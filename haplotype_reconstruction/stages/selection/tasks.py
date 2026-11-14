"""Declare selection block/context tasks and resume the initial, l1 and l2 rounds.

Owns LS23-LS31 orchestration: evidence slices, explicit CPU teams, independent
work units, deterministic gathers, publication and the context-assembly boundary.
Science never schedules or reads records. The engine owns admission and memory;
working-set estimates below are declarations, never private scientific budgets.
Method: docs/methods/local_selection.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..configure import ResolvedConfiguration
    from ...contracts.genetic_map import GeneticMapSet
    from ...core.ties import TieSettings

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any
from typing import Protocol, cast

import numpy as np

from ...contracts.assembly import AssemblyProduct
from ...contracts.discovery import DiscoveredBlock
from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.likelihoods import CodedLikelihoods, gather_likelihoods
from ...contracts.observations import PairArray, dense_observed
from ...contracts.panels import ComponentPanel, LocalPanel, materialise
from ...contracts.selection import ContextProposal, RefitSummary
from ...core.errors import InvariantError, SettingsError
from ...core.ties import TieCount, TieCounts
from ...engine.cpu import share_reader, thread_team
from ...engine.lanes import minimum_teams
from ...science.assembly.run import AssemblyResult
from ...science.discovery.parameters import DiscoveryParameters
from ...science.selection.context_refit import (
    ContextRefitTask,
    RefitFinished,
    finish_component,
    refit_component,
    refit_task_bytes,
    refit_tasks,
)
from ...science.selection.diagnostics import CensusSink
from ...science.selection.parameters import (
    CHUNK_BLOCKS,
    CHUNK_FLOOR,
    CHUNK_TASKS_PER_CPU,
    CHUNK_WAVES,
    SHARE_REREAD_SECONDS,
    LANE_MIN_WIDTH,
    LANE_WIDTH,
    SMALL_COHORT_CHUNK_BLOCKS,
    SMALL_COHORT_TASKS_PER_CPU,
    SelectionParameters,
)
from ...science.selection.rounds import BlockSelection, BlockTask, block_task_bytes, select_task
from ...science.selection.run import initial_tasks, round_tasks
from ...store.protocol import ClaimBusy, claim
from ...store.store import CheckpointStore
from ..framework import Expansion, RecordKey, ResourceRequest, ResultRef, Task, TaskKey
from .records import ROUNDS, SelectionRecords

__all__ = [
    "round_expansion",
    "context_expansion",
    "SelectionSession",
    "selection_round_task",
]


BlockSettings = tuple[SelectionParameters, DiscoveryParameters, float]


class FounderEvidence(Protocol):
    """Immutable contig founder-view evidence, shared by rounds and context assembly."""

    positions: np.ndarray
    likelihoods: np.ndarray | CodedLikelihoods
    observed: np.ndarray | PairArray


def _run_block(
    block: BlockTask,
    settings: BlockSettings,
    evidence: FounderEvidence,
    scope: SelectionRecords | None,
    *,
    reverse: bool,
    census_sink: CensusSink | None = None,
    threads: int = 1,
    memo_scope: SelectionRecords | None = None,
    retain_memo: bool = True,
) -> BlockSelection:
    """Execute one block with its allocated team; publish memo before block when recording."""

    def compute() -> BlockSelection:
        # Recorded graph rounds carry only the previous outcomes/releases. The
        # exact memo belongs to this block's consumer, not every round continuation.
        current = block
        if memo_scope is not None and current.memo is None:
            current = replace(current, memo=memo_scope.load_memo(current.index))
        with thread_team(default=threads):
            value = select_task(
                current,
                evidence.likelihoods,
                evidence.observed,
                settings,
                reverse=reverse,
                census_sink=census_sink,
                threads=share_reader(SHARE_REREAD_SECONDS),
            )
        return value

    if scope is None:
        return compute()
    with claim(
        scope.file, scope.key("block", block.index), scope.derivation("block", block.index), blocking=False
    ) as needed:
        if not needed:
            saved = scope.load_block(block.index, memos=retain_memo)
            assert saved is not None
            return saved
        result = compute()
        scope.save_block(result)
        # Publication retains the complete memo for resume and the next round.
        # The gather needs the outcome/release, and can drop the memo immediately.
        return result if retain_memo else replace(result, memo=None)


def _run_blocks(
    blocks: tuple[BlockTask, ...],
    settings: BlockSettings,
    evidence: FounderEvidence,
    scope: SelectionRecords | None,
    *,
    reverse: bool,
    census_sink: CensusSink | None = None,
    threads: int = 1,
    memo_scope: SelectionRecords | None = None,
    retain_memos: bool = True,
) -> tuple[BlockSelection, ...]:
    """Several blocks of one round in one engine task, in turn; each claims and publishes its own records exactly as
    a task of its own did, and re-reads the task's CPU share first (it grows into CPUs its peers free)."""
    from ...engine.cpu import current_budget

    results = []
    busy = None
    for block in blocks:
        try:
            result = _run_block(
                block,
                settings,
                evidence,
                scope,
                reverse=reverse,
                census_sink=census_sink,
                threads=max(1, int(current_budget().threads())),
                memo_scope=memo_scope,
                retain_memo=retain_memos,
            )
        except ClaimBusy as error:
            busy = error
        else:
            results.append(result)
    if busy is not None:
        raise busy
    return tuple(results)


def _gather_round(
    scope: SelectionRecords | None,
    indices: tuple[int, ...],
    saved: Sequence[BlockSelection],
    completed: Sequence[BlockSelection | tuple[BlockSelection, ...]],
) -> tuple[BlockSelection, ...]:
    finished = [item for result in completed for item in (result if isinstance(result, tuple) else (result,))]
    values = {item.index: item for item in (*saved, *finished)}
    if set(values) != set(indices):
        raise InvariantError(f"selection gather got {sorted(values)}, expected {indices}")
    if scope is not None:
        scope.complete(indices)
    return tuple(values[index] for index in indices)


def round_expansion(
    blocks: Sequence[BlockTask],
    settings: BlockSettings,
    evidence: FounderEvidence,
    *,
    scope: SelectionRecords | None = None,
    contig: str = "fixture",
    round_name: str = "initial",
    reverse: bool = False,
    census_sink: CensusSink | None = None,
    max_threads: int = 1,
    memo_scope: SelectionRecords | None = None,
    retain_memos: bool = True,
) -> Expansion:
    """Declare only missing block tasks and a gather, in deterministic scientific order.

    This is the graph entry for an expanding selection round. It performs no fitting.
    Complete blocks resume from their persisted records. Recorded graph rounds
    load the previous memo only at its block consumer and omit it from gathered
    results; direct callers retain the historical in-memory memo route by default.
    Tasks declare working sets; the scheduler owns concurrency and CPU allocation.
    """
    saved, pending = [], []
    for block in blocks:
        prior = None if scope is None else scope.load_block(block.index, memos=retain_memos)
        if prior is not None:
            saved.append(prior)
            continue
        pending.append(block)

    def cost(block: BlockTask) -> int:
        size = max(
            [
                len(block.latent),
                *(len(p) for p in block.proposals),
                *(len(p) for p in block.alternatives),
                0 if block.incumbent is None else len(block.incumbent.fit.panel),
            ]
        )
        return len(block.columns) * (size + 1) ** 2

    # Heaviest first. Cost-guided chunks amortise coordinator turnaround and
    # repeated evidence metadata. Small SIMD cohorts keep at least two chunks
    # per CPU and at most four blocks per chunk; other cohorts retain the higher
    # task-count floor for variable-cost refits. Guided targets shrink as work
    # finishes, and declared costs let the engine dispatch the heaviest first.
    pending.sort(key=lambda block: (-cost(block), block.index))
    costs = [cost(block) for block in pending]
    threads = max(1, int(max_threads))
    observed_rows = evidence.observed.shape[0]
    small_cohort = observed_rows <= LANE_WIDTH
    tasks_per_cpu = SMALL_COHORT_TASKS_PER_CPU if small_cohort else CHUNK_TASKS_PER_CPU
    chunk_blocks = SMALL_COHORT_CHUNK_BLOCKS if small_cohort else max(1, len(pending))
    count = min(len(pending), max(-(-len(pending) // CHUNK_BLOCKS), tasks_per_cpu * threads))
    average = sum(costs) / count if costs else 0.0
    remaining = float(sum(costs))
    # The round's stragglers start on a minimum lane team, one thread per 64-sample lane block (engine
    # minimum_teams); a cohort within one lane block has no team. A teamed block is a chunk of its own, never
    # grouped, so its team is its own.
    teams = minimum_teams(costs, [-(-observed_rows // LANE_WIDTH)] * len(pending), threads)
    chunks: list[list[BlockTask]] = []
    weights: list[float] = []
    chunk_teams: list[int] = []
    target = average
    for block, weight, team in zip(pending, costs, teams, strict=True):
        if (
            chunks
            and team == 1
            and chunk_teams[-1] == 1
            and len(chunks[-1]) < chunk_blocks
            and weights[-1] + weight <= target
        ):
            chunks[-1].append(block)
            weights[-1] += weight
        else:
            target = max(average / CHUNK_FLOOR, min(average, remaining / (CHUNK_WAVES * threads)))
            chunks.append([block])
            weights.append(float(weight))
            chunk_teams.append(team)
        remaining -= weight
    tasks = []
    block_threads = (
        min(max_threads, max(1, -(-observed_rows // LANE_MIN_WIDTH))) if small_cohort else max_threads
    )
    for chunk, weight, team in zip(chunks, weights, chunk_teams, strict=True):
        outputs: tuple[RecordKey, ...] = ()
        if scope is not None:
            for block in chunk:
                outputs += (RecordKey(contig, scope.key("block", block.index)),)
                if block.keep_memo:
                    outputs += (RecordKey(contig, scope.key("memo", block.index)),)
        tasks.append(
            Task(
                TaskKey("selection", f"blocks:{chunk[0].index:06d}", contig, round_name),
                _run_blocks,
                args=(tuple(chunk), settings, evidence, scope),
                kwargs=dict(reverse=reverse, census_sink=census_sink, memo_scope=memo_scope, retain_memos=retain_memos),
                # Small-cohort chunks may grow within their useful-width cap.
                # At larger sample counts only singleton chunks may grow;
                # grouped chunks share the node through one CPU apiece. A
                # straggler's chunk starts on its minimum team.
                resources=ResourceRequest(
                    team,
                    max(team, block_threads if small_cohort or len(chunk) == 1 else 1),
                    max(block_task_bytes(b, observed_rows) for b in chunk),
                    cost=weight,
                ),
                outputs=outputs,
                thread_argument="threads",
            )
        )
    keys = tuple(task.key for task in tasks)
    indices = tuple(block.index for block in blocks)
    gather = Task(
        TaskKey("selection", "selected", contig, round_name),
        _gather_round,
        args=(scope, indices, tuple(saved), tuple(ResultRef(key) for key in keys)),
        depends_on=keys,
        affinity=contig,
        outputs=() if scope is None else (RecordKey(contig, scope.key("selected")),),
    )
    return Expansion(tuple(tasks), gather)


def _run_refit(
    task: ContextRefitTask,
    likelihoods: np.ndarray | CodedLikelihoods,
    observed: np.ndarray,
    *,
    reverse: bool,
    census_sink: CensusSink | None = None,
    threads: int = 1,
    scope: SelectionRecords | None = None,
) -> RefitFinished:
    """Gather one component's private evidence and finish it on its allocated team."""
    if scope is not None:
        with claim(scope.file, scope.key("refit", task.index), scope.derivation("refit", task.index),
                   blocking=False) as needed:
            if not needed:
                saved = scope.load_refit(task.index)
                assert saved is not None
                return saved
            result = _run_refit(task, likelihoods, observed, reverse=reverse, census_sink=census_sink, threads=threads)
            scope.save_refit(result)
            return result
    gl = gather_likelihoods(likelihoods, task.columns)
    seen = np.ascontiguousarray(observed[:, task.columns])
    with thread_team(default=threads):
        output = refit_component(
            task.h,
            gl,
            seen,
            task.starts,
            task.stops,
            task.intervals_morgans,
            task.rate,
            task.config,
            task.unknown_mass,
            census_key=None if task.census_key is None else f"{task.census_key}/c{task.index}",
            reverse=reverse,
            census_sink=census_sink,
            claim_threads=share_reader(SHARE_REREAD_SECONDS),  # re-read the share between kernels (refits start together, one per CPU)
        )
    return finish_component(task, output, seen)


def _gather_context(
    panels: Sequence[LocalPanel],
    tasks: Sequence[ContextRefitTask],
    skipped: int,
    results: Sequence[RefitFinished],
    scope: SelectionRecords | None,
) -> tuple[tuple[ContextProposal, ...], RefitSummary]:
    proposals = [ContextProposal(block=panel.block, covered=False, calls=panel.calls) for panel in panels]
    names = (
        "cube_sites",
        "core_cube_sites",
        "core_ball_sites",
        "bp_nonconverged",
        "ascent_backstop_hits",
        "recentre_backstop_hits",
        "complement_mode_sites",
        "multimodal_abstained",
        "guard_withheld",
    )
    totals = dict.fromkeys(names, 0)
    changes = fills = sites = half = support = max_founders = 0
    specs = {task.index: task for task in tasks}
    for index, finished, changed, filled, length, founders, report in results:
        changes += changed
        fills += filled
        sites += length
        half += cast(int, report["half_band_sites"])
        support += cast(int, report["support_band_sites"])
        max_founders = max(max_founders, founders)
        for key in totals:
            totals[key] += int(cast(int, report[key]))
        for member, proposal in zip(specs[index].members, finished, strict=True):
            proposals[member] = proposal
    ties = {
        "ls.refit.half.band": dict(decisions=sites, inside_tie=half, wide=0, boundary=0),
        "ls.refit.support.band": dict(decisions=sites, inside_tie=support, wide=0, boundary=0),
    }
    summary = RefitSummary(
        len(tasks),
        skipped,
        changes,
        fills,
        tuple(totals.items()),
        max_founders,
        TieCounts(tuple(TieCount(kind, **value) for kind, value in sorted(ties.items()))),
    )
    if scope is not None:
        scope.save_proposals(proposals, summary)
    return tuple(proposals), summary


def context_expansion(
    panels: Sequence[LocalPanel],
    components: Sequence[ComponentPanel],
    evidence: FounderEvidence,
    genetic_map: ChromosomeGeneticMap,
    parameters: SelectionParameters,
    *,
    scope: SelectionRecords | None = None,
    contig: str = "fixture",
    round_name: str = "l1",
    reverse: bool = False,
    census_sink: CensusSink | None = None,
    max_threads: int = 1,
) -> Expansion:
    """Declare independent refit components after making the kept-site mask once.

    Inputs are already materialised assembly ComponentPanels. The observed mask is a
    private copy; every member panel's kept mask is applied.
    Partial component coverage is counted by pure refit_tasks, never completed here.
    """
    observed = (
        np.array(evidence.observed, copy=True)
        if isinstance(evidence.observed, np.ndarray)
        else dense_observed(evidence.observed)
    )
    for panel in panels:
        observed[:, panel.site_start : panel.site_start + panel.sites] &= panel.kept
    specs, skipped = refit_tasks(
        panels,
        components,
        observed,
        evidence.positions,
        genetic_map,
        parameters,
        census_key=f"{contig}/{round_name}/refit",
    )
    tasks = tuple(
        Task(
            TaskKey("selection", f"refit:{spec.index:06d}", contig, round_name),
            _run_refit,
            args=(spec, evidence.likelihoods, observed),
            kwargs=dict(reverse=reverse, census_sink=census_sink, scope=scope),
            outputs=() if scope is None else (RecordKey(contig, scope.key("refit", spec.index)),),
            resources=ResourceRequest(
                1, max_threads, refit_task_bytes(observed.shape[0], len(spec.columns), len(spec.starts), len(spec.h))
            ),
            thread_argument="threads",
        )
        for spec in specs
    )
    keys = tuple(task.key for task in tasks)
    gather = Task(
        TaskKey("selection", "proposals", contig, round_name),
        _gather_context,
        args=(panels, specs, skipped, tuple(ResultRef(key) for key in keys), scope),
        depends_on=keys,
        affinity=contig,
        outputs=() if scope is None else (RecordKey(contig, scope.key("proposals")),),
    )
    return Expansion(tasks, gather)


@dataclass(frozen=True)
class SelectionSession:
    """An explicit contig capability; inputs load lazily after checking all three rounds."""

    store: CheckpointStore
    contig: str
    parameters: SelectionParameters
    discovery_parameters: DiscoveryParameters
    genetic_map: ChromosomeGeneticMap
    upstream: dict[str, str]
    load_inputs: Callable[
        [], tuple[Sequence[DiscoveredBlock | None], FounderEvidence]
    ]
    manifest: dict
    reverse: bool = False
    max_threads: int = 1
    census_sink: CensusSink | None = None


@dataclass(frozen=True)
class RoundState:
    """Inputs shared by the round continuations; no order depends on task completion."""

    session: SelectionSession
    scopes: tuple[SelectionRecords, ...]
    initial: tuple[BlockTask, ...]
    evidence: FounderEvidence
    index: int
    previous: tuple[BlockSelection, ...] | None
    selected_calls: tuple[tuple[np.ndarray, ...], ...]


def _scopes(session: SelectionSession) -> tuple[SelectionRecords, ...]:
    from .records import expected

    return expected(session.store, session.manifest, product="selected", contig=session.contig,
                    upstream=session.upstream).scopes


def _selected_calls(
    earlier: Sequence[SelectionRecords], previous: tuple[BlockSelection, ...] | None
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Each earlier round's selected calls in round order. The last earlier round's panels are those of its loaded
    selections (``previous``, the same stored panels), so they are not read a second time."""
    calls = [tuple(panel.calls for panel in scope.load_panels()) for scope in earlier[:-1]]
    if earlier:
        assert previous is not None
        calls.append(tuple(item.panel.calls for item in previous))
    return tuple(calls)


def _expand_refit(state: RoundState, assembled: AssemblyResult | AssemblyProduct) -> Expansion:
    session, scope = state.session, state.scopes[state.index]
    assert state.previous is not None
    panels = tuple(item.panel for item in state.previous)
    expansion = context_expansion(
        panels,
        materialise(assembled.components, assembled.prepared),
        state.evidence,
        session.genetic_map,
        session.parameters,
        scope=scope,
        contig=session.contig,
        round_name=scope.round_name,
        reverse=session.reverse,
        census_sink=session.census_sink,
        max_threads=session.max_threads,
    )
    gather = Task(
        TaskKey("selection", "context_ready", session.contig, scope.round_name),
        _after_refit,
        args=(state, ResultRef(expansion.gather.key)),
        depends_on=(expansion.gather.key,),
        affinity=session.contig,
    )
    return Expansion((*expansion.subtasks, expansion.gather), gather)


def _after_refit(state: RoundState, result: tuple[tuple[ContextProposal, ...], RefitSummary]) -> Expansion:
    return _expand_blocks(state, result[0])


def _expand_blocks(state: RoundState, proposals: Sequence[ContextProposal] | None) -> Expansion:
    session, scope = state.session, state.scopes[state.index]
    blocks = round_tasks(
        state.initial, state.index, state.previous, state.selected_calls, proposals, label=session.contig
    )
    settings = (session.parameters, session.discovery_parameters, session.parameters.model.redraw_rate_per_morgan)
    expansion = round_expansion(
        blocks,
        settings,
        state.evidence,
        scope=scope,
        contig=session.contig,
        round_name=scope.round_name,
        reverse=session.reverse,
        census_sink=session.census_sink,
        max_threads=session.max_threads,
        memo_scope=None if state.index == 0 else state.scopes[state.index - 1],
        retain_memos=False,
    )
    gather = Task(
        TaskKey("selection", "round_ready", session.contig, scope.round_name),
        _after_round,
        args=(ResultRef(expansion.gather.key),),
        depends_on=(expansion.gather.key,),
        affinity=session.contig,
    )
    return Expansion((*expansion.subtasks, expansion.gather), gather)


def _after_round(selected: tuple[BlockSelection, ...]) -> tuple[LocalPanel, ...]:
    panels = tuple(item.panel for item in selected)
    return panels


def _enter_round(
    session: SelectionSession,
    round_name: str,
    load_context: Callable[[], AssemblyResult | AssemblyProduct] | None,
) -> tuple[LocalPanel, ...] | Expansion:
    scopes = _scopes(session)
    index = ROUNDS.index(round_name)
    scope = scopes[index]
    if scope.selected() is not None:
        return scope.load_panels()
    # Each earlier round's completion is verified here, once; the loads below then check only the records they read.
    if any(earlier.selected() is None for earlier in scopes[:index]):
        raise InvariantError(f"selection {round_name} requires all preceding selected rounds")
    proposals = None if index == 0 else scope.load_proposals()
    discovered, evidence = session.load_inputs()
    previous = None if index == 0 else scopes[index - 1].load_selections(memos=False)
    state = RoundState(
        session,
        scopes,
        initial_tasks(discovered, session.genetic_map, label=session.contig),
        evidence,
        index,
        previous,
        _selected_calls(scopes[:index], previous),
    )
    if index == 0:
        return _expand_blocks(state, None)
    if proposals is not None:
        return _expand_blocks(state, proposals)
    if load_context is None:
        raise InvariantError(f"selection {round_name} requires its published context assembly")
    return _expand_refit(state, load_context())


def selection_round_task(
    session: SelectionSession,
    round_name: str,
    *,
    load_context: Callable[[], AssemblyResult | AssemblyProduct] | None = None,
    key: TaskKey | None = None,
    depends_on: Sequence[TaskKey] = (),
) -> Task:
    """graph/workflow action for exactly one of initial, l1 and l2.

    The caller's context.l1/context.l2 assembly node publishes the matching assembly
    product first. Its importable, picklable load_context callable verifies that
    product's independently derived identity and returns AssemblyProduct (or the
    equivalent AssemblyResult from an executed assembly task). This
    action refits that context and selects only the named round; it never starts
    another assembly or selection round. Complete rounds read panels lazily.
    """
    if round_name not in ROUNDS:
        raise InvariantError(f"unsupported selection round {round_name!r}")
    return Task(
        key or TaskKey("selection", "round", session.contig, round_name),
        _enter_round,
        args=(session, round_name, load_context),
        depends_on=tuple(depends_on),
        affinity=session.contig,
        resources=ResourceRequest(1, 1),
    )


from ..inputs.products import task_inputs
from ..context import StageRun
from ..inputs.chain import _chain_input, _chain_evidence


@dataclass(frozen=True)
class SelectionInputs:
    """Picklable capability loading checked discovery and evidence only when a round needs them."""

    root: str
    contig: str

    def __call__(self) -> Any:
        from ..discovery import records as discovery_records

        reader, store = task_inputs(self.root)
        request = discovery_records.result_reader(
            store, reader.manifest, product="discovery", contig=self.contig
        )
        if request is None:
            raise SettingsError(f"{self.contig}: required discovery publication is missing")
        discovered = _chain_input(f"discovery/{request.identity}", lambda: reader.require(request, self.contig))
        _, _, founder, view = _chain_evidence(reader, store, self.contig)
        return discovered, founder


@dataclass(frozen=True)
class SelectionContext:
    """Read the already published context assembly without creating a second assembly task."""

    root: str
    contig: str
    round_name: str

    def __call__(self) -> Any:
        from ..assembly import records

        reader, store = task_inputs(self.root)
        request = records.result_reader(
            store,
            reader.manifest,
            product="assembly",
            contig=self.contig,
            round_name=self.round_name,
        )
        context = None if request is None else reader.read(request, self.contig)
        if context is None:
            raise SettingsError(f"{self.contig}: selection needs published context {self.round_name}")
        return context


def _selected_result(root, contig, round_name):
    """Return the checked selected publication after its round expansion."""
    from . import records

    reader, store = task_inputs(root)
    request = reader.request(records, "selected", contig, stage="local_selection_" + round_name)
    reader.require_receipt(request, contig)
    return {RecordKey(contig, request.key): request.identity}


def run(
    root: str,
    contig: str,
    round_name: str,
    resolved: ResolvedConfiguration,
    maps: GeneticMapSet,
    ties: TieSettings,
    *,
    cores: int = 1,
) -> Expansion | dict[RecordKey, str]:
    """Expand precisely one selection round, consuming the preceding context assembly."""
    from ...engine.graph import Expansion, RecordKey, ResultRef, Task, TaskKey
    from ..discovery import records as discovery_records
    from ..evidence import records as evidence_records
    from ..relatedness.working_map import working_maps
    from ..selection.tasks import SelectionSession, selection_round_task

    reader, store = task_inputs(root)
    # A round whose completion is published under the identities this run derives (each promised block,
    # memo and proposal record checked against its identity) is not expanded: its selected-record receipt is the
    # gather's. Expanding would decode every panel before the gather checks that same record.
    from ..selection import records as selection_records

    done = selection_records.result_reader(
        store, reader.manifest, product="selected", contig=contig, stage="local_selection_" + round_name
    )
    if done is not None and done.scopes[done.round_index].selected(verify=True) is not None:
        if reader.receipt(done, contig) is not None:
            return {RecordKey(contig, done.key): done.identity}
    upstream = {}
    for name, adapter in (("evidence", evidence_records), ("discovery", discovery_records)):
        request = reader.request(adapter, name, contig)
        reader.require_receipt(request, contig)
        upstream[name] = request.identity
    session = SelectionSession(
        store,
        contig,
        resolved.parameters.local_selection,
        resolved.parameters.discovery,
        working_maps(store, reader.manifest, maps).for_contig(contig),
        upstream,
        SelectionInputs(root, contig),
        reverse=ties.reverse,
        census_sink=ties.census,
        max_threads=cores,
        manifest=reader.manifest,
    )
    child = selection_round_task(
        session,
        round_name,
        load_context=None if round_name == "initial" else SelectionContext(root, contig, round_name),
        key=TaskKey("selection", "select", contig, round_name),
    )
    gather = Task(
        TaskKey("selection", "published", contig, round_name),
        _selected_result,
        (root, contig, round_name),
        depends_on=(child.key,),
        affinity=contig,
    )
    return Expansion((child,), gather)


def task(key, context: StageRun) -> Task:
    """Declare exactly one production selection round."""
    return Task(
        key, run,
        (context.root, key.scope, key.round, context.resolved, context.maps, context.ties),
        kwargs={"cores": context.cores}, resources=ResourceRequest(1, context.cores),
    )
