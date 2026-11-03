"""Declare selection block/context tasks and resume the initial, l1 and l2 rounds.

Owns LS23-LS31 orchestration: evidence slices, explicit CPU teams, independent
work units, deterministic gathers, publication and the context-assembly boundary.
Science never schedules or reads records. The engine owns admission and memory;
working-set estimates below are declarations, never private scientific budgets.
Method: docs/methods/local_selection.md; WP_SE2 sections 4.10-4.14 and 8.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

import numpy as np

from ...contracts.assembly import AssemblyProduct
from ...contracts.discovery import DiscoveredBlock
from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.panels import ComponentPanel, LocalPanel, materialise
from ...contracts.selection import ContextProposal, RefitSummary
from ...core.errors import InvariantError
from ...core.ties import TieCount, TieCounts
from ...engine.graph import run_graph
from ...engine.runtime import task_team
from ...science.assembly.evidence import AssemblyEvidence
from ...science.assembly.parameters import AssemblyParameters
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
from ...science.selection.parameters import SelectionParameters
from ...science.selection.rounds import BlockSelection, BlockTask, block_task_bytes, select_task
from ...science.selection.run import initial_tasks, round_tasks
from ...store.protocol import NumericalPolicy, bound_payload_identity, claim
from ...store.store import CheckpointStore
from ..framework import Expansion, RecordKey, ResourceRequest, ResultRef, Task, TaskKey
from .records import ROUNDS, SelectionRecords, context_assembly_prefix, round_identity

__all__ = [
    "round_expansion",
    "context_expansion",
    "run_round",
    "run_context_refit",
    "run_contig",
    "SelectionSession",
    "selection_task",
    "selection_round_task",
]


BlockSettings = tuple[SelectionParameters, DiscoveryParameters, float]
CHUNK_BLOCKS = 8  # average blocks' cost per engine task in a selection round (execution only: results are per block)
# A block re-reads its task's CPU share (an engine boundary) at most this often: its lane evaluations run thousands of
# times, and a block that started on one thread beside hundreds of siblings grows as they finish.
SHARE_REREAD_SECONDS = 0.1


def live_share(interval: float = SHARE_REREAD_SECONDS) -> Callable[[], int]:
    """The calling task's live CPU share, re-read (engine claim_threads: a boundary that also sets this thread's kernel
    team) when ``interval`` seconds have passed since the last read."""
    from ...engine.scheduler import claim_threads

    read_at, share = [-math.inf], [1]

    def read() -> int:
        now = time.monotonic()
        if now - read_at[0] >= interval:
            read_at[0], share[0] = now, claim_threads()
        return share[0]

    return read


class FounderEvidence(Protocol):
    """Immutable contig founder-view evidence, shared by rounds and context assembly."""

    positions: np.ndarray
    likelihoods: np.ndarray
    observed: np.ndarray


def _run_block(
    block: BlockTask,
    settings: BlockSettings,
    evidence: FounderEvidence,
    scope: SelectionRecords | None,
    *,
    reverse: bool,
    census_sink: CensusSink | None = None,
    threads: int = 1,
) -> BlockSelection:
    """Execute one block with its allocated team; publish memo before block when recording."""

    def compute() -> BlockSelection:
        with task_team(threads):
            value = select_task(
                block,
                evidence.likelihoods,
                evidence.observed,
                settings,
                reverse=reverse,
                census_sink=census_sink,
                threads=live_share(),
            )
        return value

    if scope is None:
        return compute()
    with claim(scope.file, scope.key("block", block.index), scope.derivation("block", block.index)) as needed:
        if not needed:
            saved = scope.load_block(block.index)
            assert saved is not None
            return saved
        result = compute()
        scope.save_block(result)
        return result


def _run_blocks(
    blocks: tuple[BlockTask, ...],
    settings: BlockSettings,
    evidence: FounderEvidence,
    scope: SelectionRecords | None,
    *,
    reverse: bool,
    census_sink: CensusSink | None = None,
    threads: int = 1,
) -> tuple[BlockSelection, ...]:
    """Several blocks of one round in one engine task, in turn; each claims and publishes its own records exactly as
    a task of its own did, and re-reads the task's CPU share first (it grows into CPUs its peers free)."""
    from ...engine.scheduler import current_budget

    results = []
    for block in blocks:
        results.append(
            _run_block(
                block,
                settings,
                evidence,
                scope,
                reverse=reverse,
                census_sink=census_sink,
                threads=max(1, int(current_budget().threads())),
            )
        )
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
) -> Expansion:
    """Declare only missing block tasks and a gather, in deterministic scientific order.

    This is the G1 entry for an expanding selection round. It performs no fitting.
    Complete blocks are resumed with their exact persisted memos. Tasks declare
    working sets and outputs; the scheduler owns concurrency and CPU allocation.
    """
    saved, pending = [], []
    for block in blocks:
        prior = None if scope is None else scope.load_block(block.index)
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

    # Heaviest first, as before; then consecutive blocks are grouped into engine tasks of about CHUNK_BLOCKS average
    # blocks' cost (a block at least that heavy stays alone), so the coordinator's fixed turnaround per task (reply,
    # bookkeeping, dispatch, send) is paid per chunk rather than per sub-second block. A round never has fewer tasks
    # than twice the CPUs it may use (or its block count): heavy blocks (K = 30, about a CPU-minute each) stay about
    # one per task, so the round still fills the node.
    pending.sort(key=lambda block: (-cost(block), block.index))
    costs = [cost(block) for block in pending]
    count = min(len(pending), max(-(-len(pending) // CHUNK_BLOCKS), 2 * max(1, int(max_threads))))
    target = sum(costs) / count if costs else 0.0
    chunks: list[list[BlockTask]] = []
    total = 0.0
    for block, weight in zip(pending, costs, strict=True):
        if chunks and chunks[-1] and total + weight <= target:
            chunks[-1].append(block)
            total += weight
        else:
            chunks.append([block])
            total = weight
    tasks = []
    observed_rows = evidence.observed.shape[0]
    for chunk in chunks:
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
                kwargs=dict(reverse=reverse, census_sink=census_sink),
                # Grouped light blocks: one CPU (their fits ran at ~1.0 CPU-second per wall second whatever they
                # claimed, and larger claims kept CPUs from lanes with work ready). A block alone in its task is a
                # heavy one (K = 30): its path-model kernels spread its samples' lanes over the task's threads.
                resources=ResourceRequest(
                    1, 1 if len(chunk) > 1 else max_threads, max(block_task_bytes(b, observed_rows) for b in chunk)
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


def run_round(
    blocks: Sequence[BlockTask],
    settings: BlockSettings,
    evidence: FounderEvidence,
    *,
    scope: SelectionRecords | None = None,
    contig: str = "fixture",
    round_name: str = "initial",
    reverse: bool = False,
    census_sink: CensusSink | None = None,
    cores: int = 1,
    threads: int = 1,
    in_process: bool = True,
) -> tuple[BlockSelection, ...]:
    """Execute one declared round directly; composed graphs use round_expansion instead."""
    expansion = round_expansion(
        blocks,
        settings,
        evidence,
        scope=scope,
        contig=contig,
        round_name=round_name,
        reverse=reverse,
        census_sink=census_sink,
        max_threads=threads,
    )
    priorities = {task.key: float(len(expansion.subtasks) - i) for i, task in enumerate(expansion.subtasks)}
    results = run_graph(
        (*expansion.subtasks, expansion.gather),
        cores=cores,
        in_process=in_process,
        priorities=priorities,
        preload=(__name__,),
    )
    return results[expansion.gather.key]


def _run_refit(
    task: ContextRefitTask,
    likelihoods: np.ndarray,
    observed: np.ndarray,
    *,
    reverse: bool,
    census_sink: CensusSink | None = None,
    threads: int = 1,
) -> RefitFinished:
    """Gather one component's private evidence and finish it on its allocated team."""
    gl = np.ascontiguousarray(likelihoods[:, task.columns], dtype=np.float64)
    seen = np.ascontiguousarray(observed[:, task.columns])
    with task_team(threads):
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
            claim_threads=live_share(),  # re-read the share between kernels (refits start together, one per CPU)
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

    Inputs are already materialised AS6 ComponentPanels. The observed mask is a
    private copy; every member panel's kept mask is applied exactly as in 3.5.
    Partial component coverage is counted by pure refit_tasks, never completed here.
    """
    observed = np.array(evidence.observed, copy=True)
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
            kwargs=dict(reverse=reverse, census_sink=census_sink),
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


def run_context_refit(
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
    cores: int = 1,
    threads: int = 1,
    in_process: bool = True,
) -> tuple[tuple[ContextProposal, ...], RefitSummary]:
    """Run a context fixture or stage cut; a composed graph uses context_expansion."""
    expansion = context_expansion(
        panels,
        components,
        evidence,
        genetic_map,
        parameters,
        scope=scope,
        contig=contig,
        round_name=round_name,
        reverse=reverse,
        census_sink=census_sink,
        max_threads=threads,
    )
    result = run_graph((*expansion.subtasks, expansion.gather), cores=cores, in_process=in_process, preload=(__name__,))
    return result[expansion.gather.key]


def run_contig(
    store: CheckpointStore,
    contig: str,
    parameters: SelectionParameters,
    discovery_parameters: DiscoveryParameters,
    assembly_parameters: AssemblyParameters,
    genetic_map: ChromosomeGeneticMap,
    upstream: dict[str, str],
    *,
    load_inputs: Callable[[], tuple[Sequence, FounderEvidence, Any]],
    assemble_context: Callable,
    reverse: bool = False,
    census_sink: CensusSink | None = None,
    cores: int = 1,
    threads: int = 1,
    in_process: bool = True,
) -> tuple[LocalPanel, ...]:
    """Resume or execute the three rounds, calling AS6 only for missing context proposals.

    load_inputs lazily returns discovered blocks, founder evidence and AS6's
    evidence view. assemble_context receives panels, view, map, context parameters,
    level and records prefix and returns AssemblyResult. IN1 supplies this explicit
    AS6 boundary. An all-complete contig loads only the final panels.
    """
    file = store.contig(contig)
    scopes = []
    identities = dict(upstream)
    for name in ROUNDS:
        identity = round_identity(
            name,
            parameters,
            discovery_parameters,
            genetic_map,
            identities,
            assembly_parameters if name != "initial" else None,
            reverse=reverse,
        )
        scopes.append(SelectionRecords(file, contig, name, identity))
        identities = dict(identities, previous_round=identity)
    first = next((i for i, scope in enumerate(scopes) if scope.selected() is None), len(ROUNDS))
    if first == len(ROUNDS):
        return scopes[-1].load_panels()
    discovered, evidence, assembly_evidence = load_inputs()
    initial = initial_tasks(discovered, genetic_map, label=contig)
    selected_calls = [[panel.calls for panel in scope.load_panels()] for scope in scopes[:first]]
    previous = None if first == 0 else scopes[first - 1].load_selections()
    panels = None if previous is None else tuple(item.panel for item in previous)
    settings = (parameters, discovery_parameters, parameters.model.redraw_rate_per_morgan)
    for index in range(first, len(ROUNDS)):
        scope, name = scopes[index], ROUNDS[index]
        proposals = None
        if index:
            assert panels is not None
            proposals = scope.load_proposals()
            if proposals is None:
                result = assemble_context(
                    panels,
                    assembly_evidence,
                    genetic_map,
                    assembly_parameters,
                    level=parameters.context_levels[index - 1],
                    records_prefix=context_assembly_prefix(name),
                )
                components = materialise(result.components, result.prepared)
                proposals, _ = run_context_refit(
                    panels,
                    components,
                    evidence,
                    genetic_map,
                    parameters,
                    scope=scope,
                    contig=contig,
                    round_name=name,
                    reverse=reverse,
                    census_sink=census_sink,
                    cores=cores,
                    threads=threads,
                    in_process=in_process,
                )
        blocks = round_tasks(initial, index, previous, selected_calls, proposals, label=contig)
        previous = run_round(
            blocks,
            settings,
            evidence,
            scope=scope,
            contig=contig,
            round_name=name,
            reverse=reverse,
            census_sink=census_sink,
            cores=cores,
            threads=threads,
            in_process=in_process,
        )
        panels = tuple(item.panel for item in previous)
        selected_calls.append([panel.calls for panel in panels])
    assert panels is not None
    return panels


# The composed route expands the three rounds into the application's one graph.
# The direct run_contig/run_round helpers above remain useful for isolated seam replays.
@dataclass(frozen=True)
class SelectionSession:
    """An explicit contig capability; inputs load lazily after checking all three rounds."""

    store: CheckpointStore
    contig: str
    parameters: SelectionParameters
    discovery_parameters: DiscoveryParameters
    assembly_parameters: AssemblyParameters
    genetic_map: ChromosomeGeneticMap
    upstream: dict[str, str]
    load_inputs: Callable[[], tuple[Sequence[DiscoveredBlock | None], FounderEvidence, AssemblyEvidence]]
    reverse: bool = False
    tie_keep: tuple[str, ...] = ()
    max_threads: int = 1
    numerical_policy: NumericalPolicy | None = None
    census_sink: CensusSink | None = None


@dataclass(frozen=True)
class RoundState:
    """Inputs shared by the round continuations; no order depends on task completion."""

    session: SelectionSession
    scopes: tuple[SelectionRecords, ...]
    initial: tuple[BlockTask, ...]
    evidence: FounderEvidence
    assembly_evidence: AssemblyEvidence
    index: int
    previous: tuple[BlockSelection, ...] | None
    selected_calls: tuple[tuple[np.ndarray, ...], ...]
    stop_after_round: bool = False


def _scopes(session: SelectionSession) -> tuple[SelectionRecords, ...]:
    identities = dict(session.upstream)
    result = []
    for name in ROUNDS:
        expected = round_identity(
            name,
            session.parameters,
            session.discovery_parameters,
            session.genetic_map,
            identities,
            session.assembly_parameters if name != "initial" else None,
            reverse=session.reverse,
            tie_keep=session.tie_keep,
            numerical_policy=session.numerical_policy,
        )
        result.append(SelectionRecords(session.store.contig(session.contig), session.contig, name, expected))
        identities = dict(identities, previous_round=expected)
    return tuple(result)


def _enter_contig(session: SelectionSession) -> tuple[LocalPanel, ...] | Expansion:
    scopes = _scopes(session)
    first = next((i for i, scope in enumerate(scopes) if scope.selected() is None), len(scopes))
    if first == len(scopes):
        return scopes[-1].load_panels()
    discovered, evidence, assembly_evidence = session.load_inputs()
    initial = initial_tasks(discovered, session.genetic_map, label=session.contig)
    previous = None if first == 0 else scopes[first - 1].load_selections()
    earlier = _selected_calls(scopes[:first], previous)
    return _advance_round(RoundState(session, scopes, initial, evidence, assembly_evidence, first, previous, earlier))


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


def _advance_round(state: RoundState) -> Expansion:
    """Expand context assembly only when its proposals publication is absent."""
    session, scope = state.session, state.scopes[state.index]
    if state.index == 0:
        return _expand_blocks(state, None)
    proposals = scope.load_proposals()
    if proposals is not None:
        return _expand_blocks(state, proposals)
    from ...core.digest import json_digest
    from ...core.ties import TieSettings
    from ...science.assembly.run import AssemblyInputs, Role
    from ..assembly.final import assembly_task, bind_session

    assert state.previous is not None
    panels = tuple(item.panel for item in state.previous)
    previous = state.scopes[state.index - 1]
    upstream = dict(
        panels=previous.derivation("selected"),
        evidence=session.upstream["evidence"],
        payload=bound_payload_identity(scope.file),
        map=json_digest(session.genetic_map.identity()),
    )
    key = TaskKey("selection", "context_assembly", session.contig, scope.round_name)
    assembly = bind_session(
        AssemblyInputs(panels, state.assembly_evidence, session.genetic_map),
        session.assembly_parameters,
        file=scope.file,
        upstream=upstream,
        role=Role.CONTEXT,
        levels=session.parameters.context_levels[state.index - 1],
        records_prefix=context_assembly_prefix(scope.round_name),
        parent=key,
        max_cores=session.max_threads,
        ties=TieSettings(reverse=session.reverse, keep=session.tie_keep, census=session.census_sink),
        numerical_policy=session.numerical_policy,
    )
    child = assembly_task(assembly)
    gather = Task(
        TaskKey("selection", "context_materialise", session.contig, scope.round_name),
        _expand_refit,
        args=(state, ResultRef(child.key)),
        depends_on=(child.key,),
        affinity=session.contig,
    )
    return Expansion((child,), gather)


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
    )
    gather = Task(
        TaskKey("selection", "round_ready", session.contig, scope.round_name),
        _after_round,
        args=(state, ResultRef(expansion.gather.key)),
        depends_on=(expansion.gather.key,),
        affinity=session.contig,
    )
    return Expansion((*expansion.subtasks, expansion.gather), gather)


def _after_round(state: RoundState, selected: tuple[BlockSelection, ...]) -> tuple[LocalPanel, ...] | Expansion:
    panels = tuple(item.panel for item in selected)
    if state.stop_after_round or state.index == len(ROUNDS) - 1:
        return panels
    return _advance_round(
        replace(
            state,
            index=state.index + 1,
            previous=selected,
            selected_calls=(*state.selected_calls, tuple(panel.calls for panel in panels)),
        )
    )


def selection_task(
    session: SelectionSession, *, key: TaskKey | None = None, depends_on: Sequence[TaskKey] = ()
) -> Task:
    """Standalone graph entry: a lazy contig action expanding all rounds and contexts.

    The application owns the single run_graph call. load_inputs is an importable,
    picklable callable supplying checked discovery blocks and both evidence views.
    All-complete execution reads only selected panels and never calls load_inputs.
    """
    return Task(
        key or TaskKey("selection", "contig", session.contig),
        _enter_contig,
        args=(session,),
        depends_on=tuple(depends_on),
        affinity=session.contig,
        resources=ResourceRequest(1, 1),
        # The terminal round gather owns publication; an expanding ancestor is not a second writer.
    )


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
    discovered, evidence, assembly_evidence = session.load_inputs()
    previous = None if index == 0 else scopes[index - 1].load_selections()
    state = RoundState(
        session,
        scopes,
        initial_tasks(discovered, session.genetic_map, label=session.contig),
        evidence,
        assembly_evidence,
        index,
        previous,
        _selected_calls(scopes[:index], previous),
        stop_after_round=True,
    )
    if index == 0:
        return _expand_blocks(state, None)
    proposals = scope.load_proposals()
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
    """G1/IN1 action for exactly one of initial, l1 and l2.

    The caller's context.l1/context.l2 assembly node publishes the matching AS6
    product first. Its importable, picklable load_context callable verifies that
    product's independently derived identity and returns AssemblyProduct (or the
    equivalent AssemblyResult from an executed AS6 task). This
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
