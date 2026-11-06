"""Select and release one block while carrying its exact incumbent and memo state.

Owns LS14-LS16 and LS23-LS24: fewest-founder start filtering, outcome selection,
strict incumbent replacement, release reuse and tied-panel consensus. Must not
assemble context, schedule work, read records or inspect process state.
Method: docs/methods/local_selection.md#rounds (WP_SE2 sections 4.7 and 4.10).
The fewest fitted K filter precedes start-key deduplication. Equal panel and
frequency arrays reuse the incumbent release. Shared rows retain only calls
agreed by every tied alternative. The next incumbent uses the named convention.
block_task_bytes is an execution estimate and never changes scientific budgets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from ...contracts.likelihoods import CodedLikelihoods, gather_likelihoods
from ...contracts.observations import PairArray
from ...contracts.panels import unknown_copy_mass
from ...contracts.roles import freeze_record_arrays, record_field
from ...contracts.selection import BlockSelection, MemoHits, SelectionMemo, SelectionSummary
from ...core.ties import BOUNDARY_GUARD, TieCounter, census, census_enabled, fixed_policy, improves, tie_classes
from ..discovery.api import DiscoveryParameters, search_bytes
from .bank import candidate_bank
from .objective import ContextFounders, selection_objective
from .parameters import SEARCH_GROWTH, TASK_WORKSPACE_BASE_BYTES, SelectionParameters
from .path_model import LANE_WIDTH, RELEASE_SAMPLE_BATCH, PreparedBlock, workspace_bytes
from .release import PathRelease, local_panel, release
from .search import DECISION_MOTION, LikelihoodMemo, SearchOutcome, fitted_start, search_panel

__all__ = ["BlockTask", "select_block", "block_task_bytes", "select_task"]


@dataclass(frozen=True)
class BlockTask:
    """Inputs of one block's selection (columns: contig site indices of its discovery sites;
    label: "contig/round/i<index>", the block's key in diagnostics and the valley census;
    alternatives: discovery's NLL-tied alternative latent panels, release 2.2)."""

    index: int = record_field("state", dtype="int")
    block: int = record_field("state", dtype="int")
    site_start: int = record_field("state", dtype="int")
    positions: np.ndarray = record_field("state", axes=("L_block",), units="bp", dtype="int64")
    kept: np.ndarray = record_field("state", axes=("L_block",), units="1", dtype="bool")
    columns: np.ndarray = record_field("state", axes=("L_kept",), units="site", dtype="int64")
    intervals: np.ndarray = record_field("state", axes=("L_kept_minus_one",), units="Morgans", dtype="float64")
    latent: np.ndarray = record_field("state", axes=("K", "L_kept"), units="allele", dtype="int64")
    proposals: tuple = record_field("state", dtype="tuple")
    founders: np.ndarray | None = record_field(
        "state", axes=("K_context", "L_kept"), units="allele", dtype="int8 | None"
    )
    incumbent: SearchOutcome | None = record_field("state", dtype="SearchOutcome | None")
    incumbent_release: PathRelease | None = record_field("state", dtype="PathRelease | None")
    memo: SelectionMemo | None = record_field("state", dtype="SelectionMemo | None")
    keep_memo: bool = record_field("state", dtype="bool")
    label: str = record_field("diagnostic", dtype="str", default="")
    alternatives: tuple = record_field("state", dtype="tuple", default=())
    witness: np.ndarray | None = record_field(
        "state", axes=("K_context", "L_kept"), units="1", dtype="bool | None", default=None
    )

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


def _best_outcome(outcomes, tolerance, ties, *, reverse: bool) -> tuple[SearchOutcome, list]:
    """(the selected outcome of a multi-start search, the tied alternatives): the best tie class of
    the selection objectives, fewer founders inside it (a stated prior); the distinct panels still
    tied (one outcome each, the representative included; empty when one panel remains). The
    smallest canonical key represents them (a convention: tie_classes reverses it under
    maths.ties.reverse_conventions; the release keeps only what all of them agree on)."""
    if len(outcomes) == 1:
        return outcomes[0], []
    classes = tie_classes(
        [outcome.selection_objective for outcome in outcomes],
        tolerance,
        secondary=[(len(outcome.fit.panel),) for outcome in outcomes],
        content=[outcome.fit.panel.tobytes() for outcome in outcomes],
        descending=True,
        motion=DECISION_MOTION,
        reverse=fixed_policy("selection.start.outcome", tolerance, reverse=reverse).reverse,
    )
    winner = outcomes[int(classes.order[0])]
    fewest = len(winner.fit.panel)
    tied: dict[bytes, SearchOutcome] = {}
    for n in classes.order:
        outcome = outcomes[int(n)]
        if classes.classes[int(n)] == 0 and len(outcome.fit.panel) == fewest:
            tied.setdefault(outcome.fit.panel.tobytes(), outcome)
    ties.note(
        "ls.start.outcome", inside_tie=len(tied) > 1, wide=classes.wide_classes, boundary=len(classes.boundary_gaps)
    )
    return winner, (list(tied.values()) if len(tied) > 1 else [])


def _consensus_panel(
    task, outcome, path_release, alternatives, prepared, release_config, context=None, *, census_sink=None
):
    """The LocalPanel released for tied alternative panels (release 2.2), and the founder rows it
    withholds. A founder row that every alternative contains keeps a call where all alternatives'
    releases of that row agree (-1 elsewhere), their mean ALT probability and their smallest
    support; the other rows are withheld (calls -1, ALT probability 1/2, support 0); unknown slots
    are the largest over the alternatives (the conservative side for completion's veto); rows are
    in the order of their released content. Nothing depends on which alternative represents the
    others."""
    released = []
    for alternative in alternatives:
        own = (
            path_release
            if alternative is outcome
            else release(
                alternative.fit,
                prepared,
                release_config,
                context=context,
                witness=task.witness,
                census_sink=census_sink,
            )
        )
        released.append(
            (
                alternative.fit.panel,
                local_panel(task.block, task.site_start, task.positions, task.kept, alternative.fit, own),
            )
        )
    shared = None
    for fitted, _panel in released:
        rows = {row.tobytes() for row in fitted}
        shared = rows if shared is None else shared & rows
    representative = next(
        panel for fitted, panel in released if fitted is outcome.fit.panel or np.array_equal(fitted, outcome.fit.panel)
    )
    k, length = representative.calls.shape
    calls = np.full((k, length), -1, dtype=np.int8)
    alt_probability = np.full((k, length), 0.5)
    support = np.zeros((k, length))
    for target, key in enumerate(sorted(shared)):
        rows = []
        for fitted, panel in released:
            index = next(i for i, row in enumerate(fitted) if row.tobytes() == key)
            rows.append((panel.calls[index], panel.alt_probability[index], panel.support[index]))
        agreed = np.array(rows[0][0], copy=True)
        for other_calls, _p, _s in rows[1:]:
            agreed[agreed != other_calls] = -1
        calls[target] = agreed
        alt_probability[target] = np.mean([row[1] for row in rows], axis=0)
        support[target] = np.min([row[2] for row in rows], axis=0)
    slots = np.max(np.vstack([panel.unknown_slots for _fitted, panel in released]), axis=0).astype(np.int8)
    order = np.lexsort(np.vstack([alt_probability.T[::-1], calls.T[::-1]]))
    return (
        replace(
            representative,
            calls=np.ascontiguousarray(calls[order]),
            alt_probability=np.ascontiguousarray(alt_probability[order]),
            support=np.ascontiguousarray(support[order]),
            unknown_slots=slots,
            unknown_mass=unknown_copy_mass(slots, representative.has_depth),
        ),
        int(k - len(shared)),
    )


def select_block(
    task: BlockTask,
    likelihoods: np.ndarray,
    observed: np.ndarray,
    settings: tuple[SelectionParameters, DiscoveryParameters, float],
    *,
    reverse: bool,
    census_sink=None,
    threads: int | Callable[[], int] = 1,
) -> BlockSelection:
    """Select one block's panel (``likelihoods``/``observed``: its discovery-site columns). ``threads``: an int,
    or the caller's live share re-read at the block's lane boundaries."""
    config, discovery_config, redraw_rate = settings
    prepared = PreparedBlock(
        likelihoods,
        observed,
        task.intervals,
        redraw_rate,
        config.model.unknown_mass,
        threads=threads,
        block_id=task.label or str(task.block),
    )
    incumbent = task.incumbent
    ties = TieCounter()
    census_key = task.label or None
    starts, bank, bank_summary = candidate_bank(
        likelihoods,
        observed,
        task.latent,
        task.proposals,
        discovery_config,
        config.bank,
        incumbent_panel=None if incumbent is None else incumbent.fit.panel,
        ties=ties,
        census_key=census_key,
        alternatives=task.alternatives,
        reverse=reverse,
        census_sink=census_sink,
        threads=prepared.threads,
    )
    memo = LikelihoodMemo(task.memo)
    context = None if task.founders is None else ContextFounders(task.founders)
    tolerance = config.search.decision_tolerance
    searched = starts
    if len(starts) > 1:
        # The fitted starts' selection objectives decide which starts are searched: the best tie
        # class, fewer founders inside it (a stated prior); one search per distinct fitted panel.
        fits = [
            fitted_start(start, prepared, config.search, memo, context, reverse=reverse, census_sink=census_sink)
            for start in starts
        ]
        values = [selection_objective(fit, context) for fit in fits]
        classes = tie_classes(
            values,
            tolerance,
            descending=True,
            motion=DECISION_MOTION,
            reverse=fixed_policy("selection.start.set", tolerance, reverse=reverse).reverse,
        )
        best = [n for n in range(len(starts)) if classes.classes[n] == 0]
        fewest = min(len(fits[n].panel) for n in best)
        distinct: dict[tuple[bytes, bytes], np.ndarray] = {}
        for n in best:
            if len(fits[n].panel) == fewest:
                distinct.setdefault((fits[n].panel.tobytes(), fits[n].frequencies.tobytes()), starts[n])
        searched = [distinct[key] for key in sorted(distinct)]
        ties.note(
            "ls.start.set", inside_tie=len(searched) > 1, wide=classes.wide_classes, boundary=len(classes.boundary_gaps)
        )
    outcomes = [
        search_panel(
            start,
            bank,
            prepared,
            config.search,
            incumbent=incumbent,
            founders=task.founders,
            memo=memo,
            ties=ties,
            census_key=census_key if n == 0 or census_key is None else f"{census_key}/s{n}",
            reverse=reverse,
            census_sink=census_sink,
        )
        for n, start in enumerate(searched)
    ]
    attempted, alternatives = _best_outcome(outcomes, tolerance, ties, reverse=reverse)
    retained = False
    incumbent_margin = None
    if incumbent is not None:
        # Keep-incumbent rule (hysteresis): the new search replaces the previous round's outcome
        # only when it beats it by more than the decision tolerance (both under this round's
        # context code).
        incumbent_objective = selection_objective(incumbent.fit, context)
        incumbent_margin = float(attempted.selection_objective - incumbent_objective)
        tolerance = config.search.decision_tolerance
        retained = not improves(attempted.selection_objective, incumbent_objective, tolerance)
        ties.note(
            "ls.keep_incumbent",
            inside_tie=bool(
                abs(incumbent_margin) <= tolerance and not np.array_equal(attempted.fit.panel, incumbent.fit.panel)
            ),
            boundary=int(abs(abs(incumbent_margin) - tolerance) <= BOUNDARY_GUARD * DECISION_MOTION),
        )
        if census_key is not None and census_enabled(sink=census_sink):
            census(
                "ls.keep",
                f"{census_key}/keep",
                (-incumbent_objective, -attempted.selection_objective),
                sink=census_sink,
            )
    outcome = incumbent if retained else attempted
    assert outcome is not None
    # Context snaps decided inside a tie class (by content), for the selected fit (release 2.12).
    ties.note("ls.snap.tie", inside_tie=bool(getattr(outcome.fit, "snap_ties", 0)))
    # The release is a function of the selected (panel, frequencies) and the block data, and in a feedback round
    # also of the round's context witnesses (release 2.12): an incumbent's release is reused when they are equal and
    # the round has no witnesses and the incumbent's release used none (it may have seen another round's context).
    reuse = (
        incumbent is not None
        and task.incumbent_release is not None
        and task.witness is None
        and getattr(task.incumbent_release, "context_cells", 0) == 0
        and np.array_equal(outcome.fit.panel, incumbent.fit.panel)
        and np.array_equal(outcome.fit.frequencies, incumbent.fit.frequencies)
    )
    path_release = (
        task.incumbent_release
        if reuse
        else release(
            outcome.fit,
            prepared,
            config.release,
            ties=ties,
            census_key=census_key,
            context=context,
            witness=task.witness,
            census_sink=census_sink,
        )
    )
    assert path_release is not None
    panel = local_panel(task.block, task.site_start, task.positions, task.kept, outcome.fit, path_release)
    withheld_by_ties = 0
    if alternatives and not retained:
        panel, withheld_by_ties = _consensus_panel(
            task, outcome, path_release, alternatives, prepared, config.release, context, census_sink=census_sink
        )
    tie_counts = {
        entry.kind: {name: getattr(entry, name) for name in ("decisions", "inside_tie", "wide", "boundary")}
        for entry in ties.record().entries
    }
    summary = SelectionSummary(
        incumbent_retained=bool(retained),
        release_reused=bool(reuse),
        bank=bank_summary,
        attempted_objective=float(attempted.fit.objective),
        selection_objective=float(selection_objective(outcome.fit, context)),
        search_start=attempted.search_start,
        search_rounds=len(attempted.trace),
        start_was_incumbent_panel=bool(attempted.start_was_incumbent_panel),
        incumbent_margin=incumbent_margin,
        round_backstop=bool(attempted.round_backstop),
        update_backstop=bool(outcome.fit.update_backstop),
        trajectory_best=bool(outcome.fit.selected == "trajectory_best"),
        context_snaps=int(getattr(outcome.fit, "context_snaps", 0)),
        context_founders=None if context is None else len(context),
        free_rows=len(outcome.fit.panel) if context is None else context.free(outcome.fit.panel),
        tie_events=ties.record(),
        starts=len(starts),
        searched_starts=len(searched),
        tied_alternatives=len(alternatives),
        withheld_by_ties=withheld_by_ties,
        # Context-witness rescues of the representative release, before any tied-panel consensus withholds them.
        context_witnessed=int(path_release.context_cells),
        cut_overflow=sum(
            int(tie_counts.get(f"ls.rank.{kind}.overflow", {}).get("inside_tie", 0))
            for kind in ("drop", "merge", "add")
        ),
        memo_hits=MemoHits(**memo.hits),
    )
    return BlockSelection(
        index=task.index,
        panel=panel,
        outcome=outcome,
        release=path_release,
        memo=memo.state() if task.keep_memo else None,
        summary=summary,
    )


def _memo_bytes(memo, sites, founders) -> int:
    """Approximate bytes of a block's likelihood memo state (bit-packed fits, scores, bounds)."""
    if not memo:
        return 0
    panel = founders * -(-int(sites) // 8) + 640
    return len(memo.fits) * panel + len(memo.scores) * 160 + len(memo.bounds) * (4 * founders * founders + 200)


def _founders(task: BlockTask) -> int:
    """The largest panel a block task starts from (latent, proposals or incumbent)."""
    return max(
        [len(task.latent), *(len(panel) for panel in task.proposals), *(len(panel) for panel in task.alternatives)]
        + ([len(task.incumbent.fit.panel)] if task.incumbent is not None else [])
    )


def block_task_bytes(task: BlockTask, samples: int) -> int:
    """Planned peak additional bytes of one block's selection in a pool worker: the block's GL
    slice, emission tables and lane layouts, the trajectory scratch for the largest panel the
    search may reach (one lane block per thread; planned for a thread per lane block, the most a
    task can claim), the release E-step's batch rows, the candidate bank's discovery workspace
    (context rounds) and the incoming likelihood memo (and its growth). Private worker residence is
    planned and held separately for the pool's lifetime."""
    sites = len(task.columns)
    peak = _founders(task) + SEARCH_GROWTH
    data = 232 * int(samples) * sites
    trajectories = workspace_bytes(sites, peak, threads=-(-int(samples) // LANE_WIDTH))
    release = 4 * 8 * RELEASE_SAMPLE_BATCH * (peak + 1) * sites
    bank = search_bytes(samples, sites, peak) if task.proposals else 0
    return TASK_WORKSPACE_BASE_BYTES + data + trajectories + release + bank + 2 * _memo_bytes(task.memo, sites, peak)


def select_task(
    block: BlockTask,
    likelihoods: np.ndarray | CodedLikelihoods,
    observed: np.ndarray | PairArray,
    settings: tuple[SelectionParameters, DiscoveryParameters, float],
    *,
    reverse: bool,
    census_sink=None,
    threads: int | Callable[[], int] = 1,
) -> BlockSelection:
    """Select a block from immutable contig arrays, making the exact private 3.5 evidence copy.

    Inputs are float64[N,L,3] GL and bool[N,L] observed. Only columns in block.columns
    are read. Missing cells become the literal uniform value before both fitting and
    input-key construction. Mutates no input; shape/index failures propagate.
    """
    local_gl = gather_likelihoods(likelihoods, block.columns)
    local_observed = np.asarray(observed[:, block.columns], dtype=bool, order="C")
    local_gl[~local_observed] = 1.0 / 3.0
    return select_block(
        block, local_gl, local_observed, settings, reverse=reverse, census_sink=census_sink, threads=threads
    )
