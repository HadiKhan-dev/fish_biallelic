"""One local-selection round: every informative block's path search, release and panel.

Per block: the candidate bank (context rounds) from the raw discovery panel, every earlier
round's selected calls and the newest proposals; the path search from the bank's start, or from
the previous round's selected fit (the incumbent) when the start's fit reaches the incumbent's
panel or the incumbent scores better by more than the decision tolerance; the incumbent is kept
unless the new search beats it by more than the decision tolerance (release 2.1: 2.0 kept it only
when it beat the new search, which the monotone climb made impossible, so every upstream change
re-ran the greedy search into whatever optimum it reached); the release is reused when the
selected (panel, frequencies) equal the incumbent's. Blocks run one task each in a pool whose
workers claim freed cores at phase boundaries (or as cross-node batches); completed blocks are
recorded in chunks as they arrive. Every block task reserves its planned peak memory with the
run's memory governor before it starts (``block_task_bytes``); pacing changes timing only.

Release 2.2 (tied sets decided by the data): the bank returns a set of starts (the cavity-best tie
set of every source); each start is fitted and the best tie class of their selection objectives
(the stated prior fewer founders inside it) is searched from every member; the outcomes are
compared by the selection objective's tie classes, fewer founders inside the best class. Distinct
panels left tied after that are released as their consensus (``_consensus_panel``: founder rows
not shared by every alternative are withheld; shared rows keep the calls all alternatives' releases
agree on; summary ``tied_alternatives``, ``withheld_by_ties``). The internal representative fit (the
next round's incumbent) is the smallest canonical key, a convention reversed under
maths.ties.reverse_conventions that the released panel does not depend on.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time

import numpy as np

from ..discovery.fixed_k.workspace import search_bytes
from ..execution import BudgetedPool, claim_threads, memory_worker_limit, reserve_memory
from ..execution.batch_queue import BatchKind, map_batches, queue_key
from ..maths.ties import BOUNDARY_GUARD, TieCounter, census, census_enabled, improves, tie_classes
from ..model.local_panels import unknown_copy_mass
from . import records
from .bank import candidate_bank
from .objective import ContextFounders, selection_objective
from .path_model import LANE_WIDTH, RELEASE_SAMPLE_BATCH, PreparedBlock, workspace_bytes
from .release import PathRelease, local_panel, release
from .search import DECISION_MOTION, LikelihoodMemo, SearchOutcome, fitted_start, search_panel


@dataclass(frozen=True)
class BlockTask:
    """Inputs of one block's selection (columns: contig site indices of its discovery sites;
    label: "contig/round/i<index>", the block's key in diagnostics and the valley census;
    alternatives: discovery's NLL-tied alternative latent panels, release 2.2)."""

    index: int
    block: int
    site_start: int
    positions: np.ndarray
    kept: np.ndarray
    columns: np.ndarray
    intervals: np.ndarray
    latent: np.ndarray
    proposals: tuple
    founders: np.ndarray | None
    incumbent: SearchOutcome | None
    incumbent_release: PathRelease | None
    memo: dict | None
    keep_memo: bool
    label: str = ""
    alternatives: tuple = ()


@dataclass(frozen=True)
class BlockSelection:
    """A block's selection in one round: its panel, outcome (fit + search), release, the memo
    for the next round (None in the last round) and a small summary."""

    index: int
    panel: object
    outcome: SearchOutcome
    release: PathRelease
    memo: dict | None
    summary: dict


def _best_outcome(outcomes, tolerance, ties) -> tuple[SearchOutcome, list]:
    """(the selected outcome of a multi-start search, the tied alternatives): the best tie class of
    the selection objectives, fewer founders inside it (a stated prior); the distinct panels still
    tied (one outcome each, the representative included; empty when one panel remains). The
    smallest canonical key represents them (a convention: tie_classes reverses it under
    maths.ties.reverse_conventions; the release keeps only what all of them agree on)."""
    if len(outcomes) == 1:
        return outcomes[0], []
    classes = tie_classes([outcome.selection_objective for outcome in outcomes], tolerance,
                          secondary=[(len(outcome.fit.panel),) for outcome in outcomes],
                          content=[outcome.fit.panel.tobytes() for outcome in outcomes],
                          descending=True, motion=DECISION_MOTION)
    winner = outcomes[int(classes.order[0])]
    fewest = len(winner.fit.panel)
    tied = {}
    for n in classes.order:
        outcome = outcomes[int(n)]
        if classes.classes[int(n)] == 0 and len(outcome.fit.panel) == fewest:
            tied.setdefault(outcome.fit.panel.tobytes(), outcome)
    ties.note("ls.start.outcome", inside_tie=len(tied) > 1, wide=classes.wide_classes,
              boundary=len(classes.boundary_gaps))
    return winner, (list(tied.values()) if len(tied) > 1 else [])


def _consensus_panel(task, outcome, path_release, alternatives, prepared, release_config):
    """The LocalPanel released for tied alternative panels (release 2.2), and the founder rows it
    withholds. A founder row that every alternative contains keeps a call where all alternatives'
    releases of that row agree (-1 elsewhere), their mean ALT probability and their smallest
    support; the other rows are withheld (calls -1, ALT probability 1/2, support 0); unknown slots
    are the largest over the alternatives (the conservative side for completion's veto); rows are
    in the order of their released content. Nothing depends on which alternative represents the
    others."""
    released = []
    for alternative in alternatives:
        own = path_release if alternative is outcome else release(alternative.fit, prepared, release_config)
        released.append((alternative.fit.panel, local_panel(task.block, task.site_start, task.positions,
                                                            task.kept, alternative.fit, own)))
    shared = None
    for fitted, _panel in released:
        rows = {row.tobytes() for row in fitted}
        shared = rows if shared is None else shared & rows
    representative = next(panel for fitted, panel in released
                          if fitted is outcome.fit.panel or np.array_equal(fitted, outcome.fit.panel))
    k, length = representative.calls.shape
    calls = np.full((k, length), -1, dtype=np.int8)
    alt_probability = np.full((k, length), .5)
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
    return (replace(representative, calls=np.ascontiguousarray(calls[order]),
                    alt_probability=np.ascontiguousarray(alt_probability[order]),
                    support=np.ascontiguousarray(support[order]), unknown_slots=slots,
                    unknown_mass=unknown_copy_mass(slots, representative.has_depth)),
            int(k - len(shared)))


def select_block(task: BlockTask, likelihoods, observed, settings) -> BlockSelection:
    """Select one block's panel (``likelihoods``/``observed``: its discovery-site columns)."""
    config, discovery_config, redraw_rate = settings
    started = time.monotonic()
    claim_threads()
    prepared = PreparedBlock(likelihoods, observed, task.intervals, redraw_rate,
                             config.model.unknown_mass)
    incumbent = task.incumbent
    ties = TieCounter()
    census_key = task.label or None
    starts, bank, bank_summary = candidate_bank(
        likelihoods, observed, task.latent, task.proposals, discovery_config, config.bank,
        incumbent_panel=None if incumbent is None else incumbent.fit.panel, ties=ties,
        census_key=census_key, alternatives=task.alternatives)
    memo = LikelihoodMemo(task.memo)
    context = None if task.founders is None else ContextFounders(task.founders)
    tolerance = config.search.decision_tolerance
    searched = starts
    if len(starts) > 1:
        # The fitted starts' selection objectives decide which starts are searched: the best tie
        # class, fewer founders inside it (a stated prior); one search per distinct fitted panel.
        fits = [fitted_start(start, prepared, config.search, memo) for start in starts]
        values = [selection_objective(fit, context) for fit in fits]
        classes = tie_classes(values, tolerance, descending=True, motion=DECISION_MOTION)
        best = [n for n in range(len(starts)) if classes.classes[n] == 0]
        fewest = min(len(fits[n].panel) for n in best)
        distinct = {}
        for n in best:
            if len(fits[n].panel) == fewest:
                distinct.setdefault((fits[n].panel.tobytes(), fits[n].frequencies.tobytes()), starts[n])
        searched = [distinct[key] for key in sorted(distinct)]
        ties.note("ls.start.set", inside_tie=len(searched) > 1, wide=classes.wide_classes,
                  boundary=len(classes.boundary_gaps))
    outcomes = [search_panel(start, bank, prepared, config.search, incumbent=incumbent,
                             founders=task.founders, memo=memo, ties=ties,
                             census_key=census_key if n == 0 or census_key is None else f"{census_key}/s{n}")
                for n, start in enumerate(searched)]
    attempted, alternatives = _best_outcome(outcomes, tolerance, ties)
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
        ties.note("ls.keep_incumbent",
                  inside_tie=bool(abs(incumbent_margin) <= tolerance
                                  and not np.array_equal(attempted.fit.panel, incumbent.fit.panel)),
                  boundary=int(abs(abs(incumbent_margin) - tolerance) <= BOUNDARY_GUARD * DECISION_MOTION))
        if census_key is not None and census_enabled():
            census("ls.keep", f"{census_key}/keep", (-incumbent_objective, -attempted.selection_objective))
    outcome = incumbent if retained else attempted
    # The release is a function of the selected (panel, frequencies) and the block data: an
    # incumbent's release is reused when they are equal.
    reuse = (incumbent is not None and task.incumbent_release is not None
             and np.array_equal(outcome.fit.panel, incumbent.fit.panel)
             and np.array_equal(outcome.fit.frequencies, incumbent.fit.frequencies))
    path_release = (task.incumbent_release if reuse
                    else release(outcome.fit, prepared, config.release, ties=ties, census_key=census_key))
    panel = local_panel(task.block, task.site_start, task.positions, task.kept, outcome.fit,
                        path_release)
    withheld_by_ties = 0
    if alternatives and not retained:
        panel, withheld_by_ties = _consensus_panel(task, outcome, path_release, alternatives, prepared,
                                                   config.release)
    summary = dict(
        incumbent_retained=bool(retained), release_reused=bool(reuse),
        bank=None if bank_summary is None else asdict(bank_summary),
        attempted_objective=float(attempted.fit.objective),
        selection_objective=float(selection_objective(outcome.fit, context)),
        search_start=attempted.search_start, search_rounds=len(attempted.trace),
        start_was_incumbent_panel=bool(attempted.start_was_incumbent_panel),
        incumbent_margin=incumbent_margin,
        round_backstop=bool(attempted.round_backstop),
        update_backstop=bool(outcome.fit.update_backstop),
        context_founders=None if context is None else len(context),
        free_rows=len(outcome.fit.panel) if context is None else context.free(outcome.fit.panel),
        tie_events=ties.record(), starts=len(starts), searched_starts=len(searched),
        tied_alternatives=len(alternatives), withheld_by_ties=withheld_by_ties,
        cut_overflow=sum(int(ties.counts.get(f"ls.rank.{kind}.overflow", {}).get("inside_tie", 0))
                         for kind in ("drop", "merge", "add")),
        memo_hits=dict(memo.hits), seconds=time.monotonic() - started)
    return BlockSelection(index=task.index, panel=panel, outcome=outcome, release=path_release,
                          memo=memo.state() if task.keep_memo else None, summary=summary)


# Private process residence (runtime, compiled code and retained heap), held until its pool joins.
# Cold bounded K29 workers retained up to 674 MB after trimming; 768 MiB covers that observed residence.
WORKER_RESIDENT_BYTES = 768 << 20
# Active first-use/transient workspace beyond the named arrays below. Historical TroMau incremental
# RSS reached 3.023 GB with 356 MB named workspace. With the bounded fixture's 106 MB starting private
# memory, 2 GiB workspace plus 768 MiB residence covers that projection by 180 MB. The archived startup
# baseline is unavailable; these mixed measurements do not isolate JIT compilation from heap retention.
TASK_WORKSPACE_BASE_BYTES = 2048 << 20
# The search can outgrow its initial panel; size transient trajectories for this many extra founders.
SEARCH_GROWTH = 8


def _memo_bytes(memo, sites, founders) -> int:
    """Approximate bytes of a block's likelihood memo state (bit-packed fits, scores, bounds)."""
    if not memo:
        return 0
    panel = founders * -(-int(sites) // 8) + 640
    return (len(memo["fits"]) * panel + len(memo["scores"]) * 160
            + len(memo["bounds"]) * (4 * founders * founders + 200))


def _founders(task: BlockTask) -> int:
    """The largest panel a block task starts from (latent, proposals or incumbent)."""
    return max([len(task.latent), *(len(panel) for panel in task.proposals),
                *(len(panel) for panel in task.alternatives)]
               + ([len(task.incumbent.fit.panel)] if task.incumbent is not None else []))


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
    return (TASK_WORKSPACE_BASE_BYTES + data + trajectories + release + bank
            + 2 * _memo_bytes(task.memo, sites, peak))


def _select_task(task, shared, budget):
    """Pool task: one block (worker side); arrays are the round's shared evidence."""
    block, settings = task
    likelihoods = np.ascontiguousarray(shared["likelihoods"][:, block.columns], dtype=np.float64)
    observed = np.asarray(shared["observed"][:, block.columns], dtype=bool, order="C")
    # founder_view shares immutable raw GLs; this private block copy also enters the
    # persisted search key, which must retain the literal uniform missing-cell values.
    likelihoods[~observed] = 1.0 / 3.0
    return select_block(block, likelihoods, observed, settings)


# ---------------------------------------------------------------------------------------------
# Cross-node batches (kind "local_fit"): bundles of blocks with the evidence slice they read.
# ---------------------------------------------------------------------------------------------
_BUNDLE_BYTES = 256 << 20
_BUNDLE_TASKS = 256


def _bundles(tasks, arrays, worker_memory_gb):
    """Pack consecutive tasks with the site slice they read (about 256 MiB of evidence, at most
    256 tasks; a task is never split); columns become slice-relative. (``worker_memory_gb`` is
    part of the batch-kind interface; the executing allocation sizes its workers from the memory
    governor and the tasks' own estimates.)"""
    likelihoods, observed = arrays
    bytes_per_site = (likelihoods.shape[0] * likelihoods.shape[2] * likelihoods.dtype.itemsize
                      + observed.shape[0] * observed.dtype.itemsize)
    limit = max(1, _BUNDLE_BYTES // bytes_per_site)
    group, left, right = [], None, None

    def package():
        adjusted = [(replace(block, columns=block.columns - left), settings)
                    for block, settings in group]
        return dict(tasks=adjusted, arrays=(np.ascontiguousarray(likelihoods[:, left:right]),
                                            np.ascontiguousarray(observed[:, left:right])))

    for task in tasks:
        lo, hi = int(task[0].columns[0]), int(task[0].columns[-1]) + 1
        if group and (len(group) >= _BUNDLE_TASKS or max(right, hi) - min(left, lo) > limit):
            yield package()
            group, left, right = [], None, None
        group.append(task)
        left = lo if left is None else min(left, lo)
        right = hi if right is None else max(right, hi)
    if group:
        yield package()


def _execute_bundle(payload, cpus):
    """Run one bundle in the executing allocation (a BudgetedPool over its tasks, each reserving
    its planned peak memory; workers beyond what the memory limit admits would only wait)."""
    from ..execution import CpuBudget
    tasks = payload["tasks"]
    likelihoods, observed = payload["arrays"]
    sizes = [block_task_bytes(task, likelihoods.shape[0]) for task, _settings in tasks]
    workers = max(1, min(cpus, len(tasks)))
    planned = [size + WORKER_RESIDENT_BYTES for size in sizes]
    # The pool's shared copy is one allocation, not one copy per worker. Preserve graded headroom.
    planned[planned.index(max(planned))] += int(likelihoods.nbytes + observed.nbytes)
    workers = max(1, min(workers, memory_worker_limit(planned, workers, label="local-fit-bundle")))
    with BudgetedPool(CpuBudget.fixed(cpus), processes=workers,
                      shared=dict(likelihoods=likelihoods, observed=observed),
                      label="local-fit-bundle") as pool:
        return list(pool.imap_unordered(_select_task, tasks, task_bytes=sizes))


LOCAL_FIT = BatchKind(name="local_fit", bundles=_bundles, execute=_execute_bundle)


def run_round(round_scope, tasks, settings, evidence, budget, *, queue=None, label=""):
    """Select every task's block (resuming the recorded chunks); returns BlockSelections by task
    index. ``round_scope``: the round's bound RecordScope; ``evidence``: founder view."""
    writer = records.SelectionChunks(round_scope)
    done = {selection.index: selection for selection in writer.load()}
    pending = [task for task in tasks if task.index not in done]
    if pending:
        print(f"[Local selection] {label}: {len(pending)} blocks"
              + (f" ({len(done)} recorded earlier)" if done else ""), flush=True)
        payloads = [(task, settings) for task in pending]
        pooled_results = None
        try:
            if queue is not None:
                results = map_batches(LOCAL_FIT, payloads, (evidence.likelihoods, evidence.observed),
                                      budget, queue=queue, key=queue_key(round_scope, "select"))
            else:
                # Longest first by the path model's work per pass (sites x states^2): the round's
                # tail then holds the small blocks (order changes timing only).
                pending.sort(key=lambda task: (-len(task.columns) * (_founders(task) + 1) ** 2,
                                               task.index))
                payloads = [(task, settings) for task in pending]
                sizes = [block_task_bytes(task, evidence.likelihoods.shape[0]) for task in pending]
                workers = max(1, min(int(budget.threads()), len(payloads)))
                planned = [size + WORKER_RESIDENT_BYTES for size in sizes]
                # Plan the pool's shared copy once, before reserving its worker residence.
                planned[planned.index(max(planned))] += int(evidence.likelihoods.nbytes + evidence.observed.nbytes)
                workers = max(1, min(workers, memory_worker_limit(planned, workers,
                                                                  label="local-selection")))
                shared = dict(likelihoods=evidence.likelihoods, observed=evidence.observed)
                results = pooled_results = _pooled(budget, workers, shared, payloads, sizes)
            for count, selection in enumerate(results, 1):
                done[selection.index] = selection
                writer.add(selection)
                if count % 20 == 0 or count == len(pending):
                    print(f"[Local selection] {label}: completed {count}/{len(pending)}", flush=True)
        finally:
            try:
                # An exception while recording a yielded result must close the suspended generator now.
                if pooled_results is not None:
                    pooled_results.close()
            finally:
                writer.flush()
    writer.complete(len(tasks))
    selections = [done[task.index] for task in tasks]
    _report(selections, label)
    return selections


def _pooled(budget, workers, shared, payloads, sizes):
    # Both resources start on first iteration; closing an unstarted generator owns no pool or hold.
    with BudgetedPool(budget, processes=workers, shared=shared, label="local-selection") as pool:
        yield from pool.imap_unordered(_select_task, payloads, task_bytes=sizes)


def _report(selections, label):
    counts = dict(blocks=len(selections),
                  round_backstop=sum(s.summary["round_backstop"] for s in selections),
                  update_backstop=sum(s.summary["update_backstop"] for s in selections),
                  bank_backstop=sum(bool(s.summary["bank"] is not None
                                         and s.summary["bank"]["budget_bound"]) for s in selections),
                  incumbent_retained=sum(bool(s.summary["incumbent_retained"]) for s in selections))
    hits: dict = {}
    for selection in selections:
        for name, value in selection.summary["memo_hits"].items():
            hits[name] = hits.get(name, 0) + value
    print(f"[Local selection] {label}: backstop hits {counts}; likelihood memo hits {hits}",
          flush=True)
    ties = tie_totals(selections)
    if ties:
        print(f"[Local selection] {label}: tie events (decisions/inside tie/wide/boundary) "
              + "; ".join(f"{kind} {v['decisions']}/{v['inside_tie']}/{v['wide']}/{v['boundary']}"
                          for kind, v in sorted(ties.items())), flush=True)


def tie_totals(selections) -> dict:
    """Per decision kind, the summed tie events of a round's block summaries (blocks recorded before
    release 2.1 have none)."""
    totals = TieCounter()
    for selection in selections:
        events = selection.summary.get("tie_events") or {}
        for kind, values in events.items():
            entry = totals.counts.setdefault(kind, dict(decisions=0, inside_tie=0, wide=0, boundary=0))
            for name, value in values.items():
                entry[name] = entry.get(name, 0) + int(value)
    return totals.record()
