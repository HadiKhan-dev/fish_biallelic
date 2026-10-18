"""One local-selection round: every informative block's path search, release and panel.

Per block: the candidate bank (context rounds) from the raw discovery panel, every earlier
round's selected calls and the newest proposals; the path search from the bank's start, or from
the previous round's selected fit (the incumbent) when it scores better; the incumbent is kept
when the new search does not beat it by more than the tolerance; the release is reused when the
selected (panel, frequencies) equal the incumbent's. Blocks run one task each in a pool whose
workers claim freed cores at phase boundaries (or as cross-node batches); completed blocks are
recorded in chunks as they arrive. Every block task reserves its planned peak memory with the
run's memory governor before it starts (``block_task_bytes``); pacing changes timing only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time

import numpy as np

from ..discovery.fixed_k.workspace import search_bytes
from ..execution import BudgetedPool, claim_threads, memory_worker_limit
from ..execution.batch_queue import BatchKind, map_batches, queue_key
from . import records
from .bank import candidate_bank
from .objective import ContextFounders, selection_objective
from .path_model import LANE_WIDTH, RELEASE_SAMPLE_BATCH, PreparedBlock, workspace_bytes
from .release import PathRelease, local_panel, release
from .search import LikelihoodMemo, SearchOutcome, search_panel


@dataclass(frozen=True)
class BlockTask:
    """Inputs of one block's selection (columns: contig site indices of its discovery sites)."""

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


def select_block(task: BlockTask, likelihoods, observed, settings) -> BlockSelection:
    """Select one block's panel (``likelihoods``/``observed``: its discovery-site columns)."""
    config, discovery_config, generations = settings
    started = time.monotonic()
    claim_threads()
    prepared = PreparedBlock(likelihoods, observed, task.intervals, generations,
                             config.model.unknown_mass)
    incumbent = task.incumbent
    start, bank, bank_summary = candidate_bank(
        likelihoods, observed, task.latent, task.proposals, discovery_config, config.bank,
        incumbent_panel=None if incumbent is None else incumbent.fit.panel)
    memo = LikelihoodMemo(task.memo)
    attempted = search_panel(start, bank, prepared, config.search, incumbent=incumbent,
                             founders=task.founders, memo=memo)
    context = None if task.founders is None else ContextFounders(task.founders)
    retained = incumbent is not None and (
        selection_objective(incumbent.fit, context)
        > attempted.selection_objective + config.search.tolerance)
    outcome = incumbent if retained else attempted
    # The release is a function of the selected (panel, frequencies) and the block data: an
    # incumbent's release is reused when they are equal.
    reuse = (incumbent is not None and task.incumbent_release is not None
             and np.array_equal(outcome.fit.panel, incumbent.fit.panel)
             and np.array_equal(outcome.fit.frequencies, incumbent.fit.frequencies))
    path_release = task.incumbent_release if reuse else release(outcome.fit, prepared, config.release)
    panel = local_panel(task.block, task.site_start, task.positions, task.kept, outcome.fit,
                        path_release)
    summary = dict(
        incumbent_retained=bool(retained), release_reused=bool(reuse),
        bank=None if bank_summary is None else asdict(bank_summary),
        attempted_objective=float(attempted.fit.objective),
        selection_objective=float(selection_objective(outcome.fit, context)),
        search_start=attempted.search_start, search_rounds=len(attempted.trace),
        round_backstop=bool(attempted.round_backstop),
        update_backstop=bool(outcome.fit.update_backstop),
        context_founders=None if context is None else len(context),
        free_rows=len(outcome.fit.panel) if context is None else context.free(outcome.fit.panel),
        memo_hits=dict(memo.hits), seconds=time.monotonic() - started)
    return BlockSelection(index=task.index, panel=panel, outcome=outcome, release=path_release,
                          memo=memo.state() if task.keep_memo else None, summary=summary)


# Memory plan of one block task (the governor reservation), calibrated on AulStu chr3 (N 290,
# L 200, one thread: measured block peaks 0.20-0.23 GB above the worker's baseline at path K 13-19;
# estimate ~0.6 GB at start K 22) and reconstruct_actm3 chr8 (N 116: a worker's first task, which
# also loads the kernels, peaks at up to 0.31 GB; later tasks at up to 0.05 GB). The search's panel
# can outgrow its start by several rows (chr3 initial round: up to +12, mean +3.6), so the
# trajectories are planned for SEARCH_GROWTH more.
TASK_BASE_BYTES = 320 << 20
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
    return max([len(task.latent), *(len(panel) for panel in task.proposals)]
               + ([len(task.incumbent.fit.panel)] if task.incumbent is not None else []))


def block_task_bytes(task: BlockTask, samples: int) -> int:
    """Planned peak additional bytes of one block's selection in a pool worker: the block's GL
    slice, emission tables and lane layouts, the trajectory scratch for the largest panel the
    search may reach (one lane block per thread; planned for a thread per lane block, the most a
    task can claim), the release E-step's batch rows, the candidate bank's discovery workspace
    (context rounds) and the incoming likelihood memo (and its growth)."""
    sites = len(task.columns)
    peak = _founders(task) + SEARCH_GROWTH
    data = 232 * int(samples) * sites
    trajectories = workspace_bytes(sites, peak, threads=-(-int(samples) // LANE_WIDTH))
    release = 4 * 8 * RELEASE_SAMPLE_BATCH * (peak + 1) * sites
    bank = search_bytes(samples, sites, peak) if task.proposals else 0
    return (TASK_BASE_BYTES + data + trajectories + release + bank
            + 2 * _memo_bytes(task.memo, sites, peak))


def _select_task(task, shared, budget):
    """Pool task: one block (worker side); arrays are the round's shared evidence."""
    block, settings = task
    likelihoods = np.ascontiguousarray(shared["likelihoods"][:, block.columns], dtype=np.float64)
    observed = np.asarray(shared["observed"][:, block.columns], dtype=bool, order="C")
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
    workers = max(1, min(workers, memory_worker_limit(sizes, workers, label="local-fit-bundle")))
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
                workers = max(1, min(workers, memory_worker_limit(sizes, workers,
                                                                  label="local-selection")))
                pool = BudgetedPool(budget, processes=workers,
                                    shared=dict(likelihoods=evidence.likelihoods,
                                                observed=evidence.observed),
                                    label="local-selection")
                results = _pooled(pool, payloads, sizes)
            for count, selection in enumerate(results, 1):
                done[selection.index] = selection
                writer.add(selection)
                if count % 20 == 0 or count == len(pending):
                    print(f"[Local selection] {label}: completed {count}/{len(pending)}", flush=True)
        finally:
            writer.flush()
    writer.complete(len(tasks))
    selections = [done[task.index] for task in tasks]
    _report(selections, label)
    return selections


def _pooled(pool, payloads, sizes):
    with pool:
        yield from pool.imap_unordered(_select_task, payloads, task_bytes=sizes)


def _report(selections, label):
    counts = dict(blocks=len(selections),
                  round_backstop=sum(s.summary["round_backstop"] for s in selections),
                  update_backstop=sum(s.summary["update_backstop"] for s in selections),
                  bank_backstop=sum(bool(s.summary["bank"] is not None
                                         and s.summary["bank"]["budget_bound"]) for s in selections))
    hits: dict = {}
    for selection in selections:
        for name, value in selection.summary["memo_hits"].items():
            hits[name] = hits.get(name, 0) + value
    print(f"[Local selection] {label}: backstop hits {counts}; likelihood memo hits {hits}",
          flush=True)
