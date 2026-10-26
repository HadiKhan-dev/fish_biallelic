"""Raw block discovery stage: every informative block of every contig, one block pool.

Entry points:
* :func:`run` -- discover the pending blocks of the given contigs (records ``discovery/plan`` and
  ``discovery/blocks/<n>``; resumes interrupted contigs with their missing blocks only).
* :func:`discover_block` -- one block in memory (also used by simulation-template building).

Discovery depends only on the contig's evidence (founder view: samples flagged by the
contamination QC are unobserved). Blocks are independent and embarrassingly parallel; each block
task reserves its planned peak memory with the run's memory governor (``block_task_bytes``).
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import math
import time

import numpy as np

from ..evidence import records as evidence_records
from ..execution import (BudgetedPool, claim_threads, malloc_trim, memory_governor, memory_worker_limit,
                         release_memory, reserve_memory, resident_bytes)
from ..execution.memory import adopted, held_context
from . import records

from .config import DiscoveryConfig
from .residual_births import MASS_BAND
from .fixed_k.workspace import search_bytes
from .release import DiscoveredBlock, materialise
from .search import search_block, diagnostic_block, diagnostic_phase, backstop_counts


@dataclass(frozen=True)
class BlockEvidence:
    """The founder-view evidence of one discovery block.

    block, site_start: block index and its first contig site; positions: int64[L];
    kept: bool[L] discovery sites; likelihoods: float64[N, L, 3] normalised GLs (zero depth
    exactly 1/3); observed: bool[N, L] positive depth (False for excluded samples).
    sample_ids: the samples' names (release 2.1; optional): the search runs in the run's canonical
    sample order (maths.ties.run_sample_order), so its result does not depend on the input order.
    None means the evidence is already in canonical order.
    """

    block: int
    site_start: int
    positions: np.ndarray
    kept: np.ndarray
    likelihoods: np.ndarray
    observed: np.ndarray
    sample_ids: tuple | None = None

    @property
    def informative(self) -> bool:
        """At least one discovery site is observed in at least one sample."""
        return bool(self.kept.any() and self.observed[:, self.kept].any())


def block_evidence(evidence, block: int) -> BlockEvidence:
    """Views (no copies) of ``evidence`` (a founder-view ``ContigEvidence``) on one block."""
    start, stop = (int(value) for value in evidence.block_spans[block])
    return BlockEvidence(block=int(block), site_start=start,
                         positions=evidence.positions[start:stop],
                         kept=evidence.discovery_sites[start:stop],
                         likelihoods=evidence.likelihoods[:, start:stop],
                         observed=evidence.observed[:, start:stop],
                         sample_ids=tuple(evidence.sample_ids))


def discover_block(evidence: BlockEvidence, config: DiscoveryConfig) -> DiscoveredBlock | None:
    """Search one block's panel and release its alleles; None for an uninformative block."""
    if not evidence.informative:
        return None
    with diagnostic_block() as diagnostics:
        kept = evidence.kept
        result = search_block(np.ascontiguousarray(evidence.likelihoods[:, kept]),
                              np.ascontiguousarray(evidence.observed[:, kept]), config,
                              sample_ids=evidence.sample_ids)
        block = materialise(evidence.block, evidence.site_start, evidence.positions, kept,
                            evidence.likelihoods, evidence.observed, result, config)
        return replace(block, search=replace(block.search, **diagnostics.summary_fields()))


# Memory plan of one block task (the governor reservation): the block's evidence copies and the
# search workspace for panels of up to ESTIMATE_FOUNDERS rows (its caches are entry-bounded and
# bit-packed). Calibrated on AulStu chr3 (N 290, L 200, one thread): measured peak above the
# worker's baseline 0.73 GB at K 25 (block 2058), estimate 0.85 GB.
ESTIMATE_FOUNDERS = 32
# Historical TroMau ordinary-task incremental RSS exceeded its plan by 752 MB; the added 768 MiB
# covers that measured active-workspace shortfall by 53 MB. Worker residence is owned separately.
TASK_BASE_BYTES = 1280 << 20
# Total private worker residence after native trim reached 486 MB in the long K30 measurement. This remains
# owned between tasks and contigs; the task base above still covers Python search/hierarchy workspace.
WORKER_RESIDENT_BYTES = 512 << 20


def clustering_matrix_bytes(samples: int, config: DiscoveryConfig | None = None) -> int:
    """Additional dense clustering matrices, independently of the linear fit plan.

    Residual copy masses sum to two per sample. A passing record has mass at
    least min_soft_responsibility - MASS_BAND; round its count bound upward.
    The zero-threshold case retains the existing ESTIMATE_FOUNDERS planning
    assumption (not a hard bound on discovered K).

    Global residual clustering has one R-by-R float64 matrix: min_samples=1
    needs no reachability copy. A partner route has at most N records; its
    previous and new N-by-N submatrices may coexist during assignment. K=1
    seeding peaks at three N-by-N arrays (distance, reachability copy and the
    core-distance partition scratch); similarity has already been freed.
    Reserve both phases conservatively because allocator arenas may persist.
    This covers dense distance matrices; the existing task base/workspace still
    estimates Python hierarchy/clique and fitted-panel storage.
    """
    config = DiscoveryConfig() if config is None else config
    threshold = float(config.residual_births.min_soft_responsibility) - MASS_BAND
    rows_per_sample = (math.ceil(2.0 / threshold) if threshold > 0.0
                       else ESTIMATE_FOUNDERS)
    records = int(samples) * rows_per_sample
    n = int(samples)
    a = int(config.residual_births.landmark_budget)
    b = int(config.search.seed_landmark_budget)
    # Distance chunks + concatenate + sorted nearest ties and indexing scratch.
    residual = (5 * records * a + 4*a*a + 5*n*min(n,a) if a and records > a
                else records*records + 2*n*n)
    seeds = 5*n*b + 5*b*b if b and n > b else 3*n*n
    return 8 * (residual + seeds)


def block_task_bytes(evidence: BlockEvidence, config: DiscoveryConfig | None = None) -> int:
    """Planned peak bytes: evidence copies, canonical-order copy, fit workspace
    and clustering matrices. A=0 retains the explicit dense geometry plan."""
    samples, sites = evidence.likelihoods.shape[:2]
    kept = int(np.count_nonzero(evidence.kept))
    return (TASK_BASE_BYTES + 2 * 8 * 3 * samples * sites + 25 * samples * kept
            + search_bytes(samples, kept, ESTIMATE_FOUNDERS)
            + clustering_matrix_bytes(samples, config))


# Contigs whose evidence is prepared ahead of the block queue, on as many helper threads. A helper holds a reserved
# slice of the block pool's CPUs while it runs (BudgetedPool.helper_seat); more helpers would each need a slice
# taken from the block tasks.
PREPARE_AHEAD = 1

# Block workers return freed heap pages every this many blocks, or whenever their resident set
# reaches this size (glibc keeps arenas otherwise; per-block temporaries add up across a contig).
_TRIM_EVERY_BLOCKS = 32
_TRIM_RSS_BYTES = 1536 * 1024 * 1024
_blocks_since_trim = 0


def _discover_task(task, shared, budget):
    """Pool task: one block (worker side)."""
    global _blocks_since_trim
    evidence, config, dispatched_at = task
    with diagnostic_block(dispatched_at):
        with diagnostic_phase("idle_admission"):
            claim_threads()
        result = discover_block(evidence, config)
    _blocks_since_trim += 1
    if _blocks_since_trim >= _TRIM_EVERY_BLOCKS or resident_bytes() >= _TRIM_RSS_BYTES:
        malloc_trim()
        _blocks_since_trim = 0
    return evidence.block, result


def _queued_task(task, shared, budget):
    """Pool task of the one-queue route: (contig position, block evidence, config, dispatch time) -> (contig
    position, block, result) (worker side)."""
    position, evidence, config, dispatched_at = task
    block, result = _discover_task((evidence, config, dispatched_at), shared, budget)
    return position, block, result


class _Contig:
    """One contig's discovery bookkeeping. The dispatch thread fills it before the contig's first task is sent;
    the calling thread records results (its chunk writer) and finishes the contig."""

    def __init__(self, contig, file, empty, sizes, started):
        self.contig, self.file, self.started = contig, file, started
        self.empty = empty              # uninformative blocks: recorded as None by the calling thread
        self.sizes = sizes              # block -> planned task bytes (diagnostic)
        self.total = len(sizes)         # informative tasks
        self.received = 0
        self.writer = None
        self.done = False

    def record(self, block, result) -> None:
        if self.writer is None:
            self.writer = records.ChunkWriter(self.file)
            for empty in self.empty:
                self.writer.add(empty, None)
        if block is not None:
            self.writer.add(block, result)
            self.received += 1


def _start_contig(store, contig, config, evidence):
    """Dispatch-thread preparation of one contig: (bookkeeping or None if it is complete, its block tasks)."""
    file = store.contig(contig)
    records.bind(store, file, config)
    if records.complete(file):
        print(f"\n  Discovery of {contig}: every block recorded earlier", flush=True)
        return None, []
    if evidence is None:
        evidence = evidence_records.load(file)
    evidence = evidence.founder_view()
    planned = records.planned_blocks(file)
    if planned is None:
        file.put(records.PLAN, dict(blocks=int(evidence.n_blocks)))
    elif planned != evidence.n_blocks:
        raise RuntimeError(f"{file.label}: the discovery plan has {planned} blocks, the evidence "
                           f"{evidence.n_blocks}; use a new output root")
    recorded = records.recorded_indices(file)
    pending = [block for block in range(evidence.n_blocks) if block not in recorded]
    print(f"\n  Discovering {contig}..."
          + (f" ({len(recorded)} blocks recorded earlier)" if recorded else ""), flush=True)
    inputs = [block_evidence(evidence, block) for block in pending]
    tasks = [block_input for block_input in inputs if block_input.informative]
    empty = [block_input.block for block_input in inputs if not block_input.informative]
    sizes = {block_input.block: block_task_bytes(block_input, config) for block_input in tasks}
    return _Contig(contig, file, empty, sizes, time.monotonic()), tasks


def _finish_contig(state) -> None:
    """Calling thread: flush the contig's records, check them complete and report it."""
    state.record(None, None)             # creates the writer and records the uninformative blocks if needed
    state.writer.flush()
    state.done = True
    file, contig = state.file, state.contig
    if not records.complete(file):
        raise RuntimeError(f"{contig}: block discovery records are incomplete")
    blocks = [block for block in records.load(file) if block is not None]
    counts = [block.founders for block in blocks]
    counters = [backstop_counts(block.search) for block in blocks]
    limits = {name: sum(counts[name] for counts in counters)
              for name in backstop_counts({})}
    incomplete = sum(block.search.search_incomplete for block in blocks)
    unresolved = sum(block.search.unresolved for block in blocks)
    omission_causes = {}
    for block in blocks:
        for cause in {item["cause"] for item in block.search.omissions}:
            omission_causes[cause] = omission_causes.get(cause, 0) + 1
    print(f"    [Discovery] {contig}: {len(blocks)} blocks"
          + (f", founders/block: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}"
             if counts else "")
          + f" in {time.monotonic() - state.started:.1f}s; backstop hits {limits}"
          + f"; search incomplete {incomplete}/{len(blocks)}, unresolved {unresolved},"
          + f" omission blocks by cause {omission_causes}", flush=True)


def run(store, contigs, config: DiscoveryConfig, budget, *, prepare) -> None:
    """Discover every contig's pending blocks with one block pool and one queue across contigs (a contig's last blocks
    run beside the next contig's first, so the pool never drains between contigs); blocks are recorded as they
    complete. ``prepare(contig, budget) -> ContigEvidence | None`` makes the contig's evidence record (the evidence
    stage) and returns the evidence in memory (None: decode the record); it is called up to PREPARE_AHEAD contigs
    ahead on helper threads, each with a live share of the block pool's CPUs. Every block's result depends only on
    its own evidence."""
    contigs = list(contigs)
    if not contigs:
        return
    # Pool residence is owned between contigs; task workspaces remain individually paced.
    workers = max(1, int(budget.threads()))
    workers = memory_worker_limit([WORKER_RESIDENT_BYTES] * workers, workers, label="discovery workers")
    states = [None] * len(contigs)
    caller = held_context()

    def prepared(contig):
        """The evidence helper on a reserved slice of the block pool's CPUs (not a task contender, and not a second
        full budget)."""
        with pool.helper_seat() as share:
            return prepare(contig, share)

    def tasks():
        """The block tasks of every contig in contig order, prepared one contig ahead (pool dispatch thread)."""
        upcoming = deque(preparer.submit(prepared, contig) for contig in contigs[:PREPARE_AHEAD])
        for index, contig in enumerate(contigs):
            # The calling thread waits for this work: its reservation counts as waiting meanwhile.
            with adopted(caller), memory_governor().suspended():
                evidence = upcoming.popleft().result()
            if index + PREPARE_AHEAD < len(contigs):
                upcoming.append(preparer.submit(prepared, contigs[index + PREPARE_AHEAD]))
            states[index], blocks = _start_contig(store, contig, config, evidence)
            del evidence
            for block_input in blocks:
                yield index, block_input, config, time.perf_counter()
            blocks = block_input = None      # block views keep the whole contig's evidence alive

    # On an exception, terminate/join the pool and release its residence before waiting for the preparer.
    with ThreadPoolExecutor(PREPARE_AHEAD, thread_name_prefix="discovery-prepare") as preparer:
        with BudgetedPool(
                budget, processes=workers, label="discovery", yield_to_waiters=True) as pool:
            try:
                for index, block, result in pool.imap_unordered(
                        _queued_task, tasks(), task_bytes=lambda task: states[task[0]].sizes[task[1].block]):
                    state = states[index]
                    state.record(block, result)
                    if state.received == state.total:
                        _finish_contig(state)
                        release_memory()
            except BaseException:
                # Keep the blocks completed so far (a rerun discovers only the missing ones); unfinished contigs
                # are not finished or reported.
                for state in states:
                    if state is not None and not state.done and state.writer is not None:
                        state.writer.flush()
                raise
            for state in states:
                if state is not None and not state.done:
                    _finish_contig(state)
