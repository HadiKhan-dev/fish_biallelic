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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time

import numpy as np

from ..evidence import records as evidence_records
from ..execution import BudgetedPool, claim_threads, malloc_trim, release_memory, resident_bytes
from . import records

from .config import DiscoveryConfig
from .residual_births import MASS_BAND
from .fixed_k.workspace import search_bytes
from .release import DiscoveredBlock, materialise
from .search import search_block


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
    kept = evidence.kept
    result = search_block(np.ascontiguousarray(evidence.likelihoods[:, kept]),
                          np.ascontiguousarray(evidence.observed[:, kept]), config,
                          sample_ids=evidence.sample_ids)
    return materialise(evidence.block, evidence.site_start, evidence.positions, kept,
                       evidence.likelihoods, evidence.observed, result, config)


# Memory plan of one block task (the governor reservation): the block's evidence copies and the
# search workspace for panels of up to ESTIMATE_FOUNDERS rows (its caches are entry-bounded and
# bit-packed). Calibrated on AulStu chr3 (N 290, L 200, one thread): measured peak above the
# worker's baseline 0.73 GB at K 25 (block 2058), estimate 0.85 GB.
ESTIMATE_FOUNDERS = 32
TASK_BASE_BYTES = 512 << 20


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
    """Planned peak additional bytes of one block's discovery in a pool worker (release 2.1: plus
    the canonical-order copy of the block's GLs and mask, and a second search workspace when the
    reversed-convention search is configured), including the dense clustering matrices."""
    samples, sites = evidence.likelihoods.shape[:2]
    kept = int(np.count_nonzero(evidence.kept))
    searches = 1
    if config is not None and config.search.reversed_convention_search != "off":
        searches = 2
    return (TASK_BASE_BYTES + 2 * 8 * 3 * samples * sites + 25 * samples * kept
            + searches * (search_bytes(samples, kept, ESTIMATE_FOUNDERS)
                          + clustering_matrix_bytes(samples, config)))


# Block workers return freed heap pages every this many blocks, or whenever their resident set
# reaches this size (glibc keeps arenas otherwise; per-block temporaries add up across a contig).
_TRIM_EVERY_BLOCKS = 32
_TRIM_RSS_BYTES = 1536 * 1024 * 1024
_blocks_since_trim = 0


def _discover_task(task, shared, budget):
    """Pool task: one block (worker side)."""
    global _blocks_since_trim
    evidence, config = task
    claim_threads()
    result = discover_block(evidence, config)
    _blocks_since_trim += 1
    if _blocks_since_trim >= _TRIM_EVERY_BLOCKS or resident_bytes() >= _TRIM_RSS_BYTES:
        malloc_trim()
        _blocks_since_trim = 0
    return evidence.block, result


def _discover_contig(store, contig, config, pool, evidence=None) -> None:
    """Discover one contig's pending blocks; ``evidence``: its ContigEvidence when the caller has
    it in memory (else decoded from the record)."""
    file = store.contig(contig)
    records.bind(store, file, config)
    if records.complete(file):
        print(f"\n  Discovery of {contig}: every block recorded earlier", flush=True)
        return
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
    started = time.monotonic()
    writer = records.ChunkWriter(file)
    tasks = []
    try:
        for block in pending:
            block_input = block_evidence(evidence, block)
            if block_input.informative:
                tasks.append((block_input, config))
            else:
                writer.add(block, None)
        sizes = [block_task_bytes(block_input, config) for block_input, _config in tasks]
        for index, result in pool.imap_unordered(_discover_task, tasks, task_bytes=sizes):
            writer.add(index, result)
    finally:
        writer.flush()
    if not records.complete(file):
        raise RuntimeError(f"{contig}: block discovery records are incomplete")
    blocks = [block for block in records.load(file) if block is not None]
    counts = [block.founders for block in blocks]
    limits = {name: sum(name in block.search.limits for block in blocks)
              for name in ("expansion_backstop", "exact_score_backstop")}
    incomplete = sum(block.search.search_incomplete for block in blocks)
    unresolved = sum(block.search.unresolved for block in blocks)
    omission_causes = {}
    for block in blocks:
        for cause in {item["cause"] for item in block.search.omissions}:
            omission_causes[cause] = omission_causes.get(cause, 0) + 1
    print(f"    [Discovery] {contig}: {len(blocks)} blocks"
          + (f", founders/block: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}"
             if counts else "")
          + f" in {time.monotonic() - started:.1f}s; backstop hits {limits}"
          + f"; search incomplete {incomplete}/{len(blocks)}, unresolved {unresolved},"
          + f" omission blocks by cause {omission_causes}", flush=True)


def run(store, contigs, config: DiscoveryConfig, budget, *, prepare) -> None:
    """Discover every contig's pending blocks with one block pool; blocks are recorded as they
    complete. ``prepare(contig) -> ContigEvidence | None`` makes the contig's evidence record (the
    evidence stage) and returns the evidence in memory (None: decode the record); it is called one
    contig ahead on a helper thread so the pool never idles between contigs."""
    contigs = list(contigs)
    if not contigs:
        return
    with BudgetedPool(budget, processes=max(1, int(budget.threads())),
                      label="discovery") as pool, \
            ThreadPoolExecutor(1, thread_name_prefix="discovery-prepare") as preparer:
        upcoming = preparer.submit(prepare, contigs[0])
        for index, contig in enumerate(contigs):
            evidence = upcoming.result()
            if index + 1 < len(contigs):
                upcoming = preparer.submit(prepare, contigs[index + 1])
            _discover_contig(store, contig, config, pool, evidence)
            del evidence
            release_memory()
