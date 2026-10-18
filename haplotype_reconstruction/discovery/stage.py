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
import time

import numpy as np

from ..evidence import records as evidence_records
from ..execution import BudgetedPool, claim_threads, malloc_trim, release_memory, resident_bytes
from . import records

from .config import DiscoveryConfig
from .fixed_k.workspace import search_bytes
from .release import DiscoveredBlock, materialise
from .search import search_block


@dataclass(frozen=True)
class BlockEvidence:
    """The founder-view evidence of one discovery block.

    block, site_start: block index and its first contig site; positions: int64[L];
    kept: bool[L] discovery sites; likelihoods: float64[N, L, 3] normalised GLs (zero depth
    exactly 1/3); observed: bool[N, L] positive depth (False for excluded samples).
    """

    block: int
    site_start: int
    positions: np.ndarray
    kept: np.ndarray
    likelihoods: np.ndarray
    observed: np.ndarray

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
                         observed=evidence.observed[:, start:stop])


def discover_block(evidence: BlockEvidence, config: DiscoveryConfig) -> DiscoveredBlock | None:
    """Search one block's panel and release its alleles; None for an uninformative block."""
    if not evidence.informative:
        return None
    kept = evidence.kept
    result = search_block(np.ascontiguousarray(evidence.likelihoods[:, kept]),
                          np.ascontiguousarray(evidence.observed[:, kept]), config)
    return materialise(evidence.block, evidence.site_start, evidence.positions, kept,
                       evidence.likelihoods, evidence.observed, result, config)


# Memory plan of one block task (the governor reservation): the block's evidence copies and the
# search workspace for panels of up to ESTIMATE_FOUNDERS rows (its caches are entry-bounded and
# bit-packed). Calibrated on AulStu chr3 (N 290, L 200, one thread): measured peak above the
# worker's baseline 0.73 GB at K 25 (block 2058), estimate 0.85 GB.
ESTIMATE_FOUNDERS = 32
TASK_BASE_BYTES = 512 << 20


def block_task_bytes(evidence: BlockEvidence) -> int:
    """Planned peak additional bytes of one block's discovery in a pool worker."""
    samples, sites = evidence.likelihoods.shape[:2]
    kept = int(np.count_nonzero(evidence.kept))
    return (TASK_BASE_BYTES + 2 * 8 * 3 * samples * sites
            + search_bytes(samples, kept, ESTIMATE_FOUNDERS))


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
        sizes = [block_task_bytes(block_input) for block_input, _config in tasks]
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
    print(f"    [Discovery] {contig}: {len(blocks)} blocks"
          + (f", founders/block: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}"
             if counts else "")
          + f" in {time.monotonic() - started:.1f}s; backstop hits {limits}", flush=True)


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
