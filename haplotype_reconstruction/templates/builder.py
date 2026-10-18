"""`haplotypes templates`: founder templates for simulations from a real cross's VCF (the naive builder).

Per chromosome: every biallelic SNP's allele depths (evidence.vcf), fixed-error genotype likelihoods without
site QC or read calibration (the evidence stage with those off), founder haplotypes discovered in overlapping
physical windows (WINDOW_BP every WINDOW_SHIFT_BP from position 0 to the contig length) with the discovery stage's
block search, and the naive linker (templates.linker) chaining them into up to TEMPLATE_HAPLOTYPES
chromosome-length haplotypes. Output OUTPUT/<contig>.npz: positions int64 [L], allele_probabilities float64
[H, L, 2] = (P(REF), P(ALT)) per haplotype and site, the format [simulation] templates reads.

Templates only generate simulated founders: they are not inferred chromosomes, and the pipeline never reads them.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time

import numpy as np

WINDOW_BP = 100_000
WINDOW_SHIFT_BP = 50_000
TEMPLATE_HAPLOTYPES = 6
# Planned peak of discovering one window in a pool worker: its evidence arrays (the worker's copy) times
# WINDOW_EVIDENCE_FACTOR, plus WINDOW_ALLOWANCE_BYTES for the block search's working memory, which grows with the
# founder count searched rather than with the inputs. Measured on AcTm chr8 (N 116, 283-2,111 sites, K 2-13):
# 177-1,178 MiB per window (mean 552 MiB) in a warm process; a worker's first window adds its kernel loading.
WINDOW_EVIDENCE_FACTOR = 8
WINDOW_ALLOWANCE_BYTES = 1280 << 20
# Bytes per sample and site of a contig's fixed-error evidence, held by the builder while the contig's windows are
# discovered (float64 likelihoods [N, L, 3], masks, slack; the evidence step measured 15 B on AcTm chr8).
CONTIG_CELL_BYTES = 32
# Contig length assumed when the VCF header declares none.
UNDECLARED_LENGTH_BP = 300_000_000


def fixed_error_evidence(reads, budget):
    """ContigEvidence of every input SNP: fixed-error (binomial) likelihoods, no site/sample QC."""
    from ..evidence import prepare_contig
    from ..evidence.config import EvidenceConfig
    config = EvidenceConfig()
    config = replace(config, site_quality=replace(config.site_quality, enabled=False),
                     sample_quality=replace(config.sample_quality, enabled=False),
                     read_model=replace(config.read_model, enabled=False))
    return prepare_contig(reads, config, budget).evidence


def window_ranges(positions: np.ndarray, length_bp: int) -> list[tuple[int, int]]:
    """[first, stop) site ranges of the windows [start, start + WINDOW_BP) bp, start = 0, WINDOW_SHIFT_BP, ...
    below ``length_bp`` (empty windows included)."""
    starts = np.arange(0, length_bp, WINDOW_SHIFT_BP, dtype=np.int64)
    first = np.searchsorted(positions, starts, side="left")
    stop = np.searchsorted(positions, starts + WINDOW_BP, side="left")
    return list(zip(first.tolist(), stop.tolist()))


def _discover_window(task, shared, budget):
    """Pool task: discover one window's founder rows (None when it is uninformative)."""
    from ..discovery.stage import discover_block
    from ..execution import claim_threads
    index, block_evidence, config = task
    claim_threads()
    return index, discover_block(block_evidence, config)


def window_bytes(window) -> int:
    """Planned peak bytes of one window's discovery task (see WINDOW_EVIDENCE_FACTOR)."""
    arrays = (window.positions, window.kept, window.likelihoods, window.observed)
    return WINDOW_EVIDENCE_FACTOR * sum(int(np.asarray(array).nbytes) for array in arrays) + WINDOW_ALLOWANCE_BYTES


def discover_windows(evidence, ranges, budget) -> list:
    """The DiscoveredBlock (or None) of every window, in window order; each window task holds its planned peak
    of the run's memory while it runs (window_bytes)."""
    from ..discovery.config import DiscoveryConfig
    from ..discovery.stage import BlockEvidence
    from ..execution import BudgetedPool
    config = DiscoveryConfig()
    tasks = []
    for index, (first, stop) in enumerate(ranges):
        if stop <= first:
            continue
        window = BlockEvidence(block=index, site_start=first, positions=evidence.positions[first:stop],
                               kept=evidence.discovery_sites[first:stop],
                               likelihoods=evidence.likelihoods[:, first:stop],
                               observed=evidence.observed[:, first:stop])
        if window.informative:
            tasks.append((index, window, config))
    results = [None] * len(ranges)
    with BudgetedPool(budget, processes=max(1, min(len(tasks), int(budget.threads()))),
                      label="template windows") as pool:
        for index, block in pool.imap_unordered(_discover_window, tasks,
                                                task_bytes=[window_bytes(task[1]) for task in tasks]):
            results[index] = block
    return results


def window_haplotypes(block):
    """A discovered window's rows as linker input: (P(REF), P(ALT)) per site, 0.5 / 0.5 where uncalled."""
    from .linker import WindowHaplotypes
    alt = np.where(block.calls >= 0, block.alt_probability, 0.5)
    return WindowHaplotypes(positions=np.asarray(block.positions, np.int64), kept=np.asarray(block.kept, bool),
                            haplotypes=np.stack([1.0 - alt, alt], axis=-1))


def contig_templates(reads, length_bp: int, budget, *, count: int = TEMPLATE_HAPLOTYPES):
    """(positions int64 [L], allele_probabilities float64 [H, L, 2]) of one chromosome (module docstring)."""
    from .linker import link_windows
    started = time.monotonic()
    evidence = fixed_error_evidence(reads, budget)
    ranges = window_ranges(evidence.positions, length_bp)
    blocks = discover_windows(evidence, ranges, budget)
    windows = [window_haplotypes(block) for block in blocks if block is not None]
    print(f"  [Discovery] {reads.contig}: {len(windows)} of {len(ranges)} windows informative in "
          f"{time.monotonic() - started:.1f} s", flush=True)
    started = time.monotonic()
    positions, haplotypes = link_windows(windows, count)
    print(f"  [Template construction] {reads.contig}: chained {len(haplotypes)} haplotypes over "
          f"{len(positions)} sites in {time.monotonic() - started:.1f} s", flush=True)
    if not haplotypes:
        return positions, np.zeros((0, len(positions), 2))
    return positions, np.stack(haplotypes)


def build_templates(vcf, output, *, contigs, cores=None) -> Path:
    """Write OUTPUT/<contig>.npz for every contig of ``vcf``; OUTPUT must be new or empty."""
    from .. import execution
    from ..evidence import vcf as vcf_reader
    available = execution.affinity_cpus()
    cores = available if cores is None else int(cores)
    if not 1 <= cores <= available:
        raise ValueError("template cores must fit the current CPU affinity")
    contigs = tuple(contigs or ())
    if not contigs or len(set(contigs)) != len(contigs):
        raise ValueError("templates need the unique contigs to model (--contigs)")
    header = vcf_reader.inspect(vcf)
    missing = [contig for contig in contigs if contig not in header.contigs]
    if missing:
        raise ValueError(f"{vcf}: contigs not in the header: {', '.join(missing)}")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("template output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    budget = execution.CpuBudget.fixed(cores)
    for contig in contigs:
        started = time.monotonic()
        reads = vcf_reader.read_contig(vcf, contig, budget)
        print(f"  [Loader] {contig}: {reads.n_sites} biallelic SNPs, {reads.n_samples} samples", flush=True)
        length = header.lengths.get(contig) or UNDECLARED_LENGTH_BP
        with execution.reserve_memory(CONTIG_CELL_BYTES * reads.n_samples * reads.n_sites, label="template contig"):
            positions, probabilities = contig_templates(reads, length, budget)
        del reads
        if len(probabilities) == 0:
            print(f"  [Template construction] {contig}: fewer than two informative windows; no template",
                  flush=True)
            continue
        temporary = output / f".{contig}.npz.tmp"
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, positions=positions, allele_probabilities=probabilities)
        temporary.replace(output / f"{contig}.npz")
        print(f"  {contig}: {output / (contig + '.npz')} in {time.monotonic() - started:.1f} s", flush=True)
        execution.release_memory()
    return output


def templates_command(args) -> int:
    """`haplotypes templates --vcf FILE --output DIR --contigs C ... [--cores N]`."""
    from .. import execution
    from ..execution.kernel_cache import persistent_kernel_cache
    available = execution.affinity_cpus()
    cores = available if args.cores is None else int(args.cores)
    if not 1 <= cores <= available:
        raise SystemExit(f"templates: --cores must be between 1 and the CPU affinity ({available})")
    if not args.contigs:
        raise SystemExit("templates: give --contigs (the chromosomes to model)")
    execution.start_run(cores, preload=(__name__, "haplotype_reconstruction.discovery.stage"))
    with persistent_kernel_cache(cores):
        build_templates(args.vcf, args.output, contigs=args.contigs, cores=cores)
    for line in execution.memory_governor().summary_lines():
        print(line, flush=True)
    return 0
