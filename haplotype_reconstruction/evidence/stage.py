"""The evidence stage of one contig: reads -> site/sample QC -> read model -> likelihoods -> blocks.

``prepare_contig`` is the pure computation (no store); ``run_contig`` reads the contig from the
observation source, prepares it and records it (record ``evidence``, ``records.py``), or returns the
summary of an existing record bound to the same identity. Both are thread-safe: the driver may
prepare the next contig on a helper thread while discovery runs the current one. Results do not
depend on the CPU budget.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import time

import numpy as np
from numba import njit, prange

from ..execution import numba_threads, reserve_memory, stage_timer
from ..execution.budget import as_budget
from ..model.evidence import ContigEvidence, ContigReads
from . import records
from .config import EvidenceConfig
from .read_model import calibrate as calibration
from .read_model.fit import fixed_model
from .reports import EvidenceReport, EvidenceSummary, log_lines, summarise
from .site_quality import filter_sites

# Normalisation tolerance of the likelihood rows (max-shift, exp, divide: a few ULP from 1).
_ROW_SUM_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PreparedEvidence:
    evidence: ContigEvidence
    report: EvidenceReport


@njit(cache=True, parallel=True)
def _invalid_likelihood_cells(likelihoods, counts, tolerance):
    """Cells whose likelihoods are not finite, negative, not normalised, or not 1/3 at zero depth."""
    n, length = likelihoods.shape[:2]
    bad = np.zeros(n, np.int64)
    third = 1. / 3.
    for sample in prange(n):
        for site in range(length):
            p0, p1, p2 = likelihoods[sample, site, 0], likelihoods[sample, site, 1], likelihoods[sample, site, 2]
            if counts[sample, site, 0] + counts[sample, site, 1] == 0:
                if p0 != third or p1 != third or p2 != third:
                    bad[sample] += 1
            elif (not (np.isfinite(p0) and np.isfinite(p1) and np.isfinite(p2))
                  or p0 < 0. or p1 < 0. or p2 < 0. or abs(p0 + p1 + p2 - 1.) > tolerance):
                bad[sample] += 1
    return bad.sum()


def _block_spans(n_sites: int, width: int) -> np.ndarray:
    starts = np.arange(0, n_sites, width, dtype=np.int64)
    return np.column_stack((starts, np.minimum(starts + width, n_sites)))


def prepare_contig(reads: ContigReads, config: EvidenceConfig, budget) -> PreparedEvidence:
    """The evidence of one contig from its allele depths.

    Order (as base): site filter and contamination QC on all input SNPs; read-model calibration
    on the kept sites of every analysed sample (flagged samples included); normalised
    likelihoods; observation mask; discovery sites (all samples); 200-site blocks.
    """
    budget = as_budget(budget)
    if reads.n_sites == 0:
        raise ValueError(f"{reads.contig}: no biallelic SNPs to analyse")
    contig = reads.contig
    timings = {}
    started = time.monotonic()
    threads = budget.threads()
    with stage_timer("site_quality", contig), numba_threads(threads):
        site = filter_sites(reads.positions, reads.counts, config.site_quality,
                            config.sample_quality, threads=threads)
    timings["site_quality"] = time.monotonic() - started
    kept = reads if site.kept.all() else reads.select_sites(site.kept)
    if kept.n_sites == 0:
        raise ValueError(f"{contig}: the site filter kept no sites")
    counts, positions = kept.counts, kept.positions

    started = time.monotonic()
    threads = budget.threads()
    with stage_timer("read_calibration", contig), numba_threads(threads):
        fitted, report = calibration.calibrate(counts, positions, config.read_model, threads=threads)
    timings["read_model"] = time.monotonic() - started
    model = fitted if fitted is not None else fixed_model(config.fixed_error, kept.n_samples)

    started = time.monotonic()
    with stage_timer("likelihoods", contig), numba_threads(budget.threads()):
        likelihoods = calibration.likelihoods(counts, model, config.read_model.lookup_depth)
        invalid = _invalid_likelihood_cells(likelihoods, counts, _ROW_SUM_TOLERANCE)
        if invalid:
            raise RuntimeError(f"{contig}: {invalid} genotype-likelihood cells are not normalised "
                               "probabilities (zero depth must be exactly 1/3)")
        observed = (counts[:, :, 0] > 0) | (counts[:, :, 1] > 0)
        total_depth = counts.sum(axis=(0, 2), dtype=np.int64)
        discovery_sites = total_depth >= config.discovery_sites.threshold(kept.n_samples)
    founder_excluded = (np.zeros(kept.n_samples, bool) if site.sample_quality is None
                        else np.asarray(site.sample_quality.flagged, bool))
    timings["likelihoods"] = time.monotonic() - started
    evidence = ContigEvidence(
        contig=contig, sample_ids=reads.sample_ids, positions=positions, counts=counts,
        likelihoods=likelihoods, observed=observed, discovery_sites=discovery_sites,
        founder_excluded=founder_excluded, block_spans=_block_spans(kept.n_sites, config.block_sites))
    return PreparedEvidence(evidence, EvidenceReport(
        contig=contig, input_positions=reads.positions, dropped=dict(reads.dropped),
        site_quality=site, calibration=report, timings=timings))


def preparation_bytes(samples: int, sites: int) -> int:
    """Peak bytes of preparing one contig's evidence beyond its allele depths (the reservation of
    run_contig, execution.memory): float64 likelihoods (24 per sample and site), the observation mask,
    the site-quality depth arrays (two float64 sites x samples arrays, real data) and the depth
    totals, plus 256 MiB of tiles and model fitting."""
    return 41 * samples * sites + (256 << 20)


def run_contig(store, source, contig: str, config: EvidenceConfig, budget, *,
               keep: bool = False) -> EvidenceSummary:
    """Record the evidence of one contig unless it is recorded under the same identity.

    ``source``: an observation source (``reads(contig, budget) -> ContigReads``, ``identity() -> dict``).
    A different identity bound to the contig raises (use a new output root). With ``keep``, the
    summary also carries the in-memory evidence (``summary.prepared``; loaded from the record, with
    counts, when the contig was recorded earlier), so the next stage need not decode the record again.
    """
    file = store.contig(contig)
    store.bind_identity(file, records.STAGE, records.identity(config, source.identity()))
    with stage_timer("evidence", contig):
        if records.complete(file):
            report = records.load_report(file)
            prepared = None
            if keep:
                info = file.info(records.EVIDENCE)
                with reserve_memory(0 if info is None else info.decoded_bytes, label="evidence decode"):
                    evidence = records.load(file, counts=True)
                prepared = PreparedEvidence(evidence, report)
                sites, sample_ids = evidence.n_sites, evidence.sample_ids
            else:
                payload = file.get(records.EVIDENCE, items=("sample_ids", "positions"))
                sites, sample_ids = len(payload["positions"]), payload["sample_ids"]
            return replace(summarise(report, sites, sample_ids, resumed=True), prepared=prepared)
        started = time.monotonic()
        with stage_timer("read_input", contig):
            reads = source.reads(contig, budget)
        loaded = time.monotonic() - started
        # Admitted by the run's memory governor (the next contig's preparation runs beside discovery).
        with reserve_memory(preparation_bytes(reads.n_samples, reads.n_sites), label="evidence preparation"):
            prepared = prepare_contig(reads, config, budget)
        del reads
        started = time.monotonic()
        with stage_timer("record", contig):
            records.write(file, prepared.evidence, prepared.report)
    report = replace(prepared.report, timings=dict(prepared.report.timings, read_input=loaded,
                                                   record=time.monotonic() - started))
    summary = summarise(report, prepared.evidence.n_sites, prepared.evidence.sample_ids, resumed=False)
    for line in log_lines(summary, report):
        print(line, flush=True)
    return replace(summary, prepared=PreparedEvidence(prepared.evidence, report) if keep else None)


def run(store, source, contigs, config: EvidenceConfig, budget) -> list[EvidenceSummary]:
    """run_contig for every contig, in order."""
    return [run_contig(store, source, contig, config, budget) for contig in contigs]
