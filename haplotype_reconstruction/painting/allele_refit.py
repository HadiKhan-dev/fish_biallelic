"""Conditional founder-allele refit from the fixed chromosome painting.

Assembly releases chromosome rows as concatenations of block-local discovery
rows whose alleles are never re-called. Where block-level model selection
merged two founders that differ at a few sites, the row of the unrepresented
founder borrows the other founder's alleles there, for all of its carriers.

Given the painting, this stage performs one conditional M-step: the allele of
row ``k`` at site ``j`` maximises the robust genotype log-likelihood of the
samples painted as carrying ``k`` at ``j``, with each sample's other painted
homologue held at its current allele. Each site is updated by coordinate
ascent over single-row flips and joint two-row flips (a fixed number of
sweeps); a move is accepted only when its gain reaches ``minimum_gain_nats``.
The two-row moves correct allele swaps between two rows: a swap keeps the
genotype of every sample painted with that pair, so a single flip loses
likelihood on those samples. On the reviewed simulations the result equals an
exact per-site search except at a few dozen of 14 million cells. Unknown alleles stay unknown; row
paths, founder counts and the painting itself are unchanged. Samples that the
sample-quality QC keeps out of a contig's founder construction
(core/sample_quality.py) are left out of the refit there too. The painter and
the pedigree keep consuming the frozen assembly snapshot, so nothing feeds back
into carrier-state evidence. Family phase decodes ``inference_alleles``;
evaluation and export read ``released_alleles``. A repaint/refit iteration
(EM) could reuse :func:`refit_component_alleles` unchanged; it is not run here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from numba import njit, prange

from haplotype_reconstruction import PACKAGE_ROOT
from haplotype_reconstruction.core.run_record import timed_stage


ALLELE_REFIT_STAGE = "allele_refit"
SCHEMA = "painted-carrier-founder-allele-refit-v1"
# Sites per parallel task; each task owns one founder-pair table.
_SITE_TILE = 1024


@dataclass(frozen=True)
class AlleleRefitConfig:
    """Scientific settings of the refit; the gain threshold acts as a prior.

    ``minimum_gain_nats``: a called allele changes only if the painted carriers'
    robust log-likelihood improves by at least this much (5 nats: the assembled
    allele keeps prior odds of about 150 to 1). ``robustness_epsilon`` is the
    painter's uniform genotype-error mixture. ``pair_moves=False`` restricts the
    ascent to single-row flips (the reviewed reference computation; it cannot
    correct two-row allele swaps).
    """

    minimum_gain_nats: float = 5.0
    coordinate_sweeps: int = 3
    robustness_epsilon: float = 0.01
    pair_moves: bool = True

    def __post_init__(self):
        if not math.isfinite(self.minimum_gain_nats) or self.minimum_gain_nats < 0.0:
            raise ValueError("minimum_gain_nats must be finite and non-negative")
        if int(self.coordinate_sweeps) != self.coordinate_sweeps or self.coordinate_sweeps < 1:
            raise ValueError("coordinate_sweeps must be a positive integer")
        if not 0.0 <= self.robustness_epsilon <= 1.0:
            raise ValueError("robustness_epsilon must lie in [0, 1]")


@dataclass(frozen=True)
class RefitComponentPanel:
    """One painted component's founder alleles after the refit.

    Rows follow ``founder_keys`` (the painting's component-local order) and
    columns follow ``positions``. ``released_alleles`` refits the published
    assembly calls (the frozen snapshot plus completion fills), which act as
    the partner alleles. ``inference_alleles`` is the same matrix restricted to
    the frozen snapshot's called cells (-1 elsewhere), the only alleles family
    phase decodes. Released cells that changed are listed with their accepted
    gain in nats.
    """

    component_index: int
    founder_keys: tuple
    positions: np.ndarray
    inference_alleles: np.ndarray
    released_alleles: np.ndarray
    changed_rows: np.ndarray
    changed_sites: np.ndarray
    changed_gain_nats: np.ndarray


@dataclass(frozen=True)
class AlleleRefitPanel:
    """Per-chromosome refit product, bound to one painting checkpoint."""

    schema: str
    identity: dict
    contig: str
    sample_ids: tuple
    components: tuple


def refit_code_identity():
    name = "painting/allele_refit.py"
    return {name: hashlib.sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest()}


def _record_sha256(record):
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def refit_identity(config, painting_checkpoint, excluded_samples=()):
    """Scientific identity: model, settings, code and the bound painting.

    ``excluded_samples`` (IDs whose reads the sample-quality QC keeps out of
    this contig's founder evidence) is recorded when nonempty.
    """
    return {
        "schema": SCHEMA,
        "config": asdict(config),
        "code": refit_code_identity(),
        "painting_release_sha256": _record_sha256(painting_checkpoint.release_identity.record()),
        "painting_product_sha256": _record_sha256(
            painting_checkpoint.painting_product_identity.record()),
        **({"founder_evidence_excluded": tuple(excluded_samples)} if excluded_samples else {}),
    }


@njit(cache=True)
def _pair_flip_gain(table, alleles, a, b):
    """Log-likelihood change from flipping called rows a < b together."""
    xa, xb = alleles[a], alleles[b]
    ya, yb = 1 - xa, 1 - xb
    old = table[a, a, 2 * xa] + table[b, b, 2 * xb] + table[a, b, xa + xb]
    new = table[a, a, 2 * ya] + table[b, b, 2 * yb] + table[a, b, ya + yb]
    for partner in range(len(alleles)):
        other = alleles[partner]
        if partner == a or partner == b or other < 0:
            continue
        old += table[min(a, partner), max(a, partner), xa + other]
        new += table[min(a, partner), max(a, partner), ya + other]
        old += table[min(b, partner), max(b, partner), xb + other]
        new += table[min(b, partner), max(b, partner), yb + other]
    return new - old


@njit(cache=True, parallel=True, fastmath=False)
def _refit_kernel(calls, first, second, evidence, observed, site_index,
                  robustness_epsilon, minimum_gain, sweeps, pair_moves):
    """Coordinate ascent per site on a founder-pair log-likelihood table.

    ``table[a, b, g]`` (a <= b) sums log((1-eps) * P(reads | dosage g) + eps/3)
    over observed samples painted with the unordered pair {a, b}; a row's
    objective then needs only its partners' current alleles. Pairs whose
    partner allele is unknown contribute nothing, as do unpainted tracks.
    Each sweep tries every single-row flip, then (``pair_moves``) every joint
    flip of two rows: allele 1 carried by the wrong one of two rows keeps the
    painted pair's genotype, so a single flip loses on those samples.
    """
    founders, sites = calls.shape
    samples = first.shape[0]
    refit = calls.copy()
    gain = np.zeros((founders, sites), dtype=np.float64)
    tiles = (sites + _SITE_TILE - 1) // _SITE_TILE
    for tile in prange(tiles):
        table = np.zeros((founders, founders, 3), dtype=np.float64)
        for site in range(tile * _SITE_TILE, min(sites, (tile + 1) * _SITE_TILE)):
            table[:] = 0.0
            column = site_index[site]
            carried = False
            for sample in range(samples):
                a, b = first[sample, site], second[sample, site]
                if a < 0 or b < 0 or not observed[sample, column]:
                    continue
                if a > b:
                    a, b = b, a
                x0 = np.float64(evidence[sample, column, 0])
                x1 = np.float64(evidence[sample, column, 1])
                x2 = np.float64(evidence[sample, column, 2])
                total = (x0 + x1) + x2
                if total > 0.0:
                    x0, x1, x2 = x0 / total, x1 / total, x2 / total
                else:
                    x0 = x1 = x2 = 1.0 / 3.0
                table[a, b, 0] += math.log((1.0 - robustness_epsilon) * x0 + robustness_epsilon / 3.0)
                table[a, b, 1] += math.log((1.0 - robustness_epsilon) * x1 + robustness_epsilon / 3.0)
                table[a, b, 2] += math.log((1.0 - robustness_epsilon) * x2 + robustness_epsilon / 3.0)
                carried = True
            if not carried:
                continue
            for _ in range(sweeps):
                for row in range(founders):
                    current = refit[row, site]
                    if current < 0:
                        continue
                    ll0 = table[row, row, 0]
                    ll1 = table[row, row, 2]
                    for partner in range(founders):
                        other = refit[partner, site]
                        if partner == row or other < 0:
                            continue
                        lo, hi = min(row, partner), max(row, partner)
                        ll0 += table[lo, hi, other]
                        ll1 += table[lo, hi, other + 1]
                    delta = ll1 - ll0 if current == 0 else ll0 - ll1
                    if delta > 0.0 and delta >= minimum_gain:
                        refit[row, site] = 1 - current
                        gain[row, site] = delta
                if not pair_moves:
                    continue
                for a in range(founders):
                    for b in range(a + 1, founders):
                        if refit[a, site] < 0 or refit[b, site] < 0:
                            continue
                        delta = _pair_flip_gain(table, refit[:, site], a, b)
                        if delta > 0.0 and delta >= minimum_gain:
                            refit[a, site] = 1 - refit[a, site]
                            refit[b, site] = 1 - refit[b, site]
                            gain[a, site] = delta
                            gain[b, site] = delta
    return refit, gain


def refit_component_alleles(calls, first, second, evidence, observed, site_index, config):
    """Refit a K-by-L allele matrix (-1 unknown, never filled) given painted pairs.

    ``first``/``second`` are (samples, L) founder rows of the two painted
    tracks (-1 unknown); ``site_index`` maps panel columns to the evidence
    site axis. Returns the refit matrix and the accepted gain per cell (zero
    where unchanged; a cell flipped twice keeps its last gain).
    """
    return _refit_kernel(
        np.ascontiguousarray(calls, dtype=np.int8), np.ascontiguousarray(first, dtype=np.int16),
        np.ascontiguousarray(second, dtype=np.int16), evidence,
        np.ascontiguousarray(observed, dtype=np.bool_),
        np.ascontiguousarray(site_index, dtype=np.int64),
        float(config.robustness_epsilon), float(config.minimum_gain_nats),
        int(config.coordinate_sweeps), bool(config.pair_moves))


def site_founder_pairs(painting, positions, sample_count):
    """Per-site painted founder rows; chunks are half-open [start, end) in bp."""
    first = np.full((sample_count, len(positions)), -1, dtype=np.int16)
    second = np.full((sample_count, len(positions)), -1, dtype=np.int16)
    for sample in painting.samples:
        for chunk in sample.chunks:
            lo, hi = np.searchsorted(positions, (chunk.start, chunk.end))
            first[sample.sample_index, lo:hi] = chunk.hap1
            second[sample.sample_index, lo:hi] = chunk.hap2
    return first, second


def refit_painting_alleles(painting_checkpoint, genotype_likelihoods, positions, observed,
                           *, contig, config=AlleleRefitConfig(), founder_evidence_excluded=None):
    """Refit every painted component of one chromosome; the painting is unchanged.

    ``founder_evidence_excluded`` (the contig's sample-quality flags, as the
    release-identity check returns them) keeps flagged samples' reads out of
    the re-estimation, as in founder construction: their cells enter through
    ``core.sample_quality.founder_evidence``, unobserved, so no row is scored
    on them.
    """
    painting_checkpoint = painting_checkpoints.validate_painting_checkpoint(painting_checkpoint)
    positions = np.asarray(positions, dtype=np.int64)
    evidence = np.asarray(genotype_likelihoods)
    if evidence.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        evidence = evidence.astype(np.float64)
    evidence = np.ascontiguousarray(evidence)
    observed = np.ascontiguousarray(observed, dtype=np.bool_)
    samples = len(painting_checkpoint.sample_ids)
    if evidence.shape != (samples, len(positions), 3) or observed.shape != evidence.shape[:2]:
        raise ValueError("genotype evidence does not match the painting samples and sites")
    evidence, observed = core_sample_quality.founder_evidence(
        evidence, observed, founder_evidence_excluded)
    excluded = () if founder_evidence_excluded is None else tuple(
        painting_checkpoint.sample_ids[index]
        for index in np.flatnonzero(founder_evidence_excluded))
    blocks = core_runtime.validate_phase_component_manifest(painting_checkpoint.component_manifest)
    components = []
    for index, (block, component) in enumerate(zip(blocks, painting_checkpoint.painting_bundle.components)):
        block_positions = np.asarray(block.positions, dtype=np.int64)
        site_index = np.searchsorted(positions, block_positions)
        if (np.any(site_index >= len(positions))
                or not np.array_equal(positions[site_index], block_positions)):
            raise ValueError(f"{contig}: component {index} sites are absent from the evidence")
        frozen = np.asarray(block.missing_aware_inference_discrete_haps, dtype=np.int8)
        published = np.asarray(block.discrete_haps, dtype=np.int8)
        first, second = site_founder_pairs(component.painting, block_positions, samples)
        core_chromosome_parallel.current_threads()
        refit, gain = refit_component_alleles(
            published, first, second, evidence, observed, site_index, config)
        del first, second
        rows, sites = np.nonzero(refit != published)
        components.append(RefitComponentPanel(
            component_index=index, founder_keys=tuple(component.founder_keys),
            positions=block_positions,
            inference_alleles=np.where(frozen >= 0, refit, -1).astype(np.int8),
            released_alleles=refit,
            changed_rows=rows.astype(np.int32), changed_sites=sites.astype(np.int64),
            changed_gain_nats=gain[rows, sites]))
    return AlleleRefitPanel(SCHEMA, refit_identity(config, painting_checkpoint, excluded), str(contig),
                            tuple(painting_checkpoint.sample_ids), tuple(components))


def inference_alleles(refit_panel, painting_checkpoint):
    """Return per-component refit alleles after checking the painting binding."""
    if not isinstance(refit_panel, AlleleRefitPanel) or refit_panel.schema != SCHEMA:
        raise TypeError("allele refit requires a typed refit panel")
    if tuple(refit_panel.sample_ids) != tuple(painting_checkpoint.sample_ids):
        raise ValueError("allele refit and painting sample order differ")
    expected = _record_sha256(painting_checkpoint.release_identity.record()), _record_sha256(
        painting_checkpoint.painting_product_identity.record())
    if (refit_panel.identity["painting_release_sha256"],
            refit_panel.identity["painting_product_sha256"]) != expected:
        raise ValueError("allele refit was computed from a different painting")
    blocks = core_runtime.validate_phase_component_manifest(painting_checkpoint.component_manifest)
    if len(blocks) != len(refit_panel.components):
        raise ValueError("allele refit and painting component counts differ")
    result = []
    for block, component in zip(blocks, refit_panel.components):
        frozen = np.asarray(block.missing_aware_inference_discrete_haps)
        alleles = np.asarray(component.inference_alleles)
        # The refit changes called alleles only: its unknown cells are the snapshot's.
        if (alleles.shape != frozen.shape or not np.array_equal(alleles < 0, frozen < 0)
                or not np.array_equal(component.positions, block.positions)):
            raise ValueError("allele refit rows/sites differ from the painted component")
        result.append(alleles)
    return tuple(result)


def _summary(panel, elapsed):
    called = sum(int(np.sum(c.released_alleles >= 0)) for c in panel.components)
    gains = np.concatenate([c.changed_gain_nats for c in panel.components])
    return {
        "contig": panel.contig, "components": len(panel.components),
        "founder_rows": sum(len(c.founder_keys) for c in panel.components),
        "sites": sum(len(c.positions) for c in panel.components),
        "called_alleles": called, "changed_alleles": int(len(gains)),
        "changed_inference_alleles": sum(
            int(np.sum(c.inference_alleles[c.changed_rows, c.changed_sites] >= 0))
            for c in panel.components),
        "changed_sites": sum(len(np.unique(c.changed_sites)) for c in panel.components),
        "median_changed_gain_nats": float(np.median(gains)) if len(gains) else None,
        "minimum_gain_nats": panel.identity["config"]["minimum_gain_nats"],
        "elapsed_seconds": elapsed,
    }


def _refit_contig(contig, state, phase, common):
    """Chromosome-worker handler: load, refit and checkpoint one chromosome."""
    from haplotype_reconstruction.pedigree import components as pedigree_components

    started = time.perf_counter()
    file = core_runtime.CheckpointStore(common["checkpoint_root"]).contig(contig)
    if file.has(ALLELE_REFIT_STAGE):
        panel = file.get(ALLELE_REFIT_STAGE, nthreads=core_chromosome_parallel.current_threads())
        if panel.schema != SCHEMA or tuple(panel.sample_ids) != common["names"]:
            raise ValueError(f"{contig}: allele refit checkpoint differs from this run")
    else:
        painting = file.get(common["painting"], nthreads=core_chromosome_parallel.current_threads())
        if tuple(painting.sample_ids) != common["names"]:
            raise ValueError(f"{contig}: painting sample order differs from this run")
        gl, pos, observed = core_products.load_evidence(
            file, nthreads=core_chromosome_parallel.current_threads())
        flags, digests = core_products.evidence_checks(file)
        excluded = pedigree_components._validate_release_array_identity(
            painting, np.asarray(gl), np.asarray(pos), np.asarray(observed), flags, digests)
        panel = refit_painting_alleles(painting, gl, pos, observed, contig=contig,
                                       config=common["config"], founder_evidence_excluded=excluded)
        del painting, gl, pos, observed
        gc.collect()
        file.put(ALLELE_REFIT_STAGE, panel, nthreads=core_chromosome_parallel.current_threads())
    summary = _summary(panel, time.perf_counter() - started)
    excluded_ids = panel.identity.get("founder_evidence_excluded", ())
    print(f"[allele refit {contig}] {summary['changed_alleles']} of {summary['called_alleles']} "
          f"called alleles changed at {summary['changed_sites']} sites"
          + (f"; sample-quality QC excludes the reads of {', '.join(excluded_ids)}"
             if excluded_ids else ""), flush=True)
    del panel
    gc.collect()
    core_parallel.malloc_trim()
    return None, summary


def _refit_resources(store, contigs, workers, painting):
    """Decoded evidence (read in place) plus the painting, pair arrays and 2 GiB/process."""
    estimates = []
    for contig in contigs:
        file = store.contig(contig)
        evidence = file.info(core_products.EVIDENCE).decoded_bytes
        estimates.append(evidence * 5 // 4 + 2 * file.info(painting).decoded_bytes + (2 << 30))
    return estimates, core_chromosome_parallel.memory_worker_limit(
        estimates, workers, core_runtime.available_memory_bytes(), label="ALLELE REFIT")


@timed_stage(ALLELE_REFIT_STAGE)
def run_allele_refit(store, contigs, sample_ids, *, output_dir, n_workers=None, config=None):
    """Checkpoint one refit panel per chromosome; family phase consumes them."""
    from haplotype_reconstruction.workflows.reconstruction import PAINTING_STAGE

    config = AlleleRefitConfig() if config is None else config
    workers = core_runtime.available_cpu_count() if n_workers is None else int(n_workers)
    if not 1 <= workers <= core_runtime.available_cpu_count():
        raise ValueError("allele refit CPU budget exceeds allocation")
    names = tuple(map(str, sample_ids))
    contigs = tuple(map(str, contigs))
    identity = {"schema": SCHEMA, "config": asdict(config), "code": refit_code_identity(),
                "sample_ids": names, "contigs": contigs,
                "source_records": core_products.record_identities(
                    store, (PAINTING_STAGE, core_products.EVIDENCE), contigs)}
    store.bind_identity(store.run, ALLELE_REFIT_STAGE, identity)
    weights, process_limit = _refit_resources(store, contigs, workers, PAINTING_STAGE)
    common = dict(checkpoint_root=store.root, names=names, config=config, painting=PAINTING_STAGE)
    with core_chromosome_parallel.ChromosomeExecutor(
            contigs, _refit_contig, n_workers=workers, weights=weights,
            max_workers=process_limit, label="ALLELE REFIT") as executor:
        summaries = executor.run("refit", common)
    core_runtime.require_records(store, ALLELE_REFIT_STAGE, contigs)
    output = Path(output_dir) / ALLELE_REFIT_STAGE
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / ".summary.csv.tmp"
    pd.DataFrame(summaries).to_csv(temporary, index=False)
    temporary.replace(output / "summary.csv")
    store.mark_stage_complete(ALLELE_REFIT_STAGE)
    return summaries


import haplotype_reconstruction.core.chromosome_parallel as core_chromosome_parallel
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.core.products as core_products
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.core.sample_quality as core_sample_quality
import haplotype_reconstruction.painting.checkpoints as painting_checkpoints
