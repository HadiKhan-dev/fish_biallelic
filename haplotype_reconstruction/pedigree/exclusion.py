"""Full-marker Mendelian exclusion, separate from pedigree resampling scores.

At each site A is the normalized likelihood mixture over the two incompatible
dyad genotypes, C the maximum over seven compatible genotypes, and U the
unrestricted maximum. D=(1-delta)*C+delta*U bounds every compatible read model
with at most delta arbitrary genotype replacement. A/D has null expectation
at most one under calibrated read likelihoods. Products of convex bets retain
that bound conditional on fixed genotypes and independent read errors; SNP
inheritance itself need not be independent. Average fixed windows/bets/contigs,
never an unpenalized maximum. Mapping/ascertainment errors remain limitations.

This evidence is symmetric. Direction-aware count release lives in release.py
and is additionally conditional on the already-supported family directions.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

from ..core import runtime
from ..core import chromosome_parallel as scheduling
from . import cache, eligibility, exclusion_patterns, release


MODEL = "full-marker-mendelian-directed-v1"
STAGE = "pedigree_exclusion"
BET_SIZES = np.asarray([1/1024, 1/256, 1/64, 1/16, 1/4, 1.0])
WINDOW_WIDTHS = (1_000_000, 5_000_000, 20_000_000)
# Reassess the shared thread budget between independent candidate batches.
_CANDIDATE_BATCH_SIZE = 256


def physical_windows(positions):
    """Pre-specified half-overlapping windows, plus the whole chromosome."""
    spans = [(0, len(positions))]
    if len(positions):
        for width in WINDOW_WIDTHS:
            for start in range(0, int(positions[-1])+1, width//2):
                a, b = np.searchsorted(positions, [start, start+width])
                if b > a:
                    spans.append((int(a), int(b)))
    return np.asarray(sorted(set(spans)), dtype=np.int64)


@njit(cache=True, parallel=True)
def score_dyads(gl, observed, rows, spans, delta):
    """Return window/bet-averaged log E and jointly observed site count."""
    output = np.empty((len(rows), 2))
    for row in prange(len(rows)):
        a, b = rows[row]
        ratios = np.ones(gl.shape[1])
        exposure = 0
        for site in range(gl.shape[1]):
            if not observed[a, site] or not observed[b, site]:
                continue
            exposure += 1
            x0, x1, x2 = gl[a, site]
            y0, y1, y2 = gl[b, site]
            compatible = max(x0*max(y0, y1), x1*max(y0, y1, y2), x2*max(y1, y2))
            unrestricted = max(x0, x1, x2)*max(y0, y1, y2)
            bound = max(compatible, (1-delta)*compatible + delta*unrestricted)
            ratios[site] = (x0*y2+x2*y0)/2/max(bound, 1e-300)
        combined = -np.inf
        prefix = np.empty(gl.shape[1]+1)
        prefix[0] = 0.0
        for bet in BET_SIZES:
            for site in range(gl.shape[1]):
                prefix[site+1] = prefix[site]+np.log(max(1-bet+bet*ratios[site], 1e-300))
            for start, stop in spans:
                combined = np.logaddexp(combined, prefix[stop]-prefix[start])
        output[row, 0] = combined - np.log(len(BET_SIZES)*len(spans))
        output[row, 1] = exposure
    return output


def _score_chromosome(contig, state, phase, common):
    from .pipeline import _load_raw_evidence

    started = time.perf_counter()
    store = runtime.CheckpointStore(common['root'], nthreads=scheduling.current_threads())
    rows, delta = common['rows'], common['delta']
    gl, positions, observed, gl_payload, sites_payload = _load_raw_evidence(
        store, contig, **common['source'])
    if gl.shape != (common['samples'], len(positions), 3) or observed.shape != gl.shape[:2]:
        raise ValueError(f"{contig}: full-marker evidence axes differ from the pedigree")
    spans = physical_windows(positions)
    # Exact catalogs are shared across candidate batches. Samples with
    # unusually diverse GLs retain the direct scorer, without rounding.
    scheduling.current_threads()
    codes, catalog, counts = exclusion_patterns.encode_patterns(gl, observed)
    values = np.empty((len(rows), 2))
    for start in range(0, len(rows), _CANDIDATE_BATCH_SIZE):
        scheduling.current_threads()
        stop = min(len(rows), start+_CANDIDATE_BATCH_SIZE)
        batch = rows[start:stop]
        encoded = (counts[batch[:, 0]] >= 0) & (counts[batch[:, 1]] >= 0)
        batch_values = values[start:stop]
        if encoded.any():
            batch_values[encoded] = exclusion_patterns.score_patterns(
                codes, catalog, counts, batch[encoded], spans, delta, BET_SIZES)
        if not encoded.all():
            scheduling.current_threads()
            batch_values[~encoded] = score_dyads(gl, observed, batch[~encoded], spans, delta)
    summary = dict(contig=contig, markers=len(positions), candidates=len(rows),
                   seconds=time.perf_counter()-started, final_threads=scheduling.current_threads(),
                   pattern_fallback_samples=int(np.count_nonzero(counts < 0)))
    del gl, positions, observed, gl_payload, sites_payload, codes, catalog, counts
    store.nthreads = scheduling.current_threads()
    store.save_contig(common['stage'], contig, dict(values=values, summary=summary))
    return None, summary


def refine_parent_counts(store, contigs, sample_ids, result, *, parent_eligibility=None,
                         n_workers=None, raw_gl_stage, raw_sites_stage,
                         raw_gl_key="global_probs", raw_sites_key="global_sites",
                         raw_observed_mask_key="global_observed_mask"):
    """Checkpoint full-marker evidence and apply the default Tier-B release.

    Called after ordinary genotype/painting inference, before publication.
    This never adds likelihood scores or feeds fixed exclusions into bootstrap
    fits. Lower-level score-only APIs cannot run it without their raw markers.
    """
    settings = result.config
    names = tuple(sample_ids)
    resolved = eligibility._resolve_parent_eligibility(parent_eligibility, names)
    rows = release.exclusion_candidate_pairs(result.tier_b_relationships, resolved)
    metadata = dict(model=MODEL, enabled=settings.full_marker_exclusion,
                    alpha=settings.mendelian_exclusion_alpha,
                    genotype_replacement_probability=settings.mendelian_genotype_replacement_probability,
                    conditional_on_supported_edge_directions=True,
                    candidate_dyads=len(rows), contigs=len(contigs))
    result.exclusion_evidence = pd.DataFrame(columns=["Sample", "CandidateParent", "LogE", "ObservedSites"])
    if not settings.full_marker_exclusion or not len(rows):
        metadata['status'] = 'disabled' if not settings.full_marker_exclusion else 'no_unresolved_candidates'
        release.apply_full_marker_release(result, rows[:0], np.empty(0), resolved)
        return metadata

    cpus = runtime.available_cpu_count() if n_workers is None else int(n_workers)
    if not 1 <= cpus <= runtime.available_cpu_count():
        raise ValueError("exclusion workers must fit the current CPU affinity")
    source = dict(raw_gl_stage=raw_gl_stage, raw_sites_stage=raw_sites_stage,
                  raw_gl_key=raw_gl_key, raw_sites_key=raw_sites_key,
                  raw_observed_mask_key=raw_observed_mask_key)
    identity = dict(model=MODEL, sample_ids=list(names), contigs=list(contigs), rows=rows.tolist(),
                    genotype_replacement_probability=settings.mendelian_genotype_replacement_probability,
                    bets=BET_SIZES.tolist(), window_widths=list(WINDOW_WIDTHS), source=source,
                    source_files=cache.source_files(store, (raw_gl_stage, raw_sites_stage), contigs),
                    code={Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for path in (__file__, exclusion_patterns.__file__)})
    stage = cache.versioned(STAGE, identity)
    store.bind_stage_identity(stage, identity)
    pending = [c for c in contigs if not store.contig_done(stage, c)]
    started = time.perf_counter()
    if pending:
        from ..core.checkpoints import contig_path
        weights = [Path(contig_path(store.root, raw_gl_stage, contig)).stat().st_size
                   for contig in pending]
        common = dict(root=store.root, stage=stage, samples=len(names), rows=rows, source=source,
                      delta=settings.mendelian_genotype_replacement_probability)
        with scheduling.ChromosomeExecutor(pending, _score_chromosome,
                                           n_workers=cpus, weights=weights) as executor:
            print(f"  Full-marker exclusion: {len(rows)} dyads, {len(pending)} chromosomes, "
                  f"{executor.worker_count} workers sharing {executor.cpu_budget} CPUs", flush=True)
            executor.run('exclusion', common)
    payloads = [store.load_contig(stage, contig) for contig in contigs]
    log_e = np.logaddexp.reduce(np.stack([p['values'][:, 0] for p in payloads]), axis=0)-np.log(len(contigs))
    observed = np.sum([p['values'][:, 1] for p in payloads], axis=0).astype(np.int64)
    store.mark_stage_complete(stage)
    result.exclusion_evidence = pd.DataFrame(dict(
        Sample=[names[c] for c, _ in rows], CandidateParent=[names[p] for _, p in rows],
        LogE=log_e, ObservedSites=observed))
    release.apply_full_marker_release(result, rows, log_e, resolved)
    metadata.update(status='evaluated', seconds=time.perf_counter()-started,
                    chromosome_checkpoints_resumed=len(contigs)-len(pending),
                    checkpoint_stage=stage, chromosomes=[p['summary'] for p in payloads],
                    released=int(result.exclusion_diagnostics['Released'].sum()))
    return metadata
