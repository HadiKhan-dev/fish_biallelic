"""Family phase stage: pedigree-conditioned phase, genotypes and released uncertainty of every chromosome.

Per chromosome and painting component (docs/methods/family_phase.md): the founder-mosaic pedigree model on the
painting's labels and emissions (model.py) is solved by exact block coordinate ascent under both tie conventions;
each sample's release comes from its block's exact posterior in each solve and only their agreement is released
(release.py, ``_agreement``): genotypes at ``genotype_posterior``,
phase at ``phase_posterior``, phase sets and unphased heterozygotes. Calls are released in the final homologue order
(a sample with a released slot-k parent: homologue k is the slot assigned to that parent). Nothing feeds back
upstream.
"""
from __future__ import annotations

import time
from functools import partial

import numpy as np

from ..evidence import records as evidence_records
from ..execution import (BudgetedPool, ChromosomeExecutor, CpuBudget, affinity_cpus, claim_threads,
                         memory_worker_limit, release_memory, timed_stage)
from ..founder_refit import records as refit_records
from ..painting import hmm as painting_hmm
from ..painting import records as painting_records
from ..store import CheckpointStore
from ..store.mapped import mapped_array, discard_pages
from . import records
from .allele_evidence import EVIDENCE_CHUNK, allele_evidence, chromosome_prior, flip_probabilities
from .model import LabelModel, haldane
from .painting_inputs import component_bin_edges, coverage, tie_boundaries
from .records import FamilyPhaseDiagnostics, FamilyPhaseProduct
from .release import release_sample
from .window import SampleSites, SITE_WINDOW, site_windows, use_windows


@timed_stage("family_phase")
def run(store, contigs, edges, config, genetic_maps, budget):
    """Final phase of every chromosome (one executor task each, sharing the stage's CPU budget).

    Returns the per-contig FamilyPhaseSummary rows (also stored as run record ``family_phase``). Contigs whose
    product exists are not recomputed; an interrupted chromosome is recomputed (the stage keeps no work state).
    """
    contigs = tuple(map(str, contigs))
    store.require(painting_records.KEY, contigs)
    store.require(refit_records.KEY, contigs)
    estimates = [_task_bytes(store.contig(c), len(edges.sample_ids)) for c in contigs]
    workers = memory_worker_limit(estimates, budget.threads(), label="FAMILY PHASE")
    common = dict(root=store.root, edges=edges, config=config, genetic_maps=genetic_maps)
    with ChromosomeExecutor(contigs, _phase_contig, budget=budget, max_workers=workers, label="FAMILY PHASE",
                            task_bytes=estimates, envelope=True, frontload=True) as executor:
        summaries = tuple(executor.run("family_phase", common))
    store.run.put(records.KEY, summaries)
    store.mark_stage_complete(records.STAGE)
    return summaries


def _task_bytes(file, samples) -> int:
    """Diagnostic estimate: full component bin emissions and release-pool shared copy, plus 2 GiB.

    Above the evidence-byte budget, site projections are windowed and final arrays are file-backed.
    Below it, include c34fin's resident evidence/release arrays. Each release worker reports
    its bin tables and single-sample results (including CAL2 own terms) separately (_release_component). One checked compressed cache serves
    all workers in the coordinating process.
    CAL2 retains both solves' S x sites evidence/flip matrices.
    Dispatch still follows measured memory pressure.
    """
    if file.has(records.KEY):
        return 2 * file.info(records.KEY).decoded_bytes + (1 << 30)
    shape = painting_records.load_summary(file)
    largest = max((8 * samples * s.states * s.states * s.bins for s in shape.components), default=0)
    sites = len(evidence_records.load_arrays(file, ("positions",))["positions"])
    states = max((s.states for s in shape.components), default=0)
    with file.open_arrays(evidence_records.EVIDENCE, ("likelihoods",)) as arrays:
        gl = arrays["likelihoods"]
        windowed = use_windows(int(np.prod(gl.shape)) * gl.dtype.itemsize)
    resident = 0 if windowed else 48 * samples * sites
    return int(4 * largest + resident + 32 * states * sites + (2 << 30))


def _phase_contig(contig, state, phase, common, budget):
    store = CheckpointStore(common["root"])
    file = store.contig(contig)
    edges, config = common["edges"], common["config"]
    chromosome_map = common["genetic_maps"].for_contig(contig)
    identity = records.stage_identity(file, edges, config, chromosome_map)
    store.bind_identity(file, records.STAGE, identity)
    if file.has(records.KEY):
        return state, records.summary(records.load(file, nthreads=claim_threads()))
    painting_records.require_evidence_binding(file)
    product = phase_chromosome(file, contig, edges, config, chromosome_map, budget, root=store.root)
    records.write(file, product)
    summary = records.summary(product)
    del product
    release_memory()
    print(f"[family phase {contig}] released: {summary.phased_heterozygotes} phased / "
          f"{summary.unphased_heterozygotes} unphased heterozygous calls, {summary.phase_sets} phase sets "
          f"({summary.withheld_joins} withheld junction joins, {summary.tie_breaks} tie breaks)"
          + ("" if summary.settled else "; block ascent stopped at the sweep cap")
          + f", {summary.elapsed_seconds:.1f} s", flush=True)
    return state, summary


def _component_emissions(painted, chromosome_painting, file, positions, samples, windowed=True):
    """float64 [N, S, S, B] tempered ordered-pair emissions: the painting's stored cache, else recomputed with the
    painting's own kernel and design effect (the cache is optional; values are identical)."""
    S = painted.states.n_states
    if painted.packed_emissions is not None:
        return painting_hmm.unpack_emissions(painted.packed_emissions, S).reshape(samples, S, S, -1)
    if not windowed:
        evidence = evidence_records.load(file, nthreads=claim_threads())
        sites = np.searchsorted(evidence.positions, np.asarray(painted.states.positions, dtype=np.int64))
        gl, observed = painting_hmm.gather_sites(evidence.likelihoods, evidence.observed, sites)
        del evidence
        pcfg = chromosome_painting.config
        emissions = painting_hmm.binned_emissions(gl, observed, painted.states.alt_probability(),
                                                  painted.states.background_frequency, painted.bins.starts,
                                                  painted.bins.stops, robustness_epsilon=pcfg.robustness_epsilon,
                                                  log_floor=pcfg.emission_log_floor, before_block=claim_threads)
        emissions /= painting_hmm.design_effect(gl, observed, painted.bins.starts,
                                                painted.transitions.site_correlation)[:, None, :]
        return emissions.reshape(samples, S, S, -1)
    sites = np.searchsorted(positions, np.asarray(painted.states.positions, dtype=np.int64))
    pcfg = chromosome_painting.config
    starts, stops = painted.bins.starts, painted.bins.stops
    emissions = np.empty((samples, S * S, len(starts)), dtype=np.float64)
    alt, background = painted.states.alt_probability(), painted.states.background_frequency
    # Optional packed painting caches need not exist. Recompute the same complete bins, one sample at a time.
    with file.open_arrays(evidence_records.EVIDENCE, ("likelihoods", "observed")) as arrays:
        for n in range(samples):
            first = 0
            while first < len(starts):
                last = first + 1
                while last < len(starts) and stops[last] - starts[first] <= SITE_WINDOW:
                    last += 1
                left, right = int(starts[first]), int(stops[last - 1])
                gl = arrays["likelihoods"].gather(slice(n, n + 1), sites[left:right])
                observed = arrays["observed"].gather(slice(n, n + 1), sites[left:right])
                raw = painting_hmm.binned_emissions(gl, observed, alt[:, left:right], background[left:right],
                    starts[first:last] - left, stops[first:last] - left,
                    robustness_epsilon=pcfg.robustness_epsilon, log_floor=pcfg.emission_log_floor,
                    before_block=claim_threads)
                raw /= painting_hmm.design_effect(gl, observed, starts[first:last] - left,
                                                  painted.transitions.site_correlation)[:, None, :]
                emissions[n:n + 1, :, first:last] = raw
                first = last
    return emissions.reshape(samples, S, S, -1)


def _state_alleles(states, panel_inference, n_states):
    """int8 [S, Lc]: each painting state's allele at every panel site: the common refit allele of its rows (all
    called and agreeing), else -1; BACKGROUND -1."""
    out = np.full((n_states, panel_inference.shape[1]), -1, dtype=np.int8)
    for c, rows in enumerate(states.classes):
        rows = list(rows)
        allele = panel_inference[rows[0]].astype(np.int16)
        ok = allele >= 0
        for r in rows[1:]:
            ok &= panel_inference[r] == allele
        out[c] = np.where(ok, allele, -1)
    return out


def _edge_rates(painted, chromosome_map, config, protected):
    """(theta_l, theta_m, alpha, beta) per bin edge: the painting's per-track label switch (its pair stay
    probability is the square of a track's; at most (S - 1) / S, the uniform-destination limit), the Haldane meiosis
    switch between bin centres (1/2 across protected intervals), and the not-a-copy entry and exit."""
    centres = np.asarray(painted.bins.centers, dtype=np.float64)
    dbp = np.diff(centres)
    S = painted.states.n_states
    theta_l = np.clip(1.0 - np.sqrt(np.exp(np.asarray(painted.transitions.log_same, dtype=np.float64))), 1e-15,
                      (S - 1) / S)
    morgans = np.asarray(chromosome_map.interval_morgans(centres[:-1], centres[1:]), dtype=np.float64)
    theta_m = np.clip(haldane(morgans), 1e-12, 0.5)
    for start, end in protected:                                  # transmission resets at order breakpoints
        theta_m[(centres[1:] >= start) & (centres[:-1] <= end)] = 0.5
    alpha = np.clip(1.0 - np.exp(-config.not_copy_rate_per_bp * dbp), 1e-12, 0.5)
    beta = np.clip(1.0 - np.exp(-dbp / config.not_copy_mean_bp), 1e-6, 1.0 - 1e-6)
    return theta_l, theta_m, alpha, beta


def _phase_set_count(phased, component_ids, breaks):
    """Released phase sets holding at least one phased heterozygote: per sample, components split at its phase
    breaks (as export and evaluation form them)."""
    L = len(component_ids)
    edge = np.zeros(L, dtype=bool)
    edge[1:] = component_ids[1:] != component_ids[:-1]
    count = 0
    for n, sample_breaks in enumerate(breaks):
        sites = phased[n] & (component_ids >= 0)
        if not sites.any():
            continue
        cut = edge.copy()
        starts = np.asarray(sorted(set(sample_breaks)), dtype=np.int64) + 1
        cut[starts[(starts > 0) & (starts < L)]] = True
        count += len(np.unique(np.cumsum(cut)[sites]))
    return count


def _release_bytes(model, sites):
    """Peak bytes of one sample's release: its block tables (emissions, three transition factors, forward and
    backward), three per-bin label-pair posteriors, its site likelihoods, per-site projections and CAL2 own terms."""
    states = (model.S + 2) ** 2 * 2
    return int(8 * model.B * (4 * states + 3 * model.S * model.S) + 220 * sites)


# Each pool serves a component: both solved models, with site likelihoods sent in bounded sample batches.
_WORKER_MODEL = [None, None]


def _paired_release(models, n, ctx, uncertainty):
    releases = []
    for model, terms in zip(models, uncertainty):
        common = {k: v for k, v in terms.items() if k != "excluded"}
        releases.append(release_sample(model, n, dict(ctx, excluded=bool(terms["excluded"][n]), **common)))
    first, second = releases
    bridged = len(set(ctx["tie_bins"]) - set(first[2]) - set(second[2]))
    return n, _agreement(first, second, ctx["site_bin"]), first[3]["chain"], second[3]["chain"], bridged


def _release_task(task, shared, budget):
    """One sample, both conventions; only this sample's site results leave the worker."""
    n, break_bins, tie_bins, site_gl, token, config, scores = task
    if _WORKER_MODEL[0] != token:
        # Preserve the baseline worker descendant order for BOTH attached solves: it determines the exact
        # summation order in block_tables, independently of the convention that produced X and V.
        models = []
        for prefix in ("", "mirror_"):
            model = LabelModel(shared["E"], shared["par"], shared["theta_l"], shared["theta_m"], shared["alpha"],
                               shared["beta"], config)
            model.X, model.V = shared[prefix + "X"], shared[prefix + "V"]
            models.append(model)
        _WORKER_MODEL[:] = [token, models]
    ctx = dict(config=config, site_bin=shared["site_bin"], site_alleles=shared["site_alleles"], site_gl=site_gl,
               break_bins=break_bins, tie_bins=tie_bins, scores=scores, chain=True)
    uncertainty = [dict(allele_evidence=shared[prefix + "allele_evidence"], flips=shared[prefix + "flips"],
                        logit_prior=float(shared["flip_params"][0]),
                        robust_epsilon=float(shared["flip_params"][1]), excluded=shared["excluded"])
                   for prefix in ("", "mirror_")]
    return _paired_release(_WORKER_MODEL[1], n, ctx, uncertainty)


def _release_component(model, mirror, rates, site_bin, site_alleles, file, columns, break_bins, tie_bins, config,
                       budget, token, uncertainty, scores=False):
    """Yield completed sample releases. Bin posteriors stay global; evidence/projections use whole-bin windows."""
    task = _release_bytes(model, len(site_bin))
    workers = min(model.N, max(1, int(budget.threads())))
    if workers <= 1:
        with file.open_arrays(evidence_records.EVIDENCE, ("likelihoods",), nthreads=1) as arrays:
            for n in range(model.N):
                source = SampleSites(arrays["likelihoods"], n, columns)
                site_gl = np.empty((len(site_bin), 3), dtype=np.float32)
                for left, right in site_windows(site_bin):
                    site_gl[left:right] = source[left:right]
                ctx = dict(config=config, site_bin=site_bin, site_alleles=site_alleles,
                           site_gl=site_gl, break_bins=break_bins[n],
                           tie_bins=tie_bins[n], scores=scores, chain=True)
                yield _paired_release((model, mirror), n, ctx, uncertainty)
        return
    shared = dict(E=model.E, X=model.X, V=model.V, mirror_X=mirror.X, mirror_V=mirror.V, par=model.par,
                  theta_l=rates[0], theta_m=rates[1], alpha=rates[2], beta=rates[3], site_bin=site_bin,
                  site_alleles=site_alleles,
                  flip_params=np.asarray([uncertainty[0]["logit_prior"], uncertainty[0]["robust_epsilon"]]),
                  excluded=np.asarray(uncertainty[0]["excluded"], dtype=np.bool_))
    for prefix, terms in zip(("", "mirror_"), uncertainty):
        shared[prefix + "allele_evidence"] = terms["allele_evidence"]
        shared[prefix + "flips"] = terms["flips"]
    # One checked compressed cache per chromosome, not one per release worker. Likelihoods cross the process
    # boundary as float32 sample rows (12*m bytes each); at most 2*workers rows are queued. Decoding and all site
    # projections still use whole-bin windows, and the reader follows the sample-major record in order.
    with file.open_arrays(evidence_records.EVIDENCE, ("likelihoods",), nthreads=1) as arrays:
        with BudgetedPool(budget, processes=workers, shared=shared, label="family-phase release") as pool:
            for first in range(0, model.N, 2 * workers):
                tasks = []
                for n in range(first, min(model.N, first + 2 * workers)):
                    source = SampleSites(arrays["likelihoods"], n, columns)
                    site_gl = np.empty((len(site_bin), 3), dtype=np.float32)
                    for left, right in site_windows(site_bin):
                        site_gl[left:right] = source[left:right]
                    tasks.append((n, break_bins[n], tie_bins[n], site_gl, token, config, scores))
                yield from pool.imap_unordered(_release_task, tasks, task_bytes=task)


_RESIDENT_WORKER_MODEL = [None, None]


def _resident_release_task(task, shared, budget):
    """Pool task (worker side): one sample's release from the shared converged state of its component."""
    n, break_bins, tie_bins, site_gl, token, config, scores = task
    if _RESIDENT_WORKER_MODEL[0] != token:
        model = LabelModel(shared["E"], shared["par"], shared["theta_l"], shared["theta_m"], shared["alpha"],
                           shared["beta"], config)
        model.X, model.V = shared["X"], shared["V"]
        _RESIDENT_WORKER_MODEL[:] = [token, model]
    ctx = dict(windowed=False, config=config, site_bin=shared["site_bin"], site_alleles=shared["site_alleles"], site_gl=site_gl,
               break_bins=break_bins, tie_bins=tie_bins, scores=scores, chain=True,
               allele_evidence=shared["allele_evidence"], flips=shared["flips"],
               logit_prior=float(shared["flip_params"][0]), robust_epsilon=float(shared["flip_params"][1]),
               excluded=bool(shared["excluded"][n]))
    return release_sample(_RESIDENT_WORKER_MODEL[1], n, ctx)


def _resident_release_component(model, rates, site_bin, site_alleles, site_gl, break_bins, tie_bins, config, budget, token,
                       uncertainty, scores=False):
    """Every sample's release (in sample order, with its meiosis chains' posteriors: release_sample's "chain") from
    the solved component (site_gl float32 [N, m, 3]: the samples' genotype likelihoods at the released sites;
    break_bins and tie_bins per sample, release_sample's; ``uncertainty``: the solve's allele uncertainty, a dict of
    allele_evidence and flips [S, m], logit_prior, robust_epsilon and excluded bool [N], as release_sample reads it).
    Releases are independent given the converged state: one process when the budget has one CPU, else an
    execution.BudgetedPool sharing that state."""
    task = _release_bytes(model, len(site_bin))            # reported to the pool (its memory pacing measures)
    workers = min(model.N, max(1, int(budget.threads())))
    if workers <= 1:
        common = {k: v for k, v in uncertainty.items() if k != "excluded"}
        return [release_sample(model, n, dict(windowed=False, config=config, site_bin=site_bin, site_alleles=site_alleles,
                                              site_gl=site_gl[n], break_bins=break_bins[n], tie_bins=tie_bins[n],
                                              scores=scores, chain=True, excluded=bool(uncertainty["excluded"][n]),
                                              **common))
                for n in range(model.N)]
    shared = dict(E=model.E, X=model.X, V=model.V, par=model.par, theta_l=rates[0], theta_m=rates[1],
                  alpha=rates[2], beta=rates[3], site_bin=site_bin, site_alleles=site_alleles,
                  allele_evidence=uncertainty["allele_evidence"], flips=uncertainty["flips"],
                  flip_params=np.asarray([uncertainty["logit_prior"], uncertainty["robust_epsilon"]]),
                  excluded=np.asarray(uncertainty["excluded"], dtype=np.bool_))
    tasks = [(n, break_bins[n], tie_bins[n], site_gl[n], token, config, scores) for n in range(model.N)]
    with BudgetedPool(budget, processes=workers, shared=shared, label="family-phase release") as pool:
        return pool.map(_resident_release_task, tasks, task_bytes=task)


def _resident_releases(model, mirror, rates, site_bin, site_alleles, site_gl, break_bins, tie_bins, config,
                       budget, token, excluded, logit_prior, robust_epsilon, scores):
    """c34fin's resident CAL2 calculation and per-solve scheduling, sharing the same release arithmetic."""
    uncertainty = []
    for solved in (model, mirror):
        evidence = allele_evidence(solved.X, site_bin, site_alleles, site_gl, excluded, robust_epsilon)
        uncertainty.append(dict(allele_evidence=evidence,
                                flips=flip_probabilities(evidence, site_alleles, logit_prior),
                                logit_prior=logit_prior, robust_epsilon=robust_epsilon, excluded=excluded))
    first = _resident_release_component(model, rates, site_bin, site_alleles, site_gl, break_bins, tie_bins,
                                        config, budget, token, uncertainty[0], scores)
    second = _resident_release_component(mirror, rates, site_bin, site_alleles, site_gl, break_bins, tie_bins,
                                         config, budget, token + ":mirror", uncertainty[1], scores)
    for n, (a, b) in enumerate(zip(first, second)):
        bridged = len(set(tie_bins[n]) - set(a[2]) - set(b[2]))
        yield n, _agreement(a, b, site_bin), a[3]["chain"], b[3]["chain"], bridged


def _resident_array(shape, dtype, fill=0):
    """Match mapped_array's allocation signature for resident output buffers."""
    return np.full(shape, fill, dtype=dtype)


def _allele_uncertainty(models, file, columns, site_bin, site_alleles, excluded, logit_prior, robust_epsilon, *, root):
    """CAL2's unchanged carrier sums over its original EVIDENCE_CHUNK boundaries.

    A temporary float32 spool supplies N x chunk likelihoods. Within each site CAL2 adds all homologue-0 terms in
    sample order, then all homologue-1 terms. The S x sites evidence/flip outputs remain whole, and the original
    flip_probabilities call sees its original shape; CAL2's arithmetic, prior and leave-one-out terms are unchanged.
    """
    # Stage sample-major reads once; gathering all samples from the compressed record for each site window
    # would repeatedly decompress that record. The temporary float32 spool has the baseline's shape/dtype.
    site_gl = mapped_array((models[0].N, len(site_bin), 3), np.float32, root=root)
    with file.open_arrays(evidence_records.EVIDENCE, ("likelihoods",), nthreads=1) as arrays:
        for n in range(models[0].N):
            source = SampleSites(arrays["likelihoods"], n, columns)
            for left, right in site_windows(site_bin):
                site_gl[n, left:right] = source[left:right]
            discard_pages(site_gl[n])
    evidence = [np.zeros(site_alleles.shape, dtype=np.float64) for _ in models]
    for left in range(0, len(site_bin), EVIDENCE_CHUNK):
        right = min(len(site_bin), left + EVIDENCE_CHUNK)
        for solved, target in zip(models, evidence):
            target[:, left:right] = allele_evidence(solved.X, site_bin[left:right],
                site_alleles[:, left:right], site_gl[:, left:right], excluded, robust_epsilon)
        discard_pages(site_gl)
    del site_gl
    return [dict(allele_evidence=e, flips=flip_probabilities(e, site_alleles, logit_prior),
                 logit_prior=logit_prior, robust_epsilon=robust_epsilon, excluded=excluded) for e in evidence]


def _set_start_bins(set_starts, site_bin, B):
    """(reset bool [B], blocked bool [B]) of one sample's released phase-set starts (indices into the component's
    released sites): a meiosis chain resets at the bin holding each start; a start inside a bin (after its first
    released site) also blocks that bin, whose sites belong to two sets, and resets after it."""
    reset = np.zeros(B, dtype=bool)
    blocked = np.zeros(B, dtype=bool)
    for i in set_starts:
        b = int(site_bin[i])
        reset[b] = True
        if i > 0 and site_bin[i - 1] == b:
            blocked[b] = True
            if b + 1 < B:
                reset[b + 1] = True
    return reset, blocked


def _edge_chains(model, mirror, chains, mirrored_chains, theta_m, starts, tau):
    """{(parent, child, slot): (source int8 [B], support float32 [B], reset bool [B])} for every linked homologue of
    the component, from both solves' release chains (release_sample's "chain").

    A chain resets at the component's first bin, at meiosis resets (theta_m = 1/2) and at the parent's and the
    child's released phase-set starts (_set_start_bins). Source 0/1 refers to the parent's homologue order: slot
    order for a parent with a linked parent, else each solve's own gauge, so within a segment the mirrored solve's
    sources are aligned to the default solve by the parent's labels (a segment whose two label orders disagree is
    blocked), and a bin is released only where the parent's aligned label pair is the same in both solves, so that
    a source names the same copied labels. The released source of a bin is the parent homologue that both aligned
    solves give a posterior of at least ``tau`` (else -1), its support the smaller of the two posteriors.
    """
    B = model.B
    base = np.zeros(B, dtype=bool)
    base[0] = True
    base[1:] |= np.asarray(theta_m) >= 0.5 - 1e-12
    out = {}
    for c in range(model.N):
        for k in (0, 1):
            p = int(model.par[c, k])
            if p < 0:
                continue
            a, b = chains[c][k], mirrored_chains[c][k]
            reset = base | starts[p][0] | starts[c][0]
            blocked = starts[p][1] | starts[c][1]
            segment = np.cumsum(reset)
            flip = np.zeros(B, dtype=bool)
            if np.all(model.par[p] < 0):
                X, Y = model.X[p], mirror.X[p]
                comparable = (X[0] != X[1]) & (np.minimum(X[0], X[1]) == np.minimum(Y[0], Y[1])) \
                    & (np.maximum(X[0], X[1]) == np.maximum(Y[0], Y[1]))
                swapped = comparable & (Y[0] == X[1])
                for g in np.unique(segment[comparable]):
                    inside = segment == g
                    n_swapped = int(np.sum(swapped & inside))
                    if 0 < n_swapped < int(np.sum(comparable & inside)):
                        blocked = blocked | inside
                    elif n_swapped:
                        flip |= inside
            X, Y = model.X[p], mirror.X[p]
            blocked = blocked | (np.where(flip, Y[1], Y[0]) != X[0]) | (np.where(flip, Y[0], Y[1]) != X[1])
            va = np.where(a[0] >= tau, 0, np.where(a[1] >= tau, 1, -1))
            vb = np.where(b[0] >= tau, 0, np.where(b[1] >= tau, 1, -1))
            vb = np.where((vb >= 0) & flip, 1 - vb, vb)
            source = np.where((va >= 0) & (va == vb) & ~blocked, va, -1).astype(np.int8)
            support = np.where(source >= 0, np.minimum(np.maximum(a[0], a[1]), np.maximum(b[0], b[1])),
                               0.0).astype(np.float32)
            out[(p, c, k)] = (source, support, reset)
    return out


def _agreement(first, second, site_bin):
    """One sample's release from two solves under opposite tie conventions (released outputs must not depend on a
    convention): genotypes where both call the same genotype (else unknown); a heterozygote is phased where both phase
    it; phase sets break at both solves' breaks and wherever their relative orientation changes (so within a set the
    two agree up to one global flip, and the first solve's orientation is released). Returns (calls, undetermined,
    site indices where a new phase set starts) and, when both releases carry scores, float32 [2, m]: the smaller of
    the two solves' scores, -1 where the agreement releases no genotype (row 0) or no phased heterozygote (row 1)."""
    calls_a, und_a, breaks_a = first[:3]
    calls_b, und_b, breaks_b = second[:3]
    scores_a = first[3].get("scores") if len(first) > 3 else None
    scores_b = second[3].get("scores") if len(second) > 3 else None
    m = len(site_bin)
    gt_a = np.where((calls_a >= 0).all(axis=1), calls_a.sum(axis=1, dtype=np.int64), -1)
    gt_b = np.where((calls_b >= 0).all(axis=1), calls_b.sum(axis=1, dtype=np.int64), -1)
    agree = gt_a == gt_b
    calls = np.where(agree[:, None], calls_a, np.int8(-1)).astype(np.int8)
    het = agree & (gt_a == 1)
    phased = het & ~und_a & ~und_b
    starts = np.zeros(m, dtype=bool)
    cut_bins = np.asarray(sorted(set(breaks_a) | set(breaks_b)), dtype=np.int64)
    first_sites = np.searchsorted(site_bin, cut_bins, side="left")
    starts[first_sites[first_sites < m]] = True
    idx = np.flatnonzero(phased)
    if len(idx) > 1:
        flipped = (calls_a[idx, 0] != calls_b[idx, 0]).astype(np.int8)
        same_set = np.diff(np.cumsum(starts)[idx]) == 0
        starts[idx[1:][(np.diff(flipped) != 0) & same_set]] = True
    starts[:1] = False
    if scores_a is not None and scores_b is not None:
        scores = np.minimum(scores_a, scores_b)
        scores[0, ~agree | (gt_a < 0)] = -1.0                     # no genotype released
        scores[1, ~phased] = -1.0                                   # no phased heterozygote released
        return calls, het & ~phased, np.flatnonzero(starts), scores
    return calls, het & ~phased, np.flatnonzero(starts)


def phase_chromosome(file, contig, edges, config, chromosome_map, budget=None, scores=False, *, root, storage="auto"):
    """FamilyPhaseProduct of one chromosome (``budget``: the CPUs of its releases; None = this process's
    affinity). With ``scores``, also float32 [N, L, 2]: each released site's genotype score and ordered allele score
    (_agreement; -1 where nothing is released), for reporting how the scores behave.
    Storage is selected by decoded likelihood bytes; ``storage="resident"`` or ``"windowed"`` forces a mode for
    numerical/replay checks. It changes buffers and site batching only, never the model or scientific records.
    ``root`` is the checkpoint store's root, also used for anonymous windowed scratch files."""
    started = time.perf_counter()
    budget = CpuBudget.fixed(affinity_cpus()) if budget is None else budget
    loaded = evidence_records.load_arrays(file, ("positions", "founder_excluded"), nthreads=claim_threads())
    positions = np.asarray(loaded["positions"], dtype=np.int64)
    excluded = np.asarray(loaded["founder_excluded"], dtype=np.bool_)
    del loaded
    sample_ids = tuple(edges.sample_ids)
    N, L = len(sample_ids), len(positions)
    with file.open_arrays(evidence_records.EVIDENCE, ("likelihoods",)) as arrays:
        gl = arrays["likelihoods"]
        if gl.shape[:2] != (N, L):
            raise ValueError(f"{contig}: the evidence's likelihoods do not match the pedigree's samples and sites")
        windowed = use_windows(int(np.prod(gl.shape)) * gl.dtype.itemsize, storage)
    likelihoods = None
    if not windowed:
        loaded_gl = evidence_records.load_arrays(file, ("likelihoods",), nthreads=claim_threads())
        likelihoods = np.asarray(loaded_gl["likelihoods"], dtype=np.float32)
        del loaded_gl
    new_array = partial(mapped_array, root=root) if windowed else _resident_array
    par = np.full((N, 2), -1, dtype=np.int64)
    for p, c, k in zip(edges.parents, edges.children, edges.slots):
        par[int(c), int(k)] = int(p)
    chromosome_painting = painting_records.load(file, panels=False, emissions=True)
    if tuple(chromosome_painting.sample_ids) != sample_ids:
        raise ValueError(f"{contig}: the painting's samples differ from the pedigree's")
    painting = chromosome_painting.components
    panels = refit_records.founder_panels(file)
    logit_prior = chromosome_prior(panels)
    robust_epsilon = float(chromosome_painting.config.robustness_epsilon)
    ties = tie_boundaries(painting, positions, np.arange(N))
    cover, cover_offsets = coverage(painting, positions, N)
    calls = new_array((N, L, 2), np.int8, -1)
    undetermined = new_array((N, L), np.int8)
    site_scores = new_array((N, L, 2), np.float32, -1) if scores else None
    component_ids = np.full(L, -1, dtype=np.int32)
    breaks = [[] for _ in range(N)]
    all_edges, sweeps, settled, objective, candidates, mirrored_solves, chain_parts = [], [], [], [], [], [], []
    pairing_rows = []
    all_centers = []
    withheld_joins = tie_breaks = 0
    protected = config.protected_for(contig)
    for index, (painted, panel) in enumerate(zip(painting, panels)):
        source = np.asarray(panel.positions, dtype=np.int64)
        bin_edges = component_bin_edges(painted, source)
        all_edges.append(bin_edges)
        # the painting's bin centres (mean position of each bin's sites: the transitions' distances); midpoints of
        # the fallback bins of a component without painting bins, which carry no chain
        all_centers.append(np.asarray(painted.bins.centers, dtype=np.float64) if len(painted.bins.edges) > 1
                           else (bin_edges[:-1] + bin_edges[1:]) / 2.0)
        lo = int(np.searchsorted(positions, source[0], side="left"))
        hi = int(np.searchsorted(positions, source[-1], side="right"))
        component_ids[lo:hi] = index
        if len(painted.bins.edges) <= 1:
            sweeps.append(0)
            settled.append(True)
            objective.append(())
            candidates.append(0)
            mirrored_solves.append({})
            pairing_rows.append({})
            continue
        E = _component_emissions(painted, chromosome_painting, file, positions, N, windowed)
        S, B = E.shape[1], E.shape[3]
        rates = _edge_rates(painted, chromosome_map, config, protected)
        # both tie conventions, whatever the run's (maths.ties.reverse_conventions): the release is their agreement
        model = LabelModel(E, par, *rates, config, reverse=False)
        mirror = LabelModel(E, par, *rates, config, reverse=True)
        del E
        claim_threads()
        model.solve()
        mirror.solve()
        sweeps.append(len(model.history) - 1)
        settled.append(model.settled and mirror.settled)
        objective.append(tuple(model.history))
        junction = model.junction_candidates()
        mirror_junction = mirror.junction_candidates()
        candidates.append(len({c[1:] for c in junction} | {c[1:] for c in mirror_junction}))
        mirrored_solves.append(dict(sweeps=len(mirror.history) - 1, settled=bool(mirror.settled),
                                    objective=list(mirror.history), junction_candidates=len(mirror_junction),
                                    default_settled=bool(model.settled),
                                    default_junction_candidates=len(junction)))
        withheld = model.withheld_joins(junction)
        for n, bins_n in mirror.withheld_joins(mirror_junction).items():
            withheld[n] = sorted(set(withheld.get(n, ())) | set(bins_n))
        # parents' pairings the data do not decide (model.ambiguous_pairings, in either solve): the parent and every
        # child break there; the orientation gates never bridge these breaks
        pairing = {}
        for solved in (model, mirror):
            for p, j in solved.ambiguous_pairings():
                for c in [p] + solved.children_of(p):
                    pairing.setdefault(c, set()).add(int(j))
        pairing_rows.append(dict(candidates=len(model.pairings), moves=len(model.pairing_moves),
                                 mirrored_moves=len(mirror.pairing_moves),
                                 ambiguous=len({pj for m in (model, mirror) for pj in m.ambiguous_pairings()}),
                                 breaks=int(sum(len(v) for v in pairing.values()))))
        local = positions[lo:hi]
        cols = np.searchsorted(source, local)
        valid = cols < len(source)
        valid[valid] &= source[cols[valid]] == local[valid]
        bins = np.searchsorted(bin_edges, local, side="right") - 1
        inside = valid & (bins >= 0) & (bins < B)
        site_idx = np.flatnonzero(inside)
        site_bin = bins[site_idx]
        site_alleles = _state_alleles(painted.states, np.asarray(panel.inference, dtype=np.int8), S)[:, cols[site_idx]]
        break_bins, tie_cuts = [], []
        for n in range(N):
            tie_bins = {int(np.searchsorted(bin_edges, positions[m + 1], side="right") - 1)
                        for m in ties.get(n, ()) if lo <= m < hi - 1}
            # painting ties matter only to a sample with no linked parent (a linked homologue is ordered by origin)
            tie_bins = {b for b in tie_bins if 0 < b < B} if np.all(par[n] < 0) else set()
            joins = set(withheld.get(n, ()))
            pairs = {b for b in pairing.get(n, ()) if 0 < b < B}
            tie_breaks += len(tie_bins)
            withheld_joins += len(joins - tie_bins)
            break_bins.append(sorted(tie_bins | joins | pairs))
            tie_cuts.append(sorted(tie_bins - joins - pairs))   # a sample anchored by its children may bridge these
        starts = [None] * N
        chains, mirrored_chains = [None] * N, [None] * N
        token = f"{contig}:{index}:{id(model)}"
        if windowed:
            uncertainty = _allele_uncertainty((model, mirror), file, lo + site_idx, site_bin, site_alleles,
                                             excluded, logit_prior, robust_epsilon, root=root)
            releases = _release_component(model, mirror, rates, site_bin, site_alleles, file, lo + site_idx,
                                          break_bins, tie_cuts, config, budget, token, uncertainty, scores)
        else:
            site_gl = np.ascontiguousarray(likelihoods[:, lo + site_idx])
            releases = _resident_releases(model, mirror, rates, site_bin, site_alleles, site_gl, break_bins,
                                          tie_cuts, config, budget, token, excluded, logit_prior,
                                          robust_epsilon, scores)
        for n, agreed, chain, mirrored_chain, bridged in releases:
            tie_breaks -= bridged
            sample_calls, sample_undetermined, set_starts = agreed[:3]
            if site_scores is not None:
                site_scores[n, lo + site_idx] = agreed[3].T
                discard_pages(site_scores[n])
            calls[n, lo + site_idx] = sample_calls
            undetermined[n, lo + site_idx] = np.where(sample_undetermined, 3, 0)
            discard_pages(calls[n])
            discard_pages(undetermined[n])
            breaks[n].extend(lo + int(site_idx[i]) - 1 for i in set_starts)
            starts[n] = _set_start_bins(set_starts, site_bin, B)
            chains[n], mirrored_chains[n] = chain, mirrored_chain
        chain_parts.append((index, _edge_chains(model, mirror, chains, mirrored_chains, rates[1], starts,
                                                config.phase_posterior)))
        del model, mirror, releases, chains, mirrored_chains
        if windowed:
            del uncertainty
        else:
            del site_gl
    called_alleles = phased_heterozygotes = unphased_heterozygotes = sets = 0
    # Summaries read one sample at a time; no chromosome-sized boolean temporaries.
    for n in range(N):
        row = calls[n]
        het = (row[:, 0] >= 0) & (row[:, 1] >= 0) & (row[:, 0] != row[:, 1])
        unphased = het & (undetermined[n] > 0)
        called_alleles += int(np.sum(row >= 0))
        phased_heterozygotes += int(np.sum(het & ~unphased))
        unphased_heterozygotes += int(np.sum(unphased))
        sets += _phase_set_count((het & ~unphased)[None, :], component_ids, [breaks[n]])
        discard_pages(row)
        discard_pages(undetermined[n])
    phase_breaks = np.asarray([b for sample in breaks for b in sorted(set(sample))], dtype=np.int64)
    break_offsets = np.r_[0, np.cumsum([len(set(sample)) for sample in breaks])].astype(np.int64)
    diagnostics = FamilyPhaseDiagnostics(
        sweeps=tuple(sweeps), settled=tuple(settled), objective=tuple(objective),
        junction_candidates=tuple(candidates), withheld_joins=withheld_joins, tie_breaks=tie_breaks,
        called_alleles=called_alleles, phased_heterozygotes=phased_heterozygotes,
        unphased_heterozygotes=unphased_heterozygotes, phase_sets=sets,
        elapsed_seconds=time.perf_counter() - started, extra=dict(mirrored_solves=mirrored_solves,
                                                                  pairings=pairing_rows))
    edge_offsets = np.r_[0, np.cumsum([len(e) for e in all_edges])].astype(np.int64)
    chain = _chain_arrays(edges, all_edges, chain_parts, windowed, root=root)
    zeros = new_array((N, L), np.int8)
    product = FamilyPhaseProduct(
        contig=contig, sample_ids=sample_ids, positions=positions, component_ids=component_ids,
        bin_edges=np.concatenate(all_edges).astype(np.int64) if all_edges else np.zeros(0, dtype=np.int64),
        bin_edge_offsets=edge_offsets,
        bin_centers=np.concatenate(all_centers).astype(np.float64) if all_centers else np.zeros(0),
        calls=calls, displayed_frame=zeros, decoder_frame=new_array((N, L), np.int8),
        coverage=cover, coverage_offsets=cover_offsets, edges_identity=edges.identity, diagnostics=diagnostics,
        frame_undetermined=undetermined, phase_breaks=phase_breaks, phase_break_offsets=break_offsets, **chain)
    return (product, site_scores) if scores else product


def _chain_arrays(edges, all_edges, chain_parts, windowed=True, *, root):
    """The product's meiosis-chain arrays, one row per pedigree edge (edges order) over the chromosome's painting
    bins (components concatenated in bin_edges order): chain_source int8 (-1 = no released source), chain_support
    float32 and chain_reset bool (True at each component's first bin and at every reset)."""
    rows = {(int(p), int(c), int(k)): i for i, (p, c, k) in enumerate(zip(edges.parents, edges.children, edges.slots))}
    sizes = [len(e) - 1 for e in all_edges]
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    total = int(offsets[-1])
    new_array = partial(mapped_array, root=root) if windowed else _resident_array
    source = new_array((len(rows), total), np.int8, -1)
    support = new_array((len(rows), total), np.float32)
    reset = new_array((len(rows), total), bool)
    reset[:, offsets[:-1][np.asarray(sizes) > 0]] = True
    for index, part in chain_parts:
        first = int(offsets[index])
        for key, (s, w, r) in part.items():
            i = rows[key]
            stop = first + len(s)
            source[i, first:stop], support[i, first:stop], reset[i, first:stop] = s, w, r
            for array in (source, support, reset):
                discard_pages(array[i])
    return dict(chain_source=source, chain_support=support, chain_reset=reset)
