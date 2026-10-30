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

import numpy as np

from ..evidence import records as evidence_records
from ..execution import (BudgetedPool, ChromosomeExecutor, CpuBudget, affinity_cpus, claim_threads,
                         memory_worker_limit, release_memory, timed_stage)
from ..founder_refit import records as refit_records
from ..painting import hmm as painting_hmm
from ..painting import records as painting_records
from ..store import CheckpointStore
from . import records
from .model import LabelModel, haldane
from .painting_inputs import component_bin_edges, coverage, tie_boundaries
from .records import FamilyPhaseDiagnostics, FamilyPhaseProduct
from .release import release_sample


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
    """Peak estimate of one chromosome: the dense pair emissions of its largest component (float64 N x S x S x B)
    and their shared copy for the release pool, the release's per-sample arrays over the chromosome's sites (24
    bytes per sample and site), the chromosome's float32 genotype likelihoods and a component's slice of them for the
    site model (12 + 12), and 2 GiB for Python and the kernels. Release workers report their own per-sample estimate (_release_bytes) to the
    pool, whose dispatch follows measured memory pressure (execution.pools)."""
    if file.has(records.KEY):
        return 2 * file.info(records.KEY).decoded_bytes + (1 << 30)
    shape = painting_records.load_summary(file)
    sites = len(evidence_records.load_arrays(file, ("positions",))["positions"])
    largest = max((8 * samples * s.states * s.states * s.bins for s in shape.components), default=0)
    return int(4 * largest + 48 * samples * sites + (2 << 30))


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
    product = phase_chromosome(file, contig, edges, config, chromosome_map, budget)
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


def _component_emissions(painted, chromosome_painting, file, positions, samples):
    """float64 [N, S, S, B] tempered ordered-pair emissions: the painting's stored cache, else recomputed with the
    painting's own kernel and design effect (the cache is optional; values are identical)."""
    S = painted.states.n_states
    if painted.packed_emissions is not None:
        return painting_hmm.unpack_emissions(painted.packed_emissions, S).reshape(samples, S, S, -1)
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
    backward), three per-bin label-pair posteriors, its site likelihoods and the per-site projections."""
    states = (model.S + 2) ** 2 * 2
    return int(8 * model.B * (4 * states + 3 * model.S * model.S) + 92 * sites)


# Worker-side model of the component a release pool serves: (token, LabelModel) built once per worker.
_WORKER_MODEL = [None, None]


def _release_task(task, shared, budget):
    """Pool task (worker side): one sample's release from the shared converged state of its component."""
    n, break_bins, site_gl, token, config, scores = task
    if _WORKER_MODEL[0] != token:
        model = LabelModel(shared["E"], shared["par"], shared["theta_l"], shared["theta_m"], shared["alpha"],
                           shared["beta"], config)
        model.X, model.V = shared["X"], shared["V"]
        _WORKER_MODEL[:] = [token, model]
    ctx = dict(config=config, site_bin=shared["site_bin"], site_alleles=shared["site_alleles"], site_gl=site_gl,
               break_bins=break_bins, scores=scores, chain=True)
    return release_sample(_WORKER_MODEL[1], n, ctx)


def _release_component(model, rates, site_bin, site_alleles, site_gl, break_bins, config, budget, token,
                       scores=False):
    """Every sample's release (in sample order, with its meiosis chains' posteriors: release_sample's "chain") from
    the solved component (site_gl float32 [N, m, 3]: the samples' genotype likelihoods at the released sites).
    Releases are independent given the converged state: one process when the budget has one CPU, else an
    execution.BudgetedPool sharing that state."""
    task = _release_bytes(model, len(site_bin))            # reported to the pool (its memory pacing measures)
    workers = min(model.N, max(1, int(budget.threads())))
    if workers <= 1:
        return [release_sample(model, n, dict(config=config, site_bin=site_bin, site_alleles=site_alleles,
                                              site_gl=site_gl[n], break_bins=break_bins[n], scores=scores,
                                              chain=True))
                for n in range(model.N)]
    shared = dict(E=model.E, X=model.X, V=model.V, par=model.par, theta_l=rates[0], theta_m=rates[1],
                  alpha=rates[2], beta=rates[3], site_bin=site_bin, site_alleles=site_alleles)
    tasks = [(n, break_bins[n], site_gl[n], token, config, scores) for n in range(model.N)]
    with BudgetedPool(budget, processes=workers, shared=shared, label="family-phase release") as pool:
        return pool.map(_release_task, tasks, task_bytes=task)


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


def phase_chromosome(file, contig, edges, config, chromosome_map, budget=None, scores=False):
    """FamilyPhaseProduct of one chromosome (``budget``: the CPUs of its releases; None = this process's
    affinity). With ``scores``, also float32 [N, L, 2]: each released site's genotype score and ordered allele score
    (_agreement; -1 where nothing is released), for reporting how the scores behave."""
    started = time.perf_counter()
    budget = CpuBudget.fixed(affinity_cpus()) if budget is None else budget
    loaded = evidence_records.load_arrays(file, ("positions", "likelihoods"), nthreads=claim_threads())
    positions = np.asarray(loaded["positions"], dtype=np.int64)
    likelihoods = np.asarray(loaded["likelihoods"], dtype=np.float32)    # [N, L, 3] P(reads | genotype), run order
    del loaded
    sample_ids = tuple(edges.sample_ids)
    N, L = len(sample_ids), len(positions)
    if likelihoods.shape[:2] != (N, L):
        raise ValueError(f"{contig}: the evidence's likelihoods do not match the pedigree's samples and sites")
    par = np.full((N, 2), -1, dtype=np.int64)
    for p, c, k in zip(edges.parents, edges.children, edges.slots):
        par[int(c), int(k)] = int(p)
    chromosome_painting = painting_records.load(file, panels=False, emissions=True)
    if tuple(chromosome_painting.sample_ids) != sample_ids:
        raise ValueError(f"{contig}: the painting's samples differ from the pedigree's")
    painting = chromosome_painting.components
    panels = refit_records.founder_panels(file)
    ties = tie_boundaries(painting, positions, np.arange(N))
    cover, cover_offsets = coverage(painting, positions, N)
    calls = np.full((N, L, 2), -1, dtype=np.int8)
    undetermined = np.zeros((N, L), dtype=np.int8)
    site_scores = np.full((N, L, 2), -1.0, dtype=np.float32) if scores else None
    component_ids = np.full(L, -1, dtype=np.int32)
    breaks = [[] for _ in range(N)]
    all_edges, sweeps, settled, objective, candidates, mirrored_solves, chain_parts = [], [], [], [], [], [], []
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
            continue
        E = _component_emissions(painted, chromosome_painting, file, positions, N)
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
        local = positions[lo:hi]
        cols = np.searchsorted(source, local)
        valid = cols < len(source)
        valid[valid] &= source[cols[valid]] == local[valid]
        bins = np.searchsorted(bin_edges, local, side="right") - 1
        inside = valid & (bins >= 0) & (bins < B)
        site_idx = np.flatnonzero(inside)
        site_bin = bins[site_idx]
        site_alleles = _state_alleles(painted.states, np.asarray(panel.inference, dtype=np.int8), S)[:, cols[site_idx]]
        break_bins = []
        for n in range(N):
            tie_bins = {int(np.searchsorted(bin_edges, positions[m + 1], side="right") - 1)
                        for m in ties.get(n, ()) if lo <= m < hi - 1}
            # painting ties matter only to a sample with no linked parent (a linked homologue is ordered by origin)
            tie_bins = {b for b in tie_bins if 0 < b < B} if np.all(par[n] < 0) else set()
            joins = set(withheld.get(n, ()))
            tie_breaks += len(tie_bins)
            withheld_joins += len(joins - tie_bins)
            break_bins.append(sorted(tie_bins | joins))
        site_gl = np.ascontiguousarray(likelihoods[:, lo + site_idx])
        releases = _release_component(model, rates, site_bin, site_alleles, site_gl, break_bins, config, budget,
                                      f"{contig}:{index}:{id(model)}", scores)
        mirrored = _release_component(mirror, rates, site_bin, site_alleles, site_gl, break_bins, config, budget,
                                      f"{contig}:{index}:{id(mirror)}", scores)
        del site_gl
        starts = []
        for n, (first, second) in enumerate(zip(releases, mirrored)):
            agreed = _agreement(first, second, site_bin)
            sample_calls, sample_undetermined, set_starts = agreed[:3]
            if site_scores is not None:
                site_scores[n, lo + site_idx] = agreed[3].T
            calls[n, lo + site_idx] = sample_calls
            undetermined[n, lo + site_idx] = np.where(sample_undetermined, 3, 0)
            breaks[n].extend(lo + int(site_idx[i]) - 1 for i in set_starts)
            starts.append(_set_start_bins(set_starts, site_bin, B))
        chain_parts.append((index, _edge_chains(model, mirror, [r[3]["chain"] for r in releases],
                                                [r[3]["chain"] for r in mirrored], rates[1], starts,
                                                config.phase_posterior)))
        del model, mirror, releases, mirrored
    het = (calls[..., 0] >= 0) & (calls[..., 1] >= 0) & (calls[..., 0] != calls[..., 1])
    unphased = het & (undetermined > 0)
    phase_breaks = np.asarray([b for sample in breaks for b in sorted(set(sample))], dtype=np.int64)
    break_offsets = np.r_[0, np.cumsum([len(set(sample)) for sample in breaks])].astype(np.int64)
    sets = _phase_set_count(het & ~unphased, component_ids, breaks)
    diagnostics = FamilyPhaseDiagnostics(
        sweeps=tuple(sweeps), settled=tuple(settled), objective=tuple(objective),
        junction_candidates=tuple(candidates), withheld_joins=withheld_joins, tie_breaks=tie_breaks,
        called_alleles=int(np.sum(calls >= 0)), phased_heterozygotes=int(np.sum(het & ~unphased)),
        unphased_heterozygotes=int(np.sum(unphased)), phase_sets=sets,
        elapsed_seconds=time.perf_counter() - started, extra=dict(mirrored_solves=mirrored_solves))
    edge_offsets = np.r_[0, np.cumsum([len(e) for e in all_edges])].astype(np.int64)
    chain = _chain_arrays(edges, all_edges, chain_parts)
    zeros = np.zeros((N, L), dtype=np.int8)
    product = FamilyPhaseProduct(
        contig=contig, sample_ids=sample_ids, positions=positions, component_ids=component_ids,
        bin_edges=np.concatenate(all_edges).astype(np.int64) if all_edges else np.zeros(0, dtype=np.int64),
        bin_edge_offsets=edge_offsets,
        bin_centers=np.concatenate(all_centers).astype(np.float64) if all_centers else np.zeros(0),
        calls=calls, displayed_frame=zeros, decoder_frame=zeros.copy(),
        coverage=cover, coverage_offsets=cover_offsets, edges_identity=edges.identity, diagnostics=diagnostics,
        frame_undetermined=undetermined, phase_breaks=phase_breaks, phase_break_offsets=break_offsets, **chain)
    return (product, site_scores) if scores else product


def _chain_arrays(edges, all_edges, chain_parts):
    """The product's meiosis-chain arrays, one row per pedigree edge (edges order) over the chromosome's painting
    bins (components concatenated in bin_edges order): chain_source int8 (-1 = no released source), chain_support
    float32 and chain_reset bool (True at each component's first bin and at every reset)."""
    rows = {(int(p), int(c), int(k)): i for i, (p, c, k) in enumerate(zip(edges.parents, edges.children, edges.slots))}
    sizes = [len(e) - 1 for e in all_edges]
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    total = int(offsets[-1])
    source = np.full((len(rows), total), -1, dtype=np.int8)
    support = np.zeros((len(rows), total), dtype=np.float32)
    reset = np.zeros((len(rows), total), dtype=bool)
    reset[:, offsets[:-1][np.asarray(sizes) > 0]] = True
    for index, part in chain_parts:
        first = int(offsets[index])
        for key, (s, w, r) in part.items():
            i = rows[key]
            stop = first + len(s)
            source[i, first:stop], support[i, first:stop], reset[i, first:stop] = s, w, r
    return dict(chain_source=source, chain_support=support, chain_reset=reset)
