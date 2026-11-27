"""Run calibration statistics: weight-one likelihood, and for every candidate evidence divisor two held-out scores of
the hidden part of every reference window (hidden_part): the posterior predictive density (the confidence objective: how well the chain's
probabilities predict the hidden reads) and the path score (the path objective: how well the decoded label pair
predicts them). No release decision enters either. Each error track emits as BACKGROUND independently, as in the
copying model's own site likelihood.
"""

from dataclasses import dataclass, replace
from math import fsum
import numpy as np
from ...contracts.observations import gather_observed
from ...core.errors import InvariantError
from ...core.numerics import ordered_sum
from ..shared.material import haldane_coordinates, partition_coordinates
from ..shared.observation import refined_maximum
from ..shared.recombination import haldane_fraction
from . import emissions, fidelity
from .parameters import EMISSION_CACHE_MAX_BYTES
from .label_continuity import relabel_component
from .run import component_model, decoding_grid, decoding_model, _raw_batches
from .states import build_bins, build_states
from .transitions import edge_morgans


@dataclass(frozen=True)
class HiddenObserved:
    source: object
    hidden: np.ndarray

    @property
    def shape(self):
        return self.source.shape

    def gather(self, samples, columns):
        flags = gather_observed(self.source, columns, samples).copy()
        flags[:, self.hidden[columns]] = False
        return flags


def hidden_part(bins, b, fold, folds):
    """[low, high) of the selected sites of reference window b hidden in ``fold``: the fold-th of ``folds`` contiguous
    near-equal parts of the window. Every window keeps the rest of its own sites, so the posterior that is scored is
    the one the painting uses (a window's state given its own data): the evidence divisor corrects for dependence
    among the sites within a window, which only a prediction of part of a window from the rest of it can test. Over
    the folds every site is hidden exactly once."""
    start = int(bins.starts[b])
    width = int(bins.stops[b]) - start
    return start + fold * width // folds, start + (fold + 1) * width // folds


def correlation(divisor, reference_sites):
    return (float(divisor) - 1.0) / (int(reference_sites) - 1)


def _cache_fits(panel, observed, config):
    states, bins = decoding_grid(panel, config)
    return (
        observed.shape[0] * states.n_states * (states.n_states + 1) // 2 * bins.n_bins * 8 <= EMISSION_CACHE_MAX_BYTES
    )


def component_coordinates(panel, config, genetic_map, switch_clip) -> np.ndarray:
    """float64 [B]: the working-map coordinate (Morgans) of each of the component's B bins at ``config``'s resolution,
    from its first bin, built as the pedigree builds its material axis (science.pedigree.prepare and blocks): each
    edge's meiotic switch probability is the Haldane fraction of the working-map distance between the bin centres
    (the distances the copying model reads, transitions.edge_morgans), and the coordinate is the cumulative Haldane
    distance of those probabilities clipped at ``switch_clip`` (the run's MaterialParameters.switch_clip;
    science.shared.material.haldane_coordinates): the cumulative working-map distance, except at the clip. The
    pedigree's selector floor keeps a selector's switch cost finite and is no part of a coordinate."""
    states = build_states(panel.positions, panel.inference, panel.kept)
    bins = build_bins(states.positions, config.snps_per_bin, config.reference_snps_per_bin)
    return haldane_coordinates(haldane_fraction(edge_morgans(bins.centers, genetic_map)), switch_clip)


def outlier_unit(
    panels, evidence, config, genetic_map, epsilon, *, batch_samples, emission_builder, per_sample=False,
    switch_clip=None,
):
    """The log-likelihood of the copying model at its returned fit (reference resolution, weight one), at one outlier
    rate, summed over components. The rates of the grid are independent: the stage runs each as its own task, whose
    kernels take the task's CPU share; outlier_scores computes them in sequence, with the same values.

    With ``per_sample`` (the stage's fits) also each sample's summed terms, the returned fits' regime counts and, per
    informative component (a fit with at least one eligible sample: the components whose likelihood enters the sum,
    in panel order), its bins' working-map coordinates (component_coordinates, clipped at ``switch_clip``) and its
    per-bin likelihood terms summed over the samples (ChainModel.fitted_bin_log_likelihood): the material units'
    curves of the outlier profile (outlier_profile, ADVICE2 11.2)."""
    reference = replace(config, snps_per_bin=config.reference_snps_per_bin)
    scores = []
    samples = np.zeros(evidence.likelihoods.shape[0]) if per_sample else None
    regime = np.zeros(3)  # the returned fits' label changes, anchored faithful-state bins and anchored error-state bins
    curves = []  # (bin coordinates, per-bin terms) of each informative component
    for panel in panels:
        model = component_model(
            panel,
            evidence.likelihoods,
            evidence.observed,
            panel.site_indices(),
            reference,
            genetic_map,
            robustness_epsilon=epsilon,
            site_correlation=0.0,
            batch_samples=batch_samples,
            cache_emissions=_cache_fits(panel, evidence.observed, reference),
            emission_builder=emission_builder,
            final_likelihood=True,
        )
        # the likelihood at the returned fit, one pass after its last update (ADVICE2 14.1 item 6)
        scores.append(0.0 if model is None or not model.trajectory else model.fitted_log_likelihood)
        if per_sample and model is not None and model.trajectory:
            samples += model.fitted_sample_log_likelihood  # every component spans the whole sample axis
            regime += np.asarray(model.fitted_counts, dtype=np.float64)
            coordinates = component_coordinates(panel, reference, genetic_map, switch_clip)
            terms = np.asarray(model.fitted_bin_log_likelihood, dtype=np.float64)
            if len(coordinates) != len(terms):
                raise InvariantError("a calibration fit's bins differ from its component's coordinates")
            curves.append((coordinates, terms))
    # with ``per_sample`` also each sample's summed terms (A11's per-sample rates, ADVICE2 14.1 item 9), the returned
    # fits' regime counts (ADVICE2 11.2: per-rate switch counts and error-track occupancy) and the informative
    # components' per-bin curves (the material units of outlier_profile)
    if per_sample:
        return fsum(scores), samples, tuple(float(v) for v in regime), tuple(curves)
    return fsum(scores)


def outlier_scores(panels, evidence, config, genetic_map, *, batch_samples, emission_builder):
    """Per-epsilon last recorded likelihood sums at reference resolution and weight one (outlier_unit of every rate
    of the grid, in sequence in one process)."""
    grid = sorted(set(config.robustness_grid))
    if len(grid) == 1:
        return ()
    return tuple(
        (
            float(epsilon),
            outlier_unit(
                panels, evidence, config, genetic_map, epsilon,
                batch_samples=batch_samples, emission_builder=emission_builder,
            ),
        )
        for epsilon in grid
    )


PLAIN_CUTOFF = 3.84
"""Chi-square(1) 95 % point: the plain support set of the outlier profile is {eps : 2 [l(best) - l(eps)] <= 3.84}."""


def sample_rates(grid, sample_values):
    """A11 (ADVICE2 14.1 item 9, 23): each sample's own point estimate of the outlier rate: the rate of the grid (with
    the midpoints) maximising its log-likelihood at the returned fit summed over contigs (``sample_values`` float64
    [N, G]; ties to the smaller rate), refined between that maximum's grid neighbours by the three-point parabola in
    log e (science.shared.observation.refined_maximum: clamped to the neighbours, concave only; the grid value at a
    bound), and whether it sits at the grid's top (unresolved: the sample's rate may be larger). A working exclusion
    score's input, not an upper limit."""
    grid = np.asarray(grid, dtype=np.float64)
    values = np.asarray(sample_values, dtype=np.float64).reshape(-1, len(grid))
    best = np.argmax(values, axis=1)  # first maximum: ties to the smaller rate
    rates = np.array([refined_maximum(grid, row, int(j)) for row, j in zip(values, best)], dtype=np.float64)
    return rates, best == len(grid) - 1


def midpoints(grid, values) -> tuple[float, ...]:
    """The two rates at the geometric midpoints around the grid maximum of the pooled ``values`` (ties to the smaller
    rate; one at a boundary maximum): the profile's refinement (ADVICE2 14.1 item 6)."""
    grid = np.asarray(grid, dtype=np.float64)
    j = int(np.argmax(np.asarray(values, dtype=np.float64)))
    return tuple(float(np.sqrt(grid[k] * grid[k + 1])) for k in (j - 1, j) if 0 <= k and k + 1 < len(grid))


def _contiguous(members) -> bool:
    members = np.flatnonzero(members)
    return len(members) > 0 and int(members[-1] - members[0]) + 1 == len(members)


def material_units(components, unit_morgans, tail_morgans):
    """The pedigree's equal-material units over the calibration's informative components (ADVICE2 11.2; owner, 16
    Nov: map-length units as clusters, as the pedigree does, no contig minimum). ``components``: (coordinates
    float64 [B], curves float64 [G, B]) of every informative component, the analysed contigs in configured order and
    each contig's components in map order; coordinates from the component's first bin (component_coordinates), curves
    each candidate rate's per-bin likelihood terms summed over the samples. As science.pedigree.regimes and blocks
    build the genome axis, each component starts at the running sum of the working Morgans before it (gaps between
    components and contigs add nothing), and science.shared.material.partition_coordinates cuts the axis at every
    multiple of ``unit_morgans`` (the run's MaterialParameters): a bin exactly at a cut belongs to the unit on its
    right, a final piece shorter than ``tail_morgans`` joins the unit before it, empty units are omitted, and units
    straddle components and contigs. No chain restarts at a cut: a bin's term is conditional on the earlier bins of
    its component, so a unit's sum is its stretch's share of the likelihood and the units' sums add up to the
    pooled likelihood (to rounding). Returns (values float64 [U, G]: each occupied unit's summed terms, correctly
    rounded; lengths: each unit's working Morgans; morgans: the axis length, the run's informative material)."""
    if not components:
        return np.zeros((0, 0)), (), 0.0
    axis, curves, offset = [], [], 0.0
    for coordinates, curve in components:
        axis.append(offset + np.asarray(coordinates, dtype=np.float64))
        curves.append(np.asarray(curve, dtype=np.float64))
        offset = float(axis[-1][-1])
    axis = np.concatenate(axis)
    curves = np.concatenate(curves, axis=1)
    _, units = partition_coordinates(axis, unit_morgans, tail_morgans)
    values = np.stack([ordered_sum(curves[:, first:stop], axis=1) for first, stop, _, _ in units])
    return values, tuple(float(length) for _, _, length, _ in units), offset


def outlier_profile(grid, contig_values, unit_values, morgans, assessable_morgans) -> dict:
    """The support curve of the eight-update fit over the outlier rates (ADVICE2 11.2, 14.1 item 6, 15.3 item 19):
    ``grid`` the ascending rates (the literal grid with the two midpoints around its maximum), ``contig_values``
    float64 [C, G] each contig's summed log-likelihood at the returned fit at every rate, ``unit_values`` float64
    [U, G] the same likelihood split into the pedigree's equal-material units (material_units: the clusters) and
    ``morgans`` their working Morgans, the run's informative material. The estimate is the maximum of the contigs'
    sum (ties to the smaller rate), with its boundary state and the plain support set (cutoff 3.84); none of them
    reads the units. With theta = log eps, at the refined maximum j and its neighbours (unequal gaps
    h1 = theta_j - theta_{j-1}, h2 = theta_{j+1} - theta_j):

        g_u = [h1^2 l_u(j+1) - h2^2 l_u(j-1) + (h2^2 - h1^2) l_u(j)] / (h1 h2 (h1 + h2))    unit score
        H   = -2 [(l(j+1) - l(j)) / h2 - (l(j) - l(j-1)) / h1] / (h1 + h2)                  curvature of the sum
        J   = U / (U - 1) sum_u (g_u - mean g)^2,     d = max(1, J / H)

    and the inflated support set uses the cutoff 3.84 d. A unit's terms are conditional on the earlier bins of their
    components (forward-recursion increments). At fixed true parameters, conditional score increments are martingale
    differences under the model; here J / H is a working dispersion adjustment from finite differences of the
    nuisance-refitted eight-update support curve: independent unit scores and coverage are not established (Sol C10
    review 1). It is a support set, never an interval: no coverage is claimed,
    and no consumer receives an uncertainty (the run uses the estimate). The uncertainty is unresolved (the plain set
    then descriptive only) below ``assessable_morgans`` (the pedigree's MaterialParameters.assessable_morgans) working
    Morgans of informative material, with one occupied unit, a maximum at the grid boundary, a non-positive
    curvature, two competing modes (a support set that is not contiguous on the grid), or an inflated set that fills
    the grid or touches a bound. No count of chromosomes, contigs or components enters it. A plain set covering the
    whole grid is "not identified"."""
    grid = np.asarray(grid, dtype=np.float64)
    values = np.asarray(contig_values, dtype=np.float64).reshape(-1, len(grid))
    total = values.sum(axis=0)
    j = int(np.argmax(total))  # first maximum: ties to the smaller rate
    best = float(total[j])
    in_plain = 2.0 * (best - total) <= PLAIN_CUTOFF
    plain = tuple(float(e) for e in grid[in_plain])
    boundary = "lower" if j == 0 else "upper" if j == len(grid) - 1 else "none"
    units = np.asarray(unit_values, dtype=np.float64).reshape(-1, len(grid))
    clusters = units.shape[0]
    out = dict(
        estimate=float(grid[j]), boundary=boundary, plain_set=plain, identified=len(plain) < len(grid),
        clusters=int(clusters), working_morgans=float(morgans), inflation=None, adjusted_set=None,
        uncertainty="unresolved", name="support curve of the eight-update fit",
    )
    if boundary != "none":
        out["reason"] = "maximum at the grid boundary"
        return out
    if not morgans >= assessable_morgans:
        out["reason"] = f"below {assessable_morgans:g} working Morgans of informative material"
        return out
    if clusters < 2:
        out["reason"] = "one occupied material unit"
        return out
    if not _contiguous(in_plain):
        out["reason"] = "two competing modes (the plain support set is not contiguous)"
        return out
    theta = np.log(grid)
    h1, h2 = theta[j] - theta[j - 1], theta[j + 1] - theta[j]
    curvature = -2.0 * ((total[j + 1] - total[j]) / h2 - (total[j] - total[j - 1]) / h1) / (h1 + h2)
    if not curvature > 0.0:
        out["reason"] = "no positive curvature at the maximum"
        return out
    scores = (h1 * h1 * units[:, j + 1] - h2 * h2 * units[:, j - 1] + (h2 * h2 - h1 * h1) * units[:, j]) / (
        h1 * h2 * (h1 + h2)
    )
    information = clusters / (clusters - 1) * float(((scores - scores.mean()) ** 2).sum())
    inflation = max(1.0, information / curvature)
    in_adjusted = 2.0 * (best - total) <= PLAIN_CUTOFF * inflation
    out.update(
        inflation=float(inflation), curvature=float(curvature), score_variance=information,
        adjusted_set=tuple(float(e) for e in grid[in_adjusted]),
    )
    if in_adjusted.all() or in_adjusted[0] or in_adjusted[-1]:
        out["reason"] = "the inflated support set fills the grid or touches a bound"
    elif not _contiguous(in_adjusted):
        out["reason"] = "two competing modes (the inflated support set is not contiguous)"
    else:
        out["uncertainty"] = "estimated"
    return out


def _normalised(values, cohort):
    """(gl, B): the cells' genotype likelihoods normalised to sum one (1/3 where they sum to zero) and their cohort
    likelihoods B = q . gl, ``cohort`` float64[L, 3] the sites' cohort genotype frequencies (the outlier law)."""
    total = values.sum(axis=2, keepdims=True)
    gl = np.full(values.shape, 1.0 / 3.0)
    np.divide(values, total, out=gl, where=total > 0)
    return gl, np.einsum("nlg,lg->nl", gl, np.asarray(cohort, dtype=np.float64))


def predictive_score(mass, probabilities, values, observed, epsilon, cohort):
    """Total proper log predictive density and observed-cell count for one hidden bin.

    mass is [N,S,S], including independently background-mapped error tracks;
    probabilities is [S,L] and values [N,L,3] are normalized here as in site emissions; cohort [L,3] the sites' cohort
    genotype frequencies: a cell is redrawn from them at rate epsilon (science.shared.observation).
    """
    gl, background = _normalised(values, cohort)
    p2 = np.einsum("nij,il,jl->nl", mass, probabilities, probabilities, optimize=True)
    ref = 1.0 - probabilities
    p0 = np.einsum("nij,il,jl->nl", mass, ref, ref, optimize=True)
    p1 = np.maximum(0.0, mass.sum(axis=(1, 2))[:, None] - p0 - p2)
    prediction = (1.0 - epsilon) * (p0 * gl[:, :, 0] + p1 * gl[:, :, 1] + p2 * gl[:, :, 2]) + epsilon * background
    return float(np.log(prediction[observed]).sum()), int(np.count_nonzero(observed))


def path_score(paths, probabilities, values, observed, epsilon, cohort):
    """Total held-out log score of the decoded label pair for one hidden bin: the model's weight-one site emission at
    the pair. paths is int[N, 2] (label states of the bin), probabilities [S, L], values [N, L, 3], cohort [L, 3]."""
    gl, background = _normalised(values, cohort)
    first = probabilities[paths[:, 0]]
    second = probabilities[paths[:, 1]]
    p2 = first * second
    p0 = (1.0 - first) * (1.0 - second)
    p1 = np.maximum(0.0, 1.0 - p0 - p2)
    prediction = (1.0 - epsilon) * (p0 * gl[:, :, 0] + p1 * gl[:, :, 1] + p2 * gl[:, :, 2]) + epsilon * background
    return float(np.log(prediction[observed]).sum())


SCORE_CHUNK_BYTES = 64 << 20  # working set of one scoring chunk (the gathered pair masses of its sites)


def hidden_scores(mass, paths, site_bin, probabilities, gl, observed, epsilon, background):
    """(predictive log density [N], path score [N], observed cells) of all hidden sites of one sample batch: one
    score per sample.

    The same quantities as predictive_score and path_score summed over the hidden parts of every window, computed
    over the sites in chunks instead of window by window: mass [N, S*S, B] emission-label pair masses, paths
    int[N, 2, B] decoded label states, site_bin int[H] the bin of each hidden site, probabilities [S, H] ALT
    probabilities of the labels at those sites, gl [N, H, 3] normalised genotype likelihoods, observed bool[N, H],
    background [N, H] their cohort likelihoods B = q . gl (a cell is redrawn from the cohort at rate epsilon).

    A sample's scores are the same floating-point numbers whatever batch it is scored in and however the sites are
    chunked (the batch follows the run's core count, and a run must publish the same record on any core count):
    every cell's label-pair sum is accumulated pair by pair in pair order, and a sample's score is one sum over its
    own row of all hidden sites. The caller adds the samples' scores exactly (fsum)."""
    samples, pairs, _ = mass.shape
    states = probabilities.shape[0]
    sites = len(site_bin)
    chunk = max(1, SCORE_CHUNK_BYTES // (8 * samples * pairs))
    reference = 1.0 - probabilities
    predictive = np.zeros((samples, sites))
    path = np.zeros((samples, sites))
    for low in range(0, sites, chunk):
        high = min(sites, low + chunk)
        bins = site_bin[low:high]
        alt, ref = probabilities[:, low:high], reference[:, low:high]
        seen = observed[:, low:high]
        g0, g1, g2 = gl[:, low:high, 0], gl[:, low:high, 1], gl[:, low:high, 2]
        redraw = epsilon * background[:, low:high]
        m = mass[:, :, bins]
        total = np.zeros((samples, high - low))
        p0 = np.zeros((samples, high - low))
        p2 = np.zeros((samples, high - low))
        for pair in range(pairs):
            first, second = divmod(pair, states)
            weight = m[:, pair, :]
            total += weight
            p2 += weight * (alt[first] * alt[second])
            p0 += weight * (ref[first] * ref[second])
        del m
        p1 = np.maximum(0.0, total - p0 - p2)
        prediction = (1.0 - epsilon) * (p0 * g0 + p1 * g1 + p2 * g2) + redraw
        np.log(prediction, out=predictive[:, low:high], where=seen)
        columns = np.arange(low, high)[None, :]
        first = probabilities[paths[:, 0, bins], columns]
        second = probabilities[paths[:, 1, bins], columns]
        p2 = first * second
        p0 = (1.0 - first) * (1.0 - second)
        p1 = np.maximum(0.0, 1.0 - p0 - p2)
        prediction = (1.0 - epsilon) * (p0 * g0 + p1 * g1 + p2 * g2) + redraw
        np.log(prediction, out=path[:, low:high], where=seen)
    return predictive.sum(axis=1), path.sum(axis=1), int(np.count_nonzero(observed))


def _fold_view(states, bins, sites, evidence, fold, folds, batch_samples, emission_builder):
    """One fold's view of a component, read once for every divisor: the observed flags with the fold's part of every
    window hidden, the shared weight-one evidence build, the hidden sites (selected-site indices, in order) with
    their bins and evidence columns and, per sample batch, their normalised genotype likelihoods and original
    observed flags."""
    parts = [hidden_part(bins, b, fold, folds) for b in range(bins.n_bins)]
    hidden_sites = np.concatenate([np.arange(low, high, dtype=np.int64) for low, high in parts])
    site_bin = np.repeat(np.arange(bins.n_bins, dtype=np.int64), [high - low for low, high in parts])
    hidden_columns = sites[states.site_indices][hidden_sites]
    hidden = np.zeros(evidence.observed.shape[1], dtype=np.bool_)
    hidden[hidden_columns] = True
    held = {}
    n_samples = evidence.observed.shape[0]
    for start in range(0, n_samples, batch_samples):
        stop = min(start + batch_samples, n_samples)
        values, seen = emissions.gather_sites(
            evidence.likelihoods, evidence.observed, hidden_columns, slice(start, stop)
        )
        gl, background = _normalised(values, evidence.cohort_frequencies[hidden_columns])
        held[start] = (gl, seen, background)
        del values
    return (
        HiddenObserved(evidence.observed, hidden),
        emissions.SharedEmissions(emission_builder, 1, EMISSION_CACHE_MAX_BYTES),
        hidden_sites,
        site_bin,
        hidden_columns,
        held,
    )


def _divisor_unit(panel, evidence, sites, reference, genetic_map, epsilon, batch_samples, reverse, view, divisor):
    """(predictive log density, path score, observed cells) of one fold view's hidden sites at one divisor: the
    model fit, the label-continuity search, one forward-backward and decode per sample batch, and the scores."""
    masked, shared, hidden_sites, site_bin, hidden_columns, held = view
    r = correlation(divisor, reference.reference_snps_per_bin)
    model = component_model(
        panel,
        evidence.likelihoods,
        masked,
        sites,
        reference,
        genetic_map,
        robustness_epsilon=epsilon,
        site_correlation=r,
        batch_samples=batch_samples,
        cache_emissions=False,
        emission_builder=shared,
    )
    painted, _ = relabel_component(
        panel,
        evidence.likelihoods,
        masked,
        sites,
        reference,
        genetic_map,
        model=model,
        robustness_epsilon=epsilon,
        site_correlation=r,
        batch_samples=batch_samples,
        reverse=reverse,
        emission_cache_bytes=EMISSION_CACHE_MAX_BYTES,
        emission_builder=shared,
    )
    states, bins = decoding_grid(painted, reference)
    capable, _, local_keys, _, selected = decoding_model(states, bins, sites, genetic_map, model)
    if not np.array_equal(selected[hidden_sites], hidden_columns):
        raise ValueError("label continuity changed the selected sites of a calibration component")
    tracks = fidelity.track_states(capable)
    probabilities = np.vstack((states.alt_probability(), states.background_frequency))[:, hidden_sites]
    scores, path_scores, cells = [], [], 0
    for raw, _, start, stop, _, batch_observed in _raw_batches(
        states, bins, evidence.likelihoods, masked, selected, reference, batch_samples, shared, epsilon, r
    )():
        structured = fidelity.structured_emissions(raw, tracks)
        *_, masses = fidelity.chain_statistics(
            structured,
            tracks,
            capable,
            model.theta,
            model.entry,
            model.exit,
            np.ones(stop - start, dtype=np.bool_),
            emission_masses=True,
        )
        del structured
        paths, *_ = fidelity.decode_candidates(
            raw,
            model,
            capable,
            local_keys,
            states.named_alleles,
            batch_observed,
            bins.starts,
            bins.stops,
            states.n_classes,
            reverse=reverse,
            tie=reference.viterbi_tie,
        )
        gl, seen, background = held[start]
        score, path, count = hidden_scores(masses, paths, site_bin, probabilities, gl, seen, epsilon, background)
        scores.extend(score.tolist())
        path_scores.extend(path.tolist())
        cells += count
    return fsum(scores), fsum(path_scores), cells


def divisor_unit(
    panels, evidence, config, genetic_map, epsilon, fold, divisor, *, batch_samples, reverse, emission_builder
):
    """(predictive log density, observed cells, path score) of one fold at one divisor, summed over components.

    The (fold, divisor) units of a contig are independent of one another: the stage runs each as its own one-CPU
    task, so a contig's calibration uses as many CPUs as it has units instead of one team that idles through every
    one-thread section. divisor_scores computes the same units in sequence, with the same values."""
    reference = replace(config, snps_per_bin=config.reference_snps_per_bin)
    scores, paths, cells = [], [], 0
    for panel in panels:
        states, bins = decoding_grid(panel, reference)
        if not bins.n_bins:
            continue
        sites = panel.site_indices()
        view = _fold_view(
            states, bins, sites, evidence, fold, config.calibration_folds, batch_samples, emission_builder
        )
        score, path, count = _divisor_unit(
            panel, evidence, sites, reference, genetic_map, epsilon, batch_samples, reverse, view, divisor
        )
        scores.append(score)
        paths.append(path)
        cells += count
    return fsum(scores), cells, fsum(paths)


def divisor_scores(panels, evidence, config, genetic_map, epsilon, *, batch_samples, reverse, emission_builder,
                   grid=None, folds=None):
    """(divisor, fold, predictive log density, observed cells, path score), summed over components: every unit of
    divisor_unit, in sequence in one process.

    For each component and fold the evidence is read once (_fold_view): the emissions are built at weight one and
    served to every candidate divisor (emissions.SharedEmissions: the same values as a build per divisor), and the
    hidden sites' genotype likelihoods are gathered once per sample batch and scored for all windows together
    (hidden_scores). The cost per divisor is then the model fit, the label-continuity search and one
    forward-backward and decode (_divisor_unit)."""
    if grid is None:
        grid = sorted(set(config.divisor_grid))
        if len(grid) == 1:
            return ()
    reference = replace(config, snps_per_bin=config.reference_snps_per_bin)
    folds = range(config.calibration_folds) if folds is None else folds
    totals = {(divisor, fold): [[], [], 0] for divisor in grid for fold in folds}
    for panel in panels:
        states, bins = decoding_grid(panel, reference)
        if not bins.n_bins:
            continue
        sites = panel.site_indices()
        for fold in folds:
            view = _fold_view(
                states, bins, sites, evidence, fold, config.calibration_folds, batch_samples, emission_builder
            )
            for divisor in grid:
                score, path, count = _divisor_unit(
                    panel, evidence, sites, reference, genetic_map, epsilon, batch_samples, reverse, view, divisor
                )
                entry = totals[(divisor, fold)]
                entry[0].append(score)
                entry[1].append(path)
                entry[2] += count
            del view
    return tuple(
        (float(divisor), fold, fsum(totals[(divisor, fold)][0]), totals[(divisor, fold)][2],
         fsum(totals[(divisor, fold)][1]))
        for divisor in grid
        for fold in folds
    )
