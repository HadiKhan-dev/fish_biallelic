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
from . import emissions, fidelity
from .parameters import EMISSION_CACHE_MAX_BYTES
from .label_continuity import relabel_component
from .run import component_model, decoding_grid, decoding_model, _raw_batches


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


def outlier_unit(panels, evidence, config, genetic_map, epsilon, *, batch_samples, emission_builder):
    """The last recorded likelihood sum of the copying model at reference resolution and weight one, at one outlier
    rate, summed over components. The rates of the grid are independent: the stage runs each as its own one-CPU
    task; outlier_scores computes them in sequence, with the same values."""
    reference = replace(config, snps_per_bin=config.reference_snps_per_bin)
    scores = []
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
        )
        scores.append(0.0 if model is None or not model.trajectory else model.trajectory[-1][3])
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


def predictive_score(mass, probabilities, values, observed, epsilon):
    """Total proper log predictive density and observed-cell count for one hidden bin.

    mass is [N,S,S], including independently background-mapped error tracks;
    probabilities is [S,L] and values [N,L,3] are normalized here as in site emissions.
    """
    total = values.sum(axis=2, keepdims=True)
    gl = np.full(values.shape, 1.0 / 3.0)
    np.divide(values, total, out=gl, where=total > 0)
    p2 = np.einsum("nij,il,jl->nl", mass, probabilities, probabilities, optimize=True)
    ref = 1.0 - probabilities
    p0 = np.einsum("nij,il,jl->nl", mass, ref, ref, optimize=True)
    p1 = np.maximum(0.0, mass.sum(axis=(1, 2))[:, None] - p0 - p2)
    prediction = (1.0 - epsilon) * (p0 * gl[:, :, 0] + p1 * gl[:, :, 1] + p2 * gl[:, :, 2]) + epsilon / 3.0
    return float(np.log(prediction[observed]).sum()), int(np.count_nonzero(observed))


def path_score(paths, probabilities, values, observed, epsilon):
    """Total held-out log score of the decoded label pair for one hidden bin: the model's weight-one site emission at
    the pair. paths is int[N, 2] (label states of the bin), probabilities [S, L], values [N, L, 3]."""
    total = values.sum(axis=2, keepdims=True)
    gl = np.full(values.shape, 1.0 / 3.0)
    np.divide(values, total, out=gl, where=total > 0)
    first = probabilities[paths[:, 0]]
    second = probabilities[paths[:, 1]]
    p2 = first * second
    p0 = (1.0 - first) * (1.0 - second)
    p1 = np.maximum(0.0, 1.0 - p0 - p2)
    prediction = (1.0 - epsilon) * (p0 * gl[:, :, 0] + p1 * gl[:, :, 1] + p2 * gl[:, :, 2]) + epsilon / 3.0
    return float(np.log(prediction[observed]).sum())


SCORE_CHUNK_BYTES = 64 << 20  # working set of one scoring chunk (the gathered pair masses of its sites)


def hidden_scores(mass, paths, site_bin, probabilities, gl, observed, epsilon):
    """(predictive log density [N], path score [N], observed cells) of all hidden sites of one sample batch: one
    score per sample.

    The same quantities as predictive_score and path_score summed over the hidden parts of every window, computed
    over the sites in chunks instead of window by window: mass [N, S*S, B] emission-label pair masses, paths
    int[N, 2, B] decoded label states, site_bin int[H] the bin of each hidden site, probabilities [S, H] ALT
    probabilities of the labels at those sites, gl [N, H, 3] normalised genotype likelihoods, observed bool[N, H].

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
        prediction = (1.0 - epsilon) * (p0 * g0 + p1 * g1 + p2 * g2) + epsilon / 3.0
        np.log(prediction, out=predictive[:, low:high], where=seen)
        columns = np.arange(low, high)[None, :]
        first = probabilities[paths[:, 0, bins], columns]
        second = probabilities[paths[:, 1, bins], columns]
        p2 = first * second
        p0 = (1.0 - first) * (1.0 - second)
        p1 = np.maximum(0.0, 1.0 - p0 - p2)
        prediction = (1.0 - epsilon) * (p0 * g0 + p1 * g1 + p2 * g2) + epsilon / 3.0
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
        total = values.sum(axis=2, keepdims=True)
        gl = np.full(values.shape, 1.0 / 3.0)
        np.divide(values, total, out=gl, where=total > 0)
        held[start] = (gl, seen)
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
        gl, seen = held[start]
        score, path, count = hidden_scores(masses, paths, site_bin, probabilities, gl, seen, epsilon)
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


def divisor_scores(panels, evidence, config, genetic_map, epsilon, *, batch_samples, reverse, emission_builder):
    """(divisor, fold, predictive log density, observed cells, path score), summed over components: every unit of
    divisor_unit, in sequence in one process.

    For each component and fold the evidence is read once (_fold_view): the emissions are built at weight one and
    served to every candidate divisor (emissions.SharedEmissions: the same values as a build per divisor), and the
    hidden sites' genotype likelihoods are gathered once per sample batch and scored for all windows together
    (hidden_scores). The cost per divisor is then the model fit, the label-continuity search and one
    forward-backward and decode (_divisor_unit)."""
    grid = sorted(set(config.divisor_grid))
    if len(grid) == 1:
        return ()
    reference = replace(config, snps_per_bin=config.reference_snps_per_bin)
    folds = range(config.calibration_folds)
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
