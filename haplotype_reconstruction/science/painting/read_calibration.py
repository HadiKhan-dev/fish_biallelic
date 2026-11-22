"""Read-aware calibration of the painting's evidence strengths (PaintingParameters.read_classes > 1).

PATH strength: fixed at PaintingParameters.path_divisor (path_divisor below; the one place the choice is made). On the
four truth designs measured, path divisor 4 decoded the fewest unmatched switch events at equal misses, and the
within-window path score did not select it.

CONFIDENCE strength: chosen per run from held-out reference-window parts under the read model. For every fold
(calibration.hidden_part: the fold-th contiguous part of every reference window is hidden) and candidate divisor c of
PaintingParameters.confidence_grid, the copying model is fitted (run.component_model) and the labels made continuous
(label_continuity.relabel_component) at r(c) with the read-state builder on the data with the hidden sites' allele
evidence removed (their depths and the class transport through them kept; design weights from the visible
informative sites). Each sample's score is the sum over windows of log P(hidden block | visible data) under the
read-state chain (_block_scores): the hidden block's probability under the normalised hidden law
(read_factor._hidden_matrices: for each class the read model's proper PMF of the ALT count at the observed depth, the
pair's configuration law, the outlier term at the cohort genotype law), contracted between the visible data's forward
and backward messages over (track pair, read class), minus the same contraction without it. The same hidden law serves
every candidate. Selection (paired_one_se): each sample's totals over folds and analysed contigs; the best candidate
has the largest total; the choice is the smallest divisor whose paired difference to the best, summed over samples,
is within one standard error (sqrt(n) x the standard deviation of the per-sample differences).

MATERIAL (material_prefix): the confidence fits run on a declared amount of genetic map,
PaintingParameters.calibration_morgans: the analysed contigs in configured order, whole, accumulated until their
working-map length reaches the budget (every contig when the run has less). The units are Morgans, never contig
counts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ...contracts.observations import gather_observed
from .calibration import HiddenObserved, correlation, hidden_part
from .fidelity import initial_log, structured_emissions, track_states
from .label_continuity import relabel_component
from .parameters import EMISSION_CACHE_MAX_BYTES
from .read_chain import _block_scores, track_pair_maps
from .read_factor import swap_classes
from .read_inputs import design_weights
from .run import _raw_batches, component_model, decoding_grid, decoding_model

__all__ = ["path_divisor", "paired_one_se", "material_prefix", "block_scores", "confidence_unit"]


def path_divisor(config) -> float:
    """The path strength's full-reference-window divisor of the read-state painting: PaintingParameters.path_divisor
    (fixed; the one function to replace when the path choice becomes data-driven)."""
    return float(config.path_divisor)


def paired_one_se(grid, totals):
    """The paired one-standard-error choice among the candidate divisors ``grid`` from each sample's totals
    ``totals`` float64[C, N] (C candidates, N samples). best: the largest summed total (ties to the smaller divisor);
    for each candidate the per-sample differences d_i = totals[c, i] - totals[best, i], their sum and
    SE = sqrt(N) x sd(d, ddof=1); choice: the smallest divisor with sum(d) >= -SE. Deterministic. Returns dict(best,
    choice, sums, differences, standard_errors) with plain floats."""
    grid = [float(c) for c in grid]
    totals = np.asarray(totals, dtype=np.float64)
    sums = [float(np.sum(row)) for row in totals]
    best = min(range(len(grid)), key=lambda k: (-sums[k], grid[k]))
    n = totals.shape[1]
    differences, errors = [], []
    for k in range(len(grid)):
        d = totals[k] - totals[best]
        differences.append(float(np.sum(d)))
        errors.append(float(np.sqrt(n) * np.std(d, ddof=1)) if n > 1 else 0.0)
    eligible = [k for k in range(len(grid)) if differences[k] >= -errors[k]]
    choice = min(eligible, key=lambda k: grid[k])
    return dict(best=grid[best], choice=grid[choice], sums=tuple(sums), differences=tuple(differences),
                standard_errors=tuple(errors))


def material_prefix(lengths, budget):
    """The calibration material: ((name, Morgans), ...) the leading items of ``lengths`` (an iterable of (name,
    working-map Morgans) in configured order), whole, up to and including the first whose running sum reaches
    ``budget`` Morgans; every item when the sum stays below it. Stops reading ``lengths`` there, so a lazy iterable
    computes no length past the material."""
    chosen, total = [], 0.0
    for name, morgans in lengths:
        chosen.append((name, float(morgans)))
        total += float(morgans)
        if total >= budget:
            break
    return tuple(chosen)


def _label_pairs(values, n_labels):
    """[n, X*X, B] label-pair values from per-unordered-pair values [n, U, B] (symmetric)."""
    first, second = np.triu_indices(n_labels)
    out = np.empty((values.shape[0], n_labels * n_labels, values.shape[2]))
    for k, (x, y) in enumerate(zip(first, second, strict=True)):
        out[:, x * n_labels + y] = values[:, k]
        out[:, y * n_labels + x] = values[:, k]
    return out


def block_scores(raw, read_hidden, read_log_hidden, hidden_scalar, tracks, capable, model):
    """float64[n, B]: each bin's log P(hidden block | visible data) of the batch (_block_scores); ``raw`` the batch's
    visible ReadEmissions, the hidden arrays from ReadEmissionBuilder.hidden_matrices, ``model`` its ChainModel."""
    capable = np.ascontiguousarray(capable, dtype=np.int64)
    n_labels = len(capable)
    emission_label = np.where(tracks.fidelity == 1, n_labels - 1, tracks.labels).astype(np.int64)
    unordered, swap = track_pair_maps(tracks.labels, emission_label, n_labels)
    emissions = np.ascontiguousarray(structured_emissions(raw.raw, tracks))
    if hidden_scalar.any():
        scalar = np.ascontiguousarray(structured_emissions(_label_pairs(hidden_scalar, n_labels), tracks))
    else:
        scalar = np.zeros_like(emissions)
    out = np.zeros((emissions.shape[0], emissions.shape[2]))
    _block_scores(emissions, np.ascontiguousarray(raw.read), np.ascontiguousarray(raw.read_log), scalar,
                  np.ascontiguousarray(read_hidden), np.ascontiguousarray(read_log_hidden), unordered, swap,
                  swap_classes(raw.minor), np.ascontiguousarray(raw.initial), tracks.labels, tracks.fidelity,
                  tracks.error_of, capable, initial_log(tracks, capable, model.entry, model.exit),
                  np.array(model.theta, dtype=np.float64), float(model.entry), float(model.exit), out)
    return out


def confidence_unit(panels, evidence, config, genetic_map, *, epsilon, reverse, batch_samples, emission_builder,
                    grid=None, folds=None):
    """((candidate, fold, per-sample scores tuple[N], hidden observed cells), ...) of one contig: every fold and
    candidate divisor (PaintingParameters.confidence_grid unless ``grid``), summed over its components. The
    read-state builder ``emission_builder`` (read_inputs.ReadEmissionBuilder, built without a confidence strength)
    caches the label-independent parts, so every candidate and fold reuses them."""
    builder = emission_builder
    reference = replace(config, snps_per_bin=config.reference_snps_per_bin)
    grid = tuple(sorted({float(c) for c in config.confidence_grid})) if grid is None else tuple(grid)
    folds = tuple(range(config.calibration_folds)) if folds is None else tuple(folds)
    n_samples = evidence.observed.shape[0]
    scores = {(c, f): np.zeros(n_samples) for c in grid for f in folds}
    cells = {(c, f): 0 for c in grid for f in folds}
    for panel in panels:
        states, bins = decoding_grid(panel, reference)
        if not bins.n_bins:
            continue
        sites = panel.site_indices()
        selected = np.asarray(sites, dtype=np.int64)[states.site_indices]
        size = builder.batch_samples(selected, states.n_states, bins.n_bins, False, batch_samples)
        n_packed = states.n_states * (states.n_states + 1) // 2
        hold = (8 * n_samples * n_packed * bins.n_bins
                + n_samples * builder.matrix_bytes(states.n_states, bins.n_bins)) <= EMISSION_CACHE_MAX_BYTES
        for fold in folds:
            parts = [hidden_part(bins, b, fold, config.calibration_folds) for b in range(bins.n_bins)]
            hidden_sites = np.concatenate([np.arange(low, high, dtype=np.int64) for low, high in parts])
            hidden_columns = selected[hidden_sites]
            hidden = np.zeros(evidence.observed.shape[1], dtype=np.bool_)
            hidden[hidden_columns] = True
            masked = HiddenObserved(evidence.observed, hidden)
            unit = np.zeros(len(selected), dtype=np.bool_)
            unit[hidden_sites] = True
            for candidate in grid:
                r = correlation(candidate, reference.reference_snps_per_bin)
                model = component_model(
                    panel, evidence.likelihoods, masked, sites, reference, genetic_map, robustness_epsilon=epsilon,
                    site_correlation=r, batch_samples=size, cache_emissions=hold, emission_builder=builder,
                )
                painted, _ = relabel_component(
                    panel, evidence.likelihoods, masked, sites, reference, genetic_map, model=model,
                    robustness_epsilon=epsilon, site_correlation=r, batch_samples=size, reverse=reverse,
                    emission_cache_bytes=EMISSION_CACHE_MAX_BYTES, emission_builder=builder,
                )
                painted_states, painted_bins = decoding_grid(painted, reference)
                capable, _, _, _, painted_sites = decoding_model(
                    painted_states, painted_bins, sites, genetic_map, model
                )
                if not np.array_equal(painted_sites[hidden_sites], hidden_columns):
                    raise ValueError("label continuity changed the selected sites of a calibration component")
                tracks = track_states(capable)
                for raw, _, start, stop, informative, _ in _raw_batches(
                    painted_states, painted_bins, evidence.likelihoods, masked, painted_sites, reference, size,
                    builder, epsilon, r,
                )():
                    weight = design_weights(informative, painted_bins.starts, r, 1)
                    read_hidden, log_hidden, scalar = builder.hidden_matrices(
                        evidence.likelihoods, evidence.observed, painted_sites, slice(start, stop),
                        painted_states.alt_probability(), painted_states.background_frequency, painted_bins.starts,
                        painted_bins.stops, epsilon, weight, unit,
                    )
                    out = block_scores(raw, read_hidden, log_hidden, scalar, tracks, capable, model)
                    scores[(candidate, fold)][start:stop] += out.sum(axis=1)
                    cells[(candidate, fold)] += int(
                        np.count_nonzero(gather_observed(evidence.observed, hidden_columns, slice(start, stop)))
                    )
    return tuple(
        (candidate, fold, tuple(float(v) for v in scores[(candidate, fold)]), cells[(candidate, fold)])
        for candidate in grid
        for fold in folds
    )
