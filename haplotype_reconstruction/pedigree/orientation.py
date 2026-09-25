"""Finite continuous orientation, reciprocal and short-ancestry family scores.

Finite composite-evidence corrections, not independent biological direction
probabilities. Four synchronous family-message passes exclude immediate reverse
feedback. A joint-family correction checks reverse two-edge paths; its evidence
is then frozen while a final reciprocal solve reconciles the adjusted scores.
The final reciprocal messages converge; the loopy/global model remains approximate.
Genetic scoring and final parent-state priors are unchanged.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit
from . import direction
from .ancestry_paths import joint_path_correction


@njit(cache=True, nogil=True)
def _continuous_probability(junctions, callable_bins, total_bins, multiplicity):
    """Paired chromosome contrast softened by missingness and count variance.

    This is a summary-level MAR approximation, not a common-interval recount.
    Inclusion-exclusion bounds common exposure; absent overlap is neutral.
    """
    chromosomes, samples = junctions.shape
    result = np.full((samples, samples), 0.5)
    for child in range(samples):
        for parent in range(child):
            count = mean_sum = square_sum = missing_variance = 0.0
            for chrom in range(chromosomes):
                if multiplicity[chrom] <= 0 or total_bins[chrom] <= 0:
                    continue
                fc = min(1.0, callable_bins[chrom, child] / (2.0 * total_bins[chrom]))
                fp = min(1.0, callable_bins[chrom, parent] / (2.0 * total_bins[chrom]))
                common = max(0.0, fc + fp - 1.0)
                if common <= 0:
                    continue
                weight = multiplicity[chrom] * common
                dc = junctions[chrom, child] / fc
                dp = junctions[chrom, parent] / fp
                delta = dc - dp
                count += weight
                mean_sum += weight * delta
                square_sum += weight * delta * delta
                missing_variance += weight * ((1.0-fc)*dc/fc + (1.0-fp)*dp/fp)
            if count <= 2.0:
                continue
            variance = max(0.0, square_sum - mean_sum*mean_sum/count)
            variance *= count / (count-1.0)
            variance += missing_variance + 1.0
            probability = 0.5 * (1.0 + math.erf(mean_sum / math.sqrt(2.0*variance)))
            result[child, parent] = probability
            result[parent, child] = 1.0-probability
    return result


def fit_direction(junctions, callable_bins, total_bins, weights, settings,
                  *, component_count=None):
    """Use the selected model in full, bootstrap and LOCO evaluations."""
    if settings.parent_state_direction_model == "cluster":
        return direction._fit_ancestry_depth_model(
            weights @ junctions, weights @ callable_bins, settings.bootstrap_seed,
            component_count=component_count)
    counts = weights @ junctions
    exposure = weights @ callable_bins
    full = float(weights @ (2.0 * total_bins))
    fractions = np.divide(exposure, full, out=np.zeros_like(exposure), where=full > 0)
    adjusted = np.divide(counts, fractions, out=np.full_like(counts, np.nan), where=fractions > 0)
    probabilities = _continuous_probability(junctions, callable_bins, total_bins, weights)
    if settings.parent_state_direction_model == "family":
        probabilities[:] = 0.5
    return direction._AncestryDepthModel(
        adjusted, fractions, np.ones((len(counts), 1)),
        np.asarray([np.nan]), np.asarray([np.nan]), np.asarray([1.0]),
        np.nan, (), probabilities)


@njit(cache=True, nogil=True)
def _family_log_odds(log_scores, alternatives, states, log_multiplicity, samples):
    """Stable included/excluded family log odds in O(A + N**2).

    Included masses have independent scaling for each parent. A tiny included
    probability must survive removal of a strong incoming reverse message.
    Exclusions containing the maximum row can share its scale; at most two
    exclusions per fish require a separate scan.
    """
    output = np.full((samples, samples), -np.inf)
    starts = np.searchsorted(alternatives[:, 0], np.arange(samples+1))
    for child in range(samples):
        start, stop = starts[child], starts[child+1]
        maximum = -np.inf
        best = -1
        included_max = np.full(samples, -np.inf)
        for row in range(start, stop):
            value = log_scores[row] + log_multiplicity[child, states[row]]
            if value > maximum:
                maximum, best = value, row
            for slot in range(1, 3):
                parent = alternatives[row, slot]
                if parent >= 0:
                    included_max[parent] = max(included_max[parent], value)
        if best < 0:
            continue
        mass = np.zeros(samples)
        included_sum = np.zeros(samples)
        total = 0.0
        for row in range(start, stop):
            value = log_scores[row] + log_multiplicity[child, states[row]]
            scaled = math.exp(value - maximum)
            total += scaled
            for slot in range(1, 3):
                parent = alternatives[row, slot]
                if parent >= 0 and np.isfinite(included_max[parent]):
                    mass[parent] += scaled
                    included_sum[parent] += math.exp(value-included_max[parent])
        for parent in range(samples):
            if not np.isfinite(included_max[parent]):
                continue
            if parent == alternatives[best, 1] or parent == alternatives[best, 2]:
                excluded_max = -np.inf
                for row in range(start, stop):
                    if alternatives[row, 1] != parent and alternatives[row, 2] != parent:
                        value = log_scores[row] + log_multiplicity[child, states[row]]
                        excluded_max = max(excluded_max, value)
                excluded_sum = 0.0
                for row in range(start, stop):
                    if alternatives[row, 1] != parent and alternatives[row, 2] != parent:
                        excluded_sum += math.exp(log_scores[row] + log_multiplicity[child, states[row]] - excluded_max)
                log_excluded = excluded_max + math.log(excluded_sum)
            else:
                log_excluded = maximum + math.log(max(1.0, total-mass[parent]))
            output[child, parent] = (
                included_max[parent] + math.log(included_sum[parent]) - log_excluded)
    return output


@njit(cache=True, nogil=True)
def _family_avoidance_one_pass(log_scores, alternatives, states, log_multiplicity, samples):
    """One-step sum-product messages; also the small reference-test seam."""
    odds = _family_log_odds(log_scores, alternatives, states, log_multiplicity, samples)
    return -np.maximum(0.0, odds) - np.log1p(np.exp(-np.abs(odds)))


@njit(cache=True, nogil=True)
def _family_avoidance(log_scores, alternatives, states, log_multiplicity, samples, passes=4):
    """Bounded synchronous cavity-message passes, excluding immediate feedback.

    Loopy BP approximation for pairwise anti-reciprocity factors. Finite pass
    budget, no called-pedigree scaffold, no new genotypes or candidate pairs.
    """
    previous = np.zeros((samples, samples))
    for _ in range(passes):
        adjusted = log_scores.copy()
        for row in range(len(alternatives)):
            child = alternatives[row, 0]
            for slot in range(1, 3):
                parent = alternatives[row, slot]
                if parent >= 0:
                    adjusted[row] += previous[parent, child]
        odds = _family_log_odds(
            adjusted, alternatives, states, log_multiplicity, samples)
        for child in range(samples):
            for parent in range(samples):
                # Remove incoming evidence before converting odds to support.
                odds[child, parent] -= previous[parent, child]
        previous = -np.maximum(0.0, odds) - np.log1p(np.exp(-np.abs(odds)))
    return previous


@njit(cache=True, nogil=True)
def _converged_family_avoidance(scores, alternatives, states, multiplicity,
                               samples, maximum, tolerance, damping=1.0):
    """Same cavity equations, stopped by the undamped fixed-point residual."""
    previous = np.zeros((samples, samples))
    residual = np.inf
    for iteration in range(maximum):
        adjusted = scores.copy()
        for row in range(len(alternatives)):
            child = alternatives[row, 0]
            for slot in (1, 2):
                parent = alternatives[row, slot]
                if parent >= 0:
                    adjusted[row] += previous[parent, child]
        odds = _family_log_odds(adjusted, alternatives, states, multiplicity, samples)
        residual = 0.0
        for child in range(samples):
            for parent in range(samples):
                value = odds[child, parent] - previous[parent, child]
                value = -max(0.0, value) - np.log1p(np.exp(-abs(value)))
                residual = max(residual, abs(value - previous[child, parent]))
                odds[child, parent] = value
        if residual <= tolerance:
            return odds, iteration + 1, residual
        previous = (1.0-damping)*previous + damping*odds
    return previous, maximum, residual


def _family_scores(avoidance, alternatives, eligible, explicit_direction):
    """Sum incoming reciprocal messages, respecting per-side chronology."""
    family = np.zeros(len(alternatives))
    for slot in (1, 2):
        present = (alternatives[:, slot] >= 0) & eligible
        c, p = alternatives[present, 0], alternatives[present, slot]
        values = avoidance[p, c]
        if explicit_direction is not None:
            values = np.where(explicit_direction[c, p], 0.0, values)
        family[present] += values
    return family


def adjust_scores(state_scores, identity_scores, alternatives, states,
                  full_counts, model, settings, explicit_direction=None, *, score_diagnostics=None):
    """Apply finite evidence to both state and identity; exposure remains hard."""
    if score_diagnostics is not None:
        for name in ("direction", "reciprocal_family", "ancestry_paths"):
            score_diagnostics[name] = np.zeros(len(alternatives))
    probability = model.edge_probability
    if probability is None:
        return state_scores, identity_scores
    samples = probability.shape[0]
    eligible = np.isfinite(identity_scores)
    correction = np.zeros(len(alternatives))
    if settings.parent_state_direction_model != "family":
        epsilon = max(settings.parent_state_contamination_probability, np.finfo(float).eps)
        log_direction = np.log(2.0*((1.0-epsilon)*probability + 0.5*epsilon))
        if explicit_direction is not None:
            log_direction = np.where(explicit_direction, 0.0, log_direction)
        for slot in (1, 2):
            present = (alternatives[:, slot] >= 0) & eligible
            correction[present] += log_direction[alternatives[present, 0], alternatives[present, slot]]
    if score_diagnostics is not None:
        score_diagnostics["direction"] = correction.copy()
    state_scores = state_scores + correction
    identity_scores = identity_scores + correction
    if settings.parent_state_direction_model in {"family", "continuous_family"}:
        # Equal state weights in this likelihood-only message. Final configurable
        # M0/M1/M2 priors and their sensitivity analysis remain downstream.
        log_multiplicity = -np.log(np.maximum(1, full_counts))
        avoidance = _family_avoidance(
            identity_scores, alternatives, states, log_multiplicity, samples,
            settings.parent_state_family_message_passes)
        family = _family_scores(
            avoidance, alternatives, eligible, explicit_direction)
        if score_diagnostics is not None:
            score_diagnostics["reciprocal_family"] = family.copy()
        state_scores = state_scores + family
        identity_scores = identity_scores + family
        if settings.parent_state_ancestry_path_budget:
            supported = (np.zeros((samples, samples), dtype=np.bool_)
                         if explicit_direction is None else explicit_direction)
            paths = joint_path_correction(
                identity_scores, alternatives, states, log_multiplicity,
                avoidance, settings.parent_state_ancestry_path_budget, supported,
            )
            # Compute this factor once from the original supporting beliefs.
            # The final reciprocal solve below must not recompute path evidence.
            if score_diagnostics is not None:
                score_diagnostics["ancestry_paths"] = paths.copy()
            state_scores = state_scores + paths
            identity_scores = identity_scores + paths

        # Path evidence can change which relatives compete as parents.
        # Replace the earlier reciprocal term with messages for these
        # adjusted unary scores; do not add two copies or feed the new
        # messages back into the path factor.
        state_scores = state_scores - family
        identity_scores = identity_scores - family
        avoidance, iterations, residual = _converged_family_avoidance(
            identity_scores, alternatives, states, log_multiplicity, samples,
            settings.parent_state_family_final_max_iterations,
            settings.parent_state_family_final_tolerance)
        retry_iterations = 0
        if residual > settings.parent_state_family_final_tolerance:
            # Loopy messages can oscillate in a chromosome resample. Restart
            # the same equations with damping, retaining the undamped residual
            # criterion. Successful initial solves are untouched.
            avoidance, retry_iterations, residual = _converged_family_avoidance(
                identity_scores, alternatives, states, log_multiplicity, samples,
                settings.parent_state_family_retry_iterations,
                settings.parent_state_family_final_tolerance,
                settings.parent_state_family_retry_damping)
        if residual > settings.parent_state_family_final_tolerance:
            raise FloatingPointError(
                f"Final reciprocal family messages did not converge in {iterations} "
                f"initial + {retry_iterations} damped iterations "
                f"(undamped log-message residual {residual:g})")
        family = _family_scores(avoidance, alternatives, eligible, explicit_direction)
        state_scores = state_scores + family
        identity_scores = identity_scores + family
        if score_diagnostics is not None:
            score_diagnostics["reciprocal_family"] = family.copy()
            score_diagnostics["final_family_iterations"] = iterations + retry_iterations
            score_diagnostics["final_family_retry_iterations"] = retry_iterations
            score_diagnostics["final_family_residual"] = residual
    return state_scores, identity_scores
