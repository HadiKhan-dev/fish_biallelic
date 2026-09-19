"""Finite continuous orientation and reciprocal-family cavity scores.

Finite composite-evidence corrections, not independent biological direction
probabilities. Four synchronous family-message passes exclude immediate reverse
feedback; loopy graphs are approximate, not solved exactly or to convergence.
Genetic scoring and final parent-state priors are unchanged.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit
from . import direction


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


def adjust_scores(state_scores, identity_scores, alternatives, states,
                  full_counts, model, settings, explicit_direction=None):
    """Apply finite evidence to both state and identity; exposure remains hard."""
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
    state_scores = state_scores + correction
    identity_scores = identity_scores + correction
    if settings.parent_state_direction_model in {"family", "continuous_family"}:
        # Equal state weights in this likelihood-only message. Final configurable
        # M0/M1/M2 priors and their sensitivity analysis remain downstream.
        log_multiplicity = -np.log(np.maximum(1, full_counts))
        avoidance = _family_avoidance(
            identity_scores, alternatives, states, log_multiplicity, samples,
            settings.parent_state_family_message_passes)
        family = np.zeros(len(alternatives))
        for slot in (1, 2):
            present = (alternatives[:, slot] >= 0) & eligible
            c, p = alternatives[present, 0], alternatives[present, slot]
            values = avoidance[p, c]
            if explicit_direction is not None:
                values = np.where(explicit_direction[c, p], 0.0, values)
            family[present] += values
        state_scores = state_scores + family
        identity_scores = identity_scores + family
    return state_scores, identity_scores
