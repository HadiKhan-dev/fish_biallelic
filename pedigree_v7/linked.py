"""Linked transmission likelihoods for reconstructed parental haplotypes.

The kernels in this module deliberately do not infer whether a biological
parent is present in the candidate set.  They compare *relative* candidate
parent likelihoods.  Each candidate parent's two painted haplotype tracks are
converted to local allele probabilities from the reconstructed founder
haplotypes.  A small HMM then marginalises which parental track was
transmitted, with changes of track governed by genomic distance.

This formulation has two useful properties for the real cichlid cross:

* founder labels that are locally IBS-equivalent give the same (or, after
  caller-side IBS smoothing, deliberately equivalent) allele emissions; and
* the child emission uses its raw genotype likelihood rather than treating a
  noisy child founder-label call as observed truth.

The module contains no dataset paths or acceptance thresholds.  Reporting and
cohort-level uncertainty belong in the handoff builder.
"""

import math
from typing import NamedTuple

import numba
import numpy as np
from numba import njit, prange

from .transmission_math import (
    TINY as _TINY,
    diploid_distribution as _diploid_distribution,
    recombination_fraction as _recombination_fraction,
    state_emission as _emission,
)


class TwoParentTransmissionDiagnostics(NamedTuple):
    """Linked likelihood and auditable crossover summaries.

    Expected switch counts are posterior expectations under the same forward
    model used for ``log_likelihood``. Viterbi counts describe the single
    highest-probability transmission path and are diagnostics, not independent
    evidence.
    """

    log_likelihood: np.ndarray
    expected_parent1_switches: np.ndarray
    expected_parent2_switches: np.ndarray
    viterbi_parent1_switches: np.ndarray
    viterbi_parent2_switches: np.ndarray
    viterbi_log_likelihood: np.ndarray


@njit(cache=True, parallel=True)
def _score_two_parent_kernel(
    child_likelihoods,
    parent_tracks,
    parent_pairs,
    positions,
    recombination_rate,
    error_rate,
):
    n_children = child_likelihoods.shape[0]
    n_pairs = parent_pairs.shape[0]
    n_sites = child_likelihoods.shape[1]
    output = np.empty((n_children, n_pairs), dtype=np.float64)

    for flat_index in prange(n_children * n_pairs):
        child = flat_index // n_pairs
        pair = flat_index - child * n_pairs
        parent1 = parent_pairs[pair, 0]
        parent2 = parent_pairs[pair, 1]

        forward = np.full(4, 0.25, dtype=np.float64)
        updated = np.empty(4, dtype=np.float64)
        total_loglik = 0.0

        for site in range(n_sites):
            if site > 0:
                theta = _recombination_fraction(
                    float(positions[site] - positions[site - 1]),
                    recombination_rate,
                )
                stay = 1.0 - theta
                for current in range(4):
                    current_track1 = current >> 1
                    current_track2 = current & 1
                    transition_sum = 0.0
                    for previous in range(4):
                        previous_track1 = previous >> 1
                        previous_track2 = previous & 1
                        probability1 = (
                            stay
                            if current_track1 == previous_track1
                            else theta
                        )
                        probability2 = (
                            stay
                            if current_track2 == previous_track2
                            else theta
                        )
                        transition_sum += (
                            forward[previous] * probability1 * probability2
                        )
                    updated[current] = transition_sum
                for state in range(4):
                    forward[state] = updated[state]

            scale = 0.0
            for state in range(4):
                track1 = state >> 1
                track2 = state & 1
                emission = _emission(
                    child_likelihoods[child, site],
                    parent_tracks[parent1, site, track1, 1],
                    parent_tracks[parent2, site, track2, 1],
                    error_rate,
                )
                forward[state] *= emission
                scale += forward[state]
            scale = max(scale, _TINY)
            total_loglik += math.log(scale)
            inverse_scale = 1.0 / scale
            for state in range(4):
                forward[state] *= inverse_scale

        output[child, pair] = total_loglik
    return output


@njit(cache=True, parallel=True)
def _score_two_parent_diagnostic_kernel(
    child_likelihoods,
    parent_tracks,
    parent_pairs,
    positions,
    recombination_rate,
    error_rate,
):
    """Score pairs and propagate exact expected transition counts."""
    n_children = child_likelihoods.shape[0]
    n_pairs = parent_pairs.shape[0]
    n_sites = child_likelihoods.shape[1]
    shape = (n_children, n_pairs)
    log_likelihood = np.empty(shape, dtype=np.float64)
    expected_parent1 = np.empty(shape, dtype=np.float64)
    expected_parent2 = np.empty(shape, dtype=np.float64)
    viterbi_parent1 = np.empty(shape, dtype=np.int64)
    viterbi_parent2 = np.empty(shape, dtype=np.int64)
    viterbi_log_likelihood = np.empty(shape, dtype=np.float64)

    for flat_index in prange(n_children * n_pairs):
        child = flat_index // n_pairs
        pair = flat_index - child * n_pairs
        parent1 = parent_pairs[pair, 0]
        parent2 = parent_pairs[pair, 1]

        forward = np.full(4, 0.25, dtype=np.float64)
        count1 = np.zeros(4, dtype=np.float64)
        count2 = np.zeros(4, dtype=np.float64)
        updated = np.empty(4, dtype=np.float64)
        updated_count1 = np.empty(4, dtype=np.float64)
        updated_count2 = np.empty(4, dtype=np.float64)

        viterbi = np.full(4, math.log(0.25), dtype=np.float64)
        viterbi_count1 = np.zeros(4, dtype=np.int64)
        viterbi_count2 = np.zeros(4, dtype=np.int64)
        updated_viterbi = np.empty(4, dtype=np.float64)
        updated_viterbi_count1 = np.empty(4, dtype=np.int64)
        updated_viterbi_count2 = np.empty(4, dtype=np.int64)
        total_loglik = 0.0

        for site in range(n_sites):
            if site > 0:
                theta = _recombination_fraction(
                    float(positions[site] - positions[site - 1]),
                    recombination_rate,
                )
                stay = 1.0 - theta
                for current in range(4):
                    current_track1 = current >> 1
                    current_track2 = current & 1
                    probability_sum = 0.0
                    count_sum1 = 0.0
                    count_sum2 = 0.0
                    best_value = -math.inf
                    best_count1 = 0
                    best_count2 = 0
                    for previous in range(4):
                        previous_track1 = previous >> 1
                        previous_track2 = previous & 1
                        switched1 = current_track1 != previous_track1
                        switched2 = current_track2 != previous_track2
                        probability1 = theta if switched1 else stay
                        probability2 = theta if switched2 else stay
                        transition = probability1 * probability2
                        probability_sum += forward[previous] * transition
                        count_sum1 += transition * (
                            count1[previous]
                            + forward[previous] * int(switched1)
                        )
                        count_sum2 += transition * (
                            count2[previous]
                            + forward[previous] * int(switched2)
                        )
                        candidate = viterbi[previous] + math.log(transition)
                        if candidate > best_value:
                            best_value = candidate
                            best_count1 = (
                                viterbi_count1[previous] + int(switched1)
                            )
                            best_count2 = (
                                viterbi_count2[previous] + int(switched2)
                            )
                    updated[current] = probability_sum
                    updated_count1[current] = count_sum1
                    updated_count2[current] = count_sum2
                    updated_viterbi[current] = best_value
                    updated_viterbi_count1[current] = best_count1
                    updated_viterbi_count2[current] = best_count2
                for state in range(4):
                    forward[state] = updated[state]
                    count1[state] = updated_count1[state]
                    count2[state] = updated_count2[state]
                    viterbi[state] = updated_viterbi[state]
                    viterbi_count1[state] = updated_viterbi_count1[state]
                    viterbi_count2[state] = updated_viterbi_count2[state]

            scale = 0.0
            for state in range(4):
                track1 = state >> 1
                track2 = state & 1
                emission = _emission(
                    child_likelihoods[child, site],
                    parent_tracks[parent1, site, track1, 1],
                    parent_tracks[parent2, site, track2, 1],
                    error_rate,
                )
                forward[state] *= emission
                count1[state] *= emission
                count2[state] *= emission
                viterbi[state] += math.log(emission)
                scale += forward[state]
            scale = max(scale, _TINY)
            total_loglik += math.log(scale)
            inverse_scale = 1.0 / scale
            for state in range(4):
                forward[state] *= inverse_scale
                count1[state] *= inverse_scale
                count2[state] *= inverse_scale

        best_state = 0
        for state in range(1, 4):
            if viterbi[state] > viterbi[best_state]:
                best_state = state
        log_likelihood[child, pair] = total_loglik
        expected_parent1[child, pair] = np.sum(count1)
        expected_parent2[child, pair] = np.sum(count2)
        viterbi_parent1[child, pair] = viterbi_count1[best_state]
        viterbi_parent2[child, pair] = viterbi_count2[best_state]
        viterbi_log_likelihood[child, pair] = viterbi[best_state]

    return (
        log_likelihood,
        expected_parent1,
        expected_parent2,
        viterbi_parent1,
        viterbi_parent2,
        viterbi_log_likelihood,
    )


@njit(cache=True, parallel=True)
def _score_two_parent_binned_diagnostic_kernel(
    child_likelihoods,
    parent_tracks,
    parent_pairs,
    positions,
    recombination_rate,
    error_rate,
    markers_per_block,
    effective_markers_per_block,
):
    """Robust linked HMM with geometric-mean composite block emissions."""
    n_children = child_likelihoods.shape[0]
    n_pairs = parent_pairs.shape[0]
    n_sites = child_likelihoods.shape[1]
    n_blocks = (n_sites + markers_per_block - 1) // markers_per_block
    shape = (n_children, n_pairs)
    log_likelihood = np.empty(shape, dtype=np.float64)
    expected_parent1 = np.empty(shape, dtype=np.float64)
    expected_parent2 = np.empty(shape, dtype=np.float64)
    viterbi_parent1 = np.empty(shape, dtype=np.int64)
    viterbi_parent2 = np.empty(shape, dtype=np.int64)
    viterbi_log_likelihood = np.empty(shape, dtype=np.float64)

    for flat_index in prange(n_children * n_pairs):
        child = flat_index // n_pairs
        pair = flat_index - child * n_pairs
        parent1 = parent_pairs[pair, 0]
        parent2 = parent_pairs[pair, 1]
        forward = np.full(4, 0.25, dtype=np.float64)
        count1 = np.zeros(4, dtype=np.float64)
        count2 = np.zeros(4, dtype=np.float64)
        updated = np.empty(4, dtype=np.float64)
        updated_count1 = np.empty(4, dtype=np.float64)
        updated_count2 = np.empty(4, dtype=np.float64)
        viterbi = np.full(4, math.log(0.25), dtype=np.float64)
        viterbi_count1 = np.zeros(4, dtype=np.int64)
        viterbi_count2 = np.zeros(4, dtype=np.int64)
        updated_viterbi = np.empty(4, dtype=np.float64)
        updated_viterbi_count1 = np.empty(4, dtype=np.int64)
        updated_viterbi_count2 = np.empty(4, dtype=np.int64)
        block_log_emission = np.empty(4, dtype=np.float64)
        total_loglik = 0.0
        previous_center = 0.0

        for block in range(n_blocks):
            start = block * markers_per_block
            end = min(start + markers_per_block, n_sites)
            center = 0.5 * float(positions[start] + positions[end - 1])
            if block > 0:
                theta = _recombination_fraction(
                    center - previous_center, recombination_rate
                )
                stay = 1.0 - theta
                for current in range(4):
                    current_track1 = current >> 1
                    current_track2 = current & 1
                    probability_sum = 0.0
                    count_sum1 = 0.0
                    count_sum2 = 0.0
                    best_value = -math.inf
                    best_count1 = 0
                    best_count2 = 0
                    for previous in range(4):
                        previous_track1 = previous >> 1
                        previous_track2 = previous & 1
                        switched1 = current_track1 != previous_track1
                        switched2 = current_track2 != previous_track2
                        probability1 = theta if switched1 else stay
                        probability2 = theta if switched2 else stay
                        transition = probability1 * probability2
                        probability_sum += forward[previous] * transition
                        count_sum1 += transition * (
                            count1[previous]
                            + forward[previous] * int(switched1)
                        )
                        count_sum2 += transition * (
                            count2[previous]
                            + forward[previous] * int(switched2)
                        )
                        candidate = viterbi[previous] + math.log(transition)
                        if candidate > best_value:
                            best_value = candidate
                            best_count1 = (
                                viterbi_count1[previous] + int(switched1)
                            )
                            best_count2 = (
                                viterbi_count2[previous] + int(switched2)
                            )
                    updated[current] = probability_sum
                    updated_count1[current] = count_sum1
                    updated_count2[current] = count_sum2
                    updated_viterbi[current] = best_value
                    updated_viterbi_count1[current] = best_count1
                    updated_viterbi_count2[current] = best_count2
                for state in range(4):
                    forward[state] = updated[state]
                    count1[state] = updated_count1[state]
                    count2[state] = updated_count2[state]
                    viterbi[state] = updated_viterbi[state]
                    viterbi_count1[state] = updated_viterbi_count1[state]
                    viterbi_count2[state] = updated_viterbi_count2[state]

            exponent = min(
                effective_markers_per_block, float(end - start)
            ) / float(end - start)
            maximum_log_emission = -math.inf
            for state in range(4):
                track1 = state >> 1
                track2 = state & 1
                value = 0.0
                for site in range(start, end):
                    value += math.log(_emission(
                        child_likelihoods[child, site],
                        parent_tracks[parent1, site, track1, 1],
                        parent_tracks[parent2, site, track2, 1],
                        error_rate,
                    ))
                value *= exponent
                block_log_emission[state] = value
                if value > maximum_log_emission:
                    maximum_log_emission = value

            scale = 0.0
            for state in range(4):
                emission = math.exp(
                    block_log_emission[state] - maximum_log_emission
                )
                forward[state] *= emission
                count1[state] *= emission
                count2[state] *= emission
                viterbi[state] += block_log_emission[state]
                scale += forward[state]
            scale = max(scale, _TINY)
            total_loglik += maximum_log_emission + math.log(scale)
            inverse_scale = 1.0 / scale
            for state in range(4):
                forward[state] *= inverse_scale
                count1[state] *= inverse_scale
                count2[state] *= inverse_scale
            previous_center = center

        best_state = 0
        for state in range(1, 4):
            if viterbi[state] > viterbi[best_state]:
                best_state = state
        log_likelihood[child, pair] = total_loglik
        expected_parent1[child, pair] = np.sum(count1)
        expected_parent2[child, pair] = np.sum(count2)
        viterbi_parent1[child, pair] = viterbi_count1[best_state]
        viterbi_parent2[child, pair] = viterbi_count2[best_state]
        viterbi_log_likelihood[child, pair] = viterbi[best_state]

    return (
        log_likelihood,
        expected_parent1,
        expected_parent2,
        viterbi_parent1,
        viterbi_parent2,
        viterbi_log_likelihood,
    )


@njit(cache=True, parallel=True)
def _score_one_parent_kernel(
    child_likelihoods,
    parent_tracks,
    unknown_alt_probability,
    positions,
    recombination_rate,
    error_rate,
):
    n_children = child_likelihoods.shape[0]
    n_parents = parent_tracks.shape[0]
    n_sites = child_likelihoods.shape[1]
    output = np.empty((n_children, n_parents), dtype=np.float64)

    for flat_index in prange(n_children * n_parents):
        child = flat_index // n_parents
        parent = flat_index - child * n_parents
        forward0 = 0.5
        forward1 = 0.5
        total_loglik = 0.0

        for site in range(n_sites):
            if site > 0:
                theta = _recombination_fraction(
                    float(positions[site] - positions[site - 1]),
                    recombination_rate,
                )
                stay = 1.0 - theta
                previous0 = forward0
                previous1 = forward1
                forward0 = previous0 * stay + previous1 * theta
                forward1 = previous1 * stay + previous0 * theta

            emission0 = _emission(
                child_likelihoods[child, site],
                parent_tracks[parent, site, 0, 1],
                unknown_alt_probability[site],
                error_rate,
            )
            emission1 = _emission(
                child_likelihoods[child, site],
                parent_tracks[parent, site, 1, 1],
                unknown_alt_probability[site],
                error_rate,
            )
            forward0 *= emission0
            forward1 *= emission1
            scale = max(forward0 + forward1, _TINY)
            total_loglik += math.log(scale)
            forward0 /= scale
            forward1 /= scale

        output[child, parent] = total_loglik
    return output


def _validate_common(child_likelihoods, parent_tracks, positions, error_rate):
    child = np.ascontiguousarray(child_likelihoods, dtype=np.float64)
    tracks = np.ascontiguousarray(parent_tracks, dtype=np.float64)
    marker_positions = np.ascontiguousarray(positions, dtype=np.int64)
    if child.ndim != 3 or child.shape[2] != 3:
        raise ValueError("child_likelihoods must have shape (children, sites, 3)")
    if tracks.ndim != 4 or tracks.shape[2:] != (2, 2):
        raise ValueError("parent_tracks must have shape (parents, sites, 2, 2)")
    if child.shape[1] != tracks.shape[1]:
        raise ValueError("child and parent arrays must contain the same sites")
    if marker_positions.shape != (child.shape[1],):
        raise ValueError("positions must contain one coordinate per site")
    if len(marker_positions) and np.any(np.diff(marker_positions) <= 0):
        raise ValueError("positions must be strictly increasing")
    if not 0.0 <= error_rate < 1.0:
        raise ValueError("error_rate must lie in [0, 1)")
    if not np.all(np.isfinite(child)) or np.any(child < 0.0):
        raise ValueError("child likelihoods must be finite and non-negative")
    if not np.all(np.isfinite(tracks)) or np.any(tracks < 0.0):
        raise ValueError("parent track probabilities must be finite and non-negative")
    track_sums = np.sum(tracks, axis=3)
    if not np.allclose(track_sums, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("each parent-track allele distribution must sum to one")
    return child, tracks, marker_positions


def score_two_parent_transmission(
    child_likelihoods,
    parent_tracks,
    parent_pairs,
    positions,
    recombination_rate=5e-8,
    error_rate=0.01,
):
    """Score every child against every candidate two-parent configuration."""
    child, tracks, marker_positions = _validate_common(
        child_likelihoods, parent_tracks, positions, error_rate
    )
    pairs = np.ascontiguousarray(parent_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("parent_pairs must have shape (pairs, 2)")
    if len(pairs) and (np.min(pairs) < 0 or np.max(pairs) >= len(tracks)):
        raise ValueError("parent_pairs contains an invalid parent index")
    if not np.isfinite(recombination_rate) or recombination_rate < 0.0:
        raise ValueError("recombination_rate must be finite and non-negative")
    return _score_two_parent_kernel(
        child,
        tracks,
        pairs,
        marker_positions,
        float(recombination_rate),
        float(error_rate),
    )


def score_two_parent_transmission_diagnostics(
    child_likelihoods,
    parent_tracks,
    parent_pairs,
    positions,
    recombination_rate=5e-8,
    error_rate=0.01,
):
    """Return linked scores with expected and Viterbi crossover counts."""
    child, tracks, marker_positions = _validate_common(
        child_likelihoods, parent_tracks, positions, error_rate
    )
    pairs = np.ascontiguousarray(parent_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("parent_pairs must have shape (pairs, 2)")
    if len(pairs) and (np.min(pairs) < 0 or np.max(pairs) >= len(tracks)):
        raise ValueError("parent_pairs contains an invalid parent index")
    if not np.isfinite(recombination_rate) or recombination_rate < 0.0:
        raise ValueError("recombination_rate must be finite and non-negative")
    return TwoParentTransmissionDiagnostics(
        *_score_two_parent_diagnostic_kernel(
            child,
            tracks,
            pairs,
            marker_positions,
            float(recombination_rate),
            float(error_rate),
        )
    )


def score_two_parent_binned_transmission_diagnostics(
    child_likelihoods,
    parent_tracks,
    parent_pairs,
    positions,
    recombination_rate=5e-8,
    error_rate=0.01,
    markers_per_block=100,
    effective_markers_per_block=1.0,
):
    """Score linked transmissions with robust composite block emissions.

    All markers contribute, but the product of emissions within a block is
    tempered to ``effective_markers_per_block`` independent observations.
    This prevents dense LD and correlated reconstruction errors from creating
    biologically implausible crossover paths.
    """
    child, tracks, marker_positions = _validate_common(
        child_likelihoods, parent_tracks, positions, error_rate
    )
    pairs = np.ascontiguousarray(parent_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("parent_pairs must have shape (pairs, 2)")
    if len(pairs) and (np.min(pairs) < 0 or np.max(pairs) >= len(tracks)):
        raise ValueError("parent_pairs contains an invalid parent index")
    if not np.isfinite(recombination_rate) or recombination_rate < 0.0:
        raise ValueError("recombination_rate must be finite and non-negative")
    if int(markers_per_block) != markers_per_block or markers_per_block < 1:
        raise ValueError("markers_per_block must be a positive integer")
    if (
        not np.isfinite(effective_markers_per_block)
        or effective_markers_per_block <= 0.0
    ):
        raise ValueError("effective_markers_per_block must be finite and positive")
    return TwoParentTransmissionDiagnostics(
        *_score_two_parent_binned_diagnostic_kernel(
            child,
            tracks,
            pairs,
            marker_positions,
            float(recombination_rate),
            float(error_rate),
            int(markers_per_block),
            float(effective_markers_per_block),
        )
    )


def score_one_parent_transmission(
    child_likelihoods,
    parent_tracks,
    unknown_alt_probability,
    positions,
    recombination_rate=5e-8,
    error_rate=0.01,
):
    """Score every child against every observed parent with one unknown mate."""
    child, tracks, marker_positions = _validate_common(
        child_likelihoods, parent_tracks, positions, error_rate
    )
    unknown = np.ascontiguousarray(unknown_alt_probability, dtype=np.float64)
    if unknown.shape != (child.shape[1],):
        raise ValueError("unknown_alt_probability must contain one value per site")
    if np.any(~np.isfinite(unknown)) or np.any((unknown < 0.0) | (unknown > 1.0)):
        raise ValueError("unknown_alt_probability must lie in [0, 1]")
    if not np.isfinite(recombination_rate) or recombination_rate < 0.0:
        raise ValueError("recombination_rate must be finite and non-negative")
    return _score_one_parent_kernel(
        child,
        tracks,
        unknown,
        marker_positions,
        float(recombination_rate),
        float(error_rate),
    )


def numba_thread_capacity():
    """Return the Numba thread ceiling visible to this process."""
    return int(numba.config.NUMBA_NUM_THREADS)
