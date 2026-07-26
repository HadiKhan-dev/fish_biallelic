"""Multigeneration transmission primitives for the Tropheops pedigree.

The functions here reconstruct a child's two homologs from a specified pair
of observed parents.  Unlike a generic sample painting, the returned tracks
are anchored by parental origin: track 0 came from parent 1 and track 1 came
from parent 2.  The primary reconstruction integrates unphased parental
genotype posteriors and is therefore invariant to arbitrary local swaps of a
parent's two painted tracks.  A linked block-composite forward--backward HMM
is retained for datasets where parental track phase is defensible, and as a
diagnostic in the separate Tropheops v7 analysis.

This module contains no dataset paths, candidate eligibility rules, reporting
thresholds, or cohort labels.  It is used by the separate Tropheops v7
builder; existing pipeline stages remain unchanged.
"""

import math
from typing import NamedTuple

import numpy as np
from numba import njit, prange


_TINY = np.finfo(np.float64).tiny


class InheritedTrackPosterior(NamedTuple):
    """Posterior child homologs and diagnostics for specified parent pairs."""

    log_likelihood: np.ndarray
    tracks: np.ndarray
    expected_parent1_switches: np.ndarray
    expected_parent2_switches: np.ndarray
    mean_state_entropy: np.ndarray


class UnphasedParentalOriginPosterior(NamedTuple):
    """Sitewise parental-origin homologs without parental phase assumptions."""

    tracks: np.ndarray
    mean_parent1_allele_entropy: np.ndarray
    mean_parent2_allele_entropy: np.ndarray


class MissingParentTransmissionScores(NamedTuple):
    """Composite scores for zero or one observed F1 parent."""

    zero_parent: np.ndarray
    father_only: np.ndarray
    mother_only: np.ndarray
    unknown_father_alt_probability: np.ndarray
    unknown_mother_alt_probability: np.ndarray


@njit(cache=True, inline="always")
def _recombination_fraction(distance_bp, recombination_rate):
    value = 0.5 * (1.0 - math.exp(-2.0 * distance_bp * recombination_rate))
    if value < 1e-15:
        return 1e-15
    if value > 0.5:
        return 0.5
    return value


@njit(cache=True, inline="always")
def _state_emission(child_likelihood, first_alt, second_alt, error_rate):
    first_ref = 1.0 - first_alt
    second_ref = 1.0 - second_alt
    p0 = first_ref * second_ref
    p1 = first_alt * second_ref + first_ref * second_alt
    p2 = first_alt * second_alt
    retained = 1.0 - error_rate
    background = error_rate / 3.0
    value = (
        child_likelihood[0] * (retained * p0 + background)
        + child_likelihood[1] * (retained * p1 + background)
        + child_likelihood[2] * (retained * p2 + background)
    )
    return max(value, _TINY)


@njit(cache=True, inline="always")
def _conditional_allele_marginals(
    child_likelihood,
    first_alt,
    second_alt,
    error_rate,
):
    """Return parental-allele marginals conditional on one site's data."""
    likelihood_sum = (
        child_likelihood[0]
        + child_likelihood[1]
        + child_likelihood[2]
    )
    first_one = 0.0
    second_one = 0.0
    total = 0.0
    for first in range(2):
        first_probability = first_alt if first else 1.0 - first_alt
        for second in range(2):
            second_probability = second_alt if second else 1.0 - second_alt
            observation = (
                (1.0 - error_rate) * child_likelihood[first + second]
                + (error_rate / 3.0) * likelihood_sum
            )
            weight = first_probability * second_probability * observation
            total += weight
            first_one += weight * first
            second_one += weight * second
    if total <= _TINY:
        return first_alt, second_alt
    return first_one / total, second_one / total


@njit(cache=True, parallel=True)
def _condition_tracks_kernel(tracks, likelihoods, error_rate):
    n_samples = tracks.shape[0]
    n_sites = tracks.shape[1]
    output = np.empty_like(tracks)
    for flat_index in prange(n_samples * n_sites):
        sample = flat_index // n_sites
        site = flat_index - sample * n_sites
        first, second = _conditional_allele_marginals(
            likelihoods[sample, site],
            tracks[sample, site, 0, 1],
            tracks[sample, site, 1, 1],
            error_rate,
        )
        output[sample, site, 0, 0] = 1.0 - first
        output[sample, site, 0, 1] = first
        output[sample, site, 1, 0] = 1.0 - second
        output[sample, site, 1, 1] = second
    return output


def condition_tracks_on_genotype_likelihoods(
    tracks,
    genotype_likelihoods,
    error_rate=0.01,
):
    """Update two probabilistic tracks using the sample's diploid GLs."""
    values = np.ascontiguousarray(tracks, dtype=np.float64)
    likelihoods = np.ascontiguousarray(
        genotype_likelihoods, dtype=np.float64
    )
    if values.ndim != 4 or values.shape[2:] != (2, 2):
        raise ValueError("tracks must have shape (samples, sites, 2, 2)")
    if likelihoods.shape != values.shape[:2] + (3,):
        raise ValueError("genotype_likelihoods must match samples and sites")
    if not 0.0 <= error_rate < 1.0:
        raise ValueError("error_rate must lie in [0, 1)")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("tracks must be finite and non-negative")
    if np.any(~np.isfinite(likelihoods)) or np.any(likelihoods < 0.0):
        raise ValueError("likelihoods must be finite and non-negative")
    if not np.allclose(np.sum(values, axis=3), 1.0, atol=1e-8):
        raise ValueError("track allele distributions must sum to one")
    return _condition_tracks_kernel(values, likelihoods, float(error_rate))


@njit(cache=True, inline="always")
def _genotype_posterior_from_tracks(track, likelihood):
    """Return P(genotype | unordered painted tracks, genotype likelihood)."""
    first_alt = track[0, 1]
    second_alt = track[1, 1]
    prior0 = (1.0 - first_alt) * (1.0 - second_alt)
    prior1 = (
        first_alt * (1.0 - second_alt)
        + (1.0 - first_alt) * second_alt
    )
    prior2 = first_alt * second_alt
    value0 = prior0 * likelihood[0]
    value1 = prior1 * likelihood[1]
    value2 = prior2 * likelihood[2]
    total = value0 + value1 + value2
    if total <= _TINY:
        # A zero likelihood vector is treated as missing rather than erasing
        # the genotype information already present in the painted tracks.
        total = max(prior0 + prior1 + prior2, _TINY)
        return prior0 / total, prior1 / total, prior2 / total
    return value0 / total, value1 / total, value2 / total


@njit(cache=True, parallel=True)
def _unphased_parental_origin_kernel(
    child_likelihoods,
    parent_tracks,
    parent_likelihoods,
    selected_pairs,
    error_rate,
):
    n_children = child_likelihoods.shape[0]
    n_sites = child_likelihoods.shape[1]
    output = np.empty((n_children, n_sites, 2, 2), dtype=np.float64)
    entropy1 = np.zeros(n_children, dtype=np.float64)
    entropy2 = np.zeros(n_children, dtype=np.float64)
    for child in prange(n_children):
        parent1 = selected_pairs[child, 0]
        parent2 = selected_pairs[child, 1]
        child_entropy1 = 0.0
        child_entropy2 = 0.0
        for site in range(n_sites):
            posterior1 = _genotype_posterior_from_tracks(
                parent_tracks[parent1, site],
                parent_likelihoods[parent1, site],
            )
            posterior2 = _genotype_posterior_from_tracks(
                parent_tracks[parent2, site],
                parent_likelihoods[parent2, site],
            )
            likelihood_sum = (
                child_likelihoods[child, site, 0]
                + child_likelihoods[child, site, 1]
                + child_likelihoods[child, site, 2]
            )
            total = 0.0
            first_alt_weight = 0.0
            second_alt_weight = 0.0
            for genotype1 in range(3):
                transmitted_alt1 = 0.5 * genotype1
                for genotype2 in range(3):
                    transmitted_alt2 = 0.5 * genotype2
                    genotype_weight = (
                        posterior1[genotype1] * posterior2[genotype2]
                    )
                    for allele1 in range(2):
                        transmission1 = (
                            transmitted_alt1
                            if allele1
                            else 1.0 - transmitted_alt1
                        )
                        for allele2 in range(2):
                            transmission2 = (
                                transmitted_alt2
                                if allele2
                                else 1.0 - transmitted_alt2
                            )
                            observation = (
                                (1.0 - error_rate)
                                * child_likelihoods[
                                    child, site, allele1 + allele2
                                ]
                                + (error_rate / 3.0) * likelihood_sum
                            )
                            weight = (
                                genotype_weight
                                * transmission1
                                * transmission2
                                * observation
                            )
                            total += weight
                            first_alt_weight += weight * allele1
                            second_alt_weight += weight * allele2
            if total <= _TINY:
                # Missing or contradictory child evidence: retain the
                # Mendelian transmission marginal from each parent.
                first_alt = 0.5 * (
                    posterior1[1] + 2.0 * posterior1[2]
                )
                second_alt = 0.5 * (
                    posterior2[1] + 2.0 * posterior2[2]
                )
            else:
                first_alt = first_alt_weight / total
                second_alt = second_alt_weight / total
            first_alt = min(1.0, max(0.0, first_alt))
            second_alt = min(1.0, max(0.0, second_alt))
            output[child, site, 0, 0] = 1.0 - first_alt
            output[child, site, 0, 1] = first_alt
            output[child, site, 1, 0] = 1.0 - second_alt
            output[child, site, 1, 1] = second_alt
            if 0.0 < first_alt < 1.0:
                child_entropy1 -= (
                    first_alt * math.log(first_alt)
                    + (1.0 - first_alt) * math.log(1.0 - first_alt)
                )
            if 0.0 < second_alt < 1.0:
                child_entropy2 -= (
                    second_alt * math.log(second_alt)
                    + (1.0 - second_alt) * math.log(1.0 - second_alt)
                )
        entropy1[child] = child_entropy1 / max(n_sites, 1)
        entropy2[child] = child_entropy2 / max(n_sites, 1)
    return output, entropy1, entropy2


def reconstruct_parental_origin_tracks_unphased(
    child_likelihoods,
    parent_tracks,
    parent_likelihoods,
    selected_parent_pairs,
    error_rate=0.01,
):
    """Reconstruct child homologs by parent of origin without parental phase.

    Each parent's unordered painted tracks supply a diploid-genotype prior.
    That prior is updated by the parent's raw genotype likelihood, then both
    Mendelian transmissions are integrated jointly with the child's genotype
    likelihood.  Consequently, locally swapping either parent's two input
    tracks cannot change the result.
    """
    child = np.ascontiguousarray(child_likelihoods, dtype=np.float64)
    tracks = np.ascontiguousarray(parent_tracks, dtype=np.float64)
    parent_gl = np.ascontiguousarray(parent_likelihoods, dtype=np.float64)
    pairs = np.ascontiguousarray(selected_parent_pairs, dtype=np.int64)
    if child.ndim != 3 or child.shape[2] != 3:
        raise ValueError("child_likelihoods must have shape (children, sites, 3)")
    if tracks.ndim != 4 or tracks.shape[2:] != (2, 2):
        raise ValueError("parent_tracks must have shape (parents, sites, 2, 2)")
    if parent_gl.shape != tracks.shape[:2] + (3,):
        raise ValueError("parent_likelihoods must match parent tracks and sites")
    if child.shape[1] != tracks.shape[1]:
        raise ValueError("child and parent inputs must contain the same sites")
    if pairs.shape != (len(child), 2):
        raise ValueError("selected_parent_pairs must have one pair per child")
    if len(pairs) and (np.min(pairs) < 0 or np.max(pairs) >= len(tracks)):
        raise ValueError("selected_parent_pairs contains an invalid index")
    if not 0.0 <= error_rate < 1.0:
        raise ValueError("error_rate must lie in [0, 1)")
    for name, values in (
        ("child likelihoods", child),
        ("parent tracks", tracks),
        ("parent likelihoods", parent_gl),
    ):
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
    if not np.allclose(np.sum(tracks, axis=3), 1.0, atol=1e-8):
        raise ValueError("track allele distributions must sum to one")
    output, entropy1, entropy2 = _unphased_parental_origin_kernel(
        child, tracks, parent_gl, pairs, float(error_rate)
    )
    return UnphasedParentalOriginPosterior(output, entropy1, entropy2)


@njit(cache=True, parallel=True)
def _binned_one_parent_kernel(
    child_likelihoods,
    parent_tracks,
    unknown_alt_probability,
    positions,
    recombination_rate,
    error_rate,
    markers_per_block,
    effective_markers_per_block,
):
    n_children = child_likelihoods.shape[0]
    n_parents = parent_tracks.shape[0]
    n_sites = child_likelihoods.shape[1]
    n_blocks = (n_sites + markers_per_block - 1) // markers_per_block
    output = np.empty((n_children, n_parents), dtype=np.float64)
    for flat_index in prange(n_children * n_parents):
        child = flat_index // n_parents
        parent = flat_index - child * n_parents
        forward0 = 0.5
        forward1 = 0.5
        previous_center = 0.0
        total_loglik = 0.0
        for block in range(n_blocks):
            start = block * markers_per_block
            end = min(start + markers_per_block, n_sites)
            center = 0.5 * float(positions[start] + positions[end - 1])
            exponent = min(
                effective_markers_per_block, float(end - start)
            ) / float(end - start)
            log_emission0 = 0.0
            log_emission1 = 0.0
            for site in range(start, end):
                log_emission0 += math.log(_state_emission(
                    child_likelihoods[child, site],
                    parent_tracks[parent, site, 0, 1],
                    unknown_alt_probability[site],
                    error_rate,
                ))
                log_emission1 += math.log(_state_emission(
                    child_likelihoods[child, site],
                    parent_tracks[parent, site, 1, 1],
                    unknown_alt_probability[site],
                    error_rate,
                ))
            log_emission0 *= exponent
            log_emission1 *= exponent
            maximum = max(log_emission0, log_emission1)
            emission0 = math.exp(log_emission0 - maximum)
            emission1 = math.exp(log_emission1 - maximum)
            if block > 0:
                theta = _recombination_fraction(
                    center - previous_center, recombination_rate
                )
                previous0 = forward0
                previous1 = forward1
                forward0 = (
                    previous0 * (1.0 - theta) + previous1 * theta
                )
                forward1 = (
                    previous1 * (1.0 - theta) + previous0 * theta
                )
            forward0 *= emission0
            forward1 *= emission1
            scale = max(forward0 + forward1, _TINY)
            total_loglik += maximum + math.log(scale)
            forward0 /= scale
            forward1 /= scale
            previous_center = center
        output[child, parent] = total_loglik
    return output


@njit(cache=True, parallel=True)
def _binned_zero_parent_kernel(
    child_likelihoods,
    unknown_father_alt,
    unknown_mother_alt,
    error_rate,
    markers_per_block,
    effective_markers_per_block,
):
    n_children = child_likelihoods.shape[0]
    n_sites = child_likelihoods.shape[1]
    n_blocks = (n_sites + markers_per_block - 1) // markers_per_block
    output = np.empty(n_children, dtype=np.float64)
    for child in prange(n_children):
        total_loglik = 0.0
        for block in range(n_blocks):
            start = block * markers_per_block
            end = min(start + markers_per_block, n_sites)
            exponent = min(
                effective_markers_per_block, float(end - start)
            ) / float(end - start)
            block_loglik = 0.0
            for site in range(start, end):
                block_loglik += math.log(_state_emission(
                    child_likelihoods[child, site],
                    unknown_father_alt[site],
                    unknown_mother_alt[site],
                    error_rate,
                ))
            total_loglik += exponent * block_loglik
        output[child] = total_loglik
    return output


def score_binned_missing_parent_models(
    child_likelihoods,
    parent_tracks,
    father_indices,
    mother_indices,
    positions,
    recombination_rate=5e-8,
    error_rate=0.01,
    markers_per_block=100,
    effective_markers_per_block=1.0,
):
    """Score zero-, father-only-, and mother-only observed-parent models.

    An unobserved gamete is represented by the sex-specific empirical allele
    mixture across eligible F1 homolog posteriors.  Candidate-count and state
    priors are deliberately not applied here; the v7 builder integrates those
    hypotheses during chromosome resampling.
    """
    child = np.ascontiguousarray(child_likelihoods, dtype=np.float64)
    tracks = np.ascontiguousarray(parent_tracks, dtype=np.float64)
    fathers = np.ascontiguousarray(father_indices, dtype=np.int64)
    mothers = np.ascontiguousarray(mother_indices, dtype=np.int64)
    marker_positions = np.ascontiguousarray(positions, dtype=np.int64)
    if child.ndim != 3 or child.shape[2] != 3:
        raise ValueError("child_likelihoods must have shape (children, sites, 3)")
    if tracks.ndim != 4 or tracks.shape[2:] != (2, 2):
        raise ValueError("parent_tracks must have shape (parents, sites, 2, 2)")
    if child.shape[1] != tracks.shape[1] or child.shape[1] == 0:
        raise ValueError("child and parent inputs must share at least one site")
    if fathers.ndim != 1 or mothers.ndim != 1:
        raise ValueError("father_indices and mother_indices must be one-dimensional")
    if len(fathers) == 0 or len(mothers) == 0:
        raise ValueError("both candidate-parent groups must be non-empty")
    if (
        np.min(fathers) < 0
        or np.max(fathers) >= len(tracks)
        or np.min(mothers) < 0
        or np.max(mothers) >= len(tracks)
    ):
        raise ValueError("candidate parent index is out of range")
    if marker_positions.shape != (child.shape[1],):
        raise ValueError("positions must contain one coordinate per site")
    if np.any(np.diff(marker_positions) <= 0):
        raise ValueError("positions must be strictly increasing")
    if not np.isfinite(recombination_rate) or recombination_rate < 0.0:
        raise ValueError("recombination_rate must be finite and non-negative")
    if not 0.0 <= error_rate < 1.0:
        raise ValueError("error_rate must lie in [0, 1)")
    if int(markers_per_block) != markers_per_block or markers_per_block < 1:
        raise ValueError("markers_per_block must be a positive integer")
    if (
        not np.isfinite(effective_markers_per_block)
        or effective_markers_per_block <= 0.0
    ):
        raise ValueError("effective_markers_per_block must be finite and positive")
    if np.any(~np.isfinite(child)) or np.any(child < 0.0):
        raise ValueError("child likelihoods must be finite and non-negative")
    if np.any(~np.isfinite(tracks)) or np.any(tracks < 0.0):
        raise ValueError("parent tracks must be finite and non-negative")
    if not np.allclose(np.sum(tracks, axis=3), 1.0, atol=1e-8):
        raise ValueError("track allele distributions must sum to one")
    unknown_father_alt = np.ascontiguousarray(
        np.mean(tracks[fathers, :, :, 1], axis=(0, 2)),
        dtype=np.float64,
    )
    unknown_mother_alt = np.ascontiguousarray(
        np.mean(tracks[mothers, :, :, 1], axis=(0, 2)),
        dtype=np.float64,
    )
    father_only = _binned_one_parent_kernel(
        child,
        np.ascontiguousarray(tracks[fathers]),
        unknown_mother_alt,
        marker_positions,
        float(recombination_rate),
        float(error_rate),
        int(markers_per_block),
        float(effective_markers_per_block),
    )
    mother_only = _binned_one_parent_kernel(
        child,
        np.ascontiguousarray(tracks[mothers]),
        unknown_father_alt,
        marker_positions,
        float(recombination_rate),
        float(error_rate),
        int(markers_per_block),
        float(effective_markers_per_block),
    )
    zero_parent = _binned_zero_parent_kernel(
        child,
        unknown_father_alt,
        unknown_mother_alt,
        float(error_rate),
        int(markers_per_block),
        float(effective_markers_per_block),
    )
    return MissingParentTransmissionScores(
        zero_parent,
        father_only,
        mother_only,
        unknown_father_alt,
        unknown_mother_alt,
    )


@njit(cache=True, parallel=True)
def _reconstruct_selected_pairs_kernel(
    child_likelihoods,
    parent_tracks,
    selected_pairs,
    positions,
    parent1_recombination_rate,
    parent2_recombination_rate,
    error_rate,
    markers_per_block,
    effective_markers_per_block,
):
    n_children = child_likelihoods.shape[0]
    n_sites = child_likelihoods.shape[1]
    n_blocks = (n_sites + markers_per_block - 1) // markers_per_block
    output_tracks = np.empty((n_children, n_sites, 2, 2), dtype=np.float64)
    log_likelihood = np.empty(n_children, dtype=np.float64)
    expected1 = np.empty(n_children, dtype=np.float64)
    expected2 = np.empty(n_children, dtype=np.float64)
    mean_entropy = np.empty(n_children, dtype=np.float64)

    for child in prange(n_children):
        parent1 = selected_pairs[child, 0]
        parent2 = selected_pairs[child, 1]
        block_emission = np.empty((n_blocks, 4), dtype=np.float64)
        normalized_emission = np.empty((n_blocks, 4), dtype=np.float64)
        block_center = np.empty(n_blocks, dtype=np.float64)

        for block in range(n_blocks):
            start = block * markers_per_block
            end = min(start + markers_per_block, n_sites)
            block_center[block] = 0.5 * float(
                positions[start] + positions[end - 1]
            )
            exponent = min(
                effective_markers_per_block, float(end - start)
            ) / float(end - start)
            maximum = -math.inf
            for state in range(4):
                selector1 = state >> 1
                selector2 = state & 1
                value = 0.0
                for site in range(start, end):
                    value += math.log(_state_emission(
                        child_likelihoods[child, site],
                        parent_tracks[parent1, site, selector1, 1],
                        parent_tracks[parent2, site, selector2, 1],
                        error_rate,
                    ))
                value *= exponent
                block_emission[block, state] = value
                if value > maximum:
                    maximum = value
            for state in range(4):
                normalized_emission[block, state] = math.exp(
                    block_emission[block, state] - maximum
                )

        alpha = np.empty((n_blocks, 4), dtype=np.float64)
        scales = np.empty(n_blocks, dtype=np.float64)
        total_loglik = 0.0
        scale = 0.0
        maximum = block_emission[0, 0]
        for state in range(1, 4):
            if block_emission[0, state] > maximum:
                maximum = block_emission[0, state]
        for state in range(4):
            alpha[0, state] = 0.25 * normalized_emission[0, state]
            scale += alpha[0, state]
        scale = max(scale, _TINY)
        scales[0] = scale
        total_loglik += maximum + math.log(scale)
        for state in range(4):
            alpha[0, state] /= scale

        for block in range(1, n_blocks):
            theta1 = _recombination_fraction(
                block_center[block] - block_center[block - 1],
                parent1_recombination_rate,
            )
            theta2 = _recombination_fraction(
                block_center[block] - block_center[block - 1],
                parent2_recombination_rate,
            )
            scale = 0.0
            for current in range(4):
                selector1 = current >> 1
                selector2 = current & 1
                value = 0.0
                for previous in range(4):
                    transition1 = (
                        1.0 - theta1
                        if selector1 == previous >> 1
                        else theta1
                    )
                    transition2 = (
                        1.0 - theta2
                        if selector2 == (previous & 1)
                        else theta2
                    )
                    value += alpha[block - 1, previous] * transition1 * transition2
                alpha[block, current] = (
                    value * normalized_emission[block, current]
                )
                scale += alpha[block, current]
            scale = max(scale, _TINY)
            scales[block] = scale
            maximum = block_emission[block, 0]
            for state in range(1, 4):
                if block_emission[block, state] > maximum:
                    maximum = block_emission[block, state]
            total_loglik += maximum + math.log(scale)
            for state in range(4):
                alpha[block, state] /= scale

        beta = np.ones((n_blocks, 4), dtype=np.float64)
        for block in range(n_blocks - 2, -1, -1):
            theta1 = _recombination_fraction(
                block_center[block + 1] - block_center[block],
                parent1_recombination_rate,
            )
            theta2 = _recombination_fraction(
                block_center[block + 1] - block_center[block],
                parent2_recombination_rate,
            )
            beta_total = 0.0
            for previous in range(4):
                value = 0.0
                for current in range(4):
                    transition1 = (
                        1.0 - theta1
                        if (previous >> 1) == (current >> 1)
                        else theta1
                    )
                    transition2 = (
                        1.0 - theta2
                        if (previous & 1) == (current & 1)
                        else theta2
                    )
                    value += (
                        transition1
                        * transition2
                        * normalized_emission[block + 1, current]
                        * beta[block + 1, current]
                    )
                beta[block, previous] = value
                beta_total += value
            # Backward messages are defined only up to a common factor.  An
            # explicit per-block normalization stays finite for long,
            # nearly deterministic chromosomes.
            beta_total = max(beta_total, _TINY)
            for previous in range(4):
                beta[block, previous] /= beta_total

        gamma = np.empty((n_blocks, 4), dtype=np.float64)
        entropy_sum = 0.0
        for block in range(n_blocks):
            total = 0.0
            for state in range(4):
                gamma[block, state] = alpha[block, state] * beta[block, state]
                total += gamma[block, state]
            total = max(total, _TINY)
            entropy = 0.0
            for state in range(4):
                gamma[block, state] /= total
                if gamma[block, state] > 0.0:
                    entropy -= gamma[block, state] * math.log(gamma[block, state])
            entropy_sum += entropy

        switches1 = 0.0
        switches2 = 0.0
        for block in range(1, n_blocks):
            theta1 = _recombination_fraction(
                block_center[block] - block_center[block - 1],
                parent1_recombination_rate,
            )
            theta2 = _recombination_fraction(
                block_center[block] - block_center[block - 1],
                parent2_recombination_rate,
            )
            denominator = 0.0
            numerator1 = 0.0
            numerator2 = 0.0
            for previous in range(4):
                for current in range(4):
                    changed1 = (previous >> 1) != (current >> 1)
                    changed2 = (previous & 1) != (current & 1)
                    transition1 = theta1 if changed1 else 1.0 - theta1
                    transition2 = theta2 if changed2 else 1.0 - theta2
                    value = (
                        alpha[block - 1, previous]
                        * transition1
                        * transition2
                        * normalized_emission[block, current]
                        * beta[block, current]
                    )
                    denominator += value
                    numerator1 += value * int(changed1)
                    numerator2 += value * int(changed2)
            denominator = max(denominator, _TINY)
            switches1 += numerator1 / denominator
            switches2 += numerator2 / denominator

        for block in range(n_blocks):
            start = block * markers_per_block
            end = min(start + markers_per_block, n_sites)
            for site in range(start, end):
                first_alt = 0.0
                second_alt = 0.0
                for state in range(4):
                    selector1 = state >> 1
                    selector2 = state & 1
                    conditional1, conditional2 = _conditional_allele_marginals(
                        child_likelihoods[child, site],
                        parent_tracks[parent1, site, selector1, 1],
                        parent_tracks[parent2, site, selector2, 1],
                        error_rate,
                    )
                    first_alt += gamma[block, state] * conditional1
                    second_alt += gamma[block, state] * conditional2
                first_alt = min(1.0, max(0.0, first_alt))
                second_alt = min(1.0, max(0.0, second_alt))
                output_tracks[child, site, 0, 0] = 1.0 - first_alt
                output_tracks[child, site, 0, 1] = first_alt
                output_tracks[child, site, 1, 0] = 1.0 - second_alt
                output_tracks[child, site, 1, 1] = second_alt

        log_likelihood[child] = total_loglik
        expected1[child] = switches1
        expected2[child] = switches2
        mean_entropy[child] = entropy_sum / n_blocks

    return output_tracks, log_likelihood, expected1, expected2, mean_entropy


def reconstruct_inherited_tracks(
    child_likelihoods,
    parent_tracks,
    selected_parent_pairs,
    positions,
    parent1_recombination_rate=5e-8,
    parent2_recombination_rate=5e-8,
    error_rate=0.01,
    markers_per_block=100,
    effective_markers_per_block=1.0,
):
    """Reconstruct parental-origin homolog posteriors for specified trios."""
    child = np.ascontiguousarray(child_likelihoods, dtype=np.float64)
    tracks = np.ascontiguousarray(parent_tracks, dtype=np.float64)
    pairs = np.ascontiguousarray(selected_parent_pairs, dtype=np.int64)
    marker_positions = np.ascontiguousarray(positions, dtype=np.int64)
    if child.ndim != 3 or child.shape[2] != 3:
        raise ValueError("child_likelihoods must have shape (children, sites, 3)")
    if tracks.ndim != 4 or tracks.shape[2:] != (2, 2):
        raise ValueError("parent_tracks must have shape (parents, sites, 2, 2)")
    if child.shape[1] != tracks.shape[1]:
        raise ValueError("child and parent tracks must contain the same sites")
    if pairs.shape != (len(child), 2):
        raise ValueError("selected_parent_pairs must have one pair per child")
    if len(pairs) and (np.min(pairs) < 0 or np.max(pairs) >= len(tracks)):
        raise ValueError("selected_parent_pairs contains an invalid index")
    if marker_positions.shape != (child.shape[1],):
        raise ValueError("positions must contain one position per site")
    if len(marker_positions) and np.any(np.diff(marker_positions) <= 0):
        raise ValueError("positions must be strictly increasing")
    for rate in (parent1_recombination_rate, parent2_recombination_rate):
        if not np.isfinite(rate) or rate < 0.0:
            raise ValueError("recombination rates must be finite and non-negative")
    if not 0.0 <= error_rate < 1.0:
        raise ValueError("error_rate must lie in [0, 1)")
    if int(markers_per_block) != markers_per_block or markers_per_block < 1:
        raise ValueError("markers_per_block must be a positive integer")
    if effective_markers_per_block <= 0.0:
        raise ValueError("effective_markers_per_block must be positive")
    if np.any(~np.isfinite(child)) or np.any(child < 0.0):
        raise ValueError("child likelihoods must be finite and non-negative")
    if np.any(~np.isfinite(tracks)) or np.any(tracks < 0.0):
        raise ValueError("parent tracks must be finite and non-negative")
    if not np.allclose(np.sum(tracks, axis=3), 1.0, atol=1e-8):
        raise ValueError("track allele distributions must sum to one")
    result = _reconstruct_selected_pairs_kernel(
        child,
        tracks,
        pairs,
        marker_positions,
        float(parent1_recombination_rate),
        float(parent2_recombination_rate),
        float(error_rate),
        int(markers_per_block),
        float(effective_markers_per_block),
    )
    output_tracks, log_likelihood, expected1, expected2, entropy = result
    return InheritedTrackPosterior(
        log_likelihood,
        output_tracks,
        expected1,
        expected2,
        entropy,
    )
