"""Ragged parental ancestry factors and source posterior calculations."""
from __future__ import annotations


from dataclasses import dataclass
import math
import time
from typing import Any

from numba import prange
import numpy as np
from typing import Iterable

from numba import njit
import haplotype_reconstruction.core.parallel as core_parallel

core_parallel.ensure_numba_registry_warmup()


@dataclass(frozen=True)
class RaggedFounderModel:
    """Unique named trajectories, BACKGROUND, and their site/bin alignment."""

    named_alleles: np.ndarray
    called: np.ndarray
    allele_probability: np.ndarray
    background_alt_probability: np.ndarray
    bin_site_indices: tuple[np.ndarray, ...]
    site_to_bin: np.ndarray
    background_index: int

    @property
    def n_named(self) -> int:
        return int(self.background_index)

    @property
    def n_states(self) -> int:
        return int(self.background_index + 1)

    @property
    def n_sites(self) -> int:
        return int(self.named_alleles.shape[1])

    @property
    def n_bins(self) -> int:
        return len(self.bin_site_indices)


@dataclass(frozen=True)
class RaggedExpectedStructure:
    edge_matched_bins: np.ndarray
    edge_exposed_bins: np.ndarray
    pair_explained_bins: np.ndarray
    pair_exposed_bins: np.ndarray
    structure_total_bins: float


njit = core_parallel.original_njit


@dataclass(frozen=True)
class HammingTransition:
    """Normalized T09 ordered-diplotype weights for every boundary."""

    same: np.ndarray
    one_change: np.ndarray
    two_changes: np.ndarray
    n_states: int
    double_recomb_factor: float

    @property
    def n_boundaries(self) -> int:
        return int(self.same.shape[0])


@njit(cache=True, parallel=True, fastmath=False)
def _structure_features(
    posterior,
    state_anchored,
    pair_first,
    pair_second,
    pair_diagonal,
    pair_anchored,
    available,
):
    """Compress ordered child masses and parent-presence intersections."""

    n_samples, n_bins, n_states, _ = posterior.shape
    n_features = len(pair_first)
    both = np.zeros((n_samples, n_bins), dtype=np.float64)
    any_presence = np.zeros((n_samples, n_bins, n_states), dtype=np.float64)
    full_presence = np.zeros((n_samples, n_bins, n_states), dtype=np.float64)
    pair_mass = np.zeros((n_samples, n_bins, n_features), dtype=np.float64)
    joint_any = np.zeros_like(pair_mass)
    joint_full = np.zeros_like(pair_mass)
    for sample in prange(n_samples):
        if not available[sample]:
            continue
        for block in range(n_bins):
            for first in range(n_states):
                any_mass = 0.0
                full_mass = 0.0
                for other in range(n_states):
                    ordered_mass = (
                        posterior[sample, block, first, other]
                        + posterior[sample, block, other, first]
                    )
                    any_mass += ordered_mass
                    full_mass += ordered_mass * state_anchored[other]
                diagonal_mass = posterior[sample, block, first, first]
                anchored_first = state_anchored[first]
                any_presence[sample, block, first] = anchored_first * (
                    any_mass - diagonal_mass
                )
                full_presence[sample, block, first] = anchored_first * (
                    full_mass - diagonal_mass
                )

            for feature in range(n_features):
                first = pair_first[feature]
                second = pair_second[feature]
                off_diagonal = 1.0 - pair_diagonal[feature]
                mass = posterior[sample, block, first, second]
                mass += off_diagonal * posterior[sample, block, second, first]
                pair_mass[sample, block, feature] = mass
                anchored_pair_mass = mass * pair_anchored[feature]
                joint_any[sample, block, feature] = (
                    pair_diagonal[feature] * any_presence[sample, block, first]
                    + off_diagonal * anchored_pair_mass
                )
                joint_full[sample, block, feature] = (
                    pair_diagonal[feature] * full_presence[sample, block, first]
                    + off_diagonal * anchored_pair_mass
                )
                both[sample, block] += anchored_pair_mass
    return both, any_presence, full_presence, pair_mass, joint_any, joint_full


@dataclass(frozen=True)
class RaggedSourceBatchFactors:
    """Compact posterior-chain factors for all candidate sources."""

    initial_probability: np.ndarray  # candidates, S, S
    right_weight: np.ndarray  # candidates, boundaries, S, S
    available: np.ndarray  # candidates
    informative_site_count: np.ndarray  # candidates
    transition: Any
    robustness_epsilon: float
    preparation_seconds: float

    @property
    def n_candidates(self) -> int:
        return int(self.initial_probability.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.initial_probability.shape[1])

    @property
    def n_bins(self) -> int:
        return int(self.right_weight.shape[1] + 1)

    @property
    def retained_bytes(self) -> int:
        return int(
            self.initial_probability.nbytes
            + self.right_weight.nbytes
            + self.available.nbytes
            + self.informative_site_count.nbytes
        )


def build_ragged_founder_model(
    named_alleles: np.ndarray,
    bin_site_indices: Iterable[np.ndarray] | None=None,
) -> RaggedFounderModel:
    """Build T09 allele priors from unique ``{-1,0,1}`` trajectory classes.

    Beta(1,1) smoothing is applied independently at every site across the
    unique named classes.  Sites with no named call are rejected because T09
    excludes them before painting: retaining them would create ancestry
    evidence solely from diagonal versus off-diagonal latent structure.
    """

    alleles = np.asarray(named_alleles, dtype=np.int8)
    if alleles.ndim != 2 or alleles.shape[0] < 1 or alleles.shape[1] < 1:
        raise ValueError("named_alleles must have shape (named_states, sites)")
    if np.any(~np.isin(alleles, (-1, 0, 1))):
        raise ValueError("named alleles must be -1 (missing), 0, or 1")
    called = alleles >= 0
    n_called = np.sum(called, axis=0)
    if np.any(n_called == 0):
        raise ValueError("every retained T09 site must have a named founder call")
    alt_called = np.sum(np.where(called, alleles, 0), axis=0)
    frequency = (1.0 + alt_called) / (2.0 + n_called)
    q = np.where(called, alleles, frequency[None,:]).astype(np.float64)

    n_sites = alleles.shape[1]
    if bin_site_indices is None:
        bins = tuple(np.array([site], dtype=np.int64) for site in range(n_sites))
    else:
        bins = tuple(np.asarray(index, dtype=np.int64).copy() for index in bin_site_indices)
        if not bins or any(index.ndim != 1 or len(index) == 0 for index in bins):
            raise ValueError("bin_site_indices must contain non-empty 1D arrays")
    concatenated = np.concatenate(bins)
    if (
        concatenated.shape != (n_sites,)
        or np.any(concatenated < 0)
        or np.any(concatenated >= n_sites)
        or not np.array_equal(np.sort(concatenated), np.arange(n_sites))
    ):
        raise ValueError("bin_site_indices must partition every site exactly once")
    site_to_bin = np.empty(n_sites, dtype=np.int64)
    for bin_index, sites in enumerate(bins):
        site_to_bin[sites] = bin_index

    return RaggedFounderModel(
        named_alleles=alleles.copy(),
        called=called,
        allele_probability=q,
        background_alt_probability=frequency.astype(np.float64),
        bin_site_indices=bins,
        site_to_bin=site_to_bin,
        background_index=int(alleles.shape[0]),
    )


@njit(cache=True, inline="always")
def _unit(value):
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class RaggedSourceBatchWorkspace:
    """Optional large arrays that may be reused by an immediate M2 call."""

    denominator_reciprocal: np.ndarray
    candidate_transmitted_alt_probability: np.ndarray


def build_t09_hamming_transition(
    bin_centers: np.ndarray,
    n_states: int,
    *,
    recomb_rate: float=1e-8,
    switch_penalty_per_snp: float=1.0,
    snps_per_bin: int=100,
    double_recomb_factor: float=1.5,
    chromosome_map=None,
) -> HammingTransition:
    """Build T09's normalized non-separable source transition."""

    centers = np.asarray(bin_centers, dtype=np.float64)
    if centers.ndim != 1 or len(centers) < 1 or np.any(~np.isfinite(centers)):
        raise ValueError("bin_centers must be a non-empty finite vector")
    if n_states < 2:
        raise ValueError("the ragged model requires a named state plus BACKGROUND")
    if not math.isfinite(recomb_rate) or recomb_rate < 0.0:
        raise ValueError("recomb_rate must be finite and non-negative")
    if not math.isfinite(switch_penalty_per_snp) or switch_penalty_per_snp < 0.0:
        raise ValueError("switch_penalty_per_snp must be finite and non-negative")
    if isinstance(snps_per_bin, bool) or int(snps_per_bin) != snps_per_bin or snps_per_bin < 1:
        raise ValueError("snps_per_bin must be a positive integer")
    if not math.isfinite(double_recomb_factor) or double_recomb_factor <= 0.0:
        raise ValueError("double_recomb_factor must be finite and positive")

    same = np.empty(len(centers) - 1, dtype=np.float64)
    one = np.empty_like(same)
    two = np.empty_like(same)
    log_count = math.log(float(n_states - 1))
    genetic_distances = None
    if chromosome_map is not None:
        recomb_rate = chromosome_map.fallback_rate_per_bp
        if chromosome_map.has_map:
            genetic_distances = chromosome_map.interval_morgans(centers[:-1], centers[1:])
    for boundary in range(len(same)):
        distance = max(1.0, float(centers[boundary + 1] - centers[boundary]))
        theta = min(0.5, max(1e-15, distance * recomb_rate))
        if genetic_distances is not None:
            theta = min(0.5, max(1e-15, genetic_distances[boundary]))
        log_switch = math.log(theta) - float(switch_penalty_per_snp) * int(snps_per_bin)
        log_weights = np.array(
            [
                2.0 * math.log1p(-theta),
                log_switch + math.log1p(-theta) - log_count,
                float(double_recomb_factor) * log_switch - 2.0 * log_count,
            ],
            dtype=np.float64,
        )
        multiplicity = np.array(
            [1.0, 2.0 * (n_states - 1), float((n_states - 1) ** 2)],
            dtype=np.float64,
        )
        maximum = float(np.max(log_weights))
        weights = np.exp(log_weights - maximum)
        normalizer = float(np.dot(multiplicity, weights))
        same[boundary], one[boundary], two[boundary] = weights / normalizer
    return HammingTransition(same, one, two, int(n_states), float(double_recomb_factor))


@njit(cache=True, parallel=True)
def _edge_kernel(
    pair_mass,
    pair_first,
    pair_second,
    pair_has_anchor,
    pair_anchored,
    available,
    diagonal_required,
    edge_first,
    edge_second,
    both,
    any_presence,
    full_presence,
    joint_any,
    joint_full,
):
    n_samples, n_bins, n_features = pair_mass.shape
    matched = np.zeros((n_samples, n_samples), dtype=np.float64)
    exposed = np.zeros((n_samples, n_samples), dtype=np.float64)
    for child in prange(n_samples):
        if available[child] and diagonal_required[child]:
            diagonal = 0.0
            for block in range(n_bins):
                present = 0.0
                for feature in range(n_features):
                    present += (
                        pair_mass[child, block, feature]
                        * pair_has_anchor[feature]
                    )
                diagonal += _unit(present)
            matched[child, child] = diagonal
            exposed[child, child] = diagonal

    # Each requested upper-triangle edge is an equal-sized parallel task.
    # This avoids the triangular tail produced by parallelizing over children.
    for edge in prange(len(edge_first)):
        child = edge_first[edge]
        parent = edge_second[edge]
        edge_match = 0.0
        edge_exposure = 0.0
        for block in range(n_bins):
            block_match = 0.0
            full_nonmatch = 0.0
            for feature in range(n_features):
                first = pair_first[feature]
                second = pair_second[feature]
                child_mass = pair_mass[child, block, feature]
                match_probability = (
                    any_presence[parent, block, first]
                    + any_presence[parent, block, second]
                    - joint_any[parent, block, feature]
                )
                full_nonmatch_probability = pair_anchored[feature] * (
                    both[parent, block]
                    - full_presence[parent, block, first]
                    - full_presence[parent, block, second]
                    + joint_full[parent, block, feature]
                )
                block_match += child_mass * match_probability
                full_nonmatch += child_mass * full_nonmatch_probability
            block_match = _unit(block_match)
            edge_match += block_match
            # This is a disjoint probability mass. Inclusion-exclusion can
            # cancel to a tiny negative value; retain its nonnegative bound
            # before adding it, so exposure cannot round below the match.
            edge_exposure += _unit(
                block_match + max(0.0, full_nonmatch))
        matched[child, parent] = edge_match
        matched[parent, child] = edge_match
        exposed[child, parent] = edge_exposure
        exposed[parent, child] = edge_exposure
    return matched, exposed


@dataclass(frozen=True)
class RaggedSourceBatchScores:
    """Conditional M0/M1/M2 scores plus bounded-memory diagnostics."""

    zero_observed: np.ndarray
    one_observed: np.ndarray
    two_observed: np.ndarray
    candidate_source_available: np.ndarray
    candidate_source_informative_site_count: np.ndarray
    child_informative_site_count: np.ndarray
    m2_active_trio_count: int
    m2_reduced_trio_count: int
    m2_batch_size: int
    m2_batch_count: int
    m2_bytes_per_active_task: int
    peak_m2_working_bytes: int
    transmission_probability_bytes: int
    transmission_preparation_seconds: float
    m0_scoring_seconds: float
    m1_scoring_seconds: float
    m2_scoring_seconds: float
    reusable_workspace: RaggedSourceBatchWorkspace | None = None


@njit(cache=True, parallel=True)
def _pair_kernel(
    pair_mass,
    pair_first,
    pair_second,
    pair_anchored,
    available,
    trios,
    both,
    any_presence,
    full_presence,
    joint_any,
    joint_full,
):
    n_bins = pair_mass.shape[1]
    n_features = pair_mass.shape[2]
    explained = np.zeros(len(trios), dtype=np.float64)
    exposed = np.zeros(len(trios), dtype=np.float64)
    for row in prange(len(trios)):
        child = int(trios[row, 0])
        parent1 = int(trios[row, 1])
        parent2 = int(trios[row, 2])
        if (
            not available[child]
            or not available[parent1]
            or not available[parent2]
        ):
            continue
        for block in range(n_bins):
            block_explained = 0.0
            full_nonexplained = 0.0
            for feature in range(n_features):
                first = pair_first[feature]
                second = pair_second[feature]
                child_mass = pair_mass[child, block, feature]
                assignment = (
                    any_presence[parent1, block, first]
                    * any_presence[parent2, block, second]
                    + any_presence[parent1, block, second]
                    * any_presence[parent2, block, first]
                    - joint_any[parent1, block, feature]
                    * joint_any[parent2, block, feature]
                )
                block_explained += child_mass * assignment
                full_assignment = (
                    full_presence[parent1, block, first]
                    * full_presence[parent2, block, second]
                    + full_presence[parent1, block, second]
                    * full_presence[parent2, block, first]
                    - joint_full[parent1, block, feature]
                    * joint_full[parent2, block, feature]
                )
                full_nonexplained += (
                    child_mass
                    * pair_anchored[feature]
                    * (
                        both[parent1, block] * both[parent2, block]
                        - full_assignment
                    )
                )
            block_explained = _unit(block_explained)
            explained[row] += block_explained
            exposed[row] += _unit(
                block_explained + max(0.0, full_nonexplained))
    return explained, exposed


@njit(cache=True, fastmath=False)
def _hamming_contract_workspace(
    matrix, same, one, two, output, rows, columns
):
    states = matrix.shape[0]
    rows[:] = 0.0
    columns[:] = 0.0
    total = 0.0
    for first in range(states):
        for second in range(states):
            value = matrix[first, second]
            rows[first] += value
            columns[second] += value
            total += value
    for first in range(states):
        for second in range(states):
            cell = matrix[first, second]
            one_mass = rows[first] + columns[second] - 2.0 * cell
            two_mass = total - rows[first] - columns[second] + cell
            output[first, second] = (
                same * cell + one * one_mass + two * two_mass
            )


def _validated_marginals(values: Any) -> np.ndarray:
    posterior = np.asarray(values, dtype=np.float64)
    if (
        posterior.ndim != 4
        or min(posterior.shape) < 1
        or posterior.shape[3] != posterior.shape[2]
        or np.any(~np.isfinite(posterior))
        or np.any(posterior < 0.0)
    ):
        raise ValueError(
            "posterior_marginals must have finite non-negative shape "
            "(samples, bins, states, states)"
        )
    totals = np.sum(posterior, axis=(2, 3))
    if np.any(~np.isclose(totals, 1.0, rtol=0.0, atol=1e-10)):
        raise ValueError("every sample/bin posterior marginal must sum to one")
    # Normalization is part of the input contract and has just been checked.
    # Avoid materializing another full (N, B, S, S) tensor when the caller
    # already supplies the native contiguous float64 representation.
    return np.ascontiguousarray(posterior, dtype=np.float64)


@njit(cache=True, fastmath=False)
def _pedigree_ragged_source_batch_hamming_contract(matrix, same, one, two, output):
    """Apply one symmetric ordered-diplotype Hamming transition in O(S**2)."""

    states = matrix.shape[0]
    rows = np.empty(states, dtype=np.float64)
    columns = np.empty(states, dtype=np.float64)
    _hamming_contract_workspace(
        matrix, same, one, two, output, rows, columns
    )


@njit(cache=True, inline="always", fastmath=False)
def _pedigree_ragged_structure_hamming_contract(matrix, same, one, two, output, rows, columns):
    """Apply a symmetric ordered-diplotype Hamming kernel in O(S**2)."""

    states = matrix.shape[0]
    rows[:] = 0.0
    columns[:] = 0.0
    total = 0.0
    for first in range(states):
        for second in range(states):
            value = matrix[first, second]
            rows[first] += value
            columns[second] += value
            total += value
    for first in range(states):
        for second in range(states):
            cell = matrix[first, second]
            output[first, second] = (
                same * cell
                + one * (rows[first] + columns[second] - 2.0 * cell)
                + two * (total - rows[first] - columns[second] + cell)
            )


@njit(cache=True, parallel=True, fastmath=False)
def _infer_factor_kernel(emission, same, one, two, requested_available):
    candidates, bins, states, _ = emission.shape
    initial = np.zeros((candidates, states, states), dtype=np.float64)
    right = np.ones(
        (candidates, max(0, bins - 1), states, states), dtype=np.float64
    )
    numerically_valid = np.ones(candidates, dtype=np.bool_)
    for candidate in prange(candidates):
        if not requested_available[candidate]:
            initial[candidate] = 1.0 / float(states * states)
            continue
        beta = np.ones((states, states), dtype=np.float64)
        contraction = np.empty((states, states), dtype=np.float64)
        valid = True
        for boundary in range(bins - 2, -1, -1):
            maximum = -np.inf
            for first in range(states):
                for second in range(states):
                    maximum = max(
                        maximum, emission[candidate, boundary + 1, first, second]
                    )
            if not np.isfinite(maximum):
                valid = False
                break
            maximum_right = 0.0
            for first in range(states):
                for second in range(states):
                    value = (
                        math.exp(
                            emission[candidate, boundary + 1, first, second]
                            - maximum
                        )
                        * beta[first, second]
                    )
                    right[candidate, boundary, first, second] = value
                    maximum_right = max(maximum_right, value)
            if maximum_right <= 0.0:
                valid = False
                break
            right[candidate, boundary] /= maximum_right
            _pedigree_ragged_source_batch_hamming_contract(
                right[candidate, boundary], same[boundary], one[boundary],
                two[boundary], contraction,
            )
            maximum_beta = np.max(contraction)
            if maximum_beta <= 0.0 or not np.isfinite(maximum_beta):
                valid = False
                break
            beta = contraction / maximum_beta

        if valid:
            maximum = np.max(emission[candidate, 0])
            total = 0.0
            for first in range(states):
                for second in range(states):
                    value = (
                        math.exp(emission[candidate, 0, first, second] - maximum)
                        * beta[first, second]
                    )
                    initial[candidate, first, second] = value
                    total += value
            if total <= 0.0 or not np.isfinite(total):
                valid = False
            else:
                initial[candidate] /= total
                # Unphased source GLs have an arbitrary homolog gauge.  Remove
                # only floating-point asymmetry; do not alter unordered mass.
                for first in range(states):
                    for second in range(first + 1, states):
                        value = 0.5 * (
                            initial[candidate, first, second]
                            + initial[candidate, second, first]
                        )
                        initial[candidate, first, second] = value
                        initial[candidate, second, first] = value
        if not valid:
            numerically_valid[candidate] = False
            initial[candidate] = 1.0 / float(states * states)
            right[candidate] = 1.0

    return initial, right, numerically_valid


@njit(cache=True, parallel=True, fastmath=False)
def _source_posterior_marginal_kernel(initial, right, same, one, two):
    candidates, states, _ = initial.shape
    boundaries = right.shape[1]
    posterior = np.empty(
        (candidates, boundaries + 1, states, states), dtype=np.float64
    )
    valid = np.ones(candidates, dtype=np.bool_)
    for candidate in prange(candidates):
        rows = np.empty(states, dtype=np.float64)
        columns = np.empty(states, dtype=np.float64)
        workspace = np.empty((states, states), dtype=np.float64)

        total = 0.0
        for first in range(states):
            for second in range(states):
                value = initial[candidate, first, second]
                posterior[candidate, 0, first, second] = value
                total += value
        for first in range(states):
            for second in range(states):
                posterior[candidate, 0, first, second] /= total

        for boundary in range(boundaries):
            scale = 0.0
            for first in range(states):
                for second in range(states):
                    scale = max(scale, right[candidate, boundary, first, second])
            if scale <= 0.0 or not np.isfinite(scale):
                valid[candidate] = False
                break

            # K(u,v) is unchanged by a common rescaling of r(v). Store that
            # scaled r temporarily in the next output slice, then reuse the
            # sole S-by-S workspace first for its denominator and then for
            # p_b(u) / denominator(u).
            for first in range(states):
                for second in range(states):
                    posterior[candidate, boundary + 1, first, second] = (
                        right[candidate, boundary, first, second] / scale
                    )
            _pedigree_ragged_structure_hamming_contract(
                posterior[candidate, boundary + 1],
                same[boundary],
                one[boundary],
                two[boundary],
                workspace,
                rows,
                columns,
            )
            for first in range(states):
                for second in range(states):
                    denominator = workspace[first, second]
                    if denominator <= 0.0 or not np.isfinite(denominator):
                        valid[candidate] = False
                    else:
                        workspace[first, second] = (
                            posterior[candidate, boundary, first, second]
                            / denominator
                        )
            if not valid[candidate]:
                break

            _pedigree_ragged_structure_hamming_contract(
                workspace,
                same[boundary],
                one[boundary],
                two[boundary],
                posterior[candidate, boundary + 1],
                rows,
                columns,
            )
            total = 0.0
            for first in range(states):
                for second in range(states):
                    value = (
                        posterior[candidate, boundary + 1, first, second]
                        * (right[candidate, boundary, first, second] / scale)
                    )
                    posterior[candidate, boundary + 1, first, second] = value
                    total += value
            if total <= 0.0 or not np.isfinite(total):
                valid[candidate] = False
                break
            for first in range(states):
                for second in range(states):
                    posterior[candidate, boundary + 1, first, second] /= total
    return posterior, valid


def infer_candidate_source_factors_batch(
    binned_log_emission: np.ndarray,
    transition,
    informative_site_count: np.ndarray,
    *,
    minimum_informative_sites: int=1,
    robustness_epsilon: float=0.01,
) -> RaggedSourceBatchFactors:
    """Infer exact source posterior-chain factors from precomputed T09 emissions.

    ``binned_log_emission`` accepts the native T09 layout
    ``(candidates, S*S, bins)`` or the explicit layout
    ``(candidates, bins, S, S)``.  The factors retain O(N*B*S**2) numbers.
    Candidate availability is evidence-based and is not inferred from a MAP
    release threshold.
    """

    started = time.perf_counter()
    states = int(transition.n_states)
    value = np.asarray(binned_log_emission, dtype=np.float64)
    if value.ndim == 3:
        if value.shape[1] != states * states:
            raise ValueError("T09 emissions must have S*S diplotype rows")
        value = np.transpose(value, (0, 2, 1)).reshape(
            value.shape[0], value.shape[2], states, states
        )
    elif value.ndim == 4:
        if value.shape[2:] != (states, states):
            raise ValueError("explicit emissions must have shape (N, B, S, S)")
    else:
        raise ValueError("binned emissions must have rank three or four")
    if value.shape[1] < 1:
        raise ValueError("at least one bin is required")
    if np.any(np.isnan(value)) or np.any(np.isposinf(value)):
        raise ValueError("emissions may contain finite values or -inf only")
    counts = np.asarray(informative_site_count, dtype=np.int64)
    if counts.shape != (value.shape[0],) or np.any(counts < 0):
        raise ValueError("informative_site_count must be nonnegative shape (N,)")
    if (
        isinstance(minimum_informative_sites, bool)
        or int(minimum_informative_sites) != minimum_informative_sites
        or minimum_informative_sites < 1
    ):
        raise ValueError("minimum_informative_sites must be a positive integer")
    if not np.isfinite(robustness_epsilon) or not 0.0 <= robustness_epsilon <= 1.0:
        raise ValueError("robustness_epsilon must lie in [0, 1]")
    boundaries = value.shape[1] - 1
    arrays = (
        np.asarray(transition.same, dtype=np.float64),
        np.asarray(transition.one_change, dtype=np.float64),
        np.asarray(transition.two_changes, dtype=np.float64),
    )
    if any(array.shape != (boundaries,) for array in arrays):
        raise ValueError("transition boundary count disagrees with emissions")
    if any(np.any(~np.isfinite(array)) or np.any(array < 0.0) for array in arrays):
        raise ValueError("transition weights must be finite and nonnegative")
    row_mass = arrays[0] + 2.0 * (states - 1) * arrays[1] + (states - 1) ** 2 * arrays[2]
    if not np.allclose(row_mass, 1.0, rtol=2e-13, atol=2e-15):
        raise ValueError("Hamming transition rows must be normalized")
    available = counts >= int(minimum_informative_sites)
    initial, right, valid = _infer_factor_kernel(
        np.ascontiguousarray(value), *(np.ascontiguousarray(a) for a in arrays),
        np.ascontiguousarray(available),
    )
    if np.any(available & ~valid):
        raise FloatingPointError("candidate source factor inference lost all supported mass")
    return RaggedSourceBatchFactors(
        initial_probability=initial,
        right_weight=right,
        available=available,
        informative_site_count=counts.copy(),
        transition=transition,
        robustness_epsilon=float(robustness_epsilon),
        preparation_seconds=time.perf_counter() - started,
    )


def source_posterior_marginals(
    factors: "RaggedSourceBatchFactors",
) -> np.ndarray:
    """Expand compact source-chain factors to exact per-bin marginals.

    For boundary b, the retained factors define
    K_b(u,v) = A_b(u,v) r_b(v) / sum_w A_b(u,w) r_b(w).
    Symmetry of the Hamming-category transition permits both contractions in
    O(S**2), so the bridge costs O(N*B*S**2) time and only one S-by-S workspace
    per parallel candidate in addition to its required dense output.

    Availability is deliberately not a gate here: evidence-free candidate
    rows still receive normalized marginals and are gated later by
    sample_available when expected structure is accumulated.
    """


    if not isinstance(factors, RaggedSourceBatchFactors):
        raise ValueError("factors must be RaggedSourceBatchFactors")

    initial = np.asarray(factors.initial_probability, dtype=np.float64)
    right = np.asarray(factors.right_weight, dtype=np.float64)
    if (
        initial.ndim != 3
        or initial.shape[0] < 1
        or initial.shape[1] < 2
        or initial.shape[2] != initial.shape[1]
    ):
        raise ValueError("factor initial probabilities must have shape (N, S, S)")
    candidates, states, _ = initial.shape
    if right.ndim != 4 or right.shape[0] != candidates or right.shape[2:] != (
        states,
        states,
    ):
        raise ValueError("factor right weights must have shape (N, B-1, S, S)")
    boundaries = right.shape[1]
    if (
        np.any(~np.isfinite(initial))
        or np.any(initial < 0.0)
        or np.any(~np.isfinite(right))
        or np.any(right < 0.0)
    ):
        raise ValueError("factor probabilities must be finite and nonnegative")
    if not np.allclose(
        np.sum(initial, axis=(1, 2)), 1.0, rtol=2e-13, atol=2e-15
    ):
        raise ValueError("each factor initial distribution must sum to one")

    available = np.asarray(factors.available)
    counts = np.asarray(factors.informative_site_count)
    if available.dtype != np.bool_ or available.shape != (candidates,):
        raise ValueError("factor availability must be a boolean vector of shape (N,)")
    if (
        counts.shape != (candidates,)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0)
    ):
        raise ValueError("factor informative-site counts must be nonnegative integers")
    try:
        epsilon = float(factors.robustness_epsilon)
        preparation_seconds = float(factors.preparation_seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("factor scalar metadata are invalid") from exc
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("factor robustness epsilon must lie in [0, 1]")
    if not math.isfinite(preparation_seconds) or preparation_seconds < 0.0:
        raise ValueError("factor preparation time must be finite and nonnegative")

    transition = factors.transition
    try:
        transition_states = int(transition.n_states)
        same = np.asarray(transition.same, dtype=np.float64)
        one = np.asarray(transition.one_change, dtype=np.float64)
        two = np.asarray(transition.two_changes, dtype=np.float64)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("factor transition is malformed") from exc
    if transition_states != states or any(
        array.shape != (boundaries,) for array in (same, one, two)
    ):
        raise ValueError("factor transition state or boundary count is wrong")
    if any(
        np.any(~np.isfinite(array)) or np.any(array < 0.0)
        for array in (same, one, two)
    ):
        raise ValueError("factor transition weights must be finite and nonnegative")
    row_mass = same + 2.0 * (states - 1) * one + (states - 1) ** 2 * two
    if not np.allclose(row_mass, 1.0, rtol=2e-13, atol=2e-15):
        raise ValueError("factor Hamming transition rows must be normalized")

    posterior, valid = _source_posterior_marginal_kernel(
        np.ascontiguousarray(initial),
        np.ascontiguousarray(right),
        np.ascontiguousarray(same),
        np.ascontiguousarray(one),
        np.ascontiguousarray(two),
    )
    if np.any(~valid):
        raise ValueError("factor transition has a source state with zero supported mass")
    return posterior


def _normalise_gl(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"{name} must have shape (samples, sites, 3)")
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = np.sum(value, axis=2, keepdims=True)
    result = np.full(value.shape, 1.0 / 3.0, dtype=np.float64)
    np.divide(value, total, out=result, where=total > 0.0)
    return np.ascontiguousarray(result)


def _validated_trios(values: Any, n_samples: int) -> np.ndarray:
    raw = np.asarray(values)
    if (
        raw.ndim != 2
        or raw.shape[1] != 3
        or np.any(~np.isfinite(raw))
        or np.any(raw != np.floor(raw))
    ):
        raise ValueError("trios must be an integer array with shape (rows, 3)")
    trios = np.ascontiguousarray(raw, dtype=np.int64)
    if len(trios):
        if np.any(trios < 0) or np.any(trios >= n_samples):
            raise ValueError("trio index lies outside the sample array")
        if (
            np.any(trios[:, 0] == trios[:, 1])
            or np.any(trios[:, 0] == trios[:, 2])
            or np.any(trios[:, 1] == trios[:, 2])
        ):
            raise ValueError("trios cannot contain self or duplicate parents")
    return trios


def _child_likelihood_coefficients(child_gl, mismatch):
    """Precompute L(a,b) = c0 + c1*(a+b) + c2*a*b."""

    g0 = child_gl[:,:, 0]
    g1 = child_gl[:,:, 1]
    g2 = child_gl[:,:, 2]
    linear_genotype = g1 - g0
    quadratic_genotype = g0 - 2.0 * g1 + g2
    inherited_scale = 1.0 - 2.0 * mismatch
    coefficient = np.empty(child_gl.shape, dtype=np.float64)
    coefficient[:,:, 0] = 3.0 * (
        g0
        + 2.0 * mismatch * linear_genotype
        + mismatch * mismatch * quadratic_genotype
    )
    coefficient[:,:, 1] = 3.0 * inherited_scale * (
        linear_genotype + mismatch * quadratic_genotype
    )
    coefficient[:,:, 2] = (
        3.0 * inherited_scale * inherited_scale * quadratic_genotype
    )
    return np.ascontiguousarray(coefficient)


def posterior_expected_structure(
    posterior_marginals: Any,
    anchored_states: Any,
    required_edges: Any,
    trios: Any,
    *,
    sample_available: Any | None=None,
) -> RaggedExpectedStructure:
    """Integrate component-local edge and M2 compatibility.

    Pooled named equivalence classes are anchored states. Unanchored named
    trajectories and BACKGROUND must be marked false. sample_available gates
    prior-only samples so they contribute zero exposure.
    """

    posterior = _validated_marginals(posterior_marginals)
    n_samples, n_bins, n_states, _ = posterior.shape
    anchored = np.asarray(anchored_states)
    if anchored.dtype != np.bool_ or anchored.shape != (n_states,):
        raise ValueError("anchored_states must be a boolean state vector")
    anchored = np.ascontiguousarray(anchored)

    edges = np.asarray(required_edges)
    if (
        edges.dtype != np.bool_
        or edges.shape != (n_samples, n_samples)
        or not np.array_equal(edges, edges.T)
    ):
        raise ValueError("required_edges must be a symmetric boolean matrix")
    edges = np.ascontiguousarray(edges)

    if sample_available is None:
        available = np.ones(n_samples, dtype=np.bool_)
    else:
        available = np.asarray(sample_available)
        if available.dtype != np.bool_ or available.shape != (n_samples,):
            raise ValueError("sample_available must be a boolean sample vector")
        available = np.ascontiguousarray(available)

    trio_array = _validated_trios(trios, n_samples)

    pair_first, pair_second = np.triu_indices(n_states)
    pair_first = np.ascontiguousarray(pair_first, dtype=np.int64)
    pair_second = np.ascontiguousarray(pair_second, dtype=np.int64)
    state_anchored = np.ascontiguousarray(anchored, dtype=np.float64)
    pair_diagonal = np.ascontiguousarray(
        pair_first == pair_second, dtype=np.float64
    )
    pair_anchored = np.ascontiguousarray(
        state_anchored[pair_first] * state_anchored[pair_second]
    )
    pair_has_anchor = np.ascontiguousarray(
        np.maximum(state_anchored[pair_first], state_anchored[pair_second])
    )

    active_edges = np.triu(
        edges & available[:, None] & available[None,:], k=1
    )
    edge_first, edge_second = np.nonzero(active_edges)
    edge_first = np.ascontiguousarray(edge_first, dtype=np.int64)
    edge_second = np.ascontiguousarray(edge_second, dtype=np.int64)
    diagonal_required = np.ascontiguousarray(np.diag(edges), dtype=np.bool_)

    features = _structure_features(
        posterior,
        state_anchored,
        pair_first,
        pair_second,
        pair_diagonal,
        pair_anchored,
        available,
    )
    both, any_presence, full_presence, pair_mass, joint_any, joint_full = features
    edge_matched, edge_exposed = _edge_kernel(
        pair_mass,
        pair_first,
        pair_second,
        pair_has_anchor,
        pair_anchored,
        available,
        diagonal_required,
        edge_first,
        edge_second,
        both,
        any_presence,
        full_presence,
        joint_any,
        joint_full,
    )
    pair_explained, pair_exposed = _pair_kernel(
        pair_mass,
        pair_first,
        pair_second,
        pair_anchored,
        available,
        trio_array,
        both,
        any_presence,
        full_presence,
        joint_any,
        joint_full,
    )
    return RaggedExpectedStructure(
        edge_matched,
        edge_exposed,
        pair_explained,
        pair_exposed,
        float(n_bins),
    )


@njit(cache=True, parallel=True, fastmath=False)
def _compact_observed_sites_kernel(child_observed, bin_start, bin_stop):
    """Build child/bin observed-site CSR without Python-sized tiny arrays."""

    children = child_observed.shape[0]
    bins = len(bin_start)
    start = np.empty((children, bins), dtype=np.int64)
    stop = np.empty((children, bins), dtype=np.int64)
    for task in prange(children * bins):
        child = task // bins
        block = task - child * bins
        count = 0
        for site in range(int(bin_start[block]), int(bin_stop[block])):
            count += int(child_observed[child, site])
        stop[child, block] = count

    cursor = 0
    for child in range(children):
        for block in range(bins):
            count = stop[child, block]
            start[child, block] = cursor
            cursor += count
            stop[child, block] = cursor

    sites = np.empty(cursor, dtype=np.int64)
    for child in prange(children):
        for block in range(bins):
            destination = start[child, block]
            for site in range(int(bin_start[block]), int(bin_stop[block])):
                if child_observed[child, site]:
                    sites[destination] = site
                    destination += 1
    return sites, start, stop


def _compact_observed_sites(child_observed, bin_start, bin_stop):
    """Flatten observed sites once so state loops contain no missingness branch."""

    return _compact_observed_sites_kernel(
        np.ascontiguousarray(child_observed),
        np.ascontiguousarray(bin_start, dtype=np.int64),
        np.ascontiguousarray(bin_stop, dtype=np.int64),
    )


def _compact_model_arrays(model, selected_site_indices):
    states = int(model.n_states)
    sites = int(model.n_sites)
    bins = int(model.n_bins)
    model_bins = [np.asarray(index, dtype=np.int64) for index in model.bin_site_indices]
    if len(model_bins) != bins or any(index.ndim != 1 for index in model_bins):
        raise ValueError("model bin indices are malformed")
    full_order = np.concatenate(model_bins) if model_bins else np.empty(0, dtype=np.int64)
    if len(full_order) != sites or not np.array_equal(np.sort(full_order), np.arange(sites)):
        raise ValueError("model bins must partition all sites exactly once")
    if selected_site_indices is None:
        selected = np.arange(sites, dtype=np.int64)
    else:
        selected = np.asarray(selected_site_indices, dtype=np.int64)
        if (
            selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= sites)
            or len(np.unique(selected)) != len(selected)
        ):
            raise ValueError("selected site indices must be unique valid model sites")
    slot_for_site = {int(site): slot for slot, site in enumerate(selected)}
    order_parts = [
        np.asarray(
            [slot_for_site[int(site)] for site in model_bin if int(site) in slot_for_site],
            dtype=np.int64,
        )
        for model_bin in model_bins
    ]
    order = np.concatenate(order_parts) if order_parts else np.empty(0, dtype=np.int64)
    site_order = selected[order]
    sizes = np.asarray([len(index) for index in order_parts], dtype=np.int64)
    stop = np.cumsum(sizes)
    bin_start = stop - sizes
    state_alt = np.empty((states, len(selected)), dtype=np.float64)
    state_alt[:-1] = np.asarray(model.allele_probability, dtype=np.float64)[:, site_order]
    state_alt[-1] = np.asarray(model.background_alt_probability, dtype=np.float64)[site_order]
    if int(model.background_index) != states - 1:
        raise ValueError("the ragged BACKGROUND state must be last")
    if np.any(~np.isfinite(state_alt)) or np.any((state_alt < 0.0) | (state_alt > 1.0)):
        raise ValueError("state allele probabilities must lie in [0, 1]")
    return order, np.ascontiguousarray(state_alt), bin_start, stop


def _probability_matrix(value, rows, columns, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 0:
        value = np.full((rows, columns), float(value), dtype=np.float64)
    elif value.ndim == 1 and value.shape == (columns,):
        value = np.broadcast_to(value, (rows, columns)).copy()
    if value.shape != (rows, columns) or np.any(~np.isfinite(value)) or np.any(
        (value < 0.0) | (value > 1.0)
    ):
        raise ValueError(f"{name} must be scalar, (boundaries,), or requested matrix in [0, 1]")
    return np.ascontiguousarray(value)
