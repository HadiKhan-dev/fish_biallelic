"""Post-L4 whole-bin factorial cavity refinement for the final founder panel.

For a focal site, the complete HMM bin containing that site is withheld while
ordered diploid founder-state probabilities are inferred from the rest of the
chromosome. Raw focal genotype likelihoods then score an immutable-baseline
family containing the current founder alleles, every single-founder flip, the
complete complement, and every single-founder flip of that complement.

The exact 2*K+2 family is evaluated in O(samples*K**2) work per site using
founder-pair dosage updates; no exponential candidate array is constructed.
A non-current configuration must exceed the current by a conservative
chromosome-wide multiplicity penalty. The returned copy is the canonical final
post-L4 panel: painting, pedigree inference and phase correction must all use
that same panel and its matched painting checkpoint.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Callable, Optional, Sequence, Union

import numba
import numpy as np
from numba import njit, prange

from bhd_config import (
    TERMINAL_CAVITY_EMISSION_UNIFORM_MIX,
    TERMINAL_CAVITY_LOG_EMISSION_FLOOR,
    TERMINAL_CAVITY_RHO,
    TERMINAL_CAVITY_SAMPLE_CHUNK_SIZE,
    TERMINAL_CAVITY_SITE_CHUNK_SIZE,
    TERMINAL_CAVITY_SNPS_PER_BIN,
)
from bhd_mode_canonicalization import canonicalize_binary_panel
from bhd_results import BlockResult, BlockResults


ThreadCount = Optional[Union[int, Callable[[], int]]]

TERMINAL_CAVITY_MODEL_REVISION = "whole_bin_factorial_cavity_2k_plus_2_v1"


@dataclass(frozen=True)
class TerminalCavityResult:
    """One immutable-baseline exact-family refinement result.

    Candidate encoding is in caller founder order: 0=current, 1..K=single
    flips, K+1=complement, and K+2..2K+1=single flips of the complement.
    ``margin_nats`` is the best non-current candidate's raw log-likelihood
    minus the current candidate's raw log-likelihood and the joint-site
    penalty. A candidate is accepted only when this value is strictly positive.

    ``selected_flip_mask`` is the accepted site-by-founder XOR mask relative
    to the input. ``canonical_order`` maps canonical internal row indices to
    caller row indices, while ``canonical_inverse`` maps caller indices to
    canonical ones. The HMM, scoring, and ties use canonical full-sequence row
    order; returned founder-indexed values use caller order.
    """

    haplotypes: np.ndarray
    evaluated_sites: np.ndarray
    selected_candidate: np.ndarray
    margin_nats: np.ndarray
    changed_sites: np.ndarray
    penalty_nats: float
    snps_per_bin: int
    rho: float
    sample_chunk_size: int
    site_chunk_size: int
    selected_flip_mask: Optional[np.ndarray] = None
    canonical_order: Optional[np.ndarray] = None
    canonical_inverse: Optional[np.ndarray] = None



def _resolved_threads(num_threads: ThreadCount) -> Optional[int]:
    if num_threads is None:
        return None
    value = num_threads() if callable(num_threads) else num_threads
    value = int(value)
    if value < 1:
        raise ValueError("num_threads must resolve to a positive integer")
    numba.set_num_threads(value)
    return value


def _joint_site_penalty(n_sites: int, n_haplotypes: int) -> float:
    """Return log(L*(2**K-1)) without constructing 2**K candidates."""
    exponent = n_haplotypes * math.log(2.0)
    return math.log(n_sites) + exponent + math.log1p(-math.exp(-exponent))


def terminal_cavity_chromosome_penalty(
    block_site_counts: Sequence[int],
    block_haplotype_counts: Sequence[int],
) -> float:
    """Return ``log(sum_b L_b*(2**K_b-1))`` stably.

    This conservative multiplicity penalty is shared by independently scored
    blocks on one chromosome. It accounts for different site and founder
    counts without constructing any of the candidate configurations.
    """
    site_counts = tuple(int(value) for value in block_site_counts)
    haplotype_counts = tuple(int(value) for value in block_haplotype_counts)
    if not site_counts or len(site_counts) != len(haplotype_counts):
        raise ValueError(
            "block count sequences must be non-empty and equal length"
        )
    if any(value < 1 for value in site_counts):
        raise ValueError("every block must contain at least one site")
    if any(value < 1 for value in haplotype_counts):
        raise ValueError("every block must contain at least one haplotype")

    # Retain the exact historical one-block evaluation path.
    if len(site_counts) == 1:
        return _joint_site_penalty(site_counts[0], haplotype_counts[0])

    log_terms = [
        _joint_site_penalty(n_sites, n_haplotypes)
        for n_sites, n_haplotypes in zip(site_counts, haplotype_counts)
    ]
    maximum = max(log_terms)
    return maximum + math.log(
        sum(math.exp(value - maximum) for value in log_terms)
    )


@njit(cache=True, inline="always", fastmath=False)
def _robust_log_emission(probability):
    mixture = TERMINAL_CAVITY_EMISSION_UNIFORM_MIX
    value = probability * (1.0 - mixture) + mixture / 3.0
    if value < 1e-300:
        value = 1e-300
    value = math.log(value)
    return max(value, TERMINAL_CAVITY_LOG_EMISSION_FLOOR)


@njit(cache=True, parallel=True, fastmath=False)
def _build_robust_bin_emissions(probabilities, haplotypes, snps_per_bin):
    """Build float32 robust emissions for one sample chunk."""
    n_samples, n_sites, _ = probabilities.shape
    k = haplotypes.shape[0]
    n_bins = (n_sites + snps_per_bin - 1) // snps_per_bin
    emissions = np.zeros((n_samples, n_bins, k, k), np.float32)
    for sample in prange(n_samples):
        for site in range(n_sites):
            l0 = _robust_log_emission(probabilities[sample, site, 0])
            l1 = _robust_log_emission(probabilities[sample, site, 1])
            l2 = _robust_log_emission(probabilities[sample, site, 2])
            b = site // snps_per_bin
            for left in range(k):
                for right in range(k):
                    dosage = haplotypes[left, site] + haplotypes[right, site]
                    if dosage == 0:
                        emissions[sample, b, left, right] += l0
                    elif dosage == 1:
                        emissions[sample, b, left, right] += l1
                    else:
                        emissions[sample, b, left, right] += l2
    return emissions


@njit(cache=True, fastmath=False)
def _transition_forward(source, destination, d, q, row, col):
    k = source.shape[0]
    total = 0.0
    for i in range(k):
        row[i] = 0.0
        col[i] = 0.0
    for i in range(k):
        for j in range(k):
            value = source[i, j]
            row[i] += value
            col[j] += value
            total += value
    for i in range(k):
        for j in range(k):
            destination[i, j] = (
                d * d * source[i, j]
                + q * d * (row[i] + col[j])
                + q * q * total
            )


@njit(cache=True, parallel=True, fastmath=False)
def _emissions_to_whole_bin_cavity_weights(emissions, rho):
    """Replace emissions by P(ordered diplotype in b | data outside b).

    Forward messages and the overwritten output use float32 storage; sums and
    normalisations use float64. The backward message is streamed, so there is
    no second sample-chunk by bins by K-squared tensor.
    """
    n_samples, n_bins, k, _ = emissions.shape
    alpha = np.empty_like(emissions)
    switch = math.exp(-rho)
    normalizer = 1.0 + (k - 1) * switch
    q = switch / normalizer
    d = (1.0 - switch) / normalizer
    n_states = k * k

    for sample in prange(n_samples):
        row = np.empty(k, np.float64)
        col = np.empty(k, np.float64)
        predicted = np.empty((k, k), np.float64)

        maximum = -math.inf
        for i in range(k):
            for j in range(k):
                value = emissions[sample, 0, i, j]
                if value > maximum:
                    maximum = value
        total = 0.0
        for i in range(k):
            for j in range(k):
                value = math.exp(emissions[sample, 0, i, j] - maximum)
                alpha[sample, 0, i, j] = value
                total += value
        for i in range(k):
            for j in range(k):
                alpha[sample, 0, i, j] /= total

        for b in range(1, n_bins):
            _transition_forward(alpha[sample, b - 1], predicted, d, q, row, col)
            maximum = -math.inf
            for i in range(k):
                for j in range(k):
                    value = emissions[sample, b, i, j]
                    if value > maximum:
                        maximum = value
            total = 0.0
            for i in range(k):
                for j in range(k):
                    value = predicted[i, j] * math.exp(
                        emissions[sample, b, i, j] - maximum
                    )
                    alpha[sample, b, i, j] = value
                    total += value
            for i in range(k):
                for j in range(k):
                    alpha[sample, b, i, j] /= total

        beta = np.full((k, k), 1.0 / n_states, np.float64)
        next_beta = np.empty((k, k), np.float64)
        future = np.empty((k, k), np.float64)
        left = np.empty((k, k), np.float64)
        for b in range(n_bins - 1, -1, -1):
            # beta[b-1] needs emission[b] and beta[b]. Calculate it before
            # overwriting emission[b] with the cavity distribution.
            if b > 0:
                maximum = -math.inf
                for i in range(k):
                    for j in range(k):
                        value = emissions[sample, b, i, j]
                        if value > maximum:
                            maximum = value
                for i in range(k):
                    for j in range(k):
                        future[i, j] = beta[i, j] * math.exp(
                            emissions[sample, b, i, j] - maximum
                        )
                _transition_forward(future, next_beta, d, q, row, col)
                total = 0.0
                for i in range(k):
                    for j in range(k):
                        total += next_beta[i, j]
                for i in range(k):
                    for j in range(k):
                        next_beta[i, j] /= total

            if b == 0:
                for i in range(k):
                    for j in range(k):
                        left[i, j] = 1.0 / n_states
            else:
                _transition_forward(alpha[sample, b - 1], left, d, q, row, col)

            total = 0.0
            for i in range(k):
                for j in range(k):
                    value = left[i, j] * beta[i, j]
                    emissions[sample, b, i, j] = value
                    total += value
            for i in range(k):
                for j in range(k):
                    emissions[sample, b, i, j] /= total

            if b > 0:
                temporary = beta
                beta = next_beta
                next_beta = temporary
    return emissions


@njit(cache=True, parallel=True, fastmath=False)
def _add_two_basin_site_chunk_scores(cavity_weights, probabilities, haplotypes,
                                     sites, snps_per_bin, scores):
    """Accumulate raw focal log likelihoods for the exact 2*K+2 family.

    Founder-pair mass is first grouped by dosage. A flip moves the relevant
    row/column mass between dosage groups, avoiding cancellation from adding a
    likelihood delta to a possibly tiny predictive probability.
    """
    n_samples = probabilities.shape[0]
    k = haplotypes.shape[0]

    for offset in prange(sites.size):
        site = sites[offset]
        b = site // snps_per_bin
        for sample in range(n_samples):
            p0 = np.float64(probabilities[sample, site, 0])
            p1 = np.float64(probabilities[sample, site, 1])
            p2 = np.float64(probabilities[sample, site, 2])
            mass0 = 0.0
            mass1 = 0.0
            mass2 = 0.0
            for i in range(k):
                xi = haplotypes[i, site]
                for j in range(k):
                    dosage = xi + haplotypes[j, site]
                    weight = np.float64(cavity_weights[sample, b, i, j])
                    if dosage == 0:
                        mass0 += weight
                    elif dosage == 1:
                        mass1 += weight
                    else:
                        mass2 += weight

            current = mass0 * p0 + mass1 * p1 + mass2 * p2
            complement = mass2 * p0 + mass1 * p1 + mass0 * p2
            scores[offset, 0] += math.log(max(current, 1e-300))
            scores[offset, k + 1] += math.log(max(complement, 1e-300))

            for flipped in range(k):
                old = haplotypes[flipped, site]
                new = 1 - old
                old0 = old1 = old2 = 0.0
                new0 = new1 = new2 = 0.0
                comp_old0 = comp_old1 = comp_old2 = 0.0
                comp_new0 = comp_new1 = comp_new2 = 0.0
                other_zeros = 0
                other_ones = 0
                for partner in range(k):
                    if partner == flipped:
                        continue
                    partner_bit = haplotypes[partner, site]
                    if partner_bit == 0:
                        other_zeros += 1
                    else:
                        other_ones += 1
                    ordered_weight = (
                        np.float64(cavity_weights[sample, b, flipped, partner])
                        + np.float64(
                            cavity_weights[sample, b, partner, flipped]
                        )
                    )
                    old_dosage = old + partner_bit
                    new_dosage = new + partner_bit
                    if old_dosage == 0:
                        old0 += ordered_weight
                        comp_old2 += ordered_weight
                    elif old_dosage == 1:
                        old1 += ordered_weight
                        comp_old1 += ordered_weight
                    else:
                        old2 += ordered_weight
                        comp_old0 += ordered_weight
                    if new_dosage == 0:
                        new0 += ordered_weight
                    elif new_dosage == 1:
                        new1 += ordered_weight
                    else:
                        new2 += ordered_weight
                    comp_new_dosage = old + (1 - partner_bit)
                    if comp_new_dosage == 0:
                        comp_new0 += ordered_weight
                    elif comp_new_dosage == 1:
                        comp_new1 += ordered_weight
                    else:
                        comp_new2 += ordered_weight

                diagonal_weight = float(
                    cavity_weights[sample, b, flipped, flipped]
                )
                if old == 0:
                    old0 += diagonal_weight
                    new2 += diagonal_weight
                    comp_old2 += diagonal_weight
                    comp_new0 += diagonal_weight
                else:
                    old2 += diagonal_weight
                    new0 += diagonal_weight
                    comp_old0 += diagonal_weight
                    comp_new2 += diagonal_weight

                # Remove incident baseline mass, set structurally impossible
                # residual dosage groups exactly to zero, and clamp any
                # remaining subtraction roundoff at the nonnegative boundary.
                remaining0 = mass0 - old0 if other_zeros else 0.0
                remaining1 = (
                    mass1 - old1 if other_zeros and other_ones else 0.0
                )
                remaining2 = mass2 - old2 if other_ones else 0.0
                m0 = max(remaining0, 0.0) + new0
                m1 = max(remaining1, 0.0) + new1
                m2 = max(remaining2, 0.0) + new2
                current_flip = m0 * p0 + m1 * p1 + m2 * p2

                comp_remaining0 = mass2 - comp_old0 if other_ones else 0.0
                comp_remaining1 = (
                    mass1 - comp_old1 if other_zeros and other_ones else 0.0
                )
                comp_remaining2 = mass0 - comp_old2 if other_zeros else 0.0
                cm0 = max(comp_remaining0, 0.0) + comp_new0
                cm1 = max(comp_remaining1, 0.0) + comp_new1
                cm2 = max(comp_remaining2, 0.0) + comp_new2
                complement_flip = cm0 * p0 + cm1 * p1 + cm2 * p2
                scores[offset, 1 + flipped] += math.log(
                    max(current_flip, 1e-300)
                )
                scores[offset, k + 2 + flipped] += math.log(
                    max(complement_flip, 1e-300)
                )


def _select_exact_candidates(total_scores, penalty):
    """Select the first best alternative and apply the strict penalty gate.

    Candidate order defines exact-score ties. The current candidate is never
    accepted on a tie; among non-current candidates the lowest encoded index
    wins. ``margin_nats`` describes the best alternative even when rejected.
    """
    if total_scores.ndim != 2 or total_scores.shape[1] < 2:
        raise ValueError("total_scores must contain current and alternatives")
    selected = np.zeros(total_scores.shape[0], np.int32)
    margins = np.full(total_scores.shape[0], -math.inf, np.float64)
    if total_scores.shape[0]:
        rows = np.arange(total_scores.shape[0])
        best_alternative = 1 + np.argmax(total_scores[:, 1:], axis=1)
        margins = (
            total_scores[rows, best_alternative]
            - total_scores[:, 0]
            - penalty
        )
        accepted = margins > 0.0
        selected[accepted] = best_alternative[accepted]
    return selected, margins


def _candidate_from_flip_mask(flip_mask):
    """Encode one exact-family XOR mask in caller founder order."""
    k = flip_mask.size
    flipped = np.flatnonzero(flip_mask)
    if flipped.size == 0:
        return 0
    if flipped.size == 1:
        return 1 + int(flipped[0])
    if flipped.size == k:
        return k + 1
    if flipped.size == k - 1:
        unflipped = int(np.flatnonzero(~flip_mask)[0])
        return k + 2 + unflipped
    return -1


def _validate_array_inputs(probabilities, haplotypes):
    probs = np.asarray(probabilities)
    haps = np.asarray(haplotypes)
    if probs.ndim != 3 or probs.shape[2] != 3:
        raise ValueError("probabilities must have shape (samples, sites, 3)")
    if haps.ndim != 2:
        raise ValueError("haplotypes must have shape (K, sites)")
    if probs.shape[1] != haps.shape[1]:
        raise ValueError("probabilities and haplotypes must have the same sites")
    if probs.shape[0] == 0 or haps.shape[0] == 0 or haps.shape[1] == 0:
        raise ValueError("samples, haplotypes, and sites must all be non-empty")
    if probs.dtype not in (np.float32, np.float64):
        probs = probs.astype(np.float32)
    # Probabilities are trusted pipeline objects. In particular, do not scan
    # or materialize the chromosome-sized tensor here. The caller contract is
    # finite, non-negative, per-sample-site normalized genotype likelihoods.
    if not np.all((haps == 0) | (haps == 1)):
        raise ValueError("haplotypes must contain only discrete alleles 0 and 1")
    return probs, np.ascontiguousarray(haps, np.int8)


def _duplicate_canonical_row_members(haplotypes):
    """Mark founders in an exact full-sequence equivalence class."""
    members = np.zeros(haplotypes.shape[0], np.bool_)
    for founder in range(1, haplotypes.shape[0]):
        if np.array_equal(haplotypes[founder - 1], haplotypes[founder]):
            members[founder - 1] = True
            members[founder] = True
    return members


def refine_terminal_cavity(
    probabilities,
    haplotypes,
    *,
    candidate_sites: Optional[Sequence[int]] = None,
    snps_per_bin: int = TERMINAL_CAVITY_SNPS_PER_BIN,
    rho: float = TERMINAL_CAVITY_RHO,
    sample_chunk_size: int = TERMINAL_CAVITY_SAMPLE_CHUNK_SIZE,
    site_chunk_size: int = TERMINAL_CAVITY_SITE_CHUNK_SIZE,
    penalty_nats: Optional[float] = None,
    num_threads: ThreadCount = None,
) -> TerminalCavityResult:
    """Refine a discrete chromosome-length founder panel without mutation.

    The only supported search scores the exact baseline, K single flips,
    complement, and K complement-single-flip family in that order. Work is
    O(N*L*K**2), including construction of the whole-bin factorial HMM cavity.
    ``probabilities`` must be normalized genotype likelihoods. ``penalty_nats``
    can supply a chromosome-wide multiplicity penalty when several
    heterogeneous blocks represent one chromosome. The block wrapper derives
    this shared penalty automatically when it is not supplied.

    Label-specific candidates that split exact full-sequence founder-row
    equivalence classes are excluded because their identity is not estimable.
    """
    previous_threads = numba.get_num_threads()
    try:
        _resolved_threads(num_threads)
        return _refine_terminal_cavity_impl(
            probabilities,
            haplotypes,
            candidate_sites=candidate_sites,
            snps_per_bin=snps_per_bin,
            rho=rho,
            sample_chunk_size=sample_chunk_size,
            site_chunk_size=site_chunk_size,
            penalty_nats=penalty_nats,
            num_threads=num_threads,
        )
    finally:
        if num_threads is not None:
            numba.set_num_threads(previous_threads)


def _refine_terminal_cavity_impl(
    probabilities,
    haplotypes,
    *,
    candidate_sites,
    snps_per_bin,
    rho,
    sample_chunk_size,
    site_chunk_size,
    penalty_nats,
    num_threads,
):
    probs, input_baseline = _validate_array_inputs(probabilities, haplotypes)
    empty_assignments = np.empty((0, 2), np.int64)
    (baseline, _, canonical_order, canonical_inverse,
     _) = canonicalize_binary_panel(input_baseline, empty_assignments)
    baseline = np.ascontiguousarray(baseline, np.int8)
    duplicate_members = _duplicate_canonical_row_members(baseline)
    n_samples, n_sites = probs.shape[:2]
    k = baseline.shape[0]
    if snps_per_bin < 1 or sample_chunk_size < 1 or site_chunk_size < 1:
        raise ValueError("chunk sizes and snps_per_bin must be positive")
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and non-negative")

    if candidate_sites is None:
        sites = np.arange(n_sites, dtype=np.int64)
    else:
        sites = np.asarray(candidate_sites, dtype=np.int64)
        if sites.ndim != 1:
            raise ValueError("candidate_sites must be one-dimensional")
        if sites.size and (np.any(sites < 0) or np.any(sites >= n_sites)):
            raise ValueError("candidate_sites contains an out-of-range index")
        if sites.size != np.unique(sites).size:
            raise ValueError("candidate_sites must not contain duplicates")
        sites = np.ascontiguousarray(sites)

    if penalty_nats is None:
        penalty = _joint_site_penalty(n_sites, k)
    else:
        penalty = float(penalty_nats)
        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError("penalty_nats must be finite and non-negative")

    total_scores = np.zeros((sites.size, 2 * k + 2), np.float64)
    n_bins = (n_sites + snps_per_bin - 1) // snps_per_bin

    # A one-bin chromosome has no outside-bin evidence and is a no-op.
    if n_bins > 1 and sites.size:
        # Persistent float32 cavity storage is
        # N*ceil(L/snps_per_bin)*K**2*4 bytes. Building in sample chunks adds
        # only the chunk-local emissions/forward-message working tensors.
        cavity_weights = np.empty(
            (n_samples, n_bins, k, k), dtype=np.float32
        )
        for sample_start in range(0, n_samples, sample_chunk_size):
            sample_end = min(sample_start + sample_chunk_size, n_samples)
            probability_chunk = np.ascontiguousarray(
                probs[sample_start:sample_end]
            )
            _resolved_threads(num_threads)
            chunk_weights = _build_robust_bin_emissions(
                probability_chunk, baseline, int(snps_per_bin)
            )
            _resolved_threads(num_threads)
            chunk_weights = _emissions_to_whole_bin_cavity_weights(
                chunk_weights, float(rho)
            )
            cavity_weights[sample_start:sample_end] = chunk_weights

        for site_start in range(0, sites.size, site_chunk_size):
            site_end = min(site_start + site_chunk_size, sites.size)
            site_chunk = np.ascontiguousarray(sites[site_start:site_end])
            _resolved_threads(num_threads)
            _add_two_basin_site_chunk_scores(
                cavity_weights, probs, baseline, site_chunk,
                int(snps_per_bin), total_scores[site_start:site_end],
            )

        # A correction assigned to only one of two chromosome-wide identical
        # rows is label-arbitrary. Keep the label-invariant current and full
        # complement candidates, but exclude single-row candidates that split
        # an exact equivalence class.
        for founder in np.flatnonzero(duplicate_members):
            total_scores[:, 1 + founder] = -math.inf
            total_scores[:, k + 2 + founder] = -math.inf

        selected_internal, margins = _select_exact_candidates(
            total_scores, penalty
        )
    else:
        selected_internal = np.zeros(sites.size, np.int32)
        margins = np.full(sites.size, -math.inf, np.float64)

    selected_flip_mask = np.zeros((sites.size, k), np.bool_)
    refined = baseline.copy()
    for offset, site in enumerate(sites):
        choice = int(selected_internal[offset])
        if choice == 0:
            continue
        if choice <= k:
            selected_flip_mask[offset, choice - 1] = True
        elif choice == k + 1:
            selected_flip_mask[offset] = True
        else:
            selected_flip_mask[offset] = True
            selected_flip_mask[offset, choice - (k + 2)] = False
        refined[:, site] ^= selected_flip_mask[offset]

    # HMM scoring and ties use canonical full-sequence row order. Translate
    # every founder-indexed public value back to caller order.
    refined_original = np.ascontiguousarray(refined[canonical_inverse])
    selected_flip_mask_original = np.zeros_like(selected_flip_mask)
    selected_flip_mask_original[:, canonical_order] = selected_flip_mask
    selected = np.zeros(sites.size, np.int32)
    for offset in range(sites.size):
        if selected_internal[offset] != 0:
            selected[offset] = _candidate_from_flip_mask(
                selected_flip_mask_original[offset]
            )

    changed = sites[selected != 0]
    return TerminalCavityResult(
        haplotypes=refined_original,
        evaluated_sites=sites.copy(),
        selected_candidate=selected,
        margin_nats=margins,
        changed_sites=changed.copy(),
        penalty_nats=penalty,
        snps_per_bin=int(snps_per_bin),
        rho=float(rho),
        sample_chunk_size=int(sample_chunk_size),
        site_chunk_size=int(site_chunk_size),
        selected_flip_mask=selected_flip_mask_original,
        canonical_order=canonical_order.copy(),
        canonical_inverse=canonical_inverse.copy(),
    )


def _block_haplotype_matrix(block):
    keys = list(block.haplotypes.keys())
    if not keys:
        raise ValueError("terminal cavity refinement requires haplotypes")
    positions = np.asarray(block.positions)
    rows = []
    layouts = []
    for key in keys:
        value = np.asarray(block.haplotypes[key])
        if value.shape == (positions.size, 2):
            rows.append(np.argmax(value, axis=1).astype(np.int8))
            layouts.append(2)
        elif value.shape == (positions.size,):
            if not np.all((value == 0) | (value == 1)):
                raise ValueError("one-dimensional haplotypes must be discrete")
            rows.append(value.astype(np.int8))
            layouts.append(1)
        else:
            raise ValueError(
                f"haplotype {key!r} has shape {value.shape}; expected "
                f"({positions.size},) or ({positions.size}, 2)"
            )
    return keys, np.ascontiguousarray(rows), layouts


def _exact_probability_slice(block_positions, global_positions, probabilities):
    block_positions = np.asarray(block_positions)
    global_positions = np.asarray(global_positions)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 3 or probabilities.shape[2] != 3:
        raise ValueError("probabilities must have shape (samples, sites, 3)")
    if block_positions.ndim != 1 or global_positions.ndim != 1:
        raise ValueError("block and global positions must be one-dimensional")
    if global_positions.size != probabilities.shape[1]:
        raise ValueError("global positions and probabilities have different lengths")
    if global_positions.size > 1 and np.any(global_positions[1:] <= global_positions[:-1]):
        raise ValueError("global positions must be strictly increasing")
    if block_positions.size > 1 and np.any(block_positions[1:] <= block_positions[:-1]):
        raise ValueError("block positions must be strictly increasing")
    indices = np.searchsorted(global_positions, block_positions)
    if (np.any(indices >= global_positions.size)
            or not np.array_equal(global_positions[indices], block_positions)):
        raise ValueError("every block position must exactly match a global position")
    if indices.size == 0:
        return probabilities[:, :0, :]
    if indices.size == 1 or np.all(indices[1:] == indices[:-1] + 1):
        return probabilities[:, indices[0]:indices[-1] + 1, :]
    return probabilities[:, indices, :]


def refine_terminal_cavity_block(
    block,
    global_positions,
    probabilities,
    *,
    return_diagnostics: bool = False,
    **kwargs,
):
    """Return a refined copy of one BlockResult and optional diagnostics."""
    keys, matrix, layouts = _block_haplotype_matrix(block)
    local_probabilities = _exact_probability_slice(
        block.positions, global_positions, probabilities
    )
    result = refine_terminal_cavity(local_probabilities, matrix, **kwargs)

    new_haplotypes = {}
    changed = result.haplotypes != matrix
    for row, (key, layout) in enumerate(zip(keys, layouts)):
        source = np.asarray(block.haplotypes[key])
        destination = source.copy()
        if layout == 1:
            destination[changed[row]] = result.haplotypes[row, changed[row]]
        else:
            row_changed = changed[row]
            destination[row_changed, 0] = 1 - result.haplotypes[row, row_changed]
            destination[row_changed, 1] = result.haplotypes[row, row_changed]
        new_haplotypes[key] = destination

    # Preserve soft/dynamic attributes while independently owning core fields.
    new_block = copy.copy(block)
    new_block.positions = np.asarray(block.positions).copy()
    new_block.haplotypes = new_haplotypes
    if getattr(block, "keep_flags", None) is not None:
        new_block.keep_flags = np.asarray(block.keep_flags).copy()
    if return_diagnostics:
        return new_block, result
    return new_block


def refine_terminal_cavity_blocks(
    blocks,
    global_positions,
    probabilities,
    *,
    return_diagnostics: bool = False,
    **kwargs,
):
    """Refine copied blocks with one shared chromosome-wide penalty."""
    block_list = list(blocks)
    if len(block_list) > 1 and kwargs.get("penalty_nats") is None:
        kwargs = dict(kwargs)
        kwargs["penalty_nats"] = terminal_cavity_chromosome_penalty(
            [np.asarray(block.positions).size for block in block_list],
            [len(block.haplotypes) for block in block_list],
        )

    refined = []
    diagnostics = []
    for block in block_list:
        new_block, result = refine_terminal_cavity_block(
            block, global_positions, probabilities,
            return_diagnostics=True, **kwargs,
        )
        refined.append(new_block)
        diagnostics.append(result)
    if isinstance(blocks, BlockResults):
        output = copy.copy(blocks)
        output.blocks = refined
    else:
        output = BlockResults(refined)
    if return_diagnostics:
        return output, tuple(diagnostics)
    return output


def summarize_terminal_cavity_results(results):
    """Return compact JSON/pickle-friendly terminal-refinement diagnostics."""
    results = tuple(results)
    total_evaluated_sites = 0
    total_changed_sites = 0
    total_changed_cells = 0
    minimum_accepted_margin_nats = None
    per_block = []

    for result in results:
        flip_mask = result.selected_flip_mask
        if flip_mask is None:
            raise ValueError("terminal cavity result has no selected flip mask")
        accepted = np.asarray(result.selected_candidate) != 0
        if np.any(accepted):
            accepted_minimum = float(np.min(result.margin_nats[accepted]))
            if minimum_accepted_margin_nats is None:
                minimum_accepted_margin_nats = accepted_minimum
            else:
                minimum_accepted_margin_nats = min(
                    minimum_accepted_margin_nats, accepted_minimum
                )

        total_evaluated_sites += int(np.asarray(result.evaluated_sites).size)
        total_changed_sites += int(np.asarray(result.changed_sites).size)
        total_changed_cells += int(np.count_nonzero(flip_mask))
        per_block.append(
            {
                "penalty_nats": float(result.penalty_nats),
                "snps_per_bin": int(result.snps_per_bin),
                "rho": float(result.rho),
                "sample_chunk_size": int(result.sample_chunk_size),
                "site_chunk_size": int(result.site_chunk_size),
            }
        )

    return {
        "model": {
            "revision": TERMINAL_CAVITY_MODEL_REVISION,
            "candidate_family": "exact_2K_plus_2_two_basin",
            "complexity": "O(samples * sites * K**2)",
            "whole_bin_cavity": True,
            "immutable_baseline": True,
            "emission_uniform_mix": float(
                TERMINAL_CAVITY_EMISSION_UNIFORM_MIX
            ),
            "log_emission_floor": float(
                TERMINAL_CAVITY_LOG_EMISSION_FLOOR
            ),
        },
        "n_blocks": len(results),
        "evaluated_sites": total_evaluated_sites,
        "changed_sites": total_changed_sites,
        "changed_founder_cells": total_changed_cells,
        "minimum_accepted_margin_nats": minimum_accepted_margin_nats,
        "per_block": per_block,
    }


def _direct_two_basin_reference(weights, probabilities, haplotypes, sites,
                                snps_per_bin):
    k = haplotypes.shape[0]
    result = np.zeros((len(sites), 2 * k + 2), np.float64)
    for offset, site in enumerate(sites):
        current = haplotypes[:, site]
        configs = [current.copy()]
        for founder in range(k):
            candidate = current.copy()
            candidate[founder] ^= 1
            configs.append(candidate)
        complement = 1 - current
        configs.append(complement.copy())
        for founder in range(k):
            candidate = complement.copy()
            candidate[founder] ^= 1
            configs.append(candidate)
        for candidate_index, candidate in enumerate(configs):
            for sample in range(probabilities.shape[0]):
                predictive = 0.0
                for i in range(k):
                    for j in range(k):
                        predictive += (
                            weights[sample, site // snps_per_bin, i, j]
                            * probabilities[sample, site, candidate[i] + candidate[j]]
                        )
                result[offset, candidate_index] += math.log(max(predictive, 1e-300))
    return result


def self_test():
    """Focused exact-score, determinism, immutability, and wrapper checks."""
    rng = np.random.default_rng(821071)
    n_samples, k, n_sites, spb = 5, 4, 9, 3
    probabilities = rng.random((n_samples, n_sites, 3), dtype=np.float32)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    haplotypes = rng.integers(0, 2, size=(k, n_sites), dtype=np.int8)
    probability_copy = probabilities.copy()
    haplotype_copy = haplotypes.copy()
    sites = np.asarray([0, 2, 5, 8], np.int64)

    # Compare every optimized candidate score with an independent direct sum.
    for test_k in range(1, 9):
        test_haplotypes = rng.integers(
            0, 2, size=(test_k, n_sites), dtype=np.int8
        )
        test_weights = rng.random(
            (n_samples, math.ceil(n_sites / spb), test_k, test_k),
            dtype=np.float32,
        )
        test_weights /= test_weights.sum(axis=(2, 3), keepdims=True)
        optimized = np.zeros((sites.size, 2 * test_k + 2), np.float64)
        _add_two_basin_site_chunk_scores(
            test_weights, probabilities, test_haplotypes, sites, spb, optimized
        )
        direct = _direct_two_basin_reference(
            test_weights, probabilities, test_haplotypes, sites, spb
        )
        if not np.allclose(optimized, direct, atol=1e-6, rtol=1e-7):
            raise AssertionError(
                f"exact scorer disagrees with direct scoring at K={test_k}"
            )

        extreme_weights = np.exp(rng.uniform(
            -30.0, 0.0, size=(n_samples, 1, test_k, test_k)
        )).astype(np.float32)
        extreme_weights /= extreme_weights.sum(axis=(2, 3), keepdims=True)
        extreme_probabilities = np.exp(rng.uniform(
            -80.0, 0.0, size=(n_samples, 1, 3)
        )).astype(np.float32)
        extreme_probabilities /= extreme_probabilities.sum(
            axis=2, keepdims=True
        )
        extreme_haplotypes = rng.integers(
            0, 2, size=(test_k, 1), dtype=np.int8
        )
        extreme_optimized = np.zeros((1, 2 * test_k + 2), np.float64)
        _add_two_basin_site_chunk_scores(
            extreme_weights, extreme_probabilities, extreme_haplotypes,
            np.asarray([0], np.int64), 1, extreme_optimized,
        )
        extreme_direct = _direct_two_basin_reference(
            extreme_weights, extreme_probabilities, extreme_haplotypes,
            np.asarray([0], np.int64), 1,
        )
        if not np.allclose(
            extreme_optimized, extreme_direct, atol=2e-6, rtol=2e-7
        ):
            raise AssertionError(
                f"extreme-range exact scores disagree at K={test_k}"
            )

    tied_scores = np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, 2.0]])
    tied_selected, tied_margins = _select_exact_candidates(tied_scores, 0.0)
    if (not np.array_equal(tied_selected, np.asarray([0, 1], np.int32))
            or not np.array_equal(tied_margins, np.asarray([0.0, 2.0]))):
        raise AssertionError("candidate order or exact-tie behavior changed")
    strict_selected, strict_margin = _select_exact_candidates(
        np.asarray([[0.0, 1.0, 0.5]]), 1.0
    )
    if strict_selected[0] != 0 or strict_margin[0] != 0.0:
        raise AssertionError("penalty equality was not rejected strictly")
    dominant_selected, dominant_margin = _select_exact_candidates(
        np.asarray([[3.0, 2.0, 1.0]]), 0.0
    )
    if dominant_selected[0] != 0 or dominant_margin[0] != -1.0:
        raise AssertionError("best-alternative margin semantics changed")

    one_block_penalty = terminal_cavity_chromosome_penalty([n_sites], [k])
    if one_block_penalty != _joint_site_penalty(n_sites, k):
        raise AssertionError("one-block penalty evaluation changed")
    heterogeneous_penalty = terminal_cavity_chromosome_penalty([4, 3], [2, 3])
    if not math.isclose(heterogeneous_penalty, math.log(33.0), abs_tol=1e-14):
        raise AssertionError("heterogeneous-block penalty is incorrect")
    large_penalty = terminal_cavity_chromosome_penalty([1, 1], [2000, 1999])
    large_expected = 2000.0 * math.log(2.0) + math.log(1.5)
    if (not math.isfinite(large_penalty)
            or not math.isclose(
                large_penalty, large_expected, rel_tol=0.0, abs_tol=1e-12
            )):
        raise AssertionError("large-K chromosome penalty is unstable")

    masks_and_codes = (
        (np.asarray([False, False, False, False]), 0),
        (np.asarray([False, False, True, False]), 3),
        (np.asarray([True, True, True, True]), 5),
        (np.asarray([True, False, True, True]), 7),
    )
    for mask, expected_code in masks_and_codes:
        if _candidate_from_flip_mask(mask) != expected_code:
            raise AssertionError("caller-order candidate encoding changed")

    chunk_a = refine_terminal_cavity(
        probabilities, haplotypes, snps_per_bin=spb,
        sample_chunk_size=1, site_chunk_size=1, penalty_nats=0.0,
        num_threads=1,
    )
    chunk_b = refine_terminal_cavity(
        probabilities, haplotypes, snps_per_bin=spb,
        sample_chunk_size=4, site_chunk_size=5, penalty_nats=0.0,
        num_threads=2,
    )
    for name in ("haplotypes", "selected_candidate", "selected_flip_mask"):
        if not np.array_equal(getattr(chunk_a, name), getattr(chunk_b, name)):
            raise AssertionError(f"chunk/thread choice changed {name}")
    if not np.array_equal(chunk_a.margin_nats, chunk_b.margin_nats):
        raise AssertionError("chunk/thread choice changed exact cavity margins")
    for offset, site in enumerate(chunk_a.evaluated_sites):
        reconstructed = haplotypes[:, site].copy()
        reconstructed ^= chunk_a.selected_flip_mask[offset]
        if not np.array_equal(reconstructed, chunk_a.haplotypes[:, site]):
            raise AssertionError("selected flip mask does not reconstruct site")
        if _candidate_from_flip_mask(
            chunk_a.selected_flip_mask[offset]
        ) != chunk_a.selected_candidate[offset]:
            raise AssertionError("candidate code and caller-order mask disagree")

    reversed_sites = refine_terminal_cavity(
        probabilities, haplotypes, candidate_sites=sites[::-1],
        snps_per_bin=spb, sample_chunk_size=2, site_chunk_size=2,
        penalty_nats=0.0, num_threads=1,
    )
    forward_sites = refine_terminal_cavity(
        probabilities, haplotypes, candidate_sites=sites,
        snps_per_bin=spb, sample_chunk_size=2, site_chunk_size=3,
        penalty_nats=0.0, num_threads=1,
    )
    if not np.array_equal(reversed_sites.haplotypes, forward_sites.haplotypes):
        raise AssertionError("candidate-site permutation changed the panel")
    reverse_lookup = {
        int(site): index for index, site in enumerate(reversed_sites.evaluated_sites)
    }
    for index, site in enumerate(forward_sites.evaluated_sites):
        reverse_index = reverse_lookup[int(site)]
        if (forward_sites.selected_candidate[index]
                != reversed_sites.selected_candidate[reverse_index]
                or forward_sites.margin_nats[index]
                != reversed_sites.margin_nats[reverse_index]):
            raise AssertionError("candidate-site order changed site diagnostics")

    previous_threads = numba.get_num_threads()
    overridden = refine_terminal_cavity(
        probabilities, haplotypes, snps_per_bin=spb,
        sample_chunk_size=2, site_chunk_size=3,
        penalty_nats=1e12, num_threads=1,
    )
    if overridden.penalty_nats != 1e12 or overridden.changed_sites.size:
        raise AssertionError("explicit penalty was not reported/applied")
    if (np.any(overridden.selected_candidate)
            or np.any(overridden.selected_flip_mask)):
        raise AssertionError("rejected alternatives leaked into final output")
    summary = summarize_terminal_cavity_results((chunk_a, overridden))
    accepted_margins = [
        float(margin)
        for result in (chunk_a, overridden)
        for margin, selected in zip(
            result.margin_nats, result.selected_candidate
        )
        if selected != 0
    ]
    expected_minimum = min(accepted_margins) if accepted_margins else None
    expected_per_block = [
        {
            "penalty_nats": float(result.penalty_nats),
            "snps_per_bin": int(result.snps_per_bin),
            "rho": float(result.rho),
            "sample_chunk_size": int(result.sample_chunk_size),
            "site_chunk_size": int(result.site_chunk_size),
        }
        for result in (chunk_a, overridden)
    ]
    expected_model = {
        "revision": TERMINAL_CAVITY_MODEL_REVISION,
        "candidate_family": "exact_2K_plus_2_two_basin",
        "complexity": "O(samples * sites * K**2)",
        "whole_bin_cavity": True,
        "immutable_baseline": True,
        "emission_uniform_mix": float(
            TERMINAL_CAVITY_EMISSION_UNIFORM_MIX
        ),
        "log_emission_floor": float(
            TERMINAL_CAVITY_LOG_EMISSION_FLOOR
        ),
    }
    if (
        summary["model"] != expected_model
        or summary["n_blocks"] != 2
        or summary["evaluated_sites"] != 2 * n_sites
        or summary["changed_sites"] != chunk_a.changed_sites.size
        or summary["changed_founder_cells"]
        != np.count_nonzero(chunk_a.selected_flip_mask)
        or summary["minimum_accepted_margin_nats"] != expected_minimum
        or summary["per_block"] != expected_per_block
    ):
        raise AssertionError("terminal cavity summary is incorrect")

    if numba.get_num_threads() != previous_threads:
        raise AssertionError("refinement did not restore the Numba thread count")
    if (not np.array_equal(probabilities, probability_copy)
            or not np.array_equal(haplotypes, haplotype_copy)):
        raise AssertionError("refinement mutated an input array")

    try:
        refine_terminal_cavity(probabilities, haplotypes, beam_width=1)
    except TypeError as error:
        if "beam_width" not in str(error):
            raise
    else:
        raise AssertionError("obsolete beam_width argument was silently accepted")

    duplicate_haplotypes = np.zeros((3, 6), np.int8)
    duplicate_probabilities = np.empty((40, 6, 3), np.float32)
    duplicate_probabilities[:] = np.asarray([0.998, 0.001, 0.001])
    duplicate_probabilities[:, 0] = np.asarray([0.001, 0.998, 0.001])
    duplicate_result = refine_terminal_cavity(
        duplicate_probabilities,
        duplicate_haplotypes,
        snps_per_bin=3,
        penalty_nats=0.0,
        num_threads=1,
    )
    if (duplicate_result.changed_sites.size
            or np.any(duplicate_result.selected_flip_mask)):
        raise AssertionError(
            "an exact duplicate founder class received a label-specific edit"
        )


    one_bin = refine_terminal_cavity(
        probabilities[:, :spb], haplotypes[:, :spb], snps_per_bin=spb,
        sample_chunk_size=2, site_chunk_size=2, num_threads=1,
    )
    if (not np.array_equal(one_bin.haplotypes, haplotypes[:, :spb])
            or one_bin.changed_sites.size
            or not np.all(np.isneginf(one_bin.margin_nats))):
        raise AssertionError("a one-bin chromosome must be a reported no-op")

    # Canonical full-sequence row order makes unique-score choices equivariant.
    permutation = np.asarray([2, 0, 3, 1])
    permuted = refine_terminal_cavity(
        probabilities, haplotypes[permutation], candidate_sites=sites,
        snps_per_bin=spb, sample_chunk_size=2, site_chunk_size=2,
        penalty_nats=0.0, num_threads=1,
    )
    unpermuted = np.empty_like(permuted.haplotypes)
    unpermuted[permutation] = permuted.haplotypes
    if not np.array_equal(unpermuted, forward_sites.haplotypes):
        raise AssertionError("founder permutation changed canonical endpoints")

    global_positions = np.asarray([5, 10, 20, 30, 40, 50, 60], np.int64)
    block_positions = np.asarray([10, 30, 50, 60], np.int64)
    probability_indices = np.asarray([1, 3, 5, 6])
    block_haps = {
        9: np.column_stack(
            (1 - haplotypes[0, :4], haplotypes[0, :4])
        ).astype(np.float32),
        2: np.column_stack(
            (1 - haplotypes[1, :4], haplotypes[1, :4])
        ).astype(np.float32),
    }
    block = BlockResult(
        block_positions.copy(), block_haps, keep_flags=np.ones(4, bool)
    )
    block.soft_diagnostic = {"preserve": True}
    source_copies = {key: value.copy() for key, value in block_haps.items()}
    global_probs = rng.random(
        (n_samples, global_positions.size, 3), dtype=np.float32
    )
    global_probs /= global_probs.sum(axis=2, keepdims=True)
    wrapped, wrapped_diagnostics = refine_terminal_cavity_block(
        block, global_positions, global_probs,
        snps_per_bin=2, sample_chunk_size=2, site_chunk_size=2,
        num_threads=1, return_diagnostics=True,
    )
    if list(wrapped.haplotypes) != [9, 2]:
        raise AssertionError("non-contiguous haplotype keys were not preserved")
    if (wrapped.soft_diagnostic is not block.soft_diagnostic
            or np.shares_memory(wrapped.positions, block.positions)
            or np.shares_memory(wrapped.keep_flags, block.keep_flags)
            or any(np.shares_memory(wrapped.haplotypes[key], block.haplotypes[key])
                   for key in block_haps)):
        raise AssertionError("wrapper did not shallow-copy shell/core fields")
    if any(
        not np.array_equal(block.haplotypes[key], source_copies[key])
        for key in source_copies
    ):
        raise AssertionError("wrapper mutated the source BlockResult")
    expected_local = global_probs[:, probability_indices, :]
    direct_local = refine_terminal_cavity(
        expected_local,
        np.asarray([np.argmax(block_haps[key], axis=1) for key in (9, 2)]),
        snps_per_bin=2, sample_chunk_size=2, site_chunk_size=2,
        num_threads=1,
    )
    if not np.array_equal(
        wrapped_diagnostics.haplotypes, direct_local.haplotypes
    ):
        raise AssertionError("wrapper did not use the exact position slice")

    block_results = BlockResults([block])
    block_results.soft_diagnostic = "container"
    wrapped_blocks, block_diagnostics = refine_terminal_cavity_blocks(
        block_results, global_positions, global_probs,
        snps_per_bin=2, sample_chunk_size=2, site_chunk_size=2,
        num_threads=1, return_diagnostics=True,
    )
    if (wrapped_blocks.soft_diagnostic != "container"
            or len(wrapped_blocks) != 1 or len(block_diagnostics) != 1):
        raise AssertionError("BlockResults shell attributes were not preserved")

    if block_diagnostics[0].penalty_nats != _joint_site_penalty(4, 2):
        raise AssertionError("one-block wrapper penalty behavior changed")
    for field in (
        "haplotypes", "evaluated_sites", "selected_candidate", "margin_nats",
        "changed_sites", "selected_flip_mask", "canonical_order",
        "canonical_inverse",
    ):
        if not np.array_equal(
            getattr(block_diagnostics[0], field),
            getattr(wrapped_diagnostics, field),
        ):
            raise AssertionError(f"one-block wrapper changed {field}")
    if any(
        not np.array_equal(
            wrapped_blocks.blocks[0].haplotypes[key],
            wrapped.haplotypes[key],
        )
        for key in block_haps
    ):
        raise AssertionError("one-block wrapper numerical behavior changed")

    second_positions = np.asarray([5, 20, 40], np.int64)
    second_haps = {
        5: haplotypes[0, :3].copy(),
        7: haplotypes[1, :3].copy(),
        11: haplotypes[2, :3].copy(),
    }
    second_block = BlockResult(
        second_positions, second_haps, keep_flags=np.ones(3, bool)
    )
    multi_results = BlockResults([block, second_block])
    _, multi_diagnostics = refine_terminal_cavity_blocks(
        multi_results, global_positions, global_probs,
        snps_per_bin=2, sample_chunk_size=2, site_chunk_size=2,
        num_threads=1, return_diagnostics=True,
    )
    shared_penalty = terminal_cavity_chromosome_penalty([4, 3], [2, 3])
    if any(
        diagnostic.penalty_nats != shared_penalty
        for diagnostic in multi_diagnostics
    ):
        raise AssertionError("multi-block wrapper did not share its penalty")
    _, overridden_diagnostics = refine_terminal_cavity_blocks(
        multi_results, global_positions, global_probs,
        snps_per_bin=2, sample_chunk_size=2, site_chunk_size=2,
        penalty_nats=4.25, num_threads=1, return_diagnostics=True,
    )
    if any(result.penalty_nats != 4.25 for result in overridden_diagnostics):
        raise AssertionError("explicit multi-block penalty was not preserved")

    try:
        refine_terminal_cavity_block(
            block, global_positions[:-1], global_probs[:, :-1],
            snps_per_bin=2, num_threads=1,
        )
    except ValueError as error:
        if "exactly match" not in str(error):
            raise
    else:
        raise AssertionError("missing block position was not rejected")

    emission_dtype = _build_robust_bin_emissions(
        probabilities[:2], haplotypes, spb
    ).dtype
    if emission_dtype != np.float32:
        raise AssertionError("chunk-local HMM storage is not float32")
    return {
        "exact_2k_plus_2_scores_k1_to_k8": "pass",
        "extreme_range_scores_k1_to_k8": "pass",
        "candidate_selection_and_ties": "pass",
        "chunk_thread_site_order_determinism": "pass",
        "canonical_order_and_input_immutability": "pass",
        "duplicate_founder_class_identifiability": "pass",
        "strict_penalty_single_bin_and_beam_rejection": "pass",
        "block_wrappers": "pass",
        "chromosome_wide_penalty_and_multiblock_plumbing": "pass",
        "result_summary": "pass",
        "float32_hmm_storage": "pass",
    }


if __name__ == "__main__":
    print(self_test())
