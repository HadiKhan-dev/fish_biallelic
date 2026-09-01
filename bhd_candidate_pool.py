"""Blind combined-soft candidate augmentation from unexplained strands.

This module proposes additional block haplotypes from allele depths and the
current fit's assignments. It never receives cohort labels, founder identities,
a requested K, or biological truth, and it never decides whether a proposal
belongs in the final model.

Every unordered diplotype is scored under a neutral state prior. Posterior
expected-copy responsibilities condition each usable subtractor to infer the
other strand. Missing-aware residuals are clustered both globally and within
subtractor/partner routes; each biological sample contributes at most once to
a soft consensus. Final inclusion remains the responsibility of the separate
cavity model selector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
from numba import njit

from bhd_config import (
    DEFAULT_READ_ERROR_PROBABILITY,
    CANDIDATE_DEDUP_HAMMING_PERCENT,
)
from bhd_kernels import MASK
from bhd_genotype_evidence import (
    allele_depths_to_raw_genotype_likelihoods,
    validate_normalized_genotype_evidence,
)

PROPOSAL_MODE_SOFT_RESIDUAL = "soft_residual"
PROPOSAL_MODE_SOFT_SPLIT = "soft_split"


def allele_depths_to_likelihoods(
    reads: np.ndarray,
    read_error_probability: float = DEFAULT_READ_ERROR_PROBABILITY,
) -> np.ndarray:
    """Return normalized raw P(reads | genotype) for genotypes 0, 1, 2.

    No population-frequency or HWE prior is applied.  Per-cell normalization
    removes only a model-independent constant.  Zero-depth cells are exactly
    uniform, hence make the same constant contribution to every model.
    """

    return allele_depths_to_raw_genotype_likelihoods(
        reads, read_error_probability
    )


def _candidate_alt_probabilities(
    candidates: np.ndarray | Mapping[object, np.ndarray] | Sequence[np.ndarray],
) -> np.ndarray:
    if isinstance(candidates, Mapping):
        values = [candidates[key] for key in sorted(candidates, key=lambda x: str(x))]
        array = np.asarray(values, dtype=np.float64)
    else:
        array = np.asarray(candidates, dtype=np.float64)
    if array.ndim == 3 and array.shape[2] == 2:
        denominator = np.sum(array, axis=2)
        array = np.divide(
            array[:, :, 1],
            denominator,
            out=np.full(denominator.shape, 0.5, dtype=np.float64),
            where=denominator > 0.0,
        )
    if array.ndim != 2:
        raise ValueError(
            "candidates must have shape (K, sites), (K, sites, 2), "
            "or be a mapping/sequence of those rows"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("candidate probabilities must be finite")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("candidate probabilities must lie in [0, 1]")
    return np.ascontiguousarray(array, dtype=np.float64)


def diplotype_genotype_probabilities(
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unordered-pair indices and P(genotype | candidate pair).

    For off-diagonal pairs, the two uncertain fixed haplotype alleles are
    marginalized independently.  For a diagonal pair, both chromosome copies
    share one latent founder allele and therefore cannot be heterozygous.
    """

    q = _candidate_alt_probabilities(candidates)
    pair_i, pair_j = np.triu_indices(len(q))
    genotype = np.empty((len(pair_i), q.shape[1], 3), dtype=np.float64)
    qi = q[pair_i]
    qj = q[pair_j]
    genotype[:, :, 0] = (1.0 - qi) * (1.0 - qj)
    genotype[:, :, 2] = qi * qj
    genotype[:, :, 1] = 1.0 - genotype[:, :, 0] - genotype[:, :, 2]
    diagonal = pair_i == pair_j
    if np.any(diagonal):
        genotype[diagonal, :, 0] = 1.0 - qi[diagonal]
        genotype[diagonal, :, 1] = 0.0
        genotype[diagonal, :, 2] = qi[diagonal]
    # Guard only roundoff; this is also an invariant check for future edits.
    genotype = np.clip(genotype, 0.0, 1.0)
    genotype /= np.sum(genotype, axis=2, keepdims=True)
    return pair_i.astype(np.int64), pair_j.astype(np.int64), genotype


@dataclass(frozen=True)
class ResidualRecord:
    """One anonymous sample's conditionally inferred unexplained strand."""

    sample_index: int
    assignments: tuple[int, int]
    subtractor_index: int
    unexplained_assignment: int
    soft_alt_probability: np.ndarray
    hard_calls: np.ndarray
    compatible_mask: np.ndarray
    hard_known_fraction: float
    record_kind: str = PROPOSAL_MODE_SOFT_RESIDUAL
    responsibility_weight: float = 1.0
    dominant_partner_index: int | None = None
    dominant_partner_probability: float | None = None


@dataclass(frozen=True)
class ProposalDiagnostic:
    """Auditable provenance and disposition of one residual proposal."""

    source_kind: str
    sample_indices: tuple[int, ...]
    cluster_label: int | None
    unique_sample_support: int
    max_pairwise_hamming: float | None
    known_fraction: float
    closest_existing_hamming: float | None
    closest_existing_joint_known_fraction: float | None
    closest_existing_candidate_coverage: float | None
    closest_existing_other_coverage: float | None
    emitted: bool
    reason: str
    proposal_mode: str = PROPOSAL_MODE_SOFT_RESIDUAL
    subtractor_indices: tuple[int, ...] = ()
    dominant_partner_indices: tuple[int, ...] = ()
    dominant_partner_probabilities: tuple[float, ...] = ()
    responsibility_weights: tuple[float, ...] = ()
    effective_sample_support: float = 0.0
    canonical_candidate_digest: str | None = None


@dataclass(frozen=True)
class CandidateProvenance:
    """One-to-one source record for a row in the returned candidate matrix."""

    candidate_index: int
    source_class: str
    canonical_candidate_digest: str
    proposal_diagnostic_index: int | None


@dataclass(frozen=True)
class CandidatePoolAugmentation:
    """Base candidates plus combined soft-residual proposals."""

    candidates: np.ndarray
    n_input_base_candidates: int
    n_discrete_candidates_added: int
    n_base_candidates: int
    n_residual_records: int
    n_residual_clusters: int
    n_hdbscan_clusters: int
    n_complete_link_clusters: int
    n_hdbscan_initial_noise: int
    n_unclustered_singletons: int
    n_emitted_candidates: int
    residual_records: tuple[ResidualRecord, ...]
    proposal_diagnostics: tuple[ProposalDiagnostic, ...]
    n_soft_records: int = 0
    n_soft_residual_clusters: int = 0
    n_soft_split_clusters: int = 0
    n_soft_candidates_emitted: int = 0
    candidate_provenance: tuple[CandidateProvenance, ...] = ()


@dataclass(frozen=True)
class ResidualInputWorkspace:
    """Block-invariant likelihood and depth arrays for residual proposals.

    Candidate augmentation may be called repeatedly for different fitted
    panels from the same block. Preparing these arrays once avoids repeating
    identical read-likelihood normalization.  The floored log likelihood is
    also evidence-only and lets complete binary panels gather log values
    without recomputing the same logarithms for every diplotype.
    """

    likelihood: np.ndarray
    log_likelihood: np.ndarray
    depth: np.ndarray
    read_error_probability: float
    cluster_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict, compare=False, repr=False
    )


def prepare_residual_inputs(
    reads_array: np.ndarray,
    read_error_probability: float = DEFAULT_READ_ERROR_PROBABILITY,
    *,
    likelihood: np.ndarray | None = None,
) -> ResidualInputWorkspace:
    """Prepare reusable, model-identical inputs for residual extraction."""

    reads = np.asarray(reads_array)
    if reads.ndim != 3 or reads.shape[2] != 2:
        raise ValueError("reads_array must have shape (samples, sites, 2)")
    if not np.all(np.isfinite(reads)) or np.any(reads < 0):
        raise ValueError("allele depths must be finite and non-negative")
    if likelihood is None:
        likelihood = allele_depths_to_likelihoods(
            reads, read_error_probability=read_error_probability
        )
    else:
        likelihood = validate_normalized_genotype_evidence(
            likelihood, n_samples=reads.shape[0], n_sites=reads.shape[1]
        )
    log_likelihood = np.ascontiguousarray(
        np.log(np.maximum(likelihood, np.finfo(np.float64).tiny))
    )
    depth = np.ascontiguousarray(np.sum(reads, axis=2))
    return ResidualInputWorkspace(
        likelihood=likelihood,
        log_likelihood=log_likelihood,
        depth=depth,
        read_error_probability=float(read_error_probability),
    )


def _validate_residual_input_workspace(
    workspace: ResidualInputWorkspace,
    reads_shape: tuple[int, int, int],
    read_error_probability: float,
) -> ResidualInputWorkspace:
    if not isinstance(workspace, ResidualInputWorkspace):
        raise TypeError("residual_input_workspace must be ResidualInputWorkspace")
    expected_likelihood_shape = (reads_shape[0], reads_shape[1], 3)
    expected_depth_shape = reads_shape[:2]
    likelihood = np.asarray(workspace.likelihood)
    log_likelihood = np.asarray(workspace.log_likelihood)
    depth = np.asarray(workspace.depth)
    if (
        likelihood.shape != expected_likelihood_shape
        or log_likelihood.shape != expected_likelihood_shape
        or depth.shape != expected_depth_shape
    ):
        raise ValueError("residual_input_workspace and reads_array disagree")
    if float(workspace.read_error_probability) != float(read_error_probability):
        raise ValueError(
            "residual_input_workspace uses a different read_error_probability"
        )
    return workspace


def _base_candidate_matrix(block_result: Any, n_sites: int) -> np.ndarray:
    haplotypes = getattr(block_result, "haplotypes", None)
    if not haplotypes:
        return np.empty((0, n_sites), dtype=np.float64)
    rows: list[np.ndarray] = []
    for key in sorted(haplotypes, key=lambda item: str(item)):
        haplotype = np.asarray(haplotypes[key], dtype=np.float64)
        if haplotype.ndim == 1 and haplotype.shape == (n_sites,):
            alt = haplotype.copy()
            alt[np.isclose(alt, MASK, rtol=0.0, atol=1e-12)] = 0.5
        elif haplotype.ndim == 2 and haplotype.shape == (n_sites, 2):
            denominator = np.sum(haplotype, axis=1)
            alt = np.divide(
                haplotype[:, 1],
                denominator,
                out=np.full(n_sites, 0.5, dtype=np.float64),
                where=denominator > 0.0,
            )
        else:
            raise ValueError(
                f"candidate {key!r} has unsupported shape {haplotype.shape}"
            )
        rows.append(np.clip(alt, 0.0, 1.0))
    return np.ascontiguousarray(np.stack(rows), dtype=np.float64)


def _validate_base_candidates(
    base_candidates: np.ndarray | None,
    block_result: Any,
    n_sites: int,
) -> np.ndarray:
    if base_candidates is None:
        return _base_candidate_matrix(block_result, n_sites)
    candidates = np.asarray(base_candidates, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != n_sites:
        raise ValueError("base_candidates must have shape (K, sites)")
    if np.any(~np.isfinite(candidates)):
        raise ValueError("base_candidates must be finite")
    if np.any((candidates < 0.0) | (candidates > 1.0)):
        raise ValueError("base_candidates must lie in [0, 1]")
    return np.ascontiguousarray(candidates)


def _soft_conditional_residual(
    genotype_likelihood: np.ndarray,
    read_depth: np.ndarray,
    subtractor_alt_probability: np.ndarray,
    keep_mask: np.ndarray,
    hard_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer the other allele while integrating subtractor uncertainty.

    For residual allele ``b``, its likelihood is the sum over subtractor
    allele ``a`` of ``P(a) P(reads | a + b)``.  This is deliberately softer
    than :func:`_conditional_residual`: no MAP-genotype compatibility gate is
    used, and uncertain subtractor sites remain uncertain rather than being
    silently forced to either allele.
    """

    likelihood = np.asarray(genotype_likelihood, dtype=np.float64)
    depth = np.asarray(read_depth)
    subtractor_q = np.asarray(
        subtractor_alt_probability, dtype=np.float64
    )
    if likelihood.ndim != 2 or likelihood.shape[1] != 3:
        raise ValueError("genotype_likelihood must have shape (sites, 3)")
    n_sites = likelihood.shape[0]
    if depth.shape != (n_sites,) or subtractor_q.shape != (n_sites,):
        raise ValueError("depth and subtractor must match the site dimension")
    if keep_mask.shape != (n_sites,):
        raise ValueError("keep_mask must match the site dimension")
    if np.any(~np.isfinite(subtractor_q)) or np.any(
        (subtractor_q < 0.0) | (subtractor_q > 1.0)
    ):
        raise ValueError("subtractor probabilities must lie in [0, 1]")

    residual_zero = (
        (1.0 - subtractor_q) * likelihood[:, 0]
        + subtractor_q * likelihood[:, 1]
    )
    residual_one = (
        (1.0 - subtractor_q) * likelihood[:, 1]
        + subtractor_q * likelihood[:, 2]
    )
    denominator = residual_zero + residual_one
    observed = keep_mask & (depth > 0) & (denominator > 0.0)
    soft_alt = np.full(n_sites, 0.5, dtype=np.float64)
    soft_alt[observed] = residual_one[observed] / denominator[observed]
    hard = np.full(n_sites, MASK, dtype=np.int8)
    hard[observed & (soft_alt >= hard_probability)] = 1
    hard[observed & (soft_alt <= 1.0 - hard_probability)] = 0
    return soft_alt, hard, observed


@njit(cache=True, nogil=True, fastmath=False)
def _soft_residual_numeric_kernel(
    responsibility: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    n_haplotypes: int,
    usable_indices: np.ndarray,
    usable_probabilities: np.ndarray,
    likelihood: np.ndarray,
    depth: np.ndarray,
    keep_mask: np.ndarray,
    hard_probability: float,
    minimum_responsibility: float,
    mask_value: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Accumulate posterior copies and condition accepted residual strands."""

    n_samples, n_states = responsibility.shape
    n_usable, n_sites = usable_probabilities.shape
    maximum_records = n_samples * n_usable
    sample_indices = np.empty(maximum_records, dtype=np.int64)
    subtractor_indices = np.empty(maximum_records, dtype=np.int64)
    subtractor_local_indices = np.empty(maximum_records, dtype=np.int64)
    dominant_partners = np.empty(maximum_records, dtype=np.int64)
    weights = np.empty(maximum_records, dtype=np.float64)
    partner_probabilities = np.empty(maximum_records, dtype=np.float64)

    copy_mass = np.zeros(n_haplotypes, dtype=np.float64)
    partner_mass = np.zeros(
        (n_haplotypes, n_haplotypes), dtype=np.float64
    )
    tiny = np.finfo(np.float64).tiny
    record_count = 0
    for sample_index in range(n_samples):
        copy_mass.fill(0.0)
        partner_mass.fill(0.0)
        for state_index in range(n_states):
            first = int(pair_i[state_index])
            second = int(pair_j[state_index])
            mass = responsibility[sample_index, state_index]
            if first == second:
                copy_mass[first] += 2.0 * mass
                partner_mass[first, first] += 2.0 * mass
            else:
                copy_mass[first] += mass
                copy_mass[second] += mass
                partner_mass[first, second] += mass
                partner_mass[second, first] += mass

        for subtractor_local_index in range(n_usable):
            subtractor_index = int(usable_indices[subtractor_local_index])
            expected_copies = copy_mass[subtractor_index]
            weight = min(1.0, expected_copies)
            if weight + 1e-12 < minimum_responsibility:
                continue

            dominant_partner = 0
            dominant_mass = partner_mass[subtractor_index, 0]
            for partner_index in range(1, n_haplotypes):
                candidate_mass = partner_mass[
                    subtractor_index, partner_index
                ]
                if candidate_mass > dominant_mass:
                    dominant_partner = partner_index
                    dominant_mass = candidate_mass
            denominator = max(expected_copies, tiny)
            sample_indices[record_count] = sample_index
            subtractor_indices[record_count] = subtractor_index
            subtractor_local_indices[record_count] = subtractor_local_index
            dominant_partners[record_count] = dominant_partner
            weights[record_count] = weight
            partner_probabilities[record_count] = (
                dominant_mass / denominator
            )
            record_count += 1

    soft_alt = np.full(
        (record_count, n_sites), 0.5, dtype=np.float64
    )
    hard_calls = np.full(
        (record_count, n_sites), mask_value, dtype=np.int8
    )
    compatible = np.zeros((record_count, n_sites), dtype=np.bool_)
    for record_index in range(record_count):
        sample_index = sample_indices[record_index]
        subtractor_local_index = subtractor_local_indices[record_index]
        for site_index in range(n_sites):
            subtractor_q = usable_probabilities[
                subtractor_local_index, site_index
            ]
            residual_zero = (
                (1.0 - subtractor_q) * likelihood[sample_index, site_index, 0]
                + subtractor_q * likelihood[sample_index, site_index, 1]
            )
            residual_one = (
                (1.0 - subtractor_q) * likelihood[sample_index, site_index, 1]
                + subtractor_q * likelihood[sample_index, site_index, 2]
            )
            denominator = residual_zero + residual_one
            if (
                keep_mask[site_index]
                and depth[sample_index, site_index] > 0
                and denominator > 0.0
            ):
                value = residual_one / denominator
                soft_alt[record_index, site_index] = value
                compatible[record_index, site_index] = True
                if value >= hard_probability:
                    hard_calls[record_index, site_index] = 1
                if value <= 1.0 - hard_probability:
                    hard_calls[record_index, site_index] = 0

    return (
        sample_indices[:record_count],
        subtractor_indices[:record_count],
        dominant_partners[:record_count],
        weights[:record_count],
        partner_probabilities[:record_count],
        soft_alt,
        hard_calls,
        compatible,
    )


@njit(cache=True, nogil=True, fastmath=False)
def _gather_binary_predictive(
    likelihood: np.ndarray,
    haplotypes: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    predictive: np.ndarray,
) -> None:
    """Gather one exact likelihood per hard diplotype/site into F-order output."""

    n_samples = likelihood.shape[0]
    n_pairs = pair_i.shape[0]
    n_sites = haplotypes.shape[1]
    # Match the F-order output's memory traversal: sample, then pair, then site.
    for site_index in range(n_sites):
        for pair_index in range(n_pairs):
            genotype = (
                haplotypes[pair_i[pair_index], site_index]
                + haplotypes[pair_j[pair_index], site_index]
            )
            for sample_index in range(n_samples):
                predictive[sample_index, pair_index, site_index] = likelihood[
                    sample_index, site_index, genotype
                ]


def _binary_panel_responsibility(
    likelihood: np.ndarray,
    haplotypes: np.ndarray,
    keep_mask: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    prepared_log_likelihood: np.ndarray | None = None,
) -> np.ndarray:
    """Exact hard-panel responsibilities with a bounded predictive workspace.

    For a complete binary panel, the genotype distribution of every
    diplotype/site is one-hot at dosage ``hap_i + hap_j``.  Gathering that
    exact likelihood entry avoids materializing the generic three-genotype
    tensor and multiplying its two structural zeroes.  The established
    predictive-array layout, NumPy ``log``/axis-2 ``sum``, and full-panel
    softmax are retained so pair/site reduction and tie behaviour do not
    change.  Uncertain panels never enter this helper.
    """

    haplotypes = np.asarray(haplotypes)
    if haplotypes.ndim != 2 or not np.all(
        (haplotypes == 0) | (haplotypes == 1)
    ):
        raise ValueError("binary responsibility requires a complete 0/1 panel")
    expected_i, expected_j = np.triu_indices(len(haplotypes))
    if not np.array_equal(pair_i, expected_i) or not np.array_equal(
        pair_j, expected_j
    ):
        raise ValueError("pair indices do not match the binary panel")

    likelihood = np.asarray(likelihood)
    if prepared_log_likelihood is None:
        log_likelihood = np.log(
            np.maximum(likelihood, np.finfo(np.float64).tiny)
        )
    else:
        log_likelihood = np.asarray(prepared_log_likelihood)
        if log_likelihood.shape != likelihood.shape:
            raise ValueError("prepared log likelihood and likelihood disagree")
    kept_log_likelihood = log_likelihood[:, keep_mask, :]
    kept_haplotypes = np.ascontiguousarray(haplotypes[:, keep_mask])
    n_samples = len(likelihood)
    n_pairs = len(pair_i)
    if n_pairs == 1:
        # The generic one-state softmax is identically one; avoiding its
        # emission calculation also avoids degenerate singleton layouts.
        return np.ones((n_samples, 1), dtype=np.float64)

    # Bound only the largest temporary.  Pair chunking is not equivalent here:
    # it changes the predictive array's site stride and therefore NumPy's
    # floating-point summation order on near ties.
    maximum_predictive_bytes = 32 * 1024 * 1024
    bytes_per_sample = max(
        1,
        n_pairs * kept_log_likelihood.shape[1] * np.dtype(np.float64).itemsize,
    )
    samples_per_chunk = max(
        1,
        min(n_samples, maximum_predictive_bytes // bytes_per_sample),
    )
    log_emission_order = "F" if n_samples > 1 else "C"
    log_emission = np.empty(
        (n_samples, n_pairs), dtype=np.float64, order=log_emission_order
    )
    for start in range(0, n_samples, samples_per_chunk):
        stop = min(start + samples_per_chunk, n_samples)
        log_predictive = np.empty(
            (stop - start, n_pairs, kept_log_likelihood.shape[1]),
            dtype=np.float64,
            order="F",
        )
        _gather_binary_predictive(
            kept_log_likelihood[start:stop],
            kept_haplotypes,
            pair_i,
            pair_j,
            log_predictive,
        )
        log_emission[start:stop] = np.sum(log_predictive, axis=2)

    row_maximum = np.max(log_emission, axis=1, keepdims=True)
    responsibility = np.exp(log_emission - row_maximum)
    responsibility /= np.sum(responsibility, axis=1, keepdims=True)
    return np.ascontiguousarray(responsibility)


def _extract_soft_residual_records(
    block_result: Any,
    reads_array: np.ndarray,
    keep_mask: np.ndarray,
    read_error_probability: float,
    usable_founder_known_fraction: float,
    hard_probability: float,
    minimum_responsibility: float,
    *,
    residual_input_workspace: ResidualInputWorkspace | None = None,
    binary_panel_fast_path: bool = False,
) -> tuple[ResidualRecord, ...]:
    """Create all-assignment residuals from neutral diplotype posteriors.

    Every unordered diplotype made from well-resolved discrete rows is scored
    against the supplied read counts.  These counts are the only evidence the
    function sees; callers must therefore pass the training partition.  A
    row is used as a subtractor in proportion to its posterior expected copy
    count, even when the hard fit assigned two ordinary usable rows.
    """

    discrete = np.asarray(getattr(block_result, "discrete_haps", None))
    assignments = np.asarray(getattr(block_result, "pair_assignments", None))
    reads = np.asarray(reads_array)
    if discrete.ndim != 2:
        raise ValueError("block_result.discrete_haps must have shape (K, sites)")
    if assignments.ndim != 2 or assignments.shape[1] != 2:
        raise ValueError("block_result.pair_assignments must have shape (samples, 2)")
    if reads.ndim != 3 or reads.shape[2] != 2:
        raise ValueError("reads_array must have shape (samples, sites, 2)")
    if reads.shape[:2] != (assignments.shape[0], discrete.shape[1]):
        raise ValueError("reads, assignments, and discrete haplotypes disagree")
    if np.any(reads < 0):
        raise ValueError("allele depths must be non-negative")

    n_haplotypes, n_sites = discrete.shape
    if int(getattr(block_result, "K_final", n_haplotypes)) != n_haplotypes:
        raise ValueError("K_final and discrete_haps use inconsistent coordinates")
    if np.any((assignments < 0) | (assignments > n_haplotypes)):
        raise ValueError("pair_assignments contain an invalid non-wildcard index")

    minimum_known = int(
        math.ceil(usable_founder_known_fraction * int(np.sum(keep_mask)))
    )
    founder_known = ((discrete == 0) | (discrete == 1)) & keep_mask[None, :]
    usable_indices = np.flatnonzero(
        np.sum(founder_known, axis=1) >= minimum_known
    )
    if len(usable_indices) == 0:
        return ()

    usable_probabilities = np.full(
        (len(usable_indices), n_sites), 0.5, dtype=np.float64
    )
    usable_rows = discrete[usable_indices]
    known = (usable_rows == 0) | (usable_rows == 1)
    usable_probabilities[known] = usable_rows[known]
    workspace = (
        prepare_residual_inputs(reads, read_error_probability)
        if residual_input_workspace is None
        else _validate_residual_input_workspace(
            residual_input_workspace, reads.shape, read_error_probability
        )
    )
    likelihood = np.asarray(workspace.likelihood)
    depth = np.asarray(workspace.depth)
    use_binary_fast_path = bool(binary_panel_fast_path) and bool(
        np.all(known[:, keep_mask])
    )
    if use_binary_fast_path:
        pair_i_local, pair_j_local = np.triu_indices(len(usable_indices))
        responsibility = _binary_panel_responsibility(
            np.ascontiguousarray(likelihood),
            np.ascontiguousarray(usable_rows, dtype=np.int64),
            np.ascontiguousarray(keep_mask),
            np.ascontiguousarray(pair_i_local, dtype=np.int64),
            np.ascontiguousarray(pair_j_local, dtype=np.int64),
            prepared_log_likelihood=np.asarray(workspace.log_likelihood),
        )
    else:
        pair_i_local, pair_j_local, genotype = diplotype_genotype_probabilities(
            usable_probabilities
        )
        predictive = np.einsum(
            "nlg,plg->npl",
            likelihood[:, keep_mask, :],
            genotype[:, keep_mask, :],
            optimize=True,
        )
        log_emission = np.sum(
            np.log(np.maximum(predictive, np.finfo(np.float64).tiny)), axis=2
        )
        row_maximum = np.max(log_emission, axis=1, keepdims=True)
        responsibility = np.exp(log_emission - row_maximum)
        responsibility /= np.sum(responsibility, axis=1, keepdims=True)
    pair_i = usable_indices[pair_i_local]
    pair_j = usable_indices[pair_j_local]

    (
        record_sample_indices,
        record_subtractor_indices,
        record_dominant_partners,
        record_weights,
        record_partner_probabilities,
        record_soft_alt,
        record_hard_calls,
        record_compatible,
    ) = _soft_residual_numeric_kernel(
        np.ascontiguousarray(responsibility),
        np.ascontiguousarray(pair_i),
        np.ascontiguousarray(pair_j),
        n_haplotypes,
        np.ascontiguousarray(usable_indices),
        np.ascontiguousarray(usable_probabilities),
        np.ascontiguousarray(likelihood),
        np.ascontiguousarray(depth),
        np.ascontiguousarray(keep_mask),
        hard_probability,
        minimum_responsibility,
        int(MASK),
    )

    records: list[ResidualRecord] = []
    for record_index, sample_index_value in enumerate(record_sample_indices):
        sample_index = int(sample_index_value)
        hard = record_hard_calls[record_index]
        dominant_partner = int(record_dominant_partners[record_index])
        records.append(
            ResidualRecord(
                sample_index=sample_index,
                assignments=tuple(
                    int(value) for value in assignments[sample_index]
                ),
                subtractor_index=int(
                    record_subtractor_indices[record_index]
                ),
                unexplained_assignment=dominant_partner,
                soft_alt_probability=record_soft_alt[record_index],
                hard_calls=hard,
                compatible_mask=record_compatible[record_index],
                hard_known_fraction=float(
                    np.mean(hard[keep_mask] != MASK)
                ),
                record_kind="posterior_all_assignment",
                responsibility_weight=float(record_weights[record_index]),
                dominant_partner_index=dominant_partner,
                dominant_partner_probability=float(
                    record_partner_probabilities[record_index]
                ),
            )
        )
    return tuple(records)


def _effective_unique_sample_support(
    records: Sequence[ResidualRecord], member_indices: Sequence[int]
) -> float:
    by_sample: dict[int, float] = {}
    for index in member_indices:
        record = records[int(index)]
        by_sample[record.sample_index] = min(
            1.0,
            by_sample.get(record.sample_index, 0.0)
            + float(record.responsibility_weight),
        )
    return float(math.fsum(by_sample.values()))


def _candidate_digest(candidate: np.ndarray) -> str:
    canonical = np.round(np.asarray(candidate, dtype=np.float64), 12).astype(
        "<f8", copy=False
    )
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


@njit(cache=True, fastmath=False)
def _missing_aware_hamming_reference_kernel(
    hard_calls: np.ndarray,
    keep_mask: np.ndarray,
    minimum_joint: int,
    n_kept: int,
    mask_value: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar reference for exact missing-aware pairwise Hamming counts."""

    n_records, n_sites = hard_calls.shape
    distance = np.zeros((n_records, n_records), dtype=np.float64)
    joint_fraction = np.ones((n_records, n_records), dtype=np.float64)
    for first in range(n_records - 1):
        for second in range(first + 1, n_records):
            n_joint = 0
            n_mismatch = 0
            for site in range(n_sites):
                if keep_mask[site]:
                    first_value = hard_calls[first, site]
                    second_value = hard_calls[second, site]
                    if (
                        first_value != mask_value
                        and second_value != mask_value
                    ):
                        n_joint += 1
                        if first_value != second_value:
                            n_mismatch += 1
            joint_value = n_joint / n_kept
            joint_fraction[first, second] = joint_value
            joint_fraction[second, first] = joint_value
            if n_joint < minimum_joint:
                value = 1.0
            else:
                value = n_mismatch / n_joint
            distance[first, second] = value
            distance[second, first] = value
    return distance, joint_fraction


@njit(cache=True, inline="always")
def _popcount_uint64(value: np.uint64) -> int:
    """Population count with a fixed-cost SWAR reduction.

    Numba does not expose one portable ``uint64.bit_count`` implementation
    across all versions used on CSD3.  These unsigned operations are the
    standard exact 64-bit population-count reduction and compile to a small
    register-only sequence.
    """

    value = value - (
        (value >> np.uint64(1)) & np.uint64(0x5555555555555555)
    )
    value = (
        (value & np.uint64(0x3333333333333333))
        + ((value >> np.uint64(2)) & np.uint64(0x3333333333333333))
    )
    value = (value + (value >> np.uint64(4))) & np.uint64(
        0x0F0F0F0F0F0F0F0F
    )
    return int(
        (value * np.uint64(0x0101010101010101)) >> np.uint64(56)
    )


@njit(cache=True, fastmath=False)
def _missing_aware_hamming_distance_kernel(
    hard_calls: np.ndarray,
    keep_mask: np.ndarray,
    minimum_joint: int,
    mask_value: int,
) -> np.ndarray:
    """Distance-only form of the exact bit-packed Hamming calculation."""

    n_records, n_sites = hard_calls.shape
    n_words = (n_sites + 63) // 64
    known_bits = np.zeros((n_records, n_words), dtype=np.uint64)
    allele_bits = np.zeros((n_records, n_words), dtype=np.uint64)
    for record in range(n_records):
        for site in range(n_sites):
            value = hard_calls[record, site]
            if keep_mask[site] and value != mask_value:
                word = site >> 6
                bit = np.uint64(1) << np.uint64(site & 63)
                known_bits[record, word] |= bit
                if value == 1:
                    allele_bits[record, word] |= bit

    distance = np.zeros((n_records, n_records), dtype=np.float64)
    for first in range(n_records - 1):
        for second in range(first + 1, n_records):
            n_joint = 0
            n_mismatch = 0
            for word in range(n_words):
                joint = known_bits[first, word] & known_bits[second, word]
                mismatch = (
                    allele_bits[first, word] ^ allele_bits[second, word]
                ) & joint
                n_joint += _popcount_uint64(joint)
                n_mismatch += _popcount_uint64(mismatch)
            if n_joint < minimum_joint:
                value = 1.0
            else:
                value = n_mismatch / n_joint
            distance[first, second] = value
            distance[second, first] = value
    return distance


def _missing_aware_hamming_distance_matrix(
    records: Sequence[ResidualRecord],
    keep_mask: np.ndarray,
    minimum_joint_known_fraction: float,
) -> np.ndarray:
    """Return only the matrix consumed by residual clustering."""

    n_records = len(records)
    if n_records == 0:
        return np.zeros((0, 0), dtype=np.float64)
    n_kept = int(np.sum(keep_mask))
    minimum_joint = max(
        1, int(math.ceil(minimum_joint_known_fraction * n_kept))
    )
    hard_calls = np.ascontiguousarray(
        np.stack([record.hard_calls for record in records])
    )
    return _missing_aware_hamming_distance_kernel(
        hard_calls,
        np.ascontiguousarray(keep_mask),
        minimum_joint,
        int(MASK),
    )


def _cluster_residuals(
    distance: np.ndarray,
    maximum_cluster_hamming: float,
) -> tuple[
    tuple[tuple[str, int | None, tuple[int, ...], float], ...],
    tuple[int, ...],
    int,
]:
    """Find cohesive residual groups without forcing a cluster count.

    HDBSCAN supplies the density-based primary partition.  Small, genuine
    founder groups can nevertheless be labelled noise beside a denser common
    group, so complete-link clustering is applied only to HDBSCAN's remaining
    noise.  Every retained group must satisfy the same explicit maximum
    pairwise Hamming bound; complete-link prevents density chaining.
    """

    n_records = distance.shape[0]
    if n_records < 2:
        return (), tuple(range(n_records)), n_records
    if (
        n_records == 2
        and np.all(np.isfinite(distance))
        and np.array_equal(distance, distance.T)
        and distance[0, 0] == 0.0
        and distance[1, 1] == 0.0
    ):
        pair_distance = float(distance[0, 1])
        if pair_distance <= maximum_cluster_hamming + 1e-12:
            return (
                (("hdbscan_cluster", 0, (0, 1), pair_distance),),
                (),
                0,
            )
        return (), (0, 1), 0

    import hdbscan

    labels = np.asarray(
        hdbscan.hdbscan(
            distance,
            min_cluster_size=2,
            min_samples=1,
            alpha=1.0,
            cluster_selection_epsilon=0.0,
            cluster_selection_persistence=0.0,
            max_cluster_size=0,
            metric="precomputed",
            p=None,
            leaf_size=40,
            algorithm="best",
            approx_min_span_tree=True,
            gen_min_span_tree=False,
            core_dist_n_jobs=1,
            cluster_selection_method="eom",
            allow_single_cluster=True,
            match_reference_implementation=False,
            cluster_selection_epsilon_max=np.inf,
        )[0],
        dtype=np.int64,
    )
    clusters: list[tuple[str, int | None, tuple[int, ...], float]] = []
    initial_noise_count = int(np.sum(labels == -1))
    noise = set(np.flatnonzero(labels == -1).tolist())
    for label in sorted(int(value) for value in np.unique(labels) if value >= 0):
        members = tuple(np.flatnonzero(labels == label).tolist())
        if len(members) < 2:
            noise.update(members)
            continue
        subdistance = distance[np.ix_(members, members)]
        max_distance = float(np.max(subdistance))
        if max_distance <= maximum_cluster_hamming + 1e-12:
            clusters.append(("hdbscan_cluster", label, members, max_distance))
        else:
            # HDBSCAN is a density algorithm, not a complete-link guarantee.
            noise.update(members)

    # Rescue cohesive small groups that HDBSCAN labelled as noise.  This does
    # not set K: it partitions only residual proposals at an absolute sequence
    # disagreement bound; downstream cavity scoring still decides final
    # panel inclusion.
    remaining = tuple(sorted(noise))
    complete_link_members: set[int] = set()
    if len(remaining) >= 2:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        subdistance = distance[np.ix_(remaining, remaining)]
        hierarchy = linkage(
            squareform(subdistance, checks=False), method="complete"
        )
        group_labels = fcluster(
            hierarchy, t=float(maximum_cluster_hamming), criterion="distance"
        )
        for group_label in sorted(np.unique(group_labels)):
            local = np.flatnonzero(group_labels == group_label)
            if len(local) < 2:
                continue
            members = tuple(remaining[int(index)] for index in local)
            max_distance = float(np.max(distance[np.ix_(members, members)]))
            if max_distance <= maximum_cluster_hamming + 1e-12:
                clusters.append(
                    ("complete_link_noise_cluster", None, members, max_distance)
                )
                complete_link_members.update(members)

    clustered = {member for _, _, members, _ in clusters for member in members}
    noise = set(range(n_records)) - clustered
    return tuple(clusters), tuple(sorted(noise)), initial_noise_count


def _cluster_residuals_cached(
    distance: np.ndarray,
    maximum_cluster_hamming: float,
    cache: dict[tuple[Any, ...], Any] | None,
) -> tuple[
    tuple[tuple[str, int | None, tuple[int, ...], float], ...],
    tuple[int, ...],
    int,
]:
    """Reuse exact deterministic partitions within one block search.

    Reversible proposal routes repeatedly produce identical small distance
    matrices.  Raw matrix bytes, shape, dtype, and the clustering threshold
    form an exact key; Python dictionaries also compare the complete bytes on
    a hash collision.  The cache is block-local through
    :class:`ResidualInputWorkspace` and therefore remains naturally bounded.
    """

    if cache is None:
        return _cluster_residuals(distance, maximum_cluster_hamming)
    distance_value = np.ascontiguousarray(distance)
    key = (
        float(maximum_cluster_hamming),
        tuple(distance_value.shape),
        distance_value.dtype.str,
        distance_value.tobytes(order="C"),
    )
    cached = cache.get(key)
    if cached is None:
        cached = _cluster_residuals(distance, maximum_cluster_hamming)
        cache[key] = cached
    return cached


def _soft_record_log_odds(records: Sequence[ResidualRecord]) -> np.ndarray:
    """Precompute per-record/site log odds reused by overlapping clusters."""

    if not records:
        return np.empty((0, 0), dtype=np.float64)
    probability = np.stack(
        [record.soft_alt_probability for record in records], axis=0
    )
    tiny = np.finfo(np.float64).tiny
    upper = 1.0 - np.finfo(np.float64).eps
    probability = np.clip(probability, tiny, upper)
    return np.ascontiguousarray(
        np.log(probability) - np.log1p(-probability)
    )


@njit(cache=True, fastmath=False)
def _accumulate_soft_consensus_log_odds(
    member_indices: np.ndarray,
    record_sample_indices: np.ndarray,
    record_weights: np.ndarray,
    record_compatible: np.ndarray,
    record_log_odds: np.ndarray,
    keep_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate the established per-sample capped soft consensus exactly.

    ``member_indices`` is already ordered by sample and route by the caller.
    Moving only this repeatedly executed grouping and site loop into Numba
    avoids Python dictionaries and per-sample NumPy temporaries.  The
    posterior transform remains in :func:`_soft_consensus_candidate`, so the
    numerical model and its threshold comparisons are unchanged.
    """

    n_sites = keep_mask.shape[0]
    total_log_odds = np.zeros(n_sites, dtype=np.float64)
    n_contributing_samples = np.zeros(n_sites, dtype=np.int32)
    weighted_log_odds = np.zeros(n_sites, dtype=np.float64)
    weight_at_site = np.zeros(n_sites, dtype=np.float64)
    if member_indices.shape[0] == 0:
        return total_log_odds, n_contributing_samples

    current_sample = record_sample_indices[member_indices[0]]
    for member_position in range(member_indices.shape[0]):
        record_index = member_indices[member_position]
        sample_index = record_sample_indices[record_index]
        if sample_index != current_sample:
            for site in range(n_sites):
                weight = weight_at_site[site]
                if weight > 0.0:
                    sample_weight = 1.0 if weight > 1.0 else weight
                    total_log_odds[site] += sample_weight * (
                        weighted_log_odds[site] / weight
                    )
                    n_contributing_samples[site] += 1
                    weighted_log_odds[site] = 0.0
                    weight_at_site[site] = 0.0
            current_sample = sample_index

        record_weight = record_weights[record_index]
        for site in range(n_sites):
            if keep_mask[site] and record_compatible[record_index, site]:
                weighted_log_odds[site] += (
                    record_weight * record_log_odds[record_index, site]
                )
                weight_at_site[site] += record_weight

    for site in range(n_sites):
        weight = weight_at_site[site]
        if weight > 0.0:
            sample_weight = 1.0 if weight > 1.0 else weight
            total_log_odds[site] += sample_weight * (
                weighted_log_odds[site] / weight
            )
            n_contributing_samples[site] += 1
    return total_log_odds, n_contributing_samples


def _soft_consensus_candidate(
    records: Sequence[ResidualRecord],
    member_indices: Sequence[int],
    keep_mask: np.ndarray,
    candidate_call_probability: float,
    *,
    record_log_odds: np.ndarray | None = None,
    record_sample_indices: np.ndarray | None = None,
    record_weights: np.ndarray | None = None,
    record_compatible: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return a continuous posterior consensus with one vote per sample.

    Multiple assignment routes from one biological sample are averaged in
    responsibility-weighted log-odds space.  Their total weight is capped at
    one independently at every site, so duplicate routes cannot manufacture
    replication.  The returned candidate keeps
    the posterior allele probabilities rather than replacing them by 0/0.5/1.
    """

    n_sites = len(keep_mask)
    if record_log_odds is not None:
        log_odds_values = np.asarray(record_log_odds, dtype=np.float64)
        if log_odds_values.shape != (len(records), n_sites):
            raise ValueError("record_log_odds and records disagree")
    else:
        log_odds_values = None
    prepared_numeric = (
        log_odds_values is not None
        and record_sample_indices is not None
        and record_weights is not None
        and record_compatible is not None
    )
    if prepared_numeric:
        sample_indices_value = np.ascontiguousarray(
            record_sample_indices, dtype=np.int64
        )
        weights_value = np.ascontiguousarray(record_weights, dtype=np.float64)
        compatible_value = np.ascontiguousarray(
            record_compatible, dtype=np.bool_
        )
        if (
            sample_indices_value.shape != (len(records),)
            or weights_value.shape != (len(records),)
            or compatible_value.shape != (len(records), n_sites)
        ):
            raise ValueError("prepared soft-record arrays and records disagree")
        total_log_odds, n_contributing_samples = (
            _accumulate_soft_consensus_log_odds(
                np.ascontiguousarray(member_indices, dtype=np.int64),
                sample_indices_value,
                weights_value,
                compatible_value,
                np.ascontiguousarray(log_odds_values),
                np.ascontiguousarray(keep_mask, dtype=np.bool_),
            )
        )
    else:
        by_sample: dict[int, list[int]] = {}
        for index in member_indices:
            record = records[int(index)]
            by_sample.setdefault(record.sample_index, []).append(int(index))

        total_log_odds = np.zeros(n_sites, dtype=np.float64)
        n_contributing_samples = np.zeros(n_sites, dtype=np.int32)
        tiny = np.finfo(np.float64).tiny
        upper = 1.0 - np.finfo(np.float64).eps
        for sample_index in sorted(by_sample):
            weighted_log_odds = np.zeros(n_sites, dtype=np.float64)
            weight_at_site = np.zeros(n_sites, dtype=np.float64)
            for index in sorted(by_sample[sample_index]):
                record = records[index]
                informative = keep_mask & record.compatible_mask
                weight = float(record.responsibility_weight)
                if log_odds_values is None:
                    q1 = np.clip(record.soft_alt_probability, tiny, upper)
                    values = (
                        np.log(q1[informative])
                        - np.log1p(-q1[informative])
                    )
                else:
                    values = log_odds_values[index, informative]
                weighted_log_odds[informative] += weight * values
                weight_at_site[informative] += weight
            informative = weight_at_site > 0.0
            if not np.any(informative):
                continue
            mean_log_odds = np.zeros(n_sites, dtype=np.float64)
            mean_log_odds[informative] = (
                weighted_log_odds[informative] / weight_at_site[informative]
            )
            sample_weight = np.minimum(1.0, weight_at_site)
            total_log_odds[informative] += (
                sample_weight[informative] * mean_log_odds[informative]
            )
            n_contributing_samples[informative] += 1

    posterior = np.full(n_sites, 0.5, dtype=np.float64)
    observed = n_contributing_samples > 0
    positive = observed & (total_log_odds >= 0.0)
    negative = observed & ~positive
    posterior[positive] = 1.0 / (1.0 + np.exp(-total_log_odds[positive]))
    exponential = np.exp(total_log_odds[negative])
    posterior[negative] = exponential / (1.0 + exponential)
    confident = observed & (
        (posterior >= candidate_call_probability)
        | (posterior <= 1.0 - candidate_call_probability)
    )
    known_fraction = float(np.mean(confident[keep_mask]))
    return np.ascontiguousarray(posterior), known_fraction


def _candidate_known(
    candidate: np.ndarray, known_probability: float = 1.0
) -> np.ndarray:
    if not 0.5 < known_probability <= 1.0:
        raise ValueError("known_probability must lie in (0.5, 1]")
    values = np.asarray(candidate, dtype=np.float64)
    return (values <= 1.0 - known_probability + 1e-12) | (
        values >= known_probability - 1e-12
    )


def _closest_existing_reference(
    candidate: np.ndarray,
    existing: Sequence[np.ndarray],
    keep_mask: np.ndarray,
    minimum_joint_known_fraction: float,
    *,
    known_probability: float = 1.0,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return distance and bidirectional overlap for the nearest candidate."""

    n_kept = int(np.sum(keep_mask))
    minimum_joint = max(1, int(math.ceil(minimum_joint_known_fraction * n_kept)))
    candidate_known = _candidate_known(candidate, known_probability) & keep_mask
    n_candidate_known = int(np.sum(candidate_known))
    closest_distance: float | None = None
    closest_joint: float | None = None
    closest_candidate_coverage: float | None = None
    closest_other_coverage: float | None = None
    for other in existing:
        other = np.asarray(other, dtype=np.float64)
        other_known = _candidate_known(other, known_probability) & keep_mask
        joint = candidate_known & other_known
        n_joint = int(np.sum(joint))
        if n_joint < minimum_joint:
            continue
        distance = float(
            np.mean(np.rint(candidate[joint]) != np.rint(other[joint]))
        )
        joint_fraction = n_joint / n_kept
        if (
            closest_distance is None
            or distance < closest_distance - 1e-12
            or (
                abs(distance - closest_distance) <= 1e-12
                and joint_fraction > (closest_joint or 0.0)
            )
        ):
            closest_distance = distance
            closest_joint = joint_fraction
            closest_candidate_coverage = n_joint / max(1, n_candidate_known)
            closest_other_coverage = n_joint / max(1, int(np.sum(other_known)))
    return (
        closest_distance,
        closest_joint,
        closest_candidate_coverage,
        closest_other_coverage,
    )


@njit(cache=True, fastmath=False)
def _closest_existing_kernel(
    candidate: np.ndarray,
    existing: np.ndarray,
    keep_mask: np.ndarray,
    minimum_joint: int,
    n_kept: int,
    known_probability: float,
) -> tuple[bool, float, float, float, float]:
    """Exact allocation-free nearest-candidate scan for one prepared panel.

    The scalar reference constructs two known masks, a joint mask and several
    temporary reductions for every existing row.  Here the candidate-known
    mask and rounded calls are prepared once, while joint-known, mismatch and
    coverage counts are accumulated in one loop over each existing row.  The
    comparison tolerance, joint-overlap tie break and coverage denominators
    are deliberately identical to :func:`_closest_existing_reference`.
    """

    n_sites = candidate.shape[0]
    lower = 1.0 - known_probability + 1e-12
    upper = known_probability - 1e-12
    candidate_known = np.zeros(n_sites, dtype=np.bool_)
    candidate_calls = np.empty(n_sites, dtype=np.float64)
    n_candidate_known = 0
    for site in range(n_sites):
        value = candidate[site]
        known = keep_mask[site] and (value <= lower or value >= upper)
        candidate_known[site] = known
        if known:
            candidate_calls[site] = np.rint(value)
            n_candidate_known += 1

    found = False
    closest_distance = 0.0
    closest_joint = 0.0
    closest_candidate_coverage = 0.0
    closest_other_coverage = 0.0
    for other_index in range(existing.shape[0]):
        n_other_known = 0
        n_joint = 0
        n_mismatch = 0
        for site in range(n_sites):
            if not keep_mask[site]:
                continue
            other_value = existing[other_index, site]
            other_known = other_value <= lower or other_value >= upper
            if not other_known:
                continue
            n_other_known += 1
            if candidate_known[site]:
                n_joint += 1
                if candidate_calls[site] != np.rint(other_value):
                    n_mismatch += 1
        if n_joint < minimum_joint:
            continue
        distance = n_mismatch / n_joint
        joint_fraction = n_joint / n_kept
        if (
            not found
            or distance < closest_distance - 1e-12
            or (
                abs(distance - closest_distance) <= 1e-12
                and joint_fraction > closest_joint
            )
        ):
            found = True
            closest_distance = distance
            closest_joint = joint_fraction
            closest_candidate_coverage = n_joint / max(1, n_candidate_known)
            closest_other_coverage = n_joint / max(1, n_other_known)
    return (
        found,
        closest_distance,
        closest_joint,
        closest_candidate_coverage,
        closest_other_coverage,
    )


def _closest_existing(
    candidate: np.ndarray,
    existing: Sequence[np.ndarray],
    keep_mask: np.ndarray,
    minimum_joint_known_fraction: float,
    *,
    known_probability: float = 1.0,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return the exact nearest-candidate metrics using one compiled scan."""

    if not 0.5 < known_probability <= 1.0:
        raise ValueError("known_probability must lie in (0.5, 1]")
    candidate_value = np.ascontiguousarray(candidate, dtype=np.float64)
    keep_value = np.ascontiguousarray(keep_mask, dtype=np.bool_)
    n_kept = int(np.sum(keep_value))
    minimum_joint = max(
        1, int(math.ceil(minimum_joint_known_fraction * n_kept))
    )
    if len(existing) == 0:
        return (None, None, None, None)
    existing_value = np.ascontiguousarray(
        np.asarray(existing, dtype=np.float64)
    )
    found, distance, joint, candidate_coverage, other_coverage = (
        _closest_existing_kernel(
            candidate_value,
            existing_value,
            keep_value,
            minimum_joint,
            n_kept,
            float(known_probability),
        )
    )
    if not found:
        return (None, None, None, None)
    return (distance, joint, candidate_coverage, other_coverage)


def _is_confirmed_duplicate(
    closest: tuple[float | None, float | None, float | None, float | None],
    maximum_hamming: float,
    minimum_bidirectional_coverage: float,
) -> bool:
    distance, _joint, candidate_coverage, other_coverage = closest
    return bool(
        distance is not None
        and distance <= maximum_hamming + 1e-12
        and candidate_coverage is not None
        and candidate_coverage >= minimum_bidirectional_coverage
        and other_coverage is not None
        and other_coverage >= minimum_bidirectional_coverage
    )


def _candidate_discrete_rows(block_result: Any, n_sites: int) -> np.ndarray:
    """Return the canonical candidate-only rows with explicit provenance.

    Final discrete_haps rows share coordinates with final assignments and can
    therefore differ from the permissive proposal rows. Canonical results
    preserve those proposal rows explicitly; a result without that provenance
    is not a supported Stage-1 input.
    """

    precleanup = getattr(
        block_result, "precleanup_candidate_discrete_haps", None
    )
    if precleanup is None:
        raise ValueError(
            "canonical Stage-1 results require "
            "precleanup_candidate_discrete_haps"
        )
    source_name = "precleanup_candidate_discrete_haps"
    rows = np.asarray(precleanup)
    candidate_k = getattr(block_result, "precleanup_candidate_k", None)
    if candidate_k is None or isinstance(candidate_k, (bool, np.bool_)):
        raise ValueError(
            "pre-cleanup candidate provenance requires an integer "
            "precleanup_candidate_k"
        )
    try:
        candidate_k = int(np.asarray(candidate_k).item())
    except (TypeError, ValueError) as error:
        raise ValueError(
            "precleanup_candidate_k must be a scalar integer"
        ) from error
    raw_candidate_k = np.asarray(
        getattr(block_result, "precleanup_candidate_k")
    )
    if (
        raw_candidate_k.ndim != 0
        or not np.issubdtype(raw_candidate_k.dtype, np.integer)
    ):
        raise ValueError("precleanup_candidate_k must be a scalar integer")

    if rows.ndim != 2 or rows.shape[1] != n_sites:
        raise ValueError(
            f"block_result.{source_name} must have shape (K, sites)"
        )
    if candidate_k != rows.shape[0]:
        raise ValueError(
            "precleanup_candidate_k and "
            "precleanup_candidate_discrete_haps disagree"
        )
    if not np.all((rows == 0) | (rows == 1) | (rows == MASK)):
        raise ValueError(
            f"block_result.{source_name} must contain only 0, 1, or MASK"
        )
    return np.ascontiguousarray(rows)

def _add_usable_discrete_candidates(
    base: np.ndarray,
    block_result: Any,
    keep_mask: np.ndarray,
    usable_founder_known_fraction: float,
    minimum_joint_known_fraction: float,
    dedup_hamming_fraction: float,
    minimum_bidirectional_coverage: float,
) -> tuple[np.ndarray, int]:
    """Ensure every well-resolved pre-cleanup row reaches the permissive pool."""

    discrete = _candidate_discrete_rows(block_result, len(keep_mask))
    # Reversible complete-panel search supplies the fitted binary rows as both
    # the base and the explicit pre-cleanup candidate panel. Every scan below
    # would therefore rediscover an exact duplicate and add nothing.
    if base.shape == discrete.shape and np.array_equal(base, discrete):
        return base.copy(), 0
    minimum_known = int(
        math.ceil(usable_founder_known_fraction * int(np.sum(keep_mask)))
    )
    existing = [row.copy() for row in base]
    n_added = 0
    for row in discrete:
        known = keep_mask & ((row == 0) | (row == 1))
        if int(np.sum(known)) < minimum_known:
            continue
        candidate = np.full(len(keep_mask), 0.5, dtype=np.float64)
        candidate[known] = row[known]
        closest = _closest_existing(
            candidate, existing, keep_mask, minimum_joint_known_fraction
        )
        if _is_confirmed_duplicate(
            closest, dedup_hamming_fraction, minimum_bidirectional_coverage
        ):
            continue
        existing.append(candidate)
        n_added += 1
    if not n_added:
        return base.copy(), 0
    return np.ascontiguousarray(np.stack(existing)), n_added


def _soft_cluster_sources(
    records: Sequence[ResidualRecord],
    keep_mask: np.ndarray,
    minimum_joint_known_fraction: float,
    maximum_cluster_hamming: float,
    cluster_cache: dict[tuple[Any, ...], Any] | None = None,
) -> tuple[
    tuple[tuple[str, str, int | None, tuple[int, ...], float], ...],
    int,
    int,
    int,
    int,
    int,
]:
    """Build global and/or assignment-route soft clusters.

    Soft modes deliberately expose cluster consensuses only. Member-level
    singleton candidates are not emitted merely because all-assignment
    enumeration created a record.
    """

    sources: list[tuple[str, str, int | None, tuple[int, ...], float]] = []
    n_residual_clusters = 0
    n_split_clusters = 0
    n_hdbscan = 0
    n_complete_link = 0
    initial_noise = 0
    global_distance: np.ndarray | None = None
    if len(records) >= 2:
        global_distance = _missing_aware_hamming_distance_matrix(
            records, keep_mask, minimum_joint_known_fraction
        )
        clusters, _noise, n_initial_noise = _cluster_residuals_cached(
            global_distance, maximum_cluster_hamming, cluster_cache
        )
        initial_noise += n_initial_noise
        for source_kind, label, members, max_distance in clusters:
            sources.append(
                (
                    f"soft_residual_{source_kind}",
                    PROPOSAL_MODE_SOFT_RESIDUAL,
                    label,
                    members,
                    max_distance,
                )
            )
            n_residual_clusters += 1
            n_hdbscan += int(source_kind == "hdbscan_cluster")
            n_complete_link += int(
                source_kind == "complete_link_noise_cluster"
            )

    by_route: dict[tuple[int, int], list[int]] = {}
    for index, record in enumerate(records):
        partner = record.dominant_partner_index
        if partner is None:
            continue
        by_route.setdefault(
            (record.subtractor_index, int(partner)), []
        ).append(index)
    for route in sorted(by_route):
        member_pool = tuple(sorted(by_route[route]))
        if len(member_pool) < 2:
            continue
        if global_distance is None:
            # With fewer than two records the route cannot cluster; otherwise
            # combined construction already owns the global distance matrix.
            route_records = tuple(records[index] for index in member_pool)
            distance = _missing_aware_hamming_distance_matrix(
                route_records, keep_mask, minimum_joint_known_fraction
            )
        else:
            distance = global_distance[np.ix_(member_pool, member_pool)]
        clusters, _noise, n_initial_noise = _cluster_residuals_cached(
            distance, maximum_cluster_hamming, cluster_cache
        )
        initial_noise += n_initial_noise
        for source_kind, label, local_members, max_distance in clusters:
            members = tuple(member_pool[index] for index in local_members)
            sources.append(
                (
                    f"soft_split_{source_kind}",
                    PROPOSAL_MODE_SOFT_SPLIT,
                    label,
                    members,
                    max_distance,
                )
            )
            n_split_clusters += 1
            n_hdbscan += int(source_kind == "hdbscan_cluster")
            n_complete_link += int(
                source_kind == "complete_link_noise_cluster"
            )
    return (
        tuple(sources),
        n_residual_clusters,
        n_split_clusters,
        n_hdbscan,
        n_complete_link,
        initial_noise,
    )


def augment_combined_soft_candidates(
    block_result: Any,
    reads_array: np.ndarray,
    *,
    base_candidates: np.ndarray | None = None,
    keep_flags: np.ndarray | None = None,
    read_error_probability: float = DEFAULT_READ_ERROR_PROBABILITY,
    usable_founder_known_fraction: float = 0.80,
    residual_hard_probability: float = 0.80,
    minimum_residual_joint_known_fraction: float = 0.10,
    maximum_cluster_hamming: float = 0.10,
    candidate_call_probability: float = 0.90,
    minimum_candidate_known_fraction: float = 0.80,
    minimum_dedup_joint_known_fraction: float = 0.60,
    minimum_dedup_bidirectional_coverage: float = 0.95,
    dedup_hamming_fraction: float = CANDIDATE_DEDUP_HAMMING_PERCENT / 100.0,
    minimum_soft_responsibility: float = 0.25,
    minimum_soft_unique_sample_support: int = 2,
    minimum_soft_effective_sample_support: float = 1.50,
    residual_input_workspace: ResidualInputWorkspace | None = None,
    binary_panel_fast_path: bool = False,
) -> CandidatePoolAugmentation:
    """Add combined posterior-residual proposals to a candidate pool.

    Global and route-specific soft residual clusters are both evaluated. The
    returned pool is intentionally permissive: this function does not decide K
    or accept a candidate as a founder. Final inclusion is delegated to the
    downstream cavity selector. Every proposal uses only
    ``reads_array``; in cross-validation that must be the training partition.
    Multi-sample support is required by default and consensus allele
    probabilities remain soft for downstream model selection.

    Repeated internal calls for fitted panels from the same block can pass a
    :class:`ResidualInputWorkspace` made by :func:`prepare_residual_inputs`.
    This reuses only block-invariant genotype likelihood, floored log
    likelihood, and depth arrays; candidate-dependent quantities are still
    recomputed exactly.

    ``binary_panel_fast_path=True`` gathers the exact likelihood term selected
    by each hard pair/site dosage when every usable panel allele is known and
    binary. It preserves generic reduction and proposal ordering, falling back
    automatically for incomplete or uncertain panels.
    """

    if not isinstance(binary_panel_fast_path, (bool, np.bool_)):
        raise TypeError("binary_panel_fast_path must be boolean")
    binary_panel_fast_path = bool(binary_panel_fast_path)

    reads = np.asarray(reads_array)
    if reads.ndim != 3 or reads.shape[2] != 2:
        raise ValueError("reads_array must have shape (samples, sites, 2)")
    n_sites = reads.shape[1]
    if keep_flags is None:
        source_flags = getattr(block_result, "keep_flags", None)
        keep_mask = (
            np.ones(n_sites, dtype=bool)
            if source_flags is None
            else np.asarray(source_flags) > 0
        )
    else:
        keep_mask = np.asarray(keep_flags) > 0
    if keep_mask.shape != (n_sites,) or not np.any(keep_mask):
        raise ValueError("keep_flags must retain at least one site")

    probability_parameters = {
        "read_error_probability": read_error_probability,
        "usable_founder_known_fraction": usable_founder_known_fraction,
        "residual_hard_probability": residual_hard_probability,
        "minimum_residual_joint_known_fraction": minimum_residual_joint_known_fraction,
        "maximum_cluster_hamming": maximum_cluster_hamming,
        "candidate_call_probability": candidate_call_probability,
        "minimum_candidate_known_fraction": minimum_candidate_known_fraction,
        "minimum_dedup_joint_known_fraction": minimum_dedup_joint_known_fraction,
        "minimum_dedup_bidirectional_coverage": minimum_dedup_bidirectional_coverage,
        "dedup_hamming_fraction": dedup_hamming_fraction,
        "minimum_soft_responsibility": minimum_soft_responsibility,
    }
    for name, value in probability_parameters.items():
        if not 0.0 < float(value) < 1.0:
            raise ValueError(f"{name} must lie in (0, 1)")
    if read_error_probability >= 0.5:
        raise ValueError("read_error_probability must be less than 0.5")
    if residual_hard_probability <= 0.5 or candidate_call_probability <= 0.5:
        raise ValueError("calling probabilities must exceed 0.5")
    if (
        int(minimum_soft_unique_sample_support)
        != minimum_soft_unique_sample_support
        or int(minimum_soft_unique_sample_support) < 1
    ):
        raise ValueError(
            "minimum_soft_unique_sample_support must be a positive integer"
        )
    if not 0.0 < float(minimum_soft_effective_sample_support) <= float(
        minimum_soft_unique_sample_support
    ):
        raise ValueError(
            "minimum_soft_effective_sample_support must lie in (0, unique support]"
        )

    workspace = (
        prepare_residual_inputs(reads, read_error_probability)
        if residual_input_workspace is None
        else _validate_residual_input_workspace(
            residual_input_workspace, reads.shape, read_error_probability
        )
    )
    input_base = _validate_base_candidates(base_candidates, block_result, n_sites)
    base, n_discrete_added = _add_usable_discrete_candidates(
        input_base,
        block_result,
        keep_mask,
        usable_founder_known_fraction,
        minimum_dedup_joint_known_fraction,
        dedup_hamming_fraction,
        minimum_dedup_bidirectional_coverage,
    )
    soft_records = _extract_soft_residual_records(
        block_result, reads, keep_mask, read_error_probability,
        usable_founder_known_fraction, residual_hard_probability,
        minimum_soft_responsibility, residual_input_workspace=workspace,
        binary_panel_fast_path=binary_panel_fast_path,
    )
    (
        soft_sources, n_soft_residual_clusters, n_soft_split_clusters,
        n_soft_hdbscan_clusters, n_soft_complete_link_clusters,
        soft_hdbscan_initial_noise,
    ) = _soft_cluster_sources(
        soft_records, keep_mask, minimum_residual_joint_known_fraction,
        maximum_cluster_hamming, workspace.cluster_cache,
    )


    existing_capacity = len(base) + len(soft_sources)
    existing = np.empty((existing_capacity, len(keep_mask)), dtype=np.float64)
    existing_count = len(base)
    existing[:existing_count] = base
    emitted: list[np.ndarray] = []
    emitted_candidate_digests: list[str] = []
    emitted_source_classes: list[str] = []
    emitted_diagnostic_indices: list[int] = []
    n_soft_emitted = 0
    diagnostics: list[ProposalDiagnostic] = []
    prepared_soft: list[tuple[Any, ...]] = []
    soft_log_odds = _soft_record_log_odds(soft_records)
    soft_sample_indices = np.fromiter(
        (record.sample_index for record in soft_records),
        dtype=np.int64,
        count=len(soft_records),
    )
    soft_weights = np.fromiter(
        (record.responsibility_weight for record in soft_records),
        dtype=np.float64,
        count=len(soft_records),
    )
    soft_compatible = (
        np.ascontiguousarray(
            np.stack(
                [record.compatible_mask for record in soft_records], axis=0
            ),
            dtype=np.bool_,
        )
        if soft_records
        else np.empty((0, len(keep_mask)), dtype=np.bool_)
    )
    consensus_by_members: dict[tuple[int, ...], tuple[Any, ...]] = {}
    for source_kind, source_mode, label, members, max_distance in soft_sources:
        ordered_members = tuple(
            sorted(
                members,
                key=lambda index: (
                    soft_records[index].sample_index,
                    soft_records[index].subtractor_index,
                    soft_records[index].dominant_partner_index,
                    index,
                ),
            )
        )
        cached_consensus = consensus_by_members.get(ordered_members)
        if cached_consensus is None:
            unique_samples = tuple(
                sorted(
                    {
                        soft_records[index].sample_index
                        for index in ordered_members
                    }
                )
            )
            effective_support = _effective_unique_sample_support(
                soft_records, ordered_members
            )
            candidate, known_fraction = _soft_consensus_candidate(
                soft_records,
                ordered_members,
                keep_mask,
                candidate_call_probability,
                record_log_odds=soft_log_odds,
                record_sample_indices=soft_sample_indices,
                record_weights=soft_weights,
                record_compatible=soft_compatible,
            )
            cached_consensus = (
                _candidate_digest(candidate),
                unique_samples,
                effective_support,
                candidate,
                known_fraction,
            )
            consensus_by_members[ordered_members] = cached_consensus
        (
            candidate_digest,
            unique_samples,
            effective_support,
            candidate,
            known_fraction,
        ) = cached_consensus
        prepared_soft.append(
            (
                candidate_digest,
                source_kind,
                source_mode,
                label,
                ordered_members,
                max_distance,
                unique_samples,
                effective_support,
                candidate,
                known_fraction,
            )
        )
    prepared_soft.sort(key=lambda item: (item[0], item[1], item[3] or -1))

    for prepared in prepared_soft:
        (
            candidate_digest,
            source_kind,
            source_mode,
            label,
            ordered_members,
            max_distance,
            unique_samples,
            effective_support,
            candidate,
            known_fraction,
        ) = prepared
        closest = _closest_existing(
            candidate,
            existing[:existing_count],
            keep_mask,
            minimum_dedup_joint_known_fraction,
            known_probability=candidate_call_probability,
        )
        (
            closest_distance,
            closest_joint,
            closest_candidate_coverage,
            closest_other_coverage,
        ) = closest
        if len(unique_samples) < int(minimum_soft_unique_sample_support):
            emitted_flag = False
            reason = "insufficient_unique_sample_support"
        elif effective_support + 1e-12 < minimum_soft_effective_sample_support:
            emitted_flag = False
            reason = "insufficient_effective_sample_support"
        elif known_fraction < minimum_candidate_known_fraction:
            emitted_flag = False
            reason = "insufficient_candidate_known_fraction"
        elif _is_confirmed_duplicate(
            closest,
            dedup_hamming_fraction,
            minimum_dedup_bidirectional_coverage,
        ):
            emitted_flag = False
            reason = "duplicate_existing_candidate"
        else:
            emitted_flag = True
            reason = "emitted_for_cavity_selection"
            emitted.append(candidate)
            emitted_candidate_digests.append(candidate_digest)
            existing[existing_count] = candidate
            existing_count += 1
            emitted_source_classes.append(source_mode)
            emitted_diagnostic_indices.append(len(diagnostics))
            n_soft_emitted += 1

        diagnostics.append(
            ProposalDiagnostic(
                source_kind=source_kind,
                sample_indices=unique_samples,
                cluster_label=label,
                unique_sample_support=len(unique_samples),
                max_pairwise_hamming=max_distance,
                known_fraction=known_fraction,
                closest_existing_hamming=closest_distance,
                closest_existing_joint_known_fraction=closest_joint,
                closest_existing_candidate_coverage=closest_candidate_coverage,
                closest_existing_other_coverage=closest_other_coverage,
                emitted=emitted_flag,
                reason=reason,
                proposal_mode=source_mode,
                subtractor_indices=tuple(
                    soft_records[index].subtractor_index
                    for index in ordered_members
                ),
                dominant_partner_indices=tuple(
                    (
                        -1
                        if soft_records[index].dominant_partner_index is None
                        else int(soft_records[index].dominant_partner_index)
                    )
                    for index in ordered_members
                ),
                dominant_partner_probabilities=tuple(
                    (
                        0.0
                        if soft_records[index].dominant_partner_probability is None
                        else float(
                            soft_records[index].dominant_partner_probability
                        )
                    )
                    for index in ordered_members
                ),
                responsibility_weights=tuple(
                    float(soft_records[index].responsibility_weight)
                    for index in ordered_members
                ),
                effective_sample_support=effective_support,
                canonical_candidate_digest=candidate_digest,
            )
        )

    soft_clustered_members = {
        member for _, _, _, members, _ in soft_sources for member in members
    }

    if emitted:
        combined = np.ascontiguousarray(np.vstack([base, *emitted]))
    else:
        combined = base.copy()
    base_source_classes = (
        ("input_base_candidate",) * len(input_base)
        + ("usable_discrete_candidate",) * n_discrete_added
    )
    candidate_source_classes = base_source_classes + tuple(
        emitted_source_classes
    )
    candidate_diagnostic_indices = (None,) * len(base) + tuple(
        emitted_diagnostic_indices
    )
    if (
        len(candidate_source_classes) != len(combined)
        or len(candidate_diagnostic_indices) != len(combined)
    ):
        raise AssertionError("candidate provenance and candidate rows disagree")
    candidate_digests = tuple(_candidate_digest(row) for row in base) + tuple(
        emitted_candidate_digests
    )
    candidate_provenance = tuple(
        CandidateProvenance(
            candidate_index=index,
            source_class=candidate_source_classes[index],
            canonical_candidate_digest=candidate_digests[index],
            proposal_diagnostic_index=candidate_diagnostic_indices[index],
        )
        for index in range(len(combined))
    )

    return CandidatePoolAugmentation(
        candidates=combined,
        n_input_base_candidates=len(input_base),
        n_discrete_candidates_added=n_discrete_added,
        n_base_candidates=len(base),
        n_residual_records=len(soft_records),
        n_residual_clusters=n_soft_residual_clusters + n_soft_split_clusters,
        n_hdbscan_clusters=n_soft_hdbscan_clusters,
        n_complete_link_clusters=n_soft_complete_link_clusters,
        n_hdbscan_initial_noise=soft_hdbscan_initial_noise,
        n_unclustered_singletons=(
            len(soft_records) - len(soft_clustered_members)
        ),
        n_emitted_candidates=len(emitted),
        residual_records=soft_records,
        proposal_diagnostics=tuple(diagnostics),
        n_soft_records=len(soft_records),
        n_soft_residual_clusters=n_soft_residual_clusters,
        n_soft_split_clusters=n_soft_split_clusters,
        n_soft_candidates_emitted=n_soft_emitted,
        candidate_provenance=candidate_provenance,
    )


def _selftest() -> None:
    reads = np.asarray([[[20, 0]], [[10, 10]], [[0, 20]], [[0, 0]]])
    likelihood = allele_depths_to_likelihoods(reads)
    assert np.allclose(likelihood[3, 0], 1.0 / 3.0)
    keep = np.ones(1, dtype=bool)
    for subtractor, expected in ((0.0, (0, 1, 1, MASK)), (1.0, (0, 0, 1, MASK))):
        _, hard, compatible = _soft_conditional_residual(
            likelihood[:, 0], np.sum(reads[:, 0], axis=1),
            np.full(4, subtractor), np.ones(4, dtype=bool), 0.8,
        )
        assert tuple(hard) == expected
        assert tuple(compatible) == (True, True, True, False)

    rng = np.random.default_rng(20260809)
    hard_calls = rng.integers(0, 2, (17, 131), dtype=np.int8)
    hard_calls[rng.random(hard_calls.shape) < 0.23] = MASK
    kept = rng.random(131) > 0.17
    minimum_joint = max(1, int(math.ceil(0.1 * np.sum(kept))))
    reference, _ = _missing_aware_hamming_reference_kernel(
        hard_calls, kept, minimum_joint, int(np.sum(kept)), int(MASK)
    )
    packed = _missing_aware_hamming_distance_kernel(
        hard_calls, kept, minimum_joint, int(MASK)
    )
    assert np.array_equal(packed, reference)

    def record(sample_index: int, q: float) -> ResidualRecord:
        return ResidualRecord(
            sample_index=sample_index, assignments=(0, 1), subtractor_index=0,
            unexplained_assignment=1, soft_alt_probability=np.asarray([q]),
            hard_calls=np.asarray([int(q >= 0.5)], np.int8),
            compatible_mask=np.ones(1, bool), hard_known_fraction=1.0,
            responsibility_weight=1.0, dominant_partner_index=1,
            dominant_partner_probability=1.0,
        )
    duplicate = (record(0, 0.8), record(0, 0.8))
    independent = (record(0, 0.8), record(1, 0.8))
    duplicate_q, duplicate_known = _soft_consensus_candidate(duplicate, (0, 1), keep, 0.9)
    independent_q, independent_known = _soft_consensus_candidate(independent, (0, 1), keep, 0.9)
    assert np.isclose(duplicate_q[0], 0.8) and duplicate_known == 0.0
    assert independent_q[0] > 0.9 and independent_known == 1.0
    assert _effective_unique_sample_support(duplicate, (0, 1)) == 1.0
    assert _effective_unique_sample_support(independent, (0, 1)) == 2.0

    n_sites = 80
    common = np.zeros(n_sites, dtype=np.int8)
    alternate = (np.arange(n_sites) % 2).astype(np.int8)
    near_clone = alternate.copy()
    near_clone[np.arange(0, n_sites, 10)] ^= 1
    discrete = np.vstack([common, alternate])
    assignments = np.asarray([[0, 1], [0, 1], [0, 0], [1, 1]], np.int64)
    block_reads = np.zeros((4, n_sites, 2), np.int64)
    diplotypes = ((common, near_clone), (common, near_clone), (common, common), (alternate, alternate))
    for sample_index, (first, second) in enumerate(diplotypes):
        genotype = first + second
        block_reads[sample_index, :, 0] = np.where(genotype == 0, 30, np.where(genotype == 1, 15, 0))
        block_reads[sample_index, :, 1] = np.where(genotype == 0, 0, np.where(genotype == 1, 15, 30))
    block = SimpleNamespace(
        discrete_haps=discrete, precleanup_candidate_discrete_haps=discrete,
        precleanup_candidate_k=2, pair_assignments=assignments, K_final=2,
        keep_flags=np.ones(n_sites, np.int8),
        haplotypes={
            0: np.column_stack([1.0 - common, common]),
            1: np.column_stack([1.0 - alternate, alternate]),
        },
    )
    generic = augment_combined_soft_candidates(block, block_reads)
    repeated = augment_combined_soft_candidates(block, block_reads)
    prepared = augment_combined_soft_candidates(
        block, block_reads, residual_input_workspace=prepare_residual_inputs(block_reads)
    )
    binary = augment_combined_soft_candidates(
        block, block_reads, residual_input_workspace=prepare_residual_inputs(block_reads),
        binary_panel_fast_path=True,
    )
    for other in (repeated, prepared, binary):
        assert np.array_equal(other.candidates, generic.candidates)
        assert other.proposal_diagnostics == generic.proposal_diagnostics
        assert other.candidate_provenance == generic.candidate_provenance
    emitted = generic.candidates[generic.n_base_candidates:]
    assert any(np.array_equal((row >= 0.5).astype(np.int8), near_clone) for row in emitted)
    assert generic.n_soft_records == generic.n_residual_records
    assert generic.n_soft_candidates_emitted == generic.n_emitted_candidates
    assert generic.n_soft_residual_clusters > 0 and generic.n_soft_split_clusters > 0
    assert len(generic.candidate_provenance) == len(generic.candidates)
    for index, (provenance, candidate) in enumerate(zip(generic.candidate_provenance, generic.candidates)):
        assert provenance.candidate_index == index
        assert provenance.canonical_candidate_digest == _candidate_digest(candidate)

    singleton = SimpleNamespace(
        discrete_haps=discrete, precleanup_candidate_discrete_haps=discrete,
        precleanup_candidate_k=2, pair_assignments=assignments[[0, 2, 3]],
        K_final=2, keep_flags=np.ones(n_sites, np.int8), haplotypes=block.haplotypes,
    )
    singleton_result = augment_combined_soft_candidates(
        singleton, block_reads[[0, 2, 3]]
    )
    assert singleton_result.n_soft_split_clusters == 0

    zero_depth = augment_combined_soft_candidates(block, np.zeros_like(block_reads))
    assert zero_depth.n_emitted_candidates == 0
    assert zero_depth.candidates.shape == (2, n_sites)
    print("bhd_candidate_pool selftest: PASS")


if __name__ == "__main__":
    _selftest()
