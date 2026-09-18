"""Read-supported residual evidence and candidate proposal diagnostics."""
from __future__ import annotations


from dataclasses import dataclass, field

import math

from typing import Any, Mapping, Sequence
import numpy as np
from numba import njit
import haplotype_reconstruction.core.config as core_config

PROPOSAL_MODE_SOFT_RESIDUAL = "soft_residual"


PROPOSAL_MODE_SOFT_SPLIT = "soft_split"


def allele_depths_to_likelihoods(
    reads: np.ndarray,
    read_error_probability: float=core_config.DEFAULT_READ_ERROR_PROBABILITY,
) -> np.ndarray:
    """Return normalized raw P(reads | genotype) for genotypes 0, 1, 2.

    No population-frequency or HWE prior is applied.  Per-cell normalization
    removes only a model-independent constant.  Zero-depth cells are exactly
    uniform, hence make the same constant contribution to every model.
    """

    return core_genotypes.allele_depths_to_raw_genotype_likelihoods(
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
            array[:,:, 1],
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
    genotype[:,:, 0] = (1.0 - qi) * (1.0 - qj)
    genotype[:,:, 2] = qi * qj
    genotype[:,:, 1] = 1.0 - genotype[:,:, 0] - genotype[:,:, 2]
    diagonal = pair_i == pair_j
    if np.any(diagonal):
        genotype[diagonal,:, 0] = 1.0 - qi[diagonal]
        genotype[diagonal,:, 1] = 0.0
        genotype[diagonal,:, 2] = qi[diagonal]
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
    read_error_probability: float=core_config.DEFAULT_READ_ERROR_PROBABILITY,
    *,
    likelihood: np.ndarray | None=None,
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
        likelihood = core_genotypes.validate_normalized_genotype_evidence(
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
            alt[np.isclose(alt, discovery_objectives.MASK, rtol=0.0, atol=1e-12)] = 0.5
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
    prepared_log_likelihood: np.ndarray | None=None,
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
    kept_log_likelihood = log_likelihood[:, keep_mask,:]
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
    residual_input_workspace: ResidualInputWorkspace | None=None,
    binary_panel_fast_path: bool=False,
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
    founder_known = ((discrete == 0) | (discrete == 1)) & keep_mask[None,:]
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
            likelihood[:, keep_mask,:],
            genotype[:, keep_mask,:],
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
        int(discovery_objectives.MASK),
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
                    np.mean(hard[keep_mask] != discovery_objectives.MASK)
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

import haplotype_reconstruction.core.genotypes as core_genotypes
import haplotype_reconstruction.discovery.objectives as discovery_objectives
