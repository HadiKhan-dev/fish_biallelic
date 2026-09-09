"""discovery / candidates for the canonical reconstruction pipeline."""
from __future__ import annotations


import hashlib
import math

from typing import Any, Sequence
import numpy as np
from numba import njit
import haplotype_reconstruction.core.config as core_config

def _effective_unique_sample_support(
    records: Sequence[discovery_residuals.ResidualRecord], member_indices: Sequence[int]
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
    records: Sequence[discovery_residuals.ResidualRecord],
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
        int(discovery_objectives.MASK),
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


def _soft_record_log_odds(records: Sequence[discovery_residuals.ResidualRecord]) -> np.ndarray:
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
    records: Sequence[discovery_residuals.ResidualRecord],
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
    if not np.all((rows == 0) | (rows == 1) | (rows == discovery_objectives.MASK)):
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
    records: Sequence[discovery_residuals.ResidualRecord],
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
                    discovery_residuals.PROPOSAL_MODE_SOFT_RESIDUAL,
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
                    discovery_residuals.PROPOSAL_MODE_SOFT_SPLIT,
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
    read_error_probability: float = core_config.DEFAULT_READ_ERROR_PROBABILITY,
    usable_founder_known_fraction: float = 0.80,
    residual_hard_probability: float = 0.80,
    minimum_residual_joint_known_fraction: float = 0.10,
    maximum_cluster_hamming: float = 0.10,
    candidate_call_probability: float = 0.90,
    minimum_candidate_known_fraction: float = 0.80,
    minimum_dedup_joint_known_fraction: float = 0.60,
    minimum_dedup_bidirectional_coverage: float = 0.95,
    dedup_hamming_fraction: float = core_config.CANDIDATE_DEDUP_HAMMING_PERCENT / 100.0,
    minimum_soft_responsibility: float = 0.25,
    minimum_soft_unique_sample_support: int = 2,
    minimum_soft_effective_sample_support: float = 1.50,
    residual_input_workspace: discovery_residuals.ResidualInputWorkspace | None = None,
    binary_panel_fast_path: bool = False,
) -> discovery_residuals.CandidatePoolAugmentation:
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
        discovery_residuals.prepare_residual_inputs(reads, read_error_probability)
        if residual_input_workspace is None
        else discovery_residuals._validate_residual_input_workspace(
            residual_input_workspace, reads.shape, read_error_probability
        )
    )
    input_base = discovery_residuals._validate_base_candidates(base_candidates, block_result, n_sites)
    base, n_discrete_added = _add_usable_discrete_candidates(
        input_base,
        block_result,
        keep_mask,
        usable_founder_known_fraction,
        minimum_dedup_joint_known_fraction,
        dedup_hamming_fraction,
        minimum_dedup_bidirectional_coverage,
    )
    soft_records = discovery_residuals._extract_soft_residual_records(
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
    diagnostics: list[discovery_residuals.ProposalDiagnostic] = []
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
            discovery_residuals.ProposalDiagnostic(
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
        discovery_residuals.CandidateProvenance(
            candidate_index=index,
            source_class=candidate_source_classes[index],
            canonical_candidate_digest=candidate_digests[index],
            proposal_diagnostic_index=candidate_diagnostic_indices[index],
        )
        for index in range(len(combined))
    )

    return discovery_residuals.CandidatePoolAugmentation(
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

import haplotype_reconstruction.discovery.objectives as discovery_objectives
import haplotype_reconstruction.discovery.residuals as discovery_residuals
