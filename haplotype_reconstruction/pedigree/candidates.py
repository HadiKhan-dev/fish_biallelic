"""Candidate ancestry trajectories and chromosome-level screening likelihoods."""
from __future__ import annotations


from dataclasses import dataclass


from typing import Any

from numba import prange
import numpy as np

import haplotype_reconstruction.pedigree.models as pedigree_models

@dataclass(frozen=True)
class ChromosomeLikelihoods:
    """Comparable forward likelihoods and ancestry depth for one contig."""

    zero_observed: np.ndarray
    one_observed: np.ndarray
    two_observed: np.ndarray
    ancestry_junction_counts: np.ndarray
    ancestry_callable_haplotype_bins: np.ndarray
    one_parent_identity_information: np.ndarray | None = None
    two_parent_edge_information: np.ndarray | None = None
    candidate_source_mode_requested: str = "hard_painted"
    candidate_source_mode_applied: str = "hard_painted"
    candidate_source_fallback: bool = False
    candidate_source_fallback_reason: str = ""
    complete_founder_marker_count: int | None = None
    excluded_founder_marker_count: int | None = None
    candidate_source_available: np.ndarray | None = None
    candidate_source_informative_marker_count: np.ndarray | None = None
    child_complete_informative_marker_count: np.ndarray | None = None
    candidate_initial_max_probability: np.ndarray | None = None
    candidate_initial_point_mass: np.ndarray | None = None
    peak_streamed_tensor_bytes: int = 0
    candidate_source_posterior: Any = None
    edge_matched_bins: np.ndarray | None = None
    edge_exposed_bins: np.ndarray | None = None
    pair_explained_bins: np.ndarray | None = None
    pair_exposed_bins: np.ndarray | None = None
    structure_total_bins: float | None = None


@pedigree_models.njit(cache=True, parallel=True)
def _local_ibs_class_kernel(founders):
    """Assign first-occurrence local classes independently by cache bin."""
    n_states, n_bins, n_snps = founders.shape
    mapping = np.empty((n_bins, n_states), dtype=np.int16)
    class_counts = np.empty(n_bins, dtype=np.int16)
    pooled_founders = np.full(
        (n_states, n_bins, n_snps), -1, dtype=np.int8
    )
    active = np.zeros((n_bins, n_states), dtype=np.bool_)
    for block in prange(n_bins):
        number_of_classes = 0
        for state in range(n_states):
            local_class = -1
            for previous_state in range(state):
                equivalent = True
                for snp in range(n_snps):
                    if (
                        founders[state, block, snp]
                        != founders[previous_state, block, snp]
                    ):
                        equivalent = False
                        break
                if equivalent:
                    local_class = int(mapping[block, previous_state])
                    break
            if local_class < 0:
                local_class = number_of_classes
                number_of_classes += 1
                active[block, local_class] = True
                for snp in range(n_snps):
                    pooled_founders[local_class, block, snp] = (
                        founders[state, block, snp]
                    )
            mapping[block, state] = local_class
        class_counts[block] = number_of_classes
    return mapping, class_counts, pooled_founders, active


@pedigree_models.njit(cache=True, parallel=True)
def _pool_local_label_kernel(labels, mapping):
    n_samples, n_bins, _ = labels.shape
    pooled_labels = np.full_like(labels, -1, dtype=np.int16)
    for flat_index in prange(n_samples * n_bins):
        sample = flat_index // n_bins
        block = flat_index - sample * n_bins
        for track in range(2):
            label = int(labels[sample, block, track])
            if label >= 0:
                pooled_labels[sample, block, track] = mapping[block, label]
    return pooled_labels


@pedigree_models.njit(cache=True)
def _unique_trajectory_state_kernel(mapping):
    """Deduplicate complete founder trajectories in first-occurrence order."""
    n_bins, n_states = mapping.shape
    unique_states = np.empty(n_states, dtype=np.int64)
    number_of_unique_states = 0
    for state in range(n_states):
        duplicate = False
        for unique_index in range(number_of_unique_states):
            representative = int(unique_states[unique_index])
            equivalent = True
            for block in range(n_bins):
                if mapping[block, state] != mapping[block, representative]:
                    equivalent = False
                    break
            if equivalent:
                duplicate = True
                break
        if not duplicate:
            unique_states[number_of_unique_states] = state
            number_of_unique_states += 1
    return unique_states[:number_of_unique_states]


@pedigree_models.njit(cache=True, parallel=True)
def _continuation_bridge_kernel(
    mapping,
    class_counts,
    unique_trajectory_states,
    maximum_classes,
):
    n_bins = mapping.shape[0]
    destination_counts = np.zeros(
        (n_bins, maximum_classes, maximum_classes), dtype=np.int64
    )
    continuation_bridge = np.zeros(
        (n_bins, maximum_classes, maximum_classes), dtype=np.float64
    )
    for block in prange(1, n_bins):
        for unique_index in range(len(unique_trajectory_states)):
            state = int(unique_trajectory_states[unique_index])
            previous = int(mapping[block - 1, state])
            current = int(mapping[block, state])
            destination_counts[block, previous, current] += 1
        for previous in range(int(class_counts[block - 1])):
            denominator = 0
            for current in range(int(class_counts[block])):
                denominator += destination_counts[block, previous, current]
            for current in range(int(class_counts[block])):
                count = destination_counts[block, previous, current]
                if count > 0:
                    continuation_bridge[block, previous, current] = (
                        float(count) / float(denominator)
                    )
    return continuation_bridge


def _pool_local_ibs_states(
    stacked_labels: np.ndarray,
    founder_alleles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool numeric labels that are locally IBS-equivalent.

    Reconstructed founder-label numbers are not biological identities.  On
    every cache bin, founder rows with the same observed allele vector are
    collapsed into one local state before external-parent frequencies and
    transitions are estimated.  The transition model is then estimated
    between these pooled local classes, so splitting one IBS haplotype into
    duplicate numeric labels cannot manufacture ancestry switches.
    """
    labels = np.asarray(stacked_labels, dtype=np.int16)
    founders = np.asarray(founder_alleles, dtype=np.int8)
    mapping, class_counts, pooled_founders, active = (
        _local_ibs_class_kernel(founders)
    )
    maximum_classes = int(np.max(class_counts))
    pooled_founders = np.ascontiguousarray(
        pooled_founders[:maximum_classes]
    )
    active = np.ascontiguousarray(active[:,:maximum_classes])
    pooled_labels = _pool_local_label_kernel(labels, mapping)

    # Class numbers are local to a bin.  Preserve physical ancestry continuity
    # by following each distinct chromosome-wide reconstructed founder path
    # into its next-bin pooled class.  Globally duplicate trajectories receive
    # one representative, so duplicating an arbitrary numeric label cannot
    # alter merge/split probabilities.
    unique_trajectory_states = _unique_trajectory_state_kernel(mapping)
    trajectory_classes = np.ascontiguousarray(
        mapping[:, unique_trajectory_states].T, dtype=np.int16
    )
    continuation_bridge = _continuation_bridge_kernel(
        mapping,
        class_counts,
        unique_trajectory_states,
        maximum_classes,
    )
    return (
        np.ascontiguousarray(pooled_labels),
        pooled_founders,
        active,
        np.ascontiguousarray(continuation_bridge),
        trajectory_classes,
    )


@pedigree_models.njit(cache=True, parallel=True)
def _parenthood_structure_count_kernel(labels, trios, required_edges=None):
    """Count strict, missing-aware IBS support in pooled-label bins.

    Production callers pass the sparse symmetric edge mask.  Omitting it
    retains the historical full-matrix private API used by diagnostics and
    focused reference tests.
    """
    n_samples, n_bins, _ = labels.shape
    edge_matched = np.zeros((n_samples, n_samples), dtype=np.float64)
    edge_exposed = np.zeros((n_samples, n_samples), dtype=np.float64)
    for first in prange(n_samples):
        for second in range(n_samples):
            if (
                required_edges is not None
                and not required_edges[first, second]
            ):
                continue
            for block in range(n_bins):
                a0 = labels[first, block, 0]
                a1 = labels[first, block, 1]
                b0 = labels[second, block, 0]
                b1 = labels[second, block, 1]
                if a0 < 0 or a1 < 0 or b0 < 0 or b1 < 0:
                    continue
                edge_exposed[first, second] += 1.0
                if a0 == b0 or a0 == b1 or a1 == b0 or a1 == b1:
                    edge_matched[first, second] += 1.0

    pair_explained = np.zeros(len(trios), dtype=np.float64)
    pair_exposed = np.zeros(len(trios), dtype=np.float64)
    for row in prange(len(trios)):
        child = int(trios[row, 0])
        parent1 = int(trios[row, 1])
        parent2 = int(trios[row, 2])
        for block in range(n_bins):
            c0 = labels[child, block, 0]
            c1 = labels[child, block, 1]
            p10 = labels[parent1, block, 0]
            p11 = labels[parent1, block, 1]
            p20 = labels[parent2, block, 0]
            p21 = labels[parent2, block, 1]
            if (
                c0 < 0 or c1 < 0 or p10 < 0 or p11 < 0
                or p20 < 0 or p21 < 0
            ):
                continue
            pair_exposed[row] += 1.0
            first_assignment = (
                (c0 == p10 or c0 == p11)
                and (c1 == p20 or c1 == p21)
            )
            second_assignment = (
                (c1 == p10 or c1 == p11)
                and (c0 == p20 or c0 == p21)
            )
            if first_assignment or second_assignment:
                pair_explained[row] += 1.0
    return edge_matched, edge_exposed, pair_explained, pair_exposed


@pedigree_models.njit(cache=True, parallel=True)
def _ancestry_junction_count_kernel(pooled_labels, trajectory_classes):
    """Minimum chromosome-wide founder switches for each diploid painting.

    The hidden state is an unordered pair drawn from deduplicated whole-
    chromosome founder trajectories (the two entries may be equal). Local IBS
    equivalence is an
    emission ambiguity, not permission to splice two different trajectories
    together for free. A transition costs zero for the same unordered pair,
    one when the pairs share one trajectory, and two otherwise. Per-founder
    and global minima reduce each dynamic-programming update from quartic to
    quadratic in the number of unique trajectories.

    Missing painted labels are uninformative. The second returned vector is
    the number of observed haplotype-bin labels and is carried into the depth
    model so incomplete paintings cannot masquerade as shallow ancestry merely
    because fewer switches were detectable.
    """
    n_samples, n_bins, _ = pooled_labels.shape
    n_trajectories = trajectory_classes.shape[0]
    n_pairs = n_trajectories * (n_trajectories + 1) // 2
    pair_first = np.empty(n_pairs, dtype=np.int64)
    pair_second = np.empty(n_pairs, dtype=np.int64)
    pair_index = 0
    for first in range(n_trajectories):
        for second in range(first, n_trajectories):
            pair_first[pair_index] = first
            pair_second[pair_index] = second
            pair_index += 1

    output = np.zeros(n_samples, dtype=np.int64)
    callable_bins = np.zeros(n_samples, dtype=np.int64)
    unreachable = 1 << 30
    for sample in prange(n_samples):
        previous_cost = np.zeros(n_pairs, dtype=np.int64)
        current_cost = np.empty(n_pairs, dtype=np.int64)
        per_trajectory_minimum = np.empty(n_trajectories, dtype=np.int64)
        observed_count = 0
        for block in range(n_bins):
            observed0 = int(pooled_labels[sample, block, 0])
            observed1 = int(pooled_labels[sample, block, 1])
            observed_count += int(observed0 >= 0) + int(observed1 >= 0)

            if block > 0:
                global_minimum = unreachable
                for trajectory in range(n_trajectories):
                    per_trajectory_minimum[trajectory] = unreachable
                for state in range(n_pairs):
                    value = int(previous_cost[state])
                    if value < global_minimum:
                        global_minimum = value
                    first = int(pair_first[state])
                    second = int(pair_second[state])
                    if value < per_trajectory_minimum[first]:
                        per_trajectory_minimum[first] = value
                    if value < per_trajectory_minimum[second]:
                        per_trajectory_minimum[second] = value

            for state in range(n_pairs):
                first = int(pair_first[state])
                second = int(pair_second[state])
                first_class = int(trajectory_classes[first, block])
                second_class = int(trajectory_classes[second, block])
                if observed0 < 0 and observed1 < 0:
                    compatible = True
                elif observed0 < 0:
                    compatible = (
                        first_class == observed1 or second_class == observed1
                    )
                elif observed1 < 0:
                    compatible = (
                        first_class == observed0 or second_class == observed0
                    )
                else:
                    compatible = (
                        first_class == observed0 and second_class == observed1
                    ) or (
                        first_class == observed1 and second_class == observed0
                    )
                if not compatible:
                    current_cost[state] = unreachable
                elif block == 0:
                    current_cost[state] = 0
                else:
                    shared = min(
                        int(per_trajectory_minimum[first]),
                        int(per_trajectory_minimum[second]),
                    ) + 1
                    unrelated = int(global_minimum) + 2
                    current_cost[state] = min(
                        int(previous_cost[state]), shared, unrelated
                    )
            swap = previous_cost
            previous_cost = current_cost
            current_cost = swap
        output[sample] = int(np.min(previous_cost))
        callable_bins[sample] = observed_count
    return output, callable_bins


@pedigree_models.njit(cache=True, parallel=True)
def _gl_information_exponent_kernel(
    genotype_likelihoods,
    selected_markers_per_bin,
    markers_per_information_block,
    effective_markers_per_information_block,
):
    """Temper raw-GL evidence by child, using only nonuniform GL vectors."""
    n_samples, n_bins, _, _ = genotype_likelihoods.shape
    information_block = np.empty(n_bins, dtype=np.int64)
    group = 0
    markers_in_group = 0
    for block in range(n_bins):
        if block > 0 and markers_in_group >= markers_per_information_block:
            group += 1
            markers_in_group = 0
        information_block[block] = group
        markers_in_group += max(int(selected_markers_per_bin[block]), 1)
    n_information_blocks = group + 1

    exponent = np.zeros((n_samples, n_bins), dtype=np.float64)
    for child in prange(n_samples):
        informative_per_group = np.zeros(
            n_information_blocks, dtype=np.int64
        )
        for block in range(n_bins):
            group_index = int(information_block[block])
            for snp in range(int(selected_markers_per_bin[block])):
                gl0 = genotype_likelihoods[child, block, snp, 0]
                gl1 = genotype_likelihoods[child, block, snp, 1]
                gl2 = genotype_likelihoods[child, block, snp, 2]
                if gl0 != gl1 or gl1 != gl2:
                    informative_per_group[group_index] += 1
        for block in range(n_bins):
            informative = informative_per_group[information_block[block]]
            if informative > 0:
                exponent[child, block] = (
                    min(
                        effective_markers_per_information_block,
                        float(informative),
                    )
                    / float(informative)
                )
    return exponent


def _robust_parent_screen(
    pair_scores: np.ndarray,
    marker_counts: np.ndarray,
    config: module_pedigree_config.PedigreeConfig,
    eligibility: pedigree_eligibility._ResolvedParentEligibility,
) -> np.ndarray:
    """Information-weighted utilities for eligible fixed-screen parents."""
    n_contigs, n_samples, _ = pair_scores.shape
    output = np.full((n_samples, n_samples), -np.inf, dtype=np.float64)
    contig_weights = pedigree_states._information_weights(marker_counts, config)
    blocks = np.maximum(
        np.ceil(marker_counts / config.markers_per_information_block), 1.0
    )
    tempering = blocks ** config.information_tempering_power
    for child in range(n_samples):
        if not eligibility.eligible_children[child]:
            continue
        parents = np.flatnonzero(eligibility.eligible_parents[child])
        if not len(parents):
            continue
        utility = np.empty((n_contigs, len(parents)), dtype=np.float64)
        for contig_index in range(n_contigs):
            values = pair_scores[contig_index, child, parents]
            if np.any(~np.isfinite(values)):
                raise pedigree_models.PedigreeEvidenceError("standard pair-HMM scores must be finite")
            centered = (values - np.max(values)) / tempering[contig_index]
            soft = np.exp(np.clip(centered, -60.0, 0.0))
            soft /= np.sum(soft)
            soft = (
                (1.0 - config.chromosome_contamination) * soft
                + config.chromosome_contamination / len(parents)
            )
            ranks = pedigree_states._tied_rank_probabilities(values)
            utility[contig_index] = (
                config.rank_weight * ranks
                + (1.0 - config.rank_weight) * soft
            )
        output[child, parents] = np.sum(
            utility * contig_weights[:, None], axis=0
        )
    return output


def _fixed_trio_panel(
    parent_scores: np.ndarray,
    top_k: int,
    anchor_k: int,
    use_anchor_union: bool,
    eligibility: pedigree_eligibility._ResolvedParentEligibility,
) -> np.ndarray:
    n_samples = parent_scores.shape[0]
    if n_samples < 3:
        raise pedigree_models.PedigreeEvidenceError("at least three samples are required")
    if int(top_k) != top_k or top_k < 1:
        raise pedigree_models.PedigreeEvidenceError("top_k must be a positive integer")
    if int(anchor_k) != anchor_k or anchor_k < 0:
        raise pedigree_models.PedigreeEvidenceError("anchor_k must be a non-negative integer")
    rows = []
    for child in range(n_samples):
        if not eligibility.eligible_children[child]:
            continue
        parents = np.flatnonzero(eligibility.eligible_parents[child])
        order = np.lexsort((parents, -parent_scores[child, parents]))
        leading = parents[order[:min(int(top_k), len(parents))]].tolist()
        pairs = {
            tuple(sorted((int(leading[first]), int(leading[second]))))
            for first in range(len(leading))
            for second in range(first + 1, len(leading))
            if pedigree_eligibility._eligible_parent_pair(
                eligibility, child, leading[first], leading[second]
            )
        }
        if use_anchor_union:
            for anchor in leading[:min(int(anchor_k), len(leading))]:
                for other in parents:
                    if (
                        int(other) != int(anchor)
                        and pedigree_eligibility._eligible_parent_pair(
                            eligibility, child, int(anchor), int(other)
                        )
                    ):
                        pairs.add(tuple(sorted((int(anchor), int(other)))))
        rows.extend((child, first, second) for first, second in sorted(pairs))
    return np.asarray(rows, dtype=np.int64).reshape((-1, 3))


import haplotype_reconstruction.pedigree.config as module_pedigree_config
import haplotype_reconstruction.pedigree.eligibility as pedigree_eligibility

import haplotype_reconstruction.pedigree.states as pedigree_states
