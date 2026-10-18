"""Minimum painted founder switches of each sample (orientation evidence input).

Founder trajectory numbers are not biological identities: in each bin, trajectory classes with the
same selected-marker alleles are pooled into one local class, so splitting one IBS haplotype into
duplicate numbers cannot manufacture switches. The hidden state of a sample is an unordered pair of
deduplicated whole-component trajectories (possibly equal); staying costs 0, keeping one
trajectory 1 and changing both 2, with per-trajectory and global minima making each update
quadratic (not quartic) in the number of trajectories. Unknown painted labels are uninformative;
the observed haplotype-bin labels are counted alongside, so incomplete paintings cannot pass as
shallow ancestry.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _local_classes(named_alleles, bin_start, bin_stop):
    """int16 [B, C]: first-occurrence local IBS class of each trajectory class in each bin."""
    n_classes = named_alleles.shape[0]
    n_bins = len(bin_start)
    mapping = np.empty((n_bins, n_classes), dtype=np.int16)
    for block in prange(n_bins):
        n_local = 0
        for state in range(n_classes):
            local = -1
            for previous in range(state):
                equivalent = True
                for site in range(bin_start[block], bin_stop[block]):
                    if named_alleles[state, site] != named_alleles[previous, site]:
                        equivalent = False
                        break
                if equivalent:
                    local = int(mapping[block, previous])
                    break
            if local < 0:
                local = n_local
                n_local += 1
            mapping[block, state] = local
    return mapping


@njit(cache=True, parallel=True)
def _pooled_labels(labels, mapping):
    n_samples, n_bins, _ = labels.shape
    pooled = np.full_like(labels, -1, dtype=np.int16)
    for flat in prange(n_samples * n_bins):
        sample = flat // n_bins
        block = flat - sample * n_bins
        for track in range(2):
            label = int(labels[sample, block, track])
            if label >= 0:
                pooled[sample, block, track] = mapping[block, label]
    return pooled


@njit(cache=True)
def _unique_trajectories(mapping):
    """Indices of the first occurrence of each distinct whole-component local-class trajectory."""
    n_bins, n_classes = mapping.shape
    unique = np.empty(n_classes, dtype=np.int64)
    count = 0
    for state in range(n_classes):
        duplicate = False
        for index in range(count):
            representative = int(unique[index])
            equivalent = True
            for block in range(n_bins):
                if mapping[block, state] != mapping[block, representative]:
                    equivalent = False
                    break
            if equivalent:
                duplicate = True
                break
        if not duplicate:
            unique[count] = state
            count += 1
    return unique[:count]


@njit(cache=True, parallel=True)
def _minimum_switches(pooled_labels, trajectory_classes):
    """(minimum switch count [N], observed haplotype-bin labels [N])."""
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
        global_minimum = unreachable
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
                    compatible = first_class == observed1 or second_class == observed1
                elif observed1 < 0:
                    compatible = first_class == observed0 or second_class == observed0
                else:
                    compatible = ((first_class == observed0 and second_class == observed1)
                                  or (first_class == observed1 and second_class == observed0))
                if not compatible:
                    current_cost[state] = unreachable
                elif block == 0:
                    current_cost[state] = 0
                else:
                    shared = min(int(per_trajectory_minimum[first]),
                                 int(per_trajectory_minimum[second])) + 1
                    unrelated = int(global_minimum) + 2
                    current_cost[state] = min(int(previous_cost[state]), shared, unrelated)
            swap = previous_cost
            previous_cost = current_cost
            current_cost = swap
        output[sample] = int(np.min(previous_cost))
        callable_bins[sample] = observed_count
    return output, callable_bins


def junction_counts(state_classes: np.ndarray, named_alleles: np.ndarray,
                    bin_sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(minimum switches int64 [N], observed haplotype-bin labels int64 [N]) of one component.

    ``state_classes`` int16 [N, B, 2] painted classes (-1 unknown) index the rows of
    ``named_alleles`` int8 [C, M] (selected markers, bin-major by ``bin_sizes``).
    """
    stops = np.cumsum(bin_sizes).astype(np.int64)
    starts = stops - bin_sizes
    mapping = _local_classes(np.ascontiguousarray(named_alleles), starts, stops)
    pooled = _pooled_labels(np.ascontiguousarray(state_classes, dtype=np.int16), mapping)
    trajectories = np.ascontiguousarray(mapping[:, _unique_trajectories(mapping)].T, dtype=np.int16)
    return _minimum_switches(pooled, trajectories)
