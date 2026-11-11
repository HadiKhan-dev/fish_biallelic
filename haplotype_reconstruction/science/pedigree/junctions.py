"""Minimum painted founder switches of each sample (orientation evidence input).

Founder trajectory numbers are not biological identities: in each bin, trajectory classes with the
same selected-marker alleles are pooled into one local class, so splitting one IBS haplotype into
duplicate numbers cannot manufacture switches. The hidden state of a sample is an unordered pair of
deduplicated whole-component trajectories (possibly equal); staying costs 0, keeping one
trajectory 1 and changing both 2, with per-trajectory and global minima making each update
quadratic (not quartic) in the number of trajectories. Unknown painted labels are uninformative;
the observed haplotype-bin labels are counted alongside, so incomplete paintings cannot pass as
shallow ancestry.

Method: docs/methods/pedigree.md#exposure-and-junction-counts
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import prange

from ...core.kernels import kernel
from ...core.ties import policy
from .kernels.signatures import OVERLOADS
from .parameters import JUNCTION_UNREACHABLE_COUNT, JUNCTION_UNREACHABLE_STATE


@kernel(overloads=OVERLOADS["junctions._local_classes"], cache=True, parallel=True, nogil=True)
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


@kernel(overloads=OVERLOADS["junctions._pooled_labels"], cache=True, parallel=True, nogil=True)
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


@kernel(overloads=OVERLOADS["junctions._unique_trajectories"], cache=True, nogil=True)
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


@kernel(overloads=OVERLOADS["junctions._minimum_switches"], cache=True, parallel=True, nogil=True)
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
    unreachable = JUNCTION_UNREACHABLE_STATE
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
                    compatible = (first_class == observed0 and second_class == observed1) or (
                        first_class == observed1 and second_class == observed0
                    )
                if not compatible:
                    current_cost[state] = unreachable
                elif block == 0:
                    current_cost[state] = 0
                else:
                    shared = min(int(per_trajectory_minimum[first]), int(per_trajectory_minimum[second])) + 1
                    unrelated = int(global_minimum) + 2
                    current_cost[state] = min(int(previous_cost[state]), shared, unrelated)
            swap = previous_cost
            previous_cost = current_cost
            current_cost = swap
        output[sample] = int(np.min(previous_cost))
        callable_bins[sample] = observed_count
    return output, callable_bins


def junction_counts(
    state_classes: np.ndarray, named_alleles: np.ndarray, bin_sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


# --------------------------------------------------------------------------- linked transmission
@kernel(overloads=OVERLOADS["junctions._trace_schedule"], cache=True, nogil=True)
def _trace_schedule(labels, trajectories, costs, reverse):
    """One sample's minimum-switch unordered trajectory pair per bin (int64 [B, 2]), by traceback through the
    same 0/1/2 costs as _minimum_switches; ties take the first pair in pair order, the last under the convention
    reversal (trajectories are in canonical content order, so the choice does not depend on painted track order or
    class numbering; it is an arbitrary placement among equally parsimonious paths)."""
    n_trajectories, n_bins = trajectories.shape
    n_pairs = n_trajectories * (n_trajectories + 1) // 2
    pairs = np.empty((n_pairs, 2), dtype=np.int64)
    index = 0
    for first in range(n_trajectories):
        for second in range(first, n_trajectories):
            pairs[index, 0] = first
            pairs[index, 1] = second
            index += 1
    unreachable = JUNCTION_UNREACHABLE_COUNT
    per = np.empty(n_trajectories, dtype=np.int64)
    for block in range(n_bins):
        global_minimum = unreachable
        if block > 0:
            for trajectory in range(n_trajectories):
                per[trajectory] = unreachable
            for state in range(n_pairs):
                value = costs[block - 1, state]
                per[pairs[state, 0]] = min(per[pairs[state, 0]], value)
                per[pairs[state, 1]] = min(per[pairs[state, 1]], value)
                global_minimum = min(global_minimum, value)
        observed0 = int(labels[block, 0])
        observed1 = int(labels[block, 1])
        for state in range(n_pairs):
            first_class = int(trajectories[pairs[state, 0], block])
            second_class = int(trajectories[pairs[state, 1], block])
            if observed0 < 0 and observed1 < 0:
                compatible = True
            elif observed0 < 0:
                compatible = first_class == observed1 or second_class == observed1
            elif observed1 < 0:
                compatible = first_class == observed0 or second_class == observed0
            else:
                compatible = (first_class == observed0 and second_class == observed1) or (
                    first_class == observed1 and second_class == observed0
                )
            if not compatible:
                costs[block, state] = unreachable
            elif block == 0:
                costs[block, state] = 0
            else:
                costs[block, state] = min(
                    costs[block - 1, state], min(per[pairs[state, 0]], per[pairs[state, 1]]) + 1, global_minimum + 2
                )
    trace = np.empty((n_bins, 2), dtype=np.int64)
    current = 0
    for state in range(n_pairs):
        value = costs[n_bins - 1, state]
        if value < costs[n_bins - 1, current] or (reverse and value == costs[n_bins - 1, current]):
            current = state
    trace[n_bins - 1, 0] = pairs[current, 0]
    trace[n_bins - 1, 1] = pairs[current, 1]
    for block in range(n_bins - 2, -1, -1):
        a = pairs[current, 0]
        b = pairs[current, 1]
        best = unreachable
        chosen = 0
        for state in range(n_pairs):
            c = pairs[state, 0]
            d = pairs[state, 1]
            change = min(int(a != c) + int(b != d), int(a != d) + int(b != c))
            value = costs[block, state] + change
            if value < best or (reverse and value == best):
                best = value
                chosen = state
        current = chosen
        trace[block, 0] = pairs[current, 0]
        trace[block, 1] = pairs[current, 1]
    return trace


@kernel(overloads=OVERLOADS["junctions._trace_schedules"], cache=True, parallel=True, nogil=True)
def _trace_schedules(pooled, trajectories, reverse):
    n_samples, n_bins, _ = pooled.shape
    n_trajectories = trajectories.shape[0]
    n_pairs = n_trajectories * (n_trajectories + 1) // 2
    output = np.zeros((n_samples, n_bins, 2), dtype=np.int64)
    for sample in prange(n_samples):
        costs = np.empty((n_bins, n_pairs), dtype=np.int64)
        output[sample] = _trace_schedule(pooled[sample], trajectories, costs, reverse)
    return output


@dataclass(frozen=True)
class ComponentSchedule:
    """Transient component direction inputs and the count summaries used for graph ordering.

    pairs int64 [N, B, 2]: minimum-switch trajectory indices; classes int64 [N, B, 2]:
    corresponding local classes; known bool [N, B]: both painted labels present;
    n_trajectories: distinct whole-component trajectories T. counts and callable_bins
    int64 [N] retain the minimum switch count and observed haplotype-bin count.
    """

    pairs: np.ndarray
    classes: np.ndarray
    known: np.ndarray
    n_trajectories: int
    counts: np.ndarray
    callable_bins: np.ndarray


def schedules(
    state_classes: np.ndarray, named_alleles: np.ndarray, bin_sizes: np.ndarray, *, reverse: bool
) -> ComponentSchedule:
    """Build a component's schedule, local classes, known-label mask, T and count summaries once.

    Trajectories are numbered in canonical allele-content order, so painted track order
    and class numbering do not matter. Counts sum the minimum boundary alignment costs
    on the traced path, exactly the minimum in ``junction_counts``. Pooling and deduplication
    share one marker pass with all transmission inputs; no second count pass is needed.
    """
    labels = np.asarray(state_classes, dtype=np.int16)
    named = np.asarray(named_alleles, dtype=np.int8)
    sizes = np.asarray(bin_sizes, dtype=np.int64)
    n_samples, n_bins, _ = labels.shape
    if n_bins == 0 or len(named) == 0:
        empty = np.zeros((n_samples, n_bins, 2), dtype=np.int64)
        return ComponentSchedule(
            empty,
            empty.copy(),
            np.zeros((n_samples, n_bins), dtype=np.bool_),
            1,
            np.zeros(n_samples, dtype=np.int64),
            np.zeros(n_samples, dtype=np.int64),
        )
    order = np.asarray(sorted(range(len(named)), key=lambda row: named[row].tobytes()), dtype=np.int64)
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
    relabelled = labels.copy()
    known = labels >= 0
    relabelled[known] = inverse[labels[known]]
    stops = np.cumsum(sizes).astype(np.int64)
    starts = stops - sizes
    mapping = _local_classes(np.ascontiguousarray(named[order]), starts, stops)
    pooled = _pooled_labels(np.ascontiguousarray(relabelled), mapping)
    trajectories = np.ascontiguousarray(mapping[:, _unique_trajectories(mapping)].T, dtype=np.int16)
    pairs = _trace_schedules(
        pooled, trajectories, policy("pedigree.junction.trace", kind="probability", reverse=reverse).reverse
    )
    bins = np.arange(n_bins)
    classes = np.stack([trajectories[pairs[:, :, h], bins[None, :]] for h in range(2)], axis=2).astype(np.int64)
    present = pooled >= 0
    straight = np.count_nonzero(pairs[:, :-1] != pairs[:, 1:], axis=2)
    crossed = np.count_nonzero(pairs[:, :-1] != pairs[:, 1:, ::-1], axis=2)
    counts = np.sum(np.minimum(straight, crossed), axis=1, dtype=np.int64)
    return ComponentSchedule(
        pairs,
        classes,
        np.all(present, axis=2),
        int(len(trajectories)),
        counts,
        np.count_nonzero(present, axis=(1, 2)).astype(np.int64),
    )
