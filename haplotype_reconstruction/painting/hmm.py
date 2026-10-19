"""Numerical kernels of the painting HMM: emissions, Viterbi, forward-backward.

State of a sample in a bin: an ordered pair (x1, x2) of track states, flat
index x1 * S + x2 (S = classes + BACKGROUND). Emissions are binned sums of
robust per-site genotype log likelihoods, centred by log(1/3) so unobserved
and flat sites are neutral. Transitions are the Hamming table of
painting/transitions.py, so every recursion contracts in O(S^2) per bin from
row, column and complement sums (forward-backward) or from the best and
second-best entries of each row and column (Viterbi). Samples are independent;
every per-sample reduction runs in a fixed order, so results do not depend on
the thread count.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .transitions import PaintingTransitions

# Spread of a normalised GL row above which the row is informative (not flat).
INFORMATIVE_SPREAD = 16.0 * np.finfo(np.float64).eps
# Viterbi tie quantum (nats; release 2.1, design A-4): predecessors whose path scores are within this of the best are
# tied, and the back-pointer then takes the fewest track changes (stay, then one change, then two), then the
# smallest flat index; the final state is the smallest index within it of the best. Equal-score paths arise where
# two switch placements cost the same (equal theta on a cold, zero-cM stretch of the map, and the states' emissions
# identical in the bins between) and their sums differ only in rounding, which GL noise reorders. The path scores
# themselves stay exact maxima. Census (TroMau chr18 painting, iid 1e-9; work/IE/census/viterbi_chr18_*): 4,980
# decisions within 1e-2, gap motion <= 1.9e-9, genuine gaps from 1e-5 (2 in [1e-5, 3e-5), then >= 3e-4): 2^-20 =
# 9.5e-7 sits ~500x above the motion and ~10x below the first genuine gaps.
VITERBI_TIE = 2.0 ** -20
# Candidate states of a track are int64 bit masks (painting switch intervals, release 2.2): S = classes + BACKGROUND
# must fit 63 bits (K <= 30 founders gives S <= 31).
MAXIMUM_MASK_STATES = 63


@njit(cache=True, inline="always")
def _before(value, index, reference_value, reference_index, reverse):
    """True if (value, index) ranks before (reference_value, reference_index): the larger value, and at an exact
    tie the smaller flat index (the canonical rule), or the larger one under the convention reversal (release 2.2,
    maths.ties.reverse_conventions)."""
    if value > reference_value:
        return True
    if value == reference_value:
        return index > reference_index if reverse else index < reference_index
    return False


@njit(cache=True, parallel=True)
def informative_sites(likelihoods, observed):
    """bool[N, L]: observed and max - min of the normalised GL row > 16 eps."""
    n_samples, n_sites, _ = likelihoods.shape
    result = np.empty((n_samples, n_sites), dtype=np.bool_)
    for row in prange(n_samples * n_sites):
        sample, site = row // n_sites, row % n_sites
        a = likelihoods[sample, site, 0]
        b = likelihoods[sample, site, 1]
        c = likelihoods[sample, site, 2]
        result[sample, site] = (observed[sample, site]
                                and max(a, b, c) - min(a, b, c) > INFORMATIVE_SPREAD)
    return result


@njit(cache=True, parallel=True)
def _gather_kernel(likelihoods, observed, sites):
    n_samples = likelihoods.shape[0]
    values = np.empty((n_samples, len(sites), 3), dtype=np.float64)
    flags = np.empty((n_samples, len(sites)), dtype=np.bool_)
    for sample in prange(n_samples):
        for site in range(len(sites)):
            source = sites[site]
            values[sample, site, 0] = likelihoods[sample, source, 0]
            values[sample, site, 1] = likelihoods[sample, source, 1]
            values[sample, site, 2] = likelihoods[sample, source, 2]
            flags[sample, site] = observed[sample, source]
    return values, flags


def gather_sites(likelihoods: np.ndarray, observed: np.ndarray, sites: np.ndarray,
                 samples: slice = slice(None)) -> tuple[np.ndarray, np.ndarray]:
    """C-contiguous float64[n, len(sites), 3] GL rows and bool[n, len(sites)] flags of ``samples``.

    ``sites`` are strictly increasing evidence site indices: a consecutive run is
    copied as one slice, other selections with a parallel gather.
    """
    sites = np.asarray(sites, dtype=np.int64)
    if len(sites) == 0 or int(sites[-1]) - int(sites[0]) + 1 == len(sites):
        window = slice(int(sites[0]), int(sites[-1]) + 1) if len(sites) else slice(0, 0)
        return (np.ascontiguousarray(likelihoods[samples, window], dtype=np.float64),
                np.ascontiguousarray(observed[samples, window], dtype=np.bool_))
    return _gather_kernel(likelihoods[samples], observed[samples], sites)


@njit(cache=True, parallel=True)
def _emission_kernel(likelihoods, observed, alt_probability, background_frequency,
                     bin_starts, bin_stops, robustness_epsilon, log_floor):
    """float64[N, S*S, B] binned log emissions, one task per sample x unordered pair.

    The ordered states (a, b) and (b, a) have equal emissions (the genotype law
    is symmetric in the two homologues), so each unordered pair is evaluated
    once and stored at both.
    """
    n_samples = likelihoods.shape[0]
    n_classes = alt_probability.shape[0]
    n_states = n_classes + 1
    n_pairs = n_states * (n_states + 1) // 2
    n_bins = len(bin_starts)
    result = np.empty((n_samples, n_states * n_states, n_bins), dtype=np.float64)
    uniform_log = max(math.log(1.0 / 3.0), log_floor)
    for task in prange(n_samples * n_pairs):
        sample = task // n_pairs
        # Unordered pairs first <= second in row-major upper-triangle order.
        pair = task % n_pairs
        first = 0
        while pair >= n_states - first:
            pair -= n_states - first
            first += 1
        second = first + pair
        state = first * n_states + second
        mirror = second * n_states + first
        for bin_index in range(n_bins):
            score = 0.0
            for site in range(bin_starts[bin_index], bin_stops[bin_index]):
                if not observed[sample, site]:
                    continue
                first_q = (background_frequency[site] if first == n_classes
                           else alt_probability[first, site])
                second_q = (background_frequency[site] if second == n_classes
                            else alt_probability[second, site])
                if first == second and first < n_classes:
                    # Two homologues of one class share one latent allele.
                    p0 = 1.0 - first_q
                    p1 = 0.0
                    p2 = first_q
                else:
                    p0 = (1.0 - first_q) * (1.0 - second_q)
                    p2 = first_q * second_q
                    p1 = 1.0 - p0 - p2
                likelihood = (likelihoods[sample, site, 0] * p0
                              + likelihoods[sample, site, 1] * p1
                              + likelihoods[sample, site, 2] * p2)
                probability = (1.0 - robustness_epsilon) * likelihood + robustness_epsilon / 3.0
                if probability <= 0.0:
                    log_probability = log_floor
                else:
                    log_probability = math.log(probability)
                    if log_probability < log_floor:
                        log_probability = log_floor
                score += log_probability - uniform_log
            result[sample, state, bin_index] = score
            result[sample, mirror, bin_index] = score
    return result


def binned_emissions(likelihoods: np.ndarray, observed: np.ndarray, alt_probability: np.ndarray,
                     background_frequency: np.ndarray, bin_starts: np.ndarray,
                     bin_stops: np.ndarray, *, robustness_epsilon: float, log_floor: float,
                     sample_block: int = 0, before_block=None) -> np.ndarray:
    """float64[N, S*S, B] log emissions of contiguous site runs [bin_starts, bin_stops).

    ``likelihoods`` float64[N, L, 3] normalised GL rows and ``observed``
    bool[N, L] at the component's selected sites; ``alt_probability``
    float64[C, L] and ``background_frequency`` float64[L] from the states.
    O(N S^2 L / 2). ``sample_block`` > 0 evaluates that many samples at a time
    (values are identical) and calls ``before_block()`` before each block,
    e.g. to let a chromosome worker's CPU lease grow.
    """
    arguments = (np.ascontiguousarray(alt_probability, dtype=np.float64),
                 np.ascontiguousarray(background_frequency, dtype=np.float64),
                 np.ascontiguousarray(bin_starts, dtype=np.int64),
                 np.ascontiguousarray(bin_stops, dtype=np.int64),
                 float(robustness_epsilon), float(log_floor))
    n_samples = likelihoods.shape[0]
    if sample_block <= 0 or sample_block >= n_samples:
        return _emission_kernel(likelihoods, observed, *arguments)
    n_states = alt_probability.shape[0] + 1
    result = np.empty((n_samples, n_states * n_states, len(bin_starts)), dtype=np.float64)
    for start in range(0, n_samples, sample_block):
        if before_block is not None:
            before_block()
        stop = min(start + sample_block, n_samples)
        result[start:stop] = _emission_kernel(likelihoods[start:stop], observed[start:stop],
                                              *arguments)
    return result


@njit(cache=True, parallel=True)
def pack_emissions(emissions, packed, sample_start, n_states):
    """Store the upper triangle (first <= second) of symmetric ordered-pair emissions
    into ``packed`` [N, S(S+1)/2, B] at rows sample_start + i."""
    for task in prange(emissions.shape[0] * n_states):
        sample, first = task // n_states, task % n_states
        row_start = first * n_states - first * (first - 1) // 2
        for second in range(first, n_states):
            for block in range(emissions.shape[2]):
                packed[sample_start + sample, row_start + second - first, block] = (
                    emissions[sample, first * n_states + second, block])


@njit(cache=True, parallel=True)
def unpack_emissions(packed, n_states):
    """float64[N, S*S, B] ordered-pair emissions from their packed upper triangle."""
    output = np.empty((packed.shape[0], n_states * n_states, packed.shape[2]))
    for task in prange(packed.shape[0] * n_states):
        sample, first = task // n_states, task % n_states
        row_start = first * n_states - first * (first - 1) // 2
        for second in range(first, n_states):
            for block in range(packed.shape[2]):
                value = packed[sample, row_start + second - first, block]
                output[sample, first * n_states + second, block] = value
                output[sample, second * n_states + first, block] = value
    return output


@njit(cache=True, inline="always")
def hamming_max_step(previous, cost_same, cost_one, cost_two, n_states, out_values, out_index,
                     row_best_value, row_second_value, row_best_index, row_second_index,
                     column_best_value, column_second_value, column_best_index,
                     column_second_index, tie, reverse):
    """out_values[z] = max_y previous[y] + T(y, z) and out_index[z] its argmax, O(S^2).

    T charges cost_same, cost_one or cost_two for zero, one or two changed
    tracks. The best y excluding row/column/both of z comes from the best and
    second-best entries of each row and column. out_values are exact maxima; the
    back-pointer takes, among the candidates within ``tie`` of the maximum, the one
    with the fewest changed tracks (stay, one, two), and between the best one-change
    candidates of the row and of the column the smaller flat index when they are
    within ``tie`` (release 2.1; exact ties inside a row or column go to the
    smallest flat index, the canonical tie rule). ``reverse`` (release 2.2 convention
    reversal) takes the largest flat index instead at every such index tie; the
    fewest-changes preference (a stated prior) is unchanged.
    """
    for row in range(n_states):
        best_value = -np.inf
        second_value = -np.inf
        best_index = -1
        second_index = -1
        for column in range(n_states):
            index = row * n_states + column
            value = previous[index]
            if _before(value, index, best_value, best_index, reverse):
                second_value = best_value
                second_index = best_index
                best_value = value
                best_index = index
            elif _before(value, index, second_value, second_index, reverse):
                second_value = value
                second_index = index
        row_best_value[row] = best_value
        row_second_value[row] = second_value
        row_best_index[row] = best_index
        row_second_index[row] = second_index

    for column in range(n_states):
        best_value = -np.inf
        second_value = -np.inf
        best_index = -1
        second_index = -1
        for row in range(n_states):
            index = row * n_states + column
            value = previous[index]
            if _before(value, index, best_value, best_index, reverse):
                second_value = best_value
                second_index = best_index
                best_value = value
                best_index = index
            elif _before(value, index, second_value, second_index, reverse):
                second_value = value
                second_index = index
        column_best_value[column] = best_value
        column_second_value[column] = second_value
        column_best_index[column] = best_index
        column_second_index[column] = second_index

    for current_row in range(n_states):
        # Best previous state outside the current row and outside one column.
        complement_best_value = -np.inf
        complement_second_value = -np.inf
        complement_best_index = -1
        complement_second_index = -1
        complement_best_column = -1
        for previous_column in range(n_states):
            index = column_best_index[previous_column]
            value = column_best_value[previous_column]
            if index // n_states == current_row:
                index = column_second_index[previous_column]
                value = column_second_value[previous_column]
            if _before(value, index, complement_best_value, complement_best_index, reverse):
                complement_second_value = complement_best_value
                complement_second_index = complement_best_index
                complement_best_value = value
                complement_best_index = index
                complement_best_column = previous_column
            elif _before(value, index, complement_second_value, complement_second_index, reverse):
                complement_second_value = value
                complement_second_index = index

        for current_column in range(n_states):
            current = current_row * n_states + current_column
            stay_value = previous[current] + cost_same

            column_index = column_best_index[current_column]
            column_value = column_best_value[current_column]
            if column_index // n_states == current_row:
                column_index = column_second_index[current_column]
                column_value = column_second_value[current_column]
            row_index = row_best_index[current_row]
            row_value = row_best_value[current_row]
            if row_index % n_states == current_column:
                row_index = row_second_index[current_row]
                row_value = row_second_value[current_row]
            one_value = max(row_value, column_value)
            if row_value > column_value + tie:
                one_index = row_index
            elif column_value > row_value + tie:
                one_index = column_index
            elif row_index < 0 or (column_index >= 0 and (column_index > row_index if reverse
                                                          else column_index < row_index)):
                one_index = column_index
            else:
                one_index = row_index
            one_candidate = one_value + cost_one

            complement_value = complement_best_value
            complement_index = complement_best_index
            if complement_best_column == current_column:
                complement_value = complement_second_value
                complement_index = complement_second_index
            two_candidate = complement_value + cost_two

            best_value = max(stay_value, max(one_candidate, two_candidate))
            if stay_value >= best_value - tie:
                best_index = current
            elif one_candidate >= best_value - tie:
                best_index = one_index
            else:
                best_index = complement_index

            out_values[current] = best_value
            out_index[current] = best_index


@njit(cache=True, inline="always")
def hamming_max_values(values, cost_same, cost_one, cost_two, n_states, out, row_first, row_second,
                       column_first, column_second, row_first_column, column_first_row):
    """out[z] = max_y values[y] + T(y, z): :func:`hamming_max_step` without back-pointers.

    The maxima are the same; without argmax tie bookkeeping each excluded
    maximum needs only the best and second-best values of a row or column and
    where the best one lies. Used where only scores matter (label continuity).
    """
    for index in range(n_states):
        row_first[index] = -np.inf
        row_second[index] = -np.inf
        column_first[index] = -np.inf
        column_second[index] = -np.inf
        row_first_column[index] = -1
        column_first_row[index] = -1
    for row in range(n_states):
        for column in range(n_states):
            value = values[row * n_states + column]
            if value > row_first[row]:
                row_second[row] = row_first[row]
                row_first[row] = value
                row_first_column[row] = column
            elif value > row_second[row]:
                row_second[row] = value
            if value > column_first[column]:
                column_second[column] = column_first[column]
                column_first[column] = value
                column_first_row[column] = row
            elif value > column_second[column]:
                column_second[column] = value
    for row in range(n_states):
        # Best entry outside this row, per column, then outside one column.
        complement_first = -np.inf
        complement_second = -np.inf
        complement_first_column = -1
        for column in range(n_states):
            value = (column_first[column] if column_first_row[column] != row
                     else column_second[column])
            if value > complement_first:
                complement_second = complement_first
                complement_first = value
                complement_first_column = column
            elif value > complement_second:
                complement_second = value
        for column in range(n_states):
            best = values[row * n_states + column] + cost_same
            same_row = (row_first[row] if row_first_column[row] != column else row_second[row])
            same_column = (column_first[column] if column_first_row[column] != row
                           else column_second[column])
            candidate = max(same_row, same_column) + cost_one
            if candidate > best:
                best = candidate
            candidate = (complement_first if complement_first_column != column
                         else complement_second) + cost_two
            if candidate > best:
                best = candidate
            out[row * n_states + column] = best


@njit(cache=True, parallel=True)
def _viterbi_kernel(emissions, n_states, log_same, log_one, log_two, tie, reverse):
    n_samples, n_diplotypes, n_bins = emissions.shape
    back = np.full((n_samples, n_bins, n_diplotypes), -1, dtype=np.int32)
    final_scores = np.empty((n_samples, n_diplotypes), dtype=np.float64)
    for sample in prange(n_samples):
        previous = emissions[sample, :, 0].copy()
        previous -= np.max(previous)
        current_scores = np.empty(n_diplotypes, dtype=np.float64)
        row_best_value = np.empty(n_states, dtype=np.float64)
        row_second_value = np.empty(n_states, dtype=np.float64)
        row_best_index = np.empty(n_states, dtype=np.int64)
        row_second_index = np.empty(n_states, dtype=np.int64)
        column_best_value = np.empty(n_states, dtype=np.float64)
        column_second_value = np.empty(n_states, dtype=np.float64)
        column_best_index = np.empty(n_states, dtype=np.int64)
        column_second_index = np.empty(n_states, dtype=np.int64)
        for bin_index in range(1, n_bins):
            hamming_max_step(previous, log_same[bin_index - 1], log_one[bin_index - 1],
                             log_two[bin_index - 1], n_states, current_scores,
                             back[sample, bin_index],
                             row_best_value, row_second_value, row_best_index,
                             row_second_index, column_best_value, column_second_value,
                             column_best_index, column_second_index, tie, reverse)
            for current in range(n_diplotypes):
                current_scores[current] += emissions[sample, current, bin_index]
            # Per-bin centring prevents long-path overflow; paths and ties are unchanged.
            current_scores -= np.max(current_scores)
            previous, current_scores = current_scores, previous
        final_scores[sample] = previous

    result = np.empty((n_samples, 2, n_bins), dtype=np.int32)
    for sample in prange(n_samples):
        # The smallest final state within the tie quantum of the best (release 2.1; the largest under the
        # convention reversal).
        top = np.max(final_scores[sample])
        current = 0
        for rank in range(n_diplotypes):
            state = n_diplotypes - 1 - rank if reverse else rank
            if final_scores[sample, state] >= top - tie:
                current = state
                break
        result[sample, 0, n_bins - 1] = current // n_states
        result[sample, 1, n_bins - 1] = current % n_states
        for bin_index in range(n_bins - 1, 0, -1):
            current = back[sample, bin_index, current]
            result[sample, 0, bin_index - 1] = current // n_states
            result[sample, 1, bin_index - 1] = current % n_states
    return result


def viterbi_paths(emissions: np.ndarray, transitions: PaintingTransitions, *,
                  tie: float = VITERBI_TIE, reverse: bool | None = None) -> np.ndarray:
    """int32[N, 2, B] state of each track on the Viterbi path (needs B >= 1).

    Exact O(N B S^2) maxima; predecessors within ``tie`` nats of the best are tied
    and resolved by the fewest track changes, then the smallest flat state
    (VITERBI_TIE; tie = 0 gives 2.0's exact rule up to the stay preference).
    Holds int32 back-pointers [N, B, S^2], the largest painting array.

    The traceback runs right to left and stays while staying is tied, so among
    co-optimal placements a switch lands on the leftmost tied edge; that is a
    convention, not a released result. ``reverse``
    (default: maths.ties.reverse_conventions(), release 2.2) mirrors it: the
    recursion runs over the bins in reverse order (a switch lands on the
    rightmost tied edge) and index ties take the largest flat state.
    """
    from ..maths import ties
    if reverse is None:
        reverse = ties.reverse_conventions()
    if not reverse:
        return _viterbi_kernel(emissions, transitions.n_states, transitions.log_same,
                               transitions.log_one, transitions.log_two, float(tie), False)
    mirrored = _viterbi_kernel(np.ascontiguousarray(emissions[:, :, ::-1]), transitions.n_states,
                               np.ascontiguousarray(transitions.log_same[::-1]),
                               np.ascontiguousarray(transitions.log_one[::-1]),
                               np.ascontiguousarray(transitions.log_two[::-1]), float(tie), True)
    return np.ascontiguousarray(mirrored[:, :, ::-1])


@njit(cache=True)
def _allele_visible_change(previous, current, bin_index, local_keys, n_classes):
    """1 if a track change previous -> current changes the track's alleles."""
    if previous == current:
        return 0
    if previous >= n_classes or current >= n_classes:
        return 1
    for shared_bin in (bin_index - 1, bin_index):
        key = local_keys[shared_bin, previous]
        if key < n_classes and key == local_keys[shared_bin, current]:
            return 0
    return 1


@njit(cache=True, parallel=True)
def parsimonious_track_order(paths, local_keys, n_classes):
    """Among equal-score Viterbi paths, the track order with fewer allele changes.

    Emissions are symmetric in track order and the Hamming transition is
    invariant to swapping both tracks, so swapping the whole path suffix after
    boundary t-1|t changes the score only through that transition: when the
    ordered and swapped continuations change equally many tracks, both are
    exact Viterbi paths. At such ties the suffix is swapped iff the swapped
    order has strictly fewer allele-visible track changes (a change between
    rows sharing a local key < n_classes in bin t-1 or t is invisible).
    Unordered states, and hence every posterior summary, are unchanged. O(N B).
    """
    n_samples, _, n_bins = paths.shape
    result = np.empty_like(paths)
    for sample in prange(n_samples):
        if n_bins == 0:
            continue
        swapped = False
        result[sample, 0, 0] = paths[sample, 0, 0]
        result[sample, 1, 0] = paths[sample, 1, 0]
        for bin_index in range(1, n_bins):
            first_before = result[sample, 0, bin_index - 1]
            second_before = result[sample, 1, bin_index - 1]
            first = paths[sample, 0, bin_index]
            second = paths[sample, 1, bin_index]
            if swapped:
                first, second = second, first
            direct_changes = int(first_before != first) + int(second_before != second)
            crossed_changes = int(first_before != second) + int(second_before != first)
            if direct_changes == crossed_changes:
                direct_visible = (
                    _allele_visible_change(first_before, first, bin_index, local_keys, n_classes)
                    + _allele_visible_change(second_before, second, bin_index, local_keys,
                                             n_classes))
                crossed_visible = (
                    _allele_visible_change(first_before, second, bin_index, local_keys, n_classes)
                    + _allele_visible_change(second_before, first, bin_index, local_keys,
                                             n_classes))
                if crossed_visible < direct_visible:
                    swapped = not swapped
                    first, second = second, first
            result[sample, 0, bin_index] = first
            result[sample, 1, bin_index] = second
    return result


@njit(cache=True, parallel=True)
def _release_posterior_kernel(emissions, n_states, weight_same, weight_one, weight_two,
                              masks, other_masks, public_ids, local_keys):
    n_samples, _, n_bins = emissions.shape
    qv = np.zeros((n_samples, n_bins), dtype=np.float64)
    qa = np.zeros((n_samples, n_bins), dtype=np.float64)
    for sample in prange(n_samples):
        # Scaled forward pass: one sample's B x S x S table is live per thread.
        forward = np.zeros((n_bins, n_states, n_states), dtype=np.float64)
        scales = np.empty(n_bins, dtype=np.float64)
        emission_max = np.max(emissions[sample, :, 0])
        scale = 0.0
        for row in range(n_states):
            for column in range(n_states):
                value = math.exp(emissions[sample, row * n_states + column, 0] - emission_max)
                forward[0, row, column] = value
                scale += value
        scales[0] = scale
        forward[0] /= scale

        row_sums = np.zeros(n_states, dtype=np.float64)
        column_sums = np.zeros(n_states, dtype=np.float64)
        for bin_index in range(1, n_bins):
            same = weight_same[bin_index - 1]
            one = weight_one[bin_index - 1]
            two = weight_two[bin_index - 1]
            row_sums[:] = 0.0
            column_sums[:] = 0.0
            total = 0.0
            for row in range(n_states):
                for column in range(n_states):
                    value = forward[bin_index - 1, row, column]
                    row_sums[row] += value
                    column_sums[column] += value
                    total += value
            emission_max = np.max(emissions[sample, :, bin_index])
            scale = 0.0
            for row in range(n_states):
                for column in range(n_states):
                    cell = forward[bin_index - 1, row, column]
                    one_change = max(0.0, row_sums[row] + column_sums[column] - 2.0 * cell)
                    two_changes = max(0.0, total - row_sums[row] - column_sums[column] + cell)
                    prediction = same * cell + one * one_change + two * two_changes
                    weight = math.exp(
                        emissions[sample, row * n_states + column, bin_index] - emission_max)
                    value = prediction * weight
                    forward[bin_index, row, column] = value
                    scale += value
            scales[bin_index] = scale
            forward[bin_index] /= scale

        backward = np.ones((n_states, n_states), dtype=np.float64)
        weighted_next = np.empty((n_states, n_states), dtype=np.float64)
        previous_backward = np.empty((n_states, n_states), dtype=np.float64)
        for bin_index in range(n_bins - 1, -1, -1):
            posterior_total = 0.0
            for row in range(n_states):
                for column in range(n_states):
                    posterior_total += forward[bin_index, row, column] * backward[row, column]
            # qV: mass of the unordered public pairs consistent with the two tracks'
            # candidate sets (the Viterbi state, plus the other side of a switch
            # interval; release 2.2); qA: the same over local allele keys (coarser,
            # so qA >= qV). With one candidate per track this is the mass of the
            # Viterbi state's unordered public (local-key) pair, summed over the
            # same terms in the same order as release 2.1. Masses are added in
            # state order, as one pass over the full pair table would.
            # The released masses are the smaller of the two decodings' (``other_masks``: the mirrored traceback
            # convention's Viterbi pair; release 2.2), so a bin passes only if both conventions' pairs do.
            for decoding in range(2):
                source = masks if decoding == 0 else other_masks
                mask_first = source[sample, 0, bin_index]
                mask_second = source[sample, 1, bin_index]
                public_first_mask = np.int64(0)
                public_second_mask = np.int64(0)
                local_first_mask = np.int64(0)
                local_second_mask = np.int64(0)
                for state in range(n_states):
                    if (mask_first >> state) & 1:
                        public_first_mask |= np.int64(1) << public_ids[state]
                        local_first_mask |= np.int64(1) << local_keys[bin_index, state]
                    if (mask_second >> state) & 1:
                        public_second_mask |= np.int64(1) << public_ids[state]
                        local_second_mask |= np.int64(1) << local_keys[bin_index, state]
                public_mass = 0.0
                local_mass = 0.0
                for first in range(n_states):
                    first_public = public_ids[first]
                    first_local = local_keys[bin_index, first]
                    for second in range(n_states):
                        mass = forward[bin_index, first, second] * backward[first, second] / posterior_total
                        second_public = public_ids[second]
                        if ((((public_first_mask >> first_public) & 1)
                             and ((public_second_mask >> second_public) & 1))
                                or (((public_second_mask >> first_public) & 1)
                                    and ((public_first_mask >> second_public) & 1))):
                            public_mass += mass
                        second_local = local_keys[bin_index, second]
                        if ((((local_first_mask >> first_local) & 1) and ((local_second_mask >> second_local) & 1))
                                or (((local_second_mask >> first_local) & 1)
                                    and ((local_first_mask >> second_local) & 1))):
                            local_mass += mass
                public_value = min(1.0, max(0.0, public_mass))
                local_value = min(1.0, max(0.0, local_mass))
                if decoding == 0:
                    qv[sample, bin_index] = public_value
                    qa[sample, bin_index] = local_value
                else:
                    qv[sample, bin_index] = min(qv[sample, bin_index], public_value)
                    qa[sample, bin_index] = min(qa[sample, bin_index], local_value)
            if bin_index == 0:
                continue

            same = weight_same[bin_index - 1]
            one = weight_one[bin_index - 1]
            two = weight_two[bin_index - 1]
            emission_max = np.max(emissions[sample, :, bin_index])
            row_sums[:] = 0.0
            column_sums[:] = 0.0
            total = 0.0
            for row in range(n_states):
                for column in range(n_states):
                    value = (math.exp(emissions[sample, row * n_states + column, bin_index]
                                      - emission_max) * backward[row, column])
                    weighted_next[row, column] = value
                    row_sums[row] += value
                    column_sums[column] += value
                    total += value
            for row in range(n_states):
                for column in range(n_states):
                    cell = weighted_next[row, column]
                    one_change = max(0.0, row_sums[row] + column_sums[column] - 2.0 * cell)
                    two_changes = max(0.0, total - row_sums[row] - column_sums[column] + cell)
                    previous_backward[row, column] = (
                        same * cell + one * one_change + two * two_changes) / scales[bin_index]
            backward, previous_backward = previous_backward, backward
    return qv, qa


def release_posteriors(emissions: np.ndarray, transitions: PaintingTransitions, paths: np.ndarray,
                       public_ids: np.ndarray, local_keys: np.ndarray, masks: np.ndarray | None = None,
                       other_paths: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(qV, qA) float64[N, B]: posterior masses of the unordered public class pairs and of the
    unordered local allele-key pairs consistent with each track's candidate states (exact scaled
    forward-backward).

    ``masks`` int64[N, 2, B]: per track and bin the candidate states as bits (state x = bit x);
    None = the Viterbi state alone, i.e. the mass of the Viterbi state's unordered public (local-key)
    pair (release 2.1). ``other_paths`` (the mirrored traceback convention's Viterbi path, release 2.2):
    each mass is the smaller of the two paths' Viterbi pair masses, so the release gate never passes a
    bin that the 2.1 gate (the default convention's pair) rejects, and does not depend on the convention.
    """
    same, one, two = transitions.probabilities()
    if masks is None:
        masks = np.left_shift(np.int64(1), np.asarray(paths, dtype=np.int64))
    other = masks if other_paths is None else np.left_shift(np.int64(1), np.asarray(other_paths, dtype=np.int64))
    if transitions.n_states > MAXIMUM_MASK_STATES:
        raise ValueError(f"painting candidate masks hold at most {MAXIMUM_MASK_STATES} states")
    return _release_posterior_kernel(
        emissions, transitions.n_states, same, one, two,
        np.ascontiguousarray(masks, dtype=np.int64), np.ascontiguousarray(other, dtype=np.int64),
        np.ascontiguousarray(public_ids, dtype=np.int32),
        np.ascontiguousarray(local_keys, dtype=np.int32))


def minimum_switch_counts(paths: np.ndarray) -> np.ndarray:
    """int64[N] minimum track changes of each path; a pure track swap counts as none."""
    if paths.shape[2] < 2:
        return np.zeros(paths.shape[0], dtype=np.int64)
    previous = paths[:, :, :-1]
    current = paths[:, :, 1:]
    direct = ((previous[:, 0] != current[:, 0]).astype(np.int8)
              + (previous[:, 1] != current[:, 1]).astype(np.int8))
    swapped = ((previous[:, 0] != current[:, 1]).astype(np.int8)
               + (previous[:, 1] != current[:, 0]).astype(np.int8))
    return np.sum(np.minimum(direct, swapped), axis=1, dtype=np.int64)


@njit(cache=True, parallel=True)
def direct_support(paths, named_alleles, informative, bin_starts, bin_stops, background):
    """bool[N, 2, B]: the track's Viterbi class is called at an informative site of the bin."""
    n_samples, _, n_bins = paths.shape
    result = np.zeros(paths.shape, dtype=np.bool_)
    for task in prange(n_samples * 2):
        sample, track = task // 2, task % 2
        for bin_index in range(n_bins):
            state = paths[sample, track, bin_index]
            if state == background:
                continue
            for site in range(bin_starts[bin_index], bin_stops[bin_index]):
                if named_alleles[state, site] >= 0 and informative[sample, site]:
                    result[sample, track, bin_index] = True
                    break
    return result


@njit(cache=True)
def _distinguishable(first, second, sample, bin_index, named_alleles, observed, bin_starts, bin_stops,
                     n_classes):
    """True if states ``first`` and ``second`` have different calls at a site of the bin that the
    sample observes (read depth > 0; BACKGROUND is uncalled everywhere; uncalled = -1 is a value).
    Discrete: the observed mask and the calls, so GL noise cannot move it."""
    for site in range(bin_starts[bin_index], bin_stops[bin_index]):
        if not observed[sample, site]:
            continue
        first_call = named_alleles[first, site] if first < n_classes else -1
        second_call = named_alleles[second, site] if second < n_classes else -1
        if first_call != second_call:
            return True
    return False


@njit(cache=True, parallel=True)
def switch_interval_masks(paths, named_alleles, observed, bin_starts, bin_stops, n_classes):
    """int64[N, 2, B] candidate states of each track and bin as bit masks (release 2.2 switch intervals).

    A track change a -> b of the (gauge-fixed) Viterbi path at the edge after bin e can sit at any
    edge of the maximal run of bins around e in which a and b are indistinguishable for the sample
    (no observed site of the bin where their calls differ, :func:`_distinguishable`): the data cannot
    place the switch inside the run (its emissions are equal for both), so only the
    transition prior places the switch there, and the traceback convention breaks exact ties. The
    bins of that run get both a and b as candidates; elsewhere the candidate is the Viterbi state.
    A run stops at the track's neighbouring switches (each switch keeps one bin of its own
    segments), so it never removes a switch. Exact discrete test, O(N L).
    """
    n_samples, _, n_bins = paths.shape
    masks = np.empty((n_samples, 2, n_bins), dtype=np.int64)
    for task in prange(n_samples * 2):
        sample, track = task // 2, task % 2
        for bin_index in range(n_bins):
            masks[sample, track, bin_index] = np.int64(1) << paths[sample, track, bin_index]
        previous_edge = -1
        edge = 0
        while edge < n_bins - 1:
            before = paths[sample, track, edge]
            after = paths[sample, track, edge + 1]
            if before == after:
                edge += 1
                continue
            next_edge = edge + 1
            while next_edge < n_bins - 1 and paths[sample, track, next_edge] == paths[sample, track, next_edge + 1]:
                next_edge += 1
            # Left: bins left_edge+1 .. edge could carry ``after`` (bin previous_edge + 1 keeps ``before``).
            left_edge = edge
            while (left_edge - 1 >= previous_edge + 1
                   and not _distinguishable(before, after, sample, left_edge, named_alleles, observed,
                                            bin_starts, bin_stops, n_classes)):
                left_edge -= 1
            # Right: bins edge+1 .. right_edge could carry ``before`` (bin next_edge keeps ``after``;
            # next_edge = B - 1 at the last switch).
            right_edge = edge
            while (right_edge + 1 <= next_edge - 1
                   and not _distinguishable(before, after, sample, right_edge + 1, named_alleles, observed,
                                            bin_starts, bin_stops, n_classes)):
                right_edge += 1
            for bin_index in range(left_edge + 1, right_edge + 1):
                masks[sample, track, bin_index] |= (np.int64(1) << before) | (np.int64(1) << after)
            previous_edge = edge
            edge = next_edge
    return masks


@njit(cache=True, inline="always")
def _bit_count(value):
    count = 0
    while value:
        value &= value - 1
        count += 1
    return count


@njit(cache=True, parallel=True)
def combine_candidate_masks(primary, secondary):
    """int64[N, 2, B]: per bin the smallest unordered cover of two decodings' candidate sets, in ``primary``'s
    track order (release 2.2).

    ``primary`` and ``secondary`` are the candidate masks of two co-optimal Viterbi paths (the default and the
    mirrored traceback conventions, each with its switch intervals). Per bin the two track matchings give the
    covers (P0 | S0, P1 | S1) and (P0 | S1, P1 | S0); the one with fewer candidates is released (a pure track swap
    of the secondary path costs nothing). When both are equally small, they are the same cover if they are equal as
    unordered pairs, and otherwise both tracks get every candidate. Symmetric in the two decodings up to the track
    order, so the released unordered candidate pairs do not depend on which convention is primary. O(N B S).
    """
    n_samples, _, n_bins = primary.shape
    result = np.empty_like(primary)
    for task in prange(n_samples * n_bins):
        sample, bin_index = task // n_bins, task % n_bins
        p0 = primary[sample, 0, bin_index]
        p1 = primary[sample, 1, bin_index]
        s0 = secondary[sample, 0, bin_index]
        s1 = secondary[sample, 1, bin_index]
        direct0, direct1 = p0 | s0, p1 | s1
        crossed0, crossed1 = p0 | s1, p1 | s0
        direct_size = _bit_count(direct0) + _bit_count(direct1)
        crossed_size = _bit_count(crossed0) + _bit_count(crossed1)
        same_cover = ((direct0 == crossed0 and direct1 == crossed1)
                      or (direct0 == crossed1 and direct1 == crossed0))
        if direct_size < crossed_size or (direct_size == crossed_size and same_cover):
            result[sample, 0, bin_index] = direct0
            result[sample, 1, bin_index] = direct1
        elif crossed_size < direct_size:
            result[sample, 0, bin_index] = crossed0
            result[sample, 1, bin_index] = crossed1
        else:
            everything = p0 | p1 | s0 | s1
            result[sample, 0, bin_index] = everything
            result[sample, 1, bin_index] = everything
    return result


def decode_candidates(emissions, transitions, local_keys, named_alleles, observed, bin_starts, bin_stops,
                      n_classes, *, reverse=None):
    """(paths int32[N, 2, B], candidate masks int64[N, 2, B], switch counts int64[N], the mirrored convention's
    paths int32[N, 2, B]) of one batch (release 2.2).

    The gauge-fixed Viterbi path of the run's convention (maths.ties.reverse_conventions(); ``reverse``
    overrides it) gives the released track order; the mirrored convention's path is decoded too, each path's
    switch intervals are taken (:func:`switch_interval_masks`) and the two candidate sets are combined
    (:func:`combine_candidate_masks`), so no released candidate set depends on the traceback convention. Switch
    counts are the fewer of the two paths' minimum track changes (fewest switches, a stated prior).
    """
    from ..maths import ties
    if reverse is None:
        reverse = ties.reverse_conventions()
    decoded = []
    for convention in (bool(reverse), not bool(reverse)):
        paths = parsimonious_track_order(viterbi_paths(emissions, transitions, reverse=convention),
                                         local_keys, n_classes)
        masks = switch_interval_masks(paths, named_alleles, observed, bin_starts, bin_stops, n_classes)
        decoded.append((paths, masks))
    (paths, primary), (other_paths, secondary) = decoded
    counts = np.minimum(minimum_switch_counts(paths), minimum_switch_counts(other_paths))
    return paths, combine_candidate_masks(primary, secondary), counts, other_paths
