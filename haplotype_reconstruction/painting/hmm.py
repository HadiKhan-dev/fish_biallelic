"""Numerical kernels of the painting HMM: emissions, the label-level max-product step, track order and the
switch intervals of decoded paths.

State of a sample in a bin: an ordered pair (x1, x2) of labels, flat
index x1 * S + x2 (S = classes + BACKGROUND). Emissions are binned sums of
robust per-site genotype log likelihoods, centred by log(1/3) so unobserved
and flat sites are neutral (the decoder divides each sample's bin by its design effect, design_effect).
The label-level transition is the Hamming table of painting/transitions.py, so its
max-product step contracts in O(S^2) per bin from the best and second-best entries
of each row and column (label continuity). Painting's decoder, the class x fidelity
chain, is painting/fidelity.py. Samples are independent; every per-sample reduction
runs in a fixed order, so results do not depend on the thread count.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ..maths.ties import noise_quantum

# Spread of a normalised GL row above which the row is informative (not flat).
INFORMATIVE_SPREAD = 16.0 * np.finfo(np.float64).eps
# Viterbi tie quantum (nats; release 2.1, design A-4): predecessors whose path scores are within this of the best are
# tied, and the back-pointer then takes the fewest track changes (painting.fidelity: per track, staying, then the same
# label, then a label change), then the smallest index; the final state is the smallest index within it of the best.
# Since 2.14 the scores are of tempered emissions (divided by their design effect, design_effect), so the quantum sits
# further above the GL-noise motion, and more near-ties take the fewest changes. Equal-score paths arise where two
# switch placements cost the same (equal theta on a cold, zero-cM stretch of the map, and the states' emissions
# identical in the bins between) and their sums differ only in rounding, which GL noise reorders. The path scores
# themselves stay exact maxima. Census (TroMau chr18 painting, iid 1e-9; work/IE/census/viterbi_chr18_*): 4,980
# decisions within 1e-2, gap motion <= 1.9e-9, genuine gaps from 1e-5 (2 in [1e-5, 3e-5), then >= 3e-4): 2^-20 = 9.5e-7
# sat ~500x above that motion. Release 2.13 places it against 1e-6 relative GL noise (maths.ties.GL_NOISE; gap motion <=
# 1.9e-6, the census times 10^3): noise_quantum = 2^-15 = 3.1e-5; gaps below it take the fewest track changes (a safety
# margin over the motion; it also covers the census's genuine gaps in [1e-5, 3e-5)).
VITERBI_TIE = noise_quantum(1.9e-6)
# Candidate states of a track are int64 bit masks (painting switch intervals, release 2.2): S = classes + BACKGROUND
# must fit 63 bits (K <= 30 founders gives S <= 31).
MAXIMUM_MASK_STATES = 63


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


def design_effect(likelihoods: np.ndarray, observed: np.ndarray, bin_starts: np.ndarray,
                  correlation: float) -> np.ndarray:
    """float64[N, B]: 1 + (n - 1) r for the n informative sites (informative_sites) of each sample in each bin (1 when
    n <= 1), r = ``correlation``; ``likelihoods``/``observed`` are gathered at the component's selected sites and
    ``bin_starts`` are the bins' first selected sites (release 2.14). The variance inflation of a sum of n
    equal-variance terms with common correlation r (Kish's design effect), used as a tuned working weight of a bin's
    log emission: under that assumption the bin counts as n / (1 + (n - 1) r) variance-equivalent sites. Not an
    estimated dependence correction; the weight tends to 1 as n -> 1, and a bin without informative sites has 1."""
    counts = np.add.reduceat(informative_sites(likelihoods, observed).astype(np.int64),
                             np.asarray(bin_starts, dtype=np.int64), axis=1)
    return 1.0 + np.maximum(counts - 1, 0) * float(correlation)


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
def hamming_max_values(values, cost_same, cost_one, cost_two, n_states, out, row_first, row_second,
                       column_first, column_second, row_first_column, column_first_row):
    """out[z] = max_y values[y] + T(y, z) under the label-level Hamming table, without back-pointers.

    Each excluded maximum needs only the best and second-best values of a row or column and
    where the best one lies. Used where only scores matter (label continuity's screen).
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
def _distinguishable(first, second, partner, sample, bin_index, named_alleles, observed, bin_starts, bin_stops,
                     n_classes):
    """True if the pairs (first, partner) and (second, partner) have different genotype laws at a site of the bin
    that the sample observes (read depth > 0), i.e. their emissions can differ there. Discrete: the observed mask and
    the calls, so GL noise cannot move it.

    Different calls (0, 1 or uncalled = -1; BACKGROUND is uncalled everywhere) differ. Equal calls give equal laws,
    except where both are uncalled and exactly one of the two classes is the partner's own class: two
    tracks in one class share one latent allele (no heterozygote), two different classes do not (release 2.14; the
    interval used to treat every equal call, -1 included, as interchangeable)."""
    first_shared = first == partner and first < n_classes
    second_shared = second == partner and second < n_classes
    for site in range(bin_starts[bin_index], bin_stops[bin_index]):
        if not observed[sample, site]:
            continue
        first_call = named_alleles[first, site] if first < n_classes else -1
        second_call = named_alleles[second, site] if second < n_classes else -1
        if first_call != second_call:
            return True
        if first_call < 0 and first_shared != second_shared:
            return True
    return False


@njit(cache=True, parallel=True)
def switch_interval_masks(paths, named_alleles, observed, bin_starts, bin_stops, n_classes):
    """int64[N, 2, B] candidate states of each track and bin as bit masks (release 2.2 switch intervals).

    A track change a -> b of the (gauge-fixed) Viterbi path at the edge after bin e can sit at any
    edge of the maximal run of bins around e in which a and b are indistinguishable for the sample
    (paired with the other track's state, no observed site of the bin where their genotype laws differ,
    :func:`_distinguishable`): the data cannot
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
                   and not _distinguishable(before, after, paths[sample, 1 - track, left_edge], sample, left_edge,
                                            named_alleles, observed, bin_starts, bin_stops, n_classes)):
                left_edge -= 1
            # Right: bins edge+1 .. right_edge could carry ``before`` (bin next_edge keeps ``after``;
            # next_edge = B - 1 at the last switch).
            right_edge = edge
            while (right_edge + 1 <= next_edge - 1
                   and not _distinguishable(before, after, paths[sample, 1 - track, right_edge + 1], sample,
                                            right_edge + 1, named_alleles, observed, bin_starts, bin_stops,
                                            n_classes)):
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


@njit(cache=True, inline="always")
def _mask_change(previous, current, bin_index, local_keys, n_classes, n_states):
    """(changed, visible) of one track moving between candidate masks: (0, 0) when the masks share a state; else
    changed, and allele-invisible, (1, 0), when some pair of their states shares a local key below C in bin_index - 1
    or bin_index (:func:`_allele_visible_change`), else (1, 1)."""
    if previous & current:
        return 0, 0
    for first in range(n_states):
        if not (previous >> first) & 1:
            continue
        for second in range(n_states):
            if (current >> second) & 1 and _allele_visible_change(first, second, bin_index, local_keys,
                                                                  n_classes) == 0:
                return 1, 0
    return 1, 1


@njit(cache=True, parallel=True)
def canonical_track_order(masks, local_keys, n_classes):
    """bool[N, B]: whether to exchange the two tracks of each bin, so that the track order (the H1/H2 gauge) is a
    function of the unordered candidate pairs alone (release 2.14).

    The unordered pair of candidate masks of a bin does not depend on the traceback convention
    (:func:`combine_candidate_masks`); the order read from one convention's Viterbi path did (its final state and
    traceback run from opposite ends). Right to left, as the default traceback: the last bin puts the smaller mask on
    track 0; every earlier bin takes the orientation with fewer track changes towards the bin after it (a track
    keeps its identity while consecutive masks share a state), then fewer allele-visible changes, and when both
    orientations are equal, the smaller mask on track 0. Unordered states, posteriors, switch counts and releases are
    unchanged. O(N B) for single-state masks (O(N B S^2) at worst).
    """
    n_samples, _, n_bins = masks.shape
    n_states = local_keys.shape[1]
    swap = np.zeros((n_samples, n_bins), dtype=np.bool_)
    for sample in prange(n_samples):
        if n_bins == 0:
            continue
        last = n_bins - 1
        first = masks[sample, 0, last]
        second = masks[sample, 1, last]
        exchange = second < first
        swap[sample, last] = exchange
        next_first = second if exchange else first
        next_second = first if exchange else second
        for bin_index in range(n_bins - 2, -1, -1):
            first = masks[sample, 0, bin_index]
            second = masks[sample, 1, bin_index]
            # The change between bin_index and bin_index + 1 is judged on the local keys of both bins.
            direct_first, direct_visible_first = _mask_change(
                next_first, first, bin_index + 1, local_keys, n_classes, n_states)
            direct_second, direct_visible_second = _mask_change(
                next_second, second, bin_index + 1, local_keys, n_classes, n_states)
            crossed_first, crossed_visible_first = _mask_change(
                next_first, second, bin_index + 1, local_keys, n_classes, n_states)
            crossed_second, crossed_visible_second = _mask_change(
                next_second, first, bin_index + 1, local_keys, n_classes, n_states)
            direct = direct_first + direct_second
            crossed = crossed_first + crossed_second
            if direct != crossed:
                exchange = crossed < direct
            elif direct_visible_first + direct_visible_second != crossed_visible_first + crossed_visible_second:
                exchange = (crossed_visible_first + crossed_visible_second
                            < direct_visible_first + direct_visible_second)
            else:
                exchange = second < first
            swap[sample, bin_index] = exchange
            next_first = second if exchange else first
            next_second = first if exchange else second
    return swap


def apply_track_order(swap: np.ndarray, *arrays: np.ndarray) -> None:
    """Exchange the two tracks of each [N, 2, B] array in place where ``swap`` [N, B] is set."""
    rows, columns = np.nonzero(swap)
    for array in arrays:
        first = array[rows, 0, columns].copy()
        array[rows, 0, columns] = array[rows, 1, columns]
        array[rows, 1, columns] = first
