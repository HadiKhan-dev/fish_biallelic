"""Exact conditional Potts scans sharing states that exclude the focal founder.

Only K unordered states contain the focal founder. For the remaining states,
precompute emission-prefix sums and best uninterrupted segment scores. Every
candidate then carries K explicit scores and one scalar entry score per bin,
not a separate K²-state trajectory. Entries include switches between background
states, so this does not freeze their ancestry or approximate the HMM.

For t bins, shared preparation is O(K² t²); a candidate costs O(K t+t²).
Retained beam paths still materialize K² states. Forward and adjoint/backward
operators use the same representation, including transitions at block edges.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def background(emission, known, reverse, backward, first, second):
    states, bins, focal = len(first), emission.shape[2], len(known)
    prefix = np.zeros((states, bins + 1))
    shift = np.zeros(states)
    affected = np.empty(focal + 1, np.int64)
    count = 0
    initial_site = bins - 1 if reverse else 0
    for state in range(states):
        a, b = first[state], second[state]
        if b == focal:
            affected[count] = state
            count += 1
            continue
        if backward:
            shift[state] = emission[known[a], known[b], initial_site]
        for step in range(bins):
            offset = step + int(backward)
            value = 0.0
            if offset < bins:
                site = bins - 1 - offset if reverse else offset
                value = emission[known[a], known[b], site]
            prefix[state, step + 1] = prefix[state, step] + value
    segment = np.full((bins + 1, bins + 1), -np.inf)
    for state in range(states):
        if second[state] == focal:
            continue
        for start in range(bins):
            for stop in range(start + 1, bins + 1):
                value = prefix[state, stop] - prefix[state, start]
                segment[start, stop] = max(segment[start, stop], value)
    return prefix, shift, segment, affected


@njit(cache=True, inline="always")
def cached_background(emission, known, reverse, backward, first, second,
                      prepared, focal_index, sample):
    if prepared is None:
        return background(emission, known, reverse, backward, first, second)
    return (prepared[0][sample], prepared[1][sample],
            prepared[2][sample, focal_index], np.flatnonzero(second == len(known)))


@njit(cache=True, inline="always")
def initial_background(previous, prefix, shift, second, focal, mapping=None, out=None):
    best = np.empty(prefix.shape[1]) if out is None else out
    best[:] = -np.inf
    for state in range(len(previous)):
        if second[state] != focal:
            for stop in range(len(best)):
                index = state if mapping is None else mapping[state]
                best[stop] = max(best[stop], previous[state] + shift[index] + prefix[index, stop])
    return best


@njit(cache=True, inline="always")
def candidate_into(previous, emission, known, choice, penalty, reverse, backward,
                   first, affected, initial, segment, row, entries):
    bins, focal = emission.shape[2], len(known)
    initial_site = bins - 1 if reverse else 0
    for index, state in enumerate(affected):
        mate = choice if first[state] == focal else known[first[state]]
        row[index] = previous[state]
        if backward:
            row[index] += emission[mate, choice, initial_site]
    if bins == 1:
        entry = max(initial[0], np.max(row)) - penalty
        entries[0] = entry
        for index, state in enumerate(affected):
            mate = choice if first[state] == focal else known[first[state]]
            value = 0.0 if backward else emission[mate, choice, 0]
            row[index] = max(row[index], entry) + value
        return
    outside = initial[0]
    for step in range(bins):
        entry = max(outside, np.max(row)) - penalty
        entries[step] = entry
        offset = step + int(backward)
        for index, state in enumerate(affected):
            value = 0.0
            if offset < bins:
                site = bins - 1 - offset if reverse else offset
                mate = choice if first[state] == focal else known[first[state]]
                value = emission[mate, choice, site]
            row[index] = max(row[index], entry) + value
        outside = initial[step + 1]
        for start in range(step + 1):
            outside = max(outside, entries[start] + segment[start, step + 1])



@njit(cache=True, inline="always")
def candidate(previous, emission, known, choice, penalty, reverse, backward,
              first, affected, initial, segment):
    """Owned-result wrapper for callers that do not already own scratch arrays."""
    row, entries = np.empty(len(affected)), np.empty(emission.shape[2])
    candidate_into(previous, emission, known, choice, penalty, reverse, backward,
                   first, affected, initial, segment, row, entries)
    return row, entries


@njit(cache=True, inline="always")
def terminal_background(previous, prefix, shift, second, focal, flank, mapping=None, out=None):
    bins = prefix.shape[1] - 1
    # Last slot is the initial-state contribution, other slots are entries.
    values = np.empty(bins + 1) if out is None else out
    values[:] = -np.inf
    for state in range(len(previous)):
        if second[state] == focal:
            continue
        index = state if mapping is None else mapping[state]
        end = prefix[index, bins] + flank[state]
        values[bins] = max(values[bins], previous[state] + shift[index] + end)
        for start in range(bins):
            values[start] = max(values[start], end - prefix[index, start])
    return values


@njit(cache=True, inline="always")
def terminal_score(row, entries, affected, flank, terminal):
    value = terminal[-1]
    for start in range(len(entries)):
        value = max(value, entries[start] + terminal[start])
    for index, state in enumerate(affected):
        value = max(value, row[index] + flank[state])
    return value


@njit(cache=True, inline="always")
def combine(previous, prefix, shift, second, focal, affected, rows, entries, weights, mapping=None):
    bins = prefix.shape[1] - 1
    result = np.full(len(previous), -np.inf)
    best_entries = np.full(bins, -np.inf)
    best_weight = np.max(weights)
    for choice in range(len(weights)):
        for index, state in enumerate(affected):
            result[state] = max(result[state], rows[choice, index] + weights[choice])
        for start in range(bins):
            best_entries[start] = max(best_entries[start], entries[choice, start] + weights[choice])
    for state in range(len(previous)):
        if second[state] == focal:
            continue
        index = state if mapping is None else mapping[state]
        value = previous[state] + shift[index] + best_weight
        for start in range(bins):
            value = max(value, best_entries[start] - prefix[index, start])
        result[state] = value + prefix[index, bins]
    return result


@njit(cache=True, inline="always")
def backward_sample(previous, emission, known, choices, weights, penalty,
                    reverse, first, second, prepared=None, mapping=None,
                    focal_index=0, sample=0):
    prefix, shift, segment, affected = cached_background(
        emission, known, reverse, True, first, second, prepared, focal_index, sample)
    initial = initial_background(previous, prefix, shift, second, len(known), mapping)
    rows = np.empty((len(choices), len(affected)))
    entries = np.empty((len(choices), emission.shape[2]))
    for index, choice in enumerate(choices):
        candidate_into(previous, emission, known, choice, penalty,
            reverse, True, first, affected, initial, segment, rows[index], entries[index])
    return combine(previous, prefix, shift, second, len(known), affected, rows, entries, weights, mapping)


@njit(cache=True, parallel=True, nogil=True)
def forward_details(previous, emission, known, choices, penalty, reverse, first, second, suffix,
                    prepared=None, mapping=None, focal_index=0, weights=None):
    samples, bins, founders = len(previous), emission.shape[3], len(known) + 1
    rows = np.empty((samples, len(choices), founders))
    entries = np.empty((samples, len(choices), bins))
    scores = np.empty((samples, len(choices)))
    for sample in prange(samples):
        prefix, shift, segment, affected = cached_background(
            emission[sample], known, reverse, False, first, second, prepared, focal_index, sample)
        initial = initial_background(previous[sample], prefix, shift, second, len(known), mapping)
        terminal = terminal_background(previous[sample], prefix, shift, second, len(known), suffix[sample], mapping)
        for index, choice in enumerate(choices):
            candidate_into(previous[sample], emission[sample], known, choice,
                penalty, reverse, False, first, affected, initial, segment,
                rows[sample, index], entries[sample, index])
            scores[sample, index] = terminal_score(rows[sample, index],
                entries[sample, index], affected, suffix[sample], terminal)
        if weights is not None:
            for index in range(len(choices)):
                scores[sample, index] += weights[sample, index]
            normalizer = np.max(scores[sample])
            for index in range(len(choices)):
                scores[sample, index] -= normalizer
    return rows, entries, scores


@njit(cache=True, parallel=True, nogil=True)
def combine_forward(previous, emission, known, reverse, first, second, rows, entries, weights,
                    prepared=None, mapping=None, focal_index=0, marginals=None,
                    common=None, normalizers=None):
    result = np.empty_like(previous)
    for sample in prange(len(previous)):
        bins, focal = emission.shape[3], len(known)
        affected = np.flatnonzero(second == focal)
        if prepared is None:
            prefix = np.zeros((len(first), bins + 1))
            shift = np.zeros(len(first))
            for state in range(len(first)):
                if second[state] != focal:
                    for step in range(bins):
                        site = bins - 1 - step if reverse else step
                        prefix[state, step + 1] = prefix[state, step] + emission[sample, known[first[state]], known[second[state]], site]
        else:
            prefix, shift = prepared[0][sample], prepared[1][sample]
        if marginals is not None:
            for index in range(len(weights[sample])):
                weights[sample, index] += common[index] - marginals[sample, index]
        result[sample] = combine(previous[sample], prefix, shift, second,
            focal, affected, rows[sample], entries[sample], weights[sample], mapping)
        if normalizers is not None:
            normalizer = np.max(result[sample])
            result[sample] -= normalizer
            normalizers[sample] = normalizer
    return result


@njit(cache=True, parallel=True, nogil=True)
def _score_general_branches(dp, emission, known, choices, penalty, reverse, suffix,
                   upper_suffix, first, second, two_flanks, prepared=None, mapping=None, focal_index=0):
    beams, samples = dp.shape[:2]
    values = np.empty((samples, beams * len(choices)))
    upper = np.empty_like(values) if two_flanks else np.empty((0, 0))
    for sample in prange(samples):
        prefix, shift, segment, affected = cached_background(
            emission[sample], known, reverse, False, first, second, prepared, focal_index, sample)
        row, entries = np.empty(len(affected)), np.empty(emission.shape[3])
        initial = np.empty(prefix.shape[1])
        terminal, optimistic = np.empty(prefix.shape[1]), np.empty(prefix.shape[1])
        for beam in range(beams):
            previous = dp[beam, sample]
            initial_background(previous, prefix, shift, second, len(known), mapping, initial)
            terminal_background(previous, prefix, shift, second, len(known), suffix[sample], mapping, terminal)
            if two_flanks:
                terminal_background(previous, prefix, shift, second, len(known), upper_suffix[sample], mapping, optimistic)
            for index, choice in enumerate(choices):
                candidate_into(previous, emission[sample], known, choice, penalty,
                    reverse, False, first, affected, initial, segment, row, entries)
                output = beam * len(choices) + index
                values[sample, output] = terminal_score(row, entries, affected, suffix[sample], terminal)
                if two_flanks:
                    upper[sample, output] = terminal_score(
                        row, entries, affected, upper_suffix[sample], optimistic)
    scores = values.sum(axis=0)
    if two_flanks:
        return scores, upper.sum(axis=0)
    return scores, scores


@njit(cache=True, parallel=True, nogil=True)
def _score_short_branches(dp, emission, known, choices, penalty, reverse, suffix,
                          upper_suffix, first, second, two_flanks):
    beams, samples = dp.shape[:2]
    focal, bins = len(known), emission.shape[3]
    values = np.empty((samples, beams * len(choices)))
    upper = np.empty_like(values) if two_flanks else np.empty((0, 0))
    affected = np.flatnonzero(second == focal)
    s0, s1 = (bins - 1, 0) if reverse else (0, bins - 1)
    for sample in prange(samples):
        focal_previous = np.empty(len(affected))
        for beam in range(beams):
            switched = np.max(dp[beam, sample]) - penalty
            bg_best, bg_stay, bg_enter = -np.inf, -np.inf, -np.inf
            up_stay, up_enter = -np.inf, -np.inf
            for state in range(len(first)):
                if second[state] == focal:
                    continue
                a, b = known[first[state]], known[second[state]]
                value = max(dp[beam, sample, state], switched) + emission[sample, a, b, s0]
                bg_best = max(bg_best, value)
                if bins == 1:
                    bg_stay = max(bg_stay, value + suffix[sample, state])
                    if two_flanks:
                        up_stay = max(up_stay, value + upper_suffix[sample, state])
                else:
                    e = emission[sample, a, b, s1]
                    bg_stay = max(bg_stay, (value + e) + suffix[sample, state])
                    bg_enter = max(bg_enter, e + suffix[sample, state])
                    if two_flanks:
                        up_stay = max(up_stay, (value + e) + upper_suffix[sample, state])
                        up_enter = max(up_enter, e + upper_suffix[sample, state])
            for candidate, choice in enumerate(choices):
                best = bg_best
                for index, state in enumerate(affected):
                    mate = choice if first[state] == focal else known[first[state]]
                    value = max(dp[beam, sample, state], switched) + emission[sample, mate, choice, s0]
                    focal_previous[index] = value
                    best = max(best, value)
                changed = best - penalty
                value = bg_stay if bins == 1 else max(bg_stay, changed + bg_enter)
                upper_value = up_stay if bins == 1 else max(up_stay, changed + up_enter)
                for index, state in enumerate(affected):
                    row = focal_previous[index]
                    if bins == 2:
                        mate = choice if first[state] == focal else known[first[state]]
                        row = max(row, changed) + emission[sample, mate, choice, s1]
                    value = max(value, row + suffix[sample, state])
                    if two_flanks:
                        upper_value = max(upper_value, row + upper_suffix[sample, state])
                output = beam * len(choices) + candidate
                values[sample, output] = value
                if two_flanks:
                    upper[sample, output] = upper_value
    scores = values.sum(axis=0)
    return (scores, upper.sum(axis=0)) if two_flanks else (scores, scores)


def score_branches(dp, emission, known, choices, penalty, reverse, suffix,
                   upper_suffix, first, second, two_flanks, prepared=None,
                   mapping=None, focal_index=0):
    """Exact short-bin specialization; generic sparse scan for macro blocks.

    Sample-major scratch gives each worker contiguous, private writes. At one
    or two bins the background reduces to stay/enter maxima; each candidate
    scans only the K affected states, with all transitions and bins retained.
    """
    if emission.shape[3] <= 2:
        return _score_short_branches(dp, emission, known, choices, penalty,
            reverse, suffix, upper_suffix, first, second, two_flanks)
    return _score_general_branches(dp, emission, known, choices, penalty,
        reverse, suffix, upper_suffix, first, second, two_flanks,
        prepared, mapping, focal_index)
