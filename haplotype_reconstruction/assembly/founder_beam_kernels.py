"""Compiled bounded-beam kernels for conditional founder paths.

All blocks retain ordinary unnormalized states and stable branch ordering.
Short blocks share immutable emission lookups across their retained paths.
Macro blocks use the original sparse scores inside compiled multi-block scans.
No beam slots are merged or dropped. Normalization is deliberately avoided:
floating ties changed downstream founder paths/counts in validation.
"""
import numpy as np
from numba import njit, prange
from . import founder_sparse

@njit(cache=True, parallel=True, nogil=True)
def score(data, base, size, width, known, block, choices, begin, end, penalty, reverse, first, second, dp, beams, flank, upper_flank, two_flanks, values, uppers):
    focal, samples = (len(known), len(data))
    affected = np.flatnonzero(second == focal)
    s0, s1 = (width - 1, 0) if reverse else (0, width - 1)
    branches = end - begin
    for sample in prange(samples):
        rows = np.empty(len(affected))
        bg0 = np.empty(len(first))
        bg1 = np.empty(len(first))
        focal0 = np.empty((branches, len(affected)))
        focal1 = np.empty_like(focal0)
        for state in range(len(first)):
            if second[state] != focal:
                index = base + (known[first[state], block] * size + known[second[state], block]) * width
                bg0[state] = data[sample, index + s0]
                bg1[state] = data[sample, index + s1]
        for candidate in range(branches):
            choice = choices[begin + candidate]
            for a in range(len(affected)):
                state = affected[a]
                mate = choice if first[state] == focal else known[first[state], block]
                index = base + (mate * size + choice) * width
                focal0[candidate, a] = data[sample, index + s0]
                focal1[candidate, a] = data[sample, index + s1]
        for beam in range(beams):
            switched = -np.inf
            for state in range(len(first)):
                switched = max(switched, dp[beam, sample, state])
            switched -= penalty
            bg_best, bg_stay, bg_enter = (-np.inf, -np.inf, -np.inf)
            up_stay, up_enter = (-np.inf, -np.inf)
            for state in range(len(first)):
                if second[state] == focal:
                    continue
                value = max(dp[beam, sample, state], switched) + bg0[state]
                bg_best = max(bg_best, value)
                if width == 1:
                    bg_stay = max(bg_stay, value + flank[sample, state])
                    if two_flanks:
                        up_stay = max(up_stay, value + upper_flank[sample, state])
                else:
                    e = bg1[state]
                    bg_stay = max(bg_stay, value + e + flank[sample, state])
                    bg_enter = max(bg_enter, e + flank[sample, state])
                    if two_flanks:
                        up_stay = max(up_stay, value + e + upper_flank[sample, state])
                        up_enter = max(up_enter, e + upper_flank[sample, state])
            for candidate in range(branches):
                choice = choices[begin + candidate]
                best = bg_best
                for a in range(len(affected)):
                    state = affected[a]
                    value = max(dp[beam, sample, state], switched) + focal0[candidate, a]
                    rows[a] = value
                    best = max(best, value)
                changed = best - penalty
                value = bg_stay if width == 1 else max(bg_stay, changed + bg_enter)
                upvalue = up_stay if width == 1 else max(up_stay, changed + up_enter)
                for a in range(len(affected)):
                    state = affected[a]
                    row = rows[a]
                    if width == 2:
                        row = max(row, changed) + focal1[candidate, a]
                    value = max(value, row + flank[sample, state])
                    if two_flanks:
                        upvalue = max(upvalue, row + upper_flank[sample, state])
                index = beam * branches + candidate
                values[sample, index] = value
                if two_flanks:
                    uppers[sample, index] = upvalue

@njit(cache=True, parallel=True, nogil=True)
def retain(data, base, size, bins, known, block, choices, begin, branches, penalty, reverse, first, second, dp, order, count, output):
    samples, states = (len(data), len(first))
    focal = len(known)
    addresses = np.empty((branches, states), np.int64)
    for branch in range(branches):
        choice = choices[begin + branch]
        for state in range(states):
            a = known[first[state], block] if first[state] < focal else choice
            b = known[second[state], block] if second[state] < focal else choice
            addresses[branch, state] = base + (a * size + b) * bins
    for task in prange(count * samples):
        kept, sample = (task // samples, task % samples)
        candidate = order[kept]
        parent, branch = (candidate // branches, candidate % branches)
        for state in range(states):
            output[kept, sample, state] = dp[parent, sample, state]
        for step in range(bins):
            marker = bins - 1 - step if reverse else step
            switched = -np.inf
            for state in range(states):
                switched = max(switched, output[kept, sample, state])
            switched -= penalty
            for state in range(states):
                output[kept, sample, state] = max(output[kept, sample, state], switched) + data[sample, addresses[branch, state] + marker]


@njit(cache=True, nogil=True)
def ordered(scores, uppers, ranking, width=0):
    """Stable descending keys, preserving exact original candidate-index ties."""
    if ranking == 0:
        if width > 0 and len(scores) >= 4096 and (width < len(scores)):
            threshold = np.partition(scores, len(scores) - width)[len(scores) - width]
            better = np.flatnonzero(scores > threshold)
            ties = np.flatnonzero(scores == threshold)[:width - len(better)]
            chosen = np.sort(np.concatenate((better, ties)))
            return chosen[np.argsort(-scores[chosen], kind='mergesort')]
        return np.argsort(-scores, kind='mergesort')
    if ranking == 1:
        indices = np.argsort(-uppers, kind='mergesort')
        return indices[np.argsort(-scores[indices], kind='mergesort')]
    indices = np.argsort(-scores, kind='mergesort')
    return indices[np.argsort(-uppers[indices], kind='mergesort')]


@njit(cache=True, nogil=True)
def chunk(data, bases, sizes, bins, known, choices, offsets, penalty, reverse, first, second, suffix, upper_suffix, two_flanks, upper_start, best_seen, ranking, start, stop, dp, alternate, beams, beam_width, values, uppers, ancestry, local_rows, trace_upper, trace_equivalent):
    last_scores = np.empty(0)
    last_order = np.empty(0, np.int64)
    equivalent = True
    for step in range(start, stop):
        block = len(bins) - 1 - step if reverse else step
        begin, end = (offsets[block], offsets[block + 1])
        branches = end - begin
        flank = suffix[step + 1]
        upper_flank = upper_suffix[step - upper_start + 1] if two_flanks else flank
        score(data, bases[block], sizes[block], bins[block], known, block, choices, begin, end, penalty, reverse, first, second, dp, beams, flank, upper_flank, two_flanks, values, uppers)
        length = beams * branches
        scores = np.zeros(length)
        upper = np.zeros(length)
        for sample in range(len(data)):
            for candidate in range(length):
                scores[candidate] += values[sample, candidate]
                if two_flanks:
                    upper[candidate] += uppers[sample, candidate]
        if two_flanks:
            remaining = np.max(upper)
            trace_upper[step] = remaining
            margin = 1e-10 * max(1.0, abs(remaining), abs(best_seen))
            if remaining + margin <= best_seen + 1e-06:
                return (dp, alternate, beams, scores, last_order, True, equivalent)
        order = ordered(scores, upper, ranking, beam_width)
        count = min(beam_width, length)
        if two_flanks:
            ordinary = ordered(scores, upper, 0, beam_width)
            tied = ordered(scores, upper, 1, beam_width)
            for i in range(count):
                if ordinary[i] != tied[i]:
                    equivalent = False
                    trace_equivalent[step] = False
        for i in range(count):
            ancestry[step, i] = order[i] // branches
            local_rows[step, i] = choices[begin + order[i] % branches]
        retain(data, bases[block], sizes[block], bins[block], known, block, choices, begin, branches, penalty, reverse, first, second, dp, order, count, alternate)
        dp, alternate = (alternate, dp)
        beams = count
        last_scores = scores
        last_order = order[:count]
    return (dp, alternate, beams, last_scores, last_order, False, equivalent)

@njit(cache=True, nogil=True)
def branch_scores(dp, emission, known, choices, penalty, reverse, suffix, upper_suffix, first, second, two_flanks, prepared=None, mapping=None, focal_index=0):
    if emission.shape[3] <= 2:
        return founder_sparse._score_short_branches(dp, emission, known, choices, penalty, reverse, suffix, upper_suffix, first, second, two_flanks)
    return founder_sparse._score_general_branches(dp, emission, known, choices, penalty, reverse, suffix, upper_suffix, first, second, two_flanks, prepared, mapping, focal_index)

@njit(cache=True, nogil=True)
def macro_chunk(emissions, data, bases, sizes, bins, known, choices, offsets, penalty, reverse, first, second, suffix, upper_suffix, two_flanks, upper_start, best_seen, ranking, start, stop, dp, alternate, beams, width, ancestry, rows, prepared=None, mapping=None, focal=0):
    scores = np.empty(0)
    order = np.empty(0, np.int64)
    equivalent = True
    for step in range(start, stop):
        block = len(bins) - 1 - step if reverse else step
        begin, end = (offsets[block], offsets[block + 1])
        branches = end - begin
        flank = suffix[step + 1]
        bound = upper_suffix[step - upper_start + 1] if two_flanks else flank
        if prepared is None:
            scores, upper = branch_scores(dp[:beams], emissions[block], np.ascontiguousarray(known[:, block]), choices[begin:end], penalty, reverse, flank, bound, first, second, two_flanks)
        else:
            scores, upper = branch_scores(dp[:beams], emissions[block], np.ascontiguousarray(known[:, block]), choices[begin:end], penalty, reverse, flank, bound, first, second, two_flanks, prepared[block], mapping, focal)
        if two_flanks:
            remaining = np.max(upper)
            margin = 1e-10 * max(1.0, abs(remaining), abs(best_seen))
            if remaining + margin <= best_seen + 1e-06:
                return (dp, alternate, beams, scores, order, True, equivalent)
        order = ordered(scores, upper, ranking, width)[:width]
        if two_flanks:
            ordinary = ordered(scores, upper, 0, width)[:width]
            tied = ordered(scores, upper, 1, width)[:width]
            for i in range(len(order)):
                if ordinary[i] != tied[i]:
                    equivalent = False
        for i in range(len(order)):
            ancestry[step, i] = order[i] // branches
            rows[step, i] = choices[begin + order[i] % branches]
        retain(data, bases[block], sizes[block], bins[block], known, block, choices, begin, branches, penalty, reverse, first, second, dp, order, len(order), alternate)
        dp, alternate = (alternate, dp)
        beams = len(order)
    return (dp, alternate, beams, scores, order, False, equivalent)
