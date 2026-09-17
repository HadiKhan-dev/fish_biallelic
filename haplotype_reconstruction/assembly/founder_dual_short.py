"""Exact one/two-bin founder searches with sample-contiguous coordinate updates.

The forward and adjoint scans retain every bin, candidate and diploid state.
Sample-contiguous workspaces expose independent SIMD lanes without changing
the generic recurrence's evaluation order or adding per-block thread barriers.
Keep its prefix/segment arithmetic and message-update parentheses: equivalent
algebraic simplification changed tie-sensitive paths in chromosome controls. No fast-math
flags are used. Generic multi-bin searches live in founder_dual_search.

Packed transposes belong to the immutable PreparedModels owner and are shared
only among its conditional queries. Coordinate messages remain query-local.
"""
import numpy as np
from numba import njit, prange

@njit(cache=True, parallel=True, nogil=True)
def dual_suffix(data, bases, sizes, bins, known, choices, offsets, messages, penalty, reverse, first, second):
    blocks, samples, states = (len(bins), len(data), len(first))
    focal = len(known)
    affected = np.flatnonzero(second == focal)
    result = np.zeros((blocks + 1, samples, states))
    scales = np.zeros(samples)
    for sample in prange(samples):
        shift = np.empty(states)
        endbg = np.empty(states)
        row = np.empty(len(affected))
        bestrow = np.empty(len(affected))
        for step in range(blocks - 1, -1, -1):
            block = blocks - 1 - step if reverse else step
            base, size, width = (bases[block], sizes[block], bins[block])
            s0, s1 = (0, width - 1) if reverse else (width - 1, 0)
            initial0, initial1, segment0 = (-np.inf, -np.inf, -np.inf)
            for state in range(states):
                if second[state] == focal:
                    continue
                index = base + (known[first[state], block] * size + known[second[state], block]) * width
                shift[state] = data[sample, index + s0]
                endbg[state] = data[sample, index + s1] if width == 2 else 0.0
                value = result[step + 1, sample, state] + shift[state]
                initial0 = max(initial0, value + 0.0)
                initial1 = max(initial1, value + endbg[state])
                segment0 = max(segment0, endbg[state])
            bestrow[:] = -np.inf
            bestweight, bestentry0, bestentry1 = (-np.inf, -np.inf, -np.inf)
            for local in range(offsets[block], offsets[block + 1]):
                choice = choices[local]
                weight = messages[sample, local]
                bestweight = max(bestweight, weight)
                highest = -np.inf
                for a in range(len(affected)):
                    state = affected[a]
                    mate = choice if first[state] == focal else known[first[state], block]
                    index = base + (mate * size + choice) * width
                    row[a] = result[step + 1, sample, state] + data[sample, index + s0]
                    highest = max(highest, row[a])
                entry0 = max(initial0, highest) - penalty
                highest = -np.inf
                for a in range(len(affected)):
                    state = affected[a]
                    mate = choice if first[state] == focal else known[first[state], block]
                    index = base + (mate * size + choice) * width
                    value = data[sample, index + s1] if width == 2 else 0.0
                    row[a] = max(row[a], entry0) + value
                    highest = max(highest, row[a])
                bestentry0 = max(bestentry0, entry0 + weight)
                if width == 2:
                    outside = max(initial1, entry0 + segment0)
                    entry1 = max(outside, highest) - penalty
                    bestentry1 = max(bestentry1, entry1 + weight)
                    for a in range(len(affected)):
                        row[a] = max(row[a], entry1) + 0.0
                for a in range(len(affected)):
                    bestrow[a] = max(bestrow[a], row[a] + weight)
            for state in range(states):
                if second[state] == focal:
                    continue
                value = result[step + 1, sample, state] + shift[state] + bestweight
                value = max(value, bestentry0 - 0.0)
                if width == 2:
                    value = max(value, bestentry1 - endbg[state])
                result[step, sample, state] = value + endbg[state]
            for a in range(len(affected)):
                result[step, sample, affected[a]] = bestrow[a]
            normalizer = np.max(result[step, sample])
            result[step, sample] -= normalizer
            scales[sample] += normalizer
    return (result, scales.sum())


@njit(cache=True, parallel=True, nogil=True)
def window_suffix(data, bases, sizes, bins, known, choices, offsets, penalty, start, stop, terminal, first, second):
    blocks, samples, states = (stop - start, len(data), len(first))
    focal = len(known)
    affected = np.flatnonzero(second == focal)
    result = np.zeros((blocks + 1, samples, states))
    result[-1] = terminal
    for sample in prange(samples):
        bg = np.empty(states)
        aff = np.empty(len(affected))
        values = np.empty(len(affected))
        for step in range(blocks - 1, -1, -1):
            block = start + step
            base, size, width = (bases[block], sizes[block], bins[block])
            s0, s1 = (width - 1, 0)
            begin, end = (offsets[block], offsets[block + 1])
            bgmax, bgstay, bgemax = (-np.inf, -np.inf, -np.inf)
            for state in range(states):
                if second[state] == focal:
                    continue
                index = base + (known[first[state], block] * size + known[second[state], block]) * width
                bg[state] = result[step + 1, sample, state] + data[sample, index + s0]
                bgmax = max(bgmax, bg[state])
                if width == 2:
                    e = data[sample, index + s1]
                    bgstay = max(bgstay, bg[state] + e)
                    bgemax = max(bgemax, e)
            aff[:] = -np.inf
            weightmax, entrymax = (-np.inf, -np.inf)
            for local in range(begin, end):
                choice, weight = (choices[local], 0.0)
                weightmax = max(weightmax, weight)
                highest = bgmax
                for a in range(len(affected)):
                    state = affected[a]
                    mate = choice if first[state] == focal else known[first[state], block]
                    index = base + (mate * size + choice) * width
                    value = result[step + 1, sample, state] + data[sample, index + s0]
                    values[a] = value
                    highest = max(highest, value)
                if width == 1:
                    for a in range(len(affected)):
                        aff[a] = max(aff[a], values[a] + weight)
                else:
                    entry = highest - penalty
                    entrymax = max(entrymax, entry + weight)
                    for a in range(len(affected)):
                        state = affected[a]
                        mate = choice if first[state] == focal else known[first[state], block]
                        index = base + (mate * size + choice) * width
                        value = max(values[a], entry) + data[sample, index + s1] + weight
                        aff[a] = max(aff[a], value)
            top = bgmax + weightmax if width == 1 else max(bgstay + weightmax, bgemax + entrymax)
            for a in range(len(affected)):
                top = max(top, aff[a])
            entry = top - penalty
            for state in range(states):
                if second[state] == focal:
                    continue
                value = bg[state] + weightmax
                if width == 2:
                    index = base + (known[first[state], block] * size + known[second[state], block]) * width
                    value = max(value, entrymax) + data[sample, index + s1]
                result[step, sample, state] = max(value, entry)
            for a in range(len(affected)):
                result[step, sample, affected[a]] = max(aff[a], entry)
    return result

def pack_coordinate(models, packed):
    if not hasattr(models, '_pack_lock'):
        return (np.ascontiguousarray(packed[0].T), *packed[1:])
    with models._pack_lock:
        if not hasattr(models, 'sample_contiguous_dual'):
            models.sample_contiguous_dual = (np.ascontiguousarray(packed[0].T), *packed[1:])
        return models.sample_contiguous_dual

@njit(cache=True, nogil=True)
def coordinate_sweep(data, bases, sizes, bins, known, incumbent, choices, offsets, messages, suffixes, penalty, reverse, first, second):
    blocks, samples, states = (len(bins), data.shape[1], len(first))
    focal = len(known)
    affected = np.flatnonzero(second == focal)
    max_choices = np.max(np.diff(offsets))
    prefix = np.zeros((states, samples))
    scales = np.zeros(samples)
    bg0 = np.empty((states, samples))
    bgend = np.empty((states, samples))
    rows = np.empty((max_choices, len(affected), samples))
    entries0 = np.empty((max_choices, samples))
    entries1 = np.empty_like(entries0)
    marginals = np.empty((max_choices, samples))
    common = np.empty(max_choices)
    decoded = np.empty(blocks, np.int64)
    predicted = 0.0
    initial0 = np.empty(samples)
    initial1 = np.empty(samples)
    segment0 = np.empty(samples)
    terminal = np.empty(samples)
    term0 = np.empty(samples)
    term1 = np.empty(samples)
    normalizer = np.empty(samples)
    highest = np.empty(samples)
    bestweight = np.empty(samples)
    bestentry0 = np.empty(samples)
    bestentry1 = np.empty(samples)
    for step in range(blocks):
        block = blocks - 1 - step if reverse else step
        base, size, width = (bases[block], sizes[block], bins[block])
        s0, s1 = (width - 1, 0) if reverse else (0, width - 1)
        begin, end = (offsets[block], offsets[block + 1])
        count = end - begin
        initial0[:] = -np.inf
        initial1[:] = -np.inf
        segment0[:] = -np.inf
        terminal[:] = -np.inf
        term0[:] = -np.inf
        term1[:] = -np.inf
        for state in range(states):
            if second[state] == focal:
                continue
            index = base + (known[first[state], block] * size + known[second[state], block]) * width
            for sample in range(samples):
                e0 = 0.0 + data[index + s0, sample]
                eend = e0 + data[index + s1, sample] if width == 2 else e0
                bg0[state, sample] = e0
                bgend[state, sample] = eend
                previous = prefix[state, sample]
                initial0[sample] = max(initial0[sample], previous + 0.0 + 0.0)
                initial1[sample] = max(initial1[sample], previous + 0.0 + e0)
                segment0[sample] = max(segment0[sample], e0 - 0.0)
                ending = eend + suffixes[step + 1, state, sample]
                terminal[sample] = max(terminal[sample], previous + 0.0 + ending)
                term0[sample] = max(term0[sample], ending - 0.0)
                if width == 2:
                    term1[sample] = max(term1[sample], ending - e0)
        normalizer[:] = -np.inf
        for local in range(count):
            choice = choices[begin + local]
            highest[:] = -np.inf
            for a in range(len(affected)):
                state = affected[a]
                for sample in range(samples):
                    highest[sample] = max(highest[sample], prefix[state, sample])
            for sample in range(samples):
                entries0[local, sample] = max(initial0[sample], highest[sample]) - penalty
            highest[:] = -np.inf
            for a in range(len(affected)):
                state = affected[a]
                mate = choice if first[state] == focal else known[first[state], block]
                index = base + (mate * size + choice) * width
                for sample in range(samples):
                    value = max(prefix[state, sample], entries0[local, sample]) + data[index + s0, sample]
                    rows[local, a, sample] = value
                    highest[sample] = max(highest[sample], value)
            if width == 2:
                for sample in range(samples):
                    outside = max(initial1[sample], entries0[local, sample] + segment0[sample])
                    entries1[local, sample] = max(outside, highest[sample]) - penalty
                for a in range(len(affected)):
                    state = affected[a]
                    mate = choice if first[state] == focal else known[first[state], block]
                    index = base + (mate * size + choice) * width
                    for sample in range(samples):
                        rows[local, a, sample] = max(rows[local, a, sample], entries1[local, sample]) + data[index + s1, sample]
            for sample in range(samples):
                value = max(terminal[sample], entries0[local, sample] + term0[sample])
                if width == 2:
                    value = max(value, entries1[local, sample] + term1[sample])
                marginals[local, sample] = value
            for a in range(len(affected)):
                state = affected[a]
                for sample in range(samples):
                    marginals[local, sample] = max(marginals[local, sample], rows[local, a, sample] + suffixes[step + 1, state, sample])
            for sample in range(samples):
                marginals[local, sample] += messages[begin + local, sample]
                normalizer[sample] = max(normalizer[sample], marginals[local, sample])
        for local in range(count):
            for sample in range(samples):
                marginals[local, sample] -= normalizer[sample]
            total = 0.0
            for sample in range(samples):
                total += marginals[local, sample]
            common[local] = total / samples
        best = 0
        for local in range(1, count):
            if common[local] > common[best]:
                best = local
        for local in range(count):
            if choices[begin + local] == incumbent[block] and common[local] == common[best]:
                best = local
        decoded[block] = choices[begin + best]
        predicted += samples * common[best]
        bestweight[:] = -np.inf
        bestentry0[:] = -np.inf
        bestentry1[:] = -np.inf
        for local in range(count):
            for sample in range(samples):
                weight = messages[begin + local, sample] + (common[local] - marginals[local, sample])
                messages[begin + local, sample] = weight
                bestweight[sample] = max(bestweight[sample], weight)
                bestentry0[sample] = max(bestentry0[sample], entries0[local, sample] + weight)
                if width == 2:
                    bestentry1[sample] = max(bestentry1[sample], entries1[local, sample] + weight)
        normalizer[:] = -np.inf
        for state in range(states):
            if second[state] == focal:
                continue
            for sample in range(samples):
                value = prefix[state, sample] + 0.0 + bestweight[sample]
                value = max(value, bestentry0[sample] - 0.0)
                if width == 2:
                    value = max(value, bestentry1[sample] - bg0[state, sample])
                prefix[state, sample] = value + bgend[state, sample]
                normalizer[sample] = max(normalizer[sample], prefix[state, sample])
        for a in range(len(affected)):
            state = affected[a]
            highest[:] = -np.inf
            for local in range(count):
                for sample in range(samples):
                    highest[sample] = max(highest[sample], rows[local, a, sample] + messages[begin + local, sample])
            for sample in range(samples):
                prefix[state, sample] = highest[sample]
                normalizer[sample] = max(normalizer[sample], highest[sample])
        for state in range(states):
            for sample in range(samples):
                prefix[state, sample] -= normalizer[sample]
        for sample in range(samples):
            scales[sample] += normalizer[sample]
    return (decoded, scales.sum(), predicted)


def coordinate(data, bases, sizes, bins, known, incumbent, choices, offsets, messages, suffixes, penalty, reverse, first, second):
    weights = np.ascontiguousarray(messages.T)
    suffix = np.ascontiguousarray(suffixes.transpose(0, 2, 1))
    result = coordinate_sweep(data, bases, sizes, bins, known, incumbent, choices, offsets, weights, suffix, penalty, reverse, first, second)
    messages[:] = weights.T
    return result
