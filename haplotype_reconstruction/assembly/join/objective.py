"""The join objective: one Viterbi recursion over unordered pairs of panel paths.

For a panel of K founder paths (``rows[f, j]`` = the local row of block j that path f copies) and
the binned emissions e of :class:`.emissions.EmissionLayout`, sample s scores

    V_s(P) = max over state sequences z_1 .. z_T of sum_t e_s(z_t, t) - penalty x #{t: z_t != z_(t-1)},

a state being an unordered pair {a, c} of panel paths (its emission at bin t of block j is the
layout's value for the local rows (rows[a, j], rows[c, j])). The recursion per bin is

    S_t(z) = max(S_(t-1)(z), max_z' S_(t-1)(z') - penalty) + e(z, t),   V_s = max_z S_T(z).

Unphased emissions and a uniform penalty make the ordered states (a, c) and (c, a) carry equal
scores at every bin, so the K(K + 1)/2 pairs a <= c (in the order (0, 0), (0, 1), .., (K-1, K-1))
carry the programme. Tie rules (they define the painting traceback): the switch source is the
first maximiser of S_(t-1) in pair order, and a state stays when stay >= switch.

Kernels: :func:`forward` (one sample's forward maxima, optionally with the vertex covers of each
bin's maximiser set that certification reads), :func:`score_samples` (V_s of a subset of samples;
the join score of a panel is its sum over all samples) and :func:`paint` (the traceback). Each
sample is computed by one thread; panel totals sum the per-sample vector in sample order
(:func:`column_sums` for matrices), so no result depends on the thread count. Maxima use
:func:`max_range` (exact; eight independent chains let comparisons overlap); no kernel is compiled
with fast-math (B10).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ...maths import ties


@njit(cache=True, inline="always")
def max_range(values, lo, hi, initial):
    """max(initial, values[lo:hi]) through eight independent running maxima (two below eight
    values). A maximum is exact, so the value equals one sequential scan's; the independent chains
    let comparisons overlap."""
    if hi - lo < 8:
        m0 = initial
        m1 = -np.inf
        z = lo
        while z + 2 <= hi:
            v = values[z]
            m0 = v if v > m0 else m0
            v = values[z + 1]
            m1 = v if v > m1 else m1
            z += 2
        if z < hi:
            v = values[z]
            m0 = v if v > m0 else m0
        return m1 if m1 > m0 else m0
    m0 = initial
    m1 = m2 = m3 = m4 = m5 = m6 = m7 = -np.inf
    z = lo
    while z + 8 <= hi:
        v = values[z]
        m0 = v if v > m0 else m0
        v = values[z + 1]
        m1 = v if v > m1 else m1
        v = values[z + 2]
        m2 = v if v > m2 else m2
        v = values[z + 3]
        m3 = v if v > m3 else m3
        v = values[z + 4]
        m4 = v if v > m4 else m4
        v = values[z + 5]
        m5 = v if v > m5 else m5
        v = values[z + 6]
        m6 = v if v > m6 else m6
        v = values[z + 7]
        m7 = v if v > m7 else m7
        z += 8
    while z < hi:
        v = values[z]
        m0 = v if v > m0 else m0
        z += 1
    m0 = m1 if m1 > m0 else m0
    m2 = m3 if m3 > m2 else m2
    m4 = m5 if m5 > m4 else m4
    m6 = m7 if m7 > m6 else m6
    m0 = m2 if m2 > m0 else m0
    m4 = m6 if m6 > m4 else m4
    return m4 if m4 > m0 else m0


@njit(cache=True, inline="always")
def _first_maximiser(values, count):
    """(z, max) with z the first index < count whose value equals the maximum (the first
    maximiser): one scan for short vectors, else the maximum by max_range and a scan for it."""
    if count < 16:
        best = -np.inf
        index = 0
        for z in range(count):
            if values[z] > best:
                best = values[z]
                index = z
        return index, best
    best = max_range(values, 0, count, -np.inf)
    for z in range(count):
        if values[z] == best:
            return z, best
    return 0, best


@njit(cache=True)
def pair_lists(allowed):
    """The unordered pairs (a, b), a <= b, of the allowed paths in state order: (first, second)."""
    count = 0
    for a in range(len(allowed)):
        if allowed[a]:
            for b in range(a, len(allowed)):
                if allowed[b]:
                    count += 1
    first = np.empty(count, np.int64)
    second = np.empty(count, np.int64)
    index = 0
    for a in range(len(allowed)):
        if allowed[a]:
            for b in range(a, len(allowed)):
                if allowed[b]:
                    first[index] = a
                    second[index] = b
                    index += 1
    return first, second


@njit(cache=True)
def widest_block(block_bins):
    """The largest bin count of a block (at least 1): the size of a per-block gather buffer."""
    widest = 1
    for block in range(len(block_bins)):
        widest = max(widest, block_bins[block])
    return widest


@njit(cache=True)
def record_covers(scores, best, first, second, singles, pairs, record_pairs):
    """Vertex covers of size <= 2 of this bin's maximiser set.

    singles[f]: f lies in every maximising state. pairs[f, j] (f < j): {f, j} hits every
    maximising state while neither is a single cover of this bin.
    """
    z0 = -1
    i0 = -1
    i1 = -1
    count = 0
    for z in range(len(scores)):
        if scores[z] == best:
            a = first[z]
            b = second[z]
            if z0 < 0:
                z0 = z
                i0 = a
                i1 = b if b != a else -1
            else:
                if i0 >= 0 and i0 != a and i0 != b:
                    i0 = -1
                if i1 >= 0 and i1 != a and i1 != b:
                    i1 = -1
            count += 1
    if i0 >= 0:
        singles[i0] = True
    if i1 >= 0:
        singles[i1] = True
    if not record_pairs or count < 2:
        return
    # A size-2 cover hits the first maximiser: x in it, and its partner lies
    # in every maximiser that avoids x.
    for side in range(2):
        x = first[z0] if side == 0 else second[z0]
        if side == 1 and x == first[z0]:
            break
        if x == i0 or x == i1:
            continue
        w0 = -1
        w1 = -1
        started = False
        for z in range(len(scores)):
            if scores[z] == best:
                a = first[z]
                b = second[z]
                if a == x or b == x:
                    continue
                if not started:
                    w0 = a
                    w1 = b if b != a else -1
                    started = True
                else:
                    if w0 >= 0 and w0 != a and w0 != b:
                        w0 = -1
                    if w1 >= 0 and w1 != a and w1 != b:
                        w1 = -1
        if not started:
            continue
        if w0 >= 0 and w0 != x and w0 != i0 and w0 != i1:
            pairs[min(x, w0), max(x, w0)] = True
        if w1 >= 0 and w1 != x and w1 != i0 and w1 != i1:
            pairs[min(x, w1), max(x, w1)] = True


@njit(cache=True, inline="always")
def _gather_states(flat, base, haps, bins, rows, first, second, block, gathered):
    """gathered[t, z]: the emissions of states z (pairs first[z] <= second[z] of panel paths) at
    every bin t of one block, contiguous per bin (base: the block's first value of the sample)."""
    for z in range(len(first)):
        at = base + (rows[first[z], block] * haps + rows[second[z], block]) * bins
        for marker in range(bins):
            gathered[marker, z] = flat[at + marker]


@njit(cache=True, inline="always")
def _start(scores, n, gathered, marker):
    """S_0(z) = e(z, 0) for the n states; returns max_z S_0(z)."""
    for z in range(n):
        scores[z] = gathered[marker, z]
    return max_range(scores, 0, n, -np.inf)


@njit(cache=True, inline="always")
def _advance(scores, n, gathered, marker, switch_base):
    """The recursion's step: S_t(z) = max(S_(t-1)(z), switch_base) + e(z, t) for the n states
    (switch_base = max_z' S_(t-1)(z') - penalty); returns max_z S_t(z)."""
    for z in range(n):
        stay = scores[z]
        scores[z] = (stay if stay > switch_base else switch_base) + gathered[marker, z]
    return max_range(scores, 0, n, -np.inf)


@njit(cache=True)
def forward(flat, block_start, block_haps, block_bins, rows, first, second, penalty, sample, out,
            singles, pairs, record, record_pairs, gathered):
    """Forward maxima out[t] = max_z S_t(z) of one sample over the given unordered states.

    rows: int64[K, n] local rows of the panel paths; first, second: the states (pair_lists);
    gathered: a (widest block bins, states) buffer. With ``record``, the single covers (and with
    ``record_pairs`` the disjoint pair covers) of every bin's maximiser set are marked in singles
    and pairs (record_covers). out[T - 1] = V_s.
    """
    n = len(first)
    scores = np.empty(n)
    best = -np.inf
    t = 0
    for block in range(len(block_start)):
        haps, bins = block_haps[block], block_bins[block]
        _gather_states(flat, block_start[block] + sample * haps * haps * bins, haps, bins, rows,
                       first, second, block, gathered)
        for marker in range(bins):
            if t == 0:
                best = _start(scores, n, gathered, marker)
            else:
                # best: the previous bin's maximum (the scores have not changed since)
                best = _advance(scores, n, gathered, marker, best - penalty)
            out[t] = best
            if record:
                record_covers(scores, best, first, second, singles, pairs, record_pairs)
            t += 1


@njit(cache=True, parallel=True, nogil=True)
def score_samples(flat, block_start, block_haps, block_bins, rows, penalty, sample_ids):
    """V_s of the panel ``rows`` (int64[K, n]) for the samples ``sample_ids`` (float64, in that
    order; emissions: an EmissionLayout's arrays)."""
    k = rows.shape[0]
    first, second = pair_lists(np.ones(k, np.bool_))
    n = len(first)
    widest = widest_block(block_bins)
    result = np.empty(len(sample_ids), dtype=np.float64)
    for position in prange(len(sample_ids)):
        sample = sample_ids[position]
        scores = np.empty(n)
        gathered = np.empty((widest, n))
        best = -np.inf
        t = 0
        for block in range(len(block_start)):
            haps, bins = block_haps[block], block_bins[block]
            _gather_states(flat, block_start[block] + sample * haps * haps * bins, haps, bins,
                           rows, first, second, block, gathered)
            for marker in range(bins):
                if t == 0:
                    best = _start(scores, n, gathered, marker)
                else:
                    best = _advance(scores, n, gathered, marker, best - penalty)
                t += 1
        result[position] = best
    return result


@njit(cache=True, parallel=True, nogil=True)
def paint(flat, block_start, block_haps, block_bins, rows, penalty, start, stop, relative, reverse=False):
    """Viterbi paintings of samples start .. stop - 1: per bin the state a * K + c (a <= c) of the
    maximising sequence (the traceback of :func:`forward`'s recursion; same values). Tie rules within
    the band b = relative x max(1, |the bin's best score|) (release 2.1; relative 0 gives base's exact
    rules): the switch source is the first state in pair order within b of the maximum, a state stays
    when stay >= switch - b, the last bin's state is the first within b of the maximum. The recursion's
    values stay exact maxima; a traceback choice inside the band costs the painted sequence at most b,
    and near-equal sequences (equal scores reached through different summation orders, which a tiny
    input perturbation reorders) are painted by the content rule (stay, then the lower pair, the panel
    being in canonical order; under the convention reversal ``reverse`` the last state in pair order
    instead of the first, "stay" being a stated prior). int32[stop - start, total bins]."""
    k = rows.shape[0]
    first, second = pair_lists(np.ones(k, np.bool_))
    pairs = len(first)
    code = np.empty(pairs, np.int32)
    for z in range(pairs):
        code[z] = first[z] * k + second[z]
    total = 0
    for block in range(len(block_bins)):
        total += block_bins[block]
    widest = widest_block(block_bins)
    sample_paths = np.zeros((stop - start, total), dtype=np.int32)
    for position in prange(stop - start):
        sample = start + position
        current = np.empty(pairs, dtype=np.float64)
        backptrs = np.zeros((total, pairs), dtype=np.int32)
        gathered = np.empty((widest, pairs))
        t = 0
        for block in range(len(block_start)):
            haps, bins = block_haps[block], block_bins[block]
            _gather_states(flat, block_start[block] + sample * haps * haps * bins, haps, bins, rows,
                           first, second, block, gathered)
            for marker in range(bins):
                if t == 0:
                    for z in range(pairs):
                        current[z] = gathered[marker, z]
                else:
                    best = max_range(current, 0, pairs, -np.inf)
                    band = relative * max(1.0, abs(best))
                    source = 0
                    for y in range(pairs):
                        z = pairs - 1 - y if reverse else y
                        if current[z] >= best - band:
                            source = z
                            break
                    switch_base = best - penalty
                    for z in range(pairs):
                        emission = gathered[marker, z]
                        stay = current[z] + emission
                        switch = switch_base + emission
                        if stay >= switch - band:
                            backptrs[t, z] = z
                        else:
                            backptrs[t, z] = source
                        current[z] = stay if stay >= switch else switch
                t += 1
        best = max_range(current, 0, pairs, -np.inf)
        band = relative * max(1.0, abs(best))
        state = 0
        for y in range(pairs):
            z = pairs - 1 - y if reverse else y
            if current[z] >= best - band:
                state = z
                break
        sample_paths[position, total - 1] = code[state]
        for t in range(total - 1, 0, -1):
            state = backptrs[t, state]
            sample_paths[position, t - 1] = code[state]
    return sample_paths


@njit(cache=True, parallel=True, nogil=True)
def column_sums(values):
    """Sum each column of a (samples, columns) matrix in sample order (one thread per column)."""
    samples, count = values.shape
    result = np.empty(count)
    for column in prange(count):
        total = 0.0
        for sample in range(samples):
            total += values[sample, column]
        result[column] = total
    return result


def panel_scores(layout, rows, penalty, samples=None) -> np.ndarray:
    """Per-sample join scores V_s (float64) of the panel ``rows`` (int[K, n], local rows per
    block), for every sample of the layout or for the given sample indices."""
    sample_ids = (np.arange(layout.samples, dtype=np.int64) if samples is None
                  else np.ascontiguousarray(samples, dtype=np.int64))
    return score_samples(*layout.arrays(), np.ascontiguousarray(rows, dtype=np.int64), float(penalty),
                         sample_ids)


def painting(layout, rows, penalty, tensor_budget_mb, claim, reverse=None) -> np.ndarray:
    """The Viterbi paintings int32[S, total bins] of the panel ``rows`` at ``penalty``, in sample
    chunks whose back pointers (int32, bins x states per sample) fit ``tensor_budget_mb``;
    ``claim()`` re-reads the thread budget before each chunk. Traceback ties within the band of
    :func:`paint` (tie_rules ASSEMBLY_QUANTA["paint"] relative)."""
    from ..tie_rules import ASSEMBLY_QUANTA
    relative = float(ASSEMBLY_QUANTA["paint"][1])
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    k = rows.shape[0]
    samples, bins = layout.samples, layout.total_bins
    bytes_per_sample = max(1, k * (k + 1) // 2 * bins * 4)
    chunk = max(1, min(samples, int(tensor_budget_mb * 1024 ** 2) // bytes_per_sample))
    result = np.empty((samples, bins), dtype=np.int32)
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        claim()
        result[start:stop] = paint(*layout.arrays(), rows, float(penalty), start, stop, relative,
                                   bool(ties.reverse_conventions() if reverse is None else reverse))
    return result


@njit(cache=True, parallel=True, nogil=True)
def minimum_switches(flat, block_start, block_haps, block_bins, rows, penalty, samples):
    """Exact minimum state changes among score-optimal paths, per sample.

    The primary recurrence is the ordinary join score, without a traceback band. Counts
    break exact score ties only; the stored representative prior never changes its LL.
    """
    first, second = pair_lists(np.ones(len(rows), np.bool_))
    result = np.zeros(samples, np.int64)
    for sample in prange(samples):
        values = np.zeros(len(first))
        counts = np.zeros(len(first), np.int64)
        started = False
        for block in range(len(block_start)):
            haps, bins = block_haps[block], block_bins[block]
            base = block_start[block] + sample * haps * haps * bins
            for marker in range(bins):
                best = -np.inf
                fewest = np.iinfo(np.int64).max
                for state in range(len(first)):
                    if values[state] > best:
                        best, fewest = values[state], counts[state]
                    elif values[state] == best:
                        fewest = min(fewest, counts[state])
                for state in range(len(first)):
                    stay = values[state]
                    switch = best - penalty
                    if started and (switch > stay or (switch == stay and fewest + 1 < counts[state])):
                        values[state] = switch
                        counts[state] = fewest + 1
                    at = base + (rows[first[state], block] * haps + rows[second[state], block]) * bins
                    values[state] += flat[at + marker]
                started = True
        best = np.max(values)
        minimum = np.iinfo(np.int64).max
        for state in range(len(first)):
            if values[state] == best:
                minimum = min(minimum, counts[state])
        result[sample] = minimum
    return result
