"""Exact block-boundary messages for converged founder sweeps on the canonical F.

F(panel) = sum over samples of the full-site Potts Viterbi score (founder/scoring.py):
unordered founder-pair states, informative complete sites only, SWITCH_PENALTY per
state change. For every sample the score is a max-plus chain, so for ANY panel edit
confined to blocks [a, c) the exact change of F is

    dF = sum_s [max_q (f_s(a) (x) T_new[a, c) (x) g_s(c))_q - max_q (f_s(a) (x) T_old[a, c) (x) g_s(c))_q]

with f the carried forward message (switch-then-emission per informative site) and g
the stored backward message (emission-then-switch), both normalised to max 0 at every
block boundary; normalisers cancel. A suffix exchange of founders a, b relabels the
suffix states only, so its gain needs the two messages at its junction alone.

The optional switch counts form a lexicographic max-plus pair (score first, then fewer
switches among exact score ties). They give the genotype-fit term E = F + penalty x
switches of a maximum-score path, decomposed exactly like F; only the converged
sweep's coarse-move guard reads them (and traces those paths back to locate their
switches). They never change F.

Memory layout of the recursions: a state's pair dosage haps[first[q], j] +
haps[second[q], j] is the same for every sample, so the message kernels read it from a
site-major int8 table built once per call (shared by all samples; the candidate kernels
build one per candidate, with its moved founder's new rows) and read a site's three
emissions once per sample; the chromosome-wide kernels and the candidate kernels build
the table per segment of whole blocks of at most DOSAGE_BYTES, so it stays small at
large K. Per-site chain leaders are stored sample-major, the order the per-sample loops
write and read them. The arithmetic is unchanged operation for operation (bitwise the
same values). The chromosome-wide message kernels compute the samples of a range [s0, s1)
into arrays the caller allocates: every sample's recursion is its own, so a search can
run them range by range, re-reading its thread share between ranges.
"""
import math

import numpy as np
from numba import njit, prange

from .scoring import _unordered_pairs

CENTER = math.log(1.0 / 3.0)
# Largest pair-dosage table (sites x states int8) of one segment of a chromosome-wide
# message kernel or of a candidate's blocks; a block larger than this forms its own segment.
DOSAGE_BYTES = 32 * 1024 ** 2


@njit(cache=True, nogil=True)
def swap_permutations(founders):
    """perm[p, q] = state of q after swapping the founders of pair p (a < b, triu order)."""
    first, second = _unordered_pairs(founders)
    index = np.empty((founders, founders), np.int64)
    for q in range(len(first)):
        index[first[q], second[q]] = q
        index[second[q], first[q]] = q
    pairs = founders * (founders - 1) // 2
    perm = np.empty((pairs, len(first)), np.int64)
    pa = np.empty(pairs, np.int64)
    pb = np.empty(pairs, np.int64)
    p = 0
    for a in range(founders):
        for b in range(a + 1, founders):
            pa[p] = a
            pb[p] = b
            for q in range(len(first)):
                i, j = first[q], second[q]
                i2 = b if i == a else (a if i == b else i)
                j2 = b if j == a else (a if j == b else j)
                perm[p, q] = index[i2, j2]
            p += 1
    return perm, pa, pb


@njit(inline="always")
def _emission(logs, s, j, d):
    return logs[s, j, d] - CENTER


@njit(inline="always")
def _site_emissions(logs, s, j):
    """The centred emissions of dosages 0, 1, 2 of sample s at site j."""
    return logs[s, j, 0] - CENTER, logs[s, j, 1] - CENTER, logs[s, j, 2] - CENTER


@njit(inline="always")
def _pick(d, e0, e1, e2):
    return e0 if d == 0 else (e1 if d == 1 else e2)


@njit(cache=True, parallel=True, nogil=True)
def _pair_dosages(haps, first, second, lo, hi):
    """Site-major pair dosages of sites [lo, hi): out[j - lo, q] = haps[first[q], j] +
    haps[second[q], j]."""
    states = len(first)
    out = np.empty((hi - lo, states), np.int8)
    for j in prange(lo, hi):
        for q in range(states):
            out[j - lo, q] = haps[first[q], j] + haps[second[q], j]
    return out


@njit(cache=True, nogil=True)
def _segments(offsets, states):
    """Boundaries 0 = t_0 < ... < t_n = B of consecutive segments of whole blocks whose
    pair-dosage tables fit DOSAGE_BYTES (a larger block is a segment of its own)."""
    blocks = len(offsets) - 1
    limit = max(1, DOSAGE_BYTES // max(1, states))
    bounds = np.empty(blocks + 2, np.int64)
    bounds[0] = 0
    n = 0
    for b in range(1, blocks):
        if offsets[b + 1] - offsets[bounds[n]] > limit:
            n += 1
            bounds[n] = b
    if blocks > 0:
        n += 1
        bounds[n] = blocks
    return bounds[:n + 1]


@njit(cache=True, parallel=True, nogil=True)
def messages(haps, logs, offsets, penalty, s0, s1, fwd, bwd, fscale, bscale):
    """Normalised forward (carried) and backward (stored) messages at every block boundary,
    of the samples [s0, s1).

    Written into fwd, bwd with shape (B + 1, S, Q) and the per-block normalisers fscale,
    bscale with shape (B, S), all zero on entry (allocate them with message_arrays):
    fwd[b + 1] = (fwd[b] processed over block b) - fscale[b] and
    bwd[b] = (bwd[b + 1] processed backwards over block b) - bscale[b], so the
    unnormalised backward message at x is bwd[x] + sum_{b >= x} bscale[b].
    """
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    bounds = _segments(offsets, states)
    for t in range(len(bounds) - 1):
        b0, b1 = bounds[t], bounds[t + 1]
        lo = offsets[b0]
        dos = _pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = fwd[b0, s].copy()
            for b in range(b0, b1):
                for j in range(offsets[b], offsets[b + 1]):
                    if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                        continue
                    e0, e1, e2 = _site_emissions(logs, s, j)
                    sw = np.max(row) - penalty
                    for q in range(states):
                        row[q] = max(row[q], sw) + _pick(dos[j - lo, q], e0, e1, e2)
                m = np.max(row)
                fscale[b, s] = m
                for q in range(states):
                    row[q] -= m
                    fwd[b + 1, s, q] = row[q]
    for t in range(len(bounds) - 2, -1, -1):
        b0, b1 = bounds[t], bounds[t + 1]
        lo = offsets[b0]
        dos = _pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = bwd[b1, s].copy()
            for b in range(b1 - 1, b0 - 1, -1):
                for j in range(offsets[b + 1] - 1, offsets[b] - 1, -1):
                    if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                        continue
                    e0, e1, e2 = _site_emissions(logs, s, j)
                    for q in range(states):
                        row[q] += _pick(dos[j - lo, q], e0, e1, e2)
                    sw = np.max(row) - penalty
                    for q in range(states):
                        row[q] = max(row[q], sw)
                m = np.max(row)
                bscale[b, s] = m
                for q in range(states):
                    row[q] -= m
                    bwd[b, s, q] = row[q]


def message_arrays(founders, blocks, samples, lex):
    """Zeroed message arrays of messages() (lex: messages_lex(), with switch counts):
    (fwd, bwd, fcnt, bcnt, fscale, bscale); the counts are None without lex."""
    states = founders * (founders + 1) // 2
    counts = ((np.zeros((blocks + 1, samples, states), np.int64),
               np.zeros((blocks + 1, samples, states), np.int64)) if lex else (None, None))
    return (np.zeros((blocks + 1, samples, states)), np.zeros((blocks + 1, samples, states)),
            *counts, np.zeros((blocks, samples)), np.zeros((blocks, samples)))


@njit(cache=True, parallel=True, nogil=True)
def reverse_sites(logs):
    """A C-contiguous copy of logs (samples x sites x 3) with the site axis reversed (the
    reversed orientation of a search), copied in parallel over samples."""
    samples, sites, genotypes = logs.shape
    out = np.empty_like(logs)
    for s in prange(samples):
        for j in range(sites):
            for g in range(genotypes):
                out[s, sites - 1 - j, g] = logs[s, j, g]
    return out


@njit(inline="always")
def _lex_best(row, cnt):
    """Maximum score and the fewest switches among exact maxima."""
    m = row[0]
    c = cnt[0]
    for q in range(1, len(row)):
        if row[q] > m or (row[q] == m and cnt[q] < c):
            m = row[q]
            c = cnt[q]
    return m, c


@njit(cache=True, parallel=True, nogil=True)
def messages_lex(haps, logs, offsets, penalty, s0, s1, fwd, bwd, fcnt, bcnt, fscale, bscale):
    """messages() plus the switch counts of the lexicographic (score, -switches) maxima, of
    the samples [s0, s1), into zeroed arrays (message_arrays(..., lex=True))."""
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    bounds = _segments(offsets, states)
    for t in range(len(bounds) - 1):
        b0, b1 = bounds[t], bounds[t + 1]
        lo = offsets[b0]
        dos = _pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = fwd[b0, s].copy()
            cnt = fcnt[b0, s].copy()
            for b in range(b0, b1):
                for j in range(offsets[b], offsets[b + 1]):
                    if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                        continue
                    e0, e1, e2 = _site_emissions(logs, s, j)
                    m, c = _lex_best(row, cnt)
                    sw = m - penalty
                    for q in range(states):
                        if sw > row[q] or (sw == row[q] and c + 1 < cnt[q]):
                            row[q] = sw
                            cnt[q] = c + 1
                        row[q] += _pick(dos[j - lo, q], e0, e1, e2)
                m = np.max(row)
                fscale[b, s] = m
                for q in range(states):
                    row[q] -= m
                    fwd[b + 1, s, q] = row[q]
                    fcnt[b + 1, s, q] = cnt[q]
    for t in range(len(bounds) - 2, -1, -1):
        b0, b1 = bounds[t], bounds[t + 1]
        lo = offsets[b0]
        dos = _pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = bwd[b1, s].copy()
            cnt = bcnt[b1, s].copy()
            for b in range(b1 - 1, b0 - 1, -1):
                for j in range(offsets[b + 1] - 1, offsets[b] - 1, -1):
                    if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                        continue
                    e0, e1, e2 = _site_emissions(logs, s, j)
                    for q in range(states):
                        row[q] += _pick(dos[j - lo, q], e0, e1, e2)
                    m, c = _lex_best(row, cnt)
                    sw = m - penalty
                    for q in range(states):
                        if sw > row[q] or (sw == row[q] and c + 1 < cnt[q]):
                            row[q] = sw
                            cnt[q] = c + 1
                m = np.max(row)
                bscale[b, s] = m
                for q in range(states):
                    row[q] -= m
                    bwd[b, s, q] = row[q]
                    bcnt[b, s, q] = cnt[q]


@njit(cache=True, parallel=True, nogil=True)
def update_messages(haps, logs, offsets, penalty, changed, fwd, bwd, fscale, bscale, backward_known,
                    s0, s1):
    """messages() of the current panel (haps) for the samples [s0, s1), in place on those of
    an earlier panel whose alleles differ only on the blocks where changed[b]: fwd, fscale
    always, bwd, bscale when backward_known (otherwise every block is computed into them;
    bwd[B] must hold zeros).

    A block's forward step reads only the message at its start and the block's alleles, so
    a sample's messages over an unchanged block whose start message is unchanged are the
    earlier ones. Per sample, a block is recomputed (messages()' recursion, operation for
    operation, from the stored start message) only if it changed or its start message did;
    a recomputed end message that equals the earlier one ends the run. Likewise backwards.
    Normalised messages hold no negative zero (x - x = +0), so equal values are equal bits.
    Bitwise messages() of the current panel.
    """
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    bounds = _segments(offsets, states)
    # dirty[s - s0]: sample s's message at the current segment boundary changed.
    dirty = np.zeros(s1 - s0, np.bool_)
    for t in range(len(bounds) - 1):
        b0, b1 = bounds[t], bounds[t + 1]
        if not (dirty.any() or changed[b0:b1].any()):
            continue
        lo = offsets[b0]
        dos = _pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = np.empty(states)
            for b in range(b0, b1):
                if not (dirty[s - s0] or changed[b]):
                    continue
                row[:] = fwd[b, s]
                for j in range(offsets[b], offsets[b + 1]):
                    if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                        continue
                    e0, e1, e2 = _site_emissions(logs, s, j)
                    sw = np.max(row) - penalty
                    for q in range(states):
                        row[q] = max(row[q], sw) + _pick(dos[j - lo, q], e0, e1, e2)
                m = np.max(row)
                fscale[b, s] = m
                same = True
                for q in range(states):
                    row[q] -= m
                    same = same and row[q] == fwd[b + 1, s, q]
                    fwd[b + 1, s, q] = row[q]
                dirty[s - s0] = not same
    dirty[:] = not backward_known
    for t in range(len(bounds) - 2, -1, -1):
        b0, b1 = bounds[t], bounds[t + 1]
        if not (dirty.any() or changed[b0:b1].any()):
            continue
        lo = offsets[b0]
        dos = _pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = np.empty(states)
            for b in range(b1 - 1, b0 - 1, -1):
                if not (dirty[s - s0] or changed[b]):
                    continue
                row[:] = bwd[b + 1, s]
                for j in range(offsets[b + 1] - 1, offsets[b] - 1, -1):
                    if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                        continue
                    e0, e1, e2 = _site_emissions(logs, s, j)
                    for q in range(states):
                        row[q] += _pick(dos[j - lo, q], e0, e1, e2)
                    sw = np.max(row) - penalty
                    for q in range(states):
                        row[q] = max(row[q], sw)
                m = np.max(row)
                bscale[b, s] = m
                same = backward_known
                for q in range(states):
                    row[q] -= m
                    same = same and row[q] == bwd[b, s, q]
                    bwd[b, s, q] = row[q]
                dirty[s - s0] = not same


@njit(cache=True, parallel=True, nogil=True)
def update_messages_lex(haps, logs, offsets, penalty, changed, fwd, bwd, fcnt, bcnt, fscale, bscale,
                        backward_known, s0, s1):
    """update_messages() for messages_lex(): the messages and their switch counts (a run ends
    where both equal the earlier ones), samples [s0, s1). Bitwise messages_lex() of the
    current panel."""
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    bounds = _segments(offsets, states)
    dirty = np.zeros(s1 - s0, np.bool_)
    for t in range(len(bounds) - 1):
        b0, b1 = bounds[t], bounds[t + 1]
        if not (dirty.any() or changed[b0:b1].any()):
            continue
        lo = offsets[b0]
        dos = _pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = np.empty(states)
            cnt = np.empty(states, np.int64)
            for b in range(b0, b1):
                if not (dirty[s - s0] or changed[b]):
                    continue
                row[:] = fwd[b, s]
                cnt[:] = fcnt[b, s]
                for j in range(offsets[b], offsets[b + 1]):
                    if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                        continue
                    e0, e1, e2 = _site_emissions(logs, s, j)
                    m, c = _lex_best(row, cnt)
                    sw = m - penalty
                    for q in range(states):
                        if sw > row[q] or (sw == row[q] and c + 1 < cnt[q]):
                            row[q] = sw
                            cnt[q] = c + 1
                        row[q] += _pick(dos[j - lo, q], e0, e1, e2)
                m = np.max(row)
                fscale[b, s] = m
                same = True
                for q in range(states):
                    row[q] -= m
                    same = same and row[q] == fwd[b + 1, s, q] and cnt[q] == fcnt[b + 1, s, q]
                    fwd[b + 1, s, q] = row[q]
                    fcnt[b + 1, s, q] = cnt[q]
                dirty[s - s0] = not same
    dirty[:] = not backward_known
    for t in range(len(bounds) - 2, -1, -1):
        b0, b1 = bounds[t], bounds[t + 1]
        if not (dirty.any() or changed[b0:b1].any()):
            continue
        lo = offsets[b0]
        dos = _pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = np.empty(states)
            cnt = np.empty(states, np.int64)
            for b in range(b1 - 1, b0 - 1, -1):
                if not (dirty[s - s0] or changed[b]):
                    continue
                row[:] = bwd[b + 1, s]
                cnt[:] = bcnt[b + 1, s]
                for j in range(offsets[b + 1] - 1, offsets[b] - 1, -1):
                    if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                        continue
                    e0, e1, e2 = _site_emissions(logs, s, j)
                    for q in range(states):
                        row[q] += _pick(dos[j - lo, q], e0, e1, e2)
                    m, c = _lex_best(row, cnt)
                    sw = m - penalty
                    for q in range(states):
                        if sw > row[q] or (sw == row[q] and c + 1 < cnt[q]):
                            row[q] = sw
                            cnt[q] = c + 1
                m = np.max(row)
                bscale[b, s] = m
                same = backward_known
                for q in range(states):
                    row[q] -= m
                    same = same and row[q] == bwd[b, s, q] and cnt[q] == bcnt[b, s, q]
                    bwd[b, s, q] = row[q]
                    bcnt[b, s, q] = cnt[q]
                dirty[s - s0] = not same


@njit(cache=True, parallel=True, nogil=True)
def advance(haps, logs, offsets, penalty, start, a, c):
    """Carry normalised forward messages from boundary a over blocks [a, c)."""
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    out = np.empty((samples, states))
    lo = offsets[a]
    dos = _pair_dosages(haps, first, second, lo, offsets[c])
    for s in prange(samples):
        row = start[s].copy()
        for b in range(a, c):
            for j in range(offsets[b], offsets[b + 1]):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                e0, e1, e2 = _site_emissions(logs, s, j)
                sw = np.max(row) - penalty
                for q in range(states):
                    row[q] = max(row[q], sw) + _pick(dos[j - lo, q], e0, e1, e2)
            m = np.max(row)
            for q in range(states):
                row[q] -= m
        out[s] = row
    return out


@njit(cache=True, parallel=True, nogil=True)
def advance_lex(haps, logs, offsets, penalty, start, start_cnt, a, c):
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    out = np.empty((samples, states))
    out_cnt = np.empty((samples, states), np.int64)
    lo = offsets[a]
    dos = _pair_dosages(haps, first, second, lo, offsets[c])
    for s in prange(samples):
        row = start[s].copy()
        cnt = start_cnt[s].copy()
        for b in range(a, c):
            for j in range(offsets[b], offsets[b + 1]):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                e0, e1, e2 = _site_emissions(logs, s, j)
                m, k = _lex_best(row, cnt)
                sw = m - penalty
                for q in range(states):
                    if sw > row[q] or (sw == row[q] and k + 1 < cnt[q]):
                        row[q] = sw
                        cnt[q] = k + 1
                    row[q] += _pick(dos[j - lo, q], e0, e1, e2)
            m = np.max(row)
            for q in range(states):
                row[q] -= m
        out[s] = row
        out_cnt[s] = cnt
    return out, out_cnt


@njit(cache=True, parallel=True, nogil=True)
def backward_chain(haps, logs, offsets, penalty, end, gperm, a, c):
    """Exact stored-convention backward messages at boundaries a..c (index 0..c-a).

    end is the stored message at c, read through the label permutation gperm. Also
    returns the per-block normalisers (index b - a for blocks a..c-1).
    """
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    out = np.empty((c - a + 1, samples, states))
    scale = np.zeros((max(c - a, 0), samples))
    lo = offsets[a]
    dos = _pair_dosages(haps, first, second, lo, offsets[c])
    for s in prange(samples):
        row = np.empty(states)
        for q in range(states):
            row[q] = end[s, gperm[q]]
        out[c - a, s] = row
        for b in range(c - 1, a - 1, -1):
            for j in range(offsets[b + 1] - 1, offsets[b] - 1, -1):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                e0, e1, e2 = _site_emissions(logs, s, j)
                for q in range(states):
                    row[q] += _pick(dos[j - lo, q], e0, e1, e2)
                sw = np.max(row) - penalty
                for q in range(states):
                    row[q] = max(row[q], sw)
            m = np.max(row)
            scale[b - a, s] = m
            for q in range(states):
                row[q] -= m
            out[b - a, s] = row
    return out, scale


@njit(cache=True, parallel=True, nogil=True)
def backward_chain_lex(haps, logs, offsets, penalty, end, end_cnt, gperm, a, c):
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    out = np.empty((c - a + 1, samples, states))
    out_cnt = np.empty((c - a + 1, samples, states), np.int64)
    scale = np.zeros((max(c - a, 0), samples))
    lo = offsets[a]
    dos = _pair_dosages(haps, first, second, lo, offsets[c])
    for s in prange(samples):
        row = np.empty(states)
        cnt = np.empty(states, np.int64)
        for q in range(states):
            row[q] = end[s, gperm[q]]
            cnt[q] = end_cnt[s, gperm[q]]
        out[c - a, s] = row
        out_cnt[c - a, s] = cnt
        for b in range(c - 1, a - 1, -1):
            for j in range(offsets[b + 1] - 1, offsets[b] - 1, -1):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                e0, e1, e2 = _site_emissions(logs, s, j)
                for q in range(states):
                    row[q] += _pick(dos[j - lo, q], e0, e1, e2)
                m, k = _lex_best(row, cnt)
                sw = m - penalty
                for q in range(states):
                    if sw > row[q] or (sw == row[q] and k + 1 < cnt[q]):
                        row[q] = sw
                        cnt[q] = k + 1
            m = np.max(row)
            scale[b - a, s] = m
            for q in range(states):
                row[q] -= m
            out[b - a, s] = row
            out_cnt[b - a, s] = cnt
    return out, out_cnt, scale


@njit(cache=True, parallel=True, nogil=True)
def window_values(haps, logs, offsets, penalty, f, g, a, c):
    """Per-sample max_q [f (x) interior over [a, c) (x) g] with the interior normalisers.

    Reference evaluation of a whole window (validation path of candidate_gains).
    """
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    vals = np.empty(samples)
    for s in prange(samples):
        row = f[s].copy()
        scale = 0.0
        for b in range(a, c):
            for j in range(offsets[b], offsets[b + 1]):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                sw = np.max(row) - penalty
                for q in range(states):
                    row[q] = max(row[q], sw) + _emission(
                        logs, s, j, haps[first[q], j] + haps[second[q], j])
            m = np.max(row)
            scale += m
            for q in range(states):
                row[q] -= m
        best = -np.inf
        for q in range(states):
            best = max(best, row[q] + g[s, q])
        vals[s] = best + scale
    return vals


@njit(cache=True, parallel=True, nogil=True)
def window_values_lex(haps, logs, offsets, penalty, f, fcnt, g, gcnt, a, c):
    """Per-sample (score, switch count) of the lexicographic maximum through [a, c)."""
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    vals = np.empty(samples)
    counts = np.empty(samples, np.int64)
    for s in prange(samples):
        row = f[s].copy()
        cnt = fcnt[s].copy()
        scale = 0.0
        for b in range(a, c):
            for j in range(offsets[b], offsets[b + 1]):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                m, k = _lex_best(row, cnt)
                sw = m - penalty
                for q in range(states):
                    if sw > row[q] or (sw == row[q] and k + 1 < cnt[q]):
                        row[q] = sw
                        cnt[q] = k + 1
                    row[q] += _emission(logs, s, j, haps[first[q], j] + haps[second[q], j])
            m = np.max(row)
            scale += m
            for q in range(states):
                row[q] -= m
        best = -np.inf
        best_cnt = 0
        for q in range(states):
            v = row[q] + g[s, q]
            k = cnt[q] + gcnt[s, q]
            if v > best or (v == best and k < best_cnt):
                best = v
                best_cnt = k
        vals[s] = best + scale
        counts[s] = best_cnt
    return vals, counts


@njit(inline="always")
def _lex_best_index(row, cnt):
    """_lex_best() plus the first state attaining it (the switch source of a path)."""
    m = row[0]
    c = cnt[0]
    best = 0
    for q in range(1, len(row)):
        if row[q] > m or (row[q] == m and cnt[q] < c):
            m = row[q]
            c = cnt[q]
            best = q
    return m, c, best


@njit(inline="always")
def _max_lead(row):
    """Maximum of row and the first state attaining it."""
    m = row[0]
    lead = 0
    for q in range(1, len(row)):
        if row[q] > m:
            m = row[q]
            lead = q
    return m, lead


@njit(cache=True, parallel=True, nogil=True)
def forward_chain(haps, logs, offsets, penalty, start, a, c):
    """Carried forward messages at boundaries a..c (index 0..c-a) from start at a.

    Also the chain's leaders (see candidate_gains): per site j of blocks [a, c) (sample
    major, index [s, j - offsets[a]]) the maximum of the message before the site's
    update and the first state attaining it (informative sites only), and per block the
    maximum before the block's normalisation and its first state.
    """
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    base = offsets[a]
    out = np.empty((c - a + 1, samples, states))
    site_max = np.empty((samples, offsets[c] - base))
    site_lead = np.empty((samples, offsets[c] - base), np.int32)
    block_max = np.empty((c - a, samples))
    block_lead = np.empty((c - a, samples), np.int32)
    dos = _pair_dosages(haps, first, second, base, offsets[c])
    for s in prange(samples):
        row = start[s].copy()
        out[0, s] = row
        for b in range(a, c):
            for j in range(offsets[b], offsets[b + 1]):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                e0, e1, e2 = _site_emissions(logs, s, j)
                top, lead = _max_lead(row)
                site_max[s, j - base] = top
                site_lead[s, j - base] = lead
                sw = top - penalty
                for q in range(states):
                    row[q] = max(row[q], sw) + _pick(dos[j - base, q], e0, e1, e2)
            m, lead = _max_lead(row)
            block_max[b - a, s] = m
            block_lead[b - a, s] = lead
            for q in range(states):
                row[q] -= m
            out[b - a + 1, s] = row
    return out, site_max, site_lead, block_max, block_lead


@njit(cache=True, parallel=True, nogil=True)
def forward_chain_lex(haps, logs, offsets, penalty, start, start_cnt, a, c):
    """forward_chain with the switch counts: the same messages and block leaders, and per
    site the lexicographic maximum (score, switch count) of the message before the site's
    update (site_max, site_cnt) and the first state attaining both (site_lead: a state
    attaining the maximum score, as candidate_gains requires)."""
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    base = offsets[a]
    out = np.empty((c - a + 1, samples, states))
    out_cnt = np.empty((c - a + 1, samples, states), np.int64)
    site_max = np.empty((samples, offsets[c] - base))
    site_lead = np.empty((samples, offsets[c] - base), np.int32)
    site_cnt = np.empty((samples, offsets[c] - base), np.int32)
    block_max = np.empty((c - a, samples))
    block_lead = np.empty((c - a, samples), np.int32)
    dos = _pair_dosages(haps, first, second, base, offsets[c])
    for s in prange(samples):
        row = start[s].copy()
        cnt = start_cnt[s].copy()
        out[0, s] = row
        out_cnt[0, s] = cnt
        for b in range(a, c):
            for j in range(offsets[b], offsets[b + 1]):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                e0, e1, e2 = _site_emissions(logs, s, j)
                m, k, lead = _lex_best_index(row, cnt)
                site_max[s, j - base] = m
                site_lead[s, j - base] = lead
                site_cnt[s, j - base] = k
                sw = m - penalty
                for q in range(states):
                    if sw > row[q] or (sw == row[q] and k + 1 < cnt[q]):
                        row[q] = sw
                        cnt[q] = k + 1
                    row[q] += _pick(dos[j - base, q], e0, e1, e2)
            m, lead = _max_lead(row)
            block_max[b - a, s] = m
            block_lead[b - a, s] = lead
            for q in range(states):
                row[q] -= m
            out[b - a + 1, s] = row
            out_cnt[b - a + 1, s] = cnt
    return out, out_cnt, site_max, site_lead, block_max, block_lead, site_cnt


@njit(cache=True, parallel=True, nogil=True)
def candidate_gains_reference(haps, leaves, logs, offsets, penalty, f_start, g_start, focal,
                              first_block, end_block, rows, row_offsets, g_end, scale_sum):
    """Validation reference of candidate_gains: every sample by the dense recursion."""
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    count = len(focal)
    gains = np.empty((count, samples))
    for task in prange(count * samples):
        i, s = task // samples, task % samples
        gains[i, s] = _candidate_dense(haps, leaves, logs, offsets, penalty, f_start, g_start, focal,
                                       first_block, end_block, rows, row_offsets, g_end, scale_sum,
                                       first, second, i, s)
    return gains


@njit(cache=True, nogil=True)
def _candidate_dense_blocks(haps, leaves, logs, offsets, penalty, row, k, rows, row_start, start, b0,
                            b1, s, scale, first, second):
    """The all-state recursion of a candidate (founder k's rows replaced by rows[row_start + b -
    start] on blocks b) for sample s over blocks [b0, b1), in place on row; returns scale plus
    the blocks' normalisers, added in block order."""
    states = len(first)
    for b in range(b0, b1):
        r = rows[row_start + b - start]
        leaf = leaves[b]
        base = offsets[b]
        for j in range(offsets[b], offsets[b + 1]):
            if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                continue
            sw = np.max(row) - penalty
            for q in range(states):
                a = first[q]
                e = second[q]
                d = ((leaf[r, j - base] if a == k else haps[a, j])
                     + (leaf[r, j - base] if e == k else haps[e, j]))
                row[q] = max(row[q], sw) + _emission(logs, s, j, d)
        m = np.max(row)
        scale += m
        for q in range(states):
            row[q] -= m
    return scale


@njit(inline="always")
def _max_close(row, g):
    """max over the states of row + g."""
    best = -np.inf
    for q in range(len(row)):
        best = max(best, row[q] + g[q])
    return best


@njit(cache=True, nogil=True)
def _candidate_dense(haps, leaves, logs, offsets, penalty, f_start, g_start, focal, first_block,
                     end_block, rows, row_offsets, g_end, scale_sum, first, second, i, s):
    """One candidate's exact gain for one sample: every state over every site."""
    old = _max_close(f_start[i, s], g_start[i, s])
    row = f_start[i, s].copy()
    start = first_block[i]
    scale = _candidate_dense_blocks(haps, leaves, logs, offsets, penalty, row, focal[i], rows,
                                    row_offsets[i], start, start, end_block[i], s, 0.0, first,
                                    second)
    return _max_close(row, g_end[i, s]) + scale - old - scale_sum[i, s]


@njit(cache=True, nogil=True)
def _focal_states(first, second, founders):
    """focal_states[k]: the K states containing founder k (in state order)."""
    focal_states = np.empty((founders, founders), np.int64)
    filled = np.zeros(founders, np.int64)
    for q in range(len(first)):
        a, b = first[q], second[q]
        focal_states[a, filled[a]] = q
        filled[a] += 1
        if b != a:
            focal_states[b, filled[b]] = q
            filled[b] += 1
    return focal_states


@njit(cache=True, parallel=True, nogil=True)
def _item_dosages(haps, leaves, offsets, k, rows, row_start, b0, b1, first, second):
    """Site-major pair dosages of one candidate on blocks [b0, b1): founder k's rows replaced
    by rows[row_start + b - b0] on block b, the other founders' current alleles (haps);
    out[j - offsets[b0], q] = allele of first[q] + allele of second[q] at site j."""
    states = len(first)
    lo = offsets[b0]
    hi = offsets[b1]
    allele = np.empty(hi - lo, np.int8)
    for b in range(b0, b1):
        r = rows[row_start + b - b0]
        leaf = leaves[b]
        base = offsets[b]
        for j in range(base, offsets[b + 1]):
            allele[j - lo] = leaf[r, j - base]
    out = np.empty((hi - lo, states), np.int8)
    for j in prange(lo, hi):
        for q in range(states):
            a = first[q]
            e = second[q]
            out[j - lo, q] = ((allele[j - lo] if a == k else haps[a, j])
                              + (allele[j - lo] if e == k else haps[e, j]))
    return out


@njit(cache=True, nogil=True)
def _dense_tab(dos, lo, logs, offsets, penalty, row, b0, b1, s, scale):
    """_candidate_dense_blocks on a candidate's dosage table (dos[j - lo, q], _item_dosages):
    the all-state recursion of sample s over blocks [b0, b1), in place on row; returns scale
    plus the blocks' normalisers, added in block order (the same arithmetic)."""
    states = len(row)
    for b in range(b0, b1):
        for j in range(offsets[b], offsets[b + 1]):
            if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                continue
            e0, e1, e2 = _site_emissions(logs, s, j)
            sw = np.max(row) - penalty
            for q in range(states):
                row[q] = max(row[q], sw) + _pick(dos[j - lo, q], e0, e1, e2)
        m = np.max(row)
        scale += m
        for q in range(states):
            row[q] -= m
    return scale


@njit(cache=True, parallel=True, nogil=True)
def candidate_gains(haps, leaves, logs, offsets, penalty, f_start, g_start, focal, first_block,
                    end_block, rows, row_offsets, g_end, scale_sum, site_max, site_lead, block_max,
                    block_lead, chain_block, chain_values):
    """Exact per-sample F gains of one-founder content moves.

    Candidate i replaces the rows of founder focal[i] on blocks [first_block[i],
    end_block[i]) (rows[row_offsets[i]:row_offsets[i+1]]); the current panel (haps) is
    unchanged elsewhere. With f_start[i] / g_start[i] the current normalised forward /
    backward messages at first_block[i], g_end[i] the current backward message at
    end_block[i] and scale_sum[i] the backward normalisers of those blocks, the gain of
    sample s is max_q [f_start (x) T_new (x) g_end] + new normalisers - max_q [f_start +
    g_start] - scale_sum: the exact change of the sample's canonical Viterbi score.

    Only the K states containing the moved founder k see new emissions; the others (the
    background) change only through the switch term max(row) - penalty. The current
    panel's forward chain from block chain_block (forward_chain: its messages
    chain_values at block boundaries through every end_block[i], its leaders site_max /
    site_lead per site and block_max / block_lead per block) gives that term. While the
    chain's leader is a background state and the new focal states do not exceed it, the
    switch terms and block normalisers are the chain's, so the background states equal
    the chain's and only the focal states are advanced: O(K) instead of O(Q) per site,
    the same arithmetic. Where a sample's leader contains k, or its new focal states
    would lead, inside block b, the dense recursion (candidate_gains_reference) resumes
    at boundary b from the chain's background states and the focal states held there.
    Bitwise the reference's gains.

    Candidates are scored one after another, their samples in parallel: a candidate's
    pair dosages (the moved founder's new rows with the others' current alleles) are the
    same for every sample, so they are read from one site-major table built per segment
    of whole blocks of at most DOSAGE_BYTES (_segments; a window candidate is normally
    one segment), with a site's three emissions read once per sample. Each sample's
    state (focal states, or every state once the dense recursion has taken over, and
    the normalisers added so far) carries from segment to segment.
    """
    first, second = _unordered_pairs(len(haps))
    founders = len(haps)
    states = len(first)
    samples = logs.shape[0]
    count = len(focal)
    gains = np.empty((count, samples))
    focal_states = _focal_states(first, second, founders)
    site_base = offsets[chain_block]
    focal_all = np.empty((samples, founders))
    held_all = np.empty((samples, founders))
    row_all = np.empty((samples, states))
    scale_all = np.empty(samples)
    dense_all = np.zeros(samples, np.bool_)
    for i in range(count):
        k = focal[i]
        fs = focal_states[k]
        start, stop = first_block[i], end_block[i]
        bounds = _segments(offsets[start:stop + 1], states)
        for u in range(len(bounds) - 1):
            b0, b1 = start + bounds[u], start + bounds[u + 1]
            lo = offsets[b0]
            dos = _item_dosages(haps, leaves, offsets, k, rows, row_offsets[i] + b0 - start, b0, b1,
                                first, second)
            for s in prange(samples):
                xf = focal_all[s]
                held = held_all[s]
                row = row_all[s]
                if b0 == start:
                    for t in range(founders):
                        xf[t] = f_start[i, s, fs[t]]
                    scale = 0.0
                    dense = False
                else:
                    scale = scale_all[s]
                    dense = dense_all[s]
                b = b0
                if not dense:
                    while b < b1:
                        held[:] = xf
                        fast = True
                        for j in range(offsets[b], offsets[b + 1]):
                            if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                                continue
                            lead = site_lead[s, j - site_base]
                            top = site_max[s, j - site_base]
                            if first[lead] == k or second[lead] == k:
                                fast = False
                                break
                            for t in range(founders):
                                if xf[t] > top:
                                    fast = False
                            if not fast:
                                break
                            sw = top - penalty
                            e0, e1, e2 = _site_emissions(logs, s, j)
                            for t in range(founders):
                                xf[t] = max(xf[t], sw) + _pick(dos[j - lo, fs[t]], e0, e1, e2)
                        m = 0.0
                        if fast:
                            lead = block_lead[b - chain_block, s]
                            m = block_max[b - chain_block, s]
                            if first[lead] == k or second[lead] == k:
                                fast = False
                            for t in range(founders):
                                if xf[t] > m:
                                    fast = False
                        if not fast:
                            xf[:] = held
                            break
                        scale += m
                        for t in range(founders):
                            xf[t] -= m
                        b += 1
                    if b < b1:
                        # The candidate's states at boundary b: background from the chain,
                        # focal held; the dense recursion takes over from there.
                        if b == start:
                            row[:] = f_start[i, s]
                        else:
                            row[:] = chain_values[b - chain_block, s]
                        for t in range(founders):
                            row[fs[t]] = xf[t]
                        dense = True
                if dense:
                    scale = _dense_tab(dos, lo, logs, offsets, penalty, row, b, b1, s, scale)
                scale_all[s] = scale
                dense_all[s] = dense
                if b1 == stop:
                    old = _max_close(f_start[i, s], g_start[i, s])
                    if not dense:
                        row[:] = chain_values[stop - chain_block, s]
                        for t in range(founders):
                            row[fs[t]] = xf[t]
                    gains[i, s] = _max_close(row, g_end[i, s]) + scale - old - scale_sum[i, s]
    return gains


@njit(cache=True, nogil=True)
def _candidate_dense_lex(haps, leaves, logs, offsets, penalty, row, cnt, k, rows, row_start, start,
                         b0, b1, s, scale, first, second):
    """The all-state lexicographic recursion of a candidate (founder k's rows replaced by
    rows[row_start + b - start] on blocks b) for sample s over blocks [b0, b1), in place on
    (row, cnt); returns scale plus the blocks' normalisers, added in block order."""
    states = len(first)
    for b in range(b0, b1):
        r = rows[row_start + b - start]
        leaf = leaves[b]
        base = offsets[b]
        for j in range(offsets[b], offsets[b + 1]):
            if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                continue
            m, c = _lex_best(row, cnt)
            sw = m - penalty
            for q in range(states):
                if sw > row[q] or (sw == row[q] and c + 1 < cnt[q]):
                    row[q] = sw
                    cnt[q] = c + 1
                a = first[q]
                e = second[q]
                d = ((leaf[r, j - base] if a == k else haps[a, j])
                     + (leaf[r, j - base] if e == k else haps[e, j]))
                row[q] += _emission(logs, s, j, d)
        m = np.max(row)
        scale += m
        for q in range(states):
            row[q] -= m
    return scale


@njit(inline="always")
def _lex_close(row, cnt, g, gcnt):
    """The lexicographic maximum of (row + g, cnt + gcnt) over the states."""
    best = -np.inf
    best_cnt = 0
    for q in range(len(row)):
        v = row[q] + g[q]
        n = cnt[q] + gcnt[q]
        if v > best or (v == best and n < best_cnt):
            best = v
            best_cnt = n
    return best, best_cnt


@njit(inline="always")
def _lex_old(f, fcnt, g, gcnt):
    """The current lexicographic maximum (score, switches) of one sample at a boundary."""
    old = -np.inf
    old_cnt = 0
    for q in range(len(f)):
        v = f[q] + g[q]
        n = fcnt[q] + gcnt[q]
        if v > old or (v == old and n < old_cnt):
            old = v
            old_cnt = n
    return old, old_cnt


@njit(cache=True, parallel=True, nogil=True)
def candidate_gains_lex_reference(haps, leaves, logs, offsets, penalty, f_start, fcnt_start, g_start,
                                  gcnt_start, focal, first_block, end_block, rows, row_offsets, g_end,
                                  gcnt_end, scale_sum):
    """Validation reference of candidate_gains_lex: every sample by the all-state recursion."""
    first, second = _unordered_pairs(len(haps))
    samples = logs.shape[0]
    count = len(focal)
    gains = np.empty((count, samples))
    switches = np.empty((count, samples), np.int64)
    for task in prange(count * samples):
        i, s = task // samples, task % samples
        old, old_cnt = _lex_old(f_start[i, s], fcnt_start[i, s], g_start[i, s], gcnt_start[i, s])
        row = f_start[i, s].copy()
        cnt = fcnt_start[i, s].copy()
        start = first_block[i]
        scale = _candidate_dense_lex(haps, leaves, logs, offsets, penalty, row, cnt, focal[i], rows,
                                     row_offsets[i], start, start, end_block[i], s, 0.0, first,
                                     second)
        best, best_cnt = _lex_close(row, cnt, g_end[i, s], gcnt_end[i, s])
        gains[i, s] = best + scale - old - scale_sum[i, s]
        switches[i, s] = best_cnt - old_cnt
    return gains, switches


@njit(cache=True, nogil=True)
def _dense_lex_tab(dos, lo, logs, offsets, penalty, row, cnt, b0, b1, s, scale):
    """_candidate_dense_lex on a candidate's dosage table (dos[j - lo, q], _item_dosages):
    the all-state lexicographic recursion of sample s over blocks [b0, b1), in place on
    (row, cnt); returns scale plus the blocks' normalisers, added in block order (the same
    arithmetic)."""
    states = len(row)
    for b in range(b0, b1):
        for j in range(offsets[b], offsets[b + 1]):
            if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                continue
            e0, e1, e2 = _site_emissions(logs, s, j)
            m, c = _lex_best(row, cnt)
            sw = m - penalty
            for q in range(states):
                if sw > row[q] or (sw == row[q] and c + 1 < cnt[q]):
                    row[q] = sw
                    cnt[q] = c + 1
                row[q] += _pick(dos[j - lo, q], e0, e1, e2)
        m = np.max(row)
        scale += m
        for q in range(states):
            row[q] -= m
    return scale


@njit(cache=True, parallel=True, nogil=True)
def candidate_gains_lex(haps, leaves, logs, offsets, penalty, f_start, fcnt_start, g_start,
                        gcnt_start, focal, first_block, end_block, rows, row_offsets, g_end,
                        gcnt_end, scale_sum, chain_values, chain_counts, site_max, site_lead,
                        block_max, block_lead, site_cnt, chain_block, chain_end):
    """candidate_gains() plus the per-sample change of the lexicographic switch count.

    The background fast path of candidate_gains for the lexicographic pair (score, switch
    count): the current panel's lexicographic forward chain from block chain_block
    (forward_chain_lex; first_block[i] >= chain_block) has messages and counts at
    boundaries chain_block..chain_end, per site the lexicographic maximum (site_max,
    site_cnt) before the site's update and the first state attaining it (site_lead), per
    block the maximum before the normalisation and its first state. While the site leader
    is a background state (not containing the moved founder k) and no new focal state
    exceeds its score or ties it with fewer switches, the candidate's lexicographic maximum
    is the chain's, so the switch terms, the background states' scores and counts are the
    chain's; while the block leader is a background state and no focal state exceeds the
    block maximum, so is the block normaliser. Only the K focal states are then advanced.
    Where this fails inside block b (or the chain ends at b), the all-state recursion
    resumes at boundary b from the chain's background states and the focal states held
    there. Every value is the all-state recursion's: bitwise candidate_gains_lex_reference.
    Candidates one after another, their samples in parallel, on per-segment dosage tables
    as candidate_gains.
    """
    first, second = _unordered_pairs(len(haps))
    founders = len(haps)
    states = len(first)
    samples = logs.shape[0]
    count = len(focal)
    gains = np.empty((count, samples))
    switches = np.empty((count, samples), np.int64)
    focal_states = _focal_states(first, second, founders)
    site_base = offsets[chain_block]
    focal_all = np.empty((samples, founders))
    focal_cnt_all = np.empty((samples, founders), np.int64)
    held_all = np.empty((samples, founders))
    held_cnt_all = np.empty((samples, founders), np.int64)
    row_all = np.empty((samples, states))
    cnt_all = np.empty((samples, states), np.int64)
    scale_all = np.empty(samples)
    dense_all = np.zeros(samples, np.bool_)
    for i in range(count):
        k = focal[i]
        fs = focal_states[k]
        start, stop = first_block[i], end_block[i]
        bounds = _segments(offsets[start:stop + 1], states)
        for u in range(len(bounds) - 1):
            b0, b1 = start + bounds[u], start + bounds[u + 1]
            lo = offsets[b0]
            dos = _item_dosages(haps, leaves, offsets, k, rows, row_offsets[i] + b0 - start, b0, b1,
                                first, second)
            for s in prange(samples):
                xf = focal_all[s]
                xc = focal_cnt_all[s]
                held_f = held_all[s]
                held_c = held_cnt_all[s]
                row = row_all[s]
                cnt = cnt_all[s]
                if b0 == start:
                    for t in range(founders):
                        xf[t] = f_start[i, s, fs[t]]
                        xc[t] = fcnt_start[i, s, fs[t]]
                    scale = 0.0
                    dense = False
                else:
                    scale = scale_all[s]
                    dense = dense_all[s]
                b = b0
                if not dense:
                    while b < min(b1, chain_end):
                        held_f[:] = xf
                        held_c[:] = xc
                        fast = True
                        for j in range(offsets[b], offsets[b + 1]):
                            if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                                continue
                            lead = site_lead[s, j - site_base]
                            if first[lead] == k or second[lead] == k:
                                fast = False
                                break
                            m = site_max[s, j - site_base]
                            c = site_cnt[s, j - site_base]
                            for t in range(founders):
                                if xf[t] > m or (xf[t] == m and xc[t] < c):
                                    fast = False
                            if not fast:
                                break
                            sw = m - penalty
                            e0, e1, e2 = _site_emissions(logs, s, j)
                            for t in range(founders):
                                if sw > xf[t] or (sw == xf[t] and c + 1 < xc[t]):
                                    xf[t] = sw
                                    xc[t] = c + 1
                                xf[t] += _pick(dos[j - lo, fs[t]], e0, e1, e2)
                        m = 0.0
                        if fast:
                            lead = block_lead[b - chain_block, s]
                            m = block_max[b - chain_block, s]
                            if first[lead] == k or second[lead] == k:
                                fast = False
                            for t in range(founders):
                                if xf[t] > m:
                                    fast = False
                        if not fast:
                            xf[:] = held_f
                            xc[:] = held_c
                            break
                        scale += m
                        for t in range(founders):
                            xf[t] -= m
                        b += 1
                    if b < b1:
                        # The all-state recursion from boundary b: background states from the chain.
                        if b == start:
                            row[:] = f_start[i, s]
                            cnt[:] = fcnt_start[i, s]
                        else:
                            row[:] = chain_values[b - chain_block, s]
                            cnt[:] = chain_counts[b - chain_block, s]
                        for t in range(founders):
                            row[fs[t]] = xf[t]
                            cnt[fs[t]] = xc[t]
                        dense = True
                if dense:
                    scale = _dense_lex_tab(dos, lo, logs, offsets, penalty, row, cnt, b, b1, s, scale)
                scale_all[s] = scale
                dense_all[s] = dense
                if b1 == stop:
                    old, old_cnt = _lex_old(f_start[i, s], fcnt_start[i, s], g_start[i, s],
                                            gcnt_start[i, s])
                    if not dense:
                        row[:] = chain_values[stop - chain_block, s]
                        cnt[:] = chain_counts[stop - chain_block, s]
                        for t in range(founders):
                            row[fs[t]] = xf[t]
                            cnt[fs[t]] = xc[t]
                    best, best_cnt = _lex_close(row, cnt, g_end[i, s], gcnt_end[i, s])
                    gains[i, s] = best + scale - old - scale_sum[i, s]
                    switches[i, s] = best_cnt - old_cnt
    return gains, switches


@njit(cache=True, parallel=True, nogil=True)
def path_switch_events(haps, leaves, logs, offsets, penalty, f_start, fcnt_start, g_end, gcnt_end,
                       a, c, samples, focal, p1, p2, rows):
    """Switch events of each listed sample's lexicographic maximum path on blocks [a, c).

    The path is the traceback of the accumulators of messages_lex (maximum score, then
    the fewest switches among exact maxima, first state on ties) started from the
    current messages f_start / fcnt_start at boundary a and closed by g_end / gcnt_end at
    boundary c (one row per listed sample). With focal >= 0 founder focal's rows are
    replaced on blocks [p1, p2) by `rows`, as in candidate_gains_lex. Returns the path's
    state at boundaries a and c, its whole switch count (prefix and suffix counts from
    the messages) and its events inside [a, c): (site, state before, state after).
    """
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    n = len(samples)
    lo, hi = offsets[a], offsets[c]
    width = max(hi - lo, 1)
    start_state = np.empty(n, np.int64)
    end_state = np.empty(n, np.int64)
    total = np.empty(n, np.int64)
    counts = np.zeros(n, np.int64)
    events = np.empty((n, width, 3), np.int64)
    for i in prange(n):
        s = samples[i]
        switched = np.zeros((width, states), np.bool_)
        source = np.full(width, -1, np.int64)
        row = f_start[i].copy()
        cnt = fcnt_start[i].copy()
        for b in range(a, c):
            override = focal >= 0 and p1 <= b < p2
            r = rows[b - p1] if override else 0
            leaf = leaves[b]
            base = offsets[b]
            for j in range(offsets[b], offsets[b + 1]):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                m, k, best = _lex_best_index(row, cnt)
                sw = m - penalty
                source[j - lo] = best
                for q in range(states):
                    if sw > row[q] or (sw == row[q] and k + 1 < cnt[q]):
                        row[q] = sw
                        cnt[q] = k + 1
                        switched[j - lo, q] = True
                    x = first[q]
                    y = second[q]
                    if override:
                        d = ((leaf[r, j - base] if x == focal else haps[x, j])
                             + (leaf[r, j - base] if y == focal else haps[y, j]))
                    else:
                        d = haps[x, j] + haps[y, j]
                    row[q] += _emission(logs, s, j, d)
            m = np.max(row)
            for q in range(states):
                row[q] -= m
        best_value = -np.inf
        best_cnt = 0
        end = 0
        for q in range(states):
            v = row[q] + g_end[i, q]
            k = cnt[q] + gcnt_end[i, q]
            if v > best_value or (v == best_value and k < best_cnt):
                best_value = v
                best_cnt = k
                end = q
        q = end
        nev = 0
        for j in range(hi - 1, lo - 1, -1):
            if source[j - lo] < 0:
                continue
            if switched[j - lo, q]:
                events[i, nev, 0] = j
                events[i, nev, 1] = source[j - lo]
                events[i, nev, 2] = q
                nev += 1
                q = source[j - lo]
        start_state[i] = q
        end_state[i] = end
        counts[i] = nev
        total[i] = fcnt_start[i, q] + nev + gcnt_end[i, end]
    return start_state, end_state, total, counts, events


@njit(cache=True, nogil=True)
def transition_intervals(haps, logs, samples, from_founder, to_founder, sites, lo_site, hi_site):
    """Evidence interval (lo, hi] of founder transitions in sample paths.

    For the transition of sample samples[e] from founder from_founder[e] to
    to_founder[e] placed at site sites[e], lo is the last site before it and hi the
    first site at or after it where the sample is informative and the two founders
    carry different alleles (lo_site - 1 and hi_site when there is none inside
    [lo_site, hi_site)). The sample's data cannot place the breakpoint elsewhere than
    in (lo, hi]: inside it the two paths have equal emissions.
    """
    n = len(samples)
    out = np.empty((n, 2), np.int64)
    for e in range(n):
        s = samples[e]
        u = from_founder[e]
        v = to_founder[e]
        j = sites[e]
        lo = lo_site - 1
        for i in range(j - 1, lo_site - 1, -1):
            if haps[u, i] != haps[v, i] and not (
                    logs[s, i, 0] == 0.0 and logs[s, i, 1] == 0.0 and logs[s, i, 2] == 0.0):
                lo = i
                break
        hi = hi_site
        for i in range(j, hi_site):
            if haps[u, i] != haps[v, i] and not (
                    logs[s, i, 0] == 0.0 and logs[s, i, 1] == 0.0 and logs[s, i, 2] == 0.0):
                hi = i
                break
        out[e, 0] = lo
        out[e, 1] = hi
    return out


@njit(cache=True, parallel=True, nogil=True)
def exchange_gains(f, g, gperm, perms):
    """Exact dF of every founder-pair suffix exchange at one junction (O(S K^3)).

    g is read through gperm (lazy relabelling of stored messages). The exchange of
    founders a < b relabels the suffix states; the junction value of sample s is
    max_q f[q] + g[perm(q)]. A state touching neither a nor b, and the state (a, b)
    itself, is fixed by the transposition, so the maximum over those states is the
    first of them in the per-sample descending order of t[q] = f[q] + g[q] (at most
    2K - 1 entries are skipped); only the 2K - 2 states touching exactly one of a, b
    are evaluated swapped: (a, a) <-> (b, b) and (a, c) <-> (b, c). This is the sorted
    scan of the former guarded exchange pass in the sweep's message convention (g
    already carries the switch into the suffix, so that scan's explicit switch term is
    implied). The maximum is taken over exactly the float sums f[q] + g[perm(q)] of the
    direct scan, and the per-sample differences are summed in the same order, so the
    gains are bitwise identical to the O(S K^4) all-state scan they replace.
    `perms` (the pair order, a < b in triu order) is kept for the callers' relabelling.
    """
    samples, states = f.shape
    npairs = perms.shape[0]
    founders = 1
    while founders * (founders + 1) // 2 < states:
        founders += 1
    first, second = _unordered_pairs(founders)
    index = np.empty((founders, founders), np.int64)
    for q in range(states):
        index[first[q], second[q]] = q
        index[second[q], first[q]] = q
    per = np.zeros((samples, npairs))
    for s in prange(samples):
        terms = np.empty(states)
        base = -np.inf
        for q in range(states):
            terms[q] = f[s, q] + g[s, gperm[q]]
            base = max(base, terms[q])
        order = np.argsort(-terms)
        p = 0
        for a in range(founders):
            for b in range(a + 1, founders):
                best = -np.inf
                for position in range(states):
                    q = order[position]
                    i = first[q]
                    j = second[q]
                    if (i != a and i != b and j != a and j != b) or (i == a and j == b):
                        best = terms[q]
                        break
                best = max(best, f[s, index[a, a]] + g[s, gperm[index[b, b]]])
                best = max(best, f[s, index[b, b]] + g[s, gperm[index[a, a]]])
                for c in range(founders):
                    if c != a and c != b:
                        best = max(best, f[s, index[a, c]] + g[s, gperm[index[b, c]]])
                        best = max(best, f[s, index[b, c]] + g[s, gperm[index[a, c]]])
                per[s, p] = best - base
                p += 1
    out = np.zeros(npairs)
    for p in range(npairs):
        for s in range(samples):
            out[p] += per[s, p]
    return out


@njit(cache=True, parallel=True, nogil=True)
def exchange_gains_reference(f, g, gperm, perms):
    """The direct O(S K^4) all-state scan (validation reference of exchange_gains)."""
    samples, states = f.shape
    npairs = perms.shape[0]
    per = np.zeros((samples, npairs))
    for s in prange(samples):
        base = -np.inf
        for q in range(states):
            base = max(base, f[s, q] + g[s, gperm[q]])
        for p in range(npairs):
            best = -np.inf
            for q in range(states):
                best = max(best, f[s, q] + g[s, gperm[perms[p, q]]])
            per[s, p] = best - base
    out = np.zeros(npairs)
    for p in range(npairs):
        for s in range(samples):
            out[p] += per[s, p]
    return out


@njit(cache=True, parallel=True, nogil=True)
def clamp_reverse(carried, penalty):
    """Stored messages of the reversed orientation from carried forward messages.

    out[B - c] = max(m, max(m) - penalty) of carried[c], renormalised.
    """
    nb, samples, states = carried.shape
    out = np.empty_like(carried)
    for c in prange(nb):
        for s in range(samples):
            m = -np.inf
            for q in range(states):
                m = max(m, carried[c, s, q])
            for q in range(states):
                out[nb - 1 - c, s, q] = max(carried[c, s, q], m - penalty) - m
    return out


@njit(cache=True, parallel=True, nogil=True)
def clamp_reverse_lex(carried, carried_cnt, penalty):
    nb, samples, states = carried.shape
    out = np.empty_like(carried)
    out_cnt = np.empty_like(carried_cnt)
    for c in prange(nb):
        for s in range(samples):
            m, k = _lex_best(carried[c, s], carried_cnt[c, s])
            for q in range(states):
                v = carried[c, s, q]
                n = carried_cnt[c, s, q]
                sw = m - penalty
                if sw > v or (sw == v and k + 1 < n):
                    v = sw
                    n = k + 1
                out[nb - 1 - c, s, q] = v - m
                out_cnt[nb - 1 - c, s, q] = n
    return out, out_cnt


# ---------------------------------------------------------------------- compiled pass driver
# The Gauss-Seidel acceptance pass (driven by converged_sweep._gs_pass) as successive
# nogil calls of gs_pass_chunk: the junction loop, exchange scoring and application,
# candidate bookkeeping, ranking, the coarse-move guard (incl. the shared-junction test),
# the lookahead forward chain and the backward chain from the frontier. It calls the
# kernels above (per-sample parallel tasks); its own arithmetic is the former
# Python-driven pass's operation for operation, including NumPy's pairwise summation of
# per-sample gains (_numpy_sum), so its decisions are bitwise identical.
ACC_KEYS = ("fine", "coarse", "exchange", "fine_gain", "coarse_gain", "exchange_gain", "evaluated",
            "false_positive", "guard_rejected", "guard_shared_tests", "guard_relaxed",
            "guard_shared_switches", "guard_traced_blocks", "chain_blocks", "ahead_blocks",
            "changed_blocks")
_A_FINE, _A_COARSE, _A_EXCHANGE = 0, 1, 2
_A_FINE_GAIN, _A_COARSE_GAIN, _A_EXCHANGE_GAIN = 3, 4, 5
_A_EVALUATED, _A_FALSE_POSITIVE, _A_GUARD_REJECTED = 6, 7, 8
_A_GUARD_TESTS, _A_GUARD_RELAXED, _A_GUARD_SWITCHES = 9, 10, 11
_A_TRACED, _A_CHAIN, _A_AHEAD, _A_CHANGED = 12, 13, 14, 15
MOVE_EXCHANGE, MOVE_FINE, MOVE_COARSE = 0, 1, 2
GUARD_NONE, GUARD_GENOTYPE_FIT, GUARD_SHARED_JUNCTION = 0, 1, 2


@njit(cache=True, nogil=True)
def _numpy_sum(a):
    """np.sum of a contiguous float64 vector, bitwise (NumPy's pairwise summation)."""
    n = len(a)
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[i]
        return res
    if n <= 128:
        r0, r1, r2, r3 = a[0], a[1], a[2], a[3]
        r4, r5, r6, r7 = a[4], a[5], a[6], a[7]
        i = 8
        stop = n - (n % 8)
        while i < stop:
            r0 += a[i]
            r1 += a[i + 1]
            r2 += a[i + 2]
            r3 += a[i + 3]
            r4 += a[i + 4]
            r5 += a[i + 5]
            r6 += a[i + 6]
            r7 += a[i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _numpy_sum(a[:n2]) + _numpy_sum(a[n2:])


@njit(cache=True, nogil=True)
def _add_column_sums(total, a, lo, hi):
    """total += a[lo:hi].sum(axis=0) (NumPy's order: rows added in turn)."""
    if hi <= lo:
        return
    part = a[lo].copy()
    for r in range(lo + 1, hi):
        for s in range(len(part)):
            part[s] += a[r, s]
    for s in range(len(part)):
        total[s] += part[s]


@njit(cache=True, nogil=True)
def _transition_list(first, second, before, after, out):
    """Founder transitions (u -> v) of one homologue turning `before` into `after`
    (converged_sweep._transitions); writes them to out (4 x 2), returns how many."""
    x1, y1, x2, y2 = first[before], second[before], first[after], second[after]
    if x1 == x2:
        out[0, 0], out[0, 1] = y1, y2
        return 1
    if x1 == y2:
        out[0, 0], out[0, 1] = y1, x2
        return 1
    if y1 == x2:
        out[0, 0], out[0, 1] = x1, y2
        return 1
    if y1 == y2:
        out[0, 0], out[0, 1] = x1, x2
        return 1
    out[0, 0], out[0, 1] = x1, x2
    out[1, 0], out[1, 1] = y1, y2
    out[2, 0], out[2, 1] = x1, y2
    out[3, 0], out[3, 1] = y1, x2
    return 4


@njit(cache=True, nogil=True)
def _sorted_events(events, count):
    """A path's switch events (site, before, after) as a lexicographically sorted copy."""
    out = events[:count].copy()
    order = np.arange(count)
    for i in range(1, count):
        j = i
        while j > 0:
            a, b = order[j - 1], order[j]
            if (events[a, 0], events[a, 1], events[a, 2]) > (events[b, 0], events[b, 1], events[b, 2]):
                order[j - 1], order[j] = b, a
                j -= 1
            else:
                break
    for i in range(count):
        out[i] = events[order[i]]
    return out


@njit(cache=True, nogil=True)
def _removed_events(old_events, old_count, new_events, new_count):
    """sorted(set(old) - set(new)) of two paths' switch events."""
    a = _sorted_events(old_events, old_count)
    b = _sorted_events(new_events, new_count)
    keep = np.zeros(len(a), np.bool_)
    n = 0
    j = 0
    for i in range(len(a)):
        if i > 0 and a[i, 0] == a[i - 1, 0] and a[i, 1] == a[i - 1, 1] and a[i, 2] == a[i - 1, 2]:
            continue  # set semantics
        ka = (a[i, 0], a[i, 1], a[i, 2])
        while j < len(b) and (b[j, 0], b[j, 1], b[j, 2]) < ka:
            j += 1
        if j < len(b) and (b[j, 0], b[j, 1], b[j, 2]) == ka:
            continue
        keep[i] = True
        n += 1
    out = np.empty((n, 3), np.int64)
    m = 0
    for i in range(len(a)):
        if keep[i]:
            out[m] = a[i]
            m += 1
    return out


@njit(cache=True, nogil=True)
def _shared_savings(rem_sample, rem_offsets, rem_events, saved, focal, first, second, haps, logs,
                    total_sites):
    """Shared one-way switch savings of the guard on flattened removed events: sample
    rem_sample[i] has events rem_events[rem_offsets[i]:rem_offsets[i + 1]] (sorted).
    saved[s]: net switches sample s saves. Returns the uncredited total."""
    trans = np.empty((4, 2), np.int64)
    # one-way events (FIFO pairing of a focal founder's excursions per sample)
    n_events = len(rem_events)
    one_way = np.ones(n_events, np.bool_)
    for i in range(len(rem_sample)):
        lo, hi = rem_offsets[i], rem_offsets[i + 1]
        open_flag = np.zeros(hi - lo, np.bool_)
        key_u = np.full(hi - lo, -1, np.int64)
        key_v = np.full(hi - lo, -1, np.int64)
        for e in range(hi - lo):
            ev = rem_events[lo + e]
            n = _transition_list(first, second, ev[1], ev[2], trans)
            if n != 1:
                continue
            u, v = trans[0, 0], trans[0, 1]
            if focal != u and focal != v:
                continue
            key_u[e], key_v[e] = u, v
            partner = -1
            for w in range(e):
                if open_flag[w] and key_u[w] == v and key_v[w] == u:
                    partner = w
                    break
            if partner >= 0:
                open_flag[partner] = False
                one_way[lo + partner] = False
                one_way[lo + e] = False
            else:
                open_flag[e] = True
    # entries: every transition of every one-way event
    n_entries = 0
    for i in range(len(rem_sample)):
        for e in range(rem_offsets[i], rem_offsets[i + 1]):
            if one_way[e]:
                n_entries += _transition_list(first, second, rem_events[e, 1], rem_events[e, 2], trans)
    if n_entries == 0:
        return 0
    ent_sample = np.empty(n_entries, np.int64)
    ent_event = np.empty(n_entries, np.int64)
    ent_u = np.empty(n_entries, np.int64)
    ent_v = np.empty(n_entries, np.int64)
    ent_site = np.empty(n_entries, np.int64)
    m = 0
    for i in range(len(rem_sample)):
        for e in range(rem_offsets[i], rem_offsets[i + 1]):
            if not one_way[e]:
                continue
            n = _transition_list(first, second, rem_events[e, 1], rem_events[e, 2], trans)
            for t in range(n):
                ent_sample[m] = rem_sample[i]
                ent_event[m] = e
                ent_u[m] = trans[t, 0]
                ent_v[m] = trans[t, 1]
                ent_site[m] = rem_events[e, 0]
                m += 1
    intervals = transition_intervals(haps, logs, ent_sample, ent_u, ent_v, ent_site, 0, total_sites)
    # group by (u, v), sort by (lo, hi, sample, event), mark overlapping pairs of samples
    order = np.arange(n_entries)
    for i in range(1, n_entries):
        j = i
        while j > 0:
            a, b = order[j - 1], order[j]
            ka = (ent_u[a], ent_v[a], intervals[a, 0], intervals[a, 1], ent_sample[a], ent_event[a])
            kb = (ent_u[b], ent_v[b], intervals[b, 0], intervals[b, 1], ent_sample[b], ent_event[b])
            if ka > kb:
                order[j - 1], order[j] = b, a
                j -= 1
            else:
                break
    shared = np.zeros(n_events, np.bool_)
    for x in range(n_entries):
        a = order[x]
        for y in range(x + 1, n_entries):
            b = order[y]
            if ent_u[b] != ent_u[a] or ent_v[b] != ent_v[a]:
                break
            if intervals[b, 0] >= intervals[a, 1]:
                break
            if ent_sample[a] != ent_sample[b] and max(intervals[a, 0], intervals[b, 0]) < min(
                    intervals[a, 1], intervals[b, 1]):
                shared[ent_event[a]] = True
                shared[ent_event[b]] = True
    total = 0
    for i in range(len(rem_sample)):
        count = 0
        for e in range(rem_offsets[i], rem_offsets[i + 1]):
            if shared[e]:
                count += 1
        if count:
            total += min(saved[rem_sample[i]], count)
    return total


@njit(cache=True, nogil=True)
def _set_rows(haps, leaves, offsets, founder, start, rows, lo, hi):
    """haps[founder] on blocks start + (0 .. hi - lo) from rows[lo:hi] (Orientation.set_rows)."""
    for i in range(hi - lo):
        block = start + i
        leaf = leaves[block]
        r = rows[lo + i]
        for j in range(offsets[block], offsets[block + 1]):
            haps[founder, j] = leaf[r, j - offsets[block]]


@njit(cache=True, nogil=True)
def _stored_at(stored, x, gperm, permuted):
    if permuted:
        return np.ascontiguousarray(stored[x][:, gperm])
    return stored[x]


@njit(cache=True, nogil=True)
def _ensure_chain(chain, frontier, p, haps, logs, offsets, penalty, stored, stored_cnt, gperm,
                  lexmode, acc):
    """Exact current backward messages (counts, normalisers) on boundaries [p, frontier]
    (the backward chain from the frontier); chain = (ok, start, end, values, counts,
    scales), exact on boundaries [start, end] when ok. A chain that still ends at the
    frontier but starts right of p is extended over blocks [p, start) from its exact value
    at start (the recursion from the frontier passes through that value), otherwise the
    chain is recomputed from the frontier."""
    ok, start, end, values, counts, scales = chain
    if ok and end == frontier and p >= start:
        return chain
    if ok and end == frontier:
        ident = np.arange(values.shape[2])
        if lexmode:
            head, head_cnt, head_scales = backward_chain_lex(haps, logs, offsets, penalty, values[0],
                                                             counts[0], ident, p, start)
            counts = np.concatenate((head_cnt[:-1], counts))
        else:
            head, head_scales = backward_chain(haps, logs, offsets, penalty, values[0], ident, p, start)
        acc[_A_CHAIN] += start - p
        return (True, p, end, np.concatenate((head[:-1], values)), counts,
                np.concatenate((head_scales, scales)))
    if lexmode:
        values, counts, scales = backward_chain_lex(haps, logs, offsets, penalty, stored[frontier],
                                                    stored_cnt[frontier], gperm, p, frontier)
    else:
        values, scales = backward_chain(haps, logs, offsets, penalty, stored[frontier], gperm, p,
                                        frontier)
    acc[_A_CHAIN] += frontier - p
    return (True, p, frontier, values, counts, scales)


@njit(cache=True, nogil=True)
def _chain_from(chain, x, lexmode):
    """The backward chain after an edit of blocks left of boundary x that leaves the frontier
    in place: its values and counts at boundaries >= x and the normalisers of blocks >= x
    depend only on blocks >= x and stay exact; boundaries left of x are dropped
    (_ensure_chain extends the chain from x when they are needed)."""
    ok, start, end, values, counts, scales = chain
    if not (ok and start <= x <= end):
        return (False, start, end, values, counts, scales)
    k = x - start
    return (True, x, end, values[k:], counts[k:] if lexmode else counts, scales[k:])


@njit(cache=True, parallel=True, nogil=True)
def _relabel_chain(chain, p, perm, lexmode):
    """The backward chain after the suffix exchange perm at junction p: on every block from p
    on two founders' rows are swapped, so state q there now holds the pair dosages of the
    former state perm[q]; the backward recursion is the same up to that state order (its
    maxima and lexicographic maxima are order-free), so the exact messages and switch counts
    at boundaries >= p are the former ones with their states relabelled and the block
    normalisers are unchanged. Boundaries left of p are dropped (the pass never reads them
    again)."""
    ok, start, end, values, counts, scales = chain
    if not (ok and start <= p <= end):
        return (False, start, end, values, counts, scales)
    k = p - start
    boundaries, samples, states = end - p + 1, values.shape[1], values.shape[2]
    relabelled = np.empty((boundaries, samples, states))
    relabelled_cnt = counts
    if lexmode:
        relabelled_cnt = np.empty((boundaries, samples, states), np.int64)
    for task in prange(boundaries * samples):
        x, s = task // samples, task % samples
        for q in range(states):
            relabelled[x, s, q] = values[k + x, s, perm[q]]
        if lexmode:
            for q in range(states):
                relabelled_cnt[x, s, q] = counts[k + x, s, perm[q]]
    return (True, p, end, relabelled, relabelled_cnt, scales[k:])


@njit(cache=True, nogil=True)
def _g_at(chain, x, p, frontier, haps, logs, offsets, penalty, stored, stored_cnt, gperm, permuted,
          lexmode, acc):
    """Current backward message (and counts) at boundary x; returns (g, gc, chain)."""
    if x >= frontier:
        g = _stored_at(stored, x, gperm, permuted)
        gc = _stored_at(stored_cnt, x, gperm, permuted) if lexmode else stored_cnt[0]
        return g, gc, chain
    chain = _ensure_chain(chain, frontier, p, haps, logs, offsets, penalty, stored, stored_cnt, gperm,
                          lexmode, acc)
    g = chain[3][x - chain[1]]
    gc = chain[4][x - chain[1]] if lexmode else stored_cnt[0]
    return g, gc, chain


@njit(cache=True, nogil=True)
def _forward_to(ahead, p, need, f, fcnt, haps, logs, offsets, penalty, lexmode, ahead_blocks, acc):
    """Current forward messages valid on boundaries [p, need] with the chain's leaders
    (the lookahead chain); ahead = (ok, start, valid, values, counts, site_max,
    site_lead, block_max, block_lead, site_cnt) (counts and site_cnt with lexmode)."""
    ok, start, valid, values, counts, site_max, site_lead, block_max, block_lead, site_cnt = ahead
    blocks = len(offsets) - 1
    if ok and valid >= need and start <= p:
        return ahead
    keep = ok and start <= p <= valid
    if keep:
        a0 = valid
        start_v = values[a0 - start]
        start_c = counts[a0 - start] if lexmode else fcnt
    else:
        a0 = p
        start_v = f
        start_c = fcnt
    stop = min(blocks, max(need, p + ahead_blocks))
    if lexmode:
        nv, nc, nsm, nsl, nbm, nbl, nsc = forward_chain_lex(haps, logs, offsets, penalty, start_v,
                                                            start_c, a0, stop)
    else:
        nv, nsm, nsl, nbm, nbl = forward_chain(haps, logs, offsets, penalty, start_v, a0, stop)
        nc = counts
        nsc = site_cnt
    acc[_A_AHEAD] += stop - a0
    if keep:
        k0, k1 = p - start, a0 - start
        base = offsets[start]
        s0, s1 = offsets[p] - base, offsets[a0] - base
        nv = np.concatenate((values[k0:k1], nv))
        if lexmode:
            nc = np.concatenate((counts[k0:k1], nc))
            nsc = np.concatenate((site_cnt[:, s0:s1], nsc), axis=1)
        nsm = np.concatenate((site_max[:, s0:s1], nsm), axis=1)
        nsl = np.concatenate((site_lead[:, s0:s1], nsl), axis=1)
        nbm = np.concatenate((block_max[k0:k1], nbm))
        nbl = np.concatenate((block_lead[k0:k1], nbl))
        return (True, p, stop, nv, nc, nsm, nsl, nbm, nbl, nsc)
    return (True, a0, stop, nv, nc, nsm, nsl, nbm, nbl, nsc)


@njit(cache=True, nogil=True)
def _uncredited(haps, leaves, logs, offsets, penalty, p, j0, k, p1, p2, cand_rows, rs, gains_s,
                switches_s, frontier, base_hist, base_hist_cnt, local, local_cnt, stored, stored_cnt,
                gperm, permuted, f, fcnt, first, second, chain, ahead, acc, ahead_blocks):
    """Uncredited saved switches of a coarse candidate (the shared-junction test):
    returns (uncredited saved switches, chain, ahead)."""
    samples = len(gains_s)
    blocks = len(offsets) - 1
    rows = cand_rows[rs:rs + p2 - p1].copy()
    n_aff = 0
    affected = np.empty(samples, np.int64)
    for s in range(samples):
        if switches_s[s] != 0 or abs(gains_s[s]) > 1e-9:
            affected[n_aff] = s
            n_aff += 1
    affected = affected[:n_aff]
    # removed events per settled sample (flattened, in settling order)
    rem_sample = np.empty(samples, np.int64)
    rem_offsets = np.zeros(samples + 1, np.int64)
    rem_events = np.zeros((0, 3), np.int64)
    n_rem = 0
    width = 2
    while len(affected):
        lo, hi = max(0, p1 - width), min(blocks, p2 + width)
        # forward_at(lo, p)
        if lo < j0:
            f_lo, c_lo = base_hist[lo], base_hist_cnt[lo]
        elif lo <= p:
            f_lo, c_lo = local[lo - j0], local_cnt[lo - j0]
        else:
            ahead = _forward_to(ahead, p, lo, f, fcnt, haps, logs, offsets, penalty, True, ahead_blocks,
                                acc)
            f_lo, c_lo = ahead[3][lo - ahead[1]], ahead[4][lo - ahead[1]]
        g_hi, gc_hi, chain = _g_at(chain, hi, p, frontier, haps, logs, offsets, penalty, stored,
                                   stored_cnt, gperm, permuted, True, acc)
        fa = np.empty((len(affected), f_lo.shape[1]))
        ca = np.empty((len(affected), f_lo.shape[1]), np.int64)
        ga = np.empty((len(affected), f_lo.shape[1]))
        gca = np.empty((len(affected), f_lo.shape[1]), np.int64)
        for i in range(len(affected)):
            fa[i] = f_lo[affected[i]]
            ca[i] = c_lo[affected[i]]
            ga[i] = g_hi[affected[i]]
            gca[i] = gc_hi[affected[i]]
        old = path_switch_events(haps, leaves, logs, offsets, penalty, fa, ca, ga, gca, lo, hi,
                                 affected, -1, 0, 0, rows)
        new = path_switch_events(haps, leaves, logs, offsets, penalty, fa, ca, ga, gca, lo, hi,
                                 affected, k, p1, p2, rows)
        acc[_A_TRACED] += (hi - lo) * len(affected)
        everything = lo == 0 and hi == blocks
        keep = np.ones(len(affected), np.bool_)
        for i in range(len(affected)):
            if everything or (old[0][i] == new[0][i] and old[1][i] == new[1][i]):
                keep[i] = False
                gone = _removed_events(old[4][i], old[3][i], new[4][i], new[3][i])
                if len(gone):
                    rem_sample[n_rem] = affected[i]
                    rem_events = np.concatenate((rem_events, gone))
                    rem_offsets[n_rem + 1] = len(rem_events)
                    n_rem += 1
        affected = affected[keep]
        width *= 2
    saved = np.zeros(samples, np.int64)
    for s in range(samples):
        if switches_s[s] < 0:
            saved[s] = -switches_s[s]
    if n_rem == 0:
        return 0, chain, ahead
    total = _shared_savings(rem_sample[:n_rem], rem_offsets[:n_rem + 1], rem_events, saved, k, first,
                            second, haps, logs, offsets[blocks])
    return total, chain, ahead


@njit(cache=True, nogil=True)
def _grow_moves(moves, move_gain, n_moves):
    if n_moves < len(moves):
        return moves, move_gain
    grown = np.zeros((2 * len(moves), 8), np.int64)
    grown[:n_moves] = moves[:n_moves]
    grown_gain = np.zeros(2 * len(moves))
    grown_gain[:n_moves] = move_gain[:n_moves]
    return grown, grown_gain


@njit(cache=True, nogil=True)
def _active_items(q, X, lab, cand_start, cand_kind, cand_end, cand_k0, cand_row_start, cand_rows,
                  accepted, it_cand, it_kind, it_k, it_p1, it_p2, it_rs):
    """Write the candidates anchored at junction q that differ from the current panel
    (skipping the junction's i-th candidate where accepted[i]) to the item arrays;
    returns how many."""
    n_items = 0
    for i in range(cand_start[q + 1] - cand_start[q]):
        if accepted[i]:
            continue
        cand = cand_start[q] + i
        k = lab[cand_k0[cand]]
        rs = cand_row_start[cand]
        lo = -1
        hi = -1
        for x in range(q, cand_end[cand]):
            if X[k, x] != cand_rows[rs + x - q]:
                if lo < 0:
                    lo = x - q
                hi = x - q + 1
        if lo < 0:
            continue
        it_cand[n_items] = cand
        it_kind[n_items] = cand_kind[cand]
        it_k[n_items] = k
        it_p1[n_items] = q + lo
        it_p2[n_items] = q + hi
        it_rs[n_items] = rs + lo
        n_items += 1
    return n_items


@njit(cache=True, nogil=True)
def _score_items(n_items, it_kind, it_k, it_p1, it_p2, it_rs, cand_rows, p, frontier, haps, leaves, logs,
                 offsets, penalty, stored, stored_cnt, stored_scale, gperm, permuted, lexmode, f, fcnt,
                 chain, ahead, acc, ahead_blocks):
    """Score the items at junction p: guarded runs the coarse items (lexicographic
    kernel) first, then the rest; returns (gain sums (NumPy pairwise), switch sums,
    per-sample gains, per-sample switch changes, chain, ahead)."""
    samples, states = stored.shape[1], stored.shape[2]
    gain = np.zeros(n_items)
    switch = np.zeros(n_items)
    gains_s = np.zeros((n_items, samples))
    switches_s = np.zeros((n_items, samples), np.int64)
    for group in range(2):
        members = np.empty(n_items, np.int64)
        n_mem = 0
        for i in range(n_items):
            if lexmode:
                if (group == 0) == (it_kind[i] == MOVE_COARSE):
                    members[n_mem] = i
                    n_mem += 1
            elif group == 1:
                members[n_mem] = i
                n_mem += 1
        if n_mem == 0:
            continue
        lex = lexmode and group == 0
        members = members[:n_mem]
        # The lookahead covers every member's blocks: the background fast paths of both
        # kernels read the current chain up to the candidates' ends.
        last = it_p2[members[0]]
        for i in members:
            last = max(last, it_p2[i])
        ahead = _forward_to(ahead, p, last, f, fcnt, haps, logs, offsets, penalty, lexmode, ahead_blocks,
                            acc)
        a0 = ahead[1]
        focal = np.empty(n_mem, np.int64)
        first_block = np.empty(n_mem, np.int64)
        end_block = np.empty(n_mem, np.int64)
        row_offsets = np.zeros(n_mem + 1, np.int64)
        for m in range(n_mem):
            i = members[m]
            focal[m] = it_k[i]
            first_block[m] = it_p1[i]
            end_block[m] = it_p2[i]
            row_offsets[m + 1] = row_offsets[m] + it_p2[i] - it_p1[i]
        flat = np.empty(row_offsets[n_mem], np.int64)
        for m in range(n_mem):
            i = members[m]
            for t in range(row_offsets[m + 1] - row_offsets[m]):
                flat[row_offsets[m] + t] = cand_rows[it_rs[i] + t]
        f_start = np.empty((n_mem, samples, states))
        for m in range(n_mem):
            f_start[m] = ahead[3][first_block[m] - a0]
        g_start = np.empty((n_mem, samples, states))
        g_end = np.empty((n_mem, samples, states))
        gc_start = np.empty((n_mem, samples, states), np.int64)
        gc_end = np.empty((n_mem, samples, states), np.int64)
        for m in range(n_mem):
            g, gc, chain = _g_at(chain, first_block[m], p, frontier, haps, logs, offsets, penalty, stored,
                                 stored_cnt, gperm, permuted, lexmode, acc)
            g_start[m] = g
            if lex:
                gc_start[m] = gc
        for m in range(n_mem):
            g, gc, chain = _g_at(chain, end_block[m], p, frontier, haps, logs, offsets, penalty, stored,
                                 stored_cnt, gperm, permuted, lexmode, acc)
            g_end[m] = g
            if lex:
                gc_end[m] = gc
        sums = np.zeros((n_mem, samples))
        for m in range(n_mem):
            lo_b, x = first_block[m], end_block[m]
            if lo_b < min(x, frontier):
                chain = _ensure_chain(chain, frontier, p, haps, logs, offsets, penalty, stored, stored_cnt,
                                      gperm, lexmode, acc)
                hi_b = min(x, frontier)
                _add_column_sums(sums[m], chain[5], lo_b - chain[1], hi_b - chain[1])
                lo_b = hi_b
            if lo_b < x:
                _add_column_sums(sums[m], stored_scale, lo_b, x)
        if lex:
            fc_start = np.empty((n_mem, samples, states), np.int64)
            for m in range(n_mem):
                fc_start[m] = ahead[4][first_block[m] - a0]
            gains2, switches2 = candidate_gains_lex(
                haps, leaves, logs, offsets, penalty, f_start, fc_start, g_start, gc_start, focal,
                first_block, end_block, flat, row_offsets, g_end, gc_end, sums, ahead[3], ahead[4],
                ahead[5], ahead[6], ahead[7], ahead[8], ahead[9], a0, ahead[2])
            for m in range(n_mem):
                i = members[m]
                gain[i] = _numpy_sum(np.ascontiguousarray(gains2[m]))
                gains_s[i] = gains2[m]
                switch[i] = float(np.sum(switches2[m]))
                switches_s[i] = switches2[m]
        else:
            gains2 = candidate_gains(
                haps, leaves, logs, offsets, penalty, f_start, g_start, focal, first_block, end_block,
                flat, row_offsets, g_end, sums, ahead[5], ahead[6], ahead[7], ahead[8], a0, ahead[3])
            for m in range(n_mem):
                i = members[m]
                gain[i] = _numpy_sum(np.ascontiguousarray(gains2[m]))
                gains_s[i] = gains2[m]
    return gain, switch, gains_s, switches_s, chain, ahead


@njit(cache=True, nogil=True)
def gs_pass_chunk(haps, X, leaves, logs, offsets, penalty, j0, j1, frontier0, f0, fcnt0, local, local_cnt,
                  stored, stored_cnt, stored_scale, gperm0, lab, cand_start, cand_kind, cand_end, cand_k0,
                  cand_row_start, cand_rows, perms, pa, pb, guard, exchanges, ahead_blocks, tol,
                  guard_tolerance, acc, ahead0, chain0, start_fwd, start_fcnt, pristine0):
    """Junctions [j0, j1) of one exact Gauss-Seidel acceptance pass (converged_sweep._gs_pass),
    compiled: exchanges, then the content candidates anchored at every junction, the
    coarse-move guard, the lookahead forward chain and the backward chain from the
    frontier, all as the pass's specification.
    State in: frontier0, the carried forward message f0 (counts fcnt0) at j0, gperm0
    (current state -> stored state), lab (proposal founder label -> current row; modified
    in place, as haps and X), the guard's history `local` of boundaries 0 .. j0 (filled
    here for j0 .. j1 - 1), the lookahead and backward-chain caches ahead0 / chain0 of the
    previous call (valid as their bounds state: nothing changes between calls).
    pristine0: no move has been accepted before j0 in a pass whose start panel has the
    forward messages start_fwd / start_fcnt (one-element placeholders otherwise); while
    nothing is accepted the carried forward message is read from them (the same
    recursion on the same panel) instead of being advanced. The candidates anchored at
    junction q are cand_start[q] .. cand_start[q + 1] - 1 in proposal order (kind
    MOVE_FINE / MOVE_COARSE, end block, founder label, rows cand_rows[cand_row_start[i]:]
    over [q, end)). guard: GUARD_NONE /
    GUARD_GENOTYPE_FIT / GUARD_SHARED_JUNCTION; arrays of switch counts are one-element
    placeholders when unused. acc (ACC_KEYS) is accumulated in place. The junctions are
    evaluated one by one.
    Returns (moves (n, 8) int64 rows (kind, p, index, row, p1, p2, a, b), move gains,
    junction_gain and junction_pair of the junctions, gperm, the forward message and
    counts at j1, frontier, whether a move was accepted, the lookahead and backward-chain
    caches)."""
    founders, blocks = X.shape
    samples, states = stored.shape[1], stored.shape[2]
    first, second = _unordered_pairs(founders)
    lexmode = guard != GUARD_NONE
    shared_guard = guard == GUARD_SHARED_JUNCTION
    total_sites = offsets[blocks]
    ident = np.arange(states)
    gperm = gperm0.copy()
    permuted = False
    for q in range(states):
        if gperm[q] != q:
            permuted = True
    junction_gain = np.full(j1 - j0, -np.inf)
    junction_pair = np.zeros((j1 - j0, 2), np.int64)
    moves = np.zeros((64, 8), np.int64)
    move_gain = np.zeros(64)
    n_moves = 0
    frontier = frontier0
    chain = chain0
    ahead = ahead0
    pristine = pristine0
    f = f0
    fcnt = fcnt0
    placeholder = np.zeros((1, 1, 1))
    placeholder_cnt = np.zeros((1, 1, 1), np.int64)
    accepted_any = False
    for p in range(j0, j1):
        if shared_guard:
            local[p] = f
            local_cnt[p] = fcnt
        accepted_here = False
        if p >= 1 and exchanges:
            first_look = True
            while True:
                g, _, chain = _g_at(chain, p, p, frontier, haps, logs, offsets, penalty, stored,
                                    stored_cnt, gperm, permuted, lexmode, acc)
                gains = exchange_gains(f, np.ascontiguousarray(g), ident, perms)
                best = int(np.argmax(gains))
                if first_look:
                    junction_gain[p - j0] = gains[best]
                    junction_pair[p - j0, 0] = pa[best]
                    junction_pair[p - j0, 1] = pb[best]
                    first_look = False
                if gains[best] <= tol:
                    break
                a, b = pa[best], pb[best]
                for x in range(p, blocks):
                    t = X[a, x]
                    X[a, x] = X[b, x]
                    X[b, x] = t
                for j in range(offsets[p], total_sites):
                    t8 = haps[a, j]
                    haps[a, j] = haps[b, j]
                    haps[b, j] = t8
                gperm = gperm[perms[best]]
                permuted = True
                for m in range(founders):
                    if lab[m] == a:
                        lab[m] = b
                    elif lab[m] == b:
                        lab[m] = a
                chain = _relabel_chain(chain, p, perms[best], lexmode)
                ahead = (ahead[0], ahead[1], min(ahead[2], p), ahead[3], ahead[4], ahead[5], ahead[6],
                         ahead[7], ahead[8], ahead[9])
                acc[_A_EXCHANGE] += 1
                acc[_A_EXCHANGE_GAIN] += gains[best]
                moves, move_gain = _grow_moves(moves, move_gain, n_moves)
                moves[n_moves, 0] = MOVE_EXCHANGE
                moves[n_moves, 1] = p
                moves[n_moves, 6] = a
                moves[n_moves, 7] = b
                move_gain[n_moves] = gains[best]
                n_moves += 1
                accepted_here = True
        c0, c1 = cand_start[p], cand_start[p + 1]
        if c1 > c0:
            n_here = c1 - c0
            accepted = np.zeros(n_here, np.bool_)
            evaluated = np.zeros(n_here, np.bool_)
            while True:
                it_cand = np.empty(n_here, np.int64)
                it_kind = np.empty(n_here, np.int64)
                it_k = np.empty(n_here, np.int64)
                it_p1 = np.empty(n_here, np.int64)
                it_p2 = np.empty(n_here, np.int64)
                it_rs = np.empty(n_here, np.int64)
                n_act = _active_items(p, X, lab, cand_start, cand_kind, cand_end, cand_k0,
                                      cand_row_start, cand_rows, accepted, it_cand, it_kind, it_k,
                                      it_p1, it_p2, it_rs)
                if n_act == 0:
                    break
                act_gain, act_switch, act_gains_s, act_switches_s, chain, ahead = _score_items(
                    n_act, it_kind, it_k, it_p1, it_p2, it_rs, cand_rows, p, frontier, haps, leaves,
                    logs, offsets, penalty, stored, stored_cnt, stored_scale, gperm, permuted, lexmode,
                    f, fcnt, chain, ahead, acc, ahead_blocks)
                # the best guard-passing candidate: decreasing gain, earliest on ties
                order = np.empty(n_act, np.int64)
                n_ranked = 0
                for i in range(n_act):
                    acc[_A_EVALUATED] += 1
                    evaluated[it_cand[i] - c0] = True
                    if act_gain[i] > tol:
                        order[n_ranked] = i
                        n_ranked += 1
                for i in range(1, n_ranked):
                    j = i
                    while j > 0:
                        u, v = order[j - 1], order[j]
                        if -act_gain[u] > -act_gain[v] or (-act_gain[u] == -act_gain[v] and u > v):
                            order[j - 1], order[j] = v, u
                            j -= 1
                        else:
                            break
                chosen = -1
                for r in range(n_ranked):
                    i = order[r]
                    gain = act_gain[i]
                    if lexmode and it_kind[i] == MOVE_COARSE and (
                            gain + penalty * act_switch[i] < -guard_tolerance):
                        # the move lowers the genotype fit E = F + penalty x switches
                        relaxed = False
                        if shared_guard:
                            acc[_A_GUARD_TESTS] += 1
                            shared, chain, ahead = _uncredited(
                                haps, leaves, logs, offsets, penalty, p, 0, it_k[i], it_p1[i],
                                it_p2[i], cand_rows, it_rs[i], act_gains_s[i], act_switches_s[i],
                                frontier, placeholder, placeholder_cnt, local, local_cnt, stored,
                                stored_cnt, gperm, permuted, f, fcnt, first, second, chain, ahead, acc,
                                ahead_blocks)
                            relaxed = gain - penalty * shared > tol
                            if not relaxed:
                                acc[_A_GUARD_SWITCHES] += shared
                        if not relaxed:
                            acc[_A_GUARD_REJECTED] += 1
                            continue
                        acc[_A_GUARD_RELAXED] += 1
                    chosen = i
                    break
                if chosen < 0:
                    break
                i = chosen
                gain = act_gain[i]
                accepted[it_cand[i] - c0] = True
                k, p1, p2 = it_k[i], it_p1[i], it_p2[i]
                for x in range(p1, p2):
                    X[k, x] = cand_rows[it_rs[i] + x - p1]
                _set_rows(haps, leaves, offsets, k, p1, cand_rows, it_rs[i], it_rs[i] + p2 - p1)
                if p2 <= frontier:
                    chain = _chain_from(chain, p2, lexmode)
                else:
                    frontier = p2
                    chain = (False, chain[1], chain[2], chain[3], chain[4], chain[5])
                ahead = (ahead[0], ahead[1], min(ahead[2], p1), ahead[3], ahead[4], ahead[5], ahead[6],
                         ahead[7], ahead[8], ahead[9])
                if it_kind[i] == MOVE_COARSE:
                    acc[_A_COARSE] += 1
                    acc[_A_COARSE_GAIN] += gain
                else:
                    acc[_A_FINE] += 1
                    acc[_A_FINE_GAIN] += gain
                acc[_A_CHANGED] += p2 - p1
                moves, move_gain = _grow_moves(moves, move_gain, n_moves)
                moves[n_moves, 0] = it_kind[i]
                moves[n_moves, 1] = p
                moves[n_moves, 2] = it_cand[i] - c0
                moves[n_moves, 3] = k
                moves[n_moves, 4] = p1
                moves[n_moves, 5] = p2
                move_gain[n_moves] = gain
                n_moves += 1
                accepted_here = True
            for i in range(n_here):
                if evaluated[i] and not accepted[i]:
                    acc[_A_FALSE_POSITIVE] += 1
        if accepted_here:
            pristine = False
        if p + 1 < blocks:
            if pristine:
                f = start_fwd[p + 1]
                if lexmode:
                    fcnt = start_fcnt[p + 1]
            elif ahead[0] and ahead[1] <= p + 1 <= ahead[2]:
                f = ahead[3][p + 1 - ahead[1]]
                if lexmode:
                    fcnt = ahead[4][p + 1 - ahead[1]]
            elif lexmode:
                f, fcnt = advance_lex(haps, logs, offsets, penalty, f, fcnt, p, p + 1)
            else:
                f = advance(haps, logs, offsets, penalty, f, p, p + 1)
        if accepted_here:
            accepted_any = True
    return (moves[:n_moves].copy(), move_gain[:n_moves].copy(), junction_gain, junction_pair, gperm,
            np.ascontiguousarray(f), np.ascontiguousarray(fcnt), frontier, accepted_any, ahead, chain)
