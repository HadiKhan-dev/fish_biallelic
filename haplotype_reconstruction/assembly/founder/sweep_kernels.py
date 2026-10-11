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
"""
import math

import numpy as np
from numba import njit, prange

from .scoring import _unordered_pairs

CENTER = math.log(1.0 / 3.0)


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


@njit(cache=True, parallel=True, nogil=True)
def messages(haps, logs, offsets, penalty):
    """Normalised forward (carried) and backward (stored) messages at every block boundary.

    Returns fwd, bwd with shape (B + 1, S, Q) and the per-block normalisers fscale,
    bscale with shape (B, S): fwd[b + 1] = (fwd[b] processed over block b) - fscale[b] and
    bwd[b] = (bwd[b + 1] processed backwards over block b) - bscale[b], so the
    unnormalised backward message at x is bwd[x] + sum_{b >= x} bscale[b].
    """
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples, blocks = logs.shape[0], len(offsets) - 1
    fwd = np.zeros((blocks + 1, samples, states))
    bwd = np.zeros((blocks + 1, samples, states))
    fscale = np.zeros((blocks, samples))
    bscale = np.zeros((blocks, samples))
    for s in prange(samples):
        row = np.zeros(states)
        for b in range(blocks):
            for j in range(offsets[b], offsets[b + 1]):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                sw = np.max(row) - penalty
                for q in range(states):
                    row[q] = max(row[q], sw) + _emission(
                        logs, s, j, haps[first[q], j] + haps[second[q], j])
            m = np.max(row)
            fscale[b, s] = m
            for q in range(states):
                row[q] -= m
                fwd[b + 1, s, q] = row[q]
        row[:] = 0.0
        for b in range(blocks - 1, -1, -1):
            for j in range(offsets[b + 1] - 1, offsets[b] - 1, -1):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                for q in range(states):
                    row[q] += _emission(logs, s, j, haps[first[q], j] + haps[second[q], j])
                sw = np.max(row) - penalty
                for q in range(states):
                    row[q] = max(row[q], sw)
            m = np.max(row)
            bscale[b, s] = m
            for q in range(states):
                row[q] -= m
                bwd[b, s, q] = row[q]
    return fwd, bwd, fscale, bscale


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
def messages_lex(haps, logs, offsets, penalty):
    """messages() plus the switch counts of the lexicographic (score, -switches) maxima."""
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples, blocks = logs.shape[0], len(offsets) - 1
    fwd = np.zeros((blocks + 1, samples, states))
    bwd = np.zeros((blocks + 1, samples, states))
    fcnt = np.zeros((blocks + 1, samples, states), np.int64)
    bcnt = np.zeros((blocks + 1, samples, states), np.int64)
    fscale = np.zeros((blocks, samples))
    bscale = np.zeros((blocks, samples))
    for s in prange(samples):
        row = np.zeros(states)
        cnt = np.zeros(states, np.int64)
        for b in range(blocks):
            for j in range(offsets[b], offsets[b + 1]):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                m, c = _lex_best(row, cnt)
                sw = m - penalty
                for q in range(states):
                    if sw > row[q] or (sw == row[q] and c + 1 < cnt[q]):
                        row[q] = sw
                        cnt[q] = c + 1
                    row[q] += _emission(logs, s, j, haps[first[q], j] + haps[second[q], j])
            m = np.max(row)
            fscale[b, s] = m
            for q in range(states):
                row[q] -= m
                fwd[b + 1, s, q] = row[q]
                fcnt[b + 1, s, q] = cnt[q]
        row[:] = 0.0
        cnt[:] = 0
        for b in range(blocks - 1, -1, -1):
            for j in range(offsets[b + 1] - 1, offsets[b] - 1, -1):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                for q in range(states):
                    row[q] += _emission(logs, s, j, haps[first[q], j] + haps[second[q], j])
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
    return fwd, bwd, fcnt, bcnt, fscale, bscale


@njit(cache=True, parallel=True, nogil=True)
def advance(haps, logs, offsets, penalty, start, a, c):
    """Carry normalised forward messages from boundary a over blocks [a, c)."""
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    out = np.empty((samples, states))
    for s in prange(samples):
        row = start[s].copy()
        for b in range(a, c):
            for j in range(offsets[b], offsets[b + 1]):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                sw = np.max(row) - penalty
                for q in range(states):
                    row[q] = max(row[q], sw) + _emission(
                        logs, s, j, haps[first[q], j] + haps[second[q], j])
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
    for s in prange(samples):
        row = start[s].copy()
        cnt = start_cnt[s].copy()
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
    for s in prange(samples):
        row = np.empty(states)
        for q in range(states):
            row[q] = end[s, gperm[q]]
        out[c - a, s] = row
        for b in range(c - 1, a - 1, -1):
            for j in range(offsets[b + 1] - 1, offsets[b] - 1, -1):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                for q in range(states):
                    row[q] += _emission(logs, s, j, haps[first[q], j] + haps[second[q], j])
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
                for q in range(states):
                    row[q] += _emission(logs, s, j, haps[first[q], j] + haps[second[q], j])
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
def _lex_best_lead(row, cnt):
    """_lex_best and the first state attaining the maximum score."""
    m = row[0]
    c = cnt[0]
    lead = 0
    for q in range(1, len(row)):
        if row[q] > m:
            m = row[q]
            c = cnt[q]
            lead = q
        elif row[q] == m and cnt[q] < c:
            c = cnt[q]
    return m, c, lead


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

    Also the chain's leaders (see candidate_gains): per site j of blocks [a, c) (index
    j - offsets[a]) the maximum of the message before the site's update and the first
    state attaining it (informative sites only), and per block the maximum before the
    block's normalisation and its first state.
    """
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    base = offsets[a]
    out = np.empty((c - a + 1, samples, states))
    site_max = np.empty((offsets[c] - base, samples))
    site_lead = np.empty((offsets[c] - base, samples), np.int32)
    block_max = np.empty((c - a, samples))
    block_lead = np.empty((c - a, samples), np.int32)
    for s in prange(samples):
        row = start[s].copy()
        out[0, s] = row
        for b in range(a, c):
            for j in range(offsets[b], offsets[b + 1]):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                top, lead = _max_lead(row)
                site_max[j - base, s] = top
                site_lead[j - base, s] = lead
                sw = top - penalty
                for q in range(states):
                    row[q] = max(row[q], sw) + _emission(
                        logs, s, j, haps[first[q], j] + haps[second[q], j])
            m, lead = _max_lead(row)
            block_max[b - a, s] = m
            block_lead[b - a, s] = lead
            for q in range(states):
                row[q] -= m
            out[b - a + 1, s] = row
    return out, site_max, site_lead, block_max, block_lead


@njit(cache=True, parallel=True, nogil=True)
def forward_chain_lex(haps, logs, offsets, penalty, start, start_cnt, a, c):
    """forward_chain with the switch counts (same messages and leaders)."""
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    base = offsets[a]
    out = np.empty((c - a + 1, samples, states))
    out_cnt = np.empty((c - a + 1, samples, states), np.int64)
    site_max = np.empty((offsets[c] - base, samples))
    site_lead = np.empty((offsets[c] - base, samples), np.int32)
    block_max = np.empty((c - a, samples))
    block_lead = np.empty((c - a, samples), np.int32)
    for s in prange(samples):
        row = start[s].copy()
        cnt = start_cnt[s].copy()
        out[0, s] = row
        out_cnt[0, s] = cnt
        for b in range(a, c):
            for j in range(offsets[b], offsets[b + 1]):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                m, k, lead = _lex_best_lead(row, cnt)
                site_max[j - base, s] = m
                site_lead[j - base, s] = lead
                sw = m - penalty
                for q in range(states):
                    if sw > row[q] or (sw == row[q] and k + 1 < cnt[q]):
                        row[q] = sw
                        cnt[q] = k + 1
                    row[q] += _emission(logs, s, j, haps[first[q], j] + haps[second[q], j])
            m, lead = _max_lead(row)
            block_max[b - a, s] = m
            block_lead[b - a, s] = lead
            for q in range(states):
                row[q] -= m
            out[b - a + 1, s] = row
            out_cnt[b - a + 1, s] = cnt
    return out, out_cnt, site_max, site_lead, block_max, block_lead


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
def _candidate_dense(haps, leaves, logs, offsets, penalty, f_start, g_start, focal, first_block,
                     end_block, rows, row_offsets, g_end, scale_sum, first, second, i, s):
    """One candidate's exact gain for one sample: every state over every site."""
    states = len(first)
    k = focal[i]
    old = -np.inf
    row = np.empty(states)
    for q in range(states):
        old = max(old, f_start[i, s, q] + g_start[i, s, q])
        row[q] = f_start[i, s, q]
    scale = 0.0
    start = first_block[i]
    for b in range(start, end_block[i]):
        r = rows[row_offsets[i] + b - start]
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
    best = -np.inf
    for q in range(states):
        best = max(best, row[q] + g_end[i, s, q])
    return best + scale - old - scale_sum[i, s]


@njit(cache=True, parallel=True, nogil=True)
def candidate_gains(haps, leaves, logs, offsets, penalty, f_start, g_start, focal, first_block,
                    end_block, rows, row_offsets, g_end, scale_sum, site_max, site_lead, block_max,
                    block_lead, chain_block, cur_end):
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
    panel's forward chain from block chain_block (forward_chain: its leaders site_max /
    site_lead per site, block_max / block_lead per block, and cur_end[i], its message at
    end_block[i]) gives that term. While the chain's leader is a background state and
    the new focal states do not exceed it, the switch terms and block normalisers are the
    chain's, so the background states equal the chain's and only the focal states are
    advanced: O(K) instead of O(Q) per site, the same arithmetic. A sample whose leader
    contains k, or whose new focal states would lead, takes the dense recursion
    (candidate_gains_reference). Bitwise the reference's gains.
    """
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    founders = len(haps)
    samples = logs.shape[0]
    count = len(focal)
    gains = np.empty((count, samples))
    # States containing founder k, per k.
    focal_states = np.empty((founders, founders), np.int64)
    filled = np.zeros(founders, np.int64)
    for q in range(states):
        focal_states[first[q], filled[first[q]]] = q
        filled[first[q]] += 1
        if second[q] != first[q]:
            focal_states[second[q], filled[second[q]]] = q
            filled[second[q]] += 1
    site_base = offsets[chain_block]
    for task in prange(count * samples):
        i, s = task // samples, task % samples
        k = focal[i]
        fs = focal_states[k]
        xf = np.empty(founders)
        for t in range(founders):
            xf[t] = f_start[i, s, fs[t]]
        scale = 0.0
        start = first_block[i]
        fast = True
        for b in range(start, end_block[i]):
            r = rows[row_offsets[i] + b - start]
            leaf = leaves[b]
            base = offsets[b]
            for j in range(offsets[b], offsets[b + 1]):
                if logs[s, j, 0] == 0.0 and logs[s, j, 1] == 0.0 and logs[s, j, 2] == 0.0:
                    continue
                lead = site_lead[j - site_base, s]
                top = site_max[j - site_base, s]
                if first[lead] == k or second[lead] == k:
                    fast = False
                    break
                for t in range(founders):
                    if xf[t] > top:
                        fast = False
                if not fast:
                    break
                sw = top - penalty
                for t in range(founders):
                    q = fs[t]
                    a = first[q]
                    e = second[q]
                    d = ((leaf[r, j - base] if a == k else haps[a, j])
                         + (leaf[r, j - base] if e == k else haps[e, j]))
                    xf[t] = max(xf[t], sw) + _emission(logs, s, j, d)
            if not fast:
                break
            lead = block_lead[b - chain_block, s]
            m = block_max[b - chain_block, s]
            if first[lead] == k or second[lead] == k:
                fast = False
                break
            for t in range(founders):
                if xf[t] > m:
                    fast = False
            if not fast:
                break
            scale += m
            for t in range(founders):
                xf[t] -= m
        if not fast:
            gains[i, s] = _candidate_dense(haps, leaves, logs, offsets, penalty, f_start, g_start,
                                           focal, first_block, end_block, rows, row_offsets, g_end,
                                           scale_sum, first, second, i, s)
            continue
        old = -np.inf
        for q in range(states):
            old = max(old, f_start[i, s, q] + g_start[i, s, q])
        best = -np.inf
        for q in range(states):
            if first[q] != k and second[q] != k:
                best = max(best, cur_end[i, s, q] + g_end[i, s, q])
        for t in range(founders):
            best = max(best, xf[t] + g_end[i, s, fs[t]])
        gains[i, s] = best + scale - old - scale_sum[i, s]
    return gains


@njit(cache=True, parallel=True, nogil=True)
def candidate_gains_lex(haps, leaves, logs, offsets, penalty, f_start, fcnt_start, g_start,
                        gcnt_start, focal, first_block, end_block, rows, row_offsets, g_end,
                        gcnt_end, scale_sum):
    """candidate_gains() plus the per-sample change of the lexicographic switch count."""
    first, second = _unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    count = len(focal)
    gains = np.empty((count, samples))
    switches = np.empty((count, samples), np.int64)
    for task in prange(count * samples):
        i, s = task // samples, task % samples
        k = focal[i]
        old = -np.inf
        old_cnt = 0
        row = np.empty(states)
        cnt = np.empty(states, np.int64)
        for q in range(states):
            v = f_start[i, s, q] + g_start[i, s, q]
            n = fcnt_start[i, s, q] + gcnt_start[i, s, q]
            if v > old or (v == old and n < old_cnt):
                old = v
                old_cnt = n
            row[q] = f_start[i, s, q]
            cnt[q] = fcnt_start[i, s, q]
        scale = 0.0
        start = first_block[i]
        for b in range(start, end_block[i]):
            r = rows[row_offsets[i] + b - start]
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
        best = -np.inf
        best_cnt = 0
        for q in range(states):
            v = row[q] + g_end[i, s, q]
            n = cnt[q] + gcnt_end[i, s, q]
            if v > best or (v == best and n < best_cnt):
                best = v
                best_cnt = n
        gains[i, s] = best + scale - old - scale_sum[i, s]
        switches[i, s] = best_cnt - old_cnt
    return gains, switches


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


@njit(cache=True, nogil=True)
def _exchange_sample(f, g, gperm, first, second, index, founders, out):
    """Per-sample exchange gains of every pair (the body of exchange_gains)."""
    states = len(f)
    terms = np.empty(states)
    base = -np.inf
    for q in range(states):
        terms[q] = f[q] + g[gperm[q]]
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
            best = max(best, f[index[a, a]] + g[gperm[index[b, b]]])
            best = max(best, f[index[b, b]] + g[gperm[index[a, a]]])
            for c in range(founders):
                if c != a and c != b:
                    best = max(best, f[index[a, c]] + g[gperm[index[b, c]]])
                    best = max(best, f[index[b, c]] + g[gperm[index[a, c]]])
            out[p] = best - base
            p += 1


@njit(cache=True, parallel=True, nogil=True)
def exchange_gains_batch(f, g, perms):
    """exchange_gains at several junctions at once: f, g of shape (J, S, Q), g already
    relabelled. Row j equals exchange_gains(f[j], g[j], identity, perms) bitwise (the same
    per-sample values, summed over samples in the same order)."""
    junctions, samples, states = f.shape
    npairs = perms.shape[0]
    founders = 1
    while founders * (founders + 1) // 2 < states:
        founders += 1
    first, second = _unordered_pairs(founders)
    index = np.empty((founders, founders), np.int64)
    for q in range(states):
        index[first[q], second[q]] = q
        index[second[q], first[q]] = q
    ident = np.arange(states)
    per = np.zeros((junctions, samples, npairs))
    for task in prange(junctions * samples):
        j = task // samples
        s = task % samples
        _exchange_sample(f[j, s], g[j, s], ident, first, second, index, founders, per[j, s])
    out = np.zeros((junctions, npairs))
    for j in range(junctions):
        for p in range(npairs):
            for s in range(samples):
                out[j, p] += per[j, s, p]
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
# kernels above (per-sample / per-(candidate, sample) parallel tasks); its own arithmetic
# is the former Python-driven pass's operation for operation, including NumPy's pairwise
# summation of per-sample gains (_numpy_sum), so its decisions are bitwise identical.
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
    scales)."""
    ok, start, end, values, counts, scales = chain
    if (not ok) or end != frontier or p < start:
        if lexmode:
            values, counts, scales = backward_chain_lex(haps, logs, offsets, penalty, stored[frontier],
                                                        stored_cnt[frontier], gperm, p, frontier)
        else:
            values, scales = backward_chain(haps, logs, offsets, penalty, stored[frontier], gperm, p,
                                            frontier)
        acc[_A_CHAIN] += frontier - p
        return (True, p, frontier, values, counts, scales)
    return chain


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
    site_lead, block_max, block_lead)."""
    ok, start, valid, values, counts, site_max, site_lead, block_max, block_lead = ahead
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
        nv, nc, nsm, nsl, nbm, nbl = forward_chain_lex(haps, logs, offsets, penalty, start_v, start_c,
                                                       a0, stop)
    else:
        nv, nsm, nsl, nbm, nbl = forward_chain(haps, logs, offsets, penalty, start_v, a0, stop)
        nc = counts
    acc[_A_AHEAD] += stop - a0
    if keep:
        k0, k1 = p - start, a0 - start
        base = offsets[start]
        s0, s1 = offsets[p] - base, offsets[a0] - base
        nv = np.concatenate((values[k0:k1], nv))
        if lexmode:
            nc = np.concatenate((counts[k0:k1], nc))
        nsm = np.concatenate((site_max[s0:s1], nsm))
        nsl = np.concatenate((site_lead[s0:s1], nsl))
        nbm = np.concatenate((block_max[k0:k1], nbm))
        nbl = np.concatenate((block_lead[k0:k1], nbl))
        return (True, p, stop, nv, nc, nsm, nsl, nbm, nbl)
    return (True, a0, stop, nv, nc, nsm, nsl, nbm, nbl)


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
                  accepted, it_cand, it_kind, it_k, it_p1, it_p2, it_rs, n_items):
    """Append the candidates anchored at junction q that differ from the current panel
    (skipping accepted[i] of the junction's i-th candidate) to the
    item arrays from position n_items; returns the new count."""
    for i in range(cand_start[q + 1] - cand_start[q]):
        if len(accepted) and accepted[i]:
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
        last = it_p1[members[0]]
        for i in members:
            last = max(last, it_p1[i])
        if not lex:
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
                first_block, end_block, flat, row_offsets, g_end, gc_end, sums)
            for m in range(n_mem):
                i = members[m]
                gain[i] = _numpy_sum(np.ascontiguousarray(gains2[m]))
                gains_s[i] = gains2[m]
                switch[i] = float(np.sum(switches2[m]))
                switches_s[i] = switches2[m]
        else:
            cur_end = np.empty((n_mem, samples, states))
            for m in range(n_mem):
                cur_end[m] = ahead[3][end_block[m] - a0]
            gains2 = candidate_gains(
                haps, leaves, logs, offsets, penalty, f_start, g_start, focal, first_block, end_block,
                flat, row_offsets, g_end, sums, ahead[5], ahead[6], ahead[7], ahead[8], a0, cur_end)
            for m in range(n_mem):
                i = members[m]
                gain[i] = _numpy_sum(np.ascontiguousarray(gains2[m]))
                gains_s[i] = gains2[m]
    return gain, switch, gains_s, switches_s, chain, ahead


@njit(cache=True, nogil=True)
def gs_pass_chunk(haps, X, leaves, logs, offsets, penalty, j0, j1, frontier0, f0, fcnt0, local, local_cnt,
                  stored, stored_cnt, stored_scale, gperm0, lab, cand_start, cand_kind, cand_end, cand_k0,
                  cand_row_start, cand_rows, perms, pa, pb, guard, exchanges, ahead_blocks, tol,
                  guard_tolerance, chunk, acc):
    """Junctions [j0, j1) of one exact Gauss-Seidel acceptance pass (converged_sweep._gs_pass),
    compiled: exchanges, then the content candidates anchored at every junction, the
    coarse-move guard, the lookahead forward chain and the backward chain from the
    frontier, all as the pass's specification.
    State in: frontier0, the carried forward message f0 (counts fcnt0) at j0, gperm0
    (current state -> stored state), lab (proposal founder label -> current row; modified
    in place, as haps and X), the guard's history `local` of boundaries 0 .. j0 (filled
    here for j0 .. j1 - 1). The candidates anchored at junction q are cand_start[q] ..
    cand_start[q + 1] - 1 in proposal order (kind MOVE_FINE / MOVE_COARSE, end block,
    founder label, rows cand_rows[cand_row_start[i]:] over [q, end)). guard: GUARD_NONE /
    GUARD_GENOTYPE_FIT / GUARD_SHARED_JUNCTION; arrays of switch counts are one-element
    placeholders when unused. acc (ACC_KEYS) is accumulated in place.
    chunk > 0 (a speculative chunk of j1 - j0 junctions): the first look at every junction
    (its exchange gains and its candidates' gains) depends only on the panel at j0 until
    something is accepted, so it is computed for all of them at once by the same kernels
    on the same messages (bitwise the junction-by-junction values); the call returns after
    the first junction that accepts a move (the rest of the chunk is discarded). chunk = 0:
    the junctions are evaluated one by one. The lookahead and backward-chain caches start
    empty in every call, which changes no value.
    Returns (moves (n, 8) int64 rows (kind, p, index, row, p1, p2, a, b), move gains,
    junction_gain and junction_pair of the junctions done, gperm, the forward message and
    counts at the next junction, frontier, the next junction, whether a move was accepted)."""
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
    chain = (False, 0, 0, np.zeros((1, 1, 1)), np.zeros((1, 1, 1), np.int64), np.zeros((1, 1)))
    ahead = (False, 0, -1, np.zeros((1, 1, 1)), np.zeros((1, 1, 1), np.int64), np.zeros((1, 1)),
             np.zeros((1, 1), np.int32), np.zeros((1, 1)), np.zeros((1, 1), np.int32))
    # speculative chunk: junctions [spec_q0, spec_q1); exchange gains of junctions
    # max(spec_q0, 1) .. spec_q1 - 1; candidate values by slot cand - cand_start[spec_q0]
    spec_ok = False
    spec_q0 = 0
    spec_q1 = 0
    spec_exch = np.zeros((1, 1))
    spec_gain = np.zeros(1)
    spec_switch = np.zeros(1)
    spec_gains_s = np.zeros((1, 1))
    spec_switches_s = np.zeros((1, 1), np.int64)
    f = f0
    fcnt = fcnt0
    empty_flags = np.zeros(0, np.bool_)
    placeholder = np.zeros((1, 1, 1))
    placeholder_cnt = np.zeros((1, 1, 1), np.int64)
    p_next = j1
    accepted_any = False
    for p in range(j0, j1):
        if shared_guard:
            local[p] = f
            local_cnt[p] = fcnt
        accepted_here = False
        if chunk > 0 and p == j0:
            # first looks of junctions [p, q1) against the current panel
            q1 = j1
            n_tot = cand_start[q1] - cand_start[p]
            it_cand = np.empty(n_tot, np.int64)
            it_kind = np.empty(n_tot, np.int64)
            it_k = np.empty(n_tot, np.int64)
            it_p1 = np.empty(n_tot, np.int64)
            it_p2 = np.empty(n_tot, np.int64)
            it_rs = np.empty(n_tot, np.int64)
            n_items = 0
            for q in range(p, q1):
                n_items = _active_items(q, X, lab, cand_start, cand_kind, cand_end, cand_k0,
                                        cand_row_start, cand_rows, empty_flags, it_cand, it_kind, it_k,
                                        it_p1, it_p2, it_rs, n_items)
            need = q1
            for i in range(n_items):
                need = max(need, it_p1[i])
            ahead = _forward_to(ahead, p, min(blocks, need), f, fcnt, haps, logs, offsets, penalty,
                                lexmode, ahead_blocks, acc)
            if exchanges and q1 > max(p, 1):
                jq0 = max(p, 1)
                fb = np.empty((q1 - jq0, samples, states))
                gb = np.empty((q1 - jq0, samples, states))
                for q in range(jq0, q1):
                    fb[q - jq0] = ahead[3][q - ahead[1]]
                    g, gc, chain = _g_at(chain, q, p, frontier, haps, logs, offsets, penalty, stored,
                                         stored_cnt, gperm, permuted, lexmode, acc)
                    gb[q - jq0] = g
                spec_exch = exchange_gains_batch(fb, gb, perms)
            if n_items:
                g1, s1, gs1, ss1, chain, ahead = _score_items(
                    n_items, it_kind, it_k, it_p1, it_p2, it_rs, cand_rows, p, frontier, haps, leaves,
                    logs, offsets, penalty, stored, stored_cnt, stored_scale, gperm, permuted, lexmode, f,
                    fcnt, chain, ahead, acc, ahead_blocks)
                spec_gain = np.zeros(max(1, n_tot))
                spec_switch = np.zeros(max(1, n_tot))
                spec_gains_s = np.zeros((max(1, n_tot), samples))
                spec_switches_s = np.zeros((max(1, n_tot), samples), np.int64)
                for i in range(n_items):
                    slot = it_cand[i] - cand_start[p]
                    spec_gain[slot] = g1[i]
                    spec_switch[slot] = s1[i]
                    spec_gains_s[slot] = gs1[i]
                    spec_switches_s[slot] = ss1[i]
            spec_ok = True
            spec_q0 = p
            spec_q1 = q1
        if p >= 1 and exchanges:
            first_look = True
            while True:
                if first_look and spec_ok:
                    gains = spec_exch[p - max(spec_q0, 1)].copy()
                else:
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
                chain = (False, chain[1], chain[2], chain[3], chain[4], chain[5])
                ahead = (ahead[0], ahead[1], min(ahead[2], p), ahead[3], ahead[4], ahead[5], ahead[6],
                         ahead[7], ahead[8])
                acc[_A_EXCHANGE] += 1
                acc[_A_EXCHANGE_GAIN] += gains[best]
                moves, move_gain = _grow_moves(moves, move_gain, n_moves)
                moves[n_moves, 0] = MOVE_EXCHANGE
                moves[n_moves, 1] = p
                moves[n_moves, 6] = a
                moves[n_moves, 7] = b
                move_gain[n_moves] = gains[best]
                n_moves += 1
                spec_ok = False
                accepted_here = True
        c0, c1 = cand_start[p], cand_start[p + 1]
        if c1 > c0:
            n_here = c1 - c0
            accepted = np.zeros(n_here, np.bool_)
            evaluated = np.zeros(n_here, np.bool_)
            first_eval = True
            while True:
                it_cand = np.empty(n_here, np.int64)
                it_kind = np.empty(n_here, np.int64)
                it_k = np.empty(n_here, np.int64)
                it_p1 = np.empty(n_here, np.int64)
                it_p2 = np.empty(n_here, np.int64)
                it_rs = np.empty(n_here, np.int64)
                n_act = _active_items(p, X, lab, cand_start, cand_kind, cand_end, cand_k0,
                                      cand_row_start, cand_rows, accepted, it_cand, it_kind, it_k,
                                      it_p1, it_p2, it_rs, 0)
                if n_act == 0:
                    break
                if first_eval and spec_ok:
                    act_gain = np.empty(n_act)
                    act_switch = np.empty(n_act)
                    act_gains_s = np.empty((n_act, samples))
                    act_switches_s = np.empty((n_act, samples), np.int64)
                    for i in range(n_act):
                        slot = it_cand[i] - cand_start[spec_q0]
                        act_gain[i] = spec_gain[slot]
                        act_switch[i] = spec_switch[slot]
                        act_gains_s[i] = spec_gains_s[slot]
                        act_switches_s[i] = spec_switches_s[slot]
                else:
                    act_gain, act_switch, act_gains_s, act_switches_s, chain, ahead = _score_items(
                        n_act, it_kind, it_k, it_p1, it_p2, it_rs, cand_rows, p, frontier, haps, leaves,
                        logs, offsets, penalty, stored, stored_cnt, stored_scale, gperm, permuted, lexmode,
                        f, fcnt, chain, ahead, acc, ahead_blocks)
                first_eval = False
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
                frontier = max(frontier, p2)
                chain = (False, chain[1], chain[2], chain[3], chain[4], chain[5])
                ahead = (ahead[0], ahead[1], min(ahead[2], p1), ahead[3], ahead[4], ahead[5], ahead[6],
                         ahead[7], ahead[8])
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
                spec_ok = False
                accepted_here = True
            for i in range(n_here):
                if evaluated[i] and not accepted[i]:
                    acc[_A_FALSE_POSITIVE] += 1
        if p + 1 < blocks:
            if ahead[0] and ahead[1] <= p + 1 <= ahead[2]:
                f = ahead[3][p + 1 - ahead[1]]
                if lexmode:
                    fcnt = ahead[4][p + 1 - ahead[1]]
            elif lexmode:
                f, fcnt = advance_lex(haps, logs, offsets, penalty, f, fcnt, p, p + 1)
            else:
                f = advance(haps, logs, offsets, penalty, f, p, p + 1)
        if accepted_here:
            accepted_any = True
            if chunk > 0:
                p_next = p + 1
                break
    done = p_next - j0
    return (moves[:n_moves].copy(), move_gain[:n_moves].copy(), junction_gain[:done].copy(),
            junction_pair[:done].copy(), gperm, np.ascontiguousarray(f), np.ascontiguousarray(fcnt),
            frontier, p_next, accepted_any)
