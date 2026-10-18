"""Exact normalised block-boundary messages of the objective F (objective.py).

For every sample F_s is a max-plus chain, so for any panel edit confined to blocks [a, c) the exact
change of F is

    dF = sum_s [max_q (f_s(a) (x) T_new[a, c) (x) g_s(c))_q - max_q (f_s(a) (x) T_old[a, c) (x) g_s(c))_q]

with f the carried forward message (switch-then-emission per informative site) and g the stored
backward message (emission-then-switch), both normalised to maximum 0 at every block boundary
(the per-block normalisers are kept: fscale, bscale). A suffix exchange of two founders relabels
the suffix states only, so its gain needs the two messages at its junction alone.

The optional switch counts form a lexicographic max-plus pair (score first, then fewer switches
among exact score ties). They give the genotype-fit term E = F + penalty x switches of a
maximum-score path, decomposed exactly like F; only the coarse-move guard reads them. They never
change F.

The chromosome-wide kernels compute the samples of a range [s0, s1) into arrays the caller
allocates (every sample's recursion is its own, so a search runs them range by range, re-reading
its thread share between ranges) and build the shared pair-dosage table per segment of whole
blocks of at most DOSAGE_BYTES, so it stays small at large K. All kernels use the inline site
steps of objective.py: the forward recursion of messages(), its in-place update, advance() and the
forward chain perform the same operations, so their messages are bitwise equal (the updates' early
stops and the proposal cache rely on it).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .objective import (backward_sites, backward_sites_lex, forward_sites, forward_sites_lex,
                        is_skipped, lex_best, normalise, pair_dosages, row_max, site_emissions,
                        switch_then_emit, switch_then_emit_lex, unordered_pairs)

# Largest pair-dosage table (sites x states int8) of one segment of a chromosome-wide message
# kernel or of a candidate's blocks; a block larger than this forms its own segment.
DOSAGE_BYTES = 32 * 1024 ** 2


@njit(cache=True, nogil=True)
def segments(offsets, states):
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


def message_arrays(founders, blocks, samples, lex):
    """Zeroed message arrays (fwd, bwd, fcnt, bcnt, fscale, bscale): messages float64
    (B + 1, S, Q), switch counts int64 (B + 1, S, Q) only with lex (else None), per-block
    normalisers float64 (B, S)."""
    states = founders * (founders + 1) // 2
    counts = ((np.zeros((blocks + 1, samples, states), np.int64),
               np.zeros((blocks + 1, samples, states), np.int64)) if lex else (None, None))
    return (np.zeros((blocks + 1, samples, states)), np.zeros((blocks + 1, samples, states)),
            *counts, np.zeros((blocks, samples)), np.zeros((blocks, samples)))


@njit(cache=True, parallel=True, nogil=True)
def messages(haps, logs, offsets, penalty, s0, s1, fwd, bwd, fscale, bscale):
    """Normalised forward (carried) and backward (stored) messages at every block boundary of the
    samples [s0, s1), into zeroed arrays (message_arrays): fwd[b + 1] = (fwd[b] processed over
    block b) - fscale[b] and bwd[b] = (bwd[b + 1] processed backwards over block b) - bscale[b], so
    the unnormalised backward message at x is bwd[x] + sum_{b >= x} bscale[b]."""
    first, second = unordered_pairs(len(haps))
    bounds = segments(offsets, len(first))
    for t in range(len(bounds) - 1):
        b0, b1 = bounds[t], bounds[t + 1]
        lo = offsets[b0]
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = fwd[b0, s].copy()
            for b in range(b0, b1):
                forward_sites(row, dos, lo, logs, s, offsets[b], offsets[b + 1], penalty)
                fscale[b, s] = normalise(row)
                fwd[b + 1, s] = row
    for t in range(len(bounds) - 2, -1, -1):
        b0, b1 = bounds[t], bounds[t + 1]
        lo = offsets[b0]
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = bwd[b1, s].copy()
            for b in range(b1 - 1, b0 - 1, -1):
                backward_sites(row, dos, lo, logs, s, offsets[b], offsets[b + 1], penalty)
                bscale[b, s] = normalise(row)
                bwd[b, s] = row


@njit(cache=True, parallel=True, nogil=True)
def messages_lex(haps, logs, offsets, penalty, s0, s1, fwd, bwd, fcnt, bcnt, fscale, bscale):
    """messages() plus the switch counts of the lexicographic (score, -switches) maxima, of the
    samples [s0, s1), into zeroed arrays (message_arrays(..., lex=True))."""
    first, second = unordered_pairs(len(haps))
    bounds = segments(offsets, len(first))
    for t in range(len(bounds) - 1):
        b0, b1 = bounds[t], bounds[t + 1]
        lo = offsets[b0]
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = fwd[b0, s].copy()
            cnt = fcnt[b0, s].copy()
            for b in range(b0, b1):
                forward_sites_lex(row, cnt, dos, lo, logs, s, offsets[b], offsets[b + 1], penalty)
                fscale[b, s] = normalise(row)
                fwd[b + 1, s] = row
                fcnt[b + 1, s] = cnt
    for t in range(len(bounds) - 2, -1, -1):
        b0, b1 = bounds[t], bounds[t + 1]
        lo = offsets[b0]
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = bwd[b1, s].copy()
            cnt = bcnt[b1, s].copy()
            for b in range(b1 - 1, b0 - 1, -1):
                backward_sites_lex(row, cnt, dos, lo, logs, s, offsets[b], offsets[b + 1], penalty)
                bscale[b, s] = normalise(row)
                bwd[b, s] = row
                bcnt[b, s] = cnt


@njit(cache=True, parallel=True, nogil=True)
def update_messages(haps, logs, offsets, penalty, changed, fwd, bwd, fscale, bscale, backward_known,
                    s0, s1):
    """messages() of the current panel (haps) for the samples [s0, s1), in place on those of an
    earlier panel whose alleles differ only on the blocks where changed[b]: fwd, fscale always,
    bwd, bscale when backward_known (otherwise every block is computed into them; bwd[B] must
    hold zeros).

    A block's forward step reads only the message at its start and the block's alleles, so a
    sample's messages over an unchanged block whose start message is unchanged are the earlier
    ones. Per sample, a block is recomputed (messages()' recursion from the stored start message)
    only if it changed or its start message did; a recomputed end message that equals the earlier
    one ends the run. Likewise backwards. Bitwise messages() of the current panel."""
    first, second = unordered_pairs(len(haps))
    states = len(first)
    bounds = segments(offsets, states)
    # dirty[s - s0]: sample s's message at the current segment boundary changed.
    dirty = np.zeros(s1 - s0, np.bool_)
    for t in range(len(bounds) - 1):
        b0, b1 = bounds[t], bounds[t + 1]
        if not (dirty.any() or changed[b0:b1].any()):
            continue
        lo = offsets[b0]
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = np.empty(states)
            for b in range(b0, b1):
                if not (dirty[s - s0] or changed[b]):
                    continue
                row[:] = fwd[b, s]
                forward_sites(row, dos, lo, logs, s, offsets[b], offsets[b + 1], penalty)
                fscale[b, s] = normalise(row)
                same = True
                for q in range(states):
                    same = same and row[q] == fwd[b + 1, s, q]
                    fwd[b + 1, s, q] = row[q]
                dirty[s - s0] = not same
    dirty[:] = not backward_known
    for t in range(len(bounds) - 2, -1, -1):
        b0, b1 = bounds[t], bounds[t + 1]
        if not (dirty.any() or changed[b0:b1].any()):
            continue
        lo = offsets[b0]
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = np.empty(states)
            for b in range(b1 - 1, b0 - 1, -1):
                if not (dirty[s - s0] or changed[b]):
                    continue
                row[:] = bwd[b + 1, s]
                backward_sites(row, dos, lo, logs, s, offsets[b], offsets[b + 1], penalty)
                bscale[b, s] = normalise(row)
                same = backward_known
                for q in range(states):
                    same = same and row[q] == bwd[b, s, q]
                    bwd[b, s, q] = row[q]
                dirty[s - s0] = not same


@njit(cache=True, parallel=True, nogil=True)
def update_messages_lex(haps, logs, offsets, penalty, changed, fwd, bwd, fcnt, bcnt, fscale, bscale,
                        backward_known, s0, s1):
    """update_messages() for messages_lex(): the messages and their switch counts (a run ends where
    both equal the earlier ones), samples [s0, s1). Bitwise messages_lex() of the current panel."""
    first, second = unordered_pairs(len(haps))
    states = len(first)
    bounds = segments(offsets, states)
    dirty = np.zeros(s1 - s0, np.bool_)
    for t in range(len(bounds) - 1):
        b0, b1 = bounds[t], bounds[t + 1]
        if not (dirty.any() or changed[b0:b1].any()):
            continue
        lo = offsets[b0]
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = np.empty(states)
            cnt = np.empty(states, np.int64)
            for b in range(b0, b1):
                if not (dirty[s - s0] or changed[b]):
                    continue
                row[:] = fwd[b, s]
                cnt[:] = fcnt[b, s]
                forward_sites_lex(row, cnt, dos, lo, logs, s, offsets[b], offsets[b + 1], penalty)
                fscale[b, s] = normalise(row)
                same = True
                for q in range(states):
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
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = np.empty(states)
            cnt = np.empty(states, np.int64)
            for b in range(b1 - 1, b0 - 1, -1):
                if not (dirty[s - s0] or changed[b]):
                    continue
                row[:] = bwd[b + 1, s]
                cnt[:] = bcnt[b + 1, s]
                backward_sites_lex(row, cnt, dos, lo, logs, s, offsets[b], offsets[b + 1], penalty)
                bscale[b, s] = normalise(row)
                same = backward_known
                for q in range(states):
                    same = same and row[q] == bwd[b, s, q] and cnt[q] == bcnt[b, s, q]
                    bwd[b, s, q] = row[q]
                    bcnt[b, s, q] = cnt[q]
                dirty[s - s0] = not same


@njit(cache=True, parallel=True, nogil=True)
def reverse_sites(logs):
    """A C-contiguous copy of logs (S x L x 3) with the site axis reversed (the reversed
    orientation of a search), copied in parallel over samples."""
    samples, sites, genotypes = logs.shape
    out = np.empty_like(logs)
    for s in prange(samples):
        for j in range(sites):
            for g in range(genotypes):
                out[s, sites - 1 - j, g] = logs[s, j, g]
    return out


@njit(cache=True, parallel=True, nogil=True)
def advance(haps, logs, offsets, penalty, start, a, c):
    """Carry the normalised forward messages start (S x Q) from boundary a over blocks [a, c)."""
    first, second = unordered_pairs(len(haps))
    samples = logs.shape[0]
    out = np.empty((samples, len(first)))
    lo = offsets[a]
    dos = pair_dosages(haps, first, second, lo, offsets[c])
    for s in prange(samples):
        row = start[s].copy()
        for b in range(a, c):
            forward_sites(row, dos, lo, logs, s, offsets[b], offsets[b + 1], penalty)
            normalise(row)
        out[s] = row
    return out


@njit(cache=True, parallel=True, nogil=True)
def advance_lex(haps, logs, offsets, penalty, start, start_cnt, a, c):
    """advance() with the lexicographic switch counts."""
    first, second = unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    out = np.empty((samples, states))
    out_cnt = np.empty((samples, states), np.int64)
    lo = offsets[a]
    dos = pair_dosages(haps, first, second, lo, offsets[c])
    for s in prange(samples):
        row = start[s].copy()
        cnt = start_cnt[s].copy()
        for b in range(a, c):
            forward_sites_lex(row, cnt, dos, lo, logs, s, offsets[b], offsets[b + 1], penalty)
            normalise(row)
        out[s] = row
        out_cnt[s] = cnt
    return out, out_cnt


@njit(cache=True, parallel=True, nogil=True)
def backward_chain(haps, logs, offsets, penalty, end, gperm, a, c):
    """Exact stored-convention backward messages at boundaries a..c (index 0..c-a) and the
    per-block normalisers (index b - a for blocks a..c-1). end is the stored message at c, read
    through the label permutation gperm (current state -> stored state)."""
    first, second = unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    out = np.empty((c - a + 1, samples, states))
    scale = np.zeros((max(c - a, 0), samples))
    lo = offsets[a]
    dos = pair_dosages(haps, first, second, lo, offsets[c])
    for s in prange(samples):
        row = np.empty(states)
        for q in range(states):
            row[q] = end[s, gperm[q]]
        out[c - a, s] = row
        for b in range(c - 1, a - 1, -1):
            backward_sites(row, dos, lo, logs, s, offsets[b], offsets[b + 1], penalty)
            scale[b - a, s] = normalise(row)
            out[b - a, s] = row
    return out, scale


@njit(cache=True, parallel=True, nogil=True)
def backward_chain_lex(haps, logs, offsets, penalty, end, end_cnt, gperm, a, c):
    """backward_chain() with the lexicographic switch counts."""
    first, second = unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    out = np.empty((c - a + 1, samples, states))
    out_cnt = np.empty((c - a + 1, samples, states), np.int64)
    scale = np.zeros((max(c - a, 0), samples))
    lo = offsets[a]
    dos = pair_dosages(haps, first, second, lo, offsets[c])
    for s in prange(samples):
        row = np.empty(states)
        cnt = np.empty(states, np.int64)
        for q in range(states):
            row[q] = end[s, gperm[q]]
            cnt[q] = end_cnt[s, gperm[q]]
        out[c - a, s] = row
        out_cnt[c - a, s] = cnt
        for b in range(c - 1, a - 1, -1):
            backward_sites_lex(row, cnt, dos, lo, logs, s, offsets[b], offsets[b + 1], penalty)
            scale[b - a, s] = normalise(row)
            out[b - a, s] = row
            out_cnt[b - a, s] = cnt
    return out, out_cnt, scale


@njit(inline="always")
def max_lead(row):
    """Maximum of row and the first state attaining it (four interleaved running maxima with
    their first states, then combined: the pair a sequential scan finds)."""
    n = len(row)
    m0 = row[0]
    i0 = 0
    m1 = m0
    i1 = 0
    m2 = m0
    i2 = 0
    m3 = m0
    i3 = 0
    q = 1
    while q + 4 <= n:
        if row[q] > m0:
            m0 = row[q]
            i0 = q
        if row[q + 1] > m1:
            m1 = row[q + 1]
            i1 = q + 1
        if row[q + 2] > m2:
            m2 = row[q + 2]
            i2 = q + 2
        if row[q + 3] > m3:
            m3 = row[q + 3]
            i3 = q + 3
        q += 4
    while q < n:
        if row[q] > m0:
            m0 = row[q]
            i0 = q
        q += 1
    if m1 > m0 or (m1 == m0 and i1 < i0):
        m0 = m1
        i0 = i1
    if m2 > m0 or (m2 == m0 and i2 < i0):
        m0 = m2
        i0 = i2
    if m3 > m0 or (m3 == m0 and i3 < i0):
        m0 = m3
        i0 = i3
    return m0, i0


@njit(inline="always")
def lex_best_index(row, cnt):
    """lex_best() plus the first state attaining it (the switch source of a path); a sequential
    scan (interleaved running lexicographic maxima measured slower at every K from 6 to 30)."""
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
def forward_chain(haps, logs, offsets, penalty, start, a, c):
    """Carried forward messages at boundaries a..c (index 0..c-a) from start at a, with the
    chain's leaders (gains.candidate_gains): per informative site j of blocks [a, c) (sample major,
    index [s, j - offsets[a]]) the maximum of the message before the site's update and the first
    state attaining it, and per block the maximum before the block's normalisation and its first
    state."""
    first, second = unordered_pairs(len(haps))
    states = len(first)
    samples = logs.shape[0]
    base = offsets[a]
    out = np.empty((c - a + 1, samples, states))
    site_max = np.empty((samples, offsets[c] - base))
    site_lead = np.empty((samples, offsets[c] - base), np.int32)
    block_max = np.empty((c - a, samples))
    block_lead = np.empty((c - a, samples), np.int32)
    dos = pair_dosages(haps, first, second, base, offsets[c])
    for s in prange(samples):
        row = start[s].copy()
        out[0, s] = row
        for b in range(a, c):
            for j in range(offsets[b], offsets[b + 1]):
                if is_skipped(logs, s, j):
                    continue
                e0, e1, e2 = site_emissions(logs, s, j)
                top, lead = max_lead(row)
                site_max[s, j - base] = top
                site_lead[s, j - base] = lead
                switch_then_emit(row, top - penalty, dos, j - base, e0, e1, e2)
            m, lead = max_lead(row)
            block_max[b - a, s] = m
            block_lead[b - a, s] = lead
            for q in range(states):
                row[q] -= m
            out[b - a + 1, s] = row
    return out, site_max, site_lead, block_max, block_lead


@njit(cache=True, parallel=True, nogil=True)
def forward_chain_lex(haps, logs, offsets, penalty, start, start_cnt, a, c):
    """forward_chain with the switch counts: the same messages and block leaders, and per site the
    lexicographic maximum (score, switch count) of the message before the site's update
    (site_max, site_cnt) and the first state attaining both (site_lead: a state attaining the
    maximum score, as gains.candidate_gains requires)."""
    first, second = unordered_pairs(len(haps))
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
    dos = pair_dosages(haps, first, second, base, offsets[c])
    for s in prange(samples):
        row = start[s].copy()
        cnt = start_cnt[s].copy()
        out[0, s] = row
        out_cnt[0, s] = cnt
        for b in range(a, c):
            for j in range(offsets[b], offsets[b + 1]):
                if is_skipped(logs, s, j):
                    continue
                e0, e1, e2 = site_emissions(logs, s, j)
                m, k, lead = lex_best_index(row, cnt)
                site_max[s, j - base] = m
                site_lead[s, j - base] = lead
                site_cnt[s, j - base] = k
                switch_then_emit_lex(row, cnt, m - penalty, k, dos, j - base, e0, e1, e2)
            m, lead = max_lead(row)
            block_max[b - a, s] = m
            block_lead[b - a, s] = lead
            for q in range(states):
                row[q] -= m
            out[b - a + 1, s] = row
            out_cnt[b - a + 1, s] = cnt
    return out, out_cnt, site_max, site_lead, block_max, block_lead, site_cnt


@njit(cache=True, parallel=True, nogil=True)
def clamp_reverse(carried, penalty):
    """Stored messages of the reversed orientation from carried forward messages:
    out[B - c] = max(m, max(m) - penalty) of carried[c], renormalised."""
    nb, samples, states = carried.shape
    out = np.empty_like(carried)
    for c in prange(nb):
        for s in range(samples):
            m = row_max(carried[c, s])
            for q in range(states):
                out[nb - 1 - c, s, q] = max(carried[c, s, q], m - penalty) - m
    return out


@njit(cache=True, parallel=True, nogil=True)
def clamp_reverse_lex(carried, carried_cnt, penalty):
    """clamp_reverse() with the lexicographic switch counts."""
    nb, samples, states = carried.shape
    out = np.empty_like(carried)
    out_cnt = np.empty_like(carried_cnt)
    for c in prange(nb):
        for s in range(samples):
            m, k = lex_best(carried[c, s], carried_cnt[c, s])
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
