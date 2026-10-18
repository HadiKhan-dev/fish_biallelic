"""Exact F gains of one-founder content moves and of founder-pair suffix exchanges.

A content move replaces founder k's rows on blocks [p1, p2); its exact per-sample gain is
max_q [f(p1) (x) T_new[p1, p2) (x) g(p2)] + new normalisers - max_q [f(p1) + g(p1)] - old
normalisers, from the current messages at its two ends (messages.py). A suffix exchange of
founders a < b at a junction relabels the suffix states, so its gain needs the messages at the
junction only.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .messages import segments
from .objective import (forward_sites, forward_sites_lex, is_skipped, lex_switch, normalise, pick,
                        site_emissions, unordered_pairs)


@njit(cache=True, nogil=True)
def pair_index(founders):
    """index[a, b] = index[b, a] = the state of the unordered pair {a, b}."""
    first, second = unordered_pairs(founders)
    index = np.empty((founders, founders), np.int64)
    for q in range(len(first)):
        index[first[q], second[q]] = q
        index[second[q], first[q]] = q
    return index


@njit(cache=True, nogil=True)
def exchange_pairs(founders):
    """The founder pairs a < b of the suffix exchanges, in exchange_gains' order."""
    pairs = founders * (founders - 1) // 2
    pa = np.empty(pairs, np.int64)
    pb = np.empty(pairs, np.int64)
    p = 0
    for a in range(founders):
        for b in range(a + 1, founders):
            pa[p] = a
            pb[p] = b
            p += 1
    return pa, pb


@njit(cache=True, nogil=True)
def transposition(founders, a, b):
    """perm[q] = the state of q after swapping founders a and b (applied to one accepted
    exchange; computed on demand instead of a K(K-1)/2 x Q table)."""
    first, second = unordered_pairs(founders)
    index = pair_index(founders)
    perm = np.empty(len(first), np.int64)
    for q in range(len(first)):
        i, j = first[q], second[q]
        i2 = b if i == a else (a if i == b else i)
        j2 = b if j == a else (a if j == b else j)
        perm[q] = index[i2, j2]
    return perm


@njit(cache=True, parallel=True, nogil=True)
def exchange_gains(f, g):
    """Exact dF of every founder-pair suffix exchange at one junction (O(S K^3)), in
    exchange_pairs' order. f: carried forward, g: current stored backward messages (S x Q).

    The exchange of founders a < b relabels the suffix states; sample s's junction value is
    max_q f[q] + g[perm(q)]. A state touching neither a nor b, and the state (a, b) itself, is
    fixed by the transposition, so the maximum over those states is the first of them in the
    per-sample descending order of t[q] = f[q] + g[q] (at most 2K - 1 entries are skipped); only
    the 2K - 2 states touching exactly one of a, b are evaluated swapped: (a, a) <-> (b, b) and
    (a, c) <-> (b, c). The maximum is over exactly the sums f[q] + g[perm(q)] of the all-state
    scan, so the gains equal the O(S K^4) scan's; per-sample differences are summed in sample
    order."""
    samples, states = f.shape
    founders = 1
    while founders * (founders + 1) // 2 < states:
        founders += 1
    npairs = founders * (founders - 1) // 2
    first, second = unordered_pairs(founders)
    index = pair_index(founders)
    per = np.zeros((samples, npairs))
    for s in prange(samples):
        terms = np.empty(states)
        base = -np.inf
        for q in range(states):
            terms[q] = f[s, q] + g[s, q]
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
                best = max(best, f[s, index[a, a]] + g[s, index[b, b]])
                best = max(best, f[s, index[b, b]] + g[s, index[a, a]])
                for c in range(founders):
                    if c != a and c != b:
                        best = max(best, f[s, index[a, c]] + g[s, index[b, c]])
                        best = max(best, f[s, index[b, c]] + g[s, index[a, c]])
                per[s, p] = best - base
                p += 1
    out = np.zeros(npairs)
    for p in range(npairs):
        for s in range(samples):
            out[p] += per[s, p]
    return out


@njit(cache=True, nogil=True)
def focal_states(first, second, founders):
    """focal[k]: the K states containing founder k (in state order)."""
    focal = np.empty((founders, founders), np.int64)
    filled = np.zeros(founders, np.int64)
    for q in range(len(first)):
        a, b = first[q], second[q]
        focal[a, filled[a]] = q
        filled[a] += 1
        if b != a:
            focal[b, filled[b]] = q
            filled[b] += 1
    return focal


@njit(cache=True, parallel=True, nogil=True)
def item_dosages(haps, leaves, offsets, k, rows, row_start, b0, b1, first, second):
    """Site-major pair dosages of one candidate on blocks [b0, b1): founder k's rows replaced by
    rows[row_start + b - b0] on block b, the other founders' current alleles (haps)."""
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


@njit(inline="always")
def dense_blocks(dos, lo, logs, offsets, penalty, row, b0, b1, s, scale):
    """The all-state recursion of sample s over blocks [b0, b1) on a candidate's dosage table, in
    place on row; returns scale plus the blocks' normalisers, added in block order."""
    for b in range(b0, b1):
        forward_sites(row, dos, lo, logs, s, offsets[b], offsets[b + 1], penalty)
        scale += normalise(row)
    return scale


@njit(inline="always")
def dense_blocks_lex(dos, lo, logs, offsets, penalty, row, cnt, b0, b1, s, scale):
    """dense_blocks() with the lexicographic switch counts."""
    for b in range(b0, b1):
        forward_sites_lex(row, cnt, dos, lo, logs, s, offsets[b], offsets[b + 1], penalty)
        scale += normalise(row)
    return scale


@njit(inline="always")
def max_close(row, g):
    """max over the states of row + g."""
    best = -np.inf
    for q in range(len(row)):
        best = max(best, row[q] + g[q])
    return best


@njit(inline="always")
def lex_close(row, cnt, g, gcnt):
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


@njit(cache=True, parallel=True, nogil=True)
def candidate_gains(haps, leaves, logs, offsets, penalty, f_start, g_start, focal, first_block,
                    end_block, rows, row_offsets, g_end, scale_sum, site_max, site_lead, block_max,
                    block_lead, chain_block, chain_values):
    """Exact per-sample F gains (count x S) of one-founder content moves.

    Candidate i replaces the rows of founder focal[i] on blocks [first_block[i], end_block[i])
    (rows[row_offsets[i]:row_offsets[i + 1]]); the current panel (haps) is unchanged elsewhere.
    With f_start[i] / g_start[i] the current normalised forward / backward messages at
    first_block[i], g_end[i] the current backward message at end_block[i] and scale_sum[i] the
    backward normalisers of those blocks, the gain of sample s is max_q [f_start (x) T_new (x)
    g_end] + new normalisers - max_q [f_start + g_start] - scale_sum.

    Only the K states containing the moved founder k see new emissions; the others (the
    background) change only through the switch term max(row) - penalty. The current panel's
    forward chain from block chain_block (messages.forward_chain: messages chain_values at block
    boundaries through every end_block[i], leaders site_max / site_lead per site, block_max /
    block_lead per block) gives that term. While the chain's leader is a background state and the
    new focal states do not exceed it, the switch terms and block normalisers are the chain's, so
    the background states equal the chain's and only the focal states advance: O(K) instead of
    O(Q) per site, the same arithmetic. Where a sample's leader contains k, or its new focal states
    would lead, inside block b, the all-state recursion resumes at boundary b from the chain's
    background states and the focal states held there (the same values as the all-state
    recursion from the start).

    Candidates are scored one after another, their samples in parallel, on per-segment dosage
    tables (item_dosages; a window candidate is normally one segment); each sample's state carries
    from segment to segment.
    """
    first, second = unordered_pairs(len(haps))
    founders = len(haps)
    states = len(first)
    samples = logs.shape[0]
    count = len(focal)
    gains = np.empty((count, samples))
    focal_all_states = focal_states(first, second, founders)
    site_base = offsets[chain_block]
    focal_all = np.empty((samples, founders))
    held_all = np.empty((samples, founders))
    row_all = np.empty((samples, states))
    scale_all = np.empty(samples)
    dense_all = np.zeros(samples, np.bool_)
    for i in range(count):
        k = focal[i]
        fs = focal_all_states[k]
        start, stop = first_block[i], end_block[i]
        bounds = segments(offsets[start:stop + 1], states)
        for u in range(len(bounds) - 1):
            b0, b1 = start + bounds[u], start + bounds[u + 1]
            lo = offsets[b0]
            dos = item_dosages(haps, leaves, offsets, k, rows, row_offsets[i] + b0 - start, b0, b1,
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
                            if is_skipped(logs, s, j):
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
                            e0, e1, e2 = site_emissions(logs, s, j)
                            for t in range(founders):
                                xf[t] = max(xf[t], sw) + pick(dos[j - lo, fs[t]], e0, e1, e2)
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
                        # The candidate's states at boundary b: background from the chain, focal
                        # held; the all-state recursion takes over from there.
                        if b == start:
                            row[:] = f_start[i, s]
                        else:
                            row[:] = chain_values[b - chain_block, s]
                        for t in range(founders):
                            row[fs[t]] = xf[t]
                        dense = True
                if dense:
                    scale = dense_blocks(dos, lo, logs, offsets, penalty, row, b, b1, s, scale)
                scale_all[s] = scale
                dense_all[s] = dense
                if b1 == stop:
                    old = max_close(f_start[i, s], g_start[i, s])
                    if not dense:
                        row[:] = chain_values[stop - chain_block, s]
                        for t in range(founders):
                            row[fs[t]] = xf[t]
                    gains[i, s] = max_close(row, g_end[i, s]) + scale - old - scale_sum[i, s]
    return gains


@njit(cache=True, parallel=True, nogil=True)
def candidate_gains_lex(haps, leaves, logs, offsets, penalty, f_start, fcnt_start, g_start,
                        gcnt_start, focal, first_block, end_block, rows, row_offsets, g_end,
                        gcnt_end, scale_sum, chain_values, chain_counts, site_max, site_lead,
                        block_max, block_lead, site_cnt, chain_block, chain_end):
    """candidate_gains() plus the per-sample change of the lexicographic switch count.

    The background fast path for the lexicographic pair (score, switch count): the current
    panel's lexicographic forward chain from block chain_block (messages.forward_chain_lex;
    first_block[i] >= chain_block) has messages and counts at boundaries chain_block..chain_end,
    per site the lexicographic maximum (site_max, site_cnt) before the site's update and the first
    state attaining it (site_lead), per block the maximum before the normalisation and its first
    state. While the site leader is a background state (not containing the moved founder k) and no
    new focal state exceeds its score or ties it with fewer switches, the candidate's
    lexicographic maximum is the chain's, so the switch terms, the background states' scores and
    counts are the chain's; while the block leader is a background state and no focal state
    exceeds the block maximum, so is the block normaliser. Only the K focal states are then
    advanced. Where this fails inside block b (or the chain ends at b), the all-state recursion
    resumes at boundary b from the chain's background states and the focal states held there.
    Every value is the all-state recursion's. Returns (gains, switch-count changes), count x S.
    """
    first, second = unordered_pairs(len(haps))
    founders = len(haps)
    states = len(first)
    samples = logs.shape[0]
    count = len(focal)
    gains = np.empty((count, samples))
    switches = np.empty((count, samples), np.int64)
    focal_all_states = focal_states(first, second, founders)
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
        fs = focal_all_states[k]
        start, stop = first_block[i], end_block[i]
        bounds = segments(offsets[start:stop + 1], states)
        for u in range(len(bounds) - 1):
            b0, b1 = start + bounds[u], start + bounds[u + 1]
            lo = offsets[b0]
            dos = item_dosages(haps, leaves, offsets, k, rows, row_offsets[i] + b0 - start, b0, b1,
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
                            if is_skipped(logs, s, j):
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
                            e0, e1, e2 = site_emissions(logs, s, j)
                            for t in range(founders):
                                lex_switch(xf, xc, t, sw, c)
                                xf[t] += pick(dos[j - lo, fs[t]], e0, e1, e2)
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
                    scale = dense_blocks_lex(dos, lo, logs, offsets, penalty, row, cnt, b, b1, s,
                                             scale)
                scale_all[s] = scale
                dense_all[s] = dense
                if b1 == stop:
                    old, old_cnt = lex_close(f_start[i, s], fcnt_start[i, s], g_start[i, s],
                                             gcnt_start[i, s])
                    if not dense:
                        row[:] = chain_values[stop - chain_block, s]
                        cnt[:] = chain_counts[stop - chain_block, s]
                        for t in range(founders):
                            row[fs[t]] = xf[t]
                            cnt[fs[t]] = xc[t]
                    best, best_cnt = lex_close(row, cnt, g_end[i, s], gcnt_end[i, s])
                    gains[i, s] = best + scale - old - scale_sum[i, s]
                    switches[i, s] = best_cnt - old_cnt
    return gains, switches
