"""One exact Gauss-Seidel acceptance pass of a fixed-K search (compiled; search.py drives it).

The pass walks the junctions p = 0 .. B - 1 of the panel X in one orientation (search.Orientation:
forward or reversed site order). At every junction every founder-pair suffix exchange is scored
exactly from the carried forward and the current stored backward messages and the best is applied
while its gain exceeds TOL; then the content candidates anchored there (one founder's rows on
[p1, p2): the blocks where a window proposal differs from the current panel) are scored exactly
from the current messages at p1 and p2 (gains.py), best first, each applied while its gain exceeds
TOL; coarse candidates also pass the shared-junction guard (guard.py); then the forward message
advances one block.

Ties (release 2.1, design 5.3): "best" is the best tie class of the exact gains (anchored single
linkage with the refinement quantum q_R = tie_rules.quantum("refine_F", F), maths.ties rules: a
class chains consecutive gaps <= q; a class wider than 4q keeps value order), and inside a class
the content-smallest move: an exchange by its resulting pair of founder paths (each pair sorted,
compared lexicographically in the pass's block order), a content move by the moved founder's
resulting path, then its slot, then proposal order (2.0: first np.argmax in pair order; decreasing
gain, earliest on ties). The acceptance threshold TOL is a continuous threshold and unchanged; only
moves above it are ranked.

Invariants: (i) the current panel equals sigma(start panel) on every block at or after `frontier`
(the furthest right end of a block changed by an accepted content move), sigma the composition of
the accepted suffix exchanges: the stored backward messages (and their per-block normalisers) read
through gperm are exact there, and an exact backward chain from the frontier is recomputed from the
current alleles before it is needed (only where an edit changed it: after a content move left of
the frontier the chain right of the move's end stays exact, after a suffix exchange its states are
relabelled); (ii) the carried forward message f is exact at every junction because blocks behind
it never change again in this pass; a lookahead of current forward messages is kept only up to the
first block a later edit touched.

The pass runs as successive nogil calls of JUNCTION_BATCH junctions; before every call its Numba
team is re-read from the budget, so cores freed by finished fits reach the running ones. The
lookahead and backward-chain caches carry over from call to call. Decisions, order and results
never depend on the thread count or the batching.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
from numba import njit, prange

from .gains import candidate_gains, candidate_gains_lex, exchange_gains, exchange_pairs, transposition
from .guard import canonical_sites, path_switch_events, removed_events, shared_savings
from .messages import advance, advance_lex, backward_chain, backward_chain_lex, forward_chain, forward_chain_lex
from .objective import unordered_pairs
from ... import execution
from ..tie_rules import quantum as tie_quantum

# A move is accepted while its exact F gain exceeds TOL (log units).
TOL = 1e-4
# A coarse move "lowers E" when gain + penalty x switch change < -GUARD_TOLERANCE.
GUARD_TOLERANCE = 1e-6
# Forward messages of the current panel computed ahead of the pass position at a time.
AHEAD_BLOCKS = 16
# Junctions per nogil call (the Numba team is re-read between calls).
JUNCTION_BATCH = 16
# A tie class wider than this many quanta keeps value order (maths.ties.WIDE_CLASS_QUANTA).
WIDE_QUANTA = 4.0

MOVE_FINE, MOVE_COARSE = 1, 2
# Pass counters (the fit history's per-sweep record).
ACC_KEYS = ("fine", "coarse", "exchange", "fine_gain", "coarse_gain", "exchange_gain", "evaluated",
            "false_positive", "guard_rejected", "guard_shared_tests", "guard_relaxed",
            "guard_shared_switches", "guard_traced_blocks", "chain_blocks", "ahead_blocks",
            "changed_blocks", "exchange_tie", "content_tie", "tie_wide")
INTEGER_KEYS = frozenset(ACC_KEYS) - {"fine_gain", "coarse_gain", "exchange_gain"}
(_A_FINE, _A_COARSE, _A_EXCHANGE, _A_FINE_GAIN, _A_COARSE_GAIN, _A_EXCHANGE_GAIN, _A_EVALUATED,
 _A_FALSE_POSITIVE, _A_GUARD_REJECTED, _A_GUARD_TESTS, _A_GUARD_RELAXED, _A_GUARD_SWITCHES,
 _A_TRACED, _A_CHAIN, _A_AHEAD, _A_CHANGED, _A_EXCHANGE_TIE, _A_CONTENT_TIE,
 _A_TIE_WIDE) = range(len(ACC_KEYS))

# Exact current backward messages on boundaries [start, end] (with counts under the guard) and the
# normalisers of blocks [start, end); valid only when ok.
Chain = namedtuple("Chain", ["ok", "start", "end", "values", "counts", "scales"])
# Exact current forward messages on boundaries [start, valid] with the forward chain's leaders
# (messages.forward_chain(_lex)); counts and site_cnt only under the guard.
Ahead = namedtuple("Ahead", ["ok", "start", "valid", "values", "counts", "site_max", "site_lead",
                             "block_max", "block_lead", "site_cnt"])


# ------------------------------------------------------------------ message caches
@njit(cache=True, nogil=True)
def _stored_at(stored, x, gperm, permuted):
    if permuted:
        return np.ascontiguousarray(stored[x][:, gperm])
    return stored[x]


@njit(cache=True, nogil=True)
def _ensure_chain(chain, frontier, p, haps, logs, offsets, penalty, stored, stored_cnt, gperm,
                  guarded, acc):
    """The backward chain exact on boundaries [p, frontier]. A chain that still ends at the frontier
    but starts right of p is extended over blocks [p, start) from its exact value at start (the
    recursion from the frontier passes through that value), otherwise it is recomputed from the
    frontier."""
    if chain.ok and chain.end == frontier and p >= chain.start:
        return chain
    if chain.ok and chain.end == frontier:
        ident = np.arange(chain.values.shape[2])
        counts = chain.counts
        if guarded:
            head, head_cnt, head_scales = backward_chain_lex(haps, logs, offsets, penalty,
                                                             chain.values[0], chain.counts[0], ident,
                                                             p, chain.start)
            counts = np.concatenate((head_cnt[:-1], chain.counts))
        else:
            head, head_scales = backward_chain(haps, logs, offsets, penalty, chain.values[0], ident,
                                               p, chain.start)
        acc[_A_CHAIN] += chain.start - p
        return Chain(True, p, chain.end, np.concatenate((head[:-1], chain.values)), counts,
                     np.concatenate((head_scales, chain.scales)))
    counts = chain.counts
    if guarded:
        values, counts, scales = backward_chain_lex(haps, logs, offsets, penalty, stored[frontier],
                                                    stored_cnt[frontier], gperm, p, frontier)
    else:
        values, scales = backward_chain(haps, logs, offsets, penalty, stored[frontier], gperm, p,
                                        frontier)
    acc[_A_CHAIN] += frontier - p
    return Chain(True, p, frontier, values, counts, scales)


@njit(cache=True, nogil=True)
def _chain_from(chain, x, guarded):
    """The backward chain after an edit of blocks left of boundary x that leaves the frontier in
    place: its values and counts at boundaries >= x and the normalisers of blocks >= x depend only
    on blocks >= x and stay exact; boundaries left of x are dropped."""
    if not (chain.ok and chain.start <= x <= chain.end):
        return Chain(False, chain.start, chain.end, chain.values, chain.counts, chain.scales)
    k = x - chain.start
    return Chain(True, x, chain.end, chain.values[k:], chain.counts[k:] if guarded else chain.counts,
                 chain.scales[k:])


@njit(cache=True, parallel=True, nogil=True)
def _relabel_chain(chain, p, perm, guarded):
    """The backward chain after the suffix exchange perm at junction p: on every block from p on two
    founders' rows are swapped, so state q there holds the pair dosages of the former state perm[q];
    the backward recursion is the same up to that state order (its maxima and lexicographic maxima
    are order-free), so the exact messages and counts at boundaries >= p are the former ones
    relabelled and the block normalisers are unchanged. Boundaries left of p are dropped."""
    if not (chain.ok and chain.start <= p <= chain.end):
        return Chain(False, chain.start, chain.end, chain.values, chain.counts, chain.scales)
    k = p - chain.start
    boundaries, samples, states = chain.end - p + 1, chain.values.shape[1], chain.values.shape[2]
    relabelled = np.empty((boundaries, samples, states))
    relabelled_cnt = chain.counts
    if guarded:
        relabelled_cnt = np.empty((boundaries, samples, states), np.int64)
    values = chain.values
    counts = chain.counts
    for task in prange(boundaries * samples):
        x, s = task // samples, task % samples
        for q in range(states):
            relabelled[x, s, q] = values[k + x, s, perm[q]]
        if guarded:
            for q in range(states):
                relabelled_cnt[x, s, q] = counts[k + x, s, perm[q]]
    return Chain(True, p, chain.end, relabelled, relabelled_cnt, chain.scales[k:])


@njit(cache=True, nogil=True)
def _g_at(chain, x, p, frontier, haps, logs, offsets, penalty, stored, stored_cnt, gperm, permuted,
          guarded, acc):
    """Current backward message (and counts) at boundary x; returns (g, gc, chain)."""
    if x >= frontier:
        g = _stored_at(stored, x, gperm, permuted)
        gc = _stored_at(stored_cnt, x, gperm, permuted) if guarded else stored_cnt[0]
        return g, gc, chain
    chain = _ensure_chain(chain, frontier, p, haps, logs, offsets, penalty, stored, stored_cnt, gperm,
                          guarded, acc)
    g = chain.values[x - chain.start]
    gc = chain.counts[x - chain.start] if guarded else stored_cnt[0]
    return g, gc, chain


@njit(cache=True, nogil=True)
def _forward_to(ahead, p, need, f, fcnt, haps, logs, offsets, penalty, guarded, acc):
    """The lookahead: current forward messages valid on boundaries [p, need] with the chain's
    leaders, extended from what is still valid (at least AHEAD_BLOCKS blocks ahead)."""
    blocks = len(offsets) - 1
    if ahead.ok and ahead.valid >= need and ahead.start <= p:
        return ahead
    keep = ahead.ok and ahead.start <= p <= ahead.valid
    if keep:
        a0 = ahead.valid
        start_v = ahead.values[a0 - ahead.start]
        start_c = ahead.counts[a0 - ahead.start] if guarded else fcnt
    else:
        a0 = p
        start_v = f
        start_c = fcnt
    stop = min(blocks, max(need, p + AHEAD_BLOCKS))
    if guarded:
        nv, nc, nsm, nsl, nbm, nbl, nsc = forward_chain_lex(haps, logs, offsets, penalty, start_v,
                                                            start_c, a0, stop)
    else:
        nv, nsm, nsl, nbm, nbl = forward_chain(haps, logs, offsets, penalty, start_v, a0, stop)
        nc = ahead.counts
        nsc = ahead.site_cnt
    acc[_A_AHEAD] += stop - a0
    if keep:
        k0, k1 = p - ahead.start, a0 - ahead.start
        base = offsets[ahead.start]
        s0, s1 = offsets[p] - base, offsets[a0] - base
        nv = np.concatenate((ahead.values[k0:k1], nv))
        if guarded:
            nc = np.concatenate((ahead.counts[k0:k1], nc))
            nsc = np.concatenate((ahead.site_cnt[:, s0:s1], nsc), axis=1)
        nsm = np.concatenate((ahead.site_max[:, s0:s1], nsm), axis=1)
        nsl = np.concatenate((ahead.site_lead[:, s0:s1], nsl), axis=1)
        nbm = np.concatenate((ahead.block_max[k0:k1], nbm))
        nbl = np.concatenate((ahead.block_lead[k0:k1], nbl))
        return Ahead(True, p, stop, nv, nc, nsm, nsl, nbm, nbl, nsc)
    return Ahead(True, a0, stop, nv, nc, nsm, nsl, nbm, nbl, nsc)


@njit(cache=True, nogil=True)
def _truncate_ahead(ahead, x):
    """The lookahead after an edit at block x: valid no further than boundary x."""
    return Ahead(ahead.ok, ahead.start, min(ahead.valid, x), ahead.values, ahead.counts,
                 ahead.site_max, ahead.site_lead, ahead.block_max, ahead.block_lead, ahead.site_cnt)


# ------------------------------------------------------------------ candidates
@njit(cache=True, nogil=True)
def _set_rows(haps, leaves, offsets, founder, start, rows, lo, hi):
    """haps[founder] on blocks start + (0 .. hi - lo) from rows[lo:hi]."""
    for i in range(hi - lo):
        block = start + i
        leaf = leaves[block]
        r = rows[lo + i]
        for j in range(offsets[block], offsets[block + 1]):
            haps[founder, j] = leaf[r, j - offsets[block]]


@njit(cache=True, nogil=True)
def _active_items(q, X, lab, cand_start, cand_kind, cand_end, cand_k0, cand_row_start, cand_rows,
                  accepted, it_cand, it_kind, it_k, it_p1, it_p2, it_rs):
    """Write the candidates anchored at junction q that differ from the current panel (skipping the
    junction's i-th candidate where accepted[i]) to the item arrays; returns how many. An item is
    the candidate restricted to the blocks [p1, p2) where it differs."""
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
def _score_items(n_items, it_kind, it_k, it_p1, it_p2, it_rs, cand_rows, p, frontier, haps, leaves,
                 logs, offsets, penalty, stored, stored_cnt, stored_scale, gperm, permuted, guarded,
                 f, fcnt, chain, ahead, acc):
    """Exact gains of the items at junction p: under the guard the coarse items with the
    lexicographic kernel first, then the rest. Returns (total gains, total switch changes,
    per-sample gains, per-sample switch changes, chain, ahead); totals are summed in sample order."""
    samples, states = stored.shape[1], stored.shape[2]
    gain = np.zeros(n_items)
    switch = np.zeros(n_items)
    gains_s = np.zeros((n_items, samples))
    switches_s = np.zeros((n_items, samples), np.int64)
    for group in range(2):
        members = np.empty(n_items, np.int64)
        n_mem = 0
        for i in range(n_items):
            if guarded:
                if (group == 0) == (it_kind[i] == MOVE_COARSE):
                    members[n_mem] = i
                    n_mem += 1
            elif group == 1:
                members[n_mem] = i
                n_mem += 1
        if n_mem == 0:
            continue
        lex = guarded and group == 0
        members = members[:n_mem]
        # The lookahead covers every member's blocks: the background fast paths of both kernels
        # read the current chain up to the candidates' ends.
        last = it_p2[members[0]]
        for i in members:
            last = max(last, it_p2[i])
        ahead = _forward_to(ahead, p, last, f, fcnt, haps, logs, offsets, penalty, guarded, acc)
        a0 = ahead.start
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
            f_start[m] = ahead.values[first_block[m] - a0]
        g_start = np.empty((n_mem, samples, states))
        g_end = np.empty((n_mem, samples, states))
        # Switch counts at the candidates' ends only for the lexicographic kernel.
        count_rows = n_mem if lex else 1
        gc_start = np.zeros((count_rows, samples if lex else 1, states if lex else 1), np.int64)
        gc_end = np.zeros((count_rows, samples if lex else 1, states if lex else 1), np.int64)
        for m in range(n_mem):
            g, gc, chain = _g_at(chain, first_block[m], p, frontier, haps, logs, offsets, penalty,
                                 stored, stored_cnt, gperm, permuted, guarded, acc)
            g_start[m] = g
            if lex:
                gc_start[m] = gc
        for m in range(n_mem):
            g, gc, chain = _g_at(chain, end_block[m], p, frontier, haps, logs, offsets, penalty,
                                 stored, stored_cnt, gperm, permuted, guarded, acc)
            g_end[m] = g
            if lex:
                gc_end[m] = gc
        # Backward normalisers of each candidate's blocks: from the chain left of the frontier,
        # the stored ones from it on (added in block order).
        sums = np.zeros((n_mem, samples))
        for m in range(n_mem):
            lo_b, x = first_block[m], end_block[m]
            if lo_b < min(x, frontier):
                chain = _ensure_chain(chain, frontier, p, haps, logs, offsets, penalty, stored,
                                      stored_cnt, gperm, guarded, acc)
                hi_b = min(x, frontier)
                for r in range(lo_b - chain.start, hi_b - chain.start):
                    for s in range(samples):
                        sums[m, s] += chain.scales[r, s]
                lo_b = hi_b
            for r in range(lo_b, x):
                for s in range(samples):
                    sums[m, s] += stored_scale[r, s]
        if lex:
            fc_start = np.empty((n_mem, samples, states), np.int64)
            for m in range(n_mem):
                fc_start[m] = ahead.counts[first_block[m] - a0]
            member_gains, member_switches = candidate_gains_lex(
                haps, leaves, logs, offsets, penalty, f_start, fc_start, g_start, gc_start, focal,
                first_block, end_block, flat, row_offsets, g_end, gc_end, sums, ahead.values,
                ahead.counts, ahead.site_max, ahead.site_lead, ahead.block_max, ahead.block_lead,
                ahead.site_cnt, a0, ahead.valid)
            for m in range(n_mem):
                i = members[m]
                total = 0.0
                changes = 0
                for s in range(samples):
                    total += member_gains[m, s]
                    changes += member_switches[m, s]
                gain[i] = total
                gains_s[i] = member_gains[m]
                switch[i] = float(changes)
                switches_s[i] = member_switches[m]
        else:
            member_gains = candidate_gains(
                haps, leaves, logs, offsets, penalty, f_start, g_start, focal, first_block, end_block,
                flat, row_offsets, g_end, sums, ahead.site_max, ahead.site_lead, ahead.block_max,
                ahead.block_lead, a0, ahead.values)
            for m in range(n_mem):
                i = members[m]
                total = 0.0
                for s in range(samples):
                    total += member_gains[m, s]
                gain[i] = total
                gains_s[i] = member_gains[m]
    return gain, switch, gains_s, switches_s, chain, ahead


@njit(cache=True, nogil=True)
def _uncredited_switches(haps, leaves, logs, offsets, penalty, p, k, p1, p2, cand_rows, rs, gains_s,
                         switches_s, frontier, local, local_cnt, stored, stored_cnt, gperm, permuted,
                         f, fcnt, first, second, chain, ahead, acc):
    """N_shared of a coarse candidate (founder k's rows cand_rows[rs:] on [p1, p2)): the samples it
    affects are traced with and without it on a window around [p1, p2) that doubles until the two
    paths of a sample agree at both window ends (or the window is the whole component); their
    removed switch events go to guard.shared_savings. Returns (N_shared, chain, ahead)."""
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
        if lo <= p:
            f_lo, c_lo = local[lo], local_cnt[lo]
        else:
            ahead = _forward_to(ahead, p, lo, f, fcnt, haps, logs, offsets, penalty, True, acc)
            f_lo, c_lo = ahead.values[lo - ahead.start], ahead.counts[lo - ahead.start]
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
                s_i = affected[i]
                old_c = canonical_sites(old[4][i], old[3][i], s_i, haps, leaves, logs, offsets, first,
                                        second, -1, 0, 0, rows, offsets[lo])
                new_c = canonical_sites(new[4][i], new[3][i], s_i, haps, leaves, logs, offsets, first,
                                        second, k, p1, p2, rows, offsets[lo])
                gone = removed_events(old_c, len(old_c), new_c, len(new_c))
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
    total = shared_savings(rem_sample[:n_rem], rem_offsets[:n_rem + 1], rem_events, saved, k, first,
                           second, haps, logs, offsets[blocks])
    return total, chain, ahead


# ------------------------------------------------------------------ tie rules (design 5.3)
@njit(cache=True, nogil=True)
def _virtual_cmp(X, p, f1, f2, g1, g2):
    """Lexicographic comparison (-1, 0, 1) of two virtual founder paths: path(u, v) = X[u, :p], X[v, p:]."""
    for x in range(X.shape[1]):
        a = X[f1, x] if x < p else X[f2, x]
        b = X[g1, x] if x < p else X[g2, x]
        if a != b:
            return -1 if a < b else 1
    return 0


@njit(cache=True, nogil=True)
def _exchange_cmp(X, p, a1, b1, a2, b2):
    """Content order of two suffix exchanges at junction p: their resulting pairs of founder paths (each pair
    sorted, smaller path first) compared lexicographically."""
    if _virtual_cmp(X, p, a1, b1, b1, a1) <= 0:
        l1a, l1b, h1a, h1b = a1, b1, b1, a1
    else:
        l1a, l1b, h1a, h1b = b1, a1, a1, b1
    if _virtual_cmp(X, p, a2, b2, b2, a2) <= 0:
        l2a, l2b, h2a, h2b = a2, b2, b2, a2
    else:
        l2a, l2b, h2a, h2b = b2, a2, a2, b2
    c = _virtual_cmp(X, p, l1a, l1b, l2a, l2b)
    if c != 0:
        return c
    return _virtual_cmp(X, p, h1a, h1b, h2a, h2b)


@njit(cache=True, nogil=True)
def _exchange_winner(gains, pa, pb, X, p, q, floor):
    """The chosen suffix exchange among those with gain > floor: the best tie class of the exact gains (quantum
    q), inside it the content-smallest exchange (_exchange_cmp); a class wider than WIDE_QUANTA x q keeps value
    order (content among exactly equal values). Returns (index or -1, class size, wide)."""
    n = len(gains)
    eligible = np.empty(n, np.int64)
    m = 0
    for i in range(n):
        if gains[i] > floor:
            eligible[m] = i
            m += 1
    if m == 0:
        return -1, 0, False
    values = np.empty(m)
    for t in range(m):
        values[t] = -gains[eligible[t]]
    order = eligible[np.argsort(values)]
    best = gains[order[0]]
    size = 1
    previous = best
    while size < m:
        value = gains[order[size]]
        if not (previous - value <= q):
            break
        previous = value
        size += 1
    wide = (best - previous) > WIDE_QUANTA * q
    winner = order[0]
    for t in range(1, size):
        i = order[t]
        if wide and gains[i] != gains[winner]:
            if gains[i] > gains[winner]:
                winner = i
            continue
        if _exchange_cmp(X, p, pa[i], pb[i], pa[winner], pb[winner]) < 0:
            winner = i
    return winner, size, wide


@njit(cache=True, nogil=True)
def _move_cmp(X, cand_rows, k1, a1, c1, r1, i1, k2, a2, c2, r2, i2):
    """Content order of two content moves: the moved founder's resulting path (rows cand_rows[r:] on blocks
    [a, c), its current rows elsewhere), then the founder slot, then proposal order."""
    for x in range(X.shape[1]):
        u = cand_rows[r1 + x - a1] if a1 <= x < c1 else X[k1, x]
        v = cand_rows[r2 + x - a2] if a2 <= x < c2 else X[k2, x]
        if u != v:
            return -1 if u < v else 1
    if k1 != k2:
        return -1 if k1 < k2 else 1
    if i1 != i2:
        return -1 if i1 < i2 else 1
    return 0


@njit(cache=True, nogil=True)
def _move_before(u, v, gain, X, cand_rows, it_k, it_p1, it_p2, it_rs, it_cand, by_value):
    """Decision order of ranked content moves u, v (indices into the scored items): by exact gain (when by_value),
    then content (_move_cmp)."""
    if by_value and gain[u] != gain[v]:
        return gain[u] > gain[v]
    return _move_cmp(X, cand_rows, it_k[u], it_p1[u], it_p2[u], it_rs[u], it_cand[u], it_k[v], it_p1[v],
                     it_p2[v], it_rs[v], it_cand[v]) < 0


@njit(cache=True, nogil=True)
def _rank_moves(order, n_ranked, gain, X, cand_rows, it_k, it_p1, it_p2, it_rs, it_cand, q, class_size, acc):
    """Sorts order[:n_ranked] (moves above TOL) into decision order: tie classes of the exact gains (descending,
    quantum q; a class wider than WIDE_QUANTA x q keeps value order), content order inside a class. class_size[r]
    receives the size of the class of position r."""
    for i in range(1, n_ranked):
        j = i
        while j > 0 and _move_before(order[j], order[j - 1], gain, X, cand_rows, it_k, it_p1, it_p2, it_rs,
                                     it_cand, True):
            order[j - 1], order[j] = order[j], order[j - 1]
            j -= 1
    start = 0
    while start < n_ranked:
        end = start + 1
        while end < n_ranked and gain[order[end - 1]] - gain[order[end]] <= q:
            end += 1
        if end - start > 1:
            if gain[order[start]] - gain[order[end - 1]] > WIDE_QUANTA * q:
                acc[_A_TIE_WIDE] += 1
            else:
                for i in range(start + 1, end):
                    j = i
                    while j > start and _move_before(order[j], order[j - 1], gain, X, cand_rows, it_k, it_p1,
                                                     it_p2, it_rs, it_cand, False):
                        order[j - 1], order[j] = order[j], order[j - 1]
                        j -= 1
        for r in range(start, end):
            class_size[r] = end - start
        start = end


# ------------------------------------------------------------------ the pass
@njit(cache=True, nogil=True)
def pass_chunk(haps, X, leaves, logs, offsets, penalty, j0, j1, frontier0, f0, fcnt0, local,
               local_cnt, stored, stored_cnt, stored_scale, gperm0, lab, cand_start, cand_kind,
               cand_end, cand_k0, cand_row_start, cand_rows, pa, pb, guarded, exchanges, acc, ahead0,
               chain0, start_fwd, start_fcnt, pristine0, tie_q):
    """Junctions [j0, j1) of one acceptance pass (module docstring); haps, X and lab are modified in
    place, acc (ACC_KEYS) accumulated.

    State in: frontier0, the carried forward message f0 (counts fcnt0) at j0, gperm0 (current state
    -> stored state), lab (proposal founder label -> current row), the guard's history `local` of
    the carried messages at boundaries 0 .. j0 (filled here for j0 .. j1 - 1), the caches ahead0 /
    chain0 of the previous call. pristine0: no move has been accepted before j0 in a pass whose
    start panel has the forward messages start_fwd / start_fcnt (one-element placeholders
    otherwise); while nothing is accepted the carried forward message is read from them (the same
    recursion on the same panel) instead of being advanced. The candidates anchored at junction q
    are cand_start[q] .. cand_start[q + 1] - 1 in proposal order (kind, end block, founder label,
    rows cand_rows[cand_row_start[i]:] over [q, end)). Arrays of switch counts are one-element
    placeholders without the guard. tie_q: the tie quantum of the gains (module docstring).
    Returns (junction_gain, junction_pair of the best exchange at each junction (first look), gperm,
    the forward message and counts at j1, frontier, whether a move was accepted, ahead, chain).
    """
    founders, blocks = X.shape
    states = stored.shape[2]
    first, second = unordered_pairs(founders)
    total_sites = offsets[blocks]
    gperm = gperm0.copy()
    permuted = False
    for q in range(states):
        if gperm[q] != q:
            permuted = True
    junction_gain = np.full(j1 - j0, -np.inf)
    junction_pair = np.zeros((j1 - j0, 2), np.int64)
    frontier = frontier0
    chain = chain0
    ahead = ahead0
    pristine = pristine0
    f = f0
    fcnt = fcnt0
    accepted_any = False
    for p in range(j0, j1):
        if guarded:
            local[p] = f
            local_cnt[p] = fcnt
        accepted_here = False
        if p >= 1 and exchanges:
            first_look = True
            while True:
                g, _, chain = _g_at(chain, p, p, frontier, haps, logs, offsets, penalty, stored,
                                    stored_cnt, gperm, permuted, guarded, acc)
                gains = exchange_gains(f, np.ascontiguousarray(g))
                if first_look:
                    # the junction's best gain (value) and its exchange by the tie rule (all pairs)
                    top, top_size, top_wide = _exchange_winner(gains, pa, pb, X, p, tie_q, -np.inf)
                    if top < 0:
                        top = 0
                    junction_gain[p - j0] = np.max(gains)
                    junction_pair[p - j0, 0] = pa[top]
                    junction_pair[p - j0, 1] = pb[top]
                    first_look = False
                best, tied, wide = _exchange_winner(gains, pa, pb, X, p, tie_q, TOL)
                if best < 0:
                    break
                if tied > 1:
                    acc[_A_EXCHANGE_TIE] += 1
                if wide:
                    acc[_A_TIE_WIDE] += 1
                a, b = pa[best], pb[best]
                for x in range(p, blocks):
                    t = X[a, x]
                    X[a, x] = X[b, x]
                    X[b, x] = t
                for j in range(offsets[p], total_sites):
                    t8 = haps[a, j]
                    haps[a, j] = haps[b, j]
                    haps[b, j] = t8
                perm = transposition(founders, a, b)
                gperm = gperm[perm]
                permuted = True
                for m in range(founders):
                    if lab[m] == a:
                        lab[m] = b
                    elif lab[m] == b:
                        lab[m] = a
                chain = _relabel_chain(chain, p, perm, guarded)
                ahead = _truncate_ahead(ahead, p)
                acc[_A_EXCHANGE] += 1
                acc[_A_EXCHANGE_GAIN] += gains[best]
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
                    logs, offsets, penalty, stored, stored_cnt, stored_scale, gperm, permuted,
                    guarded, f, fcnt, chain, ahead, acc)
                # the best guard-passing candidate: tie classes of the gains, content order inside
                order = np.empty(n_act, np.int64)
                n_ranked = 0
                for i in range(n_act):
                    acc[_A_EVALUATED] += 1
                    evaluated[it_cand[i] - c0] = True
                    if act_gain[i] > TOL:
                        order[n_ranked] = i
                        n_ranked += 1
                class_size = np.ones(max(n_ranked, 1), np.int64)
                _rank_moves(order, n_ranked, act_gain, X, cand_rows, it_k, it_p1, it_p2, it_rs, it_cand,
                            tie_q, class_size, acc)
                chosen = -1
                chosen_rank = -1
                for r in range(n_ranked):
                    i = order[r]
                    gain = act_gain[i]
                    if guarded and it_kind[i] == MOVE_COARSE and (
                            gain + penalty * act_switch[i] < -GUARD_TOLERANCE):
                        # the move lowers the genotype fit E = F + penalty x switches
                        acc[_A_GUARD_TESTS] += 1
                        shared, chain, ahead = _uncredited_switches(
                            haps, leaves, logs, offsets, penalty, p, it_k[i], it_p1[i], it_p2[i],
                            cand_rows, it_rs[i], act_gains_s[i], act_switches_s[i], frontier, local,
                            local_cnt, stored, stored_cnt, gperm, permuted, f, fcnt, first, second,
                            chain, ahead, acc)
                        if not gain - penalty * shared > TOL:
                            acc[_A_GUARD_SWITCHES] += shared
                            acc[_A_GUARD_REJECTED] += 1
                            continue
                        acc[_A_GUARD_RELAXED] += 1
                    chosen = i
                    chosen_rank = r
                    break
                if chosen < 0:
                    break
                if class_size[chosen_rank] > 1:
                    acc[_A_CONTENT_TIE] += 1
                i = chosen
                gain = act_gain[i]
                accepted[it_cand[i] - c0] = True
                k, p1, p2 = it_k[i], it_p1[i], it_p2[i]
                for x in range(p1, p2):
                    X[k, x] = cand_rows[it_rs[i] + x - p1]
                _set_rows(haps, leaves, offsets, k, p1, cand_rows, it_rs[i], it_rs[i] + p2 - p1)
                if p2 <= frontier:
                    chain = _chain_from(chain, p2, guarded)
                else:
                    frontier = p2
                    chain = Chain(False, chain.start, chain.end, chain.values, chain.counts,
                                  chain.scales)
                ahead = _truncate_ahead(ahead, p1)
                if it_kind[i] == MOVE_COARSE:
                    acc[_A_COARSE] += 1
                    acc[_A_COARSE_GAIN] += gain
                else:
                    acc[_A_FINE] += 1
                    acc[_A_FINE_GAIN] += gain
                acc[_A_CHANGED] += p2 - p1
                accepted_here = True
            for i in range(n_here):
                if evaluated[i] and not accepted[i]:
                    acc[_A_FALSE_POSITIVE] += 1
        if accepted_here:
            pristine = False
        if p + 1 < blocks:
            if pristine:
                f = start_fwd[p + 1]
                if guarded:
                    fcnt = start_fcnt[p + 1]
            elif ahead.ok and ahead.start <= p + 1 <= ahead.valid:
                f = ahead.values[p + 1 - ahead.start]
                if guarded:
                    fcnt = ahead.counts[p + 1 - ahead.start]
            elif guarded:
                f, fcnt = advance_lex(haps, logs, offsets, penalty, f, fcnt, p, p + 1)
            else:
                f = advance(haps, logs, offsets, penalty, f, p, p + 1)
        if accepted_here:
            accepted_any = True
    return (junction_gain, junction_pair, gperm, np.ascontiguousarray(f), np.ascontiguousarray(fcnt),
            frontier, accepted_any, ahead, chain)


def candidate_arrays(props, blocks):
    """The proposals anchored at every junction as arrays: start offsets per junction, kind, end
    block, founder label, row offsets and flat rows. props: {junction: [(kind, end, label, rows)]}
    in proposal order."""
    start = np.zeros(blocks + 1, np.int64)
    kind, end, label, row_start, rows = [], [], [], [], []
    total = 0
    for q in range(blocks):
        start[q] = len(kind)
        for move, c, k0, r in props.get(q, ()):
            kind.append(move)
            end.append(c)
            label.append(k0)
            row_start.append(total)
            rows.append(np.asarray(r, np.int64))
            total += len(r)
    start[blocks] = len(kind)
    return (start, np.asarray(kind, np.int64), np.asarray(end, np.int64),
            np.asarray(label, np.int64), np.asarray(row_start, np.int64),
            np.concatenate(rows) if rows else np.zeros(0, np.int64))


def accept_pass(orientation, X, stored, stored_cnt, stored_scale, props, penalty, guarded, budget,
                exchanges=True, start_fwd=None, start_fcnt=None, scale=0.0):
    """One exact acceptance pass on `orientation` (search.Orientation); X (K x B, that orientation's
    block order) is modified in place.

    stored, stored_cnt, stored_scale: the stored backward messages, counts (guard only) and
    normalisers of the pass's start panel in this orientation. props: the content proposals
    {junction: [(MOVE_FINE | MOVE_COARSE, end block, founder, rows)]}. guarded: the shared-junction
    guard applies to coarse moves (a search with an L1 context). exchanges=False scores and applies
    content proposals only (an addition round's copy-only pass). start_fwd / start_fcnt: the
    forward messages (and counts) of X itself in this orientation, if known; until the first
    accepted move the carried forward message is read from them. budget: the live thread share
    (execution.CpuBudget), re-read before every JUNCTION_BATCH junctions. scale: |F| of the pass's
    start panel (the tie quantum's scale, tie_rules.quantum("refine_F", scale)).
    Returns (counters {ACC_KEYS}, junction_gain (B + 1), junction_pair (B + 1, 2)).
    """
    tie_q = float(tie_quantum("refine_F", scale))
    founders, blocks = X.shape
    samples, states = stored.shape[1], stored.shape[2]
    haps = orientation.haps(X)
    cands = candidate_arrays(props, blocks)
    count2 = np.zeros((1, 1), np.int64)
    count3 = np.zeros((1, 1, 1), np.int64)
    if guarded:
        # Carried forward messages of the junctions passed (the guard's path tracing).
        local = np.empty((blocks + 1, samples, states))
        local_cnt = np.empty((blocks + 1, samples, states), np.int64)
    else:
        local, local_cnt = np.zeros((1, 1, 1)), count3
    acc = np.zeros(len(ACC_KEYS))
    f = np.zeros((samples, states))
    fcnt = np.zeros((samples, states), np.int64) if guarded else count2
    gperm = np.arange(states, dtype=np.int64)
    lab = np.arange(founders, dtype=np.int64)
    pa, pb = exchange_pairs(founders)
    ahead = Ahead(False, 0, -1, np.zeros((1, 1, 1)), count3, np.zeros((1, 1)),
                  np.zeros((1, 1), np.int32), np.zeros((1, 1)), np.zeros((1, 1), np.int32),
                  np.zeros((1, 1), np.int32))
    chain = Chain(False, 0, 0, np.zeros((1, 1, 1)), count3, np.zeros((1, 1)))
    pristine = start_fwd is not None
    if start_fwd is None:
        start_fwd = np.zeros((1, 1, 1))
    if start_fcnt is None:
        start_fcnt = count3
    junction_gain = np.full(blocks + 1, -np.inf)
    junction_pair = np.zeros((blocks + 1, 2), np.int64)
    frontier, p = 0, 0
    while p < blocks:
        stop = min(blocks, p + JUNCTION_BATCH)
        with execution.numba_threads(max(1, int(budget.threads()))):
            (jg, jp, gperm, f, fcnt, frontier, accepted, ahead, chain) = pass_chunk(
                haps, X, orientation.leaves, orientation.logs, orientation.offsets, float(penalty),
                p, stop, frontier, f, fcnt, local, local_cnt, stored,
                stored_cnt if guarded else count3, stored_scale, gperm, lab, *cands, pa, pb,
                bool(guarded), bool(exchanges), acc, ahead, chain, start_fwd, start_fcnt, pristine, tie_q)
        pristine = pristine and not accepted
        junction_gain[p:stop] = jg
        junction_pair[p:stop] = jp
        p = stop
    record = {key: (int(round(value)) if key in INTEGER_KEYS else float(value))
              for key, value in zip(ACC_KEYS, acc.tolist())}
    return record, junction_gain, junction_pair
