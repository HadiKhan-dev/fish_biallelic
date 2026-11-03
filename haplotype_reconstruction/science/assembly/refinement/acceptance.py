"""One exact Gauss-Seidel acceptance pass of a fixed-K search (compiled; search.py drives it).

Owns Gauss-Seidel move order and guard verdicts; never logs or opens storage.
The long junction kernel stays whole so exchange, full ranking, stale rescoring, message invalidation
and advancement retain one explicit state transition order.
Method: docs/methods/founder_refinement.md#acceptance (RS08–RS10).

Owns the Gauss-Seidel acceptance transition and its decision receipts.
Must not choose K, read storage or configure runtime.
Method: docs/methods/founder_refinement.md#acceptance (RS08-RS10).
STYLE_40 exception: numerical kernels retain flat scalar/array arguments because
Numba cannot lower the frozen Python boundary records; the stage uses those records.
Hot recurrences remain whole to preserve their loop/update order and avoid temporary
materialisations. Parallel kernels write disjoint sample slices; sample reductions
are fixed serial folds, independent of the explicit stage team's size.


The pass walks the junctions p = 0 .. B - 1 of the panel X in one orientation (search.Orientation:
forward or reversed site order). At every junction every founder-pair suffix exchange is scored
exactly from the carried forward and the current stored backward messages and the best is applied
while its gain exceeds TOL (at most EXCHANGE_ROUNDS times); then the content candidates anchored
there (one founder's rows on [p1, p2): the blocks where a window proposal differs from the current
panel) are scored exactly from the current messages at p1 and p2 (gains.py), best first, each
applied while its gain exceeds TOL (after CONTENT_FULL_ROUNDS acceptances the rest one at a time in
the last full ranking's order); coarse candidates also pass the shared-junction guard (guard.py);
then the forward message advances one block.

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
from numba import prange, types
from numba.typed import List

from haplotype_reconstruction.core.errors import InvariantError
from haplotype_reconstruction.core.kernels import kernel
from haplotype_reconstruction.core.ties import WIDE_CLASS_QUANTA, policy

from ..orders import _exchange_winner, _move_cmp
from .checkpoints import read as read_checkpoint
from .gains import (
    candidate_gains,
    candidate_gains_lex,
    exchange_gains,
    exchange_pairs,
    parallel_exchanges,
    transposition,
)
from .guard import canonical_sites, path_switch_events, removed_events, shared_savings
from .messages import advance, advance_lex, backward_chain, backward_chain_lex, forward_chain, forward_chain_lex
from .objective import unordered_pairs
from .overloads import overloads_for
from .parameters import (
    ACCEPTANCE_GAIN_FLOOR,
    AFFECTED_GAIN_TOLERANCE,
    CONTENT_FULL_ROUNDS,
    EXCHANGE_ROUNDS,
    GUARD_TOLERANCE,
)
from .window import cover
from .workspace import sample_threads

# A move is accepted while its exact F gain exceeds TOL (log units), the refine_F noise floor.
# The 2^-9 quantum covers measured motion under 1e-6 relative GL noise (p99 9e-5 nats over ~4,300 contrasts).
TOL = ACCEPTANCE_GAIN_FLOOR
# A coarse move "lowers E" when gain + switch-cost change < -GUARD_TOLERANCE (E = F + the
# switch cost of the samples' lexicographic maximum paths). This guards dE, rather than a score tie:
# raising it would let moves with small negative dE bypass the guard. Its contrast under 1e-6 GL noise is unmeasured.
# Forward messages of the current panel computed ahead of the pass position at a time: at least this many blocks,
# further when a candidate starts further ahead. An acceptance truncates the lookahead at its edit and discards
# what lay beyond it, so a short floor limits wasted work. The extent never changes a value.
AHEAD_BLOCKS = 4
# Junctions per nogil call (the Numba team is re-read between calls).
JUNCTION_BATCH = 16
# On reaching a junction the pass covers the blocks it reads first (the lookahead and a fine
# window ahead of it), so a streaming window slides with the pass instead of jumping between its reads.
JUNCTION_REACH = 16 + 8  # Streaming window reach, in blocks.
# A tie class wider than this many quanta keeps value order (maths.ties.WIDE_CLASS_QUANTA).
WIDE_QUANTA = WIDE_CLASS_QUANTA
# K-independent bounds on rescoring at one junction, so a pass stays
# O(K^3). Content candidates: the first CONTENT_FULL_ROUNDS acceptances rescore every remaining item
# (best first); after that the remaining items of the last full ranking are rescored one at a time in
# that ranking's order and each accepted while its own exact gain exceeds TOL (and the guard passes),
# so a junction costs at most (CONTENT_FULL_ROUNDS + 1) x its candidates evaluations. Suffix
# exchanges: at most EXCHANGE_ROUNDS accepted per junction and pass. A pass with no acceptance is
# unaffected (every item is scored once), so the fit's fixed point is the same kind; the binding
# counters (round_cap, exchange_cap) record every junction where a cap changed the search order.

MOVE_FINE, MOVE_COARSE = 1, 2
# Pass counters (the fit history's per-sweep record). false_positive: candidates evaluated but never accepted
# (once each); content_rounds: full rankings at junctions with candidates; round_cap: junctions where an item of
# the last full ranking was rescored alone (a junction that reaches the cap with no positive untried item exits
# without it); stale_evaluated: those single rescorings (their decisions compare one item, so content_tie does
# not count ties there); exchange_cap: junctions left with a positive exchange after EXCHANGE_ROUNDS; lex_items:
# coarse items above TOL in guarded rounds, whether or not the ranking reaches their lexicographic rescore
# (see _score_items); this count decides the search's next sweep mode;
# lex_abort: plain passes stopped for a guarded rerun.
ACC_KEYS = (
    "fine",
    "coarse",
    "exchange",
    "fine_gain",
    "coarse_gain",
    "exchange_gain",
    "evaluated",
    "false_positive",
    "guard_rejected",
    "guard_shared_tests",
    "guard_relaxed",
    "guard_shared_switches",
    "guard_traced_blocks",
    "chain_blocks",
    "ahead_blocks",
    "changed_blocks",
    "exchange_tie",
    "content_tie",
    "tie_wide",
    "content_rounds",
    "round_cap",
    "stale_evaluated",
    "exchange_cap",
    "lex_items",
    "lex_abort",
)
INTEGER_KEYS = frozenset(ACC_KEYS) - {"fine_gain", "coarse_gain", "exchange_gain", "guard_shared_switches"}
(
    _A_FINE,
    _A_COARSE,
    _A_EXCHANGE,
    _A_FINE_GAIN,
    _A_COARSE_GAIN,
    _A_EXCHANGE_GAIN,
    _A_EVALUATED,
    _A_FALSE_POSITIVE,
    _A_GUARD_REJECTED,
    _A_GUARD_TESTS,
    _A_GUARD_RELAXED,
    _A_GUARD_SWITCHES,
    _A_TRACED,
    _A_CHAIN,
    _A_AHEAD,
    _A_CHANGED,
    _A_EXCHANGE_TIE,
    _A_CONTENT_TIE,
    _A_TIE_WIDE,
    _A_ROUNDS,
    _A_ROUND_CAP,
    _A_STALE,
    _A_EXCHANGE_CAP,
    _A_LEX_ITEMS,
    _A_LEX_ABORT,
) = range(len(ACC_KEYS))

# Exact current backward messages on boundaries [start, end] (with counts under the guard) and the
# normalisers of blocks [start, end); valid only when ok.
Chain = namedtuple("Chain", ["ok", "start", "end", "values", "counts", "scales"])
# Exact current forward messages on boundaries [start, valid] with the forward chain's founder maxima
# (messages.forward_chain(_lex): site_max = R1, site_lead = P1, site_cnt = R2, each S x sites x K) and
# block leaders; counts only under the guard.
Ahead = namedtuple(
    "Ahead",
    ["ok", "start", "valid", "values", "counts", "site_max", "site_lead", "block_max", "block_lead", "site_cnt"],
)


# ------------------------------------------------------------------ message caches
@kernel(overloads=overloads_for(__name__, "_ensure_chain"), cache=True, nogil=True)
def _ensure_chain(chain, frontier, p, haps, win, offsets, stored, stored_cnt, gperm, guarded, acc):
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
            head, head_cnt, head_scales = backward_chain_lex(
                haps, win, offsets, chain.values[0], chain.counts[0], ident, p, chain.start
            )
            counts = np.concatenate((head_cnt[:-1], chain.counts))
        else:
            head, head_scales = backward_chain(haps, win, offsets, chain.values[0], ident, p, chain.start)
        acc[_A_CHAIN] += chain.start - p
        return Chain(
            True,
            p,
            chain.end,
            np.concatenate((head[:-1], chain.values)),
            counts,
            np.concatenate((head_scales, chain.scales)),
        )
    counts = chain.counts
    end, end_cnt = read_checkpoint(stored, frontier)
    if guarded:
        values, counts, scales = backward_chain_lex(haps, win, offsets, end, end_cnt, gperm, p, frontier)
    else:
        values, scales = backward_chain(haps, win, offsets, end, gperm, p, frontier)
    acc[_A_CHAIN] += frontier - p
    return Chain(True, p, frontier, values, counts, scales)


@kernel(overloads=overloads_for(__name__, "_chain_from"), cache=True, nogil=True)
def _chain_from(chain, x, guarded):
    """The backward chain after an edit of blocks left of boundary x that leaves the frontier in
    place: its values and counts at boundaries >= x and the normalisers of blocks >= x depend only
    on blocks >= x and stay exact; boundaries left of x are dropped."""
    if not (chain.ok and chain.start <= x <= chain.end):
        return Chain(False, chain.start, chain.end, chain.values, chain.counts, chain.scales)
    k = x - chain.start
    return Chain(True, x, chain.end, chain.values[k:], chain.counts[k:] if guarded else chain.counts, chain.scales[k:])


@kernel(overloads=overloads_for(__name__, "_relabel_chain"), cache=True, parallel=True, nogil=True)
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
        relabelled_cnt = np.empty((boundaries, samples, states))
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


@kernel(overloads=overloads_for(__name__, "_g_at"), cache=True, nogil=True)
def _g_at(chain, x, p, frontier, haps, win, offsets, stored, stored_cnt, gperm, permuted, guarded, acc):
    """Current backward message (and counts) at boundary x; returns (g, gc, chain)."""
    if x >= frontier:
        g, gc = read_checkpoint(stored, x)
        if permuted:
            g = np.ascontiguousarray(g[:, gperm])
            if guarded:
                gc = np.ascontiguousarray(gc[:, gperm])
        return g, gc, chain
    chain = _ensure_chain(chain, frontier, p, haps, win, offsets, stored, stored_cnt, gperm, guarded, acc)
    g = chain.values[x - chain.start]
    gc = chain.counts[x - chain.start] if guarded else stored_cnt[0]
    return g, gc, chain


@kernel(overloads=overloads_for(__name__, "_forward_to"), cache=True, nogil=True)
def _forward_to(ahead, p, need, f, fcnt, haps, win, offsets, guarded, acc):
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
        nv, nc, nsm, nsl, nbm, nbl, nsc = forward_chain_lex(haps, win, offsets, start_v, start_c, a0, stop)
    else:
        nv, nsm, nsl, nbm, nbl = forward_chain(haps, win, offsets, start_v, a0, stop)
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


@kernel(overloads=overloads_for(__name__, "_truncate_ahead"), cache=True, nogil=True)
def _truncate_ahead(ahead, x):
    """The lookahead after an edit at block x: valid no further than boundary x."""
    return Ahead(
        ahead.ok,
        ahead.start,
        min(ahead.valid, x),
        ahead.values,
        ahead.counts,
        ahead.site_max,
        ahead.site_lead,
        ahead.block_max,
        ahead.block_lead,
        ahead.site_cnt,
    )


# ------------------------------------------------------------------ candidates
@kernel(overloads=overloads_for(__name__, "_set_rows"), cache=True, nogil=True)
def _set_rows(haps, leaves, offsets, founder, start, rows, lo, hi):
    """haps[founder] on blocks start + (0 .. hi - lo) from rows[lo:hi]."""
    for i in range(hi - lo):
        block = start + i
        leaf = leaves[block]
        r = rows[lo + i]
        for j in range(offsets[block], offsets[block + 1]):
            haps[founder, j] = leaf[r, j - offsets[block]]


@kernel(overloads=overloads_for(__name__, "_active_items"), cache=True, nogil=True)
def _active_items(
    q,
    X,
    lab,
    cand_start,
    cand_kind,
    cand_end,
    cand_k0,
    cand_row_start,
    cand_rows,
    accepted,
    it_cand,
    it_kind,
    it_k,
    it_p1,
    it_p2,
    it_rs,
):
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


@kernel(overloads=overloads_for(__name__, "_score_items"), cache=True, nogil=True)
def _score_items(
    n_items,
    it_kind,
    it_k,
    it_p1,
    it_p2,
    it_rs,
    cand_rows,
    p,
    frontier,
    haps,
    leaves,
    win,
    offsets,
    stored,
    stored_cnt,
    stored_scale,
    gperm,
    permuted,
    guarded,
    f,
    fcnt,
    chain,
    ahead,
    acc,
    lex_item,
):
    """Exact gains of the items at junction p: with lex_item < 0 every item with the plain kernel (switch
    changes zero); with lex_item = i >= 0 item i alone with the lexicographic kernel, for its switch-cost
    changes (its gains equal the plain ones: the lexicographic recursion's scores are the plain recursion's).
    Lazy lexicographic scoring: under the guard the ranking loop rescores lexicographically
    only the coarse items it reaches in decision order (up to the chosen one). The round makes no
    decision on an item it does not reach. Returns (total gains, total switch changes,
    per-sample gains, per-sample switch changes, chain, ahead); totals are summed in sample order."""
    samples, states = stored[0].shape[1], stored[0].shape[2]
    gain = np.zeros(n_items)
    switch = np.zeros(n_items)
    gains_s = np.zeros((n_items, samples))
    switches_s = np.zeros((n_items, samples))
    for group in range(2):
        members = np.empty(n_items, np.int64)
        n_mem = 0
        for i in range(n_items):
            if (group == 0 and lex_item < 0) or (group == 1 and i == lex_item):
                members[n_mem] = i
                n_mem += 1
        if n_mem == 0:
            continue
        lex = group == 1
        members = members[:n_mem]
        # The lookahead covers every member's first block, the only forward message the candidate kernels
        # read (f_start); accepted edits invalidate forward messages beyond their starts.
        last = it_p1[members[0]]
        for i in members:
            last = max(last, it_p1[i])
        ahead = _forward_to(ahead, p, last, f, fcnt, haps, win, offsets, guarded, acc)
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
        gc_start = np.zeros((count_rows, samples if lex else 1, states if lex else 1))
        gc_end = np.zeros((count_rows, samples if lex else 1, states if lex else 1))
        for m in range(n_mem):
            g, gc, chain = _g_at(
                chain,
                first_block[m],
                p,
                frontier,
                haps,
                win,
                offsets,
                stored,
                stored_cnt,
                gperm,
                permuted,
                guarded,
                acc,
            )
            g_start[m] = g
            if lex:
                gc_start[m] = gc
        for m in range(n_mem):
            g, gc, chain = _g_at(
                chain, end_block[m], p, frontier, haps, win, offsets, stored, stored_cnt, gperm, permuted, guarded, acc
            )
            g_end[m] = g
            if lex:
                gc_end[m] = gc
        # Backward normalisers of each candidate's blocks: from the chain left of the frontier,
        # the stored ones from it on (added in block order).
        sums = np.zeros((n_mem, samples))
        for m in range(n_mem):
            lo_b, x = first_block[m], end_block[m]
            if lo_b < min(x, frontier):
                chain = _ensure_chain(chain, frontier, p, haps, win, offsets, stored, stored_cnt, gperm, guarded, acc)
                hi_b = min(x, frontier)
                for r in range(lo_b - chain.start, hi_b - chain.start):
                    for s in range(samples):
                        sums[m, s] += chain.scales[r, s]
                lo_b = hi_b
            for r in range(lo_b, x):
                for s in range(samples):
                    sums[m, s] += stored_scale[r, s]
        if lex:
            fc_start = np.empty((n_mem, samples, states))
            for m in range(n_mem):
                fc_start[m] = ahead.counts[first_block[m] - a0]
            member_gains, member_switches = candidate_gains_lex(
                haps,
                leaves,
                win,
                offsets,
                f_start,
                fc_start,
                g_start,
                gc_start,
                focal,
                first_block,
                end_block,
                flat,
                row_offsets,
                g_end,
                gc_end,
                sums,
            )
            for m in range(n_mem):
                i = members[m]
                total = 0.0
                changes = 0.0
                for s in range(samples):
                    total += member_gains[m, s]
                    changes += member_switches[m, s]
                gain[i] = total
                gains_s[i] = member_gains[m]
                switch[i] = changes
                switches_s[i] = member_switches[m]
        else:
            member_gains = candidate_gains(
                haps,
                leaves,
                win,
                offsets,
                f_start,
                g_start,
                focal,
                first_block,
                end_block,
                flat,
                row_offsets,
                g_end,
                sums,
            )
            for m in range(n_mem):
                i = members[m]
                total = 0.0
                for s in range(samples):
                    total += member_gains[m, s]
                gain[i] = total
                gains_s[i] = member_gains[m]
    if guarded and lex_item < 0:
        # 3.4's count of lexicographic rescorings (ACC_KEYS), which the next sweep's mode reads.
        for i in range(n_items):
            if it_kind[i] == MOVE_COARSE and gain[i] > TOL:
                acc[_A_LEX_ITEMS] += 1
    return gain, switch, gains_s, switches_s, chain, ahead


@kernel(overloads=overloads_for(__name__, "_uncredited_switches"), cache=True, nogil=True)
def _uncredited_switches(
    haps,
    leaves,
    win,
    offsets,
    p,
    k,
    p1,
    p2,
    cand_rows,
    rs,
    gains_s,
    switches_s,
    frontier,
    local,
    local_cnt,
    stored,
    stored_cnt,
    gperm,
    permuted,
    f,
    fcnt,
    first,
    second,
    chain,
    ahead,
    acc,
    reverse,
):
    """C_shared of a coarse candidate (founder k's rows cand_rows[rs:] on [p1, p2)): the samples it
    affects are traced with and without it on a window around [p1, p2) that doubles until the two
    paths of a sample agree at both window ends (or the window is the whole component); their
    removed switch events (with their costs) go to guard.shared_savings. Returns (C_shared, chain,
    ahead)."""
    samples = len(gains_s)
    blocks = len(offsets) - 1
    rows = cand_rows[rs : rs + p2 - p1].copy()
    n_aff = 0
    affected = np.empty(samples, np.int64)
    for s in range(samples):
        if switches_s[s] != 0.0 or abs(gains_s[s]) > AFFECTED_GAIN_TOLERANCE:
            affected[n_aff] = s
            n_aff += 1
    affected = affected[:n_aff]
    # removed events per settled sample (flattened, in settling order)
    rem_sample = np.empty(samples, np.int64)
    rem_offsets = np.zeros(samples + 1, np.int64)
    rem_events = np.zeros((0, 3), np.int64)
    rem_costs = np.zeros(0)
    n_rem = 0
    width = 2
    while len(affected):
        lo, hi = max(0, p1 - width), min(blocks, p2 + width)
        if lo <= p:
            # Passed prefix alleles never change: reconstruct from the nearest
            # carried-message checkpoint with the same block-normalized recursion.
            anchor = (lo // JUNCTION_BATCH) * JUNCTION_BATCH
            f_lo, c_lo = local[lo // JUNCTION_BATCH], local_cnt[lo // JUNCTION_BATCH]
            if anchor != lo:
                f_lo, c_lo = advance_lex(haps, win, offsets, f_lo, c_lo, anchor, lo)
        else:
            ahead = _forward_to(ahead, p, lo, f, fcnt, haps, win, offsets, True, acc)
            f_lo, c_lo = ahead.values[lo - ahead.start], ahead.counts[lo - ahead.start]
        g_hi, gc_hi, chain = _g_at(
            chain, hi, p, frontier, haps, win, offsets, stored, stored_cnt, gperm, permuted, True, acc
        )
        fa = np.empty((len(affected), f_lo.shape[1]))
        ca = np.empty((len(affected), f_lo.shape[1]))
        ga = np.empty((len(affected), f_lo.shape[1]))
        gca = np.empty((len(affected), f_lo.shape[1]))
        for i in range(len(affected)):
            fa[i] = f_lo[affected[i]]
            ca[i] = c_lo[affected[i]]
            ga[i] = g_hi[affected[i]]
            gca[i] = gc_hi[affected[i]]
        old = path_switch_events(haps, leaves, win, offsets, fa, ca, ga, gca, lo, hi, affected, -1, 0, 0, rows, reverse)
        new = path_switch_events(
            haps, leaves, win, offsets, fa, ca, ga, gca, lo, hi, affected, k, p1, p2, rows, reverse
        )
        acc[_A_TRACED] += (hi - lo) * len(affected)
        everything = lo == 0 and hi == blocks
        keep = np.ones(len(affected), np.bool_)
        for i in range(len(affected)):
            if everything or (old[0][i] == new[0][i] and old[1][i] == new[1][i]):
                keep[i] = False
                s_i = affected[i]
                old_c = canonical_sites(
                    old[4][i], old[3][i], s_i, haps, leaves, win, offsets, first, second, -1, 0, 0, rows, offsets[lo]
                )
                new_c = canonical_sites(
                    new[4][i], new[3][i], s_i, haps, leaves, win, offsets, first, second, k, p1, p2, rows, offsets[lo]
                )
                gone, gone_costs = removed_events(old_c, len(old_c), new_c, len(new_c), old[5][i])
                if len(gone):
                    rem_sample[n_rem] = affected[i]
                    rem_events = np.concatenate((rem_events, gone))
                    rem_costs = np.concatenate((rem_costs, gone_costs))
                    rem_offsets[n_rem + 1] = len(rem_events)
                    n_rem += 1
        affected = affected[keep]
        width *= 2
    saved = np.zeros(samples)
    for s in range(samples):
        if switches_s[s] < 0.0:
            saved[s] = -switches_s[s]
    if n_rem == 0:
        return 0.0, chain, ahead
    total = shared_savings(
        rem_sample[:n_rem],
        rem_offsets[: n_rem + 1],
        rem_events,
        rem_costs,
        saved,
        k,
        first,
        second,
        haps,
        win,
        offsets[blocks],
    )
    return total, chain, ahead


# ------------------------------------------------------------------ tie rules (design 5.3)


@kernel(overloads=overloads_for(__name__, "_move_before"), cache=True, nogil=True)
def _move_before(u, v, gain, X, cand_rows, it_k, it_p1, it_p2, it_rs, it_cand, by_value, reverse):
    """Decision order of ranked content moves u, v (indices into the scored items): by exact gain (when by_value),
    then content (_move_cmp)."""
    if by_value and gain[u] != gain[v]:
        return gain[u] > gain[v]
    return (
        _move_cmp(
            X,
            cand_rows,
            it_k[u],
            it_p1[u],
            it_p2[u],
            it_rs[u],
            it_cand[u],
            it_k[v],
            it_p1[v],
            it_p2[v],
            it_rs[v],
            it_cand[v],
            reverse,
        )
        < 0
    )


@kernel(overloads=overloads_for(__name__, "_rank_moves"), cache=True, nogil=True)
def _rank_moves(order, n_ranked, gain, X, cand_rows, it_k, it_p1, it_p2, it_rs, it_cand, q, class_size, acc, reverse):
    """Sorts order[:n_ranked] (moves above TOL) into decision order: tie classes of the exact gains (descending,
    quantum q; a class wider than WIDE_QUANTA x q keeps value order), content order inside a class. class_size[r]
    receives the size of the class of position r."""
    for i in range(1, n_ranked):
        j = i
        while j > 0 and _move_before(
            order[j], order[j - 1], gain, X, cand_rows, it_k, it_p1, it_p2, it_rs, it_cand, True, reverse
        ):
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
                    while j > start and _move_before(
                        order[j], order[j - 1], gain, X, cand_rows, it_k, it_p1, it_p2, it_rs, it_cand, False, reverse
                    ):
                        order[j - 1], order[j] = order[j], order[j - 1]
                        j -= 1
        for r in range(start, end):
            class_size[r] = end - start
        start = end


# ------------------------------------------------------------------ the pass
@kernel(overloads=overloads_for(__name__, "pass_chunk"), cache=True, nogil=True)
def pass_chunk(
    haps,
    X,
    leaves,
    win,
    offsets,
    j0,
    j1,
    frontier0,
    f0,
    fcnt0,
    local,
    local_cnt,
    stored,
    stored_cnt,
    stored_scale,
    gperm0,
    lab,
    cand_start,
    cand_kind,
    cand_end,
    cand_k0,
    cand_row_start,
    cand_rows,
    pa,
    pb,
    guarded,
    exchanges,
    exchanges_parallel,
    acc,
    ahead0,
    chain0,
    start_fwd,
    start_fcnt,
    pristine0,
    tie_q,
    reverse,
    optimistic,
    events,
):
    """Junctions [j0, j1) of one acceptance pass (module docstring); haps, X and lab are modified in
    place, acc (ACC_KEYS) accumulated.

    State in: frontier0, the carried forward message f0 (counts fcnt0) at j0, gperm0 (current state
    -> stored state), lab (proposal founder label -> current row), the guard's sparse history `local` of
    carried messages at multiples of JUNCTION_BATCH through j0 (extended here), the caches ahead0 /
    chain0 of the previous call. pristine0: no move has been accepted before j0 in a pass whose
    start panel has the forward messages start_fwd / start_fcnt (one-element placeholders
    otherwise); while nothing is accepted the carried forward message is read from them (the same
    recursion on the same panel) instead of being advanced. The candidates anchored at junction q
    are cand_start[q] .. cand_start[q + 1] - 1 in proposal order (kind, end block, founder label,
    rows cand_rows[cand_row_start[i]:] over [q, end)). Arrays of switch counts are one-element
    placeholders without the guard. exchanges_parallel: each junction's exchange gains run as one parallel
    region on the caller's team (gains.parallel_exchanges). tie_q: the tie quantum of the gains (module
    docstring). reverse: the convention reversal (maths.ties.reverse_conventions(); content orders inverted). optimistic
    (unguarded only): stop at the first coarse item whose gain exceeds TOL (the first
    point where a guarded pass reads switch costs) with acc[lex_abort] = 1; the caller discards the
    pass and runs it guarded.
    Returns (junction_gain, junction_pair of the best exchange at each junction (first look), gperm,
    the forward message and counts at j1, frontier, whether a move was accepted, ahead, chain).
    """
    founders, blocks = X.shape
    states = stored[0].shape[2]
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
        cover(win, offsets[p], offsets[min(blocks, p + JUNCTION_REACH)], 0, stored[0].shape[1], False)
        if guarded and p % JUNCTION_BATCH == 0:
            local[p // JUNCTION_BATCH] = f
            local_cnt[p // JUNCTION_BATCH] = fcnt
        accepted_here = False
        if p >= 1 and exchanges:
            first_look = True
            n_exchanged = 0
            while True:
                g, _, chain = _g_at(
                    chain, p, p, frontier, haps, win, offsets, stored, stored_cnt, gperm, permuted, guarded, acc
                )
                gains = exchange_gains(f, np.ascontiguousarray(g), exchanges_parallel)
                for ei in range(len(gains)):
                    events.append(
                        (
                            0.0,
                            float(p),
                            float(pa[ei]),
                            float(pb[ei]),
                            float(p),
                            float(blocks),
                            float(ei),
                            gains[ei],
                            TOL,
                            (1.0 if gains[ei] > TOL else 0.0),
                        )
                    )
                if first_look:
                    # the junction's best gain (value) and its exchange by the tie rule (all pairs)
                    top, top_size, top_wide = _exchange_winner(gains, pa, pb, X, p, tie_q, -np.inf, reverse)
                    if top < 0:
                        top = 0
                    junction_gain[p - j0] = np.max(gains)
                    junction_pair[p - j0, 0] = pa[top]
                    junction_pair[p - j0, 1] = pb[top]
                    first_look = False
                best, tied, wide = _exchange_winner(gains, pa, pb, X, p, tie_q, TOL, reverse)
                if best < 0:
                    break
                if n_exchanged == EXCHANGE_ROUNDS:
                    # a positive exchange remains; the next pass looks again (O(K^3) per junction)
                    acc[_A_EXCHANGE_CAP] += 1
                    break
                n_exchanged += 1
                if tied > 1:
                    acc[_A_EXCHANGE_TIE] += 1
                if wide:
                    acc[_A_TIE_WIDE] += 1
                a, b = pa[best], pb[best]
                events.append(
                    (3.0, float(p), float(a), float(b), float(p), float(blocks), float(best), gains[best], TOL, 1.0)
                )
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
            # rank of each candidate in the last full ranking (gain > TOL), -1 otherwise or once tried
            stale_rank = np.full(n_here, -1, np.int64)
            rounds = 0
            while True:
                it_cand = np.empty(n_here, np.int64)
                it_kind = np.empty(n_here, np.int64)
                it_k = np.empty(n_here, np.int64)
                it_p1 = np.empty(n_here, np.int64)
                it_p2 = np.empty(n_here, np.int64)
                it_rs = np.empty(n_here, np.int64)
                n_act = _active_items(
                    p,
                    X,
                    lab,
                    cand_start,
                    cand_kind,
                    cand_end,
                    cand_k0,
                    cand_row_start,
                    cand_rows,
                    accepted,
                    it_cand,
                    it_kind,
                    it_k,
                    it_p1,
                    it_p2,
                    it_rs,
                )
                if n_act == 0:
                    break
                stale = rounds >= CONTENT_FULL_ROUNDS
                if stale:
                    # the untried item ranked first by the last full ranking, rescored alone
                    pick = -1
                    for i in range(n_act):
                        rk = stale_rank[it_cand[i] - c0]
                        if rk >= 0 and (pick < 0 or rk < stale_rank[it_cand[pick] - c0]):
                            pick = i
                    if pick < 0:
                        break
                    if rounds == CONTENT_FULL_ROUNDS:
                        acc[_A_ROUND_CAP] += 1
                    stale_rank[it_cand[pick] - c0] = -1
                    it_cand[0] = it_cand[pick]
                    it_kind[0] = it_kind[pick]
                    it_k[0] = it_k[pick]
                    it_p1[0] = it_p1[pick]
                    it_p2[0] = it_p2[pick]
                    it_rs[0] = it_rs[pick]
                    n_act = 1
                    acc[_A_STALE] += 1
                else:
                    acc[_A_ROUNDS] += 1
                rounds += 1
                act_gain, act_switch, act_gains_s, act_switches_s, chain, ahead = _score_items(
                    n_act,
                    it_kind,
                    it_k,
                    it_p1,
                    it_p2,
                    it_rs,
                    cand_rows,
                    p,
                    frontier,
                    haps,
                    leaves,
                    win,
                    offsets,
                    stored,
                    stored_cnt,
                    stored_scale,
                    gperm,
                    permuted,
                    guarded,
                    f,
                    fcnt,
                    chain,
                    ahead,
                    acc,
                    -1,
                )
                if optimistic:
                    for i in range(n_act):
                        if it_kind[i] == MOVE_COARSE and act_gain[i] > TOL:
                            acc[_A_LEX_ABORT] = 1.0
                            return (
                                junction_gain,
                                junction_pair,
                                gperm,
                                np.ascontiguousarray(f),
                                np.ascontiguousarray(fcnt),
                                frontier,
                                accepted_any,
                                ahead,
                                chain,
                            )
                # the best guard-passing candidate: tie classes of the gains, content order inside
                order = np.empty(n_act, np.int64)
                n_ranked = 0
                for i in range(n_act):
                    acc[_A_EVALUATED] += 1
                    evaluated[it_cand[i] - c0] = True
                    events.append(
                        (
                            1.0,
                            float(p),
                            float(it_k[i]),
                            -1.0,
                            float(it_p1[i]),
                            float(it_p2[i]),
                            float(it_cand[i]),
                            act_gain[i],
                            TOL,
                            (1.0 if act_gain[i] > TOL else 0.0),
                        )
                    )
                    if act_gain[i] > TOL:
                        order[n_ranked] = i
                        n_ranked += 1
                class_size = np.ones(max(n_ranked, 1), np.int64)
                _rank_moves(
                    order,
                    n_ranked,
                    act_gain,
                    X,
                    cand_rows,
                    it_k,
                    it_p1,
                    it_p2,
                    it_rs,
                    it_cand,
                    tie_q,
                    class_size,
                    acc,
                    reverse,
                )
                if not stale:
                    stale_rank[:] = -1
                    for r in range(n_ranked):
                        stale_rank[it_cand[order[r]] - c0] = r
                chosen = -1
                chosen_rank = -1
                for r in range(n_ranked):
                    i = order[r]
                    gain = act_gain[i]
                    if guarded and it_kind[i] == MOVE_COARSE:
                        # Lazy lexicographic rescoring of the coarse item the loop reaches (_score_items).
                        lex_gain, lex_switch, lex_gains_s, lex_switches_s, chain, ahead = _score_items(
                            n_act,
                            it_kind,
                            it_k,
                            it_p1,
                            it_p2,
                            it_rs,
                            cand_rows,
                            p,
                            frontier,
                            haps,
                            leaves,
                            win,
                            offsets,
                            stored,
                            stored_cnt,
                            stored_scale,
                            gperm,
                            permuted,
                            guarded,
                            f,
                            fcnt,
                            chain,
                            ahead,
                            acc,
                            i,
                        )
                        if lex_gain[i] != gain:
                            raise InvariantError("founder refinement: lexicographic and plain gains differ")
                        act_switch[i] = lex_switch[i]
                        act_gains_s[i] = lex_gains_s[i]
                        act_switches_s[i] = lex_switches_s[i]
                    if guarded and it_kind[i] == MOVE_COARSE:
                        events.append(
                            (
                                5.0,
                                float(p),
                                float(it_k[i]),
                                -1.0,
                                float(it_p1[i]),
                                float(it_p2[i]),
                                float(it_cand[i]),
                                gain + act_switch[i],
                                -GUARD_TOLERANCE,
                                (1.0 if gain + act_switch[i] < -GUARD_TOLERANCE else 0.0),
                            )
                        )
                    if guarded and it_kind[i] == MOVE_COARSE and (gain + act_switch[i] < -GUARD_TOLERANCE):
                        # the move lowers the genotype fit E = F + switch cost
                        acc[_A_GUARD_TESTS] += 1
                        shared, chain, ahead = _uncredited_switches(
                            haps,
                            leaves,
                            win,
                            offsets,
                            p,
                            it_k[i],
                            it_p1[i],
                            it_p2[i],
                            cand_rows,
                            it_rs[i],
                            act_gains_s[i],
                            act_switches_s[i],
                            frontier,
                            local,
                            local_cnt,
                            stored,
                            stored_cnt,
                            gperm,
                            permuted,
                            f,
                            fcnt,
                            first,
                            second,
                            chain,
                            ahead,
                            acc,
                            reverse,
                        )
                        events.append(
                            (
                                2.0,
                                float(p),
                                float(it_k[i]),
                                -1.0,
                                float(it_p1[i]),
                                float(it_p2[i]),
                                float(it_cand[i]),
                                gain - shared,
                                TOL,
                                (1.0 if gain - shared > TOL else 0.0),
                            )
                        )
                        if not gain - shared > TOL:
                            acc[_A_GUARD_SWITCHES] += shared
                            acc[_A_GUARD_REJECTED] += 1
                            continue
                        acc[_A_GUARD_RELAXED] += 1
                    chosen = i
                    chosen_rank = r
                    break
                if chosen < 0:
                    if stale:
                        continue
                    break
                if class_size[chosen_rank] > 1:
                    acc[_A_CONTENT_TIE] += 1
                i = chosen
                gain = act_gain[i]
                accepted[it_cand[i] - c0] = True
                k, p1, p2 = it_k[i], it_p1[i], it_p2[i]
                events.append(
                    (
                        4.0,
                        float(p),
                        float(k),
                        float(it_kind[i]),
                        float(p1),
                        float(p2),
                        float(it_cand[i]),
                        gain,
                        TOL,
                        1.0,
                    )
                )
                for x in range(p1, p2):
                    X[k, x] = cand_rows[it_rs[i] + x - p1]
                _set_rows(haps, leaves, offsets, k, p1, cand_rows, it_rs[i], it_rs[i] + p2 - p1)
                if p2 <= frontier:
                    chain = _chain_from(chain, p2, guarded)
                else:
                    frontier = p2
                    chain = Chain(False, chain.start, chain.end, chain.values, chain.counts, chain.scales)
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
                f, fcnt = advance_lex(haps, win, offsets, f, fcnt, p, p + 1)
            else:
                f = advance(haps, win, offsets, f, p, p + 1)
        if accepted_here:
            accepted_any = True
    return (
        junction_gain,
        junction_pair,
        gperm,
        np.ascontiguousarray(f),
        np.ascontiguousarray(fcnt),
        frontier,
        accepted_any,
        ahead,
        chain,
    )


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
    return (
        start,
        np.asarray(kind, np.int64),
        np.asarray(end, np.int64),
        np.asarray(label, np.int64),
        np.asarray(row_start, np.int64),
        np.concatenate(rows) if rows else np.zeros(0, np.int64),
    )


def accept_pass(
    orientation,
    X,
    stored,
    stored_cnt,
    stored_scale,
    props,
    window,
    guarded,
    budget,
    exchanges=True,
    start_fwd=None,
    start_fcnt=None,
    scale=0.0,
    *,
    reverse,
    optimistic=False,
):
    """One exact acceptance pass on `orientation` (search.Orientation); X (K x B, that orientation's
    block order) is modified in place.

    stored, stored_cnt, stored_scale: the stored backward messages, switch costs (guard only) and
    normalisers of the pass's start panel in this orientation. window: the log evidence and switch
    costs of the panel's K in this orientation (window.Windows.view; the pass covers the stretch
    ahead of its junction as it moves). props: the content proposals
    {junction: [(MOVE_FINE | MOVE_COARSE, end block, founder, rows)]}. guarded: the shared-junction
    guard applies to coarse moves (a search with an L1 context). exchanges=False scores and applies
    content proposals only (an addition round's copy-only pass). start_fwd / start_fcnt: the
    forward messages (and counts) of X itself in this orientation, if known; until the first
    accepted move the carried forward message is read from them. budget: the live thread share
    (the explicit stage team), re-read before every JUNCTION_BATCH junctions. scale: |F| of the pass's
    start panel (the tie quantum's scale, tie_rules.quantum("refine_F", scale)). reverse (required): the content-order
    convention of the pass. The fixed-K search passes ``first == "rev"``; a copy-only pass
    passes the run's TieSettings reversal.
    optimistic (unguarded passes only): the pass stops, with counters["lex_abort"] = 1 and X
    partly edited, at the first coarse item whose gain exceeds TOL; until then a guarded pass of the
    same inputs makes exactly the same decisions (its gains and orders equal the plain ones).
    Returns (counters {ACC_KEYS}, junction_gain (B + 1), junction_pair (B + 1, 2)).
    """
    tie_q = policy("assembly.refine.accept", reverse=reverse).quantum(scale)
    founders, blocks = X.shape
    if stored.dense:
        # Preserve the original pristine forward shortcut on the dense path.
        if start_fwd is not None:
            start_fwd = start_fwd.data
    else:
        start_fwd = start_fcnt = None
    stored.clear()
    stored = stored.source()
    samples, states = stored[0].shape[1], stored[0].shape[2]
    haps = orientation.haps(X)
    cands = candidate_arrays(props, blocks)
    count2 = np.zeros((1, 1))
    count3 = np.zeros((1, 1, 1))
    if guarded:
        # Sparse carried-message checkpoints. A guard query reconstructs at most
        # JUNCTION_BATCH - 1 passed blocks; suffix edits cannot change that prefix.
        checkpoints = (blocks - 1) // JUNCTION_BATCH + 1
        local = np.empty((checkpoints, samples, states))
        local_cnt = np.empty((checkpoints, samples, states))
    else:
        local, local_cnt = np.zeros((1, 1, 1)), count3
    events = List.empty_list(types.UniTuple(types.float64, 10))
    acc = np.zeros(len(ACC_KEYS))
    f = np.zeros((samples, states))
    fcnt = np.zeros((samples, states)) if guarded else count2
    gperm = np.arange(states, dtype=np.int64)
    lab = np.arange(founders, dtype=np.int64)
    pa, pb = exchange_pairs(founders)
    ahead = Ahead(
        False,
        0,
        -1,
        np.zeros((1, 1, 1)),
        count3,
        np.zeros((1, 0)),
        np.zeros((1, 0), np.int32),
        np.zeros((1, 1)),
        np.zeros((1, 1), np.int32),
        np.zeros((1, 0)),
    )
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
        team = sample_threads(samples, budget)
        with budget.scope(team):
            (jg, jp, gperm, f, fcnt, frontier, accepted, ahead, chain) = pass_chunk(
                haps,
                X,
                orientation.leaves,
                window,
                orientation.offsets,
                p,
                stop,
                frontier,
                f,
                fcnt,
                local,
                local_cnt,
                stored,
                stored_cnt if guarded else count3,
                stored_scale,
                gperm,
                lab,
                *cands,
                pa,
                pb,
                bool(guarded),
                bool(exchanges),
                parallel_exchanges(samples, founders, team),
                acc,
                ahead,
                chain,
                start_fwd,
                start_fcnt,
                pristine,
                tie_q,
                bool(reverse),
                bool(optimistic and not guarded),
                events,
            )
        pristine = pristine and not accepted
        junction_gain[p:stop] = jg
        junction_pair[p:stop] = jp
        p = stop
        if acc[_A_LEX_ABORT]:
            break
    record = {
        key: (int(round(value)) if key in INTEGER_KEYS else float(value))
        for key, value in zip(ACC_KEYS, acc.tolist(), strict=True)
    }
    record["decision_trace"] = np.asarray(list(events), dtype=np.float64).reshape(-1, 10)
    return record, junction_gain, junction_pair


__all__ = ["pass_chunk", "candidate_arrays", "accept_pass"]
