"""Exact normalised block-boundary messages of the objective F (objective.py).

Owns fixed-order message recurrences and their clamped reverse views; no storage or thread admission.
Method: docs/methods/founder_refinement.md#messages (RC05–RC07).

Owns dense forward/backward messages and carried block chains.
Must not choose K, read storage or configure runtime.
Method: docs/methods/founder_refinement.md#messages (RC04-RC06).
STYLE_40 exception: numerical kernels retain flat scalar/array arguments because
Numba cannot lower the frozen Python boundary records; the stage uses those records.
Hot recurrences remain whole to preserve their loop/update order and avoid temporary
materialisations. Parallel kernels write disjoint sample slices; sample reductions
are fixed serial folds, independent of the explicit stage team's size.


For every sample F_s is a max-plus chain, so for any panel edit confined to blocks [a, c) the exact
change of F is

    dF = sum_s [max_q (f_s(a) (x) T_new[a, c) (x) g_s(c))_q - max_q (f_s(a) (x) T_old[a, c) (x) g_s(c))_q]

with f the carried forward message (transition-then-emission per informative site) and g the
stored backward message (emission-then-transition), both normalised to maximum 0 at every block
boundary (the per-block normalisers are kept: fscale, bscale). A suffix exchange of two founders
relabels the suffix states only (the transition is founder-symmetric), so its gain needs the two
messages at its junction alone.

The optional switch costs form a lexicographic max-plus pair (score first, then the lower switch
cost among exact score ties). They give the genotype-fit term E = F + switch cost of a
maximum-score path, decomposed exactly like F; only the coarse-move guard reads them. They never
change F.

Every kernel reads the log evidence and the switch costs of its orientation and K through a window
(window.py): it covers each stretch of whole blocks before it reads it, a stretch at
most the window's capacity (the whole component for a resident window), and carries every sample's
message from stretch to stretch, so the recursion and its values are those of one pass. The
chromosome-wide kernels compute the samples of a range [s0, s1) into arrays the caller allocates
(every sample's recursion is its own, so a search runs them range by range, re-reading its thread
share between ranges) and build the shared pair-dosage table per segment of whole blocks of at most
DOSAGE_BYTES, so it stays small at large K. All kernels use the inline site steps of objective.py:
the forward recursion of messages(), its in-place update, advance() and the forward chain perform the
same operations, so their messages are bitwise equal (the updates' early stops and the proposal
cache rely on it).
"""

from __future__ import annotations

import numpy as np
from numba import prange

from haplotype_reconstruction.core.kernels import kernel

from .objective import (
    backward_sites,
    backward_sites_lex,
    forward_sites,
    forward_sites_lex,
    normalise,
    pair_dosages,
    row_max,
    transition,
    transition_lex,
    unordered_pairs,
)
from .overloads import overloads_for
from .window import FOUNDERS, capacity, cover, samples_of, stretch, stretch_back
from .window import boundary_costs as _boundary_costs

# Largest pair-dosage table (sites x states int8) of one segment of a chromosome-wide message
# kernel or of a candidate's blocks; a block larger than this forms its own segment.
DOSAGE_BYTES = 32 * 1024**2


@kernel(overloads=overloads_for(__name__, "segments"), cache=True, nogil=True)
def segments(offsets, states, limit):
    """Boundaries 0 = t_0 < ... < t_n = B of consecutive segments of whole blocks whose
    pair-dosage tables fit DOSAGE_BYTES and that hold at most `limit` sites (a window's capacity); a
    larger block is a segment of its own."""
    blocks = len(offsets) - 1
    limit = min(limit, max(1, DOSAGE_BYTES // max(1, states)))
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
    return bounds[: n + 1]


@kernel(overloads=overloads_for(__name__, "messages"), cache=True, parallel=True, nogil=True)
def messages(haps, win, offsets, s0, s1, fwd, bwd, fscale, bscale):
    """Normalised forward (carried) and backward (stored) messages at every block boundary of the
    samples [s0, s1), into zeroed arrays (message_arrays): fwd[b + 1] = (fwd[b] processed over
    block b) - fscale[b] and bwd[b] = (bwd[b + 1] processed backwards over block b) - bscale[b], so
    the unnormalised backward message at x is bwd[x] + sum_{b >= x} bscale[b]."""
    founders = len(haps)
    first, second = unordered_pairs(founders)
    bounds = segments(offsets, len(first), capacity(win))
    for t in range(len(bounds) - 1):
        b0, b1 = bounds[t], bounds[t + 1]
        lo = offsets[b0]
        cover(win, lo, offsets[b1], s0, s1, False)
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            R = np.empty(founders)
            e = np.empty(7)
            row = fwd[b0, s].copy()
            for b in range(b0, b1):
                forward_sites(row, dos, lo, win, s, offsets[b], offsets[b + 1], R, e)
                fscale[b, s] = normalise(row)
                fwd[b + 1, s] = row
    for t in range(len(bounds) - 2, -1, -1):
        b0, b1 = bounds[t], bounds[t + 1]
        lo = offsets[b0]
        cover(win, lo, offsets[b1], s0, s1, True)
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            R = np.empty(founders)
            e = np.empty(7)
            row = bwd[b1, s].copy()
            for b in range(b1 - 1, b0 - 1, -1):
                backward_sites(row, dos, lo, win, s, offsets[b], offsets[b + 1], R, e)
                bscale[b, s] = normalise(row)
                bwd[b, s] = row


@kernel(overloads=overloads_for(__name__, "messages_lex"), cache=True, parallel=True, nogil=True)
def messages_lex(haps, win, offsets, s0, s1, fwd, bwd, fcnt, bcnt, fscale, bscale):
    """messages() plus the switch costs of the lexicographic (score, -switch cost) maxima, of the
    samples [s0, s1), into zeroed arrays (message_arrays(..., lex=True))."""
    founders = len(haps)
    first, second = unordered_pairs(founders)
    bounds = segments(offsets, len(first), capacity(win))
    for t in range(len(bounds) - 1):
        b0, b1 = bounds[t], bounds[t + 1]
        lo = offsets[b0]
        cover(win, lo, offsets[b1], s0, s1, False)
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            R = np.empty(founders)
            Rc = np.empty(founders)
            e = np.empty(7)
            row = fwd[b0, s].copy()
            cnt = fcnt[b0, s].copy()
            for b in range(b0, b1):
                forward_sites_lex(row, cnt, dos, lo, win, s, offsets[b], offsets[b + 1], R, Rc, e)
                fscale[b, s] = normalise(row)
                fwd[b + 1, s] = row
                fcnt[b + 1, s] = cnt
    for t in range(len(bounds) - 2, -1, -1):
        b0, b1 = bounds[t], bounds[t + 1]
        lo = offsets[b0]
        cover(win, lo, offsets[b1], s0, s1, True)
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            R = np.empty(founders)
            Rc = np.empty(founders)
            e = np.empty(7)
            row = bwd[b1, s].copy()
            cnt = bcnt[b1, s].copy()
            for b in range(b1 - 1, b0 - 1, -1):
                backward_sites_lex(row, cnt, dos, lo, win, s, offsets[b], offsets[b + 1], R, Rc, e)
                bscale[b, s] = normalise(row)
                bwd[b, s] = row
                bcnt[b, s] = cnt


# The range kernels below run over stretches of whole blocks of at most the window's capacity, left to
# right (forward recursions) or right to left (backward ones), each covered before it is read; a sample's
# message is carried from stretch to stretch in the output arrays.
@kernel(overloads=overloads_for(__name__, "advance"), cache=True, parallel=True, nogil=True)
def advance(haps, win, offsets, start, a, c):
    """Carry the normalised forward messages start (S x Q) from boundary a over blocks [a, c)."""
    founders = len(haps)
    first, second = unordered_pairs(founders)
    samples = samples_of(win)
    out = start.copy()
    b0 = a
    while b0 < c:
        b1 = stretch(offsets, b0, c, capacity(win))
        lo = offsets[b0]
        cover(win, lo, offsets[b1], 0, samples, False)
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(samples):
            R = np.empty(founders)
            e = np.empty(7)
            row = out[s]
            for b in range(b0, b1):
                forward_sites(row, dos, lo, win, s, offsets[b], offsets[b + 1], R, e)
                normalise(row)
        b0 = b1
    return out


@kernel(overloads=overloads_for(__name__, "advance_lex"), cache=True, parallel=True, nogil=True)
def advance_lex(haps, win, offsets, start, start_cnt, a, c):
    """advance() with the lexicographic switch costs."""
    founders = len(haps)
    first, second = unordered_pairs(founders)
    samples = samples_of(win)
    out = start.copy()
    out_cnt = start_cnt.copy()
    b0 = a
    while b0 < c:
        b1 = stretch(offsets, b0, c, capacity(win))
        lo = offsets[b0]
        cover(win, lo, offsets[b1], 0, samples, False)
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(samples):
            R = np.empty(founders)
            Rc = np.empty(founders)
            e = np.empty(7)
            row = out[s]
            cnt = out_cnt[s]
            for b in range(b0, b1):
                forward_sites_lex(row, cnt, dos, lo, win, s, offsets[b], offsets[b + 1], R, Rc, e)
                normalise(row)
        b0 = b1
    return out, out_cnt


@kernel(overloads=overloads_for(__name__, "forward_boundaries"), cache=True, parallel=True, nogil=True)
def forward_boundaries(haps, win, offsets, start, a, c):
    """Normalised carried forward messages at boundaries a..c (index 0..c-a) from start (S x Q) at
    a over blocks [a, c) of the panel haps (advance()'s recursion, every boundary kept)."""
    founders = len(haps)
    first, second = unordered_pairs(founders)
    states = len(first)
    samples = samples_of(win)
    out = np.empty((c - a + 1, samples, states))
    out[0] = start
    b0 = a
    while b0 < c:
        b1 = stretch(offsets, b0, c, capacity(win))
        lo = offsets[b0]
        cover(win, lo, offsets[b1], 0, samples, False)
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(samples):
            R = np.empty(founders)
            e = np.empty(7)
            row = out[b0 - a, s].copy()
            for b in range(b0, b1):
                forward_sites(row, dos, lo, win, s, offsets[b], offsets[b + 1], R, e)
                normalise(row)
                out[b - a + 1, s] = row
        b0 = b1
    return out


@kernel(overloads=overloads_for(__name__, "backward_chain"), cache=True, parallel=True, nogil=True)
def backward_chain(haps, win, offsets, end, gperm, a, c):
    """Exact stored-convention backward messages at boundaries a..c (index 0..c-a) and the
    per-block normalisers (index b - a for blocks a..c-1). end is the stored message at c, read
    through the label permutation gperm (current state -> stored state)."""
    founders = len(haps)
    first, second = unordered_pairs(founders)
    states = len(first)
    samples = samples_of(win)
    out = np.empty((c - a + 1, samples, states))
    scale = np.zeros((max(c - a, 0), samples))
    for s in prange(samples):
        for q in range(states):
            out[c - a, s, q] = end[s, gperm[q]]
    b1 = c
    while b1 > a:
        b0 = stretch_back(offsets, a, b1, capacity(win))
        lo = offsets[b0]
        cover(win, lo, offsets[b1], 0, samples, True)
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(samples):
            R = np.empty(founders)
            e = np.empty(7)
            row = out[b1 - a, s].copy()
            for b in range(b1 - 1, b0 - 1, -1):
                backward_sites(row, dos, lo, win, s, offsets[b], offsets[b + 1], R, e)
                scale[b - a, s] = normalise(row)
                out[b - a, s] = row
        b1 = b0
    return out, scale


@kernel(overloads=overloads_for(__name__, "backward_chain_lex"), cache=True, parallel=True, nogil=True)
def backward_chain_lex(haps, win, offsets, end, end_cnt, gperm, a, c):
    """backward_chain() with the lexicographic switch costs."""
    founders = len(haps)
    first, second = unordered_pairs(founders)
    states = len(first)
    samples = samples_of(win)
    out = np.empty((c - a + 1, samples, states))
    out_cnt = np.empty((c - a + 1, samples, states))
    scale = np.zeros((max(c - a, 0), samples))
    for s in prange(samples):
        for q in range(states):
            out[c - a, s, q] = end[s, gperm[q]]
            out_cnt[c - a, s, q] = end_cnt[s, gperm[q]]
    b1 = c
    while b1 > a:
        b0 = stretch_back(offsets, a, b1, capacity(win))
        lo = offsets[b0]
        cover(win, lo, offsets[b1], 0, samples, True)
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(samples):
            R = np.empty(founders)
            Rc = np.empty(founders)
            e = np.empty(7)
            row = out[b1 - a, s].copy()
            cnt = out_cnt[b1 - a, s].copy()
            for b in range(b1 - 1, b0 - 1, -1):
                backward_sites_lex(row, cnt, dos, lo, win, s, offsets[b], offsets[b + 1], R, Rc, e)
                scale[b - a, s] = normalise(row)
                out[b - a, s] = row
                out_cnt[b - a, s] = cnt
        b1 = b0
    return out, out_cnt, scale


@kernel(overloads=overloads_for(__name__, "max_lead"), inline="always", nogil=True)
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


@kernel(overloads=overloads_for(__name__, "forward_chain"), cache=True, parallel=True, nogil=True)
def forward_chain(haps, win, offsets, start, a, c):
    """Carried forward messages at boundaries a..c (index 0..c-a) from start at a, and per block
    the maximum before the block's normalisation and its first state. (Release 2.8: the candidate
    kernels are dense, so the per-site leaders of 2.6's fast path are empty (S x 0) arrays.)"""
    founders = len(haps)
    first, second = unordered_pairs(founders)
    states = len(first)
    samples = samples_of(win)
    out = np.empty((c - a + 1, samples, states))
    site_max = np.empty((samples, 0))
    site_lead = np.empty((samples, 0), np.int32)
    block_max = np.empty((c - a, samples))
    block_lead = np.empty((c - a, samples), np.int32)
    out[0] = start
    b0 = a
    while b0 < c:
        b1 = stretch(offsets, b0, c, capacity(win))
        base = offsets[b0]
        cover(win, base, offsets[b1], 0, samples, False)
        dos = pair_dosages(haps, first, second, base, offsets[b1])
        for s in prange(samples):
            R = np.empty(founders)
            e = np.empty(7)
            row = out[b0 - a, s].copy()
            for b in range(b0, b1):
                forward_sites(row, dos, base, win, s, offsets[b], offsets[b + 1], R, e)
                m, lead = max_lead(row)
                block_max[b - a, s] = m
                block_lead[b - a, s] = lead
                for q in range(states):
                    row[q] -= m
                out[b - a + 1, s] = row
        b0 = b1
    return out, site_max, site_lead, block_max, block_lead


@kernel(overloads=overloads_for(__name__, "forward_chain_lex"), cache=True, parallel=True, nogil=True)
def forward_chain_lex(haps, win, offsets, start, start_cnt, a, c):
    """forward_chain with the lexicographic switch costs (messages and costs at boundaries a..c,
    block leaders; empty per-site arrays)."""
    founders = len(haps)
    first, second = unordered_pairs(founders)
    states = len(first)
    samples = samples_of(win)
    out = np.empty((c - a + 1, samples, states))
    out_cnt = np.empty((c - a + 1, samples, states))
    site_max = np.empty((samples, 0))
    site_lead = np.empty((samples, 0), np.int32)
    site_cnt = np.empty((samples, 0))
    block_max = np.empty((c - a, samples))
    block_lead = np.empty((c - a, samples), np.int32)
    out[0] = start
    out_cnt[0] = start_cnt
    b0 = a
    while b0 < c:
        b1 = stretch(offsets, b0, c, capacity(win))
        base = offsets[b0]
        cover(win, base, offsets[b1], 0, samples, False)
        dos = pair_dosages(haps, first, second, base, offsets[b1])
        for s in prange(samples):
            R = np.empty(founders)
            Rc = np.empty(founders)
            e = np.empty(7)
            row = out[b0 - a, s].copy()
            cnt = out_cnt[b0 - a, s].copy()
            for b in range(b0, b1):
                forward_sites_lex(row, cnt, dos, base, win, s, offsets[b], offsets[b + 1], R, Rc, e)
                m, lead = max_lead(row)
                block_max[b - a, s] = m
                block_lead[b - a, s] = lead
                for q in range(states):
                    row[q] -= m
                out[b - a + 1, s] = row
                out_cnt[b - a + 1, s] = cnt
        b0 = b1
    return out, out_cnt, site_max, site_lead, block_max, block_lead, site_cnt


def boundary_costs(win, offsets):
    """float64 (B + 1, S): the switch cost (the forward window's K) entering each sample's first
    informative site at or after every block boundary (+inf when there is none, or it is the
    sample's first)."""
    return _boundary_costs(win[0], win[1], win[3], offsets, int(win[8][FOUNDERS]))


@kernel(overloads=overloads_for(__name__, "clamp_reverse"), cache=True, parallel=True, nogil=True)
def clamp_reverse(carried, entering):
    """Stored messages of the reversed orientation from carried forward messages: out[B - c] is
    carried[c] taken through the transition entering the sample's next informative site (forward
    order; the reversed orientation's stored convention ends with it), renormalised. entering:
    boundary_costs() of the forward orientation."""
    nb, samples, states = carried.shape
    founders = 1
    while founders * (founders + 1) // 2 < states:
        founders += 1
    out = np.empty_like(carried)
    for c in prange(nb):
        R = np.empty(founders)
        row = np.empty(states)
        for s in range(samples):
            m = row_max(carried[c, s])
            row[:] = carried[c, s]
            transition(row, entering[c, s], R)
            for q in range(states):
                out[nb - 1 - c, s, q] = row[q] - m
    return out


@kernel(overloads=overloads_for(__name__, "clamp_reverse_lex"), cache=True, parallel=True, nogil=True)
def clamp_reverse_lex(carried, carried_cnt, entering):
    """clamp_reverse() with the lexicographic switch costs."""
    nb, samples, states = carried.shape
    founders = 1
    while founders * (founders + 1) // 2 < states:
        founders += 1
    out = np.empty_like(carried)
    out_cnt = np.empty_like(carried_cnt)
    for c in prange(nb):
        R = np.empty(founders)
        Rc = np.empty(founders)
        row = np.empty(states)
        cnt = np.empty(states)
        for s in range(samples):
            m = row_max(carried[c, s])
            row[:] = carried[c, s]
            cnt[:] = carried_cnt[c, s]
            transition_lex(row, cnt, entering[c, s], R, Rc)
            for q in range(states):
                out[nb - 1 - c, s, q] = row[q] - m
                out_cnt[nb - 1 - c, s, q] = cnt[q]
    return out, out_cnt


__all__ = [
    "segments",
    "messages",
    "messages_lex",
    "advance",
    "advance_lex",
    "forward_boundaries",
    "backward_chain",
    "backward_chain_lex",
    "max_lead",
    "forward_chain",
    "forward_chain_lex",
    "boundary_costs",
    "clamp_reverse",
    "clamp_reverse_lex",
]


@kernel(overloads=overloads_for(__name__, "update_messages"), cache=True, parallel=True, nogil=True)
def update_messages(haps, win, offsets, changed, fwd, bwd, fscale, bscale, backward_known, s0, s1):
    """messages() of the current panel (haps) for the samples [s0, s1), in place on those of an
    earlier panel whose alleles differ only on the blocks where changed[b]: fwd, fscale always,
    bwd, bscale when backward_known (otherwise every block is computed into them; bwd[B] must
    hold zeros).

    A block's forward step reads only the message at its start and the block's alleles, so a
    sample's messages over an unchanged block whose start message is unchanged are the earlier
    ones. Per sample, a block is recomputed (messages()' recursion from the stored start message)
    only if it changed or its start message did; a recomputed end message that equals the earlier
    one ends the run. Likewise backwards. Bitwise messages() of the current panel. Only the
    segments it recomputes are covered."""
    founders = len(haps)
    first, second = unordered_pairs(founders)
    states = len(first)
    bounds = segments(offsets, states, capacity(win))
    # dirty[s - s0]: sample s's message at the current segment boundary changed.
    dirty = np.zeros(s1 - s0, np.bool_)
    for t in range(len(bounds) - 1):
        b0, b1 = bounds[t], bounds[t + 1]
        if not (dirty.any() or changed[b0:b1].any()):
            continue
        lo = offsets[b0]
        cover(win, lo, offsets[b1], s0, s1, False)
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = np.empty(states)
            R = np.empty(founders)
            e = np.empty(7)
            for b in range(b0, b1):
                if not (dirty[s - s0] or changed[b]):
                    continue
                row[:] = fwd[b, s]
                forward_sites(row, dos, lo, win, s, offsets[b], offsets[b + 1], R, e)
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
        cover(win, lo, offsets[b1], s0, s1, True)
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = np.empty(states)
            R = np.empty(founders)
            e = np.empty(7)
            for b in range(b1 - 1, b0 - 1, -1):
                if not (dirty[s - s0] or changed[b]):
                    continue
                row[:] = bwd[b + 1, s]
                backward_sites(row, dos, lo, win, s, offsets[b], offsets[b + 1], R, e)
                bscale[b, s] = normalise(row)
                same = backward_known
                for q in range(states):
                    same = same and row[q] == bwd[b, s, q]
                    bwd[b, s, q] = row[q]
                dirty[s - s0] = not same


@kernel(overloads=overloads_for(__name__, "update_messages_lex"), cache=True, parallel=True, nogil=True)
def update_messages_lex(haps, win, offsets, changed, fwd, bwd, fcnt, bcnt, fscale, bscale, backward_known, s0, s1):
    """update_messages() for messages_lex(): the messages and their switch costs (a run ends where
    both equal the earlier ones), samples [s0, s1). Bitwise messages_lex() of the current panel."""
    founders = len(haps)
    first, second = unordered_pairs(founders)
    states = len(first)
    bounds = segments(offsets, states, capacity(win))
    dirty = np.zeros(s1 - s0, np.bool_)
    for t in range(len(bounds) - 1):
        b0, b1 = bounds[t], bounds[t + 1]
        if not (dirty.any() or changed[b0:b1].any()):
            continue
        lo = offsets[b0]
        cover(win, lo, offsets[b1], s0, s1, False)
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = np.empty(states)
            cnt = np.empty(states)
            R = np.empty(founders)
            Rc = np.empty(founders)
            e = np.empty(7)
            for b in range(b0, b1):
                if not (dirty[s - s0] or changed[b]):
                    continue
                row[:] = fwd[b, s]
                cnt[:] = fcnt[b, s]
                forward_sites_lex(row, cnt, dos, lo, win, s, offsets[b], offsets[b + 1], R, Rc, e)
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
        cover(win, lo, offsets[b1], s0, s1, True)
        dos = pair_dosages(haps, first, second, lo, offsets[b1])
        for s in prange(s0, s1):
            row = np.empty(states)
            cnt = np.empty(states)
            R = np.empty(founders)
            Rc = np.empty(founders)
            e = np.empty(7)
            for b in range(b1 - 1, b0 - 1, -1):
                if not (dirty[s - s0] or changed[b]):
                    continue
                row[:] = bwd[b + 1, s]
                cnt[:] = bcnt[b + 1, s]
                backward_sites_lex(row, cnt, dos, lo, win, s, offsets[b], offsets[b + 1], R, Rc, e)
                bscale[b, s] = normalise(row)
                same = backward_known
                for q in range(states):
                    same = same and row[q] == bwd[b, s, q] and cnt[q] == bcnt[b, s, q]
                    bwd[b, s, q] = row[q]
                    bcnt[b, s, q] = cnt[q]
                dirty[s - s0] = not same
