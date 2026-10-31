"""Sparse block messages, reconstructed by their original block-normalised recursion.

Large, wide message sets keep both ends and every CHECKPOINT_BLOCKS boundary,
plus per-block normalisers. Others retain every boundary and use the original kernels.
Updates stop only at unchanged checkpoints (scores and switch costs). A reversed pass
reconstructs the clamped original forward recursion, rather than a differently rounded
reverse backward recursion. Reader caches are private; checkpoints and start-panel alleles
are read-only until the next sweep, after all readers have finished.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from . import messages as M
from .objective import (backward_sites, backward_sites_lex, forward_sites, forward_sites_lex,
                        normalise, pair_dosages, unordered_pairs)
from .window import capacity, cover, stretch, stretch_back

# Fixed storage policy; measured by the SW5 replay/timing campaign.
CHECKPOINT_BLOCKS = 16
# Two segments accommodate a junction and a proposal's other flank.
CACHE_SEGMENTS = 2
# Require both a substantial total footprint and a wide boundary before paying
# for recomputation. The length allowance keeps long, modest-N searches fast.
MESSAGE_BYTES = 512 << 20
MESSAGE_BOUNDARY_BYTES = 1 << 20


def stride(founders, blocks, samples, guarded):
    arrays = 4 if guarded else 2
    per_boundary = arrays * 8 * samples * (founders * (founders + 1) // 2)
    dense_bytes = (blocks + 1) * per_boundary
    allowance = max(MESSAGE_BYTES, (blocks + 1) * MESSAGE_BOUNDARY_BYTES)
    return CHECKPOINT_BLOCKS if dense_bytes > allowance else 1


def boundaries(blocks, step=CHECKPOINT_BLOCKS):
    return np.asarray([*range(0, blocks, step), blocks], np.int64)


@njit(cache=True, parallel=True, nogil=True)
def update(haps, win, offsets, bounds, changed, data, counts, scales, known,
           forward, lex, s0, s1):
    """Update one direction, comparing only checkpoint ends, over samples [s0, s1)."""
    founders = len(haps)
    first, second = unordered_pairs(founders)
    states = len(first)
    dirty = np.full(s1 - s0, not known, np.bool_)
    carry = np.empty((s1 - s0, states))
    carry_cnt = np.empty((s1 - s0, states)) if lex else np.empty((1, 1))
    n = len(bounds) - 1
    for step in range(n):
        t = step if forward else n - 1 - step
        a, c = bounds[t], bounds[t + 1]
        edited = changed[a:c].any()
        if not (edited or dirty.any()):
            continue
        source, target = (t, t + 1) if forward else (t + 1, t)
        for s in prange(s0, s1):
            if dirty[s - s0] or edited:
                carry[s - s0] = data[source, s]
                if lex:
                    carry_cnt[s - s0] = counts[source, s]
        edge = a if forward else c
        while (edge < c if forward else edge > a):
            if forward:
                b0 = edge
                b1 = stretch(offsets, b0, c, min(capacity(win), M.DOSAGE_BYTES // max(1, states)))
            else:
                b1 = edge
                b0 = stretch_back(offsets, a, b1, min(capacity(win), M.DOSAGE_BYTES // max(1, states)))
            lo = offsets[b0]
            cover(win, lo, offsets[b1], s0, s1, not forward)
            dos = pair_dosages(haps, first, second, lo, offsets[b1])
            for s in prange(s0, s1):
                if not (dirty[s - s0] or edited):
                    continue
                row = carry[s - s0]
                cnt = carry_cnt[s - s0] if lex else carry_cnt[0]
                R, Rc, e = np.empty(founders), np.empty(founders), np.empty(7)
                for j in range(b1 - b0):
                    b = b0 + j if forward else b1 - 1 - j
                    if forward:
                        if lex:
                            forward_sites_lex(row, cnt, dos, lo, win, s, offsets[b], offsets[b + 1], R, Rc, e)
                        else:
                            forward_sites(row, dos, lo, win, s, offsets[b], offsets[b + 1], R, e)
                    else:
                        if lex:
                            backward_sites_lex(row, cnt, dos, lo, win, s, offsets[b], offsets[b + 1], R, Rc, e)
                        else:
                            backward_sites(row, dos, lo, win, s, offsets[b], offsets[b + 1], R, e)
                    scales[b, s] = normalise(row)
            edge = b1 if forward else b0
        for s in prange(s0, s1):
            if not (dirty[s - s0] or edited):
                continue
            same = known
            for q in range(states):
                same = same and carry[s - s0, q] == data[target, s, q]
                data[target, s, q] = carry[s - s0, q]
                if lex:
                    same = same and carry_cnt[s - s0, q] == counts[target, s, q]
                    counts[target, s, q] = carry_cnt[s - s0, q]
            dirty[s - s0] = not same


@njit(cache=True, nogil=True)
def read(source, x):
    """One boundary from a private two-segment cache. Views last until its slot is refilled."""
    data, counts, bounds, haps, win, offsets, forward, reverse, entering, cache, cc, tags, lex = source
    # Dense sources are already in pass order, including an eagerly clamped
    # reverse pass. No search, reconstruction cache or recomputation on this path.
    if len(bounds) == len(offsets):
        return data[x], counts[x] if lex else counts[0]
    y = len(offsets) - 1 - x if reverse else x
    t = min(len(bounds) - 2, np.searchsorted(bounds, y, side="right") - 1)
    a, c = bounds[t], bounds[t + 1]
    if not reverse and y == a:
        return data[t], counts[t] if lex else counts[0]
    if not reverse and y == c:
        return data[t + 1], counts[t + 1] if lex else counts[0]
    slot = t % len(tags)
    if tags[slot] != t:
        if forward:
            if lex:
                values, costs, _, _, _, _, _ = M.forward_chain_lex(
                    haps, win, offsets, data[t], counts[t], a, c)
            else:
                values = M.forward_boundaries(haps, win, offsets, data[t], a, c)
                costs = counts
            if reverse:
                if lex:
                    values, costs = M.clamp_reverse_lex(values, costs, entering[a:c + 1])
                else:
                    values = M.clamp_reverse(values, entering[a:c + 1])
        else:
            ident = np.arange(data.shape[2])
            if lex:
                values, costs, _ = M.backward_chain_lex(
                    haps, win, offsets, data[t + 1], counts[t + 1], ident, a, c)
            else:
                values, _ = M.backward_chain(haps, win, offsets, data[t + 1], ident, a, c)
                costs = counts
        cache[slot, :c - a + 1] = values
        if lex:
            cc[slot, :c - a + 1] = costs
        tags[slot] = t
    i = c - y if reverse else y - a
    return cache[slot, i], cc[slot, i] if lex else counts[0]


class BoundaryMessages:
    """Boundary indexing for proposals/polish; Numba acceptance reads source().

    Parallel proposal solves receive flank dictionaries, so they do not share a mutable
    reconstruction cache or allocate a streaming window per proposal worker.
    """

    def __init__(self, data, counts, bounds, haps, orientation, forward, reverse=False, entering=None):
        self.data, self.counts, self.bounds = data, counts, bounds
        self.haps, self.orientation, self.forward = haps, orientation, forward
        self.reverse, self.entering = reverse, entering
        self.shape = (orientation.blocks + 1, *data.shape[1:])
        self._source = None
        self.dense = len(bounds) == self.shape[0]

    @property
    def nbytes(self):
        return self.data.nbytes

    def source(self):
        samples, states = self.data.shape[1:]
        lex = self.counts is not None
        width = int(np.max(np.diff(self.bounds))) + 1
        shape = (1, 1, 1, 1) if self.dense else (CACHE_SEGMENTS, width, samples, states)
        return (self.data, self.counts if lex else np.zeros((1, 1, 1)), self.bounds, self.haps,
                self.orientation.window(len(self.haps)), self.orientation.offsets, self.forward, self.reverse,
                self.entering if self.entering is not None else np.zeros((1, 1)), np.empty(shape),
                np.empty(shape) if lex else np.empty((1, 1, 1, 1)), np.full(CACHE_SEGMENTS, -1, np.int64), lex)

    def __getitem__(self, index):
        if self.dense:
            return self.data[index]
        if np.ndim(index):
            return np.asarray([self[int(i)] for i in index])
        if self._source is None:
            self._source = self.source()
        return read(self._source, int(index))[0].copy()

    def clear(self):
        self._source = None

    def flanks(self, indices):
        if self.dense:
            return self.data
        result = {int(i): self[int(i)] for i in sorted(set(indices))}
        self.clear()
        return result

    def reversed(self, entering):
        if self.dense:
            if self.counts is None:
                data, counts = M.clamp_reverse(self.data, entering), None
            else:
                data, counts = M.clamp_reverse_lex(self.data, self.counts, entering)
            return BoundaryMessages(data, counts, self.bounds, self.haps, self.orientation, False)
        return BoundaryMessages(self.data, self.counts, self.bounds, self.haps, self.orientation,
                                True, reverse=True, entering=entering)


def allocate(founders, blocks, samples, lex, haps, orientation, step):
    bounds = boundaries(blocks, step)
    shape = (len(bounds), samples, founders * (founders + 1) // 2)
    fc, bc = (np.zeros(shape), np.zeros(shape)) if lex else (None, None)
    fwd = BoundaryMessages(np.zeros(shape), fc, bounds, haps, orientation, True)
    bwd = BoundaryMessages(np.zeros(shape), bc, bounds, haps, orientation, False)
    return fwd, bwd, fc, bc, np.zeros((blocks, samples)), np.zeros((blocks, samples))
