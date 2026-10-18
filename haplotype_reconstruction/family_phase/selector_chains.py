"""The family solver's incremental four-state selector chains (its hot loop).

``update_selector_chains`` keeps every child's joint selector chain (maths.chains: two independent selector bits) in
256-marker blocks and recomputes only blocks whose emissions changed, propagating forward/backward until the
block-boundary messages are exactly unchanged, so every stored value equals a full recomputation
(maths.chains.selector_cavity). The recurrences are those of selector_cavity, hand-unrolled over the four states.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..maths.chains import mix4, selector_cavity


BLOCK_MARKERS = 256
# Selector caches hold three float64 four-state streams plus float32 emissions per supported marker
# (113 bytes with block metadata); groups beyond this budget are recomputed in full each iteration.
MAX_SELECTOR_CACHE_BYTES = 24 * 1024 ** 3
SELECTOR_CACHE_BYTES_PER_MARKER = 113
# Factor sweeps mark changed emissions in tiles of this many sites (factors.FACTOR_TILE_SITES).
DIRTY_TILE_SITES = 32


@njit(cache=True, parallel=True)
def _tile_blocks(offsets, indices, tiles):
    """First and stop cache block of every group's supported markers inside each dirty tile."""
    first = np.full((len(offsets) - 1, tiles), -1, dtype=np.int32)
    stop = np.zeros_like(first)
    for group in prange(len(offsets) - 1):
        begin = offsets[group]
        end = offsets[group + 1]
        cursor = begin
        for tile in range(tiles):
            left = cursor
            while cursor < end and indices[cursor] < (tile + 1) * DIRTY_TILE_SITES:
                cursor += 1
            if cursor > left:
                first[group, tile] = (left - begin) // BLOCK_MARKERS
                stop[group, tile] = (cursor - 1 - begin) // BLOCK_MARKERS + 1
    return first, stop


class SelectorChainCache:
    """Incremental workspaces of the selector chains of the first groups that fit ``max_bytes``.

    ``support`` = (offsets int64[G+1], marker indices int32[M], transition probabilities float64[M]) of the
    supported markers of every group (factors.selector_support). Workspaces are transient (never saved).
    """

    def __init__(self, support, max_bytes=MAX_SELECTOR_CACHE_BYTES):
        offsets = support[0]
        self.groups = max(0, int(np.searchsorted(offsets, max_bytes // SELECTOR_CACHE_BYTES_PER_MARKER,
                                                 side='right') - 1))
        count = int(offsets[self.groups])
        lengths = np.diff(offsets[:self.groups + 1])
        blocks = np.r_[0, np.cumsum((lengths + BLOCK_MARKERS - 1) // BLOCK_MARKERS)].astype(np.int64)
        nblocks = int(blocks[-1])
        # saved emissions, alpha, predicted, beta, block offsets, block boundary messages, valid groups,
        # settled blocks, block residuals.
        self.arrays = (np.empty((count, 4), dtype=np.float32),
                       np.empty((count, 4)), np.empty((count, 4)), np.empty((count, 4)), blocks,
                       np.empty((nblocks, 4)), np.zeros(self.groups, dtype=np.bool_),
                       np.zeros(nblocks, dtype=np.bool_), np.zeros(nblocks))
        self.tile_mapping = None
        self.damping = None

    def invalidate(self):
        """Forget every cached chain (after a gauge move permutes selector states)."""
        self.arrays[6].fill(False)
        self.arrays[7].fill(False)


@njit(cache=True, parallel=True)
def _incremental_update(emissions, current, damping, offsets, indices, theta,
                        saved, alpha, predicted, beta, block_offsets, boundary, valid, settled,
                        block_delta, producer_dirty, tile_mapping):
    delta = np.zeros(len(emissions))
    for group in prange(len(emissions)):
        start = offsets[group]
        stop = offsets[group + 1]
        length = stop - start
        if not length:
            continue
        if group >= len(valid):
            local = np.empty((length, 4), dtype=np.float32)
            for i in range(start, stop):
                for state in range(4):
                    local[i - start, state] = emissions[group, indices[i], state]
            cavity = selector_cavity(local, theta[start:stop - 1])
            for i in range(start, stop):
                for state in range(4):
                    old = current[group, indices[i], state]
                    new = (1 - damping) * old + damping * cavity[i - start, state]
                    current[group, indices[i], state] = new
                    delta[group] = max(delta[group], abs(new - old))
            continue
        was_valid = valid[group]
        first_block = block_offsets[group]
        blocks = block_offsets[group + 1] - first_block
        dirty = np.zeros(blocks, dtype=np.bool_)
        forward = np.zeros(blocks, dtype=np.bool_)
        backward = np.zeros(blocks, dtype=np.bool_)
        inspect = np.ones(blocks, dtype=np.bool_)
        if was_valid:
            inspect[:] = False
            first, stop_block = tile_mapping
            for tile in range(producer_dirty.shape[1]):
                if producer_dirty[group, tile] and first[group, tile] >= 0:
                    for block in range(first[group, tile], stop_block[group, tile]):
                        inspect[block] = True
        for block in range(blocks):
            if not inspect[block]:
                continue
            lo = start + block * BLOCK_MARKERS
            hi = min(stop, lo + BLOCK_MARKERS)
            for i in range(lo, hi):
                different = not was_valid
                for state in range(4):
                    value = emissions[group, indices[i], state]
                    if was_valid and value != saved[i, state]:
                        different = True
                    saved[i, state] = value
                if different:
                    dirty[block] = True
        carry = not was_valid
        for block in range(blocks):
            lo = start + block * BLOCK_MARKERS
            hi = min(stop, lo + BLOCK_MARKERS)
            if not (dirty[block] or carry):
                continue
            old0 = alpha[hi - 1, 0] if was_valid else 0.
            old1 = alpha[hi - 1, 1] if was_valid else 0.
            old2 = alpha[hi - 1, 2] if was_valid else 0.
            old3 = alpha[hi - 1, 3] if was_valid else 0.
            for i in range(lo, hi):
                p = ((.25, .25, .25, .25) if i == start else
                     mix4(alpha[i - 1, 0], alpha[i - 1, 1], alpha[i - 1, 2], alpha[i - 1, 3], theta[i - 1]))
                total = 0.
                for state in range(4):
                    predicted[i, state] = p[state]
                    alpha[i, state] = p[state] * saved[i, state]
                    total += alpha[i, state]
                for state in range(4):
                    alpha[i, state] /= total
            carry = (not was_valid or alpha[hi - 1, 0] != old0 or alpha[hi - 1, 1] != old1 or
                     alpha[hi - 1, 2] != old2 or alpha[hi - 1, 3] != old3)
            forward[block] = True
        carry = not was_valid
        for block in range(blocks - 1, -1, -1):
            lo = start + block * BLOCK_MARKERS
            hi = min(stop, lo + BLOCK_MARKERS)
            key = first_block + block
            if not (dirty[block] or carry):
                continue
            old0 = boundary[key, 0] if was_valid else 0.
            old1 = boundary[key, 1] if was_valid else 0.
            old2 = boundary[key, 2] if was_valid else 0.
            old3 = boundary[key, 3] if was_valid else 0.
            for i in range(hi - 1, lo - 1, -1):
                if i == stop - 1:
                    for state in range(4):
                        beta[i, state] = 1.
                else:
                    values = mix4(saved[i + 1, 0] * beta[i + 1, 0], saved[i + 1, 1] * beta[i + 1, 1],
                                  saved[i + 1, 2] * beta[i + 1, 2], saved[i + 1, 3] * beta[i + 1, 3], theta[i])
                    total = sum(values)
                    for state in range(4):
                        beta[i, state] = values[state] / total
            for state in range(4):
                boundary[key, state] = saved[lo, state] * beta[lo, state]
            # The emission at a block's first marker affects the preceding block even when this block's
            # beta vector itself is unchanged.
            carry = (not was_valid or boundary[key, 0] != old0 or boundary[key, 1] != old1 or
                     boundary[key, 2] != old2 or boundary[key, 3] != old3)
            backward[block] = True
        for block in range(blocks):
            key = first_block + block
            lo = start + block * BLOCK_MARKERS
            hi = min(stop, lo + BLOCK_MARKERS)
            if forward[block] or backward[block] or not settled[key]:
                same = True
                change = 0.
                for i in range(lo, hi):
                    total = 0.
                    for state in range(4):
                        total += predicted[i, state] * beta[i, state]
                    for state in range(4):
                        target = predicted[i, state] * beta[i, state] / total
                        old = current[group, indices[i], state]
                        new = (1 - damping) * old + damping * target
                        current[group, indices[i], state] = new
                        if np.float32(new) != old:
                            same = False
                        change = max(change, abs(new - old))
                settled[key] = same
                block_delta[key] = change
            # Retain the actual residual of an update that rounded to the same stored messages.
            delta[group] = max(delta[group], block_delta[key])
        valid[group] = True
    return delta


def update_selector_chains(emissions, current, damping, support, cache, dirty_tiles):
    """Damped cavity update of every group's selector chain in ``current`` (float32 [G, L, 4], in place).

    ``emissions`` float32 [G, L, 4] are the copy-factor selector emissions, ``dirty_tiles`` bool
    [G, ceil(L / 32)] marks tiles whose emissions the last factor sweep changed. Returns the largest message
    change per group.
    """
    if cache.damping != damping:
        cache.arrays[7].fill(False)
        cache.damping = damping
    if cache.tile_mapping is None:
        cache.tile_mapping = _tile_blocks(support[0], support[1], dirty_tiles.shape[1])
    return _incremental_update(emissions, current, damping, *support, *cache.arrays,
                               dirty_tiles, cache.tile_mapping)
