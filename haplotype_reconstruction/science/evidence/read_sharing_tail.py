"""Conservative added sample-QC weights in the fitted read-sharing range's first omitted octave."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numba import get_num_threads, prange, types
from scipy.special import ndtri

from ...core.kernels import bounded_team, kernel, sample_team
from .read_sharing import ReadSharing, _band

_ANCHOR_TARGET = 4096
_BLOCK_BP = 1_000_000
_ALPHA = 0.05
"""Nominal one-sided level, split over every occupied bin in the first omitted octave."""


def _partner_intervals(positions, anchors, edges, selected):
    """Two directional slabs per anchor and bin, excluding pairs at distance zero."""
    at = positions[anchors]
    intervals = np.empty((len(selected), 2 * len(at), 2), dtype=np.int64)
    for i, b in enumerate(selected):
        low, high = max(1, int(edges[b])), int(edges[b + 1])
        intervals[i, :len(at), 0] = positions.searchsorted(at - high, side="right")
        intervals[i, :len(at), 1] = positions.searchsorted(at - low, side="right")
        intervals[i, len(at):, 0] = positions.searchsorted(at + low)
        intervals[i, len(at):, 1] = positions.searchsorted(at + high)
    return intervals


@kernel(overloads=((types.int64, types.int64, types.float64,
                    types.float64[::1], types.float64[::1]),), cache=True, inline="always")
def _slab_moments(start, stop, a, prefix, prefix2):
    n = stop - start
    p = prefix[stop] - prefix[start]
    q = prefix2[stop] - prefix2[start]
    return n * a * a + q - 2 * a * p, n * a + p


_I1 = types.Array(types.int64, 1, "C")
_I2 = types.Array(types.int64, 2, "C")
_I3 = types.Array(types.int64, 3, "C")
_F1 = types.Array(types.float64, 1, "C")
_F3 = types.Array(types.float64, 3, "C")
_F4 = types.Array(types.float64, 4, "C")


@kernel(overloads=((_I2, _I1, _I3, _I1, _I1, types.int64, _F3, _F4, _F1),),
        cache=True, parallel=True, nogil=True)
def _paired_moments(depth, anchors, intervals, blocks, cuts, pair_count, rows, omitted, bounds):
    """Full directed moments by sample and deleted moments by lane and physical block.

    A block deletion removes each pair with either endpoint in the block once. For a slab of partners, the
    two partial blocks use prefixes, while complete intervening blocks use range additions of (1, a, a²).
    The anchor's block cannot be an intervening block in a strictly one-sided slab.
    """
    lanes = omitted.shape[0]
    n_rows, n_sites = depth.shape
    n_blocks = len(cuts) - 1
    n_anchors = len(anchors)
    for lane in prange(lanes):
        first = lane * n_rows // lanes
        last = (lane + 1) * n_rows // lanes
        prefix = np.zeros(n_sites + 1)
        prefix2 = np.zeros(n_sites + 1)
        change = np.empty((n_blocks + 1, 3))
        for r in range(first, last):
            highest = 1.0
            for j in range(n_sites):
                d = float(depth[r, j])
                prefix[j + 1] = prefix[j] + d
                prefix2[j + 1] = prefix2[j] + d * d
                highest = max(highest, d)
            # Bound integer magnitudes in float64, including the expansion of (a - d)**2 before cancellation.
            bounds[r] = max(n_sites, 4.0 * pair_count) * highest * highest
            for b in range(intervals.shape[0]):
                change.fill(0.0)
                for q in range(2 * n_anchors):
                    start, stop = intervals[b, q]
                    if stop <= start:
                        continue
                    anchor = anchors[q % n_anchors]
                    a = float(depth[r, anchor])
                    own, left, right = blocks[anchor], blocks[start], blocks[stop - 1]
                    square, exposure = _slab_moments(start, stop, a, prefix, prefix2)
                    rows[r, b, 0] += square
                    rows[r, b, 1] += exposure
                    omitted[lane, b, own, 0] += square
                    omitted[lane, b, own, 1] += exposure
                    if left != own:
                        square, exposure = _slab_moments(start, min(stop, cuts[left + 1]), a, prefix, prefix2)
                        omitted[lane, b, left, 0] += square
                        omitted[lane, b, left, 1] += exposure
                    if right != left and right != own:
                        square, exposure = _slab_moments(cuts[right], stop, a, prefix, prefix2)
                        omitted[lane, b, right, 0] += square
                        omitted[lane, b, right, 1] += exposure
                    if left + 1 < right:
                        change[left + 1, 0] += 1
                        change[right, 0] -= 1
                        change[left + 1, 1] += a
                        change[right, 1] -= a
                        change[left + 1, 2] += a * a
                        change[right, 2] -= a * a
                number, total, squared = 0.0, 0.0, 0.0
                for j in range(n_blocks):
                    number += change[j, 0]
                    total += change[j, 1]
                    squared += change[j, 2]
                    if number > 0:
                        start, stop = cuts[j], cuts[j + 1]
                        sites = stop - start
                        p = prefix[stop] - prefix[start]
                        q2 = prefix2[stop] - prefix2[start]
                        omitted[lane, b, j, 0] += sites * squared + number * q2 - 2 * total * p
                        omitted[lane, b, j, 1] += sites * total + number * p


def _statistics(depth, anchors, intervals, blocks, cuts):
    n_rows, n_sites = depth.shape
    n_bins = len(intervals)
    pair_count = int((intervals[:, :, 1] - intervals[:, :, 0]).sum())
    lanes = sample_team(n_rows, get_num_threads(), n_sites)
    rows = np.zeros((n_rows, n_bins, 2))
    omitted = np.zeros((lanes, n_bins, len(cuts) - 1, 2))
    bounds = np.empty(n_rows)
    with bounded_team(lanes):
        _paired_moments(depth, anchors, intervals, blocks, cuts, pair_count, rows, omitted, bounds)
    if bounds.sum() >= 2**53:
        # Depths outside exact-integer support of the prefix expansion leave the fitted working curve in force.
        return None
    return rows, omitted.sum(axis=0).transpose(1, 0, 2)


def _values(moments):
    square, exposure = moments[..., 0], moments[..., 1]
    return np.divide(exposure - square, exposure, out=np.full_like(exposure, np.nan), where=exposure > 0)


def _lower_weight(total, rows, blocks, z):
    if np.count_nonzero(rows[:, 1]) < 2 or np.count_nonzero(blocks[:, 1]) < 2:
        return 0.0
    deleted_rows, deleted_blocks = _values(total - rows), _values(total - blocks)
    point = _values(total)
    if not (np.isfinite(point) and np.isfinite(deleted_rows).all() and np.isfinite(deleted_blocks).all()):
        return 0.0
    sample_se = np.sqrt(len(rows) - 1) * np.std(deleted_rows)
    block_se = np.sqrt(len(blocks) - 1) * np.std(deleted_blocks)
    return max(0.0, float(point - z * (sample_se + block_se)))


def apply_tail(sharing: ReadSharing, depth: np.ndarray, positions: np.ndarray) -> ReadSharing:
    """Add working weights on the final evidence axis, preserving the fitted range and signed estimates."""
    if sharing.status != "resolved_within" or sharing.range_bp <= 0:
        return sharing
    edges = sharing.edges
    stop = int(edges.searchsorted(sharing.range_bp))
    if stop >= len(edges) - 1:
        return sharing
    selected = np.flatnonzero(_band(stop, edges[:-1], edges[1:], sharing.pairs > 0))
    if not len(selected):
        return sharing
    occupied, starts, blocks = np.unique(positions // _BLOCK_BP, return_index=True, return_inverse=True)
    if len(occupied) < 2:
        return sharing
    cuts = np.r_[starts, len(positions)]
    anchors = np.arange(0, len(positions), max(1, len(positions) // _ANCHOR_TARGET))
    intervals = _partner_intervals(positions, anchors, edges, selected)
    stats = _statistics(depth, anchors, intervals, blocks, cuts)
    if stats is None:
        return sharing
    rows, blocks = stats
    total = rows.sum(axis=0)
    z = ndtri(1.0 - _ALPHA / len(selected))
    applied = np.zeros(len(edges) - 1)
    for i, b in enumerate(selected):
        applied[b] = _lower_weight(total[i], rows[:, i], blocks[:, i], z)
    if not applied.any():
        return sharing
    added_mass = float((sharing.pairs @ applied) * (2.0 / sharing.sites))
    return replace(sharing, applied_tail=applied, neighbour_weight=sharing.neighbour_weight + added_mass)
