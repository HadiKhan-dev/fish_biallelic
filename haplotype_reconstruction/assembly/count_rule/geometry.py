"""Row geometry of the count rule: private cells, nearest rows, shadows and novelty.

A panel is a K x B matrix of rows (one local row per founder and block). A cell (block, row) is
*private* to a founder when no other founder of the panel uses that row at that block. The
shadow X+^sigma of a decided founder f replaces each of f's cells that the smaller panel does not
use by the row the smaller panel uses at that block nearest to it (Hamming distance over sites
called in both rows; ties to the lowest row). A founder without a private cell is a pure mosaic
of the others: its sequence part q is exactly zero.

Distances are measured on the pre-fill inference calls, the calls every likelihood of the join and
of founder refinement reads (-1 at a completion fill and at every not-kept site). A completion fill
is a published call that no likelihood sees, so it never makes two rows differ or agree here: the
nearest rows, shadows and near-identity tests see exactly the alleles the scores they feed see.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class RowGeometry:
    """Called-site distances between the rows of a batch's blocks (computed lazily, cached).

    ``calls``: per block, int8[K_b, L_b] alleles of its rows (-1 = no call).
    nearest(b, r, covered): sigma_b(r), the covered row nearest to r (Hamming over sites called in
    both; ties to the lowest row). span_mismatch(p, g): the called-site mismatch rate of two paths
    over the batch's span.
    """

    def __init__(self, calls: Sequence[np.ndarray]):
        self.calls = [np.asarray(value, dtype=np.int8) for value in calls]
        self._nearest = {}
        self._pairs = {}

    @classmethod
    def from_panels(cls, panels) -> "RowGeometry":
        """Geometry of a sequence of panels over their ``inference`` calls (prepared panels for
        founder refinement, a batch's materialised components for the join)."""
        return cls([panel.inference for panel in panels])

    def nearest(self, block: int, row: int, covered: tuple[int, ...]) -> int:
        key = (block, row, covered)
        found = self._nearest.get(key)
        if found is None:
            calls = self.calls[block]
            target = calls[row]
            candidates = calls[list(covered)]
            both = (target >= 0)[None, :] & (candidates >= 0)
            distance = ((candidates != target[None, :]) & both).sum(axis=1)
            found = covered[int(np.argmin(distance))]   # covered is sorted: ties -> lowest row
            self._nearest[key] = found
        return found

    def _pair(self, block, first, second):
        key = (block, first, second) if first <= second else (block, second, first)
        found = self._pairs.get(key)
        if found is None:
            a, c = self.calls[block][first], self.calls[block][second]
            both = (a >= 0) & (c >= 0)
            found = (int(((a != c) & both).sum()), int(both.sum()))
            self._pairs[key] = found
        return found

    def span_mismatch(self, path, other) -> float:
        """Called-site mismatch rate of two paths over the span (0 when no site is called in both)."""
        mismatch = called = 0
        for block, (first, second) in enumerate(zip(path, other)):
            m, n = self._pair(block, int(first), int(second))
            mismatch += m
            called += n
        return mismatch / called if called else 0.0

    def near_identical(self, path, others, threshold: float) -> bool:
        """True iff some other path is within ``threshold`` mismatch rate over the span."""
        return any(self.span_mismatch(path, other) <= threshold for other in others)


def covered_rows(rows) -> list[tuple[int, ...]]:
    """Per block, the sorted tuple of rows used by the panel (rows: int[K, B])."""
    rows = np.asarray(rows)
    return [tuple(sorted(set(int(r) for r in rows[:, b]))) for b in range(rows.shape[1])]


def private_mask(rows) -> np.ndarray:
    """bool[K, B]: private[i, b] iff path i is the only path using its row at block b."""
    rows = np.asarray(rows, dtype=np.int64)
    private = np.zeros(rows.shape, dtype=bool)
    for b in range(rows.shape[1]):
        _, inverse, counts = np.unique(rows[:, b], return_inverse=True, return_counts=True)
        private[:, b] = counts[inverse] == 1
    return private


def shadow_path(path, smaller_cover, geometry: RowGeometry):
    """f^sigma: f's cells not used by the smaller panel replaced by its nearest rows there.

    Returns (shadow path as a tuple, number of replaced cells).
    """
    result = list(path)
    lost = 0
    for b, row in enumerate(path):
        if row not in smaller_cover[b]:
            result[b] = geometry.nearest(b, int(row), smaller_cover[b])
            lost += 1
    return tuple(result), lost


def novel_paths(panel, geometry: RowGeometry, near_identity: float) -> np.ndarray:
    """novel[i]: path i owns sequence the other paths lack: a private cell, and no other path within
    ``near_identity`` of it over the span. A path without it has q = 0 (a pure mosaic or a
    near-duplicate): never added, deleted first, never the result of a same-count move (the join's
    rule; founder refinement uses private cells only)."""
    rows = np.asarray(panel, dtype=np.int64)
    private = private_mask(rows).any(axis=1)
    novel = private.copy()
    for i in np.flatnonzero(private).tolist():
        others = [path for j, path in enumerate(panel) if j != i]
        if geometry.near_identical(panel[i], others, near_identity):
            novel[i] = False
    return novel


def has_novelty(path, others, cover, geometry: RowGeometry, near_identity: float) -> bool:
    """A candidate path owns sequence a panel (``cover`` = its covered rows) lacks."""
    if not any(row not in cover[b] for b, row in enumerate(path)):
        return False
    return not geometry.near_identical(path, others, near_identity)
