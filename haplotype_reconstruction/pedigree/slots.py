"""Released order of a child's two observed parents (release 2.2; Parent1/Parent2, family-phase slots).

Scoring treats a parent pair as unordered; only the released products order it. The order carries biology only
when the configured sample metadata records the parents' sexes:

* ``dam_sire``: the two parents have recorded sexes exactly {F, M}: Parent1 / slot 0 = the dam (F), Parent2 /
  slot 1 = the sire (M). Family phase gives the child's homologue 0 to slot 0, so the phased GT of such a child
  reads maternal | paternal.
* ``unordered``: otherwise (no metadata, unknown or equal sexes) the slots carry no biological meaning. Their order
  is a canonical label that does not depend on sample names or input order: the parent with more observed sites
  (read depth > 0) first, compared over the run's contigs in configured order until they differ; the sample name
  decides only between parents whose observed patterns have identical counts on every contig. Renaming samples
  changes no released order; the convention reversal (maths.ties) flips the label.
* ``single``: one observed parent: slot 1 when its recorded sex is M (the sire), else slot 0.

Sexes are never inferred: a blank or other label is unknown.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..maths import ties

DAM, SIRE = "F", "M"
BASES = ("dam_sire", "unordered", "single", "none")


@dataclass(frozen=True)
class ParentOrientation:
    """Orders released parent pairs: ``sexes`` (recorded labels per sample, or None) and ``ranks`` int64 [N]
    (canonical rank of each sample, lower first; :func:`canonical_ranks`)."""

    sexes: tuple[str, ...] | None
    ranks: np.ndarray

    def _sex(self, sample: int) -> str:
        return "" if self.sexes is None else self.sexes[sample]

    def order(self, first: int, second: int) -> tuple[tuple[int, int], str]:
        """((slot 0, slot 1), basis) of a released parent pair (-1 = no parent)."""
        named = [int(parent) for parent in (first, second) if parent >= 0]
        if not named:
            return (-1, -1), "none"
        if len(named) == 1:
            parent = named[0]
            return ((-1, parent) if self._sex(parent) == SIRE else (parent, -1)), "single"
        a, b = named
        sexes = (self._sex(a), self._sex(b))
        if set(sexes) == {DAM, SIRE}:
            return ((a, b) if sexes[0] == DAM else (b, a)), "dam_sire"
        before = (int(self.ranks[a]) <= int(self.ranks[b])) != ties.reverse_conventions()
        return ((a, b) if before else (b, a)), "unordered"

    def orient(self, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """(oriented int64 [N, 2], moved bool [N]: the pair's slots changed, basis per row) of parents [N, 2]."""
        parents = np.asarray(parents, dtype=np.int64)
        oriented = np.full(parents.shape, -1, dtype=np.int64)
        bases = []
        for row, (first, second) in enumerate(parents.tolist()):
            pair, basis = self.order(first, second)
            oriented[row] = pair
            bases.append(basis)
        moved = np.any(oriented != parents, axis=1)
        return oriented, moved, tuple(bases)


def canonical_ranks(observed_counts, sample_ids, pairs=None) -> np.ndarray:
    """int64 [N]: canonical sample ranks by (-observed sites per contig, compared contig by contig, then name).

    ``observed_counts`` is an iterable yielding int64 [N] observed-site counts per contig in configured order (it
    may be a generator: consumption stops once every pair of ``pairs``, an iterable of (a, b), differs on some
    contig; None consumes all). The name is a last resort for identical count vectors.
    """
    names = [str(name) for name in sample_ids]
    n = len(names)
    pending = None if pairs is None else {(int(a), int(b)) for a, b in pairs if a >= 0 and b >= 0 and a != b}
    columns = []
    for counts in observed_counts:
        counts = np.asarray(counts, dtype=np.int64)
        if counts.shape != (n,):
            raise ValueError("observed-site counts must have one entry per sample")
        columns.append(counts)
        if pending is not None:
            pending = {(a, b) for a, b in pending if counts[a] == counts[b]}
            if not pending:
                break
    keys = [tuple(-int(column[sample]) for column in columns) + (names[sample],) for sample in range(n)]
    order = sorted(range(n), key=lambda sample: keys[sample])
    ranks = np.empty(n, dtype=np.int64)
    ranks[np.asarray(order, dtype=np.int64)] = np.arange(n, dtype=np.int64)
    return ranks


def orient_edges(parents: np.ndarray, children: np.ndarray, slots: np.ndarray,
                 orientation: ParentOrientation) -> np.ndarray:
    """Slots of released parent edges (parent, child, internal slot) in the released orientation: a child's edges
    are re-slotted together by :meth:`ParentOrientation.order` of its supported parents."""
    parents, children, slots = (np.asarray(values, dtype=np.int64) for values in (parents, children, slots))
    result = slots.copy()
    for child in np.unique(children):
        edges = np.flatnonzero(children == child)
        pair = [-1, -1]
        for edge in edges:
            pair[int(slots[edge])] = int(parents[edge])
        oriented, _ = orientation.order(*pair)
        for edge in edges:
            result[edge] = oriented.index(int(parents[edge]))
    return result


def orient_frame(frame, sample_ids, orientation: ParentOrientation, *, first: str = "Parent1",
                 second: str = "Parent2"):
    """A report table with its Parent1*/Parent2* column pairs swapped where the released order of the row's
    (``first``, ``second``) names moves them, and a ParentSlotBasis column (dam_sire / unordered / single /
    none). Rows whose parent columns are empty keep their values."""
    import pandas as pd
    if frame is None or first not in frame.columns or second not in frame.columns or not len(frame):
        return frame
    index = {str(name): position for position, name in enumerate(sample_ids)}
    frame = frame.copy()
    pairs = [(column, column.replace(first, second, 1)) for column in frame.columns
             if first in column and column.replace(first, second, 1) in frame.columns]
    bases, moved = [], []
    for a, b in zip(frame[first].tolist(), frame[second].tolist()):
        p = index.get(str(a), -1) if a is not None and not (isinstance(a, float) and a != a) else -1
        q = index.get(str(b), -1) if b is not None and not (isinstance(b, float) and b != b) else -1
        oriented, basis = orientation.order(p, q)
        bases.append(basis)
        moved.append(oriented != (p, q))
    moved = np.asarray(moved, dtype=bool)
    if np.any(moved):
        for a, b in pairs:
            left, right = frame[a].tolist(), frame[b].tolist()
            for row in np.flatnonzero(moved).tolist():
                left[row], right[row] = right[row], left[row]
            same = frame[a].dtype == frame[b].dtype and frame[a].dtype != object
            dtype = frame[a].dtype if same else object
            frame[a] = pd.Series(left, index=frame.index, dtype=dtype)
            frame[b] = pd.Series(right, index=frame.index, dtype=dtype)
    position = list(frame.columns).index(second) + 1
    frame.insert(position, "ParentSlotBasis", pd.Series(bases, index=frame.index, dtype=object))
    return frame
