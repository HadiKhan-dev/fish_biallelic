"""Assembly content orders for the refinement pilot, using core tie mechanisms.

Owns canonical founder-path ordering, not search trajectories or storage.
Method: docs/methods/founder_refinement.md#ties (RS09, RS12).
"""

import numpy as np

from haplotype_reconstruction.core import ties
from haplotype_reconstruction.core.kernels import kernel
from haplotype_reconstruction.core.ties import (
    WIDE_CLASS_QUANTA,
    SharedCounter,
    add_decision,
    census,
    motion,
    quantum,
    truncate,
)

from .refinement.overloads import overloads_for


def canonical_order(rows) -> np.ndarray:
    """int64[K] permutation putting the rows of an int matrix (K x B) in lexicographic order (stable)."""
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[0] < 2:
        return np.arange(rows.shape[0] if rows.ndim else 0, dtype=np.int64)
    return np.lexsort(tuple(rows[:, column] for column in range(rows.shape[1] - 1, -1, -1))).astype(np.int64)


def canonical_rows(rows) -> tuple[np.ndarray, np.ndarray]:
    """(rows in canonical order, the permutation used: canonical[i] = rows[order[i]])."""
    rows = np.asarray(rows)
    order = canonical_order(rows)
    return np.ascontiguousarray(rows[order]), order


def path_key(path) -> tuple:
    """Exact content key of one founder path (its row indices)."""
    return tuple(int(row) for row in path)


def panel_key(panel) -> tuple:
    """Exact content key of a panel: its paths as sorted tuples (multiplicity kept)."""
    return tuple(sorted(path_key(path) for path in panel))


__all__ = [
    "canonical_order",
    "canonical_rows",
    "path_key",
    "panel_key",
    "quantum",
    "motion",
    "add_decision",
    "census",
    "truncate",
    "SharedCounter",
]


@kernel(overloads=overloads_for(__name__, "_virtual_cmp"), cache=True, nogil=True)
def _virtual_cmp(X, p, f1, f2, g1, g2):
    """Lexicographic comparison (-1, 0, 1) of two virtual founder paths: path(u, v) = X[u, :p], X[v, p:]."""
    for x in range(X.shape[1]):
        a = X[f1, x] if x < p else X[f2, x]
        b = X[g1, x] if x < p else X[g2, x]
        if a != b:
            return -1 if a < b else 1
    return 0


@kernel(overloads=overloads_for(__name__, "_exchange_cmp"), cache=True, nogil=True)
def _exchange_cmp(X, p, a1, b1, a2, b2, reverse):
    """Content order of two suffix exchanges at junction p: their resulting pairs of founder paths (each pair
    sorted, smaller path first) compared lexicographically; inverted under the convention reversal."""
    if _virtual_cmp(X, p, a1, b1, b1, a1) <= 0:
        l1a, l1b, h1a, h1b = a1, b1, b1, a1
    else:
        l1a, l1b, h1a, h1b = b1, a1, a1, b1
    if _virtual_cmp(X, p, a2, b2, b2, a2) <= 0:
        l2a, l2b, h2a, h2b = a2, b2, b2, a2
    else:
        l2a, l2b, h2a, h2b = b2, a2, a2, b2
    c = _virtual_cmp(X, p, l1a, l1b, l2a, l2b)
    if c == 0:
        c = _virtual_cmp(X, p, h1a, h1b, h2a, h2b)
    return -c if reverse else c


@kernel(overloads=overloads_for(__name__, "_exchange_winner"), cache=True, nogil=True)
def _exchange_winner(gains, pa, pb, X, p, q, floor, reverse):
    """The chosen suffix exchange among those with gain > floor: the best tie class of the exact gains (quantum
    q), inside it the content-smallest exchange (_exchange_cmp); a class wider than WIDE_CLASS_QUANTA x q keeps value
    order (content among exactly equal values). Returns (index or -1, class size, wide)."""
    # L3: equal values and equal resulting paths keep pair enumeration order.
    # This final stable order is independent of NumPy/Numba quicksort internals.
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
    order = eligible[np.argsort(values, kind="mergesort")]
    best = gains[order[0]]
    size = 1
    previous = best
    while size < m:
        value = gains[order[size]]
        if not (previous - value <= q):
            break
        previous = value
        size += 1
    wide = (best - previous) > WIDE_CLASS_QUANTA * q
    winner = order[0]
    for t in range(1, size):
        i = order[t]
        if wide and gains[i] != gains[winner]:
            if gains[i] > gains[winner]:
                winner = i
            continue
        if _exchange_cmp(X, p, pa[i], pb[i], pa[winner], pb[winner], reverse) < 0:
            winner = i
    return winner, size, wide


@kernel(overloads=overloads_for(__name__, "_move_cmp"), cache=True, nogil=True)
def _move_cmp(X, cand_rows, k1, a1, c1, r1, i1, k2, a2, c2, r2, i2, reverse):
    """Content order of two content moves: the moved founder's resulting path (rows cand_rows[r:] on blocks
    [a, c), its current rows elsewhere), then the founder slot, then proposal order."""
    c = 0
    for x in range(X.shape[1]):
        u = cand_rows[r1 + x - a1] if a1 <= x < c1 else X[k1, x]
        v = cand_rows[r2 + x - a2] if a2 <= x < c2 else X[k2, x]
        if u != v:
            c = -1 if u < v else 1
            break
    if c == 0 and k1 != k2:
        c = -1 if k1 < k2 else 1
    if c == 0 and i1 != i2:
        c = -1 if i1 < i2 else 1
    return -c if reverse else c


def sort_panel(panel) -> list:
    """A panel (sequence of path tuples) in canonical order."""
    return sorted(path_key(path) for path in panel)


def path_codes(paths) -> np.ndarray | None:
    """int64[n] mixed-radix codes of the rows of an int matrix whose numeric order is the rows' lexicographic
    order (each column shifted by its minimum, radix its range; the first column most significant), or None
    when the product of the ranges does not fit below 2^62 (callers then sort by the columns)."""
    paths = np.asarray(paths, dtype=np.int64)
    if paths.ndim != 2 or paths.shape[0] == 0:
        return np.zeros(paths.shape[0] if paths.ndim else 0, dtype=np.int64)
    low = paths.min(axis=0)
    ranges = paths.max(axis=0) - low + 1
    total = 1
    for value in ranges.tolist():
        total *= int(value)
    if total >= 1 << 62:
        return None
    codes = np.zeros(paths.shape[0], dtype=np.int64)
    for column in range(paths.shape[1]):
        codes *= ranges[column]
        codes += paths[:, column] - low[column]
    return codes


def path_class_order(
    values,
    paths,
    quantum_value: float,
    *,
    descending: bool = False,
    motion_value: float | None = None,
    reverse: bool = False,
):
    """Vectorised tie classes of ``values`` with the lexicographic ``paths`` (int[n, B]) as content keys, the same
    rules as maths.ties.tie_classes (single-linkage classes over the exactly sorted values, content order inside a
    class, value order inside a class wider than WIDE_CLASS_QUANTA x q).

    Returns (order int64[n] in decision order, classes int64[n] per element (0 = best), wide classes, boundary gaps)."""
    values = np.asarray(values, dtype=np.float64)
    paths = np.asarray(paths, dtype=np.int64).reshape(len(values), -1)
    n = len(values)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), 0, ()
    if np.isnan(values).any():
        raise ValueError("tie classes of NaN are undefined")
    sign = -1.0 if descending else 1.0
    signed = sign * values
    # One mixed-radix code per path orders the paths exactly as their columns do (two sort keys
    # instead of B + 1; the same stable sorts, so the same permutations).
    codes = path_codes(paths)
    columns = (codes,) if codes is not None else tuple(paths[:, column] for column in range(paths.shape[1] - 1, -1, -1))
    exact = np.lexsort(columns + (signed,))
    ordered = signed[exact]
    with np.errstate(invalid="ignore"):
        gaps = np.diff(ordered)
    new = np.ones(n, dtype=bool)
    # A new class where the gap exceeds q (equal values, inf - inf included, never split).
    new[1:] = (ordered[1:] != ordered[:-1]) & ~(gaps <= quantum_value)
    class_sorted = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], n] - 1
    with np.errstate(invalid="ignore"):
        spans = ordered[ends] - ordered[starts]
    spans = np.where(np.isfinite(spans), spans, 0.0)
    wide = spans > ties.WIDE_CLASS_QUANTA * quantum_value
    value_key = np.where(wide[class_sorted], ordered, 0.0)
    # Content order inside a class: lexicographic paths, reversed under reverse (release 2.2
    # convention reversal; classes and values are not reversed).
    direction = -1 if reverse else 1
    if codes is not None:
        content = (direction * codes[exact],)
    else:
        sorted_paths = paths[exact]
        content = tuple(direction * sorted_paths[:, column] for column in range(paths.shape[1] - 1, -1, -1))
    inner = np.lexsort(content + (value_key, class_sorted))
    order = exact[inner]
    classes = np.empty(n, dtype=np.int64)
    classes[exact] = class_sorted
    boundary: tuple[float, ...] = ()
    if motion_value is not None and n > 1:
        guard = ties.BOUNDARY_GUARD * float(motion_value)
        finite = np.isfinite(gaps) & (gaps != 0.0)
        boundary = tuple(float(gap) for gap in gaps[finite & (np.abs(gaps - quantum_value) <= guard)])
    return order.astype(np.int64), classes, int(np.count_nonzero(wide)), boundary


def linkage_order(result: ties.TieClasses, links, contents, kind: str = "beam", *, reverse: bool = False) -> list:
    """Decision order of a tie-class decision with a data criterion inside the classes (release 2.2): inside every
    class of ``result`` (not wider than WIDE_CLASS_QUANTA x q; those keep value order) the members are ordered by
    tie classes of ``links`` (descending; quantum ``kind`` at the largest |link| of the class) and only then by the
    content key (reversed under the convention reversal). links: the batch linker's log-probability of each
    candidate's paths (join/beam.path_linkage), the population linkage the sample likelihood could not use to
    separate them. Returns element indices in decision order; symmetric_winner(result, links, order) tells whether
    the winner was still decided by content (a fully symmetric tie, counted and flagged by the callers)."""
    order = [int(i) for i in result.order]
    if links is None or len(order) < 2:
        return order
    classes = result.classes
    out = []
    start = 0
    while start < len(order):
        c = int(classes[order[start]])
        end = start
        while end < len(order) and int(classes[order[end]]) == c:
            end += 1
        members = order[start:end]
        wide = float(result.spans[c]) > ties.WIDE_CLASS_QUANTA * float(result.quantum)
        if len(members) > 1 and not wide:
            values = [float(links[m]) for m in members]
            finite = [abs(v) for v in values if np.isfinite(v)]
            sub = ties.tie_classes(
                values,
                quantum(kind, max(finite) if finite else 0.0),
                content=[contents[m] for m in members],
                descending=True,
                reverse=reverse,
            )
            members = [members[k] for k in sub.order]
        out.extend(members)
        start = end
    return out


def symmetric_members(result: ties.TieClasses, links, kind: str = "beam", *, reverse: bool = False) -> list:
    """The elements a tie-class decision's winner is still tied with once the data criteria are used: the members
    of its class (not wider than WIDE_CLASS_QUANTA x q) that are also tied with it on the linker score (all of them
    without links). Empty when the data decide. A non-empty list is a fully symmetric decision, decided by the
    content key (release 2.2: counted as kind "<decision>.symmetric" and its alternatives released in the events)."""
    if not result.tied_winner:
        return []
    c = int(result.classes[result.winner])
    if float(result.spans[c]) > ties.WIDE_CLASS_QUANTA * float(result.quantum):
        return []
    members = [int(i) for i in result.members(c)]
    if links is None:
        return members
    values = [float(links[m]) for m in members]
    finite = [abs(v) for v in values if np.isfinite(v)]
    sub = ties.tie_classes(values, quantum(kind, max(finite) if finite else 0.0), descending=True, reverse=reverse)
    if not sub.tied_winner:
        return []
    return [members[k] for k in sub.members(int(sub.classes[sub.winner]))]


__all__ += ["sort_panel", "path_codes", "path_class_order", "linkage_order", "symmetric_members"]
