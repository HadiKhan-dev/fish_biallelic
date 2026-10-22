"""Tie quanta, canonical founder order and tie-class orders of the assembly stage (release 2.1, design 5.1-5.4).

Assembly decides on float sums over its float32 evidence (the joins, the endpoint beam and founder refinement) and on
the fitted linker transitions. A tiny input perturbation moves such a sum without changing its meaning, so a discrete
choice must never follow from which of two values that differ by less than that motion happens to be larger, nor from
array order. The rules (``maths.ties``, owner-reviewed 18 Oct):

* **Canonical founder order** (5.1): the founders (row paths) of every joined component, every refinement fit and the
  join search's own panel are kept in lexicographic order of their prepared-row index paths (prepared rows are in
  lexicographic order of their fitted alleles, so this is a content order). Every downstream tie-break that falls back
  to a founder slot or a pool index is then a content rule. Likelihoods do not change; founder labels do.
* **Tie classes** (``maths.ties.tie_classes``): values closer than the decision kind's quantum chain into one class;
  a winner is the content-smallest member of the best class, a truncation (top-q proposals, beam quotas, refit
  shortlists) keeps every member of the class it cuts (capped at twice the budget) and orders inside a class by
  content; a class spanning more than ``maths.ties.WIDE_CLASS_QUANTA`` quanta is reported (width guard) and ordered
  by value inside. Content keys are exact discrete data: prepared-row index paths and panels as sorted path tuples.
* **Quanta** (:data:`ASSEMBLY_QUANTA`): one per decision kind, a power of two at least max(floor, factor x |scale|),
  placed in the kind's valley between the motion of the compared differences under 1e-9 relative GL noise (including
  its one-ulp float32 consequences) and the first genuine differences, by the valley census of design A-8
  (``maths.ties.census``; work/N2/harness/census.py). :data:`ASSEMBLY_MOTION` records the measured motions (boundary
  guard).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..maths import ties

# (floor, relative factor of the scale) per assembly decision kind: q = power of two >= max(floor, factor x |scale|).
#   refine_F      founder refinement totals F (log units): the refit picked at a count step, forward vs reverse
#                 search, best-F keep/restore, addition copy gains; scale = the incumbent's F. Also the per-sample
#                 band of the zero-gain cleanup (floor only).
#   join_ll       join log likelihoods and exact gains of one-move alternatives (nats): tier-1 and tier-2 winners,
#                 birth and seed choices; scale = the current panel's log likelihood.
#   join_ll_wide  join log likelihoods of panels that differ in whole paths (nats): the start panel and the death
#                 candidate (V(P minus f) over f; their sparse per-sample scores move more); scale = the panel's LL.
#   join_proxy    the join's proxy scores (conditional-field, static and novelty gains of tier-1 proposals, bound gains
#                 of chimera-trap indicators): proposal cuts and novelty ranking; scale = the cut's largest |score|.
#                 The floor alone is the band of the novelty test at its exact zero.
#   beam          endpoint-beam path scores (sums of mean log transitions); scale = the largest |score| of the cut.
#   count_margin  effective-carrier point margins in units of the founder cost (refinement shortlists); no scale.
#   paint         the join painting's Viterbi traceback (proposal generation only): relative to the bin's best
#                 per-sample score (factor only; floor 2^-30 nats).
# Valley census (design A-8; fixed l2 panels of actm3 chr8/13/19, std_n80 chr8/13/18 and TroMau chr18/chr3, test tree
# ic2, reference vs iid 1e-9; work/IC/census): largest motion of the compared differences (relative to the scale) /
# smallest genuine (non-rounding, non-zero) difference relative to the scale:
#   join_ll (tier 1): 2.7e-12 / 4.4e-12 (actm3; 1.1e-6 nats, merged), 5.9e-11 (merged), then 4.9e-10 (TroMau);
#     tier 2, births: <= 1.3e-7 nats / 4.7e-4 nats
#   join_ll_wide (deaths, start): 5.7e-11 (std_n80) / 2.1e-8 (actm3 deaths) and 1.8e-7 (start)
#   refine_F: 3.6e-12 (std_n80) / 6.5e-13 and 8.3e-12 (actm3 fwd vs rev, merged), 4.9e-11 (TroMau), refits 8.4e-9
#   join_proxy: <= max(1.1e-7, 2.1e-9 |score|) / continuum (no wide valley: classes cut by content, the cut class kept
#     whole); 2.0 had O(1) jumps where a painting near-tie flipped (fixed by the painting band, kind paint)
#   beam: 6.5e-8 nats / 1.2e-8 .. 3e-7 (a few dozen of ~1.3e6 differences below q, merged), dense from 1e-4
#   count_margin: 1.3e-10 / 0.009 (units of c); zero-gain per-sample losses exactly 0 or >= 1.87 nats (motion 1.2e-5).
ASSEMBLY_QUANTA: dict[str, tuple[float, float]] = {
    "refine_F": (2.0 ** -14, 2.0 ** -35),
    "join_ll": (2.0 ** -19, 2.0 ** -34),
    "join_ll_wide": (2.0 ** -19, 2.0 ** -30),
    "join_proxy": (2.0 ** -19, 2.0 ** -24),
    "beam": (2.0 ** -18, 2.0 ** -30),
    "count_margin": (2.0 ** -26, 0.0),
    "paint": (2.0 ** -30, 2.0 ** -30),
}
# Measured motion of the compared differences under 1e-9 relative GL noise (census above), as (absolute, relative to
# the decision's scale): max(absolute, relative x |scale|) enables the boundary guard of maths.ties.tie_classes.
ASSEMBLY_MOTION: dict[str, tuple[float, float]] = {
    "refine_F": (5.1e-7, 3.6e-12),
    "join_ll": (1.3e-7, 2.7e-12),
    "join_ll_wide": (1.3e-7, 5.7e-11),
    "join_proxy": (1.1e-7, 2.1e-9),
    "beam": (6.5e-8, 0.0),
    "count_margin": (1.3e-10, 0.0),
    "paint": (0.0, 0.0),
}


def quantum(kind: str, scale: float = 0.0) -> float:
    """The tie quantum of an assembly decision kind at ``scale`` (a power of two)."""
    floor, factor = ASSEMBLY_QUANTA[kind]
    target = max(floor, factor * abs(float(scale)) if math.isfinite(scale) else floor)
    return math.ldexp(1.0, int(math.ceil(math.log2(target))))


def motion(kind: str, scale: float = 0.0) -> float | None:
    """The measured noise motion of a decision kind's compared differences at ``scale`` (boundary guard)."""
    value = ASSEMBLY_MOTION.get(kind)
    if value is None:
        return None
    absolute, relative = value
    scale = abs(float(scale)) if math.isfinite(scale) else 0.0
    return max(absolute, relative * scale) or None


# ------------------------------------------------------------------------------------------- canonical order
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


def sort_panel(panel) -> list:
    """A panel (sequence of path tuples) in canonical order."""
    return sorted((path_key(path) for path in panel))


# ------------------------------------------------------------------------------------------- tie-class orders
def path_class_order(values, paths, quantum_value: float, *, descending: bool = False,
                     motion_value: float | None = None):
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
    columns = tuple(paths[:, column] for column in range(paths.shape[1] - 1, -1, -1))
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
    sorted_paths = paths[exact]
    # Content order inside a class: lexicographic paths, reversed under ties.reverse_conventions() (release 2.2
    # convention reversal; classes and values are not reversed).
    direction = -1 if ties.reverse_conventions() else 1
    inner = np.lexsort(tuple(direction * sorted_paths[:, column] for column in range(paths.shape[1] - 1, -1, -1))
                       + (value_key, class_sorted))
    order = exact[inner]
    classes = np.empty(n, dtype=np.int64)
    classes[exact] = class_sorted
    boundary = ()
    if motion_value is not None and n > 1:
        guard = ties.BOUNDARY_GUARD * float(motion_value)
        finite = np.isfinite(gaps) & (gaps != 0.0)
        boundary = tuple(float(gap) for gap in gaps[finite & (np.abs(gaps - quantum_value) <= guard)])
    return order.astype(np.int64), classes, int(np.count_nonzero(wide)), boundary


def linkage_order(result: ties.TieClasses, links, contents, kind: str = "beam") -> list:
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
            sub = ties.tie_classes(values, quantum(kind, max(finite) if finite else 0.0),
                                   content=[contents[m] for m in members], descending=True)
            members = [members[k] for k in sub.order]
        out.extend(members)
        start = end
    return out


def symmetric_members(result: ties.TieClasses, links, kind: str = "beam") -> list:
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
    sub = ties.tie_classes(values, quantum(kind, max(finite) if finite else 0.0), descending=True)
    if not sub.tied_winner:
        return []
    return [members[k] for k in sub.members(int(sub.classes[sub.winner]))]


def relevant_boundaries(values, ties_value: ties.TieClasses, quantum_value: float, motion_value: float | None, *,
                        descending: bool, cut: int | None = None) -> int:
    """Boundary-guard events of one decision: the gaps that decide it lying within BOUNDARY_GUARD noise motions of
    q -- for a winner the gap between the best class and the next, for a truncation that keeps ``cut`` items the
    gaps entering and leaving the class the cut falls in. (maths.ties reports every consecutive gap of the list;
    gaps elsewhere do not decide anything.)"""
    if motion_value is None or len(ties_value.classes) < 2:
        return 0
    signed = (-1.0 if descending else 1.0) * np.asarray(values, dtype=np.float64)
    classes = np.asarray(ties_value.classes)
    count = int(classes.max()) + 1
    low = np.full(count, np.inf)
    high = np.full(count, -np.inf)
    np.minimum.at(low, classes, signed)
    np.maximum.at(high, classes, signed)
    if cut is None:
        targets = [0]
    else:
        if cut <= 0 or cut >= len(ties_value.order):
            return 0
        cut_class = int(classes[ties_value.order[cut - 1]])
        targets = [cut_class - 1, cut_class]
    guard = ties.BOUNDARY_GUARD * float(motion_value)
    events = 0
    for c in targets:
        if 0 <= c < count - 1:
            gap = low[c + 1] - high[c]
            if np.isfinite(gap) and abs(gap - quantum_value) <= guard:
                events += 1
    return events


def add_decision(counter, kind: str, ties_value: ties.TieClasses, values, quantum_value: float,
                 motion_value: float | None, *, descending: bool, cut: int | None = None) -> None:
    """Count one tie-class decision with its decision-relevant boundary events (see relevant_boundaries): a
    winner is inside a tie when its class has more than one member, a truncation when the cut falls inside a
    class (members on both sides of it)."""
    if counter is None:
        return
    if cut is None:
        inside = ties_value.tied_winner
    else:
        order = ties_value.order
        inside = bool(0 < cut < len(order) and ties_value.classes[order[cut - 1]] == ties_value.classes[order[cut]])
    boundary = relevant_boundaries(values, ties_value, quantum_value, motion_value, descending=descending, cut=cut)
    if hasattr(counter, "add_counts"):
        counter.add_counts(kind, 1, int(inside), int(ties_value.wide_classes), boundary)
        return
    entry = counter.counts.setdefault(kind, dict(decisions=0, inside_tie=0, wide=0, boundary=0))
    entry["decisions"] += 1
    entry["inside_tie"] += int(inside)
    entry["wide"] += int(ties_value.wide_classes)
    entry["boundary"] += boundary


def classes_record(counter: ties.TieCounter | None, kind: str, ties_value: ties.TieClasses,
                   decided_inside: bool | None = None) -> None:
    """Add one decision to a TieCounter (no-op without one)."""
    if counter is not None:
        counter.add(kind, ties_value, decided_inside=decided_inside)


def count_vector_classes(counter: ties.TieCounter | None, kind: str, classes: np.ndarray, wide: int,
                         boundary: Sequence, decided_inside: bool) -> None:
    """Add one vectorised tie-class decision (path_class_order) to a TieCounter."""
    if counter is None:
        return
    if hasattr(counter, "add_counts"):
        counter.add_counts(kind, 1, int(bool(decided_inside)), int(wide), len(boundary))
        return
    entry = counter.counts.setdefault(kind, dict(decisions=0, inside_tie=0, wide=0, boundary=0))
    entry["decisions"] += 1
    entry["inside_tie"] += int(bool(decided_inside))
    entry["wide"] += int(wide)
    entry["boundary"] += len(boundary)


def truncate(items: list, ties_value: ties.TieClasses, limit: int, cap: int | None = None) -> list:
    """The first ``limit`` items in decision order, extended by the members of the class the cut falls in, so the
    whole cut class is kept (release 2.2: order-invariant). When the cut class would take the list beyond ``cap``
    items (default 2 x limit), the whole class is dropped instead and only the classes before it are kept (2.1 kept
    the class's first members in content order); a first class larger than the cap is kept whole. ``items`` are in
    the original indexing of ``ties_value``."""
    order = [int(i) for i in ties_value.order]
    if len(order) <= limit:
        return [items[i] for i in order]
    cap = 2 * limit if cap is None else cap
    classes = ties_value.classes
    cut_class = classes[order[limit - 1]]
    start = limit - 1
    while start > 0 and classes[order[start - 1]] == cut_class:
        start -= 1
    end = limit
    while end < len(order) and classes[order[end]] == cut_class:
        end += 1
    kept = order[:end] if (end <= cap or start == 0) else order[:start]
    return [items[i] for i in kept]


def census_key(*parts) -> str:
    """A census decision key from parts (joined with '/')."""
    return "/".join(str(part) for part in parts)


class SharedCounter(ties.TieCounter):
    """A maths.ties.TieCounter that concurrent threads of one task may add to (a lock around every update)."""

    def __init__(self):
        super().__init__()
        import threading
        self._lock = threading.Lock()

    def add(self, kind, ties_value, *, decided_inside=None):
        with self._lock:
            super().add(kind, ties_value, decided_inside=decided_inside)

    def note(self, kind, *, inside_tie=False, wide=0, boundary=0):
        with self._lock:
            super().note(kind, inside_tie=inside_tie, wide=wide, boundary=boundary)

    def add_counts(self, kind, decisions=0, inside_tie=0, wide=0, boundary=0):
        with self._lock:
            entry = self.counts.setdefault(kind, dict(decisions=0, inside_tie=0, wide=0, boundary=0))
            entry["decisions"] += int(decisions)
            entry["inside_tie"] += int(inside_tie)
            entry["wide"] += int(wide)
            entry["boundary"] += int(boundary)


def census(kind: str, key: str | None, values) -> None:
    """maths.ties.census when enabled and a key is given (values are only gathered then)."""
    if key is not None and ties.census_enabled():
        ties.census(kind, key, [float(value) for value in values])
