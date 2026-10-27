"""Tie classes, per-kind tie quanta and exact content orders (release 2.1 noise stability).

Floating values that differ only by rounding, or by the effect of input perturbations far below any real difference,
must not decide a discrete outcome, and neither may array order. This module gives every stage one way to say
"these values are tied" and one way to break such ties by content.

* **Tie quanta per decision kind** (:func:`tie_quantum`, :class:`BlockQuanta`). A quantum is a power of two far above
  the rounding of the compared sums and above the typical movement that relative GL perturbations of size GL_NOISE = 1e-6
  cause in the compared contrasts, and otherwise as small as possible: the smallest power of two at least
  NOISE_MARGIN = 16 times that motion (:func:`noise_quantum`). Release 2.1 placed the quanta against 1e-9 relative
  noise (N1's census, review R2); release 2.13 places them against 1e-6 (owner, 27 Oct: no decision tolerance below
  the noise floor). The motions are the 2.1 census values times 10^3 (the motion scales linearly with the noise
  amplitude; NOISE_MOTION below), or measured under a 1e-6 jitter where the 2.1 motion came from fit convergence; a
  relative factor with no recorded motion is the 2.1 factor times NOISE_SCALE = 2^10. Typical matched-history
  motions and linear-response extrapolations guide the tolerances; bounded replays test the resulting policy. This is
  not a worst-case noise certificate. Differences smaller than a
  quantum are decided by content (exact keys) or a stated prior, never by which of two values happens to be larger;
  the margin 16 is a safety factor over the motion, not a significance level, so this includes some differences whose
  sign is stable. Kinds:

  ============  =========================  ====================================================================
  kind          quantum (power of two >=)   compared quantities
  ============  =========================  ====================================================================
  nll           max(2^-10, 2^-20 x scale)   block NLLs (archive and beam order, dedup, cycles), bank BIC,
                                            cavity scores (K selection), gauge site sums
  h_step        2^-12                       H-step allele cost sums
  improvement   2^-10                       "improved" tests
  sample        max(2^-13, 2^-26 x scale)   per-sample sums: data-birth excess, A-step state costs
  cell          2^-14                       per-cell evidence contributions (supporter tests)
  probability   2^-15                       probabilities and masses (bands at 1/2, partner masses)
  ============  =========================  ====================================================================

  ``scale`` is the decision's magnitude (for a discovery block its null NLL, every panel's NLL being at most that).

* **Anchored tie classes** (:func:`tie_classes`; owner-reviewed rules, 18 Oct):
  (a) classes are single-linkage chains over the exactly sorted values (a consecutive gap <= q joins), so class
  membership is transitive by construction; (b) the winner's class is anchored at the best value (everything chained
  from it) and cut by content; full orders (beams, top-k cuts, budgets) chain along the whole sorted list with content
  order inside each class, i.e. every order is (class index, secondary exact keys, content key); (c) width guard: a
  class spanning more than WIDE_CLASS_QUANTA x q is reported and its internal order falls back to value order, so
  genuinely different values chained together are never reordered by content; (d) boundary guard: every consecutive
  gap within BOUNDARY_GUARD x the measured noise motion of the kind around q is reported (the empty-valley
  assumption failing must be visible); (e) quantities with no valley (decisiveness, refinement count margins) use
  content keys or the separate ambiguity policy, never chaining. Two values whose gap is much smaller than q stay in
  one class wherever they sit, unlike values snapped to a fixed grid (review R1). :class:`TieCounter` accumulates
  per-kind counts (decisions taken inside a tie, width-guard and boundary-guard events) for the stage records:
  a stage stores ``counter.record()`` as the ``tie_events`` field of the unit's summary record (kinds named
  "<stage>.<decision>"); the run's tie report sums them (reports/ties.py: OUTPUT/ties/, docs/methods/ties.md).

* **Exact content keys**: discrete data only (int8 row bytes, index paths, integer counts, sample names), never
  rounded floats. Elements that are equal on a content key must be exchangeable for the decision that uses it.

* **Canonical orders**: :func:`canonical_sample_order` (a block's samples by observation/dosage pattern, then exact
  GL bytes) and :func:`run_sample_order` (a run's samples by name, applied once at the evidence entry).

* :func:`census` (diagnostic valley census, design A-8; on only when HAPLOTYPES_TIE_CENSUS names a directory).

* **Convention reversal** (release 2.2 acceptance test, design/V22_BRIEF.md): :func:`reverse_conventions` is True when
  the environment sets HAPLOTYPES_TIE_REVERSE=1. Every remaining ARBITRARY convention then runs in reverse: index
  order, content-key order, binary default states (0/1), sample-name order and keep-first. Stated priors are NOT
  reversed (smaller K, fewer switches or parents, the conservative side of an exact threshold, keep-current inside a
  tie class), nor canonical labels (e.g. :func:`rows_key`). A released biological output that changes under reversal
  (outside released ambiguity) was selected by a convention. Helpers: :func:`conv_key` (reverse the last, arbitrary
  component of a sort key), :func:`conv_index`, :func:`conv_order`, :func:`conv_first`, :func:`conv_argmin`,
  :func:`conv_argmax`, :func:`conv_state`, :func:`conv_sorted`. :func:`tie_classes` (hence :func:`class_sort` and
  :func:`class_winner`) and :func:`run_sample_order` apply it themselves. Default: off, and every helper is the
  identity (no-op). Numba kernels take the flag as an argument (pass ``reverse_conventions()``).

* :func:`grid_key` only for values that are grid points by construction (e.g. the excess of an all-unknown sample,
  n_obs x 2 lambda nats); :func:`improves`; :func:`threshold_band` (owner decision 5; see design A-7).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

import numba
import numpy as np

# Relative GL noise the tie quanta are placed against (release 2.13; 2.1-2.12: 1e-9), the margin of a quantum over
# the motion of its compared contrast under that noise, and the factor (a power of two, the ratio of the noise levels)
# by which 2.13 multiplies a 2.1 relative factor that has no recorded motion.
GL_NOISE = 1e-6
NOISE_MARGIN = 16.0
NOISE_SCALE = 2.0 ** 10


def noise_quantum(motion: float) -> float:
    """The smallest power of two at least NOISE_MARGIN times ``motion`` (a contrast's motion under GL_NOISE)."""
    return math.ldexp(1.0, int(math.ceil(math.log2(NOISE_MARGIN * float(motion)))))


# Movement of the compared contrasts under GL_NOISE relative GL noise (nats; per kind): the N1 census measured under
# 1e-9 noise (review F1/F3: nll 5.5e-8, h_step 1e-8, improvement 5.5e-8, sample 5e-9, cell 2e-9, probability 1e-9)
# times 10^3, the motion scaling linearly with the noise amplitude.
NOISE_MOTION: dict[str, float] = {
    "nll": 5.5e-5, "h_step": 1e-5, "improvement": 5.5e-5, "sample": 5e-6, "cell": 2e-6, "probability": 1e-6,
}
# (floor, relative factor of the scale) per decision kind; quantum = power of two at least max(floor, factor x scale).
# Floors: noise_quantum of the kind's motion; factors: the 2.1 factors (no recorded relative motion) times NOISE_SCALE.
QUANTUM_KINDS: dict[str, tuple[float, float]] = {
    "nll": (noise_quantum(NOISE_MOTION["nll"]), 2.0 ** -30 * NOISE_SCALE),
    "h_step": (noise_quantum(NOISE_MOTION["h_step"]), 0.0),
    "improvement": (noise_quantum(NOISE_MOTION["improvement"]), 0.0),
    "sample": (noise_quantum(NOISE_MOTION["sample"]), 2.0 ** -36 * NOISE_SCALE),
    "cell": (noise_quantum(NOISE_MOTION["cell"]), 0.0),
    "probability": (noise_quantum(NOISE_MOTION["probability"]), 0.0),
}
# A tie class wider than this many quanta has no valley around it: reported, and ordered by value inside.
WIDE_CLASS_QUANTA = 4.0
# A consecutive gap within this many noise motions of q is a boundary-guard event.
BOUNDARY_GUARD = 10.0


def tie_quantum(scale: float, kind: str) -> float:
    """The power of two at least max(floor, factor x |scale|) of ``kind`` (see QUANTUM_KINDS)."""
    floor, factor = QUANTUM_KINDS[kind]
    target = max(floor, factor * abs(float(scale)))
    return math.ldexp(1.0, int(math.ceil(math.log2(target))))


@dataclass(frozen=True)
class BlockQuanta:
    """The per-kind tie quanta of one decision scope (e.g. one discovery block); ``scale`` is its magnitude."""

    scale: float
    nll: float
    h_step: float
    improvement: float
    sample: float
    cell: float
    probability: float

    @classmethod
    def for_scale(cls, scale: float) -> "BlockQuanta":
        scale = abs(float(scale))
        return cls(scale=scale, **{kind: tie_quantum(scale, kind) for kind in QUANTUM_KINDS})

    @classmethod
    def floors(cls) -> "BlockQuanta":
        """Quanta without a scale (every kind at its floor)."""
        return cls.for_scale(0.0)


def block_scale(likelihood: np.ndarray, observed: np.ndarray, unknown_copy_penalty: float) -> float:
    """A discovery block's null NLL: every sample's genotype-oracle NLL over its observed sites plus two unknown
    copies there (sum_s sum_obs (-log max_g GL + 2 lambda)), exactly rounded. Every panel's NLL is at most this."""
    likelihood = np.asarray(likelihood, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    maximum = np.maximum(likelihood.max(axis=2), np.finfo(np.float64).tiny)
    # -log through scalar libm (a Numba kernel calling math.log; NumPy's vectorised log takes CPU-dependent SIMD
    # paths that can differ by an ulp), so the block scale, hence its quanta, is the same on every machine
    oracle = _negative_log_where(np.ascontiguousarray(maximum), np.ascontiguousarray(observed))
    return math.fsum(oracle.tolist()) + 2.0 * float(unknown_copy_penalty) * float(np.count_nonzero(observed))


@numba.njit(cache=True, nogil=True)
def _negative_log_where(values, mask):
    """float64[count of mask]: -log(values) at the masked cells (row-major order), by scalar libm."""
    out = np.empty(int(mask.sum()), dtype=np.float64)
    k = 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if mask[i, j]:
                out[k] = -math.log(values[i, j])
                k += 1
    return out


@dataclass(frozen=True)
class TieClasses:
    """Anchored tie classes of a value array.

    order: int64[n] element indices in decision order: (class, secondary, content), or (class, value, secondary,
        content) inside a wide class (width guard).
    classes: int64[n] class index of each element (original indexing; 0 = the best class).
    spans: float64[C] max - min of each class.
    quantum: the q used.
    wide_classes: number of classes spanning more than WIDE_CLASS_QUANTA x q (width-guard events).
    boundary_gaps: consecutive gaps (sorted values) within the boundary guard of q, i.e. |gap - q| <= guard
        (boundary-guard events; empty when no motion is given).
    """

    order: np.ndarray
    classes: np.ndarray
    spans: np.ndarray
    quantum: float
    wide_classes: int = 0
    boundary_gaps: tuple = ()

    @property
    def wide(self) -> bool:
        return self.wide_classes > 0

    @property
    def tied_winner(self) -> bool:
        """The winner's class has more than one member (the decision was taken inside a tie)."""
        return bool(np.count_nonzero(self.classes == 0) > 1)

    @property
    def winner(self) -> int:
        return int(self.order[0])

    def members(self, class_index: int) -> np.ndarray:
        """Element indices of one class, in decision order."""
        return self.order[self.classes[self.order] == class_index]


def _exact_key(value):
    """A sort key for exact content data (bytes, str, ints, tuples of them, numpy integer arrays)."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuSU":
            raise TypeError("content keys must be exact discrete data, not floats")
        return (value.shape, np.ascontiguousarray(value).tobytes())
    if isinstance(value, float):
        raise TypeError("content keys must be exact discrete data, not floats")
    return value


def tie_classes(values: Sequence[float], quantum: float, *, content: Sequence | None = None,
                secondary: Sequence | None = None, descending: bool = False,
                motion: float | None = None) -> TieClasses:
    """Anchored tie classes of ``values`` and the decision order.

    values: finite or infinite floats (NaN is refused). quantum: q > 0. content: one exact key per element (bytes,
    str, ints, tuples, integer arrays); None means the elements are exchangeable and their index order is used.
    secondary: optional exact keys applied after the class and before the content key (e.g. founder count).
    descending: larger values first (scores); the default puts smaller values first (NLLs, costs).
    motion: the measured noise motion of the compared quantity (e.g. NOISE_MOTION[kind]); enables the boundary guard.
    Sorting is by exact value (ties by content key); a new class starts where the gap to the previous value exceeds q;
    inside a class wider than WIDE_CLASS_QUANTA x q the order stays by value (width guard). Under
    reverse_conventions() the content key (or index) order inside a class is reversed; classes, values and
    secondary keys (stated preferences) are not.
    """
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    if np.isnan(x).any():
        raise ValueError("tie classes of NaN are undefined")
    if quantum <= 0.0:
        raise ValueError("the tie quantum must be positive")
    keys = [(_exact_key(content[i]) if content is not None else i) for i in range(n)]
    sign = -1.0 if descending else 1.0
    exact_order = sorted(range(n), key=lambda i: (sign * x[i], keys[i]))
    classes = np.empty(n, dtype=np.int64)
    spans = []
    boundary = []
    guard = None if motion is None else BOUNDARY_GUARD * float(motion)
    current = -1
    start_value = previous = None
    for i in exact_order:
        value = sign * x[i]
        if previous is not None and guard is not None and value != previous:
            gap = value - previous
            if np.isfinite(gap) and abs(gap - quantum) <= guard:
                boundary.append(float(gap))
        if previous is None or (value != previous and not (value - previous <= quantum)):
            if previous is not None:
                spans.append(0.0 if previous == start_value else previous - start_value)
            current += 1
            start_value = value
        classes[i] = current
        previous = value
    if previous is not None:
        spans.append(0.0 if previous == start_value else previous - start_value)
    spans_array = np.asarray([0.0 if not np.isfinite(span) else span for span in spans], dtype=np.float64)
    wide = spans_array > WIDE_CLASS_QUANTA * quantum
    second = [(_exact_key(secondary[i]) if secondary is not None else ()) for i in range(n)]

    last = [conv_key(key) for key in keys] if reverse_conventions() else keys     # the arbitrary component

    def decision_key(i):
        c = classes[i]
        if wide[c]:
            return (c, sign * x[i], second[i], last[i])
        return (c, 0.0, second[i], last[i])

    order = np.asarray(sorted(range(n), key=decision_key), dtype=np.int64)
    return TieClasses(order=order, classes=classes, spans=spans_array, quantum=float(quantum),
                      wide_classes=int(np.count_nonzero(wide)), boundary_gaps=tuple(boundary))


class TieCounter:
    """Per-kind counts for stage records (review r9; owner rule: the harness counts guard events per stage):
    decisions taken inside a tie (the winner's class or a truncation cut had several members), width-guard events
    (wide classes) and boundary-guard events (gaps within the boundary guard of q)."""

    def __init__(self):
        self.counts: dict[str, dict[str, int]] = {}

    def add(self, kind: str, ties: TieClasses, *, decided_inside: bool | None = None) -> None:
        entry = self.counts.setdefault(kind, dict(decisions=0, inside_tie=0, wide=0, boundary=0))
        entry["decisions"] += 1
        entry["inside_tie"] += int(ties.tied_winner if decided_inside is None else decided_inside)
        entry["wide"] += int(ties.wide_classes)
        entry["boundary"] += len(ties.boundary_gaps)

    def note(self, kind: str, *, inside_tie: bool = False, wide: int = 0, boundary: int = 0) -> None:
        """Count one decision taken without a TieClasses object (e.g. a threshold band: inside_tie = in the band)."""
        entry = self.counts.setdefault(kind, dict(decisions=0, inside_tie=0, wide=0, boundary=0))
        entry["decisions"] += 1
        entry["inside_tie"] += int(bool(inside_tie))
        entry["wide"] += int(wide)
        entry["boundary"] += int(boundary)

    def merge(self, other: "TieCounter") -> None:
        for kind, values in other.counts.items():
            entry = self.counts.setdefault(kind, dict(decisions=0, inside_tie=0, wide=0, boundary=0))
            for name, value in values.items():
                entry[name] += value

    def record(self) -> dict:
        return {kind: dict(values) for kind, values in sorted(self.counts.items())}


def class_sort(items: Iterable, value: Callable, quantum: float, *, secondary: Callable | None = None,
               content: Callable | None = None, descending: bool = False, motion: float | None = None,
               counter: TieCounter | None = None, kind: str = "") -> list:
    """``items`` in tie-class decision order (value classes, then class-internal secondary and content keys)."""
    items = list(items)
    if not items:
        return []
    ties = tie_classes([value(item) for item in items], quantum,
                       content=None if content is None else [content(item) for item in items],
                       secondary=None if secondary is None else [secondary(item) for item in items],
                       descending=descending, motion=motion)
    if counter is not None:
        counter.add(kind or "unnamed", ties)
    return [items[i] for i in ties.order]


def class_winner(items: Iterable, value: Callable, quantum: float, *, secondary: Callable | None = None,
                 content: Callable | None = None, descending: bool = False, motion: float | None = None,
                 counter: TieCounter | None = None, kind: str = ""):
    """The first item of :func:`class_sort`: the best class (anchored at the best value, everything chained from it),
    cut by secondary and content keys."""
    ordered = class_sort(items, value, quantum, secondary=secondary, content=content, descending=descending,
                         motion=motion, counter=counter, kind=kind)
    return ordered[0] if ordered else None


def improves(new: float, old: float, quantum: float, *, higher_is_better: bool = True) -> bool:
    """Whether ``new`` beats ``old`` by more than the quantum."""
    return (new > old + quantum) if higher_is_better else (new < old - quantum)


def grid_key(value: float, quantum: float) -> int:
    """Integer grid key floor(value / q + 1/2). Use only for values that are grid points by construction (a grid
    key splits values that straddle a cell boundary, review R1)."""
    return int(math.floor(float(value) / quantum + 0.5))


def threshold_band(value: float, threshold: float, quantum: float) -> int:
    """-1 below the band [threshold - q, threshold + q], 0 inside, +1 above. For threshold rules that give values
    sitting exactly at a threshold a fixed conservative decision (owner decision 5; design A-7: use the band only where
    values can sit at the threshold mathematically)."""
    if value < threshold - quantum:
        return -1
    if value > threshold + quantum:
        return 1
    return 0


def rows_key(matrix: np.ndarray) -> bytes:
    """Content key of a set of binary rows: the int8 bytes of the rows in lexicographic order."""
    rows = np.asarray(matrix, dtype=np.int8)
    if rows.ndim != 2:
        raise ValueError("rows_key expects a 2-D matrix")
    order = sorted(range(len(rows)), key=lambda index: rows[index].tobytes())
    return np.ascontiguousarray(rows[order]).tobytes()


def canonical_sample_order(likelihood: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """int64[N] permutation putting a block's samples in content order: observation and argmax-dosage pattern
    (0 = unobserved, 1-3 = dosage + 1), then the exact GLs (non-negative float64 compare as big-endian bytes). Only
    samples with bitwise identical evidence share a key; samples equal on the dosage pattern are ordered by their
    exact GLs, and any tie-break that uses this order must treat such samples as exchangeable."""
    likelihood = np.asarray(likelihood, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    dosage = np.where(observed, np.argmax(likelihood, axis=2) + 1, 0).astype(np.uint8)
    exact = np.ascontiguousarray(likelihood, dtype=">f8")
    keys = [dosage[s].tobytes() + exact[s].tobytes() for s in range(likelihood.shape[0])]
    return np.asarray(sorted(range(len(keys)), key=lambda s: conv_key(keys[s])), dtype=np.int64)


def run_sample_order(sample_ids: Sequence[str]) -> np.ndarray:
    """int64[N] permutation putting a run's samples in canonical order (by name; names are unique and travel with the
    data, so the order does not depend on the input order). Applied once at the evidence entry (design A-11).
    Reverse name order under reverse_conventions()."""
    names = [str(name) for name in sample_ids]
    if len(set(names)) != len(names):
        raise ValueError("sample names must be unique for the canonical run order")
    return np.asarray(sorted(range(len(names)), key=lambda index: names[index], reverse=reverse_conventions()),
                      dtype=np.int64)


def inverse_permutation(order: np.ndarray) -> np.ndarray:
    """The inverse of a permutation (inverse[order[i]] = i)."""
    order = np.asarray(order, dtype=np.int64)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order), dtype=np.int64)
    return inverse


# Valley census (design A-8; diagnostic only). When HAPLOTYPES_TIE_CENSUS names a directory (set by the lab harness,
# never by a configuration), every census() call appends one JSON line {"kind", "key", "values"} to
# DIR/census.<pid>.jsonl; work/N2/harness/census.py histograms the gaps of a reference run and the motion of the same
# decisions in perturbed reruns. ``key`` must identify the decision uniquely and stably across reruns of the same input
# (e.g. "chr8/block123/archive/7"). Disabled (the default) it costs one dictionary lookup per call.
_CENSUS: dict = {"dir": None, "pid": None, "handle": None}


def census_enabled() -> bool:
    """Whether the valley census is collecting (callers may skip gathering values when it is not)."""
    import os
    if _CENSUS["pid"] is None:
        _CENSUS["dir"] = os.environ.get("HAPLOTYPES_TIE_CENSUS") or None
        _CENSUS["pid"] = -1
    return _CENSUS["dir"] is not None


def census(kind: str, key: str, values: Iterable[float]) -> None:
    """Append one decision's compared values to the valley census (no-op unless enabled)."""
    if not census_enabled():
        return
    import json
    import os
    pid = os.getpid()
    if _CENSUS["pid"] != pid:
        os.makedirs(_CENSUS["dir"], exist_ok=True)
        _CENSUS["handle"] = open(os.path.join(_CENSUS["dir"], f"census.{pid}.jsonl"), "a", buffering=1)
        _CENSUS["pid"] = pid
    _CENSUS["handle"].write(json.dumps({"kind": str(kind), "key": str(key),
                                        "values": [float(value) for value in values]}) + "\n")


# Convention reversal (see the module docstring). Read once per process from the environment, which worker processes
# inherit; set_reverse_conventions() overrides it for tests and harness tools (never midway through a run).
_REVERSE: dict = {"value": None}


def reverse_conventions() -> bool:
    """Whether the run reverses its arbitrary conventions (HAPLOTYPES_TIE_REVERSE set to 1/true/yes/on)."""
    value = _REVERSE["value"]
    if value is None:
        import os
        text = os.environ.get("HAPLOTYPES_TIE_REVERSE", "").strip().lower()
        value = text not in ("", "0", "false", "no", "off")
        _REVERSE["value"] = value
    return value


def set_reverse_conventions(value: bool | None) -> None:
    """Force the reversal on (True) or off (False), or go back to the environment (None). Processes started earlier
    keep their own value; a run must not change it midway."""
    _REVERSE["value"] = None if value is None else bool(value)


class _Descending:
    """A sort key that inverts the order of an exact key (conv_key under reversal)."""

    __slots__ = ("key",)

    def __init__(self, key):
        self.key = key

    def __lt__(self, other):
        return other.key < self.key

    def __gt__(self, other):
        return other.key > self.key

    def __le__(self, other):
        return other.key <= self.key

    def __ge__(self, other):
        return other.key >= self.key

    def __eq__(self, other):
        return isinstance(other, _Descending) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        return f"_Descending({self.key!r})"


def conv_key(key):
    """An exact key in convention order: ``key`` itself, or a wrapper that sorts in reverse under the reversal. Use it
    only for the arbitrary (last) component of a sort key, never for values or stated preferences."""
    return _Descending(key) if reverse_conventions() else key


def conv_index(index: int, size: int) -> int:
    """An index as a convention rank: ``index``, or ``size - 1 - index`` under the reversal (so "the lowest index"
    becomes "the highest")."""
    return int(size - 1 - index) if reverse_conventions() else int(index)


def conv_order(size: int) -> np.ndarray:
    """int64[size]: the convention's traversal order of ``size`` exchangeable items (0 .. size-1, or reversed)."""
    order = np.arange(int(size), dtype=np.int64)
    return order[::-1].copy() if reverse_conventions() else order


def conv_first(items: Sequence):
    """Keep-first: the first element of a sequence, or the last under the reversal (None if empty)."""
    if len(items) == 0:
        return None
    return items[-1] if reverse_conventions() else items[0]


def conv_argmin(values) -> int:
    """np.argmin over the flattened values with exact ties going to the lowest index, or to the highest under the
    reversal."""
    a = np.ravel(np.asarray(values))
    if not reverse_conventions():
        return int(np.argmin(a))
    return int(len(a) - 1 - np.argmin(a[::-1]))


def conv_argmax(values) -> int:
    """np.argmax over the flattened values with exact ties going to the lowest index, or to the highest under the
    reversal."""
    a = np.ravel(np.asarray(values))
    if not reverse_conventions():
        return int(np.argmax(a))
    return int(len(a) - 1 - np.argmax(a[::-1]))


def conv_state(state: int) -> int:
    """A binary default state (0/1; e.g. "frame 0 = scaffold orientation", "state 0 on a tie") in the convention:
    ``state``, or ``1 - state`` under the reversal."""
    return 1 - int(state) if reverse_conventions() else int(state)


def conv_sorted(items: Iterable, key: Callable | None = None) -> list:
    """sorted(items, key=key), or its exact reverse under the reversal (equal keys in reversed order too). For orders
    that are wholly conventions (e.g. a traversal by name or index); for (value, ..., content) orders use conv_key on
    the content component instead."""
    ordered = sorted(items, key=key)
    return ordered[::-1] if reverse_conventions() else ordered
