"""Keep decisions stable under numerical noise through explicit, named tie policies.

Owns the six shared quanta, assembly quanta, single-linkage classes, width and boundary
 guards, content conventions and typed counters (P0 EC30-34). Assembly path/linkage
 orders belong to science.assembly.orders. Arithmetic is relocated from 3.5 unchanged;
 L8 replaces reversal, discovery keep switches and census environment reads with
 explicit arguments. Census sinks are supplied by stages; this module performs no I/O.

Caller guide (STYLE_40 §6; P0 EC30-34): name the decision and its original quantity,
units, direction and scale beside the call. State the exact discrete content key and
why equal keys are exchangeable. State each secondary preference separately: it is
science, not an arbitrary convention, and never reverses. Pass the same explicit
reverse setting through the whole decision. Record relevant boundary operands and
return typed diagnostics; only stage adapters turn them into payload dictionaries.
Do not use these classes for an incumbent-preference rule with a different mechanism.

Unless stated otherwise, public helpers do not mutate their inputs. Arrays returned
by TieClasses are newly allocated; the record is frozen but its arrays are owned by
the caller. Counters are mutable task-local accumulators; record() returns a frozen
snapshot. Inputs must have the documented shapes and exact, mutually comparable
keys: NumPy shape/conversion errors, IndexError for short keys, and TypeError for
incomparable keys propagate. There is no fallback that changes a decision on error.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numba
import numpy as np

from .kernels import kernel

# Relative GL noise the tie quanta are placed against (release 2.13; 2.1-2.12: 1e-9), the margin of a quantum over
# the motion of its compared contrast under that noise, and the factor (a power of two, the ratio of the noise levels)
# by which 2.13 multiplies a 2.1 relative factor that has no recorded motion.
GL_NOISE = 1e-6
NOISE_MARGIN = 16.0
NOISE_SCALE = 2.0**10


def noise_quantum(motion: float) -> float:
    """The smallest power of two at least NOISE_MARGIN times ``motion`` (a contrast's motion under GL_NOISE).

    No mutation. motion must be positive and finite; math domain/overflow errors propagate.
    """
    return math.ldexp(1.0, int(math.ceil(math.log2(NOISE_MARGIN * float(motion)))))


# Movement of the compared contrasts under GL_NOISE relative GL noise (nats; per kind): the N1 census measured under
# 1e-9 noise (review F1/F3: nll 5.5e-8, h_step 1e-8, improvement 5.5e-8, sample 5e-9, cell 2e-9, probability 1e-9)
# times 10^3, the motion scaling linearly with the noise amplitude.
NOISE_MOTION: dict[str, float] = {
    "nll": 5.5e-5,
    "h_step": 1e-5,
    "improvement": 5.5e-5,
    "sample": 5e-6,
    "cell": 2e-6,
    "probability": 1e-6,
}
# (floor, relative factor of the scale) per decision kind; quantum = power of two at least max(floor, factor x scale).
# Floors: noise_quantum of the kind's motion; factors: the 2.1 factors (no recorded relative motion) times NOISE_SCALE.
QUANTUM_KINDS: dict[str, tuple[float, float]] = {
    "nll": (noise_quantum(NOISE_MOTION["nll"]), 2.0**-30 * NOISE_SCALE),
    "h_step": (noise_quantum(NOISE_MOTION["h_step"]), 0.0),
    "improvement": (noise_quantum(NOISE_MOTION["improvement"]), 0.0),
    "sample": (noise_quantum(NOISE_MOTION["sample"]), 2.0**-36 * NOISE_SCALE),
    "cell": (noise_quantum(NOISE_MOTION["cell"]), 0.0),
    "probability": (noise_quantum(NOISE_MOTION["probability"]), 0.0),
}
# A tie class wider than this many quanta has no valley around it: reported, and ordered by value inside.
WIDE_CLASS_QUANTA = 4.0
# A consecutive gap within this many noise motions of q is a boundary-guard event.
BOUNDARY_GUARD = 10.0


def tie_quantum(scale: float, kind: str) -> float:
    """The power of two at least max(floor, factor x |scale|) of ``kind`` (see QUANTUM_KINDS).

    No mutation. Unknown kind raises KeyError; numeric conversion/domain errors propagate.
    """
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
    def for_scale(cls, scale: float) -> BlockQuanta:
        """Build frozen quanta without mutation; float conversion/domain errors propagate."""
        scale = abs(float(scale))
        return cls(scale=scale, **{kind: tie_quantum(scale, kind) for kind in QUANTUM_KINDS})

    @classmethod
    def floors(cls) -> BlockQuanta:
        """Quanta without a scale (every kind at its floor)."""
        return cls.for_scale(0.0)


def block_scale(likelihood: np.ndarray, observed: np.ndarray, unknown_copy_penalty: float) -> float:
    """A discovery block's null NLL: every sample's genotype-oracle NLL over its observed sites plus two unknown
    copies there (sum_s sum_obs (-log max_g GL + 2 lambda)), exactly rounded. Every panel's NLL is at most this.

    Inputs float64[N,L,3] GL and bool[N,L] mask are unchanged. Returns nats; invalid shapes raise NumPy errors.
    """
    likelihood = np.asarray(likelihood, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    maximum = np.maximum(likelihood.max(axis=2), np.finfo(np.float64).tiny)
    # -log through scalar libm (a Numba kernel calling math.log; NumPy's vectorised log takes CPU-dependent SIMD
    # paths that can differ by an ulp), so the block scale, hence its quanta, is the same on every machine
    oracle = _negative_log_where(np.ascontiguousarray(maximum), np.ascontiguousarray(observed))
    return math.fsum(oracle.tolist()) + 2.0 * float(unknown_copy_penalty) * float(np.count_nonzero(observed))


@kernel(
    overloads=tuple(
        (
            numba.types.Array(numba.float64, 2, "C", readonly=readonly),
            numba.types.Array(numba.boolean, 2, "C", readonly=mask_readonly),
        )
        for readonly in (False, True)
        for mask_readonly in (False, True)
    ),
    cache=True,
    nogil=True,
)
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
        """Report any wide class without mutation or additional validation."""
        return self.wide_classes > 0

    @property
    def tied_winner(self) -> bool:
        """The winner's class has more than one member (the decision was taken inside a tie)."""
        return bool(np.count_nonzero(self.classes == 0) > 1)

    @property
    def winner(self) -> int:
        """Return the winning original index; no mutation, IndexError for an empty decision."""
        return int(self.order[0])

    def members(self, class_index: int) -> np.ndarray:
        """Return new int64 member indices in decision order; unknown classes give an empty array."""
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


@kernel(
    overloads=tuple(
        (
            numba.types.Array(numba.float64, 1, layout, readonly=readonly),
            numba.int64[::1],
            numba.float64,
            numba.float64,
            numba.float64,
            numba.boolean,
        )
        for layout in ("C", "A")
        for readonly in (False, True)
    ),
    cache=True,
    nogil=True,
)
def _class_membership(x, exact_order, sign, quantum, guard, with_guard):
    """The original consecutive-gap scan, with the same float64 arithmetic and guards."""
    n = len(x)
    classes = np.empty(n, dtype=np.int64)
    spans = np.empty(n, dtype=np.float64)
    boundary = np.empty(max(0, n - 1), dtype=np.float64)
    current = -1
    boundary_count = 0
    previous = start_value = 0.0
    for position in range(n):
        i = exact_order[position]
        value = sign * x[i]
        if position and with_guard and value != previous:
            gap = value - previous
            if np.isfinite(gap) and abs(gap - quantum) <= guard:
                boundary[boundary_count] = gap
                boundary_count += 1
        if position == 0 or (value != previous and not (value - previous <= quantum)):
            if position:
                span = 0.0 if previous == start_value else previous - start_value
                spans[current] = span if np.isfinite(span) else 0.0
            current += 1
            start_value = value
        classes[i] = current
        previous = value
    if n:
        span = 0.0 if previous == start_value else previous - start_value
        spans[current] = span if np.isfinite(span) else 0.0
    return classes, spans[: current + 1], boundary[:boundary_count]


def tie_classes(
    values: Sequence[float] | np.ndarray,
    quantum: float,
    *,
    content: Sequence | None = None,
    secondary: Sequence | None = None,
    descending: bool = False,
    motion: float | None = None,
    reverse: bool = False,
) -> TieClasses:
    """Anchored tie classes of ``values`` and the decision order.

    values: finite or infinite floats (NaN is refused). quantum: q > 0. content: one exact key per element (bytes,
    str, ints, tuples, integer arrays); None means the elements are exchangeable and their index order is used.
    secondary: optional exact keys applied after the class and before the content key (e.g. founder count).
    descending: larger values first (scores); the default puts smaller values first (NLLs, costs).
    motion: the measured noise motion of the compared quantity (e.g. NOISE_MOTION[kind]); enables the boundary guard.
    Sorting is by exact value (ties by content key); a new class starts where the gap to the previous value exceeds q;
    inside a class wider than WIDE_CLASS_QUANTA x q the order stays by value (width guard). Under
    reverse_conventions(reverse) the content key (or index) order inside a class is reversed; classes, values and
    secondary keys (stated preferences) are not.

        No inputs are mutated. Raises ValueError for NaN or nonpositive q, TypeError for floating content keys; key-
    length/index errors propagate.
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
    classes, spans_array, boundary = _class_membership(
        x,
        np.asarray(exact_order, dtype=np.int64),
        sign,
        float(quantum),
        0.0 if motion is None else BOUNDARY_GUARD * float(motion),
        motion is not None,
    )
    wide = spans_array > WIDE_CLASS_QUANTA * quantum
    second = [(_exact_key(secondary[i]) if secondary is not None else ()) for i in range(n)]

    last = (
        [conv_key(key, reverse=reverse) for key in keys] if reverse_conventions(reverse) else keys
    )  # the arbitrary component

    def decision_key(i):
        c = classes[i]
        if wide[c]:
            return (c, sign * x[i], second[i], last[i])
        return (c, 0.0, second[i], last[i])

    order = np.asarray(sorted(range(n), key=decision_key), dtype=np.int64)
    return TieClasses(
        order=order,
        classes=classes,
        spans=spans_array,
        quantum=float(quantum),
        wide_classes=int(np.count_nonzero(wide)),
        boundary_gaps=tuple(boundary.tolist()),
    )


@dataclass(frozen=True)
class TieCount:
    """One kind's integer decision counts; no scientific input is retained or mutated."""

    kind: str = field(metadata={"role": "diagnostic", "axes": (), "units": "1", "dtype": "str"})
    decisions: int = field(default=0, metadata={"role": "diagnostic", "axes": (), "units": "1", "dtype": "int64"})
    inside_tie: int = field(default=0, metadata={"role": "diagnostic", "axes": (), "units": "1", "dtype": "int64"})
    wide: int = field(default=0, metadata={"role": "diagnostic", "axes": (), "units": "1", "dtype": "int64"})
    boundary: int = field(default=0, metadata={"role": "diagnostic", "axes": (), "units": "1", "dtype": "int64"})


@dataclass(frozen=True)
class TieCounts:
    """Immutable, kind-sorted snapshot for a stage's typed diagnostics."""

    entries: tuple[TieCount, ...] = field(
        metadata={"role": "diagnostic", "axes": ("kind",), "units": "1", "dtype": "record"}
    )


@dataclass(frozen=True)
class TieCensus:
    """Compared float values in original draw/decision order for one named decision."""

    kind: str = field(metadata={"role": "diagnostic", "axes": (), "units": "1", "dtype": "str"})
    key: str = field(metadata={"role": "diagnostic", "axes": (), "units": "1", "dtype": "str"})
    values: tuple[float, ...] = field(
        metadata={"role": "diagnostic", "axes": ("candidate",), "units": "policy kind", "dtype": "float64"}
    )


@dataclass(frozen=True)
class TieSettings:
    """Explicit run conventions and a task-local diagnostic sink, with no environment reads.

    Reverse and keep belong in derivation identities; the sink never does. Runtime
    constructs the settings; science only consumes them. ``keep`` applies to the
    named discovery conventions that 3.5 allowed callers to hold at default.
    """

    reverse: bool = False
    keep: tuple[str, ...] = ()
    census: Callable[[TieCensus], None] | None = None

    def reversed(self, name: str) -> bool:
        """Return a discovery convention's effective reversal without changing settings."""
        return rev(name, reverse=self.reverse, keep=self.keep)


class TieCounter:
    """Accumulate per-kind decisions, ties, wide classes and boundary gaps (P0 EC30-34).

    Updates mutate only this accumulator; record() returns independent frozen values.
    Use SharedCounter for concurrent updates. Numeric conversion errors propagate.
    """

    def __init__(self):
        self._counts: dict[str, TieCount] = {}

    def add_counts(self, kind: str, decisions=0, inside_tie=0, wide=0, boundary=0) -> None:
        """Add integer counts to this accumulator; int conversion errors propagate."""
        entry = self._counts.get(kind, TieCount(kind))
        self._counts[kind] = TieCount(
            kind,
            entry.decisions + int(decisions),
            entry.inside_tie + int(inside_tie),
            entry.wide + int(wide),
            entry.boundary + int(boundary),
        )

    def add(self, kind: str, ties: TieClasses, *, decided_inside: bool | None = None) -> None:
        """Mutate this counter by one decision; leave ties and its arrays unchanged."""
        self.add_counts(
            kind,
            1,
            int(ties.tied_winner if decided_inside is None else decided_inside),
            int(ties.wide_classes),
            len(ties.boundary_gaps),
        )

    def note(self, kind: str, *, inside_tie: bool = False, wide: int = 0, boundary: int = 0) -> None:
        """Mutate this counter by one threshold decision without a TieClasses value."""
        self.add_counts(kind, 1, int(bool(inside_tie)), int(wide), int(boundary))

    def merge(self, other: TieCounter) -> None:
        """Add another counter's snapshot to this counter; leave the other unchanged."""
        for entry in other.record().entries:
            self.add_counts(entry.kind, entry.decisions, entry.inside_tie, entry.wide, entry.boundary)

    def record(self) -> TieCounts:
        """Return a frozen kind-sorted snapshot; neither counter nor earlier snapshots change."""
        return TieCounts(tuple(self._counts[kind] for kind in sorted(self._counts)))


def class_sort(
    items: Iterable,
    value: Callable,
    quantum: float,
    *,
    secondary: Callable | None = None,
    content: Callable | None = None,
    descending: bool = False,
    motion: float | None = None,
    counter: TieCounter | None = None,
    kind: str = "",
    reverse: bool = False,
) -> list:
    """``items`` in tie-class decision order (value classes, then class-internal secondary and content keys).

        Consumes items once, returns a new list, and increments counter when nonempty. Does not mutate items or
    their contents. Callback exceptions and tie_classes errors propagate.
    """
    items = list(items)
    if not items:
        return []
    ties = tie_classes(
        [value(item) for item in items],
        quantum,
        content=None if content is None else [content(item) for item in items],
        secondary=None if secondary is None else [secondary(item) for item in items],
        descending=descending,
        motion=motion,
        reverse=reverse,
    )
    if counter is not None:
        counter.add(kind or "unnamed", ties)
    return [items[i] for i in ties.order]


def class_winner(
    items: Iterable,
    value: Callable,
    quantum: float,
    *,
    secondary: Callable | None = None,
    content: Callable | None = None,
    descending: bool = False,
    motion: float | None = None,
    counter: TieCounter | None = None,
    kind: str = "",
    reverse: bool = False,
):
    """The first item of :func:`class_sort`: the best class (anchored at the best value, everything chained from it),
    cut by secondary and content keys.

        Returns None for empty items. Mutates a supplied counter through class_sort; all callback/classification
    errors propagate.
    """
    ordered = class_sort(
        items,
        value,
        quantum,
        secondary=secondary,
        content=content,
        descending=descending,
        motion=motion,
        counter=counter,
        kind=kind,
        reverse=reverse,
    )
    return ordered[0] if ordered else None


def improves(new: float, old: float, quantum: float, *, higher_is_better: bool = True) -> bool:
    """Whether ``new`` beats ``old`` by more than the quantum.

    No mutation or custom errors; operands and quantum use the original strict floating comparison.
    """
    return (new > old + quantum) if higher_is_better else (new < old - quantum)


def threshold_band(value: float, threshold: float, quantum: float) -> int:
    """-1 below the band [threshold - q, threshold + q], 0 inside, +1 above. For threshold rules that give values
    sitting exactly at a threshold a fixed conservative decision (owner decision 5; design A-7: use the band only where
    values can sit at the threshold mathematically).

    No mutation or custom errors; callers supply finite operands and the policy quantum.
    """
    if value < threshold - quantum:
        return -1
    if value > threshold + quantum:
        return 1
    return 0


def run_sample_order(sample_ids: Sequence[str], *, reverse: bool = False) -> np.ndarray:
    """int64[N] permutation putting a run's samples in canonical order (by name; names are unique and travel with the
    data, so the order does not depend on the input order). Applied once at the evidence entry (design A-11).
    Reverse name order under reverse_conventions(reverse).

    Does not mutate names. Duplicate stringified names raise ValueError.
    """
    names = [str(name) for name in sample_ids]
    if len(set(names)) != len(names):
        raise ValueError("sample names must be unique for the canonical run order")
    return np.asarray(
        sorted(range(len(names)), key=lambda index: names[index], reverse=reverse_conventions(reverse)), dtype=np.int64
    )


def inverse_permutation(order: np.ndarray) -> np.ndarray:
    """The inverse of a permutation (inverse[order[i]] = i).

    Input int64[N] must be a permutation of range(N); it is unchanged. Out-of-range indices raise IndexError.
    """
    order = np.asarray(order, dtype=np.int64)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order), dtype=np.int64)
    return inverse


def census_enabled(sink: Callable[[TieCensus], None] | None = None) -> bool:
    """Whether an explicit diagnostic sink is collecting compared values.

    Does not call or mutate the sink; no custom errors are raised.
    """
    return sink is not None


def census(
    kind: str, key: str | None, values: Iterable[float], *, sink: Callable[[TieCensus], None] | None = None
) -> None:
    """Deliver the original census record to the caller's sink; never open files.

        Consumes values only when sink and key are present. Inputs are unchanged; the sink may mutate its own
    collector. Float conversion and sink exceptions propagate. The stage adapter owns dictionary serialization.
    """
    if sink is not None and key is not None:
        sink(TieCensus(str(kind), str(key), tuple(float(value) for value in values)))


def reverse_conventions(reverse: bool = False) -> bool:
    """Return the explicit reversal hook; no process-global or environment state.

    No input mutation; returns bool(reverse), propagating a custom __bool__ error if supplied.
    """
    return bool(reverse)


def rev(name: str, *, reverse: bool = False, keep: Sequence[str] = ()) -> bool:
    """Reverse a discovery convention unless its name (or 'all') is explicitly kept.

    No mutation; keep is a sequence of exact policy names (or all), with no environment parsing.
    """
    return bool(reverse) and "all" not in keep and name not in keep


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


def conv_key(key, *, reverse: bool = False):
    """An exact key in convention order: ``key`` itself, or a wrapper that sorts in reverse under the reversal. Use it
    only for the arbitrary (last) component of a sort key, never for values or stated preferences.

    No mutation. The wrapped discrete key must support comparison; comparison errors propagate.
    """
    return _Descending(key) if reverse_conventions(reverse) else key


def conv_order(size: int, *, reverse: bool = False) -> np.ndarray:
    """int64[size]: the convention's traversal order of ``size`` exchangeable items (0 .. size-1, or reversed).

    Returns a new int64 permutation; no mutation. Size must be a nonnegative integer.
    """
    order = np.arange(int(size), dtype=np.int64)
    return order[::-1].copy() if reverse_conventions(reverse) else order


def conv_argmin(values, *, reverse: bool = False) -> int:
    """np.argmin over the flattened values with exact ties going to the lowest index, or to the highest under the
    reversal.

    No mutation; NumPy errors for empty or invalid value arrays propagate.
    """
    a = np.ravel(np.asarray(values))
    if not reverse_conventions(reverse):
        return int(np.argmin(a))
    return int(len(a) - 1 - np.argmin(a[::-1]))


def conv_argmax(values, *, reverse: bool = False) -> int:
    """np.argmax over the flattened values with exact ties going to the lowest index, or to the highest under the
    reversal.

    No mutation; NumPy errors for empty or invalid value arrays propagate.
    """
    a = np.ravel(np.asarray(values))
    if not reverse_conventions(reverse):
        return int(np.argmax(a))
    return int(len(a) - 1 - np.argmax(a[::-1]))


def conv_sorted(items: Iterable, key: Callable | None = None, *, reverse: bool = False) -> list:
    """sorted(items, key=key), or its exact reverse under the reversal (equal keys in reversed order too). For orders
    that are wholly conventions (e.g. a traversal by name or index); for (value, ..., content) orders use conv_key on
    the content component instead.

    Consumes items once, returns a new list; key/comparison exceptions propagate.
    """
    ordered = sorted(items, key=key)
    return ordered[::-1] if reverse_conventions(reverse) else ordered


ASSEMBLY_QUANTA: dict[str, tuple[float, float]] = {
    "refine_F": (2.0**-9, 2.0**-26),
    "join_ll": (2.0**-19, 2.0**-34),
    "join_ll_wide": (2.0**-19, 2.0**-30),
    "join_proxy": (2.0**-19, 2.0**-24),
    "beam": (2.0**-18, 2.0**-30),
    "count_margin": (2.0**-26, 0.0),
    "paint": (2.0**-30, 2.0**-30),
}
# Measured motion of compared differences (P0 EC30; 1e-9 relative GL noise, refine_F at 1e-6 since release
# 2.13), as (absolute, relative to the decision's scale): max(absolute, relative x |scale|) enables the boundary guard
# of tie_classes.
ASSEMBLY_MOTION: dict[str, tuple[float, float]] = {
    "refine_F": (9e-5, 5e-10),
    "join_ll": (1.3e-7, 2.7e-12),
    "join_ll_wide": (1.3e-7, 5.7e-11),
    "join_proxy": (1.1e-7, 2.1e-9),
    "beam": (6.5e-8, 0.0),
    "count_margin": (1.3e-10, 0.0),
    "paint": (0.0, 0.0),
}


def quantum(kind: str, scale: float = 0.0) -> float:
    """The tie quantum of an assembly decision kind at ``scale`` (a power of two).

    No mutation; an unknown kind raises KeyError and numeric conversion errors propagate.
    """
    floor, factor = ASSEMBLY_QUANTA[kind]
    target = max(floor, factor * abs(float(scale)) if math.isfinite(scale) else floor)
    return math.ldexp(1.0, int(math.ceil(math.log2(target))))


def motion(kind: str, scale: float = 0.0) -> float | None:
    """The measured noise motion of a decision kind's compared differences at ``scale`` (boundary guard).

    No mutation; an unknown kind raises KeyError and numeric conversion errors propagate.
    """
    value = ASSEMBLY_MOTION.get(kind)
    if value is None:
        return None
    absolute, relative = value
    scale = abs(float(scale)) if math.isfinite(scale) else 0.0
    return max(absolute, relative * scale) or None


def relevant_boundaries(
    values,
    ties_value: TieClasses,
    quantum_value: float,
    motion_value: float | None,
    *,
    descending: bool,
    cut: int | None = None,
) -> int:
    """Boundary-guard events of one decision: the gaps that decide it lying within BOUNDARY_GUARD noise motions of
    q -- for a winner the gap between the best class and the next, for a truncation that keeps ``cut`` items the
    gaps entering and leaving the class the cut falls in. (maths.ties reports every consecutive gap of the list;
    gaps elsewhere do not decide anything.)

    Inputs are unchanged; values and classes must share one element axis. NumPy/index errors propagate.
    """
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
    guard = BOUNDARY_GUARD * float(motion_value)
    events = 0
    for c in targets:
        if 0 <= c < count - 1:
            gap = low[c + 1] - high[c]
            if np.isfinite(gap) and abs(gap - quantum_value) <= guard:
                events += 1
    return events


def add_decision(
    counter,
    kind: str,
    ties_value: TieClasses,
    values,
    quantum_value: float,
    motion_value: float | None,
    *,
    descending: bool,
    cut: int | None = None,
) -> None:
    """Count one tie-class decision with its decision-relevant boundary events (see relevant_boundaries): a
    winner is inside a tie when its class has more than one member, a truncation when the cut falls inside a
    class (members on both sides of it).

        Mutates only counter (or no-ops when None); classification inputs are unchanged. Shape/index errors from
    relevant_boundaries propagate.
    """
    if counter is None:
        return
    if cut is None:
        inside = ties_value.tied_winner
    else:
        order = ties_value.order
        inside = bool(0 < cut < len(order) and ties_value.classes[order[cut - 1]] == ties_value.classes[order[cut]])
    boundary = relevant_boundaries(values, ties_value, quantum_value, motion_value, descending=descending, cut=cut)
    counter.add_counts(kind, 1, int(inside), int(ties_value.wide_classes), boundary)


def count_vector_classes(
    counter: TieCounter | None, kind: str, classes: np.ndarray, wide: int, boundary: Sequence, decided_inside: bool
) -> None:
    """Add one vectorised tie-class decision (path_class_order) to a TieCounter.

    Mutates only counter; classes is retained for the original calling interface. Count conversion errors propagate.
    """
    if counter is None:
        return
    counter.add_counts(kind, 1, int(bool(decided_inside)), int(wide), len(boundary))


def truncate(items: list, ties_value: TieClasses, limit: int, cap: int | None = None) -> list:
    """The first ``limit`` items in decision order, extended by the members of the class the cut falls in, so the
    whole cut class is kept (release 2.2: order-invariant). When the cut class would take the list beyond ``cap``
    items (default 2 x limit), the whole class is dropped instead and only the classes before it are kept (2.1 kept
    the class's first members in content order); a first class larger than the cap is kept whole. ``items`` are in
    the original indexing of ``ties_value``.

        Does not mutate items or tie arrays. Nonempty inputs require a positive limit and matching element axes;
    index errors propagate.
    """
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
    """A census decision key from parts (joined with '/').

    No mutation; string conversion errors from parts propagate.
    """
    return "/".join(str(part) for part in parts)


class SharedCounter(TieCounter):
    """Protect each counter update and immutable snapshot across a task's threads."""

    def __init__(self):
        super().__init__()
        import threading

        self._lock = threading.Lock()

    def add_counts(self, kind, decisions=0, inside_tie=0, wide=0, boundary=0):
        """Mutate this counter under its lock; conversion errors propagate to the caller."""
        with self._lock:
            super().add_counts(kind, decisions, inside_tie, wide, boundary)

    def record(self) -> TieCounts:
        """Return a frozen snapshot under the update lock without mutating any counts."""
        with self._lock:
            return super().record()


@dataclass(frozen=True)
class TiePolicy:
    """A named decision's quantum and arbitrary content convention, with no hidden state.

    Domain-specific content keys are supplied by the caller; secondary keys are priors
    and never reverse. Instances are immutable and can travel to process workers.
    """

    name: str
    kind: str
    reverse: bool = False
    fixed_quantum: float | None = None

    def quantum(self, scale: float = 0.0) -> float:
        """Return this decision's original power-of-two quantum at its magnitude."""
        if self.fixed_quantum is not None:
            return self.fixed_quantum
        return quantum(self.kind, scale) if self.kind in ASSEMBLY_QUANTA else tie_quantum(scale, self.kind)

    def motion(self, scale: float = 0.0) -> float | None:
        """Return the original noise motion used by the boundary guard."""
        if self.fixed_quantum is not None:
            return None
        return motion(self.kind, scale) if self.kind in ASSEMBLY_QUANTA else NOISE_MOTION[self.kind]

    def classes(
        self,
        values: Sequence[float] | np.ndarray,
        *,
        scale: float = 0.0,
        content: Sequence | None = None,
        secondary: Sequence | None = None,
        descending: bool = False,
    ) -> TieClasses:
        """Classify values with this policy; inputs are not mutated."""
        return tie_classes(
            values,
            self.quantum(scale),
            content=content,
            secondary=secondary,
            descending=descending,
            motion=self.motion(scale),
            reverse=self.reverse,
        )


# Semantic names are aliases of existing kinds, never new numerical tolerances.
_POLICY_KINDS = {
    "assembly.refine.accept": "refine_F",
    "assembly.join.panel": "join_ll",
    "assembly.join.wide": "join_ll_wide",
    "assembly.join.proxy": "join_proxy",
    "assembly.beam": "beam",
    "assembly.count.margin": "count_margin",
    "assembly.paint": "paint",
}


def policy(name: str, *, kind: str | None = None, reverse: bool = False) -> TiePolicy:
    """Bind a named decision to an existing kind; unknown kinds raise ValueError.

    The six shared names and seven assembly kinds work directly. A new semantic name
    requires an explicit kind, so a missing policy never silently picks a tolerance.

    No mutation. Unknown policy/kind raises ValueError; the caller documents content and secondary order.
    """
    selected = kind if kind is not None else _POLICY_KINDS.get(name, name)
    if selected not in QUANTUM_KINDS and selected not in ASSEMBLY_QUANTA:
        raise ValueError(f"unknown tie policy {name!r} (kind {selected!r})")
    return TiePolicy(name, selected, bool(reverse))


@dataclass(frozen=True)
class TieBand:
    """A named scalar width with no content order or reversible convention."""

    name: str
    width: float

    def quantum(self) -> float:
        """Return the caller's unchanged width; no mutation or numerical transformation."""
        return self.width


def fixed_band(name: str, quantum: float) -> TieBand:
    """Name an existing scalar band that does not choose among tied contents.

    For example, discovery release suppresses a within-band contribution before
    its fixed ALT probability test. That test has no reversible content order.
    The caller retains its native inequality and parameter provenance. No mutation;
    nonpositive or nonfinite widths raise ValueError exactly as fixed_policy does.
    """
    if not math.isfinite(quantum) or quantum <= 0:
        raise ValueError("a fixed tie quantum must be positive and finite")
    return TieBand(name, float(quantum))


def fixed_policy(name: str, quantum: float, *, reverse: bool) -> TiePolicy:
    """Bind a named single-linkage decision to its caller's original fixed quantum.

    The owning parameters module documents the constant and provenance. This does
    not convert anchored bands or incumbent preferences into class rules. No input
    mutation; nonpositive or nonfinite quanta raise ValueError. No motion guard is
    invented for a fixed quantum that had none in 3.5.
    """
    if not math.isfinite(quantum) or quantum <= 0:
        raise ValueError("a fixed tie quantum must be positive and finite")
    return TiePolicy(name, "fixed", bool(reverse), float(quantum))


__all__ = [
    "noise_quantum",
    "tie_quantum",
    "BlockQuanta",
    "block_scale",
    "TieClasses",
    "tie_classes",
    "TieCounter",
    "TieCount",
    "TieCounts",
    "TieCensus",
    "class_sort",
    "class_winner",
    "improves",
    "threshold_band",
    "run_sample_order",
    "inverse_permutation",
    "census_enabled",
    "census",
    "reverse_conventions",
    "rev",
    "conv_key",
    "conv_order",
    "conv_argmin",
    "conv_argmax",
    "conv_sorted",
    "quantum",
    "motion",
    "relevant_boundaries",
    "add_decision",
    "count_vector_classes",
    "truncate",
    "census_key",
    "SharedCounter",
    "TiePolicy",
    "TieSettings",
    "fixed_policy",
    "TieBand",
    "fixed_band",
    "policy",
    "GL_NOISE",
    "NOISE_MARGIN",
    "NOISE_SCALE",
    "NOISE_MOTION",
    "QUANTUM_KINDS",
    "WIDE_CLASS_QUANTA",
    "BOUNDARY_GUARD",
    "ASSEMBLY_QUANTA",
    "ASSEMBLY_MOTION",
]
