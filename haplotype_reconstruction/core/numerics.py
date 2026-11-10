"""Provide stable natural-log sums with explicit, worker-independent reduction order.

Scalar log addition and contiguous-vector log sums handle non-finite inputs
explicitly. No tolerance, model or search rule is selected here. Callers own the
scientific order of their input vector.
"""

from __future__ import annotations

import math
from typing import overload

import numba
import numpy as np

from .kernels import kernel

__all__ = [
    "log_add",
    "log_sum",
    "ordered_sum",
    "expansion_sum",
    "max_range",
]

# Skip exp(-gap) only beyond float64 underflow; the gap is in natural-log units.
_EXP_UNDERFLOW = 750.0


@kernel(overloads=((numba.float64, numba.float64),), cache=True, nogil=True, inline="always")
def log_add(a: float, b: float) -> float:
    """Return log(exp(a) + exp(b)) for finite/-inf nats, propagating NaN."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    gap = abs(a - b)
    return max(a, b) + (0.0 if gap > _EXP_UNDERFLOW else math.log1p(math.exp(-gap)))


def log_sum(values: np.ndarray) -> float:
    """Return log(sum(exp(values))) over a float64 vector, or -inf when empty.

    Non-finite maxima are returned unchanged, as in the pedigree scorer. A C-order
    copy fixes the reduction independently of a caller's array strides.
    Inputs are never mutated.
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    if len(values) == 0:
        return -np.inf
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


@overload
def ordered_sum(values: np.ndarray, *, axis: None = None) -> float: ...


@overload
def ordered_sum(values: np.ndarray, *, axis: int) -> np.ndarray: ...


def ordered_sum(values: np.ndarray, *, axis: int | None = None) -> float | np.ndarray:
    """Accurately sum float64 values, retaining contributions lost by ordinary addition.

    With axis=None, return one scalar; otherwise reduce that axis and retain the
    remaining axes. Scalar sums use math.fsum; axis sums use its expansion algorithm
    in a compiled loop with shared scratch, independently of layout or batching.
    Finite sums are independent of input order. Empty reductions return positive
    zero; non-finite values and intermediate overflow follow math.fsum. Inputs are
    never mutated.
    """
    values = np.asarray(values, dtype=np.float64)
    if axis is None:
        return math.fsum(values.ravel(order="C").tolist())
    rows = np.moveaxis(values, axis, -1)
    shape = rows.shape[:-1]
    count = math.prod(shape)
    rows = rows.reshape(count, rows.shape[-1])
    return _axis_sum(rows).reshape(shape)


@kernel(
    overloads=tuple(
        (numba.types.Array(numba.float64, 1, layout, readonly=readonly), scratch)
        for layout in ("C", "A")
        for readonly in (False, True)
        for scratch in (numba.types.Omitted(None), numba.types.none, numba.float64[::1])
    ),
    cache=True,
    nogil=True,
)
def expansion_sum(values, partials=None):
    """Finite-value expansion sum (the fsum reference, including halfway correction).

    The partials retain low bits discarded by ordinary addition. This makes population and
    cluster reductions independent of sample order, including cancellation of log odds.
    Values are finite float64; they are not mutated. Optional float64 scratch must
    have at least len(values) elements and is overwritten. A single -0.0 stays
    negative.
    """
    if partials is None:
        partials = np.empty(len(values), dtype=np.float64)
    n = 0
    for value in values:
        x = value
        i = 0
        for j in range(n):
            y = partials[j]
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo != 0.0:
                partials[i] = lo
                i += 1
            x = hi
        partials[i] = x
        n = i + 1
    hi = 0.0
    lo = 0.0
    if n:
        n -= 1
        hi = partials[n]
        while n:
            x = hi
            n -= 1
            y = partials[n]
            hi = x + y
            lo = y - (hi - x)
            if lo != 0.0:
                break
        if n and ((lo < 0.0 and partials[n - 1] < 0.0) or (lo > 0.0 and partials[n - 1] > 0.0)):
            y = lo * 2.0
            x = hi + y
            if y == x - hi:
                hi = x
    return hi


@kernel(
    overloads=tuple(
        (numba.types.Array(numba.float64, 2, layout, readonly=readonly),)
        for layout in ("C", "F", "A")
        for readonly in (False, True)
    ),
    cache=True,
    nogil=True,
)
def _axis_sum(rows):
    """Exactly sum each row, reusing expansion scratch; match math.fsum's special values.

    The finite path uses expansion_sum unchanged. An exceptional row is split at
    its non-finite values: fsum resets its finite partials there, but still raises
    on intermediate overflow in any finite segment or on opposing infinities.
    """
    result = np.empty(rows.shape[0], dtype=np.float64)
    partials = np.empty(rows.shape[1], dtype=np.float64)
    for item in range(rows.shape[0]):
        row = rows[item]
        total = expansion_sum(row, partials)
        if math.isfinite(total):
            # fsum starts at +0.0, whereas expansion_sum preserves a single -0.0.
            result[item] = total if total != 0.0 else 0.0
            continue
        special_sum = 0.0
        inf_sum = 0.0
        start = 0
        for index in range(len(row)):
            value = row[index]
            if not math.isfinite(value):
                if not math.isfinite(expansion_sum(row[start:index], partials)):
                    raise OverflowError("intermediate overflow in fsum")
                special_sum += value
                if math.isinf(value):
                    inf_sum += value
                start = index + 1
        if not math.isfinite(expansion_sum(row[start:], partials)):
            raise OverflowError("intermediate overflow in fsum")
        if math.isnan(inf_sum):
            raise ValueError("-inf + inf in fsum")
        result[item] = special_sum
    return result


@kernel(
    overloads=tuple(
        (numba.types.Array(dtype, 1, layout, readonly=readonly), numba.int64, numba.int64, numba.float64)
        for dtype in (numba.float32, numba.float64)
        for layout in ("C", "A")
        for readonly in (False, True)
    ),
    cache=True,
    inline="always",
    nogil=True,
)
def max_range(values, lo, hi, initial):
    """max(initial, values[lo:hi]) through eight independent running maxima (two below eight
    values). A maximum is exact, so the value equals one sequential scan's; the independent chains
    let comparisons overlap."""
    if hi - lo < 8:
        m0 = initial
        m1 = -np.inf
        z = lo
        while z + 2 <= hi:
            v = values[z]
            m0 = v if v > m0 else m0
            v = values[z + 1]
            m1 = v if v > m1 else m1
            z += 2
        if z < hi:
            v = values[z]
            m0 = v if v > m0 else m0
        return m1 if m1 > m0 else m0
    m0 = initial
    m1 = m2 = m3 = m4 = m5 = m6 = m7 = -np.inf
    z = lo
    while z + 8 <= hi:
        v = values[z]
        m0 = v if v > m0 else m0
        v = values[z + 1]
        m1 = v if v > m1 else m1
        v = values[z + 2]
        m2 = v if v > m2 else m2
        v = values[z + 3]
        m3 = v if v > m3 else m3
        v = values[z + 4]
        m4 = v if v > m4 else m4
        v = values[z + 5]
        m5 = v if v > m5 else m5
        v = values[z + 6]
        m6 = v if v > m6 else m6
        v = values[z + 7]
        m7 = v if v > m7 else m7
        z += 8
    while z < hi:
        v = values[z]
        m0 = v if v > m0 else m0
        z += 1
    m0 = m1 if m1 > m0 else m0
    m2 = m3 if m3 > m2 else m2
    m4 = m5 if m5 > m4 else m4
    m6 = m7 if m7 > m6 else m6
    m0 = m2 if m2 > m0 else m0
    m4 = m6 if m6 > m4 else m4
    return m4 if m4 > m0 else m0
