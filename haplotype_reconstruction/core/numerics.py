"""Provide stable natural-log sums and correctly rounded sums independent of workers, layout and input order.

Scalar log addition and contiguous-vector log sums handle non-finite inputs
explicitly. ordered_sum and expansion_sum return math.fsum's correctly rounded value;
the axis form of ordered_sum also takes per-term weights. No tolerance, model or
search rule is selected here.
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

# Exact axis sums (_column_sums). Columns per block: four float64 running values per column,
# 16 KiB, stay in L1 cache while every row of the block streams through.
_COLUMN_BLOCK = 512
# Certify only columns whose sum of |terms| is at most 2**1000 (no partial sum of any order can
# overflow, so fsum cannot raise) and, when the error sum was itself rounded, whose rounded total is
# at least 2**-500 (the error bound and the float spacing are normal numbers). Other columns take the
# fsum route; these are scale limits, not tolerances, and change no value.
_CERTIFIED_MAXIMUM = 2.0**1000
_CERTIFIED_MINIMUM = 2.0**-500


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
def ordered_sum(values: np.ndarray, *, axis: int, weights: np.ndarray | None = None) -> np.ndarray: ...


def ordered_sum(
    values: np.ndarray, *, axis: int | None = None, weights: np.ndarray | None = None
) -> float | np.ndarray:
    """Return the correctly rounded sum of float64 values: math.fsum's value, independent of input order.

    With axis=None, return one scalar (math.fsum). Otherwise reduce that axis and retain the
    remaining axes; ``weights`` (float64, one per position along ``axis``) then multiply each
    term first, so a term is the rounded product weights[i] * value (a zero weight still
    multiplies: 0 * inf is NaN) and no weighted copy of ``values`` is made. Empty reductions
    return positive zero; non-finite values and intermediate overflow follow math.fsum.
    Inputs are never mutated.
    """
    values = np.asarray(values, dtype=np.float64)
    if axis is None:
        return math.fsum(values.ravel(order="C").tolist())
    columns = np.moveaxis(values, axis, 0)
    shape = columns.shape[1:]
    terms = columns.shape[0]
    columns = columns.reshape(terms, math.prod(shape))
    if weights is None:
        weights = np.ones(terms, dtype=np.float64)
    else:
        weights = np.array(weights, dtype=np.float64).reshape(-1)
        if len(weights) != terms:
            raise ValueError(f"{len(weights)} weights for an axis of {terms} terms")
    return _column_sums(columns, weights).reshape(shape)


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
    overloads=((numba.float64[::1], numba.float64[::1]),),
    cache=True,
    nogil=True,
    inline="always",
)
def _fsum_row(row, partials):
    """math.fsum of one contiguous row, including its special values; ``partials`` is scratch of len(row).

    The finite path uses expansion_sum unchanged. An exceptional row is split at
    its non-finite values: fsum resets its finite partials there, but still raises
    on intermediate overflow in any finite segment or on opposing infinities.
    """
    total = expansion_sum(row, partials)
    if math.isfinite(total):
        # fsum starts at +0.0, whereas expansion_sum preserves a single -0.0.
        return total if total != 0.0 else 0.0
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
    return special_sum


@kernel(
    overloads=tuple(
        (numba.types.Array(numba.float64, 2, layout, readonly=readonly), numba.float64[::1])
        for layout in ("C", "F", "A")
        for readonly in (False, True)
    ),
    cache=True,
    nogil=True,
)
def _column_sums(values, weights):
    """float64 [J]: for every column j of ``values`` [I, J], math.fsum of the rounded products
    weights[i] * values[i, j] over i: the correctly rounded value of their exact sum.

    Columns are summed in blocks whose running values stay in cache while the I rows stream
    through, with no weighted copy. Each column carries an error-free (TwoSum) running sum s,
    the TwoSum sum sigma of s's exact addition errors, and the float sum D of the |errors| of
    sigma itself, so that the exact total is s + sigma + d with |d| <= 2 D. Then r = s + sigma
    (rounded) is the correctly rounded total, hence fsum's value: always when D = 0 (s + sigma
    is the exact total), and otherwise when |e| + 2 D stays below half the float spacing around
    r (the smaller spacing when |r| is a power of two), e being the exact error of r. A column
    that does not certify (r = 0, non-finite or extreme magnitudes, a total too close to a
    rounding boundary) is summed by _fsum_row in its original order, so special values,
    exceptions and every finite result are exactly fsum's.
    """
    terms, n_columns = values.shape
    result = np.empty(n_columns, dtype=np.float64)
    running = np.empty(_COLUMN_BLOCK, dtype=np.float64)
    errors = np.empty(_COLUMN_BLOCK, dtype=np.float64)
    inexact = np.empty(_COLUMN_BLOCK, dtype=np.float64)
    magnitude = np.empty(_COLUMN_BLOCK, dtype=np.float64)
    row = np.empty(terms, dtype=np.float64)
    partials = np.empty(terms, dtype=np.float64)
    for start in range(0, n_columns, _COLUMN_BLOCK):
        width = min(_COLUMN_BLOCK, n_columns - start)
        running[:width] = 0.0
        errors[:width] = 0.0
        inexact[:width] = 0.0
        magnitude[:width] = 0.0
        for i in range(terms):
            weight = weights[i]
            line = values[i, start : start + width]
            for b in range(width):
                term = weight * line[b]
                before = running[b]
                total = before + term
                virtual = total - before
                error = (before - (total - virtual)) + (term - virtual)
                running[b] = total
                low = errors[b]
                low_total = low + error
                low_virtual = low_total - low
                inexact[b] += abs((low - (low_total - low_virtual)) + (error - low_virtual))
                errors[b] = low_total
                magnitude[b] += abs(term)
        for b in range(width):
            size = magnitude[b]
            if size == 0.0:
                # Every term is a signed zero; fsum returns +0.0.
                result[start + b] = 0.0
                continue
            if size <= _CERTIFIED_MAXIMUM:
                total = running[b]
                low = errors[b]
                rounded = total + low
                missing = inexact[b]
                if missing == 0.0:
                    if rounded != 0.0:
                        result[start + b] = rounded
                        continue
                elif abs(rounded) >= _CERTIFIED_MINIMUM:
                    virtual = rounded - total
                    residual = (total - (rounded - virtual)) + (low - virtual)
                    mantissa, exponent = math.frexp(abs(rounded))
                    half_spacing = math.ldexp(1.0, exponent - (55 if mantissa == 0.5 else 54))
                    bound = 2.0 * missing
                    if bound < 0.5 * half_spacing and half_spacing - abs(residual) > bound:
                        result[start + b] = rounded
                        continue
            for i in range(terms):
                row[i] = weights[i] * values[i, start + b]
            result[start + b] = _fsum_row(row, partials)
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
