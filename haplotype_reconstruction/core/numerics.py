"""Provide stable natural-log sums with explicit, worker-independent reduction order.

log_add relocates the live pedigree.family._logaddexp scalar primitive; log_sum
relocates pedigree.alternatives.logsumexp_finite on a contiguous float64 vector.
The unused maths/logspace module is not carried. No tolerance, model or search rule
is selected here. Callers own the scientific order of their input vector.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from .kernels import kernel

__all__ = [
    "log_add",
    "log_sum",
    "ordered_sum",
    "chromosome_sum",
    "weighted_chromosome_sum",
    "expansion_sum",
    "contiguous_axis_sum",
    "max_range",
]

# The live 3.5 pedigree primitive skips exp(-gap) only after float64 underflow.
_EXP_UNDERFLOW = 750.0


@kernel(overloads=((numba.float64, numba.float64),), cache=True, nogil=True, inline="always")
def log_add(a: float, b: float) -> float:
    """Return log(exp(a) + exp(b)) for finite/-inf nats, propagating NaN as in 3.5."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    gap = abs(a - b)
    return max(a, b) + (0.0 if gap > _EXP_UNDERFLOW else math.log1p(math.exp(-gap)))


def log_sum(values: np.ndarray) -> float:
    """Return log(sum(exp(values))) over a float64 vector, or -inf when empty.

    Non-finite maxima are returned unchanged, as in the pedigree scorer. A C-order
    copy fixes the reduction independently of a caller's array strides (ledger L2).
    Inputs are never mutated.
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    if len(values) == 0:
        return -np.inf
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def ordered_sum(values: np.ndarray) -> float:
    """Accurately sum float64 values in natural C index order, independent of batching.

    This is the shared fsum form already used for block scales and phase gains in
    3.5. Replacing a batch-dependent reduction with it is ledger L2; the caller must
    report any decision boundary crossed by that replacement.
    """
    return math.fsum(np.asarray(values, dtype=np.float64).ravel(order="C").tolist())


def chromosome_sum(values: np.ndarray) -> np.ndarray:
    """Fold float64[C,...] in configured contig order, returning float64[...].

    This is the pedigree pilot's declared L2 left fold, promoted unchanged. It is
    distinct from fsum: each output keeps the original ordered float64 additions.
    Callers bind contig order and report boundary flips when adopting it.
    """
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros(values.shape[1:], dtype=np.float64)
    for row in values:
        result += row
    return result


@kernel(
    overloads=tuple(
        (
            numba.types.Array(numba.float64, 1, "C", readonly=weights_readonly),
            numba.types.Array(numba.float64, 2, layout, readonly=values_readonly),
            numba.float64[::1],
        )
        for weights_readonly in (False, True)
        for layout in ("C", "F", "A")
        for values_readonly in (False, True)
    ),
    cache=True,
    nogil=True,
)
def _weighted_fold(weights, values, result):
    """result[m] = (...((0 + w0 v0m) + w1 v1m) ...): each product rounded, then added, as NumPy's row loop does."""
    for contig in range(values.shape[0]):
        weight = weights[contig]
        for item in range(values.shape[1]):
            product = weight * values[contig, item]
            result[item] = result[item] + product


def weighted_chromosome_sum(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fold weights[C] times values[C,...] in configured contig order (ledger L2).

    Zero weights still multiply their row, preserving IEEE non-finite behavior.
    This is the pedigree pilot's helper, independent of C/F strides. A two-axis input
    runs the same per-element products and additions in one compiled pass (no
    temporary per contig; ped40c PERF: 9 % of each resampling fit at N = 2,503).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2 and len(values):
        result = np.zeros(values.shape[1], dtype=np.float64)
        _weighted_fold(np.ascontiguousarray(weights, dtype=np.float64), values, result)
        return result
    result = np.zeros(values.shape[1:], dtype=np.float64)
    for index, row in enumerate(values):
        result += weights[index] * row
    return result


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
    negative, as in 3.5 discovery.residual_births._accurate_sum (P0 DI1/DI2).
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


def contiguous_axis_sum(values: np.ndarray) -> np.ndarray:
    """Reduce float64[C,n] along a contiguous C axis using NumPy's fixed blocking.

    This reproduces 3.5 selective_m1's advanced-indexing gather, including its
    pairwise rounding. It is dimension-fixed, not a left fold (WP_PE2 §4). The
    numerical-policy identity binds NumPy's version. Inputs are never mutated.
    """
    return np.add.reduce(np.asfortranarray(values, dtype=np.float64), axis=0)


@kernel(
    overloads=tuple(
        (numba.types.Array(dtype, 1, layout, readonly=readonly), numba.int64, numba.int64, numba.float64)
        for dtype in (numba.float32, numba.float64)
        for layout in ("C", "A")
        for readonly in (False, True)
    ),
    cache=True,
    inline="always",
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
