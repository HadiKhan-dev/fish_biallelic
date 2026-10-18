"""Log-space sums inside Numba kernels: log(exp(a) + exp(b)) without overflow (inlined into the calling kernel).

An -inf operand returns the other one unchanged; NaN propagates.
"""
from __future__ import annotations

import math

import numba


@numba.njit(cache=True, inline="always")
def logaddexp(a, b):
    """log(exp(a) + exp(b)) for two floats."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))

