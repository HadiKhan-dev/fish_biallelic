"""Enumerate unordered sample pairs for family fitting and pairwise IBD."""

import numba
import numpy as np

from ...core.kernels import kernel


@kernel(
    overloads=(
        (
            numba.int64,
            numba.int64,
            numba.int64,
        ),
    ),
    cache=True,
    nogil=True,
)
def pair_indices(n, lo, hi):
    """Unordered sample pairs [lo, hi) in row-major upper-triangle order, without triu temporaries."""
    out = np.empty((hi - lo, 2), dtype=np.int64)
    f, offset = 0, 0
    while f < n - 1 and offset + n - f - 1 <= lo:
        offset += n - f - 1
        f += 1
    g = f + 1 + lo - offset
    for t in range(hi - lo):
        out[t, 0], out[t, 1] = f, g
        g += 1
        if g == n:
            f += 1
            g = f + 1
    return out
