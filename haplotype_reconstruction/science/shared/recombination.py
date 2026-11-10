"""Define the single Haldane recombination fraction used throughout the science layer.

Owns the stable transform from nonnegative genetic distance to meiotic switch
probability; must not choose maps, thresholds or protected intervals. Method:
docs/methods/family_phase.md.
"""

from __future__ import annotations

import numpy as np
from numba import types

from ...core.kernels import kernel

__all__ = ["haldane_fraction"]


@kernel(
    overloads=(
        (types.float64,),
        (types.Array(types.float64, 1, "C"),),
        (types.Array(types.float64, 1, "C", readonly=True),),
    ),
    cache=True,
    inline="always",
    nogil=True,
)
def haldane_fraction(distance_morgans):
    """Return -expm1(-2M)/2 for float64 scalar/vector M in Morgans; mutate nothing.

    The domain is nonnegative distances. No clipping is imposed here; each model
    owns its probability bounds. This expression preserves short-interval accuracy.
    """
    return -0.5 * np.expm1(-2.0 * distance_morgans)
