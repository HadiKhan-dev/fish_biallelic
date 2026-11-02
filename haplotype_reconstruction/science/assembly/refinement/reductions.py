"""Reduce per-sample exact scores into the panel objective in one fixed order.

Owns only F = sum_s F_s, in original sample order. Must not reorder samples or let
team size, strides or a memory budget choose the reduction. Method:
docs/methods/founder_refinement.md#objective (RS01/RS11).

The C1 contiguous-axis reduction fixes NumPy's pairwise order from dimensions.
The N-by-1 view preserves the original sample-vector sum, including empty inputs.
"""

import numpy as np
from numpy.typing import NDArray

from haplotype_reconstruction.core.numerics import contiguous_axis_sum

__all__ = ["panel_objective"]


def panel_objective(scores: NDArray[np.float64]) -> float:
    """Return the nats objective of float64[N] sample scores without mutating them.

    A private contiguous view normalizes layout before NumPy's declared pairwise
    reduction. The sample axis keeps its original order; empty input sums to zero.
    """
    return float(contiguous_axis_sum(np.asarray(scores, np.float64).reshape(-1, 1))[0])
