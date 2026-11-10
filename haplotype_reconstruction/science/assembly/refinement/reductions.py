"""Reduce per-sample scores accurately into the panel objective F = sum_s F_s.

The exact-rounded sum is independent of sample order, team size and strides.
Method: docs/methods/founder_refinement.md#objective (RS01/RS11).
"""

import numpy as np
from numpy.typing import NDArray

from haplotype_reconstruction.core.numerics import ordered_sum

__all__ = ["panel_objective"]


def panel_objective(scores: NDArray[np.float64]) -> float:
    """Return the nats objective of float64[N] scores; empty input sums to zero."""
    return ordered_sum(scores)
