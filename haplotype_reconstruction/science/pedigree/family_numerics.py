"""Log-space arithmetic and lane sizing shared by pedigree family fits."""

import math
import numpy as np

from ...core.kernels import kernel
from .kernels.decision import OVERLOADS
from .numerical_parameters import FAMILY_EXP_UNDERFLOW as _EXP_UNDERFLOW
from .workspace_parameters import FAMILY_LANES_PER_THREAD

_LANES_PER_THREAD = FAMILY_LANES_PER_THREAD
# exp(x) is exactly +0.0 in float64 for every x < -745.14 (below half the smallest subnormal), and log(1) and log1p(0)
# are exactly 0: the kernels skip those evaluations, whose results they already know, so every value is unchanged
# (bit for bit) while far-below-maximum rows, which dominate label-free panels at large N, cost no transcendental.


@kernel(overloads=OVERLOADS["family_numerics._exp_or_zero"], cache=True, nogil=True, inline="always")
def _exp_or_zero(x):
    """math.exp(x), skipped where it is exactly zero (x < -750; NaN still evaluates)."""
    return 0.0 if x < -_EXP_UNDERFLOW else math.exp(x)


@kernel(overloads=OVERLOADS["family_numerics._log_or_zero"], cache=True, nogil=True, inline="always")
def _log_or_zero(x):
    """math.log(x), skipped where it is exactly zero (x == 1)."""
    return 0.0 if x == 1.0 else math.log(x)


@kernel(overloads=OVERLOADS["family_numerics._softplus_tail"], cache=True, nogil=True, inline="always")
def _softplus_tail(magnitude):
    """np.log1p(np.exp(-magnitude)) for magnitude >= 0, skipped where it is exactly zero (magnitude > 750)."""
    return 0.0 if magnitude > _EXP_UNDERFLOW else np.log1p(np.exp(-magnitude))


@kernel(overloads=OVERLOADS["family_numerics._lane_count"], cache=True, nogil=True)
def _lane_count(items, threads):
    return max(1, min(items, _LANES_PER_THREAD * threads))
