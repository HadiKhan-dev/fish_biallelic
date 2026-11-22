"""The run's outlier rate before the calibration barrier, from the pilot's strict fits.

Outlier rate (ADVICE2 section 8 and its notes of 08:45 and 09:12). The rate is the one-rate maximum
likelihood over every known-state cell of the pilot blocks' STRICT fits (fits at the numerical lower bound of
the rate, so that no fit tolerates the discordance it is asked to measure), at least 0.5 / cells. The two-level
mixture below (units "explained" at eps or "unexplained" at a larger rate) is published beside it as a diagnostic:
it is not identified once the panel is good.

    maximise  sum_u log[(1 - pi) L_u(eps) + pi L_u(eps_bad)]   over eps < eps_bad on the grid and pi in (0, 1)

(for each pair of grid rates pi is the maximiser of a concave function, found by bisection of its derivative; pairs
are scanned in parallel). Ties go to the smaller rate.

"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import prange, types

from ...core.kernels import kernel
from ..shared.observation import refined_maximum

__all__ = ["OutlierEstimate", "two_level_rate", "rate_grid", "refined_maximum"]


def rate_grid(low: float, high: float, points: int) -> np.ndarray:
    """float64[points] log-spaced rates from ``low`` to ``high``."""
    return np.exp(np.linspace(math.log(low), math.log(high), int(points)))


@kernel(
    overloads=((types.Array(types.float64, 2, "C", readonly=True), types.int64, types.int64, types.float64),),
    cache=True,
    inline="always",
    nogil=True,
)
def _share_derivative(units, i, j, pi):
    """d/dpi of sum_u log[(1 - pi) L_u(rate i) + pi L_u(rate j)] (overflow-safe in the log-likelihood ratio)."""
    total = 0.0
    for u in range(units.shape[0]):
        t = units[u, j] - units[u, i]
        if t > 30.0:
            e = math.exp(-t)
            total += (1.0 - e) / (pi + (1.0 - pi) * e)
        else:
            x = math.exp(t)
            total += (x - 1.0) / ((1.0 - pi) + pi * x)
    return total


@kernel(
    overloads=(
        (
            types.Array(types.float64, 2, "C", readonly=True),
            types.Array(types.int64, 1, "C", readonly=True),
            types.Array(types.int64, 1, "C", readonly=True),
            types.Array(types.float64, 1, "C"),
            types.Array(types.float64, 1, "C"),
        ),
    ),
    cache=True,
    parallel=True,
    nogil=True,
)
def _scan_pairs(units, first, second, log_likelihood, share):
    """For each grid pair (first[k] < second[k]): the maximised two-level log likelihood and its share pi."""
    n_units = units.shape[0]
    for k in prange(first.shape[0]):
        i = first[k]
        j = second[k]
        low = 1e-12
        high = 1.0 - 1e-12
        if _share_derivative(units, i, j, low) <= 0.0:
            pi = low
        elif _share_derivative(units, i, j, high) >= 0.0:
            pi = high
        else:
            for _ in range(80):
                mid = 0.5 * (low + high)
                if _share_derivative(units, i, j, mid) > 0.0:
                    low = mid
                else:
                    high = mid
            pi = 0.5 * (low + high)
        total = 0.0
        for u in range(n_units):
            a = units[u, i] + math.log1p(-pi)
            b = units[u, j] + math.log(pi)
            m = a if a > b else b
            total += m + math.log(math.exp(a - m) + math.exp(b - m))
        log_likelihood[k] = total
        share[k] = pi


@dataclass(frozen=True)
class OutlierEstimate:
    """The two-level estimate and what it rests on."""

    outlier_rate: float  # eps of explained units, at least floor
    unexplained_rate: float  # eps_bad
    unexplained_share: float  # pi
    one_rate: float  # the one-rate maximum on the grid
    floor: float  # 0.5 / cells
    cells: int
    units: int
    gain: float  # two-level over one-rate log likelihood (nats)
    surface: tuple[float, ...]  # one-rate log likelihood on the grid (relative to its maximum)


def two_level_rate(units: np.ndarray, cells: int, grid: np.ndarray, fallback: float) -> OutlierEstimate:
    """The two-level estimate (module docstring) from unit log likelihoods ``units`` float64[M, G] on ``grid``;
    ``cells`` the observed cells of known states they sum over. Without any unit (a run too small or sparse for the
    pilot) the estimate is ``fallback`` (the pilot's declared start rate), published with zero units. The floor
    0.5 / cells is a regularising lower bound against a zero estimate, not a confidence bound."""
    units = np.ascontiguousarray(units, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    floor = 0.5 / max(int(cells), 1)
    if units.shape[0] == 0:
        return OutlierEstimate(float(fallback), float(fallback), 0.0, float(fallback), floor, 0, 0, 0.0,
                               tuple(0.0 for _ in grid))
    single = units.sum(axis=0)
    j1 = int(np.argmax(single))  # first maximum: the smaller rate
    first, second = np.triu_indices(len(grid), k=1)
    first = np.ascontiguousarray(first, dtype=np.int64)
    second = np.ascontiguousarray(second, dtype=np.int64)
    log_likelihood = np.empty(len(first))
    share = np.empty(len(first))
    frozen = units.copy()
    frozen.flags.writeable = False
    first.flags.writeable = False
    second.flags.writeable = False
    _scan_pairs(frozen, first, second, log_likelihood, share)
    best = int(np.argmax(log_likelihood))  # pairs ascend in (eps, eps_bad): the first maximum has the smaller rates
    if log_likelihood[best] <= single[j1]:
        rate, bad, pi, gain = float(grid[j1]), float(grid[j1]), 0.0, 0.0
    else:
        rate, bad, pi = float(grid[first[best]]), float(grid[second[best]]), float(share[best])
        gain = float(log_likelihood[best] - single[j1])
    return OutlierEstimate(
        max(rate, floor), bad, pi, float(grid[j1]), floor, int(cells), int(units.shape[0]), gain,
        tuple(float(v) for v in single - single[j1]),
    )
