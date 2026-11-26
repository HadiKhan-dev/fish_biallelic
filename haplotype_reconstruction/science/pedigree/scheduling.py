"""Balance independent forward-chain work across static OpenMP ranges."""

import heapq

import numpy as np
from numba import types

from ...core.kernels import kernel
from .kernels.read_layout import _I64_1, _I64_2


@kernel(overloads=((_I64_1, types.int64),), cache=True, nogil=True)
def balanced_units(cost, threads):
    """Return unit ids grouped by thread and the offsets of those groups.

    OpenMP assigns contiguous slices of a prange. Deal longest work first to
    the least loaded group, then use one prange iteration per group. Costs
    affect only execution order. Callers pass their current team size outside
    the compiled function so these kernels can use Numba's disk cache.
    """
    groups = min(threads, len(cost))
    heap = [(np.int64(0), group) for group in range(groups)]
    assignment = np.empty(len(cost), np.int64)
    counts = np.zeros(groups, np.int64)
    order = np.argsort(cost)[::-1]
    for unit in order:
        load, group = heapq.heappop(heap)
        assignment[unit] = group
        counts[group] += 1
        heapq.heappush(heap, (load + cost[unit], group))
    offsets = np.empty(groups + 1, np.int64)
    offsets[0] = 0
    for group in range(groups):
        offsets[group + 1] = offsets[group] + counts[group]
        counts[group] = offsets[group]
    units = np.empty(len(cost), np.int64)
    for unit in order:
        group = assignment[unit]
        units[counts[group]] = unit
        counts[group] += 1
    return units, offsets


@kernel(overloads=((_I64_2, _I64_2, _I64_2),), cache=True, nogil=True)
def chain_work(site_start, site_stop, width):
    """A child's work grows with both observed emissions and carried class states."""
    bins = site_start.shape[1]
    cost = np.ones(site_start.shape[0], dtype=np.int64)
    if bins:
        for child in range(len(cost)):
            cost[child] += site_stop[child, bins - 1] - site_start[child, 0]
            for block in range(bins):
                cost[child] += width[child, block] * width[child, block + 1]
    return cost
