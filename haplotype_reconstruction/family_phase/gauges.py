"""MAP selector paths of children's joint selector chains (root-frame repair, frames.py).

Release 2.1: the joint root/children relabelling moves of 2.0's belief propagation became exact moves of the
family MAP solver (map_solver.py); this module keeps the selector-path decoding the repair's votes use.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..maths import chains


@njit(cache=True, parallel=True)
def _selector_paths(emissions, theta, groups, result, reverse):
    for job in prange(len(groups)):
        group = groups[job]
        result[group] = chains.selector_viterbi(emissions[group], theta, chains.CHAIN_TIE, reverse)


def selector_paths(emissions, theta, groups, reverse=False):
    """MAP selector paths int8 [G, L] of ``groups`` (other rows are not written and must not be read). ``reverse``:
    the convention reversal of chains.selector_viterbi (maths.ties.reverse_conventions)."""
    paths = np.empty(emissions.shape[:2], dtype=np.int8)
    if len(groups):
        _selector_paths(emissions, theta, np.asarray(groups, dtype=np.int64), paths, bool(reverse))
    return paths
