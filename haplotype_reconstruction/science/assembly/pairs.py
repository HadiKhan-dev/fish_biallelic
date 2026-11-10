"""The three distinct pair orders used by assembly (AJ03, AJ04, AJ19).

Packed emission pairs include the diagonal, ordered by their larger row. Objective
states include the diagonal, ordered by their first row. Splice columns exclude
the diagonal. None of these indices may be substituted for another.

Owns: the three independent emission, objective-state and splice pair orders.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: docs/methods/assembly.md#aj03.
Invariant: These orders are deliberately different and cannot be substituted for one another.
"""

import numpy as np

from haplotype_reconstruction.core.kernels import kernel

from .join.kernels import overloads_for


@kernel(overloads=overloads_for(__name__, "emission_pair_index"), cache=True, inline="always", nogil=True)
def emission_pair_index(first: int, second: int) -> int:
    """Packed triangular emission index, valid for nonnegative local row indices."""
    high = max(first, second)
    return high * (high + 1) // 2 + min(first, second)


@kernel(overloads=overloads_for(__name__, "state_pairs"), cache=True, nogil=True)
def state_pairs(founders: int, allowed=None):
    """Two int64[Q] arrays in nested first <= second order.

    Optional bool[K] allowed mask excludes a path without renumbering its slot.
    Counting and filling both visit the objective states in nested pair order.
    """
    if allowed is None:
        allowed = np.ones(founders, np.bool_)
    count = 0
    for a in range(founders):
        if allowed[a]:
            for b in range(a, founders):
                if allowed[b]:
                    count += 1
    first = np.empty(count, np.int64)
    second = np.empty(count, np.int64)
    index = 0
    for a in range(founders):
        if allowed[a]:
            for b in range(a, founders):
                if allowed[b]:
                    first[index] = a
                    second[index] = b
                    index += 1
    return first, second


@kernel(overloads=overloads_for(__name__, "splice_column"), cache=True, inline="always", nogil=True)
def splice_column(first: int, second: int, founders: int) -> int:
    """Strict first < second triangular column of the K(K-1)/2 splice table."""
    return first * (2 * founders - first - 1) // 2 + second - first - 1


__all__ = ["emission_pair_index", "state_pairs", "splice_column"]
