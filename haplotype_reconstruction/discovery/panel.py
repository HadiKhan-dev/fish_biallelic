"""Fixed-K solutions of a block and their one canonical form.

A *panel* is K distinct binary rows (founder haplotypes at the block's discovery sites). A
:class:`FixedKSolution` is a panel with each sample's unordered row pair (the unknown copy is
row index K), the per-sample cost and the binned switching-Viterbi NLL of its fit.

Canonical form: rows in lexicographic order of their 0/1 sequences (the order of the rows'
int8 bytes compared as unsigned bytes; equal rows keep their input order), assignments remapped
to that order with each pair sorted ascending. The canonical key is the rows' int8 bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from numba import njit

from ..maths.ties import BlockQuanta, class_sort

# Release 2.1: NLL comparisons use the block's ``nll`` tie quantum (BlockQuanta.nll; the old absolute
# NLL_TOLERANCE = 1e-10 sat at the 1e-9-noise scale). Duplicates keep the first fit seen unless the
# other is better by more than the quantum.


@dataclass(frozen=True)
class FixedKSolution:
    """A fitted fixed-K factorisation in canonical form (arrays read-only).

    haplotypes: int64[K, L] binary rows, canonical order.
    assignments: int64[N, 2] sorted row pairs, K = unknown copy.
    per_sample_cost: float64[N] per-sample binned switching-Viterbi NLL.
    unknown_slots: int64[N] unknown copies in the sample's pair (0, 1, 2).
    iterations: coordinate-descent iterations the fit reported.
    nll: total NLL (sum of per_sample_cost, as the fitter reported it).
    fixed_point: the fitter proved haplotypes an A(H), H(A) fixed point in this workspace.
    key: canonical key (int8 bytes of the rows).
    quanta: the block's BlockQuanta the solution was fitted under (None for solutions built outside a
        workspace); orders of solutions use tie classes of kind ``nll`` with them (maths.ties).
    """

    haplotypes: np.ndarray
    assignments: np.ndarray
    per_sample_cost: np.ndarray
    unknown_slots: np.ndarray
    iterations: int
    nll: float
    fixed_point: bool = field(compare=False)
    key: bytes = field(repr=False)
    quanta: BlockQuanta | None = field(default=None, compare=False, repr=False)

    @property
    def k(self) -> int:
        return int(self.haplotypes.shape[0])

    @property
    def sites(self) -> int:
        return int(self.haplotypes.shape[1])

    def order(self) -> tuple[float, int, bytes]:
        """Exact (NLL, K, key). Tie-aware orders of several solutions use :func:`sort_solutions`
        (tie classes of kind ``nll``), not this tuple."""
        return (self.nll, self.k, self.key)


def _readonly(array):
    array.setflags(write=False)
    return array


@njit(cache=True, nogil=True)
def _canonical_panel_kernel(haplotypes, assignments):
    """Stable insertion sort of the rows by their int8 bytes compared as unsigned values
    (NumPy's stable sort order of the rows' void keys) and the remapped, sorted assignments."""
    k = haplotypes.shape[0]
    n_sites = haplotypes.shape[1]
    byte_rows = np.empty((k, n_sites), dtype=np.int8)
    for row in range(k):
        for site in range(n_sites):
            byte_rows[row, site] = np.int8(haplotypes[row, site] & 0xFF)
    order = np.arange(k)
    for index in range(1, k):
        current = order[index]
        position = index
        while position > 0:
            previous = order[position - 1]
            less = False
            for site in range(n_sites):
                a = np.uint8(byte_rows[current, site])
                b = np.uint8(byte_rows[previous, site])
                if a != b:
                    less = a < b
                    break
            if not less:
                break
            order[position] = previous
            position -= 1
        order[position] = current
    inverse = np.empty(k, dtype=np.int64)
    for index in range(k):
        inverse[order[index]] = index
    canonical_assignments = np.empty((assignments.shape[0], 2), dtype=np.int64)
    for sample in range(assignments.shape[0]):
        first = assignments[sample, 0]
        second = assignments[sample, 1]
        first = inverse[first] if first < k else k
        second = inverse[second] if second < k else k
        if second < first:
            first, second = second, first
        canonical_assignments[sample, 0] = first
        canonical_assignments[sample, 1] = second
    canonical_haplotypes = np.empty((k, n_sites), dtype=np.int64)
    canonical_bytes = np.empty((k, n_sites), dtype=np.int8)
    for index in range(k):
        source = order[index]
        for site in range(n_sites):
            canonical_haplotypes[index, site] = haplotypes[source, site]
            canonical_bytes[index, site] = byte_rows[source, site]
    return canonical_haplotypes, canonical_assignments, canonical_bytes


def canonical_solution(fit, workspace, quanta=None) -> FixedKSolution:
    """The canonical solution of a fit ``(H, A, per_sample_cost, unknown_slots, iterations, nll)``.

    ``fixed_point`` is set when ``workspace`` certifies the canonical rows (in that exact order)
    as a fixed point of its coordinate map; None means never certified. ``quanta``: the block's
    BlockQuanta (default: the workspace's when given).
    """
    if quanta is None and workspace is not None:
        quanta = workspace.quanta
    haplotypes, assignments, key_bytes = _canonical_panel_kernel(
        np.ascontiguousarray(fit[0], dtype=np.int64), np.ascontiguousarray(fit[1], dtype=np.int64))
    certified = bool(workspace is not None and workspace.certifies_fixed_point(haplotypes))
    return FixedKSolution(
        haplotypes=_readonly(haplotypes), assignments=_readonly(assignments),
        per_sample_cost=fit[2], unknown_slots=fit[3], iterations=int(fit[4]),
        nll=float(fit[5]), fixed_point=certified, key=key_bytes.tobytes(), quanta=quanta)


def restricted(solution: FixedKSolution, samples: np.ndarray) -> FixedKSolution:
    """The same panel with only the ``samples`` rows of the sample arrays (nll re-summed)."""
    cost = solution.per_sample_cost[samples]
    return FixedKSolution(
        haplotypes=solution.haplotypes, assignments=solution.assignments[samples],
        per_sample_cost=cost, unknown_slots=solution.unknown_slots[samples],
        iterations=solution.iterations, nll=math.fsum(cost.tolist()),
        fixed_point=solution.fixed_point, key=solution.key, quanta=solution.quanta)


def has_distinct_rows(solution: FixedKSolution) -> bool:
    """Whether the solution's K rows are pairwise distinct (a complete K panel)."""
    width = solution.sites
    key = solution.key
    if width == 0:
        return solution.k == 1
    return len({key[start:start + width] for start in range(0, len(key), width)}) == solution.k


def _nll_quantum(solutions, quanta):
    if quanta is not None:
        return quanta.nll
    for solution in solutions:
        if solution.quanta is not None:
            return solution.quanta.nll
    return BlockQuanta.floors().nll


def sort_solutions(solutions, quanta=None, *, by_k=True, counter=None) -> list:
    """Solutions in tie-class order of their NLLs (kind ``nll``; quanta: the block's, default the
    solutions' own), then smaller K, then the canonical key."""
    solutions = list(solutions)
    q = _nll_quantum(solutions, quanta)
    return class_sort(solutions, lambda s: s.nll, q, secondary=(lambda s: (s.k,)) if by_k else None,
                      content=lambda s: s.key, counter=counter, kind="nll_order")


def deduplicate(solutions, beam_width: int | None = None, *, quanta=None) -> tuple[FixedKSolution, ...]:
    """Distinct complete panels (by key), each with its lowest-NLL fit (a later fit replaces an
    earlier one only when better by more than the ``nll`` quantum), in tie-class order of NLL, then
    smaller K, then key (sort_solutions; 2.0's (nll, K, key) order with tie classes), optionally the
    first beam_width (the cut falls inside a class by K and key)."""
    solutions = [solution for solution in solutions if has_distinct_rows(solution)]
    q = _nll_quantum(solutions, quanta)
    unique: dict[bytes, FixedKSolution] = {}
    for solution in solutions:
        previous = unique.get(solution.key)
        if previous is None or solution.nll < previous.nll - q:
            unique[solution.key] = solution
    ordered = sort_solutions(unique.values(), quanta)
    return tuple(ordered if beam_width is None else ordered[:beam_width])


def unique_rows(matrix: np.ndarray) -> np.ndarray:
    """Distinct binary rows in lexicographic order (first occurrence kept), C-contiguous."""
    rows = np.asarray(matrix)
    if len(rows) <= 1:
        return np.array(rows, copy=True, order="C")
    packed = np.packbits(rows, axis=1, bitorder="big")
    first_index_by_key: dict[bytes, int] = {}
    for index, packed_row in enumerate(packed):
        first_index_by_key.setdefault(packed_row.tobytes(), index)
    indices = [first_index_by_key[key] for key in sorted(first_index_by_key)]
    return np.ascontiguousarray(rows[indices])


def panel_key(haplotypes: np.ndarray) -> bytes:
    """Canonical key of a (not necessarily canonical) panel: its rows' int8 bytes in canonical
    order (equal to ``FixedKSolution.key`` of the same rows)."""
    matrix = np.asarray(haplotypes, dtype=np.int8)
    order = sorted(range(len(matrix)), key=lambda index: (matrix[index].tobytes(), index))
    return np.ascontiguousarray(matrix[order]).tobytes()


_LOG_TWO = math.log(2.0)


def log_haplotype_set_count(n_sites: int, k: int) -> float:
    """log C(2^L, K): the code length (nats) of an unordered set of K distinct binary rows of
    length L, as K stable log-ratio terms (no 2^L integer is formed)."""
    universe_log = n_sites * _LOG_TWO
    terms = [universe_log + math.log1p(-math.ldexp(float(index), -n_sites)) - math.log(index + 1)
             for index in range(k)]
    result = math.fsum(terms)
    if -1e-12 < result < 0.0:
        return 0.0
    return result
