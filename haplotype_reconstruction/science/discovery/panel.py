"""Fixed-K solutions of a block and their one canonical form.

A *panel* is K distinct binary rows (founder haplotypes at the block's discovery sites). A
:class:`FixedKSolution` is a panel with each sample's unordered row pair (the unknown copy is
row index K), the per-sample cost and the binned switching-Viterbi NLL of its fit.

Canonical form: rows in lexicographic order of their 0/1 sequences (the order of the rows'
int8 bytes compared as unsigned bytes; equal rows keep their input order), assignments remapped
to that order with each pair sorted ascending. The canonical key is the rows' int8 bytes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar

import numba
import numpy as np

from haplotype_reconstruction.core.ties import BlockQuanta, TieCounter, class_sort, fixed_policy

from ...contracts.roles import record_field
from ...core.kernels import kernel
from .parameters import SET_COUNT_ROUNDING_FLOOR

if TYPE_CHECKING:
    from .fixed_k.workspace import FixedKWorkspace


def state_layout(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kind and endpoints of row-major known pairs, row-unknown states and the all-unknown state.

    Endpoints are int64; unused endpoints are -1 until state_membership maps them to K.
    """
    pair_i, pair_j = np.triu_indices(k)
    n_rr = len(pair_i)
    n_states = n_rr + k + 1
    kind = np.empty(n_states, dtype=np.int64)
    first = np.full(n_states, -1, dtype=np.int64)
    second = np.full(n_states, -1, dtype=np.int64)
    kind[:n_rr] = 0
    first[:n_rr] = pair_i
    second[:n_rr] = pair_j
    kind[n_rr : n_rr + k] = 1
    first[n_rr : n_rr + k] = np.arange(k)
    kind[-1] = 2
    return kind, first, second


FitResult = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray | None, np.ndarray | None]
_T = TypeVar("_T")

# Release 2.1: NLL comparisons use the block's ``nll`` tie quantum (BlockQuanta.nll; the old absolute
# NLL_TOLERANCE = 1e-10 sat at the 1e-9-noise scale). Duplicates keep the first fit seen unless the
# other is better by more than the quantum.


@dataclass(frozen=True)
class FixedKSolution:
    """A fitted fixed-K factorisation in canonical form (arrays read-only).

    haplotypes: int64[K, L] binary rows, canonical order.
    assignments: int64[N, 2] sorted row pairs, K = unknown copy or a copy absent from the meet of the tied states.
        This pair is a summary; tie_offsets/tie_states retain fractional evidence membership.
    per_sample_cost: float64[N] per-sample binned switching-Viterbi NLL.
    unknown_slots: int64[N] unknown copies of the sample's (tied) A-step states (0, 1, 2); without ties the unknown
        copies in its pair.
    iterations: coordinate-descent iterations the fit reported.
    nll: total NLL (sum of per_sample_cost, as the fitter reported it).
    fixed_point: the fitter proved haplotypes an A(H), H(A) fixed point in this workspace.
    key: canonical key (int8 bytes of the rows).
    quanta: the block's BlockQuanta the solution was fitted under (None for solutions built outside a
        workspace); orders of solutions use tie classes of kind ``nll`` with them (maths.ties).
    """

    haplotypes: np.ndarray = record_field("state", axes=("K", "L"), dtype="int64")
    assignments: np.ndarray = record_field("state", axes=("N", "2"), dtype="int64")
    per_sample_cost: np.ndarray = record_field("state", axes=("N",), units="nats", dtype="float64")
    unknown_slots: np.ndarray = record_field("state", axes=("N",), units="copies", dtype="int64")
    iterations: int = record_field("state", units="iterations", dtype="int")
    nll: float = record_field("state", units="nats", dtype="float")
    fixed_point: bool = record_field("state", dtype="bool", compare=False)
    key: bytes = record_field("state", dtype="bytes", repr=False)
    quanta: BlockQuanta | None = record_field(
        "state", dtype="BlockQuanta|None", default=None, compare=False, repr=False
    )
    tie_offsets: np.ndarray | None = record_field(
        "state", axes=("N+1",), dtype="int64|None", default=None, compare=False, repr=False
    )
    tie_states: np.ndarray | None = record_field(
        "state", axes=("M",), dtype="int32|None", default=None, compare=False, repr=False
    )

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


@kernel(
    overloads=(
        (
            numba.types.Array(numba.int64, 2, "C", readonly=False),
            numba.types.Array(numba.int64, 2, "C", readonly=False),
        ),
        (
            numba.types.Array(numba.int64, 2, "C", readonly=True),
            numba.types.Array(numba.int64, 2, "C", readonly=True),
        ),
    ),
    cache=True,
    nogil=True,
)
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
    return canonical_haplotypes, canonical_assignments, canonical_bytes, inverse


@kernel(
    overloads=(
        (numba.types.Array(numba.boolean, 2, "C", readonly=False),),
        (numba.types.Array(numba.int64, 2, "C", readonly=False),),
        (numba.types.Array(numba.uint8, 2, "C", readonly=False),),
        (numba.types.Array(numba.int64, 2, "C", readonly=True),),
    ),
    cache=True,
    nogil=True,
)
def lexicographic_row_order(rows):
    """The stable lexicographic order of a panel's rows compared numerically, first site most significant: the
    permutation ``np.lexsort(rows.T[::-1])``, computed by one compiled insertion sort."""
    k = rows.shape[0]
    n_sites = rows.shape[1]
    order = np.arange(k)
    for index in range(1, k):
        current = order[index]
        position = index
        while position > 0:
            previous = order[position - 1]
            less = False
            for site in range(n_sites):
                a = rows[current, site]
                b = rows[previous, site]
                if a != b:
                    less = a < b
                    break
            if not less:
                break
            order[position] = previous
            position -= 1
        order[position] = current
    return order


@lru_cache(maxsize=64)
def _state_pairs(k):
    """Read-only state endpoints, shared by panels with the same founder count."""
    first, second = np.triu_indices(k)
    return (
        _readonly(np.concatenate((first, np.arange(k), [k]))),
        _readonly(np.concatenate((second, np.full(k + 1, k)))),
    )


def canonical_solution(
    fit: FitResult, workspace: FixedKWorkspace | None, quanta: BlockQuanta | None = None
) -> FixedKSolution:
    """The canonical solution of a fit ``(H, A, per_sample_cost, unknown_slots, iterations, nll)``.

    ``fixed_point`` is set when ``workspace`` certifies the canonical rows (in that exact order)
    as a fixed point of its coordinate map; None means never certified. ``quanta``: the block's
    BlockQuanta (default: the workspace's when given).
    """
    if quanta is None and workspace is not None:
        quanta = workspace.quanta
    haplotypes, assignments, key_bytes, inverse = _canonical_panel_kernel(
        np.ascontiguousarray(fit[0], dtype=np.int64), np.ascontiguousarray(fit[1], dtype=np.int64)
    )
    offsets, states = (fit[6], fit[7]) if len(fit) > 6 else (None, None)
    if offsets is not None:
        k = len(haplotypes)
        state_first, state_second = _state_pairs(k)
        mapping = np.append(inverse, k)
        a = mapping[state_first[states]]
        b = mapping[state_second[states]]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        n_pairs = k * (k + 1) // 2
        remapped = np.where(
            hi < k, lo * k - lo * (lo - 1) // 2 + hi - lo, np.where(lo < k, n_pairs + lo, n_pairs + k)
        ).astype(np.int64)
        # Each sample's state IDs sorted ascending, all segments in one sort: state IDs lie in
        # [0, n_pairs + k], so segment * (n_pairs + k + 1) + ID orders by segment first, then ID.
        segment = np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets))
        stride = segment * (n_pairs + k + 1)
        states = _readonly((np.sort(stride + remapped) - stride).astype(np.int32))
    certified = bool(workspace is not None and workspace.certifies_fixed_point(haplotypes))
    return FixedKSolution(
        haplotypes=_readonly(haplotypes),
        assignments=_readonly(assignments),
        per_sample_cost=fit[2],
        unknown_slots=fit[3],
        iterations=int(fit[4]),
        nll=float(fit[5]),
        fixed_point=certified,
        key=key_bytes.tobytes(),
        quanta=quanta,
        tie_offsets=offsets,
        tie_states=states,
    )


def restricted(solution: FixedKSolution, samples: np.ndarray) -> FixedKSolution:
    """The same panel with only the ``samples`` rows of the sample arrays (nll re-summed)."""
    cost = solution.per_sample_cost[samples]
    offsets, states = subset_ties(solution, samples)
    return FixedKSolution(
        haplotypes=solution.haplotypes,
        assignments=_readonly(solution.assignments[samples]),
        per_sample_cost=_readonly(cost),
        unknown_slots=_readonly(solution.unknown_slots[samples]),
        iterations=solution.iterations,
        nll=math.fsum(cost.tolist()),
        fixed_point=solution.fixed_point,
        key=solution.key,
        quanta=solution.quanta,
        tie_offsets=offsets,
        tie_states=states,
    )


@kernel(overloads=((numba.types.Array(numba.uint8, 2, "C", readonly=False),),), cache=True, nogil=True)
def _compact_ties(mask):
    offsets = np.zeros(len(mask) + 1, dtype=np.int64)
    for sample in range(len(mask)):
        count = 0
        for state in range(mask.shape[1]):
            count += int(mask[sample, state] != 0)
        offsets[sample + 1] = offsets[sample] + (count if count > 1 else 0)
    states = np.empty(offsets[-1], dtype=np.int32)
    for sample in range(len(mask)):
        if offsets[sample + 1] == offsets[sample]:
            continue
        position = offsets[sample]
        for state in range(mask.shape[1]):
            if mask[sample, state]:
                states[position] = state
                position += 1
    return offsets, states


def compact_ties(mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Retain state IDs only for multi-state samples (singletons use the pair summary)."""
    offsets, states = _compact_ties(mask)
    if len(states) == 0:
        return None, None
    return _readonly(offsets), _readonly(states)


def subset_ties(solution: FixedKSolution, samples: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compact membership for a sample restriction or permutation."""
    if solution.tie_offsets is None:
        return None, None
    old = solution.tie_offsets
    assert solution.tie_states is not None  # compact_ties sets both or neither
    counts = np.diff(old)[samples]
    if not np.any(counts):
        return None, None
    offsets = np.zeros(len(samples) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    states = np.concatenate([solution.tie_states[old[s] : old[s + 1]] for s in samples])
    return _readonly(offsets), _readonly(states)


def solution_tie_mask(solution: FixedKSolution) -> np.ndarray:
    """Full winning-category membership, including singletons; old fixtures use their pair."""
    return assignment_tie_mask(solution.assignments, solution.k, solution.tie_offsets, solution.tie_states)


@kernel(
    overloads=(
        (
            numba.types.Array(numba.int64, 2, "C", readonly=False),
            numba.int64,
            numba.types.Array(numba.int64, 1, "C", readonly=True),
            numba.types.Array(numba.int32, 1, "C", readonly=True),
        ),
        (
            numba.types.Array(numba.int64, 2, "C", readonly=False),
            numba.int64,
            numba.types.Omitted(None),
            numba.types.Omitted(None),
        ),
        (
            numba.types.Array(numba.int64, 2, "C", readonly=False),
            numba.int64,
            numba.types.none,
            numba.types.none,
        ),
        (
            numba.types.Array(numba.int64, 2, "C", readonly=True),
            numba.int64,
            numba.types.Array(numba.int64, 1, "C", readonly=True),
            numba.types.Array(numba.int32, 1, "C", readonly=True),
        ),
        (
            numba.types.Array(numba.int64, 2, "C", readonly=True),
            numba.int64,
            numba.types.none,
            numba.types.none,
        ),
    ),
    cache=True,
    nogil=True,
)
def assignment_tie_mask(assignments, k, offsets=None, states=None):
    """Full state membership; explicit multi-state segments replace the singleton pair."""
    n_pairs = k * (k + 1) // 2
    mask = np.zeros((len(assignments), n_pairs + k + 1), dtype=np.uint8)
    for sample in range(len(assignments)):
        if offsets is not None and states is not None:
            if offsets[sample + 1] != offsets[sample]:
                for position in range(offsets[sample], offsets[sample + 1]):
                    mask[sample, states[position]] = 1
                continue
        a, b = assignments[sample, 0], assignments[sample, 1]
        if b < k:
            state = a * k - a * (a - 1) // 2 + b - a
        elif a < k:
            state = n_pairs + a
        else:
            state = n_pairs + k
        mask[sample, state] = 1
    return mask


def has_distinct_rows(solution: FixedKSolution) -> bool:
    """Whether the solution's K rows are pairwise distinct (a complete K panel)."""
    width = solution.sites
    key = solution.key
    if width == 0:
        return solution.k == 1
    return len({key[start : start + width] for start in range(0, len(key), width)}) == solution.k


def _nll_quantum(solutions, quanta):
    if quanta is not None:
        return quanta.nll
    for solution in solutions:
        if solution.quanta is not None:
            return solution.quanta.nll
    return BlockQuanta.floors().nll


def sort_solutions(
    solutions: Iterable[FixedKSolution],
    quanta: BlockQuanta | None = None,
    *,
    by_k: bool = True,
    counter: TieCounter | None = None,
    reverse: bool,
) -> list[FixedKSolution]:
    """Solutions in tie-class order of their NLLs (kind ``nll``; quanta: the block's, default the
    solutions' own), then smaller K, then the canonical key."""
    solutions = list(solutions)
    q = _nll_quantum(solutions, quanta)
    return class_sort(
        solutions,
        lambda s: s.nll,
        q,
        secondary=(lambda s: (s.k,)) if by_k else None,
        content=lambda s: s.key,
        counter=counter,
        kind="nll_order",
        reverse=fixed_policy("discovery.panel.order", q, reverse=reverse).reverse,
    )


def deduplicate(
    solutions: Iterable[FixedKSolution],
    beam_width: int | None = None,
    *,
    quanta: BlockQuanta | None = None,
    reverse: bool,
) -> tuple[FixedKSolution, ...]:
    """Distinct complete panels (by key), each with its lowest-NLL fit (a later fit replaces an
    earlier one only when better by more than the ``nll`` quantum), in tie-class order of NLL, then
    smaller K, then key (sort_solutions; 2.0's (nll, K, key) order with tie classes), optionally the
    first beam_width (the cut falls inside a class by K and key)."""
    solutions = [solution for solution in solutions if has_distinct_rows(solution)]
    q = _nll_quantum(solutions, quanta)
    q = fixed_policy("discovery.panel.dedup", q, reverse=reverse).quantum()
    unique: dict[bytes, FixedKSolution] = {}
    for solution in solutions:
        previous = unique.get(solution.key)
        if previous is None or solution.nll < previous.nll - q:
            unique[solution.key] = solution
    ordered = sort_solutions(unique.values(), quanta, reverse=reverse)
    return tuple(ordered if beam_width is None else ordered[:beam_width])


@kernel(
    overloads=(
        (
            numba.types.Array(numba.boolean, 2, "C", readonly=False),
            numba.int64,
            numba.types.Array(numba.uint8, 2, "C", readonly=False),
        ),
    ),
    cache=True,
    nogil=True,
)
def _unknown_alleles_from_membership(observed, k, mask):
    """Exact Boolean carrier union, without per-founder fancy-index temporaries."""
    n, length = observed.shape
    carriers = np.zeros((k, n), dtype=np.bool_)
    state = 0
    for first in range(k):
        for second in range(first, k):
            for sample in range(n):
                if mask[sample, state]:
                    carriers[first, sample] = True
                    carriers[second, sample] = True
            state += 1
    for founder in range(k):
        for sample in range(n):
            if mask[sample, state + founder]:
                carriers[founder, sample] = True
    known = np.zeros((k, length), dtype=np.bool_)
    for founder in range(k):
        for sample in range(n):
            if carriers[founder, sample]:
                for site in range(length):
                    known[founder, site] |= observed[sample, site]
    return ~known


def unknown_alleles(
    assignments: np.ndarray, observed: np.ndarray, k: int, tie_mask: np.ndarray | None = None
) -> np.ndarray:
    """bool[K, L]: True where no sample assigned to the founder (either copy; K = unknown copy) observes the site
    (exact integer counts): the founder's allele there is unknown (release 2.2)."""
    assignments = np.asarray(assignments, dtype=np.int64)
    observed = np.asarray(observed, dtype=np.bool_)
    if tie_mask is not None:
        return _unknown_alleles_from_membership(observed, k, tie_mask)
    known = np.zeros((k, observed.shape[1]), dtype=np.bool_)
    for copy in (0, 1):
        rows = assignments[:, copy]
        for founder in np.unique(rows[rows < k]):
            known[founder] |= observed[rows == founder].any(axis=0)
    return ~known


@kernel(
    overloads=(
        (
            numba.types.Array(numba.int64, 2, "C", readonly=False),
            numba.types.Array(numba.boolean, 2, "C", readonly=False),
        ),
        (
            numba.types.Array(numba.int64, 2, "C", readonly=True),
            numba.types.Array(numba.boolean, 2, "C", readonly=False),
        ),
    ),
    cache=True,
    nogil=True,
)
def _known_content_bytes(haplotypes, unknown):
    k, length = haplotypes.shape
    masked = np.empty((k, length), dtype=np.uint8)
    for row in range(k):
        for site in range(length):
            masked[row, site] = np.uint8(255) if unknown[row, site] else np.uint8(haplotypes[row, site])
    # Unsigned byte order is the original bytes order, including unknown (-1 -> 255).
    order = lexicographic_row_order(masked)
    result = np.empty(k * length + max(0, k - 1), dtype=np.uint8)
    position = 0
    for index in range(k):
        if index:
            result[position] = 124  # the existing b"|" row separator
            position += 1
        for site in range(length):
            result[position] = masked[order[index], site]
            position += 1
    return result


def known_content_key(haplotypes: np.ndarray, unknown: np.ndarray) -> bytes:
    """A panel's science: its rows with unknown alleles masked (-1), as a sorted multiset (bytes)."""
    haplotypes = np.asarray(haplotypes, dtype=np.int64)
    unknown = np.asarray(unknown, dtype=bool)
    return _known_content_bytes(haplotypes, unknown).tobytes() + b"#" + str(haplotypes.shape).encode()


def whole_class_cut(
    ordered: Sequence[_T],
    classes: Sequence[int],
    width: int,
    omissions: list[dict] | None = None,
    *,
    cause: str = "beam",
    scope: str = "NLL classes",
) -> tuple[list[_T], bool]:
    """Keep complete ranked classes until reaching ``width``, extending a straddling class
    through 2 x width. Larger straddling classes are omitted whole. Lower-ranked classes after
    the stopping boundary are ordinary truncations. Returns (kept, a straddling class was omitted).
    """
    kept: list[_T] = []
    position = 0
    cut = False
    while position < len(ordered):
        stop = position + 1
        while stop < len(ordered) and classes[stop] == classes[position]:
            stop += 1
        if position and len(kept) >= width:
            break
        if len(kept) + stop - position > 2 * width:
            cut = True
            if omissions is not None:
                omissions.append(
                    dict(
                        kind="tie_class_at_budget",
                        cause=cause + "_tie_class_at_budget",
                        count=stop - position,
                        scope=scope,
                        count_kind="exact",
                    )
                )
            position = stop
            break
        kept.extend(ordered[position:stop])
        position = stop
    if position < len(ordered) and omissions is not None:
        omissions.append(
            dict(
                kind="ranked_truncation",
                cause=cause + "_ranked_truncation",
                count=len(ordered) - position,
                scope=scope,
                count_kind="exact",
            )
        )
    return kept, cut


@kernel(
    overloads=(
        (numba.types.Array(numba.int64, 2, "C", readonly=False),),
        (numba.types.Array(numba.int64, 2, "C", readonly=True),),
    ),
    cache=True,
    nogil=True,
)
def _unique_binary_indices(rows):
    # packbits compares truth values. Preserve its equivalence classes and keep-first rule.
    binary = rows != 0
    order = lexicographic_row_order(binary)
    keep = np.empty(len(rows), dtype=np.int64)
    count = 0
    for position in range(len(order)):
        distinct = position == 0
        if position:
            for site in range(rows.shape[1]):
                if binary[order[position], site] != binary[order[position - 1], site]:
                    distinct = True
                    break
        if distinct:
            keep[count] = order[position]
            count += 1
    return keep[:count]


def unique_rows(matrix: np.ndarray) -> np.ndarray:
    """Distinct binary rows in lexicographic order (first occurrence kept), C-contiguous."""
    rows = np.asarray(matrix)
    if len(rows) <= 1:
        return np.array(rows, copy=True, order="C")
    return np.ascontiguousarray(rows[_unique_binary_indices(rows)])


def panel_key(haplotypes: np.ndarray) -> bytes:
    """Canonical key of a (not necessarily canonical) panel: its rows' int8 bytes in canonical
    order (equal to ``FixedKSolution.key`` of the same rows)."""
    matrix = np.asarray(haplotypes, dtype=np.int8)
    order = sorted(range(len(matrix)), key=lambda index: (matrix[index].tobytes(), index))
    return np.ascontiguousarray(matrix[order]).tobytes()


_LOG_TWO = math.log(2.0)


@lru_cache(maxsize=4096)
def log_haplotype_set_count(n_sites: int, k: int) -> float:
    """log C(2^L, K): the code length (nats) of an unordered set of K distinct binary rows of
    length L, as K stable log-ratio terms (no 2^L integer is formed). Memoised: a pure function of two
    integers, evaluated for every objective of a block search."""
    universe_log = n_sites * _LOG_TWO
    terms = [universe_log + math.log1p(-math.ldexp(float(index), -n_sites)) - math.log(index + 1) for index in range(k)]
    result = math.fsum(terms)
    if SET_COUNT_ROUNDING_FLOOR < result < 0.0:
        return 0.0
    return result


@kernel(
    overloads=(
        (
            numba.types.Array(numba.int64, 2, "C", readonly=False),
            numba.int64,
            numba.types.Array(numba.uint8, 2, "C", readonly=False),
            numba.types.Array(numba.int64, 1, "C", readonly=False),
            numba.types.Array(numba.int64, 1, "C", readonly=False),
            numba.types.Array(numba.int64, 1, "C", readonly=False),
        ),
        (
            numba.types.Array(numba.int64, 2, "C", readonly=True),
            numba.int64,
            numba.types.Array(numba.uint8, 2, "C", readonly=False),
            numba.types.Array(numba.int64, 1, "C", readonly=False),
            numba.types.Array(numba.int64, 1, "C", readonly=False),
            numba.types.Array(numba.int64, 1, "C", readonly=False),
        ),
    ),
    cache=True,
    nogil=True,
)
def state_membership(assignments, k, tie_mask, state_kind, state_first, state_second):
    """Compact full membership and actual founder pairs (K denotes the unknown copy)."""
    n_samples = assignments.shape[0]
    n_states = state_kind.shape[0]
    first = np.empty(n_states, dtype=np.int64)
    second = np.empty(n_states, dtype=np.int64)
    lookup = np.full((k + 1, k + 1), -1, dtype=np.int64)
    for state in range(n_states):
        kind = state_kind[state]
        a = state_first[state] if kind < 2 else k
        b = state_second[state] if kind == 0 else k
        first[state], second[state] = a, b
        lookup[a, b] = state
    offsets = np.zeros(n_samples + 1, dtype=np.int64)
    with_ties = tie_mask.shape[0] == n_samples
    for sample in range(n_samples):
        size = 0
        if with_ties:
            for state in range(n_states):
                size += int(tie_mask[sample, state] != 0)
        offsets[sample + 1] = offsets[sample] + max(1, size)
    members = np.empty(offsets[-1], dtype=np.int64)
    for sample in range(n_samples):
        offset = offsets[sample]
        if with_ties:
            for state in range(n_states):
                if tie_mask[sample, state]:
                    members[offset] = state
                    offset += 1
        if offset == offsets[sample]:
            members[offset] = lookup[assignments[sample, 0], assignments[sample, 1]]
    return offsets, members, first, second


__all__ = [
    "FixedKSolution",
    "canonical_solution",
    "restricted",
    "compact_ties",
    "subset_ties",
    "solution_tie_mask",
    "assignment_tie_mask",
    "has_distinct_rows",
    "sort_solutions",
    "deduplicate",
    "unknown_alleles",
    "known_content_key",
    "whole_class_cut",
    "unique_rows",
    "panel_key",
    "log_haplotype_set_count",
    "lexicographic_row_order",
    "state_membership",
]
