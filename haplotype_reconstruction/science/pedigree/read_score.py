"""Dispatch the read-carried M0, M1 and M2 scores for one prepared component."""

from __future__ import annotations

from collections.abc import Callable

import numba
import numpy as np
from numba import prange

from ...core.kernels import kernel
from .forward import (
    ChildTerms,
    model_arguments,
    terms_args,
)
from .kernels.component_scores import OVERLOADS
from .projection import Projection
from .read_forward import read_arguments, score_path
from .scheduling import balanced_units, chain_work


@kernel(overloads=OVERLOADS["read_score._m0_kernel"], cache=True, parallel=True, nogil=True)
def _m0_kernel(
    pi,
    candidate_alt,
    stay,
    alpha,
    beta,
    row_arm,
    column_arm,
    pivot,
    excursion,
    branch,
    state_alt,
    child_gl,
    coefficient,
    site_index,
    site_start,
    site_stop,
    null_diagonal,
    null_off,
    null_frequency,
    mismatch,
    eligible_child,
    read_offset,
    read_initial,
    read_reset, read_width, read_operator,
    read_ratio_table,
):
    full = np.empty(child_gl.shape[0], dtype=np.float64)
    for child in prange(child_gl.shape[0]):
        if eligible_child[child]:
            full[child] = score_path(
                child, 0, 0, 0, pi, candidate_alt, stay, alpha, beta, row_arm, column_arm, pivot, excursion, branch,
                state_alt, child_gl, coefficient, site_index, site_start, site_stop, null_diagonal, null_off,
                null_frequency, mismatch, None, 0, read_offset, read_initial, read_reset, read_width, read_operator,
                read_ratio_table, None,
            )
        else:
            full[child] = -np.inf
    return full


@kernel(overloads=OVERLOADS["read_score._dyads"], cache=True, parallel=True, nogil=True)
def _dyads(
    pairs,
    pi,
    candidate_alt,
    stay,
    alpha,
    beta,
    row_arm,
    column_arm,
    pivot,
    excursion,
    branch,
    available,
    state_alt,
    gl,
    coefficient,
    indices,
    starts,
    stops,
    null_diag,
    null_off,
    null_frequency,
    mismatch,
    zero,
    products,
    first_child,
    read_offset,
    read_initial,
    read_reset, read_width, read_operator,
    read_ratio_table,
    class_cache=None,
):
    output = np.empty(len(pairs))
    for i in prange(len(pairs)):
        c, p = pairs[i]
        if not available[p]:
            output[i] = zero[c]
        else:
            output[i] = score_path(
                c, p, 0, 1, pi, candidate_alt, stay, alpha, beta, row_arm, column_arm, pivot, excursion, branch,
                state_alt, gl, coefficient, indices, starts, stops, null_diag, null_off, null_frequency, mismatch,
                products, first_child, read_offset, read_initial, read_reset, read_width, read_operator,
                read_ratio_table, class_cache,
            )
    return output


def component_m0(
    p: Projection, t: ChildTerms, eligible_children: np.ndarray, claim_threads: Callable[[], int | None]
) -> np.ndarray:
    claim_threads()
    return _m0_kernel(*model_arguments(p), *terms_args(p, t), np.ascontiguousarray(eligible_children),
                      *read_arguments(t))


# Pair chunks per M1 query call: the calling thread re-reads its CPU share before each chunk, so a
# chromosome whose query outlasts its peers grows into the CPUs they free (a kernel's team is fixed for the whole call).
# Pairs are scored independently, so chunking does not change any value.
_CLAIM_CHUNKS = 32
_CLAIM_CHUNK_MINIMUM = 256


def component_m1(
    p: Projection,
    t: ChildTerms,
    pairs: np.ndarray,
    zero_scores: np.ndarray,
    first_child: int,
    products: np.ndarray | None,
    claim_threads: Callable[[], int | None],
    classes: np.ndarray | None = None,
) -> np.ndarray:
    """M1 of ``pairs`` (child, candidate); ``products`` the scalar cache (the M1 screen's completed one,
    ChildTerms.common_null_second) and ``classes`` the class-transfer cache (ChildTerms.class_cache) of the children
    from ``first_child`` (None: none)."""
    pairs = np.ascontiguousarray(pairs, dtype=np.int64)
    model, terms, reads = model_arguments(p), terms_args(p, t), read_arguments(t)
    output = np.empty(len(pairs))
    step = max(_CLAIM_CHUNK_MINIMUM, -(-len(pairs) // _CLAIM_CHUNKS))
    for start in range(0, max(1, len(pairs)), step):
        claim_threads()
        chunk = pairs[start : start + step]
        output[start : start + len(chunk)] = _dyads(
            chunk, *model, p.available, *terms, zero_scores, products, first_child, *reads, classes
        )
    return output


@kernel(overloads=OVERLOADS["read_score._m2_kernel"], cache=True, parallel=True, nogil=True)
def _m2_kernel(
    trios,
    units,
    offsets,
    available,
    pi,
    candidate_alt,
    stay,
    alpha,
    beta,
    row_arm,
    column_arm,
    pivot,
    excursion,
    branch,
    state_alt,
    child_gl,
    coefficient,
    site_index,
    site_start,
    site_stop,
    null_diagonal,
    null_off,
    null_frequency,
    mismatch,
    common_products,
    first_cached_child,
    read_offset,
    read_initial,
    read_reset, read_width, read_operator,
    read_ratio_table,
    class_cache=None,
):
    full = np.empty(len(trios), dtype=np.float64)
    for group in prange(len(offsets) - 1):
        for index in range(offsets[group], offsets[group + 1]):
            row = units[index]
            child, first, second = trios[row]
            first_available, second_available = available[first], available[second]
            mode = np.int64(first_available) + np.int64(second_available)
            if not first_available and second_available:
                first = second
            full[row] = score_path(
                child, first, second, mode, pi, candidate_alt, stay, alpha, beta, row_arm,
                column_arm, pivot, excursion, branch, state_alt, child_gl, coefficient, site_index, site_start,
                site_stop, null_diagonal, null_off, null_frequency, mismatch, common_products, first_cached_child,
                read_offset, read_initial, read_reset, read_width, read_operator, read_ratio_table, class_cache,
            )
    return full


def component_m2(
    projection: Projection,
    terms: ChildTerms,
    trios: np.ndarray,
    zero: np.ndarray | None = None,
    one: np.ndarray | None = None,
    claim_threads=lambda: None,
) -> np.ndarray:
    """Full-data M2 scores, with exact M1/M0 reductions on unavailable components."""
    children = terms.gl.shape[0]
    two = np.empty(len(trios), dtype=np.float64)
    if zero is None or one is None:
        active_rows = np.arange(len(trios))
    else:
        first_available = projection.available[trios[:, 1]]
        second_available = projection.available[trios[:, 2]]
        active = first_available & second_available
        for row in np.flatnonzero(~active):
            child, first, second = trios[row]
            two[row] = (
                one[child, first] if first_available[row] else one[child, second] if second_available[row] else zero[child]
            )
        active_rows = np.flatnonzero(active)
    if not len(active_rows):
        return two
    active_children = trios[active_rows, 0]
    if np.any(active_children[1:] < active_children[:-1]):
        order = np.argsort(active_children, kind="stable")
        active_rows = active_rows[order]
        active_children = active_children[order]
    model, args, reads = model_arguments(projection), terms_args(projection, terms), read_arguments(terms)
    from .read_block_forward import row_chunk, row_tiles, score_block_rows

    batch = terms.trio_batch
    cuts = np.searchsorted(active_children, np.arange(0, children + batch, batch))
    child_cost = chain_work(terms.site_start, terms.site_stop, terms.read_width)
    single_block = np.zeros(projection.n_bins, dtype=np.int64)
    availability = np.array(projection.available, copy=True)
    for start, stop in zip(cuts[:-1], cuts[1:], strict=True):
        if start == stop:
            continue
        child_start, child_stop = int(active_children[start]), int(active_children[stop - 1]) + 1
        claim_threads()
        products = terms.common(projection, child_start, child_stop)
        claim_threads()
        classes = terms.class_cache(projection, child_start, child_stop, trios=True)
        # Keep window caches resident while successive tiles refresh the live CPU claim.
        first = start
        while first < stop:
            claim_threads()
            threads = numba.get_num_threads()
            last = min(stop, first + row_chunk(threads))
            batch_rows = active_rows[first:last]
            chosen = np.ascontiguousarray(trios[batch_rows])
            if classes is None:
                units, offsets = balanced_units(child_cost[chosen[:, 0]], threads)
                two[batch_rows] = _m2_kernel(
                    chosen, units, offsets, projection.available, *model, *args, products, child_start, *reads, None,
                )
            else:
                tiles = row_tiles(chosen[:, 0], threads)
                scores = score_block_rows(chosen, tiles, threads, child_cost, single_block, 1, availability,
                                          *model, *args, *reads, classes, child_start, products)
                two[batch_rows] = scores[:, 0]
            first = last
        del products, classes
    return two
