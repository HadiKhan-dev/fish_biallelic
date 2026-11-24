"""Dispatch the read-carried M0, M1 and M2 scores for one prepared component."""

from __future__ import annotations

from collections.abc import Callable

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
    for row in prange(len(trios)):
        full[row] = score_path(
            trios[row, 0], trios[row, 1], trios[row, 2], 2, pi, candidate_alt, stay, alpha, beta, row_arm,
            column_arm, pivot, excursion, branch, state_alt, child_gl, coefficient, site_index, site_start,
            site_stop, null_diagonal, null_off, null_frequency, mismatch, common_products, first_cached_child,
            read_offset, read_initial, read_reset, read_width, read_operator, read_ratio_table, class_cache,
        )
    return full


def component_m2(
    projection: Projection,
    terms: ChildTerms,
    trios: np.ndarray,
    zero: np.ndarray,
    one: np.ndarray,
    claim_threads=lambda: None,
) -> np.ndarray:
    """M2 log likelihoods [T] of the panel rows; a row with an evidence-free parent reduces to the other
    parent's M1 (or M0) screen value. (Production scores the panel through component_block_scores; this whole-chain
    form serves the reference checks.)"""
    children = terms.gl.shape[0]
    first_available = projection.available[trios[:, 1]]
    second_available = projection.available[trios[:, 2]]
    active = first_available & second_available
    two = np.empty(len(trios), dtype=np.float64)
    for row in np.flatnonzero(~active):
        child, first, second = trios[row]
        two[row] = (
            one[child, first] if first_available[row] else one[child, second] if second_available[row] else zero[child]
        )
    active_rows = np.flatnonzero(active)
    if not len(active_rows):
        return two
    active_children = trios[active_rows, 0]
    for child_start in range(0, children, terms.batch):
        child_stop = min(children, child_start + terms.batch)
        batch_rows = active_rows[(active_children >= child_start) & (active_children < child_stop)]
        if not len(batch_rows):
            continue
        claim_threads()
        products = terms.common(projection, child_start, child_stop)
        classes = terms.class_cache(projection, child_start, child_stop)
        claim_threads()
        two[batch_rows] = _m2_kernel(
            np.ascontiguousarray(trios[batch_rows]), *model_arguments(projection), *terms_args(projection, terms),
            products, child_start, *read_arguments(terms), classes,
        )
        del products, classes
    return two
