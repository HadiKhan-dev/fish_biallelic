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
from .read_forward import (
    _apply_log_emission,
    _apply_probability_emission,
    _bin_products,
    _initial_state,
    advance_classes,
    read_arguments,
    score_path,
)
from .read_transfer import _carry_background, carry_bin, named_transfers
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


@kernel(overloads=OVERLOADS["read_score._m1_children"], cache=True, parallel=True, nogil=True)
def _m1_children(
    unit_child,
    unit_start,
    unit_stop,
    candidates,
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
    output,
):
    """M1 (score_path, mode 1) of one child against the candidates candidates[unit_start[u]:unit_stop[u]] per unit u,
    the candidates' chains advancing bin by bin together: a structured bin's class transfers of the state pairs with
    a named first state are the same for every candidate (their alleles are the panel's; only BACKGROUND's is the
    candidate's), so they are built once per bin (named_transfers) and contracted into each candidate's chain, as
    the class cache does (read_transfer._carry_bin); BACKGROUND's row is carried per candidate (_carry_background).
    A bin without the child's read slots, or whose transfers are not finite, is carried directly (carry_bin). Each
    candidate's chain is otherwise score_path's, so a value differs from it only by the association of the bin's
    class products. ``output`` [len(candidates)] in candidate order."""
    states = state_alt.shape[0]
    bins = starts.shape[1]
    classes = read_initial.shape[1]
    named = states - 1
    cells = states * states
    plane = named * states
    for u in prange(len(unit_child)):
        child = unit_child[u]
        lo, hi = unit_start[u], unit_stop[u]
        count = hi - lo
        chains = np.empty((count, classes, states, states))
        emissions = np.empty((count, states, states))
        logs = np.zeros(count)
        live = np.empty(count, dtype=np.bool_)
        mix = np.empty((classes, states, states))
        scratch = np.empty((states, states))
        prefix = np.empty(states + 1)
        suffix = np.empty(states + 1)
        factor = np.empty(classes)
        column = np.empty(classes)
        moved = np.empty(classes)
        transfers = np.empty((classes, classes, states, states))
        bank = np.empty((states, states, classes, classes))
        word = np.empty(states, dtype=np.int64)
        for j in range(count):
            q = candidates[lo + j]
            live[j] = available[q]
            if live[j]:
                _initial_state(chains[j], child, q, 0, 1, pi, null_frequency, read_initial)
        carried = classes > 1 and read_offset[child + 1] != read_offset[child]
        base = read_offset[child] - starts[child, 0]
        flat = transfers.reshape((classes, classes, cells))
        for block in range(bins):
            enter, leave = read_width[child, block], read_width[child, block + 1]
            first_slot, stop_slot = starts[child, block], stops[child, block]
            shared = carried and first_slot != stop_slot
            if shared:
                shared = named_transfers(transfers, bank, word, factor, moved, child, block, state_alt, gl, indices,
                                         starts, stops, read_offset, read_reset, read_width, read_operator,
                                         read_ratio_table)
            for j in range(count):
                if not live[j]:
                    continue
                q = candidates[lo + j]
                current = chains[j]
                if block > 0:
                    advance_classes(current[:enter], scratch, child, q, 0, 1, block - 1, stay, alpha, beta, row_arm,
                                    column_arm, pivot, excursion, branch, null_diag, null_off, null_frequency,
                                    prefix, suffix)
                stable = _bin_products(emissions[j], child, q, 0, 1, block, candidate_alt, state_alt, gl,
                                       coefficient, indices, starts, stops, mismatch, products, first_child)
                if shared:
                    source = current.reshape((classes, cells))
                    target = mix.reshape((classes, cells))
                    for out in range(leave):
                        for cell in range(plane):
                            target[out, cell] = 0.0
                    for i in range(enter):
                        for out in range(leave):
                            for cell in range(plane):
                                target[out, cell] += source[i, cell] * flat[i, out, cell]
                    for out in range(leave):
                        for cell in range(plane):
                            source[out, cell] = target[out, cell]
                    _carry_background(current, mix, factor, child, q, np.int64(-1), candidate_alt, state_alt, gl,
                                      indices, read_reset, read_operator, read_ratio_table, base, first_slot,
                                      stop_slot, enter, leave)
                else:
                    carry_bin(current, mix, factor, column, child, block, q, -1, candidate_alt, state_alt, gl,
                              indices, starts, stops, read_offset, read_reset, read_width, read_operator,
                              read_ratio_table, None, child - first_child, moved)
                if stable:
                    increment = _apply_probability_emission(current[:leave], emissions[j])
                else:
                    increment = _apply_log_emission(current[:leave], emissions[j])
                if not np.isfinite(increment):
                    live[j] = False
                    logs[j] = -np.inf
                    continue
                logs[j] += increment
        for j in range(count):
            q = candidates[lo + j]
            output[lo + j] = logs[j] if available[q] else zero[child]


# Pair chunks per M1 query call: the calling thread re-reads its CPU share before each chunk, so a
# chromosome whose query outlasts its peers grows into the CPUs they free (a kernel's team is fixed for the whole call).
# Pairs are scored independently, so chunking does not change any value.
_CLAIM_CHUNKS = 32
_CLAIM_CHUNK_MINIMUM = 256
# Fewest candidates of one child-major unit: a unit builds each bin's shared transfers once for its candidates.
_CHILD_MAJOR_MINIMUM = 16


def child_major_cost(t: ChildTerms, states: int, candidates: float) -> tuple[float, float]:
    """(per-pair chain route, child-major route) carry cost per (pair, bin) in multiply-adds, from the structured
    class operator: a marker moves a state pair's C classes in own^2 + 4 C (own block, foreign sum, renewal, factor),
    m markers per child-bin; the child-major route contracts each named-first pair's C x C transfer, carries
    BACKGROUND's S pairs by marker and builds the S (S - 1) transfers (C rows each) once per child-bin for its
    ``candidates``. O(S^2 (own^2 + C)) per marker against O(S^2 C^2) per bin: the child-major route wins only while C
    is small beside the markers it replaces, so the per-pair route is kept at large K (AGENTS 2.5)."""
    classes = t.n_classes
    own = t.read_operator.shape[1] - 1
    markers = float(np.mean(t.site_stop - t.site_start)) if t.site_start.size else 0.0
    move = markers * (own * own + 4 * classes)
    per_pair = states * states * move
    named = (states - 1) * states
    child_major = named * classes * classes + states * move + named * classes * move / max(candidates, 1.0)
    return per_pair, child_major


def _child_units(pairs: np.ndarray, threads: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(order, unit child, unit start, unit stop): ``pairs`` grouped by child (stable), each child's candidates split
    into equal units, the fewest per child that make the units at least twice the team where the children allow
    (_CHILD_MAJOR_MINIMUM candidates per unit at least): each unit builds its bins' transfers once, so splitting a
    child trades that sharing for balance. Values do not depend on the split."""
    order = np.argsort(pairs[:, 0], kind="stable")
    children, first = np.unique(pairs[order, 0], return_index=True)
    bounds = np.append(first, len(pairs))
    wanted = max(1, -(-2 * max(1, threads) // len(children)))
    child, start, stop = [], [], []
    for k, c in enumerate(children):
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        parts = max(1, min(wanted, (hi - lo) // _CHILD_MAJOR_MINIMUM))
        edges = lo + (np.arange(parts + 1) * (hi - lo)) // parts
        child.extend([int(c)] * parts)
        start.extend(edges[:-1].tolist())
        stop.extend(edges[1:].tolist())
    return (order, np.asarray(child, dtype=np.int64), np.asarray(start, dtype=np.int64),
            np.asarray(stop, dtype=np.int64))


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
    from ``first_child`` (None: none). Structured read-state gaps without that cache run child-major where it is
    cheaper (child_major_cost)."""
    pairs = np.ascontiguousarray(pairs, dtype=np.int64)
    model, terms, reads = model_arguments(p), terms_args(p, t), read_arguments(t)
    output = np.empty(len(pairs))
    structured = t.n_classes > 1 and t.read_operator.shape[1] != t.read_operator.shape[2]
    if classes is None and structured and len(pairs):
        claim_threads()
        threads = numba.get_num_threads()
        order, unit_child, unit_start, unit_stop = _child_units(pairs, threads)
        per_pair, child_major = child_major_cost(t, p.n_states, len(pairs) / len(unit_child))
        if child_major < per_pair:
            candidates = np.ascontiguousarray(pairs[order, 1])
            scored = np.empty(len(pairs))
            step = max(2 * threads, -(-len(unit_child) // _CLAIM_CHUNKS))
            for first in range(0, len(unit_child), step):
                if first:
                    claim_threads()
                last = min(len(unit_child), first + step)
                _m1_children(unit_child[first:last], unit_start[first:last], unit_stop[first:last], candidates,
                             *model, p.available, *terms, zero_scores, products, first_child, *reads, scored)
            output[order] = scored
            return output
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
