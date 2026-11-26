"""Score configuration likelihood contributions by block, tiling rows that share a child's class transfers."""

from __future__ import annotations

import numba
import numpy as np
from numba import prange

from ...core.kernels import kernel
from .kernels.block_scores import OVERLOADS
from .read_forward import (
    _apply_log_emission,
    _apply_probability_emission,
    _bin_products,
    _initial_state,
    advance_classes,
    read_arguments,
)
from .read_transfer import carry_bin
from .scheduling import balanced_units, chain_work


@kernel(overloads=OVERLOADS["read_block_forward._score_tile"], cache=True, nogil=True)
def _score_tile(
    resolved,
    start,
    stop,
    block_of_bin,
    n_blocks,
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
    class_cache,
    scores,
):
    """``scores[r, h]`` = the log increments of score_path credited to block ``block_of_bin`` h, without chain
    resets, for the rows ``start`` to ``stop`` of ``resolved`` (child, first parent, second parent, mode). The rows
    advance together bin by bin, each row's chain exactly as alone (the same operations in the same order), so a tile
    of one child's rows reads each bin's class transfers once while they are in cache."""
    count = stop - start
    states = state_alt.shape[0]
    bins = site_start.shape[1]
    classes = read_initial.shape[1]
    full = np.empty((count, classes, states, states), dtype=np.float64)
    mix = np.empty((classes, states, states), dtype=np.float64)
    work = np.empty((states, states), dtype=np.float64)
    scratch = np.empty_like(work)
    prefix = np.empty(states + 1, dtype=np.float64)
    suffix = np.empty(states + 1, dtype=np.float64)
    factor = np.empty(classes, dtype=np.float64)
    column = np.empty(classes, dtype=np.float64)
    moved = np.empty(classes, dtype=np.float64)
    alive = np.ones(count, dtype=np.bool_)
    background_row = np.empty(states - 1, dtype=np.float64)
    for r in range(count):
        row = start + r
        _initial_state(full[r], resolved[row, 0], resolved[row, 1], resolved[row, 2], resolved[row, 3], pi,
                       null_frequency, read_initial)
        for block in range(n_blocks):
            scores[row, block] = 0.0
    for block in range(bins):
        target = block_of_bin[block]
        cached_child, cached_parent = -1, -1
        for r in range(count):
            if not alive[r]:
                continue
            row = start + r
            child, first_parent, second_parent, mode = resolved[row, 0], resolved[row, 1], resolved[row, 2], resolved[
                row, 3]
            first_source = first_parent if mode >= 1 else -1
            second_source = second_parent if mode == 2 else -1
            if block > 0:
                advance_classes(full[r][:read_width[child, block]], scratch, child, first_parent, second_parent, mode,
                                block - 1, stay, alpha, beta, row_arm, column_arm, pivot, excursion, branch,
                                null_diagonal, null_off, null_frequency, prefix, suffix)
            if count > 1 and first_source >= 0:
                reuse = child == cached_child and first_source == cached_parent
                stable_product = _bin_products(
                    work, child, first_parent, second_parent, mode, block, candidate_alt, state_alt, child_gl,
                    coefficient, site_index, site_start, site_stop, mismatch, common_products, first_cached_child,
                    background_row, reuse,
                )
                cached_child, cached_parent = (child, first_source) if stable_product else (-1, -1)
            else:
                stable_product = _bin_products(work, child, first_parent, second_parent, mode, block, candidate_alt,
                                               state_alt, child_gl, coefficient, site_index, site_start, site_stop,
                                               mismatch, common_products, first_cached_child)
            carry_bin(full[r], mix, factor, column, child, block, first_source, second_source, candidate_alt,
                      state_alt, child_gl, site_index, site_start, site_stop, read_offset, read_reset, read_width,
                      read_operator, read_ratio_table, class_cache, child - first_cached_child, moved)
            if stable_product:
                increment = _apply_probability_emission(full[r][:read_width[child, block + 1]], work)
            else:
                increment = _apply_log_emission(full[r][:read_width[child, block + 1]], work)
            if np.isfinite(increment):
                scores[row, target] += increment
            else:
                alive[r] = False
                for later in range(target, n_blocks):
                    scores[row, later] = -np.inf


@kernel(overloads=OVERLOADS["read_block_forward.score_block_rows"], cache=True, parallel=True, nogil=True)
def score_block_rows(rows, tiles, threads, child_cost, block_of_bin, n_blocks, available, pi, candidate_alt, stay, alpha, beta,
                     row_arm, column_arm, pivot, excursion, branch, state_alt, child_gl, coefficient,
                     site_index, site_start, site_stop, null_diagonal, null_off, null_frequency, mismatch,
                     read_offset, read_initial, read_reset, read_width, read_operator, read_ratio_table,
                     class_cache=None, first_child=0, common_products=None):
    """[R, H] block contributions; unavailable parental slots follow the original raw M1/M0 fallback.
    ``tiles``: row offsets of the tiles scored together (_score_tile; any grouping gives the same values: rows of
    one child share the class transfers). ``class_cache``: class_transfers of the rows' children from
    ``first_child`` (None: none)."""
    resolved = np.empty((len(rows), 4), dtype=np.int64)
    for row in range(len(rows)):
        child, first, second = rows[row, 0], rows[row, 1], rows[row, 2]
        if first >= 0 and not available[first]:
            first = -1
        if second >= 0 and not available[second]:
            second = -1
        if first < 0 and second >= 0:
            first, second = second, -1
        resolved[row, 0] = child
        resolved[row, 1] = max(first, 0)
        resolved[row, 2] = max(second, 0)
        resolved[row, 3] = int(first >= 0) + int(second >= 0)
    output = np.empty((len(rows), n_blocks))
    cost = np.empty(len(tiles) - 1, np.int64)
    for tile in range(len(cost)):
        start, stop = tiles[tile], tiles[tile + 1]
        cost[tile] = child_cost[resolved[start, 0]] * (stop - start)
    units, offsets = balanced_units(cost, threads)
    for group in prange(len(offsets) - 1):
        for cell in range(offsets[group], offsets[group + 1]):
            tile = units[cell]
            _score_tile(resolved, tiles[tile], tiles[tile + 1], block_of_bin, n_blocks, pi, candidate_alt, stay, alpha,
                        beta, row_arm, column_arm, pivot, excursion, branch, state_alt, child_gl, coefficient,
                        site_index, site_start, site_stop, null_diagonal, null_off, null_frequency, mismatch,
                        common_products, first_child, read_offset, read_initial, read_reset, read_width, read_operator,
                        read_ratio_table, class_cache, output)
    return output


# Rows per scoring tile (_score_tile): a tile of one child's rows reads each bin's class transfers (S^2 C^2 float64
# values) once instead of once per row. The tiles shrink (to one row) when there are too few tiles per thread to
# balance. One-class rows (no transfers to share) are tiles of one row. The values do not depend on the tiling.
TILE_ROWS = 16
TILES_PER_THREAD = 8


def row_tiles(children: np.ndarray, threads: int, largest: int = TILE_ROWS) -> np.ndarray:
    """int64 row offsets of the tiles of rows sorted by child (``children``): each child's run cut into tiles of at most
    ``largest`` rows, smaller while there are fewer than TILES_PER_THREAD tiles per thread (measured on the simulated
    cohorts' chromosome 19, 76 threads: 2-20 % less CPU than one row per task at about the same wall time)."""
    children = np.asarray(children, dtype=np.int64)
    runs = np.flatnonzero(np.diff(children)) + 1
    starts = np.concatenate(([0], runs))
    stops = np.concatenate((runs, [len(children)]))
    size = max(1, int(largest))
    while True:
        offsets = np.concatenate([np.arange(a, b, size) for a, b in zip(starts, stops, strict=True)]
                                 + [np.array([len(children)])]).astype(np.int64)
        if size == 1 or len(offsets) - 1 >= TILES_PER_THREAD * threads:
            return offsets
        size //= 2


def row_chunk(threads: int) -> int:
    # Reclaim CPUs between calls. Each call has enough rows for several
    # tiles per thread, and at least 512 on narrow teams to amortise dispatch.
    return max(512, TILE_ROWS * TILES_PER_THREAD * threads)


def component_block_scores(projection, terms, rows, block_of_bin, n_blocks, claim_threads=lambda: None):
    """[R, H] block contributions of ``rows`` (child, first, second; any order); with the read state the rows
    are scored by child batches that share their class-transfer cache (ChildTerms.class_cache), each batch's rows in
    tiles of one child (row_tiles)."""
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    if not len(rows):
        return np.empty((0, n_blocks))
    # Collapse unavailable slots before dispatch. Many panel rows then share the same M1 or M0 chain.
    available = projection.available
    rows = rows.copy()
    for slot in (1, 2):
        candidate = rows[:, slot]
        candidate[(candidate >= 0) & ~available[candidate]] = -1
    swap = (rows[:, 1] < 0) & (rows[:, 2] >= 0)
    rows[swap, 1] = rows[swap, 2]
    rows[swap, 2] = -1
    rows, inverse = np.unique(rows, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    arguments = (np.array(available, copy=True), projection.pi, projection.candidate_alt, projection.stay,
                 projection.alpha, projection.beta, projection.row_arm, projection.column_arm, projection.pivot,
                 projection.excursion, projection.branch, projection.state_alt, terms.gl, terms.coefficient,
                 terms.site_index, terms.site_start, terms.site_stop, terms.null_diagonal,
                 terms.null_off, terms.null_frequency, terms.mismatch, *read_arguments(terms))
    output = np.empty((len(rows), n_blocks))
    children, batch = terms.gl.shape[0], terms.trio_batch
    child_cost = chain_work(terms.site_start, terms.site_stop, terms.read_width)
    cuts = np.searchsorted(rows[:, 0], np.arange(0, children + batch, batch))
    for start, stop in zip(cuts[:-1], cuts[1:], strict=True):
        if start == stop:
            continue
        # Avoid building tables for unused children at either end of this window.
        child_start, child_stop = int(rows[start, 0]), int(rows[stop - 1, 0]) + 1
        claim_threads()
        products = terms.common(projection, child_start, child_stop)
        claim_threads()
        cache = terms.class_cache(projection, child_start, child_stop, trios=True)
        first = start
        while first < stop:
            claim_threads()
            threads = numba.get_num_threads()
            chunk = slice(first, min(stop, first + row_chunk(threads)))
            chosen = rows[chunk]
            if cache is None:
                tiles = np.arange(len(chosen) + 1, dtype=np.int64)
            else:
                tiles = row_tiles(chosen[:, 0], threads)
            output[chunk] = score_block_rows(
                chosen, tiles, threads, child_cost, block_of_bin, n_blocks, *arguments, cache, child_start, products
            )
            first = chunk.stop
        del cache, products
    return output[inverse]
