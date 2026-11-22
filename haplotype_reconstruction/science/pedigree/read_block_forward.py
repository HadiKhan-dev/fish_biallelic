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
    _tempered_increment,
    advance_classes,
    read_arguments,
)
from .read_transfer import carry_bin


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
    exponent,
    null_diagonal,
    null_off,
    null_frequency,
    mismatch,
    common_products,
    first_cached_child,
    common_powered,
    read_offset,
    read_initial,
    read_reset, read_width, read_operator,
    read_ratio_table,
    read_powered,
    class_cache,
    views,
):
    """``views[r, h]`` = (tempered, full) log increments of score_path credited to block ``block_of_bin`` h, without
    chain resets, for the rows ``start`` to ``stop`` of ``resolved`` (child, first parent, second parent, mode). The
    rows advance together bin by bin, each row's chain exactly as alone (the same operations in the same order), so
    a tile of one child's rows reads each bin's class transfers once while they are in cache."""
    count = stop - start
    states = state_alt.shape[0]
    bins = exponent.shape[1]
    classes = read_initial.shape[1]
    tempered = np.empty((count, classes, states, states), dtype=np.float64)
    full = np.empty_like(tempered)
    mix = np.empty((classes, states, states), dtype=np.float64)
    work = np.empty((states, states), dtype=np.float64)
    scratch = np.empty_like(work)
    prefix = np.empty(states + 1, dtype=np.float64)
    suffix = np.empty(states + 1, dtype=np.float64)
    factor = np.empty(classes, dtype=np.float64)
    column = np.empty(classes, dtype=np.float64)
    moved = np.empty(classes, dtype=np.float64)
    tempered_alive = np.ones(count, dtype=np.bool_)
    full_alive = np.ones(count, dtype=np.bool_)
    for r in range(count):
        row = start + r
        _initial_state(tempered[r], resolved[row, 0], resolved[row, 1], resolved[row, 2], resolved[row, 3], pi,
                       null_frequency, read_initial)
        full[r, :, :, :] = tempered[r]
        for block in range(n_blocks):
            views[row, block, 0] = 0.0
            views[row, block, 1] = 0.0
    for block in range(bins):
        target = block_of_bin[block]
        for r in range(count):
            if not tempered_alive[r] and not full_alive[r]:
                continue
            row = start + r
            child, first_parent, second_parent, mode = resolved[row, 0], resolved[row, 1], resolved[row, 2], resolved[
                row, 3]
            first_source = first_parent if mode >= 1 else -1
            second_source = second_parent if mode == 2 else -1
            if block > 0:
                boundary = block - 1
                if tempered_alive[r]:
                    advance_classes(tempered[r][:read_width[child, block]], scratch, child, first_parent, second_parent, mode, boundary, stay,
                                    alpha, beta, row_arm, column_arm, pivot, excursion, branch, null_diagonal,
                                    null_off, null_frequency, prefix, suffix)
                if full_alive[r]:
                    advance_classes(full[r][:read_width[child, block]], scratch, child, first_parent, second_parent, mode, boundary, stay, alpha,
                                    beta, row_arm, column_arm, pivot, excursion, branch, null_diagonal, null_off,
                                    null_frequency, prefix, suffix)
            stable_product = _bin_products(work, child, first_parent, second_parent, mode, block, candidate_alt,
                                           state_alt, child_gl, coefficient, site_index, site_start, site_stop,
                                           mismatch, common_products, first_cached_child)
            if tempered_alive[r]:
                eta = exponent[child, block]
                carry_bin(tempered[r], mix, factor, column, child, block, first_source, second_source, candidate_alt,
                          state_alt, child_gl, site_index, site_start, site_stop, read_offset, read_reset, read_width, read_operator,
                          read_ratio_table, read_powered, True, eta, class_cache, child - first_cached_child, moved)
                powered = None if common_powered is None else common_powered[child - first_cached_child, block]
                increment = _tempered_increment(tempered[r][:read_width[child, block + 1]], work, scratch, eta, stable_product, powered)
                if np.isfinite(increment):
                    views[row, target, 0] += increment
                else:
                    tempered_alive[r] = False
                    for later in range(target, n_blocks):
                        views[row, later, 0] = -np.inf
            if full_alive[r]:
                carry_bin(full[r], mix, factor, column, child, block, first_source, second_source, candidate_alt,
                          state_alt, child_gl, site_index, site_start, site_stop, read_offset, read_reset, read_width, read_operator,
                          read_ratio_table, read_powered, False, 1.0, class_cache, child - first_cached_child, moved)
                if stable_product:
                    increment = _apply_probability_emission(full[r][:read_width[child, block + 1]], work, 1.0, None)
                else:
                    increment = _apply_log_emission(full[r][:read_width[child, block + 1]], work)
                if np.isfinite(increment):
                    views[row, target, 1] += increment
                else:
                    full_alive[r] = False
                    for later in range(target, n_blocks):
                        views[row, later, 1] = -np.inf


@kernel(overloads=OVERLOADS["read_block_forward.score_path_blocks"], cache=True, nogil=True)
def score_path_blocks(
    child,
    first_parent,
    second_parent,
    mode,
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
    exponent,
    null_diagonal,
    null_off,
    null_frequency,
    mismatch,
    common_products,
    first_cached_child,
    common_powered,
    block_of_bin,
    n_blocks,
    read_offset,
    read_initial,
    read_reset, read_width, read_operator,
    read_ratio_table,
    read_powered,
    class_cache=None,
):
    """[H, 2] tempered/full log increments of score_path, credited to the bin's block without chain resets (one row
    of _score_tile)."""
    resolved = np.empty((1, 4), dtype=np.int64)
    resolved[0, 0] = child
    resolved[0, 1] = first_parent
    resolved[0, 2] = second_parent
    resolved[0, 3] = mode
    views = np.empty((1, n_blocks, 2))
    _score_tile(resolved, np.int64(0), np.int64(1), block_of_bin, n_blocks, pi, candidate_alt, stay, alpha, beta,
                row_arm, column_arm, pivot, excursion, branch, state_alt, child_gl, coefficient, site_index,
                site_start, site_stop, exponent, null_diagonal, null_off, null_frequency, mismatch, common_products,
                first_cached_child, common_powered, read_offset, read_initial, read_reset, read_width, read_operator, read_ratio_table,
                read_powered, class_cache, views)
    return views[0]


@kernel(overloads=OVERLOADS["read_block_forward.score_block_rows"], cache=True, parallel=True, nogil=True)
def score_block_rows(rows, tiles, block_of_bin, n_blocks, available, pi, candidate_alt, stay, alpha, beta,
                     row_arm, column_arm, pivot, excursion, branch, state_alt, child_gl, coefficient,
                     site_index, site_start, site_stop, exponent, null_diagonal, null_off, null_frequency, mismatch,
                     read_offset, read_initial, read_reset, read_width, read_operator, read_ratio_table, read_powered,
                     class_cache=None, first_child=0):
    """Pooled [R,H] contributions; unavailable parental slots follow the original raw M1/M0 fallback.
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
    views = np.empty((len(rows), n_blocks, 2))
    for tile in prange(len(tiles) - 1):
        _score_tile(resolved, tiles[tile], tiles[tile + 1], block_of_bin, n_blocks, pi, candidate_alt, stay, alpha,
                    beta, row_arm, column_arm, pivot, excursion, branch, state_alt, child_gl, coefficient, site_index,
                    site_start, site_stop, exponent, null_diagonal, null_off, null_frequency, mismatch, None,
                    first_child, None, read_offset, read_initial, read_reset, read_width, read_operator, read_ratio_table, read_powered,
                    class_cache, views)
    output = np.empty((len(rows), n_blocks))
    for row in prange(len(rows)):
        for block in range(n_blocks):
            output[row, block] = 0.5 * (views[row, block, 0] + views[row, block, 1])
    return output


# Rows per scoring tile (_score_tile): a tile of one child's rows reads each bin's class transfers (2 S^2 C^2 float64
# values) once instead of once per row; the tiles shrink (to one row) while fewer than TILES_PER_THREAD per thread
# would leave threads idle at the end (tiles are handed out one at a time); one-class rows (no transfers to share)
# are tiles of one row. The values do not depend on the tiling.
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


def component_block_scores(projection, terms, rows, block_of_bin, n_blocks):
    """[R, H] pooled block contributions of ``rows`` (child, first, second; any order); with the read state the rows
    are scored by child batches that share their class-transfer cache (ChildTerms.class_cache), each batch's rows in
    tiles of one child (row_tiles)."""
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    threads = numba.get_num_threads()
    arguments = (np.array(projection.available, copy=True), projection.pi, projection.candidate_alt, projection.stay,
                 projection.alpha, projection.beta, projection.row_arm, projection.column_arm, projection.pivot,
                 projection.excursion, projection.branch, projection.state_alt, terms.gl, terms.coefficient,
                 terms.site_index, terms.site_start, terms.site_stop, terms.exponent, terms.null_diagonal,
                 terms.null_off, terms.null_frequency, terms.mismatch, *read_arguments(terms))
    output = np.empty((len(rows), n_blocks))
    if not len(rows):
        return output
    children = terms.gl.shape[0]
    batch = children if terms.n_classes == 1 else terms.batch
    for child_start in range(0, children, batch):
        child_stop = min(children, child_start + batch)
        selected = np.flatnonzero((rows[:, 0] >= child_start) & (rows[:, 0] < child_stop))
        if not len(selected):
            continue
        selected = selected[np.argsort(rows[selected, 0], kind="stable")]
        chosen = np.ascontiguousarray(rows[selected])
        cache = None if terms.n_classes == 1 else terms.class_cache(projection, child_start, child_stop)
        if cache is None:
            # one row per tile, split statically as before: without class transfers to share, tiles would only
            # coarsen the split
            tiles = np.arange(len(chosen) + 1, dtype=np.int64)
            output[selected] = score_block_rows(chosen, tiles, block_of_bin, n_blocks, *arguments, cache, child_start)
        else:
            # tiles differ in cost (their rows' configurations, their child's markers): hand them out one at a time
            with numba.parallel_chunksize(1):
                output[selected] = score_block_rows(chosen, row_tiles(chosen[:, 0], threads), block_of_bin,
                                                    n_blocks, *arguments, cache, child_start)
        del cache
    return output
