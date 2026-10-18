"""Certification of the join's same-count moves: exclusion forwards and two-region bounds.

Nothing here accepts a panel. For the join objective (:mod:`.objective`; LL = sum over samples of
V_s) the kernels give, per sample,

* exclusion forwards: the forward maxima G^{-F}(t) of the panel without the paths in F
  (|F| <= 2). G^{-F} = G at every bin iff at every bin some maximising state avoids F (induction
  over the Potts recursion: surviving states see the same switch base). The forward records, per
  bin, the vertex covers of size <= 2 of its maximiser set, so G^{-F} is recomputed only where F
  covers the maximisers at some bin; this is exact, also under ties. The count step reads
  V_s(P minus f) for every row f from them;
* the two-region bound: an upper bound on the value of the panel with the paths F removed and new
  paths H (|H| <= 2) added. Runs through the unchanged states A (pairs of retained paths) are
  bounded by telescoped increments of G^{-F} (the A-only programme can take its best prefix and
  switch in, so entering A is free); runs through the new states N (pairs containing a new path)
  follow the exact Potts dynamics (entering N or switching inside N costs the penalty). The DP is
  max-plus linear in its state, so a structural move (local, onesided, splice) is bounded in two
  halves: the state before the part where the move departs from a reference path, plus the
  backward vector after it (structural_bounds: per founder, and per pair of founders, one forward
  and one backward pass serve every block and boundary). The halves are summed in another order
  than one forward pass, so values agree up to float rounding, which the exclusion's float margin
  covers (moves.best_improvement).

A move is excluded only when its bound proves that it cannot raise the join likelihood. Per-sample
values are computed by one thread each and summed in sample order, so every bound is independent
of the thread count. Costs per sample: forward O(m k^2); a bound O(m k); all structural moves
O(m k (k^2 + sum_b K_b)).
"""
from __future__ import annotations

import numba
import numpy as np
from numba import njit, prange

from .objective import (column_sums, forward, max_range, pair_lists, score_samples,
                        widest_block)

# Paths per birth_bounds call (per-sample values of one call are held at once).
_BIRTH_CHUNK = 4096
# Bytes of per-sample replacement bounds held at once by pool_bounds.
_REPLACE_CHUNK_BYTES = 1 << 26


@njit(cache=True, parallel=True, nogil=True)
def forward_covers(flat, block_start, block_haps, block_bins, rows, penalty, total_bins, samples):
    """G[s, t] and the per-sample single covers and disjoint pair covers."""
    k = rows.shape[0]
    forward_values = np.empty((samples, total_bins))
    singles = np.zeros((samples, k), np.bool_)
    pairs = np.zeros((samples, k, k), np.bool_)
    first, second = pair_lists(np.ones(k, np.bool_))
    widest = widest_block(block_bins)
    for sample in prange(samples):
        gathered = np.empty((widest, len(first)))
        forward(flat, block_start, block_haps, block_bins, rows, first, second, penalty, sample,
                forward_values[sample], singles[sample], pairs[sample], True, True, gathered)
    return forward_values, singles, pairs


@njit(cache=True, parallel=True, nogil=True)
def single_exclusions(flat, block_start, block_haps, block_bins, rows, penalty, total_bins, index,
                      slots):
    """G^{-f} for every (sample, f) with index >= 0, and the single covers of
    each such forward (second[s, f, j]: j covers a maximiser set of P \\ f)."""
    k = rows.shape[0]
    samples = index.shape[0]
    excluded = np.empty((slots, total_bins))
    second = np.zeros((samples, k, k), np.bool_)
    unused = np.zeros((1, 1), np.bool_)
    widest = widest_block(block_bins)
    for sample in prange(samples):
        allowed = np.ones(k, np.bool_)
        gathered = np.empty((widest, k * (k + 1) // 2))
        for founder in range(k):
            slot = index[sample, founder]
            if slot < 0:
                continue
            allowed[founder] = False
            first, other = pair_lists(allowed)
            forward(flat, block_start, block_haps, block_bins, rows, first, other, penalty,
                    sample, excluded[slot], second[sample, founder], unused, True, False,
                    gathered)
            allowed[founder] = True
    return excluded, second


@njit(cache=True)
def region_source(index, second, pairs, sample, a, b, k):
    """Where G^{-F} of one sample lives, F = {a, b} (a < b; -1 = absent).

    0: the full forward; 1: a stored single exclusion (slot); 2: recompute
    (P \\ {a, b} has a maximiser set covered by F); 3: P \\ F is empty.
    """
    removed = (1 if a >= 0 else 0) + (1 if b >= 0 else 0)
    if removed == k:
        return 3, -1
    if a < 0:
        return 0, -1
    if b < 0:
        slot = index[sample, a]
        return (1, slot) if slot >= 0 else (0, -1)
    slot_a = index[sample, a]
    slot_b = index[sample, b]
    if slot_a >= 0:
        return (2, -1) if second[sample, a, b] else (1, slot_a)
    if slot_b >= 0:
        return (2, -1) if second[sample, b, a] else (1, slot_b)
    return (2, -1) if pairs[sample, a, b] else (0, -1)


@njit(cache=True)
def _gather_move(flat, base, haps, bins, r1, r2, pair, rows, block, removed_a, removed_b,
                 gathered):
    """gathered[t, z]: the emissions of a move's new states at every bin t of one block, in
    _two_region's state order: (r1, x) [and (r2, x)] for every retained row x, then (r1, r1)
    [, (r1, r2), (r2, r2)] (EmissionLayout values, base: the block's first value of the sample).
    Returns the number of new states."""
    k = rows.shape[0]
    z = 0
    for x in range(k):
        if x == removed_a or x == removed_b:
            continue
        source = rows[x, block]
        at = base + (r1 * haps + source) * bins
        for marker in range(bins):
            gathered[marker, z] = flat[at + marker]
        z += 1
        if pair:
            at = base + (r2 * haps + source) * bins
            for marker in range(bins):
                gathered[marker, z] = flat[at + marker]
            z += 1
    at = base + (r1 * haps + r1) * bins
    for marker in range(bins):
        gathered[marker, z] = flat[at + marker]
    z += 1
    if pair:
        at = base + (r1 * haps + r2) * bins
        other = base + (r2 * haps + r2) * bins
        for marker in range(bins):
            gathered[marker, z] = flat[at + marker]
            gathered[marker, z + 1] = flat[other + marker]
        z += 2
    return z


@njit(cache=True)
def _two_region(flat, block_start, block_haps, block_bins, rows, penalty, sample, region,
                has_region, removed_a, removed_b, new_paths, new_first, new_second, values_buffer,
                gathered):
    pair = new_second >= 0
    a_value = -np.inf
    count = 0
    t = 0
    for block in range(len(block_start)):
        haps, bins = block_haps[block], block_bins[block]
        r1 = new_paths[new_first, block]
        r2 = new_paths[new_second, block] if pair else r1
        count = _gather_move(flat, block_start[block] + sample * haps * haps * bins, haps, bins,
                             r1, r2, pair, rows, block, removed_a, removed_b, gathered)
        for marker in range(bins):
            if t == 0:
                a_value = region[0] if has_region else -np.inf
                for z in range(count):
                    values_buffer[z] = gathered[marker, z]
            else:
                best = max_range(values_buffer, 0, count, a_value)
                if has_region:
                    a_value = best + (region[t] - region[t - 1])
                switch = best - penalty
                for z in range(count):
                    value = values_buffer[z]
                    values_buffer[z] = (value if value > switch else switch) + gathered[marker, z]
            t += 1
    return max_range(values_buffer, 0, count, a_value)


@njit(cache=True, parallel=True, nogil=True)
def move_bounds(flat, block_start, block_haps, block_bins, rows, penalty, total_bins, forward_values,
                index, excluded, second, pairs, new_rows, first_removed, second_removed,
                sample_ids):
    """Per-sample two-region bounds of one move, for the samples sample_ids.

    The move replaces the panel rows first_removed < second_removed (or -1) by the
    one or two local paths new_rows (rows of a 2-d array).
    """
    k = rows.shape[0]
    a = first_removed
    b = second_removed
    second_new = 1 if new_rows.shape[0] > 1 else -1
    result = np.empty(len(sample_ids))
    widest = widest_block(block_bins)
    for position in prange(len(sample_ids)):
        sample = sample_ids[position]
        values_buffer = np.empty(2 * k + 3)
        gathered = np.empty((widest, 2 * k + 3))
        source, slot = region_source(index, second, pairs, sample, a, b, k)
        if source == 1:
            region = excluded[slot]
        elif source == 2:
            region = np.empty(total_bins)
            allowed = np.ones(k, np.bool_)
            allowed[a] = False
            allowed[b] = False
            first, other = pair_lists(allowed)
            forward(flat, block_start, block_haps, block_bins, rows, first, other, penalty,
                    sample, region, np.zeros(1, np.bool_), np.zeros((1, 1), np.bool_), False,
                    False, np.empty((widest, len(first))))
        else:
            region = forward_values[sample]
        result[position] = _two_region(
            flat, block_start, block_haps, block_bins, rows, penalty, sample, region, source != 3,
            a, b, new_rows, 0, second_new, values_buffer, gathered)
    return result


@njit(cache=True)
def _advance_block(flat, base, haps, bins, r1, r2, pair, rows, block, removed_a, removed_b,
                   region, has_region, t0, penalty, state, count, gathered):
    """Advance a move's two-region DP over the bins of one block (_two_region's operations in
    the same order). state[0]: the bound on the retained states; state[1:count + 1]: the new
    states; t0: global index of the block's first bin; base: the block's first value of the
    sample in the EmissionLayout."""
    _gather_move(flat, base, haps, bins, r1, r2, pair, rows, block, removed_a, removed_b,
                 gathered)
    for marker in range(bins):
        t = t0 + marker
        if t == 0:
            state[0] = region[0] if has_region else -np.inf
            for z in range(count):
                state[1 + z] = gathered[marker, z]
        else:
            best = max_range(state, 1, count + 1, state[0])
            if has_region:
                state[0] = best + (region[t] - region[t - 1])
            switch = best - penalty
            for z in range(count):
                value = state[1 + z]
                state[1 + z] = (value if value > switch else switch) + gathered[marker, z]


@njit(cache=True)
def _retreat_block(flat, base, haps, bins, r1, r2, pair, rows, block, removed_a, removed_b,
                   region, has_region, t0, penalty, beta, count, gathered, work):
    """Carry the DP's backward vector over the bins of one block, last bin first.

    The two-region DP is max-plus linear in its state v, so the value over the remaining bins
    is max_j v[j] + beta[j]. On entry beta applies to the state after the block's last bin, on
    exit to the state before its first bin: beta'[0] = c, beta'[z] = max(c, e_z + beta[z]) with
    c = max(region increment + beta[0], max_z (e_z + beta[z]) - penalty)."""
    _gather_move(flat, base, haps, bins, r1, r2, pair, rows, block, removed_a, removed_b,
                 gathered)
    for marker in range(bins - 1, -1, -1):
        t = t0 + marker
        if t == 0:
            break
        for z in range(count):
            work[z] = gathered[marker, z] + beta[1 + z]
        carry = max_range(work, 0, count, -np.inf) - penalty
        if has_region:
            value = (region[t] - region[t - 1]) + beta[0]
            if value > carry:
                carry = value
            beta[0] = carry
        for z in range(count):
            value = work[z]
            beta[1 + z] = value if value > carry else carry


@njit(cache=True)
def _meet(state, beta, count):
    """The move's bound from the state before a point and the backward vector after it."""
    best = -np.inf
    for z in range(count + 1):
        value = state[z] + beta[z]
        if value > best:
            best = value
    return best


@njit(cache=True, parallel=True, nogil=True)
def structural_bounds(flat, block_start, block_haps, block_bins, rows, penalty, total_bins,
                      forward_values, index, excluded, second, pairs, offsets, active, local_moves,
                      onesided_table, splice_table, columns):
    """Per-sample two-region bounds of the local, onesided and splice moves (samples x columns).

    Each value is move_bounds' recursion evaluated in two halves: the DP is max-plus linear in
    its state, so a move's bound is max_j (state before the part where the move differs from
    a reference)[j] + (backward vector after that part)[j], equal in exact arithmetic (float
    rounding differs by the summation order only). Per founder f, one forward and one backward
    pass with new path f give the prefix states and suffix vectors of every local move of f
    (it differs from f at one block only, which is then advanced from the prefix state); one
    backward pass per ordered pair (f, g) with new path g gives the suffixes of every onesided
    move f[:t] + g[t:] (its prefix is f's); one forward and one backward pass per unordered
    pair give every splice. Regions G^{-F} are chosen as in move_bounds (region_source).
    local_moves rows: (column, founder, block, new local row); onesided_table[f, g, t] and
    splice_table[f, g, t] (f < g): column of that move or -1; active[f]: f has local or
    onesided moves here. O(m k (k^2 + sum_b K_b)) per sample instead of move_bounds'
    O(m k B (k^2 + sum_b K_b)).
    """
    k, blocks = rows.shape
    samples = forward_values.shape[0]
    result = np.empty((samples, columns))
    onesided_any = np.zeros((k, k), np.bool_)
    splice_any = np.zeros((k, k), np.bool_)
    for f in range(k):
        for g in range(k):
            for t in range(blocks):
                if onesided_table[f, g, t] >= 0:
                    onesided_any[f, g] = True
                if splice_table[f, g, t] >= 0:
                    splice_any[f, g] = True
    width = 2 * k + 2
    widest = widest_block(block_bins)
    for sample in prange(samples):
        gathered = np.empty((widest, width))
        work = np.empty(width)
        state = np.empty(width)
        beta = np.empty(width)
        prefix = np.empty((k, blocks, k + 1))
        suffix = np.empty((k, blocks, k + 1))
        pair_prefix = np.empty((blocks, width))
        pair_region = np.empty(total_bins)
        pair_gathered = np.empty((widest, k * (k + 1) // 2))
        # The sample's first value of every block in the EmissionLayout.
        base = np.empty(blocks, np.int64)
        for block in range(blocks):
            base[block] = block_start[block] + sample * block_haps[block] ** 2 * block_bins[block]
        allowed = np.ones(k, np.bool_)
        unused_singles = np.zeros(1, np.bool_)
        unused_pairs = np.zeros((1, 1), np.bool_)
        has_single = k > 1
        for f in range(k):
            if not active[f]:
                continue
            slot = index[sample, f]
            region = excluded[slot] if slot >= 0 else forward_values[sample]
            for block in range(blocks):
                _advance_block(flat, base[block], block_haps[block], block_bins[block],
                               rows[f, block], rows[f, block], False, rows, block, f, -1, region,
                               has_single, offsets[block], penalty, state, k, gathered)
                for z in range(k + 1):
                    prefix[f, block, z] = state[z]
            for z in range(k + 1):
                beta[z] = 0.0
            if not has_single:
                beta[0] = -np.inf
            for z in range(k + 1):
                suffix[f, blocks - 1, z] = beta[z]
            for block in range(blocks - 1, 0, -1):
                _retreat_block(flat, base[block], block_haps[block], block_bins[block],
                               rows[f, block], rows[f, block], False, rows, block, f, -1, region,
                               has_single, offsets[block], penalty, beta, k, gathered, work)
                for z in range(k + 1):
                    suffix[f, block - 1, z] = beta[z]
        for i in range(local_moves.shape[0]):
            column = local_moves[i, 0]
            f = local_moves[i, 1]
            block = local_moves[i, 2]
            row = local_moves[i, 3]
            slot = index[sample, f]
            region = excluded[slot] if slot >= 0 else forward_values[sample]
            if block > 0:
                for z in range(k + 1):
                    state[z] = prefix[f, block - 1, z]
            _advance_block(flat, base[block], block_haps[block], block_bins[block], row, row,
                           False, rows, block, f, -1, region, has_single, offsets[block], penalty,
                           state, k, gathered)
            result[sample, column] = _meet(state, suffix[f, block], k)
        for f in range(k):
            slot = index[sample, f]
            region = excluded[slot] if slot >= 0 else forward_values[sample]
            for g in range(k):
                if not onesided_any[f, g]:
                    continue
                for z in range(k + 1):
                    beta[z] = 0.0
                if not has_single:
                    beta[0] = -np.inf
                for block in range(blocks - 1, 0, -1):
                    _retreat_block(flat, base[block], block_haps[block], block_bins[block],
                                   rows[g, block], rows[g, block], False, rows, block, f, -1,
                                   region, has_single, offsets[block], penalty, beta, k, gathered,
                                   work)
                    column = onesided_table[f, g, block]
                    if column >= 0:
                        result[sample, column] = _meet(prefix[f, block - 1], beta, k)
        count = 2 * (k - 2) + 3
        for f in range(k):
            for g in range(f + 1, k):
                if not splice_any[f, g]:
                    continue
                source, slot = region_source(index, second, pairs, sample, f, g, k)
                has_pair = source != 3
                if source == 1:
                    region = excluded[slot]
                elif source == 2:
                    allowed[f] = False
                    allowed[g] = False
                    first, other = pair_lists(allowed)
                    forward(flat, block_start, block_haps, block_bins, rows, first, other,
                            penalty, sample, pair_region, unused_singles, unused_pairs, False,
                            False, pair_gathered)
                    allowed[f] = True
                    allowed[g] = True
                    region = pair_region
                else:
                    region = forward_values[sample]
                for block in range(blocks - 1):
                    _advance_block(flat, base[block], block_haps[block], block_bins[block],
                                   rows[f, block], rows[g, block], True, rows, block, f, g, region,
                                   has_pair, offsets[block], penalty, state, count, gathered)
                    for z in range(count + 1):
                        pair_prefix[block, z] = state[z]
                for z in range(count + 1):
                    beta[z] = 0.0
                if not has_pair:
                    beta[0] = -np.inf
                for block in range(blocks - 1, 0, -1):
                    _retreat_block(flat, base[block], block_haps[block], block_bins[block],
                                   rows[g, block], rows[f, block], True, rows, block, f, g, region,
                                   has_pair, offsets[block], penalty, beta, count, gathered, work)
                    column = splice_table[f, g, block]
                    if column >= 0:
                        result[sample, column] = _meet(pair_prefix[block - 1], beta, count)
    return result


@njit(cache=True)
def _region_block(gathered, bins, removed, has_region, region, t0, penalty, a_value, state):
    """Advance one single-path two-region DP over the bins of one block.

    The operations are _two_region's, in the same order. state[x]: the new state of the path
    with panel row x, state[k]: with itself (emissions from _gather_flat); a removed row's
    state does not exist and is held at -inf, so it never reaches a maximum.
    """
    width = gathered.shape[1]
    for marker in range(bins):
        t = t0 + marker
        if t == 0:
            a_value = region[0] if has_region else -np.inf
            for z in range(width):
                state[z] = gathered[marker, z]
        else:
            best = max_range(state, 0, width, a_value)
            if has_region:
                a_value = best + (region[t] - region[t - 1])
            switch = best - penalty
            for z in range(width):
                value = state[z]
                state[z] = (value if value > switch else switch) + gathered[marker, z]
        if removed >= 0:
            state[removed] = -np.inf
    return a_value


@njit(cache=True)
def _gather_flat(flat, base, haps, bins, r1, rows, block, gathered):
    """gathered[t, x] = emission of (r1, panel row x) and gathered[t, k] = (r1, r1) for every bin t
    of one block (EmissionLayout values, base: the block's first value of the sample): the new
    states' emissions, contiguous per bin."""
    k = rows.shape[0]
    for x in range(k):
        at = base + (r1 * haps + rows[x, block]) * bins
        for marker in range(bins):
            gathered[marker, x] = flat[at + marker]
    at = base + (r1 * haps + r1) * bins
    for marker in range(bins):
        gathered[marker, k] = flat[at + marker]


@njit(cache=True, parallel=True, nogil=True)
def birth_bounds(flat, block_start, block_haps, block_bins, rows, penalty, forward_values, paths,
                 common, offsets, start, stop, parts):
    """Per-sample upper bounds on V_s(P + c) (the two-region bound of birth(c), F empty) for the
    sorted paths start..stop-1 (emissions: an EmissionLayout's arrays).

    common[i] counts the leading blocks path i shares with path i-1: the DP state after a
    shared prefix is reused (as in proposals.prefix_scores), so the work follows the
    distinct prefixes and the values equal the per-move recursion exactly. Each task is one
    sample and one contiguous part of the path range.
    """
    k, blocks = rows.shape
    samples = forward_values.shape[0]
    count = stop - start
    widest = widest_block(block_bins)
    result = np.empty((samples, count))
    for task in prange(samples * parts):
        sample = task // parts
        part = task % parts
        lo = start + (count * part) // parts
        hi = start + (count * (part + 1)) // parts
        if lo >= hi:
            continue
        prefix_a = np.empty(blocks + 1)
        prefix_v = np.empty((blocks + 1, k + 1))
        buffer = np.empty(k + 1)
        gathered = np.empty((widest, k + 1))
        region = forward_values[sample]
        for i in range(lo, hi):
            shared = 0 if i == lo else min(common[i], blocks - 1)
            for block in range(shared, blocks):
                haps, bins = block_haps[block], block_bins[block]
                _gather_flat(flat, block_start[block] + sample * haps * haps * bins, haps, bins,
                             paths[i, block], rows, block, gathered)
                if block == 0:
                    a_value = -np.inf
                else:
                    a_value = prefix_a[block]
                    for z in range(k + 1):
                        buffer[z] = prefix_v[block, z]
                a_value = _region_block(gathered, bins, -1, True, region,
                                        offsets[block], penalty, a_value, buffer)
                prefix_a[block + 1] = a_value
                for z in range(k + 1):
                    prefix_v[block + 1, z] = buffer[z]
            result[sample, i - start] = max_range(prefix_v[blocks], 0, k + 1, prefix_a[blocks])
    return result


@njit(cache=True)
def _region_retreat(gathered, bins, removed, has_region, region, t0, penalty, beta, work):
    """Carry the backward vector of _region_block's DP over one block, last bin first: on entry
    beta[0] (retained states) and beta[1:] (the new states, _region_block's layout) apply to the
    state after the block's last bin, on exit to the state before its first bin (see
    _retreat_block; the removed row's state does not exist)."""
    width = gathered.shape[1]
    for marker in range(bins - 1, -1, -1):
        t = t0 + marker
        if t == 0:
            break
        for z in range(width):
            work[z] = gathered[marker, z] + beta[1 + z]
        if removed >= 0:
            work[removed] = -np.inf
        carry = max_range(work, 0, width, -np.inf) - penalty
        if has_region:
            value = (region[t] - region[t - 1]) + beta[0]
            if value > carry:
                carry = value
            beta[0] = carry
        for z in range(width):
            value = work[z]
            beta[1 + z] = value if value > carry else carry


@njit(cache=True, parallel=True, nogil=True)
def replace_bounds(flat, block_start, block_haps, block_bins, rows, penalty, index, excluded,
                   offsets, split, prefixes, prefix_common, prefix_of, suffixes, suffix_common,
                   suffix_of, width, wanted):
    """Per-sample two-region bounds of replace(f, c) for every row f with a single cover in the
    sample (index >= 0) and every path c of one group: out[s, i, j] for the j-th such row of
    sample s (rows in increasing order).

    The paths are given as their distinct prefixes (blocks < split; sorted; prefix_common = the
    leading blocks shared with the previous prefix) and distinct suffixes (blocks >= split; sorted
    from the last block; suffix_common = the trailing blocks shared with the previous suffix). The
    DP is max-plus linear in its state, so a path's bound is max_j F[j] + beta[j] for the forward
    state F after its prefix and the backward vector beta of its suffix: every distinct prefix and
    suffix is run once, reusing the state over shared leading (trailing) blocks. Equal to
    _two_region's recursion in exact arithmetic; float rounding differs by the summation order.
    Only the rows with wanted[f] are bounded: each row's DP is independent of the others, so
    their values are unchanged, and the other rows get -inf (their replacements are not searched).
    Emissions: an EmissionLayout's arrays.
    """
    k, blocks = rows.shape
    samples = index.shape[0]
    count = len(prefix_of)
    widest = widest_block(block_bins)
    out = np.empty((samples, count, width))
    has_region = k > 1
    for sample in prange(samples):
        regions = np.empty(k, np.int64)
        columns = np.empty(k, np.int64)
        nreg = 0
        column = 0
        for founder in range(k):
            if index[sample, founder] >= 0:
                if wanted[founder]:
                    regions[nreg] = founder
                    columns[nreg] = column
                    nreg += 1
                else:
                    for i in range(count):
                        out[sample, i, column] = -np.inf
                column += 1
        if nreg == 0:
            continue
        size = k + 2                       # the retained-state bound, then k + 1 new states
        gathered = np.empty((widest, k + 1))
        buffer = np.empty(size)
        work = np.empty(k + 1)
        stack = np.empty((split + 1, nreg, size))
        forward_states = np.empty((prefixes.shape[0], nreg, size))
        for p in range(prefixes.shape[0]):
            shared = 0 if p == 0 else prefix_common[p]
            for block in range(shared, split):
                haps, bins = block_haps[block], block_bins[block]
                _gather_flat(flat, block_start[block] + sample * haps * haps * bins, haps, bins,
                             prefixes[p, block], rows, block, gathered)
                for rg in range(nreg):
                    region = excluded[index[sample, regions[rg]]]
                    a_value = -np.inf
                    if block > 0:
                        a_value = stack[block, rg, 0]
                        for z in range(k + 1):
                            buffer[z] = stack[block, rg, 1 + z]
                    a_value = _region_block(gathered, bins, regions[rg], has_region,
                                            region, offsets[block], penalty, a_value, buffer)
                    stack[block + 1, rg, 0] = a_value
                    for z in range(k + 1):
                        stack[block + 1, rg, 1 + z] = buffer[z]
            for rg in range(nreg):
                for z in range(size):
                    forward_states[p, rg, z] = stack[split, rg, z]
        after = np.empty((blocks - split + 1, nreg, size))
        for rg in range(nreg):
            after[blocks - split, rg, 0] = 0.0 if has_region else -np.inf
            for z in range(k + 1):
                after[blocks - split, rg, 1 + z] = 0.0
        backward = np.empty((suffixes.shape[0], nreg, size))
        for q in range(suffixes.shape[0]):
            shared = 0 if q == 0 else suffix_common[q]
            for block in range(blocks - 1 - shared, split - 1, -1):
                haps, bins = block_haps[block], block_bins[block]
                _gather_flat(flat, block_start[block] + sample * haps * haps * bins, haps, bins,
                             suffixes[q, block - split], rows, block, gathered)
                for rg in range(nreg):
                    region = excluded[index[sample, regions[rg]]]
                    for z in range(size):
                        buffer[z] = after[block - split + 1, rg, z]
                    _region_retreat(gathered, bins, regions[rg], has_region, region,
                                    offsets[block], penalty, buffer, work)
                    for z in range(size):
                        after[block - split, rg, z] = buffer[z]
            for rg in range(nreg):
                for z in range(size):
                    backward[q, rg, z] = after[0, rg, z]
        for i in range(count):
            p = prefix_of[i]
            q = suffix_of[i]
            for rg in range(nreg):
                for z in range(size):
                    work_value = forward_states[p, rg, z] + backward[q, rg, z]
                    buffer[z] = work_value
                out[sample, i, columns[rg]] = max_range(buffer, 0, size, -np.inf)
    return out


@njit(cache=True, parallel=True, nogil=True)
def pool_sums(out, births, index):
    """Summed bounds [birth(c), replace(0, c) .. replace(k-1, c)] per path, summed over samples
    in sample order (one thread per path): replace(f, c) takes birth(c)'s value where f has no
    single cover (G^{-f} = G; valid since (P \\ f) + c is a subset of P + c)."""
    samples, count, _ = out.shape
    k = index.shape[1]
    result = np.empty((count, k + 1))
    for i in prange(count):
        totals = np.zeros(k + 1)
        for sample in range(samples):
            birth = births[sample, i]
            totals[0] += birth
            rg = 0
            for founder in range(k):
                if index[sample, founder] >= 0:
                    totals[1 + founder] += out[sample, i, rg]
                    rg += 1
                else:
                    totals[1 + founder] += birth
        for column in range(k + 1):
            result[i, column] = totals[column]
    return result


def leading_shared(rows):
    """For sorted distinct rows: the number of leading columns each shares with the previous."""
    shared = np.zeros(len(rows), dtype=np.int64)
    if len(rows) > 1 and rows.shape[1]:
        equal = rows[1:] == rows[:-1]
        shared[1:] = np.where(equal.all(axis=1), rows.shape[1], np.argmin(equal, axis=1))
    return shared


def _sorted_rows(rows):
    """Lexicographic order of integer rows (first column first), the sorted rows, and for each
    sorted row the number of leading columns it shares with the previous one (all if equal)."""
    order = np.lexsort(rows.T[::-1]) if rows.shape[1] else np.arange(len(rows))
    ordered = rows[order]
    return order, ordered, leading_shared(ordered)


def _unique_rows(rows):
    """np.unique(rows, axis=0, return_inverse=True) for integer rows by one lexicographic sort:
    the distinct rows in lexicographic order and each row's index among them (int64)."""
    order, ordered, shared = _sorted_rows(rows)
    first = shared < rows.shape[1]
    first[:1] = True
    inverse = np.empty(len(rows), dtype=np.int64)
    inverse[order] = np.cumsum(first) - 1
    return np.ascontiguousarray(ordered[first]), inverse


def _best_split(paths):
    """The block split t (1 <= t <= B) with the fewest distinct prefixes of lengths 1..t plus
    distinct suffixes of lengths 1..B - t: the block passes of replace_bounds.

    In lexicographic order the rows sharing a prefix of length d are consecutive, so the number
    of distinct prefixes of length d is 1 plus the number of sorted rows that share fewer than
    d leading columns with their predecessor (suffixes: the same on the reversed rows)."""
    blocks = paths.shape[1]
    shared = _sorted_rows(paths)[2][1:]
    prefixes = [1 + int(np.count_nonzero(shared < depth)) for depth in range(1, blocks + 1)]
    shared = _sorted_rows(paths[:, ::-1])[2][1:]
    suffixes = [1 + int(np.count_nonzero(shared < depth)) for depth in range(1, blocks)]
    costs = [sum(prefixes[:split]) + sum(suffixes[:blocks - split]) for split in range(1, blocks + 1)]
    return 1 + int(np.argmin(costs))


class PanelForwards:
    """Exclusion forwards of one panel; shared by every bound of one pass.

    layout: the search's EmissionLayout; rows: int64[K, n] the panel's local rows (row order
    matters). forward float64[S, T]: G; index int64[S, K]: slot of each (sample, row) whose
    single exclusion was recomputed (-1: G^{-f} = G); excluded float64[slots, T]: those
    G^{-f}; second, pairs: the covers that route two-row exclusions (region_source).
    births: per-sample birth bounds by local path (bytes key), shared between the count step
    and the pool bounds of this panel (and, by the search, between panels of one path set).
    """

    def __init__(self, layout, rows, penalty, total_bins):
        self.layout = layout
        self.rows = np.ascontiguousarray(rows, dtype=np.int64)
        self.penalty = float(penalty)
        self.total_bins = int(total_bins)
        self.forward, singles, self.pairs = forward_covers(
            *layout.arrays(), self.rows, self.penalty, self.total_bins, layout.samples)
        index = np.full(singles.shape, -1, dtype=np.int64)
        flat = np.flatnonzero(singles.ravel())
        index.ravel()[flat] = np.arange(len(flat))
        self.index = index
        self.excluded, self.second = single_exclusions(
            *layout.arrays(), self.rows, self.penalty, self.total_bins, index, len(flat))
        # Mean number of rows per sample whose exclusion was recomputed.
        self.recomputed_per_sample = len(flat) / max(1, singles.shape[0])
        self.births = {}

    def removal_scores(self):
        """Exact per-sample V_s(P minus f) for every row f (samples x k)."""
        last = self.forward.shape[1] - 1
        values = np.repeat(self.forward[:, last:last + 1], self.index.shape[1], axis=1)
        mask = self.index >= 0
        values[mask] = self.excluded[self.index[mask], last]
        return values

    def birth_values(self, paths, offsets):
        """Per-sample birth bounds (samples x paths) of local paths, by birth_bounds over the
        sorted distinct paths not yet known for this panel (self.births keeps them by path, for
        the count step and the pool bounds of this panel)."""
        paths = np.asarray(paths, dtype=np.int64)
        samples = self.forward.shape[0]
        keys = [row.tobytes() for row in paths]
        unique = list(dict.fromkeys(keys))
        missing = [key for key in unique if key not in self.births]
        if missing:
            todo = np.asarray([np.frombuffer(key, dtype=np.int64) for key in missing])
            count, blocks = todo.shape
            order = np.lexsort(todo[:, ::-1].T)
            ordered = np.ascontiguousarray(todo[order])
            common = leading_shared(ordered)
            threads = max(1, numba.get_num_threads())
            offsets = np.ascontiguousarray(offsets, dtype=np.int64)
            for start in range(0, count, _BIRTH_CHUNK):
                stop = min(count, start + _BIRTH_CHUNK)
                parts = max(1, min(stop - start, -(-2 * threads // samples)))
                values = birth_bounds(*self.layout.arrays(), self.rows, self.penalty,
                                      self.forward, ordered, common, offsets, start, stop, parts)
                for column, position in enumerate(order[start:stop].tolist()):
                    self.births[missing[position]] = values[:, column].copy()
        if not keys:
            return np.empty((samples, 0))
        return np.stack([self.births[key] for key in keys], axis=1)

    def pool_bounds(self, paths, offsets, keep, replaced=None):
        """Summed bounds of [birth(c), replace(0, c) .. replace(k-1, c)] per pool path (rows of
        paths), as one (len(paths), k + 1) array, and the per-sample bounds of the replacements
        whose sum keep(sums) retains ({(path, column): vector}).

        Births come from birth_values (exact, shared with the count step); replacements from
        replace_bounds over groups of paths, split where the distinct prefixes and suffixes
        need the fewest block passes, with at most about 2**26 bytes of per-sample values.
        replaced: if given, only replacements of these rows are bounded; a row outside it sums
        to -inf wherever it has a single cover (lower_bounds then excludes its moves)."""
        k, blocks = self.rows.shape
        samples = self.forward.shape[0]
        paths = np.ascontiguousarray(paths, dtype=np.int64)
        result = np.empty((len(paths), k + 1))
        kept = {}
        if not len(paths):
            return result, kept
        births = self.birth_values(paths, offsets)
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        wanted = np.ones(k, dtype=np.bool_)
        if replaced is not None:
            wanted[:] = False
            wanted[list(replaced)] = True
        singles = self.index >= 0
        width = max(1, int(singles.sum(axis=1).max()))
        slot = np.where(singles, np.cumsum(singles, axis=1) - 1, -1)
        rows_of = np.arange(samples)
        step = max(256, _REPLACE_CHUNK_BYTES // (8 * samples * width))
        for start in range(0, len(paths), step):
            group = paths[start:start + step]
            split = _best_split(group)
            prefixes, prefix_of = _unique_rows(group[:, :split])
            prefix_common = leading_shared(prefixes)
            tails, suffix_of = _unique_rows(group[:, split:][:, ::-1])
            suffix_common = leading_shared(tails)
            suffixes = np.ascontiguousarray(tails[:, ::-1])
            out = replace_bounds(*self.layout.arrays(), self.rows, self.penalty, self.index,
                                 self.excluded, offsets, split, np.ascontiguousarray(prefixes),
                                 prefix_common, prefix_of, suffixes, suffix_common, suffix_of,
                                 width, wanted)
            part = np.ascontiguousarray(births[:, start:start + len(group)])
            sums = pool_sums(out, part, self.index)
            result[start:start + len(group)] = sums
            for path, column in zip(*np.nonzero(keep(sums))):
                if column:
                    founder = column - 1
                    vector = part[:, path].copy()
                    single = slot[:, founder] >= 0
                    vector[single] = out[rows_of[single], path, slot[single, founder]]
                    kept[(start + int(path), int(column))] = vector
        return result, kept

    def replacement_scores(self, founder, new_row, trial_rows, removal, sample_bounds=None):
        """Exact per-sample scores V_s(P') of the panel P' = P with row ``founder`` replaced by the
        local path ``new_row`` (``trial_rows``: P' as int64[K, n]; ``removal``: removal_scores()).

        P minus f is contained in P', so lower_s = V_s(P minus f) <= V_s(P') <= U_s (the move's
        two-region bound). V_s(P') is computed (score_samples of a sample subset) only where U_s
        exceeds lower_s; elsewhere the new path's states never exceed the running maximum of
        P minus f, so V_s(P') = lower_s exactly. sample_bounds (optional): the move's per-sample
        bounds from the bulk bounds of its neighbourhood, which equal U_s or exceed it, up to float
        rounding (a 1e-6 relative margin covers it); U_s is recomputed only where they leave the
        comparison open, so the scored samples are those of the full comparison. Returns
        (scores, samples scored). At a one-path panel P minus f is empty (lower_s = -inf), so
        every sample is scored.
        """
        lower = removal[:, founder]
        tolerance = np.full(len(lower), -np.inf)
        finite = np.isfinite(lower)
        tolerance[finite] = lower[finite] + 1e-9 * (1.0 + np.abs(lower[finite]))
        if sample_bounds is None:
            candidates = np.arange(len(lower), dtype=np.int64)
        else:
            margin = 1e-6 * (1.0 + np.abs(sample_bounds) + np.abs(lower))
            candidates = np.flatnonzero(sample_bounds + margin > tolerance).astype(np.int64)
        values = lower.copy()
        if not len(candidates):
            return values, 0
        upper = move_bounds(*self.layout.arrays(), self.rows, self.penalty, self.total_bins,
                            self.forward, self.index, self.excluded, self.second, self.pairs,
                            np.ascontiguousarray(np.reshape(new_row, (1, -1)), dtype=np.int64),
                            int(founder), -1, candidates)
        need = candidates[upper > tolerance[candidates]]
        if len(need):
            values[need] = score_samples(*self.layout.arrays(),
                                         np.ascontiguousarray(trial_rows, dtype=np.int64),
                                         self.penalty, need)
        return values, len(need)


__all__ = ["PanelForwards", "birth_bounds", "column_sums", "forward_covers", "leading_shared",
           "move_bounds", "pool_sums", "region_source", "replace_bounds", "single_exclusions",
           "structural_bounds"]
