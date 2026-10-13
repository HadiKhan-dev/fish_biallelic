"""Certified same-count panel search: exclusion forwards and two-region bounds.

Join objective (``panel_scoring.score_emissions``): per sample, the maximum over
sequences of unordered panel-path pairs of binned emissions minus a uniform
penalty ``pen`` per state change; LL = sum over samples. Same-count moves are
compared by the join score -2 LL; the panel size is decided by ``panel_count``.

Nothing here accepts a panel. The kernels give, per sample,

* exclusion forwards: the forward maxima G^{-F}(t) of the panel without the
  paths in F (|F| <= 2). G^{-F} = G at every bin iff at every bin some
  maximising state avoids F (induction over the Potts recursion: surviving
  states see the same switch base). The forward records, per bin, the vertex
  covers of size <= 2 of its maximiser set, so G^{-F} is recomputed only where
  F covers the maximisers at some bin; this is exact, also under ties. The count
  step reads V_n(P minus f) for every row f from them;
* the two-region bound: an upper bound on the value of the panel with the
  paths F removed and new paths H (|H| <= 2) added. Runs through the unchanged
  states A (pairs of retained paths) are bounded by telescoped increments of
  G^{-F} (the A-only programme can take its best prefix and switch in, so
  entering A is free); runs through the new states N (pairs containing a new
  path) follow the exact Potts dynamics (entering N or switching inside N
  costs pen). The DP is max-plus linear in its state, so a structural move
  (local, onesided, splice) is bounded in two halves: the state before the part
  where the move departs from a reference path, plus the backward vector after
  it (structural_bounds: per founder, and per pair of founders, one forward and
  one backward pass serve every block and boundary). The halves are summed in
  another order than one forward pass, so values agree up to float rounding,
  which the exclusion's float margin covers.

A move is excluded only when its bound proves that it cannot raise the join
likelihood. Per-sample values are computed by one thread each and summed in
sample order, so every bound is independent of the thread count. Costs per
sample: forward O(m k^2); a bound O(m k); all structural moves O(m k (k^2 +
sum_b K_b)).
"""
import numba
import numpy as np
from numba import njit, prange
from numba.typed import List

# Same-count moves; birth(c) is bounded alongside replace(f, c) (pool_bounds) but never
# accepted here: count changes are decided by panel_count.
KINDS = ('replace', 'local', 'onesided', 'splice', 'birth')
REPLACE, LOCAL, ONESIDED, SPLICE, BIRTH = range(5)
_MOVES_PER_CHUNK = 8192
_BIRTH_CHUNK = 4096


@njit(cache=True, inline='always')
def _max_range(values, lo, hi, initial):
    """max(initial, values[lo:hi]) through eight independent running maxima. A maximum is exact,
    so the value equals one sequential scan's; the independent chains let comparisons overlap."""
    m0 = initial
    m1 = m2 = m3 = m4 = m5 = m6 = m7 = -np.inf
    z = lo
    while z + 8 <= hi:
        v = values[z]
        m0 = v if v > m0 else m0
        v = values[z + 1]
        m1 = v if v > m1 else m1
        v = values[z + 2]
        m2 = v if v > m2 else m2
        v = values[z + 3]
        m3 = v if v > m3 else m3
        v = values[z + 4]
        m4 = v if v > m4 else m4
        v = values[z + 5]
        m5 = v if v > m5 else m5
        v = values[z + 6]
        m6 = v if v > m6 else m6
        v = values[z + 7]
        m7 = v if v > m7 else m7
        z += 8
    while z < hi:
        v = values[z]
        m0 = v if v > m0 else m0
        z += 1
    m0 = m1 if m1 > m0 else m0
    m2 = m3 if m3 > m2 else m2
    m4 = m5 if m5 > m4 else m4
    m6 = m7 if m7 > m6 else m6
    m0 = m2 if m2 > m0 else m0
    m4 = m6 if m6 > m4 else m4
    return m4 if m4 > m0 else m0


@njit(cache=True)
def _pair_lists(allowed):
    count = 0
    for a in range(len(allowed)):
        if allowed[a]:
            for b in range(a, len(allowed)):
                if allowed[b]:
                    count += 1
    first = np.empty(count, np.int64)
    second = np.empty(count, np.int64)
    index = 0
    for a in range(len(allowed)):
        if allowed[a]:
            for b in range(a, len(allowed)):
                if allowed[b]:
                    first[index] = a
                    second[index] = b
                    index += 1
    return first, second


@njit(cache=True)
def _record_covers(scores, best, first, second, singles, pairs, record_pairs):
    """Vertex covers of size <= 2 of this bin's maximiser set.

    singles[f]: f lies in every maximising state. pairs[f, j] (f < j): {f, j}
    hits every maximising state while neither is a single cover of this bin.
    """
    z0 = -1
    i0 = -1
    i1 = -1
    count = 0
    for z in range(len(scores)):
        if scores[z] == best:
            a = first[z]
            b = second[z]
            if z0 < 0:
                z0 = z
                i0 = a
                i1 = b if b != a else -1
            else:
                if i0 >= 0 and i0 != a and i0 != b:
                    i0 = -1
                if i1 >= 0 and i1 != a and i1 != b:
                    i1 = -1
            count += 1
    if i0 >= 0:
        singles[i0] = True
    if i1 >= 0:
        singles[i1] = True
    if not record_pairs or count < 2:
        return
    # A size-2 cover hits the first maximiser: x in it, and its partner lies
    # in every maximiser that avoids x.
    for side in range(2):
        x = first[z0] if side == 0 else second[z0]
        if side == 1 and x == first[z0]:
            break
        if x == i0 or x == i1:
            continue
        w0 = -1
        w1 = -1
        started = False
        for z in range(len(scores)):
            if scores[z] == best:
                a = first[z]
                b = second[z]
                if a == x or b == x:
                    continue
                if not started:
                    w0 = a
                    w1 = b if b != a else -1
                    started = True
                else:
                    if w0 >= 0 and w0 != a and w0 != b:
                        w0 = -1
                    if w1 >= 0 and w1 != a and w1 != b:
                        w1 = -1
        if not started:
            continue
        if w0 >= 0 and w0 != x and w0 != i0 and w0 != i1:
            pairs[min(x, w0), max(x, w0)] = True
        if w1 >= 0 and w1 != x and w1 != i0 and w1 != i1:
            pairs[min(x, w1), max(x, w1)] = True


@njit(cache=True)
def _forward(emissions, rows, first, second, penalty, sample, out, singles, pairs,
             record, record_pairs, gathered):
    """Forward maxima of one sample over the given unordered states.

    The recursion and its operation order are the production scorer's. Each block's state
    emissions are first copied into gathered (bins x states), contiguous per bin.
    """
    n = len(first)
    scores = np.empty(n)
    best = -np.inf
    t = 0
    for block in range(len(emissions)):
        values = emissions[block]
        bins = values.shape[3]
        for z in range(n):
            row = rows[first[z], block]
            column = rows[second[z], block]
            for marker in range(bins):
                gathered[marker, z] = values[sample, row, column, marker]
        for marker in range(bins):
            if t == 0:
                for z in range(n):
                    scores[z] = gathered[marker, z]
            else:
                # best: the previous bin's maximum (the scores have not changed since)
                base = best - penalty
                for z in range(n):
                    stay = scores[z]
                    scores[z] = (stay if stay > base else base) + gathered[marker, z]
            best = _max_range(scores, 0, n, -np.inf)
            out[t] = best
            if record:
                _record_covers(scores, best, first, second, singles, pairs,
                               record_pairs)
            t += 1


@njit(cache=True)
def _widest(emissions):
    widest = 1
    for block in range(len(emissions)):
        widest = max(widest, emissions[block].shape[3])
    return widest


@njit(cache=True, parallel=True)
def forward_covers(emissions, rows, penalty, total_bins):
    """G[s, t] and the per-sample single covers and disjoint pair covers."""
    k = rows.shape[0]
    samples = emissions[0].shape[0]
    forward = np.empty((samples, total_bins))
    singles = np.zeros((samples, k), np.bool_)
    pairs = np.zeros((samples, k, k), np.bool_)
    first, second = _pair_lists(np.ones(k, np.bool_))
    widest = _widest(emissions)
    for sample in prange(samples):
        gathered = np.empty((widest, len(first)))
        _forward(emissions, rows, first, second, penalty, sample, forward[sample],
                 singles[sample], pairs[sample], True, True, gathered)
    return forward, singles, pairs


@njit(cache=True, parallel=True)
def single_exclusions(emissions, rows, penalty, total_bins, index, slots):
    """G^{-f} for every (sample, f) with index >= 0, and the single covers of
    each such forward (second[s, f, j]: j covers a maximiser set of P \\ f)."""
    k = rows.shape[0]
    samples = index.shape[0]
    excluded = np.empty((slots, total_bins))
    second = np.zeros((samples, k, k), np.bool_)
    unused = np.zeros((1, 1), np.bool_)
    widest = _widest(emissions)
    for sample in prange(samples):
        allowed = np.ones(k, np.bool_)
        gathered = np.empty((widest, k * (k + 1) // 2))
        for founder in range(k):
            slot = index[sample, founder]
            if slot < 0:
                continue
            allowed[founder] = False
            first, other = _pair_lists(allowed)
            _forward(emissions, rows, first, other, penalty, sample, excluded[slot],
                     second[sample, founder], unused, True, False, gathered)
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
def _gather_move(values, sample, r1, r2, pair, rows, block, removed_a, removed_b, gathered):
    """gathered[t, z]: the emissions of a move's new states at every bin t of one block, in
    _two_region's state order: (r1, x) [and (r2, x)] for every retained row x, then (r1, r1)
    [, (r1, r2), (r2, r2)]. Returns the number of new states."""
    k = rows.shape[0]
    bins = values.shape[3]
    z = 0
    for x in range(k):
        if x == removed_a or x == removed_b:
            continue
        source = rows[x, block]
        for marker in range(bins):
            gathered[marker, z] = values[sample, r1, source, marker]
        z += 1
        if pair:
            for marker in range(bins):
                gathered[marker, z] = values[sample, r2, source, marker]
            z += 1
    for marker in range(bins):
        gathered[marker, z] = values[sample, r1, r1, marker]
    z += 1
    if pair:
        for marker in range(bins):
            gathered[marker, z] = values[sample, r1, r2, marker]
            gathered[marker, z + 1] = values[sample, r2, r2, marker]
        z += 2
    return z


@njit(cache=True)
def _two_region(emissions, rows, penalty, sample, region, has_region, removed_a,
                removed_b, new_paths, new_first, new_second, values_buffer, gathered):
    pair = new_second >= 0
    a_value = -np.inf
    count = 0
    t = 0
    for block in range(len(emissions)):
        values = emissions[block]
        r1 = new_paths[new_first, block]
        r2 = new_paths[new_second, block] if pair else r1
        count = _gather_move(values, sample, r1, r2, pair, rows, block, removed_a, removed_b,
                             gathered)
        for marker in range(values.shape[3]):
            if t == 0:
                a_value = region[0] if has_region else -np.inf
                for z in range(count):
                    values_buffer[z] = gathered[marker, z]
            else:
                best = _max_range(values_buffer, 0, count, a_value)
                if has_region:
                    a_value = best + (region[t] - region[t - 1])
                switch = best - penalty
                for z in range(count):
                    value = values_buffer[z]
                    values_buffer[z] = (value if value > switch else switch) + gathered[marker, z]
            t += 1
    return _max_range(values_buffer, 0, count, a_value)


@njit(cache=True, parallel=True)
def move_bounds(emissions, rows, penalty, total_bins, forward, index, excluded, second,
                pairs, new_rows, first_removed, second_removed, sample_ids):
    """Per-sample two-region bounds of one move, for the samples sample_ids.

    The move replaces the panel rows first_removed < second_removed (or -1) by the
    one or two local paths new_rows (rows of a 2-d array).
    """
    k = rows.shape[0]
    a = first_removed
    b = second_removed
    second_new = 1 if new_rows.shape[0] > 1 else -1
    result = np.empty(len(sample_ids))
    widest = _widest(emissions)
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
            first, other = _pair_lists(allowed)
            _forward(emissions, rows, first, other, penalty, sample, region,
                     np.zeros(1, np.bool_), np.zeros((1, 1), np.bool_), False, False,
                     np.empty((_widest(emissions), len(first))))
        else:
            region = forward[sample]
        result[position] = _two_region(
            emissions, rows, penalty, sample, region, source != 3, a, b,
            new_rows, 0, second_new, values_buffer, gathered)
    return result


@njit(cache=True)
def _advance_block(values, sample, r1, r2, pair, rows, block, removed_a, removed_b, region,
                   has_region, t0, penalty, state, count, gathered):
    """Advance a move's two-region DP over the bins of one block (_two_region's operations in
    the same order). state[0]: the bound on the retained states; state[1:count + 1]: the new
    states; t0: global index of the block's first bin."""
    _gather_move(values, sample, r1, r2, pair, rows, block, removed_a, removed_b, gathered)
    for marker in range(values.shape[3]):
        t = t0 + marker
        if t == 0:
            state[0] = region[0] if has_region else -np.inf
            for z in range(count):
                state[1 + z] = gathered[marker, z]
        else:
            best = _max_range(state, 1, count + 1, state[0])
            if has_region:
                state[0] = best + (region[t] - region[t - 1])
            switch = best - penalty
            for z in range(count):
                value = state[1 + z]
                state[1 + z] = (value if value > switch else switch) + gathered[marker, z]


@njit(cache=True)
def _retreat_block(values, sample, r1, r2, pair, rows, block, removed_a, removed_b, region,
                   has_region, t0, penalty, beta, count, gathered, work):
    """Carry the DP's backward vector over the bins of one block, last bin first.

    The two-region DP is max-plus linear in its state v, so the value over the remaining bins
    is max_j v[j] + beta[j]. On entry beta applies to the state after the block's last bin, on
    exit to the state before its first bin: beta'[0] = c, beta'[z] = max(c, e_z + beta[z]) with
    c = max(region increment + beta[0], max_z (e_z + beta[z]) - penalty)."""
    _gather_move(values, sample, r1, r2, pair, rows, block, removed_a, removed_b, gathered)
    for marker in range(values.shape[3] - 1, -1, -1):
        t = t0 + marker
        if t == 0:
            break
        for z in range(count):
            work[z] = gathered[marker, z] + beta[1 + z]
        carry = _max_range(work, 0, count, -np.inf) - penalty
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


@njit(cache=True, parallel=True)
def structural_bounds(emissions, rows, penalty, total_bins, forward, index, excluded, second,
                      pairs, offsets, active, local_moves, onesided_table, splice_table, columns):
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
    samples = forward.shape[0]
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
    widest = _widest(emissions)
    for sample in prange(samples):
        gathered = np.empty((widest, width))
        work = np.empty(width)
        state = np.empty(width)
        beta = np.empty(width)
        prefix = np.empty((k, blocks, k + 1))
        suffix = np.empty((k, blocks, k + 1))
        pair_prefix = np.empty((blocks, width))
        pair_region = np.empty(total_bins)
        pair_gathered = np.empty((_widest(emissions), k * (k + 1) // 2))
        allowed = np.ones(k, np.bool_)
        unused_singles = np.zeros(1, np.bool_)
        unused_pairs = np.zeros((1, 1), np.bool_)
        has_single = k > 1
        for f in range(k):
            if not active[f]:
                continue
            slot = index[sample, f]
            region = excluded[slot] if slot >= 0 else forward[sample]
            for block in range(blocks):
                _advance_block(emissions[block], sample, rows[f, block], rows[f, block], False,
                               rows, block, f, -1, region, has_single, offsets[block], penalty,
                               state, k, gathered)
                for z in range(k + 1):
                    prefix[f, block, z] = state[z]
            for z in range(k + 1):
                beta[z] = 0.0
            if not has_single:
                beta[0] = -np.inf
            for z in range(k + 1):
                suffix[f, blocks - 1, z] = beta[z]
            for block in range(blocks - 1, 0, -1):
                _retreat_block(emissions[block], sample, rows[f, block], rows[f, block], False,
                               rows, block, f, -1, region, has_single, offsets[block], penalty,
                               beta, k, gathered, work)
                for z in range(k + 1):
                    suffix[f, block - 1, z] = beta[z]
        for i in range(local_moves.shape[0]):
            column = local_moves[i, 0]
            f = local_moves[i, 1]
            block = local_moves[i, 2]
            row = local_moves[i, 3]
            slot = index[sample, f]
            region = excluded[slot] if slot >= 0 else forward[sample]
            if block > 0:
                for z in range(k + 1):
                    state[z] = prefix[f, block - 1, z]
            _advance_block(emissions[block], sample, row, row, False, rows, block, f, -1, region,
                           has_single, offsets[block], penalty, state, k, gathered)
            result[sample, column] = _meet(state, suffix[f, block], k)
        for f in range(k):
            slot = index[sample, f]
            region = excluded[slot] if slot >= 0 else forward[sample]
            for g in range(k):
                if not onesided_any[f, g]:
                    continue
                for z in range(k + 1):
                    beta[z] = 0.0
                if not has_single:
                    beta[0] = -np.inf
                for block in range(blocks - 1, 0, -1):
                    _retreat_block(emissions[block], sample, rows[g, block], rows[g, block],
                                   False, rows, block, f, -1, region, has_single, offsets[block],
                                   penalty, beta, k, gathered, work)
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
                    first, other = _pair_lists(allowed)
                    _forward(emissions, rows, first, other, penalty, sample, pair_region,
                             unused_singles, unused_pairs, False, False, pair_gathered)
                    allowed[f] = True
                    allowed[g] = True
                    region = pair_region
                else:
                    region = forward[sample]
                for block in range(blocks - 1):
                    _advance_block(emissions[block], sample, rows[f, block], rows[g, block], True,
                                   rows, block, f, g, region, has_pair, offsets[block], penalty,
                                   state, count, gathered)
                    for z in range(count + 1):
                        pair_prefix[block, z] = state[z]
                for z in range(count + 1):
                    beta[z] = 0.0
                if not has_pair:
                    beta[0] = -np.inf
                for block in range(blocks - 1, 0, -1):
                    _retreat_block(emissions[block], sample, rows[g, block], rows[f, block], True,
                                   rows, block, f, g, region, has_pair, offsets[block], penalty,
                                   beta, count, gathered, work)
                    column = splice_table[f, g, block]
                    if column >= 0:
                        result[sample, column] = _meet(pair_prefix[block - 1], beta, count)
    return result


@njit(cache=True)
def _gather_block(values, sample, r1, rows, block, gathered):
    """gathered[t, x] = emission of (r1, panel row x) and gathered[t, k] = (r1, r1)
    for every bin t of one block: the new states' emissions, contiguous per bin."""
    k = rows.shape[0]
    bins = values.shape[3]
    for x in range(k):
        source = rows[x, block]
        for marker in range(bins):
            gathered[marker, x] = values[sample, r1, source, marker]
    for marker in range(bins):
        gathered[marker, k] = values[sample, r1, r1, marker]


@njit(cache=True)
def _region_block(gathered, bins, removed, has_region, region, t0, penalty, a_value, state):
    """Advance one single-path two-region DP over the bins of one block.

    The operations are _two_region's, in the same order. state[x]: the new state of the path
    with panel row x, state[k]: with itself (emissions from _gather_block); a removed row's
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
            best = _max_range(state, 0, width, a_value)
            if has_region:
                a_value = best + (region[t] - region[t - 1])
            switch = best - penalty
            for z in range(width):
                value = state[z]
                state[z] = (value if value > switch else switch) + gathered[marker, z]
        if removed >= 0:
            state[removed] = -np.inf
    return a_value


@njit(cache=True, parallel=True)
def birth_bounds(emissions, rows, penalty, forward, paths, common, offsets, start, stop, parts):
    """Per-sample upper bounds on V_s(P + c) (the two-region bound of birth(c), F empty) for the
    sorted paths start..stop-1.

    common[i] counts the leading blocks path i shares with path i-1: the DP state after a
    shared prefix is reused (as in panel_candidates.prefix_scores), so the work follows the
    distinct prefixes and the values equal the per-move recursion exactly. Each task is one
    sample and one contiguous part of the path range.
    """
    k, blocks = rows.shape
    samples = forward.shape[0]
    count = stop - start
    widest = 1
    for block in range(blocks):
        widest = max(widest, emissions[block].shape[3])
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
        region = forward[sample]
        for i in range(lo, hi):
            shared = 0 if i == lo else min(common[i], blocks - 1)
            for block in range(shared, blocks):
                values = emissions[block]
                _gather_block(values, sample, paths[i, block], rows, block, gathered)
                if block == 0:
                    a_value = -np.inf
                else:
                    a_value = prefix_a[block]
                    for z in range(k + 1):
                        buffer[z] = prefix_v[block, z]
                a_value = _region_block(gathered, values.shape[3], -1, True, region,
                                        offsets[block], penalty, a_value, buffer)
                prefix_a[block + 1] = a_value
                for z in range(k + 1):
                    prefix_v[block + 1, z] = buffer[z]
            result[sample, i - start] = _max_range(prefix_v[blocks], 0, k + 1, prefix_a[blocks])
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
        carry = _max_range(work, 0, width, -np.inf) - penalty
        if has_region:
            value = (region[t] - region[t - 1]) + beta[0]
            if value > carry:
                carry = value
            beta[0] = carry
        for z in range(width):
            value = work[z]
            beta[1 + z] = value if value > carry else carry


@njit(cache=True, parallel=True)
def replace_bounds(emissions, rows, penalty, index, excluded, offsets, split, prefixes,
                   prefix_common, prefix_of, suffixes, suffix_common, suffix_of, width, wanted):
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
    """
    k, blocks = rows.shape
    samples = index.shape[0]
    count = len(prefix_of)
    widest = 1
    for block in range(blocks):
        widest = max(widest, emissions[block].shape[3])
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
                values = emissions[block]
                _gather_block(values, sample, prefixes[p, block], rows, block, gathered)
                for rg in range(nreg):
                    region = excluded[index[sample, regions[rg]]]
                    a_value = -np.inf
                    if block > 0:
                        a_value = stack[block, rg, 0]
                        for z in range(k + 1):
                            buffer[z] = stack[block, rg, 1 + z]
                    a_value = _region_block(gathered, values.shape[3], regions[rg], has_region,
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
                values = emissions[block]
                _gather_block(values, sample, suffixes[q, block - split], rows, block, gathered)
                for rg in range(nreg):
                    region = excluded[index[sample, regions[rg]]]
                    for z in range(size):
                        buffer[z] = after[block - split + 1, rg, z]
                    _region_retreat(gathered, values.shape[3], regions[rg], has_region, region,
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
                out[sample, i, columns[rg]] = _max_range(buffer, 0, size, -np.inf)
    return out


@njit(cache=True, parallel=True)
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


@njit(cache=True, parallel=True)
def column_sums(values):
    """Sum each column in sample order (one thread per column)."""
    samples, count = values.shape
    result = np.empty(count)
    for column in prange(count):
        total = 0.0
        for sample in range(samples):
            total += values[sample, column]
        result[column] = total
    return result


class PanelForwards:
    """Exclusion forwards of one panel; shared by every bound of one pass."""

    def __init__(self, emissions, rows, penalty, total_bins):
        self.emissions = emissions
        self.rows = np.ascontiguousarray(rows, dtype=np.int64)
        self.penalty = float(penalty)
        self.total_bins = int(total_bins)
        self.forward, singles, self.pairs = forward_covers(
            emissions, self.rows, self.penalty, self.total_bins)
        index = np.full(singles.shape, -1, dtype=np.int64)
        flat = np.flatnonzero(singles.ravel())
        index.ravel()[flat] = np.arange(len(flat))
        self.index = index
        self.excluded, self.second = single_exclusions(
            emissions, self.rows, self.penalty, self.total_bins, index, len(flat))
        # Mean number of rows per sample whose exclusion was recomputed.
        self.recomputed_per_sample = len(flat) / max(1, singles.shape[0])
        # Per-sample birth bounds by local path (bytes): panel_count's count step and the
        # pool bounds of this panel share them (the same recursion, computed once).
        self.births = {}

    def removal_scores(self):
        """Exact per-sample V_n(P minus f) for every row f (samples x k)."""
        last = self.forward.shape[1] - 1
        values = np.repeat(self.forward[:, last:last + 1], self.index.shape[1], axis=1)
        mask = self.index >= 0
        values[mask] = self.excluded[self.index[mask], last]
        return values

    def structural(self, moves, local_new, offsets, keep, limit=_MOVES_PER_CHUNK):
        """Summed bounds of the structural moves (local, onesided, splice) in move order, and
        the per-sample bounds of the moves whose sum keep(sums) retains ({position: vector}).

        Moves are bounded by structural_bounds in groups of consecutive founders (the removed
        row) holding at most ``limit`` moves where possible, so the per-sample values of one
        group are held at a time."""
        lo, hi = moves.pool_moves, moves.kernel_moves
        result = np.empty(hi - lo)
        kept = {}
        if hi <= lo:
            return result, kept
        k, blocks = self.rows.shape
        kind = moves.kind[lo:hi]
        founder = moves.first_removed[lo:hi]
        partner = moves.canonical[lo:hi, 2]
        position = moves.canonical[lo:hi, 3]
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        per_founder = np.bincount(founder, minlength=k)
        start = 0
        while start < k:
            stop = start + 1
            size = per_founder[start]
            while stop < k and size + per_founder[stop] <= limit:
                size += per_founder[stop]
                stop += 1
            chosen = np.flatnonzero((founder >= start) & (founder < stop))
            start = stop
            if not len(chosen):
                continue
            columns = np.arange(len(chosen))
            group_kind = kind[chosen]
            active = np.zeros(k, dtype=np.bool_)
            active[founder[chosen][group_kind != SPLICE]] = True
            local = group_kind == LOCAL
            moves_local = np.ascontiguousarray(np.stack([
                columns[local], founder[chosen][local], position[chosen][local],
                local_new[moves.first_new[lo:hi][chosen][local], position[chosen][local]]], axis=1),
                dtype=np.int64)
            onesided = np.full((k, k, blocks), -1, dtype=np.int64)
            mask = group_kind == ONESIDED
            onesided[founder[chosen][mask], partner[chosen][mask], position[chosen][mask]] = columns[mask]
            splice = np.full((k, k, blocks), -1, dtype=np.int64)
            mask = group_kind == SPLICE
            splice[founder[chosen][mask], partner[chosen][mask], position[chosen][mask]] = columns[mask]
            values = structural_bounds(
                self.emissions, self.rows, self.penalty, self.total_bins, self.forward, self.index,
                self.excluded, self.second, self.pairs, offsets, active, moves_local, onesided,
                splice, len(chosen))
            sums = column_sums(values)
            result[chosen] = sums
            for column in np.flatnonzero(keep(sums)).tolist():
                kept[int(chosen[column])] = values[:, column].copy()
        return result, kept

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
            common = np.zeros(count, dtype=np.int64)
            if count > 1:
                equal = ordered[1:] == ordered[:-1]
                common[1:] = np.where(equal.all(axis=1), blocks, np.argmin(equal, axis=1))
            threads = max(1, numba.get_num_threads())
            offsets = np.ascontiguousarray(offsets, dtype=np.int64)
            for start in range(0, count, _BIRTH_CHUNK):
                stop = min(count, start + _BIRTH_CHUNK)
                parts = max(1, min(stop - start, -(-2 * threads // samples)))
                values = birth_bounds(self.emissions, self.rows, self.penalty, self.forward,
                                      ordered, common, offsets, start, stop, parts)
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
        step = max(256, (1 << 26) // (8 * samples * width))
        for start in range(0, len(paths), step):
            group = paths[start:start + step]
            split = _best_split(group)
            prefixes, prefix_of = np.unique(group[:, :split], axis=0, return_inverse=True)
            prefix_common = _leading_shared(prefixes)
            tails, suffix_of = np.unique(group[:, split:][:, ::-1], axis=0, return_inverse=True)
            suffix_common = _leading_shared(tails)
            suffixes = np.ascontiguousarray(tails[:, ::-1])
            out = replace_bounds(self.emissions, self.rows, self.penalty, self.index,
                                 self.excluded, offsets, split, np.ascontiguousarray(prefixes),
                                 prefix_common, prefix_of.ravel().astype(np.int64), suffixes,
                                 suffix_common, suffix_of.ravel().astype(np.int64), width, wanted)
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


def _leading_shared(rows):
    """For sorted distinct rows: the number of leading columns each shares with the previous."""
    shared = np.zeros(len(rows), dtype=np.int64)
    if len(rows) > 1 and rows.shape[1]:
        equal = rows[1:] == rows[:-1]
        shared[1:] = np.where(equal.all(axis=1), rows.shape[1], np.argmin(equal, axis=1))
    return shared


def _best_split(paths):
    """The block split t (1 <= t <= B) with the fewest distinct prefixes of lengths 1..t plus
    distinct suffixes of lengths 1..B - t: the block passes of replace_bounds."""
    blocks = paths.shape[1]
    prefixes = [len(np.unique(paths[:, :depth], axis=0)) for depth in range(1, blocks + 1)]
    suffixes = [len(np.unique(paths[:, blocks - depth:], axis=0)) for depth in range(1, blocks)]
    costs = [sum(prefixes[:split]) + sum(suffixes[:blocks - split]) for split in range(1, blocks + 1)]
    return 1 + int(np.argmin(costs))


def _sparse_scores(forwards, lower, founder, new_row, trial_rows, score_samples, sample_bounds=None):
    """Exact per-sample scores V_n(P') of the panel P' with row ``founder`` replaced by the local
    path ``new_row``.

    P minus f is contained in P', so lower_n = V_n(P minus f) <= V_n(P') <= U_n (the move's
    two-region bound). V_n(P') is computed (score_samples, the full-score recursion of a sample
    subset) only where U_n exceeds lower_n; elsewhere the new path's states never exceed the
    running maximum of P minus f, so V_n(P') = lower_n exactly. sample_bounds (optional): the
    move's per-sample bounds from the bulk bounds of its neighbourhood, which equal U_n or exceed
    it, up to float rounding (a 1e-6 relative margin covers it); U_n is recomputed only where
    they leave the comparison open, so the scored samples are those of the full comparison.
    Returns (scores, samples scored). At a one-path panel P minus f is empty (lower_n = -inf), so
    every sample is scored.
    """
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
    upper = move_bounds(forwards.emissions, forwards.rows, forwards.penalty, forwards.total_bins,
                        forwards.forward, forwards.index, forwards.excluded, forwards.second,
                        forwards.pairs, np.ascontiguousarray(np.reshape(new_row, (1, -1)),
                                                             dtype=np.int64),
                        int(founder), -1, candidates)
    need = candidates[upper > tolerance[candidates]]
    if len(need):
        values[need] = score_samples(forwards.emissions, np.ascontiguousarray(trial_rows.T),
                                     forwards.penalty, need)
    return values, len(need)


def replaced_row(moves, move):
    """The panel row f a move replaces by one new path, or None for any other move."""
    first = int(moves.first_removed[move])
    if first < 0 or int(moves.second_removed[move]) >= 0 or int(moves.second_new[move]) >= 0:
        return None
    return first


def one_path_move_scores(forwards, moves, local_new, removal, move, trial_rows, score_samples,
                         sample_bounds=None):
    """Exact per-sample scores (_sparse_scores) of a move that replaces one panel row f by one new
    path, or None for any other move."""
    first = replaced_row(moves, move)
    if first is None:
        return None
    return _sparse_scores(forwards, removal[:, first], first, local_new[int(moves.first_new[move])],
                          trial_rows, score_samples, sample_bounds)


def replacement_scores(forwards, removal, founder, new_row, trial_rows, score_samples):
    """Exact per-sample scores of the panel with row ``founder`` replaced by ``new_row`` (local
    rows), sparse over samples (_sparse_scores); returns (scores, samples scored)."""
    return _sparse_scores(forwards, removal[:, founder], founder, new_row, trial_rows, score_samples)


def emission_list(sub):
    arrays = List()
    for item in sub:
        arrays.append(np.ascontiguousarray(item['bin_emissions']))
    return arrays




class Moves:
    """Same-count neighbourhood moves as arrays in kernel order, with canonical keys.

    canonical rows: (kind, founder, partner, block, row or pool index), -1 for
    unused fields, kind order as in KINDS; pool indices are global (offset).
    Pool moves come first (per pool path in lexicographic order: birth, then
    replace(f) for every row f; bounded by PanelForwards.pool_bounds over shared
    prefixes and suffixes; birth(c) has size_change 1 and is excluded by
    lower_bounds), then the
    structural moves (structural_bounds): local, onesided and splice moves whose
    result keeps k distinct paths.
    """

    _FIELDS = ('kind', 'first_removed', 'second_removed', 'first_new', 'second_new',
               'size_change')

    def __init__(self, rows, capacity):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.capacity = capacity
        self.lookup = {}
        for index, row in enumerate(map(tuple, self.rows.tolist())):
            self.lookup.setdefault(row, index)
        self._parts = []
        self._paths = []
        self._path_count = 0
        self._count = 0
        self.pool_paths = np.zeros((0, self.rows.shape[1]), dtype=np.int64)
        self.pool_births = False
        self.pool_moves = 0

    def _add_paths(self, paths):
        start = self._path_count
        self._paths.append(np.asarray(paths, dtype=np.int64).reshape(-1, self.rows.shape[1]))
        self._path_count += len(paths)
        return start

    def _members(self, paths):
        return np.asarray([self.lookup.get(path, -1) for path in map(tuple, paths.tolist())],
                          dtype=np.int64)

    def _add(self, kind, first_removed, second_removed, first_new, second_new, size_change,
             canonical):
        count = len(kind)
        if not count:
            return
        arrays = [np.broadcast_to(np.asarray(value, dtype=np.int64), (count,)).copy()
                  for value in (kind, first_removed, second_removed, first_new, second_new,
                                size_change)]
        self._parts.append((arrays, np.asarray(canonical, dtype=np.int64).reshape(count, 5)))
        self._count += count

    def add_pool(self, pool, pool_offset=0):
        """birth(c) and replace(f, c) for every pool path c not in the panel.

        Must be the first part added. Returns the number of pool paths tested.
        """
        k, blocks = self.rows.shape
        pool = np.asarray(pool, dtype=np.int64).reshape(-1, blocks)
        eligible = np.flatnonzero(self._members(pool) < 0)
        if not len(eligible):
            return 0
        eligible = eligible[np.lexsort(pool[eligible][:, ::-1].T)]
        paths = np.ascontiguousarray(pool[eligible])
        self.pool_paths = paths
        self.pool_births = births = k + 1 <= self.capacity
        group = k + int(births)
        shape = (len(eligible), group)
        kind = np.full(shape, REPLACE)
        first_removed = np.full(shape, -1)
        change = np.zeros(shape, dtype=np.int64)
        canonical = np.full(shape + (5,), -1, dtype=np.int64)
        canonical[..., 4] = (pool_offset + eligible)[:, None]
        columns = slice(1, None) if births else slice(None)
        first_removed[:, columns] = np.arange(k)[None,:]
        canonical[:, columns, 0] = REPLACE
        canonical[:, columns, 1] = np.arange(k)[None,:]
        if births:
            kind[:, 0] = BIRTH
            change[:, 0] = 1
            canonical[:, 0, 0] = BIRTH
        path_index = self._add_paths(paths) + np.arange(len(eligible))
        self._add(kind.ravel(), first_removed.ravel(), -1, np.repeat(path_index, group), -1,
                  change.ravel(), canonical.reshape(-1, 5))
        self.pool_moves = self._count
        return len(eligible)

    def add_structural(self, block_sizes, slots=None):
        """local, onesided and splice moves whose result keeps k distinct paths.

        slots: if given, only moves that change at least one of these rows (a seed refit)."""
        rows = self.rows
        k, blocks = rows.shape
        mine = np.ones(k, dtype=bool) if slots is None else np.isin(np.arange(k), list(slots))
        founders, positions, alternatives = [], [], []
        for block in range(blocks):
            choices = np.arange(block_sizes[block])
            for founder in np.flatnonzero(mine).tolist():
                other = choices[choices != rows[founder, block]]
                founders.append(np.full(len(other), founder))
                positions.append(np.full(len(other), block))
                alternatives.append(other)
        if founders:
            founder = np.concatenate(founders)
            block = np.concatenate(positions)
            row = np.concatenate(alternatives)
            paths = rows[founder].copy()
            paths[np.arange(len(founder)), block] = row
            member = self._members(paths)
            index = self._add_paths(paths) + np.arange(len(founder))
            canonical = np.stack([np.full(len(founder), LOCAL), founder,
                                  np.full(len(founder), -1), block, row], axis=1)
            keep = member < 0
            self._add(np.full(int(keep.sum()), LOCAL), founder[keep], -1, index[keep], -1, 0,
                      canonical[keep])
        if k >= 2 and blocks >= 2:
            pairs = np.asarray([(a, b) for a in range(k) for b in range(k) if a != b],
                               dtype=np.int64)
            ordinal = np.full((k, k), -1, dtype=np.int64)
            ordinal[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs))
            boundaries = np.arange(1, blocks)
            first = np.repeat(pairs[:, 0], len(boundaries))
            second = np.repeat(pairs[:, 1], len(boundaries))
            boundary = np.tile(boundaries, len(pairs))
            paths = np.where(np.arange(blocks)[None,:] < boundary[:, None],
                             rows[first], rows[second])
            member = self._members(paths)
            start = self._add_paths(paths)
            index = start + np.arange(len(first))
            zeros = np.full(len(first), -1)
            base = np.stack([zeros, first, second, boundary, zeros], axis=1)

            def canonical_for(kind, mask):
                values = base[mask].copy()
                values[:, 0] = kind
                return values
            keep = (member < 0) & mine[first]
            self._add(np.full(int(keep.sum()), ONESIDED), first[keep], -1, index[keep], -1, 0,
                      canonical_for(ONESIDED, keep))
            partner = start + ordinal[second, first] * len(boundaries) + boundary - 1
            keep = ((first < second) & (member < 0) & (member[partner - start] < 0)
                    & (mine[first] | mine[second]))
            self._add(np.full(int(keep.sum()), SPLICE), first[keep], second[keep],
                      index[keep], partner[keep], 0, canonical_for(SPLICE, keep))

    def finish(self):
        self.kernel_moves = self._count
        if self._parts:
            for position, name in enumerate(self._FIELDS):
                setattr(self, name, np.concatenate([arrays[position] for arrays, _ in self._parts]))
            self.canonical = np.concatenate([canonical for _, canonical in self._parts])
        else:
            for name in self._FIELDS:
                setattr(self, name, np.zeros(0, dtype=np.int64))
            self.canonical = np.zeros((0, 5), dtype=np.int64)
        self.new_paths = (np.concatenate(self._paths) if self._paths
                          else np.zeros((0, self.rows.shape[1]), dtype=np.int64))
        del self._parts, self._paths
        return self

    def __len__(self):
        return len(self.kind)

    def counts(self):
        values = np.bincount(self.kind, minlength=len(KINDS))
        return {name: int(values[index]) for index, name in enumerate(KINDS) if values[index]}

    def description(self, move, candidates_offset=0):
        kind, founder, partner, block, index = (int(x) for x in self.canonical[move])
        if kind == REPLACE:
            return ('replace', founder, index)
        if kind == LOCAL:
            return ('local', founder, block, index)
        if kind in (ONESIDED, SPLICE):
            return (KINDS[kind], founder, partner, block)
        return ('birth', (index,))


def lower_bounds(moves, forwards, likelihood, to_local, offsets, replaced=None):
    """Lower bound of every move's change of the join score -2 LL: -2 (sum_n U_n - LL(P)).

    to_local maps dense paths to emission rows; offsets are the first bin of
    every block. birth(c) moves are bounded with their replace(f, c) moves but get
    +inf: the panel size is decided by panel_count. Also returns the per-sample
    bounds of every move best_improvement's exclusion test keeps ({move: vector}),
    with which the exact scores skip samples. replaced: bound only the replacements
    of these panel rows (PanelForwards.pool_bounds); the caller excludes the others."""
    totals = np.empty(len(moves))
    likelihood = float(likelihood)
    eps = 1e-9 * (1.0 + abs(-2.0 * likelihood))
    threshold = max(1e-8, eps)

    def keep(sums):
        return -2.0 * (sums - likelihood) <= -threshold + eps
    sample_bounds = {}
    pool = moves.pool_moves
    if pool:
        values, kept = forwards.pool_bounds(to_local(moves.pool_paths), offsets, keep, replaced)
        group = values.shape[1] - (0 if moves.pool_births else 1)
        shift = 0 if moves.pool_births else 1
        totals[:pool] = (values if moves.pool_births else values[:, 1:]).ravel()
        for (path, column), vector in kept.items():
            sample_bounds[path * group + column - shift] = vector
    if moves.kernel_moves > pool:
        totals[pool:moves.kernel_moves], kept = forwards.structural(
            moves, to_local(moves.new_paths), offsets, keep)
        for position, vector in kept.items():
            sample_bounds[pool + position] = vector
    result = -2.0 * (totals - likelihood)
    result[moves.size_change != 0] = np.inf
    return result, sample_bounds


def best_improvement(bounds, canonical, materialize, score, current_bic, *, signature,
                     admissible):
    """Best-first exact scoring of the moves a bound cannot exclude.

    materialize(move) -> panel; score(panel, move) -> (exact join score -2 LL, payload);
    admissible(panel) is False for a move outside the searched set (a changed path
    left without sequence of its own). eps = 1e-9 (1 + |score|) is the float
    resolution allowed for the exact score and the bounds (measured disagreement
    <= 1.2e-10 relative). A move improves iff its exact change < -t, t = max(1e-8,
    eps); it is excluded iff its bound > -t + eps. Survivors are scored in increasing
    bound order while the next bound <= min(best admissible exact change, -t) + eps.
    The best move has the least exact change, ties by canonical order; identical
    panels are scored once and take the path order of their first move in canonical
    order. Returns ((change, canonical, move, panel, payload) or None, survivors, scored).
    """
    eps = 1e-9 * (1.0 + abs(float(current_bic)))
    threshold = max(1e-8, eps)
    survivors = np.flatnonzero(bounds <= -threshold + eps)
    if not len(survivors):
        return None, 0, 0
    keys = canonical[survivors]
    order = survivors[np.lexsort(tuple(keys[:, column] for column in range(
        keys.shape[1] - 1, -1, -1)) + (bounds[survivors],))]
    seen = {}
    best_change = None
    scored = 0
    for move in order.tolist():
        limit = -threshold if best_change is None else min(-threshold, best_change)
        if bounds[move] > limit + eps:
            break
        panel = materialize(move)
        key = signature(panel)
        canon = tuple(int(x) for x in canonical[move])
        entry = seen.get(key)
        if entry is not None:
            # The same panel (same paths with multiplicity) from an earlier move
            # in canonical order: keep that move's description and path order.
            if canon < entry[1]:
                entry[1] = canon
                entry[2] = move
                if entry[3] is not None:
                    entry[3] = panel
            continue
        if not admissible(panel):
            seen[key] = [np.inf, canon, move, None, None]
            continue
        exact_bic, payload = score(panel, move)
        scored += 1
        change = float(exact_bic) - float(current_bic)
        improving = change < -threshold
        seen[key] = [change, canon, move, panel if improving else None,
                     payload if improving else None]
        if improving and (best_change is None or change < best_change):
            best_change = change
    improving = [entry for entry in seen.values() if entry[0] < -threshold]
    if not improving:
        return None, len(survivors), scored
    return tuple(min(improving, key=lambda entry: (entry[0], entry[1]))), len(survivors), scored
