"""Fixed-K fits of many starts: one frontier engine over the deterministic coordinate map.

A fit of an ordered panel H0 is the trajectory H0 -> H1 = H(A(H0)) -> ... of the A step
(assign.py) and the H step (update.py). It stops at a fixed point (an H step that changes
nothing: the fit's assignments are then A of the final panel), when the trajectory enters a
cycle (it then returns the cycle's lowest-NLL state, unconverged) or after ``max_iterations``
steps (the assignments are then recomputed for the last panel). A fit result is the tuple
``(H, A, per_sample_cost, unknown_slots, iterations, nll)`` of read-only arrays; an unconverged
fit reports ``iterations = max_iterations``.

The map is deterministic, so every state's transition A(H), H(A(H)) is memoised in the block's
workspace and all starts of a batch advance together as a frontier: equal states are evaluated
once, a frontier of several uncached states runs one batched kernel (emissions shared across
the frontier's rows and row pairs, sample tiles in parallel), a single uncached state runs the
internally parallel A and H steps. Results never depend on batch composition or threads.
"""
from __future__ import annotations

import math

import numba
import numpy as np
from numba import njit, prange

from ...execution import claim_threads
from ..panel import canonical_solution, deduplicate, unique_rows
from .assign import assign, haplotype_patterns, pair_indices
from .update import founder_buckets, sweep_one_thread, update_founders, usage_order
from .workspace import FixedKWorkspace

# Thread cap of one block's fixed-K fitting phase (a thread limit, not a K limit): the frontier
# and A/H kernels stop scaling well above it.
FIT_MAX_THREADS = 16
# Shared-emission workspace per numerical worker of the batched frontier kernel, and its
# sample-tile width; a frontier whose tile would not fit is split (identical results).
_FRONTIER_TILE_BYTES = 8 * 1024 * 1024
_FRONTIER_TILE_SAMPLES = 16
# The batched kernel parallelises over sample tiles; beyond this many threads per frontier
# state a wider Numba team only adds start-up cost (measured low-frontier saturation).
_FRONTIER_MIN_THREADS = 4


def _readonly(*arrays):
    for array in arrays:
        array.setflags(write=False)


def _assignment_record(assignments, cost, unknown_slots):
    """Immutable A(H) record: (assignments, per-sample cost, unknown slots, total NLL)."""
    _readonly(assignments, cost, unknown_slots)
    # The exactly rounded total: independent of sample order and summation layout (release 2.1).
    return (assignments, cost, unknown_slots, math.fsum(cost.tolist()))


def _fit_result(haplotypes, record, iterations):
    """A fit result tuple from an ordered panel (taken over: made read-only) and its
    assignment record."""
    haplotypes.setflags(write=False)
    return (haplotypes, record[0], record[1], record[2], int(iterations), record[3])


def _assignment_scalar(workspace, haplotypes, key):
    record = _assignment_record(*assign(workspace, haplotypes))
    workspace.remember_assignment(key, record)
    return record


def _transition_scalar(workspace, haplotypes, key):
    assignments, cost, unknown_slots = assign(workspace, haplotypes)
    following = np.array(haplotypes, dtype=np.int64, order="C", copy=True)
    changes = update_founders(workspace, following, assignments)
    record = _assignment_record(assignments, cost, unknown_slots)
    return workspace.remember_transition(key, record, workspace.state_key(following), changes)


# ---------------------------------------------------------------------------------------------
# Batched transitions of a frontier of distinct ordered panels (same K).
# ---------------------------------------------------------------------------------------------
def _distinct_byte_rows(rows):
    """(first index, inverse) of the distinct rows of a 2-D uint8 matrix, the distinct rows in
    lexicographic byte order: ``np.unique`` of the rows viewed as one ``np.void`` each (stable:
    first occurrence), by one stable lexsort. numpy 2.4 keeps memory for every void/structured
    dtype instance it sorts (~10 KB per fresh ``np.dtype((np.void, n))``), which a per-call view
    leaked in long-lived discovery workers."""
    n, width = rows.shape
    if width == 0:
        return np.zeros(min(n, 1), dtype=np.intp), np.zeros(n, dtype=np.intp)
    order = np.lexsort(rows.T[::-1])
    ordered = rows[order]
    new = np.ones(n, dtype=bool)
    new[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    inverse = np.empty(n, dtype=np.intp)
    inverse[order] = np.cumsum(new) - 1
    return order[new], inverse


def _frontier_layout(batch, first, second):
    """Distinct rows of the batch, distinct (row, row) pairs, and per panel the shared index of
    each of its row-pair states (row-major) and row-unknown states."""
    b, k, length = batch.shape
    packed = np.packbits(batch, axis=2, bitorder="little")
    row_first, row_inverse = _distinct_byte_rows(packed.reshape(b * k, -1))
    row_inverse = row_inverse.reshape(b, k)
    n_rows = len(row_first)
    pair_keys = row_inverse[:, first] * n_rows + row_inverse[:, second]
    unique_pairs, pair_inverse = np.unique(pair_keys, return_inverse=True)
    n_pairs = len(unique_pairs)
    state_indices = np.empty((b, len(first) + k), dtype=np.int64)
    state_indices[:, :len(first)] = pair_inverse.reshape(b, -1)
    state_indices[:, len(first):] = row_inverse + n_pairs
    rows = np.ascontiguousarray(batch.reshape(b * k, length)[row_first])
    return rows, unique_pairs // n_rows, unique_pairs % n_rows, state_indices


@njit(cache=True, parallel=True, nogil=True)
def _assign_frontier(c0b, diff1_table, w_table, kw_cb, kw_diff_table, ww_bin_emissions,
                     ww_total_cost, row_slots, pair_i, pair_j, pair_slots, state_indices, first,
                     second, penalty, k, n_bins, tile_width, tie):
    """A step of every panel of a frontier, sharing the emissions of distinct rows and row
    pairs within sample tiles; per panel the state order, argmin tie rule and Viterbi are those
    of the single-panel A step (identical values)."""
    n = c0b.shape[0]
    b_count = state_indices.shape[0]
    n_pairs = pair_i.shape[0]
    n_rows = row_slots.shape[0]
    n_shared = n_pairs + n_rows
    n_rr = first.shape[0]
    n_states = n_rr + k + 1
    assignments = np.empty((b_count, n, 2), dtype=np.int64)
    cost = np.empty((b_count, n), dtype=np.float64)
    unknown_slots = np.empty((b_count, n), dtype=np.int64)
    for tile in prange((n + tile_width - 1) // tile_width):
        start_sample = tile * tile_width
        width = min(tile_width, n - start_sample)
        emissions = np.empty((n_shared, n_bins, width), dtype=np.float64)
        totals = np.zeros((n_shared, width), dtype=np.float64)
        for p in range(n_pairs):
            i = pair_i[p]
            j = pair_j[p]
            for b in range(n_bins):
                si = row_slots[i, b]
                sj = row_slots[j, b]
                sij = pair_slots[p, b]
                for t in range(width):
                    s = start_sample + t
                    e = c0b[s, b] + diff1_table[b, si, s] + diff1_table[b, sj, s] + w_table[b, sij, s]
                    emissions[p, b, t] = e
                    totals[p, t] += e
        for row in range(n_rows):
            state = n_pairs + row
            for b in range(n_bins):
                slot = row_slots[row, b]
                for t in range(width):
                    s = start_sample + t
                    e = kw_cb[s, b] + kw_diff_table[b, slot, s]
                    emissions[state, b, t] = e
                    totals[state, t] += e
        alpha = np.empty((n_states, width), dtype=np.float64)
        best_cost = np.empty(width, dtype=np.float64)
        best_state = np.empty(width, dtype=np.int64)
        best_previous = np.empty(width, dtype=np.float64)
        for panel in range(b_count):
            for t in range(width):
                best_cost[t] = np.inf
                best_state[t] = 0
            for state in range(n_states - 1):
                shared = state_indices[panel, state]
                for t in range(width):
                    candidate = -totals[shared, t]
                    if candidate < best_cost[t]:
                        best_cost[t] = candidate
                    alpha[state, t] = emissions[shared, 0, t]
            for t in range(width):
                s = start_sample + t
                if ww_total_cost[s] < best_cost[t]:
                    best_cost[t] = ww_total_cost[s]
                alpha[n_states - 1, t] = ww_bin_emissions[s, 0]
                # The first state (state order) within the tie quantum of the cheapest, as assign.py.
                threshold = best_cost[t] + tie
                chosen = n_states - 1
                for st in range(n_states - 1):
                    if -totals[state_indices[panel, st], t] <= threshold:
                        chosen = st
                        break
                best_state[t] = chosen
                state = best_state[t]
                if state < n_rr:
                    assignments[panel, s, 0] = first[state]
                    assignments[panel, s, 1] = second[state]
                    unknown_slots[panel, s] = 0
                elif state < n_rr + k:
                    assignments[panel, s, 0] = state - n_rr
                    assignments[panel, s, 1] = k
                    unknown_slots[panel, s] = 1
                else:
                    assignments[panel, s, 0] = k
                    assignments[panel, s, 1] = k
                    unknown_slots[panel, s] = 2
            for b in range(1, n_bins):
                for t in range(width):
                    best_previous[t] = -np.inf
                for state in range(n_states):
                    for t in range(width):
                        if alpha[state, t] > best_previous[t]:
                            best_previous[t] = alpha[state, t]
                for state in range(n_states - 1):
                    shared = state_indices[panel, state]
                    for t in range(width):
                        switch_base = best_previous[t] - penalty
                        stay = alpha[state, t]
                        e = emissions[shared, b, t]
                        if stay > switch_base:
                            alpha[state, t] = stay + e
                        else:
                            alpha[state, t] = switch_base + e
                for t in range(width):
                    switch_base = best_previous[t] - penalty
                    stay = alpha[n_states - 1, t]
                    e = ww_bin_emissions[start_sample + t, b]
                    if stay > switch_base:
                        alpha[n_states - 1, t] = stay + e
                    else:
                        alpha[n_states - 1, t] = switch_base + e
            for t in range(width):
                best_final = -np.inf
                for state in range(n_states):
                    if alpha[state, t] > best_final:
                        best_final = alpha[state, t]
                cost[panel, start_sample + t] = -best_final
    return assignments, cost, unknown_slots


@njit(cache=True, parallel=True, nogil=True)
def _update_frontier(batch, assignments, unknown_slots, uninformative, genotype_cost_by_sample,
                     unknown_cost_by_sample, tie):
    """Uninformative samples to (U, U), then every panel's one-thread H sweep (parallel over
    panels)."""
    following = batch.copy()
    changes = np.empty(batch.shape[0], dtype=np.int64)
    k = batch.shape[1]
    for panel in prange(batch.shape[0]):
        for sample in range(assignments.shape[1]):
            if uninformative[sample]:
                assignments[panel, sample, 0] = k
                assignments[panel, sample, 1] = k
                unknown_slots[panel, sample] = 2
        (homozygous, with_partner, with_unknown, partner, n_homozygous, n_with_partner,
         n_with_unknown, usage) = founder_buckets(assignments[panel], k)
        order = usage_order(usage)
        changes[panel] = sweep_one_thread(
            following[panel], order, homozygous, with_partner, with_unknown, partner,
            n_homozygous, n_with_partner, n_with_unknown, genotype_cost_by_sample,
            unknown_cost_by_sample, tie)
    return following, changes


def _row_patterns(rows, snps_per_bin, n_bins):
    first = np.zeros(0, dtype=np.int64)
    patterns, _ = haplotype_patterns(rows, first, first, snps_per_bin, n_bins)
    return patterns


def _batched_transitions(workspace, batch):
    """(following panels, assignments, cost, unknown slots, changes) of a frontier batch."""
    b, k, _ = batch.shape
    first, second = pair_indices(k)
    rows, pair_i, pair_j, state_indices = _frontier_layout(batch, first, second)
    n_bins = workspace.n_bins
    bytes_per_sample = 8 * ((len(pair_i) + len(rows)) * (n_bins + 1) + len(first) + k + 4)
    tile_width = min(_FRONTIER_TILE_SAMPLES, _FRONTIER_TILE_BYTES // bytes_per_sample)
    if tile_width < 1:
        if b > 1:
            # Too many distinct rows/pairs for one tile: split the frontier (each panel's
            # transition is independent of its batch-mates, so the results are identical).
            half = b // 2
            left = _batched_transitions(workspace, batch[:half])
            right = _batched_transitions(workspace, batch[half:])
            return tuple(np.concatenate((x, y)) for x, y in zip(left, right))
        tile_width = 1
    tables = workspace.patterns
    patterns = _row_patterns(rows, workspace.snps_per_bin, n_bins)
    row_slots = tables.slots(patterns)
    pair_slots = tables.slots(patterns[pair_i] & patterns[pair_j])
    assignments, cost, unknown_slots = _assign_frontier(
        workspace.c0b, tables.diff1_table, tables.w_table, workspace.kw_cb, tables.kw_diff_table,
        workspace.ww_bin_emissions, workspace.ww_total_cost, row_slots, pair_i, pair_j,
        pair_slots, state_indices, first, second, workspace.switch_penalty, k, n_bins,
        tile_width, float(workspace.quanta.sample))
    following, changes = _update_frontier(
        batch, assignments, unknown_slots, workspace.uninformative,
        workspace.genotype_cost_by_sample, workspace.unknown_cost_by_sample,
        float(workspace.quanta.h_step))
    return following, assignments, cost, unknown_slots, changes


def _transition_batch(workspace, matrices, keys):
    """Compute and remember the transitions of distinct uncached states of one frontier."""
    batch = np.ascontiguousarray(np.stack(matrices), dtype=np.int64)
    active = numba.get_num_threads()
    threads = min(active, max(_FRONTIER_MIN_THREADS, batch.shape[0]))
    if threads != active:
        numba.set_num_threads(threads)
    try:
        following, assignments, cost, unknown_slots, changes = _batched_transitions(workspace, batch)
    finally:
        if threads != active:
            numba.set_num_threads(active)
    for index, key in enumerate(keys):
        # The batched A step has no capped/uncapped split: cost is the one per-sample NLL.
        record = _assignment_record(assignments[index], cost[index], unknown_slots[index])
        workspace.remember_transition(key, record, workspace.state_key(following[index]),
                                      int(changes[index]))


# ---------------------------------------------------------------------------------------------
# The frontier engine.
# ---------------------------------------------------------------------------------------------
def _best_cycle_state(workspace, cycle_keys):
    """(key, assignment record) of the lowest-NLL state of a cycle of the coordinate map.

    ``cycle_keys`` lists the cycle's states in the order the fit first visited them; NLLs within
    the block's ``nll`` tie quantum of each other are ties, which keep that order. A cyclic fit returns this state
    unconverged: it reports the iteration limit and is never certified a fixed point. (Base
    returned the state the iteration-by-iteration loop would stop on after the limit, so the
    endpoint depended on the limit's parity rather than on the objective: discovery B2.)
    """
    best_key, best = None, None
    for key in cycle_keys:
        record = workspace.cached_assignment(key)
        if record is None:
            # Evicted from the bounded cache: recompute (a pure function of the state).
            record = _transition_scalar(workspace, workspace.state_from_key(key), key)[0]
        if best is None or record[3] < best[3] - workspace.quanta.nll:
            best_key, best = key, record
    return best_key, best


def _frontier_fits(workspace, starts, max_iterations, with_initial):
    """Fit distinct same-K starts together; returns final fits (and initial assignment-only fits
    when ``with_initial``) in start order, remembering both in the workspace."""
    count = len(starts)
    limit = max(0, int(max_iterations))
    finals = [None] * count
    initials = [None] * count
    fit_keys = [None] * count
    initial_keys = [None] * count
    state_keys = [None] * count
    previous_assignments = [None] * count
    completed = [0] * count
    paths = [None] * count
    first_visits = [None] * count
    cached_finals = [None] * count
    active = []
    for index, start in enumerate(starts):
        fit_keys[index] = workspace.fit_key(start, max_iterations)
        cached_finals[index] = workspace.cached_fit(fit_keys[index])
        if with_initial:
            initial_keys[index] = workspace.fit_key(start, 0)
            initials[index] = workspace.cached_fit(initial_keys[index])
        if cached_finals[index] is not None and (not with_initial or initials[index] is not None):
            finals[index] = cached_finals[index]
            continue
        key = workspace.state_key(start)
        state_keys[index] = key
        paths[index] = [key]
        first_visits[index] = {key: 0}
        active.append(index)

    def finish(index, result, fixed_point):
        finals[index] = result
        workspace.remember_fit(fit_keys[index], result, fixed_point=fixed_point)
        if with_initial:
            workspace.remember_fit(initial_keys[index], initials[index], fixed_point=False)

    assignment_only = limit == 0
    while active:
        missing = {}
        for index in active:
            key = state_keys[index]
            cached = (workspace.cached_assignment(key) if assignment_only
                      else workspace.cached_transition(key))
            if cached is None:
                missing.setdefault(key, None)
        if missing:
            matrices = [workspace.state_from_key(key) for key in missing]
            if assignment_only:
                for key, matrix in zip(missing, matrices):
                    _assignment_scalar(workspace, matrix, key)
            elif len(matrices) == 1:
                # A single state keeps the internally parallel A and H steps.
                _transition_scalar(workspace, matrices[0], next(iter(missing)))
            else:
                _transition_batch(workspace, matrices, tuple(missing))
        next_active = []
        for index in active:
            key = state_keys[index]
            if assignment_only:
                record, next_key, changes = workspace.cached_assignment(key), None, 0
            else:
                record, next_key, changes = workspace.cached_transition(key)
            if with_initial and initials[index] is None:
                initials[index] = _fit_result(starts[index].copy(), record, 0)
            if cached_finals[index] is not None:
                finish(index, cached_finals[index], False)
                continue
            if completed[index] >= limit:
                finish(index, _fit_result(workspace.state_from_key(key), record, limit), False)
                continue
            completed[index] += 1
            if changes == 0:
                # A fixed point: the assignments are A of the final panel. A fit whose
                # assignments still changed in this step reports one more iteration (the
                # confirming step of the iteration-by-iteration loop) when within the limit.
                reported = completed[index]
                previous = previous_assignments[index]
                if (previous is None or not np.array_equal(record[0], previous)) \
                        and completed[index] < limit:
                    reported += 1
                finish(index, _fit_result(workspace.state_from_key(key), record, reported), True)
                continue
            previous_assignments[index] = record[0]
            cycle_start = first_visits[index].get(next_key)
            if cycle_start is not None:
                # The deterministic map entered a cycle (period >= 2: an H step that changes
                # nothing is a fixed point), so no fixed point is reachable from this start.
                endpoint_key, endpoint = _best_cycle_state(
                    workspace, paths[index][int(cycle_start):])
                finish(index, _fit_result(workspace.state_from_key(endpoint_key), endpoint, limit),
                       False)
                continue
            state_keys[index] = next_key
            first_visits[index][next_key] = completed[index]
            paths[index].append(next_key)
            next_active.append(index)
        active = next_active
    if with_initial:
        return list(zip(initials, finals))
    return finals


def fit_starts(workspace, starts, max_iterations, *, with_initial=False, quanta=None):
    """Fit binary int64 starts of one K (in order); duplicate starts are fitted once.

    Returns the fit tuples (with ``with_initial``: (initial assignment-only fit, final fit)
    pairs) in start order. ``quanta`` (release 2.1 interface): must be None or the workspace's own
    BlockQuanta (the fits' tie rules are those of the workspace).
    """
    _check_quanta(workspace, quanta)
    starts = [np.ascontiguousarray(start, dtype=np.int64) for start in starts]
    if not starts:
        return []
    # A phase boundary: tail blocks claim cores released by finished workers.
    claim_threads(max_threads=FIT_MAX_THREADS)
    unique: dict[bytes, int] = {}
    distinct = []
    replay = []
    for start in starts:
        key = start.tobytes()
        index = unique.get(key)
        if index is None:
            index = unique[key] = len(distinct)
            distinct.append(start)
        replay.append(index)
    results = _frontier_fits(workspace, distinct, max_iterations, with_initial)
    return [results[index] for index in replay]


def fit_panels(workspace, starts, max_iterations, *, quanta=None):
    """Fit every distinct-row start and return (synchronized, refitted) canonical solutions.

    ``quanta``: None (the workspace's) or the workspace's own BlockQuanta; solutions carry them and both
    collections are deduplicated and ordered with them (tie classes of kind ``nll``).

    For each start (its distinct rows), the synchronized endpoint is its assignment-only fit
    (A of the start, no H step) and the refitted endpoint its complete fit. A refit that collapses
    rows re-enters as a start at the lower K (refitted collection only). Both collections are
    deduplicated (distinct complete panels, lowest NLL, ordered by (nll, key)). A synchronized
    endpoint is marked a fixed point only if its start was certified before this call.
    """
    _check_quanta(workspace, quanta)
    quanta = workspace.quanta
    pending = []
    seen = set()
    certified_before = {}
    for start in starts:
        rows = np.ascontiguousarray(unique_rows(np.asarray(start)), dtype=np.int64)
        key = (rows.shape, rows.tobytes())
        if key in seen:
            continue
        seen.add(key)
        pending.append(rows)
        certified_before[key] = workspace.certifies_fixed_point(rows)
    synchronized, refitted = [], []
    capture_initial = True
    while pending:
        grouped: dict[int, list] = {}
        for rows in pending:
            grouped.setdefault(len(rows), []).append(rows)
        pending = []
        for k in sorted(grouped):
            group = grouped[k]
            if capture_initial:
                records = fit_starts(workspace, group, max_iterations, with_initial=True)
            else:
                records = [(None, fit) for fit in fit_starts(workspace, group, max_iterations)]
            for start, (initial, final) in zip(group, records):
                if initial is not None:
                    key = (start.shape, start.tobytes())
                    synchronized.append(canonical_solution(
                        initial, workspace if certified_before[key] else None, quanta))
                solution = canonical_solution(final, workspace, quanta)
                rows = np.ascontiguousarray(unique_rows(solution.haplotypes), dtype=np.int64)
                if len(rows) == solution.k:
                    refitted.append(solution)
                    continue
                key = (rows.shape, rows.tobytes())
                if key not in seen:
                    seen.add(key)
                    pending.append(rows)
        capture_initial = False
    return deduplicate(synchronized, quanta=quanta), deduplicate(refitted, quanta=quanta)


def fit_one(workspace, haplotypes, max_iterations, *, quanta=None):
    """The fit tuple of one start."""
    return fit_starts(workspace, [haplotypes], max_iterations, quanta=quanta)[0]


def _check_quanta(workspace, quanta):
    if quanta is not None and quanta != workspace.quanta:
        raise ValueError("a fixed-K fit uses its workspace's tie quanta; pass None or workspace.quanta")


def prepare_workspace(likelihoods, observed, model, quanta=None) -> FixedKWorkspace:
    """A block's fixed-K workspace (see workspace.FixedKWorkspace); ``quanta``: the block's BlockQuanta
    (None: derived from the block's null NLL, maths.ties.block_scale)."""
    return FixedKWorkspace(likelihoods, observed, model, quanta)
