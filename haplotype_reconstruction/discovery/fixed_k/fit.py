"""Fixed-K fits of many starts: one frontier engine over the deterministic coordinate map.

Biological consumers fit complete families of the two consistent usage-tie schedules.
The second schedule is run only if the first encounters a positive-usage tie. A(H) and
untied H transitions are shared; tied H transitions, full fits and fixed-point proofs
are schedule-specific. A scalar fit of an ordered panel H0 is the trajectory H0 -> H1 = H(A(H0)) -> ... of the A step
(assign.py) and the H step (update.py). It stops at a fixed point (an H step that changes
nothing: the fit's assignments are then A of the final panel), when the trajectory enters a
cycle (it then returns the cycle's lowest-NLL state, unconverged) or after ``max_iterations``
steps (the assignments are then recomputed for the last panel). A fit result is the tuple
``(H, A, per_sample_cost, unknown_slots, iterations, nll, tie_offsets, tie_states)``.
Arrays are read-only; the last two arrays compactly store multi-state evidence membership.
An unconverged fit reports ``iterations = max_iterations``.

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
from ..conventions import rev
from ..panel import (assignment_tie_mask, canonical_solution, compact_ties, deduplicate,
                     panel_key, unique_rows)
from .assign import assign, haplotype_patterns, meet_pairs, pair_indices
from .update import (founder_buckets, fractional_buckets, fractional_sweep_one_thread,
                     sweep_one_thread, update_founders, usage_order,
                     _fractional_usage_commutes, _singleton_usage_commutes)
from .workspace import FixedKWorkspace

# Thread cap of one block's fixed-K fitting phase (a thread limit, not a K limit): the frontier
# and A/H kernels stop scaling well above it.
FIT_MAX_THREADS = 16
# Shared-emission workspace per numerical worker of the batched frontier kernel, and its
# sample-tile width; a frontier whose tile would not fit is split (identical results).
_FRONTIER_TILE_BYTES = 8 * 1024 * 1024
_FRONTIER_TILE_SAMPLES = 16
# Dense membership exists only during a transition; cached records keep sparse multi-state IDs.
_FRONTIER_MEMBERSHIP_BYTES = 32 * 1024 * 1024
# The batched kernel parallelises over sample tiles; beyond this many threads per frontier
# state a wider Numba team only adds start-up cost (measured low-frontier saturation).
_FRONTIER_MIN_THREADS = 4


def _readonly(*arrays):
    for array in arrays:
        array.setflags(write=False)


def _assignment_record(assignments, cost, unknown_slots, mask):
    """Immutable A(H) record: (assignments, per-sample cost, unknown slots, total NLL)."""
    _readonly(assignments, cost, unknown_slots)
    # The exactly rounded total: independent of sample order and summation layout (release 2.1).
    offsets, states = compact_ties(mask)
    return (assignments, cost, unknown_slots, math.fsum(cost.tolist()), offsets, states)


def _fit_result(haplotypes, record, iterations):
    """A fit result tuple from an ordered panel (taken over: made read-only) and its
    assignment record."""
    haplotypes.setflags(write=False)
    return (haplotypes, record[0], record[1], record[2], int(iterations), record[3], record[4], record[5])


def _assignment_scalar(workspace, haplotypes, key):
    workspace.fit_schedule_counts["a_evaluations"] += 1
    record = _assignment_record(*assign(workspace, haplotypes, with_ties=True))
    workspace.remember_assignment(key, record)
    return record


def _transition_scalar(workspace, haplotypes, key, schedule_reverse=False):
    record = workspace.cached_assignment(key)
    if record is None:
        record = _assignment_scalar(workspace, haplotypes, key)
    else:
        workspace.fit_schedule_counts["a_reuses_for_h"] += 1
    mask = assignment_tie_mask(record[0], len(haplotypes), record[4], record[5])
    following = np.array(haplotypes, dtype=np.int64, order="C", copy=True)
    changes, tied, commuting = update_founders(
        workspace, following, record[0], mask, schedule_reverse=schedule_reverse,
        with_transition_certificate=True)
    workspace.fit_schedule_counts["h_sweeps"] += 1
    workspace.fit_schedule_counts["tied_h_sweeps"] += int(tied)
    workspace.fit_schedule_counts["commuting_tied_h_sweeps"] += int(tied and commuting)
    workspace.fit_schedule_counts["stationary_tied_h_sweeps"] += int(tied and not commuting and changes == 0)
    return workspace.remember_transition(key, record, workspace.state_key(following), changes,
                                         schedule_reverse, tied, commuting)


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
    mask = np.zeros((b_count, n, n_states), dtype=np.uint8)
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
                threshold = best_cost[t] + tie
                m0 = k
                m1 = k
                category_won = 2
                for category in range(2):
                    lo = 0 if category == 0 else n_rr
                    hi = n_rr if category == 0 else n_states - 1
                    tied = 0
                    for st in range(lo, hi):
                        if -totals[state_indices[panel, st], t] <= threshold:
                            mask[panel, s, st] = 1
                            if st < n_rr:
                                c = first[st]
                                d = second[st]
                            else:
                                c = st - n_rr
                                d = k
                            if tied == 0:
                                m0 = c
                                m1 = d
                            else:
                                m0, m1 = meet_pairs(m0, m1, c, d, k)
                            tied += 1
                    if tied > 0:
                        category_won = category
                        break
                if category_won == 2:
                    mask[panel, s, n_states - 1] = 1
                assignments[panel, s, 0] = m0
                assignments[panel, s, 1] = m1
                unknown_slots[panel, s] = category_won
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
    return assignments, cost, unknown_slots, mask


@njit(cache=True, parallel=True, nogil=True)
def _update_frontier(batch, assignments, unknown_slots, mask, uninformative, genotype_cost_by_sample,
                     unknown_cost_by_sample, tie, reverse, commuting=None, cost_bound=np.inf):
    """Uninformative samples to (U, U), then every panel's one-thread H sweep (parallel over
    panels)."""
    following = batch.copy()
    changes = np.empty(batch.shape[0], dtype=np.int64)
    usage_ties = np.zeros(batch.shape[0], dtype=np.bool_)
    k = batch.shape[1]
    for panel in prange(batch.shape[0]):
        multiple = False
        for sample in range(assignments.shape[1]):
            if uninformative[sample]:
                assignments[panel, sample, 0] = k
                assignments[panel, sample, 1] = k
                unknown_slots[panel, sample] = 2
                mask[panel, sample, :] = 0
                mask[panel, sample, -1] = 1
            size = 0
            for state in range(mask.shape[2]):
                size += mask[panel, sample, state]
            if size > 1:
                multiple = True
        if multiple:
            buckets = fractional_buckets(mask[panel], k)
            order = usage_order(buckets[-1], reverse)
            for pos in range(1, len(order)):
                if buckets[-1][order[pos]] == buckets[-1][order[pos - 1]]:
                    usage_ties[panel] = True
            if commuting is not None:
                commuting[panel] = (not usage_ties[panel] or _fractional_usage_commutes(
                    buckets[-1], buckets[0], buckets[5], buckets[6]))
            changes[panel] = fractional_sweep_one_thread(
                following[panel], order, buckets[0], buckets[1], buckets[2], buckets[3],
                buckets[4], buckets[5], buckets[6], genotype_cost_by_sample, unknown_cost_by_sample,
                tie, cost_bound)
            continue
        (homozygous, with_partner, with_unknown, partner, n_homozygous, n_with_partner,
         n_with_unknown, usage) = founder_buckets(assignments[panel], k)
        order = usage_order(usage, reverse)
        for pos in range(1, len(order)):
            if usage[order[pos]] == usage[order[pos - 1]]:
                usage_ties[panel] = True
        if commuting is not None:
            commuting[panel] = (not usage_ties[panel] or
                                _singleton_usage_commutes(usage, partner, n_with_partner))
        changes[panel] = sweep_one_thread(
            following[panel], order, homozygous, with_partner, with_unknown, partner,
            n_homozygous, n_with_partner, n_with_unknown, genotype_cost_by_sample,
            unknown_cost_by_sample, tie)
    return following, changes, usage_ties


def _row_patterns(rows, snps_per_bin, n_bins):
    first = np.zeros(0, dtype=np.int64)
    patterns, _ = haplotype_patterns(rows, first, first, snps_per_bin, n_bins)
    return patterns


def _batched_assignments(workspace, batch):
    """Exact A(H) for a batch, sharing row/pair emissions without an H sweep."""
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
            left = _batched_assignments(workspace, batch[:half])
            right = _batched_assignments(workspace, batch[half:])
            return tuple(np.concatenate((x, y)) for x, y in zip(left, right))
        tile_width = 1
    tables = workspace.patterns
    patterns = _row_patterns(rows, workspace.snps_per_bin, n_bins)
    row_slots = tables.slots(patterns)
    pair_slots = tables.slots(patterns[pair_i] & patterns[pair_j])
    assignments, cost, unknown_slots, mask = _assign_frontier(
        workspace.c0b, tables.diff1_table, tables.w_table, workspace.kw_cb, tables.kw_diff_table,
        workspace.ww_bin_emissions, workspace.ww_total_cost, row_slots, pair_i, pair_j,
        pair_slots, state_indices, first, second, workspace.switch_penalty, k, n_bins,
        tile_width, float(workspace.quanta.sample))
    # The full transition historically applied this correction in _update_frontier.
    # A-only results need the same neutral (U,U) membership as scalar assign.
    inactive = workspace.uninformative
    if np.any(inactive):
        assignments[:, inactive, :] = k
        unknown_slots[:, inactive] = 2
        mask[:, inactive, :] = 0
        mask[:, inactive, -1] = 1
    return assignments, cost, unknown_slots, mask


def _batched_transitions(workspace, batch, schedule_reverse=None):
    """A(H) then H(A), retaining the exact assignment-only batch as a reusable phase."""
    assignments, cost, unknown_slots, mask = _batched_assignments(workspace, batch)
    following, changes, _usage_ties = _update_frontier(
        batch, assignments, unknown_slots, mask, workspace.uninformative,
        workspace.genotype_cost_by_sample, workspace.unknown_cost_by_sample,
        float(workspace.quanta.h_step), rev("d3") if schedule_reverse is None else bool(schedule_reverse))
    return following, assignments, cost, unknown_slots, changes, mask


def _assignment_batch(workspace, matrices, keys, *, with_dense=False):
    """Cache exact A-only frontiers, with the same bounded dense membership workspace."""
    k = matrices[0].shape[0]
    membership_bytes = len(matrices) * workspace.samples * (k * (k + 1) // 2 + k + 1)
    if len(matrices) > 1 and membership_bytes > _FRONTIER_MEMBERSHIP_BYTES:
        half = len(matrices) // 2
        left = _assignment_batch(workspace, matrices[:half], keys[:half], with_dense=with_dense)
        right = _assignment_batch(workspace, matrices[half:], keys[half:], with_dense=with_dense)
        if with_dense:
            return tuple(np.concatenate((x, y)) for x, y in zip(left, right))
        return
    workspace.fit_schedule_counts["a_evaluations"] += len(matrices)
    batch = np.ascontiguousarray(np.stack(matrices), dtype=np.int64)
    active = numba.get_num_threads()
    threads = min(active, max(_FRONTIER_MIN_THREADS, len(batch)))
    if threads != active:
        numba.set_num_threads(threads)
    try:
        assignments, cost, unknown_slots, mask = _batched_assignments(workspace, batch)
    finally:
        if threads != active:
            numba.set_num_threads(active)
    for index, key in enumerate(keys):
        record = _assignment_record(assignments[index], cost[index], unknown_slots[index], mask[index])
        workspace.remember_assignment(key, record)
    if with_dense:
        return assignments, cost, unknown_slots, mask


def _transition_batch(workspace, matrices, keys, schedule_reverse=False):
    """Share each exact A(H); only H transitions and certificates depend on schedule."""
    k = matrices[0].shape[0]
    membership_bytes = len(matrices) * workspace.samples * (k * (k + 1) // 2 + k + 1)
    if len(matrices) > 1 and membership_bytes > _FRONTIER_MEMBERSHIP_BYTES:
        half = len(matrices) // 2
        _transition_batch(workspace, matrices[:half], keys[:half], schedule_reverse)
        _transition_batch(workspace, matrices[half:], keys[half:], schedule_reverse)
        return
    missing = [(matrix, key) for matrix, key in zip(matrices, keys)
               if workspace.cached_assignment(key) is None]
    workspace.fit_schedule_counts["a_reuses_for_h"] += len(matrices) - len(missing)
    dense = None
    if len(missing) == 1:
        _assignment_scalar(workspace, missing[0][0], missing[0][1])
    elif missing:
        dense = _assignment_batch(workspace, [item[0] for item in missing], [item[1] for item in missing],
                                  with_dense=len(missing) == len(matrices))
    records = [workspace.cached_assignment(key) for key in keys]
    batch = np.ascontiguousarray(np.stack(matrices), dtype=np.int64)
    if dense is not None:
        assignments, _cost, unknown_slots, mask = dense
        workspace.fit_schedule_counts["dense_a_h_reuses"] += len(matrices)
    else:
        assignments = np.stack([record[0] for record in records])
        unknown_slots = np.stack([record[2] for record in records])
        mask = np.stack([assignment_tie_mask(record[0], k, record[4], record[5]) for record in records])
    commuting = np.empty(len(matrices), dtype=np.bool_)
    active = numba.get_num_threads()
    threads = min(active, max(_FRONTIER_MIN_THREADS, len(batch)))
    if threads != active:
        numba.set_num_threads(threads)
    try:
        following, changes, usage_ties = _update_frontier(
            batch, assignments, unknown_slots, mask, workspace.uninformative,
            workspace.genotype_cost_by_sample, workspace.unknown_cost_by_sample,
            float(workspace.quanta.h_step), bool(schedule_reverse), commuting, workspace.h_term_bound)
    finally:
        if threads != active:
            numba.set_num_threads(active)
    workspace.fit_schedule_counts["h_sweeps"] += len(keys)
    workspace.fit_schedule_counts["tied_h_sweeps"] += int(np.count_nonzero(usage_ties))
    workspace.fit_schedule_counts["commuting_tied_h_sweeps"] += int(np.count_nonzero(usage_ties & commuting))
    workspace.fit_schedule_counts["stationary_tied_h_sweeps"] += int(np.count_nonzero(
        usage_ties & ~commuting & (changes == 0)))
    for index, key in enumerate(keys):
        workspace.remember_transition(key, records[index], workspace.state_key(following[index]),
                                      int(changes[index]), schedule_reverse, bool(usage_ties[index]),
                                      bool(commuting[index]))


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
            record = _assignment_scalar(workspace, workspace.state_from_key(key), key)
        if best is None or record[3] < best[3] - workspace.quanta.nll:
            best_key, best = key, record
    return best_key, best


def _frontier_fits(workspace, starts, max_iterations, with_initial, schedule_reverse=False):
    """Fit distinct same-K starts together; returns final fits (and initial assignment-only fits
    when ``with_initial``) in start order, remembering both in the workspace."""
    count = len(starts)
    limit = max(0, int(max_iterations))
    finals = [None] * count
    usage_ties = [False] * count
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
        fit_keys[index] = workspace.fit_key(start, max_iterations, schedule_reverse)
        cached_finals[index] = workspace.cached_fit(fit_keys[index])
        if cached_finals[index] is not None:
            usage_ties[index] = workspace.fit_encountered_usage_tie(fit_keys[index])
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
        workspace.remember_fit(fit_keys[index], result, fixed_point=fixed_point,
                               schedule_reverse=schedule_reverse, usage_tie=usage_ties[index])
        if with_initial:
            workspace.remember_fit(initial_keys[index], initials[index], fixed_point=False)

    assignment_only = limit == 0
    while active:
        missing = {}
        for index in active:
            key = state_keys[index]
            cached = (workspace.cached_assignment(key) if assignment_only
                      else workspace.cached_transition(key, schedule_reverse))
            if cached is None:
                missing.setdefault(key, None)
        if missing:
            matrices = [workspace.state_from_key(key) for key in missing]
            if assignment_only:
                if len(matrices) == 1:
                    _assignment_scalar(workspace, matrices[0], next(iter(missing)))
                else:
                    _assignment_batch(workspace, matrices, tuple(missing))
            elif len(matrices) == 1:
                # A single state keeps the internally parallel A and H steps.
                _transition_scalar(workspace, matrices[0], next(iter(missing)), schedule_reverse)
            else:
                _transition_batch(workspace, matrices, tuple(missing), schedule_reverse)
        next_active = []
        for index in active:
            key = state_keys[index]
            if assignment_only:
                record, next_key, changes, tied = workspace.cached_assignment(key), None, 0, False
            else:
                record, next_key, changes, tied = workspace.cached_transition(key, schedule_reverse)
            if with_initial and initials[index] is None:
                initials[index] = _fit_result(starts[index].copy(), record, 0)
            if cached_finals[index] is not None:
                finish(index, cached_finals[index], False)
                continue
            if completed[index] >= limit:
                finish(index, _fit_result(workspace.state_from_key(key), record, limit), False)
                continue
            usage_ties[index] |= tied
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
        return list(zip(initials, finals)), usage_ties
    return finals, usage_ties


def _scheduled_starts(workspace, starts, max_iterations, with_initial, quanta, schedule_reverse):
    _check_quanta(workspace, quanta)
    starts = [np.ascontiguousarray(start, dtype=np.int64) for start in starts]
    if not starts:
        return [], []
    claim_threads(max_threads=FIT_MAX_THREADS)
    unique, distinct, replay = {}, [], []
    for start in starts:
        key = start.tobytes()
        index = unique.get(key)
        if index is None:
            index = unique[key] = len(distinct)
            distinct.append(start)
        replay.append(index)
    records, tied = _frontier_fits(workspace, distinct, max_iterations, with_initial, schedule_reverse)
    return [records[index] for index in replay], [tied[index] for index in replay]


def fit_starts(workspace, starts, max_iterations, *, with_initial=False, quanta=None,
               schedule_reverse=None):
    """Explicit single-schedule fit primitive; max_iterations=0 is unchanged exact A(H).

    Biological discovery and bank consumers use fit_families. The optional legacy schedule
    default remains for reference callers; it never chooses a production family endpoint.
    """
    reverse = rev("d3") if schedule_reverse is None else bool(schedule_reverse)
    return _scheduled_starts(workspace, starts, max_iterations, with_initial, quanta, reverse)[0]


def fit_families(workspace, starts, max_iterations, *, with_initial=False, quanta=None):
    """Complete endpoints of both consistent usage-tie schedules, per canonical start.

    No second trajectory is run when the first encounters no positive-usage tie. Exact A(H)
    and untied H transitions are shared. Assignment-only calls have one unchanged endpoint.
    """
    starts = [np.ascontiguousarray(start, dtype=np.int64) for start in starts]
    if max_iterations > 0:
        starts = [np.ascontiguousarray(start[np.lexsort(start.T[::-1])]) for start in starts]
    primary, tied = _scheduled_starts(workspace, starts, max_iterations, with_initial, quanta, False)
    branch = [index for index, value in enumerate(tied) if value]
    workspace.fit_schedule_counts["family_starts"] += len(starts)
    workspace.fit_schedule_counts["branched_starts"] += len(branch)
    alternate = {}
    if branch:
        records, _ = _scheduled_starts(workspace, [starts[index] for index in branch],
                                       max_iterations, False, quanta, True)
        alternate = dict(zip(branch, records))
    result = []
    for index, record in enumerate(primary):
        initial, final = record if with_initial else (None, record)
        endpoints = [final]
        if index in alternate:
            endpoints.append(alternate[index])
        unique = {}
        for endpoint in endpoints:
            unique.setdefault(panel_key(endpoint[0]), endpoint)
        family = tuple(unique[key] for key in sorted(unique))
        result.append((initial, family) if with_initial else family)
    return result


def fit_panels(workspace, starts, max_iterations, *, quanta=None, with_families=False):
    """Synchronized and refitted solutions, retaining both schedule endpoints per start.

    Row collapse recurses at lower K. With ``with_families``, a third collection gives each
    original start's complete canonical endpoint keys, including its synchronized endpoint
    and all lower-K descendants. Bank score limits admit those families atomically.
    """
    _check_quanta(workspace, quanta)
    quanta = workspace.quanta
    pending, roots, seen, certified_before = [], [], set(), {}
    for start in starts:
        rows = np.ascontiguousarray(unique_rows(np.asarray(start)), dtype=np.int64)
        key = workspace.state_key(rows)
        if key in seen:
            continue
        seen.add(key)
        roots.append(key)
        pending.append(rows)
        certified_before[key] = workspace.certifies_fixed_point(rows)
    synchronized, refitted = [], []
    initial_keys, endpoints, collapsed = {}, {}, {}
    capture_initial = True
    while pending:
        grouped = {}
        for rows in pending:
            grouped.setdefault(len(rows), []).append(rows)
        pending = []
        for k in sorted(grouped):
            group = grouped[k]
            if capture_initial:
                records = fit_families(workspace, group, max_iterations, with_initial=True)
            else:
                records = [(None, family) for family in fit_families(workspace, group, max_iterations)]
            for start, (initial, family) in zip(group, records):
                source_key = workspace.state_key(start)
                endpoints.setdefault(source_key, set())
                collapsed.setdefault(source_key, set())
                if initial is not None:
                    solution = canonical_solution(initial,
                        workspace if certified_before[source_key] else None, quanta)
                    synchronized.append(solution)
                    initial_keys[source_key] = solution.key
                for final in family:
                    solution = canonical_solution(final, workspace, quanta)
                    rows = np.ascontiguousarray(unique_rows(solution.haplotypes), dtype=np.int64)
                    if len(rows) == solution.k:
                        refitted.append(solution)
                        endpoints[source_key].add(solution.key)
                        continue
                    key = workspace.state_key(rows)
                    collapsed[source_key].add(key)
                    if key not in seen:
                        seen.add(key)
                        pending.append(rows)
        capture_initial = False
    synchronized = deduplicate(synchronized, quanta=quanta)
    refitted = deduplicate(refitted, quanta=quanta)
    if not with_families:
        return synchronized, refitted
    families = []
    for root in roots:
        keys = {initial_keys[root]}
        queue, reached = [root], set()
        while queue:
            source = queue.pop()
            if source in reached:
                continue
            reached.add(source)
            keys.update(endpoints[source])
            queue.extend(collapsed[source])
        families.append(tuple(sorted(keys)))
    return synchronized, refitted, tuple(families)


def fit_one(workspace, haplotypes, max_iterations, *, quanta=None, schedule_reverse=None):
    """The fit tuple of one start."""
    return fit_starts(workspace, [haplotypes], max_iterations, quanta=quanta,
                      schedule_reverse=schedule_reverse)[0]


def _check_quanta(workspace, quanta):
    if quanta is not None and quanta != workspace.quanta:
        raise ValueError("a fixed-K fit uses its workspace's tie quanta; pass None or workspace.quanta")


def prepare_workspace(likelihoods, observed, model, quanta=None) -> FixedKWorkspace:
    """A block's fixed-K workspace with uniform winning-category tie evidence in every path."""
    return FixedKWorkspace(likelihoods, observed, model, quanta)
