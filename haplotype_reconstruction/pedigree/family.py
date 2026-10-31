"""Finite family evidence: orientation correction, reciprocal cavity messages, short ancestry paths.

All terms are composite-evidence corrections added to both the state and the identity scores of a
configuration; genetic scoring and the parent-count priors are unchanged.
1. Orientation: log(2 P(parent -> child | one orientation holds)) from the junction log odds, mixed
   with a pair-level painting-failure probability epsilon: within [log(epsilon), log(2 - epsilon)].
2. Reciprocal family: pairwise anti-reciprocity factors (p cannot be c's parent and c p's), solved by
   synchronous cavity messages that exclude immediate reverse feedback (loopy BP).
3. Short ancestry paths: a joint factor against reverse two-edge paths (c -> g -> p when p is c's
   parent) over a bounded top-R configuration panel per sample; shared intermediates count once,
   both focal parents are excluded, omitted panel mass is neutral (truncation only weakens it).
The path factor is prepared from four synchronous passes, then frozen while a converged final
reciprocal solve replaces the preparatory reciprocal term. Every edge's direction is inferred: no
edge is exempt from the three terms (release 3.4: no design chronology from sample metadata).
"""
from __future__ import annotations

import math

import numpy as np
from numba import get_num_threads, njit, prange

from .config import DirectionConfig
from .orientation import Orientation


class FamilyMessageNonConvergence(FloatingPointError):
    """The final reciprocal solve missed its tolerance: an error in the full-data fit, an abstaining
    replicate in resampling."""


# Parallel layout (release 3.4, PEDFAST): every kernel below is parallel over independent children, parents or
# rows, and each output element is computed by the same operations in the same order as the serial kernels of 3.3
# (no floating-point reduction is split across threads: the residual's maximum is folded serially), so the results do
# not depend on the thread count. Work is dealt to threads in interleaved lanes (item = lane, lane + lanes, ...) so
# that children with many configuration rows spread over the team.
_LANES_PER_THREAD = 8
# exp(x) is exactly +0.0 in float64 for every x < -745.14 (below half the smallest subnormal), and log(1) and log1p(0)
# are exactly 0: the kernels skip those evaluations, whose results they already know, so every value is unchanged
# (bit for bit) while far-below-maximum rows, which dominate label-free panels at large N, cost no transcendental.
_EXP_UNDERFLOW = 750.0


@njit(cache=True, nogil=True, inline="always")
def _exp_or_zero(x):
    """math.exp(x), skipped where it is exactly zero (x < -750; NaN still evaluates)."""
    return 0.0 if x < -_EXP_UNDERFLOW else math.exp(x)


@njit(cache=True, nogil=True, inline="always")
def _log_or_zero(x):
    """math.log(x), skipped where it is exactly zero (x == 1)."""
    return 0.0 if x == 1.0 else math.log(x)


@njit(cache=True, nogil=True, inline="always")
def _softplus_tail(magnitude):
    """np.log1p(np.exp(-magnitude)) for magnitude >= 0, skipped where it is exactly zero (magnitude > 750)."""
    return 0.0 if magnitude > _EXP_UNDERFLOW else np.log1p(np.exp(-magnitude))


@njit(cache=True, nogil=True, inline="always")
def _logaddexp(a, b):
    """maths.logspace.logaddexp with its log1p(exp(-|a - b|)) skipped where that is exactly zero."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    gap = abs(a - b)
    return max(a, b) + (0.0 if gap > _EXP_UNDERFLOW else math.log1p(math.exp(-gap)))


@njit(cache=True, nogil=True)
def _lane_count(items):
    return max(1, min(items, _LANES_PER_THREAD * get_num_threads()))


@njit(cache=True, nogil=True)
def _child_log_odds(log_scores, rows, states, log_multiplicity, samples, child, start, stop, output):
    """One child's row of _family_log_odds (the serial 3.3 body, unchanged)."""
    maximum = -np.inf
    best = -1
    included_max = np.full(samples, -np.inf)
    for row in range(start, stop):
        value = log_scores[row] + log_multiplicity[child, states[row]]
        if value > maximum:
            maximum, best = value, row
        for slot in range(1, 3):
            parent = rows[row, slot]
            if parent >= 0:
                included_max[parent] = max(included_max[parent], value)
    if best < 0:
        return
    mass = np.zeros(samples)
    included_sum = np.zeros(samples)
    total = 0.0
    for row in range(start, stop):
        value = log_scores[row] + log_multiplicity[child, states[row]]
        scaled = _exp_or_zero(value - maximum)
        total += scaled
        for slot in range(1, 3):
            parent = rows[row, slot]
            if parent >= 0 and np.isfinite(included_max[parent]):
                mass[parent] += scaled
                included_sum[parent] += _exp_or_zero(value - included_max[parent])
    for parent in range(samples):
        if not np.isfinite(included_max[parent]):
            continue
        if parent == rows[best, 1] or parent == rows[best, 2]:
            excluded_max = -np.inf
            for row in range(start, stop):
                if rows[row, 1] != parent and rows[row, 2] != parent:
                    value = log_scores[row] + log_multiplicity[child, states[row]]
                    excluded_max = max(excluded_max, value)
            excluded_sum = 0.0
            for row in range(start, stop):
                if rows[row, 1] != parent and rows[row, 2] != parent:
                    excluded_sum += _exp_or_zero(
                        log_scores[row] + log_multiplicity[child, states[row]] - excluded_max)
            log_excluded = excluded_max + _log_or_zero(excluded_sum)
        else:
            # Exact: the best row excludes this parent, so the excluded mass is >= 1.
            log_excluded = maximum + _log_or_zero(max(1.0, total - mass[parent]))
        output[child, parent] = included_max[parent] + _log_or_zero(included_sum[parent]) - log_excluded


@njit(cache=True, nogil=True, parallel=True)
def _family_log_odds(log_scores, rows, states, log_multiplicity, samples):
    """Included/excluded log odds of every [child, parent] in O(A + N^2), stably.

    Included masses have an independent scale per parent, so a tiny included probability survives
    the removal of a strong incoming reverse message; exclusions containing the maximum row share
    its scale and at most two exclusions per child need a separate scan.
    """
    output = np.full((samples, samples), -np.inf)
    starts = np.searchsorted(rows[:, 0], np.arange(samples + 1))
    lanes = _lane_count(samples)
    for lane in prange(lanes):
        for child in range(lane, samples, lanes):
            _child_log_odds(log_scores, rows, states, log_multiplicity, samples, child, starts[child],
                            starts[child + 1], output)
    return output


@njit(cache=True, nogil=True, parallel=True)
def _with_incoming(log_scores, rows, incoming):
    """log_scores plus each row's incoming messages, added slot by slot; ``incoming[child, parent]`` holds the message
    previous[parent, child] (the transposed messages: a child's rows read one contiguous row of it)."""
    adjusted = np.empty_like(log_scores)
    for row in prange(len(rows)):
        value = log_scores[row]
        child = rows[row, 0]
        for slot in range(1, 3):
            parent = rows[row, slot]
            if parent >= 0:
                value += incoming[child, parent]
        adjusted[row] = value
    return adjusted


@njit(cache=True, nogil=True, parallel=True)
def _cavity_passes(log_scores, rows, states, log_multiplicity, samples, passes):
    """Bounded synchronous cavity-message passes (immediate reverse feedback excluded)."""
    previous = np.zeros((samples, samples))
    for _ in range(passes):
        incoming = np.ascontiguousarray(previous.T)
        adjusted = _with_incoming(log_scores, rows, incoming)
        odds = _family_log_odds(adjusted, rows, states, log_multiplicity, samples)
        updated = np.empty((samples, samples))
        for child in prange(samples):
            for parent in range(samples):
                # Remove incoming evidence before converting odds to support.
                value = odds[child, parent] - incoming[child, parent]
                updated[child, parent] = -np.maximum(0.0, value) - _softplus_tail(np.abs(value))
        previous = updated
    return previous


@njit(cache=True, nogil=True, parallel=True)
def _cavity_iterations(scores, rows, states, log_multiplicity, samples, previous, history, first, stop,
                       tolerance, damping, window, horizon, residual):
    """Iterations ``first`` .. ``stop - 1`` of the converged cavity solve (see _converged_cavity), resumable.

    Returns (messages, iterations done, residual, status): status 1 = converged (the messages are the converged
    odds), 2 = paused, 0 = ``stop`` reached (the messages are the damped state to continue from). ``history`` (the
    pause rule's residual ring) is updated in place.
    """
    for iteration in range(first, stop):
        incoming = np.ascontiguousarray(previous.T)
        adjusted = _with_incoming(scores, rows, incoming)
        odds = _family_log_odds(adjusted, rows, states, log_multiplicity, samples)
        change = np.empty((samples, samples))
        for child in prange(samples):
            for parent in range(samples):
                value = odds[child, parent] - incoming[child, parent]
                value = -max(0.0, value) - _softplus_tail(abs(value))
                change[child, parent] = abs(value - previous[child, parent])
                odds[child, parent] = value
        # The residual is folded serially, in the 3.3 order.
        residual = 0.0
        for child in range(samples):
            for parent in range(samples):
                residual = max(residual, change[child, parent])
        if residual <= tolerance:
            return odds, iteration + 1, residual, 1
        previous = (1.0 - damping) * previous + damping * odds
        if window > 0:
            done = iteration + 1
            position = done % window
            earlier = history[position]
            history[position] = residual
            if done >= 2 * window and done % window == 0:
                # Pause without progress over the last window, or when that window's geometric rate
                # needs more than ``horizon`` iterations in all to reach the tolerance.
                if residual >= earlier or done + window * math.log(tolerance / residual) / math.log(
                        residual / earlier) > horizon:
                    return previous, done, residual, 2
    return previous, stop, residual, 0


# A fit claims its CPU share (execution.claim_threads) before each family kernel and between chunks of the converged
# solve, so that resampling fits widen as the pool drains; a fit whose rows and messages (A + N^2) are fewer than this
# runs single-threaded (parallel start-up would cost more than it saves).
_PARALLEL_MINIMUM_WORK = 200_000
# The converged solve claims again after about this much work (A + N^2 per iteration).
_CLAIM_WORK = 20_000_000


def _claim(work):
    """Apply this process's current CPU share to the Numba team (one thread for small fits)."""
    from ..execution import claim_threads
    return claim_threads(max_threads=1 if work < _PARALLEL_MINIMUM_WORK else None)


def _converged_cavity(scores, rows, states, log_multiplicity, samples, maximum, tolerance,
                      damping, start, window, horizon):
    """The same cavity equations iterated from ``start`` to the undamped fixed-point residual.

    Returns (messages, iterations, residual, paused). With window > 0 the solve pauses (paused True, the
    messages being the state to resume from) once two windows have run and the residual's geometric rate over
    the last window cannot reach ``tolerance`` within ``horizon`` iterations; window 0 never pauses.
    Runs in chunks of iterations (the same iterations as one pass), claiming threads between chunks.
    """
    work = len(rows) + samples * samples
    chunk = max(1, _CLAIM_WORK // max(1, work))
    previous = start.copy()
    residual = np.inf
    history = np.full(max(window, 1), np.inf)
    done = 0
    while done < maximum:
        _claim(work)
        previous, done, residual, status = _cavity_iterations(
            scores, rows, states, log_multiplicity, samples, previous, history, done, min(maximum, done + chunk),
            tolerance, damping, window, horizon, residual)
        if status == 1:
            return previous, done, residual, False
        if status == 2:
            return previous, done, residual, True
    return previous, maximum, residual, False


def _incoming_messages(avoidance, rows, eligible):
    """Per row, the sum of the incoming reciprocal messages of its parents."""
    family = np.zeros(len(rows))
    for slot in (1, 2):
        present = (rows[:, slot] >= 0) & eligible
        child, parent = rows[present, 0], rows[present, slot]
        family[present] += avoidance[parent, child]
    return family


def _pair_orientation_log_odds(adjusted, rows, states, log_multiplicity, samples, applied):
    """[c, p]: log odds that p is c's parent rather than c p's, given a link.

    ``adjusted`` already carries the applied reciprocal messages; ``applied[p, c]`` is the one added
    to c's configurations naming p. With cavity inclusion log odds a = O[c, p] and b = O[p, c], the
    pair's joint states p -> c, c -> p and neither weigh e^a : e^b : 1. Infinite where only one
    orientation is scored; NaN where neither is.
    """
    cavity = _family_log_odds(adjusted, rows, states, log_multiplicity, samples) - applied.T
    with np.errstate(invalid="ignore"):
        return cavity - cavity.T


@njit(cache=True, nogil=True)
def _sample_panel(scores, rows, states, log_multiplicity, parent, start, stop, budget, selected, weights, tail):
    """One sample's row of _configuration_panels (the serial 3.3 body, unchanged)."""
    for row in range(start, stop):
        value = scores[row] + log_multiplicity[parent, states[row]]
        if not np.isfinite(value):
            continue
        minimum = np.argmin(weights[parent])
        if value > weights[parent, minimum]:
            tail[parent] = _logaddexp(tail[parent], weights[parent, minimum])
            weights[parent, minimum], selected[parent, minimum] = value, row
        else:
            tail[parent] = _logaddexp(tail[parent], value)
    boundary = np.min(weights[parent])
    if np.isfinite(boundary):
        found = 0
        for row in range(start, stop):
            if scores[row] + log_multiplicity[parent, states[row]] >= boundary:
                found += 1
        if found > budget:
            for j in range(budget):
                if weights[parent, j] == boundary:
                    tail[parent] = _logaddexp(tail[parent], weights[parent, j])
                    weights[parent, j], selected[parent, j] = -np.inf, -1
    # Keep even tiny omitted mass in log space until all exclusions are applied.
    center = max(tail[parent], np.max(weights[parent]))
    if np.isfinite(center):
        tail[parent] -= center
        weights[parent] -= center


@njit(cache=True, nogil=True, parallel=True)
def _configuration_panels(scores, rows, states, log_multiplicity, samples, budget):
    """Top ``budget`` configurations per sample; boundary ties move to the neutral tail."""
    starts = np.searchsorted(rows[:, 0], np.arange(samples + 1))
    selected = np.full((samples, budget), -1, dtype=np.int64)
    weights = np.full((samples, budget), -np.inf)
    tail = np.full(samples, -np.inf)
    lanes = _lane_count(samples)
    for lane in prange(lanes):
        for parent in range(lane, samples, lanes):
            _sample_panel(scores, rows, states, log_multiplicity, parent, starts[parent], starts[parent + 1],
                          budget, selected, weights, tail)
    return selected, weights, tail


@njit(cache=True, nogil=True)
def _path_singles(parent, n, rows, selected, weights, tail, excluded, budget, retained, denominator, single):
    """One parent's row of the single-parent path factor (the serial 3.3 body, unchanged)."""
    for child in range(n):
        numerator = tail[parent]
        for j in range(budget):
            row = selected[parent, j]
            if row < 0:
                continue
            a, b = rows[row, 1], rows[row, 2]
            if a == child or b == child:
                continue
            value = weights[parent, j]
            retained[parent, child] = _logaddexp(retained[parent, child], value)
            if a >= 0:
                value += excluded[a, child]
            if b >= 0:
                value += excluded[b, child]
            numerator = _logaddexp(numerator, value)
        denominator[parent, child] = _logaddexp(tail[parent], retained[parent, child])
        if np.isfinite(denominator[parent, child]):
            single[parent, child] = min(0., numerator - denominator[parent, child])


@njit(cache=True, nogil=True)
def _path_target(target, rows, scores, selected, weights, tail, retained, denominator, single, excluded_by_child,
                 budget):
    """One configuration row's path factor (the serial 3.3 body; excluded_by_child[c, g] = excluded[g, c], the same
    values laid out by child)."""
    child, p, q = rows[target, 0], rows[target, 1], rows[target, 2]
    if p < 0 or not np.isfinite(scores[target]):
        return 0.0
    if q < 0:
        return single[p, child]
    if not np.isfinite(denominator[p, child] + denominator[q, child]):
        return 0.0
    # Tail configurations are fully compatible, including omitted direct reversals; this can
    # only dilute the conditional penalty.
    numerator = _logaddexp(tail[p] + denominator[q, child], retained[p, child] + tail[q])
    for i in range(budget):
        ri = selected[p, i]
        if ri < 0:
            continue
        a, b = rows[ri, 1], rows[ri, 2]
        if a == child or b == child:
            continue
        for j in range(budget):
            rj = selected[q, j]
            if rj < 0:
                continue
            d, e = rows[rj, 1], rows[rj, 2]
            if d == child or e == child:
                continue
            value = weights[p, i] + weights[q, j]
            for g, slot in ((a, 0), (b, 1), (d, 2), (e, 3)):
                if g < 0 or g == p or g == q:
                    continue
                if slot >= 1 and g == a:
                    continue
                if slot >= 2 and g == b:
                    continue
                if slot >= 3 and g == d:
                    continue
                value += excluded_by_child[child, g]
            numerator = _logaddexp(numerator, value)
    return min(0., numerator - denominator[p, child] - denominator[q, child])


@njit(cache=True, nogil=True, parallel=True)
def _path_correction(scores, rows, states, log_multiplicity, excluded, budget):
    """Reverse two-edge path factor per row; O(R A + R N^2 + R^2 A2), R = budget.

    ``excluded[g, c]`` is the outgoing cavity log probability that g does not choose c as parent.
    """
    n = len(excluded)
    selected, weights, tail = _configuration_panels(scores, rows, states, log_multiplicity, n,
                                                    budget)
    retained = np.full((n, n), -np.inf)
    denominator = np.full((n, n), -np.inf)
    single = np.zeros((n, n))
    lanes = _lane_count(n)
    for lane in prange(lanes):
        for parent in range(lane, n, lanes):
            _path_singles(parent, n, rows, selected, weights, tail, excluded, budget, retained, denominator, single)
    excluded_by_child = np.ascontiguousarray(excluded.T)
    result = np.zeros(len(states))
    count = len(states)
    lanes = _lane_count(count)
    for lane in prange(lanes):
        for target in range(lane, count, lanes):
            result[target] = _path_target(target, rows, scores, selected, weights, tail, retained, denominator,
                                          single, excluded_by_child, budget)
    return result


def adjust(state_scores: np.ndarray, identity_scores: np.ndarray, rows: np.ndarray,
           states: np.ndarray, full_counts: np.ndarray, orientation: Orientation,
           config: DirectionConfig, trace: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Add the orientation, path and converged reciprocal terms to state and identity scores.

    Exposure-untestable identities (-inf) stay impossible. With ``trace`` (full-data fit only) the
    per-row terms, the pair orientation log odds and the final solve are recorded in it.
    Raises FamilyMessageNonConvergence when the final solve misses its tolerance.
    """
    samples = orientation.log_odds.shape[0]
    eligible = np.isfinite(identity_scores)
    # log(2 P(parent -> child | one orientation holds)); a sample-level painting excess repeats on
    # every chromosome, so a pair-level failure mixture bounds it. Zero for equal support.
    epsilon = max(config.pair_failure_probability, np.finfo(float).eps)
    log_support = -np.logaddexp(0.0, -orientation.log_odds)
    log_direction = math.log(2.0) + np.logaddexp(math.log1p(-epsilon) + log_support,
                                                 math.log(0.5 * epsilon))
    direction = np.zeros(len(rows))
    for slot in (1, 2):
        present = (rows[:, slot] >= 0) & eligible
        direction[present] += log_direction[rows[present, 0], rows[present, slot]]
    state_scores = state_scores + direction
    identity_scores = identity_scores + direction

    # Equal state weights in this likelihood-only message; the configured priors act later.
    log_multiplicity = -np.log(np.maximum(1, full_counts))
    work = len(rows) + samples * samples
    _claim(work)
    avoidance = _cavity_passes(identity_scores, rows, states, log_multiplicity, samples,
                               config.message_passes)
    preparatory = _incoming_messages(avoidance, rows, eligible)
    _claim(work)
    paths = _path_correction(identity_scores + preparatory, rows, states, log_multiplicity,
                             avoidance, config.ancestry_path_budget)
    # The path factor is frozen; the final reciprocal solve replaces the preparatory term.
    state_scores = state_scores + paths
    identity_scores = identity_scores + paths
    zero = np.zeros((samples, samples))
    avoidance, iterations, residual, paused = _converged_cavity(
        identity_scores, rows, states, log_multiplicity, samples, config.final_max_iterations,
        config.final_tolerance, 1.0, zero, config.final_progress_window,
        config.final_pause_budgets * config.final_max_iterations)
    retry_iterations = 0
    if residual > config.final_tolerance:
        # Loopy messages can oscillate in a resample: restart damped, keeping the undamped residual.
        undamped = avoidance
        avoidance, retry_iterations, residual, _ = _converged_cavity(
            identity_scores, rows, states, log_multiplicity, samples, config.retry_iterations,
            config.final_tolerance, config.retry_damping, zero, 0, 0)
        if residual > config.final_tolerance and paused:
            # The undamped solve paused for lack of progress; give it the rest of its budget.
            avoidance, resumed, residual, _ = _converged_cavity(
                identity_scores, rows, states, log_multiplicity, samples,
                config.final_max_iterations - iterations, config.final_tolerance, 1.0, undamped, 0, 0)
            iterations += resumed
    if residual > config.final_tolerance:
        raise FamilyMessageNonConvergence(
            f"Final reciprocal family messages did not converge in {iterations} initial + "
            f"{retry_iterations} damped iterations (undamped log-message residual {residual:g})")
    family = _incoming_messages(avoidance, rows, eligible)
    state_scores = state_scores + family
    identity_scores = identity_scores + family
    if trace is not None:
        _claim(work)
        trace.update(
            direction=direction, ancestry_paths=paths, reciprocal_family=family,
            orientation_log_odds=_pair_orientation_log_odds(
                identity_scores, rows, states, log_multiplicity, samples, avoidance),
            final_family_iterations=iterations + retry_iterations,
            final_family_retry_iterations=retry_iterations, final_family_residual=residual)
    return state_scores, identity_scores
