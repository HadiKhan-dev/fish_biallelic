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
edge is exempt from the three terms; sample metadata supplies no design chronology.

Method: docs/methods/pedigree.md#converged-final-reciprocal-solve
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import prange

from ...core.kernels import kernel
from .decision_parameters import DirectionParameters
from .family_numerics import _exp_or_zero, _lane_count, _log_or_zero, _softplus_tail
from .family_paths import _path_correction
from .kernels.decision import OVERLOADS
from .orientation import Orientation
from .workspace_parameters import FAMILY_CLAIM_WORK, FAMILY_TRANSPOSE_BLOCK


@dataclass(frozen=True)
class FamilySolveDiagnostics:
    """The actual pause, retry and resume path of one final reciprocal solve."""

    initial_iterations: int
    damped_iterations: int
    resumed_iterations: int
    initial_paused: bool
    residual: float
    converged: bool


class FamilyMessageNonConvergence(FloatingPointError):
    """A failed full fit, or an abstaining resample, with typed solve evidence."""

    def __init__(self, message: str, diagnostics: FamilySolveDiagnostics | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics


# Parallel layout: every kernel below is parallel over independent children, parents or
# rows. Each output element follows its serial operation order; no floating-point reduction is split across
# threads, and the residual's maximum is folded serially, so the results do
# not depend on the thread count. Work is dealt to threads in interleaved lanes (item = lane, lane + lanes, ...) so
# that children with many configuration rows spread over the team.
# Read the current thread count at each Python entry after claiming the CPU share. Passing it into
# compiled code keeps the same lane schedule without embedding a dynamic runtime address in its cache.
@kernel(overloads=OVERLOADS["family._child_log_odds"], cache=True, nogil=True)
def _child_log_odds(log_scores, rows, states, log_multiplicity, samples, child, start, stop, output):
    """One child's parent-inclusion log odds from finite scores and state multiplicities."""
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
                    excluded_sum += _exp_or_zero(log_scores[row] + log_multiplicity[child, states[row]] - excluded_max)
            log_excluded = excluded_max + _log_or_zero(excluded_sum)
        else:
            # Exact: the best row excludes this parent, so the excluded mass is >= 1.
            log_excluded = maximum + _log_or_zero(max(1.0, total - mass[parent]))
        output[child, parent] = included_max[parent] + _log_or_zero(included_sum[parent]) - log_excluded


@kernel(overloads=OVERLOADS["family._family_log_odds"], cache=True, nogil=True, parallel=True)
def _family_log_odds(log_scores, rows, states, log_multiplicity, samples, threads):
    """Included/excluded log odds of every [child, parent] in O(A + N^2), stably.

    Included masses have an independent scale per parent, so a tiny included probability survives
    the removal of a strong incoming reverse message; exclusions containing the maximum row share
    its scale and at most two exclusions per child need a separate scan.
    """
    output = np.full((samples, samples), -np.inf)
    starts = np.searchsorted(rows[:, 0], np.arange(samples + 1))
    lanes = _lane_count(samples, threads)
    for lane in prange(lanes):
        for child in range(lane, samples, lanes):
            _child_log_odds(
                log_scores, rows, states, log_multiplicity, samples, child, starts[child], starts[child + 1], output
            )
    return output


@kernel(overloads=OVERLOADS["family._with_incoming"], cache=True, nogil=True, parallel=True)
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


@kernel(overloads=OVERLOADS["family._cavity_passes"], cache=True, nogil=True, parallel=True)
def _cavity_passes(log_scores, rows, states, log_multiplicity, samples, passes, threads):
    """Bounded synchronous cavity-message passes (immediate reverse feedback excluded)."""
    previous = np.zeros((samples, samples))
    for _ in range(passes):
        incoming = np.ascontiguousarray(previous.T)
        adjusted = _with_incoming(log_scores, rows, incoming)
        odds = _family_log_odds(adjusted, rows, states, log_multiplicity, samples, threads)
        updated = np.empty((samples, samples))
        for child in prange(samples):
            for parent in range(samples):
                # Remove incoming evidence before converting odds to support.
                value = odds[child, parent] - incoming[child, parent]
                updated[child, parent] = -np.maximum(0.0, value) - _softplus_tail(np.abs(value))
        previous = updated
    return previous


@kernel(overloads=OVERLOADS["family._child_cavity"], cache=True, nogil=True)
def _child_cavity(
    scores,
    rows,
    states,
    log_multiplicity,
    samples,
    child,
    start,
    stop,
    incoming,
    previous,
    values,
    included_max,
    mass,
    included_sum,
    row_odds,
    messages,
):
    """One child's row of a final-solve iteration: _with_incoming, _child_log_odds and the cavity update of
    _cavity_iterations fused. The arithmetic follows each row's slot order: adjusted row values are computed
    once and reused by every scan, and the best row's two
    parents share one pair of exclusion scans (each parent's own maximum and sum, over the rows in order).
    Writes messages[child] and returns the row's residual (the maximum of its changes) and whether a change is NaN.
    """
    maximum = -np.inf
    best = -1
    for parent in range(samples):
        included_max[parent] = -np.inf
        row_odds[parent] = -np.inf
    for row in range(start, stop):
        # _with_incoming, then _child_log_odds's value.
        value = scores[row]
        for slot in range(1, 3):
            parent = rows[row, slot]
            if parent >= 0:
                value += incoming[child, parent]
        value = value + log_multiplicity[child, states[row]]
        values[row - start] = value
        if value > maximum:
            maximum, best = value, row
        for slot in range(1, 3):
            parent = rows[row, slot]
            if parent >= 0:
                included_max[parent] = max(included_max[parent], value)
    if best >= 0:
        for parent in range(samples):
            mass[parent] = 0.0
            included_sum[parent] = 0.0
        total = 0.0
        for row in range(start, stop):
            value = values[row - start]
            scaled = _exp_or_zero(value - maximum)
            total += scaled
            for slot in range(1, 3):
                parent = rows[row, slot]
                if parent >= 0 and np.isfinite(included_max[parent]):
                    mass[parent] += scaled
                    included_sum[parent] += _exp_or_zero(value - included_max[parent])
        # The best row's parents: excluded masses need their own scans (both in one pass).
        first, second = rows[best, 1], rows[best, 2]
        scan_first = first >= 0 and np.isfinite(included_max[first])
        scan_second = second >= 0 and np.isfinite(included_max[second])
        first_max = -np.inf
        second_max = -np.inf
        first_excluded = -np.inf
        second_excluded = -np.inf
        if scan_first or scan_second:
            for row in range(start, stop):
                a, b = rows[row, 1], rows[row, 2]
                value = values[row - start]
                if scan_first and a != first and b != first:
                    first_max = max(first_max, value)
                if scan_second and a != second and b != second:
                    second_max = max(second_max, value)
            first_sum = 0.0
            second_sum = 0.0
            for row in range(start, stop):
                a, b = rows[row, 1], rows[row, 2]
                value = values[row - start]
                if scan_first and a != first and b != first:
                    first_sum += _exp_or_zero(value - first_max)
                if scan_second and a != second and b != second:
                    second_sum += _exp_or_zero(value - second_max)
            first_excluded = first_max + _log_or_zero(first_sum)
            second_excluded = second_max + _log_or_zero(second_sum)
        for parent in range(samples):
            if not np.isfinite(included_max[parent]):
                continue
            if parent == first:
                log_excluded = first_excluded
            elif parent == second:
                log_excluded = second_excluded
            else:
                # Exact: the best row excludes this parent, so the excluded mass is >= 1.
                log_excluded = maximum + _log_or_zero(max(1.0, total - mass[parent]))
            row_odds[parent] = included_max[parent] + _log_or_zero(included_sum[parent]) - log_excluded
    residual = 0.0
    undefined = False
    for parent in range(samples):
        value = row_odds[parent] - incoming[child, parent]
        value = -max(0.0, value) - _softplus_tail(abs(value))
        change = abs(value - previous[child, parent])
        messages[child, parent] = value
        residual = max(residual, change)
        undefined |= change != change
    return residual, undefined


@kernel(overloads=OVERLOADS["family._lane_cavity"], cache=True, nogil=True)
def _lane_cavity(
    scores,
    rows,
    states,
    log_multiplicity,
    samples,
    lane,
    lanes,
    starts,
    widest,
    incoming,
    previous,
    messages,
    row_residual,
    row_undefined,
):
    """The children lane, lane + lanes, ... of one final-solve iteration, with this lane's own scratch rows."""
    values = np.empty(widest)
    included_max = np.empty(samples)
    mass = np.empty(samples)
    included_sum = np.empty(samples)
    row_odds = np.empty(samples)
    for child in range(lane, samples, lanes):
        residual, undefined = _child_cavity(
            scores,
            rows,
            states,
            log_multiplicity,
            samples,
            child,
            starts[child],
            starts[child + 1],
            incoming,
            previous,
            values,
            included_max,
            mass,
            included_sum,
            row_odds,
            messages,
        )
        row_residual[child] = residual
        row_undefined[child] = undefined


@kernel(overloads=OVERLOADS["family._cavity_iterations"], cache=True, nogil=True, parallel=True)
def _cavity_iterations(
    scores,
    rows,
    states,
    log_multiplicity,
    samples,
    previous,
    history,
    first,
    stop,
    tolerance,
    damping,
    window,
    horizon,
    residual,
    threads,
):
    """Iterations ``first`` .. ``stop - 1`` of the converged cavity solve (see _converged_cavity), resumable.

    Returns (messages, iterations done, residual, status): status 1 = converged (the messages are the converged
    odds), 2 = paused, 0 = ``stop`` reached (the messages are the damped state to continue from). ``history`` (the
    pause rule's residual ring) is updated in place. ``previous`` is read, never written.

    Each iteration fuses incoming-message adjustment, family log odds and the cavity update in one parallel
    pass (_child_cavity), reusing buffers. The residual is a maximum of non-negative changes, which any grouping
    returns exactly; a NaN change triggers a serial fold in child/parent slot order because NaN handling depends
    on order.
    """
    starts = np.searchsorted(rows[:, 0], np.arange(samples + 1))
    widest = 0
    for child in range(samples):
        widest = max(widest, starts[child + 1] - starts[child])
    lanes = _lane_count(samples, threads)
    incoming = np.empty((samples, samples))
    messages = np.empty((samples, samples))
    current = previous
    following = np.empty((samples, samples))
    spare = np.empty((samples, samples))
    row_residual = np.empty(samples)
    row_undefined = np.zeros(samples, dtype=np.bool_)
    block = FAMILY_TRANSPOSE_BLOCK
    blocks = (samples + block - 1) // block
    for iteration in range(first, stop):
        # incoming = current.T (blocked transpose).
        for tile in prange(blocks * blocks):
            row0 = (tile // blocks) * block
            column0 = (tile % blocks) * block
            for i in range(row0, min(row0 + block, samples)):
                for j in range(column0, min(column0 + block, samples)):
                    incoming[i, j] = current[j, i]
        for lane in prange(lanes):
            _lane_cavity(
                scores,
                rows,
                states,
                log_multiplicity,
                samples,
                lane,
                lanes,
                starts,
                widest,
                incoming,
                current,
                messages,
                row_residual,
                row_undefined,
            )
        residual = 0.0
        if np.any(row_undefined):
            # Fold changes in child/parent slot order, recomputing from the same operands.
            for child in range(samples):
                for parent in range(samples):
                    residual = max(residual, abs(messages[child, parent] - current[child, parent]))
        else:
            for child in range(samples):
                residual = max(residual, row_residual[child])
        if residual <= tolerance:
            return messages, iteration + 1, residual, 1
        for child in prange(samples):
            for parent in range(samples):
                following[child, parent] = (1.0 - damping) * current[child, parent] + damping * messages[child, parent]
        # The damped state becomes current; its old buffer (never the caller's array) is reused next.
        if iteration == first:
            current, following = following, spare
        else:
            current, following = following, current
        if window > 0:
            done = iteration + 1
            position = done % window
            earlier = history[position]
            history[position] = residual
            if done >= 2 * window and done % window == 0:
                # Pause without progress over the last window, or when that window's geometric rate
                # needs more than ``horizon`` iterations in all to reach the tolerance.
                if (
                    residual >= earlier
                    or done + window * math.log(tolerance / residual) / math.log(residual / earlier) > horizon
                ):
                    return current, done, residual, 2
    return current, stop, residual, 0


# Chunk the same solve at a bounded amount of work; the stage supplies thread count explicitly.
_CLAIM_WORK = FAMILY_CLAIM_WORK


def _converged_cavity(
    scores, rows, states, log_multiplicity, samples, maximum, tolerance, damping, start, window, horizon, threads=1
):
    """The same cavity equations iterated from ``start`` to the undamped fixed-point residual.

    Returns (messages, iterations, residual, paused). With window > 0 the solve pauses (paused True, the
    messages being the state to resume from) once two windows have run and the residual's geometric rate over
    the last window cannot reach ``tolerance`` within ``horizon`` iterations; window 0 never pauses.
    Runs in work-sized iteration chunks, carrying the same state and residual history between chunks.
    """
    work = len(rows) + samples * samples
    chunk = max(1, _CLAIM_WORK // max(1, work))
    previous = start.copy()
    residual = np.inf
    history = np.full(max(window, 1), np.inf)
    done = 0
    while done < maximum:
        previous, done, residual, status = _cavity_iterations(
            scores,
            rows,
            states,
            log_multiplicity,
            samples,
            previous,
            history,
            done,
            min(maximum, done + chunk),
            tolerance,
            damping,
            window,
            horizon,
            residual,
            threads,
        )
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


def _pair_orientation_log_odds(adjusted, rows, states, log_multiplicity, samples, applied, threads=1):
    """[c, p]: log odds that p is c's parent rather than c p's, given a link.

    ``adjusted`` already carries the applied reciprocal messages; ``applied[p, c]`` is the one added
    to c's configurations naming p. With cavity inclusion log odds a = O[c, p] and b = O[p, c], the
    pair's joint states p -> c, c -> p and neither weigh e^a : e^b : 1. Infinite where only one
    orientation is scored; NaN where neither is.
    """
    cavity = _family_log_odds(adjusted, rows, states, log_multiplicity, samples, threads) - applied.T
    with np.errstate(invalid="ignore"):
        return cavity - cavity.T


def adjust(
    state_scores: np.ndarray,
    identity_scores: np.ndarray,
    rows: np.ndarray,
    states: np.ndarray,
    full_counts: np.ndarray,
    orientation: Orientation,
    config: DirectionParameters,
    trace: dict | None = None,
    *,
    threads: int = 1,
    diagnostics: list[FamilySolveDiagnostics] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
    log_direction = math.log(2.0) + np.logaddexp(math.log1p(-epsilon) + log_support, math.log(0.5 * epsilon))
    direction = np.zeros(len(rows))
    for slot in (1, 2):
        present = (rows[:, slot] >= 0) & eligible
        direction[present] += log_direction[rows[present, 0], rows[present, slot]]
    state_scores = state_scores + direction
    identity_scores = identity_scores + direction

    # Equal state weights in this likelihood-only message; the configured priors act later.
    log_multiplicity = -np.log(np.maximum(1, full_counts))
    avoidance = _cavity_passes(identity_scores, rows, states, log_multiplicity, samples, config.message_passes, threads)
    preparatory = _incoming_messages(avoidance, rows, eligible)
    paths = _path_correction(
        identity_scores + preparatory, rows, states, log_multiplicity, avoidance, config.ancestry_path_budget, threads
    )
    # The path factor is frozen; the final reciprocal solve replaces the preparatory term.
    state_scores = state_scores + paths
    identity_scores = identity_scores + paths
    zero = np.zeros((samples, samples))
    avoidance, iterations, residual, paused = _converged_cavity(
        identity_scores,
        rows,
        states,
        log_multiplicity,
        samples,
        config.final_max_iterations,
        config.final_tolerance,
        1.0,
        zero,
        config.final_progress_window,
        config.final_pause_budgets * config.final_max_iterations,
        threads,
    )
    initial_iterations = iterations
    resumed = 0
    retry_iterations = 0
    if residual > config.final_tolerance:
        # Loopy messages can oscillate in a resample: restart damped, keeping the undamped residual.
        undamped = avoidance
        avoidance, retry_iterations, residual, _ = _converged_cavity(
            identity_scores,
            rows,
            states,
            log_multiplicity,
            samples,
            config.retry_iterations,
            config.final_tolerance,
            config.retry_damping,
            zero,
            0,
            0,
            threads,
        )
        if residual > config.final_tolerance and paused:
            # The undamped solve paused for lack of progress; give it the rest of its budget.
            avoidance, resumed, residual, _ = _converged_cavity(
                identity_scores,
                rows,
                states,
                log_multiplicity,
                samples,
                config.final_max_iterations - iterations,
                config.final_tolerance,
                1.0,
                undamped,
                0,
                0,
                threads,
            )
            iterations += resumed
    solve = FamilySolveDiagnostics(
        initial_iterations,
        retry_iterations,
        resumed,
        bool(paused),
        float(residual),
        bool(residual <= config.final_tolerance),
    )
    if diagnostics is not None:
        diagnostics.append(solve)
    if residual > config.final_tolerance:
        raise FamilyMessageNonConvergence(
            f"Final reciprocal family messages did not converge in {iterations} initial + "
            f"{retry_iterations} damped iterations (undamped log-message residual {residual:g})",
            solve,
        )
    family = _incoming_messages(avoidance, rows, eligible)
    state_scores = state_scores + family
    identity_scores = identity_scores + family
    if trace is not None:
        trace.update(
            direction=direction,
            ancestry_paths=paths,
            reciprocal_family=family,
            orientation_log_odds=_pair_orientation_log_odds(
                identity_scores, rows, states, log_multiplicity, samples, avoidance, threads
            ),
            final_family_iterations=iterations + retry_iterations,
            final_family_retry_iterations=retry_iterations,
            final_family_residual=residual,
        )
    return state_scores, identity_scores
