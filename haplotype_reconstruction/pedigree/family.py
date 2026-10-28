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
reciprocal solve replaces the preparatory reciprocal term. Design chronology exempts an edge's
side from all three terms (not an uncertain co-parent).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from ..maths.logspace import logaddexp
from .config import DirectionConfig
from .orientation import Orientation


class FamilyMessageNonConvergence(FloatingPointError):
    """The final reciprocal solve missed its tolerance: an error in the full-data fit, an abstaining
    replicate in resampling."""


@njit(cache=True, nogil=True)
def _family_log_odds(log_scores, rows, states, log_multiplicity, samples):
    """Included/excluded log odds of every [child, parent] in O(A + N^2), stably.

    Included masses have an independent scale per parent, so a tiny included probability survives
    the removal of a strong incoming reverse message; exclusions containing the maximum row share
    its scale and at most two exclusions per child need a separate scan.
    """
    output = np.full((samples, samples), -np.inf)
    starts = np.searchsorted(rows[:, 0], np.arange(samples + 1))
    for child in range(samples):
        start, stop = starts[child], starts[child + 1]
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
            continue
        mass = np.zeros(samples)
        included_sum = np.zeros(samples)
        total = 0.0
        for row in range(start, stop):
            value = log_scores[row] + log_multiplicity[child, states[row]]
            scaled = math.exp(value - maximum)
            total += scaled
            for slot in range(1, 3):
                parent = rows[row, slot]
                if parent >= 0 and np.isfinite(included_max[parent]):
                    mass[parent] += scaled
                    included_sum[parent] += math.exp(value - included_max[parent])
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
                        excluded_sum += math.exp(
                            log_scores[row] + log_multiplicity[child, states[row]] - excluded_max)
                log_excluded = excluded_max + math.log(excluded_sum)
            else:
                # Exact: the best row excludes this parent, so the excluded mass is >= 1.
                log_excluded = maximum + math.log(max(1.0, total - mass[parent]))
            output[child, parent] = included_max[parent] + math.log(included_sum[parent]) - log_excluded
    return output


@njit(cache=True, nogil=True)
def _cavity_passes(log_scores, rows, states, log_multiplicity, samples, passes):
    """Bounded synchronous cavity-message passes (immediate reverse feedback excluded)."""
    previous = np.zeros((samples, samples))
    for _ in range(passes):
        adjusted = log_scores.copy()
        for row in range(len(rows)):
            child = rows[row, 0]
            for slot in range(1, 3):
                parent = rows[row, slot]
                if parent >= 0:
                    adjusted[row] += previous[parent, child]
        odds = _family_log_odds(adjusted, rows, states, log_multiplicity, samples)
        for child in range(samples):
            for parent in range(samples):
                # Remove incoming evidence before converting odds to support.
                odds[child, parent] -= previous[parent, child]
        previous = -np.maximum(0.0, odds) - np.log1p(np.exp(-np.abs(odds)))
    return previous


@njit(cache=True, nogil=True)
def _converged_cavity(scores, rows, states, log_multiplicity, samples, maximum, tolerance,
                      damping, start, window, horizon):
    """The same cavity equations iterated from ``start`` to the undamped fixed-point residual.

    Returns (messages, iterations, residual, paused). With window > 0 the solve pauses (paused True, the
    messages being the state to resume from) once two windows have run and the residual's geometric rate over
    the last window cannot reach ``tolerance`` within ``horizon`` iterations; window 0 never pauses.
    """
    previous = start.copy()
    residual = np.inf
    history = np.full(max(window, 1), np.inf)
    for iteration in range(maximum):
        adjusted = scores.copy()
        for row in range(len(rows)):
            child = rows[row, 0]
            for slot in (1, 2):
                parent = rows[row, slot]
                if parent >= 0:
                    adjusted[row] += previous[parent, child]
        odds = _family_log_odds(adjusted, rows, states, log_multiplicity, samples)
        residual = 0.0
        for child in range(samples):
            for parent in range(samples):
                value = odds[child, parent] - previous[parent, child]
                value = -max(0.0, value) - np.log1p(np.exp(-abs(value)))
                residual = max(residual, abs(value - previous[child, parent]))
                odds[child, parent] = value
        if residual <= tolerance:
            return odds, iteration + 1, residual, False
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
                    return previous, done, residual, True
    return previous, maximum, residual, False


def _incoming_messages(avoidance, rows, eligible, chronology):
    """Per row, the sum of the incoming reciprocal messages of its parents (chronology exempt)."""
    family = np.zeros(len(rows))
    for slot in (1, 2):
        present = (rows[:, slot] >= 0) & eligible
        child, parent = rows[present, 0], rows[present, slot]
        family[present] += np.where(chronology[child, parent], 0.0, avoidance[parent, child])
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
def _configuration_panels(scores, rows, states, log_multiplicity, samples, budget):
    """Top ``budget`` configurations per sample; boundary ties move to the neutral tail."""
    starts = np.searchsorted(rows[:, 0], np.arange(samples + 1))
    selected = np.full((samples, budget), -1, dtype=np.int64)
    weights = np.full((samples, budget), -np.inf)
    tail = np.full(samples, -np.inf)
    for parent in range(samples):
        for row in range(starts[parent], starts[parent + 1]):
            value = scores[row] + log_multiplicity[parent, states[row]]
            if not np.isfinite(value):
                continue
            minimum = np.argmin(weights[parent])
            if value > weights[parent, minimum]:
                tail[parent] = logaddexp(tail[parent], weights[parent, minimum])
                weights[parent, minimum], selected[parent, minimum] = value, row
            else:
                tail[parent] = logaddexp(tail[parent], value)
        boundary = np.min(weights[parent])
        if np.isfinite(boundary):
            found = 0
            for row in range(starts[parent], starts[parent + 1]):
                if scores[row] + log_multiplicity[parent, states[row]] >= boundary:
                    found += 1
            if found > budget:
                for j in range(budget):
                    if weights[parent, j] == boundary:
                        tail[parent] = logaddexp(tail[parent], weights[parent, j])
                        weights[parent, j], selected[parent, j] = -np.inf, -1
        # Keep even tiny omitted mass in log space until all exclusions are applied.
        center = max(tail[parent], np.max(weights[parent]))
        if np.isfinite(center):
            tail[parent] -= center
            weights[parent] -= center
    return selected, weights, tail


@njit(cache=True, nogil=True)
def _path_correction(scores, rows, states, log_multiplicity, excluded, budget, chronology):
    """Reverse two-edge path factor per row; O(R A + R N^2 + R^2 A2), R = budget.

    ``excluded[g, c]`` is the outgoing cavity log probability that g does not choose c as parent.
    Chronology exempts that focal parental side, not an uncertain co-parent.
    """
    n = len(excluded)
    selected, weights, tail = _configuration_panels(scores, rows, states, log_multiplicity, n,
                                                    budget)
    retained = np.full((n, n), -np.inf)
    denominator = np.full((n, n), -np.inf)
    single = np.zeros((n, n))
    for parent in range(n):
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
                retained[parent, child] = logaddexp(retained[parent, child], value)
                if a >= 0:
                    value += excluded[a, child]
                if b >= 0:
                    value += excluded[b, child]
                numerator = logaddexp(numerator, value)
            denominator[parent, child] = logaddexp(tail[parent], retained[parent, child])
            if np.isfinite(denominator[parent, child]):
                single[parent, child] = min(0., numerator - denominator[parent, child])

    result = np.zeros(len(states))
    for target in range(len(states)):
        child, p, q = rows[target]
        if p < 0 or not np.isfinite(scores[target]):
            continue
        supported_p = chronology[child, p]
        if q < 0:
            if not supported_p:
                result[target] = single[p, child]
            continue
        supported_q = chronology[child, q]
        if supported_p and supported_q:
            continue
        if supported_p or supported_q:
            # Only the uncertain side contributes paths; both focal parents stay excluded (a path
            # through the co-parent is already forbidden by direct reciprocity).
            parent = q if supported_p else p
            if not np.isfinite(denominator[parent, child]):
                continue
            numerator = tail[parent]
            for j in range(budget):
                row = selected[parent, j]
                if row < 0:
                    continue
                a, b = rows[row, 1], rows[row, 2]
                if a == child or b == child:
                    continue
                value = weights[parent, j]
                if a >= 0 and a != p and a != q:
                    value += excluded[a, child]
                if b >= 0 and b != p and b != q:
                    value += excluded[b, child]
                numerator = logaddexp(numerator, value)
            result[target] = min(0., numerator - denominator[parent, child])
            continue
        if not np.isfinite(denominator[p, child] + denominator[q, child]):
            continue
        # Tail configurations are fully compatible, including omitted direct reversals; this can
        # only dilute the conditional penalty.
        numerator = logaddexp(tail[p] + denominator[q, child], retained[p, child] + tail[q])
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
                    value += excluded[g, child]
                numerator = logaddexp(numerator, value)
        result[target] = min(0., numerator - denominator[p, child] - denominator[q, child])
    return result


def adjust(state_scores: np.ndarray, identity_scores: np.ndarray, rows: np.ndarray,
           states: np.ndarray, full_counts: np.ndarray, orientation: Orientation,
           config: DirectionConfig, chronology: np.ndarray, trace: dict | None = None,
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
    log_direction = math.log(2.0) + np.logaddexp(math.log1p(-epsilon) + log_support,
                                                 math.log(0.5 * epsilon))
    log_direction = np.where(chronology, 0.0, log_direction)
    direction = np.zeros(len(rows))
    for slot in (1, 2):
        present = (rows[:, slot] >= 0) & eligible
        direction[present] += log_direction[rows[present, 0], rows[present, slot]]
    state_scores = state_scores + direction
    identity_scores = identity_scores + direction

    # Equal state weights in this likelihood-only message; the configured priors act later.
    log_multiplicity = -np.log(np.maximum(1, full_counts))
    avoidance = _cavity_passes(identity_scores, rows, states, log_multiplicity, samples,
                               config.message_passes)
    preparatory = _incoming_messages(avoidance, rows, eligible, chronology)
    paths = _path_correction(identity_scores + preparatory, rows, states, log_multiplicity,
                             avoidance, config.ancestry_path_budget, chronology)
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
    family = _incoming_messages(avoidance, rows, eligible, chronology)
    state_scores = state_scores + family
    identity_scores = identity_scores + family
    if trace is not None:
        applied = np.where(chronology.T, 0.0, avoidance)
        trace.update(
            direction=direction, ancestry_paths=paths, reciprocal_family=family,
            orientation_log_odds=_pair_orientation_log_odds(
                identity_scores, rows, states, log_multiplicity, samples, applied),
            final_family_iterations=iterations + retry_iterations,
            final_family_retry_iterations=retry_iterations, final_family_residual=residual)
    return state_scores, identity_scores
