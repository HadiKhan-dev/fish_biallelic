"""Correct family scores for retained indirect paths."""

from __future__ import annotations

import numpy as np
from numba import prange

from ...core.kernels import kernel
from ...core.numerics import log_add as _logaddexp
from .family_numerics import _exp_or_zero, _lane_count, _log_or_zero
from .kernels.decision import OVERLOADS


@kernel(overloads=OVERLOADS["family_paths._sample_panel"], cache=True, nogil=True)
def _sample_panel(scores, rows, states, log_multiplicity, parent, start, stop, budget, selected, weights, tail):
    """One sample's bounded panel of finite configurations, retaining the boundary tie class in the tail."""
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


@kernel(overloads=OVERLOADS["family_paths._configuration_panels"], cache=True, nogil=True, parallel=True)
def _configuration_panels(scores, rows, states, log_multiplicity, samples, budget, threads):
    """Top ``budget`` configurations per sample; boundary ties move to the neutral tail."""
    starts = np.searchsorted(rows[:, 0], np.arange(samples + 1))
    selected = np.full((samples, budget), -1, dtype=np.int64)
    weights = np.full((samples, budget), -np.inf)
    tail = np.full(samples, -np.inf)
    lanes = _lane_count(samples, threads)
    for lane in prange(lanes):
        for parent in range(lane, samples, lanes):
            _sample_panel(
                scores,
                rows,
                states,
                log_multiplicity,
                parent,
                starts[parent],
                starts[parent + 1],
                budget,
                selected,
                weights,
                tail,
            )
    return selected, weights, tail


@kernel(overloads=OVERLOADS["family_paths._path_singles"], cache=True, nogil=True)
def _path_singles(parent, n, rows, selected, weights, tail, excluded, budget, retained, denominator, single):
    """One parent's row of the single-parent ancestry-avoidance factor."""
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
            single[parent, child] = min(0.0, numerator - denominator[parent, child])


@kernel(overloads=OVERLOADS["family_paths._path_target"], cache=True, nogil=True)
def _path_target(
    target, rows, scores, selected, weights, tail, retained, denominator, single, excluded_by_child, budget
):
    """One configuration row's path factor; excluded_by_child[c, g] = excluded[g, c] lays out exclusions by child."""
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
    # Stable scaled sum of the same terms: one exponential per panel pair and one final logarithm,
    # rather than a logarithm at every addition. Rescale whenever a larger term is encountered.
    maximum, total = numerator, 1.0
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
            if value > maximum:
                total = total * _exp_or_zero(maximum - value) + 1.0
                maximum = value
            elif value != -np.inf:
                total += _exp_or_zero(value - maximum)
    numerator = maximum + _log_or_zero(total)
    return min(0.0, numerator - denominator[p, child] - denominator[q, child])


@kernel(overloads=OVERLOADS["family_paths._path_correction"], cache=True, nogil=True, parallel=True)
def _path_correction(scores, rows, states, log_multiplicity, excluded, budget, threads):
    """Reverse two-edge path factor per row; O(R A + R N^2 + R^2 A2), R = budget.

    ``excluded[g, c]`` is the outgoing cavity log probability that g does not choose c as parent.
    """
    n = len(excluded)
    selected, weights, tail = _configuration_panels(scores, rows, states, log_multiplicity, n, budget, threads)
    retained = np.full((n, n), -np.inf)
    denominator = np.full((n, n), -np.inf)
    single = np.zeros((n, n))
    lanes = _lane_count(n, threads)
    for lane in prange(lanes):
        for parent in range(lane, n, lanes):
            _path_singles(parent, n, rows, selected, weights, tail, excluded, budget, retained, denominator, single)
    excluded_by_child = np.ascontiguousarray(excluded.T)
    result = np.zeros(len(states))
    count = len(states)
    lanes = _lane_count(count, threads)
    for lane in prange(lanes):
        for target in range(lane, count, lanes):
            result[target] = _path_target(
                target, rows, scores, selected, weights, tail, retained, denominator, single, excluded_by_child, budget
            )
    return result
