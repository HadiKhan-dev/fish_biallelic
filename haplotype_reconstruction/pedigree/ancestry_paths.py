"""Finite joint-family evidence against reverse two-edge ancestry paths.

Average over uncertain parent configurations, conditioning out direct reverse
edges already handled by reciprocal messages. Shared intermediates count once;
the other focal parent is excluded, preserving legitimate backcrosses. This is
a one-way composite cavity approximation, not a global pedigree posterior.
Omitted bounded-panel mass is neutral, so truncation only weakens the penalty.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _logadd(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


@njit(cache=True, nogil=True)
def _configuration_panels(scores, alternatives, states, multiplicity, n, budget):
    """Top joint configurations, with boundary ties moved to the neutral tail."""
    starts = np.searchsorted(alternatives[:, 0], np.arange(n + 1))
    selected = np.full((n, budget), -1, dtype=np.int64)
    weights = np.full((n, budget), -np.inf)
    tail = np.full(n, -np.inf)
    for parent in range(n):
        for row in range(starts[parent], starts[parent + 1]):
            value = scores[row] + multiplicity[parent, states[row]]
            if not np.isfinite(value):
                continue
            minimum = np.argmin(weights[parent])
            if value > weights[parent, minimum]:
                tail[parent] = _logadd(tail[parent], weights[parent, minimum])
                weights[parent, minimum], selected[parent, minimum] = value, row
            else:
                tail[parent] = _logadd(tail[parent], value)
        boundary = np.min(weights[parent])
        if np.isfinite(boundary):
            found = 0
            for row in range(starts[parent], starts[parent + 1]):
                if scores[row] + multiplicity[parent, states[row]] >= boundary:
                    found += 1
            if found > budget:
                for j in range(budget):
                    if weights[parent, j] == boundary:
                        tail[parent] = _logadd(tail[parent], weights[parent, j])
                        weights[parent, j], selected[parent, j] = -np.inf, -1
        # Avoid cancellation between chromosome-wide likelihood totals. Keep
        # even tiny omitted mass in log space until all exclusions are applied.
        center = max(tail[parent], np.max(weights[parent]))
        if np.isfinite(center):
            tail[parent] -= center
            weights[parent] -= center
    return selected, weights, tail


@njit(cache=True, nogil=True)
def joint_path_correction(scores, alternatives, states, multiplicity, excluded,
                          budget, explicit_direction):
    """O(R*A + R*N**2 + R**2*A2), with R=budget and A2=M2 row count.

``excluded[g,c]`` is the outgoing cavity log probability that g does not
choose c as parent. Explicit direction support exempts that focal parental
side, not an uncertain co-parent. It does not assert individual parentage.
The caller only invokes this kernel with a positive budget.
    """
    n = len(excluded)
    selected, weights, tail = _configuration_panels(
        scores, alternatives, states, multiplicity, n, budget)
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
                a, b = alternatives[row, 1], alternatives[row, 2]
                if a == child or b == child:
                    continue
                value = weights[parent, j]
                retained[parent, child] = _logadd(retained[parent, child], value)
                if a >= 0:
                    value += excluded[a, child]
                if b >= 0:
                    value += excluded[b, child]
                numerator = _logadd(numerator, value)
            denominator[parent, child] = _logadd(tail[parent], retained[parent, child])
            if np.isfinite(denominator[parent, child]):
                single[parent, child] = min(0., numerator - denominator[parent, child])

    result = np.zeros(len(states))
    for target in range(len(states)):
        child, p, q = alternatives[target]
        if p < 0 or not np.isfinite(scores[target]):
            continue
        supported_p = explicit_direction[child, p]
        if q < 0:
            if not supported_p:
                result[target] = single[p, child]
            continue
        supported_q = explicit_direction[child, q]
        if supported_p and supported_q:
            continue
        if supported_p or supported_q:
            # Only the uncertain side contributes paths. Still exclude BOTH
            # focal parents: a path through its co-parent is already forbidden
            # by direct reciprocity, irrespective of chronology metadata.
            parent = q if supported_p else p
            if not np.isfinite(denominator[parent, child]):
                continue
            numerator = tail[parent]
            for j in range(budget):
                row = selected[parent, j]
                if row < 0:
                    continue
                a, b = alternatives[row, 1], alternatives[row, 2]
                if a == child or b == child:
                    continue
                value = weights[parent, j]
                if a >= 0 and a != p and a != q:
                    value += excluded[a, child]
                if b >= 0 and b != p and b != q:
                    value += excluded[b, child]
                numerator = _logadd(numerator, value)
            result[target] = min(0., numerator - denominator[parent, child])
            continue
        if not np.isfinite(denominator[p, child] + denominator[q, child]):
            continue
        # Tail configurations are fully compatible, including any omitted
        # direct reversals. This can only dilute the conditional penalty.
        numerator = _logadd(tail[p] + denominator[q, child], retained[p, child] + tail[q])
        for i in range(budget):
            ri = selected[p, i]
            if ri < 0:
                continue
            a, b = alternatives[ri, 1], alternatives[ri, 2]
            if a == child or b == child:
                continue
            for j in range(budget):
                rj = selected[q, j]
                if rj < 0:
                    continue
                d, e = alternatives[rj, 1], alternatives[rj, 2]
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
                numerator = _logadd(numerator, value)
        result[target] = min(0., numerator - denominator[p, child] - denominator[q, child])
    return result
