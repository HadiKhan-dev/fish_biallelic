"""Tie-aware winners and ranks of composite log evidence.

Two scores closer than a scale-relative tolerance (512 ULP of the largest magnitude, at least 1)
are tied: floating evaluation order must not decide a parent state, an identity or a rank.
Margins (differences of scores) are compared on the same scale: the magnitude of the scores they
came from, not their own (graph.tie_conflict_children).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

CONTRAST_ULP_FACTOR = 512.0


def contrast_tolerance(values: np.ndarray) -> float:
    """Tolerance below which differences among finite ``values`` count as ties."""
    scale = max(1.0, float(np.max(np.abs(np.asarray(values, dtype=np.float64)))))
    return CONTRAST_ULP_FACTOR * np.finfo(np.float64).eps * scale


def magnitude(values: np.ndarray) -> float:
    """Largest finite |value|, at least 1 (1 without finite values): the scale of the rounding that a
    difference of ``values`` carries."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.abs(values[np.isfinite(values)])
    return max(1.0, float(np.max(finite))) if finite.size else 1.0


def tied_rank_probabilities(scores: np.ndarray) -> np.ndarray:
    """Rank evidence (n - average rank) / n of finite scores, ties sharing their average rank,
    normalised to sum to one."""
    values = np.asarray(scores, dtype=np.float64)
    n_values = len(values)
    if n_values == 1:
        return np.ones(1, dtype=np.float64)
    tolerance = contrast_tolerance(values)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(n_values, dtype=np.float64)
    start = 0
    while start < n_values:
        end = start + 1
        while end < n_values and abs(values[order[end]] - values[order[start]]) <= tolerance:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    evidence = (n_values - ranks) / n_values
    evidence /= np.sum(evidence)
    return evidence


@njit(cache=True)
def _top_two_finite(rows, scores):
    """Stable O(n) scan: first best row, best and runner-up value, finite count, max |value| (>= 1)."""
    best_row = -1
    best_value = -math.inf
    second_value = -math.inf
    finite_count = 0
    maximum_absolute_value = 1.0
    for position in range(len(rows)):
        row = int(rows[position])
        value = float(scores[row])
        if not math.isfinite(value):
            continue
        finite_count += 1
        absolute_value = abs(value)
        if absolute_value > maximum_absolute_value:
            maximum_absolute_value = absolute_value
        if best_row < 0 or value > best_value:
            if best_row >= 0:
                second_value = best_value
            best_row = row
            best_value = value
        elif value > second_value:
            # Strict comparison keeps the first occurrence as the winner while an equal later
            # value still becomes the runner-up.
            second_value = value
    return best_row, best_value, second_value, finite_count, maximum_absolute_value


def unique_finite_winner(rows: np.ndarray, scores: np.ndarray) -> tuple[int | None, float]:
    """The row of ``rows`` with the uniquely largest finite score, and its margin.

    Returns (None, 0.0) without finite scores, (row, inf) for a single finite score and
    (None, margin) when the margin to the runner-up is within the contrast tolerance.
    """
    best_row, best_value, second_value, finite_count, maximum_absolute_value = _top_two_finite(
        np.asarray(rows, dtype=np.int64), np.asarray(scores, dtype=np.float64))
    if finite_count == 0:
        return None, 0.0
    if finite_count == 1:
        return int(best_row), np.inf
    margin = float(best_value - second_value)
    if margin <= CONTRAST_ULP_FACTOR * np.finfo(np.float64).eps * maximum_absolute_value:
        return None, margin
    return int(best_row), margin
