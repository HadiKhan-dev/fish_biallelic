"""Tie-aware winners and ranks of composite log evidence.

Two scores closer than the tolerance (the larger of 512 ULP of the largest magnitude, at least 1, and
the absolute floor NOISE_FLOOR = 2^-15 nats) are tied: floating evaluation order must not decide a
parent state, an identity or a rank.
Margins (differences of scores) are compared on the same scale: the magnitude of the scores they
came from, not their own (graph.tie_conflict_children).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from ..maths import ties as tie_rules

CONTRAST_ULP_FACTOR = 512.0
# Absolute floor of the winner tolerance (nats; release 2.1, review R12): 512 ULP of the magnitude (1e-10..1e-8 nats)
# absorbs rounding but not the 1e-9 GL noise motion of composite scores. Placed by the valley census of the decisions
# themselves (best vs runner-up; pedigree.winner, std_n80 full-data fit, 780 decisions; work/IE/probes/std_t1_*,
# work/IE/top2_census.py): their gap moves by <= 6.2e-9 under iid 1e-9 noise (values by <= 1.3e-7), the smallest
# decision gaps lie in [1e-3, 1e-2); 2^-15 = 3.05e-5 sits ~5000x above the motion and ~30x below them. 0.0 = 2.0's
# rule.
NOISE_FLOOR = 2.0 ** -15
# Census context (lab instrumentation, design A-8): decision keys are recorded only while it is set (the full-data
# fit; resampled fits are not recorded), and so are the tie counts of FULL_FIT_COUNTER (a maths.ties.TieCounter the
# decision step installs: winner decisions, and those whose margin is within the tolerance but above 2.0's 512 ULP).
CENSUS_CONTEXT: str | None = None
FULL_FIT_COUNTER = None


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
    tolerance = max(contrast_tolerance(values), NOISE_FLOOR)
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


def winner_tolerance(maximum_absolute_value: float) -> float:
    """Margin at or below which two scores of magnitude up to ``maximum_absolute_value`` are tied."""
    return max(CONTRAST_ULP_FACTOR * np.finfo(np.float64).eps * maximum_absolute_value, NOISE_FLOOR)


def unique_finite_winner(rows: np.ndarray, scores: np.ndarray, key: str | None = None) -> tuple[int | None, float]:
    """The row of ``rows`` with the uniquely largest finite score, and its margin.

    Returns (None, 0.0) without finite scores, (row, inf) for a single finite score and
    (None, margin) when the margin to the runner-up is within the tie tolerance (winner_tolerance).
    ``key``: the decision's census key (recorded while CENSUS_CONTEXT is set and the census is enabled).
    """
    rows = np.asarray(rows, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    best_row, best_value, second_value, finite_count, maximum_absolute_value = _top_two_finite(rows, scores)
    if key is not None and CENSUS_CONTEXT is not None and finite_count >= 2 and tie_rules.census_enabled():
        # The decision compares the best and the runner-up only (their gap is what the tolerance must separate).
        tie_rules.census("pedigree.winner", f"{CENSUS_CONTEXT}/{key}", [best_value, second_value])
    if finite_count == 0:
        return None, 0.0
    if finite_count == 1:
        return int(best_row), np.inf
    margin = float(best_value - second_value)
    tolerance = winner_tolerance(maximum_absolute_value)
    if key is not None and CENSUS_CONTEXT is not None and FULL_FIT_COUNTER is not None:
        FULL_FIT_COUNTER.note("pedigree.winner", inside_tie=margin <= tolerance)
        if margin <= tolerance and margin > CONTRAST_ULP_FACTOR * np.finfo(np.float64).eps * maximum_absolute_value:
            FULL_FIT_COUNTER.note("pedigree.winner.noise_floor", inside_tie=True)
    if margin <= tolerance:
        return None, margin
    return int(best_row), margin
