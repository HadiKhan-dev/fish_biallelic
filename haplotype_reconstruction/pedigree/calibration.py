"""Cross-chromosome predictive weighting of composite pedigree evidence.

One nuisance scale multiplies every M0/M1/M2 log likelihood, fitted conditional on the existing
paintings and candidate panel by predicting held-out chromosomes. It is not calibration of
biological posterior probabilities: painting, screening, direction and relatives are not cross-fit.
Priors, candidate universe, direction model and release rules stay fixed; the full-marker exclusion
is never tempered. Whole chromosomes define folds (index mod 4); folds 0-2 are each predicted by the
other two, fold 3 is a held-out test that never selects the scale. Only tempering (scale < 1) with a
positive, interior predictive gain is applied; resampling conditions on the fitted scale.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from ..maths import ties
from .alternatives import Alternatives
from .config import PedigreeConfig
from .records import GenomeScores

# Broad numerical bounds, not fitted constants: a boundary optimum is unidentified and keeps
# scale 1. The 33-point grid only brackets optima (the objective need not be unimodal).
LOG_SCALE_BOUNDS = (float(np.log(0.001)), float(np.log(100.0)))
GRID_POINTS = 33
OPTIMIZER_XATOL = 1e-5
MINIMUM_GAIN = 1e-8          # nats of pooled predictive log likelihood
BOUNDARY_MARGIN = 0.01       # log-scale distance to a search bound treated as the boundary
# Release 2.1 (review R12): Brent's bounded search stops anywhere within OPTIMIZER_XATOL of an interior optimum, and
# where depends on its path, which rounding and 1e-9 GL noise alter; the scale multiplies every log likelihood, so
# a 1e-5 shift moves every margin by 1e-5 relative. Each interior maximum is therefore refined to the root of the
# objective's analytic derivative (safeguarded Newton, bracketed), to rounding precision; a step below this
# (relative to max(1, |x|)) ends the refinement.
REFINE_TOLERANCE = 1e-14
REFINE_ITERATIONS = 60


def fit_scale(scores: GenomeScores, alternatives: Alternatives,
              config: PedigreeConfig) -> tuple[float, dict]:
    """(scale to apply, JSON-safe report). Scale 1 when calibration is off or not selected."""
    report = dict(
        mode=config.evidence_calibration, scale=1.0, selected=False, status="disabled",
        truth_used=False, parent_count_prior_fitted=False,
        conditional_on_fixed_paintings_and_candidate_panel=True,
        resampling_conditions_on_fitted_scale=True,
        interpretation="composite predictive evidence weight, not posterior calibration")
    if config.evidence_calibration == "off":
        return 1.0, report
    started = time.perf_counter()
    report.update(_fit(scores, alternatives, config))
    report["seconds"] = time.perf_counter() - started
    return float(report["scale"]), report


def _refine(derivatives, start, low, high):
    """The stationary point of a smooth maximum near ``start`` inside [low, high]: the root of the first
    derivative (``derivatives(x) -> (f', f'')``), bracketed around ``start`` and found by Newton steps safeguarded
    by bisection. Returns ``start`` when no sign change of f' (+ to -) brackets a maximum."""
    width = 4.0 * OPTIMIZER_XATOL
    left = right = None
    for _ in range(30):
        a, b = max(low, start - width), min(high, start + width)
        fa, fb = derivatives(a)[0], derivatives(b)[0]
        if fa >= 0.0 >= fb:
            left, right = (a, fa), (b, fb)
            break
        if a == low and b == high:
            break
        width *= 4.0
    if left is None:
        return float(start)
    (a, fa), (b, fb) = left, right
    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)
    x = min(max(start, a), b)
    for _ in range(REFINE_ITERATIONS):
        first, second = derivatives(x)
        if first == 0.0:
            return float(x)
        if first > 0.0:
            a = x
        else:
            b = x
        step = -first / second if second < 0.0 else None
        candidate = x + step if step is not None else None
        if candidate is None or not (a < candidate < b):
            candidate = 0.5 * (a + b)
        if abs(candidate - x) <= REFINE_TOLERANCE * max(1.0, abs(x)) or b - a <= REFINE_TOLERANCE * max(1.0, abs(x)):
            return float(candidate)
        x = candidate
    return float(x)


def _maximize(prediction, derivatives=None):
    """Bracket the visible maxima of a scalar objective on the grid, then refine each (Brent, then the root of the
    derivative when ``derivatives`` is given). Maxima whose values lie in one tie class (nll quantum) are tied: the
    one closest to scale 1 (smallest |log scale|) is taken."""
    grid = np.linspace(*LOG_SCALE_BOUNDS, GRID_POINTS)
    values = np.asarray([prediction(x) for x in grid])
    candidates = [(float(values[i]), float(grid[i])) for i in (0, len(grid) - 1)]
    converged = True
    for i in range(1, len(grid) - 1):
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            if values[i] == values[i - 1] == values[i + 1]:
                continue
            found = minimize_scalar(lambda x: -prediction(x), bounds=(grid[i - 1], grid[i + 1]),
                                    method="bounded", options={"xatol": OPTIMIZER_XATOL})
            converged &= bool(found.success)
            if found.success:
                x = float(found.x)
                if derivatives is not None:
                    x = _refine(derivatives, x, float(grid[i - 1]), float(grid[i + 1]))
                candidates.append((prediction(x), x))
    values_only = [value for value, _ in candidates]
    finite = [value for value in values_only if np.isfinite(value)]
    if finite:
        quantum = ties.tie_quantum(max(abs(value) for value in finite), "nll")
        ranked = ties.tie_classes(values_only, quantum, descending=True)
        tied = [candidates[index] for index in ranked.members(0)]
        # Prior: the scale nearest 1 (smallest |log scale|); then the smaller scale by convention (the larger under
        # the convention reversal, release 2.2).
        best, log_scale = min(tied, key=lambda item: (abs(item[1]), ties.conv_key(item[1])))
    else:
        best, log_scale = max(candidates, key=lambda item: (item[0], ties.conv_key(item[1])))
    return best, log_scale, converged, grid, values


def _reciprocal_dominant_mask(rows: np.ndarray, zero: np.ndarray, one: np.ndarray,
                              contigs: np.ndarray) -> np.ndarray:
    """0 / -inf objective weights excluding reciprocal-dominant rows.

    A row naming p as a parent of c is excluded when, on ``contigs``, M1(p with parent c) - M0(p) >=
    M1(c with parent p) - M0(c): genetics cannot order such reciprocal hypotheses (direction and
    family messages settle them). Excluded from the scale's objective only, never from decisions.
    """
    with np.errstate(invalid="ignore"):
        # Ineligible children have -inf M0 and M1; their NaN gains never win.
        gain = one[contigs].sum(axis=0) - zero[contigs].sum(axis=0)[:, None]
    mask = np.zeros(len(rows), dtype=np.float64)
    child = rows[:, 0]
    for column in (1, 2):
        present = np.flatnonzero(rows[:, column] >= 0)
        parent = rows[present, column]
        dominant = gain[parent, child[present]] >= gain[child[present], parent]
        mask[present[dominant]] = -np.inf
    return mask


def _fit(scores: GenomeScores, alternatives: Alternatives, config: PedigreeConfig) -> dict:
    folds = np.arange(len(scores.contigs)) % 4
    groups = [np.flatnonzero(folds == i) for i in range(4)]
    report = dict(
        scale=1.0, selected=False, status="insufficient_contigs",
        estimator="pooled-three-fold-predictive-v2-reciprocal-excluded",
        partitions={f"fold_{i}": [str(scores.contigs[j]) for j in group]
                    for i, group in enumerate(groups)},
        testing_fold="fold_3",
        training_scheme="each non-test fold predicted by the other two; test uses all non-test contigs",
        search_scale_bounds=list(np.exp(LOG_SCALE_BOUNDS)))
    if min(map(len, groups)) < 2:
        return report
    rows, states = alternatives.rows, alternatives.states
    likelihoods = alternatives.log_likelihoods
    held = np.stack([likelihoods[folds == i].sum(axis=0) for i in range(3)])
    training = np.stack([likelihoods[(folds < 3) & (folds != i)].sum(axis=0) for i in range(3)])
    testing = likelihoods[folds == 3].sum(axis=0)
    all_training = likelihoods[folds < 3].sum(axis=0)
    prior = np.log(np.asarray(config.selection.state_priors)[states])
    prior -= np.log(np.maximum(1, alternatives.full_counts[rows[:, 0], states]))
    fit_mask = np.stack([_reciprocal_dominant_mask(rows, scores.zero, scores.one,
                                                   (folds < 3) & (folds != i)) for i in range(3)])
    test_mask = _reciprocal_dominant_mask(rows, scores.zero, scores.one, folds < 3)
    prepared, omitted = [], 0
    for child_rows in alternatives.child_rows:
        finite = np.isfinite(likelihoods[:, child_rows]).all(axis=0)
        omitted += int(np.count_nonzero(~finite))
        valid = child_rows[finite]
        if len(valid) < 2:
            continue
        # A finite M0 row is never masked, so every fold keeps a maximum.
        fit_kept = np.isfinite(fit_mask[:, valid])
        test_kept = np.isfinite(test_mask[valid])
        if not (fit_kept.any(axis=1).all() and test_kept.any()):
            continue
        train, validation = training[:, valid], held[:, valid]
        train = train - train.max(axis=1, keepdims=True)
        validation = validation - np.max(np.where(fit_kept, validation, -np.inf), axis=1,
                                         keepdims=True)
        combined = all_training[valid] - all_training[valid].max()
        test = testing[valid] - testing[valid][test_kept].max()
        prepared.append((train, validation, combined, test,
                         prior[valid][None, :] + fit_mask[:, valid], prior[valid] + test_mask[valid]))
    report.update(eligible_children=len(prepared), omitted_nonfinite_alternatives=omitted,
                  reciprocal_dominant_rows_excluded_per_fold=[
                      int(np.count_nonzero(~np.isfinite(mask))) for mask in fit_mask])
    if not prepared:
        report["status"] = "insufficient_alternatives"
        return report

    def prediction(log_scale, test=False):
        scale, total = np.exp(log_scale), 0.0
        for train, validation, combined, held_test, p, p_test in prepared:
            if test:
                weights = scale * combined + p_test
                total += logsumexp(weights + held_test) - logsumexp(weights)
            else:
                weights = scale * train + p
                total += np.sum(logsumexp(weights + validation, axis=1)
                                - logsumexp(weights, axis=1))
        return float(total)

    def derivatives(log_scale):
        """(d/dx, d2/dx2) of prediction(x), x = log scale: s (E'[t] - E[t]) and s (E'[t] - E[t]) + s^2 (Var'[t] -
        Var[t]) summed over children and folds, with E/Var under softmax(s t + p) and E'/Var' under
        softmax(s t + p + validation) (t = train)."""
        scale = float(np.exp(log_scale))
        mean_difference = variance_difference = 0.0
        for train, validation, _, _, p, _ in prepared:
            base = scale * train + p
            for logits, sign in ((base + validation, 1.0), (base, -1.0)):
                top = np.max(logits, axis=1, keepdims=True)
                weights = np.exp(logits - top)
                weights /= np.sum(weights, axis=1, keepdims=True)
                mean = np.sum(weights * train, axis=1)
                variance = np.sum(weights * (train - mean[:, None]) ** 2, axis=1)
                mean_difference += sign * float(np.sum(mean))
                variance_difference += sign * float(np.sum(variance))
        return scale * mean_difference, scale * mean_difference + scale * scale * variance_difference

    best, log_scale, success, grid, values = _maximize(prediction, derivatives)
    gain = best - prediction(0.0)
    boundary = min(log_scale - LOG_SCALE_BOUNDS[0], LOG_SCALE_BOUNDS[1] - log_scale) < BOUNDARY_MARGIN
    # Tempering only: a scale above one would sharpen evidence without calibration support.
    above_unit = log_scale >= 0.0
    selected = bool(success and np.isfinite(gain) and gain > MINIMUM_GAIN and not boundary
                    and not above_unit)
    applied_log_scale = log_scale if selected else 0.0
    report.update(
        scale=float(np.exp(log_scale)) if selected else 1.0,
        fitted_scale=float(np.exp(log_scale)), selected=selected,
        # Release 2.1: without predictive gain the optimum is unidentified and its location (boundary or not) is
        # rounding- and noise-driven on the flat objective, so the gain test comes before the boundary label.
        status=("selected" if selected else "optimizer_failed" if not success else
                "no_predictive_gain" if not (np.isfinite(gain) and gain > MINIMUM_GAIN) else
                "boundary_optimum" if boundary else "above_unit_scale"),
        optimizer_success=success, at_search_boundary=bool(boundary),
        selection_gain=float(gain), search_log_scale_grid=grid.tolist(),
        search_predictive_grid=values.tolist(),
        # Neither test diagnostic selects; they separate the unapplied candidate from the fallback.
        untouched_test_gain=prediction(applied_log_scale, True) - prediction(0.0, True),
        fitted_untouched_test_gain=prediction(log_scale, True) - prediction(0.0, True))
    return report
