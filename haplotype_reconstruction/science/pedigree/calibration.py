"""Cross-chromosome predictive weighting of composite pedigree evidence.

One nuisance scale multiplies every M0/M1/M2 log likelihood, fitted conditional on the existing
paintings and candidate panel by predicting held-out chromosomes. It is not calibration of
biological posterior probabilities: painting, screening, direction and relatives are not cross-fit.
Priors, candidate universe, direction model and release rules stay fixed; the full-marker exclusion
is never tempered. Whole chromosomes define folds (index mod 4); folds 0-2 are each predicted by the
other two, fold 3 is a held-out test that never selects the scale. Only tempering (scale < 1) with a
positive, interior predictive gain is applied; resampling conditions on the fitted scale.

PD03 objective: F(x) = sum_c sum_(f=0..2) [logsumexp(exp(x)*train + prior + mask + held)
- logsumexp(exp(x)*train + prior + mask)]. Reciprocal masks exclude reverse gain >=
forward gain, including ties. Center train over all valid rows, held over mask-kept
rows, combined training over all valid rows, and test over test-kept rows, before
M1 aggregation. This centering is objective identity, not a normalization. The
33-point log-scale grid, every Brent success flag, safeguarded 60-step refiner,
nearest-unit tie winner, and strict positive-gain/interior/tempering predicates
remain the baseline rules. Per-child terms use C layout; the final child fold is
serial in input order even when the executor evaluates terms concurrently.

Method: docs/methods/pedigree.md#cross-chromosome-evidence-calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from ...contracts.pedigree import CalibrationRecord, GenomeScores, edge_index
from ...core import ties
from ...core.numerics import chromosome_sum
from .alternatives import Alternatives
from .parameters import CALIBRATION_NUMERICS, PedigreeConfig

__all__ = ["fit_scale", "CalibrationExecutor", "CalibrationProblem", "problem", "solve", "uses_executor"]


class CalibrationExecutor(Protocol):
    """Stage-owned term evaluation; returned child rows retain input order."""

    @property
    def workers(self) -> int: ...

    def calibration_terms(self, prepared, kind: str, scales: tuple[float, ...]) -> np.ndarray:
        """Every child's terms at each scale, without cross-child reduction."""
        ...


# These aliases keep the numerical vocabulary of the objective and its replay hooks.
LOG_SCALE_BOUNDS = CALIBRATION_NUMERICS.log_scale_bounds
GRID_POINTS = CALIBRATION_NUMERICS.grid_points
OPTIMIZER_XATOL = CALIBRATION_NUMERICS.optimizer_xatol
MINIMUM_GAIN = CALIBRATION_NUMERICS.minimum_gain
BOUNDARY_MARGIN = CALIBRATION_NUMERICS.boundary_margin
REFINE_TOLERANCE = CALIBRATION_NUMERICS.refine_tolerance
REFINE_ITERATIONS = CALIBRATION_NUMERICS.refine_iterations
BRACKET_EXPANSIONS = CALIBRATION_NUMERICS.bracket_expansions


@dataclass(frozen=True)
class CalibrationProblem:
    """The objective's prepared per-child terms and the report so far (``problem``), before the scale search.

    ``prepared`` is None when calibration is disabled (the report is final) and empty when no scale is
    identifiable (too few contigs or alternatives); otherwise each child's C-layout terms in child order.
    """

    report: dict
    prepared: tuple[tuple[np.ndarray, ...], ...] | None
    reverse: bool


def fit_scale(
    scores: GenomeScores,
    alternatives: Alternatives,
    config: PedigreeConfig,
    executor: CalibrationExecutor | None = None,
) -> tuple[float, CalibrationRecord]:
    """Applied scale and typed diagnostics; use scale 1 when disabled or unidentified.

    Independent child terms may run through the supplied stage executor. Their
    final fold always follows child order, including every adaptive objective call.
    """
    return solve(problem(scores, alternatives, config), executor)


def problem(scores: GenomeScores, alternatives: Alternatives, config: PedigreeConfig) -> CalibrationProblem:
    """Folds, centred terms and reciprocal masks of the PD03 objective (fit_scale's first part)."""
    report = dict(
        mode=config.evidence_calibration,
        scale=1.0,
        selected=False,
        status="disabled",
        truth_used=False,
        parent_count_prior_fitted=False,
        conditional_on_fixed_paintings_and_candidate_panel=True,
        resampling_conditions_on_fitted_scale=True,
        interpretation="composite predictive evidence weight, not posterior calibration",
    )
    if config.evidence_calibration == "off":
        return CalibrationProblem(report, None, config.reverse_conventions)
    fitted, prepared = _prepare(scores, alternatives, config)
    report.update(fitted)
    return CalibrationProblem(report, tuple(prepared), config.reverse_conventions)


def solve(
    calibration: CalibrationProblem, executor: CalibrationExecutor | None = None
) -> tuple[float, CalibrationRecord]:
    """The scale search on a prepared problem (fit_scale's second part); terms may run through ``executor``."""
    report = dict(calibration.report)
    if calibration.prepared is None:
        return 1.0, CalibrationRecord(**report)
    if calibration.prepared:
        with _Objective(calibration.prepared, executor) as objective:
            report.update(
                _select(objective.prediction, objective.derivatives, objective.predictions, reverse=calibration.reverse)
            )
    report["seconds"] = 0.0
    return float(report["scale"]), CalibrationRecord(**report)


def uses_executor(prepared: tuple[tuple[np.ndarray, ...], ...] | None, workers: int) -> bool:
    """Whether the objective sends its child terms to a term executor of ``workers`` (else serial)."""
    if not prepared:
        return False
    terms = sum(len(term[2]) for term in prepared)
    return workers > 1 and terms >= CALIBRATION_NUMERICS.parallel_minimum_terms and len(prepared) > 1


def _refine(derivatives, start, low, high):
    """The stationary point of a smooth maximum near ``start`` inside [low, high]: the root of the first
    derivative (``derivatives(x) -> (f', f'')``), bracketed around ``start`` and found by Newton steps safeguarded
    by bisection. Returns ``start`` when no sign change of f' (+ to -) brackets a maximum."""
    width = 4.0 * OPTIMIZER_XATOL
    left = right = None
    for _ in range(BRACKET_EXPANSIONS):
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


def _maximize(prediction, derivatives=None, predictions=None, *, reverse=False):
    """Bracket the visible maxima of a scalar objective on the grid, then refine each (Brent, then the root of the
    derivative when ``derivatives`` is given). Maxima whose values lie in one tie class (nll quantum) are tied: the
    one closest to scale 1 (smallest |log scale|) is taken. ``predictions`` (optional) evaluates prediction at
    several points at once (the grid), with the same values."""
    grid = np.linspace(*LOG_SCALE_BOUNDS, GRID_POINTS)
    values = np.asarray(predictions(grid) if predictions is not None else [prediction(x) for x in grid])
    candidates = [(float(values[i]), float(grid[i])) for i in (0, len(grid) - 1)]
    converged = True
    for i in range(1, len(grid) - 1):
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            if values[i] == values[i - 1] == values[i + 1]:
                continue
            found = minimize_scalar(
                lambda x: -prediction(x),
                bounds=(grid[i - 1], grid[i + 1]),
                method="bounded",
                options={"xatol": OPTIMIZER_XATOL},
            )
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
        best, log_scale = min(tied, key=lambda item: (abs(item[1]), ties.conv_key(item[1], reverse=reverse)))
    else:
        best, log_scale = max(candidates, key=lambda item: (item[0], ties.conv_key(item[1], reverse=reverse)))
    return best, log_scale, converged, grid, values


def _reciprocal_slots(rows: np.ndarray, edges: np.ndarray, n_samples: int) -> list:
    """Per parent slot of the configuration rows: (rows naming a parent there, child, parent, edge-axis
    positions of (child, parent) and of the reverse (parent, child), -1 where the reverse is not an eligible
    edge); ``edges`` is GenomeScores.edges."""
    slots = []
    for column in (1, 2):
        present = np.flatnonzero(rows[:, column] >= 0)
        child, parent = rows[present, 0], rows[present, column]
        slots.append(
            (
                present,
                child,
                parent,
                edge_index(edges, n_samples, child, parent),
                edge_index(edges, n_samples, parent, child),
            )
        )
    return slots


def _reciprocal_dominant_mask(
    n_rows: int, zero: np.ndarray, one: np.ndarray, slots: list, contigs: np.ndarray
) -> np.ndarray:
    """0 / -inf objective weights excluding reciprocal-dominant rows.

    A row naming p as a parent of c is excluded when, on ``contigs``, M1(p with parent c) - M0(p) >=
    M1(c with parent p) - M0(c): genetics cannot order such reciprocal hypotheses (direction and
    family messages settle them). Excluded from the scale's objective only, never from decisions.
    ``one`` float64 [C, E] per eligible edge; ``slots`` from ``_reciprocal_slots``.
    """
    with np.errstate(invalid="ignore"):
        one_total = chromosome_sum(one[contigs])
        zero_total = chromosome_sum(zero[contigs])
    mask = np.zeros(n_rows, dtype=np.float64)
    for present, child, parent, forward, reverse in slots:
        with np.errstate(invalid="ignore"):
            # A reverse dyad off the eligible edges has M1 -inf on every chromosome; ineligible children have
            # -inf M0 as well, and their NaN gains never win.
            reverse_gain = np.where(reverse >= 0, one_total[reverse], -np.inf) - zero_total[parent]
            forward_gain = one_total[forward] - zero_total[child]
        dominant = reverse_gain >= forward_gain
        mask[present[dominant]] = -np.inf
    return mask


def _aggregate_m1(prepared, states, chromosomes):
    """Identical M1 chromosome vectors and objective masks share a counted term.

    The prior still uses the full candidate universe. Masks are part of the key:
    genetic floor equality alone does not imply reciprocal-mask equality.
    """
    train, validation, combined, test, prior, prior_test = prepared
    groups, representatives, multiplicity = {}, [], []
    for row, state in enumerate(states):
        key = None
        if state == 1:
            key = (chromosomes[:, row].tobytes(), prior[:, row].tobytes(), float(prior_test[row]))
        old = groups.get(key) if key is not None else None
        if old is None:
            old = len(representatives)
            if key is not None:
                groups[key] = old
            representatives.append(row)
            multiplicity.append(1)
        else:
            multiplicity[old] += 1
    if len(representatives) == len(states):
        return prepared
    take = np.asarray(representatives, np.int64)
    log_count = np.log(np.asarray(multiplicity, np.float64))
    return (
        train[:, take],
        validation[:, take],
        combined[take],
        test[take],
        prior[:, take] + log_count[None, :],
        prior_test[take] + log_count,
    )


def _prediction_term(term, scale, test):
    """One child's term of the pooled predictive log likelihood (``prepared`` entry ``term``) at ``scale``."""
    train, validation, combined, held_test, p, p_test = term
    if test:
        weights = scale * combined + p_test
        return logsumexp(weights + held_test) - logsumexp(weights)
    weights = scale * train + p
    return np.sum(logsumexp(weights + validation, axis=1) - logsumexp(weights, axis=1))


def _derivative_terms(term, scale):
    """One child's (sum of E'[t], sum of Var'[t], sum of E[t], sum of Var[t]) over the folds (see ``derivatives``)."""
    train, validation, _, _, p, _ = term
    base = scale * train + p
    values = []
    for logits in (base + validation, base):
        top = np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits - top)
        weights /= np.sum(weights, axis=1, keepdims=True)
        mean = np.sum(weights * train, axis=1)
        variance = np.sum(weights * (train - mean[:, None]) ** 2, axis=1)
        values += [float(np.sum(mean)), float(np.sum(variance))]
    return values


def _fold_prediction(values) -> float:
    """The children's terms summed in child order (the serial objective's order)."""
    total = 0.0
    for value in values:
        total += value
    return float(total)


def _fold_derivatives(values, scale):
    """(d/dx, d2/dx2) from the children's derivative terms, summed in child order (the serial order)."""
    mean_difference = variance_difference = 0.0
    for mean_plus, variance_plus, mean_minus, variance_minus in values:
        for mean, variance, sign in ((mean_plus, variance_plus, 1.0), (mean_minus, variance_minus, -1.0)):
            mean_difference += sign * mean
            variance_difference += sign * variance
    return scale * mean_difference, scale * mean_difference + scale * scale * variance_difference


class _Objective:
    """Fold independent predictive terms in child order, preserving PD03 arithmetic."""

    def __init__(self, prepared, executor: CalibrationExecutor | None = None):
        self.prepared = prepared
        self.executor = executor if executor is not None and uses_executor(prepared, executor.workers) else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def prediction(self, log_scale, test=False):
        scale = np.exp(log_scale)
        if self.executor is None:
            return _fold_prediction(_prediction_term(term, scale, test) for term in self.prepared)
        terms = self.executor.calibration_terms(self.prepared, "test" if test else "prediction", (scale,))
        return _fold_prediction(terms[:, 0])

    def predictions(self, log_scales):
        if self.executor is None:
            return [self.prediction(x) for x in log_scales]
        terms = self.executor.calibration_terms(self.prepared, "prediction", tuple(np.exp(x) for x in log_scales))
        return [_fold_prediction(terms[:, index]) for index in range(len(log_scales))]

    def derivatives(self, log_scale):
        scale = float(np.exp(log_scale))
        if self.executor is None:
            return _fold_derivatives((_derivative_terms(term, scale) for term in self.prepared), scale)
        terms = self.executor.calibration_terms(self.prepared, "derivative", (scale,))
        return _fold_derivatives(terms[:, 0], scale)


def _prepare(
    scores: GenomeScores,
    alternatives: Alternatives,
    config: PedigreeConfig,
) -> tuple[dict, list]:
    folds = np.arange(len(scores.contigs)) % 4
    groups = [np.flatnonzero(folds == i) for i in range(4)]
    report = dict(
        scale=1.0,
        selected=False,
        status="insufficient_contigs",
        estimator="pooled-three-fold-predictive-v2-reciprocal-excluded",
        partitions={f"fold_{i}": [str(scores.contigs[j]) for j in group] for i, group in enumerate(groups)},
        testing_fold="fold_3",
        training_scheme="each non-test fold predicted by the other two; test uses all non-test contigs",
        search_scale_bounds=list(np.exp(LOG_SCALE_BOUNDS)),
    )
    if min(map(len, groups)) < 2:
        return report, []
    rows, states = alternatives.rows, alternatives.states
    likelihoods = alternatives.log_likelihoods
    held = np.stack([chromosome_sum(likelihoods[folds == i]) for i in range(3)])
    training = np.stack([chromosome_sum(likelihoods[(folds < 3) & (folds != i)]) for i in range(3)])
    testing = chromosome_sum(likelihoods[folds == 3])
    all_training = chromosome_sum(likelihoods[folds < 3])
    prior = np.log(np.asarray(config.selection.state_priors)[states])
    prior -= np.log(np.maximum(1, alternatives.full_counts[rows[:, 0], states]))
    slots = _reciprocal_slots(rows, scores.edges, len(scores.sample_ids))
    fit_mask = np.stack(
        [
            _reciprocal_dominant_mask(len(rows), scores.zero, scores.one, slots, (folds < 3) & (folds != i))
            for i in range(3)
        ]
    )
    test_mask = _reciprocal_dominant_mask(len(rows), scores.zero, scores.one, slots, folds < 3)
    prepared, omitted = [], 0
    expanded_terms = aggregated_terms = 0
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
        validation = validation - np.max(np.where(fit_kept, validation, -np.inf), axis=1, keepdims=True)
        combined = all_training[valid] - all_training[valid].max()
        test = testing[valid] - testing[valid][test_kept].max()
        terms = (
            train,
            validation,
            combined,
            test,
            prior[valid][None, :] + fit_mask[:, valid],
            prior[valid] + test_mask[valid],
        )
        compact = _aggregate_m1(terms, states[valid], likelihoods[:, valid])
        expanded_terms += len(valid)
        aggregated_terms += len(compact[2])
        prepared.append(tuple(np.ascontiguousarray(value) for value in compact))
    report.update(
        eligible_children=len(prepared),
        omitted_nonfinite_alternatives=omitted,
        expanded_objective_terms=expanded_terms,
        aggregated_objective_terms=aggregated_terms,
        objective_aggregation="identical-M1-vectors-and-reciprocal-masks-counted",
        reciprocal_dominant_rows_excluded_per_fold=[int(np.count_nonzero(~np.isfinite(mask))) for mask in fit_mask],
    )
    if not prepared:
        report["status"] = "insufficient_alternatives"
    return report, prepared


def _select(prediction, derivatives, predictions=None, *, reverse=False) -> dict:
    """The fitted scale, its selection and the report's search and test diagnostics."""
    report: dict[str, object] = {}
    best, log_scale, success, grid, values = _maximize(prediction, derivatives, predictions, reverse=reverse)
    gain = best - prediction(0.0)
    boundary = min(log_scale - LOG_SCALE_BOUNDS[0], LOG_SCALE_BOUNDS[1] - log_scale) < BOUNDARY_MARGIN
    # Tempering only: a scale above one would sharpen evidence without calibration support.
    above_unit = log_scale >= 0.0
    selected = bool(success and np.isfinite(gain) and gain > MINIMUM_GAIN and not boundary and not above_unit)
    applied_log_scale = log_scale if selected else 0.0
    report.update(
        scale=float(np.exp(log_scale)) if selected else 1.0,
        fitted_scale=float(np.exp(log_scale)),
        selected=selected,
        # Release 2.1: without predictive gain the optimum is unidentified and its location (boundary or not) is
        # rounding- and noise-driven on the flat objective, so the gain test comes before the boundary label.
        status=(
            "selected"
            if selected
            else "optimizer_failed"
            if not success
            else "no_predictive_gain"
            if not (np.isfinite(gain) and gain > MINIMUM_GAIN)
            else "boundary_optimum"
            if boundary
            else "above_unit_scale"
        ),
        optimizer_success=success,
        at_search_boundary=bool(boundary),
        selection_gain=float(gain),
        search_log_scale_grid=grid.tolist(),
        search_predictive_grid=values.tolist(),
        # Neither test diagnostic selects; they separate the unapplied candidate from the fallback.
        untouched_test_gain=prediction(applied_log_scale, True) - prediction(0.0, True),
        fitted_untouched_test_gain=prediction(log_scale, True) - prediction(0.0, True),
    )
    return report
