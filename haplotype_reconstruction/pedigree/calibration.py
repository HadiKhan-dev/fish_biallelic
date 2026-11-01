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
from .records import GenomeScores, edge_index

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
# Release 3.5: the objective is a sum of independent per-child terms. With at least this many (aggregated) terms and
# more than one CPU, each evaluation is computed by a process pool (every child's term by the same NumPy/SciPy
# operations on copies of the same arrays) and folded in child order, so every value is the serial one.
POOL_MINIMUM_TERMS = 200_000
# Pool tasks per worker in one evaluation: one contiguous child range of about equal cost per worker (each task
# costs the parent a dispatch; ~100 evaluations are sequential).
POOL_TASKS_PER_WORKER = 1


def fit_scale(scores: GenomeScores, alternatives: Alternatives,
              config: PedigreeConfig, budget=None) -> tuple[float, dict]:
    """(scale to apply, JSON-safe report). Scale 1 when calibration is off or not selected.

    ``budget``: the CPU budget (execution.CpuBudget) for evaluating the objective in parallel; None evaluates it
    serially (the same values)."""
    report = dict(
        mode=config.evidence_calibration, scale=1.0, selected=False, status="disabled",
        truth_used=False, parent_count_prior_fitted=False,
        conditional_on_fixed_paintings_and_candidate_panel=True,
        resampling_conditions_on_fitted_scale=True,
        interpretation="composite predictive evidence weight, not posterior calibration")
    if config.evidence_calibration == "off":
        return 1.0, report
    started = time.perf_counter()
    report.update(_fit(scores, alternatives, config, budget))
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


def _maximize(prediction, derivatives=None, predictions=None):
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


def _reciprocal_slots(rows: np.ndarray, edges: np.ndarray, n_samples: int) -> list:
    """Per parent slot of the configuration rows: (rows naming a parent there, child, parent, edge-axis
    positions of (child, parent) and of the reverse (parent, child), -1 where the reverse is not an eligible
    edge); ``edges`` is GenomeScores.edges."""
    slots = []
    for column in (1, 2):
        present = np.flatnonzero(rows[:, column] >= 0)
        child, parent = rows[present, 0], rows[present, column]
        slots.append((present, child, parent, edge_index(edges, n_samples, child, parent),
                      edge_index(edges, n_samples, parent, child)))
    return slots


def _reciprocal_dominant_mask(n_rows: int, zero: np.ndarray, one: np.ndarray, slots: list,
                              contigs: np.ndarray) -> np.ndarray:
    """0 / -inf objective weights excluding reciprocal-dominant rows.

    A row naming p as a parent of c is excluded when, on ``contigs``, M1(p with parent c) - M0(p) >=
    M1(c with parent p) - M0(c): genetics cannot order such reciprocal hypotheses (direction and
    family messages settle them). Excluded from the scale's objective only, never from decisions.
    ``one`` float64 [C, E] per eligible edge; ``slots`` from ``_reciprocal_slots``.
    """
    with np.errstate(invalid="ignore"):
        one_total = one[contigs].sum(axis=0)
        zero_total = zero[contigs].sum(axis=0)
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
            key = (chromosomes[:, row].tobytes(), prior[:, row].tobytes(),
                   float(prior_test[row]))
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
    return (train[:, take], validation[:, take], combined[take], test[take],
            prior[:, take] + log_count[None, :], prior_test[take] + log_count)


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


_TERM_FIELDS = (("train", 3), ("validation", 3), ("combined", 1), ("test", 1), ("prior", 3), ("prior_test", 1))


def _fortran_layouts(prepared) -> np.ndarray | None:
    """bool [children, fields]: whether each [3, terms] array of ``prepared`` is Fortran-ordered (as fancy indexing
    makes them); None if some array is neither C- nor Fortran-contiguous. NumPy's reductions follow the memory
    layout, so a pooled copy keeps it to reproduce every value."""
    layouts = np.zeros((len(prepared), len(_TERM_FIELDS)), dtype=np.bool_)
    for child, term in enumerate(prepared):
        for index, array in enumerate(term):
            if array.flags.c_contiguous:
                continue
            if not array.flags.f_contiguous:
                return None
            layouts[child, index] = True
    return layouts


def _flatten_terms(prepared, layouts) -> dict[str, np.ndarray]:
    """Shared-memory layout of ``prepared``: each field's per-child blocks (in their own memory order) concatenated
    in child order, with int64 ``offsets`` [children + 1] of the children's term counts and the ``layouts``."""
    counts = np.asarray([len(term[2]) for term in prepared], dtype=np.int64)
    arrays = {"offsets": np.concatenate(([0], np.cumsum(counts))).astype(np.int64), "layouts": layouts}
    for index, (name, _) in enumerate(_TERM_FIELDS):
        arrays[name] = np.concatenate([np.ravel(term[index], order="F" if layouts[child, index] else "C")
                                       for child, term in enumerate(prepared)])
    return arrays


def _shared_term(shared, child):
    """Child ``child``'s ``prepared`` entry, copied out of the shared layout into fresh arrays of the original
    shapes and memory orders."""
    start, stop = int(shared["offsets"][child]), int(shared["offsets"][child + 1])
    count = stop - start
    term = []
    for index, (name, rows) in enumerate(_TERM_FIELDS):
        block = shared[name][rows * start:rows * stop]
        order = "F" if shared["layouts"][child, index] else "C"
        term.append(block.reshape((3, count) if rows == 3 else (count,), order=order).copy(order=order))
    return tuple(term)


def _pool_terms(task, shared, budget):
    """Pool worker: the terms of children ``start`` .. ``stop - 1`` at each of ``scales`` (float64 [children,
    scales] for predictions, [children, scales, 4] for derivatives)."""
    kind, scales, start, stop = task
    terms = [_shared_term(shared, child) for child in range(start, stop)]
    if kind == "derivatives":
        return np.asarray([[_derivative_terms(term, scale) for scale in scales] for term in terms],
                          dtype=np.float64).reshape(len(terms), len(scales), 4)
    test = kind == "test"
    return np.asarray([[_prediction_term(term, scale, test) for scale in scales] for term in terms],
                      dtype=np.float64).reshape(len(terms), len(scales))


class _Objective:
    """The predictive objective and its derivatives over ``prepared``: serial, or by a pool of the budget's CPUs
    for large inputs. Either way every child's term is computed by _prediction_term / _derivative_terms and the
    terms are folded in child order, so the values do not depend on the CPU count. Use as a context manager."""

    def __init__(self, prepared, budget):
        self.prepared = prepared
        self.pool = None
        self.chunks = None
        terms = sum(len(term[2]) for term in prepared)
        cpus = 1 if budget is None else int(budget.threads())
        layouts = None
        if budget is not None and cpus > 1 and terms >= POOL_MINIMUM_TERMS and len(prepared) > 1:
            layouts = _fortran_layouts(prepared)
        if layouts is not None:
            from ..execution import BudgetedPool
            workers = min(cpus, len(prepared))
            # Contiguous child ranges of about equal term counts (plus a per-child overhead).
            cost = np.cumsum(np.asarray([len(term[2]) for term in prepared], dtype=np.float64) + 500.0)
            pieces = min(len(prepared), workers * POOL_TASKS_PER_WORKER)
            cuts = np.searchsorted(cost, cost[-1] * np.arange(1, pieces) / pieces, side="right")
            bounds = np.unique(np.concatenate(([0], cuts, [len(prepared)])))
            self.chunks = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
            self.pool = BudgetedPool(budget, processes=min(workers, len(self.chunks)),
                                     shared=_flatten_terms(prepared, layouts), label="pedigree calibration")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.__exit__(*exc)
            self.pool = None
        return False

    def _terms(self, kind, scales):
        """Every child's terms at each of ``scales``, children in order: [children, scales(, 4)]."""
        results = self.pool.map(_pool_terms, [(kind, scales, a, b) for a, b in self.chunks])
        return np.concatenate(results)

    def prediction(self, log_scale, test=False):
        scale = np.exp(log_scale)
        if self.pool is None:
            return _fold_prediction(_prediction_term(term, scale, test) for term in self.prepared)
        return _fold_prediction(self._terms("test" if test else "prediction", [scale])[:, 0])

    def predictions(self, log_scales):
        """[prediction(x) for x in log_scales], with one pool batch for all of them."""
        if self.pool is None:
            return [self.prediction(x) for x in log_scales]
        terms = self._terms("prediction", [np.exp(x) for x in log_scales])
        return [_fold_prediction(terms[:, index]) for index in range(len(log_scales))]

    def derivatives(self, log_scale):
        """(d/dx, d2/dx2) of prediction(x), x = log scale: s (E'[t] - E[t]) and s (E'[t] - E[t]) + s^2 (Var'[t] -
        Var[t]) summed over children and folds, with E/Var under softmax(s t + p) and E'/Var' under
        softmax(s t + p + validation) (t = train)."""
        scale = float(np.exp(log_scale))
        if self.pool is None:
            return _fold_derivatives((_derivative_terms(term, scale) for term in self.prepared), scale)
        return _fold_derivatives(self._terms("derivatives", [scale])[:, 0], scale)


def _fit(scores: GenomeScores, alternatives: Alternatives, config: PedigreeConfig, budget=None) -> dict:
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
    slots = _reciprocal_slots(rows, scores.edges, len(scores.sample_ids))
    fit_mask = np.stack([_reciprocal_dominant_mask(len(rows), scores.zero, scores.one, slots,
                                                   (folds < 3) & (folds != i)) for i in range(3)])
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
        validation = validation - np.max(np.where(fit_kept, validation, -np.inf), axis=1,
                                         keepdims=True)
        combined = all_training[valid] - all_training[valid].max()
        test = testing[valid] - testing[valid][test_kept].max()
        terms = (train, validation, combined, test,
                 prior[valid][None, :] + fit_mask[:, valid], prior[valid] + test_mask[valid])
        compact = _aggregate_m1(terms, states[valid], likelihoods[:, valid])
        expanded_terms += len(valid)
        aggregated_terms += len(compact[2])
        prepared.append(compact)
    report.update(eligible_children=len(prepared), omitted_nonfinite_alternatives=omitted,
                  expanded_objective_terms=expanded_terms, aggregated_objective_terms=aggregated_terms,
                  objective_aggregation="identical-M1-vectors-and-reciprocal-masks-counted",
                  reciprocal_dominant_rows_excluded_per_fold=[
                      int(np.count_nonzero(~np.isfinite(mask))) for mask in fit_mask])
    if not prepared:
        report["status"] = "insufficient_alternatives"
        return report

    with _Objective(prepared, budget) as objective:
        report.update(_select(objective.prediction, objective.derivatives, objective.predictions))
    return report


def _select(prediction, derivatives, predictions=None) -> dict:
    """The fitted scale, its selection and the report's search and test diagnostics."""
    report = {}
    best, log_scale, success, grid, values = _maximize(prediction, derivatives, predictions)
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
