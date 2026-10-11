"""Cross-chromosome predictive weighting of composite pedigree evidence.

This fits one nuisance parameter conditional on the existing paintings and
candidate panel. It is not calibration of biological posterior probabilities:
painting, screening, direction, and relatives are not independently cross-fit.
The parent-count prior, candidate universe, direction model and release rules
remain fixed. Raw-marker Mendelian exclusion is never tempered here.
"""
from dataclasses import replace
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from . import eligibility, states


# Broad numerical bounds, not fitted simulation constants. A boundary optimum
# is unidentified and retains the unmodified model. The grid only brackets
# numerical optima; predictive mixtures are not guaranteed to be unimodal.
_LOG_SCALE_BOUNDS = (float(np.log(0.001)), float(np.log(100.0)))


def prepare_decision_scores(scored, settings, parent_eligibility=None):
    """Return ephemeral scaled scores and a serializable fit/fallback record.

    Stored genetic evidence is not mutated. The decision cache includes this
    module and the calibration setting; its upstream genetic-score cache does not.
    Bootstrap and LOCO refits condition on the single fitted scale, rather than
    estimating it again within each replicate.
    """
    settings.validated()
    report = dict(
        mode=settings.evidence_calibration, scale=1.0, selected=False,
        status="disabled", truth_used=False, parent_count_prior_fitted=False,
        conditional_on_fixed_paintings_and_candidate_panel=True,
        resampling_conditions_on_fitted_scale=True,
        interpretation="composite predictive evidence weight, not posterior calibration",
    )
    if settings.evidence_calibration == "off":
        return scored, report
    started = time.perf_counter()
    resolved = eligibility._resolve_parent_eligibility(parent_eligibility, scored.sample_ids)
    report.update(_fit(scored, settings, resolved))
    report["seconds"] = time.perf_counter() - started
    scale = report["scale"]
    if not report["selected"]:
        return scored, report
    chromosomes = tuple(replace(
        c,
        zero_parent_log_likelihoods=c.zero_parent_log_likelihoods * scale,
        one_parent_log_likelihoods=c.one_parent_log_likelihoods * scale,
        two_parent_log_likelihoods=c.two_parent_log_likelihoods * scale,
    ) for c in scored.chromosomes)
    return replace(
        scored, chromosomes=chromosomes, runtime_chromosome_results=None,
        score_identity=dict(scored.score_identity, predictive_evidence_scale=scale),
    ), report


def _maximize_predictive(prediction):
    """Bracket visible maxima without assuming a unimodal scalar objective."""
    grid = np.linspace(*_LOG_SCALE_BOUNDS, 33)
    values = np.asarray([prediction(x) for x in grid])
    candidates = [(float(values[i]), float(grid[i])) for i in (0, len(grid)-1)]
    converged = True
    for i in range(1, len(grid)-1):
        if values[i] >= values[i-1] and values[i] >= values[i+1]:
            if values[i] == values[i-1] == values[i+1]:
                continue
            found = minimize_scalar(
                lambda x: -prediction(x), bounds=(grid[i-1], grid[i+1]),
                method="bounded", options={"xatol": 1e-5})
            converged &= bool(found.success)
            if found.success:
                candidates.append((prediction(found.x), float(found.x)))
    best, log_scale = max(candidates)
    return best, log_scale, converged, grid, values


def _reciprocal_dominant_mask(alternatives, zero, one, contigs):
    """Return 0/-inf objective weights excluding reciprocal-dominant rows.

    An M1/M2 row proposing p as a parent of c is excluded when, on the given
    contigs, M1(p with parent c) - M0(p) >= M1(c with parent p) - M0(c). The
    genetic score cannot order such reciprocal hypotheses; direction and
    reciprocal family messages settle them. They are excluded from the scale's
    objective only, never from decisions.
    """
    with np.errstate(invalid="ignore"):
        # Ineligible children have -inf M0 and M1; their NaN gains never win.
        gain = one[contigs].sum(axis=0) - zero[contigs].sum(axis=0)[:, None]
    mask = np.zeros(len(alternatives), dtype=np.float64)
    child = alternatives[:, 0]
    for column in (1, 2):
        rows = np.flatnonzero(alternatives[:, column] >= 0)
        parent = alternatives[rows, column]
        dominant = gain[parent, child[rows]] >= gain[child[rows], parent]
        mask[rows[dominant]] = -np.inf
    return mask


def _fit(scored, settings, resolved):
    # Whole physical contigs, not adjacent markers, define the split. Each of
    # folds 0/1/2 is predicted by the other two; fold 3 never selects the scale.
    folds = np.arange(len(scored.chromosomes)) % 4
    groups = [np.flatnonzero(folds == i) for i in range(4)]
    report = dict(
        scale=1.0, selected=False, status="insufficient_contigs",
        estimator="pooled-three-fold-predictive-v2-reciprocal-excluded",
        partitions={f"fold_{i}": [str(scored.contig_names[j]) for j in group]
                    for i, group in enumerate(groups)},
        testing_fold="fold_3",
        training_scheme="each non-test fold predicted by the other two; test uses all non-test contigs",
        search_scale_bounds=list(np.exp(_LOG_SCALE_BOUNDS)),
    )
    if min(map(len, groups)) < 2:
        return report
    zero = np.stack([c.zero_parent_log_likelihoods for c in scored.chromosomes])
    one = np.stack([c.one_parent_log_likelihoods for c in scored.chromosomes])
    alternatives, state, scores, by_child, full_counts, _ = states._parent_state_alternatives(
        scored.trios,
        zero,
        one,
        np.stack([c.two_parent_log_likelihoods for c in scored.chromosomes]),
        settings.parent_state_contamination_probability, resolved,
        candidate_source_mode=settings.parent_state_candidate_source_mode,
    )
    held = np.stack([scores[folds == i].sum(axis=0) for i in range(3)])
    training = np.stack([scores[(folds < 3) & (folds != i)].sum(axis=0)
                         for i in range(3)])
    testing = scores[folds == 3].sum(axis=0)
    all_training = scores[folds < 3].sum(axis=0)
    prior = np.log(np.asarray(settings.parent_state_priors)[state])
    prior -= np.log(np.maximum(1, full_counts[alternatives[:, 0], state]))
    # Objective-only masks (0 or -inf), from the contigs that fit each fold.
    fit_mask = np.stack([
        _reciprocal_dominant_mask(alternatives, zero, one, (folds < 3) & (folds != i))
        for i in range(3)])
    test_mask = _reciprocal_dominant_mask(alternatives, zero, one, folds < 3)
    prepared, omitted = [], 0
    for rows in by_child:
        finite = np.isfinite(scores[:, rows]).all(axis=0)
        omitted += int(np.count_nonzero(~finite))
        valid = rows[finite]
        if len(valid) < 2:
            continue
        # A finite M0 row is never masked, so every fold keeps a maximum.
        fit_kept = np.isfinite(fit_mask[:, valid])
        test_kept = np.isfinite(test_mask[valid])
        if not (fit_kept.any(axis=1).all() and test_kept.any()):
            continue
        train, validation = training[:, valid], held[:, valid]
        train = train - train.max(axis=1, keepdims=True)
        validation = validation - np.max(
            np.where(fit_kept, validation, -np.inf), axis=1, keepdims=True)
        combined = all_training[valid] - all_training[valid].max()
        test = testing[valid] - testing[valid][test_kept].max()
        prepared.append((train, validation, combined, test,
                         prior[valid][None, :] + fit_mask[:, valid],
                         prior[valid] + test_mask[valid]))
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

    best, log_scale, success, grid, values = _maximize_predictive(prediction)
    gain = best - prediction(0.0)
    boundary = min(log_scale-_LOG_SCALE_BOUNDS[0], _LOG_SCALE_BOUNDS[1]-log_scale) < 0.01
    # Tempering only: a scale above one would sharpen evidence without calibration support.
    above_unit = log_scale >= 0.0
    selected = bool(success and np.isfinite(gain) and gain > 1e-8 and not boundary
                    and not above_unit)
    applied_log_scale = log_scale if selected else 0.0
    report.update(
        scale=float(np.exp(log_scale)) if selected else 1.0,
        fitted_scale=float(np.exp(log_scale)), selected=selected,
        status=("selected" if selected else "boundary_optimum" if boundary else
                "optimizer_failed" if not success else
                "above_unit_scale" if above_unit and gain > 1e-8 else "no_predictive_gain"),
        optimizer_success=success, at_search_boundary=bool(boundary),
        selection_gain=float(gain), search_log_scale_grid=grid.tolist(),
        search_predictive_grid=values.tolist(),
        # Neither test diagnostic participates in selection. Distinguish the
        # unapplied candidate from the fallback actually used downstream.
        untouched_test_gain=prediction(applied_log_scale, True)-prediction(0.0, True),
        fitted_untouched_test_gain=prediction(log_scale, True)-prediction(0.0, True),
    )
    return report
