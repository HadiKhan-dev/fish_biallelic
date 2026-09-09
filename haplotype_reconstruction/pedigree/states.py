"""pedigree / states for the canonical reconstruction pipeline."""
from __future__ import annotations


from dataclasses import dataclass
import math


from typing import Any, Mapping, Optional, Sequence


import numpy as np
import pandas as pd
import haplotype_reconstruction.pedigree.direction as pedigree_direction
import haplotype_reconstruction.pedigree.models as pedigree_models

_CONTRAST_ULP_FACTOR = 512.0


def _contrast_tolerance(values: np.ndarray) -> float:
    """Tolerance covering scale-dependent floating evaluation-order noise."""
    finite = np.asarray(values, dtype=np.float64)
    if finite.ndim != 1 or len(finite) == 0 or np.any(~np.isfinite(finite)):
        raise pedigree_models.PedigreeEvidenceError("candidate utilities must be finite vectors")
    scale = max(1.0, float(np.max(np.abs(finite))))
    return _CONTRAST_ULP_FACTOR * np.finfo(np.float64).eps * scale


def _tied_rank_probabilities(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    n_values = len(values)
    if n_values == 1:
        return np.ones(1, dtype=np.float64)
    tolerance = _contrast_tolerance(values)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(n_values, dtype=np.float64)
    start = 0
    while start < n_values:
        end = start + 1
        while (
            end < n_values
            and abs(values[order[end]] - values[order[start]]) <= tolerance
        ):
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    evidence = (n_values - ranks) / n_values
    evidence /= np.sum(evidence)
    return evidence


def _information_weights(
    markers: np.ndarray, config: module_pedigree_config.PedigreeConfig
) -> np.ndarray:
    weights = np.sqrt(np.asarray(markers, dtype=np.float64))
    median = float(np.median(weights))
    if median <= 0.0:
        raise pedigree_models.PedigreeEvidenceError("marker information is empty")
    ratio = config.maximum_contig_weight_ratio
    weights = np.clip(weights / median, 1.0 / ratio, ratio)
    return weights / np.sum(weights)


def _parent_state_alternatives(
    trios: np.ndarray,
    zero: np.ndarray,
    one: np.ndarray,
    two: np.ndarray,
    contamination: float,
    eligibility: Optional[pedigree_eligibility._ResolvedParentEligibility] = None,
    candidate_source_mode: str = "hard_painted",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """Build M0/M1/M2 configurations inside the eligible universe."""
    n_contigs, n_samples = zero.shape
    if eligibility is None:
        eligibility = pedigree_eligibility._resolve_parent_eligibility(None, range(n_samples))
    trio_children = trios[:, 0]
    child_starts = np.searchsorted(
        trio_children,
        np.arange(n_samples + 1, dtype=np.int64),
    ).astype(np.int64)
    eligible_two_rows = []
    two_counts = np.zeros(n_samples, dtype=np.int64)
    for child in range(n_samples):
        rows = np.arange(
            int(child_starts[child]),
            int(child_starts[child + 1]),
            dtype=np.int64,
        )
        if len(rows):
            child_trios = trios[rows]
            child_rows = rows[
                pedigree_eligibility._eligible_parent_pair_mask(
                    eligibility,
                    child,
                    child_trios[:, 1],
                    child_trios[:, 2],
                )
            ]
        else:
            child_rows = np.empty(0, dtype=np.int64)
        eligible_two_rows.append(child_rows)
        two_counts[child] = len(child_rows)
    parent_counts = np.count_nonzero(
        eligibility.eligible_parents, axis=1
    ).astype(np.int64)
    n_rows = int(np.sum(
        eligibility.eligible_children * (
            1 + parent_counts + two_counts

        )
    ))
    alternatives = np.empty((n_rows, 3), dtype=np.int64)
    states = np.empty(n_rows, dtype=np.int8)
    log_likelihoods = np.empty((n_contigs, n_rows), dtype=np.float64)
    by_child = []
    scored_counts = np.zeros((n_samples, 3), dtype=np.int64)
    offset = 0
    for child in range(n_samples):
        start = offset
        if not eligibility.eligible_children[child]:
            by_child.append(np.empty(0, dtype=np.int64))
            continue
        alternatives[offset] = (child, pedigree_models._EXTERNAL_PARENT, pedigree_models._EXTERNAL_PARENT)
        states[offset] = pedigree_models._ZERO_OBSERVED
        log_likelihoods[:, offset] = zero[:, child]
        offset += 1

        parents = np.flatnonzero(eligibility.eligible_parents[child])
        one_end = offset + len(parents)
        alternatives[offset:one_end, 0] = child
        alternatives[offset:one_end, 1] = parents
        alternatives[offset:one_end, 2] = pedigree_models._EXTERNAL_PARENT
        states[offset:one_end] = pedigree_models._ONE_OBSERVED
        log_likelihoods[:, offset:one_end] = one[:, child, parents]
        offset = one_end

        child_two_rows = eligible_two_rows[child]
        two_end = offset + len(child_two_rows)
        alternatives[offset:two_end] = trios[child_two_rows]
        states[offset:two_end] = pedigree_models._TWO_OBSERVED
        log_likelihoods[:, offset:two_end] = two[:, child_two_rows]
        offset = two_end
        rows = np.arange(start, offset, dtype=np.int64)
        by_child.append(rows)
        scored_counts[child] = (
            1,
            int(parent_counts[child]),
            len(child_two_rows),
        )
    if offset != n_rows:
        raise AssertionError("internal parent-state alternative count mismatch")


    full_counts = np.zeros((n_samples, 3), dtype=np.int64)
    full_counts[:, pedigree_models._ZERO_OBSERVED] = eligibility.eligible_children.astype(
        np.int64
    )
    full_counts[:, pedigree_models._ONE_OBSERVED] = np.count_nonzero(
        eligibility.eligible_parents, axis=1
    )
    full_counts[:, pedigree_models._TWO_OBSERVED] = pedigree_eligibility._eligible_parent_pair_counts(eligibility)
    return (
        alternatives,
        states,
        log_likelihoods,
        by_child,
        full_counts,
        scored_counts,
    )


def _apply_aggregate_parent_state_contamination(
    aggregate: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    contamination: float,
) -> np.ndarray:
    """Mix each aggregated candidate relationship with child M0 once."""
    values = np.asarray(aggregate, dtype=np.float64)
    if contamination == 0.0:
        return values
    mixed = values.copy()
    nonzero_rows = np.flatnonzero(states != pedigree_models._ZERO_OBSERVED)
    if not len(nonzero_rows):
        return mixed
    m0_rows = np.flatnonzero(states == pedigree_models._ZERO_OBSERVED)
    represented_children = np.unique(alternatives[:, 0])
    n_samples = int(represented_children[-1]) + 1
    m0_children = alternatives[m0_rows, 0]
    m0_counts = np.bincount(m0_children, minlength=n_samples)
    if np.any(m0_counts[represented_children] != 1):
        raise pedigree_models.PedigreeEvidenceError(
            "aggregate contamination requires exactly one M0 row per child"
        )
    m0_by_child = np.full(n_samples, -1, dtype=np.int64)
    m0_by_child[m0_children] = m0_rows
    children = alternatives[nonzero_rows, 0]
    mixed[nonzero_rows] = np.logaddexp(
        math.log1p(-contamination) + values[nonzero_rows],
        math.log(contamination) + values[m0_by_child[children]],
    )
    return mixed


def _logsumexp_finite(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return -np.inf
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def _softmax_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.zeros_like(values)
    finite = np.isfinite(values)
    if not np.any(finite):
        return output
    maximum = float(np.max(values[finite]))
    output[finite] = np.exp(values[finite] - maximum)
    output /= np.sum(output)
    return output


@pedigree_models.njit(cache=True)
def _linear_top_two_finite(rows, scores):
    """Stable O(n) top-two scan used by repeated bootstrap selections."""
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
            # Strict comparison retains the first occurrence as the stable
            # winner while still making an equal later value the runner-up.
            second_value = value
    return (
        best_row,
        best_value,
        second_value,
        finite_count,
        maximum_absolute_value,
    )


def _unique_finite_winner(
    rows: np.ndarray,
    scores: np.ndarray,
) -> tuple[Optional[int], float]:
    candidate_rows = np.asarray(rows, dtype=np.int64)
    candidate_scores = np.asarray(scores, dtype=np.float64)
    (
        best_row,
        best_value,
        second_value,
        finite_count,
        maximum_absolute_value,
    ) = _linear_top_two_finite(candidate_rows, candidate_scores)
    if finite_count == 0:
        return None, 0.0
    if finite_count == 1:
        return int(best_row), np.inf
    margin = float(best_value - second_value)
    tolerance = (
        _CONTRAST_ULP_FACTOR
        * np.finfo(np.float64).eps
        * maximum_absolute_value
    )
    if margin <= tolerance:
        return None, margin
    return int(best_row), margin


_ANCESTRY_DEPTH_MAX_COMPONENTS = 6


_ANCESTRY_DEPTH_GMM_N_INIT = 10


_ANCESTRY_DEPTH_GMM_MAX_ITERATIONS = 500


_ANCESTRY_DEPTH_GMM_REGULARIZATION = 1e-3


@dataclass(frozen=True)
class _ParentStateSelection:
    state_log_evidence: np.ndarray
    state_scores: np.ndarray
    state_support: np.ndarray
    decision_scores: np.ndarray
    fitted_prior_parameters: np.ndarray
    loo_state_priors: np.ndarray
    local_states: dict[int, int]
    local_rows: dict[int, int]
    graph_rows: dict[int, int]
    graph_tie_conflicts: frozenset[int]
    graph_direction_resolved_children: frozenset[int]
    graph_parent_role_probabilities: dict[int, float]
    ancestry_depth_model: Optional[pedigree_direction._AncestryDepthModel]
    state_margins: np.ndarray
    identity_margins: np.ndarray
    unresolved_reasons: tuple[Optional[str], ...]
    m1_over_m0_edge_gains: np.ndarray
    m2_over_first_m1_edge_gains: np.ndarray
    m2_over_second_m1_edge_gains: np.ndarray
    predictive_fold_count: int
    m1_direction_state_supported: Optional[np.ndarray] = None


def _integrated_parent_state_log_evidence(
    aggregate_log_likelihoods: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
) -> np.ndarray:
    """Log-mean identity evidence, using the full eligible multiplicity."""
    evidence = np.full((len(by_child), 3), -np.inf, dtype=np.float64)
    for child, child_rows in enumerate(by_child):
        for state in range(3):
            rows = child_rows[states[child_rows] == state]
            if len(rows):
                evidence[child, state] = (
                    _logsumexp_finite(aggregate_log_likelihoods[rows])
                    - math.log(float(full_counts[child, state]))
                )
    return evidence


def _balanced_predictive_fold_weights(
    contig_weights: np.ndarray,
    contig_information_weights: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Pair held-out chromosomes by effective information weight.

    Stable heavy--light pairing yields eleven two-chromosome test folds for a
    22-chromosome run, each trained on the complementary twenty. A bootstrap
    chromosome and all of its multiplicity stay in one fold; missing contigs
    are removed before pairing. Odd active counts produce one singleton test
    fold. Fewer than three unique chromosomes cannot form disjoint non-empty
    train/test folds and are explicitly unresolved.
    """
    weights = np.asarray(contig_weights, dtype=np.float64)
    information = np.asarray(contig_information_weights, dtype=np.float64)
    if information.shape != weights.shape:
        raise pedigree_models.PedigreeEvidenceError(
            "predictive contig information weights must match contig weights"
        )
    active = np.flatnonzero(weights > 0.0)
    if len(active) < 3:
        return ()
    effective_information = information[active] * weights[active]
    order = np.lexsort((active, effective_information))
    ordered = active[order]
    test_groups = []
    left = 0
    right = len(ordered) - 1
    while left < right:
        test_groups.append((int(ordered[left]), int(ordered[right])))
        left += 1
        right -= 1
    if left == right:
        test_groups.append((int(ordered[left]),))

    folds = []
    for test_indices in test_groups:
        train = weights.copy()
        test = np.zeros_like(weights)
        for index in test_indices:
            train[index] = 0.0
            test[index] = weights[index]
        folds.append((train, test))
    return tuple(folds)


def _fit_hierarchical_parent_state_prior(
    state_log_evidence: np.ndarray,
    base_probabilities: Sequence[float],
    strength: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a weak Dirichlet mixture and return leave-one-child-out priors.

    A child's own responsibility is subtracted before its predictive state
    prior is formed. This prevents the evidence being evaluated from directly
    setting its own prior while allowing the cohort to share information about
    the prevalence of zero-, one-, and two-observed-parent states.
    """
    evidence = np.asarray(state_log_evidence, dtype=np.float64)
    base = np.asarray(base_probabilities, dtype=np.float64)
    alpha = strength * base
    active = np.any(np.isfinite(evidence), axis=1)
    responsibilities = np.zeros_like(evidence)
    mixture = base.copy()
    for _ in range(max_iterations):
        previous = mixture.copy()
        log_mixture = np.log(mixture)
        for child in np.flatnonzero(active):
            responsibilities[child] = _softmax_finite(
                evidence[child] + log_mixture
            )
        posterior = alpha + np.sum(responsibilities[active], axis=0)
        mixture = posterior / np.sum(posterior)
        if float(np.max(np.abs(mixture - previous))) <= tolerance:
            break

    posterior = alpha + np.sum(responsibilities[active], axis=0)
    loo_priors = np.tile(base, (len(evidence), 1))
    for child in np.flatnonzero(active):
        parameters = posterior - responsibilities[child]
        loo_priors[child] = parameters / np.sum(parameters)
    return posterior, loo_priors


def _parent_state_score_components(
    aggregate_log_likelihoods: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
    child_state_priors: np.ndarray,
    state_log_evidence: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build marginal state and conditional-identity decision evidence."""
    n_samples = len(by_child)
    if state_log_evidence is None:
        state_log_evidence = _integrated_parent_state_log_evidence(
            aggregate_log_likelihoods, states, by_child, full_counts
        )
    priors = np.asarray(child_state_priors, dtype=np.float64)
    if priors.shape == (3,):
        priors = np.tile(priors, (n_samples, 1))
    if (
        priors.shape != (n_samples, 3)
        or np.any(~np.isfinite(priors))
        or np.any(priors <= 0.0)
    ):
        raise pedigree_models.PedigreeEvidenceError(
            "child parent-state priors must be positive with shape (samples, 3)"
        )
    state_scores = state_log_evidence + np.log(priors)
    state_support = np.zeros((n_samples, 3), dtype=np.float64)
    decision_scores = np.full(
        len(aggregate_log_likelihoods), -np.inf, dtype=np.float64
    )
    for child, child_rows in enumerate(by_child):
        for state in range(3):
            rows = child_rows[states[child_rows] == state]
            if not len(rows) or not np.isfinite(state_scores[child, state]):
                continue
            values = aggregate_log_likelihoods[rows]
            maximum = float(np.max(values))
            decision_scores[rows] = (
                state_scores[child, state] + values - maximum
            )
        state_support[child] = _softmax_finite(state_scores[child])
    return state_log_evidence, state_scores, state_support, decision_scores


def _local_parent_state_winners(
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    state_scores: np.ndarray,
    decision_scores: np.ndarray,
) -> tuple[
    dict[int, int],
    dict[int, int],
    np.ndarray,
    np.ndarray,
    tuple[Optional[str], ...],
]:
    """Choose the marginal state first, then an identity within that state."""
    n_samples = len(by_child)
    local_states = {}
    local_rows = {}
    state_margins = np.full(n_samples, np.nan, dtype=np.float64)
    identity_margins = np.full(n_samples, np.nan, dtype=np.float64)
    reasons: list[Optional[str]] = [None] * n_samples
    state_indices = np.arange(3, dtype=np.int64)
    for child, rows in enumerate(by_child):
        state, state_margin = _unique_finite_winner(
            state_indices, state_scores[child]
        )
        state_margins[child] = state_margin
        if state is None:
            reasons[child] = "no_unique_marginal_parent_state"
            continue
        state = int(state)
        local_states[child] = state
        state_rows = rows[states[rows] == state]
        row, identity_margin = _unique_finite_winner(
            state_rows, decision_scores
        )
        identity_margins[child] = identity_margin
        if row is None:
            reasons[child] = "parent_state_resolved_identity_unresolved"
            continue
        local_rows[child] = int(row)
    return (
        local_states,
        local_rows,
        state_margins,
        identity_margins,
        tuple(reasons),
    )


def _evaluate_parent_state_aggregate(
    aggregate_log_likelihoods: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
    base_priors: Sequence[float],
    prior_strength: float,
    prior_max_iterations: int,
    prior_tolerance: float,
    n_samples: int,
    local_search_passes: int,
    ancestry_depth_model: Optional[pedigree_direction._AncestryDepthModel] = None,
    *,
    state_log_evidence_override: Optional[np.ndarray] = None,
    use_cohort_prior: bool = True,
    algorithm_mode: str = "b0",
    m1_over_m0_edge_gains: Optional[np.ndarray] = None,
    m2_over_first_m1_edge_gains: Optional[np.ndarray] = None,
    m2_over_second_m1_edge_gains: Optional[np.ndarray] = None,
    predictive_fold_count: int = 0,
    graph_downward_fallback: bool = False,
    identity_log_likelihoods_override: Optional[np.ndarray] = None,
    use_fixed_base_priors: bool = False,
) -> _ParentStateSelection:
    state_log_evidence = (
        _integrated_parent_state_log_evidence(
            aggregate_log_likelihoods, states, by_child, full_counts
        )
        if state_log_evidence_override is None
        else np.asarray(state_log_evidence_override, dtype=np.float64).copy()
    )
    if use_cohort_prior:
        fitted_parameters, loo_state_priors = (
            _fit_hierarchical_parent_state_prior(
                state_log_evidence,
                base_priors,
                prior_strength,
                prior_max_iterations,
                prior_tolerance,
            )
        )
    else:
        fitted_parameters = np.full(3, np.nan, dtype=np.float64)
        loo_state_priors = (
            np.tile(np.asarray(base_priors, dtype=np.float64), (n_samples, 1))
            if use_fixed_base_priors
            else np.full((n_samples, 3), 1.0 / 3.0)
        )
    components = _parent_state_score_components(
        aggregate_log_likelihoods,
        states,
        by_child,
        full_counts,
        loo_state_priors,
        state_log_evidence,
    )
    (
        state_log_evidence,
        state_scores,
        state_support,
        decision_scores,
    ) = components
    if identity_log_likelihoods_override is not None:
        identity_values = np.asarray(
            identity_log_likelihoods_override, dtype=np.float64
        )
        if identity_values.shape != aggregate_log_likelihoods.shape:
            raise pedigree_models.PedigreeEvidenceError(
                "identity likelihood override must match aggregate rows"
            )
        decision_scores = np.full(len(identity_values), -np.inf, dtype=np.float64)
        for child, child_rows in enumerate(by_child):
            for state in range(3):
                rows = child_rows[states[child_rows] == state]
                finite = rows[np.isfinite(identity_values[rows])]
                if not len(finite) or not np.isfinite(state_scores[child, state]):
                    continue
                maximum = float(np.max(identity_values[finite]))
                decision_scores[finite] = (
                    state_scores[child, state]
                    + identity_values[finite]
                    - maximum
                )
    n_alternatives = len(alternatives)
    nan_edges = np.full(n_alternatives, np.nan, dtype=np.float64)
    m1_edges = (
        nan_edges.copy()
        if m1_over_m0_edge_gains is None
        else np.asarray(m1_over_m0_edge_gains, dtype=np.float64).copy()
    )
    m2_first_edges = (
        nan_edges.copy()
        if m2_over_first_m1_edge_gains is None
        else np.asarray(
            m2_over_first_m1_edge_gains, dtype=np.float64
        ).copy()
    )
    m2_second_edges = (
        nan_edges.copy()
        if m2_over_second_m1_edge_gains is None
        else np.asarray(
            m2_over_second_m1_edge_gains, dtype=np.float64
        ).copy()
    )
    if algorithm_mode == "b3":
        # The aggregate gate is identity-only. B3 state evidence/support was
        # already computed with fold-specific masks learned exclusively from
        # training chromosomes; full-data gains must not alter those scores.
        for child, child_rows in enumerate(by_child):
            for row in child_rows:
                row = int(row)
                state = int(states[row])
                if state == pedigree_models._ONE_OBSERVED:
                    gain = m1_edges[row]
                    eligible = bool(
                        np.isfinite(gain)
                        and gain > _contrast_tolerance(
                            np.asarray((gain, 0.0), dtype=np.float64)
                        )
                    )
                elif state == pedigree_models._TWO_OBSERVED:
                    gains = np.asarray((
                        m2_first_edges[row], m2_second_edges[row], 0.0
                    ), dtype=np.float64)
                    finite = gains[np.isfinite(gains)]
                    tolerance = _contrast_tolerance(finite)
                    eligible = bool(
                        np.isfinite(m2_first_edges[row])
                        and np.isfinite(m2_second_edges[row])
                        and m2_first_edges[row] > tolerance
                        and m2_second_edges[row] > tolerance
                    )
                else:
                    eligible = True
                if not eligible:
                    decision_scores[row] = -np.inf
    (
        local_states,
        local_rows,
        state_margins,
        identity_margins,
        unresolved_reasons,
    ) = _local_parent_state_winners(
        states, by_child, state_scores, decision_scores
    )
    local_row_vector = np.full(n_samples, -1, dtype=np.int64)
    for child, row in local_rows.items():
        local_row_vector[child] = row
    depth_posterior = (
        None
        if ancestry_depth_model is None
        else ancestry_depth_model.posterior
    )
    if pedigree_bootstrap.is_acyclic_parent_rows(local_row_vector, alternatives):
        graph_tie_conflicts = frozenset()
        graph_selection = pedigree_graph._acyclic_local_parent_state_selection(
            alternatives,
            local_rows,
            state_margins,
            identity_margins,
            depth_posterior,
        )
    else:
        # Cyclic local calls retain the exact tie peeling, deterministic
        # fallback, local search, and direction diagnostics used previously.
        graph_tie_conflicts = pedigree_graph._graph_tie_conflict_children(
            alternatives,
            local_rows,
            state_margins,
            identity_margins,
            n_samples,
        )
        graph_eligible_rows = {
            child: row
            for child, row in local_rows.items()
            if child not in graph_tie_conflicts
        }
        graph_selection = pedigree_graph._acyclic_parent_state_selection(
            alternatives,
            states,
            decision_scores,
            by_child,
            graph_eligible_rows,
            state_margins,
            identity_margins,
            n_samples,
            local_search_passes,
            depth_posterior=depth_posterior,
            downward_fallback=graph_downward_fallback,
        )
    return _ParentStateSelection(
        state_log_evidence=state_log_evidence,
        state_scores=state_scores,
        state_support=state_support,
        decision_scores=decision_scores,
        fitted_prior_parameters=fitted_parameters,
        loo_state_priors=loo_state_priors,
        local_states=local_states,
        local_rows=local_rows,
        graph_rows=graph_selection.rows,
        graph_tie_conflicts=graph_tie_conflicts,
        graph_direction_resolved_children=(
            graph_selection.direction_resolved_children
        ),
        graph_parent_role_probabilities=(
            graph_selection.selected_parent_role_probabilities
        ),
        ancestry_depth_model=ancestry_depth_model,
        state_margins=state_margins,
        identity_margins=identity_margins,
        unresolved_reasons=unresolved_reasons,
        m1_over_m0_edge_gains=m1_edges,
        m2_over_first_m1_edge_gains=m2_first_edges,
        m2_over_second_m1_edge_gains=m2_second_edges,
        predictive_fold_count=int(predictive_fold_count),
    )


def _structure_pair_indices(
    alternatives: np.ndarray,
    states: np.ndarray,
    trios: np.ndarray,
) -> np.ndarray:
    """Map M2 alternative rows to the fixed-trio structure-count rows."""
    indices = np.full(len(alternatives), -1, dtype=np.int64)
    m2_rows = np.flatnonzero(states == pedigree_models._TWO_OBSERVED)
    if not len(m2_rows):
        return indices
    n_samples = int(max(np.max(alternatives), np.max(trios))) + 1
    trio_keys = (trios[:, 0] * n_samples + trios[:, 1]) * n_samples + trios[:, 2]
    order = np.argsort(trio_keys, kind="stable")
    sorted_keys = trio_keys[order]
    selected = alternatives[m2_rows]
    selected_keys = (selected[:, 0] * n_samples + selected[:, 1]) * n_samples + selected[:, 2]
    locations = np.searchsorted(sorted_keys, selected_keys)
    if np.any(locations >= len(sorted_keys)):
        raise pedigree_models.PedigreeEvidenceError("M2 alternative is absent from the structure panel")
    if np.any(sorted_keys[locations] != selected_keys):
        raise pedigree_models.PedigreeEvidenceError("M2 alternative is absent from the structure panel")
    indices[m2_rows] = order[locations]
    return indices


def _resolved_structure_cx_veto_policy(
    settings: module_pedigree_config.PedigreeConfig,
) -> str:
    """Resolve the sensitivity policy without changing source defaults."""

    return "diagnostic_only_independent_source_posterior_collision_v1"


def _edge_direction_state_compatibility(
    direction: np.ndarray,
    depth_posterior: Optional[np.ndarray],
    explicit_direction: np.ndarray,
    settings: module_pedigree_config.PedigreeConfig,
) -> np.ndarray:
    """Return edges allowed to contribute non-M0 parent-count evidence.

    Direction is testable only when a multi-layer depth model supplies
    posterior mass for both samples. Missing, single-layer, or otherwise
    neutral direction evidence therefore leaves genetic state evidence
    unchanged. Caller-supplied direction support overrides either gate.
    """
    values = np.asarray(direction, dtype=np.float64)
    compatible = np.ones(values.shape, dtype=np.bool_)
    policy = settings.parent_state_direction_state_policy
    if policy == "identity_only":
        return compatible
    if depth_posterior is None or depth_posterior.shape[1] < 2:
        return compatible

    posterior_mass = np.sum(depth_posterior, axis=1)
    observed = np.isfinite(posterior_mass) & (posterior_mass > 0.0)
    testable = observed[:, None] & observed[None, :]
    threshold = float(settings.parent_state_minimum_direction_probability)
    if policy == "strict_gate":
        contradicted = testable & (values < threshold)
    elif policy == "reverse_contradiction_gate":
        contradicted = (
            testable
            & (values < threshold)
            & (values.T >= 1.0 - threshold)
        )
    else:  # validated configuration makes this unreachable.
        raise pedigree_models.PedigreeEvidenceError(
            f"unsupported direction-state policy {policy!r}"
        )
    contradicted &= ~np.asarray(explicit_direction, dtype=np.bool_)
    compatible[contradicted] = False
    return compatible


def _parent_state_structure_mask(
    weights: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    pair_indices: np.ndarray,
    edge_matched_by_contig: np.ndarray,
    edge_exposed_by_contig: np.ndarray,
    pair_explained_by_contig: np.ndarray,
    pair_exposed_by_contig: np.ndarray,
    structure_total_bins_by_contig: np.ndarray,
    depth_posterior: Optional[np.ndarray],
    settings: module_pedigree_config.PedigreeConfig,
    edge_exposure_presence_words: Optional[np.ndarray] = None,
    pair_exposure_presence_words: Optional[np.ndarray] = None,
    direction_supported_parents: Optional[np.ndarray] = None,
    scaffold_descendant_veto: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, ...]:
    """Separate C/X diagnostics, direction-aware state, and identity gates.

    The returned third mask is the complete parent-count state gate.
    """
    selected_contigs = np.asarray(weights) > 0.0
    edge_matched = np.tensordot(weights, edge_matched_by_contig, axes=(0, 0))
    edge_exposed = np.tensordot(weights, edge_exposed_by_contig, axes=(0, 0))
    pair_explained = weights @ pair_explained_by_contig
    pair_exposed = weights @ pair_exposed_by_contig
    total_bins = float(np.dot(weights, structure_total_bins_by_contig))
    edge_coverage = np.divide(
        edge_matched,
        edge_exposed,
        out=np.full_like(edge_matched, np.nan),
        where=edge_exposed > 0.0,
    )
    pair_explainability = np.divide(
        pair_explained,
        pair_exposed,
        out=np.full_like(pair_explained, np.nan),
        where=pair_exposed > 0.0,
    )
    edge_exposed_fraction = np.divide(
        edge_exposed,
        total_bins,
        out=np.zeros_like(edge_exposed),
        where=total_bins > 0.0,
    )
    pair_exposed_fraction = np.divide(
        pair_exposed,
        total_bins,
        out=np.zeros_like(pair_exposed),
        where=total_bins > 0.0,
    )
    edge_exposed_contigs = (
        np.count_nonzero(
            (
                (edge_exposed_by_contig > 0.0)
                & (edge_exposed_by_contig >= (
                    settings.parent_state_minimum_edge_exposed_bins
                ))
            )
            & selected_contigs[:, None, None],
            axis=0,
        )
        if edge_exposure_presence_words is None
        else pedigree_bootstrap.count_exposed_contigs(edge_exposure_presence_words, weights)
    )
    pair_exposed_contigs = (
        np.count_nonzero(
            (
                (pair_exposed_by_contig > 0.0)
                & (pair_exposed_by_contig >= (
                    settings.parent_state_minimum_pair_exposed_bins
                ))
            ) & selected_contigs[:, None],
            axis=0,
        )
        if pair_exposure_presence_words is None
        else pedigree_bootstrap.count_exposed_contigs(pair_exposure_presence_words, weights)
    )
    edge_exposure_ok = (
        (edge_exposed >= settings.parent_state_minimum_edge_exposed_bins)
        & (edge_exposed_fraction >= settings.parent_state_minimum_exposed_fraction)
        & (edge_exposed_contigs >= settings.parent_state_minimum_exposed_contigs)
    )
    pair_exposure_ok = (
        (pair_exposed >= settings.parent_state_minimum_pair_exposed_bins)
        & (pair_exposed_fraction >= settings.parent_state_minimum_exposed_fraction)
        & (pair_exposed_contigs >= settings.parent_state_minimum_exposed_contigs)
    )

    n_samples = edge_coverage.shape[0]
    direction = np.zeros((n_samples, n_samples), dtype=np.float64)
    depth_available = bool(
        depth_posterior is not None and depth_posterior.shape[1] >= 2
    )
    if depth_available:
        lower_depth_probability = (
            np.cumsum(depth_posterior, axis=1) - depth_posterior
        )
        direction = np.clip(
            depth_posterior @ lower_depth_probability.T, 0.0, 1.0
        )
    explicit_direction = (
        np.zeros((n_samples, n_samples), dtype=np.bool_)
        if direction_supported_parents is None
        else np.asarray(direction_supported_parents, dtype=np.bool_)
    )
    direction_supported = explicit_direction | (
        direction >= settings.parent_state_minimum_direction_probability
    )

    m0_rows = np.flatnonzero(states == pedigree_models._ZERO_OBSERVED)
    m1_rows = np.flatnonzero(states == pedigree_models._ONE_OBSERVED)
    m2_rows = np.flatnonzero(states == pedigree_models._TWO_OBSERVED)
    m1_children = alternatives[m1_rows, 0]
    m1_parents = alternatives[m1_rows, 1]
    m2_children = alternatives[m2_rows, 0]
    m2_first = alternatives[m2_rows, 1]
    m2_second = alternatives[m2_rows, 2]
    m2_pairs = pair_indices[m2_rows]

    exposure_testable = np.zeros(len(alternatives), dtype=np.bool_)
    exposure_testable[m0_rows] = True
    exposure_testable[m1_rows] = edge_exposure_ok[m1_children, m1_parents]
    exposure_testable[m2_rows] = (
        edge_exposure_ok[m2_children, m2_first]
        & edge_exposure_ok[m2_children, m2_second]
        & (m2_pairs >= 0)
        & pair_exposure_ok[m2_pairs]
    )
    child_evaluable = np.zeros(n_samples, dtype=np.bool_)
    np.logical_or.at(
        child_evaluable,
        m1_children,
        exposure_testable[m1_rows],
    )

    cx_compatible = np.zeros(len(alternatives), dtype=np.bool_)
    cx_compatible[m0_rows] = True
    cx_compatible[m1_rows] = (
        exposure_testable[m1_rows]
        & (
            edge_coverage[m1_children, m1_parents]
            >= settings.parent_state_minimum_edge_coverage
        )
    )
    cx_compatible[m2_rows] = (
        exposure_testable[m2_rows]
        & (
            edge_coverage[m2_children, m2_first]
            >= settings.parent_state_minimum_edge_coverage
        )
        & (
            edge_coverage[m2_children, m2_second]
            >= settings.parent_state_minimum_edge_coverage
        )
        & (
            pair_explainability[m2_pairs]
            >= settings.parent_state_minimum_pair_explainability
        )
    )
    selection_compatible = (
        cx_compatible.copy()
        if _resolved_structure_cx_veto_policy(settings) == "hard_gate_v1"
        else exposure_testable.copy()
    )
    direction_state_compatible = _edge_direction_state_compatibility(
        direction,
        depth_posterior,
        explicit_direction,
        settings,
    )
    state_compatible = selection_compatible.copy()
    state_compatible[m1_rows] &= direction_state_compatible[
        m1_children, m1_parents
    ]
    state_compatible[m2_rows] &= (
        direction_state_compatible[m2_children, m2_first]
        & direction_state_compatible[m2_children, m2_second]
    )
    identity_eligible = selection_compatible.copy()
    identity_eligible[m1_rows] &= direction_supported[
        m1_children, m1_parents
    ]
    identity_eligible[m2_rows] &= (
        direction_supported[m2_children, m2_first]
        & direction_supported[m2_children, m2_second]
    )

    return (
        exposure_testable,
        cx_compatible,
        state_compatible,
        identity_eligible,
        child_evaluable,
        edge_coverage,
        pair_explainability,
        direction,
        direction_supported,
    )


def _structure_state_and_identity_aggregates(
    raw_aggregate: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    exposure_testable: np.ndarray,
    selection_compatible: np.ndarray,
    identity_eligible: Optional[np.ndarray] = None,
    *,
    direction_available: Optional[bool] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build separate state-marginal and selectable-identity score rows.

    Exposure-untestable alternatives contribute the child's M0 likelihood to
    the state mean but remain impossible identities. Effective C/X or selected
    direction-policy contradictions contribute no state mass. The stricter
    identity direction gate independently controls which parent can be named.
    """
    raw = np.asarray(raw_aggregate, dtype=np.float64)
    state_scores = raw.copy()
    identity_scores = raw.copy()
    nonzero = states != pedigree_models._ZERO_OBSERVED
    testable = np.asarray(exposure_testable, dtype=np.bool_)
    compatible = np.asarray(selection_compatible, dtype=np.bool_)
    identity = (
        compatible
        if identity_eligible is None
        else np.asarray(identity_eligible, dtype=np.bool_)
    )
    m0_rows = np.flatnonzero(states == pedigree_models._ZERO_OBSERVED)
    represented_children = np.unique(alternatives[:, 0])
    n_samples = int(represented_children[-1]) + 1
    m0_children = alternatives[m0_rows, 0]
    m0_counts = np.bincount(m0_children, minlength=n_samples)
    if np.any(m0_counts[represented_children] != 1):
        raise pedigree_models.PedigreeEvidenceError(
            "every structurally evaluated child requires exactly one M0 row"
        )
    m0_by_child = np.full(n_samples, -1, dtype=np.int64)
    m0_by_child[m0_children] = m0_rows
    underexposed = nonzero & ~testable
    contradicted = nonzero & testable & ~compatible
    children = alternatives[underexposed, 0]
    state_scores[underexposed] = raw[m0_by_child[children]]
    state_scores[contradicted] = -np.inf
    identity_scores[nonzero & ~identity] = -np.inf
    return state_scores, identity_scores


def _prepare_parent_state_weighted_contigs(
    contig_log_likelihoods: np.ndarray,
    contig_weights: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    settings: module_pedigree_config.PedigreeConfig,
    ancestry_depth_model: pedigree_direction._AncestryDepthModel,
    structure_pair_indices: Optional[np.ndarray],
    edge_matched_by_contig: Optional[np.ndarray],
    edge_exposed_by_contig: Optional[np.ndarray],
    pair_explained_by_contig: Optional[np.ndarray],
    pair_exposed_by_contig: Optional[np.ndarray],
    structure_total_bins_by_contig: Optional[np.ndarray],
    edge_exposure_presence_words: Optional[np.ndarray],
    pair_exposure_presence_words: Optional[np.ndarray],
    direction_supported_parents: Optional[np.ndarray] = None,
    scaffold_prepared: Any = None,
    contig_information_weights: Optional[np.ndarray] = None,
    scaffold_descendant_veto: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare likelihood and exposure/C/X/direction state and identity rows."""
    weights = np.asarray(contig_weights, dtype=np.float64)

    aggregate = weights @ contig_log_likelihoods
    aggregate = _apply_aggregate_parent_state_contamination(
        aggregate,
        alternatives,
        states,
        settings.parent_state_contamination_probability,
    )
    structure_values = (
        structure_pair_indices,
        edge_matched_by_contig,
        edge_exposed_by_contig,
        pair_explained_by_contig,
        pair_exposed_by_contig,
        structure_total_bins_by_contig,
    )
    if any(value is None for value in structure_values):
        raise pedigree_models.PedigreeEvidenceError(
            "combined_v1 requires per-contig parenthood structure evidence"
        )
    (
        exposure_testable,
        _,
        selection_compatible,
        identity_eligible,
        _,
        _,
        _,
        _,
        _,
    ) = _parent_state_structure_mask(
        weights,
        alternatives,
        states,
        structure_pair_indices,
        edge_matched_by_contig,
        edge_exposed_by_contig,
        pair_explained_by_contig,
        pair_exposed_by_contig,
        structure_total_bins_by_contig,
        ancestry_depth_model.posterior,
        settings,
        direction_supported_parents=direction_supported_parents,
        edge_exposure_presence_words=edge_exposure_presence_words,
        pair_exposure_presence_words=pair_exposure_presence_words,
        scaffold_descendant_veto=scaffold_descendant_veto,
    )
    return _structure_state_and_identity_aggregates(
        aggregate,
        alternatives,
        states,
        exposure_testable,
        selection_compatible,
        identity_eligible,
    )


def _evaluate_parent_state_weighted_contigs(
    contig_log_likelihoods: np.ndarray,
    contig_weights: np.ndarray,
    contig_information_weights: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
    settings: module_pedigree_config.PedigreeConfig,
    n_samples: int,
    ancestry_depth_model: Optional[pedigree_direction._AncestryDepthModel] = None,
    base_priors: Optional[Sequence[float]] = None,
    structure_pair_indices: Optional[np.ndarray] = None,
    edge_matched_by_contig: Optional[np.ndarray] = None,
    edge_exposed_by_contig: Optional[np.ndarray] = None,
    pair_explained_by_contig: Optional[np.ndarray] = None,
    pair_exposed_by_contig: Optional[np.ndarray] = None,
    structure_total_bins_by_contig: Optional[np.ndarray] = None,
    edge_exposure_presence_words: Optional[np.ndarray] = None,
    pair_exposure_presence_words: Optional[np.ndarray] = None,
    direction_supported_parents: Optional[np.ndarray] = None,
    prepared_aggregates: Optional[tuple[np.ndarray, np.ndarray]] = None,
    scaffold_prepared: Any = None,
) -> _ParentStateSelection:
    """Evaluate the combined method using internal B1 likelihood evidence."""
    if ancestry_depth_model is None:
        raise pedigree_models.PedigreeEvidenceError(
            "combined_v1 requires an ancestry-depth model in every evaluation"
        )
    preparation_settings = settings

    if prepared_aggregates is None:
        aggregate, identity_aggregate = (
            _prepare_parent_state_weighted_contigs(
                contig_log_likelihoods,
                contig_weights,
                alternatives,
                states,
                preparation_settings,
                ancestry_depth_model,
                structure_pair_indices,
                edge_matched_by_contig,
                edge_exposed_by_contig,
                pair_explained_by_contig,
                pair_exposed_by_contig,
                structure_total_bins_by_contig,
                edge_exposure_presence_words,
                pair_exposure_presence_words,
                direction_supported_parents,
                scaffold_prepared,
                contig_information_weights,
            )
        )
    else:
        aggregate, identity_aggregate = prepared_aggregates
    effective_priors = (
        settings.parent_state_priors if base_priors is None else base_priors
    )
    selection = _evaluate_parent_state_aggregate(
        aggregate,
        alternatives,
        states,
        by_child,
        full_counts,
        effective_priors,
        settings.parent_state_prior_strength,
        settings.parent_state_prior_max_iterations,
        settings.parent_state_prior_tolerance,
        n_samples,
        settings.dag_local_search_passes,
        ancestry_depth_model,
        use_cohort_prior=False,
        algorithm_mode=pedigree_models._PARENT_STATE_LIKELIHOOD,
        use_fixed_base_priors=True,
        graph_downward_fallback=True,
        identity_log_likelihoods_override=identity_aggregate,
    )

    return selection


def _parent_state_frame(
    sample_ids: Sequence[Any],
    alternatives: np.ndarray,
    states: np.ndarray,
    rows_by_child: Mapping[int, Optional[int]],
    state_by_child: Mapping[int, int],
    status_by_child: Mapping[int, str],
) -> pd.DataFrame:
    rows = []
    for child, sample in enumerate(sample_ids):
        selected = rows_by_child.get(child)
        state = state_by_child.get(child)
        first = second = None
        if selected is not None:
            first_index = int(alternatives[selected, 1])
            second_index = int(alternatives[selected, 2])
            first = None if first_index < 0 else sample_ids[first_index]
            second = None if second_index < 0 else sample_ids[second_index]
        rows.append({
            "Sample": sample,
            "Generation": "Unknown",
            "Parent1": first,
            "Parent2": second,
            "ParentState": (
                "unresolved" if state is None else pedigree_models._PARENT_STATE_NAMES[state]
            ),
            "ObservedParentCount": (
                np.nan if state is None else int(state)
            ),
            "InferenceStatus": status_by_child.get(child, "unresolved"),
        })
    return pd.DataFrame(rows)


def _configuration_support_text(
    child_rows: np.ndarray,
    counts: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    sample_ids: Sequence[Any],
    coverage: float,
) -> str:
    total = float(np.sum(counts[child_rows]))
    if total <= 0.0:
        return ""
    ordered = child_rows[np.argsort(-counts[child_rows], kind="stable")]
    pieces = []
    cumulative = 0.0
    for row in ordered:
        probability = float(counts[row] / total)
        state = int(states[row])
        parents = [
            str(sample_ids[int(parent)])
            for parent in alternatives[row, 1:]
            if int(parent) >= 0
        ]
        label = "+".join(parents) if parents else "external+external"
        pieces.append(
            f"{pedigree_models._PARENT_STATE_NAMES[state]}:{label}:{probability:.3f}"
        )
        cumulative += probability
        if cumulative >= coverage:
            break
    return ";".join(pieces)


def _coverage_rows_from_scores(
    rows: np.ndarray,
    scores: np.ndarray,
    coverage: float,
) -> tuple[int, ...]:
    """Small deterministic configuration set carrying requested score mass."""
    candidate_rows = np.asarray(rows, dtype=np.int64)
    probabilities = _softmax_finite(scores[candidate_rows])
    positive = probabilities > 0.0
    candidate_rows = candidate_rows[positive]
    probabilities = probabilities[positive]
    if not len(candidate_rows):
        return ()
    order = np.lexsort((candidate_rows, -probabilities))
    selected = []
    cumulative = 0.0
    for position in order:
        selected.append(int(candidate_rows[position]))
        cumulative += float(probabilities[position])
        if cumulative >= coverage:
            break
    return tuple(selected)


def _evidence_parent_support_sets(
    child_rows: np.ndarray,
    selected_state: Optional[int],
    selected_row: Optional[int],
    alternatives: np.ndarray,
    states: np.ndarray,
    decision_scores: np.ndarray,
    coverage: float,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Factor a high-evidence M1/M2 configuration set into parent sides.

    M2 parents are unordered. If all supported pairs share one parent, that
    singleton becomes the fixed side and the other side is the union of its
    alternatives. Otherwise, sides are defined conditionally by holding each
    parent of the unique leading pair fixed. The final tuple contains the
    supported configuration rows, preserving non-factorable ambiguity.
    """
    if selected_state not in {pedigree_models._ONE_OBSERVED, pedigree_models._TWO_OBSERVED}:
        return (), (), ()
    state_rows = child_rows[states[child_rows] == selected_state]
    support_rows = _coverage_rows_from_scores(
        state_rows, decision_scores, coverage
    )
    if not support_rows:
        return (), (), ()
    if selected_state == pedigree_models._ONE_OBSERVED:
        parents = tuple(dict.fromkeys(
            int(alternatives[row, 1]) for row in support_rows
        ))
        return parents, (), support_rows

    pairs = [
        frozenset((
            int(alternatives[row, 1]), int(alternatives[row, 2])
        ))
        for row in support_rows
    ]
    if len(pairs) == 1:
        row = support_rows[0]
        return (
            (int(alternatives[row, 1]),),
            (int(alternatives[row, 2]),),
            support_rows,
        )
    common = set(pairs[0])
    for pair in pairs[1:]:
        common.intersection_update(pair)
    if len(common) == 1:
        fixed = next(iter(common))
        variable = tuple(sorted({
            parent for pair in pairs for parent in pair if parent != fixed
        }))
        if selected_row is not None and int(alternatives[selected_row, 1]) == fixed:
            return (fixed,), variable, support_rows
        return variable, (fixed,), support_rows
    return (), (), support_rows


def _sample_set_text(indices: Sequence[int], sample_ids: Sequence[Any]) -> str:
    return "{" + ",".join(str(sample_ids[index]) for index in indices) + "}"

import haplotype_reconstruction.pedigree.bootstrap as pedigree_bootstrap
import haplotype_reconstruction.pedigree.config as module_pedigree_config
import haplotype_reconstruction.pedigree.eligibility as pedigree_eligibility
import haplotype_reconstruction.pedigree.graph as pedigree_graph

