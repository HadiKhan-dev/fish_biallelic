"""Aggregate parent-state likelihoods and structure evidence across chromosomes."""
from __future__ import annotations


import math

import time
from typing import Any, Callable, Mapping, Sequence
import numpy as np
import haplotype_reconstruction.pedigree.models as pedigree_models

def _sum_optional(values: list[np.ndarray | None]) -> np.ndarray | None:
    if not values or any(value is None for value in values):
        return None
    return np.sum(np.stack(values), axis=0)


def _validated_trios(trios: Any) -> np.ndarray:
    values = np.asarray(trios)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or np.any(~np.isfinite(values))
        or np.any(values != np.floor(values))
    ):
        raise pedigree_models.PedigreeEvidenceError("trios must be an integer array of shape (rows, 3)")
    return np.ascontiguousarray(values, dtype=np.int64)


def _hard_structure_only(
        component: pedigree_components.PreparedComponentPedigree,
        trio_array: np.ndarray,
        eligible_parent_mask: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, float,
]:
    """Reproduce hard-path ancestry structure without likelihood scoring."""

    cache = component.cache
    founders = np.asarray(cache.founder_alleles, dtype=np.int8).copy()
    marker_counts = np.asarray(
        cache.selected_markers_per_bin, dtype=np.int64
    )
    for block, marker_count in enumerate(marker_counts):
        founders[:, block, int(marker_count):] = -1
    pooled_labels, _, _, _, trajectory_classes = pedigree_candidates._pool_local_ibs_states(
        cache.stacked_labels, founders
    )
    required_edges = np.ascontiguousarray(
        eligible_parent_mask | eligible_parent_mask.T
    )
    np.fill_diagonal(required_edges, True)
    edge_matched, edge_exposed, pair_explained, pair_exposed = (
        pedigree_candidates._parenthood_structure_count_kernel(
            pooled_labels, trio_array, required_edges
        )
    )
    junctions, callable_bins = pedigree_candidates._ancestry_junction_count_kernel(
        pooled_labels, trajectory_classes
    )
    return (
        junctions,
        callable_bins,
        edge_matched,
        edge_exposed,
        pair_explained,
        pair_exposed,
        float(pooled_labels.shape[1]),
    )


def _compact_component_gl(component: pedigree_components.PreparedComponentPedigree) -> np.ndarray:
    retained = getattr(component, "compact_genotype_likelihoods", None)
    if retained is not None:
        return retained
    # Compatibility for compact prepared runs created before the retained
    # ragged GL field was added.
    cache = component.cache
    return np.ascontiguousarray(np.concatenate([
        cache.genotype_likelihoods[:, block,:int(count),:]
        for block, count in enumerate(cache.selected_markers_per_bin)
    ], axis=1))


def _projected_m1_screen(
        component: pedigree_components.PreparedComponentPedigree,
        exponent: np.ndarray,
        settings: module_pedigree_config.PedigreeConfig,
        eligible_child_mask: np.ndarray,
        eligible_parent_mask: np.ndarray,
) -> pedigree_components._ProjectedParentScreen:
    model = component.ragged_model
    factors = component.ragged_source_factors
    observed = component.compact_observed
    if model is None or factors is None or observed is None:
        raise pedigree_models.PedigreeEvidenceError(
            "projected T09 component lacks compact ragged source factors"
        )
    compact_gl = _compact_component_gl(component)
    source_marginal_started = time.perf_counter()
    structure_marginals = np.ascontiguousarray(
        pedigree_sources.source_posterior_marginals(factors)
    )
    source_marginal_seconds = time.perf_counter() - source_marginal_started
    structure_marginals.setflags(write=False)
    projected = pedigree_transmission.prepare_projected_ragged_quadratic(
        factors,
        model,
        compact_gl,
        observed,
        precomputed_source_marginals=structure_marginals,
        candidate_selector_switch_probability=(
            component.cache.switch_probabilities[1:]
        ),
    )
    batch = pedigree_transmission.score_projected_ragged_quadratic(
        projected,
        factors.transition,
        compact_gl,
        observed,
        exponent,
        np.empty((0, 3), dtype=np.int64),
        eligible_children=eligible_child_mask,
        eligible_parent_edges=eligible_parent_mask,
        null_selector_switch_probability=(
            component.cache.switch_probabilities[1:]
        ),
        mismatch_probability=settings.parent_state_mismatch_probability,
    )
    result = np.asarray(batch.one_observed, dtype=np.float64).copy()
    result[~eligible_parent_mask] = -math.inf
    return pedigree_components._ProjectedParentScreen(
        result, batch, projected, structure_marginals,
        source_marginal_seconds,
    )


def _score_projected_component_with_diagnostics(
        component: pedigree_components.PreparedComponentPedigree,
        exponent: np.ndarray,
        trio_array: np.ndarray,
        settings: module_pedigree_config.PedigreeConfig,
        eligible_child_mask: np.ndarray,
        eligible_parent_mask: np.ndarray,
        *,
        reuse_screen: pedigree_components._ProjectedParentScreen | None=None,
) -> tuple[
    pedigree_candidates.ChromosomeLikelihoods,
    pedigree_components.ProjectedComponentScoringDiagnostics
]:
    model = component.ragged_model
    factors = component.ragged_source_factors
    observed = component.compact_observed
    if model is None or factors is None or observed is None:
        raise pedigree_models.PedigreeEvidenceError(
            "projected T09 component lacks compact ragged source factors"
        )
    compact_gl = _compact_component_gl(component)
    if reuse_screen is None:
        source_marginal_started = time.perf_counter()
        structure_marginals = np.ascontiguousarray(
            pedigree_sources.source_posterior_marginals(factors)
        )
        source_marginal_seconds = (
            time.perf_counter() - source_marginal_started
        )
        structure_marginals.setflags(write=False)
        projected = pedigree_transmission.prepare_projected_ragged_quadratic(
            factors,
            model,
            compact_gl,
            observed,
            precomputed_source_marginals=structure_marginals,
            candidate_selector_switch_probability=(
                component.cache.switch_probabilities[1:]
            ),
        )
        reuse_scores = None
    else:
        structure_marginals = reuse_screen.structure_marginals
        source_marginal_seconds = (
            reuse_screen.source_marginal_preparation_seconds
        )
        projected = reuse_screen.projected_model
        reuse_scores = reuse_screen.batch_scores
    batch = pedigree_transmission.score_projected_ragged_quadratic(
        projected,
        factors.transition,
        compact_gl,
        observed,
        exponent,
        trio_array,
        eligible_children=eligible_child_mask,
        eligible_parent_edges=eligible_parent_mask,
        null_selector_switch_probability=(
            component.cache.switch_probabilities[1:]
        ),
        mismatch_probability=settings.parent_state_mismatch_probability,
        reuse_scores=reuse_scores,
    )

    zero = np.asarray(batch.zero_observed, dtype=np.float64).copy()
    one = np.asarray(batch.one_observed, dtype=np.float64).copy()
    two = np.asarray(batch.two_observed, dtype=np.float64).copy()
    zero[~eligible_child_mask] = -math.inf
    one[~eligible_parent_mask] = -math.inf
    if len(trio_array):
        valid_trio = (
            eligible_child_mask[trio_array[:, 0]]
            & eligible_parent_mask[trio_array[:, 0], trio_array[:, 1]]
            & eligible_parent_mask[trio_array[:, 0], trio_array[:, 2]]
        )
        two[~valid_trio] = -math.inf

    hard_structure_started = time.perf_counter()
    hard_structure = _hard_structure_only(
        component, trio_array, eligible_parent_mask
    )
    hard_structure_seconds = time.perf_counter() - hard_structure_started
    junctions, callable_bins = hard_structure[:2]
    anchored_states = component.ragged_anchored_states
    if anchored_states is None:
        raise pedigree_models.PedigreeEvidenceError(
            "projected T09 component lacks its anchored-state identity mask"
        )
    required_edges = np.ascontiguousarray(
        eligible_parent_mask | eligible_parent_mask.T
    )
    np.fill_diagonal(required_edges, True)
    posterior_structure_started = time.perf_counter()
    expected_structure = pedigree_sources.posterior_expected_structure(
        structure_marginals,
        anchored_states,
        required_edges,
        trio_array,
        sample_available=factors.available,
    )
    posterior_structure_seconds = (
        time.perf_counter() - posterior_structure_started
    )
    initial_max = np.max(
        factors.initial_probability.reshape(factors.n_candidates, -1), axis=1
    )
    complete = int(np.count_nonzero(
        np.all(model.named_alleles >= 0, axis=0)
    ))
    excluded = int(model.n_sites - complete)
    score = pedigree_candidates.ChromosomeLikelihoods(
        zero_observed=zero,
        one_observed=one,
        two_observed=two,
        ancestry_junction_counts=junctions,
        ancestry_callable_haplotype_bins=callable_bins,
        candidate_source_mode_requested=(
            pedigree_models.RAGGED_QUADRATIC_MODEL
        ),
        candidate_source_mode_applied=(
            pedigree_models.RAGGED_QUADRATIC_MODEL
        ),
        candidate_source_fallback=False,
        candidate_source_fallback_reason="",
        complete_founder_marker_count=complete,
        excluded_founder_marker_count=excluded,
        candidate_source_available=batch.candidate_source_available.copy(),
        candidate_source_informative_marker_count=(
            batch.candidate_source_informative_site_count.copy()
        ),
        child_complete_informative_marker_count=(
            batch.child_informative_site_count.copy()
        ),
        candidate_initial_max_probability=initial_max,
        candidate_initial_point_mass=initial_max == 1.0,
        peak_streamed_tensor_bytes=int(
            batch.complexity.peak_working_bytes_per_task
        ),
        candidate_source_posterior=factors,
        edge_matched_bins=expected_structure.edge_matched_bins,
        edge_exposed_bins=expected_structure.edge_exposed_bins,
        pair_explained_bins=expected_structure.pair_explained_bins,
        pair_exposed_bins=expected_structure.pair_exposed_bins,
        structure_total_bins=expected_structure.structure_total_bins,
    )
    phase_batches = (batch,) if reuse_scores is None else (reuse_scores, batch)
    branch_counts = np.bincount(
        projected.bridge_branch.ravel(), minlength=4
    )
    diagnostic = pedigree_components.ProjectedComponentScoringDiagnostics(
        approximation_name=pedigree_transmission.APPROXIMATION_NAME,
        component_index=component.component_index,
        source_preparation_seconds=float(factors.preparation_seconds),
        source_marginal_preparation_seconds=float(
            source_marginal_seconds
        ),
        hard_structure_seconds=float(hard_structure_seconds),
        posterior_expected_structure_seconds=float(
            posterior_structure_seconds
        ),
        projection_preparation_seconds=float(projected.preparation_seconds),
        m0_scoring_seconds=sum(
            float(value.m0_scoring_seconds) for value in phase_batches
        ),
        m1_scoring_seconds=sum(
            float(value.m1_scoring_seconds) for value in phase_batches
        ),
        m2_scoring_seconds=float(batch.m2_scoring_seconds),
        m2_active_trio_count=int(batch.m2_active_trio_count),
        m2_reduced_trio_count=int(batch.m2_reduced_trio_count),
        projection_retained_bytes=int(batch.projection_retained_bytes),
        projected_hidden_state_count=int(
            batch.complexity.projected_hidden_state_count
        ),
        exact_m2_hidden_state_count=int(
            batch.complexity.exact_m2_hidden_state_count
        ),
        maximum_bridge_marginal_residual=float(
            batch.maximum_bridge_marginal_residual
        ),
        bridge_branch_counts=tuple(
            int(value) for value in branch_counts[:4]
        ),
        maximum_bridge_solver_iterations=(
            int(np.max(projected.sinkhorn_iterations))
            if projected.sinkhorn_iterations.size else 0
        ),
        reused_screen_scores=bool(batch.reused_lower_order_scores),
    )
    return score, diagnostic


def _score_prepared_chromosome(
        prepared: pedigree_components.PreparedChromosome,
        trio_array: np.ndarray,
        settings: module_pedigree_config.PedigreeConfig,
        eligible_child_mask: np.ndarray,
        eligible_parent_mask: np.ndarray,
        *,
        ragged_screen_scores: Sequence[
            pedigree_sources.RaggedSourceBatchScores | pedigree_components._ProjectedParentScreen
        ] | None=None,
) -> pedigree_components.ComponentPedigreeChromosomeResult:
    if not prepared.components:
        return pedigree_components.ComponentPedigreeChromosomeResult(
            prepared.contig,
            None,
            None,
            prepared.component_count,
            0,
            0,
            prepared.omitted_reason,
        )

    if any(
        component.source_mode != prepared.source_mode
        for component in prepared.components
    ):
        raise pedigree_models.PedigreeEvidenceError("prepared T09 component source modes disagree")
    if ragged_screen_scores is None:
        reuse_by_component = (None,) * len(prepared.components)
    else:
        reuse_by_component = tuple(ragged_screen_scores)
        if (
            len(reuse_by_component) != len(prepared.components)
            or any(
                not isinstance(value, pedigree_components._ProjectedParentScreen)
                for value in reuse_by_component
            )
        ):
            raise pedigree_models.PedigreeEvidenceError(
                "projected screening scores must match prepared components"
            )
    scored_components = [
        _score_projected_component_with_diagnostics(
            component,
            exponent,
            trio_array,
            settings,
            eligible_child_mask,
            eligible_parent_mask,
            reuse_screen=reuse_screen,
        )
        for component, exponent, reuse_screen in zip(
            prepared.components,
            prepared.information_exponents,
            reuse_by_component,
        )
    ]
    component_scores = [value[0] for value in scored_components]
    projected_diagnostics = tuple(
        value[1] for value in scored_components
    )
    ragged_diagnostics = ()

    zero = np.sum(np.stack([
        value.zero_observed for value in component_scores
    ]), axis=0)
    one = np.sum(np.stack([
        value.one_observed for value in component_scores
    ]), axis=0)
    two = np.sum(np.stack([
        value.two_observed for value in component_scores
    ]), axis=0)
    junctions = np.sum(np.stack([
        value.ancestry_junction_counts for value in component_scores
    ]), axis=0)
    callable_bins = np.sum(np.stack([
        value.ancestry_callable_haplotype_bins for value in component_scores
    ]), axis=0)
    edge_matched = _sum_optional([
        value.edge_matched_bins for value in component_scores
    ])
    edge_exposed = _sum_optional([
        value.edge_exposed_bins for value in component_scores
    ])
    pair_explained = _sum_optional([
        value.pair_explained_bins for value in component_scores
    ])
    pair_exposed = _sum_optional([
        value.pair_exposed_bins for value in component_scores
    ])
    total_bins = float(sum(
        component.cache.stacked_labels.shape[1]
        for component in prepared.components
    ))
    informative_markers = int(sum(
        component.cache.informative_markers for component in prepared.components
    ))
    aggregate = pedigree_candidates.ChromosomeLikelihoods(
        zero,
        one,
        two,
        junctions,
        callable_bins,
        _sum_optional([
            value.one_parent_identity_information for value in component_scores
        ]),
        _sum_optional([
            value.two_parent_edge_information for value in component_scores
        ]),
        candidate_source_mode_requested=prepared.source_mode,
        candidate_source_mode_applied=prepared.source_mode,
        candidate_source_fallback=False,
        complete_founder_marker_count=sum(
            int(value.complete_founder_marker_count or 0)
            for value in component_scores
        ),
        excluded_founder_marker_count=sum(
            int(value.excluded_founder_marker_count or 0)
            for value in component_scores
        ),
        candidate_source_available=(
            np.any(np.stack([
                value.candidate_source_available
                for value in component_scores
            ]), axis=0)
        ),
        candidate_source_informative_marker_count=(
            np.sum(np.stack([
                value.candidate_source_informative_marker_count
                for value in component_scores
            ]), axis=0)
        ),
        child_complete_informative_marker_count=(
            np.sum(np.stack([
                value.child_complete_informative_marker_count
                for value in component_scores
            ]), axis=0)
        ),
        candidate_initial_max_probability=(
            np.max(np.stack([
                value.candidate_initial_max_probability
                for value in component_scores
            ]), axis=0)
        ),
        candidate_initial_point_mass=(
            np.any(np.stack([
                value.candidate_initial_point_mass
                for value in component_scores
            ]), axis=0)
        ),
        peak_streamed_tensor_bytes=max(
            value.peak_streamed_tensor_bytes for value in component_scores
        ),
        candidate_source_posterior=(
            tuple(value.candidate_source_posterior for value in component_scores)
        ),
        edge_matched_bins=edge_matched,
        edge_exposed_bins=edge_exposed,
        pair_explained_bins=pair_explained,
        pair_exposed_bins=pair_exposed,
        structure_total_bins=total_bins,
    )
    evidence = pedigree_models.ParentStateEvidence(
        contig=prepared.contig,
        trios=trio_array,
        zero_parent_log_likelihoods=zero,
        one_parent_log_likelihoods=one,
        two_parent_log_likelihoods=two,
        informative_markers=informative_markers,
        edge_matched_bins=edge_matched,
        edge_exposed_bins=edge_exposed,
        pair_explained_bins=pair_explained,
        pair_exposed_bins=pair_exposed,
        structure_total_bins=total_bins,
    )
    return pedigree_components.ComponentPedigreeChromosomeResult(
        prepared.contig,
        evidence,
        aggregate,
        prepared.component_count,
        len(prepared.components),
        informative_markers,
        ragged_component_diagnostics=ragged_diagnostics,
        projected_component_diagnostics=projected_diagnostics,
    )


def _compact_chromosome_evidence(
        result: pedigree_components.ComponentPedigreeChromosomeResult,
) -> pedigree_components.ScoredT09ChromosomeEvidence:
    evidence = result.evidence
    scores = result.state_scores
    if evidence is None or scores is None or result.omitted_reason is not None:
        raise pedigree_models.PedigreeEvidenceError(
            "only informative chromosome results can enter the evidence cache"
        )
    required_structure = (
        evidence.edge_matched_bins,
        evidence.edge_exposed_bins,
        evidence.pair_explained_bins,
        evidence.pair_exposed_bins,
        evidence.structure_total_bins,
    )
    if any(value is None for value in required_structure):
        raise pedigree_models.PedigreeEvidenceError(
            "scored chromosomes require complete C/X structure evidence"
        )
    return pedigree_components.ScoredT09ChromosomeEvidence(
        contig=result.contig,
        zero_parent_log_likelihoods=evidence.zero_parent_log_likelihoods,
        one_parent_log_likelihoods=evidence.one_parent_log_likelihoods,
        two_parent_log_likelihoods=evidence.two_parent_log_likelihoods,
        edge_matched_bins=evidence.edge_matched_bins,
        edge_exposed_bins=evidence.edge_exposed_bins,
        pair_explained_bins=evidence.pair_explained_bins,
        pair_exposed_bins=evidence.pair_exposed_bins,
        structure_total_bins=float(evidence.structure_total_bins),
        ancestry_junction_counts=scores.ancestry_junction_counts,
        ancestry_callable_haplotype_bins=(
            scores.ancestry_callable_haplotype_bins
        ),
        informative_markers=int(result.informative_markers),
        component_count=int(result.component_count),
        scored_component_count=int(result.scored_component_count),
        one_parent_identity_information=scores.one_parent_identity_information,
        two_parent_edge_information=scores.two_parent_edge_information,
        candidate_source_mode_requested=scores.candidate_source_mode_requested,
        candidate_source_mode_applied=scores.candidate_source_mode_applied,
        candidate_source_fallback=bool(scores.candidate_source_fallback),
        candidate_source_fallback_reason=scores.candidate_source_fallback_reason,
        complete_founder_marker_count=scores.complete_founder_marker_count,
        excluded_founder_marker_count=scores.excluded_founder_marker_count,
        candidate_source_available=scores.candidate_source_available,
        candidate_source_informative_marker_count=(
            scores.candidate_source_informative_marker_count
        ),
        child_complete_informative_marker_count=(
            scores.child_complete_informative_marker_count
        ),
        candidate_initial_max_probability=(
            scores.candidate_initial_max_probability
        ),
        candidate_initial_point_mass=scores.candidate_initial_point_mass,
        peak_streamed_tensor_bytes=int(scores.peak_streamed_tensor_bytes),
        ragged_component_diagnostics=result.ragged_component_diagnostics,
        projected_component_diagnostics=result.projected_component_diagnostics,
    )


def _chromosome_result_from_scored(
        chromosome: pedigree_components.ScoredT09ChromosomeEvidence,
        trios: np.ndarray,
) -> pedigree_components.ComponentPedigreeChromosomeResult:
    evidence = pedigree_models.ParentStateEvidence(
        contig=chromosome.contig,
        trios=trios,
        zero_parent_log_likelihoods=chromosome.zero_parent_log_likelihoods,
        one_parent_log_likelihoods=chromosome.one_parent_log_likelihoods,
        two_parent_log_likelihoods=chromosome.two_parent_log_likelihoods,
        informative_markers=chromosome.informative_markers,
        edge_matched_bins=chromosome.edge_matched_bins,
        edge_exposed_bins=chromosome.edge_exposed_bins,
        pair_explained_bins=chromosome.pair_explained_bins,
        pair_exposed_bins=chromosome.pair_exposed_bins,
        structure_total_bins=chromosome.structure_total_bins,
    )
    scores = pedigree_candidates.ChromosomeLikelihoods(
        chromosome.zero_parent_log_likelihoods,
        chromosome.one_parent_log_likelihoods,
        chromosome.two_parent_log_likelihoods,
        chromosome.ancestry_junction_counts,
        chromosome.ancestry_callable_haplotype_bins,
        chromosome.one_parent_identity_information,
        chromosome.two_parent_edge_information,
        candidate_source_mode_requested=(
            chromosome.candidate_source_mode_requested
        ),
        candidate_source_mode_applied=chromosome.candidate_source_mode_applied,
        candidate_source_fallback=chromosome.candidate_source_fallback,
        candidate_source_fallback_reason=(
            chromosome.candidate_source_fallback_reason
        ),
        complete_founder_marker_count=(
            chromosome.complete_founder_marker_count
        ),
        excluded_founder_marker_count=(
            chromosome.excluded_founder_marker_count
        ),
        candidate_source_available=chromosome.candidate_source_available,
        candidate_source_informative_marker_count=(
            chromosome.candidate_source_informative_marker_count
        ),
        child_complete_informative_marker_count=(
            chromosome.child_complete_informative_marker_count
        ),
        candidate_initial_max_probability=(
            chromosome.candidate_initial_max_probability
        ),
        candidate_initial_point_mass=chromosome.candidate_initial_point_mass,
        peak_streamed_tensor_bytes=chromosome.peak_streamed_tensor_bytes,
        candidate_source_posterior=None,
        edge_matched_bins=chromosome.edge_matched_bins,
        edge_exposed_bins=chromosome.edge_exposed_bins,
        pair_explained_bins=chromosome.pair_explained_bins,
        pair_exposed_bins=chromosome.pair_exposed_bins,
        structure_total_bins=chromosome.structure_total_bins,
    )
    return pedigree_components.ComponentPedigreeChromosomeResult(
        chromosome.contig,
        evidence,
        scores,
        chromosome.component_count,
        chromosome.scored_component_count,
        chromosome.informative_markers,
        ragged_component_diagnostics=chromosome.ragged_component_diagnostics,
        projected_component_diagnostics=(
            chromosome.projected_component_diagnostics
        ),
    )


def _validate_scored_chromosome(
        value: Any,
        *,
        contig: str,
        n_samples: int,
        n_trios: int,
) -> pedigree_components.ScoredT09ChromosomeEvidence:
    if not isinstance(value, pedigree_components.ScoredT09ChromosomeEvidence):
        raise pedigree_models.PedigreeEvidenceError(
            "chromosome evidence callback must return scored T09 evidence"
        )
    if value.contig != contig:
        raise pedigree_models.PedigreeEvidenceError("cached chromosome evidence contig mismatch")
    shapes = {
        "zero-parent": (value.zero_parent_log_likelihoods, (n_samples,)),
        "one-parent": (
            value.one_parent_log_likelihoods, (n_samples, n_samples)
        ),
        "two-parent": (value.two_parent_log_likelihoods, (n_trios,)),
        "edge matched": (value.edge_matched_bins, (n_samples, n_samples)),
        "edge exposed": (value.edge_exposed_bins, (n_samples, n_samples)),
        "pair explained": (value.pair_explained_bins, (n_trios,)),
        "pair exposed": (value.pair_exposed_bins, (n_trios,)),
        "ancestry junction": (value.ancestry_junction_counts, (n_samples,)),
        "ancestry callability": (
            value.ancestry_callable_haplotype_bins, (n_samples,)
        ),
    }
    for name, (raw, expected) in shapes.items():
        if np.asarray(raw).shape != expected:
            raise pedigree_models.PedigreeEvidenceError(
                f"cached {name} evidence has shape {np.asarray(raw).shape}; "
                f"expected {expected}"
            )
    if (
        value.informative_markers < 1
        or value.component_count < value.scored_component_count
        or value.scored_component_count < 1
        or not np.isfinite(value.structure_total_bins)
        or value.structure_total_bins <= 0.0
    ):
        raise pedigree_models.PedigreeEvidenceError("cached chromosome evidence metadata is invalid")
    return value


def _panel_diagnostics(
        requested_initial_top_k: int | None,
        outer_top_k: int,
        chromosome_count: int,
        eligibility: Any,
        selected_k_by_child: Sequence[int],
        trio_row_count: int,
        *,
        applied: bool,
        fallback_reason: str | None=None,
) -> pedigree_components.AdaptiveParentPanelDiagnostics:
    selected = tuple(int(value) for value in selected_k_by_child)
    eligible_counts = [
        selected[child]
        for child in range(len(selected))
        if eligibility.eligible_children[child]
    ]
    distribution = tuple(
        (value, eligible_counts.count(value))
        for value in sorted(set(eligible_counts))
    )
    requested = requested_initial_top_k is not None
    return pedigree_components.AdaptiveParentPanelDiagnostics(
        requested_initial_top_k=requested_initial_top_k,
        outer_top_k=int(outer_top_k),
        informative_chromosome_count=int(chromosome_count),
        adaptive_requested=requested,
        adaptive_applied=bool(applied),
        adaptive_fallback=bool(requested and not applied),
        fallback_reason=fallback_reason,
        selected_k_by_child=selected,
        selected_k_distribution=distribution,
        trio_row_count=int(trio_row_count),
    )


def _adaptive_trio_panel(
        pair_score_array: np.ndarray,
        marker_count_array: np.ndarray,
        parent_screen_scores: np.ndarray,
        top_k: int,
        anchor_k: int,
        use_anchor_union: bool,
        eligibility: Any,
        settings: module_pedigree_config.PedigreeConfig,
        adaptive_initial_top_k: int | None,
) -> tuple[np.ndarray, pedigree_components.AdaptiveParentPanelDiagnostics]:
    """Build an optional LOCO-stability subset of the fixed M2 panel."""

    fixed_trios = pedigree_candidates._fixed_trio_panel(
        parent_screen_scores, top_k, anchor_k, use_anchor_union, eligibility
    )
    outer_top_k = int(top_k)
    n_samples = parent_screen_scores.shape[0]
    outer_by_child: list[list[int]] = [[] for _ in range(n_samples)]
    for child in range(n_samples):
        if not eligibility.eligible_children[child]:
            continue
        parents = np.flatnonzero(eligibility.eligible_parents[child])
        order = np.lexsort((parents, -parent_screen_scores[child, parents]))
        outer_by_child[child] = parents[order[:min(
            outer_top_k, len(parents)
        )]].astype(np.int64).tolist()
    outer_counts = tuple(len(values) for values in outer_by_child)

    if adaptive_initial_top_k is None:
        return fixed_trios, _panel_diagnostics(
            None, outer_top_k, len(pair_score_array), eligibility, outer_counts,
            len(fixed_trios), applied=False,
        )
    initial_top_k = pedigree_components._positive_integer(
        adaptive_initial_top_k, "adaptive_initial_top_k"
    )
    if initial_top_k >= outer_top_k:
        raise pedigree_models.PedigreeEvidenceError(
            "adaptive_initial_top_k must be smaller than top_k"
        )

    fallback_reason = None
    if len(pair_score_array) < 3:
        fallback_reason = "fewer_than_three_informative_chromosomes"
    elif (
        marker_count_array.shape != (len(pair_score_array),)
        or np.any(~np.isfinite(marker_count_array))
        or np.any(marker_count_array <= 0.0)
    ):
        fallback_reason = "invalid_or_uninformative_loco"

    loco_scores = []
    if fallback_reason is None:
        for omitted_index in range(len(pair_score_array)):
            keep = np.arange(len(pair_score_array)) != omitted_index
            try:
                score = pedigree_candidates._robust_parent_screen(
                    pair_score_array[keep], marker_count_array[keep],
                    settings, eligibility,
                )
            except pedigree_models.PedigreeEvidenceError:
                fallback_reason = "invalid_or_uninformative_loco"
                break
            if score.shape != parent_screen_scores.shape:
                fallback_reason = "invalid_or_uninformative_loco"
                break
            invalid = False
            for child in range(n_samples):
                parents = np.flatnonzero(eligibility.eligible_parents[child])
                if (
                    eligibility.eligible_children[child]
                    and len(parents)
                    and np.any(~np.isfinite(score[child, parents]))
                ):
                    invalid = True
                    break
            if invalid:
                fallback_reason = "invalid_or_uninformative_loco"
                break
            loco_scores.append(score)

    if fallback_reason is not None:
        return fixed_trios, _panel_diagnostics(
            initial_top_k, outer_top_k, len(pair_score_array), eligibility,
            outer_counts, len(fixed_trios), applied=False,
            fallback_reason=fallback_reason,
        )

    rows = []
    selected_counts = [0] * n_samples
    for child, outer in enumerate(outer_by_child):
        if not eligibility.eligible_children[child]:
            continue
        selected_set = set(outer[:min(initial_top_k, len(outer))])
        eligible_parents = np.flatnonzero(eligibility.eligible_parents[child])
        outer_set = set(outer)
        for score in loco_scores:
            order = np.lexsort((
                eligible_parents, -score[child, eligible_parents]
            ))
            for parent in eligible_parents[order[:min(
                    initial_top_k, len(eligible_parents)
            )]]:
                parent_value = int(parent)
                if parent_value in outer_set:
                    selected_set.add(parent_value)
        selected = [parent for parent in outer if parent in selected_set]
        selected = selected[:len(outer)]
        selected_counts[child] = len(selected)
        pairs = {
            tuple(sorted((selected[first], selected[second])))
            for first in range(len(selected))
            for second in range(first + 1, len(selected))
            if pedigree_eligibility._eligible_parent_pair(
                eligibility, child, selected[first], selected[second]
            )
        }
        if use_anchor_union:
            # Anchor identities remain those from the full-data fixed panel.
            for anchor in outer[:min(int(anchor_k), len(outer))]:
                for other in eligible_parents:
                    if (
                        int(other) != anchor
                        and pedigree_eligibility._eligible_parent_pair(
                            eligibility, child, anchor, int(other)
                        )
                    ):
                        pairs.add(tuple(sorted((anchor, int(other)))))
        rows.extend((child, first, second) for first, second in sorted(pairs))
    trios = np.asarray(rows, dtype=np.int64).reshape((-1, 3))
    return trios, _panel_diagnostics(
        initial_top_k, outer_top_k, len(pair_score_array), eligibility,
        selected_counts, len(trios), applied=True,
    )


def score_prepared_t09_parent_state_evidence(
        prepared: pedigree_components.PreparedPedigree,
        *,
        parent_eligibility: Any=None,
        config: module_pedigree_config.PedigreeConfig | None=None,
        top_k: int=20,
        adaptive_initial_top_k: int | None=None,
        anchor_k: int=5,
        use_anchor_union: bool=False,
        mismatch_penalty: float=pedigree_models.DEFAULT_MISMATCH_PENALTY,
        candidate_source_mode: str | None=None,
        evidence_identity: Mapping[str, Any] | None=None,
        chromosome_evidence_callback: Callable[[
            pedigree_components.Stage10ChromosomeEvidenceRequest,
            Callable[[], pedigree_components.ScoredT09ChromosomeEvidence],
        ], pedigree_components.ScoredT09ChromosomeEvidence] | None=None,
) -> pedigree_components.ScoredT09ParentStateEvidence:
    """Compute or resume all expensive score-stage evidence.

    The callback is invoked once per informative physical chromosome after the
    common screened trio panel is fixed. It may return a matching cached value
    without calling ``producer``, or call the producer and atomically persist
    its lean result. Decision-only thresholds, direction policy, priors, graph
    selection, and bootstrap settings do not enter the score identity.
    """

    scoring_started = time.perf_counter()
    settings = (config or module_pedigree_config.PedigreeConfig()).validated()
    source_mode = pedigree_components._resolve_source_mode(settings, candidate_source_mode)

    if not isinstance(prepared, pedigree_components.PreparedPedigree):
        raise pedigree_models.PedigreeEvidenceError("prepared must be a PreparedT09PedigreeRun")
    if (
        not isinstance(prepared.sample_ids, tuple)
        or not prepared.sample_ids
        or len(set(prepared.sample_ids)) != len(prepared.sample_ids)
        or not isinstance(prepared.chromosomes, tuple)
        or not isinstance(prepared.omitted_chromosomes, tuple)
    ):
        raise pedigree_models.PedigreeEvidenceError("prepared T09 run has an invalid ordered layout")
    names = tuple(value.contig for value in prepared.chromosomes)
    omitted_names = tuple(value.contig for value in prepared.omitted_chromosomes)
    if (
        len(set(names + omitted_names)) != len(names) + len(omitted_names)
        or any(
            not isinstance(value, pedigree_components.PreparedChromosome)
            or value.sample_ids != prepared.sample_ids
            or not value.components
            for value in prepared.chromosomes
        )
    ):
        raise pedigree_models.PedigreeEvidenceError(
            "prepared T09 chromosomes have inconsistent names or sample order"
        )
    if (
        prepared.markers_per_information_block
        != settings.markers_per_information_block
        or prepared.effective_markers_per_information_block
        != settings.parent_state_effective_markers_per_information_block
    ):
        raise pedigree_models.PedigreeEvidenceError(
            "prepared T09 information tempering does not match config"
        )
    if (
        prepared.source_mode != source_mode
        or any(value.source_mode != source_mode for value in prepared.chromosomes)
    ):
        raise pedigree_models.PedigreeEvidenceError(
            "prepared T09 source mode does not match requested inference mode"
        )
    if not np.isfinite(mismatch_penalty) or mismatch_penalty >= 0.0:
        raise pedigree_models.PedigreeEvidenceError("mismatch_penalty must be finite and negative")
    if not isinstance(use_anchor_union, (bool, np.bool_)):
        raise pedigree_models.PedigreeEvidenceError("use_anchor_union must be boolean")
    if chromosome_evidence_callback is not None and not callable(
            chromosome_evidence_callback):
        raise pedigree_models.PedigreeEvidenceError("chromosome_evidence_callback must be callable")
    if not prepared.chromosomes:
        omitted = ", ".join(
            value.contig for value in prepared.omitted_chromosomes
        )
        raise pedigree_models.PedigreeEvidenceError(
            "no physical chromosome contains observed nonuniform evidence"
            + (f": {omitted}" if omitted else "")
        )

    eligibility = pedigree_eligibility._resolve_parent_eligibility(
        parent_eligibility, prepared.sample_ids
    )
    pair_scores = []
    marker_counts = []
    ragged_screen_scores_by_chromosome = []
    for chromosome in prepared.chromosomes:
        component_screens = [
            _projected_m1_screen(
                component,
                exponent,
                settings,
                eligibility.eligible_children,
                eligibility.eligible_parents,
            )
            for component, exponent in zip(
                chromosome.components, chromosome.information_exponents
            )
        ]
        component_pair_scores = [
            value.one_observed for value in component_screens
        ]
        ragged_screen_scores_by_chromosome.append(tuple(component_screens))
        pair_scores.append(np.sum(np.stack(component_pair_scores), axis=0))
        marker_counts.append(sum(
            component.cache.informative_markers
            for component in chromosome.components
        ))
    pair_score_array = np.stack(pair_scores)
    marker_count_array = np.asarray(marker_counts, dtype=np.float64)
    parent_screen_scores = pedigree_candidates._robust_parent_screen(
        pair_score_array,
        marker_count_array,
        settings,
        eligibility,
    )
    trios, panel_diagnostics = _adaptive_trio_panel(
        pair_score_array,
        marker_count_array,
        parent_screen_scores,
        top_k,
        anchor_k,
        bool(use_anchor_union),
        eligibility,
        settings,
        adaptive_initial_top_k,
    )
    score_identity = pedigree_components._parent_state_score_identity(
        prepared,
        settings,
        eligibility,
        trios,
        top_k=top_k,
        adaptive_initial_top_k=adaptive_initial_top_k,
        anchor_k=anchor_k,
        use_anchor_union=bool(use_anchor_union),
        mismatch_penalty=float(mismatch_penalty),
        external_identity=evidence_identity,
    )
    run_digest = pedigree_components._identity_digest(score_identity)

    scored_chromosomes = []
    runtime_results = []
    for chromosome, screen_scores in zip(
            prepared.chromosomes, ragged_screen_scores_by_chromosome):
        produced_result = None

        def produce(
                chromosome=chromosome,
                screen_scores=screen_scores,
        ) -> pedigree_components.ScoredT09ChromosomeEvidence:
            nonlocal produced_result
            produced_result = _score_prepared_chromosome(
                chromosome,
                trios,
                settings,
                eligibility.eligible_children,
                eligibility.eligible_parents,
                ragged_screen_scores=screen_scores,
            )
            return _compact_chromosome_evidence(produced_result)

        chromosome_identity = {
            "run_score_identity_sha256": run_digest,
            "contig": chromosome.contig,
            "source_identity": getattr(chromosome, "source_identity", None),
        }
        request = pedigree_components.Stage10ChromosomeEvidenceRequest(
            chromosome.contig,
            score_identity,
            pedigree_components._identity_digest(chromosome_identity),
            len(trios),
        )
        if chromosome_evidence_callback is None:
            scored_chromosome = produce()
        else:
            scored_chromosome = chromosome_evidence_callback(request, produce)
        scored_chromosome = _validate_scored_chromosome(
            scored_chromosome,
            contig=chromosome.contig,
            n_samples=len(prepared.sample_ids),
            n_trios=len(trios),
        )
        scored_chromosomes.append(scored_chromosome)
        runtime_results.append(produced_result)

    return pedigree_components.ScoredT09ParentStateEvidence(
        sample_ids=prepared.sample_ids,
        contig_names=names,
        chromosomes=tuple(scored_chromosomes),
        trios=trios,
        parent_screen_scores=parent_screen_scores,
        omitted_chromosomes=prepared.omitted_chromosomes,
        parent_panel_diagnostics=panel_diagnostics,
        source_mode=source_mode,
        score_identity=score_identity,
        input_preparation_seconds=float(prepared.input_preparation_seconds),
        screening_and_scoring_seconds=time.perf_counter() - scoring_started,
        runtime_chromosome_results=(
            tuple(runtime_results)
            if all(value is not None for value in runtime_results) else None
        ),
    )


def infer_scored_t09_parent_state_evidence(
        scored: pedigree_components.ScoredT09ParentStateEvidence,
        *,
        parent_eligibility: Any=None,
        config: module_pedigree_config.PedigreeConfig | None=None,
        n_workers: int | None=None,
) -> pedigree_components.ComponentPedigreeRunResult:
    """Run decision policy and bootstraps without T09/raw/scorer reload."""

    inference_started = time.perf_counter()
    if not isinstance(scored, pedigree_components.ScoredT09ParentStateEvidence):
        raise pedigree_models.PedigreeEvidenceError(
            "scored must be a ScoredT09ParentStateEvidence"
        )
    settings = (config or module_pedigree_config.PedigreeConfig()).validated()
    score_identity = pedigree_components._canonical_identity(
        scored.score_identity, "stored parent-state score identity"
    )
    if score_identity.get("schema") != pedigree_components._EVIDENCE_SCORE_IDENTITY_SCHEMA:
        raise pedigree_models.PedigreeEvidenceError("stored parent-state score identity is unknown")
    if score_identity.get("scoring_config") != pedigree_components._canonical_identity(
            pedigree_components._score_config_identity(settings), "current scoring config"):
        raise pedigree_models.PedigreeEvidenceError(
            "decision replay changed a score-stage configuration value"
        )
    eligibility = pedigree_eligibility._resolve_parent_eligibility(
        parent_eligibility, scored.sample_ids
    )
    if score_identity.get("scoring_eligibility") != pedigree_components._canonical_identity(
            pedigree_components._eligibility_score_identity(eligibility),
            "current scoring eligibility",
    ):
        raise pedigree_models.PedigreeEvidenceError(
            "decision replay changed child, parent, or pair scoring eligibility"
        )


    trios = _validated_trios(scored.trios)
    expected_names = tuple(value.contig for value in scored.chromosomes)
    if (
        scored.contig_names != expected_names
        or tuple(score_identity.get("ordered_sample_ids", ()))
        != scored.sample_ids
        or tuple(score_identity.get("ordered_informative_contigs", ()))
        != scored.contig_names
        or score_identity.get("candidate_panel", {}).get("trios_sha256")
        != pedigree_components._array_digest(trios)
        or np.asarray(scored.parent_screen_scores).shape
        != (len(scored.sample_ids), len(scored.sample_ids))
    ):
        raise pedigree_models.PedigreeEvidenceError("stored parent-state run axes or panel mismatch")
    chromosomes = tuple(
        _validate_scored_chromosome(
            chromosome,
            contig=name,
            n_samples=len(scored.sample_ids),
            n_trios=len(trios),
        )
        for name, chromosome in zip(scored.contig_names, scored.chromosomes)
    )
    chromosome_results = (
        scored.runtime_chromosome_results
        if scored.runtime_chromosome_results is not None
        else tuple(
            _chromosome_result_from_scored(chromosome, trios)
            for chromosome in chromosomes
        )
    )
    evidence = tuple(
        _chromosome_result_from_scored(chromosome, trios).evidence
        for chromosome in chromosomes
    )
    junctions = np.stack([
        chromosome.ancestry_junction_counts for chromosome in chromosomes
    ])
    callable_bins = np.stack([
        chromosome.ancestry_callable_haplotype_bins
        for chromosome in chromosomes
    ])
    scaffold_inputs = {}

    parent_state_started = time.perf_counter()
    pedigree_result = pedigree_inference.infer_from_parent_state_evidence(
        evidence,
        scored.sample_ids,
        config=settings,
        parent_eligibility=eligibility,
        ancestry_junction_counts=junctions,
        ancestry_callable_haplotype_bins=callable_bins,
        n_workers=n_workers,
        **scaffold_inputs,
    )
    parent_state_seconds = time.perf_counter() - parent_state_started

    replay_seconds = time.perf_counter() - inference_started
    execution_diagnostics = pedigree_components.ComponentPedigreeExecutionDiagnostics(
        input_preparation_seconds=float(scored.input_preparation_seconds),
        screening_and_scoring_seconds=float(
            scored.screening_and_scoring_seconds
        ),
        parent_state_aggregation_bootstrap_seconds=float(parent_state_seconds),
        prepared_inference_seconds=float(
            scored.screening_and_scoring_seconds + replay_seconds
        ),
        end_to_end_seconds=float(
            scored.input_preparation_seconds
            + scored.screening_and_scoring_seconds
            + replay_seconds
        ),
    )
    return pedigree_components.ComponentPedigreeRunResult(
        pedigree_result,
        chromosome_results,
        trios,
        scored.parent_screen_scores,
        scored.omitted_chromosomes,
        scored.parent_panel_diagnostics,
        execution_diagnostics,
    )


def infer_prepared_t09_component_pedigree(
        prepared: pedigree_components.PreparedPedigree,
        *,
        parent_eligibility: Any=None,
        config: module_pedigree_config.PedigreeConfig | None=None,
        top_k: int=20,
        adaptive_initial_top_k: int | None=None,
        anchor_k: int=5,
        use_anchor_union: bool=False,
        mismatch_penalty: float=pedigree_models.DEFAULT_MISMATCH_PENALTY,
        n_workers: int | None=None,
        candidate_source_mode: str | None=None,
        evidence_identity: Mapping[str, Any] | None=None,
        chromosome_evidence_callback: Callable[[
            pedigree_components.Stage10ChromosomeEvidenceRequest,
            Callable[[], pedigree_components.ScoredT09ChromosomeEvidence],
        ], pedigree_components.ScoredT09ChromosomeEvidence] | None=None,
) -> pedigree_components.ComponentPedigreeRunResult:
    """Score prepared chromosomes, then apply parent-state decision policy."""

    settings = (config or module_pedigree_config.PedigreeConfig()).validated()
    source_mode = pedigree_components._resolve_source_mode(settings, candidate_source_mode)

    scored = score_prepared_t09_parent_state_evidence(
        prepared,
        parent_eligibility=parent_eligibility,
        config=settings,
        top_k=top_k,
        adaptive_initial_top_k=adaptive_initial_top_k,
        anchor_k=anchor_k,
        use_anchor_union=use_anchor_union,
        mismatch_penalty=mismatch_penalty,
        candidate_source_mode=candidate_source_mode,
        evidence_identity=evidence_identity,
        chromosome_evidence_callback=chromosome_evidence_callback,
    )
    return infer_scored_t09_parent_state_evidence(
        scored,
        parent_eligibility=parent_eligibility,
        config=settings,
        n_workers=n_workers,
    )


import haplotype_reconstruction.pedigree.candidates as pedigree_candidates
import haplotype_reconstruction.pedigree.components as pedigree_components
import haplotype_reconstruction.pedigree.config as module_pedigree_config
import haplotype_reconstruction.pedigree.eligibility as pedigree_eligibility
import haplotype_reconstruction.pedigree.inference as pedigree_inference
import haplotype_reconstruction.pedigree.sources as pedigree_sources
import haplotype_reconstruction.pedigree.transmission as pedigree_transmission
