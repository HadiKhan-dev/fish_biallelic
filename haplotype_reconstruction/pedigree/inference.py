"""pedigree / inference for the canonical reconstruction pipeline."""
from __future__ import annotations


from typing import Any, Mapping, Optional, Sequence


import numpy as np
import pandas as pd


def infer_from_parent_state_evidence(
    evidence: Sequence[pedigree_models.ParentStateEvidence],
    sample_ids: Sequence[Any],
    config: Optional[module_pedigree_config.PedigreeConfig] = None,
    *,
    parent_eligibility: Optional[pedigree_eligibility.ParentEligibility | Mapping[str, Any]] = None,
    ancestry_junction_counts: Optional[np.ndarray] = None,
    ancestry_callable_haplotype_bins: Optional[np.ndarray] = None,
    n_workers: Optional[int] = None,
    candidate_source_available: Optional[np.ndarray] = None,
    child_informative_marker_count: Optional[np.ndarray] = None,
    scaffold_contig_names: Optional[Sequence[str]] = None,
) -> pedigree_results.PedigreeResult:
    """Infer a DAG from comparable 0/1/2-observed-parent likelihoods.

    State evidence is integrated over candidate identities only after contig
    log likelihoods have been summed. The sole supported combined method uses
    fixed B1 state priors, exposure and source-mode-specific C/X selection,
    configurable ancestry-direction parent-count screening, and a separate
    ancestry-depth-or-explicit direction identity gate; other analysed
    children do not alter a focal child's parent-count prior. Parent-count,
    conditional identity, and graph support are resampled separately so the DAG
    cannot create biological confidence.
    Every contig must supply parenthood-structure counts. Per-contig ancestry
    junction and callability matrices are also required, must have shape
    ``(len(evidence), len(sample_ids))``, and must be supplied together.
    The opt-in scaffold policy additionally requires source/child availability
    on those axes plus explicit matching scaffold_contig_names. Its biological
    M1 release uses direction-qualified mass and resample support; genetic local
    state diagnostics remain separate. Prior sensitivity re-fits the scaffold.
    """
    settings = (config or module_pedigree_config.PedigreeConfig()).validated()
    samples = list(sample_ids)
    n_samples = len(samples)
    if n_samples < 3 or len(set(samples)) != n_samples:
        raise pedigree_models.PedigreeEvidenceError(
            "sample_ids must contain at least three unique IDs"
        )
    eligibility = pedigree_eligibility._resolve_parent_eligibility(parent_eligibility, samples)
    (
        trios,
        zero,
        one,
        two,
        markers,
        edge_matched_by_contig,
        edge_exposed_by_contig,
        pair_explained_by_contig,
        pair_exposed_by_contig,
        structure_total_bins_by_contig,
        contig_names,
    ) = pedigree_evidence._canonical_parent_state_evidence(evidence, n_samples, eligibility)
    contig_information_weights = np.ceil(
        markers / settings.markers_per_information_block
    ).astype(np.float64)
    edge_exposure_presence_words = (
        None
        if edge_exposed_by_contig is None
        else pedigree_bootstrap.pack_contig_presence(
            (edge_exposed_by_contig > 0.0)
            & (edge_exposed_by_contig >= (
                settings.parent_state_minimum_edge_exposed_bins
            ))
        )
    )
    pair_exposure_presence_words = (
        None
        if pair_exposed_by_contig is None
        else pedigree_bootstrap.pack_contig_presence(
            (pair_exposed_by_contig > 0.0)
            & (pair_exposed_by_contig >= (
                settings.parent_state_minimum_pair_exposed_bins
            ))
        )
    )
    if (
        ancestry_junction_counts is None
        or ancestry_callable_haplotype_bins is None
    ):
        if not (
            ancestry_junction_counts is None
            and ancestry_callable_haplotype_bins is None
        ):
            raise pedigree_models.PedigreeEvidenceError(
                "ancestry junction counts and callable haplotype bins must "
                "be supplied together"
            )
        junction_matrix = callable_matrix = None
    else:
        junction_matrix = np.asarray(
            ancestry_junction_counts, dtype=np.float64
        )
        callable_matrix = np.asarray(
            ancestry_callable_haplotype_bins, dtype=np.float64
        )
        expected_shape = (len(contig_names), n_samples)
        if (
            junction_matrix.shape != expected_shape
            or callable_matrix.shape != expected_shape
            or np.any(~np.isfinite(junction_matrix))
            or np.any(~np.isfinite(callable_matrix))
            or np.any(junction_matrix < 0.0)
            or np.any(callable_matrix < 0.0)
        ):
            raise pedigree_models.PedigreeEvidenceError(
                "per-contig ancestry junction counts and callability must be "
                f"finite, non-negative arrays with shape {expected_shape}"
            )
    if edge_matched_by_contig is None:
        raise pedigree_models.PedigreeEvidenceError(
            "combined_v1 requires parenthood structure evidence on every contig"
        )
    if junction_matrix is None:
        raise pedigree_models.PedigreeEvidenceError(
            "combined_v1 requires ancestry junction and callability evidence"
        )
    if len(contig_names) < settings.parent_state_minimum_exposed_contigs:
        raise pedigree_models.PedigreeEvidenceError(
            "combined_v1 requires at least parent_state_minimum_exposed_contigs "
            "contigs"
        )
    (
        alternatives,
        states,
        contig_log_likelihoods,
        by_child,
        full_counts,
        scored_counts,
    ) = pedigree_states._parent_state_alternatives(
        trios,
        zero,
        one,
        two,
        settings.parent_state_contamination_probability,
        eligibility,
        candidate_source_mode=settings.parent_state_candidate_source_mode,
    )

    structure_pair_indices = pedigree_states._structure_pair_indices(
        alternatives, states, trios
    )
    scaffold_data = scaffold_prepared = None


    def evaluate(
        weights: np.ndarray,
        base_priors: Sequence[float] = settings.parent_state_priors,
        depth_model: Optional[pedigree_direction._AncestryDepthModel] = None,
        prepared_aggregates: Optional[
            tuple[np.ndarray, np.ndarray]
        ] = None,
    ) -> pedigree_states._ParentStateSelection:
        return pedigree_states._evaluate_parent_state_weighted_contigs(
            contig_log_likelihoods,
            weights,
            contig_information_weights,
            alternatives,
            states,
            by_child,
            full_counts,
            settings,
            n_samples,
            depth_model,
            base_priors,
            structure_pair_indices=structure_pair_indices,
            edge_matched_by_contig=edge_matched_by_contig,
            edge_exposed_by_contig=edge_exposed_by_contig,
            pair_explained_by_contig=pair_explained_by_contig,
            pair_exposed_by_contig=pair_exposed_by_contig,
            structure_total_bins_by_contig=structure_total_bins_by_contig,
            edge_exposure_presence_words=edge_exposure_presence_words,
            pair_exposure_presence_words=pair_exposure_presence_words,
            direction_supported_parents=(
                eligibility.direction_supported_parents
            ),
            prepared_aggregates=prepared_aggregates,
            scaffold_prepared=scaffold_prepared,
        )

    full_weights = np.ones(len(contig_names), dtype=np.float64)
    full_aggregate = np.sum(contig_log_likelihoods, axis=0)
    full_aggregate = pedigree_states._apply_aggregate_parent_state_contamination(
        full_aggregate,
        alternatives,
        states,
        settings.parent_state_contamination_probability,
    )
    total_junction_counts = np.sum(junction_matrix, axis=0)
    total_callable_bins = np.sum(callable_matrix, axis=0)
    full_depth_model = pedigree_direction._fit_ancestry_depth_model(
        total_junction_counts,
        total_callable_bins,
        settings.bootstrap_seed,
    )
    full_scaffold_veto = full_scaffold_result = None

    full_prepared_aggregates = pedigree_states._prepare_parent_state_weighted_contigs(
        contig_log_likelihoods,
        full_weights,
        alternatives,
        states,
        settings,
        full_depth_model,
        structure_pair_indices,
        edge_matched_by_contig,
        edge_exposed_by_contig,
        pair_explained_by_contig,
        pair_exposed_by_contig,
        structure_total_bins_by_contig,
        edge_exposure_presence_words,
        pair_exposure_presence_words,
        eligibility.direction_supported_parents,
        scaffold_prepared=scaffold_prepared,
        contig_information_weights=contig_information_weights,
        scaffold_descendant_veto=full_scaffold_veto,
    )
    full_selection = evaluate(
        full_weights, depth_model=full_depth_model,
        prepared_aggregates=full_prepared_aggregates,
    )

    (
        full_exposure_testable,
        full_cx_compatible,
        full_selection_compatible,
        full_identity_eligible,
        full_structure_child_evaluable,
        full_edge_coverage,
        full_pair_explainability,
        full_edge_direction,
        full_edge_direction_supported,
    ) = pedigree_states._parent_state_structure_mask(
        full_weights,
        alternatives,
        states,
        structure_pair_indices,
        edge_matched_by_contig,
        edge_exposed_by_contig,
        pair_explained_by_contig,
        pair_exposed_by_contig,
        structure_total_bins_by_contig,
        full_depth_model.posterior,
        settings,
        direction_supported_parents=eligibility.direction_supported_parents,
        edge_exposure_presence_words=edge_exposure_presence_words,
        pair_exposure_presence_words=pair_exposure_presence_words,
    )
    full_pre_direction_selection = (
        full_cx_compatible
        if pedigree_states._resolved_structure_cx_veto_policy(settings) == "hard_gate_v1"
        else full_exposure_testable
    )
    full_direction_state_rejected = (
        (states != pedigree_models._ZERO_OBSERVED)
        & full_pre_direction_selection
        & ~full_selection_compatible
    )
    full_scaffold_state_rejected = np.zeros(len(alternatives), dtype=np.bool_)
    if full_scaffold_veto is not None:
        for parent_column in (1, 2):
            rows = np.flatnonzero(alternatives[:, parent_column] >= 0)
            full_scaffold_state_rejected[rows] |= full_scaffold_veto[
                alternatives[rows, 0], alternatives[rows, parent_column]]
        full_scaffold_state_rejected &= full_exposure_testable
        full_selection_compatible[full_scaffold_state_rejected] = False
        full_identity_eligible[full_scaffold_state_rejected] = False
    informative = np.zeros(
        (len(contig_names), n_samples), dtype=np.bool_
    )
    base_log_prior = np.zeros(3, dtype=np.float64)
    state_indices = np.arange(3, dtype=np.int64)
    for contig_index in range(len(contig_names)):
        contig_state_evidence = pedigree_states._integrated_parent_state_log_evidence(
            contig_log_likelihoods[contig_index],
            states,
            by_child,
            full_counts,
        ) + base_log_prior
        for child in range(n_samples):
            winner, _ = pedigree_states._unique_finite_winner(
                state_indices, contig_state_evidence[child]
            )
            informative[contig_index, child] = winner is not None

    n_alternatives = len(alternatives)
    local_configuration_counts = np.zeros(
        n_alternatives, dtype=np.int64
    )
    graph_configuration_counts = np.zeros(
        n_alternatives, dtype=np.int64
    )
    local_state_counts = np.zeros((n_samples, 3), dtype=np.int64)
    graph_state_counts = None
    local_parent_counts = np.zeros(
        (n_samples, n_samples), dtype=np.int64
    )
    graph_parent_counts = np.zeros(
        (n_samples, n_samples), dtype=np.int64
    )

    def accumulate(
        selection: pedigree_states._ParentStateSelection,
        configuration_counts: np.ndarray,
        state_counts: Optional[np.ndarray],
        parent_counts: np.ndarray,
        graph: bool,
    ) -> None:
        selected_rows = selection.graph_rows if graph else selection.local_rows
        selected_states = (
            {
                child: int(states[row])
                for child, row in selection.graph_rows.items()
            }
            if graph
            else selection.local_states
        )
        if state_counts is not None:
            for child, state in selected_states.items():
                state_counts[child, state] += 1
        for child, row in selected_rows.items():
            configuration_counts[row] += 1
            for parent in alternatives[row, 1:]:
                parent = int(parent)
                if parent >= 0:
                    parent_counts[child, parent] += 1

    m1_direction_state_counts = (
        None)
    bootstrap_worker_count, bootstrap_depth_refits = (
        pedigree_bootstrap._run_parent_state_bootstraps(
            contig_log_likelihoods,
            alternatives,
            states,
            by_child,
            full_counts,
            junction_matrix,
            callable_matrix,
            settings,
            n_workers,
            local_configuration_counts,
            graph_configuration_counts,
            local_state_counts,
            graph_state_counts,
            local_parent_counts,
            graph_parent_counts,
            contig_information_weights=contig_information_weights,
            structure_pair_indices=structure_pair_indices,
            edge_matched_by_contig=edge_matched_by_contig,
            edge_exposed_by_contig=edge_exposed_by_contig,
            pair_explained_by_contig=pair_explained_by_contig,
            pair_exposed_by_contig=pair_exposed_by_contig,
            structure_total_bins_by_contig=structure_total_bins_by_contig,
            edge_exposure_presence_words=edge_exposure_presence_words,
            pair_exposure_presence_words=pair_exposure_presence_words,
            direction_supported_parents=(
                eligibility.direction_supported_parents
            ),
            scaffold_data=scaffold_data,
            m1_direction_state_counts=m1_direction_state_counts,
            depth_component_count=full_depth_model.posterior.shape[1],
        )
    )

    loco_local_configuration_counts = np.zeros(
        n_alternatives, dtype=np.int64
    )
    loco_graph_configuration_counts = np.zeros(
        n_alternatives, dtype=np.int64
    )
    loco_local_state_counts = np.zeros(
        (n_samples, 3), dtype=np.int64
    )
    loco_graph_state_counts = None
    loco_local_parent_counts = np.zeros(
        (n_samples, n_samples), dtype=np.int64
    )
    loco_graph_parent_counts = np.zeros(
        (n_samples, n_samples), dtype=np.int64
    )
    m1_direction_loco_counts = (
        None)
    n_loco = 0
    if len(contig_names) > 1:
        for omitted in range(len(contig_names)):
            loco_weights = full_weights.copy()
            loco_weights[omitted] = 0.0
            loco_depth_model = pedigree_direction._fit_ancestry_depth_model(
                total_junction_counts
                - junction_matrix[omitted],
                total_callable_bins
                - callable_matrix[omitted],
                settings.bootstrap_seed,
                component_count=full_depth_model.posterior.shape[1],
            )
            selection = evaluate(loco_weights, depth_model=loco_depth_model)
            if m1_direction_loco_counts is not None:
                m1_direction_loco_counts += selection.m1_direction_state_supported
            n_loco += 1
            accumulate(
                selection,
                loco_local_configuration_counts,
                loco_local_state_counts,
                loco_local_parent_counts,
                False,
            )
            accumulate(
                selection,
                loco_graph_configuration_counts,
                loco_graph_state_counts,
                loco_graph_parent_counts,
                True,
            )

    sensitivity_runs = []
    sensitivity_summary_rows = []
    scaffold_prior_sensitivity_refits = 0
    for base_priors in settings.parent_state_prior_sensitivity:

        selection = (
            full_selection
            if tuple(base_priors) == tuple(settings.parent_state_priors)
            else evaluate(
                full_weights, base_priors, full_depth_model,
                full_prepared_aggregates,
            )
        )
        sensitivity_runs.append((base_priors, selection))
        state_call_counts = [
            sum(
                int(state == target)
                for state in selection.local_states.values()
            )
            for target in range(3)
        ]
        sensitivity_summary_rows.append({
            "BasePrior0": base_priors[0],
            "BasePrior1": base_priors[1],
            "BasePrior2": base_priors[2],
            "FittedDirichletParameter0": (
                selection.fitted_prior_parameters[0]
            ),
            "FittedDirichletParameter1": (
                selection.fitted_prior_parameters[1]
            ),
            "FittedDirichletParameter2": (
                selection.fitted_prior_parameters[2]
            ),
            "LocalZeroObservedCalls": state_call_counts[0],
            "LocalOneObservedCalls": state_call_counts[1],
            "LocalTwoObservedCalls": state_call_counts[2],
            "LocalIdentityResolvedCalls": len(selection.local_rows),
            "GraphConfigurationCalls": len(selection.graph_rows),
        })

    bootstrap_denominator = float(settings.bootstrap_replicates)
    loco_denominator = float(n_loco) if n_loco else np.nan
    complete_rows: dict[int, Optional[int]] = {}
    complete_states: dict[int, int] = {}
    complete_status = {}
    scaffold_direction_unresolved = np.zeros(n_samples, dtype=np.bool_)
    tier_a_rows: dict[int, Optional[int]] = {}
    tier_b_rows: dict[int, Optional[int]] = {}
    tier_a_states: dict[int, int] = {}
    tier_b_states: dict[int, int] = {}
    tier_a_status = {}
    tier_b_status = {}
    tier_a_parent_flags: dict[int, tuple[bool, bool]] = {}
    tier_b_parent_flags: dict[int, tuple[bool, bool]] = {}
    diagnostics = []
    state_call_rows = []
    trio_candidates = {}
    parent_candidates = {}
    trio_scores = {}
    evidence_support_sets = {}

    def stable_fraction(count: int, denominator: float) -> float:
        if not np.isfinite(denominator) or denominator <= 0.0:
            return 0.0
        return float(count / denominator)

    for child, sample in enumerate(samples):
        child_rows = by_child[child]
        m1_rows = child_rows[states[child_rows] == pedigree_models._ONE_OBSERVED]
        m2_rows = child_rows[states[child_rows] == pedigree_models._TWO_OBSERVED]
        best_m1_row, best_m1_margin = pedigree_states._unique_finite_winner(
            m1_rows, full_selection.decision_scores
        )
        best_m2_row, best_m2_margin = pedigree_states._unique_finite_winner(
            m2_rows, full_selection.decision_scores
        )
        best_m1_parent = (
            None if best_m1_row is None
            else int(alternatives[best_m1_row, 1])
        )
        best_m2_parents = (() if best_m2_row is None else tuple(
            int(parent) for parent in alternatives[best_m2_row, 1:]
        ))
        local_state = full_selection.local_states.get(child)
        local_row = full_selection.local_rows.get(child)
        graph_row = full_selection.graph_rows.get(child)
        graph_tie_conflict = child in full_selection.graph_tie_conflicts
        graph_direction_resolved = (
            child in full_selection.graph_direction_resolved_children
        )
        selected_parent_role_probability = (
            full_selection.graph_parent_role_probabilities.get(child, np.nan)
        )
        local_parent_role_probability = (
            np.nan
            if local_row is None
            else pedigree_direction._parent_role_probability(
                local_row,
                alternatives,
                (
                    None
                    if full_depth_model is None
                    else full_depth_model.posterior
                ),
            )
        )
        graph_conflict = bool(
            graph_tie_conflict
            or (local_row is not None and graph_row != local_row)
        )
        graph_displaced = (
            local_row is not None
            and graph_row is not None
            and graph_row != local_row
        )
        graph_state = (
            None if graph_row is None else int(states[graph_row])
        )
        informative_count = int(np.count_nonzero(informative[:, child]))
        enough_contigs = (
            informative_count >= settings.minimum_informative_contigs
        )

        if local_state is None:
            state_bootstrap = state_loco = np.nan
        else:
            state_bootstrap = stable_fraction(
                local_state_counts[child, local_state],
                bootstrap_denominator,
            )
            state_loco = stable_fraction(
                loco_local_state_counts[child, local_state],
                loco_denominator,
            )
        if local_row is None:
            local_configuration_bootstrap = np.nan
            graph_configuration_bootstrap = np.nan
            local_configuration_loco = np.nan
            graph_configuration_loco = np.nan
            selected_parents: tuple[int, ...] = ()
        else:
            local_configuration_bootstrap = stable_fraction(
                local_configuration_counts[local_row],
                bootstrap_denominator,
            )
            graph_configuration_bootstrap = stable_fraction(
                graph_configuration_counts[local_row],
                bootstrap_denominator,
            )
            local_configuration_loco = stable_fraction(
                loco_local_configuration_counts[local_row],
                loco_denominator,
            )
            graph_configuration_loco = stable_fraction(
                loco_graph_configuration_counts[local_row],
                loco_denominator,
            )
            selected_parents = tuple(
                int(parent)
                for parent in alternatives[local_row, 1:]
                if int(parent) >= 0
            )

        parent_bootstrap = [
            stable_fraction(
                local_parent_counts[child, parent], bootstrap_denominator
            )
            for parent in selected_parents
        ]
        parent_loco = [
            stable_fraction(
                loco_local_parent_counts[child, parent], loco_denominator
            )
            for parent in selected_parents
        ]
        first_bootstrap = (
            parent_bootstrap[0] if parent_bootstrap else np.nan
        )
        second_bootstrap = (
            parent_bootstrap[1] if len(parent_bootstrap) > 1 else np.nan
        )
        first_loco = parent_loco[0] if parent_loco else np.nan
        second_loco = parent_loco[1] if len(parent_loco) > 1 else np.nan

        m0_sensitivity_state_agreement = bool(
            local_state != pedigree_models._ZERO_OBSERVED
            or all(
                selection.local_states.get(child) == pedigree_models._ZERO_OBSERVED
                for _, selection in sensitivity_runs
            )
        )
        m0_sensitivity_graph_agreement = bool(
            local_state != pedigree_models._ZERO_OBSERVED
            or all(
                (
                    selection.graph_rows.get(child) is not None
                    and int(states[selection.graph_rows[child]])
                    == pedigree_models._ZERO_OBSERVED
                )
                for _, selection in sensitivity_runs
            )
        )
        m0_sensitivity_tier_veto = bool(
            local_state == pedigree_models._ZERO_OBSERVED
            and not m0_sensitivity_state_agreement
        )

        scaffold_direction_unresolved[child] = bool(
            scaffold_prepared is not None and local_state == pedigree_models._ONE_OBSERVED
            and not full_selection.m1_direction_state_supported[child])

        def tier_decision(
            state_bootstrap_cutoff: float,
            parent_bootstrap_cutoff: float,
            loco_cutoff: float,
        ) -> tuple[bool, bool, tuple[bool, bool]]:
            state_pass = bool(
                local_state is not None
                and enough_contigs
                and state_bootstrap >= state_bootstrap_cutoff
                and state_loco >= loco_cutoff
                and m0_sensitivity_state_agreement
                and not scaffold_direction_unresolved[child]
                and (scaffold_prepared is None or local_state != pedigree_models._ONE_OBSERVED or (
                    stable_fraction(m1_direction_state_counts[child], bootstrap_denominator)
                    >= state_bootstrap_cutoff
                    and stable_fraction(m1_direction_loco_counts[child], loco_denominator)
                    >= loco_cutoff
                ))
            )
            parent_flags = tuple(
                bool(
                    state_pass
                    and parent_bootstrap[index]
                    >= parent_bootstrap_cutoff
                    and parent_loco[index] >= loco_cutoff
                )
                for index in range(len(selected_parents))
            )
            identity_pass = bool(
                state_pass
                and local_row is not None
                and graph_row == local_row
                and not graph_tie_conflict
                and local_configuration_bootstrap
                >= state_bootstrap_cutoff
                and graph_configuration_bootstrap
                >= state_bootstrap_cutoff
                and local_configuration_loco >= loco_cutoff
                and graph_configuration_loco >= loco_cutoff
                and all(parent_flags)
                and m0_sensitivity_graph_agreement
            )
            padded_flags = (
                parent_flags[0] if len(parent_flags) > 0 else False,
                parent_flags[1] if len(parent_flags) > 1 else False,
            )
            return state_pass, identity_pass, padded_flags

        tier_a_state_pass, tier_a_identity_pass, tier_a_flags = tier_decision(
            settings.tier_a_pair_bootstrap,
            settings.tier_a_parent_bootstrap,
            settings.tier_a_loco_fraction,
        )
        tier_b_state_pass, tier_b_identity_pass, tier_b_flags = tier_decision(
            settings.tier_b_pair_bootstrap,
            settings.tier_b_parent_bootstrap,
            settings.tier_b_loco_fraction,
        )
        tier_a_parent_flags[child] = tier_a_flags
        tier_b_parent_flags[child] = tier_b_flags

        if not eligibility.eligible_children[child]:
            complete_rows[child] = None
            complete_status[child] = "excluded_by_parent_eligibility"
        elif scaffold_direction_unresolved[child]:
            complete_rows[child] = None
            complete_status[child] = "genetic_m1_directionally_unresolved"
        elif graph_tie_conflict:
            complete_rows[child] = None
            complete_states[child] = local_state
            complete_status[child] = (
                "graph_tie_conflict_parent_identity_unresolved"
            )
        elif graph_row is not None:
            complete_rows[child] = graph_row
            complete_states[child] = graph_state
            if graph_row == local_row:
                complete_status[child] = (
                    f"selected_{pedigree_models._PARENT_STATE_NAMES[graph_state]}"
                )
            elif graph_direction_resolved:
                complete_status[child] = (
                    "graph_displaced_direction_resolved_same_state_"
                    "hypothesis_not_tier_eligible"
                )
            else:
                complete_status[child] = (
                    "graph_displaced_hypothesis_not_tier_eligible"
                )
        elif local_state is not None:
            complete_rows[child] = None
            complete_states[child] = local_state
            complete_status[child] = (
                full_selection.unresolved_reasons[child]
                or "parent_state_resolved_identity_unresolved"
            )
        else:
            complete_rows[child] = None
            complete_status[child] = (
                full_selection.unresolved_reasons[child]
                or "unresolved_parent_state"
            )

        def populate_tier(
            state_pass: bool,
            identity_pass: bool,
            tier_rows: dict[int, Optional[int]],
            tier_states: dict[int, int],
            tier_status: dict[int, str],
            label: str,
        ) -> None:
            if not eligibility.eligible_children[child]:
                tier_rows[child] = None
                tier_status[child] = "excluded_by_parent_eligibility"
                return
            if scaffold_direction_unresolved[child]:
                tier_rows[child] = None
                tier_status[child] = "genetic_m1_directionally_unresolved"
                return
            if not state_pass or local_state is None:
                if m0_sensitivity_tier_veto:
                    tier_rows[child] = None
                    tier_status[child] = (
                        f"unresolved_{label}_m0_prior_sensitivity"
                    )
                    return
                tier_rows[child] = None
                tier_status[child] = f"unresolved_below_{label}_state_support"
                return
            tier_states[child] = local_state
            if identity_pass:
                tier_rows[child] = local_row
                tier_status[child] = (
                    f"{label}_supported_{pedigree_models._PARENT_STATE_NAMES[local_state]}"
                )
            else:
                tier_rows[child] = None
                suffix = (
                    "graph_tie_conflict"
                    if graph_tie_conflict
                    else (
                        "graph_conflict"
                        if graph_conflict
                        else "identity_unresolved"
                    )
                )
                tier_status[child] = (
                    f"{label}_parent_state_supported_{suffix}"
                )

        populate_tier(
            tier_a_state_pass,
            tier_a_identity_pass,
            tier_a_rows,
            tier_a_states,
            tier_a_status,
            "tier_a",
        )
        populate_tier(
            tier_b_state_pass,
            tier_b_identity_pass,
            tier_b_rows,
            tier_b_states,
            tier_b_status,
            "tier_b",
        )

        sensitivity_state_agreement = []
        sensitivity_identity_agreement = []
        sensitivity_graph_agreement = []
        sensitivity_selected_support = []
        for _, selection in sensitivity_runs:
            sensitivity_state_agreement.append(
                selection.local_states.get(child) == local_state
            )
            sensitivity_identity_agreement.append(
                selection.local_rows.get(child) == local_row
            )
            sensitivity_graph_agreement.append(
                selection.graph_rows.get(child) == graph_row
            )
            if local_state is not None:
                sensitivity_selected_support.append(
                    selection.state_support[child, local_state]
                )

        def sensitivity_agreement(values):
            if not eligibility.eligible_children[child]:
                return np.nan
            return float(np.mean(values))

        graph_parents = (
            ()
            if graph_row is None
            else tuple(
                int(parent)
                for parent in alternatives[graph_row, 1:]
                if int(parent) >= 0
            )
        )
        selected_utility = (
            np.nan
            if graph_row is None
            else float(full_selection.decision_scores[graph_row])
        )
        selected_graph_configuration_bootstrap = (
            np.nan
            if graph_row is None
            else stable_fraction(
                graph_configuration_counts[graph_row],
                bootstrap_denominator,
            )
        )
        selected_graph_configuration_loco = (
            np.nan
            if graph_row is None
            else stable_fraction(
                loco_graph_configuration_counts[graph_row],
                loco_denominator,
            )
        )
        selected_graph_parent_bootstrap = [
            stable_fraction(
                graph_parent_counts[child, parent], bootstrap_denominator
            )
            for parent in graph_parents
        ]
        selected_graph_parent_loco = [
            stable_fraction(
                loco_graph_parent_counts[child, parent], loco_denominator
            )
            for parent in graph_parents
        ]
        unconstrained_best = (
            -np.inf
            if not len(child_rows)
            else float(np.max(full_selection.decision_scores[child_rows]))
        )
        selected_minus_best = (
            np.nan
            if graph_row is None
            else selected_utility - unconstrained_best
        )
        local_margin = (
            np.nan
            if local_state is None
            else float(full_selection.state_margins[child])
        )
        identity_margin = (
            np.nan
            if local_row is None
            else float(full_selection.identity_margins[child])
        )
        unresolved_reason = complete_status[child]
        support_text = pedigree_states._configuration_support_text(
            child_rows,
            local_configuration_counts,
            alternatives,
            states,
            samples,
            settings.support_set_coverage,
        )
        parent1_set, parent2_set, configuration_set_rows = (
            pedigree_states._evidence_parent_support_sets(
                child_rows,
                local_state,
                local_row,
                alternatives,
                states,
                full_selection.decision_scores,
                settings.support_set_coverage,
            )
        )
        evidence_support_sets[child] = (
            parent1_set, parent2_set, configuration_set_rows
        )
        selected_m1_gain = (
            np.nan
            if local_row is None
            else full_selection.m1_over_m0_edge_gains[local_row]
        )
        selected_m2_first_gain = (
            np.nan
            if local_row is None
            else full_selection.m2_over_first_m1_edge_gains[local_row]
        )
        selected_m2_second_gain = (
            np.nan
            if local_row is None
            else full_selection.m2_over_second_m1_edge_gains[local_row]
        )
        raw_junction_count = (
            np.nan
            if junction_matrix is None
            else float(np.sum(junction_matrix[:, child]))
        )
        callable_haplotype_bin_count = (
            np.nan
            if callable_matrix is None
            else float(np.sum(callable_matrix[:, child]))
        )
        if full_depth_model is None:
            adjusted_junction_burden = np.nan
            ancestry_callability = np.nan
            depth_posterior_text = ""
            depth_map = np.nan
            depth_component_count = 0
            depth_component_means = ""
            depth_component_standard_deviations = ""
            depth_component_weights = ""
            depth_selected_bic = np.nan
            depth_tested_bics = ""
        else:
            adjusted_junction_burden = float(
                full_depth_model.adjusted_junction_burden[child]
            )
            ancestry_callability = float(
                full_depth_model.callability_fraction[child]
            )
            child_depth_posterior = full_depth_model.posterior[child]
            depth_posterior_text = ";".join(
                f"{value:.8g}" for value in child_depth_posterior
            )
            depth_map = (
                np.nan
                if float(np.sum(child_depth_posterior)) <= 0.0
                else int(np.argmax(child_depth_posterior))
            )
            depth_component_count = len(
                full_depth_model.component_means
            )
            depth_component_means = ";".join(
                f"{value:.8g}"
                for value in full_depth_model.component_means
            )
            depth_component_standard_deviations = ";".join(
                f"{value:.8g}"
                for value in full_depth_model.component_standard_deviations
            )
            depth_component_weights = ";".join(
                f"{value:.8g}"
                for value in full_depth_model.component_weights
            )
            depth_selected_bic = full_depth_model.selected_bic
            depth_tested_bics = ";".join(
                f"{value:.8g}" for value in full_depth_model.tested_bics
            )

        structure_selected_cx_compatible = bool(
            local_row is not None and full_cx_compatible[local_row]
        )
        structure_selected_selection_compatible = bool(
            local_row is not None and full_selection_compatible[local_row]
        )
        structure_selected_identity_eligible = bool(
            local_row is not None and full_identity_eligible[local_row]
        )
        structure_parent1_coverage = np.nan
        structure_parent2_coverage = np.nan
        structure_parent1_direction = np.nan
        structure_parent2_direction = np.nan
        structure_parent1_direction_supported = np.nan
        structure_parent2_direction_supported = np.nan
        structure_pair_explainability = np.nan
        if local_row is not None and selected_parents:
            structure_parent1_coverage = full_edge_coverage[
                child, selected_parents[0]
            ]
            structure_parent1_direction = full_edge_direction[
                child, selected_parents[0]
            ]
            if len(selected_parents) > 1:
                structure_parent2_coverage = full_edge_coverage[
                    child, selected_parents[1]
                ]
                structure_parent2_direction = full_edge_direction[
                    child, selected_parents[1]
                ]
                pair_index = int(structure_pair_indices[local_row])
                if pair_index >= 0:
                    structure_pair_explainability = (
                        full_pair_explainability[pair_index]
                    )
        diagnostics.append({
            "Sample": sample,
            "ParentStateAlgorithmMode": pedigree_models._PARENT_STATE_LIKELIHOOD,
            "ParentStateStructureMode": pedigree_models._PARENT_STATE_METHOD,
            "StructureCXVetoPolicy": pedigree_states._resolved_structure_cx_veto_policy(
                settings
            ),
            "ParentStateDirectionStatePolicy": (
                settings.parent_state_direction_state_policy
            ),
            "DirectionStateRejectedAlternativeCount": int(np.count_nonzero(
                full_direction_state_rejected[child_rows]
            )),
            "M0PriorSensitivityStateAgreement": (
                m0_sensitivity_state_agreement
            ),
            "M0PriorSensitivityGraphAgreement": (
                m0_sensitivity_graph_agreement
            ),
            "M0PriorSensitivityTierVeto": m0_sensitivity_tier_veto,
            "StructureChildEvaluable": bool(
                full_structure_child_evaluable[child]
            ),
            "StructureSelectedRowExposureTestable": bool(
                local_row is not None and full_exposure_testable[local_row]
            ),
            "StructureSelectedRowCXCompatible": (
                structure_selected_cx_compatible
            ),
            "StructureSelectedRowSelectionCompatible": (
                structure_selected_selection_compatible
            ),
            "StructureSelectedRowStateCompatible": (
                structure_selected_selection_compatible
            ),
            "StructureSelectedRowIdentityEligible": (
                structure_selected_identity_eligible
            ),
            "StructureSelectedRowEligible": structure_selected_identity_eligible,
            "StructureParent1Coverage": structure_parent1_coverage,
            "StructureParent2Coverage": structure_parent2_coverage,
            "StructurePairExplainability": structure_pair_explainability,
            "StructureParent1DirectionProbability": structure_parent1_direction,
            "StructureParent2DirectionProbability": structure_parent2_direction,
            "StructureParent1DirectionSupported": (
                np.nan if len(selected_parents) < 1 else bool(
                    full_edge_direction_supported[child, selected_parents[0]]
                )
            ),
            "StructureParent2DirectionSupported": (
                np.nan if len(selected_parents) < 2 else bool(
                    full_edge_direction_supported[child, selected_parents[1]]
                )
            ),
            "CohortStatePriorUsed": False,
            "BestM1Row": (
                np.nan if best_m1_row is None else int(best_m1_row)
            ),
            "BestM1Parent": (
                None if best_m1_parent is None else samples[best_m1_parent]
            ),
            "BestM1ConditionalIdentityMargin": best_m1_margin,
            "BestM2Row": (
                np.nan if best_m2_row is None else int(best_m2_row)
            ),
            "BestM2Parent1": (
                None if len(best_m2_parents) < 1
                else samples[best_m2_parents[0]]
            ),
            "BestM2Parent2": (
                None if len(best_m2_parents) < 2
                else samples[best_m2_parents[1]]
            ),
            "BestM2ConditionalIdentityMargin": best_m2_margin,
            "LocalParent1": (
                None if len(selected_parents) < 1
                else samples[selected_parents[0]]
            ),
            "LocalParent2": (
                None if len(selected_parents) < 2
                else samples[selected_parents[1]]
            ),
            "EligibilityPolicy": eligibility.policy_name,
            "EligibleChild": bool(eligibility.eligible_children[child]),
            "EligibleParentCount": int(full_counts[child, 1]),
            "EligibleParentPairCount": int(full_counts[child, 2]),
            "CompleteParent1": (
                None if len(graph_parents) < 1 else samples[graph_parents[0]]
            ),
            "CompleteParent2": (
                None if len(graph_parents) < 2 else samples[graph_parents[1]]
            ),
            "ParentOrderMeaning": "unordered_sample_array_index",
            "SelectedParentState": (
                None if graph_state is None else pedigree_models._PARENT_STATE_NAMES[graph_state]
            ),
            "LocalWinnerParentState": (
                None if local_state is None else pedigree_models._PARENT_STATE_NAMES[local_state]
            ),
            "ObservedParentCount": (
                np.nan if graph_state is None else graph_state
            ),
            "LocalObservedParentCount": (
                np.nan if local_state is None else local_state
            ),
            "Identifiable": bool(
                local_row is not None and graph_row == local_row
            ),
            "DAGDisplaced": graph_displaced,
            "GraphConflict": graph_conflict,
            "GraphTieConflict": graph_tie_conflict,
            "GraphDirectionResolvedAlternative": graph_direction_resolved,
            "GraphFallbackPolicy": (
                "local_if_feasible_else_unique_finite_downward_state_or_M0"
            ),
            "LocalParentRoleProbability": local_parent_role_probability,
            "SelectedParentRoleProbability": (
                selected_parent_role_probability
            ),
            "RawAncestryJunctionCount": raw_junction_count,
            "CallableAncestryHaplotypeBinCount": (
                callable_haplotype_bin_count
            ),
            "AdjustedAncestryJunctionBurden": adjusted_junction_burden,
            "AncestryPaintingCallabilityFraction": ancestry_callability,
            "LatentAncestryDepthMAP": depth_map,
            "LatentAncestryDepthPosterior": depth_posterior_text,
            "LatentAncestryDepthComponentCount": depth_component_count,
            "LatentAncestryDepthComponentMeans": depth_component_means,
            "LatentAncestryDepthComponentStandardDeviations": (
                depth_component_standard_deviations
            ),
            "LatentAncestryDepthComponentWeights": depth_component_weights,
            "LatentAncestryDepthSelectedBIC": depth_selected_bic,
            "LatentAncestryDepthTestedBICs": depth_tested_bics,
            "AncestryDepthResampling": "conditional_full_data_component_count",
            "InformativeContigCount": informative_count,
            "LocalStateBootstrapFraction": state_bootstrap,
            "LocalConfigurationBootstrapFraction": (
                local_configuration_bootstrap
            ),
            "GraphConfigurationBootstrapFraction": (
                graph_configuration_bootstrap
            ),
            "SelectedGraphConfigurationBootstrapFraction": (
                selected_graph_configuration_bootstrap
            ),
            "SelectedGraphParent1BootstrapFraction": (
                selected_graph_parent_bootstrap[0]
                if len(selected_graph_parent_bootstrap) > 0 else np.nan
            ),
            "SelectedGraphParent2BootstrapFraction": (
                selected_graph_parent_bootstrap[1]
                if len(selected_graph_parent_bootstrap) > 1 else np.nan
            ),
            "PairBootstrapFraction": local_configuration_bootstrap,
            "Parent1BootstrapFraction": first_bootstrap,
            "Parent2BootstrapFraction": second_bootstrap,
            "LocalStateLOCOFraction": state_loco,
            "LocalConfigurationLOCOFraction": local_configuration_loco,
            "GraphConfigurationLOCOFraction": graph_configuration_loco,
            "SelectedGraphConfigurationLOCOFraction": (
                selected_graph_configuration_loco
            ),
            "SelectedGraphParent1LOCOFraction": (
                selected_graph_parent_loco[0]
                if len(selected_graph_parent_loco) > 0 else np.nan
            ),
            "SelectedGraphParent2LOCOFraction": (
                selected_graph_parent_loco[1]
                if len(selected_graph_parent_loco) > 1 else np.nan
            ),
            "PairLOCOFraction": local_configuration_loco,
            "Parent1LOCOFraction": first_loco,
            "Parent2LOCOFraction": second_loco,
            "StateWinnerMargin": local_margin,
            "ConditionalIdentityMargin": identity_margin,
            "UnconstrainedWinnerMargin": (
                min(local_margin, identity_margin)
                if np.isfinite(identity_margin)
                else local_margin
            ),
            "SelectedAggregateUtility": selected_utility,
            "SelectedMinusUnconstrainedBest": selected_minus_best,
            "StateLogEvidence0": full_selection.state_log_evidence[child, 0],
            "StateLogEvidence1": full_selection.state_log_evidence[child, 1],
            "StateLogEvidence2": full_selection.state_log_evidence[child, 2],
            "StateSupport0": full_selection.state_support[child, 0],
            "StateSupport1": full_selection.state_support[child, 1],
            "StateSupport2": full_selection.state_support[child, 2],
            "LOOStatePrior0": full_selection.loo_state_priors[child, 0],
            "LOOStatePrior1": full_selection.loo_state_priors[child, 1],
            "LOOStatePrior2": full_selection.loo_state_priors[child, 2],
            "ScoredCandidateCount0": scored_counts[child, 0],
            "ScoredCandidateCount1": scored_counts[child, 1],
            "ScoredCandidateCount2": scored_counts[child, 2],
            "FullCandidateCount0": full_counts[child, 0],
            "FullCandidateCount1": full_counts[child, 1],
            "FullCandidateCount2": full_counts[child, 2],
            "CandidatePairCount": scored_counts[child, 2],
            "FullCandidatePairCount": full_counts[child, 2],
            "M2CandidateScreenIncomplete": bool(
                scored_counts[child, 2] < full_counts[child, 2]
            ),
            "M2StateEvidenceIsLowerBound": bool(
                scored_counts[child, 2] < full_counts[child, 2]
            ),
            "M2PredictiveScreenLowerBoundGuarantee": (
                "not_applicable_integrated_evidence"
            ),
            "B3HeldOutStateMaskPolicy": "none",
            "B3AggregateIdentityMaskPolicy": "none",
            "PriorSensitivityLocalStateAgreementFraction": (
                sensitivity_agreement(sensitivity_state_agreement)
            ),
            "PriorSensitivityLocalIdentityAgreementFraction": (
                sensitivity_agreement(sensitivity_identity_agreement)
            ),
            "PriorSensitivityGraphAgreementFraction": (
                sensitivity_agreement(sensitivity_graph_agreement)
            ),
            "PriorSensitivitySelectedStateMinimumSupport": (
                np.nan
                if not sensitivity_selected_support
                else float(np.min(sensitivity_selected_support))
            ),
            "UnresolvedReason": unresolved_reason,
            "InferenceStatus": complete_status[child],
            "TierAStateCall": tier_a_state_pass,
            "TierBStateCall": tier_b_state_pass,
            "TierAExactConfiguration": tier_a_identity_pass,
            "TierBExactConfiguration": tier_b_identity_pass,
            "TierAExactPair": bool(
                tier_a_identity_pass and local_state == pedigree_models._TWO_OBSERVED
            ),
            "TierBExactPair": bool(
                tier_b_identity_pass and local_state == pedigree_models._TWO_OBSERVED
            ),
            "TierAParent1": tier_a_flags[0],
            "TierAParent2": tier_a_flags[1],
            "TierBParent1": tier_b_flags[0],
            "TierBParent2": tier_b_flags[1],
            "PairSupportSet": support_text,
            "ConfigurationSupportSet": support_text,
            "Parent1CandidateSet": pedigree_states._sample_set_text(parent1_set, samples),
            "Parent2CandidateSet": pedigree_states._sample_set_text(parent2_set, samples),
            "EvidenceConfigurationSet": ";".join(
                "+".join(
                    str(samples[int(parent)])
                    for parent in alternatives[row, 1:]
                    if int(parent) >= 0
                )
                for row in configuration_set_rows
            ),
            "HeldOutPredictiveFoldCount": (
                full_selection.predictive_fold_count
            ),
            "M1OverM0AggregateHeldOutGain": selected_m1_gain,
            "M2OverParent1M1AggregateHeldOutGain": selected_m2_first_gain,
            "M2OverParent2M1AggregateHeldOutGain": selected_m2_second_gain,
            "Interpretation": (
                "composite forward-likelihood model support and internal "
                "chromosome-resampling stability; not calibrated biological "
                "posterior probability"
            ),
        })
        state_call_rows.append({
            "Sample": sample,
            "SelectedParentState": (
                None if graph_state is None else pedigree_models._PARENT_STATE_NAMES[graph_state]
            ),
            "LocalWinnerParentState": (
                None if local_state is None else pedigree_models._PARENT_STATE_NAMES[local_state]
            ),
            "ObservedParentCount": (
                np.nan if graph_state is None else graph_state
            ),
            "LocalObservedParentCount": (
                np.nan if local_state is None else local_state
            ),
            "InferenceStatus": complete_status[child],
            "DAGDisplaced": graph_displaced,
            "GraphConflict": graph_conflict,
            "GraphTieConflict": graph_tie_conflict,
            "GraphDirectionResolvedAlternative": graph_direction_resolved,
            "SelectedParentRoleProbability": (
                selected_parent_role_probability
            ),
            "TierAStateCall": tier_a_state_pass,
            "TierBStateCall": tier_b_state_pass,
        })

        two_rows = child_rows[states[child_rows] == pedigree_models._TWO_OBSERVED]
        ordered_two = two_rows[np.argsort(
            -full_aggregate[two_rows], kind="stable"
        )]
        trio_candidates[sample] = [
            (
                samples[int(alternatives[row, 1])],
                samples[int(alternatives[row, 2])],
                float(full_aggregate[row]),
            )
            for row in ordered_two
        ]
        parent_score = {}
        for row in child_rows[states[child_rows] != pedigree_models._ZERO_OBSERVED]:
            for parent in alternatives[row, 1:]:
                parent = int(parent)
                if parent >= 0:
                    parent_score[parent] = max(
                        parent_score.get(parent, -np.inf),
                        float(full_aggregate[row]),
                    )
        parent_candidates[sample] = [
            (samples[parent], score)
            for parent, score in sorted(
                parent_score.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        trio_scores[sample] = selected_utility

    complete_frame = pedigree_states._parent_state_frame(
        samples,
        alternatives,
        states,
        complete_rows,
        complete_states,
        complete_status,
    )
    tier_a_frame = pedigree_states._parent_state_frame(
        samples,
        alternatives,
        states,
        tier_a_rows,
        tier_a_states,
        tier_a_status,
    )
    tier_b_frame = pedigree_states._parent_state_frame(
        samples,
        alternatives,
        states,
        tier_b_rows,
        tier_b_states,
        tier_b_status,
    )

    def partial_frame(
        exact: pd.DataFrame,
        tier_states: Mapping[int, int],
        tier_rows: Mapping[int, Optional[int]],
        parent_flags: Mapping[int, tuple[bool, bool]],
        label: str,
    ) -> pd.DataFrame:
        partial = exact.copy(deep=True)
        for child, state in tier_states.items():
            if tier_rows.get(child) is not None:
                continue
            local_row = full_selection.local_rows.get(child)
            if local_row is None or full_selection.graph_rows.get(child) != local_row:
                continue
            flags = parent_flags[child]
            parent_indices = [
                int(parent)
                for parent in alternatives[local_row, 1:]
                if int(parent) >= 0
            ]
            retained = 0
            for index, parent in enumerate(parent_indices):
                if index < len(flags) and flags[index]:
                    partial.at[child, f"Parent{index + 1}"] = samples[parent]
                    retained += 1
            if retained:
                partial.at[child, "InferenceStatus"] = (
                    f"{label}_partial_parent_support"
                )
        return partial

    tier_a_partial_frame = partial_frame(
        tier_a_frame,
        tier_a_states,
        tier_a_rows,
        tier_a_parent_flags,
        "tier_a",
    )
    tier_b_partial_frame = partial_frame(
        tier_b_frame,
        tier_b_states,
        tier_b_rows,
        tier_b_parent_flags,
        "tier_b",
    )

    candidate_set_rows = []
    for child, sample in enumerate(samples):
        state = tier_b_states.get(child)
        parent1_set, parent2_set, configuration_rows = (
            evidence_support_sets.get(child, ((), (), ()))
            if state is not None
            else ((), (), ())
        )

        def supported_singleton(indices: tuple[int, ...]) -> Optional[int]:
            if len(indices) != 1:
                return None
            parent = int(indices[0])
            bootstrap = stable_fraction(
                local_parent_counts[child, parent], bootstrap_denominator
            )
            loco = stable_fraction(
                loco_local_parent_counts[child, parent], loco_denominator
            )
            if (
                bootstrap >= settings.tier_b_parent_bootstrap
                and loco >= settings.tier_b_loco_fraction
            ):
                return parent
            return None

        supported_first = supported_singleton(parent1_set)
        supported_second = supported_singleton(parent2_set)
        exact = tier_b_rows.get(child) is not None
        if state is None:
            status = (
                "genetic_m1_directionally_unresolved"
                if scaffold_direction_unresolved[child]
                else "unresolved_below_tier_b_state_support"
            )
        elif exact:
            status = "tier_b_exact_configuration"
        elif supported_first is not None or supported_second is not None:
            status = "tier_b_partial_parent_with_candidate_set"
        elif parent1_set or parent2_set or configuration_rows:
            status = "tier_b_candidate_set_identity_unresolved"
        else:
            status = "tier_b_parent_state_only"
        candidate_set_rows.append({
            "Sample": sample,
            "ParentState": (
                "unresolved" if state is None else pedigree_models._PARENT_STATE_NAMES[state]
            ),
            "ObservedParentCount": (
                np.nan if state is None else int(state)
            ),
            "Parent1": (
                None if supported_first is None else samples[supported_first]
            ),
            "Parent2": (
                None if supported_second is None else samples[supported_second]
            ),
            "Parent1Candidates": tuple(samples[index] for index in parent1_set),
            "Parent2Candidates": tuple(samples[index] for index in parent2_set),
            "ConfigurationCandidates": tuple(
                tuple(
                    samples[int(parent)]
                    for parent in alternatives[row, 1:]
                    if int(parent) >= 0
                )
                for row in configuration_rows
            ),
            "ExactConfigurationResolved": exact,
            "InferenceStatus": status,
        })
    tier_b_candidate_sets = pd.DataFrame(candidate_set_rows)

    primary = {
        "tier_a": tier_a_frame,
        "tier_b": tier_b_frame,
        "complete": complete_frame,
    }[settings.primary_view].copy(deep=True)
    effective_blocks = int(np.sum(np.ceil(
        markers / settings.markers_per_information_block
    )))
    result = pedigree_results.PedigreeResult(
        samples,
        primary,
        parent_candidates,
        None,
        [],
        None,
        None,
        trio_scores=trio_scores,
        total_bins=effective_blocks,
    )
    result.trio_candidate_scores = trio_candidates
    result.mode = True
    result.parent_state_model = True
    result.pair_only_compatibility_mode = False
    result.bootstrap_worker_count = bootstrap_worker_count
    result.bootstrap_depth_refit_count = bootstrap_depth_refits
    result.config = settings
    result.parent_state_candidate_source_mode = (
        settings.parent_state_candidate_source_mode
    )
    result.candidate_source_approximation = (
        "transmitted-marginal-persistence-maxent-quadratic"
    )
    result.parent_state_structure_mode = pedigree_models._PARENT_STATE_METHOD
    result.structure_cx_veto_policy = (
        pedigree_states._resolved_structure_cx_veto_policy(settings)
    )
    result.direction_state_policy = (
        settings.parent_state_direction_state_policy
    )
    result.parent_state_algorithm_mode = pedigree_models._PARENT_STATE_LIKELIHOOD
    result.b3_heldout_state_mask_policy = "none"
    result.b3_aggregate_identity_mask_policy = "none"
    result.tier_a_partial_relationships = tier_a_partial_frame
    result.tier_b_partial_relationships = tier_b_partial_frame
    result.tier_b_candidate_sets = tier_b_candidate_sets
    result.tier_a_relationships = tier_a_frame
    result.tier_b_relationships = tier_b_frame
    result.complete_relationships = complete_frame
    result.parent_state_calls = pd.DataFrame(state_call_rows)
    result.diagnostics = pd.DataFrame(diagnostics)

    result.prior_sensitivity_summary = pd.DataFrame(
        sensitivity_summary_rows
    )
    result.fitted_parent_state_prior_parameters = (
        full_selection.fitted_prior_parameters.copy()
    )
    evidence_summary = {
        "Contig": contig_names,
        "InformativeMarkers": markers.astype(np.int64),
        "AggregationWeight": np.ones(len(contig_names), dtype=np.float64),
    }
    if junction_matrix is not None:
        evidence_summary["MeanMinimumAncestryJunctions"] = np.mean(
            junction_matrix, axis=1
        )
        evidence_summary["MeanCallableHaplotypeBins"] = np.mean(
            callable_matrix, axis=1
        )
    result.evidence_summary = pd.DataFrame(evidence_summary)
    result.predictive_folds = pd.DataFrame()
    result.ancestry_depth_model = full_depth_model
    result.ancestry_depth_model_available = bool(
        full_depth_model is not None
        and full_depth_model.posterior.shape[1] >= 2
    )
    result.parent_eligibility_policy_label = eligibility.policy_name
    result.parent_eligibility_supplied = eligibility.supplied
    result.parent_eligibility_record = (
        pedigree_eligibility._parent_eligibility_result_record(eligibility)
    )
    result.ancestry_depth_model_parameters = {
        "maximum_components": pedigree_states._ANCESTRY_DEPTH_MAX_COMPONENTS,
        "gmm_initializations": pedigree_states._ANCESTRY_DEPTH_GMM_N_INIT,
        "gmm_max_iterations": pedigree_states._ANCESTRY_DEPTH_GMM_MAX_ITERATIONS,
        "standardized_covariance_regularization": (
            pedigree_states._ANCESTRY_DEPTH_GMM_REGULARIZATION
        ),
        "component_selection": "minimum_BIC",
        "direction_decision": (
            "identity_gate_at_or_above_configured_probability_"
            "or_explicit_support"
        ),
        "direction_identity_probability_threshold": float(
            settings.parent_state_minimum_direction_probability
        ),
        "direction_state_policy": (
            settings.parent_state_direction_state_policy
        ),
        "direction_state_reverse_probability_threshold": 1.0 - float(
            settings.parent_state_minimum_direction_probability
        ),
        "graph_direction_use": (
            "continuous_parent_role_probability_ordering"
        ),
    }
    result.ancestry_depth_model_specification = (
        "BIC-selected one-dimensional Gaussian mixture over callability-"
        "adjusted, chromosome-wide phase-invariant minimum founder-trajectory "
        "switch burden; components are ordered relative ancestry layers, not "
        "generation labels. Its posterior supplies per-edge direction "
        "probabilities. Candidate identity requires every observed-parent edge "
        "to meet the configured probability threshold or have explicit caller "
        "support. The configured parent-count policy either leaves state mass "
        "unchanged, restores the strict testable forward-direction gate, or "
        "removes only testable edges with both sub-threshold forward support "
        "and strong reverse support. Missing, single-layer, and zero-posterior "
        "direction evidence is state-neutral. Direction does not enter the "
        "forward B1 likelihood; its probability also orders graph construction "
        "and conflict resolution."
    )
    result.selection_method = (
        "marginal parent-state selection followed by conditional identity; "
        "deterministic ancestry-direction-then-confidence-ordered variable-edge "
        "DAG with coordinate local search. A graph-conflicted local row may use "
        "a unique finite lower-parent-count alternative supported by downward "
        "ancestry direction; otherwise it falls to M0. Local support is measured "
        "before the DAG."
    )
    result.candidate_screening_scope = (
        "M0 and every M1 identity are scored; M2 uses a fixed candidate panel. "
        "The M2 identity prior denominator is the full eligible pair "
        "space, so integrated M2 evidence is an explicit lower bound when the "
        "screen is incomplete."
    )
    result.missing_parent_model = (
        "Normalized forward HMMs compare zero, one, and two observed parents. "
        "An external parent is a linked child-left-out mixture of locally "
        "IBS-pooled reconstructed founder haplotypes; zero observed parents "
        "does not assert biological founder status."
    )
    result.limitations = (
        (
            "No cohort, sex, breeding record, or sample-name eligibility was used; "
            "parent order is arbitrary. "
            if not eligibility.supplied
            else (
                "Candidate eligibility was supplied by caller policy "
                f"{eligibility.policy_name!r}; Smart did not infer eligibility "
                "from cohort, sex, breeding records, or sample names. Eligibility "
                "constraints are design assumptions, not individual parentage "
                "ground truth. Parent order remains arbitrary. "
            )
        )
        + "State support is based on the fixed-prior tempered B1 composite "
        "likelihood, not a calibrated posterior probability. Incomplete-screen "
        "integrated B1 evidence is a lower bound. Relative ancestry depth is "
        "unsupervised and painting-dependent: it is inferred from a conservative "
        "minimum-switch burden, not from known generation, age, or breeding "
        "metadata. Callability tempers each sample's component posterior, but "
        "callability-adjusted burdens still enter mixture fitting and BIC equally; "
        "highly incomplete paintings therefore remain a validation risk. Depth "
        "supplies a configurable candidate-identity direction gate unless caller "
        "eligibility explicitly supports the edge, and its probabilities order "
        "graph construction. Under the configured direction-state policy it can "
        "also exclude testable direction contradictions from parent-count state "
        "mass; unavailable direction remains neutral. It can therefore affect "
        "state calls, resolved identities, the DAG-selected configuration, and "
        "Tier A/B release; it is not an independent likelihood source. Reconstructed "
        "paintings and hard founder alleles "
        "inherit upstream errors; raw genotype likelihoods are not double-"
        "counted as an independent source. Zero observed parents may mean a "
        "top-level individual or unsequenced biological parents and is not by "
        "itself safe founder-recolour eligibility. For API compatibility, trio_scores "
        "contains hierarchical decision utilities and total_bins an effective "
        "information-block count; neither is comparable to legacy Viterbi "
        "scores or raw bin counts."
    )
    result.limitations += (
        " Candidate-source transmission uses the explicit projected "
        "quadratic approximation: ordered-diplotype dependence is collapsed "
        "to transmitted-state marginals with a maximum-entropy bridge."
    )
    return result

import haplotype_reconstruction.pedigree.bootstrap as pedigree_bootstrap
import haplotype_reconstruction.pedigree.config as module_pedigree_config
import haplotype_reconstruction.pedigree.direction as pedigree_direction
import haplotype_reconstruction.pedigree.eligibility as pedigree_eligibility
import haplotype_reconstruction.pedigree.evidence as pedigree_evidence
import haplotype_reconstruction.pedigree.models as pedigree_models
import haplotype_reconstruction.pedigree.results as pedigree_results
import haplotype_reconstruction.pedigree.states as pedigree_states
