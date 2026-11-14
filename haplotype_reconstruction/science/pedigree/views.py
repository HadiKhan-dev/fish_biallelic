"""Compute numeric pedigree report views from the released decision, without rendering.

Owns PD25 and the slot-order and sibling-guard transformations. No DataFrames, storage,
scoring, names or formatted support strings enter these producers.
Method: docs/methods/pedigree.md#published-tables (PD25).
Identity: count overwrite precedes slots; guard filters existing slots; the complete
view and selection diagnostics retain pre-guard values.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np

from ...contracts.pedigree import (
    M0,
    M1,
    M2,
    NO_PARENT,
    UNRESOLVED,
    AlternativeExplanationView,
    CallExplanationView,
    CandidateView,
    ChromosomeSummaryView,
    ConfigurationSupport,
    CountRelease,
    Decision,
    DiagnosticView,
    EvidenceSummaryView,
    ExclusionEvidenceView,
    GuardRecord,
    GuardView,
    GuardViewRow,
    ParentStateView,
    PartialEdges,
    PartialRelationshipView,
    PedigreeViews,
    RelationshipView,
    RelativeContrasts,
    SearchDiagnosticView,
    StructuralCompatibility,
    TierCalls,
)
from ...core.numerics import ordered_sum
from .orientation import parent_role_probability
from .parameters import PedigreeParameters
from .release import support_sides
from .slots import ParentOrientation
from .ties import unique_finite_winner

__all__ = ["build", "orient", "apply_guard", "diagnostics", "candidate_sets", "configuration_support"]


def _index(value: int | None) -> int:
    return NO_PARENT if value is None else int(value)


def _row_parents(rows: np.ndarray, row: int) -> tuple[int, ...]:
    return () if row < 0 else tuple(int(parent) for parent in rows[row, 1:] if parent >= 0)


def _count(state: int) -> float:
    return float("nan") if state == UNRESOLVED else int(state)


def configuration_support(
    child_rows: np.ndarray, counts: np.ndarray, rows: np.ndarray, states: np.ndarray, coverage: float
) -> tuple[ConfigurationSupport, ...]:
    """Select support-set rows by stable count order, stopping after coverage (PD25)."""
    total = float(np.sum(counts[child_rows]))
    if total <= 0.0:
        return ()
    result = []
    cumulative = 0.0
    for row in child_rows[np.argsort(-counts[child_rows], kind="stable")]:
        probability = float(counts[row] / total)
        result.append(ConfigurationSupport(int(states[row]), _row_parents(rows, int(row)), probability))
        cumulative += probability
        if cumulative >= coverage:
            break
    return tuple(result)


def _structure(decision: Decision, config: PedigreeParameters) -> StructuralCompatibility:
    """Full-data C/X diagnostics: edge coverage [E] per eligible edge (GenomeScores.edges), pair
    explainability [T], C/X-compatible and exposure-testable rows [A], children with a testable M1 row [N]."""
    scores, alternatives = decision.scores, decision.alternatives
    edge_matched = ordered_sum(scores.edge_matched, axis=0)
    edge_exposed = ordered_sum(scores.edge_exposed, axis=0)
    pair_explained = ordered_sum(scores.pair_explained, axis=0)
    pair_exposed = ordered_sum(scores.pair_exposed, axis=0)
    coverage = np.divide(edge_matched, edge_exposed, out=np.full_like(edge_matched, np.nan), where=edge_exposed > 0.0)
    explainability = np.divide(
        pair_explained, pair_exposed, out=np.full_like(pair_explained, np.nan), where=pair_exposed > 0.0
    )
    # By key, as every trace mapping (records and registered fixtures alike) supports; the testable flags are bool [A].
    testable = cast(np.ndarray, decision.trace["testable"])
    rows, states = alternatives.rows, alternatives.states
    row_edges = decision.inputs.exposure.row_edges
    compatible = np.zeros(len(rows), dtype=np.bool_)
    compatible[states == M0] = True
    m1 = np.flatnonzero(states == M1)
    compatible[m1] = testable[m1] & (coverage[row_edges[m1, 0]] >= config.exposure.report_edge_coverage)
    m2 = np.flatnonzero(states == M2)
    compatible[m2] = (
        testable[m2]
        & (coverage[row_edges[m2, 0]] >= config.exposure.report_edge_coverage)
        & (coverage[row_edges[m2, 1]] >= config.exposure.report_edge_coverage)
        & (explainability[alternatives.panel_row[m2]] >= config.exposure.report_pair_explainability)
    )
    evaluable = np.zeros(alternatives.n_samples, dtype=np.bool_)
    np.logical_or.at(evaluable, rows[m1, 0], testable[m1])
    return StructuralCompatibility(coverage, explainability, compatible, testable, evaluable)


def diagnostics(decision: Decision, config: PedigreeParameters) -> tuple[DiagnosticView, ...]:
    """One row per sample: selections, margins, support fractions, structure and tier flags."""
    full, alternatives, evidence = decision.full, decision.alternatives, decision.evidence
    rows, states = alternatives.rows, alternatives.states
    sample_ids = decision.scores.sample_ids
    boot = decision.resampling.bootstrap
    coverage, explainability, compatible, testable, evaluable = _structure(decision, config)
    row_edges = decision.inputs.exposure.row_edges
    junctions, callable_bins = decision.scores.junctions, decision.scores.callable_bins
    eligible_children = decision.eligibility.children
    records = []
    for child in range(len(sample_ids)):
        child_rows = alternatives.child_rows[child]
        best_m1, best_m1_margin = unique_finite_winner(child_rows[states[child_rows] == M1], full.decision_scores)
        best_m2, best_m2_margin = unique_finite_winner(child_rows[states[child_rows] == M2], full.decision_scores)
        local_state, local_row = int(full.local_state[child]), int(full.local_row[child])
        graph_row = int(full.graph_row[child])
        graph_state = int(evidence.graph_state[child])
        selected = _row_parents(rows, local_row)
        graph_parents = _row_parents(rows, graph_row)
        best_m2_parents = _row_parents(rows, -1 if best_m2 is None else best_m2) + (None, None)
        sensitivity = decision.sensitivity

        def agreement(values, child=child):
            return np.nan if not eligible_children[child] else float(np.mean(values))

        supports = [
            selection.state_support[child, local_state] for _, selection in sensitivity if local_state != UNRESOLVED
        ]
        selected_utility = np.nan if graph_row < 0 else float(full.decision_scores[graph_row])
        best = -np.inf if not len(child_rows) else float(np.max(full.decision_scores[child_rows]))
        state_margin = np.nan if local_state == UNRESOLVED else float(full.state_margin[child])
        identity_margin = np.nan if local_row < 0 else float(full.identity_margin[child])
        first_set, second_set, configuration_rows = support_sides(
            child_rows,
            local_state,
            local_row,
            alternatives,
            full.decision_scores,
            config.release.support_set_coverage,
            reverse=config.reverse_conventions,
        )
        parent_coverage = [np.nan, np.nan]
        parent_direction = [np.nan, np.nan]
        pair_explainability = np.nan
        if local_row >= 0 and selected:
            for slot, parent in enumerate(selected):
                parent_coverage[slot] = coverage[row_edges[local_row, slot]]
                parent_direction[slot] = decision.orientation.order_probability[child, parent]
            if len(selected) > 1 and alternatives.panel_row[local_row] >= 0:
                pair_explainability = explainability[alternatives.panel_row[local_row]]
        tier_a, tier_b = decision.tier_a, decision.tier_b
        # Tier views always carry their pass flags (only the complete view leaves them None).
        assert tier_a.state_pass is not None and tier_a.identity_pass is not None
        assert tier_b.state_pass is not None and tier_b.identity_pass is not None
        records.append(
            DiagnosticView(
                sample=child,
                m0_prior_sensitivity_state_agreement=bool(evidence.m0_state_agreement[child]),
                m0_prior_sensitivity_graph_agreement=bool(evidence.m0_graph_agreement[child]),
                m0_prior_sensitivity_tier_veto=bool(evidence.m0_veto[child]),
                structure_child_evaluable=bool(evaluable[child]),
                structure_selected_row_exposure_testable=bool(local_row >= 0 and testable[local_row]),
                structure_selected_row_cx_compatible=bool(local_row >= 0 and compatible[local_row]),
                structure_parent1_coverage=parent_coverage[0],
                structure_parent2_coverage=parent_coverage[1],
                structure_pair_explainability=pair_explainability,
                structure_parent1_direction_probability=parent_direction[0],
                structure_parent2_direction_probability=parent_direction[1],
                best_m1_parent=NO_PARENT if best_m1 is None else int(rows[best_m1, 1]),
                best_m1_conditional_identity_margin=best_m1_margin,
                best_m2_parent1=_index(best_m2_parents[0]),
                best_m2_parent2=_index(best_m2_parents[1]),
                best_m2_conditional_identity_margin=best_m2_margin,
                local_parent1=_index((selected + (None,))[0]),
                local_parent2=_index((selected + (None, None))[1]),
                eligible_child=bool(eligible_children[child]),
                eligible_parent_count=int(alternatives.full_counts[child, 1]),
                eligible_parent_pair_count=int(alternatives.full_counts[child, 2]),
                complete_parent1=_index((graph_parents + (None,))[0]),
                complete_parent2=_index((graph_parents + (None, None))[1]),
                selected_parent_state=graph_state,
                local_winner_parent_state=local_state,
                observed_parent_count=np.nan if graph_state == UNRESOLVED else graph_state,
                local_observed_parent_count=np.nan if local_state == UNRESOLVED else local_state,
                identifiable=bool(local_row >= 0 and graph_row == local_row),
                dag_displaced=bool(local_row >= 0 and graph_row >= 0 and graph_row != local_row),
                graph_conflict=bool(evidence.graph_conflict[child]),
                graph_tie_conflict=bool(full.tie_conflict[child]),
                graph_direction_resolved_alternative=bool(full.direction_resolved[child]),
                local_parent_role_probability=(
                    np.nan if local_row < 0 else parent_role_probability(local_row, rows, decision.orientation)
                ),
                selected_parent_role_probability=full.role_probability[child],
                raw_ancestry_junction_count=float(np.sum(junctions[:, child])),
                callable_ancestry_haplotype_bin_count=float(np.sum(callable_bins[:, child])),
                adjusted_ancestry_junction_burden=float(decision.orientation.adjusted_burden[child]),
                ancestry_painting_callability_fraction=float(decision.orientation.callability[child]),
                informative_unit_count=int(decision.informative[child]),
                informative_morgans=float(decision.informative_morgans[child]),
                local_state_bootstrap_fraction=evidence.state_bootstrap[child],
                local_configuration_bootstrap_fraction=evidence.configuration_bootstrap[child],
                graph_configuration_bootstrap_fraction=evidence.configuration_graph_bootstrap[child],
                selected_graph_configuration_bootstrap_fraction=evidence.graph_configuration_bootstrap[child],
                selected_graph_parent1_bootstrap_fraction=evidence.graph_parent_bootstrap[child, 0],
                selected_graph_parent2_bootstrap_fraction=evidence.graph_parent_bootstrap[child, 1],
                parent1_bootstrap_fraction=evidence.parent_bootstrap[child, 0],
                parent2_bootstrap_fraction=evidence.parent_bootstrap[child, 1],
                local_state_lobo_fraction=evidence.state_loco[child],
                local_configuration_lobo_fraction=evidence.configuration_loco[child],
                graph_configuration_lobo_fraction=evidence.configuration_graph_loco[child],
                selected_graph_configuration_lobo_fraction=evidence.graph_configuration_loco[child],
                selected_graph_parent1_lobo_fraction=evidence.graph_parent_loco[child, 0],
                selected_graph_parent2_lobo_fraction=evidence.graph_parent_loco[child, 1],
                parent1_lobo_fraction=evidence.parent_loco[child, 0],
                parent2_lobo_fraction=evidence.parent_loco[child, 1],
                state_winner_margin=state_margin,
                conditional_identity_margin=identity_margin,
                unconstrained_winner_margin=(
                    min(state_margin, identity_margin) if np.isfinite(identity_margin) else state_margin
                ),
                selected_aggregate_utility=selected_utility,
                selected_minus_unconstrained_best=(np.nan if graph_row < 0 else selected_utility - best),
                state_log_evidence0=full.state_log_evidence[child, 0],
                state_log_evidence1=full.state_log_evidence[child, 1],
                state_log_evidence2=full.state_log_evidence[child, 2],
                state_support0=full.state_support[child, 0],
                state_support1=full.state_support[child, 1],
                state_support2=full.state_support[child, 2],
                scored_candidate_count2=alternatives.scored_counts[child, 2],
                m2_candidate_screen_incomplete=bool(
                    alternatives.scored_counts[child, 2] < alternatives.full_counts[child, 2]
                ),
                prior_sensitivity_local_state_agreement_fraction=agreement(
                    [selection.local_state[child] == local_state for _, selection in sensitivity]
                ),
                prior_sensitivity_local_identity_agreement_fraction=agreement(
                    [selection.local_row[child] == local_row for _, selection in sensitivity]
                ),
                prior_sensitivity_graph_agreement_fraction=agreement(
                    [selection.graph_row[child] == graph_row for _, selection in sensitivity]
                ),
                prior_sensitivity_selected_state_minimum_support=(np.nan if not supports else float(np.min(supports))),
                inference_status=decision.complete.status[child],
                tier_a_state_call=bool(tier_a.state_pass[child]),
                tier_b_state_call=bool(tier_b.state_pass[child]),
                tier_a_exact_configuration=bool(tier_a.identity_pass[child]),
                tier_b_exact_configuration=bool(tier_b.identity_pass[child]),
                tier_a_exact_pair=bool(tier_a.identity_pass[child] and local_state == M2),
                tier_b_exact_pair=bool(tier_b.identity_pass[child] and local_state == M2),
                tier_a_parent1=bool(decision.partial_a.supported[child, 0]),
                tier_a_parent2=bool(decision.partial_a.supported[child, 1]),
                tier_b_parent1=bool(decision.partial_b.supported[child, 0]),
                tier_b_parent2=bool(decision.partial_b.supported[child, 1]),
                configuration_support_set=configuration_support(
                    child_rows, boot.local_configuration, rows, states, config.release.support_set_coverage
                ),
                parent1_candidate_set=tuple(int(i) for i in first_set),
                parent2_candidate_set=tuple(int(i) for i in second_set),
                evidence_configuration_set=tuple(_row_parents(rows, row) for row in configuration_rows),
            )
        )
    return tuple(records)


def relationships(calls: TierCalls, rows: np.ndarray) -> tuple[RelationshipView, ...]:
    """Materialize full relationship rows in sample order; unresolved counts are NaN."""
    result = []
    for child, state in enumerate(calls.state):
        parents = _row_parents(rows, int(calls.row[child])) + (NO_PARENT, NO_PARENT)
        result.append(
            RelationshipView(child, parents[0], parents[1], int(state), _count(int(state)), calls.status[child])
        )
    return tuple(result)


def partial_relationships(partial: PartialEdges, release_evidence: bool) -> tuple[PartialRelationshipView, ...]:
    """Materialize marginal parents, support, exactness and count bounds without mutation."""
    return tuple(
        PartialRelationshipView(
            child,
            int(partial.parents[child, 0]),
            int(partial.parents[child, 1]),
            int(state),
            _count(int(state)),
            partial.status[child],
            bool(partial.supported[child, 0]),
            bool(partial.supported[child, 1]),
            bool(partial.exact[child]),
            int(partial.count_minimum[child]),
            int(partial.count_maximum[child]),
            release_evidence=partial.release_evidence[child] if release_evidence else None,
        )
        for child, state in enumerate(partial.state)
    )


def candidate_sets(decision: Decision, release: CountRelease) -> tuple[CandidateView, ...]:
    """Apply the final count release to candidate rows, preserving the M0 configuration ((),)."""
    sets, partial = decision.candidates, decision.partial_b
    final, final_partial = release.tier_b, release.partial_b
    result = []
    for child, state in enumerate(decision.tier_b.state):
        first, second = sets.side_candidates[child]
        record = CandidateView(
            child,
            int(state),
            _count(int(state)),
            int(sets.parents[child, 0]),
            int(sets.parents[child, 1]),
            tuple(int(i) for i in first),
            tuple(int(i) for i in second),
            tuple(tuple(int(i) for i in configuration) for configuration in sets.configurations[child]),
            bool(sets.exact[child]),
            sets.status[child],
            bool(partial.supported[child, 0]),
            bool(partial.supported[child, 1]),
            int(partial.count_minimum[child]),
            int(partial.count_maximum[child]),
        )
        if release.released[child]:
            kept = tuple(int(i) for i in release.kept[child])
            record = replace(
                record,
                parent_state=int(final.state[child]),
                observed_parent_count=_count(int(final.state[child])),
                parent1=kept[0] if kept else NO_PARENT,
                parent2=NO_PARENT,
                inference_status=final.status[child],
                exact_configuration_resolved=True,
                parent1_candidates=kept,
                parent2_candidates=(),
                configuration_candidates=(kept,),
                parent1_supported=bool(final_partial.supported[child, 0]),
                parent2_supported=bool(final_partial.supported[child, 1]),
                observed_parent_count_minimum=int(final_partial.count_minimum[child]),
                observed_parent_count_maximum=int(final_partial.count_maximum[child]),
            )
        result.append(record)
    return tuple(result)


def explanations(
    decision: Decision,
    candidates: tuple[CandidateView, ...],
    diagnostic: tuple[DiagnosticView, ...],
    release: CountRelease,
) -> tuple[tuple[CallExplanationView, ...], tuple[AlternativeExplanationView, ...]]:
    """Select local/graph and leading three finite alternatives per state in configuration row order."""
    full, alternatives = decision.full, decision.alternatives
    rows, states = alternatives.rows, alternatives.states
    state_scores, identity_scores = decision.adjusted
    terms = decision.trace
    summaries, records = [], []
    for child, child_rows in enumerate(alternatives.child_rows):
        local, graph = int(full.local_row[child]), int(full.graph_row[child])
        chosen = {row for row in (local, graph) if row >= 0}
        ranks = {}
        for state in range(3):
            group = child_rows[(states[child_rows] == state) & np.isfinite(identity_scores[child_rows])]
            order = group[np.argsort(-identity_scores[group], kind="stable")]
            chosen.update(map(int, order[:3]))
            ranks.update({int(row): rank + 1 for rank, row in enumerate(order)})
        for row in sorted(chosen):
            state = int(states[row])
            records.append(
                AlternativeExplanationView(
                    child,
                    state,
                    int(rows[row, 1]),
                    int(rows[row, 2]),
                    row == local,
                    row == graph,
                    ranks.get(row),
                    terms.genetic[row],
                    terms.structural_state[row],
                    bool(np.isfinite(terms.structural_identity[row])),
                    terms.direction[row],
                    terms.reciprocal_family[row],
                    terms.ancestry_paths[row],
                    state_scores[row],
                    identity_scores[row],
                    np.log(full.priors[child, state]),
                    np.log(max(1, alternatives.full_counts[child, state])),
                    full.state_log_evidence[child, state],
                    full.decision_scores[row],
                )
            )
        candidate, diag = candidates[child], diagnostic[child]
        summaries.append(
            CallExplanationView(
                candidate.sample,
                candidate.parent_state,
                candidate.observed_parent_count,
                candidate.parent1,
                candidate.parent2,
                candidate.parent1_candidates,
                candidate.parent2_candidates,
                candidate.configuration_candidates,
                candidate.exact_configuration_resolved,
                candidate.inference_status,
                candidate.parent1_supported,
                candidate.parent2_supported,
                candidate.observed_parent_count_minimum,
                candidate.observed_parent_count_maximum,
                diag.local_winner_parent_state,
                diag.selected_parent_state,
                diag.local_parent1,
                diag.local_parent2,
                diag.graph_conflict,
                diag.state_winner_margin,
                diag.conditional_identity_margin,
                diag.local_state_bootstrap_fraction,
                diag.local_state_lobo_fraction,
                int(alternatives.full_counts[child, 2]),
                int(alternatives.scored_counts[child, 2]),
                bool(alternatives.scored_counts[child, 2] < alternatives.full_counts[child, 2]),
                graph != local,
                (
                    "fixed-observed-genome exclusion conditional on supported family directions; original resampling "
                    "stability is unchanged; M0 is not founder generation"
                )
                if release.released[child]
                else "composite model support; M0 means no observed parent, not founder generation",
            )
        )
    return tuple(summaries), tuple(records)


def final_state_calls(release: CountRelease, rows: np.ndarray) -> np.ndarray:
    """A final state is called only when its observed-parent count equals the released row count."""
    final = release.tier_b
    return np.asarray(
        [
            int(state) != UNRESOLVED and int(state) == len(_row_parents(rows, int(final.row[child])))
            for child, state in enumerate(final.state)
        ],
        dtype=np.bool_,
    )


def build(
    decision: Decision,
    release: CountRelease,
    search: tuple[SearchDiagnosticView, ...],
    chromosome_summary: tuple[ChromosomeSummaryView, ...],
    config: PedigreeParameters,
    *,
    relative_contrasts: RelativeContrasts | None = None,
) -> PedigreeViews:
    """Compute all pre-slot numeric views from supplied decision/reports; no input is mutated."""
    rows = decision.alternatives.rows
    final = final_state_calls(release, rows)
    evidence = release.partial_b.release_evidence
    diagnostic = diagnostics(decision, config)
    candidates = candidate_sets(decision, release)
    call_rows, alternative_rows = explanations(decision, candidates, diagnostic, release)
    diagnostic = tuple(
        replace(r, tier_b_final_state_call=bool(final[r.sample]), tier_b_release_evidence=evidence[r.sample])
        for r in diagnostic
    )
    candidates = tuple(
        replace(r, tier_b_final_state_call=bool(final[r.sample]), tier_b_release_evidence=evidence[r.sample])
        for r in candidates
    )
    call_rows = tuple(
        replace(r, tier_b_final_state_call=bool(final[r.sample]), tier_b_release_evidence=evidence[r.sample])
        for r in call_rows
    )
    state_calls = tuple(
        ParentStateView(
            r.sample,
            r.selected_parent_state,
            r.local_winner_parent_state,
            r.observed_parent_count,
            r.local_observed_parent_count,
            r.inference_status,
            r.dag_displaced,
            r.graph_conflict,
            r.graph_tie_conflict,
            r.graph_direction_resolved_alternative,
            r.selected_parent_role_probability,
            r.tier_a_state_call,
            r.tier_b_state_call,
            r.tier_b_final_state_call,
            r.tier_b_release_evidence,
        )
        for r in diagnostic
    )
    dyads = np.asarray(release.dyads, dtype=np.int64).reshape(-1, 2)
    lower = np.array(release.log_e, dtype=np.float64)
    observed = np.array(release.observed_sites, dtype=np.int64)
    if release.log_e_upper is None:
        exclusion = ExclusionEvidenceView(dyads[:, 0].copy(), dyads[:, 1].copy(), lower, observed)
    else:
        upper = np.array(release.log_e_upper, dtype=np.float64)
        exact = lower == upper
        exclusion = ExclusionEvidenceView(
            dyads[:, 0].copy(), dyads[:, 1].copy(), np.where(exact, lower, np.nan), observed, lower, upper, exact
        )
    scores = decision.scores
    if scores.contig_junctions is not None and scores.contig_callable_bins is not None:
        # Per contig from its pieces, before they are summed into (possibly straddling) units.
        assert scores.contig_markers is not None
        evidence_summary = tuple(
            EvidenceSummaryView(c, int(scores.contig_markers[i]), float(np.mean(scores.contig_junctions[i])),
                                float(np.mean(scores.contig_callable_bins[i])))
            for i, c in enumerate(scores.real_contigs))
    else:
        mean_junctions = np.mean(scores.junctions, axis=1)
        mean_callable = np.mean(scores.callable_bins, axis=1)
        evidence_summary = tuple(
            EvidenceSummaryView(c, int(scores.informative_markers[i]), float(mean_junctions[i]),
                                float(mean_callable[i]))
            for i, c in enumerate(scores.contigs)
        )
    return PedigreeViews(
        tuple(replace(r, release_evidence=evidence[r.sample]) for r in relationships(release.tier_b, rows)),
        partial_relationships(release.partial_b, True),
        relationships(decision.tier_a, rows),
        partial_relationships(decision.partial_a, False),
        relationships(decision.complete, rows),
        candidates,
        state_calls,
        diagnostic,
        call_rows,
        alternative_rows,
        search,
        exclusion,
        release.records,
        evidence_summary,
        chromosome_summary,
        relative_contrasts,
        None,
        release.log_e_upper is not None,
    )


def _orient_relationship(row: RelationshipView, orientation: ParentOrientation) -> RelationshipView:
    pair, basis = orientation.order(row.parent1, row.parent2)
    return replace(row, parent1=pair[0], parent2=pair[1], parent_slot_basis=basis)


def _orient_partial(row: PartialRelationshipView, orientation: ParentOrientation) -> PartialRelationshipView:
    pair, basis = orientation.order(row.parent1, row.parent2)
    moved = pair != (row.parent1, row.parent2)
    return replace(
        row,
        parent1=pair[0],
        parent2=pair[1],
        parent_slot_basis=basis,
        parent1_supported=row.parent2_supported if moved else row.parent1_supported,
        parent2_supported=row.parent1_supported if moved else row.parent2_supported,
    )


def _orient_candidate(row: CandidateView, orientation: ParentOrientation) -> CandidateView:
    pair, basis = orientation.order(row.parent1, row.parent2)
    moved = pair != (row.parent1, row.parent2)
    return replace(
        row,
        parent1=pair[0],
        parent2=pair[1],
        parent_slot_basis=basis,
        parent1_supported=row.parent2_supported if moved else row.parent1_supported,
        parent2_supported=row.parent1_supported if moved else row.parent2_supported,
        parent1_candidates=row.parent2_candidates if moved else row.parent1_candidates,
        parent2_candidates=row.parent1_candidates if moved else row.parent2_candidates,
    )


def orient(views: PedigreeViews, orientation: ParentOrientation) -> PedigreeViews:
    """Orient only the six relationship/candidate views, preserving every paired slot column."""
    return replace(
        views,
        tier_b_relationships=tuple(_orient_relationship(r, orientation) for r in views.tier_b_relationships),
        tier_a_relationships=tuple(_orient_relationship(r, orientation) for r in views.tier_a_relationships),
        complete_relationships=tuple(_orient_relationship(r, orientation) for r in views.complete_relationships),
        tier_b_partial_relationships=tuple(_orient_partial(r, orientation) for r in views.tier_b_partial_relationships),
        tier_a_partial_relationships=tuple(_orient_partial(r, orientation) for r in views.tier_a_partial_relationships),
        tier_b_candidate_sets=tuple(_orient_candidate(r, orientation) for r in views.tier_b_candidate_sets),
    )


def _guard_relationship(row: RelationshipView, children: frozenset[int], status: str) -> RelationshipView:
    if row.sample not in children:
        return row
    return replace(
        row,
        parent1=NO_PARENT,
        parent2=NO_PARENT,
        relationship_basis="",
        parents_before_guard=(row.parent1, row.parent2),
        parent_state=UNRESOLVED,
        observed_parent_count=float("nan"),
        inference_status=status,
        parent_slot_basis="none",
        release_evidence="sibling_guard" if row.release_evidence is not None else None,
    )


def _guard_partial(
    row: PartialRelationshipView, withheld: frozenset[tuple[int, int]], children: frozenset[int], status: str
) -> PartialRelationshipView:
    if row.sample not in children:
        return row
    first = (row.parent1, row.sample) in withheld
    second = (row.parent2, row.sample) in withheld
    parent1, parent2 = NO_PARENT if first else row.parent1, NO_PARENT if second else row.parent2
    supported1, supported2 = row.parent1_supported and not first, row.parent2_supported and not second
    return replace(
        row,
        parent1=parent1,
        parent2=parent2,
        parents_before_guard=(row.parent1, row.parent2),
        parent1_supported=supported1,
        parent2_supported=supported2,
        observed_parent_count_minimum=int(supported1 and parent1 >= 0) + int(supported2 and parent2 >= 0),
        observed_parent_count_maximum=2,
        exact_configuration_resolved=False,
        parent_state=UNRESOLVED,
        observed_parent_count=float("nan"),
        inference_status=status,
        parent_slot_basis="none" if parent1 < 0 and parent2 < 0 else row.parent_slot_basis,
        release_evidence="sibling_guard" if row.release_evidence is not None else None,
    )


def _guard_candidate(
    row: CandidateView, withheld: frozenset[tuple[int, int]], children: frozenset[int]
) -> CandidateView:
    if row.sample not in children:
        return row
    first = (row.parent1, row.sample) in withheld
    second = (row.parent2, row.sample) in withheld
    parent1, parent2 = NO_PARENT if first else row.parent1, NO_PARENT if second else row.parent2
    supported1, supported2 = row.parent1_supported and not first, row.parent2_supported and not second
    # Candidate tuples deliberately retain the withheld identities (PD23).
    return replace(
        row,
        parent1=parent1,
        parent2=parent2,
        parents_before_guard=(row.parent1, row.parent2),
        parent1_supported=supported1,
        parent2_supported=supported2,
        observed_parent_count_minimum=int(supported1 and parent1 >= 0) + int(supported2 and parent2 >= 0),
        observed_parent_count_maximum=2,
        exact_configuration_resolved=False,
        parent_state=UNRESOLVED,
        observed_parent_count=float("nan"),
        inference_status="tier_b_sibling_guard_withheld",
        parent_slot_basis="none" if parent1 < 0 and parent2 < 0 else row.parent_slot_basis,
        tier_b_final_state_call=False,
        tier_b_release_evidence="sibling_guard",
    )


def apply_guard(views: PedigreeViews, guard: GuardRecord | GuardView) -> PedigreeViews:
    """Withdraw affected calls and slots while keeping candidate lists and pre-guard diagnostics."""
    guard = guard_view(guard) if isinstance(guard, GuardRecord) else guard
    withheld = frozenset((r.parent, r.child) for r in guard.rows if r.call == "withheld")
    children = frozenset(c for _, c in withheld)
    basis = {(r.parent, r.child): r.relationship_basis for r in guard.rows if r.call != "withheld"}

    def relationship_basis(row):
        parents = [
            p
            for slot, p in enumerate((row.parent1, row.parent2))
            if p >= 0
            and (
                not isinstance(row, PartialRelationshipView)
                or (row.parent1_supported if slot == 0 else row.parent2_supported)
            )
        ]
        if not parents:
            return replace(row, relationship_basis="")
        value = "family" if all(basis.get((p, row.sample)) == "family" for p in parents) else "pair_only"
        return replace(row, relationship_basis=value)

    return replace(
        views,
        tier_b_relationships=tuple(
            relationship_basis(_guard_relationship(r, children, "tier_b_sibling_guard_withheld"))
            for r in views.tier_b_relationships
        ),
        tier_a_relationships=tuple(
            relationship_basis(_guard_relationship(r, children, "tier_a_sibling_guard_withheld"))
            for r in views.tier_a_relationships
        ),
        tier_b_partial_relationships=tuple(
            relationship_basis(_guard_partial(r, withheld, children, "tier_b_sibling_guard_withheld"))
            for r in views.tier_b_partial_relationships
        ),
        tier_a_partial_relationships=tuple(
            relationship_basis(_guard_partial(r, withheld, children, "tier_a_sibling_guard_withheld"))
            for r in views.tier_a_partial_relationships
        ),
        tier_b_candidate_sets=tuple(_guard_candidate(r, withheld, children) for r in views.tier_b_candidate_sets),
        parent_state_calls=tuple(
            replace(r, tier_b_final_state_call=False, tier_b_release_evidence="sibling_guard")
            if r.sample in children
            else r
            for r in views.parent_state_calls
        ),
        sibling_guard=guard,
    )


def guard_view(guard: GuardRecord) -> GuardView:
    """Project the final relationship evidence and fixed-point receipt."""
    return GuardView(
        tuple(
            GuardViewRow(
                v.family.parent,
                v.family.child,
                len(v.family.tested),
                v.set_aside,
                v.against,
                v.supporting,
                v.pair_score,
                v.family_score,
                v.score,
                v.call,
                v.evidence_call,
                v.withdrawal_iteration,
                v.withholding_basis,
                v.relationship_basis,
            )
            for v in guard.votes
        ),
        guard.cleaning_iterations,
        guard.cleaning_converged,
        guard.cleaning_iteration_bound,
    )
