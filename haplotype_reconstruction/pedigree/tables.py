"""Report tables of the pedigree product (one DataFrame per CSV; written by the reporting layer).

Relationship views (tier_b, tier_b_partial, tier_a, tier_a_partial, complete), Tier-B candidate sets,
per-child diagnostics and state calls, call and alternative explanations, the screen boundary, the
full-marker exclusion evidence and decisions, and per-chromosome summaries. Scores are composite
log evidence, not calibrated probabilities; M0 means no observed parent, not founder generation.

Columns removed relative to 92b3daf (constant, duplicate or relict): relationships Generation;
diagnostics ParentStateAlgorithmMode, ParentStateStructureMode, StructureCXVetoPolicy,
ParentStateDirectionStatePolicy, DirectionStateRejectedAlternativeCount (always 0),
StructureSelectedRow{Selection,State}Compatible / {IdentityEligible,Eligible} (= ExposureTestable),
StructureParent{1,2}DirectionSupported (always true), CohortStatePriorUsed, BestM1Row, BestM2Row
(internal row numbers), EligibilityPolicy, ParentOrderMeaning, GraphFallbackPolicy,
LatentAncestryDepth* (cluster model), DirectionModel, FamilyMessagePasses, FamilyReconciliation,
AncestryPathBudget, AncestryDepthResampling, PairBootstrapFraction / PairLOCOFraction (= local
configuration fractions), LOOStatePrior0-2 (fixed priors), ScoredCandidateCount0/1, FullCandidateCount0-2,
CandidatePairCount, FullCandidatePairCount (= EligibleChild / EligibleParent(Pair)Count /
ScoredCandidateCount2), M2StateEvidenceIsLowerBound (= M2CandidateScreenIncomplete),
M2PredictiveScreenLowerBoundGuarantee, B3*, UnresolvedReason (= InferenceStatus), PairSupportSet (=
ConfigurationSupportSet), HeldOutPredictiveFoldCount, *AggregateHeldOutGain, Interpretation;
evidence_summary AggregationWeight; search_diagnostics Note.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PedigreeConfig
from .decision import Decision
from .exclusion import CountRelease
from .orientation import parent_role_probability
from .records import M0, M1, M2, STATE_NAMES, UNRESOLVED, PedigreeReport
from .release import PartialEdges, TierCalls, support_sides
from .ties import unique_finite_winner

EXCLUSION_DIAGNOSTIC_COLUMNS = (
    "Sample", "EligibleCandidates", "CompatibleCandidates", "SupportedReverseCandidates",
    "RemainingCandidates", "WeakestExclusionLogE", "ThresholdLogE", "CompleteCandidateEvidence",
    "Released", "ReleaseConditionalOnSupportedDirections")


def _name(sample_ids, index) -> str | None:
    return None if index is None or index < 0 else sample_ids[int(index)]


def _state_name(state: int) -> str:
    return "unresolved" if state == UNRESOLVED else STATE_NAMES[state]


def _count(state: int):
    return np.nan if state == UNRESOLVED else int(state)


def _row_parents(rows: np.ndarray, row: int) -> tuple[int, ...]:
    return () if row < 0 else tuple(int(parent) for parent in rows[row, 1:] if parent >= 0)


def relationships(calls: TierCalls, rows: np.ndarray, sample_ids: tuple[str, ...]) -> pd.DataFrame:
    """Sample, Parent1, Parent2, ParentState, ObservedParentCount, InferenceStatus."""
    records = []
    for child, sample in enumerate(sample_ids):
        parents = _row_parents(rows, int(calls.row[child])) + (None, None)
        state = int(calls.state[child])
        records.append({"Sample": sample, "Parent1": _name(sample_ids, parents[0]),
                        "Parent2": _name(sample_ids, parents[1]), "ParentState": _state_name(state),
                        "ObservedParentCount": _count(state), "InferenceStatus": calls.status[child]})
    return pd.DataFrame(records)


def partial_relationships(partial: PartialEdges, sample_ids: tuple[str, ...],
                          release_evidence: bool) -> pd.DataFrame:
    """The marginal-edge view: released edges per slot with count bounds."""
    records = []
    for child, sample in enumerate(sample_ids):
        state = int(partial.state[child])
        record = {"Sample": sample, "Parent1": _name(sample_ids, partial.parents[child, 0]),
                  "Parent2": _name(sample_ids, partial.parents[child, 1]),
                  "ParentState": _state_name(state), "ObservedParentCount": _count(state),
                  "InferenceStatus": partial.status[child],
                  "Parent1Supported": bool(partial.supported[child, 0]),
                  "Parent2Supported": bool(partial.supported[child, 1]),
                  "ExactConfigurationResolved": bool(partial.exact[child]),
                  "ObservedParentCountMinimum": int(partial.count_minimum[child]),
                  "ObservedParentCountMaximum": int(partial.count_maximum[child])}
        if release_evidence:
            record["ReleaseEvidence"] = partial.release_evidence[child]
        records.append(record)
    return pd.DataFrame(records)


def _support_text(child_rows, counts, rows, states, sample_ids, coverage) -> str:
    total = float(np.sum(counts[child_rows]))
    if total <= 0.0:
        return ""
    pieces = []
    cumulative = 0.0
    for row in child_rows[np.argsort(-counts[child_rows], kind="stable")]:
        probability = float(counts[row] / total)
        parents = [str(sample_ids[parent]) for parent in rows[row, 1:] if parent >= 0]
        label = "+".join(parents) if parents else "external+external"
        pieces.append(f"{STATE_NAMES[int(states[row])]}:{label}:{probability:.3f}")
        cumulative += probability
        if cumulative >= coverage:
            break
    return ";".join(pieces)


def _set_text(indices, sample_ids) -> str:
    return "{" + ",".join(str(sample_ids[index]) for index in indices) + "}"


def _structure(decision: Decision, config: PedigreeConfig):
    """Full-data C/X diagnostics: edge coverage [N, N], pair explainability [T], C/X-compatible and
    exposure-testable rows [A], children with a testable M1 row [N]."""
    scores, alternatives = decision.scores, decision.alternatives
    weights = np.ones(len(scores.contigs))
    edge_matched = np.tensordot(weights, scores.edge_matched, axes=(0, 0))
    edge_exposed = np.tensordot(weights, scores.edge_exposed, axes=(0, 0))
    pair_explained = weights @ scores.pair_explained
    pair_exposed = weights @ scores.pair_exposed
    coverage = np.divide(edge_matched, edge_exposed, out=np.full_like(edge_matched, np.nan),
                         where=edge_exposed > 0.0)
    explainability = np.divide(pair_explained, pair_exposed,
                               out=np.full_like(pair_explained, np.nan), where=pair_exposed > 0.0)
    testable = decision.trace["testable"]
    rows, states = alternatives.rows, alternatives.states
    compatible = np.zeros(len(rows), dtype=np.bool_)
    compatible[states == M0] = True
    m1 = np.flatnonzero(states == M1)
    compatible[m1] = testable[m1] & (coverage[rows[m1, 0], rows[m1, 1]]
                                     >= config.exposure.report_edge_coverage)
    m2 = np.flatnonzero(states == M2)
    compatible[m2] = (testable[m2]
                      & (coverage[rows[m2, 0], rows[m2, 1]] >= config.exposure.report_edge_coverage)
                      & (coverage[rows[m2, 0], rows[m2, 2]] >= config.exposure.report_edge_coverage)
                      & (explainability[alternatives.panel_row[m2]]
                         >= config.exposure.report_pair_explainability))
    evaluable = np.zeros(alternatives.n_samples, dtype=np.bool_)
    np.logical_or.at(evaluable, rows[m1, 0], testable[m1])
    return coverage, explainability, compatible, testable, evaluable


def diagnostics(decision: Decision, config: PedigreeConfig) -> pd.DataFrame:
    """One row per sample: selections, margins, support fractions, structure and tier flags."""
    full, alternatives, evidence = decision.full, decision.alternatives, decision.evidence
    rows, states = alternatives.rows, alternatives.states
    sample_ids = decision.scores.sample_ids
    boot = decision.resampling.bootstrap
    coverage, explainability, compatible, testable, evaluable = _structure(decision, config)
    junctions, callable_bins = decision.scores.junctions, decision.scores.callable_bins
    eligible_children = decision.eligibility.children
    records = []
    for child, sample in enumerate(sample_ids):
        child_rows = alternatives.child_rows[child]
        best_m1, best_m1_margin = unique_finite_winner(child_rows[states[child_rows] == M1],
                                                       full.decision_scores)
        best_m2, best_m2_margin = unique_finite_winner(child_rows[states[child_rows] == M2],
                                                       full.decision_scores)
        local_state, local_row = int(full.local_state[child]), int(full.local_row[child])
        graph_row = int(full.graph_row[child])
        graph_state = int(evidence.graph_state[child])
        selected = _row_parents(rows, local_row)
        graph_parents = _row_parents(rows, graph_row)
        best_m2_parents = _row_parents(rows, -1 if best_m2 is None else best_m2) + (None, None)
        sensitivity = decision.sensitivity
        agreement = lambda values: (np.nan if not eligible_children[child]  # noqa: E731
                                    else float(np.mean(values)))
        supports = [selection.state_support[child, local_state] for _, selection in sensitivity
                    if local_state != UNRESOLVED]
        selected_utility = np.nan if graph_row < 0 else float(full.decision_scores[graph_row])
        best = -np.inf if not len(child_rows) else float(np.max(full.decision_scores[child_rows]))
        state_margin = np.nan if local_state == UNRESOLVED else float(full.state_margin[child])
        identity_margin = np.nan if local_row < 0 else float(full.identity_margin[child])
        first_set, second_set, configuration_rows = support_sides(
            child_rows, local_state, local_row, alternatives, full.decision_scores,
            config.release.support_set_coverage)
        parent_coverage = [np.nan, np.nan]
        parent_direction = [np.nan, np.nan]
        pair_explainability = np.nan
        if local_row >= 0 and selected:
            for slot, parent in enumerate(selected):
                parent_coverage[slot] = coverage[child, parent]
                parent_direction[slot] = decision.orientation.order_probability[child, parent]
            if len(selected) > 1 and alternatives.panel_row[local_row] >= 0:
                pair_explainability = explainability[alternatives.panel_row[local_row]]
        tier_a, tier_b = decision.tier_a, decision.tier_b
        records.append({
            "Sample": sample,
            "M0PriorSensitivityStateAgreement": bool(evidence.m0_state_agreement[child]),
            "M0PriorSensitivityGraphAgreement": bool(evidence.m0_graph_agreement[child]),
            "M0PriorSensitivityTierVeto": bool(evidence.m0_veto[child]),
            "StructureChildEvaluable": bool(evaluable[child]),
            "StructureSelectedRowExposureTestable": bool(local_row >= 0 and testable[local_row]),
            "StructureSelectedRowCXCompatible": bool(local_row >= 0 and compatible[local_row]),
            "StructureParent1Coverage": parent_coverage[0],
            "StructureParent2Coverage": parent_coverage[1],
            "StructurePairExplainability": pair_explainability,
            "StructureParent1DirectionProbability": parent_direction[0],
            "StructureParent2DirectionProbability": parent_direction[1],
            "BestM1Parent": None if best_m1 is None else sample_ids[rows[best_m1, 1]],
            "BestM1ConditionalIdentityMargin": best_m1_margin,
            "BestM2Parent1": _name(sample_ids, best_m2_parents[0]),
            "BestM2Parent2": _name(sample_ids, best_m2_parents[1]),
            "BestM2ConditionalIdentityMargin": best_m2_margin,
            "LocalParent1": _name(sample_ids, (selected + (None,))[0]),
            "LocalParent2": _name(sample_ids, (selected + (None, None))[1]),
            "EligibleChild": bool(eligible_children[child]),
            "EligibleParentCount": int(alternatives.full_counts[child, 1]),
            "EligibleParentPairCount": int(alternatives.full_counts[child, 2]),
            "CompleteParent1": _name(sample_ids, (graph_parents + (None,))[0]),
            "CompleteParent2": _name(sample_ids, (graph_parents + (None, None))[1]),
            "SelectedParentState": None if graph_state == UNRESOLVED else STATE_NAMES[graph_state],
            "LocalWinnerParentState": None if local_state == UNRESOLVED else STATE_NAMES[local_state],
            "ObservedParentCount": np.nan if graph_state == UNRESOLVED else graph_state,
            "LocalObservedParentCount": np.nan if local_state == UNRESOLVED else local_state,
            "Identifiable": bool(local_row >= 0 and graph_row == local_row),
            "DAGDisplaced": bool(local_row >= 0 and graph_row >= 0 and graph_row != local_row),
            "GraphConflict": bool(evidence.graph_conflict[child]),
            "GraphTieConflict": bool(full.tie_conflict[child]),
            "GraphDirectionResolvedAlternative": bool(full.direction_resolved[child]),
            "LocalParentRoleProbability": (np.nan if local_row < 0 else parent_role_probability(
                local_row, rows, decision.orientation)),
            "SelectedParentRoleProbability": full.role_probability[child],
            "RawAncestryJunctionCount": float(np.sum(junctions[:, child])),
            "CallableAncestryHaplotypeBinCount": float(np.sum(callable_bins[:, child])),
            "AdjustedAncestryJunctionBurden": float(decision.orientation.adjusted_burden[child]),
            "AncestryPaintingCallabilityFraction": float(decision.orientation.callability[child]),
            "InformativeContigCount": int(decision.informative[child]),
            "LocalStateBootstrapFraction": evidence.state_bootstrap[child],
            "LocalConfigurationBootstrapFraction": evidence.configuration_bootstrap[child],
            "GraphConfigurationBootstrapFraction": evidence.configuration_graph_bootstrap[child],
            "SelectedGraphConfigurationBootstrapFraction": evidence.graph_configuration_bootstrap[child],
            "SelectedGraphParent1BootstrapFraction": evidence.graph_parent_bootstrap[child, 0],
            "SelectedGraphParent2BootstrapFraction": evidence.graph_parent_bootstrap[child, 1],
            "Parent1BootstrapFraction": evidence.parent_bootstrap[child, 0],
            "Parent2BootstrapFraction": evidence.parent_bootstrap[child, 1],
            "LocalStateLOCOFraction": evidence.state_loco[child],
            "LocalConfigurationLOCOFraction": evidence.configuration_loco[child],
            "GraphConfigurationLOCOFraction": evidence.configuration_graph_loco[child],
            "SelectedGraphConfigurationLOCOFraction": evidence.graph_configuration_loco[child],
            "SelectedGraphParent1LOCOFraction": evidence.graph_parent_loco[child, 0],
            "SelectedGraphParent2LOCOFraction": evidence.graph_parent_loco[child, 1],
            "Parent1LOCOFraction": evidence.parent_loco[child, 0],
            "Parent2LOCOFraction": evidence.parent_loco[child, 1],
            "StateWinnerMargin": state_margin,
            "ConditionalIdentityMargin": identity_margin,
            "UnconstrainedWinnerMargin": (min(state_margin, identity_margin)
                                          if np.isfinite(identity_margin) else state_margin),
            "SelectedAggregateUtility": selected_utility,
            "SelectedMinusUnconstrainedBest": (np.nan if graph_row < 0
                                               else selected_utility - best),
            "StateLogEvidence0": full.state_log_evidence[child, 0],
            "StateLogEvidence1": full.state_log_evidence[child, 1],
            "StateLogEvidence2": full.state_log_evidence[child, 2],
            "StateSupport0": full.state_support[child, 0],
            "StateSupport1": full.state_support[child, 1],
            "StateSupport2": full.state_support[child, 2],
            "ScoredCandidateCount2": alternatives.scored_counts[child, 2],
            "M2CandidateScreenIncomplete": bool(alternatives.scored_counts[child, 2]
                                                < alternatives.full_counts[child, 2]),
            "PriorSensitivityLocalStateAgreementFraction": agreement(
                [selection.local_state[child] == local_state for _, selection in sensitivity]),
            "PriorSensitivityLocalIdentityAgreementFraction": agreement(
                [selection.local_row[child] == local_row for _, selection in sensitivity]),
            "PriorSensitivityGraphAgreementFraction": agreement(
                [selection.graph_row[child] == graph_row for _, selection in sensitivity]),
            "PriorSensitivitySelectedStateMinimumSupport": (np.nan if not supports
                                                            else float(np.min(supports))),
            "InferenceStatus": decision.complete.status[child],
            "TierAStateCall": bool(tier_a.state_pass[child]),
            "TierBStateCall": bool(tier_b.state_pass[child]),
            "TierAExactConfiguration": bool(tier_a.identity_pass[child]),
            "TierBExactConfiguration": bool(tier_b.identity_pass[child]),
            "TierAExactPair": bool(tier_a.identity_pass[child] and local_state == M2),
            "TierBExactPair": bool(tier_b.identity_pass[child] and local_state == M2),
            "TierAParent1": bool(decision.partial_a.supported[child, 0]),
            "TierAParent2": bool(decision.partial_a.supported[child, 1]),
            "TierBParent1": bool(decision.partial_b.supported[child, 0]),
            "TierBParent2": bool(decision.partial_b.supported[child, 1]),
            "ConfigurationSupportSet": _support_text(child_rows, boot.local_configuration, rows,
                                                     states, sample_ids,
                                                     config.release.support_set_coverage),
            "Parent1CandidateSet": _set_text(first_set, sample_ids),
            "Parent2CandidateSet": _set_text(second_set, sample_ids),
            "EvidenceConfigurationSet": ";".join(
                "+".join(str(sample_ids[parent]) for parent in rows[row, 1:] if parent >= 0)
                for row in configuration_rows),
        })
    return pd.DataFrame(records)


def candidate_sets(decision: Decision, release: CountRelease) -> pd.DataFrame:
    """Tier-B candidate sets with retained marginal edges and the full-marker release applied."""
    sets, partial = decision.candidates, decision.partial_b
    sample_ids = decision.scores.sample_ids
    final, final_partial = release.tier_b, release.partial_b
    records = []
    for child, sample in enumerate(sample_ids):
        state = int(decision.tier_b.state[child])
        first, second = sets.side_candidates[child]
        record = {
            "Sample": sample, "ParentState": _state_name(state), "ObservedParentCount": _count(state),
            "Parent1": _name(sample_ids, sets.parents[child, 0]),
            "Parent2": _name(sample_ids, sets.parents[child, 1]),
            "Parent1Candidates": tuple(sample_ids[i] for i in first),
            "Parent2Candidates": tuple(sample_ids[i] for i in second),
            "ConfigurationCandidates": tuple(tuple(sample_ids[i] for i in configuration)
                                             for configuration in sets.configurations[child]),
            "ExactConfigurationResolved": bool(sets.exact[child]),
            "InferenceStatus": sets.status[child],
            "Parent1Supported": bool(partial.supported[child, 0]),
            "Parent2Supported": bool(partial.supported[child, 1]),
            "ObservedParentCountMinimum": int(partial.count_minimum[child]),
            "ObservedParentCountMaximum": int(partial.count_maximum[child]),
        }
        if release.released[child]:
            kept = tuple(sample_ids[i] for i in release.kept[child])
            record.update(ParentState=_state_name(int(final.state[child])),
                          ObservedParentCount=_count(int(final.state[child])),
                          Parent1=kept[0] if kept else None, Parent2=None,
                          InferenceStatus=final.status[child], ExactConfigurationResolved=True,
                          Parent1Candidates=kept, Parent2Candidates=(),
                          ConfigurationCandidates=(kept,),
                          Parent1Supported=bool(final_partial.supported[child, 0]),
                          Parent2Supported=bool(final_partial.supported[child, 1]),
                          ObservedParentCountMinimum=int(final_partial.count_minimum[child]),
                          ObservedParentCountMaximum=int(final_partial.count_maximum[child]))
        records.append(record)
    return pd.DataFrame(records)


def explanations(decision: Decision, candidate_table: pd.DataFrame, diagnostic_table: pd.DataFrame,
                 release: CountRelease) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(call explanations per sample, the leading alternatives behind each call)."""
    full, alternatives = decision.full, decision.alternatives
    rows, states = alternatives.rows, alternatives.states
    sample_ids = decision.scores.sample_ids
    state_scores, identity_scores = decision.adjusted
    terms = decision.trace
    summaries, records = [], []
    for child, child_rows in enumerate(alternatives.child_rows):
        summary = dict(candidate_table.iloc[child])
        if not len(child_rows):
            summaries.append(dict(summary, Interpretation=(
                "excluded as an inference target by caller eligibility")))
            continue
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
            records.append(dict(
                Sample=sample_ids[child], ParentState=f"M{state}",
                Parent1=_name(sample_ids, rows[row, 1]), Parent2=_name(sample_ids, rows[row, 2]),
                LocalWinner=row == local, GraphWinner=row == graph, WithinStateRank=ranks.get(row),
                GeneticLogEvidence=terms["genetic"][row],
                StructuralStateLogEvidence=terms["structural_state"][row],
                StructuralIdentityEligible=bool(np.isfinite(terms["structural_identity"][row])),
                DirectionLogContribution=terms["direction"][row],
                ReciprocalFamilyLogContribution=terms["reciprocal_family"][row],
                AncestryPathLogContribution=terms["ancestry_paths"][row],
                AdjustedStateLogEvidence=state_scores[row],
                AdjustedIdentityLogEvidence=identity_scores[row],
                ParentStateLogPrior=np.log(full.priors[child, state]),
                LogEligibleIdentityMultiplicity=np.log(max(1, alternatives.full_counts[child, state])),
                MarginalStateLogEvidence=full.state_log_evidence[child, state],
                GraphDecisionUtility=full.decision_scores[row]))
        diagnostic = diagnostic_table.iloc[child]
        for name in ("LocalWinnerParentState", "SelectedParentState", "LocalParent1",
                     "LocalParent2", "GraphConflict", "StateWinnerMargin",
                     "ConditionalIdentityMargin", "LocalStateBootstrapFraction",
                     "LocalStateLOCOFraction"):
            summary[name] = diagnostic[name]
        summary.update(
            EligibleM2Pairs=int(alternatives.full_counts[child, 2]),
            ScoredM2Pairs=int(alternatives.scored_counts[child, 2]),
            IncompleteM2Search=bool(alternatives.scored_counts[child, 2]
                                    < alternatives.full_counts[child, 2]),
            GraphChangedLocalConfiguration=graph != local,
            Interpretation=("fixed-observed-genome exclusion conditional on supported family "
                            "directions; original resampling stability is unchanged; M0 is not "
                            "founder generation") if release.released[child] else (
                "composite model support; M0 means no observed parent, not founder generation"))
        summaries.append(summary)
    return pd.DataFrame(summaries), pd.DataFrame(records)


def final_state_columns(release: CountRelease, sample_ids, rows) -> dict[str, np.ndarray]:
    """TierBFinalStateCall (state called and every called parent released) and TierBReleaseEvidence."""
    final = release.tier_b
    called = np.asarray([
        int(final.state[child]) != UNRESOLVED
        and int(final.state[child]) == len(_row_parents(rows, int(final.row[child])))
        for child in range(len(sample_ids))])
    return {"TierBFinalStateCall": called,
            "TierBReleaseEvidence": np.asarray(release.partial_b.release_evidence, dtype=object)}


def build(decision: Decision, release: CountRelease, search: pd.DataFrame,
          chromosome_summary: pd.DataFrame, config: PedigreeConfig) -> PedigreeReport:
    """All report tables of one pedigree."""
    sample_ids = decision.scores.sample_ids
    rows = decision.alternatives.rows
    final_columns = final_state_columns(release, sample_ids, rows)
    diagnostic_table = diagnostics(decision, config)
    candidate_table = candidate_sets(decision, release)
    call_table, alternative_table = explanations(decision, candidate_table, diagnostic_table, release)
    state_calls = diagnostic_table[[
        "Sample", "SelectedParentState", "LocalWinnerParentState", "ObservedParentCount",
        "LocalObservedParentCount", "InferenceStatus", "DAGDisplaced", "GraphConflict",
        "GraphTieConflict", "GraphDirectionResolvedAlternative", "SelectedParentRoleProbability",
        "TierAStateCall", "TierBStateCall"]].copy()
    for table in (diagnostic_table, state_calls, candidate_table, call_table):
        for name, values in final_columns.items():
            table[name] = values
    tier_b = relationships(release.tier_b, rows, sample_ids)
    tier_b["ReleaseEvidence"] = list(release.partial_b.release_evidence)
    exclusion_evidence = pd.DataFrame(dict(
        Sample=[sample_ids[c] for c, _ in release.dyads],
        CandidateParent=[sample_ids[p] for _, p in release.dyads],
        LogE=release.log_e, ObservedSites=release.observed_sites),
        columns=["Sample", "CandidateParent", "LogE", "ObservedSites"])
    scores = decision.scores
    evidence_summary = pd.DataFrame({
        "Contig": list(scores.contigs),
        "InformativeMarkers": scores.informative_markers.astype(np.int64),
        "MeanMinimumAncestryJunctions": np.mean(scores.junctions, axis=1),
        "MeanCallableHaplotypeBins": np.mean(scores.callable_bins, axis=1)})
    return PedigreeReport(
        tier_b_relationships=tier_b,
        tier_b_partial_relationships=partial_relationships(release.partial_b, sample_ids, True),
        tier_a_relationships=relationships(decision.tier_a, rows, sample_ids),
        tier_a_partial_relationships=partial_relationships(decision.partial_a, sample_ids, False),
        complete_relationships=relationships(decision.complete, rows, sample_ids),
        tier_b_candidate_sets=candidate_table, parent_state_calls=state_calls,
        diagnostics=diagnostic_table, call_explanations=call_table,
        alternative_explanations=alternative_table, search_diagnostics=search,
        exclusion_evidence=exclusion_evidence,
        exclusion_diagnostics=pd.DataFrame(list(release.records),
                                           columns=list(EXCLUSION_DIAGNOSTIC_COLUMNS)),
        evidence_summary=evidence_summary, chromosome_summary=chromosome_summary)
