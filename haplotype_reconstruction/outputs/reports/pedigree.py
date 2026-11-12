"""Render typed pedigree views as the 17 published CSVs and two summaries.

Owns OU15/PD25 names, tuple text, column order and missing-value text inference.
All selected rows, probabilities, guard transformations and support sets are pedigree
values. Method: docs/methods/pedigree.md.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from ...contracts import pedigree as records
from .tables import record_mapping, write_frame, write_json, write_rows

__all__ = ["render", "write_pedigree", "COLUMNS"]

# Fixed published table schemas; report construction never reads expected data.
COLUMNS = {
    "tier_b_relationships": [
        "Sample",
        "Parent1",
        "Parent2",
        "ParentState",
        "ObservedParentCount",
        "InferenceStatus",
        "ReleaseEvidence",
        "RelationshipBasis",
    ],
    "tier_b_partial_relationships": [
        "Sample",
        "Parent1",
        "Parent2",
        "ParentState",
        "ObservedParentCount",
        "InferenceStatus",
        "Parent1Supported",
        "Parent2Supported",
        "ExactConfigurationResolved",
        "ObservedParentCountMinimum",
        "ObservedParentCountMaximum",
        "ReleaseEvidence",
        "RelationshipBasis",
    ],
    "tier_a_relationships": [
        "Sample",
        "Parent1",
        "Parent2",
        "ParentState",
        "ObservedParentCount",
        "InferenceStatus",
        "RelationshipBasis",
    ],
    "tier_a_partial_relationships": [
        "Sample",
        "Parent1",
        "Parent2",
        "ParentState",
        "ObservedParentCount",
        "InferenceStatus",
        "Parent1Supported",
        "Parent2Supported",
        "ExactConfigurationResolved",
        "ObservedParentCountMinimum",
        "ObservedParentCountMaximum",
        "RelationshipBasis",
    ],
    "complete_relationships": ["Sample", "Parent1", "Parent2", "ParentState", "ObservedParentCount", "InferenceStatus"],
    "tier_b_candidate_sets": [
        "Sample",
        "ParentState",
        "ObservedParentCount",
        "Parent1",
        "Parent2",
        "Parent1Candidates",
        "Parent2Candidates",
        "ConfigurationCandidates",
        "ExactConfigurationResolved",
        "InferenceStatus",
        "Parent1Supported",
        "Parent2Supported",
        "ObservedParentCountMinimum",
        "ObservedParentCountMaximum",
        "TierBFinalStateCall",
        "TierBReleaseEvidence",
    ],
    "parent_state_calls": [
        "Sample",
        "SelectedParentState",
        "LocalWinnerParentState",
        "ObservedParentCount",
        "LocalObservedParentCount",
        "InferenceStatus",
        "DAGDisplaced",
        "GraphConflict",
        "GraphTieConflict",
        "GraphDirectionResolvedAlternative",
        "SelectedParentRoleProbability",
        "TierAStateCall",
        "TierBStateCall",
        "TierBFinalStateCall",
        "TierBReleaseEvidence",
    ],
    "diagnostics": [
        "Sample",
        "M0PriorSensitivityStateAgreement",
        "M0PriorSensitivityGraphAgreement",
        "M0PriorSensitivityTierVeto",
        "StructureChildEvaluable",
        "StructureSelectedRowExposureTestable",
        "StructureSelectedRowCXCompatible",
        "StructureParent1Coverage",
        "StructureParent2Coverage",
        "StructurePairExplainability",
        "StructureParent1DirectionProbability",
        "StructureParent2DirectionProbability",
        "BestM1Parent",
        "BestM1ConditionalIdentityMargin",
        "BestM2Parent1",
        "BestM2Parent2",
        "BestM2ConditionalIdentityMargin",
        "LocalParent1",
        "LocalParent2",
        "EligibleChild",
        "EligibleParentCount",
        "EligibleParentPairCount",
        "CompleteParent1",
        "CompleteParent2",
        "SelectedParentState",
        "LocalWinnerParentState",
        "ObservedParentCount",
        "LocalObservedParentCount",
        "Identifiable",
        "DAGDisplaced",
        "GraphConflict",
        "GraphTieConflict",
        "GraphDirectionResolvedAlternative",
        "LocalParentRoleProbability",
        "SelectedParentRoleProbability",
        "RawAncestryJunctionCount",
        "CallableAncestryHaplotypeBinCount",
        "AdjustedAncestryJunctionBurden",
        "AncestryPaintingCallabilityFraction",
        "InformativeContigCount",
        "InformativeMapUnits",
        "LocalStateBootstrapFraction",
        "LocalConfigurationBootstrapFraction",
        "GraphConfigurationBootstrapFraction",
        "SelectedGraphConfigurationBootstrapFraction",
        "SelectedGraphParent1BootstrapFraction",
        "SelectedGraphParent2BootstrapFraction",
        "Parent1BootstrapFraction",
        "Parent2BootstrapFraction",
        "LocalStateLOCOFraction",
        "LocalConfigurationLOCOFraction",
        "GraphConfigurationLOCOFraction",
        "SelectedGraphConfigurationLOCOFraction",
        "SelectedGraphParent1LOCOFraction",
        "SelectedGraphParent2LOCOFraction",
        "Parent1LOCOFraction",
        "Parent2LOCOFraction",
        "StateWinnerMargin",
        "ConditionalIdentityMargin",
        "UnconstrainedWinnerMargin",
        "SelectedAggregateUtility",
        "SelectedMinusUnconstrainedBest",
        "StateLogEvidence0",
        "StateLogEvidence1",
        "StateLogEvidence2",
        "StateSupport0",
        "StateSupport1",
        "StateSupport2",
        "ScoredCandidateCount2",
        "M2CandidateScreenIncomplete",
        "PriorSensitivityLocalStateAgreementFraction",
        "PriorSensitivityLocalIdentityAgreementFraction",
        "PriorSensitivityGraphAgreementFraction",
        "PriorSensitivitySelectedStateMinimumSupport",
        "InferenceStatus",
        "TierAStateCall",
        "TierBStateCall",
        "TierAExactConfiguration",
        "TierBExactConfiguration",
        "TierAExactPair",
        "TierBExactPair",
        "TierAParent1",
        "TierAParent2",
        "TierBParent1",
        "TierBParent2",
        "ConfigurationSupportSet",
        "Parent1CandidateSet",
        "Parent2CandidateSet",
        "EvidenceConfigurationSet",
        "TierBFinalStateCall",
        "TierBReleaseEvidence",
    ],
    "call_explanations": [
        "Sample",
        "ParentState",
        "ObservedParentCount",
        "Parent1",
        "Parent2",
        "Parent1Candidates",
        "Parent2Candidates",
        "ConfigurationCandidates",
        "ExactConfigurationResolved",
        "InferenceStatus",
        "Parent1Supported",
        "Parent2Supported",
        "ObservedParentCountMinimum",
        "ObservedParentCountMaximum",
        "LocalWinnerParentState",
        "SelectedParentState",
        "LocalParent1",
        "LocalParent2",
        "GraphConflict",
        "StateWinnerMargin",
        "ConditionalIdentityMargin",
        "LocalStateBootstrapFraction",
        "LocalStateLOCOFraction",
        "EligibleM2Pairs",
        "ScoredM2Pairs",
        "IncompleteM2Search",
        "GraphChangedLocalConfiguration",
        "Interpretation",
        "TierBFinalStateCall",
        "TierBReleaseEvidence",
    ],
    "alternative_explanations": [
        "Sample",
        "ParentState",
        "Parent1",
        "Parent2",
        "LocalWinner",
        "GraphWinner",
        "WithinStateRank",
        "GeneticLogEvidence",
        "StructuralStateLogEvidence",
        "StructuralIdentityEligible",
        "DirectionLogContribution",
        "ReciprocalFamilyLogContribution",
        "AncestryPathLogContribution",
        "AdjustedStateLogEvidence",
        "AdjustedIdentityLogEvidence",
        "ParentStateLogPrior",
        "LogEligibleIdentityMultiplicity",
        "MarginalStateLogEvidence",
        "GraphDecisionUtility",
    ],
    "search_diagnostics": [
        "Sample",
        "EligibleParents",
        "ScoredM2Pairs",
        "M2PanelParents",
        "ScreenBoundaryRank",
        "ScreenBoundaryGap",
        "OmittedParentCount",
        "StrongestOmittedParent",
        "Parent1ScreenRank",
        "Parent2ScreenRank",
        "SelectedParentAtScreenBoundary",
        "M2PanelCapped",
    ],
    "exclusion_evidence": ["Sample", "CandidateParent", "LogE", "ObservedSites"],
    "exclusion_diagnostics": [
        "Sample",
        "EligibleCandidates",
        "CompatibleCandidates",
        "SupportedReverseCandidates",
        "RemainingCandidates",
        "WeakestExclusionLogE",
        "ThresholdLogE",
        "CompleteCandidateEvidence",
        "Released",
        "ReleaseConditionalOnSupportedDirections",
    ],
    "evidence_summary": ["Contig", "InformativeMarkers", "MeanMinimumAncestryJunctions", "MeanCallableHaplotypeBins"],
    "chromosome_summary": ["Contig", "ComponentCount", "ScoredComponentCount", "InformativeMarkers", "OmittedReason"],
    "relative_contrasts": [
        "Sample",
        "CandidateParent",
        "Views",
        "ProxyModel",
        "M1MinusSecondDegreeProxyNats",
        "M1MinusFullSibProxyNats",
        "UsableChromosomes",
        "NegativeChromosomesSecondDegree",
        "NegativeChromosomesFullSib",
        "MinChromosomeContrastSecondDegree",
    ],
    "sibling_guard": [
        "Parent",
        "Child",
        "Witnesses",
        "SetAside",
        "Against",
        "Supporting",
        "PairScore",
        "FamilyScore",
        "Score",
        "Call",
        "EvidenceCall",
        "WithdrawalIteration",
        "WithholdingBasis",
    ],
}


# Empty row-dict frames have no header. Fixed-column frames retain their full header.
EMPTY_COLUMNS = {
    "tier_b_relationships": ("ReleaseEvidence", "RelationshipBasis"),
    "tier_b_partial_relationships": ("RelationshipBasis",),
    "tier_a_relationships": ("RelationshipBasis",),
    "tier_a_partial_relationships": ("RelationshipBasis",),
    "complete_relationships": (),
    "tier_b_candidate_sets": ("TierBFinalStateCall", "TierBReleaseEvidence"),
    "diagnostics": ("TierBFinalStateCall", "TierBReleaseEvidence"),
    "call_explanations": ("TierBFinalStateCall", "TierBReleaseEvidence"),
    "alternative_explanations": (),
    "search_diagnostics": (),
    "chromosome_summary": (),
}


def snake(name):
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)).lower()


REFERENCE_FIELDS = {
    "Sample",
    "Parent1",
    "Parent2",
    "CandidateParent",
    "BestM1Parent",
    "BestM2Parent1",
    "BestM2Parent2",
    "LocalParent1",
    "LocalParent2",
    "CompleteParent1",
    "CompleteParent2",
    "StrongestOmittedParent",
    "Parent",
    "Child",
}
STATE_FIELDS = {"ParentState", "SelectedParentState", "LocalWinnerParentState"}
SET_FIELDS = {
    "Parent1Candidates",
    "Parent2Candidates",
    "CompatibleCandidates",
    "SupportedReverseCandidates",
    "RemainingCandidates",
}


def _value(column, value, names, table):
    # Absent search ranks are None before pandas infers the column dtype.
    if column in {"Parent1ScreenRank", "Parent2ScreenRank"}:
        return None if value is None or np.isnan(value) else int(value)
    if column == "ScreenBoundaryRank":
        return int(value)
    if column in REFERENCE_FIELDS:
        return None if value is None or value < 0 else names[int(value)]
    if column in STATE_FIELDS:
        if value < 0:
            return "unresolved" if column == "ParentState" else None
        return f"M{value}" if table == "alternative_explanations" else records.STATE_NAMES[int(value)]
    if column in SET_FIELDS:
        return tuple(names[i] for i in value)
    if column == "ConfigurationCandidates":
        return tuple(tuple(names[i] for i in configuration) for configuration in value)
    if column in {"Parent1CandidateSet", "Parent2CandidateSet"}:
        return "{" + ",".join(str(names[i]) for i in value) + "}"
    if column == "EvidenceConfigurationSet":
        return ";".join("+".join(str(names[i]) for i in configuration) for configuration in value)
    if column == "ConfigurationSupportSet":
        return ";".join(
            f"{records.STATE_NAMES[v.state]}:"
            + ("+".join(str(names[i]) for i in v.parents) if v.parents else "external+external")
            + f":{v.probability:.3f}"
            for v in value
        )
    if column == "DroppedMembers":
        return ";".join(f"{names[i]}:{reason}" for i, reason in value)
    return value


def _count_row(value, names):
    row = dict(
        Sample=names[value.child],
        EligibleCandidates=value.eligible_candidates,
        CompatibleCandidates=tuple(names[i] for i in value.compatible),
        SupportedReverseCandidates=tuple(names[i] for i in value.reverse),
        RemainingCandidates=tuple(names[i] for i in value.remaining),
        WeakestExclusionLogE=value.weakest_log_e,
        ThresholdLogE=value.threshold_log_e,
        CompleteCandidateEvidence=value.complete_evidence,
        Released=value.released,
        ReleaseConditionalOnSupportedDirections=value.uses_supported_directions,
    )
    if value.weakest_exact is not None:
        row.update(
            WeakestExclusionLogELower=value.weakest_lower,
            WeakestExclusionLogEUpper=value.weakest_upper,
            WeakestExclusionLogEExact=value.weakest_exact,
        )
    return row


def _exclusion_frame(view: records.ExclusionEvidenceView, names: Sequence[str], intervals: bool) -> pd.DataFrame:
    """The exclusion evidence table, column by column from its columnar view (same columns and dtypes as row-wise:
    names as text, LogE and the interval floats as float64, ObservedSites int64, LogEExact bool)."""
    table = "exclusion_evidence"
    columns: dict[str, object] = dict(
        Sample=[_value("Sample", i, names, table) for i in view.sample.tolist()],
        CandidateParent=[_value("CandidateParent", i, names, table) for i in view.candidate_parent.tolist()],
        LogE=np.asarray(view.log_e, dtype=np.float64),
        ObservedSites=np.asarray(view.observed_sites, dtype=np.int64),
    )
    if intervals:
        columns.update(
            LogELower=np.asarray(view.log_e_lower, dtype=np.float64),
            LogEUpper=np.asarray(view.log_e_upper, dtype=np.float64),
            LogEExact=np.asarray(view.log_e_exact, dtype=np.bool_),
        )
    return pd.DataFrame(columns)


def render(product: records.PedigreeViews, names: Sequence[str]) -> dict[str, pd.DataFrame | None]:
    """Render numeric rows and text only; never choose or recompute a scientific value."""
    result = {}
    for table, fixed_columns in COLUMNS.items():
        values = getattr(product, table)
        columns = list(fixed_columns)
        if table in EMPTY_COLUMNS and not values:
            result[table] = pd.DataFrame(columns=list(EMPTY_COLUMNS[table]))
            continue
        if table == "relative_contrasts":
            if values is None:
                result[table] = None
                continue
            result[table] = pd.DataFrame(
                [
                    dict(
                        Sample=names[int(c)],
                        CandidateParent=names[int(p)],
                        Views=";".join(values.views[i]),
                        ProxyModel="relative-proxy-v1",
                        M1MinusSecondDegreeProxyNats=values.second_degree[i],
                        M1MinusFullSibProxyNats=values.full_sib[i],
                        UsableChromosomes=int(values.usable_chromosomes[i]),
                        NegativeChromosomesSecondDegree=int(values.negative_second[i]),
                        NegativeChromosomesFullSib=int(values.negative_full[i]),
                        MinChromosomeContrastSecondDegree=values.minimum_second[i],
                    )
                    for i, (c, p) in enumerate(values.pairs)
                ],
                columns=columns,
            )
            continue
        if table == "sibling_guard":
            if values is None:
                result[table] = None
                continue
            guard = values
            values = guard.rows
        if table == "exclusion_evidence":
            result[table] = _exclusion_frame(values, names, product.exclusion_intervals)
            continue
        if table == "exclusion_diagnostics":
            if product.exclusion_intervals:
                columns += ["WeakestExclusionLogELower", "WeakestExclusionLogEUpper", "WeakestExclusionLogEExact"]
            result[table] = pd.DataFrame([_count_row(v, names) for v in values], columns=columns)
            continue
        oriented = table in (
            "tier_b_relationships",
            "tier_b_partial_relationships",
            "tier_a_relationships",
            "tier_a_partial_relationships",
            "complete_relationships",
            "tier_b_candidate_sets",
        )
        if oriented and values and values[0].parent_slot_basis is not None:
            columns.insert(columns.index("Parent2") + 1, "ParentSlotBasis")
        result[table] = pd.DataFrame(
            [{c: _value(c, getattr(v, snake(c)), names, table) for c in columns} for v in values], columns=columns
        )
        if oriented:
            for slot, column in enumerate(("Parent1", "Parent2")):
                if any(v.parents_before_guard is not None and v.parents_before_guard[slot] >= 0 for v in values):
                    result[table][column] = pd.Series(result[table][column].tolist(), dtype="str")
        if oriented and "ParentSlotBasis" in columns:
            # Swapped names and bases use explicit object dtype.
            result[table]["ParentSlotBasis"] = pd.Series([v.parent_slot_basis for v in values], dtype=object)
        if table == "sibling_guard":
            if not values:
                # Assign empty lists after constructing the frame to preserve column dtypes.
                for column in ("WithdrawalIteration", "WithholdingBasis"):
                    result[table][column] = np.empty(0, dtype=np.float64)
            result[table].attrs.update(
                cleaning_iterations=guard.cleaning_iterations,
                cleaning_converged=guard.cleaning_converged,
                cleaning_iteration_bound=guard.cleaning_iteration_bound,
            )
    return result


# Summary insertion order: chromosome details precede the released count.
# Record field layout is a storage contract, separate from JSON key order.
EXCLUSION_SUMMARY_FIELDS = (
    "model",
    "alpha",
    "genotype_replacement_probability",
    "conditional_on_supported_edge_directions",
    "candidate_dyads",
    "contigs",
    "status",
    "chromosomes",
    "released",
)


def write_pedigree(
    output: str | Path, product: records.Pedigree, input_ids: Sequence[str] | None = None, *, map_work=None
) -> list[Path]:
    """Write every table of ``product.views``, the run summary; returns the paths written.
    ``input_ids``: the run's input sample order (the store manifest's, evidence.sample_order.store_input_order);
    per-sample tables list their rows in it (the records keep the canonical order)."""
    from .tables import frame_in_input_order

    frames = render(product.views, product.sample_ids)
    exclusion = record_mapping(product.exclusion)
    directory = Path(output) / "pedigree"
    items = separate_collections(frames, product.views, product.sample_ids, input_ids)

    def write_table(name):
        return write_frame(directory / f"{name}.csv", frame_in_input_order(frames[name], input_ids))

    names = [field.name for field in fields(product.views) if field.name in frames and frames[field.name] is not None]
    written = list(map(write_table, names)) if map_work is None else map_work(write_table, names)
    written.append(write_rows(directory / "cell_items.csv", items, columns=CELL_ITEM_COLUMNS))
    written.append(
        write_json(
            directory / "summary.json",
            dict(
                informative_contigs=list(product.informative_contigs),
                omitted=dict(product.omitted),
                reach=record_mapping(product.reach),
                family_solve=record_mapping(product.family_solve),
                exclusion={name: exclusion[name] for name in EXCLUSION_SUMMARY_FIELDS if name in exclusion},
                eligibility=record_mapping(product.eligibility),
                sibling_guard=record_mapping(product.guard_summary),
            ),
        )
    )
    return written


# Published report column schemas.
# Row-dict tables use a headerless empty constructor; these constants do not
# impose columns on those tables. Scientific values remain upstream contract fields.
TIER_B_RELATIONSHIPS_COLUMNS = (
    "Sample",
    "Parent1",
    "Parent2",
    "ParentState",
    "ObservedParentCount",
    "InferenceStatus",
    "ReleaseEvidence",
    "RelationshipBasis",
)
"""Columns in published order.

    Sample: sample identifier (text).
    Parent1: first parent identifier; empty for an unobserved parent (text).
    Parent2: second parent identifier; empty for an unobserved parent (text).
    ParentState: released M0/M1/M2 state label (text).
    ObservedParentCount: released sequenced-parent count; unresolved values remain NaN (1).
    InferenceStatus: inference status (1).
    ReleaseEvidence: release evidence (1).
    RelationshipBasis: family when every released parent passes the family score threshold; pair_only otherwise (text).
"""

TIER_B_PARTIAL_RELATIONSHIPS_COLUMNS = (
    "Sample",
    "Parent1",
    "Parent2",
    "ParentState",
    "ObservedParentCount",
    "InferenceStatus",
    "Parent1Supported",
    "Parent2Supported",
    "ExactConfigurationResolved",
    "ObservedParentCountMinimum",
    "ObservedParentCountMaximum",
    "ReleaseEvidence",
    "RelationshipBasis",
)
"""Columns in published order.

    Sample: sample identifier (text).
    Parent1: first parent identifier; empty for an unobserved parent (text).
    Parent2: second parent identifier; empty for an unobserved parent (text).
    ParentState: released M0/M1/M2 state label (text).
    ObservedParentCount: released sequenced-parent count; unresolved values remain NaN (1).
    InferenceStatus: inference status (1).
    Parent1Supported: parent1 supported (1).
    Parent2Supported: parent2 supported (1).
    ExactConfigurationResolved: exact configuration resolved (1).
    ObservedParentCountMinimum: observed parent count minimum (1).
    ObservedParentCountMaximum: observed parent count maximum (1).
    ReleaseEvidence: release evidence (1).
    RelationshipBasis: family when every released parent passes the family score threshold; pair_only otherwise (text).
"""

TIER_A_RELATIONSHIPS_COLUMNS = (
    "Sample",
    "Parent1",
    "Parent2",
    "ParentState",
    "ObservedParentCount",
    "InferenceStatus",
    "RelationshipBasis",
)
"""Columns in published order.

    Sample: sample identifier (text).
    Parent1: first parent identifier; empty for an unobserved parent (text).
    Parent2: second parent identifier; empty for an unobserved parent (text).
    ParentState: released M0/M1/M2 state label (text).
    ObservedParentCount: released sequenced-parent count; unresolved values remain NaN (1).
    InferenceStatus: inference status (1).
    RelationshipBasis: family when every released parent passes the family score threshold; pair_only otherwise (text).
"""

TIER_A_PARTIAL_RELATIONSHIPS_COLUMNS = (
    "Sample",
    "Parent1",
    "Parent2",
    "ParentState",
    "ObservedParentCount",
    "InferenceStatus",
    "Parent1Supported",
    "Parent2Supported",
    "ExactConfigurationResolved",
    "ObservedParentCountMinimum",
    "ObservedParentCountMaximum",
    "RelationshipBasis",
)
"""Columns in published order.

    Sample: sample identifier (text).
    Parent1: first parent identifier; empty for an unobserved parent (text).
    Parent2: second parent identifier; empty for an unobserved parent (text).
    ParentState: released M0/M1/M2 state label (text).
    ObservedParentCount: released sequenced-parent count; unresolved values remain NaN (1).
    InferenceStatus: inference status (1).
    Parent1Supported: parent1 supported (1).
    Parent2Supported: parent2 supported (1).
    ExactConfigurationResolved: exact configuration resolved (1).
    ObservedParentCountMinimum: observed parent count minimum (1).
    ObservedParentCountMaximum: observed parent count maximum (1).
    RelationshipBasis: family when every released parent passes the family score threshold; pair_only otherwise (text).
"""

COMPLETE_RELATIONSHIPS_COLUMNS = (
    "Sample",
    "Parent1",
    "Parent2",
    "ParentState",
    "ObservedParentCount",
    "InferenceStatus",
)
"""Columns in published order.

    Sample: sample identifier (text).
    Parent1: first parent identifier; empty for an unobserved parent (text).
    Parent2: second parent identifier; empty for an unobserved parent (text).
    ParentState: released M0/M1/M2 state label (text).
    ObservedParentCount: released sequenced-parent count; unresolved values remain NaN (1).
    InferenceStatus: inference status (1).
"""

TIER_B_CANDIDATE_SETS_COLUMNS = (
    "Sample",
    "ParentState",
    "ObservedParentCount",
    "Parent1",
    "Parent2",
    "Parent1Candidates",
    "Parent2Candidates",
    "ConfigurationCandidates",
    "ExactConfigurationResolved",
    "InferenceStatus",
    "Parent1Supported",
    "Parent2Supported",
    "ObservedParentCountMinimum",
    "ObservedParentCountMaximum",
    "TierBFinalStateCall",
    "TierBReleaseEvidence",
)
"""Columns in published order.

    Sample: sample identifier (text).
    ParentState: released M0/M1/M2 state label (text).
    ObservedParentCount: released sequenced-parent count; unresolved values remain NaN (1).
    Parent1: first parent identifier; empty for an unobserved parent (text).
    Parent2: second parent identifier; empty for an unobserved parent (text).
    Parent1Candidates: number of ordered items in pedigree/cell_items.csv (count).
    Parent2Candidates: number of ordered items in pedigree/cell_items.csv (count).
    ConfigurationCandidates: number of ordered items in pedigree/cell_items.csv (count).
    ExactConfigurationResolved: exact configuration resolved (1).
    InferenceStatus: inference status (1).
    Parent1Supported: parent1 supported (1).
    Parent2Supported: parent2 supported (1).
    ObservedParentCountMinimum: observed parent count minimum (1).
    ObservedParentCountMaximum: observed parent count maximum (1).
    TierBFinalStateCall: tier b final state call (1).
    TierBReleaseEvidence: tier b release evidence (1).
"""

PARENT_STATE_CALLS_COLUMNS = (
    "Sample",
    "SelectedParentState",
    "LocalWinnerParentState",
    "ObservedParentCount",
    "LocalObservedParentCount",
    "InferenceStatus",
    "DAGDisplaced",
    "GraphConflict",
    "GraphTieConflict",
    "GraphDirectionResolvedAlternative",
    "SelectedParentRoleProbability",
    "TierAStateCall",
    "TierBStateCall",
    "TierBFinalStateCall",
    "TierBReleaseEvidence",
)
"""Columns in published order.

    Sample: sample identifier (text).
    SelectedParentState: selected M0/M1/M2 state label (text).
    LocalWinnerParentState: local winning M0/M1/M2 state label (text).
    ObservedParentCount: released sequenced-parent count; unresolved values remain NaN (1).
    LocalObservedParentCount: local observed parent count (1).
    InferenceStatus: inference status (1).
    DAGDisplaced: dag displaced (1).
    GraphConflict: graph conflict (1).
    GraphTieConflict: graph tie conflict (1).
    GraphDirectionResolvedAlternative: graph direction resolved alternative (1).
    SelectedParentRoleProbability: selected parent role probability (1).
    TierAStateCall: tier a state call (1).
    TierBStateCall: tier b state call (1).
    TierBFinalStateCall: tier b final state call (1).
    TierBReleaseEvidence: tier b release evidence (1).
"""

DIAGNOSTICS_COLUMNS = (
    "Sample",
    "M0PriorSensitivityStateAgreement",
    "M0PriorSensitivityGraphAgreement",
    "M0PriorSensitivityTierVeto",
    "StructureChildEvaluable",
    "StructureSelectedRowExposureTestable",
    "StructureSelectedRowCXCompatible",
    "StructureParent1Coverage",
    "StructureParent2Coverage",
    "StructurePairExplainability",
    "StructureParent1DirectionProbability",
    "StructureParent2DirectionProbability",
    "BestM1Parent",
    "BestM1ConditionalIdentityMargin",
    "BestM2Parent1",
    "BestM2Parent2",
    "BestM2ConditionalIdentityMargin",
    "LocalParent1",
    "LocalParent2",
    "EligibleChild",
    "EligibleParentCount",
    "EligibleParentPairCount",
    "CompleteParent1",
    "CompleteParent2",
    "SelectedParentState",
    "LocalWinnerParentState",
    "ObservedParentCount",
    "LocalObservedParentCount",
    "Identifiable",
    "DAGDisplaced",
    "GraphConflict",
    "GraphTieConflict",
    "GraphDirectionResolvedAlternative",
    "LocalParentRoleProbability",
    "SelectedParentRoleProbability",
    "RawAncestryJunctionCount",
    "CallableAncestryHaplotypeBinCount",
    "AdjustedAncestryJunctionBurden",
    "AncestryPaintingCallabilityFraction",
    "InformativeContigCount",
    "InformativeMapUnits",
    "LocalStateBootstrapFraction",
    "LocalConfigurationBootstrapFraction",
    "GraphConfigurationBootstrapFraction",
    "SelectedGraphConfigurationBootstrapFraction",
    "SelectedGraphParent1BootstrapFraction",
    "SelectedGraphParent2BootstrapFraction",
    "Parent1BootstrapFraction",
    "Parent2BootstrapFraction",
    "LocalStateLOCOFraction",
    "LocalConfigurationLOCOFraction",
    "GraphConfigurationLOCOFraction",
    "SelectedGraphConfigurationLOCOFraction",
    "SelectedGraphParent1LOCOFraction",
    "SelectedGraphParent2LOCOFraction",
    "Parent1LOCOFraction",
    "Parent2LOCOFraction",
    "StateWinnerMargin",
    "ConditionalIdentityMargin",
    "UnconstrainedWinnerMargin",
    "SelectedAggregateUtility",
    "SelectedMinusUnconstrainedBest",
    "StateLogEvidence0",
    "StateLogEvidence1",
    "StateLogEvidence2",
    "StateSupport0",
    "StateSupport1",
    "StateSupport2",
    "ScoredCandidateCount2",
    "M2CandidateScreenIncomplete",
    "PriorSensitivityLocalStateAgreementFraction",
    "PriorSensitivityLocalIdentityAgreementFraction",
    "PriorSensitivityGraphAgreementFraction",
    "PriorSensitivitySelectedStateMinimumSupport",
    "InferenceStatus",
    "TierAStateCall",
    "TierBStateCall",
    "TierAExactConfiguration",
    "TierBExactConfiguration",
    "TierAExactPair",
    "TierBExactPair",
    "TierAParent1",
    "TierAParent2",
    "TierBParent1",
    "TierBParent2",
    "ConfigurationSupportSet",
    "Parent1CandidateSet",
    "Parent2CandidateSet",
    "EvidenceConfigurationSet",
    "TierBFinalStateCall",
    "TierBReleaseEvidence",
)
"""Columns in published order.

    Sample: sample identifier (text).
    M0PriorSensitivityStateAgreement: m0 prior sensitivity state agreement (1).
    M0PriorSensitivityGraphAgreement: m0 prior sensitivity graph agreement (1).
    M0PriorSensitivityTierVeto: m0 prior sensitivity tier veto (1).
    StructureChildEvaluable: structure child evaluable (1).
    StructureSelectedRowExposureTestable: structure selected row exposure testable (1).
    StructureSelectedRowCXCompatible: structure selected row cx compatible (1).
    StructureParent1Coverage: structure parent1 coverage (1).
    StructureParent2Coverage: structure parent2 coverage (1).
    StructurePairExplainability: structure pair explainability (1).
    StructureParent1DirectionProbability: structure parent1 direction probability (1).
    StructureParent2DirectionProbability: structure parent2 direction probability (1).
    BestM1Parent: best m1 parent (1).
    BestM1ConditionalIdentityMargin: best m1 conditional identity margin (nats).
    BestM2Parent1: best m2 parent1 (1).
    BestM2Parent2: best m2 parent2 (1).
    BestM2ConditionalIdentityMargin: best m2 conditional identity margin (nats).
    LocalParent1: local parent1 (1).
    LocalParent2: local parent2 (1).
    EligibleChild: eligible child (1).
    EligibleParentCount: eligible parent count (1).
    EligibleParentPairCount: eligible parent pair count (1).
    CompleteParent1: complete parent1 (1).
    CompleteParent2: complete parent2 (1).
    SelectedParentState: selected M0/M1/M2 state label (text).
    LocalWinnerParentState: local winning M0/M1/M2 state label (text).
    ObservedParentCount: released sequenced-parent count; unresolved values remain NaN (1).
    LocalObservedParentCount: local observed parent count (1).
    Identifiable: identifiable (1).
    DAGDisplaced: dag displaced (1).
    GraphConflict: graph conflict (1).
    GraphTieConflict: graph tie conflict (1).
    GraphDirectionResolvedAlternative: graph direction resolved alternative (1).
    LocalParentRoleProbability: local parent role probability (1).
    SelectedParentRoleProbability: selected parent role probability (1).
    RawAncestryJunctionCount: raw ancestry junction count (1).
    CallableAncestryHaplotypeBinCount: callable ancestry haplotype bin count (1).
    AdjustedAncestryJunctionBurden: adjusted ancestry junction burden (1).
    AncestryPaintingCallabilityFraction: ancestry painting callability fraction (1).
    InformativeContigCount: informative contig count (1).
    InformativeMapUnits: informative map units (1).
    LocalStateBootstrapFraction: local state bootstrap fraction (1).
    LocalConfigurationBootstrapFraction: local configuration bootstrap fraction (1).
    GraphConfigurationBootstrapFraction: graph configuration bootstrap fraction (1).
    SelectedGraphConfigurationBootstrapFraction: selected graph configuration bootstrap fraction (1).
    SelectedGraphParent1BootstrapFraction: selected graph parent1 bootstrap fraction (1).
    SelectedGraphParent2BootstrapFraction: selected graph parent2 bootstrap fraction (1).
    Parent1BootstrapFraction: parent1 bootstrap fraction (1).
    Parent2BootstrapFraction: parent2 bootstrap fraction (1).
    LocalStateLOCOFraction: local state loco fraction (1).
    LocalConfigurationLOCOFraction: local configuration loco fraction (1).
    GraphConfigurationLOCOFraction: graph configuration loco fraction (1).
    SelectedGraphConfigurationLOCOFraction: selected graph configuration loco fraction (1).
    SelectedGraphParent1LOCOFraction: selected graph parent1 loco fraction (1).
    SelectedGraphParent2LOCOFraction: selected graph parent2 loco fraction (1).
    Parent1LOCOFraction: parent1 loco fraction (1).
    Parent2LOCOFraction: parent2 loco fraction (1).
    StateWinnerMargin: state winner margin (nats).
    ConditionalIdentityMargin: conditional identity margin (nats).
    UnconstrainedWinnerMargin: unconstrained winner margin (nats).
    SelectedAggregateUtility: selected aggregate utility (nats).
    SelectedMinusUnconstrainedBest: selected minus unconstrained best (1).
    StateLogEvidence0: state log evidence0 (nats).
    StateLogEvidence1: state log evidence1 (nats).
    StateLogEvidence2: state log evidence2 (nats).
    StateSupport0: state support0 (1).
    StateSupport1: state support1 (1).
    StateSupport2: state support2 (1).
    ScoredCandidateCount2: scored candidate count2 (1).
    M2CandidateScreenIncomplete: m2 candidate screen incomplete (1).
    PriorSensitivityLocalStateAgreementFraction: prior sensitivity local state agreement fraction (1).
    PriorSensitivityLocalIdentityAgreementFraction: prior sensitivity local identity agreement fraction (1).
    PriorSensitivityGraphAgreementFraction: prior sensitivity graph agreement fraction (1).
    PriorSensitivitySelectedStateMinimumSupport: prior sensitivity selected state minimum support (1).
    InferenceStatus: inference status (1).
    TierAStateCall: tier a state call (1).
    TierBStateCall: tier b state call (1).
    TierAExactConfiguration: tier a exact configuration (1).
    TierBExactConfiguration: tier b exact configuration (1).
    TierAExactPair: tier a exact pair (1).
    TierBExactPair: tier b exact pair (1).
    TierAParent1: tier a parent1 (1).
    TierAParent2: tier a parent2 (1).
    TierBParent1: tier b parent1 (1).
    TierBParent2: tier b parent2 (1).
    ConfigurationSupportSet: number of ordered items in pedigree/cell_items.csv (count).
    Parent1CandidateSet: number of ordered items in pedigree/cell_items.csv (count).
    Parent2CandidateSet: number of ordered items in pedigree/cell_items.csv (count).
    EvidenceConfigurationSet: number of ordered items in pedigree/cell_items.csv (count).
    TierBFinalStateCall: tier b final state call (1).
    TierBReleaseEvidence: tier b release evidence (1).
"""

CALL_EXPLANATIONS_COLUMNS = (
    "Sample",
    "ParentState",
    "ObservedParentCount",
    "Parent1",
    "Parent2",
    "Parent1Candidates",
    "Parent2Candidates",
    "ConfigurationCandidates",
    "ExactConfigurationResolved",
    "InferenceStatus",
    "Parent1Supported",
    "Parent2Supported",
    "ObservedParentCountMinimum",
    "ObservedParentCountMaximum",
    "LocalWinnerParentState",
    "SelectedParentState",
    "LocalParent1",
    "LocalParent2",
    "GraphConflict",
    "StateWinnerMargin",
    "ConditionalIdentityMargin",
    "LocalStateBootstrapFraction",
    "LocalStateLOCOFraction",
    "EligibleM2Pairs",
    "ScoredM2Pairs",
    "IncompleteM2Search",
    "GraphChangedLocalConfiguration",
    "Interpretation",
    "TierBFinalStateCall",
    "TierBReleaseEvidence",
)
"""Columns in published order.

    Sample: sample identifier (text).
    ParentState: released M0/M1/M2 state label (text).
    ObservedParentCount: released sequenced-parent count; unresolved values remain NaN (1).
    Parent1: first parent identifier; empty for an unobserved parent (text).
    Parent2: second parent identifier; empty for an unobserved parent (text).
    Parent1Candidates: number of ordered items in pedigree/cell_items.csv (count).
    Parent2Candidates: number of ordered items in pedigree/cell_items.csv (count).
    ConfigurationCandidates: number of ordered items in pedigree/cell_items.csv (count).
    ExactConfigurationResolved: exact configuration resolved (1).
    InferenceStatus: inference status (1).
    Parent1Supported: parent1 supported (1).
    Parent2Supported: parent2 supported (1).
    ObservedParentCountMinimum: observed parent count minimum (1).
    ObservedParentCountMaximum: observed parent count maximum (1).
    LocalWinnerParentState: local winning M0/M1/M2 state label (text).
    SelectedParentState: selected M0/M1/M2 state label (text).
    LocalParent1: local parent1 (1).
    LocalParent2: local parent2 (1).
    GraphConflict: graph conflict (1).
    StateWinnerMargin: state winner margin (nats).
    ConditionalIdentityMargin: conditional identity margin (nats).
    LocalStateBootstrapFraction: local state bootstrap fraction (1).
    LocalStateLOCOFraction: local state loco fraction (1).
    EligibleM2Pairs: eligible m2 pairs (1).
    ScoredM2Pairs: scored m2 pairs (1).
    IncompleteM2Search: incomplete m2 search (1).
    GraphChangedLocalConfiguration: graph changed local configuration (1).
    Interpretation: interpretation (1).
    TierBFinalStateCall: tier b final state call (1).
    TierBReleaseEvidence: tier b release evidence (1).
"""

ALTERNATIVE_EXPLANATIONS_COLUMNS = (
    "Sample",
    "ParentState",
    "Parent1",
    "Parent2",
    "LocalWinner",
    "GraphWinner",
    "WithinStateRank",
    "GeneticLogEvidence",
    "StructuralStateLogEvidence",
    "StructuralIdentityEligible",
    "DirectionLogContribution",
    "ReciprocalFamilyLogContribution",
    "AncestryPathLogContribution",
    "AdjustedStateLogEvidence",
    "AdjustedIdentityLogEvidence",
    "ParentStateLogPrior",
    "LogEligibleIdentityMultiplicity",
    "MarginalStateLogEvidence",
    "GraphDecisionUtility",
)
"""Columns in published order.

    Sample: sample identifier (text).
    ParentState: released M0/M1/M2 state label (text).
    Parent1: first parent identifier; empty for an unobserved parent (text).
    Parent2: second parent identifier; empty for an unobserved parent (text).
    LocalWinner: local winner (1).
    GraphWinner: graph winner (1).
    WithinStateRank: within state rank (1).
    GeneticLogEvidence: genetic log evidence (nats).
    StructuralStateLogEvidence: structural state log evidence (nats).
    StructuralIdentityEligible: structural identity eligible (1).
    DirectionLogContribution: direction log contribution (nats).
    ReciprocalFamilyLogContribution: reciprocal family log contribution (nats).
    AncestryPathLogContribution: ancestry path log contribution (nats).
    AdjustedStateLogEvidence: adjusted state log evidence (nats).
    AdjustedIdentityLogEvidence: adjusted identity log evidence (nats).
    ParentStateLogPrior: parent state log prior (nats).
    LogEligibleIdentityMultiplicity: log eligible identity multiplicity (nats).
    MarginalStateLogEvidence: marginal state log evidence (nats).
    GraphDecisionUtility: graph decision utility (nats).
"""

SEARCH_DIAGNOSTICS_COLUMNS = (
    "Sample",
    "EligibleParents",
    "ScoredM2Pairs",
    "M2PanelParents",
    "ScreenBoundaryRank",
    "ScreenBoundaryGap",
    "OmittedParentCount",
    "StrongestOmittedParent",
    "Parent1ScreenRank",
    "Parent2ScreenRank",
    "SelectedParentAtScreenBoundary",
    "M2PanelCapped",
)
"""Columns in published order.

    Sample: sample identifier (text).
    EligibleParents: eligible parents (1).
    ScoredM2Pairs: scored m2 pairs (1).
    M2PanelParents: m2 panel parents (1).
    ScreenBoundaryRank: screen boundary rank (1).
    ScreenBoundaryGap: screen boundary gap (nats).
    OmittedParentCount: omitted parent count (1).
    StrongestOmittedParent: strongest omitted parent (1).
    Parent1ScreenRank: parent1 screen rank (1).
    Parent2ScreenRank: parent2 screen rank (1).
    SelectedParentAtScreenBoundary: selected parent at screen boundary (1).
    M2PanelCapped: the child's admitted M2 pairs exceeded the per-child cap, so its panel is incomplete (1).
"""

EXCLUSION_EVIDENCE_COLUMNS = ("Sample", "CandidateParent", "LogE", "ObservedSites")
"""Columns in published order.

    Sample: sample identifier (text).
    CandidateParent: candidate parent (1).
    LogE: exclusion log-evidence (nats).
    ObservedSites: observed sites (1).
"""

EXCLUSION_DIAGNOSTICS_COLUMNS = (
    "Sample",
    "EligibleCandidates",
    "CompatibleCandidates",
    "SupportedReverseCandidates",
    "RemainingCandidates",
    "WeakestExclusionLogE",
    "ThresholdLogE",
    "CompleteCandidateEvidence",
    "Released",
    "ReleaseConditionalOnSupportedDirections",
)
"""Columns in published order.

    Sample: sample identifier (text).
    EligibleCandidates: eligible candidates (count).
    CompatibleCandidates: number of ordered items in pedigree/cell_items.csv (count).
    SupportedReverseCandidates: number of ordered items in pedigree/cell_items.csv (count).
    RemainingCandidates: number of ordered items in pedigree/cell_items.csv (count).
    WeakestExclusionLogE: weakest exclusion log-evidence used for the released count (nats).
    ThresholdLogE: required exclusion log-evidence threshold (nats).
    CompleteCandidateEvidence: complete candidate evidence (1).
    Released: released (1).
    ReleaseConditionalOnSupportedDirections: release conditional on supported directions (1).
"""

EVIDENCE_SUMMARY_COLUMNS = ("Contig", "InformativeMarkers", "MeanMinimumAncestryJunctions", "MeanCallableHaplotypeBins")
"""Columns in published order.

    Contig: contig identifier (text).
    InformativeMarkers: informative markers (1).
    MeanMinimumAncestryJunctions: mean minimum ancestry junctions (painting bins).
    MeanCallableHaplotypeBins: mean callable haplotype bins (1).
"""

CHROMOSOME_SUMMARY_COLUMNS = ("Contig", "ComponentCount", "ScoredComponentCount", "InformativeMarkers", "OmittedReason")
"""Columns in published order.

    Contig: contig identifier (text).
    ComponentCount: component count (1).
    ScoredComponentCount: scored component count (1).
    InformativeMarkers: informative markers (1).
    OmittedReason: omitted reason (1).
"""

RELATIVE_CONTRASTS_COLUMNS = (
    "Sample",
    "CandidateParent",
    "Views",
    "ProxyModel",
    "M1MinusSecondDegreeProxyNats",
    "M1MinusFullSibProxyNats",
    "UsableChromosomes",
    "NegativeChromosomesSecondDegree",
    "NegativeChromosomesFullSib",
    "MinChromosomeContrastSecondDegree",
)
"""Columns in published order.

    Sample: sample identifier (text).
    CandidateParent: candidate parent (1).
    Views: views (1).
    ProxyModel: proxy model (1).
    M1MinusSecondDegreeProxyNats: m1 minus second degree proxy nats (nats).
    M1MinusFullSibProxyNats: m1 minus full sib proxy nats (nats).
    UsableChromosomes: usable chromosomes (1).
    NegativeChromosomesSecondDegree: negative chromosomes second degree (1).
    NegativeChromosomesFullSib: negative chromosomes full sib (1).
    MinChromosomeContrastSecondDegree: min chromosome contrast second degree (1).
"""

SIBLING_GUARD_COLUMNS = (
    "Parent",
    "Child",
    "Witnesses",
    "SetAside",
    "Against",
    "Supporting",
    "PairScore",
    "FamilyScore",
    "Score",
    "Call",
    "EvidenceCall",
    "WithdrawalIteration",
    "WithholdingBasis",
)
"""Columns in published order.

    Parent: proposed sequenced parent identifier (text).
    Child: child identifier (text).
    Witnesses: final family members after linked cleaning, excluding child and proposed parent (count).
    SetAside: witnesses at their prior-odds price under both hypotheses (count).
    Against: witnesses contributing less than minus one nat (count).
    Supporting: witnesses contributing more than one nat (count).
    PairScore: feasible minus best four-process pair log likelihood (nats).
    FamilyScore: sum of priced witness contributions (nats).
    Score: pair plus family score Lambda (nats).
    Call: withheld, kept or untested; withdrawals are irreversible (text).
    EvidenceCall: call under final retained-edge premises (text).
    WithdrawalIteration: first withdrawal pass, zero when retained (count).
    WithholdingBasis: pair, family or none; first-withdrawal basis is retained (text).
"""

EXCLUSION_INTERVAL_COLUMNS = ("LogELower", "LogEUpper", "LogEExact")
"""Columns in published order.

    LogELower: lower exclusion log-evidence bound (nats).
    LogEUpper: upper exclusion log-evidence bound (nats).
    LogEExact: whether exclusion log-evidence is exact (1).
"""

COUNT_INTERVAL_COLUMNS = ("WeakestExclusionLogELower", "WeakestExclusionLogEUpper", "WeakestExclusionLogEExact")
"""Columns in published order.

    WeakestExclusionLogELower: weakest exclusion log e lower (nats).
    WeakestExclusionLogEUpper: weakest exclusion log e upper (nats).
    WeakestExclusionLogEExact: weakest exclusion log e exact (1).
"""

PARENT_SLOT_COLUMNS = ("ParentSlotBasis",)
"""Columns in published order.

    ParentSlotBasis: recorded unordered-parent orientation basis; no sex inference (text).
"""


# These sets grow with the cohort, including under diffuse support. Probability
# coverage is not a cardinality bound. Keep render()'s record-facing view intact;
# only publication separates the cells.
COLLECTION_COLUMNS = {
    "tier_b_candidate_sets": ("Parent1Candidates", "Parent2Candidates", "ConfigurationCandidates"),
    "call_explanations": ("Parent1Candidates", "Parent2Candidates", "ConfigurationCandidates"),
    "diagnostics": (
        "ConfigurationSupportSet",
        "Parent1CandidateSet",
        "Parent2CandidateSet",
        "EvidenceConfigurationSet",
    ),
    "exclusion_diagnostics": ("CompatibleCandidates", "SupportedReverseCandidates", "RemainingCandidates"),
    "sibling_guard": (),
}

CELL_ITEM_COLUMNS = ("source_table", "sample", "parent", "child", "column", "item", "value")
"""Columns in published order.

    source_table: parent pedigree table basename, without .csv (text).
    sample: parent row's sample key; empty for sibling_guard (text).
    parent: sibling_guard parent key; empty for per-sample tables (text).
    child: sibling_guard child key; empty for per-sample tables (text).
    column: original collection column in snake_case (text).
    item: zero-based item position in the original collection (1).
    value: JSON string or parent-configuration list of at most two sample names (text).
"""


def separate_collections(frames, views, names, input_ids=None):
    """Replace cohort-sized cells by counts and return their ordered lossless items.

    Tuple cells use one JSON value per tuple element. The custom brace/semicolon
    cells use their original formatted item text, preserving rounding and names.
    Empty collections have count zero and no child rows.
    """
    from ...core.columns import column_name

    items = []
    exclusion_fields = {
        "CompatibleCandidates": "compatible",
        "SupportedReverseCandidates": "reverse",
        "RemainingCandidates": "remaining",
    }
    for table, columns in COLLECTION_COLUMNS.items():
        frame = frames[table]
        if frame is None or frame.empty:
            continue
        rows = getattr(views, table)
        if table == "sibling_guard":
            rows = rows.rows
        order = list(range(len(rows)))
        if input_ids is not None and table != "sibling_guard":
            position = {name: index for index, name in enumerate(input_ids)}
            order.sort(key=lambda i: position.get(frame.iloc[i]["Sample"], len(position)))
        for column in columns:
            counts = [0] * len(rows)
            for index in order:
                row = rows[index]
                field = exclusion_fields.get(column, snake(column))
                value = getattr(row, field)
                if column in SET_FIELDS or column == "ConfigurationCandidates":
                    values = _value(column, value, names, table)
                elif column in {"Parent1CandidateSet", "Parent2CandidateSet"}:
                    values = tuple(str(names[i]) for i in value)
                else:
                    # Each item has at most two parent names (support/configuration)
                    # or one sample/reason pair (the sibling guard).
                    values = tuple(_value(column, (item,), names, table) for item in value)
                counts[index] = len(values)
                keys = (
                    dict(sample=None, parent=frame.iloc[index]["Parent"], child=frame.iloc[index]["Child"])
                    if table == "sibling_guard"
                    else dict(sample=frame.iloc[index]["Sample"], parent=None, child=None)
                )
                for ordinal, item in enumerate(values):
                    items.append(
                        dict(
                            source_table=table,
                            **keys,
                            column=column_name(column),
                            item=ordinal,
                            value=json.dumps(item, sort_keys=True),
                        )
                    )
            frame[column] = counts
    return items
