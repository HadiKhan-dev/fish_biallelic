"""Pedigree tables derived from V7-margin assignments and explicit seeds."""

import numpy as np
import pandas as pd


MODEL_NAME = "v7_phase_invariant__bounded_likelihood_margin_aggregation"


def _empty_row(individual):
    return {
        "EntityID": individual.Sample,
        "Sample": individual.Sample,
        "Alias": individual.Alias,
        "SantosID": individual.SantosID,
        "EntityType": "sequenced_individual",
        "MetadataGeneration": individual.Generation,
        "PedigreeGeneration": individual.Generation,
        "Sex": individual.Sex,
        "Parent1": None,
        "Parent2": None,
        "Parent1Alias": None,
        "Parent2Alias": None,
        "CandidateParent1": None,
        "CandidateParent2": None,
        "CandidateParent1Alias": None,
        "CandidateParent2Alias": None,
        "Parent1Role": None,
        "Parent2Role": None,
        "Parent1Alternatives95": None,
        "Parent2Alternatives95": None,
        "PairAlternatives95": None,
        "PairDiagnosticScope": None,
        "Parent1Stable": None,
        "Parent2Stable": None,
        "ExactPairStable": None,
        "ExactPairBootstrapFraction": np.nan,
        "Parent1BootstrapFraction": np.nan,
        "Parent2BootstrapFraction": np.nan,
        "InferenceModel": MODEL_NAME,
        "EvidencePolicy": None,
        "EvidenceTier": None,
        "EvidenceStatus": None,
        "SeedStatus": None,
        "SeedReportParentEdges": None,
        "ParentCountMAPState": None,
        "ParentCountStable": None,
        "ParentCountBootstrapFraction": np.nan,
        "MissingBiologicalParentCount": np.nan,
        "Notes": None,
    }


def _set_parent(row, slot, sample, alias):
    row[f"Parent{slot}"] = sample
    row[f"Parent{slot}Alias"] = alias


def _f1_row(base, seed, policy):
    row = dict(base)
    if not isinstance(seed.report_parent_edges, (bool, np.bool_)):
        raise ValueError("report_parent_edges must be a parsed boolean")
    report_edges = bool(seed.report_parent_edges)
    seed_status = str(seed.seed_status)
    reportable_statuses = {
        "exact_stable_inferred", "documented_breeding_record",
    }
    if seed_status not in reportable_statuses | {"computational_seed_only"}:
        raise ValueError(f"unrecognized seed_status: {seed_status!r}")
    if report_edges and seed_status not in reportable_statuses:
        raise ValueError(
            "a computational_seed_only pair cannot be emitted as parent edges"
        )
    row.update({
        "CandidateParent1": seed.father,
        "CandidateParent2": seed.mother,
        "CandidateParent1Alias": seed.father_alias,
        "CandidateParent2Alias": seed.mother_alias,
        "Parent1Role": "father",
        "Parent2Role": "mother",
        "Parent1Stable": report_edges,
        "Parent2Stable": report_edges,
        "ExactPairStable": report_edges,
        "EvidencePolicy": policy,
        "EvidenceTier": f"fixed_input__{seed_status}",
        "EvidenceStatus": f"explicit_G0_pair_seed__{seed_status}",
        "SeedStatus": seed_status,
        "SeedReportParentEdges": report_edges,
        "ParentCountMAPState": (
            "fixed_input_two_parent" if report_edges
            else "fixed_input_pair_not_reported"
        ),
        "ParentCountStable": report_edges,
        "MissingBiologicalParentCount": 0 if report_edges else np.nan,
        "Notes": (
            "This G0 pair was supplied explicitly to reconstruct the F1's "
            "parent-of-origin homologs. Its status is fixed input provenance, "
            "not a result inferred by this run. "
            + (
                "The input authorizes reporting these parent edges. "
                if report_edges else
                "The pair is computational-only and its parent edges are not "
                "reported. "
            )
            + f"Seed status: {seed_status}. Basis: {seed.seed_basis}"
        ),
    })
    if report_edges:
        _set_parent(row, 1, seed.father, seed.father_alias)
        _set_parent(row, 2, seed.mother, seed.mother_alias)
    return row


def _f2_row(base, assignment, policy):
    row = dict(base)
    father_present = pd.notna(assignment.reported_father)
    mother_present = pd.notna(assignment.reported_mother)
    if father_present:
        row["CandidateParent1"] = assignment.reported_father
        row["CandidateParent1Alias"] = assignment.reported_father_alias
    if mother_present:
        row["CandidateParent2"] = assignment.reported_mother
        row["CandidateParent2Alias"] = assignment.reported_mother_alias
    father_alternatives = assignment.father_resampling_95_set
    mother_alternatives = assignment.mother_resampling_95_set
    father_bootstrap = (
        assignment.chromosome_bootstrap_father_selection_fraction
    )
    mother_bootstrap = (
        assignment.chromosome_bootstrap_mother_selection_fraction
    )
    if assignment.parent_count_MAP_state == "father_only":
        father_alternatives = assignment.father_only_resampling_95_set
        father_bootstrap = (
            assignment.father_only_bootstrap_selection_fraction
        )
    if assignment.parent_count_MAP_state == "mother_only":
        mother_alternatives = assignment.mother_only_resampling_95_set
        mother_bootstrap = (
            assignment.mother_only_bootstrap_selection_fraction
        )
    if not father_present:
        father_alternatives = (
            "no observed father in leading parent-count state"
        )
        father_bootstrap = np.nan
    if not mother_present:
        mother_alternatives = (
            "no observed mother in leading parent-count state"
        )
        mother_bootstrap = np.nan
    pair_alternatives = assignment.pair_resampling_95_set
    if assignment.parent_count_MAP_state == "two_parent":
        pair_diagnostic_scope = "selected_two_parent_state"
    else:
        pair_diagnostic_scope = (
            "conditional_two_parent_identity_diagnostic_not_selected_state"
        )
        pair_alternatives = (
            "CONDITIONAL_TWO_PARENT_DIAGNOSTIC_NOT_SELECTED_STATE; "
            f"{pair_alternatives}"
        )
    row.update({
        "Parent1Role": "father",
        "Parent2Role": "mother",
        "Parent1Alternatives95": father_alternatives,
        "Parent2Alternatives95": mother_alternatives,
        "PairAlternatives95": pair_alternatives,
        "PairDiagnosticScope": pair_diagnostic_scope,
        "ExactPairBootstrapFraction": (
            assignment.chromosome_bootstrap_pair_selection_fraction
        ),
        "Parent1BootstrapFraction": father_bootstrap,
        "Parent2BootstrapFraction": mother_bootstrap,
        "EvidencePolicy": policy,
        "EvidenceTier": assignment.tiered_evidence_class,
        "EvidenceStatus": assignment.evidence_class,
        "ParentCountMAPState": assignment.parent_count_MAP_state,
        "ParentCountStable": bool(assignment.parent_count_state_stable),
        "ParentCountBootstrapFraction": (
            assignment.parent_count_bootstrap_selection_fraction
        ),
        "MissingBiologicalParentCount": (
            {
                "zero_parent": 2,
                "father_only": 1,
                "mother_only": 1,
                "two_parent": 0,
            }[assignment.parent_count_MAP_state]
            if assignment.parent_count_state_stable
            else np.nan
        ),
        "Notes": (
            "Candidate identities are leading V7-margin hypotheses. Tier flags "
            "describe internal model and chromosome stability, not calibrated "
            "real-data error probabilities. "
            + (
                "Pair-level fields are conditional two-parent diagnostics "
                "because the selected parent-count state is not two-parent."
                if assignment.parent_count_MAP_state != "two_parent" else ""
            )
        ),
    })
    if policy == "leading_hypothesis":
        father_call = father_present
        mother_call = mother_present
        exact_call = bool(assignment.tier_A_exact_pair)
        father_stability = bool(assignment.tier_A_father)
        mother_stability = bool(assignment.tier_A_mother)
    elif policy == "tier_A":
        father_call = father_present and bool(assignment.tier_A_father)
        mother_call = mother_present and bool(assignment.tier_A_mother)
        exact_call = bool(assignment.tier_A_exact_pair)
        father_stability = bool(assignment.tier_A_father)
        mother_stability = bool(assignment.tier_A_mother)
    elif policy == "tier_B_or_better":
        father_call = father_present and bool(
            assignment.tier_B_or_better_father
        )
        mother_call = mother_present and bool(
            assignment.tier_B_or_better_mother
        )
        exact_call = bool(assignment.tier_B_or_better_exact_pair)
        father_stability = bool(assignment.tier_B_or_better_father)
        mother_stability = bool(assignment.tier_B_or_better_mother)
    else:
        raise ValueError(f"unknown pedigree evidence policy: {policy}")
    row["Parent1Stable"] = father_stability
    row["Parent2Stable"] = mother_stability
    row["ExactPairStable"] = exact_call
    if father_call:
        _set_parent(
            row, 1, assignment.reported_father,
            assignment.reported_father_alias,
        )
    if mother_call:
        _set_parent(
            row, 2, assignment.reported_mother,
            assignment.reported_mother_alias,
        )
    return row


def build_pedigree(metadata, seeds, assignments, policy):
    """Build one pedigree table under a named evidence reporting policy."""
    seeds_by_child = seeds.set_index("child_index")
    assignments_by_child = assignments.set_index("child_index")
    rows = []
    for individual in metadata.itertuples(index=False):
        base = _empty_row(individual)
        if individual.Generation == "G0":
            base.update({
                "EvidencePolicy": policy,
                "EvidenceTier": "metadata_root",
                "EvidenceStatus": "sequenced_G0_candidate_root",
                "MissingBiologicalParentCount": 0,
                "Notes": (
                    "Sequenced G0 cohort member; no parent assigned. Cohort "
                    "membership alone does not establish any descendant."
                ),
            })
            rows.append(base)
        elif individual.Generation == "F1":
            if individual.sample_index not in seeds_by_child.index:
                raise RuntimeError("missing explicit G0 seed for an F1 individual")
            rows.append(_f1_row(
                base, seeds_by_child.loc[individual.sample_index], policy
            ))
        elif individual.Generation == "F2":
            if individual.sample_index not in assignments_by_child.index:
                raise RuntimeError("missing V7-margin assignment for an F2 individual")
            rows.append(_f2_row(
                base, assignments_by_child.loc[individual.sample_index], policy
            ))
        else:
            raise RuntimeError(
                f"V7 compatibility output cannot place generation "
                f"{individual.Generation!r}"
            )
    result = pd.DataFrame(rows)
    validate_pedigree(result, metadata)
    return result


def validate_pedigree(pedigree, metadata):
    """Validate entity references and the explicit V7 eligibility constraints."""
    if len(pedigree) != len(metadata) or pedigree["EntityID"].duplicated().any():
        raise RuntimeError("pedigree entity set does not equal metadata")
    by_sample = metadata.set_index("Sample")
    entities = set(pedigree["EntityID"])
    for child in pedigree.itertuples(index=False):
        for slot, expected_sex in ((1, "M"), (2, "F")):
            parent = getattr(child, f"Parent{slot}")
            alias = getattr(child, f"Parent{slot}Alias")
            if pd.isna(parent):
                if pd.notna(alias):
                    raise RuntimeError("parent alias exists without parent")
                continue
            if parent not in entities:
                raise RuntimeError(f"unknown parent entity {parent}")
            if by_sample.at[parent, "Alias"] != alias:
                raise RuntimeError("parent alias does not match metadata")
            if by_sample.at[parent, "Sex"] != expected_sex:
                raise RuntimeError("reported parent has ineligible sex")
            expected_generation = "G0" if child.MetadataGeneration == "F1" else "F1"
            if by_sample.at[parent, "Generation"] != expected_generation:
                raise RuntimeError("reported parent has ineligible generation")


def pedigree_edges(pedigree):
    """Return one row per reported parent-child edge."""
    rows = []
    for child in pedigree.itertuples(index=False):
        for slot in (1, 2):
            parent = getattr(child, f"Parent{slot}")
            if pd.isna(parent):
                continue
            rows.append({
                "ParentEntityID": parent,
                "ParentAlias": getattr(child, f"Parent{slot}Alias"),
                "ChildEntityID": child.EntityID,
                "ChildAlias": child.Alias,
                "ChildGeneration": child.PedigreeGeneration,
                "ParentSlot": slot,
                "EvidencePolicy": child.EvidencePolicy,
                "EvidenceTier": child.EvidenceTier,
                "EvidenceStatus": child.EvidenceStatus,
                "SeedStatus": child.SeedStatus,
                "SeedReportParentEdges": child.SeedReportParentEdges,
            })
    return pd.DataFrame(rows)


def pedigree_fam(pedigree, family_id="Tropheops_cross"):
    """Return a six-column PLINK-compatible FAM representation."""
    values = []
    for row in pedigree.itertuples(index=False):
        sex = 1 if row.Sex == "M" else 2 if row.Sex == "F" else 0
        values.append({
            "FID": family_id,
            "IID": row.EntityID,
            "PID": row.Parent1 if pd.notna(row.Parent1) else "0",
            "MID": row.Parent2 if pd.notna(row.Parent2) else "0",
            "SEX": sex,
            "PHENOTYPE": -9,
        })
    return pd.DataFrame(values)
