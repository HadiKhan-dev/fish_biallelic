"""Exact V7 bounded likelihood-margin aggregation and tier policies."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .design import CONTIGS
from .provenance import validate_cache_manifest
from .ranking import descending_rank_votes
from . import model


MODEL_REVISION = "v7_phase_invariant__bounded_likelihood_margin_aggregation"
RANK_WEIGHT = 0.9
MARGIN_WEIGHT = 0.1
BLOCK_TEMPERING_POWER = 0.5
CHROMOSOME_CONTAMINATION = 0.01
TIER_B_VARIANT_MINIMUM = 5
TIER_B_LOCO_MINIMUM = 18


def robust_information_weighted_utilities(
    scores,
    marker_counts,
    markers_per_block,
    rank_weight=RANK_WEIGHT,
    tempering_power=BLOCK_TEMPERING_POWER,
    contamination=CHROMOSOME_CONTAMINATION,
):
    """Blend robust ranks with bounded soft likelihood-margin evidence.

    ``scores`` contains one block-composite linked-HMM log likelihood for each
    chromosome, child, and candidate pair.  Dividing log-likelihood margins by
    ``n_blocks ** 0.5`` retains square-root information growth without allowing
    long chromosomes to dominate linearly.  A uniform contamination mixture
    bounds the contribution of a badly misspecified chromosome.
    """
    values = np.asarray(scores, dtype=np.float64)
    markers = np.asarray(marker_counts, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("scores must have shape (contigs, children, pairs)")
    if markers.shape != (values.shape[0],):
        raise ValueError("marker_counts must contain one value per contig")
    if values.shape[2] < 2:
        raise ValueError("at least two candidate pairs are required")
    if np.any(~np.isfinite(values)):
        raise ValueError("scores must be finite")
    if np.any(~np.isfinite(markers)) or np.any(markers <= 0.0):
        raise ValueError("marker counts must be finite and positive")
    if int(markers_per_block) != markers_per_block or markers_per_block < 1:
        raise ValueError("markers_per_block must be a positive integer")
    if not 0.0 <= rank_weight <= 1.0:
        raise ValueError("rank_weight must lie in [0, 1]")
    if not np.isfinite(tempering_power) or tempering_power < 0.0:
        raise ValueError("tempering_power must be finite and non-negative")
    if not 0.0 <= contamination < 1.0:
        raise ValueError("contamination must lie in [0, 1)")

    n_blocks = np.maximum(
        np.ceil(markers / float(markers_per_block)),
        1.0,
    )
    centered = values - np.max(values, axis=2, keepdims=True)
    centered /= n_blocks[:, None, None] ** float(tempering_power)
    soft_evidence = np.exp(centered)
    soft_evidence /= np.sum(soft_evidence, axis=2, keepdims=True)
    soft_evidence = (
        (1.0 - contamination) * soft_evidence
        + contamination / values.shape[2]
    )
    rank_evidence = descending_rank_votes(values)
    utilities = (
        float(rank_weight) * rank_evidence
        + (1.0 - float(rank_weight)) * soft_evidence
    )
    if np.any(~np.isfinite(utilities)):
        raise RuntimeError("non-finite margin aggregation utility")
    return utilities


def _utility_transform(marker_counts):
    specifications = model._variant_specifications()

    def transform(scores, variant_index):
        return robust_information_weighted_utilities(
            scores,
            marker_counts,
            specifications[int(variant_index)][1],
        )

    return transform


def _tier_b_flags(assignments):
    """Return requested 5/7-variant and 18/22-LOCO Tier B flags."""
    result = assignments.copy()
    variant_minimum = TIER_B_VARIANT_MINIMUM / len(model.VARIANT_LABELS)
    loco_minimum = TIER_B_LOCO_MINIMUM / len(CONTIGS)
    as_bool = lambda column: result[column].astype(bool)

    state_support = (
        result["parent_count_MAP_state"].eq("two_parent")
        & (result["parent_count_bootstrap_selection_fraction"] > 0.5)
        & (result["parent_count_variant_fraction"] >= variant_minimum)
        & (result["parent_count_leave_one_fraction"] >= loco_minimum)
        & (result["parent_count_prior_fraction"] >= (2.0 / 3.0))
    )
    pair_identity = (
        as_bool("joint_and_marginal_father_agree")
        & as_bool("joint_and_marginal_mother_agree")
        & (result["chromosome_bootstrap_pair_selection_fraction"] > 0.5)
        & (result["chromosome_bootstrap_father_selection_fraction"] > 0.5)
        & (result["chromosome_bootstrap_mother_selection_fraction"] > 0.5)
        & (result["pair_variant_fraction"] >= variant_minimum)
        & (result["father_variant_fraction"] >= variant_minimum)
        & (result["mother_variant_fraction"] >= variant_minimum)
        & (result["pair_leave_one_fraction"] >= loco_minimum)
        & (result["father_leave_one_fraction"] >= loco_minimum)
        & (result["mother_leave_one_fraction"] >= loco_minimum)
    )
    role_flags = {}
    for role in ("father", "mother"):
        identity = (
            as_bool(f"joint_and_marginal_{role}_agree")
            & (
                result[
                    f"chromosome_bootstrap_{role}_selection_fraction"
                ]
                > 0.5
            )
            & (result[f"{role}_variant_fraction"] >= variant_minimum)
            & (result[f"{role}_leave_one_fraction"] >= loco_minimum)
        )
        presence = (
            as_bool(f"{role}_observed_MAP")
            & (result[f"{role}_observed_bootstrap_fraction"] > 0.5)
            & (
                result[f"{role}_observed_variant_fraction"]
                >= variant_minimum
            )
            & (
                result[f"{role}_observed_leave_one_fraction"]
                >= loco_minimum
            )
            & (result[f"{role}_observed_prior_fraction"] >= (2.0 / 3.0))
        )
        role_flags[role] = identity & presence

    result["tier_A_father"] = as_bool("father_stable")
    result["tier_A_mother"] = as_bool("mother_stable")
    result["tier_A_exact_pair"] = as_bool("exact_pair_stable")
    result["tier_B_or_better_father"] = (
        role_flags["father"] | result["tier_A_father"]
    )
    result["tier_B_or_better_mother"] = (
        role_flags["mother"] | result["tier_A_mother"]
    )
    result["tier_B_or_better_exact_pair"] = (
        (state_support & pair_identity) | result["tier_A_exact_pair"]
    )
    if np.any(
        result["tier_B_or_better_exact_pair"]
        & ~(
            result["tier_B_or_better_father"]
            & result["tier_B_or_better_mother"]
        )
    ):
        raise RuntimeError("Tier B exact pair lacks both Tier B parent edges")

    labels = []
    for row in result.itertuples(index=False):
        if row.tier_A_exact_pair:
            label = "A_exact_pair"
        elif row.tier_B_or_better_exact_pair:
            label = "B_exact_pair"
        elif row.tier_A_father and row.tier_A_mother:
            label = "A_both_parent_identities_pair_ambiguous"
        elif row.tier_B_or_better_father and row.tier_B_or_better_mother:
            label = "B_both_parent_identities_pair_ambiguous"
        elif row.tier_A_father or row.tier_A_mother:
            label = "A_one_parent_identity"
        elif row.tier_B_or_better_father or row.tier_B_or_better_mother:
            label = "B_one_parent_identity"
        else:
            label = "C_not_yet_defined"
        labels.append(label)
    result["tiered_evidence_class"] = labels
    return result

def _held_out_metrics(raw_scores, utilities, marker_counts, block_size):
    raw = np.asarray(raw_scores, dtype=np.float64)
    ranks = descending_rank_votes(raw)
    total = np.sum(utilities, axis=0)
    rows = np.arange(raw.shape[1])
    rank_values = []
    sqrt_regrets = []
    n_blocks = np.maximum(
        np.ceil(np.asarray(marker_counts) / float(block_size)),
        1.0,
    )
    for contig in range(raw.shape[0]):
        winner = np.argmax(total - utilities[contig], axis=1)
        rank_values.extend(ranks[contig, rows, winner])
        regret = (
            np.max(raw[contig], axis=1)
            - raw[contig, rows, winner]
        )
        sqrt_regrets.extend(regret / np.sqrt(n_blocks[contig]))
    return {
        "mean_held_out_rank_utility": float(np.mean(rank_values)),
        "mean_held_out_sqrt_block_regret": float(np.mean(sqrt_regrets)),
    }


@dataclass(frozen=True)
class V7MarginResult:
    """In-memory outputs from exact V7-margin compatibility aggregation."""

    assignments: pd.DataFrame
    candidate_evidence: pd.DataFrame
    parent_states: pd.DataFrame
    arrays: dict
    marker_counts: np.ndarray


def aggregate_v7_margin(
    cache_dir,
    metadata,
    candidates,
    expected_cache_manifest,
    bootstrap_replicates=2000,
    allow_legacy_cache_without_manifest=False,
):
    """Aggregate complete V7 contig caches using the selected margin model."""
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    cache_dir = Path(cache_dir)
    validate_cache_manifest(
        cache_dir,
        expected_cache_manifest,
        allow_legacy_cache_without_manifest=(
            allow_legacy_cache_without_manifest
        ),
    )
    marker_counts = np.asarray([
        int(np.load(cache_dir / f"v7_{contig}.npz")["markers"])
        for contig in CONTIGS
    ], dtype=np.int64)
    arrays = model._load_contig_arrays(cache_dir)
    scoring = {
        "f1_pairs": np.asarray(candidates["f1_pairs"], dtype=np.int64),
        "f2_children": np.asarray(candidates["f2_children"], dtype=np.int64),
    }
    pair_assignments, candidate_evidence = model._build_assignment_tables(
        arrays,
        scoring,
        metadata,
        bootstrap_replicates,
        utility_transform=_utility_transform(marker_counts),
    )
    candidate_evidence = candidate_evidence.rename(columns={
        "primary_mean_rank_utility": "primary_mean_margin_utility"
    })
    parent_states = model._build_parent_state_table(
        arrays, scoring, metadata, bootstrap_replicates
    )
    assignments = model._apply_parent_state_support(
        pair_assignments, parent_states
    )
    assignments = _tier_b_flags(assignments)
    return V7MarginResult(
        assignments=assignments,
        candidate_evidence=candidate_evidence,
        parent_states=parent_states,
        arrays=arrays,
        marker_counts=marker_counts,
    )
