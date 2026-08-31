"""Stable evaluation primitives for simulated pedigree truth comparisons."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def observed_parent_set(values: Iterable[object]) -> set[object]:
    """Return non-missing parent labels without imposing parental order."""
    return {value for value in values if not pd.isna(value)}


OBSERVED_PARENT_STATES = (
    "zero_observed_parents",
    "one_observed_parent",
    "two_observed_parents",
)


def observed_parent_state(
    parents: Iterable[object], observed_samples: Iterable[object]
) -> str:
    """Classify truth by the number of distinct parents present in the cohort."""
    observed = set(observed_samples)
    present = {
        value for value in parents
        if not pd.isna(value) and value in observed
    }
    if len(present) > 2:
        raise ValueError(
            "a diploid pedigree row cannot have over two observed parents"
        )
    return OBSERVED_PARENT_STATES[len(present)]


def observed_parent_sets_match(
    true_parents: Iterable[object], inferred_parents: Iterable[object],
    observed_samples: Iterable[object],
) -> bool:
    """Compare only biological parents represented by sequenced samples."""
    observed = set(observed_samples)
    true_observed = {
        value for value in true_parents
        if not pd.isna(value) and value in observed
    }
    return true_observed == observed_parent_set(inferred_parents)


def truth_parent_state_series(pedigree: pd.DataFrame) -> pd.Series:
    """Return known-truth M0/M1/M2 states in pedigree row order."""
    required = {"Sample", "Parent1", "Parent2"}
    missing = required.difference(pedigree.columns)
    if missing:
        raise ValueError(f"truth pedigree is missing columns {sorted(missing)}")
    observed = set(pedigree["Sample"])
    return pedigree.apply(
        lambda row: observed_parent_state(
            (row["Parent1"], row["Parent2"]), observed
        ),
        axis=1,
    )


def unordered_parents_match(
    true_parents: Iterable[object],
    inferred_parents: Iterable[object],
    *,
    unobserved_founder_label: str = "Founder",
) -> bool:
    """Apply the simulation pipeline's unordered parent-matching rule."""
    true_parent_set = observed_parent_set(true_parents)
    inferred_parent_set = observed_parent_set(inferred_parents)
    if any(unobserved_founder_label in str(value) for value in true_parent_set):
        return not inferred_parent_set
    return true_parent_set == inferred_parent_set


def parent_columns_match(row) -> bool:
    """Adapt the standard merged simulation DataFrame columns."""
    return unordered_parents_match(
        (row["Parent1_True"], row["Parent2_True"]),
        (row["Parent1_Inf"], row["Parent2_Inf"]),
    )


def compare_relationships_to_truth(
    truth_pedigree: pd.DataFrame, inferred_relationships: pd.DataFrame
) -> pd.DataFrame:
    """Align one inferred view with known truth without inventing generations."""
    truth_required = {"Sample", "Generation", "Parent1", "Parent2"}
    inferred_required = {"Sample", "ParentState", "Parent1", "Parent2"}
    missing_truth = truth_required.difference(truth_pedigree.columns)
    missing_inferred = inferred_required.difference(inferred_relationships.columns)
    if missing_truth:
        raise ValueError(
            f"truth pedigree is missing columns {sorted(missing_truth)}"
        )
    if missing_inferred:
        raise ValueError(
            "inferred relationships are missing columns "
            f"{sorted(missing_inferred)}"
        )

    truth = truth_pedigree.loc[:, [
        "Sample", "Generation", "Parent1", "Parent2"
    ]].copy()
    truth = truth.rename(columns={"Generation": "TruthGeneration"})
    truth["ParentState"] = truth_parent_state_series(truth_pedigree)
    inferred = inferred_relationships.loc[:, [
        "Sample", "ParentState", "Parent1", "Parent2"
    ]]
    comparison = pd.merge(
        truth, inferred, on="Sample", suffixes=("_True", "_Inf"),
        validate="one_to_one",
    )
    comparison["ParentState_Match"] = (
        comparison["ParentState_True"] == comparison["ParentState_Inf"]
    )
    observed_samples = set(truth_pedigree["Sample"])
    comparison["Parents_Match"] = comparison.apply(
        lambda row: observed_parent_sets_match(
            (row["Parent1_True"], row["Parent2_True"]),
            (row["Parent1_Inf"], row["Parent2_Inf"]),
            observed_samples,
        ),
        axis=1,
    )
    return comparison
