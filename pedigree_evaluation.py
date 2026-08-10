"""Stable evaluation primitives for simulated pedigree truth comparisons."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def observed_parent_set(values: Iterable[object]) -> set[object]:
    """Return non-missing parent labels without imposing parental order."""
    return {value for value in values if not pd.isna(value)}


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
