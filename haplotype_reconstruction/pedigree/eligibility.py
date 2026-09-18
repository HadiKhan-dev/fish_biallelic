"""Explicit child, parent and parent-pair eligibility rules."""
from __future__ import annotations


from dataclasses import dataclass, field


from typing import Any, Mapping, Optional, Sequence


import numpy as np


@dataclass(frozen=True)
class ParentEligibility:
    """Versioned caller-supplied child and observed-parent candidate universe.

    Sample order must exactly match inference ``sample_ids``. The optional M2
    mask is derived from M1 eligibility when omitted and otherwise must be
    symmetric in its final two axes. Smart never derives masks from metadata.
    """

    format_version: int
    sample_ids: Sequence[Any]
    eligible_children: np.ndarray
    eligible_parents: np.ndarray
    eligible_parent_pairs: Optional[np.ndarray] = None
    policy_name: str = "caller_supplied_parent_eligibility_v1"
    source_fields: Sequence[str] = ()
    assumptions: Sequence[str] = ()
    individual_parentage_ground_truth: bool = False

    direction_supported_parents: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _ResolvedParentEligibility:
    supplied: bool
    format_version: int
    sample_ids: tuple[Any, ...]
    eligible_children: np.ndarray
    eligible_parents: np.ndarray
    eligible_parent_pairs: Optional[np.ndarray]
    direction_supported_parents: np.ndarray
    policy_name: str
    source_fields: tuple[str, ...]
    assumptions: tuple[str, ...]
    individual_parentage_ground_truth: bool
    pair_policy: str


def _parent_eligibility_text_tuple(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise pedigree_models.PedigreeEvidenceError(f"parent eligibility {field} must be a sequence")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise pedigree_models.PedigreeEvidenceError(
            f"parent eligibility {field} must be a sequence"
        ) from exc
    if any(not isinstance(item, str) for item in values):
        raise pedigree_models.PedigreeEvidenceError(
            f"parent eligibility {field} entries must be strings"
        )
    return values


def _resolve_parent_eligibility(
    value: Optional[ParentEligibility | Mapping[str, Any]],
    sample_ids: Sequence[Any],
) -> _ResolvedParentEligibility:
    """Validate eligibility without expanding derived parent-pair policies."""
    samples = tuple(sample_ids)
    n_samples = len(samples)
    diagonal = np.arange(n_samples)
    if value is None:
        children = np.ones(n_samples, dtype=np.bool_)
        parents = np.ones((n_samples, n_samples), dtype=np.bool_)
        np.fill_diagonal(parents, False)
        return _ResolvedParentEligibility(
            supplied=False,
            format_version=pedigree_models.PARENT_ELIGIBILITY_FORMAT_VERSION,
            sample_ids=samples,
            eligible_children=children,
            eligible_parents=parents,
            eligible_parent_pairs=None,
            direction_supported_parents=np.zeros_like(parents),
            policy_name="all_samples_eligible_default_v1",
            source_fields=(),
            assumptions=(),
            individual_parentage_ground_truth=False,
            pair_policy="all_unordered_pairs_of_eligible_parents",
        )

    if isinstance(value, _ResolvedParentEligibility):
        if value.sample_ids != samples:
            raise pedigree_models.PedigreeEvidenceError(
                "parent eligibility sample_ids must exactly match inference sample order"
            )
        return value
    if isinstance(value, ParentEligibility):
        record = value
    elif isinstance(value, Mapping):
        required = (
            "format_version", "sample_ids", "eligible_children",
            "eligible_parents",
        )
        missing = [field for field in required if field not in value]
        if missing:
            raise pedigree_models.PedigreeEvidenceError(
                f"parent eligibility is missing required field {missing[0]!r}"
            )
        record = ParentEligibility(
            format_version=value["format_version"],
            sample_ids=value["sample_ids"],
            eligible_children=value["eligible_children"],
            eligible_parents=value["eligible_parents"],
            eligible_parent_pairs=value.get("eligible_parent_pairs"),
            direction_supported_parents=value.get(
                "direction_supported_parents"
            ),
            policy_name=value.get(
                "policy_name", "caller_supplied_parent_eligibility_v1"
            ),
            source_fields=value.get("source_fields", ()),
            assumptions=value.get("assumptions", ()),
            individual_parentage_ground_truth=value.get(
                "individual_parentage_ground_truth", False
            ),
        )
    else:
        raise pedigree_models.PedigreeEvidenceError(
            "parent_eligibility must be SmartParentEligibility or a mapping"
        )
    if (
        isinstance(record.format_version, (bool, np.bool_))
        or record.format_version != pedigree_models.PARENT_ELIGIBILITY_FORMAT_VERSION
    ):
        raise pedigree_models.PedigreeEvidenceError("unsupported parent eligibility format_version")
    try:
        record_samples = tuple(record.sample_ids)
    except TypeError as exc:
        raise pedigree_models.PedigreeEvidenceError(
            "parent eligibility sample_ids must be an ordered sequence"
        ) from exc
    if len(record_samples) != n_samples or any(
        observed != expected
        for observed, expected in zip(record_samples, samples)
    ):
        raise pedigree_models.PedigreeEvidenceError(
            "parent eligibility sample_ids must exactly match inference sample order"
        )

    children_raw = np.asarray(record.eligible_children)
    parents_raw = np.asarray(record.eligible_parents)
    if children_raw.shape != (n_samples,) or children_raw.dtype != np.bool_:
        raise pedigree_models.PedigreeEvidenceError(
            "eligible_children must be a boolean array with shape (samples,)"
        )
    if (
        parents_raw.shape != (n_samples, n_samples)
        or parents_raw.dtype != np.bool_
    ):
        raise pedigree_models.PedigreeEvidenceError(
            "eligible_parents must be a boolean array with shape "
            "(samples, samples)"
        )
    children = np.ascontiguousarray(children_raw)
    parents = np.ascontiguousarray(parents_raw)
    if np.any(parents[diagonal, diagonal]):
        raise pedigree_models.PedigreeEvidenceError("eligible_parents cannot admit self-parenting")
    if np.any(parents[~children]):
        raise pedigree_models.PedigreeEvidenceError(
            "excluded children cannot have eligible parent identities"
        )

    if record.direction_supported_parents is None:
        direction_supported = np.zeros_like(parents)
    else:
        direction_raw = np.asarray(record.direction_supported_parents)
        if (
            direction_raw.shape != (n_samples, n_samples)
            or direction_raw.dtype != np.bool_
        ):
            raise pedigree_models.PedigreeEvidenceError(
                "direction_supported_parents must be a boolean array with "
                "shape (samples, samples)"
            )
        direction_supported = np.ascontiguousarray(direction_raw)
        if np.any(direction_supported[diagonal, diagonal]):
            raise pedigree_models.PedigreeEvidenceError(
                "direction_supported_parents cannot admit self-parenting"
            )
        if np.any(direction_supported[~children]):
            raise pedigree_models.PedigreeEvidenceError(
                "excluded children cannot have explicitly direction-supported "
                "parents"
            )
        if np.any(direction_supported & ~parents):
            raise pedigree_models.PedigreeEvidenceError(
                "direction_supported_parents must be a subset of "
                "eligible_parents"
            )

    if record.eligible_parent_pairs is None:
        pairs = None
        pair_policy = "all_unordered_pairs_of_eligible_parents"
    else:
        pair_raw = np.asarray(record.eligible_parent_pairs)
        if (
            pair_raw.shape != (n_samples, n_samples, n_samples)
            or pair_raw.dtype != np.bool_
        ):
            raise pedigree_models.PedigreeEvidenceError(
                "eligible_parent_pairs must be a boolean array with shape "
                "(samples, samples, samples)"
            )
        pairs = np.ascontiguousarray(pair_raw)
        if not np.array_equal(pairs, np.swapaxes(pairs, 1, 2)):
            raise pedigree_models.PedigreeEvidenceError(
                "eligible_parent_pairs must be symmetric in the parent axes"
            )
        if np.any(pairs[:, diagonal, diagonal]):
            raise pedigree_models.PedigreeEvidenceError(
                "eligible_parent_pairs cannot contain duplicate parents"
            )
        for child in range(n_samples):
            child_allowed = (
                parents[child,:, None] & parents[child, None,:]
            )
            if np.any(pairs[child] & ~child_allowed):
                raise pedigree_models.PedigreeEvidenceError(
                    "eligible_parent_pairs may contain only eligible parent identities"
                )
        pair_policy = "explicit_symmetric_pair_mask"

    if not isinstance(record.policy_name, str) or not record.policy_name:
        raise pedigree_models.PedigreeEvidenceError(
            "parent eligibility policy_name must be a non-empty string"
        )
    source_fields = _parent_eligibility_text_tuple(
        record.source_fields, "source_fields"
    )
    assumptions = _parent_eligibility_text_tuple(
        record.assumptions, "assumptions"
    )
    if not isinstance(record.individual_parentage_ground_truth, (bool, np.bool_)):
        raise pedigree_models.PedigreeEvidenceError(
            "individual_parentage_ground_truth must be boolean"
        )
    return _ResolvedParentEligibility(
        True, pedigree_models.PARENT_ELIGIBILITY_FORMAT_VERSION, samples,
        children.copy(), parents.copy(), (
            None if pairs is None else np.ascontiguousarray(pairs).copy()
        ),
        direction_supported.copy(),
        record.policy_name, source_fields, assumptions,
        bool(record.individual_parentage_ground_truth), pair_policy,
    )


def _eligible_parent_pair(
    eligibility: _ResolvedParentEligibility,
    child: int,
    first_parent: int,
    second_parent: int,
) -> bool:
    """Return exact M2 membership for one unordered parent pair."""
    if first_parent == second_parent:
        return False
    pairs = eligibility.eligible_parent_pairs
    if pairs is not None:
        return bool(pairs[child, first_parent, second_parent])
    parents = eligibility.eligible_parents
    return bool(
        parents[child, first_parent] and parents[child, second_parent]
    )


def _eligible_parent_pair_mask(
    eligibility: _ResolvedParentEligibility,
    children: np.ndarray | int,
    first_parents: np.ndarray,
    second_parents: np.ndarray,
) -> np.ndarray:
    """Vectorized exact M2 membership without deriving a dense pair cube."""
    pairs = eligibility.eligible_parent_pairs
    if pairs is not None:
        return pairs[children, first_parents, second_parents]
    parents = eligibility.eligible_parents
    return (
        (first_parents != second_parents)
        & parents[children, first_parents]
        & parents[children, second_parents]
    )


def _eligible_parent_pair_counts(
    eligibility: _ResolvedParentEligibility,
) -> np.ndarray:
    """Count each child's unordered M2 universe without implicit expansion."""
    pairs = eligibility.eligible_parent_pairs
    if pairs is None:
        parent_counts = np.count_nonzero(
            eligibility.eligible_parents, axis=1
        ).astype(np.int64)
        return parent_counts * (parent_counts - 1) // 2

    n_samples = len(eligibility.sample_ids)
    counts = np.empty(n_samples, dtype=np.int64)
    for child in range(n_samples):
        counts[child] = np.count_nonzero(np.triu(pairs[child], k=1))
    return counts


def _parent_eligibility_result_record(
    eligibility: _ResolvedParentEligibility,
) -> dict[str, Any]:
    """Serialize resolved eligibility without expanding an implicit policy."""
    pairs = eligibility.eligible_parent_pairs
    record = {
        "format_version": eligibility.format_version,
        "policy_name": eligibility.policy_name,
        "sample_ids": eligibility.sample_ids,
        "eligible_children": eligibility.eligible_children.copy(),
        "eligible_parents": eligibility.eligible_parents.copy(),
        "eligible_parent_pairs": None if pairs is None else pairs.copy(),
        "direction_supported_parents": (
            eligibility.direction_supported_parents.copy()
        ),
        "pair_policy": eligibility.pair_policy,
        "source_fields": eligibility.source_fields,
        "assumptions": eligibility.assumptions,
        "individual_parentage_ground_truth": (
            eligibility.individual_parentage_ground_truth
        ),
    }

    if pairs is None:
        record["eligible_parent_pair_counts"] = (
            _eligible_parent_pair_counts(eligibility)
        )

    return record

import haplotype_reconstruction.pedigree.models as pedigree_models
