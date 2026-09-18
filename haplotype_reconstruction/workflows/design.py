"""Real-cross eligibility and chronology without assumed individual parentage."""
from __future__ import annotations


import hashlib
import json
from collections.abc import Sequence
from typing import Any
import numpy as np
import pandas as pd


PEDIGREE_BACKEND = "component_ragged_parent_state_v1"


TROPHEOPS_CANDIDATE_SOURCE_MODE = "t09_ragged_projected_quadratic_v1"


PARENT_ELIGIBILITY_FORMAT_VERSION = 1


TROPHEOPS_F2_F1_POLICY = "tropheops_f2_from_f1_opposite_sex_v2"


TROPHEOPS_F2_F1_NO_SEX_POLICY = "tropheops_f2_from_f1_any_pair_v2"


ASAC_GENERATION_STEP_POLICY = (
    "asac_f2_from_f1_f3_from_f2_opposite_sex_v2"
)


ASAC_GENERATION_STEP_NO_SEX_POLICY = (
    "asac_f2_from_f1_f3_from_f2_any_pair_v2"
)


_IMPLICIT_PARENT_PAIR_RULE = "all_unordered_pairs_of_eligible_parents_v1"


def _eligible_parent_pair_template(
    eligible_parent_samples: np.ndarray,
    sex: np.ndarray,
    *,
    require_opposite_sex_pair: bool,
) -> np.ndarray:
    """Return the symmetric two-parent mask for one candidate cohort."""
    parent_samples = np.asarray(eligible_parent_samples, dtype=np.bool_)
    pairs = np.logical_and(parent_samples[:, None], parent_samples[None,:])
    if require_opposite_sex_pair:
        female = sex == "F"
        male = sex == "M"
        pairs &= (
            (female[:, None] & male[None,:])
            | (male[:, None] & female[None,:])
        )
    np.fill_diagonal(pairs, False)
    return np.ascontiguousarray(pairs, dtype=np.bool_)


def _upper_triangle_counts(values: np.ndarray) -> np.ndarray:
    """Count true upper-triangle entries per leading slice without a cube copy."""
    counts = np.zeros(values.shape[0], dtype=np.intp)
    for first in range(values.shape[1] - 1):
        counts += np.count_nonzero(values[:, first, first + 1:], axis=1)
    return counts


def _explicit_direction_supported_parents(
    record: dict[str, Any],
) -> np.ndarray | None:
    """Validate and return the optional explicit design-direction mask."""
    if record.get("direction_supported_parents") is None:
        return None

    n_samples = len(tuple(record["sample_ids"]))
    children = np.asarray(record["eligible_children"])
    parents = np.asarray(record["eligible_parents"])
    direction = np.asarray(record["direction_supported_parents"])
    if children.shape != (n_samples,) or children.dtype != np.bool_:
        raise ValueError(
            "eligible_children must be a boolean array with shape (samples,)"
        )
    if parents.shape != (n_samples, n_samples) or parents.dtype != np.bool_:
        raise ValueError(
            "eligible_parents must be a boolean array with shape "
            "(samples, samples)"
        )
    if (
        direction.shape != (n_samples, n_samples)
        or direction.dtype != np.bool_
    ):
        raise ValueError(
            "direction_supported_parents must be a boolean array with shape "
            "(samples, samples)"
        )
    diagonal = np.arange(n_samples)
    if np.any(direction[diagonal, diagonal]):
        raise ValueError(
            "direction_supported_parents cannot admit self-parenting"
        )
    if np.any(direction[~children]):
        raise ValueError(
            "excluded children cannot have explicitly direction-supported "
            "parents"
        )
    if np.any(direction & ~parents):
        raise ValueError(
            "direction_supported_parents must be a subset of eligible_parents"
        )
    return np.ascontiguousarray(direction)


def build_current_pedigree_config(*, bootstrap_replicates: int=1000):
    """Return the promoted ragged quadratic / strict-direction configuration."""


    return module_pedigree_config.PedigreeConfig(
        bootstrap_replicates=bootstrap_replicates,
        primary_view="tier_b",
        parent_state_effective_markers_per_information_block=3.0,
        parent_state_minimum_edge_coverage=0.95,
        parent_state_minimum_pair_explainability=0.95,
        parent_state_minimum_edge_exposed_bins=1.0,
        parent_state_minimum_pair_exposed_bins=1.0,
        parent_state_minimum_exposed_fraction=0.10,
        parent_state_minimum_exposed_contigs=3,
        parent_state_minimum_direction_probability=0.01,
    ).validated()


def build_tropheops_parent_eligibility(
    metadata: pd.DataFrame,
    sample_ids: Sequence[Any],
    *,
    require_opposite_sex_pair: bool=True,
) -> dict[str, Any]:
    """Build the explicit exploratory F2 <- F1 candidate universe.

    ``generation`` supplies explicit F1-to-F2 design chronology as well as the
    candidate-eligibility constraint, but never establishes individual
    parentage. ``sex`` constrains an M2 pair when requested; it does not order
    Parent1/Parent2. Every VCF sample must occur exactly once in ``metadata``
    so the record is reproducibly bound to sample order.
    """
    required = {"primary_ID", "generation", "sex"}
    missing_columns = sorted(required.difference(metadata.columns))
    if missing_columns:
        raise ValueError(
            "Tropheops parent eligibility requires metadata columns: "
            + ", ".join(missing_columns)
        )

    ordered_ids = tuple(str(value) for value in sample_ids)
    if len(ordered_ids) < 3 or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("sample_ids must contain at least three unique IDs")

    rows = metadata.loc[:, ["primary_ID", "generation", "sex"]].copy()
    rows["primary_ID"] = rows["primary_ID"].astype(str)
    relevant = rows[rows["primary_ID"].isin(ordered_ids)]
    duplicated = relevant["primary_ID"].duplicated(keep=False)
    if bool(duplicated.any()):
        raise ValueError("metadata contains duplicate rows for a VCF sample")
    by_id = relevant.set_index("primary_ID")
    absent = [sample for sample in ordered_ids if sample not in by_id.index]
    if absent:
        raise ValueError(
            f"metadata is missing {len(absent)} ordered VCF sample(s)"
        )
    aligned = by_id.loc[list(ordered_ids)]

    generation = aligned["generation"].fillna("").astype(str).to_numpy()
    sex = aligned["sex"].fillna("").astype(str).str.upper().to_numpy()
    eligible_children = generation == "F2"
    eligible_parent_samples = generation == "F1"
    eligible_parents = (
        eligible_children[:, None] & eligible_parent_samples[None,:]
    )
    np.fill_diagonal(eligible_parents, False)

    n_samples = len(ordered_ids)
    pair_template = _eligible_parent_pair_template(
        eligible_parent_samples,
        sex,
        require_opposite_sex_pair=require_opposite_sex_pair,
    )
    eligible_pairs = np.empty(
        (n_samples, n_samples, n_samples), dtype=np.bool_
    )
    np.logical_and(
        eligible_children[:, None, None],
        pair_template[None,:,:],
        out=eligible_pairs,
    )

    policy_name = (
        TROPHEOPS_F2_F1_POLICY
        if require_opposite_sex_pair
        else TROPHEOPS_F2_F1_NO_SEX_POLICY
    )
    assumptions = [
        "Only metadata-labelled F2 samples are inference targets.",
        "Only metadata-labelled F1 samples are legitimate observed-parent candidates.",
        "Generation labels constrain eligibility but do not establish individual parentage.",
        "The explicit F1-to-F2 generation step supports candidate direction, not individual parentage truth.",
        "G0 and F2 samples are excluded from the observed-parent candidate pool.",
    ]
    if require_opposite_sex_pair:
        assumptions.append(
            "Two-observed-parent configurations require one recorded female "
            "and one recorded male F1; parent order remains arbitrary."
        )

    return {
        "format_version": PARENT_ELIGIBILITY_FORMAT_VERSION,
        "policy_name": policy_name,
        "sample_ids": ordered_ids,
        "eligible_children": np.ascontiguousarray(
            eligible_children, dtype=np.bool_
        ),
        "eligible_parents": np.ascontiguousarray(
            eligible_parents, dtype=np.bool_
        ),
        "direction_supported_parents": np.ascontiguousarray(
            eligible_parents, dtype=np.bool_
        ).copy(),
        "eligible_parent_pairs": np.ascontiguousarray(
            eligible_pairs, dtype=np.bool_
        ),
        "source_fields": ("primary_ID", "generation", "sex"),
        "assumptions": tuple(assumptions),
        "individual_parentage_ground_truth": False,
    }


def build_asac_parent_eligibility(
    metadata: pd.DataFrame,
    sample_ids: Sequence[Any],
    *,
    require_opposite_sex_pair: bool=True,
) -> dict[str, Any]:
    """Build the exploratory AsAc F2<-F1 and F3<-F2 universe.

    Generation supplies explicit F1-to-F2 and F2-to-F3 design chronology as
    well as candidate cohorts. It does not assert any individual parentage.
    Recorded sex constrains M2 feasibility when requested, but it never orders
    the parent fields or removes M1 candidates. Species-labelled G0 samples
    are explicitly excluded.
    """
    required = {"primary_ID", "generation", "sex"}
    missing_columns = sorted(required.difference(metadata.columns))
    if missing_columns:
        raise ValueError(
            "AsAc parent eligibility requires metadata columns: "
            + ", ".join(missing_columns)
        )

    ordered_ids = tuple(str(value) for value in sample_ids)
    if len(ordered_ids) < 3 or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("sample_ids must contain at least three unique IDs")

    rows = metadata.loc[:, ["primary_ID", "generation", "sex"]].copy()
    rows = rows[rows["primary_ID"].notna()]
    rows["primary_ID"] = rows["primary_ID"].astype(str)
    relevant = rows[rows["primary_ID"].isin(ordered_ids)]
    duplicated = relevant["primary_ID"].duplicated(keep=False)
    if bool(duplicated.any()):
        raise ValueError("metadata contains duplicate rows for a VCF sample")
    by_id = relevant.set_index("primary_ID")
    absent = [sample for sample in ordered_ids if sample not in by_id.index]
    if absent:
        raise ValueError(
            f"metadata is missing {len(absent)} ordered VCF sample(s)"
        )
    aligned = by_id.loc[list(ordered_ids)]

    raw_generation = aligned["generation"].fillna("").astype(str)
    generation = raw_generation.where(
        ~raw_generation.str.startswith("G0"), "G0"
    ).to_numpy()
    allowed_generations = {"G0", "F1", "F2", "F3"}
    invalid_generations = sorted(set(generation).difference(allowed_generations))
    if invalid_generations:
        raise ValueError(
            "AsAc metadata contains unsupported generation labels: "
            + ", ".join(repr(value) for value in invalid_generations)
        )
    sex = aligned["sex"].fillna("").astype(str).str.upper().to_numpy()
    invalid_sexes = sorted(set(sex).difference({"F", "M"}))
    if invalid_sexes:
        raise ValueError(
            "AsAc metadata contains unsupported sex labels: "
            + ", ".join(repr(value) for value in invalid_sexes)
        )

    f2_children = generation == "F2"
    f3_children = generation == "F3"
    eligible_children = f2_children | f3_children
    f1_parent_samples = generation == "F1"
    f2_parent_samples = generation == "F2"
    n_samples = len(ordered_ids)
    eligible_parents = np.zeros((n_samples, n_samples), dtype=np.bool_)
    eligible_parents[f2_children,:] = f1_parent_samples
    eligible_parents[f3_children,:] = f2_parent_samples
    np.fill_diagonal(eligible_parents, False)

    eligible_pairs = np.zeros(
        (n_samples, n_samples, n_samples), dtype=np.bool_
    )
    f1_pair_template = _eligible_parent_pair_template(
        f1_parent_samples,
        sex,
        require_opposite_sex_pair=require_opposite_sex_pair,
    )
    f2_pair_template = _eligible_parent_pair_template(
        f2_parent_samples,
        sex,
        require_opposite_sex_pair=require_opposite_sex_pair,
    )
    eligible_pairs[f2_children,:,:] = f1_pair_template
    eligible_pairs[f3_children,:,:] = f2_pair_template

    policy_name = (
        ASAC_GENERATION_STEP_POLICY
        if require_opposite_sex_pair
        else ASAC_GENERATION_STEP_NO_SEX_POLICY
    )
    assumptions = [
        "Only metadata-labelled F2 and F3 samples are inference targets.",
        "F2 admits F1 and F3 admits F2 observed-parent candidates.",
        "Generation labels constrain eligibility but do not establish individual parentage.",
        "The explicit F1-to-F2 and F2-to-F3 generation steps support candidate direction, not individual parentage truth.",
        "Species-labelled G0 samples are neither targets nor candidates.",
    ]
    if require_opposite_sex_pair:
        assumptions.append(
            "M2 requires one recorded female and one recorded male within "
            "the eligible parent cohort; parent order remains arbitrary."
        )

    return {
        "format_version": PARENT_ELIGIBILITY_FORMAT_VERSION,
        "policy_name": policy_name,
        "sample_ids": ordered_ids,
        "eligible_children": np.ascontiguousarray(
            eligible_children, dtype=np.bool_
        ),
        "eligible_parents": np.ascontiguousarray(
            eligible_parents, dtype=np.bool_
        ),
        "direction_supported_parents": np.ascontiguousarray(
            eligible_parents, dtype=np.bool_
        ).copy(),
        "eligible_parent_pairs": np.ascontiguousarray(
            eligible_pairs, dtype=np.bool_
        ),
        "source_fields": ("primary_ID", "generation", "sex"),
        "assumptions": tuple(assumptions),
        "individual_parentage_ground_truth": False,
    }


def parent_eligibility_identity(record: dict[str, Any]) -> str:
    """Return a deterministic compatibility identity for one policy record."""
    scalar = {
        "format_version": int(record["format_version"]),
        "policy_name": str(record["policy_name"]),
        "sample_ids": [str(value) for value in record["sample_ids"]],
        "source_fields": list(record.get("source_fields", ())),
        "assumptions": list(record.get("assumptions", ())),
        "individual_parentage_ground_truth": bool(
            record.get("individual_parentage_ground_truth", False)
        ),
    }
    digest = hashlib.sha256(json.dumps(
        scalar,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8"))
    for field in ("eligible_children", "eligible_parents"):
        values = np.ascontiguousarray(record[field], dtype=np.bool_)
        digest.update(field.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(values).cast("B"))
    pairs = record["eligible_parent_pairs"]
    if pairs is None:
        # ``None`` is the compact policy: every unordered pair admitted by
        # the corresponding eligible-parent row. Hash the rule rather than
        # expanding its child x parent x parent truth table.
        digest.update(b"eligible_parent_pairs")
        digest.update(_IMPLICIT_PARENT_PAIR_RULE.encode("ascii"))
    else:
        # Keep the established dense-record byte stream exactly unchanged so
        # existing checkpoint identities remain compatible.
        values = np.ascontiguousarray(pairs, dtype=np.bool_)
        digest.update(b"eligible_parent_pairs")
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(values).cast("B"))
    direction_supported = _explicit_direction_supported_parents(record)
    if direction_supported is not None:
        digest.update(b"direction_supported_parents")
        digest.update(
            np.asarray(direction_supported.shape, dtype=np.int64).tobytes()
        )
        digest.update(memoryview(direction_supported).cast("B"))
    return digest.hexdigest()


def summarize_parent_eligibility(record: dict[str, Any]) -> dict[str, Any]:
    """Return compact, JSON-safe counts for manifests and logs."""
    children = np.asarray(record["eligible_children"], dtype=np.bool_)
    parents = np.asarray(record["eligible_parents"], dtype=np.bool_)
    parent_counts = np.count_nonzero(parents, axis=1)
    pairs = record["eligible_parent_pairs"]
    if pairs is None:
        # Each row represents all unordered pairs of its eligible parents.
        # The exact count is analytic and needs no O(N^3) pair cube.
        pair_counts = parent_counts * (parent_counts - 1) // 2
    else:
        pair_counts = _upper_triangle_counts(
            np.asarray(pairs, dtype=np.bool_)
        )
    target_indices = np.flatnonzero(children)
    direction_supported = _explicit_direction_supported_parents(record)
    return {
        "format_version": int(record["format_version"]),
        "policy_name": str(record["policy_name"]),
        "sample_count": int(len(children)),
        "eligible_child_count": int(np.count_nonzero(children)),
        "candidate_parent_sample_count": int(
            np.count_nonzero(np.any(parents, axis=0))
        ),
        "direction_supported_parent_edge_count": int(
            0 if direction_supported is None
            else np.count_nonzero(direction_supported)
        ),
        "minimum_parent_candidates_per_target": int(
            np.min(parent_counts[target_indices]) if len(target_indices) else 0
        ),
        "maximum_parent_candidates_per_target": int(
            np.max(parent_counts[target_indices]) if len(target_indices) else 0
        ),
        "minimum_parent_pairs_per_target": int(
            np.min(pair_counts[target_indices]) if len(target_indices) else 0
        ),
        "maximum_parent_pairs_per_target": int(
            np.max(pair_counts[target_indices]) if len(target_indices) else 0
        ),
        "individual_parentage_ground_truth": bool(
            record.get("individual_parentage_ground_truth", False)
        ),
        "assumptions": list(record.get("assumptions", ())),
    }


def pedigree_stage_names(backend: str=PEDIGREE_BACKEND) -> tuple[str, str]:
    """Return evidence preparation and genome-wide inference stages."""
    if backend != PEDIGREE_BACKEND:
        raise ValueError(f"unsupported pedigree backend identity {backend!r}")
    return (
        "10_pedigree_evidence",
        "10_pedigree",
    )

import haplotype_reconstruction.pedigree.config as module_pedigree_config
