"""Default metadata-free, parent-state pedigree inference.

Standard tolerance-painting/founder-block inputs and explicit parent-state
evidence are scored under one Smart model for zero, one, and two observed
parents. Unknown or pair-only schemas are rejected; this module never silently
switches to a different pedigree algorithm.

The model separates parent-count state, conditional candidate identity,
chromosome-resampling stability, and joint acyclic graph selection. Direct
results retain the configured scientific view in ``relationships``. The
pipeline entry point adds a separate ``pipeline_control_relationships`` table
for downstream code that cannot represent missing-parent states. A row without
reported parents can be unresolved or have no observed parents; it does not by
itself establish biological founder status.

With no breeding records, generation labels, or sex information, parent roles
are unordered and direction can be weakly identifiable. Bootstrap and LOCO
fractions describe internal stability rather than calibrated biological
posterior probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import operator
import os
import warnings
from typing import Any, Mapping, Optional, Sequence

import thread_config  # must precede NumPy/Numba imports
import dynamic_threads

import numba
from numba import prange

thread_config.ensure_numba_registry_warmup()
# Smart kernels explicitly declare their cache policy.  thread_config performs
# the one process-wide Numba registry warm-up and exposes the unwrapped
# decorator, so importing this module no longer mutates numba.njit temporarily.
njit = thread_config.original_njit

import numpy as np
import pandas as pd
from bhd_config import DEFAULT_READ_ERROR_PROBABILITY
from pedigree_bootstrap_kernels import (
    accumulate_bootstrap_counts_into,
    count_exposed_contigs,
    is_acyclic_parent_rows,
    pack_contig_presence,
)
from bhd_genotype_evidence import allele_depths_to_raw_genotype_likelihoods
from pedigree_candidate_source_posterior import (
    infer_candidate_source_posterior,
    score_candidate_source_batch_exact,
    score_candidate_source_batch_matched_null_exact,
)
from pedigree_depth_gmm import fit_bic_selected_gaussian_mixture_1d
from multiprocessing_runtime import (
    forkserver_context,
    safe_forkserver_pool,
)
import pedigree_hard_painting as _hard_painting
from pedigree_hmm import poisson_switch_stay_terms
from pedigree_result import PedigreeResult
from pedigree_visualization import draw_pedigree_tree
from shared_array import (
    attach_shared_array,
    create_shared_array,
    shared_memory_cleanup,
)

DEFAULT_MISMATCH_PENALTY = _hard_painting.DEFAULT_MISMATCH_PENALTY


class SmartEvidenceError(ValueError):
    """Raised when explicit smart evidence is absent or internally invalid."""


_PARENT_STATE_NAMES = (
    "zero_observed_parents",
    "one_observed_parent",
    "two_observed_parents",
)
_ZERO_OBSERVED = 0
_ONE_OBSERVED = 1
_TWO_OBSERVED = 2
_EXTERNAL_PARENT = -1
SMART_PARENT_ELIGIBILITY_FORMAT_VERSION = 1
PARENT_ELIGIBILITY_FORMAT_VERSION = SMART_PARENT_ELIGIBILITY_FORMAT_VERSION
_PARENT_STATE_METHOD = "combined_v1"
_PARENT_STATE_LIKELIHOOD = "b1"


@dataclass(frozen=True)
class SmartParentEligibility:
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


@dataclass(frozen=True)
class _ResolvedParentEligibility:
    supplied: bool
    format_version: int
    sample_ids: tuple[Any, ...]
    eligible_children: np.ndarray
    eligible_parents: np.ndarray
    eligible_parent_pairs: Optional[np.ndarray]
    policy_name: str
    source_fields: tuple[str, ...]
    assumptions: tuple[str, ...]
    individual_parentage_ground_truth: bool
    pair_policy: str


def _parent_eligibility_text_tuple(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise SmartEvidenceError(f"parent eligibility {field} must be a sequence")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise SmartEvidenceError(
            f"parent eligibility {field} must be a sequence"
        ) from exc
    if any(not isinstance(item, str) for item in values):
        raise SmartEvidenceError(
            f"parent eligibility {field} entries must be strings"
        )
    return values


def _resolve_parent_eligibility(
    value: Optional[SmartParentEligibility | Mapping[str, Any]],
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
            False,
            SMART_PARENT_ELIGIBILITY_FORMAT_VERSION,
            samples,
            children,
            parents,
            None,
            "all_samples_eligible_default_v1",
            (),
            (),
            False,
            "all_unordered_pairs_of_eligible_parents",
        )

    if isinstance(value, _ResolvedParentEligibility):
        if value.sample_ids != samples:
            raise SmartEvidenceError(
                "parent eligibility sample_ids must exactly match inference sample order"
            )
        return value
    if isinstance(value, SmartParentEligibility):
        record = value
    elif isinstance(value, Mapping):
        required = (
            "format_version", "sample_ids", "eligible_children",
            "eligible_parents",
        )
        missing = [field for field in required if field not in value]
        if missing:
            raise SmartEvidenceError(
                f"parent eligibility is missing required field {missing[0]!r}"
            )
        record = SmartParentEligibility(
            format_version=value["format_version"],
            sample_ids=value["sample_ids"],
            eligible_children=value["eligible_children"],
            eligible_parents=value["eligible_parents"],
            eligible_parent_pairs=value.get("eligible_parent_pairs"),
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
        raise SmartEvidenceError(
            "parent_eligibility must be SmartParentEligibility or a mapping"
        )
    if (
        isinstance(record.format_version, (bool, np.bool_))
        or record.format_version != SMART_PARENT_ELIGIBILITY_FORMAT_VERSION
    ):
        raise SmartEvidenceError("unsupported parent eligibility format_version")
    try:
        record_samples = tuple(record.sample_ids)
    except TypeError as exc:
        raise SmartEvidenceError(
            "parent eligibility sample_ids must be an ordered sequence"
        ) from exc
    if len(record_samples) != n_samples or any(
        observed != expected
        for observed, expected in zip(record_samples, samples)
    ):
        raise SmartEvidenceError(
            "parent eligibility sample_ids must exactly match inference sample order"
        )

    children_raw = np.asarray(record.eligible_children)
    parents_raw = np.asarray(record.eligible_parents)
    if children_raw.shape != (n_samples,) or children_raw.dtype != np.bool_:
        raise SmartEvidenceError(
            "eligible_children must be a boolean array with shape (samples,)"
        )
    if (
        parents_raw.shape != (n_samples, n_samples)
        or parents_raw.dtype != np.bool_
    ):
        raise SmartEvidenceError(
            "eligible_parents must be a boolean array with shape "
            "(samples, samples)"
        )
    children = np.ascontiguousarray(children_raw)
    parents = np.ascontiguousarray(parents_raw)
    if np.any(parents[diagonal, diagonal]):
        raise SmartEvidenceError("eligible_parents cannot admit self-parenting")
    if np.any(parents[~children]):
        raise SmartEvidenceError(
            "excluded children cannot have eligible parent identities"
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
            raise SmartEvidenceError(
                "eligible_parent_pairs must be a boolean array with shape "
                "(samples, samples, samples)"
            )
        pairs = np.ascontiguousarray(pair_raw)
        if not np.array_equal(pairs, np.swapaxes(pairs, 1, 2)):
            raise SmartEvidenceError(
                "eligible_parent_pairs must be symmetric in the parent axes"
            )
        if np.any(pairs[:, diagonal, diagonal]):
            raise SmartEvidenceError(
                "eligible_parent_pairs cannot contain duplicate parents"
            )
        for child in range(n_samples):
            child_allowed = (
                parents[child, :, None] & parents[child, None, :]
            )
            if np.any(pairs[child] & ~child_allowed):
                raise SmartEvidenceError(
                    "eligible_parent_pairs may contain only eligible parent identities"
                )
        pair_policy = "explicit_symmetric_pair_mask"

    if not isinstance(record.policy_name, str) or not record.policy_name:
        raise SmartEvidenceError(
            "parent eligibility policy_name must be a non-empty string"
        )
    source_fields = _parent_eligibility_text_tuple(
        record.source_fields, "source_fields"
    )
    assumptions = _parent_eligibility_text_tuple(
        record.assumptions, "assumptions"
    )
    if not isinstance(record.individual_parentage_ground_truth, (bool, np.bool_)):
        raise SmartEvidenceError(
            "individual_parentage_ground_truth must be boolean"
        )
    return _ResolvedParentEligibility(
        True, SMART_PARENT_ELIGIBILITY_FORMAT_VERSION, samples,
        children.copy(), parents.copy(), (
            None if pairs is None else np.ascontiguousarray(pairs).copy()
        ),
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


@dataclass(frozen=True)
class SmartPedigreeConfig:
    """Numerical and reporting policy for metadata-free inference.

    Thresholds label evidence tiers; they are not biological acceptance
    probabilities. Defaults deliberately require evidence from at least
    three independently aggregated contigs.
    """

    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 20260725
    markers_per_information_block: int = 100
    information_tempering_power: float = 0.5
    maximum_contig_weight_ratio: float = 4.0
    rank_weight: float = 0.35
    chromosome_contamination: float = 0.02
    linked_evidence_weight: float = 0.65
    genotype_evidence_weight: float = 0.35
    minimum_informative_contigs: int = 3
    tier_a_pair_bootstrap: float = 0.95
    tier_a_parent_bootstrap: float = 0.95
    tier_a_loco_fraction: float = 0.95
    tier_b_pair_bootstrap: float = 0.70
    tier_b_parent_bootstrap: float = 0.70
    tier_b_loco_fraction: float = 0.80
    support_set_coverage: float = 0.95
    primary_view: str = "tier_b"
    # Persisted provenance identities, deliberately excluded from __init__ so
    # callers cannot select the retired unscreened or prototype engines.
    parent_state_algorithm_mode: str = field(
        default=_PARENT_STATE_LIKELIHOOD, init=False
    )
    parent_state_structure_mode: str = field(
        default=_PARENT_STATE_METHOD, init=False
    )
    parent_state_minimum_edge_coverage: float = 0.95
    parent_state_minimum_pair_explainability: float = 0.95
    parent_state_minimum_edge_exposed_bins: float = 1.0
    parent_state_minimum_pair_exposed_bins: float = 1.0
    parent_state_minimum_exposed_fraction: float = 0.10
    parent_state_minimum_exposed_contigs: int = 3
    parent_state_minimum_direction_probability: float = 0.01
    parent_state_candidate_source_mode: str = "hard_painted"
    parent_state_candidate_source_path_switch_probability: float | None = None
    dag_local_search_passes: int = 3
    parent_state_mismatch_probability: float = 0.01
    parent_state_phase_switch_probability: float = 0.01
    parent_state_contamination_probability: float = 0.02
    parent_state_effective_markers_per_information_block: float = 3.0
    parent_state_external_state_pseudocount: float = 1.0
    parent_state_external_transition_pseudocount: float = 20.0
    parent_state_priors: tuple[float, float, float] = (
        1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    )
    parent_state_prior_sensitivity: tuple[
        tuple[float, float, float], ...
    ] = (
        (0.50, 0.30, 0.20),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (0.20, 0.30, 0.50),
    )
    parent_state_prior_strength: float = 3.0
    parent_state_prior_max_iterations: int = 100
    parent_state_prior_tolerance: float = 1e-10

    def validated(self) -> "SmartPedigreeConfig":
        def require_integer(name: str, minimum: int) -> None:
            raw_value = getattr(self, name)
            if isinstance(raw_value, (bool, np.bool_)):
                raise SmartEvidenceError(
                    f"{name} must be an integer of at least {minimum}"
                )
            try:
                value = operator.index(raw_value)
            except TypeError as exc:
                raise SmartEvidenceError(
                    f"{name} must be an integer of at least {minimum}"
                ) from exc
            if value < minimum:
                raise SmartEvidenceError(
                    f"{name} must be an integer of at least {minimum}"
                )

        for integer_name, minimum in (
            ("bootstrap_replicates", 1),
            ("bootstrap_seed", 0),
            ("markers_per_information_block", 1),
            ("minimum_informative_contigs", 1),
            ("parent_state_minimum_exposed_contigs", 1),
            ("dag_local_search_passes", 0),
            ("parent_state_prior_max_iterations", 1),
        ):
            require_integer(integer_name, minimum)
        if (
            not np.isfinite(self.maximum_contig_weight_ratio)
            or self.maximum_contig_weight_ratio < 1.0
        ):
            raise SmartEvidenceError(
                "maximum_contig_weight_ratio must be finite and at least one"
            )
        for name in (
            "information_tempering_power",
            "rank_weight",
            "chromosome_contamination",
            "linked_evidence_weight",
            "genotype_evidence_weight",
            "tier_a_pair_bootstrap",
            "tier_a_parent_bootstrap",
            "tier_a_loco_fraction",
            "tier_b_pair_bootstrap",
            "tier_b_parent_bootstrap",
            "tier_b_loco_fraction",
            "support_set_coverage",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise SmartEvidenceError(f"{name} must be finite")
        if not 0.0 <= self.rank_weight <= 1.0:
            raise SmartEvidenceError("rank_weight must lie in [0, 1]")
        if not 0.0 <= self.chromosome_contamination < 1.0:
            raise SmartEvidenceError(
                "chromosome_contamination must lie in [0, 1)"
            )
        for name in (
            "tier_a_pair_bootstrap",
            "tier_a_parent_bootstrap",
            "tier_a_loco_fraction",
            "tier_b_pair_bootstrap",
            "tier_b_parent_bootstrap",
            "tier_b_loco_fraction",
            "support_set_coverage",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise SmartEvidenceError(f"{name} must lie in [0, 1]")
        if self.linked_evidence_weight < 0.0 or self.genotype_evidence_weight < 0.0:
            raise SmartEvidenceError("evidence weights must be non-negative")
        if self.linked_evidence_weight + self.genotype_evidence_weight <= 0.0:
            raise SmartEvidenceError("at least one evidence weight must be positive")
        if self.primary_view not in {"tier_a", "tier_b", "complete"}:
            raise SmartEvidenceError(
                "primary_view must be 'tier_a', 'tier_b', or 'complete'"
            )
        # Old pickles can carry values for init=False fields. Reject them
        # explicitly rather than silently treating a retired configuration as
        # the combined method.
        if self.parent_state_algorithm_mode != _PARENT_STATE_LIKELIHOOD:
            raise SmartEvidenceError(
                "only the combined_v1 pedigree method with internal B1 "
                "likelihood evidence is supported"
            )
        if self.parent_state_structure_mode != _PARENT_STATE_METHOD:
            raise SmartEvidenceError(
                "the retired unscreened pedigree method is not supported"
            )
        for name in (
            "parent_state_minimum_edge_coverage",
            "parent_state_minimum_pair_explainability",
            "parent_state_minimum_direction_probability",
            "parent_state_minimum_exposed_fraction",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SmartEvidenceError(f"{name} must lie in [0, 1]")
        for name in (
            "parent_state_minimum_edge_exposed_bins",
            "parent_state_minimum_pair_exposed_bins",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise SmartEvidenceError(
                    f"{name} must be finite and non-negative"
                )
        if self.parent_state_candidate_source_mode not in {
            "hard_painted", "exact_raw_gl_v1", "matched_null_raw_gl_v2"
        }:
            raise SmartEvidenceError(
                "parent_state_candidate_source_mode must be 'hard_painted', "
                "'exact_raw_gl_v1', or 'matched_null_raw_gl_v2'"
            )
        source_path_switch = (
            self.parent_state_candidate_source_path_switch_probability
        )
        if self.parent_state_candidate_source_mode == "matched_null_raw_gl_v2":
            try:
                source_path_switch_value = float(source_path_switch)
            except (TypeError, ValueError, OverflowError):
                source_path_switch_value = np.nan
            if (
                isinstance(source_path_switch, (bool, np.bool_))
                or not np.isfinite(source_path_switch_value)
                or not 0.0 <= source_path_switch_value <= 0.5
            ):
                raise SmartEvidenceError(
                    "matched_null_raw_gl_v2 requires an explicit finite "
                    "parent_state_candidate_source_path_switch_probability "
                    "in [0, 0.5]"
                )
        if (
            not np.isfinite(self.parent_state_mismatch_probability)
            or not 0.0 < self.parent_state_mismatch_probability < 0.5
        ):
            raise SmartEvidenceError(
                "parent_state_mismatch_probability must lie in (0, 0.5)"
            )
        if (
            not np.isfinite(self.parent_state_phase_switch_probability)
            or not 0.0 <= self.parent_state_phase_switch_probability <= 0.5
        ):
            raise SmartEvidenceError(
                "parent_state_phase_switch_probability must lie in [0, 0.5]"
            )
        if (
            not np.isfinite(self.parent_state_contamination_probability)
            or not 0.0 <= self.parent_state_contamination_probability < 1.0
        ):
            raise SmartEvidenceError(
                "parent_state_contamination_probability must lie in [0, 1)"
            )
        for name in (
            "parent_state_effective_markers_per_information_block",
            "parent_state_external_state_pseudocount",
            "parent_state_external_transition_pseudocount",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise SmartEvidenceError(f"{name} must be finite and positive")
        priors = np.asarray(self.parent_state_priors, dtype=np.float64)
        if (
            priors.shape != (3,)
            or np.any(~np.isfinite(priors))
            or np.any(priors <= 0.0)
            or not np.isclose(np.sum(priors), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise SmartEvidenceError(
                "parent_state_priors must be three positive probabilities "
                "summing to one"
            )
        if (
            not np.isfinite(self.parent_state_prior_strength)
            or self.parent_state_prior_strength <= 0.0
        ):
            raise SmartEvidenceError(
                "parent_state_prior_strength must be finite and positive"
            )
        if (
            not np.isfinite(self.parent_state_prior_tolerance)
            or self.parent_state_prior_tolerance <= 0.0
        ):
            raise SmartEvidenceError(
                "parent_state_prior_tolerance must be finite and positive"
            )
        sensitivity = np.asarray(
            self.parent_state_prior_sensitivity, dtype=np.float64
        )
        if (
            sensitivity.ndim != 2
            or sensitivity.shape[1] != 3
            or len(sensitivity) == 0
            or np.any(~np.isfinite(sensitivity))
            or np.any(sensitivity <= 0.0)
            or np.any(~np.isclose(
                np.sum(sensitivity, axis=1), 1.0, rtol=0.0, atol=1e-12
            ))
        ):
            raise SmartEvidenceError(
                "parent_state_prior_sensitivity must contain positive "
                "probability triples summing to one"
            )
        return self


@dataclass(frozen=True)
class SmartContigEvidence:
    """Explicit chromosome evidence for a fixed candidate-trio panel.

    ``trios`` is an integer ``(n_rows, 3)`` array with columns
    ``child, parent1, parent2``. Every contig must contain the same canonical
    trio keys, although row order may differ. Scores are relative log-like
    evidence and may be on different scales: each source is robustly
    transformed within chromosome and child before aggregation.
    """

    contig: str
    trios: np.ndarray
    linked_log_likelihoods: np.ndarray
    genotype_log_likelihoods: np.ndarray
    informative_markers: int


@dataclass(frozen=True)
class SmartParentStateEvidence:
    """Comparable forward evidence for 0/1/2 observed-parent models.

    The zero-parent vector is indexed by child, the one-parent matrix by
    ``[child, observed_parent]`` (with a ``-inf`` diagonal), and the two-parent
    vector by ``trios``. Scores must be normalized forward log likelihoods
    from the same observation model, not ranks, Viterbi maxima, or unrelated
    score scales. The two-parent panel may be screened; aggregation always
    uses the full eligible pair count as its identity-prior denominator.
    """

    contig: str
    trios: np.ndarray
    zero_parent_log_likelihoods: np.ndarray
    one_parent_log_likelihoods: np.ndarray
    two_parent_log_likelihoods: np.ndarray
    informative_markers: int
    edge_matched_bins: np.ndarray | None = None
    edge_exposed_bins: np.ndarray | None = None
    pair_explained_bins: np.ndarray | None = None
    pair_exposed_bins: np.ndarray | None = None
    structure_total_bins: float | None = None


# Canonical names for the default engine. Smart-prefixed names remain stable
# aliases for historical checkpoints and frozen analysis scripts.
PedigreeConfig = SmartPedigreeConfig
PedigreeEvidenceError = SmartEvidenceError
ParentEligibility = SmartParentEligibility
ParentStateEvidence = SmartParentStateEvidence


@dataclass(frozen=True)
class _StandardContigCache:
    """Allele-grid representation derived from one historical contig input."""

    contig: str
    stacked_alleles: np.ndarray
    stacked_hom_mask: np.ndarray
    switch_costs: np.ndarray
    stay_costs: np.ndarray
    informative_markers: int
    stacked_labels: np.ndarray
    founder_alleles: np.ndarray
    selected_markers_per_bin: np.ndarray
    switch_probabilities: np.ndarray
    genotype_likelihoods: np.ndarray | None = None
    selected_positions: np.ndarray | None = None
    state_evidence_mode: str = "hard_allele"



_TINY = np.finfo(np.float64).tiny


def _apply_smart_dynamic_threads() -> int:
    """Settle the shared remainder allocation before a parallel kernel.

    ``dynamic_threads`` intentionally reads its active counter lock-free. If
    several workers release stale remainder claims together, the first pass
    can temporarily leave a few remainder cores idle. A second public-API
    pass immediately lets non-holders reclaim those cores without changing
    the foundational allocator or the total thread budget.
    """
    dynamic_threads.apply_dynamic_threads()
    return dynamic_threads.apply_dynamic_threads()


@dataclass(frozen=True)
class _ParentStateContigScores:
    """Comparable forward likelihoods and ancestry depth for one contig."""

    zero_observed: np.ndarray
    one_observed: np.ndarray
    two_observed: np.ndarray
    ancestry_junction_counts: np.ndarray
    ancestry_callable_haplotype_bins: np.ndarray
    one_parent_identity_information: np.ndarray | None = None
    two_parent_edge_information: np.ndarray | None = None
    candidate_source_mode_requested: str = "hard_painted"
    candidate_source_mode_applied: str = "hard_painted"
    candidate_source_fallback: bool = False
    candidate_source_fallback_reason: str = ""
    complete_founder_marker_count: int | None = None
    excluded_founder_marker_count: int | None = None
    candidate_source_available: np.ndarray | None = None
    candidate_source_informative_marker_count: np.ndarray | None = None
    child_complete_informative_marker_count: np.ndarray | None = None
    candidate_initial_max_probability: np.ndarray | None = None
    candidate_initial_point_mass: np.ndarray | None = None
    peak_streamed_tensor_bytes: int = 0
    candidate_source_posterior: Any = None
    edge_matched_bins: np.ndarray | None = None
    edge_exposed_bins: np.ndarray | None = None
    pair_explained_bins: np.ndarray | None = None
    pair_exposed_bins: np.ndarray | None = None
    structure_total_bins: float | None = None


@njit(cache=True, inline="always")
def _source_alt_probability(hard_allele, background_alt):
    if hard_allele < 0:
        return background_alt
    return float(hard_allele)


@njit(cache=True, inline="always")
def _observed_allele_log_probability(observed, source_alt, mismatch_probability):
    if observed < 0:
        return 0.0
    if observed == 1:
        probability = (
            (1.0 - mismatch_probability) * source_alt
            + mismatch_probability * (1.0 - source_alt)
        )
    else:
        probability = (
            (1.0 - mismatch_probability) * (1.0 - source_alt)
            + mismatch_probability * source_alt
        )
    return math.log(max(probability, _TINY))


@njit(cache=True, inline="always")
def _phase_switch_probability(
    previous_homozygous,
    current_homozygous,
    phase_switch_probability,
):
    # Reconstructed homolog orientation is unidentified across a homozygous or
    # wholly missing block. Resetting to 1/2 is normalized and avoids treating
    # arbitrary painting phase as biological transmission evidence.
    if previous_homozygous or current_homozygous:
        return 0.5
    return phase_switch_probability


@njit(cache=True, parallel=True)
def _local_ibs_class_kernel(founders):
    """Assign first-occurrence local classes independently by cache bin."""
    n_states, n_bins, n_snps = founders.shape
    mapping = np.empty((n_bins, n_states), dtype=np.int16)
    class_counts = np.empty(n_bins, dtype=np.int16)
    pooled_founders = np.full(
        (n_states, n_bins, n_snps), -1, dtype=np.int8
    )
    active = np.zeros((n_bins, n_states), dtype=np.bool_)
    for block in prange(n_bins):
        number_of_classes = 0
        for state in range(n_states):
            local_class = -1
            for previous_state in range(state):
                equivalent = True
                for snp in range(n_snps):
                    if (
                        founders[state, block, snp]
                        != founders[previous_state, block, snp]
                    ):
                        equivalent = False
                        break
                if equivalent:
                    local_class = int(mapping[block, previous_state])
                    break
            if local_class < 0:
                local_class = number_of_classes
                number_of_classes += 1
                active[block, local_class] = True
                for snp in range(n_snps):
                    pooled_founders[local_class, block, snp] = (
                        founders[state, block, snp]
                    )
            mapping[block, state] = local_class
        class_counts[block] = number_of_classes
    return mapping, class_counts, pooled_founders, active


@njit(cache=True, parallel=True)
def _pool_local_label_kernel(labels, mapping):
    n_samples, n_bins, _ = labels.shape
    pooled_labels = np.full_like(labels, -1, dtype=np.int16)
    for flat_index in prange(n_samples * n_bins):
        sample = flat_index // n_bins
        block = flat_index - sample * n_bins
        for track in range(2):
            label = int(labels[sample, block, track])
            if label >= 0:
                pooled_labels[sample, block, track] = mapping[block, label]
    return pooled_labels


@njit(cache=True)
def _unique_trajectory_state_kernel(mapping):
    """Deduplicate complete founder trajectories in first-occurrence order."""
    n_bins, n_states = mapping.shape
    unique_states = np.empty(n_states, dtype=np.int64)
    number_of_unique_states = 0
    for state in range(n_states):
        duplicate = False
        for unique_index in range(number_of_unique_states):
            representative = int(unique_states[unique_index])
            equivalent = True
            for block in range(n_bins):
                if mapping[block, state] != mapping[block, representative]:
                    equivalent = False
                    break
            if equivalent:
                duplicate = True
                break
        if not duplicate:
            unique_states[number_of_unique_states] = state
            number_of_unique_states += 1
    return unique_states[:number_of_unique_states]


@njit(cache=True, parallel=True)
def _continuation_bridge_kernel(
    mapping,
    class_counts,
    unique_trajectory_states,
    maximum_classes,
):
    n_bins = mapping.shape[0]
    destination_counts = np.zeros(
        (n_bins, maximum_classes, maximum_classes), dtype=np.int64
    )
    continuation_bridge = np.zeros(
        (n_bins, maximum_classes, maximum_classes), dtype=np.float64
    )
    for block in prange(1, n_bins):
        for unique_index in range(len(unique_trajectory_states)):
            state = int(unique_trajectory_states[unique_index])
            previous = int(mapping[block - 1, state])
            current = int(mapping[block, state])
            destination_counts[block, previous, current] += 1
        for previous in range(int(class_counts[block - 1])):
            denominator = 0
            for current in range(int(class_counts[block])):
                denominator += destination_counts[block, previous, current]
            for current in range(int(class_counts[block])):
                count = destination_counts[block, previous, current]
                if count > 0:
                    continuation_bridge[block, previous, current] = (
                        float(count) / float(denominator)
                    )
    return continuation_bridge


def _pool_local_ibs_states(
    stacked_labels: np.ndarray,
    founder_alleles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool numeric labels that are locally IBS-equivalent.

    Reconstructed founder-label numbers are not biological identities.  On
    every cache bin, founder rows with the same observed allele vector are
    collapsed into one local state before external-parent frequencies and
    transitions are estimated.  The transition model is then estimated
    between these pooled local classes, so splitting one IBS haplotype into
    duplicate numeric labels cannot manufacture ancestry switches.
    """
    labels = np.asarray(stacked_labels, dtype=np.int16)
    founders = np.asarray(founder_alleles, dtype=np.int8)
    mapping, class_counts, pooled_founders, active = (
        _local_ibs_class_kernel(founders)
    )
    maximum_classes = int(np.max(class_counts))
    pooled_founders = np.ascontiguousarray(
        pooled_founders[:maximum_classes]
    )
    active = np.ascontiguousarray(active[:, :maximum_classes])
    pooled_labels = _pool_local_label_kernel(labels, mapping)

    # Class numbers are local to a bin.  Preserve physical ancestry continuity
    # by following each distinct chromosome-wide reconstructed founder path
    # into its next-bin pooled class.  Globally duplicate trajectories receive
    # one representative, so duplicating an arbitrary numeric label cannot
    # alter merge/split probabilities.
    unique_trajectory_states = _unique_trajectory_state_kernel(mapping)
    trajectory_classes = np.ascontiguousarray(
        mapping[:, unique_trajectory_states].T, dtype=np.int16
    )
    continuation_bridge = _continuation_bridge_kernel(
        mapping,
        class_counts,
        unique_trajectory_states,
        maximum_classes,
    )
    return (
        np.ascontiguousarray(pooled_labels),
        pooled_founders,
        active,
        np.ascontiguousarray(continuation_bridge),
        trajectory_classes,
    )


@njit(cache=True, parallel=True)
def _parenthood_structure_count_kernel(labels, trios, required_edges=None):
    """Count strict, missing-aware IBS support in pooled-label bins.

    Production callers pass the sparse symmetric edge mask.  Omitting it
    retains the historical full-matrix private API used by diagnostics and
    focused reference tests.
    """
    n_samples, n_bins, _ = labels.shape
    edge_matched = np.zeros((n_samples, n_samples), dtype=np.float64)
    edge_exposed = np.zeros((n_samples, n_samples), dtype=np.float64)
    for first in prange(n_samples):
        for second in range(n_samples):
            if (
                required_edges is not None
                and not required_edges[first, second]
            ):
                continue
            for block in range(n_bins):
                a0 = labels[first, block, 0]
                a1 = labels[first, block, 1]
                b0 = labels[second, block, 0]
                b1 = labels[second, block, 1]
                if a0 < 0 or a1 < 0 or b0 < 0 or b1 < 0:
                    continue
                edge_exposed[first, second] += 1.0
                if a0 == b0 or a0 == b1 or a1 == b0 or a1 == b1:
                    edge_matched[first, second] += 1.0

    pair_explained = np.zeros(len(trios), dtype=np.float64)
    pair_exposed = np.zeros(len(trios), dtype=np.float64)
    for row in prange(len(trios)):
        child = int(trios[row, 0])
        parent1 = int(trios[row, 1])
        parent2 = int(trios[row, 2])
        for block in range(n_bins):
            c0 = labels[child, block, 0]
            c1 = labels[child, block, 1]
            p10 = labels[parent1, block, 0]
            p11 = labels[parent1, block, 1]
            p20 = labels[parent2, block, 0]
            p21 = labels[parent2, block, 1]
            if (
                c0 < 0 or c1 < 0 or p10 < 0 or p11 < 0
                or p20 < 0 or p21 < 0
            ):
                continue
            pair_exposed[row] += 1.0
            first_assignment = (
                (c0 == p10 or c0 == p11)
                and (c1 == p20 or c1 == p21)
            )
            second_assignment = (
                (c1 == p10 or c1 == p11)
                and (c0 == p20 or c0 == p21)
            )
            if first_assignment or second_assignment:
                pair_explained[row] += 1.0
    return edge_matched, edge_exposed, pair_explained, pair_exposed


def _deduplicate_global_founder_trajectories(
    physical_founders: np.ndarray,
    physical_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate complete B-by-K founder rows in first-occurrence order."""
    founders = np.asarray(physical_founders, dtype=np.int8)
    labels = np.asarray(physical_labels, dtype=np.int16)
    key_to_unique: dict[tuple[int, ...], int] = {}
    representatives = []
    old_to_unique = np.empty(founders.shape[0], dtype=np.int16)
    for founder in range(founders.shape[0]):
        key = tuple(int(value) for value in founders[founder].ravel())
        unique = key_to_unique.get(key)
        if unique is None:
            unique = len(representatives)
            key_to_unique[key] = unique
            representatives.append(founder)
        old_to_unique[founder] = unique
    remapped_labels = labels.copy()
    called = remapped_labels >= 0
    remapped_labels[called] = old_to_unique[remapped_labels[called]]
    return (
        np.ascontiguousarray(founders[representatives]),
        np.ascontiguousarray(remapped_labels),
        old_to_unique,
    )


def _lift_pooled_external_chains_to_physical_founders(
    pooled_state_probability: np.ndarray,
    pooled_transition_probability: np.ndarray,
    physical_founders: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed the pooled external chain in exact-founder coordinates.

    This is an auxiliary strongly-lumpable embedding, not a physical external-
    ancestry model. Each pooled destination-class mass is divided among the
    chromosome-global exact founders in that local class. Projecting the lift
    back over class members exactly recovers the pooled initial and transition
    laws. Since class members have identical emissions at that bin, G^2 and
    F^2 G child likelihoods are preserved while candidate states keep one
    fixed chromosome-global Markov dimension.
    """
    physical = np.asarray(physical_founders, dtype=np.int8)
    mapping, _, _, _ = _local_ibs_class_kernel(physical)
    pooled_initial = np.asarray(pooled_state_probability, dtype=np.float64)
    pooled_transition = np.asarray(
        pooled_transition_probability, dtype=np.float64
    )
    n_children, n_bins, _ = pooled_initial.shape
    n_founders = physical.shape[0]
    membership = np.zeros((n_bins, n_founders), dtype=np.int64)
    for block in range(n_bins):
        membership[block] = np.bincount(
            mapping[block], minlength=n_founders
        )

    initial = np.empty((n_children, n_founders), dtype=np.float64)
    for founder in range(n_founders):
        state = int(mapping[0, founder])
        initial[:, founder] = (
            pooled_initial[:, 0, state] / membership[0, state]
        )

    transition = np.empty(
        (n_children, max(n_bins - 1, 0), n_founders, n_founders),
        dtype=np.float64,
    )
    for boundary in range(1, n_bins):
        for previous in range(n_founders):
            previous_class = int(mapping[boundary - 1, previous])
            for current in range(n_founders):
                current_class = int(mapping[boundary, current])
                transition[:, boundary - 1, previous, current] = (
                    pooled_transition[
                        :, boundary, previous_class, current_class
                    ]
                    / membership[boundary, current_class]
                )
    return initial, transition


@njit(cache=True, parallel=True)
def _ancestry_junction_count_kernel(pooled_labels, trajectory_classes):
    """Minimum chromosome-wide founder switches for each diploid painting.

    The hidden state is an unordered pair drawn from deduplicated whole-
    chromosome founder trajectories (the two entries may be equal). Local IBS
    equivalence is an
    emission ambiguity, not permission to splice two different trajectories
    together for free. A transition costs zero for the same unordered pair,
    one when the pairs share one trajectory, and two otherwise. Per-founder
    and global minima reduce each dynamic-programming update from quartic to
    quadratic in the number of unique trajectories.

    Missing painted labels are uninformative. The second returned vector is
    the number of observed haplotype-bin labels and is carried into the depth
    model so incomplete paintings cannot masquerade as shallow ancestry merely
    because fewer switches were detectable.
    """
    n_samples, n_bins, _ = pooled_labels.shape
    n_trajectories = trajectory_classes.shape[0]
    n_pairs = n_trajectories * (n_trajectories + 1) // 2
    pair_first = np.empty(n_pairs, dtype=np.int64)
    pair_second = np.empty(n_pairs, dtype=np.int64)
    pair_index = 0
    for first in range(n_trajectories):
        for second in range(first, n_trajectories):
            pair_first[pair_index] = first
            pair_second[pair_index] = second
            pair_index += 1

    output = np.zeros(n_samples, dtype=np.int64)
    callable_bins = np.zeros(n_samples, dtype=np.int64)
    unreachable = 1 << 30
    for sample in prange(n_samples):
        previous_cost = np.zeros(n_pairs, dtype=np.int64)
        current_cost = np.empty(n_pairs, dtype=np.int64)
        per_trajectory_minimum = np.empty(n_trajectories, dtype=np.int64)
        observed_count = 0
        for block in range(n_bins):
            observed0 = int(pooled_labels[sample, block, 0])
            observed1 = int(pooled_labels[sample, block, 1])
            observed_count += int(observed0 >= 0) + int(observed1 >= 0)

            if block > 0:
                global_minimum = unreachable
                for trajectory in range(n_trajectories):
                    per_trajectory_minimum[trajectory] = unreachable
                for state in range(n_pairs):
                    value = int(previous_cost[state])
                    if value < global_minimum:
                        global_minimum = value
                    first = int(pair_first[state])
                    second = int(pair_second[state])
                    if value < per_trajectory_minimum[first]:
                        per_trajectory_minimum[first] = value
                    if value < per_trajectory_minimum[second]:
                        per_trajectory_minimum[second] = value

            for state in range(n_pairs):
                first = int(pair_first[state])
                second = int(pair_second[state])
                first_class = int(trajectory_classes[first, block])
                second_class = int(trajectory_classes[second, block])
                if observed0 < 0 and observed1 < 0:
                    compatible = True
                elif observed0 < 0:
                    compatible = (
                        first_class == observed1 or second_class == observed1
                    )
                elif observed1 < 0:
                    compatible = (
                        first_class == observed0 or second_class == observed0
                    )
                else:
                    compatible = (
                        first_class == observed0 and second_class == observed1
                    ) or (
                        first_class == observed1 and second_class == observed0
                    )
                if not compatible:
                    current_cost[state] = unreachable
                elif block == 0:
                    current_cost[state] = 0
                else:
                    shared = min(
                        int(per_trajectory_minimum[first]),
                        int(per_trajectory_minimum[second]),
                    ) + 1
                    unrelated = int(global_minimum) + 2
                    current_cost[state] = min(
                        int(previous_cost[state]), shared, unrelated
                    )
            swap = previous_cost
            previous_cost = current_cost
            current_cost = swap
        output[sample] = int(np.min(previous_cost))
        callable_bins[sample] = observed_count
    return output, callable_bins


@njit(cache=True, parallel=True)
def _background_totals(
    stacked_alleles,
    stacked_labels,
    n_states,
):
    n_samples, n_bins, _, n_snps = stacked_alleles.shape
    state_counts = np.zeros((n_bins, n_states), dtype=np.int64)
    alt_counts = np.zeros((n_bins, n_snps), dtype=np.int64)
    allele_counts = np.zeros((n_bins, n_snps), dtype=np.int64)
    transition_counts = np.zeros(
        (n_bins, n_states, n_states), dtype=np.int64
    )
    for block in prange(n_bins):
        for sample in range(n_samples):
            for track in range(2):
                label = int(stacked_labels[sample, block, track])
                if label >= 0:
                    state_counts[block, label] += 1
                for snp in range(n_snps):
                    allele = int(stacked_alleles[sample, block, track, snp])
                    if allele >= 0:
                        allele_counts[block, snp] += 1
                        alt_counts[block, snp] += allele
                if block > 0:
                    previous = int(stacked_labels[sample, block - 1, track])
                    if previous >= 0 and label >= 0:
                        transition_counts[block, previous, label] += 1
    return state_counts, alt_counts, allele_counts, transition_counts


@njit(cache=True, parallel=True)
def _prepare_leave_child_out_background(
    stacked_alleles,
    stacked_labels,
    selected_markers_per_bin,
    active_states,
    continuation_bridge,
    switch_probability,
    state_counts,
    alt_counts,
    allele_counts,
    transition_counts,
    external_state_pseudocount,
    external_transition_pseudocount,
    markers_per_information_block,
    effective_markers_per_information_block,
):
    """Create child-left-out pooled external-parent distributions.

    Transition rows use empirical pooled local-IBS transitions from all other
    sampled homologs.  Their Dirichlet target follows the same reconstructed
    ancestry label across adjacent local IBS classes with probability
    ``1 - theta`` and redraws from next-bin occupancy with probability
    ``theta``.  Shrinking directly to occupancy at every bin would create a
    bin-resolution-dependent ancestry switch process and systematically
    penalize each unobserved parental gamete.
    """
    n_samples, n_bins, _, n_snps = stacked_alleles.shape
    n_states = state_counts.shape[1]
    state_probability = np.zeros(
        (n_samples, n_bins, n_states), dtype=np.float64
    )
    external_transition_probability = np.zeros(
        (n_samples, n_bins, n_states, n_states), dtype=np.float64
    )
    background_alt_probability = np.empty(
        (n_samples, n_bins, n_snps), dtype=np.float64
    )

    information_block = np.empty(n_bins, dtype=np.int64)
    group = 0
    markers_in_group = 0
    for block in range(n_bins):
        if block > 0 and markers_in_group >= markers_per_information_block:
            group += 1
            markers_in_group = 0
        information_block[block] = group
        markers_in_group += max(int(selected_markers_per_bin[block]), 1)
    n_information_blocks = group + 1
    information_exponent = np.zeros((n_samples, n_bins), dtype=np.float64)

    for child in prange(n_samples):
        valid_sites_per_group = np.zeros(
            n_information_blocks, dtype=np.int64
        )
        for block in range(n_bins):
            count_sum = 0
            active_count = 0
            for state in range(n_states):
                if active_states[block, state]:
                    active_count += 1
                count = int(state_counts[block, state])
                for track in range(2):
                    count -= int(
                        int(stacked_labels[child, block, track]) == state
                    )
                count = max(count, 0)
                state_probability[child, block, state] = count
                count_sum += count
            active_count = max(active_count, 1)
            denominator = count_sum + external_state_pseudocount
            for state in range(n_states):
                if active_states[block, state]:
                    state_probability[child, block, state] = (
                        state_probability[child, block, state]
                        + external_state_pseudocount / active_count
                    ) / denominator
                else:
                    state_probability[child, block, state] = 0.0

            for snp in range(n_snps):
                count = int(allele_counts[block, snp])
                alts = int(alt_counts[block, snp])
                for track in range(2):
                    allele = int(stacked_alleles[child, block, track, snp])
                    if allele >= 0:
                        count -= 1
                        alts -= allele
                background_alt_probability[child, block, snp] = (
                    alts + 0.5
                ) / max(count + 1.0, 1.0)
                if (
                    int(stacked_alleles[child, block, 0, snp]) >= 0
                    or int(stacked_alleles[child, block, 1, snp]) >= 0
                ):
                    valid_sites_per_group[information_block[block]] += 1

            if block > 0:
                for previous in range(n_states):
                    row_total = 0
                    for current in range(n_states):
                        count = int(
                            transition_counts[block, previous, current]
                        )
                        for track in range(2):
                            child_previous = int(
                                stacked_labels[child, block - 1, track]
                            )
                            child_current = int(
                                stacked_labels[child, block, track]
                            )
                            if (
                                child_previous == previous
                                and child_current == current
                            ):
                                count -= 1
                        count = max(count, 0)
                        external_transition_probability[
                            child, block, previous, current
                        ] = count
                        row_total += count
                    transition_denominator = (
                        row_total + external_transition_pseudocount
                    )
                    theta = switch_probability[block]
                    for current in range(n_states):
                        linked_homolog_transition = (
                            external_transition_probability[
                                child, block, previous, current
                            ]
                            + external_transition_pseudocount
                            * continuation_bridge[block, previous, current]
                        ) / transition_denominator
                        external_transition_probability[
                            child, block, previous, current
                        ] = (
                            (1.0 - theta) * linked_homolog_transition
                            + theta
                            * state_probability[child, block, current]
                        )

        for block in range(n_bins):
            valid_sites = valid_sites_per_group[information_block[block]]
            if valid_sites > 0:
                information_exponent[child, block] = (
                    min(
                        effective_markers_per_information_block,
                        float(valid_sites),
                    )
                    / float(valid_sites)
                )

    return (
        state_probability,
        external_transition_probability,
        background_alt_probability,
        information_exponent,
    )


@njit(cache=True, inline="always")
def _external_transition_vector(source, destination, transition):
    """Apply one row-normalized pooled external-state transition."""
    n_states = source.shape[0]
    for current in range(n_states):
        value = 0.0
        for previous in range(n_states):
            value += source[previous] * transition[previous, current]
        destination[current] = value


@njit(cache=True, inline="always")
def _log_emission_hard_sources(
    child_alleles,
    first_source,
    second_source,
    background_alt,
    orientation,
    mismatch_probability,
    exponent,
):
    n_snps = child_alleles.shape[1]
    value = 0.0
    first_child_track = orientation
    second_child_track = 1 - orientation
    for snp in range(n_snps):
        first_alt = _source_alt_probability(
            int(first_source[snp]), background_alt[snp]
        )
        second_alt = _source_alt_probability(
            int(second_source[snp]), background_alt[snp]
        )
        value += _observed_allele_log_probability(
            int(child_alleles[first_child_track, snp]),
            first_alt,
            mismatch_probability,
        )
        value += _observed_allele_log_probability(
            int(child_alleles[second_child_track, snp]),
            second_alt,
            mismatch_probability,
        )
    return exponent * value


@njit(cache=True, parallel=True)
def _score_zero_parent_forward_kernel(
    stacked_alleles,
    stacked_hom_mask,
    founder_alleles,
    state_probability,
    external_transition_probability,
    background_alt_probability,
    information_exponent,
    phase_switch_probability,
    mismatch_probability,
    eligible_children,
):
    n_samples, n_bins, _, _ = stacked_alleles.shape
    n_states = founder_alleles.shape[0]
    output = np.empty(n_samples, dtype=np.float64)

    for child in prange(n_samples):
        if not eligible_children[child]:
            output[child] = -math.inf
            continue
        forward = np.empty((2, n_states, n_states), dtype=np.float64)
        work1 = np.empty_like(forward)
        work2 = np.empty_like(forward)
        source = np.empty(n_states, dtype=np.float64)
        destination = np.empty(n_states, dtype=np.float64)
        total_log_likelihood = 0.0
        for orientation in range(2):
            for first in range(n_states):
                for second in range(n_states):
                    forward[orientation, first, second] = (
                        0.5
                        * state_probability[child, 0, first]
                        * state_probability[child, 0, second]
                    )

        for block in range(n_bins):
            if block > 0:
                transition = external_transition_probability[child, block]
                for orientation in range(2):
                    for second in range(n_states):
                        for first in range(n_states):
                            source[first] = forward[orientation, first, second]
                        _external_transition_vector(
                            source, destination, transition
                        )
                        for first in range(n_states):
                            work1[orientation, first, second] = destination[first]
                    for first in range(n_states):
                        for second in range(n_states):
                            source[second] = work1[orientation, first, second]
                        _external_transition_vector(
                            source, destination, transition
                        )
                        for second in range(n_states):
                            work2[orientation, first, second] = destination[second]
                phase_rho = _phase_switch_probability(
                    bool(stacked_hom_mask[child, block - 1]),
                    bool(stacked_hom_mask[child, block]),
                    phase_switch_probability,
                )
                for first in range(n_states):
                    for second in range(n_states):
                        value0 = work2[0, first, second]
                        value1 = work2[1, first, second]
                        forward[0, first, second] = (
                            (1.0 - phase_rho) * value0 + phase_rho * value1
                        )
                        forward[1, first, second] = (
                            (1.0 - phase_rho) * value1 + phase_rho * value0
                        )

            maximum = -math.inf
            for orientation in range(2):
                for first in range(n_states):
                    for second in range(n_states):
                        log_emission = _log_emission_hard_sources(
                            stacked_alleles[child, block],
                            founder_alleles[first, block],
                            founder_alleles[second, block],
                            background_alt_probability[child, block],
                            orientation,
                            mismatch_probability,
                            information_exponent[child, block],
                        )
                        work1[orientation, first, second] = log_emission
                        maximum = max(maximum, log_emission)
            scale = 0.0
            for orientation in range(2):
                for first in range(n_states):
                    for second in range(n_states):
                        forward[orientation, first, second] *= math.exp(
                            work1[orientation, first, second] - maximum
                        )
                        scale += forward[orientation, first, second]
            scale = max(scale, _TINY)
            total_log_likelihood += maximum + math.log(scale)
            for orientation in range(2):
                for first in range(n_states):
                    for second in range(n_states):
                        forward[orientation, first, second] /= scale
        output[child] = total_log_likelihood
    return output


@njit(cache=True, parallel=True)
def _score_one_parent_forward_kernel(
    stacked_alleles,
    stacked_hom_mask,
    founder_alleles,
    state_probability,
    external_transition_probability,
    background_alt_probability,
    information_exponent,
    switch_probability,
    phase_switch_probability,
    mismatch_probability,
):
    n_samples, n_bins, _, _ = stacked_alleles.shape
    n_states = founder_alleles.shape[0]
    output = np.full((n_samples, n_samples), -math.inf, dtype=np.float64)

    for flat_index in prange(n_samples * n_samples):
        child = flat_index // n_samples
        parent = flat_index - child * n_samples
        if parent == child:
            continue
        forward = np.empty((2, 2, n_states), dtype=np.float64)
        work1 = np.empty_like(forward)
        work2 = np.empty_like(forward)
        source = np.empty(n_states, dtype=np.float64)
        destination = np.empty(n_states, dtype=np.float64)
        total_log_likelihood = 0.0
        for orientation in range(2):
            for track in range(2):
                for state in range(n_states):
                    forward[orientation, track, state] = (
                        0.25 * state_probability[child, 0, state]
                    )

        for block in range(n_bins):
            if block > 0:
                transition = external_transition_probability[child, block]
                for orientation in range(2):
                    for track in range(2):
                        for state in range(n_states):
                            source[state] = forward[orientation, track, state]
                        _external_transition_vector(
                            source, destination, transition
                        )
                        for state in range(n_states):
                            work1[orientation, track, state] = destination[state]
                theta = _phase_switch_probability(
                    bool(stacked_hom_mask[parent, block - 1]),
                    bool(stacked_hom_mask[parent, block]),
                    switch_probability[block],
                )
                for orientation in range(2):
                    for state in range(n_states):
                        value0 = work1[orientation, 0, state]
                        value1 = work1[orientation, 1, state]
                        work2[orientation, 0, state] = (
                            (1.0 - theta) * value0 + theta * value1
                        )
                        work2[orientation, 1, state] = (
                            (1.0 - theta) * value1 + theta * value0
                        )
                phase_rho = _phase_switch_probability(
                    bool(stacked_hom_mask[child, block - 1]),
                    bool(stacked_hom_mask[child, block]),
                    phase_switch_probability,
                )
                for track in range(2):
                    for state in range(n_states):
                        value0 = work2[0, track, state]
                        value1 = work2[1, track, state]
                        forward[0, track, state] = (
                            (1.0 - phase_rho) * value0 + phase_rho * value1
                        )
                        forward[1, track, state] = (
                            (1.0 - phase_rho) * value1 + phase_rho * value0
                        )

            maximum = -math.inf
            for orientation in range(2):
                for track in range(2):
                    for state in range(n_states):
                        log_emission = _log_emission_hard_sources(
                            stacked_alleles[child, block],
                            stacked_alleles[parent, block, track],
                            founder_alleles[state, block],
                            background_alt_probability[child, block],
                            orientation,
                            mismatch_probability,
                            information_exponent[child, block],
                        )
                        work1[orientation, track, state] = log_emission
                        maximum = max(maximum, log_emission)
            scale = 0.0
            for orientation in range(2):
                for track in range(2):
                    for state in range(n_states):
                        forward[orientation, track, state] *= math.exp(
                            work1[orientation, track, state] - maximum
                        )
                        scale += forward[orientation, track, state]
            scale = max(scale, _TINY)
            total_log_likelihood += maximum + math.log(scale)
            for orientation in range(2):
                for track in range(2):
                    for state in range(n_states):
                        forward[orientation, track, state] /= scale
        output[child, parent] = total_log_likelihood
    return output


@njit(cache=True, parallel=True)
def _score_two_parent_forward_kernel(
    stacked_alleles,
    stacked_hom_mask,
    trios,
    background_alt_probability,
    information_exponent,
    switch_probability,
    phase_switch_probability,
    mismatch_probability,
):
    n_samples, n_bins, _, _ = stacked_alleles.shape
    output = np.full(len(trios), -math.inf, dtype=np.float64)
    for row in prange(len(trios)):
        child = int(trios[row, 0])
        parent1 = int(trios[row, 1])
        parent2 = int(trios[row, 2])
        if (
            child < 0
            or child >= n_samples
            or parent1 < 0
            or parent1 >= n_samples
            or parent2 < 0
            or parent2 >= n_samples
            or parent1 == child
            or parent2 == child
            or parent1 == parent2
        ):
            continue
        forward = np.full((2, 2, 2), 0.125, dtype=np.float64)
        work1 = np.empty_like(forward)
        work2 = np.empty_like(forward)
        total_log_likelihood = 0.0
        for block in range(n_bins):
            if block > 0:
                theta1 = _phase_switch_probability(
                    bool(stacked_hom_mask[parent1, block - 1]),
                    bool(stacked_hom_mask[parent1, block]),
                    switch_probability[block],
                )
                theta2 = _phase_switch_probability(
                    bool(stacked_hom_mask[parent2, block - 1]),
                    bool(stacked_hom_mask[parent2, block]),
                    switch_probability[block],
                )
                stay1 = 1.0 - theta1
                stay2 = 1.0 - theta2
                for orientation in range(2):
                    for second in range(2):
                        value0 = forward[orientation, 0, second]
                        value1 = forward[orientation, 1, second]
                        work1[orientation, 0, second] = (
                            stay1 * value0 + theta1 * value1
                        )
                        work1[orientation, 1, second] = (
                            stay1 * value1 + theta1 * value0
                        )
                    for first in range(2):
                        value0 = work1[orientation, first, 0]
                        value1 = work1[orientation, first, 1]
                        work2[orientation, first, 0] = (
                            stay2 * value0 + theta2 * value1
                        )
                        work2[orientation, first, 1] = (
                            stay2 * value1 + theta2 * value0
                        )
                phase_rho = _phase_switch_probability(
                    bool(stacked_hom_mask[child, block - 1]),
                    bool(stacked_hom_mask[child, block]),
                    phase_switch_probability,
                )
                for first in range(2):
                    for second in range(2):
                        value0 = work2[0, first, second]
                        value1 = work2[1, first, second]
                        forward[0, first, second] = (
                            (1.0 - phase_rho) * value0 + phase_rho * value1
                        )
                        forward[1, first, second] = (
                            (1.0 - phase_rho) * value1 + phase_rho * value0
                        )

            maximum = -math.inf
            for orientation in range(2):
                for first in range(2):
                    for second in range(2):
                        log_emission = _log_emission_hard_sources(
                            stacked_alleles[child, block],
                            stacked_alleles[parent1, block, first],
                            stacked_alleles[parent2, block, second],
                            background_alt_probability[child, block],
                            orientation,
                            mismatch_probability,
                            information_exponent[child, block],
                        )
                        work1[orientation, first, second] = log_emission
                        maximum = max(maximum, log_emission)
            scale = 0.0
            for orientation in range(2):
                for first in range(2):
                    for second in range(2):
                        forward[orientation, first, second] *= math.exp(
                            work1[orientation, first, second] - maximum
                        )
                        scale += forward[orientation, first, second]
            scale = max(scale, _TINY)
            total_log_likelihood += maximum + math.log(scale)
            for orientation in range(2):
                for first in range(2):
                    for second in range(2):
                        forward[orientation, first, second] /= scale
        output[row] = total_log_likelihood
    return output


@njit(cache=True, inline="always")
def _single_source_log_emission(
    child_track_alleles,
    source_alleles,
    background_alt,
    mismatch_probability,
    hard_match_log_probability,
    hard_mismatch_log_probability,
):
    """Log emission contributed by one parental source.

    Hard reconstructed alleles have only two possible probabilities, so their
    logarithms are computed once per kernel rather than once per
    source/state/SNP/candidate. Missing source alleles retain the exact
    background-mixture calculation used by
    :func:`_observed_allele_log_probability`.
    """
    value = 0.0
    for snp in range(child_track_alleles.shape[0]):
        observed = int(child_track_alleles[snp])
        if observed < 0:
            continue
        source = int(source_alleles[snp])
        if source >= 0:
            if source == observed:
                value += hard_match_log_probability
            else:
                value += hard_mismatch_log_probability
        else:
            source_alt = background_alt[snp]
            if observed == 1:
                probability = (
                    (1.0 - mismatch_probability) * source_alt
                    + mismatch_probability * (1.0 - source_alt)
                )
            else:
                probability = (
                    (1.0 - mismatch_probability) * (1.0 - source_alt)
                    + mismatch_probability * source_alt
                )
            value += math.log(max(probability, _TINY))
    return value


@njit(cache=True, parallel=True)
def _effective_switch_probability_kernel(
    stacked_hom_mask,
    switch_probability,
    phase_switch_probability,
):
    """Precompute transmission and child-orientation switch probabilities."""
    n_samples, n_bins = stacked_hom_mask.shape
    transmission = np.empty((n_samples, n_bins), dtype=np.float64)
    phase = np.empty((n_samples, n_bins), dtype=np.float64)
    for sample in prange(n_samples):
        transmission[sample, 0] = 0.0
        phase[sample, 0] = 0.0
        for block in range(1, n_bins):
            reset = (
                bool(stacked_hom_mask[sample, block - 1])
                or bool(stacked_hom_mask[sample, block])
            )
            if reset:
                transmission[sample, block] = 0.5
                phase[sample, block] = 0.5
            else:
                transmission[sample, block] = switch_probability[block]
                phase[sample, block] = phase_switch_probability
    return transmission, phase


@njit(cache=True, parallel=True)
def _score_one_parent_forward_kernel_grouped(
    stacked_alleles,
    founder_alleles,
    state_probability,
    external_transition_probability,
    background_alt_probability,
    information_exponent,
    eligible_parents,
    effective_transmission_probability,
    effective_phase_probability,
    mismatch_probability,
    child_start,
    child_end,
):
    """Score M1 while reusing emissions across every parent of one child.

    This is the same normalized forward model as
    :func:`_score_one_parent_forward_kernel`, reorganised from
    ``(child,parent)->blocks`` to ``child->blocks->parents``. A known-parent
    source emission is computed once per block and reused for every external
    founder state; an external-founder emission is computed once and reused
    for every candidate parent.
    """
    n_samples, n_bins, _, _ = stacked_alleles.shape
    n_states = founder_alleles.shape[0]
    n_children = child_end - child_start
    output = np.full((n_children, n_samples), -math.inf, dtype=np.float64)
    hard_match_log_probability = math.log(1.0 - mismatch_probability)
    hard_mismatch_log_probability = math.log(mismatch_probability)

    for local_child in prange(n_children):
        child = child_start + local_child
        if not np.any(eligible_parents[child]):
            continue
        forward = np.empty(
            (n_samples, 2, 2, n_states), dtype=np.float64
        )
        work1 = np.empty_like(forward)
        work2 = np.empty_like(forward)
        totals = np.zeros(n_samples, dtype=np.float64)
        known_emission = np.empty((n_samples, 2, 2), dtype=np.float64)
        external_emission = np.empty((n_states, 2), dtype=np.float64)

        for parent in range(n_samples):
            if not eligible_parents[child, parent]:
                continue
            for orientation in range(2):
                for track in range(2):
                    for state in range(n_states):
                        forward[parent, orientation, track, state] = (
                            0.25 * state_probability[child, 0, state]
                        )

        for block in range(n_bins):
            if block > 0:
                transition = external_transition_probability[child, block]
                phase_rho = effective_phase_probability[child, block]
                phase_stay = 1.0 - phase_rho
                for parent in range(n_samples):
                    if not eligible_parents[child, parent]:
                        continue
                    for orientation in range(2):
                        for track in range(2):
                            for current in range(n_states):
                                value = 0.0
                                for previous in range(n_states):
                                    value += (
                                        forward[
                                            parent,
                                            orientation,
                                            track,
                                            previous,
                                        ]
                                        * transition[previous, current]
                                    )
                                work1[
                                    parent, orientation, track, current
                                ] = value
                    theta = effective_transmission_probability[parent, block]
                    stay = 1.0 - theta
                    for orientation in range(2):
                        for state in range(n_states):
                            value0 = work1[parent, orientation, 0, state]
                            value1 = work1[parent, orientation, 1, state]
                            work2[parent, orientation, 0, state] = (
                                stay * value0 + theta * value1
                            )
                            work2[parent, orientation, 1, state] = (
                                stay * value1 + theta * value0
                            )
                    for track in range(2):
                        for state in range(n_states):
                            value0 = work2[parent, 0, track, state]
                            value1 = work2[parent, 1, track, state]
                            forward[parent, 0, track, state] = (
                                phase_stay * value0 + phase_rho * value1
                            )
                            forward[parent, 1, track, state] = (
                                phase_stay * value1 + phase_rho * value0
                            )

            child_block = stacked_alleles[child, block]
            background_block = background_alt_probability[child, block]
            for parent in range(n_samples):
                if not eligible_parents[child, parent]:
                    continue
                for track in range(2):
                    for child_track in range(2):
                        known_emission[parent, track, child_track] = (
                            _single_source_log_emission(
                                child_block[child_track],
                                stacked_alleles[parent, block, track],
                                background_block,
                                mismatch_probability,
                                hard_match_log_probability,
                                hard_mismatch_log_probability,
                            )
                        )
            for state in range(n_states):
                for child_track in range(2):
                    external_emission[state, child_track] = (
                        _single_source_log_emission(
                            child_block[child_track],
                            founder_alleles[state, block],
                            background_block,
                            mismatch_probability,
                            hard_match_log_probability,
                            hard_mismatch_log_probability,
                        )
                    )

            exponent = information_exponent[child, block]
            for parent in range(n_samples):
                if not eligible_parents[child, parent]:
                    continue
                maximum = -math.inf
                for orientation in range(2):
                    for track in range(2):
                        for state in range(n_states):
                            log_emission = exponent * (
                                known_emission[parent, track, orientation]
                                + external_emission[state, 1 - orientation]
                            )
                            work1[
                                parent, orientation, track, state
                            ] = log_emission
                            maximum = max(maximum, log_emission)
                scale = 0.0
                for orientation in range(2):
                    for track in range(2):
                        for state in range(n_states):
                            forward[
                                parent, orientation, track, state
                            ] *= math.exp(
                                work1[
                                    parent, orientation, track, state
                                ] - maximum
                            )
                            scale += forward[
                                parent, orientation, track, state
                            ]
                scale = max(scale, _TINY)
                totals[parent] += maximum + math.log(scale)
                inverse_scale = 1.0 / scale
                for orientation in range(2):
                    for track in range(2):
                        for state in range(n_states):
                            forward[
                                parent, orientation, track, state
                            ] *= inverse_scale

        for parent in range(n_samples):
            if eligible_parents[child, parent]:
                output[local_child, parent] = totals[parent]
    return output


@njit(cache=True, parallel=True)
def _score_two_parent_forward_kernel_grouped(
    stacked_alleles,
    sorted_trios,
    child_row_starts,
    background_alt_probability,
    information_exponent,
    effective_transmission_probability,
    effective_phase_probability,
    mismatch_probability,
    child_start,
    child_end,
):
    """Score M2 by child while reusing each parent-source emission."""
    n_samples, n_bins, _, _ = stacked_alleles.shape
    output_row_start = int(child_row_starts[child_start])
    output_row_end = int(child_row_starts[child_end])
    output = np.full(
        output_row_end - output_row_start, -math.inf, dtype=np.float64
    )
    hard_match_log_probability = math.log(1.0 - mismatch_probability)
    hard_mismatch_log_probability = math.log(mismatch_probability)

    for local_child in prange(child_end - child_start):
        child = child_start + local_child
        start = int(child_row_starts[child])
        end = int(child_row_starts[child + 1])
        n_pairs = end - start
        if n_pairs <= 0:
            continue
        forward = np.full((n_pairs, 2, 2, 2), 0.125, dtype=np.float64)
        work1 = np.empty_like(forward)
        work2 = np.empty_like(forward)
        totals = np.zeros(n_pairs, dtype=np.float64)
        source_emission = np.empty(
            (n_samples, 2, 2), dtype=np.float64
        )

        for block in range(n_bins):
            if block > 0:
                phase_rho = effective_phase_probability[child, block]
                phase_stay = 1.0 - phase_rho
                for local_row in range(n_pairs):
                    row = start + local_row
                    parent1 = int(sorted_trios[row, 1])
                    parent2 = int(sorted_trios[row, 2])
                    theta1 = effective_transmission_probability[
                        parent1, block
                    ]
                    theta2 = effective_transmission_probability[
                        parent2, block
                    ]
                    stay1 = 1.0 - theta1
                    stay2 = 1.0 - theta2
                    for orientation in range(2):
                        for second in range(2):
                            value0 = forward[
                                local_row, orientation, 0, second
                            ]
                            value1 = forward[
                                local_row, orientation, 1, second
                            ]
                            work1[
                                local_row, orientation, 0, second
                            ] = stay1 * value0 + theta1 * value1
                            work1[
                                local_row, orientation, 1, second
                            ] = stay1 * value1 + theta1 * value0
                        for first in range(2):
                            value0 = work1[
                                local_row, orientation, first, 0
                            ]
                            value1 = work1[
                                local_row, orientation, first, 1
                            ]
                            work2[
                                local_row, orientation, first, 0
                            ] = stay2 * value0 + theta2 * value1
                            work2[
                                local_row, orientation, first, 1
                            ] = stay2 * value1 + theta2 * value0
                    for first in range(2):
                        for second in range(2):
                            value0 = work2[local_row, 0, first, second]
                            value1 = work2[local_row, 1, first, second]
                            forward[local_row, 0, first, second] = (
                                phase_stay * value0 + phase_rho * value1
                            )
                            forward[local_row, 1, first, second] = (
                                phase_stay * value1 + phase_rho * value0
                            )

            child_block = stacked_alleles[child, block]
            background_block = background_alt_probability[child, block]
            for parent in range(n_samples):
                for track in range(2):
                    for child_track in range(2):
                        source_emission[parent, track, child_track] = (
                            _single_source_log_emission(
                                child_block[child_track],
                                stacked_alleles[parent, block, track],
                                background_block,
                                mismatch_probability,
                                hard_match_log_probability,
                                hard_mismatch_log_probability,
                            )
                        )

            exponent = information_exponent[child, block]
            for local_row in range(n_pairs):
                row = start + local_row
                parent1 = int(sorted_trios[row, 1])
                parent2 = int(sorted_trios[row, 2])
                maximum = -math.inf
                for orientation in range(2):
                    for first in range(2):
                        for second in range(2):
                            log_emission = exponent * (
                                source_emission[
                                    parent1, first, orientation
                                ]
                                + source_emission[
                                    parent2, second, 1 - orientation
                                ]
                            )
                            work1[
                                local_row, orientation, first, second
                            ] = log_emission
                            maximum = max(maximum, log_emission)
                scale = 0.0
                for orientation in range(2):
                    for first in range(2):
                        for second in range(2):
                            forward[
                                local_row, orientation, first, second
                            ] *= math.exp(
                                work1[
                                    local_row, orientation, first, second
                                ] - maximum
                            )
                            scale += forward[
                                local_row, orientation, first, second
                            ]
                scale = max(scale, _TINY)
                totals[local_row] += maximum + math.log(scale)
                inverse_scale = 1.0 / scale
                for orientation in range(2):
                    for first in range(2):
                        for second in range(2):
                            forward[
                                local_row, orientation, first, second
                            ] *= inverse_scale

        for local_row in range(n_pairs):
            output[start + local_row - output_row_start] = totals[local_row]
    return output


def score_parent_state_hmms(
    stacked_alleles,
    stacked_labels,
    stacked_hom_mask,
    founder_alleles,
    selected_markers_per_bin,
    switch_probability,
    trios,
    *,
    mismatch_probability=0.01,
    phase_switch_probability=0.01,
    markers_per_information_block=100,
    _eligible_children=None,
    _eligible_parents=None,
    effective_markers_per_information_block=1.0,
    external_state_pseudocount=1.0,
    external_transition_pseudocount=20.0,
    _dynamic_rebalance=False,
    _dynamic_child_chunk_floor=32,
    _dynamic_child_chunk_scale=4,
):
    """Score M0, M1 and M2 with one comparable normalized forward model."""
    raw_alleles = np.asarray(stacked_alleles)
    if raw_alleles.ndim == 3:
        raw_alleles = raw_alleles[..., None]
    if (
        raw_alleles.ndim != 4
        or min(raw_alleles.shape) < 1
        or np.any(~np.isin(raw_alleles, (-1, 0, 1)))
    ):
        raise SmartEvidenceError(
            "stacked_alleles must be non-empty hard alleles in {-1, 0, 1}"
        )
    alleles = np.ascontiguousarray(raw_alleles, dtype=np.int8)

    raw_founders = np.asarray(founder_alleles)
    if raw_founders.ndim == 2:
        raw_founders = raw_founders[..., None]
    if (
        raw_founders.ndim != 3
        or min(raw_founders.shape) < 1
        or np.any(~np.isin(raw_founders, (-1, 0, 1)))
    ):
        raise SmartEvidenceError(
            "founder_alleles must be non-empty hard alleles in {-1, 0, 1}"
        )
    founders = np.ascontiguousarray(raw_founders, dtype=np.int8)
    if founders.shape[1:] != (alleles.shape[1], alleles.shape[3]):
        raise SmartEvidenceError(
            "founder alleles must have shape (states, bins, SNPs)"
        )

    raw_labels = np.asarray(stacked_labels)
    if (
        raw_labels.shape != alleles.shape[:3]
        or np.any(~np.isfinite(raw_labels))
        or np.any(raw_labels != np.floor(raw_labels))
        or np.any(raw_labels < -1)
        or np.any(raw_labels >= founders.shape[0])
    ):
        raise SmartEvidenceError(
            "stacked_labels must have shape (samples, bins, 2) and values "
            "in {-1, 0, ..., states-1}"
        )
    labels = np.ascontiguousarray(raw_labels, dtype=np.int16)
    hom = np.ascontiguousarray(stacked_hom_mask, dtype=np.bool_)
    if hom.shape != alleles.shape[:2]:
        raise SmartEvidenceError(
            "stacked_hom_mask must have shape (samples, bins)"
        )

    raw_marker_counts = np.asarray(selected_markers_per_bin)
    if (
        raw_marker_counts.shape != (alleles.shape[1],)
        or np.any(~np.isfinite(raw_marker_counts))
        or np.any(raw_marker_counts != np.floor(raw_marker_counts))
        or np.any(raw_marker_counts < 0)
        or np.any(raw_marker_counts > alleles.shape[3])
        or np.sum(raw_marker_counts) < 1
    ):
        raise SmartEvidenceError(
            "selected_markers_per_bin must be integer counts between zero and "
            "the SNP-slot count, with at least one selected marker"
        )
    marker_counts = np.ascontiguousarray(
        raw_marker_counts, dtype=np.int64
    )
    founders = founders.copy()
    for block, marker_count in enumerate(marker_counts):
        founders[:, block, int(marker_count):] = -1
    theta = np.ascontiguousarray(switch_probability, dtype=np.float64)
    if (
        theta.shape != (alleles.shape[1],)
        or np.any(~np.isfinite(theta))
        or np.any((theta < 0.0) | (theta > 0.5))
    ):
        raise SmartEvidenceError(
            "switch probabilities must be finite with shape (bins,) and lie "
            "in [0, 0.5]"
        )

    raw_trios = np.asarray(trios)
    if (
        raw_trios.ndim != 2
        or raw_trios.shape[1] != 3
        or np.any(~np.isfinite(raw_trios))
        or np.any(raw_trios != np.floor(raw_trios))
    ):
        raise SmartEvidenceError("trios must be an integer array of shape (rows, 3)")
    trio_array = np.ascontiguousarray(raw_trios, dtype=np.int64)
    if len(trio_array):
        if np.any(trio_array < 0) or np.any(trio_array >= alleles.shape[0]):
            raise SmartEvidenceError("trio index outside the sample array")
        if (
            np.any(trio_array[:, 0] == trio_array[:, 1])
            or np.any(trio_array[:, 0] == trio_array[:, 2])
            or np.any(trio_array[:, 1] == trio_array[:, 2])
        ):
            raise SmartEvidenceError(
                "trios cannot contain self-parents or duplicate parents"
            )

    eligible_children, eligible_parents = _scoring_eligibility_masks(
        alleles.shape[0], _eligible_children, _eligible_parents
    )
    required_edges = np.ascontiguousarray(
        eligible_parents | eligible_parents.T
    )
    np.fill_diagonal(required_edges, True)

    if (
        not np.isfinite(mismatch_probability)
        or not 0.0 < mismatch_probability < 0.5
    ):
        raise SmartEvidenceError("mismatch_probability must lie in (0, 0.5)")
    if (
        not np.isfinite(phase_switch_probability)
        or not 0.0 <= phase_switch_probability <= 0.5
    ):
        raise SmartEvidenceError("phase_switch_probability must lie in [0, 0.5]")
    try:
        marker_block_size = float(markers_per_information_block)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SmartEvidenceError(
            "markers_per_information_block must be a positive integer"
        ) from exc
    if (
        isinstance(markers_per_information_block, (bool, np.bool_))
        or not np.isfinite(marker_block_size)
        or marker_block_size != math.floor(marker_block_size)
        or marker_block_size < 1
    ):
        raise SmartEvidenceError(
            "markers_per_information_block must be a positive integer"
        )
    for name, value in (
        (
            "effective_markers_per_information_block",
            effective_markers_per_information_block,
        ),
        ("external_state_pseudocount", external_state_pseudocount),
        ("external_transition_pseudocount", external_transition_pseudocount),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise SmartEvidenceError(f"{name} must be finite and positive")

    try:
        child_chunk_floor = operator.index(_dynamic_child_chunk_floor)
    except TypeError as exc:
        raise SmartEvidenceError(
            "_dynamic_child_chunk_floor must be a positive integer"
        ) from exc
    if child_chunk_floor < 1:
        raise SmartEvidenceError(
            "_dynamic_child_chunk_floor must be a positive integer"
        )
    try:
        child_chunk_scale = operator.index(_dynamic_child_chunk_scale)
    except TypeError as exc:
        raise SmartEvidenceError(
            "_dynamic_child_chunk_scale must be a non-negative integer"
        ) from exc
    if child_chunk_scale < 0:
        raise SmartEvidenceError(
            "_dynamic_child_chunk_scale must be a non-negative integer"
        )

    (
        labels,
        founders,
        active_states,
        continuation_bridge,
        trajectory_classes,
    ) = _pool_local_ibs_states(labels, founders)
    structure_counts = _parenthood_structure_count_kernel(
        labels, trio_array, required_edges
    )
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    totals = _background_totals(alleles, labels, founders.shape[0])
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    background = _prepare_leave_child_out_background(
        alleles,
        labels,
        marker_counts,
        active_states,
        continuation_bridge,
        theta,
        *totals,
        float(external_state_pseudocount),
        float(external_transition_pseudocount),
        int(marker_block_size),
        float(effective_markers_per_information_block),
    )
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    zero = _score_zero_parent_forward_kernel(
        alleles,
        hom,
        founders,
        *background,
        float(phase_switch_probability),
        float(mismatch_probability),
        eligible_children,
    )
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    effective_transmission, effective_phase = (
        _effective_switch_probability_kernel(
            hom,
            theta,
            float(phase_switch_probability),
        )
    )
    one = np.full(
        (alleles.shape[0], alleles.shape[0]), -math.inf, dtype=np.float64
    )
    child_start = 0
    while child_start < alleles.shape[0]:
        applied_threads = (
            _apply_smart_dynamic_threads()
            if _dynamic_rebalance
            else alleles.shape[0]
        )
        child_chunk_size = max(child_chunk_floor, applied_threads)
        if child_chunk_scale:
            child_chunk_size = max(
                child_chunk_size,
                min(64, child_chunk_scale * applied_threads),
            )
        child_end = min(
            child_start + child_chunk_size,
            alleles.shape[0],
        )
        one[child_start:child_end] = (
            _score_one_parent_forward_kernel_grouped(
                alleles,
                founders,
                *background,
                eligible_parents,
                effective_transmission,
                effective_phase,
                float(mismatch_probability),
                child_start,
                child_end,
            )
        )
        child_start = child_end
    if len(trio_array):
        if np.all(trio_array[1:, 0] >= trio_array[:-1, 0]):
            trio_order = None
            sorted_trios = trio_array
        else:
            trio_order = np.argsort(
                trio_array[:, 0], kind="stable"
            ).astype(np.int64)
            sorted_trios = np.ascontiguousarray(
                trio_array[trio_order], dtype=np.int64
            )
        child_row_starts = np.searchsorted(
            sorted_trios[:, 0],
            np.arange(alleles.shape[0] + 1, dtype=np.int64),
        ).astype(np.int64)
    else:
        trio_order = None
        sorted_trios = trio_array
        child_row_starts = np.zeros(
            alleles.shape[0] + 1, dtype=np.int64
        )
    sorted_two = np.full(len(sorted_trios), -math.inf, dtype=np.float64)
    child_start = 0
    while child_start < alleles.shape[0]:
        applied_threads = (
            _apply_smart_dynamic_threads()
            if _dynamic_rebalance
            else alleles.shape[0]
        )
        child_chunk_size = max(child_chunk_floor, applied_threads)
        if child_chunk_scale:
            child_chunk_size = max(
                child_chunk_size,
                min(64, child_chunk_scale * applied_threads),
            )
        child_end = min(
            child_start + child_chunk_size,
            alleles.shape[0],
        )
        row_start = int(child_row_starts[child_start])
        row_end = int(child_row_starts[child_end])
        sorted_two[row_start:row_end] = (
            _score_two_parent_forward_kernel_grouped(
                alleles,
                sorted_trios,
                child_row_starts,
                background[2],
                background[3],
                effective_transmission,
                effective_phase,
                float(mismatch_probability),
                child_start,
                child_end,
            )
        )
        child_start = child_end
    if trio_order is None:
        two = sorted_two
    else:
        two = np.empty_like(sorted_two)
        two[trio_order] = sorted_two
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    junctions, callable_bins = _ancestry_junction_count_kernel(
        labels, trajectory_classes
    )
    return _ParentStateContigScores(
        zero, one, two, junctions, callable_bins,
        edge_matched_bins=structure_counts[0],
        edge_exposed_bins=structure_counts[1],
        pair_explained_bins=structure_counts[2],
        pair_exposed_bins=structure_counts[3],
        structure_total_bins=float(labels.shape[1]),
    )



@njit(cache=True, parallel=True)
def _gl_information_exponent_kernel(
    genotype_likelihoods,
    selected_markers_per_bin,
    markers_per_information_block,
    effective_markers_per_information_block,
):
    """Temper raw-GL evidence by child, using only nonuniform GL vectors."""
    n_samples, n_bins, _, _ = genotype_likelihoods.shape
    information_block = np.empty(n_bins, dtype=np.int64)
    group = 0
    markers_in_group = 0
    for block in range(n_bins):
        if block > 0 and markers_in_group >= markers_per_information_block:
            group += 1
            markers_in_group = 0
        information_block[block] = group
        markers_in_group += max(int(selected_markers_per_bin[block]), 1)
    n_information_blocks = group + 1

    exponent = np.zeros((n_samples, n_bins), dtype=np.float64)
    for child in prange(n_samples):
        informative_per_group = np.zeros(
            n_information_blocks, dtype=np.int64
        )
        for block in range(n_bins):
            group_index = int(information_block[block])
            for snp in range(int(selected_markers_per_bin[block])):
                gl0 = genotype_likelihoods[child, block, snp, 0]
                gl1 = genotype_likelihoods[child, block, snp, 1]
                gl2 = genotype_likelihoods[child, block, snp, 2]
                if gl0 != gl1 or gl1 != gl2:
                    informative_per_group[group_index] += 1
        for block in range(n_bins):
            informative = informative_per_group[information_block[block]]
            if informative > 0:
                exponent[child, block] = (
                    min(
                        effective_markers_per_information_block,
                        float(informative),
                    )
                    / float(informative)
                )
    return exponent


@njit(cache=True, inline="always")
def _gl_source_alt_probability(
    hard_allele,
    background_alt_probability,
    mismatch_probability,
):
    """Return a source ALT probability after the hard-call error model."""
    if hard_allele < 0:
        source_alt = background_alt_probability
    else:
        source_alt = float(hard_allele)
    return (
        mismatch_probability
        + (1.0 - 2.0 * mismatch_probability) * source_alt
    )


@njit(cache=True, inline="always")
def _gl_diploid_log_emission(
    genotype_likelihoods,
    first_source,
    second_source,
    background_alt_probability,
    marker_count,
    mismatch_probability,
    exponent,
):
    """Log ``3 dot(GL, Q)`` for two independent transmitted alleles.

    ``Q`` is the diploid 0/1/2-ALT genotype distribution implied by the two
    source alleles after mismatch convolution. The factor three makes a
    uniform normalized GL exactly neutral rather than adding a marker-count
    dependent constant to every parent-count model.
    """
    value = 0.0
    for snp in range(marker_count):
        gl0 = genotype_likelihoods[snp, 0]
        gl1 = genotype_likelihoods[snp, 1]
        gl2 = genotype_likelihoods[snp, 2]
        if gl0 == gl1 and gl1 == gl2:
            continue
        first_alt = _gl_source_alt_probability(
            int(first_source[snp]),
            background_alt_probability[snp],
            mismatch_probability,
        )
        second_alt = _gl_source_alt_probability(
            int(second_source[snp]),
            background_alt_probability[snp],
            mismatch_probability,
        )
        first_ref = 1.0 - first_alt
        second_ref = 1.0 - second_alt
        q0 = first_ref * second_ref
        q1 = first_alt * second_ref + first_ref * second_alt
        q2 = first_alt * second_alt
        likelihood = gl0 * q0 + gl1 * q1 + gl2 * q2
        value += math.log(max(3.0 * likelihood, _TINY))
    return exponent * value


@njit(cache=True, inline="always")
def _gl_log_weighted_pair_sum(log_value0, weight0, log_value1, weight1):
    """Log-sum two reachable weighted paths, preserving exact zero edges."""
    first = -math.inf
    second = -math.inf
    if weight0 > 0.0 and math.isfinite(log_value0):
        first = log_value0 + math.log(weight0)
    if weight1 > 0.0 and math.isfinite(log_value1):
        second = log_value1 + math.log(weight1)
    if first == -math.inf:
        return second
    if second == -math.inf:
        return first
    maximum = max(first, second)
    return maximum + math.log(
        math.exp(first - maximum) + math.exp(second - maximum)
    )


@njit(cache=True, inline="always")
def _gl_logsumexp_reachable(log_values, log_probabilities):
    """Sum only finite predecessor paths joined by positive-probability edges."""
    maximum = -math.inf
    for index in range(len(log_values)):
        if (
            math.isfinite(log_values[index])
            and math.isfinite(log_probabilities[index])
        ):
            maximum = max(
                maximum, log_values[index] + log_probabilities[index]
            )
    if maximum == -math.inf:
        return -math.inf
    scale = 0.0
    for index in range(len(log_values)):
        if (
            math.isfinite(log_values[index])
            and math.isfinite(log_probabilities[index])
        ):
            scale += math.exp(
                log_values[index] + log_probabilities[index] - maximum
            )
    return maximum + math.log(scale)


@njit(cache=True, parallel=True)
def _gl_external_log_transition_inplace_kernel(transition_buffer):
    """Convert a now-dead external transition buffer to log space in place."""
    n_children, n_bins, n_states, _ = transition_buffer.shape
    for child in prange(n_children):
        for previous in range(n_states):
            for current in range(n_states):
                transition_buffer[child, 0, previous, current] = -math.inf
    n_boundaries = n_bins - 1
    if n_boundaries <= 0:
        return transition_buffer
    for flat_index in prange(n_children * n_boundaries):
        child = flat_index // n_boundaries
        block = flat_index - child * n_boundaries + 1
        for previous in range(n_states):
            for current in range(n_states):
                probability = transition_buffer[
                    child, block, previous, current
                ]
                transition_buffer[child, block, previous, current] = (
                    math.log(probability) if probability > 0.0 else -math.inf
                )
    return transition_buffer


def _gl_external_log_transition_kernel(external_transition_probability):
    """Return log transitions without mutating a direct helper caller's input."""
    output = np.ascontiguousarray(
        external_transition_probability, dtype=np.float64
    ).copy()
    return _gl_external_log_transition_inplace_kernel(output)


@njit(cache=True, parallel=True)
def _score_zero_parent_gl_forward_kernel(
    genotype_likelihoods,
    founder_alleles,
    selected_markers_per_bin,
    state_probability,
    external_log_transition_probability,
    background_alt_probability,
    information_exponent,
    mismatch_probability,
    eligible_children,
):
    """Log-space M0 forward score for two child-left-out external chains."""
    n_samples, n_bins, _, _ = genotype_likelihoods.shape
    n_states = founder_alleles.shape[0]
    output = np.empty(n_samples, dtype=np.float64)

    for child in prange(n_samples):
        if not eligible_children[child]:
            output[child] = -math.inf
            continue
        informative = False
        for block in range(n_bins):
            informative = informative or information_exponent[child, block] > 0.0
        if not informative:
            output[child] = 0.0
            continue

        forward = np.empty((n_states, n_states), dtype=np.float64)
        work1 = np.empty_like(forward)
        for first in range(n_states):
            for second in range(n_states):
                first_probability = state_probability[child, 0, first]
                second_probability = state_probability[child, 0, second]
                if first_probability > 0.0 and second_probability > 0.0:
                    forward[first, second] = (
                        math.log(first_probability)
                        + math.log(second_probability)
                    )
                else:
                    forward[first, second] = -math.inf
        total = 0.0

        for block in range(n_bins):
            if block > 0:
                log_transition = external_log_transition_probability[child, block]
                for first in range(n_states):
                    for second in range(n_states):
                        work1[first, second] = _gl_logsumexp_reachable(
                            forward[:, second], log_transition[:, first]
                        )
                for first in range(n_states):
                    for second in range(n_states):
                        forward[first, second] = _gl_logsumexp_reachable(
                            work1[first, :], log_transition[:, second]
                        )

            maximum = -math.inf
            for first in range(n_states):
                for second in range(n_states):
                    if math.isfinite(forward[first, second]):
                        forward[first, second] += _gl_diploid_log_emission(
                            genotype_likelihoods[child, block],
                            founder_alleles[first, block],
                            founder_alleles[second, block],
                            background_alt_probability[child, block],
                            int(selected_markers_per_bin[block]),
                            mismatch_probability,
                            information_exponent[child, block],
                        )
                        maximum = max(maximum, forward[first, second])
            if maximum == -math.inf:
                total = -math.inf
                break
            scale = 0.0
            for first in range(n_states):
                for second in range(n_states):
                    if math.isfinite(forward[first, second]):
                        scale += math.exp(forward[first, second] - maximum)
            normalizer = maximum + math.log(scale)
            total += normalizer
            for first in range(n_states):
                for second in range(n_states):
                    if math.isfinite(forward[first, second]):
                        forward[first, second] -= normalizer
        output[child] = total
    return output


@njit(cache=True, parallel=True)
def _score_one_parent_gl_forward_kernel(
    genotype_likelihoods,
    stacked_alleles,
    founder_alleles,
    selected_markers_per_bin,
    state_probability,
    external_log_transition_probability,
    background_alt_probability,
    information_exponent,
    effective_transmission_probability,
    mismatch_probability,
    eligible=None,
    child_start=0,
    child_end=None,
):
    """Log-space M1 forward score for a parent track and external chain."""
    n_samples, n_bins, _, _ = genotype_likelihoods.shape
    n_states = founder_alleles.shape[0]
    if child_end is None:
        child_end = n_samples
    n_children = child_end - child_start
    output = np.full((n_children, n_samples), -math.inf, dtype=np.float64)

    for flat_index in prange(n_children * n_samples):
        local_child = flat_index // n_samples
        child = child_start + local_child
        parent = flat_index - local_child * n_samples
        if (
            parent == child
            or (eligible is not None and not eligible[child, parent])
        ):
            continue
        informative = False
        for block in range(n_bins):
            informative = informative or information_exponent[child, block] > 0.0
        if not informative:
            output[local_child, parent] = 0.0
            continue

        forward = np.empty((2, n_states), dtype=np.float64)
        work1 = np.empty_like(forward)
        for track in range(2):
            for state in range(n_states):
                probability = state_probability[child, 0, state]
                if probability > 0.0:
                    forward[track, state] = math.log(0.5 * probability)
                else:
                    forward[track, state] = -math.inf
        total = 0.0

        for block in range(n_bins):
            if block > 0:
                log_transition = external_log_transition_probability[child, block]
                for track in range(2):
                    for state in range(n_states):
                        work1[track, state] = _gl_logsumexp_reachable(
                            forward[track, :], log_transition[:, state]
                        )
                theta = effective_transmission_probability[parent, block]
                stay = 1.0 - theta
                for state in range(n_states):
                    forward[0, state] = _gl_log_weighted_pair_sum(
                        work1[0, state], stay, work1[1, state], theta
                    )
                    forward[1, state] = _gl_log_weighted_pair_sum(
                        work1[1, state], stay, work1[0, state], theta
                    )

            maximum = -math.inf
            for track in range(2):
                for state in range(n_states):
                    if math.isfinite(forward[track, state]):
                        forward[track, state] += _gl_diploid_log_emission(
                            genotype_likelihoods[child, block],
                            stacked_alleles[parent, block, track],
                            founder_alleles[state, block],
                            background_alt_probability[child, block],
                            int(selected_markers_per_bin[block]),
                            mismatch_probability,
                            information_exponent[child, block],
                        )
                        maximum = max(maximum, forward[track, state])
            if maximum == -math.inf:
                total = -math.inf
                break
            scale = 0.0
            for track in range(2):
                for state in range(n_states):
                    if math.isfinite(forward[track, state]):
                        scale += math.exp(forward[track, state] - maximum)
            normalizer = maximum + math.log(scale)
            total += normalizer
            for track in range(2):
                for state in range(n_states):
                    if math.isfinite(forward[track, state]):
                        forward[track, state] -= normalizer
        output[local_child, parent] = total
    return output


@njit(cache=True, parallel=True)
def _score_two_parent_gl_forward_kernel(
    genotype_likelihoods,
    stacked_alleles,
    selected_markers_per_bin,
    trios,
    background_alt_probability,
    information_exponent,
    effective_transmission_probability,
    mismatch_probability,
    row_start=0,
    row_end=None,
):
    """Log-space M2 forward score using four scalar homolog-track states."""
    n_bins = genotype_likelihoods.shape[1]
    if row_end is None:
        row_end = len(trios)
    n_rows = row_end - row_start
    output = np.empty(n_rows, dtype=np.float64)
    initial = math.log(0.25)

    for local_row in prange(n_rows):
        row = row_start + local_row
        child = int(trios[row, 0])
        parent1 = int(trios[row, 1])
        parent2 = int(trios[row, 2])
        informative = False
        for block in range(n_bins):
            informative = informative or information_exponent[child, block] > 0.0
        if not informative:
            output[local_row] = 0.0
            continue

        forward00 = initial
        forward01 = initial
        forward10 = initial
        forward11 = initial
        total = 0.0

        for block in range(n_bins):
            if block > 0:
                theta1 = effective_transmission_probability[parent1, block]
                theta2 = effective_transmission_probability[parent2, block]
                stay1 = 1.0 - theta1
                stay2 = 1.0 - theta2
                work00 = _gl_log_weighted_pair_sum(
                    forward00, stay1, forward10, theta1
                )
                work10 = _gl_log_weighted_pair_sum(
                    forward10, stay1, forward00, theta1
                )
                work01 = _gl_log_weighted_pair_sum(
                    forward01, stay1, forward11, theta1
                )
                work11 = _gl_log_weighted_pair_sum(
                    forward11, stay1, forward01, theta1
                )
                forward00 = _gl_log_weighted_pair_sum(
                    work00, stay2, work01, theta2
                )
                forward01 = _gl_log_weighted_pair_sum(
                    work01, stay2, work00, theta2
                )
                forward10 = _gl_log_weighted_pair_sum(
                    work10, stay2, work11, theta2
                )
                forward11 = _gl_log_weighted_pair_sum(
                    work11, stay2, work10, theta2
                )

            maximum = -math.inf
            if math.isfinite(forward00):
                forward00 += _gl_diploid_log_emission(
                    genotype_likelihoods[child, block],
                    stacked_alleles[parent1, block, 0],
                    stacked_alleles[parent2, block, 0],
                    background_alt_probability[child, block],
                    int(selected_markers_per_bin[block]),
                    mismatch_probability,
                    information_exponent[child, block],
                )
                maximum = max(maximum, forward00)
            if math.isfinite(forward01):
                forward01 += _gl_diploid_log_emission(
                    genotype_likelihoods[child, block],
                    stacked_alleles[parent1, block, 0],
                    stacked_alleles[parent2, block, 1],
                    background_alt_probability[child, block],
                    int(selected_markers_per_bin[block]),
                    mismatch_probability,
                    information_exponent[child, block],
                )
                maximum = max(maximum, forward01)
            if math.isfinite(forward10):
                forward10 += _gl_diploid_log_emission(
                    genotype_likelihoods[child, block],
                    stacked_alleles[parent1, block, 1],
                    stacked_alleles[parent2, block, 0],
                    background_alt_probability[child, block],
                    int(selected_markers_per_bin[block]),
                    mismatch_probability,
                    information_exponent[child, block],
                )
                maximum = max(maximum, forward10)
            if math.isfinite(forward11):
                forward11 += _gl_diploid_log_emission(
                    genotype_likelihoods[child, block],
                    stacked_alleles[parent1, block, 1],
                    stacked_alleles[parent2, block, 1],
                    background_alt_probability[child, block],
                    int(selected_markers_per_bin[block]),
                    mismatch_probability,
                    information_exponent[child, block],
                )
                maximum = max(maximum, forward11)
            if maximum == -math.inf:
                total = -math.inf
                break
            scale = 0.0
            if math.isfinite(forward00):
                scale += math.exp(forward00 - maximum)
            if math.isfinite(forward01):
                scale += math.exp(forward01 - maximum)
            if math.isfinite(forward10):
                scale += math.exp(forward10 - maximum)
            if math.isfinite(forward11):
                scale += math.exp(forward11 - maximum)
            normalizer = maximum + math.log(scale)
            total += normalizer
            if math.isfinite(forward00):
                forward00 -= normalizer
            if math.isfinite(forward01):
                forward01 -= normalizer
            if math.isfinite(forward10):
                forward10 -= normalizer
            if math.isfinite(forward11):
                forward11 -= normalizer
        output[local_row] = total
    return output


@njit(cache=True, parallel=True)
def _gl_candidate_identity_information_kernel(
    genotype_likelihoods,
    stacked_alleles,
    selected_markers_per_bin,
    information_exponent,
    eligible_parents,
):
    """Measure candidate calls at child-informative selected markers."""
    n_samples, n_bins, _, _ = genotype_likelihoods.shape
    information = np.zeros((n_samples, n_samples), dtype=np.float64)
    fully_called = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for flat_index in prange(n_samples * n_samples):
        child = flat_index // n_samples
        parent = flat_index - child * n_samples
        if not eligible_parents[child, parent]:
            continue
        value = 0.0
        complete = True
        for block in range(n_bins):
            exponent = information_exponent[child, block]
            for snp in range(int(selected_markers_per_bin[block])):
                gl0 = genotype_likelihoods[child, block, snp, 0]
                gl1 = genotype_likelihoods[child, block, snp, 1]
                gl2 = genotype_likelihoods[child, block, snp, 2]
                if gl0 == gl1 and gl1 == gl2:
                    continue
                first_called = stacked_alleles[parent, block, 0, snp] >= 0
                second_called = stacked_alleles[parent, block, 1, snp] >= 0
                if first_called or second_called:
                    value += exponent
                if not first_called or not second_called:
                    complete = False
        information[child, parent] = value
        fully_called[child, parent] = complete
    return information, fully_called


@njit(cache=True, inline="always")
def _gl_one_parent_linked_log_emission(
    genotype_likelihoods,
    candidate_source,
    candidate_fallback,
    external_source,
    background_alt_probability,
    marker_count,
    mismatch_probability,
    exponent,
):
    """Raw-GL emission with a linked founder fallback for candidate gaps."""
    value = 0.0
    for snp in range(marker_count):
        gl0 = genotype_likelihoods[snp, 0]
        gl1 = genotype_likelihoods[snp, 1]
        gl2 = genotype_likelihoods[snp, 2]
        if gl0 == gl1 and gl1 == gl2:
            continue
        candidate_allele = int(candidate_source[snp])
        if candidate_allele < 0:
            candidate_allele = int(candidate_fallback[snp])
        first_alt = _gl_source_alt_probability(
            candidate_allele,
            background_alt_probability[snp],
            mismatch_probability,
        )
        second_alt = _gl_source_alt_probability(
            int(external_source[snp]),
            background_alt_probability[snp],
            mismatch_probability,
        )
        first_ref = 1.0 - first_alt
        second_ref = 1.0 - second_alt
        q0 = first_ref * second_ref
        q1 = first_alt * second_ref + first_ref * second_alt
        q2 = first_alt * second_alt
        likelihood = gl0 * q0 + gl1 * q1 + gl2 * q2
        value += math.log(max(3.0 * likelihood, _TINY))
    return exponent * value


@njit(cache=True, inline="always")
def _gl_two_parent_linked_log_emission(
    genotype_likelihoods,
    first_candidate_source,
    first_candidate_fallback,
    second_candidate_source,
    second_candidate_fallback,
    background_alt_probability,
    marker_count,
    mismatch_probability,
    exponent,
):
    """Raw-GL emission with an independent linked fallback per candidate."""
    value = 0.0
    for snp in range(marker_count):
        gl0 = genotype_likelihoods[snp, 0]
        gl1 = genotype_likelihoods[snp, 1]
        gl2 = genotype_likelihoods[snp, 2]
        if gl0 == gl1 and gl1 == gl2:
            continue
        first_allele = int(first_candidate_source[snp])
        if first_allele < 0:
            first_allele = int(first_candidate_fallback[snp])
        second_allele = int(second_candidate_source[snp])
        if second_allele < 0:
            second_allele = int(second_candidate_fallback[snp])
        first_alt = _gl_source_alt_probability(
            first_allele,
            background_alt_probability[snp],
            mismatch_probability,
        )
        second_alt = _gl_source_alt_probability(
            second_allele,
            background_alt_probability[snp],
            mismatch_probability,
        )
        first_ref = 1.0 - first_alt
        second_ref = 1.0 - second_alt
        q0 = first_ref * second_ref
        q1 = first_alt * second_ref + first_ref * second_alt
        q2 = first_alt * second_alt
        likelihood = gl0 * q0 + gl1 * q1 + gl2 * q2
        value += math.log(max(3.0 * likelihood, _TINY))
    return exponent * value


@njit(cache=True, parallel=True)
def _score_one_parent_gl_linked_fallback_kernel(
    genotype_likelihoods,
    stacked_alleles,
    founder_alleles,
    selected_markers_per_bin,
    state_probability,
    external_log_transition_probability,
    background_alt_probability,
    information_exponent,
    effective_transmission_probability,
    identity_information,
    fully_called,
    mismatch_probability,
    eligible_parents,
    child_start=0,
    child_end=None,
):
    """Score partial M1 states (parent track, fallback, external source)."""
    n_samples, n_bins, _, _ = genotype_likelihoods.shape
    n_states = founder_alleles.shape[0]
    if child_end is None:
        child_end = n_samples
    n_children = child_end - child_start
    output = np.full((n_children, n_samples), -math.inf, dtype=np.float64)

    for flat_index in prange(n_children * n_samples):
        local_child = flat_index // n_samples
        child = child_start + local_child
        parent = flat_index - local_child * n_samples
        if (
            parent == child
            or not eligible_parents[child, parent]
            or identity_information[child, parent] <= 0.0
            or fully_called[child, parent]
        ):
            continue

        forward = np.empty(
            (2, n_states, n_states), dtype=np.float64
        )
        work1 = np.empty_like(forward)
        for track in range(2):
            for fallback in range(n_states):
                fallback_probability = state_probability[
                    child, 0, fallback
                ]
                for external in range(n_states):
                    external_probability = state_probability[
                        child, 0, external
                    ]
                    if (
                        fallback_probability > 0.0
                        and external_probability > 0.0
                    ):
                        forward[track, fallback, external] = (
                            math.log(0.5)
                            + math.log(fallback_probability)
                            + math.log(external_probability)
                        )
                    else:
                        forward[track, fallback, external] = -math.inf
        total = 0.0

        for block in range(n_bins):
            if block > 0:
                log_transition = external_log_transition_probability[child, block]
                for track in range(2):
                    for fallback in range(n_states):
                        for external in range(n_states):
                            work1[track, fallback, external] = (
                                _gl_logsumexp_reachable(
                                    forward[track, :, external],
                                    log_transition[:, fallback],
                                )
                            )
                for track in range(2):
                    for fallback in range(n_states):
                        for external in range(n_states):
                            forward[track, fallback, external] = (
                                _gl_logsumexp_reachable(
                                    work1[track, fallback, :],
                                    log_transition[:, external],
                                )
                            )
                theta = effective_transmission_probability[parent, block]
                stay = 1.0 - theta
                for fallback in range(n_states):
                    for external in range(n_states):
                        value0 = forward[0, fallback, external]
                        value1 = forward[1, fallback, external]
                        forward[0, fallback, external] = (
                            _gl_log_weighted_pair_sum(
                                value0,
                                stay,
                                value1,
                                theta,
                            )
                        )
                        forward[1, fallback, external] = (
                            _gl_log_weighted_pair_sum(
                                value1,
                                stay,
                                value0,
                                theta,
                            )
                        )

            maximum = -math.inf
            for track in range(2):
                for fallback in range(n_states):
                    for external in range(n_states):
                        if math.isfinite(
                            forward[track, fallback, external]
                        ):
                            forward[track, fallback, external] += (
                                _gl_one_parent_linked_log_emission(
                                    genotype_likelihoods[child, block],
                                    stacked_alleles[
                                        parent, block, track
                                    ],
                                    founder_alleles[fallback, block],
                                    founder_alleles[external, block],
                                    background_alt_probability[
                                        child, block
                                    ],
                                    int(
                                        selected_markers_per_bin[block]
                                    ),
                                    mismatch_probability,
                                    information_exponent[child, block],
                                )
                            )
                            maximum = max(
                                maximum,
                                forward[track, fallback, external],
                            )
            if maximum == -math.inf:
                total = -math.inf
                break
            scale = 0.0
            for track in range(2):
                for fallback in range(n_states):
                    for external in range(n_states):
                        if math.isfinite(
                            forward[track, fallback, external]
                        ):
                            scale += math.exp(
                                forward[track, fallback, external]
                                - maximum
                            )
            normalizer = maximum + math.log(scale)
            total += normalizer
            for track in range(2):
                for fallback in range(n_states):
                    for external in range(n_states):
                        if math.isfinite(
                            forward[track, fallback, external]
                        ):
                            forward[
                                track, fallback, external
                            ] -= normalizer
        output[local_child, parent] = total
    return output


@njit(cache=True, parallel=True)
def _score_two_parent_gl_linked_fallback_kernel(
    genotype_likelihoods,
    stacked_alleles,
    founder_alleles,
    selected_markers_per_bin,
    trios,
    state_probability,
    external_log_transition_probability,
    background_alt_probability,
    information_exponent,
    effective_transmission_probability,
    mismatch_probability,
    row_start=0,
    row_end=None,
):
    """Score general partial M2 states without a Kronecker transition."""
    n_bins = genotype_likelihoods.shape[1]
    n_states = founder_alleles.shape[0]
    if row_end is None:
        row_end = len(trios)
    n_rows = row_end - row_start
    output = np.empty(n_rows, dtype=np.float64)

    for local_row in prange(n_rows):
        row = row_start + local_row
        child = int(trios[row, 0])
        parent1 = int(trios[row, 1])
        parent2 = int(trios[row, 2])
        forward = np.empty(
            (2, n_states, 2, n_states), dtype=np.float64
        )
        work1 = np.empty_like(forward)
        for track1 in range(2):
            for fallback1 in range(n_states):
                probability1 = state_probability[child, 0, fallback1]
                for track2 in range(2):
                    for fallback2 in range(n_states):
                        probability2 = state_probability[
                            child, 0, fallback2
                        ]
                        if probability1 > 0.0 and probability2 > 0.0:
                            forward[
                                track1, fallback1, track2, fallback2
                            ] = (
                                math.log(0.25)
                                + math.log(probability1)
                                + math.log(probability2)
                            )
                        else:
                            forward[
                                track1, fallback1, track2, fallback2
                            ] = -math.inf
        total = 0.0

        for block in range(n_bins):
            if block > 0:
                log_transition = external_log_transition_probability[child, block]
                for track1 in range(2):
                    for fallback1 in range(n_states):
                        for track2 in range(2):
                            for fallback2 in range(n_states):
                                work1[
                                    track1,
                                    fallback1,
                                    track2,
                                    fallback2,
                                ] = _gl_logsumexp_reachable(
                                    forward[
                                        track1, :, track2, fallback2
                                    ],
                                    log_transition[:, fallback1],
                                )
                theta1 = effective_transmission_probability[
                    parent1, block
                ]
                stay1 = 1.0 - theta1
                for fallback1 in range(n_states):
                    for track2 in range(2):
                        for fallback2 in range(n_states):
                            forward[
                                0, fallback1, track2, fallback2
                            ] = _gl_log_weighted_pair_sum(
                                work1[
                                    0, fallback1, track2, fallback2
                                ],
                                stay1,
                                work1[
                                    1, fallback1, track2, fallback2
                                ],
                                theta1,
                            )
                            forward[
                                1, fallback1, track2, fallback2
                            ] = _gl_log_weighted_pair_sum(
                                work1[
                                    1, fallback1, track2, fallback2
                                ],
                                stay1,
                                work1[
                                    0, fallback1, track2, fallback2
                                ],
                                theta1,
                            )
                for track1 in range(2):
                    for fallback1 in range(n_states):
                        for track2 in range(2):
                            for fallback2 in range(n_states):
                                work1[
                                    track1,
                                    fallback1,
                                    track2,
                                    fallback2,
                                ] = _gl_logsumexp_reachable(
                                    forward[
                                        track1, fallback1, track2, :
                                    ],
                                    log_transition[:, fallback2],
                                )
                theta2 = effective_transmission_probability[
                    parent2, block
                ]
                stay2 = 1.0 - theta2
                for track1 in range(2):
                    for fallback1 in range(n_states):
                        for fallback2 in range(n_states):
                            forward[
                                track1, fallback1, 0, fallback2
                            ] = _gl_log_weighted_pair_sum(
                                work1[
                                    track1, fallback1, 0, fallback2
                                ],
                                stay2,
                                work1[
                                    track1, fallback1, 1, fallback2
                                ],
                                theta2,
                            )
                            forward[
                                track1, fallback1, 1, fallback2
                            ] = _gl_log_weighted_pair_sum(
                                work1[
                                    track1, fallback1, 1, fallback2
                                ],
                                stay2,
                                work1[
                                    track1, fallback1, 0, fallback2
                                ],
                                theta2,
                            )

            maximum = -math.inf
            for track1 in range(2):
                for fallback1 in range(n_states):
                    for track2 in range(2):
                        for fallback2 in range(n_states):
                            if math.isfinite(
                                forward[
                                    track1,
                                    fallback1,
                                    track2,
                                    fallback2,
                                ]
                            ):
                                forward[
                                    track1,
                                    fallback1,
                                    track2,
                                    fallback2,
                                ] += _gl_two_parent_linked_log_emission(
                                    genotype_likelihoods[child, block],
                                    stacked_alleles[
                                        parent1, block, track1
                                    ],
                                    founder_alleles[fallback1, block],
                                    stacked_alleles[
                                        parent2, block, track2
                                    ],
                                    founder_alleles[fallback2, block],
                                    background_alt_probability[
                                        child, block
                                    ],
                                    int(
                                        selected_markers_per_bin[block]
                                    ),
                                    mismatch_probability,
                                    information_exponent[child, block],
                                )
                                maximum = max(
                                    maximum,
                                    forward[
                                        track1,
                                        fallback1,
                                        track2,
                                        fallback2,
                                    ],
                                )
            if maximum == -math.inf:
                total = -math.inf
                break
            scale = 0.0
            for track1 in range(2):
                for fallback1 in range(n_states):
                    for track2 in range(2):
                        for fallback2 in range(n_states):
                            if math.isfinite(
                                forward[
                                    track1,
                                    fallback1,
                                    track2,
                                    fallback2,
                                ]
                            ):
                                scale += math.exp(
                                    forward[
                                        track1,
                                        fallback1,
                                        track2,
                                        fallback2,
                                    ]
                                    - maximum
                                )
            normalizer = maximum + math.log(scale)
            total += normalizer
            for track1 in range(2):
                for fallback1 in range(n_states):
                    for track2 in range(2):
                        for fallback2 in range(n_states):
                            if math.isfinite(
                                forward[
                                    track1,
                                    fallback1,
                                    track2,
                                    fallback2,
                                ]
                            ):
                                forward[
                                    track1,
                                    fallback1,
                                    track2,
                                    fallback2,
                                ] -= normalizer
        output[local_row] = total
    return output


def score_parent_state_gl_hmms(
    genotype_likelihoods,
    stacked_alleles,
    stacked_labels,
    stacked_hom_mask,
    founder_alleles,
    selected_markers_per_bin,
    switch_probability,
    trios,
    *,
    mismatch_probability=0.01,
    phase_switch_probability=0.01,
    markers_per_information_block=100,
    effective_markers_per_information_block=1.0,
    external_state_pseudocount=1.0,
    external_transition_pseudocount=20.0,
    candidate_source_mode="hard_painted",
    candidate_source_path_switch_probability=None,
    _precomputed_candidate_source=None,
    _dynamic_rebalance=False,
    _eligible_children=None,
    _eligible_parents=None,
    _dynamic_child_chunk_floor=32,
    _dynamic_child_chunk_scale=4,
):
    """Score M0/M1/M2 forward models from unphased child genotype GLs.

    ``genotype_likelihoods`` has shape ``(samples, bins, SNP slots, 3)`` in
    0/0, 0/1, 1/1 order. Each finite, non-negative linear relative-likelihood
    vector must already sum to one. Given source ALT probabilities ``p1`` and
    ``p2`` after mismatch convolution, its diploid emission is
    ``log(3 * dot(GL, Q(p1, p2)))``. Thus exactly uniform GL is neutral and
    returns zero evidence in every parent-count model.

    M0 follows two external local-IBS chains, M1 one candidate-parent homolog
    track plus one external chain, and M2 two candidate-parent homolog tracks.
    The external distributions and transitions are estimated from the same
    child-left-out painted population as :func:`score_parent_state_hmms`.
    Candidate-parent hard sources are error-convolved. At a candidate ``-1``,
    the source delegates to that candidate's own child-left-out founder-state
    fallback chain. The fallback evolves through called as well as missing
    bins, preserving linkage across a missing-called-missing interval. Fully
    called candidates retain the reduced fast state spaces.

    Identity information sums the shared child information exponent over
    nonuniform selected-marker GLs where either candidate homolog is called.
    It is diagnostic only. A candidate with zero identity information is
    canonically external: M1 equals M0 exactly; M2 with one such candidate
    equals the other candidate's M1, and M2 with two equals M0. Information
    tempering remains shared by every hidden state and candidate for a child
    and is never divided by the identity-information diagnostic.

    The child's hard alleles, homolog orientation and homozygosity mask are
    never observations or hidden states in this scorer. Accordingly
    ``phase_switch_probability`` is accepted and validated for scientific-
    tuning parity with the hard-call API but cannot affect an unphased GL
    score. Painted labels are still returned through the ancestry-depth
    diagnostics and are used only to construct algebraically child-left-out
    external-chain distributions.

    ``matched_null_raw_gl_v2`` instead requires an explicit adult mosaic
    ``candidate_source_path_switch_probability``. It uses an
    ordered-independent candidate root and two matched synthetic null parents.
    ``switch_probability`` remains the offspring transmission theta shared by
    candidate and null parents; painted hard-homo resets are not applied.
    Candidate-source normalizers are excluded from all parent-state scores.
    """
    raw_alleles = np.asarray(stacked_alleles)
    if raw_alleles.ndim == 3:
        raw_alleles = raw_alleles[..., None]
    if (
        raw_alleles.ndim != 4
        or raw_alleles.shape[2] != 2
        or min(raw_alleles.shape) < 1
        or np.any(~np.isin(raw_alleles, (-1, 0, 1)))
    ):
        raise SmartEvidenceError(
            "stacked_alleles must have exactly two homologs and contain "
            "non-empty hard alleles in {-1, 0, 1}"
        )
    alleles = np.ascontiguousarray(raw_alleles, dtype=np.int8)

    raw_gl = np.asarray(genotype_likelihoods)
    if raw_gl.shape != alleles.shape[:2] + (alleles.shape[3], 3):
        raise SmartEvidenceError(
            "genotype_likelihoods must have shape "
            "(samples, bins, SNP slots, 3) matching stacked_alleles"
        )
    try:
        gl = np.ascontiguousarray(raw_gl, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SmartEvidenceError(
            "genotype_likelihoods must be finite normalized linear values"
        ) from exc
    gl_totals = np.sum(gl, axis=3)
    if (
        np.any(~np.isfinite(gl))
        or np.any(gl < 0.0)
        or np.any(~np.isclose(gl_totals, 1.0, rtol=0.0, atol=1e-12))
    ):
        raise SmartEvidenceError(
            "every genotype-likelihood vector must contain finite, "
            "non-negative linear values summing to one"
        )

    raw_founders = np.asarray(founder_alleles)
    if raw_founders.ndim == 2:
        raw_founders = raw_founders[..., None]
    if (
        raw_founders.ndim != 3
        or min(raw_founders.shape) < 1
        or np.any(~np.isin(raw_founders, (-1, 0, 1)))
    ):
        raise SmartEvidenceError(
            "founder_alleles must be non-empty hard alleles in {-1, 0, 1}"
        )
    founders = np.ascontiguousarray(raw_founders, dtype=np.int8)
    physical_founders = founders.copy()
    if founders.shape[1:] != (alleles.shape[1], alleles.shape[3]):
        raise SmartEvidenceError(
            "founder alleles must have shape (states, bins, SNPs)"
        )

    raw_labels = np.asarray(stacked_labels)
    if (
        raw_labels.ndim != 3
        or raw_labels.shape[2] != 2
        or raw_labels.shape != alleles.shape[:3]
        or np.any(~np.isfinite(raw_labels))
        or np.any(raw_labels != np.floor(raw_labels))
        or np.any(raw_labels < -1)
        or np.any(raw_labels >= founders.shape[0])
    ):
        raise SmartEvidenceError(
            "stacked_labels must have shape (samples, bins, 2) and values "
            "in {-1, 0, ..., states-1}"
        )
    labels = np.ascontiguousarray(raw_labels, dtype=np.int16)
    physical_labels = labels.copy()
    raw_hom = np.asarray(stacked_hom_mask)
    if (
        raw_hom.shape != alleles.shape[:2]
        or raw_hom.dtype != np.dtype(np.bool_)
    ):
        raise SmartEvidenceError(
            "stacked_hom_mask must be a boolean array with shape "
            "(samples, bins)"
        )
    hom = np.ascontiguousarray(raw_hom, dtype=np.bool_)

    raw_marker_counts = np.asarray(selected_markers_per_bin)
    if (
        raw_marker_counts.shape != (alleles.shape[1],)
        or np.any(~np.isfinite(raw_marker_counts))
        or np.any(raw_marker_counts != np.floor(raw_marker_counts))
        or np.any(raw_marker_counts < 0)
        or np.any(raw_marker_counts > alleles.shape[3])
        or np.sum(raw_marker_counts) < 1
    ):
        raise SmartEvidenceError(
            "selected_markers_per_bin must be integer counts between zero and "
            "the SNP-slot count, with at least one selected marker"
        )
    marker_counts = np.ascontiguousarray(raw_marker_counts, dtype=np.int64)
    founders = founders.copy()
    for block, marker_count in enumerate(marker_counts):
        founders[:, block, int(marker_count):] = -1
        physical_founders[:, block, int(marker_count):] = -1
    theta = np.ascontiguousarray(switch_probability, dtype=np.float64)
    if (
        theta.shape != (alleles.shape[1],)
        or np.any(~np.isfinite(theta))
        or np.any((theta < 0.0) | (theta > 0.5))
    ):
        raise SmartEvidenceError(
            "switch probabilities must be finite with shape (bins,) and lie "
            "in [0, 0.5]"
        )

    raw_trios = np.asarray(trios)
    if (
        raw_trios.ndim != 2
        or raw_trios.shape[1] != 3
        or np.any(~np.isfinite(raw_trios))
        or np.any(raw_trios != np.floor(raw_trios))
    ):
        raise SmartEvidenceError("trios must be an integer array of shape (rows, 3)")
    trio_array = np.ascontiguousarray(raw_trios, dtype=np.int64)
    if len(trio_array):
        if np.any(trio_array < 0) or np.any(trio_array >= alleles.shape[0]):
            raise SmartEvidenceError("trio index outside the sample array")
        if (
            np.any(trio_array[:, 0] == trio_array[:, 1])
            or np.any(trio_array[:, 0] == trio_array[:, 2])
            or np.any(trio_array[:, 1] == trio_array[:, 2])
        ):
            raise SmartEvidenceError(
                "trios cannot contain self-parents or duplicate parents"
            )

    eligible_children, eligible_parents = _scoring_eligibility_masks(
        alleles.shape[0], _eligible_children, _eligible_parents
    )
    required_edges = np.ascontiguousarray(
        eligible_parents | eligible_parents.T
    )
    np.fill_diagonal(required_edges, True)
    identity_required = eligible_parents.copy()
    np.fill_diagonal(identity_required, True)
    if candidate_source_mode not in {
        "hard_painted", "exact_raw_gl_v1", "matched_null_raw_gl_v2"
    }:
        raise SmartEvidenceError(
            "candidate_source_mode must be 'hard_painted', "
            "'exact_raw_gl_v1', or 'matched_null_raw_gl_v2'"
        )
    if candidate_source_mode == "matched_null_raw_gl_v2":
        source_path_switch = candidate_source_path_switch_probability
        try:
            source_path_switch_value = float(source_path_switch)
        except (TypeError, ValueError, OverflowError):
            source_path_switch_value = np.nan
        if (
            isinstance(source_path_switch, (bool, np.bool_))
            or not np.isfinite(source_path_switch_value)
            or not 0.0 <= source_path_switch_value <= 0.5
        ):
            raise SmartEvidenceError(
                "matched_null_raw_gl_v2 requires explicit "
                "candidate_source_path_switch_probability in [0, 0.5]"
            )
    if (
        not np.isfinite(mismatch_probability)
        or not 0.0 < mismatch_probability < 0.5
    ):
        raise SmartEvidenceError("mismatch_probability must lie in (0, 0.5)")
    if (
        not np.isfinite(phase_switch_probability)
        or not 0.0 <= phase_switch_probability <= 0.5
    ):
        raise SmartEvidenceError("phase_switch_probability must lie in [0, 0.5]")
    try:
        marker_block_size = float(markers_per_information_block)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SmartEvidenceError(
            "markers_per_information_block must be a positive integer"
        ) from exc
    if (
        isinstance(markers_per_information_block, (bool, np.bool_))
        or not np.isfinite(marker_block_size)
        or marker_block_size != math.floor(marker_block_size)
        or marker_block_size < 1
    ):
        raise SmartEvidenceError(
            "markers_per_information_block must be a positive integer"
        )
    for name, value in (
        (
            "effective_markers_per_information_block",
            effective_markers_per_information_block,
        ),
        ("external_state_pseudocount", external_state_pseudocount),
        ("external_transition_pseudocount", external_transition_pseudocount),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise SmartEvidenceError(f"{name} must be finite and positive")
    try:
        child_chunk_floor = operator.index(_dynamic_child_chunk_floor)
    except TypeError as exc:
        raise SmartEvidenceError(
            "_dynamic_child_chunk_floor must be a positive integer"
        ) from exc
    if child_chunk_floor < 1:
        raise SmartEvidenceError(
            "_dynamic_child_chunk_floor must be a positive integer"
        )
    try:
        child_chunk_scale = operator.index(_dynamic_child_chunk_scale)
    except TypeError as exc:
        raise SmartEvidenceError(
            "_dynamic_child_chunk_scale must be a non-negative integer"
        ) from exc
    if child_chunk_scale < 0:
        raise SmartEvidenceError(
            "_dynamic_child_chunk_scale must be a non-negative integer"
        )

    (
        labels,
        founders,
        active_states,
        continuation_bridge,
        trajectory_classes,
    ) = _pool_local_ibs_states(labels, founders)
    structure_counts = _parenthood_structure_count_kernel(
        labels, trio_array, required_edges
    )
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    totals = _background_totals(alleles, labels, founders.shape[0])
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    background = _prepare_leave_child_out_background(
        alleles,
        labels,
        marker_counts,
        active_states,
        continuation_bridge,
        theta,
        *totals,
        float(external_state_pseudocount),
        float(external_transition_pseudocount),
        int(marker_block_size),
        float(effective_markers_per_information_block),
    )
    information_exponent = _gl_information_exponent_kernel(
        gl,
        marker_counts,
        int(marker_block_size),
        float(effective_markers_per_information_block),
    )
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    (
        identity_information,
        fully_called_identity,
    ) = _gl_candidate_identity_information_kernel(
        gl,
        alleles,
        marker_counts,
        information_exponent,
        identity_required,
    )
    effective_transmission, _ = _effective_switch_probability_kernel(
        hom,
        theta,
        float(phase_switch_probability),
    )
    real_site = (
        np.arange(alleles.shape[3])[None, :] < marker_counts[:, None]
    )
    exact_source_requested = candidate_source_mode in {
        "exact_raw_gl_v1",
        "matched_null_raw_gl_v2",
    }
    if exact_source_requested:
        exact_founders, exact_labels, _ = (
            _deduplicate_global_founder_trajectories(
                physical_founders, physical_labels
            )
        )
        completeness_panel = exact_founders
    else:
        completeness_panel = physical_founders
    complete_founder_site = (
        np.all(completeness_panel >= 0, axis=0) & real_site
    )
    complete_founder_marker_count = int(np.sum(complete_founder_site))
    excluded_founder_marker_count = int(
        np.sum(real_site) - complete_founder_marker_count
    )
    exact_source_fallback = bool(
        exact_source_requested and excluded_founder_marker_count > 0
    )
    if exact_source_requested and not exact_source_fallback:
        candidate_source = _precomputed_candidate_source
        if candidate_source is None:
            source_path_switch = (
                theta
                if candidate_source_mode == "exact_raw_gl_v1"
                else float(candidate_source_path_switch_probability)
            )
            candidate_source = infer_candidate_source_posterior(
                gl,
                exact_founders,
                marker_counts,
                source_path_switch,
                eta=np.where(
                    information_exponent > 0.0,
                    information_exponent,
                    1.0,
                ),
                painted_track_labels=(
                    exact_labels
                    if candidate_source_mode == "exact_raw_gl_v1"
                    else None
                ),
                return_lumped_posterior=True,
                lumped_root_prior_mode=(
                    "uniform_unordered"
                    if candidate_source_mode == "exact_raw_gl_v1"
                    else "ordered_independent_uniform"
                ),
            )
        if candidate_source_mode == "exact_raw_gl_v1":
            external_initial, external_transition = (
                _lift_pooled_external_chains_to_physical_founders(
                    background[0], background[1], exact_founders
                )
            )
            batch = score_candidate_source_batch_exact(
                candidate_source,
                gl,
                exact_founders,
                marker_counts,
                information_exponent,
                effective_transmission[:, 1:],
                external_initial,
                external_transition,
                trio_array,
                mismatch_probability=float(mismatch_probability),
            )
        else:
            batch = score_candidate_source_batch_matched_null_exact(
                candidate_source,
                gl,
                exact_founders,
                marker_counts,
                information_exponent,
                theta,
                trio_array,
                mismatch_probability=float(mismatch_probability),
            )
        exact_one = np.asarray(batch.one_observed, dtype=np.float64).copy()
        np.fill_diagonal(exact_one, -math.inf)
        if _dynamic_rebalance:
            _apply_smart_dynamic_threads()
        junctions, callable_bins = _ancestry_junction_count_kernel(
            labels, trajectory_classes
        )
        return _ParentStateContigScores(
            np.asarray(batch.zero_observed, dtype=np.float64),
            exact_one,
            np.asarray(batch.two_observed, dtype=np.float64),
            junctions,
            callable_bins,
            np.asarray(
                batch.one_parent_identity_information, dtype=np.float64
            ),
            np.asarray(batch.two_parent_edge_information, dtype=np.float64),
            edge_matched_bins=structure_counts[0],
            edge_exposed_bins=structure_counts[1],
            pair_explained_bins=structure_counts[2],
            pair_exposed_bins=structure_counts[3],
            structure_total_bins=float(labels.shape[1]),
            candidate_source_mode_requested=candidate_source_mode,
            candidate_source_mode_applied=candidate_source_mode,
            complete_founder_marker_count=(
                batch.complete_founder_marker_count
            ),
            excluded_founder_marker_count=(
                batch.excluded_founder_marker_count
            ),
            candidate_source_available=(
                batch.candidate_source_available.copy()
            ),
            candidate_source_informative_marker_count=(
                batch.candidate_source_informative_marker_count.copy()
            ),
            child_complete_informative_marker_count=(
                batch.child_complete_informative_marker_count.copy()
            ),
            candidate_initial_max_probability=(
                batch.candidate_initial_max_probability.copy()
            ),
            candidate_initial_point_mass=(
                batch.candidate_initial_point_mass.copy()
            ),
            peak_streamed_tensor_bytes=int(batch.peak_streamed_tensor_bytes),
            candidate_source_posterior=candidate_source,
        )
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    external_log_transition = _gl_external_log_transition_inplace_kernel(
        background[1]
    )
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    zero = _score_zero_parent_gl_forward_kernel(
        gl,
        founders,
        marker_counts,
        background[0],
        external_log_transition,
        background[2],
        information_exponent,
        float(mismatch_probability),
        eligible_children,
    )
    fast_one_identity = (
        fully_called_identity & (identity_information > 0.0)
    )
    fast_one_eligible = fast_one_identity & eligible_parents
    one = np.full(
        (alleles.shape[0], alleles.shape[0]), -math.inf, dtype=np.float64
    )
    child_start = 0
    while child_start < alleles.shape[0]:
        if _dynamic_rebalance:
            applied_threads = _apply_smart_dynamic_threads()
            child_chunk_size = max(child_chunk_floor, applied_threads)
            if child_chunk_scale:
                child_chunk_size = max(
                    child_chunk_size,
                    min(64, child_chunk_scale * applied_threads),
                )
        else:
            child_chunk_size = alleles.shape[0]
        child_end = min(child_start + child_chunk_size, alleles.shape[0])
        one[child_start:child_end] = _score_one_parent_gl_forward_kernel(
            gl,
            alleles,
            founders,
            marker_counts,
            background[0],
            external_log_transition,
            background[2],
            information_exponent,
            effective_transmission,
            float(mismatch_probability),
            fast_one_eligible,
            child_start,
            child_end,
        )
        child_start = child_end

    partial_identity = (
        (identity_information > 0.0) & ~fully_called_identity & eligible_parents
    )
    if np.any(partial_identity):
        child_start = 0
        while child_start < alleles.shape[0]:
            if _dynamic_rebalance:
                applied_threads = _apply_smart_dynamic_threads()
                child_chunk_size = max(child_chunk_floor, applied_threads)
                if child_chunk_scale:
                    child_chunk_size = max(
                        child_chunk_size,
                        min(64, child_chunk_scale * applied_threads),
                    )
            else:
                child_chunk_size = alleles.shape[0]
            child_end = min(
                child_start + child_chunk_size, alleles.shape[0]
            )
            partial_scores = _score_one_parent_gl_linked_fallback_kernel(
                gl,
                alleles,
                founders,
                marker_counts,
                background[0],
                external_log_transition,
                background[2],
                information_exponent,
                effective_transmission,
                identity_information,
                fully_called_identity,
                float(mismatch_probability),
                eligible_parents,
                child_start,
                child_end,
            )
            chunk_mask = partial_identity[child_start:child_end]
            target = one[child_start:child_end]
            target[chunk_mask] = partial_scores[chunk_mask]
            child_start = child_end
    zero_identity = (identity_information <= 0.0) & eligible_parents
    zero_by_candidate = np.broadcast_to(zero[:, None], one.shape)
    one[zero_identity] = zero_by_candidate[zero_identity]
    np.fill_diagonal(one, -math.inf)

    n_trios = len(trio_array)
    two = np.empty(n_trios, dtype=np.float64)
    edge_information = np.empty((n_trios, 2), dtype=np.float64)
    if n_trios:
        children = trio_array[:, 0]
        first_parents = trio_array[:, 1]
        second_parents = trio_array[:, 2]
        edge_information[:, 0] = identity_information[
            children, first_parents
        ]
        edge_information[:, 1] = identity_information[
            children, second_parents
        ]
        first_has_identity = edge_information[:, 0] > 0.0
        second_has_identity = edge_information[:, 1] > 0.0
        neither_has_identity = (
            ~first_has_identity & ~second_has_identity
        )
        only_first_has_identity = (
            first_has_identity & ~second_has_identity
        )
        only_second_has_identity = (
            ~first_has_identity & second_has_identity
        )
        both_have_identity = (
            first_has_identity & second_has_identity
        )
        two[neither_has_identity] = zero[
            children[neither_has_identity]
        ]
        two[only_first_has_identity] = one[
            children[only_first_has_identity],
            first_parents[only_first_has_identity],
        ]
        two[only_second_has_identity] = one[
            children[only_second_has_identity],
            second_parents[only_second_has_identity],
        ]

        fully_called_pair = (
            both_have_identity
            & fully_called_identity[children, first_parents]
            & fully_called_identity[children, second_parents]
        )
        partial_pair = both_have_identity & ~fully_called_pair
        if np.any(fully_called_pair):
            fast_trios = np.ascontiguousarray(
                trio_array[fully_called_pair]
            )
            fast_values = np.empty(len(fast_trios), dtype=np.float64)
            row_start = 0
            while row_start < len(fast_trios):
                if _dynamic_rebalance:
                    applied_threads = _apply_smart_dynamic_threads()
                    row_chunk_size = max(64, 16 * applied_threads)
                else:
                    row_chunk_size = len(fast_trios)
                row_end = min(row_start + row_chunk_size, len(fast_trios))
                fast_values[row_start:row_end] = (
                    _score_two_parent_gl_forward_kernel(
                        gl,
                        alleles,
                        marker_counts,
                        fast_trios,
                        background[2],
                        information_exponent,
                        effective_transmission,
                        float(mismatch_probability),
                        row_start,
                        row_end,
                    )
                )
                row_start = row_end
            two[fully_called_pair] = fast_values
        if np.any(partial_pair):
            partial_trios = np.ascontiguousarray(
                trio_array[partial_pair]
            )
            partial_values = np.empty(len(partial_trios), dtype=np.float64)
            row_start = 0
            while row_start < len(partial_trios):
                if _dynamic_rebalance:
                    applied_threads = _apply_smart_dynamic_threads()
                    row_chunk_size = max(64, 16 * applied_threads)
                else:
                    row_chunk_size = len(partial_trios)
                row_end = min(
                    row_start + row_chunk_size, len(partial_trios)
                )
                partial_values[row_start:row_end] = (
                    _score_two_parent_gl_linked_fallback_kernel(
                        gl,
                        alleles,
                        founders,
                        marker_counts,
                        partial_trios,
                        background[0],
                        external_log_transition,
                        background[2],
                        information_exponent,
                        effective_transmission,
                        float(mismatch_probability),
                        row_start,
                        row_end,
                    )
                )
                row_start = row_end
            two[partial_pair] = partial_values
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    junctions, callable_bins = _ancestry_junction_count_kernel(
        labels, trajectory_classes
    )
    return _ParentStateContigScores(
        zero,
        one,
        two,
        junctions,
        callable_bins,
        identity_information,
        edge_information,
        edge_matched_bins=structure_counts[0],
        edge_exposed_bins=structure_counts[1],
        pair_explained_bins=structure_counts[2],
        pair_exposed_bins=structure_counts[3],
        structure_total_bins=float(labels.shape[1]),
        candidate_source_mode_requested=candidate_source_mode,
        candidate_source_mode_applied="hard_painted",
        candidate_source_fallback=exact_source_fallback,
        candidate_source_fallback_reason=(
            "founder_missing_selected_real_site_whole_contig_hard_fallback"
            if exact_source_fallback else ""
        ),
        complete_founder_marker_count=complete_founder_marker_count,
        excluded_founder_marker_count=excluded_founder_marker_count,
    )


def _normalise_genotype_likelihoods(values: np.ndarray, kind: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise SmartEvidenceError(
            "genotype likelihoods must have shape (samples, markers, 3)"
        )
    key = str(kind).lower()
    if key in {"pl", "phred"}:
        invalid_rows = np.any(~np.isfinite(array) | (array < 0.0), axis=2)
        sanitized = array.copy()
        sanitized[invalid_rows] = 0.0
        shifted = sanitized - np.min(sanitized, axis=2, keepdims=True)
        likelihoods = np.power(10.0, -shifted / 10.0)
    elif key in {"log", "ln", "log_likelihood"}:
        if np.any(~np.isfinite(array)):
            raise SmartEvidenceError("log genotype likelihoods must be finite")
        shifted = array - np.max(array, axis=2, keepdims=True)
        likelihoods = np.exp(shifted)
    elif key in {"probability", "probabilities", "linear"}:
        if np.any(~np.isfinite(array)):
            raise SmartEvidenceError("linear genotype likelihoods must be finite")
        if np.any(array < 0.0):
            raise SmartEvidenceError("linear genotype likelihoods cannot be negative")
        likelihoods = array.copy()
    else:
        raise SmartEvidenceError(
            "likelihood_kind must be 'PL', 'log', or 'probability'"
        )
    totals = np.sum(likelihoods, axis=2, keepdims=True)
    if np.any(totals <= 0.0):
        raise SmartEvidenceError("every sample-marker likelihood vector needs mass")
    likelihoods /= totals
    return likelihoods


def _empirical_allele_frequencies(
    likelihoods: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate HWE allele frequencies and genotype posteriors by EM.

    Normalized PL/GL vectors are relative likelihoods, not genotype
    posteriors. This empirical-Bayes step supplies the required genotype prior
    before parental genotype uncertainty is integrated over transmissions.
    """
    if iterations < 1:
        raise SmartEvidenceError("frequency_em_iterations must be positive")
    n_samples = likelihoods.shape[0]
    frequencies = np.full(likelihoods.shape[1], 0.5, dtype=np.float64)
    posterior = np.empty_like(likelihoods)
    for _ in range(iterations):
        prior = np.stack((
            (1.0 - frequencies) ** 2,
            2.0 * frequencies * (1.0 - frequencies),
            frequencies ** 2,
        ), axis=1)
        posterior[:] = likelihoods * prior[None, :, :]
        normalizer = np.sum(posterior, axis=2, keepdims=True)
        posterior /= np.maximum(normalizer, np.finfo(np.float64).tiny)
        updated = np.sum(
            posterior[:, :, 1] + 2.0 * posterior[:, :, 2], axis=0
        ) / (2.0 * n_samples)
        frequencies = np.clip(updated, 1e-6, 1.0 - 1e-6)
    return frequencies, posterior


def _mendelian_table() -> np.ndarray:
    """P(child genotype | two unphased diploid parent genotypes)."""
    table = np.zeros((3, 3, 3), dtype=np.float64)
    gametes = np.asarray(((1.0, 0.0), (0.5, 0.5), (0.0, 1.0)))
    for first in range(3):
        for second in range(3):
            for first_alt in range(2):
                for second_alt in range(2):
                    child = first_alt + second_alt
                    table[first, second, child] += (
                        gametes[first, first_alt] * gametes[second, second_alt]
                    )
    return table


_MENDELIAN = _mendelian_table()


@njit(cache=True, parallel=True, fastmath=False)
def _score_genotype_likelihood_kernel(
    likelihoods: np.ndarray,
    genotype_posteriors: np.ndarray,
    null_child: np.ndarray,
    trios: np.ndarray,
    mendelian_table: np.ndarray,
) -> np.ndarray:
    """Parallel two-parent minus population-null composite likelihood."""
    scores = np.empty(len(trios), dtype=np.float64)
    epsilon = 2.2250738585072014e-308
    for row in prange(len(trios)):
        child = trios[row, 0]
        first = trios[row, 1]
        second = trios[row, 2]
        score = 0.0
        for marker in range(likelihoods.shape[1]):
            mendelian = 0.0
            for first_genotype in range(3):
                first_probability = genotype_posteriors[
                    first, marker, first_genotype
                ]
                for second_genotype in range(3):
                    parental_probability = (
                        first_probability
                        * genotype_posteriors[second, marker, second_genotype]
                    )
                    for child_genotype in range(3):
                        mendelian += (
                            parental_probability
                            * mendelian_table[
                                first_genotype, second_genotype, child_genotype
                            ]
                            * likelihoods[child, marker, child_genotype]
                        )
            null = 0.0
            for child_genotype in range(3):
                null += (
                    null_child[marker, child_genotype]
                    * likelihoods[child, marker, child_genotype]
                )
            score += math.log(max(mendelian, epsilon))
            score -= math.log(max(null, epsilon))
        scores[row] = score
    return scores


def _score_genotype_likelihood_reference(
    likelihoods: np.ndarray,
    genotype_posteriors: np.ndarray,
    null_child: np.ndarray,
    trios: np.ndarray,
) -> np.ndarray:
    """Independent vectorized oracle for kernel regression tests."""
    scores = np.empty(len(trios), dtype=np.float64)
    epsilon = np.finfo(np.float64).tiny
    for row, (child, first, second) in enumerate(trios):
        expected_child = np.einsum(
            "mg,mh,ghc->mc",
            genotype_posteriors[int(first)],
            genotype_posteriors[int(second)],
            _MENDELIAN,
            optimize=True,
        )
        mendelian = np.sum(expected_child * likelihoods[int(child)], axis=1)
        null = np.sum(null_child * likelihoods[int(child)], axis=1)
        scores[row] = float(
            np.sum(
                np.log(np.maximum(mendelian, epsilon))
                - np.log(np.maximum(null, epsilon))
            )
        )
    return scores


def smart_numba_thread_capacity() -> int:
    """Return the active Numba thread capacity without changing it."""
    return int(numba.get_num_threads())


def score_genotype_likelihood_evidence(
    contig: str,
    genotype_likelihoods: np.ndarray,
    candidate_trios: np.ndarray,
    *,
    likelihood_kind: str = "PL",
    positions: Optional[np.ndarray] = None,
    linked_log_likelihoods: Optional[np.ndarray] = None,
    allele_frequencies: Optional[np.ndarray] = None,
    frequency_em_iterations: int = 12,
) -> SmartContigEvidence:
    """Score explicit candidate trios from raw genotype likelihoods.

    The score is the marker-summed log likelihood ratio between Mendelian
    transmission from the candidate pair and an empirical-population child
    genotype distribution. It is intentionally an unlinked complement to a
    painting/HMM score, not a substitute for recombination-aware evidence.
    Candidate eligibility is supplied numerically by the caller and is never
    inferred from identifiers. Unless ``allele_frequencies`` is supplied, the
    genotype prior is an empirical HWE estimate; closed crosses can violate
    that approximation, so linked painting/HMM evidence should remain primary.
    Candidate trios are evaluated by a memory-bounded Numba ``prange`` kernel
    using the currently configured thread capacity; this function does not
    force a thread count or create nested parallelism.
    """
    likelihoods = _normalise_genotype_likelihoods(
        genotype_likelihoods, likelihood_kind
    )
    trios = np.asarray(candidate_trios, dtype=np.int64)
    if trios.ndim != 2 or trios.shape[1] != 3 or len(trios) == 0:
        raise SmartEvidenceError("candidate_trios must have shape (n, 3)")
    n_samples, n_markers, _ = likelihoods.shape
    if np.any(trios < 0) or np.any(trios >= n_samples):
        raise SmartEvidenceError("candidate_trios contains an invalid sample index")
    if np.any(trios[:, 0] == trios[:, 1]) or np.any(trios[:, 0] == trios[:, 2]):
        raise SmartEvidenceError("a child cannot be its own parent")
    if np.any(trios[:, 1] == trios[:, 2]):
        raise SmartEvidenceError("the two candidate parents must differ")
    if positions is not None:
        marker_positions = np.asarray(positions)
        if marker_positions.ndim != 1 or len(marker_positions) != n_markers:
            raise SmartEvidenceError("positions must have one value per marker")
        if len(marker_positions) > 1 and np.any(np.diff(marker_positions) <= 0):
            raise SmartEvidenceError("positions must be strictly increasing")

    if allele_frequencies is None:
        allele_frequency, genotype_posteriors = _empirical_allele_frequencies(
            likelihoods, frequency_em_iterations
        )
    else:
        allele_frequency = np.asarray(allele_frequencies, dtype=np.float64)
        if allele_frequency.shape != (n_markers,):
            raise SmartEvidenceError(
                "allele_frequencies must contain one value per marker"
            )
        if np.any(~np.isfinite(allele_frequency)) or np.any(
            (allele_frequency <= 0.0) | (allele_frequency >= 1.0)
        ):
            raise SmartEvidenceError(
                "allele_frequencies must be finite and strictly inside (0, 1)"
            )
        prior = np.stack((
            (1.0 - allele_frequency) ** 2,
            2.0 * allele_frequency * (1.0 - allele_frequency),
            allele_frequency ** 2,
        ), axis=1)
        genotype_posteriors = likelihoods * prior[None, :, :]
        genotype_posteriors /= np.maximum(
            np.sum(genotype_posteriors, axis=2, keepdims=True),
            np.finfo(np.float64).tiny,
        )
    null_child = np.column_stack((
        (1.0 - allele_frequency) ** 2,
        2.0 * allele_frequency * (1.0 - allele_frequency),
        allele_frequency ** 2,
    ))

    # Parallelism is over candidate trios. The function respects the active
    # Numba thread pool configured by thread_config/the caller and never opens
    # a nested process pool or changes the requested thread count.
    genotype_scores = _score_genotype_likelihood_kernel(
        np.ascontiguousarray(likelihoods),
        np.ascontiguousarray(genotype_posteriors),
        np.ascontiguousarray(null_child),
        np.ascontiguousarray(trios),
        _MENDELIAN,
    )

    if linked_log_likelihoods is None:
        linked = np.zeros(len(trios), dtype=np.float64)
    else:
        linked = np.asarray(linked_log_likelihoods, dtype=np.float64)
        if linked.shape != (len(trios),) or np.any(~np.isfinite(linked)):
            raise SmartEvidenceError(
                "linked_log_likelihoods must be one finite score per trio"
            )
    informative_marker_mask = np.any(
        np.ptp(likelihoods, axis=2) > 1e-12, axis=0
    )
    informative_markers = int(np.sum(informative_marker_mask))
    if informative_markers == 0:
        raise SmartEvidenceError(
            "genotype likelihoods contain no informative marker evidence"
        )
    return SmartContigEvidence(
        contig=str(contig),
        trios=trios.copy(),
        linked_log_likelihoods=linked.copy(),
        genotype_log_likelihoods=genotype_scores,
        informative_markers=informative_markers,
    )


def load_bcf_genotype_likelihoods(
    bcf_path: Any,
    contig: str,
    sample_ids: Sequence[Any],
    *,
    selected_positions: Optional[np.ndarray] = None,
    threads: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Load PL evidence from one explicitly named BCF/VCF contig.

    The header sample order must exactly equal ``sample_ids``. No path,
    sample, contig, or candidate discovery is attempted. When positions are
    supplied every requested position must be present with a valid three-state
    PL vector, preventing silent panel changes across chromosomes.
    """
    try:
        from cyvcf2 import VCF
    except ImportError as error:
        raise SmartEvidenceError(
            "cyvcf2 is required to load an explicitly supplied BCF/VCF"
        ) from error
    if int(threads) != threads or threads < 1:
        raise SmartEvidenceError("threads must be a positive integer")
    requested = None
    if selected_positions is not None:
        requested_array = np.asarray(selected_positions, dtype=np.int64)
        if requested_array.ndim != 1 or len(requested_array) == 0:
            raise SmartEvidenceError(
                "selected_positions must be a non-empty one-dimensional array"
            )
        if np.any(np.diff(requested_array) <= 0):
            raise SmartEvidenceError(
                "selected_positions must be unique and strictly increasing"
            )
        requested = set(int(value) for value in requested_array)

    reader = VCF(str(bcf_path), threads=int(threads))
    try:
        if list(reader.samples) != list(sample_ids):
            raise SmartEvidenceError(
                "BCF/VCF header sample order does not exactly match sample_ids"
            )
        values_by_position = {}
        for variant in reader(str(contig)):
            position = int(variant.POS)
            if requested is not None and position not in requested:
                continue
            values = variant.format("PL")
            if values is None:
                continue
            values = np.asarray(values, dtype=np.float64)
            if values.ndim != 2 or values.shape != (len(sample_ids), 3):
                continue
            # cyvcf2 uses negative sentinels for a missing sample call. Keep
            # the marker and make only that sample's PL row uninformative;
            # dropping the whole marker would let one missing sample erase
            # evidence for every other trio.
            invalid_rows = np.any(~np.isfinite(values) | (values < 0), axis=1)
            values = values.copy()
            values[invalid_rows] = 0.0
            if position in values_by_position:
                raise SmartEvidenceError(
                    f"duplicate PL-bearing position {position} on {contig}"
                )
            values_by_position[position] = values
    finally:
        reader.close()
    if requested is not None:
        missing = requested.difference(values_by_position)
        if missing:
            raise SmartEvidenceError(
                f"{len(missing)} requested PL positions are absent on {contig}"
            )
    if not values_by_position:
        raise SmartEvidenceError(f"no valid three-state PL evidence on {contig}")
    positions = np.asarray(sorted(values_by_position), dtype=np.int64)
    likelihoods = np.stack(
        [values_by_position[int(position)] for position in positions], axis=1
    )
    return likelihoods, positions



def load_bcf_raw_genotype_likelihoods(
    bcf_path: Any,
    contig: str,
    sample_ids: Sequence[Any],
    *,
    selected_positions: np.ndarray,
    threads: int = 1,
    read_error_probability: float = DEFAULT_READ_ERROR_PROBABILITY,
) -> tuple[np.ndarray, np.ndarray]:
    """Load selected BCF AD rows as normalized raw genotype likelihoods.

    This uses the same prior-free binomial read model as T01. AD is available
    at retained sites where caller PL is absent, so using AD uniformly avoids
    a PL-availability-dependent marker panel and keeps one observation model
    across chromosomes.
    """
    try:
        from cyvcf2 import VCF
    except ImportError as error:
        raise SmartEvidenceError(
            "cyvcf2 is required to load an explicitly supplied BCF/VCF"
        ) from error
    if int(threads) != threads or threads < 1:
        raise SmartEvidenceError("threads must be a positive integer")
    requested_array = np.asarray(selected_positions, dtype=np.int64)
    if requested_array.ndim != 1 or len(requested_array) == 0:
        raise SmartEvidenceError(
            "selected_positions must be a non-empty one-dimensional array"
        )
    if np.any(np.diff(requested_array) <= 0):
        raise SmartEvidenceError(
            "selected_positions must be unique and strictly increasing"
        )
    requested = set(int(value) for value in requested_array)

    reader = VCF(str(bcf_path), threads=int(threads))
    values_by_position = {}
    try:
        if list(reader.samples) != list(sample_ids):
            raise SmartEvidenceError(
                "BCF/VCF header sample order does not exactly match sample_ids"
            )
        for variant in reader(str(contig)):
            position = int(variant.POS)
            if position not in requested:
                continue
            values = variant.format("AD")
            if values is None:
                continue
            values = np.asarray(values)
            if (
                values.ndim != 2
                or values.shape[0] != len(sample_ids)
                or values.shape[1] < 2
            ):
                continue
            values = np.ascontiguousarray(values[:, :2]).copy()
            values[values < 0] = 0
            if position in values_by_position:
                raise SmartEvidenceError(
                    f"duplicate AD-bearing position {position} on {contig}"
                )
            values_by_position[position] = values
    finally:
        reader.close()
    missing = requested.difference(values_by_position)
    if missing:
        raise SmartEvidenceError(
            f"{len(missing)} requested AD positions are absent on {contig}"
        )
    positions = np.asarray(sorted(values_by_position), dtype=np.int64)
    allele_depths = np.stack(
        [values_by_position[int(position)] for position in positions], axis=1
    )
    try:
        likelihoods = allele_depths_to_raw_genotype_likelihoods(
            allele_depths,
            read_error_probability=read_error_probability,
            require_nonempty=True,
            require_integer=True,
        )
    except ValueError as error:
        raise SmartEvidenceError(
            f"invalid AD evidence on {contig}: {error}"
        ) from error
    return likelihoods, positions

def _as_contig_evidence(value: Any) -> SmartContigEvidence:
    if isinstance(value, SmartContigEvidence):
        evidence = value
    elif isinstance(value, Mapping):
        try:
            trios = np.asarray(value["trios"], dtype=np.int64)
            evidence = SmartContigEvidence(
                contig=str(value["contig"]),
                trios=trios,
                linked_log_likelihoods=np.asarray(
                    value.get("linked_log_likelihoods", np.zeros(len(trios))),
                    dtype=np.float64,
                ),
                genotype_log_likelihoods=np.asarray(
                    value.get("genotype_log_likelihoods", np.zeros(len(trios))),
                    dtype=np.float64,
                ),
                informative_markers=int(value["informative_markers"]),
            )
        except KeyError as error:
            raise SmartEvidenceError(
                f"smart evidence is missing required field {error.args[0]!r}"
            ) from error
    else:
        raise SmartEvidenceError(
            "smart_evidence must be SmartContigEvidence or an explicit mapping"
        )
    trios = np.asarray(evidence.trios, dtype=np.int64)
    if trios.ndim != 2 or trios.shape[1] != 3 or len(trios) == 0:
        raise SmartEvidenceError("each contig needs a non-empty (n, 3) trio array")
    linked = np.asarray(evidence.linked_log_likelihoods, dtype=np.float64)
    genotype = np.asarray(evidence.genotype_log_likelihoods, dtype=np.float64)
    if linked.shape != (len(trios),) or genotype.shape != (len(trios),):
        raise SmartEvidenceError("each score array must have one value per trio")
    if np.any(~np.isfinite(linked)) or np.any(~np.isfinite(genotype)):
        raise SmartEvidenceError("smart evidence scores must be finite")
    if evidence.informative_markers < 1:
        raise SmartEvidenceError("informative_markers must be positive")
    return replace(
        evidence,
        trios=trios.copy(),
        linked_log_likelihoods=linked.copy(),
        genotype_log_likelihoods=genotype.copy(),
    )


def _canonical_evidence(
    evidence: Sequence[SmartContigEvidence], n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    keys_reference: Optional[list[tuple[int, int, int]]] = None
    linked_rows = []
    genotype_rows = []
    markers = []
    names = []
    seen_contigs = set()
    for raw in evidence:
        item = _as_contig_evidence(raw)
        if item.contig in seen_contigs:
            raise SmartEvidenceError(f"duplicate contig identifier {item.contig!r}")
        seen_contigs.add(item.contig)
        keys = []
        score_lookup = {}
        for row, triple in enumerate(item.trios):
            child, first, second = (int(value) for value in triple)
            if not (0 <= child < n_samples and 0 <= first < n_samples
                    and 0 <= second < n_samples):
                raise SmartEvidenceError("trio index outside sample array")
            if child in (first, second) or first == second:
                raise SmartEvidenceError("invalid self-parent or duplicate-parent trio")
            if second < first:
                first, second = second, first
            key = (child, first, second)
            if key in score_lookup:
                raise SmartEvidenceError(f"duplicate trio key {key} on {item.contig}")
            score_lookup[key] = (
                float(item.linked_log_likelihoods[row]),
                float(item.genotype_log_likelihoods[row]),
            )
            keys.append(key)
        keys = sorted(keys)
        if keys_reference is None:
            keys_reference = keys
        elif keys != keys_reference:
            raise SmartEvidenceError(
                "every contig must score the same canonical candidate-trio panel"
            )
        linked_rows.append([score_lookup[key][0] for key in keys])
        genotype_rows.append([score_lookup[key][1] for key in keys])
        markers.append(item.informative_markers)
        names.append(item.contig)
    if not linked_rows or keys_reference is None:
        raise SmartEvidenceError("at least one explicit contig is required")
    return (
        np.asarray(keys_reference, dtype=np.int64),
        np.asarray(linked_rows, dtype=np.float64),
        np.asarray(genotype_rows, dtype=np.float64),
        np.asarray(markers, dtype=np.float64),
        names,
    )


_CONTRAST_ULP_FACTOR = 512.0


def _contrast_tolerance(values: np.ndarray) -> float:
    """Tolerance covering scale-dependent floating evaluation-order noise."""
    finite = np.asarray(values, dtype=np.float64)
    if finite.ndim != 1 or len(finite) == 0 or np.any(~np.isfinite(finite)):
        raise SmartEvidenceError("candidate utilities must be finite vectors")
    scale = max(1.0, float(np.max(np.abs(finite))))
    return _CONTRAST_ULP_FACTOR * np.finfo(np.float64).eps * scale


def _has_genuine_contrast(values: np.ndarray) -> bool:
    """Return whether scores contain any non-numerical candidate range."""
    scores = np.asarray(values, dtype=np.float64)
    if len(scores) < 2:
        return False
    return float(np.ptp(scores)) > _contrast_tolerance(scores)


def _unique_winner_contrast(values: np.ndarray) -> tuple[bool, float]:
    """Return unique-winner identifiability and its top-two score margin."""
    scores = np.asarray(values, dtype=np.float64)
    if len(scores) < 2:
        return False, 0.0
    ordered = np.sort(scores)
    margin = float(ordered[-1] - ordered[-2])
    return margin > _contrast_tolerance(scores), margin


def _tied_rank_probabilities(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    n_values = len(values)
    if n_values == 1:
        return np.ones(1, dtype=np.float64)
    tolerance = _contrast_tolerance(values)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(n_values, dtype=np.float64)
    start = 0
    while start < n_values:
        end = start + 1
        while (
            end < n_values
            and abs(values[order[end]] - values[order[start]]) <= tolerance
        ):
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    evidence = (n_values - ranks) / n_values
    evidence /= np.sum(evidence)
    return evidence


def _source_utilities(
    scores: np.ndarray,
    trios: np.ndarray,
    markers: np.ndarray,
    config: SmartPedigreeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform one source and mark any genuine child-contig score range."""
    utilities = np.zeros_like(scores, dtype=np.float64)
    children = np.unique(trios[:, 0])
    contrast = np.zeros(
        (scores.shape[0], int(np.max(children)) + 1), dtype=np.bool_
    )
    n_blocks = np.maximum(
        np.ceil(markers / config.markers_per_information_block), 1.0
    )
    temper = n_blocks ** config.information_tempering_power
    for contig_index in range(scores.shape[0]):
        for child in children:
            rows = np.flatnonzero(trios[:, 0] == child)
            values = scores[contig_index, rows]
            has_contrast = _has_genuine_contrast(values)
            if not has_contrast:
                utilities[contig_index, rows] = 1.0 / len(rows)
                continue
            contrast[contig_index, int(child)] = True
            centered = (values - np.max(values)) / temper[contig_index]
            centered = np.clip(centered, -60.0, 0.0)
            soft = np.exp(centered)
            soft /= np.sum(soft)
            soft = (
                (1.0 - config.chromosome_contamination) * soft
                + config.chromosome_contamination / len(rows)
            )
            ranks = _tied_rank_probabilities(values)
            utilities[contig_index, rows] = (
                config.rank_weight * ranks + (1.0 - config.rank_weight) * soft
            )
    return utilities, contrast


def _information_weights(
    markers: np.ndarray, config: SmartPedigreeConfig
) -> np.ndarray:
    weights = np.sqrt(np.asarray(markers, dtype=np.float64))
    median = float(np.median(weights))
    if median <= 0.0:
        raise SmartEvidenceError("marker information is empty")
    ratio = config.maximum_contig_weight_ratio
    weights = np.clip(weights / median, 1.0 / ratio, ratio)
    return weights / np.sum(weights)


def _combined_utilities(
    linked: np.ndarray,
    genotype: np.ndarray,
    trios: np.ndarray,
    markers: np.ndarray,
    config: SmartPedigreeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine sources with child-contig-specific identifiability masks."""
    source_arrays = []
    source_contrasts = []
    source_weights = []
    if config.linked_evidence_weight > 0.0:
        values, contrast = _source_utilities(
            linked, trios, markers, config
        )
        source_arrays.append(values)
        source_contrasts.append(contrast)
        source_weights.append(config.linked_evidence_weight)
    if config.genotype_evidence_weight > 0.0:
        values, contrast = _source_utilities(
            genotype, trios, markers, config
        )
        source_arrays.append(values)
        source_contrasts.append(contrast)
        source_weights.append(config.genotype_evidence_weight)

    weights = np.asarray(source_weights, dtype=np.float64)
    utilities = np.zeros_like(source_arrays[0])
    combined_contrast = np.logical_or.reduce(source_contrasts)
    children = np.unique(trios[:, 0])
    for contig_index in range(linked.shape[0]):
        for child in children:
            rows = np.flatnonzero(trios[:, 0] == child)
            active = np.asarray([
                contrast[contig_index, int(child)]
                for contrast in source_contrasts
            ], dtype=np.bool_)
            if not np.any(active):
                utilities[contig_index, rows] = 1.0 / len(rows)
                continue
            active_weights = weights[active]
            active_weights /= np.sum(active_weights)
            active_indices = np.flatnonzero(active)
            for source_weight, source_index in zip(
                active_weights, active_indices
            ):
                utilities[contig_index, rows] += (
                    source_weight
                    * source_arrays[int(source_index)][contig_index, rows]
                )
    return utilities, _information_weights(markers, config), combined_contrast


def _path_exists(adjacency: list[set[int]], start: int, target: int) -> bool:
    if start == target:
        return True
    stack = [start]
    seen = {start}
    while stack:
        node = stack.pop()
        for neighbour in adjacency[node]:
            if neighbour == target:
                return True
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return False



def _observed_parents(
    alternatives: np.ndarray, row: int
) -> tuple[int, ...]:
    return tuple(
        int(parent) for parent in alternatives[row, 1:] if int(parent) >= 0
    )

def _acyclic_selection(
    trios: np.ndarray,
    scores: np.ndarray,
    n_samples: int,
    local_search_passes: int,
) -> dict[int, int]:
    """Select a DAG without resolving score ties by canonical row order."""
    by_child = {}
    margins = {}
    for child in np.unique(trios[:, 0]):
        rows = np.flatnonzero(trios[:, 0] == child)
        identifiable, margin = _unique_winner_contrast(scores[rows])
        if identifiable:
            by_child[int(child)] = rows
            margins[int(child)] = margin
    child_order = sorted(by_child, key=lambda value: (-margins[value], value))
    adjacency = [set() for _ in range(n_samples)]
    selected: dict[int, int] = {}

    def can_add(row: int) -> bool:
        child, first, second = (int(value) for value in trios[row])
        return not (
            _path_exists(adjacency, child, first)
            or _path_exists(adjacency, child, second)
        )

    def add(row: int) -> None:
        child, first, second = (int(value) for value in trios[row])
        adjacency[first].add(child)
        adjacency[second].add(child)
        selected[child] = row

    def remove(child: int) -> Optional[int]:
        row = selected.pop(child, None)
        if row is not None:
            _, first, second = (int(value) for value in trios[row])
            adjacency[first].discard(child)
            adjacency[second].discard(child)
        return row

    def best_feasible(child: int) -> Optional[int]:
        rows = by_child[child]
        ordered = rows[np.argsort(-scores[rows], kind="stable")]
        feasible = np.asarray(
            [int(row) for row in ordered if can_add(int(row))],
            dtype=np.int64,
        )
        if len(feasible) == 0:
            return None
        if len(feasible) == 1:
            return int(feasible[0])
        identifiable, _ = _unique_winner_contrast(scores[feasible])
        if not identifiable:
            return None
        return int(feasible[0])

    for child in child_order:
        row = best_feasible(child)
        if row is not None:
            add(row)

    for _ in range(local_search_passes):
        changed = False
        for child in child_order:
            previous = remove(child)
            replacement = best_feasible(child)
            if replacement is not None:
                add(replacement)
            changed |= replacement != previous
        if not changed:
            break
    return selected


def _aggregate_scores(
    utilities: np.ndarray, weights: np.ndarray, selected_contigs: np.ndarray
) -> np.ndarray:
    chosen_weights = weights[selected_contigs]
    chosen_weights = chosen_weights / np.sum(chosen_weights)
    return np.sum(utilities[selected_contigs] * chosen_weights[:, None], axis=0)


def _generation_frame(
    sample_ids: Sequence[Any], selected: Mapping[int, tuple[Optional[int], Optional[int]]]
) -> pd.DataFrame:
    """Build an all-sample relationship frame without invented cohorts.

    A both-null row can be an unscored root or an unresolved descendant. With
    no metadata those states are not distinguishable, and graph depth cannot
    establish biological G0/F1/F2 labels. Generation therefore remains
    ``Unknown`` for every row; selected parent edges carry the inferred
    topology.
    """
    rows = []
    for child, sample in enumerate(sample_ids):
        first, second = selected.get(child, (None, None))
        rows.append({
            "Sample": sample,
            "Generation": "Unknown",
            "Parent1": None if first is None else sample_ids[first],
            "Parent2": None if second is None else sample_ids[second],
        })
    return pd.DataFrame(rows)


def _support_text(
    rows: np.ndarray,
    probabilities: np.ndarray,
    trios: np.ndarray,
    sample_ids: Sequence[Any],
    coverage: float,
) -> str:
    order = rows[np.argsort(-probabilities[rows], kind="stable")]
    pieces = []
    cumulative = 0.0
    denominator = float(np.sum(probabilities[rows]))
    if denominator <= 0.0:
        return ""
    for row in order:
        _, first, second = (int(value) for value in trios[row])
        probability = float(probabilities[row] / denominator)
        pieces.append(f"{sample_ids[first]}+{sample_ids[second]}:{probability:.3f}")
        cumulative += probability
        if cumulative >= coverage:
            break
    return ";".join(pieces)


def _as_parent_state_evidence(
    value: Any,
    n_samples: int,
    eligibility: Optional[_ResolvedParentEligibility] = None,
) -> SmartParentStateEvidence:
    """Validate one comparable parent-state evidence object."""
    if isinstance(value, SmartParentStateEvidence):
        item = value
    elif isinstance(value, Mapping):
        try:
            item = SmartParentStateEvidence(
                contig=str(value["contig"]),
                trios=np.asarray(value["trios"], dtype=np.int64),
                zero_parent_log_likelihoods=np.asarray(
                    value["zero_parent_log_likelihoods"], dtype=np.float64
                ),
                one_parent_log_likelihoods=np.asarray(
                    value["one_parent_log_likelihoods"], dtype=np.float64
                ),
                two_parent_log_likelihoods=np.asarray(
                    value["two_parent_log_likelihoods"], dtype=np.float64
                ),
                informative_markers=int(value["informative_markers"]),
                edge_matched_bins=value.get("edge_matched_bins"),
                edge_exposed_bins=value.get("edge_exposed_bins"),
                pair_explained_bins=value.get("pair_explained_bins"),
                pair_exposed_bins=value.get("pair_exposed_bins"),
                structure_total_bins=value.get("structure_total_bins"),
            )
        except KeyError as error:
            raise SmartEvidenceError(
                f"parent-state evidence is missing {error.args[0]!r}"
            ) from error
    else:
        raise SmartEvidenceError(
            "parent-state evidence must be SmartParentStateEvidence or a mapping"
        )

    trios = np.asarray(item.trios, dtype=np.int64)
    zero = np.asarray(item.zero_parent_log_likelihoods, dtype=np.float64)
    one = np.asarray(item.one_parent_log_likelihoods, dtype=np.float64)
    two = np.asarray(item.two_parent_log_likelihoods, dtype=np.float64)
    if eligibility is None:
        required_children = np.ones(n_samples, dtype=np.bool_)
        required_parents = ~np.eye(n_samples, dtype=np.bool_)
    else:
        required_children = eligibility.eligible_children
        required_parents = eligibility.eligible_parents
    if trios.ndim != 2 or trios.shape[1] != 3:
        raise SmartEvidenceError("parent-state trios must have shape (rows, 3)")
    if (
        zero.shape != (n_samples,)
        or np.any(~np.isfinite(zero[required_children]))
        or np.any(np.isnan(zero) | np.isposinf(zero))
    ):
        raise SmartEvidenceError(
            "zero-parent evidence must be one finite log likelihood per child"
        )
    if one.shape != (n_samples, n_samples):
        raise SmartEvidenceError(
            "one-parent evidence must have shape (samples, samples)"
        )
    if (
        np.any(~np.isfinite(one[required_parents]))
        or np.any(np.isnan(one) | np.isposinf(one))
    ):
        raise SmartEvidenceError(
            "off-diagonal one-parent log likelihoods must be finite"
        )
    if not np.all(np.isneginf(np.diag(one))):
        raise SmartEvidenceError(
            "the one-parent self-parent diagonal must be negative infinity"
        )
    if two.shape != (len(trios),) or np.any(~np.isfinite(two)):
        raise SmartEvidenceError(
            "two-parent evidence must be one finite log likelihood per trio"
        )
    if item.informative_markers < 1:
        raise SmartEvidenceError("informative_markers must be positive")

    structure_values = (
        getattr(item, "edge_matched_bins", None),
        getattr(item, "edge_exposed_bins", None),
        getattr(item, "pair_explained_bins", None),
        getattr(item, "pair_exposed_bins", None),
        getattr(item, "structure_total_bins", None),
    )
    if any(raw is None for raw in structure_values):
        raise SmartEvidenceError(
            "combined_v1 requires all parenthood structure arrays and "
            "structure_total_bins on every contig"
        )
    else:
        edge_matched = np.asarray(structure_values[0], dtype=np.float64)
        edge_exposed = np.asarray(structure_values[1], dtype=np.float64)
        pair_explained = np.asarray(structure_values[2], dtype=np.float64)
        pair_exposed = np.asarray(structure_values[3], dtype=np.float64)
        structure_total_bins = float(structure_values[4])
        arrays = (edge_matched, edge_exposed, pair_explained, pair_exposed)
        if (
            edge_matched.shape != (n_samples, n_samples)
            or edge_exposed.shape != edge_matched.shape
            or pair_explained.shape != (len(trios),)
            or pair_exposed.shape != pair_explained.shape
            or any(np.any(~np.isfinite(array)) for array in arrays)
            or any(np.any(array < 0.0) for array in arrays)
            or np.any(edge_matched > edge_exposed)
            or np.any(pair_explained > pair_exposed)
            or not np.isfinite(structure_total_bins)
            or structure_total_bins <= 0.0
        ):
            raise SmartEvidenceError(
                "parenthood structure counts must be finite non-negative arrays "
                "with valid totals and matched counts no larger than exposed"
            )
        if not (
            np.array_equal(edge_matched, edge_matched.T)
            and np.array_equal(edge_exposed, edge_exposed.T)
        ):
            raise SmartEvidenceError("edge structure counts must be symmetric")

    return SmartParentStateEvidence(
        contig=str(item.contig),
        trios=trios,
        zero_parent_log_likelihoods=zero,
        one_parent_log_likelihoods=one,
        two_parent_log_likelihoods=two,
        informative_markers=int(item.informative_markers),
        edge_matched_bins=edge_matched,
        edge_exposed_bins=edge_exposed,
        pair_explained_bins=(
            pair_explained
        ),
        pair_exposed_bins=pair_exposed,
        structure_total_bins=structure_total_bins,
    )


def _canonical_parent_state_evidence(
    evidence: Sequence[SmartParentStateEvidence],
    n_samples: int,
    eligibility: Optional[_ResolvedParentEligibility] = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
    Optional[np.ndarray], Optional[np.ndarray], list[str],
]:
    """Align a fixed two-parent screen while retaining full M0/M1 evidence.

    Internally generated panels are already canonical and identically ordered.
    Validate that common case with vector operations and append its score array
    directly. Permuted caller-supplied panels are canonically sorted once,
    preserving the historical support for arbitrary external row order without
    rebuilding a 539,200-entry Python dictionary for every contig.
    """
    reference_trios: Optional[np.ndarray] = None
    zero_rows = []
    one_rows = []
    two_rows = []
    edge_matched_rows = []
    edge_exposed_rows = []
    pair_explained_rows = []
    pair_exposed_rows = []
    structure_total_rows = []
    structure_presence: Optional[bool] = None
    markers = []
    names = []
    seen_contigs = set()
    for raw in evidence:
        item = _as_parent_state_evidence(raw, n_samples, eligibility)
        if item.contig in seen_contigs:
            raise SmartEvidenceError(f"duplicate contig identifier {item.contig!r}")
        seen_contigs.add(item.contig)
        edge_matched = item.edge_matched_bins
        edge_exposed = item.edge_exposed_bins
        pair_explained = item.pair_explained_bins
        structure_total = item.structure_total_bins
        pair_exposed = item.pair_exposed_bins
        item_has_structure = edge_matched is not None
        if structure_presence is None:
            structure_presence = item_has_structure
        elif structure_presence != item_has_structure:
            raise SmartEvidenceError(
                "parenthood structure evidence must be present on every contig "
                "or absent from every contig"
            )

        canonical = np.asarray(item.trios, dtype=np.int64)
        scores = np.asarray(
            item.two_parent_log_likelihoods, dtype=np.float64
        )
        if len(canonical):
            child = canonical[:, 0]
            first = canonical[:, 1]
            second = canonical[:, 2]
            if (
                np.any(child < 0)
                or np.any(child >= n_samples)
                or np.any(first < 0)
                or np.any(first >= n_samples)
                or np.any(second < 0)
                or np.any(second >= n_samples)
            ):
                raise SmartEvidenceError("trio index outside sample array")
            if np.any(child == first) or np.any(child == second):
                raise SmartEvidenceError(
                    "invalid self-parent or duplicate-parent trio"
                )
            swap = second < first
            if np.any(swap):
                canonical = canonical.copy()
                temporary = canonical[swap, 1].copy()
                canonical[swap, 1] = canonical[swap, 2]
                canonical[swap, 2] = temporary
            if np.any(canonical[:, 1] == canonical[:, 2]):
                raise SmartEvidenceError(
                    "invalid self-parent or duplicate-parent trio"
                )

            previous = canonical[:-1]
            current = canonical[1:]
            ordered = np.all(
                (current[:, 0] > previous[:, 0])
                | (
                    (current[:, 0] == previous[:, 0])
                    & (
                        (current[:, 1] > previous[:, 1])
                        | (
                            (current[:, 1] == previous[:, 1])
                            & (current[:, 2] >= previous[:, 2])
                        )
                    )
                )
            )
            if not ordered:
                order = np.lexsort((
                    canonical[:, 2], canonical[:, 1], canonical[:, 0]
                ))
                canonical = np.ascontiguousarray(canonical[order])
                scores = np.ascontiguousarray(scores[order])
                if pair_explained is not None:
                    pair_explained = np.ascontiguousarray(pair_explained[order])
                    pair_exposed = np.ascontiguousarray(pair_exposed[order])
            duplicate = np.all(canonical[1:] == canonical[:-1], axis=1)
            if np.any(duplicate):
                key = tuple(
                    int(value) for value in canonical[1:][duplicate][0]
                )
                raise SmartEvidenceError(
                    f"duplicate trio key {key} on {item.contig}"
                )

        if reference_trios is None:
            reference_trios = canonical
        elif not np.array_equal(canonical, reference_trios):
            raise SmartEvidenceError(
                "every contig must score the same canonical two-parent panel"
            )
        zero_rows.append(item.zero_parent_log_likelihoods)
        one_rows.append(item.one_parent_log_likelihoods)
        two_rows.append(scores)
        markers.append(item.informative_markers)
        names.append(item.contig)
        if item_has_structure:
            edge_matched_rows.append(edge_matched)
            edge_exposed_rows.append(edge_exposed)
            structure_total_rows.append(structure_total)
            pair_explained_rows.append(pair_explained)
            pair_exposed_rows.append(pair_exposed)
    if not zero_rows or reference_trios is None:
        raise SmartEvidenceError("at least one parent-state contig is required")
    return (
        np.asarray(reference_trios, dtype=np.int64).reshape((-1, 3)),
        np.asarray(zero_rows, dtype=np.float64),
        np.asarray(one_rows, dtype=np.float64),
        np.asarray(two_rows, dtype=np.float64),
        np.asarray(markers, dtype=np.float64),
        (
            None if not structure_presence
            else np.asarray(edge_matched_rows, dtype=np.float64)
        ),
        (
            None if not structure_presence
            else np.asarray(edge_exposed_rows, dtype=np.float64)
        ),
        (
            None if not structure_presence
            else np.asarray(pair_explained_rows, dtype=np.float64)
        ),
        (
            None if not structure_presence
            else np.asarray(pair_exposed_rows, dtype=np.float64)
        ),
        (
            None if not structure_presence
            else np.asarray(structure_total_rows, dtype=np.float64)
        ),
        names,
    )

def _parent_state_alternatives(
    trios: np.ndarray,
    zero: np.ndarray,
    one: np.ndarray,
    two: np.ndarray,
    contamination: float,
    eligibility: Optional[_ResolvedParentEligibility] = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """Build M0/M1/M2 configurations inside the eligible universe."""
    n_contigs, n_samples = zero.shape
    if eligibility is None:
        eligibility = _resolve_parent_eligibility(None, range(n_samples))
    trio_children = trios[:, 0]
    child_starts = np.searchsorted(
        trio_children,
        np.arange(n_samples + 1, dtype=np.int64),
    ).astype(np.int64)
    eligible_two_rows = []
    two_counts = np.zeros(n_samples, dtype=np.int64)
    for child in range(n_samples):
        rows = np.arange(
            int(child_starts[child]),
            int(child_starts[child + 1]),
            dtype=np.int64,
        )
        if len(rows):
            child_trios = trios[rows]
            child_rows = rows[
                _eligible_parent_pair_mask(
                    eligibility,
                    child,
                    child_trios[:, 1],
                    child_trios[:, 2],
                )
            ]
        else:
            child_rows = np.empty(0, dtype=np.int64)
        eligible_two_rows.append(child_rows)
        two_counts[child] = len(child_rows)
    parent_counts = np.count_nonzero(
        eligibility.eligible_parents, axis=1
    ).astype(np.int64)
    n_rows = int(np.sum(
        eligibility.eligible_children * (
            1 + parent_counts + two_counts

        )
    ))
    alternatives = np.empty((n_rows, 3), dtype=np.int64)
    states = np.empty(n_rows, dtype=np.int8)
    log_likelihoods = np.empty((n_contigs, n_rows), dtype=np.float64)
    by_child = []
    scored_counts = np.zeros((n_samples, 3), dtype=np.int64)
    offset = 0
    for child in range(n_samples):
        start = offset
        if not eligibility.eligible_children[child]:
            by_child.append(np.empty(0, dtype=np.int64))
            continue
        alternatives[offset] = (child, _EXTERNAL_PARENT, _EXTERNAL_PARENT)
        states[offset] = _ZERO_OBSERVED
        log_likelihoods[:, offset] = zero[:, child]
        offset += 1

        parents = np.flatnonzero(eligibility.eligible_parents[child])
        one_end = offset + len(parents)
        alternatives[offset:one_end, 0] = child
        alternatives[offset:one_end, 1] = parents
        alternatives[offset:one_end, 2] = _EXTERNAL_PARENT
        states[offset:one_end] = _ONE_OBSERVED
        log_likelihoods[:, offset:one_end] = one[:, child, parents]
        offset = one_end

        child_two_rows = eligible_two_rows[child]
        two_end = offset + len(child_two_rows)
        alternatives[offset:two_end] = trios[child_two_rows]
        states[offset:two_end] = _TWO_OBSERVED
        log_likelihoods[:, offset:two_end] = two[:, child_two_rows]
        offset = two_end
        rows = np.arange(start, offset, dtype=np.int64)
        by_child.append(rows)
        scored_counts[child] = (
            1,
            int(parent_counts[child]),
            len(child_two_rows),
        )
    if offset != n_rows:
        raise AssertionError("internal parent-state alternative count mismatch")

    if contamination > 0.0:
        log_primary = math.log1p(-contamination)
        log_null = math.log(contamination)
        nonzero_rows = np.flatnonzero(states != _ZERO_OBSERVED)
        children = alternatives[nonzero_rows, 0]
        log_likelihoods[:, nonzero_rows] = np.logaddexp(
            log_primary + log_likelihoods[:, nonzero_rows],
            log_null + zero[:, children],
        )

    full_counts = np.zeros((n_samples, 3), dtype=np.int64)
    full_counts[:, _ZERO_OBSERVED] = eligibility.eligible_children.astype(
        np.int64
    )
    full_counts[:, _ONE_OBSERVED] = np.count_nonzero(
        eligibility.eligible_parents, axis=1
    )
    full_counts[:, _TWO_OBSERVED] = _eligible_parent_pair_counts(eligibility)
    return (
        alternatives,
        states,
        log_likelihoods,
        by_child,
        full_counts,
        scored_counts,
    )


def _logsumexp_finite(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return -np.inf
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def _softmax_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.zeros_like(values)
    finite = np.isfinite(values)
    if not np.any(finite):
        return output
    maximum = float(np.max(values[finite]))
    output[finite] = np.exp(values[finite] - maximum)
    output /= np.sum(output)
    return output


@njit(cache=True)
def _linear_top_two_finite(rows, scores):
    """Stable O(n) top-two scan used by repeated bootstrap selections."""
    best_row = -1
    best_value = -math.inf
    second_value = -math.inf
    finite_count = 0
    maximum_absolute_value = 1.0
    for position in range(len(rows)):
        row = int(rows[position])
        value = float(scores[row])
        if not math.isfinite(value):
            continue
        finite_count += 1
        absolute_value = abs(value)
        if absolute_value > maximum_absolute_value:
            maximum_absolute_value = absolute_value
        if best_row < 0 or value > best_value:
            if best_row >= 0:
                second_value = best_value
            best_row = row
            best_value = value
        elif value > second_value:
            # Strict comparison retains the first occurrence as the stable
            # winner while still making an equal later value the runner-up.
            second_value = value
    return (
        best_row,
        best_value,
        second_value,
        finite_count,
        maximum_absolute_value,
    )


def _unique_finite_winner(
    rows: np.ndarray,
    scores: np.ndarray,
) -> tuple[Optional[int], float]:
    candidate_rows = np.asarray(rows, dtype=np.int64)
    candidate_scores = np.asarray(scores, dtype=np.float64)
    (
        best_row,
        best_value,
        second_value,
        finite_count,
        maximum_absolute_value,
    ) = _linear_top_two_finite(candidate_rows, candidate_scores)
    if finite_count == 0:
        return None, 0.0
    if finite_count == 1:
        return int(best_row), np.inf
    margin = float(best_value - second_value)
    tolerance = (
        _CONTRAST_ULP_FACTOR
        * np.finfo(np.float64).eps
        * maximum_absolute_value
    )
    if margin <= tolerance:
        return None, margin
    return int(best_row), margin


_ANCESTRY_DEPTH_MAX_COMPONENTS = 6
_ANCESTRY_DEPTH_GMM_N_INIT = 10
_ANCESTRY_DEPTH_GMM_MAX_ITERATIONS = 500
_ANCESTRY_DEPTH_GMM_REGULARIZATION = 1e-3


@dataclass(frozen=True)
class _AncestryDepthModel:
    """Unsupervised relative ancestry-depth model from reconstructed paintings."""

    adjusted_junction_burden: np.ndarray
    callability_fraction: np.ndarray
    posterior: np.ndarray
    component_means: np.ndarray
    component_standard_deviations: np.ndarray
    component_weights: np.ndarray
    selected_bic: float
    tested_bics: tuple[float, ...]


@dataclass(frozen=True)
class _GraphParentStateSelection:
    rows: dict[int, int]
    direction_resolved_children: frozenset[int]
    selected_parent_role_probabilities: dict[int, float]


def _fit_ancestry_depth_model(
    junction_counts: np.ndarray,
    callable_haplotype_bins: np.ndarray,
    seed: int,
) -> _AncestryDepthModel:
    """Fit a deterministic BIC-selected mixture of relative ancestry burdens.

    Junction counts are scaled to the maximum observed painting callability.
    Component likelihoods are then tempered by each sample's relative
    callability, so incomplete paintings move toward the mixture prevalence
    rather than becoming spuriously certain shallow samples. Components are
    ordered by increasing burden and represent relative ancestry depth only;
    they are not generation labels.
    """
    counts = np.asarray(junction_counts, dtype=np.float64)
    callable_bins = np.asarray(callable_haplotype_bins, dtype=np.float64)
    if (
        counts.ndim != 1
        or callable_bins.shape != counts.shape
        or np.any(~np.isfinite(counts))
        or np.any(~np.isfinite(callable_bins))
        or np.any(counts < 0.0)
        or np.any(callable_bins < 0.0)
    ):
        raise SmartEvidenceError(
            "ancestry junction counts and callability must be finite, "
            "non-negative sample vectors"
        )

    adjusted = np.full(len(counts), np.nan, dtype=np.float64)
    callability = np.zeros(len(counts), dtype=np.float64)
    valid = callable_bins > 0.0
    if not np.any(valid):
        return _AncestryDepthModel(
            adjusted,
            callability,
            np.zeros((len(counts), 1), dtype=np.float64),
            np.asarray((np.nan,)),
            np.asarray((np.nan,)),
            np.asarray((1.0,)),
            np.nan,
            (np.nan,),
        )

    reference_callability = float(np.max(callable_bins[valid]))
    callability[valid] = callable_bins[valid] / reference_callability
    adjusted[valid] = (
        counts[valid] * reference_callability / callable_bins[valid]
    )
    observed = adjusted[valid]
    center = float(np.mean(observed))
    scale = float(np.std(observed))
    distinct = np.unique(observed)
    if (
        len(observed) < 4
        or len(distinct) < 2
        or scale <= _contrast_tolerance(np.asarray((center,)))
    ):
        posterior = np.zeros((len(counts), 1), dtype=np.float64)
        posterior[valid, 0] = 1.0
        return _AncestryDepthModel(
            adjusted,
            callability,
            posterior,
            np.asarray((center,)),
            np.asarray((scale,)),
            np.asarray((1.0,)),
            np.nan,
            (np.nan,),
        )

    standardized = np.sort((observed - center) / scale)
    maximum_components = min(
        _ANCESTRY_DEPTH_MAX_COMPONENTS,
        len(distinct),
        max(1, len(observed) // 2),
    )
    fitted = fit_bic_selected_gaussian_mixture_1d(
        standardized,
        maximum_components,
        int(seed),
        n_init=_ANCESTRY_DEPTH_GMM_N_INIT,
        max_iter=_ANCESTRY_DEPTH_GMM_MAX_ITERATIONS,
        reg_covar=_ANCESTRY_DEPTH_GMM_REGULARIZATION,
    )
    if not fitted.converged:
        posterior = np.zeros((len(counts), 1), dtype=np.float64)
        posterior[valid, 0] = 1.0
        return _AncestryDepthModel(
            adjusted,
            callability,
            posterior,
            np.asarray((center,)),
            np.asarray((scale,)),
            np.asarray((1.0,)),
            np.nan,
            fitted.tested_bics,
        )
    standardized_means = fitted.means
    standardized_variances = fitted.variances
    weights = fitted.weights

    posterior = np.zeros((len(counts), len(weights)), dtype=np.float64)
    valid_standardized = (adjusted[valid] - center) / scale
    log_density = (
        -0.5 * np.log(2.0 * math.pi * standardized_variances)[None, :]
        -0.5 * (
            valid_standardized[:, None] - standardized_means[None, :]
        ) ** 2 / standardized_variances[None, :]
    )
    log_scores = (
        np.log(weights)[None, :]
        + callability[valid, None] * log_density
    )
    log_scores -= np.max(log_scores, axis=1, keepdims=True)
    probabilities = np.exp(log_scores)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    posterior[valid] = probabilities
    return _AncestryDepthModel(
        adjusted,
        callability,
        posterior,
        center + scale * standardized_means,
        scale * np.sqrt(standardized_variances),
        weights,
        fitted.selected_bic,
        fitted.tested_bics,
    )


def _parent_role_probability(
    row: int,
    alternatives: np.ndarray,
    depth_posterior: Optional[np.ndarray],
) -> float:
    """Probability all observed parents occupy an earlier latent depth."""
    parents = tuple(
        int(parent)
        for parent in alternatives[row, 1:]
        if int(parent) >= 0
    )
    if not parents:
        return 1.0
    if depth_posterior is None or depth_posterior.shape[1] < 2:
        return 0.0
    child = int(alternatives[row, 0])
    child_posterior = depth_posterior[child]
    if float(np.sum(child_posterior)) <= 0.0:
        return 0.0
    lower_depth_probability = (
        np.cumsum(depth_posterior, axis=1) - depth_posterior
    )
    probability = 0.0
    for depth, child_probability in enumerate(child_posterior):
        term = float(child_probability)
        for parent in parents:
            term *= float(lower_depth_probability[parent, depth])
        probability += term
    return float(np.clip(probability, 0.0, 1.0))


@dataclass(frozen=True)
class _ParentStateSelection:
    state_log_evidence: np.ndarray
    state_scores: np.ndarray
    state_support: np.ndarray
    decision_scores: np.ndarray
    fitted_prior_parameters: np.ndarray
    loo_state_priors: np.ndarray
    local_states: dict[int, int]
    local_rows: dict[int, int]
    graph_rows: dict[int, int]
    graph_tie_conflicts: frozenset[int]
    graph_direction_resolved_children: frozenset[int]
    graph_parent_role_probabilities: dict[int, float]
    ancestry_depth_model: Optional[_AncestryDepthModel]
    state_margins: np.ndarray
    identity_margins: np.ndarray
    unresolved_reasons: tuple[Optional[str], ...]
    m1_over_m0_edge_gains: np.ndarray
    m2_over_first_m1_edge_gains: np.ndarray
    m2_over_second_m1_edge_gains: np.ndarray
    predictive_fold_count: int


def _integrated_parent_state_log_evidence(
    aggregate_log_likelihoods: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
) -> np.ndarray:
    """Log-mean identity evidence, using the full eligible multiplicity."""
    evidence = np.full((len(by_child), 3), -np.inf, dtype=np.float64)
    for child, child_rows in enumerate(by_child):
        for state in range(3):
            rows = child_rows[states[child_rows] == state]
            if len(rows):
                evidence[child, state] = (
                    _logsumexp_finite(aggregate_log_likelihoods[rows])
                    - math.log(float(full_counts[child, state]))
                )
    return evidence


def _balanced_predictive_fold_weights(
    contig_weights: np.ndarray,
    contig_information_weights: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Pair held-out chromosomes by effective information weight.

    Stable heavy--light pairing yields eleven two-chromosome test folds for a
    22-chromosome run, each trained on the complementary twenty. A bootstrap
    chromosome and all of its multiplicity stay in one fold; missing contigs
    are removed before pairing. Odd active counts produce one singleton test
    fold. Fewer than three unique chromosomes cannot form disjoint non-empty
    train/test folds and are explicitly unresolved.
    """
    weights = np.asarray(contig_weights, dtype=np.float64)
    information = np.asarray(contig_information_weights, dtype=np.float64)
    if information.shape != weights.shape:
        raise SmartEvidenceError(
            "predictive contig information weights must match contig weights"
        )
    active = np.flatnonzero(weights > 0.0)
    if len(active) < 3:
        return ()
    effective_information = information[active] * weights[active]
    order = np.lexsort((active, effective_information))
    ordered = active[order]
    test_groups = []
    left = 0
    right = len(ordered) - 1
    while left < right:
        test_groups.append((int(ordered[left]), int(ordered[right])))
        left += 1
        right -= 1
    if left == right:
        test_groups.append((int(ordered[left]),))

    folds = []
    for test_indices in test_groups:
        train = weights.copy()
        test = np.zeros_like(weights)
        for index in test_indices:
            train[index] = 0.0
            test[index] = weights[index]
        folds.append((train, test))
    return tuple(folds)


def _training_edge_eligibility_mask(
    training_scores: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    child_rows: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Select B3 configurations using training chromosomes only."""
    eligible = np.zeros(len(alternatives), dtype=np.bool_)
    zero_rows = child_rows[states[child_rows] == _ZERO_OBSERVED]
    one_rows = child_rows[states[child_rows] == _ONE_OBSERVED]
    two_rows = child_rows[states[child_rows] == _TWO_OBSERVED]
    if len(zero_rows) != 1:
        return eligible
    zero_row = int(zero_rows[0])
    eligible[zero_row] = True

    def positive(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left = np.asarray(left, dtype=np.float64)
        right = np.broadcast_to(np.asarray(right, dtype=np.float64), left.shape)
        output = np.zeros(left.shape, dtype=np.bool_)
        finite = np.isfinite(left) & np.isfinite(right)
        tolerance = np.zeros(left.shape, dtype=np.float64)
        tolerance[finite] = (
            _CONTRAST_ULP_FACTOR
            * np.finfo(np.float64).eps
            * np.maximum(
                np.maximum(np.abs(left[finite]), np.abs(right[finite])), 1.0
            )
        )
        output[finite] = (left[finite] - right[finite]) > tolerance[finite]
        return output

    if len(one_rows):
        eligible[one_rows] = positive(
            training_scores[one_rows], training_scores[zero_row]
        )
    if len(two_rows) and len(one_rows):
        one_by_parent = np.full(n_samples, -1, dtype=np.int64)
        one_by_parent[alternatives[one_rows, 1]] = one_rows
        first_m1 = one_by_parent[alternatives[two_rows, 1]]
        second_m1 = one_by_parent[alternatives[two_rows, 2]]
        valid = (first_m1 >= 0) & (second_m1 >= 0)
        valid_two = two_rows[valid]
        first_m1 = first_m1[valid]
        second_m1 = second_m1[valid]
        eligible[valid_two] = positive(
            training_scores[valid_two], training_scores[first_m1]
        ) & positive(
            training_scores[valid_two], training_scores[second_m1]
        )
    return eligible


def _cross_fitted_parent_state_scores(
    contig_log_likelihoods: np.ndarray,
    contig_weights: np.ndarray,
    contig_information_weights: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
    use_training_edge_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Score model order by prediction on chromosomes excluded from fitting.

    Within each state, training chromosomes determine a probability mixture
    over candidate identities. Each held-out chromosome pair is scored under
    that frozen mixture. This is deliberately not a sequential
    rewriting of the whole-genome marginal likelihood: candidate weights are
    learned on a disjoint chromosome set before held-out evaluation.

    Edge gains sum disjoint held-out fold contrasts, so every chromosome
    contributes exactly once. They compare M1(p) with M0 and M2(p,q)
    separately with M1(p) and M1(q). Bootstrap and LOCO measure stability.
    """
    folds = _balanced_predictive_fold_weights(
        contig_weights, contig_information_weights
    )
    n_samples = len(by_child)
    n_alternatives = len(alternatives)
    state_evidence = np.full((n_samples, 3), -np.inf, dtype=np.float64)
    m1_gain = np.full(n_alternatives, np.nan, dtype=np.float64)
    m2_over_first = np.full(n_alternatives, np.nan, dtype=np.float64)
    m2_over_second = np.full(n_alternatives, np.nan, dtype=np.float64)
    if not folds:
        return (
            state_evidence, m1_gain, m2_over_first, m2_over_second, 0
        )

    train_scores = [train @ contig_log_likelihoods for train, _ in folds]
    test_scores = [test @ contig_log_likelihoods for _, test in folds]
    for child, child_rows in enumerate(by_child):
        if not len(child_rows):
            continue
        training_masks = (
            [
                _training_edge_eligibility_mask(
                    train, alternatives, states, child_rows, n_samples
                )
                for train in train_scores
            ]
            if use_training_edge_mask
            else [None] * len(folds)
        )
        state_contributions = np.zeros(3, dtype=np.float64)
        state_available = np.ones(3, dtype=np.bool_)
        for state in range(3):
            all_rows = child_rows[states[child_rows] == state]
            if not len(all_rows):
                state_available[state] = False
                continue
            for train, test, training_mask in zip(
                train_scores, test_scores, training_masks
            ):
                rows = (
                    all_rows
                    if training_mask is None
                    else all_rows[training_mask[all_rows]]
                )
                if not len(rows):
                    state_available[state] = False
                    break
                probabilities = _softmax_finite(train[rows])
                positive = probabilities > 0.0
                if not np.any(positive):
                    state_available[state] = False
                    break
                state_contributions[state] += _logsumexp_finite(
                    test[rows[positive]] + np.log(probabilities[positive])
                )
            if state_available[state]:
                # This correction labels omitted screen mass but does not make
                # cross-fitted predictive mixtures monotone under panel growth.
                state_contributions[state] += math.log(
                    float(len(all_rows)) / float(full_counts[child, state])
                )
                state_evidence[child, state] = state_contributions[state]

        zero_rows = child_rows[states[child_rows] == _ZERO_OBSERVED]
        one_rows = child_rows[states[child_rows] == _ONE_OBSERVED]
        two_rows = child_rows[states[child_rows] == _TWO_OBSERVED]
        if len(zero_rows) != 1:
            continue
        zero_row = int(zero_rows[0])
        one_by_parent = {
            int(alternatives[row, 1]): int(row) for row in one_rows
        }
        for row in one_rows:
            row = int(row)
            m1_gain[row] = sum(
                float(test[row] - test[zero_row]) for test in test_scores
            )
        for row in two_rows:
            row = int(row)
            first = int(alternatives[row, 1])
            second = int(alternatives[row, 2])
            first_m1 = one_by_parent.get(first)
            second_m1 = one_by_parent.get(second)
            if first_m1 is None or second_m1 is None:
                continue
            # Removing the second parent leaves M1(first), and vice versa.
            m2_over_first[row] = sum(
                float(test[row] - test[first_m1]) for test in test_scores
            )
            m2_over_second[row] = sum(
                float(test[row] - test[second_m1]) for test in test_scores
            )
    return (
        state_evidence,
        m1_gain,
        m2_over_first,
        m2_over_second,
        len(folds),
    )


def _fit_hierarchical_parent_state_prior(
    state_log_evidence: np.ndarray,
    base_probabilities: Sequence[float],
    strength: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a weak Dirichlet mixture and return leave-one-child-out priors.

    A child's own responsibility is subtracted before its predictive state
    prior is formed. This prevents the evidence being evaluated from directly
    setting its own prior while allowing the cohort to share information about
    the prevalence of zero-, one-, and two-observed-parent states.
    """
    evidence = np.asarray(state_log_evidence, dtype=np.float64)
    base = np.asarray(base_probabilities, dtype=np.float64)
    alpha = strength * base
    active = np.any(np.isfinite(evidence), axis=1)
    responsibilities = np.zeros_like(evidence)
    mixture = base.copy()
    for _ in range(max_iterations):
        previous = mixture.copy()
        log_mixture = np.log(mixture)
        for child in np.flatnonzero(active):
            responsibilities[child] = _softmax_finite(
                evidence[child] + log_mixture
            )
        posterior = alpha + np.sum(responsibilities[active], axis=0)
        mixture = posterior / np.sum(posterior)
        if float(np.max(np.abs(mixture - previous))) <= tolerance:
            break

    posterior = alpha + np.sum(responsibilities[active], axis=0)
    loo_priors = np.tile(base, (len(evidence), 1))
    for child in np.flatnonzero(active):
        parameters = posterior - responsibilities[child]
        loo_priors[child] = parameters / np.sum(parameters)
    return posterior, loo_priors


def _parent_state_score_components(
    aggregate_log_likelihoods: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
    child_state_priors: np.ndarray,
    state_log_evidence: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build marginal state and conditional-identity decision evidence."""
    n_samples = len(by_child)
    if state_log_evidence is None:
        state_log_evidence = _integrated_parent_state_log_evidence(
            aggregate_log_likelihoods, states, by_child, full_counts
        )
    priors = np.asarray(child_state_priors, dtype=np.float64)
    if priors.shape == (3,):
        priors = np.tile(priors, (n_samples, 1))
    if (
        priors.shape != (n_samples, 3)
        or np.any(~np.isfinite(priors))
        or np.any(priors <= 0.0)
    ):
        raise SmartEvidenceError(
            "child parent-state priors must be positive with shape (samples, 3)"
        )
    state_scores = state_log_evidence + np.log(priors)
    state_support = np.zeros((n_samples, 3), dtype=np.float64)
    decision_scores = np.full(
        len(aggregate_log_likelihoods), -np.inf, dtype=np.float64
    )
    for child, child_rows in enumerate(by_child):
        for state in range(3):
            rows = child_rows[states[child_rows] == state]
            if not len(rows) or not np.isfinite(state_scores[child, state]):
                continue
            values = aggregate_log_likelihoods[rows]
            maximum = float(np.max(values))
            decision_scores[rows] = (
                state_scores[child, state] + values - maximum
            )
        state_support[child] = _softmax_finite(state_scores[child])
    return state_log_evidence, state_scores, state_support, decision_scores


def _local_parent_state_winners(
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    state_scores: np.ndarray,
    decision_scores: np.ndarray,
) -> tuple[
    dict[int, int],
    dict[int, int],
    np.ndarray,
    np.ndarray,
    tuple[Optional[str], ...],
]:
    """Choose the marginal state first, then an identity within that state."""
    n_samples = len(by_child)
    local_states = {}
    local_rows = {}
    state_margins = np.full(n_samples, np.nan, dtype=np.float64)
    identity_margins = np.full(n_samples, np.nan, dtype=np.float64)
    reasons: list[Optional[str]] = [None] * n_samples
    state_indices = np.arange(3, dtype=np.int64)
    for child, rows in enumerate(by_child):
        state, state_margin = _unique_finite_winner(
            state_indices, state_scores[child]
        )
        state_margins[child] = state_margin
        if state is None:
            reasons[child] = "no_unique_marginal_parent_state"
            continue
        state = int(state)
        local_states[child] = state
        state_rows = rows[states[rows] == state]
        row, identity_margin = _unique_finite_winner(
            state_rows, decision_scores
        )
        identity_margins[child] = identity_margin
        if row is None:
            reasons[child] = "parent_state_resolved_identity_unresolved"
            continue
        local_rows[child] = int(row)
    return (
        local_states,
        local_rows,
        state_margins,
        identity_margins,
        tuple(reasons),
    )


def _acyclic_local_parent_state_selection(
    alternatives: np.ndarray,
    local_rows: Mapping[int, int],
    state_margins: np.ndarray,
    identity_margins: np.ndarray,
    depth_posterior: Optional[np.ndarray],
) -> _GraphParentStateSelection:
    """Return the exact graph result when all local rows already form a DAG."""
    confidence = {}
    local_role_probability = {}
    for child, row in local_rows.items():
        margins = (state_margins[child], identity_margins[child])
        confidence[child] = min(
            value for value in margins if not np.isnan(value)
        )
        local_role_probability[child] = _parent_role_probability(
            row, alternatives, depth_posterior
        )
    child_order = sorted(
        local_rows,
        key=lambda child: (
            -local_role_probability[child],
            -confidence[child],
            child,
        ),
    )
    selected = {}
    role_probabilities = {}
    for child in child_order:
        row = int(local_rows[child])
        selected[child] = row
        role_probabilities[child] = _parent_role_probability(
            row, alternatives, depth_posterior
        )
    return _GraphParentStateSelection(
        selected,
        frozenset(),
        role_probabilities,
    )


def _acyclic_parent_state_selection(
    alternatives: np.ndarray,
    states: np.ndarray,
    decision_scores: np.ndarray,
    by_child: Sequence[np.ndarray],
    local_rows: Mapping[int, int],
    state_margins: np.ndarray,
    identity_margins: np.ndarray,
    n_samples: int,
    local_search_passes: int,
    depth_posterior: Optional[np.ndarray] = None,
    downward_fallback: bool = False,
) -> _GraphParentStateSelection:
    """Choose a DAG without converting graph feasibility into parent evidence.

    A unique locally preferred row is retained whenever it is acyclic. Legacy
    mode may substitute another same-state identity with depth support, then M0.
    Combined-v1 may instead search deterministic unique finite winners from the
    local state downward (M2 to M1 to M0, or M1 to M0); it never promotes a
    child to a higher parent-count state. Every non-M0 combined candidate has
    already passed the calibrated coverage, explainability, exposure, and
    direction screens.
    """
    confidence = {}
    local_role_probability = {}
    for child, row in local_rows.items():
        margins = (state_margins[child], identity_margins[child])
        confidence[child] = min(
            value for value in margins if not np.isnan(value)
        )
        local_role_probability[child] = _parent_role_probability(
            row, alternatives, depth_posterior
        )
    child_order = sorted(
        local_rows,
        key=lambda child: (
            -local_role_probability[child],
            -confidence[child],
            child,
        ),
    )
    adjacency = [set() for _ in range(n_samples)]
    selected: dict[int, int] = {}
    direction_resolved: set[int] = set()
    role_probabilities: dict[int, float] = {}

    def can_add(row: int) -> bool:
        child = int(alternatives[row, 0])
        return not any(
            _path_exists(adjacency, child, parent)
            for parent in _observed_parents(alternatives, row)
        )

    def add(row: int, displaced_local: bool = False) -> None:
        child = int(alternatives[row, 0])
        for parent in _observed_parents(alternatives, row):
            adjacency[parent].add(child)
        selected[child] = row
        role_probabilities[child] = _parent_role_probability(
            row, alternatives, depth_posterior
        )
        if displaced_local and int(states[row]) != _ZERO_OBSERVED:
            direction_resolved.add(child)
        else:
            direction_resolved.discard(child)

    def remove(child: int) -> Optional[int]:
        row = selected.pop(child, None)
        role_probabilities.pop(child, None)
        direction_resolved.discard(child)
        if row is not None:
            for parent in _observed_parents(alternatives, row):
                adjacency[parent].discard(child)
        return row

    def best_feasible(child: int) -> tuple[Optional[int], bool]:
        local = local_rows.get(child)
        if local is not None and can_add(local):
            return local, False
        if local is not None and downward_fallback:
            local_state = int(states[local])
            for target_state in range(local_state, -1, -1):
                feasible = np.asarray([
                    int(row) for row in by_child[child]
                    if (
                        int(states[row]) == target_state
                        and np.isfinite(decision_scores[row])
                        and can_add(int(row))
                    )
                ], dtype=np.int64)
                winner, _ = _unique_finite_winner(
                    feasible, decision_scores
                )
                if winner is not None:
                    return winner, winner != local
            return None, False
        if local is not None and depth_posterior is not None:
            local_state = int(states[local])
            eligible = []
            for row in by_child[child]:
                row = int(row)
                if (
                    row != local
                    and int(states[row]) == local_state
                    and np.isfinite(decision_scores[row])
                    and can_add(row)
                ):
                    role_probability = _parent_role_probability(
                        row, alternatives, depth_posterior
                    )
                    tolerance = _contrast_tolerance(np.asarray(
                        (role_probability, 0.5), dtype=np.float64
                    ))
                    if role_probability > 0.5 + tolerance:
                        eligible.append(row)
            if eligible:
                winner, _ = _unique_finite_winner(
                    np.asarray(eligible, dtype=np.int64), decision_scores
                )
                if winner is not None:
                    return winner, True
        zero_rows = [
            int(row)
            for row in by_child[child]
            if int(states[row]) == _ZERO_OBSERVED
        ]
        if len(zero_rows) != 1:
            raise SmartEvidenceError(
                "each child must have exactly one zero-observed-parent row"
            )
        zero = zero_rows[0]
        return (
            (zero, local is not None and zero != local)
            if np.isfinite(decision_scores[zero])
            else (None, False)
        )

    for child in child_order:
        row, displaced = best_feasible(child)
        if row is not None:
            add(row, displaced)
    for _ in range(local_search_passes):
        changed = False
        for child in child_order:
            previous = remove(child)
            replacement, displaced = best_feasible(child)
            if replacement is not None:
                add(replacement, displaced)
            changed |= replacement != previous
        if not changed:
            break
    return _GraphParentStateSelection(
        selected,
        frozenset(direction_resolved),
        role_probabilities,
    )


def _graph_tie_conflict_children(
    alternatives: np.ndarray,
    local_rows: Mapping[int, int],
    state_margins: np.ndarray,
    identity_margins: np.ndarray,
    n_samples: int,
) -> frozenset[int]:
    """Find tied local rows that a DAG cannot resolve without arbitrariness.

    Cycles are peeled at their least-supported child configuration.  A unique
    weakest row can be left for the normal DAG optimizer to displace.  If two
    or more weakest rows are numerically tied, all are marked ambiguous and
    excluded from graph selection; selecting one by sample-array order would
    manufacture exact confidence from a graph constraint.
    """
    active = {int(child): int(row) for child, row in local_rows.items()}
    ambiguous: set[int] = set()

    def confidence(child: int) -> float:
        values = (
            float(state_margins[child]),
            float(identity_margins[child]),
        )
        return min(value for value in values if not np.isnan(value))

    def cyclic_components() -> list[frozenset[int]]:
        adjacency = [set() for _ in range(n_samples)]
        reverse = [set() for _ in range(n_samples)]
        for child, row in active.items():
            for parent in _observed_parents(alternatives, row):
                adjacency[parent].add(child)
                reverse[child].add(parent)

        def reachable(graph: Sequence[set[int]], start: int) -> set[int]:
            seen = {start}
            stack = [start]
            while stack:
                node = stack.pop()
                for neighbour in graph[node]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            return seen

        remaining = set(range(n_samples))
        components = []
        while remaining:
            start = min(remaining)
            component = reachable(adjacency, start) & reachable(reverse, start)
            remaining.difference_update(component)
            if len(component) > 1:
                components.append(frozenset(component))
        return components

    while True:
        components = cyclic_components()
        if not components:
            break
        removed = set()
        for component in components:
            implicated = [
                child
                for child, row in active.items()
                if child in component
                and any(
                    parent in component
                    for parent in _observed_parents(alternatives, row)
                )
            ]
            if not implicated:
                continue
            values = np.asarray(
                [confidence(child) for child in implicated], dtype=np.float64
            )
            minimum = float(np.min(values))
            if np.isfinite(minimum):
                finite = values[np.isfinite(values)]
                tolerance = _contrast_tolerance(finite)
                tied = [
                    child for child, value in zip(implicated, values)
                    if np.isfinite(value) and abs(float(value) - minimum) <= tolerance
                ]
            else:
                tied = [
                    child for child, value in zip(implicated, values)
                    if float(value) == minimum
                ]
            if len(tied) > 1:
                ambiguous.update(tied)
                removed.update(tied)
            else:
                removed.add(tied[0])
        if not removed:
            raise SmartEvidenceError("failed to peel a cyclic local pedigree")
        for child in removed:
            active.pop(child, None)
    return frozenset(ambiguous)



def _evaluate_parent_state_aggregate(
    aggregate_log_likelihoods: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
    base_priors: Sequence[float],
    prior_strength: float,
    prior_max_iterations: int,
    prior_tolerance: float,
    n_samples: int,
    local_search_passes: int,
    ancestry_depth_model: Optional[_AncestryDepthModel] = None,
    *,
    state_log_evidence_override: Optional[np.ndarray] = None,
    use_cohort_prior: bool = True,
    algorithm_mode: str = "b0",
    m1_over_m0_edge_gains: Optional[np.ndarray] = None,
    m2_over_first_m1_edge_gains: Optional[np.ndarray] = None,
    m2_over_second_m1_edge_gains: Optional[np.ndarray] = None,
    predictive_fold_count: int = 0,
    graph_downward_fallback: bool = False,
    identity_log_likelihoods_override: Optional[np.ndarray] = None,
    use_fixed_base_priors: bool = False,
) -> _ParentStateSelection:
    state_log_evidence = (
        _integrated_parent_state_log_evidence(
            aggregate_log_likelihoods, states, by_child, full_counts
        )
        if state_log_evidence_override is None
        else np.asarray(state_log_evidence_override, dtype=np.float64).copy()
    )
    if use_cohort_prior:
        fitted_parameters, loo_state_priors = (
            _fit_hierarchical_parent_state_prior(
                state_log_evidence,
                base_priors,
                prior_strength,
                prior_max_iterations,
                prior_tolerance,
            )
        )
    else:
        fitted_parameters = np.full(3, np.nan, dtype=np.float64)
        loo_state_priors = (
            np.tile(np.asarray(base_priors, dtype=np.float64), (n_samples, 1))
            if use_fixed_base_priors
            else np.full((n_samples, 3), 1.0 / 3.0)
        )
    components = _parent_state_score_components(
        aggregate_log_likelihoods,
        states,
        by_child,
        full_counts,
        loo_state_priors,
        state_log_evidence,
    )
    (
        state_log_evidence,
        state_scores,
        state_support,
        decision_scores,
    ) = components
    if identity_log_likelihoods_override is not None:
        identity_values = np.asarray(
            identity_log_likelihoods_override, dtype=np.float64
        )
        if identity_values.shape != aggregate_log_likelihoods.shape:
            raise SmartEvidenceError(
                "identity likelihood override must match aggregate rows"
            )
        decision_scores = np.full(len(identity_values), -np.inf, dtype=np.float64)
        for child, child_rows in enumerate(by_child):
            for state in range(3):
                rows = child_rows[states[child_rows] == state]
                finite = rows[np.isfinite(identity_values[rows])]
                if not len(finite) or not np.isfinite(state_scores[child, state]):
                    continue
                maximum = float(np.max(identity_values[finite]))
                decision_scores[finite] = (
                    state_scores[child, state]
                    + identity_values[finite]
                    - maximum
                )
    n_alternatives = len(alternatives)
    nan_edges = np.full(n_alternatives, np.nan, dtype=np.float64)
    m1_edges = (
        nan_edges.copy()
        if m1_over_m0_edge_gains is None
        else np.asarray(m1_over_m0_edge_gains, dtype=np.float64).copy()
    )
    m2_first_edges = (
        nan_edges.copy()
        if m2_over_first_m1_edge_gains is None
        else np.asarray(
            m2_over_first_m1_edge_gains, dtype=np.float64
        ).copy()
    )
    m2_second_edges = (
        nan_edges.copy()
        if m2_over_second_m1_edge_gains is None
        else np.asarray(
            m2_over_second_m1_edge_gains, dtype=np.float64
        ).copy()
    )
    if algorithm_mode == "b3":
        # The aggregate gate is identity-only. B3 state evidence/support was
        # already computed with fold-specific masks learned exclusively from
        # training chromosomes; full-data gains must not alter those scores.
        for child, child_rows in enumerate(by_child):
            for row in child_rows:
                row = int(row)
                state = int(states[row])
                if state == _ONE_OBSERVED:
                    gain = m1_edges[row]
                    eligible = bool(
                        np.isfinite(gain)
                        and gain > _contrast_tolerance(
                            np.asarray((gain, 0.0), dtype=np.float64)
                        )
                    )
                elif state == _TWO_OBSERVED:
                    gains = np.asarray((
                        m2_first_edges[row], m2_second_edges[row], 0.0
                    ), dtype=np.float64)
                    finite = gains[np.isfinite(gains)]
                    tolerance = _contrast_tolerance(finite)
                    eligible = bool(
                        np.isfinite(m2_first_edges[row])
                        and np.isfinite(m2_second_edges[row])
                        and m2_first_edges[row] > tolerance
                        and m2_second_edges[row] > tolerance
                    )
                else:
                    eligible = True
                if not eligible:
                    decision_scores[row] = -np.inf
    (
        local_states,
        local_rows,
        state_margins,
        identity_margins,
        unresolved_reasons,
    ) = _local_parent_state_winners(
        states, by_child, state_scores, decision_scores
    )
    local_row_vector = np.full(n_samples, -1, dtype=np.int64)
    for child, row in local_rows.items():
        local_row_vector[child] = row
    depth_posterior = (
        None
        if ancestry_depth_model is None
        else ancestry_depth_model.posterior
    )
    if is_acyclic_parent_rows(local_row_vector, alternatives):
        graph_tie_conflicts = frozenset()
        graph_selection = _acyclic_local_parent_state_selection(
            alternatives,
            local_rows,
            state_margins,
            identity_margins,
            depth_posterior,
        )
    else:
        # Cyclic local calls retain the exact tie peeling, deterministic
        # fallback, local search, and direction diagnostics used previously.
        graph_tie_conflicts = _graph_tie_conflict_children(
            alternatives,
            local_rows,
            state_margins,
            identity_margins,
            n_samples,
        )
        graph_eligible_rows = {
            child: row
            for child, row in local_rows.items()
            if child not in graph_tie_conflicts
        }
        graph_selection = _acyclic_parent_state_selection(
            alternatives,
            states,
            decision_scores,
            by_child,
            graph_eligible_rows,
            state_margins,
            identity_margins,
            n_samples,
            local_search_passes,
            depth_posterior=depth_posterior,
            downward_fallback=graph_downward_fallback,
        )
    return _ParentStateSelection(
        state_log_evidence=state_log_evidence,
        state_scores=state_scores,
        state_support=state_support,
        decision_scores=decision_scores,
        fitted_prior_parameters=fitted_parameters,
        loo_state_priors=loo_state_priors,
        local_states=local_states,
        local_rows=local_rows,
        graph_rows=graph_selection.rows,
        graph_tie_conflicts=graph_tie_conflicts,
        graph_direction_resolved_children=(
            graph_selection.direction_resolved_children
        ),
        graph_parent_role_probabilities=(
            graph_selection.selected_parent_role_probabilities
        ),
        ancestry_depth_model=ancestry_depth_model,
        state_margins=state_margins,
        identity_margins=identity_margins,
        unresolved_reasons=unresolved_reasons,
        m1_over_m0_edge_gains=m1_edges,
        m2_over_first_m1_edge_gains=m2_first_edges,
        m2_over_second_m1_edge_gains=m2_second_edges,
        predictive_fold_count=int(predictive_fold_count),
    )


def _edge_direction_probability(
    child: int,
    parent: int,
    depth_posterior: Optional[np.ndarray],
) -> float:
    """Return P(depth_parent < depth_child) under the fitted latent mixture."""
    if depth_posterior is None or depth_posterior.shape[1] < 2:
        return 0.0
    child_probability = depth_posterior[child]
    parent_probability = depth_posterior[parent]
    if np.sum(child_probability) <= 0.0 or np.sum(parent_probability) <= 0.0:
        return 0.0
    lower_parent = np.cumsum(parent_probability) - parent_probability
    return float(np.clip(np.dot(child_probability, lower_parent), 0.0, 1.0))


def _structure_pair_indices(
    alternatives: np.ndarray,
    states: np.ndarray,
    trios: np.ndarray,
) -> np.ndarray:
    """Map M2 alternative rows to the fixed-trio structure-count rows."""
    indices = np.full(len(alternatives), -1, dtype=np.int64)
    m2_rows = np.flatnonzero(states == _TWO_OBSERVED)
    if not len(m2_rows):
        return indices
    n_samples = int(max(np.max(alternatives), np.max(trios))) + 1
    trio_keys = (trios[:, 0] * n_samples + trios[:, 1]) * n_samples + trios[:, 2]
    order = np.argsort(trio_keys, kind="stable")
    sorted_keys = trio_keys[order]
    selected = alternatives[m2_rows]
    selected_keys = (selected[:, 0] * n_samples + selected[:, 1]) * n_samples + selected[:, 2]
    locations = np.searchsorted(sorted_keys, selected_keys)
    if np.any(locations >= len(sorted_keys)):
        raise SmartEvidenceError("M2 alternative is absent from the structure panel")
    if np.any(sorted_keys[locations] != selected_keys):
        raise SmartEvidenceError("M2 alternative is absent from the structure panel")
    indices[m2_rows] = order[locations]
    return indices


def _parent_state_structure_mask(
    weights: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    pair_indices: np.ndarray,
    edge_matched_by_contig: np.ndarray,
    edge_exposed_by_contig: np.ndarray,
    pair_explained_by_contig: np.ndarray,
    pair_exposed_by_contig: np.ndarray,
    structure_total_bins_by_contig: np.ndarray,
    depth_posterior: Optional[np.ndarray],
    settings: SmartPedigreeConfig,
    edge_exposure_presence_words: Optional[np.ndarray] = None,
    pair_exposure_presence_words: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply C/X/D as structural screens, never as likelihood multipliers."""
    selected_contigs = np.asarray(weights) > 0.0
    edge_matched = np.tensordot(weights, edge_matched_by_contig, axes=(0, 0))
    edge_exposed = np.tensordot(weights, edge_exposed_by_contig, axes=(0, 0))
    pair_explained = weights @ pair_explained_by_contig
    pair_exposed = weights @ pair_exposed_by_contig
    total_bins = float(np.dot(weights, structure_total_bins_by_contig))
    edge_coverage = np.divide(
        edge_matched,
        edge_exposed,
        out=np.full_like(edge_matched, np.nan),
        where=edge_exposed > 0.0,
    )
    pair_explainability = np.divide(
        pair_explained,
        pair_exposed,
        out=np.full_like(pair_explained, np.nan),
        where=pair_exposed > 0.0,
    )
    edge_exposed_fraction = np.divide(
        edge_exposed,
        total_bins,
        out=np.zeros_like(edge_exposed),
        where=total_bins > 0.0,
    )
    pair_exposed_fraction = np.divide(
        pair_exposed,
        total_bins,
        out=np.zeros_like(pair_exposed),
        where=total_bins > 0.0,
    )
    edge_exposed_contigs = (
        np.count_nonzero(
            (edge_exposed_by_contig > 0.0)
            & selected_contigs[:, None, None],
            axis=0,
        )
        if edge_exposure_presence_words is None
        else count_exposed_contigs(edge_exposure_presence_words, weights)
    )
    pair_exposed_contigs = (
        np.count_nonzero(
            (pair_exposed_by_contig > 0.0) & selected_contigs[:, None],
            axis=0,
        )
        if pair_exposure_presence_words is None
        else count_exposed_contigs(pair_exposure_presence_words, weights)
    )
    edge_exposure_ok = (
        (edge_exposed >= settings.parent_state_minimum_edge_exposed_bins)
        & (edge_exposed_fraction >= settings.parent_state_minimum_exposed_fraction)
        & (edge_exposed_contigs >= settings.parent_state_minimum_exposed_contigs)
    )
    pair_exposure_ok = (
        (pair_exposed >= settings.parent_state_minimum_pair_exposed_bins)
        & (pair_exposed_fraction >= settings.parent_state_minimum_exposed_fraction)
        & (pair_exposed_contigs >= settings.parent_state_minimum_exposed_contigs)
    )

    n_samples = edge_coverage.shape[0]
    direction = np.zeros((n_samples, n_samples), dtype=np.float64)
    depth_available = bool(
        depth_posterior is not None and depth_posterior.shape[1] >= 2
    )
    if depth_available:
        lower_depth_probability = (
            np.cumsum(depth_posterior, axis=1) - depth_posterior
        )
        direction = np.clip(
            depth_posterior @ lower_depth_probability.T, 0.0, 1.0
        )

    m0_rows = np.flatnonzero(states == _ZERO_OBSERVED)
    m1_rows = np.flatnonzero(states == _ONE_OBSERVED)
    m2_rows = np.flatnonzero(states == _TWO_OBSERVED)
    m1_children = alternatives[m1_rows, 0]
    m1_parents = alternatives[m1_rows, 1]
    m2_children = alternatives[m2_rows, 0]
    m2_first = alternatives[m2_rows, 1]
    m2_second = alternatives[m2_rows, 2]
    m2_pairs = pair_indices[m2_rows]

    exposure_testable = np.zeros(len(alternatives), dtype=np.bool_)
    exposure_testable[m0_rows] = True
    exposure_testable[m1_rows] = edge_exposure_ok[m1_children, m1_parents]
    exposure_testable[m2_rows] = (
        edge_exposure_ok[m2_children, m2_first]
        & edge_exposure_ok[m2_children, m2_second]
        & (m2_pairs >= 0)
        & pair_exposure_ok[m2_pairs]
    )
    child_evaluable = np.zeros(n_samples, dtype=np.bool_)
    np.logical_or.at(
        child_evaluable,
        m1_children,
        exposure_testable[m1_rows],
    )
    if not depth_available:
        child_evaluable[:] = False

    eligible = np.zeros(len(alternatives), dtype=np.bool_)
    eligible[m0_rows] = True
    eligible[m1_rows] = (
        exposure_testable[m1_rows]
        & (
            edge_coverage[m1_children, m1_parents]
            >= settings.parent_state_minimum_edge_coverage
        )
        & (
            direction[m1_children, m1_parents]
            >= settings.parent_state_minimum_direction_probability
        )
    )
    eligible[m2_rows] = (
        exposure_testable[m2_rows]
        & (
            edge_coverage[m2_children, m2_first]
            >= settings.parent_state_minimum_edge_coverage
        )
        & (
            edge_coverage[m2_children, m2_second]
            >= settings.parent_state_minimum_edge_coverage
        )
        & (
            direction[m2_children, m2_first]
            >= settings.parent_state_minimum_direction_probability
        )
        & (
            direction[m2_children, m2_second]
            >= settings.parent_state_minimum_direction_probability
        )
        & (
            pair_explainability[m2_pairs]
            >= settings.parent_state_minimum_pair_explainability
        )
    )
    return (
        exposure_testable,
        eligible,
        child_evaluable,
        edge_coverage,
        pair_explainability,
        direction,
    )


def _structure_state_and_identity_aggregates(
    raw_aggregate: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    exposure_testable: np.ndarray,
    structure_eligible: np.ndarray,
    *,
    direction_available: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build separate state-marginal and selectable-identity score rows.

    Untestable alternatives contribute the child's M0 likelihood to the state
    mean but remain impossible identities. Testable C/X/D failures contribute
    no state mass and also remain impossible identities.
    """
    raw = np.asarray(raw_aggregate, dtype=np.float64)
    state_scores = raw.copy()
    identity_scores = raw.copy()
    nonzero = states != _ZERO_OBSERVED
    testable = np.asarray(exposure_testable, dtype=np.bool_).copy()
    if not direction_available:
        testable[nonzero] = False
    eligible = np.asarray(structure_eligible, dtype=np.bool_)
    m0_rows = np.flatnonzero(states == _ZERO_OBSERVED)
    n_samples = int(np.max(alternatives[:, 0])) + 1
    m0_by_child = np.full(n_samples, -1, dtype=np.int64)
    m0_by_child[alternatives[m0_rows, 0]] = m0_rows
    if np.any(m0_by_child < 0):
        raise SmartEvidenceError(
            "every structurally evaluated child requires exactly one M0 row"
        )
    underexposed = nonzero & ~testable
    contradicted = nonzero & testable & ~eligible
    children = alternatives[underexposed, 0]
    state_scores[underexposed] = raw[m0_by_child[children]]
    state_scores[contradicted] = -np.inf
    identity_scores[nonzero & ~eligible] = -np.inf
    return state_scores, identity_scores


def _prepare_parent_state_weighted_contigs(
    contig_log_likelihoods: np.ndarray,
    contig_weights: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    settings: SmartPedigreeConfig,
    ancestry_depth_model: _AncestryDepthModel,
    structure_pair_indices: Optional[np.ndarray],
    edge_matched_by_contig: Optional[np.ndarray],
    edge_exposed_by_contig: Optional[np.ndarray],
    pair_explained_by_contig: Optional[np.ndarray],
    pair_exposed_by_contig: Optional[np.ndarray],
    structure_total_bins_by_contig: Optional[np.ndarray],
    edge_exposure_presence_words: Optional[np.ndarray],
    pair_exposure_presence_words: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare prior-independent full-data likelihood and C/X/D rows."""
    weights = np.asarray(contig_weights, dtype=np.float64)
    aggregate = weights @ contig_log_likelihoods
    structure_values = (
        structure_pair_indices,
        edge_matched_by_contig,
        edge_exposed_by_contig,
        pair_explained_by_contig,
        pair_exposed_by_contig,
        structure_total_bins_by_contig,
    )
    if any(value is None for value in structure_values):
        raise SmartEvidenceError(
            "combined_v1 requires per-contig parenthood structure evidence"
        )
    (
        exposure_testable,
        structure_eligible,
        _,
        _,
        _,
        _,
    ) = _parent_state_structure_mask(
        weights,
        alternatives,
        states,
        structure_pair_indices,
        edge_matched_by_contig,
        edge_exposed_by_contig,
        pair_explained_by_contig,
        pair_exposed_by_contig,
        structure_total_bins_by_contig,
        ancestry_depth_model.posterior,
        settings,
        edge_exposure_presence_words,
        pair_exposure_presence_words,
    )
    return _structure_state_and_identity_aggregates(
        aggregate,
        alternatives,
        states,
        exposure_testable,
        structure_eligible,
        direction_available=ancestry_depth_model.posterior.shape[1] >= 2,
    )


def _evaluate_parent_state_weighted_contigs(
    contig_log_likelihoods: np.ndarray,
    contig_weights: np.ndarray,
    contig_information_weights: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
    settings: SmartPedigreeConfig,
    n_samples: int,
    ancestry_depth_model: Optional[_AncestryDepthModel] = None,
    base_priors: Optional[Sequence[float]] = None,
    structure_pair_indices: Optional[np.ndarray] = None,
    edge_matched_by_contig: Optional[np.ndarray] = None,
    edge_exposed_by_contig: Optional[np.ndarray] = None,
    pair_explained_by_contig: Optional[np.ndarray] = None,
    pair_exposed_by_contig: Optional[np.ndarray] = None,
    structure_total_bins_by_contig: Optional[np.ndarray] = None,
    edge_exposure_presence_words: Optional[np.ndarray] = None,
    pair_exposure_presence_words: Optional[np.ndarray] = None,
    prepared_aggregates: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> _ParentStateSelection:
    """Evaluate the combined method using internal B1 likelihood evidence."""
    if ancestry_depth_model is None:
        raise SmartEvidenceError(
            "combined_v1 requires an ancestry-depth model in every evaluation"
        )
    if prepared_aggregates is None:
        aggregate, identity_aggregate = (
            _prepare_parent_state_weighted_contigs(
                contig_log_likelihoods,
                contig_weights,
                alternatives,
                states,
                settings,
                ancestry_depth_model,
                structure_pair_indices,
                edge_matched_by_contig,
                edge_exposed_by_contig,
                pair_explained_by_contig,
                pair_exposed_by_contig,
                structure_total_bins_by_contig,
                edge_exposure_presence_words,
                pair_exposure_presence_words,
            )
        )
    else:
        aggregate, identity_aggregate = prepared_aggregates
    effective_priors = (
        settings.parent_state_priors if base_priors is None else base_priors
    )
    return _evaluate_parent_state_aggregate(
        aggregate,
        alternatives,
        states,
        by_child,
        full_counts,
        effective_priors,
        settings.parent_state_prior_strength,
        settings.parent_state_prior_max_iterations,
        settings.parent_state_prior_tolerance,
        n_samples,
        settings.dag_local_search_passes,
        ancestry_depth_model,
        use_cohort_prior=False,
        algorithm_mode=_PARENT_STATE_LIKELIHOOD,
        use_fixed_base_priors=True,
        graph_downward_fallback=True,
        identity_log_likelihoods_override=identity_aggregate,
    )


def _parent_state_frame(
    sample_ids: Sequence[Any],
    alternatives: np.ndarray,
    states: np.ndarray,
    rows_by_child: Mapping[int, Optional[int]],
    state_by_child: Mapping[int, int],
    status_by_child: Mapping[int, str],
) -> pd.DataFrame:
    rows = []
    for child, sample in enumerate(sample_ids):
        selected = rows_by_child.get(child)
        state = state_by_child.get(child)
        first = second = None
        if selected is not None:
            first_index = int(alternatives[selected, 1])
            second_index = int(alternatives[selected, 2])
            first = None if first_index < 0 else sample_ids[first_index]
            second = None if second_index < 0 else sample_ids[second_index]
        rows.append({
            "Sample": sample,
            "Generation": "Unknown",
            "Parent1": first,
            "Parent2": second,
            "ParentState": (
                "unresolved" if state is None else _PARENT_STATE_NAMES[state]
            ),
            "ObservedParentCount": (
                np.nan if state is None else int(state)
            ),
            "InferenceStatus": status_by_child.get(child, "unresolved"),
        })
    return pd.DataFrame(rows)


def _configuration_support_text(
    child_rows: np.ndarray,
    counts: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    sample_ids: Sequence[Any],
    coverage: float,
) -> str:
    total = float(np.sum(counts[child_rows]))
    if total <= 0.0:
        return ""
    ordered = child_rows[np.argsort(-counts[child_rows], kind="stable")]
    pieces = []
    cumulative = 0.0
    for row in ordered:
        probability = float(counts[row] / total)
        state = int(states[row])
        parents = [
            str(sample_ids[int(parent)])
            for parent in alternatives[row, 1:]
            if int(parent) >= 0
        ]
        label = "+".join(parents) if parents else "external+external"
        pieces.append(
            f"{_PARENT_STATE_NAMES[state]}:{label}:{probability:.3f}"
        )
        cumulative += probability
        if cumulative >= coverage:
            break
    return ";".join(pieces)


def _coverage_rows_from_scores(
    rows: np.ndarray,
    scores: np.ndarray,
    coverage: float,
) -> tuple[int, ...]:
    """Small deterministic configuration set carrying requested score mass."""
    candidate_rows = np.asarray(rows, dtype=np.int64)
    probabilities = _softmax_finite(scores[candidate_rows])
    positive = probabilities > 0.0
    candidate_rows = candidate_rows[positive]
    probabilities = probabilities[positive]
    if not len(candidate_rows):
        return ()
    order = np.lexsort((candidate_rows, -probabilities))
    selected = []
    cumulative = 0.0
    for position in order:
        selected.append(int(candidate_rows[position]))
        cumulative += float(probabilities[position])
        if cumulative >= coverage:
            break
    return tuple(selected)


def _evidence_parent_support_sets(
    child_rows: np.ndarray,
    selected_state: Optional[int],
    selected_row: Optional[int],
    alternatives: np.ndarray,
    states: np.ndarray,
    decision_scores: np.ndarray,
    coverage: float,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Factor a high-evidence M1/M2 configuration set into parent sides.

    M2 parents are unordered. If all supported pairs share one parent, that
    singleton becomes the fixed side and the other side is the union of its
    alternatives. Otherwise, sides are defined conditionally by holding each
    parent of the unique leading pair fixed. The final tuple contains the
    supported configuration rows, preserving non-factorable ambiguity.
    """
    if selected_state not in {_ONE_OBSERVED, _TWO_OBSERVED}:
        return (), (), ()
    state_rows = child_rows[states[child_rows] == selected_state]
    support_rows = _coverage_rows_from_scores(
        state_rows, decision_scores, coverage
    )
    if not support_rows:
        return (), (), ()
    if selected_state == _ONE_OBSERVED:
        parents = tuple(dict.fromkeys(
            int(alternatives[row, 1]) for row in support_rows
        ))
        return parents, (), support_rows

    pairs = [
        frozenset((
            int(alternatives[row, 1]), int(alternatives[row, 2])
        ))
        for row in support_rows
    ]
    if len(pairs) == 1:
        row = support_rows[0]
        return (
            (int(alternatives[row, 1]),),
            (int(alternatives[row, 2]),),
            support_rows,
        )
    common = set(pairs[0])
    for pair in pairs[1:]:
        common.intersection_update(pair)
    if len(common) == 1:
        fixed = next(iter(common))
        variable = tuple(sorted({
            parent for pair in pairs for parent in pair if parent != fixed
        }))
        if selected_row is not None and int(alternatives[selected_row, 1]) == fixed:
            return (fixed,), variable, support_rows
        return variable, (fixed,), support_rows
    return (), (), support_rows


def _sample_set_text(indices: Sequence[int], sample_ids: Sequence[Any]) -> str:
    return "{" + ",".join(str(sample_ids[index]) for index in indices) + "}"


_SMART_BOOTSTRAP_SHARED: dict[str, Any] = {}
_SMART_BOOTSTRAP_SHM_REFS: list[Any] = []
_SMART_BOOTSTRAP_MIN_WORK_ITEMS = 100_000


def _smart_bootstrap_worker_count(
    n_alternatives: int,
    bootstrap_replicates: int,
    cpu_budget: int,
) -> int:
    """Choose one process per usable CPU for a substantial bootstrap.

    Replicates are independent tasks, and their dominant selection work scans
    the candidate alternatives.  Gate pool startup on that combined work
    rather than on candidate count alone, then consume the complete caller-
    bounded CPU budget when enough replicates are available.
    """
    work_items = int(n_alternatives) * int(bootstrap_replicates)
    if (
        cpu_budget <= 1
        or bootstrap_replicates < 32
        or work_items < _SMART_BOOTSTRAP_MIN_WORK_ITEMS
    ):
        return 1
    return min(int(cpu_budget), int(bootstrap_replicates))


def _init_smart_bootstrap_worker(shared: Mapping[str, Any]) -> None:
    """Attach read-only bootstrap arrays once in each forkserver worker."""
    global _SMART_BOOTSTRAP_SHARED, _SMART_BOOTSTRAP_SHM_REFS
    for handle in _SMART_BOOTSTRAP_SHM_REFS:
        try:
            handle.close()
        except Exception:
            pass
    _SMART_BOOTSTRAP_SHM_REFS = []
    _SMART_BOOTSTRAP_SHARED = {}
    for key, value in shared.items():
        if isinstance(value, Mapping) and "shm_name" in value:
            handle, array = attach_shared_array(value)
            _SMART_BOOTSTRAP_SHM_REFS.append(handle)
            _SMART_BOOTSTRAP_SHARED[key] = array
        else:
            _SMART_BOOTSTRAP_SHARED[key] = value




def _evaluate_smart_bootstrap_chunk(
    shared: Mapping[str, Any],
    multiplicities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Evaluate a deterministic chunk of chromosome bootstrap replicates."""
    alternatives = shared["alternatives"]
    states = shared["states"]
    contig_log_likelihoods = shared["contig_log_likelihoods"]
    by_child = shared["by_child"]
    full_counts = shared["full_counts"]
    junction_matrix = shared["junction_matrix"]
    callable_matrix = shared["callable_matrix"]
    n_samples = int(shared["n_samples"])
    n_replicates = len(multiplicities)
    local_rows = np.full((n_replicates, n_samples), -1, dtype=np.int64)
    graph_rows = np.full((n_replicates, n_samples), -1, dtype=np.int64)
    local_states = np.full((n_replicates, n_samples), -1, dtype=np.int8)
    depth_refits = 0

    def evaluate(
        weights: np.ndarray,
        depth_model: Optional[_AncestryDepthModel],
    ) -> _ParentStateSelection:
        return _evaluate_parent_state_weighted_contigs(
            contig_log_likelihoods,
            weights,
            shared["contig_information_weights"],
            alternatives,
            states,
            by_child,
            full_counts,
            shared["settings"],
            n_samples,
            depth_model,
            structure_pair_indices=shared["structure_pair_indices"],
            edge_matched_by_contig=shared["edge_matched_by_contig"],
            edge_exposed_by_contig=shared["edge_exposed_by_contig"],
            pair_explained_by_contig=shared["pair_explained_by_contig"],
            pair_exposed_by_contig=shared["pair_exposed_by_contig"],
            structure_total_bins_by_contig=shared[
                "structure_total_bins_by_contig"
            ],
            edge_exposure_presence_words=shared.get(
                "edge_exposure_presence_words"
            ),
            pair_exposure_presence_words=shared.get(
                "pair_exposure_presence_words"
            ),
        )

    for replicate, multiplicity in enumerate(multiplicities):
        depth_model = _fit_ancestry_depth_model(
            multiplicity @ junction_matrix,
            multiplicity @ callable_matrix,
            int(shared["bootstrap_seed"]),
        )
        selection = evaluate(multiplicity, depth_model)
        depth_refits += 1
        for child, state in selection.local_states.items():
            local_states[replicate, child] = state
        for child, row in selection.local_rows.items():
            local_rows[replicate, child] = row
        for child, row in selection.graph_rows.items():
            graph_rows[replicate, child] = row
    return (
        local_rows,
        graph_rows,
        local_states,
        depth_refits,
    )


def _smart_bootstrap_worker(
    multiplicities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Module-scope forkserver callback for one bootstrap chunk."""
    return _evaluate_smart_bootstrap_chunk(
        _SMART_BOOTSTRAP_SHARED, multiplicities
    )


def _accumulate_smart_bootstrap_chunk(
    chunk: tuple[np.ndarray, np.ndarray, np.ndarray, int],
    alternatives: np.ndarray,
    local_configuration_counts: np.ndarray,
    graph_configuration_counts: np.ndarray,
    local_state_counts: np.ndarray,
    _graph_state_counts: Optional[np.ndarray],
    local_parent_counts: np.ndarray,
    graph_parent_counts: np.ndarray,
) -> int:
    """Reduce one worker result with order-independent integer additions."""
    local_rows, graph_rows, local_states, depth_refits = chunk
    # Graph-state counts have always been an unused compatibility argument;
    # leave them untouched while compiling every count that is consumed.
    accumulate_bootstrap_counts_into(
        local_rows,
        graph_rows,
        local_states,
        alternatives,
        None,
        local_configuration_counts,
        graph_configuration_counts,
        local_state_counts,
        None,
        local_parent_counts,
        graph_parent_counts,
    )
    return int(depth_refits)


def _run_parent_state_bootstraps(
    contig_log_likelihoods: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
    junction_matrix: Optional[np.ndarray],
    callable_matrix: Optional[np.ndarray],
    settings: SmartPedigreeConfig,
    n_workers: Optional[int],
    local_configuration_counts: np.ndarray,
    graph_configuration_counts: np.ndarray,
    local_state_counts: np.ndarray,
    graph_state_counts: np.ndarray,
    local_parent_counts: np.ndarray,
    graph_parent_counts: np.ndarray,
    *,
    contig_information_weights: Optional[np.ndarray] = None,
    structure_pair_indices: Optional[np.ndarray] = None,
    edge_matched_by_contig: Optional[np.ndarray] = None,
    edge_exposed_by_contig: Optional[np.ndarray] = None,
    pair_explained_by_contig: Optional[np.ndarray] = None,
    pair_exposed_by_contig: Optional[np.ndarray] = None,
    structure_total_bins_by_contig: Optional[np.ndarray] = None,
    edge_exposure_presence_words: Optional[np.ndarray] = None,
    pair_exposure_presence_words: Optional[np.ndarray] = None,
) -> tuple[int, int]:
    """Run fixed-seed bootstraps serially or in a shared-memory pool."""
    n_contigs = contig_log_likelihoods.shape[0]
    if contig_information_weights is None:
        information_weights = np.ones(n_contigs, dtype=np.float64)
    else:
        information_weights = np.asarray(
            contig_information_weights, dtype=np.float64
        )
        if information_weights.shape != (n_contigs,):
            raise SmartEvidenceError("contig information weights must match contigs")
    rng = np.random.default_rng(settings.bootstrap_seed)
    multiplicities = np.empty(
        (settings.bootstrap_replicates, n_contigs), dtype=np.float64
    )
    for replicate in range(settings.bootstrap_replicates):
        draws = rng.integers(0, n_contigs, size=n_contigs)
        multiplicities[replicate] = np.bincount(
            draws, minlength=n_contigs
        ).astype(np.float64)

    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = os.cpu_count() or 1
    capacity = int(numba.config.NUMBA_NUM_THREADS)
    if n_workers is None:
        cpu_budget = min(available_cpus, capacity)
    else:
        if int(n_workers) != n_workers or n_workers < 1:
            raise SmartEvidenceError("n_workers must be a positive integer")
        cpu_budget = min(int(n_workers), available_cpus, capacity)
    worker_count = _smart_bootstrap_worker_count(
        len(alternatives),
        settings.bootstrap_replicates,
        cpu_budget,
    )

    ordinary_shared = {
        "by_child": tuple(np.asarray(rows, dtype=np.int64) for rows in by_child),
        "settings": settings,
        "n_samples": len(by_child),
        "bootstrap_seed": settings.bootstrap_seed,
        "contig_information_weights": information_weights,
    }
    depth_refits = 0
    if worker_count == 1:
        shared = {
            **ordinary_shared,
            "contig_log_likelihoods": contig_log_likelihoods,
            "alternatives": alternatives,
            "states": states,
            "full_counts": full_counts,
            "junction_matrix": junction_matrix,
            "callable_matrix": callable_matrix,
            "structure_pair_indices": structure_pair_indices,
            "edge_matched_by_contig": edge_matched_by_contig,
            "edge_exposed_by_contig": edge_exposed_by_contig,
            "pair_explained_by_contig": pair_explained_by_contig,
            "pair_exposed_by_contig": pair_exposed_by_contig,
            "structure_total_bins_by_contig": structure_total_bins_by_contig,
            "edge_exposure_presence_words": edge_exposure_presence_words,
            "pair_exposure_presence_words": pair_exposure_presence_words,
        }
        results = (_evaluate_smart_bootstrap_chunk(shared, multiplicities),)
        for chunk in results:
            depth_refits += _accumulate_smart_bootstrap_chunk(
                chunk,
                alternatives,
                local_configuration_counts,
                graph_configuration_counts,
                local_state_counts,
                graph_state_counts,
                local_parent_counts,
                graph_parent_counts,
            )
        return worker_count, depth_refits

    handles = []
    shared = dict(ordinary_shared)
    for key, array in (
        ("contig_log_likelihoods", contig_log_likelihoods),
        ("alternatives", alternatives),
        ("states", states),
        ("full_counts", full_counts),
        ("junction_matrix", junction_matrix),
        ("callable_matrix", callable_matrix),
        ("structure_pair_indices", structure_pair_indices),
        ("edge_matched_by_contig", edge_matched_by_contig),
        ("edge_exposed_by_contig", edge_exposed_by_contig),
        ("pair_explained_by_contig", pair_explained_by_contig),
        ("pair_exposed_by_contig", pair_exposed_by_contig),
        ("structure_total_bins_by_contig", structure_total_bins_by_contig),
        ("edge_exposure_presence_words", edge_exposure_presence_words),
        ("pair_exposure_presence_words", pair_exposure_presence_words),
    ):
        if array is None:
            shared[key] = None
        else:
            try:
                handle, metadata = create_shared_array(array)
            except BaseException:
                with shared_memory_cleanup(handles):
                    pass
                raise
            handles.append(handle)
            shared[key] = metadata

    chunk_size = max(
        1,
        int(math.ceil(
            settings.bootstrap_replicates / float(worker_count * 4)
        )),
    )
    tasks = [
        np.ascontiguousarray(multiplicities[start:start + chunk_size])
        for start in range(0, settings.bootstrap_replicates, chunk_size)
    ]
    with shared_memory_cleanup(handles), safe_forkserver_pool(
        worker_count,
        initializer=_init_smart_bootstrap_worker,
        initargs=(shared,),
    ) as pool:
        for chunk in pool.imap_unordered(
            _smart_bootstrap_worker, tasks, chunksize=1
        ):
            depth_refits += _accumulate_smart_bootstrap_chunk(
                chunk,
                alternatives,
                local_configuration_counts,
                graph_configuration_counts,
                local_state_counts,
                graph_state_counts,
                local_parent_counts,
                graph_parent_counts,
            )
    return worker_count, depth_refits


def infer_from_parent_state_evidence(
    evidence: Sequence[SmartParentStateEvidence],
    sample_ids: Sequence[Any],
    config: Optional[SmartPedigreeConfig] = None,
    *,
    parent_eligibility: Optional[SmartParentEligibility | Mapping[str, Any]] = None,
    ancestry_junction_counts: Optional[np.ndarray] = None,
    ancestry_callable_haplotype_bins: Optional[np.ndarray] = None,
    n_workers: Optional[int] = None,
) -> PedigreeResult:
    """Infer a DAG from comparable 0/1/2-observed-parent likelihoods.

    State evidence is integrated over candidate identities only after contig
    log likelihoods have been summed. The sole supported combined method uses
    fixed B1 state priors, C/X/D structural screening, and ancestry-depth
    direction evidence; other analysed children do not alter a focal child's
    parent-count prior. Parent-count, conditional identity, and graph support
    are resampled separately so the DAG cannot create biological confidence.
    Every contig must supply parenthood-structure counts. Per-contig ancestry
    junction and callability matrices are also required, must have shape
    ``(len(evidence), len(sample_ids))`, and must be supplied together.
    """
    settings = (config or SmartPedigreeConfig()).validated()
    samples = list(sample_ids)
    n_samples = len(samples)
    if n_samples < 3 or len(set(samples)) != n_samples:
        raise SmartEvidenceError(
            "sample_ids must contain at least three unique IDs"
        )
    eligibility = _resolve_parent_eligibility(parent_eligibility, samples)
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
    ) = _canonical_parent_state_evidence(evidence, n_samples, eligibility)
    contig_information_weights = np.ceil(
        markers / settings.markers_per_information_block
    ).astype(np.float64)
    edge_exposure_presence_words = (
        None
        if edge_exposed_by_contig is None
        else pack_contig_presence(edge_exposed_by_contig)
    )
    pair_exposure_presence_words = (
        None
        if pair_exposed_by_contig is None
        else pack_contig_presence(pair_exposed_by_contig)
    )
    if (
        ancestry_junction_counts is None
        or ancestry_callable_haplotype_bins is None
    ):
        if not (
            ancestry_junction_counts is None
            and ancestry_callable_haplotype_bins is None
        ):
            raise SmartEvidenceError(
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
            raise SmartEvidenceError(
                "per-contig ancestry junction counts and callability must be "
                f"finite, non-negative arrays with shape {expected_shape}"
            )
    if edge_matched_by_contig is None:
        raise SmartEvidenceError(
            "combined_v1 requires parenthood structure evidence on every contig"
        )
    if junction_matrix is None:
        raise SmartEvidenceError(
            "combined_v1 requires ancestry junction and callability evidence"
        )
    if len(contig_names) < settings.parent_state_minimum_exposed_contigs:
        raise SmartEvidenceError(
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
    ) = _parent_state_alternatives(
        trios,
        zero,
        one,
        two,
        settings.parent_state_contamination_probability,
        eligibility,
    )

    structure_pair_indices = _structure_pair_indices(
        alternatives, states, trios
    )

    def evaluate(
        weights: np.ndarray,
        base_priors: Sequence[float] = settings.parent_state_priors,
        depth_model: Optional[_AncestryDepthModel] = None,
        prepared_aggregates: Optional[
            tuple[np.ndarray, np.ndarray]
        ] = None,
    ) -> _ParentStateSelection:
        return _evaluate_parent_state_weighted_contigs(
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
            prepared_aggregates=prepared_aggregates,
        )

    full_weights = np.ones(len(contig_names), dtype=np.float64)
    full_aggregate = np.sum(contig_log_likelihoods, axis=0)
    total_junction_counts = np.sum(junction_matrix, axis=0)
    total_callable_bins = np.sum(callable_matrix, axis=0)
    full_depth_model = _fit_ancestry_depth_model(
        total_junction_counts,
        total_callable_bins,
        settings.bootstrap_seed,
    )
    full_prepared_aggregates = _prepare_parent_state_weighted_contigs(
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
    )
    full_selection = evaluate(
        full_weights, depth_model=full_depth_model,
        prepared_aggregates=full_prepared_aggregates,
    )

    (
        full_exposure_testable,
        full_structure_eligible,
        full_structure_child_evaluable,
        full_edge_coverage,
        full_pair_explainability,
        full_edge_direction,
    ) = _parent_state_structure_mask(
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
        edge_exposure_presence_words,
        pair_exposure_presence_words,
    )
    informative = np.zeros(
        (len(contig_names), n_samples), dtype=np.bool_
    )
    base_log_prior = np.zeros(3, dtype=np.float64)
    state_indices = np.arange(3, dtype=np.int64)
    for contig_index in range(len(contig_names)):
        contig_state_evidence = _integrated_parent_state_log_evidence(
            contig_log_likelihoods[contig_index],
            states,
            by_child,
            full_counts,
        ) + base_log_prior
        for child in range(n_samples):
            winner, _ = _unique_finite_winner(
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
        selection: _ParentStateSelection,
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

    bootstrap_worker_count, bootstrap_depth_refits = (
        _run_parent_state_bootstraps(
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
    n_loco = 0
    if len(contig_names) > 1:
        for omitted in range(len(contig_names)):
            loco_weights = full_weights.copy()
            loco_weights[omitted] = 0.0
            loco_depth_model = _fit_ancestry_depth_model(
                total_junction_counts
                - junction_matrix[omitted],
                total_callable_bins
                - callable_matrix[omitted],
                settings.bootstrap_seed,
            )
            selection = evaluate(loco_weights, depth_model=loco_depth_model)
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
        m1_rows = child_rows[states[child_rows] == _ONE_OBSERVED]
        m2_rows = child_rows[states[child_rows] == _TWO_OBSERVED]
        best_m1_row, best_m1_margin = _unique_finite_winner(
            m1_rows, full_selection.decision_scores
        )
        best_m2_row, best_m2_margin = _unique_finite_winner(
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
            else _parent_role_probability(
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
            local_state != _ZERO_OBSERVED
            or all(
                selection.local_states.get(child) == _ZERO_OBSERVED
                for _, selection in sensitivity_runs
            )
        )
        m0_sensitivity_graph_agreement = bool(
            local_state != _ZERO_OBSERVED
            or all(
                (
                    selection.graph_rows.get(child) is not None
                    and int(states[selection.graph_rows[child]])
                    == _ZERO_OBSERVED
                )
                for _, selection in sensitivity_runs
            )
        )
        m0_sensitivity_tier_veto = bool(
            local_state == _ZERO_OBSERVED
            and not m0_sensitivity_state_agreement
        )

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
                    f"selected_{_PARENT_STATE_NAMES[graph_state]}"
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
                    f"{label}_supported_{_PARENT_STATE_NAMES[local_state]}"
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
        support_text = _configuration_support_text(
            child_rows,
            local_configuration_counts,
            alternatives,
            states,
            samples,
            settings.support_set_coverage,
        )
        parent1_set, parent2_set, configuration_set_rows = (
            _evidence_parent_support_sets(
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

        structure_selected_eligible = bool(
            local_row is not None and full_structure_eligible[local_row]
        )
        structure_parent1_coverage = np.nan
        structure_parent2_coverage = np.nan
        structure_parent1_direction = np.nan
        structure_parent2_direction = np.nan
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
            "ParentStateAlgorithmMode": _PARENT_STATE_LIKELIHOOD,
            "ParentStateStructureMode": _PARENT_STATE_METHOD,
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
            "StructureSelectedRowEligible": structure_selected_eligible,
            "StructureParent1Coverage": structure_parent1_coverage,
            "StructureParent2Coverage": structure_parent2_coverage,
            "StructurePairExplainability": structure_pair_explainability,
            "StructureParent1DirectionProbability": structure_parent1_direction,
            "StructureParent2DirectionProbability": structure_parent2_direction,
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
                None if graph_state is None else _PARENT_STATE_NAMES[graph_state]
            ),
            "LocalWinnerParentState": (
                None if local_state is None else _PARENT_STATE_NAMES[local_state]
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
                tier_a_identity_pass and local_state == _TWO_OBSERVED
            ),
            "TierBExactPair": bool(
                tier_b_identity_pass and local_state == _TWO_OBSERVED
            ),
            "TierAParent1": tier_a_flags[0],
            "TierAParent2": tier_a_flags[1],
            "TierBParent1": tier_b_flags[0],
            "TierBParent2": tier_b_flags[1],
            "PairSupportSet": support_text,
            "ConfigurationSupportSet": support_text,
            "Parent1CandidateSet": _sample_set_text(parent1_set, samples),
            "Parent2CandidateSet": _sample_set_text(parent2_set, samples),
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
                None if graph_state is None else _PARENT_STATE_NAMES[graph_state]
            ),
            "LocalWinnerParentState": (
                None if local_state is None else _PARENT_STATE_NAMES[local_state]
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

        two_rows = child_rows[states[child_rows] == _TWO_OBSERVED]
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
        for row in child_rows[states[child_rows] != _ZERO_OBSERVED]:
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

    complete_frame = _parent_state_frame(
        samples,
        alternatives,
        states,
        complete_rows,
        complete_states,
        complete_status,
    )
    tier_a_frame = _parent_state_frame(
        samples,
        alternatives,
        states,
        tier_a_rows,
        tier_a_states,
        tier_a_status,
    )
    tier_b_frame = _parent_state_frame(
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
            status = "unresolved_below_tier_b_state_support"
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
                "unresolved" if state is None else _PARENT_STATE_NAMES[state]
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
    result = PedigreeResult(
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
    result.smart_mode = True
    result.smart_parent_state_model = True
    result.smart_pair_only_compatibility_mode = False
    result.smart_bootstrap_worker_count = bootstrap_worker_count
    result.smart_bootstrap_depth_refit_count = bootstrap_depth_refits
    result.smart_config = settings
    result.smart_parent_state_structure_mode = _PARENT_STATE_METHOD
    result.smart_parent_state_algorithm_mode = _PARENT_STATE_LIKELIHOOD
    result.smart_b3_heldout_state_mask_policy = "none"
    result.smart_b3_aggregate_identity_mask_policy = "none"
    result.tier_a_partial_relationships = tier_a_partial_frame
    result.tier_b_partial_relationships = tier_b_partial_frame
    result.tier_b_candidate_sets = tier_b_candidate_sets
    result.tier_a_relationships = tier_a_frame
    result.tier_b_relationships = tier_b_frame
    result.complete_relationships = complete_frame
    result.smart_parent_state_calls = pd.DataFrame(state_call_rows)
    result.smart_diagnostics = pd.DataFrame(diagnostics)
    result.smart_prior_sensitivity_summary = pd.DataFrame(
        sensitivity_summary_rows
    )
    result.smart_fitted_parent_state_prior_parameters = (
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
    result.smart_evidence_summary = pd.DataFrame(evidence_summary)
    result.smart_predictive_folds = pd.DataFrame()
    result.smart_ancestry_depth_model = full_depth_model
    result.smart_ancestry_depth_model_available = bool(
        full_depth_model is not None
        and full_depth_model.posterior.shape[1] >= 2
    )
    result.smart_parent_eligibility_policy_label = eligibility.policy_name
    result.smart_parent_eligibility_supplied = eligibility.supplied
    result.smart_parent_eligibility_record = (
        _parent_eligibility_result_record(eligibility)
    )
    result.smart_ancestry_depth_model_parameters = {
        "maximum_components": _ANCESTRY_DEPTH_MAX_COMPONENTS,
        "gmm_initializations": _ANCESTRY_DEPTH_GMM_N_INIT,
        "gmm_max_iterations": _ANCESTRY_DEPTH_GMM_MAX_ITERATIONS,
        "standardized_covariance_regularization": (
            _ANCESTRY_DEPTH_GMM_REGULARIZATION
        ),
        "component_selection": "minimum_BIC",
        "direction_decision": "strict_MAP_probability_above_0.5",
    }
    result.smart_ancestry_depth_model_specification = (
        "BIC-selected one-dimensional Gaussian mixture over callability-"
        "adjusted, chromosome-wide phase-invariant minimum founder-trajectory "
        "switch burden; components are ordered relative ancestry layers, not "
        "generation labels. Its posterior supplies the per-edge D structural "
        "screen and orders graph conflicts. It never enters the forward B1 "
        "likelihood, but structural screening can mask M1/M2 alternatives and "
        "therefore affect local winners and Tier A/B release."
    )
    result.smart_selection_method = (
        "marginal parent-state selection followed by conditional identity; "
        "deterministic ancestry-direction-then-confidence-ordered variable-edge "
        "DAG with coordinate local search. A graph-conflicted local row may use "
        "a unique finite lower-parent-count alternative supported by downward "
        "ancestry direction; otherwise it falls to M0. Local support is measured "
        "before the DAG."
    )
    result.smart_candidate_screening_scope = (
        "M0 and every M1 identity are scored; M2 uses a fixed candidate panel. "
        "The M2 identity prior denominator is the full eligible pair "
        "space, so integrated M2 evidence is an explicit lower bound when the "
        "screen is incomplete."
    )
    result.smart_missing_parent_model = (
        "Normalized forward HMMs compare zero, one, and two observed parents. "
        "An external parent is a linked child-left-out mixture of locally "
        "IBS-pooled reconstructed founder haplotypes; zero observed parents "
        "does not assert biological founder status."
    )
    result.smart_limitations = (
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
        "supplies the D structural screen and graph ordering. Because that screen "
        "can mask parent alternatives, it can affect local calls and Tier A/B "
        "release; this is part of the combined method, not an independent "
        "likelihood source. Reconstructed paintings and hard founder alleles "
        "inherit upstream errors; raw genotype likelihoods are not double-"
        "counted as an independent source. Zero observed parents may mean a "
        "top-level individual or unsequenced biological parents and is not by "
        "itself safe founder-recolour eligibility. For API compatibility, trio_scores "
        "contains hierarchical decision utilities and total_bins an effective "
        "information-block count; neither is comparable to legacy Viterbi "
        "scores or raw bin counts."
    )
    return result


_STANDARD_RAW_GL_MODE = "raw_likelihood"
_STANDARD_COMPACT_RAW_GL_KEY = "standard_compact_raw_gl"
_STANDARD_COMPACT_RAW_GL_FORMAT_VERSION = 1
_STANDARD_COMPACT_RAW_GL_FIELDS = (
    "format_version",
    "state_evidence_mode",
    "genotype_likelihoods",
    "selected_positions",
    "sample_ids",
    "selection_parameters",
)
_STANDARD_COMPACT_SELECTION_FIELDS = (
    "snps_per_bin",
    "max_snps_per_bin",
    "recombination_rate",
)

_STANDARD_RAW_GL_INPUT_KEYS = (
    "standard_state_evidence_mode",
    "standard_raw_genotype_likelihoods",
    "standard_raw_positions",
    "standard_raw_sample_ids",
)


def _standard_raw_gl_transport(
    item: Mapping[str, Any],
) -> str | None:
    """Validate and identify a full or compact raw-likelihood transport."""
    full_present = [
        key in item for key in _STANDARD_RAW_GL_INPUT_KEYS
    ]
    compact_present = _STANDARD_COMPACT_RAW_GL_KEY in item
    if compact_present and any(full_present):
        raise SmartEvidenceError(
            "standard raw likelihood input cannot contain both full and "
            "compact transports"
        )
    if any(full_present) and not all(full_present):
        raise SmartEvidenceError(
            "standard raw likelihood input requires all of "
            "standard_state_evidence_mode, "
            "standard_raw_genotype_likelihoods, standard_raw_positions, "
            "and standard_raw_sample_ids"
        )
    if all(full_present):
        if item["standard_state_evidence_mode"] != _STANDARD_RAW_GL_MODE:
            raise SmartEvidenceError(
                "standard_state_evidence_mode must be exactly "
                "'raw_likelihood'"
            )
        return "full"
    if not compact_present:
        return None

    bundle = item[_STANDARD_COMPACT_RAW_GL_KEY]
    if not isinstance(bundle, Mapping):
        raise SmartEvidenceError(
            "standard_compact_raw_gl must be a field mapping"
        )
    missing = [
        field
        for field in _STANDARD_COMPACT_RAW_GL_FIELDS
        if field not in bundle
    ]
    if missing:
        raise SmartEvidenceError(
            "standard_compact_raw_gl is missing required field "
            f"{missing[0]}"
        )
    if (
        bundle["format_version"]
        != _STANDARD_COMPACT_RAW_GL_FORMAT_VERSION
    ):
        raise SmartEvidenceError(
            "unsupported standard_compact_raw_gl format_version"
        )
    if bundle["state_evidence_mode"] != _STANDARD_RAW_GL_MODE:
        raise SmartEvidenceError(
            "compact state_evidence_mode must be exactly 'raw_likelihood'"
        )
    selection = bundle["selection_parameters"]
    if not isinstance(selection, Mapping):
        raise SmartEvidenceError(
            "compact selection_parameters must be a field mapping"
        )
    missing_selection = [
        field
        for field in _STANDARD_COMPACT_SELECTION_FIELDS
        if field not in selection
    ]
    if missing_selection:
        raise SmartEvidenceError(
            "compact selection_parameters is missing required field "
            f"{missing_selection[0]}"
        )
    return "compact"


def _standard_raw_gl_mode(item: Mapping[str, Any]) -> bool:
    """Return whether either explicit raw-likelihood transport is present."""
    return _standard_raw_gl_transport(item) is not None


def _standard_input_schema(
    contig_data_list: Sequence[Any],
) -> bool:
    """Return whether every item has the standard numerical input schema."""
    if not contig_data_list:
        return False
    status = []
    raw_gl_status = []
    for item in contig_data_list:
        if not isinstance(item, Mapping):
            return False
        raw_gl_status.append(_standard_raw_gl_mode(item))
        has_painting = "tolerance_painting" in item
        has_founder = "founder_block" in item
        if has_painting != has_founder:
            raise SmartEvidenceError(
                "each standard contig needs both tolerance_painting and "
                "founder_block"
            )
        status.append(has_painting)
    if any(status) and not all(status):
        raise SmartEvidenceError(
            "standard tolerance-painting evidence must be supplied for every "
            "contig or for none"
        )
    if any(raw_gl_status) and not all(raw_gl_status):
        raise SmartEvidenceError(
            "standard raw likelihood evidence must be supplied for every "
            "contig or for none"
        )
    if any(raw_gl_status) and not all(status):
        raise SmartEvidenceError(
            "standard raw likelihood evidence requires tolerance_painting "
            "and founder_block on every contig"
        )
    return all(status)


def _hard_founder_haplotypes(founder_block: Any) -> tuple[list[int], np.ndarray]:
    positions = np.asarray(getattr(founder_block, "positions", None))
    haplotypes = getattr(founder_block, "haplotypes", None)
    if positions.ndim != 1 or len(positions) == 0:
        raise SmartEvidenceError("founder_block.positions must be non-empty")
    if len(positions) > 1 and np.any(np.diff(positions) <= 0):
        raise SmartEvidenceError("founder positions must be strictly increasing")
    if not isinstance(haplotypes, Mapping) or not haplotypes:
        raise SmartEvidenceError("founder_block.haplotypes must be non-empty")
    keys = sorted(int(value) for value in haplotypes)
    if any(value < 0 for value in keys) or len(set(keys)) != len(keys):
        raise SmartEvidenceError("founder labels must be unique non-negative integers")
    hard = np.empty((len(keys), len(positions)), dtype=np.int8)
    for local, key in enumerate(keys):
        values = np.asarray(haplotypes[key])
        if values.ndim == 2:
            if values.shape != (len(positions), 2):
                raise SmartEvidenceError(
                    "probabilistic founder haplotypes need shape (markers, 2)"
                )
            if np.any(~np.isfinite(values)):
                raise SmartEvidenceError("founder haplotype probabilities must be finite")
            values = np.argmax(values, axis=1)
        elif values.ndim != 1 or len(values) != len(positions):
            raise SmartEvidenceError(
                "hard founder haplotypes need one allele per founder position"
            )
        if np.any(~np.isin(values, (-1, 0, 1))):
            raise SmartEvidenceError("hard founder alleles must be -1, 0, or 1")
        hard[local] = values.astype(np.int8)
    return keys, hard


def _load_standard_compact_raw_genotype_likelihoods(
    bundle: Mapping[str, Any],
    sample_ids: Sequence[Any] | None,
    n_samples: int,
    selected_positions: np.ndarray,
    snps_per_bin: int,
    recombination_rate: float,
    max_snps_per_bin: int,
) -> np.ndarray:
    """Validate a versioned selected-marker raw-GL checkpoint bundle."""
    if sample_ids is None:
        raise SmartEvidenceError(
            "sample_ids are required for compact raw likelihood alignment"
        )
    if isinstance(bundle["sample_ids"], (str, bytes)):
        raise SmartEvidenceError(
            "compact sample_ids must be an ordered sample sequence"
        )
    try:
        compact_sample_ids = tuple(bundle["sample_ids"])
    except TypeError as exc:
        raise SmartEvidenceError(
            "compact sample_ids must be an ordered sample sequence"
        ) from exc
    expected_sample_ids = tuple(sample_ids)
    try:
        sample_order_matches = (
            compact_sample_ids == expected_sample_ids
        )
    except (TypeError, ValueError):
        sample_order_matches = False
    if (
        len(compact_sample_ids) != n_samples
        or not isinstance(sample_order_matches, (bool, np.bool_))
        or not bool(sample_order_matches)
    ):
        raise SmartEvidenceError(
            "compact sample_ids must exactly match ordered sample_ids"
        )

    expected_selection = {
        "snps_per_bin": int(snps_per_bin),
        "max_snps_per_bin": int(max_snps_per_bin),
        "recombination_rate": float(recombination_rate),
    }
    selection = bundle["selection_parameters"]
    for field, expected in expected_selection.items():
        try:
            matches = selection[field] == expected
        except (TypeError, ValueError):
            matches = False
        if (
            not isinstance(matches, (bool, np.bool_))
            or not bool(matches)
        ):
            raise SmartEvidenceError(
                "compact selection parameter mismatch for "
                f"{field}"
            )

    try:
        compact_position_values = np.asarray(
            bundle["selected_positions"], dtype=np.float64
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise SmartEvidenceError(
            "compact selected_positions must be integer coordinates"
        ) from exc
    if (
        compact_position_values.shape != selected_positions.shape
        or np.any(~np.isfinite(compact_position_values))
        or np.any(
            compact_position_values
            != np.floor(compact_position_values)
        )
        or np.any(
            compact_position_values < np.iinfo(np.int64).min
        )
        or np.any(
            compact_position_values > np.iinfo(np.int64).max
        )
    ):
        raise SmartEvidenceError(
            "compact selected_positions has invalid shape or coordinates"
        )
    compact_positions = compact_position_values.astype(np.int64)
    if not np.array_equal(compact_positions, selected_positions):
        raise SmartEvidenceError(
            "compact selected_positions does not match current selection"
        )

    try:
        compact = np.asarray(
            bundle["genotype_likelihoods"], dtype=np.float64
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise SmartEvidenceError(
            "compact genotype_likelihoods must be numeric linear values"
        ) from exc
    expected_shape = (
        n_samples,
        selected_positions.shape[0],
        selected_positions.shape[1],
        3,
    )
    if compact.shape != expected_shape:
        raise SmartEvidenceError(
            "compact genotype_likelihoods must have shape "
            "(samples, bins, SNP slots, 3)"
        )
    row_totals = np.sum(compact, axis=3)
    if (
        np.any(~np.isfinite(compact))
        or np.any(compact < 0.0)
        or np.any(~np.isfinite(row_totals))
        or np.any(row_totals <= 0.0)
    ):
        raise SmartEvidenceError(
            "every compact genotype-likelihood row must contain finite, "
            "non-negative linear values with a positive total"
        )
    padding = selected_positions < 0
    padded = compact[:, padding, :]
    if (
        np.any(padded[:, :, 0] != padded[:, :, 1])
        or np.any(padded[:, :, 1] != padded[:, :, 2])
    ):
        raise SmartEvidenceError(
            "compact genotype-likelihood padding must be uniform"
        )
    normalized = compact / row_totals[:, :, :, None]
    return np.ascontiguousarray(normalized, dtype=np.float64)


def _compact_standard_raw_genotype_likelihoods(
    item: Mapping[str, Any],
    sample_ids: Sequence[Any] | None,
    n_samples: int,
    selected_positions: np.ndarray,
    snps_per_bin: int,
    recombination_rate: float,
    max_snps_per_bin: int,
) -> np.ndarray | None:
    """Load either raw-likelihood transport into the selected GL grid."""
    transport = _standard_raw_gl_transport(item)
    if transport is None:
        return None
    if transport == "compact":
        return _load_standard_compact_raw_genotype_likelihoods(
            item[_STANDARD_COMPACT_RAW_GL_KEY],
            sample_ids,
            n_samples,
            selected_positions,
            snps_per_bin,
            recombination_rate,
            max_snps_per_bin,
        )
    if sample_ids is None:
        raise SmartEvidenceError(
            "sample_ids are required for standard raw likelihood alignment"
        )
    if isinstance(
        item["standard_raw_sample_ids"], (str, bytes)
    ):
        raise SmartEvidenceError(
            "standard_raw_sample_ids must be an ordered sample sequence"
        )
    try:
        raw_sample_ids = tuple(item["standard_raw_sample_ids"])
    except TypeError as exc:
        raise SmartEvidenceError(
            "standard_raw_sample_ids must be an ordered sample sequence"
        ) from exc
    expected_sample_ids = tuple(sample_ids)
    try:
        sample_order_matches = raw_sample_ids == expected_sample_ids
    except (TypeError, ValueError):
        sample_order_matches = False
    if (
        len(raw_sample_ids) != n_samples
        or not isinstance(sample_order_matches, (bool, np.bool_))
        or not bool(sample_order_matches)
    ):
        raise SmartEvidenceError(
            "standard_raw_sample_ids must exactly match ordered sample_ids"
        )

    try:
        raw_position_values = np.asarray(
            item["standard_raw_positions"], dtype=np.float64
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise SmartEvidenceError(
            "standard_raw_positions must be finite integer coordinates"
        ) from exc
    if (
        raw_position_values.ndim != 1
        or len(raw_position_values) < 1
        or np.any(~np.isfinite(raw_position_values))
        or np.any(raw_position_values != np.floor(raw_position_values))
        or np.any(raw_position_values < np.iinfo(np.int64).min)
        or np.any(raw_position_values > np.iinfo(np.int64).max)
    ):
        raise SmartEvidenceError(
            "standard_raw_positions must be finite integer coordinates"
        )
    raw_positions = raw_position_values.astype(np.int64)
    if len(raw_positions) > 1 and np.any(np.diff(raw_positions) <= 0):
        raise SmartEvidenceError(
            "standard_raw_positions must be strictly increasing and unique"
        )

    try:
        raw_likelihoods = np.asarray(
            item["standard_raw_genotype_likelihoods"],
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise SmartEvidenceError(
            "standard raw genotype likelihoods must be numeric linear values"
        ) from exc
    expected_shape = (n_samples, len(raw_positions), 3)
    if raw_likelihoods.shape != expected_shape:
        raise SmartEvidenceError(
            "standard_raw_genotype_likelihoods must have shape "
            "(samples, raw_positions, 3)"
        )
    row_totals = np.sum(raw_likelihoods, axis=2)
    if (
        np.any(~np.isfinite(raw_likelihoods))
        or np.any(raw_likelihoods < 0.0)
        or np.any(~np.isfinite(row_totals))
        or np.any(row_totals <= 0.0)
    ):
        raise SmartEvidenceError(
            "every standard raw genotype-likelihood row must contain finite, "
            "non-negative linear values with a positive total"
        )

    selected_mask = selected_positions >= 0
    selected_coordinates = selected_positions[selected_mask]
    if len(np.unique(selected_coordinates)) != len(selected_coordinates):
        raise SmartEvidenceError(
            "selected founder coordinates are duplicated or ambiguous"
        )
    raw_indices = np.searchsorted(raw_positions, selected_coordinates)
    matched = raw_indices < len(raw_positions)
    if np.any(matched):
        matched[matched] &= (
            raw_positions[raw_indices[matched]]
            == selected_coordinates[matched]
        )
    if not np.all(matched):
        missing = selected_coordinates[~matched]
        raise SmartEvidenceError(
            "standard_raw_positions is missing selected founder coordinate "
            f"{int(missing[0])}"
        )

    compact = np.full(
        (
            n_samples,
            selected_positions.shape[0],
            selected_positions.shape[1],
            3,
        ),
        1.0 / 3.0,
        dtype=np.float64,
    )
    bin_indices, slot_indices = np.nonzero(selected_mask)
    selected_likelihoods = raw_likelihoods[:, raw_indices, :]
    selected_totals = row_totals[:, raw_indices, None]
    compact[:, bin_indices, slot_indices, :] = (
        selected_likelihoods / selected_totals
    )
    return np.ascontiguousarray(compact, dtype=np.float64)


def _build_standard_contig_cache(
    item: Mapping[str, Any],
    contig_index: int,
    n_samples: int,
    snps_per_bin: int,
    recombination_rate: float,
    max_snps_per_bin: int,
    sample_ids: Sequence[Any] | None = None,
) -> _StandardContigCache:
    """Build the selected hard grid and optional aligned raw-GL grid."""

    if int(snps_per_bin) != snps_per_bin or snps_per_bin < 1:
        raise SmartEvidenceError("snps_per_bin must be a positive integer")
    if int(max_snps_per_bin) != max_snps_per_bin or max_snps_per_bin < 1:
        raise SmartEvidenceError("max_snps_per_bin must be a positive integer")
    if not np.isfinite(recombination_rate) or recombination_rate <= 0.0:
        raise SmartEvidenceError("recomb_rate must be finite and positive")
    painting = item["tolerance_painting"]
    founder_block = item["founder_block"]
    try:
        painting_samples = len(painting)
        start_position = float(painting.start_pos)
        end_position = float(painting.end_pos)
    except (AttributeError, TypeError) as error:
        raise SmartEvidenceError("invalid tolerance_painting container") from error
    if painting_samples != n_samples:
        raise SmartEvidenceError(
            "tolerance painting sample count does not match sample_ids"
        )
    if not (np.isfinite(start_position) and np.isfinite(end_position)):
        raise SmartEvidenceError("painting coordinates must be finite")
    if end_position <= start_position:
        raise SmartEvidenceError("painting end must be greater than start")

    positions = np.asarray(founder_block.positions, dtype=np.int64)
    founder_keys, hard_founders = _hard_founder_haplotypes(founder_block)
    approximate_bp_per_bin = int(snps_per_bin) * 100
    n_bins = max(
        100,
        int((end_position - start_position) / approximate_bp_per_bin),
    )
    bin_edges = np.linspace(start_position, end_position, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = (
        float(bin_centers[1] - bin_centers[0]) if n_bins > 1 else 10000.0
    )

    id_grids = np.full((n_samples, n_bins, 2), -1, dtype=np.int32)
    for sample_index in range(n_samples):
        sample = painting[sample_index]
        chunks = list(getattr(sample, "chunks", ()) or ())
        if not chunks:
            continue
        starts = np.asarray([chunk.start for chunk in chunks], dtype=np.float64)
        ends = np.asarray([chunk.end for chunk in chunks], dtype=np.float64)
        first_ids = np.asarray([chunk.hap1 for chunk in chunks], dtype=np.int64)
        second_ids = np.asarray([chunk.hap2 for chunk in chunks], dtype=np.int64)
        if (
            np.any(~np.isfinite(starts))
            or np.any(~np.isfinite(ends))
            or np.any(ends <= starts)
            or np.any(starts[1:] < starts[:-1])
            or np.any(ends[1:] < ends[:-1])
        ):
            raise SmartEvidenceError("painting chunks must be sorted valid intervals")
        chunk_indices = np.searchsorted(ends, bin_centers)
        chunk_indices = np.clip(chunk_indices, 0, len(chunks) - 1)
        valid = bin_centers >= starts[chunk_indices]
        id_grids[sample_index, :, 0] = np.where(
            valid, first_ids[chunk_indices], -1
        )
        id_grids[sample_index, :, 1] = np.where(
            valid, second_ids[chunk_indices], -1
        )

    local_ids = np.full_like(id_grids, -1)
    for local, founder_id in enumerate(founder_keys):
        local_ids[id_grids == founder_id] = local

    if max_snps_per_bin > 1:
        # Non-overlapping half-open bins assign an exact internal boundary SNP
        # to the bin on its right. The final chromosome endpoint is inclusive.
        start_indices = np.searchsorted(
            positions, bin_edges[:-1], side="left"
        )
        end_indices = np.searchsorted(
            positions, bin_edges[1:], side="left"
        )
        end_indices[-1] = np.searchsorted(
            positions, bin_edges[-1], side="right"
        )
        selected = np.full((n_bins, max_snps_per_bin), -1, dtype=np.int64)
        for bin_index in range(n_bins):
            first = int(start_indices[bin_index])
            last = int(end_indices[bin_index])
            count = last - first
            if count <= 0:
                continue
            if count <= max_snps_per_bin:
                chosen = np.arange(first, last, dtype=np.int64)
            else:
                step = count / float(max_snps_per_bin)
                chosen = np.asarray([
                    first + int(slot * step)
                    for slot in range(max_snps_per_bin)
                ], dtype=np.int64)
            selected[bin_index, :len(chosen)] = chosen
        informative_markers = int(np.sum(selected >= 0))
        selected_markers_per_bin = np.sum(selected >= 0, axis=1).astype(
            np.int64
        )
        lookup = np.full(
            (len(founder_keys), n_bins, max_snps_per_bin),
            -1,
            dtype=np.int8,
        )
        for bin_index in range(n_bins):
            chosen = selected[bin_index]
            valid_slots = chosen >= 0
            if np.any(valid_slots):
                lookup[:, bin_index, valid_slots] = hard_founders[
                    :, chosen[valid_slots]
                ]
        safe_ids = np.maximum(local_ids, 0)
        bin_axis = np.arange(n_bins)[None, :]
        first_track = lookup[safe_ids[:, :, 0], bin_axis].copy()
        second_track = lookup[safe_ids[:, :, 1], bin_axis].copy()
        first_track[local_ids[:, :, 0] < 0] = -1
        second_track[local_ids[:, :, 1] < 0] = -1
        stacked = np.stack((first_track, second_track), axis=2)
        jointly_valid = (first_track != -1) & (second_track != -1)
        hom_mask = (
            ~np.any(jointly_valid, axis=2)
            | np.all(~jointly_valid | (first_track == second_track), axis=2)
        )
        selected_grid = selected
    else:
        selected = np.searchsorted(positions, bin_centers)
        selected = np.clip(selected, 0, len(positions) - 1)
        valid_marker = (
            np.abs(positions[selected] - bin_centers) <= bin_width / 2.0
        )
        selected_grid = np.full(
            (n_bins, 1), -1, dtype=np.int64
        )
        selected_grid[valid_marker, 0] = selected[valid_marker]
        informative_markers = int(np.sum(valid_marker))
        lookup = hard_founders[:, selected].copy()
        lookup[:, ~valid_marker] = -1
        selected_markers_per_bin = valid_marker.astype(np.int64)
        safe_ids = np.maximum(local_ids, 0)
        bin_axis = np.arange(n_bins)[None, :]
        first_track = lookup[safe_ids[:, :, 0], bin_axis].copy()
        second_track = lookup[safe_ids[:, :, 1], bin_axis].copy()
        first_track[local_ids[:, :, 0] < 0] = -1
        second_track[local_ids[:, :, 1] < 0] = -1
        stacked = np.stack((first_track, second_track), axis=2)
        hom_mask = (
            (first_track == second_track)
            | (first_track == -1)
            | (second_track == -1)
        )
    if informative_markers < 1:
        raise SmartEvidenceError("a standard contig contains no sampled markers")
    selected_positions = np.full_like(selected_grid, -1)
    selected_mask = selected_grid >= 0
    selected_positions[selected_mask] = positions[
        selected_grid[selected_mask]
    ]
    genotype_likelihoods = (
        _compact_standard_raw_genotype_likelihoods(
            item,
            sample_ids,
            n_samples,
            selected_positions,
            snps_per_bin,
            recombination_rate,
            max_snps_per_bin,
        )
    )
    state_evidence_mode = (
        _STANDARD_RAW_GL_MODE
        if genotype_likelihoods is not None
        else "hard_allele"
    )

    theta, switch_costs, stay_costs = poisson_switch_stay_terms(
        bin_centers, recombination_rate
    )
    name = str(item.get("contig", f"contig_{contig_index + 1}"))
    return _StandardContigCache(
        contig=name,
        stacked_alleles=np.ascontiguousarray(stacked, dtype=np.int8),
        stacked_hom_mask=np.ascontiguousarray(hom_mask, dtype=np.bool_),
        switch_costs=switch_costs,
        stay_costs=stay_costs,
        informative_markers=informative_markers,
        stacked_labels=np.ascontiguousarray(local_ids, dtype=np.int16),
        founder_alleles=np.ascontiguousarray(
            lookup if lookup.ndim == 3 else lookup[:, :, None],
            dtype=np.int8,
        ),
        selected_markers_per_bin=np.ascontiguousarray(
            selected_markers_per_bin, dtype=np.int64
        ),
        switch_probabilities=np.ascontiguousarray(theta, dtype=np.float64),
        genotype_likelihoods=genotype_likelihoods,
        selected_positions=np.ascontiguousarray(
            selected_positions, dtype=np.int64
        ),
        state_evidence_mode=state_evidence_mode,
    )


def prepare_standard_compact_raw_gl(
    tolerance_painting: Any,
    founder_block: Any,
    sample_ids: Sequence[Any],
    raw_genotype_likelihoods: Any,
    raw_positions: Any,
    *,
    snps_per_bin: int,
    recombination_rate: float,
    max_snps_per_bin: int,
    contig: str = "compact_raw_gl",
) -> dict[str, Any]:
    """Prepare a selected-marker raw-GL checkpoint transport.

    Assign the returned mapping to a standard item's
    standard_compact_raw_gl key alongside the same tolerance_painting and
    founder_block. Selection is performed only by the standard cache builder;
    the full raw-position tensor is absent from the returned checkpoint.
    """
    ordered_sample_ids = tuple(sample_ids)
    full_item = {
        "contig": str(contig),
        "tolerance_painting": tolerance_painting,
        "founder_block": founder_block,
        "standard_state_evidence_mode": _STANDARD_RAW_GL_MODE,
        "standard_raw_genotype_likelihoods": raw_genotype_likelihoods,
        "standard_raw_positions": raw_positions,
        "standard_raw_sample_ids": ordered_sample_ids,
    }
    cache = _build_standard_contig_cache(
        full_item,
        0,
        len(ordered_sample_ids),
        snps_per_bin,
        recombination_rate,
        max_snps_per_bin,
        sample_ids=ordered_sample_ids,
    )
    return {
        "format_version": _STANDARD_COMPACT_RAW_GL_FORMAT_VERSION,
        "state_evidence_mode": _STANDARD_RAW_GL_MODE,
        "genotype_likelihoods": np.ascontiguousarray(
            cache.genotype_likelihoods, dtype=np.float64
        ),
        "selected_positions": np.ascontiguousarray(
            cache.selected_positions, dtype=np.int64
        ),
        "sample_ids": ordered_sample_ids,
        "selection_parameters": {
            "snps_per_bin": int(snps_per_bin),
            "max_snps_per_bin": int(max_snps_per_bin),
            "recombination_rate": float(recombination_rate),
        },
    }



def prepare_standard_compact_raw_gl_from_bcf(
    tolerance_painting: Any,
    founder_block: Any,
    sample_ids: Sequence[Any],
    bcf_path: Any,
    *,
    bcf_contig: str,
    snps_per_bin: int,
    recombination_rate: float,
    max_snps_per_bin: int,
    bcf_threads: int = 1,
    read_error_probability: float = DEFAULT_READ_ERROR_PROBABILITY,
) -> dict[str, Any]:
    """Prepare compact raw likelihoods from exact selected BCF AD rows.

    Every selected site uses the same prior-free binomial read model as T01,
    including sites where caller PL is absent. Only founder coordinates chosen
    by the standard cache builder are requested from the indexed BCF.
    """
    ordered_sample_ids = tuple(sample_ids)
    selection_item = {
        "contig": str(bcf_contig),
        "tolerance_painting": tolerance_painting,
        "founder_block": founder_block,
    }
    selection_cache = _build_standard_contig_cache(
        selection_item,
        0,
        len(ordered_sample_ids),
        snps_per_bin,
        recombination_rate,
        max_snps_per_bin,
        sample_ids=ordered_sample_ids,
    )
    selected_positions = np.unique(
        selection_cache.selected_positions[
            selection_cache.selected_positions >= 0
        ]
    )
    del selection_cache
    if not len(selected_positions):
        raise SmartEvidenceError(
            "standard marker selection contains no BCF coordinates"
        )
    linear_likelihoods, raw_positions = load_bcf_raw_genotype_likelihoods(
        bcf_path,
        str(bcf_contig),
        ordered_sample_ids,
        selected_positions=selected_positions,
        threads=bcf_threads,
        read_error_probability=read_error_probability,
    )
    bundle = prepare_standard_compact_raw_gl(
        tolerance_painting,
        founder_block,
        ordered_sample_ids,
        linear_likelihoods,
        raw_positions,
        snps_per_bin=snps_per_bin,
        recombination_rate=recombination_rate,
        max_snps_per_bin=max_snps_per_bin,
        contig=str(bcf_contig),
    )
    bundle["source_evidence_mode"] = "bcf_ad_binomial_raw_likelihood_v1"
    bundle["read_error_probability"] = float(read_error_probability)
    return bundle


_RUN_PAIR_HMM = _hard_painting.run_phase_agnostic_hmm
_RUN_PAIR_HMM_MULTISNP = _hard_painting.run_phase_agnostic_hmm_multisnp


@njit(fastmath=True, cache=True, parallel=True)
def _score_all_pairs_kernel_multisnp(
    stacked_alleles,
    stacked_hom_mask,
    switch_costs,
    stay_costs,
    error_penalty,
    phase_penalty,
    mismatch_penalty,
    eligible_parents,
):
    """Score the full child-parent matrix in one parallel launch."""
    n_samples = stacked_alleles.shape[0]
    output = np.empty((n_samples, n_samples), dtype=np.float64)
    for flat_index in prange(n_samples * n_samples):
        child = flat_index // n_samples
        parent = flat_index - child * n_samples
        if not eligible_parents[child, parent]:
            output[child, parent] = -math.inf
        else:
            output[child, parent] = _RUN_PAIR_HMM_MULTISNP(
                stacked_alleles[child],
                stacked_hom_mask[child],
                stacked_alleles[parent],
                switch_costs,
                stay_costs,
                error_penalty,
                phase_penalty,
                mismatch_penalty,
            )
    return output


@njit(fastmath=True, cache=True, parallel=True)
def _score_all_pairs_kernel(
    stacked_alleles,
    stacked_hom_mask,
    switch_costs,
    stay_costs,
    error_penalty,
    phase_penalty,
    mismatch_penalty,
    eligible_parents,
):
    """Non-multisnp counterpart of the flattened pair-screen kernel."""
    n_samples = stacked_alleles.shape[0]
    output = np.empty((n_samples, n_samples), dtype=np.float64)
    for flat_index in prange(n_samples * n_samples):
        child = flat_index // n_samples
        parent = flat_index - child * n_samples
        if not eligible_parents[child, parent]:
            output[child, parent] = -math.inf
        else:
            output[child, parent] = _RUN_PAIR_HMM(
                stacked_alleles[child],
                stacked_hom_mask[child],
                stacked_alleles[parent],
                switch_costs,
                stay_costs,
                error_penalty,
                phase_penalty,
                mismatch_penalty,
            )
    return output


def _score_pair_hmm_contig(
    cache: _StandardContigCache,
    mismatch_penalty: float,
    eligibility: Optional[_ResolvedParentEligibility] = None,
) -> np.ndarray:
    error_penalty = -math.log(1e-2)
    phase_penalty = 50.0
    if eligibility is None:
        _, eligible_parents = _scoring_eligibility_masks(
            cache.stacked_alleles.shape[0], None, None
        )
    else:
        eligible_parents = eligibility.eligible_parents

    if cache.stacked_alleles.ndim == 4:
        return _score_all_pairs_kernel_multisnp(
            cache.stacked_alleles,
            cache.stacked_hom_mask,
            cache.switch_costs,
            cache.stay_costs,
            error_penalty,
            phase_penalty,
            mismatch_penalty,
            eligible_parents,
        )
    return _score_all_pairs_kernel(
        cache.stacked_alleles,
        cache.stacked_hom_mask,
        cache.switch_costs,
        cache.stay_costs,
        error_penalty,
        phase_penalty,
        mismatch_penalty,
        eligible_parents,
    )


def _robust_parent_screen(
    pair_scores: np.ndarray,
    marker_counts: np.ndarray,
    config: SmartPedigreeConfig,
    eligibility: _ResolvedParentEligibility,
) -> np.ndarray:
    """Information-weighted utilities for eligible fixed-screen parents."""
    n_contigs, n_samples, _ = pair_scores.shape
    output = np.full((n_samples, n_samples), -np.inf, dtype=np.float64)
    contig_weights = _information_weights(marker_counts, config)
    blocks = np.maximum(
        np.ceil(marker_counts / config.markers_per_information_block), 1.0
    )
    tempering = blocks ** config.information_tempering_power
    for child in range(n_samples):
        if not eligibility.eligible_children[child]:
            continue
        parents = np.flatnonzero(eligibility.eligible_parents[child])
        if not len(parents):
            continue
        utility = np.empty((n_contigs, len(parents)), dtype=np.float64)
        for contig_index in range(n_contigs):
            values = pair_scores[contig_index, child, parents]
            if np.any(~np.isfinite(values)):
                raise SmartEvidenceError("standard pair-HMM scores must be finite")
            centered = (values - np.max(values)) / tempering[contig_index]
            soft = np.exp(np.clip(centered, -60.0, 0.0))
            soft /= np.sum(soft)
            soft = (
                (1.0 - config.chromosome_contamination) * soft
                + config.chromosome_contamination / len(parents)
            )
            ranks = _tied_rank_probabilities(values)
            utility[contig_index] = (
                config.rank_weight * ranks
                + (1.0 - config.rank_weight) * soft
            )
        output[child, parents] = np.sum(
            utility * contig_weights[:, None], axis=0
        )
    return output


def _fixed_trio_panel(
    parent_scores: np.ndarray,
    top_k: int,
    anchor_k: int,
    use_anchor_union: bool,
    eligibility: _ResolvedParentEligibility,
) -> np.ndarray:
    n_samples = parent_scores.shape[0]
    if n_samples < 3:
        raise SmartEvidenceError("at least three samples are required")
    if int(top_k) != top_k or top_k < 1:
        raise SmartEvidenceError("top_k must be a positive integer")
    if int(anchor_k) != anchor_k or anchor_k < 0:
        raise SmartEvidenceError("anchor_k must be a non-negative integer")
    rows = []
    for child in range(n_samples):
        if not eligibility.eligible_children[child]:
            continue
        parents = np.flatnonzero(eligibility.eligible_parents[child])
        order = np.lexsort((parents, -parent_scores[child, parents]))
        leading = parents[order[:min(int(top_k), len(parents))]].tolist()
        pairs = {
            tuple(sorted((int(leading[first]), int(leading[second]))))
            for first in range(len(leading))
            for second in range(first + 1, len(leading))
            if _eligible_parent_pair(
                eligibility, child, leading[first], leading[second]
            )
        }
        if use_anchor_union:
            for anchor in leading[:min(int(anchor_k), len(leading))]:
                for other in parents:
                    if (
                        int(other) != int(anchor)
                        and _eligible_parent_pair(
                            eligibility, child, int(anchor), int(other)
                        )
                    ):
                        pairs.add(tuple(sorted((int(anchor), int(other)))))
        rows.extend((child, first, second) for first, second in sorted(pairs))
    return np.asarray(rows, dtype=np.int64).reshape((-1, 3))


def _score_standard_contig_parent_states(
    cache: _StandardContigCache,
    trios: np.ndarray,
    config: SmartPedigreeConfig,
    eligible_children: Optional[np.ndarray] = None,
    eligible_parents: Optional[np.ndarray] = None,
    *,
    dynamic_rebalance: bool = False,
    dynamic_child_chunk_floor: int = 32,
    dynamic_child_chunk_scale: int = 4,
    precomputed_candidate_source=None,
) -> _ParentStateContigScores:
    """Dispatch standard state evidence without changing the hard screen."""
    common = dict(
        _eligible_children=eligible_children,
        _eligible_parents=eligible_parents,
        mismatch_probability=config.parent_state_mismatch_probability,
        phase_switch_probability=(
            config.parent_state_phase_switch_probability
        ),
        markers_per_information_block=config.markers_per_information_block,
        effective_markers_per_information_block=(
            config.parent_state_effective_markers_per_information_block
        ),
        external_state_pseudocount=(
            config.parent_state_external_state_pseudocount
        ),
        external_transition_pseudocount=(
            config.parent_state_external_transition_pseudocount
        ),
    )
    if (
        config.parent_state_candidate_source_mode in {
            "exact_raw_gl_v1", "matched_null_raw_gl_v2"
        }
        and cache.genotype_likelihoods is None
    ):
        raise SmartEvidenceError(
            f"{config.parent_state_candidate_source_mode} candidate sources "
            "require raw genotype likelihoods"
        )
    if cache.genotype_likelihoods is not None:
        return score_parent_state_gl_hmms(
            cache.genotype_likelihoods,
            cache.stacked_alleles,
            cache.stacked_labels,
            cache.stacked_hom_mask,
            cache.founder_alleles,
            cache.selected_markers_per_bin,
            cache.switch_probabilities,
            trios,
            candidate_source_mode=(
                config.parent_state_candidate_source_mode
            ),
            candidate_source_path_switch_probability=(
                config.parent_state_candidate_source_path_switch_probability
            ),
            _precomputed_candidate_source=(
                precomputed_candidate_source
            ),
            _dynamic_rebalance=dynamic_rebalance,
            _dynamic_child_chunk_floor=(
                dynamic_child_chunk_floor
            ),
            _dynamic_child_chunk_scale=(
                dynamic_child_chunk_scale
            ),
            **common,
        )
    hard_dynamic = {}
    if dynamic_rebalance:
        hard_dynamic = {
            "_dynamic_rebalance": True,
            "_dynamic_child_chunk_floor": dynamic_child_chunk_floor,
            "_dynamic_child_chunk_scale": dynamic_child_chunk_scale,
        }
    return score_parent_state_hmms(
        cache.stacked_alleles,
        cache.stacked_labels,
        cache.stacked_hom_mask,
        cache.founder_alleles,
        cache.selected_markers_per_bin,
        cache.switch_probabilities,
        trios,
        **common,
        **hard_dynamic,
    )


_SMART_STATE_TRIOS = None
_SMART_STATE_TRIO_SHM = None
_SMART_STATE_CONFIG = None
_SMART_STATE_START_BARRIER = None
_SMART_STATE_ELIGIBLE_CHILDREN = None
_SMART_STATE_ELIGIBLE_PARENTS = None
_SMART_STATE_TARGET_THREADS_PER_PROCESS = 8
_SMART_STATE_CHILD_CHUNK_FLOOR = 32
_SMART_STATE_CHILD_CHUNK_SCALE = 4


def _init_smart_state_worker(
    trio_meta,
    config,
    active_counter,
    total_cores,
    extra_counter,
    start_barrier,
    child_chunk_floor,
    child_chunk_scale,
    eligible_children,
    eligible_parents,
):
    """Attach the fixed trio panel and initialize bounded dynamic threads."""
    global _SMART_STATE_TRIOS, _SMART_STATE_TRIO_SHM
    global _SMART_STATE_CONFIG, _SMART_STATE_START_BARRIER
    global _SMART_STATE_CHILD_CHUNK_FLOOR
    global _SMART_STATE_CHILD_CHUNK_SCALE
    global _SMART_STATE_ELIGIBLE_CHILDREN, _SMART_STATE_ELIGIBLE_PARENTS
    _SMART_STATE_TRIO_SHM, _SMART_STATE_TRIOS = attach_shared_array(
        trio_meta
    )
    _SMART_STATE_CONFIG = config
    _SMART_STATE_START_BARRIER = start_barrier
    _SMART_STATE_CHILD_CHUNK_FLOOR = child_chunk_floor
    _SMART_STATE_CHILD_CHUNK_SCALE = child_chunk_scale
    _SMART_STATE_ELIGIBLE_CHILDREN = eligible_children
    _SMART_STATE_ELIGIBLE_PARENTS = eligible_parents
    dynamic_threads.set_dynamic_thread_state(
        total_cores, active_counter, extra_counter
    )
    numba.set_num_threads(1)


def _score_standard_contig_state_worker(bundle):
    """Score one static contig bundle with dynamic intra-process threads."""
    dynamic_threads.increment_active()
    try:
        _SMART_STATE_START_BARRIER.wait(timeout=180.0)
        settings = _SMART_STATE_CONFIG
        indexed = []
        for contig_index, cache in bundle:
            _apply_smart_dynamic_threads()
            scores = _score_standard_contig_parent_states(
                cache,
                _SMART_STATE_TRIOS,
                settings,
                _SMART_STATE_ELIGIBLE_CHILDREN,
                _SMART_STATE_ELIGIBLE_PARENTS,
                dynamic_rebalance=True,
                dynamic_child_chunk_scale=(
                    _SMART_STATE_CHILD_CHUNK_SCALE
                ),
                dynamic_child_chunk_floor=(
                    _SMART_STATE_CHILD_CHUNK_FLOOR
                ),
            )
            indexed.append((contig_index, scores))
        return indexed
    finally:
        dynamic_threads.release_dynamic_extra()
        dynamic_threads.decrement_active()


def _warm_smart_parent_state_kernels(include_raw_gl=False):
    """Compile each standard-input smart signature once before forkserver."""
    previous_threads = int(numba.get_num_threads())
    numba.set_num_threads(1)
    try:
        founders = np.asarray(
            (((0,), (1,)), ((1,), (0,))), dtype=np.int8
        )
        labels = np.asarray(
            (
                ((0, 1), (0, 1)),
                ((0, 0), (1, 1)),
                ((1, 0), (1, 0)),
                ((1, 1), (0, 0)),
            ),
            dtype=np.int16,
        )
        alleles = np.empty((4, 2, 2, 1), dtype=np.int8)
        for sample in range(4):
            for block in range(2):
                for track in range(2):
                    alleles[sample, block, track, 0] = founders[
                        labels[sample, block, track], block, 0
                    ]
        trios = []
        for child in range(4):
            parents = [value for value in range(4) if value != child]
            for first_index in range(len(parents)):
                for second_index in range(first_index + 1, len(parents)):
                    trios.append(
                        (child, parents[first_index], parents[second_index])
                    )
        score_parent_state_hmms(
            alleles,
            labels,
            np.zeros((4, 2), dtype=np.bool_),
            founders,
            np.ones(2, dtype=np.int64),
            np.asarray((0.0, 0.01), dtype=np.float64),
            np.asarray(trios, dtype=np.int64),
        )
        if include_raw_gl:
            raw_alleles = alleles.copy()
            raw_alleles[1, 0, 0, 0] = -1
            genotype_likelihoods = np.empty(
                (4, 2, 1, 3), dtype=np.float64
            )
            genotype_likelihoods[:, 0, 0] = np.asarray(
                (0.82, 0.15, 0.03), dtype=np.float64
            )
            genotype_likelihoods[:, 1, 0] = np.asarray(
                (0.08, 0.84, 0.08), dtype=np.float64
            )
            score_parent_state_gl_hmms(
                genotype_likelihoods,
                raw_alleles,
                labels,
                np.zeros((4, 2), dtype=np.bool_),
                founders,
                np.ones(2, dtype=np.int64),
                np.asarray((0.0, 0.01), dtype=np.float64),
                np.asarray(trios, dtype=np.int64),
            )
    finally:
        numba.set_num_threads(previous_threads)


def _balanced_state_contig_bundles(
    caches,
    worker_count,
):
    """Assign whole contigs to deterministic least-loaded static bundles."""
    if int(worker_count) != worker_count or worker_count < 1:
        raise SmartEvidenceError("state_worker_count must be a positive integer")
    bundle_count = min(int(worker_count), len(caches))
    bundles = [[] for _ in range(bundle_count)]
    loads = [0 for _ in range(bundle_count)]
    weighted = sorted(
        (
            (
                int(cache.stacked_alleles.shape[1]),
                index,
                cache,
            )
            for index, cache in enumerate(caches)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for weight, index, cache in weighted:
        target = min(range(bundle_count), key=lambda item: (loads[item], item))
        bundles[target].append((index, cache))
        loads[target] += weight
    return [tuple(bundle) for bundle in bundles]


def _score_standard_state_contigs(
    caches,
    trios,
    config,
    requested_threads,
    eligibility,
    state_worker_count=None,
    dynamic_child_chunk_floor=32,
    dynamic_child_chunk_scale=4,
):
    """Score contigs serially or with a fixed, dynamically threaded pool."""
    n_contigs = len(caches)
    try:
        child_chunk_floor = operator.index(dynamic_child_chunk_floor)
    except TypeError as exc:
        raise SmartEvidenceError(
            "dynamic_child_chunk_floor must be a positive integer"
        ) from exc
    if child_chunk_floor < 1:
        raise SmartEvidenceError(
            "dynamic_child_chunk_floor must be a positive integer"
        )
    try:
        child_chunk_scale = operator.index(dynamic_child_chunk_scale)
    except TypeError as exc:
        raise SmartEvidenceError(
            "dynamic_child_chunk_scale must be a non-negative integer"
        ) from exc
    if child_chunk_scale < 0:
        raise SmartEvidenceError(
            "dynamic_child_chunk_scale must be a non-negative integer"
        )
    if state_worker_count is None:
        target = _SMART_STATE_TARGET_THREADS_PER_PROCESS
        if n_contigs < target:
            worker_count = min(n_contigs, requested_threads)
        else:
            worker_count = min(
                n_contigs,
                max(1, (requested_threads + target - 1) // target),
            )
    else:
        try:
            worker_count = operator.index(state_worker_count)
        except TypeError as exc:
            raise SmartEvidenceError(
                "state_worker_count must be a positive integer"
            ) from exc
        if worker_count < 1:
            raise SmartEvidenceError("state_worker_count must be positive")
        worker_count = min(worker_count, n_contigs)
    use_pool = (
        worker_count > 1
        and requested_threads >= worker_count
    )
    if not use_pool:
        numba.set_num_threads(requested_threads)
        return [
            _score_standard_contig_parent_states(
                cache, trios, config,
                eligibility.eligible_children,
                eligibility.eligible_parents,
            )
            for cache in caches
        ], 1

    _warm_smart_parent_state_kernels(
        include_raw_gl=caches[0].genotype_likelihoods is not None
    )
    active_counter = forkserver_context.Value("i", 0)
    extra_counter = forkserver_context.Value("i", 0)
    start_barrier = forkserver_context.Barrier(worker_count)
    trio_shm, trio_meta = create_shared_array(trios)
    tasks = _balanced_state_contig_bundles(caches, worker_count)
    with shared_memory_cleanup([trio_shm]), safe_forkserver_pool(
        worker_count,
        initializer=_init_smart_state_worker,
        initargs=(
            trio_meta,
            config,
            active_counter,
            requested_threads,
            extra_counter,
            start_barrier,
            child_chunk_floor,
            child_chunk_scale,
            eligibility.eligible_children,
            eligibility.eligible_parents,
        ),
    ) as pool:
        nested = list(pool.imap_unordered(
            _score_standard_contig_state_worker, tasks, chunksize=1
        ))
    indexed = [
        item for bundle in nested for item in bundle
    ]
    indexed.sort(key=operator.itemgetter(0))
    return [scores for _, scores in indexed], worker_count


def _standard_contig_evidence(
    contig_data_list: Sequence[Any],
    sample_ids: Sequence[Any],
    config: SmartPedigreeConfig,
    *,
    eligibility: Optional[_ResolvedParentEligibility] = None,
    top_k: int,
    snps_per_bin: int,
    recomb_rate: float,
    mismatch_penalty: float,
    max_snps_per_bin: int,
    n_workers: Optional[int],
    anchor_k: int,
    use_anchor_union: bool,
) -> tuple[
    list[SmartParentStateEvidence],
    np.ndarray,
    int,
    int,
    np.ndarray,
    np.ndarray,
    str,
    list[_ParentStateContigScores],
]:
    if not np.isfinite(mismatch_penalty) or mismatch_penalty >= 0.0:
        raise SmartEvidenceError("mismatch_penalty must be finite and negative")
    n_samples = len(sample_ids)
    eligibility = _resolve_parent_eligibility(eligibility, sample_ids)
    if n_samples < 3 or len(set(sample_ids)) != n_samples:
        raise SmartEvidenceError(
            "sample_ids must contain at least three unique IDs"
        )
    caches = [
        _build_standard_contig_cache(
            item,
            contig_index,
            n_samples,
            snps_per_bin,
            recomb_rate,
            max_snps_per_bin,
            sample_ids=sample_ids,
        )
        for contig_index, item in enumerate(contig_data_list)
    ]
    evidence_modes = {cache.state_evidence_mode for cache in caches}
    if len(evidence_modes) != 1:
        raise SmartEvidenceError(
            "cannot mix hard and raw-likelihood standard state evidence"
        )
    state_evidence_mode = evidence_modes.pop()
    names = [cache.contig for cache in caches]
    if len(set(names)) != len(names):
        raise SmartEvidenceError("standard contig identifiers must be unique")
    marker_counts = np.asarray(
        [cache.informative_markers for cache in caches], dtype=np.float64
    )

    capacity = int(numba.config.NUMBA_NUM_THREADS)
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = os.cpu_count() or 1
    if n_workers is None:
        requested_threads = min(capacity, available_cpus)
    else:
        if int(n_workers) != n_workers or n_workers < 1:
            raise SmartEvidenceError("n_workers must be a positive integer")
        requested_threads = min(int(n_workers), available_cpus, capacity)
    previous_threads = int(numba.get_num_threads())
    numba.set_num_threads(requested_threads)
    try:
        exact_source = config.parent_state_candidate_source_mode in {
            "exact_raw_gl_v1", "matched_null_raw_gl_v2"
        }
        if exact_source:
            if state_evidence_mode != _STANDARD_RAW_GL_MODE:
                raise SmartEvidenceError(
                    f"{config.parent_state_candidate_source_mode} candidate "
                    "sources require raw genotype likelihoods on every "
                    "standard contig"
                )
            empty_trios = np.empty((0, 3), dtype=np.int64)
            prescreen_scores = [
                _score_standard_contig_parent_states(
                    cache, empty_trios, config,
                    eligibility.eligible_children,
                    eligibility.eligible_parents,
                )
                for cache in caches
            ]
            pair_scores = np.asarray([
                scores.one_observed for scores in prescreen_scores
            ])
        else:
            pair_scores = np.asarray([
                _score_pair_hmm_contig(cache, mismatch_penalty, eligibility)
                for cache in caches
            ])
        parent_scores = _robust_parent_screen(
            pair_scores, marker_counts, config, eligibility
        )
        trios = _fixed_trio_panel(
            parent_scores, top_k, anchor_k, use_anchor_union, eligibility
        )
        if exact_source:
            state_scores = [
                _score_standard_contig_parent_states(
                    cache,
                    trios,
                    config,
                    eligibility.eligible_children,
                    eligibility.eligible_parents,
                    precomputed_candidate_source=(
                        prescreen_scores[index].candidate_source_posterior
                    ),
                )
                for index, cache in enumerate(caches)
            ]
            state_processes = 1
        else:
            state_scores, state_processes = _score_standard_state_contigs(
                caches, trios, config, requested_threads, eligibility
            )
    finally:
        if previous_threads is not None:
            numba.set_num_threads(previous_threads)

    evidence = [
        SmartParentStateEvidence(
            contig=cache.contig,
            trios=trios,
            zero_parent_log_likelihoods=state_scores[index].zero_observed,
            one_parent_log_likelihoods=state_scores[index].one_observed,
            two_parent_log_likelihoods=state_scores[index].two_observed,
            informative_markers=cache.informative_markers,
            edge_matched_bins=state_scores[index].edge_matched_bins,
            edge_exposed_bins=state_scores[index].edge_exposed_bins,
            pair_explained_bins=state_scores[index].pair_explained_bins,
            pair_exposed_bins=state_scores[index].pair_exposed_bins,
            structure_total_bins=state_scores[index].structure_total_bins,
        )
        for index, cache in enumerate(caches)
    ]
    junction_counts = np.stack([
        scores.ancestry_junction_counts for scores in state_scores
    ])
    callable_haplotype_bins = np.stack([
        scores.ancestry_callable_haplotype_bins for scores in state_scores
    ])
    return (
        evidence,
        trios,
        requested_threads,
        state_processes,
        junction_counts,
        callable_haplotype_bins,
        state_scores,
        state_evidence_mode,
    )


def _config_from_contig_inputs(
    contig_data_list: Sequence[Any],
    explicit_config: Optional[SmartPedigreeConfig] = None,
) -> SmartPedigreeConfig:
    if explicit_config is not None:
        if not isinstance(explicit_config, SmartPedigreeConfig):
            raise SmartEvidenceError("config must be a SmartPedigreeConfig")
        return explicit_config.validated()
    config: Any = None
    if (
        contig_data_list
        and isinstance(contig_data_list[0], Mapping)
        and "smart_config" in contig_data_list[0]
    ):
        config = contig_data_list[0]["smart_config"]
        if isinstance(config, Mapping):
            config_values = dict(config)
            for field_name, expected in (
                ("parent_state_algorithm_mode", _PARENT_STATE_LIKELIHOOD),
                ("parent_state_structure_mode", _PARENT_STATE_METHOD),
            ):
                if field_name not in config_values:
                    continue
                if config_values.pop(field_name) != expected:
                    raise SmartEvidenceError(
                        f"embedded {field_name} selects a retired pedigree method"
                    )
            config = SmartPedigreeConfig(**config_values)
        if not isinstance(config, SmartPedigreeConfig):
            raise SmartEvidenceError(
                "smart_config must be SmartPedigreeConfig or a field mapping"
            )
    return (config or SmartPedigreeConfig()).validated()



def _parent_eligibility_from_contig_inputs(
    contig_data_list: Sequence[Any],
) -> Optional[SmartParentEligibility | Mapping[str, Any]]:
    """Read the run-level eligibility record from the first standard item."""
    if not contig_data_list:
        return None
    later = [
        index for index, item in enumerate(contig_data_list[1:], start=1)
        if "smart_parent_eligibility" in item
    ]
    if later:
        raise SmartEvidenceError(
            "smart_parent_eligibility is run-level metadata and may appear "
            "only on the first standard contig item"
        )
    return contig_data_list[0].get("smart_parent_eligibility")


def _infer_standard_inputs(
    contig_data_list: Sequence[Any],
    sample_ids: Sequence[Any],
    config: SmartPedigreeConfig,
    *,
    parent_eligibility: Optional[SmartParentEligibility | Mapping[str, Any]] = None,
    top_k: int,
    snps_per_bin: int,
    recomb_rate: float,
    mismatch_penalty: float,
    max_snps_per_bin: int,
    n_workers: Optional[int],
    anchor_k: int,
    use_anchor_union: bool,
) -> PedigreeResult:
    eligibility = _resolve_parent_eligibility(
        parent_eligibility, sample_ids
    )
    (
        evidence,
        trios,
        used_threads,
        state_processes,
        junction_counts,
        callable_haplotype_bins,
        source_scores,
        state_evidence_mode,
    ) = _standard_contig_evidence(
        contig_data_list,
        sample_ids,
        config,
        eligibility=eligibility,
        top_k=top_k,
        snps_per_bin=snps_per_bin,
        recomb_rate=recomb_rate,
        mismatch_penalty=mismatch_penalty,
        max_snps_per_bin=max_snps_per_bin,
        n_workers=n_workers,
        anchor_k=anchor_k,
        use_anchor_union=use_anchor_union,
    )
    result = infer_from_parent_state_evidence(
        evidence,
        sample_ids,
        config=config,
        parent_eligibility=eligibility,
        ancestry_junction_counts=junction_counts,
        ancestry_callable_haplotype_bins=callable_haplotype_bins,
        n_workers=used_threads,
    )
    structure_edges = (
        eligibility.eligible_parents | eligibility.eligible_parents.T
    )
    np.fill_diagonal(structure_edges, True)
    result.smart_standard_scored_eligible_child_count = int(
        np.count_nonzero(eligibility.eligible_children)
    )
    result.smart_standard_scored_eligible_parent_edge_count = int(
        np.count_nonzero(eligibility.eligible_parents)
    )
    result.smart_standard_scored_structure_edge_count = int(
        np.count_nonzero(structure_edges)
    )
    result.smart_standard_eligibility_scoring_scope = (
        "dense_exact_candidate_source_likelihoods_with_sparse_structure"
        if config.parent_state_candidate_source_mode in {
            "exact_raw_gl_v1", "matched_null_raw_gl_v2"
        }
        else "resolved_eligibility_likelihoods_and_structure"
    )
    result.smart_parent_state_candidate_source_mode = (
        config.parent_state_candidate_source_mode
    )
    result.smart_parent_state_candidate_source_path_switch_probability = (
        config.parent_state_candidate_source_path_switch_probability
    )
    result.smart_candidate_source_fallback_contig_count = int(sum(
        bool(scores.candidate_source_fallback) for scores in source_scores
    ))
    summary = result.smart_evidence_summary
    summary["CandidateSourceModeRequested"] = [
        scores.candidate_source_mode_requested for scores in source_scores
    ]
    summary["CandidateSourceModeApplied"] = [
        scores.candidate_source_mode_applied for scores in source_scores
    ]
    summary["CandidateSourcePathSwitchProbability"] = [
        (
            config.parent_state_candidate_source_path_switch_probability
            if scores.candidate_source_mode_requested
            == "matched_null_raw_gl_v2"
            else np.nan
        )
        for scores in source_scores
    ]
    summary["OffspringTransmissionSelector"] = [
        (
            "shared_biological_theta_no_hard_homo_reset"
            if scores.candidate_source_mode_applied
            == "matched_null_raw_gl_v2"
            else "legacy_mode_specific"
        )
        for scores in source_scores
    ]
    summary["CandidateSourceWholeContigFallback"] = [
        bool(scores.candidate_source_fallback) for scores in source_scores
    ]
    summary["CandidateSourceFallbackReason"] = [
        scores.candidate_source_fallback_reason for scores in source_scores
    ]
    summary["CompleteFounderMarkerCount"] = [
        scores.complete_founder_marker_count for scores in source_scores
    ]
    summary["ExcludedFounderMarkerCount"] = [
        scores.excluded_founder_marker_count for scores in source_scores
    ]
    summary["CandidateSourceAvailableCount"] = [
        (
            np.nan if scores.candidate_source_available is None
            else int(np.count_nonzero(scores.candidate_source_available))
        )
        for scores in source_scores
    ]
    summary["CandidateSourceMeanInformativeMarkerCount"] = [
        (
            np.nan
            if scores.candidate_source_informative_marker_count is None
            else float(np.mean(
                scores.candidate_source_informative_marker_count
            ))
        )
        for scores in source_scores
    ]
    summary["PeakStreamedTensorBytes"] = [
        int(scores.peak_streamed_tensor_bytes) for scores in source_scores
    ]
    source_rows = []
    for contig, scores in zip(summary["Contig"], source_scores):
        for candidate, sample in enumerate(sample_ids):
            source_rows.append({
                "Contig": contig,
                "Candidate": sample,
                "SourceModeApplied": scores.candidate_source_mode_applied,
                "SourcePathSwitchProbability": (
                    config.parent_state_candidate_source_path_switch_probability
                    if scores.candidate_source_mode_requested
                    == "matched_null_raw_gl_v2"
                    else np.nan
                ),
                "WholeContigFallback": bool(scores.candidate_source_fallback),
                "SourceAvailable": (
                    np.nan if scores.candidate_source_available is None
                    else bool(scores.candidate_source_available[candidate])
                ),
                "SourceInformativeMarkerCount": (
                    np.nan
                    if scores.candidate_source_informative_marker_count is None
                    else int(scores.candidate_source_informative_marker_count[candidate])
                ),
                "InitialMaxProbability": (
                    np.nan if scores.candidate_initial_max_probability is None
                    else float(scores.candidate_initial_max_probability[candidate])
                ),
            })
    result.smart_candidate_source_diagnostics = pd.DataFrame(source_rows)
    result.smart_standard_state_evidence_mode = state_evidence_mode
    if state_evidence_mode == _STANDARD_RAW_GL_MODE:
        result.smart_evidence_source = (
            "normalized per-contig zero/one/two-observed-parent forward HMMs "
            "from exact-position-aligned raw linear genotype likelihoods; "
            "tolerance_painting + reconstructed founder_block still define "
            "candidate screening, external ancestry states, and linked "
            "candidate fallback states; locally IBS-equivalent founder labels "
            "are pooled; no metadata; parent direction is resolved only for "
            "graph-conflicted rows using a phase-invariant chromosome-wide "
            "minimum-switch model over unique founder trajectories"
        )
    else:
        result.smart_evidence_source = (
            "normalized per-contig zero/one/two-observed-parent forward HMMs "
            "from tolerance_painting + reconstructed founder_block; locally "
            "IBS-equivalent founder labels are pooled; no metadata and no raw "
            "genotype likelihoods; parent direction is resolved only for "
            "graph-conflicted rows using a phase-invariant chromosome-wide "
            "minimum-switch model over unique founder trajectories"
        )
    if config.parent_state_candidate_source_mode == "exact_raw_gl_v1":
        result.smart_evidence_source = (
            "exact_raw_gl_v1 candidate founder-source posteriors inferred once "
            "per contig from normalized raw GLs, followed by exact conditional "
            "batch M0/M1/M2 child likelihoods. Existing child-left-out external "
            "chains, child information exponents, physical recombination, track "
            "switches, founders and marker counts are retained; candidate-source "
            "normalizers never enter pedigree scores."
        )
    if config.parent_state_candidate_source_mode == "matched_null_raw_gl_v2":
        result.smart_evidence_source = (
            "matched_null_raw_gl_v2 uses an ordered-independent candidate "
            "founder-source root and two independent synthetic null-parent "
            "draws from the same compound source process. Adult source-path "
            "rho is explicit and separate from the shared offspring "
            "transmission theta; painted hard-homo resets and candidate-source "
            "normalizers never enter M0/M1/M2 pedigree scores."
        )
    result.smart_candidate_screening_scope = (
        f"M0 plus every M1 identity and a fixed {len(trios):,}-trio M2 panel; "
        "the M2 panel was selected once from robust, information-weighted "
        "full-data pair-HMM scores. Bootstrap and LOCO are conditional on the "
        "screen, while the M2 identity prior still uses the full eligible "
        "pair count, making incomplete-screen M2 state evidence a labelled "
        "lower bound."
    )
    if config.parent_state_candidate_source_mode in {
        "exact_raw_gl_v1", "matched_null_raw_gl_v2"
    }:
        source_label = config.parent_state_candidate_source_mode
        result.smart_candidate_screening_scope = (
            f"M0 plus every {source_label} M1 identity and a fixed "
            f"{len(trios):,}-trio M2 panel selected from information-weighted "
            f"{source_label} M1 raw-GL likelihoods. Bootstrap and LOCO are "
            "conditional on this screen. The hard-painted pair screen is not "
            "consulted."
        )
    if (
        state_evidence_mode == _STANDARD_RAW_GL_MODE
        and config.parent_state_candidate_source_mode == "hard_painted"
    ):
        result.smart_candidate_screening_scope += (
            " The fixed candidate panel remains selected by the unchanged "
            "hard-painted pair HMM; raw likelihoods affect parent-state "
            "scoring only."
        )
    elif config.parent_state_candidate_source_mode in {
        "exact_raw_gl_v1", "matched_null_raw_gl_v2"
    }:
        result.smart_candidate_screening_scope += (
            " Candidate screening uses the selected probabilistic-source M1 "
            "raw-GL likelihoods; "
            "the hard-painted pair screen is not consulted. Founder-missing "
            "contigs use an explicit whole-contig hard/linked raw-GL fallback "
            "for both screening and state scoring."
        )
    result.smart_standard_input_threads = used_threads
    result.smart_standard_input_processes = state_processes
    result.smart_missing_parent_model = (
        "Comparable normalized forward likelihoods explicitly model zero, one, "
        "and two observed parents. Missing biological parents are linked "
        "external ancestry paths over locally IBS-pooled reconstructed "
        "haplotypes; zero observed parents does not assert founder status."
    )
    if config.parent_state_candidate_source_mode == "matched_null_raw_gl_v2":
        result.smart_missing_parent_model = (
            "Comparable matched-null likelihoods explicitly model zero, one, "
            "and two observed parents. Each missing biological parent is an "
            "independent synthetic draw from the same ordered-independent "
            "founder-source process used as the candidate prior; zero observed "
            "parents does not assert founder status."
        )
    if (
        state_evidence_mode == _STANDARD_RAW_GL_MODE
        and config.parent_state_candidate_source_mode == "hard_painted"
    ):
        result.smart_limitations += (
            " Standard-input parent-state evidence used explicitly supplied "
            "raw linear genotype likelihoods normalized after exact sample "
            "and selected-position alignment. The candidate screen remains "
            "hard-painted and is not independent validation of the raw-GL "
            "state scores. PL, AD, and HWE-posterior rows are not converted or "
            "silently reinterpreted. No post-hoc label consistency cutoff is "
            "applied to the parent-state result."
        )
    elif config.parent_state_candidate_source_mode == "exact_raw_gl_v1":
        result.smart_limitations += (
            " Exact-source candidate posteriors and child likelihoods use the "
            "same raw GL vectors in a conditional factorization; candidate HMM "
            "normalizers are excluded. Any founder-missing selected real site "
            "sends the entire contig through the unchanged hard/linked raw-GL "
            "scorer; independent likelihoods are never spliced."
        )
    elif config.parent_state_candidate_source_mode == "matched_null_raw_gl_v2":
        result.smart_limitations += (
            " Matched-null source posteriors and child likelihoods use the same "
            "raw GL vectors conditionally, without candidate HMM normalizers. "
            "The source-path rho is caller-specified because no metadata-free "
            "release estimator has yet been validated. Any founder-missing "
            "selected real site sends the whole contig through the unchanged "
            "hard/linked raw-GL fallback; likelihood families are not spliced."
        )
    else:
        result.smart_limitations += (
            " Standard-input evidence comes from reconstructed paintings "
            "rather than raw VCF genotype likelihoods. No post-hoc label "
            "consistency cutoff is applied to the parent-state result."
        )
    if eligibility.supplied:
        result.smart_evidence_source = result.smart_evidence_source.replace(
            "; no metadata;", "; caller-supplied parent eligibility;"
        ).replace(
            "; no metadata and no raw",
            "; caller-supplied parent eligibility and no raw",
        )
    return result


def _explicit_evidence_from_inputs(
    contig_data_list: Sequence[Any],
    sample_ids: Optional[Sequence[Any]] = None,
) -> Optional[list[SmartParentStateEvidence | SmartContigEvidence]]:
    """Return explicit parent-state evidence or leave standard inputs untouched."""
    if not contig_data_list:
        return None
    direct_records = [
        isinstance(item, (SmartParentStateEvidence, SmartContigEvidence))
        for item in contig_data_list
    ]
    if any(direct_records):
        if not all(direct_records):
            raise SmartEvidenceError(
                "explicit evidence records cannot be mixed with mapping inputs"
            )
        if sample_ids is None:
            raise SmartEvidenceError(
                "sample_ids are required for explicit pedigree evidence"
            )
        return [
            _as_parent_state_evidence(item, len(sample_ids))
            if isinstance(item, SmartParentStateEvidence) else item
            for item in contig_data_list
        ]

    present = [
        "smart_parent_state_evidence" in item for item in contig_data_list
    ]
    if any(present) and not all(present):
        raise SmartEvidenceError(
            "smart_parent_state_evidence must be supplied for every contig "
            "or for none"
        )
    if not all(present):
        return None
    if sample_ids is None:
        raise SmartEvidenceError(
            "sample_ids are required for parent-state evidence"
        )
    return [
        _as_parent_state_evidence(
            item["smart_parent_state_evidence"], len(sample_ids)
        )
        for item in contig_data_list
    ]


def infer_pedigree(
    contig_data_list: Sequence[Any],
    sample_ids: Sequence[Any],
    *,
    parent_eligibility: Optional[ParentEligibility | Mapping[str, Any]] = None,
    config: Optional[PedigreeConfig] = None,
    scoring_kwargs: Optional[Mapping[str, Any]] = None,
    ancestry_junction_counts: Optional[np.ndarray] = None,
    ancestry_callable_haplotype_bins: Optional[np.ndarray] = None,
) -> PedigreeResult:
    """Run the default parent-state pedigree engine.

    Accepted inputs are explicit :class:`ParentStateEvidence` records or
    the standard tolerance-painting plus founder-block schema. Unknown and
    pair-only schemas are rejected rather than delegated to another scientific
    model. ``scoring_kwargs`` configures only standard-input candidate
    screening and compact painting construction.
    """
    evidence = _explicit_evidence_from_inputs(contig_data_list, sample_ids)
    settings = _config_from_contig_inputs(contig_data_list, config)
    if evidence is not None:
        if not all(isinstance(item, ParentStateEvidence) for item in evidence):
            raise SmartEvidenceError(
                "the default engine requires ParentStateEvidence; "
                "pair-only SmartContigEvidence is not a supported inference path"
            )
        if scoring_kwargs:
            raise TypeError(
                "scoring_kwargs apply only to standard painting inputs"
            )
        return infer_from_parent_state_evidence(
            evidence,
            sample_ids,
            config=settings,
            parent_eligibility=parent_eligibility,
            ancestry_junction_counts=ancestry_junction_counts,
            ancestry_callable_haplotype_bins=ancestry_callable_haplotype_bins,
        )

    if _standard_input_schema(contig_data_list):
        if (
            ancestry_junction_counts is not None
            or ancestry_callable_haplotype_bins is not None
        ):
            raise TypeError(
                "ancestry matrices apply only to explicit parent-state evidence"
            )
        embedded_eligibility = _parent_eligibility_from_contig_inputs(
            contig_data_list
        )
        if parent_eligibility is not None and embedded_eligibility is not None:
            raise SmartEvidenceError(
                "parent eligibility was supplied both as an argument and in inputs"
            )
        effective_eligibility = (
            parent_eligibility
            if parent_eligibility is not None
            else embedded_eligibility
        )
        kwargs = dict(scoring_kwargs or {})
        result = _infer_standard_inputs(
            contig_data_list,
            sample_ids,
            settings,
            parent_eligibility=effective_eligibility,
            top_k=kwargs.pop("top_k", 20),
            snps_per_bin=kwargs.pop("snps_per_bin", 100),
            recomb_rate=kwargs.pop("recomb_rate", 5e-8),
            mismatch_penalty=kwargs.pop(
                "mismatch_penalty", DEFAULT_MISMATCH_PENALTY
            ),
            max_snps_per_bin=kwargs.pop("max_snps_per_bin", 10),
            n_workers=kwargs.pop("n_workers", None),
            anchor_k=kwargs.pop("anchor_k", 5),
            use_anchor_union=kwargs.pop("use_anchor_union", True),
        )
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected scoring_kwargs: {unexpected}")
        return result

    raise SmartEvidenceError(
        "pedigree inference requires explicit parent-state evidence or the "
        "standard tolerance-painting and founder-block input schema"
    )



_PIPELINE_CONTROL_SENTINEL = "0"
_PIPELINE_CONTROL_WARNING = (
    "pipeline_control_relationships is an internal control table for the "
    "unchanged phase-correction pipeline, not an export pedigree. Use "
    "result.relationships for scientific reporting."
)
_CONDITIONAL_SCREEN_WARNING = (
    "M2 identity stability is conditional on the fixed candidate panel; "
    "incomplete-screen combined B1 integrated evidence is a lower bound. "
    "State support and resampling "
    "fractions are not calibrated biological posterior probabilities."
)
_PIPELINE_DIAGNOSTIC_COLUMNS = (
    "Identifiable",
    "InformativeContigCount",
    "PairBootstrapFraction",
    "Parent1BootstrapFraction",
    "Parent2BootstrapFraction",
    "PairLOCOFraction",
    "Parent1LOCOFraction",
    "Parent2LOCOFraction",
    "UnconstrainedWinnerMargin",
    "SelectedAggregateUtility",
    "SelectedMinusUnconstrainedBest",
    "CandidatePairCount",
    "UnresolvedReason",
    "PairSupportSet",
    "TierAExactPair",
    "TierBExactPair",
    "SelectedParentState",
    "LocalWinnerParentState",
    "GraphTieConflict",
    "GraphDirectionResolvedAlternative",
    "SelectedParentRoleProbability",
    "AdjustedAncestryJunctionBurden",
    "AncestryPaintingCallabilityFraction",
    "LatentAncestryDepthMAP",
    "LatentAncestryDepthPosterior",
    "LocalStateBootstrapFraction",
    "LocalConfigurationBootstrapFraction",
    "GraphConfigurationBootstrapFraction",
    "LocalStateLOCOFraction",
    "M2StateEvidenceIsLowerBound",
    "TierAStateCall",
    "TierBStateCall",
)


def _require_pipeline_sentinel_available(sample_ids: Sequence[Any]) -> None:
    if any(str(sample) == _PIPELINE_CONTROL_SENTINEL for sample in sample_ids):
        raise SmartEvidenceError(
            "sample ID string '0' is reserved by the exact-signature "
            "pipeline_control_adapter"
        )


def _attach_pipeline_control_adapter(result: PedigreeResult) -> PedigreeResult:
    """Guard the unchanged downstream pipeline from unsupported parent rows.

    Only a Tier-B-supported, resolved, non-displaced M2 call is representable
    by the historical phase-correction interface. M0, M1, unresolved, weak
    complete-view, and graph-displaced rows
    receive the control sentinel.  In particular M0 is *not* converted to a
    both-null F1 row: zero observed parents does not establish founder-recolour
    eligibility when biological parents may simply be unsequenced.
    """
    scientific = result.relationships
    expected_samples = list(result.samples)
    if (
        len(scientific) != len(expected_samples)
        or scientific["Sample"].tolist() != expected_samples
    ):
        raise SmartEvidenceError(
            "smart scientific relationships must retain every sample in order"
        )
    has_first = scientific["Parent1"].notna()
    has_second = scientific["Parent2"].notna()
    if "ParentState" not in scientific.columns:
        raise SmartEvidenceError(
            "pipeline inference requires parent-state scientific relationships"
        )
    diagnostics = result.smart_diagnostics
    if diagnostics["Sample"].duplicated().any():
        raise SmartEvidenceError("smart diagnostics contain duplicate samples")
    diagnostic_lookup = diagnostics.set_index("Sample")
    tier_column = "TierBExactConfiguration"
    if tier_column not in diagnostic_lookup.columns:
        raise SmartEvidenceError(
            f"smart diagnostics lack required adapter gate {tier_column!r}"
        )
    tier_b_exact = scientific["Sample"].map(
        diagnostic_lookup[tier_column]
    )
    if tier_b_exact.isna().any():
        raise SmartEvidenceError(
            "smart diagnostics do not cover every scientific sample"
        )
    tier_b_exact = tier_b_exact.astype(bool)
    parent_state = scientific["ParentState"].astype(object)
    zero = parent_state.eq(_PARENT_STATE_NAMES[_ZERO_OBSERVED])
    one = parent_state.eq(_PARENT_STATE_NAMES[_ONE_OBSERVED])
    two = parent_state.eq(_PARENT_STATE_NAMES[_TWO_OBSERVED])
    unresolved_state = parent_state.eq("unresolved") | parent_state.isna()
    if np.any(zero & (has_first | has_second)):
        raise SmartEvidenceError("M0 scientific rows cannot contain parents")
    if np.any(one & has_first & has_second):
        raise SmartEvidenceError(
            "M1 scientific rows cannot contain two observed parents"
        )
    if np.any(two & (has_first ^ has_second)):
        raise SmartEvidenceError(
            "M2 scientific rows need an exact pair or unresolved identity"
        )
    if np.any(unresolved_state & (has_first | has_second)):
        raise SmartEvidenceError(
            "unresolved scientific rows cannot contain observed parents"
        )
    scientific_status = scientific["InferenceStatus"].astype(str)
    disallowed = scientific_status.str.contains(
        "graph_displaced|graph_conflict|unresolved", regex=True
    )
    pass_to_pipeline = (
        two & has_first & has_second & ~disallowed & tier_b_exact
    )

    scientific.attrs["table_role"] = "scientific_relationships"
    scientific.attrs["warning"] = (
        "ParentState distinguishes zero observed parents from unresolved "
        "identity where the parent-state model is available. Zero observed "
        "parents does not establish biological founder status."
    )
    adapter = scientific.copy(deep=True)
    adapter["ScientificParent1"] = scientific["Parent1"].astype(object)
    adapter["ScientificParent2"] = scientific["Parent2"].astype(object)
    adapter["ScientificParentState"] = parent_state
    adapter["ScientificInferenceStatus"] = scientific_status
    adapter["InferenceStatus"] = np.where(
        pass_to_pipeline,
        "resolved_exact_pair",
        "pipeline_control_sentinel_" + parent_state.astype(str),
    )

    for column in _PIPELINE_DIAGNOSTIC_COLUMNS:
        if column in diagnostic_lookup.columns:
            adapter[column] = adapter["Sample"].map(
                diagnostic_lookup[column]
            )
    adapter["ConditionalScreenWarning"] = _CONDITIONAL_SCREEN_WARNING
    adapter["ParentFieldRole"] = "pipeline_control_adapter_not_export"
    adapter["PipelineAdapterWarning"] = _PIPELINE_CONTROL_WARNING

    sentinel_rows = ~pass_to_pipeline
    adapter["Parent1"] = adapter["Parent1"].astype(object)
    adapter["Parent2"] = adapter["Parent2"].astype(object)
    adapter.loc[sentinel_rows, "Parent1"] = None
    adapter.loc[sentinel_rows, "Parent2"] = _PIPELINE_CONTROL_SENTINEL
    adapter.loc[sentinel_rows, "Generation"] = "Unknown"
    adapter.attrs["table_role"] = "pipeline_control_adapter"
    adapter.attrs["warning"] = _PIPELINE_CONTROL_WARNING

    result.scientific_relationships = result.relationships
    result.pipeline_control_relationships = adapter
    result.pipeline_control_adapter = True
    result.pipeline_control_adapter_name = "pipeline_control_adapter"
    result.pipeline_control_adapter_warning = _PIPELINE_CONTROL_WARNING
    result.pipeline_control_sentinel = _PIPELINE_CONTROL_SENTINEL
    result.relationships_is_export_pedigree = True
    result.pipeline_control_status_summary = {
        str(name): int(count)
        for name, count in adapter["InferenceStatus"].value_counts().items()
    }
    warnings.warn(_PIPELINE_CONTROL_WARNING, RuntimeWarning, stacklevel=2)
    return result


def infer_pedigree_for_pipeline(
    contig_data_list,
    sample_ids,
    top_k=20,
    snps_per_bin=100,
    recomb_rate=5e-8,
    mismatch_penalty=DEFAULT_MISMATCH_PENALTY,
    max_snps_per_bin=10,
    n_workers=None,
    anchor_k=5,
    use_anchor_union=True,
    *,
    parent_eligibility=None,
    config=None,
    ancestry_junction_counts=None,
    ancestry_callable_haplotype_bins=None,
):
    """Run the default engine and attach an explicit downstream-control table.

    ``result.relationships`` remains the configured scientific view. The
    historical phase-correction representation is available separately as
    ``result.pipeline_control_relationships`` and must be selected explicitly
    by a pipeline caller.
    """
    _require_pipeline_sentinel_available(sample_ids)
    scoring_kwargs = None
    if _standard_input_schema(contig_data_list):
        scoring_kwargs = {
            "top_k": top_k,
            "snps_per_bin": snps_per_bin,
            "recomb_rate": recomb_rate,
            "mismatch_penalty": mismatch_penalty,
            "max_snps_per_bin": max_snps_per_bin,
            "n_workers": n_workers,
            "anchor_k": anchor_k,
            "use_anchor_union": use_anchor_union,
        }
    result = infer_pedigree(
        contig_data_list,
        sample_ids,
        parent_eligibility=parent_eligibility,
        config=config,
        ancestry_junction_counts=ancestry_junction_counts,
        ancestry_callable_haplotype_bins=ancestry_callable_haplotype_bins,
        scoring_kwargs=scoring_kwargs,
    )
    return _attach_pipeline_control_adapter(result)


__all__ = [
    "DEFAULT_MISMATCH_PENALTY",
    "PARENT_ELIGIBILITY_FORMAT_VERSION",
    "ParentEligibility",
    "ParentStateEvidence",
    "PedigreeConfig",
    "PedigreeEvidenceError",
    "PedigreeResult",
    "draw_pedigree_tree",
    "infer_from_parent_state_evidence",
    "infer_pedigree",
    "infer_pedigree_for_pipeline",
    "load_bcf_raw_genotype_likelihoods",
    "prepare_standard_compact_raw_gl",
    "prepare_standard_compact_raw_gl_from_bcf",
    "score_parent_state_gl_hmms",
    "score_parent_state_hmms",
]

def _scoring_eligibility_masks(
    n_samples: int,
    eligible_children: Optional[np.ndarray],
    eligible_parents: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return dense private scoring masks without changing public defaults."""
    if eligible_children is None and eligible_parents is None:
        children = np.ones(n_samples, dtype=np.bool_)
        parents = np.ones((n_samples, n_samples), dtype=np.bool_)
        np.fill_diagonal(parents, False)
        return children, parents
    if eligible_children is None or eligible_parents is None:
        raise SmartEvidenceError(
            "eligible child and parent scoring masks must be supplied together"
        )
    children = np.asarray(eligible_children)
    parents = np.asarray(eligible_parents)
    if children.shape != (n_samples,) or children.dtype != np.bool_:
        raise SmartEvidenceError(
            "eligible child scoring mask must be boolean with shape (samples,)"
        )
    if (
        parents.shape != (n_samples, n_samples)
        or parents.dtype != np.bool_
    ):
        raise SmartEvidenceError(
            "eligible parent scoring mask must be boolean with shape "
            "(samples, samples)"
        )
    return (
        np.ascontiguousarray(children),
        np.ascontiguousarray(parents & children[:, None]),
    )
