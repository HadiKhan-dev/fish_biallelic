"""Scientific configuration for parent-count, identity and direction inference."""
from __future__ import annotations


from dataclasses import dataclass, field

import operator


import numpy as np

import haplotype_reconstruction.pedigree.models as pedigree_models

@dataclass(frozen=True)
class PedigreeConfig:
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
        default=pedigree_models._PARENT_STATE_LIKELIHOOD, init=False
    )
    parent_state_structure_mode: str = field(
        default=pedigree_models._PARENT_STATE_METHOD, init=False
    )
    parent_state_minimum_edge_coverage: float = 0.95
    parent_state_minimum_pair_explainability: float = 0.95
    parent_state_minimum_edge_exposed_bins: float = 1.0
    parent_state_minimum_pair_exposed_bins: float = 1.0
    parent_state_minimum_exposed_fraction: float = 0.10
    parent_state_minimum_exposed_contigs: int = 3
    parent_state_minimum_direction_probability: float = 0.01
    parent_state_direction_state_policy: str = field(default="strict_gate", init=False)
    parent_state_scaffold_policy: str = field(default="none", init=False)
    parent_state_candidate_source_mode: str = field(
        default=pedigree_models.RAGGED_QUADRATIC_MODEL, init=False
    )
    parent_state_cx_veto_policy: str = field(default="source_default", init=False)
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

    def validated(self) -> "PedigreeConfig":
        def require_integer(name: str, minimum: int) -> None:
            raw_value = getattr(self, name)
            if isinstance(raw_value, (bool, np.bool_)):
                raise pedigree_models.PedigreeEvidenceError(
                    f"{name} must be an integer of at least {minimum}"
                )
            try:
                value = operator.index(raw_value)
            except TypeError as exc:
                raise pedigree_models.PedigreeEvidenceError(
                    f"{name} must be an integer of at least {minimum}"
                ) from exc
            if value < minimum:
                raise pedigree_models.PedigreeEvidenceError(
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
            raise pedigree_models.PedigreeEvidenceError(
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
                raise pedigree_models.PedigreeEvidenceError(f"{name} must be finite")
        if not 0.0 <= self.rank_weight <= 1.0:
            raise pedigree_models.PedigreeEvidenceError("rank_weight must lie in [0, 1]")
        if not 0.0 <= self.chromosome_contamination < 1.0:
            raise pedigree_models.PedigreeEvidenceError(
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
                raise pedigree_models.PedigreeEvidenceError(f"{name} must lie in [0, 1]")
        if self.linked_evidence_weight < 0.0 or self.genotype_evidence_weight < 0.0:
            raise pedigree_models.PedigreeEvidenceError("evidence weights must be non-negative")
        if self.linked_evidence_weight + self.genotype_evidence_weight <= 0.0:
            raise pedigree_models.PedigreeEvidenceError("at least one evidence weight must be positive")
        if self.primary_view not in {"tier_a", "tier_b", "complete"}:
            raise pedigree_models.PedigreeEvidenceError(
                "primary_view must be 'tier_a', 'tier_b', or 'complete'"
            )
        # Old pickles can carry values for init=False fields. Reject them
        # explicitly rather than silently treating a retired configuration as
        # the combined method.
        if self.parent_state_algorithm_mode != pedigree_models._PARENT_STATE_LIKELIHOOD:
            raise pedigree_models.PedigreeEvidenceError(
                "only the combined_v1 pedigree method with internal B1 "
                "likelihood evidence is supported"
            )
        if self.parent_state_structure_mode != pedigree_models._PARENT_STATE_METHOD:
            raise pedigree_models.PedigreeEvidenceError(
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
                raise pedigree_models.PedigreeEvidenceError(f"{name} must lie in [0, 1]")
        for name in (
            "parent_state_minimum_edge_exposed_bins",
            "parent_state_minimum_pair_exposed_bins",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise pedigree_models.PedigreeEvidenceError(
                    f"{name} must be finite and non-negative"
                )


        if (
            not np.isfinite(self.parent_state_mismatch_probability)
            or not 0.0 < self.parent_state_mismatch_probability < 0.5
        ):
            raise pedigree_models.PedigreeEvidenceError(
                "parent_state_mismatch_probability must lie in (0, 0.5)"
            )
        if (
            not np.isfinite(self.parent_state_phase_switch_probability)
            or not 0.0 <= self.parent_state_phase_switch_probability <= 0.5
        ):
            raise pedigree_models.PedigreeEvidenceError(
                "parent_state_phase_switch_probability must lie in [0, 0.5]"
            )
        if (
            not np.isfinite(self.parent_state_contamination_probability)
            or not 0.0 <= self.parent_state_contamination_probability < 1.0
        ):
            raise pedigree_models.PedigreeEvidenceError(
                "parent_state_contamination_probability must lie in [0, 1)"
            )
        for name in (
            "parent_state_effective_markers_per_information_block",
            "parent_state_external_state_pseudocount",
            "parent_state_external_transition_pseudocount",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise pedigree_models.PedigreeEvidenceError(f"{name} must be finite and positive")
        priors = np.asarray(self.parent_state_priors, dtype=np.float64)
        if (
            priors.shape != (3,)
            or np.any(~np.isfinite(priors))
            or np.any(priors <= 0.0)
            or not np.isclose(np.sum(priors), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise pedigree_models.PedigreeEvidenceError(
                "parent_state_priors must be three positive probabilities "
                "summing to one"
            )
        if (
            not np.isfinite(self.parent_state_prior_strength)
            or self.parent_state_prior_strength <= 0.0
        ):
            raise pedigree_models.PedigreeEvidenceError(
                "parent_state_prior_strength must be finite and positive"
            )
        if (
            not np.isfinite(self.parent_state_prior_tolerance)
            or self.parent_state_prior_tolerance <= 0.0
        ):
            raise pedigree_models.PedigreeEvidenceError(
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
            raise pedigree_models.PedigreeEvidenceError(
                "parent_state_prior_sensitivity must contain positive "
                "probability triples summing to one"
            )
        return self
