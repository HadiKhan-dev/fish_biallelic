"""Prepare component-local paintings and raw likelihoods for pedigree scoring."""
from __future__ import annotations
from haplotype_reconstruction import PACKAGE_ROOT

from dataclasses import dataclass, field
import hashlib
import json
import math

import time
from typing import Any, Mapping, Sequence
import numpy as np
from numba import njit, prange
import haplotype_reconstruction.pedigree.candidates as pedigree_candidates
import haplotype_reconstruction.pedigree.models as pedigree_models
import haplotype_reconstruction.pedigree.sources as pedigree_sources
import haplotype_reconstruction.pedigree.transmission as pedigree_transmission

_HARD_PAINTED_MODE = "hard_painted"


_T09_LOG_FLOOR = -50.0


_EVIDENCE_SCORE_IDENTITY_SCHEMA = "t10-parent-state-score-identity-v1"


_EVIDENCE_SCORE_CODE_VERSION = "component-local-parent-state-score-v1"


_EVIDENCE_SCORE_CODE_FILES = (
    'pedigree/sources.py',
    'pedigree/transmission.py',
    'pedigree/transmission_projection.py',
    'pedigree/transmission_scoring.py',
    'core/genetic_map.py',
    'painting/model.py'
)


@dataclass(frozen=True)
class PreparedComponentPedigree:
    """One independently rooted component and its direct-evidence labels."""

    component_index: int
    cache: pedigree_models.ComponentEvidenceArrays
    direct_labels: np.ndarray
    source_mode: str = _HARD_PAINTED_MODE
    ragged_model: pedigree_sources.RaggedFounderModel | None = None
    ragged_source_factors: pedigree_sources.RaggedSourceBatchFactors | None = None
    compact_genotype_likelihoods: np.ndarray | None = None
    compact_observed: np.ndarray | None = None
    ragged_anchored_states: np.ndarray | None = None


@dataclass(frozen=True)
class PreparedChromosome:
    """Validated component caches plus chromosome-global information weights."""

    contig: str
    sample_ids: tuple[str, ...]
    components: tuple[PreparedComponentPedigree, ...]
    information_exponents: tuple[np.ndarray, ...]
    component_count: int
    omitted_reason: str | None = None
    source_mode: str = _HARD_PAINTED_MODE
    source_identity: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ComponentPedigreeChromosomeResult:
    """Exactly zero or one T10 evidence record for one physical chromosome."""

    contig: str
    evidence: pedigree_models.ParentStateEvidence | None
    state_scores: pedigree_candidates.ChromosomeLikelihoods | None
    component_count: int
    scored_component_count: int
    informative_markers: int
    omitted_reason: str | None = None
    ragged_component_diagnostics: tuple[
        "RaggedComponentScoringDiagnostics", ...
    ] = ()
    projected_component_diagnostics: tuple[
        "ProjectedComponentScoringDiagnostics", ...
    ] = ()

    @property
    def ragged_scoring_diagnostics(self) -> "RaggedScoringDiagnostics | None":
        return _aggregate_ragged_diagnostics(
            getattr(self, "ragged_component_diagnostics", ())
        )

    @property
    def projected_scoring_diagnostics(
            self,
    ) -> "ProjectedScoringDiagnostics | None":
        return _aggregate_projected_diagnostics(
            getattr(self, "projected_component_diagnostics", ())
        )


@dataclass(frozen=True)
class OmittedT09Chromosome:
    """A physical chromosome excluded because it contains no evidence."""

    contig: str
    component_count: int
    reason: str


@dataclass(frozen=True)
class PreparedPedigree:
    """Ordered informative chromosomes and explicitly omitted chromosomes."""
    recombination_rate: float
    max_snps_per_bin: int
    markers_per_information_block: int
    effective_markers_per_information_block: float

    sample_ids: tuple[str, ...]
    chromosomes: tuple[PreparedChromosome, ...]
    omitted_chromosomes: tuple[OmittedT09Chromosome, ...]
    source_mode: str = _HARD_PAINTED_MODE
    input_preparation_seconds: float = 0.0


@dataclass(frozen=True)
class ScoredT09ChromosomeEvidence:
    """Lean replay evidence for one physical chromosome.

    The common trio panel is stored once at run level. Candidate posterior and
    scorer factor tensors are deliberately absent: neither is consumed by
    parent-state policy aggregation.
    """

    contig: str
    zero_parent_log_likelihoods: np.ndarray
    one_parent_log_likelihoods: np.ndarray
    two_parent_log_likelihoods: np.ndarray
    edge_matched_bins: np.ndarray
    edge_exposed_bins: np.ndarray
    pair_explained_bins: np.ndarray
    pair_exposed_bins: np.ndarray
    structure_total_bins: float
    ancestry_junction_counts: np.ndarray
    ancestry_callable_haplotype_bins: np.ndarray
    informative_markers: int
    component_count: int
    scored_component_count: int
    one_parent_identity_information: np.ndarray | None = None
    two_parent_edge_information: np.ndarray | None = None
    candidate_source_mode_requested: str = _HARD_PAINTED_MODE
    candidate_source_mode_applied: str = _HARD_PAINTED_MODE
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
    ragged_component_diagnostics: tuple[
        "RaggedComponentScoringDiagnostics", ...
    ] = ()
    projected_component_diagnostics: tuple[
        "ProjectedComponentScoringDiagnostics", ...
    ] = ()


@dataclass(frozen=True)
class ScoredT09ParentStateEvidence:
    """All score-stage products needed for decision-policy replay."""

    sample_ids: tuple[str, ...]
    contig_names: tuple[str, ...]
    chromosomes: tuple[ScoredT09ChromosomeEvidence, ...]
    trios: np.ndarray
    parent_screen_scores: np.ndarray
    omitted_chromosomes: tuple[OmittedT09Chromosome, ...]
    parent_panel_diagnostics: "AdaptiveParentPanelDiagnostics"
    source_mode: str
    score_identity: Mapping[str, Any]
    input_preparation_seconds: float = 0.0
    screening_and_scoring_seconds: float = 0.0
    runtime_chromosome_results: (
        tuple[ComponentPedigreeChromosomeResult, ...] | None
    ) = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class Stage10ChromosomeEvidenceRequest:
    """Identity presented to an optional per-chromosome cache callback."""

    contig: str
    run_score_identity: Mapping[str, Any]
    chromosome_score_identity_sha256: str
    trio_count: int


@dataclass(frozen=True)
class ComponentPedigreeExecutionDiagnostics:
    """Wall-time attribution for one prepared component-pedigree run."""

    input_preparation_seconds: float
    screening_and_scoring_seconds: float
    parent_state_aggregation_bootstrap_seconds: float
    prepared_inference_seconds: float
    end_to_end_seconds: float


@dataclass(frozen=True)
class ComponentPedigreeRunResult:
    """Genome-wide inference plus its physical-chromosome evidence seam."""

    pedigree_result: Any
    chromosome_results: tuple[ComponentPedigreeChromosomeResult, ...]
    trios: np.ndarray
    parent_screen_scores: np.ndarray
    omitted_chromosomes: tuple[OmittedT09Chromosome, ...]
    parent_panel_diagnostics: "AdaptiveParentPanelDiagnostics | None" = None
    execution_diagnostics: "ComponentPedigreeExecutionDiagnostics | None" = None

    @property
    def adaptive_panel_diagnostics(self) -> "AdaptiveParentPanelDiagnostics | None":
        return self.parent_panel_diagnostics

    @property
    def ragged_scoring_diagnostics(self) -> "RaggedScoringDiagnostics | None":
        return _aggregate_ragged_diagnostics(tuple(
            component_diagnostic
            for chromosome in self.chromosome_results
            for component_diagnostic in getattr(
                chromosome, "ragged_component_diagnostics", ()
            )
        ))

    @property
    def projected_scoring_diagnostics(
            self,
    ) -> "ProjectedScoringDiagnostics | None":
        return _aggregate_projected_diagnostics(tuple(
            component_diagnostic
            for chromosome in self.chromosome_results
            for component_diagnostic in getattr(
                chromosome, "projected_component_diagnostics", ()
            )
        ))


@dataclass(frozen=True)
class AdaptiveParentPanelDiagnostics:
    """Selection-size and fallback diagnostics for the M2 candidate panel."""

    requested_initial_top_k: int | None
    outer_top_k: int
    informative_chromosome_count: int
    adaptive_requested: bool
    adaptive_applied: bool
    adaptive_fallback: bool
    fallback_reason: str | None
    selected_k_by_child: tuple[int, ...]
    selected_k_distribution: tuple[tuple[int, int], ...]
    trio_row_count: int


@dataclass(frozen=True)
class RaggedComponentScoringDiagnostics:
    """Exact batch phases for one independently scored T09 component."""

    component_index: int
    source_preparation_seconds: float
    source_marginal_preparation_seconds: float
    hard_structure_seconds: float
    posterior_expected_structure_seconds: float
    transmission_preparation_seconds: float
    m0_scoring_seconds: float
    m1_scoring_seconds: float
    m2_scoring_seconds: float
    m2_active_trio_count: int
    m2_reduced_trio_count: int
    m2_batch_size: int
    m2_batch_count: int
    m2_bytes_per_active_task: int
    peak_m2_working_bytes: int
    transmission_probability_bytes: int
    reused_screen_scores: bool


@dataclass(frozen=True)
class RaggedScoringDiagnostics:
    """Additive phase timings and bounded-memory maxima for a larger scope."""

    component_count: int
    source_preparation_seconds: float
    source_marginal_preparation_seconds: float
    hard_structure_seconds: float
    posterior_expected_structure_seconds: float
    transmission_preparation_seconds: float
    m0_scoring_seconds: float
    m1_scoring_seconds: float
    m2_scoring_seconds: float
    m2_active_trio_count: int
    m2_reduced_trio_count: int
    m2_batch_count: int
    maximum_m2_batch_size: int
    maximum_m2_bytes_per_active_task: int
    peak_m2_working_bytes: int
    maximum_transmission_probability_bytes: int
    reused_screen_component_count: int


def _aggregate_ragged_diagnostics(
        values: Sequence[RaggedComponentScoringDiagnostics],
) -> RaggedScoringDiagnostics | None:
    if not values:
        return None
    return RaggedScoringDiagnostics(
        component_count=len(values),
        source_preparation_seconds=sum(
            value.source_preparation_seconds for value in values
        ),
        source_marginal_preparation_seconds=sum(
            value.source_marginal_preparation_seconds for value in values
        ),
        hard_structure_seconds=sum(
            value.hard_structure_seconds for value in values
        ),
        posterior_expected_structure_seconds=sum(
            value.posterior_expected_structure_seconds for value in values
        ),
        transmission_preparation_seconds=sum(
            value.transmission_preparation_seconds for value in values
        ),
        m0_scoring_seconds=sum(value.m0_scoring_seconds for value in values),
        m1_scoring_seconds=sum(value.m1_scoring_seconds for value in values),
        m2_scoring_seconds=sum(value.m2_scoring_seconds for value in values),
        m2_active_trio_count=sum(value.m2_active_trio_count for value in values),
        m2_reduced_trio_count=sum(
            value.m2_reduced_trio_count for value in values
        ),
        m2_batch_count=sum(value.m2_batch_count for value in values),
        maximum_m2_batch_size=max(value.m2_batch_size for value in values),
        maximum_m2_bytes_per_active_task=max(
            value.m2_bytes_per_active_task for value in values
        ),
        peak_m2_working_bytes=max(
            value.peak_m2_working_bytes for value in values
        ),
        maximum_transmission_probability_bytes=max(
            value.transmission_probability_bytes for value in values
        ),
        reused_screen_component_count=sum(
            int(value.reused_screen_scores) for value in values
        ),
    )


@dataclass(frozen=True)
class ProjectedComponentScoringDiagnostics:
    """Projected quadratic phases for one independently scored component."""

    approximation_name: str
    component_index: int
    source_preparation_seconds: float
    source_marginal_preparation_seconds: float
    hard_structure_seconds: float
    posterior_expected_structure_seconds: float
    projection_preparation_seconds: float
    m0_scoring_seconds: float
    m1_scoring_seconds: float
    m2_scoring_seconds: float
    m2_active_trio_count: int
    m2_reduced_trio_count: int
    projection_retained_bytes: int
    projected_hidden_state_count: int
    exact_m2_hidden_state_count: int
    maximum_bridge_marginal_residual: float
    bridge_branch_counts: tuple[int, int, int, int]
    maximum_bridge_solver_iterations: int
    reused_screen_scores: bool


@dataclass(frozen=True)
class ProjectedScoringDiagnostics:
    """Aggregate projected quadratic timing, state, and bridge diagnostics."""

    approximation_name: str
    component_count: int
    source_preparation_seconds: float
    source_marginal_preparation_seconds: float
    hard_structure_seconds: float
    posterior_expected_structure_seconds: float
    projection_preparation_seconds: float
    m0_scoring_seconds: float
    m1_scoring_seconds: float
    m2_scoring_seconds: float
    m2_active_trio_count: int
    m2_reduced_trio_count: int
    maximum_projection_retained_bytes: int
    maximum_projected_hidden_state_count: int
    maximum_exact_m2_hidden_state_count: int
    maximum_bridge_marginal_residual: float
    bridge_branch_counts: tuple[int, int, int, int]
    maximum_bridge_solver_iterations: int
    reused_screen_component_count: int


def _aggregate_projected_diagnostics(
        values: Sequence[ProjectedComponentScoringDiagnostics],
) -> ProjectedScoringDiagnostics | None:
    if not values:
        return None
    return ProjectedScoringDiagnostics(
        approximation_name=pedigree_transmission.APPROXIMATION_NAME,
        component_count=len(values),
        source_preparation_seconds=sum(
            value.source_preparation_seconds for value in values
        ),
        source_marginal_preparation_seconds=sum(
            value.source_marginal_preparation_seconds for value in values
        ),
        hard_structure_seconds=sum(
            value.hard_structure_seconds for value in values
        ),
        posterior_expected_structure_seconds=sum(
            value.posterior_expected_structure_seconds for value in values
        ),
        projection_preparation_seconds=sum(
            value.projection_preparation_seconds for value in values
        ),
        m0_scoring_seconds=sum(value.m0_scoring_seconds for value in values),
        m1_scoring_seconds=sum(value.m1_scoring_seconds for value in values),
        m2_scoring_seconds=sum(value.m2_scoring_seconds for value in values),
        m2_active_trio_count=sum(value.m2_active_trio_count for value in values),
        m2_reduced_trio_count=sum(
            value.m2_reduced_trio_count for value in values
        ),
        maximum_projection_retained_bytes=max(
            value.projection_retained_bytes for value in values
        ),
        maximum_projected_hidden_state_count=max(
            value.projected_hidden_state_count for value in values
        ),
        maximum_exact_m2_hidden_state_count=max(
            value.exact_m2_hidden_state_count for value in values
        ),
        maximum_bridge_marginal_residual=max(
            value.maximum_bridge_marginal_residual for value in values
        ),
        bridge_branch_counts=tuple(
            sum(value.bridge_branch_counts[index] for value in values)
            for index in range(4)
        ),
        maximum_bridge_solver_iterations=max(
            value.maximum_bridge_solver_iterations for value in values
        ),
        reused_screen_component_count=sum(
            int(value.reused_screen_scores) for value in values
        ),
    )


@dataclass(frozen=True)
class _ProjectedParentScreen:
    """Projected M1 values plus reusable projection and exact marginals."""

    one_observed: np.ndarray
    batch_scores: pedigree_transmission.ProjectedRaggedQuadraticScores
    projected_model: pedigree_transmission.ProjectedRaggedQuadraticModel
    structure_marginals: np.ndarray
    source_marginal_preparation_seconds: float


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise pedigree_models.PedigreeEvidenceError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise pedigree_models.PedigreeEvidenceError(f"{name} must be a positive integer") from exc
    if result != value or result < 1:
        raise pedigree_models.PedigreeEvidenceError(f"{name} must be a positive integer")
    return result


def _resolve_source_mode(settings, requested):
    """Use the canonical ragged quadratic candidate-source model."""
    if requested not in (None, pedigree_models.RAGGED_QUADRATIC_MODEL):
        raise pedigree_models.PedigreeEvidenceError("only ragged quadratic candidate sources are supported")
    return pedigree_models.RAGGED_QUADRATIC_MODEL


def _t09_painting_config(checkpoint: Any) -> dict[str, Any]:
    """Return the exact scientific T09 painting parameters."""

    try:
        record = checkpoint.painting_product_identity.record()
        config = record["config"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise pedigree_models.PedigreeEvidenceError(
            "T09 painting identity lacks its scientific config"
        ) from exc
    required = (
        "recombination_rate",
        "switch_penalty_per_snp",
        "robustness_epsilon",
        "double_recomb_factor",
        "snps_per_bin",
    )
    if not isinstance(config, dict) or any(name not in config for name in required):
        raise pedigree_models.PedigreeEvidenceError(
            "T09 painting identity lacks exact ragged-HMM parameters"
        )
    try:
        values = {
            "recombination_rate": float(config["recombination_rate"]),
            "switch_penalty_per_snp": float(config["switch_penalty_per_snp"]),
            "robustness_epsilon": float(config["robustness_epsilon"]),
            "double_recomb_factor": float(config["double_recomb_factor"]),
            "snps_per_bin": _positive_integer(
                config["snps_per_bin"], "T09 snps_per_bin"
            ),
        }
    except (TypeError, ValueError, OverflowError) as exc:
        raise pedigree_models.PedigreeEvidenceError(
            "T09 painting identity has invalid ragged-HMM parameters"
        ) from exc
    if (
        any(not math.isfinite(values[name]) or values[name] < 0.0 for name in (
            "recombination_rate", "switch_penalty_per_snp",
            "robustness_epsilon", "double_recomb_factor",
        ))
        or values["robustness_epsilon"] > 1.0
        or values["double_recomb_factor"] <= 0.0
    ):
        raise pedigree_models.PedigreeEvidenceError(
            "T09 painting identity has invalid ragged-HMM parameters"
        )
    if "genetic_map" in config:

        values["chromosome_map"] = core_genetic_map.ChromosomeGeneticMap(**config["genetic_map"])
    return values


def _array_digest(value: Any) -> str:
    """Match the Stage-2 release array identity exactly."""

    array = np.asarray(value)
    contiguous = array if array.flags.c_contiguous else np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    raw = memoryview(contiguous).cast("B")
    chunk_bytes = 64 * 1024 * 1024
    for start in range(0, len(raw), chunk_bytes):
        digest.update(raw[start:start + chunk_bytes])
    return digest.hexdigest()


def _canonical_identity(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise pedigree_models.PedigreeEvidenceError(f"{name} must be a mapping")
    try:
        return json.loads(json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ))
    except (TypeError, ValueError) as exc:
        raise pedigree_models.PedigreeEvidenceError(f"{name} must be canonical JSON data") from exc


def _identity_digest(value: Mapping[str, Any]) -> str:
    canonical = _canonical_identity(value, "identity")
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stage10_evidence_scoring_code_identity(
        digest_overrides: Mapping[str, str] | None=None,
) -> dict[str, Any]:
    """Return the code identity for evidence preparation and scoring only.

    Decision-policy code is intentionally outside this identity. The semantic
    version covers private scoring helpers imported from ``pedigree_inference``;
    it must be bumped if one of those helpers changes scientifically.
    """

    overrides = {} if digest_overrides is None else dict(digest_overrides)
    unknown = set(overrides).difference(_EVIDENCE_SCORE_CODE_FILES)
    if unknown:
        raise pedigree_models.PedigreeEvidenceError(
            "unknown evidence-scoring code identity overrides: "
            + ", ".join(sorted(unknown))
        )
    root = PACKAGE_ROOT
    files = {}
    for filename in _EVIDENCE_SCORE_CODE_FILES:
        digest = overrides.get(filename)
        if digest is None:
            digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        digest = str(digest).lower()
        if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest):
            raise pedigree_models.PedigreeEvidenceError(f"invalid SHA256 identity for {filename}")
        files[filename] = digest
    return {
        "semantic_version": _EVIDENCE_SCORE_CODE_VERSION,
        "file_sha256": files,
    }


def _t09_source_identity(checkpoint: Any) -> dict[str, Any]:
    release = checkpoint.release_identity.record()
    painting = checkpoint.painting_product_identity.record()
    return _canonical_identity({
        "t09_release_identity": release,
        "t09_painting_product_identity": painting,
        "raw_input_array_sha256": release.get("input_array_sha256"),
    }, "T09/raw source identity")


def _score_config_identity(settings: module_pedigree_config.PedigreeConfig) -> dict[str, Any]:
    names = (
        "markers_per_information_block",
        "information_tempering_power",
        "maximum_contig_weight_ratio",
        "rank_weight",
        "chromosome_contamination",
        "parent_state_candidate_source_mode",
        "parent_state_candidate_source_path_switch_probability",
        "parent_state_mismatch_probability",
        "parent_state_phase_switch_probability",
        "parent_state_effective_markers_per_information_block",
        "parent_state_external_state_pseudocount",
        "parent_state_external_transition_pseudocount",
    )
    return {name: getattr(settings, name) for name in names}


def _eligibility_score_identity(eligibility: Any) -> dict[str, Any]:
    pairs = eligibility.eligible_parent_pairs
    return {
        "sample_ids": list(eligibility.sample_ids),
        "eligible_children_sha256": _array_digest(
            eligibility.eligible_children
        ),
        "eligible_parents_sha256": _array_digest(
            eligibility.eligible_parents
        ),
        "eligible_parent_pairs_sha256": (
            None if pairs is None else _array_digest(pairs)
        ),
        "pair_policy": str(eligibility.pair_policy),
    }


def _parent_state_score_identity(
        prepared: PreparedPedigree,
        settings: module_pedigree_config.PedigreeConfig,
        eligibility: Any,
        trios: np.ndarray,
        *,
        top_k: int,
        adaptive_initial_top_k: int | None,
        anchor_k: int,
        use_anchor_union: bool,
        mismatch_penalty: float,
        external_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    sources = []
    for chromosome in prepared.chromosomes:
        source = getattr(chromosome, "source_identity", None)
        sources.append({
            "contig": chromosome.contig,
            "source_identity": (
                None if source is None else _canonical_identity(
                    source, f"{chromosome.contig} source identity"
                )
            ),
        })
    return _canonical_identity({
        "schema": _EVIDENCE_SCORE_IDENTITY_SCHEMA,
        "ordered_sample_ids": prepared.sample_ids,
        "ordered_informative_contigs": tuple(
            chromosome.contig for chromosome in prepared.chromosomes
        ),
        "omitted_contigs": tuple(
            value.contig for value in prepared.omitted_chromosomes
        ),
        "sources": sources,
        "preparation": {
            "recombination_rate": prepared.recombination_rate,
            "max_snps_per_bin": prepared.max_snps_per_bin,
        },
        "scoring_config": _score_config_identity(settings),
        "scoring_eligibility": _eligibility_score_identity(eligibility),
        "candidate_panel": {
            "top_k": int(top_k),
            "adaptive_initial_top_k": adaptive_initial_top_k,
            "anchor_k": int(anchor_k),
            "use_anchor_union": bool(use_anchor_union),
            "trios_sha256": _array_digest(trios),
        },
        "hard_screen_mismatch_penalty": float(mismatch_penalty),
        "scoring_code_identity": stage10_evidence_scoring_code_identity(),
        "external_identity": (
            None if external_identity is None else _canonical_identity(
                external_identity, "external evidence identity"
            )
        ),
    }, "parent-state score identity")


def _validate_release_array_identity(
        checkpoint: Any,
        genotype_likelihoods: np.ndarray,
        positions: np.ndarray,
        observed: np.ndarray,
) -> None:
    release = checkpoint.release_identity.record()
    expected = release.get("input_array_sha256")
    if (not isinstance(expected, dict)
            or set(expected) != {
                "global_probs", "global_sites", "global_observed_mask"
            }):
        raise pedigree_models.PedigreeEvidenceError(
            "T09 release lacks the exact three-array input identity"
        )
    arrays = {
        "global_probs": genotype_likelihoods,
        "global_sites": positions,
        "global_observed_mask": observed,
    }
    for name, value in arrays.items():
        digest = expected.get(name)
        if not isinstance(digest, str) or _array_digest(value) != digest:
            raise pedigree_models.PedigreeEvidenceError(
                f"raw {name} does not match the typed T09 release identity"
            )


@njit(cache=True, parallel=True, fastmath=False)
def _normalize_raw_evidence(raw_gl, observed):
    """Fuse strict raw-row validation, float64 normalization and missingness."""
    samples, sites, _ = raw_gl.shape
    normalized = np.empty((samples, sites, 3), dtype=np.float64)
    invalid = 0
    for row in prange(samples * sites):
        sample, site = row // sites, row % sites
        a = np.float64(raw_gl[sample, site, 0])
        b = np.float64(raw_gl[sample, site, 1])
        c = np.float64(raw_gl[sample, site, 2])
        # Match the float64 three-genotype NumPy sum without reassociation.
        total = (a + b) + c
        valid = (
            np.isfinite(a) and np.isfinite(b) and np.isfinite(c)
            and a >= 0.0 and b >= 0.0 and c >= 0.0
            and np.isfinite(total) and total > 0.0
        )
        invalid += int(not valid)
        # Unobserved rows still have to satisfy the raw evidence contract.
        if valid and observed[sample, site]:
            normalized[sample, site, 0] = a / total
            normalized[sample, site, 1] = b / total
            normalized[sample, site, 2] = c / total
        else:
            normalized[sample, site, 0] = 1.0 / 3.0
            normalized[sample, site, 1] = 1.0 / 3.0
            normalized[sample, site, 2] = 1.0 / 3.0
    return normalized, invalid


def _validated_raw_evidence(
        checkpoint,
        raw_genotype_likelihoods: Any,
        raw_positions: Any,
        raw_observed_mask: Any,
        raw_sample_ids: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_ids = checkpoint.sample_ids
    if isinstance(raw_sample_ids, (str, bytes)):
        raise pedigree_models.PedigreeEvidenceError("raw_sample_ids must be an ordered sequence")
    try:
        observed_ids = tuple(str(value) for value in raw_sample_ids)
    except TypeError as exc:
        raise pedigree_models.PedigreeEvidenceError(
            "raw_sample_ids must be an ordered sequence"
        ) from exc
    if observed_ids != expected_ids:
        raise pedigree_models.PedigreeEvidenceError(
            "raw_sample_ids must exactly match the typed T09 sample order"
        )

    raw_position_values = np.asarray(raw_positions)
    if (
        raw_position_values.ndim != 1
        or len(raw_position_values) < 1
        or np.any(~np.isfinite(raw_position_values))
        or np.any(raw_position_values != np.floor(raw_position_values))
    ):
        raise pedigree_models.PedigreeEvidenceError(
            "raw_positions must be a nonempty vector of integer coordinates"
        )
    positions = np.ascontiguousarray(raw_position_values, dtype=np.int64)
    if len(positions) > 1 and np.any(positions[1:] <= positions[:-1]):
        raise pedigree_models.PedigreeEvidenceError(
            "raw_positions must be strictly increasing and unique"
        )

    raw_gl = np.asarray(raw_genotype_likelihoods)
    expected_shape = (len(expected_ids), len(positions), 3)
    if raw_gl.shape != expected_shape:
        raise pedigree_models.PedigreeEvidenceError(
            "raw_genotype_likelihoods must have shape "
            "(T09 samples, raw positions, 3)"
        )
    try:
        # Native float32/64 inputs can be cast row-wise inside the fused pass.
        # Keep NumPy's established conversion for other supported input dtypes.
        gl = (
            raw_gl if raw_gl.dtype in (np.dtype(np.float32), np.dtype(np.float64))
            else np.asarray(raw_gl, dtype=np.float64)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise pedigree_models.PedigreeEvidenceError(
            "raw genotype likelihoods must be finite non-negative values"
        ) from exc
    observed = np.asarray(raw_observed_mask)
    if observed.dtype != np.dtype(np.bool_) or observed.shape != expected_shape[:2]:
        raise pedigree_models.PedigreeEvidenceError(
            "raw_observed_mask must be an exact boolean T09-sample-by-site mask"
        )
    normalized, invalid = _normalize_raw_evidence(gl, observed)
    if invalid:
        raise pedigree_models.PedigreeEvidenceError(
            "every raw genotype-likelihood row must have positive finite mass"
        )
    _validate_release_array_identity(
        checkpoint,
        raw_gl,
        raw_position_values,
        observed,
    )
    observed = np.ascontiguousarray(observed, dtype=np.bool_)
    return normalized, positions, observed


def _diagnostic_arrays(
        diagnostic: Any,
        component_index: int,
        component: Any,
        n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"component {component_index}"
    try:
        positions = np.asarray(diagnostic.selected_positions)
        site_indices = np.asarray(diagnostic.selected_site_indices)
        centers = np.asarray(diagnostic.bin_centers, dtype=np.float64)
        edges = np.asarray(diagnostic.bin_edges)
        named_alleles = np.asarray(diagnostic.named_alleles)
        equivalence_classes = tuple(diagnostic.equivalence_classes)
        label_grid = np.asarray(diagnostic.map_label_grid)
        state_grid = np.asarray(diagnostic.map_state_class_grid)
        statuses = np.asarray(diagnostic.track_status_grid)
        direct = np.asarray(diagnostic.map_direct_callability)
    except AttributeError as exc:
        raise pedigree_models.PedigreeEvidenceError(
            f"{prefix} lacks required ragged-painting diagnostics"
        ) from exc

    if (
        positions.ndim != 1
        or site_indices.shape != positions.shape
        or np.any(~np.isfinite(positions))
        or np.any(positions != np.floor(positions))
        or np.any(~np.isfinite(site_indices))
        or np.any(site_indices != np.floor(site_indices))
        or np.any(site_indices < 0)
    ):
        raise pedigree_models.PedigreeEvidenceError(
            f"{prefix} diagnostic sites must be aligned integer vectors"
        )
    positions = np.ascontiguousarray(positions, dtype=np.int64)
    if len(positions) > 1 and np.any(positions[1:] <= positions[:-1]):
        raise pedigree_models.PedigreeEvidenceError(
            f"{prefix} diagnostic positions must be strictly increasing"
        )
    if len(positions):
        interval = tuple(component.interval)
        if positions[0] < interval[0] or positions[-1] > interval[1]:
            raise pedigree_models.PedigreeEvidenceError(
                f"{prefix} diagnostic positions lie outside its T09 interval"
            )

    n_bins = len(centers)
    if (
        edges.shape != (n_bins + 1,)
        or np.any(~np.isfinite(centers))
        or np.any(~np.isfinite(edges))
        or np.any(edges != np.floor(edges))
        or (n_bins and np.any(edges[1:] <= edges[:-1]))
        or (n_bins and np.any((centers < edges[:-1]) | (centers >= edges[1:])))
    ):
        raise pedigree_models.PedigreeEvidenceError(f"{prefix} has invalid diagnostic bins")
    edges = np.ascontiguousarray(edges, dtype=np.int64)

    if (
        named_alleles.ndim != 2
        or named_alleles.shape[1] != len(positions)
        or (len(positions) and named_alleles.shape[0] < 1)
        or np.any(~np.isin(named_alleles, (-1, 0, 1)))
    ):
        raise pedigree_models.PedigreeEvidenceError(
            f"{prefix} named alleles must have shape (states, sites) in -1/0/1"
        )
    named_alleles = np.ascontiguousarray(named_alleles, dtype=np.int8)
    if len(equivalence_classes) != named_alleles.shape[0]:
        raise pedigree_models.PedigreeEvidenceError(
            f"{prefix} equivalence classes must match named-allele rows"
        )
    original_ids = set()
    for members in equivalence_classes:
        if not isinstance(members, tuple) or not members:
            raise pedigree_models.PedigreeEvidenceError(
                f"{prefix} has an invalid equivalence class"
            )
        for original in members:
            if (
                original in original_ids
                or int(original) != original
                or original < 0
            ):
                raise pedigree_models.PedigreeEvidenceError(
                    f"{prefix} has invalid original founder IDs"
                )
            original_ids.add(int(original))
    if label_grid.shape != (n_samples, 2, n_bins):
        raise pedigree_models.PedigreeEvidenceError(
            f"{prefix} MAP labels must have shape (samples, 2, bins)"
        )
    if (
        np.any(~np.isfinite(label_grid))
        or np.any(label_grid != np.floor(label_grid))
        or np.any(label_grid < -1)
    ):
        raise pedigree_models.PedigreeEvidenceError(
            f"{prefix} MAP labels contain a nonlocal founder ID"
        )
    known_originals = np.asarray(tuple(sorted(original_ids)), dtype=np.int64)
    if np.any((label_grid >= 0) & ~np.isin(label_grid, known_originals)):
        raise pedigree_models.PedigreeEvidenceError(
            f"{prefix} MAP labels contain an unknown founder ID"
        )
    if (
        state_grid.shape != label_grid.shape
        or np.any(~np.isfinite(state_grid))
        or np.any(state_grid != np.floor(state_grid))
        or np.any(state_grid < -1)
        or np.any(state_grid > named_alleles.shape[0])
    ):
        raise pedigree_models.PedigreeEvidenceError(
            f"{prefix} MAP state classes contain an invalid class ID"
        )
    if direct.dtype != np.dtype(np.bool_) or direct.shape != label_grid.shape:
        raise pedigree_models.PedigreeEvidenceError(
            f"{prefix} direct callability must be a boolean MAP-grid mask"
        )
    if statuses.dtype != np.dtype(np.uint8) or statuses.shape != label_grid.shape:
        raise pedigree_models.PedigreeEvidenceError(
            f"{prefix} track status must be a uint8 MAP-grid annotation"
        )
    unknown = (
        (statuses == int(module_painting_model.PaintingTrackStatus.BACKGROUND))
        | (statuses == int(module_painting_model.PaintingTrackStatus.UNANCHORED_TRAJECTORY))
    )
    if np.any(direct & ((state_grid < 0) | unknown)):
        raise pedigree_models.PedigreeEvidenceError(
            f"{prefix} unknown ancestry cannot be directly founder-callable"
        )
    source_grid = np.ascontiguousarray(state_grid, dtype=np.int16).copy()
    source_grid[unknown] = -1
    return (
        positions,
        centers,
        edges,
        named_alleles,
        source_grid,
        np.ascontiguousarray(direct, dtype=np.bool_),
    )


def _ragged_state_and_binning(
        diagnostic: Any,
        positions: np.ndarray,
        centers: np.ndarray,
        edges: np.ndarray,
        named_alleles: np.ndarray,
) -> tuple[module_painting_model.RaggedStateSpace, module_painting_model.RaggedBinning]:
    """Reconstruct the exact frozen T09 state/bin definitions."""

    called = named_alleles >= 0
    called_count = np.sum(called, axis=0)
    if np.any(called_count == 0):
        raise pedigree_models.PedigreeEvidenceError(
            "T09 diagnostic retained a site with no named founder call"
        )
    frequency = (
        1.0 + np.sum(np.where(called, named_alleles, 0), axis=0)
    ) / (2.0 + called_count)
    q = named_alleles.astype(np.float64)
    q[~called] = np.broadcast_to(frequency, q.shape)[~called]
    state = module_painting_model.RaggedStateSpace(
        positions=np.ascontiguousarray(positions),
        site_indices=np.ascontiguousarray(
            np.asarray(diagnostic.selected_site_indices, dtype=np.int64)
        ),
        q=np.ascontiguousarray(q),
        called=np.ascontiguousarray(called),
        active=np.ones(called.shape, dtype=np.bool_),
        equivalence_classes=tuple(diagnostic.equivalence_classes),
        background_index=int(named_alleles.shape[0]),
        background_alt_probability=np.ascontiguousarray(frequency),
    )
    bins = []
    for block in range(len(centers)):
        first = int(np.searchsorted(positions, edges[block]))
        last = int(np.searchsorted(positions, edges[block + 1]))
        if first == last:
            raise pedigree_models.PedigreeEvidenceError("T09 diagnostic contains an empty HMM bin")
        bins.append(np.arange(first, last, dtype=np.int64))
    if (
        not bins
        or not np.array_equal(
            np.concatenate(bins), np.arange(len(positions), dtype=np.int64)
        )
    ):
        raise pedigree_models.PedigreeEvidenceError(
            "T09 diagnostic bins do not partition selected sites"
        )
    return state, module_painting_model.RaggedBinning(
        indices=tuple(bins),
        centers=np.ascontiguousarray(centers, dtype=np.float64),
        edges=np.ascontiguousarray(edges, dtype=np.int64),
        sizes=np.asarray([len(value) for value in bins], dtype=np.int64),
    )


def _raw_indices_for_positions(
        raw_positions: np.ndarray,
        selected_positions: np.ndarray,
        component_index: int,
) -> np.ndarray:
    indices = np.searchsorted(raw_positions, selected_positions)
    matched = indices < len(raw_positions)
    if np.any(matched):
        matched[matched] &= (
            raw_positions[indices[matched]] == selected_positions[matched]
        )
    if not np.all(matched):
        missing = selected_positions[~matched]
        raise pedigree_models.PedigreeEvidenceError(
            f"component {component_index} raw positions lack selected coordinate "
            f"{int(missing[0])}"
        )
    return np.ascontiguousarray(indices, dtype=np.int64)


def _compact_ragged_model(
        founder_grid: np.ndarray,
        marker_counts: np.ndarray,
) -> pedigree_sources.RaggedFounderModel:
    parts = [
        founder_grid[:, block,:int(count)]
        for block, count in enumerate(marker_counts)
    ]
    alleles = np.ascontiguousarray(np.concatenate(parts, axis=1))
    sizes = np.asarray(marker_counts, dtype=np.int64)
    stops = np.cumsum(sizes)
    starts = stops - sizes
    bin_sites = tuple(
        np.arange(start, stop, dtype=np.int64)
        for start, stop in zip(starts, stops)
    )
    return pedigree_sources.build_ragged_founder_model(alleles, bin_sites)


def _selected_marker_grid(
        diagnostic_positions: np.ndarray,
        bin_edges: np.ndarray,
        max_snps_per_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_bins = len(bin_edges) - 1
    selected = np.full((n_bins, max_snps_per_bin), -1, dtype=np.int64)
    marker_counts = np.zeros(n_bins, dtype=np.int64)
    for block in range(n_bins):
        first = int(np.searchsorted(diagnostic_positions, bin_edges[block]))
        last = int(np.searchsorted(diagnostic_positions, bin_edges[block + 1]))
        count = last - first
        if count < 1:
            continue
        if count <= max_snps_per_bin:
            chosen = np.arange(first, last, dtype=np.int64)
        else:
            step = count / float(max_snps_per_bin)
            chosen = np.asarray(
                [first + int(slot * step) for slot in range(max_snps_per_bin)],
                dtype=np.int64,
            )
        marker_counts[block] = len(chosen)
        selected[block,:len(chosen)] = chosen
    return selected, marker_counts


def _prepare_component(
        diagnostic: Any,
        component_index: int,
        component: Any,
        raw_gl: np.ndarray,
        raw_positions: np.ndarray,
        raw_observed: np.ndarray,
        recombination_rate: float,
        max_snps_per_bin: int,
        source_mode: str,
        painting_config: dict[str, Any] | None,
        chromosome_map=None,
) -> PreparedComponentPedigree | None:
    (
        diagnostic_positions,
        centers,
        edges,
        named_alleles,
        label_grid,
        direct,
    ) = _diagnostic_arrays(
        diagnostic, component_index, component, raw_gl.shape[0]
    )
    if len(diagnostic_positions) == 0 or len(centers) == 0:
        return None

    ragged_factors = None
    ragged_anchored_states = None
    if painting_config is None:
        raise pedigree_models.PedigreeEvidenceError("posterior source mode lacks T09 config")
    state, binning = _ragged_state_and_binning(
        diagnostic, diagnostic_positions, centers, edges, named_alleles
    )
    # Structure identity follows whole-component T09 trajectory classes.
    # Derive anchoring before Stage-10 marker subsampling: a trajectory
    # called only at an omitted marker remains a represented identity.
    ragged_anchored_states = np.ascontiguousarray(np.concatenate((
        np.any(named_alleles >= 0, axis=1),
        np.asarray((False,), dtype=np.bool_),  # BACKGROUND
    )), dtype=np.bool_)
    all_raw_indices = _raw_indices_for_positions(
        raw_positions, diagnostic_positions, component_index
    )
    # Raw evidence identity/sample order and T09 config were checked by the
    # chromosome entry point. Reuse the exact frozen emission axis when kept
    # by the painter; high-K/memory-limited products compute it here as usual.
    source_emissions = getattr(diagnostic, "source_log_emission_upper", None)
    if source_emissions is None:
        source_gl, source_observed, informative_counts = painting_evidence.gather_component_evidence(
            raw_gl, raw_observed, all_raw_indices, raw_gl.dtype.type(1e-12))
        source_emissions, _ = module_painting_model.calculate_ragged_binned_emissions(
            source_gl, source_observed, state, binning,
            robustness_epsilon=painting_config["robustness_epsilon"], log_floor=_T09_LOG_FLOOR)
        del source_gl, source_observed
    else:
        states = state.background_index + 1
        if source_emissions.shape != (raw_gl.shape[0], states * (states + 1) // 2, len(centers)):
            raise pedigree_models.PedigreeEvidenceError("cached T09 emissions have incompatible axes")
        source_emissions = painting_evidence.expand_symmetric_emissions(source_emissions, states)
        informative_counts = painting_evidence.count_component_information(
            raw_gl, raw_observed, all_raw_indices, raw_gl.dtype.type(1e-12))
    transition = pedigree_sources.build_t09_hamming_transition(
        centers,
        state.background_index + 1,
        recomb_rate=painting_config["recombination_rate"],
        switch_penalty_per_snp=painting_config["switch_penalty_per_snp"],
        snps_per_bin=painting_config["snps_per_bin"],
        double_recomb_factor=painting_config["double_recomb_factor"],
        chromosome_map=painting_config.get("chromosome_map"),
    )
    ragged_factors = pedigree_sources.infer_candidate_source_factors_batch(
        source_emissions,
        transition,
        informative_counts,
        robustness_epsilon=painting_config["robustness_epsilon"],
    )
    del source_emissions

    selected, marker_counts = _selected_marker_grid(
        diagnostic_positions, edges, max_snps_per_bin
    )
    if np.sum(marker_counts) < 1:
        return None
    selected_mask = selected >= 0
    diagnostic_indices = selected[selected_mask]
    selected_positions = np.full_like(selected, -1)
    selected_positions[selected_mask] = diagnostic_positions[diagnostic_indices]
    raw_indices = _raw_indices_for_positions(
        raw_positions, selected_positions[selected_mask], component_index
    )

    compact_gl = np.full(
        (raw_gl.shape[0], len(centers), max_snps_per_bin, 3),
        1.0 / 3.0,
        dtype=np.float64,
    )
    blocks, slots = np.nonzero(selected_mask)
    compact_gl[:, blocks, slots,:] = raw_gl[:, raw_indices,:]
    compact_observed_grid = np.zeros(
        (raw_gl.shape[0], len(centers), max_snps_per_bin), dtype=np.bool_
    )
    compact_observed_grid[:, blocks, slots] = raw_observed[:, raw_indices]
    nonuniform = (
        (compact_gl[..., 0] != compact_gl[..., 1])
        | (compact_gl[..., 1] != compact_gl[..., 2])
    )
    real_slots = (
        np.arange(max_snps_per_bin)[None,:] < marker_counts[:, None]
    )
    if not np.any(nonuniform & real_slots[None,:,:]):
        return None

    founder_grid = np.full(
        (named_alleles.shape[0], len(centers), max_snps_per_bin),
        -1,
        dtype=np.int8,
    )
    founder_grid[:, blocks, slots] = named_alleles[:, diagnostic_indices]
    ragged_model = (
        _compact_ragged_model(founder_grid, marker_counts)
    )
    labels = np.ascontiguousarray(np.transpose(label_grid, (0, 2, 1))).copy()
    direct_bin_track = np.transpose(direct, (0, 2, 1)).copy()
    informative_by_sample_bin = np.any(
        nonuniform & real_slots[None,:,:], axis=2
    )
    direct_bin_track &= informative_by_sample_bin[:,:, None]
    direct_labels = labels.copy()
    direct_labels[~direct_bin_track] = -1

    safe_labels = np.maximum(labels, 0)
    bin_axis = np.arange(len(centers))[None,:, None]
    stacked = np.empty(
        (raw_gl.shape[0], len(centers), 2, max_snps_per_bin),
        dtype=np.int8,
    )
    for track in range(2):
        stacked[:,:, track,:] = founder_grid[
            safe_labels[:,:, track], bin_axis[:,:, 0],:
        ]
        stacked[:,:, track,:][labels[:,:, track] < 0] = -1
    jointly_valid = (stacked[:,:, 0] >= 0) & (stacked[:,:, 1] >= 0)
    hom = (
        ~np.any(jointly_valid, axis=2)
        | np.all(
            ~jointly_valid | (stacked[:,:, 0] == stacked[:,:, 1]), axis=2
        )
    )
    theta, switch_costs, stay_costs = core_genetic_map.poisson_switch_stay_terms(
        centers, recombination_rate, chromosome_map=chromosome_map,
    )
    cache = pedigree_models.ComponentEvidenceArrays(
        contig=f"component_{component_index}",
        stacked_alleles=np.ascontiguousarray(stacked),
        stacked_hom_mask=np.ascontiguousarray(hom, dtype=np.bool_),
        switch_costs=switch_costs,
        stay_costs=stay_costs,
        informative_markers=int(np.sum(marker_counts)),
        stacked_labels=labels,
        founder_alleles=np.ascontiguousarray(founder_grid),
        selected_markers_per_bin=np.ascontiguousarray(marker_counts),
        switch_probabilities=np.ascontiguousarray(theta),
        genotype_likelihoods=np.ascontiguousarray(compact_gl),
        selected_positions=np.ascontiguousarray(selected_positions),
        state_evidence_mode=(
            f"raw_gl_{source_mode}"
        ),
    )
    compact_genotype_likelihoods = np.ascontiguousarray(
        compact_gl[:, blocks, slots,:]
    )
    compact_observed = np.ascontiguousarray(
        compact_observed_grid[:, blocks, slots]
    )
    return PreparedComponentPedigree(
        component_index=component_index,
        cache=cache,
        direct_labels=direct_labels,
        source_mode=source_mode,
        ragged_model=ragged_model,
        ragged_source_factors=ragged_factors,
        compact_genotype_likelihoods=compact_genotype_likelihoods,
        compact_observed=compact_observed,
        ragged_anchored_states=ragged_anchored_states,
    )


def prepare_t09_chromosome_components(
        checkpoint: Any,
        raw_genotype_likelihoods: Any,
        raw_positions: Any,
        raw_observed_mask: Any,
        raw_sample_ids: Sequence[Any],
        *,
        contig: str,
        recombination_rate: float=1e-8,
        max_snps_per_bin: int=10,
        markers_per_information_block: int=100,
        effective_markers_per_information_block: float=1.0,
        candidate_source_mode: str=pedigree_models.RAGGED_QUADRATIC_MODEL,
        chromosome_map=None,
) -> PreparedChromosome:
    """Validate and prepare independently rooted component HMM inputs."""

    checkpoint = painting_checkpoints.validate_t09_component_checkpoint(checkpoint)
    source_identity = _t09_source_identity(checkpoint)
    if chromosome_map is not None:
        source_identity = dict(source_identity, selector_genetic_map=chromosome_map.identity())
        recombination_rate = chromosome_map.fallback_rate_per_bp
    name = str(contig)
    if not name:
        raise pedigree_models.PedigreeEvidenceError("contig must be nonempty")
    source_mode = str(candidate_source_mode)

    painting_config = (
        _t09_painting_config(checkpoint)
    )
    max_snps = _positive_integer(max_snps_per_bin, "max_snps_per_bin")
    information_block_size = _positive_integer(
        markers_per_information_block, "markers_per_information_block"
    )
    if not np.isfinite(recombination_rate) or recombination_rate < 0.0:
        raise pedigree_models.PedigreeEvidenceError("recombination_rate must be finite and nonnegative")
    if (
        not np.isfinite(effective_markers_per_information_block)
        or effective_markers_per_information_block <= 0.0
    ):
        raise pedigree_models.PedigreeEvidenceError(
            "effective_markers_per_information_block must be finite and positive"
        )
    blocks = core_runtime.validate_phase_component_manifest(
        checkpoint.component_manifest
    )
    components = checkpoint.painting_bundle.components
    diagnostic_items = tuple(
        component.ragged_diagnostics for component in components
    )
    raw_gl, positions, raw_observed = _validated_raw_evidence(
        checkpoint,
        raw_genotype_likelihoods,
        raw_positions,
        raw_observed_mask,
        raw_sample_ids,
    )

    prepared = []
    for component_index, (block, component, diagnostic) in enumerate(
            zip(blocks, components, diagnostic_items)):
        model = component.painting_model
        if model != painting_components.PAINTING_MODEL_RAGGED:
            raise pedigree_models.PedigreeEvidenceError(
                f"component {component_index} is not a unified open-set painting"
            )
        if diagnostic is None:
            if component.informative_site_count == 0:
                continue
            raise pedigree_models.PedigreeEvidenceError(
                f"component {component_index} lacks ragged diagnostics"
            )
        value = _prepare_component(
            diagnostic,
            component_index,
            component,
            raw_gl,
            positions,
            raw_observed,
            float(recombination_rate),
            max_snps,
            source_mode,
            painting_config,
            chromosome_map,
        )
        if value is not None:
            prepared.append(value)
    if not prepared:
        return PreparedChromosome(
            contig=name,
            sample_ids=checkpoint.sample_ids,
            components=(),
            information_exponents=(),
            component_count=len(components),
            omitted_reason="no_observed_nonuniform_component_evidence",
            source_mode=source_mode,
            source_identity=source_identity,
        )

    concatenated_gl = np.concatenate(
        [item.cache.genotype_likelihoods for item in prepared], axis=1
    )
    concatenated_markers = np.concatenate(
        [item.cache.selected_markers_per_bin for item in prepared]
    )
    global_exponent = pedigree_candidates._gl_information_exponent_kernel(
        concatenated_gl,
        concatenated_markers,
        information_block_size,
        float(effective_markers_per_information_block),
    )
    exponents = []
    offset = 0
    for item in prepared:
        n_bins = item.cache.stacked_labels.shape[1]
        exponents.append(np.ascontiguousarray(
            global_exponent[:, offset:offset + n_bins]
        ))
        offset += n_bins
    return PreparedChromosome(
        contig=name,
        sample_ids=checkpoint.sample_ids,
        components=tuple(prepared),
        information_exponents=tuple(exponents),
        component_count=len(components),
        source_mode=source_mode,
        source_identity=source_identity,
    )

import haplotype_reconstruction.core.genetic_map as core_genetic_map
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.painting.checkpoints as painting_checkpoints
import haplotype_reconstruction.painting.components as painting_components
import haplotype_reconstruction.painting.model as module_painting_model
import haplotype_reconstruction.painting.evidence as painting_evidence
import haplotype_reconstruction.pedigree.config as module_pedigree_config
