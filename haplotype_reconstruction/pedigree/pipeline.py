"""Checkpointed preparation, inference and export of genome-wide pedigrees."""
from __future__ import annotations
from ..core.run_record import timed_stage
from haplotype_reconstruction import PACKAGE_ROOT

import copy
from dataclasses import asdict, dataclass, fields
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import numpy as np
import haplotype_reconstruction.painting.checkpoints as painting_checkpoints
from . import cache as pedigree_cache
import haplotype_reconstruction.pedigree.components as pedigree_components
import haplotype_reconstruction.pedigree.models as pedigree_models
import haplotype_reconstruction.workflows.design as workflows_design
import haplotype_reconstruction.workflows.reconstruction as workflows_reconstruction

PEDIGREE_PREPARATION_SCHEMA = "pedigree_preparation-component-pedigree-preparation-v5"


PEDIGREE_PREPARATION_BACKEND = "component-local-ragged-source-preparation-v4"


PEDIGREE_GLOBAL_SCHEMA = "pedigree_scoring-component-pedigree-result-v5"


PEDIGREE_GLOBAL_BACKEND = "component-local-parent-state-combined-v4"


EVIDENCE_STAGE, PEDIGREE_STAGE = (
    workflows_design.pedigree_stage_names()
)


PEDIGREE_INFERENCE_CODE_FILES = (
    'pedigree/pipeline.py',
    'pedigree/cache.py',
    'pedigree/explanations.py',
    'pedigree/components.py',
    'pedigree/likelihoods.py',
    'pedigree/bootstrap.py',
    'pedigree/candidates.py',
    'pedigree/config.py',
    'pedigree/direction.py',
    'pedigree/orientation.py',
    'pedigree/ancestry_paths.py',
    'pedigree/eligibility.py',
    'pedigree/evidence.py',
    'pedigree/graph.py',
    'pedigree/inference.py',
    'pedigree/models.py',
    'pedigree/states.py',
    'pedigree/sources.py',
    'pedigree/transmission.py',
    'pedigree/transmission_projection.py',
    'pedigree/transmission_scoring.py',
    'painting/model.py',
    'painting/evidence.py',
    'core/genetic_map.py',
    'core/raw_evidence.py',
    'pedigree/results.py'
)


@dataclass(frozen=True)
class EvidenceConfig:
    """Scientific preparation and fixed candidate-panel settings."""

    recombination_rate: float = 5e-8
    max_snps_per_bin: int = 10
    top_k: int = 20
    anchor_k: int = 5
    use_anchor_union: bool = False
    mismatch_penalty: float = pedigree_models.DEFAULT_MISMATCH_PENALTY
    genetic_maps: Any = None

    def __post_init__(self) -> None:
        if self.genetic_maps is not None:
            object.__setattr__(self, "recombination_rate",
                               self.genetic_maps.default_rate_cm_per_mb / 1e8)
        if not math.isfinite(self.recombination_rate) or self.recombination_rate < 0:
            raise ValueError("recombination_rate must be finite and nonnegative")
        for name in ("max_snps_per_bin", "top_k", "anchor_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.use_anchor_union, bool):
            raise TypeError("use_anchor_union must be boolean")
        if not math.isfinite(self.mismatch_penalty) or self.mismatch_penalty >= 0:
            raise ValueError("mismatch_penalty must be finite and negative")
        if self.top_k != 20:
            raise ValueError("production top_k is fixed at 20")
        if self.anchor_k != 5:
            raise ValueError("production anchor_k is fixed at 5")
        if self.mismatch_penalty != pedigree_models.DEFAULT_MISMATCH_PENALTY:
            raise ValueError(
                "production mismatch_penalty must use the engine default"
            )


@dataclass(frozen=True)
class PedigreeEvidenceCheckpoint:
    """Compact pedigree preparation product bound to its exact typed painting identities."""

    schema: str
    contig: str
    sample_ids: tuple[str, ...]
    prepared_chromosome: pedigree_components.PreparedChromosome
    assembly_identity: painting_checkpoints.ScientificIdentity
    tpainting_identity: painting_checkpoints.ScientificIdentity
    preparation_identity: painting_checkpoints.ScientificIdentity


@dataclass(frozen=True)
class PedigreeContigSummary:
    """Small operational summary for one prepared physical chromosome."""

    contig: str
    resumed: bool
    component_count: int
    scored_component_count: int
    omitted_reason: str | None


def _canonical_sample_ids(sample_ids: Sequence[Any]) -> tuple[str, ...]:
    result = tuple(str(value) for value in sample_ids)
    if not result or len(result) != len(set(result)):
        raise ValueError("ordered sample IDs must be nonempty and unique")
    return result


def _canonical_contigs(contigs: Sequence[Any], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in contigs)
    if not result or any(not value for value in result):
        raise ValueError(f"{name} must be nonempty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique physical contigs")
    return result


def _canonical_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty mapping")
    try:
        return json.loads(json.dumps(
            copy.deepcopy(dict(value)),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be JSON-compatible") from error


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_mapping(value, "identity"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pedigree_inference_code_identity(
        digest_overrides: Mapping[str, str] | None=None) -> dict[str, str]:
    """Hash the complete scientific pedigree inference implementation closure."""

    overrides = {} if digest_overrides is None else dict(digest_overrides)
    unknown = set(overrides).difference(PEDIGREE_INFERENCE_CODE_FILES)
    if unknown:
        raise ValueError(
            "unknown pedigree code identity override files: "
            + ", ".join(sorted(unknown))
        )
    root = PACKAGE_ROOT
    result = {}
    for filename in PEDIGREE_INFERENCE_CODE_FILES:
        if filename in overrides:
            digest = str(overrides[filename]).lower()
            if len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"invalid SHA256 override for {filename}")
        else:
            digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        result[filename] = digest
    return result


def _eligibility_provenance(parent_eligibility, sample_ids):
    if parent_eligibility is not None:
        record_ids = tuple(str(value) for value in parent_eligibility["sample_ids"])
        if record_ids != sample_ids:
            raise ValueError("parent eligibility sample order mismatch")
        return (
            workflows_design.parent_eligibility_identity(parent_eligibility),
            workflows_design.summarize_parent_eligibility(parent_eligibility),
        )

    sample_count = len(sample_ids)
    unrestricted = {
        "policy_name": "unrestricted_no_metadata_v1",
        "ordered_sample_ids": sample_ids,
        "individual_parentage_ground_truth": False,
    }
    candidates = max(0, sample_count - 1)
    summary = {
        "policy_name": unrestricted["policy_name"],
        "sample_count": sample_count,
        "eligible_child_count": sample_count,
        "candidate_parent_sample_count": sample_count,
        "minimum_parent_candidates_per_target": candidates,
        "maximum_parent_candidates_per_target": candidates,
        "minimum_parent_pairs_per_target": candidates * (candidates - 1) // 2,
        "maximum_parent_pairs_per_target": candidates * (candidates - 1) // 2,
        "individual_parentage_ground_truth": False,
        "assumptions": [
            "No metadata eligibility or simulated parentage truth enters inference."
        ],
    }
    return _canonical_digest(unrestricted), summary


def pedigree_preparation_identity(
        preparation_config,
        pedigree_config,
        *,
        sample_ids,
        contigs,
        painting_stage,
        raw_gl_stage,
        raw_sites_stage,
        raw_gl_key,
        raw_sites_key,
        raw_observed_mask_key,
        parent_eligibility_identity,
        code_digest_overrides=None,
        source_files=None,
) -> dict[str, Any]:
    """Return the common pedigree preparation identity, excluding runtime scheduling."""

    if not isinstance(preparation_config, EvidenceConfig):
        raise TypeError("preparation_config must be EvidenceConfig")
    if not isinstance(pedigree_config, module_pedigree_config.PedigreeConfig):
        raise TypeError("pedigree_config must be PedigreeConfig")
    settings = pedigree_config.validated()
    return _canonical_mapping({
        "schema": PEDIGREE_PREPARATION_SCHEMA,
        "backend": PEDIGREE_PREPARATION_BACKEND,
        "config": {
            "preparation": {
                **{f.name: getattr(preparation_config, f.name)
                   for f in fields(preparation_config) if f.name != "genetic_maps"},
                **({"genetic_maps": preparation_config.genetic_maps.identity()}
                   if preparation_config.genetic_maps is not None else {}),
            },
            "pedigree": {name: getattr(settings, name) for name in (
                "markers_per_information_block",
                "parent_state_effective_markers_per_information_block",
                "parent_state_candidate_source_mode")},
        },
        "ordered_sample_ids": _canonical_sample_ids(sample_ids),
        "ordered_contigs": _canonical_contigs(contigs, "contigs"),
        "source": {
            "painting_stage": str(painting_stage),
            "painting_schema": painting_checkpoints.PAINTING_COMPONENT_CHECKPOINT_SCHEMA,
            "raw_gl_stage": str(raw_gl_stage),
            "raw_sites_stage": str(raw_sites_stage),
            "raw_gl_key": str(raw_gl_key),
            "raw_sites_key": str(raw_sites_key),
            "raw_observed_mask_key": str(raw_observed_mask_key),
            "genotype_evidence_mode": workflows_reconstruction.SUPPORTED_GENOTYPE_EVIDENCE_MODE,
            "observed_mask_mode": workflows_reconstruction.EXACT_OBSERVED_MASK_MODE,
        },
        "source_files": source_files,
        "preparation_code_identity": pedigree_cache.preparation_code_identity(),
    }, "pedigree preparation identity")


def pedigree_global_identity(preparation_identity, settings, eligibility_identity, scoring_identity) -> dict[str, Any]:
    """Decision identity, independent of the reusable genetic-score product."""

    prepared = _canonical_mapping(preparation_identity, "preparation_identity")
    return _canonical_mapping({
        "schema": PEDIGREE_GLOBAL_SCHEMA,
        "backend": PEDIGREE_GLOBAL_BACKEND,
        "config": {"preparation_identity_sha256": _canonical_digest(prepared)},
        "preparation_identity": prepared,
        "scoring_identity": scoring_identity,
        "pedigree_config": asdict(settings),
        "parent_eligibility_identity": eligibility_identity,
        "decision_code_identity": pedigree_inference_code_identity(),
    }, "pedigree global identity")


def _validate_source_modes(payload, stage, contig) -> None:
    if payload.get("genotype_evidence_mode") != workflows_reconstruction.SUPPORTED_GENOTYPE_EVIDENCE_MODE:
        raise RuntimeError(
            f"{stage}/{contig} lacks normalized raw-likelihood provenance"
        )
    if payload.get("observed_call_mask_mode") != workflows_reconstruction.EXACT_OBSERVED_MASK_MODE:
        raise RuntimeError(
            f"{stage}/{contig} lacks exact observation-mask provenance"
        )


def _load_raw_evidence(
        checkpoint_store,
        contig,
        *,
        raw_gl_stage,
        raw_sites_stage,
        raw_gl_key,
        raw_sites_key,
        raw_observed_mask_key,
):
    cached = raw_evidence.load(
        checkpoint_store, contig, raw_gl_stage=raw_gl_stage,
        raw_sites_stage=raw_sites_stage, raw_gl_key=raw_gl_key,
        raw_sites_key=raw_sites_key, raw_observed_mask_key=raw_observed_mask_key)
    if cached is not None:
        _validate_source_modes(cached, raw_evidence.STAGE, contig)
        return (cached["global_probs"], cached["global_sites"],
                cached["global_observed_mask"], cached, cached)
    sites_payload = checkpoint_store.load_contig(raw_sites_stage, contig)
    gl_payload = None
    try:
        _validate_source_modes(sites_payload, raw_sites_stage, contig)
        if raw_gl_stage == raw_sites_stage:
            gl_payload = sites_payload
        else:
            gl_payload = checkpoint_store.load_contig(raw_gl_stage, contig)
        for payload, key, stage in (
            (gl_payload, raw_gl_key, raw_gl_stage),
            (sites_payload, raw_sites_key, raw_sites_stage),
            (sites_payload, raw_observed_mask_key, raw_sites_stage),
        ):
            if key not in payload:
                raise KeyError(f"{stage}/{contig} lacks {key!r}")
        return (
            gl_payload[raw_gl_key],
            sites_payload[raw_sites_key],
            sites_payload[raw_observed_mask_key],
            gl_payload,
            sites_payload,
        )
    except Exception:
        del gl_payload, sites_payload
        raise


def _validate_ragged_component(
        component: pedigree_components.PreparedComponentPedigree,
        sample_count: int,
        bin_count: int,
) -> None:
    """Validate the compact painting posterior law persisted in one pedigree preparation component."""

    model = component.ragged_model
    factors = component.ragged_source_factors
    observed = np.asarray(component.compact_observed)
    if not isinstance(model, pedigree_sources.RaggedFounderModel):
        raise ValueError("pedigree preparation ragged component lacks its founder model")
    if not isinstance(factors, pedigree_sources.RaggedSourceBatchFactors):
        raise ValueError("pedigree preparation ragged component lacks source posterior factors")
    if observed.dtype != np.dtype(np.bool_) or observed.ndim != 2:
        raise ValueError("pedigree preparation ragged observed mask is invalid")

    named = np.asarray(model.named_alleles)
    called = np.asarray(model.called)
    allele_probability = np.asarray(model.allele_probability)
    background = np.asarray(model.background_alt_probability)
    site_to_bin = np.asarray(model.site_to_bin)
    anchored_states = np.asarray(component.ragged_anchored_states)
    named_count = int(model.background_index)
    state_count = named_count + 1
    if (
        named.ndim != 2
        or named_count != named.shape[0]
        or named_count < 1
        or named.shape[1] < 1
        or np.any(~np.isin(named, (-1, 0, 1)))
        or called.dtype != np.dtype(np.bool_)
        or called.shape != named.shape
        or not np.array_equal(called, named >= 0)
        or allele_probability.shape != named.shape
        or np.any(~np.isfinite(allele_probability))
        or np.any((allele_probability < 0.0) | (allele_probability > 1.0))
        or background.shape != (named.shape[1],)
        or np.any(~np.isfinite(background))
        or np.any((background < 0.0) | (background > 1.0))
        or site_to_bin.shape != (named.shape[1],)
        or anchored_states.dtype != np.dtype(np.bool_)
        or anchored_states.shape != (state_count,)
        or bool(anchored_states[-1])
        or np.any(np.any(called, axis=1) & ~anchored_states[:-1])
        or not np.any(anchored_states[:-1])
        or len(model.bin_site_indices) != bin_count
    ):
        raise ValueError("pedigree preparation ragged founder model shape/content is invalid")
    expected_frequency = (
        1.0 + np.sum(np.where(called, named, 0), axis=0)
    ) / (2.0 + np.sum(called, axis=0))
    expected_probability = np.where(called, named, expected_frequency[None,:])
    if (
        not np.array_equal(background, expected_frequency)
        or not np.array_equal(allele_probability, expected_probability)
    ):
        raise ValueError("pedigree preparation ragged founder allele probabilities changed")

    cache = component.cache
    founder_grid = np.asarray(cache.founder_alleles)
    marker_counts = np.asarray(cache.selected_markers_per_bin)
    if (
        founder_grid.ndim != 3
        or founder_grid.shape[0] != named_count
        or founder_grid.shape[1] != bin_count
        or marker_counts.shape != (bin_count,)
        or np.any(~np.isfinite(marker_counts))
        or np.any(marker_counts != np.floor(marker_counts))
        or np.any(marker_counts < 1)
        or np.any(marker_counts > founder_grid.shape[2])
    ):
        raise ValueError("pedigree preparation ragged marker/model alignment is invalid")
    compact_named = np.ascontiguousarray(np.concatenate([
        founder_grid[:, block,:int(count)]
        for block, count in enumerate(marker_counts)
    ], axis=1))
    if not np.array_equal(named, compact_named):
        raise ValueError("pedigree preparation ragged founder model disagrees with compact markers")
    expected_bins = []
    offset = 0
    for count in marker_counts:
        stop = offset + int(count)
        expected_bins.append(np.arange(offset, stop, dtype=np.int64))
        offset = stop
    if any(
        np.asarray(actual).ndim != 1
        or not np.array_equal(np.asarray(actual), expected)
        for actual, expected in zip(model.bin_site_indices, expected_bins)
    ):
        raise ValueError("pedigree preparation ragged bin/site partition is invalid")
    expected_site_to_bin = np.repeat(
        np.arange(bin_count, dtype=np.int64), marker_counts.astype(np.int64)
    )
    if not np.array_equal(site_to_bin, expected_site_to_bin):
        raise ValueError("pedigree preparation ragged site-to-bin alignment is invalid")
    if observed.shape != (sample_count, named.shape[1]):
        raise ValueError("pedigree preparation ragged observed-mask shape is invalid")

    initial = np.asarray(factors.initial_probability)
    right = np.asarray(factors.right_weight)
    available = np.asarray(factors.available)
    informative = np.asarray(factors.informative_site_count)
    transition = factors.transition
    if (
        initial.shape != (sample_count, state_count, state_count)
        or right.shape
        != (sample_count, max(0, bin_count - 1), state_count, state_count)
        or available.dtype != np.dtype(np.bool_)
        or available.shape != (sample_count,)
        or informative.shape != (sample_count,)
        or np.any(~np.isfinite(informative))
        or np.any(informative != np.floor(informative))
        or np.any(informative < 0)
        or not np.array_equal(available, informative >= 1)
        or np.any(~np.isfinite(initial))
        or np.any(initial < 0.0)
        or np.any(~np.isfinite(right))
        or np.any(right < 0.0)
        or not isinstance(transition, pedigree_sources.HammingTransition)
        or int(transition.n_states) != state_count
    ):
        raise ValueError("pedigree preparation ragged source-factor shape/content is invalid")
    initial_mass = np.sum(initial, axis=(1, 2))
    if np.any(
        ~np.isclose(initial_mass, 1.0, rtol=1e-12, atol=1e-14)
    ):
        raise ValueError("pedigree preparation ragged source initial probabilities are invalid")
    same = np.asarray(transition.same)
    one = np.asarray(transition.one_change)
    two = np.asarray(transition.two_changes)
    boundaries = max(0, bin_count - 1)
    if (
        any(value.shape != (boundaries,) for value in (same, one, two))
        or any(np.any(~np.isfinite(value)) or np.any(value < 0.0)
               for value in (same, one, two))
        or not math.isfinite(float(transition.double_recomb_factor))
        or float(transition.double_recomb_factor) <= 0.0
    ):
        raise ValueError("pedigree preparation ragged transition is invalid")
    row_mass = same + 2.0 * (state_count - 1) * one + (
        state_count - 1
    ) ** 2 * two
    if not np.allclose(row_mass, 1.0, rtol=2e-13, atol=2e-15):
        raise ValueError("pedigree preparation ragged transition rows are not normalized")
    if (
        not math.isfinite(float(factors.robustness_epsilon))
        or not 0.0 <= float(factors.robustness_epsilon) <= 1.0
        or not math.isfinite(float(factors.preparation_seconds))
        or float(factors.preparation_seconds) < 0.0
        or hasattr(factors, "denominator")
    ):
        raise ValueError("pedigree preparation ragged factor metadata is invalid")


def _validate_prepared_checkpoint(
        payload,
        *,
        expected_contig,
        expected_sample_ids,
        expected_preparation_identity,
        expected_painting=None,
) -> PedigreeEvidenceCheckpoint:
    if not isinstance(payload, PedigreeEvidenceCheckpoint):
        raise TypeError("pedigree preparation requires a typed prepared checkpoint")
    if payload.schema != PEDIGREE_PREPARATION_SCHEMA:
        raise ValueError("unknown pedigree preparation checkpoint schema")
    if payload.contig != str(expected_contig):
        raise ValueError("pedigree preparation physical contig mismatch")
    sample_ids = _canonical_sample_ids(expected_sample_ids)
    if payload.sample_ids != sample_ids:
        raise ValueError("pedigree preparation sample order mismatch")
    expected_identity = painting_checkpoints.ScientificIdentity.from_record(
        expected_preparation_identity
    )
    if payload.preparation_identity != expected_identity:
        raise ValueError("pedigree preparation preparation identity mismatch")
    prepared = payload.prepared_chromosome
    try:
        expected_source_mode = str(expected_identity.record()["config"][
            "pedigree"
        ]["parent_state_candidate_source_mode"])
    except (KeyError, TypeError) as exc:
        raise ValueError("pedigree preparation preparation identity lacks source mode") from exc
    if (
        not isinstance(prepared, pedigree_components.PreparedChromosome)
        or prepared.contig != payload.contig
        or prepared.sample_ids != sample_ids
        or isinstance(prepared.component_count, bool)
        or int(prepared.component_count) != prepared.component_count
        or prepared.component_count < len(prepared.components)
        or (bool(prepared.components) == bool(prepared.omitted_reason))
        or len(prepared.information_exponents) != len(prepared.components)
        or prepared.source_mode != expected_source_mode
        or prepared.source_mode not in {
            "hard_painted",
            pedigree_models.PAINTING_RAGGED_POSTERIOR_MODE,
            pedigree_models.RAGGED_QUADRATIC_MODEL,
        }
    ):
        raise ValueError("pedigree preparation prepared chromosome layout is invalid")
    component_ids = []
    for component, exponent in zip(
            prepared.components, prepared.information_exponents):
        if not isinstance(component, pedigree_components.PreparedComponentPedigree):
            raise ValueError("pedigree preparation component is not a typed prepared component")
        labels = np.asarray(component.cache.stacked_labels)
        weights = np.asarray(exponent)
        if (labels.ndim != 3 or labels.shape[0] != len(sample_ids)
                or labels.shape[2] != 2
                or weights.shape != labels.shape[:2]
                or np.any(~np.isfinite(weights))
                or np.any(weights < 0.0)
                or np.any(weights > 1.0)
                or component.source_mode != prepared.source_mode):
            raise ValueError("pedigree preparation component information weights/mode are invalid")
        _validate_ragged_component(
            component, len(sample_ids), labels.shape[1]
        )
        component_ids.append(int(component.component_index))
    if component_ids != sorted(component_ids) or len(set(component_ids)) != len(
            component_ids):
        raise ValueError("pedigree preparation component IDs are not ordered and unique")
    payload.assembly_identity.record()
    payload.tpainting_identity.record()
    if expected_painting is not None:
        painting_checkpoint = painting_checkpoints.validate_painting_checkpoint(
            expected_painting, expected_sample_ids=sample_ids
        )
        if payload.assembly_identity != painting_checkpoint.release_identity:
            raise ValueError("pedigree preparation source painting release identity mismatch")
        if payload.tpainting_identity != painting_checkpoint.painting_product_identity:
            raise ValueError("pedigree preparation source sample painting identity mismatch")
    return payload


def _checkpoint_source_identity(checkpoint: PedigreeEvidenceCheckpoint) -> str:
    return _canonical_digest({
        "contig": checkpoint.contig,
        "preparation_identity": checkpoint.preparation_identity.record(),
        "assembly_identity": checkpoint.assembly_identity.record(),
        "tpainting_identity": checkpoint.tpainting_identity.record(),
    })


def _serialize_ragged_scoring_diagnostics(diagnostics) -> dict[str, Any]:
    """Return stable pedigree preparation/pedigree scoring phase diagnostics for the global summary.

    The component-level timings are additive diagnostic work times, not an
    alternative wall-clock total.  ``runtime.inference_elapsed_seconds``
    remains the authoritative end-to-end pedigree scoring elapsed time.
    """

    return {
        "component_count": int(diagnostics.component_count),
        "phase_timings_seconds": {
            "pedigree_preparation_source_preparation": float(
                diagnostics.source_preparation_seconds
            ),
            "pedigree_scoring_source_marginal_preparation": float(
                diagnostics.source_marginal_preparation_seconds
            ),
            "pedigree_scoring_hard_structure": float(diagnostics.hard_structure_seconds),
            "pedigree_scoring_posterior_expected_structure": float(
                diagnostics.posterior_expected_structure_seconds
            ),
            "pedigree_scoring_transmission_preparation": float(
                diagnostics.transmission_preparation_seconds
            ),
            "pedigree_scoring_m0_scoring": float(diagnostics.m0_scoring_seconds),
            "pedigree_scoring_m1_scoring": float(diagnostics.m1_scoring_seconds),
            "pedigree_scoring_m2_scoring": float(diagnostics.m2_scoring_seconds),
        },
        "m2_row_counts": {
            "active": int(diagnostics.m2_active_trio_count),
            "reduced": int(diagnostics.m2_reduced_trio_count),
        },
        "m2_batch_count": int(diagnostics.m2_batch_count),
        "maximum_m2_batch_size": int(diagnostics.maximum_m2_batch_size),
        "memory_maxima_bytes": {
            "m2_per_active_task": int(
                diagnostics.maximum_m2_bytes_per_active_task
            ),
            "m2_working": int(diagnostics.peak_m2_working_bytes),
            "transmission_probability": int(
                diagnostics.maximum_transmission_probability_bytes
            ),
        },
        "reused_screen_component_count": int(
            diagnostics.reused_screen_component_count
        ),
    }


def _serialize_projected_scoring_diagnostics(diagnostics) -> dict[str, Any]:
    """Return stable diagnostics for the projected-quadratic approximation.

    Projection and bridge diagnostics are kept separate from the additive
    phase work times so a persisted production result states both the
    approximation that was applied and the numerical quality of its
    maximum-entropy bridge construction.
    """

    branch_counts = tuple(
        int(value) for value in diagnostics.bridge_branch_counts
    )
    if len(branch_counts) != 4:
        raise ValueError("projected bridge diagnostics must have four branches")
    return {
        "approximation": {
            "name": str(diagnostics.approximation_name),
            "candidate_source_mode": pedigree_models.RAGGED_QUADRATIC_MODEL,
        },
        "component_count": int(diagnostics.component_count),
        "phase_timings_seconds": {
            "pedigree_preparation_source_preparation": float(
                diagnostics.source_preparation_seconds
            ),
            "pedigree_scoring_source_marginal_preparation": float(
                diagnostics.source_marginal_preparation_seconds
            ),
            "pedigree_scoring_hard_structure": float(diagnostics.hard_structure_seconds),
            "pedigree_scoring_posterior_expected_structure": float(
                diagnostics.posterior_expected_structure_seconds
            ),
            "pedigree_scoring_projection_preparation": float(
                diagnostics.projection_preparation_seconds
            ),
            "pedigree_scoring_m0_scoring": float(diagnostics.m0_scoring_seconds),
            "pedigree_scoring_m1_scoring": float(diagnostics.m1_scoring_seconds),
            "pedigree_scoring_m2_scoring": float(diagnostics.m2_scoring_seconds),
        },
        "m2_row_counts": {
            "active": int(diagnostics.m2_active_trio_count),
            "reduced": int(diagnostics.m2_reduced_trio_count),
        },
        "projection": {
            "maximum_retained_bytes": int(
                diagnostics.maximum_projection_retained_bytes
            ),
            "maximum_projected_hidden_state_count": int(
                diagnostics.maximum_projected_hidden_state_count
            ),
            "maximum_exact_m2_hidden_state_count": int(
                diagnostics.maximum_exact_m2_hidden_state_count
            ),
        },
        "bridge": {
            "maximum_marginal_residual": float(
                diagnostics.maximum_bridge_marginal_residual
            ),
            "branch_counts": {
                "diagonal": branch_counts[0],
                "all_small": branch_counts[1],
                "one_large": branch_counts[2],
                "star": branch_counts[3],
            },
            "maximum_solver_iterations": int(
                diagnostics.maximum_bridge_solver_iterations
            ),
        },
        "reused_screen_component_count": int(
            diagnostics.reused_screen_component_count
        ),
    }


def _prepare_one_contig(
        checkpoint_store,
        contig,
        sample_ids,
        preparation_identity,
        preparation_config,
        pedigree_config,
        *,
        painting_stage,
        raw_gl_stage,
        raw_sites_stage,
        raw_gl_key,
        raw_sites_key,
        raw_observed_mask_key,
        target_stage,
) -> PedigreeContigSummary:
    painting_checkpoint = painting_checkpoints.validate_painting_checkpoint(
        checkpoint_store.load_contig(painting_stage, contig),
        expected_sample_ids=sample_ids,
    )
    if checkpoint_store.contig_done(target_stage, contig):
        checkpoint = _validate_prepared_checkpoint(
            checkpoint_store.load_contig(target_stage, contig),
            expected_contig=contig,
            expected_sample_ids=sample_ids,
            expected_preparation_identity=preparation_identity,
            expected_painting=painting_checkpoint,
        )
        prepared = checkpoint.prepared_chromosome
        return PedigreeContigSummary(
            contig, True, prepared.component_count, len(prepared.components),
            prepared.omitted_reason,
        )

    raw_gl = raw_sites = raw_observed = gl_payload = sites_payload = None
    try:
        raw_gl, raw_sites, raw_observed, gl_payload, sites_payload = (
            _load_raw_evidence(
                checkpoint_store,
                contig,
                raw_gl_stage=raw_gl_stage,
                raw_sites_stage=raw_sites_stage,
                raw_gl_key=raw_gl_key,
                raw_sites_key=raw_sites_key,
                raw_observed_mask_key=raw_observed_mask_key,
            )
        )
        prepared = pedigree_components.prepare_painted_chromosome_components(
            painting_checkpoint,
            raw_gl,
            raw_sites,
            raw_observed,
            sample_ids,
            contig=contig,
            recombination_rate=preparation_config.recombination_rate,
            max_snps_per_bin=preparation_config.max_snps_per_bin,
            chromosome_map=(None if preparation_config.genetic_maps is None
                            else preparation_config.genetic_maps.for_contig(contig)),
            markers_per_information_block=(
                pedigree_config.markers_per_information_block
            ),
            effective_markers_per_information_block=(
                pedigree_config.parent_state_effective_markers_per_information_block
            ),
            candidate_source_mode=(
                pedigree_config.parent_state_candidate_source_mode
            ),
        )
        checkpoint = PedigreeEvidenceCheckpoint(
            schema=PEDIGREE_PREPARATION_SCHEMA,
            contig=contig,
            sample_ids=sample_ids,
            prepared_chromosome=prepared,
            assembly_identity=painting_checkpoint.release_identity,
            tpainting_identity=painting_checkpoint.painting_product_identity,
            preparation_identity=painting_checkpoints.ScientificIdentity.from_record(
                preparation_identity
            ),
        )
        _validate_prepared_checkpoint(
            checkpoint, expected_contig=contig, expected_sample_ids=sample_ids,
            expected_preparation_identity=preparation_identity, expected_painting=painting_checkpoint)
        # Atomic checkpoint I/O already propagates write failures. Validate
        # this exact object before writing; global inference validates the
        # persisted copy on load, without an immediate duplicate disk read.
        checkpoint_store.save_contig(target_stage, contig, checkpoint)
    finally:
        del raw_gl, raw_sites, raw_observed, gl_payload, sites_payload, painting_checkpoint
        gc.collect()
        core_parallel.malloc_trim()

    prepared = checkpoint.prepared_chromosome
    return PedigreeContigSummary(
        contig, False, prepared.component_count, len(prepared.components),
        prepared.omitted_reason,
    )


@timed_stage("pedigree_evidence")
def prepare_pedigree_contigs(
        checkpoint_store,
        contigs,
        sample_ids,
        *,
        pedigree_config,
        parent_eligibility=None,
        preparation_config=EvidenceConfig(),
        painting_stage=workflows_reconstruction.PAINTING_STAGE,
        raw_gl_stage,
        raw_sites_stage,
        raw_gl_key="global_probs",
        raw_sites_key="global_sites",
        raw_observed_mask_key="global_observed_mask",
        all_contigs=None,
        publish_completion=True,
        target_stage=EVIDENCE_STAGE,
) -> tuple[PedigreeContigSummary, ...]:
    """Prepare requested pedigree preparation contigs; shards never publish full completion."""

    if not isinstance(preparation_config, EvidenceConfig):
        raise TypeError("preparation_config must be EvidenceConfig")
    settings = pedigree_config.validated()
    requested = _canonical_contigs(contigs, "contigs")
    ordered_ids = _canonical_sample_ids(sample_ids)
    all_ordered = (
        requested if all_contigs is None
        else _canonical_contigs(all_contigs, "all_contigs")
    )
    requested_set = set(requested)
    if not requested_set.issubset(all_ordered):
        raise ValueError("contigs must be a subset of all_contigs")
    if tuple(value for value in all_ordered if value in requested_set) != requested:
        raise ValueError("contigs must preserve configured physical-contig order")
    if not isinstance(publish_completion, bool):
        raise TypeError("publish_completion must be boolean")
    if publish_completion and requested != all_ordered:
        raise ValueError(
            "only the complete configured contig set may publish pedigree preparation completion"
        )
    eligibility_identity, _ = _eligibility_provenance(
        parent_eligibility, ordered_ids
    )
    identity = pedigree_preparation_identity(
        preparation_config,
        settings,
        sample_ids=ordered_ids,
        contigs=all_ordered,
        painting_stage=painting_stage,
        raw_gl_stage=raw_gl_stage,
        raw_sites_stage=raw_sites_stage,
        raw_gl_key=raw_gl_key,
        raw_sites_key=raw_sites_key,
        raw_observed_mask_key=raw_observed_mask_key,
        parent_eligibility_identity=eligibility_identity,
        source_files=pedigree_cache.source_files(
            checkpoint_store, (painting_stage, raw_gl_stage, raw_sites_stage), all_ordered),
    )
    target_stage = pedigree_cache.versioned(target_stage, identity)
    checkpoint_store.bind_stage_identity(target_stage, identity)
    if checkpoint_store.stage_complete(target_stage) and checkpoint_store.global_done(target_stage):
        stored = checkpoint_store.load_global(target_stage)
        return tuple(PedigreeContigSummary(**dict(row, resumed=True))
                     for row in stored["summaries"] if row["contig"] in requested_set)
    if checkpoint_store.stage_complete(target_stage):
        missing = [
            contig for contig in all_ordered
            if not checkpoint_store.contig_done(target_stage, contig)
        ]
        if missing:
            raise RuntimeError(
                f"{target_stage} is marked complete but lacks: {missing}"
            )
    summaries = tuple(
        _prepare_one_contig(
            checkpoint_store,
            contig,
            ordered_ids,
            identity,
            preparation_config,
            settings,
            painting_stage=painting_stage,
            raw_gl_stage=raw_gl_stage,
            raw_sites_stage=raw_sites_stage,
            raw_gl_key=raw_gl_key,
            raw_sites_key=raw_sites_key,
            raw_observed_mask_key=raw_observed_mask_key,
            target_stage=target_stage,
        )
        for contig in requested
    )
    core_runtime.require_contig_checkpoints(
        checkpoint_store, target_stage, requested
    )
    if publish_completion:
        checkpoint_store.save_global(target_stage, {"summaries": [asdict(row) for row in summaries]})
        if not checkpoint_store.stage_complete(painting_stage):
            raise RuntimeError(f"{painting_stage} must be complete before pedigree preparation completion")
        core_runtime.require_contig_checkpoints(
            checkpoint_store, painting_stage, all_ordered
        )
        core_runtime.require_contig_checkpoints(
            checkpoint_store, target_stage, all_ordered
        )
        if not checkpoint_store.stage_complete(target_stage):
            checkpoint_store.mark_stage_complete(target_stage)
    return summaries


def _load_prepared_run(
        checkpoint_store,
        contigs,
        sample_ids,
        preparation_identity,
        pedigree_config,
        preparation_config,
        *,
        painting_stage,
        preparation_stage,
):
    chromosomes = []
    omissions = []
    source_identities = []
    for contig in contigs:
        painting_checkpoint = checkpoint_store.load_contig(painting_stage, contig)
        prepared_checkpoint = _validate_prepared_checkpoint(
            checkpoint_store.load_contig(preparation_stage, contig),
            expected_contig=contig,
            expected_sample_ids=sample_ids,
            expected_preparation_identity=preparation_identity,
            expected_painting=painting_checkpoint,
        )
        source_identities.append(_checkpoint_source_identity(prepared_checkpoint))
        chromosome = prepared_checkpoint.prepared_chromosome
        if chromosome.components:
            chromosomes.append(chromosome)
        else:
            omissions.append(pedigree_components.OmittedPaintingChromosome(
                contig,
                chromosome.component_count,
                chromosome.omitted_reason or "no_component_evidence",
            ))
        del painting_checkpoint, prepared_checkpoint, chromosome
    return (
        pedigree_components.PreparedPedigree(
            recombination_rate=preparation_config.recombination_rate,
            max_snps_per_bin=preparation_config.max_snps_per_bin,
            markers_per_information_block=(
                pedigree_config.markers_per_information_block
            ),
            effective_markers_per_information_block=(
                pedigree_config.parent_state_effective_markers_per_information_block
            ),
            sample_ids=sample_ids,
            chromosomes=tuple(chromosomes),
            omitted_chromosomes=tuple(omissions),
            source_mode=pedigree_config.parent_state_candidate_source_mode,
        ),
        tuple(source_identities),
    )


@timed_stage("pedigree")
def run_global_pedigree_inference(
        checkpoint_store,
        contigs,
        sample_ids,
        *,
        pedigree_config,
        parent_eligibility=None,
        preparation_config=EvidenceConfig(),
        n_workers=None,
        painting_stage=workflows_reconstruction.PAINTING_STAGE,
        preparation_stage=EVIDENCE_STAGE,
        target_stage=PEDIGREE_STAGE,
        raw_gl_stage,
        raw_sites_stage,
        raw_gl_key="global_probs",
        raw_sites_key="global_sites",
        raw_observed_mask_key="global_observed_mask",
):
    """Require complete painting/pedigree preparation inputs and run or resume global pedigree scoring."""

    ordered_contigs = _canonical_contigs(contigs, "contigs")
    ordered_ids = _canonical_sample_ids(sample_ids)
    settings = pedigree_config.validated()
    eligibility_identity, eligibility_summary = _eligibility_provenance(
        parent_eligibility, ordered_ids
    )
    preparation_identity = pedigree_preparation_identity(
        preparation_config,
        settings,
        sample_ids=ordered_ids,
        contigs=ordered_contigs,
        painting_stage=painting_stage,
        raw_gl_stage=raw_gl_stage,
        raw_sites_stage=raw_sites_stage,
        raw_gl_key=raw_gl_key,
        raw_sites_key=raw_sites_key,
        raw_observed_mask_key=raw_observed_mask_key,
        parent_eligibility_identity=eligibility_identity,
        source_files=pedigree_cache.source_files(
            checkpoint_store, (painting_stage, raw_gl_stage, raw_sites_stage), ordered_contigs),
    )
    scoring_identity = pedigree_cache.scoring_identity(
        preparation_identity, settings, parent_eligibility, ordered_ids, preparation_config)
    global_identity = pedigree_global_identity(
        preparation_identity, settings, eligibility_identity, scoring_identity)
    preparation_stage = pedigree_cache.versioned(preparation_stage, preparation_identity)
    published_stage = target_stage
    target_stage = pedigree_cache.versioned(target_stage, global_identity)
    checkpoint_store.bind_stage_identity(preparation_stage, preparation_identity)
    checkpoint_store.bind_stage_identity(target_stage, global_identity)
    for required_stage in (painting_stage, preparation_stage):
        if not checkpoint_store.stage_complete(required_stage):
            raise RuntimeError(
                f"{required_stage} must be complete before global pedigree inference"
            )
        core_runtime.require_contig_checkpoints(
            checkpoint_store, required_stage, ordered_contigs
        )

    target_complete = checkpoint_store.stage_complete(target_stage)
    target_global = checkpoint_store.global_done(target_stage)
    if target_complete and not target_global:
        raise RuntimeError(f"{target_stage} is complete but lacks _global")
    if target_global:
        payload = checkpoint_store.load_global(target_stage)
        if (
            payload.get("schema") != PEDIGREE_GLOBAL_SCHEMA
            or payload.get("backend") != PEDIGREE_GLOBAL_BACKEND
            or payload.get("identity") != global_identity
            or tuple(payload.get("ordered_sample_ids", ())) != ordered_ids
            or tuple(payload.get("ordered_contigs", ())) != ordered_contigs
        ):
            raise RuntimeError("global pedigree checkpoint identity/order mismatch")
        if not target_complete:
            checkpoint_store.mark_stage_complete(target_stage)
        pedigree_cache.publish_decision(checkpoint_store, published_stage, payload)
        return payload

    started = time.perf_counter()
    def prepare():
        return _load_prepared_run(
            checkpoint_store, ordered_contigs, ordered_ids, preparation_identity,
            settings, preparation_config, painting_stage=painting_stage,
            preparation_stage=preparation_stage)

    scored, source_identities, scores_resumed = pedigree_cache.load_or_score(
        checkpoint_store, scoring_identity, prepare, settings, parent_eligibility, preparation_config)
    scoring_elapsed = time.perf_counter() - started
    decision_started = time.perf_counter()
    result = pedigree_likelihoods.infer_scored_parent_state_evidence(
        scored, parent_eligibility=parent_eligibility, config=settings, n_workers=n_workers)
    elapsed = time.perf_counter() - started
    pedigree_result = result.pedigree_result
    tables = {
        "scientific_relationships": pedigree_result.relationships,
        "complete_relationships": pedigree_result.complete_relationships,
        "tier_a_relationships": pedigree_result.tier_a_relationships,
        "tier_b_relationships": pedigree_result.tier_b_relationships,
    }
    for name, frame in tables.items():
        if frame["Sample"].tolist() != list(ordered_ids):
            raise RuntimeError(f"{name} changed configured sample order")
    scored_by_contig = {
        value.contig: value for value in result.chromosome_results
    }
    omitted_by_contig = {
        value.contig: value for value in result.omitted_chromosomes
    }
    chromosome_summary = []
    for contig in ordered_contigs:
        if contig in scored_by_contig:
            value = scored_by_contig[contig]
            chromosome_summary.append({
                "contig": contig,
                "component_count": int(value.component_count),
                "scored_component_count": int(value.scored_component_count),
                "informative_markers": int(value.informative_markers),
            })
        elif contig in omitted_by_contig:
            value = omitted_by_contig[contig]
            chromosome_summary.append({
                "contig": contig,
                "component_count": int(value.component_count),
                "scored_component_count": 0,
                "informative_markers": 0,
            })
        else:
            raise RuntimeError(f"pedigree result lacks physical contig {contig}")
    payload = {
        "schema": PEDIGREE_GLOBAL_SCHEMA,
        "backend": PEDIGREE_GLOBAL_BACKEND,
        "identity": global_identity,
        "ordered_sample_ids": ordered_ids,
        "ordered_contigs": ordered_contigs,
        "prepared_source_identity_sha256": source_identities,
        **tables,
        'diagnostics': pedigree_result.diagnostics,
        'call_explanations': pedigree_result.call_explanations,
        'alternative_explanations': pedigree_result.alternative_explanations,
        'search_diagnostics': pedigree_result.search_diagnostics,
        'tier_b_candidate_sets': pedigree_result.tier_b_candidate_sets,
        'parent_state_calls': pedigree_result.parent_state_calls,
        'evidence_summary': pedigree_result.evidence_summary,
        'config': settings,
        "parent_eligibility": parent_eligibility,
        "parent_eligibility_identity": eligibility_identity,
        "parent_eligibility_summary": eligibility_summary,
        "chromosome_summary": tuple(chromosome_summary),
        "omitted_chromosomes": tuple({
            "contig": value.contig,
            "component_count": int(value.component_count),
            "reason": value.reason,
        } for value in result.omitted_chromosomes),
        "runtime": {
            "inference_elapsed_seconds": float(elapsed),
            "scoring_elapsed_seconds": float(scoring_elapsed),
            "decision_elapsed_seconds": time.perf_counter() - decision_started,
            "genetic_scores_resumed": scores_resumed,
            "requested_n_workers": (
                None if n_workers is None else int(n_workers)
            ),
            "bootstrap_worker_count": int(
                pedigree_result.bootstrap_worker_count
            ),
            "checkpoint_compression_threads": int(checkpoint_store.nthreads),
        },
    }
    ragged_diagnostics = getattr(result, "ragged_scoring_diagnostics", None)
    if ragged_diagnostics is not None:
        payload["ragged_scoring_diagnostics"] = (
            _serialize_ragged_scoring_diagnostics(ragged_diagnostics)
        )
    projected_diagnostics = getattr(
        result, "projected_scoring_diagnostics", None
    )
    if projected_diagnostics is not None:
        payload["projected_scoring_diagnostics"] = (
            _serialize_projected_scoring_diagnostics(projected_diagnostics)
        )
    checkpoint_store.save_global(target_stage, payload)
    if not checkpoint_store.global_done(target_stage):
        raise OSError(f"failed to checkpoint {target_stage}/_global")
    checkpoint_store.mark_stage_complete(target_stage)
    pedigree_cache.publish_decision(checkpoint_store, published_stage, payload)
    return payload


def write_pedigree_outputs(payload, output_dir):
    """Publish the primary pedigree, evidence tiers, and supporting diagnostics."""
    output = Path(output_dir) / "pedigree"
    output.mkdir(parents=True, exist_ok=True)
    names = (
        "scientific_relationships", "complete_relationships",
        "tier_a_relationships", "tier_b_relationships",
        'diagnostics',
        'parent_state_calls', 'evidence_summary',
        'call_explanations', 'alternative_explanations', 'search_diagnostics',
        'tier_b_candidate_sets',
    )
    for name in names:
        temporary = output / f".{name}.csv.tmp"
        payload[name].to_csv(temporary, index=False)
        temporary.replace(output / f"{name}.csv")
    return output


def run_pedigree(
        checkpoint_store, contigs, sample_ids, *, output_dir,
        raw_gl_stage, raw_sites_stage, raw_gl_key="global_probs",
        n_workers=None, parent_eligibility=None, all_contigs=None,
        publish_global=True, genetic_maps=None, recombination_rate=5e-8):
    """Canonical entry-point bridge: painting plus raw GL -> Tier-B pedigree.

    Numerical scoring uses one process with the allocated Numba threads;
    bootstrap uses up to that many single-threaded workers, in a later phase.
    No phase-correction or recombination-map stage is invoked.
    """
    workers = core_runtime.available_cpu_count() if n_workers is None else int(n_workers)
    if not 1 <= workers <= core_runtime.available_cpu_count():
        raise ValueError("pedigree workers must fit the current CPU affinity")
    print(f"PEDIGREE: ragged quadratic, finite family direction, top-20, Tier B; {workers} CPUs")
    print("Direction is a model assumption; same-depth and missing-parent crosses need caution.")
    with core_parallel.numba_thread_scope(workers):
        summaries, payload = run_or_resume_pedigree(
            checkpoint_store, contigs, sample_ids,
            pedigree_config=workflows_design.build_current_pedigree_config(),
            preparation_config=EvidenceConfig(
                recombination_rate=recombination_rate, genetic_maps=genetic_maps),
            parent_eligibility=parent_eligibility, n_workers=workers,
            raw_gl_stage=raw_gl_stage, raw_sites_stage=raw_sites_stage,
            raw_gl_key=raw_gl_key, all_contigs=all_contigs,
            publish_global=publish_global,
        )
    if payload is not None:
        output = write_pedigree_outputs(payload, output_dir)
        print(f"Pedigree output: {output / 'tier_b_relationships.csv'}")
    else:
        print("pedigree preparation shard cached; genome-wide pedigree awaits the complete contig set.")
    print("[COMPLETE] Pedigree ready for family refinement and final phase correction.")
    return summaries, payload


def run_or_resume_pedigree(
        checkpoint_store,
        contigs,
        sample_ids,
        *,
        pedigree_config,
        parent_eligibility=None,
        preparation_config=EvidenceConfig(),
        n_workers=None,
        painting_stage=workflows_reconstruction.PAINTING_STAGE,
        raw_gl_stage,
        raw_sites_stage,
        raw_gl_key="global_probs",
        raw_sites_key="global_sites",
        raw_observed_mask_key="global_observed_mask",
        all_contigs=None,
        publish_global=True,
):
    """Prepare pedigree preparation and, for a complete unsharded run, publish global pedigree scoring."""

    summaries = prepare_pedigree_contigs(
        checkpoint_store,
        contigs,
        sample_ids,
        pedigree_config=pedigree_config,
        parent_eligibility=parent_eligibility,
        preparation_config=preparation_config,
        painting_stage=painting_stage,
        raw_gl_stage=raw_gl_stage,
        raw_sites_stage=raw_sites_stage,
        raw_gl_key=raw_gl_key,
        raw_sites_key=raw_sites_key,
        raw_observed_mask_key=raw_observed_mask_key,
        all_contigs=all_contigs,
        publish_completion=publish_global,
    )
    if not publish_global:
        return summaries, None
    configured_contigs = contigs if all_contigs is None else all_contigs
    payload = run_global_pedigree_inference(
        checkpoint_store,
        configured_contigs,
        sample_ids,
        pedigree_config=pedigree_config,
        parent_eligibility=parent_eligibility,
        preparation_config=preparation_config,
        n_workers=n_workers,
        painting_stage=painting_stage,
        raw_gl_stage=raw_gl_stage,
        raw_sites_stage=raw_sites_stage,
        raw_gl_key=raw_gl_key,
        raw_sites_key=raw_sites_key,
        raw_observed_mask_key=raw_observed_mask_key,
    )
    return summaries, payload

import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.pedigree.config as module_pedigree_config
import haplotype_reconstruction.pedigree.likelihoods as pedigree_likelihoods
import haplotype_reconstruction.pedigree.sources as pedigree_sources

from haplotype_reconstruction.core import raw_evidence
