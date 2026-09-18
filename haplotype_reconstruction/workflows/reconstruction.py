"""Connect local feedback, component assembly, painting and evidence caches."""
from __future__ import annotations
from haplotype_reconstruction import PACKAGE_ROOT

import copy
import gc
from dataclasses import dataclass, field
import hashlib
import json
import math

from typing import Any, Mapping
import numpy as np
import haplotype_reconstruction.assembly.pipeline as assembly_pipeline
import haplotype_reconstruction.painting.components as painting_components
from haplotype_reconstruction.core import raw_evidence
from haplotype_reconstruction.workflows import block_feedback

STAGE2_PRODUCTION_SCHEMA = "stage2-production-v4"


STAGE2_PRODUCTION_BACKEND = (
    "feedback-selected-stage1-preprocess-hierarchy-unified-open-set-painting-t09-v4"
)


PAINTING_STAGE = "09_painting"


STAGE2_PAINTING_CODE_FILES = (
    'core/numerics.py',
    'core/genetic_map.py',
    'painting/components.py',
    'assembly/observations.py',
    'painting/model.py',
    'painting/evidence.py'
)


EXACT_OBSERVED_MASK_MODE = "positive_read_depth_v1"


SUPPORTED_GENOTYPE_EVIDENCE_MODE = "normalized_raw_linear_likelihood_v1"


STAGE2_PRODUCTION_CODE_FILES = (
    'painting/components.py',
    'core/runtime.py',
    'core/raw_evidence.py',
    'painting/checkpoints.py',
    'workflows/reconstruction.py',
    'painting/model.py',
)


def _default_release_config() -> assembly_pipeline.AssemblyConfig:
    return assembly_pipeline.AssemblyConfig()


@dataclass(frozen=True)
class ReconstructionConfig:
    """Assembly release and component painting settings."""

    release_config: assembly_pipeline.AssemblyConfig = field(
        default_factory=_default_release_config
    )
    feedback_config: block_feedback.BlockFeedbackConfig = field(
        default_factory=block_feedback.BlockFeedbackConfig)
    paint_recombination_rate: float = 5e-8
    paint_switch_penalty_per_snp: float = 1.0
    paint_robustness_epsilon: float = 1e-2
    paint_double_recomb_factor: float = 1.5
    paint_snps_per_bin: int = 100
    paint_batch_size: int = 1
    paint_cores: int = 1
    paint_ragged_working_memory_gb: float | None = None
    # Predeclared 10% within-model posterior-error budget; validate externally.
    paint_minimum_viterbi_public_class_posterior: float = 0.90

    def __post_init__(self) -> None:
        if not isinstance(self.release_config, assembly_pipeline.AssemblyConfig):
            raise TypeError("release_config must be an AssemblyConfig")
        for name in ("paint_snps_per_bin", "paint_batch_size", "paint_cores"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "paint_recombination_rate",
            "paint_switch_penalty_per_snp",
            "paint_robustness_epsilon",
            "paint_double_recomb_factor",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.paint_robustness_epsilon <= 1.0:
            raise ValueError("paint_robustness_epsilon must lie in [0, 1]")
        if self.paint_ragged_working_memory_gb is not None:
            value = float(self.paint_ragged_working_memory_gb)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "paint_ragged_working_memory_gb must be finite and positive"
                )
        posterior_threshold = float(
            self.paint_minimum_viterbi_public_class_posterior
        )
        if (not math.isfinite(posterior_threshold)
                or not 0.5 < posterior_threshold <= 1.0):
            raise ValueError(
                "paint_minimum_viterbi_public_class_posterior must lie in (0.5, 1]"
            )


@dataclass(frozen=True)
class Stage2ContigRunSummary:
    """Small operational summary; scientific detail remains in the checkpoint."""

    contig: str
    resumed: bool
    observed_mask_mode: str
    component_count: int
    evidence_eligible_component_sample_pairs: int
    total_component_sample_pairs: int


def observed_call_mask_from_read_counts(read_counts) -> np.ndarray:
    """Return the exact sample-by-site positive-depth observation mask."""

    reads = np.asarray(read_counts)
    if reads.ndim != 3 or reads.shape[2] < 1:
        raise ValueError("read_counts must have shape (samples, sites, alleles)")
    if np.any(~np.isfinite(reads)) or np.any(reads < 0):
        raise ValueError("read counts must be finite and non-negative")
    return np.ascontiguousarray(np.any(reads > 0, axis=2), dtype=np.bool_)


def _canonical_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty mapping")
    encoded = json.dumps(
        copy.deepcopy(dict(value)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    result = json.loads(encoded)
    if not isinstance(result, dict) or not result:
        raise ValueError(f"{name} must encode a nonempty mapping")
    return result


def _canonical_sample_ids(sample_ids) -> tuple[str, ...]:
    result = tuple(str(value) for value in sample_ids)
    if not result or len(result) != len(set(result)):
        raise ValueError("ordered sample IDs must be nonempty and unique")
    return result


def stage2_inputs_from_stage1(
    payload,
    *,
    expected_stage1_identity,
    expected_sample_ids,
):
    """Validate one Stage-1 payload and return canonical Stage-2 inputs."""

    if not isinstance(payload, Mapping):
        raise TypeError("Stage-1 payload must be a mapping")
    required = (
        "block_results", "global_probs", "global_sites",
        "global_observed_mask", "observed_call_mask_mode",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise KeyError("Stage-1 payload lacks: " + ", ".join(missing))

    expected_identity = _canonical_mapping(
        expected_stage1_identity, "expected_stage1_identity"
    )
    observed_identity = _canonical_mapping({
        "backend": payload.get("stage1_backend"),
        "config": payload.get("stage1_config"),
    }, "Stage-1 identity")
    if observed_identity != expected_identity:
        raise RuntimeError("Stage-1 payload identity mismatch")

    blocks = payload["block_results"]
    if not isinstance(blocks, core_haplotypes.BlockResults) or not blocks:
        raise TypeError("Stage-1 block_results must be a nonempty BlockResults")
    probabilities = np.asarray(payload["global_probs"])
    sites = np.asarray(payload["global_sites"])
    sample_ids = _canonical_sample_ids(expected_sample_ids)
    if probabilities.ndim != 3 or probabilities.shape[2] != 3:
        raise ValueError(
            "Stage-1 global_probs must have shape (samples, sites, 3)"
        )
    if probabilities.shape[0] != len(sample_ids):
        raise ValueError(
            "Stage-1 probability sample axis does not match sample IDs"
        )
    if sites.shape != (probabilities.shape[1],):
        raise ValueError("Stage-1 global_sites does not match global_probs")
    if payload.get("genotype_evidence_mode") != SUPPORTED_GENOTYPE_EVIDENCE_MODE:
        raise RuntimeError(
            "Stage-1 payload lacks normalized raw-likelihood provenance"
        )

    mask = np.ascontiguousarray(payload["global_observed_mask"], dtype=np.bool_)
    if mask.shape != probabilities.shape[:2]:
        raise ValueError(
            "Stage-1 global_observed_mask must have shape (samples, sites)"
        )
    mask_mode = payload["observed_call_mask_mode"]
    if mask_mode != EXACT_OBSERVED_MASK_MODE:
        raise RuntimeError("Stage-1 observation-mask provenance mismatch")

    return blocks, probabilities, sites, mask, mask_mode


def _validate_stage1_global_source(
    checkpoint_store,
    source_stage,
    contigs,
    ordered_ids,
    stage1_identity,
    *,
    require_complete,
):
    """Bind cached Stage-1 arrays to persisted sample and contig order."""

    if require_complete and not checkpoint_store.stage_complete(source_stage):
        raise RuntimeError(f"{source_stage} is not marked complete")
    if not checkpoint_store.global_done(source_stage):
        raise RuntimeError(f"{source_stage} lacks its global checkpoint")
    payload = checkpoint_store.load_global(source_stage)
    try:
        stored_ids = _canonical_sample_ids(payload.get("sample_ids", ()))
        if stored_ids != ordered_ids:
            raise RuntimeError(f"{source_stage} cached sample order mismatch")
        stored_contigs = tuple(
            str(value) for value in payload.get("contigs", ())
        )
        if stored_contigs != contigs:
            raise RuntimeError(f"{source_stage} cached contig order mismatch")
        observed_identity = _canonical_mapping({
            "backend": payload.get("stage1_backend"),
            "config": payload.get("stage1_config"),
        }, "global Stage-1 identity")
        if observed_identity != _canonical_mapping(
                stage1_identity, "stage1_identity"):
            raise RuntimeError(f"{source_stage} global Stage-1 identity mismatch")
        if payload.get("genotype_evidence_mode") != (
                SUPPORTED_GENOTYPE_EVIDENCE_MODE):
            raise RuntimeError(
                f"{source_stage} global genotype-evidence provenance mismatch"
            )
    finally:
        del payload


def _production_code_identity() -> dict[str, str]:
    root = PACKAGE_ROOT
    return {
        filename: hashlib.sha256((root / filename).read_bytes()).hexdigest()
        for filename in STAGE2_PRODUCTION_CODE_FILES
    }


def stage2_painting_code_identity(digest_overrides=None) -> dict[str, str]:
    """Hash the scientific component-painting implementation closure."""

    overrides = {} if digest_overrides is None else dict(digest_overrides)
    unknown = set(overrides) - set(STAGE2_PAINTING_CODE_FILES)
    if unknown:
        raise ValueError(
            "unknown painting code identity override files: "
            + ", ".join(sorted(unknown))
        )
    root = PACKAGE_ROOT
    result = {}
    for filename in STAGE2_PAINTING_CODE_FILES:
        if filename in overrides:
            value = str(overrides[filename]).lower()
            if (len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError(
                    f"painting code digest override for {filename} is not SHA256"
                )
            result[filename] = value
        else:
            result[filename] = hashlib.sha256(
                (root / filename).read_bytes()
            ).hexdigest()
    return result


def stage2_painting_product_identity(
        config, *, code_digest_overrides=None, chromosome_map=None) -> dict[str, Any]:
    """Return the scientific painting identity, excluding scheduling knobs."""

    if not isinstance(config, ReconstructionConfig):
        raise TypeError("config must be a ReconstructionConfig")
    return painting_checkpoints.component_painting_product_identity(
        {
            "recombination_rate": (
                config.paint_recombination_rate if chromosome_map is None
                else chromosome_map.fallback_rate_per_bp),
            **({"genetic_map": chromosome_map.record()}
               if chromosome_map is not None and chromosome_map.has_map else {}),
            "switch_penalty_per_snp": config.paint_switch_penalty_per_snp,
            "robustness_epsilon": config.paint_robustness_epsilon,
            "double_recomb_factor": config.paint_double_recomb_factor,
            "snps_per_bin": config.paint_snps_per_bin,
            "minimum_viterbi_public_class_posterior": (
                config.paint_minimum_viterbi_public_class_posterior
            ),
            "minimum_viterbi_public_class_posterior_status": (
                "predeclared-90-percent-within-model-support-v1"
            ),
            "founder_panel": "unified-open-set-frozen-calls-v3",
            "open_set_founder_model": (
                "fixed-trajectories-beta11-unanchored-shared-"
                "background-independent-v2"
            ),
            "public_state_partition": (
                "anchored-classes-distinct-unanchored-and-background-unknown-v1"
            ),
            "trajectory_equivalence": "selected-site-exact-minus1-0-1-v1",
            "background_equivalent_founder_weight": 1.0,
            "posterior_summary": (
                "unordered-public-class-forward-backward-v3"
            ),
            "release_policy": (
                "whole-bin-viterbi-induced-public-class-posterior-threshold-v2"
            ),
            "sample_observation_policy": (
                "positive-depth-raw-gl-normalized-exact-neutral-v2"
            ),
        },
        stage2_painting_code_identity(code_digest_overrides),
    )


def _ragged_working_memory_bytes(config) -> int | None:
    value = config.paint_ragged_working_memory_gb
    return None if value is None else int(float(value) * (1024 ** 3))


def _painting_runtime_provenance(
        config, checkpoint_threads: int, painting_bundle=None) -> dict[str, Any]:
    """Record execution choices that must not define the scientific product."""

    ragged_components = []
    if painting_bundle is not None:
        for component in painting_bundle.components:
            diagnostics = component.ragged_diagnostics
            if diagnostics is None:
                continue
            ragged_components.append({
                "component_index": int(component.component_index),
                "working_memory_budget_bytes": int(
                    diagnostics.hmm_working_memory_budget_bytes
                ),
                "estimated_bytes_per_sample": int(
                    diagnostics.hmm_estimated_bytes_per_sample
                ),
                "effective_batch_size": int(diagnostics.hmm_batch_size),
                "numba_thread_count": int(diagnostics.hmm_thread_count),
            })
    return {
        "release_core_ceiling": int(config.release_config.num_processes),
        "painting_core_ceiling": int(config.paint_cores),
        "painting_batch_size": int(config.paint_batch_size),
        "ragged_working_memory_override_bytes": (
            _ragged_working_memory_bytes(config)
        ),
        "ragged_components": tuple(ragged_components),
        "checkpoint_threads": int(checkpoint_threads),
        "release_and_painting_phases_overlap": False,
    }


def stage2_production_stage_identity(
    config,
    *,
    stage1_identity,
    sample_ids,
    contigs,
    source_stage,
    target_stage,
    probabilities_stage,
    probabilities_key,
    genetic_maps=None,
) -> dict[str, Any]:
    """Return the shared multi-contig production-stage resume identity."""

    if not isinstance(config, ReconstructionConfig):
        raise TypeError("config must be a ReconstructionConfig")
    return _canonical_mapping({
        "schema": STAGE2_PRODUCTION_SCHEMA,
        "backend": STAGE2_PRODUCTION_BACKEND,
        "stage1_identity": _canonical_mapping(stage1_identity, "stage1_identity"),
        "ordered_sample_ids": _canonical_sample_ids(sample_ids),
        "ordered_contigs": tuple(str(value) for value in contigs),
        "source_stage": str(source_stage),
        "probabilities_stage": (
            None if probabilities_stage is None else str(probabilities_stage)
        ),
        "probabilities_key": str(probabilities_key),
        "target_stage": str(target_stage),
        "config": {
            "block_feedback": block_feedback.scientific_identity(config.feedback_config),
            "release_scientific": (
                assembly_pipeline._release_scientific_config(
                    config.release_config
                )
            ),
            "painting_product": stage2_painting_product_identity(config),
            **({"genetic_maps": genetic_maps.identity()} if genetic_maps is not None else {}),
        },
        "release_code_identity_sha256": (
            assembly_pipeline.stage2_release_code_identity()
        ),
        "production_code_identity_sha256": _production_code_identity(),
        "downstream_boundary": {
            "component_local_founder_ids": True,
            "component_aware_downstream_supported": True,
        },
    }, "Stage-2 production identity")


def _painting_kwargs(config, chromosome_map=None):
    return {
        **({"chromosome_map": chromosome_map} if chromosome_map is not None else {}),
        "recomb_rate": config.paint_recombination_rate,
        "switch_penalty_per_snp": config.paint_switch_penalty_per_snp,
        "robustness_epsilon": config.paint_robustness_epsilon,
        "double_recomb_factor": config.paint_double_recomb_factor,
        "snps_per_bin": config.paint_snps_per_bin,
        "batch_size": config.paint_batch_size,
        "working_memory_bytes": _ragged_working_memory_bytes(config),
        "minimum_viterbi_public_class_posterior": (
            config.paint_minimum_viterbi_public_class_posterior
        ),
    }


def _checkpoint_summary(contig, checkpoint, resumed, mask_mode):
    bundle = checkpoint.painting_bundle
    evidence_eligible = sum(
        int(np.sum(component.evidence_eligible_sample_mask))
        for component in bundle.components
    )
    return Stage2ContigRunSummary(
        contig=str(contig),
        resumed=bool(resumed),
        observed_mask_mode=str(mask_mode),
        component_count=len(bundle.components),
        evidence_eligible_component_sample_pairs=evidence_eligible,
        total_component_sample_pairs=len(bundle.components) * bundle.num_samples,
    )


def _run_or_resume_one_contig(
    checkpoint_store,
    contig,
    ordered_ids,
    *,
    stage1_identity,
    config,
    source_stage,
    probabilities_stage,
    probabilities_key,
    target_stage,
    painter_factory,
    chromosome_map=None,
):
    """Run one contig without retaining large arrays across boundaries."""

    source = checkpoint_store.load_contig(source_stage, contig)
    probabilities_payload = None
    try:
        if probabilities_stage is not None:
            if "global_probs" in source:
                raise ValueError(
                    f"{contig}: source payload already contains global_probs"
                )
            probabilities_payload = checkpoint_store.load_contig(
                probabilities_stage, contig
            )
            if probabilities_key not in probabilities_payload:
                raise KeyError(
                    f"{probabilities_stage}/{contig} lacks "
                    f"{probabilities_key!r}"
                )
            probabilities = probabilities_payload[probabilities_key]
            source = dict(source)
            source["global_probs"] = probabilities
            probabilities_payload = None
            gc.collect()
        return _run_or_resume_loaded_contig(
            checkpoint_store,
            contig,
            ordered_ids,
            source,
            stage1_identity=stage1_identity,
            config=config,
            target_stage=target_stage,
            painter_factory=painter_factory,
            chromosome_map=chromosome_map,
            raw_evidence_source=dict(
                raw_gl_stage=probabilities_stage or source_stage,
                raw_sites_stage=source_stage,
                raw_gl_key=probabilities_key if probabilities_stage else "global_probs",
            ),
        )
    finally:
        del source, probabilities_payload
        gc.collect()
        core_parallel.malloc_trim()


def _run_or_resume_loaded_contig(
    checkpoint_store,
    contig,
    ordered_ids,
    source,
    *,
    stage1_identity,
    config,
    target_stage,
    painter_factory,
    chromosome_map=None,
    raw_evidence_source=None,
):
    blocks, probabilities, sites, observed, mask_mode = (
        stage2_inputs_from_stage1(
            source,
            expected_stage1_identity=stage1_identity,
            expected_sample_ids=ordered_ids,
        )
    )
    if raw_evidence_source is not None:
        raw_evidence.save(checkpoint_store, contig, probabilities, sites, observed,
                          source, **raw_evidence_source)
    with core_parallel.numba_thread_scope(config.release_config.num_processes):
        chromosome_evidence = assembly_pipeline.prepare_chromosome_evidence(
            blocks, probabilities, sites, observed)
    blocks, feedback_identity = block_feedback.run_block_feedback(
        checkpoint_store, contig, blocks, probabilities, sites, observed, ordered_ids,
        stage1_identity=stage1_identity, assembly_config=config.release_config,
        config=config.feedback_config, chromosome_map=chromosome_map,
        chromosome_evidence=chromosome_evidence)
    release_stage1_identity = _canonical_mapping(
        stage1_identity, "stage1_identity"
    )
    release_stage1_identity["observed_call_mask_mode"] = mask_mode
    release_stage1_identity["block_feedback"] = feedback_identity
    expected_release_identity = (
        assembly_pipeline.stage2_release_identity_record(
            config.release_config,
            stage1_identity=release_stage1_identity,
            sample_ids=ordered_ids,
            input_blocks=blocks,
            global_probs=probabilities,
            global_sites=sites,
            global_observed_mask=observed,
            chromosome_map=chromosome_map,
            chromosome_evidence=chromosome_evidence,
        )
    )
    expected_painting_identity = stage2_painting_product_identity(
        config, chromosome_map=chromosome_map)
    runtime_provenance = _painting_runtime_provenance(
        config, checkpoint_store.nthreads
    )

    if checkpoint_store.contig_done(target_stage, contig):
        checkpoint = painting_checkpoints.validate_t09_component_checkpoint(
            checkpoint_store.load_contig(target_stage, contig),
            expected_sample_ids=ordered_ids,
            expected_release_identity=expected_release_identity,
            expected_painting_product_identity=expected_painting_identity,
        )
        return _checkpoint_summary(contig, checkpoint, True, mask_mode)

    # Release pools and painting pools never overlap. In particular, no idle
    # painter workers retain memory or process slots during hierarchical work.
    release_checkpoints = assembly_checkpoints.AssemblyCheckpointStore(
        checkpoint_store,
        work_stage=f"{target_stage}_release_work",
        contig=contig,
    )
    with core_parallel.numba_thread_scope(config.release_config.num_processes):
        release = assembly_pipeline.assemble_chromosome(
            blocks,
            probabilities,
            sites,
            observed,
            ordered_ids,
            stage1_identity=release_stage1_identity,
            config=config.release_config,
            release_checkpoints=release_checkpoints,
            chromosome_map=chromosome_map,
            chromosome_evidence=chromosome_evidence,
        )
    if release.get("identity") != expected_release_identity:
        raise RuntimeError(
            f"{contig}: Stage-2 release identity changed during execution"
        )
    components = release.get("components")
    manifest = release.get("component_manifest")
    manifest_blocks = core_runtime.validate_phase_component_manifest(
        manifest
    )
    if tuple(components) != manifest_blocks:
        raise RuntimeError(
            f"{contig}: component manifest differs from release output"
        )
    del release
    gc.collect()
    core_parallel.malloc_trim()

    with painter_factory(num_processes=config.paint_cores) as painter:
        painting_bundle = painter.paint_components(
            components,
            probabilities,
            sites,
            sample_observed_mask=observed,
            **_painting_kwargs(config, chromosome_map),
        )
    runtime_provenance = _painting_runtime_provenance(
        config, checkpoint_store.nthreads, painting_bundle
    )
    checkpoint = painting_checkpoints.build_t09_component_checkpoint(
        manifest,
        painting_bundle,
        ordered_ids,
        expected_release_identity,
        expected_painting_identity,
        runtime_provenance,
    )
    checkpoint_store.save_contig(target_stage, contig, checkpoint)
    checkpoint = painting_checkpoints.validate_t09_component_checkpoint(
        checkpoint_store.load_contig(target_stage, contig),
        expected_sample_ids=ordered_ids,
        expected_release_identity=expected_release_identity,
        expected_painting_product_identity=expected_painting_identity,
    )
    return _checkpoint_summary(contig, checkpoint, False, mask_mode)


def run_reconstruction(
    checkpoint_store,
    contigs,
    sample_ids,
    *,
    stage1_identity,
    config,
    source_stage="T00_founder_templates",
    target_stage=PAINTING_STAGE,
    probabilities_stage=None,
    probabilities_key="global_probs",
    all_contigs=None,
    publish_completion=True,
    painter_factory=painting_components.ComponentPainter,
    genetic_maps=None,
):
    """Run the canonical Stage-2 route and atomically checkpoint every contig."""

    if not isinstance(config, ReconstructionConfig):
        raise TypeError("config must be a ReconstructionConfig")
    if probabilities_stage is not None:
        probabilities_stage = str(probabilities_stage)
        if not probabilities_stage:
            raise ValueError("probabilities_stage must be nonempty")
    probabilities_key = str(probabilities_key)
    if not probabilities_key:
        raise ValueError("probabilities_key must be nonempty")
    contigs = tuple(str(value) for value in contigs)
    if not contigs or len(contigs) != len(set(contigs)):
        raise ValueError("contigs must be a nonempty unique ordered collection")
    if all_contigs is None:
        all_contigs = contigs
    else:
        all_contigs = tuple(str(value) for value in all_contigs)
    if not all_contigs or len(all_contigs) != len(set(all_contigs)):
        raise ValueError(
            "all_contigs must be a nonempty unique ordered collection"
        )
    requested = set(contigs)
    if not requested.issubset(all_contigs):
        raise ValueError("contigs must be a subset of all_contigs")
    if tuple(value for value in all_contigs if value in requested) != contigs:
        raise ValueError("contigs must preserve all_contigs order")
    if not isinstance(publish_completion, bool):
        raise TypeError("publish_completion must be boolean")
    ordered_ids = _canonical_sample_ids(sample_ids)
    available = core_runtime.available_cpu_count()
    requested = max(config.release_config.num_processes, config.paint_cores)
    if requested > available:
        raise ValueError(
            f"requested core ceiling ({requested}) exceeds process affinity "
            f"({available})"
        )

    _validate_stage1_global_source(
        checkpoint_store,
        source_stage,
        all_contigs,
        ordered_ids,
        stage1_identity,
        require_complete=publish_completion,
    )
    stage_identity = stage2_production_stage_identity(
        config,
        stage1_identity=stage1_identity,
        sample_ids=ordered_ids,
        contigs=all_contigs,
        source_stage=source_stage,
        target_stage=target_stage,
        probabilities_stage=probabilities_stage,
        probabilities_key=probabilities_key,
        genetic_maps=genetic_maps,
    )
    checkpoint_store.bind_stage_identity(target_stage, stage_identity)
    missing = [
        contig for contig in contigs
        if not checkpoint_store.contig_done(target_stage, contig)
    ]
    if checkpoint_store.stage_complete(target_stage) and missing:
        raise RuntimeError(
            f"{target_stage} is marked complete but lacks: {missing}"
        )

    summaries = tuple(
        _run_or_resume_one_contig(
            checkpoint_store,
            contig,
            ordered_ids,
            stage1_identity=stage1_identity,
            config=config,
            source_stage=source_stage,
            probabilities_stage=probabilities_stage,
            probabilities_key=probabilities_key,
            target_stage=target_stage,
            painter_factory=painter_factory,
            chromosome_map=(None if genetic_maps is None
                            else genetic_maps.for_contig(contig)),
        )
        for contig in contigs
    )
    core_runtime.require_contig_checkpoints(
        checkpoint_store, target_stage, contigs
    )
    if publish_completion:
        core_runtime.require_contig_checkpoints(
            checkpoint_store, target_stage, all_contigs
        )
        if not checkpoint_store.stage_complete(target_stage):
            checkpoint_store.mark_stage_complete(target_stage)
    return summaries

import haplotype_reconstruction.assembly.checkpoints as assembly_checkpoints
import haplotype_reconstruction.core.haplotypes as core_haplotypes
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.painting.checkpoints as painting_checkpoints
