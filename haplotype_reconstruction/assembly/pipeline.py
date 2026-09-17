"""assembly / pipeline for the canonical reconstruction pipeline."""
from __future__ import annotations
from haplotype_reconstruction import PACKAGE_ROOT

import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

import time
from typing import Any, Mapping
import numpy as np
import haplotype_reconstruction.assembly.completion as assembly_completion
from .structured_transitions import StructuredTransitionConfig, configured_transition
from .panel_search import PanelSearchConfig, configured_panel_search
from . import founder_refinement, evidence as assembly_evidence

STAGE2_RELEASE_SCHEMA = "stage2-release-v1"


STAGE2_RELEASE_BACKEND = (
    "preprocess-partial-founder-hierarchy-phase-refinement-v18"
)


INTERSECTION_SAMPLE_RULE = "intersection_across_linking_proxy_batch_v2"


STAGE2_RELEASE_CODE_IDENTITY_FILES = tuple(sorted(set(
    assembly_completion.STAGE2_PREPROCESS_SCIENTIFIC_DEPENDENCIES + ('core/numerics.py', 'core/config.py', 'core/genotypes.py', 'discovery/objectives.py', 'discovery/blocks.py', 'assembly/paths.py', 'assembly/linking.py', 'assembly/chimera_kernels.py', 'assembly/chimera_resolution.py', 'assembly/chimera_scoring.py', 'core/parallel.py', 'assembly/hierarchy.py', 'core/genetic_map.py', 'assembly/micro_hmm.py', 'assembly/micro_hmm_log.py', 'assembly/macro_hmm.py', 'assembly/edge_counts.py', 'core/runtime.py', 'core/environment.py', 'assembly/pipeline.py', 'assembly/checkpoints.py')
)))


STAGE2_RELEASE_CODE_IDENTITY_FILES += (
    'assembly/structured_transitions.py', 'assembly/panel_search.py',
    'assembly/panel_scoring.py', 'assembly/panel_candidates.py', 'assembly/partial_emissions.py',
    'assembly/founder_refinement.py', 'assembly/founder_path_search.py',
    'assembly/founder_scoring.py', 'assembly/founder_dual_search.py',
    'assembly/founder_exchanges.py', 'assembly/founder_windows.py',
    'assembly/founder_intervals.py',
    'assembly/founder_count.py', 'assembly/founder_count_bound.py', 'assembly/evidence.py',
    'assembly/founder_workspace.py', 'assembly/founder_count_workers.py',
    'assembly/founder_candidates.py', 'assembly/founder_sparse.py',
    'assembly/founder_background.py', 'assembly/founder_delta.py',
    'assembly/founder_packing.py', 'assembly/founder_checkpoints.py',
    'assembly/founder_site_kernels.py', 'assembly/founder_dual_short.py',
    'assembly/founder_evidence.py',
    'assembly/founder_beam.py', 'assembly/founder_beam_kernels.py',
)

_RELEASE_RUNTIME_CONFIG_FIELDS = frozenset((
    "num_processes",
    "maxtasksperchild",
    "min_gb_per_worker",
    "preprocess_diagnostics_mode",
    "verbose",
))


def _configured_founder_refinement():
    from ..core.environment import assembly_founder_refinement
    return founder_refinement.FounderRefinementConfig(
        enabled=assembly_founder_refinement())


@dataclass(frozen=True)
class AssemblyConfig:
    """Dense transitions and bounded panel search by default; one core ceiling."""

    preprocess_config: assembly_completion.CompletionConfig = field(
        default_factory=assembly_completion.CompletionConfig
    )
    max_level: int = 4
    l1_batch_size: int = 10
    higher_level_batch_size: int = 10
    min_boundary_informative_samples: int = 4
    beam_width: int = 200
    max_founders: int = 12
    max_sites_for_linking: int = 2000
    recombination_rate: float = 5e-8
    n_generations: int = 3
    recombination_tolerance: float = 0.5
    top_n_swap: int = 20
    max_cr_iterations: int = 10
    paint_penalty: float = 10.0
    min_hotspot_samples: int = 5
    cc_scale: float = 0.5
    num_processes: int = 1
    maxtasksperchild: int | None = None
    min_gb_per_worker: float = 4.0
    preprocess_diagnostics_mode: str = "compact"
    verbose: bool = False
    structured_transition_config: StructuredTransitionConfig | None = field(default_factory=configured_transition)
    panel_search_config: PanelSearchConfig | None = field(default_factory=configured_panel_search)
    founder_refinement_config: founder_refinement.FounderRefinementConfig = field(
        default_factory=_configured_founder_refinement)

    def __post_init__(self) -> None:
        if not isinstance(self.founder_refinement_config, founder_refinement.FounderRefinementConfig):
            raise TypeError("founder_refinement_config must be FounderRefinementConfig")
        if self.panel_search_config is not None and not isinstance(self.panel_search_config, PanelSearchConfig):
            raise TypeError("panel_search_config must be PanelSearchConfig or None")
        if (self.structured_transition_config is not None
                and not isinstance(self.structured_transition_config, StructuredTransitionConfig)):
            raise TypeError("structured_transition_config must be StructuredTransitionConfig or None")
        if not isinstance(self.preprocess_config, assembly_completion.CompletionConfig):
            raise TypeError("preprocess_config must be a CompletionConfig")
        for name in (
            "max_level", "l1_batch_size", "higher_level_batch_size",
            "min_boundary_informative_samples", "beam_width", "max_founders",
            "max_sites_for_linking", "n_generations", "top_n_swap",
            "max_cr_iterations", "min_hotspot_samples", "num_processes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_level > 4:
            raise ValueError("the Stage-2 release supports hierarchy L1-L4")
        if self.maxtasksperchild is not None and (
                isinstance(self.maxtasksperchild, bool)
                or int(self.maxtasksperchild) != self.maxtasksperchild
                or self.maxtasksperchild < 1):
            raise ValueError("maxtasksperchild must be None or positive")
        for name in (
            "recombination_rate", "recombination_tolerance", "paint_penalty",
            "cc_scale", "min_gb_per_worker",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.min_gb_per_worker == 0.0:
            raise ValueError("min_gb_per_worker must be positive")
        if self.preprocess_diagnostics_mode not in {"full", "compact"}:
            raise ValueError(
                "preprocess_diagnostics_mode must be 'full' or 'compact'"
            )


def _release_scientific_config(config: AssemblyConfig) -> dict[str, Any]:
    values = asdict(config)
    values.pop("preprocess_config")
    for name in _RELEASE_RUNTIME_CONFIG_FIELDS:
        values.pop(name)
    return values


def _release_execution_config(config: AssemblyConfig) -> dict[str, Any]:
    values = asdict(config)
    return {
        name: values[name]
        for name in sorted(_RELEASE_RUNTIME_CONFIG_FIELDS)
    }


def _canonical_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty mapping")
    text = json.dumps(
        copy.deepcopy(dict(value)), sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )
    result = json.loads(text)
    if not isinstance(result, dict) or not result:
        raise ValueError(f"{name} must encode a nonempty mapping")
    return result


def _canonical_sample_ids(sample_ids, n_samples: int) -> tuple[str, ...]:
    values = tuple(str(value) for value in sample_ids)
    if len(values) != n_samples:
        raise ValueError("ordered sample IDs must match the sample axis")
    if not values or len(values) != len(set(values)):
        raise ValueError("ordered sample IDs must be nonempty and unique")
    return values


def _array_digest(value) -> str:
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


def _update_array_digest(digest, name: str, value) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(_array_digest(value).encode("ascii"))


def _block_discrete(block: core_haplotypes.BlockResult) -> np.ndarray:
    keys = tuple(sorted(block.haplotypes))
    positions = np.asarray(block.positions)
    discrete = np.asarray(getattr(block, "discrete_haps", None))
    if not keys or positions.ndim != 1 or positions.size == 0:
        raise ValueError("every Stage-2 input block must be nonempty")
    if discrete.shape != (len(keys), positions.size):
        raise ValueError("block.discrete_haps is not founder-by-site aligned")
    if np.any(~np.isin(discrete, (-1, 0, 1))):
        raise ValueError("discrete founder alleles must be -1, 0, or 1")
    return discrete


def _layout(blocks):
    return tuple(
        (
            tuple(int(value) for value in np.asarray(block.positions)),
            tuple(sorted(block.haplotypes)),
        )
        for block in blocks
    )


def _input_block_digest(input_blocks) -> str:
    digest = hashlib.sha256()
    for index, block in enumerate(input_blocks):
        positions = np.asarray(block.positions, dtype=np.int64)
        discrete = _block_discrete(block)
        keys = tuple(sorted(block.haplotypes))
        digest.update(str(index).encode("ascii"))
        digest.update(json.dumps(
            keys, default=str, separators=(",", ":")
        ).encode())
        _update_array_digest(digest, "positions", positions)
        _update_array_digest(digest, "discrete_haps", discrete)
        flags = getattr(block, "keep_flags", None)
        flags = (
            np.ones(len(positions), dtype=np.bool_)
            if flags is None else np.asarray(flags)
        )
        if flags.shape != positions.shape:
            raise ValueError("keep_flags must match block positions")
        _update_array_digest(digest, "keep_flags", flags)
        for key in keys:
            _update_array_digest(
                digest, f"haplotype:{key}", block.haplotypes[key]
            )
        for name in (
            "founder_alt_pseudo_probability",
            "n_directional_site_supporters",
        ):
            value = getattr(block, name, None)
            digest.update(name.encode("ascii"))
            if value is None:
                digest.update(b"none")
            else:
                _update_array_digest(digest, name, value)
        for name in (
            "sample_has_observed_kept_depth",
            "wildcard_slots",
            "wildcard_mass",
        ):
            value = getattr(block, name, None)
            digest.update(name.encode("ascii"))
            if value is None:
                digest.update(b"absent")
            else:
                digest.update(b"present")
                _update_array_digest(digest, name, value)
        for name in (
            "missing_aware_break_before", "missing_aware_break_after",
            "missing_aware_break_reason_before",
            "missing_aware_break_reason_after",
        ):
            digest.update(json.dumps(
                [name, getattr(block, name, None)],
                sort_keys=True, separators=(",", ":"), default=str,
            ).encode())
    return digest.hexdigest()


def stage2_release_code_identity(digest_overrides=None) -> dict[str, str]:
    overrides = {} if digest_overrides is None else dict(digest_overrides)
    unknown = set(overrides) - set(STAGE2_RELEASE_CODE_IDENTITY_FILES)
    if unknown:
        raise ValueError(
            "unknown code identity override files: " + ", ".join(sorted(unknown))
        )
    root = PACKAGE_ROOT
    result = {}
    for filename in STAGE2_RELEASE_CODE_IDENTITY_FILES:
        if filename in overrides:
            value = str(overrides[filename]).lower()
            if (len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError(
                    f"code digest override for {filename} is not SHA256"
                )
            result[filename] = value
        else:
            result[filename] = hashlib.sha256(
                (root / filename).read_bytes()
            ).hexdigest()
    return result


def _validate_global_inputs(
    input_blocks, global_probs, global_sites, global_observed_mask,
    chromosome_evidence=None,
):
    if not isinstance(input_blocks, core_haplotypes.BlockResults) or not input_blocks:
        raise TypeError("input_blocks must be a nonempty BlockResults")
    probabilities = np.asarray(global_probs)
    sites = np.asarray(global_sites)
    observed = np.asarray(global_observed_mask, dtype=np.bool_)
    if probabilities.ndim != 3 or probabilities.shape[2] != 3:
        raise ValueError("global_probs must have shape (samples, sites, 3)")
    if sites.shape != (probabilities.shape[1],):
        raise ValueError("global_sites must match global_probs")
    if observed.shape != probabilities.shape[:2]:
        raise ValueError("global_observed_mask must have shape (samples, sites)")
    if sites.size > 1 and np.any(sites[1:] <= sites[:-1]):
        raise ValueError("global_sites must be strictly increasing")
    if chromosome_evidence is not None:
        chromosome_evidence.check_arrays(probabilities, sites, observed)
    elif not assembly_evidence.finite_nonnegative(probabilities):
        raise ValueError("global_probs must be finite and non-negative")
    previous = None
    for block in input_blocks:
        _block_discrete(block)
        positions = np.asarray(block.positions)
        if positions.size > 1 and np.any(positions[1:] <= positions[:-1]):
            raise ValueError("block positions must be strictly increasing")
        indices = np.searchsorted(sites, positions)
        if (np.any(indices >= sites.size)
                or not np.array_equal(sites[indices], positions)):
            raise ValueError("every block position must occur in global_sites")
        if previous is not None and positions[0] <= previous:
            raise ValueError("input blocks must be ordered and non-overlapping")
        previous = positions[-1]
    return probabilities, sites, observed


def prepare_chromosome_evidence(input_blocks, probabilities, sites, observed):
    """Validate/hash raw evidence once during one read-only chromosome run."""
    probabilities, sites, observed = _validate_global_inputs(
        input_blocks, probabilities, sites, observed)
    return assembly_evidence.ChromosomeEvidence(
        probabilities, sites, observed, {
            "global_probs": _array_digest(probabilities),
            "global_sites": _array_digest(sites),
            "global_observed_mask": _array_digest(observed),
        })


def _preprocess_inputs(input_blocks, probabilities, sites, observed,
                       chromosome_evidence=None):
    if chromosome_evidence is not None:
        chromosome_evidence.check_arrays(probabilities, sites, observed)
        return chromosome_evidence.prepare(input_blocks)
    indices, kept = assembly_evidence.block_indices_and_keep(input_blocks, sites)
    neutral, effective_observed = assembly_evidence.neutral_evidence(
        probabilities, observed, kept)
    return assembly_evidence.block_views(neutral, effective_observed, indices)


def stage2_release_identity_record(
    config: AssemblyConfig,
    *,
    stage1_identity: Mapping[str, Any],
    sample_ids,
    input_blocks,
    global_probs,
    global_sites,
    global_observed_mask,
    fold_assignments=None,
    preprocess_identity=None,
    code_digest_overrides=None,
    chromosome_map=None,
    chromosome_evidence=None,
) -> dict[str, Any]:
    """Return the exact stable scientific and input identity for one release."""

    if not isinstance(config, AssemblyConfig):
        raise TypeError("config must be an AssemblyConfig")
    probabilities, sites, observed = _validate_global_inputs(
        input_blocks, global_probs, global_sites, global_observed_mask,
        chromosome_evidence,
    )
    ordered_ids = _canonical_sample_ids(sample_ids, probabilities.shape[0])
    if preprocess_identity is None:
        preprocess_identity = assembly_completion.stage2_preprocess_identity(
            config.preprocess_config,
            fold_assignments=fold_assignments,
            n_samples=probabilities.shape[0],
        ).record()
    else:
        preprocess_identity = _canonical_mapping(
            preprocess_identity, "preprocess_identity"
        )
    record = {
        "schema": STAGE2_RELEASE_SCHEMA,
        "backend": STAGE2_RELEASE_BACKEND,
        "stage1_identity": _canonical_mapping(
            stage1_identity, "stage1_identity"
        ),
        "ordered_sample_ids": ordered_ids,
        "preprocess_identity": preprocess_identity,
        "config": {
            "scientific_hierarchy": _release_scientific_config(config),
        },
        "input_block_sha256": _input_block_digest(input_blocks),
        "input_array_sha256": (
            chromosome_evidence.digests if chromosome_evidence is not None else {
                "global_probs": _array_digest(probabilities),
                "global_sites": _array_digest(sites),
                "global_observed_mask": _array_digest(observed),
            }),
        "code_identity_sha256": stage2_release_code_identity(
            code_digest_overrides
        ),
        "hierarchy_model": {
            "input": "preprocess_result.prepared_blocks_original_layout",
            "founder_alleles": "shared_latent_with_explicit_missingness",
            "batch_sample_rule": INTERSECTION_SAMPLE_RULE,
            "linking": "distance_aware_double_hmm",
            "linker_fit": "coherent_expected_counts",
            "genotype_emission": "partial_founder_predictive_uniform_mixture",
            "max_linking_iterations": assembly_linking.MAX_LINKING_ITERATIONS,
            "observed_false_genotype_evidence": "uniform_state_neutral",
            "final_founder_refinement": "full_site_potts_phase_count_staggered_intervals",
        },
    }
    if chromosome_map is not None:
        record["config"]["scientific_hierarchy"]["recombination_rate"] = (
            chromosome_map.fallback_rate_per_bp)
        if chromosome_map.has_map:
            record["genetic_map"] = chromosome_map.identity()
    return _canonical_mapping(record, "stage2_release_identity")


def _expected_inference_snapshot(source: core_haplotypes.BlockResult) -> np.ndarray:
    result = np.asarray(_block_discrete(source), dtype=np.int8).copy()
    flags = getattr(source, "keep_flags", None)
    if flags is not None:
        kept = np.asarray(flags) > 0
        if kept.shape != (result.shape[1],):
            raise ValueError("keep_flags must match block positions")
        result[:, ~kept] = -1
    return result


def _freeze_inference_snapshots(blocks) -> None:
    for block in blocks:
        released = _block_discrete(block)
        snapshot = np.asarray(
            getattr(block, "missing_aware_inference_discrete_haps", None)
        )
        if (snapshot.shape != released.shape
                or np.any(~np.isin(snapshot, (-1, 0, 1)))):
            raise ValueError("block lost its founder-by-site inference snapshot")
        frozen = np.ascontiguousarray(snapshot, dtype=np.int8).copy()
        frozen.setflags(write=False)
        block.missing_aware_inference_discrete_haps = frozen


def _validate_preprocess_result(source, result) -> core_haplotypes.BlockResults:
    prepared = getattr(result, "prepared_blocks", None)
    direct = getattr(result, "direct_components", None)
    if not isinstance(prepared, core_haplotypes.BlockResults) or not prepared:
        raise TypeError("Stage-2 preprocessing must return prepared_blocks")
    if not isinstance(direct, core_haplotypes.BlockResults) or not direct:
        raise TypeError(
            "Stage-2 preprocessing must return diagnostic direct_components"
        )
    if _layout(prepared) != _layout(source):
        raise ValueError("preprocessing changed the original Stage-1 block layout")
    for raw, block in zip(source, prepared):
        snapshot = np.asarray(
            getattr(block, "missing_aware_inference_discrete_haps", None)
        )
        if not np.array_equal(snapshot, _expected_inference_snapshot(raw)):
            raise ValueError("preprocessing changed the frozen pre-fill snapshot")
        for side in ("before", "after"):
            for prefix in (
                    "missing_aware_break_", "missing_aware_break_reason_"):
                name = prefix + side
                if getattr(block, name, None) != getattr(raw, name, None):
                    raise ValueError(
                        "partial-link diagnostics may not create persistent "
                        "breaks on prepared hierarchy input"
                    )
    _freeze_inference_snapshots(prepared)
    return prepared


def _row_metadata(block, indices, row):
    keys = tuple(sorted(block.haplotypes))
    return (
        np.asarray(block.haplotypes[keys[row]])[indices],
        np.asarray(block.discrete_haps)[row, indices],
        np.asarray(block.missing_aware_inference_discrete_haps)[row, indices],
        None if getattr(block, "founder_alt_pseudo_probability", None) is None
        else np.asarray(block.founder_alt_pseudo_probability)[row, indices],
        None if getattr(block, "n_directional_site_supporters", None) is None
        else np.asarray(block.n_directional_site_supporters)[row, indices],
    )


def _metadata_equal(left, right) -> bool:
    for first, second in zip(left, right):
        if first is None or second is None:
            if first is not None or second is not None:
                return False
        elif np.issubdtype(np.asarray(first).dtype, np.inexact):
            if not np.allclose(
                    first, second, rtol=1e-6, atol=1e-7, equal_nan=True):
                return False
        elif not np.array_equal(first, second):
            return False
    return True


def _selected_path_snapshot(blocks):
    """Copy founder metadata that the hierarchy may mutate in place."""
    result = []
    for block in blocks:
        keys = tuple(sorted(block.haplotypes))
        result.append({
            "positions": np.asarray(block.positions).copy(),
            "haplotypes": tuple(
                np.asarray(block.haplotypes[key]).copy() for key in keys
            ),
            "discrete": np.asarray(block.discrete_haps).copy(),
            "inference": np.asarray(
                block.missing_aware_inference_discrete_haps
            ).copy(),
            "probability": None if getattr(
                block, "founder_alt_pseudo_probability", None
            ) is None else np.asarray(
                block.founder_alt_pseudo_probability
            ).copy(),
            "support": None if getattr(
                block, "n_directional_site_supporters", None
            ) is None else np.asarray(
                block.n_directional_site_supporters
            ).copy(),
        })
    return tuple(result)


def _validate_hierarchy_selected_paths(before, after) -> None:
    covered_sources = []
    for output in after:
        output_positions = np.asarray(output.positions)
        output_sources = 0
        for source_index, source in enumerate(before):
            source_positions = source["positions"]
            if (source_positions[0] < output_positions[0]
                    or source_positions[-1] > output_positions[-1]):
                continue
            indices = np.searchsorted(output_positions, source_positions)
            if (np.any(indices >= len(output_positions))
                    or not np.array_equal(
                        output_positions[indices], source_positions)):
                raise ValueError("hierarchy split or reordered an input component")
            covered_sources.append(source_index)
            output_sources += 1
            source_indices = np.arange(len(source_positions))
            candidates = [
                (
                    source["haplotypes"][row][source_indices],
                    source["discrete"][row, source_indices],
                    source["inference"][row, source_indices],
                    None if source["probability"] is None else
                    source["probability"][row, source_indices],
                    None if source["support"] is None else
                    source["support"][row, source_indices],
                )
                for row in range(len(source["haplotypes"]))
            ]
            for row in range(len(output.haplotypes)):
                observed = _row_metadata(output, indices, row)
                if not any(
                        _metadata_equal(observed, candidate)
                        for candidate in candidates):
                    raise ValueError(
                        "hierarchy produced noncommuting founder metadata"
                    )
        if output_sources == 0:
            raise ValueError("hierarchy split or invented an input component")
    if covered_sources != list(range(len(before))):
        raise ValueError("hierarchy split, duplicated, or reordered components")


def _persistent_breaks(blocks):
    result = []
    source = list(blocks)
    for left, right in zip(source, source[1:]):
        if not (
                getattr(left, "missing_aware_break_after", False)
                or getattr(right, "missing_aware_break_before", False)):
            continue
        reason = (
            getattr(left, "missing_aware_break_reason_after", None)
            or getattr(right, "missing_aware_break_reason_before", None)
        )
        result.append((
            int(np.asarray(left.positions)[-1]),
            int(np.asarray(right.positions)[0]),
            reason,
        ))
    return tuple(result)


def _canonicalize_persistent_breaks(blocks, boundaries) -> None:
    source = list(blocks)
    for left_end, right_start, reason in boundaries:
        left = next(
            block for block in source
            if int(np.asarray(block.positions)[-1]) == left_end
        )
        right = next(
            block for block in source
            if int(np.asarray(block.positions)[0]) == right_start
        )
        left.missing_aware_break_after = True
        right.missing_aware_break_before = True
        if reason is not None:
            left.missing_aware_break_reason_after = reason
            right.missing_aware_break_reason_before = reason


def _validate_persistent_breaks(blocks, boundaries) -> None:
    source = list(blocks)
    for left_end, right_start, reason in boundaries:
        if any(
                np.asarray(block.positions)[0] <= left_end
                and np.asarray(block.positions)[-1] >= right_start
                for block in source):
            raise ValueError("hierarchy crossed a persistent Stage-1 phase break")
        left = next((
            block for block in source
            if int(np.asarray(block.positions)[-1]) == left_end
        ), None)
        right = next((
            block for block in source
            if int(np.asarray(block.positions)[0]) == right_start
        ), None)
        if left is None or right is None:
            raise ValueError("hierarchy lost a persistent Stage-1 phase break")
        if not (
                getattr(left, "missing_aware_break_after", False)
                and getattr(right, "missing_aware_break_before", False)):
            raise ValueError("hierarchy lost persistent Stage-1 break markers")
        if reason is not None and not (
                getattr(left, "missing_aware_break_reason_after", None) == reason
                and getattr(right, "missing_aware_break_reason_before", None) == reason):
            raise ValueError("hierarchy changed a persistent Stage-1 break reason")


def _assert_position_coverage(blocks, expected_positions) -> None:
    if not blocks:
        raise ValueError("hierarchy produced no phase components")
    observed = np.concatenate([np.asarray(block.positions) for block in blocks])
    if not np.array_equal(observed, expected_positions):
        raise ValueError("hierarchy changed Stage-2 position coverage or order")


def _hierarchy_stop_reason(level: int, before: int, after: int) -> str | None:
    if after == 1:
        return "single_component_irreducible"
    if level >= 2 and after >= before:
        return "component_count_not_reduced"
    return None


def _preprocess_result_diagnostics_mode(preprocess_result) -> str:
    execution = getattr(preprocess_result, "execution", None)
    mode = getattr(execution, "diagnostics_mode", None)
    if mode not in {"compact", "full"}:
        raise ValueError(
            "preprocess result lacks a recognized diagnostics mode"
        )
    return mode


def assemble_chromosome(
    input_blocks,
    global_probs,
    global_sites,
    global_observed_mask,
    sample_ids,
    *,
    stage1_identity: Mapping[str, Any],
    config: AssemblyConfig = AssemblyConfig(),
    fold_assignments: np.ndarray | None = None,
    code_digest_overrides=None,
    release_checkpoints: assembly_checkpoints.AssemblyCheckpointIO | None = None,
    chromosome_map=None,
    chromosome_evidence=None,
) -> dict[str, Any]:
    """Preprocess, assemble the hierarchy, then refine final local row choices."""

    if not isinstance(config, AssemblyConfig):
        raise TypeError("config must be an AssemblyConfig")
    probabilities, sites, observed = _validate_global_inputs(
        input_blocks, global_probs, global_sites, global_observed_mask,
        chromosome_evidence,
    )
    ordered_ids = _canonical_sample_ids(sample_ids, probabilities.shape[0])
    source_layout = _layout(input_blocks)
    source_calls = tuple(
        _block_discrete(block).copy() for block in input_blocks
    )
    expected_positions = np.concatenate([
        np.asarray(block.positions).copy() for block in input_blocks
    ])
    persistent_breaks = _persistent_breaks(input_blocks)
    neutral_probs, evidence_by_block, observed_by_block = _preprocess_inputs(
        input_blocks, probabilities, sites, observed, chromosome_evidence
    )
    expected_preprocess_identity = (
        assembly_completion.stage2_preprocess_identity(
            config.preprocess_config,
            fold_assignments=fold_assignments,
            n_samples=probabilities.shape[0],
        ).record()
    )
    identity = stage2_release_identity_record(
        config,
        stage1_identity=stage1_identity,
        sample_ids=ordered_ids,
        input_blocks=input_blocks,
        global_probs=probabilities,
        global_sites=sites,
        global_observed_mask=observed,
        fold_assignments=fold_assignments,
        preprocess_identity=expected_preprocess_identity,
        code_digest_overrides=code_digest_overrides,
        chromosome_map=chromosome_map,
        chromosome_evidence=chromosome_evidence,
    )
    resumed_phases = []
    preprocess_checkpoint_upgraded = False
    if release_checkpoints is not None:
        release_checkpoints.bind(identity)
        preprocess_result = release_checkpoints.load("preprocess")
    else:
        preprocess_result = None
    preprocess_was_loaded = preprocess_result is not None
    if preprocess_was_loaded:
        observed_preprocess_identity = (
            preprocess_result.config_identity.record()
        )
        if observed_preprocess_identity != expected_preprocess_identity:
            raise RuntimeError(
                "preprocess checkpoint has the wrong scientific identity"
            )
        stored_diagnostics_mode = _preprocess_result_diagnostics_mode(
            preprocess_result
        )
        if (
                config.preprocess_diagnostics_mode == "full"
                and stored_diagnostics_mode == "compact"):
            preprocess_result = None
            preprocess_was_loaded = False
            preprocess_checkpoint_upgraded = True
        else:
            resumed_phases.append("preprocess")
    if preprocess_result is None:
        preprocess_result = assembly_completion.run_stage2_preprocess(
            input_blocks,
            evidence_by_block,
            observed_by_block,
            fold_assignments=fold_assignments,
            config=config.preprocess_config,
            num_processes=config.num_processes,
            diagnostics_mode=config.preprocess_diagnostics_mode,
        )
    effective_preprocess_diagnostics_mode = (
        _preprocess_result_diagnostics_mode(preprocess_result)
    )
    if (
            not preprocess_was_loaded
            and effective_preprocess_diagnostics_mode
            != config.preprocess_diagnostics_mode):
        raise RuntimeError(
            "fresh preprocess result has the wrong diagnostics mode"
        )
    observed_preprocess_identity = preprocess_result.config_identity.record()
    if release_checkpoints is not None:
        if observed_preprocess_identity != expected_preprocess_identity:
            raise RuntimeError(
                "preprocess checkpoint has the wrong scientific identity"
            )
    else:
        identity = stage2_release_identity_record(
            config,
            stage1_identity=stage1_identity,
            sample_ids=ordered_ids,
            input_blocks=input_blocks,
            global_probs=probabilities,
            global_sites=sites,
            global_observed_mask=observed,
            fold_assignments=fold_assignments,
            preprocess_identity=observed_preprocess_identity,
            code_digest_overrides=code_digest_overrides,
            chromosome_map=chromosome_map,
            chromosome_evidence=chromosome_evidence,
        )
    working = _validate_preprocess_result(input_blocks, preprocess_result)
    core_runtime.strip_block_evidence(working)
    core_runtime.strip_block_evidence(preprocess_result.direct_components)
    if release_checkpoints is not None and not preprocess_was_loaded:
        release_checkpoints.save("preprocess", preprocess_result)
    # Preprocessing has materialized every retained profile and diagnostic.
    # These block-local evidence arrays duplicate slices of ``neutral_probs``
    # and are not consumed by hierarchy or returned in the release product.
    del evidence_by_block, observed_by_block
    _canonicalize_persistent_breaks(working, persistent_breaks)

    # The hierarchy has always consumed float32 evidence. Cast once for all
    # levels instead of repeating a chromosome-sized cast at each level.
    hierarchy_probs = None
    refinement_context = None
    level_diagnostics = []
    stop_reason = "maximum_level_reached"
    for level in range(1, config.max_level + 1):
        before_count = len(working)
        batch_size = (
            config.l1_batch_size if level == 1
            else config.higher_level_batch_size
        )
        phase = f"hierarchy_l{level}"
        checkpoint_payload = (
            None if release_checkpoints is None
            else release_checkpoints.load(phase)
        )
        selected_path_input = _selected_path_snapshot(working)
        if checkpoint_payload is None:
            started = time.perf_counter()
            if hierarchy_probs is None:
                hierarchy_probs = assembly_evidence.float32_evidence(neutral_probs)
            output = assembly_hierarchy.run_hierarchical_step(
                working,
                neutral_probs,
                sites,
                scoring_probs=hierarchy_probs,
                batch_size=batch_size,
                recomb_rate=config.recombination_rate,
                chromosome_map=chromosome_map,
                beam_width=config.beam_width,
                max_founders=config.max_founders,
                max_sites_for_linking=config.max_sites_for_linking,
                n_generations=config.n_generations if level > 1 else None,
                recomb_tolerance=config.recombination_tolerance,
                top_n_swap=config.top_n_swap,
                max_cr_iterations=config.max_cr_iterations,
                paint_penalty=config.paint_penalty,
                min_hotspot_samples=config.min_hotspot_samples,
                cc_scale=config.cc_scale,
                num_processes=config.num_processes,
                maxtasksperchild=config.maxtasksperchild,
                min_gb_per_worker=config.min_gb_per_worker,
                verbose=config.verbose,
                min_boundary_informative_samples=(
                    config.min_boundary_informative_samples
                ),
                structured_transition_config=config.structured_transition_config,
                panel_search_config=config.panel_search_config,
            )
            elapsed_seconds = time.perf_counter() - started
            stored_diagnostic = None
        else:
            if (
                    not isinstance(checkpoint_payload, Mapping)
                    or checkpoint_payload.get("schema")
                    != "stage2-release-hierarchy-phase-v1"
                    or checkpoint_payload.get("level") != level):
                raise ValueError("unrecognized hierarchy-level checkpoint")
            output = checkpoint_payload.get("blocks")
            stored_diagnostic = checkpoint_payload.get("diagnostic")
            if not isinstance(stored_diagnostic, Mapping):
                raise TypeError("hierarchy checkpoint lacks its diagnostics")
            elapsed_seconds = stored_diagnostic.get("elapsed_seconds")
            resumed_phases.append(phase)
        if output is None:
            raise ValueError("hierarchy failed to produce phase components")
        if not isinstance(output, core_haplotypes.BlockResults):
            raise TypeError("hierarchy checkpoint must contain BlockResults")
        next_blocks = core_haplotypes.BlockResults(list(output))
        _assert_position_coverage(next_blocks, expected_positions)
        _validate_persistent_breaks(next_blocks, persistent_breaks)
        _validate_hierarchy_selected_paths(selected_path_input, next_blocks)
        _freeze_inference_snapshots(next_blocks)
        core_runtime.strip_block_evidence(next_blocks)
        after_count = len(next_blocks)
        stopped = _hierarchy_stop_reason(level, before_count, after_count)
        expected_diagnostic = {
            "level": level,
            "linking": "hmm",
            "linker_fit": "coherent_expected_counts",
            "genotype_emission": "partial_founder_predictive_uniform_mixture",
            "max_linking_iterations": assembly_linking.MAX_LINKING_ITERATIONS,
            "batch_size": batch_size,
            "component_count_before": before_count,
            "component_count_after": after_count,
            "component_count_reduced": after_count < before_count,
            "stop_reason": stopped,
        }
        if stored_diagnostic is not None:
            if not isinstance(stored_diagnostic, Mapping):
                raise TypeError("hierarchy checkpoint lacks its diagnostics")
            for name, expected in expected_diagnostic.items():
                if stored_diagnostic.get(name) != expected:
                    raise ValueError(
                        f"hierarchy checkpoint diagnostic mismatch: {name}"
                    )
            if (
                    isinstance(elapsed_seconds, bool)
                    or not isinstance(elapsed_seconds, (int, float))
                    or not math.isfinite(elapsed_seconds)
                    or elapsed_seconds < 0.0):
                raise ValueError(
                    "hierarchy checkpoint has invalid elapsed_seconds"
                )
            diagnostic = dict(stored_diagnostic)
        else:
            diagnostic = {
                **expected_diagnostic,
                "elapsed_seconds": elapsed_seconds,
            }
            if release_checkpoints is not None:
                release_checkpoints.save(phase, {
                    "schema": "stage2-release-hierarchy-phase-v1",
                    "level": level,
                    "blocks": next_blocks,
                    "diagnostic": diagnostic,
                })
        working = next_blocks
        if level == 1 and config.max_level == 4:
            refinement_context = next_blocks
        level_diagnostics.append(diagnostic)
        if stopped is not None:
            stop_reason = stopped
            break

    refinement_diagnostics = {"enabled": False, "components": []}
    # L1/L2 context passes must not feed chromosome-wide refinement back into
    # local discovery. Only a final release (including early-irreducible ones)
    # reopens the prepared local rows discarded by the hierarchy.
    if config.max_level == 4 and config.founder_refinement_config.enabled:
        phase = "founder_refinement"
        refined = (None if release_checkpoints is None
                   else release_checkpoints.load(phase))
        if refined is None:
            output, refinement_diagnostics = founder_refinement.refine_components(
                preprocess_result.prepared_blocks, working, neutral_probs, sites,
                config=config.founder_refinement_config,
                num_threads=config.num_processes, checkpoints=release_checkpoints,
                l1_blocks=refinement_context, cc_scale=config.cc_scale)
        else:
            output, refinement_diagnostics = refined["blocks"], refined["diagnostics"]
            resumed_phases.append(phase)
        _assert_position_coverage(output, expected_positions)
        _validate_persistent_breaks(output, persistent_breaks)
        _validate_hierarchy_selected_paths(
            _selected_path_snapshot(preprocess_result.prepared_blocks), output)
        _freeze_inference_snapshots(output)
        core_runtime.strip_block_evidence(output)
        working = output
        if refined is None and release_checkpoints is not None:
            release_checkpoints.save(phase, {
                "blocks": working, "diagnostics": refinement_diagnostics})

    for component_id, block in enumerate(working):
        block.missing_aware_phase_component_id = component_id
        block.stage2_component_identity = copy.deepcopy(identity)
    component_manifest = core_runtime.create_phase_component_manifest(working)
    core_runtime.validate_phase_component_manifest(component_manifest)

    if _layout(input_blocks) != source_layout or any(
            not np.array_equal(value, _block_discrete(block))
            for value, block in zip(source_calls, input_blocks)):
        raise AssertionError("the Stage-2 release mutated its Stage-1 inputs")

    return {
        "schema": STAGE2_RELEASE_SCHEMA,
        "identity": identity,
        "preprocess_result": preprocess_result,
        "components": working,
        "component_manifest": component_manifest,
        "level_diagnostics": tuple(level_diagnostics),
        "founder_refinement_diagnostics": refinement_diagnostics,
        "stop_reason": stop_reason,
        "release_metadata": {
                "hierarchy_input": "preprocess_result.prepared_blocks",
            "direct_components_role": "diagnostic_only",
            "direct_component_count": len(preprocess_result.direct_components),
            "execution_config": _release_execution_config(config),
            "requested_preprocess_diagnostics_mode": (
                config.preprocess_diagnostics_mode
            ),
            "effective_preprocess_diagnostics_mode": (
                effective_preprocess_diagnostics_mode
            ),
            "preprocess_checkpoint_upgraded": preprocess_checkpoint_upgraded,
            "resumed_phases": tuple(resumed_phases),
            "inference_snapshot": "immutable_pre_fill_stage1_calls",
            "batch_sample_rule": INTERSECTION_SAMPLE_RULE,
            "ordered_sample_ids": ordered_ids,
            "phase_core_ceiling": config.num_processes,
            "phases_overlap": False,
        },
    }

import haplotype_reconstruction.assembly.checkpoints as assembly_checkpoints
import haplotype_reconstruction.assembly.hierarchy as assembly_hierarchy
import haplotype_reconstruction.assembly.linking as assembly_linking
import haplotype_reconstruction.core.haplotypes as core_haplotypes
import haplotype_reconstruction.core.runtime as core_runtime
