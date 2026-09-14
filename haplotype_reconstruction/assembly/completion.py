"""assembly / completion for the canonical reconstruction pipeline."""
from __future__ import annotations
from haplotype_reconstruction import PACKAGE_ROOT

import atexit
import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import hashlib
import json

from typing import Any, Callable, Mapping, Sequence
import numpy as np
from numba import njit, prange, types
from numba.typed import List
import haplotype_reconstruction.assembly.boundaries as assembly_boundaries
import haplotype_reconstruction.assembly.joint_completion as assembly_joint_completion
import haplotype_reconstruction.assembly.observations as assembly_observations
import haplotype_reconstruction.assembly.occupancy as assembly_occupancy
import haplotype_reconstruction.assembly.partial_links as assembly_partial_links
import haplotype_reconstruction.core.haplotypes as core_haplotypes

STAGE2_PREPROCESS_SCHEMA = "stage2-missing-aware-preprocess-v2"


STAGE2_PREPROCESS_BACKEND = (
    "joint-crossfit-partial-link-cavity-variable-occupancy-v2"
)


STAGE2_PREPROCESS_SCIENTIFIC_DEPENDENCIES = tuple(sorted((
    'core/haplotypes.py',
    'assembly/components.py',
    'assembly/joint_completion.py',
    'assembly/joint_statistics.py',
    'assembly/allele_polynomials.py',
    'assembly/boundaries.py',
    'assembly/observations.py',
    'assembly/partial_links.py',
    'assembly/completion.py',
    'assembly/occupancy.py',
)))


def _scientific_dependency_digests() -> dict[str, str]:
    root = PACKAGE_ROOT
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in STAGE2_PREPROCESS_SCIENTIFIC_DEPENDENCIES
    }


def _joint_block_release_config() -> assembly_joint_completion.JointBlockConfig:
    return assembly_joint_completion.JointBlockConfig(
        max_unresolved_founders=6,
        minimum_call_probability=0.99,
    )


def _whole_bin_cavity_rule() -> assembly_boundaries.CavityFillRule:
    return assembly_boundaries.CavityFillRule(
        snps_per_bin=50,
        minimum_posterior_probability=0.99,
        maximum_enumerated_unresolved_founders=6,
    )


@dataclass(frozen=True)
class CompletionConfig:
    """Scientific settings for missing-aware founder completion before assembly."""

    joint_block_config: assembly_joint_completion.JointBlockConfig = field(
        default_factory=_joint_block_release_config
    )
    partial_link_config: assembly_partial_links.PartialLinkConfig = field(
        default_factory=assembly_partial_links.PartialLinkConfig
    )
    cavity_fill_rule: assembly_boundaries.CavityFillRule = field(
        default_factory=_whole_bin_cavity_rule
    )
    occupancy_rule: assembly_occupancy.CrossBlockOccupancyRule = field(
        default_factory=assembly_occupancy.CrossBlockOccupancyRule
    )
    maximum_stage1_wildcard_mass_for_fill_release: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.joint_block_config, assembly_joint_completion.JointBlockConfig):
            raise TypeError("joint_block_config must be a JointBlockConfig")
        if not isinstance(self.partial_link_config, assembly_partial_links.PartialLinkConfig):
            raise TypeError("partial_link_config must be a PartialLinkConfig")
        if not isinstance(self.cavity_fill_rule, assembly_boundaries.CavityFillRule):
            raise TypeError("cavity_fill_rule must be a CavityFillRule")
        if not isinstance(self.occupancy_rule, assembly_occupancy.CrossBlockOccupancyRule):
            raise TypeError("occupancy_rule must be a CrossBlockOccupancyRule")
        if self.joint_block_config.minimum_call_probability != 0.99:
            raise ValueError("Stage-2 joint-block release probability must be 0.99")
        if self.cavity_fill_rule.minimum_posterior_probability != 0.99:
            raise ValueError("Stage-2 cavity release probability must be 0.99")
        if self.cavity_fill_rule.maximum_enumerated_unresolved_founders != 6:
            raise ValueError("Stage-2 cavity unresolved-founder cap must be 6")
        threshold = self.maximum_stage1_wildcard_mass_for_fill_release
        if (
                isinstance(threshold, bool)
                or not np.isfinite(threshold)
                or threshold < 0.0
                or threshold > 1.0):
            raise ValueError(
                "maximum Stage-1 wildcard mass for fill release must be "
                "finite and in [0, 1]"
            )


@dataclass(frozen=True)
class FrozenPreprocessIdentity:
    """Canonical checkpoint-compatible scientific configuration identity."""

    canonical_json: str

    def record(self) -> dict[str, Any]:
        value = json.loads(self.canonical_json)
        if (
                value.get("schema") != STAGE2_PREPROCESS_SCHEMA
                or value.get("backend") != STAGE2_PREPROCESS_BACKEND):
            raise ValueError("Stage-2 preprocessing identity schema mismatch")
        return value


def stage2_preprocess_identity(
    config: CompletionConfig,
    *,
    fold_assignments: np.ndarray | None = None,
    n_samples: int | None = None,
) -> FrozenPreprocessIdentity:
    if not isinstance(config, CompletionConfig):
        raise TypeError("config must be a CompletionConfig")
    if fold_assignments is None:
        if n_samples is not None and (
                isinstance(n_samples, bool)
                or int(n_samples) != n_samples
                or n_samples < 1):
            raise ValueError("n_samples must be a positive integer")
        fold_semantics = {
            "mode": "deterministic_two_folds",
            "construction_rule": "deterministic_two_folds(n_samples, seed)",
            "deterministic_seed": int(
                config.joint_block_config.deterministic_seed
            ),
            "sample_count": None if n_samples is None else int(n_samples),
        }
        if n_samples is not None:
            ordered_folds = assembly_joint_completion.deterministic_two_folds(
                int(n_samples),
                config.joint_block_config.deterministic_seed,
            )
            fold_semantics["ordered_vector_sha256"] = hashlib.sha256(
                np.ascontiguousarray(ordered_folds, dtype=np.int8).tobytes()
            ).hexdigest()
    else:
        ordered_folds = np.asarray(fold_assignments, dtype=np.int8)
        if (
                ordered_folds.ndim != 1
                or len(ordered_folds) < 1
                or set(ordered_folds.tolist()) != {0, 1}):
            raise ValueError(
                "fold_assignments must contain ordered labels 0 and 1"
            )
        if n_samples is not None and int(n_samples) != len(ordered_folds):
            raise ValueError("n_samples does not match fold_assignments")
        fold_semantics = {
            "mode": "explicit_ordered_sample_axis",
            "sample_count": len(ordered_folds),
            "ordered_vector_sha256": hashlib.sha256(
                np.ascontiguousarray(ordered_folds, dtype=np.int8).tobytes()
            ).hexdigest(),
        }
    record = {
        "schema": STAGE2_PREPROCESS_SCHEMA,
        "backend": STAGE2_PREPROCESS_BACKEND,
        "scientific_config": asdict(config),
        "scientific_code_sha256": _scientific_dependency_digests(),
        "fold_semantics": fold_semantics,
    }
    return FrozenPreprocessIdentity(json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ))


@dataclass(frozen=True)
class BlockCavityGateDiagnostic:
    """Raw cavity posterior and occupancy release decision for one block."""

    candidate_fills: np.ndarray
    kept_fills: np.ndarray
    rejected_fills: np.ndarray
    posterior_alt: np.ndarray
    rejected_posterior_alt: np.ndarray


@dataclass(frozen=True)
class Stage1WildcardDiagnostic:
    """Validated Stage-1 wildcard evidence behind one block veto."""

    block_index: int
    metadata_present: bool
    observed_kept_depth_sample_count: int | None
    observed_wildcard_sample_count: int | None
    observed_wildcard_slot_count: int | None
    wildcard_mass: float | None
    maximum_mass_for_fill_release: float
    inference_eligible_unknown_founder_cell_count: int
    underfit: bool
    reason: str


@dataclass(frozen=True)
class CrossFitFoldDiagnostic:
    """Compact convergence and release audit for one crossfit fold."""

    fold: int
    training_sample_count: int
    heldout_sample_count: int
    best_start: int
    elbo: float
    converged_start_count: int
    start_count: int
    maximum_iterations: int
    minimum_elbo_increment: float
    released_cell_count: int


@dataclass(frozen=True)
class CompactCrossFitBlock:
    """Link/gate profiles plus bounded fold diagnostics."""

    profiles: assembly_joint_completion.CarrierProfiles
    folds: tuple[CrossFitFoldDiagnostic, ...]


@dataclass(frozen=True)
class PreprocessExecutionDiagnostics:
    """Observed process/thread and retained-result memory footprint."""

    requested_workers: int
    used_workers: int
    threads_per_worker: int
    backend: str
    diagnostics_mode: str
    shared_input_bytes: int
    retained_crossfit_array_bytes: int


@dataclass(frozen=True)
class Stage2PreprocessResult:
    """Typed output of truth-free Stage-2 preprocessing."""

    config_identity: FrozenPreprocessIdentity
    fold_assignments: np.ndarray
    raw_panels: tuple[assembly_observations.FounderPanel, ...]
    stage1_wildcard_diagnostics: tuple[Stage1WildcardDiagnostic, ...]
    crossfit_blocks: tuple[assembly_joint_completion.CrossFitBlock | CompactCrossFitBlock, ...]
    partial_links: tuple[assembly_partial_links.PartialBoundaryLink, ...]
    cavity_fills: tuple[assembly_boundaries.CavityFillResult, ...]
    occupancy_gate: assembly_occupancy.VariableKOccupancyGateResult
    cavity_gate_diagnostics: tuple[BlockCavityGateDiagnostic, ...]
    prepared_blocks: core_haplotypes.BlockResults
    direct_components: core_haplotypes.BlockResults
    execution: PreprocessExecutionDiagnostics

    @property
    def crossfit_profiles(self) -> tuple[assembly_joint_completion.CarrierProfiles, ...]:
        return tuple(value.profiles for value in self.crossfit_blocks)


def _readonly(array, dtype=None):
    value = np.array(array, dtype=dtype, order="C", copy=True)
    value.setflags(write=False)
    return value


def _freeze_founder_panel(block: core_haplotypes.BlockResult) -> assembly_observations.FounderPanel:
    panel = assembly_observations.founder_panel_from_block_result(block)
    keep_flags = getattr(block, "keep_flags", None)
    kept = (
        np.ones(len(panel.positions), dtype=np.bool_)
        if keep_flags is None
        else np.asarray(keep_flags) > 0
    )
    if kept.shape != (len(panel.positions),):
        raise ValueError("keep_flags must match founder-panel positions")
    q = panel.q.copy()
    called = panel.called.copy()
    q[:, ~kept] = 0.5
    called[:, ~kept] = False
    support = None if panel.support is None else panel.support.copy()
    if support is not None:
        support[:, ~kept] = 0.0
    result = assembly_observations.FounderPanel(
        positions=panel.positions,
        keys=panel.keys,
        q=q,
        called=called,
        support=support,
    )
    result.positions.setflags(write=False)
    result.q.setflags(write=False)
    result.called.setflags(write=False)
    if result.support is not None:
        result.support.setflags(write=False)
    return result


def _hard_panel(panel: assembly_observations.FounderPanel) -> assembly_joint_completion.HardFounderPanel:
    alleles = np.full(panel.q.shape, -1, dtype=np.int8)
    alleles[panel.called] = panel.q[panel.called].astype(np.int8)
    result = assembly_joint_completion.HardFounderPanel(panel.positions, panel.keys, alleles)
    result.positions.setflags(write=False)
    result.alleles.setflags(write=False)
    return result


@njit(cache=True, parallel=True, fastmath=False)
def _copy_validated_evidence(source, destination):
    """Own the input snapshot and check finite/nonnegative GLs in one scan."""
    samples = source[0].shape[0]
    valid = np.ones(len(source) * samples, np.bool_)
    for task in prange(len(source) * samples):
        block, sample = task // samples, task % samples
        values, copied = source[block], destination[block]
        for site in range(values.shape[1]):
            for genotype in range(3):
                value = values[sample, site, genotype]
                copied[sample, site, genotype] = value
                if not np.isfinite(value) or value < 0.0:
                    valid[task] = False
    return valid


def _validate_and_clone_inputs(
    raw_blocks: core_haplotypes.BlockResults,
    genotype_evidence: Sequence[np.ndarray],
    observed_masks: Sequence[np.ndarray],
    num_threads=1,
):
    if not isinstance(raw_blocks, core_haplotypes.BlockResults):
        raise TypeError("raw_blocks must be a BlockResults")
    source = tuple(raw_blocks)
    if not source:
        raise ValueError("at least one raw Stage-2 block is required")
    evidence_values = tuple(genotype_evidence)
    observed_values = tuple(observed_masks)
    if (
            len(evidence_values) != len(source)
            or len(observed_values) != len(source)):
        raise ValueError(
            "blocks, per-block genotype evidence, and observed masks must align"
        )

    cloned = copy.deepcopy(raw_blocks)
    blocks = tuple(cloned)
    evidence = []
    source_evidence = List.empty_list(types.Array(types.float64, 3, "A", readonly=True))
    observed = []
    n_samples = None
    previous_end = None
    for index, (block, block_evidence, block_observed) in enumerate(
            zip(blocks, evidence_values, observed_values)):
        positions = np.asarray(block.positions)
        keys = tuple(sorted(block.haplotypes))
        discrete = np.asarray(getattr(block, "discrete_haps", None))
        if positions.ndim != 1 or len(positions) == 0:
            raise ValueError(f"block {index} must contain positions")
        if len(positions) > 1 and np.any(positions[1:] <= positions[:-1]):
            raise ValueError(f"block {index} positions must be strictly increasing")
        if previous_end is not None and positions[0] <= previous_end:
            raise ValueError("raw blocks must be ordered and non-overlapping")
        if not keys:
            raise ValueError(f"block {index} must contain founders")
        if discrete.shape != (len(keys), len(positions)):
            raise ValueError(
                f"block {index} discrete_haps must have shape (founders, sites)"
            )
        if np.any(~np.isin(discrete, (-1, 0, 1))):
            raise ValueError("discrete_haps must contain only -1, 0, or 1")
        keep_flags = getattr(block, "keep_flags", None)
        kept_sites = (
            np.ones(len(positions), dtype=np.bool_)
            if keep_flags is None
            else np.asarray(keep_flags) > 0
        )
        if kept_sites.shape != (len(positions),):
            raise ValueError(
                f"block {index} keep_flags must match positions"
            )

        source_values = np.asarray(block_evidence)
        owns_conversion = source_values.dtype != np.float64
        values = (np.asarray(source_values, dtype=np.float64, order="C")
                  if owns_conversion else source_values)
        mask = np.asarray(block_observed, dtype=np.bool_).copy()
        if values.ndim != 3 or values.shape[1:] != (len(positions), 3):
            raise ValueError(
                f"block {index} evidence must have shape (samples, sites, 3)"
            )
        if mask.shape != values.shape[:2]:
            raise ValueError(
                f"block {index} observed mask must have shape (samples, sites)"
            )
        mask &= kept_sites[None, :]
        if n_samples is None:
            n_samples = values.shape[0]
        elif values.shape[0] != n_samples:
            raise ValueError("all blocks must share the same sample axis")

        inference_discrete = np.ascontiguousarray(discrete, dtype=np.int8).copy()
        inference_discrete[:, ~kept_sites] = -1
        snapshot = _readonly(inference_discrete, np.int8)
        block.discrete_haps = np.ascontiguousarray(discrete, dtype=np.int8).copy()
        block.missing_aware_inference_discrete_haps = snapshot
        frozen_values = values.view()
        frozen_values.setflags(write=False)
        source_evidence.append(frozen_values)
        # A dtype conversion already owns its snapshot. Do not retain a
        # second chromosome-sized float64 copy for float32 callers.
        evidence.append(values if owns_conversion
                        else np.empty(values.shape, dtype=np.float64))
        mask.setflags(write=False)
        observed.append(mask)
        previous_end = positions[-1]

    assert n_samples is not None
    with core_parallel.numba_thread_scope(num_threads):
        valid = _copy_validated_evidence(source_evidence, List(evidence))
    if not np.all(valid):
        raise ValueError("genotype evidence must be finite and non-negative")
    for value in evidence:
        value.setflags(write=False)
    return cloned, tuple(evidence), tuple(observed), int(n_samples)


def _validated_wildcard_mass(value, block_index: int) -> float:
    mass_array = np.asarray(value, dtype=np.float64)
    if mass_array.shape != ():
        raise ValueError(f"block {block_index} wildcard_mass must be scalar")
    mass = float(mass_array)
    if not np.isfinite(mass) or not 0.0 <= mass <= 1.0:
        raise ValueError(
            f"block {block_index} wildcard_mass must be finite and in [0, 1]"
        )
    return mass


def _stage1_underfit_flags(
    blocks: Sequence[core_haplotypes.BlockResult],
    n_samples: int,
    maximum_mass_for_fill_release: float,
) -> tuple[tuple[bool, ...], tuple[Stage1WildcardDiagnostic, ...]]:
    """Validate Stage-1 wildcard provenance and flag fitted-panel underfit."""

    flags = []
    diagnostics = []
    for index, block in enumerate(blocks):
        discrete = np.asarray(block.discrete_haps, dtype=np.int8)
        keep_flags = getattr(block, "keep_flags", None)
        kept = (
            np.ones(discrete.shape[1], dtype=np.bool_)
            if keep_flags is None
            else np.asarray(keep_flags) > 0
        )
        inference_eligible_unknown_count = int(np.count_nonzero(
            (discrete < 0) & kept[None, :]
        ))
        depth_value = getattr(
            block, "sample_has_observed_kept_depth", None
        )
        slots_value = getattr(block, "wildcard_slots", None)
        mass_value = getattr(block, "wildcard_mass", None)
        has_depth = depth_value is not None
        has_slots = slots_value is not None
        has_mass = mass_value is not None
        if not (has_depth or has_slots):
            if not has_mass:
                underfit = inference_eligible_unknown_count > 0
                flags.append(underfit)
                diagnostics.append(Stage1WildcardDiagnostic(
                    block_index=index,
                    metadata_present=False,
                    observed_kept_depth_sample_count=None,
                    observed_wildcard_sample_count=None,
                    observed_wildcard_slot_count=None,
                    wildcard_mass=None,
                    maximum_mass_for_fill_release=(
                        maximum_mass_for_fill_release
                    ),
                    inference_eligible_unknown_founder_cell_count=(
                        inference_eligible_unknown_count
                    ),
                    underfit=underfit,
                    reason=(
                        "stage1_wildcard_metadata_absent_with_unknown_"
                        "founder_cells"
                        if underfit
                        else "stage1_wildcard_metadata_absent_no_unknown_"
                        "founder_cells"
                    ),
                ))
                continue
            summary_mass = _validated_wildcard_mass(mass_value, index)
            underfit = summary_mass > maximum_mass_for_fill_release
            flags.append(underfit)
            diagnostics.append(Stage1WildcardDiagnostic(
                block_index=index,
                metadata_present=True,
                observed_kept_depth_sample_count=None,
                observed_wildcard_sample_count=None,
                observed_wildcard_slot_count=None,
                wildcard_mass=summary_mass,
                maximum_mass_for_fill_release=(
                    maximum_mass_for_fill_release
                ),
                inference_eligible_unknown_founder_cell_count=(
                    inference_eligible_unknown_count
                ),
                underfit=underfit,
                reason="stage1_wildcard_mass_summary_only",
            ))
            continue
        if not (has_depth and has_slots):
            raise ValueError(
                f"block {index} has incomplete Stage-1 wildcard metadata"
            )

        raw_depth = np.asarray(depth_value)
        raw_slots = np.asarray(slots_value)
        if raw_depth.shape != (n_samples,):
            raise ValueError(
                f"block {index} sample_has_observed_kept_depth must "
                "match samples"
            )
        if raw_slots.shape != (n_samples,):
            raise ValueError(
                f"block {index} wildcard_slots must match samples"
            )
        if np.any(~np.isin(raw_depth, (False, True))):
            raise ValueError(
                "sample_has_observed_kept_depth must be boolean"
            )
        slots_float = np.asarray(raw_slots, dtype=np.float64)
        if (
                np.any(~np.isfinite(slots_float))
                or np.any(slots_float != np.floor(slots_float))
                or np.any((slots_float < 0.0) | (slots_float > 2.0))):
            raise ValueError(
                "wildcard_slots must contain integer strand counts in [0, 2]"
            )
        depth = np.asarray(raw_depth, dtype=np.bool_)
        slots = np.asarray(slots_float, dtype=np.int64)
        observed_wildcard = depth & (slots > 0)
        expected_mass = float(
            np.sum(slots[depth], dtype=np.float64)
            / max(2 * int(np.sum(depth)), 1)
        )
        if has_mass:
            mass = _validated_wildcard_mass(mass_value, index)
            if not np.isclose(
                    mass, expected_mass, rtol=0.0,
                    atol=np.finfo(np.float64).eps):
                raise ValueError(
                    f"block {index} wildcard_mass disagrees with "
                    "wildcard_slots"
                )

        underfit = bool(
            np.any(observed_wildcard)
            and expected_mass > maximum_mass_for_fill_release
        )
        if underfit:
            reason = "stage1_wildcard_panel_underfit"
        elif np.any(observed_wildcard):
            reason = "stage1_wildcard_mass_within_configured_threshold"
        else:
            reason = "no_observed_kept_depth_wildcards"
        flags.append(underfit)
        diagnostics.append(Stage1WildcardDiagnostic(
            block_index=index,
            metadata_present=True,
            observed_kept_depth_sample_count=int(np.sum(depth)),
            observed_wildcard_sample_count=int(
                np.sum(observed_wildcard)
            ),
            observed_wildcard_slot_count=int(
                np.sum(slots[depth], dtype=np.int64)
            ),
            wildcard_mass=expected_mass,
            maximum_mass_for_fill_release=maximum_mass_for_fill_release,
            inference_eligible_unknown_founder_cell_count=(
                inference_eligible_unknown_count
            ),
            underfit=underfit,
            reason=reason,
        ))
    return tuple(flags), tuple(diagnostics)


def _validated_folds(
    n_samples: int,
    config: CompletionConfig,
    fold_assignments,
) -> np.ndarray:
    if fold_assignments is None:
        folds = assembly_joint_completion.deterministic_two_folds(
            n_samples, config.joint_block_config.deterministic_seed
        )
    else:
        folds = np.asarray(fold_assignments, dtype=np.int8)
        if folds.shape != (n_samples,) or set(folds.tolist()) != {0, 1}:
            raise ValueError(
                "fold_assignments must contain deterministic labels 0 and 1"
            )
    return _readonly(folds, np.int8)


def _compact_crossfit_block(value: assembly_joint_completion.CrossFitBlock) -> CompactCrossFitBlock:
    diagnostics = []
    for fold in value.folds:
        starts = tuple(fold.fit.starts)
        diagnostics.append(CrossFitFoldDiagnostic(
            fold=int(fold.fold),
            training_sample_count=len(fold.training_indices),
            heldout_sample_count=len(fold.heldout_indices),
            best_start=int(fold.fit.best_start),
            elbo=float(fold.fit.elbo),
            converged_start_count=sum(
                int(start.converged) for start in starts
            ),
            start_count=len(starts),
            maximum_iterations=max(
                (int(start.iterations) for start in starts), default=0
            ),
            minimum_elbo_increment=min(
                (
                    float(start.minimum_elbo_increment)
                    for start in starts
                ),
                default=float("nan"),
            ),
            released_cell_count=int(np.sum(fold.released.released)),
        ))
    return CompactCrossFitBlock(value.profiles, tuple(diagnostics))


def _retain_crossfit(value: assembly_joint_completion.CrossFitBlock, diagnostics_mode: str):
    if not isinstance(value, assembly_joint_completion.CrossFitBlock):
        raise TypeError("crossfit function must return CrossFitBlock values")
    return (
        value
        if diagnostics_mode == "full"
        else _compact_crossfit_block(value)
    )


def _retained_array_bytes(values) -> int:
    """Estimate resident NumPy payload bytes without double-counting aliases."""

    seen_objects = set()
    seen_arrays = set()

    def visit(value):
        object_id = id(value)
        if isinstance(value, np.ndarray):
            if object_id in seen_arrays:
                return 0
            seen_arrays.add(object_id)
            return int(value.nbytes)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)
        if is_dataclass(value):
            return sum(visit(getattr(value, item.name)) for item in fields(value))
        if isinstance(value, Mapping):
            return sum(visit(key) + visit(item) for key, item in value.items())
        if isinstance(value, (tuple, list)):
            return sum(visit(item) for item in value)
        return 0

    return visit(values)


_PREPROCESS_WORKER_EVIDENCE = None


_PREPROCESS_WORKER_OBSERVED = None


_PREPROCESS_WORKER_OFFSETS = None


_PREPROCESS_WORKER_FOLDS = None


_PREPROCESS_WORKER_JOINT_CONFIG = None


_PREPROCESS_WORKER_CAVITY_RULE = None


_PREPROCESS_WORKER_DIAGNOSTICS_MODE = None


_PREPROCESS_WORKER_HANDLES = []


def _close_preprocess_worker_shared_memory() -> None:
    global _PREPROCESS_WORKER_HANDLES
    core_parallel.close_shared_memory(_PREPROCESS_WORKER_HANDLES)
    _PREPROCESS_WORKER_HANDLES = []


def _initialize_preprocess_worker(
    evidence_metadata,
    observed_metadata,
    offsets,
    folds,
    joint_config,
    cavity_rule,
    diagnostics_mode,
) -> None:
    """Attach shared read-only inputs and enforce one numerical thread."""

    core_environment.force_single_threaded_numeric_libraries()
    import numba

    numba.set_num_threads(1)
    _close_preprocess_worker_shared_memory()
    handles = []
    try:
        evidence_handle, evidence = core_parallel.attach_shared_array(evidence_metadata)
        handles.append(evidence_handle)
        observed_handle, observed = core_parallel.attach_shared_array(observed_metadata)
        handles.append(observed_handle)
    except BaseException:
        core_parallel.close_shared_memory(handles)
        raise

    global _PREPROCESS_WORKER_EVIDENCE
    global _PREPROCESS_WORKER_OBSERVED
    global _PREPROCESS_WORKER_OFFSETS
    global _PREPROCESS_WORKER_FOLDS
    global _PREPROCESS_WORKER_JOINT_CONFIG
    global _PREPROCESS_WORKER_CAVITY_RULE
    global _PREPROCESS_WORKER_DIAGNOSTICS_MODE
    global _PREPROCESS_WORKER_HANDLES
    _PREPROCESS_WORKER_EVIDENCE = evidence
    _PREPROCESS_WORKER_OBSERVED = observed
    _PREPROCESS_WORKER_OFFSETS = np.asarray(offsets, dtype=np.int64)
    _PREPROCESS_WORKER_FOLDS = np.asarray(folds, dtype=np.int8)
    _PREPROCESS_WORKER_JOINT_CONFIG = joint_config
    _PREPROCESS_WORKER_CAVITY_RULE = cavity_rule
    _PREPROCESS_WORKER_DIAGNOSTICS_MODE = diagnostics_mode
    _PREPROCESS_WORKER_HANDLES = handles
    atexit.register(_close_preprocess_worker_shared_memory)


def _preprocess_worker_arrays(block_index: int):
    if (
            _PREPROCESS_WORKER_EVIDENCE is None
            or _PREPROCESS_WORKER_OBSERVED is None
            or _PREPROCESS_WORKER_OFFSETS is None):
        raise RuntimeError("Stage-2 preprocess worker is not initialized")
    start = int(_PREPROCESS_WORKER_OFFSETS[block_index])
    stop = int(_PREPROCESS_WORKER_OFFSETS[block_index + 1])
    evidence = np.ascontiguousarray(
        _PREPROCESS_WORKER_EVIDENCE[:, start:stop, :]
    )
    observed = np.ascontiguousarray(
        _PREPROCESS_WORKER_OBSERVED[:, start:stop]
    )
    return evidence, observed


def _preprocess_crossfit_worker(task):
    block_index, panel = task
    evidence, observed = _preprocess_worker_arrays(block_index)
    value = assembly_joint_completion.crossfit_block(
        panel,
        evidence,
        observed,
        fold_assignments=_PREPROCESS_WORKER_FOLDS,
        config=_PREPROCESS_WORKER_JOINT_CONFIG,
    )
    return (
        block_index,
        _retain_crossfit(value, _PREPROCESS_WORKER_DIAGNOSTICS_MODE),
    )


def _preprocess_cavity_worker(task):
    block_index, panel = task
    evidence, observed = _preprocess_worker_arrays(block_index)
    value = assembly_boundaries.cavity_fill_unknown_alleles(
        panel,
        evidence,
        observed,
        _PREPROCESS_WORKER_CAVITY_RULE,
    )
    return block_index, value


def _preprocess_partial_link_worker(task):
    """Independent adjacent boundaries share the already-running block pool."""
    index, left_panel, right_panel, left, right, config = task
    return index, assembly_partial_links.link_partial_profiles(
        left_panel, right_panel, left, right,
        fold_assignments=_PREPROCESS_WORKER_FOLDS, config=config)


def _ordered_worker_results(records, count, value_name):
    output = [None] * count
    for index, value in records:
        if not 0 <= index < count or output[index] is not None:
            raise RuntimeError(
                f"parallel {value_name} returned an invalid block index"
            )
        output[index] = value
    if any(value is None for value in output):
        raise RuntimeError(f"parallel {value_name} omitted a block")
    return tuple(output)


def _infer_partial_links(
    hard_panels,
    crossfit_blocks,
    folds,
    config,
    partial_link_fn,
):
    links = tuple(
        partial_link_fn(
            hard_panels[index],
            hard_panels[index + 1],
            crossfit_blocks[index].profiles,
            crossfit_blocks[index + 1].profiles,
            fold_assignments=folds,
            config=config.partial_link_config,
        )
        for index in range(len(hard_panels) - 1)
    )
    if any(not isinstance(value, assembly_partial_links.PartialBoundaryLink) for value in links):
        raise TypeError(
            "partial-link function must return PartialBoundaryLink values"
        )
    return links


def _run_sequential_block_science(
    raw_panels,
    hard_panels,
    evidence,
    observed,
    folds,
    config,
    diagnostics_mode,
    crossfit_fn,
    cavity_fn,
    partial_link_fn,
):
    crossfit_blocks = tuple(
        _retain_crossfit(
            crossfit_fn(
                panel,
                block_evidence,
                block_observed,
                fold_assignments=folds,
                config=config.joint_block_config,
            ),
            diagnostics_mode,
        )
        for panel, block_evidence, block_observed in zip(
            hard_panels, evidence, observed
        )
    )
    links = _infer_partial_links(
        hard_panels, crossfit_blocks, folds, config, partial_link_fn
    )
    cavity_fills = tuple(
        cavity_fn(
            panel,
            block_evidence,
            block_observed,
            config.cavity_fill_rule,
        )
        for panel, block_evidence, block_observed in zip(
            raw_panels, evidence, observed
        )
    )
    return crossfit_blocks, links, cavity_fills


@njit(cache=True, parallel=True)
def _pack_preprocess_arrays(evidence, observed, offsets):
    """Copy independent block/sample slabs into the worker transport layout."""
    samples = evidence[0].shape[0]
    blocks = len(evidence)
    values = np.empty((samples, offsets[-1], 3), np.float64)
    masks = np.empty((samples, offsets[-1]), np.bool_)
    for task in prange(samples * blocks):
        sample, block = task // blocks, task % blocks
        start, stop = offsets[block], offsets[block + 1]
        values[sample, start:stop] = evidence[block][sample]
        masks[sample, start:stop] = observed[block][sample]
    return values, masks


def _run_parallel_block_science(
    raw_panels,
    hard_panels,
    evidence,
    observed,
    folds,
    config,
    diagnostics_mode,
    worker_count,
    partial_link_fn,
):
    offsets = np.zeros(len(evidence) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(
        [value.shape[1] for value in evidence], dtype=np.int64
    )
    with core_parallel.numba_thread_scope(worker_count):
        combined_evidence, combined_observed = _pack_preprocess_arrays(
            List(evidence), List(observed), offsets)
    shared_input_bytes = (
        int(combined_evidence.nbytes) + int(combined_observed.nbytes)
    )
    handles = []
    try:
        evidence_handle, evidence_metadata = core_parallel.create_shared_array(
            combined_evidence, copy_threads=worker_count
        )
        handles.append(evidence_handle)
        observed_handle, observed_metadata = core_parallel.create_shared_array(
            combined_observed, copy_threads=worker_count
        )
        handles.append(observed_handle)
        del combined_evidence, combined_observed

        with core_parallel.safe_forkserver_pool(
            worker_count,
            initializer=_initialize_preprocess_worker,
            initargs=(
                evidence_metadata,
                observed_metadata,
                offsets,
                folds,
                config.joint_block_config,
                config.cavity_fill_rule,
                diagnostics_mode,
            ),
        ) as pool:
            crossfit_blocks = _ordered_worker_results(
                pool.imap_unordered(
                    _preprocess_crossfit_worker,
                    tuple(enumerate(hard_panels)),
                    chunksize=1,
                ),
                len(hard_panels),
                "crossfit",
            )
            if partial_link_fn is assembly_partial_links.link_partial_profiles:
                # Boundaries depend only on the adjacent completed crossfits,
                # never on another boundary's decision. Preserve result order.
                links = _ordered_worker_results(pool.imap_unordered(
                    _preprocess_partial_link_worker,
                    ((index, hard_panels[index], hard_panels[index + 1],
                      crossfit_blocks[index].profiles,
                      crossfit_blocks[index + 1].profiles,
                      config.partial_link_config)
                     for index in range(len(hard_panels) - 1)),
                    chunksize=4), len(hard_panels) - 1, "partial links")
            else:
                links = _infer_partial_links(
                    hard_panels, crossfit_blocks, folds, config, partial_link_fn)
            cavity_fills = _ordered_worker_results(
                pool.imap_unordered(
                    _preprocess_cavity_worker,
                    tuple(enumerate(raw_panels)),
                    chunksize=1,
                ),
                len(raw_panels),
                "cavity",
            )
    finally:
        core_parallel.close_shared_memory(handles, unlink=True)

    return crossfit_blocks, links, cavity_fills, shared_input_bytes


def _validate_cavity_fill(
    fill: assembly_boundaries.CavityFillResult,
    panel: assembly_observations.FounderPanel,
    block: core_haplotypes.BlockResult,
    block_index: int,
) -> None:
    if not isinstance(fill, assembly_boundaries.CavityFillResult):
        raise TypeError("cavity function must return a CavityFillResult")
    shape = panel.q.shape
    for name in ("filled", "posterior_alt", "effective_supporters"):
        if np.asarray(getattr(fill, name)).shape != shape:
            raise ValueError(
                f"block {block_index} cavity {name} is not panel-aligned"
            )
    candidate = np.asarray(fill.filled, dtype=np.bool_)
    if np.any(candidate & panel.called):
        raise ValueError("cavity fills may not replace original called alleles")
    keep_flags = getattr(block, "keep_flags", None)
    kept = (
        np.ones(panel.q.shape[1], dtype=np.bool_)
        if keep_flags is None
        else np.asarray(keep_flags) > 0
    )
    if kept.shape != (panel.q.shape[1],):
        raise ValueError("keep_flags must match cavity-panel positions")
    if np.any(candidate[:, ~kept]):
        raise ValueError("cavity fills may not use keep_flags=False sites")
    posterior = np.asarray(fill.posterior_alt, dtype=np.float64)
    if np.any(~np.isfinite(posterior)) or np.any(
            (posterior < 0.0) | (posterior > 1.0)):
        raise ValueError("cavity posterior_alt must lie in [0, 1]")


def _cavity_gate_diagnostics(
    cavity_fills: Sequence[assembly_boundaries.CavityFillResult],
    gate: assembly_occupancy.VariableKOccupancyGateResult,
) -> tuple[BlockCavityGateDiagnostic, ...]:
    diagnostics = []
    for fill, kept_value in zip(cavity_fills, gate.kept_fills):
        candidate = np.asarray(fill.filled, dtype=np.bool_)
        kept = np.asarray(kept_value, dtype=np.bool_)
        if kept.shape != candidate.shape or np.any(kept & ~candidate):
            raise AssertionError("occupancy gate released a non-cavity fill")
        rejected = candidate & ~kept
        posterior = np.asarray(fill.posterior_alt, dtype=np.float64)
        rejected_posterior = np.full(posterior.shape, np.nan, dtype=np.float64)
        rejected_posterior[rejected] = posterior[rejected]
        diagnostics.append(BlockCavityGateDiagnostic(
            candidate_fills=_readonly(candidate, np.bool_),
            kept_fills=_readonly(kept, np.bool_),
            rejected_fills=_readonly(rejected, np.bool_),
            posterior_alt=_readonly(posterior, np.float64),
            rejected_posterior_alt=_readonly(
                rejected_posterior, np.float64
            ),
        ))
    return tuple(diagnostics)


def _materialize_kept_fills(
    block: core_haplotypes.BlockResult,
    raw_panel: assembly_observations.FounderPanel,
    fill: assembly_boundaries.CavityFillResult,
    diagnostic: BlockCavityGateDiagnostic,
) -> None:
    raw_discrete = np.asarray(
        getattr(block, "discrete_haps", None), dtype=np.int8
    )
    if raw_discrete.shape != raw_panel.q.shape:
        raise ValueError("public discrete_haps are not founder-by-site aligned")
    raw_discrete = raw_discrete.copy()
    original_public_called = raw_discrete >= 0
    kept = np.asarray(diagnostic.kept_fills, dtype=np.bool_)
    posterior = np.asarray(fill.posterior_alt, dtype=np.float64)

    discrete = raw_discrete.copy()
    discrete[kept] = (posterior[kept] > 0.5).astype(np.int8)
    if not np.array_equal(
            discrete[original_public_called],
            raw_discrete[original_public_called]):
        raise AssertionError("materialization changed an original called allele")
    if not np.array_equal(
            (discrete >= 0) & ~original_public_called, kept):
        raise AssertionError("materialization released cells outside the gate")

    keys = tuple(sorted(block.haplotypes))
    if keys != raw_panel.keys:
        raise ValueError("public haplotype keys disagree with the raw panel")
    public_haplotypes = {}
    for row, key in enumerate(keys):
        values = np.asarray(block.haplotypes[key])
        if values.shape == (raw_panel.q.shape[1],):
            updated = values.copy()
            updated[kept[row]] = discrete[row, kept[row]]
        elif values.shape == (raw_panel.q.shape[1], 2):
            updated = values.copy()
            if np.issubdtype(updated.dtype, np.floating):
                updated[kept[row], 0] = 1.0 - posterior[row, kept[row]]
                updated[kept[row], 1] = posterior[row, kept[row]]
            else:
                updated[kept[row]] = 0
                updated[kept[row], discrete[row, kept[row]]] = 1
        else:
            raise ValueError("public haplotypes are not founder-by-site aligned")
        public_haplotypes[key] = np.ascontiguousarray(updated)

    original_probability = getattr(
        block, "founder_alt_pseudo_probability", None
    )
    public_probability = np.asarray(original_probability)
    if public_probability.shape != raw_panel.q.shape:
        raise ValueError(
            "founder_alt_pseudo_probability is not founder-by-site aligned"
        )
    if not np.issubdtype(public_probability.dtype, np.floating):
        raise TypeError("founder_alt_pseudo_probability must be floating point")
    public_probability = public_probability.copy()
    public_probability[kept] = posterior[kept]

    block.discrete_haps = np.ascontiguousarray(discrete)
    block.haplotypes = public_haplotypes
    block.founder_alt_pseudo_probability = np.ascontiguousarray(
        public_probability
    )
    block.cavity_candidate_filled = diagnostic.candidate_fills
    block.cavity_kept_filled = diagnostic.kept_fills
    block.cavity_rejected_filled = diagnostic.rejected_fills
    block.cavity_posterior_alt = diagnostic.posterior_alt
    block.cavity_rejected_posterior_alt = diagnostic.rejected_posterior_alt
    block.cavity_effective_supporters = _readonly(
        fill.effective_supporters, np.float64
    )
    block.genotype_evidence_mode = "stage2_missing_aware_preprocessed"


def run_stage2_preprocess(
    raw_blocks: core_haplotypes.BlockResults,
    genotype_evidence: Sequence[np.ndarray],
    observed_masks: Sequence[np.ndarray],
    *,
    fold_assignments: np.ndarray | None = None,
    config: CompletionConfig = CompletionConfig(),
    num_processes: int = 1,
    diagnostics_mode: str = "full",
    crossfit_fn: Callable[..., assembly_joint_completion.CrossFitBlock] = assembly_joint_completion.crossfit_block,
    cavity_fn: Callable[..., assembly_boundaries.CavityFillResult] = assembly_boundaries.cavity_fill_unknown_alleles,
    partial_link_fn: Callable[..., assembly_partial_links.PartialBoundaryLink] = assembly_partial_links.link_partial_profiles,
) -> Stage2PreprocessResult:
    """Run the truth-free Stage-2 reference or parallel production path.

    Process parallelism is limited to independent block computations.  Every
    worker is restricted to one numerical thread, so the aggregate compute
    budget is used-workers times one.
    """

    if not isinstance(config, CompletionConfig):
        raise TypeError("config must be a CompletionConfig")
    prepared, evidence, observed, n_samples = _validate_and_clone_inputs(
        raw_blocks, genotype_evidence, observed_masks, num_threads=num_processes
    )
    blocks = tuple(prepared)
    block_underfit_flags, wildcard_diagnostics = _stage1_underfit_flags(
        blocks,
        n_samples,
        config.maximum_stage1_wildcard_mass_for_fill_release,
    )
    raw_panels = tuple(_freeze_founder_panel(block) for block in blocks)
    hard_panels = tuple(_hard_panel(panel) for panel in raw_panels)
    folds = _validated_folds(n_samples, config, fold_assignments)

    if (
            isinstance(num_processes, bool)
            or int(num_processes) != num_processes
            or num_processes < 1):
        raise ValueError("num_processes must be a positive integer")
    requested_workers = int(num_processes)
    available_workers = core_runtime.available_cpu_count()
    if requested_workers > available_workers:
        raise ValueError(
            f"num_processes exceeds the available CPU affinity "
            f"({requested_workers} > {available_workers})"
        )
    if diagnostics_mode not in {"full", "compact"}:
        raise ValueError("diagnostics_mode must be 'full' or 'compact'")

    production_callbacks = (
        crossfit_fn is assembly_joint_completion.crossfit_block
        and cavity_fn is assembly_boundaries.cavity_fill_unknown_alleles
        and partial_link_fn is assembly_partial_links.link_partial_profiles
    )
    eligible_workers = min(requested_workers, len(blocks))
    if eligible_workers > 1 and production_callbacks:
        (
            crossfit_blocks,
            partial_links,
            cavity_fills,
            shared_input_bytes,
        ) = _run_parallel_block_science(
            raw_panels,
            hard_panels,
            evidence,
            observed,
            folds,
            config,
            diagnostics_mode,
            eligible_workers,
            partial_link_fn,
        )
        used_workers = eligible_workers
        execution_backend = "forkserver_shared_arrays"
    else:
        crossfit_blocks, partial_links, cavity_fills = (
            _run_sequential_block_science(
                raw_panels,
                hard_panels,
                evidence,
                observed,
                folds,
                config,
                diagnostics_mode,
                crossfit_fn,
                cavity_fn,
                partial_link_fn,
            )
        )
        used_workers = 1
        shared_input_bytes = 0
        if requested_workers == 1:
            execution_backend = "sequential_reference"
        elif not production_callbacks:
            execution_backend = "sequential_custom_callbacks"
        else:
            execution_backend = "sequential_insufficient_blocks"

    for index, (fill, panel, block) in enumerate(
            zip(cavity_fills, raw_panels, blocks)):
        _validate_cavity_fill(fill, panel, block, index)

    with core_parallel.numba_thread_scope(requested_workers):
        focal_masks = assembly_occupancy.focal_informative_masks(evidence, observed)
    occupancy_gate = assembly_occupancy.apply_variable_k_occupancy_gate(
        tuple(np.asarray(value.filled, dtype=np.bool_) for value in cavity_fills),
        tuple(value.profiles for value in crossfit_blocks),
        focal_masks,
        partial_links,
        config.occupancy_rule,
        block_underfit_flags=block_underfit_flags,
    )
    diagnostics = _cavity_gate_diagnostics(cavity_fills, occupancy_gate)
    for block, panel, fill, diagnostic in zip(
            blocks, raw_panels, cavity_fills, diagnostics):
        _materialize_kept_fills(block, panel, fill, diagnostic)

    boundary_mappings = []
    for index, link in enumerate(partial_links):
        left = blocks[index]
        right = blocks[index + 1]
        persistent_break = bool(
            getattr(left, "missing_aware_break_after", False)
            or getattr(right, "missing_aware_break_before", False)
        )
        if link.full_stitch and not persistent_break:
            if link.left_to_right is None:
                raise ValueError("a full partial link must provide left_to_right")
            boundary_mappings.append(assembly_components.BoundaryMapping(
                np.asarray(link.left_to_right, dtype=np.int64),
                link.reason,
            ))
            continue

        reason = link.reason
        if persistent_break:
            reason = (
                getattr(left, "missing_aware_break_reason_after", None)
                or getattr(right, "missing_aware_break_reason_before", None)
                or "persistent_input_break"
            )
        boundary_mappings.append(assembly_components.BoundaryMapping(None, reason))

    components = assembly_components.assemble_phase_components(blocks, boundary_mappings)
    identity = stage2_preprocess_identity(
        config,
        fold_assignments=(folds if fold_assignments is not None else None),
        n_samples=n_samples,
    )
    for block in blocks:
        block.stage2_preprocess_identity = identity.record()
    for block in components:
        block.stage2_preprocess_identity = identity.record()

    return Stage2PreprocessResult(
        config_identity=identity,
        fold_assignments=folds,
        raw_panels=raw_panels,
        stage1_wildcard_diagnostics=wildcard_diagnostics,
        crossfit_blocks=crossfit_blocks,
        partial_links=partial_links,
        cavity_fills=cavity_fills,
        occupancy_gate=occupancy_gate,
        cavity_gate_diagnostics=diagnostics,
        prepared_blocks=prepared,
        direct_components=components,
        execution=PreprocessExecutionDiagnostics(
            requested_workers=requested_workers,
            used_workers=used_workers,
            threads_per_worker=1,
            backend=execution_backend,
            diagnostics_mode=diagnostics_mode,
            shared_input_bytes=shared_input_bytes,
            retained_crossfit_array_bytes=_retained_array_bytes(
                crossfit_blocks
            ),
        ),
    )

import haplotype_reconstruction.assembly.components as assembly_components
import haplotype_reconstruction.core.environment as core_environment
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.core.runtime as core_runtime
