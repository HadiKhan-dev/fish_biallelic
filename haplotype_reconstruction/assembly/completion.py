"""Freeze evidence-supported local founder panels before hierarchical assembly."""
from __future__ import annotations
from haplotype_reconstruction import PACKAGE_ROOT

import atexit
from contextlib import ExitStack
import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import hashlib
import json
from multiprocessing.shared_memory import SharedMemory
import queue
import threading

from typing import Any, Callable, Mapping, Sequence
import numpy as np
from numba import njit, prange, types
from numba.typed import List
import haplotype_reconstruction.assembly.boundaries as assembly_boundaries
import haplotype_reconstruction.assembly.joint_completion as assembly_joint_completion
import haplotype_reconstruction.assembly.observations as assembly_observations
import haplotype_reconstruction.assembly.occupancy as assembly_occupancy
import haplotype_reconstruction.assembly.partial_links as assembly_partial_links
import haplotype_reconstruction.assembly.site_posterior as assembly_site_posterior
import haplotype_reconstruction.core.haplotypes as core_haplotypes

ASSEMBLY_PREPROCESS_SCHEMA = "assembly-missing-aware-preprocess-v2"


ASSEMBLY_PREPROCESS_BACKEND = (
    "joint-crossfit-partial-link-cavity-variable-occupancy-v2"
)


ASSEMBLY_PREPROCESS_SCIENTIFIC_DEPENDENCIES = tuple(sorted((
    'core/haplotypes.py',
    'core/config.py',
    'assembly/components.py',
    'assembly/joint_completion.py',
    'assembly/joint_statistics.py',
    'assembly/allele_polynomials.py',
    'assembly/site_posterior.py',
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
        for name in ASSEMBLY_PREPROCESS_SCIENTIFIC_DEPENDENCIES
    }


def _joint_block_release_config() -> assembly_joint_completion.JointBlockConfig:
    return assembly_joint_completion.JointBlockConfig(
        minimum_call_probability=0.99,
    )


def _whole_bin_cavity_rule() -> assembly_boundaries.CavityFillRule:
    return assembly_boundaries.CavityFillRule(
        snps_per_bin=50,
        minimum_posterior_probability=0.99,
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
    maximum_discovery_wildcard_mass_for_fill_release: float = 0.0

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
            raise ValueError("assembly joint-block release probability must be 0.99")
        if self.cavity_fill_rule.minimum_posterior_probability != 0.99:
            raise ValueError("assembly cavity release probability must be 0.99")
        threshold = self.maximum_discovery_wildcard_mass_for_fill_release
        if (
                isinstance(threshold, bool)
                or not np.isfinite(threshold)
                or threshold < 0.0
                or threshold > 1.0):
            raise ValueError(
                "maximum block discovery wildcard mass for fill release must be "
                "finite and in [0, 1]"
            )


@dataclass(frozen=True)
class FrozenPreprocessIdentity:
    """Canonical checkpoint-compatible scientific configuration identity."""

    canonical_json: str

    def record(self) -> dict[str, Any]:
        value = json.loads(self.canonical_json)
        if (
                value.get("schema") != ASSEMBLY_PREPROCESS_SCHEMA
                or value.get("backend") != ASSEMBLY_PREPROCESS_BACKEND):
            raise ValueError("assembly preprocessing identity schema mismatch")
        return value


def assembly_preprocess_identity(
    config: CompletionConfig,
    *,
    fold_assignments: np.ndarray | None=None,
    n_samples: int | None=None,
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
        "schema": ASSEMBLY_PREPROCESS_SCHEMA,
        "backend": ASSEMBLY_PREPROCESS_BACKEND,
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
class DiscoveryWildcardDiagnostic:
    """Validated block discovery wildcard evidence behind one block veto."""

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
    guard_skipped_cell_count: int
    site_posterior_counts: tuple


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
class FounderCompletionResult:
    """Typed output of truth-free assembly preprocessing."""

    config_identity: FrozenPreprocessIdentity
    fold_assignments: np.ndarray
    raw_panels: tuple[assembly_observations.FounderPanel, ...]
    discovery_wildcard_diagnostics: tuple[DiscoveryWildcardDiagnostic, ...]
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


def _aligned_inputs(raw_blocks, genotype_evidence, observed_masks):
    if not isinstance(raw_blocks, core_haplotypes.BlockResults):
        raise TypeError("raw_blocks must be a BlockResults")
    source = tuple(raw_blocks)
    if not source:
        raise ValueError("at least one raw assembly block is required")
    evidence_values = tuple(genotype_evidence)
    observed_values = tuple(observed_masks)
    if (
            len(evidence_values) != len(source)
            or len(observed_values) != len(source)):
        raise ValueError(
            "blocks, per-block genotype evidence, and observed masks must align"
        )
    return source, evidence_values, observed_values


def _block_layout(index, block, block_evidence, block_observed, n_samples):
    """Kept sites, float64 evidence and observed mask of one block, shape-checked.

    The evidence is the caller's array unless a dtype conversion was needed
    (then the fourth value is True and the conversion owns its values).
    """
    positions = np.asarray(block.positions)
    if positions.ndim != 1 or len(positions) == 0:
        raise ValueError(f"block {index} must contain positions")
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
    mask = np.asarray(block_observed, dtype=np.bool_)
    if values.ndim != 3 or values.shape[1:] != (len(positions), 3):
        raise ValueError(
            f"block {index} evidence must have shape (samples, sites, 3)"
        )
    if mask.shape != values.shape[:2]:
        raise ValueError(
            f"block {index} observed mask must have shape (samples, sites)"
        )
    if n_samples is not None and values.shape[0] != n_samples:
        raise ValueError("all blocks must share the same sample axis")
    return kept_sites, values, mask, owns_conversion


def _freeze_block_calls(index, block, kept_sites, previous_end):
    """Check one cloned block's calls and store them with the immutable pre-fill snapshot.

    Returns the block's last position, which the next block must exceed.
    """
    positions = np.asarray(block.positions)
    keys = tuple(sorted(block.haplotypes))
    discrete = np.asarray(getattr(block, "discrete_haps", None))
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
    inference_discrete = np.ascontiguousarray(discrete, dtype=np.int8).copy()
    inference_discrete[:, ~kept_sites] = -1
    snapshot = _readonly(inference_discrete, np.int8)
    block.discrete_haps = np.ascontiguousarray(discrete, dtype=np.int8).copy()
    block.missing_aware_inference_discrete_haps = snapshot
    return positions[-1]


def _validate_and_clone_inputs(
    raw_blocks: core_haplotypes.BlockResults,
    genotype_evidence: Sequence[np.ndarray],
    observed_masks: Sequence[np.ndarray],
    num_threads=1,
):
    """Clone blocks, evidence and kept observed masks for the in-process path."""
    _, evidence_values, observed_values = _aligned_inputs(
        raw_blocks, genotype_evidence, observed_masks)
    cloned = copy.deepcopy(raw_blocks)
    evidence = []
    source_evidence = List.empty_list(types.Array(types.float64, 3, "A", readonly=True))
    observed = []
    n_samples = None
    previous_end = None
    for index, (block, block_evidence, block_observed) in enumerate(
            zip(tuple(cloned), evidence_values, observed_values)):
        kept_sites, values, mask, owns_conversion = _block_layout(
            index, block, block_evidence, block_observed, n_samples)
        previous_end = _freeze_block_calls(
            index, block, kept_sites, previous_end)
        n_samples = values.shape[0]
        mask = mask & kept_sites[None,:]
        frozen_values = values.view()
        frozen_values.setflags(write=False)
        source_evidence.append(frozen_values)
        # A dtype conversion already owns its snapshot. Do not retain a
        # second chromosome-sized float64 copy for float32 callers.
        evidence.append(values if owns_conversion
                        else np.empty(values.shape, dtype=np.float64))
        mask.setflags(write=False)
        observed.append(mask)

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


def _discovery_underfit_flag(
    index: int,
    block: core_haplotypes.BlockResult,
    n_samples: int,
    maximum_mass_for_fill_release: float,
) -> tuple[bool, DiscoveryWildcardDiagnostic]:
    """Validate one block's discovery wildcard provenance and flag fitted-panel underfit."""

    discrete = np.asarray(block.discrete_haps, dtype=np.int8)
    keep_flags = getattr(block, "keep_flags", None)
    kept = (
        np.ones(discrete.shape[1], dtype=np.bool_)
        if keep_flags is None
        else np.asarray(keep_flags) > 0
    )
    inference_eligible_unknown_count = int(np.count_nonzero(
        (discrete < 0) & kept[None,:]
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
            return underfit, DiscoveryWildcardDiagnostic(
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
                    "discovery_wildcard_metadata_absent_with_unknown_"
                    "founder_cells"
                    if underfit
                    else "discovery_wildcard_metadata_absent_no_unknown_"
                    "founder_cells"
                ),
            )
        summary_mass = _validated_wildcard_mass(mass_value, index)
        underfit = summary_mass > maximum_mass_for_fill_release
        return underfit, DiscoveryWildcardDiagnostic(
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
            reason="discovery_wildcard_mass_summary_only",
        )
    if not (has_depth and has_slots):
        raise ValueError(
            f"block {index} has incomplete block discovery wildcard metadata"
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
        reason = "discovery_wildcard_panel_underfit"
    elif np.any(observed_wildcard):
        reason = "discovery_wildcard_mass_within_configured_threshold"
    else:
        reason = "no_observed_kept_depth_wildcards"
    return underfit, DiscoveryWildcardDiagnostic(
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
    )


def _discovery_underfit_flags(
    blocks: Sequence[core_haplotypes.BlockResult],
    n_samples: int,
    maximum_mass_for_fill_release: float,
) -> tuple[tuple[bool, ...], tuple[DiscoveryWildcardDiagnostic, ...]]:
    """Validate block discovery wildcard provenance and flag fitted-panel underfit."""

    values = tuple(
        _discovery_underfit_flag(
            index, block, n_samples, maximum_mass_for_fill_release)
        for index, block in enumerate(blocks)
    )
    return (tuple(flag for flag, _ in values),
            tuple(diagnostic for _, diagnostic in values))


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
            guard_skipped_cell_count=int(np.sum(fold.released.guard_skipped)),
            site_posterior_counts=assembly_site_posterior.summary_counts(
                fold.fit.site_posterior_info),
        ))
    return CompactCrossFitBlock(value.profiles, tuple(diagnostics))


def _site_posterior_totals(crossfit_blocks, cavity_fills):
    """Sum the above-MAX_EXACT_BITS site-posterior counts of every fold and cavity fill."""
    totals = np.zeros(len(assembly_site_posterior.SUMMARY_FIELDS), dtype=np.int64)
    guard = 0
    for value in crossfit_blocks:
        for fold in value.folds:
            if isinstance(fold, CrossFitFoldDiagnostic):
                totals += fold.site_posterior_counts
                guard += fold.guard_skipped_cell_count
            else:
                totals += assembly_site_posterior.summary_counts(fold.fit.site_posterior_info)
                guard += int(np.sum(fold.released.guard_skipped))
    for fill in cavity_fills:
        totals += fill.site_posterior_counts
        guard += fill.n_guard_skipped
    return dict(zip(assembly_site_posterior.SUMMARY_FIELDS, map(int, totals)),
                guard_skipped_cells=int(guard))


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


_PREPROCESS_WORKER_LINK_CONFIG = None


_PREPROCESS_WORKER_DIAGNOSTICS_MODE = None


_PREPROCESS_WORKER_HANDLES = []


def _close_preprocess_worker_shared_memory() -> None:
    global _PREPROCESS_WORKER_HANDLES
    core_parallel.close_shared_memory(_PREPROCESS_WORKER_HANDLES)
    _PREPROCESS_WORKER_HANDLES = []


def _initialize_preprocess_worker(
    shared_metadata,
    offsets,
    folds,
    joint_config,
    cavity_rule,
    link_config,
    diagnostics_mode,
) -> None:
    """Attach the shared block slabs and enforce one numerical thread."""

    core_environment.force_single_threaded_numeric_libraries()
    import numba

    numba.set_num_threads(1)
    _close_preprocess_worker_shared_memory()
    handles = []
    arrays = []
    try:
        for metadata in shared_metadata:
            handle, array = core_parallel.attach_shared_array(metadata)
            handles.append(handle)
            arrays.append(array)
    except BaseException:
        core_parallel.close_shared_memory(handles)
        raise

    global _PREPROCESS_WORKER_EVIDENCE
    global _PREPROCESS_WORKER_OBSERVED
    global _PREPROCESS_WORKER_OFFSETS
    global _PREPROCESS_WORKER_FOLDS
    global _PREPROCESS_WORKER_JOINT_CONFIG
    global _PREPROCESS_WORKER_CAVITY_RULE
    global _PREPROCESS_WORKER_LINK_CONFIG
    global _PREPROCESS_WORKER_DIAGNOSTICS_MODE
    global _PREPROCESS_WORKER_HANDLES
    _PREPROCESS_WORKER_EVIDENCE, _PREPROCESS_WORKER_OBSERVED = arrays
    _PREPROCESS_WORKER_OFFSETS = np.asarray(offsets, dtype=np.int64)
    _PREPROCESS_WORKER_FOLDS = np.asarray(folds, dtype=np.int8)
    _PREPROCESS_WORKER_JOINT_CONFIG = joint_config
    _PREPROCESS_WORKER_CAVITY_RULE = cavity_rule
    _PREPROCESS_WORKER_LINK_CONFIG = link_config
    _PREPROCESS_WORKER_DIAGNOSTICS_MODE = diagnostics_mode
    _PREPROCESS_WORKER_HANDLES = handles
    atexit.register(_close_preprocess_worker_shared_memory)


def _block_slab(flat, offsets, samples, block, genotypes=None):
    """One block's contiguous (samples, sites[, genotypes]) slab of a block-major array.

    A block's rows are stored together, so a worker that reads a block maps
    only that block's pages of the shared segment (a sample-major layout made
    every worker map, and unmap at exit, nearly all of it).
    """
    width = 1 if genotypes is None else genotypes
    start, stop = int(offsets[block]), int(offsets[block + 1])
    shape = ((samples, stop - start) if genotypes is None
             else (samples, stop - start, genotypes))
    return flat[samples * start * width:samples * stop * width].reshape(shape)


def _preprocess_block_worker(task):
    """Crossfit and cavity fill of consecutive blocks, in order."""
    return "blocks", tuple(_block_science(*item) for item in task)


def _block_science(block_index, hard_panel, raw_panel):
    """Crossfit and cavity fill of one block; both only read its private evidence copy."""
    if _PREPROCESS_WORKER_EVIDENCE is None or _PREPROCESS_WORKER_OFFSETS is None:
        raise RuntimeError("assembly preprocess worker is not initialized")
    # The fold vector has one label per sample.
    samples = len(_PREPROCESS_WORKER_FOLDS)
    offsets = _PREPROCESS_WORKER_OFFSETS
    evidence = np.array(_block_slab(
        _PREPROCESS_WORKER_EVIDENCE, offsets, samples, block_index, 3))
    observed = np.array(_block_slab(
        _PREPROCESS_WORKER_OBSERVED, offsets, samples, block_index))
    crossfit = assembly_joint_completion.crossfit_block(
        hard_panel,
        evidence,
        observed,
        fold_assignments=_PREPROCESS_WORKER_FOLDS,
        config=_PREPROCESS_WORKER_JOINT_CONFIG,
    )
    fill = assembly_boundaries.cavity_fill_unknown_alleles(
        raw_panel,
        evidence,
        observed,
        _PREPROCESS_WORKER_CAVITY_RULE,
    )
    return (
        block_index,
        _retain_crossfit(crossfit, _PREPROCESS_WORKER_DIAGNOSTICS_MODE),
        fill,
    )


def _preprocess_link_worker(task):
    """Links of consecutive boundaries; each needs only its two adjacent crossfits."""
    first, panels, profiles = task
    return "links", first, tuple(
        assembly_partial_links.link_partial_profiles(
            panels[offset], panels[offset + 1],
            profiles[offset], profiles[offset + 1],
            fold_assignments=_PREPROCESS_WORKER_FOLDS,
            config=_PREPROCESS_WORKER_LINK_CONFIG)
        for offset in range(len(panels) - 1)
    )


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


# Consecutive blocks per task. A block takes 40-120 ms, so pairs halve the
# coordinator's dispatch work per block (task pickling, result handling, thread
# wake-ups), which limits the pool's occupancy at 24-48 workers (chr3: -4% wall
# at 24, -2% at 48), while thousands of tasks still keep the tail short.
_BLOCKS_PER_TASK = 2


# Block tasks in flight per worker (one running, one queued): no worker waits for
# the coordinator between tasks, and a boundary link, sent as soon as its two
# crossfits exist, never queues behind the rest of the chromosome.
_TASKS_IN_FLIGHT_PER_WORKER = 2


# Consecutive boundary links sent as one task (one link is about 2 ms of work).
_LINKS_PER_TASK = 8


# Threads of the validated copy into fresh shared memory. As for
# core.parallel.create_shared_array, first-touch page faults saturate at 16.
_SHARED_COPY_THREADS = 16


# Normalised genotype evidence whose range is at most this is uniform, hence
# not focal-informative for the occupancy gate (the default of
# occupancy.focal_informative_masks, which the in-process path calls).
_FOCAL_UNIFORM_TOLERANCE = 1e-12


@njit(cache=True, parallel=True, fastmath=False)
def _pack_block_slabs(evidence, observed, kept, offsets, tolerance, values, masks, focal):
    """Copy every block into its contiguous slab of the shared worker inputs.

    One task per independent (block, sample) row and one scan: genotype
    evidence is copied and checked finite and non-negative; the mask is the
    observed and kept cells; the focal-informative mask (the occupancy gate's
    input) is computed with occupancy.focal_informative_masks' operations
    (divide before subtracting; observed and kept, range above tolerance),
    so the coordinator needs no second pass over a private evidence copy.
    The slab layout is _block_slab's.
    """
    samples = evidence[0].shape[0]
    blocks = len(evidence)
    valid = np.ones(blocks * samples, np.bool_)
    for task in prange(blocks * samples):
        block, sample = task // samples, task % samples
        start = offsets[block]
        width = offsets[block + 1] - start
        row = samples * start + sample * width
        source, mask = evidence[block], observed[block]
        for site in range(width):
            seen = mask[sample, site] and kept[start + site]
            masks[row + site] = seen
            a, b, c = source[sample, site]
            values[3 * (row + site)] = a
            values[3 * (row + site) + 1] = b
            values[3 * (row + site) + 2] = c
            if (not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c)
                    or a < 0.0 or b < 0.0 or c < 0.0):
                valid[task] = False
                focal[row + site] = False
                continue
            total = (a + b) + c
            if total > 0.0:
                a, b, c = a / total, b / total, c / total
                focal[row + site] = seen and max(a, b, c) - min(a, b, c) > tolerance
            else:
                focal[row + site] = False
    return valid


def _empty_shared_array(length, dtype):
    """A new shared segment for a 1-D array: ``(handle, array, metadata)``."""
    dtype = np.dtype(dtype)
    handle = SharedMemory(create=True, size=max(int(length) * dtype.itemsize, 1))
    try:
        array = np.ndarray((int(length),), dtype=dtype, buffer=handle.buf)
    except BaseException:
        core_parallel.close_shared_memory([handle], unlink=True)
        raise
    return handle, array, {
        "shm_name": handle.name, "shape": (int(length),), "dtype": str(dtype)}


def _stream_block_science(pool, blocks, kept_sites, n_samples, config, worker_count):
    """Dependency-driven block science in one pool, dispatched in block order.

    A block's science (crossfit and cavity fill) needs only that block, so
    each block is prepared (calls checked, discovery underfit flag, panels
    frozen) and dispatched, in tasks of _BLOCKS_PER_TASK consecutive blocks,
    while earlier blocks compute. A boundary link needs only its two adjacent
    crossfits, so links go out in runs as soon as their crossfits exist,
    interleaved with block tasks rather than after a barrier. Cavity fills are
    validated as they arrive. Results are stored by index, so nothing depends
    on completion order.
    """
    count = len(blocks)
    raw_panels = [None] * count
    hard_panels = [None] * count
    crossfits = [None] * count
    fills = [None] * count
    links = [None] * (count - 1)
    flags = []
    wildcard_diagnostics = []
    retained_bytes = 0
    results = queue.SimpleQueue()

    def submit(function, task):
        pool.apply_async(function, (task,), callback=results.put,
                         error_callback=results.put)

    maximum_in_flight = (
        _TASKS_IN_FLIGHT_PER_WORKER * _BLOCKS_PER_TASK * worker_count)
    next_block = blocks_in_flight = link_tasks_in_flight = 0
    crossfit_prefix = linked = 0
    previous_end = None
    while next_block < count or blocks_in_flight or link_tasks_in_flight:
        while next_block < count and blocks_in_flight < maximum_in_flight:
            task = []
            stop = min(count, next_block + _BLOCKS_PER_TASK)
            for block_index in range(next_block, stop):
                block = blocks[block_index]
                previous_end = _freeze_block_calls(
                    block_index, block, kept_sites[block_index], previous_end)
                flag, diagnostic = _discovery_underfit_flag(
                    block_index, block, n_samples,
                    config.maximum_discovery_wildcard_mass_for_fill_release)
                flags.append(flag)
                wildcard_diagnostics.append(diagnostic)
                raw_panels[block_index] = _freeze_founder_panel(block)
                hard_panels[block_index] = _hard_panel(raw_panels[block_index])
                task.append((
                    block_index, hard_panels[block_index], raw_panels[block_index]))
            submit(_preprocess_block_worker, tuple(task))
            next_block += len(task)
            blocks_in_flight += len(task)
        record = results.get()
        if isinstance(record, BaseException):
            raise record
        if record[0] == "links":
            _, first, values = record
            links[first:first + len(values)] = values
            link_tasks_in_flight -= 1
            continue
        for index, crossfit, fill in record[1]:
            blocks_in_flight -= 1
            _validate_cavity_fill(fill, raw_panels[index], blocks[index], index)
            crossfits[index] = crossfit
            fills[index] = fill
            # Every block's crossfit is computed on its own, so blocks share
            # no arrays and per-block sums equal one walk over all blocks.
            retained_bytes += _retained_array_bytes(crossfit)
        while crossfit_prefix < count and crossfits[crossfit_prefix] is not None:
            crossfit_prefix += 1
        # Boundaries [0, crossfit_prefix - 1) have both crossfits. Send full
        # runs as they form, and the remainder once every crossfit is done.
        ready = crossfit_prefix - 1
        while (ready - linked >= _LINKS_PER_TASK
               or (crossfit_prefix == count and linked < ready)):
            last = min(ready, linked + _LINKS_PER_TASK)
            submit(_preprocess_link_worker, (
                linked,
                tuple(hard_panels[linked:last + 1]),
                tuple(crossfits[block].profiles
                      for block in range(linked, last + 1)),
            ))
            link_tasks_in_flight += 1
            linked = last
    return (raw_panels, flags, wildcard_diagnostics, crossfits, links, fills,
            retained_bytes)


def _run_parallel_block_science(
    raw_blocks,
    source,
    evidence_values,
    observed_values,
    fold_assignments,
    config,
    diagnostics_mode,
    worker_count,
):
    """Production path: one validated copy into shared memory, then streamed blocks.

    The evidence of every block is copied once, block-major, into shared
    memory, in the scan that validates it and computes the focal-informative
    masks (no separate clone, packing copy or mask pass). The worker pool is
    then created and starts up while the coordinator clones the blocks, and
    block science streams through it (_stream_block_science). Once every task
    is done, a background thread releases the workers and the shared segments
    while the coordinator runs the occupancy gate and finishes; the returned
    callable joins it.
    """
    kept_sites = []
    evidence = List.empty_list(types.Array(types.float64, 3, "A", readonly=True))
    observed = List.empty_list(types.Array(types.boolean, 2, "A", readonly=True))
    n_samples = None
    for index, (block, block_evidence, block_observed) in enumerate(
            zip(source, evidence_values, observed_values)):
        kept, values, mask, _ = _block_layout(
            index, block, block_evidence, block_observed, n_samples)
        n_samples = values.shape[0]
        kept_sites.append(kept)
        frozen_values, frozen_mask = values.view(), mask.view()
        frozen_values.setflags(write=False)
        frozen_mask.setflags(write=False)
        evidence.append(frozen_values)
        observed.append(frozen_mask)
    folds = _validated_folds(n_samples, config, fold_assignments)
    offsets = np.zeros(len(source) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(value) for value in kept_sites], dtype=np.int64)
    cells = n_samples * int(offsets[-1])
    focal = np.empty(cells, dtype=np.bool_)

    handles = []
    shared = []
    pool_stack = ExitStack()
    release_errors = []

    def release():
        try:
            pool_stack.close()
            # Views of the segments must go before their handles close.
            shared.clear()
            core_parallel.close_shared_memory(handles, unlink=True)
        except BaseException as error:
            release_errors.append(error)

    try:
        for length, dtype in ((3 * cells, np.float64), (cells, np.bool_)):
            handle, array, metadata = _empty_shared_array(length, dtype)
            handles.append(handle)
            shared.append((array, metadata))
        with core_parallel.numba_thread_scope(min(worker_count, _SHARED_COPY_THREADS)):
            valid = _pack_block_slabs(
                evidence, observed, np.concatenate(kept_sites), offsets,
                _FOCAL_UNIFORM_TOLERANCE, shared[0][0], shared[1][0], focal)
        if not np.all(valid):
            raise ValueError("genotype evidence must be finite and non-negative")
        pool = pool_stack.enter_context(core_parallel.safe_forkserver_pool(
            worker_count,
            initializer=_initialize_preprocess_worker,
            initargs=(
                tuple(metadata for _, metadata in shared),
                offsets,
                folds,
                config.joint_block_config,
                config.cavity_fill_rule,
                config.partial_link_config,
                diagnostics_mode,
            ),
        ))
        # The workers start up while the coordinator clones the blocks.
        prepared = copy.deepcopy(raw_blocks)
        (raw_panels, flags, wildcard_diagnostics, crossfit_blocks, partial_links,
         cavity_fills, retained_bytes) = _stream_block_science(
            pool, tuple(prepared), kept_sites, n_samples, config, worker_count)
    except BaseException:
        release()
        raise
    releaser = threading.Thread(
        target=release, name="preprocess-release", daemon=True)
    releaser.start()
    try:
        occupancy_gate = _occupancy_gate(
            config, cavity_fills, crossfit_blocks,
            tuple(_block_slab(focal, offsets, n_samples, block)
                  for block in range(len(source))),
            partial_links, flags)
    except BaseException:
        releaser.join()
        raise

    def join_release():
        releaser.join()
        if release_errors:
            raise release_errors[0]

    return (
        prepared,
        folds,
        n_samples,
        tuple(raw_panels),
        tuple(flags),
        tuple(wildcard_diagnostics),
        tuple(crossfit_blocks),
        tuple(partial_links),
        tuple(cavity_fills),
        occupancy_gate,
        # Worker inputs: evidence and observed masks.
        int(cells * 3 * np.dtype(np.float64).itemsize + cells),
        retained_bytes,
        join_release,
    )


def _validate_cavity_fill(
    fill: assembly_boundaries.CavityFillResult,
    panel: assembly_observations.FounderPanel,
    block: core_haplotypes.BlockResult,
    block_index: int,
) -> None:
    if not isinstance(fill, assembly_boundaries.CavityFillResult):
        raise TypeError("cavity function must return a CavityFillResult")
    shape = panel.q.shape
    for name in ("filled", "posterior_alt", "supporters"):
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
    block.cavity_supporters = _readonly(
        fill.supporters, np.float64
    )
    block.genotype_evidence_mode = "assembly_missing_aware_preprocessed"


def _occupancy_gate(
    config,
    cavity_fills,
    crossfit_blocks,
    focal_masks,
    partial_links,
    block_underfit_flags,
):
    return assembly_occupancy.apply_variable_k_occupancy_gate(
        tuple(np.asarray(value.filled, dtype=np.bool_) for value in cavity_fills),
        tuple(value.profiles for value in crossfit_blocks),
        focal_masks,
        tuple(partial_links),
        config.occupancy_rule,
        block_underfit_flags=tuple(block_underfit_flags),
    )


def run_assembly_preprocess(
    raw_blocks: core_haplotypes.BlockResults,
    genotype_evidence: Sequence[np.ndarray],
    observed_masks: Sequence[np.ndarray],
    *,
    fold_assignments: np.ndarray | None=None,
    config: CompletionConfig=CompletionConfig(),
    num_processes: int | Callable[[], int]=1,
    diagnostics_mode: str="full",
    crossfit_fn: Callable[..., assembly_joint_completion.CrossFitBlock]=assembly_joint_completion.crossfit_block,
    cavity_fn: Callable[..., assembly_boundaries.CavityFillResult]=assembly_boundaries.cavity_fill_unknown_alleles,
    partial_link_fn: Callable[..., assembly_partial_links.PartialBoundaryLink]=assembly_partial_links.link_partial_profiles,
) -> FounderCompletionResult:
    """Run the truth-free assembly reference or parallel production path.

    Process parallelism is limited to independent block computations.  Every
    worker is restricted to one numerical thread, so the aggregate compute
    budget is used-workers times one. ``num_processes`` is an int or a live
    budget (core.parallel.budget_value), read once: the pool cannot grow.
    """

    if not isinstance(config, CompletionConfig):
        raise TypeError("config must be a CompletionConfig")
    source, evidence_values, observed_values = _aligned_inputs(
        raw_blocks, genotype_evidence, observed_masks)
    if callable(num_processes):
        num_processes = core_parallel.budget_value(num_processes)
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
    eligible_workers = min(requested_workers, len(source))
    if eligible_workers > 1 and production_callbacks:
        (
            prepared,
            folds,
            n_samples,
            raw_panels,
            block_underfit_flags,
            wildcard_diagnostics,
            crossfit_blocks,
            partial_links,
            cavity_fills,
            occupancy_gate,
            shared_input_bytes,
            retained_crossfit_bytes,
            join_release,
        ) = _run_parallel_block_science(
            raw_blocks,
            source,
            evidence_values,
            observed_values,
            fold_assignments,
            config,
            diagnostics_mode,
            eligible_workers,
        )
        used_workers = eligible_workers
        execution_backend = "forkserver_shared_arrays"
    else:
        prepared, evidence, observed, n_samples = _validate_and_clone_inputs(
            raw_blocks, genotype_evidence, observed_masks,
            num_threads=requested_workers,
        )
        blocks = tuple(prepared)
        block_underfit_flags, wildcard_diagnostics = _discovery_underfit_flags(
            blocks,
            n_samples,
            config.maximum_discovery_wildcard_mass_for_fill_release,
        )
        raw_panels = tuple(_freeze_founder_panel(block) for block in blocks)
        hard_panels = tuple(_hard_panel(panel) for panel in raw_panels)
        folds = _validated_folds(n_samples, config, fold_assignments)
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
        for index, (fill, panel, block) in enumerate(
                zip(cavity_fills, raw_panels, blocks)):
            _validate_cavity_fill(fill, panel, block, index)
        with core_parallel.numba_thread_scope(requested_workers):
            focal_masks = assembly_occupancy.focal_informative_masks(
                evidence, observed, uniform_tolerance=_FOCAL_UNIFORM_TOLERANCE)
        occupancy_gate = _occupancy_gate(
            config, cavity_fills, crossfit_blocks, focal_masks, partial_links,
            block_underfit_flags)
        used_workers = 1
        shared_input_bytes = 0
        retained_crossfit_bytes = _retained_array_bytes(crossfit_blocks)
        join_release = None
        if requested_workers == 1:
            execution_backend = "sequential_reference"
        elif not production_callbacks:
            execution_backend = "sequential_custom_callbacks"
        else:
            execution_backend = "sequential_insufficient_blocks"

    blocks = tuple(prepared)
    site_totals = _site_posterior_totals(crossfit_blocks, cavity_fills)
    if any(site_totals.values()):
        # Sites with more than MAX_EXACT_BITS unresolved founders (none at
        # today's founder counts), their backstop hits and guard abstentions.
        print("[Preprocess] site posterior",
              " ".join(f"{key}={value}" for key, value in site_totals.items()), flush=True)

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
    identity = assembly_preprocess_identity(
        config,
        fold_assignments=(folds if fold_assignments is not None else None),
        n_samples=n_samples,
    )
    for block in blocks:
        block.assembly_preprocess_identity = identity.record()
    for block in components:
        block.assembly_preprocess_identity = identity.record()
    if join_release is not None:
        join_release()

    return FounderCompletionResult(
        config_identity=identity,
        fold_assignments=folds,
        raw_panels=raw_panels,
        discovery_wildcard_diagnostics=wildcard_diagnostics,
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
            retained_crossfit_array_bytes=retained_crossfit_bytes,
        ),
    )

import haplotype_reconstruction.assembly.components as assembly_components
import haplotype_reconstruction.core.environment as core_environment
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.core.runtime as core_runtime
