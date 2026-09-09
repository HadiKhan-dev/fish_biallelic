"""painting / components for the canonical reconstruction pipeline."""
from __future__ import annotations


import numpy as np


from dataclasses import dataclass
from typing import Any, List, Tuple, NamedTuple


import haplotype_reconstruction.painting.model as module_painting_model

PAINTING_MODEL_RAGGED = "unified-open-set-public-unknown-v3"


try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    import networkx as nx
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


np.seterr(divide='ignore', invalid='ignore')


class PaintedChunk(NamedTuple):
    start: int
    end: int
    hap1: int
    hap2: int


class SamplePainting:
    def __init__(self, sample_index: int, chunks: List[PaintedChunk]):
        self.sample_index = sample_index
        self.chunks = chunks
        self.num_recombinations = max(0, len(self.chunks) - 1)

    def __repr__(self):
        return f"<SamplePainting ID {self.sample_index}: {len(self.chunks)} chunks>"

    def __iter__(self):
        return iter(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


class BlockPainting:
    def __init__(self, block_position_range: Tuple[int, int], samples: List[SamplePainting]):
        self.start_pos = block_position_range[0]
        self.end_pos = block_position_range[1]
        self.samples = samples
        self.num_samples = len(samples)

    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return self.samples[idx]
    def __iter__(self): return iter(self.samples)


@dataclass(frozen=True)
class ComponentPaintingResult:
    """Painting and provenance for one independently painted component.

    Founder IDs in ``painting`` are local row indices.  ``founder_keys`` maps
    those local IDs back to the component's published founder keys.  Structural
    break reasons are copied from the missing-aware hierarchy output, while
    sample_break_reasons explains sample-level abstentions.
    evidence_eligible_sample_mask records only whether a sample had at
    least one observed, non-uniform genotype-likelihood row and could enter
    the Viterbi painter. It is not a posterior statement that the path is
    resolved. ``painting_model`` identifies the complete exact fast path or
    the ragged BACKGROUND model; only the latter carries typed diagnostics and
    may emit the unresolved label ``-1``.
    """

    component_index: int
    interval: Tuple[int, int]
    informative_site_count: int
    founder_keys: Tuple[Any, ...]
    painting: BlockPainting
    evidence_eligible_sample_mask: np.ndarray
    sample_break_reasons: Tuple[str | None, ...]
    break_reason_before: str | None = None
    break_reason_after: str | None = None
    painting_model: str = PAINTING_MODEL_RAGGED
    ragged_diagnostics: module_painting_model.RaggedPaintingDiagnostics | None = None


@dataclass(frozen=True)
class ComponentPaintingBundle:
    """Ordered, independently namespaced paintings for one chromosome."""

    components: Tuple[ComponentPaintingResult, ...]
    num_samples: int

    def __len__(self):
        return len(self.components)

    def __getitem__(self, idx):
        return self.components[idx]

    def __iter__(self):
        return iter(self.components)


def _validated_component_painting_inputs(
        components, sample_probs_matrix, sample_sites, sample_observed_mask=None):
    """Validate and normalize inputs shared by all component paintings."""

    components = tuple(components)
    sample_probs = np.asarray(sample_probs_matrix)
    sites = np.asarray(sample_sites)
    if sample_probs.ndim != 3 or sample_probs.shape[2] != 3:
        raise ValueError(
            "sample_probs_matrix must have shape (samples, sites, 3)"
        )
    if sites.ndim != 1:
        raise ValueError("sample_sites must be one-dimensional")
    if sample_probs.shape[1] != len(sites):
        raise ValueError(
            "sample_probs_matrix and sample_sites disagree on the site dimension"
        )
    if len(sites) > 1 and np.any(sites[1:] <= sites[:-1]):
        raise ValueError("sample_sites must be strictly increasing")
    if sample_observed_mask is None:
        observed = None
    else:
        observed = np.asarray(sample_observed_mask, dtype=np.bool_)
        if observed.shape != sample_probs.shape[:2]:
            raise ValueError(
                "sample_observed_mask must have shape (samples, sites)"
            )

    previous_end = None
    for component_index, block in enumerate(components):
        positions = np.asarray(block.positions)
        if positions.ndim != 1 or len(positions) == 0:
            raise ValueError(
                f"component {component_index} positions must be a non-empty vector"
            )
        if len(positions) > 1 and np.any(positions[1:] <= positions[:-1]):
            raise ValueError(
                f"component {component_index} positions must be strictly increasing"
            )
        if previous_end is not None and positions[0] <= previous_end:
            raise ValueError(
                "component positions must be ordered, increasing, and non-overlapping"
            )

        site_indices = np.searchsorted(sites, positions)
        if (
                np.any(site_indices >= len(sites))
                or not np.array_equal(sites[site_indices], positions)):
            raise ValueError(
                f"component {component_index} positions are not aligned to sample_sites"
            )

        keep_flags = getattr(block, "keep_flags", None)
        if (
                keep_flags is not None
                and np.asarray(keep_flags).shape != positions.shape):
            raise ValueError(
                f"component {component_index} keep_flags must match positions"
            )
        previous_end = positions[-1]

    return components, sample_probs, sites, observed


def _component_break_reason(block, side):
    """Return a published hierarchy break reason, retaining unknown breaks."""

    reason = getattr(block, f"missing_aware_break_reason_{side}", None)
    if reason is not None:
        return str(reason)
    if bool(getattr(block, f"missing_aware_break_{side}", False)):
        return "unspecified_component_break"
    return None


class ComponentPainter:
    """
    Persistent pool manager for painting multiple chromosomes efficiently.

    Creates the multiprocessing Pool ONCE and reuses it across chromosomes.
    SharedMemory is created per chromosome; workers lazy-initialize when they
    detect a new chromosome ID.

    Usage:
        with paint_samples.PaintingPoolManager(num_processes=112) as painter:
            result = painter.paint_components(
                components,
                sample_probs_matrix,
                sample_sites,
                sample_observed_mask=observed,
            )

    Saves ~10s per chromosome by avoiding repeated Pool creation/teardown.
    """

    def __init__(self, num_processes=16):
        self.num_processes = int(num_processes)


    def paint_components(
            self, components, sample_probs_matrix, sample_sites, *,
            sample_observed_mask, **painting_kwargs):
        """Paint every nonempty component with the unified open-set HMM.

        Fixed whole-component trajectory equivalence classes plus BACKGROUND
        are used even for fully called panels: panel exhaustiveness is not a
        scientific assumption of Stage 2. Published post-cavity calls never
        feed back into the model.
        """


        components, sample_probs, sites, observed = (
            _validated_component_painting_inputs(
                components, sample_probs_matrix, sample_sites,
                sample_observed_mask,
            )
        )
        num_samples = sample_probs.shape[0]
        component_results = []

        ragged_keys = {
            "recomb_rate", "switch_penalty_per_snp", "robustness_epsilon",
            "double_recomb_factor", "snps_per_bin", "batch_size",
            "working_memory_bytes", "chromosome_map",
            "minimum_viterbi_public_class_posterior",
        }
        ragged_kwargs = {
            key: value for key, value in painting_kwargs.items()
            if key in ragged_keys
        }

        for component_index, block in enumerate(components):
            positions = np.asarray(block.positions)
            interval = (int(positions[0]), int(positions[-1]))
            founder_count = len(block.haplotypes)
            frozen_snapshot = getattr(
                block, "missing_aware_inference_discrete_haps", None
            )
            if frozen_snapshot is None:
                raise ValueError(
                    f"component {component_index} lacks its required frozen "
                    "pre-fill founder snapshot"
                )
            frozen_snapshot = np.asarray(frozen_snapshot)
            if frozen_snapshot.shape != (founder_count, len(positions)):
                raise ValueError(
                    f"component {component_index} frozen founder snapshot "
                    "is not founder-by-site aligned"
                )
            if np.any(~np.isin(frozen_snapshot, (-1, 0, 1))):
                raise ValueError(
                    f"component {component_index} frozen founder snapshot "
                    "contains a value outside -1, 0, 1"
                )
            if frozen_snapshot.flags.writeable:
                raise ValueError(
                    f"component {component_index} pre-fill founder snapshot "
                    "is not frozen read-only"
                )

            panel = assembly_observations.founder_inference_panel_from_block_result(block)
            kept = (
                np.ones(len(positions), dtype=np.bool_)
                if getattr(block, "keep_flags", None) is None
                else np.asarray(block.keep_flags) > 0
            )
            component_site_indices = np.searchsorted(sites, positions)
            component_evidence = np.ascontiguousarray(
                sample_probs[:, component_site_indices, :]
            )
            component_observed = (
                np.ones(component_evidence.shape[:2], dtype=np.bool_)
                if observed is None
                else np.ascontiguousarray(observed[:, component_site_indices])
            )
            break_reason_before = _component_break_reason(block, "before")
            break_reason_after = _component_break_reason(block, "after")

            with core_parallel.numba_thread_scope(getattr(self, "num_processes", 1)):
                ragged = module_painting_model.paint_ragged_component(
                    panel, component_evidence, component_observed, kept,
                    **ragged_kwargs,
                )
            informative_site_count = len(ragged.state_space.positions)
            resolved = ragged.evidence_eligible_sample_mask
            painting = ragged.painting
            if informative_site_count == 0:
                painting = BlockPainting(interval, painting.samples)
            if founder_count == 0:
                reason = "no_founders"
            elif informative_site_count == 0:
                reason = "no_called_founder_sites"
            else:
                reason = "no_observed_nonuniform_evidence"
            sample_reasons = tuple(
                None if value else reason for value in resolved
            )
            painting_model = PAINTING_MODEL_RAGGED
            ragged_diagnostics = ragged.diagnostics

            resolved = np.asarray(resolved, dtype=np.bool_)
            resolved.flags.writeable = False
            component_results.append(ComponentPaintingResult(
                component_index=component_index,
                interval=interval,
                informative_site_count=informative_site_count,
                founder_keys=panel.keys,
                painting=painting,
                evidence_eligible_sample_mask=resolved,
                sample_break_reasons=sample_reasons,
                break_reason_before=break_reason_before,
                break_reason_after=break_reason_after,
                painting_model=painting_model,
                ragged_diagnostics=ragged_diagnostics,
            ))

        return ComponentPaintingBundle(tuple(component_results), num_samples)

    def close(self):
        """No unused process pool: painting uses scoped Numba threads."""
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

import haplotype_reconstruction.assembly.observations as assembly_observations
import haplotype_reconstruction.core.parallel as core_parallel
