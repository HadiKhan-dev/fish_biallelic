"""Complete founder panels from immutable evidence before the assembly hierarchy.

Owns the pure U03-U07 transition, block/link task inputs and working-set estimates.
Must not create pools, reserve resources, read storage, log or inspect execution policy.
Method: docs/methods/assembly.md#completion.
Identity: block and boundary results are assembled by index; all original normalization,
fold, likelihood, release and occupancy orders are retained. A caller may provide an
ordered task map; the default evaluates the same tasks sequentially.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import numpy as np
from numba import prange, types

from haplotype_reconstruction.contracts.panels import LocalPanel, PreparedPanel, unknown_copy_mass
from haplotype_reconstruction.contracts.roles import record_field
from haplotype_reconstruction.core.kernels import kernel
from haplotype_reconstruction.core.ties import TieCounts, TieSettings

from ..alleles import site_posterior
from ..alleles.parameters import SitePosteriorParameters
from .cavity_fill import CavityFill, cavity_fill
from .crossfit import CrossFit, FoldPlan, content_fold_plan, crossfit_block
from .links import BoundaryLink, link_boundary
from .occupancy import occupancy_gate
from .parameters import UNIFORM_TOLERANCE, CompletionParameters

T = TypeVar("T")
U = TypeVar("U")


class OrderedMap(Protocol):
    """A caller-owned task map that returns results in submission order."""

    def __call__(self, function: Callable[[T], U], items: Sequence[T]) -> Sequence[U]: ...


class CompletionEvidence(Protocol):
    """Read-only contig evidence; hashing uses the complete observation axis.

    neutral is None only after a final join released it (AssemblyEvidence.release_neutral);
    completion runs before any join, so complete() requires it.
    """

    @property
    def neutral(self) -> np.ndarray | None: ...

    @property
    def observed(self) -> np.ndarray: ...

    @property
    def depth_keys(self) -> tuple[bytes, ...] | None: ...

    @property
    def samples(self) -> int: ...


@dataclass(frozen=True)
class CompletionWorkspace:
    """Explicit policies and optional ordered task map; no scheduler is constructed."""

    ties: TieSettings = field(default_factory=TieSettings)
    site_parameters: SitePosteriorParameters = field(default_factory=SitePosteriorParameters)
    map_tasks: OrderedMap | None = None


@dataclass(frozen=True)
class AbsentTraining:
    """One fold whose profiles used unchanged pre-fill calls because training was absent."""

    block: int = record_field("report", dtype="int64")
    fold: int = record_field("report", dtype="int64")
    heldout_informative_samples: int = record_field("report", dtype="int64")


@dataclass(frozen=True)
class CompletionSummary:
    """Completion counters and scientific veto flags, in input block order.

    Arrays and tuples are newly owned results. Stages encode these fields; no storage
    payload or formatted log message is constructed here.
    """

    candidate_fills: np.ndarray = record_field("report", axes=("B",), units="count", dtype="int64")
    released_fills: np.ndarray = record_field("report", axes=("B",), units="count", dtype="int64")
    unknown_mass_veto: np.ndarray = record_field("state", axes=("B",), dtype="bool")
    gate_reasons: tuple[tuple[str, int], ...] = record_field("report", dtype="tuple[tuple[str, int], ...]")
    rare_without_focal_observation: int = record_field("report", units="count", dtype="int64")
    site_posterior: tuple[int, ...] = record_field("report", units="count", dtype="tuple[int, ...]")
    guard_skipped: int = record_field("report", units="count", dtype="int64")
    tie_events: TieCounts = record_field("diagnostic", default_factory=lambda: TieCounts(()), dtype="TieCounts")
    fold_key_source: str = record_field("report", default="unrecorded", dtype="str")
    fold_group_sizes: tuple[int, ...] = record_field("report", default=(), dtype="tuple[int, ...]")
    fold_informative_sizes: tuple[int, ...] = record_field("report", default=(), dtype="tuple[int, ...]")
    fold_absent_informative_training: tuple[AbsentTraining, ...] = record_field(
        "report", default=(), dtype="tuple[AbsentTraining, ...]"
    )


def block_memory_bytes(samples: int, founders: int, sites: int, config: CompletionParameters) -> int:
    """Upper estimate of one block's working memory in a completion worker (one numeric thread).

    The block's private evidence copy (float64 x 3 and the observed mask per sample and site); the
    cavity fill's emissions of one bin, (N, K, K, bin sites) float64, with the einsum's likelihood,
    probability and logarithm temporaries (five such arrays at the peak); its per-bin pair-state
    scores with their prefix and suffix sums (bins x N x K(K+1)/2 each); the cross-fit's pair-state
    scores and posteriors (N x K(K+1)/2). Measured in process at K = 30, N = 320, 200 sites: 591 MB
    (estimate 601 MB); on 300 tropheops chr8 / actm3 chr13 blocks (N 116): peak - estimate <= 1.2 MB.
    The stage task adds its declared pool overhead; ``config``: CompletionConfig.
    """
    pairs = founders * (founders + 1) // 2
    width = min(sites, config.cavity.snps_per_bin)
    bins = -(-sites // config.cavity.snps_per_bin)
    return int(
        25 * samples * sites
        + 5 * 8 * samples * founders * founders * width
        + 3 * 8 * bins * samples * pairs
        + 8 * 8 * samples * pairs
    )


def link_memory_bytes(samples: int, founders: int) -> int:
    """Upper estimate of one boundary link's working memory: both blocks' carrier profiles and class
    summaries (a few N x K float64 arrays) and K x K cost and rank matrices."""
    return int(16 * 8 * samples * founders + 8 * 8 * founders * founders)


def _unknown_mass_veto(panel: LocalPanel, maximum_mass: float) -> bool:
    """Use the one unknown-copy mass formula; integer sums preserve the original value."""
    depth = np.asarray(panel.has_depth, dtype=np.bool_)
    slots = np.asarray(panel.unknown_slots, dtype=np.int64)
    mass = unknown_copy_mass(slots, depth)
    return bool(np.any(depth & (slots > 0)) and mass > maximum_mass)


@kernel(
    overloads=(
        (
            types.Array(types.float64, 3, "C", readonly=False),
            types.Array(types.boolean, 2, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.boolean, 1, "C", readonly=False),
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False),
        ),
        (
            types.Array(types.float64, 3, "C", readonly=True),
            types.Array(types.boolean, 2, "C", readonly=True),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.boolean, 1, "C", readonly=False),
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False),
        ),
    ),
    cache=True,
    parallel=True,
    nogil=True,
)
def _focal_mask(neutral, observed, starts, offsets, kept, tolerance, focal):
    """Block-major focal-informative mask, in one scan over (block, sample) rows: block b occupies
    cells [N offsets[b], N offsets[b + 1]) as a (N, L_b) array; a cell is focal when the sample is
    observed at a kept site and its normalised evidence is non-uniform (divide before subtracting)."""
    samples = neutral.shape[0]
    blocks = len(starts)
    for task in prange(blocks * samples):
        block, sample = task // samples, task % samples
        start = offsets[block]
        width = offsets[block + 1] - start
        row = samples * start + sample * width
        first = starts[block]
        for site in range(width):
            seen = observed[sample, first + site] and kept[start + site]
            a = neutral[sample, first + site, 0]
            b = neutral[sample, first + site, 1]
            c = neutral[sample, first + site, 2]
            total = (a + b) + c
            if total > 0.0:
                a, b, c = a / total, b / total, c / total
                focal[row + site] = seen and max(a, b, c) - min(a, b, c) > tolerance
            else:
                focal[row + site] = False


def _block_view(flat, samples, start, stop):
    """Block [start, stop) of a block-major mask as (N, L)."""
    return flat[samples * start : samples * stop].reshape(samples, stop - start)


@dataclass(frozen=True)
class CompletionBlock:
    """One independent block's immutable inputs; evidence is a read-only contig slice, copied when its task starts."""

    alleles: np.ndarray
    evidence: np.ndarray
    observed: np.ndarray
    folds: np.ndarray
    parameters: CompletionParameters
    settings: np.ndarray


@dataclass(frozen=True)
class CompletionBlockResult:
    """Held-out profiles and cavity candidates for one submitted block."""

    crossfit: CrossFit
    fill: CavityFill


def complete_block(block: CompletionBlock) -> CompletionBlockResult:
    """Compute U04/U05 for one block; inputs are unchanged, outputs newly allocated."""
    evidence = np.array(block.evidence)
    crossfit = crossfit_block(
        block.alleles, evidence, block.observed, block.folds, block.parameters.crossfit, block.settings
    )
    called = block.alleles >= 0
    fill = cavity_fill(
        np.where(called, block.alleles, 0.5), called, evidence, block.observed, block.parameters.cavity, block.settings
    )
    return CompletionBlockResult(crossfit, fill)


@dataclass(frozen=True)
class CompletionBoundary:
    """The adjacent block inputs required by one independent U06 link computation."""

    left: CompletionBlock
    right: CompletionBlock
    left_result: CompletionBlockResult
    right_result: CompletionBlockResult


def complete_boundary(boundary: CompletionBoundary) -> BoundaryLink:
    """Compute one boundary correspondence without mutating either block."""
    return link_boundary(
        boundary.left.alleles,
        boundary.right.alleles,
        boundary.left_result.crossfit.profiles,
        boundary.right_result.crossfit.profiles,
        boundary.left.folds,
        boundary.left.parameters.links,
    )


@dataclass(frozen=True)
class CompletionPlan:
    """A contig's completion inputs shared by every block, computed once before the blocks run (U03).

    fold_plan: the content fold plan of the observed axis; veto: bool (B) unknown-mass vetoes; inference:
    per block int8 (R_b, L_b) calls with unkept sites unknown; offsets: int64 (B + 1) block site offsets;
    focal: block-major focal-informative mask (_focal_mask); settings: the site-posterior parameter vector."""

    fold_plan: FoldPlan
    veto: np.ndarray
    inference: tuple[np.ndarray, ...]
    offsets: np.ndarray
    focal: np.ndarray
    settings: np.ndarray


def plan_completion(
    panels: Sequence[LocalPanel],
    evidence: CompletionEvidence,
    parameters: CompletionParameters,
    workspace: CompletionWorkspace,
) -> CompletionPlan:
    """The shared inputs of complete() for non-empty ``panels``."""
    samples = evidence.samples
    fold_plan = content_fold_plan(evidence.observed, parameters.crossfit.fold_seed, evidence.depth_keys)
    veto = np.array(
        [_unknown_mass_veto(panel, parameters.maximum_discovery_unknown_mass) for panel in panels], dtype=np.bool_
    )
    inference = []
    for panel in panels:
        alleles = np.array(panel.calls, dtype=np.int8)
        alleles[:, ~panel.kept] = -1
        inference.append(alleles)
    blocks = len(panels)
    offsets: np.ndarray = np.zeros(blocks + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([panel.sites for panel in panels])
    starts = np.array([panel.site_start for panel in panels], dtype=np.int64)
    kept = np.concatenate([panel.kept for panel in panels]).astype(np.bool_)
    focal: np.ndarray = np.empty(samples * int(offsets[-1]), dtype=np.bool_)
    assert evidence.neutral is not None  # released only after the final join (see CompletionEvidence)
    _focal_mask(evidence.neutral, evidence.observed, starts, offsets, kept, UNIFORM_TOLERANCE, focal)
    settings = workspace.site_parameters.vector(reverse=workspace.ties.reverse)
    return CompletionPlan(fold_plan, veto, tuple(inference), offsets, focal, settings)


def completion_blocks(
    panels: Sequence[LocalPanel],
    inference: Sequence[np.ndarray],
    neutral: np.ndarray,
    observed: np.ndarray,
    parameters: CompletionParameters,
    folds: np.ndarray,
    settings: np.ndarray,
) -> tuple[CompletionBlock, ...]:
    """The independent inputs of ``panels`` (any consecutive run of a plan's blocks, with their ``inference`` calls)
    from the contig's evidence arrays; evidence is sliced, not copied."""
    return tuple(
        CompletionBlock(
            alleles,
            neutral[:, panel.site_start : panel.site_start + panel.sites],
            observed[:, panel.site_start : panel.site_start + panel.sites] & panel.kept[None, :],
            folds,
            parameters,
            settings,
        )
        for panel, alleles in zip(panels, inference, strict=True)
    )


def plan_blocks(
    panels: Sequence[LocalPanel], evidence: CompletionEvidence, parameters: CompletionParameters, plan: CompletionPlan
) -> tuple[CompletionBlock, ...]:
    """Every block's inputs under ``plan``."""
    assert evidence.neutral is not None
    return completion_blocks(
        panels, plan.inference, evidence.neutral, evidence.observed, parameters, plan.fold_plan.folds, plan.settings
    )


def finish_completion(
    panels: Sequence[LocalPanel],
    evidence: CompletionEvidence,
    parameters: CompletionParameters,
    workspace: CompletionWorkspace,
    plan: CompletionPlan,
    results: Sequence[CompletionBlockResult],
) -> tuple[tuple[PreparedPanel, ...], CompletionSummary]:
    """Link adjacent blocks, gate their candidate fills and release the prepared panels (U06-U08), from every
    block's result in block order."""
    panels = tuple(panels)
    blocks = len(panels)
    samples = evidence.samples
    fold_plan, veto, inference, offsets, focal = plan.fold_plan, plan.veto, plan.inference, plan.offsets, plan.focal
    block_inputs = plan_blocks(panels, evidence, parameters, plan)
    evaluate = workspace.map_tasks
    crossfits = tuple(result.crossfit for result in results)
    fills = tuple(result.fill for result in results)
    boundary_inputs = tuple(
        CompletionBoundary(block_inputs[i], block_inputs[i + 1], results[i], results[i + 1]) for i in range(blocks - 1)
    )
    links = tuple(
        map(complete_boundary, boundary_inputs) if evaluate is None else evaluate(complete_boundary, boundary_inputs)
    )
    gate = occupancy_gate(
        tuple(fill.filled for fill in fills),
        tuple(crossfit.profiles for crossfit in crossfits),
        tuple(_block_view(focal, samples, int(offsets[index]), int(offsets[index + 1])) for index in range(blocks)),
        links,
        parameters.occupancy,
        veto,
        keys=tuple(f"p{int(panel.positions[0])}/b{index}" for index, panel in enumerate(panels)),
        census=workspace.ties.census,
    )
    prepared = []
    for panel, alleles, fill, released_fills in zip(panels, inference, fills, gate.released, strict=False):
        released = np.array(panel.calls, dtype=np.int8)
        released[released_fills] = (fill.posterior_alt[released_fills] > 0.5).astype(np.int8)
        alt_probability = np.array(panel.alt_probability, dtype=np.float64)
        alt_probability[released_fills] = fill.posterior_alt[released_fills]
        prepared.append(
            PreparedPanel(
                block=int(panel.block),
                site_start=int(panel.site_start),
                positions=panel.positions,
                kept=panel.kept,
                released=released,
                inference=alleles,
                alt_probability=alt_probability,
                support=panel.support,
            )
        )
    totals: np.ndarray = np.zeros(len(site_posterior.SUMMARY_FIELDS), dtype=np.int64)
    guard_skipped = 0
    for crossfit, fill in zip(crossfits, fills, strict=False):
        totals += crossfit.site_posterior_counts
        totals += fill.site_posterior_counts
        guard_skipped += crossfit.guard_skipped + fill.guard_skipped
    summary = CompletionSummary(
        candidate_fills=np.array([np.sum(fill.filled) for fill in fills], dtype=np.int64),
        released_fills=np.array([np.sum(mask) for mask in gate.released], dtype=np.int64),
        unknown_mass_veto=veto,
        gate_reasons=gate.reason_counts,
        rare_without_focal_observation=gate.rare_without_focal_observation,
        site_posterior=tuple(int(value) for value in totals),
        guard_skipped=int(guard_skipped),
        tie_events=gate.tie_events or TieCounts(()),
        fold_key_source=fold_plan.key_source,
        fold_group_sizes=fold_plan.group_sizes,
        fold_informative_sizes=fold_plan.informative_sizes,
        fold_absent_informative_training=tuple(
            AbsentTraining(block, fold, count)
            for block, crossfit in enumerate(crossfits)
            for fold, count in crossfit.absent_informative_training
        ),
    )
    return tuple(prepared), summary


def complete(
    panels: Sequence[LocalPanel],
    evidence: CompletionEvidence,
    parameters: CompletionParameters,
    workspace: CompletionWorkspace | None = None,
) -> tuple[tuple[PreparedPanel, ...], CompletionSummary]:
    """Complete local panels in genomic order from full-contig founder evidence.

    panels contain int8[R_b,L_b] calls. Evidence is float64[N,L,3], observed bool[N,L]
    and full-contig depth hashes; folds must never be based on a block crop accidentally.
    Inputs are unchanged. Returns newly owned calls and counters; position/support arrays
    remain read-only shared views. An ELBO decrease raises InvariantError.
    """
    panels = tuple(panels)
    workspace = CompletionWorkspace() if workspace is None else workspace
    if not panels:
        empty: np.ndarray = np.zeros(0, dtype=np.int64)
        return (), CompletionSummary(
            empty, empty.copy(), np.zeros(0, dtype=np.bool_), (), 0, (0,) * len(site_posterior.SUMMARY_FIELDS), 0
        )
    plan = plan_completion(panels, evidence, parameters, workspace)
    block_inputs = plan_blocks(panels, evidence, parameters, plan)
    evaluate = workspace.map_tasks
    results = tuple(map(complete_block, block_inputs) if evaluate is None else evaluate(complete_block, block_inputs))
    return finish_completion(panels, evidence, parameters, workspace, plan, results)


__all__ = [
    "CompletionEvidence",
    "CompletionWorkspace",
    "CompletionSummary",
    "AbsentTraining",
    "CompletionBlock",
    "CompletionBlockResult",
    "CompletionBoundary",
    "CompletionPlan",
    "plan_completion",
    "completion_blocks",
    "plan_blocks",
    "finish_completion",
    "complete",
    "complete_block",
    "complete_boundary",
    "block_memory_bytes",
    "link_memory_bytes",
]
