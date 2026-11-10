"""Declare one family-phase component task with its true upstream record dependencies.

Owns the engine boundary and the explicit storage-route choice, never science.
The component task is serial; painting's independent emission cells may use its
thread team. Memory estimates are diagnostics and do not choose a reduction order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..configure import ResolvedConfiguration
    from ...contracts.genetic_map import GeneticMapSet

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from ...contracts.evidence import ContigEvidence
from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.painting import Painting
from ...contracts.pedigree import ParentEdges
from ...contracts.phase import PhasedSamples, PhaseInputIdentities, PhaseObservations
from ...contracts.refit import FounderRefit
from ...science.phase.component import PhaseWorkspace
from ...science.phase.parameters import PhaseParameters
from ...store.container import CheckpointFile
from ...store.protocol import Derivation
from ..framework import RecordKey, ResourceRequest, Task
from .boundaries import validate_parent_edges, validate_sample_order

T = TypeVar("T")
R = TypeVar("R")

__all__ = ["execute_contig", "publish_contig", "task"]


RESIDENT_EVIDENCE_BYTES = 8 << 30


def use_windows(samples: int, sites: int, storage: str = "auto") -> bool:
    """Select the native capability threshold from decoded dimensions only (FP22)."""
    from ...core.errors import SettingsError

    if storage not in ("auto", "resident", "windowed"):
        raise SettingsError(f"invalid phase storage route {storage!r}")
    return 24 * samples * sites > RESIDENT_EVIDENCE_BYTES if storage == "auto" else storage == "windowed"


def execute_contig(
    evidence: ContigEvidence,
    painting: Painting,
    refit: FounderRefit,
    edges: ParentEdges,
    genetic_map: ChromosomeGeneticMap,
    parameters: PhaseParameters,
    identities: PhaseInputIdentities,
    *,
    storage: str = "auto",
    root: str | Path | None = None,
) -> PhasedSamples:
    """Check canonical boundaries, then compute every component and assemble a complete contig."""
    import numpy as np

    from ...core.errors import InvariantError
    from ...engine.cpu import current_budget
    from ...science.phase.component import phase_contig
    from ...science.phase.window import site_windows
    from ...store.mapped import discard_pages, mapped_array
    from .records import canonical_component_inputs, validate_canonical_inputs

    validate_canonical_inputs(evidence, painting, refit, edges, genetic_map, identities)
    windowed = use_windows(len(evidence.sample_ids), len(evidence.positions), storage)
    if windowed and root is None:
        raise InvariantError("windowed phase requires a task-owned scratch root")
    components = tuple(
        canonical_component_inputs(
            evidence, painting, refit, edges, genetic_map, parameters, component=index, identities=identities
        )
        for index in range(len(painting.components))
    )
    # Empty contigs still validate independent upstream sample axes and pedigree.
    validate_sample_order(
        evidence.sample_ids, painting=painting.sample_ids, refit=refit.sample_ids, pedigree=edges.sample_ids
    )
    validate_parent_edges(edges)
    if len(painting.components) != len(refit.components):
        raise InvariantError("phase painting/refit component counts differ")

    def allocate(shape: tuple[int, ...], fill: int, *, dtype: Any) -> np.ndarray:
        return mapped_array(shape, dtype, fill, root=root)

    budget = current_budget()

    def prepare_likelihoods(observations: PhaseObservations, indices: np.ndarray, site_bins: np.ndarray) -> np.ndarray:
        spool = mapped_array((len(observations.sample_ids), len(indices), 3), np.float32, root=root)
        windows = tuple(site_windows(site_bins))

        def row(sample: int) -> None:
            for left, right in windows:
                spool[sample, left:right] = observations.likelihoods[sample, indices[left:right]]
            discard_pages(spool[sample])

        # Each sample fills its own spool row, so row writes can run concurrently.
        budget.thread_map(row, range(len(observations.sample_ids)), label="family-phase likelihood spool")
        return spool

    def map_work(function: Callable[[T], R], items: Iterable[T]) -> list[R]:
        return budget.thread_map(function, items, label="family-phase independent work")

    def concurrent(functions: list[Callable[[Callable[..., list[Any]]], R]]) -> list[R]:
        """each function(map) at once on a live share of the task's CPUs, ``map`` running one-thread
        calls on that share (a solve's blocks); results in order."""

        def on_share(function: Callable[[Callable[..., list[Any]]], R]) -> Callable[[Any], R]:
            def run(share: Any) -> R:
                return function(lambda work, items: share.thread_map(work, items, label="family-phase blocks"))

            return run

        results: list[Any] = [None] * len(functions)
        for index, value in budget.concurrent([on_share(f) for f in functions], label="family-phase solves"):
            results[index] = value
        return results

    return phase_contig(
        evidence.contig,
        evidence.sample_ids,
        evidence.positions,
        components,
        parameters,
        edges,
        PhaseWorkspace(
            windowed=windowed,
            scores=False,
            prepare_likelihoods=prepare_likelihoods if windowed else None,
            map_work=map_work,
            concurrent=concurrent,
        ),
        allocate=allocate if windowed else None,
    )


def publish_contig(
    file: CheckpointFile,
    evidence: ContigEvidence,
    painting: Painting,
    refit: FounderRefit,
    edges: ParentEdges,
    genetic_map: ChromosomeGeneticMap,
    parameters: PhaseParameters,
    identities: PhaseInputIdentities,
    expected: Derivation,
    *,
    storage: str = "auto",
    root: str | Path | None = None,
) -> PhasedSamples:
    """Validate external identities before resume; write completion after all scientific records."""
    from ...core.errors import InvariantError
    from ...store.protocol import mark_complete
    from . import records

    records.validate_canonical_inputs(evidence, painting, refit, edges, genetic_map, identities)
    actual = records.derivation(
        evidence.contig,
        evidence.sample_ids,
        identities,
        parameters,
        reverse=expected.tie_reverse,
        code=expected.code,
        policy=expected.numerical_policy,
    )
    if actual.identity != expected.identity:
        raise InvariantError("phase task inputs differ from its supplied derivation")
    saved = records.load(file, expected)
    if saved is not None:
        return saved
    product = execute_contig(
        evidence, painting, refit, edges, genetic_map, parameters, identities, storage=storage, root=root
    )
    records.write(file, expected, product)
    mark_complete(file, records.KEY, {records.KEY: expected.identity})
    return product


from ..inputs.products import task_inputs, required_product, required_dense_evidence
from ..context import StageRun
from ..context import default_scratch


def run(
    root: str, contig: str, resolved: ResolvedConfiguration, maps: GeneticMapSet, scratch: str | None
) -> str:
    """Bind the canonical typed upstream products to family phase's complete-contig publisher."""
    from ...contracts.phase import PhaseInputIdentities
    from ...core.digest import json_digest
    from ..painting import records as painting_records
    from ..pedigree import records as pedigree_records
    from ..phase import records
    from ..phase.tasks import publish_contig
    from ..refit import records as refit_records

    reader, store = task_inputs(root)
    evidence = required_dense_evidence(reader, contig)
    painting = required_product(reader, painting_records, "painting", contig)
    refit = required_product(reader, refit_records, "refit", contig)
    pedigree = required_product(reader, pedigree_records, "pedigree", None)
    expected = records.expected(store, reader.manifest, product="phase", contig=contig)
    identities = PhaseInputIdentities(**expected.inputs)
    publish_contig(
        store.contig(contig),
        evidence,
        painting,
        refit,
        pedigree.edges,
        maps.for_contig(contig),
        resolved.parameters.family_phase,
        identities,
        expected,
        root=default_scratch(Path(root)) / "phase" / json_digest(contig)[:16]
        if scratch is None
        else Path(scratch) / "phase" / json_digest(contig)[:16],
    )
    return expected.identity


def task(key, context: StageRun) -> Task:
    """Declare the production complete-contig phase publisher."""
    return Task(
        key, run, (context.root, key.scope, context.resolved, context.maps, context.scratch),
        outputs=(RecordKey(key.scope, "phase"),), resources=ResourceRequest(1, context.cores),
    )
