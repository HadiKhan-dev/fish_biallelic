"""Declare one family-phase component task with its true upstream record dependencies.

Owns the engine boundary and the explicit storage-route choice, never science.
The component task is serial; painting's independent emission cells may use its
thread team. Memory estimates are diagnostics and do not choose a reduction order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from ...contracts.evidence import ContigEvidence
from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.painting import Painting
from ...contracts.pedigree import ParentEdges
from ...contracts.phase import (
    PhaseComponentInputs,
    PhaseComponentResult,
    PhasedSamples,
    PhaseInputIdentities,
    PhaseObservations,
)
from ...contracts.refit import FounderRefit
from ...science.phase.component import PhaseWorkspace, phase_component
from ...science.phase.parameters import PhaseParameters
from ...store.container import CheckpointFile
from ...store.protocol import Derivation
from ..framework import RecordKey, ResourceRequest, Task, TaskKey
from .boundaries import validate_parent_edges, validate_sample_order

T = TypeVar("T")
R = TypeVar("R")

__all__ = ["component_task", "execute_component"]


def execute_component(
    inputs: PhaseComponentInputs, parameters: PhaseParameters, *, windowed: bool = False
) -> PhaseComponentResult:
    """Compute the typed component result; worker exceptions propagate to the engine."""
    validate_sample_order(
        inputs.observations.sample_ids, painting=inputs.painting.sample_ids, pedigree=inputs.pedigree.sample_ids
    )
    validate_parent_edges(inputs.pedigree)
    return phase_component(inputs, parameters, PhaseWorkspace(windowed=windowed, scores=True)).result


def component_task(
    inputs: PhaseComponentInputs,
    parameters: PhaseParameters,
    *,
    dependencies: tuple[TaskKey, ...] = (),
    windowed: bool = False,
    max_cores: int = 1,
) -> Task:
    """Declare the already-adapted component seam for in-process or forkserver execution.

    Dependencies are producer tasks supplied by the caller. The four scientific
    record dependencies are always declared, even in a fixture-fed isolated task.
    """
    contig = inputs.contig
    samples = len(inputs.observations.sample_ids)
    states = inputs.painting.states.n_states
    bins = len(inputs.painting.bins.starts)
    sites = len(inputs.observations.positions)
    # FP23 resident estimate: four emission-sized workspaces,
    # resident observations/releases, CAL2 state, and the fixed runtime allowance.
    # The pilot retains both solves' releases, so do not estimate only CAL2/E.
    working_set = 32 * samples * states * states * bins + 48 * samples * sites + 32 * states * sites + 2 * 1024**3
    return Task(
        key=TaskKey("phase", "component", contig, str(inputs.component)),
        function=execute_component,
        args=(inputs, parameters),
        kwargs={"windowed": windowed},
        depends_on=dependencies,
        inputs=(
            RecordKey("run", "pedigree"),
            RecordKey(contig, "painting"),
            RecordKey(contig, "founder_refit"),
            RecordKey(contig, "evidence"),
        ),
        outputs=(RecordKey(contig, f"phase/component/{inputs.component}"),),
        resources=ResourceRequest(1, max_cores, working_set, 0),
        affinity=contig,
    )


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
    from ...engine.scheduler import current_budget
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

    def prepare_likelihoods(observations: PhaseObservations, indices: np.ndarray, site_bins: np.ndarray) -> np.ndarray:
        spool = mapped_array((len(observations.sample_ids), len(indices), 3), np.float32, root=root)
        for sample in range(len(observations.sample_ids)):
            for left, right in site_windows(site_bins):
                spool[sample, left:right] = observations.likelihoods[sample, indices[left:right]]
            discard_pages(spool[sample])
        return spool

    budget = current_budget()

    def map_work(function: Callable[[T], R], items: Iterable[T]) -> list[R]:
        return budget.thread_map(function, items, label="family-phase independent work")

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
