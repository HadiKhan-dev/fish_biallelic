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
    # Original 3.5 FP23 resident estimate: four emission-sized workspaces,
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


def phase_task(
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
    dependencies: tuple[TaskKey, ...] = (),
    max_cores: int = 1,
    file: CheckpointFile | None = None,
    expected: Derivation | None = None,
) -> Task:
    """Declare one complete-contig phase task; upstream record keys use canonical stage spellings."""
    from ...core.errors import InvariantError

    if (file is None) != (expected is None):
        raise InvariantError("phase publication requires both file and expected derivation")
    n, length = len(evidence.sample_ids), len(evidence.positions)
    largest = max((len(c.bins.starts) * c.states.n_states**2 for c in painting.components), default=0)
    working_set = 32 * n * largest + 48 * n * length + (2 << 30)
    return Task(
        key=TaskKey("phase", "contig", evidence.contig),
        function=execute_contig if file is None else publish_contig,
        args=(evidence, painting, refit, edges, genetic_map, parameters, identities)
        if file is None
        else (file, evidence, painting, refit, edges, genetic_map, parameters, identities, expected),
        kwargs=dict(storage=storage, root=root),
        depends_on=dependencies,
        inputs=(
            RecordKey("run", "pedigree"),
            RecordKey(evidence.contig, "evidence"),
            RecordKey(evidence.contig, "painting"),
            RecordKey(evidence.contig, "founder_refit"),
        ),
        outputs=(RecordKey(evidence.contig, "phase"),),
        resources=ResourceRequest(1, max_cores, working_set, 0),
        affinity=evidence.contig,
    )


def load_evidence(
    file: CheckpointFile,
    identity: str,
    *,
    storage: str = "auto",
    root: str | Path | None = None,
) -> ContigEvidence:
    """Decode checked evidence directly into mapped buffers when its dimensions require windows.

    ST1 verifies the immutable publication before exposing array descriptors. The
    resident record and mapped record have the same canonical evidence contract;
    only owned buffer allocation differs. No source file is reopened.
    """
    from ...contracts.evidence import ContigEvidence
    from ...contracts.roles import require_schema
    from ...core.errors import InvariantError
    from ...store.mapped import discard_pages, mapped_array
    from ...store.protocol import open_windows, read_items
    from ..evidence import records as evidence_records

    header = read_items(
        file, evidence_records.EVIDENCE, identity, ("schema", "contig", "sample_ids", "payload_identity")
    )
    if header is None:
        raise InvariantError("phase requires complete evidence")
    require_schema(header.payload, evidence_records.SCHEMA)
    names = tuple(name for name in evidence_records.ARRAYS if name != "counts")
    # ST1 windows address row slices, not one-dimensional axis vectors.
    window_names = ("likelihoods", "observed", "block_spans")
    buffers = {}
    with open_windows(file, evidence_records.EVIDENCE, identity, window_names) as arrays:
        if arrays is None:
            raise InvariantError("phase evidence arrays are absent")
        n, length, _ = arrays["likelihoods"].shape
        if use_windows(n, length, storage):
            if root is None:
                raise InvariantError("mapped phase evidence requires scratch root")
            for name, value in arrays.items():
                if len(value.shape) > 1:
                    buffers[name] = mapped_array(value.shape, value.dtype, root=root)
    values = evidence_records.load_arrays(file, names, expected_identity=identity, out=buffers or None)
    for value in buffers.values():
        discard_pages(value)
    payload = header.payload
    return ContigEvidence(
        contig=payload["contig"],
        sample_ids=tuple(payload["sample_ids"]),
        counts=None,
        payload_identity=payload["payload_identity"],
        positions=values["positions"],
        likelihoods=values["likelihoods"],
        observed=values["observed"],
        discovery_sites=values["discovery_sites"],
        founder_excluded=values["founder_excluded"],
        block_spans=values["block_spans"],
    )
