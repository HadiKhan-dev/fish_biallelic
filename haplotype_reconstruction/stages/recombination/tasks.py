"""Declare one recombination contig task and its identity-checked publication boundary.

Science reads only PhasedSamples, conditioned ParentEdges and the shared map.
The task owns no pipeline orchestration, hidden cache or source-data reopening.
"""

from __future__ import annotations

import numpy as np

from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.pedigree import ParentEdges
from ...contracts.phase import PhasedSamples
from ...contracts.recombination import Recombination
from ...core.errors import InvariantError
from ...science.recombination.follow import map_chromosome
from ...science.recombination.parameters import RecombinationParameters
from ...store.container import CheckpointFile
from ...store.protocol import Derivation, mark_complete
from ..framework import RecordKey, ResourceRequest, Task, TaskKey
from . import records


def validate_inputs(phase: PhasedSamples, edges: ParentEdges, genetic_map: ChromosomeGeneticMap) -> None:
    """Reject mismatched conditioning and axes before any completion check or calculation."""
    if phase.edges_identity != edges.identity:
        raise InvariantError(f"{phase.contig}: family phase was conditioned on a different pedigree")
    if phase.sample_ids != edges.sample_ids or phase.contig != genetic_map.contig:
        raise InvariantError("recombination sample order or map contig differs from phase")
    if phase.chain_source is None:
        raise InvariantError(f"{phase.contig}: the phase record has no meiosis chains")
    expected = (len(edges.parents), len(phase.bin_centers))
    if any(np.shape(array) != expected for array in (phase.chain_source, phase.chain_support, phase.chain_reset)):
        raise InvariantError("recombination chain axes disagree with conditioned edges and painting bins")


def execute_recombination(
    phase: PhasedSamples,
    edges: ParentEdges,
    parameters: RecombinationParameters,
    genetic_map: ChromosomeGeneticMap,
    *,
    reverse: bool = False,
) -> Recombination:
    """Compute a complete conditional map; all worker exceptions propagate."""
    validate_inputs(phase, edges, genetic_map)
    return map_chromosome(phase, edges, parameters, genetic_map, reverse=reverse)


def publish_contig(
    file: CheckpointFile,
    phase: PhasedSamples,
    edges: ParentEdges,
    parameters: RecombinationParameters,
    genetic_map: ChromosomeGeneticMap,
    expected: Derivation,
    *,
    reverse: bool = False,
) -> Recombination:
    """Bind before resume, then publish and mark completion only after all checks pass."""
    validate_inputs(phase, edges, genetic_map)
    actual = records.derivation(
        phase.contig,
        phase.sample_ids,
        expected.inputs["phase"],
        edges,
        parameters,
        genetic_map,
        reverse=reverse,
        code=expected.code,
        policy=expected.numerical_policy,
    )
    if actual.identity != expected.identity:
        raise InvariantError("recombination task inputs differ from its supplied derivation")
    saved = records.load(file, expected)
    if saved is not None:
        return saved
    product = execute_recombination(phase, edges, parameters, genetic_map, reverse=reverse)
    records.write(file, expected, product)
    mark_complete(file, records.KEY, {records.KEY: expected.identity})
    return product


def recombination_task(
    phase: PhasedSamples,
    edges: ParentEdges,
    parameters: RecombinationParameters,
    genetic_map: ChromosomeGeneticMap,
    *,
    reverse: bool = False,
    dependencies: tuple[TaskKey, ...] = (),
    decoded_phase_bytes: int | None = None,
    file: CheckpointFile | None = None,
    expected: Derivation | None = None,
) -> Task:
    """Declare one serial contig computation with the native decoded-phase memory estimate."""
    if (file is None) != (expected is None):
        raise InvariantError("recombination publication requires both file and expected derivation")
    if decoded_phase_bytes is None:
        decoded_phase_bytes = sum(
            getattr(phase, name).nbytes
            for name in ("calls", "frame_undetermined", "chain_source", "chain_support", "chain_reset")
        )
    return Task(
        key=TaskKey("recombination", "contig", phase.contig),
        function=execute_recombination if file is None else publish_contig,
        args=(phase, edges, parameters, genetic_map)
        if file is None
        else (file, phase, edges, parameters, genetic_map, expected),
        kwargs=dict(reverse=reverse),
        depends_on=dependencies,
        inputs=(RecordKey(phase.contig, "phase"), RecordKey("run", "pedigree")),
        outputs=(RecordKey(phase.contig, records.KEY),),
        resources=ResourceRequest(1, 1, 2 * decoded_phase_bytes + (2 << 30), 0),
        affinity=phase.contig,
    )
