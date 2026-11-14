"""Compute and publish one identity-checked recombination contig.

Science reads only PhasedSamples, conditioned ParentEdges and the shared map.
The task owns no pipeline orchestration, hidden cache or source-data reopening.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..configure import ResolvedConfiguration
    from ...contracts.genetic_map import GeneticMapSet

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


from ..inputs.products import task_inputs, required_product
from ..context import StageRun
from ..framework import Task, RecordKey, ResourceRequest


def run(
    root: str, contig: str, resolved: ResolvedConfiguration, maps: GeneticMapSet
) -> str:
    """Execute recombination against checked phase and conditioned edges, binding before resume."""
    from ..pedigree import records as pedigree_records
    from ..phase import records as phase_records
    from ..recombination import records
    from ..recombination.tasks import publish_contig
    from ..relatedness.working_map import working_maps

    reader, store = task_inputs(root)
    phase = required_product(reader, phase_records, "phase", contig)
    pedigree = required_product(reader, pedigree_records, "pedigree", None)
    reverse = reader.manifest["ties"]["reverse"]
    expected = records.expected(store, reader.manifest, product="recombination", contig=contig,
                                parameters=resolved.parameters.recombination)
    publish_contig(
        store.contig(contig),
        phase,
        pedigree.edges,
        resolved.parameters.recombination,
        working_maps(store, reader.manifest, maps).for_contig(contig),
        expected,
        reverse=reverse,
    )
    return expected.identity


def task(key, context: StageRun) -> Task:
    """Declare the production single-core conditional recombination action."""
    return Task(key, run, (context.root, key.scope, context.resolved, context.maps),
                outputs=(RecordKey(key.scope, "recombination"),), resources=ResourceRequest(1, 1))
