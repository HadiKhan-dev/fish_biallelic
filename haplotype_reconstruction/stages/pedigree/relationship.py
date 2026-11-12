"""Requested painting relationship evidence, independently published per contig (PD28)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...contracts.payload import from_payload, to_payload
from ...contracts.pedigree import PreparedChromosome, RelationshipChance, RelationshipEvidence, RelationshipRequest
from ...contracts.roles import require_schema
from ...core.digest import array_digest
from ...core.errors import InputError
from ...core.random import stream
from ...engine.cpu import claim_threads, thread_team
from ...science.pedigree import junctions
from ...science.pedigree import relationship as science
from ...science.pedigree.parameters import PEDIGREE_WORKSPACE, PedigreeParameters
from ...store.container import CheckpointFile
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy
from ..framework import RecordKey, ResourceRequest, Task, TaskKey, load_product, publish_product

KEY = "pedigree/relationship"
SEQUENCE_CHUNK = PEDIGREE_WORKSPACE.relationship_sequence_chunk


class RelationshipAdapter:
    """Owned chromosome likelihoods and component chance laws; no repair on decode."""

    schema = 1

    def encode(self, value: RelationshipEvidence) -> Mapping[str, Any]:
        return {"schema": self.schema, **to_payload(value)}

    def decode(self, payload: Mapping[str, Any]) -> RelationshipEvidence:
        require_schema(payload, self.schema)
        return from_payload(RelationshipEvidence, {k: v for k, v in payload.items() if k != "schema"})


def derivation(
    parameters: PedigreeParameters,
    prepared_identity: str,
    request: RelationshipRequest,
    contig_index: int,
    *,
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind schedules' preparation, request, axes, process settings and random stream."""
    return Derivation(
        code_dependency_identity((__name__,)),
        dict(
            relationship=parameters.relationship,
            sample_ids=request.sample_ids,
            contigs=request.contigs,
            contig_index=contig_index,
            bootstrap_seed=parameters.resampling.bootstrap_seed,
            stream="pedigree.relationship.reference",
            schema=RelationshipAdapter.schema,
        ),
        numerical_policy or NumericalPolicy.current(),
        dict(prepared=prepared_identity, dyads=array_digest(request.dyads), trios=array_digest(request.trios)),
        parameters.reverse_conventions,
    ).identity


def reference_triples(sample_ids: tuple[str, ...], count: int, rng: np.random.Generator) -> np.ndarray:
    """Distinct samples per triple, drawn on sorted identifier order; keys do not score."""
    if len(sample_ids) < 3 or len(set(sample_ids)) != len(sample_ids):
        raise InputError("relationship references require at least three distinct sample identifiers")
    order = np.array(sorted(range(len(sample_ids)), key=lambda i: sample_ids[i]), dtype=np.int64)
    return np.array([order[rng.choice(len(order), 3, replace=False)] for _ in range(count)], dtype=np.int64)


def score_schedules(
    contig: str,
    sample_ids: tuple[str, ...],
    components: Sequence[junctions.ComponentSchedule],
    thetas: Sequence[np.ndarray],
    component_indices: Sequence[int],
    request: RelationshipRequest,
    parameters: PedigreeParameters,
    contig_index: int,
    *,
    sequence_chunk: int = SEQUENCE_CHUNK,
) -> RelationshipEvidence:
    """Production scoring seam, with bounded sequence chunks and ordered component sums."""
    if tuple(sample_ids) != request.sample_ids or request.contigs[contig_index] != contig:
        raise InputError("relationship task axes differ from request")
    if isinstance(sequence_chunk, bool) or int(sequence_chunk) != sequence_chunk or sequence_chunk < 1:
        raise InputError("relationship sequence_chunk must be a positive integer")
    sequence_chunk = int(sequence_chunk)
    pair = np.zeros((len(request.dyads), 4))
    trio = np.zeros((len(request.trios), 4))
    chances = []
    rates = science.process_rates(parameters.relationship)
    for part, theta, index in zip(components, thetas, component_indices, strict=True):
        if part.classes.shape[0] != len(sample_ids):
            raise InputError("relationship schedules differ from requested sample axis")
        claim_threads()
        rng = stream(
            "pedigree.relationship.reference",
            seed=parameters.resampling.bootstrap_seed,
            index=contig_index,
            key=int(index),
        )
        references = reference_triples(sample_ids, parameters.relationship.reference_triples, rng)
        q = science.pair_chance(part.classes, part.known)
        s = science.pair_persistence(part.classes, part.known)
        qt, st = science.trio_reference(part.classes, part.known, references)
        distances = science.haldane_distances(theta)
        if len(distances) != part.classes.shape[1]:
            raise InputError("relationship theta and scheduled bin axes differ")
        for rows, result, observe, chance, persistence in (
            (request.dyads, pair, science.pair_sequence, q, s),
            (request.trios, trio, science.trio_sequence, qt, st),
        ):
            for first in range(0, len(rows), sequence_chunk):
                last = min(first + sequence_chunk, len(rows))
                claim_threads()
                x = observe(part.classes, part.known, rows[first:last])
                result[first:last] += science.process_log_likelihoods(
                    x, distances, chance, persistence, rates, isolated_error=parameters.relationship.isolated_error
                )
                del x
        chances.append(RelationshipChance(int(index), q, s, qt, st, references))
    return RelationshipEvidence(contig, tuple(sample_ids), request.dyads, request.trios, pair, trio, tuple(chances))


def run(
    prepared: PreparedChromosome,
    parameters: PedigreeParameters,
    request: RelationshipRequest,
    contig_index: int,
    *,
    threads: int = 1,
    file: CheckpointFile | None = None,
    identity: str | None = None,
) -> RelationshipEvidence:
    """Rebuild schedules exactly as scoring, resume complete contigs, publish atomically."""
    if (file is None) != (identity is None):
        raise InputError("publication requires both file and identity")
    adapter = RelationshipAdapter()
    if file is not None:
        assert identity is not None
        previous = load_product(file, KEY, identity, adapter)
        if previous is not None:
            return previous
    with thread_team(default=threads):
        parts = []
        for component in prepared.components:
            claim_threads()
            parts.append(
                junctions.schedules(
                    component.state_classes,
                    component.named_alleles,
                    component.bin_sizes,
                    reverse=parameters.reverse_conventions,
                )
            )
        result = score_schedules(
            prepared.contig,
            request.sample_ids,
            parts,
            [c.selector_switch_probability for c in prepared.components],
            [c.component_index for c in prepared.components],
            request,
            parameters,
            contig_index,
        )
    if file is not None:
        assert identity is not None
        publish_product(file, KEY, identity, result, adapter)
    return adapter.decode(adapter.encode(result))


def task(
    prepared: PreparedChromosome,
    parameters: PedigreeParameters,
    request: RelationshipRequest,
    contig_index: int,
    *,
    cores: int = 1,
    file: CheckpointFile | None = None,
    identity: str | None = None,
) -> Task:
    """Independent, affinity-local contig work with its own publication key."""
    return Task(
        TaskKey("pedigree", "relationship", prepared.contig),
        run,
        args=(prepared, parameters, request, contig_index),
        kwargs=dict(file=file, identity=identity),
        inputs=(RecordKey(prepared.contig, "pedigree/prepared"),),
        outputs=(RecordKey(prepared.contig, KEY),),
        affinity=prepared.contig,
        resources=ResourceRequest(max_cores=cores),
        thread_argument="threads",
    )
