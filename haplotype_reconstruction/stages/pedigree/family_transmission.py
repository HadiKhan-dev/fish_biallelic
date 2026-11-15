"""Per-contig family transmission evidence of the frozen PD29 request (no checkpoint; recomputed on resume)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ...contracts.pedigree import PreparedChromosome, TransmissionEvidence, TransmissionRequest
from ...core.errors import InputError
from ...engine.cpu import thread_team
from ...science.pedigree import family_transmission as science
from ...science.pedigree import junctions
from ...science.pedigree.parameters import TRANSMISSION_SWITCH_FLOOR, PedigreeParameters
from ..framework import RecordKey, ResourceRequest, Task, TaskKey

PREPARED = "pedigree/prepared"


def run(
    prepared: PreparedChromosome,
    parameters: PedigreeParameters,
    request: TransmissionRequest,
    *,
    threads: int = 1,
) -> TransmissionEvidence:
    """Every requested edge's segment log likelihoods on this contig's components, in component order."""
    if any(c.state_classes.shape[0] != len(request.sample_ids) for c in prepared.components):
        raise InputError("family transmission request differs from the prepared sample axis")
    parts = [
        (
            junctions.schedules(c.state_classes, c.named_alleles, c.bin_sizes, reverse=parameters.reverse_conventions),
            np.asarray(c.selector_switch_probability, dtype=np.float64),
        )
        for c in prepared.components
    ]
    delta = parameters.direction.mismatch_probability

    def edge(e: int) -> tuple[np.ndarray, np.ndarray]:
        members = (int(request.parents[e]), int(request.children[e]),
                   *map(int, request.members[request.offsets[e]:request.offsets[e + 1]]))
        rows = [science.score_component(s, theta, members, parameters, delta, TRANSMISSION_SWITCH_FLOOR)
                for s, theta in parts]
        return (np.concatenate([r[0] for r in rows], axis=0) if rows else np.zeros((0, 3)),
                np.concatenate([r[1] for r in rows]) if rows else np.zeros(0))

    with thread_team(default=1):
        workers = max(1, int(threads))
        if workers == 1 or len(request.parents) <= 1:
            results = [edge(e) for e in range(len(request.parents))]
        else:
            with ThreadPoolExecutor(workers) as pool:
                results = list(pool.map(edge, range(len(request.parents))))
    offsets = np.zeros(len(results) + 1, dtype=np.int64)
    for e, (ll, _) in enumerate(results):
        offsets[e + 1] = offsets[e] + len(ll)
    return TransmissionEvidence(
        prepared.contig,
        request.sample_ids,
        request.digest,
        offsets,
        np.concatenate([r[0] for r in results], axis=0) if results else np.zeros((0, 3)),
        np.concatenate([r[1] for r in results]) if results else np.zeros(0),
    )


def task(
    prepared: PreparedChromosome,
    parameters: PedigreeParameters,
    request: TransmissionRequest,
    *,
    cores: int = 1,
) -> Task:
    """Independent, affinity-local contig work of the frozen request."""
    return Task(
        TaskKey("pedigree", "family_transmission", prepared.contig),
        run,
        args=(prepared, parameters, request),
        inputs=(RecordKey(prepared.contig, PREPARED),),
        affinity=prepared.contig,
        resources=ResourceRequest(max_cores=cores),
        thread_argument="threads",
    )
