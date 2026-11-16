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


def _mapper(width: int):
    """The vote's map factory (science.family_transmission.Mapper): map(function, items) -> list in input order, on
    up to ``width`` threads. The fits release the interpreter lock and write no array they share."""

    def threaded(function, items):
        items = list(items)
        if len(items) <= 1 or width <= 1:
            return [function(item) for item in items]
        with ThreadPoolExecutor(min(width, len(items))) as pool:
            return list(pool.map(function, items))

    return threaded


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
    plans = [science.component_plan(schedule, theta, parameters) for schedule, theta in parts]
    edges = range(len(request.parents))

    def members(e: int) -> tuple[int, ...]:
        return (int(request.parents[e]), int(request.children[e]),
                *map(int, request.members[request.offsets[e]:request.offsets[e + 1]]))

    # Every segment of every component of every edge is fitted independently (science.segment_unit), each on one
    # thread. Mapped over edges only, a contig with one or two requested edges fitted all their segments in sequence
    # on one thread of its team (a sibling-rich cross: 1 of 19 threads busy for the whole task, while the stage's
    # one-CPU resampling tasks waited for the CPUs this task held).
    jobs = [
        (e, c, u) for e in edges for c, plan in enumerate(plans) if plan is not None for u in range(len(plan[2]))
    ]

    # A contig often has only a few segments (one requested edge and two 1-Morgan segments on a sibling-rich cross):
    # the threads left over serve the independent fits inside each segment's search (science.segment_models).
    width = max(1, int(threads) // max(1, len(jobs)))

    def fit(job: tuple[int, int, int]) -> tuple[float, float, float]:
        e, c, u = job
        schedule, theta = parts[c]
        return science.segment_unit(
            schedule, plans[c], theta, members(e), u, parameters, delta, TRANSMISSION_SWITCH_FLOOR, width=width,
            mapper=_mapper,
        )

    with thread_team(default=1):
        workers = min(max(1, int(threads)), len(jobs))
        if workers <= 1:
            values = [fit(job) for job in jobs]
        else:
            with ThreadPoolExecutor(workers) as pool:
                values = list(pool.map(fit, jobs))
    fitted: dict[int, list[tuple[float, float, float]]] = {e: [] for e in edges}
    for (e, _, _), value in zip(jobs, values, strict=True):
        fitted[e].append(value)
    morgans = [plan[2] for plan in plans if plan is not None]
    lengths = np.concatenate(morgans) if morgans else np.zeros(0)
    results = [(np.asarray(fitted[e], dtype=np.float64).reshape(-1, 3), lengths) for e in edges]
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
