"""Per-contig family transmission evidence of the frozen PD29 request (no checkpoint; recomputed on resume)."""

from __future__ import annotations

import numpy as np

from ...contracts.pedigree import PreparedChromosome, TransmissionEvidence, TransmissionRequest
from ...core.errors import InputError
from ...engine.cpu import CpuBudget, current_budget, thread_team
from ...science.pedigree import family_transmission as science
from ...science.pedigree import junctions
from ...science.pedigree.numerical_parameters import TRANSMISSION_SWITCH_FLOOR
from ...science.pedigree.parameters import PedigreeParameters
from ..framework import RecordKey, ResourceRequest, Task, TaskKey

PREPARED = "pedigree/prepared"


def _mapper(width: int):
    """Ordered map for direct callers with an explicit fixed thread budget."""
    return CpuBudget.fixed(width).thread_map


def _concurrent(function, items):
    """Use live child CPU shares, restoring the caller's input order."""
    jobs = [lambda share, item=item: function(item) for item in items]
    values = dict(current_budget().concurrent(jobs, label="family transmission"))
    return [values[i] for i in range(len(jobs))]


def _fit_map(function, items):
    return current_budget().thread_map(function, items, label="family transmission starts")


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
            junctions.schedules(c.state_classes, c.named_alleles, c.bin_sizes, reverse=parameters.reverse_conventions,
                                label_error=c.label_error, search_alleles_reversed=prepared.search_alleles_reversed),
            np.asarray(c.selector_switch_probability, dtype=np.float64),
        )
        for c in prepared.components
    ]
    delta = parameters.direction.mismatch_probability  # the floor of every label's allowance (science.member_allowances)
    plans = [science.component_plan(schedule, theta, parameters) for schedule, theta in parts]
    edges = range(len(request.parents))
    members = tuple(
        (int(request.parents[e]), int(request.children[e]),
         *map(int, request.members[request.offsets[e]:request.offsets[e + 1]]))
        for e in edges
    )
    jobs = [(e, c, u) for e in edges for c, plan in enumerate(plans) if plan is not None for u in range(len(plan[2]))]

    def work(job: tuple[int, int, int]) -> int:
        e, c, u = job
        _, offsets, _ = plans[c]
        return len(members[e]) * int(offsets[u + 1] - offsets[u])

    def fit(job: tuple[int, int, int]) -> tuple[float, float, float]:
        e, c, u = job
        schedule, theta = parts[c]
        return science.segment_unit(
            schedule, plans[c], theta, members[e], u, parameters, delta, TRANSMISSION_SWITCH_FLOOR,
            map_searches=_concurrent, map_fits=_fit_map,
        )

    # Advance the most expensive segments first. The shared engine grants
    # their search orders spare CPUs for native starts as other jobs finish.
    ordered = sorted(jobs, key=work, reverse=True)
    with thread_team(default=threads):
        values = dict(zip(ordered, _concurrent(fit, ordered), strict=True))
    fitted: dict[int, list[tuple[float, float, float]]] = {e: [] for e in edges}
    for job in jobs:
        fitted[job[0]].append(values[job])
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
        resources=ResourceRequest(max_cores=cores, width=cores),
        thread_argument="threads",
    )
