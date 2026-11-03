"""Evaluate report-only relative proxies per contig in configured order (PD26)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...contracts.pedigree import PreparedChromosome, RelativeContrasts
from ...contracts.roles import record_field
from ...engine.runtime import task_team
from ...engine.scheduler import claim_threads
from ...science.pedigree import relatives as science
from ...science.pedigree.parameters import PedigreeConfig
from ..framework import RecordKey, ResourceRequest, Task, TaskKey


@dataclass(frozen=True)
class ContigContrasts:
    """Owned, report-only chromosome contrasts; usable is the component-wise OR."""

    second: np.ndarray = record_field("report", axes=("Q",), units="nats", dtype="float64")
    full: np.ndarray = record_field("report", axes=("Q",), units="nats", dtype="float64")
    usable: np.ndarray = record_field("report", axes=("Q",), units="1", dtype="bool")


def run(
    prepared: PreparedChromosome,
    parameters: PedigreeConfig,
    pairs: np.ndarray,
    *,
    threads: int = 1,
    reverse: bool | None = None,
) -> ContigContrasts:
    """Sum components in painting order, gating only candidate-parent availability."""
    second = np.zeros(len(pairs))
    full = np.zeros(len(pairs))
    usable: np.ndarray = np.zeros(len(pairs), bool)
    with task_team(threads):
        for component in prepared.components:
            # claim_threads: the science's kernel boundaries re-read this task's share (it grows as peers finish).
            a, b, use = science.component_contrasts(component, parameters, pairs, claim_threads, reverse=reverse)
            second += a
            full += b
            usable |= use
    for value in (second, full, usable):
        value.flags.writeable = False
    return ContigContrasts(second, full, usable)


def aggregate(
    pairs: np.ndarray, views: tuple[tuple[str, ...], ...], contigs: tuple[ContigContrasts, ...]
) -> RelativeContrasts:
    """Aggregate the configured chromosome axis using the declared report sum."""
    if contigs:
        second = np.stack([x.second for x in contigs])
        full = np.stack([x.full for x in contigs])
        usable = np.stack([x.usable for x in contigs])
    else:
        second = np.empty((0, len(pairs)))
        full = second.copy()
        usable = np.empty(second.shape, bool)
    return science.aggregate(pairs, views, second, full, usable)


def task(
    prepared: PreparedChromosome,
    parameters: PedigreeConfig,
    pairs: np.ndarray,
    *,
    cores: int = 1,
    reverse: bool | None = None,
    depends_on: tuple[TaskKey, ...] = (),
) -> Task:
    """One independent contig task; completion order never selects report order."""
    return Task(
        TaskKey("pedigree", "relatives", prepared.contig),
        run,
        args=(prepared, parameters, pairs),
        kwargs=dict(reverse=reverse),
        depends_on=depends_on,
        inputs=(RecordKey(prepared.contig, "pedigree/prepared"),),
        resources=ResourceRequest(max_cores=cores),
        affinity=prepared.contig,
        thread_argument="threads",
    )
