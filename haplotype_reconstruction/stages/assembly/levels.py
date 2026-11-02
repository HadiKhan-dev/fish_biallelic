"""Expand a hierarchy level into independently resumable batch tasks.

Owns derivation lookup, execution teams and incremental join publication.
Pure planning/composition lives in science.assembly.hierarchy.level; each
completed join is durable before the level gather runs (WP_AS6 §4.6/8).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from time import perf_counter
from typing import TypeVar

import numpy as np

from haplotype_reconstruction.contracts.assembly import JoinRecord, JoinSettings, JoinTask, ReleaseFamily
from haplotype_reconstruction.contracts.panels import FounderComponent, PreparedPanel
from haplotype_reconstruction.engine.graph import ResourceRequest, ResultRef, Task, TaskKey
from haplotype_reconstruction.engine.runtime import task_team
from haplotype_reconstruction.engine.scheduler import current_budget
from haplotype_reconstruction.engine.workers import SharedArrayRef
from haplotype_reconstruction.science.assembly.evidence import AssemblyEvidence
from haplotype_reconstruction.science.assembly.hierarchy.batch import run_batch
from haplotype_reconstruction.science.assembly.hierarchy.costs import batch_memory_bytes
from haplotype_reconstruction.science.assembly.hierarchy.level import LevelOutput, LevelPlan, compose_level, plan_level
from haplotype_reconstruction.stages.framework import Expansion
from haplotype_reconstruction.store.protocol import claim

from .completion import attached_scoring
from .records import JOINS, RecordCapability, join_derivation, load_join, save_join

T = TypeVar("T")


def execute_join(
    task: JoinTask,
    scoring: np.ndarray | SharedArrayRef,
    records: RecordCapability | None,
    identity: str,
    *,
    cores: int = 1,
) -> JoinRecord:
    """Execute one whole batch and atomically publish it before returning."""
    if records is not None:
        with claim(records.file, JOINS + identity, identity) as needed:
            if not needed:
                stored = load_join(records.file, identity, index=task.index)
                assert stored is not None
                return stored
            result = execute_join(task, scoring, None, identity, cores=cores)
            save_join(records.file, identity, result)
            return result
    began = perf_counter()
    with attached_scoring(scoring) as data, task_team(cores):
        # The task's live allowance: the join search re-reads it at its boundaries, so a long batch grows into CPUs
        # its finished peers left (a fixed budget kept a tail batch on the cores it started with).
        result = run_batch(task, data, current_budget())
    return replace(result, seconds=perf_counter() - began)


def gather_level(
    planned: LevelPlan,
    cached: Mapping[int, JoinRecord],
    finished: Sequence[JoinRecord],
    identities: Mapping[int, str],
    callback: Callable[[LevelOutput], T],
) -> T:
    """Place results by batch index, then call the continuation in parent context."""
    results = dict(cached)
    results.update((record.index, record) for record in finished)
    output = compose_level(planned, results, identities=identities, reused=len(cached))
    return callback(output)


def expand_level(
    components: Sequence[FounderComponent],
    prepared: Sequence[PreparedPanel],
    evidence: AssemblyEvidence,
    settings: JoinSettings,
    *,
    batch_size: int,
    min_informative: int,
    records: RecordCapability | None,
    release_alternatives: Sequence[ReleaseFamily],
    parent: TaskKey,
    callback: Callable[[LevelOutput], T],
    max_cores: int = 1,
    cores: int = 1,
    scoring_ref: SharedArrayRef | None = None,
) -> T | Expansion:
    """Create one engine task per missing join, preserving input-order gathering."""
    with task_team(cores):
        planned = plan_level(
            components,
            prepared,
            evidence,
            settings,
            batch_size=batch_size,
            min_informative=min_informative,
            release_alternatives=release_alternatives,
        )
    identities = {
        t.index: join_derivation(t, evidence.scoring, evidence_identity=evidence.identity).identity
        for t in planned.tasks
    }
    plan = [(a, b, identities.get(index)) for index, (a, b) in enumerate(planned.partition.ranges)]
    if records is not None and not records.has("plan"):
        records.save("plan", dict(schema=1, batches=plan))
    cached, pending = {}, []
    for item in planned.tasks:
        stored = None if records is None else load_join(records.file, identities[item.index], index=item.index)
        if stored is not None:
            cached[item.index] = stored
            continue
        key = replace(parent, unit=f"{parent.unit}.batch{item.index}")
        # Batches are independent (each publishes its own join under a claim; the gather places them by index), so
        # they are not pinned to the parent's worker: they run side by side across the contig's lane, as 3.5's
        # hierarchy pool ran them. Pinned, a contig's batches ran one at a time on a few threads.
        pending.append(
            Task(
                key,
                execute_join,
                args=(item, evidence.scoring if scoring_ref is None else scoring_ref, records, identities[item.index]),
                resources=ResourceRequest(1, max_cores, batch_memory_bytes(item)),
                thread_argument="cores",
            )
        )
    if not pending:
        return gather_level(planned, cached, (), identities, callback)
    gather = Task(
        replace(parent, unit=parent.unit + ".gather"),
        gather_level,
        args=(planned, cached, tuple(ResultRef(t.key) for t in pending), identities, callback),
        depends_on=tuple(t.key for t in pending),
        affinity=parent.scope,
    )
    return Expansion(tuple(pending), gather)


def run_level_stage(
    components: Sequence[FounderComponent],
    prepared: Sequence[PreparedPanel],
    evidence: AssemblyEvidence,
    settings: JoinSettings,
    *,
    batch_size: int,
    min_informative: int,
    records: RecordCapability | None,
    release_alternatives: Sequence[ReleaseFamily] = (),
    cores: int = 1,
) -> LevelOutput:
    """Synchronous adapter with the same incremental-publication protocol.

    Used for bounded composition cuts and context callbacks with an existing
    engine task; it creates no pool or new CPU authority.
    """
    with task_team(cores):
        planned = plan_level(
            components,
            prepared,
            evidence,
            settings,
            batch_size=batch_size,
            min_informative=min_informative,
            release_alternatives=release_alternatives,
        )
    identities = {
        t.index: join_derivation(t, evidence.scoring, evidence_identity=evidence.identity).identity
        for t in planned.tasks
    }
    if records is not None and not records.has("plan"):
        records.save(
            "plan",
            dict(schema=1, batches=[(a, b, identities.get(i)) for i, (a, b) in enumerate(planned.partition.ranges)]),
        )
    results, reused = {}, 0
    for task in planned.tasks:
        found = None if records is None else load_join(records.file, identities[task.index], index=task.index)
        if found is not None:
            reused += 1
        results[task.index] = (
            found
            if found is not None
            else execute_join(task, evidence.scoring, records, identities[task.index], cores=cores)
        )
    return compose_level(planned, results, identities=identities, reused=reused)
