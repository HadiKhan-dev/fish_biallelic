"""Plan and compose one assembly level without storage or scheduling effects.

Owns common-domain masks, symmetric partitioning and ordered family collection
(P0 U11-U13). The stage executes returned JoinTasks and persists each result.
Method: docs/methods/assembly.md#hierarchy. Borrowed inputs are never mutated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import cast

import numpy as np

from haplotype_reconstruction.contracts.assembly import JoinRecord, JoinSettings, JoinTask, ReleaseFamily
from haplotype_reconstruction.contracts.panels import FounderComponent

from ..alternatives import common_masks, common_site_masks, component_choices, joint_choices
from ..release import UNRESOLVED_JOINT_BUDGET
from .batches import LevelPartition, informative_masks, partition


@dataclass(frozen=True)
class LevelPlan:
    """Complete batch choices; tasks are indexed in partition range order."""

    partition: LevelPartition
    tasks: tuple[JoinTask, ...]
    retained: tuple[ReleaseFamily, ...]


@dataclass(frozen=True)
class LevelOutput:
    """Composed components and source families in baseline collection order."""

    components: tuple[FounderComponent, ...]
    plan: tuple[tuple[int, int, str | None], ...]
    joins: tuple
    log_likelihoods: tuple[float, ...]
    reused: int
    partition: LevelPartition
    release_alternatives: tuple[ReleaseFamily, ...] = ()
    shadows: tuple = ()


def _task(index, components, prepared, samples, settings) -> JoinTask:
    """A batch task with its components re-based on the prepared panels it covers."""
    first = components[0].start
    stop = components[-1].stop
    shifted = tuple(FounderComponent(c.start - first, c.stop - first, c.rows) for c in components)
    return JoinTask(
        index=index, components=shifted, prepared=tuple(prepared[first:stop]), samples=samples, settings=settings
    )


def plan_level(
    components,
    prepared,
    evidence,
    settings: JoinSettings,
    *,
    batch_size: int,
    min_informative: int,
    release_alternatives=(),
) -> LevelPlan:
    """Choose common evidence domains before sampling each proxy (A40-8)."""
    choices, consumable, inherited = component_choices(components, release_alternatives)
    shared_sites, lost_sites = common_site_masks(choices, prepared) if consumable else ((), ())
    masks = (
        common_masks(choices, prepared, evidence.neutral, settings.linker.max_sites, usable_masks=shared_sites)
        if consumable
        else None
    )
    split = partition(
        components,
        prepared,
        evidence.neutral,
        batch_size=batch_size,
        min_informative=min_informative,
        max_sites=settings.linker.max_sites,
        masks=masks,
    )
    tasks = {}
    retained = list(inherited)
    for index, (first, end) in enumerate(split.ranges):
        samples = split.batch_samples[index]
        source = [
            family
            for family in consumable
            if components[first].start <= family["start"] and family["stop"] <= components[end - 1].stop
        ]
        branches, combinations = joint_choices(choices[first:end])
        if end - first >= 2 and len(samples):
            task = _task(index, split.components[first:end], prepared, samples, settings)
            if source:
                task = replace(
                    task, common_usable=shared_sites[first:end], common_usable_lost=sum(lost_sites[first:end])
                )
            if source and combinations > 1:
                if branches:
                    origin = components[first].start
                    relative = tuple(
                        tuple(FounderComponent(c.start - origin, c.stop - origin, c.rows) for c in branch)
                        for branch in branches
                    )
                    # NumPy 2.5's reduce stubs omit the plain allocating overload (np.ufunc: same object, generic type).
                    nominal_mask = cast(np.ufunc, np.logical_and).reduce(
                        informative_masks(components[first:end], prepared, evidence.neutral, settings.linker.max_sites)
                    )
                    task = replace(
                        task,
                        branches=relative,
                        branch_sources=tuple(source),
                        common_mask_lost=int(nominal_mask.sum()) - len(samples),
                    )
                else:
                    task = replace(task, overflow=True)
                    retained.extend(
                        family.updated(
                            unresolved_search=True,
                            status=UNRESOLVED_JOINT_BUDGET,
                            joint_combinations=combinations,
                            lost_usable_sites=task.common_usable_lost,
                            usable_site_reference="union_of_input_alternatives",
                        )
                        for family in source
                    )
            tasks[index] = task
        else:
            retained.extend(source)
    return LevelPlan(split, tuple(tasks.values()), tuple(retained))


def compose_level(
    planned: LevelPlan,
    results: Mapping[int, JoinRecord],
    *,
    identities: Mapping[int, str] | None = None,
    reused: int = 0,
) -> LevelOutput:
    """Compose joins in batch order, preserving inherited then joined families."""
    split = planned.partition
    tasks = {task.index: task for task in planned.tasks}
    identities = {} if identities is None else identities
    output: list[FounderComponent] = []
    joins, likelihoods = [], []
    families = list(planned.retained)
    shadows = []
    for index, (first, end) in enumerate(split.ranges):
        inputs = split.components[first:end]
        if index not in tasks:
            output.extend(inputs)
            continue
        result = results[index]
        if result.shadow is not None:
            shadows.append(dict(result.shadow, start=inputs[0].start, stop=inputs[-1].stop))
        families.extend(
            ReleaseFamily.from_mapping(item).updated(
                start=inputs[0].start + int(item["start"]), stop=inputs[0].start + int(item["stop"])
            )
            for item in result.release_alternatives
        )
        output.append(
            FounderComponent(
                inputs[0].start,
                inputs[-1].stop,
                result.rows,
                break_before=inputs[0].break_before,
                break_after=inputs[-1].break_after,
            )
        )
        joins.append(result.summary)
        likelihoods.append(float(result.log_likelihood))
    plan = tuple(
        (first, end, identities.get(index, str(index)) if index in tasks else None)
        for index, (first, end) in enumerate(split.ranges)
    )
    return LevelOutput(
        tuple(output), plan, tuple(joins), tuple(likelihoods), reused, split, tuple(families), tuple(shadows)
    )


def run_level(
    components,
    prepared,
    evidence,
    settings: JoinSettings,
    *,
    batch_size: int,
    min_informative: int,
    budget,
    release_alternatives=(),
) -> LevelOutput:
    """Execute a pure context level using the caller's explicit execution capability."""
    from .batch import run_batch

    planned = plan_level(
        components,
        prepared,
        evidence,
        settings,
        batch_size=batch_size,
        min_informative=min_informative,
        release_alternatives=release_alternatives,
    )
    results = {task.index: run_batch(task, evidence.scoring, budget) for task in planned.tasks}
    return compose_level(planned, results)
