"""Declare the production relatedness gather."""

from __future__ import annotations

from ..framework import RecordKey, ResourceRequest, Task
from ..context import StageRun


def task(key, context: StageRun) -> Task:
    """Declare the families task with the unchanged live team argument."""
    from .families_task import run_stage

    return Task(
        key, run_stage,
        args=(context.root, context.resolved, context.maps),
        kwargs={"reverse": bool(context.manifest["ties"]["reverse"])},
        inputs=tuple(RecordKey(c, "evidence") for c in context.resolved.contigs),
        outputs=(RecordKey("run", "relatedness/families"),),
        resources=ResourceRequest(min_cores=1, max_cores=context.cores),
        thread_argument="team",
    )
