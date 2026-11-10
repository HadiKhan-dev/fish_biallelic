"""Declare the production pedigree pipeline over its stored contig preparations."""

from __future__ import annotations

from ..context import StageRun
from ..framework import ResourceRequest, Task
from .prepare import pipeline_start


def task(key, context: StageRun) -> Task:
    """Enter preparation, selective screening, scoring and the terminal decision."""
    return Task(key, pipeline_start, args=(context.root, context.maps, context.manifest, context.cores),
                resources=ResourceRequest(max_cores=context.cores))
