"""Declare the production founder-release action over checked typed products."""

from __future__ import annotations

from ..inputs.products import task_inputs, required_product
from ..context import StageRun
from ..framework import Task, TaskKey, RecordKey, ResourceRequest


def run(root: str, contig: str, *, team: int = 1) -> str:
    """Pass checked typed inputs to assembly's release operation without transforming them."""
    from ..assembly import records as assembly_records
    from .compute import ReleaseInputs, execute_release
    from . import records
    from ..evidence import records as evidence_records
    from ..painting import records as painting_records
    from ..refit import records as refit_records

    reader, store = task_inputs(root)
    assembly = required_product(reader, assembly_records, "assembly", contig)
    painting = required_product(reader, painting_records, "painting", contig)
    refit = required_product(reader, refit_records, "refit", contig)
    evidence = required_product(reader, evidence_records, "evidence", contig)
    identity = records.expected(store, reader.manifest, product="release", contig=contig).identity
    execute_release(
        ReleaseInputs(assembly, painting, refit, evidence), store.contig(contig), identity, cores=team
    )
    return identity


def task(key, context: StageRun) -> Task:
    """Declare option-independent founder release with its existing dependencies."""
    return Task(key, run, (context.root, key.scope),
                depends_on=tuple(TaskKey(stage, "contig", key.scope) for stage in ("assembly", "painting", "refit", "evidence")),
                outputs=(RecordKey(key.scope, "release"),), resources=ResourceRequest(1, context.cores),
                thread_argument="team")
