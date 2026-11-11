"""Orchestrate export placement and the ordered manifest after checked releases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...core.errors import SettingsError
from ...engine.cpu import TASK_THREADS
from ...results.reader import Results
from ...store.store import CheckpointStore
from ..context import StageRun
from ..framework import Task, TaskKey, RecordKey, ResourceRequest, ResultRef
from . import records


def _open_run(output, root):
    """Read immutable source products and publish outputs through a separate store capability."""
    return Results(output, checkpoints=root, nthreads=TASK_THREADS, exclude_contigs=()), CheckpointStore(root, nthreads=TASK_THREADS)


def map_writers(function, items):
    """Map in input order, giving each output writer a live CPU share for compression and indexing."""
    from ...engine.cpu import current_budget

    items = list(items)
    results: list[Any] = [None] * len(items)
    calls = [lambda share, item=item: function(item) for item in items]
    for index, value in current_budget().concurrent(calls, label="output writers"):
        results[index] = value
    return results


def run(root: str, output: str, contig: str, full_length: bool) -> str:
    """Render checked release products and publish the output selection after its files."""
    from ...contracts.roles import require_schema
    from ...outputs.export.records import SCHEMA, ExportConfig, ExportRecord
    from ...outputs.export.task import write
    from ...store.protocol import publish_output, read_output

    reader, store = _open_run(output, root)
    config = ExportConfig(full_length=full_length)
    identity = records.expected(reader, contig, config)
    file = store.contig(contig)
    previous = read_output(file, records.KEY, identity=lambda payload: payload["identity"])
    if previous is not None:
        require_schema(previous.payload, SCHEMA)
        if previous.payload["identity"] != identity:
            # the files are about to be replaced one product at a time. Until the new record is
            # published, the old record must not certify them (an interrupted replacement could leave one product
            # rerendered under it); it keeps listing its files so the replacement still retires them.
            replacing = dict(previous.payload, identity=f"replacing:{identity}")
            publish_output(file, records.KEY, replacing["identity"], replacing, supersede=True)
            previous = read_output(file, records.KEY, identity=lambda payload: payload["identity"])
    record = write(
        reader,
        Path(output) / "export",
        contig,
        config,
        identity,
        None if previous is None else ExportRecord(**previous.payload),
        map_work=map_writers,
    )
    publish_output(
        file,
        records.KEY,
        record.identity,
        dict(schema=record.schema, identity=record.identity, entry=dict(record.entry)),
        supersede=True,
    )
    return identity


def manifest_part(root: str, output: str, contig: str, full_length: bool):
    """Check one published export and collect its released identity flags."""
    from ...contracts.roles import require_schema
    from ...outputs.export.records import SCHEMA, ExportConfig
    from ...store.protocol import read_output

    reader, store = _open_run(output, root)
    record = read_output(store.contig(contig), records.KEY, identity=lambda p: p["identity"])
    if record is None:
        raise SettingsError(f"{contig}: export record is missing")
    expected = records.expected(reader, contig, ExportConfig(full_length=full_length))
    if record.identity != expected:
        raise SettingsError(f"{contig}: export differs from current published products or options")
    require_schema(record.payload, SCHEMA)
    release = reader.release(contig)
    if release is None:
        raise SettingsError(f"{contig}: released founder identity flags are missing")
    return dict(record.payload["entry"]), (contig, "founder_refit", release.identity_flags)


def write_manifest(root: str, output: str, contigs: tuple[str, ...], full_length: bool, parts=None) -> None:
    """Publish the manifest after ordered, independently checked chromosome parts."""
    from ...outputs.export.records import ExportConfig
    from ...outputs.export.run import write_founder_identity, write_manifest as publish_manifest

    reader, store = _open_run(output, root)
    if parts is None:
        parts = tuple(manifest_part(root, output, contig, full_length) for contig in contigs)
    entries, flags = zip(*parts, strict=True)
    write_founder_identity(
        Path(output) / "export", reader, contigs, ExportConfig(full_length=full_length).products, collected=flags
    )
    publish_manifest(
        Path(output) / "export",
        Path(root),
        synthetic=reader.synthetic,
        reference=reader.manifest["source"].get("reference_vcf"),
        contigs=list(entries),
        settings=ExportConfig(full_length=full_length),
        analysis=reader.analysis.document() if reader.analysis.excluded else None,
    )


def task(key, context: StageRun, *, full_length=False) -> Task:
    """Declare export after sites, release and phase, with the existing writer team."""
    return Task(key, run, (context.root, context.output, key.scope, full_length),
                depends_on=tuple(TaskKey(stage, "contig", key.scope) for stage in ("sites", "release", "phase")),
                outputs=(RecordKey(key.scope, records.KEY),), resources=ResourceRequest(1, context.cores))


def manifest_tasks(context: StageRun, *, full_length=False, contigs=None):
    """Gather independent export checks in analysed chromosome order."""
    contigs = context.resolved.analysis.included if contigs is None else tuple(contigs)
    parts = tuple(Task(TaskKey("report", "contig", c, "export_manifest"), manifest_part,
                       (context.root, context.output, c, full_length),
                       depends_on=(TaskKey("export", "contig", c),), resources=ResourceRequest(1, context.cores))
                  for c in contigs)
    gather = Task(TaskKey("report", "run", "run", "export_manifest"), write_manifest,
                  (context.root, context.output, contigs, full_length,
                   tuple(ResultRef(task.key) for task in parts)), depends_on=tuple(task.key for task in parts))
    return (*parts, gather)
