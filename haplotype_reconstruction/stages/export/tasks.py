"""Orchestrate export placement and the ordered manifest after checked releases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...core.errors import SettingsError
from ...engine.cpu import TASK_THREADS
from ...results.reader import Results
from ...store.store import CheckpointStore
from ..context import StageRun
from ..framework import Expansion, Task, TaskKey, RecordKey, ResourceRequest, ResultRef
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


def _previous(file):
    from ...contracts.roles import require_schema
    from ...outputs.export.records import SCHEMA, ExportRecord
    from ...store.protocol import read_output

    saved = read_output(file, records.KEY, identity=lambda payload: payload["identity"])
    if saved is None:
        return None
    require_schema(saved.payload, SCHEMA)
    return ExportRecord(**saved.payload)


def run(root: str, output: str, contig: str, full_length: bool, checked=None) -> str:
    """Render checked release products and publish the output selection after its files."""
    from ...outputs.export.records import ExportConfig, complete
    from ...outputs.export.task import write
    from ...store.protocol import publish_output

    reader, store = _open_run(output, root)
    config = ExportConfig(full_length=full_length)
    identity = records.expected(reader, contig, config, checked=checked)
    file = store.contig(contig)
    destination = Path(output) / "export"
    previous = _previous(file)
    if complete(previous, identity, destination):
        return identity
    if previous is not None and previous.identity != identity:
        # Withhold the old receipt during replacement while keeping its list for stale-file retirement.
        replacing = f"replacing:{identity}"
        publish_output(
            file, records.KEY, replacing,
            dict(schema=previous.schema, identity=replacing, entry=dict(previous.entry)), supersede=True,
        )
    record = write(reader, destination, contig, config, identity, previous, map_work=map_writers)
    publish_output(
        file,
        records.KEY,
        record.identity,
        dict(schema=record.schema, identity=record.identity, entry=dict(record.entry)),
        supersede=True,
    )
    return identity


def _resume_or_render(key, root, output, full_length, checked, cores):
    """Check an existing export with one CPU and admit a writer team on a miss."""
    from ...outputs.export.records import ExportConfig, complete

    reader, store = _open_run(output, root)
    identity = records.expected(reader, key.scope, ExportConfig(full_length=full_length), checked=checked)
    if complete(_previous(store.contig(key.scope)), identity, Path(output) / "export"):
        return identity
    return Expansion((), Task(key, run, (root, output, key.scope, full_length, checked),
                              resources=ResourceRequest(1, cores)))


def manifest_part(root: str, output: str, contig: str, full_length: bool, identity=None, release_identity=None):
    """Check one published export and collect just its released identity flags."""
    from ...contracts.assembly import FounderIdentityFlag
    from ...contracts.payload import from_payload
    from ...contracts.roles import require_schema
    from ...outputs.export.records import ExportConfig
    from ...store.protocol import read_items

    reader, store = _open_run(output, root)
    file = store.contig(contig)
    record = _previous(file)
    if record is None:
        raise SettingsError(f"{contig}: export record is missing")
    if identity is None:
        identity = records.expected(reader, contig, ExportConfig(full_length=full_length))
    if record.identity != identity:
        raise SettingsError(f"{contig}: export differs from current published products or options")
    if release_identity is None:
        release_identity = reader.product_identity("release", contig)
    release = None if release_identity is None else read_items(file, "release", release_identity, ("schema", "identity_flags"))
    if release is None:
        raise SettingsError(f"{contig}: released founder identity flags are missing")
    require_schema(release.payload, 1)
    flags = tuple(from_payload(FounderIdentityFlag, flag) for flag in release.payload.get("identity_flags", ()))
    return dict(record.entry), (contig, "founder_refit", flags)


def write_manifest(root: str, output: str, contigs: tuple[str, ...], full_length: bool, parts=None) -> list[Path]:
    """Publish the manifest after ordered, independently checked chromosome parts."""
    from ...outputs.export.records import ExportConfig
    from ...outputs.export.run import write_founder_identity, write_manifest as publish_manifest

    reader, store = _open_run(output, root)
    if parts is None:
        parts = tuple(manifest_part(root, output, contig, full_length) for contig in contigs)
    entries, flags = zip(*parts, strict=True)
    flags_path = write_founder_identity(
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
    manifest_path = Path(output) / "export" / "manifest.json"
    return [manifest_path] if flags_path is None else [flags_path, manifest_path]


def task(key, context: StageRun, *, full_length=False) -> Task:
    """Declare export after sites, release and phase, with the existing writer team."""
    dependencies = tuple(TaskKey(stage, "contig", key.scope) for stage in ("sites", "release", "phase"))
    checked = {dependency.stage: ResultRef(dependency) for dependency in dependencies}
    existing = CheckpointStore(context.root).contig(key.scope).has(records.KEY)
    function = _resume_or_render if existing else run
    args = (context.root, context.output, key.scope, full_length, checked)
    if existing:
        args = (key, context.root, context.output, full_length, checked, context.cores)
    return Task(key, function, args, depends_on=dependencies, outputs=(RecordKey(key.scope, records.KEY),),
                resources=ResourceRequest(1, 1 if existing else context.cores))


def manifest_tasks(context: StageRun, *, full_length=False, contigs=None):
    """Gather independent export checks in analysed chromosome order."""
    contigs = context.resolved.analysis.included if contigs is None else tuple(contigs)
    parts = tuple(Task(TaskKey("report", "contig", c, "export_manifest"), manifest_part,
                       (context.root, context.output, c, full_length, ResultRef(TaskKey("export", "contig", c)),
                        ResultRef(TaskKey("release", "contig", c))),
                       depends_on=(TaskKey("export", "contig", c), TaskKey("release", "contig", c)))
                  for c in contigs)
    gather = Task(TaskKey("report", "run", "run", "export_manifest"), write_manifest,
                  (context.root, context.output, contigs, full_length,
                   tuple(ResultRef(task.key) for task in parts)), depends_on=tuple(task.key for task in parts))
    return (*parts, gather)
