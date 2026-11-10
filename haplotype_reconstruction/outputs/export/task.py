"""Declare one formatting-only export task for each released contig.

Owns output resume and stale-file replacement (OU08), using results readers only.
Method: docs/methods/export.md. The export stage publishes the returned output record;
no scientific identity, record or inference is changed by this task.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ...engine.graph import RecordKey, ResourceRequest, Task, TaskKey
from .records import ExportConfig, ExportRecord, complete
from .run import export_records

if TYPE_CHECKING:
    from ...contracts.assembly import FounderRelease
    from ...contracts.phase import PhasedSamples
    from ...contracts.release import ReleasedSites

__all__ = ["ExportResults", "write", "make_task"]


class ExportResults(Protocol):
    """The released products and run metadata an export output task consumes."""

    @property
    def input_sample_ids(self) -> tuple[str, ...] | None: ...

    @property
    def manifest(self) -> dict: ...

    def sites(self, contig: str) -> ReleasedSites | None: ...
    def release(self, contig: str) -> FounderRelease | None: ...
    def phase(self, contig: str) -> PhasedSamples | None: ...


def write(
    results: ExportResults,
    destination: Path,
    contig: str,
    settings: ExportConfig,
    identity: str,
    previous: ExportRecord | None = None,
    *,
    map_work=None,
) -> ExportRecord:
    """Return the matching record or render and remove files superseded by this export.

    A failed render leaves the prior completion record intact. Successful data/index
    placement precedes the caller's publication of the returned record.
    """
    destination = Path(destination)
    if previous is not None and complete(previous, identity, destination):
        return previous
    products = set(settings.products)
    entry = export_records(
        destination,
        contig,
        settings=settings,
        sites=results.sites(contig) if products & {"founders", "samples"} else None,
        founders=results.release(contig) if products & {"founders", "tracks"} else None,
        phase=results.phase(contig) if "samples" in products else None,
        input_sample_ids=results.input_sample_ids,
        synthetic=results.manifest["source"]["kind"] == "simulation",
        map_work=map_work,
    )
    if previous is not None:
        kept = set(entry["files"])
        for name in set(previous.entry["files"]) - kept:
            # The other index of a data file this render kept was retired by its placement, under the file's lock
            #; removing it here, after the lock, could remove a later writer's current index.
            if any(str(name) == f"{data}{index}" for data in kept for index in (".tbi", ".csi")):
                continue
            (destination / str(name)).unlink(missing_ok=True)
    return ExportRecord(identity=identity, entry=entry)


def make_task(
    results: ExportResults,
    destination: Path,
    contig: str,
    settings: ExportConfig,
    identity: str,
    *,
    previous: ExportRecord | None = None,
    depends_on: tuple[TaskKey, ...] = (),
    working_set_bytes: int = 0,
    max_cores: int = 1,
    map_work=None,
) -> Task:
    """Declare a contig writer with bounded parallel files; the interface supplies release/sites/phase dependencies."""
    products = set(settings.products)
    inputs = []
    if products & {"founders", "samples"}:
        inputs.append(RecordKey(contig, "sites"))
    if products & {"founders", "tracks"}:
        inputs.append(RecordKey(contig, "release"))
    if "samples" in products:
        inputs.append(RecordKey(contig, "phase"))
    return Task(
        TaskKey("export", "contig", scope=contig),
        write,
        args=(results, destination, contig, settings, identity, previous),
        kwargs={"map_work": map_work},
        depends_on=depends_on,
        inputs=tuple(inputs),
        outputs=(RecordKey(contig, "export"),),
        resources=ResourceRequest(max_cores=max_cores, working_set_bytes=working_set_bytes),
    )
