"""Keep run and contig checkpoint containers under one versioned layout.

At rest the directory holds ``run.ckpt`` (genome-wide records), one append-only ``contigs/<contig>.ckpt`` per
chromosome and optional ``markers/`` (store.container describes the format and concurrency rules).
Everything here is cheap and picklable.

A run marks its run.ckpt with this release's layout (``layout`` record, :meth:`CheckpointStore.bind_layout`) before
it writes anything else. Container byte compatibility alone does not establish compatible record schemas;
the layout marker prevents reuse across incompatible layouts. Publication and resume by derivation live in store.protocol.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

from ..core.errors import InvariantError, SettingsError
from . import codec
from .container import CheckpointFile, CheckpointMismatch, RecordInfo
from .identity import jsonable
from .releases import CURRENT_RELEASE, unsupported_release

CONTIG_DIRECTORY = "contigs"
MARKER_DIRECTORY = "markers"
RUN_FILE = "run.ckpt"
CONTIG_SUFFIX = ".ckpt"
LAYOUT_KEY = "layout"  # run.ckpt record marking this release's checkpoint layout
LAYOUT = {"release": CURRENT_RELEASE}  # Scientific products require this layout marker.


def _path_token(value, name) -> str:
    result = str(value)
    if not result or result in {".", ".."} or os.sep in result or (os.altsep is not None and os.altsep in result):
        raise ValueError(f"{name} must be a nonempty path-component token")
    return result


class CheckpointStore:
    """A run's checkpoint directory (see the module docstring).

    ``nthreads`` is the default encode/decode CPU budget of its containers (a count, or a live CPU budget re-read
    at each encode/decode); ``clevel`` their default ZSTD level.
    A directory holding other subdirectories than contigs/ and markers/ belongs to another layout and is refused.
    """

    def __init__(self, root, *, nthreads=1, clevel=None):
        self.root = os.fspath(root)
        self.nthreads = codec.thread_setting(nthreads)
        self.clevel = codec.CLEVEL if clevel is None else int(clevel)
        os.makedirs(self.root, exist_ok=True)
        for entry in os.scandir(self.root):
            if entry.is_dir() and entry.name not in (CONTIG_DIRECTORY, MARKER_DIRECTORY):
                raise unsupported_release(f"{self.root} (checkpoint layout {entry.name}/)")

    def __repr__(self):
        return f"CheckpointStore({self.root!r})"

    @property
    def run(self) -> CheckpointFile:
        """The genome-wide container, run.ckpt."""
        return CheckpointFile(os.path.join(self.root, RUN_FILE), kind="run", nthreads=self.nthreads, clevel=self.clevel)

    def contig(self, name) -> CheckpointFile:
        """The container of one chromosome (created by its first put)."""
        name = _path_token(name, "contig")
        return CheckpointFile(
            os.path.join(self.root, CONTIG_DIRECTORY, name + CONTIG_SUFFIX),
            kind="contig",
            name=name,
            nthreads=self.nthreads,
            clevel=self.clevel,
        )

    def contig_names(self) -> tuple[str, ...]:
        """Chromosomes that have a container, sorted by name."""
        try:
            names = os.listdir(os.path.join(self.root, CONTIG_DIRECTORY))
        except FileNotFoundError:
            return ()
        return tuple(
            sorted(
                name[: -len(CONTIG_SUFFIX)]
                for name in names
                if name.endswith(CONTIG_SUFFIX) and not name.startswith(".")
            )
        )

    def with_threads(self, nthreads) -> CheckpointStore:
        """The same store with another default encode/decode CPU budget."""
        store = copy.copy(self)
        store.nthreads = codec.thread_setting(nthreads)
        return store

    def check_layout(self) -> None:
        """Refuse (CheckpointMismatch) a directory this release did not create; reads only, writes nothing.

        run.ckpt carries this release's ``layout`` marker from a run's first write on. A run.ckpt or contig file that
        holds records without it may use incompatible record schemas under the same keys,
        as may a run.ckpt with another marker; directories of older layouts are refused by the constructor.
        """
        run = self.run
        try:
            if run.has(LAYOUT_KEY):
                marker = run.get(LAYOUT_KEY)
                if marker == LAYOUT:
                    return
                raise unsupported_release(
                    f"{self.root} (checkpoint layout)", marker.get("release") if isinstance(marker, dict) else None
                )
            foreign = (
                (run.exists() and bool(run.keys()))
                or any(self.contig(name).keys() for name in self.contig_names())
                or any(
                    entry.is_file()
                    and entry.name not in (RUN_FILE, RUN_FILE + ".index.json", ".init.lock")
                    and not (entry.name.startswith(".run.ckpt.") and entry.name.endswith((".claim", ".tmp")))
                    for entry in os.scandir(self.root)
                )
            )
        except ValueError as error:  # a container header of another kind/name/format
            raise unsupported_release(f"{self.root} (checkpoint layout: {error})") from None
        if foreign:
            raise unsupported_release(f"{self.root} (checkpoint layout marker absent)")

    def bind_layout(self) -> None:
        """Check and bind the layout under the same append lock as first publication.

        Two initialisers may start on an empty directory together. Holding the
        lock across the check avoids mistaking the other initialiser's freshly
        written layout record for an unmarked container with existing records.
        """
        # Reject unmarked foreign files before opening a current container. A
        # marked run can have a current atomic index temporary alongside it.
        # Marker validation remains under the append lock for concurrent init.
        foreign_files = any(
            entry.is_file()
            and entry.name not in (RUN_FILE, RUN_FILE + ".index.json", ".init.lock")
            and not (entry.name.startswith(".run.ckpt.") and entry.name.endswith((".claim", ".tmp")))
            for entry in os.scandir(self.root)
        )
        run = self.run
        try:
            if foreign_files and not run.has(LAYOUT_KEY):
                raise unsupported_release(f"{self.root} (checkpoint layout)")
        except ValueError as error:
            raise unsupported_release(f"{self.root} (checkpoint layout: {error})") from None
        with run._append_lock() as (fd, index):
            marker = index.records.get(LAYOUT_KEY)
            if marker is not None:
                saved = run._read(marker, 1)
                if saved != LAYOUT:
                    raise unsupported_release(
                        f"{self.root} (checkpoint layout)", saved.get("release") if isinstance(saved, dict) else None
                    )
                return
            if index.records or any(self.contig(name).keys() for name in self.contig_names()):
                raise unsupported_release(f"{self.root} (checkpoint layout marker absent)")
            run._append(fd, index, LAYOUT_KEY, LAYOUT, None, None)

    def bind_run_manifest(self, resolved) -> None:
        """Atomically bind the resolved scientific configuration once, after layout.

        Equal concurrent initialisers are no-ops. A mismatch raises SettingsError
        without replacing the winner's manifest; execution/output options must
        already have been removed by the caller.
        """
        self.bind_layout()
        try:
            self.run.put_once("manifest/run", jsonable(resolved))
        except CheckpointMismatch:
            raise SettingsError(
                "this output root was initialised with a different configuration; use a new output root"
            ) from None

    def scatter_view(self, contig: str) -> ScatterView:
        """Give a scatter task its one writable container and a read-only run reader."""
        source = self.run
        reader = ReadOnlyCheckpointFile(source.path, kind="run", nthreads=source.nthreads, clevel=source.clevel)
        return ScatterView(self.contig(contig), reader)

    def bind_manifest(self, key: str, payload: dict) -> None:
        """Publish one JSON-compatible run manifest exactly once (run.ckpt ``manifest/<key>``)."""
        if not isinstance(payload, dict) or not payload:
            raise ValueError("a run manifest must be a nonempty dictionary")
        try:
            self.run.put_once(f"manifest/{key}", jsonable(payload))
        except CheckpointMismatch:
            raise CheckpointMismatch(
                f"manifest '{key}' does not match the current run; use a new output root"
            ) from None

    def manifest(self, key: str) -> dict | None:
        """The run manifest ``manifest/<key>`` (e.g. "run": kind, contigs, sample_ids, source), or None."""
        run = self.run
        name = f"manifest/{key}"
        return run.get(name) if run.has(name) else None

    def control(self, key: str) -> dict | None:
        """A run-scope control record bound at ``init`` (e.g. the simulation manifest), or None.

        Control records are bindings, not scientific publications: they carry the configuration identity that
        callers compare with their own. A scientific publication envelope is refused here; it is read with its
        independently expected identity (read_items/resume).
        """
        try:
            record = self.run.get(key)
        except KeyError:
            return None
        if isinstance(record, dict) and {"derivation", "payload", "publication_schema"} <= record.keys():
            raise CheckpointMismatch(f"run:{key}: a scientific publication is read with its expected identity")
        return record

    def require(self, key: str, contigs) -> None:
        """Require ``key`` in every listed contig's container (OSError naming the ones lacking it)."""
        missing = [str(contig) for contig in contigs if not self.contig(contig).has(key)]
        if missing:
            raise OSError(f"checkpoint record {key!r} is missing for: " + ", ".join(missing))


class RecordScope:
    """The raw records under ``prefix`` of one container.

    Cheap and picklable: pool workers record through pickled copies (each record is one append, safe across
    processes). ``scoped`` narrows to ``<prefix>/<sub>``. Scientific products use
    store.protocol publication and derivation identities instead of this raw prefix helper.
    Records are written at ``clevel`` (FAST_CLEVEL by default: transient or superseded state) with
    ``nthreads`` compression threads (the file's default when None).
    """

    def __init__(self, file: CheckpointFile, prefix: str, *, nthreads=None, clevel=codec.FAST_CLEVEL):
        if not prefix or prefix.startswith("/") or prefix.endswith("/"):
            raise ValueError(f"invalid record prefix: {prefix!r}")
        self.file = file
        self.prefix = prefix
        self.nthreads = nthreads
        self.clevel = clevel

    def __repr__(self):
        return f"RecordScope({self.file.label!r}, {self.prefix!r})"

    def scoped(self, sub: str, *, nthreads=None) -> RecordScope:
        """The records under ``<prefix>/<sub>``, within the same container."""
        scope = copy.copy(self)
        scope.prefix = self.key(sub)
        if nthreads is not None:
            scope.nthreads = nthreads
        return scope

    def key(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def has(self, name: str) -> bool:
        return self.file.has(self.key(name))

    def load(self, name: str):
        """The record ``name``, or None when it is absent."""
        key = self.key(name)
        return self.file.get(key, nthreads=self.nthreads) if self.file.has(key) else None

    def save(self, name: str, payload) -> RecordInfo:
        return self.file.put(self.key(name), payload, nthreads=self.nthreads, clevel=self.clevel)

    def keys(self, prefix: str = "") -> tuple[str, ...]:
        """Names (relative to this scope) of the live records starting with ``prefix``."""
        start = f"{self.prefix}/"
        return tuple(key[len(start) :] for key in self.file.keys(start + prefix))


class ReadOnlyCheckpointFile(CheckpointFile):
    """A run-container reader that refuses every normal append entry point."""

    def _append_lock(self):
        raise InvariantError("scatter tasks cannot append run.ckpt")


@dataclass(frozen=True)
class ScatterView:
    """The durable storage capabilities owned by one contig scatter task."""

    contig: CheckpointFile
    run: ReadOnlyCheckpointFile


__all__ = [
    "CheckpointStore",
    "RecordScope",
    "ScatterView",
    "ReadOnlyCheckpointFile",
    "LAYOUT",
    "LAYOUT_KEY",
    "CONTIG_DIRECTORY",
    "MARKER_DIRECTORY",
    "RUN_FILE",
    "CONTIG_SUFFIX",
]
