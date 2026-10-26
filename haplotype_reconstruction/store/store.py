"""A run's checkpoint directory: run.ckpt, contigs/<contig>.ckpt and transient work/ states.

At rest the directory holds ``run.ckpt`` (genome-wide records), one append-only ``contigs/<contig>.ckpt`` per
chromosome, and ``work/`` only while transient resume-only state is live (store.container describes the
container format and its concurrency rules). Everything here is cheap and picklable.

A run marks its run.ckpt with this release's layout (``layout`` record, :meth:`CheckpointStore.bind_layout`) before
it writes anything else: 92b3daf wrote the same container format with other contents under the same keys, so a
directory whose run.ckpt or contig files hold records but no marker is refused instead of being appended to.
"""
from __future__ import annotations

import copy
import os

from . import codec
from .container import CheckpointFile, CheckpointMismatch, RecordInfo
from .identity import canonical_json, jsonable

CONTIG_DIRECTORY = "contigs"
WORK_DIRECTORY = "work"
RUN_FILE = "run.ckpt"
CONTIG_SUFFIX = ".ckpt"
LAYOUT_KEY = "layout"          # run.ckpt record marking this release's checkpoint layout
LAYOUT = {"release": 2}        # 92b3daf (release 1) wrote the same containers without it


def _path_token(value, name) -> str:
    result = str(value)
    if (not result or result in {".", ".."} or os.sep in result
            or (os.altsep is not None and os.altsep in result)):
        raise ValueError(f"{name} must be a nonempty path-component token")
    return result


class CheckpointStore:
    """A run's checkpoint directory (see the module docstring).

    ``nthreads`` is the default encode/decode CPU budget of its containers (a count, or a live CPU budget re-read
    at each encode/decode); ``clevel`` their default ZSTD level.
    A directory holding other subdirectories than contigs/ and work/ belongs to another layout and is refused.
    """

    def __init__(self, root, *, nthreads=1, clevel=None):
        self.root = os.fspath(root)
        self.nthreads = codec.thread_setting(nthreads)
        self.clevel = codec.CLEVEL if clevel is None else int(clevel)
        os.makedirs(self.root, exist_ok=True)
        for entry in os.scandir(self.root):
            if entry.is_dir() and entry.name not in (CONTIG_DIRECTORY, WORK_DIRECTORY):
                raise RuntimeError(
                    f"{self.root} holds '{entry.name}/', which is not part of this release's checkpoint "
                    "layout (run.ckpt, contigs/, work/): use a new output root")

    def __repr__(self):
        return f"CheckpointStore({self.root!r})"

    @property
    def run(self) -> CheckpointFile:
        """The genome-wide container, run.ckpt."""
        return CheckpointFile(os.path.join(self.root, RUN_FILE), kind="run",
                              nthreads=self.nthreads, clevel=self.clevel)

    def contig(self, name) -> CheckpointFile:
        """The container of one chromosome (created by its first put)."""
        name = _path_token(name, "contig")
        return CheckpointFile(os.path.join(self.root, CONTIG_DIRECTORY, name + CONTIG_SUFFIX),
                              kind="contig", name=name, nthreads=self.nthreads, clevel=self.clevel)

    def contig_names(self) -> tuple[str, ...]:
        """Chromosomes that have a container, sorted by name."""
        try:
            names = os.listdir(os.path.join(self.root, CONTIG_DIRECTORY))
        except FileNotFoundError:
            return ()
        return tuple(sorted(name[:-len(CONTIG_SUFFIX)] for name in names
                            if name.endswith(CONTIG_SUFFIX) and not name.startswith(".")))

    def with_threads(self, nthreads) -> "CheckpointStore":
        """The same store with another default encode/decode CPU budget."""
        store = copy.copy(self)
        store.nthreads = codec.thread_setting(nthreads)
        return store

    def bind_identity(self, file: CheckpointFile, section: str, identity) -> None:
        """Bind ``identity/<section>`` of a container to one exact identity.

        The first binder records it; later runs must match exactly (canonical JSON), else CheckpointMismatch
        names the section.
        """
        try:
            file.put_once(f"identity/{section}", jsonable(identity))
        except CheckpointMismatch:
            raise CheckpointMismatch(
                f"{file.label}: checkpoint identity '{section}' does not match the current "
                "model/configuration; use a new output root") from None

    def check_layout(self) -> None:
        """Refuse (CheckpointMismatch) a directory this release did not create; reads only, writes nothing.

        run.ckpt carries this release's ``layout`` marker from a run's first write on. A run.ckpt or contig file that
        holds records without it was written by 92b3daf (same containers, other record schemas under the same keys),
        and so is a run.ckpt with another marker; directories of older layouts are refused by the constructor.
        """
        run = self.run
        try:
            if run.has(LAYOUT_KEY):
                marker = run.get(LAYOUT_KEY)
                if marker == LAYOUT:
                    return
                raise CheckpointMismatch(f"{self.root}: checkpoint layout {marker} is not this release's {LAYOUT}; "
                                         "use a new output root")
            foreign = (run.exists() and bool(run.keys())) or any(self.contig(name).keys()
                                                                  for name in self.contig_names())
        except ValueError as error:   # a container header of another kind/name/format
            raise CheckpointMismatch(f"{self.root} holds checkpoint files this release cannot use ({error}); "
                                     "use a new output root") from None
        if foreign:
            raise CheckpointMismatch(f"{self.root} holds checkpoints written by an earlier release (no layout "
                                     "marker in run.ckpt); use a new output root")

    def bind_layout(self) -> None:
        """check_layout, then mark run.ckpt with this release's layout (a run calls it before writing anything)."""
        self.check_layout()
        if not self.run.has(LAYOUT_KEY):
            self.run.put_once(LAYOUT_KEY, LAYOUT)

    def bind_manifest(self, key: str, payload: dict) -> None:
        """Publish one JSON-compatible run manifest exactly once (run.ckpt ``manifest/<key>``)."""
        if not isinstance(payload, dict) or not payload:
            raise ValueError("a run manifest must be a nonempty dictionary")
        try:
            self.run.put_once(f"manifest/{key}", jsonable(payload))
        except CheckpointMismatch:
            raise CheckpointMismatch(
                f"manifest '{key}' does not match the current run; use a new output root") from None

    def manifest(self, key: str) -> dict | None:
        """The run manifest ``manifest/<key>`` (e.g. "run": kind, contigs, sample_ids, source), or None."""
        run = self.run
        name = f"manifest/{key}"
        return run.get(name) if run.has(name) else None

    def stage_complete(self, stage: str) -> bool:
        return self.run.has(f"complete/{stage}")

    def mark_stage_complete(self, stage: str) -> None:
        if not self.stage_complete(stage):
            self.run.put_once(f"complete/{stage}", True)
            print(f"  [Checkpoint] Stage '{stage}' marked complete", flush=True)

    def require(self, key: str, contigs) -> None:
        """Require ``key`` in every listed contig's container (OSError naming the ones lacking it)."""
        missing = [str(contig) for contig in contigs if not self.contig(contig).has(key)]
        if missing:
            raise OSError(f"checkpoint record {key!r} is missing for: " + ", ".join(missing))

    def work(self, contig, name) -> "WorkState":
        """Transient resume-only state of one contig step (see WorkState)."""
        return WorkState(self.root, contig, name, nthreads=self.nthreads)


class WorkState:
    """Resume-only state of one contig step: ``work/<contig>/<name>.p5.b2``.

    Saved by atomic, durable replacement; cleared (with the directories it leaves empty) once its step's
    product is recorded, so a completed run has no ``work/`` directory.
    """

    def __init__(self, root, contig, name, *, nthreads=1):
        self.root = os.fspath(root)
        self.contig = _path_token(contig, "contig")
        self.name = _path_token(name, "work state name")
        self.nthreads = codec.thread_setting(nthreads)

    @property
    def path(self) -> str:
        return os.path.join(self.root, WORK_DIRECTORY, self.contig, self.name + codec.PAYLOAD_SUFFIX)

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def save(self, payload) -> int:
        """Replace the state atomically; return its size in bytes."""
        return codec.write_payload(self.path, payload, nthreads=codec.cpu_threads(self.nthreads),
                                   clevel=codec.FAST_CLEVEL)

    def load(self):
        """The saved state, or None when there is none."""
        try:
            return codec.read_payload(self.path, nthreads=codec.cpu_threads(self.nthreads))
        except FileNotFoundError:
            return None

    def clear(self) -> None:
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        directory = os.path.dirname(self.path)
        for path in (directory, os.path.dirname(directory)):
            try:
                os.rmdir(path)
            except OSError:
                return


class RecordScope:
    """The records under ``prefix`` of one container, bound to one identity before anything is written.

    Cheap and picklable: pool workers record through pickled copies (each record is one append, safe across
    processes). ``bind`` records ``identity/<prefix>``; ``scoped`` narrows to ``<prefix>/<sub>`` under the same
    binding. Records are written at ``clevel`` (FAST_CLEVEL by default: transient or superseded state) with
    ``nthreads`` compression threads (the file's default when None).
    """

    def __init__(self, file: CheckpointFile, prefix: str, *, nthreads=None, clevel=codec.FAST_CLEVEL):
        if not prefix or prefix.startswith("/") or prefix.endswith("/"):
            raise ValueError(f"invalid record prefix: {prefix!r}")
        self.file = file
        self.prefix = prefix
        self.nthreads = nthreads
        self.clevel = clevel
        # Canonical JSON of the bound identity (None until bound).
        self.identity = None

    def __repr__(self):
        return f"RecordScope({self.file.label!r}, {self.prefix!r})"

    def bind(self, identity) -> None:
        """Bind this scope to ``identity``; a different bound identity is a CheckpointMismatch."""
        identity = jsonable(identity)
        try:
            self.file.put_once(f"identity/{self.prefix}", identity)
        except CheckpointMismatch:
            raise CheckpointMismatch(
                f"{self.file.label}: checkpoint identity '{self.prefix}' does not match the current "
                "model/configuration; use a new output root") from None
        self.identity = canonical_json(identity)

    def scoped(self, sub: str, *, nthreads=None) -> "RecordScope":
        """The records under ``<prefix>/<sub>``, written under this binding."""
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
        if self.identity is None:
            raise RuntimeError(f"{self!r} must be bound before it records anything")
        return self.file.put(self.key(name), payload, nthreads=self.nthreads, clevel=self.clevel)

    def keys(self, prefix: str = "") -> tuple[str, ...]:
        """Names (relative to this scope) of the live records starting with ``prefix``."""
        start = f"{self.prefix}/"
        return tuple(key[len(start):] for key in self.file.keys(start + prefix))
