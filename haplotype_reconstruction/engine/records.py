"""Record invocation provenance, task resources and failure state atomically.

ENG-30 retains nested inclusive/exclusive stage timers and the stdout tee.
RunRecord is an immutable snapshot; RunRecorder owns the atomic JSON writer.
Neither elapsed time, diagnostics nor output paths enter scientific identities.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
import os
import platform
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import wraps
from importlib import metadata
from pathlib import Path
from typing import IO, Any, TypedDict

from ..core.digest import canonical_json, jsonable
from .runtime import affinity_cpus

_LOG = logging.getLogger(__name__)
__all__ = [
    "source_record",
    "environment_record",
    "RunRecord",
    "RunRecorder",
    "record_run",
    "worker_stages",
    "stage_timer",
    "timed_stage",
    "record_product",
    "TeeOutput",
    "run_log",
]

DEPENDENCIES = (
    "numpy",
    "numba",
    "llvmlite",
    "scipy",
    "pandas",
    "cyvcf2",
    "blosc2",
    "matplotlib",
    "networkx",
    "tqdm",
    "openpyxl",
)
_ACTIVE = None
_ROUTE_CAPTURE: tuple[int, list[dict[str, Any]]] | None = None
_ROUTE_LOCK = threading.RLock()


@contextmanager
def capture_storage_routes():
    """Collect one task's route receipts, including its engine-owned threads."""
    global _ROUTE_CAPTURE
    previous = _ROUTE_CAPTURE
    rows: list[dict[str, Any]] = []
    _ROUTE_CAPTURE = (os.getpid(), rows)
    try:
        yield rows
    finally:
        _ROUTE_CAPTURE = previous


def record_storage_routes(rows) -> None:
    """Publish operational route receipts at the invocation boundary."""
    record = _active()
    if record is not None and rows:
        with record.lock:
            record.data.setdefault("storage_routes", []).extend(copy.deepcopy(rows))
        record.publish()


def record_storage_route(
    *, route: str, shape: tuple[int, ...], dtype: str, consumers: int, window_axis: int, window_width: int
) -> None:
    """Capture only dimensions and route choice; no payload or path identity."""
    row = dict(
        route=route, shape=shape, dtype=dtype, consumers=consumers, window_axis=window_axis, window_width=window_width
    )
    with _ROUTE_LOCK:
        if _ROUTE_CAPTURE is not None and _ROUTE_CAPTURE[0] == os.getpid():
            _ROUTE_CAPTURE[1].append(row)
            return
    record_storage_routes((row,))


def _git_commit(directory: Path) -> str | None:
    """The checked-out commit of the work tree containing ``directory`` (read from .git, no subprocess)."""
    for parent in (directory, *directory.parents):
        git = parent / ".git"
        if git.is_file():
            text = git.read_text().strip()
            if not text.startswith("gitdir:"):
                return None
            git = (parent / text.split(":", 1)[1].strip()).resolve()
        if not git.is_dir():
            continue
        try:
            head = (git / "HEAD").read_text().strip()
            if not head.startswith("ref:"):
                return head
            reference = head.split(":", 1)[1].strip()
            common = git
            if (git / "commondir").is_file():
                common = (git / (git / "commondir").read_text().strip()).resolve()
            for base in (git, common):
                if (base / reference).is_file():
                    return (base / reference).read_text().strip()
            packed = common / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + reference):
                        return line.split()[0]
        except OSError:
            pass
        return None
    return None


def source_record() -> dict:
    from ..store.identity import PACKAGE_ROOT, source_digest

    return dict(package=str(PACKAGE_ROOT), package_sha256=source_digest(), git_commit=_git_commit(PACKAGE_ROOT.parent))


def environment_record() -> dict:
    versions: dict[str, str | None] = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return dict(
        python=platform.python_version(), platform=platform.platform(), executable=sys.executable, packages=versions
    )


def _thread_label() -> str:
    thread = threading.current_thread()
    return "main" if thread is threading.main_thread() else thread.name


def task_log_path(record: Path) -> Path:
    """The append-only task log beside a run record."""
    return record.with_suffix(".tasks.jsonl")


def record_tasks(record: Path, data: dict | None = None) -> list[dict]:
    """The finished tasks of one run record: its task list once final, else its task log (a partly written last
    line, from a command still running, is skipped)."""
    data = json.loads(record.read_text()) if data is None else data
    if "tasks" in data:
        return list(data["tasks"])
    try:
        lines = task_log_path(record).read_text().splitlines()
    except FileNotFoundError:
        # No task yet, or the command finished meanwhile: its final record holds the list (written before the log
        # is deleted).
        return list(json.loads(record.read_text()).get("tasks", ()))
    rows = []
    for number, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines) - 1:
                raise
    return rows


@dataclass(frozen=True)
class RunRecord:
    """An immutable canonical snapshot of one invocation's operational record."""

    schema: str
    state: str
    document_json: str

    def document(self) -> dict[str, Any]:
        return json.loads(self.document_json)


class _StageFrame(TypedDict):
    stage: str
    contig: str | None
    thread: str
    start: float
    child_seconds: float


class RunRecorder:
    """The run record of one invocation (see the module docstring).

    Finished tasks are appended, one compact JSON line each, to the task log beside the record (``*.tasks.jsonl``):
    a constant cost per task. The main record omits the task list while the command runs and is rewritten at
    most every ``PUBLISH_SECONDS`` (a timer writes the last change of a burst); the final record, written when the
    command ends, holds the complete task list, and the task log is then deleted (it survives only a command that
    was killed before finishing). Rewriting the whole record after every task cost time quadratic in the number of
    tasks (bc8: 15,000 tasks, an 8 MB record, about 50 minutes of a 53-minute run).
    """

    PUBLISH_SECONDS = 5.0

    def __init__(
        self, output, command: str, log, *, config: dict | None = None, checkpoints=None, source_descriptor: Any = None
    ):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        self.path = Path(output) / "run_records" / f"{stamp}_{os.getpid()}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tasks_path = task_log_path(self.path)
        self._tasks_log: IO[str] | None = None
        self._published_at: float | None = None
        self._timer: threading.Timer | None = None
        self.pid, self.start = (os.getpid(), time.monotonic())
        self.stacks: dict[int, list[_StageFrame]] = {}
        self.lock = threading.RLock()
        self.checkpoints = None if checkpoints is None else str(checkpoints)
        self._clean_stop = False
        self.data = dict(
            schema="run-record-v4",
            command=command,
            state="running",
            started_utc=datetime.now(UTC).isoformat(),
            pid=self.pid,
            host=platform.node(),
            log=str(log),
            argv=sys.argv,
            cpu_affinity=affinity_cpus(),
            config=jsonable(config or {}),
            source_descriptor=jsonable(source_descriptor),
            source=source_record(),
            environment=environment_record(),
            stages=[],
        )
        from ..store.protocol import NumericalPolicy

        self.data["numerical_policy"] = jsonable(NumericalPolicy.current())
        self.data["tasks"] = []
        self.data["storage_routes"] = []
        self.data["published_identities"] = {}
        from .scheduler import _GOVERNOR

        if _GOVERNOR is not None:
            _LOG.info(f"[memory] {_GOVERNOR.describe()}")
        self.publish(force=True)

    def publish(self, *, force: bool = False):
        """Rewrite the main record: at once when forced, otherwise at most every PUBLISH_SECONDS."""
        with self.lock:
            now = time.monotonic()
            wait = 0.0 if force or self._published_at is None else self._published_at + self.PUBLISH_SECONDS - now
            if wait > 0:
                if self._timer is None:
                    self._timer = threading.Timer(wait, self._publish_due)
                    self._timer.args = (self._timer,)
                    self._timer.daemon = True
                    self._timer.start()
                return
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._published_at = now
            self.data["updated_utc"] = datetime.now(UTC).isoformat()
            self.data["wall_seconds"] = now - self.start
            self.data["active_stages"] = [
                dict(
                    stage=frame["stage"],
                    contig=frame["contig"],
                    thread=frame["thread"],
                    elapsed_seconds=now - frame["start"],
                )
                for stack in self.stacks.values()
                for frame in stack
            ]
            document = dict(self.data, task_log=self.tasks_path.name, task_count=len(self.data["tasks"]))
            if document["state"] == "running":
                del document["tasks"]  # read the task log while the command runs (record_tasks)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(document, indent=2) + "\n")
            temporary.replace(self.path)

    def _publish_due(self, timer: threading.Timer) -> None:
        with self.lock:
            if self._timer is not timer:
                return  # cancelled while it waited for the lock: that publish wrote the record
            self._timer = None
        self.publish()

    @contextmanager
    def stage(self, name: str, contig: str | None):
        ident = threading.get_ident()
        frame: _StageFrame = dict(
            stage=name, contig=contig, thread=_thread_label(), start=time.monotonic(), child_seconds=0.0
        )
        with self.lock:
            self.stacks.setdefault(ident, []).append(frame)
        self.publish()
        outcome = "complete"
        try:
            yield
        except BaseException:
            outcome = "failed"
            raise
        finally:
            duration = time.monotonic() - frame["start"]
            with self.lock:
                stack = self.stacks[ident]
                stack.pop()
                if stack:
                    stack[-1]["child_seconds"] += duration
                else:
                    del self.stacks[ident]
                self.data["stages"].append(
                    dict(
                        stage=name,
                        contig=contig,
                        thread=frame["thread"],
                        state=outcome,
                        inclusive_seconds=duration,
                        exclusive_seconds=max(0.0, duration - frame["child_seconds"]),
                    )
                )
            self.publish()

    def current_contig(self) -> str | None:
        with self.lock:
            stack = self.stacks.get(threading.get_ident())
            return stack[-1]["contig"] if stack else None

    def add_product(self, name: str, contig, summary) -> None:
        with self.lock:
            self.data.setdefault("products", {}).setdefault(name, {})[str(contig)] = copy.deepcopy(summary)
        self.publish()

    def clean_stop(self) -> None:
        """Mark a successful workflow stop; record_run finalizes it exactly once."""
        self._clean_stop = True

    def finish(self, error=None, *, clean_stop: bool = False):
        with self.lock:
            self.data["state"] = "failed" if error else "clean_stop" if clean_stop else "complete"
            self.data["finished_utc"] = datetime.now(UTC).isoformat()
            if error is not None:
                self.data["error"] = f"{type(error).__name__}: {error}"
            main = sum(row["exclusive_seconds"] for row in self.data["stages"] if row["thread"] == "main")
            self.data["unclassified_seconds"] = max(0.0, time.monotonic() - self.start - main)
            self._record_memory()
            if self._tasks_log is not None:
                self._tasks_log.close()
                self._tasks_log = None
        self.publish(force=True)
        # The final record now holds every task; the log is kept only by a command that never reached here.
        self.tasks_path.unlink(missing_ok=True)

    def snapshot(self) -> RunRecord:
        with self.lock:
            return RunRecord(self.data["schema"], self.data["state"], canonical_json(jsonable(self.data)))

    def task_finished(self, task, reply) -> None:
        """Scheduler callback, emitted after every complete task/gather boundary; appends one task-log line.

        ``finished_seconds`` is when the coordinator received the reply, in seconds since the run started; with
        ``wall_seconds`` it places the task on the run's timeline (where wall time goes, stage by stage)."""
        with self.lock:
            entry = dict(
                key=asdict(task.key),
                finished_seconds=round(time.monotonic() - self.start, 3),
                wall_seconds=reply.wall_seconds,
                cpu_seconds=reply.cpu_seconds,
                threads_start=reply.threads[0],
                threads_peak=reply.threads[1],
                peak_bytes=reply.peak_bytes,
                working_set_bytes=task.resources.working_set_bytes,
                scratch_bytes=task.resources.scratch_bytes,
                outputs=[asdict(key) for key in task.outputs],
                retired=reply.retire,
            )
            self.data["tasks"].append(entry)
            if self._tasks_log is None:
                self._tasks_log = self.tasks_path.open("a", buffering=1)  # line-buffered: each line lands whole
            self._tasks_log.write(json.dumps(entry, separators=(",", ":")) + "\n")
            from .scheduler import memory_statistics

            self.data["memory"] = memory_statistics()
        self.publish()

    def published(self, identities: dict[str, str]) -> None:
        with self.lock:
            self.data["published_identities"].update(identities)
        self.publish()

    def _record_memory(self):
        """The memory governor's pacing statistics (execution.memory), also printed to the log."""
        from .scheduler import _GOVERNOR

        if _GOVERNOR is None:
            return
        self.data["memory"] = _GOVERNOR.statistics()
        for line in _GOVERNOR.summary_lines():
            _LOG.info(line)


@contextmanager
def record_run(
    output, command: str, log, *, config: dict | None = None, checkpoints=None, source_descriptor: Any = None
):
    """Make a RunRecord the active record of this process for the duration of a command."""
    global _ACTIVE
    previous = _ACTIVE
    record = RunRecorder(
        output, command, log, config=config, checkpoints=checkpoints, source_descriptor=source_descriptor
    )
    _ACTIVE = record
    from .runtime import signal_boundary

    try:
        with signal_boundary(lambda: None):
            yield record
    except BaseException as error:
        record.finish(error)
        raise
    else:
        record.finish(clean_stop=record._clean_stop)
    finally:
        _ACTIVE = previous


class _WorkerStages(RunRecorder):
    """Stage timer of a worker process; it writes nothing to the run directory."""

    def __init__(self):
        self.pid, self.start = (os.getpid(), time.monotonic())
        self.stacks = {}
        self.lock = threading.RLock()
        self.checkpoints = None
        self.data = dict(stages=[])

    def publish(self, *, force: bool = False):
        pass


@contextmanager
def worker_stages():
    """Collect a worker's timed stages (yielded list); its coordinator records them."""
    global _ACTIVE
    previous = _ACTIVE
    record = _WorkerStages()
    _ACTIVE = record
    from .runtime import signal_boundary

    try:
        with signal_boundary(lambda: None):
            yield record.data["stages"]
    finally:
        _ACTIVE = previous


def _active():
    record = _ACTIVE
    return record if record is not None and record.pid == os.getpid() else None


def stage_timer(name: str, contig=None):
    """Time a block as stage ``name`` (contig defaults to the enclosing stage's); no-op outside a run."""
    record = _active()
    if record is None:
        return nullcontext()
    if contig is None:
        contig = record.current_contig()
    return record.stage(name, None if contig is None else str(contig))


def timed_stage(name: str, contig_arg: str | None = None):
    """Decorator: time a function as stage ``name``; ``contig_arg`` names the argument holding the contig."""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            if _active() is None:
                return function(*args, **kwargs)
            contig = None
            if contig_arg:
                contig = signature.bind_partial(*args, **kwargs).arguments.get(contig_arg)
            with stage_timer(name, contig):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def record_product(name: str, contig, summary) -> None:
    """Attach a small JSON-compatible summary of one contig's product to the active run record."""
    record = _active()
    if record is not None:
        record.add_product(name, contig, summary)


class TeeOutput:
    """Mirror writes to the original stream and a line-buffered log, proxying other stream attributes."""

    def __init__(self, log_path, original):
        object.__setattr__(self, "_log_file", open(log_path, "a", buffering=1))
        object.__setattr__(self, "_original", original)

    def write(self, message):
        self._original.write(message)
        try:
            self._log_file.write(message)
        except (ValueError, OSError):
            pass
        return None

    def flush(self):
        self._original.flush()
        try:
            self._log_file.flush()
        except (ValueError, OSError):
            pass

    def close(self):
        self._log_file.close()

    def __getattr__(self, name):
        return getattr(self._original, name)


@contextmanager
def run_log(log_path):
    """Mirror stdout into ``log_path`` (appending) and restore it on exit, also on failure."""
    original = sys.stdout
    stream = TeeOutput(log_path, original)
    sys.stdout = stream
    logger = logging.getLogger("haplotype_reconstruction.engine")
    handler = logging.StreamHandler(stream)
    previous_level, previous_propagation = logger.level, logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagation
        sys.stdout = original
        stream.flush()
        stream.close()
