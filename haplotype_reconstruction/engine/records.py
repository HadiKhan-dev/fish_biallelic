"""Record invocation provenance, task resources and failure state atomically.

ENG-30 retains nested inclusive/exclusive stage timers and the stdout tee.
RunRecord is an immutable snapshot; RunRecorder owns the atomic JSON writer.
Neither elapsed time, diagnostics nor output paths enter scientific identities.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import platform
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import IO, Any, TypedDict

from ..core.digest import canonical_json, jsonable
from ..core.files import atomic_path
from ..store.releases import CURRENT_RELEASE, require_current_release, unsupported_release
from .runtime import affinity_cpus

_LOG = logging.getLogger(__name__)
__all__ = [
    "source_record",
    "environment_record",
    "RunRecord",
    "RunRecorder",
    "record_run",
    "record_diagnostic",
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
_DIAGNOSTIC_LOCK = threading.RLock()
_DIAGNOSTIC_CAPTURE: tuple[int, list[str]] | None = None


@contextmanager
def capture_diagnostics():
    """Collect a task's small human summaries for its coordinator's run log."""
    global _DIAGNOSTIC_CAPTURE
    previous = _DIAGNOSTIC_CAPTURE
    messages: list[str] = []
    _DIAGNOSTIC_CAPTURE = (os.getpid(), messages)
    try:
        yield messages
    finally:
        _DIAGNOSTIC_CAPTURE = previous


def record_diagnostic(message: str) -> None:
    """Emit a stage summary here, or return it with a worker's existing reply."""
    with _DIAGNOSTIC_LOCK:
        if _DIAGNOSTIC_CAPTURE is not None and _DIAGNOSTIC_CAPTURE[0] == os.getpid():
            _DIAGNOSTIC_CAPTURE[1].append(message)
            return
    _LOG.info("%s", message)


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


class ThreadUsage:
    """Integrate a task's claimed CPUs between its existing allowance boundaries.

    Claims include CPUs retained by running children. This measures allocation,
    like threads_start/threads_peak, rather than native threads doing useful work.
    The histogram is bounded by the task's CPU ceiling, regardless of its duration.
    """

    def __init__(self, threads: int, *, start: float) -> None:
        self.threads = threads
        self.changed = start
        self.seconds: dict[int, float] = {}
        self.lock = threading.Lock()

    def observe(self, threads: int) -> None:
        with self.lock:
            if threads != self.threads:
                self._close(time.perf_counter())
                self.threads = threads

    def _close(self, now: float) -> None:
        self.seconds[self.threads] = self.seconds.get(self.threads, 0.0) + now - self.changed
        self.changed = now

    def finish(self, stop: float) -> tuple[tuple[int, float], ...]:
        with self.lock:
            self._close(stop)
            return tuple(sorted(self.seconds.items()))


def task_log_path(record: Path) -> Path:
    """The append-only task log beside a run record."""
    return record.with_suffix(".tasks.jsonl")


def record_tasks(record: Path, data: dict | None = None) -> list[dict]:
    """Read detailed tasks from this release's retained JSONL log.

    A partly written final line from a running or interrupted invocation is skipped.
    Status uses task_summaries instead and never opens this log.
    """
    data = json.loads(record.read_text()) if data is None else data
    require_current_release(data, record)
    if data.get("schema") != "run-record-v5" or "tasks" in data:
        raise unsupported_release(record, data.get("release"))
    try:
        lines = task_log_path(record).read_text().splitlines()
    except FileNotFoundError:
        return []  # A current invocation creates its task log only on its first task.
    rows = []
    for number, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines) - 1:
                raise
    return rows


def _add_task_summary(summaries: dict, entry: dict) -> None:
    """One constant-size resource total per stage, including failed task outcomes."""
    row = summaries.setdefault(
        entry["key"]["stage"],
        dict(tasks=0, wall_seconds=0.0, cpu_seconds=0.0, publish_lock_wait_seconds=0.0,
             publish_fsync_seconds=0.0, peak_bytes=None, threads_start=0, threads_peak=0, failures=0),
    )
    row["tasks"] += 1
    row["failures"] += int(entry.get("state") == "failed")
    for name in ("wall_seconds", "cpu_seconds", "publish_lock_wait_seconds", "publish_fsync_seconds"):
        value = entry.get(name)
        if value is not None:
            row[name] += value
    for name in ("peak_bytes", "threads_start", "threads_peak"):
        value = entry.get(name)
        if value is not None:
            row[name] = value if row[name] is None else max(row[name], value)


def task_summaries(data: dict) -> dict[str, dict]:
    """Read this release's bounded per-stage aggregates without opening task detail."""
    require_current_release(data, "invocation record")
    if data.get("schema") != "run-record-v5" or "tasks" in data:
        raise unsupported_release("invocation record", data.get("release"))
    return data["task_summaries"]



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

    Task detail stays in the append-only ``*.tasks.jsonl`` log after completion.
    The main record retains only per-stage resource aggregates and is rewritten at
    most every ``PUBLISH_SECONDS`` (a timer writes the last change of a burst).
    Both accumulation and task-log append cost a constant amount per task; status
    never reads the task log. Stage timers and products remain invocation summaries.
    """

    PUBLISH_SECONDS = 5.0
    PUBLISH_WAIT_WINDOW = 32  # most recent completed tasks per contig
    PUBLISH_WAIT_MIN_TASKS = 8  # avoid warning on a single short task

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
        self._publish_windows: dict[str, deque[tuple[float, float]]] = {}
        self._publish_warned: set[str] = set()
        self.data = dict(
            schema="run-record-v5",
            release=CURRENT_RELEASE,
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
        self.data["task_count"] = 0
        self.data["task_summaries"] = {}
        self.data["published_identities"] = {}
        from .memory import current_governor

        governor = current_governor()

        if governor is not None:
            _LOG.info(f"[memory] {governor.describe()}")
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
            document = dict(self.data, task_log=self.tasks_path.name)
            with atomic_path(self.path) as temporary:
                temporary.write_text(json.dumps(document, indent=2) + "\n")

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
        from .runtime import RunInterrupted

        outcome = "complete"
        try:
            yield
        except (KeyboardInterrupt, RunInterrupted):
            outcome = "interrupted"
            raise
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
        from .runtime import RunInterrupted

        with self.lock:
            if isinstance(error, (KeyboardInterrupt, RunInterrupted)):
                self.data["state"] = "interrupted"
            elif error is not None:
                self.data["state"] = "failed"
            else:
                self.data["state"] = "clean_stop" if clean_stop else "complete"
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

    def snapshot(self) -> RunRecord:
        with self.lock:
            return RunRecord(self.data["schema"], self.data["state"], canonical_json(jsonable(self.data)))

    def task_finished(self, task, reply) -> None:
        """Scheduler callback, emitted after every complete task/gather boundary; appends one task-log line.

        ``finished_seconds`` is when the coordinator received the reply, in seconds since the run started; with
        ``wall_seconds`` it places the task on the run's timeline (where wall time goes, stage by stage)."""
        with self.lock:
            thread_seconds = reply.thread_seconds
            observed = sum(seconds for _, seconds in thread_seconds)
            entry = dict(
                key=asdict(task.key),
                finished_seconds=round(time.monotonic() - self.start, 3),
                wall_seconds=reply.wall_seconds,
                cpu_seconds=reply.cpu_seconds,
                publish_lock_wait_seconds=reply.publish_lock_wait_seconds,
                publish_fsync_seconds=reply.publish_fsync_seconds,
                threads_start=reply.threads[0],
                threads_peak=reply.threads[1],
                thread_seconds=thread_seconds,
                threads_mean=(sum(threads * seconds for threads, seconds in thread_seconds) / observed
                              if observed else None),
                threads_one_seconds=dict(thread_seconds).get(1, 0.0) if observed else None,
                peak_bytes=reply.peak_bytes,
                working_set_bytes=task.resources.working_set_bytes,
                scratch_bytes=task.resources.scratch_bytes,
                outputs=[asdict(key) for key in task.outputs],
                retired=reply.retire,
            )
            self._append_task(entry)
            self._publish_progress(task.key.scope, entry)
            from .memory import memory_statistics

            self.data["memory"] = memory_statistics() or {}
        self.publish()

    def _publish_progress(self, contig: str, entry: dict) -> None:
        """Warn once per invocation/contig when recent completed tasks mostly waited to publish."""
        if contig == "run" or contig in self._publish_warned:
            return
        window = self._publish_windows.setdefault(contig, deque(maxlen=self.PUBLISH_WAIT_WINDOW))
        window.append((entry["wall_seconds"], entry["publish_lock_wait_seconds"]))
        wall, wait = map(sum, zip(*window, strict=True))
        if len(window) >= self.PUBLISH_WAIT_MIN_TASKS and wait > 0.5 * wall:
            self._publish_warned.add(contig)
            del self._publish_windows[contig]
            _LOG.warning(
                "[progress] %s: checkpoint publish lock wait %.1fs / %.1fs task wall time "
                "(%.0f%% over %d tasks); checkpoint storage is limiting throughput",
                contig, wait, wall, 100 * wait / wall, len(window),
            )

    def _append_task(self, entry: dict) -> None:
        if self._tasks_log is None:
            self._tasks_log = self.tasks_path.open("a", buffering=1)
        self._tasks_log.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self.data["task_count"] += 1
        _add_task_summary(self.data["task_summaries"], entry)

    def task_failed(self, key, error: BaseException) -> None:
        """Record an attributed scheduler failure; no resource measurement is invented."""
        with self.lock:
            self._append_task(dict(
                key=asdict(key), state="failed", error=f"{type(error).__name__}: {error}",
                publish_lock_wait_seconds=getattr(error, "publish_lock_wait_seconds", None),
                publish_fsync_seconds=getattr(error, "publish_fsync_seconds", None),
            ))
        self.publish()

    def published(self, identities: dict[str, str]) -> None:
        with self.lock:
            self.data["published_identities"].update(identities)
        self.publish()

    def _record_memory(self):
        """The memory governor's pacing statistics (engine.scheduler), also printed to the log."""
        from .memory import current_governor

        governor = current_governor()

        if governor is None:
            return
        self.data["memory"] = governor.statistics()
        for line in governor.summary_lines():
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




def _active():
    record = _ACTIVE
    return record if record is not None and record.pid == os.getpid() else None


def record_task_failure(key, error: BaseException) -> None:
    """Attach a known failed task to the coordinator's operational record."""
    record = _active()
    if record is not None:
        record.task_failed(key, error)




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
