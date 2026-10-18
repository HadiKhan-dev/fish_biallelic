"""Per-invocation provenance and non-overlapping wall-time accounting (OUTPUT/run_records/<stamp>_<pid>.json).

Only the coordinating process writes the record. Stage timings nest per thread: a stage's exclusive time
excludes the stages timed inside it on the same thread, and stages timed on helper threads (e.g. the evidence
stage preparing the next contig while discovery runs) are recorded with their thread's name instead of being
dropped. Unclassified time is the wall time not covered by main-thread stages. Worker processes collect their
stage rows with `worker_stages()` and return them to the coordinator. Records describe completed work and
resume overhead, not hypothetical cold runs.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
from datetime import datetime, timezone
from functools import wraps
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time

from .resources import affinity_cpus

DEPENDENCIES = ("numpy", "numba", "llvmlite", "scipy", "pandas", "cyvcf2", "blosc2", "matplotlib",
                "networkx", "tqdm", "openpyxl")

_ACTIVE = None


def _git_commit(directory: Path) -> str | None:
    """The checked-out commit of the work tree containing ``directory`` (read from .git, no subprocess)."""
    for parent in (directory, *directory.parents):
        git = parent / ".git"
        if git.is_file():       # a worktree: "gitdir: <path>"
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
    return dict(package=str(PACKAGE_ROOT), package_sha256=source_digest(),
                git_commit=_git_commit(PACKAGE_ROOT.parent))


def environment_record() -> dict:
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), executable=sys.executable,
                packages=versions)


def _thread_label() -> str:
    thread = threading.current_thread()
    return "main" if thread is threading.main_thread() else thread.name


class RunRecord:
    """The run record of one invocation (see the module docstring)."""

    def __init__(self, output, command: str, log, *, config: dict | None = None, checkpoints=None):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.path = Path(output) / "run_records" / f"{stamp}_{os.getpid()}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pid, self.start = os.getpid(), time.monotonic()
        self.stacks: dict[int, list[dict]] = {}
        self.lock = threading.RLock()
        self.checkpoints = None if checkpoints is None else str(checkpoints)
        self.data = dict(schema="run-record-v2", command=command, state="running",
                         started_utc=datetime.now(timezone.utc).isoformat(), pid=self.pid,
                         host=platform.node(), log=str(log), argv=sys.argv, cpu_affinity=affinity_cpus(),
                         config=config or {}, source=source_record(), environment=environment_record(),
                         stages=[])
        from .memory import _GOVERNOR
        if _GOVERNOR is not None:
            print(f"[memory] {_GOVERNOR.describe()}", flush=True)
        self.publish()

    def publish(self):
        with self.lock:
            now = time.monotonic()
            self.data["updated_utc"] = datetime.now(timezone.utc).isoformat()
            self.data["wall_seconds"] = now - self.start
            self.data["active_stages"] = [
                dict(stage=frame["stage"], contig=frame["contig"], thread=frame["thread"],
                     elapsed_seconds=now - frame["start"])
                for stack in self.stacks.values() for frame in stack]
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.data, indent=2) + "\n")
            temporary.replace(self.path)

    @contextmanager
    def stage(self, name: str, contig: str | None):
        ident = threading.get_ident()
        frame = dict(stage=name, contig=contig, thread=_thread_label(), start=time.monotonic(),
                     child_seconds=0.)
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
                self.data["stages"].append(dict(
                    stage=name, contig=contig, thread=frame["thread"], state=outcome, inclusive_seconds=duration,
                    exclusive_seconds=max(0., duration - frame["child_seconds"])))
            self.publish()

    def current_contig(self) -> str | None:
        with self.lock:
            stack = self.stacks.get(threading.get_ident())
            return stack[-1]["contig"] if stack else None

    def add_product(self, name: str, contig, summary) -> None:
        with self.lock:
            self.data.setdefault("products", {}).setdefault(name, {})[str(contig)] = copy.deepcopy(summary)
        self.publish()

    def finish(self, error=None):
        with self.lock:
            self.data["state"] = "failed" if error else "complete"
            self.data["finished_utc"] = datetime.now(timezone.utc).isoformat()
            if error is not None:
                self.data["error"] = f"{type(error).__name__}: {error}"
            if self.checkpoints and os.path.exists(os.path.join(self.checkpoints, "run.ckpt")):
                from ..store.container import CheckpointFile
                run = CheckpointFile(os.path.join(self.checkpoints, "run.ckpt"), kind="run")
                self.data["checkpoint_identities"] = {
                    key[len("identity/"):]: run.get(key) for key in run.keys("identity/")}
            main = sum(row["exclusive_seconds"] for row in self.data["stages"] if row["thread"] == "main")
            self.data["unclassified_seconds"] = max(0., time.monotonic() - self.start - main)
            self._record_memory()
        self.publish()

    def _record_memory(self):
        """The memory governor's pacing statistics (execution.memory), also printed to the log."""
        from .memory import _GOVERNOR
        if _GOVERNOR is None:
            return
        self.data["memory"] = _GOVERNOR.statistics()
        for line in _GOVERNOR.summary_lines():
            print(line, flush=True)


@contextmanager
def record_run(output, command: str, log, *, config: dict | None = None, checkpoints=None):
    """Make a RunRecord the active record of this process for the duration of a command."""
    global _ACTIVE
    previous = _ACTIVE
    record = RunRecord(output, command, log, config=config, checkpoints=checkpoints)
    _ACTIVE = record
    try:
        yield record
    except BaseException as error:
        clean_stop = isinstance(error, SystemExit) and error.code in (None, 0)
        record.finish(None if clean_stop else error)
        raise
    else:
        record.finish()
    finally:
        _ACTIVE = previous


class _WorkerStages(RunRecord):
    """Stage timer of a worker process; it writes nothing to the run directory."""

    def __init__(self):
        self.pid, self.start = os.getpid(), time.monotonic()
        self.stacks = {}
        self.lock = threading.RLock()
        self.checkpoints = None
        self.data = dict(stages=[])

    def publish(self):
        pass


@contextmanager
def worker_stages():
    """Collect a worker's timed stages (yielded list); its coordinator records them."""
    global _ACTIVE
    previous = _ACTIVE
    record = _WorkerStages()
    _ACTIVE = record
    try:
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
