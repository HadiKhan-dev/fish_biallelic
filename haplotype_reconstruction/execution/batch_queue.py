"""Opt-in shared-filesystem queue that lets helper allocations run independent batch bundles of a stage.

Used only when the run sets ``[execution] batch_queue`` (a shared directory); the driver passes it to the stages
that support it. Only the chromosome coordinator records results in the checkpoint store. Helpers return
immutable results under attempt-specific names; a renewable claim fences late writers after an interruption.
No scheduler submission happens here.

A stage describes its work as a `BatchKind`: ``bundles(tasks, arrays, worker_memory_gb)`` packs tasks with the
slices of the shared arrays they need into payload dicts (never splitting a scientific task), and
``execute(payload, cpus)`` (module level) runs one bundle in the executing allocation (typically through a
BudgetedPool) and returns its task results. Lifecycle: `map_batches` publishes bundles -> helpers (`serve`, the
``batch-worker`` command) and the coordinator itself claim and execute them -> the coordinator yields results.

Transport files are temporary: a consumed bundle's input is deleted, and a completed stage leaves neither files
nor records. Until then a keyed stage (named by an exact scientific identity, `queue_key`) keeps its results,
and its failed bundles their inputs and tracebacks, for a later attempt, which retries those bundles. A stage
without a key is removed as soon as its attempt stops.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import threading
import time
import traceback
from typing import Callable
import uuid

from ..store.codec import read_payload, write_payload
from ..store.identity import source_digest
from .budget import as_budget

LEASE_SECONDS = 600
HEARTBEAT_SECONDS = 20
IDLE_POLL_SECONDS = 2
_PACKAGE = __name__.split(".")[0]


@dataclass(frozen=True)
class BatchKind:
    """A stage's batch work: how to pack tasks into bundles and how to execute one bundle (module level)."""

    name: str
    bundles: Callable
    execute: Callable


def _reference(function) -> str:
    reference = f"{function.__module__}:{function.__qualname__}"
    if not function.__module__.startswith(_PACKAGE + ".") or "<locals>" in function.__qualname__:
        raise ValueError(f"batch executors must be module-level functions of the package, not {reference}")
    return reference


def _resolve(reference: str):
    module, _, qualname = reference.partition(":")
    if not module.startswith(_PACKAGE + "."):
        raise ValueError(f"refusing to execute {reference!r} from a batch queue")
    value = importlib.import_module(module)
    for part in qualname.split("."):
        value = getattr(value, part)
    return value


def queue_key(scope, phase: str) -> str | None:
    """Stage key of one phase of a bound RecordScope (None without one): results are reused only under that
    exact scientific identity."""
    if scope is None or scope.identity is None:
        return None
    value = (scope.identity, scope.file.name, scope.prefix, phase)
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def _write_json(path: Path, data) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def _state(root: Path):
    """Serialise short metadata updates; numerical work never holds this lock."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = root / "queue.json"
        data = json.loads(path.read_text()) if path.exists() else dict(stages={}, bundles={})
        try:
            yield data
        finally:
            _write_json(path, data)
            fcntl.flock(handle, fcntl.LOCK_UN)


def _remove_stage(state, stage):
    """Forget a stage and its bundles; return the removed bundle records."""
    state["stages"].pop(stage, None)
    keys = [key for key, record in state["bundles"].items() if record["stage"] == stage]
    return [state["bundles"].pop(key) for key in keys]


def _claim(root: Path, stage=None):
    """Claim ready work, or reclaim a bundle whose worker stopped heartbeating.

    A stage whose coordinator's allocation has ended stops; without a key it is removed with its files.
    """
    claim, ended = None, []
    try:
        with _state(root) as state:
            now = time.time()
            for name, parent in list(state["stages"].items()):
                if parent.get("owner_expiry") and now > parent["owner_expiry"]:
                    parent["state"] = "stopped"
                    if not parent.get("resumable", True):
                        _remove_stage(state, name)
                        ended.append(name)
            for key, record in state["bundles"].items():
                if stage is not None and record["stage"] != stage:
                    continue
                parent = state["stages"][record["stage"]]
                if parent["state"] != "active":
                    continue
                if record["state"] == "running" and now - record["heartbeat"] > LEASE_SECONDS:
                    record["state"] = "ready"
                if record["state"] != "ready":
                    continue
                if parent["source"] != source_digest():
                    raise RuntimeError("batch worker source differs from its coordinator")
                token = uuid.uuid4().hex
                record.update(state="running", token=token, heartbeat=now, host=socket.gethostname(),
                              pid=os.getpid(), job=os.environ.get("SLURM_JOB_ID"), started=now)
                claim = key, dict(record)
                break
    finally:
        for name in ended:
            shutil.rmtree(root / name, ignore_errors=True)
    return claim


def _finish(root: Path, key, token, result_path: Path, outcome) -> bool:
    """Publish a claim's terminal state; return False if the claim was lost (its result is then removed)."""
    with _state(root) as state:
        current = state["bundles"].get(key)
        if current is not None and current.get("token") == token:
            current.update(outcome)
            return True
        result_path.unlink(missing_ok=True)
        if result_path.parent.name not in state["stages"]:
            try:
                result_path.parent.rmdir()
            except OSError:
                pass
    return False


def serve_one(root, cores: int, *, stage=None) -> bool:
    """Use this allocation for one ready bundle; return False when none is ready.

    A failure under a lost claim is not the bundle's failure (the current claim or the coordinator decides it),
    so the allocation keeps serving.
    """
    root = Path(root)
    claim = _claim(root, stage)
    if claim is None:
        return False
    key, record = claim
    token = record["token"]
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(HEARTBEAT_SECONDS):
            with _state(root) as state:
                current = state["bundles"].get(key)
                if current is None or current.get("token") != token:
                    return
                current["heartbeat"] = time.time()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    result_path = root / record["stage"] / f'{record["number"]:06d}.{token}.result.p5.b2'
    started = time.monotonic()
    try:
        payload = read_payload(str(root / record["input"]), nthreads=cores)
        results = _resolve(payload["executor"])(payload, cores)
        write_payload(str(result_path), results, nthreads=cores)
    except BaseException as error:
        held = _finish(root, key, token, result_path, dict(
            state="failed", error=traceback.format_exc(), seconds=time.monotonic() - started, cores=cores))
        if held or not isinstance(error, Exception):
            raise
    else:
        _finish(root, key, token, result_path, dict(
            state="done", result=str(result_path.relative_to(root)), seconds=time.monotonic() - started,
            cores=cores, finished=time.time()))
    finally:
        stop.set()
        thread.join()
    return True


def map_batches(kind: BatchKind, tasks, arrays, budget, *, queue, key=None, worker_memory_gb=4.0):
    """Yield the tasks' results (as ``kind.execute`` returns them), the coordinator helping its own queue.

    ``queue`` is the shared directory; ``key`` (see queue_key) names a stage a later attempt can resume: its
    consumed results stay until the stage completes, and bundles that failed in an earlier attempt are published
    afresh. A failure within this attempt stops it. Without a key, results go once consumed and a stopped stage
    at once; helpers still running its bundles then find their claims gone.
    """
    root = Path(queue)
    cores = as_budget(budget).threads()
    executor = _reference(kind.execute)
    stage = key or uuid.uuid4().hex
    directory = root / stage
    directory.mkdir(parents=True, exist_ok=True)
    with _state(root) as state:
        existing = state["stages"].get(stage)
        if existing and existing["source"] != source_digest():
            raise RuntimeError("queued stage belongs to different source code")
        state["stages"][stage] = dict(
            source=source_digest(), state="active", kind=kind.name, resumable=key is not None,
            owner_job=os.environ.get("SLURM_JOB_ID"), owner_expiry=int(os.environ.get("SLURM_JOB_END_TIME", "0")),
            host=socket.gethostname(), pid=os.getpid())
    tasks = list(tasks)
    keys, retried = [], 0
    try:
        for number, payload in enumerate(kind.bundles(tasks, arrays, worker_memory_gb)):
            task_key = f"{stage}/{number:06d}"
            keys.append(task_key)
            with _state(root) as state:
                record = state["bundles"].get(task_key)
            if record is not None:
                if record["state"] != "failed":
                    continue
                retried += 1
            filename = directory / f"{number:06d}.input.p5.b2"
            write_payload(str(filename), dict(payload, executor=executor), nthreads=cores)
            with _state(root) as state:
                state["bundles"][task_key] = dict(stage=stage, number=number, state="ready",
                                                  input=str(filename.relative_to(root)),
                                                  tasks=len(payload["tasks"]))
        print(f"[Cross-node {kind.name}] published {len(keys)} bundles / {len(tasks)} tasks"
              + (f"; retrying {retried} that failed in an earlier attempt" if retried else ""), flush=True)
        pending = set(keys)
        while pending:
            with _state(root) as state:
                records = {item: dict(state["bundles"][item]) for item in keys if item in pending}
            for item, record in records.items():
                if record["state"] == "failed":
                    raise RuntimeError(f"cross-node batch failed: {item}\n{record['error']}")
                if record["state"] == "done":
                    for result in read_payload(str(root / record["result"]), nthreads=cores):
                        yield result
                    pending.remove(item)
                    (root / record["input"]).unlink(missing_ok=True)
                    if key is None:
                        (root / record["result"]).unlink(missing_ok=True)
            # The coordinator can help another chromosome while its own last bundles run elsewhere.
            if pending and not serve_one(root, cores, stage=stage) and not serve_one(root, cores):
                time.sleep(IDLE_POLL_SECONDS)
    except BaseException:
        with _state(root) as state:
            if key is None:
                _remove_stage(state, stage)
            else:
                state["stages"][stage]["state"] = "stopped"
        if key is None:
            shutil.rmtree(directory, ignore_errors=True)
        raise
    # Records go first: an interruption in between can leave files without records (a keyed stage's next
    # attempt overwrites or removes them), never a done record without its result.
    with _state(root) as state:
        records = _remove_stage(state, stage)
    shutil.rmtree(directory, ignore_errors=True)
    executors = Counter(f'{record.get("host")}:{record.get("pid")}' for record in records)
    print(f"[Cross-node {kind.name}] {len(records)} bundles complete, by {dict(executors)}; "
          "transport files removed", flush=True)


def serve(queue, cores: int, idle_seconds: float = 120) -> None:
    """The batch-worker command: serve bundles with this allocation until the queue has been idle for
    ``idle_seconds`` (or two minutes before the allocation's end)."""
    from .resources import affinity_cpus
    if not 1 <= cores <= affinity_cpus():
        raise ValueError("batch worker cores must fit its CPU affinity")
    last_work = time.monotonic()
    while time.monotonic() - last_work < idle_seconds:
        expiry = int(os.environ.get("SLURM_JOB_END_TIME", "0"))
        if expiry and time.time() > expiry - 120:
            break
        if serve_one(queue, cores):
            last_work = time.monotonic()
        else:
            time.sleep(IDLE_POLL_SECONDS)
