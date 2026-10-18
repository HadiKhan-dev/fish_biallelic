"""Concurrent contig lanes: independent contigs reconstructed side by side within one CPU ceiling.

Lane coordinators are persistent ChromosomeExecutor workers (non-daemonic forkserver processes) sharing the run's
cores as leases. Each runs the per-contig work (a module-level function ``work(store, contig, context, budget)``)
for its contigs, longest first, with its lease as the budget: pools, Numba teams and checkpoint compression inside
a lane stay within it, so concurrent contigs never exceed the ceiling. Leases start in proportion to each lane's
work and grow, never inside a running kernel, as finished lanes release theirs: at contig starts, at the phase
starts the stages mark (budget.phase), and inside pool phases (pools re-read their live budget every second).

The plan: one lane per CONTIG_LANE_CORES cores (or the requested count), at most one per pending contig, then
limited by memory: while a lane works on a contig it holds LANE_RESIDENT_FACTOR x the contig's planning bytes of
the run's memory (execution.memory; the coordinator's resident contig data), and the lanes of the largest
concurrent group may hold at most LANE_RESIDENT_SHARE of the memory limit. Everything else a lane runs (pools,
executors, concurrent fits) reserves its own bytes as it starts, so memory paces the work inside lanes. A contig's
computation never reads another contig's work in progress, and core counts and pacing are execution choices
outside every scientific identity.

The first failing contig stops the run: the other lanes are stopped (SIGTERM) and keep their contigs' records for
resume. A lane also stops as soon as the main process exits for any reason, so no lane keeps computing or writing
without an owner; a stop that Python drops inside a callback (as during Numba compilation) is re-armed and sent
again.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import faulthandler
import functools
import multiprocessing
import os
import re
import signal
import sys
import threading
import time
import traceback

from .budget import CpuBudget
from .chromosome_workers import ChromosomeExecutor
from .memory import headroom_bytes, memory_governor, release_memory
from .run_log import run_log
from .run_record import record_product, timed_stage, worker_stages

# Narrow lanes need fewer core-seconds: the same contig at 16 threads took 7% fewer than at 32-37 (N=320 chr19 and
# chr8 shards; path selection scales fully, hierarchy, refinement tails and serial checkpoint phases do not).
CONTIG_LANE_CORES = 16

# Resident bytes of a lane per planning byte of its contig (contig_work.planning_bytes: discovery records, two
# genotype-likelihood copies and the observation mask): the coordinator's decoded evidence and records. Its pools,
# executors and concurrent tasks reserve their own memory (shared arrays included) as they start.
LANE_RESIDENT_FACTOR = 1.0
# The resident bytes of concurrent lanes may take at most this share of the memory limit (the rest is for the work
# inside the lanes, which the governor paces).
LANE_RESIDENT_SHARE = 0.5

# Planning sizes come from record indexes that wait on shared-filesystem latency: contigs are scanned concurrently.
SIZE_SCAN_THREADS = 16

_FAILURE = "contig lane failed on"


def scan_concurrently(function, items):
    """[function(item) for item in items], with up to SIZE_SCAN_THREADS concurrent calls (I/O latency)."""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(min(SIZE_SCAN_THREADS, len(items))) as scanner:
        return list(scanner.map(function, items))


def lane_resident_bytes(size) -> int:
    """Bytes a lane holds while it works on a contig of ``size`` planning bytes."""
    return int(LANE_RESIDENT_FACTOR * float(size))


def plan_lanes(sizes, pending, *, cores: int, requested: int | None, available: int | None = None):
    """Concurrent lanes and longest-first work weights per contig.

    ``sizes``: planning bytes per contig; ``pending``: whether each contig still needs work. Automatic: one lane
    per CONTIG_LANE_CORES cores, at most one per pending contig; memory then limits the concurrent group so that
    its lanes' resident bytes fit LANE_RESIDENT_SHARE of the run's memory limit (or of ``available`` minus the
    headroom when given); it never lowers the core ceiling. Returns (1, None) when lanes would not help.
    """
    count = sum(bool(flag) for flag in pending)
    limit = max(1, cores // CONTIG_LANE_CORES) if requested is None else requested
    lanes = min(count, limit, cores)
    if lanes < 2:
        return 1, None
    weights = [float(size) if flag else 0. for size, flag in zip(sizes, pending)]
    residents = sorted((lane_resident_bytes(size) for size, flag in zip(sizes, pending) if flag), reverse=True)
    memory = memory_governor().limit if available is None else max(0, available - headroom_bytes(available))
    allowance = LANE_RESIDENT_SHARE * memory
    group = total = 0
    for resident in residents[:lanes]:
        if group and total + resident > allowance:
            break
        group += 1
        total += resident
    print(f"[memory] RECONSTRUCTION: {group} concurrent lanes (of {lanes} by cores); resident bytes of the "
          f"largest group {total / 2**30:.1f} GiB of {allowance / 2**30:.1f} GiB (half the limit)", flush=True)
    return (group, weights) if group >= 2 else (1, None)


def run_lanes(store, contigs, work, context, *, cores: int, lanes: int, weights, log_directory, phase_progress,
              done=None, label="reconstruction"):
    """Run ``work(store, contig, context, budget)`` for every contig in ``lanes`` concurrent lanes.

    Returns the work results in contig order. Each lane writes its contig's stdout to
    ``log_directory/<label>_<contig>_<stamp>_<pid>.log``; the coordinator records one run-record product per
    contig with the lane's timed stages. ``done(contig) -> bool`` names unfinished contigs after a failure.
    """
    contigs = tuple(contigs)
    executor = ChromosomeExecutor(contigs, _lane_contig, budget=cores, weights=weights, max_workers=lanes,
                                  label=label.upper(), phase_progress=phase_progress,
                                  task_bytes=None if weights is None else [lane_resident_bytes(w) for w in weights])
    if executor.worker_count < 2:
        return run_sequentially(store, contigs, work, context, cores=cores)
    os.makedirs(log_directory, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logs = {contig: os.path.join(log_directory, f"{label}_{contig}_{stamp}_{os.getpid()}.log")
            for contig in contigs}
    partitions = [[contig for _, contig in partition] for partition in executor.partitions]
    print(f"  Contigs: {executor.worker_count} concurrent lanes share {executor.cpu_budget} cores (leases in "
          "proportion to work, growing at contig and phase starts and inside pool phases); longest first per "
          "lane: " + "; ".join(", ".join(partition) for partition in partitions), flush=True)
    for contig in contigs:
        print(f"    {contig} log: {logs[contig]}", flush=True)
    common = dict(root=store.root, work=work, context=context, logs=logs)
    try:
        results = _run_executor(executor, common)
    except RuntimeError as error:
        message = _failure_message(str(error), partitions, logs, done or (lambda contig: False))
        print(f"  [FAILED] {message}", flush=True)
        raise RuntimeError(message) from error
    summaries = []
    for contig, result in zip(contigs, results):
        summaries.append(result["summary"])
        print(f"  [done] {contig}: {result['threads']}-core lease, {result['wall_seconds']:.1f} s; "
              f"log {result['log']}", flush=True)
        record_product(label, contig, dict(threads=int(result["threads"]), wall_seconds=float(result["wall_seconds"]),
                                           log=result["log"], pid=int(result["pid"]), stages=result["stages"]))
    return summaries


def run_sequentially(store, contigs, work, context, *, cores: int):
    """The one-lane route: every contig in this process with the whole ceiling (freed heap pages are returned
    after each contig, as the lane executor does after each of its tasks)."""
    print(f"  Contigs run sequentially with a {cores}-core ceiling", flush=True)
    budget = CpuBudget.fixed(cores)
    results = []
    for contig in contigs:
        results.append(work(store, contig, context, budget))
        release_memory()
    return results


@timed_stage("contig_lanes")
def _run_executor(executor, common):
    with executor:
        return executor.run("reconstruct", common)


def _failure_message(text, partitions, logs, done):
    """Name the failed contig from its lane's message (or, if the lane died, its first unfinished contig)."""
    match = re.search(re.escape(_FAILURE) + r" (\S+?):", text)
    failed = match.group(1) if match else None
    died = re.match(r"chromosome worker (\d+) (?:exited unexpectedly|closed its pipe)", text)
    if failed not in logs and died is not None and int(died.group(1)) < len(partitions):
        failed = next((contig for contig in partitions[int(died.group(1))] if not done(contig)), None)
    unfinished = [contig for partition in partitions for contig in partition if contig != failed and not done(contig)]
    head = (f"reconstruction failed in contig {failed} (log {logs[failed]})" if failed in logs
            else "a contig lane failed")
    return (f"{head}; the other lanes were stopped and these contigs resume from their own records on rerun: "
            f"{unfinished or 'none'}")


class _Stopped(SystemExit):
    """SIGTERM: another contig failed, the run was stopped, or the main process exited."""


_STOPPING = False


def _stop(signum, frame):
    # Raise once: a later SIGTERM (a signal to the whole job, then the parent watch's) must not cut the
    # unwinding's cleanup short. Only a raise that Python dropped instead of unwinding is re-armed.
    global _STOPPING
    if not _STOPPING:
        _STOPPING = True
        raise _Stopped(f"stopped by signal {signum}")


_PARENT_EXITED = threading.Event()
_SEND_STOP = threading.Event()
_RESEND_SECONDS = 0.1
_parent_watch = None


def _rearm_dropped_stop(previous_hook, unraisable):
    """``sys.unraisablehook`` of a lane: re-arm a stop that Python dropped.

    The SIGTERM handler runs at the main thread's next bytecode, which during Numba compilation can be a ctypes
    callback from LLVM; Python reports an exception raised there and drops it, so the contig would run on. Only
    such a provably lost stop is re-armed; _send_stops then sends SIGTERM again.
    """
    global _STOPPING
    try:
        previous_hook(unraisable)
    finally:
        if isinstance(unraisable.exc_value, _Stopped):
            _SEND_STOP.set()
            _STOPPING = False


def _watch_parent(parent):
    """Stop this lane once the main process has exited (for any reason, including SIGKILL)."""
    parent.join()
    _PARENT_EXITED.set()
    _SEND_STOP.set()


def _send_stops(main_thread):
    """Once a stop is needed, SIGTERM the main thread until one is unwinding."""
    _SEND_STOP.wait()
    while True:
        _SEND_STOP.clear()
        if not _STOPPING:
            signal.pthread_kill(main_thread, signal.SIGTERM)
        _SEND_STOP.wait(_RESEND_SECONDS)


def _lane_contig(contig, state, phase, common, budget):
    """ChromosomeExecutor handler: one contig's work under this lane's lease."""
    global _parent_watch
    from ..store.store import CheckpointStore
    parent = multiprocessing.parent_process()
    if parent is not None and (_PARENT_EXITED.is_set() or not parent.is_alive()):
        print(f"[{contig}] not started because the main process exited", flush=True)
        raise _Stopped("the main process exited")
    threads = budget.threads()
    started = time.monotonic()
    if not faulthandler.is_enabled():
        faulthandler.enable()
    if _parent_watch is None and parent is not None:
        sys.unraisablehook = functools.partial(_rearm_dropped_stop, sys.unraisablehook)
        threading.Thread(target=_send_stops, args=(threading.get_ident(),), name="lane-stop-send",
                         daemon=True).start()
        _parent_watch = threading.Thread(target=_watch_parent, args=(parent,), name="lane-parent-watch",
                                         daemon=True)
        _parent_watch.start()
    previous = signal.signal(signal.SIGTERM, _stop)
    log = common["logs"][contig]
    store = CheckpointStore(common["root"], nthreads=threads)
    try:
        with run_log(log), worker_stages() as stages:
            print(f"[{contig}] lane pid {os.getpid()}: {threads}-core lease (phases may grow it)", flush=True)
            try:
                summary = common["work"](store, contig, common["context"], budget)
            except BaseException as error:
                traceback.print_exc(file=sys.stdout)
                if isinstance(error, _Stopped):
                    reason = ("because the main process exited" if _PARENT_EXITED.is_set()
                              else "by SIGTERM (another contig failed or the run was stopped)")
                    print(f"[{contig}] stopped {reason}; its records are kept for resume", flush=True)
                    raise
                raise RuntimeError(f"{_FAILURE} {contig}: {type(error).__name__}: {error} (log {log})") from error
            wall = time.monotonic() - started
            print(f"[{contig}] done in {wall:.1f} s", flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous)
    return state, dict(summary=summary, threads=threads, wall_seconds=wall, log=log, pid=os.getpid(),
                       stages=stages)
