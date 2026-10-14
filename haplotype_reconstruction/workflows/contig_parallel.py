"""Reconstruct independent contigs concurrently inside one allocation.

Contig coordinators are persistent ``core.chromosome_parallel`` workers:
non-daemonic forkserver processes sharing the run's core ceiling as leases.
Each coordinator calls the unchanged per-contig route,
``reconstruction.run_reconstruction`` for one contig with
``publish_completion=False``, and sets that contig's release, painting and
checkpoint core ceilings to its lease. Pools, Numba teams and compression teams
inside a contig stay within its lease, so concurrent contigs never exceed the
run's ceiling. Leases start in proportion to each coordinator's work. A
coordinator that has finished its contigs returns its lease; the others claim
it, in proportion to their remaining work, at their next contig or phase start
(block-feedback round, context refinement, assembly preprocess, hierarchy and
refinement level, painting), while their pool phases run (path-selection
rounds, context refinements, hierarchy and component-refinement levels: the
coordinator raises the pool's shared thread total every second, and workers
take the threads at their next dynamic-thread boundary; the process count
stays) and at every sweep allocation of the founder refinement, never inside a
running kernel.

A contig's computation does not read other contigs' work in progress, and core
ceilings are execution choices outside every scientific identity. Each
coordinator re-binds the painting stage identity computed with its own ceiling,
so a ceiling-dependent identity would fail at once rather than resume or mix
outputs. The parent validates the source and binds that identity before any
coordinator starts; it alone records stage completion. The first failing
contig stops the run, as in the sequential route: the other coordinators are
stopped and keep their contigs' records for resume. A coordinator
also stops in the same way as soon as the workflow's main process exits for any
reason (for example a signal sent to that process alone), so no coordinator
keeps computing or writing checkpoints without an owner, and it starts no
further contig. A stop that Python drops inside a callback (as during Numba
compilation) is re-armed and sent again.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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

import haplotype_reconstruction.core.chromosome_parallel as core_chromosome_parallel
import haplotype_reconstruction.core.environment as core_environment
import haplotype_reconstruction.core.run_record as core_run_record
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.discovery.checkpoints as discovery_checkpoints
import haplotype_reconstruction.painting.components as painting_components
from haplotype_reconstruction.workflows import reconstruction

# Operational scheduling rule, not a scientific setting; core/config.py and
# core/chromosome_parallel.py are hashed into checkpoint identities. Narrow
# contig lanes need fewer core-seconds: the same contig at 16 threads took 7%
# fewer than at 32-37 (N=320 chr19 and chr8 shards; path selection scales
# fully, hierarchy, refinement tails and serial checkpoint phases do not). So
# the automatic rule gives each concurrent contig about sixteen cores, and
# leases start in proportion to each coordinator's work.
CONTIG_WORKER_CORES = 16

# Peak memory of one contig lane (coordinator, its pools and shared arrays),
# for limiting concurrency only: SOURCE_MEMORY_FACTOR x the decoded discovery
# product (discovery.checkpoints.decoded_bytes) + CONTIG_MEMORY_BYTES +
# CORE_MEMORY_BYTES per lease core. Fitted to
# node memory sampled every 10 s in the N=320 host run (3 lanes of 37 threads,
# 22 contigs, 39 GiB outside the lanes): the sum over running lanes bounds all
# 2,015 samples, including contig transitions, the node peak (236 GiB with
# chr3, chr4 and chr11), a chr3 lane implied at ~100 GiB (estimate 112 GiB)
# and a lane of chr19 (4.1 GiB decoded) alone at ~50 GiB (estimate 56 GiB).
# Resumed contigs only validate and rewrite reports.
SOURCE_MEMORY_FACTOR = 3
CONTIG_MEMORY_BYTES = 16 << 30
CORE_MEMORY_BYTES = 3 << 28

# Fraction of a contig's reconstruction done when each phase that re-reads its
# lane's lease starts (core/chromosome_parallel.lease_threads; phase names as
# passed by block_feedback, assembly.pipeline and reconstruction): lane-minutes
# summed over the 22 contigs of the N=320 live-head host run (285283b, 18-21
# threads per lane). Operational only: leases follow each lane's remaining work.
RECONSTRUCTION_PHASE_PROGRESS = {
    "feedback_round0": 0.0,
    "feedback_l1_preprocess": 0.258, "feedback_l1_hierarchy_l1": 0.269,
    "feedback_l1_refine": 0.350, "feedback_round1": 0.363,
    "feedback_l2_preprocess": 0.407, "feedback_l2_hierarchy_l1": 0.418,
    "feedback_l2_hierarchy_l2": 0.454, "feedback_l2_refine": 0.494, "feedback_round2": 0.513,
    "final_preprocess": 0.532, "final_hierarchy_l1": 0.543, "final_refinement_l1": 0.555,
    "final_hierarchy_l2": 0.590, "final_refinement_l2": 0.614,
    "final_hierarchy_l3": 0.736, "final_refinement_l3": 0.745,
    "final_hierarchy_l4": 0.892, "final_refinement_l4": 0.896,
    "painting": 0.989,
}

# Decoded sizes come from record indexes (header/trailer reads) that wait on
# shared-filesystem latency, so the contigs' files are indexed concurrently.
SIZE_SCAN_THREADS = 16

_FAILURE = "contig coordinator failed on"


def configured_contig_workers(value=None):
    """Return an explicit concurrent-contig count, or None for the automatic rule.

    ``value`` None reads HAPLOTYPES_CONTIG_WORKERS (``auto`` by default).
    """
    if value is None:
        value = os.environ.get("HAPLOTYPES_CONTIG_WORKERS", "auto")
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return None
        if not text.isdigit():
            raise ValueError("contig_workers must be 'auto' or a positive integer")
        value = int(text)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("contig_workers must be 'auto' or a positive integer")
    return value


def configured_serial_contigs():
    """Return (serial, max_contigs) from HAPLOTYPES_SERIAL_CONTIGS and HAPLOTYPES_MAX_CONTIGS.

    The CLI resolves both (CLI > TOML > environment); a contig cap implies serial contigs.
    """
    limit = os.environ.get("HAPLOTYPES_MAX_CONTIGS")
    if limit is not None:
        if not limit.strip().isdigit() or int(limit) < 1:
            raise ValueError("HAPLOTYPES_MAX_CONTIGS must be a positive integer")
        limit = int(limit)
    serial = core_environment.boolean_setting(
        os.environ.get("HAPLOTYPES_SERIAL_CONTIGS", "off"), "serial_contigs")
    return serial or limit is not None, limit


def run_serially(checkpoint_store, contigs, sample_ids, *, discover, discovery_identity, config,
                 output_dir, max_contigs=None, all_contigs=None, genetic_maps=None):
    """Discover, assemble and paint one contig at a time, each recorded before the next.

    For each of ``contigs`` (run order) still lacking its painting:
    ``discover([contig])`` when its block discovery is incomplete (the
    workflow's block discovery, whose pool is closed before reconstruction),
    then ``reconstruction.run_reconstruction`` for that contig alone: the
    sequential route with the whole core ceiling. A contig thus runs the code of
    a normal run and resumes from its own records. No stage completion is
    recorded here: once every contig is painted, the caller continues as a
    normal run, which validates the painted contigs, records stage completion
    and runs the genome-wide stages.

    ``max_contigs`` caps the contigs finished in this call, including a resumed
    partial one. Returns True when every contig is painted, or False after a
    clean stop that leaves contigs for a later invocation.
    """
    contigs = tuple(contigs)
    pending = [contig for contig in contigs
               if not checkpoint_store.contig(contig).has(reconstruction.PAINTING_STAGE)]
    selected = pending if max_contigs is None else pending[:max_contigs]
    cores = max(config.release_config.num_processes, config.paint_cores)
    print(f"\nSERIAL CONTIGS: {len(contigs) - len(pending)} of {len(contigs)} contigs painted"
          + (f"; now {', '.join(selected)}: discovery, assembly and painting, one contig at a "
             f"time with a {cores}-core ceiling" if selected else ""), flush=True)
    for done, contig in enumerate(selected, start=len(contigs) - len(pending) + 1):
        started = time.monotonic()
        if not discovery_checkpoints.complete(checkpoint_store.contig(contig)):
            discover([contig])
        reconstruction.run_reconstruction(
            checkpoint_store, (contig,), sample_ids, discovery_identity=discovery_identity,
            config=config, output_dir=output_dir,
            all_contigs=contigs if all_contigs is None else all_contigs,
            publish_completion=False, genetic_maps=genetic_maps)
        print(f"  [SERIAL] {contig} painted in {time.monotonic() - started:.1f} s; "
              f"{done} of {len(contigs)} contigs painted", flush=True)
    remaining = pending[len(selected):]
    if remaining:
        print(f"[STOP] max_contigs = {max_contigs} reached: "
              f"{len(contigs) - len(remaining)} of {len(contigs)} contigs painted; "
              f"remaining: {', '.join(remaining)}. Completed records are kept; "
              "rerun the same command to continue.", flush=True)
        return False
    return True


def plan_contig_workers(checkpoint_store, contigs, *, cores, requested):
    """Return concurrent coordinators and longest-first work weights per contig.

    Automatic: one coordinator per CONTIG_WORKER_CORES cores, at most one per
    contig still lacking its painting. A decoded-source memory estimate then
    limits the concurrent group; it never reduces the shared core ceiling.
    """
    files = [checkpoint_store.contig(contig) for contig in contigs]
    pending = [file.name for file in files if not file.has(reconstruction.PAINTING_STAGE)]
    limit = max(1, cores // CONTIG_WORKER_CORES) if requested is None else requested
    workers = min(len(pending), limit, cores)
    if workers < 2:
        return 1, None
    lease = cores // workers
    with ThreadPoolExecutor(min(SIZE_SCAN_THREADS, len(files))) as scanner:
        decoded = list(scanner.map(discovery_checkpoints.decoded_bytes, files))
    weights, estimates = [], []
    for contig, source in zip(contigs, decoded):
        if contig in pending:
            weights.append(float(source))
            estimates.append(SOURCE_MEMORY_FACTOR * source + CONTIG_MEMORY_BYTES
                             + CORE_MEMORY_BYTES * lease)
        else:
            weights.append(0.)
            estimates.append(2 * source + CONTIG_MEMORY_BYTES)
    workers = core_chromosome_parallel.memory_worker_limit(
        estimates, workers, core_runtime.available_memory_bytes(),
        label="RECONSTRUCTION")
    return workers, weights


def _validated_contigs(contigs, all_contigs, publish_completion):
    """The sequential route's contig-order checks, before choosing a width."""
    contigs = tuple(str(value) for value in contigs)
    if not contigs or len(contigs) != len(set(contigs)):
        raise ValueError("contigs must be a nonempty unique ordered collection")
    all_contigs = contigs if all_contigs is None else tuple(str(value) for value in all_contigs)
    if not all_contigs or len(all_contigs) != len(set(all_contigs)):
        raise ValueError("all_contigs must be a nonempty unique ordered collection")
    requested = set(contigs)
    if not requested.issubset(all_contigs):
        raise ValueError("contigs must be a subset of all_contigs")
    if tuple(value for value in all_contigs if value in requested) != contigs:
        raise ValueError("contigs must preserve all_contigs order")
    if not isinstance(publish_completion, bool):
        raise TypeError("publish_completion must be boolean")
    return contigs, all_contigs


def run_reconstruction(checkpoint_store, contigs, sample_ids, *, discovery_identity, config,
                       output_dir, all_contigs=None, publish_completion=True,
                       painter_factory=painting_components.ComponentPainter,
                       genetic_maps=None, contig_workers=None):
    """``reconstruction.run_reconstruction`` with independent contigs run concurrently.

    The run's core ceiling C is max(release num_processes, paint_cores); each
    concurrent contig uses its lease for both and for checkpoint compression.
    ``contig_workers`` None reads HAPLOTYPES_CONTIG_WORKERS (default auto:
    min(pending contigs, C // CONTIG_WORKER_CORES), memory-limited); 1, a single
    pending contig or a one-core ceiling run the sequential route unchanged.
    Returns the same summaries in the same contig order.
    """
    arguments = dict(
        discovery_identity=discovery_identity, config=config, output_dir=output_dir,
        all_contigs=all_contigs, publish_completion=publish_completion,
        painter_factory=painter_factory, genetic_maps=genetic_maps)
    requested = configured_contig_workers(contig_workers)
    if not isinstance(config, reconstruction.ReconstructionConfig):
        raise TypeError("config must be a ReconstructionConfig")
    contigs, all_contigs = _validated_contigs(contigs, all_contigs, publish_completion)
    cores = max(config.release_config.num_processes, config.paint_cores)
    workers, weights = 1, None
    if requested != 1 and len(contigs) > 1 and cores > 1:
        workers, weights = plan_contig_workers(
            checkpoint_store, contigs, cores=cores, requested=requested)
    if workers > 1:
        executor = core_chromosome_parallel.ChromosomeExecutor(
            contigs, _reconstruct_contig, n_workers=cores, weights=weights,
            max_workers=workers, label="RECONSTRUCTION",
            phase_progress=RECONSTRUCTION_PHASE_PROGRESS)
        if executor.worker_count > 1:
            return _run_concurrently(
                checkpoint_store, contigs, sample_ids, executor, all_contigs=all_contigs,
                arguments=arguments)
    print(f"  Contigs run sequentially with a {cores}-core ceiling", flush=True)
    return reconstruction.run_reconstruction(
        checkpoint_store, contigs, sample_ids, **arguments)


def _run_concurrently(checkpoint_store, contigs, sample_ids, executor, *, all_contigs,
                      arguments):
    """Validate and bind as the sequential route does, run coordinators, record."""
    target_stage = reconstruction.PAINTING_STAGE
    config = arguments["config"]
    ordered_ids = reconstruction._canonical_sample_ids(sample_ids)
    available = core_runtime.available_cpu_count()
    requested = max(config.release_config.num_processes, config.paint_cores)
    if requested > available:
        raise ValueError(
            f"requested core ceiling ({requested}) exceeds process affinity "
            f"({available})"
        )
    reconstruction._validate_discovery_global_source(
        checkpoint_store, all_contigs, ordered_ids, arguments["discovery_identity"],
        require_complete=arguments["publish_completion"])
    checkpoint_store.bind_identity(
        checkpoint_store.run, target_stage, reconstruction.assembly_production_stage_identity(
            config, discovery_identity=arguments["discovery_identity"],
            sample_ids=ordered_ids, contigs=all_contigs,
            genetic_maps=arguments["genetic_maps"]))
    missing = [contig for contig in contigs
               if not checkpoint_store.contig(contig).has(target_stage)]
    if checkpoint_store.stage_complete(target_stage) and missing:
        raise RuntimeError(f"{target_stage} is marked complete but lacks: {missing}")

    directory = os.path.abspath(os.environ.get("HAPLOTYPES_LOG_DIR", "work/logs"))
    os.makedirs(directory, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logs = {contig: os.path.join(directory, f"reconstruction_{contig}_{stamp}_{os.getpid()}.log")
            for contig in contigs}
    partitions = [[contig for _, contig in partition] for partition in executor.partitions]
    print(f"  Contigs: {executor.worker_count} concurrent coordinators share "
          f"{executor.cpu_budget} cores (leases in proportion to work, growing at "
          "contig and phase starts and inside pool phases); "
          "longest first per coordinator: "
          + "; ".join(", ".join(partition) for partition in partitions), flush=True)
    for contig in contigs:
        print(f"    {contig} log: {logs[contig]}", flush=True)
    common = dict(
        root=checkpoint_store.root, sample_ids=ordered_ids, logs=logs,
        arguments=dict(arguments, all_contigs=all_contigs, publish_completion=False))
    try:
        results = _run_coordinators(executor, common)
    except RuntimeError as error:
        message = _failure_message(str(error), partitions, checkpoint_store, target_stage, logs)
        print(f"  [FAILED] {message}", flush=True)
        raise RuntimeError(message) from error

    summaries = []
    for contig, result in zip(contigs, results):
        summary = result["summary"]
        summaries.append(summary)
        status = "resumed" if summary.resumed else "completed"
        print(f"  [{status}] {contig}: {result['threads']}-core ceiling, "
              f"{result['wall_seconds']:.1f} s; log {result['log']}", flush=True)
        core_run_record.record_product("reconstruction", contig, dict(
            resumed=bool(summary.resumed), threads=int(result["threads"]),
            wall_seconds=float(result["wall_seconds"]), log=result["log"],
            pid=int(result["pid"]), stages=result["stages"]))
    core_runtime.require_records(checkpoint_store, target_stage, contigs)
    if arguments["publish_completion"]:
        core_runtime.require_records(checkpoint_store, target_stage, all_contigs)
        checkpoint_store.mark_stage_complete(target_stage)
    return tuple(summaries)


@core_run_record.timed_stage("reconstruction")
def _run_coordinators(executor, common):
    with executor:
        return executor.run("reconstruct", common)


def _failure_message(text, partitions, checkpoint_store, target_stage, logs):
    """Name the failed contig from its coordinator's message, or, if the
    coordinator process died, the first unfinished contig of its partition."""
    match = re.search(re.escape(_FAILURE) + r" (\S+?):", text)
    failed = match.group(1) if match else None
    died = re.match(r"chromosome worker (\d+) (?:exited unexpectedly|closed its pipe)", text)
    if failed not in logs and died is not None and int(died.group(1)) < len(partitions):
        failed = next((contig for contig in partitions[int(died.group(1))]
                       if not checkpoint_store.contig(contig).has(target_stage)), None)
    unfinished = [contig for partition in partitions for contig in partition
                  if contig != failed
                  and not checkpoint_store.contig(contig).has(target_stage)]
    head = (f"reconstruction failed in contig {failed} (log {logs[failed]})"
            if failed in logs else "a reconstruction contig coordinator failed")
    return (f"{head}; the other coordinators were stopped and these contigs resume from "
            f"their own records on rerun: {unfinished or 'none'}")


class _Stopped(SystemExit):
    """SIGTERM: another contig failed, the run was stopped, or the main process exited."""


_STOPPING = False


def _stop(signum, frame):
    # Raise once: a later SIGTERM (a signal to the whole job, then the parent
    # watch's) must not cut the unwinding's cleanup short. Only a raise that
    # Python dropped instead of unwinding is re-armed (_rearm_dropped_stop).
    global _STOPPING
    if not _STOPPING:
        _STOPPING = True
        raise _Stopped(f"stopped by signal {signum}")


_PARENT_EXITED = threading.Event()
# Set by the parent watch and by a dropped stop: _send_stops sends SIGTERM.
_SEND_STOP = threading.Event()
# _send_stops also rechecks this often whether a stop is still needed.
_RESEND_SECONDS = 0.1
_parent_watch = None


def _rearm_dropped_stop(previous_hook, unraisable):
    """``sys.unraisablehook`` of a coordinator: re-arm a stop that was dropped.

    The SIGTERM handler runs at the main thread's next bytecode, which during
    Numba compilation can be a ctypes callback from LLVM
    (``ExecutionEngine._raw_object_cache_notify``). Python reports an exception
    raised there ("Exception ignored on calling ctypes callback function") and
    drops it, so the contig would run on. Only such a provably lost stop is
    re-armed; _send_stops then sends SIGTERM again.
    """
    global _STOPPING
    try:
        previous_hook(unraisable)
    finally:
        if isinstance(unraisable.exc_value, _Stopped):
            _SEND_STOP.set()
            # Last: no later bytecode of this hook, where a raise would be
            # dropped again, can run the handler once the stop is re-armed.
            _STOPPING = False


def _watch_parent(parent):
    """Stop this coordinator once the workflow's main process has exited.

    The main process holds the only write end of the pipe behind the forkserver
    parent sentinel, so ``join`` returns when it exits for any reason, including
    SIGKILL. The SIGTERM that _send_stops then sends unwinds a running contig as
    after a peer's failure; between contigs its default action ends this
    coordinator.
    """
    parent.join()
    _PARENT_EXITED.set()
    _SEND_STOP.set()


def _send_stops(main_thread):
    """Once a stop is needed, SIGTERM the main thread until one is unwinding.

    Each _SEND_STOP sends at once; after a dropped stop that is as soon as the
    callback returns and releases the GIL, normally before the C code that
    called it can enter another one (a fixed resend period can keep landing in
    the next callback). _STOPPING stays set from the raise that unwinds the
    contig, so no SIGTERM is sent into its cleanup; only a dropped raise clears
    it.
    """
    _SEND_STOP.wait()
    while True:
        _SEND_STOP.clear()
        if not _STOPPING:
            signal.pthread_kill(main_thread, signal.SIGTERM)
        _SEND_STOP.wait(_RESEND_SECONDS)


def _reconstruct_contig(contig, state, phase, common):
    """ChromosomeExecutor handler: one contig's unchanged route under this lease."""
    global _parent_watch
    parent = multiprocessing.parent_process()
    if parent is not None and (_PARENT_EXITED.is_set() or not parent.is_alive()):
        # The main process exited during an earlier contig of this partition:
        # stop as its parent watch would, before this contig starts.
        print(f"[{contig}] not started because the workflow's main process exited",
              flush=True)
        raise _Stopped("the workflow's main process exited")
    threads = core_chromosome_parallel.current_threads()
    started = time.monotonic()
    if not faulthandler.is_enabled():
        faulthandler.enable()
    if _parent_watch is None and parent is not None:
        sys.unraisablehook = functools.partial(_rearm_dropped_stop, sys.unraisablehook)
        threading.Thread(target=_send_stops, args=(threading.get_ident(),),
                         name="contig-stop-send", daemon=True).start()
        _parent_watch = threading.Thread(
            target=_watch_parent, args=(parent,),
            name="contig-parent-watch", daemon=True)
        _parent_watch.start()
    # Stopping after a peer's failure or the main process's exit unwinds the
    # contig's pools and shared memory.
    previous = signal.signal(signal.SIGTERM, _stop)
    log = common["logs"][contig]
    arguments = common["arguments"]
    config = arguments["config"]
    arguments = dict(arguments, config=replace(
        config, paint_cores=threads,
        release_config=replace(config.release_config, num_processes=threads)))
    store = core_runtime.CheckpointStore(common["root"], nthreads=threads)
    try:
        with core_runtime.run_log(log), core_run_record.worker_stages() as stages:
            print(f"[{contig}] contig coordinator pid {os.getpid()}: {threads}-core "
                  "lease for release, painting and checkpoint I/O (phases may grow it)",
                  flush=True)
            try:
                summary, = reconstruction.run_reconstruction(
                    store, (contig,), common["sample_ids"], **arguments)
            except BaseException as error:
                traceback.print_exc(file=sys.stdout)
                if isinstance(error, _Stopped):
                    # A signal to the whole job (Slurm cancel or time limit)
                    # usually arrives before the main process's exit is seen.
                    reason = ("because the workflow's main process exited"
                              if _PARENT_EXITED.is_set()
                              else "by SIGTERM (another contig failed or the run was stopped)")
                    print(f"[{contig}] stopped {reason}; "
                          "its records are kept for resume", flush=True)
                    raise
                raise RuntimeError(f"{_FAILURE} {contig}: {type(error).__name__}: "
                                   f"{error} (log {log})") from error
            wall = time.monotonic() - started
            print(f"[{contig}] {'resumed' if summary.resumed else 'completed'} "
                  f"in {wall:.1f} s", flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous)
    return state, dict(summary=summary, threads=threads, wall_seconds=wall, log=log,
                       pid=os.getpid(), stages=stages)
