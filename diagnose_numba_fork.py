#!/usr/bin/env python
"""
Diagnose why numba prange runs single-threaded inside the block-hap forkserver
workers (so the dynamic-thread ramp has no physical effect on the tail).

Runs the SAME CPU-bound parallel kernel in (a) the main process and (b) a
forkserver pool worker configured like _init_block_worker, and reports for each:
  - the numba threading layer actually in use,
  - wall-time at 1 thread vs all-cores -- the SPEEDUP is the test: ~Nx means
    numba really threads; ~1x means it is silently single-threaded,
  - how many distinct numba threads actually executed the loop (when the numba
    build exposes get_thread_id).

Run on a compute node in bio-env, TWICE:
    python diagnose_numba_fork.py
    NUMBA_THREADING_LAYER=workqueue python diagnose_numba_fork.py

Reading it:
  * main MULTI but worker SINGLE  -> fork issue.  If the layer is 'tbb', it is
    not fork-safe; if 'workqueue' makes the worker MULTI, that env var is the fix.
  * both SINGLE                   -> a config problem (NUMBA_NUM_THREADS / no
    usable threading layer), independent of fork.
  * both MULTI                    -> the plain setup threads fine, so something
    the pipeline does before the pool starts is poisoning it (see notes below).
"""
import os
import time
import multiprocessing as mp

TOTAL = len(os.sched_getaffinity(0))            # cores actually granted to this job
os.environ.setdefault("NUMBA_NUM_THREADS", str(TOTAL))

import numpy as np
import numba
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=False)
def _busy(out, reps):
    """Embarrassingly parallel and CPU-bound (no memory bottleneck), so a
    multi-threaded run should scale close to linearly."""
    for i in prange(out.shape[0]):
        acc = 0.0
        x = i * 1e-6
        for r in range(reps):
            acc += np.sin(x + r) * np.cos(x - r)
        out[i] = acc


def _count_distinct_threads():
    """How many numba threads actually executed a prange loop (bonus signal;
    returns -1 if this numba build doesn't expose get_thread_id)."""
    try:
        gid = numba.get_thread_id
    except AttributeError:
        return -1

    @njit(parallel=True, cache=False)
    def _who(tag):
        for i in prange(tag.shape[0]):
            tag[i] = gid()

    tag = np.zeros(200000, dtype=np.int64)
    numba.set_num_threads(TOTAL)
    _who(tag)
    return int(np.unique(tag).size)


def _probe(label):
    N, REPS = 4_000_000, 20
    out = np.empty(N)
    numba.set_num_threads(1)
    _busy(out, 1)                                # warm up: compile + launch pool
    numba.set_num_threads(1)
    t = time.perf_counter(); _busy(out, REPS); t1 = time.perf_counter() - t
    numba.set_num_threads(TOTAL)
    t = time.perf_counter(); _busy(out, REPS); tn = time.perf_counter() - t
    try:
        layer = numba.threading_layer()
    except Exception as e:
        layer = f"<unavailable: {e}>"
    distinct = _count_distinct_threads()
    speedup = t1 / tn if tn > 0 else 0.0
    print(f"[{label}] pid={os.getpid()}  layer={layer}  "
          f"cap(NUMBA_NUM_THREADS)={numba.config.NUMBA_NUM_THREADS}  "
          f"get_num_threads={numba.get_num_threads()}")
    print(f"[{label}] 1-thread={t1:6.3f}s   {TOTAL}-thread={tn:6.3f}s   "
          f"speedup={speedup:5.1f}x   distinct_threads_used={distinct}")
    print(f"[{label}] => numba runs {'MULTI-threaded' if speedup > 1.8 else 'SINGLE-threaded'} here")


def _worker(_):
    # mirror _init_block_worker's numba setup
    numba.config.NUMBA_NUM_THREADS = TOTAL
    numba.set_num_threads(1)
    _probe("forkserver-worker")
    return True


if __name__ == "__main__":
    print(f"numba {numba.__version__} | cores(sched_getaffinity)={TOTAL} | "
          f"NUMBA_THREADING_LAYER={os.environ.get('NUMBA_THREADING_LAYER', '<default>')}")
    print("-" * 78)
    _probe("main-process")
    print("-" * 78)
    ctx = mp.get_context("forkserver")
    with ctx.Pool(processes=2) as pool:
        list(pool.map(_worker, range(2)))