"""
Thread configuration — import before numpy or numba.

Ensures all internal threading libraries (BLAS, OpenMP, MKL) default
to single-threaded mode. This prevents worker processes spawned by
multiprocess.Pool from each creating their own thread pools, which
causes massive oversubscription on multi-core machines.

All BLAS/OpenMP parallelism should be controlled at the process level via Pool.

Numba's thread pool is initialised at the machine's full core count,
but each worker process should use numba_thread_scope() or
numba.set_num_threads() to restrict how many threads it actually uses.
This allows adaptive parallelism: L1 workers use 1 Numba thread each,
while L3 workers use many.

Threading layer selection (automatic):
  1. TBB  — idle threads sleep (zero CPU). Best for dynamic reallocation.
  2. OMP  — idle threads sleep with OMP_WAIT_POLICY=PASSIVE. Good fallback.
  3. workqueue — busy-wait spinloop. Always available but wastes CPU on
     idle threads. Acceptable when thread pools are small.

To force a specific layer, set NUMBA_THREADING_LAYER_OVERRIDE in the
environment before importing this module.

Forkserver preloading: configures multiprocessing.set_forkserver_preload()
with heavy modules (numpy, numba, scipy, project modules) so the
forkserver process imports them once at startup. All workers forked from
it inherit the imports via COW, avoiding ~2 GB of per-worker import
overhead. This must be called before the first Pool creation in the
session. Subsequent imports of thread_config are no-ops.
"""
import os
from contextlib import contextmanager

# =========================================================================
# BLAS / OpenMP / MKL — always single-threaded
# =========================================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# =========================================================================
# Numba — initialise pool at full machine size so set_num_threads can
# scale up to it later. Default to 1 if os.cpu_count() fails.
# =========================================================================
os.environ.setdefault("NUMBA_NUM_THREADS", str(os.cpu_count() or 1))

# =========================================================================
# Threading layer — pick the best available backend.
#
# TBB and OMP put idle threads to sleep. workqueue uses busy-wait
# spinning, which wastes CPU when thread pools are oversized (e.g.
# 112 workers each with a 112-thread pool for dynamic reallocation).
#
# Always auto-selects unless NUMBA_THREADING_LAYER_OVERRIDE is set.
# (We override NUMBA_THREADING_LAYER unconditionally because stale
# values from previous runs or shell config can persist and prevent
# auto-selection.)
# =========================================================================
_user_override = os.environ.get("NUMBA_THREADING_LAYER_OVERRIDE")

if _user_override:
    os.environ["NUMBA_THREADING_LAYER"] = _user_override
else:
    _selected_layer = "workqueue"  # fallback — always available
    
    try:
        import tbb as _tbb  # noqa: F401
        _selected_layer = "tbb"
    except (ImportError, OSError):
        # TBB unavailable — use OMP with passive wait instead
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        _selected_layer = "omp"
    
    os.environ["NUMBA_THREADING_LAYER"] = _selected_layer

# =========================================================================
# Forkserver preloading
# =========================================================================
# Tell the forkserver to pre-import heavy modules when it starts.
# Workers forked from it inherit these imports via COW (~500 MB shared)
# instead of each importing them fresh (~2 GB per worker).
#
# set_forkserver_preload must be called before any Pool creation.
# It is safe to call multiple times — only the last call before the
# forkserver starts takes effect. After the forkserver starts, further
# calls are ignored.
try:
    import multiprocessing as _mp
    _mp.set_forkserver_preload([
        # Core scientific stack
        'numpy', 'numba', 'scipy', 'math', 'time', 'hdbscan',
        # Project modules used by workers
        'thread_config',
        'block_haplotypes', 'block_linking', 'hmm_matching',
        'beam_search_core', 'analysis_utils', 'chimera_resolution',
    ])
except (AttributeError, RuntimeError):
    # set_forkserver_preload not available (old Python) or forkserver
    # already started — safe to ignore
    pass


@contextmanager
def numba_thread_scope(n_threads):
    """
    Context manager to temporarily set Numba's active thread count.
    Restores the previous value on exit, even if an exception occurs.
    
    Usage:
        with numba_thread_scope(37):
            # prange loops use 37 threads here
        # restored to previous count here
    """
    import numba
    old = numba.get_num_threads()
    numba.set_num_threads(n_threads)
    try:
        yield
    finally:
        numba.set_num_threads(old)