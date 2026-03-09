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

Uses setdefault so that advanced users can override via environment
variables if they know what they are doing.

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
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

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