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
        # paint_samples + pedigree_inference: the pedigree-inference stage now
        # runs its Phase 1/2/3 and consistency-cutoff pools on forkserver too,
        # so preload these once in the server (workers inherit via COW) instead
        # of each worker re-importing them on spawn.
        'paint_samples', 'pedigree_inference',
    ])
except (AttributeError, RuntimeError):
    # set_forkserver_preload not available (old Python) or forkserver
    # already started — safe to ignore
    pass

# =========================================================================
# Numba caching — inject cache=True into all @njit decorators
# =========================================================================
# Workers recycled via maxtasksperchild lose compiled numba code.
# cache=True persists compiled functions to disk (__pycache__), so
# respawned workers load in ~0.1s instead of recompiling in 5-15s.
# setdefault respects explicit cache=False if ever needed.
try:
    import numba as _numba
    _original_njit = _numba.njit

    def _caching_njit(*args, **kwargs):
        kwargs.setdefault('cache', True)
        return _original_njit(*args, **kwargs)

    _numba.njit = _caching_njit
except ImportError:
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

# =========================================================================
# Dynamic Numba thread reallocation (shared, pool-wide)
# =========================================================================
# Mechanism factored out of hierarchical_assembly's private implementation so
# that ANY forkserver pool whose per-task work spans multiple modules can
# re-check the live active-worker count and re-apply its numba thread
# allocation at every major / intermediate phase boundary of a task.  This is
# what lets a long-running (straggler) task GROW into cores freed as its peers
# finish, instead of being pinned for its whole run to the thread count it
# happened to get when it started.
#
# In particular block_haplotypes_discrete -> bhd_recovery / bhd_trio (Stage-3
# block-haplotype discovery) spans three modules; each calls
# apply_dynamic_threads() at its phase boundaries, all reading this one shared
# state.  hierarchical_assembly keeps its own private copy (unchanged).
#
# State (set per worker by the pool initializer via set_dynamic_thread_state):
#   _DYN_ACTIVE_COUNTER : mp.Value('i') — # tasks currently active pool-wide.
#   _DYN_TOTAL_CORES    : int           — total cores available to the pool.
#   _DYN_EXTRA_COUNTER  : mp.Value('i') — # workers currently holding +1
#                                         (remainder distribution).  Optional:
#                                         if None, the helpers fall back to a
#                                         floor-only allocation (the remainder,
#                                         at most active-1 cores, is left idle).
#   _DYN_I_HAVE_EXTRA   : per-worker-process bool, True iff this worker holds +1.
# All None/False outside a properly-initialised pool worker, so the helpers
# no-op (return 1) on the sequential path — importing this module changes
# nothing until a pool wires the counters.

_DYN_ACTIVE_COUNTER = None
_DYN_TOTAL_CORES = None
_DYN_EXTRA_COUNTER = None
_DYN_I_HAVE_EXTRA = False


def set_dynamic_thread_state(total_cores, active_counter, extra_counter=None):
    """Wire this worker process to the pool-wide dynamic-thread counters.

    Call once from the pool initializer (initializer=...).  Resets the
    per-worker extra-claim flag, which is essential for correctness when a
    Pool recycles workers (maxtasksperchild): a respawned worker must not
    inherit a stale 'I hold an extra' belief.

    Args:
        total_cores: int — total cores available to the pool.
        active_counter: mp.Value('i', 0) shared across workers — the pool
            increments/decrements this around each task.
        extra_counter: mp.Value('i', 0) for remainder distribution, or None
            for floor-only allocation.
    """
    global _DYN_ACTIVE_COUNTER, _DYN_TOTAL_CORES, _DYN_EXTRA_COUNTER, _DYN_I_HAVE_EXTRA
    _DYN_ACTIVE_COUNTER = active_counter
    _DYN_TOTAL_CORES = total_cores
    _DYN_EXTRA_COUNTER = extra_counter
    # Defensive: ensure no stale claim from worker recycling.
    _DYN_I_HAVE_EXTRA = False


def release_dynamic_extra():
    """Release any remainder-thread (+1) claim this worker currently holds,
    WITHOUT tearing down the wiring.

    Call in a worker's per-task finally block (mirrors
    hierarchical_assembly's _try_release_extra at task end): the worker keeps
    its counter wiring across tasks (Pool worker reuse) but must not carry an
    extra-claim into its idle gap, or the remainder pool leaks.  No-op when
    this worker holds no extra or no extra_counter is set.
    """
    _try_release_dyn_extra()


def clear_dynamic_thread_state():
    """Tear down the wiring and release any extra-claim this worker holds.

    For full teardown (e.g. pool shutdown).  For the common per-task release
    that must preserve wiring across worker reuse, use release_dynamic_extra().
    """
    global _DYN_ACTIVE_COUNTER, _DYN_TOTAL_CORES, _DYN_EXTRA_COUNTER, _DYN_I_HAVE_EXTRA
    _try_release_dyn_extra()
    _DYN_ACTIVE_COUNTER = None
    _DYN_TOTAL_CORES = None
    _DYN_EXTRA_COUNTER = None
    _DYN_I_HAVE_EXTRA = False


def _try_claim_dyn_extra(remainder):
    """Atomically attempt to claim an extra thread from the remainder pool.

    Returns True if successfully claimed (and sets _DYN_I_HAVE_EXTRA),
    False if the pool is exhausted or the counter isn't set up.
    Idempotent: re-calling while already holding does not double-claim.

    Race analysis: the counter increment is guarded by its own lock.
    `current_extras < remainder` is evaluated INSIDE the lock so two
    workers can't both observe `current_extras = remainder - 1` and both
    push the counter to `remainder + 1`.  The local `_DYN_I_HAVE_EXTRA =
    True` happens-after the counter increment (same thread of execution).
    """
    global _DYN_I_HAVE_EXTRA
    if _DYN_I_HAVE_EXTRA:
        return True
    if _DYN_EXTRA_COUNTER is None:
        return False
    try:
        with _DYN_EXTRA_COUNTER.get_lock():
            if _DYN_EXTRA_COUNTER.value < remainder:
                _DYN_EXTRA_COUNTER.value += 1
                _DYN_I_HAVE_EXTRA = True
                return True
    except Exception:
        pass
    return False


def _try_release_dyn_extra():
    """Atomically release this worker's extra claim, if held.

    Defensive: clears the local flag even if the shared counter mutation
    fails.
    """
    global _DYN_I_HAVE_EXTRA
    if not _DYN_I_HAVE_EXTRA:
        return False
    if _DYN_EXTRA_COUNTER is None:
        _DYN_I_HAVE_EXTRA = False
        return False
    try:
        with _DYN_EXTRA_COUNTER.get_lock():
            _DYN_EXTRA_COUNTER.value -= 1
            _DYN_I_HAVE_EXTRA = False
            return True
    except Exception:
        _DYN_I_HAVE_EXTRA = False
        return False


def get_dynamic_threads():
    """Compute this worker's thread count from the live active-peer count.

    Returns floor(total_cores / active) + (1 if this worker holds an
    extra-claim else 0), clamped to >= 1.  floor+extra is always <=
    total_cores, so total threads in use across the pool == total_cores
    when an extra_counter is supplied (zero idle cores); with no
    extra_counter it is total_cores - remainder (floor-only).

    The read of active_counter.value is intentionally lock-free — a
    slightly stale count is fine since we recheck at every phase.  The
    extra-counter lock is acquired only briefly during a claim/release
    transition (not on every call once the claim has stabilised).

    Returns 1 on the sequential path (when state is unset).
    """
    if _DYN_ACTIVE_COUNTER is None or _DYN_TOTAL_CORES is None:
        return 1
    active = max(_DYN_ACTIVE_COUNTER.value, 1)
    floor = _DYN_TOTAL_CORES // active
    remainder = _DYN_TOTAL_CORES - floor * active

    # Adjust extra-claim based on the current remainder.
    if _DYN_EXTRA_COUNTER is not None:
        try:
            current_extras = _DYN_EXTRA_COUNTER.value
        except Exception:
            current_extras = 0
        if not _DYN_I_HAVE_EXTRA:
            if current_extras < remainder:
                _try_claim_dyn_extra(remainder)
        else:
            if current_extras > remainder:
                _try_release_dyn_extra()

    return max(1, floor + (1 if _DYN_I_HAVE_EXTRA else 0))


def apply_dynamic_threads():
    """Recompute and apply this worker's numba thread allocation.

    Call at every major / intermediate phase boundary of a long-running
    task (e.g. the top of a K-sweep iteration, a recovery round, an outer
    iteration).  No-op (returns 1) on the sequential path.  Cheap: a
    lock-free counter read plus numba.set_num_threads, which only sets the
    thread count for subsequently-entered parallel regions.

    Returns the thread count applied.
    """
    n = get_dynamic_threads()
    try:
        import numba
        numba.set_num_threads(n)
    except Exception:
        pass
    return n