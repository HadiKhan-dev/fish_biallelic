"""core / parallel for the canonical reconstruction pipeline."""
from __future__ import annotations


import os
import sys
import time
import numba
import ctypes
import multiprocessing as _mp
import multiprocessing.pool as _mp_pool
from contextlib import contextmanager
import math
import operator
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import numba as _numba


_ACTIVE_COUNTER = None


try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    _LIBC = None


try:
    forkserver_context = _mp.get_context("forkserver")
except (ValueError, AttributeError):
    forkserver_context = _mp.get_context("fork")


def _array_description(metadata):
    """Return validated ``(name, shape, dtype)`` from shared-array metadata."""
    shm_name = metadata.get("shm_name")
    name = metadata.get("name")
    if shm_name is not None and name is not None and shm_name != name:
        raise ValueError("shared-array metadata contains conflicting names")
    segment_name = shm_name if shm_name is not None else name
    if not isinstance(segment_name, str) or not segment_name:
        raise ValueError("shared-array metadata needs 'shm_name' or 'name'")

    try:
        shape = tuple(operator.index(size) for size in metadata["shape"])
    except (KeyError, TypeError) as exc:
        raise ValueError("shared-array metadata has an invalid shape") from exc
    if any(size < 0 for size in shape):
        raise ValueError("shared-array shape dimensions must be non-negative")

    try:
        dtype = np.dtype(metadata["dtype"])
    except (KeyError, TypeError) as exc:
        raise ValueError("shared-array metadata has an invalid dtype") from exc
    if dtype.hasobject:
        raise TypeError("object arrays cannot be transported by shared memory")
    return segment_name, shape, dtype


os.environ.setdefault("NUMBA_NUM_THREADS", str(os.cpu_count() or 1))


_TOTAL_CORES = None


def malloc_trim():
    """Ask glibc to return free heap pages to the operating system."""
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


class ForkserverPool(_mp_pool.Pool):
    """A standard process pool bound to the shared forkserver context."""

    def __init__(self, *args, **kwargs):
        kwargs["context"] = forkserver_context
        super().__init__(*args, **kwargs)


@numba.njit(cache=True, parallel=True)
def _copy_shared_bytes(source, destination):
    chunk = 1 << 20
    for part in numba.prange((source.size + chunk - 1) // chunk):
        start = part * chunk
        stop = min(source.size, start + chunk)
        destination[start:stop] = source[start:stop]


def create_shared_array(array, *, name_key="shm_name", dtype_as_string=True,
                        copy_threads=1):
    """Copy an array into a new segment and return ``(handle, metadata)``.

    Non-contiguous inputs are copied to C-contiguous storage.  POSIX shared
    memory does not allow zero-byte segments, so empty arrays receive a
    one-byte backing segment while retaining their original shape and dtype.
    If setup or copying fails after allocation, the new segment is immediately
    closed and unlinked. ``copy_threads`` is an upper bound; parallel first-touch
    copying is capped at 16 threads where measured memory bandwidth saturates.
    """
    if name_key not in ("shm_name", "name"):
        raise ValueError("name_key must be 'shm_name' or 'name'")
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.hasobject:
        raise TypeError("object arrays cannot be transported by shared memory")

    handle = SharedMemory(create=True, size=max(int(contiguous.nbytes), 1))
    try:
        shared_view = np.ndarray(
            contiguous.shape, dtype=contiguous.dtype, buffer=handle.buf
        )
        if copy_threads > 1 and contiguous.nbytes >= 4 << 20:
            # Explicit controller-side budget: never start a second pool
            # implicitly inside an already-parallel numerical worker.
            # 512 MiB first-touch test: 16 threads beat 76 (0.058s vs 0.087s).
            with numba_thread_scope(min(copy_threads, 16)):
                _copy_shared_bytes(
                    np.frombuffer(memoryview(contiguous), dtype=np.uint8),
                    np.frombuffer(handle.buf, dtype=np.uint8,
                                  count=contiguous.nbytes))
        else:
            np.copyto(shared_view, contiguous)
        metadata = {
            name_key: handle.name,
            "shape": tuple(contiguous.shape),
            "dtype": (
                str(contiguous.dtype) if dtype_as_string else contiguous.dtype
            ),
        }
        return handle, metadata
    except BaseException:
        close_shared_memory([handle], unlink=True)
        raise


_user_override = os.environ.get("NUMBA_THREADING_LAYER_OVERRIDE")


_EXTRA_COUNTER = None


class NonDaemonicProcess(forkserver_context.Process):
    """Forkserver process allowed to create child processes."""

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


def attach_shared_array(metadata):
    """Attach to a segment described by either supported metadata format."""
    segment_name, shape, dtype = _array_description(metadata)
    handle = SharedMemory(name=segment_name, create=False)
    try:
        required_bytes = math.prod(shape) * dtype.itemsize
        if required_bytes > len(handle.buf):
            raise ValueError(
                "shared-array metadata describes more bytes than the segment"
            )
        array = np.ndarray(shape, dtype=dtype, buffer=handle.buf)
        return handle, array
    except BaseException:
        handle.close()
        raise


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

# Numba may already have been imported (including above) before this module
# chooses its backend. Updating only os.environ then leaves config pointing
# at the inherited backend until a later config reload. Apply the existing
# selection before first pool initialization; never replace a running backend.
try:
    _numba.threading_layer()
except ValueError:
    _numba.config.THREADING_LAYER = os.environ["NUMBA_THREADING_LAYER"]


_STARTED_COUNTER = None


class NonDaemonicForkserverContext(type(forkserver_context)):
    Process = NonDaemonicProcess


def close_shared_memory(handles, *, unlink=False):
    """Close handles, optionally unlinking their segments, best-effort."""
    for handle in handles:
        try:
            handle.close()
        except Exception:
            pass
        if unlink:
            try:
                handle.unlink()
            except Exception:
                pass


try:
    import multiprocessing as _mp
    _mp.set_forkserver_preload([
        # Workers inherit imports through copy-on-write, before data loading.
        'numpy', 'numba', 'scipy', 'hdbscan',
        'haplotype_reconstruction.core.parallel',
        'haplotype_reconstruction.discovery.blocks',
        'haplotype_reconstruction.assembly.linking',
        'haplotype_reconstruction.assembly.paths',
        'haplotype_reconstruction.assembly.chimera_resolution',
        'haplotype_reconstruction.painting.components',
        'haplotype_reconstruction.pedigree.inference',
    ])
except (AttributeError, RuntimeError):
    # set_forkserver_preload not available (old Python) or forkserver
    # already started — safe to ignore
    pass


_PARTICIPANT_COUNTER = None


class NonDaemonicForkserverPool(_mp_pool.Pool):
    """Forkserver pool whose workers may create nested child pools."""

    def __init__(self, *args, **kwargs):
        kwargs["context"] = NonDaemonicForkserverContext()
        super().__init__(*args, **kwargs)


@contextmanager
def shared_memory_cleanup(handles):
    """Unlink parent-owned segments after the protected lifetime ends."""
    try:
        yield
    finally:
        close_shared_memory(handles, unlink=True)


_original_njit = _numba.njit


_BATCH_GENERATION = None


@contextmanager
def main_module_guard():
    """Prevent workers from re-executing the caller's entry script.

    The original __main__.__file__ and __main__.__spec__ values are restored
    on normal and exceptional exits. This intentionally preserves the existing
    convention that an absent or None __file__ remains absent after restoration.
    """

    import sys

    main_module = sys.modules.get("__main__")
    saved_main_file = getattr(main_module, "__file__", None)
    saved_main_spec = getattr(main_module, "__spec__", None)
    if main_module is not None:
        if hasattr(main_module, "__file__"):
            del main_module.__file__
        main_module.__spec__ = None
    try:
        yield
    finally:
        if main_module is not None:
            if saved_main_file is not None:
                main_module.__file__ = saved_main_file
            main_module.__spec__ = saved_main_spec


_project_source_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


_BATCH_TASK_COUNT = None


@contextmanager
def safe_forkserver_pool(processes, initializer=None, initargs=()):
    """Yield a ForkserverPool while guarding __main__."""

    with main_module_guard():
        with ForkserverPool(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
        ) as pool:
            yield pool


@_original_njit(cache=False)
def _numba_registry_warmup(value):
    return value + 1


_STARTUP_TARGET = None


_numba_registry_is_warm = False


_STARTUP_READY = None


def ensure_numba_registry_warmup():
    """Bind Numba's lazy internal registries once, only when requested."""
    global _numba_registry_is_warm
    if _numba_registry_is_warm:
        return

    project_njit_wrapper = _numba.njit
    try:
        _numba.njit = _original_njit
        _numba_registry_warmup(0)
        _numba_registry_is_warm = True
    finally:
        _numba.njit = project_njit_wrapper


_I_HAVE_EXTRA = False


original_njit = _original_njit


_LAST_SEEN_GENERATION = None


def _is_project_njit_function(function):
    code = getattr(function, '__code__', None)
    filename = None if code is None else code.co_filename
    if not filename or filename.startswith('<'):
        return False
    try:
        source_path = os.path.realpath(filename)
        return os.path.commonpath((
            _project_source_root, source_path
        )) == _project_source_root
    except ValueError:
        return False


_LAST_APPLIED_THREADS = None


def _caching_njit(*args, **kwargs):
    if 'cache' in kwargs:
        return _original_njit(*args, **kwargs)

    # Bare @njit passes the function directly.  Parameterized forms such
    # as @njit(parallel=True) need to defer the source-path decision until
    # Numba calls the returned decorator with the function.
    if args and hasattr(args[0], '__code__'):
        resolved_kwargs = kwargs.copy()
        if _is_project_njit_function(args[0]):
            resolved_kwargs['cache'] = True
        return _original_njit(*args, **resolved_kwargs)

    def _decorate(function):
        resolved_kwargs = kwargs.copy()
        if _is_project_njit_function(function):
            resolved_kwargs['cache'] = True
        return _original_njit(
            *args, **resolved_kwargs
        )(function)

    return _decorate


_LOG = "BHD_DYNTHREAD_LOG" in os.environ


_numba.njit = _caching_njit


_LAST_LOGGED = None


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
    old = _numba.get_num_threads()
    _numba.set_num_threads(n_threads)
    try:
        yield
    finally:
        _numba.set_num_threads(old)


def _counter_read(counter):
    """Lock-free read of a shared counter's current value.  The pool wires the
    counters as mp.Value('i') (lock=True), whose `.value` property acquires the
    value's lock on EVERY read; polled in tight loops across many workers, that
    one shared lock serialises the whole pool.  Reading the underlying ctypes
    object skips the lock -- an aligned int read is atomic, and a slightly stale
    count is fine here (we recheck constantly).  Falls back to `.value` if the
    counter has no get_obj (e.g. a plain value)."""
    obj = counter.get_obj() if hasattr(counter, "get_obj") else counter
    return obj.value


def set_dynamic_thread_state(
    total_cores,
    active_counter,
    extra_counter=None,
    started_counter=None,
    participant_counter=None,
    batch_generation=None,
    batch_task_count=None,
    startup_target=None,
    startup_ready=None,
):
    """Wire this worker process to the pool-wide counters.  Call once from the
    pool initializer.  Resets the per-worker extra-claim flag (essential when a
    Pool recycles workers — a respawned worker must not inherit a stale claim).

    active_counter may be None for the single-process sequential path: then
    every helper returns threads = 1 regardless of total_cores.
    """
    global _ACTIVE_COUNTER, _TOTAL_CORES, _EXTRA_COUNTER
    global _STARTED_COUNTER, _PARTICIPANT_COUNTER, _BATCH_GENERATION
    global _BATCH_TASK_COUNT, _STARTUP_TARGET, _STARTUP_READY
    global _I_HAVE_EXTRA, _LAST_SEEN_GENERATION, _LAST_APPLIED_THREADS
    _ACTIVE_COUNTER = active_counter
    _TOTAL_CORES = total_cores
    _EXTRA_COUNTER = extra_counter
    _STARTED_COUNTER = started_counter
    _PARTICIPANT_COUNTER = participant_counter
    _BATCH_GENERATION = batch_generation
    _BATCH_TASK_COUNT = batch_task_count
    _STARTUP_TARGET = startup_target
    _STARTUP_READY = startup_ready
    _I_HAVE_EXTRA = False
    _LAST_SEEN_GENERATION = None
    _LAST_APPLIED_THREADS = None


def increment_active():
    """Register this worker as active and as started in the current batch.

    Startup readiness normally requires ``min(tasks, workers)`` distinct
    workers. Counting distinct participants, rather than task starts, prevents
    one fast worker from opening the gate before slower pool workers have
    participated. If fewer distinct workers service a short batch, starting
    every submitted task also opens the gate: no queued task can then introduce
    a late active worker, so tail expansion is safe and cannot remain stuck at
    one thread.
    """
    global _LAST_SEEN_GENERATION
    if _ACTIVE_COUNTER is not None:
        with _ACTIVE_COUNTER.get_lock():
            obj = _ACTIVE_COUNTER.get_obj()
            obj.value += 1

    started = 0
    if _STARTED_COUNTER is not None:
        with _STARTED_COUNTER.get_lock():
            obj = _STARTED_COUNTER.get_obj()
            obj.value += 1
            started = obj.value

    participants = 0
    if _PARTICIPANT_COUNTER is not None and _BATCH_GENERATION is not None:
        generation = _counter_read(_BATCH_GENERATION)
        if _LAST_SEEN_GENERATION != generation:
            with _PARTICIPANT_COUNTER.get_lock():
                obj = _PARTICIPANT_COUNTER.get_obj()
                obj.value += 1
                participants = obj.value
            _LAST_SEEN_GENERATION = generation
        else:
            participants = _counter_read(_PARTICIPANT_COUNTER)

    if _STARTUP_READY is not None:
        target = (
            _counter_read(_STARTUP_TARGET)
            if _STARTUP_TARGET is not None else 1
        )
        task_count = (
            _counter_read(_BATCH_TASK_COUNT)
            if _BATCH_TASK_COUNT is not None else target
        )
        if participants >= target or started >= task_count:
            with _STARTUP_READY.get_lock():
                _STARTUP_READY.get_obj().value = 1


def decrement_active():
    """Deregister this worker (atomic).  No-op on the sequential path."""
    if _ACTIVE_COUNTER is not None:
        with _ACTIVE_COUNTER.get_lock():
            obj = _ACTIVE_COUNTER.get_obj()
            obj.value -= 1


def active_value():
    """Current active-worker count (raw), or 1 on the sequential path.
    Lock-free read — used for diagnostics/logging."""
    if _ACTIVE_COUNTER is None:
        return 1
    return _counter_read(_ACTIVE_COUNTER)


def _try_claim_extra(remainder):
    """Atomically attempt to claim an extra thread from the remainder pool.

    Returns True if claimed (and sets _I_HAVE_EXTRA), False otherwise.
    Idempotent: re-calling while already holding does not double-claim.

    Race analysis: the increment is guarded by the counter's own lock, and
    `current < remainder` is evaluated INSIDE the lock so two workers can't
    both observe `remainder - 1` and both push the counter to `remainder + 1`.
    The local `_I_HAVE_EXTRA = True` happens-after the increment (same thread).
    """
    global _I_HAVE_EXTRA
    if _I_HAVE_EXTRA:
        return True
    if _EXTRA_COUNTER is None:
        return False
    try:
        with _EXTRA_COUNTER.get_lock():
            obj = _EXTRA_COUNTER.get_obj()
            if obj.value < remainder:
                obj.value += 1
                _I_HAVE_EXTRA = True
                return True
    except Exception:
        pass
    return False


def _try_release_extra():
    """Atomically release this worker's extra claim, if held.  Defensive: clears
    the local flag even if the shared counter mutation fails."""
    global _I_HAVE_EXTRA
    if not _I_HAVE_EXTRA:
        return False
    if _EXTRA_COUNTER is None:
        _I_HAVE_EXTRA = False
        return False
    try:
        with _EXTRA_COUNTER.get_lock():
            obj = _EXTRA_COUNTER.get_obj()
            obj.value -= 1
            _I_HAVE_EXTRA = False
            return True
    except Exception:
        _I_HAVE_EXTRA = False
        return False


def release_dynamic_extra():
    """Release any +1 claim this worker holds WITHOUT tearing down the wiring.
    Call in a worker's per-task finally (the worker keeps its counter wiring
    across tasks for Pool reuse, but must not carry an extra-claim into its
    idle gap or the remainder pool leaks).  No-op when no claim / no counter."""
    _try_release_extra()


def _validated_thread_limit(max_threads):
    """Return a positive integer phase limit, or None when uncapped."""
    if max_threads is None:
        return None
    if (
        isinstance(max_threads, bool)
        or int(max_threads) != max_threads
        or int(max_threads) < 1
    ):
        raise ValueError("max_threads must be a positive integer or None")
    return int(max_threads)


def get_dynamic_threads(max_threads=None):
    """Compute this worker's thread count from the live active-peer count:
    floor(total_cores / active) + (1 if this worker holds an extra), clamped
    to >= 1.  floor+extra is always <= total_cores.

    The counter reads are lock-free (via _counter_read) — a slightly stale
    count is fine since we recheck at every phase, and at high call rates a
    locking read would serialise the pool.  The extra-counter lock is held only
    briefly on a claim/release transition (not on every call once stabilised).

    ``max_threads`` is an optional phase-specific cap.  It limits this worker
    without changing the pool-wide allocation policy; a worker capped at or
    below the common floor releases any stale +1 remainder claim so an
    uncapped peer can use it.

    Returns 1 on the sequential path (active_counter unset).
    """
    thread_limit = _validated_thread_limit(max_threads)
    if _ACTIVE_COUNTER is None or _TOTAL_CORES is None:
        return 1

    # Keep initial tasks at one thread until min(batch size, worker count)
    # tasks have started.  The flag is latched, so normal active-count tail
    # expansion takes over permanently after the initial wave has formed.
    if _STARTUP_READY is not None and not _counter_read(_STARTUP_READY):
        _try_release_extra()
        return 1

    active = max(_counter_read(_ACTIVE_COUNTER), 1)
    floor = _TOTAL_CORES // active
    remainder = _TOTAL_CORES - floor * active

    if thread_limit is not None and thread_limit <= floor:
        _try_release_extra()
        return max(1, min(floor, thread_limit))

    # Adjust the extra-claim based on the current remainder.
    if _EXTRA_COUNTER is not None:
        try:
            current_extras = _counter_read(_EXTRA_COUNTER)
        except Exception:
            current_extras = 0
        if not _I_HAVE_EXTRA:
            if current_extras < remainder:
                _try_claim_extra(remainder)
        else:
            if current_extras > remainder:
                _try_release_extra()

    allocated = max(1, floor + (1 if _I_HAVE_EXTRA else 0))
    if thread_limit is not None:
        allocated = min(allocated, thread_limit)
    return allocated


def apply_dynamic_threads(max_threads=None):
    """Recompute and apply this worker's numba thread allocation.  Call at
    every major / intermediate phase boundary of a long task.  Cheap (a
    lock-free read + numba.set_num_threads, which only affects subsequently-
    entered parallel regions).  ``max_threads`` optionally caps this phase.
    Returns the thread count actually applied; returns 1 on the sequential
    path.
    """
    global _LAST_APPLIED_THREADS
    n = get_dynamic_threads(max_threads=max_threads)
    # Most neighbouring phase boundaries request the same allocation. Avoid
    # rebuilding Numba's thread mask in that common case, while checking the
    # live value so direct, phase-local caps are restored on the next call.
    current = int(numba.get_num_threads())
    if _LAST_APPLIED_THREADS != n or current != n:
        numba.set_num_threads(n)
        current = int(numba.get_num_threads())
    n = current
    _LAST_APPLIED_THREADS = n
    _log_alloc(n)
    return n


def _log_alloc(n):
    """Trace thread-count transitions when BHD_DYNTHREAD_LOG is set (diagnostics
    only): one stderr line per worker whenever its applied count changes, with
    the live (active, floor, remainder, extra) it was derived from.  No-op when
    logging is off or the count is unchanged, so it never floods a long run."""
    global _LAST_LOGGED
    if not _LOG or n == _LAST_LOGGED:
        return
    _LAST_LOGGED = n
    try:
        active = _counter_read(_ACTIVE_COUNTER) if _ACTIVE_COUNTER is not None else 1
        extras = _counter_read(_EXTRA_COUNTER) if _EXTRA_COUNTER is not None else 0
        floor = (_TOTAL_CORES // max(active, 1)) if _TOTAL_CORES else 1
        rem = (_TOTAL_CORES - floor * max(active, 1)) if _TOTAL_CORES else 0
        print("[dynthreads pid=%d t=%.1f] active=%d total=%s floor=%d rem=%d "
              "extras=%d mine=%s -> threads=%d"
              % (os.getpid(), time.monotonic(), active, _TOTAL_CORES, floor,
                 rem, extras, _I_HAVE_EXTRA, n),
              file=sys.stderr, flush=True)
    except Exception:
        pass
