"""
Dynamic numba thread reallocation — shared, pool-wide.

A forkserver pool processing a fixed set of tasks via imap_unordered hits a
straggler tail: when only a few long tasks remain, the pool has no more work
to hand its idle workers, so those tasks run on whatever thread count they
were given at start while the freed cores sit idle.  This module lets a task
RE-CHECK the live active-worker count at its phase boundaries and rescale its
numba threads, so a straggler grows into the freed cores.

Mechanism (one shared state per worker process, wired by the pool initializer
via set_dynamic_thread_state):

  _ACTIVE_COUNTER : mp.Value('i') — # tasks currently active pool-wide.  The
                    pool increments/decrements it around each task
                    (increment_active / decrement_active).
  _TOTAL_CORES    : int           — total cores available to the pool.
  _EXTRA_COUNTER  : mp.Value('i') — # workers currently holding a +1 thread
                    (remainder distribution).  Optional: if None, allocation
                    is floor-only (the remainder, < active, is left idle).
  _I_HAVE_EXTRA   : per-worker-process bool, True iff this worker holds the +1.

get_dynamic_threads() returns floor(total/active) + (1 if this worker holds an
extra).  With an extra_counter, exactly `total % active` workers hold the +1,
so total threads in use across the pool == total_cores (zero idle cores).

All state is None/False outside an initialised pool worker, so every helper
no-ops (threads = 1) on the sequential path.  Importing this module has no
side effects.
"""

import multiprocessing  # noqa: F401  (documents the mp.Value contract)
import os
import sys
import time


_ACTIVE_COUNTER = None
_TOTAL_CORES = None
_EXTRA_COUNTER = None
_I_HAVE_EXTRA = False

# Optional diagnostics: set BHD_DYNTHREAD_LOG=1 to trace per-worker thread-count
# transitions on stderr (one line whenever a worker's applied count changes), so
# a straggler ramping into freed cores -- or failing to -- is visible without
# flooding the log.  Off by default: zero overhead beyond an env lookup at import.
_LOG = "BHD_DYNTHREAD_LOG" in os.environ
_LAST_LOGGED = None


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


# -------------------------------------------------------------------------
# Wiring (called by the pool initializer / sequential-path setup)
# -------------------------------------------------------------------------
def set_dynamic_thread_state(total_cores, active_counter, extra_counter=None):
    """Wire this worker process to the pool-wide counters.  Call once from the
    pool initializer.  Resets the per-worker extra-claim flag (essential when a
    Pool recycles workers — a respawned worker must not inherit a stale claim).

    active_counter may be None for the single-process sequential path: then
    every helper returns threads = 1 regardless of total_cores.
    """
    global _ACTIVE_COUNTER, _TOTAL_CORES, _EXTRA_COUNTER, _I_HAVE_EXTRA
    _ACTIVE_COUNTER = active_counter
    _TOTAL_CORES = total_cores
    _EXTRA_COUNTER = extra_counter
    _I_HAVE_EXTRA = False


def clear_dynamic_thread_state():
    """Full teardown: release any extra-claim and drop all wiring."""
    global _ACTIVE_COUNTER, _TOTAL_CORES, _EXTRA_COUNTER, _I_HAVE_EXTRA
    _try_release_extra()
    _ACTIVE_COUNTER = None
    _TOTAL_CORES = None
    _EXTRA_COUNTER = None
    _I_HAVE_EXTRA = False


# -------------------------------------------------------------------------
# Active-count management (so callers never touch the counter directly)
# -------------------------------------------------------------------------
def increment_active():
    """Register this worker as active (atomic).  No-op on the sequential path."""
    if _ACTIVE_COUNTER is not None:
        with _ACTIVE_COUNTER.get_lock():
            obj = _ACTIVE_COUNTER.get_obj()
            obj.value += 1


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


# -------------------------------------------------------------------------
# Remainder distribution (extra-claim) — atomic claim/release
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# The allocation itself
# -------------------------------------------------------------------------
def get_dynamic_threads():
    """Compute this worker's thread count from the live active-peer count:
    floor(total_cores / active) + (1 if this worker holds an extra), clamped
    to >= 1.  floor+extra is always <= total_cores.

    The counter reads are lock-free (via _counter_read) — a slightly stale
    count is fine since we recheck at every phase, and at high call rates a
    locking read would serialise the pool.  The extra-counter lock is held only
    briefly on a claim/release transition (not on every call once stabilised).

    Returns 1 on the sequential path (active_counter unset).
    """
    if _ACTIVE_COUNTER is None or _TOTAL_CORES is None:
        return 1
    active = max(_counter_read(_ACTIVE_COUNTER), 1)
    floor = _TOTAL_CORES // active
    remainder = _TOTAL_CORES - floor * active

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

    return max(1, floor + (1 if _I_HAVE_EXTRA else 0))


def apply_dynamic_threads():
    """Recompute and apply this worker's numba thread allocation.  Call at
    every major / intermediate phase boundary of a long task.  Cheap (a
    lock-free read + numba.set_num_threads, which only affects subsequently-
    entered parallel regions).  Returns the thread count applied; no-op
    (returns 1) on the sequential path.
    """
    n = get_dynamic_threads()
    try:
        import numba
        numba.set_num_threads(n)
    except Exception:
        pass
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