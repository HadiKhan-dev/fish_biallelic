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


_ACTIVE_COUNTER = None
_TOTAL_CORES = None
_EXTRA_COUNTER = None
_I_HAVE_EXTRA = False


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
            _ACTIVE_COUNTER.value += 1


def decrement_active():
    """Deregister this worker (atomic).  No-op on the sequential path."""
    if _ACTIVE_COUNTER is not None:
        with _ACTIVE_COUNTER.get_lock():
            _ACTIVE_COUNTER.value -= 1


def active_value():
    """Current active-worker count (raw), or 1 on the sequential path.
    Lock-free read — used for diagnostics/logging."""
    if _ACTIVE_COUNTER is None:
        return 1
    return _ACTIVE_COUNTER.value


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
            if _EXTRA_COUNTER.value < remainder:
                _EXTRA_COUNTER.value += 1
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
            _EXTRA_COUNTER.value -= 1
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

    The active read is intentionally lock-free — a slightly stale count is fine
    since we recheck at every phase.  The extra-counter lock is held only
    briefly on a claim/release transition (not on every call once stabilised).

    Returns 1 on the sequential path (active_counter unset).
    """
    if _ACTIVE_COUNTER is None or _TOTAL_CORES is None:
        return 1
    active = max(_ACTIVE_COUNTER.value, 1)
    floor = _TOTAL_CORES // active
    remainder = _TOTAL_CORES - floor * active

    # Adjust the extra-claim based on the current remainder.
    if _EXTRA_COUNTER is not None:
        try:
            current_extras = _EXTRA_COUNTER.value
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
    return n