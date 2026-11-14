"""Reuse window results without changing template order or charged logical work.

Owns a per-search proposal cache and the component's window-result store. Must not
open storage, choose scientific budgets, or inspect free memory.
Method: docs/methods/founder_refinement.md#window-store (RS04, RS06, RS11, RC22).
Store keys are content digests of everything a window solve reads. A store hit
restores its logical work receipt while doing no new physical programme work.
"""

from __future__ import annotations

import pickle
import threading
from collections import OrderedDict

import numpy as np

from .parameters import WINDOW_STORE_ENTRY_OVERHEAD, WINDOW_STORE_TRACK_BYTES, WINDOW_STORE_TRACKED_FITS
from .polish_cache import digest
from .workspace import Orientation

__all__ = ["Orientation", "ProposalCache", "WindowStore"]


class WindowStore:
    """Window-solve results of one component attempt, shared by every fit of it.

    The count pass, every incoming root (replayed or complete), every continuation
    branch and the polish fits read and fill one store. A window solve is a pure
    function of the template (index: kind and unit span), the content convention,
    the panel's rows in the window and the two flank message arrays at its ends,
    given one immutable component solver (emissions, row domains, template list,
    coarse context alphabets, penalty, frontier, seeded-search controls; the work
    account depends only on dimensions). The store is bound to that solver object
    (``solver``): every fit of the component shares it, and SweepEngine.proposals
    refuses a store bound to another solver, since a template index means nothing
    outside its own solver's list. K is in the window rows' and flanks' shapes.

    Key convention (one for every refinement store, polish_cache.digest): a 128-bit
    BLAKE2b digest of the template index, the convention, and the dtype, shape and
    bytes of the window rows and both flanks. Keys are digests, not the inputs: two
    different problems would share a result only on a 128-bit collision, with
    probability below n^2 / 2^129 for n stored problems (about 1e-28 at n = 2e5), the
    qualification the polish stores already carry (RS15). Stored results are
    serialised; each hit returns privately owned arrays and diagnostics.

    In-flight sharing: the first requester of a key solves it and publishes the
    result; a concurrent requester of the same key waits for that result instead of
    solving it again. A solve that raises (a work stop) publishes nothing and wakes
    its waiters, one of which then solves the problem itself, so a failure can only
    repeat work, never return a result. A template that exhausts its work account
    raises before it is stored, so the store never holds a stop.

    Retention: residence (payloads plus WINDOW_STORE_ENTRY_OVERHEAD per entry, plus
    WINDOW_STORE_TRACK_BYTES per tracked key) is bounded by ``limit`` bytes. Entries
    are evicted least recently used first, except pinned ones: the keys each count
    fit used are tracked under its fit token (the last WINDOW_STORE_TRACKED_FITS
    fits), and pin(token), called when a count history adopts that fit, exempts its
    entries from eviction while pinned bytes stay within half the limit. The adopted
    path is what replays of tied roots and continuation branches meet again after
    the nominal pass has moved on. Eviction, pinning and the budget change runtime
    only: a miss solves the identical problem again.
    """

    def __init__(self, limit, solver=None):
        self.limit = int(limit)
        self.solver = solver
        self.entries = OrderedDict()
        self.pinned = {}
        self.pinned_bytes = 0
        self.tracked = OrderedDict()
        self.pending = {}
        self.lock = threading.Lock()
        self.used = 0
        self.statistics = dict(calls=0, hits=0, waits=0, evictions=0, pinned=0, peak_entry_bytes=0)

    @staticmethod
    def key(index, a, c, X, fwd, bwd, convention):
        """Digest of (template index, convention, window rows, both flanks)."""
        return digest(int(index), bool(convention), X[:, a:c], fwd[a], bwd[c])

    @staticmethod
    def transient_bytes(a, c, X, fwd, bwd):
        # A serialized/private copy of all focal paths and their small diagnostic
        # dictionaries. Existing estimates cover the solved result and numerical tables.
        return 16 * len(X) * (c - a) + 2048 * len(X) + 4096

    def _evict(self):
        while self.used > self.limit and self.entries:
            _, old = self.entries.popitem(last=False)
            self.used -= old[1]
            self.statistics["evictions"] += 1

    def _track(self, tag, key):
        if tag is None:
            return
        keys = self.tracked.get(tag)
        if keys is None:
            keys = self.tracked[tag] = []
            while len(self.tracked) > WINDOW_STORE_TRACKED_FITS:
                _, old = self.tracked.popitem(last=False)
                self.used -= WINDOW_STORE_TRACK_BYTES * len(old)
        keys.append(key)
        self.used += WINDOW_STORE_TRACK_BYTES
        self._evict()

    def _entry(self, key):
        entry = self.pinned.get(key)
        if entry is None:
            entry = self.entries.get(key)
            if entry is not None:
                self.entries.move_to_end(key)
        return entry

    def claim(self, key, tag=None):
        """The stored result of key (privately owned), or None: the caller then owns the
        solve and must call release(key) after store() or on failure. Waits while another
        requester solves the same key. tag: the requesting count fit's token, or None."""
        waited = False
        while True:
            with self.lock:
                if not waited:
                    self.statistics["calls"] += 1
                entry = self._entry(key)
                if entry is not None:
                    self.statistics["hits"] += 1
                    self._track(tag, key)
                    payload = entry[0]
                    break
                event = self.pending.get(key)
                if event is None:
                    self.pending[key] = threading.Event()
                    return None
                if not waited:
                    self.statistics["waits"] += 1
                waited = True
            event.wait()
        return pickle.loads(payload)

    def store(self, key, result, tag=None):
        payload = pickle.dumps(result, protocol=5)
        size = len(payload) + WINDOW_STORE_ENTRY_OVERHEAD
        with self.lock:
            self._track(tag, key)
            if size > self.limit or self._entry(key) is not None:
                return  # Too large, or another worker completed the identical problem.
            self.entries[key] = payload, size
            self.used += size
            self._evict()
            self.statistics["peak_entry_bytes"] = max(self.statistics["peak_entry_bytes"], self.used)

    def release(self, key):
        """End the caller's claim on key (after store() or a failure); wake its waiters."""
        with self.lock:
            event = self.pending.pop(key, None)
        if event is not None:
            event.set()

    def pin(self, tag):
        """Exempt the entries the fit ``tag`` used from eviction (within half the limit)."""
        with self.lock:
            keys = self.tracked.pop(tag, ())
            self.used -= WINDOW_STORE_TRACK_BYTES * len(keys)
            for key in keys:
                entry = self.entries.get(key)
                if entry is None or self.pinned_bytes + entry[1] > self.limit // 2:
                    continue
                del self.entries[key]
                self.pinned[key] = entry
                self.pinned_bytes += entry[1]
                self.statistics["pinned"] += 1

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.pinned.clear()
            self.tracked.clear()
            self.used = self.pinned_bytes = 0


class ProposalCache:
    """One search's last solved window inputs and results (SweepEngine.proposals).

    Invariant: the cached result of template t was solved from the rows windows[t][0] and the
    boundary messages fwd[a_t], bwd[c_t] held here (a boundary's messages are replaced only when
    every template ending there is re-solved). Every result is a pure function of its inputs (the
    hybrid search controls are fixed), fallbacks included."""

    def __init__(self):
        self.fwd = {}
        self.bwd = {}
        self.windows = {}
        self._fwd_same = self._bwd_same = None

    def lookup(self, spans, X, fwd, bwd, results):
        """Fill results[t] = cached result for templates with unchanged inputs; return the
        others' indices."""
        self._fwd_same, self._bwd_same = {}, {}
        todo = []
        for index, (a, c) in enumerate(spans):
            held = self.windows.get(index)
            if held is not None:
                if a not in self._fwd_same:
                    self._fwd_same[a] = np.array_equal(self.fwd[a], fwd[a])
                if c not in self._bwd_same:
                    self._bwd_same[c] = np.array_equal(self.bwd[c], bwd[c])
                if self._fwd_same[a] and self._bwd_same[c] and np.array_equal(held[0], X[:, a:c]):
                    results[index] = held[1]
                    continue
            todo.append(index)
        return todo

    def store(self, spans, X, fwd, bwd, todo, computed):
        for index, result in zip(todo, computed, strict=False):
            a, c = spans[index]
            self.windows[index] = (X[:, a:c].copy(), result)
            if not self._fwd_same.get(a, False):
                self.fwd[a] = fwd[a].copy()
                self._fwd_same[a] = True
            if not self._bwd_same.get(c, False):
                self.bwd[c] = bwd[c].copy()
                self._bwd_same[c] = True
