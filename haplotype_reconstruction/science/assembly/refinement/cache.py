"""Reuse window results without changing template order or charged logical work.

Owns a per-search proposal cache and the bounded cross-root
memo. Must not open storage, choose scientific budgets, or inspect free memory.
Method: docs/methods/founder_refinement.md#fixed-k-search (RS04, RS06, RS11).
Cache keys bind row/flank bytes and convention. A memo hit restores
its logical work receipt while doing no new physical programme work.
"""

from __future__ import annotations

import pickle
import threading
from collections import OrderedDict

import numpy as np

from .parameters import WINDOW_MEMO_BYTES
from .parameters import WINDOW_MEMO_ENTRIES as _WINDOW_MEMO_ENTRIES
from .parameters import WINDOW_MEMO_OVERHEAD as _WINDOW_MEMO_OVERHEAD
from .workspace import Orientation

__all__ = ["Orientation", "ProposalCache", "WindowMemo"]


# Only components with multiple incoming roots retain these optional window results.
# One MiB of the reservation covers the bounded dict/entry bookkeeping. Input-key
# copies and returned proposal arrays are covered by each worker's task estimate.


class WindowMemo:
    """Hybrid window results within one immutable component solver.

    Scope fixes emissions, row domains, templates, coarse context and penalty.
    Keys additionally compare every row/flank byte, dtype, shape and explicit
    convention. The immutable solver fixes the hybrid frontier and search controls.
    Dictionary hashing is only an index, never equality.
    Stored serialized results are immutable; each hit returns privately owned
    arrays and diagnostics. Concurrent misses may duplicate work, not decisions.
    No solver or engine reference is retained, avoiding large ownership cycles.
    """

    def __init__(self, limit=WINDOW_MEMO_BYTES - _WINDOW_MEMO_OVERHEAD):
        self.limit = int(limit)
        self.entries = OrderedDict()
        self.lock = threading.Lock()
        self.used = 0
        self.statistics = dict(calls=0, hits=0, evictions=0, peak_entry_bytes=0)

    @staticmethod
    def key(index, a, c, X, fwd, bwd, convention):
        def array(value):
            return value.dtype.str, value.shape, value.tobytes(order="C")

        return (int(index), bool(convention), array(X[:, a:c]), array(fwd[a]), array(bwd[c]))

    @staticmethod
    def transient_bytes(a, c, X, fwd, bwd):
        # Complete input key plus a serialized/private copy of all focal paths
        # and their small diagnostic dictionaries. Existing estimates cover the
        # original solved result and numerical tables.
        return int(X[:, a:c].nbytes + fwd[a].nbytes + bwd[c].nbytes) + 16 * len(X) * (c - a) + 2048 * len(X) + 4096

    @staticmethod
    def _size(value):
        # Fixed dimension estimate, independent of interpreter object layout.
        # Sixty-four bytes per scalar/tuple header plus the held byte payload.
        if isinstance(value, tuple):
            return 64 + 8 * len(value) + sum(WindowMemo._size(x) for x in value)
        if isinstance(value, (bytes, str)):
            return 64 + len(value)
        return 64

    def lookup(self, key):
        with self.lock:
            self.statistics["calls"] += 1
            entry = self.entries.get(key)
            if entry is None:
                return None
            self.entries.move_to_end(key)
            self.statistics["hits"] += 1
        return pickle.loads(entry[0])

    def store(self, key, result):
        payload = pickle.dumps(result, protocol=5)
        size = self._size(key) + (64 + len(payload)) + 256
        if size > self.limit:
            return
        with self.lock:
            if key in self.entries:
                return  # Another worker completed the identical problem.
            while self.used + size > self.limit or len(self.entries) >= _WINDOW_MEMO_ENTRIES:
                _, old = self.entries.popitem(last=False)
                self.used -= old[1]
                self.statistics["evictions"] += 1
            self.entries[key] = payload, size
            self.used += size
            self.statistics["peak_entry_bytes"] = max(self.statistics["peak_entry_bytes"], self.used)

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.used = 0


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
