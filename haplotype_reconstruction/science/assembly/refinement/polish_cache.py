"""Reuse the polish's local numerical answers within one immutable component.

Owns last-problem beam, scalar-gain and plain bridge-window stores per fixed-K search
and the bounded answers all polish fits of one component share. It never
selects variants, accepts moves or retains a workspace. Method:
docs/methods/founder_refinement.md#polish (RS15–RS16, RC22).
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import numpy as np

from .parameters import LONG_CORRECTION, LONG_STRIDE, LONG_WINDOW


def digest(*parts) -> bytes:
    """Content key of fixed-shape numerical inputs, including array dtype/shape."""
    key = hashlib.blake2b(digest_size=16)
    for part in parts:
        if isinstance(part, np.ndarray):
            key.update(repr((part.dtype.str, part.shape)).encode())
            key.update(np.ascontiguousarray(part).tobytes())
        else:
            key.update(repr(part).encode() + b"|")
    return key.digest()


def label_frame(rows):
    """Window row order, inverse founder map and unordered-pair state map.

    Equal local rows retain founder-index order. Such ties may lose a hit; equal
    complete framed keys still identify the same problem under one permutation.
    No row aliases or block-specific permutations enter the comparison.
    """
    founders = len(rows)
    order = np.asarray(sorted(range(founders), key=lambda f: (rows[f].tobytes(), f)), np.int64)
    position = np.empty(founders, np.int64)
    position[order] = np.arange(founders)
    lo, hi = np.minimum(order[:, None], order), np.maximum(order[:, None], order)
    index = lo * founders - lo * (lo - 1) // 2 + hi - lo
    states = np.ascontiguousarray(index[np.triu_indices(founders)], np.int64)
    return order, position, states


def store_bytes(founders, beam_spans, long_spans, bridge_spans):
    """Dimension-only residence bound (bytes); see PolishStores for the formula."""
    k = int(founders)
    beam = sum(512 + k * (640 + 8 * w) for w in beam_spans)
    gains = 0
    step = max(1, LONG_CORRECTION // 2)
    # At most one long beam offer per focal; across relabellings all K donors
    # can enter its corrections. One more offer per focal covers bridge gains.
    for w in long_spans:
        candidates = k * (2 + k * ((w + step - 1) // step))
        gains += 512 + candidates * (256 + 8 * w)
    gains += sum(512 + k * (256 + 8 * w) for w in bridge_spans)
    bridge = sum(512 + (w + 1) * (512 + 8 * w) for w in bridge_spans)
    return int(beam + gains + bridge + 512)


def search_store_bytes(founders, blocks, templates):
    """Store residence for the component's fixed beam templates and long windows."""
    starts = sorted(set(range(0, max(1, blocks - LONG_WINDOW + 1), LONG_STRIDE)) | {max(0, blocks - LONG_WINDOW)})
    long = [min(blocks, a + LONG_WINDOW) - a for a in starts]
    return store_bytes(
        founders, [c - a for _, a, c in templates] + long, long, [c - a for kind, a, c in templates if kind == "fine"]
    )


class PolishStores:
    """Last local problems of one polish fit's forward-first search, never recorded.

    Evidence/sample order, masks, prepared rows, binning, context alphabets,
    emissions, map costs, K and numerical/search controls are fixed by ownership.
    Each slot is replaced on an input miss; nothing crosses fits, K or resume.
    Only forward-oriented gains are stored, including after a reverse sweep.
    Beam results use a single window founder frame; scalar gains additionally
    name the mapped focal and actual candidate rows. Bridge selections retain
    plain labels because precursor/pair ties read founder indices. Long-repair
    variants are regenerated each time, never stored as a selected answer.

    Size bound per fit, independent of sweep count: for K founders, beam-template
    widths U_t, long-window block widths W_l, fine bridge widths V_b and
    J(w) = ceil(w / max(1, LONG_CORRECTION // 2)), at most
    sum_t [512 + K(640 + 8 U_t)]
    + sum_l [512 + K(2 + K J(W_l))(256 + 8 W_l)]
    + sum_b [1024 + K(256 + 8 V_b) + (V_b + 1)(512 + 8 V_b)]
    + 512 bytes. Metadata allowances include Python containers and digest keys;
    candidate rows are int64 bytes. The long bound includes every possible donor
    across index-tied relabellings, not just the two selected in one invocation.
    Overlapping long/bridge gain slots are counted twice conservatively. Beam
    spans count fine and coarse units; bridge spans count blocks. Hash collisions
    have the usual content-digest qualification. No full flank array is retained.
    Temporary frames/key gathers are covered by the numerical working set.

    share (set by the owning search, else None): the component's PolishShare. A slot
    miss then consults the answers of every polish fit of the component under the same
    complete content key (the primary's answers serve an alternative that meets the
    identical framed problem), and every slot write is offered to it. The share is
    bounded separately (PolishShare); the slots keep this bound.
    """

    def __init__(self):
        self.beam = {}
        self.gains = {}
        self.bridge = {}
        self.share = None

    def beam_problem(self, kind, a, c, width, convention, X, left, right):
        """Framed key and founder maps of a beam template (a/c are unit indices)."""
        order, position, states = label_frame(X)
        key = digest(kind, int(a), int(c), int(width), bool(convention), X[order], left[:, states], right[:, states])
        return key, order, position

    def _name(self, table):
        return "beam" if table is self.beam else "gains" if table is self.gains else "bridge"

    def lookup(self, table, slot, key):
        """Last answer in this slot, dropping the previous problem on a miss; on a miss, the
        component's shared answer to the identical keyed problem (installed in the slot)."""
        entry = table.get(slot)
        if entry is not None:
            if entry[0] == key:
                return entry[1]
            del table[slot]
        if self.share is None:
            return None
        found = self.share.get(self._name(table), key)
        if found is not None:
            table[slot] = key, found
        return found

    def put(self, table, slot, key, value):
        """Replace this slot's answer and offer it to the component's share."""
        table[slot] = key, value
        if self.share is not None:
            self.share.put(self._name(table), key, value)

    def gain_table(self, slot, key):
        """Scalar answers for this window problem, replaced together on a miss."""
        table = self.lookup(self.gains, slot, key)
        if table is None:
            table = {} if self.share is None else SharedGains(self.share, key)
            self.put(self.gains, slot, key, table)
        return table


def _answer_bytes(value) -> int:
    """Residence estimate of a stored answer: array payloads plus 64 bytes per object."""
    if isinstance(value, np.ndarray):
        return 128 + int(value.nbytes)
    if isinstance(value, (tuple, list)):
        return 64 + 8 * len(value) + sum(_answer_bytes(item) for item in value)
    if isinstance(value, dict):
        return 256 + sum(_answer_bytes(k) + _answer_bytes(v) for k, v in value.items())
    if isinstance(value, (bytes, str)):
        return 64 + len(value)
    return 64


class SharedGains(dict):
    """A scalar-gain table held in the component's share: every new candidate charges its bytes there."""

    __slots__ = ("share", "token")

    def __init__(self, share, key):
        dict.__init__(self)
        self.share, self.token = share, ("gains", key)

    def __setitem__(self, key, value):
        fresh = key not in self
        dict.__setitem__(self, key, value)
        if fresh:
            self.share.grow(self.token, _answer_bytes(key) + 64)


class PolishShare:
    """Polish answers shared by every polish fit of one component attempt.

    Entries are the per-search slot answers (beam template answers in their window founder
    frame, scalar-gain tables, plain bridge trials) keyed by (table, the complete content
    digest of the slot's problem), exactly the keys under which a single search reuses them
    (RS15-RS17): an entry is the answer to the identical problem, whichever fit computed it.
    Concurrent polish fits may complete one gain table together: its values are pure
    functions of the same framed problem and candidate. Residence is bounded by ``limit``
    bytes (estimated from the held arrays and objects; a gain table is charged as it
    grows); least recently used entries are evicted first, which changes runtime only
    (an evicted answer still held by a search's slot stays valid there).
    """

    def __init__(self, limit):
        self.limit = int(limit)
        self.entries = OrderedDict()
        self.lock = threading.Lock()
        self.used = 0
        self.statistics = dict(calls=0, hits=0, evictions=0, peak_entry_bytes=0)

    def get(self, name, key):
        with self.lock:
            self.statistics["calls"] += 1
            entry = self.entries.get((name, key))
            if entry is None:
                return None
            self.entries.move_to_end((name, key))
            self.statistics["hits"] += 1
            return entry[0]

    def _evict(self, keep=None):
        while self.used > self.limit and self.entries:
            token = next(iter(self.entries))
            if token == keep:
                if len(self.entries) == 1:
                    break
                self.entries.move_to_end(token)
                continue
            _, size = self.entries.pop(token)
            self.used -= size
            self.statistics["evictions"] += 1

    def put(self, name, key, value):
        size = 128 + len(key) + _answer_bytes(value)
        with self.lock:
            if (name, key) in self.entries or size > self.limit:
                return
            self.entries[(name, key)] = [value, size]
            self.used += size
            self._evict(keep=(name, key))
            self.statistics["peak_entry_bytes"] = max(self.statistics["peak_entry_bytes"], self.used)

    def grow(self, token, size):
        with self.lock:
            entry = self.entries.get(token)
            if entry is None:
                return  # Evicted: the table lives on only in the slots that hold it.
            entry[1] += size
            self.used += size
            self._evict(keep=token)
            self.statistics["peak_entry_bytes"] = max(self.statistics["peak_entry_bytes"], self.used)

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.used = 0


class GainBook:
    """One invocation's forward-oriented view of the fit's scalar-gain windows.

    The framed current rows and exact forward flank at a / backward flank at c
    determine g_start and scale_sum by backward recursion over [a,c). Those
    redundant inputs need no separate key under block-normalised messages.
    Boundary reconstruction stays on the search thread; only digests and scalar
    tables survive this view. Donor screening and best-variant selection stay fresh.
    """

    def __init__(self, stores, X, fwd, bwd):
        self.stores, self.X, self.fwd, self.bwd = stores, X, fwd, bwd
        self.tables = {}

    def table(self, a, c):
        """Gain table and current-to-framed focal map of window [a,c)."""
        slot = int(a), int(c)
        if slot not in self.tables:
            order, position, states = label_frame(self.X[:, a:c])
            key = digest(int(a), int(c), self.X[order, a:c], self.fwd[a][:, states], self.bwd[c][:, states])
            self.tables[slot] = self.stores.gain_table(slot, key), position
        return self.tables[slot]

    def candidate(self, cand):
        """Table and same-candidate key, without a cached variant choice."""
        a, c, focal, rows = cand
        table, position = self.table(a, c)
        return table, (int(position[focal]), np.asarray(rows, np.int64).tobytes())

    def lookup(self, cands):
        """Gains in current candidate order (None at misses), and miss indices."""
        values, missing = [], []
        for i, cand in enumerate(cands):
            table, key = self.candidate(cand)
            value = table.get(key)
            values.append(value)
            if value is None:
                missing.append(i)
        return values, missing

    def remember(self, cand, gain):
        """Retain only this candidate's scalar gain, in its window's frame."""
        table, key = self.candidate(cand)
        table[key] = float(gain)

    def window_key(self, a, c, focal, rows, gain):
        """Plain bridge key; index-dependent precursor/pair choices cannot map."""
        return digest(
            int(a),
            int(c),
            self.X[:, a:c],
            self.fwd[a],
            self.bwd[c],
            int(focal),
            np.asarray(rows, np.int64),
            float(gain).hex(),
        )
