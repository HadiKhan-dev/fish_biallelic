"""Reuse the polish's local numerical answers within one immutable fixed-K search.

Owns last-problem beam, scalar-gain and plain bridge-window stores. It never
selects variants, accepts moves or retains a workspace. Method:
docs/methods/founder_refinement.md#polish (RS15–RS16).
"""

from __future__ import annotations

import hashlib

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
    """

    def __init__(self):
        self.beam = {}
        self.gains = {}
        self.bridge = {}

    def beam_problem(self, kind, a, c, width, convention, X, left, right):
        """Framed key and founder maps of a beam template (a/c are unit indices)."""
        order, position, states = label_frame(X)
        key = digest(kind, int(a), int(c), int(width), bool(convention), X[order], left[:, states], right[:, states])
        return key, order, position

    @staticmethod
    def lookup(table, slot, key):
        """Last answer in this slot, dropping the previous problem on a miss."""
        entry = table.get(slot)
        if entry is not None:
            if entry[0] == key:
                return entry[1]
            del table[slot]
        return None

    def gain_table(self, slot, key):
        """Scalar answers for this window problem, replaced together on a miss."""
        table = self.lookup(self.gains, slot, key)
        if table is None:
            table = {}
            self.gains[slot] = key, table
        return table


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
