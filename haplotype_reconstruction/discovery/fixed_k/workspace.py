"""Evidence-only tables of one block's fixed-K model and the block's fit caches.

Everything here depends on the block's evidence and the model settings only, never on a panel,
so one workspace serves every fit of a block search (thousands of starts). Missing cells (no
depth) contribute nothing: zero log emissions, zero unknown-copy cost, and a sample without any
observed site is forced to the unknown-unknown state.

Per-site costs of a sample s at site l (lambda = unknown_copy_penalty, floor 1e-12):
* genotype g: -log max(GL[s, l, g], 1e-12) (``log_probs`` holds the logarithm);
* both copies unknown: the best genotype plus 2 lambda (``cost_ww``), the per-cell cap of the
  H step's costs;
* one known copy with allele h and one unknown copy: the best genotype in {h, h + 1} plus lambda.

The A step's binned emissions are contracted per (bin, within-bin allele pattern) of a row or of
a row pair (bitwise AND): ``PatternTables`` fills a pattern's three contraction rows the first
time a panel uses it, as one fixed ascending set-bit sum, so every value is the same float64 the
complete 2^snps_per_bin table would hold.
"""
from __future__ import annotations

from collections import OrderedDict
import math

import numpy as np
from numba import njit, prange

# Cache bounds by entry count (one block's search stays far below them).
FIT_CACHE_ENTRIES = 4096
TRANSITION_CACHE_ENTRIES = 16384
# Initial pattern slots per bin of the lazily filled tables (doubled on demand; a block uses
# 40-130 of the 1,024 patterns of a ten-site bin).
_PATTERN_INITIAL_SLOTS = 64


@njit(cache=True, parallel=True, nogil=True)
def _observed_log_emissions(likelihoods, observed, penalty):
    """log max(GL, 1e-12) at observed cells (0 elsewhere) and the unknown-unknown cost per cell
    (0 at unobserved cells)."""
    n, length, _ = likelihoods.shape
    log_probs = np.empty((n, length, 3), dtype=np.float64)
    cost_ww = np.empty((n, length), dtype=np.float64)
    floor = 1e-12
    for s in prange(n):
        for l in range(length):
            if not observed[s, l]:
                log_probs[s, l, 0] = 0.0
                log_probs[s, l, 1] = 0.0
                log_probs[s, l, 2] = 0.0
                cost_ww[s, l] = 0.0
                continue
            maximum = -math.inf
            for genotype in range(3):
                value = likelihoods[s, l, genotype]
                if value < floor:
                    value = floor
                log_value = math.log(value)
                log_probs[s, l, genotype] = log_value
                if log_value > maximum:
                    maximum = log_value
            cost_ww[s, l] = -maximum + 2.0 * penalty
    return log_probs, cost_ww


@njit(cache=True, parallel=True, nogil=True)
def _capped_update_costs(cost_ww, log_probs, observed, penalty):
    """The H step's capped costs: per genotype (N, L, 3) and per known-allele of a row paired
    with an unknown copy (N, L, 2), each capped at the cell's unknown-unknown cost; the
    unknown-unknown total per sample (summed left to right)."""
    n, length = cost_ww.shape
    ww_total = np.empty(n, dtype=np.float64)
    genotype_cost = np.empty((n, length, 3), dtype=np.float64)
    unknown_cost = np.empty((n, length, 2), dtype=np.float64)
    for s in prange(n):
        total = 0.0
        for l in range(length):
            cap = cost_ww[s, l]
            total += cap
            lp0 = log_probs[s, l, 0]
            lp1 = log_probs[s, l, 1]
            lp2 = log_probs[s, l, 2]
            for genotype in range(3):
                value = -log_probs[s, l, genotype]
                if value > cap:
                    value = cap
                genotype_cost[s, l, genotype] = value
            best0 = lp0 if lp0 > lp1 else lp1
            best1 = lp1 if lp1 > lp2 else lp2
            penalty_here = penalty if observed[s, l] else 0.0
            value0 = -best0 + penalty_here
            value1 = -best1 + penalty_here
            if value0 > cap:
                value0 = cap
            if value1 > cap:
                value1 = cap
            unknown_cost[s, l, 0] = value0
            unknown_cost[s, l, 1] = value1
        ww_total[s] = total
    return ww_total, genotype_cost, unknown_cost


@njit(cache=True, parallel=True)
def _ww_bin_emissions(cost_ww, snps_per_bin, n_bins):
    """Binned log emission of the unknown-unknown state: -sum of cost_ww over each bin, left to
    right."""
    n, length = cost_ww.shape
    out = np.empty((n, n_bins), dtype=np.float64)
    for s in prange(n):
        for b in range(n_bins):
            start = b * snps_per_bin
            end = min(start + snps_per_bin, length)
            acc = 0.0
            for l in range(start, end):
                acc -= cost_ww[s, l]
            out[s, b] = acc
    return out


@njit(cache=True, parallel=True)
def _binned_contraction_inputs(log_probs, observed, penalty, snps_per_bin, n_bins):
    """Panel-independent inputs of the A step's binned emissions.

    With a_g = log_probs[s, l, g], a row-pair state (i, j) emits at a site
    a0 + (a1 - a0)(H_i + H_j) + (a0 - 2 a1 + a2) H_i H_j, and a row-unknown state (k, U) emits
    m01 + (m12 - m01) H_k - lambda (m01 = max(a0, a1), m12 = max(a1, a2); lambda only at
    observed cells). Per (sample, bin): C0b = sum a0, kW_Cb = sum m01 - lambda x observed sites;
    per (bin, site in bin, sample): diff1 = a1 - a0, w = a0 - 2 a1 + a2, kWdiff = m12 - m01
    (zero beyond the last site).
    """
    n, length, _ = log_probs.shape
    c0b = np.zeros((n, n_bins), dtype=np.float64)
    kw_cb = np.zeros((n, n_bins), dtype=np.float64)
    diff1 = np.zeros((n_bins, snps_per_bin, n), dtype=np.float64)
    w = np.zeros((n_bins, snps_per_bin, n), dtype=np.float64)
    kw_diff = np.zeros((n_bins, snps_per_bin, n), dtype=np.float64)
    for b in prange(n_bins):
        for s in range(n):
            c0 = 0.0
            cm01 = 0.0
            observed_count = 0
            for t in range(snps_per_bin):
                l = b * snps_per_bin + t
                if l < length:
                    a0 = log_probs[s, l, 0]
                    a1 = log_probs[s, l, 1]
                    a2 = log_probs[s, l, 2]
                    m01 = a0 if a0 > a1 else a1
                    m12 = a1 if a1 > a2 else a2
                    diff1[b, t, s] = a1 - a0
                    w[b, t, s] = a0 - 2.0 * a1 + a2
                    kw_diff[b, t, s] = m12 - m01
                    c0 += a0
                    cm01 += m01
                    if observed[s, l]:
                        observed_count += 1
            c0b[s, b] = c0
            kw_cb[s, b] = cm01 - observed_count * penalty
    return c0b, diff1, w, kw_cb, kw_diff


@njit(cache=True, nogil=True)
def _assign_pattern_slots(patterns, slot_of, n_used):
    """Give every new (bin, pattern) the next free slot of its bin; returns the new entries in
    first-use order and the largest number of slots any bin now uses."""
    n_rows, n_bins = patterns.shape
    new_bins = np.empty(n_rows * n_bins, dtype=np.int64)
    new_patterns = np.empty(n_rows * n_bins, dtype=np.int64)
    new_slots = np.empty(n_rows * n_bins, dtype=np.int64)
    count = 0
    for row in range(n_rows):
        for b in range(n_bins):
            pattern = patterns[row, b]
            if slot_of[b, pattern] < 0:
                slot = n_used[b]
                slot_of[b, pattern] = slot
                n_used[b] = slot + 1
                new_bins[count] = b
                new_patterns[count] = pattern
                new_slots[count] = slot
                count += 1
    needed = 0
    for b in range(n_bins):
        if n_used[b] > needed:
            needed = n_used[b]
    return new_bins[:count], new_patterns[:count], new_slots[:count], needed


@njit(cache=True, nogil=True)
def _lookup_pattern_slots(patterns, slot_of):
    n_rows, n_bins = patterns.shape
    slots = np.empty((n_rows, n_bins), dtype=np.int64)
    for row in range(n_rows):
        for b in range(n_bins):
            slots[row, b] = slot_of[b, patterns[row, b]]
    return slots


@njit(cache=True, nogil=True)
def _fill_pattern_rows(diff1, w, kw_diff, new_bins, new_patterns, new_slots,
                       diff1_table, w_table, kw_diff_table):
    """Contract new patterns: 0.0 plus the per-site terms of the pattern's set bits, added in
    ascending bit order (the complete table's recurrence, so the values are identical)."""
    n_samples = diff1.shape[2]
    snps_per_bin = diff1.shape[1]
    for index in range(new_bins.shape[0]):
        b = new_bins[index]
        pattern = new_patterns[index]
        slot = new_slots[index]
        diff1_row = diff1_table[b, slot]
        w_row = w_table[b, slot]
        kw_diff_row = kw_diff_table[b, slot]
        for s in range(n_samples):
            diff1_row[s] = 0.0
            w_row[s] = 0.0
            kw_diff_row[s] = 0.0
        for bit in range(snps_per_bin):
            if (pattern >> bit) & 1:
                diff1_bit = diff1[b, bit]
                w_bit = w[b, bit]
                kw_diff_bit = kw_diff[b, bit]
                for s in range(n_samples):
                    diff1_row[s] = diff1_row[s] + diff1_bit[s]
                    w_row[s] = w_row[s] + w_bit[s]
                    kw_diff_row[s] = kw_diff_row[s] + kw_diff_bit[s]


class PatternTables:
    """Lazily filled contractions of the within-bin allele patterns a block's panels use.

    ``slot_of[b, pattern]`` maps a pattern of bin b to its row (slot) in the three tables
    ``diff1_table``, ``w_table``, ``kw_diff_table`` (shape (n_bins, capacity, N)); call
    :meth:`slots` before reading the tables (adding patterns can replace them).
    """

    def __init__(self, diff1, w, kw_diff):
        self.n_bins, self.snps_per_bin, self.n_samples = diff1.shape
        self.diff1, self.w, self.kw_diff = diff1, w, kw_diff
        n_patterns = 1 << self.snps_per_bin
        self.slot_of = np.full((self.n_bins, n_patterns), -1, dtype=np.int64)
        self._n_used = np.zeros(self.n_bins, dtype=np.int64)
        capacity = min(_PATTERN_INITIAL_SLOTS, n_patterns)
        self._tables = [np.empty((self.n_bins, capacity, self.n_samples)) for _ in range(3)]

    @property
    def diff1_table(self):
        return self._tables[0]

    @property
    def w_table(self):
        return self._tables[1]

    @property
    def kw_diff_table(self):
        return self._tables[2]

    def slots(self, patterns: np.ndarray) -> np.ndarray:
        """Slots of within-bin patterns (last axis = bin), contracting first-seen patterns."""
        values = np.ascontiguousarray(patterns, dtype=np.int64)
        flat = values.reshape(-1, self.n_bins)
        new_bins, new_patterns, new_slots, needed = _assign_pattern_slots(
            flat, self.slot_of, self._n_used)
        if len(new_bins):
            capacity = self._tables[0].shape[1]
            if needed > capacity:
                while capacity < needed:
                    capacity *= 2
                for index, table in enumerate(self._tables):
                    grown = np.empty((self.n_bins, capacity, self.n_samples))
                    grown[:, :table.shape[1]] = table
                    self._tables[index] = grown
            _fill_pattern_rows(self.diff1, self.w, self.kw_diff, new_bins, new_patterns,
                               new_slots, *self._tables)
        return _lookup_pattern_slots(flat, self.slot_of).reshape(values.shape)


def search_bytes(samples: int, sites: int, founders: int, snps_per_bin: int = 10) -> int:
    """Planned peak bytes of one block's workspace for panels of up to ``founders`` rows: the
    evidence tables, pattern tables at their full 1,024 slots per bin, the entry-bounded caches
    (assignment records of three per-sample vectors; bit-packed panels and keys) and the
    fixed-point set (about 7,000 panels in a K = 25 block of AulStu chr3)."""
    n, length, k = int(samples), int(sites), int(founders)
    _, n_bins = bin_layout(length, int(snps_per_bin))
    tables = 8 * n * length * 14
    patterns = 3 * 8 * n_bins * 1024 * n
    packed = -(-k * length // 8) + 120
    caches = (TRANSITION_CACHE_ENTRIES * (32 * n + 400 + packed + 150)
              + FIT_CACHE_ENTRIES * (2 * packed + 300) + 16384 * packed)
    return tables + patterns + caches


def bin_layout(n_sites: int, snps_per_bin: int) -> tuple[int, int]:
    """(snps per bin, bins): ``snps_per_bin`` sites per bin when the block has more sites than
    that, else one bin per site."""
    if 1 < snps_per_bin < n_sites:
        return snps_per_bin, int(math.ceil(n_sites / snps_per_bin))
    return 1, n_sites


class FixedKWorkspace:
    """One block's fixed-K model tables and fit caches (see the module docstring).

    Fit caches are exact memos of deterministic functions of an *ordered* panel: the A(H)
    assignment record, the A(H), H(A) transition and the complete fit per (start, iteration
    limit); ``fixed_point_keys`` holds the ordered panels proved to be fixed points.
    Caches are bounded by entry count (least recently used entries leave first).
    """

    def __init__(self, likelihoods, observed, model):
        self.penalty = float(model.unknown_copy_penalty)
        self.switch_penalty = float(model.switch_penalty)
        self.likelihoods = likelihoods
        self.observed = np.ascontiguousarray(observed, dtype=np.bool_)
        self.uninformative = np.ascontiguousarray(~self.observed.any(axis=1))
        n, length, _ = likelihoods.shape
        self.log_probs, self.cost_ww = _observed_log_emissions(
            np.ascontiguousarray(likelihoods, dtype=np.float64), self.observed, self.penalty)
        self.ww_total_cost, genotype_cost, unknown_cost = _capped_update_costs(
            self.cost_ww, self.log_probs, self.observed, self.penalty)
        # The same capped costs in two layouts: (L, N, c) for the site-parallel H sweep and
        # site-contiguous (N, c, L) for the one-thread founder-outer sweep.
        self.genotype_cost_by_site = np.ascontiguousarray(genotype_cost.transpose(1, 0, 2))
        self.unknown_cost_by_site = np.ascontiguousarray(unknown_cost.transpose(1, 0, 2))
        self.genotype_cost_by_sample = np.ascontiguousarray(genotype_cost.transpose(0, 2, 1))
        self.unknown_cost_by_sample = np.ascontiguousarray(unknown_cost.transpose(0, 2, 1))
        self.snps_per_bin, self.n_bins = bin_layout(length, int(model.snps_per_bin))
        self.ww_bin_emissions = _ww_bin_emissions(self.cost_ww, self.snps_per_bin, self.n_bins)
        (self.c0b, diff1, w, self.kw_cb, kw_diff) = _binned_contraction_inputs(
            self.log_probs, self.observed, self.penalty, self.snps_per_bin, self.n_bins)
        self.patterns = PatternTables(diff1, w, kw_diff)
        self.fits: OrderedDict = OrderedDict()
        self.assignments: OrderedDict = OrderedDict()
        self.transitions: OrderedDict = OrderedDict()
        self.fixed_point_keys: set = set()

    @property
    def samples(self) -> int:
        return int(self.observed.shape[0])

    @property
    def sites(self) -> int:
        return int(self.observed.shape[1])

    # -- ordered-panel state keys (founder order matters: assignment ties are index ordered)
    @staticmethod
    def state_key(haplotypes) -> tuple:
        matrix = np.asarray(haplotypes)
        packed = np.packbits(np.ascontiguousarray(matrix, dtype=np.uint8).reshape(-1),
                             bitorder="little")
        return (int(matrix.shape[0]), int(matrix.shape[1]), packed.tobytes())

    @staticmethod
    def state_from_key(key) -> np.ndarray:
        k, length, packed = key
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=k * length,
                             bitorder="little")
        return np.ascontiguousarray(bits.reshape((k, length)), dtype=np.int64)

    # -- caches
    def cached_assignment(self, key):
        record = self.assignments.get(key)
        if record is not None:
            self.assignments.move_to_end(key)
        return record

    def remember_assignment(self, key, record):
        self.assignments[key] = record
        self.assignments.move_to_end(key)
        while len(self.assignments) > TRANSITION_CACHE_ENTRIES:
            self.assignments.popitem(last=False)

    def cached_transition(self, key):
        record = self.transitions.get(key)
        if record is not None:
            self.transitions.move_to_end(key)
        return record

    def remember_transition(self, key, assignment, next_key, changes):
        transition = (assignment, next_key, int(changes))
        self.remember_assignment(key, assignment)
        self.transitions[key] = transition
        self.transitions.move_to_end(key)
        while len(self.transitions) > TRANSITION_CACHE_ENTRIES:
            self.transitions.popitem(last=False)
        return transition

    @staticmethod
    def fit_key(haplotypes, max_iterations) -> tuple:
        return (int(max_iterations),) + FixedKWorkspace.state_key(haplotypes)

    def cached_fit(self, key):
        """A remembered fit tuple; its panel is stored bit-packed (K x L / 8 bytes, not the
        K x L x 8 of an int64 array) and returned as a fresh read-only int64 array."""
        stored = self.fits.get(key)
        if stored is None:
            return None
        self.fits.move_to_end(key)
        packed, rest = stored
        haplotypes = self.state_from_key(packed)
        haplotypes.setflags(write=False)
        return (haplotypes, *rest)

    def remember_fit(self, key, result, *, fixed_point):
        packed = self.state_key(result[0])
        self.fits[key] = (packed, tuple(result[1:]))
        self.fits.move_to_end(key)
        while len(self.fits) > FIT_CACHE_ENTRIES:
            self.fits.popitem(last=False)
        if fixed_point:
            self.fixed_point_keys.add(packed)

    @staticmethod
    def fixed_point_key(haplotypes) -> tuple:
        return FixedKWorkspace.state_key(haplotypes)

    def certifies_fixed_point(self, haplotypes) -> bool:
        return self.fixed_point_key(haplotypes) in self.fixed_point_keys
