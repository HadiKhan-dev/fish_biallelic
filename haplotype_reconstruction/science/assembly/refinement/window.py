"""Streamed log evidence and switch costs of a component's refinement.

Owns mathematical window kernels and dimension constants; stage capabilities own evidence buffers.
Method: docs/methods/founder_refinement.md#workspace (RC02–RC04).

Holding all seven robust log-evidence codes (float64 S x L x 7), a reversed copy and seven
float32 switch-cost arrays costs about 140 bytes per sample-site. Those values are pure functions
of the component's float32 evidence, site masks and genetic map, so windows compute them where
they are read:

- a *window* holds, for a stretch of sites and a range of samples, the uncentred robust log evidence
  of pair codes 0-2 (float64), of codes 3-6 at its partial sites (sites where a prepared row of the
  block is uncalled: the only sites where any panel's pair codes reach 3-6), and the switch costs of
  one founder count K and one orientation (float32);
- every kernel reads a sample-site through `locate` / `site_logs` / `site_cost`: from the window when it
  holds the site, otherwise computed on the spot by the same arithmetic (`site_values`, `cost_value`).
  Both routes use the same operations in the same order, so windows change time and
  memory only;
- kernels `cover` the stretch they are about to read: a window that does not hold it is refilled
  (the overlap with what it held is moved, not recomputed).

A component whose tables fit RESIDENT_BYTES keeps one resident window over all its sites and samples,
shared by every search of the component (the switch costs per K and orientation cached, at most
COST_ARRAYS), with the same read arithmetic. A larger component gives each consumer (one search, or one
count computation) its own window (Windows.view) of W = max(WINDOW_MIN_SITES, WINDOW_BYTES / (28 N)) sites,
which takes 28 N W + 8 W + 32 N max(1, P_W) bytes (Windows.window_bytes; P_W the most partial sites in any
W-site stretch): WINDOW_BYTES for codes 0-2 and costs, plus the partial sites' codes 3-6, up to as much again
when every site is partial (1,097 MiB at N = 2,503). The memory of the refinement evidence is bounded by the
windows, not by the chromosome. Window sizes are fixed constants, never the memory that happens to be free;
they set work and memory only.

Site order. The buffers are always in forward site order. A reversed search indexes site j of the
reversed orientation, forward site L - 1 - j (`reverse` in the window tuple), so both orientations read
one copy of the log evidence; the reversed switch costs differ (the cost entering reversed site L - 1 - j
is the forward cost of the sample's next informative site after j) and are filled per orientation.

Window tuple (the Numba argument of every kernel):
    (ev, kept, partial, hazards, logs, plogs, pmap, costs, meta, reverse)
    ev       float32 (S, L, 3): the component's evidence (normalised genotype likelihoods), forward order
    kept     bool (L,): the objective's sites (kept, with a called prepared row)
    partial  bool (L,): sites where a prepared row is uncalled (pair codes 3-6 can be read there)
    hazards  float64 (L,): physical-edge reset hazards (component_workspace.map_hazards)
    logs     float64 (R, C, 3): codes 0-2 of rows (samples) meta[FIRST] .., columns (sites) meta[BASE] ..
    plogs    float64 (R, P, 4): codes 3-6 of the window's partial sites (slot pmap[column])
    pmap     int64 (C,): partial slot of a column, or -1
    costs    float32 (R, C): switch cost entering each site (+inf at a sample's first informative site and
             at skipped sites) of K = meta[FOUNDERS] in orientation meta[COST_ORIENT]
    meta     int64 (META,): see the field indices below
    reverse  bool: the orientation the kernel indexes in
"""

from __future__ import annotations

import math

import numpy as np
from numba import prange

from haplotype_reconstruction.core.kernels import kernel

from .emissions import normalised_likelihoods, predictive_likelihood, robust_log_likelihood
from .overloads import overloads_for

# Field indices of a window's meta array.
BASE, LENGTH, FIRST, LAST, FOUNDERS, COST_ORIENT, SITES, RESIDENT = range(8)
META = 8

# Per sample-site bytes of a window: codes 0-2 (float64) and one switch cost (float32); codes 3-6 add 32
# bytes per partial sample-site.
WINDOW_SITE_BYTES = 3 * 8 + 4
PARTIAL_SITE_BYTES = 4 * 8
# Sizes a consumer's window (one search, or one count computation): W = max(WINDOW_MIN_SITES, WINDOW_BYTES // (28 N))
# sites of codes 0-2 and costs (7,660 sites, 38 prepared blocks, at N = 2,503). The window also holds 8 W bytes of
# column map and 32 N bytes per partial site it can meet, so it takes WINDOW_BYTES plus 0 (no partial site) to
# 32 N W (every site partial) bytes: a sizing budget, not an upper bound. An acceptance pass covers the
# acceptance.JUNCTION_REACH = 24 blocks ahead of its junction, so its window moves about once every 14 blocks and
# keeps its overlap; whole-component passes fill each stretch once whatever the size.
WINDOW_BYTES = 1 << 29
WINDOW_MIN_SITES = 4096
# A component whose log evidence and switch costs fit this many bytes stays resident: one window over every site
# and sample, shared by all its consumers (its switch costs cached per K and orientation, at most COST_ARRAYS),
# so it is filled once. Above it every consumer streams its own window.
RESIDENT_BYTES = 4 << 30
# At most this many (K, orientation) switch-cost arrays of a resident component are cached (K and K +- 1 of the
# count pass in both orientations).
COST_ARRAYS = 6
# A fill remembers, per sample, the log evidence of the last likelihood triple seen in each of MEMO_SLOTS hash slots
# (a power of two): low-coverage likelihoods repeat (a simulated chromosome has ~200 distinct triples, real TroMau
# ~30,000 over 10-15M informative sample-sites), and a repeated triple's seven values are copied instead of
# recomputed. The memo holds values of the same arithmetic for the same input, so it changes time only.
MEMO_SLOTS = 2048


# ------------------------------------------------------------------ one sample-site from the evidence
@kernel(overloads=overloads_for(__name__, "informative_at"), inline="always", nogil=True)
def informative_at(ev, kept, s, f):
    """Forward site f carries evidence for sample s: an objective site where its likelihoods are informative."""
    if not kept[f]:
        return False
    informative, _, _, _ = normalised_likelihoods(ev[s, f, 0], ev[s, f, 1], ev[s, f, 2])
    return informative


@kernel(overloads=overloads_for(__name__, "site_values"), inline="always", nogil=True)
def site_values(ev, kept, partial, s, f, e):
    """e[0..6] = the uncentred robust log evidence of the seven pair codes of sample s at forward site f
    (codes 3-6 only at partial sites); returns False (e untouched) where the site is skipped."""
    if not kept[f]:
        return False
    informative, p0, p1, p2 = normalised_likelihoods(ev[s, f, 0], ev[s, f, 1], ev[s, f, 2])
    if not informative:
        return False
    e[0] = robust_log_likelihood(predictive_likelihood(p0, p1, p2, 0))
    e[1] = robust_log_likelihood(predictive_likelihood(p0, p1, p2, 1))
    e[2] = robust_log_likelihood(predictive_likelihood(p0, p1, p2, 2))
    if partial[f]:
        e[3] = robust_log_likelihood(predictive_likelihood(p0, p1, p2, 3))
        e[4] = robust_log_likelihood(predictive_likelihood(p0, p1, p2, 4))
        e[5] = robust_log_likelihood(predictive_likelihood(p0, p1, p2, 5))
        e[6] = robust_log_likelihood(predictive_likelihood(p0, p1, p2, 6))
    return True


@kernel(overloads=overloads_for(__name__, "switch_cost"), inline="always", nogil=True)
def switch_cost(rho, founders):
    """Per changed homologue log(1 - rho (1 - 1/K)) - log(rho / K) >= 0; +inf without a preceding
    informative site (rho = 0), 0 for an exact uniform reset."""
    if rho == 0.0:
        return np.inf
    if rho == 1.0:
        return 0.0
    return math.log1p(-rho * (1.0 - 1.0 / founders)) - math.log(rho) + math.log(founders)


@kernel(overloads=overloads_for(__name__, "gap_cost"), inline="always", nogil=True)
def gap_cost(mass, founders):
    """float32 switch cost of a gap of reset hazard `mass` (rho = 1 - exp(-mass))."""
    return np.float32(switch_cost(-math.expm1(-mass), founders))


@kernel(overloads=overloads_for(__name__, "cost_value"), cache=True, nogil=True)
def cost_value(ev, kept, hazards, s, f, founders, reverse):
    """The switch cost of sample s at forward site f, computed alone: forward orientation, the cost entering f
    from the sample's previous informative site; reversed orientation, the cost entering its next informative
    site after f. +inf at a skipped site and without such a site. The hazards are summed from the earlier site
    onwards, using the same summation order as the window fill."""
    sites = len(kept)
    if not informative_at(ev, kept, s, f):
        return np.float32(np.inf)
    K = float(founders)
    if reverse:
        mass = 0.0
        for t in range(f + 1, sites):
            mass += hazards[t]
            if informative_at(ev, kept, s, t):
                return gap_cost(mass, K)
        return np.float32(np.inf)
    i = f - 1
    while i >= 0 and not informative_at(ev, kept, s, i):
        i -= 1
    if i < 0:
        return np.float32(np.inf)
    mass = 0.0
    for t in range(i + 1, f + 1):
        mass += hazards[t]
    return gap_cost(mass, K)


# ------------------------------------------------------------------ kernel accessors
@kernel(overloads=overloads_for(__name__, "locate"), inline="always", nogil=True)
def locate(win, s, j):
    """(forward site, column, row) of sample s at site j of the window's orientation; column or row -1 when
    the window does not hold it."""
    meta = win[8]
    f = meta[SITES] - 1 - j if win[9] else j
    k = f - meta[BASE]
    r = s - meta[FIRST]
    if k < 0 or k >= meta[LENGTH] or r < 0 or s >= meta[LAST]:
        return f, -1, -1
    return f, k, r


@kernel(overloads=overloads_for(__name__, "site_logs"), inline="always", nogil=True)
def site_logs(win, s, f, k, r, e):
    """e[0..6] = the uncentred log evidence of the pair codes of sample s at forward site f (column k, row r
    from locate); False where the site is skipped. Codes 3-6 are set only at partial sites."""
    if k >= 0:
        logs = win[4]
        e0 = logs[r, k, 0]
        e1 = logs[r, k, 1]
        e2 = logs[r, k, 2]
        if e0 == 0.0 and e1 == 0.0 and e2 == 0.0:
            return False
        e[0] = e0
        e[1] = e1
        e[2] = e2
        slot = win[6][k]
        if slot >= 0:
            plogs = win[5]
            e[3] = plogs[r, slot, 0]
            e[4] = plogs[r, slot, 1]
            e[5] = plogs[r, slot, 2]
            e[6] = plogs[r, slot, 3]
        return True
    return site_values(win[0], win[1], win[2], s, f, e)


@kernel(overloads=overloads_for(__name__, "site_cost"), inline="always", nogil=True)
def site_cost(win, s, f, k, r):
    """float32 switch cost of sample s at forward site f in the window's orientation (column k, row r)."""
    meta = win[8]
    if k >= 0 and meta[COST_ORIENT] == (1 if win[9] else 0):
        return win[7][r, k]
    return cost_value(win[0], win[1], win[3], s, f, meta[FOUNDERS], win[9])


@kernel(overloads=overloads_for(__name__, "skipped"), inline="always", nogil=True)
def skipped(win, s, j):
    """Site j (window orientation) carries no evidence for sample s (read from the evidence itself)."""
    meta = win[8]
    f = meta[SITES] - 1 - j if win[9] else j
    return not informative_at(win[0], win[1], s, f)


@kernel(overloads=overloads_for(__name__, "capacity"), inline="always", nogil=True)
def capacity(win):
    """Sites a window can hold (the whole component for a resident window)."""
    return win[4].shape[1]


@kernel(overloads=overloads_for(__name__, "samples_of"), inline="always", nogil=True)
def samples_of(win):
    return win[0].shape[0]


@kernel(overloads=overloads_for(__name__, "sites_of"), inline="always", nogil=True)
def sites_of(win):
    return win[8][SITES]


# ------------------------------------------------------------------ filling
@kernel(overloads=overloads_for(__name__, "_memo_slot"), inline="always", nogil=True)
def _memo_slot(x0, x1, x2):
    """Hash slot of a likelihood triple (any deterministic function: a slot is used only on an exact match)."""
    h = (int(x0 * 1048573.0) * 31 + int(x1 * 1048583.0)) * 31 + int(x2 * 1048589.0)
    return h & (MEMO_SLOTS - 1)


@kernel(overloads=overloads_for(__name__, "_fill_logs"), cache=True, parallel=True, nogil=True)
def _fill_logs(ev, kept, partial, logs, plogs, pmap, base, s0, s1, c0, c1):
    """Codes 0-2 of columns [c0, c1) (forward sites base + c) and codes 3-6 of their partial sites (slots
    already in pmap) for samples [s0, s1) into rows 0 .. s1 - s0. Skipped sample-sites hold zeros. A sample's
    repeated likelihood triples take their values from its fill's memo (MEMO_SLOTS): every value is
    site_values' for that exact triple."""
    for s in prange(s0, s1):
        r = s - s0
        keys = np.empty((MEMO_SLOTS, 3), np.float32)
        used = np.zeros(MEMO_SLOTS, np.bool_)
        found = np.zeros(MEMO_SLOTS, np.bool_)
        values = np.empty((MEMO_SLOTS, 7))
        v = np.empty(7)
        for k in range(c0, c1):
            f = base + k
            if not kept[f]:
                logs[r, k, 0] = 0.0
                logs[r, k, 1] = 0.0
                logs[r, k, 2] = 0.0
                continue
            x0, x1, x2 = ev[s, f, 0], ev[s, f, 1], ev[s, f, 2]
            if not (x0 == x0 and x1 == x1 and x2 == x2 and x0 < 2.0 and x1 < 2.0 and x2 < 2.0):
                # outside the memo's domain (not a normalised likelihood): computed alone
                informative = site_values(ev, kept, partial, s, f, v)
                for c in range(7):
                    values_row = v[c] if informative else 0.0
                    if c < 3:
                        logs[r, k, c] = values_row
                    elif pmap[k] >= 0:
                        plogs[r, pmap[k], c - 3] = values_row
                continue
            h = _memo_slot(x0, x1, x2)
            if not (used[h] and keys[h, 0] == x0 and keys[h, 1] == x1 and keys[h, 2] == x2):
                informative, p0, p1, p2 = normalised_likelihoods(x0, x1, x2)
                if informative:
                    for c in range(7):
                        values[h, c] = robust_log_likelihood(predictive_likelihood(p0, p1, p2, c))
                keys[h, 0] = x0
                keys[h, 1] = x1
                keys[h, 2] = x2
                used[h] = True
                found[h] = informative
            if found[h]:
                logs[r, k, 0] = values[h, 0]
                logs[r, k, 1] = values[h, 1]
                logs[r, k, 2] = values[h, 2]
                slot = pmap[k]
                if slot >= 0:
                    plogs[r, slot, 0] = values[h, 3]
                    plogs[r, slot, 1] = values[h, 4]
                    plogs[r, slot, 2] = values[h, 5]
                    plogs[r, slot, 3] = values[h, 6]
            else:
                logs[r, k, 0] = 0.0
                logs[r, k, 1] = 0.0
                logs[r, k, 2] = 0.0


@kernel(overloads=overloads_for(__name__, "_fill_partial"), cache=True, parallel=True, nogil=True)
def _fill_partial(ev, kept, partial, plogs, pmap, base, s0, s1, c0, c1):
    """Codes 3-6 of the partial sites among columns [c0, c1) for samples [s0, s1)."""
    for s in prange(s0, s1):
        r = s - s0
        v = np.empty(7)
        for k in range(c0, c1):
            slot = pmap[k]
            if slot < 0:
                continue
            if site_values(ev, kept, partial, s, base + k, v):
                plogs[r, slot, 0] = v[3]
                plogs[r, slot, 1] = v[4]
                plogs[r, slot, 2] = v[5]
                plogs[r, slot, 3] = v[6]


@kernel(overloads=overloads_for(__name__, "_fill_costs"), cache=True, parallel=True, nogil=True)
def _fill_costs(ev, kept, hazards, costs, base, s0, s1, c0, c1, founders, reverse):
    """Switch costs of columns [c0, c1) (forward sites base + c) for samples [s0, s1) in either orientation,
    forward, the cost entering each informative site from the sample's previous
    informative site (the hazards summed onwards from it); reversed, the forward cost of its next informative
    site. A gap of one edge, the common case, reads the per-site cost table."""
    sites = len(kept)
    f0, f1 = base + c0, base + c1
    K = float(founders)
    # The cost of a one-edge gap into site t (mass = 0.0 + hazards[t], bitwise hazards[t]).
    lo = f0 if not reverse else f0 + 1
    hi = min(sites, f1 + 1)
    unit = np.empty(max(0, hi - lo), np.float32)
    for t in range(lo, hi):
        unit[t - lo] = gap_cost(0.0 + hazards[t], K)
    for s in prange(s0, s1):
        r = s - s0
        if not reverse:
            # the sample's state entering f0: its previous informative site i (if any) and the hazards since
            i = f0 - 1
            while i >= 0 and not informative_at(ev, kept, s, i):
                i -= 1
            seen = i >= 0
            mass = 0.0
            if seen:
                for t in range(i + 1, f0):
                    mass += hazards[t]
            last = i
            for f in range(f0, f1):
                if seen:
                    mass += hazards[f]
                if not informative_at(ev, kept, s, f):
                    costs[r, f - base] = np.inf
                    continue
                if not seen:
                    costs[r, f - base] = np.inf
                elif last == f - 1:
                    costs[r, f - base] = unit[f - lo]
                else:
                    costs[r, f - base] = gap_cost(mass, K)
                seen = True
                mass = 0.0
                last = f
        else:
            # the sample's next informative site at or after f1
            nxt = f1
            while nxt < sites and not informative_at(ev, kept, s, nxt):
                nxt += 1
            for f in range(f1 - 1, f0 - 1, -1):
                if not informative_at(ev, kept, s, f):
                    costs[r, f - base] = np.inf
                    continue
                if nxt >= sites:
                    costs[r, f - base] = np.inf
                elif nxt == f + 1:
                    costs[r, f - base] = unit[nxt - lo]
                else:
                    mass = 0.0
                    for t in range(f + 1, nxt + 1):
                        mass += hazards[t]
                    costs[r, f - base] = gap_cost(mass, K)
                nxt = f


@kernel(overloads=overloads_for(__name__, "_shift"), cache=True, parallel=True, nogil=True)
def _shift(logs, costs, rows, shift, c0, c1, move_costs):
    """Move columns [c0, c1) of the first `rows` rows to c - shift (shift > 0: towards column 0); the costs too
    when move_costs."""
    for r in prange(rows):
        if shift > 0:
            for k in range(c0, c1):
                logs[r, k - shift, 0] = logs[r, k, 0]
                logs[r, k - shift, 1] = logs[r, k, 1]
                logs[r, k - shift, 2] = logs[r, k, 2]
                if move_costs:
                    costs[r, k - shift] = costs[r, k]
        else:
            for k in range(c1 - 1, c0 - 1, -1):
                logs[r, k - shift, 0] = logs[r, k, 0]
                logs[r, k - shift, 1] = logs[r, k, 1]
                logs[r, k - shift, 2] = logs[r, k, 2]
                if move_costs:
                    costs[r, k - shift] = costs[r, k]


@kernel(overloads=overloads_for(__name__, "refill"), cache=True, nogil=True)
def refill(win, f0, f1, s0, s1):
    """Make the window hold forward sites [f0, f1) (f1 - f0 <= capacity) of samples [s0, s1) with the switch
    costs of its orientation. What it already holds for the same samples is moved, not recomputed (the log
    evidence is the same in both orientations; the costs only when they are of this orientation)."""
    ev, kept, partial, hazards, logs, plogs, pmap, costs, meta, reverse = win
    orient = 1 if reverse else 0
    founders = meta[FOUNDERS]
    old0, old1 = meta[BASE], meta[BASE] + meta[LENGTH]
    keep0, keep1 = max(f0, old0), min(f1, old1)
    reuse = meta[LENGTH] > 0 and meta[FIRST] == s0 and meta[LAST] == s1 and keep0 < keep1
    reuse_costs = reuse and meta[COST_ORIENT] == orient
    if reuse and f0 != old0:
        _shift(logs, costs, s1 - s0, f0 - old0, keep0 - old0, keep1 - old0, reuse_costs)
    # partial slots of the window's columns (codes 3-6 are refilled for every partial site of the window: rare)
    slot = 0
    for k in range(f1 - f0):
        if partial[f0 + k]:
            pmap[k] = slot
            slot += 1
        else:
            pmap[k] = -1
    meta[BASE] = f0
    meta[LENGTH] = f1 - f0
    meta[FIRST] = s0
    meta[LAST] = s1
    meta[COST_ORIENT] = orient
    if reuse:
        if keep0 > f0:
            _fill_logs(ev, kept, partial, logs, plogs, pmap, f0, s0, s1, 0, keep0 - f0)
        if f1 > keep1:
            _fill_logs(ev, kept, partial, logs, plogs, pmap, f0, s0, s1, keep1 - f0, f1 - f0)
        if slot:
            _fill_partial(ev, kept, partial, plogs, pmap, f0, s0, s1, keep0 - f0, keep1 - f0)
    else:
        _fill_logs(ev, kept, partial, logs, plogs, pmap, f0, s0, s1, 0, f1 - f0)
    if reuse_costs:
        if keep0 > f0:
            _fill_costs(ev, kept, hazards, costs, f0, s0, s1, 0, keep0 - f0, founders, reverse)
        if f1 > keep1:
            _fill_costs(ev, kept, hazards, costs, f0, s0, s1, keep1 - f0, f1 - f0, founders, reverse)
    else:
        _fill_costs(ev, kept, hazards, costs, f0, s0, s1, 0, f1 - f0, founders, reverse)


@kernel(overloads=overloads_for(__name__, "cover"), cache=True, nogil=True)
def cover(win, lo, hi, s0, s1, backward):
    """Before reading sites [lo, hi) (window orientation) of samples [s0, s1): refill the window unless it holds
    them with its orientation's costs. A stretch longer than the window is covered from its start (backward:
    from its end); the rest is read through the exact fallback. backward: the reads proceed towards lower
    sites (the window is placed to end at hi). A resident window is never refilled."""
    meta = win[8]
    if meta[RESIDENT] or hi <= lo:
        return
    sites = meta[SITES]
    reverse = win[9]
    f_lo, f_hi = (sites - hi, sites - lo) if reverse else (lo, hi)
    if (
        meta[LENGTH] > 0
        and f_lo >= meta[BASE]
        and f_hi <= meta[BASE] + meta[LENGTH]
        and s0 >= meta[FIRST]
        and s1 <= meta[LAST]
        and meta[COST_ORIENT] == (1 if reverse else 0)
    ):
        return
    size = min(win[4].shape[1], sites)
    # placement in the window's orientation, then in forward sites
    toward_low = backward != reverse
    if toward_low:
        f1 = min(sites, f_hi)
        f0 = max(0, f1 - size)
        f1 = min(sites, f0 + size)
    else:
        f0 = max(0, f_lo)
        f1 = min(sites, f0 + size)
        f0 = max(0, f1 - size)
    refill(win, f0, f1, s0, s1)


@kernel(overloads=overloads_for(__name__, "stretch"), inline="always", nogil=True)
def stretch(offsets, b0, b1, limit):
    """The end block of the next stretch of whole blocks from b0 (towards b1 > b0) holding at most `limit`
    sites (at least one block)."""
    end = b0 + 1
    while end < b1 and offsets[end + 1] - offsets[b0] <= limit:
        end += 1
    return end


@kernel(overloads=overloads_for(__name__, "stretch_back"), inline="always", nogil=True)
def stretch_back(offsets, b0, b1, limit):
    """The start block of the next stretch of whole blocks ending at b1 (towards b0 < b1) holding at most
    `limit` sites (at least one block)."""
    start = b1 - 1
    while start > b0 and offsets[b1] - offsets[start - 1] <= limit:
        start -= 1
    return start


# ------------------------------------------------------------------ whole-component scans of the evidence


@kernel(overloads=overloads_for(__name__, "boundary_costs"), cache=True, parallel=True, nogil=True)
def boundary_costs(ev, kept, hazards, offsets, founders):
    """float64 (B + 1, S): the forward switch cost (K = founders) entering each sample's first informative site
    at or after every block boundary (+inf when there is none, or it is the sample's first).
    Only the costs of these boundary sites are computed."""
    blocks = len(offsets) - 1
    samples = ev.shape[0]
    K = float(founders)
    out = np.empty((blocks + 1, samples))
    for s in prange(samples):
        seen = False
        mass = 0.0
        c = 0
        for f in range(offsets[blocks]):
            if seen:
                mass += hazards[f]
            if not informative_at(ev, kept, s, f):
                continue
            if c < blocks + 1 and offsets[c] <= f:
                value = np.float32(gap_cost(mass, K)) if seen else np.float32(np.inf)
                while c < blocks + 1 and offsets[c] <= f:
                    out[c, s] = value
                    c += 1
            seen = True
            mass = 0.0
        while c < blocks + 1:
            out[c, s] = np.inf
            c += 1
    return out


# ------------------------------------------------------------------ windows of a workspace
def window_sites(samples: int) -> int:
    """Sites of a consumer's streaming window for `samples` samples."""
    return max(WINDOW_MIN_SITES, WINDOW_BYTES // max(1, WINDOW_SITE_BYTES * int(samples)))


def resident_bytes(samples: int, sites: int, partials: int) -> int:
    """Bytes of a resident window (codes 0-2, the partial sites' codes 3-6, one cost array) and its cost cache."""
    return int(samples) * (int(sites) * (24 + 4 * COST_ARRAYS) + PARTIAL_SITE_BYTES * int(partials))


__all__ = [
    "informative_at",
    "site_values",
    "switch_cost",
    "gap_cost",
    "cost_value",
    "locate",
    "site_logs",
    "site_cost",
    "skipped",
    "capacity",
    "samples_of",
    "sites_of",
    "refill",
    "cover",
    "stretch",
    "stretch_back",
    "boundary_costs",
    "window_sites",
    "resident_bytes",
]


@kernel(overloads=overloads_for(__name__, "informative_samples"), cache=True, parallel=True, nogil=True)
def informative_samples(ev, kept):
    """bool (S,): samples with evidence at one of the component's objective sites. Every other sample's F_s is
    exactly 0 for every panel (no site step ever runs). A sample's scan stops at its first such site."""
    samples, sites = ev.shape[0], ev.shape[1]
    result = np.zeros(samples, np.bool_)
    for s in prange(samples):
        for f in range(sites):
            if informative_at(ev, kept, s, f):
                result[s] = True
                break
    return result


@kernel(overloads=overloads_for(__name__, "map_normalisers"), cache=True, parallel=True, nogil=True)
def map_normalisers(ev, kept, hazards, founders):
    """float64 (S,) panel-independent terms of every sample's max-path score: -2 log K at its first informative
    site (the uniform start), 2 log(1 - rho (1 - 1/K)) at every later one (0 without evidence)."""
    samples, sites = ev.shape[0], ev.shape[1]
    out = np.zeros(samples)
    for s in prange(samples):
        seen = False
        mass = 0.0
        total = 0.0
        for j in range(sites):
            if seen:
                mass += hazards[j]
            if not informative_at(ev, kept, s, j):
                continue
            if seen:
                total += 2.0 * math.log1p(-(-math.expm1(-mass)) * (1.0 - 1.0 / founders))  # noqa: B002 - stay term
            else:
                total = -2.0 * math.log(founders)
                seen = True
            mass = 0.0
        out[s] = total
    return out
