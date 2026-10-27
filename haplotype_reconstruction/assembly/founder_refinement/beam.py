"""The polish's polynomial window-beam proposer (from the 2.8 design prototype) on the 2.6 engine; proposals only.

The window problems maximise a binned unordered (Potts) surrogate H whose switch costs are
per-sample and per-bin map-derived (the Q-state reset law of rho composed between the
midpoints of a sample's informative bins), with the surrogate's own flank messages
(potts_bin_messages). For every (founder, window) a bounded beam (BEAM_WIDTH partial paths per
unit, box bounds, pairwise dominance that never lets an incumbent prefix erase a distinct one)
finds the best path that differs from the incumbent; it is offered only if its H exceeds the
incumbent's (2.6's offer rule). A window whose box bound cannot
exceed the incumbent's H by the offer quantum offers nothing and skips the beam. Exact acceptance (acceptance.py) decides every
move. The beam is an approximate H search by design; its cut counts are reported as
beam_cut / beam_binding in the proposal statistics. It reads the workspace's binned emissions (the window
programmes' surrogate: floor-2 robust emissions over kept sites with a called local row, predictive classes for
uncalled alleles). Used only by the polish (search.SweepEngine.beam_escape and long_repair).
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange
from numba.typed import List

from ... import execution
from .. import tie_rules
from ...maths import ties
from .objective import unordered_pairs, row_max
from .proposals import window_templates
from .workspace import map_hazards, switch_cost

BEAM_WIDTH = 64


@njit(inline='always')
def pair(a, b):
    return max(a, b) * (max(a, b) + 1) // 2 + min(a, b)


@njit(inline='always')
def bin_cost(rho, s, j, states):
    """Q-state switch cost of sample s entering bin j (+inf without a preceding informative bin)."""
    return switch_cost(rho[s, j], states)


@njit(cache=True, parallel=True, nogil=True)
def compose_bin_resets(flags, hazards):
    """float64 (S, bins) reset probability entering each informative bin of every sample (0 at a
    sample's first informative bin and at uninformative bins): rho = 1 - exp(-sum of the bin-midpoint
    hazards since the previous informative bin)."""
    samples, bins = flags.shape
    out = np.zeros((samples, bins))
    for s in prange(samples):
        seen = False
        mass = 0.0
        for j in range(bins):
            if seen:
                mass += hazards[j]
            if not flags[s, j]:
                continue
            if seen:
                out[s, j] = -math.expm1(-mass)
            seen = True
            mass = 0.0
    return out


@njit(cache=True, parallel=True, nogil=True)
def rank_table(emissions, bin_offsets, max_pairs):
    """(S, bins, 3 + max_pairs) surrogate table: column 0 the informative-bin flag, columns 3 + pair
    the binned pair emissions (workspace.binned_emissions laid out per bin)."""
    samples = emissions[0].shape[0]
    out = np.zeros((samples, bin_offsets[-1], 3 + max_pairs))
    for s in prange(samples):
        for b in range(len(emissions)):
            e = emissions[b]
            for t in range(e.shape[2]):
                j = bin_offsets[b] + t
                informative = False
                for z in range(e.shape[1]):
                    v = e[s, z, t]
                    out[s, j, 3 + z] = v
                    if v != 0.0:
                        informative = True
                out[s, j, 0] = 1.0 if informative else 0.0
    return out


@njit(cache=True, parallel=True, nogil=True)
def potts_bin_messages(haps, logs, offsets, rho):
    """The surrogate's own normalised flank messages at every block boundary: forward
    switch-then-emission, backward emission-then-switch, per informative bin, Q-state map costs."""
    first, second = unordered_pairs(len(haps))
    qn = len(first)
    blocks = len(offsets) - 1
    n = len(logs)
    f = np.zeros((blocks + 1, n, qn))
    g = np.zeros_like(f)
    for s in prange(n):
        row = np.zeros(qn)
        for b in range(blocks):
            for j in range(offsets[b], offsets[b + 1]):
                if logs[s, j, 0] == 0.0:
                    continue
                floor = row_max(row) - bin_cost(rho, s, j, qn)
                for q in range(qn):
                    row[q] = max(row[q], floor) + logs[s, j, 3 + pair(haps[first[q], j], haps[second[q], j])]
            row -= row_max(row)
            f[b + 1, s] = row
        row = np.zeros(qn)
        for b in range(blocks - 1, -1, -1):
            for j in range(offsets[b + 1] - 1, offsets[b] - 1, -1):
                if logs[s, j, 0] == 0.0:
                    continue
                for q in range(qn):
                    row[q] += logs[s, j, 3 + pair(haps[first[q], j], haps[second[q], j])]
                floor = row_max(row) - bin_cost(rho, s, j, qn)
                for q in range(qn):
                    row[q] = max(row[q], floor)
            row -= row_max(row)
            g[b, s] = row
    return f, g


@njit(cache=True, nogil=True)
def unit_tables(logs, cols, first, second, focal, tail):
    """Background segment maxima, shared by every child and parent at this unit."""
    n, b, _ = logs.shape
    active = np.where((first == focal) | (second == focal))[0]
    bg = np.where((first != focal) & (second != focal))[0]
    prefix = np.zeros((n, b + 1, len(bg)))
    seg = np.full((n, b, b), -np.inf)
    endseg = np.full((n, b), -np.inf)
    for s in range(n):
        for z in range(len(bg)):
            state = bg[z]
            aa = first[state]
            bb = second[state]
            for j in range(b):
                e = logs[s, j, 3 + pair(cols[aa, j], cols[bb, j])] if logs[s, j, 0] != 0 else 0.0
                prefix[s, j + 1, z] = prefix[s, j, z] + e
            for t in range(b):
                endseg[s, t] = max(endseg[s, t], prefix[s, b, z] - prefix[s, t, z] + tail[s, state])
                for j in range(t, b):
                    seg[s, t, j] = max(seg[s, t, j], prefix[s, j + 1, z] - prefix[s, t, z])
    return active, bg, prefix, seg, endseg


@njit(cache=True, nogil=True)
def rank_children(messages, logs, rho, states, cols, candidates, first, second, focal, tail,
                  active, bg, prefix, seg, endseg):
    """Child upper bounds, O(B N b Q + B R N (b K + b^2)); no Q loop inside child ranking."""
    parents, n, q = messages.shape
    b = logs.shape[1]
    rmax = len(candidates)
    k = len(active)
    totals = np.zeros((parents, rmax))
    for p in range(parents):
        for s in range(n):
            bgbase = np.full(b, -np.inf)
            endbase = -np.inf
            bgstart = -np.inf
            for zi in range(len(bg)):
                z = bg[zi]
                bgstart = max(bgstart, messages[p, s, z])
                endbase = max(endbase, messages[p, s, z] + prefix[s, b, zi] + tail[s, z])
                for j in range(b):
                    bgbase[j] = max(bgbase[j], messages[p, s, z] + prefix[s, j + 1, zi])
            for r in range(rmax):
                focalvals = messages[p, s, active].copy()
                floors = np.full(b, -np.inf)
                bgmax = bgstart
                for j in range(b):
                    if logs[s, j, 0] != 0:
                        floor = max(bgmax, np.max(focalvals)) - bin_cost(rho, s, j, states)
                        floors[j] = floor
                        for zi in range(k):
                            z = active[zi]
                            aa = first[z]
                            bb = second[z]
                            ra = candidates[r, j] if aa == focal else cols[aa, j]
                            rb = candidates[r, j] if bb == focal else cols[bb, j]
                            focalvals[zi] = max(focalvals[zi], floor) + logs[s, j, 3 + pair(ra, rb)]
                    bgmax = bgbase[j]
                    for t in range(j + 1):
                        bgmax = max(bgmax, floors[t] + seg[s, t, j])
                value = endbase
                for t in range(b):
                    value = max(value, floors[t] + endseg[s, t])
                for zi in range(k):
                    value = max(value, focalvals[zi] + tail[s, active[zi]])
                totals[p, r] += value
    return totals


@njit(cache=True, nogil=True)
def extend(message, logs, rho, states, cols, candidate, first, second, focal):
    """Full-state recursion of one unit, used only for the beam's survivors."""
    out = message.copy()
    for s in range(len(logs)):
        for j in range(logs.shape[1]):
            if logs[s, j, 0] == 0:
                continue
            floor = np.max(out[s]) - bin_cost(rho, s, j, states)
            for z in range(len(first)):
                aa = first[z]
                bb = second[z]
                ra = candidate[j] if aa == focal else cols[aa, j]
                rb = candidate[j] if bb == focal else cols[bb, j]
                out[s, z] = max(out[s, z], floor) + logs[s, j, 3 + pair(ra, rb)]
    return out


@njit(cache=True, nogil=True)
def box_unit(logs, rho, states, cols, candidates, first, second, focal, tail):
    """2.6's window box bound per unit (backward: emission then switch; absent bins identity);
    each sample may choose its own row."""
    active, bg, prefix, seg, _ = unit_tables(logs, cols, first, second, focal, tail)
    n, b, _ = logs.shape
    out = np.full_like(tail, -np.inf)
    for s in range(n):
        ue = np.full(b, -np.inf)
        for zi in range(len(bg)):
            z = bg[zi]
            for j in range(b):
                ue[j] = max(ue[j], tail[s, z] + prefix[s, b, zi] - prefix[s, j, zi])
        best_floor = np.full(b, -np.inf)
        for r in range(len(candidates)):
            vals = tail[s, active].copy()
            floors = np.full(b, -np.inf)
            for j in range(b - 1, -1, -1):
                if logs[s, j, 0] == 0:
                    continue
                top = ue[j]
                for t in range(j + 1, b):
                    top = max(top, floors[t] + seg[s, j, t - 1])
                for zi in range(len(active)):
                    z = active[zi]
                    aa = first[z]
                    bb = second[z]
                    ra = candidates[r, j] if aa == focal else cols[aa, j]
                    rb = candidates[r, j] if bb == focal else cols[bb, j]
                    vals[zi] += logs[s, j, 3 + pair(ra, rb)]
                    top = max(top, vals[zi])
                floors[j] = top - bin_cost(rho, s, j, states)
                for zi in range(len(active)):
                    vals[zi] = max(vals[zi], floors[j])
            for zi in range(len(active)):
                out[s, active[zi]] = max(out[s, active[zi]], vals[zi])
            for j in range(b):
                best_floor[j] = max(best_floor[j], floors[j])
        for zi in range(len(bg)):
            z = bg[zi]
            value = tail[s, z] + prefix[s, b, zi]
            for j in range(b):
                value = max(value, best_floor[j] + prefix[s, j, zi])
            out[s, z] = value
    return out


@njit(cache=True, nogil=True)
def normalized(message):
    out = message.copy()
    total = 0.0
    for s in range(len(out)):
        top = np.max(out[s])
        out[s] -= top
        total += top
    return out, total


@njit(cache=True, nogil=True)
def dominates(loser, loser_total, winner, winner_total, tolerance):
    """sum_s max_z (M_loser - M_winner) + T_loser - T_winner <= tolerance (normalised messages)."""
    difference = loser_total - winner_total
    for s in range(len(loser)):
        difference += np.max(loser[s] - winner[s])
        if difference > tolerance:
            return False
    return difference <= tolerance


@njit(cache=True, nogil=True)
def beam_template(logs, costs, states, cols, candidates, first, second, focal, groups, fwd, bwd, a, c,
                  width, current, quantum):
    """One (founder, window) beam: the best distinct path, its H gain over the incumbent, counts and
    the H certificate (best, largest dropped bound, gap, incumbent, best distinct)."""
    n, q = fwd.shape[1:]
    w = c - a
    upper = List()
    for u in range(w + 1):
        upper.append(np.empty((n, q)))
    upper[w] = bwd[groups[c - 1, 1]].copy()
    for u in range(c - 1, a - 1, -1):
        upper[u - a] = box_unit(logs[u], costs[u], states, cols[u], candidates[u], first, second, focal,
                                upper[u - a + 1])
    base = fwd[groups[a, 0]].copy()
    for u in range(a, c):
        base = extend(base, logs[u], costs[u], states, cols[u], candidates[u][current[u, focal]], first,
                      second, focal)
    incumbent = 0.0
    for s in range(n):
        incumbent += np.max(base[s] + upper[w][s])
    # The box bound over every path of the window (each sample choosing its own rows)
    # cannot beat the incumbent by more than the offer quantum -> no distinct path would be offered
    # (b1 gating), so the beam is skipped (2.6's box certificate). The margin covers the bound's
    # different summation order.
    bound = 0.0
    start = fwd[groups[a, 0]]
    for s in range(n):
        bound += np.max(start[s] + upper[0][s])
    if not bound > incumbent + quantum - 1e-9 * max(1.0, abs(incumbent)):
        return (np.full(w, -1, np.int64), -np.inf, np.array([1, 0, 0, 0, 0], np.int64),
                np.array([incumbent, -np.inf, 0.0, incumbent, -np.inf]))
    messages = np.empty((1, n, q))
    messages[0], initial = normalized(fwd[groups[a, 0]])
    totals = np.array([initial])
    paths = np.full((1, w), -1, np.int64)
    distinct = np.zeros(1, np.bool_)
    max_dropped = -np.inf
    children = 0
    cuts = 0
    tie_cuts = 0
    dominated = 0
    peak = 1
    for u in range(a, c):
        active, bg, prefix, seg, endseg = unit_tables(logs[u], cols[u], first, second, focal, upper[u - a + 1])
        bounds = rank_children(messages, logs[u], costs[u], states, cols[u], candidates[u], first, second,
                               focal, upper[u - a + 1], active, bg, prefix, seg, endseg)
        rmax = len(candidates[u])
        flat = bounds.ravel()
        children += len(flat)
        for v in range(len(flat)):
            flat[v] += totals[v // rmax]
        # Rank keys anchored at the unit's best bound (release 2.13): the bounds are totals (~|H|) that a GL
        # perturbation shifts in common by far more than the quantum, so a fixed grid on the totals reordered
        # near-equal children at its rounding boundaries; anchored at the best, only contrasts are quantised.
        anchor = -np.inf
        for v in range(len(flat)):
            anchor = max(anchor, flat[v])
        if not np.isfinite(anchor):
            anchor = 0.0
        keys = np.empty(len(flat))
        for v in range(len(flat)):
            keys[v] = np.floor((flat[v] - anchor) / quantum + 0.5)
        ranked = np.argsort(-keys, kind='mergesort')
        size = min(width, len(ranked))
        cohort = size
        if cohort < len(ranked):
            cutkey = keys[ranked[size - 1]]
            while cohort < min(2 * width, len(ranked)) and keys[ranked[cohort]] == cutkey:
                cohort += 1
        for t in range(cohort, len(ranked)):
            max_dropped = max(max_dropped, flat[ranked[t]])
            cuts += 1
            if keys[ranked[t]] == keys[ranked[size - 1]]:
                tie_cuts += 1
        order = ranked[:cohort]
        material = np.empty((cohort, n, q))
        mt = np.empty(cohort)
        md = np.empty(cohort, np.bool_)
        for j in range(cohort):
            v = order[j]
            p = v // rmax
            r = v % rmax
            out = extend(messages[p], logs[u], costs[u], states, cols[u], candidates[u][r], first, second, focal)
            material[j], add = normalized(out)
            mt[j] = totals[p] + add
            md[j] = distinct[p] or r != current[u, focal]
        keep = np.ones(cohort, np.bool_)
        for j in range(cohort):
            if not keep[j]:
                continue
            for v in range(cohort):
                if v == j or not keep[v] or (md[j] and not md[v]):
                    continue
                if dominates(material[j], mt[j], material[v], mt[v], 0.0):
                    mutual = dominates(material[v], mt[v], material[j], mt[j], 0.0)
                    if mutual and order[v] > order[j]:
                        continue
                    keep[j] = False
                    dominated += 1
                    break
        selected = np.empty(min(width, cohort), np.int64)
        used = 0
        for j in range(cohort):
            if not keep[j]:
                continue
            if used < width:
                selected[used] = j
                used += 1
            else:
                max_dropped = max(max_dropped, flat[order[j]])
                cuts += 1
        selected = selected[:used]
        canon = np.argsort(order[selected], kind='mergesort')
        selected = selected[canon]
        new = np.empty((used, n, q))
        newt = np.empty(used)
        newd = np.empty(used, np.bool_)
        newpaths = np.full((used, w), -1, np.int64)
        for j in range(used):
            v = selected[j]
            p = order[v] // rmax
            r = order[v] % rmax
            new[j] = material[v]
            newt[j] = mt[v]
            newd[j] = md[v]
            newpaths[j] = paths[p]
            newpaths[j, u - a] = r
        messages = new
        totals = newt
        distinct = newd
        paths = newpaths
        peak = max(peak, used)
    best = -np.inf
    chosen = np.full(w, -1, np.int64)
    best_all = incumbent
    for p in range(len(paths)):
        value = totals[p]
        for s in range(n):
            value += np.max(messages[p, s] + upper[w][s])
        best_all = max(best_all, value)
        if not distinct[p]:
            continue
        # A later distinct path replaces the best only when its difference to the best rounds to at least one quantum,
        # i.e. it is better by half the quantum or more (release 2.13: the difference is rounded, not each total on a
        # fixed grid); admission of the offer itself still needs a gain above the quantum.
        if np.all(chosen < 0) or np.floor((value - best) / quantum + 0.5) > 0.0:
            best = value
            chosen = paths[p].copy()
    cert = np.array([best_all, max_dropped, max(0.0, max_dropped - best_all), incumbent, best])
    counts = np.array([peak, children, cuts, tie_cuts, dominated], np.int64)
    return chosen, best - incumbent, counts, cert


@njit(cache=True, parallel=True, nogil=True)
def beam_batch(logs, costs, states, cols, candidates, first, second, groups, fwd, bwd, tasks, width, current,
               quantum):
    max_width = 1
    for t in range(len(tasks)):
        max_width = max(max_width, tasks[t, 1] - tasks[t, 0])
    paths = np.full((len(tasks), max_width), -1, np.int64)
    gains = np.full(len(tasks), -np.inf)
    counts = np.zeros((len(tasks), 5), np.int64)
    cert = np.empty((len(tasks), 5))
    for t in prange(len(tasks)):
        a, c, f = tasks[t]
        choice, gain, ct, receipt = beam_template(logs, costs, states, cols, candidates, first, second, f, groups,
                                                  fwd, bwd, a, c, width, current, quantum)
        paths[t, :c - a] = choice
        gains[t] = gain
        counts[t] = ct
        cert[t] = receipt
    return paths, gains, counts, cert


class WindowBeam:
    """The component's beam proposer (fixed binned inputs, templates and coarse context)."""

    def __init__(self, workspace, context, genetic_map, reset_rate, probability_floor):
        self.ws = workspace
        self.blocks = workspace.blocks
        emissions = workspace.emissions
        self.bin_counts = np.asarray([e.shape[2] for e in emissions], np.int64)
        self.bin_offsets = np.r_[0, np.cumsum(self.bin_counts)].astype(np.int64)
        max_pairs = max(e.shape[1] for e in emissions)
        self.logs = rank_table(List(emissions), self.bin_offsets, max_pairs)
        size = workspace.bin_size
        midpoints = []
        for panel in workspace.panels:
            for lo in range(0, panel.sites, size):
                hi = min(panel.sites, lo + size)
                midpoints.append((int(panel.positions[lo]) + int(panel.positions[hi - 1])) // 2)
        hazards = map_hazards(np.asarray(midpoints, np.int64), genetic_map, reset_rate, probability_floor)
        self.rho = compose_bin_resets(np.ascontiguousarray(self.logs[:, :, 0] != 0.0), hazards)
        self.coarse = None if context is None else (
            [tuple(int(x) for x in g) for g in context[0]], [np.asarray(p, np.int64) for p in context[1]])
        self.templates = window_templates(self.blocks, None if self.coarse is None else self.coarse[0])

    def nbytes(self):
        return int(self.logs.nbytes + self.rho.nbytes)

    def scratch_bytes(self, founders, threads=1, templates=None, width=None):
        n = self.ws.samples
        q = founders * (founders + 1) // 2
        spans = [(a, c) if kind == 'fine' else (self.coarse[0][a][0], self.coarse[0][c - 1][1])
                 for kind, a, c in (self.templates if templates is None else templates)]
        b = max(int(self.bin_offsets[c] - self.bin_offsets[a]) for a, c in spans)
        # + one contiguous copy of the rank table, resets and bin rows per unit kind (generate's unit lists).
        width = BEAM_WIDTH if width is None else int(width)
        return int(max(1, threads) * (8 * n * q * (6 * width + 12 + b) + 8 * n * b * b)
                   + 16 * (self.blocks + 1) * n * q + 2 * (self.logs.nbytes + self.rho.nbytes)
                   + 16 * founders * int(self.bin_offsets[-1]))

    def _units(self, X, kind):
        if kind == 'fine':
            groups = [(b, b + 1) for b in range(self.blocks)]
            paths = [np.arange(panel.rows, dtype=np.int64)[:, None] for panel in self.ws.panels]
        else:
            groups, context = self.coarse
            paths = []
            for u, (a, c) in enumerate(groups):
                unique = {tuple(map(int, row)) for row in (*X[:, a:c], *context[u])}
                paths.append(np.asarray(sorted(unique), np.int64))
        return groups, paths

    def generate(self, X, budget, convention=None, focals=None, templates=None, width=None):
        """Proposals [(kind, first block, end block, founder, rows)] for panel X and statistics. templates:
        [(kind, first unit, end unit)] to solve instead of the component's (the polish's long windows)."""
        templates = self.templates if templates is None else templates
        width = BEAM_WIDTH if width is None else int(width)
        started = time.perf_counter()
        cpu = time.thread_time()
        X = np.ascontiguousarray(X, np.int64)
        focals = np.arange(len(X)) if focals is None else np.asarray(focals, np.int64)
        reverse = bool(ties.reverse_conventions() if convention is None else convention)
        stats = dict(windows=0, certified=0, dp=0, fixing_dps=0, max_frontier=0, fallbacks=0, changed=0,
                     worker_cpu=0.0, templates_solved=0, templates_reused=0, beam_truncations=0,
                     beam_fallbacks=0, beam_cut=0, beam_tie_cut=0, beam_binding=0, dominated=0, children=0,
                     box_max_gap=0.0)
        proposals = []
        seen = set()
        first, second = unordered_pairs(len(X))
        states = len(first)
        quantum = float(tie_rules.quantum('refine_F', 0.0))
        haps = np.ascontiguousarray(np.repeat(X, self.bin_counts, axis=1), np.int64)
        threads = max(1, int(budget.threads()))
        with execution.reserve_memory(self.scratch_bytes(len(X), threads, templates, width),
                                      label="founder_refinement window beam"):
            with execution.numba_threads(threads):
                ff, gg = potts_bin_messages(haps, self.logs, self.bin_offsets, self.rho)
                for kind in ('fine', 'coarse'):
                    if kind == 'coarse' and self.coarse is None:
                        continue
                    # The alphabet and unit preparation below is serial Python: one thread (the rest returned to
                    # peers); the beam re-claims up to its reserved scratch team before its kernel.
                    execution.numba_threads_set(execution.capped_threads(budget, 1))
                    groups, alphabets = self._units(X, kind)
                    # Prepared-row content orders each unit's alphabet (unknown aliases are row-specific).
                    alphabets = [p[sorted(range(len(p)), key=lambda r: (tuple(self.ws.leaves[b][p[r, b - a]].tobytes()
                                 for b in range(a, c)), tuple(p[r])), reverse=reverse)]
                                 for (a, c), p in zip(groups, alphabets)]
                    cur = np.asarray([[next(r for r, path in enumerate(p) if np.array_equal(path, X[f, a:c]))
                                       for f in range(len(X))] for (a, c), p in zip(groups, alphabets)], np.int64)
                    logs = List()
                    cc = List()
                    cols = List()
                    choices = List()
                    for (a, c), p in zip(groups, alphabets):
                        lo, hi = self.bin_offsets[a], self.bin_offsets[c]
                        logs.append(np.ascontiguousarray(self.logs[:, lo:hi]))
                        cc.append(np.ascontiguousarray(self.rho[:, lo:hi]))
                        cols.append(np.ascontiguousarray(haps[:, lo:hi]))
                        choices.append(np.ascontiguousarray(np.concatenate(
                            [np.repeat(p[:, b - a:b - a + 1], self.bin_counts[b], axis=1) for b in range(a, c)],
                            axis=1), np.int64))
                    groups_arr = np.asarray(groups, np.int64)
                    tasks = np.asarray([(a, c, int(f)) for label, a, c in templates if label == kind
                                        for f in focals], np.int64)
                    if not len(tasks):
                        continue
                    execution.numba_threads_set(min(threads, max(1, int(budget.threads()))))
                    paths, values, counts, cert = beam_batch(logs, cc, states, cols, choices, first, second, groups_arr,
                                                             ff, gg, tasks, width, cur, quantum)
                    stats['max_frontier'] = max(stats['max_frontier'], int(counts[:, 0].max(initial=0)))
                    stats['children'] += int(counts[:, 1].sum())
                    stats['beam_cut'] += int(counts[:, 2].sum())
                    stats['beam_tie_cut'] += int(counts[:, 3].sum())
                    stats['dominated'] += int(counts[:, 4].sum())
                    stats['beam_binding'] += int(np.count_nonzero(counts[:, 2]))
                    stats['box_max_gap'] = max(stats['box_max_gap'], float(cert[:, 2].max(initial=0.0)))
                    for t, (i, j, f) in enumerate(tasks):
                        a, c = groups[i][0], groups[j - 1][1]
                        stats['certified'] += int(cert[t, 2] <= quantum)
                        path = paths[t, :j - i]
                        if np.any(path < 0):
                            continue
                        # b1 gating: offer only a surrogate improvement (2.6's offer rule).
                        if not values[t] > quantum:
                            continue
                        rows = np.concatenate([alphabets[u][path[u - i]] for u in range(i, j)])
                        if np.array_equal(rows, X[f, a:c]):
                            continue
                        key = kind, a, c, int(f), tuple(map(int, rows))
                        if key in seen:
                            continue
                        seen.add(key)
                        proposals.append((kind, a, c, int(f), rows))
                    stats['windows'] += len(tasks)
                    stats['dp'] += len(tasks)
                    stats['templates_solved'] += len(tasks) // max(1, len(focals))
        stats.update(changed=len(proposals), worker_cpu=time.thread_time() - cpu,
                     wall=time.perf_counter() - started)
        return proposals, stats
