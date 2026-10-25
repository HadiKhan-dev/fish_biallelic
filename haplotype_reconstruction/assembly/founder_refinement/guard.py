"""The coarse-move guard ("shared junction"): switch events of sample paths and shared savings.

A long founder edit (a coarse move copying an L1 span) can raise F by letting several samples drop
the switch of a crossover they share (descendants of one meiosis carry the same breakpoint),
turning the founder into an artificial recombinant. A coarse move that lowers the genotype fit
E = F + switch cost (of the samples' lexicographic maximum paths) is refused unless its F gain
exceeds the acceptance tolerance once every saved switch at a shared one-way junction is
uncredited: dF - C_shared > TOL, C_shared = sum over samples of min(switch cost the sample saves,
the cost of its removed switch events at shared one-way junctions). (Release 2.8: switch costs are
the map-derived per-homologue costs of objective.py; an event changing both homologues costs 2c.) A removed event is a founder
transition u -> v of one homologue; it is one-way unless it leaves the edited founder and the same
path later rejoins it (an excursion of the founder's own carrier, the signature of a row that is
wrong there: descendants sharing both breakpoints of a short double crossover is improbable under
crossover interference); it is shared when another sample's one-way removed event is the same
transition with an overlapping evidence interval (the stretch between the sample's nearest
informative sites where u and v differ). Moves that do not lower E, and moves whose gain comes from
switch savings at unshared junctions or from excursions, pass. The acceptance pass
(acceptance.uncredited_switches) traces the paths with and without the move.

Event identity (release 2.1, design 5.3): the traceback places a switch at one site of the stretch where the two
states' genotypes are equal for the sample (or the sample has no evidence), and which site it picks there follows
exact float ties. Removed events are therefore compared by (transition, evidence interval): every event's site is
moved to the leftmost site of its interval (canonical_sites) before the set difference, using the alleles of the
panel the path was traced on (with or without the move).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ...maths.emissions import CENTER
from .objective import is_skipped, lex_better, pair_code7, row_max, unordered_pairs


@njit(inline="always")
def _lex_take(v, k, w, l, reverse):
    """(v, k) replaces the running lexicographic maximum (w, l): strictly better, or (under the
    convention reversal) exactly equal (the last state among exact ties)."""
    return v > w or (v == w and (k < l or (reverse and k == l)))


@njit(cache=True, parallel=True, nogil=True)
def path_switch_events(haps, leaves, logs, offsets, costs, f_start, fcnt_start, g_end, gcnt_end,
                       a, c, samples, focal, p1, p2, rows, reverse):
    """Switch events of each listed sample's lexicographic maximum path on blocks [a, c).

    The path is the traceback of the accumulators of messages_lex (maximum score, then the lowest
    switch cost among exact maxima) started from the current messages f_start / fcnt_start at
    boundary a and closed by g_end / gcnt_end at boundary c (one row per listed sample). With
    focal >= 0 founder focal's rows are replaced on blocks [p1, p2) by `rows`. Per site and state
    the traceback keeps the route taken (stay; one homologue from the founder maximum R[x] of the
    state's first or second founder, whose partner is recorded; both from the overall maximum),
    first state among exact ties (last under the convention reversal). Returns the path's state at
    boundaries a and c, its whole switch cost (prefix and suffix costs from the messages), its
    number of events inside [a, c), the events (site, state before, state after) and their costs
    (c at the site per changed homologue)."""
    founders = len(haps)
    first, second = unordered_pairs(founders)
    index = np.empty((founders, founders), np.int64)
    for q in range(len(first)):
        index[first[q], second[q]] = q
        index[second[q], first[q]] = q
    states = len(first)
    n = len(samples)
    lo, hi = offsets[a], offsets[c]
    width = max(hi - lo, 1)
    start_state = np.empty(n, np.int64)
    end_state = np.empty(n, np.int64)
    total = np.empty(n)
    counts = np.zeros(n, np.int64)
    events = np.empty((n, width, 3), np.int64)
    event_costs = np.empty((n, width))
    for i in prange(n):
        s = samples[i]
        route = np.zeros((width, states), np.int8)
        partner = np.empty((width, founders), np.int16)
        source = np.full(width, -1, np.int64)
        site_cost = np.zeros(width)
        R = np.empty(founders)
        Rc = np.empty(founders)
        Ry = np.zeros(founders, np.int64)
        row = f_start[i].copy()
        cst = fcnt_start[i].copy()
        for b in range(a, c):
            override = focal >= 0 and p1 <= b < p2
            r = rows[b - p1] if override else 0
            leaf = leaves[b]
            base = offsets[b]
            for j in range(offsets[b], offsets[b + 1]):
                if is_skipped(logs, s, j):
                    continue
                cj = costs[s, j]
                if cj < np.inf:
                    G = -np.inf
                    Gc = 0.0
                    Gq = 0
                    for x in range(founders):
                        R[x] = -np.inf
                        Rc[x] = 0.0
                    for q in range(states):
                        x = first[q]
                        y = second[q]
                        v = row[q]
                        k = cst[q]
                        if _lex_take(v, k, G, Gc, reverse):
                            G, Gc, Gq = v, k, q
                        if _lex_take(v, k, R[x], Rc[x], reverse):
                            R[x], Rc[x], Ry[x] = v, k, y
                        if x != y and _lex_take(v, k, R[y], Rc[y], reverse):
                            R[y], Rc[y], Ry[y] = v, k, x
                    source[j - lo] = Gq
                    site_cost[j - lo] = cj
                    for x in range(founders):
                        partner[j - lo, x] = Ry[x]
                    both = G - 2.0 * cj
                    both_c = Gc + 2.0 * cj
                    for q in range(states):
                        x = first[q]
                        y = second[q]
                        v = row[q]
                        k = cst[q]
                        taken = 0
                        if _lex_take(R[y], Rc[y], R[x], Rc[x], reverse):
                            w, wc, wr = R[y] - cj, Rc[y] + cj, 2
                        else:
                            w, wc, wr = R[x] - cj, Rc[x] + cj, 1
                        if lex_better(w, wc, v, k):
                            v, k, taken = w, wc, wr
                        if lex_better(both, both_c, v, k):
                            v, k, taken = both, both_c, 3
                        row[q] = v
                        cst[q] = k
                        route[j - lo, q] = taken
                for q in range(states):
                    x = first[q]
                    y = second[q]
                    if override:
                        d = pair_code7(leaf[r, j - base] if x == focal else haps[x, j],
                                       leaf[r, j - base] if y == focal else haps[y, j])
                    else:
                        d = pair_code7(haps[x, j], haps[y, j])
                    row[q] += logs[s, j, d] - CENTER
            m = row_max(row)
            for q in range(states):
                row[q] -= m
        best_value = -np.inf
        best_cnt = 0.0
        end = 0
        for q in range(states):
            v = row[q] + g_end[i, q]
            k = cst[q] + gcnt_end[i, q]
            if v > best_value or (v == best_value and (k < best_cnt or (reverse and k == best_cnt))):
                best_value = v
                best_cnt = k
                end = q
        q = end
        nev = 0
        spent = 0.0
        for j in range(hi - 1, lo - 1, -1):
            if source[j - lo] < 0:
                continue
            taken = route[j - lo, q]
            if taken == 0:
                continue
            x = first[q]
            y = second[q]
            if taken == 1:
                before = index[x, partner[j - lo, x]]
            elif taken == 2:
                before = index[y, partner[j - lo, y]]
            else:
                before = source[j - lo]
            events[i, nev, 0] = j
            events[i, nev, 1] = before
            events[i, nev, 2] = q
            event_costs[i, nev] = site_cost[j - lo] * (2.0 if taken == 3 else 1.0)
            spent += event_costs[i, nev]
            nev += 1
            q = before
        start_state[i] = q
        end_state[i] = end
        counts[i] = nev
        total[i] = fcnt_start[i, q] + spent + gcnt_end[i, end]
    return start_state, end_state, total, counts, events, event_costs


@njit(cache=True, nogil=True)
def transition_intervals(haps, logs, samples, from_founder, to_founder, sites, lo_site, hi_site):
    """Evidence interval (lo, hi] of founder transitions in sample paths.

    For the transition of sample samples[e] from founder from_founder[e] to to_founder[e] placed at
    site sites[e], lo is the last site before it and hi the first site at or after it where the
    sample is informative and the two founders carry different alleles (lo_site - 1 and hi_site
    when there is none inside [lo_site, hi_site)). The sample's data cannot place the breakpoint
    elsewhere than in (lo, hi]: inside it the two paths have equal emissions."""
    n = len(samples)
    out = np.empty((n, 2), np.int64)
    for e in range(n):
        s = samples[e]
        u = from_founder[e]
        v = to_founder[e]
        j = sites[e]
        lo = lo_site - 1
        for i in range(j - 1, lo_site - 1, -1):
            if haps[u, i] != haps[v, i] and not is_skipped(logs, s, i):
                lo = i
                break
        hi = hi_site
        for i in range(j, hi_site):
            if haps[u, i] != haps[v, i] and not is_skipped(logs, s, i):
                hi = i
                break
        out[e, 0] = lo
        out[e, 1] = hi
    return out


@njit(cache=True, nogil=True)
def transition_list(first, second, before, after, out):
    """Founder transitions (u -> v) of one homologue turning state `before` into `after`;
    writes them to out (4 x 2) and returns how many (1 when one homologue changes, else the 4
    candidate pairings)."""
    x1, y1, x2, y2 = first[before], second[before], first[after], second[after]
    if x1 == x2:
        out[0, 0], out[0, 1] = y1, y2
        return 1
    if x1 == y2:
        out[0, 0], out[0, 1] = y1, x2
        return 1
    if y1 == x2:
        out[0, 0], out[0, 1] = x1, y2
        return 1
    if y1 == y2:
        out[0, 0], out[0, 1] = x1, x2
        return 1
    out[0, 0], out[0, 1] = x1, x2
    out[1, 0], out[1, 1] = y1, y2
    out[2, 0], out[2, 1] = x1, y2
    out[3, 0], out[3, 1] = y1, x2
    return 4


@njit(cache=True, nogil=True)
def event_order(events, count):
    """Indices of a path's first `count` switch events (site, before, after) in lexicographic
    order."""
    order = np.arange(count)
    for i in range(1, count):
        j = i
        while j > 0:
            a, b = order[j - 1], order[j]
            if (events[a, 0], events[a, 1], events[a, 2]) > (events[b, 0], events[b, 1], events[b, 2]):
                order[j - 1], order[j] = b, a
                j -= 1
            else:
                break
    return order


@njit(cache=True, nogil=True)
def canonical_sites(events, count, s, haps, leaves, logs, offsets, first, second, focal, p1, p2, rows,
                    lo_site):
    """A copy of one path's first `count` switch events (site, state before, state after) with every site moved to
    the leftmost site of the event's evidence interval: lo + 1, lo the last site before it (lo_site - 1 when there is
    none at or after lo_site) where sample s has evidence and the two states' genotypes differ. With focal >= 0
    founder focal's rows are replaced on blocks [p1, p2) by `rows` (the panel with the move). Inside the interval the
    two states have equal emissions, so the event is the same switch wherever the traceback placed it there."""
    out = events[:count].copy()
    lo_override = offsets[p1] if focal >= 0 else 0
    hi_override = offsets[p2] if focal >= 0 else 0
    for e in range(count):
        j = events[e, 0]
        x1, y1 = first[events[e, 1]], second[events[e, 1]]
        x2, y2 = first[events[e, 2]], second[events[e, 2]]
        lo = lo_site - 1
        b = len(offsets) - 2
        while b > 0 and offsets[b] > j - 1:
            b -= 1
        for i in range(j - 1, lo_site - 1, -1):
            while b > 0 and offsets[b] > i:
                b -= 1
            if is_skipped(logs, s, i):
                continue
            if focal >= 0 and lo_override <= i < hi_override:
                r = rows[b - p1]
                own = leaves[b][r, i - offsets[b]]
                d1 = pair_code7(own if x1 == focal else haps[x1, i], own if y1 == focal else haps[y1, i])
                d2 = pair_code7(own if x2 == focal else haps[x2, i], own if y2 == focal else haps[y2, i])
            else:
                d1 = pair_code7(haps[x1, i], haps[y1, i])
                d2 = pair_code7(haps[x2, i], haps[y2, i])
            if d1 != d2:
                lo = i
                break
        out[e, 0] = lo + 1
    return out


@njit(cache=True, nogil=True)
def removed_events(old_events, old_count, new_events, new_count, old_costs):
    """sorted(set(old) - set(new)) of two paths' switch events, with each removed event's cost on
    the old path (old_costs aligned with old_events)."""
    oa = event_order(old_events, old_count)
    ob = event_order(new_events, new_count)
    keep = np.zeros(old_count, np.bool_)
    n = 0
    j = 0
    for i in range(old_count):
        e = oa[i]
        if i > 0:
            p = oa[i - 1]
            if (old_events[e, 0] == old_events[p, 0] and old_events[e, 1] == old_events[p, 1]
                    and old_events[e, 2] == old_events[p, 2]):
                continue  # set semantics
        ka = (old_events[e, 0], old_events[e, 1], old_events[e, 2])
        while j < new_count and (new_events[ob[j], 0], new_events[ob[j], 1], new_events[ob[j], 2]) < ka:
            j += 1
        if j < new_count and (new_events[ob[j], 0], new_events[ob[j], 1], new_events[ob[j], 2]) == ka:
            continue
        keep[i] = True
        n += 1
    out = np.empty((n, 3), np.int64)
    out_costs = np.empty(n)
    m = 0
    for i in range(old_count):
        if keep[i]:
            out[m] = old_events[oa[i]]
            out_costs[m] = old_costs[oa[i]]
            m += 1
    return out, out_costs


@njit(cache=True, nogil=True)
def shared_savings(rem_sample, rem_offsets, rem_events, rem_costs, saved, focal, first, second,
                   haps, logs, total_sites):
    """Uncredited saved switch cost of the guard on flattened removed events: sample rem_sample[i]
    has the sorted events rem_events[rem_offsets[i]:rem_offsets[i + 1]] with costs rem_costs;
    saved[s] = net switch cost sample s saves. Returns the sum over samples of min(saved, the cost
    of its removed homologue transitions at shared one-way junctions).

    Release 2.8: a removed event that changes one homologue is one transition
    (cost c); one that changes both is two transitions of cost c each under one of its two
    consistent pairings (x1 -> x2, y1 -> y2) or (x1 -> y2, y1 -> x2). Every transition is tested for
    sharing separately, and a double change is charged c per shared transition of its better
    pairing (at most 2c), never its whole 2c for one shared homologue.

    Declared approximation: this is a CONSERVATIVE UPPER BOUND on the shared cost,
    not an identified attribution. Each ambiguous double change picks its own best pairing and
    other samples may match its other alternative (so the total can exceed every globally
    consistent pairing), and events are compared as whole diplotype events (removed_events), so a
    homologue transition that persists while the other homologue changes counts as removed. Both
    errors only uncredit more, so the guard can reject more coarse moves than an exact attribution
    would, never fewer; the extra rejections are counted (guard_rejected) and checked on the native
    set. Unordered tracebacks do not identify homologue pairings, and resolving the pairings jointly
    would be a combinatorial search."""
    trans = np.empty((4, 2), np.int64)
    # one-way events (FIFO pairing of the focal founder's excursions per sample)
    n_events = len(rem_events)
    one_way = np.ones(n_events, np.bool_)
    for i in range(len(rem_sample)):
        lo, hi = rem_offsets[i], rem_offsets[i + 1]
        open_flag = np.zeros(hi - lo, np.bool_)
        key_u = np.full(hi - lo, -1, np.int64)
        key_v = np.full(hi - lo, -1, np.int64)
        for e in range(hi - lo):
            ev = rem_events[lo + e]
            n = transition_list(first, second, ev[1], ev[2], trans)
            if n != 1:
                continue
            u, v = trans[0, 0], trans[0, 1]
            if focal != u and focal != v:
                continue
            key_u[e], key_v[e] = u, v
            partner = -1
            for w in range(e):
                if open_flag[w] and key_u[w] == v and key_v[w] == u:
                    partner = w
                    break
            if partner >= 0:
                open_flag[partner] = False
                one_way[lo + partner] = False
                one_way[lo + e] = False
            else:
                open_flag[e] = True
    # entries: every transition of every one-way event
    n_entries = 0
    for i in range(len(rem_sample)):
        for e in range(rem_offsets[i], rem_offsets[i + 1]):
            if one_way[e]:
                n_entries += transition_list(first, second, rem_events[e, 1], rem_events[e, 2], trans)
    if n_entries == 0:
        return 0.0
    ent_sample = np.empty(n_entries, np.int64)
    ent_event = np.empty(n_entries, np.int64)
    ent_u = np.empty(n_entries, np.int64)
    ent_v = np.empty(n_entries, np.int64)
    ent_site = np.empty(n_entries, np.int64)
    ev_first = np.full(n_events, -1, np.int64)
    ev_count = np.zeros(n_events, np.int64)
    m = 0
    for i in range(len(rem_sample)):
        for e in range(rem_offsets[i], rem_offsets[i + 1]):
            if not one_way[e]:
                continue
            n = transition_list(first, second, rem_events[e, 1], rem_events[e, 2], trans)
            ev_first[e] = m
            ev_count[e] = n
            for t in range(n):
                ent_sample[m] = rem_sample[i]
                ent_event[m] = e
                ent_u[m] = trans[t, 0]
                ent_v[m] = trans[t, 1]
                ent_site[m] = rem_events[e, 0]
                m += 1
    intervals = transition_intervals(haps, logs, ent_sample, ent_u, ent_v, ent_site, 0, total_sites)
    # group by (u, v), sort by (lo, hi, sample, event), mark overlapping pairs of samples
    order = np.arange(n_entries)
    for i in range(1, n_entries):
        j = i
        while j > 0:
            a, b = order[j - 1], order[j]
            ka = (ent_u[a], ent_v[a], intervals[a, 0], intervals[a, 1], ent_sample[a], ent_event[a])
            kb = (ent_u[b], ent_v[b], intervals[b, 0], intervals[b, 1], ent_sample[b], ent_event[b])
            if ka > kb:
                order[j - 1], order[j] = b, a
                j -= 1
            else:
                break
    shared = np.zeros(n_entries, np.bool_)
    for x in range(n_entries):
        a = order[x]
        for y in range(x + 1, n_entries):
            b = order[y]
            if ent_u[b] != ent_u[a] or ent_v[b] != ent_v[a]:
                break
            if intervals[b, 0] >= intervals[a, 1]:
                break
            if ent_sample[a] != ent_sample[b] and max(intervals[a, 0], intervals[b, 0]) < min(
                    intervals[a, 1], intervals[b, 1]):
                shared[a] = True
                shared[b] = True
    total = 0.0
    for i in range(len(rem_sample)):
        cost = 0.0
        for e in range(rem_offsets[i], rem_offsets[i + 1]):
            f = ev_first[e]
            if f < 0:
                continue
            if ev_count[e] == 1:
                if shared[f]:
                    cost += rem_costs[e]
            else:
                # rem_costs[e] = 2c: c per shared transition of the better consistent pairing
                pairing_a = int(shared[f]) + int(shared[f + 1])
                pairing_b = int(shared[f + 2]) + int(shared[f + 3])
                cost += 0.5 * rem_costs[e] * max(pairing_a, pairing_b)
        if cost > 0.0:
            total += min(saved[rem_sample[i]], cost)
    return total


def traceback_bytes(samples: int, sites: int, founders: int, threads: int) -> int:
    """Conservative peak of one guard's full-width tracing beyond its message arrays.

    Both old/new int64 event tensors coexist (48 N L bytes). Up to N L removed
    events add 24 N L. shared_savings expands each event into at most four
    transitions: five int64 entry arrays, two interval coordinates and an order
    index add 256 N L, and two masks add 2 N L. Canonicalisation, sorting and
    one-sample FIFO scratch are covered by another 200 L bytes. Each active thread
    holds int8[L,Q] traceback routes, int16[L,K] founder-maximum partners and int64 + float64[L]
    source and cost vectors, while boundary, reconstructed-prefix and per-thread messages/costs add
    64 N Q. Event costs add 24 N L (old and new traceback cost arrays and the flattened removed-event
    costs), and the per-event transition index (ev_first, ev_count: 16 N L) and
    per-transition shared flags (3 N L) another 19 N L. Width may grow to the entire component; using the whole width in admission is
    deliberate.
    """
    states = founders * (founders + 1) // 2
    active = min(samples, max(1, threads))
    return int((373 * samples + 200) * sites + active * sites * (states + 2 * founders + 16)
               + 64 * samples * states)
