"""The coarse-move guard ("shared junction"): switch events of sample paths and shared savings.

A long founder edit (a coarse move copying an L1 span) can raise F by letting several samples drop
the switch of a crossover they share (descendants of one meiosis carry the same breakpoint),
turning the founder into an artificial recombinant. A coarse move that lowers the genotype fit
E = F + penalty x switches (of the samples' lexicographic maximum paths) is refused unless its F
gain exceeds the acceptance tolerance once every saved switch at a shared one-way junction is
uncredited: dF - penalty x N_shared > TOL, N_shared = sum over samples of min(switches the sample
saves, its removed switch events at shared one-way junctions). A removed event is a founder
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
from .messages import lex_best_index
from .objective import is_skipped, row_max, unordered_pairs


@njit(cache=True, parallel=True, nogil=True)
def path_switch_events(haps, leaves, logs, offsets, penalty, f_start, fcnt_start, g_end, gcnt_end,
                       a, c, samples, focal, p1, p2, rows, reverse):
    """Switch events of each listed sample's lexicographic maximum path on blocks [a, c).

    The path is the traceback of the accumulators of messages_lex (maximum score, then the fewest
    switches among exact maxima, first state on ties) started from the current messages f_start /
    fcnt_start at boundary a and closed by g_end / gcnt_end at boundary c (one row per listed
    sample). With focal >= 0 founder focal's rows are replaced on blocks [p1, p2) by `rows`.
    Returns the path's state at boundaries a and c, its whole switch count (prefix and suffix
    counts from the messages), its number of events inside [a, c) and the events (site, state
    before, state after)."""
    first, second = unordered_pairs(len(haps))
    states = len(first)
    n = len(samples)
    lo, hi = offsets[a], offsets[c]
    width = max(hi - lo, 1)
    start_state = np.empty(n, np.int64)
    end_state = np.empty(n, np.int64)
    total = np.empty(n, np.int64)
    counts = np.zeros(n, np.int64)
    events = np.empty((n, width, 3), np.int64)
    for i in prange(n):
        s = samples[i]
        switched = np.zeros((width, states), np.bool_)
        source = np.full(width, -1, np.int64)
        row = f_start[i].copy()
        cnt = fcnt_start[i].copy()
        for b in range(a, c):
            override = focal >= 0 and p1 <= b < p2
            r = rows[b - p1] if override else 0
            leaf = leaves[b]
            base = offsets[b]
            for j in range(offsets[b], offsets[b + 1]):
                if is_skipped(logs, s, j):
                    continue
                m, k, best = _lex_best_index(row, cnt, reverse)
                sw = m - penalty
                source[j - lo] = best
                for q in range(states):
                    if sw > row[q] or (sw == row[q] and k + 1 < cnt[q]):
                        row[q] = sw
                        cnt[q] = k + 1
                        switched[j - lo, q] = True
                    x = first[q]
                    y = second[q]
                    if override:
                        d = ((leaf[r, j - base] if x == focal else haps[x, j])
                             + (leaf[r, j - base] if y == focal else haps[y, j]))
                    else:
                        d = haps[x, j] + haps[y, j]
                    row[q] += logs[s, j, d] - CENTER
            m = row_max(row)
            for q in range(states):
                row[q] -= m
        best_value = -np.inf
        best_cnt = 0
        end = 0
        for q in range(states):
            v = row[q] + g_end[i, q]
            k = cnt[q] + gcnt_end[i, q]
            if v > best_value or (v == best_value and (k < best_cnt or (reverse and k == best_cnt))):
                best_value = v
                best_cnt = k
                end = q
        q = end
        nev = 0
        for j in range(hi - 1, lo - 1, -1):
            if source[j - lo] < 0:
                continue
            if switched[j - lo, q]:
                events[i, nev, 0] = j
                events[i, nev, 1] = source[j - lo]
                events[i, nev, 2] = q
                nev += 1
                q = source[j - lo]
        start_state[i] = q
        end_state[i] = end
        counts[i] = nev
        total[i] = fcnt_start[i, q] + nev + gcnt_end[i, end]
    return start_state, end_state, total, counts, events


@njit(cache=True, nogil=True)
def _lex_best_index(row, cnt, reverse):
    """messages.lex_best_index (the lexicographic maximum: largest value, then fewest switches) with its switch
    source among exact ties: the first state, or the last under the convention reversal."""
    if not reverse:
        return lex_best_index(row, cnt)
    m = row[0]
    c = cnt[0]
    best = 0
    for q in range(1, len(row)):
        if row[q] > m or (row[q] == m and cnt[q] <= c):
            m = row[q]
            c = cnt[q]
            best = q
    return m, c, best


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
def sorted_events(events, count):
    """A path's switch events (site, before, after) as a lexicographically sorted copy."""
    out = events[:count].copy()
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
    for i in range(count):
        out[i] = events[order[i]]
    return out


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
                d1 = (own if x1 == focal else haps[x1, i]) + (own if y1 == focal else haps[y1, i])
                d2 = (own if x2 == focal else haps[x2, i]) + (own if y2 == focal else haps[y2, i])
            else:
                d1 = haps[x1, i] + haps[y1, i]
                d2 = haps[x2, i] + haps[y2, i]
            if d1 != d2:
                lo = i
                break
        out[e, 0] = lo + 1
    return out


@njit(cache=True, nogil=True)
def removed_events(old_events, old_count, new_events, new_count):
    """sorted(set(old) - set(new)) of two paths' switch events."""
    a = sorted_events(old_events, old_count)
    b = sorted_events(new_events, new_count)
    keep = np.zeros(len(a), np.bool_)
    n = 0
    j = 0
    for i in range(len(a)):
        if i > 0 and a[i, 0] == a[i - 1, 0] and a[i, 1] == a[i - 1, 1] and a[i, 2] == a[i - 1, 2]:
            continue  # set semantics
        ka = (a[i, 0], a[i, 1], a[i, 2])
        while j < len(b) and (b[j, 0], b[j, 1], b[j, 2]) < ka:
            j += 1
        if j < len(b) and (b[j, 0], b[j, 1], b[j, 2]) == ka:
            continue
        keep[i] = True
        n += 1
    out = np.empty((n, 3), np.int64)
    m = 0
    for i in range(len(a)):
        if keep[i]:
            out[m] = a[i]
            m += 1
    return out


@njit(cache=True, nogil=True)
def shared_savings(rem_sample, rem_offsets, rem_events, saved, focal, first, second, haps, logs,
                   total_sites):
    """Uncredited saved switches of the guard on flattened removed events: sample rem_sample[i] has
    the sorted events rem_events[rem_offsets[i]:rem_offsets[i + 1]]; saved[s] = net switches sample
    s saves. Returns sum over samples of min(saved, its removed events at shared one-way
    junctions)."""
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
        return 0
    ent_sample = np.empty(n_entries, np.int64)
    ent_event = np.empty(n_entries, np.int64)
    ent_u = np.empty(n_entries, np.int64)
    ent_v = np.empty(n_entries, np.int64)
    ent_site = np.empty(n_entries, np.int64)
    m = 0
    for i in range(len(rem_sample)):
        for e in range(rem_offsets[i], rem_offsets[i + 1]):
            if not one_way[e]:
                continue
            n = transition_list(first, second, rem_events[e, 1], rem_events[e, 2], trans)
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
    shared = np.zeros(n_events, np.bool_)
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
                shared[ent_event[a]] = True
                shared[ent_event[b]] = True
    total = 0
    for i in range(len(rem_sample)):
        count = 0
        for e in range(rem_offsets[i], rem_offsets[i + 1]):
            if shared[e]:
                count += 1
        if count:
            total += min(saved[rem_sample[i]], count)
    return total
