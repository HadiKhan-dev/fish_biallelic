"""Prototype snapshot compound schedule; serial MAP kernels remain the reference.

Coordinates already come from pedigree/scaffold data. A catalogue is fixed for
one sweep. Conflicts conservatively include every read row and transition guard;
only disjoint factors/windows execute together. New pairs wait for the next sweep.
"""
from dataclasses import dataclass
import heapq
import numpy as np
from numba import njit
from ..execution.budget import current_budget
from . import map_solver as ms
from .work_bounds import BoundedSet, FamilyWorkBound, configured_bound


@dataclass(frozen=True)
class Move:
    kind: int
    centre: int
    source: int
    mask: int
    bits: int
    a: int
    b: int
    lo: int
    hi: int
    read_lo: int
    read_hi: int
    rows: tuple
    triggers: tuple = ()


@njit(cache=True, nogil=True)
def _changes(selector, markers, mask):
    result = []
    for k in range(1, len(markers)):
        bits = (selector[markers[k]] ^ selector[markers[k - 1]]) & mask
        if bits:
            result.append((int(markers[k]), int(bits), int(k)))
    return result


@njit(cache=True, nogil=True)
def _valid_pair(row, markers, mask, bits, a, b, sites):
    first = np.searchsorted(markers, a)
    last = np.searchsorted(markers, b)
    if a > 0:
        if first <= 0 or first >= len(markers) or markers[first] != a:
            return False
        if ((row[markers[first]] ^ row[markers[first - 1]]) & mask) != bits:
            return False
    if b < sites:
        if last <= 0 or last >= len(markers) or markers[last] != b:
            return False
        changed = (row[markers[last]] ^ row[markers[last - 1]]) & mask
        if not changed or (a == 0 and changed != bits):
            return False
    for k in range(max(1, first + (a > 0)), min(last, len(markers))):
        if (row[markers[k]] ^ row[markers[k - 1]]) & mask:
            return False
    return True


def _neighbourhood(fm, centre, relabel=False):
    bound = configured_bound("HAPLOTYPES_COMPOUND_WORK_BOUND")
    key = (int(centre), bool(relabel), bound)
    cache = getattr(fm, '_compound_layouts', None)
    if cache is None:
        fm._compound_layouts = cache = {}
    if key in cache:
        if cache[key] is None:
            raise FamilyWorkBound(bound)
        return cache[key]
    try:
        g = fm.graph
        children = lambda p: fm.child_groups[fm.child_offsets[p]:fm.child_offsets[p + 1]]
        if relabel:
            moving = BoundedSet(children(centre), bound)
            roots = BoundedSet((centre,), bound)
        else:
            own = int(g.node_group[centre])
            moving = BoundedSet((own,), bound)
            moving.update(children(centre))
            roots = BoundedSet(bound=bound)
            for parent in g.group_parent[own]:
                if parent < 0:
                    continue
                moving.update(map(int, children(parent)))
                if fm.roots[parent]:
                    roots.add(int(parent))
                else:
                    moving.add(int(g.node_group[parent]))
        read_groups = BoundedSet(moving, bound)
        for group in moving:
            read_groups.update(map(int, children(g.group_child[group])))
            roots.update(int(p) for p in g.group_parent[group] if p >= 0 and fm.roots[p])
        for root in roots:
            read_groups.update(map(int, children(root)))
        rows = BoundedSet(roots, bound)
        for group in read_groups:
            rows.add(int(g.group_child[group]))
            rows.update(int(p) for p in g.group_parent[group] if p >= 0)
        cache[key] = (tuple(sorted(moving)), tuple(sorted(rows)))
    except FamilyWorkBound:
        cache[key] = None
        raise
    return cache[key]


def _bounds(fm, moving, a, b, lo, hi, kind):
    if kind >= 2:
        # Tolerant sibling pairs can extend by one seed span; their padding is
        # bounded by the existing COMPOUND_WIDTH/MARGIN. This is a read envelope,
        # not an altered decoding window.
        span = b - a
        width = max(ms.COMPOUND_MARGIN, ms.COMPOUND_WIDTH)
        lo = min(lo, max(0, a - span - width))
        hi = max(hi, min(fm.sites, b + span + width))
    end = hi
    for group in moving:
        end = max(end, int(ms._window_end(group, hi, fm.marker_offsets, fm.marker_sites, fm.sites)))
    left, right = max(0, lo - 1), min(fm.sites, end + 1)
    for group in moving:
        x, y = fm.marker_offsets[group:group + 2]
        markers = fm.marker_sites[x:y]
        l, r = np.searchsorted(markers, (lo, end))
        if l:
            left = min(left, int(markers[l - 1]))
        if r < len(markers):
            right = max(right, int(markers[r]) + 1)
    return left, right


def catalogue(fm, state, tier=0, relabel=False):
    g, length = fm.graph, fm.sites
    cached_changes = {}
    def changes(source, mask):
        key = (int(source), int(mask))
        if key not in cached_changes:
            a, b = fm.marker_offsets[source:source + 2]
            cached_changes[key] = _changes(state.selector[source], fm.marker_sites[a:b], mask)
        return cached_changes[key]
    result = []
    skipped = []
    fm.last_compound_skips = skipped
    def layout(centre, is_relabel=False):
        try:
            return _neighbourhood(fm, centre, is_relabel)
        except FamilyWorkBound as exc:
            # No gain was computed: this is omitted search effort, never a
            # losing proposal or evidence supporting a phase join.
            return None
    if relabel:
        roots = np.flatnonzero(fm.roots)
        for root in (roots[::-1] if fm.reverse else roots):
            starts = fm.frozen_starts if fm.workspace.frozen[root] else fm.free_starts
            by_run = {}
            for index in range(fm.child_offsets[root], fm.child_offsets[root + 1]):
                source = int(fm.child_groups[index]); mask = 1 << (1 - int(fm.child_slots[index]))
                for marker, bits, k in changes(source, mask):
                    r = int(np.searchsorted(starts[:-1], marker))
                    choices = (r, r - 1) if fm.reverse else (r - 1, r)
                    choices = [x for x in choices if 1 <= x < len(starts) - 1]
                    if not choices:
                        continue
                    run = min(choices, key=lambda x: abs(int(starts[x]) - marker))
                    if abs(int(starts[run]) - marker) <= ms.RELABEL_REACH:
                        by_run.setdefault(run, []).append((source, mask, marker, k))
            neighbourhood = layout(int(root), True)
            if neighbourhood is None:
                bound = configured_bound("HAPLOTYPES_COMPOUND_WORK_BOUND")
                for run in sorted(by_run, reverse=fm.reverse):
                    skipped.append(dict(centre=int(root), samples=[int(root)], cuts=[int(starts[run])],
                        relabel=True, tier=tier, reason='computational_bound', family_work_bound=bound,
                        sample_units_at_least=bound+1))
                continue
            moving, rows = neighbourhood
            for run in sorted(by_run, reverse=fm.reverse):
                triggers = tuple(by_run[run]); boundary = int(starts[run])
                lo = max(0, min([boundary] + [x[2] for x in triggers]) - ms.COMPOUND_MARGIN)
                hi = min(length, max([boundary] + [x[2] for x in triggers]) + ms.COMPOUND_MARGIN + 1)
                # Relabelling changes an anchored selector/frame tail. Full-row
                # conflicts deliberately forbid calling separate tails independent.
                result.append(Move(4, int(root), -1, 0, run, boundary - 1, boundary,
                                   lo, hi, 0, length, rows, triggers))
        return result
    for kind in range(2 if tier == 1 else 4):
        order = range(len(g.group_child) - 1, -1, -1) if fm.reverse else range(len(g.group_child))
        for group in order:
            centre = int(g.group_child[group])
            sources = [(group, 3)]
            if kind == 1:
                if fm.child_offsets[centre] == fm.child_offsets[centre + 1]:
                    continue
                sources += [(int(fm.child_groups[x]), 1 << (1 - int(fm.child_slots[x])))
                            for x in range(fm.child_offsets[centre], fm.child_offsets[centre + 1])]
            if fm.reverse:
                sources.reverse()
            neighbourhood = layout(centre)
            if neighbourhood is not None:
                moving, rows = neighbourhood
            for source, mask in sources:
                ch = changes(source, mask)
                pairs = [(ch[k][0], ch[k + 1][0], ch[k][1]) for k in range(len(ch) - 1)]
                if fm.reverse:
                    pairs.reverse()
                if tier == 1:
                    pairs = [x for x in pairs if x[1] - x[0] > ms.LONG_SPAN]
                    if ch:
                        terminals = [(0, ch[0][0], ch[0][1]), (ch[-1][0], length, ch[-1][1])]
                        pairs += terminals[::-1] if fm.reverse else terminals
                else:
                    pairs = [x for x in pairs if x[1] - x[0] <= ms.LONG_SPAN]
                for a, b, bits in pairs:
                    if kind == 3 and b - a > ms.COMPOUND_WIDTH:
                        continue
                    if kind == 2:
                        parent = g.group_parent[source, 0] if bits == 2 else (g.group_parent[source, 1] if bits == 1 else -1)
                        if parent < 0 or not fm.roots[parent]:
                            continue
                    if neighbourhood is None:
                        bound = configured_bound("HAPLOTYPES_COMPOUND_WORK_BOUND")
                        skipped.append(dict(centre=centre, source=int(source), kind=kind,
                            samples=sorted({centre,int(g.group_child[source])}), cuts=[int(a),int(b)],
                            relabel=False, tier=tier, reason='computational_bound', family_work_bound=bound,
                            sample_units_at_least=bound+1))
                        continue
                    width = max(ms.COMPOUND_MARGIN, min(b - a, ms.COMPOUND_WIDTH))
                    lo, hi = max(0, a - width), min(length, b + width)
                    left, right = _bounds(fm, moving, a, b, lo, hi, kind)
                    result.append(Move(kind, centre, source, mask, bits, a, b, lo, hi, left, right, rows))
    return result


def colour_moves(moves):
    """Interval sweep, with factor-row occupancy per colour; no candidate all-pairs scan."""
    active, occupancy, result = [], {}, []
    for i in sorted(range(len(moves)), key=lambda x: (moves[x].read_lo, x)):
        move = moves[i]
        while active and active[0][0] <= move.read_lo:
            _, old, colour = heapq.heappop(active)
            for row in moves[old].rows:
                counts = occupancy[row]
                counts[colour] -= 1
                if not counts[colour]:
                    del counts[colour]
        used = set()
        for row in move.rows:
            used.update(occupancy.get(row, ()))
        colour = 0
        while colour in used:
            colour += 1
        while len(result) <= colour:
            result.append([])
        result[colour].append(i)
        for row in move.rows:
            counts = occupancy.setdefault(row, {})
            counts[colour] = counts.get(colour, 0) + 1
        heapq.heappush(active, (move.read_hi, i, colour))
    return [tuple(sorted(x)) for x in result]


def evaluate(fm, state, move):
    g, w = fm.graph, fm.workspace
    if move.kind < 4:
        a, b = fm.marker_offsets[move.source:move.source + 2]
        valid = _valid_pair(state.selector[move.source], fm.marker_sites[a:b], move.mask,
                            move.bits, move.a, move.b, fm.sites)
    else:
        valid = False
        for source, mask, marker, k in move.triggers:
            previous = fm.marker_sites[fm.marker_offsets[source] + k - 1]
            if (state.selector[source, marker] ^ state.selector[source, previous]) & mask:
                valid = True
                break
    if not valid:
        return 0., False, False
    args = (w.potentials, state.genotype, state.selector, state.frame, w.orientation, fm.roots, w.frozen,
            fm.free_starts, fm.free_trans, fm.frozen_starts, fm.frozen_trans, w.anchors, g.node_group,
            g.group_parent, g.group_child, fm.child_offsets, fm.child_groups, fm.child_slots,
            fm.marker_offsets, fm.marker_sites, fm.marker_theta, fm.log_keep, fm.log_error, fm.log_match, fm.log_miss)
    if move.kind == 4:
        gain, accepted = ms._relabel_compound(move.centre, move.bits, move.lo, move.hi, fm.quantum,
                                             *args, fm.reverse)
    else:
        gain, accepted = ms._compound(move.kind, move.centre, move.bits, move.a, move.b,
                                     move.lo, move.hi, fm.quantum, *args, fm.reverse)
    return float(gain), bool(accepted), True


def sweep(fm, state, decisions, count, tier=0, relabel=False):
    moves = catalogue(fm, state, tier, relabel)
    colours = colour_moves(moves)
    total, accepted = 0., 0
    for colour in colours:
        items = [moves[x] for x in colour]
        sizes = []
        for move in items:
            moving, _ = _neighbourhood(fm, move.centre, move.kind == 4)
            end = max([move.hi] + [int(ms._window_end(group, move.hi, fm.marker_offsets,
                        fm.marker_sites, fm.sites)) for group in moving])
            span = end - move.lo if move.kind == 4 else move.read_hi - move.read_lo
            sizes.append(span * (2 * len(move.rows) + 160))
        values = current_budget().thread_map(lambda x: evaluate(fm, state, x), items,
                                             item_bytes=sizes, label='compound colour')
        for index, (gain, ok, valid) in zip(colour, values):
            move = moves[index]
            if valid and abs(gain) < 1. and count[0] < decisions.shape[0]:
                code = 6 if relabel else (3 + move.kind if move.kind < 3 else 8)
                decisions[count[0]] = (code, move.centre if relabel else move.source, move.a, gain)
                count[0] += 1
            if ok:
                accepted += 1; total += gain
    fm.compound_layout.append(dict(relabel=relabel, tier=tier, candidates=len(moves),
                                   colours=[len(x) for x in colours], accepted=accepted,
                                   skipped_neighbourhoods=fm.last_compound_skips))
    return accepted, total
