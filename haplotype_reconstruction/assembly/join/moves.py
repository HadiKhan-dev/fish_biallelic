"""The join's same-count neighbourhood: moves, their bounds and best-first exact scoring.

A same-count move changes the panel's paths without changing their number: replace(f, c) (path f
becomes pool path c), local(f, b, r) (path f takes row r at block b), onesided(f, g, t) (path f
takes g's rows from block t on) and splice(f, g, t) (f and g exchange their rows from block t on).
birth(c) is bounded alongside replace(f, c) (pool bounds) but never accepted here: count changes
are decided by the count step. :func:`lower_bounds` turns the two-region bounds of
:mod:`.certify` into lower bounds on each move's change of the join score -2 LL;
:func:`best_improvement` scores the survivors exactly, best bound first.
"""
from __future__ import annotations

import numpy as np
import numba
from ...execution import reserve_memory
from .pair_segments import workspace_bytes as segment_workspace_bytes

from .certify import structural_bounds
from .objective import column_sums

KINDS = ('replace', 'local', 'onesided', 'splice', 'birth')
REPLACE, LOCAL, ONESIDED, SPLICE, BIRTH = range(5)
# Structural moves bounded per structural_bounds call (their per-sample values are held at once).
MOVES_PER_CHUNK = 8192


class Moves:
    """Same-count neighbourhood moves as arrays in kernel order, with canonical keys.

    canonical rows: (kind, founder, partner, block, row or pool index), -1 for
    unused fields, kind order as in KINDS; pool indices are global (offset).
    Pool moves come first (per pool path in lexicographic order: birth, then
    replace(f) for every row f; bounded by PanelForwards.pool_bounds over shared
    prefixes and suffixes; birth(c) has size_change 1 and is excluded by
    lower_bounds), then the structural moves (structural_bounds): local, onesided and
    splice moves whose result keeps k distinct paths.
    """

    _FIELDS = ('kind', 'first_removed', 'second_removed', 'first_new', 'second_new',
               'size_change')

    def __init__(self, rows, capacity):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.capacity = capacity
        self.lookup = {}
        for index, row in enumerate(map(tuple, self.rows.tolist())):
            self.lookup.setdefault(row, index)
        self._parts = []
        self._paths = []
        self._path_count = 0
        self._count = 0
        self.pool_paths = np.zeros((0, self.rows.shape[1]), dtype=np.int64)
        self.pool_births = False
        self.pool_moves = 0

    def _add_paths(self, paths):
        start = self._path_count
        self._paths.append(np.asarray(paths, dtype=np.int64).reshape(-1, self.rows.shape[1]))
        self._path_count += len(paths)
        return start

    def _members(self, paths):
        return np.asarray([self.lookup.get(path, -1) for path in map(tuple, paths.tolist())],
                          dtype=np.int64)

    def _add(self, kind, first_removed, second_removed, first_new, second_new, size_change,
             canonical):
        count = len(kind)
        if not count:
            return
        arrays = [np.broadcast_to(np.asarray(value, dtype=np.int64), (count,)).copy()
                  for value in (kind, first_removed, second_removed, first_new, second_new,
                                size_change)]
        self._parts.append((arrays, np.asarray(canonical, dtype=np.int64).reshape(count, 5)))
        self._count += count

    def add_pool(self, pool, pool_offset=0):
        """birth(c) and replace(f, c) for every pool path c not in the panel.

        Must be the first part added. Returns the number of pool paths tested.
        """
        k, blocks = self.rows.shape
        pool = np.asarray(pool, dtype=np.int64).reshape(-1, blocks)
        eligible = np.flatnonzero(self._members(pool) < 0)
        if not len(eligible):
            return 0
        eligible = eligible[np.lexsort(pool[eligible][:, ::-1].T)]
        paths = np.ascontiguousarray(pool[eligible])
        self.pool_paths = paths
        self.pool_births = births = k + 1 <= self.capacity
        group = k + int(births)
        shape = (len(eligible), group)
        kind = np.full(shape, REPLACE)
        first_removed = np.full(shape, -1)
        change = np.zeros(shape, dtype=np.int64)
        canonical = np.full(shape + (5,), -1, dtype=np.int64)
        canonical[..., 4] = (pool_offset + eligible)[:, None]
        columns = slice(1, None) if births else slice(None)
        first_removed[:, columns] = np.arange(k)[None, :]
        canonical[:, columns, 0] = REPLACE
        canonical[:, columns, 1] = np.arange(k)[None, :]
        if births:
            kind[:, 0] = BIRTH
            change[:, 0] = 1
            canonical[:, 0, 0] = BIRTH
        path_index = self._add_paths(paths) + np.arange(len(eligible))
        self._add(kind.ravel(), first_removed.ravel(), -1, np.repeat(path_index, group), -1,
                  change.ravel(), canonical.reshape(-1, 5))
        self.pool_moves = self._count
        return len(eligible)

    def add_structural(self, block_sizes, slots=None):
        """local, onesided and splice moves whose result keeps k distinct paths.

        slots: if given, only moves that change at least one of these rows (a seed refit)."""
        rows = self.rows
        k, blocks = rows.shape
        mine = np.ones(k, dtype=bool) if slots is None else np.isin(np.arange(k), list(slots))
        founders, positions, alternatives = [], [], []
        for block in range(blocks):
            choices = np.arange(block_sizes[block])
            for founder in np.flatnonzero(mine).tolist():
                other = choices[choices != rows[founder, block]]
                founders.append(np.full(len(other), founder))
                positions.append(np.full(len(other), block))
                alternatives.append(other)
        if founders:
            founder = np.concatenate(founders)
            block = np.concatenate(positions)
            row = np.concatenate(alternatives)
            paths = rows[founder].copy()
            paths[np.arange(len(founder)), block] = row
            member = self._members(paths)
            index = self._add_paths(paths) + np.arange(len(founder))
            canonical = np.stack([np.full(len(founder), LOCAL), founder,
                                  np.full(len(founder), -1), block, row], axis=1)
            keep = member < 0
            self._add(np.full(int(keep.sum()), LOCAL), founder[keep], -1, index[keep], -1, 0,
                      canonical[keep])
        if k >= 2 and blocks >= 2:
            pairs = np.asarray([(a, b) for a in range(k) for b in range(k) if a != b],
                               dtype=np.int64)
            ordinal = np.full((k, k), -1, dtype=np.int64)
            ordinal[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs))
            boundaries = np.arange(1, blocks)
            first = np.repeat(pairs[:, 0], len(boundaries))
            second = np.repeat(pairs[:, 1], len(boundaries))
            boundary = np.tile(boundaries, len(pairs))
            paths = np.where(np.arange(blocks)[None, :] < boundary[:, None],
                             rows[first], rows[second])
            member = self._members(paths)
            start = self._add_paths(paths)
            index = start + np.arange(len(first))
            zeros = np.full(len(first), -1)
            base = np.stack([zeros, first, second, boundary, zeros], axis=1)

            def canonical_for(kind, mask):
                values = base[mask].copy()
                values[:, 0] = kind
                return values
            keep = (member < 0) & mine[first]
            self._add(np.full(int(keep.sum()), ONESIDED), first[keep], -1, index[keep], -1, 0,
                      canonical_for(ONESIDED, keep))
            partner = start + ordinal[second, first] * len(boundaries) + boundary - 1
            keep = ((first < second) & (member < 0) & (member[partner - start] < 0)
                    & (mine[first] | mine[second]))
            self._add(np.full(int(keep.sum()), SPLICE), first[keep], second[keep],
                      index[keep], partner[keep], 0, canonical_for(SPLICE, keep))

    def finish(self):
        self.kernel_moves = self._count
        if self._parts:
            for position, name in enumerate(self._FIELDS):
                setattr(self, name, np.concatenate([arrays[position] for arrays, _ in self._parts]))
            self.canonical = np.concatenate([canonical for _, canonical in self._parts])
        else:
            for name in self._FIELDS:
                setattr(self, name, np.zeros(0, dtype=np.int64))
            self.canonical = np.zeros((0, 5), dtype=np.int64)
        self.new_paths = (np.concatenate(self._paths) if self._paths
                          else np.zeros((0, self.rows.shape[1]), dtype=np.int64))
        del self._parts, self._paths
        return self

    def __len__(self):
        return len(self.kind)

    def counts(self):
        values = np.bincount(self.kind, minlength=len(KINDS))
        return {name: int(values[index]) for index, name in enumerate(KINDS) if values[index]}

    def description(self, move):
        """The move as the search describes it: ('replace', f, pool index), ('local', f, b, r),
        ('onesided' | 'splice', f, g, t) or ('birth', (pool index,))."""
        kind, founder, partner, block, index = (int(x) for x in self.canonical[move])
        if kind == REPLACE:
            return ('replace', founder, index)
        if kind == LOCAL:
            return ('local', founder, block, index)
        if kind in (ONESIDED, SPLICE):
            return (KINDS[kind], founder, partner, block)
        return ('birth', (index,))


def replaced_row(moves, move):
    """The panel row f a move replaces by one new path, or None for any other move."""
    first = int(moves.first_removed[move])
    if first < 0 or int(moves.second_removed[move]) >= 0 or int(moves.second_new[move]) >= 0:
        return None
    return first


def structural_move_bounds(forwards, moves, local_new, offsets, keep, limit=MOVES_PER_CHUNK):
    """Summed bounds of the structural moves (local, onesided, splice) in move order, and the
    per-sample bounds of the moves whose sum keep(sums) retains ({position: vector}).

    Moves are bounded by structural_bounds in groups of consecutive founders (the removed
    row) holding at most ``limit`` moves where possible, so the per-sample values of one
    group are held at a time."""
    lo, hi = moves.pool_moves, moves.kernel_moves
    result = np.empty(hi - lo)
    kept = {}
    if hi <= lo:
        return result, kept
    k, blocks = forwards.rows.shape
    kind = moves.kind[lo:hi]
    founder = moves.first_removed[lo:hi]
    partner = moves.canonical[lo:hi, 2]
    position = moves.canonical[lo:hi, 3]
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    per_founder = np.bincount(founder, minlength=k)
    start = 0
    while start < k:
        stop = start + 1
        size = per_founder[start]
        while stop < k and size + per_founder[stop] <= limit:
            size += per_founder[stop]
            stop += 1
        chosen = np.flatnonzero((founder >= start) & (founder < stop))
        start = stop
        if not len(chosen):
            continue
        columns = np.arange(len(chosen))
        group_kind = kind[chosen]
        active = np.zeros(k, dtype=np.bool_)
        active[founder[chosen][group_kind != SPLICE]] = True
        local = group_kind == LOCAL
        moves_local = np.ascontiguousarray(np.stack([
            columns[local], founder[chosen][local], position[chosen][local],
            local_new[moves.first_new[lo:hi][chosen][local], position[chosen][local]]], axis=1),
            dtype=np.int64)
        onesided = np.full((k, k, blocks), -1, dtype=np.int64)
        mask = group_kind == ONESIDED
        onesided[founder[chosen][mask], partner[chosen][mask], position[chosen][mask]] = columns[mask]
        splice = np.full((k, k, blocks), -1, dtype=np.int64)
        mask = group_kind == SPLICE
        splice[founder[chosen][mask], partner[chosen][mask], position[chosen][mask]] = columns[mask]
        # At most one segment scratch set per active sample thread. Concurrent seed
        # calls reserve separately; the enclosing batch envelope lends available bytes.
        query_pairs = int(np.count_nonzero(np.any(splice >= 0, axis=2)))
        scratch = segment_workspace_bytes(k, forwards.total_bins, query_pairs)
        scratch *= min(forwards.layout.samples, numba.get_num_threads())
        with reserve_memory(scratch, label="join pair segments"):
            values = structural_bounds(
                *forwards.layout.arrays(), forwards.rows, forwards.penalty, forwards.total_bins,
                forwards.forward, forwards.index, forwards.excluded, forwards.second, forwards.pairs,
                offsets, active, moves_local, onesided, splice, len(chosen))
        sums = column_sums(values)
        result[chosen] = sums
        for column in np.flatnonzero(keep(sums)).tolist():
            kept[int(chosen[column])] = values[:, column].copy()
    return result, kept


def lower_bounds(moves, forwards, likelihood, to_local, offsets, replaced=None):
    """Lower bound of every move's change of the join score -2 LL: -2 (sum_s U_s - LL(P)).

    to_local maps paths to emission rows (int64 arrays); offsets are the first bin of every
    block. birth(c) moves are bounded with their replace(f, c) moves but get +inf: the panel
    size is decided by the count step. Also returns the per-sample bounds of every move
    best_improvement's exclusion test keeps ({move: vector}), with which the exact scores skip
    samples. replaced: bound only the replacements of these panel rows
    (PanelForwards.pool_bounds); the caller excludes the others."""
    totals = np.empty(len(moves))
    likelihood = float(likelihood)
    eps = 1e-9 * (1.0 + abs(-2.0 * likelihood))
    threshold = max(1e-8, eps)

    def keep(sums):
        return -2.0 * (sums - likelihood) <= -threshold + eps
    sample_bounds = {}
    pool = moves.pool_moves
    if pool:
        values, kept = forwards.pool_bounds(to_local(moves.pool_paths), offsets, keep, replaced)
        group = values.shape[1] - (0 if moves.pool_births else 1)
        shift = 0 if moves.pool_births else 1
        totals[:pool] = (values if moves.pool_births else values[:, 1:]).ravel()
        for (path, column), vector in kept.items():
            sample_bounds[path * group + column - shift] = vector
    if moves.kernel_moves > pool:
        totals[pool:moves.kernel_moves], kept = structural_move_bounds(
            forwards, moves, to_local(moves.new_paths), offsets, keep)
        for position, vector in kept.items():
            sample_bounds[pool + position] = vector
    result = -2.0 * (totals - likelihood)
    result[moves.size_change != 0] = np.inf
    return result, sample_bounds


def best_improvement(bounds, canonical, materialize, score, current_score, *, signature,
                     admissible, reach=0.0, choose=None, observe=None):
    """Best-first exact scoring of the moves a bound cannot exclude.

    materialize(move) -> panel; score(panel, move) -> (exact join score -2 LL, payload);
    admissible(panel) is False for a move outside the searched set (a changed path
    left without sequence of its own). eps = 1e-9 (1 + |score|) is the float
    resolution allowed for the exact score and the bounds (measured disagreement
    <= 1.2e-10 relative). A move improves iff its exact change < -t, t = max(1e-8,
    eps); it is excluded iff its bound > -t + eps. Survivors are scored in increasing
    bound order while the next bound <= min(best admissible exact change + reach, -t) + eps
    (reach: how far above the best a tie class may extend, in score units; release 2.1 passes
    WIDE_CLASS_QUANTA tie quanta so every move that could share the best's tie class is scored).
    choose(improving entries) picks the move among the improving (default, base: the least exact
    change, ties by canonical order; release 2.1: tie classes of the change, the panel's content
    inside a class); identical panels are scored once and take the path order of their first move
    in canonical order. Entries are [change, canonical, move, panel, payload]. Returns
    ((change, canonical, move, panel, payload) or None, survivors, scored).
    observe(panel, exact_score, payload), when given, sees each uniquely exact-scored
    admissible panel, including flat neighbors. It does not widen the bounded search
    or make a non-improving move eligible for acceptance.
    """
    eps = 1e-9 * (1.0 + abs(float(current_score)))
    threshold = max(1e-8, eps)
    survivors = np.flatnonzero(bounds <= -threshold + eps)
    if not len(survivors):
        return None, 0, 0
    keys = canonical[survivors]
    order = survivors[np.lexsort(tuple(keys[:, column] for column in range(
        keys.shape[1] - 1, -1, -1)) + (bounds[survivors],))]
    seen = {}
    best_change = None
    scored = 0
    for move in order.tolist():
        limit = -threshold if best_change is None else min(-threshold, best_change + reach)
        if bounds[move] > limit + eps:
            break
        panel = materialize(move)
        key = signature(panel)
        canon = tuple(int(x) for x in canonical[move])
        # Admissibility depends on the changed-slot witness, not just the panel.
        if not admissible(panel):
            continue
        entry = seen.get(key)
        if entry is not None:
            # The same panel (same paths with multiplicity) from an earlier move
            # in canonical order: keep that move's description and path order.
            if canon < entry[1]:
                entry[1] = canon
                entry[2] = move
                if entry[3] is not None:
                    entry[3] = panel
            continue
        exact_score, payload = score(panel, move)
        scored += 1
        if observe is not None:
            observe(panel, float(exact_score), payload)
        change = float(exact_score) - float(current_score)
        improving = change < -threshold
        seen[key] = [change, canon, move, panel if improving else None,
                     payload if improving else None]
        if improving and (best_change is None or change < best_change):
            best_change = change
    improving = [entry for entry in seen.values() if entry[0] < -threshold]
    if not improving:
        return None, len(survivors), scored
    if choose is not None:
        return tuple(choose(improving)), len(survivors), scored
    return tuple(min(improving, key=lambda entry: (entry[0], entry[1]))), len(survivors), scored
