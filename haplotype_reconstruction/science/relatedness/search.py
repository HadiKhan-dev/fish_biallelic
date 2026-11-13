"""Fit full-sib families with both production searches.

Method: docs/methods/relatedness.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.concurrency import ThreadSource, thread_count
from .family_kernels import (
    family_loglik_parallel,
    full_sib_gains,
    half_screen,
    half_sib_gains,
    merge_gains,
    pair_indices,
    screen_all,
    screen_one,
    screen_shapes,
)
from .keys import _ranked, _top
from .parameters import FamilyParameters
from .partition import _Partition
from .support import _support

_DEFAULT_FAMILY = FamilyParameters()


@dataclass(frozen=True)
class FamilyFit:
    """The fitted families (sample indices of the run's axis).

    members: one int64 array per family (sorted). parents: int64 [F, 2] sequenced sample per parent slot, -1 latent.
    member_support: float64 [N] share of support units (``ranges``) on which the sample's family is its best placement
    (for a one-member family: on which staying alone beats joining any family; a latent one-member family is the
    sample alone, a tie, so it scores 0). parent_support: float64 [F, 2] share
    of support units on which the slot sample
    beats a latent slot (NaN for a latent slot). half_sib: int64 [H, 2] pairs of latent families sharing one latent
    parent better than not (gain above log N), half_sib_gain: float64 [H]. loglik: the objective. sweeps: search
    sweeps used; converged: whether the last sweep changed nothing. search: the search kept ("best_improvement" or
    "content_order"); search_margin: its objective minus the other search's.
    A slot sample's genotypes serve the model as a parent's would: it may be a parent, child (the two directions
    tie for a one-member family), sib or another close relative. Parent calls, direction and their support belong
    to the pedigree stage.
    """

    members: tuple[np.ndarray, ...]
    parents: np.ndarray
    member_support: np.ndarray
    parent_support: np.ndarray
    half_sib: np.ndarray
    half_sib_gain: np.ndarray
    loglik: float
    sweeps: int
    converged: bool
    search: str = "best_improvement"
    search_margin: float = float("nan")


class _PairGains:
    """Upper triangle in row order; no diagonal, lower triangle, or unused family slots."""

    def __init__(self, n):
        self.n = n
        rows = np.arange(n + 1, dtype=np.int64)
        self.offset = rows * (2 * n - rows - 1) // 2
        self.values = np.empty(n * (n - 1) // 2)

    def row(self, f):
        return self.values[self.offset[f] : self.offset[f + 1]]

    def get(self, f, g):
        return self.values[self.offset[f] + g - f - 1]

    def put(self, f, g, values):
        self.values[self.offset[f] + g - f - 1] = values

    def maximum(self, f):
        return self.row(f).max(initial=-np.inf)


def _agglomeration_entries(part: _Partition, first, second, gain, screen):
    """gain[first[t], second[t]] = the latent full-sib merge gain of the pair when positive and above its half-sib link
    gain, else -inf (pair_gains' comparison). The link is evaluated exactly only where its screen estimate is within the
    error bound of the full-sib gain (elsewhere the comparison is decided either way)."""
    first, second = np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)
    full = np.empty(len(first))
    full_sib_gains(part.S, part.ll, first, second, part.logpi, full)
    entry = np.full(len(first), -np.inf)
    positive = np.flatnonzero(full > 0)
    if len(positive):
        pi, y, Y, base = screen["pi"], screen["y"], screen["Y"], screen["half_base"]
        estimate = np.empty(len(positive))
        half_screen(part.S, part.logpi, pi, y, Y, base, first[positive], second[positive], estimate)
        value = full[positive]
        error = part.merge_error(first[positive], second[positive], estimate)
        above, below = value > estimate + error, value < estimate - error
        entry[positive[above]] = value[above]
        unsure = positive[~(above | below)]
        if len(unsure):
            half = np.empty(len(unsure))
            half_sib_gains(
                part.S,
                part.ll,
                np.ascontiguousarray(np.stack([first[unsure], second[unsure]], axis=1)),
                part.logpi,
                half,
            )
            entry[unsure] = np.where(full[unsure] > half, full[unsure], -np.inf)
    gain.put(first, second, entry)


def _agglomerate(part: _Partition):
    """Latent full-sib families from one-member families: repeatedly merge the pair with the largest full-sib gain,
    when positive and larger than its half-sib link gain.

    Only such gains are candidates, so the gain matrix keeps those (else -inf), and each row's maximum is kept and
    updated with the entries a merge changes. A screen (family_kernels.screen_all, screen_one: the full-sib gain in the
    linear domain, in float32) sets aside the pairs whose gain is below 0 by more than its error bound; only the
    others are evaluated exactly (_agglomeration_entries)."""
    N, M = part.N, part.M
    part.singletons()
    gain = _PairGains(N)
    W = np.ascontiguousarray((part.logpi[:, :, None] + part.logpi[:, None, :]).reshape(M, 9))
    W2 = W / 2
    m, A = np.empty((N, M)), np.empty((N, M, 9), dtype=np.float32)
    y, Y = np.empty((N, M)), np.empty((N, M, 3), dtype=np.float32)
    screen_shapes(part.S, part.everyone, W2, part.logpi, m, A, y, Y)
    full_base = m.sum(axis=1) - part.ll[:N]
    screen = dict(pi=np.exp(part.logpi), y=y, Y=Y, half_base=y.sum(axis=1) - part.ll[:N])
    # The screen writes directly into the packed table. Replace each bounded batch by exact gains (or -inf)
    # before the search starts; batching changes storage only, never arithmetic or candidate selection.
    thread_count(part.threads)
    screen_all(part.S, W, m, A, full_base, N, part.parameters.screen_tile, gain.values)
    for lo in range(0, len(gain.values), part.parameters.pair_batch):
        thread_count(part.threads)
        hi = min(len(gain.values), lo + part.parameters.pair_batch)
        pairs = pair_indices(N, lo, hi)
        first, second = pairs[:, 0], pairs[:, 1]
        estimate = gain.values[lo:hi]
        keep = estimate > -part.merge_error(first, second, estimate)
        first, second = first[keep], second[keep]
        gain.values[lo:hi] = -np.inf
        _agglomeration_entries(part, first, second, gain, screen)
    rowmax = np.asarray([gain.maximum(f) for f in range(N)])
    while True:
        maximum = rowmax.max()
        if maximum <= 0:
            break
        bound = maximum - part.parameters.gain_tolerance
        tied: list[tuple[int, int]] = []
        for r in np.flatnonzero(rowmax >= bound):
            row = gain.row(r)
            tied.extend((int(r), int(r + 1 + c)) for c in np.flatnonzero((row > 0) & (row >= bound)))
        f, g = tied[0] if len(tied) == 1 else max(tied, key=lambda pair: part.pair_key(*pair))
        thread_count(part.threads)
        part.merge(f, g, np.array([-1, -1]))
        # family g is gone: rows whose maximum was in column g are scanned again
        previous = np.arange(g, dtype=np.int64)
        column = gain.get(previous, g)
        gain.row(g)[:] = -np.inf
        gain.put(previous, g, -np.inf)
        recount = set(previous[(column == rowmax[:g]) & (column > -np.inf)].tolist()) | {f, g}
        others = np.asarray([h for h in part.live() if h != f], dtype=np.int64)
        if not len(others):
            break
        screen_shapes(part.S, np.asarray([f], dtype=np.int64), W2, part.logpi, m, A, y, Y)
        full_base[f] = m[f].sum() - part.ll[f]
        screen["half_base"][f] = y[f].sum() - part.ll[f]
        estimate = np.empty(len(others))
        screen_one(part.S, W, m, A, full_base, f, others, estimate)
        below = others[others < f]  # their entries (h, f) lie in column f
        before = gain.get(below, f)
        low, high = np.minimum(others, f), np.maximum(others, f)
        gain.put(low, high, -np.inf)
        candidates = estimate > -part.merge_error(low, high, estimate)
        _agglomeration_entries(part, low[candidates], high[candidates], gain, screen)
        after = gain.get(below, f)
        rise = after > rowmax[below]
        rowmax[below[rise]] = after[rise]
        recount |= set(below[(before == rowmax[below]) & (after < before) & (before > -np.inf)].tolist())
        for row_index in recount:
            rowmax[row_index] = gain.maximum(row_index)


def _offer_gain(part: _Partition, f: int, k: int, p: int, member: bool):
    """(gain, target) of filling family f's free slot k with sample p on the current partition: for a member, p
    first moves to its best other placement (target; -1 a new family). None when the offer no longer applies."""
    other = int(part.slots[f][1 - k])
    if member:
        if part.of[p] != f or len(part.members[f]) < 2:
            return None
        rest = part.member_parent(f)[p]
        out_gain, target = part.best_placement(p, exclude=(f,))
        return (rest - part.ll[f]) + (out_gain - part.leave_gain(p)), target
    if part.of[p] == f:
        return None
    kept = part._slot_logliks.get((int(f), int(part.version[f])))
    if kept is not None and not np.isnan(kept[p]):
        return kept[p] - part.ll[f], None
    return (
        family_loglik_parallel(part.S[f], part.S[f], 0.0, part.law(other), part.logpost[p], 0, part.M) - part.ll[f],
        None,
    )


def _assign_parents(part: _Partition, penalty: float, keep_versions=()) -> int:
    """One parent assignment per family with a free slot (best offers first): a sequenced non-member, or a member
    moved out to its best other placement first. Offers are ranked on the partition at the start of the sweep, and
    each is scored again on the current partition just before it is applied (earlier assignments change families),
    so an applied assignment raises the objective by more than ``penalty``; the parent graph must stay acyclic.
    Tied offers (gains within the tolerance, as a parent-offspring pair's two directions are) go to the offered
    parent with the larger content key, then the family with the larger key."""
    part.prune(keep_versions)
    live = part.live()
    open_families = [int(f) for f in live if -1 in part.slots[f]]
    movers = [p for f in open_families if len(part.members[f]) >= 2 for p in part.members[f]]
    part.sync_estimates(live)
    part.sync_leave(movers)
    part.prefetch_near_best(movers)
    offers = []
    for f in open_families:
        k = int(np.flatnonzero(part.slots[f] == -1)[0])
        other = int(part.slots[f][1 - k])
        allowed = np.ones(part.N, dtype=np.bool_)
        allowed[part.members[f]] = False
        if other >= 0:
            allowed[other] = False
        gains = np.where(allowed, part.slot_logliks(f, allowed), np.nan)
        gains -= part.ll[f]
        if np.isfinite(gains).any():
            p = _top(gains, part.evidence_rank, part.parameters.gain_tolerance)
            offers.append((float(gains[p]), int(f), k, p, False))
        if len(part.members[f]) >= 2:
            for p in part.members[f]:
                gain, _ = _offer_gain(part, int(f), k, p, True)
                offers.append((float(gain), int(f), k, p, True))
    family_key = {int(f): part.key(f) for f in part.live()}
    offers = _ranked(
        offers,
        tolerance=part.parameters.gain_tolerance,
        key=lambda offer: (part.evidence_rank[offer[3]], family_key[offer[1]]),
    )
    changed, used_f, used_p = 0, set(), set()
    for ranked, f, k, p, member in offers:
        if ranked <= penalty or f in used_f or p in used_p or not part.members[f]:
            continue
        if part.slots[f][k] != -1 or p in part.slots[f]:
            continue
        current = _offer_gain(part, f, k, p, member)
        if current is None or not current[0] > penalty:
            continue
        undo = None
        if member:
            undo = f
            part.move(p, current[1])
        part.slots[f][k] = p
        if not part.acyclic_with_parent(f, p):
            part.slots[f][k] = -1
            if undo is not None:
                part.move(p, undo)
            continue
        part.ll[f] = part.score(f)
        part.touch(f)
        used_f.add(f)
        used_p.add(p)
        changed += 1
    return changed


def _moves(part: _Partition) -> int:
    """Best-improvement single-sample moves: repeatedly the one move with the largest gain over all
    samples, each sample to its best valid placement (another family, or a new one-member family), at most N moves per
    sweep, until no move gains more than the tolerance. Ties go to content keys (_top), so the moves do not depend
    on sample names or input order.

    Every sample's leave gain is kept exactly, and its join gains as the screen's estimates (_Partition.estimate,
    within their error bound of the exact gains): a move changes only its two families, so only their rows are
    estimated again. The estimates bound each sample's best gain; the samples that may hold the largest gain (or tie
    with it), and their placements that may be their best (or tie with it), are evaluated exactly, and the move is
    chosen among them by the rule above, as from the exact gains of every sample and placement. The selected move is
    scored again on the current partition before it is applied, so each applied move raises the objective by more
    than the tolerance."""
    N = part.N
    everyone = part.everyone
    changed = 0
    for _ in range(N):
        thread_count(part.threads)
        live = part.live()
        part.sync_estimates(live)
        part.sync_leave(everyone)
        leave = part.leave
        estimate = part.estimate[live].T.copy()
        excluded = part.of[:, None] == live[None, :]
        for j, g in enumerate(live):
            if (part.slots[g] >= 0).any():
                excluded[list(part.blocked(g)), j] = True
        estimate[excluded] = -np.inf
        finite = estimate[np.isfinite(estimate)]
        # one bound for every entry: join_error at the largest magnitudes of this partition
        error = part.screen_error(
            2 * np.abs(part.ll[live]).max() + np.abs(part.solo).max() + (np.abs(finite).max() if len(finite) else 0.0)
        )
        low = np.maximum(part.solo, (estimate - error).max(axis=1))  # each sample's best placement is at least this,
        upper = leave + np.maximum(part.solo, (estimate + error).max(axis=1))  # its move gain at most this,
        lower = leave + low - part.parameters.gain_tolerance  # and at least this
        candidates = np.flatnonzero(
            (upper >= lower.max() - part.parameters.gain_tolerance) & (upper > part.parameters.gain_tolerance)
        )
        if not len(candidates):
            break
        rows, cols = np.nonzero(
            estimate[candidates] + error >= (low[candidates] - part.parameters.gain_tolerance)[:, None]
        )
        part.sync_pairs(live[cols], candidates[rows])
        gains = np.full((len(candidates), len(live)), -np.inf)
        gains[rows, cols] = part.join[live[cols], candidates[rows]]
        # Rank complete target keys once, retaining vectorised selection. Include the new singleton in the same tie
        # rule, and keep the selected target's actual gain (not the maximum of its tolerance band).
        family_keys = [part.key(g) for g in live]
        singleton_keys = part.singleton_keys
        key_rank = {key: i for i, key in enumerate(sorted(set(family_keys + singleton_keys)))}
        keys = np.asarray([key_rank[key] for key in family_keys])
        alone_keys = np.asarray([key_rank[singleton_keys[c]] for c in candidates])
        solo = part.solo[candidates]
        best = np.maximum(gains.max(axis=1), solo)
        tied = gains >= (best - part.parameters.gain_tolerance)[:, None]
        pick = np.where(tied, keys[None, :], -1).argmax(axis=1)
        family_rank = np.where(tied, keys[None, :], -1).max(axis=1)
        alone_rank = np.where(solo >= best - part.parameters.gain_tolerance, alone_keys, -1)
        target = np.where(family_rank > alone_rank, live[pick], -1)
        gain = leave[candidates] + np.where(target >= 0, gains[np.arange(len(candidates)), pick], solo)
        eligible = np.where(gain > part.parameters.gain_tolerance, gain, -np.inf)
        if not np.isfinite(eligible).any():
            break
        j = _top(eligible, part.evidence_rank[candidates], part.parameters.gain_tolerance)
        c, g = int(candidates[j]), int(target[j])
        # scored again on the current partition before it is applied
        if g < 0:
            fresh = part.leave_gain(c) + part.solo[c]
        else:
            u, v = part.slots[g]
            fresh = (
                part.leave_gain(c)
                + family_loglik_parallel(part.S[g], part.C[c], 1.0, part.law(u), part.law(v), 0, part.M)
                - part.ll[g]
            )
        if not fresh > part.parameters.gain_tolerance or (g >= 0 and c in part.blocked(g)):
            continue  # a stale score (not expected): every kept gain is current already
        part.move(c, g)
        changed += 1
    return changed


def _content_moves(part: _Partition) -> int:
    """First-improvement single-sample moves in content order (the second search): each sample once, by
    decreasing content key, moves to its best valid placement when that gains more than the tolerance. Exact join
    gains near the best are prepared for a batch of the coming samples at a time (and again for a sample whose
    candidate families changed meanwhile)."""
    changed = 0
    order = part.content_order
    part.sync_estimates(part.live())
    part.sync_leave(part.everyone)
    batch = part.parameters.prefetch_batch
    for i, c in enumerate(order):
        if i % batch == 0:
            thread_count(part.threads)
            part.prefetch_near_best(order[i : i + batch])
        gain, target = part.best_placement(int(c))
        if gain > part.parameters.gain_tolerance:
            f = int(part.of[c])
            h = part.move(int(c), target)
            changed += 1
            rows = [h] + ([f] if part.members[f] else [])
            part.sync_leave([m for r in rows for m in part.members[r]])
    return changed


def _sweeps(part: _Partition, moves, penalty: float, max_sweeps: int, keep_versions=()):
    """Sweeps of slot assignment, ``moves``, kept-slot splits and one merge; (sweeps, converged)."""
    sweeps, changed = 0, 1
    while changed and sweeps < max_sweeps:
        thread_count(part.threads)
        changed = _assign_parents(part, penalty, keep_versions) + moves(part) + _split_moves(part, penalty)
        changed += _merge_one(part)
        sweeps += 1
    return sweeps, not changed


def _split_moves(part: _Partition, penalty: float) -> int:
    """Leave a family with at least two members for a new singleton keeping one sequenced slot sample.

    Repeatedly select the largest gain strictly above log N, at most N moves per sweep. Gains within the
    tolerance tie by the member's evidence rank, then the kept sample's rank. Rescore the selected leave on the
    current partition before applying it; the singleton likelihood is partition-independent and kept for the fit.
    The kept edge already existed, so this move introduces no ancestry or cycle. The other slot stays latent
    until ordinary slot assignment in a later sweep.
    """
    changed = 0
    for _ in range(part.N):
        thread_count(part.threads)
        families = [int(f) for f in part.live() if len(part.members[f]) >= 2 and (part.slots[f] >= 0).any()]
        part.sync_leave([c for f in families for c in part.members[f]])
        offers = []
        for f in families:
            for c in part.members[f]:
                for u in part.slots[f]:
                    if u < 0:
                        continue
                    gain = float(part.leave[c] + part.singleton_slot_loglik(c, u))
                    if gain > penalty:
                        offers.append((gain, int(c), int(u)))
        if not offers:
            break
        j = _top(
            np.asarray([offer[0] for offer in offers]),
            [(part.evidence_rank[c], part.evidence_rank[u]) for _, c, u in offers],
            part.parameters.gain_tolerance,
        )
        _, c, u = offers[j]
        f = int(part.of[c])
        a, b = part.slots[f]
        fresh = (
            family_loglik_parallel(part.S[f], part.C[c], -1.0, part.law(a), part.law(b), 0, part.M)
            - part.ll[f]
            + part.singleton_slot_loglik(c, u)
        )
        if not fresh > penalty:
            continue
        h = part.move(c, -1)
        part.slots[h] = (u, -1)
        part.ll[h] = part.singleton_slot_loglik(c, u)
        part.touch(h)
        changed += 1
    return changed


def _merge_one(part: _Partition) -> int:
    """The best merge of two families whose sequenced parents are compatible (at most two, none a member); two latent
    families merge only when the full-sib gain beats their half-sib link. A pair's gains are kept for the two
    families' versions."""
    live = part.live()
    named_of = {int(f): {int(p) for p in part.slots[f] if p >= 0} for f in live}
    member_set = {int(f): set(part.members[f]) for f in live}
    row_list: list[tuple[int, int]] = []
    parent_list: list[list[int]] = []
    for a in range(len(live)):
        f = int(live[a])
        for b in range(a + 1, len(live)):
            g = int(live[b])
            union = named_of[f] | named_of[g]
            if len(union) > 2 or any(p in member_set[f] or p in member_set[g] for p in union):
                continue
            named = sorted(
                {int(p) for p in (*part.slots[f], *part.slots[g]) if p >= 0}, key=lambda p: -part.evidence_rank[p]
            )
            row_list.append((f, g))
            parent_list.append((named + [-1, -1])[:2])
    if not row_list:
        return 0
    rows, parents = np.asarray(row_list, dtype=np.int64), np.asarray(parent_list, dtype=np.int64)
    out = np.empty((len(rows), 2))
    keys = [(int(f), int(part.version[f]), int(g), int(part.version[g])) for f, g in rows]
    missing_list: list[int] = []
    for t, key in enumerate(keys):
        value = part._merge_pairs.get(key)
        if value is None:
            missing_list.append(t)
        else:
            out[t] = value
    if missing_list:
        thread_count(part.threads)
        missing = np.asarray(missing_list, dtype=np.int64)
        fresh = np.empty((len(missing), 2))
        merge_gains(
            part.S,
            part.ll,
            np.ascontiguousarray(rows[missing]),
            np.ascontiguousarray(parents[missing]),
            part.logpi,
            part.logpost,
            part.parameters.gain_tolerance,
            fresh,
        )
        out[missing] = fresh
        for t, value in zip(missing.tolist(), fresh, strict=False):
            part._merge_pairs[keys[t]] = value.copy()
    full = np.where(np.isnan(out[:, 1]) | (out[:, 0] > out[:, 1]), out[:, 0], -np.inf)
    offers = [(float(full[t]), t) for t in np.flatnonzero(full > part.parameters.gain_tolerance)]
    for _, t in _ranked(
        offers, tolerance=part.parameters.gain_tolerance, key=lambda offer: part.pair_key(*rows[offer[1]])
    ):
        f, g = int(rows[t, 0]), int(rows[t, 1])
        before = (part.slots[f].copy(), part.slots[g].copy())
        members_g = list(part.members[g])
        part.merge(f, g, parents[t])
        if part.acyclic():
            return 1
        # undo: g's members back in a family with g's old slots, f's slots restored
        h = part.new_family(members_g)
        for c in members_g:
            part.members[f].remove(c)
        part.own_sum(f)
        part.S[f] -= part.S[h]
        part.slots[f], part.slots[h] = before
        part.ll[f], part.ll[h] = part.score(f), part.score(h)
        part.touch(f)
        part.touch(h)
    return 0


def fit(
    C: np.ndarray,
    logpi: np.ndarray,
    logpost: np.ndarray,
    ranges: tuple[tuple[int, int], ...],
    *,
    max_sweeps: int,
    parameters: FamilyParameters = _DEFAULT_FAMILY,
    threads: ThreadSource = 1,
) -> FamilyFit:
    """Families over the samples of C [N, M, 9] (child terms), logpi [M, 3], logpost [N, M, 3]; ``ranges``: each
    support unit's site range (lo, hi). Two bounded production searches; screens use exact fallback gains."""
    # Partition sums borrow singleton rows before detaching on mutation. A readonly
    # task input needs an owned buffer so Numba's mutable sum list remains valid.
    C = C if C.flags.writeable else C.copy()
    logpi = logpi if logpi.flags.writeable else logpi.copy()
    logpost = logpost if logpost.flags.writeable else logpost.copy()
    part = _Partition(C, logpi, logpost, parameters, threads)
    penalty = float(np.log(part.N))
    _agglomerate(part)
    # Two order-free searches from the agglomeration, and the higher objective is kept (the first on a
    # tie): the objective has many local optima, and neither search reaches the better one on every design.
    start = part.snapshot(keep=True)
    start_versions = [(f, int(start[4]["version"][f])) for f in start[4]["S"]]
    sweeps, converged = _sweeps(part, _moves, penalty, max_sweeps, start_versions)
    first, first_total = (part.snapshot(), sweeps, converged), part.total()
    part.restore(start)
    del start
    sweeps, converged = _sweeps(part, _content_moves, penalty, max_sweeps)
    second_total = part.total()
    if second_total > first_total + part.parameters.gain_tolerance:
        search, margin = "content_order", second_total - first_total
    else:
        state, sweeps, converged = first
        part.restore(state)
        search, margin = "best_improvement", first_total - second_total
    member_support, parent_hits = _support(part, ranges)
    live = part.live()
    parents = part.slots[live].copy()
    parent_support = np.where(parents >= 0, parent_hits[live], np.nan)
    latent = np.asarray([j for j, f in enumerate(live) if (part.slots[f] < 0).all()], dtype=np.int64)
    a, b = np.triu_indices(len(latent), k=1)
    pairs = np.stack([latent[a], latent[b]], axis=1).astype(np.int64).reshape(-1, 2)
    gains = np.empty(len(pairs))
    if len(pairs):
        thread_count(part.threads)
        half_sib_gains(part.S, part.ll, np.ascontiguousarray(live[pairs]), part.logpi, gains)
    linked = gains > penalty
    return FamilyFit(
        members=tuple(np.asarray(sorted(part.members[f]), dtype=np.int64) for f in live),
        parents=parents,
        member_support=member_support,
        parent_support=parent_support,
        half_sib=pairs[linked].reshape(-1, 2),
        half_sib_gain=np.asarray(gains[linked], dtype=np.float64),
        loglik=part.total(),
        sweeps=sweeps,
        converged=converged,
        search=search,
        search_margin=float(margin),
    )


__all__ = ["FamilyFit", "fit"]
