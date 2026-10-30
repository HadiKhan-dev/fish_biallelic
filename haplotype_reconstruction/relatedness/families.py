"""Full-sib families with latent or sequenced parents, from genotype likelihoods alone.

A family is a set of samples (its members, full sibs) and two parent slots, each latent (Hardy-Weinberg at the sample
allele frequency) or a sequenced sample p. Its composite log-likelihood over the thinned sites is

    L_F = sum_i log sum_{g_u, g_v} P_u(i, g_u) P_v(i, g_v) prod_{c in F} sum_g T(g | g_u, g_v) GL_c(i, g),

with P = the Hardy-Weinberg prior for a latent slot and p's genotype posterior from its own reads for a sequenced
slot. The objective is the sum over families. Each sample's reads enter once, as a member of its own family; a family
with a sequenced parent conditions on that parent's reads (a chain-rule factor P(children | parent's reads)), which is
a valid factorisation only along an acyclic parent graph, so no assignment may make a sample its own ancestor. The
approximation drops what a sequenced parent's own family says about its genotype.

Half-sib links: two latent families may instead share one latent parent; a full-sib merge is made only when it beats
both separation and that link (the link is recorded, the families stay apart).

Search (deterministic): an agglomerative latent partition, then sweeps of (a) parent assignment (a sequenced sample, or
a member moved out first, in a free slot; gain above log N, the choice among N samples), (b) single-sample moves, (c)
one merge of families with compatible parents. Support: on each map unit alone (a short chromosome, or one without a
genetic map, is one unit), whether a member's family predicts its reads better than any other placement (its held-out-
child log-likelihood ratio), and whether a sequenced parent beats a latent one. The half-sib comparison guards the
agglomeration and the merges, not single-sample moves, so the search is no joint optimiser over shared latent parents.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernels import family_loglik, family_logliks, half_sib_loglik, join_gains, pair_gains, slot_gains

_GAIN_TOLERANCE = 1e-6     # moves and merges must improve the objective by more than this (log units)


@dataclass(frozen=True)
class FamilyFit:
    """The fitted families (sample indices of the run's axis).

    members: one int64 array per family (sorted). parents: int64 [F, 2] sequenced parent per slot, -1 latent.
    member_support: float64 [N] share of support units (``ranges``) on which the sample's family is its best placement
    (for a one-member family: on which staying alone beats joining any family). parent_support: float64 [F, 2] share
    of support units on which the sequenced parent
    beats a latent one (NaN for a latent slot). half_sib: int64 [H, 2] pairs of latent families sharing one latent
    parent better than not (gain above log N), half_sib_gain: float64 [H]. loglik: the objective. sweeps: search
    sweeps used; converged: whether the last sweep changed nothing.
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


class _Partition:
    """Mutable families over the run's samples (family indices are reused once a family empties)."""

    def __init__(self, C, logpi, logpost):
        self.C, self.logpi, self.logpost = C, logpi, logpost
        self.N, self.M = C.shape[0], C.shape[1]
        capacity = 2 * self.N + 2
        self.S = np.zeros((capacity, self.M, 9))
        self.ll = np.zeros(capacity)
        self.slots = np.full((capacity, 2), -1, dtype=np.int64)
        self.members = [[] for _ in range(capacity)]
        self.of = np.full(self.N, -1, dtype=np.int64)
        self.free = list(range(capacity - 1, -1, -1))

    def law(self, p):
        return self.logpi if p < 0 else self.logpost[p]

    def score(self, f):
        u, v = self.slots[f]
        return family_loglik(self.S[f], self.S[f], 0.0, self.law(u), self.law(v), 0, self.M)

    def new_family(self, members):
        f = self.free.pop()
        self.members[f] = sorted(members)
        self.slots[f] = -1
        self.S[f] = self.C[self.members[f]].sum(axis=0)
        for c in members:
            self.of[c] = f
        self.ll[f] = self.score(f)
        return f

    def _release(self, f):
        self.members[f] = []
        self.slots[f] = -1
        self.S[f] = 0.0
        self.ll[f] = 0.0
        self.free.append(f)

    def live(self):
        return np.asarray([f for f in range(len(self.members)) if self.members[f]], dtype=np.int64)

    def move(self, c, g):
        """Move sample c out of its family into family g (g < 0: a new one-member family); returns c's family."""
        f = int(self.of[c])
        self.members[f].remove(c)
        if self.members[f]:
            self.S[f] -= self.C[c]
            self.ll[f] = self.score(f)
        else:
            self._release(f)
        if g < 0:
            return self.new_family([c])
        self.members[g].append(c)
        self.members[g].sort()
        self.S[g] += self.C[c]
        self.of[c] = g
        self.ll[g] = self.score(g)
        return g

    def merge(self, f, g, parents):
        for c in self.members[g]:
            self.of[c] = f
        self.members[f] = sorted(self.members[f] + self.members[g])
        self.S[f] += self.S[g]
        self._release(g)
        self.slots[f] = parents
        self.ll[f] = self.score(f)

    def total(self):
        return float(sum(self.ll[f] for f in self.live()))

    def ancestors(self, x):
        """Samples that are ancestors of x through sequenced parent slots."""
        seen, stack = set(), [x]
        while stack:
            for p in self.slots[self.of[stack.pop()]]:
                if p >= 0 and p not in seen:
                    seen.add(int(p))
                    stack.append(int(p))
        return seen

    def acyclic(self):
        """Whether no sample is its own ancestor through the sequenced parent slots."""
        return all(x not in self.ancestors(x) for x in range(self.N))

    def leave_gain(self, c):
        """Log-likelihood change of c's family when c leaves it."""
        f = self.of[c]
        if len(self.members[f]) == 1:
            return -self.ll[f]
        u, v = self.slots[f]
        return family_loglik(self.S[f], self.C[c], -1.0, self.law(u), self.law(v), 0, self.M) - self.ll[f]

    def alone(self, c, lo=0, hi=None):
        hi = self.M if hi is None else hi
        return family_loglik(self.C[c], self.C[c], 0.0, self.logpi, self.logpi, lo, hi)

    def placements(self, c, exclude=(), lo=0, hi=None, base=None):
        """(families, gains): c's join gain into every live family other than its own and ``exclude`` and those that
        have c as a parent (over sites [lo, hi); base = those families' log-likelihoods over the same sites)."""
        hi = self.M if hi is None else hi
        live = self.live()
        keep = [j for j, f in enumerate(live) if f != self.of[c] and f not in exclude and c not in self.slots[f]]
        families = live[keep]
        if not len(families):
            return families, np.zeros(0)
        base = self.ll[families] if base is None else base[keep]
        out = np.empty(len(families))
        join_gains(self.S, np.ascontiguousarray(base), self.slots, families, self.logpi, self.logpost, self.C[c],
                   1.0, lo, hi, out)
        return families, out

    def best_placement(self, c, exclude=()):
        """(gain, target) of moving c out of its family to its best valid other family or a new one (target -1): the
        change of the objective. A target whose sequenced parents descend from c is not valid."""
        leave = self.leave_gain(c)
        families, gains = self.placements(c, exclude)
        best = (leave + self.alone(c), -1)
        for j in np.argsort(-gains, kind="stable"):
            if leave + gains[j] <= best[0]:
                break
            g = int(families[j])
            descendants = [p for p in self.slots[g] if p >= 0 and (p == c or c in self.ancestors(int(p)))]
            if not descendants:
                best = (leave + gains[j], g)
                break
        return best


def _agglomerate(part: _Partition, log=print):
    """Latent full-sib families from one-member families: repeatedly merge the pair with the largest full-sib gain,
    when positive and larger than its half-sib link gain."""
    N = part.N
    for c in range(N):
        part.new_family([c])
    rows = np.asarray([(f, g) for f in range(N) for g in range(f + 1, N)], dtype=np.int64).reshape(-1, 2)
    out = np.empty((len(rows), 2))
    pair_gains(part.S, part.ll, rows, np.full((len(rows), 2), -1, dtype=np.int64), part.logpi, part.logpost, out)
    gain = np.full((2 * N + 2, 2 * N + 2), -np.inf)
    full, half = out[:, 0], out[:, 1]
    gain[rows[:, 0], rows[:, 1]] = np.where(full > half, full, -np.inf)
    while True:
        f, g = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[f, g] <= 0:
            break
        part.merge(f, g, np.array([-1, -1]))
        gain[g, :] = gain[:, g] = -np.inf
        others = np.asarray([h for h in part.live() if h != f], dtype=np.int64)
        if not len(others):
            break
        rows = np.stack([np.minimum(others, f), np.maximum(others, f)], axis=1)
        out = np.empty((len(rows), 2))
        pair_gains(part.S, part.ll, rows, np.full((len(rows), 2), -1, dtype=np.int64), part.logpi, part.logpost, out)
        gain[rows[:, 0], rows[:, 1]] = np.where(out[:, 0] > out[:, 1], out[:, 0], -np.inf)
    log(f"  agglomeration: {len(part.live())} families, objective {part.total():.1f}")


def _offer_gain(part: _Partition, f: int, k: int, p: int, member: bool):
    """(gain, target) of filling family f's free slot k with sample p on the current partition: for a member, p
    first moves to its best other placement (target; -1 a new family). None when the offer no longer applies."""
    other = int(part.slots[f][1 - k])
    if member:
        if part.of[p] != f or len(part.members[f]) < 2:
            return None
        rest = family_loglik(part.S[f], part.C[p], -1.0, part.law(other), part.logpost[p], 0, part.M)
        out_gain, target = part.best_placement(p, exclude=(f,))
        return (rest - part.ll[f]) + (out_gain - part.leave_gain(p)), target
    if part.of[p] == f:
        return None
    return family_loglik(part.S[f], part.S[f], 0.0, part.law(other), part.logpost[p], 0, part.M) - part.ll[f], None


def _assign_parents(part: _Partition, penalty: float) -> int:
    """One parent assignment per family with a free slot (best offers first): a sequenced non-member, or a member
    moved out to its best other placement first. Offers are ranked on the partition at the start of the sweep, and
    each is scored again on the current partition just before it is applied (earlier assignments change families),
    so an applied assignment raises the objective by more than ``penalty``; the parent graph must stay acyclic."""
    offers = []
    for f in part.live():
        if -1 not in part.slots[f]:
            continue
        k = int(np.flatnonzero(part.slots[f] == -1)[0])
        other = int(part.slots[f][1 - k])
        allowed = np.ones(part.N, dtype=np.bool_)
        allowed[part.members[f]] = False
        if other >= 0:
            allowed[other] = False
        gains = np.empty(part.N)
        slot_gains(part.S[f], part.law(other), part.logpost, allowed, gains)
        gains -= part.ll[f]
        if np.isfinite(gains).any():
            p = int(np.nanargmax(gains))
            offers.append((float(gains[p]), int(f), k, p, False))
        if len(part.members[f]) >= 2:
            for p in part.members[f]:
                gain, _ = _offer_gain(part, int(f), k, p, True)
                offers.append((float(gain), int(f), k, p, True))
    offers.sort(key=lambda offer: (-offer[0], offer[1], offer[3]))
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
        if not part.acyclic():
            part.slots[f][k] = -1
            if undo is not None:
                part.move(p, undo)
            continue
        part.ll[f] = part.score(f)
        used_f.add(f)
        used_p.add(p)
        changed += 1
    return changed


def _moves(part: _Partition) -> int:
    changed = 0
    for c in range(part.N):
        gain, target = part.best_placement(c)
        if gain > _GAIN_TOLERANCE:
            part.move(c, target)
            changed += 1
    return changed


def _merge_one(part: _Partition) -> int:
    """The best merge of two families whose sequenced parents are compatible (at most two, none a member); two latent
    families merge only when the full-sib gain beats their half-sib link."""
    live = part.live()
    rows, parents = [], []
    for a in range(len(live)):
        for b in range(a + 1, len(live)):
            f, g = int(live[a]), int(live[b])
            named = sorted({int(p) for p in (*part.slots[f], *part.slots[g]) if p >= 0})
            if len(named) > 2 or set(named) & set(part.members[f] + part.members[g]):
                continue
            rows.append((f, g))
            parents.append((named + [-1, -1])[:2])
    if not rows:
        return 0
    rows, parents = np.asarray(rows, dtype=np.int64), np.asarray(parents, dtype=np.int64)
    out = np.empty((len(rows), 2))
    pair_gains(part.S, part.ll, rows, parents, part.logpi, part.logpost, out)
    full = np.where(np.isnan(out[:, 1]) | (out[:, 0] > out[:, 1]), out[:, 0], -np.inf)
    for t in np.argsort(-full, kind="stable"):
        if not full[t] > _GAIN_TOLERANCE:
            return 0
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
        part.S[f] -= part.S[h]
        part.slots[f], part.slots[h] = before
        part.ll[f], part.ll[h] = part.score(f), part.score(h)
    return 0


def _support(part: _Partition, ranges):
    """Per chromosome (with family sites): whether each member's family is its best placement, and whether each
    sequenced parent beats a latent one."""
    ranges = [(lo, hi) for lo, hi in ranges if hi > lo]
    live = part.live()
    hits = np.zeros(part.N)
    parent_hits = np.zeros((len(part.members), 2))
    for lo, hi in ranges:
        llk = np.empty(len(live))
        family_logliks(part.S, part.slots, live, part.logpi, part.logpost, lo, hi, llk)
        base = np.full(len(part.members), np.nan)
        base[live] = llk
        for f in live:
            u, v = part.slots[f]
            for c in part.members[f]:
                rest = (family_loglik(part.S[f], part.C[c], -1.0, part.law(u), part.law(v), lo, hi)
                        if len(part.members[f]) > 1 else 0.0)
                stay = base[f] - rest
                _, gains = part.placements(c, lo=lo, hi=hi, base=base[part.live()])
                elsewhere = max(gains.max() if len(gains) else -np.inf, part.alone(c, lo, hi))
                hits[c] += stay > elsewhere
            for j, p in enumerate((u, v)):
                if p >= 0:
                    latent = family_loglik(part.S[f], part.S[f], 0.0, part.logpi if j == 0 else part.law(u),
                                           part.logpi if j == 1 else part.law(v), lo, hi)
                    parent_hits[f, j] += base[f] > latent
    return hits / max(len(ranges), 1), parent_hits / max(len(ranges), 1)


def fit(C, logpi, logpost, ranges, *, max_sweeps: int, log=print) -> FamilyFit:
    """Families over the samples of C [N, M, 9] (child terms), logpi [M, 3], logpost [N, M, 3]; ``ranges``: each
    support unit's site range (lo, hi) (relatedness.stage.support_units)."""
    part = _Partition(C, logpi, logpost)
    penalty = float(np.log(part.N))
    _agglomerate(part, log)
    sweeps, changed = 0, 1
    while changed and sweeps < max_sweeps:
        changed = _assign_parents(part, penalty) + _moves(part) + _merge_one(part)
        sweeps += 1
        named = sum(int(p >= 0) for f in part.live() for p in part.slots[f])
        log(f"  sweep {sweeps}: {changed} changes, {len(part.live())} families, {named} sequenced parent slots, "
            f"objective {part.total():.1f}")
    member_support, parent_hits = _support(part, ranges)
    live = part.live()
    parents = part.slots[live].copy()
    parent_support = np.where(parents >= 0, parent_hits[live], np.nan)
    half, gains = [], []
    latent = [j for j, f in enumerate(live) if (part.slots[f] < 0).all()]
    for a in range(len(latent)):
        for b in range(a + 1, len(latent)):
            f, g = live[latent[a]], live[latent[b]]
            gain = half_sib_loglik(part.S[f], part.S[g], logpi, 0, part.M) - part.ll[f] - part.ll[g]
            if gain > penalty:
                half.append((latent[a], latent[b]))
                gains.append(gain)
    return FamilyFit(members=tuple(np.asarray(part.members[f], dtype=np.int64) for f in live), parents=parents,
                     member_support=member_support, parent_support=parent_support,
                     half_sib=np.asarray(half, dtype=np.int64).reshape(-1, 2),
                     half_sib_gain=np.asarray(gains, dtype=np.float64), loglik=part.total(), sweeps=sweeps,
                     converged=not changed)
