"""Reference implementation of relatedness.families: the release 3.4 search before the relatedness speed-up
(REL v3), kept unchanged for validation (``families.fit(..., reference=True)``). The model, search, tie rules and
outputs are those of relatedness.families; see its docstring and docs/methods/relatedness.md.

Full-sib families with latent or sequenced parents, from genotype likelihoods alone.

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

Search (deterministic, and independent of sample names and input order): an agglomerative latent partition, then
sweeps of (a) parent assignment (a sequenced sample, or a member moved out first, in a free slot; gain above log N, the
choice among N samples), (b) single-sample moves, (c) one merge of families with compatible parents. Release 3.4: the
sweeps run twice from the agglomeration, with best-improvement moves (always the move with the largest gain) and with
first-improvement moves in content order (each sample once, by decreasing content key), and the fit with the higher
objective is kept (the best-improvement one on a tie). Gains within the tolerance tie and go to content keys (a
sample's retained child-term vector), never to sample order. Support: on each map unit alone (a short chromosome, or one
without a genetic map, is one unit), whether a member's family predicts its reads better than any other placement (its
held-out-child log-likelihood ratio), and whether a sequenced parent beats a latent one. The half-sib comparison
guards the agglomeration and the merges, not single-sample moves, so the search is no joint optimiser over shared
latent parents.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key

import numpy as np

from .kernels import family_loglik, family_logliks, half_sib_loglik, join_gains, pair_gains, sample_gains, slot_gains

_GAIN_TOLERANCE = 1e-6     # moves and merges must improve the objective by more than this (log units); gains of
                           # competing moves or offers within it of each other tie (decided by content keys)


def _evidence_ranks(C):
    """Exact lexicographic order of the sample's own retained child-term vector, computed once per fit.

    Terms depend only on that sample's normalised GLs and observed mask, never the cohort allele frequency.
    Compare numeric values in site/parent-genotype order, in bounded chunks to stop at the first difference.
    Unlike a digest order, this does not scramble unrelated samples when an upstream fit changes a last bit.
    Equal vectors share a rank: identities within that evidence class are interchangeable, so outputs are
    defined up to their permutation. No name or axis index is appended, and no likelihood tolerance or
    rounding enters the evidence comparison. Canonical member order also fixes restoration's reduction order.
    """
    vectors = C.reshape(len(C), -1)

    def compare(a, b):
        for lo in range(0, vectors.shape[1], 256):
            left, right = vectors[a, lo:lo + 256], vectors[b, lo:lo + 256]
            different = np.flatnonzero(left != right)
            if len(different):
                j = different[0]
                return 1 if left[j] > right[j] else -1
        return 0

    order = sorted(range(len(C)), key=cmp_to_key(compare))
    ranks = np.empty(len(C), dtype=np.int64)
    rank = 0
    for j, c in enumerate(order):
        if j and compare(order[j - 1], c):
            rank += 1
        ranks[c] = rank
    return ranks


@dataclass(frozen=True)
class FamilyFit:
    """The fitted families (sample indices of the run's axis).

    members: one int64 array per family (sorted). parents: int64 [F, 2] sequenced parent per slot, -1 latent.
    member_support: float64 [N] share of support units (``ranges``) on which the sample's family is its best placement
    (for a one-member family: on which staying alone beats joining any family; a latent one-member family is the
    sample alone, a tie, so it scores 0). parent_support: float64 [F, 2] share
    of support units on which the sequenced parent
    beats a latent one (NaN for a latent slot). half_sib: int64 [H, 2] pairs of latent families sharing one latent
    parent better than not (gain above log N), half_sib_gain: float64 [H]. loglik: the objective. sweeps: search
    sweeps used; converged: whether the last sweep changed nothing. search: the search kept ("best_improvement" or
    "content_order"); search_margin: its objective minus the other search's.
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
        self.evidence_rank = _evidence_ranks(C)
        self.content_order = np.argsort(-self.evidence_rank, kind="stable")
        # Numerical gain of forming a singleton; deliberately separate from the evidence order.
        self.solo = np.asarray([family_loglik(C[c], C[c], 0.0, logpi, logpi, 0, self.M) for c in range(self.N)])

    def law(self, p):
        return self.logpi if p < 0 else self.logpost[p]

    def score(self, f):
        u, v = self.slots[f]
        return family_loglik(self.S[f], self.S[f], 0.0, self.law(u), self.law(v), 0, self.M)

    def new_family(self, members):
        f = self.free.pop()
        self.members[f] = sorted(members, key=lambda c: -self.evidence_rank[c])
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
        self.members[g].sort(key=lambda c: -self.evidence_rank[c])
        self.S[g] += self.C[c]
        self.of[c] = g
        self.ll[g] = self.score(g)
        return g

    def merge(self, f, g, parents):
        for c in self.members[g]:
            self.of[c] = f
        self.members[f] = sorted(self.members[f] + self.members[g], key=lambda c: -self.evidence_rank[c])
        self.S[f] += self.S[g]
        self._release(g)
        self.slots[f] = parents
        self.ll[f] = self.score(f)

    def total(self):
        return float(sum(self.ll[f] for f in sorted(self.live(), key=self.key)))

    def ancestors(self, x):
        """Samples that are ancestors of x through sequenced parent slots."""
        seen, stack = set(), [x]
        while stack:
            for p in self.slots[self.of[stack.pop()]]:
                if p >= 0 and p not in seen:
                    seen.add(int(p))
                    stack.append(int(p))
        return seen

    def key(self, f):
        """Complete member evidence multiset, then parent evidence (latent = -1)."""
        return (tuple(sorted((int(self.evidence_rank[c]) for c in self.members[f]), reverse=True)),
                tuple(sorted((int(self.evidence_rank[p]) if p >= 0 else -1 for p in self.slots[f]), reverse=True)))

    def pair_key(self, f, g):
        """Unordered pair of complete family keys for a merge."""
        return tuple(sorted((self.key(f), self.key(g)), reverse=True))

    def blocked(self, g):
        """Samples that may not join family g: its sequenced parents and their ancestors (none may become its own
        ancestor)."""
        out = set()
        for p in self.slots[g]:
            if p >= 0:
                out.add(int(p))
                out |= self.ancestors(int(p))
        return out

    def snapshot(self):
        """The families (members, parent slots, free indices), to be restored later."""
        return ([list(m) for m in self.members], self.slots.copy(), self.of.copy(), list(self.free))

    def restore(self, state):
        """Back to a snapshot: the family sums and log-likelihoods are computed again from the members."""
        for f in self.live():
            self.S[f] = 0.0
            self.ll[f] = 0.0
        members, slots, of, free = state
        self.members, self.slots, self.of, self.free = [list(m) for m in members], slots.copy(), of.copy(), list(free)
        for f in self.live():
            self.S[f] = self.C[self.members[f]].sum(axis=0)
            self.ll[f] = self.score(f)

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
        valid = [j for j, g in enumerate(families) if c not in self.blocked(int(g))]
        targets = [-1] + [int(families[j]) for j in valid]
        values = np.asarray([leave + self.solo[c]] + [leave + gains[j] for j in valid])
        keys = [((int(self.evidence_rank[c]),), (-1, -1))] + [self.key(g) for g in targets[1:]]
        j = _top(values, keys)
        return float(values[j]), targets[j]


def _agglomerate(part: _Partition, log=print):
    """Latent full-sib families from one-member families: repeatedly merge the pair with the largest full-sib gain,
    when positive and larger than its half-sib link gain."""
    N = part.N
    for c in part.content_order:
        part.new_family([int(c)])
    rows = np.asarray([(f, g) for f in range(N) for g in range(f + 1, N)], dtype=np.int64).reshape(-1, 2)
    out = np.empty((len(rows), 2))
    pair_gains(part.S, part.ll, rows, np.full((len(rows), 2), -1, dtype=np.int64), part.logpi, part.logpost, out)
    gain = np.full((2 * N + 2, 2 * N + 2), -np.inf)
    full, half = out[:, 0], out[:, 1]
    gain[rows[:, 0], rows[:, 1]] = np.where(full > half, full, -np.inf)
    while True:
        maximum = gain.max()
        if maximum <= 0:
            break
        tied = np.argwhere((gain > 0) & (gain >= maximum - _GAIN_TOLERANCE))
        f, g = max(tied, key=lambda pair: part.pair_key(*pair))
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


def _top(gains, keys) -> int:
    """Index of the largest gain (NaN: not a candidate). Gains within the tolerance of it tie, and the tie goes to the
    largest content key, so the choice follows the data, not sample names or input order."""
    tied = np.flatnonzero(gains >= np.nanmax(gains) - _GAIN_TOLERANCE)
    return int(max(tied, key=lambda j: keys[j]))


def _ranked(offers, key):
    """``offers`` (gain first) in decreasing gain. Gains within the tolerance of a run's first gain tie, and are
    ordered by ``key`` (content), not by sample or family index."""
    offers = sorted(offers, key=lambda offer: -offer[0])
    out, start = [], 0
    while start < len(offers):
        stop = start + 1
        while stop < len(offers) and offers[stop][0] >= offers[start][0] - _GAIN_TOLERANCE:
            stop += 1
        out.extend(sorted(offers[start:stop], key=key, reverse=True))
        start = stop
    return out


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
    so an applied assignment raises the objective by more than ``penalty``; the parent graph must stay acyclic.
    Tied offers (gains within the tolerance, as a parent-offspring pair's two directions are) go to the offered
    parent with the larger content key, then the family with the larger key."""
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
            p = _top(gains, part.evidence_rank)
            offers.append((float(gains[p]), int(f), k, p, False))
        if len(part.members[f]) >= 2:
            for p in part.members[f]:
                gain, _ = _offer_gain(part, int(f), k, p, True)
                offers.append((float(gain), int(f), k, p, True))
    family_key = {int(f): part.key(f) for f in part.live()}
    offers = _ranked(offers, key=lambda offer: (part.evidence_rank[offer[3]], family_key[offer[1]]))
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
    """Best-improvement single-sample moves (release 3.4): repeatedly the one move with the largest gain over all
    samples, each sample to its best valid placement (another family, or a new one-member family), at most N moves per
    sweep, until no move gains more than the tolerance. Ties go to content keys (``_top``), so the moves do not depend
    on sample names or input order.

    Every sample's gain on joining every live family, and on leaving its own, is kept and updated incrementally: a
    move changes only its two families, so only their columns (every sample joining them) and their members' leave
    gains are scored again. A sweep costs O(N F M) to start (one first-improvement sweep) and O(N M) per move. The
    selected move is scored again on the current partition before it is applied, so each applied move raises the
    objective by more than the tolerance."""
    N = part.N
    everyone = np.arange(N, dtype=np.int64)
    into = np.full((len(part.members), N), -np.inf)    # into[g, c]: c joining family g, without leaving its own
    leave = np.empty(N)                                # leave[c]: c's family's change when c leaves it

    def rescore(f):
        u, v = part.slots[f]
        sample_gains(part.S[f], part.C, everyone, 1.0, part.law(u), part.law(v), part.ll[f], into[f])
        members = np.asarray(part.members[f], dtype=np.int64)
        if len(members) == 1:
            leave[members] = -part.ll[f]
        else:
            out = np.empty(len(members))
            sample_gains(part.S[f], part.C, members, -1.0, part.law(u), part.law(v), part.ll[f], out)
            leave[members] = out

    for f in part.live():
        rescore(f)
    changed = 0
    for _ in range(N):
        live = part.live()
        gains = into[live].T.copy()
        gains[part.of[:, None] == live[None, :]] = -np.inf
        for j, g in enumerate(live):
            if (part.slots[g] >= 0).any():
                gains[list(part.blocked(g)), j] = -np.inf
        # Rank complete target keys once, retaining vectorised N x F selection. Include the new singleton in
        # the same tie rule, and keep the selected target's actual gain (not the maximum of its tolerance band).
        family_keys = [part.key(g) for g in live]
        singleton_keys = [((int(rank),), (-1, -1)) for rank in part.evidence_rank]
        key_rank = {key: i for i, key in enumerate(sorted(set(family_keys + singleton_keys)))}
        keys = np.asarray([key_rank[key] for key in family_keys])
        alone_keys = np.asarray([key_rank[key] for key in singleton_keys])
        best = np.maximum(gains.max(axis=1), part.solo)
        tied = gains >= (best - _GAIN_TOLERANCE)[:, None]
        pick = np.where(tied, keys[None, :], -1).argmax(axis=1)
        family_rank = np.where(tied, keys[None, :], -1).max(axis=1)
        alone_rank = np.where(part.solo >= best - _GAIN_TOLERANCE, alone_keys, -1)
        target = np.where(family_rank > alone_rank, live[pick], -1)
        gain = leave + np.where(target >= 0, gains[np.arange(N), pick], part.solo)
        eligible = np.where(gain > _GAIN_TOLERANCE, gain, -np.inf)
        if not np.isfinite(eligible).any():
            break
        c = _top(eligible, part.evidence_rank)
        f, g = int(part.of[c]), int(target[c])
        # scored again on the current partition before it is applied
        if g < 0:
            fresh = part.leave_gain(c) + part.solo[c]
        else:
            u, v = part.slots[g]
            fresh = part.leave_gain(c) + family_loglik(part.S[g], part.C[c], 1.0, part.law(u), part.law(v), 0,
                                                       part.M) - part.ll[g]
        if not fresh > _GAIN_TOLERANCE or (g >= 0 and c in part.blocked(g)):
            for h in live:      # a stale score (not expected): score every family again
                rescore(h)
            continue
        h = part.move(c, g)
        if part.members[f]:
            rescore(f)
        rescore(h)
        changed += 1
    return changed


def _content_moves(part: _Partition) -> int:
    """First-improvement single-sample moves in content order (release 3.4, the second search): each sample once, by
    decreasing content key, moves to its best valid placement when that gains more than the tolerance."""
    changed = 0
    for c in part.content_order:
        gain, target = part.best_placement(int(c))
        if gain > _GAIN_TOLERANCE:
            part.move(int(c), target)
            changed += 1
    return changed


def _sweeps(part: _Partition, moves, penalty: float, max_sweeps: int, log):
    """Sweeps of parent assignment, ``moves`` and one merge until a sweep changes nothing; (sweeps, converged)."""
    sweeps, changed = 0, 1
    while changed and sweeps < max_sweeps:
        changed = _assign_parents(part, penalty) + moves(part) + _merge_one(part)
        sweeps += 1
        named = sum(int(p >= 0) for f in part.live() for p in part.slots[f])
        log(f"  sweep {sweeps}: {changed} changes, {len(part.live())} families, {named} sequenced parent slots, "
            f"objective {part.total():.1f}")
    return sweeps, not changed


def _merge_one(part: _Partition) -> int:
    """The best merge of two families whose sequenced parents are compatible (at most two, none a member); two latent
    families merge only when the full-sib gain beats their half-sib link."""
    live = part.live()
    rows, parents = [], []
    for a in range(len(live)):
        for b in range(a + 1, len(live)):
            f, g = int(live[a]), int(live[b])
            named = sorted({int(p) for p in (*part.slots[f], *part.slots[g]) if p >= 0},
                           key=lambda p: -part.evidence_rank[p])
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
    offers = [(float(full[t]), t) for t in np.flatnonzero(full > _GAIN_TOLERANCE)]
    for _, t in _ranked(offers, key=lambda offer: part.pair_key(*rows[offer[1]])):
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
    sequenced parent beats a latent one. A margin within the tolerance is a tie, not a win (release 3.4): a latent
    one-member family is exactly its member alone, and a parent without reads on a unit exactly a latent one, so
    float noise must not decide those (it follows the sample order)."""
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
                hits[c] += stay > elsewhere + _GAIN_TOLERANCE
            for j, p in enumerate((u, v)):
                if p >= 0:
                    latent = family_loglik(part.S[f], part.S[f], 0.0, part.logpi if j == 0 else part.law(u),
                                           part.logpi if j == 1 else part.law(v), lo, hi)
                    parent_hits[f, j] += base[f] > latent + _GAIN_TOLERANCE
    return hits / max(len(ranges), 1), parent_hits / max(len(ranges), 1)


def fit(C, logpi, logpost, ranges, *, max_sweeps: int, log=print) -> FamilyFit:
    """Families over the samples of C [N, M, 9] (child terms), logpi [M, 3], logpost [N, M, 3]; ``ranges``: each
    support unit's site range (lo, hi) (relatedness.stage.support_units)."""
    part = _Partition(C, logpi, logpost)
    penalty = float(np.log(part.N))
    _agglomerate(part, log)
    # Release 3.4: two order-free searches from the agglomeration, and the higher objective is kept (the first on a
    # tie): the objective has many local optima, and neither search reaches the better one on every design.
    start = part.snapshot()
    log("  search best_improvement")
    sweeps, converged = _sweeps(part, _moves, penalty, max_sweeps, log)
    first, first_total = (part.snapshot(), sweeps, converged), part.total()
    part.restore(start)
    log("  search content_order")
    sweeps, converged = _sweeps(part, _content_moves, penalty, max_sweeps, log)
    second_total = part.total()
    if second_total > first_total + _GAIN_TOLERANCE:
        search, margin = "content_order", second_total - first_total
    else:
        state, sweeps, converged = first
        part.restore(state)
        search, margin = "best_improvement", first_total - second_total
    log(f"  kept the {search} search (objective {part.total():.1f}, margin {margin:.1f})")
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
    return FamilyFit(members=tuple(np.asarray(sorted(part.members[f]), dtype=np.int64) for f in live), parents=parents,
                     member_support=member_support, parent_support=parent_support,
                     half_sib=np.asarray(half, dtype=np.int64).reshape(-1, 2),
                     half_sib_gain=np.asarray(gains, dtype=np.float64), loglik=part.total(), sweeps=sweeps,
                     converged=converged, search=search, search_margin=float(margin))
