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

Implementation (release 3.4, the relatedness speed-up). The search is the one above, step for step; only what is
evaluated, and when, changed. Every gain is the same kernel expression with the same operands as before, so every
comparison, tie and decision is the same:
- every sample's gain on joining every family, and its family's change on its leaving, are kept between steps and
  evaluated again only for families changed since (each family carries a version that every change of its members,
  sums or slots advances), across moves, parent assignment, merges and both searches;
- a family's parent-slot offers and its members' parent offers are kept for that family version, and a pair's merge
  gains for the two families' versions;
- the agglomeration evaluates the half-sib link only where the full-sib gain is positive (elsewhere the pair is no
  candidate either way) and finds the largest gain from per-row maxima instead of a scan of the gain matrix;
- the acyclicity check of a parent assignment follows the new parent's ancestors only (the graph was acyclic before);
- support and the half-sib links are evaluated in single parallel kernels.
``fit(..., reference=True)`` runs the previous implementation (``families_reference``) for validation.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cmp_to_key

import numpy as np
from numba.typed import List

from .kernels import (alone_logliks, family_loglik_parallel, family_logliks, family_screen, full_sib_gains,
                      half_screen, half_sib_gains, join_entries, join_screen, latent_alternatives, leave_entries,
                      member_parent_logliks, merge_gains, pair_indices, posterior_screen, sample_screen, screen_all,
                      screen_one, screen_shapes, slot_gains, slot_screen, unit_terms)

_GAIN_TOLERANCE = 1e-6     # moves and merges must improve the objective by more than this (log units); gains of
                           # competing moves or offers within it of each other tie (decided by content keys)
# The screens' error bound (_Partition.screen_error), against which a candidate is set aside. Two sources, bounded
# separately (first order): (i) the float32 linear-domain terms, each product of rescaled terms within about 2^-23 of
# its value and summed without cancellation, so about 1.2e-7 log units per site, covered by 1e-6 per site; (ii) the
# float64 reductions of the estimate and of the exact gain (sums over M sites, then differences of family
# log-likelihoods), each within about (M + 16) 2^-53 times the summed magnitudes of the log-likelihoods involved,
# covered by twice that; plus 0.1 log units. Measured screen errors are far below the bound (design receipts: at most
# 2.7e-5 log units over 56,502 sites; a 10^7-site arithmetic stress test reached 6.6 against a bound of 10.1).
_SCREEN_MARGIN = 0.1
_SCREEN_MARGIN_PER_SITE = 1e-6
_SCREEN_ROUNDING = 2.0 ** -53
_SCREEN_TILE = 16
_SUM_BYTES = 32 * 1024**2  # temporary child terms in a family-sum reduction, independent of total sites
_PAIR_BATCH = 262144       # storage batch only: every unordered pair is still screened and considered
_PREFETCH_BATCH = 64        # content-order moves: samples whose exact join gains near their best are prepared at once


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
    """Mutable families over the run's samples (family indices are reused once a family empties), with the search's
    kept gains.

    version[f] changes with every change of family f (members, sums, slots); a kept value records the version it was
    computed for and is evaluated again once that differs. join[f, c]: the change of family f's log-likelihood when
    sample c joins it; leave[c]: the change of c's family's log-likelihood when c leaves it."""

    def __init__(self, C, logpi, logpost):
        self.C, self.logpi, self.logpost = C, logpi, logpost
        self.N, self.M = C.shape[0], C.shape[1]
        # Every live family contains a sample; even a split or an undone merge needs at most N slots.
        capacity = self.N
        self.empty_sum = np.empty((0, 9))
        self.S = List([self.empty_sum for _ in range(capacity)])
        self.shared_sum = np.zeros(capacity, dtype=np.bool_)
        self.ll = np.zeros(capacity)
        self.slots = np.full((capacity, 2), -1, dtype=np.int64)
        self.members = [[] for _ in range(capacity)]
        self.of = np.full(self.N, -1, dtype=np.int64)
        self.free = list(range(capacity - 1, -1, -1))
        self.evidence_rank = _evidence_ranks(C)
        self.content_order = np.argsort(-self.evidence_rank, kind="stable")
        # Numerical gain of forming a singleton; deliberately separate from the evidence order.
        self.solo = np.empty(self.N)
        alone_logliks(C, logpi, self.solo)
        self.everyone = np.arange(self.N, dtype=np.int64)
        self.clock = 0
        self.version = np.zeros(capacity, dtype=np.int64)
        self.join = np.zeros((capacity, self.N))
        self.join_at = np.full((capacity, self.N), -1, dtype=np.int64)
        self.leave = np.zeros(self.N)
        self.leave_at = np.full(self.N, -1, dtype=np.int64)
        self.singleton_keys = [((int(rank),), (-1, -1)) for rank in self.evidence_rank]
        self._live = None
        self._keys = {}
        self._slot_logliks = {}
        self._member_parent = {}
        self._merge_pairs = {}
        self._children_slots, self._children = None, {}     # families by sequenced parent, for the slots seen
        self._posterior = None                               # float32 genotype posteriors for the slot screen
        self._samples_screen = None                          # samples' linear-domain terms for the join screen
        self.estimate = np.zeros((capacity, self.N))         # the join screen's estimates of join, per row version
        self.estimate_at = np.full(capacity, -1, dtype=np.int64)
        self.screen_margin = _SCREEN_MARGIN + _SCREEN_MARGIN_PER_SITE * self.M
        self.screen_rounding = 2 * (self.M + 16) * _SCREEN_ROUNDING

    def screen_error(self, magnitude):
        """The bound on a screen estimate's error for a gain whose log-likelihoods have summed absolute values
        ``magnitude`` (see _SCREEN_MARGIN)."""
        return self.screen_margin + self.screen_rounding * magnitude

    # --- changes ------------------------------------------------------------------------------------------------

    def touch(self, f):
        self.clock += 1
        self.version[f] = self.clock

    def law(self, p):
        return self.logpi if p < 0 else self.logpost[p]

    def score(self, f):
        u, v = self.slots[f]
        return family_loglik_parallel(self.S[f], self.S[f], 0.0, self.law(u), self.law(v), 0, self.M)

    def own_sum(self, f):
        """Detach child terms or a saved start sum before an in-place update."""
        if self.shared_sum[f]:
            self.S[f] = self.S[f].copy()
            self.shared_sum[f] = False

    def sum_members(self, f):
        """The original sum(axis=0), in bounded site slices with the same member/reduction order.

        A singleton's child terms already are its sum (family_terms emits +0 at neutral cells). Borrow that row;
        multi-member rows are owned. Slice only the non-reduced site axis so NumPy adds members in the same order.
        """
        members = self.members[f]
        if len(members) == 1:
            self.S[f] = self.C[members[0]]
            self.shared_sum[f] = True
            return
        out = np.empty((self.M, 9))
        step = max(1, _SUM_BYTES // (len(members) * 9 * 8))
        for lo in range(0, self.M, step):
            out[lo:lo + step] = self.C[members, lo:lo + step].sum(axis=0)
        self.S[f] = out
        self.shared_sum[f] = False

    def new_family(self, members):
        f = self.free.pop()
        self.members[f] = sorted(members, key=lambda c: -self.evidence_rank[c])
        self.slots[f] = -1
        self.sum_members(f)
        for c in members:
            self.of[c] = f
        self.ll[f] = self.score(f)
        self.touch(f)
        self._live = None
        return f

    def singletons(self):
        """new_family([c]) for every sample c in content order (a one-member family's sums are its member's terms,
        and its log-likelihood is the member alone with latent parents, ``solo``)."""
        for c in self.content_order:
            c = int(c)
            f = self.free.pop()
            self.members[f] = [c]
            self.slots[f] = -1
            self.S[f] = self.C[c]
            self.shared_sum[f] = True
            self.of[c] = f
            self.ll[f] = self.solo[c]
            self.touch(f)
        self._live = None

    def _release(self, f):
        self.members[f] = []
        self.slots[f] = -1
        self.S[f] = self.empty_sum
        self.shared_sum[f] = False
        self.ll[f] = 0.0
        self.free.append(f)
        self.touch(f)
        self._live = None

    def live(self):
        if self._live is None:
            self._live = np.asarray([f for f in range(len(self.members)) if self.members[f]], dtype=np.int64)
        return self._live

    def move(self, c, g):
        """Move sample c out of its family into family g (g < 0: a new one-member family); returns c's family."""
        f = int(self.of[c])
        self.members[f].remove(c)
        if self.members[f]:
            self.own_sum(f)
            self.S[f] -= self.C[c]
            self.ll[f] = self.score(f)
            self.touch(f)
        else:
            self._release(f)
        if g < 0:
            return self.new_family([c])
        self.members[g].append(c)
        self.members[g].sort(key=lambda c: -self.evidence_rank[c])
        self.own_sum(g)
        self.S[g] += self.C[c]
        self.of[c] = g
        self.ll[g] = self.score(g)
        self.touch(g)
        return g

    def merge(self, f, g, parents):
        for c in self.members[g]:
            self.of[c] = f
        self.members[f] = sorted(self.members[f] + self.members[g], key=lambda c: -self.evidence_rank[c])
        self.own_sum(f)
        self.S[f] += self.S[g]
        self._release(g)
        self.slots[f] = parents
        self.ll[f] = self.score(f)
        self.touch(f)

    def snapshot(self, keep=False):
        """The families (members, parent slots, free indices), to be restored later; ``keep``: also the kept gains
        and the families' sums, so that a family restored to identical sums keeps them."""
        kept = None
        if keep:
            live = self.live()
            self.shared_sum[live] = True
            kept = dict(S={int(f): self.S[f] for f in live}, version=self.version.copy(),
                        join={int(f): self.join[f].copy() for f in live},
                        join_at={int(f): self.join_at[f].copy() for f in live},
                        estimate={int(f): self.estimate[f].copy() for f in live},
                        estimate_at=self.estimate_at.copy(), leave=self.leave.copy(), leave_at=self.leave_at.copy())
        return [list(m) for m in self.members], self.slots.copy(), self.of.copy(), list(self.free), kept

    def restore(self, state):
        """Back to a snapshot: the family sums and log-likelihoods are computed again from the members."""
        for f in self.live():
            self.S[f] = self.empty_sum
            self.shared_sum[f] = False
            self.ll[f] = 0.0
            self.touch(f)
        members, slots, of, free, kept = state
        self.members, self.slots, self.of, self.free = [list(m) for m in members], slots.copy(), of.copy(), list(free)
        self._live = None
        live = self.live()
        for f in live:
            self.sum_members(f)
        scores = np.empty(len(live))
        family_logliks(self.S, self.slots, live, self.logpi, self.logpost, 0, self.M, scores)
        self.ll[live] = scores
        for f in live:
            saved = kept["S"].get(int(f)) if kept is not None else None
            if saved is not None and np.array_equal(saved.view(np.int64), self.S[f].view(np.int64)):
                # the same members, slots and sums as at the snapshot: what was kept for that version still holds
                self.version[f] = kept["version"][f]
                self.join[f] = kept["join"][f]
                self.join_at[f] = kept["join_at"][f]
                self.estimate[f] = kept["estimate"][f]
                self.estimate_at[f] = kept["estimate_at"][f]
                mine = np.asarray(self.members[f], dtype=np.int64)
                self.leave[mine] = kept["leave"][mine]
                self.leave_at[mine] = kept["leave_at"][mine]
            else:
                self.touch(f)

    # --- reads --------------------------------------------------------------------------------------------------

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
        cached = self._keys.get(int(f))
        if cached is not None and cached[0] == self.version[f]:
            return cached[1]
        key = (tuple(sorted((int(self.evidence_rank[c]) for c in self.members[f]), reverse=True)),
               tuple(sorted((int(self.evidence_rank[p]) if p >= 0 else -1 for p in self.slots[f]), reverse=True)))
        self._keys[int(f)] = (self.version[f], key)
        return key

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

    def blocked_for(self, c):
        """The families sample c may not join (c in ``blocked``): those with a sequenced parent that is c or one of
        c's descendants."""
        if self._children_slots is None or not np.array_equal(self._children_slots, self.slots):
            self._children_slots = self.slots.copy()
            self._children = {}
            for g, x in zip(*np.nonzero(self.slots >= 0)):
                self._children.setdefault(int(self.slots[g, x]), []).append(int(g))
        out, seen, stack = set(), {int(c)}, [int(c)]
        while stack:
            x = stack.pop()
            for g in self._children.get(x, ()):
                if g not in out:
                    out.add(g)
                    for m in self.members[g]:
                        if m not in seen:
                            seen.add(m)
                            stack.append(m)
        return out

    def acyclic(self):
        """Whether no sample is its own ancestor through the sequenced parent slots."""
        return all(x not in self.ancestors(x) for x in range(self.N))

    def acyclic_with_parent(self, f, p):
        """acyclic() once p fills a free slot of family f, the graph being acyclic before: a new cycle would run
        through a new edge p -> member of f, so one exists exactly when a member of f is an ancestor of p."""
        return not (set(self.members[f]) & self.ancestors(p))

    # --- kept gains ---------------------------------------------------------------------------------------------

    def sync_pairs(self, families, samples):
        """join[families[t], samples[t]] current (exact)."""
        families, samples = np.asarray(families, dtype=np.int64), np.asarray(samples, dtype=np.int64)
        stale = self.join_at[families, samples] != self.version[families]
        if not stale.any():
            return
        families, samples = families[stale], samples[stale]
        out = np.empty(len(families))
        join_entries(self.S, self.C, self.slots, self.ll, self.logpi, self.logpost, families, samples, out)
        self.join[families, samples] = out
        self.join_at[families, samples] = self.version[families]

    def sync_estimates(self, rows):
        """estimate[rows] current: the join screen's estimate of every sample's join gain, per family version."""
        rows = np.asarray(rows, dtype=np.int64)
        rows = rows[self.estimate_at[rows] != self.version[rows]]
        if not len(rows):
            return
        if self._samples_screen is None:
            n, G = np.empty((self.N, self.M)), np.empty((self.N, self.M, 9), dtype=np.float32)
            sample_screen(self.C, n, G)
            self._samples_screen = (n, G, n.sum(axis=1))
        n, G, nsum = self._samples_screen
        h, H = np.empty(self.M), np.empty((self.M, 9))
        for g in rows:
            u, v = self.slots[g]
            family_screen(self.S[g], self.law(u), self.law(v), h, H)
            join_screen(self.S[g], self.law(u), self.law(v), h, H, h.sum() - self.ll[g], self.C, n, G, nsum,
                        self.estimate[g])
            self.estimate_at[g] = self.version[g]

    def sync_leave(self, samples):
        samples = np.asarray(samples, dtype=np.int64)
        if not len(samples):
            return
        families = self.of[samples]
        stale = self.leave_at[samples] != self.version[families]
        if not stale.any():
            return
        samples, families = samples[stale], families[stale]
        single = np.asarray([len(self.members[f]) == 1 for f in families], dtype=np.bool_)
        out = np.empty(len(samples))
        leave_entries(self.S, self.C, self.slots, self.ll, self.logpi, self.logpost, families, samples, single, out)
        self.leave[samples] = out
        self.leave_at[samples] = self.version[families]

    def leave_gain(self, c):
        """Log-likelihood change of c's family when c leaves it."""
        self.sync_leave([c])
        return self.leave[c]

    def slot_logliks(self, f, allowed):
        """float64 [N]: family f's log-likelihood with its free slot filled by sample p (slot_gains), for every allowed
        p whose value may be within the tolerance of the largest (NaN for the others), kept for f's version. A screen
        (kernels.slot_screen) estimates every allowed p; only those whose estimate plus its error bound reaches the
        largest estimate minus its bound, less the tolerance, are evaluated exactly: they include the largest value and
        every value tied with it."""
        key = (int(f), int(self.version[f]))
        out = self._slot_logliks.get(key)
        if out is None:
            k = int(np.flatnonzero(self.slots[f] == -1)[0])
            law = self.law(int(self.slots[f][1 - k]))
            if self._posterior is None:
                self._posterior = np.empty((self.N, self.M, 3), dtype=np.float32)
                posterior_screen(self.logpost, self._posterior)
            estimate = np.empty(self.N)
            slot_screen(self.S[f], law, self._posterior, self.logpost, estimate)
            estimate[~allowed] = -np.inf
            error = self.screen_error(np.abs(np.where(allowed, estimate, 0.0)) + 2 * abs(self.ll[f]))
            exact = allowed & (estimate + error >= (estimate - error).max() - _GAIN_TOLERANCE)
            out = np.empty(self.N)
            slot_gains(self.S[f], law, self.logpost, exact, out)
            self._slot_logliks[key] = out
        return out

    def member_parent(self, f):
        """{member p: family f's log-likelihood without p and with p in its free slot}, kept for f's version."""
        key = (int(f), int(self.version[f]))
        out = self._member_parent.get(key)
        if out is None:
            k = int(np.flatnonzero(self.slots[f] == -1)[0])
            members = np.asarray(self.members[f], dtype=np.int64)
            values = np.empty(len(members))
            member_parent_logliks(self.S[f], self.C, members, self.law(int(self.slots[f][1 - k])), self.logpost,
                                  values)
            out = dict(zip(members.tolist(), values))
            self._member_parent[key] = out
        return out

    def prune(self, keep_versions=()):
        """Forget kept offers of family versions that no longer exist (other than ``keep_versions``)."""
        current = {(int(f), int(self.version[f])) for f in self.live()} | set(keep_versions)
        for cache in (self._slot_logliks, self._member_parent):
            for key in [key for key in cache if key not in current]:
                del cache[key]
        self._merge_pairs = {key: value for key, value in self._merge_pairs.items()
                             if (key[0], key[1]) in current and (key[2], key[3]) in current}

    def targets(self, c, exclude=()):
        """The live families sample c may move to: not its own, not ``exclude``, none it is a parent of or blocked
        from (best_placement's candidates, in family order)."""
        live = self.live()
        keep = live != self.of[c]
        for g in exclude:
            keep &= live != g
        keep &= (self.slots[live] != c).all(axis=1)
        blocked = self.blocked_for(c)
        if blocked:
            keep &= ~np.isin(live, np.fromiter(blocked, dtype=np.int64, count=len(blocked)))
        return live[keep]

    def near_best(self, c, targets):
        """Boolean [len(targets)]: the targets whose join gain for c may lie within the tolerance of c's best placement
        (another family or alone), by the screen: every join gain within the tolerance of the best, and the best, is
        among them (each estimate is within its error bound, join_error, of the exact gain)."""
        estimate = self.estimate[targets, c]
        if not len(targets):
            return np.zeros(0, dtype=np.bool_)
        error = self.join_error(targets, c, estimate)
        return estimate + error >= max(self.solo[c], (estimate - error).max()) - _GAIN_TOLERANCE

    def join_error(self, families, samples, estimate):
        """screen_error of join estimates (families joined by samples; -inf estimates excluded): the magnitudes are
        the family's, the joined family's and the sample's own log-likelihoods."""
        ll = np.abs(self.ll[families])
        joined = np.abs(self.ll[families] + np.where(np.isfinite(estimate), estimate, 0.0))
        return self.screen_error(ll + joined + np.abs(self.solo[samples]))

    def merge_error(self, first, second, estimate):
        """screen_error of merge or half-sib-link estimates of family pairs: the magnitudes are the two families' and
        the merged (or linked) families' log-likelihoods."""
        ll_f, ll_g = self.ll[first], self.ll[second]
        merged = np.abs(ll_f + ll_g + np.where(np.isfinite(estimate), estimate, 0.0))
        return self.screen_error(np.abs(ll_f) + np.abs(ll_g) + merged)

    def prefetch_near_best(self, samples, exclude_own=True):
        """Exact join gains near each listed sample's best placement (best_placement's), in one batch."""
        families, columns = [], []
        for c in samples:
            targets = self.targets(c)
            self.sync_estimates(targets)
            near = targets[self.near_best(c, targets)]
            families.append(near)
            columns.append(np.full(len(near), c, dtype=np.int64))
        if families:
            self.sync_pairs(np.concatenate(families), np.concatenate(columns))

    def best_placement(self, c, exclude=()):
        """(gain, target) of moving c out of its family to its best valid other family or a new one (target -1): the
        change of the objective. A target whose sequenced parents descend from c is not valid. Join gains are
        evaluated exactly near the best only (the others cannot tie with it or exceed it; NaN here)."""
        leave = self.leave_gain(c)
        targets = self.targets(c, exclude)
        self.sync_estimates(targets)
        near = self.near_best(c, targets)
        self.sync_pairs(targets[near], np.full(int(near.sum()), c, dtype=np.int64))
        joins = np.full(len(targets), np.nan)
        joins[near] = self.join[targets[near], c]
        values = np.concatenate(([leave + self.solo[c]], leave + joins))

        def keyed(j):
            return ((int(self.evidence_rank[c]),), (-1, -1)) if j == 0 else self.key(int(targets[j - 1]))
        j = _top_keyed(values, keyed)
        return float(values[j]), (-1 if j == 0 else int(targets[j - 1]))


def _top(gains, keys) -> int:
    """Index of the largest gain (NaN: not a candidate). Gains within the tolerance of it tie, and the tie goes to the
    largest content key, so the choice follows the data, not sample names or input order."""
    tied = np.flatnonzero(gains >= np.nanmax(gains) - _GAIN_TOLERANCE)
    return int(max(tied, key=lambda j: keys[j]))


def _top_keyed(gains, key) -> int:
    """_top with the content keys evaluated for the tied gains only (key(j): candidate j's key)."""
    tied = np.flatnonzero(gains >= np.nanmax(gains) - _GAIN_TOLERANCE)
    return int(tied[0]) if len(tied) == 1 else int(max(tied, key=key))


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


class _PairGains:
    """Upper triangle in row order; no diagonal, lower triangle, or unused family slots."""

    def __init__(self, n):
        self.n = n
        rows = np.arange(n + 1, dtype=np.int64)
        self.offset = rows * (2 * n - rows - 1) // 2
        self.values = np.empty(n * (n - 1) // 2)

    def row(self, f):
        return self.values[self.offset[f]:self.offset[f + 1]]

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
            half_sib_gains(part.S, part.ll, np.ascontiguousarray(np.stack([first[unsure], second[unsure]], axis=1)),
                           part.logpi, half)
            entry[unsure] = np.where(full[unsure] > half, full[unsure], -np.inf)
    gain.put(first, second, entry)


def _agglomerate(part: _Partition, log=print):
    """Latent full-sib families from one-member families: repeatedly merge the pair with the largest full-sib gain,
    when positive and larger than its half-sib link gain.

    Only such gains are candidates, so the gain matrix keeps those (else -inf), and each row's maximum is kept and
    updated with the entries a merge changes. A screen (kernels.screen_all, screen_one: the full-sib gain in the
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
    screen_all(part.S, W, m, A, full_base, N, _SCREEN_TILE, gain.values)
    for lo in range(0, len(gain.values), _PAIR_BATCH):
        hi = min(len(gain.values), lo + _PAIR_BATCH)
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
        bound = maximum - _GAIN_TOLERANCE
        tied = []
        for r in np.flatnonzero(rowmax >= bound):
            row = gain.row(r)
            tied.extend((int(r), int(r + 1 + c)) for c in np.flatnonzero((row > 0) & (row >= bound)))
        f, g = tied[0] if len(tied) == 1 else max(tied, key=lambda pair: part.pair_key(*pair))
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
        below = others[others < f]           # their entries (h, f) lie in column f
        before = gain.get(below, f)
        low, high = np.minimum(others, f), np.maximum(others, f)
        gain.put(low, high, -np.inf)
        candidates = estimate > -part.merge_error(low, high, estimate)
        _agglomeration_entries(part, low[candidates], high[candidates], gain, screen)
        after = gain.get(below, f)
        rise = after > rowmax[below]
        rowmax[below[rise]] = after[rise]
        recount |= set(below[(before == rowmax[below]) & (after < before) & (before > -np.inf)].tolist())
        for r in recount:
            rowmax[r] = gain.maximum(r)
    log(f"  agglomeration: {len(part.live())} families, objective {part.total():.1f}")


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
    return (family_loglik_parallel(part.S[f], part.S[f], 0.0, part.law(other), part.logpost[p], 0, part.M)
            - part.ll[f], None)


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
    """Best-improvement single-sample moves (release 3.4): repeatedly the one move with the largest gain over all
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
        error = part.screen_error(2 * np.abs(part.ll[live]).max() + np.abs(part.solo).max()
                                  + (np.abs(finite).max() if len(finite) else 0.0))
        low = np.maximum(part.solo, (estimate - error).max(axis=1))   # each sample's best placement is at least this,
        upper = leave + np.maximum(part.solo, (estimate + error).max(axis=1))   # its move gain at most this,
        lower = leave + low - _GAIN_TOLERANCE                                  # and at least this
        candidates = np.flatnonzero((upper >= lower.max() - _GAIN_TOLERANCE) & (upper > _GAIN_TOLERANCE))
        if not len(candidates):
            break
        rows, cols = np.nonzero(estimate[candidates] + error >= (low[candidates] - _GAIN_TOLERANCE)[:, None])
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
        tied = gains >= (best - _GAIN_TOLERANCE)[:, None]
        pick = np.where(tied, keys[None, :], -1).argmax(axis=1)
        family_rank = np.where(tied, keys[None, :], -1).max(axis=1)
        alone_rank = np.where(solo >= best - _GAIN_TOLERANCE, alone_keys, -1)
        target = np.where(family_rank > alone_rank, live[pick], -1)
        gain = leave[candidates] + np.where(target >= 0, gains[np.arange(len(candidates)), pick], solo)
        eligible = np.where(gain > _GAIN_TOLERANCE, gain, -np.inf)
        if not np.isfinite(eligible).any():
            break
        j = _top(eligible, part.evidence_rank[candidates])
        c, g = int(candidates[j]), int(target[j])
        # scored again on the current partition before it is applied
        if g < 0:
            fresh = part.leave_gain(c) + part.solo[c]
        else:
            u, v = part.slots[g]
            fresh = part.leave_gain(c) + family_loglik_parallel(part.S[g], part.C[c], 1.0, part.law(u), part.law(v),
                                                                0, part.M) - part.ll[g]
        if not fresh > _GAIN_TOLERANCE or (g >= 0 and c in part.blocked(g)):
            continue            # a stale score (not expected): every kept gain is current already
        part.move(c, g)
        changed += 1
    return changed


def _content_moves(part: _Partition) -> int:
    """First-improvement single-sample moves in content order (release 3.4, the second search): each sample once, by
    decreasing content key, moves to its best valid placement when that gains more than the tolerance. Exact join
    gains near the best are prepared for a batch of the coming samples at a time (and again for a sample whose
    candidate families changed meanwhile)."""
    changed = 0
    order = part.content_order
    part.sync_estimates(part.live())
    part.sync_leave(part.everyone)
    batch = _PREFETCH_BATCH
    for i, c in enumerate(order):
        if i % batch == 0:
            part.prefetch_near_best(order[i:i + batch])
        gain, target = part.best_placement(int(c))
        if gain > _GAIN_TOLERANCE:
            f = int(part.of[c])
            h = part.move(int(c), target)
            changed += 1
            rows = [h] + ([f] if part.members[f] else [])
            part.sync_leave([m for r in rows for m in part.members[r]])
    return changed


def _sweeps(part: _Partition, moves, penalty: float, max_sweeps: int, log, keep_versions=()):
    """Sweeps of parent assignment, ``moves`` and one merge until a sweep changes nothing; (sweeps, converged)."""
    sweeps, changed = 0, 1
    while changed and sweeps < max_sweeps:
        changed = _assign_parents(part, penalty, keep_versions) + moves(part) + _merge_one(part)
        sweeps += 1
        named = sum(int(p >= 0) for f in part.live() for p in part.slots[f])
        log(f"  sweep {sweeps}: {changed} changes, {len(part.live())} families, {named} sequenced parent slots, "
            f"objective {part.total():.1f}")
    return sweeps, not changed


def _merge_one(part: _Partition) -> int:
    """The best merge of two families whose sequenced parents are compatible (at most two, none a member); two latent
    families merge only when the full-sib gain beats their half-sib link. A pair's gains are kept for the two
    families' versions."""
    live = part.live()
    named_of = {int(f): {int(p) for p in part.slots[f] if p >= 0} for f in live}
    member_set = {int(f): set(part.members[f]) for f in live}
    rows, parents = [], []
    for a in range(len(live)):
        f = int(live[a])
        for b in range(a + 1, len(live)):
            g = int(live[b])
            union = named_of[f] | named_of[g]
            if len(union) > 2 or any(p in member_set[f] or p in member_set[g] for p in union):
                continue
            named = sorted({int(p) for p in (*part.slots[f], *part.slots[g]) if p >= 0},
                           key=lambda p: -part.evidence_rank[p])
            rows.append((f, g))
            parents.append((named + [-1, -1])[:2])
    if not rows:
        return 0
    rows, parents = np.asarray(rows, dtype=np.int64), np.asarray(parents, dtype=np.int64)
    out = np.empty((len(rows), 2))
    keys = [(int(f), int(part.version[f]), int(g), int(part.version[g])) for f, g in rows]
    missing = []
    for t, key in enumerate(keys):
        value = part._merge_pairs.get(key)
        if value is None:
            missing.append(t)
        else:
            out[t] = value
    if missing:
        missing = np.asarray(missing, dtype=np.int64)
        fresh = np.empty((len(missing), 2))
        merge_gains(part.S, part.ll, np.ascontiguousarray(rows[missing]), np.ascontiguousarray(parents[missing]),
                    part.logpi, part.logpost, _GAIN_TOLERANCE, fresh)
        out[missing] = fresh
        for t, value in zip(missing.tolist(), fresh):
            part._merge_pairs[keys[t]] = value.copy()
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
        part.own_sum(f)
        part.S[f] -= part.S[h]
        part.slots[f], part.slots[h] = before
        part.ll[f], part.ll[h] = part.score(f), part.score(h)
        part.touch(f)
        part.touch(h)
    return 0


def _support(part: _Partition, ranges):
    """Per chromosome (with family sites): whether each member's family is its best placement, and whether each
    sequenced parent beats a latent one. A margin within the tolerance is a tie, not a win (release 3.4): a latent
    one-member family is exactly its member alone, and a parent without reads on a unit exactly a latent one, so
    float noise must not decide those (it follows the sample order)."""
    ranges = [(lo, hi) for lo, hi in ranges if hi > lo]
    live = part.live()
    N, F = part.N, len(live)
    hits = np.zeros(N)
    parent_hits = np.zeros((len(part.members), 2))
    position = np.full(len(part.members), -1, dtype=np.int64)
    position[live] = np.arange(F)
    family_of = position[part.of]
    single = np.asarray([len(part.members[f]) == 1 for f in live], dtype=np.bool_)[family_of]
    # no placement in the sample's own family, nor in a family it is a parent of
    excluded_rows, excluded_cols = [family_of], [part.everyone]
    for j, f in enumerate(live):
        for p in part.slots[f]:
            if p >= 0:
                excluded_rows.append(np.asarray([j], dtype=np.int64))
                excluded_cols.append(np.asarray([p], dtype=np.int64))
    excluded_rows, excluded_cols = np.concatenate(excluded_rows), np.concatenate(excluded_cols)
    named = part.slots[live] >= 0
    joins, rest, alone, latent = np.empty((F, N)), np.empty(N), np.empty(N), np.empty((F, 2))
    for lo, hi in ranges:
        llk = np.empty(F)
        family_logliks(part.S, part.slots, live, part.logpi, part.logpost, lo, hi, llk)
        unit_terms(part.S, part.C, part.slots, live, llk, family_of, part.logpi, part.logpost, lo, hi, joins, rest,
                   alone)
        rest[single] = 0.0
        stay = llk[family_of] - rest
        joins[excluded_rows, excluded_cols] = -np.inf
        elsewhere = np.maximum(joins.max(axis=0) if F else np.full(N, -np.inf), alone)
        hits += stay > elsewhere + _GAIN_TOLERANCE
        latent_alternatives(part.S, part.slots, live, part.logpi, part.logpost, lo, hi, latent)
        parent_hits[live] += named & (llk[:, None] > latent + _GAIN_TOLERANCE)
    return hits / max(len(ranges), 1), parent_hits / max(len(ranges), 1)


def fit(C, logpi, logpost, ranges, *, max_sweeps: int, log=print, reference: bool = False) -> FamilyFit:
    """Families over the samples of C [N, M, 9] (child terms), logpi [M, 3], logpost [N, M, 3]; ``ranges``: each
    support unit's site range (lo, hi) (relatedness.stage.support_units). ``reference``: run the previous
    implementation of the same search instead (families_reference; validation only)."""
    if reference:
        from . import families_reference
        fitted = families_reference.fit(C, logpi, logpost, ranges, max_sweeps=max_sweeps, log=log)
        return FamilyFit(**{field.name: getattr(fitted, field.name) for field in fields(FamilyFit)})
    part = _Partition(C, logpi, logpost)
    penalty = float(np.log(part.N))
    _agglomerate(part, log)
    # Release 3.4: two order-free searches from the agglomeration, and the higher objective is kept (the first on a
    # tie): the objective has many local optima, and neither search reaches the better one on every design.
    start = part.snapshot(keep=True)
    start_versions = [(f, int(start[4]["version"][f])) for f in start[4]["S"]]
    log("  search best_improvement")
    sweeps, converged = _sweeps(part, _moves, penalty, max_sweeps, log, start_versions)
    first, first_total = (part.snapshot(), sweeps, converged), part.total()
    part.restore(start)
    del start
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
    latent = np.asarray([j for j, f in enumerate(live) if (part.slots[f] < 0).all()], dtype=np.int64)
    a, b = np.triu_indices(len(latent), k=1)
    pairs = np.stack([latent[a], latent[b]], axis=1).astype(np.int64).reshape(-1, 2)
    gains = np.empty(len(pairs))
    if len(pairs):
        half_sib_gains(part.S, part.ll, np.ascontiguousarray(live[pairs]), part.logpi, gains)
    linked = gains > penalty
    return FamilyFit(members=tuple(np.asarray(sorted(part.members[f]), dtype=np.int64) for f in live), parents=parents,
                     member_support=member_support, parent_support=parent_support,
                     half_sib=pairs[linked].reshape(-1, 2), half_sib_gain=np.asarray(gains[linked], dtype=np.float64),
                     loglik=part.total(), sweeps=sweeps, converged=converged, search=search,
                     search_margin=float(margin))
