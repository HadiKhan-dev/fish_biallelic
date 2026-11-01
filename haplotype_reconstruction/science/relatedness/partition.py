"""Keep mutable family sums and versioned gain caches.

Owns the relocated partition units. No storage, environment or runtime policy.
Method: docs/methods/relatedness.md; WP_RL1. Legacy arithmetic retained for relocation.
"""

from __future__ import annotations

import numpy as np
from numba.typed import List

from .family_kernels import (
    alone_logliks,
    family_loglik_parallel,
    family_logliks,
    family_screen,
    join_entries,
    join_screen,
    leave_entries,
    member_parent_logliks,
    posterior_screen,
    sample_screen,
    slot_gains,
    slot_screen,
)
from .keys import _evidence_ranks, _top_keyed
from .parameters import FamilyParameters

_DEFAULT_FAMILY = FamilyParameters()


class _Partition:
    """Mutable families over the run's samples (family indices are reused once a family empties), with the search's
    kept gains.

    version[f] changes with every change of family f (members, sums, slots); a kept value records the version it was
    computed for and is evaluated again once that differs. join[f, c]: the change of family f's log-likelihood when
    sample c joins it; leave[c]: the change of c's family's log-likelihood when c leaves it."""

    def __init__(self, C, logpi, logpost, parameters: FamilyParameters = _DEFAULT_FAMILY):
        self.parameters = parameters
        self.C, self.logpi, self.logpost = C, logpi, logpost
        self.N, self.M = C.shape[0], C.shape[1]
        # Every live family contains a sample; even a split or an undone merge needs at most N slots.
        capacity = self.N
        self.empty_sum = np.empty((0, 9))
        self.S = List([self.empty_sum for _ in range(capacity)])
        self.shared_sum = np.zeros(capacity, dtype=np.bool_)
        self.ll = np.zeros(capacity)
        self.slots = np.full((capacity, 2), -1, dtype=np.int64)
        self.members: list[list[int]] = [[] for _ in range(capacity)]
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
        self._keys: dict[int, tuple[np.int64, tuple[tuple[int, ...], tuple[int, ...]]]] = {}  # f: (version, key)
        self._slot_logliks: dict[tuple[int, int], np.ndarray] = {}  # (f, version)
        self._member_parent: dict[tuple[int, int], dict[int, float]] = {}  # (f, version)
        self._merge_pairs: dict[tuple[int, int, int, int], np.ndarray] = {}  # (f, version f, g, version g)
        self._children_slots: np.ndarray | None = None  # families by sequenced parent, for the slots seen
        self._children: dict[int, list[int]] = {}
        self._posterior = None  # float32 genotype posteriors for the slot screen
        self._samples_screen = None  # samples' linear-domain terms for the join screen
        self.estimate = np.zeros((capacity, self.N))  # the join screen's estimates of join, per row version
        self.estimate_at = np.full(capacity, -1, dtype=np.int64)
        self.screen_margin = self.parameters.screen_margin + self.parameters.screen_margin_per_site * self.M
        self.screen_rounding = 2 * (self.M + self.parameters.screen_padding) * self.parameters.screen_rounding

    def screen_error(self, magnitude):
        """The bound on a screen estimate's error for a gain whose log-likelihoods have summed absolute values
        ``magnitude`` (see self.parameters.screen_margin)."""
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
        step = max(1, self.parameters.sum_bytes // (len(members) * 9 * 8))
        for lo in range(0, self.M, step):
            out[lo : lo + step] = self.C[members, lo : lo + step].sum(axis=0)
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
            kept = dict(
                S={int(f): self.S[f] for f in live},
                version=self.version.copy(),
                join={int(f): self.join[f].copy() for f in live},
                join_at={int(f): self.join_at[f].copy() for f in live},
                estimate={int(f): self.estimate[f].copy() for f in live},
                estimate_at=self.estimate_at.copy(),
                leave=self.leave.copy(),
                leave_at=self.leave_at.copy(),
            )
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
        key = (
            tuple(sorted((int(self.evidence_rank[c]) for c in self.members[f]), reverse=True)),
            tuple(sorted((int(self.evidence_rank[p]) if p >= 0 else -1 for p in self.slots[f]), reverse=True)),
        )
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
            for g, x in zip(*np.nonzero(self.slots >= 0), strict=False):
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
            join_screen(
                self.S[g], self.law(u), self.law(v), h, H, h.sum() - self.ll[g], self.C, n, G, nsum, self.estimate[g]
            )
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
            exact = allowed & (estimate + error >= (estimate - error).max() - self.parameters.gain_tolerance)
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
            member_parent_logliks(self.S[f], self.C, members, self.law(int(self.slots[f][1 - k])), self.logpost, values)
            out = dict(zip(members.tolist(), values, strict=False))
            self._member_parent[key] = out
        return out

    def prune(self, keep_versions=()):
        """Forget kept offers of family versions that no longer exist (other than ``keep_versions``)."""
        current = {(int(f), int(self.version[f])) for f in self.live()} | set(keep_versions)
        for cache in (self._slot_logliks, self._member_parent):
            for key in [key for key in cache if key not in current]:
                del cache[key]
        self._merge_pairs = {
            key: value
            for key, value in self._merge_pairs.items()
            if (key[0], key[1]) in current and (key[2], key[3]) in current
        }

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
        return estimate + error >= max(self.solo[c], (estimate - error).max()) - self.parameters.gain_tolerance

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

        j = _top_keyed(values, keyed, self.parameters.gain_tolerance)
        return float(values[j]), (-1 if j == 0 else int(targets[j - 1]))


__all__ = []
