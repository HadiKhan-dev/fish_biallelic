"""Keep mutable family sums and versioned gain caches.

Method: docs/methods/relatedness.md.
"""

from __future__ import annotations

import numpy as np
from numba.typed import List

from ...core.concurrency import ThreadSource
from .class_kernels import (
    FAMILY,
    FAMILY_FAMILY,
    FAMILY_SAMPLE,
    SAMPLE,
    class_family_table,
    class_family_tasks,
    class_half_sib_tasks,
)
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

    def __init__(
        self, C, logpi, logpost, parameters: FamilyParameters = _DEFAULT_FAMILY, threads: ThreadSource = 1,
        class_prior=None, composition=None, ranges=(),
    ):
        self.parameters = parameters
        # rl21 (two source populations): C holds the child terms over parental configuration pairs [N, M, G x G],
        # logpost each sample's configuration posterior [N, M, G] and class_prior the ancestry classes' log laws
        # over the configurations [K, M, G]. A latent slot is then the marginal over the classes (class_kernels) and
        # every gain is exact: no screen runs. None: one source, the Hardy-Weinberg law logpi and the screened kernels.
        self.class_prior = class_prior
        if class_prior is not None:
            self.laws = np.ascontiguousarray(np.exp(class_prior))
            self.logw = np.full(class_prior.shape[0], -np.log(class_prior.shape[0]))
            # each sample's ancestry along the map under parents of given transmitted shares (ancestry.composition):
            # row 0 the genome, then one row per support unit
            self.composition = None  # terms without it (a test of the class kernels alone) add nothing
            if composition is not None:
                self.composition, self.class_type, self.sample_type = composition
            self.span = {(0, C.shape[1]): 0, **{(int(lo), int(hi)): 1 + r for r, (lo, hi) in enumerate(ranges)}}
            self.span[(0, C.shape[1])] = 0  # a single unit that covers the genome is the genome
        # Read this caller-owned source at batch/sweep boundaries. A live source
        # refreshes the ambient kernel team as siblings return their CPUs.
        self.threads = threads
        self.C, self.logpi, self.logpost = C, logpi, logpost
        self.N, self.M = C.shape[0], C.shape[1]
        # Every live family contains a sample; even a split or an undone merge needs at most N slots.
        capacity = self.N
        self.empty_sum = np.empty((0, C.shape[2]))
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
        self.everyone = np.arange(self.N, dtype=np.int64)
        if class_prior is None:
            alone_logliks(C, logpi, self.solo)
        else:
            self.solo[:] = self.class_values(SAMPLE, self.everyone)
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
        self._singleton_slot: dict[tuple[int, int], float] = {}  # (member, kept slot sample), for the fit
        self._children_slots: np.ndarray | None = None  # families by sequenced parent, for the slots seen
        self._children: dict[int, list[int]] = {}
        self._posterior = None  # float32 genotype posteriors for the slot screen
        self._samples_screen = None  # samples' linear-domain terms for the join screen
        self.estimate = np.zeros((capacity, self.N))  # the join screen's estimates of join, per row version
        self.estimate_at = np.full(capacity, -1, dtype=np.int64)
        self.screen_margin = self.parameters.screen_margin + self.parameters.screen_margin_per_site * self.M
        self.screen_rounding = 2 * (self.M + self.parameters.screen_padding) * self.parameters.screen_rounding
        if class_prior is not None:
            self.screen_margin = self.screen_rounding = 0.0  # the "estimates" are exact gains

    # --- ancestry classes (rl21) ---------------------------------------------------------------------------------

    def class_values(self, kind, family, other=None, sign=1.0, slots=None, lo=0, hi=None):
        """float64 [T]: class-marginal family log-likelihoods of a batch (class_kernels.class_family_tasks) over
        sites [lo, hi): ``kind`` with families (or samples) ``family`` [T], ``other`` [T] where the kind reads it,
        ``sign`` of a sample's terms and ``slots`` [T, 2] (default: both latent). A latent slot is summed over the
        classes with their prior weights; a sequenced slot is its sample's one law."""
        family = np.array(family, dtype=np.int64).reshape(-1)
        n = len(family)
        if n == 0:
            return np.empty(0)
        other = np.zeros(n, dtype=np.int64) if other is None else np.array(other, dtype=np.int64).reshape(-1)
        slots = (
            np.full((n, 2), -1, dtype=np.int64) if slots is None
            else np.array(np.broadcast_to(np.asarray(slots, dtype=np.int64), (n, 2)))
        )
        classes = self.laws.shape[0]
        tables = np.empty((n, classes, classes))
        class_family_tasks(
            self.S, self.C, np.full(n, kind, dtype=np.int64), family, other, np.full(n, float(sign)),
            np.ascontiguousarray(slots[:, 0]), np.ascontiguousarray(slots[:, 1]), self.laws, self.logpost,
            lo, self.M if hi is None else hi, tables,
        )
        tables += np.where(slots[:, 0, None, None] < 0, self.logw[None, :, None], 0.0)
        tables += np.where(slots[:, 1, None, None] < 0, self.logw[None, None, :], 0.0)
        if self.composition is not None:
            tables += self._composition(kind, family, other, sign, slots, lo, self.M if hi is None else hi)
        tables = tables.reshape(n, -1)  # unused entries are LOG_ZERO: no weight
        top = tables.max(axis=1)
        return top + np.log(np.exp(tables - top[:, None]).sum(axis=1))

    def _held(self, kind, family, other, sign, row):
        """float64 [T, types, types]: each task's sum over the samples it holds of their composition tables."""
        table = self.composition[row]
        if kind == SAMPLE:
            return table[family]
        families, index = np.unique(family, return_inverse=True)
        out = np.stack([table[self.members[f]].sum(axis=0) for f in families])[index.reshape(-1)]
        if kind == FAMILY_SAMPLE:
            out = out + sign * table[other]
        elif kind == FAMILY_FAMILY:
            families, index = np.unique(other, return_inverse=True)
            out = out + np.stack([table[self.members[g]].sum(axis=0) for g in families])[index.reshape(-1)]
        return out

    def _composition(self, kind, family, other, sign, slots, lo, hi):
        """float64 [T, K, K]: what the held samples' ancestry along the map says about each class pair of the
        task's slots (a latent slot: each class's transmitted share; a sequenced slot: its sample's)."""
        held = self._held(kind, family, other, sign, self.span[(int(lo), int(hi))])
        u = np.where(slots[:, 0, None] < 0, self.class_type[None, :], self.sample_type[np.maximum(slots[:, 0], 0)][:, None])
        v = np.where(slots[:, 1, None] < 0, self.class_type[None, :], self.sample_type[np.maximum(slots[:, 1], 0)][:, None])
        return held[np.arange(len(family))[:, None, None], u[:, :, None], v[:, None, :]]

    def class_half_sib_gains(self, first, second):
        """float64 [T]: the half-sib link gains of latent families first[t], second[t] (one shared latent parent,
        each of the three parents marginal over its own class), over all sites."""
        first, second = np.array(first, dtype=np.int64).reshape(-1), np.array(second, dtype=np.int64).reshape(-1)
        if not len(first):
            return np.empty(0)
        classes = self.laws.shape[0]
        tables = np.empty((len(first), classes, classes, classes))
        class_half_sib_tasks(self.S, first, second, self.laws, 0, self.M, tables)
        tables += (self.logw[:, None, None] + self.logw[None, :, None] + self.logw[None, None, :])[None]
        if self.composition is not None:
            latent = np.full((len(first), 2), -1, dtype=np.int64)
            tables += self._composition(FAMILY, first, first, 1.0, latent, 0, self.M)[:, :, :, None]
            tables += self._composition(FAMILY, second, second, 1.0, latent, 0, self.M)[:, :, None, :]
        tables = tables.reshape(len(first), -1)
        top = tables.max(axis=1)
        return top + np.log(np.exp(tables - top[:, None]).sum(axis=1)) - self.ll[first] - self.ll[second]

    def class_pair_entries(self, first, second):
        """float64 [T]: the latent full-sib merge gain of each family pair when positive and above its half-sib link
        gain, else -inf (the agglomeration's entries, exact)."""
        first, second = np.array(first, dtype=np.int64).reshape(-1), np.array(second, dtype=np.int64).reshape(-1)
        full = self.class_values(FAMILY_FAMILY, first, second) - self.ll[first] - self.ll[second]
        entry = np.full(len(first), -np.inf)
        positive = np.flatnonzero(full > 0)
        if len(positive):
            half = self.class_half_sib_gains(first[positive], second[positive])
            entry[positive] = np.where(full[positive] > half, full[positive], -np.inf)
        return entry

    def slot_classes(self, families):
        """(int64 [F, 2], float64 [F]): each listed family's most probable class per latent slot (-1 for a sequenced
        slot) and that class pair's posterior probability (the classes' prior weights are equal)."""
        classes = self.laws.shape[0]
        best = np.full((len(families), 2), -1, dtype=np.int64)
        probability = np.ones(len(families))
        for j, f in enumerate(families):
            u, v = (int(s) for s in self.slots[f])
            if u >= 0 and v >= 0:
                continue
            table = np.empty((classes if u < 0 else 1, classes if v < 0 else 1))
            class_family_table(self.S[f], self.S[f], 0.0, u, v, self.laws, self.logpost, 0, self.M, table)
            if self.composition is not None:
                table += self._composition(
                    FAMILY, np.asarray([f]), np.asarray([f]), 1.0, np.asarray([[u, v]]), 0, self.M
                )[0][: table.shape[0], : table.shape[1]]
            weights = np.exp(table - table.max())
            a, b = np.unravel_index(int(np.argmax(weights)), weights.shape)
            # an all-latent family's slots are exchangeable: (a, b) and (b, a) are one class pair
            mass = weights[a, b] + (weights[b, a] if u < 0 and v < 0 and a != b else 0.0)
            probability[j] = mass / weights.sum()
            if u < 0 and v < 0:
                a, b = min(a, b), max(a, b)
            best[j] = (a if u < 0 else -1, b if v < 0 else -1)
        return best, probability

    # --- single class-marginal evaluations (rl21) -----------------------------------------------------------------

    def joined(self, g, c):
        """Family g's class-marginal log-likelihood with sample c added (its slots as they are)."""
        return float(self.class_values(FAMILY_SAMPLE, [g], [c], 1.0, self.slots[g])[0])

    def left(self, f, c):
        """Family f's class-marginal log-likelihood without its member c (its slots as they are)."""
        return float(self.class_values(FAMILY_SAMPLE, [f], [c], -1.0, self.slots[f])[0])

    def slotted(self, f, other, p):
        """Family f's class-marginal log-likelihood with slot samples ``other`` (-1: latent) and p."""
        return float(self.class_values(FAMILY, [f], slots=(other, p))[0])

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
        if self.class_prior is not None:
            return float(self.class_values(FAMILY, [f], slots=(u, v))[0])
        return family_loglik_parallel(self.S[f], self.S[f], 0.0, self.law(u), self.law(v), 0, self.M)

    def singleton_slot_loglik(self, c, u):
        """One-member family c with slot sample u and one latent slot, kept across both searches of this fit.

        This likelihood depends only on c and u, not on the partition or its family versions.
        """
        key = (int(c), int(u))
        value = self._singleton_slot.get(key)
        if value is None and self.class_prior is not None:
            value = float(self.class_values(SAMPLE, [c], slots=(u, -1))[0])
            self._singleton_slot[key] = value
        if value is None:
            value = family_loglik_parallel(self.C[c], self.C[c], 0.0, self.logpost[u], self.logpi, 0, self.M)
            self._singleton_slot[key] = value
        return value

    def own_sum(self, f):
        """Detach child terms or a saved start sum before an in-place update."""
        if self.shared_sum[f]:
            self.S[f] = self.S[f].copy()
            self.shared_sum[f] = False

    def sum_members(self, f):
        """Member sum(axis=0) in bounded site slices, retaining member and reduction order.

        A singleton's child terms already are its sum (family_terms emits +0 at neutral cells). Borrow that row;
        multi-member rows are owned. Slice only the non-reduced site axis so NumPy adds members in the same order.
        """
        members = self.members[f]
        if len(members) == 1:
            self.S[f] = self.C[members[0]]
            self.shared_sum[f] = True
            return
        width = self.C.shape[2]
        out = np.empty((self.M, width))
        step = max(1, self.parameters.sum_bytes // (len(members) * width * 8))
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
        if self.class_prior is not None:
            scores[:] = self.class_values(FAMILY, live, slots=self.slots[live])
        else:
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
        if self.class_prior is not None:
            out[:] = self.class_values(FAMILY_SAMPLE, families, samples, 1.0, self.slots[families]) - self.ll[families]
        else:
            join_entries(self.S, self.C, self.slots, self.ll, self.logpi, self.logpost, families, samples, out)
        self.join[families, samples] = out
        self.join_at[families, samples] = self.version[families]

    def sync_estimates(self, rows):
        """estimate[rows] current: the join screen's estimate of every sample's join gain, per family version."""
        rows = np.asarray(rows, dtype=np.int64)
        rows = rows[self.estimate_at[rows] != self.version[rows]]
        if not len(rows):
            return
        if self.class_prior is not None:
            # rl21: no screen; the estimates are the exact join gains of every sample, kept as such
            families = np.repeat(rows, self.N)
            exact = self.class_values(
                FAMILY_SAMPLE, families, np.tile(self.everyone, len(rows)), 1.0, self.slots[families]
            ).reshape(len(rows), self.N) - self.ll[rows][:, None]
            self.estimate[rows] = exact
            self.estimate_at[rows] = self.version[rows]
            self.join[rows] = exact
            self.join_at[rows] = self.version[rows][:, None]
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
        if self.class_prior is not None:
            out[:] = -self.ll[families]
            many = ~single
            out[many] += self.class_values(
                FAMILY_SAMPLE, families[many], samples[many], -1.0, self.slots[families[many]]
            )
        else:
            leave_entries(
                self.S, self.C, self.slots, self.ll, self.logpi, self.logpost, families, samples, single, out
            )
        self.leave[samples] = out
        self.leave_at[samples] = self.version[families]

    def leave_gain(self, c):
        """Log-likelihood change of c's family when c leaves it."""
        self.sync_leave([c])
        return self.leave[c]

    def slot_logliks(self, f, allowed):
        """float64 [N]: family f's log-likelihood with its free slot filled by sample p (slot_gains), for every allowed
        p whose value may be within the tolerance of the largest (NaN for the others), kept for f's version. A screen
        (family_kernels.slot_screen) estimates every allowed p; only those whose estimate plus its error bound reaches the
        largest estimate minus its bound, less the tolerance, are evaluated exactly: they include the largest value and
        every value tied with it."""
        key = (int(f), int(self.version[f]))
        out = self._slot_logliks.get(key)
        if out is None and self.class_prior is not None:
            k = int(np.flatnonzero(self.slots[f] == -1)[0])
            candidates = np.flatnonzero(allowed)
            slots = np.stack([np.full(len(candidates), int(self.slots[f][1 - k])), candidates], axis=1)
            out = np.full(self.N, np.nan)
            out[candidates] = self.class_values(FAMILY, np.full(len(candidates), f), slots=slots)
            self._slot_logliks[key] = out
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
            if self.class_prior is not None:
                slots = np.stack([np.full(len(members), int(self.slots[f][1 - k])), members], axis=1)
                values[:] = self.class_values(FAMILY_SAMPLE, np.full(len(members), f), members, -1.0, slots)
            else:
                member_parent_logliks(
                    self.S[f], self.C, members, self.law(int(self.slots[f][1 - k])), self.logpost, values
                )
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

    def prefetch_near_best(self, samples):
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
