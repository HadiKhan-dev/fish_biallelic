"""The family-phase model of one painting component and its MAP by exact block coordinate ascent.

Every sample has two homologues. Homologue k of a sample with a released (Tier-B) slot-k parent is a copy of one of
that parent's homologues (its meiosis chain V switches with the Haldane fraction between bin centres) or, for a run,
not a copy (V = 2: a painting or panel error, conversion or mutation, or a false edge), with a free label then.
Homologue k of a sample without a slot-k parent is a free label path under the painting's copying process. Emissions
are the painting's own tempered log emissions of the sample's unordered label pair. The MAP is found by exact block
coordinate ascent over one block per sample (kernels.block_viterbi): a block's path is applied only when the full
objective rises by more than the gain quantum, so the ascent is monotone and terminates.

Identity junctions: a panel row is a trajectory through assembled blocks, not a certified biological homologue; rows
can exchange founder identity at a block junction. Carriers of one of the two rows then switch between them at that
edge, while carriers of both keep continuous labels whose content changes homologue, which no emission of theirs can
show. Shared ancestry of samples without released parents (an unsequenced parent's own crossover, inherited by its
children) produces the same switch pattern with continuous rows, so the switches do not identify an exchange.
``junction_candidates`` lists edges where at least ``junction_min_samples`` samples switch, by free or not-a-copy
tracks, between the same two labels; the release withholds the phase join of every sample carrying both labels there
(``withheld_joins``, release.py).
"""
from __future__ import annotations

import numpy as np

from ..maths.ties import reverse_conventions
from . import kernels


def haldane(morgans):
    return 0.5 * (1.0 - np.exp(-2.0 * np.asarray(morgans, dtype=np.float64)))


class LabelModel:
    """One component.

    E float64 [N, S, S, B]: ordered-pair tempered log emissions (symmetric in the pair); par int64 [N, 2]: slot-k
    parent or -1; theta_l, theta_m, alpha, beta float64 [B - 1] per bin edge: label switch, meiosis switch, entry into
    and exit from not-a-copy. ``config`` FamilyPhaseConfig. ``reverse``: the tie convention of the ascent (None: the
    run's, maths.ties.reverse_conventions).
    """

    def __init__(self, E, par, theta_l, theta_m, alpha, beta, config, reverse=None):
        self.E = np.ascontiguousarray(E, dtype=np.float64)
        self.N, self.S, _, self.B = self.E.shape
        S = self.S
        self.config = config
        self.par = np.asarray(par, dtype=np.int64)
        self.lstay = np.log1p(-theta_l)
        self.lmove = np.log(theta_l / max(1, S - 1))
        self.mstay = np.log1p(-theta_m)
        self.mmove = np.log(theta_m)
        self.ain = np.log(alpha)
        self.astay = np.log1p(-alpha)
        self.bout = np.log(beta)
        # initial not-a-copy mass of a linked homologue: the stationary occupancy of the entry/exit process
        occupancy = config.not_copy_rate_per_bp * config.not_copy_mean_bp
        bottom = occupancy / (1.0 + occupancy)
        self.init_copy = float(np.log((1.0 - bottom) / 2.0))
        self.init_bottom = float(np.log(bottom / S))
        N = self.N
        children = [[] for _ in range(N)]
        for c in range(N):
            for k in range(2):
                if self.par[c, k] >= 0:
                    children[self.par[c, k]].append(2 * c + k)
        self.child_tracks = [np.asarray(sorted(t), dtype=np.int64) for t in children]
        depth = np.zeros(N, dtype=np.int64)
        for _ in range(N):
            changed = False
            for c in range(N):
                for k in range(2):
                    p = self.par[c, k]
                    if p >= 0 and depth[c] < depth[p] + 1:
                        depth[c] = depth[p] + 1
                        changed = True
            if not changed:
                break
        # pedigree order: ancestors first; within a depth by sample index (an arbitrary convention: reversed under
        # maths.ties.reverse_conventions)
        self.reverse = reverse_conventions() if reverse is None else bool(reverse)
        self.order = np.asarray(sorted(range(N), key=lambda i: (depth[i], -i if self.reverse else i)),
                                dtype=np.int64)
        rank = np.empty(N, dtype=np.int64)
        rank[self.order] = np.arange(N)
        self.desc = []
        for n in range(N):
            seen, stack = set(), [n]
            while stack:
                u = stack.pop()
                for t in children[u]:
                    c = int(t // 2)
                    if c not in seen:
                        seen.add(c)
                        stack.append(c)
            self.desc.append(np.asarray(sorted(seen, key=lambda i: rank[i]), dtype=np.int64))
        self.mark = np.full(2 * N, -1, dtype=np.int8)
        self.X = np.zeros((N, 2, self.B), dtype=np.int64)
        self.V = np.full((N, 2, self.B), -1, dtype=np.int8)
        for n in range(N):
            for k in range(2):
                if self.par[n, k] >= 0:
                    self.V[n, k] = 0
        self.history = []
        self.settled = False

    # ------------------------------------------------------------------------------------------- objective
    def objective(self):
        return kernels.objective(self.E, self.X, self.V, self.par, self.lstay, self.lmove, self.mstay, self.mmove,
                                 self.ain, self.astay, self.bout, self.init_copy, self.init_bottom)

    def kinds(self, n):
        return np.asarray([1 if self.par[n, k] >= 0 else 0 for k in range(2)], dtype=np.int64)

    def parents(self, n):
        return np.asarray([max(0, self.par[n, k]) for k in range(2)], dtype=np.int64)

    # ------------------------------------------------------------------------------------------- blocks
    def _current(self, n, kinds):
        cur = []
        for k in range(2):
            if kinds[k] == 1:
                v = self.V[n, k].astype(np.int64)
                cur.append(np.where(v < 2, v, 2 + self.X[n, k]).astype(np.int32))
            else:
                cur.append(self.X[n, k].astype(np.int32))
        return cur

    def block(self, n, kinds=None, use_children=True):
        kinds = self.kinds(n) if kinds is None else kinds
        cur = self._current(n, kinds)
        child_tracks = self.child_tracks[n] if use_children else np.zeros(0, dtype=np.int64)
        desc = self.desc[n] if use_children else np.zeros(0, dtype=np.int64)
        return kernels.block_viterbi(n, kinds, self.parents(n), self.E, self.X, self.V, self.par, desc,
                                     child_tracks, self.mark, self.S, self.lstay, self.lmove, self.mstay,
                                     self.mmove, self.ain, self.astay, self.bout, self.init_copy, self.init_bottom,
                                     cur[0], cur[1], self.config.tie_quantum, self.reverse)

    def tables(self, n):
        """(em, T0, T1, TG, labs0, labs1), kinds: the block's factors for kernels.forward_backward."""
        kinds = self.kinds(n)
        return kernels.block_tables(n, kinds, self.parents(n), self.E, self.X, self.V, self.par, self.desc[n],
                                    self.child_tracks[n], self.mark, self.S, self.lstay, self.lmove, self.mstay,
                                    self.mmove, self.ain, self.astay, self.bout, self.init_copy,
                                    self.init_bottom), kinds

    def apply(self, n, path0, path1, gpath, kinds):
        for k, path in ((0, path0), (1, path1)):
            if kinds[k] == 1:
                p = self.par[n, k]
                self.V[n, k] = np.where(path < 2, path, 2).astype(np.int8)
                self.X[n, k] = np.where(path >= 2, path - 2, np.where(path == 1, self.X[p, 1], self.X[p, 0]))
            else:
                self.X[n, k] = path
        if np.any(gpath == 1):
            for t in self.child_tracks[n]:
                c, k = int(t // 2), int(t % 2)
                v = self.V[c, k]
                flip = (gpath == 1) & (v < 2)
                self.V[c, k] = np.where(flip, v ^ 1, v)
        self.propagate(self.desc[n])

    def propagate(self, order):
        """Labels of copied homologues from their parents (``order`` topological)."""
        X, V, par = self.X, self.V, self.par
        for n in order:
            for k in range(2):
                p = par[n, k]
                if p < 0:
                    continue
                v = V[n, k]
                src = np.where(v == 1, X[p, 1], X[p, 0])
                X[n, k] = np.where(v < 2, src, X[n, k])

    # ------------------------------------------------------------------------------------------- ascent
    def initialise(self):
        """Every sample by its own painting Viterbi (both homologues free, no relatives), then samples with a
        released parent, ancestors first, without their children's terms: parental origin orients them."""
        free = np.zeros(2, dtype=np.int64)
        for n in range(self.N):
            path0, path1, _, _ = self.block(n, free, use_children=False)
            self.X[n, 0] = path0
            self.X[n, 1] = path1
        for n in self.order:
            if np.all(self.par[n] < 0):
                continue
            kinds = self.kinds(n)
            path0, path1, gpath, _ = self.block(n, kinds, use_children=False)
            self.apply(n, path0, path1, np.zeros_like(gpath), kinds)

    def sweeps(self):
        """Block coordinate ascent from the current state until a sweep applies no block (``settled``) or
        ``max_sweeps`` sweeps have run (every applied block raised the objective, so the state is still valid)."""
        J = self.objective()
        quantum = self.config.gain_quantum
        self.history.append(float(J))
        for _ in range(self.config.max_sweeps):
            accepted = 0
            for n in self.order:
                X0, V0 = self.X.copy(), self.V.copy()
                kinds = self.kinds(n)
                path0, path1, gpath, _ = self.block(n, kinds)
                self.apply(n, path0, path1, gpath, kinds)
                J1 = self.objective()
                if J1 > J + quantum:
                    J = J1
                    accepted += 1
                else:
                    self.X[:], self.V[:] = X0, V0
            self.history.append(float(J))
            if accepted == 0:
                self.settled = True
                return J
        self.settled = False
        return J

    def solve(self):
        self.initialise()
        return self.sweeps()

    # ------------------------------------------------------------------------------------------- junctions
    def junction_candidates(self):
        """(samples, edge bin j, a, b), strongest first (ties: later edge, then larger labels first): bin edges where
        free or not-a-copy tracks of at least ``junction_min_samples`` distinct samples switch between labels a and b
        (a copied homologue's switch is its parent's or a crossover, so it is not counted)."""
        X, V, par = self.X, self.V, self.par
        who = {}
        for n in range(self.N):
            for k in range(2):
                x = X[n, k]
                for j in np.flatnonzero(x[1:] != x[:-1]) + 1:
                    if par[n, k] >= 0 and V[n, k, j] < 2 and V[n, k, j - 1] < 2:
                        continue
                    a, b = int(x[j - 1]), int(x[j])
                    who.setdefault((int(j), min(a, b), max(a, b)), set()).add(n)
        return sorted(((len(s), j, a, b) for (j, a, b), s in who.items()
                       if len(s) >= self.config.junction_min_samples), reverse=True)

    def withheld_joins(self, candidates):
        """{sample: sorted bins}: at each candidate junction (j, a, b), the samples whose two homologues carry a and b
        on both sides of edge j and whose relative phase across j is not pinned. Their phase across j depends on
        whether the two rows continue there, and a count of relatives is no evidence of that (a relative without
        reads on one flank pins nothing). A sample is pinned only when a linked homologue copies the same parental
        homologue on both sides of j and the parent's own reads pin that homologue's content against every pair
        alternative (``homologue_pinned``); the sample's own pair then fixes its other homologue. Children are not
        used: a child's copy rules out the alternative only through a meiosis switch at exactly edge j, whose odds
        depend on the interval (none at a protected reset) and on uncertainty the children share."""
        X, V, par = self.X, self.V, self.par
        out = {}
        for _, j, a, b in candidates:
            pair = {a, b}
            for n in range(self.N):
                if {int(X[n, 0, j - 1]), int(X[n, 1, j - 1])} != pair or {int(X[n, 0, j]), int(X[n, 1, j])} != pair:
                    continue
                pinned = False
                for k in range(2):
                    p, h = int(par[n, k]), int(V[n, k, j])
                    if p >= 0 and h < 2 and int(V[n, k, j - 1]) == h and self.homologue_pinned(p, h, j, a, b):
                        pinned = True
                        break
                if not pinned:
                    out.setdefault(n, set()).add(int(j))
        return {n: sorted(v) for n, v in out.items()}

    def homologue_pinned(self, p, h, j, a, b):
        """Whether homologue h of sample p has known content across edge j from p's own emissions, against the
        exchange of labels a and b (``_flanks_pinned``: the other junction label on h, with any partner). A flank
        where it carries neither label is unaffected by the junction; a carrier of both labels is never pinned (its
        pair is symmetric under the exchange)."""
        X = self.X
        swap = {a: b, b: a}
        for t in (j - 1, j):
            if {int(X[p, h, t]), int(X[p, 1 - h, t])} == {a, b}:
                return False
        return self._flanks_pinned(p, h, j, (swap, swap), unaffected_passes=True)

    def _flanks_pinned(self, p, h, j, alternatives, unaffected_passes):
        """Both flanks of edge j: p's pair (x, y), with x on homologue h at the flank's first bin, beats every pair
        that puts the alternative label z = alternatives[flank][x] on homologue h, whatever the partner (per bin, the
        best partner label), over up to ``junction_pin_bins`` bins while p's labels stay put, by at least
        ``junction_pin_nats``. The partner is free because a pair alternative can fit the same unphased reads (two
        rows exchanged together): holding it fixed would mistake that ambiguity for evidence. A flank with no
        alternative to x (x not in the map, or mapped to itself) passes when ``unaffected_passes`` and fails
        otherwise."""
        X, E, B = self.X, self.E, self.B
        window = self.config.junction_pin_bins
        flanks = ((j - 1, max(-1, j - 1 - window), -1), (j, min(B, j + window), 1))
        for (start, stop, step), alternative in zip(flanks, alternatives):
            x, y = int(X[p, h, start]), int(X[p, 1 - h, start])
            z = alternative.get(x, x)
            if z == x:
                if unaffected_passes:
                    continue
                return False
            llr = 0.0
            for t in range(start, stop, step):
                if int(X[p, h, t]) != x or int(X[p, 1 - h, t]) != y:
                    break
                llr += E[p, x, y, t] - E[p, z, :, t].max()
            if llr < self.config.junction_pin_nats:
                return False
        return True
