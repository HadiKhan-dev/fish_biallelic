"""The family-phase model of one painting component and its MAP by exact block coordinate ascent.

Every sample has two homologues. Homologue k of a sample with a released (Tier-B) slot-k parent is a copy of one of
that parent's homologues (its meiosis chain V switches with the Haldane fraction between bin centres) or, for a run,
not a copy (V = 2: a painting or panel error, conversion or mutation, or a false edge), with a free label then.
Homologue k of a sample without a slot-k parent is a free label path under the painting's copying process. Emissions
are the painting's own tempered log emissions of the sample's unordered label pair. The MAP is found by exact block
coordinate ascent over one block per sample (kernels.block_viterbi): a block's path is applied only when it raises the
objective by more than the gain quantum, so the ascent is monotone and terminates.

Local gains: J is a sum of per-sample terms, each reading only that sample's labels and chains (kernels.row_terms).
A block of sample n changes only n's labels and chains, its children's chains (the gauge) and the copied labels of
its descendants, so its gain is the change in the terms of those samples, O(B) per sample changed instead of O(N B)
for the whole objective, and only their rows are saved and restored. The joint pairing moves use the same local
decomposition. Independent reference tests check it against full objectives;
no reference-only switch enters production science.

Identity junctions: a panel row is a trajectory through assembled blocks, not a certified biological homologue; rows
can exchange founder identity at a block junction. Carriers of one of the two rows then switch between them at that
edge, while carriers of both keep continuous labels whose content changes homologue, which no emission of theirs can
show. Shared ancestry of samples without released parents (an unsequenced parent's own crossover, inherited by its
children) produces the same switch pattern with continuous rows, so the switches do not identify an exchange.
``junction_candidates`` lists edges where at least ``junction_min_samples`` samples switch, by free or not-a-copy
tracks, between the same two labels; the release withholds the phase join of every sample carrying both labels there
(``withheld_joins``, release.py).

Parents' pairings: where both homologues of a sample without a released parent change label at one bin edge, its own
emissions (of unordered pairs) cost the same under both pairings of the old labels with the new, so only its children
decide the pairing. A child copying the arbitrary one absorbs it with a label switch of its other homologue, and no
block can undo that alone: the better configuration changes the parent and the child together. ``solve`` therefore
ends with joint moves (``pairing_pass``) that swap the parent's homologues from that edge on and give every child its
exact block optimum; a move is kept only when it raises J by more than log(set_posterior / (1 - set_posterior)), the
odds at which the release continues a set. Pairings whose alternative is closer than that (``ambiguous_pairings``)
are not decided by the data; the stage breaks the parent's and its children's phase sets there. The margin is a working
value against one optimized alternative (the children are re-solved one block at a time, not marginalized jointly),
not a calibrated posterior of the pairing.
"""

from __future__ import annotations

import threading

import numpy as np

from ...core.errors import InvariantError
from ...core.numerics import ordered_sum
from . import kernels


class LabelModel:
    """One component.

    E float64 [N, S, S, B]: ordered-pair tempered log emissions (symmetric in the pair); par int64 [N, 2]: slot-k
    parent or -1; theta_l, theta_m, alpha, beta float64 [B - 1] per bin edge: label switch, meiosis switch, entry into
    and exit from not-a-copy. ``config`` PhaseParameters. ``reverse``: the tie convention of the ascent (explicit,
    never read from the environment).
    """

    def __init__(self, E, par, theta_l, theta_m, alpha, beta, config, reverse=False, *, decision_audit=None):
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
        else:
            if N:
                raise InvariantError("phase pedigree depth did not converge: directed cycle")
        # pedigree order: ancestors first; within a depth by sample index (an arbitrary convention: reversed under
        # explicit convention reversal)
        self.reverse = bool(reverse)
        self.order = np.asarray(sorted(range(N), key=lambda i: (depth[i], -i if self.reverse else i)), dtype=np.int64)
        rank = np.empty(N, dtype=np.int64)
        rank[self.order] = np.arange(N)
        self.rank = rank
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
        # L2: release sums use ancestors first, then ascending sample index, for both
        # solves and every worker count. Search continues to use its own desc order.
        self.release_desc = [
            np.asarray(sorted(rows, key=lambda i: (depth[i], i)), dtype=np.int64) for rows in self.desc
        ]
        # the samples whose labels or chains a block of n (or a pairing move of n) can change: n and its descendants
        self.affected = [np.concatenate(([n], self.desc[n])).astype(np.int64) for n in range(N)]
        self.mark = np.full(2 * N, -1, dtype=np.int8)
        self.X = np.zeros((N, 2, self.B), dtype=np.int64)
        self.V = np.full((N, 2, self.B), -1, dtype=np.int8)
        for n in range(N):
            for k in range(2):
                if self.par[n, k] >= 0:
                    self.V[n, k] = 0
        self.history = []
        self.settled = False
        # parents' pairings at double label changes (pairing_pass): moves kept, and (parent, edge, gain) of every
        # candidate in the final state
        self.pairing_odds = float(np.log(config.set_posterior / (1.0 - config.set_posterior)))
        self.pairing_moves = []
        self.pairings = []
        self.audit = None
        self.decision_audit = decision_audit
        # Release 4.10: concurrent blocks (solve(map_work=...)). The rows a block of n writes (n and its descendants:
        # labels, chains, its children's gauge) and reads (those rows and their parents: the kernel's copied labels
        # and subtree marks, propagate) fix which blocks commute; ``_waves`` keeps the sequential order of every
        # pair that does not. Each thread has its own subtree mark scratch.
        self.map_work = None
        self.reads = []
        for rows in self.affected:
            linked = np.concatenate((rows, self.par[rows].ravel()))
            self.reads.append(np.unique(linked[linked >= 0]))
        self._scratch = threading.local()
        self._sweep_waves = None
        self._initial_waves = None

    def _waves(self, order):
        """Partition ``order`` into waves of mutually independent blocks, each wave after every block it conflicts
        with: a block goes one wave after the latest wave that wrote a row it reads or read a row it writes (writes
        are among a block's reads). Running the waves in turn, each wave's blocks in any order or at once, applies
        every conflicting pair in its sequential order and so reproduces the sequential sweep exactly."""
        last_write = np.full(self.N, -1, dtype=np.int64)
        last_read = np.full(self.N, -1, dtype=np.int64)
        waves = []
        for n in order:
            n = int(n)
            writes, reads = self.affected[n], self.reads[n]
            wave = 1 + max(int(last_write[reads].max(initial=-1)), int(last_read[writes].max(initial=-1)))
            if wave == len(waves):
                waves.append([])
            waves[wave].append(n)
            last_write[writes] = np.maximum(last_write[writes], wave)
            last_read[reads] = np.maximum(last_read[reads], wave)
        return [np.asarray(members, dtype=np.int64) for members in waves]

    def _mark(self):
        """The subtree scratch for this thread (int8 [2N], -1 outside a block call)."""
        if self.map_work is None:
            return self.mark
        mark = getattr(self._scratch, "mark", None)
        if mark is None:
            mark = self._scratch.mark = np.full(2 * self.N, -1, dtype=np.int8)
        return mark

    def _concurrent(self, function, members):
        """function(n) for every member of one wave: on the map when a wave has more than one block."""
        if len(members) > 1:
            return self.map_work(function, members.tolist())
        return [function(int(n)) for n in members]

    # ------------------------------------------------------------------------------------------- objective
    def objective(self):
        return kernels.objective(
            self.E,
            self.X,
            self.V,
            self.par,
            self.lstay,
            self.lmove,
            self.mstay,
            self.mmove,
            self.ain,
            self.astay,
            self.bout,
            self.init_copy,
            self.init_bottom,
        )

    def _terms(self, X, V, rows):
        return kernels.row_terms(
            self.E,
            X,
            V,
            self.par,
            rows,
            self.lstay,
            self.lmove,
            self.mstay,
            self.mmove,
            self.ain,
            self.astay,
            self.bout,
            self.init_copy,
            self.init_bottom,
        )

    def _gain(self, rows, X0, V0):
        """(change of J, positions in ``rows`` of the samples whose labels or chains changed) from the saved rows X0,
        V0 of samples ``rows`` to the current state; every other sample's terms are unchanged. Each changed sample's
        terms are differenced before summing (math.fsum), so unchanged samples add exactly nothing."""
        X1, V1 = self.X[rows], self.V[rows]
        changed = np.flatnonzero((X1 != X0).any(axis=(1, 2)) | (V1 != V0).any(axis=(1, 2)))
        if not len(changed):
            return 0.0, changed
        sub = rows[changed]
        after = self._terms(X1[changed], V1[changed], sub)
        before = self._terms(X0[changed], V0[changed], sub)
        return ordered_sum(after - before), changed

    def _restore(self, rows, X0, V0, changed):
        sub = rows[changed]
        self.X[sub] = X0[changed]
        self.V[sub] = V0[changed]

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
        return kernels.block_viterbi(
            n,
            kinds,
            self.parents(n),
            self.E,
            self.X,
            self.V,
            self.par,
            desc,
            child_tracks,
            self._mark(),
            self.S,
            self.lstay,
            self.lmove,
            self.mstay,
            self.mmove,
            self.ain,
            self.astay,
            self.bout,
            self.init_copy,
            self.init_bottom,
            cur[0],
            cur[1],
            self.config.tie_quantum,
            self.reverse,
        )

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

        def own(n):
            path0, path1, _, _ = self.block(n, free, use_children=False)
            self.X[n, 0] = path0
            self.X[n, 1] = path1

        def oriented(n):
            kinds = self.kinds(n)
            path0, path1, gpath, _ = self.block(n, kinds, use_children=False)
            self.apply(n, path0, path1, np.zeros_like(gpath), kinds)

        if self.map_work is not None:
            # Release 4.10: the free blocks read and write only their own sample; the oriented ones run in waves.
            self._concurrent(own, np.arange(self.N, dtype=np.int64))
            if self._initial_waves is None:
                self._initial_waves = self._waves([n for n in self.order if np.any(self.par[n] >= 0)])
            for members in self._initial_waves:
                self._concurrent(oriented, members)
            return
        for n in range(self.N):
            own(n)
        for n in self.order:
            if np.all(self.par[n] < 0):
                continue
            oriented(n)

    def _sweep_block(self, n):
        """One block of a sweep without audits: apply it when its local gain exceeds the gain quantum, else restore
        n's and its descendants' rows. Returns whether it was applied."""
        rows = self.affected[n]
        X0, V0 = self.X[rows], self.V[rows]
        kinds = self.kinds(n)
        path0, path1, gpath, _ = self.block(n, kinds)
        self.apply(n, path0, path1, gpath, kinds)
        gain, changed = self._gain(rows, X0, V0)
        if gain > self.config.gain_quantum:
            return True
        self._restore(rows, X0, V0, changed)
        return False

    def sweeps(self):
        """Block coordinate ascent from the current state until a sweep applies no block (``settled``) or
        ``max_sweeps`` sweeps have run (every applied block raised the objective, so the state is still valid). A
        block is applied when its local gain (_gain over n and its descendants) exceeds the gain quantum; J itself is
        evaluated once per sweep, for ``history``."""
        quantum = self.config.gain_quantum
        accept = lambda change: change > quantum  # noqa: E731
        J = self.objective()
        self.history.append(float(J))
        concurrent = self.map_work is not None and self.decision_audit is None and self.audit is None
        if concurrent and self._sweep_waves is None:
            self._sweep_waves = self._waves(self.order)
        for sweep_index in range(self.config.max_sweeps):
            accepted = 0
            for members in self._sweep_waves if concurrent else ():
                accepted += sum(self._concurrent(self._sweep_block, members))
            for n in () if concurrent else self.order:
                n = int(n)
                rows = self.affected[n]
                X0, V0 = self.X[rows], self.V[rows]
                kinds = self.kinds(n)
                path0, path1, gpath, _ = self.block(n, kinds)
                self.apply(n, path0, path1, gpath, kinds)
                gain, changed = self._gain(rows, X0, V0)
                taken = accept(gain)
                if self.decision_audit is not None:
                    self.decision_audit.difference("gain", gain, quantum, taken)
                if self.audit is not None:
                    self.audit.gains.append([n, float(gain), float(quantum)])
                    self.audit.emit("phase.gain", taken, gain=float(gain), threshold=float(quantum), sample=n)
                if taken:
                    accepted += 1
                else:
                    self._restore(rows, X0, V0, changed)
            J = self.objective()
            self.history.append(float(J))
            if self.audit is not None:
                self.audit.sweeps.append(
                    dict(
                        index=int(sweep_index), accepted=accepted, objective=float(J), X=self.X.copy(), V=self.V.copy()
                    )
                )
            if accepted == 0:
                if self.audit is not None:
                    self.audit.cap_events.append(
                        dict(
                            loop="sweeps",
                            used=int(sweep_index) + 1,
                            limit=self.config.max_sweeps,
                            improvements=accepted,
                        )
                    )
                    self.audit.emit(
                        "ph1.solve.caps",
                        False,
                        loop="sweeps",
                        used=int(sweep_index) + 1,
                        limit=self.config.max_sweeps,
                        improvements=accepted,
                    )
                self.settled = True
                return J
        if self.audit is not None:
            self.audit.cap_events.append(
                dict(loop="sweeps", used=self.config.max_sweeps, limit=self.config.max_sweeps, improvements=accepted)
            )
            self.audit.emit(
                "ph1.solve.caps",
                True,
                loop="sweeps",
                used=self.config.max_sweeps,
                limit=self.config.max_sweeps,
                improvements=accepted,
            )
        self.settled = False
        return J

    def solve(self, *, trace=False, map_work=None):
        """The capped ascent below (_solve); with ``map_work(function, items)`` (ordered results, one thread per call)
        its initial blocks and sweep blocks run in conflict-free waves (release 4.10) and reach the same state."""
        self.map_work = None if trace or self.decision_audit is not None else map_work
        try:
            return self._solve(trace=trace)
        finally:
            self.map_work = None

    def _solve(self, *, trace=False):
        """Perform one capped global ascent and retain its final X/V for release.

        Initialise every sample, then alternate whole-pedigree block sweeps and
        joint parent/descendant pairing passes. Each sweep round and the pairing
        loop has the unchanged max_sweeps cap. A cap leaves a valid improved
        state; a capped pairing loop runs the final keep=False pass so ambiguity
        is measured at that final state. This is not independent family solving.
        The component runs default and mirror independently, then releases their
        agreement; neither solve supplies initial conditions to the other.
        """
        if trace:
            from .diagnostics import SolveTrace

            self.audit = SolveTrace()
        self.initialise()
        if self.audit is not None:
            self.audit.initial.update(X=self.X.copy(), V=self.V.copy(), objective=float(self.objective()))
        J = self.sweeps()
        for _ in range(self.config.max_sweeps):
            if not self.pairing_pass():
                break  # the pass's gains were measured in the final state
            J = self.sweeps()
        else:
            if self.audit is not None:
                self.audit.final_nonkeeping_pass = True
            self.pairing_pass(keep=False)
        return J

    # ------------------------------------------------------------------------------------------- parents' pairings
    def pairing_candidates(self):
        """(p, j): every sample p without a released parent and with children (pedigree order) and every bin edge j
        (ascending) where both of p's homologues change label."""
        out = []
        for p in self.order:
            p = int(p)
            if np.any(self.par[p] >= 0) or not len(self.child_tracks[p]):
                continue
            x0, x1 = self.X[p, 0], self.X[p, 1]
            out.extend((p, int(j)) for j in np.flatnonzero((x0[1:] != x0[:-1]) & (x1[1:] != x1[:-1])) + 1)
        return out

    def children_of(self, p):
        """p's children (samples with a homologue linked to p), in pedigree order."""
        return sorted({int(t // 2) for t in self.child_tracks[p]}, key=lambda c: self.rank[c])

    def _pairing_move(self, p, j):
        """pairing_move, also returning the saved rows of p and its descendants (the only samples it can change)
        and the positions of those that changed: (gain, rows, X0, V0, changed)."""
        rows = self.affected[p]
        X0, V0 = self.X[rows], self.V[rows]
        self.X[p, :, j:] = self.X[p, ::-1, j:].copy()
        self.propagate(self.desc[p])
        for c in self.children_of(p):
            kinds = self.kinds(c)
            path0, path1, gpath, _ = self.block(c, kinds)
            self.apply(c, path0, path1, gpath, kinds)
        gain, changed = self._gain(rows, X0, V0)
        return gain, rows, X0, V0, changed

    def pairing_pass(self, keep=True):
        """One pass over pairing_candidates: each move is kept when it raises J by more than ``pairing_odds`` (and
        ``keep``), else undone. Records (p, j, gain) of every candidate in ``pairings``; returns the number kept. A move
        that leaves every label and chain as it was (the parent's swapped suffix is the same configuration, e.g. a
        homozygous pair on both sides) is no competing pairing and is not recorded."""
        if self.audit is not None:
            self.audit.passes.append(
                dict(keep=bool(keep), before=float(self.objective()), X_before=self.X.copy(), V_before=self.V.copy())
            )
        kept = 0
        records = []
        margin = max(self.pairing_odds, self.config.gain_quantum)
        for p, j in self.pairing_candidates():
            gain, rows, X0, V0, changed = self._pairing_move(p, j)
            if not len(changed):
                continue  # the same state: nothing to choose between
            records.append((p, j, float(gain)))
            taken = keep and gain > margin
            if keep and self.decision_audit is not None:
                self.decision_audit.difference("pairing", gain, margin, gain > margin)
            if self.audit is not None:
                self.audit.emit(
                    "phase.pairing", taken, gain=float(gain), threshold=float(margin), keep=bool(keep), parent=p, bin=j
                )
            if taken:
                kept += 1
                self.pairing_moves.append((p, j, float(gain)))
            else:
                self._restore(rows, X0, V0, changed)
        self.pairings = records
        if self.audit is not None:
            self.audit.passes[-1].update(
                kept=int(kept),
                after=float(self.objective()),
                records=list(self.pairings),
                X_after=self.X.copy(),
                V_after=self.V.copy(),
            )
            if keep:
                used = sum(row["keep"] for row in self.audit.passes)
                self.audit.cap_events.append(
                    dict(loop="pairing", used=used, limit=self.config.max_sweeps, improvements=int(kept))
                )
                self.audit.emit(
                    "ph1.solve.caps",
                    bool(used >= self.config.max_sweeps and kept > 0),
                    loop="pairing",
                    used=used,
                    limit=self.config.max_sweeps,
                    improvements=int(kept),
                )
        return kept

    def ambiguous_pairings(self):
        """(p, j) of the final state whose other pairing is within ``pairing_odds`` of J: not decided by the data."""
        result = []
        for p, j, gain in self.pairings:
            ambiguous = gain > -self.pairing_odds
            if self.decision_audit is not None:
                self.decision_audit.difference("ambiguity", gain, -self.pairing_odds, ambiguous)
            if self.audit is not None:
                self.audit.emit(
                    "phase.ambiguity", ambiguous, gain=float(gain), threshold=-self.pairing_odds, parent=p, bin=j
                )
            if ambiguous:
                result.append((p, j))
        return result

    # ------------------------------------------------------------------------------------------- junctions


__all__ = ["LabelModel"]
