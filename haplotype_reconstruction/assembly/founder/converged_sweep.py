"""Converged founder refinement: Gauss-Seidel window/exchange sweeps between converged count decisions.

Objective (unchanged): the canonical full-site F of founder/scoring.py over panels of
original prepared local rows (K x B row matrices), with the fixed complete-site mask and
SWITCH_PENALTY. Count decisions use the existing per-founder cost through the pipeline's
count rule, count_criterion.decide_with_sequence (the assembly join's rule).

Fixed-K fit C(X): repeat sweeps of
  (i) exact normalised forward/backward block messages of the panel (O(S n Q); a search
      updates its previous sweep's messages in place, recomputing per sample only the
      blocks the last pass changed and those whose start message changed; computed over
      consecutive sample ranges);
  (ii) proposals computed from the panel at the start of the sweep, independently and
      in parallel (a pure function of the panel; a window whose inputs are unchanged since
      the search last solved it reuses that solve): for every founder and every 8-block
      window (stride 4, plus the window ending at the last block) the canonical exact
      maximiser of the binned window problem (window_dp.py); where an L1 context exists,
      for every founder and every window of 4 consecutive L1 spans (stride 2) the same
      over span tuples (the founders' current tuples plus the L1 context paths, as
      production's L1-macro moves);
  (iii) one exact Gauss-Seidel pass in block order, direction alternating between sweeps:
      at every junction every founder-pair suffix exchange is scored exactly from the
      carried forward and stored backward messages and the best is applied while its gain
      exceeds TOL; then the proposals anchored there are re-scored exactly against the
      CURRENT panel, best first, each applied while its gain exceeds TOL; coarse moves also
      pass a guard (below); then the forward message advances one block;
  (iv) the canonical F is recomputed and must equal the start F plus the accepted gains.
Best of two: every fixed-K fit runs this search twice from the same start, concurrently,
once with its first pass forward and once with its first pass reversed (both then
alternate; their common sweep-0 messages and proposals are computed once). Each is a
deterministic local search; the fit keeps the one with the higher exact F (exact ties:
the forward-first one), so at a fixed start it never ends below the forward-first
search, and reports both (record["best_of_two"]).

Coarse-move guard (config.coarse_move_guard). A long founder edit can raise F by letting
several samples drop the switch of a crossover they share (descendants of one meiosis
carry the same breakpoint), turning the founder into an artificial recombinant.
"genotype_fit" (production's guard) refuses every coarse move that lowers the genotype fit
E = F + penalty x switches of the samples' maximum paths, whatever its switch savings
are. "shared_junction" refuses such a move only if its F gain does not exceed TOL once
every saved switch at a shared one-way junction is uncredited: dF - penalty x N_shared <=
TOL, N_shared = sum over samples of min(switches the sample saves, its removed switch
events at shared one-way junctions). A removed event is a founder transition u -> v of
one homologue; it is one-way unless it leaves the edited founder and the same path later
rejoins it (an excursion of the founder's own carrier, the signature of a row that is
wrong there: descendants sharing both breakpoints of a short double crossover is
improbable under crossover interference); it is shared when another sample's one-way
removed event is the same transition with an overlapping evidence interval (the stretch
between the sample's nearest informative sites where u and v differ). Moves that do not
lower E, and moves whose gain comes from switch savings at unshared junctions or from
excursions, pass. Removed events come from exact tracebacks of the samples' lexicographic
maximum paths with and without the move (sweep_kernels.path_switch_events).
A sweep that accepts nothing stops the fit. Every proposal of that sweep was computed
against the final panel and every move of the stated move set was scored exactly
against it: the result is a certified local optimum of F under that move set (F is
non-convex; no global optimum is claimed; the window proposals are optima of the binned
surrogate, so the certificate is relative to them). The sweep backstop only marks a fit
non-converged, never silently.

Count rule (the join's, panel_count): a decision between a larger panel X+ and a smaller
X- keeps X+ iff the effective-carrier test passes on the total per-sample gain
d_s = F_s(X+) - F_s(X-) AND on its sequence part q_s = F_s(X+) - F_s(X+^sigma). The shadow
X+^sigma replaces every cell (block, row) of X+ that a reference panel does not use by the
reference's row nearest to it at that block (panel_count.shadow_path); the reference is
X+ minus the decided founder f for a deletion (only f's private cells change) and the
incumbent X- for an addition (every cell the refit added). A founder without a private
cell (every cell another founder uses at that block) owns no sequence: q = 0 exactly when
X+ holds one. Otherwise q is the exact shadow difference; unlike the join, refinement
takes no near-identity shortcut (a founder within panel_count.NEAR_IDENTITY of another
over the span keeps the evidence of its differing cells), because its last level is
final and no later level can re-split a merged pair. Switch savings alone never justify a
founder.

Converged count pass: X_K = C(start); deletion rounds (founders without a private cell
are the round's candidates when there are any, as in the join; otherwise every founder. The r = COUNT_RANKED_REFITS top-ranked candidates, ranked by the criterion's point
margin on the total gain of their exact unrefitted per-sample scores, are refitted to their
own stops concurrently and the best-F of them is decided; with at most r candidates every
one is refitted; an optimistic bound may skip a round only when it proves the decision for
every candidate); then,
at the final count step only (the last level, in place of production's count-up),
duplicate-seed addition rounds (the r seeds of largest exact copy best-response gain,
every seed when K <= r, refitted concurrently, the best-F of them decided), each
accepted addition followed by one deletion round that must reject, otherwise the call
is a reported count cycle, resolved to the smaller K (the deletion round stands) with
the better-F of its two K panels; it is flagged undecided when the deletion round's call
is ambiguous and the pass reached K by a decided deletion from K + 1
(count_criterion.pre_cycle_count). At most r converged fits per
count decision. The count decisions
choose K; the panel kept at that K is the best-F converged fit computed at K during the
pass (it differs from the last panel only when the pass returned to a K it had held with
a better panel; the replacement and the restored panel's adjacent-K comparisons are
reported, and the call is flagged ambiguous when the restored panel fails its own
deletion test). Earlier levels make deletion
decisions only, as production's per-level count pass: their K is the working set of
the next hierarchy join, and the reported founder count is decided once, between
converged whole-component fits. K stays within [2, incoming K x (1 +
count_round_backstop)] (deletions stop at two founders as production's count pass does;
the upper bound is a reported backstop).
"""
from __future__ import annotations

import threading
import time
import warnings

import numpy as np
from numba.typed import List

from . import (count_bound as founder_count_bound, count_criterion, scoring as founder_scoring,
               sweep_kernels as kernels, window_dp)
from .. import panel_count
from .candidates import POLL_SECONDS, completed_candidates
from .packing import emission_arrays, gather_macro
from .workspace import resolve_threads, sample_ranges
from ...core import parallel

TOL = 1e-4
FINE_WINDOW = 8
FINE_STRIDE = 4
COARSE_WINDOW = 4
COARSE_STRIDE = 2
GUARD_TOLERANCE = 1e-6
WEAK_JUNCTION_MARGIN = founder_scoring.SWITCH_PENALTY
# Forward messages of the current panel computed ahead of the pass position at a time.
AHEAD_BLOCKS = 16
# The compiled acceptance pass (_gs_pass) runs in nogil calls of JUNCTION_BATCH junctions,
# evaluated one by one, and re-reads its thread budget before each call.
JUNCTION_BATCH = 16


def window_templates(blocks, groups):
    """Fine windows over blocks and coarse windows over L1 spans (unit indices)."""
    tasks = []
    if FINE_WINDOW >= blocks:
        starts = [0]
    else:
        starts = sorted(set(range(0, blocks - FINE_WINDOW + 1, FINE_STRIDE)) | {blocks - FINE_WINDOW})
    tasks.extend(("fine", a, min(blocks, a + FINE_WINDOW)) for a in starts)
    if groups is not None:
        units = len(groups)
        if COARSE_WINDOW >= units:
            starts = [0]
        else:
            starts = sorted(set(range(0, units - COARSE_WINDOW + 1, COARSE_STRIDE))
                            | {units - COARSE_WINDOW})
        tasks.extend(("coarse", i, min(units, i + COARSE_WINDOW)) for i in starts)
    return tasks


class Orientation:
    """Leaves, log evidence and block offsets in forward or reversed site order."""

    def __init__(self, workspace, reverse):
        sites = int(workspace.offsets[-1])
        self.reverse = reverse
        if reverse:
            self.logs = kernels.reverse_sites(workspace.logs)
            self.offsets = np.ascontiguousarray(sites - workspace.offsets[::-1], np.int64)
            self.leaves = List([np.ascontiguousarray(leaf[:, ::-1])
                                for leaf in list(workspace.leaves)[::-1]])
        else:
            self.logs = workspace.logs
            self.offsets = np.ascontiguousarray(workspace.offsets, np.int64)
            self.leaves = workspace.leaves
        self.blocks = len(self.offsets) - 1

    def haps(self, panel):
        return founder_scoring.selected_alleles(self.leaves, self.offsets,
                                                np.ascontiguousarray(panel, np.int64))

    def set_rows(self, haps, founder, start, rows):
        for i, row in enumerate(rows):
            block = start + i
            haps[founder, self.offsets[block]:self.offsets[block + 1]] = self.leaves[block][row]


_GUARD_CODES = {None: kernels.GUARD_NONE, "genotype_fit": kernels.GUARD_GENOTYPE_FIT,
                "shared_junction": kernels.GUARD_SHARED_JUNCTION}
_KIND_CODES = {"fine": kernels.MOVE_FINE, "coarse": kernels.MOVE_COARSE}
_KIND_NAMES = {kernels.MOVE_FINE: "fine", kernels.MOVE_COARSE: "coarse"}
_INT_ACC = ("fine", "coarse", "exchange", "evaluated", "false_positive", "guard_rejected",
            "guard_shared_tests", "guard_relaxed", "guard_shared_switches", "guard_traced_blocks",
            "chain_blocks", "ahead_blocks", "changed_blocks")


def _candidate_arrays(props, blocks):
    """The proposals anchored at every junction as arrays (kernels.segment_pass): start
    offsets per junction, kind, end block, founder label, row offsets and flat rows."""
    start = np.zeros(blocks + 1, np.int64)
    kind, end, label, row_start, rows = [], [], [], [], []
    total = 0
    for q in range(blocks):
        start[q] = len(kind)
        for name, c, k0, r in props.get(q, ()):
            kind.append(_KIND_CODES[name])
            end.append(c)
            label.append(k0)
            row_start.append(total)
            rows.append(np.asarray(r, np.int64))
            total += len(r)
    start[blocks] = len(kind)
    return (start, np.asarray(kind, np.int64), np.asarray(end, np.int64),
            np.asarray(label, np.int64), np.asarray(row_start, np.int64),
            np.concatenate(rows) if rows else np.zeros(0, np.int64))


def _gs_pass(o, X, stored, stored_cnt, stored_scale, props, penalty, perms, pa, pb, guard, threads,
             exchanges=True, start_fwd=None, start_fcnt=None):
    """One exact Gauss-Seidel acceptance pass on orientation o; X is modified in place.

    exchanges=False scores and applies the content proposals only (the copy-only pass of
    an addition round's best responses). start_fwd / start_fcnt: the forward messages (and
    counts) of X itself in orientation o, if known (a forward pass); until the first accepted
    move the carried forward message is read from them instead of being advanced.

    Invariants: (i) the current panel equals sigma(start panel) on every block at or after
    `frontier` (the furthest right end of a block changed by an accepted content move),
    where sigma is the composition of the accepted suffix exchanges; the stored backward
    messages (and their per-block normalisers) read through gperm are therefore exact
    there, and an exact backward chain from the frontier is recomputed from the current
    alleles before it is needed (only where an edit changed it: after a content move left
    of the frontier the chain right of the move's end stays exact, after a suffix exchange
    its states are relabelled); (ii) the carried forward message f is exact at every
    junction because blocks behind it never change again in this pass, and a lookahead of
    current forward messages is kept only up to the first block a later edit touched.
    At every junction every founder-pair suffix exchange is scored exactly and the best
    applied while its gain exceeds TOL; then the content candidates anchored there (one
    founder's rows on [p1, p2), the blocks where it differs from the current panel) are
    scored exactly from the current messages at p1 and p2:
    max_q [f(p1) (x) T_new[p1, p2) (x) g(p2)] - max_q [f(p1) + g(p1)], normalisers
    included, best first (earliest on ties), each applied while its gain exceeds TOL;
    coarse moves also pass the guard (None, "genotype_fit" or "shared_junction"; see
    SweepEngine), whose shared-junction test traces sample paths left of a candidate from
    the carried forward messages of the junctions already passed.

    Compiled (kernels.gs_pass_chunk): the pass runs as successive nogil calls of
    JUNCTION_BATCH junctions; before every call its Numba threads are re-read from
    `threads` (an int, or a callable budget whose share grows as other fits finish,
    candidates.SlotBudget), so finished work's cores flow to the running fits. The
    lookahead and backward-chain caches carry over from call to call (they hold exact
    current messages over the ranges they mark valid). Decisions, order and results never
    depend on the thread count or the batching. Returns (acc, junction_gain,
    junction_pair).
    """
    founders, blocks = X.shape
    samples, states = stored.shape[1], stored.shape[2]
    lexmode = guard is not None
    shared = guard == "shared_junction"
    haps = o.haps(X)
    cands = _candidate_arrays(props, blocks)
    count2 = np.zeros((1, 1), np.int64)
    count3 = np.zeros((1, 1, 1), np.int64)
    if shared:
        # Carried forward messages of the junctions passed (the guard's path tracing).
        local = np.empty((blocks + 1, samples, states))
        local_cnt = np.empty((blocks + 1, samples, states), np.int64)
    else:
        local, local_cnt = np.zeros((1, 1, 1)), count3
    acc = np.zeros(len(kernels.ACC_KEYS))
    f = np.zeros((samples, states))
    fcnt = np.zeros((samples, states), np.int64) if lexmode else count2
    gperm = np.arange(states, dtype=np.int64)
    lab = np.arange(founders, dtype=np.int64)
    # Lookahead and backward-chain caches, carried from call to call (empty at the start).
    ahead = (False, 0, -1, np.zeros((1, 1, 1)), np.zeros((1, 1, 1), np.int64), np.zeros((1, 1)),
             np.zeros((1, 1), np.int32), np.zeros((1, 1)), np.zeros((1, 1), np.int32),
             np.zeros((1, 1), np.int32))
    chain = (False, 0, 0, np.zeros((1, 1, 1)), np.zeros((1, 1, 1), np.int64), np.zeros((1, 1)))
    pristine = start_fwd is not None
    if start_fwd is None:
        start_fwd = np.zeros((1, 1, 1))
    if start_fcnt is None:
        start_fcnt = count3
    junction_gain = np.full(blocks + 1, -np.inf)
    junction_pair = np.zeros((blocks + 1, 2), np.int64)
    frontier, p = 0, 0
    restore = parallel._numba.get_num_threads()
    try:
        while p < blocks:
            parallel._numba.set_num_threads(max(1, int(resolve_threads(threads))))
            stop = min(blocks, p + JUNCTION_BATCH)
            (_, _, jg, jp, gperm, f, fcnt, frontier, accepted, ahead, chain) = kernels.gs_pass_chunk(
                haps, X, o.leaves, o.logs, o.offsets, float(penalty), p, stop, frontier, f, fcnt,
                local, local_cnt, stored, stored_cnt if lexmode else count3, stored_scale, gperm,
                lab, *cands, perms, pa, pb, _GUARD_CODES[guard], bool(exchanges), AHEAD_BLOCKS, TOL,
                GUARD_TOLERANCE, acc, ahead, chain, start_fwd, start_fcnt, pristine)
            pristine = pristine and not accepted
            junction_gain[p:stop] = jg
            junction_pair[p:stop] = jp
            p = stop
    finally:
        parallel._numba.set_num_threads(restore)
    record = {key: (int(round(value)) if key in _INT_ACC else float(value))
              for key, value in zip(kernels.ACC_KEYS, acc.tolist())}
    return record, junction_gain, junction_pair


class ProposalCache:
    """One search's last solved window inputs and results (SweepEngine.proposals).

    Invariant: the cached result of template t was solved from the rows windows[t][0] and
    the boundary messages fwd[a_t], bwd[c_t] held here (a boundary's messages are replaced
    only when every template ending there is re-solved)."""

    def __init__(self, source=None):
        self.fwd = {} if source is None else dict(source.fwd)
        self.bwd = {} if source is None else dict(source.bwd)
        self.windows = {} if source is None else dict(source.windows)
        self._fwd_same = self._bwd_same = None

    def lookup(self, engine, X, fwd, bwd, results):
        """Fill results[t] = (cached result, 0 CPU) for templates with unchanged inputs;
        return the others' indices."""
        self._fwd_same, self._bwd_same = {}, {}
        todo = []
        for index, (a, c) in enumerate(engine.spans):
            held = self.windows.get(index)
            if held is not None:
                if a not in self._fwd_same:
                    self._fwd_same[a] = np.array_equal(self.fwd[a], fwd[a])
                if c not in self._bwd_same:
                    self._bwd_same[c] = np.array_equal(self.bwd[c], bwd[c])
                if self._fwd_same[a] and self._bwd_same[c] and np.array_equal(held[0], X[:, a:c]):
                    results[index] = (held[1], 0.0)
                    continue
            todo.append(index)
        return todo

    def store(self, engine, X, fwd, bwd, todo, computed):
        for index, (result, _) in zip(todo, computed):
            a, c = engine.spans[index]
            self.windows[index] = (X[:, a:c].copy(), result)
            if not self._fwd_same.get(a, False):
                self.fwd[a] = fwd[a].copy()
                self._fwd_same[a] = True
            if not self._bwd_same.get(c, False):
                self.bwd[c] = bwd[c].copy()
                self._bwd_same[c] = True


class SweepEngine:
    """Immutable per-component inputs shared by every converged fit of the component."""

    def __init__(self, workspace, context, config, checkpoints=None, prefix="converged_sweep"):
        self.workspace = workspace
        self.models = workspace.models()
        self.block_emissions = [model["bin_emissions"] for model in self.models]
        self.emission_list = emission_arrays(self.models)
        self.penalty = float(workspace.penalty)
        self.blocks = len(workspace.leaves)
        self.config = config
        # "site": fine windows solve the canonical full-site F of the window (one-site
        # bins) instead of the binned surrogate; a validation option (coarse span-tuple
        # windows stay binned: their site-level pair tables would be too large).
        self.site = config.window_emission_resolution == "site"
        self.coarse = None
        if context is not None:
            groups, paths = context
            self.coarse = ([tuple(int(x) for x in group) for group in groups],
                           [np.ascontiguousarray(path, np.int64) for path in paths])
        # Coarse-move guard: None, "genotype_fit" (production's) or "shared_junction".
        self.guard = (config.coarse_move_guard
                      if config.coarse_move_guard != "none" and self.coarse is not None else None)
        self.forward = Orientation(workspace, False)
        self._reversed = Orientation(workspace, True)
        self.checkpoints = checkpoints
        self.prefix = prefix
        self.templates = window_templates(self.blocks, None if self.coarse is None else self.coarse[0])
        # Block range [a, c) of every template and the message boundaries a proposal cache
        # holds (fwd at window starts, bwd at window ends).
        self.spans = [(a, c) if kind == "fine" else (self.coarse[0][a][0], self.coarse[0][c - 1][1])
                      for kind, a, c in self.templates]
        self.cached_boundaries = (len({a for a, _ in self.spans}) + len({c for _, c in self.spans}))
        # Per-sample scores of converged fits by label, kept from their last sweep (fit_scores).
        self._fit_scores = {}

    def _load(self, label):
        return None if self.checkpoints is None else self.checkpoints.load(f"{self.prefix}.{label}")

    def _save(self, label, payload):
        if self.checkpoints is not None:
            self.checkpoints.save(f"{self.prefix}.{label}", payload)

    def per_sample(self, panel, threads):
        """Per-sample canonical scores of panel, scored over consecutive sample ranges at the
        share ``threads`` (an int or a callable budget) re-read before each range."""
        ws = self.workspace
        with parallel.numba_thread_scope(int(resolve_threads(threads))):
            alleles = founder_scoring.selected_alleles(ws.leaves, ws.offsets,
                                                       np.ascontiguousarray(panel, np.int64))
            return np.asarray(founder_scoring.score_panel(
                alleles, ws.evidence, ws.complete, ws.penalty, ws.logs,
                ranges=sample_ranges(len(ws.evidence), threads)))

    def score(self, panel, threads):
        return float(self.per_sample(panel, threads).sum())

    def fit_scores(self, label, panel, threads):
        """Per-sample scores of the converged fit `label` (whose panel is `panel`): the scores
        its last sweep computed for that panel, or computed now (a fit loaded from its
        checkpoint)."""
        scores = self._fit_scores.pop(label, None)
        if scores is None:
            scores = self.per_sample(panel, threads)
        return scores

    def fit_bytes(self, founders):
        """Memory of one fixed-K search: the message arrays its pass holds (stored backward
        values, + counts with a guard, + the forward history of the shared-junction test)."""
        samples = len(self.workspace.evidence)
        states = founders * (founders + 1) // 2
        arrays = {None: 2, "genotype_fit": 4, "shared_junction": 6}[self.guard]
        # + the boundary messages of its proposal cache (ProposalCache)
        boundaries = (self.blocks + 1) * arrays + self.cached_boundaries
        return 8 * boundaries * samples * states + 256 * 1024 ** 2

    # ------------------------------------------------------------------ proposals
    def _solve(self, problem):
        return window_dp.solve(problem, backstop=self.config.frontier_backstop,
                               screen_sweeps=self.config.screen_sweeps,
                               use_screen=self.config.window_screen)

    def fine_problems(self, X, fwd, bwd, a, c, focals=None):
        """One window problem per founder (or per listed founder) on blocks [a, c) (rows =
        all local rows)."""
        if self.site:
            ws = self.workspace
            emissions = List([window_dp.site_emissions(ws.leaves[b], ws.logs, int(ws.offsets[b]),
                                                       int(ws.offsets[b + 1])) for b in range(a, c)])
        else:
            emissions = List(self.block_emissions[a:c])
        cols = np.ascontiguousarray(X[:, a:c].T)
        sizes = [e.shape[1] for e in emissions]
        return [window_dp.WindowProblem(emissions, cols, sizes, k, X[k, a:c], fwd[a], bwd[c],
                                        self.penalty)
                for k in (range(len(X)) if focals is None else focals)]

    def coarse_problems(self, X, fwd, bwd, i, j):
        """One problem per founder over L1 spans [i, j); units choose span tuples.

        Alphabet of a span: the distinct row tuples the founders use there, then the L1
        context paths (production's macro alphabet). Returns (problems, alphabets, a, c).
        """
        groups, context = self.coarse
        alphabets, lookups = [], []
        for unit in range(i, j):
            start, stop = groups[unit]
            rows, lookup = [], {}
            for row in [*X[:, start:stop], *context[unit]]:
                key = np.ascontiguousarray(row, np.int64).tobytes()
                if key not in lookup:
                    lookup[key] = len(rows)
                    rows.append(np.ascontiguousarray(row, np.int64))
            alphabets.append(np.asarray(rows, np.int64))
            lookups.append(lookup)
        emissions = gather_macro(self.emission_list, np.asarray(groups[i:j], np.int64),
                                 List(alphabets))
        cols = np.empty((j - i, len(X)), np.int64)
        for u, unit in enumerate(range(i, j)):
            start, stop = groups[unit]
            for founder in range(len(X)):
                cols[u, founder] = lookups[u][np.ascontiguousarray(X[founder, start:stop]).tobytes()]
        a, c = groups[i][0], groups[j - 1][1]
        sizes = [len(alphabet) for alphabet in alphabets]
        problems = [window_dp.WindowProblem(emissions, cols, sizes, k, cols[:, k], fwd[a], bwd[c],
                                            self.penalty) for k in range(len(X))]
        return problems, alphabets, a, c

    def _template(self, emissions, cols, sizes, start, end, focals=None):
        """Every (listed) founder's window problem in one nogil call (window_dp.solve_template).

        Returns [(focal, choice or None, diagnostic)]; the Lagrangian-dual iterations
        (window_screen) and the site-resolution option take the per-problem solve()."""
        founders = cols.shape[1]
        focals = np.arange(founders, dtype=np.int64) if focals is None else np.asarray(focals, np.int64)
        samples, states = start.shape
        store_limit = max(1, window_dp.STORE_BYTES // max(1, 8 * samples * states))
        status, rows, box, dp, frontier, fixing = window_dp.solve_template(
            emissions, np.ascontiguousarray(cols, np.int64), np.asarray(sizes, np.int64), focals,
            self.penalty, start, end, int(self.config.frontier_backstop), int(store_limit))
        out = []
        for i, k in enumerate(focals):
            if status[i] == 2:
                raise RuntimeError(f"window dynamic programme frontier exceeded its backstop "
                                   f"{self.config.frontier_backstop}")
            if status[i] == 3:
                raise RuntimeError("window dynamic programme lost every feasible path")
            out.append((int(k), rows[i].copy() if status[i] == 1 else None,
                        dict(screened=True, certified=bool(box[i]), box_certified=bool(box[i]),
                             dp=bool(dp[i]), frontier=int(frontier[i]), fixing_dps=int(fixing[i]))))
        return out

    def _direct(self):
        return self.config.window_screen or self.site

    def _fine(self, X, fwd, bwd, a, c, focals=None):
        out = []
        if self._direct():
            problems = self.fine_problems(X, fwd, bwd, a, c, focals)
            for k, problem in zip(range(len(X)) if focals is None else focals, problems):
                rows, diagnostic = self._solve(problem)
                out.append(("fine", a, c, k, rows, diagnostic))
            return out
        emissions = window_dp.slice_list(self.emission_list, a, c)
        sizes = [self.block_emissions[b].shape[1] for b in range(a, c)]
        for k, rows, diagnostic in self._template(emissions, X[:, a:c].T, sizes, fwd[a], bwd[c],
                                                  focals):
            out.append(("fine", a, c, k, rows, diagnostic))
        return out

    def _coarse(self, X, fwd, bwd, i, j):
        problems, alphabets, a, c = self.coarse_problems(X, fwd, bwd, i, j)
        if self._direct():
            solved = [(k, *self._solve(problem)) for k, problem in enumerate(problems)]
        else:
            probe = problems[0]
            solved = self._template(probe.emissions, probe.cols, np.diff(probe.offsets),
                                    probe.start, probe.end)
        out = []
        for k, choice, diagnostic in solved:
            rows = None
            if choice is not None:
                rows = np.concatenate([alphabets[u][int(choice[u])] for u in range(j - i)])
                if np.array_equal(rows, X[k, a:c]):
                    rows = None
            out.append(("coarse", a, c, k, rows, diagnostic))
        return out

    def proposals(self, X, fwd, bwd, threads, cache=None):
        """Every window template's proposals for panel X with messages fwd, bwd.

        cache (a search's ProposalCache, updated in place): a window problem is a pure
        function of the panel's rows in the window and the messages fwd[a], bwd[c] at its
        ends, and normalised max-plus messages of panels that differ only further away are
        often bitwise equal, so a template whose three inputs equal the cached ones bitwise
        takes its cached result (the same proposals and diagnostics) instead of a solve.
        The templates to solve run one thread each, as many at once as the share
        ``threads`` (an int or a callable budget) allows (_one_thread_map).
        """
        def run(index):
            kind, a, c = self.templates[index]
            started = time.thread_time()
            if kind == "fine":
                result = self._fine(X, fwd, bwd, a, c)
            else:
                result = self._coarse(X, fwd, bwd, a, c)
            return result, time.thread_time() - started

        results = [None] * len(self.templates)
        todo = list(range(len(self.templates))) if cache is None else cache.lookup(self, X, fwd, bwd,
                                                                                 results)
        computed = _one_thread_map(run, todo, threads)
        for index, value in zip(todo, computed):
            results[index] = value
        if cache is not None:
            cache.store(self, X, fwd, bwd, todo, computed)
        proposals = []
        stats = dict(windows=0, screened=0, certified=0, dp=0, fixing_dps=0, max_frontier=0,
                     changed=0, worker_cpu=0.0)
        for result, cpu in results:
            stats["worker_cpu"] += cpu
            for kind, a, c, k, rows, diagnostic in result:
                stats["windows"] += 1
                stats["screened"] += int(diagnostic["screened"])
                stats["certified"] += int(diagnostic["certified"])
                stats["dp"] += int(diagnostic["dp"])
                stats["fixing_dps"] += int(diagnostic["fixing_dps"])
                stats["max_frontier"] = max(stats["max_frontier"], int(diagnostic["frontier"]))
                if rows is not None:
                    stats["changed"] += 1
                    proposals.append((kind, a, c, k, np.asarray(rows, np.int64)))
        return proposals, stats

    def copy_response(self, Y, threads):
        """Exact F gain of the copy's fine-window best responses (an addition round's seed).

        Y is a duplicate-seed panel whose last row copies a founder. Every fine window's
        canonical best response of the copy alone (B/4 window programmes, the other rows
        fixed, messages of Y) is applied by one copy-only Gauss-Seidel pass in block order
        (exact gains above TOL, no exchanges). Returns (gain, record).
        """
        copy = len(Y) - 1
        fwd, bwd, _, _, _, bscale = self._full_messages(Y, False, threads)

        def run(template):
            return self._fine(Y, fwd, bwd, template[1], template[2], focals=(copy,))

        fine = [template for template in self.templates if template[0] == "fine"]
        solved = _one_thread_map(run, fine, threads)
        props, windows = {}, 0
        for result in solved:
            for _, a, c, _, rows, _ in result:
                windows += 1
                if rows is not None:
                    props.setdefault(a, []).append(("fine", c, copy, np.asarray(rows, np.int64)))
        perms, pa, pb = kernels.swap_permutations(len(Y))
        with parallel.numba_thread_scope(int(resolve_threads(threads))):
            acc, _, _ = _gs_pass(self.forward, np.ascontiguousarray(Y, np.int64).copy(), bwd, None,
                                 bscale, props, self.penalty, perms, pa, pb, None, threads,
                                 exchanges=False, start_fwd=fwd)
        return float(acc["fine_gain"]), dict(windows=windows, proposals=int(sum(map(len, props.values()))),
                                            accepted=int(acc["fine"]), gain=float(acc["fine_gain"]))

    # ------------------------------------------------------------------ fixed-K fit
    def converge(self, start, label, threads):
        """C(start): the better of the forward-first and the reverse-first search.

        Both run concurrently, each with its share of the thread budget (re-read before every
        sample range of its messages and scores, window programme and acceptance-pass call,
        so the survivor takes over the finished one's cores within its current phase). The
        higher exact F is kept (exact ties: forward-first); the record is the kept search's,
        with both searches' CPU and record["best_of_two"].
        """
        done = self._load(f"{label}.done")
        if done is not None:
            return np.asarray(done["rows"], np.int64).copy(), done["record"]
        started = time.perf_counter()
        # Both searches start from the same panel, so the F (and per-sample scores), messages
        # and window proposals of their sweep 0 (pure functions of the panel) are computed
        # once, with the fit's whole thread budget, and shared; only their acceptance passes
        # differ. A search that resumes from its own checkpoints computes its sweeps itself.
        shared = None
        if all(self._load(f"{label}.{first}.{stage}") is None
               for first in ("fwd", "rev") for stage in ("done", "sweep0")):
            X = np.ascontiguousarray(start, np.int64)
            scores0 = self.per_sample(X, threads)
            cache = ProposalCache()
            shared = dict(F0=float(scores0.sum()), scores0=scores0,
                          inputs=self._sweep_inputs(X, threads, cache), cache=cache,
                          waiting={"fwd", "rev"}, readers=set(), lock=threading.Lock())
        (X_fwd, fwd), (X_rev, rev) = _parallel_map(
            [lambda budget, first=first: self._converge_direction(start, label, first,
                                                                  budget or threads, shared)
             for first in ("fwd", "rev")], threads, self.fit_bytes(len(start)))
        chosen = "rev" if rev["F"] > fwd["F"] else "fwd"
        X, kept = (X_rev, rev) if chosen == "rev" else (X_fwd, fwd)
        scores = {first: self._fit_scores.pop(f"{label}.{first}", None) for first in ("fwd", "rev")}
        if scores[chosen] is not None:
            self._fit_scores[label] = scores[chosen]
        record = dict(kept, label=label, first=chosen,
                      worker_cpu=fwd["worker_cpu"] + rev["worker_cpu"],
                      thread_cpu=fwd["thread_cpu"] + rev["thread_cpu"],
                      wall=time.perf_counter() - started,
                      best_of_two=dict(chosen=chosen, F_fwd=fwd["F"], F_rev=rev["F"],
                                       gain=float(kept["F"] - fwd["F"]),
                                       sweeps_fwd=fwd["sweeps"], sweeps_rev=rev["sweeps"],
                                       converged_fwd=fwd["converged"],
                                       converged_rev=rev["converged"],
                                       wall_fwd=fwd["wall"], wall_rev=rev["wall"]))
        self._save(f"{label}.done", dict(rows=X, record=record))
        return X, record

    def _full_messages(self, X, lex, threads):
        """messages(_lex) of panel X: (fwd, bwd, fcnt, bcnt, fscale, bscale), the counts None
        without lex; computed over consecutive sample ranges at the share ``threads`` (an int or
        a callable budget) re-read before each range (every sample's recursion is its own)."""
        logs, offsets = self.workspace.logs, self.forward.offsets
        with parallel.numba_thread_scope(int(resolve_threads(threads))):
            haps = self.forward.haps(X)
        messages = kernels.message_arrays(len(X), self.blocks, len(logs), lex)
        fwd, bwd, fcnt, bcnt, fscale, bscale = messages
        for s0, s1, n in sample_ranges(len(logs), threads):
            with parallel.numba_thread_scope(n):
                if lex:
                    kernels.messages_lex(haps, logs, offsets, self.penalty, s0, s1, fwd, bwd, fcnt,
                                         bcnt, fscale, bscale)
                else:
                    kernels.messages(haps, logs, offsets, self.penalty, s0, s1, fwd, bwd, fscale,
                                     bscale)
        return messages

    def _messages(self, X, threads, held=None):
        """The messages of panel X (with switch counts under a guard) and their wall time.

        held: (panel, messages) of an earlier panel, owned by the caller: they are updated in
        place to X's where X differs from that panel (kernels.update_messages(_lex), bitwise
        the full computation); a held backward half of None is computed whole. Computed over
        consecutive sample ranges at the share ``threads`` re-read before each range.
        """
        started = time.perf_counter()
        if held is None:
            return self._full_messages(X, bool(self.guard), threads), time.perf_counter() - started
        logs, offsets = self.workspace.logs, self.forward.offsets
        with parallel.numba_thread_scope(int(resolve_threads(threads))):
            haps = self.forward.haps(X)
        panel, (fwd, bwd, fcnt, bcnt, fscale, bscale) = held
        known = bwd is not None
        if not known:
            bwd, bscale = np.zeros_like(fwd), np.zeros_like(fscale)
            bcnt = None if fcnt is None else np.zeros_like(fcnt)
        changed = np.any(panel != X, axis=0)
        for s0, s1, n in sample_ranges(len(logs), threads):
            with parallel.numba_thread_scope(n):
                if self.guard:
                    kernels.update_messages_lex(haps, logs, offsets, self.penalty, changed, fwd, bwd,
                                                fcnt, bcnt, fscale, bscale, known, s0, s1)
                else:
                    kernels.update_messages(haps, logs, offsets, self.penalty, changed, fwd, bwd,
                                            fscale, bscale, known, s0, s1)
        return (fwd, bwd, fcnt, bcnt, fscale, bscale), time.perf_counter() - started

    def _sweep_inputs(self, X, threads, cache, held=None):
        """The messages of panel X and its window proposals (pure functions of the panel)."""
        messages, message_wall = self._messages(X, threads, held)
        started = time.perf_counter()
        proposals, stats = self.proposals(X, messages[0], messages[1], threads, cache)
        return dict(messages=messages, proposals=proposals, stats=stats, message_wall=message_wall,
                    proposal_wall=time.perf_counter() - started)

    def _converge_direction(self, start, label, first, threads, shared=None):
        """One search from start: sweeps until one accepts nothing (or the reported
        backstop); the first pass runs forward (first "fwd") or reversed ("rev"), later
        passes alternate. Checkpoints <label>.<first>.sweep<n> / .done. shared: the start
        panel's F and sweep-0 inputs computed once for both searches (converge)."""
        label = f"{label}.{first}"
        done = self._load(f"{label}.done")
        if done is not None:
            return np.asarray(done["rows"], np.int64).copy(), done["record"]
        started, started_cpu = time.perf_counter(), time.thread_time()
        X = np.ascontiguousarray(start, np.int64).copy()
        founders, blocks = X.shape
        perms, pa, pb = kernels.swap_permutations(founders)
        # Per-sample scores of the current panel X, while known: a pass that accepts nothing
        # leaves the panel, hence its scores, unchanged.
        if shared is not None:
            F0, current = shared["F0"], shared["scores0"]
        else:
            current = self.per_sample(X, threads)
            F0 = float(current.sum())
        F, history, junctions, converged, sweep = F0, [], None, False, 0
        # The search's proposal cache (seeded by the shared sweep 0).
        cache = ProposalCache(None if shared is None else shared["cache"])
        # This search's own messages of the panel its last sweep started from, updated in place
        # by the next sweep (_messages): (panel, messages); a reversed pass keeps their forward
        # half only, so the search holds no more message arrays than its forward passes do.
        held = None
        backstop = self.config.sweep_backstop
        resumed = 0
        while sweep < backstop:
            saved = self._load(f"{label}.sweep{sweep}")
            if saved is None:
                break
            X = np.asarray(saved["rows"], np.int64).copy()
            F, history, junctions = saved["F"], list(saved["history"]), saved["junctions"]
            current = None
            sweep += 1
            resumed += 1
            if saved["accepted"] == 0:
                converged = True
                break
        while not converged and sweep < backstop:
            sweep_started = time.perf_counter()
            owned = True
            if sweep == 0 and shared is not None:
                with shared["lock"]:
                    inputs = shared["inputs"]
                    shared["waiting"].discard(first)
                    if not shared["waiting"]:
                        shared["inputs"] = None
                    if inputs["messages"] is not None:
                        shared["readers"].add(first)
                if first == "rev":
                    # The shared proposals' CPU is reported once, by the forward-first search.
                    inputs = dict(inputs, stats=dict(inputs["stats"], worker_cpu=0.0))
                if inputs["messages"] is None:
                    # The other search ran first and released them (see below).
                    messages, message_wall = self._messages(X, threads)
                    inputs = dict(inputs, messages=messages, message_wall=message_wall)
                else:
                    owned = False
            else:
                inputs = self._sweep_inputs(X, threads, cache, held)
            held = None
            fwd, bwd, fcnt, bcnt, fscale, bscale = inputs["messages"]
            proposals, stats = inputs["proposals"], inputs["stats"]
            message_wall, proposal_wall = inputs["message_wall"], inputs["proposal_wall"]
            del inputs
            reverse = (sweep % 2 == 1) if first == "fwd" else (sweep % 2 == 0)
            props = {}
            for kind, a, c, k, rows in proposals:
                if reverse:
                    props.setdefault(blocks - c, []).append((kind, blocks - a, k, rows[::-1].copy()))
                else:
                    props.setdefault(a, []).append((kind, c, k, rows))
            pass_started = time.perf_counter()
            with parallel.numba_thread_scope(int(resolve_threads(threads))):
                if reverse:
                    # The reversed pass stores the clamped forward messages; their per-block
                    # normalisers are the forward ones in reversed block order.
                    o = self._reversed
                    if self.guard:
                        stored, stored_cnt = kernels.clamp_reverse_lex(fwd, fcnt, self.penalty)
                    else:
                        stored, stored_cnt = kernels.clamp_reverse(fwd, self.penalty), None
                    stored_scale = np.ascontiguousarray(fscale[::-1])
                    Xo = np.ascontiguousarray(X[:, ::-1])
                    start_fwd = start_fcnt = None
                    kept = (fwd, None, fcnt, None, fscale, None)
                else:
                    o, stored, stored_cnt, stored_scale, Xo = self.forward, bwd, bcnt, bscale, X.copy()
                    start_fwd, start_fcnt = fwd, fcnt
                    kept = (fwd, bwd, fcnt, bcnt, fscale, bscale)
                del fwd, bwd, fcnt, bcnt, fscale, bscale
                acc, gains, pairs = _gs_pass(o, Xo, stored, stored_cnt, stored_scale, props,
                                             self.penalty, perms, pa, pb, self.guard, threads,
                                             start_fwd=start_fwd, start_fcnt=start_fcnt)
                del stored, stored_cnt, stored_scale, start_fwd, start_fcnt
            if sweep == 0 and shared is not None:
                # Run one after the other (a one-thread or memory-limited budget), the other
                # search has not started: its message arrays are not kept alive through this
                # whole search; it recomputes them and keeps the shared proposals. The shared
                # messages become this search's own once no other search reads them (the last
                # one to finish its sweep-0 pass).
                with shared["lock"]:
                    if shared["inputs"] is not None:
                        shared["inputs"] = dict(shared["inputs"], messages=None)
                    if not owned:
                        shared["readers"].discard(first)
                        owned = not shared["readers"]
            if owned:
                held = (X, kept)
            del kept
            X = np.ascontiguousarray(Xo[:, ::-1]) if reverse else Xo
            accepted = acc["fine"] + acc["coarse"] + acc["exchange"]
            if accepted or current is None:
                current = self.per_sample(X, threads)
            scores = current
            F_new = float(scores.sum())
            pass_wall = time.perf_counter() - pass_started
            accounted = acc["fine_gain"] + acc["coarse_gain"] + acc["exchange_gain"]
            deviation = F_new - (F + accounted)
            if abs(deviation) > 1e-6 * max(1.0, abs(F_new)):
                raise RuntimeError(
                    f"converged sweep {self.prefix}.{label}:{sweep}: canonical F changed by "
                    f"{F_new - F}, accepted local gains {accounted}")
            junctions = _weak_junctions(gains, pairs, reverse, blocks)
            history.append(dict(sweep=sweep, reverse=bool(reverse), F=float(F_new),
                                dF=float(F_new - F), dF_accounted=float(accounted),
                                deviation=float(deviation), accepted=int(accepted), **acc,
                                proposals=stats, wall=time.perf_counter() - sweep_started,
                                message_wall=message_wall, proposal_wall=proposal_wall,
                                pass_wall=pass_wall))
            F = F_new
            self._save(f"{label}.sweep{sweep}", dict(rows=X, F=F, sweep=sweep, reverse=bool(reverse),
                                                     accepted=int(accepted), history=history,
                                                     junctions=junctions))
            sweep += 1
            if accepted == 0:
                converged = True
        if not converged:
            warnings.warn(f"converged founder sweep {self.prefix}.{label} reached its "
                          f"{backstop}-sweep backstop without converging", RuntimeWarning)
        if len(history) > resumed:
            self._fit_scores[label] = scores
        record = dict(label=label, founders=int(founders), F0=float(F0), F=float(F),
                      gain=float(F - F0), converged=bool(converged), sweeps=len(history),
                      resumed_sweeps=resumed, backstop=int(backstop),
                      first=first, accepted=_totals(history),
                      max_frontier=max([h["proposals"]["max_frontier"] for h in history] or [0]),
                      max_deviation=max([abs(h.get("deviation", 0.0)) for h in history] or [0.0]),
                      worker_cpu=float(sum(h["proposals"]["worker_cpu"] for h in history)),
                      wall=time.perf_counter() - started,
                      thread_cpu=time.thread_time() - started_cpu,
                      weak_junctions=junctions or [], history=history)
        self._save(f"{label}.done", dict(rows=X, record=record))
        return X, record


def _totals(history):
    keys = ("fine", "coarse", "exchange", "fine_gain", "coarse_gain", "exchange_gain",
            "evaluated", "false_positive", "guard_rejected", "guard_shared_tests", "guard_relaxed",
            "guard_shared_switches", "guard_traced_blocks", "changed_blocks")
    return {key: sum(h[key] for h in history) for key in keys}


def _weak_junctions(gains, pairs, reverse, blocks):
    """Junctions whose best suffix exchange costs less than one switch penalty.

    Consecutive junctions with the same founder pair are merged into one range.
    In the certifying sweep these values are exact against the final panel.
    """
    records = []
    for p in range(1, blocks):
        if not gains[p] > -WEAK_JUNCTION_MARGIN:
            continue
        junction = blocks - p if reverse else p
        records.append((junction, int(pairs[p][0]), int(pairs[p][1]), float(gains[p])))
    records.sort()
    merged = []
    for junction, a, b, gain in records:
        if merged and merged[-1]["pair"] == [a, b] and merged[-1]["last"] == junction - 1:
            merged[-1]["last"] = junction
            merged[-1]["best_gain"] = max(merged[-1]["best_gain"], gain)
        else:
            merged.append(dict(first=junction, last=junction, pair=[a, b], best_gain=gain))
    return merged


# ---------------------------------------------------------------------- count pass
def _point_margins(larger, smaller, cost):
    """The effective-carrier point margins of larger - smaller: m and the leave-one-sample-out m'."""
    differences = np.asarray(larger, np.float64) - np.asarray(smaller, np.float64)
    samples = len(differences)
    rest = np.delete(differences, int(np.argmax(differences)))
    margin = (float(np.sum(differences)) / cost
              - count_criterion.effective_carriers(differences) / (2. * samples))
    loso = float(np.sum(rest)) / cost - count_criterion.effective_carriers(rest) / (2. * samples)
    return margin, loso


def _shadow(geometry, larger, reference):
    """X+^sigma: every cell (block, row) of ``larger`` that ``reference`` does not use, replaced
    by the row ``reference`` uses there nearest to it (panel_count.shadow_path: Hamming over
    called sites, ties to the lowest row). Returns the shadow rows and the replaced cells."""
    cover = panel_count.covered_rows(np.asarray(reference, np.int64))
    rows, replaced = [], 0
    for path in larger:
        shadow, lost = panel_count.shadow_path(path, cover, geometry)
        rows.append(shadow)
        replaced += lost
    return np.asarray(rows, np.int64), replaced


def _own_sequence(rows):
    """owns[f]: founder f has a private cell (a row at some block that no other founder uses)."""
    return panel_count.private_mask(np.asarray(rows, np.int64)).any(axis=1)


def _sequence_scores(engine, geometry, larger, larger_scores, reference, threads):
    """Per-sample scores of the shadow of ``larger`` against ``reference`` (q = larger - shadow).

    q = 0 when a founder of the larger panel has no private cell or when every cell of the
    larger panel is used by the reference; otherwise the shadow is scored exactly (no
    near-identity shortcut). Returns (shadow scores, record).
    """
    owns = _own_sequence(larger)
    record = dict(without_sequence=[int(f) for f in np.flatnonzero(~owns)], shadow_cells=0)
    if not owns.all():
        return larger_scores, record
    shadow, replaced = _shadow(geometry, larger, reference)
    record["shadow_cells"] = int(replaced)
    if not replaced:
        return larger_scores, record
    return engine.per_sample(shadow, threads), record


def _loso_margin(diagnostics):
    """The joint decision's leave-one-sample-out margin: the smaller of the two parts'."""
    return float(min(diagnostics["total"]["loso_margin"], diagnostics["sequence"]["loso_margin"]))


def _death_candidate(engine, geometry, rows, threads):
    """The join's death candidate of a panel without refits: founders without a private
    cell first, then the largest unrefitted F(rows minus f) (ties: lower founder)."""
    owns = _own_sequence(rows)
    pool = np.flatnonzero(~owns) if not owns.all() else np.arange(len(rows))
    totals = [engine.score(np.delete(rows, f, axis=0), threads) for f in pool.tolist()]
    return int(pool[max(range(len(pool)), key=lambda i: (totals[i], -i))])


def _one_thread_map(function, items, threads):
    """[function(item) for item in items] in input order, each call on one Numba thread, with as
    many calls running at once as the share ``threads`` (an int or a callable budget) allows.

    Worker threads take the items in order. The caller starts another worker whenever the share,
    re-read at every completion and every POLL_SECONDS, exceeds the workers, and a worker stops
    after an item while the workers exceed the share: a search whose share grows while its window
    programmes run takes the freed threads within the phase, and never runs more threads than its
    share (each worker serves many items, so the threads and their allocator arenas stay few).
    """
    results = [None] * len(items)
    state = dict(taken=0, finished=0, workers=0, error=None)
    changed = threading.Condition()

    def share():
        return max(1, int(resolve_threads(threads)))

    def work():
        with parallel.numba_thread_scope(1):
            while True:
                with changed:
                    if (state["error"] is not None or state["taken"] == len(items)
                            or state["workers"] > share()):
                        state["workers"] -= 1
                        changed.notify()
                        return
                    index = state["taken"]
                    state["taken"] += 1
                try:
                    results[index] = function(items[index])
                except BaseException as error:
                    with changed:
                        if state["error"] is None:
                            state["error"] = error
                with changed:
                    state["finished"] += 1
                    changed.notify()

    started = []
    with changed:
        try:
            while state["finished"] < len(items) and state["error"] is None:
                while state["workers"] < min(share(), len(items) - state["taken"]):
                    state["workers"] += 1
                    started.append(threading.Thread(target=work, daemon=True))
                    started[-1].start()
                changed.wait(POLL_SECONDS)
        except BaseException as error:
            if state["error"] is None:
                state["error"] = error
    for worker in started:
        worker.join()
    if state["error"] is not None:
        raise state["error"]
    return results


def _parallel_map(functions, threads, bytes_per_task):
    """Results of independent calls in input order (each gets its share of the thread budget)."""
    results = [None] * len(functions)
    for index, value in completed_candidates(functions, threads, bytes_per_task):
        results[index] = value
    return results


def _fit_many(engine, starts, labels, threads):
    """Converge several independent starts concurrently; results in input order.

    Each fit's Numba threads are its share of the budget among the fits still running
    (re-read within every phase of its searches, so the last fits take over the finished
    ones' cores); a fit's share is split between its two concurrent searches
    (SweepEngine.converge).
    """
    per_fit = 2 * engine.fit_bytes(max(len(start) for start in starts))
    return _parallel_map([lambda budget, start=start, label=label: engine.converge(
        start, label, budget or threads) for start, label in zip(starts, labels)], threads, per_fit)


def _best_refit(chosen, values):
    """Position (in chosen) of the best-F refit; exact ties go to the lower candidate."""
    return max(range(len(chosen)), key=lambda i: (values[i], -chosen[i]))


def _decision_entry(kind, label, founders, decision, ranking, scores, chosen, values, pick,
                    larger_rows, smaller_rows, cost):
    candidate_F = [None] * int(founders)
    for candidate, value in zip(chosen, values):
        candidate_F[candidate] = float(value)
    others = sorted((v for i, v in enumerate(values) if i != pick), reverse=True)
    return dict(kind=kind, round=label, founders=int(founders), skipped_by_bound=False,
                ranking=ranking,
                candidate_scores=(None if scores is None else
                                  [None if v is None else float(v) for v in scores]),
                refitted=[int(c) for c in chosen], candidate_F=candidate_F,
                selected=int(chosen[pick]), selected_F=float(values[pick]),
                runner_up_margin_over_cost=(None if not others
                                            else float((values[pick] - others[0]) / cost)),
                accept_larger=bool(decision.accept_larger), margin=float(decision.margin),
                band=[float(x) for x in decision.band], ambiguous=bool(decision.ambiguous),
                criterion=decision.diagnostics, larger_rows=np.array(larger_rows, np.int64),
                smaller_rows=np.array(smaller_rows, np.int64))


def _restored_comparisons(engine, geometry, threads, rows, scores, best_at, decisions, cost,
                          config, seed_key):
    """Adjacent-K comparisons of a best-F restored panel from fits already computed.

    Diagnostics only (no decision reads them): the recorded count decisions with the
    restored panel on one side, and the count rule's decision between the restored panel
    and the best-F converged fit computed at K - 1 (deletion; the sequence part is that
    of the restored panel's death candidate, _death_candidate) and at K + 1 (addition;
    the shadow is taken against the restored panel) during the pass, from their stored
    per-sample scores. The restored panel fails its own deletion test when its recorded
    deletion round accepted a deletion or, without a recorded round, when the best-F
    K - 1 fit is preferred to it; the call is then flagged ambiguous.
    """
    K = len(rows)
    recorded = []
    for entry in decisions:
        if entry.get("skipped_by_bound"):
            continue
        role = ("larger" if np.array_equal(entry["larger_rows"], rows) else
                "smaller" if np.array_equal(entry["smaller_rows"], rows) else None)
        if role is not None:
            recorded.append(dict(kind=entry["kind"], round=entry["round"], role=role,
                                 accept_larger=entry["accept_larger"], margin=entry["margin"],
                                 loso_margin=_loso_margin(entry["criterion"]),
                                 band=list(entry["band"]), ambiguous=entry["ambiguous"]))

    def replay(kind, other):
        held = best_at.get(other)
        if held is None:
            return None
        if kind == "deletion":
            larger, larger_scores, smaller = rows, scores, held[3]
            founder = _death_candidate(engine, geometry, rows, threads)
            reference = np.delete(rows, founder, axis=0)
        else:
            larger, larger_scores, smaller, reference = held[1], held[3], scores, rows
        shadow, _ = _sequence_scores(engine, geometry, larger, larger_scores, reference, threads)
        decision = count_criterion.decide_with_sequence(
            larger_scores, smaller, shadow, cost, kind=kind, config=config,
            seed_key=[*seed_key, "best_f", kind])
        return dict(fit=held[2]["label"], founders=int(other), F=float(held[0]),
                    accept_larger=bool(decision.accept_larger), margin=float(decision.margin),
                    loso_margin=_loso_margin(decision.diagnostics),
                    sequence_margin=float(decision.diagnostics["sequence"]["margin"]),
                    band=[float(x) for x in decision.band], ambiguous=bool(decision.ambiguous))

    deletion = replay("deletion", K - 1) if K > 2 else None
    addition = replay("addition", K + 1)
    own = next((item for item in recorded
                if item["kind"] == "deletion" and item["role"] == "larger"), None)
    test = own if own is not None else deletion
    fails = None if test is None else not test["accept_larger"]
    return dict(recorded=recorded, deletion_vs_best_smaller=deletion,
                addition_vs_best_larger=addition,
                own_deletion_test=(None if test is None else
                                   dict(source="recorded_round" if own is not None
                                        else "best_f_smaller_fit", **test)),
                fails_own_deletion_test=fails, ambiguous=bool(fails))


def count_pass(engine, start, cost, threads, seed_key, additions=True):
    """Converged fixed-K fit and converged count decisions for one component.

    Every decision is the count rule's (count_criterion.decide_with_sequence: the
    effective-carrier test on the total gain and on its sequence part, see the module
    docstring) between converged fits, at a cost of at most r = COUNT_RANKED_REFITS
    converged fits per decision, run concurrently, plus one shadow-panel score. Deletion
    round: the candidates are the founders without a private cell when there are any
    (their sequence part is zero, so the rule deletes whichever is decided), otherwise
    every founder. With more than r candidates, their exact unrefitted per-sample scores
    rank them by the point margin min(m, m') of the total gain and the r top-ranked
    deletions are converged (each from the incumbent without that founder); otherwise every
    candidate is. The best-F of them (exact ties: the lower founder index) is decided, its
    sequence part from the incumbent with that founder's private cells replaced; a
    rejection ends the deletions (backward elimination with refit). Addition round: with
    more than r founders every duplicate seed's copy gets its exact fine-window best
    responses (one copy-only Gauss-Seidel pass, SweepEngine.copy_response) and the r seeds
    of largest gain are converged; otherwise every seed is; the best-F of them (ties: the
    lower seed) is decided, its sequence part from the refit with every cell the incumbent
    lacks replaced. An optimistic bound may skip a round only when it proves the decision
    for every candidate: the addition bound (an accepted addition needs the total-gain
    test) as before; then the per-sample addition bound, which proves every addition
    rejected and not ambiguous when no sample's optimistic gain reaches the per-carrier
    cost (count_criterion.addition_per_sample_bound_excludes); the deletion bound when
    every founder has a private cell, where its per-sample bounds also bound every shadow
    panel (count_criterion.deletion_bound_excludes). A kept deletion round reports the
    other founders whose sequence part fails the test (hidden_sequence_failures;
    diagnostics only). additions: run the addition rounds (the final count step); False
    makes deletion decisions only (earlier levels).
    """
    from ..founder_refinement import COUNT_RANKED_REFITS as ranked_refits
    config = engine.config
    incoming = len(start)
    fits, decisions = [], []
    X, record = engine.converge(start, "start", threads)
    fits.append(record)
    ws = engine.workspace
    # The join's called-site geometry of the component's prepared rows (near-identity,
    # nearest covered rows).
    geometry = panel_count.RowGeometry(ws.batch)
    scores = engine.fit_scores(record["label"], X, threads)
    samples = len(scores)
    round_index = 0
    # Reported backstop: additions stop at the incoming count plus count_round_backstop
    # times the incoming count (default: K_in + K_in).
    upper = incoming + config.count_round_backstop * incoming
    backstop = dict(upper=int(upper), reached=False)
    # Best-F converged fit computed at each K: (F, panel, fit, per-sample scores).
    best_at = {}

    def remember(panel, fit, per_sample):
        F_fit = float(per_sample.sum())
        held = best_at.get(len(panel))
        if held is None or F_fit > held[0] + TOL:
            best_at[len(panel)] = (F_fit, panel, fit, per_sample)

    remember(X, record, scores)

    def deletion_round(X, scores):
        nonlocal round_index
        K = len(X)
        label = f"r{round_index}"
        round_index += 1
        F = float(scores.sum())
        # Founders without a private cell (q = 0) are decided first, as in the join.
        without = [int(f) for f in np.flatnonzero(~_own_sequence(X))]
        optimistic = None
        if not without:
            # Every founder has a private cell: the per-sample bounds of any K - 1 panel also
            # bound every shadow panel (count_criterion.deletion_bound_excludes), so a bound that
            # proves the total-gain test for every deletion proves the sequence test too.
            with parallel.numba_thread_scope(int(resolve_threads(threads))):
                bounds, _, per_block_sample = founder_count_bound.upper_bound(
                    ws.leaves, ws.offsets, ws.logs, ws.penalty, K - 1)
            per_sample_optimistic = per_block_sample.sum(axis=0)
            optimistic = float(bounds.sum())
            margin = 1e-8 * max(1.0, abs(F), abs(optimistic), abs(K * cost - 2.0 * F))
            if count_criterion.deletion_bound_excludes(
                    F, optimistic, cost, margin=margin,
                    per_sample_score=scores, per_sample_optimistic=per_sample_optimistic):
                return dict(kind="deletion", round=label, founders=K, skipped_by_bound=True,
                            optimistic_score=optimistic, accept_larger=True, ambiguous=False,
                            without_sequence=[]), None, None
        candidates = without or list(range(K))
        if len(candidates) > ranked_refits:
            # Exact unrefitted per-sample scores of every candidate's deletion, ranked by the
            # point margin of the total gain (ties: lower founder index); the top r are converged.
            keys = [None] * K
            for d in candidates:
                unrefitted = engine.per_sample(np.delete(X, d, axis=0), threads)
                keys[d] = min(_point_margins(scores, unrefitted, cost))
            chosen = sorted(candidates, key=lambda d: (keys[d], d))[:ranked_refits]
            ranking = "unrefitted_point_margin"
        else:
            keys, chosen, ranking = None, list(candidates), "all_candidates"
        results = _fit_many(engine, [np.delete(X, d, axis=0) for d in chosen],
                            [f"{label}.del{d}" for d in chosen], threads)
        fits.extend(result[1] for result in results)
        values = [result[1]["F"] for result in results]
        pick = _best_refit(chosen, values)
        smaller, fit = results[pick]
        founder = chosen[pick]
        smaller_scores = engine.fit_scores(fit["label"], smaller, threads)
        shadow_scores, sequence = _sequence_scores(engine, geometry, X, scores,
                                                   np.delete(X, founder, axis=0), threads)
        decision = count_criterion.decide_with_sequence(
            scores, smaller_scores, shadow_scores, cost, kind="deletion", config=config,
            seed_key=[*seed_key, label, "deletion"])
        entry = _decision_entry("deletion", label, K, decision, ranking, keys, chosen, values,
                                pick, X, smaller, cost)
        entry.update(optimistic_score=optimistic, without_sequence=without,
                     shadow_cells=sequence["shadow_cells"])
        if decision.accept_larger:
            entry["hidden_sequence_failures"] = hidden_failures(X, scores, founder)
        remember(smaller, fit, smaller_scores)
        return entry, (smaller, fit), smaller_scores

    def hidden_failures(X, scores, decided):
        """Diagnostics only: founders other than the decided one whose sequence part fails the
        effective-carrier test (m <= 0 or m' <= 0 on q) at a kept deletion round."""
        failing = []
        for f in range(len(X)):
            if f == decided:
                continue
            shadow_scores, _ = _sequence_scores(engine, geometry, X, scores,
                                                np.delete(X, f, axis=0), threads)
            margin, loso = _point_margins(scores, shadow_scores, cost)
            if not (margin > 0. and loso > 0.):
                failing.append(dict(founder=int(f), margin=float(margin), loso_margin=float(loso)))
        return failing

    def addition_round(X, scores):
        nonlocal round_index
        K = len(X)
        label = f"r{round_index}"
        round_index += 1
        F = float(scores.sum())
        with parallel.numba_thread_scope(int(resolve_threads(threads))):
            bounds, _, per_block_sample = founder_count_bound.upper_bound(
                ws.leaves, ws.offsets, ws.logs, ws.penalty, K + 1)
        optimistic = float(bounds.sum())
        margin = 1e-8 * max(1.0, abs(F), abs(optimistic), abs(K * cost - 2.0 * F))
        if count_criterion.addition_bound_excludes(optimistic, F, cost, samples, margin=margin):
            return dict(kind="addition", round=label, founders=K, skipped_by_bound=True,
                        optimistic_score=optimistic, accept_larger=False, ambiguous=False), None, None
        # No sample's optimistic gain reaches the per-carrier cost c/(2S): no seed's refit (nor
        # any (K+1)-panel) can pass or make the call ambiguous, so the round is decided unrun.
        per_sample_optimistic = per_block_sample.sum(axis=0)
        tolerance = 1e-9 * max(1.0, abs(F), float(np.max(np.abs(per_sample_optimistic))))
        excluded, gain_bound, gain_sample = count_criterion.addition_per_sample_bound_excludes(
            per_sample_optimistic, scores, cost, tolerance=tolerance)
        if excluded:
            return dict(kind="addition", round=label, founders=K, skipped_by_bound=True,
                        optimistic_score=optimistic, bound="per_sample_effective_carrier",
                        per_sample_gain_bound=gain_bound, per_sample_gain_bound_sample=gain_sample,
                        threshold_per_carrier=cost / (2.0 * samples), accept_larger=False,
                        ambiguous=False), None, None
        starts = [np.vstack((X, X[f:f + 1])) for f in range(K)]
        responses = gains = None
        if K > ranked_refits:
            # Seeds ranked by the exact F gain of their copy's best responses (ties: lower
            # seed); the top r are converged.
            states = (K + 1) * (K + 2) // 2
            per_task = 24 * (engine.blocks + 1) * samples * states + 256 * 1024 ** 2
            responses = _parallel_map(
                [lambda budget, start=start: engine.copy_response(start, budget or threads)
                 for start in starts], threads, per_task)
            gains = [response[0] for response in responses]
            chosen = sorted(range(K), key=lambda f: (-gains[f], f))[:ranked_refits]
            ranking = "copy_best_response_gain"
        else:
            chosen, ranking = list(range(K)), "all_candidates"
        results = _fit_many(engine, [starts[f] for f in chosen],
                            [f"{label}.add{f}" for f in chosen], threads)
        fits.extend(result[1] for result in results)
        values = [result[1]["F"] for result in results]
        pick = _best_refit(chosen, values)
        larger, fit = results[pick]
        larger_scores = engine.fit_scores(fit["label"], larger, threads)
        # The refit may have moved any founder: its sequence part is the gain over the refit
        # with every cell the incumbent does not use replaced by the incumbent's nearest row.
        shadow_scores, sequence = _sequence_scores(engine, geometry, larger, larger_scores, X,
                                                   threads)
        decision = count_criterion.decide_with_sequence(
            larger_scores, scores, shadow_scores, cost, kind="addition", config=config,
            seed_key=[*seed_key, label, "addition"])
        entry = _decision_entry("addition", label, K, decision, ranking, gains, chosen, values,
                                pick, larger, X, cost)
        if responses is not None:
            entry["copy_responses"] = [response[1] for response in responses]
        entry.update(optimistic_score=optimistic, without_sequence=sequence["without_sequence"],
                     shadow_cells=sequence["shadow_cells"])
        remember(larger, fit, larger_scores)
        return entry, (larger, fit), larger_scores

    final = record
    # Founder count at the start and after every accepted count change, and the deletions of
    # a founder without a private cell (cleanup, not a decision): cycle diagnostics only
    # (count_criterion.pre_cycle_count).
    sizes = [incoming]
    cleanup_steps = set()
    # Deletion rounds until one is rejected; like production's count pass, panels of
    # at most two founders are not reduced.
    while len(X) > 2:
        entry, smaller, smaller_scores = deletion_round(X, scores)
        decisions.append(entry)
        if entry["accept_larger"]:
            break
        (X, final), scores = smaller, smaller_scores
        sizes.append(len(X))
        if entry["selected"] in entry["without_sequence"]:
            cleanup_steps.add(len(sizes) - 1)
    cycle = False
    cycle_undecided = False
    while additions:
        if len(X) >= backstop["upper"]:
            backstop["reached"] = True
            break
        entry, larger, larger_scores = addition_round(X, scores)
        decisions.append(entry)
        if not entry["accept_larger"]:
            break
        if len(larger[0]) <= 2:
            (X, final), scores = larger, larger_scores
            sizes.append(len(X))
            continue
        check, smaller, smaller_scores = deletion_round(larger[0], larger_scores)
        check["after_addition"] = True
        decisions.append(check)
        if check["accept_larger"]:
            (X, final), scores = larger, larger_scores
            sizes.append(len(X))
            continue
        # Adding then deleting: a count cycle. The deletion round stands (the K+1 panel
        # is not supported against its own best K refit) and the better-F of the two K
        # panels (the incumbent and that refit) is kept; the call is reported as ambiguous,
        # and as undecided when the deletion round's call is ambiguous and the cycle began with
        # a decided deletion from K + 1 (count_criterion.pre_cycle_count).
        cycle = True
        origin = count_criterion.pre_cycle_count(sizes + [len(larger[0])], len(larger[0]),
                                                 cleanup_steps)
        cycle_undecided = bool(origin >= len(larger[0]) and check["ambiguous"])
        if float(smaller_scores.sum()) > float(scores.sum()) + TOL:
            (X, final), scores = smaller, smaller_scores
        entry["ambiguous"] = check["ambiguous"] = True
        entry["count_cycle"] = check["count_cycle"] = True
        entry["cycle_start_count"] = check["cycle_start_count"] = int(origin)
        entry["cycle_undecided"] = check["cycle_undecided"] = cycle_undecided
        break
    replaced = None
    held = best_at[len(X)]
    if held[0] > float(scores.sum()) + TOL:
        # The pass returned to a K it had held with a better converged panel: the count
        # decisions chose K, the fixed-K objective chooses the panel within K.
        replaced = dict(from_fit=final["label"], to_fit=held[2]["label"],
                        gain=float(held[0] - float(scores.sum())))
        X, final, scores = held[1], held[2], held[3]
        replaced.update(_restored_comparisons(engine, geometry, threads, X, scores, best_at,
                                              decisions, cost, config, seed_key))
    if backstop["reached"]:
        warnings.warn(f"converged founder count pass {engine.prefix} reached its count backstop "
                      f"K = {backstop['upper']}", RuntimeWarning)
    return X, dict(founders_before=int(incoming), founders_after=int(len(X)),
                   F=float(scores.sum()), cost=float(cost), fits=fits, decisions=decisions,
                   final_fit=final["label"], weak_label_junctions=final["weak_junctions"],
                   count_cycle=cycle, count_cycle_undecided=cycle_undecided,
                   count_backstop=backstop, additions=bool(additions),
                   ambiguous=(any(d.get("ambiguous") for d in decisions)
                              or bool(replaced is not None and replaced["ambiguous"])),
                   best_f_replacement=replaced)


def _fit_summary(fit):
    """Compact per-fit census record (the full sweep history stays in the checkpoint)."""
    return dict(label=fit["label"], founders=fit["founders"], converged=fit["converged"],
                sweeps=fit["sweeps"], max_frontier=fit["max_frontier"], gain=fit["gain"],
                accepted=fit["accepted"], wall=fit["wall"], worker_cpu=fit["worker_cpu"],
                thread_cpu=fit.get("thread_cpu"), max_deviation=fit.get("max_deviation"),
                first=fit.get("first"), best_of_two=fit.get("best_of_two"))


# ---------------------------------------------------------------------- integration
def _reconstruct(selected, score, batch, original):
    """Refined prepared-row paths as a super-block; the original component keeps its outer
    missing-aware phase-break annotations."""
    from .. import founder_refinement as refiner, hierarchy, paths
    reconstruction = paths.reconstruct_haplotypes_from_beam(
        [(list(row), score) for row in selected], refiner._LeafKeyMap(batch), batch)
    result = hierarchy.convert_reconstruction_to_superblock(reconstruction, batch)
    for side in ("before", "after"):
        for stem in ("missing_aware_break", "missing_aware_break_reason",
                     "missing_aware_joint_informative_samples"):
            name = f"{stem}_{side}"
            default = False if stem == "missing_aware_break" else None
            setattr(result, name, getattr(original, name, default))
    return result


def refine_components(prepared_blocks, components, neutral_probs, global_sites, *, config,
                      num_threads=1, checkpoints=None, l1_blocks=None, cc_scale=0.5,
                      additions=True):
    """Converged refinement and count decisions for each component (one level).

    additions: True at the final count step (deletion and addition rounds), False at
    earlier levels (deletion rounds only).
    """
    from .. import founder_refinement as refiner, chimera_scoring
    from ...core import haplotypes
    from .workspace import component_workspace
    prepared = list(prepared_blocks)
    starts = {int(block.positions[0]): index for index, block in enumerate(prepared)}
    ends = {int(block.positions[-1]): index + 1 for index, block in enumerate(prepared)}
    results, diagnostics = [], []
    workspaces = {}
    for number, component in enumerate(components):
        token = f"converged_sweep.component{number}"
        cached = refiner._load(checkpoints, f"{token}.result")
        if cached is not None:
            results.append(cached["block"])
            diagnostics.append(cached["diagnostic"])
            continue
        started = time.perf_counter()
        batch = prepared[starts[int(component.positions[0])]:ends[int(component.positions[-1])]]
        positions = np.concatenate([block.positions for block in batch])
        if not np.array_equal(positions, component.positions):
            raise ValueError("founder refinement cannot split or reorder a prepared block")
        selected = refiner._local_selection(component, batch)
        if len(batch) < 2 or len(selected) < 2:
            results.append(component)
            diagnostics.append(dict(component=number, changed=False,
                                    reason="single_block_or_founder", search="converged_sweep"))
            continue
        # The component's set-up kernels (evidence gathering and emissions, block models, the
        # reversed orientation) run on its whole share, also in a pool worker (whose Numba
        # threads stay at one between tasks).
        with parallel.numba_thread_scope(int(resolve_threads(num_threads))):
            workspace = component_workspace(workspaces, batch, neutral_probs, global_sites,
                                            config.proposal_max_bins, num_threads,
                                            minimum_bin_size=config.proposal_min_sites_per_bin)
            context = refiner._macro_context(batch, l1_blocks)
            engine = SweepEngine(workspace, context, config, checkpoints, token)
        cost = float(chimera_scoring.compute_cc(batch, len(neutral_probs), cc_scale))
        # Deterministic bootstrap streams per component and decision (positions, size).
        seed_key = [int(positions[0]), int(positions[-1]), int(len(batch)), int(len(selected))]
        rows, count = count_pass(engine, selected, cost, num_threads, seed_key=seed_key,
                                 additions=additions)
        changed = not (rows.shape == selected.shape and np.array_equal(rows, selected))
        result = component
        if changed:
            result = _reconstruct(rows, count["F"], batch, component)
        start_fit = count["fits"][0]
        diagnostic = dict(
            component=number, changed=changed, search="converged_sweep",
            founders=int(len(rows)), founders_before=int(len(selected)),
            original_blocks=len(batch),
            changed_local_rows=(int(np.count_nonzero(rows != selected))
                                if rows.shape == selected.shape else None),
            initial_likelihood=start_fit["F0"], final_likelihood=count["F"],
            proposal_bin_size=workspace.bin_size, switch_penalty=workspace.penalty,
            coarse_units=None if engine.coarse is None else len(engine.coarse[0]),
            coarse_move_guard=engine.guard, complexity_cost=cost,
            weak_label_junctions=count["weak_label_junctions"],
            fits=[_fit_summary(fit) for fit in count["fits"]],
            nonconverged_fits=[fit["label"] for fit in count["fits"] if not fit["converged"]],
            decisions=count["decisions"], count_cycle=count["count_cycle"],
            count_cycle_undecided=count["count_cycle_undecided"],
            count_backstop=count["count_backstop"], count_additions=count["additions"],
            ambiguous_count=count["ambiguous"],
            final_fit=count["final_fit"], best_f_replacement=count["best_f_replacement"],
            elapsed_seconds=time.perf_counter() - started)
        refiner._save(checkpoints, f"{token}.result", dict(block=result, diagnostic=diagnostic))
        results.append(result)
        diagnostics.append(diagnostic)
    return haplotypes.BlockResults(results), dict(
        enabled=True, model="full_site_potts_converged_sweep_v1",
        switch_penalty=founder_scoring.SWITCH_PENALTY,
        candidate_rows="original_prepared_inference_panels",
        search_approximation="certified_local_optimum_fine8_coarse4_exchanges",
        acceptance="exact_local_full_site_gain_gauss_seidel",
        count_criterion=count_criterion.CRITERION, count_additions=bool(additions),
        components=diagnostics)


def convergence_summary(diagnostics):
    """Census of one level's converged refinement (for logs and run summaries)."""
    fits = nonconverged = count_backstops = cycles = cycles_undecided = replacements = 0
    without_sequence = sequence_rejected = hidden = 0
    sweeps, frontiers, ambiguous, weak = [], [], [], 0
    for item in diagnostics.get("components", []):
        for fit in item.get("fits", ()):
            fits += 1
            nonconverged += int(not fit["converged"])
            sweeps.append(int(fit["sweeps"]))
            frontiers.append(int(fit["max_frontier"]))
        count_backstops += int(bool(item.get("count_backstop", {}).get("reached")))
        cycles += int(bool(item.get("count_cycle")))
        cycles_undecided += int(bool(item.get("count_cycle_undecided")))
        weak += len(item.get("weak_label_junctions", ()))
        for decision in item.get("decisions", ()):
            without_sequence += int(bool(decision.get("without_sequence"))
                                    and decision["kind"] == "deletion")
            criterion = decision.get("criterion") or {}
            sequence_rejected += int(not criterion.get("accept_larger", True)
                                     and criterion.get("total", {}).get("passes", False)
                                     and not criterion.get("sequence", {}).get("passes", True))
            hidden += len(decision.get("hidden_sequence_failures") or ())
            if decision.get("ambiguous"):
                ambiguous.append(dict(component=item.get("component"), kind=decision["kind"],
                                      round=decision["round"], margin=decision.get("margin"),
                                      band=decision.get("band")))
        replaced = item.get("best_f_replacement")
        if replaced is not None:
            replacements += 1
            if replaced.get("ambiguous"):
                test = replaced.get("own_deletion_test") or {}
                ambiguous.append(dict(component=item.get("component"),
                                      kind="best_f_restored_deletion", round=test.get("round"),
                                      margin=test.get("margin"), band=test.get("band")))
    return dict(fits=fits, nonconverged_fits=nonconverged, count_backstop_hits=count_backstops,
                count_cycles=cycles, count_cycles_undecided=cycles_undecided,
                best_f_replacements=replacements,
                deletions_without_sequence=without_sequence,
                decided_by_sequence_part=sequence_rejected,
                hidden_sequence_failures=hidden,
                max_sweeps=max(sweeps or [0]),
                mean_sweeps=float(np.mean(sweeps)) if sweeps else 0.0,
                max_frontier=max(frontiers or [0]), weak_label_junctions=weak,
                ambiguous_count_calls=ambiguous)
