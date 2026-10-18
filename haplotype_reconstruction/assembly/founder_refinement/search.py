"""Fixed-K fit C(X) of one component's founder paths: Gauss-Seidel sweeps to a certified stop.

A sweep (i) computes the exact normalised forward/backward block messages of the panel (a search
updates its previous sweep's messages in place, recomputing per sample only the blocks the last
pass changed and those whose start message changed; over consecutive sample ranges); (ii) computes
the window proposals of the panel at the start of the sweep (proposals.py; a window whose inputs
are bitwise unchanged since the search last solved it reuses that solve): for every founder and
every fine window the canonical exact maximiser of the binned surrogate, and where an L1 context
exists for every founder and every coarse window the same over L1 span tuples; (iii) runs one exact
acceptance pass (acceptance.py), its direction alternating between sweeps; (iv) recomputes the
exact F, which must equal the start F plus the accepted gains. A sweep that accepts nothing stops
the search: every proposal of that sweep was computed against the final panel and every move of the
move set was scored exactly against it, so the result is a certified local optimum of F under that
move set (relative to the surrogate's proposals; F is non-convex and no global optimum is claimed).
The sweep limit only marks a fit non-converged, never silently.

Best of two: every fit runs this search twice from the same start, concurrently, once with its
first pass forward and once reversed (both then alternate; their common sweep-0 messages and
proposals are computed once). Each is a deterministic local search; the fit keeps the higher exact
F (exact ties: the forward-first one), so at a fixed start it never ends below the forward-first
search.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import warnings

import numpy as np
from numba.typed import List

from . import messages as kernels
from .acceptance import MOVE_COARSE, MOVE_FINE, accept_pass
from .objective import SWITCH_PENALTY, score_panel, selected_alleles
from .proposals import WindowSolver
from .workspace import sample_ranges
from ... import execution

# Junctions whose best suffix exchange costs less than one switch are reported as weak.
WEAK_JUNCTION_MARGIN = SWITCH_PENALTY
_MOVE_CODES = {"fine": MOVE_FINE, "coarse": MOVE_COARSE}
_TOTAL_KEYS = ("fine", "coarse", "exchange", "fine_gain", "coarse_gain", "exchange_gain",
               "evaluated", "false_positive", "guard_rejected", "guard_shared_tests",
               "guard_relaxed", "guard_shared_switches", "guard_traced_blocks", "changed_blocks")


@dataclass(frozen=True)
class WeakJunction:
    """A run of junctions whose best founder-pair suffix exchange costs less than one switch.

    first, last: junctions (prepared-block boundaries 1 .. B-1 of the component, inclusive) with the
    same best pair, merged when consecutive. pair: the two founder rows (a < b) of that exchange.
    best_gain: the largest exact F change of the exchange over the run (> -SWITCH_PENALTY), exact
    against the final panel of the fit (its certifying sweep).
    """

    first: int
    last: int
    pair: tuple[int, int]
    best_gain: float


@dataclass(frozen=True)
class FitSummary:
    """One converged fixed-K fit (the better of its forward-first and reverse-first searches).

    label: the fit's name in the count pass ("start", "r<i>.del<f>", "r<i>.add<f>").
    founders: K. F0, F: exact objective of the start and the result (log units).
    converged: the kept search ended with a sweep that accepted nothing (else it reached
    FounderRefinementConfig.sweep_limit). sweeps: sweeps of the kept search.
    first: "fwd" | "rev", the kept search (exact F ties keep "fwd"). F_fwd, F_rev: both results.
    max_frontier: largest window-programme frontier; frontier_fallbacks: window programmes that
    kept the incumbent because their frontier exceeded FounderRefinementConfig.frontier_bytes.
    accepted: totals over the kept search's sweeps (fine, coarse, exchange moves and gains,
    evaluated candidates, guard counters). max_deviation: largest |F_new - (F + accepted gains)|
    of a sweep (the exactness check). wall, worker_cpu, thread_cpu: seconds (both searches).
    """

    label: str
    founders: int
    F0: float
    F: float
    converged: bool
    sweeps: int
    first: str
    F_fwd: float
    F_rev: float
    max_frontier: int
    frontier_fallbacks: int
    accepted: dict
    max_deviation: float
    wall: float
    worker_cpu: float
    thread_cpu: float


@dataclass(frozen=True)
class Fit:
    """A converged fit: its panel (int64 K x B prepared-row indices), summary and weak junctions."""

    rows: np.ndarray
    summary: FitSummary
    weak_junctions: tuple[WeakJunction, ...]


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
        """int8 (K, L) alleles of the panel (K x B rows in this orientation's block order)."""
        return selected_alleles(self.leaves, self.offsets, np.ascontiguousarray(panel, np.int64))


class ProposalCache:
    """One search's last solved window inputs and results (SweepEngine.proposals).

    Invariant: the cached result of template t was solved from the rows windows[t][0] and the
    boundary messages fwd[a_t], bwd[c_t] held here (a boundary's messages are replaced only when
    every template ending there is re-solved). Every result is a pure function of its inputs (the
    frontier cap is a fixed setting), fallbacks included."""

    def __init__(self, source=None):
        self.fwd = {} if source is None else dict(source.fwd)
        self.bwd = {} if source is None else dict(source.bwd)
        self.windows = {} if source is None else dict(source.windows)
        self._fwd_same = self._bwd_same = None

    def lookup(self, spans, X, fwd, bwd, results):
        """Fill results[t] = (cached result, 0 CPU) for templates with unchanged inputs; return the
        others' indices."""
        self._fwd_same, self._bwd_same = {}, {}
        todo = []
        for index, (a, c) in enumerate(spans):
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

    def store(self, spans, X, fwd, bwd, todo, computed):
        for index, (result, _) in zip(todo, computed):
            a, c = spans[index]
            self.windows[index] = (X[:, a:c].copy(), result)
            if not self._fwd_same.get(a, False):
                self.fwd[a] = fwd[a].copy()
                self._fwd_same[a] = True
            if not self._bwd_same.get(c, False):
                self.bwd[c] = bwd[c].copy()
                self._bwd_same[c] = True


def _weak_junctions(gains, pairs, reverse, blocks):
    """Junctions whose best suffix exchange costs less than one switch penalty, consecutive ones
    with the same founder pair merged (exact against the final panel in the certifying sweep)."""
    records = []
    for p in range(1, blocks):
        if not gains[p] > -WEAK_JUNCTION_MARGIN:
            continue
        junction = blocks - p if reverse else p
        records.append((junction, int(pairs[p][0]), int(pairs[p][1]), float(gains[p])))
    records.sort()
    merged = []
    for junction, a, b, gain in records:
        if merged and merged[-1][2] == (a, b) and merged[-1][1] == junction - 1:
            merged[-1] = (merged[-1][0], junction, (a, b), max(merged[-1][3], gain))
        else:
            merged.append((junction, junction, (a, b), gain))
    return tuple(WeakJunction(first, last, pair, gain) for first, last, pair, gain in merged)


def _totals(history):
    return {key: sum(entry[key] for entry in history) for key in _TOTAL_KEYS}


# The pair-dosage table (L x Q int8) of the exact score is built up to this size, else the direct
# kernel runs; both give the same values, so this is a speed choice only (a fixed constant: the
# component task's memory estimate covers it).
PAIR_TABLE_BYTES = 1 << 31


class SweepEngine:
    """The fixed inputs of every converged fit of one component (workspace, window solver,
    both orientations, fit records)."""

    def __init__(self, workspace, context, config, records=None, name="component"):
        self.workspace = workspace
        self.config = config
        self.windows = WindowSolver(workspace.emissions, context, workspace.penalty)
        self.penalty = float(workspace.penalty)
        self.blocks = workspace.blocks
        # The shared-junction guard applies to coarse moves, which exist with an L1 context.
        self.guarded = context is not None
        self.forward = Orientation(workspace, False)
        self._reversed = Orientation(workspace, True)
        # The component's fit-state records (load/save by fit label), or None.
        self.records = records
        self.name = name
        spans = self.windows.spans
        self.cached_boundaries = len({a for a, _ in spans}) + len({c for _, c in spans})
        # Per-sample scores of converged fits by label, kept from their last sweep (fit_scores).
        self._fit_scores = {}

    def _load(self, label):
        return None if self.records is None else self.records.load(label)

    def _save(self, label, payload):
        if self.records is not None:
            self.records.save(label, payload)

    # ------------------------------------------------------------------ scores
    def per_sample(self, panel, budget):
        """Per-sample exact scores F_s of the panel, over consecutive sample ranges at the budget's
        live share re-read before each range."""
        ws = self.workspace
        with execution.numba_threads(max(1, int(budget.threads()))):
            alleles = selected_alleles(ws.leaves, ws.offsets, np.ascontiguousarray(panel, np.int64))
            return np.asarray(score_panel(alleles, ws.logs, ws.penalty,
                                          sample_ranges(ws.samples, budget), PAIR_TABLE_BYTES))

    def score(self, panel, budget):
        return float(self.per_sample(panel, budget).sum())

    def fit_scores(self, label, panel, budget):
        """Per-sample scores of the converged fit `label` (panel `panel`): the scores its last sweep
        computed, or computed now (a fit loaded from its record)."""
        scores = self._fit_scores.pop(label, None)
        if scores is None:
            scores = self.per_sample(panel, budget)
        return scores

    def fit_bytes(self, founders):
        """Memory of one fixed-K search (its reservation, execution.memory): the message arrays its pass holds
        (stored backward values, + counts and the forward history of the guard), its proposal cache's
        boundaries, and the pass transients per junction (f/g starts and ends of up to 3K candidates, the
        lookahead and backward chain: K30 map 3.2), plus 32 MiB."""
        states = founders * (founders + 1) // 2
        arrays = 6 if self.guarded else 2
        boundaries = (self.blocks + 1) * arrays + self.cached_boundaries
        samples = self.workspace.samples
        transients = 8 * samples * states * (3 * 3 * founders + 2 * 48)
        return 8 * boundaries * samples * states + transients + (32 << 20)

    # ------------------------------------------------------------------ messages and proposals
    def _full_messages(self, X, lex, budget):
        """messages(_lex) of panel X: (fwd, bwd, fcnt, bcnt, fscale, bscale), counts None without
        lex; over consecutive sample ranges at the budget's live share."""
        logs, offsets = self.workspace.logs, self.forward.offsets
        with execution.numba_threads(max(1, int(budget.threads()))):
            haps = self.forward.haps(X)
        arrays = kernels.message_arrays(len(X), self.blocks, len(logs), lex)
        fwd, bwd, fcnt, bcnt, fscale, bscale = arrays
        for s0, s1, n in sample_ranges(len(logs), budget):
            with execution.numba_threads(n):
                if lex:
                    kernels.messages_lex(haps, logs, offsets, self.penalty, s0, s1, fwd, bwd, fcnt,
                                         bcnt, fscale, bscale)
                else:
                    kernels.messages(haps, logs, offsets, self.penalty, s0, s1, fwd, bwd, fscale,
                                     bscale)
        return arrays

    def _messages(self, X, budget, held=None):
        """The messages of panel X (with switch counts under the guard) and their wall time.

        held: (panel, messages) of an earlier panel, owned by the caller: updated in place to X's
        where X differs from that panel (update_messages(_lex), bitwise the full computation); a
        held backward half of None is computed whole."""
        started = time.perf_counter()
        if held is None:
            return self._full_messages(X, self.guarded, budget), time.perf_counter() - started
        logs, offsets = self.workspace.logs, self.forward.offsets
        with execution.numba_threads(max(1, int(budget.threads()))):
            haps = self.forward.haps(X)
        panel, (fwd, bwd, fcnt, bcnt, fscale, bscale) = held
        known = bwd is not None
        if not known:
            bwd, bscale = np.zeros_like(fwd), np.zeros_like(fscale)
            bcnt = None if fcnt is None else np.zeros_like(fcnt)
        changed = np.any(panel != X, axis=0)
        for s0, s1, n in sample_ranges(len(logs), budget):
            with execution.numba_threads(n):
                if self.guarded:
                    kernels.update_messages_lex(haps, logs, offsets, self.penalty, changed, fwd, bwd,
                                                fcnt, bcnt, fscale, bscale, known, s0, s1)
                else:
                    kernels.update_messages(haps, logs, offsets, self.penalty, changed, fwd, bwd,
                                            fscale, bscale, known, s0, s1)
        return (fwd, bwd, fcnt, bcnt, fscale, bscale), time.perf_counter() - started

    def proposals(self, X, fwd, bwd, budget, cache=None):
        """Every window template's proposals for panel X with messages fwd, bwd.

        cache (a search's ProposalCache, updated in place): a window problem is a pure function of
        the panel's rows in the window and the messages at its ends, and normalised max-plus
        messages of panels that differ only further away are often bitwise equal, so a template
        whose inputs equal the cached ones takes its cached result. The templates to solve run one
        Numba thread each, as many at once as the live share allows (budget.thread_map)."""
        frontier_bytes = self.config.frontier_bytes

        def run(index):
            started = time.thread_time()
            result = self.windows.solve(index, X, fwd, bwd, frontier_bytes)
            return result, time.thread_time() - started

        spans = self.windows.spans
        results = [None] * len(spans)
        todo = (list(range(len(spans))) if cache is None
                else cache.lookup(spans, X, fwd, bwd, results))
        windows = self.windows
        sizes = [windows.template_bytes(*windows.templates[index], self.workspace.samples, len(X))
                 for index in todo]
        computed = budget.thread_map(run, todo, item_bytes=sizes, label="founder_refinement window solves")
        for index, value in zip(todo, computed):
            results[index] = value
        if cache is not None:
            cache.store(spans, X, fwd, bwd, todo, computed)
        proposals = []
        stats = dict(windows=0, certified=0, dp=0, fixing_dps=0, max_frontier=0, fallbacks=0,
                     changed=0, worker_cpu=0.0)
        for result, cpu in results:
            stats["worker_cpu"] += cpu
            for kind, a, c, k, rows, diagnostic in result:
                stats["windows"] += 1
                stats["certified"] += int(diagnostic["certified"])
                stats["dp"] += int(diagnostic["dp"])
                stats["fixing_dps"] += diagnostic["fixing_dps"]
                stats["max_frontier"] = max(stats["max_frontier"], diagnostic["frontier"])
                stats["fallbacks"] += int(diagnostic["fallback"])
                if rows is not None:
                    stats["changed"] += 1
                    proposals.append((kind, a, c, k, np.asarray(rows, np.int64)))
        return proposals, stats

    def copy_response(self, Y, budget):
        """Exact F gain of the copy's fine-window best responses (an addition round's seed).

        Y is a duplicate-seed panel whose last row copies a founder. Every fine window's canonical
        best response of the copy alone (the other rows fixed, messages of Y) is applied by one
        copy-only acceptance pass in block order (exact gains above TOL, no exchanges, no guard).
        Returns (gain, record)."""
        copy = len(Y) - 1
        fwd, bwd, _, _, _, bscale = self._full_messages(Y, False, budget)
        fine = [(a, c) for kind, a, c in self.windows.templates if kind == "fine"]
        frontier_bytes = self.config.frontier_bytes
        sizes = [self.windows.template_bytes("fine", a, c, self.workspace.samples, len(Y)) for a, c in fine]
        solved = budget.thread_map(
            lambda span: self.windows.fine(Y, fwd, bwd, span[0], span[1], frontier_bytes,
                                           focals=(copy,)), fine, item_bytes=sizes,
            label="founder_refinement window solves")
        props, windows = {}, 0
        for result in solved:
            for _, a, c, _, rows, _ in result:
                windows += 1
                if rows is not None:
                    props.setdefault(a, []).append((MOVE_FINE, c, copy, np.asarray(rows, np.int64)))
        acc, _, _ = accept_pass(self.forward, np.ascontiguousarray(Y, np.int64).copy(), bwd, None,
                                bscale, props, self.penalty, False, budget, exchanges=False,
                                start_fwd=fwd)
        return float(acc["fine_gain"]), dict(windows=windows,
                                             proposals=int(sum(map(len, props.values()))),
                                             accepted=int(acc["fine"]), gain=float(acc["fine_gain"]))

    def _sweep_inputs(self, X, budget, cache, held=None):
        """The messages of panel X and its window proposals (pure functions of the panel)."""
        message_state, message_wall = self._messages(X, budget, held)
        started = time.perf_counter()
        proposals, stats = self.proposals(X, message_state[0], message_state[1], budget, cache)
        return dict(messages=message_state, proposals=proposals, stats=stats,
                    message_wall=message_wall, proposal_wall=time.perf_counter() - started)

    # ------------------------------------------------------------------ fixed-K fit
    def converge(self, start, label, budget):
        """C(start): the better of the forward-first and the reverse-first search (module
        docstring). Both run concurrently, each with its share of the budget; the higher exact F is
        kept (exact ties: forward-first). Returns a Fit."""
        done = self._load(f"{label}.done")
        if done is not None:
            return Fit(np.asarray(done["rows"], np.int64).copy(), FitSummary(**done["summary"]),
                       tuple(WeakJunction(*item) for item in done["weak_junctions"]))
        started = time.perf_counter()
        # Both searches start from the same panel, so the F (and per-sample scores), messages and
        # window proposals of their sweep 0 (pure functions of the panel) are computed once, with
        # the fit's whole budget, and shared; only their acceptance passes differ. A search that
        # resumes from its own records computes its sweeps itself.
        shared = None
        if all(self._load(f"{label}.{first}.{stage}") is None
               for first in ("fwd", "rev") for stage in ("done", "sweep0")):
            X = np.ascontiguousarray(start, np.int64)
            scores0 = self.per_sample(X, budget)
            cache = ProposalCache()
            shared = dict(F0=float(scores0.sum()), scores0=scores0,
                          inputs=self._sweep_inputs(X, budget, cache), cache=cache,
                          waiting={"fwd", "rev"}, readers=set(), lock=threading.Lock())
        searches = [None, None]
        for index, result in budget.concurrent(
                [lambda child, first=first: self._converge_direction(start, label, first, child,
                                                                     shared)
                 for first in ("fwd", "rev")], self.fit_bytes(len(start)),
                label="founder_refinement searches"):
            searches[index] = result
        (X_fwd, fwd), (X_rev, rev) = searches
        chosen = "rev" if rev["F"] > fwd["F"] else "fwd"
        X, kept = (X_rev, rev) if chosen == "rev" else (X_fwd, fwd)
        scores = {first: self._fit_scores.pop(f"{label}.{first}", None) for first in ("fwd", "rev")}
        if scores[chosen] is not None:
            self._fit_scores[label] = scores[chosen]
        summary = FitSummary(
            label=label, founders=int(kept["founders"]), F0=float(kept["F0"]), F=float(kept["F"]),
            converged=bool(kept["converged"]), sweeps=int(kept["sweeps"]), first=chosen,
            F_fwd=float(fwd["F"]), F_rev=float(rev["F"]), max_frontier=int(kept["max_frontier"]),
            frontier_fallbacks=int(kept["frontier_fallbacks"]), accepted=dict(kept["accepted"]),
            max_deviation=float(kept["max_deviation"]),
            wall=time.perf_counter() - started,
            worker_cpu=float(fwd["worker_cpu"] + rev["worker_cpu"]),
            thread_cpu=float(fwd["thread_cpu"] + rev["thread_cpu"]))
        weak = tuple(WeakJunction(*item) for item in kept["weak_junctions"])
        self._save(f"{label}.done", dict(rows=X.astype(np.int32), summary=vars(summary).copy(),
                                         weak_junctions=[tuple(vars(item).values()) for item in weak]))
        return Fit(X, summary, weak)

    def _converge_direction(self, start, label, first, budget, shared=None):
        """One search from start: sweeps until one accepts nothing (or the reported sweep limit);
        the first pass runs forward (first "fwd") or reversed ("rev"), later passes alternate.
        Records <label>.<first>.sweep<n> (that sweep's state and history entry) and .done. shared:
        the start panel's F and sweep-0 inputs computed once for both searches (converge).
        Returns (rows int64 K x B, search record dict)."""
        label = f"{label}.{first}"
        done = self._load(f"{label}.done")
        if done is not None:
            return np.asarray(done["rows"], np.int64).copy(), done["record"]
        started, started_cpu = time.perf_counter(), time.thread_time()
        X = np.ascontiguousarray(start, np.int64).copy()
        founders, blocks = X.shape
        # Per-sample scores of the current panel X, while known: a pass that accepts nothing leaves
        # the panel, hence its scores, unchanged.
        if shared is not None:
            F0, current = shared["F0"], shared["scores0"]
        else:
            current = self.per_sample(X, budget)
            F0 = float(current.sum())
        F, history, junctions, converged, sweep = F0, [], (), False, 0
        # The search's proposal cache (seeded by the shared sweep 0).
        cache = ProposalCache(None if shared is None else shared["cache"])
        # This search's own messages of the panel its last sweep started from, updated in place by
        # the next sweep: (panel, messages); a reversed pass keeps their forward half only, so the
        # search holds no more message arrays than its forward passes do.
        held = None
        limit = self.config.sweep_limit
        resumed = 0
        while sweep < limit:
            saved = self._load(f"{label}.sweep{sweep}")
            if saved is None:
                break
            X = np.asarray(saved["rows"], np.int64).copy()
            F = saved["F"]
            history.append(saved["entry"])
            junctions = tuple(WeakJunction(*item) for item in saved["junctions"])
            current = None
            sweep += 1
            resumed += 1
            if saved["entry"]["accepted"] == 0:
                converged = True
                break
        while not converged and sweep < limit:
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
                    message_state, message_wall = self._messages(X, budget)
                    inputs = dict(inputs, messages=message_state, message_wall=message_wall)
                else:
                    owned = False
            else:
                inputs = self._sweep_inputs(X, budget, cache, held)
            held = None
            fwd, bwd, fcnt, bcnt, fscale, bscale = inputs["messages"]
            proposals, stats = inputs["proposals"], inputs["stats"]
            message_wall, proposal_wall = inputs["message_wall"], inputs["proposal_wall"]
            del inputs
            reverse = (sweep % 2 == 1) if first == "fwd" else (sweep % 2 == 0)
            props = {}
            for kind, a, c, k, rows in proposals:
                if reverse:
                    props.setdefault(blocks - c, []).append((_MOVE_CODES[kind], blocks - a, k,
                                                             rows[::-1].copy()))
                else:
                    props.setdefault(a, []).append((_MOVE_CODES[kind], c, k, rows))
            pass_started = time.perf_counter()
            with execution.numba_threads(max(1, int(budget.threads()))):
                if reverse:
                    # The reversed pass stores the clamped forward messages; their per-block
                    # normalisers are the forward ones in reversed block order.
                    orientation = self._reversed
                    if self.guarded:
                        stored, stored_cnt = kernels.clamp_reverse_lex(fwd, fcnt, self.penalty)
                    else:
                        stored, stored_cnt = kernels.clamp_reverse(fwd, self.penalty), None
                    stored_scale = np.ascontiguousarray(fscale[::-1])
                    Xo = np.ascontiguousarray(X[:, ::-1])
                    start_fwd = start_fcnt = None
                    kept = (fwd, None, fcnt, None, fscale, None)
                else:
                    orientation, stored, stored_cnt, stored_scale = self.forward, bwd, bcnt, bscale
                    Xo = X.copy()
                    start_fwd, start_fcnt = fwd, fcnt
                    kept = (fwd, bwd, fcnt, bcnt, fscale, bscale)
            del fwd, bwd, fcnt, bcnt, fscale, bscale
            acc, gains, pairs = accept_pass(orientation, Xo, stored, stored_cnt, stored_scale, props,
                                            self.penalty, self.guarded, budget,
                                            start_fwd=start_fwd, start_fcnt=start_fcnt)
            del stored, stored_cnt, stored_scale, start_fwd, start_fcnt
            if sweep == 0 and shared is not None:
                # Run one after the other (a one-thread or memory-limited budget), the other search
                # has not started: its message arrays are not kept alive through this whole search;
                # it recomputes them and keeps the shared proposals. The shared messages become this
                # search's own once no other search reads them.
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
                current = self.per_sample(X, budget)
            F_new = float(current.sum())
            pass_wall = time.perf_counter() - pass_started
            accounted = acc["fine_gain"] + acc["coarse_gain"] + acc["exchange_gain"]
            deviation = F_new - (F + accounted)
            # Exactness check of the accepted gains (class-A fix D3: absolute in F's scale plus a
            # rounding allowance per accepted move, instead of 1e-6 relative to F).
            if abs(deviation) > 1e-9 * max(1.0, abs(F_new)) + 1e-6 * accepted:
                raise RuntimeError(
                    f"founder refinement {self.name}.{label}:{sweep}: exact F changed by "
                    f"{F_new - F}, accepted local gains {accounted}")
            junctions = _weak_junctions(gains, pairs, reverse, blocks)
            entry = dict(sweep=sweep, reverse=bool(reverse), F=F_new, dF=F_new - F,
                         dF_accounted=float(accounted), deviation=float(deviation),
                         accepted=int(accepted), **acc, proposals=stats,
                         wall=time.perf_counter() - sweep_started, message_wall=message_wall,
                         proposal_wall=proposal_wall, pass_wall=pass_wall)
            history.append(entry)
            F = F_new
            self._save(f"{label}.sweep{sweep}", dict(
                rows=X.astype(np.int32), F=F, entry=entry,
                junctions=[tuple(vars(item).values()) for item in junctions]))
            sweep += 1
            if accepted == 0:
                converged = True
        if not converged:
            warnings.warn(f"founder refinement {self.name}.{label} reached its {limit}-sweep limit "
                          "without converging", RuntimeWarning)
        fallbacks = sum(h["proposals"]["fallbacks"] for h in history)
        if fallbacks:
            warnings.warn(f"founder refinement {self.name}.{label}: {fallbacks} window programmes kept the "
                          "incumbent at the frontier cap (FounderRefinementConfig.frontier_bytes)", RuntimeWarning)
        if len(history) > resumed:
            self._fit_scores[label] = current
        record = dict(label=label, founders=int(founders), F0=float(F0), F=float(F),
                      gain=float(F - F0), converged=bool(converged), sweeps=len(history),
                      resumed_sweeps=resumed, sweep_limit=int(limit), first=first,
                      accepted=_totals(history),
                      max_frontier=max([h["proposals"]["max_frontier"] for h in history] or [0]),
                      frontier_fallbacks=fallbacks,
                      max_deviation=max([abs(h["deviation"]) for h in history] or [0.0]),
                      worker_cpu=float(sum(h["proposals"]["worker_cpu"] for h in history)),
                      wall=time.perf_counter() - started,
                      thread_cpu=time.thread_time() - started_cpu,
                      weak_junctions=[tuple(vars(item).values()) for item in junctions],
                      history=history)
        self._save(f"{label}.done", dict(rows=X.astype(np.int32), record=record))
        return X, record
