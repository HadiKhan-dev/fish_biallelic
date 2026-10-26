"""Fixed-K fit C(X) of one component's founder paths: Gauss-Seidel sweeps to a certified stop.

A sweep (i) computes the exact normalised forward/backward block messages of the panel (a search
updates its previous sweep's messages in place, recomputing per sample only the blocks the last
pass changed and those whose start message changed; over consecutive sample ranges); (ii) computes
the window proposals of the panel at the start of the sweep (proposals.py; a window whose inputs
are bitwise unchanged since the search last solved it reuses that solve): for every founder and
every fine window the canonical maximiser of the binned surrogate (exact unless a beam-capped
programme truncates: counted as beam_truncations), and where an L1 context
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
F, decided by tie classes of F (release 2.1: two results within the quantum tie_rules.quantum("refine_F", F)
are tied; release 2.3 retains the family and uses the minimum exact-optimal switch count
as the representative prior, then a naming convention), so at a
fixed start it never ends measurably below the forward-first search. The fit's panel is returned in
canonical founder order (rows sorted lexicographically, design 5.1): founder indices of the count pass
(candidates, shortlists, tie-breaks) are content ranks.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import OrderedDict
from contextlib import contextmanager
import copy
import hashlib
import pickle
import sys
import threading
import time
import warnings

import numpy as np
from numba.typed import List

from . import gains as gains_kernels
from . import messages as kernels
from .. import tie_rules
from ...maths import ties
from .acceptance import AHEAD_BLOCKS, JUNCTION_BATCH, MOVE_COARSE, MOVE_FINE, TOL, accept_pass
from .objective import SWITCH_PENALTY, score_panel, score_switches_direct, selected_alleles
from .proposals import WindowSolver
from .workspace import sample_ranges
from ... import execution

# Precursor content scoring of the bridge search runs in batches of at most this many message bytes.
BRIDGE_BATCH_BYTES = 256 << 20
# Polish: long one-founder moves, beam offers on LONG_WINDOW-block windows at LONG_STRIDE, each
# also tried with one private correction of LONG_CORRECTION blocks (SweepEngine.long_repair).
LONG_WINDOW = 64
LONG_STRIDE = 16
LONG_CORRECTION = 8
# Donors of a long offer's corrections: the LONG_DONORS founders whose rows agree with the offer on most blocks of
# its changed span (ties: founder order), plus the incumbent's own rows; O(1) per offer keeps long_repair cubic in K.
LONG_DONORS = 2
# Acceptance counters of the polish's escape and long-repair passes merged into their sweep's counters.
_POLISH_PASS_KEYS = ("evaluated", "false_positive", "exchange_tie", "content_tie", "tie_wide", "round_cap",
                     "exchange_cap", "content_rounds", "stale_evaluated", "guard_shared_tests", "guard_rejected",
                     "guard_traced_blocks", "guard_relaxed", "guard_shared_switches", "lex_items")
# Beam width of the long windows' offers (the ordinary escape keeps beam.BEAM_WIDTH).
LONG_BEAM_WIDTH = 16
# Junctions whose best suffix exchange costs less than one switch are reported as weak (a fixed
# reporting margin of 20 nats, 2.6's diplotype switch penalty; 2.8 switch costs are
# map-derived).
WEAK_JUNCTION_MARGIN = SWITCH_PENALTY
_MOVE_CODES = {"fine": MOVE_FINE, "coarse": MOVE_COARSE}
_TOTAL_KEYS = ("fine", "coarse", "exchange", "bridge", "escape", "long", "fine_gain", "coarse_gain",
               "exchange_gain", "bridge_gain", "escape_gain", "long_gain", "long_offers", "escape_offers",
               "escape_windows", "escape_beam_cut", "escape_beam_binding", "long_windows", "long_beam_cut",
               "long_beam_binding", "long_candidates",
               "evaluated", "false_positive", "guard_rejected", "guard_shared_tests",
               "guard_relaxed", "guard_shared_switches", "guard_traced_blocks", "changed_blocks",
               "content_rounds", "round_cap", "stale_evaluated", "exchange_cap", "lex_items", "lex_abort")
# Acceptance, escape and long-repair counters reported with every fit's work counters.
_WORK_KEYS = ("guard_shared_tests", "guard_traced_blocks", "content_rounds", "round_cap",
              "stale_evaluated", "exchange_cap", "escape_windows", "escape_beam_cut",
              "escape_beam_binding", "long_windows", "long_beam_cut", "long_beam_binding", "long_candidates",
              "lex_items", "lex_abort")


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
    first: "fwd" | "rev", the kept search after objective and minimum-switch comparison.
    F_fwd, F_rev: both results (F_rev None for a polish fit, which runs the forward-first search only).
    max_frontier: largest window-programme frontier; frontier_fallbacks: window programmes that
    kept the incumbent because their frontier exceeded FounderRefinementConfig.frontier_bytes.
    accepted: totals over the kept search's sweeps (fine, coarse, exchange moves and gains,
    evaluated candidates, guard counters). max_deviation: largest |F_new - (F + accepted gains)|
    of a sweep (the exactness check). wall, worker_cpu, thread_cpu: seconds (both searches).
    tied_panels (release 2.2): when the forward- and reverse-first results are tied on F (one tie class)
    but are different panels, their panel keys (sorted): equally supported alternatives, released flagged;
    the kept one uses the minimum exact-optimal switch count prior, then a canonical
    naming convention. tied_scores and tied_switches align with tied_panels;
    minimum_switches is the chosen prior value. Empty otherwise.
    """

    label: str
    founders: int
    F0: float
    F: float
    converged: bool
    sweeps: int
    first: str
    F_fwd: float
    F_rev: float | None
    max_frontier: int
    frontier_fallbacks: int
    accepted: dict
    max_deviation: float
    wall: float
    worker_cpu: float
    thread_cpu: float
    tied_panels: tuple = ()
    tied_scores: tuple = ()
    tied_switches: tuple = ()
    minimum_switches: int | None = None
    phase_seconds: dict = field(default_factory=dict)
    work_counters: dict = field(default_factory=dict)


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
        self.workspace = workspace
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

    def costs(self, founders):
        """float32 (S, L) switch costs of K = founders in this orientation's site order."""
        return self.workspace.costs(founders, self.reverse)


# Only components with multiple incoming roots retain these optional window results.
# One MiB of the reservation covers the bounded dict/entry bookkeeping. Input-key
# copies and returned proposal arrays are covered by each worker's task estimate.
WINDOW_MEMO_BYTES = 256 << 20
_WINDOW_MEMO_OVERHEAD = 1 << 20
_WINDOW_MEMO_ENTRIES = 4096


class _WindowMemo:
    """Exact window solutions within one immutable component solver.

    Scope fixes emissions, row domains, templates, coarse context and penalty.
    Keys additionally compare every row/flank byte, dtype, shape, frontier cap and
    explicit convention. Dictionary hashing is only an index, never equality.
    Stored serialized results are immutable; each hit returns privately owned
    arrays and diagnostics. Concurrent misses may duplicate work, not decisions.
    No solver or engine reference is retained, avoiding large ownership cycles.
    """

    def __init__(self, limit=WINDOW_MEMO_BYTES - _WINDOW_MEMO_OVERHEAD):
        self.limit = int(limit)
        self.entries = OrderedDict()
        self.lock = threading.Lock()
        self.used = 0
        self.statistics = dict(calls=0, hits=0, evictions=0, peak_entry_bytes=0)

    @staticmethod
    def key(index, a, c, X, fwd, bwd, frontier_bytes, convention):
        def array(value):
            return value.dtype.str, value.shape, value.tobytes(order="C")
        return (int(index), int(frontier_bytes),
                bool(ties.reverse_conventions() if convention is None else convention),
                array(X[:, a:c]), array(fwd[a]), array(bwd[c]))

    @staticmethod
    def transient_bytes(a, c, X, fwd, bwd):
        # Complete input key plus a serialized/private copy of all focal paths
        # and their small diagnostic dictionaries. Existing estimates cover the
        # original solved result and numerical tables.
        return (int(X[:, a:c].nbytes + fwd[a].nbytes + bwd[c].nbytes)
                + 16 * len(X) * (c - a) + 2048 * len(X) + 4096)

    @staticmethod
    def _size(value):
        return sys.getsizeof(value) + (sum(_WindowMemo._size(x) for x in value)
                                      if isinstance(value, tuple) else 0)

    def lookup(self, key):
        with self.lock:
            self.statistics["calls"] += 1
            entry = self.entries.get(key)
            if entry is None:
                return None
            self.entries.move_to_end(key)
            self.statistics["hits"] += 1
        return pickle.loads(entry[0])

    def store(self, key, result):
        payload = pickle.dumps(result, protocol=5)
        size = self._size(key) + sys.getsizeof(payload) + 256
        if size > self.limit:
            return
        with self.lock:
            if key in self.entries:
                return  # Another worker completed the identical problem.
            while self.used + size > self.limit or len(self.entries) >= _WINDOW_MEMO_ENTRIES:
                _, old = self.entries.popitem(last=False)
                self.used -= old[1]
                self.statistics["evictions"] += 1
            self.entries[key] = payload, size
            self.used += size
            self.statistics["peak_entry_bytes"] = max(self.statistics["peak_entry_bytes"], self.used)

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.used = 0


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
    return {key: sum(entry.get(key, 0) for entry in history) for key in _TOTAL_KEYS}


# The pair-dosage table (L x Q int8) of the exact score is built up to this size, else the direct
# kernel runs; both give the same values, so this is a speed choice only (a fixed constant: the
# component task's memory estimate covers it).
PAIR_TABLE_BYTES = 1 << 31


class SweepEngine:
    """The fixed inputs of every converged fit of one component (workspace, window solver,
    both orientations, fit records)."""

    def __init__(self, workspace, context, config, records=None, name="component", counter=None,
                 census_prefix=None, source_start=0):
        self.workspace = workspace
        # Tie-class decisions of this component's fits (tie_rules.SharedCounter or None) and its census prefix.
        self.counter = counter
        self.census_prefix = census_prefix
        self.config = config
        self.windows = WindowSolver(workspace.emissions, context, workspace.penalty,
                                    work_coefficient=config.window_work_coefficient, source_start=source_start)
        self.penalty = float(workspace.penalty)
        self.blocks = workspace.blocks
        # The window-beam proposer on the map-aware binned surrogate for the polish
        # fits (converge(polish=True)); count-pass fits use the window programmes only.
        self.beam = None
        if getattr(config, "polish", False):
            from .beam import WindowBeam
            self.beam = WindowBeam(workspace, context, workspace.genetic_map, workspace.reset_rate,
                                   workspace.map_probability_floor)
        # The shared-junction guard applies to coarse moves, which exist with an L1 context.
        self.guarded = context is not None
        self.forward = Orientation(workspace, False)
        self._reversed = Orientation(workspace, True)
        # The component's fit-state records (load/save by fit label), or None.
        self.records = records
        self._base_records = records
        self.name = name
        spans = self.windows.spans
        self.cached_boundaries = len({a for a, _ in spans}) + len({c for _, c in spans})
        # Per-sample scores of converged fits by label, kept from their last sweep (fit_scores).
        self._fit_scores = {}
        self._minimum_switches = {}
        # Branches replay the original count pass. A fit is reusable only at the
        # same labelled input panel; its alternate direction retains its own
        # diagnostics. Workspace and completed-fit caches are shared read-only
        # between sequential branch continuations.
        self._fit_cache = {}
        self._score_cache = {}
        self._forced_panels = {}
        self._fit_invocations = {}
        self._chosen_panels = {}
        self.reused_fits = 0
        self.branch_key = None
        self._window_memo = None
        self._window_memo_statistics = {}
        # Shallow branch copies share measured work; cached fits contribute no
        # new work. Updates occur only once per directional search, under a lock.
        self._profile_lock = threading.Lock()
        self._phase_seconds = {}
        self._work_counters = {}

    @contextmanager
    def memoized_windows(self, enabled):
        """Optional bounded reuse across incoming roots; memory admission never waits.

        The fixed workspace and WindowSolver are shared by branch engines. Cache
        eviction or declined admission changes runtime only. A retained-state
        reservation is released on success or any failure of the count search.
        """
        if not enabled:
            yield None
            return
        governor = execution.memory_governor()
        live = execution.available_memory_bytes()
        if (WINDOW_MEMO_BYTES > governor.limit - governor.reserved
                or (live is not None and WINDOW_MEMO_BYTES > live - governor.headroom)):
            # Optional reuse must never invoke the governor's run-alone exception.
            yield None
            return
        request = governor.request(WINDOW_MEMO_BYTES, label="founder_refinement window memo")
        memo = None
        try:
            if request.poll():
                governor.rest(request.id)
                memo = _WindowMemo()
                self._window_memo = memo
            else:
                # This optional request will never be polled again. Remove its queue claim now,
                # before the search can make nested requests behind it.
                request.release()
            yield memo
        finally:
            self._window_memo = None
            if memo is not None:
                self._window_memo_statistics = dict(memo.statistics)
                memo.clear()
            request.release()

    def branch(self, choices, input_key=None):
        """A sequential count continuation sharing the evidence and completed fits.

        choices maps (fit label, canonical input panel) to a tied output panel.
        Replaying from the original input reconstructs the same bootstrap labels,
        count-cycle history and best-by-K state before the chosen decision.
        """
        branch = copy.copy(self)
        branch._forced_panels = dict(choices)
        branch._fit_scores = {}
        branch._fit_invocations = {}
        branch._chosen_panels = {}
        branch.reused_fits = 0
        branch.branch_key = hashlib.sha256(repr((input_key, tuple(sorted(choices.items())))).encode()).hexdigest()[:16]
        branch.records = (None if self._base_records is None else
                          self._base_records.scoped(f"branch/{branch.branch_key}", nthreads=1))
        branch.census_prefix = (None if self.census_prefix is None else
                               f"{self.census_prefix}/branch/{branch.branch_key}")
        return branch

    def _choose_fit(self, token, variants):
        wanted = self._forced_panels.get(token)
        fit = variants[0]
        if wanted is not None:
            fit = next(item for item in variants if tie_rules.panel_key(item.rows) == wanted)
            if fit is not variants[0]:
                self._fit_scores.pop(token[0], None)
        self._fit_invocations[token[0]] = token[1]
        self._chosen_panels[token] = tie_rules.panel_key(fit.rows)
        return fit

    def _load(self, label):
        return None if self.records is None else self.records.load(label)

    def _save(self, label, payload):
        if self.records is not None:
            self.records.save(label, payload)

    # ------------------------------------------------------------------ scores
    def per_sample(self, panel, budget):
        """Per-sample exact scores F_s of the panel, over consecutive sample ranges at the budget's
        live share re-read before each range."""
        # Completed fits already hold this exact score vector; do not rescore
        # an original-input branch or count shadow that reaches the same panel.
        cached = self._score_cache.get(tie_rules.panel_key(panel))
        if cached is not None:
            return cached
        ws = self.workspace
        founders = len(panel)
        with execution.numba_threads(max(1, int(budget.threads()))):
            alleles = selected_alleles(ws.leaves, ws.offsets, np.ascontiguousarray(panel, np.int64))
            scores = np.asarray(score_panel(alleles, ws.logs, ws.costs(founders),
                                            sample_ranges(ws.samples, budget), PAIR_TABLE_BYTES))
        # The panel-independent terms of K (start and stay): scores of different K compare.
        return scores + ws.normalisers(founders)

    def score(self, panel, budget):
        return float(self.per_sample(panel, budget).sum())

    def minimum_switches(self, panel, budget):
        """Fewest exact-optimal diplotype switches, summed over the common evidence."""
        key = tie_rules.panel_key(panel)
        found = self._minimum_switches.get(key)
        if found is not None:
            return found
        ws = self.workspace
        with execution.numba_threads(max(1, int(budget.threads()))):
            alleles = selected_alleles(ws.leaves, ws.offsets, np.asarray(key, np.int64))
        total = 0
        costs = ws.costs(len(alleles))
        for s0, s1, threads in sample_ranges(ws.samples, budget):
            with execution.numba_threads(threads):
                _, counts = score_switches_direct(alleles, ws.logs[s0:s1], costs[s0:s1])
            total += int(counts.sum())
        self._minimum_switches[key] = total
        return total

    def fit_scores(self, label, panel, budget):
        """Per-sample scores of the converged fit `label` (panel `panel`): the scores its last sweep
        computed, or computed now (a fit loaded from its record)."""
        key = tie_rules.panel_key(panel)
        scores = self._fit_scores.pop(label, None)
        if scores is None:
            scores = self._score_cache.get(key)
        if scores is None:
            scores = self.per_sample(panel, budget)
        self._score_cache[key] = scores
        return scores

    def fit_bytes(self, founders, polish=False):
        """Memory of one fixed-K search (its reservation, execution.memory): the message arrays its pass holds
        (stored backward values, + counts and the forward history of the guard), its proposal cache's
        boundaries, and the pass transients per junction (f/g starts and ends of up to 3K candidates, the
        lookahead and backward chain: K30 map 3.2), plus 32 MiB. Guard traceback scratch is
        reserved separately during acceptance; it is not resident during window proposals. A polish fit's chains
        and lookahead can span a long repair's LONG_WINDOW blocks."""
        states = founders * (founders + 1) // 2
        arrays = 4 if self.guarded else 2
        boundaries = (self.blocks + 1) * arrays + self.cached_boundaries
        if self.guarded:
            boundaries += 2 * ((self.blocks - 1) // JUNCTION_BATCH + 1)
        samples = self.workspace.samples
        span = max(48, LONG_WINDOW) if polish else 48
        transients = 8 * samples * states * (3 * 3 * founders + 2 * span)
        return 8 * boundaries * samples * states + transients + (32 << 20)

    # ------------------------------------------------------------------ messages and proposals
    def _full_messages(self, X, lex, budget):
        """messages(_lex) of panel X: (fwd, bwd, fcnt, bcnt, fscale, bscale), counts None without
        lex; over consecutive sample ranges at the budget's live share."""
        logs, offsets = self.workspace.logs, self.forward.offsets
        costs = self.forward.costs(len(X))
        with execution.numba_threads(max(1, int(budget.threads()))):
            haps = self.forward.haps(X)
        arrays = kernels.message_arrays(len(X), self.blocks, len(logs), lex)
        fwd, bwd, fcnt, bcnt, fscale, bscale = arrays
        for s0, s1, n in sample_ranges(len(logs), budget):
            with execution.numba_threads(n):
                if lex:
                    kernels.messages_lex(haps, logs, offsets, costs, s0, s1, fwd, bwd, fcnt,
                                         bcnt, fscale, bscale)
                else:
                    kernels.messages(haps, logs, offsets, costs, s0, s1, fwd, bwd, fscale,
                                     bscale)
        return arrays

    def _messages(self, X, budget, held=None, lex=None):
        """The messages of panel X (with switch counts when lex, default: under the guard) and their
        wall time.

        held: (panel, messages) of an earlier panel, owned by the caller: updated in place to X's
        where X differs from that panel (update_messages(_lex), bitwise the full computation); a
        held backward half of None is computed whole. Held messages without switch
        counts are not updated to lexicographic ones (computed whole); held counts are dropped
        when lex is False (the scores of both recursions are the same)."""
        started = time.perf_counter()
        lex = self.guarded if lex is None else bool(lex)
        if held is not None and lex and held[1][2] is None:
            held = None
        if held is None:
            return self._full_messages(X, lex, budget), time.perf_counter() - started
        logs, offsets = self.workspace.logs, self.forward.offsets
        costs = self.forward.costs(len(X))
        with execution.numba_threads(max(1, int(budget.threads()))):
            haps = self.forward.haps(X)
        panel, (fwd, bwd, fcnt, bcnt, fscale, bscale) = held
        if not lex:
            fcnt = bcnt = None
        known = bwd is not None
        if not known:
            bwd, bscale = np.zeros_like(fwd), np.zeros_like(fscale)
            bcnt = None if fcnt is None else np.zeros_like(fcnt)
        changed = np.any(panel != X, axis=0)
        for s0, s1, n in sample_ranges(len(logs), budget):
            with execution.numba_threads(n):
                if lex:
                    kernels.update_messages_lex(haps, logs, offsets, costs, changed, fwd, bwd,
                                                fcnt, bcnt, fscale, bscale, known, s0, s1)
                else:
                    kernels.update_messages(haps, logs, offsets, costs, changed, fwd, bwd,
                                            fscale, bscale, known, s0, s1)
        return (fwd, bwd, fcnt, bcnt, fscale, bscale), time.perf_counter() - started

    def proposals(self, X, fwd, bwd, budget, cache=None, convention=None):
        """Every window template's proposals for panel X with messages fwd, bwd.

        cache (a search's ProposalCache, updated in place): a window problem is a pure function of
        the panel's rows in the window and the messages at its ends, and normalised max-plus
        messages of panels that differ only further away are often bitwise equal, so a template
        whose inputs equal the cached ones takes its cached result. The templates to solve run one
        Numba thread each, as many at once as the live share allows (budget.thread_map)."""
        frontier_bytes = self.config.frontier_bytes

        memo = self._window_memo

        def run(index):
            key = None
            if memo is not None:
                a, c = self.windows.spans[index]
                key = memo.key(index, a, c, X, fwd, bwd, frontier_bytes, convention)
                result = memo.lookup(key)
                if result is not None:
                    return result, 0.0
            started = time.thread_time()
            result = self.windows.solve(index, X, fwd, bwd, frontier_bytes, convention)
            cpu = time.thread_time() - started
            if memo is not None:
                memo.store(key, result)
            return result, cpu

        spans = self.windows.spans
        results = [None] * len(spans)
        todo = (list(range(len(spans))) if cache is None
                else cache.lookup(spans, X, fwd, bwd, results))
        windows = self.windows
        sizes = [windows.template_bytes(*windows.templates[index], self.workspace.samples, len(X))
                 + (0 if memo is None else memo.transient_bytes(*spans[index], X, fwd, bwd))
                 for index in todo]
        # Largest templates first (their workspace estimate tracks their work), so the map's tail holds small
        # solves; each result is stored at its own index, so the order changes timing only.
        order = sorted(range(len(todo)), key=lambda i: (-sizes[i], i))
        solved = budget.thread_map(run, [todo[i] for i in order], item_bytes=[sizes[i] for i in order],
                                   label="founder_refinement window solves")
        computed = [None] * len(todo)
        for i, value in zip(order, solved):
            computed[i] = value
        for index, value in zip(todo, computed):
            results[index] = value
        if cache is not None:
            cache.store(spans, X, fwd, bwd, todo, computed)
        proposals = []
        stats = dict(windows=0, certified=0, dp=0, fixing_dps=0, max_frontier=0, fallbacks=0,
                     changed=0, worker_cpu=0.0, templates_solved=len(todo),
                     templates_reused=len(spans) - len(todo), beam_truncations=0, beam_fallbacks=0)
        for result, cpu in results:
            stats["worker_cpu"] += cpu
            for kind, a, c, k, rows, diagnostic in result:
                stats["windows"] += 1
                stats["certified"] += int(diagnostic["certified"])
                stats["dp"] += int(diagnostic["dp"])
                stats["fixing_dps"] += diagnostic["fixing_dps"]
                stats["max_frontier"] = max(stats["max_frontier"], diagnostic["frontier"])
                stats["fallbacks"] += int(diagnostic["fallback"])
                stats["beam_truncations"] += int(diagnostic.get("beam_truncations", 0))
                stats["beam_fallbacks"] += int(diagnostic.get("beam_fallback", False))
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
        props, windows, truncations, fallbacks = {}, 0, 0, 0
        for result in solved:
            for _, a, c, _, rows, diagnostic in result:
                windows += 1
                # The programmes' approximation receipts (proposals.BEAM_CAP).
                truncations += int(diagnostic.get("beam_truncations", 0))
                fallbacks += int(diagnostic.get("beam_fallback", False))
                if rows is not None:
                    props.setdefault(a, []).append((MOVE_FINE, c, copy, np.asarray(rows, np.int64)))
        acc, _, _ = accept_pass(self.forward, np.ascontiguousarray(Y, np.int64).copy(), bwd, None,
                                bscale, props, self.forward.costs(len(Y)), False, budget,
                                exchanges=False, start_fwd=fwd)
        return float(acc["fine_gain"]), dict(windows=windows,
                                             proposals=int(sum(map(len, props.values()))),
                                             accepted=int(acc["fine"]), gain=float(acc["fine_gain"]),
                                             beam_truncations=truncations, beam_fallbacks=fallbacks,
                                             round_cap=int(acc["round_cap"]),
                                             stale_evaluated=int(acc["stale_evaluated"]))

    def beam_escape(self, X, F, budget, convention):
        """Polish: one exact acceptance pass over the beam bank of panel X at an ordinary fixed point (forward
        orientation, the search's content convention). Returns (panel or None, its per-sample scores, exact
        gain, record, the bank's fine offers as bridge precursors).

        The pass runs on the panel's canonical founder order (rows sorted by content, ties by index) and its
        outcome is mapped back, so an escape depends on the panel only up to founder labels (the objective is
        label-invariant: max-plus messages; per-sample scores are exact under relabelling)."""
        founders = len(X)
        order = sorted(range(founders), key=lambda f: (tuple(int(x) for x in X[f]), f))
        canonical = np.ascontiguousarray(np.asarray(X, np.int64)[order])
        Yc, scores, gain, record, fine = self._beam_escape_canonical(canonical, F, budget, convention)
        Y = None
        if Yc is not None:
            Y = np.empty_like(Yc)
            Y[order] = Yc
        fine = [(a, c, order[k], rows) for a, c, k, rows in fine]
        return Y, scores, gain, record, fine

    def _beam_escape_canonical(self, X, F, budget, convention):
        """beam_escape on a panel in canonical founder order."""
        offers, beam_stats = self.beam.generate(X, budget, convention)
        fine = [(a, c, k, rows) for kind, a, c, k, rows in offers if kind == "fine"]
        record = dict(offers=len(offers), windows=int(beam_stats["windows"]),
                      certified=int(beam_stats["certified"]), beam_cut=int(beam_stats["beam_cut"]),
                      beam_binding=int(beam_stats["beam_binding"]),
                      box_max_gap=float(beam_stats["box_max_gap"]), beam_wall=float(beam_stats["wall"]),
                      accepted=0)
        if not offers:
            return None, None, 0.0, record, fine
        props = {}
        for kind, a, c, k, rows in offers:
            props.setdefault(a, []).append((_MOVE_CODES[kind], c, k, rows))
        Xo = np.ascontiguousarray(X, np.int64).copy()
        # This message set is held beside the search's own (fit_bytes).
        with execution.reserve_memory(self.message_bytes(len(X), self.guarded),
                                      label="founder_refinement beam escape messages"):
            fwd, bwd, fcnt, bcnt, fscale, bscale = self._full_messages(X, self.guarded, budget)
            acc, _, _ = accept_pass(self.forward, Xo, bwd, bcnt, bscale, props, self.forward.costs(len(X)),
                                    self.guarded, budget, start_fwd=fwd, start_fcnt=fcnt, scale=abs(F),
                                    reverse=convention)
            del fwd, bwd, fcnt, bcnt, fscale, bscale
        accepted = int(acc["fine"] + acc["coarse"] + acc["exchange"])
        predicted = float(acc["fine_gain"] + acc["coarse_gain"] + acc["exchange_gain"])
        record.update(accepted=accepted, fine=int(acc["fine"]), coarse=int(acc["coarse"]),
                      exchange=int(acc["exchange"]), predicted=predicted,
                      guard_rejected=int(acc["guard_rejected"]), round_cap=int(acc["round_cap"]),
                      exchange_cap=int(acc["exchange_cap"]), content_rounds=int(acc["content_rounds"]),
                      stale_evaluated=int(acc["stale_evaluated"]), guard_shared_tests=int(acc["guard_shared_tests"]),
                      guard_traced_blocks=int(acc["guard_traced_blocks"]), guard_relaxed=int(acc["guard_relaxed"]),
                      guard_shared_switches=float(acc["guard_shared_switches"]), lex_items=int(acc["lex_items"]),
                      evaluated=int(acc["evaluated"]), false_positive=int(acc["false_positive"]),
                      exchange_tie=int(acc["exchange_tie"]), content_tie=int(acc["content_tie"]),
                      tie_wide=int(acc["tie_wide"]))
        if not accepted:
            return None, None, 0.0, record, fine
        scores = self.per_sample(Xo, budget)
        actual = float(scores.sum()) - F
        record.update(full_delta=actual)
        if abs(actual - predicted) > 1e-9 * max(1.0, abs(F)) + 1e-6 * accepted:
            raise RuntimeError(f"founder refinement {self.name}: beam escape gains {predicted} do not match "
                               f"the full rescoring {actual}")
        if not actual > TOL:
            # The fully rescored gain must clear the acceptance band too (as bridges).
            raise RuntimeError(f"founder refinement {self.name}: beam escape's full rescoring {actual} is not "
                               f"above the acceptance threshold {TOL}")
        return Xo, scores, actual, record, fine

    def long_repair(self, X, F, budget, convention):
        """Polish only: long one-founder moves at an ordinary fixed point.

        WindowBeam offers on LONG_WINDOW-block windows (stride LONG_STRIDE; the window's best path that differs from
        the incumbent per founder, offered when its surrogate gain exceeds the tie quantum). Each offer, and every
        variant of it whose rows on one LONG_CORRECTION-block window inside its changed span are replaced by one of
        LONG_DONORS screened founders' rows or the incumbent's (one private correction: a long surrogate optimum is
        often exact-F negative only through a short stretch), is a one-founder content candidate on the offer's
        window, scored exactly from the panel's messages (gains.candidate_gains). Per offer the best candidate
        above TOL (content order on ties) enters ONE acceptance pass as a coarse move, so the shared-junction guard
        applies; accepted gains are verified by full rescoring and must clear TOL. Candidates per offer
        O(window), each O(S L_w Q). Returns (panel or None, its per-sample scores, exact gain, record)."""
        founders, blocks = X.shape
        X = np.ascontiguousarray(X, np.int64)
        starts = sorted(set(range(0, max(1, blocks - LONG_WINDOW + 1), LONG_STRIDE)) | {max(0, blocks - LONG_WINDOW)})
        templates = [("fine", a, min(blocks, a + LONG_WINDOW)) for a in starts]
        offers, beam_stats = self.beam.generate(X, budget, convention, templates=templates, width=LONG_BEAM_WIDTH)
        record = dict(offers=len(offers), windows=int(beam_stats["windows"]), beam_cut=int(beam_stats["beam_cut"]),
                      beam_binding=int(beam_stats["beam_binding"]), beam_wall=float(beam_stats["wall"]),
                      candidates=0, repaired=0, accepted=0)
        cands = []
        for kind, a, c, k, rows in offers:
            base = np.asarray(rows, np.int64)
            changed = np.flatnonzero(base != X[k, a:c])
            if not len(changed):
                continue
            c0, c1 = a + int(changed[0]), a + int(changed[-1]) + 1
            agree = [(-int(np.count_nonzero(X[d, c0:c1] == base[c0 - a:c1 - a])), d) for d in range(founders) if d != k]
            donors = [k] + [d for _, d in sorted(agree)[:LONG_DONORS]]
            variants = {tuple(base.tolist()): base}
            for u in range(c0, max(c0 + 1, c1 - LONG_CORRECTION + 1), max(1, LONG_CORRECTION // 2)):
                v = min(c, u + LONG_CORRECTION)
                for d in donors:
                    r = base.copy()
                    r[u - a:v - a] = X[d, u:v]
                    if not np.array_equal(r, X[k, a:c]):
                        variants.setdefault(tuple(r.tolist()), r)
            cands.extend((int(a), int(c), int(k), r) for r in variants.values())
        record["candidates"] = len(cands)
        if not cands:
            return None, None, 0.0, record
        # Held while scoring: one full message set, the batched gathers and the candidate rows built above.
        nbytes = (self.message_bytes(founders, self.guarded) + min(BRIDGE_BATCH_BYTES, 24 * self.workspace.samples
                  * founders * (founders + 1) // 2 * len(cands)) + sum(r.nbytes + 160 for _, _, _, r in cands))
        with execution.reserve_memory(nbytes, label="founder_refinement long repair"):
            fwd, bwd, fcnt, bcnt, fscale, bscale = self._full_messages(X, self.guarded, budget)
            orient = self.forward
            costs = orient.costs(founders)
            threads = max(1, int(budget.threads()))
            with execution.numba_threads(threads):
                haps = orient.haps(X)
            gains = []
            chunk = max(1, BRIDGE_BATCH_BYTES // max(1, 24 * self.workspace.samples * fwd.shape[2]))
            for m0 in range(0, len(cands), chunk):
                part = cands[m0:m0 + chunk]
                fb = np.asarray([a for a, _, _, _ in part], np.int64)
                eb = np.asarray([c for _, c, _, _ in part], np.int64)
                focal = np.asarray([k for _, _, k, _ in part], np.int64)
                ro = np.zeros(len(part) + 1, np.int64)
                ro[1:] = np.cumsum(eb - fb)
                rows = np.concatenate([r for _, _, _, r in part])
                sums = np.zeros((len(part), self.workspace.samples))
                for m, (a, c, _, _) in enumerate(part):
                    for b in range(a, c):
                        sums[m] += bscale[b]
                # Each chunk re-reads the live share (a sample's gain is its own recursion).
                with execution.numba_threads(max(1, int(budget.threads()))):
                    per = gains_kernels.candidate_gains(
                        haps, orient.leaves, self.workspace.logs, orient.offsets, costs,
                        np.ascontiguousarray(fwd[fb]), np.ascontiguousarray(bwd[fb]), focal, fb, eb, rows, ro,
                        np.ascontiguousarray(bwd[eb]), sums)
                gains.extend(float(sum(per[m])) for m in range(len(part)))
            best = {}
            for (a, c, k, r), gain in zip(cands, gains):
                key = (-gain, tuple(r.tolist()))
                if gain > TOL and ((a, c, k) not in best or key < best[(a, c, k)][0]):
                    best[(a, c, k)] = (key, r)
            record["repaired"] = len(best)
            if not best:
                return None, None, 0.0, record
            props = {}
            for (a, c, k), (_, r) in sorted(best.items()):
                props.setdefault(a, []).append((MOVE_COARSE, c, k, r))
            Xo = X.copy()
            acc, _, _ = accept_pass(self.forward, Xo, bwd, bcnt, bscale, props, costs, self.guarded, budget,
                                    start_fwd=fwd, start_fcnt=fcnt, scale=abs(F), reverse=convention)
            del fwd, bwd, fcnt, bcnt, fscale, bscale
        accepted = int(acc["fine"] + acc["coarse"] + acc["exchange"])
        predicted = float(acc["fine_gain"] + acc["coarse_gain"] + acc["exchange_gain"])
        record.update(accepted=accepted, predicted=predicted, guard_rejected=int(acc["guard_rejected"]),
                      guard_shared_tests=int(acc["guard_shared_tests"]), round_cap=int(acc["round_cap"]),
                      exchange_cap=int(acc["exchange_cap"]), content_rounds=int(acc["content_rounds"]),
                      stale_evaluated=int(acc["stale_evaluated"]), guard_traced_blocks=int(acc["guard_traced_blocks"]),
                      guard_relaxed=int(acc["guard_relaxed"]), guard_shared_switches=float(acc["guard_shared_switches"]),
                      lex_items=int(acc["lex_items"]), evaluated=int(acc["evaluated"]),
                      false_positive=int(acc["false_positive"]), exchange_tie=int(acc["exchange_tie"]),
                      content_tie=int(acc["content_tie"]), tie_wide=int(acc["tie_wide"]))
        if not accepted:
            return None, None, 0.0, record
        scores = self.per_sample(Xo, budget)
        actual = float(scores.sum()) - F
        record.update(full_delta=actual)
        if abs(actual - predicted) > 1e-9 * max(1.0, abs(F)) + 1e-6 * accepted:
            raise RuntimeError(f"founder refinement {self.name}: long repair gains {predicted} do not match "
                               f"the full rescoring {actual}")
        if not actual > TOL:
            raise RuntimeError(f"founder refinement {self.name}: long repair's full rescoring {actual} is not "
                               f"above the acceptance threshold {TOL}")
        return Xo, scores, actual, record

    def message_bytes(self, founders, lex):
        """Bytes of one full message set of a panel of `founders` rows (messages.message_arrays)."""
        states = founders * (founders + 1) // 2
        samples = self.workspace.samples
        return 8 * samples * ((self.blocks + 1) * states * (4 if lex else 2) + 2 * self.blocks)

    def bridge(self, X, fine_props, budget, convention=None):
        """bridge_search under a reservation of its transients: a full message set,
        the precursor batches' three S x Q gathers and one window's local forward/backward chains."""
        if not fine_props:
            return None
        founders = len(X)
        states = founders * (founders + 1) // 2
        per = 24 * self.workspace.samples * states
        span = max(c - a for a, c, _, _ in fine_props) + 1
        nbytes = (self.message_bytes(founders, False) + min(BRIDGE_BATCH_BYTES, per * len(fine_props))
                  + 2 * span * 8 * self.workspace.samples * states)
        with execution.reserve_memory(nbytes, label="founder_refinement bridge"):
            return self.bridge_search(X, fine_props, budget, convention)

    def bridge_search(self, X, fine_props, budget, convention=None):
        """The best fine-window/suffix bridge of panel X (forward block order), or None.

        At an ordinary fixed point the fine proposals of the sweep (the surrogate's window
        maximisers, rejected by exact acceptance) are bridge precursors whatever the sign of their
        exact gain: per fine window the precursor of largest exact gain (content order on ties) is
        applied privately, and every founder-pair suffix exchange at every cut inside the window
        is scored exactly from the private panel's local messages (prefix and suffix outside the
        window unchanged). The offer is precursor gain + exchange gain; the largest above TOL is
        returned (ties: content order of (cut, pair, window, founder, rows)) as (panel, predicted
        exact gain, record). The caller verifies the gain by full rescoring. (The 2.8 design prototype's fine-bridge
        family on the 2.6 engine's proposals.)"""
        ws = self.workspace
        founders, blocks = X.shape
        X = np.ascontiguousarray(X, np.int64)
        fwd, bwd, _, _, _, bscale = self._full_messages(X, False, budget)
        orient = self.forward
        costs = orient.costs(founders)
        logs, offsets, leaves = ws.logs, orient.offsets, orient.leaves
        threads = max(1, int(budget.threads()))
        with execution.numba_threads(threads):
            haps = orient.haps(X)
        focal = np.asarray([k for _, _, k, _ in fine_props], np.int64)
        fb = np.asarray([a for a, _, _, _ in fine_props], np.int64)
        eb = np.asarray([c for _, c, _, _ in fine_props], np.int64)
        row_offsets = np.zeros(len(fine_props) + 1, np.int64)
        row_offsets[1:] = np.cumsum([c - a for a, c, _, _ in fine_props])
        rows = np.concatenate([np.asarray(r, np.int64) for _, _, _, r in fine_props])
        content = []
        # Bounded batches: three S x Q message copies per precursor.
        chunk = max(1, BRIDGE_BATCH_BYTES // max(1, 24 * ws.samples * fwd.shape[2]))
        for m0 in range(0, len(fine_props), chunk):
            m1 = min(len(fine_props), m0 + chunk)
            sums = np.zeros((m1 - m0, ws.samples))
            for m in range(m0, m1):
                a, c = fine_props[m][0], fine_props[m][1]
                for b in range(a, c):
                    sums[m - m0] += bscale[b]
            ro = row_offsets[m0:m1 + 1] - row_offsets[m0]
            with execution.numba_threads(max(1, int(budget.threads()))):     # the live share, per chunk
                per = gains_kernels.candidate_gains(
                    haps, leaves, logs, offsets, costs, np.ascontiguousarray(fwd[fb[m0:m1]]),
                    np.ascontiguousarray(bwd[fb[m0:m1]]), focal[m0:m1], fb[m0:m1], eb[m0:m1],
                    rows[row_offsets[m0]:row_offsets[m1]], ro, np.ascontiguousarray(bwd[eb[m0:m1]]), sums)
            content.extend(float(sum(per[m])) for m in range(m1 - m0))
        best = {}
        for m, (a, c, k, r) in enumerate(fine_props):
            key = (-content[m], int(k), tuple(int(x) for x in r))
            if (a, c) not in best or key < best[(a, c)][0]:
                best[(a, c)] = (key, m)
        pa, pb = gains_kernels.exchange_pairs(founders)
        ident = np.arange(fwd.shape[2], dtype=np.int64)
        chosen = None
        evaluated = 0
        for (a, c), (_, m) in sorted(best.items()):
            k = int(focal[m])
            r = rows[row_offsets[m]:row_offsets[m + 1]]
            Y = X.copy()
            Y[k, a:c] = r
            with execution.numba_threads(max(1, int(budget.threads()))):     # the live share, per window
                haps_y = orient.haps(Y)
                local_f = kernels.forward_boundaries(haps_y, logs, offsets, costs,
                                                     np.ascontiguousarray(fwd[a]), a, c)
                local_b, _ = kernels.backward_chain(haps_y, logs, offsets, costs,
                                                    np.ascontiguousarray(bwd[c]), ident, a, c)
                for cut in range(max(1, a), min(blocks - 1, c) + 1):
                    swap = gains_kernels.exchange_gains(np.ascontiguousarray(local_f[cut - a]),
                                                        np.ascontiguousarray(local_b[cut - a]))
                    evaluated += len(swap)
                    p = int(np.argmax(swap))
                    gain = content[m] + float(swap[p])
                    key = (-gain, cut, int(pa[p]), int(pb[p]), a, c, k, tuple(int(x) for x in r))
                    if gain > TOL and (chosen is None or key < chosen[0]):
                        chosen = (key, Y, cut, int(pa[p]), int(pb[p]), gain, m)
        if chosen is None:
            return None
        _, Y, cut, u, v, gain, m = chosen
        Y = Y.copy()
        Y[[u, v], cut:] = Y[[v, u], cut:]
        record = dict(window=[int(fb[m]), int(eb[m])], founder=int(focal[m]),
                      precursor_gain=float(content[m]), cut=int(cut), pair=[u, v],
                      predicted=float(gain), windows=len(best), exchanges_evaluated=int(evaluated))
        return Y, float(gain), record

    def _sweep_inputs(self, X, budget, cache, held=None, convention=None, lex=None):
        """The messages of panel X and its window proposals (pure functions of the panel and the convention)."""
        message_state, message_wall = self._messages(X, budget, held, lex)
        started = time.perf_counter()
        proposals, stats = self.proposals(X, message_state[0], message_state[1], budget, cache, convention)
        return dict(messages=message_state, proposals=proposals, stats=stats,
                    message_wall=message_wall, proposal_wall=time.perf_counter() - started)

    # ------------------------------------------------------------------ fixed-K fit
    def converge(self, start, label, budget, polish=False):
        """C(start): the better of the forward-first search (default content-order convention) and the
        reverse-first search (reversed convention) (module docstring). Both run concurrently, each with its share
        of the budget; the higher exact F is kept (tie classes of F; within a class the canonically smaller panel,
        flagged as tied_panels when the panels differ). Returns a Fit whose rows are in canonical founder order."""
        token = (label, tie_rules.panel_key(start))
        cached = self._fit_cache.get(token)
        if cached is not None:
            self.reused_fits += 1
            return self._choose_fit(token, cached)
        done = self._load(f"{label}.done")
        if done is not None and (not done["summary"].get("tied_panels") or "variants" in done):
            payloads = done.get("variants", (done,))
            variants = tuple(Fit(np.asarray(item["rows"], np.int64).copy(), FitSummary(**item["summary"]),
                                 tuple(WeakJunction(*entry) for entry in item["weak_junctions"]))
                             for item in payloads)
            self._fit_cache[token] = variants
            return self._choose_fit(token, variants)
        started = time.perf_counter()
        # Release 2.2: the two searches differ in their start direction and in their content-order convention
        # (forward-first: the default order, reverse-first: the reversed one), so the kept fit is the better of
        # both conventions rather than a product of one (equal F with different panels: flagged tied_panels).
        # Their sweep-0 proposals then differ, so the start panel's inputs are not shared (2.1 computed them once).
        shared = None
        # The initial exact objective is independent of direction/convention;
        # only proposals and directional messages need separate computation.
        initial_scores = self.per_sample(start, budget)
        if polish:
            # Polish fits run the forward-first search only, with the whole budget (its extra move families
            # dominate its cost; the count pass's fits keep the best of two). No reverse search runs: the
            # summary aggregates the executed search only and records F_rev as None.
            for _, result in budget.concurrent(
                    [lambda child: self._converge_direction(start, label, "fwd", child, shared, initial_scores,
                                                            polish)], self.fit_bytes(len(start), polish=True),
                    label="founder_refinement searches"):
                X_fwd, fwd = result
            X_rev, rev = None, None
        else:
            searches = [None, None]
            for index, result in budget.concurrent(
                    [lambda child, first=first: self._converge_direction(start, label, first, child,
                                                                         shared, initial_scores, polish)
                     for first in ("fwd", "rev")], self.fit_bytes(len(start)),
                    label="founder_refinement searches"):
                searches[index] = result
            (X_fwd, fwd), (X_rev, rev) = searches
        executed = (fwd,) if rev is None else (fwd, rev)
        chosen = "fwd"
        tied_panels = ()
        tied_scores = tied_switches = ()
        minimum_switches = None
        result = None
        if rev is not None:
            # Tie classes of the two exact F (quantum at |F|); within a class the canonically smaller
            # panel (sorted rows, compared as row tuples) is kept.
            values = [float(fwd["F"]), float(rev["F"])]
            contents = [tie_rules.panel_key(X_fwd), tie_rules.panel_key(X_rev)]
            scale = max(abs(v) for v in values)
            quantum = tie_rules.quantum("refine_F", scale)
            result = ties.tie_classes(values, quantum, content=contents, descending=True)
            tie_rules.add_decision(self.counter, "assembly.refine_converge", result, values, quantum,
                                   tie_rules.motion("refine_F", scale), descending=True)
            if self.census_prefix is not None:
                tie_rules.census("assembly.refine_converge", f"{self.census_prefix}/{label}", [-v for v in values])
            chosen = ("fwd", "rev")[result.winner]
        if result is not None and result.tied_winner and contents[0] != contents[1]:
            tied_panels = tuple(sorted(contents))
            counts = [self.minimum_switches(X, budget) for X in (X_fwd, X_rev)]
            minimum_switches = min(counts)
            eligible = [i for i in result.order if counts[i] == minimum_switches]
            chosen = ("fwd", "rev")[eligible[0]]
            by_panel = {content: (value, count)
                        for content, value, count in zip(contents, values, counts)}
            tied_scores = tuple(by_panel[panel][0] for panel in tied_panels)
            tied_switches = tuple(by_panel[panel][1] for panel in tied_panels)
            if self.counter is not None:
                self.counter.note("assembly.refine_converge.symmetric", inside_tie=True)
        X, kept = (X_rev, rev) if chosen == "rev" else (X_fwd, fwd)
        # Canonical founder order of the fit (weak-junction founder pairs follow the permutation).
        X, order = tie_rules.canonical_rows(np.asarray(X, np.int64))
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order), dtype=np.int64)
        scores = {first: self._fit_scores.pop(f"{label}.{first}", None) for first in ("fwd", "rev")}
        if scores[chosen] is not None:
            self._fit_scores[label] = scores[chosen]
            self._score_cache[tie_rules.panel_key(X)] = scores[chosen]
        summary = FitSummary(
            label=label, founders=int(kept["founders"]), F0=float(kept["F0"]), F=float(kept["F"]),
            converged=bool(kept["converged"]), sweeps=int(kept["sweeps"]), first=chosen,
            F_fwd=float(fwd["F"]), F_rev=None if rev is None else float(rev["F"]),
            max_frontier=int(kept["max_frontier"]),
            frontier_fallbacks=int(kept["frontier_fallbacks"]), accepted=dict(kept["accepted"]),
            max_deviation=float(kept["max_deviation"]),
            wall=time.perf_counter() - started,
            worker_cpu=float(sum(search["worker_cpu"] for search in executed)),
            thread_cpu=float(sum(search["thread_cpu"] for search in executed)), tied_panels=tied_panels,
            tied_scores=tied_scores, tied_switches=tied_switches,
            minimum_switches=minimum_switches,
            phase_seconds={name: sum(search.get("phase_seconds", {}).get(name, 0.0)
                                     for search in executed)
                           for name in ("messages", "refinement_windows", "acceptance",
                                        "guarded_acceptance")},
            work_counters={name: sum(search.get("work_counters", {}).get(name, 0)
                                     for search in executed)
                           for name in ("windows", "templates_solved", "templates_reused", "dp",
                                        "fixing_dps", "beam_truncations", "beam_fallbacks") + _WORK_KEYS})
        weak = tuple(WeakJunction(first, last, tuple(sorted((int(inverse[pair[0]]), int(inverse[pair[1]])))),
                                  gain) for first, last, pair, gain in kept["weak_junctions"])
        variants = [Fit(X, summary, weak)]
        if tied_panels:
            other_name = "fwd" if chosen == "rev" else "rev"
            other_X, other = (X_fwd, fwd) if other_name == "fwd" else (X_rev, rev)
            other_X, order = tie_rules.canonical_rows(np.asarray(other_X, np.int64))
            inverse = np.empty(len(order), np.int64)
            inverse[order] = np.arange(len(order), dtype=np.int64)
            other_weak = tuple(WeakJunction(first, last,
                tuple(sorted((int(inverse[pair[0]]), int(inverse[pair[1]])))), gain)
                for first, last, pair, gain in other["weak_junctions"])
            other_summary = replace(summary, F=float(other["F"]), first=other_name,
                converged=bool(other["converged"]), sweeps=int(other["sweeps"]),
                max_frontier=int(other["max_frontier"]), frontier_fallbacks=int(other["frontier_fallbacks"]),
                accepted=dict(other["accepted"]), max_deviation=float(other["max_deviation"]),
                minimum_switches=self.minimum_switches(other_X, budget))
            variants.append(Fit(other_X, other_summary, other_weak))
            if scores[other_name] is not None:
                self._score_cache[tie_rules.panel_key(other_X)] = scores[other_name]
        payloads = [dict(rows=fit.rows.astype(np.int32), summary=vars(fit.summary).copy(),
                         weak_junctions=[tuple(vars(item).values()) for item in fit.weak_junctions])
                    for fit in variants]
        self._save(f"{label}.done", dict(payloads[0], variants=payloads))
        self._fit_cache[token] = tuple(variants)
        return self._choose_fit(token, variants)

    def _converge_direction(self, start, label, first, budget, shared=None, initial_scores=None,
                            polish=False):
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
            current = self.per_sample(X, budget) if initial_scores is None else initial_scores
            F0 = float(current.sum())
        F, history, junctions, converged, sweep = F0, [], (), False, 0
        # The search's proposal cache (seeded by the shared sweep 0).
        cache = ProposalCache(None if shared is None else shared["cache"])
        # This search's own messages of the panel its last sweep started from, updated in place by
        # the next sweep: (panel, messages); a reversed pass keeps their forward half only, so the
        # search holds no more message arrays than its forward passes do.
        held = None
        # Lazy lexicographic sweeps: a guarded sweep runs its pass on plain messages
        # unless the previous sweep read switch costs; the plain pass stops at the first coarse
        # item above TOL and the sweep is then run guarded (same decisions, see accept_pass). A
        # search's first sweep starts lexicographic (it reads switch costs in ~95% of searches).
        lex_next = True
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
                inputs = self._sweep_inputs(X, budget, cache, held, convention=(first == "rev"),
                                            lex=self.guarded and lex_next)
            held = None
            lex = inputs["messages"][2] is not None
            fwd, bwd, fcnt, bcnt, fscale, bscale = inputs["messages"]
            proposals, stats = inputs["proposals"], inputs["stats"]
            message_wall, proposal_wall = inputs["message_wall"], inputs["proposal_wall"]
            del inputs
            reverse = (sweep % 2 == 1) if first == "fwd" else (sweep % 2 == 0)
            fine_props = [(a, c, k, rows) for kind, a, c, k, rows in proposals if kind == "fine"]
            props = {}
            for kind, a, c, k, rows in proposals:
                if reverse:
                    props.setdefault(blocks - c, []).append((_MOVE_CODES[kind], blocks - a, k,
                                                             rows[::-1].copy()))
                else:
                    props.setdefault(a, []).append((_MOVE_CODES[kind], c, k, rows))
            pass_started = time.perf_counter()
            lex_aborts = 0
            while True:
                with execution.numba_threads(max(1, int(budget.threads()))):
                    if reverse:
                        # The reversed pass stores the clamped forward messages; their per-block
                        # normalisers are the forward ones in reversed block order.
                        orientation = self._reversed
                        forward_costs = self.forward.costs(founders)
                        if lex:
                            stored, stored_cnt = kernels.clamp_reverse_lex(
                                fwd, fcnt, self.forward.logs, forward_costs, self.forward.offsets)
                        else:
                            stored, stored_cnt = kernels.clamp_reverse(
                                fwd, self.forward.logs, forward_costs, self.forward.offsets), None
                        stored_scale = np.ascontiguousarray(fscale[::-1])
                        Xo = np.ascontiguousarray(X[:, ::-1])
                        start_fwd = start_fcnt = None
                        kept = (fwd, None, fcnt, None, fscale, None)
                    else:
                        orientation, stored, stored_cnt, stored_scale = self.forward, bwd, bcnt, bscale
                        Xo = X.copy()
                        start_fwd, start_fcnt = fwd, fcnt
                        kept = (fwd, bwd, fcnt, bcnt, fscale, bscale)
                acc, gains, pairs = accept_pass(orientation, Xo, stored, stored_cnt, stored_scale, props,
                                                orientation.costs(founders), lex, budget,
                                                start_fwd=start_fwd, start_fcnt=start_fcnt, scale=abs(F),
                                                reverse=(first == "rev"), optimistic=self.guarded and not lex)
                if not acc["lex_abort"]:
                    break
                # The plain pass met a coarse candidate above TOL: the sweep's lexicographic messages.
                lex_aborts += 1
                del stored, stored_cnt, stored_scale, start_fwd, start_fcnt, kept, Xo
                del fwd, bwd, fcnt, bcnt, fscale, bscale
                message_state, extra_wall = self._messages(X, budget, None, lex=True)
                message_wall += extra_wall
                fwd, bwd, fcnt, bcnt, fscale, bscale = message_state
                del message_state
                lex = True
            del fwd, bwd, fcnt, bcnt, fscale, bscale
            lex_next = lex and acc["lex_items"] > 0
            acc["lex_abort"] = lex_aborts
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
            acc.update(bridge=0, bridge_gain=0.0, escape=0, escape_gain=0.0, escape_offers=0,
                       escape_windows=0, escape_beam_cut=0, escape_beam_binding=0, long=0, long_gain=0.0,
                       long_offers=0, long_windows=0, long_beam_cut=0, long_beam_binding=0, long_candidates=0)
            bridge_record = None
            if accepted == 0 and polish:
                # Polish: a beam escape pass at the ordinary fixed point (beam_escape); its fine
                # offers are the bridge precursors.
                escape_started = time.perf_counter()
                Y, scores, gain, escape_record, fine_props = self.beam_escape(X, F_new, budget, first == "rev")
                acc.update(escape_offers=escape_record["offers"], escape_windows=escape_record["windows"],
                           escape_beam_cut=escape_record["beam_cut"],
                           escape_beam_binding=escape_record["beam_binding"])
                # The escape pass's own cap and guard counters join the sweep's.
                for name in _POLISH_PASS_KEYS:
                    acc[name] = acc.get(name, 0) + escape_record.get(name, 0)
                if Y is not None:
                    X, current, F_new = Y, scores, float(scores.sum())
                    acc.update(escape=int(escape_record["accepted"]), escape_gain=gain)
                    accepted = int(escape_record["accepted"])
                    accounted += gain
                escape_wall = time.perf_counter() - escape_started
                pass_wall += escape_wall
                stats = dict(stats, escape_wall=escape_wall, escape=escape_record)
            if accepted == 0 and polish:
                # Polish: long one-founder moves with one private correction (long_repair).
                long_started = time.perf_counter()
                Y, scores, gain, long_record = self.long_repair(X, F_new, budget, first == "rev")
                acc.update(long_offers=long_record["offers"], long_windows=long_record["windows"],
                           long_beam_cut=long_record["beam_cut"], long_beam_binding=long_record["beam_binding"],
                           long_candidates=long_record["candidates"])
                for name in _POLISH_PASS_KEYS:
                    acc[name] = acc.get(name, 0) + long_record.get(name, 0)
                if Y is not None:
                    X, current, F_new = Y, scores, float(scores.sum())
                    acc.update(long=int(long_record["accepted"]), long_gain=gain)
                    accepted = int(long_record["accepted"])
                    accounted += gain
                long_wall = time.perf_counter() - long_started
                pass_wall += long_wall
                stats = dict(stats, long_wall=long_wall, long=long_record)
            if accepted == 0 and polish and fine_props:
                # Polish: fine-window/suffix bridges at an ordinary fixed point (bridge()).
                bridge_started = time.perf_counter()
                found = self.bridge(X, fine_props, budget, convention=(first == "rev"))
                if found is not None:
                    Y, predicted, bridge_record = found
                    scores = self.per_sample(Y, budget)
                    actual = float(scores.sum()) - F_new
                    bridge_record.update(full_delta=actual, delta_error=actual - predicted)
                    if abs(actual - predicted) > 1e-9 * max(1.0, abs(F_new)) + 1e-6:
                        raise RuntimeError(
                            f"founder refinement {self.name}.{label}:{sweep}: bridge gain "
                            f"{predicted} does not match the full rescoring {actual}")
                    if not actual > TOL:
                        # The fully rescored gain must clear the acceptance band too.
                        raise RuntimeError(
                            f"founder refinement {self.name}.{label}:{sweep}: bridge's full rescoring "
                            f"{actual} is not above the acceptance threshold {TOL}")
                    X, current, F_new = Y, scores, float(scores.sum())
                    acc.update(bridge=1, bridge_gain=actual)
                    accepted = 1
                    accounted += actual
                bridge_wall = time.perf_counter() - bridge_started
                pass_wall += bridge_wall
                stats = dict(stats, bridge_wall=bridge_wall, bridge=bridge_record)
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
        binding = sum(h.get("escape_beam_binding", 0) for h in history)
        if binding:
            warnings.warn(f"founder refinement {self.name}.{label}: {binding} beam-escape windows were cut at "
                          f"beam.BEAM_WIDTH (approximate surrogate search; exact acceptance)", RuntimeWarning)
        binding = sum(h.get("long_beam_binding", 0) for h in history)
        if binding:
            warnings.warn(f"founder refinement {self.name}.{label}: {binding} long-repair windows were cut at "
                          f"search.LONG_BEAM_WIDTH = {LONG_BEAM_WIDTH} (approximate surrogate search; exact "
                          "acceptance)", RuntimeWarning)
        capped = sum(h.get("round_cap", 0) + h.get("exchange_cap", 0) for h in history)
        if capped:
            warnings.warn(f"founder refinement {self.name}.{label}: the acceptance round caps "
                          f"(acceptance.CONTENT_FULL_ROUNDS, EXCHANGE_ROUNDS) changed the move order at {capped} "
                          "junctions (exact gains; same fixed-point test)", RuntimeWarning)
        beam_fallbacks = sum(h["proposals"].get("beam_fallbacks", 0) for h in history)
        truncations = sum(h["proposals"].get("beam_truncations", 0) for h in history)
        if beam_fallbacks or truncations:
            warnings.warn(f"founder refinement {self.name}.{label}: the window beam cap (proposals.BEAM_CAP) "
                          f"truncated {truncations} programme units ({beam_fallbacks} programmes kept the "
                          "incumbent): those proposals are approximate", RuntimeWarning)
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
                      history=history,
                      # Sum of directional elapsed times, not component wall time. Guarded
                      # acceptance includes scoring and bookkeeping; kernel timings are not
                      # injected into the hot Numba loop. Guard tests/work are separate below.
                      phase_seconds={name: sum(h[key] for h in history) for name, key in
                                     (("messages", "message_wall"),
                                      ("refinement_windows", "proposal_wall"),
                                      ("acceptance", "pass_wall"))},
                      work_counters={name: sum(h["proposals"].get(name, 0) for h in history)
                                     for name in ("windows", "templates_solved", "templates_reused",
                                                  "dp", "fixing_dps", "beam_truncations",
                                                  "beam_fallbacks")})
        record["phase_seconds"]["guarded_acceptance"] = (
            record["phase_seconds"]["acceptance"] if self.guarded else 0.0)
        for name in _WORK_KEYS:
            record["work_counters"][name] = sum(h.get(name, 0) for h in history)
        fresh = history[resumed:]
        phases = {name: sum(h[key] for h in fresh) for name, key in
                  (("messages", "message_wall"), ("refinement_windows", "proposal_wall"),
                   ("acceptance", "pass_wall"))}
        phases["guarded_acceptance"] = phases["acceptance"] if self.guarded else 0.0
        work = {name: sum(h["proposals"].get(name, 0) for h in fresh)
                for name in ("windows", "templates_solved", "templates_reused", "dp", "fixing_dps",
                             "beam_truncations", "beam_fallbacks")}
        work.update(searches=1, **{name: sum(h.get(name, 0) for h in fresh) for name in _WORK_KEYS})
        with self._profile_lock:
            for name, value in phases.items():
                self._phase_seconds[name] = self._phase_seconds.get(name, 0.0) + value
            for name, value in work.items():
                self._work_counters[name] = self._work_counters.get(name, 0) + value
        self._save(f"{label}.done", dict(rows=X.astype(np.int32), record=record))
        return X, record
