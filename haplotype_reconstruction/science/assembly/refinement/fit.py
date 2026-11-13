"""Fixed-K fit C(X) of one component's founder paths: Gauss-Seidel sweeps to a certified stop.

Owns fixed-K orchestration, fit selection and polish scheduling; never opens storage or reads clocks.
The two search conventions own separate messages/proposals; their initial score vector is shared.
Method: docs/methods/founder_refinement.md#fixed-k-search (RS11–RS18).

A sweep (i) computes the exact normalised forward/backward block messages of the panel (a search
updates its previous sweep's messages in place, recomputing per sample only the blocks the last
pass changed and those whose start message changed; over consecutive sample ranges); (ii) computes
the window proposals of the panel at the start of the sweep (proposals.py; a window whose inputs
are bitwise unchanged since the search last solved it reuses that solve): for every founder and
every fine window a binned-surrogate offer: a completed exact programme gives the canonical
maximiser; if a focal frontier exceeds 64 kept paths, seeded search gives a heuristic answer,
rescored by the ordinary surrogate, without an optimality certificate or a beam_truncations count.
Where an L1 context exists, every founder and every coarse window uses the same hybrid over L1
span tuples; (iii) runs one exact
acceptance pass (acceptance.py), its direction alternating between sweeps; (iv) recomputes the
exact F, which must equal the start F plus the accepted gains. A sweep that accepts nothing stops
the search at a fixed point of the offered moves under the gain threshold and coarse-move guard.
An improving offer can remain below the threshold or fail the guard; convergence does not certify
a local or global optimum of F.
The sweep limit only marks a fit non-converged, never silently.

Best of two: every fit runs this search twice from the same start, concurrently, once with its
first pass forward and once reversed (both then alternate; each convention computes its own
proposals and directional messages). Each is a deterministic local search; the fit keeps the higher exact
F, decided by tie classes of F (two results within tie_rules.quantum("refine_F", |F|) are tied). Distinct tied
panels are retained as a family; the representative has the fewest exact-optimal switches, then the content
convention. At a fixed start the fit therefore never ends measurably below the forward-first search.
The fit's panel is returned in canonical founder order (rows sorted lexicographically), so founder
indices of the count pass (candidates and tie-breaks) are content ranks.
"""

from __future__ import annotations

import copy
import hashlib
import threading
from contextlib import contextmanager
from dataclasses import replace
from functools import partial

import numpy as np

from haplotype_reconstruction.contracts.assembly import (
    Fit,
    FitAcceptanceTotals,
    FitSummary,
    FitWorkCounters,
    SweepHistory,
    WeakJunction,
)
from haplotype_reconstruction.core import ties
from haplotype_reconstruction.core.errors import InvariantError

from .. import orders as tie_rules
from . import checkpoints, costs
from . import gains as gains_kernels
from . import messages as kernels
from .acceptance import MOVE_COARSE, MOVE_FINE, TOL, accept_pass
from .beam import WindowBeam
from .cache import Orientation, ProposalCache
from .cache import WindowMemo as _WindowMemo
from .capabilities import ComponentWorkspaceView, RefinementExecution, RefinementRecords
from .ledger import FitLedger
from .objective import score_panel, score_switches_direct, selected_alleles
from .parameters import (
    BRIDGE_BATCH_BYTES,
    EXACT_GAIN_ABSOLUTE,
    EXACT_GAIN_RELATIVE,
    LONG_BEAM_WIDTH,
    LONG_CORRECTION,
    LONG_DONORS,
    LONG_STRIDE,
    LONG_WINDOW,
    PAIR_TABLE_BYTES,
    SWITCH_PENALTY,
    RefinementParameters,
)
from .polish_cache import GainBook, PolishStores, search_store_bytes
from .proposals import WindowSolver
from .reductions import panel_objective
from .workspace import sample_ranges, sample_threads


def _top_class(values) -> np.ndarray:
    """Indices of the values in the closed band [max - q, max], q the refine_F floor quantum.

    The polish chooses among candidates whose exact gains differ by less than q (a safety margin over their
    motion under 1e-6 relative genotype-likelihood noise, not a significance level) by content, not by the last
    bits of the gains. It is a band at the maximum, not the chained tie classes of core.ties:
    (0, 0.75q, 1.5q) keeps the last two."""
    values = np.asarray(values, dtype=np.float64)
    return np.flatnonzero(values >= values.max() - tie_rules.quantum("refine_F", 0.0))


# Acceptance counters of the polish's escape and long-repair passes merged into their sweep's counters.
_POLISH_PASS_KEYS = (
    "evaluated",
    "false_positive",
    "exchange_tie",
    "content_tie",
    "tie_wide",
    "round_cap",
    "exchange_cap",
    "content_rounds",
    "stale_evaluated",
    "guard_shared_tests",
    "guard_rejected",
    "guard_traced_blocks",
    "guard_relaxed",
    "guard_shared_switches",
    "lex_items",
)
_MOVE_CODES = {"fine": MOVE_FINE, "coarse": MOVE_COARSE}
_TOTAL_KEYS = (
    "fine",
    "coarse",
    "exchange",
    "bridge",
    "escape",
    "long",
    "fine_gain",
    "coarse_gain",
    "exchange_gain",
    "bridge_gain",
    "escape_gain",
    "long_gain",
    "long_offers",
    "escape_offers",
    "escape_windows",
    "escape_beam_cut",
    "escape_beam_binding",
    "long_windows",
    "long_beam_cut",
    "long_beam_binding",
    "long_candidates",
    "evaluated",
    "false_positive",
    "guard_rejected",
    "guard_shared_tests",
    "guard_relaxed",
    "guard_shared_switches",
    "guard_traced_blocks",
    "changed_blocks",
    "content_rounds",
    "round_cap",
    "stale_evaluated",
    "exchange_cap",
    "lex_items",
    "lex_abort",
)
# Acceptance, escape and long-repair counters reported with every fit's work counters.
_WORK_KEYS = (
    "guard_shared_tests",
    "guard_traced_blocks",
    "content_rounds",
    "round_cap",
    "stale_evaluated",
    "exchange_cap",
    "escape_windows",
    "escape_beam_cut",
    "escape_beam_binding",
    "long_windows",
    "long_beam_cut",
    "long_beam_binding",
    "long_candidates",
    "lex_items",
    "lex_abort",
)


def _weak_junctions(gains, pairs, reverse, blocks):
    """Junctions whose best suffix exchange costs less than one switch penalty, consecutive ones
    with the same founder pair merged (exact against the final panel in the certifying sweep)."""
    records = []
    for p in range(1, blocks):
        if not gains[p] > -SWITCH_PENALTY:
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


def _fit_summary(payload):
    """Restore declared fit counters without altering sparse outer diagnostics."""
    values = dict(payload)
    for name, kind in (("accepted", FitAcceptanceTotals), ("work_counters", FitWorkCounters)):
        if not isinstance(values[name], kind):
            values[name] = kind(**values[name])
    return FitSummary(**values)


__all__ = ["SweepEngine", "PAIR_TABLE_BYTES"]


class SweepEngine:
    """The fixed inputs of every converged fit of one component (workspace, window solver,
    both orientations, fit records)."""

    def __init__(
        self,
        workspace: ComponentWorkspaceView,
        context,
        config: RefinementParameters,
        records: RefinementRecords | None = None,
        name="component",
        counter=None,
        census_prefix=None,
        source_start=0,
        *,
        reverse=False,
        census=None,
    ):
        self.reverse = bool(reverse)
        self.workspace = workspace
        # Tie-class decisions of this component's fits (tie_rules.SharedCounter or None) and its census prefix.
        self.counter = counter
        self.census_prefix = census_prefix
        self.census = census
        self.config = config
        self.windows = WindowSolver(
            workspace.emissions,
            context,
            workspace.penalty,
            work_coefficient=config.window_work_coefficient,
            source_start=source_start,
        )
        self.penalty = float(workspace.penalty)
        self.blocks = workspace.blocks
        # The window-beam proposer on the map-aware binned surrogate for the polish
        # fits (converge(polish=True)); count-pass fits use the window programmes only.
        self.beam = None
        if getattr(config, "polish", False):
            self.beam = WindowBeam(
                workspace, context, workspace.genetic_map, workspace.reset_rate, workspace.map_probability_floor
            )
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
        self.ledger = FitLedger()
        self.reused_fits = 0
        self.branch_key = None
        self._window_memo = None
        self._window_memo_statistics: dict[str, int] = {}
        # Shallow branch copies share measured work; cached fits contribute no
        # new work. Updates occur only once per directional search, under a lock.
        self._profile_lock = threading.Lock()
        self._work_counters: dict[str, int] = {}
        self._fit_order = [0]
        self._ordinal_lock = threading.Lock()

    def bind_fits(self, count, requests=None):
        """Bind logical driver-order fit ordinals before any concurrent dispatch."""
        with self._ordinal_lock:
            start = self._fit_order[0]
            self._fit_order[0] += int(count)
            ordinals = tuple(range(start, start + int(count)))
        if requests is not None:
            if len(requests) != count:
                raise ValueError("fit binding count differs from requests")
            for ordinal, (label, panel, kind) in zip(ordinals, requests, strict=False):
                self.windows.work.bind_fit(ordinal, label, panel, kind, self.branch_key)
        return ordinals

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
        # Dimension-bounded, deterministic optional memo. The stage owns reservation.
        memo = _WindowMemo()
        self._window_memo = memo
        try:
            yield memo
        finally:
            self._window_memo = None
            self._window_memo_statistics = dict(memo.statistics)
            memo.clear()

    def branch(self, choices, input_key=None):
        """A sequential count continuation sharing the evidence and completed fits.

        choices maps (fit label, canonical input panel) to a tied output panel.
        Replaying from the original input reconstructs the same bootstrap labels
        and best-by-K state before the chosen decision.
        """
        branch = copy.copy(self)
        branch.ledger = self.ledger.branch(choices)
        branch.reused_fits = 0
        branch.branch_key = hashlib.sha256(repr((input_key, tuple(sorted(choices.items())))).encode()).hexdigest()[:16]
        branch.records = (
            None if self._base_records is None else self._base_records.scoped(f"branch/{branch.branch_key}", nthreads=1)
        )
        branch.census_prefix = (
            None if self.census_prefix is None else f"{self.census_prefix}/branch/{branch.branch_key}"
        )
        return branch

    def _choose_fit(self, token, variants):
        wanted = self.ledger.forced_panels.get(token)
        fit = variants[0]
        if wanted is not None:
            fit = next(item for item in variants if tie_rules.panel_key(item.rows) == wanted)
            if fit is not variants[0]:
                self.ledger.fit_scores.pop(token[0], None)
        self.ledger.invocations[token[0]] = token[1]
        self.ledger.chosen_panels[token] = tie_rules.panel_key(fit.rows)
        return fit

    def _load(self, label):
        return None if self.records is None else self.records.load(label)

    def _save(self, label, payload):
        if self.records is not None:
            self.records.save(label, payload)

    # ------------------------------------------------------------------ scores
    def window(self, orientation, founders, budget):
        """Read a window, filling lazy numerical tables on the live sample team."""
        with budget.scope(sample_threads(self.workspace.samples, budget)):
            return orientation.window(founders)

    def normalisers(self, founders, budget):
        """Read the cached panel-independent terms, computing them on the live sample team."""
        with budget.scope(sample_threads(self.workspace.samples, budget)):
            return self.workspace.normalisers(founders)

    def per_sample(self, panel, budget):
        """Per-sample exact scores F_s of the panel, over consecutive sample ranges at the budget's
        live share re-read before each range."""
        # The immutable component fixes scores of a canonical panel, including
        # founder multiplicity. Fits and previous misses share this exact vector.
        key = tie_rules.panel_key(panel)
        cached = self.ledger.scores.get(key)
        if cached is not None:
            return cached
        ws = self.workspace
        founders = len(panel)
        with budget.scope(sample_threads(ws.samples, budget)):
            alleles = selected_alleles(ws.leaves, ws.offsets, np.ascontiguousarray(panel, np.int64))
        window = self.window(self.forward, founders, budget)
        with budget.scope(sample_threads(ws.samples, budget)):
            scores = np.asarray(
                score_panel(alleles, window, sample_ranges(ws.samples, budget), PAIR_TABLE_BYTES, budget)
            )
        # The panel-independent terms of K (start and stay): scores of different K compare.
        scores = scores + self.normalisers(founders, budget)
        self.ledger.remember_score(key, scores)
        return scores

    def minimum_switches(self, panel, budget):
        """Fewest exact-optimal diplotype switches, summed over the common evidence."""
        key = tie_rules.panel_key(panel)
        found = self.ledger.minimum_switches.get(key)
        if found is not None:
            return found
        ws = self.workspace
        with budget.scope(sample_threads(self.workspace.samples, budget)):
            alleles = selected_alleles(ws.leaves, ws.offsets, np.asarray(key, np.int64))
        total = 0
        with budget.scope(sample_threads(ws.samples, budget)):
            win = ws.windows.view(len(alleles))
        for s0, s1, threads in sample_ranges(ws.samples, budget):
            with budget.scope(threads):
                _, counts = score_switches_direct(alleles, win, s0, s1)
            total += int(counts.sum())
        self.ledger.minimum_switches[key] = total
        return total

    def fit_scores(self, label, panel, budget):
        """Per-sample scores of the converged fit `label` (panel `panel`): the scores its last sweep
        computed, or computed now (a fit loaded from its record)."""
        key = tie_rules.panel_key(panel)
        scores = self.ledger.fit_scores.pop(label, None)
        if scores is None:
            scores = self.ledger.scores.get(key)
        if scores is None:
            scores = self.per_sample(panel, budget)
        self.ledger.publish_score(key, scores)
        return scores

    def fit_bytes(self, founders, polish=False):
        """Declared message/acceptance working set of one search; cost only."""
        size = costs.fit_bytes(self.workspace, founders, self.guarded, self.cached_boundaries, polish)
        if polish:
            size += search_store_bytes(founders, self.blocks, self.beam.templates)
        return size

    # ------------------------------------------------------------------ messages and proposals
    def _full_messages(self, X, lex, budget):
        """messages(_lex) of panel X: (fwd, bwd, fcnt, bcnt, fscale, bscale), counts None without
        lex; over consecutive sample ranges at the budget's live share."""
        samples, offsets = self.workspace.samples, self.forward.offsets
        win = self.window(self.forward, len(X), budget)
        with budget.scope(sample_threads(self.workspace.samples, budget)):
            haps = self.forward.haps(X)
        step = self.workspace.message_step or checkpoints.stride(len(X), self.blocks, samples, self.guarded)
        arrays = checkpoints.allocate(len(X), self.blocks, samples, lex, haps, self.forward, step)
        fwd, bwd, fcnt, bcnt, fscale, bscale = arrays
        changed = np.ones(self.blocks, np.bool_)
        count3 = np.zeros((1, 1, 1))
        for s0, s1, n in sample_ranges(samples, budget):
            with budget.scope(n):
                if fwd.dense:
                    if lex:
                        kernels.messages_lex(haps, win, offsets, s0, s1, fwd.data, bwd.data, fcnt, bcnt, fscale, bscale)
                    else:
                        kernels.messages(haps, win, offsets, s0, s1, fwd.data, bwd.data, fscale, bscale)
                else:
                    for msg, cnt, scale in ((fwd, fcnt, fscale), (bwd, bcnt, bscale)):
                        with budget.scope(sample_threads(s1 - s0, budget)):
                            checkpoints.update(
                                haps,
                                win,
                                offsets,
                                msg.bounds,
                                changed,
                                msg.data,
                                cnt if lex else count3,
                                scale,
                                False,
                                msg.forward,
                                lex,
                                s0,
                                s1,
                            )
        return arrays

    def _messages(self, X, budget, held=None, lex=None):
        """The messages of panel X (with switch counts when lex, default: under the guard).

        held: (panel, messages) of an earlier panel, owned by the caller: updated in place to X's
        where X differs from that panel (update_messages(_lex), bitwise the full computation); a
        held backward half of None is computed whole. Held messages without switch
        counts are not updated to lexicographic ones (computed whole); held counts are dropped
        when lex is False (the scores of both recursions are the same)."""
        lex = self.guarded if lex is None else bool(lex)
        if held is not None and lex and held[1][2] is None:
            held = None
        if held is None:
            return self._full_messages(X, lex, budget)
        samples, offsets = self.workspace.samples, self.forward.offsets
        win = self.window(self.forward, len(X), budget)
        with budget.scope(sample_threads(self.workspace.samples, budget)):
            haps = self.forward.haps(X)
        panel, (fwd, bwd, fcnt, bcnt, fscale, bscale) = held
        if not lex:
            fcnt = bcnt = None
        known = bwd is not None
        if not known:
            bcnt = None if not lex else np.zeros_like(fcnt)
            bwd = checkpoints.BoundaryMessages(np.zeros_like(fwd.data), bcnt, fwd.bounds, haps, self.forward, False)
            bscale = np.zeros_like(fscale)
        for msg, cnt in ((fwd, fcnt), (bwd, bcnt)):
            msg.haps, msg.counts = haps, cnt
            msg.clear()
        changed = np.any(panel != X, axis=0)
        count3 = np.zeros((1, 1, 1))
        for s0, s1, n in sample_ranges(samples, budget):
            with budget.scope(n):
                if fwd.dense:
                    if lex:
                        kernels.update_messages_lex(
                            haps, win, offsets, changed, fwd.data, bwd.data, fcnt, bcnt, fscale, bscale, known, s0, s1
                        )
                    else:
                        kernels.update_messages(
                            haps, win, offsets, changed, fwd.data, bwd.data, fscale, bscale, known, s0, s1
                        )
                else:
                    for msg, cnt, scale, valid in ((fwd, fcnt, fscale, True), (bwd, bcnt, bscale, known)):
                        with budget.scope(sample_threads(s1 - s0, budget)):
                            checkpoints.update(
                                haps,
                                win,
                                offsets,
                                msg.bounds,
                                changed,
                                msg.data,
                                cnt if lex else count3,
                                scale,
                                valid,
                                msg.forward,
                                lex,
                                s0,
                                s1,
                            )
        return fwd, bwd, fcnt, bcnt, fscale, bscale

    def proposals(self, X, fwd, bwd, budget, cache=None, convention=None, prefix=None):
        """Every window template's proposals for panel X with messages fwd, bwd.

        cache (a search's ProposalCache, updated in place): a window problem is a pure function of
        the panel's rows in the window and the messages at its ends, and normalised max-plus
        messages of panels that differ only further away are often bitwise equal, so a template
        whose inputs equal the cached ones takes its cached result. The templates to solve run one
        Numba thread each, as many at once as the live share allows (budget.thread_map). With fewer templates
        than CPUs, they run as concurrent children of the share instead (budget.concurrent), each holding at most
        the CPUs its focal problems can use (WindowSolver.focal_chunk_cap), and a template's focal problems run in
        chunks on its child's map (WindowSolver._template; the results are those of one call)."""
        with budget.scope(sample_threads(self.workspace.samples, budget)):
            fwd = fwd.flanks(a for a, _ in self.windows.spans)
            bwd = bwd.flanks(c for _, c in self.windows.spans)

        memo = self._window_memo

        def run(index, team=None):
            key = None
            if memo is not None:
                a, c = self.windows.spans[index]
                key = memo.key(index, a, c, X, fwd, bwd, convention)
                result = memo.lookup(key)
                if result is not None:
                    self.windows.work.reused(ordinals[index], result)
                    return result
            result = self.windows.solve(index, X, fwd, bwd, convention, ordinal=ordinals[index], team=team)
            if memo is not None:
                memo.store(key, result)
            return result

        spans = self.windows.spans
        results = [None] * len(spans)
        todo = list(range(len(spans))) if cache is None else cache.lookup(spans, X, fwd, bwd, results)
        if prefix is None:
            prefix = (self.bind_fits(1)[0], 0, 0)
        ordinals = self.windows.work.bind(prefix, todo)
        windows = self.windows
        sizes = [
            windows.template_bytes(*windows.templates[index], self.workspace.samples, len(X))
            + (0 if memo is None else memo.transient_bytes(*spans[index], X, fwd, bwd))
            for index in todo
        ]
        # Largest templates first (their workspace estimate tracks their work), so the map's tail holds small
        # solves; each result is stored at its own index, so the order changes timing only.
        order = sorted(range(len(todo)), key=lambda i: (-sizes[i], i))
        caps = [windows.focal_chunk_cap(todo[i], self.workspace.samples, len(X)) for i in order]
        if max(caps, default=1) > 1 and int(budget.threads()) > len(todo):
            solved = [None] * len(order)
            for position, value in budget.concurrent(
                [partial(run, todo[i]) for i in order],
                [sizes[i] for i in order],
                label="founder_refinement window solves",
                caps=caps,
            ):
                solved[position] = value
        else:
            solved = budget.thread_map(
                run,
                [todo[i] for i in order],
                item_bytes=[sizes[i] for i in order],
                label="founder_refinement window solves",
            )
        computed = [None] * len(todo)
        for i, value in zip(order, solved, strict=False):
            computed[i] = value
        for index, value in zip(todo, computed, strict=False):
            results[index] = value
        if cache is not None:
            cache.store(spans, X, fwd, bwd, todo, computed)
        proposals = []
        stats = dict(
            windows=0,
            certified=0,
            dp=0,
            fixing_dps=0,
            max_frontier=0,
            fallbacks=0,
            changed=0,
            templates_solved=len(todo),
            templates_reused=len(spans) - len(todo),
            beam_truncations=0,
            beam_fallbacks=0,
        )
        for result in results:
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

    def beam_escape(self, X, F, budget, convention, stores=None):
        """Polish: one exact acceptance pass over the beam bank of panel X at an ordinary fixed point (forward
        orientation, the search's content convention). Returns (panel or None, its per-sample scores, exact
        gain, record, the bank's fine offers as bridge precursors).

        The pass runs on the panel's canonical founder order (rows sorted by content, ties by index) and its
        outcome is mapped back, so an escape depends on the panel only up to founder labels (the objective is
        label-invariant: max-plus messages; per-sample scores are exact under relabelling)."""
        founders = len(X)
        order = sorted(range(founders), key=lambda f: (tuple(int(x) for x in X[f]), f))
        canonical = np.ascontiguousarray(np.asarray(X, np.int64)[order])
        Yc, scores, gain, record, fine = self._beam_escape_canonical(canonical, F, budget, convention, stores)
        Y = None
        if Yc is not None:
            Y = np.empty_like(Yc)
            Y[order] = Yc
        fine = [(a, c, order[k], rows) for a, c, k, rows in fine]
        return Y, scores, gain, record, fine

    def _beam_escape_canonical(self, X, F, budget, convention, stores=None):
        """beam_escape on a panel in canonical founder order."""
        offers, beam_stats = self.beam.generate(X, budget, convention, stores=stores)
        fine = [(a, c, k, rows) for kind, a, c, k, rows in offers if kind == "fine"]
        record = dict(
            offers=len(offers),
            windows=int(beam_stats["windows"]),
            certified=int(beam_stats["certified"]),
            beam_cut=int(beam_stats["beam_cut"]),
            beam_binding=int(beam_stats["beam_binding"]),
            box_max_gap=float(beam_stats["box_max_gap"]),
            accepted=0,
        )
        if not offers:
            return None, None, 0.0, record, fine
        props = {}
        for kind, a, c, k, rows in offers:
            props.setdefault(a, []).append((_MOVE_CODES[kind], c, k, rows))
        Xo = np.ascontiguousarray(X, np.int64).copy()
        # This message set is held beside the search's own (fit_bytes).
        with budget.reserve(self.message_bytes(len(X), self.guarded), label="founder_refinement beam escape messages"):
            fwd, bwd, fcnt, bcnt, fscale, bscale = self._full_messages(X, self.guarded, budget)
            acc, _, _ = accept_pass(
                self.forward,
                Xo,
                bwd,
                bcnt,
                bscale,
                props,
                self.window(self.forward, len(X), budget),
                self.guarded,
                budget,
                start_fwd=fwd,
                start_fcnt=fcnt,
                scale=abs(F),
                reverse=convention,
            )
            del fwd, bwd, fcnt, bcnt, fscale, bscale
        accepted = int(acc["fine"] + acc["coarse"] + acc["exchange"])
        predicted = float(acc["fine_gain"] + acc["coarse_gain"] + acc["exchange_gain"])
        record.update(
            accepted=accepted,
            fine=int(acc["fine"]),
            coarse=int(acc["coarse"]),
            exchange=int(acc["exchange"]),
            predicted=predicted,
            guard_rejected=int(acc["guard_rejected"]),
            round_cap=int(acc["round_cap"]),
            exchange_cap=int(acc["exchange_cap"]),
            content_rounds=int(acc["content_rounds"]),
            stale_evaluated=int(acc["stale_evaluated"]),
            guard_shared_tests=int(acc["guard_shared_tests"]),
            guard_traced_blocks=int(acc["guard_traced_blocks"]),
            guard_relaxed=int(acc["guard_relaxed"]),
            guard_shared_switches=float(acc["guard_shared_switches"]),
            lex_items=int(acc["lex_items"]),
            evaluated=int(acc["evaluated"]),
            false_positive=int(acc["false_positive"]),
            exchange_tie=int(acc["exchange_tie"]),
            content_tie=int(acc["content_tie"]),
            tie_wide=int(acc["tie_wide"]),
        )
        if not accepted:
            return None, None, 0.0, record, fine
        scores = self.per_sample(Xo, budget)
        actual = panel_objective(scores) - F
        record.update(full_delta=actual)
        if abs(actual - predicted) > EXACT_GAIN_RELATIVE * max(1.0, abs(F)) + EXACT_GAIN_ABSOLUTE * accepted:
            raise InvariantError(
                f"founder refinement {self.name}: beam escape gains {predicted} do not match "
                f"the full rescoring {actual}"
            )
        if not actual > TOL:
            # The fully rescored gain must clear the acceptance band too (as bridges).
            raise InvariantError(
                f"founder refinement {self.name}: beam escape's full rescoring {actual} is not "
                f"above the acceptance threshold {TOL}"
            )
        return Xo, scores, actual, record, fine

    def long_repair(self, X, F, budget, convention, stores=None):
        """Polish only: long one-founder moves at an ordinary fixed point.

        WindowBeam offers on LONG_WINDOW-block windows (stride LONG_STRIDE; the window's best path that differs from
        the incumbent per founder, offered when its surrogate gain exceeds the tie quantum). Each offer, and every
        variant of it whose rows on one LONG_CORRECTION-block window inside its changed span are replaced by one of
        LONG_DONORS screened founders' rows or the incumbent's (one private correction: a long surrogate optimum is
        often exact-F negative only through a short stretch), is a one-founder content candidate on the offer's
        window, scored exactly from the panel's messages (gains.candidate_gains). Per offer the best candidate
        above TOL (content order among those within the refine_F floor quantum of the best) enters ONE
        acceptance pass as a coarse move, so the shared-junction guard
        applies; accepted gains are verified by full rescoring and must clear TOL. Candidates per offer
        O(window), each O(S L_w Q). Returns (panel or None, its per-sample scores, exact gain, record)."""
        founders, blocks = X.shape
        X = np.ascontiguousarray(X, np.int64)
        starts = sorted(set(range(0, max(1, blocks - LONG_WINDOW + 1), LONG_STRIDE)) | {max(0, blocks - LONG_WINDOW)})
        templates = [("fine", a, min(blocks, a + LONG_WINDOW)) for a in starts]
        offers, beam_stats = self.beam.generate(
            X, budget, convention, templates=templates, width=LONG_BEAM_WIDTH, stores=stores
        )
        record = dict(
            offers=len(offers),
            windows=int(beam_stats["windows"]),
            beam_cut=int(beam_stats["beam_cut"]),
            beam_binding=int(beam_stats["beam_binding"]),
            candidates=0,
            repaired=0,
            accepted=0,
        )
        # Regenerate in current labels: three equally agreeing donors can change
        # which two survive index ties. Only same-candidate scalar gains are reused.
        cands = []
        for _kind, a, c, k, rows in offers:
            base = np.asarray(rows, np.int64)
            changed = np.flatnonzero(base != X[k, a:c])
            if not len(changed):
                continue
            c0, c1 = a + int(changed[0]), a + int(changed[-1]) + 1
            agree = [
                (-int(np.count_nonzero(X[d, c0:c1] == base[c0 - a : c1 - a])), d) for d in range(founders) if d != k
            ]
            donors = [k] + [d for _, d in sorted(agree)[:LONG_DONORS]]
            variants = {tuple(base.tolist()): base}
            for u in range(c0, max(c0 + 1, c1 - LONG_CORRECTION + 1), max(1, LONG_CORRECTION // 2)):
                v = min(c, u + LONG_CORRECTION)
                for d in donors:
                    r = base.copy()
                    r[u - a : v - a] = X[d, u:v]
                    if not np.array_equal(r, X[k, a:c]):
                        variants.setdefault(tuple(r.tolist()), r)
            cands.extend((int(a), int(c), int(k), r) for r in variants.values())
        record["candidates"] = len(cands)
        if not cands:
            return None, None, 0.0, record
        # Held while scoring: one full message set, the batched gathers and the candidate rows built above.
        nbytes = (
            self.message_bytes(founders, self.guarded)
            + min(BRIDGE_BATCH_BYTES, 24 * self.workspace.samples * founders * (founders + 1) // 2 * len(cands))
            + sum(r.nbytes + 160 for _, _, _, r in cands)
        )
        with budget.reserve(nbytes, label="founder_refinement long repair"):
            fwd, bwd, fcnt, bcnt, fscale, bscale = self._full_messages(X, self.guarded, budget)
            orient = self.forward
            win = self.window(orient, founders, budget)
            threads = max(1, int(budget.threads()))
            with budget.scope(threads):
                haps = orient.haps(X)
            book = None if stores is None else GainBook(stores, X, fwd, bwd)
            gains, todo = ([None] * len(cands), list(range(len(cands)))) if book is None else book.lookup(cands)
            chunk = max(1, BRIDGE_BATCH_BYTES // max(1, 24 * self.workspace.samples * fwd.shape[2]))
            for m0 in range(0, len(todo), chunk):
                indices = todo[m0 : m0 + chunk]
                part = [cands[m] for m in indices]
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
                with budget.scope(sample_threads(self.workspace.samples, budget)):
                    per = gains_kernels.candidate_gains(
                        haps,
                        orient.leaves,
                        win,
                        orient.offsets,
                        np.ascontiguousarray(fwd[fb]),
                        np.ascontiguousarray(bwd[fb]),
                        focal,
                        fb,
                        eb,
                        rows,
                        ro,
                        np.ascontiguousarray(bwd[eb]),
                        sums,
                    )
                for i, m in enumerate(indices):
                    gains[m] = float(sum(per[i]))
                    if book is not None:
                        book.remember(cands[m], gains[m])
            groups = {}
            for (a, c, k, r), gain in zip(cands, gains, strict=False):
                if gain > TOL:
                    groups.setdefault((a, c, k), []).append((gain, r))
            best = {}
            for unit, members in groups.items():
                # The variants within the quantum of the best gain are tied: content (the rows) decides.
                tied = _top_class([gain for gain, _ in members])
                r = min((members[i][1] for i in tied), key=lambda rows: tuple(rows.tolist()))
                best[unit] = (None, r)
            record["repaired"] = len(best)
            if not best:
                return None, None, 0.0, record
            props = {}
            for (a, c, k), (_, r) in sorted(best.items()):
                props.setdefault(a, []).append((MOVE_COARSE, c, k, r))
            Xo = X.copy()
            acc, _, _ = accept_pass(
                self.forward,
                Xo,
                bwd,
                bcnt,
                bscale,
                props,
                win,
                self.guarded,
                budget,
                start_fwd=fwd,
                start_fcnt=fcnt,
                scale=abs(F),
                reverse=convention,
            )
            del book, fwd, bwd, fcnt, bcnt, fscale, bscale
        accepted = int(acc["fine"] + acc["coarse"] + acc["exchange"])
        predicted = float(acc["fine_gain"] + acc["coarse_gain"] + acc["exchange_gain"])
        record.update(
            accepted=accepted,
            predicted=predicted,
            guard_rejected=int(acc["guard_rejected"]),
            guard_shared_tests=int(acc["guard_shared_tests"]),
            round_cap=int(acc["round_cap"]),
            exchange_cap=int(acc["exchange_cap"]),
            content_rounds=int(acc["content_rounds"]),
            stale_evaluated=int(acc["stale_evaluated"]),
            guard_traced_blocks=int(acc["guard_traced_blocks"]),
            guard_relaxed=int(acc["guard_relaxed"]),
            guard_shared_switches=float(acc["guard_shared_switches"]),
            lex_items=int(acc["lex_items"]),
            evaluated=int(acc["evaluated"]),
            false_positive=int(acc["false_positive"]),
            exchange_tie=int(acc["exchange_tie"]),
            content_tie=int(acc["content_tie"]),
            tie_wide=int(acc["tie_wide"]),
        )
        if not accepted:
            return None, None, 0.0, record
        scores = self.per_sample(Xo, budget)
        actual = panel_objective(scores) - F
        record.update(full_delta=actual)
        if abs(actual - predicted) > EXACT_GAIN_RELATIVE * max(1.0, abs(F)) + EXACT_GAIN_ABSOLUTE * accepted:
            raise InvariantError(
                f"founder refinement {self.name}: long repair gains {predicted} do not match "
                f"the full rescoring {actual}"
            )
        if not actual > TOL:
            raise InvariantError(
                f"founder refinement {self.name}: long repair's full rescoring {actual} is not "
                f"above the acceptance threshold {TOL}"
            )
        return Xo, scores, actual, record

    def message_bytes(self, founders, lex):
        """Bytes of the selected dense or checkpointed message set, including normalisers."""
        states = founders * (founders + 1) // 2
        samples = self.workspace.samples
        return (
            8
            * samples
            * (
                len(
                    checkpoints.boundaries(
                        self.blocks, checkpoints.stride(founders, self.blocks, samples, self.guarded)
                    )
                )
                * states
                * (4 if lex else 2)
                + 2 * self.blocks
            )
        )

    def bridge(self, X, fine_props, budget, convention=None, stores=None):
        """bridge_search under a reservation of its transients: a full message set,
        the precursor batches' three S x Q gathers and one window's local forward/backward chains."""
        if not fine_props:
            return None
        founders = len(X)
        states = founders * (founders + 1) // 2
        per = 24 * self.workspace.samples * states
        span = max(c - a for a, c, _, _ in fine_props) + 1
        nbytes = (
            self.message_bytes(founders, False)
            + min(BRIDGE_BATCH_BYTES, per * len(fine_props))
            + 2 * span * 8 * self.workspace.samples * states
        )
        with budget.reserve(nbytes, label="founder_refinement bridge"):
            return self.bridge_search(X, fine_props, budget, convention, stores)

    def bridge_search(self, X, fine_props, budget, convention=None, stores=None):
        """The best fine-window/suffix bridge of panel X (forward block order), or None.

        At an ordinary fixed point the fine proposals of the sweep (the surrogate's window
        maximisers, rejected by exact acceptance) are bridge precursors whatever the sign of their
        exact gain: per fine window the precursor of largest exact gain (content order inside the refine_F floor quantum
        of it) is applied privately, and every founder-pair suffix exchange at every cut inside the window
        is scored exactly from the private panel's local messages (prefix and suffix outside the
        window unchanged). The offer is precursor gain + exchange gain; the largest above TOL is
        returned (ties within the quantum: content order of (cut, pair, window, founder, rows)) as (panel, predicted
        exact gain, record). The caller verifies the gain by full rescoring."""
        ws = self.workspace
        founders, blocks = X.shape
        X = np.ascontiguousarray(X, np.int64)
        fwd, bwd, _, _, _, bscale = self._full_messages(X, False, budget)
        orient = self.forward
        win = self.window(orient, founders, budget)
        offsets, leaves = orient.offsets, orient.leaves
        threads = max(1, int(budget.threads()))
        with budget.scope(threads):
            haps = orient.haps(X)
        focal = np.asarray([k for _, _, k, _ in fine_props], np.int64)
        fb = np.asarray([a for a, _, _, _ in fine_props], np.int64)
        eb = np.asarray([c for _, c, _, _ in fine_props], np.int64)
        row_offsets = np.zeros(len(fine_props) + 1, np.int64)
        row_offsets[1:] = np.cumsum([c - a for a, c, _, _ in fine_props])
        rows = np.concatenate([np.asarray(r, np.int64) for _, _, _, r in fine_props])
        book = None if stores is None else GainBook(stores, X, fwd, bwd)
        content, todo = (
            ([None] * len(fine_props), list(range(len(fine_props)))) if book is None else book.lookup(fine_props)
        )
        # Bounded batches: three S x Q message copies per missing precursor.
        chunk = max(1, BRIDGE_BATCH_BYTES // max(1, 24 * ws.samples * fwd.shape[2]))
        for m0 in range(0, len(todo), chunk):
            part = todo[m0 : m0 + chunk]
            sums = np.zeros((len(part), ws.samples))
            for i, m in enumerate(part):
                a, c = fine_props[m][0], fine_props[m][1]
                for b in range(a, c):
                    sums[i] += bscale[b]
            ro = np.zeros(len(part) + 1, np.int64)
            ro[1:] = np.cumsum([row_offsets[m + 1] - row_offsets[m] for m in part])
            part_rows = np.concatenate([rows[row_offsets[m] : row_offsets[m + 1]] for m in part])
            index = np.asarray(part, np.int64)
            with budget.scope(sample_threads(self.workspace.samples, budget)):
                per = gains_kernels.candidate_gains(
                    haps,
                    leaves,
                    win,
                    offsets,
                    np.ascontiguousarray(fwd[fb[index]]),
                    np.ascontiguousarray(bwd[fb[index]]),
                    np.ascontiguousarray(focal[index]),
                    np.ascontiguousarray(fb[index]),
                    np.ascontiguousarray(eb[index]),
                    part_rows,
                    ro,
                    np.ascontiguousarray(bwd[eb[index]]),
                    sums,
                )
            for i, m in enumerate(part):
                content[m] = float(sum(per[i]))
                if book is not None:
                    book.remember(fine_props[m], content[m])
        windows = {}
        for m, (a, c, _k, _r) in enumerate(fine_props):
            windows.setdefault((a, c), []).append(m)
        best = {}
        for (a, c), members in windows.items():
            # Precursors within the quantum of the largest exact gain are tied: content decides.
            tied = [members[i] for i in _top_class([content[m] for m in members])]
            m = min(tied, key=lambda m: (int(fine_props[m][2]), tuple(int(x) for x in fine_props[m][3])))
            best[(a, c)] = (None, m)
        pa, pb = gains_kernels.exchange_pairs(founders)
        ident = np.arange(fwd.shape[2], dtype=np.int64)
        offers = []
        evaluated = 0

        def evaluate(item, child):
            (a, c), (_, m), left, right = item
            k = int(focal[m])
            r = rows[row_offsets[m] : row_offsets[m + 1]]
            Y = X.copy()
            Y[k, a:c] = r
            found = []
            count = 0
            # A streaming window is private to this child. Resident log tables
            # are immutable and shared, as in independent fixed-K refits.
            with ws.windows.lease(founders):
                local_win = self.window(orient, founders, child)
                team = sample_threads(ws.samples, child)
                with child.scope(team):
                    haps_y = orient.haps(Y)
                    local_f = kernels.forward_boundaries(haps_y, local_win, offsets, left, a, c)
                    local_b, _ = kernels.backward_chain(haps_y, local_win, offsets, right, ident, a, c)
                for cut in range(max(1, a), min(blocks - 1, c) + 1):
                    team = sample_threads(ws.samples, child)
                    parallel = gains_kernels.parallel_exchanges(ws.samples, founders, team)
                    with child.scope(team):
                        swap = gains_kernels.exchange_gains(
                            np.ascontiguousarray(local_f[cut - a]), np.ascontiguousarray(local_b[cut - a]), parallel
                        )
                    count += len(swap)
                    # Pair and offer selection still use the original content convention.
                    p = int(_top_class(swap)[0])
                    gain = content[m] + float(swap[p])
                    if gain > TOL:
                        found.append(
                            (
                                (cut, int(pa[p]), int(pb[p]), a, c, k, tuple(int(x) for x in r)),
                                cut,
                                int(pa[p]),
                                int(pb[p]),
                                gain,
                                m,
                            )
                        )
            return found, count

        # Windows have no accepted edits in common: every trial starts from X.
        # Bound their aggregate scratch, and copy checkpoint flanks on the parent
        # thread before dispatch (MessageSet's reconstruction cache is mutable).
        pending = []
        cached = {}
        keys = {}
        for unit, choice in sorted(best.items()):
            a, c = unit
            _, m = choice
            found = None
            if book is not None:
                r = rows[row_offsets[m] : row_offsets[m + 1]]
                key = book.window_key(a, c, focal[m], r, content[m])
                keys[unit] = key
                found = stores.lookup(stores.bridge, unit, key)
            if found is None:
                pending.append((unit, choice))
            else:
                # Only the local trial is retained. Current proposal index and
                # global offer ordering are reconstructed on every invocation.
                trials, count = found
                cached[unit] = (
                    [
                        ((cut, u, v, a, c, int(focal[m]), tuple(int(x) for x in fine_props[m][3])), cut, u, v, gain, m)
                        for cut, u, v, gain in trials
                    ],
                    count,
                )
        per_window = [
            2 * founders * int(ws.offsets[-1])
            + 16 * (c - a + 2) * ws.samples * fwd.shape[2]
            + ws.windows.window_bytes()
            for (a, c), _ in pending
        ]
        first = 0
        while first < len(pending):
            stop, total = first, 0
            while stop < len(pending) and (stop == first or total + per_window[stop] <= BRIDGE_BATCH_BYTES):
                total += per_window[stop]
                stop += 1
            wave = [
                (unit, choice, np.ascontiguousarray(fwd[unit[0]]), np.ascontiguousarray(bwd[unit[1]]))
                for unit, choice in pending[first:stop]
            ]
            values = [None] * len(wave)
            for index, result in budget.concurrent(
                [lambda child, item=item: evaluate(item, child) for item in wave],
                per_window[first:stop],
                label="founder_refinement bridge windows",
            ):
                values[index] = result
            for (unit, _), (found, count) in zip(pending[first:stop], values, strict=True):
                cached[unit] = found, count
                if book is not None:
                    stores.bridge[unit] = keys[unit], (tuple(entry[1:5] for entry in found), count)
            first = stop
        for unit in sorted(best):
            found, count = cached[unit]
            offers.extend(found)
            evaluated += count
        if not offers:
            return None
        # Offers within the quantum of the largest gain are tied: content order of (cut, pair, window, founder, rows).
        tied = [offers[i] for i in _top_class([offer[4] for offer in offers])]
        _, cut, u, v, gain, m = min(tied, key=lambda offer: offer[0])
        Y = X.copy()
        Y[int(focal[m]), int(fb[m]) : int(eb[m])] = rows[row_offsets[m] : row_offsets[m + 1]]
        Y = Y.copy()
        Y[[u, v], cut:] = Y[[v, u], cut:]
        record = dict(
            window=[int(fb[m]), int(eb[m])],
            founder=int(focal[m]),
            precursor_gain=float(content[m]),
            cut=int(cut),
            pair=[u, v],
            predicted=float(gain),
            windows=len(best),
            exchanges_evaluated=int(evaluated),
        )
        return Y, float(gain), record

    def _sweep_inputs(self, X, budget, cache, held=None, convention=None, lex=None, prefix=None):
        """The messages of panel X and its window proposals (pure functions of the panel and the convention)."""
        message_state = self._messages(X, budget, held, lex)
        proposals, stats = self.proposals(X, message_state[0], message_state[1], budget, cache, convention, prefix)
        return dict(
            messages=message_state,
            proposals=proposals,
            stats=stats,
        )

    # ------------------------------------------------------------------ fixed-K fit
    def converge(self, start, label: str, budget: RefinementExecution, polish: bool = False, *, ordinal=None) -> Fit:
        """Run one pre-bound fit scope and seal it only after every direction joins."""
        ordinal = self.bind_fits(1, [(label, start, "polish" if polish else "fit")])[0] if ordinal is None else ordinal
        self.windows.work.begin_fit(ordinal)
        try:
            return self._converge(start, label, budget, polish, ordinal)
        finally:
            self.windows.work.end_fit(ordinal)

    def _converge(self, start, label, budget, polish, ordinal):
        """C(start): the better of the forward-first search (default content-order convention) and the
        reverse-first search (reversed convention) (module docstring). Both run concurrently, each with its share
        of the budget; the higher exact F is kept (tie classes of F; within a class the canonically smaller panel,
        flagged as tied_panels when the panels differ). Returns a Fit whose rows are in canonical founder order."""
        token = (label, tie_rules.panel_key(start))
        cached = self.ledger.fits.get(token)
        if cached is not None:
            self.reused_fits += 1
            return self._choose_fit(token, cached)
        done = self._load(f"{label}.done")
        if done is not None and (not done["summary"].get("tied_panels") or "variants" in done):
            # The fit's own tie-class decisions count once per computed fit, resumed or not.
            if self.counter is not None:
                for kind, decisions, inside_tie, wide, boundary in done["tie_counts"]:
                    self.counter.add_counts(kind, decisions, inside_tie, wide, boundary)
            payloads = done.get("variants", (done,))
            variants = tuple(
                Fit(
                    np.asarray(item["rows"], np.int64).copy(),
                    _fit_summary(item["summary"]),
                    tuple(WeakJunction(*entry) for entry in item["weak_junctions"]),
                )
                for item in payloads
            )
            self.ledger.fits[token] = variants
            return self._choose_fit(token, variants)
        # The two searches differ in their start direction and in their content-order convention
        # (forward-first: the default order, reverse-first: the reversed one), so the kept fit is the better of
        # both conventions rather than a product of one (equal F with different panels: flagged tied_panels).
        # Their sweep-0 proposals differ; only the initial exact objective, independent of direction and
        # convention, is shared.
        initial_scores = self.per_sample(start, budget)
        if polish:
            # Polish fits run the forward-first search only, with the whole budget (its extra move families
            # dominate its cost; the count pass's fits keep the best of two). No reverse search runs: the
            # summary aggregates the executed search only and records F_rev as None.
            for _, result in budget.concurrent(
                [lambda child: self._converge_direction(start, label, "fwd", child, initial_scores, polish, ordinal)],
                self.fit_bytes(len(start), polish=True),
                label="founder_refinement searches",
            ):
                X_fwd, fwd = result
            X_rev, rev = None, None
        else:
            searches = [None, None]
            for index, result in budget.concurrent(
                [
                    lambda child, first=first: self._converge_direction(
                        start, label, first, child, initial_scores, polish, ordinal
                    )
                    for first in ("fwd", "rev")
                ],
                self.fit_bytes(len(start)),
                label="founder_refinement searches",
            ):
                searches[index] = result
            (X_fwd, fwd), (X_rev, rev) = searches
        executed = (fwd,) if rev is None else (fwd, rev)
        # This fit's own tie-class decisions: added to the component's counter and kept in its done record, so
        # that a component resumed from the record reports the same tie counts.
        fit_ties = ties.TieCounter()
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
            result = ties.fixed_policy("assembly.refine_converge", quantum, reverse=self.reverse).classes(
                values, content=contents, descending=True
            )
            tie_rules.add_decision(
                fit_ties,
                "assembly.refine_converge",
                result,
                values,
                quantum,
                tie_rules.motion("refine_F", scale),
                descending=True,
            )
            if self.census_prefix is not None:
                tie_rules.census(
                    "assembly.refine_converge", f"{self.census_prefix}/{label}", [-v for v in values], sink=self.census
                )
            chosen = ("fwd", "rev")[result.winner]
        if result is not None and result.tied_winner and contents[0] != contents[1]:
            tied_panels = tuple(sorted(contents))
            counts = [self.minimum_switches(X, budget) for X in (X_fwd, X_rev)]
            minimum_switches = min(counts)
            eligible = [i for i in result.order if counts[i] == minimum_switches]
            chosen = ("fwd", "rev")[eligible[0]]
            by_panel = {
                content: (value, count) for content, value, count in zip(contents, values, counts, strict=False)
            }
            tied_scores = tuple(by_panel[panel][0] for panel in tied_panels)
            tied_switches = tuple(by_panel[panel][1] for panel in tied_panels)
            fit_ties.note("assembly.refine_converge.symmetric", inside_tie=True)
        if self.counter is not None:
            self.counter.merge(fit_ties)
        X, kept = (X_rev, rev) if chosen == "rev" else (X_fwd, fwd)
        # Canonical founder order of the fit (weak-junction founder pairs follow the permutation).
        X, order = tie_rules.canonical_rows(np.asarray(X, np.int64))
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order), dtype=np.int64)
        scores = {first: self.ledger.fit_scores.pop(f"{label}.{first}", None) for first in ("fwd", "rev")}
        if scores[chosen] is not None:
            self.ledger.fit_scores[label] = scores[chosen]
            self.ledger.publish_score(tie_rules.panel_key(X), scores[chosen])
        summary = FitSummary(
            label=label,
            founders=int(kept["founders"]),
            F0=float(kept["F0"]),
            F=float(kept["F"]),
            converged=bool(kept["converged"]),
            sweeps=int(kept["sweeps"]),
            first=chosen,
            F_fwd=float(fwd["F"]),
            F_rev=None if rev is None else float(rev["F"]),
            max_frontier=int(kept["max_frontier"]),
            frontier_fallbacks=int(kept["frontier_fallbacks"]),
            accepted=FitAcceptanceTotals(**kept["accepted"]),
            max_deviation=float(kept["max_deviation"]),
            tied_panels=tied_panels,
            tied_scores=tied_scores,
            tied_switches=tied_switches,
            minimum_switches=minimum_switches,
            work_counters=FitWorkCounters(
                **{
                    name: sum(search.get("work_counters", {}).get(name, 0) for search in executed)
                    for name in (
                        "windows",
                        "templates_solved",
                        "templates_reused",
                        "dp",
                        "fixing_dps",
                        "beam_truncations",
                        "beam_fallbacks",
                    )
                    + _WORK_KEYS
                }
            ),
        )
        weak = tuple(
            WeakJunction(first, last, tuple(sorted((int(inverse[pair[0]]), int(inverse[pair[1]])))), gain)
            for first, last, pair, gain in kept["weak_junctions"]
        )
        variants = [Fit(X, summary, weak)]
        if tied_panels:
            other_name = "fwd" if chosen == "rev" else "rev"
            other_X, other = (X_fwd, fwd) if other_name == "fwd" else (X_rev, rev)
            other_X, order = tie_rules.canonical_rows(np.asarray(other_X, np.int64))
            inverse = np.empty(len(order), np.int64)
            inverse[order] = np.arange(len(order), dtype=np.int64)
            other_weak = tuple(
                WeakJunction(first, last, tuple(sorted((int(inverse[pair[0]]), int(inverse[pair[1]])))), gain)
                for first, last, pair, gain in other["weak_junctions"]
            )
            other_summary = replace(
                summary,
                F=float(other["F"]),
                first=other_name,
                converged=bool(other["converged"]),
                sweeps=int(other["sweeps"]),
                max_frontier=int(other["max_frontier"]),
                frontier_fallbacks=int(other["frontier_fallbacks"]),
                accepted=FitAcceptanceTotals(**other["accepted"]),
                max_deviation=float(other["max_deviation"]),
                minimum_switches=self.minimum_switches(other_X, budget),
            )
            variants.append(Fit(other_X, other_summary, other_weak))
            if scores[other_name] is not None:
                self.ledger.publish_score(tie_rules.panel_key(other_X), scores[other_name])
        payloads = [
            dict(
                rows=fit.rows.astype(np.int32),
                summary=vars(fit.summary).copy(),
                weak_junctions=[tuple(vars(item).values()) for item in fit.weak_junctions],
            )
            for fit in variants
        ]
        tie_counts = [
            [entry.kind, entry.decisions, entry.inside_tie, entry.wide, entry.boundary]
            for entry in fit_ties.record().entries
        ]
        self._save(f"{label}.done", dict(payloads[0], variants=payloads, tie_counts=tie_counts))
        self.ledger.fits[token] = tuple(variants)
        return self._choose_fit(token, variants)

    def _converge_direction(self, start, label, first, budget, initial_scores=None, polish=False, ordinal=None):
        """_search under one window lease of the panel's K: every kernel of the search reads the same
        streaming window (stages.assembly.storage.Windows.lease; nothing for a resident component)."""
        with self.workspace.windows.lease(len(start)):
            return self._search(start, label, first, budget, initial_scores, polish, ordinal)

    def _search(self, start, label, first, budget, initial_scores=None, polish=False, ordinal=None):
        """One search from start: sweeps until one accepts nothing (or the reported sweep limit);
        the first pass runs forward (first "fwd") or reversed ("rev"), later passes alternate.
        Records <label>.<first>.sweep<n> (that sweep's state and history entry) and .done.
        Initial exact scores may be shared; direction-dependent proposals never are.
        Returns (rows int64 K x B, search record dict)."""
        label = f"{label}.{first}"
        done = self._load(f"{label}.done")
        if done is not None:
            return np.asarray(done["rows"], np.int64).copy(), done["record"]
        X = np.ascontiguousarray(start, np.int64).copy()
        founders, blocks = X.shape
        # Per-sample scores of the current panel X, while known: a pass that accepts nothing leaves
        # the panel, hence its scores, unchanged.
        current = self.per_sample(X, budget) if initial_scores is None else initial_scores
        F0 = panel_objective(current)
        F, history, junctions, converged, sweep = F0, [], (), False, 0
        # Each direction owns its proposal cache: the two conventions' proposals differ from sweep 0.
        cache = ProposalCache()
        # Only this polish search owns these bounded answers. A resumed search
        # starts empty; no store enters the ledger or a scientific record.
        stores = PolishStores() if polish else None
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
            history.append(SweepHistory(**saved["entry"]) if isinstance(saved["entry"], dict) else saved["entry"])
            junctions = tuple(WeakJunction(*item) for item in saved["junctions"])
            current = None
            sweep += 1
            resumed += 1
            if saved["entry"]["accepted"] == 0:
                converged = True
                break
        while not converged and sweep < limit:
            inputs = self._sweep_inputs(
                X,
                budget,
                cache,
                held,
                convention=(first == "rev"),
                lex=self.guarded and lex_next,
                prefix=(ordinal, int(first == "rev"), sweep),
            )
            held = None
            lex = inputs["messages"][2] is not None
            fwd, bwd, fcnt, bcnt, fscale, bscale = inputs["messages"]
            proposals, stats = inputs["proposals"], inputs["stats"]
            del inputs
            reverse = (sweep % 2 == 1) if first == "fwd" else (sweep % 2 == 0)
            fine_props = [(a, c, k, rows) for kind, a, c, k, rows in proposals if kind == "fine"]
            props = {}
            for kind, a, c, k, rows in proposals:
                if reverse:
                    props.setdefault(blocks - c, []).append((_MOVE_CODES[kind], blocks - a, k, rows[::-1].copy()))
                else:
                    props.setdefault(a, []).append((_MOVE_CODES[kind], c, k, rows))
            lex_aborts = 0
            while True:
                with budget.scope(sample_threads(self.workspace.samples, budget)):
                    if reverse:
                        # The reversed pass stores the clamped forward messages; their per-block
                        # normalisers are the forward ones in reversed block order.
                        orientation = self._reversed
                        entering = kernels.boundary_costs(self.forward.window(founders), self.forward.offsets)
                        stored = fwd.reversed(entering)
                        stored_cnt = stored.counts
                        stored_scale = np.ascontiguousarray(fscale[::-1])
                        Xo = np.ascontiguousarray(X[:, ::-1])
                        start_fwd = start_fcnt = None
                        kept = (fwd, None, fcnt, None, fscale, None)
                        # A reversed pass reads the clamped forward messages, and its backward half is not
                        # kept for the next sweep: drop this search's references now, so it does not hold
                        # both message sets during the pass (memory only).
                        bwd = bcnt = bscale = None
                    else:
                        orientation, stored, stored_cnt, stored_scale = self.forward, bwd, bcnt, bscale
                        Xo = X.copy()
                        start_fwd, start_fcnt = fwd, fcnt
                        kept = (fwd, bwd, fcnt, bcnt, fscale, bscale)
                acc, gains, pairs = accept_pass(
                    orientation,
                    Xo,
                    stored,
                    stored_cnt,
                    stored_scale,
                    props,
                    self.window(orientation, founders, budget),
                    lex,
                    budget,
                    start_fwd=start_fwd,
                    start_fcnt=start_fcnt,
                    scale=abs(F),
                    reverse=(first == "rev"),
                    optimistic=self.guarded and not lex,
                )
                if not acc["lex_abort"]:
                    break
                # The plain pass met a coarse candidate above TOL: the sweep's lexicographic messages.
                lex_aborts += 1
                del stored, stored_cnt, stored_scale, start_fwd, start_fcnt, kept, Xo
                del fwd, bwd, fcnt, bcnt, fscale, bscale
                message_state = self._messages(X, budget, None, lex=True)
                fwd, bwd, fcnt, bcnt, fscale, bscale = message_state
                del message_state
                lex = True
            del fwd, bwd, fcnt, bcnt, fscale, bscale
            lex_next = lex and acc["lex_items"] > 0
            acc["lex_abort"] = lex_aborts
            del stored, stored_cnt, stored_scale, start_fwd, start_fcnt
            held = (X, kept)
            del kept
            X = np.ascontiguousarray(Xo[:, ::-1]) if reverse else Xo
            accepted = acc["fine"] + acc["coarse"] + acc["exchange"]
            if accepted or current is None:
                current = self.per_sample(X, budget)
            F_new = panel_objective(current)
            accounted = acc["fine_gain"] + acc["coarse_gain"] + acc["exchange_gain"]
            deviation = F_new - (F + accounted)
            # Exactness check of the accepted gains: a relative allowance in F's scale plus a rounding
            # allowance per accepted move (each accepted gain is its own floating-point sum).
            if abs(deviation) > EXACT_GAIN_RELATIVE * max(1.0, abs(F_new)) + EXACT_GAIN_ABSOLUTE * accepted:
                raise InvariantError(
                    f"founder refinement {self.name}.{label}:{sweep}: exact F changed by "
                    f"{F_new - F}, accepted local gains {accounted}"
                )
            acc.update(
                bridge=0,
                bridge_gain=0.0,
                escape=0,
                escape_gain=0.0,
                escape_offers=0,
                escape_windows=0,
                escape_beam_cut=0,
                escape_beam_binding=0,
                long=0,
                long_gain=0.0,
                long_offers=0,
                long_windows=0,
                long_beam_cut=0,
                long_beam_binding=0,
                long_candidates=0,
            )
            bridge_record = None
            if accepted == 0 and polish:
                # Polish: a beam escape pass at the ordinary fixed point (beam_escape); its fine
                # offers are the bridge precursors.
                Y, scores, gain, escape_record, fine_props = self.beam_escape(X, F_new, budget, first == "rev", stores)
                acc.update(
                    escape_offers=escape_record["offers"],
                    escape_windows=escape_record["windows"],
                    escape_beam_cut=escape_record["beam_cut"],
                    escape_beam_binding=escape_record["beam_binding"],
                )
                # The escape pass's own cap and guard counters join the sweep's.
                for name in _POLISH_PASS_KEYS:
                    acc[name] = acc.get(name, 0) + escape_record.get(name, 0)
                if Y is not None:
                    X, current, F_new = Y, scores, panel_objective(scores)
                    acc.update(escape=int(escape_record["accepted"]), escape_gain=gain)
                    accepted = int(escape_record["accepted"])
                    accounted += gain
                stats = dict(stats, escape=escape_record)
            if accepted == 0 and polish:
                # Polish: long one-founder moves with one private correction (long_repair).
                Y, scores, gain, long_record = self.long_repair(X, F_new, budget, first == "rev", stores)
                acc.update(
                    long_offers=long_record["offers"],
                    long_windows=long_record["windows"],
                    long_beam_cut=long_record["beam_cut"],
                    long_beam_binding=long_record["beam_binding"],
                    long_candidates=long_record["candidates"],
                )
                for name in _POLISH_PASS_KEYS:
                    acc[name] = acc.get(name, 0) + long_record.get(name, 0)
                if Y is not None:
                    X, current, F_new = Y, scores, panel_objective(scores)
                    acc.update(long=int(long_record["accepted"]), long_gain=gain)
                    accepted = int(long_record["accepted"])
                    accounted += gain
                stats = dict(stats, long=long_record)
            if accepted == 0 and polish and fine_props:
                # Polish: fine-window/suffix bridges at an ordinary fixed point (bridge()).
                found = self.bridge(X, fine_props, budget, convention=(first == "rev"), stores=stores)
                if found is not None:
                    Y, predicted, bridge_record = found
                    scores = self.per_sample(Y, budget)
                    actual = panel_objective(scores) - F_new
                    bridge_record.update(full_delta=actual, delta_error=actual - predicted)
                    if abs(actual - predicted) > EXACT_GAIN_RELATIVE * max(1.0, abs(F_new)) + EXACT_GAIN_ABSOLUTE:
                        raise InvariantError(
                            f"founder refinement {self.name}.{label}:{sweep}: bridge gain "
                            f"{predicted} does not match the full rescoring {actual}"
                        )
                    if not actual > TOL:
                        # The fully rescored gain must clear the acceptance band too.
                        raise InvariantError(
                            f"founder refinement {self.name}.{label}:{sweep}: bridge's full rescoring "
                            f"{actual} is not above the acceptance threshold {TOL}"
                        )
                    X, current, F_new = Y, scores, panel_objective(scores)
                    acc.update(bridge=1, bridge_gain=actual)
                    accepted = 1
                    accounted += actual
                stats = dict(stats, bridge=bridge_record)
            junctions = _weak_junctions(gains, pairs, reverse, blocks)
            entry = SweepHistory(
                sweep=sweep,
                reverse=bool(reverse),
                F=F_new,
                dF=F_new - F,
                dF_accounted=float(accounted),
                deviation=float(deviation),
                accepted=int(accepted),
                **acc,
                proposals=stats,
            )
            history.append(entry)
            F = F_new
            self._save(
                f"{label}.sweep{sweep}",
                dict(
                    rows=X.astype(np.int32),
                    F=F,
                    entry=entry,
                    junctions=[tuple(vars(item).values()) for item in junctions],
                ),
            )
            sweep += 1
            if accepted == 0:
                converged = True
        fallbacks = sum(h["proposals"]["fallbacks"] for h in history)
        if len(history) > resumed:
            self.ledger.fit_scores[label] = current
        record = dict(
            label=label,
            founders=int(founders),
            F0=float(F0),
            F=float(F),
            gain=float(F - F0),
            converged=bool(converged),
            sweeps=len(history),
            resumed_sweeps=resumed,
            sweep_limit=int(limit),
            first=first,
            accepted=_totals(history),
            max_frontier=max([h["proposals"]["max_frontier"] for h in history] or [0]),
            frontier_fallbacks=fallbacks,
            max_deviation=max([abs(h["deviation"]) for h in history] or [0.0]),
            weak_junctions=[tuple(vars(item).values()) for item in junctions],
            history=history,
            work_counters={
                name: sum(h["proposals"].get(name, 0) for h in history)
                for name in (
                    "windows",
                    "templates_solved",
                    "templates_reused",
                    "dp",
                    "fixing_dps",
                    "beam_truncations",
                    "beam_fallbacks",
                )
            },
        )
        for name in _WORK_KEYS:
            record["work_counters"][name] = sum(h.get(name, 0) for h in history)
        fresh = history[resumed:]
        work = {
            name: sum(h["proposals"].get(name, 0) for h in fresh)
            for name in (
                "windows",
                "templates_solved",
                "templates_reused",
                "dp",
                "fixing_dps",
                "beam_truncations",
                "beam_fallbacks",
            )
        }
        work.update(searches=1, **{name: sum(h.get(name, 0) for h in fresh) for name in _WORK_KEYS})
        with self._profile_lock:
            for name, value in work.items():
                self._work_counters[name] = self._work_counters.get(name, 0) + value
        self._save(f"{label}.done", dict(rows=X.astype(np.int32), record=record))
        return X, record
