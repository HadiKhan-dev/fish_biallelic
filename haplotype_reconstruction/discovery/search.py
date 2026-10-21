"""Adaptive reversible block search with whole tied-class decisions.

Every member of each K's leading NLL class is cavity-scored (memoized by panel key). The
best cavity class, then the smaller-K prior, chooses a complete alternative set. Byte order
only labels its representative. Each expansion freezes a complete unexpanded NLL class,
merges all peers' neighbours before reselecting, and tests budgets at class boundaries.
Computational omissions are recorded separately from complete fitted alternatives.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from contextlib import contextmanager
from contextvars import ContextVar
import math
import time

import numpy as np

from ..maths.ties import (NOISE_MOTION, WIDE_CLASS_QUANTA, BlockQuanta, TieCounter, block_scale, improves,
                          inverse_permutation, run_sample_order, tie_classes)
from .cavity import CavityWorkspace, score_solutions
from .fixed_k.fit import prepare_workspace
from .moves import (MaximumCuts, SoftBirths, data_births, growth_inputs, initial_solutions,
                    ordinary_moves, replacements, soft_births)
from .panel import FixedKSolution, deduplicate, known_content_key, restricted, solution_tie_mask, unknown_alleles, whole_class_cut
from .residual_births import ResidualInputs

# Relative K of the moves a search step generates (death, same-K, birth, bridge).
_MOVE_OFFSETS = (-1, 0, 1, 2)
# Fixed block-local buckets, not an event stream. Geometry routines enter
# diagnostic_phase locally; outside a timed block it is a no-op.
_PHASES = ("fitting", "landmarks", "clique_counting", "lifting_budgeting", "cavity", "idle_admission")
_DIAGNOSTICS = ContextVar("discovery_diagnostics", default=None)


class BlockDiagnostics:
    """Exclusive wall/process-CPU times in a single block worker.

    Fitting includes surrounding search/release orchestration not inside a more
    specific bucket. CPU includes the worker's numerical threads, not other
    processes. idle_admission is elapsed dispatch preparation/admission/transport
    to worker entry plus the initial thread claim. Waiting for an undispatched
    pool slot is excluded. Block wall time starts at dispatch preparation when
    run in the pool, or at direct in-memory entry otherwise.
    """

    def __init__(self, dispatched_at=None):
        self.wall = dict.fromkeys(_PHASES, 0.0)
        self.cpu = dict.fromkeys(_PHASES, 0.0)
        self.phase = "fitting"
        self.wall_at = self.wall_start = time.perf_counter()
        self.cpu_at = self.cpu_start = time.process_time()
        if dispatched_at is not None:
            self.wall_start = dispatched_at
            self.wall["idle_admission"] = self.wall_at - dispatched_at

    def switch(self, phase):
        wall, cpu = time.perf_counter(), time.process_time()
        self.wall[self.phase] += wall - self.wall_at
        self.cpu[self.phase] += cpu - self.cpu_at
        self.wall_at, self.cpu_at = wall, cpu
        previous, self.phase = self.phase, phase
        return previous

    def summary_fields(self):
        self.switch(self.phase)
        return dict(wall_seconds=self.wall_at - self.wall_start,
                    cpu_seconds=self.cpu_at - self.cpu_start,
                    phase_seconds={name: dict(wall=self.wall[name], cpu=self.cpu[name]) for name in _PHASES})


@contextmanager
def diagnostic_block(dispatched_at=None):
    """Use one clock per block; nested search/release calls share it."""
    current = _DIAGNOSTICS.get()
    if current is not None:
        yield current
        return
    current = BlockDiagnostics(dispatched_at)
    token = _DIAGNOSTICS.set(current)
    try:
        yield current
    finally:
        _DIAGNOSTICS.reset(token)


@contextmanager
def diagnostic_phase(name):
    """Charge a named discovery phase exclusively (nested phases subtract)."""
    current = _DIAGNOSTICS.get()
    if current is None:
        yield
        return
    previous = current.switch(name)
    try:
        yield
    finally:
        current.switch(previous)


def contextual_omissions(items, k, phase):
    """Copy geometry omission scopes; never mutate cached generator receipts."""
    return [dict(item, scope=f"{item['scope']}; generating K={k} phase={phase}")
            if "landmark" in item["cause"] or "clique" in item["cause"] else item for item in items]


def backstop_counts(summary):
    """Block indicators, separating ordinary truncation from omitted ties.

    Accept a decoded summary or its saved mapping. Stop reasons also cover old
    expansion records whose omission list was not yet persisted.
    """
    get = summary.get if isinstance(summary, dict) else lambda key, default=None: getattr(summary, key, default)
    causes = {item["cause"] for item in get("omissions", ())}
    expansion_ranked = "expansion_ranked_truncation" in causes
    expansion_tied = "expansion_tie_class_at_budget" in causes
    score_ranked = "exact_score_ranked_truncation" in causes
    score_tied = "exact_score_tie_class_at_budget" in causes
    return dict(expansion_backstop=int(get("stop_reason") == "expansion_backstop" or expansion_ranked or expansion_tied),
                exact_score_backstop=int(score_ranked or score_tied),
                expansion_ranked_truncation=int(expansion_ranked), expansion_tie_class_at_budget=int(expansion_tied),
                exact_score_ranked_truncation=int(score_ranked), exact_score_tie_class_at_budget=int(score_tied))


@dataclass(frozen=True)
class SearchSummary:
    """What one block's search did (diagnostics; nothing downstream decides on it).

    k_values: int16[R] represented K values, ascending; k_log_scores: float64[R] the cavity
        log score (leave-one-out predictive + telescoping K prior + unordered set code) of each
        K's completely scored leading NLL class. Selection uses cavity classes and smaller K.
    natural_k_ceiling: identifiability ceiling min(2 N_active, 2^L).
    expansions, data_starts, exact_scores, exact_score_cache_hits: work counts.
    residual_calls, residual_rows: soft-residual birth calls and the novel rows they emitted.
    limits: omitted tied-class/component causes and mean-field nonconvergence, in first-occurrence
        order. Ranked truncations are excluded.
    heuristic_gauge_cuts: some expanded panel had K > exact_cut_max_k, so its gauge rewires came
        from the degree-ordered local-search maximum cut (reported here, not as a limit).
    stop_reason: "generated_neighbourhood_local_optimum", "incumbent_neighbourhood_search_limited"
        or "expansion_backstop".
    certified: the selected panel's generated neighbourhood was completely generated and scored
        without improvement (a local certificate, not a global one).
    Release 2.1 (defaults keep 2.0 records decodable):
    tie_quantum: the block's ``nll`` tie quantum (selection and archive orders).
    escalated, conventions: historical metadata only; the alternative modes were removed in 2.5.
    ambiguous: the winning cavity class after the wide-class rule and smaller-K prior has alternatives.
    tied_alternatives: the number of other members of that complete fitted set.
    score_margin: the selected score minus the best other represented K's score (inf if none).
    tie_events: maths.ties.TieCounter record {kind: {decisions, inside_tie, wide, boundary}}.
    search_incomplete: a straddling tied class or whole candidate component was omitted.
    omissions: dictionaries with kind, cause, count, scope and count_kind (exact or lower_bound).
        kind is tie_class_at_budget or ranked_truncation. Ordinary lower-ranked truncations are
        recorded but do not set search_incomplete or limits.
    unresolved: no scored incumbent exists; release an empty panel only in this case.
    archive_max_k: largest K ever added to the archive, even if subsequently pruned.
    wall_seconds, cpu_seconds, phase_seconds: optional block timings (BlockDiagnostics).
    selected_mean_field_failures: held-out site solves whose starts all failed, summed over
        the complete winning set, including tied selected alternatives.
    other_mean_field_failures: the same count over uniquely scored cached panels outside
        that winning set (including panels no longer in the archive). Old records use None.
    """

    k_values: np.ndarray
    k_log_scores: np.ndarray
    natural_k_ceiling: int
    expansions: int
    data_starts: int
    exact_scores: int
    exact_score_cache_hits: int
    residual_calls: int
    residual_rows: int
    limits: tuple[str, ...]
    heuristic_gauge_cuts: bool
    stop_reason: str
    certified: bool
    tie_quantum: float = 0.0
    escalated: bool = False
    conventions: int = 1
    ambiguous: bool = False
    tied_alternatives: int = 0
    score_margin: float = math.inf
    tie_events: dict = field(default_factory=dict)
    search_incomplete: bool = False
    omissions: tuple = ()
    unresolved: bool = False
    # None means not measured in older records, not zero work/failures.
    archive_max_k: int | None = None
    wall_seconds: float | None = None
    cpu_seconds: float | None = None
    phase_seconds: dict = field(default_factory=dict)
    selected_mean_field_failures: int | None = None
    other_mean_field_failures: int | None = None


@dataclass(frozen=True)
class BlockSearch:
    """Selected fitted alternative set and summary; selected=None means no scored incumbent.

    Alternatives contain the other winning cavity-class members after the smaller-K prior.
    The representative is a label for this complete set, never a search decision.
    """

    selected: FixedKSolution | None
    summary: SearchSummary
    alternatives: tuple = ()


def natural_k_ceiling(n_samples: int, n_sites: int) -> int:
    """The largest identifiable K: 2 copies per sample with depth, at most 2^L rows."""
    sample_ceiling = 2 * int(n_samples)
    if n_sites < sample_ceiling.bit_length():
        return min(sample_ceiling, 1 << int(n_sites))
    return sample_ceiling


def _readonly(array):
    array.setflags(write=False)
    return array


def _sample_subset(solution: FixedKSolution, index: np.ndarray) -> FixedKSolution:
    """The same fit with its per-sample arrays taken at ``index`` (a permutation of the samples)."""
    return replace(restricted(solution, index), nll=solution.nll)



def search_block(likelihoods, observed, config, *, sample_ids=None) -> BlockSearch:
    """Search the panel of one block's discovery sites.

    likelihoods: float64[N, L, 3] normalised GLs; observed: bool[N, L] (at least one True).
    sample_ids: the samples' names (optional). With names the samples are searched in the run's
    canonical order (maths.ties.run_sample_order); without, in the given order, which the caller
    keeps canonical (the evidence stage's run order). The selected solution's per-sample arrays
    are in the input order either way. Unobserved cells are set to exactly 1/3 before any seed,
    fit or score.
    """
    observed = np.ascontiguousarray(observed, dtype=np.bool_)
    likelihood = np.array(likelihoods, dtype=np.float64, order="C", copy=True)
    likelihood[~observed] = 1.0 / 3.0
    order = None
    if sample_ids is not None:
        if len(sample_ids) != likelihood.shape[0]:
            raise ValueError("sample_ids must name every sample of the block")
        order = run_sample_order(sample_ids)
        if np.array_equal(order, np.arange(len(order))):
            order = None
    if order is not None:
        likelihood = np.ascontiguousarray(likelihood[order])
        observed = np.ascontiguousarray(observed[order])
    with diagnostic_block() as diagnostics:
        result = _search(likelihood, observed, config)
        result = replace(result, summary=replace(result.summary, **diagnostics.summary_fields()))
    if order is None or result.selected is None:
        return result
    inverse = inverse_permutation(order)
    return BlockSearch(selected=_sample_subset(result.selected, inverse), summary=result.summary,
                       alternatives=tuple(_sample_subset(panel, inverse) for panel in result.alternatives))


def select_scores(values, quantum: float, content):
    """(representative, TieClasses): cavity classes, wide-class rule, then smaller K.

    Content labels the representative; _Search.winning_scores retains every equally ranked panel.
    """
    classes = tie_classes([score.log_score for score in values], quantum,
                          content=[content(score.solution) for score in values],
                          secondary=[(score.k,) for score in values], descending=True,
                          motion=NOISE_MOTION["nll"])
    return values[classes.winner], classes


class _Archive:
    """Per K the retained panels in decision order: tie classes of NLL (kind ``nll``), the content
    key inside a class labels members. Each add
    receives a complete proposal round. The retained list combines previous members and the new
    arrivals before a whole-class cut at ``width``; a straddling hard-bound class is omitted whole."""

    def __init__(self, width: int, quanta: BlockQuanta, content, counter: TieCounter, observed=None):
        self.width = int(width)
        self.quantum = float(quanta.nll)
        self.content = content
        self.counter = counter
        self.observed = observed
        self.fitted: dict[bytes, float] = {}
        self.beams: dict[int, list] = {}
        self.beam_classes: dict[int, list] = {}
        self.omissions: list = []
        self.science: dict[bytes, bytes] = {}
        self.max_k = 0

    def known_key(self, solution) -> bytes:
        """The panel's science (rows with alleles no assigned carrier observes masked; release 2.2)."""
        key = self.science.get(solution.key)
        if key is None:
            if self.observed is None or solution.assignments is None:
                key = solution.key
            else:
                key = known_content_key(solution.haplotypes,
                                        unknown_alleles(solution.assignments, self.observed, solution.k,
                                                        tie_mask=solution_tie_mask(solution)))
            self.science[solution.key] = key
        return key

    def add(self, solutions) -> None:
        touched: dict[int, None] = {}
        for solution in sorted(solutions, key=lambda panel: (panel.key, panel.nll)):
            previous = self.fitted.get(solution.key)
            if previous is not None and not solution.nll < previous - self.quantum:
                continue
            self.fitted[solution.key] = solution.nll
            self.max_k = max(self.max_k, solution.k)
            panels = self.beams.setdefault(solution.k, [])
            if previous is not None:
                panels[:] = [entry for entry in panels if entry.key != solution.key]
            # Panels that differ only at alleles no assigned carrier observes are the same science: one archive
            # entry, the lower NLL beyond the quantum, else the canonical (smaller) key -- a label (release 2.2).
            science = self.known_key(solution)
            twin = next((entry for entry in panels if self.known_key(entry) == science), None)
            if twin is not None:
                if solution.nll < twin.nll - self.quantum or (not twin.nll < solution.nll - self.quantum
                                                              and solution.key < twin.key):
                    panels[:] = [entry for entry in panels if entry is not twin]
                else:
                    continue
            panels.append(solution)
            touched[solution.k] = None
        for k in touched:
            ordered, classes = self._ordered(self.beams[k])
            kept, _ = whole_class_cut(ordered, classes, self.width, self.omissions,
                                      cause="archive", scope=f"K={k} archive NLL classes")
            self.beams[k] = kept
            self.beam_classes[k] = classes[:len(kept)]

    def _ordered(self, panels):
        """(panels in decision order, their NLL tie-class ids)."""
        if len(panels) == 1:
            return panels, [0]
        classes = tie_classes([panel.nll for panel in panels], self.quantum,
                              content=[self.content(panel) for panel in panels],
                              motion=NOISE_MOTION["nll"])
        ordered = [panels[index] for index in classes.order]
        cut_inside = (len(ordered) > self.width
                      and classes.classes[classes.order[self.width - 1]]
                      == classes.classes[classes.order[self.width]])
        self.counter.add("discovery.archive", classes,
                         decided_inside=bool(classes.tied_winner or cut_inside))
        return ordered, [int(classes.classes[index]) for index in classes.order]

    def first_class(self, k: int) -> list:
        """The retained panels of K in the first NLL class, in decision order."""
        panels = list(self.beams.get(k, ()))
        if len(panels) <= 1:
            return panels
        classes = tie_classes([panel.nll for panel in panels], self.quantum,
                              content=[self.content(panel) for panel in panels])
        return [panels[int(index)] for index in classes.order if classes.classes[int(index)] == 0]

    def first_class_size(self, k: int) -> int:
        """Members of the first NLL class among the retained panels of K."""
        panels = self.beams.get(k, ())
        if len(panels) <= 1:
            return len(panels)
        classes = tie_classes([panel.nll for panel in panels], self.quantum,
                              content=[self.content(panel) for panel in panels])
        return int(np.count_nonzero(classes.classes == 0))


class _Search:
    """One search of a block in canonical sample order."""

    def __init__(self, likelihood, observed, config):
        search = config.search
        self.config = config
        self.search = search
        self.likelihood = likelihood
        self.observed = observed
        self.active = np.ascontiguousarray(observed.any(axis=1))
        self.ceiling = natural_k_ceiling(int(self.active.sum()), likelihood.shape[1])
        self.quanta = BlockQuanta.for_scale(
            block_scale(likelihood, observed, config.model.unknown_copy_penalty))
        self.counter = TieCounter()
        self.content = lambda solution: solution.key
        self.workspace = prepare_workspace(likelihood, observed, config.model, self.quanta)
        self.growth = growth_inputs(likelihood, observed, self.counter)
        self.residual_inputs = ResidualInputs(likelihood, observed)
        self.cuts = MaximumCuts(search.exact_cut_max_k, search.max_cut_ties)
        self.cavity_workspace = CavityWorkspace(
            likelihood if bool(self.active.all()) else likelihood[self.active], config.cavity)
        width = search.beam_width
        self.archive = _Archive(width, self.quanta, self.content, self.counter, observed)
        self.beam_width = int(search.beam_width)
        self.replacement_limit = int(search.max_replacement_children)
        self.representatives: dict[int, FixedKSolution] = {}
        self.score_cache: dict = {}
        self.scores: dict = {}
        self.class_scores: dict[int, tuple] = {}
        self.scored_class_keys: dict[int, tuple] = {}
        self.omissions: list = []
        self.omission_keys: set = set()
        self.soft_cache: dict = {}
        self.limits: list[str] = []
        self.exact_scores = 0
        self.cache_hits = 0
        self.mean_field_failed = False
        self.ordinary_expanded: set[bytes] = set()
        self.residual_expanded: set[bytes] = set()
        self.ordinary_complete: dict[bytes, bool] = {}
        self.residual_complete: dict[bytes, bool] = {}
        self.generated_offsets: dict[bytes, set] = {}
        self.no_improvement: set[bytes] = set()
        self.residual_calls = 0
        self.residual_rows = 0
        self.heuristic_cuts = False
        self.stop_reason = ""
        self.expansions = 0
        self.data_starts = 0

    # ---------------------------------------------------------------------------------- scoring
    def omit(self, cause, count, scope, count_kind="exact", token=None, kind="tie_class_at_budget"):
        """Record a computational omission, once per complete candidate group."""
        identity = (cause, scope, token)
        if count and identity not in self.omission_keys:
            self.omission_keys.add(identity)
            self.omissions.append(dict(kind=kind, cause=cause, count=int(count), scope=scope, count_kind=count_kind))
            if kind != "ranked_truncation":
                self.limits.append(cause)

    def refresh(self, preferred_k: int) -> bool:
        """Cavity-score complete leading NLL classes, nearest K first; cache by panel key.

        When the next class cannot be fully scored, retain that K's last complete scored class.
        Cached members never allow a prefix of the new class to displace the incumbent.
        """
        omitted = stopped = False
        backstop = self.search.exact_score_backstop(self.archive.max_k)
        for k in sorted(self.archive.beams, key=lambda value: (abs(value - preferred_k), value)):
            group = self.archive.first_class(k)
            if not group:
                continue
            keys = tuple(sorted(panel.key for panel in group))
            if keys == self.scored_class_keys.get(k):
                continue
            novel = [panel for panel in group if panel.key not in self.score_cache]
            if novel and (stopped or (self.exact_scores >= backstop and self.exact_scores > 0)):
                self.omit("exact_score_ranked_truncation", len(novel), f"K={k} leading NLL class",
                          token=keys, kind="ranked_truncation")
                stopped = True
                continue
            if self.exact_scores + len(novel) > 2 * backstop:
                self.omit("exact_score_tie_class_at_budget", len(novel), f"K={k} leading NLL class", token=keys)
                omitted = stopped = True
                continue
            self.cache_hits += len(group) - len(novel)
            with diagnostic_phase("cavity"):
                scored = score_solutions(self.cavity_workspace, novel, self.active)
            for score in scored:
                self.score_cache[score.solution.key] = score
                if score.mean_field_failures:
                    self.mean_field_failed = True
            self.exact_scores += len(novel)
            values = tuple(self.score_cache[panel.key] for panel in group)
            self.class_scores[k] = values
            self.scored_class_keys[k] = keys
            representative, _ = select_scores(values, self.quanta.nll, self.content)
            self.representatives[k] = representative.solution
            self.scores[k] = representative
        return omitted

    def winning_scores(self, values=None, classes=None):
        """The complete score class after the wide-class rule and the smaller-K prior."""
        if values is None:
            _, classes, values = self.selection()
        group = [score for score, cls in zip(values, classes.classes) if cls == 0]
        if classes.spans[0] > WIDE_CLASS_QUANTA * self.quanta.nll:
            maximum = max(score.log_score for score in group)
            group = [score for score in group if score.log_score == maximum]
        best_k = min(score.k for score in group)
        return tuple(score for score in group if score.k == best_k)

    def selection_value(self):
        return max(score.log_score for score in self.winning_scores())

    def selection(self):
        """(selected PanelScore, its tie classes, the scores in class-input order): score classes
        (kind ``nll``, descending), then smaller K, then the content key."""
        values = [score for k in sorted(self.class_scores) for score in self.class_scores[k]]
        winner, classes = select_scores(values, self.quanta.nll, self.content)
        return winner, classes, values

    def selected(self):
        return self.selection()[0]

    def improved(self, after, before) -> bool:
        return improves(after.log_score, before.log_score, self.quanta.improvement)

    def margin_and_ties(self):
        """(score margin of the selected panel over the best other K, tied alternatives)."""
        best, classes, values = self.selection()
        others = [score.log_score for score in values if score.k != best.k]
        margin = self.selection_value() - max(others) if others else math.inf
        tied = len(self.winning_scores(values, classes)) - 1
        return best, margin, tied

    # ---------------------------------------------------------------------------------- moves
    def beam(self, k: int):
        """The panels of K to expand: the archive's leading whole NLL classes up to the beam width (release 2.2)."""
        panels = list(self.archive.beams.get(k, ()))
        omissions = []
        kept, _ = whole_class_cut(panels, self.archive.beam_classes.get(k, list(range(len(panels)))),
                                  self.beam_width, omissions, cause="beam", scope=f"K={k} expansion NLL classes")
        for item in omissions:
            self.omit(**item, token=tuple(sorted(panel.key for panel in panels[len(kept):])))
        return kept

    def soft(self, solution) -> SoftBirths:
        if solution.k >= self.ceiling:
            return SoftBirths((), 0, 0)
        return soft_births(self.workspace, solution, self.likelihood, self.residual_inputs,
                           self.config, self.soft_cache, counter=self.counter)

    def start(self) -> None:
        data_solutions = initial_solutions(self.workspace, self.likelihood, self.active,
                                           self.config, counter=self.counter, omissions=self.omissions)
        self.omissions = contextual_omissions(self.omissions, 1, "initial")
        self.limits.extend(item["cause"] for item in self.omissions if item.get("kind") != "ranked_truncation")
        self.data_starts = len(data_solutions)
        self.archive.add(data_solutions)
        if self.refresh(1):
            self.limits.append("exact_score_tie_class_at_budget")
        if self.mean_field_failed:
            self.limits.append("mean_field_nonconvergence")

    def parent_group(self, k, expanded):
        """The next entire unexpanded NLL class in the selected K's frozen beam."""
        panels = self.beam(k)
        if not panels:
            return ()
        classes = tie_classes([panel.nll for panel in panels], self.quanta.nll,
                              content=[self.content(panel) for panel in panels])
        for cls in sorted(set(classes.classes.tolist())):
            group = tuple(panels[int(i)] for i in classes.members(cls) if panels[int(i)].key not in expanded)
            if group:
                return group
        return ()

    def expand(self, parents, phase: str, before) -> None:
        """Generate all peers' neighbours before testing improvement or selecting a new K."""
        if isinstance(parents, FixedKSolution):
            parents = (parents,)
        config = self.config
        search = self.search
        parent_k = parents[0].k
        baseline = self.selection_value()
        generated, births = [], []
        local_limits = []
        omissions = []
        expanded = self.ordinary_expanded if phase == "ordinary" else self.residual_expanded
        expanded.update(parent.key for parent in parents)
        for parent in parents:
            if phase == "ordinary":
                moves = ordinary_moves(self.workspace, parent, self.likelihood, self.growth,
                                       self.ceiling, self.cuts, config, counter=self.counter)
                self.heuristic_cuts |= parent.k > 1 and self.cuts.heuristic(parent.k)
                generated.extend(moves.children)
                births.extend(moves.births)
                local_limits.extend(moves.limits)
                omissions.extend(contextual_omissions(moves.omissions, parent.k, phase))
            else:
                result = self.soft(parent)
                self.residual_calls += 1
                self.residual_rows += result.emitted
                generated.extend(result.children)
                births.extend(result.children)
                omissions.extend(contextual_omissions(result.omissions, parent.k, phase))
        self.archive.add(deduplicate(generated))
        if self.refresh(parent_k):
            local_limits.append("exact_score_tie_class_at_budget")
        if not improves(self.selection_value(), baseline, self.quanta.improvement):
            # Every parent has the same K; replacements depend only on the union of births.
            children, _ = replacements(self.workspace, parents[0], deduplicate(births), config,
                                        self.replacement_limit, omissions=omissions)
            generated.extend(children)
            self.archive.add(children)
            if self.refresh(parent_k):
                local_limits.append("exact_score_tie_class_at_budget")
        if (phase == "residual" and not improves(self.selection_value(), baseline, self.quanta.improvement)
                and parent_k + 2 <= self.ceiling and parent_k + 1 in self.class_scores):
            # Freeze all data-selected tied K+1 bridge parents before adding any bridge children.
            bridge_values = self.class_scores[parent_k + 1]
            _, bridge_classes = select_scores(bridge_values, self.quanta.nll, self.content)
            bridges = self.winning_scores(bridge_values, bridge_classes)
            paired = []
            for bridge_score in bridges:
                bridge = bridge_score.solution
                result = data_births(self.workspace, bridge, self.growth,
                                     config.model.max_iterations, search, counter=self.counter)
                omissions.extend(contextual_omissions(result.omissions, bridge.k, "bridge"))
                paired.extend(child for child in result.children if child.k == parent_k + 2)
                result = self.soft(bridge)
                self.residual_calls += 1
                self.residual_rows += result.emitted
                omissions.extend(contextual_omissions(result.omissions, bridge.k, "bridge"))
                paired.extend(child for child in result.children if child.k == parent_k + 2)
            paired = deduplicate(paired)
            generated.extend(paired)
            self.archive.add(paired)
            if self.refresh(parent_k):
                local_limits.append("exact_score_tie_class_at_budget")
        self.omissions.extend(omissions)
        local_limits.extend(item["cause"] for item in omissions if item.get("kind") != "ranked_truncation")
        complete = self.ordinary_complete if phase == "ordinary" else self.residual_complete
        no_gain = not improves(self.selection_value(), baseline, self.quanta.improvement)
        for parent in parents:
            complete[parent.key] = not local_limits
            if phase == "residual" and no_gain:
                self.no_improvement.add(parent.key)
            self.generated_offsets.setdefault(parent.key, set()).update(
                child.k - parent_k for child in generated if child.k - parent_k in _MOVE_OFFSETS)
        self.limits.extend(local_limits)
        if self.mean_field_failed:
            self.limits.append("mean_field_nonconvergence")
        self.expansions += len(parents)

    def run(self) -> None:
        self.start()
        if not self.scores:
            self.stop_reason = "no_incumbent_search_limited"
            return
        search = self.search
        while True:
            before = self.selected()
            parents = self.parent_group(before.k, self.ordinary_expanded)
            phase = "ordinary"
            if not parents:
                parents = self.parent_group(before.k, self.residual_expanded)
                phase = "residual"
            if not parents:
                self.stop_reason = ("generated_neighbourhood_local_optimum" if not self.limits
                                    else "incumbent_neighbourhood_search_limited")
                break
            budget = search.expansion_backstop(self.archive.max_k)
            if (self.expansions >= budget and self.expansions > 0) or self.expansions + len(parents) > 2 * budget:
                kind = "ranked_truncation" if self.expansions >= budget and self.expansions > 0 else "tie_class_at_budget"
                cause = "expansion_" + kind
                self.omit(cause, len(parents), f"K={before.k} {phase} next parent class",
                          token=tuple(sorted(parent.key for parent in parents)), kind=kind)
                self.stop_reason = "expansion_backstop"
                break
            self.expand(parents, phase, before)
        if self.mean_field_failed:
            self.limits.append("mean_field_nonconvergence")

    def certified(self, key: bytes, best_k: int, unique_limits) -> bool:
        offsets = self.generated_offsets.get(key, set())
        return bool(
            key in self.ordinary_expanded and key in self.residual_expanded
            and self.ordinary_complete.get(key, False) and self.residual_complete.get(key, False)
            and all(best_k + offset in self.scores for offset in offsets)
            and key in self.no_improvement and not unique_limits and not self.mean_field_failed)


def _search(likelihood, observed, config) -> BlockSearch:
    """The search of a block whose samples are in canonical order (see search_block)."""
    forward = _Search(likelihood, observed, config)
    forward.run()
    omissions = [*forward.omissions, *forward.archive.omissions]
    limits = list(forward.limits)
    limits.extend(item["cause"] for item in omissions if item.get("kind") != "ranked_truncation")
    unique_limits = tuple(dict.fromkeys(limits))
    best = None
    margin, tied, certified = math.inf, 0, False
    winners = ()
    if forward.scores:
        best, classes, values = forward.selection()
        _, margin, tied = forward.margin_and_ties()
        winners = forward.winning_scores(values, classes)
        forward.counter.add("discovery.selection", classes)
        certified = all(forward.certified(score.solution.key, score.k, unique_limits) for score in winners)
    selected_keys = {score.solution.key for score in winners}
    k_values = np.array(sorted(forward.scores), dtype=np.int16)
    stop_reason = forward.stop_reason
    if unique_limits and stop_reason == "generated_neighbourhood_local_optimum":
        stop_reason = "incumbent_neighbourhood_search_limited"
    summary = SearchSummary(
        k_values=k_values,
        k_log_scores=np.array([max(score.log_score for score in forward.class_scores[int(k)]) for k in k_values],
                              dtype=np.float64),
        natural_k_ceiling=int(forward.ceiling),
        expansions=int(forward.expansions),
        data_starts=int(forward.data_starts),
        exact_scores=int(forward.exact_scores),
        exact_score_cache_hits=int(forward.cache_hits),
        residual_calls=int(forward.residual_calls),
        residual_rows=int(forward.residual_rows),
        limits=unique_limits,
        heuristic_gauge_cuts=bool(forward.heuristic_cuts),
        stop_reason=stop_reason, certified=bool(certified),
        tie_quantum=float(forward.quanta.nll),
        ambiguous=bool(tied > 0), tied_alternatives=int(tied),
        score_margin=float(margin), tie_events=forward.counter.record(),
        search_incomplete=any(item.get("kind") != "ranked_truncation" for item in omissions),
        omissions=tuple(omissions), unresolved=best is None,
        archive_max_k=int(forward.archive.max_k),
        selected_mean_field_failures=sum(score.mean_field_failures for score in winners),
        other_mean_field_failures=sum(score.mean_field_failures for key, score in forward.score_cache.items()
                                      if key not in selected_keys))
    alternatives = tuple(score.solution for score in winners if score.solution.key != best.solution.key)
    return BlockSearch(selected=None if best is None else best.solution, summary=summary, alternatives=alternatives)
