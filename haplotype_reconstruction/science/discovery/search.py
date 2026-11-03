"""Adaptive reversible block search with whole tied-class decisions.

Owns: the complete search state (SearchState) and its results (BlockSearch, SearchSummary, Omission), the per-K
archive, the cavity refresh with its exact-score backstop, selection of the winning panel set (select_scores), the
expansion loop and the search status; natural_k_ceiling.
Must not: read storage, the environment or a clock, log, choose routes or own an executor; the block runner
(run.py) and the stage layer do that. Diagnostics go only to the BlockDiagnostics the caller passes.
Method: docs/methods/discovery.md, units di21-di25 (P0 DI21-DI25).
Identity: archive and parent classes cut whole at ``beam_width`` (straddling through twice it); refresh visits K in
(|K - preferred K|, K) order, counts only cache misses and keeps the previous complete scored class when a new one
cannot fit; first-scored PanelScore objects are kept and the archive's fitted-key memory is permanent; ordinary moves
precede residual births, replacements come only after no improvement, the frozen K + 1 bridge family precedes its
K + 2 children; ``exact_score_backstop`` and ``expansion_backstop`` (K-scaled) and ``max_replacement_children`` are
unchanged budgets, tested at class boundaries.

Every member of each K's leading NLL class is cavity-scored (memoized by panel key). The
best cavity class, then the smaller-K prior, chooses a complete alternative set. Byte order
only labels its representative. Each expansion freezes a complete unexpanded NLL class,
merges all peers' neighbours before reselecting, and tests budgets at class boundaries.
Computational omissions are recorded separately from complete fitted alternatives.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

from haplotype_reconstruction.contracts.roles import record_field
from haplotype_reconstruction.core.ties import (
    NOISE_MOTION,
    WIDE_CLASS_QUANTA,
    BlockQuanta,
    TieCounter,
    block_scale,
    fixed_policy,
    improves,
    tie_classes,
)

from .cavity import CavityWorkspace, PanelScore, score_solutions
from .diagnostics import BlockDiagnostics
from .fixed_k.fit import prepare_workspace
from .fixed_k.workspace import FixedKWorkspace
from .moves import (
    DataBirths,
    GrowthInputs,
    MaximumCuts,
    SoftBirths,
    data_births,
    growth_inputs,
    initial_solutions,
    ordinary_moves,
    replacements,
    soft_births,
)
from .panel import FixedKSolution, deduplicate, known_content_key, solution_tie_mask, unknown_alleles, whole_class_cut
from .parameters import DiscoveryConfig, SearchConfig
from .residual_births import ResidualInputs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from haplotype_reconstruction.core.ties import TieClasses

# Relative K of the moves a search step generates (death, same-K, birth, bridge).
_MOVE_OFFSETS = (-1, 0, 1, 2)

__all__ = ["SearchState", "SearchSummary", "BlockSearch", "natural_k_ceiling", "select_scores"]


class Omission(TypedDict):
    """A whole-class budget decision, in production order (P0 DI25)."""

    kind: str
    cause: str
    count: int
    scope: str
    count_kind: str


def contextual_omissions(items, k, phase):
    """Copy geometry omission scopes; never mutate cached generator receipts."""
    return [
        dict(item, scope=f"{item['scope']}; generating K={k} phase={phase}")
        if "landmark" in item["cause"] or "clique" in item["cause"]
        else item
        for item in items
    ]


@dataclass(frozen=True)
class SearchSummary:
    """Scientific search counts, status and selected-set decisions, plus diagnostics.

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
    Optional recorded diagnostics (defaults permit records that omit them):
    tie_quantum: the block's ``nll`` tie quantum (selection and archive orders).
    escalated, conventions: recorded metadata; neither field selects a search mode.
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
        that winning set, including panels absent from the archive. None denotes an unrecorded membership.
    """

    k_values: np.ndarray = record_field("state", axes=("R",), dtype="int16")
    k_log_scores: np.ndarray = record_field("state", axes=("R",), dtype="float64", units="nats")
    natural_k_ceiling: int = record_field("state", dtype="int")
    expansions: int = record_field("state", dtype="int")
    data_starts: int = record_field("state", dtype="int")
    exact_scores: int = record_field("state", dtype="int")
    exact_score_cache_hits: int = record_field("state", dtype="int")
    residual_calls: int = record_field("state", dtype="int")
    residual_rows: int = record_field("state", dtype="int")
    limits: tuple[str, ...] = record_field("state", dtype="tuple[str, ...]")
    heuristic_gauge_cuts: bool = record_field("state", dtype="bool")
    stop_reason: str = record_field("state", dtype="str")
    certified: bool = record_field("state", dtype="bool")
    tie_quantum: float = record_field("state", dtype="float64", units="nats", default=0.0)
    escalated: bool = record_field("diagnostic", dtype="bool", default=False)
    conventions: int = record_field("diagnostic", dtype="int", default=1)
    ambiguous: bool = record_field("call", dtype="bool", default=False)
    tied_alternatives: int = record_field("call", dtype="int", default=0)
    score_margin: float = record_field("state", dtype="float64", units="nats", default=math.inf)
    tie_events: dict = record_field("diagnostic", dtype="TieCounts", default_factory=dict)
    search_incomplete: bool = record_field("state", dtype="bool", default=False)
    omissions: tuple = record_field("state", dtype="tuple[Omission,...]", default=())
    unresolved: bool = record_field("call", dtype="bool", default=False)
    # None means not measured in older records, not zero work/failures.
    archive_max_k: int | None = record_field("state", dtype="int | None", default=None)
    wall_seconds: float | None = record_field("diagnostic", dtype="float64 | None", default=None)
    cpu_seconds: float | None = record_field("diagnostic", dtype="float64 | None", default=None)
    phase_seconds: dict = record_field("diagnostic", dtype="mapping[str,mapping[str,float64]]", default_factory=dict)
    selected_mean_field_failures: int | None = record_field("state", dtype="int | None", default=None)
    other_mean_field_failures: int | None = record_field("state", dtype="int | None", default=None)


@dataclass(frozen=True)
class BlockSearch:
    """Selected fitted alternative set and summary; selected=None means no scored incumbent.

    Alternatives contain the other winning cavity-class members after the smaller-K prior.
    The representative is a label for this complete set, never a search decision.
    """

    selected: FixedKSolution | None = record_field("state", dtype="FixedKSolution | None")
    summary: SearchSummary = record_field("state", dtype="SearchSummary")
    alternatives: tuple[FixedKSolution, ...] = record_field("state", dtype="tuple[FixedKSolution, ...]", default=())


def natural_k_ceiling(n_samples: int, n_sites: int) -> int:
    """The largest identifiable K: 2 copies per sample with depth, at most 2^L rows."""
    sample_ceiling = 2 * int(n_samples)
    if n_sites < sample_ceiling.bit_length():
        return min(sample_ceiling, 1 << int(n_sites))
    return sample_ceiling


def select_scores(
    values: Sequence[PanelScore],
    quantum: float,
    content: Callable[[FixedKSolution], bytes],
    *,
    reverse: bool,
) -> tuple[PanelScore, TieClasses]:
    """(representative, TieClasses): cavity classes, wide-class rule, then smaller K (P0 DI23).

    ``values``: the scored panels; their ``log_score`` (nats) is ranked descending in tie classes of the fixed
    ``quantum`` (policy ``discovery.selection``, motion of kind ``nll``), then fewer founders (``k``), then
    ``content(solution)`` labels members. Returns the first panel in decision order and the TieClasses over
    ``values``. Content labels the representative only; _Search.winning_scores retains every equally ranked panel.
    Mutates nothing; an empty ``values`` raises IndexError (no winner).
    """
    classes = tie_classes(
        [score.log_score for score in values],
        fixed_policy("discovery.selection", quantum, reverse=reverse).quantum(),
        content=[content(score.solution) for score in values],
        secondary=[(score.k,) for score in values],
        descending=True,
        motion=NOISE_MOTION["nll"],
        reverse=reverse,
    )
    return values[classes.winner], classes


class _Archive:
    """Per K the retained panels in decision order: tie classes of NLL (kind ``nll``), the content
    key inside a class labels members. Each add
    receives a complete proposal round. The retained list combines previous members and the new
    arrivals before a whole-class cut at ``width``; a straddling hard-bound class is omitted whole."""

    def __init__(
        self, width: int, quanta: BlockQuanta, content, counter: TieCounter, observed=None, *, reverse: bool = False
    ):
        self.reverse = reverse
        self.width = int(width)
        self.quantum = float(quanta.nll)
        self.content = content
        self.counter = counter
        self.observed = observed
        self.fitted: dict[bytes, float] = {}
        self.beams: dict[int, list] = {}
        self.beam_classes: dict[int, list] = {}
        self.omissions: list[Omission] = []
        self.science: dict[bytes, bytes] = {}
        self.max_k = 0

    def known_key(self, solution) -> bytes:
        """The panel's science (rows with alleles no assigned carrier observes masked; release 2.2)."""
        key = self.science.get(solution.key)
        if key is None:
            if self.observed is None or solution.assignments is None:
                key = solution.key
            else:
                key = known_content_key(
                    solution.haplotypes,
                    unknown_alleles(
                        solution.assignments, self.observed, solution.k, tie_mask=solution_tie_mask(solution)
                    ),
                )
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
                if solution.nll < twin.nll - self.quantum or (
                    not twin.nll < solution.nll - self.quantum and solution.key < twin.key
                ):
                    panels[:] = [entry for entry in panels if entry is not twin]
                else:
                    continue
            panels.append(solution)
            touched[solution.k] = None
        for k in touched:
            ordered, classes = self._ordered(self.beams[k])
            kept, _ = whole_class_cut(
                ordered,
                classes,
                self.width,
                cast(list[dict], self.omissions),
                cause="archive",
                scope=f"K={k} archive NLL classes",
            )
            self.beams[k] = kept
            self.beam_classes[k] = classes[: len(kept)]

    def _ordered(self, panels):
        """(panels in decision order, their NLL tie-class ids)."""
        if len(panels) == 1:
            return panels, [0]
        classes = tie_classes(
            [panel.nll for panel in panels],
            fixed_policy("discovery.archive", self.quantum, reverse=self.reverse).quantum(),
            content=[self.content(panel) for panel in panels],
            motion=NOISE_MOTION["nll"],
            reverse=self.reverse,
        )
        ordered = [panels[index] for index in classes.order]
        cut_inside = (
            len(ordered) > self.width
            and classes.classes[classes.order[self.width - 1]] == classes.classes[classes.order[self.width]]
        )
        self.counter.add("discovery.archive", classes, decided_inside=bool(classes.tied_winner or cut_inside))
        return ordered, [int(classes.classes[index]) for index in classes.order]

    def first_class(self, k: int) -> list:
        """The retained panels of K in the first NLL class, in decision order."""
        panels = list(self.beams.get(k, ()))
        if len(panels) <= 1:
            return panels
        classes = tie_classes(
            [panel.nll for panel in panels],
            fixed_policy("discovery.archive", self.quantum, reverse=self.reverse).quantum(),
            content=[self.content(panel) for panel in panels],
            reverse=self.reverse,
        )
        return [panels[int(index)] for index in classes.order if classes.classes[int(index)] == 0]


class SearchState:
    """Mutable history of one block search in canonical sample order (P0 DI21–DI25).

    The unbounded score cache and permanent fitted archive are scientific state:
    eviction changes cache-miss budgets and admissible trajectories. Arrays are
    read-only inputs; all counters, caches and phase sets belong to this search.
    """

    reverse: bool
    keep: tuple[str, ...]
    threads: int
    diagnostics: BlockDiagnostics | None
    config: DiscoveryConfig
    search: SearchConfig
    likelihood: np.ndarray
    observed: np.ndarray
    active: np.ndarray
    ceiling: int
    quanta: BlockQuanta
    counter: TieCounter
    content: Callable[[FixedKSolution], bytes]
    workspace: FixedKWorkspace
    growth: GrowthInputs
    residual_inputs: ResidualInputs
    cuts: MaximumCuts
    cavity_workspace: CavityWorkspace
    archive: _Archive
    beam_width: int
    replacement_limit: int
    exact_scores: int
    cache_hits: int
    mean_field_failed: bool
    residual_calls: int
    residual_rows: int
    heuristic_cuts: bool
    stop_reason: str
    expansions: int
    data_starts: int

    def __init__(
        self,
        likelihood: np.ndarray,
        observed: np.ndarray,
        config: DiscoveryConfig,
        *,
        reverse: bool = False,
        keep: tuple[str, ...] = (),
        diagnostics: BlockDiagnostics | None = None,
        threads: int = 1,
    ) -> None:
        self.reverse = reverse
        self.keep = tuple(keep)
        self.threads = threads
        self.diagnostics = diagnostics
        search = config.search
        self.config = config
        self.search = search
        self.likelihood = likelihood
        self.observed = observed
        self.active = np.ascontiguousarray(observed.any(axis=1))
        self.ceiling = natural_k_ceiling(int(self.active.sum()), likelihood.shape[1])
        self.quanta = BlockQuanta.for_scale(block_scale(likelihood, observed, config.model.unknown_copy_penalty))
        self.counter = TieCounter()
        self.content = lambda solution: solution.key
        self.workspace = prepare_workspace(
            likelihood, observed, config.model, self.quanta, reverse=reverse, threads=threads
        )
        self.growth = growth_inputs(likelihood, observed, self.counter, reverse=reverse, keep=keep)
        self.residual_inputs = ResidualInputs(likelihood, observed, reverse=reverse, keep=keep, diagnostics=diagnostics)
        self.cuts = MaximumCuts(search.exact_cut_max_k, search.max_cut_ties, reverse=reverse, keep=keep)
        self.cavity_workspace = CavityWorkspace(
            likelihood if bool(self.active.all()) else likelihood[self.active], config.cavity
        )
        width = search.beam_width
        self.archive = _Archive(width, self.quanta, self.content, self.counter, observed, reverse=reverse)
        self.beam_width = int(search.beam_width)
        self.replacement_limit = int(search.max_replacement_children)
        self.representatives: dict[int, FixedKSolution] = {}
        self.score_cache: dict[bytes, PanelScore] = {}
        self.scores: dict[int, PanelScore] = {}
        self.class_scores: dict[int, tuple[PanelScore, ...]] = {}
        self.scored_class_keys: dict[int, tuple[bytes, ...]] = {}
        self.omissions: list[Omission] = []
        self.omission_keys: set[tuple] = set()
        self.soft_cache: dict[tuple[bytes, bytes], SoftBirths] = {}
        self.limits: list[str] = []
        self.exact_scores = 0
        self.cache_hits = 0
        self.mean_field_failed = False
        self.ordinary_expanded: set[bytes] = set()
        self.residual_expanded: set[bytes] = set()
        self.ordinary_complete: dict[bytes, bool] = {}
        self.residual_complete: dict[bytes, bool] = {}
        self.generated_offsets: dict[bytes, set[int]] = {}
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
                self.omit(
                    "exact_score_ranked_truncation",
                    len(novel),
                    f"K={k} leading NLL class",
                    token=keys,
                    kind="ranked_truncation",
                )
                stopped = True
                continue
            if self.exact_scores + len(novel) > 2 * backstop:
                self.omit("exact_score_tie_class_at_budget", len(novel), f"K={k} leading NLL class", token=keys)
                omitted = stopped = True
                continue
            self.cache_hits += len(group) - len(novel)
            scored = score_solutions(self.cavity_workspace, novel, self.active)
            for score in scored:
                self.score_cache[score.solution.key] = score
                if score.mean_field_failures:
                    self.mean_field_failed = True
            self.exact_scores += len(novel)
            values = tuple(self.score_cache[panel.key] for panel in group)
            self.class_scores[k] = values
            self.scored_class_keys[k] = keys
            representative, _ = select_scores(values, self.quanta.nll, self.content, reverse=self.reverse)
            self.representatives[k] = representative.solution
            self.scores[k] = representative
        return omitted

    def winning_scores(self, values=None, classes=None):
        """The complete score class after the wide-class rule and the smaller-K prior."""
        if values is None:
            _, classes, values = self.selection()
        group = [score for score, cls in zip(values, classes.classes, strict=False) if cls == 0]
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
        winner, classes = select_scores(values, self.quanta.nll, self.content, reverse=self.reverse)
        return winner, classes, values

    def selected(self):
        return self.selection()[0]

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
        omissions: list[dict] = []
        kept, _ = whole_class_cut(
            panels,
            self.archive.beam_classes.get(k, list(range(len(panels)))),
            self.beam_width,
            omissions,
            cause="beam",
            scope=f"K={k} expansion NLL classes",
        )
        for item in omissions:
            self.omit(**item, token=tuple(sorted(panel.key for panel in panels[len(kept) :])))
        return kept

    def soft(self, solution) -> SoftBirths:
        if solution.k >= self.ceiling:
            return SoftBirths((), 0, 0)
        return soft_births(
            self.workspace,
            solution,
            self.likelihood,
            self.residual_inputs,
            self.config,
            self.soft_cache,
            counter=self.counter,
        )

    def start(self) -> None:
        data_solutions = initial_solutions(
            self.workspace,
            self.likelihood,
            self.active,
            self.config,
            counter=self.counter,
            # Omission records are the generator's omission dictionaries (the TypedDict view of them).
            omissions=self.omissions,  # type: ignore[arg-type]
            keep=self.keep,
        )
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
        classes = tie_classes(
            [panel.nll for panel in panels],
            fixed_policy("discovery.archive", self.quanta.nll, reverse=self.reverse).quantum(),
            content=[self.content(panel) for panel in panels],
            reverse=self.reverse,
        )
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
        generated: list[FixedKSolution] = []
        births: list[FixedKSolution] = []
        result: SoftBirths | DataBirths
        local_limits: list[str] = []
        omissions: list[dict] = []
        expanded = self.ordinary_expanded if phase == "ordinary" else self.residual_expanded
        expanded.update(parent.key for parent in parents)
        for parent in parents:
            if phase == "ordinary":
                moves = ordinary_moves(
                    self.workspace,
                    parent,
                    self.likelihood,
                    self.growth,
                    self.ceiling,
                    self.cuts,
                    config,
                    counter=self.counter,
                )
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
        self.archive.add(deduplicate(generated, reverse=self.reverse))
        if self.refresh(parent_k):
            local_limits.append("exact_score_tie_class_at_budget")
        if not improves(self.selection_value(), baseline, self.quanta.improvement):
            # Every parent has the same K; replacements depend only on the union of births.
            children, _ = replacements(
                self.workspace,
                parents[0],
                deduplicate(births, reverse=self.reverse),
                config,
                self.replacement_limit,
                omissions=omissions,
            )
            generated.extend(children)
            self.archive.add(children)
            if self.refresh(parent_k):
                local_limits.append("exact_score_tie_class_at_budget")
        if (
            phase == "residual"
            and not improves(self.selection_value(), baseline, self.quanta.improvement)
            and parent_k + 2 <= self.ceiling
            and parent_k + 1 in self.class_scores
        ):
            # Freeze all data-selected tied K+1 bridge parents before adding any bridge children.
            bridge_values = self.class_scores[parent_k + 1]
            _, bridge_classes = select_scores(bridge_values, self.quanta.nll, self.content, reverse=self.reverse)
            bridges = self.winning_scores(bridge_values, bridge_classes)
            paired: list[FixedKSolution] | tuple[FixedKSolution, ...] = []
            for bridge_score in bridges:
                bridge = bridge_score.solution
                result = data_births(
                    self.workspace, bridge, self.growth, config.model.max_iterations, search, counter=self.counter
                )
                omissions.extend(contextual_omissions(result.omissions, bridge.k, "bridge"))
                cast(list[FixedKSolution], paired).extend(child for child in result.children if child.k == parent_k + 2)
                result = self.soft(bridge)
                self.residual_calls += 1
                self.residual_rows += result.emitted
                omissions.extend(contextual_omissions(result.omissions, bridge.k, "bridge"))
                cast(list[FixedKSolution], paired).extend(child for child in result.children if child.k == parent_k + 2)
            paired = deduplicate(paired, reverse=self.reverse)
            generated.extend(paired)
            self.archive.add(paired)
            if self.refresh(parent_k):
                local_limits.append("exact_score_tie_class_at_budget")
        self.omissions.extend(cast(list[Omission], omissions))
        local_limits.extend(item["cause"] for item in omissions if item.get("kind") != "ranked_truncation")
        complete = self.ordinary_complete if phase == "ordinary" else self.residual_complete
        no_gain = not improves(self.selection_value(), baseline, self.quanta.improvement)
        for parent in parents:
            complete[parent.key] = not local_limits
            if phase == "residual" and no_gain:
                self.no_improvement.add(parent.key)
            self.generated_offsets.setdefault(parent.key, set()).update(
                child.k - parent_k for child in generated if child.k - parent_k in _MOVE_OFFSETS
            )
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
                self.stop_reason = (
                    "generated_neighbourhood_local_optimum"
                    if not self.limits
                    else "incumbent_neighbourhood_search_limited"
                )
                break
            budget = search.expansion_backstop(self.archive.max_k)
            if (self.expansions >= budget and self.expansions > 0) or self.expansions + len(parents) > 2 * budget:
                kind = (
                    "ranked_truncation" if self.expansions >= budget and self.expansions > 0 else "tie_class_at_budget"
                )
                cause = "expansion_" + kind
                self.omit(
                    cause,
                    len(parents),
                    f"K={before.k} {phase} next parent class",
                    token=tuple(sorted(parent.key for parent in parents)),
                    kind=kind,
                )
                self.stop_reason = "expansion_backstop"
                break
            self.expand(parents, phase, before)
        if self.mean_field_failed:
            self.limits.append("mean_field_nonconvergence")

    def certified(self, key: bytes, best_k: int, unique_limits) -> bool:
        offsets = self.generated_offsets.get(key, set())
        return bool(
            key in self.ordinary_expanded
            and key in self.residual_expanded
            and self.ordinary_complete.get(key, False)
            and self.residual_complete.get(key, False)
            and all(best_k + offset in self.scores for offset in offsets)
            and key in self.no_improvement
            and not unique_limits
            and not self.mean_field_failed
        )


# The frozen extractor instruments this private historical spelling.
# Both names are the same state class; there is one implementation.
_Search = SearchState


def _search(
    likelihood: np.ndarray,
    observed: np.ndarray,
    config: DiscoveryConfig,
    *,
    reverse: bool = False,
    keep: tuple[str, ...] = (),
    diagnostics: BlockDiagnostics | None = None,
    threads: int = 1,
) -> BlockSearch:
    """The search of a block whose samples are in canonical order (see search_block)."""
    forward = SearchState(
        likelihood, observed, config, reverse=reverse, keep=keep, diagnostics=diagnostics, threads=threads
    )
    forward.run()
    omissions = [*forward.omissions, *forward.archive.omissions]
    limits = list(forward.limits)
    limits.extend(item["cause"] for item in omissions if item.get("kind") != "ranked_truncation")
    unique_limits = tuple(dict.fromkeys(limits))
    best = None
    margin, tied, certified = math.inf, 0, False
    winners: list[PanelScore] | tuple[PanelScore, ...] = ()
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
        k_log_scores=np.array(
            [max(score.log_score for score in forward.class_scores[int(k)]) for k in k_values], dtype=np.float64
        ),
        natural_k_ceiling=int(forward.ceiling),
        expansions=int(forward.expansions),
        data_starts=int(forward.data_starts),
        exact_scores=int(forward.exact_scores),
        exact_score_cache_hits=int(forward.cache_hits),
        residual_calls=int(forward.residual_calls),
        residual_rows=int(forward.residual_rows),
        limits=unique_limits,
        heuristic_gauge_cuts=bool(forward.heuristic_cuts),
        stop_reason=stop_reason,
        certified=bool(certified),
        tie_quantum=float(forward.quanta.nll),
        ambiguous=bool(tied > 0),
        tied_alternatives=int(tied),
        score_margin=float(margin),
        tie_events={
            entry.kind: dict(
                decisions=entry.decisions, inside_tie=entry.inside_tie, wide=entry.wide, boundary=entry.boundary
            )
            for entry in forward.counter.record().entries
        },
        search_incomplete=any(item.get("kind") != "ranked_truncation" for item in omissions),
        omissions=tuple(omissions),
        unresolved=best is None,
        archive_max_k=int(forward.archive.max_k),
        selected_mean_field_failures=sum(score.mean_field_failures for score in winners),
        other_mean_field_failures=sum(
            score.mean_field_failures for key, score in forward.score_cache.items() if key not in selected_keys
        ),
    )
    if diagnostics is not None:
        diagnostics.capture_fit_counts(forward.workspace.fit_schedule_counts, forward.workspace.endpoint_family_counts)
    alternatives = tuple(
        score.solution for score in winners if score.solution.key != cast(PanelScore, best).solution.key
    )
    return BlockSearch(selected=None if best is None else best.solution, summary=summary, alternatives=alternatives)
