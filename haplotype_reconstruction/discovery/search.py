"""Adaptive reversible search over complete panels of one block.

The archive ranks every fitted complete panel (one per canonical key, the first fit seen unless a
later fit is better by more than the block's ``nll`` tie quantum) and keeps, per K, the panels the
search reads: the first ``beam_width`` in tie-class order of NLL; the first of each represented K is
its *representative* and is cavity-scored (score cached by key; at most
``exact_score_backstop(K_rep)`` scores, K_rep the largest represented K). The selected panel is the
representative with the best score (ties: smaller K, then key).

Each expansion takes the selected K's beam (its ``beam_width`` first archived panels): the first
panel not yet expanded ordinarily gets the ordinary moves (deaths, gauge rewires, data births;
then, if the score did not improve, birth-delete replacements); once every beam panel was
expanded ordinarily, the first not yet expanded in the residual phase gets soft-residual births
(then replacements, then, if still no improvement, a K + 2 bridge: data and soft births of the
K + 1 representative). The search stops when the beam is exhausted in both phases, or at the
reported expansion backstop. There is no K grid and no K cap besides the identifiability ceiling
min(2 N_active, 2^L).

Release 2.1 (noise stability; design/DESIGN.md section 3 and A-1..A-3):
* Samples are searched in the run's canonical order (``run_sample_order`` of the sample names when
  the caller passes them; otherwise the caller's order, which the evidence stage makes canonical),
  and the selected solution's per-sample arrays are returned in the input order: a reordering of
  the samples changes nothing.
* Every order and winner uses anchored tie classes (maths.ties) with the block's tie quanta
  (``BlockQuanta`` from the block's null NLL): the archive per K by NLL classes (kind ``nll``)
  with the canonical key inside a class; the selection by cavity-score classes (kind ``nll``),
  then smaller K, then key; improvement tests by the ``improvement`` quantum. Values that differ
  only by rounding or by perturbation-level amounts never decide, and neither does array order.
* Optional, off by default (owner decision 2): an escalation of fragile blocks and a second search
  under the reversed tie convention, both pooled into one archive; the selection rule over the
  pool is unchanged (within K the lowest NLL, across K the cavity score).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..maths.ties import (NOISE_MOTION, BlockQuanta, TieCounter, block_scale, improves,
                          inverse_permutation, run_sample_order, tie_classes)
from .cavity import CavityWorkspace, score_solutions
from .fixed_k.fit import prepare_workspace
from .moves import (MaximumCuts, SoftBirths, data_births, growth_inputs, initial_solutions,
                    ordinary_moves, replacements, soft_births)
from .panel import FixedKSolution, deduplicate
from .residual_births import ResidualInputs

# Relative K of the moves a search step generates (death, same-K, birth, bridge).
_MOVE_OFFSETS = (-1, 0, 1, 2)
# Content keys of the reversed tie convention: the complemented 0/1 row bytes (reverse lexicographic order of
# equally long keys).
_COMPLEMENT = bytes.maketrans(bytes([0, 1]), bytes([1, 0]))


@dataclass(frozen=True)
class SearchSummary:
    """What one block's search did (diagnostics; nothing downstream decides on it).

    k_values: int16[R] represented K values, ascending; k_log_scores: float64[R] the cavity
        log score (leave-one-out predictive + telescoping K prior + unordered set code) of each
        K's representative. The selected K maximises it (tie classes: smaller K, then key).
    natural_k_ceiling: identifiability ceiling min(2 N_active, 2^L).
    expansions, data_starts, exact_scores, exact_score_cache_hits: work counts.
    residual_calls, residual_rows: soft-residual birth calls and the novel rows they emitted.
    limits: reported search limitations in first-occurrence order, from
        "expansion_backstop", "exact_score_backstop", "proposal_budget",
        "replacement_proposal_budget", "soft_residual_proposal_budget",
        "gauge_max_cut_ties_truncated", "mean_field_nonconvergence", "data_birth_tie_budget",
        "escalation_backstop".
    heuristic_gauge_cuts: some expanded panel had K > exact_cut_max_k, so its gauge rewires came
        from the degree-ordered local-search maximum cut (reported here, not as a limit).
    stop_reason: "generated_neighbourhood_local_optimum", "incumbent_neighbourhood_search_limited"
        or "expansion_backstop".
    certified: the selected panel's generated neighbourhood was completely generated and scored
        without improvement (a local certificate, not a global one).
    Release 2.1 (defaults keep 2.0 records decodable):
    tie_quantum: the block's ``nll`` tie quantum (selection and archive orders).
    escalated: the escalation ran (SearchConfig.escalation).
    conventions: searches pooled (2 when the reversed-convention search ran).
    ambiguous: the selection was taken inside a tie: another representative's score is in the
        selected score's tie class, or another panel is in the first NLL class of the selected K.
    tied_alternatives: the number of such other representatives and panels.
    score_margin: the selected score minus the best other represented K's score (inf if none).
    tie_events: maths.ties.TieCounter record {kind: {decisions, inside_tie, wide, boundary}}.
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


@dataclass(frozen=True)
class BlockSearch:
    """The selected solution of a block search and its summary."""

    selected: FixedKSolution
    summary: SearchSummary


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
    return FixedKSolution(
        haplotypes=solution.haplotypes,
        assignments=_readonly(np.ascontiguousarray(solution.assignments[index])),
        per_sample_cost=_readonly(np.ascontiguousarray(solution.per_sample_cost[index])),
        unknown_slots=_readonly(np.ascontiguousarray(solution.unknown_slots[index])),
        iterations=solution.iterations, nll=solution.nll, fixed_point=solution.fixed_point,
        key=solution.key, quanta=solution.quanta)


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
    result = _search(likelihood, observed, config)
    if order is None:
        return result
    return BlockSearch(selected=_sample_subset(result.selected, inverse_permutation(order)),
                       summary=result.summary)


def select_scores(values, quantum: float, content):
    """(winner, TieClasses) of PanelScores: cavity-score classes (descending, ``quantum``), then smaller
    K, then the content key of the panel (the selection rule across K; within K the archive's NLL order
    already chose each K's representative)."""
    classes = tie_classes([score.log_score for score in values], quantum,
                          content=[content(score.solution) for score in values],
                          secondary=[(score.k,) for score in values], descending=True,
                          motion=NOISE_MOTION["nll"])
    return values[classes.winner], classes


class _Archive:
    """Per K the retained panels in decision order: tie classes of NLL (kind ``nll``), the content
    key inside a class (forward: the canonical key; reversed convention: its complement). Every
    panel ever fitted keeps the NLL of its archived fit (a later fit of the same key replaces it only
    when better by more than the quantum). A K's retained list is re-ordered over its previous
    members and the new arrivals and cut at ``width`` (a cut inside a class falls by content)."""

    def __init__(self, width: int, quanta: BlockQuanta, content, counter: TieCounter):
        self.width = int(width)
        self.quantum = float(quanta.nll)
        self.content = content
        self.counter = counter
        self.fitted: dict[bytes, float] = {}
        self.beams: dict[int, list] = {}
        self.max_k = 0

    def add(self, solutions) -> None:
        touched: dict[int, None] = {}
        for solution in solutions:
            previous = self.fitted.get(solution.key)
            if previous is not None and not solution.nll < previous - self.quantum:
                continue
            self.fitted[solution.key] = solution.nll
            self.max_k = max(self.max_k, solution.k)
            panels = self.beams.setdefault(solution.k, [])
            if previous is not None:
                panels[:] = [entry for entry in panels if entry.key != solution.key]
            panels.append(solution)
            touched[solution.k] = None
        for k in touched:
            self.beams[k] = self._ordered(self.beams[k])[:self.width]

    def _ordered(self, panels):
        if len(panels) == 1:
            return panels
        classes = tie_classes([panel.nll for panel in panels], self.quantum,
                              content=[self.content(panel) for panel in panels],
                              motion=NOISE_MOTION["nll"])
        ordered = [panels[index] for index in classes.order]
        cut_inside = (len(ordered) > self.width
                      and classes.classes[classes.order[self.width - 1]]
                      == classes.classes[classes.order[self.width]])
        self.counter.add("discovery.archive", classes,
                         decided_inside=bool(classes.tied_winner or cut_inside))
        return ordered

    def first_class_size(self, k: int) -> int:
        """Members of the first NLL class among the retained panels of K."""
        panels = self.beams.get(k, ())
        if len(panels) <= 1:
            return len(panels)
        classes = tie_classes([panel.nll for panel in panels], self.quantum,
                              content=[self.content(panel) for panel in panels])
        return int(np.count_nonzero(classes.classes == 0))


class _Search:
    """One search of a block under one tie convention (+1 forward, -1 reversed: the caller passes
    the samples in reversed canonical order and content keys are complemented)."""

    def __init__(self, likelihood, observed, config, convention: int = 1):
        search = config.search
        self.config = config
        self.search = search
        self.convention = int(convention)
        self.likelihood = likelihood
        self.observed = observed
        self.active = np.ascontiguousarray(observed.any(axis=1))
        self.ceiling = natural_k_ceiling(int(self.active.sum()), likelihood.shape[1])
        self.quanta = BlockQuanta.for_scale(
            block_scale(likelihood, observed, config.model.unknown_copy_penalty))
        self.counter = TieCounter()
        if self.convention > 0:
            self.content = lambda solution: solution.key
        else:
            self.content = lambda solution: solution.key.translate(_COMPLEMENT)
        self.workspace = prepare_workspace(likelihood, observed, config.model, self.quanta)
        self.growth = growth_inputs(likelihood, observed)
        self.residual_inputs = ResidualInputs(likelihood, observed)
        self.cuts = MaximumCuts(search.exact_cut_max_k, search.max_cut_ties)
        self.cavity_workspace = CavityWorkspace(
            likelihood if bool(self.active.all()) else likelihood[self.active], config.cavity)
        width = search.beam_width
        if search.escalation:
            width = max(width, search.escalate_beam_width)
        self.archive = _Archive(width, self.quanta, self.content, self.counter)
        self.beam_width = int(search.beam_width)
        self.replacement_limit = int(search.max_replacement_children)
        self.representatives: dict[int, FixedKSolution] = {}
        self.score_cache: dict = {}
        self.scores: dict = {}
        self.soft_cache: dict = {}
        self.limits: list[str] = []
        self.exact_scores = 0
        self.cache_hits = 0
        self.score_backstop_factor = 1
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
        self.escalated = False
        self.data_starts = 0

    # ---------------------------------------------------------------------------------- scoring
    def refresh(self, preferred_k: int) -> bool:
        """Score changed representatives (nearest K first); True if the backstop omitted any."""
        changed = []
        for k in sorted(self.archive.beams):
            representative = self.archive.beams[k][0]
            previous = self.representatives.get(k)
            if previous is None or previous.key != representative.key:
                changed.append(representative)
        changed.sort(key=lambda solution: (abs(solution.k - preferred_k), solution.k))
        novel = []
        for solution in changed:
            cached = self.score_cache.get(solution.key)
            if cached is None:
                novel.append(solution)
            else:
                self.cache_hits += 1
                self.representatives[solution.k] = solution
                self.scores[solution.k] = cached
        backstop = self.search.exact_score_backstop(self.archive.max_k) * self.score_backstop_factor
        room = max(0, backstop - self.exact_scores)
        retained = novel[:room]
        for score in score_solutions(self.cavity_workspace, retained, self.active):
            self.score_cache[score.solution.key] = score
            self.representatives[score.k] = score.solution
            self.scores[score.k] = score
            if score.mean_field_failures:
                self.mean_field_failed = True
        self.exact_scores += len(retained)
        return len(novel) > room

    def selection(self):
        """(selected PanelScore, its tie classes, the scores in class-input order): score classes
        (kind ``nll``, descending), then smaller K, then the content key."""
        values = list(self.scores.values())
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
        margin = best.log_score - max(others) if others else math.inf
        tied = int(np.count_nonzero(classes.classes == 0)) - 1
        tied += max(0, self.archive.first_class_size(best.k) - 1)
        return best, margin, tied

    # ---------------------------------------------------------------------------------- moves
    def beam(self, k: int):
        return list(self.archive.beams.get(k, ()))[:self.beam_width]

    def soft(self, solution) -> SoftBirths:
        if solution.k >= self.ceiling:
            return SoftBirths((), 0, 0)
        return soft_births(self.workspace, solution, self.likelihood, self.residual_inputs,
                           self.config, self.soft_cache, counter=self.counter)

    def start(self) -> None:
        data_solutions = initial_solutions(self.workspace, self.likelihood, self.active,
                                           self.config, counter=self.counter)
        self.data_starts = len(data_solutions)
        self.archive.add(data_solutions)
        if self.refresh(1):
            self.limits.append("exact_score_backstop")
        if self.mean_field_failed:
            self.limits.append("mean_field_nonconvergence")

    def expand(self, parent, phase: str, before) -> None:
        """One expansion of ``parent`` (see the module docstring)."""
        config = self.config
        search = self.search
        local_limits: list[str] = []
        generated: list = []
        if phase == "ordinary":
            self.ordinary_expanded.add(parent.key)
            moves = ordinary_moves(self.workspace, parent, self.likelihood, self.growth,
                                   self.ceiling, self.cuts, config, counter=self.counter)
            if parent.k > 1 and self.cuts.heuristic(parent.k):
                self.heuristic_cuts = True
            generated.extend(moves.children)
            local_limits.extend(moves.limits)
            self.archive.add(moves.children)
            if self.refresh(parent.k):
                local_limits.append("exact_score_backstop")
            if not self.improved(self.selected(), before):
                children, omitted = replacements(self.workspace, parent, moves.births, config,
                                                 self.replacement_limit)
                generated.extend(children)
                if omitted:
                    local_limits.append("replacement_proposal_budget")
                self.archive.add(children)
                if self.refresh(parent.k):
                    local_limits.append("exact_score_backstop")
            self.ordinary_complete[parent.key] = not local_limits
        else:
            self.residual_expanded.add(parent.key)
            births = self.soft(parent)
            self.residual_calls += 1
            self.residual_rows += births.emitted
            if births.omitted:
                local_limits.append("soft_residual_proposal_budget")
            generated.extend(births.children)
            self.archive.add(births.children)
            # Refresh unconditionally: a retained soft proposal counts even if others were capped.
            if self.refresh(parent.k):
                local_limits.append("exact_score_backstop")
            if not self.improved(self.selected(), before):
                children, omitted = replacements(self.workspace, parent, births.children, config,
                                                 self.replacement_limit)
                generated.extend(children)
                if omitted:
                    local_limits.append("replacement_proposal_budget")
                self.archive.add(children)
                if self.refresh(parent.k):
                    local_limits.append("exact_score_backstop")
            if (not self.improved(self.selected(), before)
                    and parent.k + 2 <= self.ceiling and parent.k + 1 in self.representatives):
                bridge = self.representatives[parent.k + 1]
                bridged = data_births(self.workspace, bridge, self.growth,
                                      config.model.max_iterations, search, counter=self.counter)
                if bridged.tie_cut:
                    local_limits.append("data_birth_tie_budget")
                paired = [child for child in bridged.children if child.k == parent.k + 2]
                bridge_births = self.soft(bridge)
                self.residual_calls += 1
                self.residual_rows += bridge_births.emitted
                if bridge_births.omitted:
                    local_limits.append("soft_residual_proposal_budget")
                paired.extend(child for child in bridge_births.children
                              if child.k == parent.k + 2)
                paired = deduplicate(paired)
                generated.extend(paired)
                self.archive.add(paired)
                if self.refresh(parent.k):
                    local_limits.append("exact_score_backstop")
            self.residual_complete[parent.key] = not local_limits
            after = self.selected()
            if after.solution.key == parent.key and not self.improved(after, before):
                self.no_improvement.add(parent.key)
        self.limits.extend(local_limits)
        if self.mean_field_failed:
            self.limits.append("mean_field_nonconvergence")
        self.generated_offsets.setdefault(parent.key, set()).update(
            child.k - parent.k for child in generated if child.k - parent.k in _MOVE_OFFSETS)
        self.expansions += 1

    def fragile(self) -> bool:
        """The escalation trigger (on the representatives, 2.0's selection): selected K >=
        escalate_min_k, a score margin below escalate_margin, or a selection inside a tie."""
        best, margin, tied = self.margin_and_ties()
        search = self.search
        return bool(best.k >= search.escalate_min_k or margin < search.escalate_margin or tied > 0)

    def escalate(self) -> bool:
        """Widen the search once (beam, replacements) for a fragile block, keeping every fit, score
        and archived panel; the widened beam is expanded again until exhausted or until
        escalate_expansions_percent % of the expansion backstop more expansions."""
        search = self.search
        if not search.escalation or self.escalated or not self.fragile():
            return False
        self.escalated = True
        self.beam_width = max(self.beam_width, int(search.escalate_beam_width))
        self.replacement_limit = max(self.replacement_limit,
                                     int(search.escalate_replacement_children))
        for collection in (self.ordinary_expanded, self.residual_expanded, self.ordinary_complete,
                           self.residual_complete, self.no_improvement):
            collection.clear()
        self.escalation_limit = self.expansions + max(
            1, search.expansion_backstop(self.archive.max_k) * int(search.escalate_expansions_percent)
            // 100)
        return True

    def run(self) -> None:
        self.start()
        search = self.search
        while True:
            if self.escalated:
                if self.expansions >= self.escalation_limit:
                    self.limits.append("escalation_backstop")
                    self.stop_reason = "expansion_backstop"
                    break
            elif self.expansions >= search.expansion_backstop(self.archive.max_k):
                self.limits.append("expansion_backstop")
                self.stop_reason = "expansion_backstop"
                if self.escalate():
                    continue
                break
            before = self.selected()
            candidates = self.beam(before.k)
            parent = next((solution for solution in candidates
                           if solution.key not in self.ordinary_expanded), None)
            phase = "ordinary"
            if parent is None:
                parent = next((solution for solution in candidates
                               if solution.key not in self.residual_expanded), None)
                phase = "residual"
            if parent is None:
                self.stop_reason = ("generated_neighbourhood_local_optimum" if not self.limits
                                    else "incumbent_neighbourhood_search_limited")
                if self.escalate():
                    continue
                break
            self.expand(parent, phase, before)
        if self.mean_field_failed:
            self.limits.append("mean_field_nonconvergence")

    # ---------------------------------------------------------------------------------- pooling
    def archived(self):
        """Every retained archived panel (all K, decision order)."""
        return [panel for k in sorted(self.archive.beams) for panel in self.archive.beams[k]]

    def adopt(self, other: "_Search", index: np.ndarray) -> None:
        """Pool another search's archived panels (per-sample arrays mapped to this search's sample
        order by ``index``) into this archive and re-select: within K the NLL classes decide, across
        K the cavity score (the shared-archive rule). The score backstop doubles."""
        panels = [_sample_subset(panel, index) for panel in other.archived()]
        self.archive.add(panels)
        self.score_backstop_factor = 2
        if self.refresh(self.selected().k):
            self.limits.append("exact_score_backstop")
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
    forward = _Search(likelihood, observed, config, convention=1)
    forward.run()
    searches = [forward]
    mode = config.search.reversed_convention_search
    if mode == "all" or (mode == "fragile" and forward.fragile()):
        reverse = np.arange(likelihood.shape[0] - 1, -1, -1, dtype=np.int64)
        second = _Search(np.ascontiguousarray(likelihood[reverse]),
                         np.ascontiguousarray(observed[reverse]), config, convention=-1)
        second.run()
        forward.adopt(second, reverse)          # reverse is its own inverse
        searches.append(second)
    limits = [name for search in searches for name in search.limits]
    unique_limits = tuple(dict.fromkeys(limits))
    best, classes, values = forward.selection()
    _, margin, tied = forward.margin_and_ties()
    forward.counter.add("discovery.selection", classes)
    certified = forward.certified(best.solution.key, best.k, unique_limits)
    if len(searches) > 1 and not certified:
        certified = searches[1].certified(best.solution.key, best.k, unique_limits)
    counter = TieCounter()
    for search in searches:
        counter.merge(search.counter)
    k_values = np.array(sorted(forward.scores), dtype=np.int16)
    summary = SearchSummary(
        k_values=k_values,
        k_log_scores=np.array([forward.scores[int(k)].log_score for k in k_values],
                              dtype=np.float64),
        natural_k_ceiling=int(forward.ceiling),
        expansions=int(sum(search.expansions for search in searches)),
        data_starts=int(forward.data_starts),
        exact_scores=int(sum(search.exact_scores for search in searches)),
        exact_score_cache_hits=int(sum(search.cache_hits for search in searches)),
        residual_calls=int(sum(search.residual_calls for search in searches)),
        residual_rows=int(sum(search.residual_rows for search in searches)),
        limits=unique_limits,
        heuristic_gauge_cuts=bool(any(search.heuristic_cuts for search in searches)),
        stop_reason=forward.stop_reason, certified=bool(certified),
        tie_quantum=float(forward.quanta.nll), escalated=bool(forward.escalated),
        conventions=len(searches), ambiguous=bool(tied > 0), tied_alternatives=int(tied),
        score_margin=float(margin), tie_events=counter.record())
    return BlockSearch(selected=best.solution, summary=summary)
