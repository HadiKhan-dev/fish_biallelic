"""Certified panel search of one joined batch: same-count moves under the join likelihood,
founder counts by the count rule.

Tier 1 is the bounded sweep: conditional fixed-painting gains and the other proposal generators
(:mod:`.proposals`) rank same-count edits, and the best improving panel among the top
``full_scores_per_kind`` proposals of each kind is accepted after an exact score (a proposal
replacing one path is scored only on the samples its two-region bound leaves open; elsewhere its
value is V(P minus f) exactly). At every tier-1 stall the count step (:mod:`.count_step`, the
pipeline's effective-carrier rule on the total gain and its sequence part) decides one death
(after deleting every path without sequence of its own) or else one birth. Once the count step is
quiescent, tier 2 bounds every same-count move of an explicit K-scaled neighbourhood
(:mod:`.certify`, :mod:`.moves`) and scores the survivors exactly (one-path moves only on the
samples their bound leaves open); an improving move returns the search to tier 1. A same-count
move must leave every changed path a private cell. At a panel certified for both, the
duplicate-seed births run (copies of the main coverers of chimera-trap indicators, refitted
concurrently) and then a pool-saturation step re-runs the endpoint beam with doubled quotas
before the panel is certified (up to 256 paths per endpoint, up to 1024 for a join whose 256 rung
accepted a move). The panel size follows from the count rule; the pool, not a cap, bounds it; a
reported backstop never decides a normal result.

Costs: each sweep O(N m K^2 + K^2 B log K) (candidate scores only where a block-restart bound
leaves a candidate's novelty open); a tier-2 pass O(N m K (|C| + K^2)) (structural moves share
forward prefixes and backward suffixes); a count step O(N m K |C|) plus sparse exact scores.

Determinism: per-sample values are computed by one thread each and summed in sample order;
concurrent refits are evaluated in seed order; caches never change results; so the selected
panel does not depend on thread counts. The Numba team is re-read from the budget at the
boundaries where the search changes phase (each full score or painting chunk, each panel's
forwards, each certify pass); a concurrent refit runs on its share of the budget.

Ties (release 2.1, design 5.4): the panel is kept in canonical order (its paths sorted, tie_rules.sort_panel) after
every change, so every slot tie-break (painting ties to the lower slot, proposal descriptions, the death candidate,
main coverers) is a content rule; the pool comes from the beam in tie-class order. Every choice between candidates
on a continuous value (start panel, tier-1 winner, the top proposals of each kind, tier-2 winner, death candidate,
birth and seed choices) uses anchored tie classes of the join log likelihood (quantum
tie_rules.quantum("join_ll", |LL|)) with the candidate panel's content (its sorted paths) deciding inside a class;
a truncation keeps every proposal of the class it cuts (at most twice the budget).
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field, fields as dataclass_fields, replace
import threading
import re
import time
from typing import Callable, Mapping, Sequence

import numba
import numpy as np

from .. import count_rule
from ...maths import ties
from .. import tie_rules
from . import emissions, objective
from .certify import PanelForwards
from .beam import CartesianBeamFamily, UnresolvedBeam
from .count_step import CountStep, main_coverer
from .moves import (MOVES_PER_CHUNK, REPLACE, SPLICE, Moves, best_improvement, lower_bounds,
                    replaced_row)
from .proposals import (ScoredEdits, boundary_proposals, boundary_vote_matrix, candidate_novelty,
                        conditional_fields, cycle_permutations, fixed_paint_delta,
                        greedy_permutations, local_candidates, prepare_candidate_scores,
                        replacement_candidates, static_replacement_gains)

# Reported backstops, never expected to bind: total sweeps (tier-1 sweeps plus accepted tier-2
# moves) <= base + per_row x max_b K_b, and the pool-saturation ladder. Every join doubles the
# endpoint quota up to POOL_QUOTA_DOUBLINGS times (16 x 2**4 = 256 paths per endpoint); a join
# whose 256 rung accepted a move climbs on, up to POOL_EXTENDED_DOUBLINGS (16 x 2**6 = 1024). A
# panel is certified only when no rung up to its ladder top adds a move; a move accepted at 1024
# is the reported pool backstop.
SWEEP_BACKSTOP_BASE = 100
SWEEP_BACKSTOP_PER_ROW = 20
# Count changes (decisions plus seed swaps) per search before births and seed swaps stop (a reported backstop, never
# expected to bind: normal searches make at most a few count changes; release 2.1 guard against count cycles that
# return through seed swaps, which the born-path freeze cannot see).
COUNT_BACKSTOP_BASE = 50
COUNT_BACKSTOP_PER_ROW = 10
POOL_QUOTA_DOUBLINGS = 4
POOL_EXTENDED_DOUBLINGS = 6
# Bytes per path of the pool's beam lists and move arrays per input block and founder pair (base
# hierarchy._hmm_batch_memory_gb's 200 B q K^2): a batch reserves them up to the ladder's normal top
# (hierarchy.level.batch_memory_bytes); a search whose ladder climbs on reserves the rest itself.
POOL_BYTES_PER_PATH_PAIR = 200


def ladder_extension_bytes(blocks: int, founders: int, paths_per_endpoint: int) -> int:
    """Memory a search reserves when its saturation ladder climbs past the normal top (quota
    paths_per_endpoint x 2**POOL_QUOTA_DOUBLINGS) to the backstop (x 2**POOL_EXTENDED_DOUBLINGS):
    beam lists and move arrays for the extra quota, O(B q K^2)."""
    extra = paths_per_endpoint * (2 ** POOL_EXTENDED_DOUBLINGS - 2 ** POOL_QUOTA_DOUBLINGS)
    return POOL_BYTES_PER_PATH_PAIR * blocks * extra * founders * founders
# Distinct-path sets whose birth bounds are kept for reuse (memory: one samples-long vector per
# pool path and set); raised to the number of sets a round of seed refits meets.
BIRTH_SETS = 4
# Panels whose exclusion forwards are kept (a sweep, its stall, tier 2 and the saturation ladder
# often share one panel) and beam re-runs kept per search.
FORWARDS_KEPT = 3
BEAM_RUNS_KEPT = 4
# The exploratory painting's penalty cap: a lower-penalty painting reveals coordinated apparent
# switches (proposals only; acceptance uses the join score at the join's penalty).
EXPLORATORY_PENALTY = 10.0
# Same-count proposal kinds of a tier-1 sweep, in evaluation order.
SAME_COUNT_KINDS = ('replace', 'static_replace', 'novel_replace', 'local', 'onesided', 'splice',
                    'cycle', 'switch_splice', 'switch_cycle')

__all__ = ["BEAM_RUNS_KEPT", "BIRTH_SETS", "CountCensus", "InitialPanel", "JoinOutcome",
           "JoinSummary", "MOVES_PER_CHUNK", "POOL_EXTENDED_DOUBLINGS", "POOL_QUOTA_DOUBLINGS",
           "PanelSearch", "census_lines", "search_panel", "summarise"]


def _plain(value):
    """A JSON-able copy (dict, list, str, int, float, bool, None) of a search record value."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"search records hold plain values, not {type(value).__name__}")


@dataclass(frozen=True)
class CountCensus:
    """The count step's tallies over one search.

    decisions: count changes decided by the rule (births + deaths); births = pool_births +
    seed_births; seed_swaps: duplicate-seed refits kept at the same size; deaths, of which
    pure_mosaic_deaths deleted a path without a private cell; ambiguous: decided changes whose
    bootstrap band contains zero; births_frozen: a count cycle froze births; start_size: K at
    the start; cycles: deletions of a path an addition of this search created; cycles_undecided:
    such cycles that began with a decided deletion and closed ambiguously; trap_indicators: the
    most chimera-trap indicators of one birth step; seed_refits: duplicate-seed refits run;
    ladder_births: births accepted at a pool-saturation rung; final_paths_failing_death_test:
    final paths that would fail their own death test (None when the search stopped at the sweep
    backstop); seconds, seed_seconds: wall time of the count steps and of their seed refits.
    state_revisits (release 2.1): count steps that would have returned to a panel this search already
    held after a count change (a count cycle through births, deaths and seed swaps; births and seed
    swaps then stop); count_backstop_reached: the search reached COUNT_BACKSTOP count changes and
    stopped births and seed swaps (reported, never expected).
    """

    decisions: int = 0
    births: int = 0
    pool_births: int = 0
    seed_births: int = 0
    seed_swaps: int = 0
    deaths: int = 0
    pure_mosaic_deaths: int = 0
    ambiguous: int = 0
    births_frozen: bool = False
    start_size: int = 0
    cycles: int = 0
    cycles_undecided: int = 0
    trap_indicators: int = 0
    seed_refits: int = 0
    ladder_births: int = 0
    final_paths_failing_death_test: int | None = None
    seconds: float = 0.0
    seed_seconds: float = 0.0
    state_revisits: int = 0
    count_backstop_reached: bool = False


@dataclass(frozen=True)
class InitialPanel:
    """One anchor panel of the search's start: anchor block, its size and log likelihood."""

    anchor: int
    panel_size: int
    likelihood: float


@dataclass(frozen=True)
class JoinSummary:
    """What one joined batch's certified panel search did (a small typed record).

    final_panel_size, final_likelihood: K and the join log likelihood (nats) of the result;
    total_full_scores: full-panel exact scores; initialization: the distinct anchor panels
    scored at the start, selected_initial_anchor the one the search started from; sweeps
    (tier-1 sweeps + accepted tier-2 moves), tier1_sweeps, sweep_backstop, tier2_passes,
    tier2_accepted, saturation_steps, saturation_accepted: work and moves by phase; quota: the
    endpoint quota reached, ladder_top / quota_backstop: the saturation ladder's top and its
    backstop (ladder_extended: the ladder climbed beyond 256); beam_seconds: wall time of beam
    re-runs; count: the count step's tallies; count_exact_scores / count_exact_sample_scores:
    exact scores the count step made; pool_original / pool_final / pool_largest: candidate pool
    sizes; certified: no move improves the panel and the count rule is quiescent at every rung;
    backstop_reached / pool_backstop: the search stopped at a backstop (not certified).
    tie_events: maths.ties.TieCounter record of the join's tie-class decisions (beam cuts and search choices):
    {kind: {decisions, inside_tie, wide, boundary}}.
    events: the search's records in order (tier-1 sweeps, count deaths / seeds / births, tier-2
    passes, pool-saturation rungs), plain JSON dicts as the reports table lists them.
    """

    final_panel_size: int
    final_likelihood: float
    total_full_scores: int
    initialization: tuple[InitialPanel, ...]
    selected_initial_anchor: int
    sweeps: int
    tier1_sweeps: int
    sweep_backstop: int
    tier2_passes: int
    tier2_accepted: int
    saturation_steps: int
    saturation_accepted: int
    quota: int
    ladder_top: int
    quota_backstop: int
    beam_seconds: float
    count: CountCensus
    count_exact_scores: int
    count_exact_sample_scores: int
    pool_original: int
    pool_final: int
    pool_largest: int
    certified: bool
    backstop_reached: bool
    pool_backstop: bool
    events: tuple[dict, ...] = field(default=(), repr=False)
    tie_events: dict = field(default_factory=dict)
    # Count time is included in join_search; counters never decide scientific results.
    phase_seconds: dict = field(default_factory=dict)
    work_counters: dict = field(default_factory=dict)

    @property
    def ladder_extended(self) -> bool:
        return self.ladder_top == self.quota_backstop

    @property
    def ambiguous_count_calls(self) -> int:
        """Count decisions (deaths and births) of this search whose band contains zero."""
        return sum(bool(event.get('ambiguous')) for event in self.events
                   if event.get('stage') in ('count_death', 'count_birth'))

    def final_record(self) -> dict:
        """The search's closing record (the last row of its reports table entries)."""
        return dict(
            total_full_scores=self.total_full_scores, final_panel_size=self.final_panel_size,
            final_likelihood=self.final_likelihood,
            initialization=[dict(anchor=item.anchor, panel_size=item.panel_size,
                                 likelihood=item.likelihood) for item in self.initialization],
            selected_initial_anchor=self.selected_initial_anchor, sweeps=self.sweeps,
            tier1_sweeps=self.tier1_sweeps, sweep_backstop=self.sweep_backstop,
            tier2_passes=self.tier2_passes, tier2_accepted=self.tier2_accepted,
            saturation_steps=self.saturation_steps, saturation_accepted=self.saturation_accepted,
            quota=self.quota, ladder_top=self.ladder_top, ladder_extended=self.ladder_extended,
            quota_backstop=self.quota_backstop, beam_seconds=self.beam_seconds,
            count={item.name: getattr(self.count, item.name)
                   for item in dataclass_fields(CountCensus)},
            count_exact_scores=self.count_exact_scores,
            count_exact_sample_scores=self.count_exact_sample_scores,
            pool_sizes=dict(original=self.pool_original, final=self.pool_final,
                            largest=self.pool_largest),
            certified=self.certified, backstop_reached=self.backstop_reached,
            pool_backstop=self.pool_backstop, tie_events=_plain(self.tie_events),
            phase_seconds=_plain(self.phase_seconds), work_counters=_plain(self.work_counters))

    def records(self) -> list[dict]:
        """Every record of the search in order, the closing record last (one reports-table row
        each); plain dicts (copies)."""
        return [_plain(event) for event in self.events] + [self.final_record()]

    def record(self) -> dict:
        """The summary as one plain JSON dict (the closing record plus ``events``)."""
        return dict(self.final_record(), events=[_plain(event) for event in self.events])

    @classmethod
    def from_record(cls, record: Mapping) -> "JoinSummary":
        """The summary :meth:`record` encoded."""
        pools = record["pool_sizes"]
        return cls(
            final_panel_size=int(record["final_panel_size"]),
            final_likelihood=float(record["final_likelihood"]),
            total_full_scores=int(record["total_full_scores"]),
            initialization=tuple(InitialPanel(int(item["anchor"]), int(item["panel_size"]),
                                              float(item["likelihood"]))
                                 for item in record["initialization"]),
            selected_initial_anchor=int(record["selected_initial_anchor"]),
            sweeps=int(record["sweeps"]), tier1_sweeps=int(record["tier1_sweeps"]),
            sweep_backstop=int(record["sweep_backstop"]), tier2_passes=int(record["tier2_passes"]),
            tier2_accepted=int(record["tier2_accepted"]),
            saturation_steps=int(record["saturation_steps"]),
            saturation_accepted=int(record["saturation_accepted"]), quota=int(record["quota"]),
            ladder_top=int(record["ladder_top"]), quota_backstop=int(record["quota_backstop"]),
            beam_seconds=float(record["beam_seconds"]),
            count=CountCensus(**{item.name: record["count"][item.name]
                                 for item in dataclass_fields(CountCensus) if item.name in record["count"]}),
            count_exact_scores=int(record["count_exact_scores"]),
            count_exact_sample_scores=int(record["count_exact_sample_scores"]),
            pool_original=int(pools["original"]), pool_final=int(pools["final"]),
            pool_largest=int(pools["largest"]), certified=bool(record["certified"]),
            backstop_reached=bool(record["backstop_reached"]),
            pool_backstop=bool(record["pool_backstop"]),
            events=tuple(_plain(event) for event in record.get("events", ())),
            tie_events=_plain(record.get("tie_events", {})),
            phase_seconds=_plain(record.get("phase_seconds", {})),
            work_counters=_plain(record.get("work_counters", {})))


@dataclass(frozen=True)
class JoinOutcome:
    """The selected panel of one join.

    paths: int32[K_out, n]; output founder f copies founder (local row) paths[f, j] of input
        block j, in canonical order (the hierarchy publishes the prepared-row form in canonical order).
    log_likelihood: the join objective of the panel (sum over the batch's samples of V_s, nats).
    summary: what the search did.
    """

    paths: np.ndarray
    log_likelihood: float
    summary: JoinSummary
    release_alternatives: tuple = ()
    minimum_switches: int | None = None


def summarise(summaries: Sequence[JoinSummary]) -> dict:
    """Totals over a level's join searches (the census log lines' counts)."""
    summary = dict(searched=0, certified=0, changed=0, tier2_accepted=0,
                   saturation_accepted=0, tier2_passes=0, backstop_reached=0,
                   pool_backstop=0, extended_ladders=0, largest_pool=0, births=0, deaths=0,
                   pure_mosaic_deaths=0, ambiguous=0, frozen=0, cycles_undecided=0,
                   most_decisions=0, start_rows=0, final_rows=0, seed_births=0, seed_swaps=0,
                   seed_refits=0, ladder_births=0, trap_blocks=0, failing_final=0, state_revisits=0,
                   count_backstops=0)
    for item in summaries:
        count = item.count
        summary['searched'] += 1
        summary['certified'] += bool(item.certified)
        summary['changed'] += item.tier2_accepted + item.saturation_accepted > 0
        summary['tier2_accepted'] += item.tier2_accepted
        summary['saturation_accepted'] += item.saturation_accepted
        summary['tier2_passes'] += item.tier2_passes
        summary['backstop_reached'] += bool(item.backstop_reached)
        summary['pool_backstop'] += bool(item.pool_backstop)
        summary['extended_ladders'] += bool(item.ladder_extended)
        summary['largest_pool'] = max(summary['largest_pool'], item.pool_largest)
        summary['births'] += count.births
        summary['deaths'] += count.deaths
        summary['pure_mosaic_deaths'] += count.pure_mosaic_deaths
        summary['ambiguous'] += count.ambiguous
        summary['frozen'] += bool(count.births_frozen)
        summary['cycles_undecided'] += count.cycles_undecided
        summary['most_decisions'] = max(summary['most_decisions'], count.decisions)
        summary['start_rows'] += count.start_size
        summary['final_rows'] += item.final_panel_size
        summary['seed_births'] += count.seed_births
        summary['seed_swaps'] += count.seed_swaps
        summary['seed_refits'] += count.seed_refits
        summary['ladder_births'] += count.ladder_births
        summary['trap_blocks'] += count.trap_indicators > 0
        summary['failing_final'] += count.final_paths_failing_death_test or 0
        summary['state_revisits'] += count.state_revisits
        summary['count_backstops'] += bool(count.count_backstop_reached)
    return summary


def census_lines(summaries: Sequence[JoinSummary]) -> list[str]:
    """The level's two census log lines (panel search; join count rule), or [] when no batch was
    searched."""
    summary = summarise(summaries)
    if not summary['searched']:
        return []
    return [
        "  Panel search: {searched} searched, {certified} certified, {changed} changed "
        "by tier 2 or pool saturation ({tier2_accepted} tier-2 and {saturation_accepted} "
        "saturation moves), {tier2_passes} tier-2 passes, sweep backstop hits "
        "{backstop_reached}, saturation ladders extended beyond 256 {extended_ladders}, "
        "pool backstop hits {pool_backstop}, largest pool {largest_pool}".format(**summary),
        "  Join count rule: rows {start_rows} at the start -> {final_rows}; {births} births, "
        "{deaths} deaths ({pure_mosaic_deaths} without a private cell), {ambiguous} "
        "ambiguous calls, births frozen by a count cycle in {frozen} super-blocks "
        "({cycles_undecided} undecided cycles), at most {most_decisions} decisions in one "
        "super-block; duplicate seeds: {seed_refits} refits, {seed_births} births, {seed_swaps} "
        "swaps; {ladder_births} saturation births; {trap_blocks} super-blocks with chimera-trap "
        "indicators; {failing_final} final paths fail their own death test; {state_revisits} count-state "
        "revisits, {count_backstops} count backstops".format(**summary),
    ]


def _materialize(description, selected, candidates):
    """Panel after a same-count move (paths keep their slots)."""
    kind, *args = description
    result = [list(path) for path in selected]
    if kind == 'replace':
        founder, candidate = args
        result[founder] = list(candidates[candidate])
    elif kind == 'local':
        founder, block, hap = args
        result[founder][block] = hap
    elif kind == 'onesided':
        first, second, boundary = args
        result[first][boundary:] = selected[second][boundary:]
    elif kind == 'splice':
        first, second, boundary = args
        result[first][boundary:], result[second][boundary:] = (result[second][boundary:],
                                                               result[first][boundary:])
    elif kind == 'cycle':
        boundary, permutation = args
        for first, second in enumerate(permutation):
            result[first][boundary:] = selected[second][boundary:]
    else:
        raise ValueError(kind)
    return [tuple(path) for path in result]


def _admissible(old, new, geometry, near_identity, exempt=frozenset()):
    """A same-count move keeps k distinct paths, and every changed path owns sequence.

    A changed path must keep a private cell and stay farther than the near-identity tolerance
    from every other path; otherwise it would be a pure mosaic or a near-duplicate (q = 0), which
    the count rule deletes, so the move would turn a justified path into a deletion the count
    rule never decided. ``exempt`` slots (a seed and its copy during their refit) may duplicate
    each other or lack novelty; their panel is settled by the count rule afterwards.
    """
    if len(new) != len(old):
        return False
    checked = [i for i, (a, b) in enumerate(zip(old, new)) if a != b and i not in exempt]
    if not exempt and len(set(new)) != len(new):
        return False
    if not checked:
        return True
    private = count_rule.private_mask(np.asarray(new, dtype=np.int64))
    for i in checked:
        if not private[i].any():
            return False
        others = [path for j, path in enumerate(new) if j != i]
        if new[i] in others or geometry.near_identical(new[i], others, near_identity):
            return False
    return True


def _improves(new_ll, old_ll) -> bool:
    """A log-likelihood rise beyond the float resolution of the join score -2 LL: t = max(1e-8,
    1e-9 (1 + 2|LL|)), as in moves.best_improvement."""
    return 2.0 * (new_ll - old_ll) > max(1e-8, 1e-9 * (1.0 + 2.0 * abs(old_ll)))


def _improvement_distance(new_ll, old_ll) -> float:
    """Signed distance (nats of LL) of a rise to the improvement threshold of :func:`_improves`."""
    return (new_ll - old_ll) - 0.5 * max(1e-8, 1e-9 * (1.0 + 2.0 * abs(old_ll)))


def _paths(beam) -> list[tuple[int, ...]]:
    """Distinct beam paths (tuples of Python ints) in beam order."""
    return list(dict.fromkeys(tuple(int(row) for row in path) for path, _ in beam))


class PanelSearch:
    """One join's certified panel search (see the module docstring); :meth:`run` executes it.

    The search state: ``selected`` (the panel, a list of path tuples), its per-sample scores
    ``baseline_scores`` and log likelihood ``likelihood``; the candidate pool (``candidates``,
    with its array, sorted workspace and capacity); the paths the search's births added
    (``born``, followed through same-count moves); the panel sizes after every count change
    (``sizes``, with the changes that deleted a path without sequence of its own in
    ``cleanup_steps``) and the count tallies; the caches (exclusion forwards of the last panels,
    birth bounds by path set, beam re-runs), which never change results.
    """

    def __init__(self, inputs, beam, beam_runner, *, config, count_rule_config, cost_scale, budget,
                 label=None, counter=None, linkage=None, choice_plan=None, shared_layout=None):
        self.choice_plan = choice_plan
        self.config = config
        # The batch linker's path score (join/beam.path_linkage; release 2.2): the data criterion inside
        # log-likelihood tie classes (None: content only, 2.1).
        self.linkage = linkage
        self.budget = budget
        # The batch's census label (None: no valley census) and its tie-class decision counter.
        self.label = label
        self.tie_counter = counter if counter is not None else tie_rules.SharedCounter()
        self._tie_lock = threading.Lock()      # concurrent seed refits record decisions too
        self._sweeps_run = 0
        self._certify_runs = 0
        self._count_steps = 0
        self.beam_runner = beam_runner
        self._local = threading.local()
        self.blocks = len(inputs.blocks)
        self.samples = int(inputs.sample_count)
        self.initial_family = beam if isinstance(beam, CartesianBeamFamily) else None
        self.initial_unresolved = beam if isinstance(beam, UnresolvedBeam) else None
        self.beam_stopped = self.initial_family is not None or self.initial_unresolved is not None
        if self.initial_family is not None:
            # Exactly the old first-covering anchor trials, without enumerating the
            # other Cartesian members. These seed paths are not a truncated pool.
            candidates = list(dict.fromkeys(beam.first_covering(block, row)
                              for block, count in enumerate(beam.founders) for row in range(count)))
        else:
            candidates = _paths(beam.incumbent_paths if self.initial_unresolved is not None else beam)
        if not candidates:
            raise ValueError("a join needs at least one beam path")
        site_counts = [len(block.positions) for block in inputs.blocks]
        self.penalty = emissions.switch_penalty(site_counts)
        self.sites_per_bin = emissions.sites_per_bin(site_counts, config.max_bins)
        self.cost = count_rule.founder_cost(site_counts, self.samples, cost_scale)
        self.layout = (emissions.binned_emissions(inputs, self.sites_per_bin, budget)
                       if shared_layout is None else shared_layout)
        self.total_bins = self.layout.total_bins
        self.block_offsets = np.ascontiguousarray(self.layout.offsets[:-1], dtype=np.int64)
        self.block_sizes = [int(k) for k in self.layout.founders]
        self.geometry = count_rule.RowGeometry.from_panels(inputs.blocks)
        self.near_identity = float(count_rule_config.near_identity)
        self.seed_refits = int(count_rule_config.ranked_refits)
        self.release_families = []
        self._plateau_candidates = {}
        self._plateau_certified = set()
        self._terminal_choices = 0
        self._seed_families = {}
        self._switch_counts = {}
        self.counter = CountStep(layout=self.layout, penalty=self.penalty, cost=self.cost,
                                 samples=self.samples, geometry=self.geometry,
                                 config=count_rule_config, score=self._score,
                                 tie_counter=self.tie_counter, label=label, linkage=linkage,
                                 choose_panel=self._prefer_panels,
                                 record_count_family=self._record_count_family)
        self.evaluations = 0
        self._evaluations_lock = threading.Lock()
        self.original_pool = candidates
        self.original_set = set(candidates)
        self.candidates = self.candidate_array = self.candidate_workspace = None
        self.capacity = 0
        self.beam_cache = {}
        self.beam_seconds = 0.0
        self.forwards_cache = {}
        self.births_by_set = {}
        self.birth_sets = BIRTH_SETS
        self._cache_lock = threading.Lock()
        self.sweep_forwards = None   # (panel, exclusion forwards, removal scores) of the last sweep
        self.born = set()
        self.births_frozen = False
        self.sweep_backstop = SWEEP_BACKSTOP_BASE + SWEEP_BACKSTOP_PER_ROW * max(self.block_sizes)
        self.count_backstop = COUNT_BACKSTOP_BASE + COUNT_BACKSTOP_PER_ROW * max(self.block_sizes)
        # Canonical panel states held after the start and after every count change (count-cycle guard).
        self.count_states = set()
        self.events = []
        self.count = dict(decisions=0, births=0, pool_births=0, seed_births=0, seed_swaps=0,
                          deaths=0, pure_mosaic_deaths=0, ambiguous=0, births_frozen=False,
                          start_size=0, cycles=0, cycles_undecided=0, trap_indicators=0,
                          seed_refits=0, ladder_births=0, final_paths_failing_death_test=None,
                          seconds=0.0, seed_seconds=0.0, state_revisits=0, count_backstop_reached=False)
        # The start: the anchor-covering panel of largest likelihood (tie classes of the join log
        # likelihood, the panel's content inside a class). It only seeds the search; the count rule
        # decides the panel size.
        self.initialization = []
        self.selected = None
        initial_panels = set()
        trials = []
        for anchor in range(self.blocks):
            seen = set()
            trial = []
            anchor_candidates = (candidates if self.initial_family is None else
                                 [self.initial_family.first_covering(anchor, row)
                                  for row in (range(self.initial_family.founders[anchor] - 1, -1, -1)
                                              if self.initial_family.reverse else
                                              range(self.initial_family.founders[anchor]))])
            for path in anchor_candidates:
                if path[anchor] not in seen:
                    trial.append(path)
                    seen.add(path[anchor])
            signature = tie_rules.panel_key(trial)
            if signature in initial_panels:
                continue
            initial_panels.add(signature)
            trial_scores = self._score(trial)
            trial_ll = float(trial_scores.sum())
            self.initialization.append(InitialPanel(anchor, len(trial), trial_ll))
            trials.append((trial_ll, signature, anchor, trial_scores))
        best = self._class_winner('assembly.join_start', trials, lambda item: item[0], lambda item: item[1],
                                  census_key='start', quantum_kind='join_ll_wide',
                                  link=lambda item: self._panel_link(item[1]), panel=lambda item: item[1])
        self._set_panel(best[1], best[3], best[0])
        self.selected_anchor = best[2]
        self.count_states.add(tie_rules.panel_key(self.selected))
        self.start_size = len(self.selected)
        self.count['start_size'] = self.start_size
        # Panel size at the start and after every count change, and the changes that deleted a
        # path without sequence of its own (count_rule.pre_cycle_count).
        self.sizes = [self.start_size]
        self.cleanup_steps = set()
        self.static_totals = [self.layout.block(j).sum(axis=2) for j in range(self.blocks)]
        self._set_pool(self.original_pool)
        if self.beam_stopped:
            self._record_beam_bound(beam, 'initial')

    def _record_beam_bound(self, beam, source, bound=None):
        """Preserve the complete unevaluated path domain; keep the current incumbent."""
        domain = (dict(kind='constant_cartesian', founders=list(beam.founders),
                       cardinality=beam.cardinality, score=beam.score, reverse=beam.reverse,
                       bound=(2 * self.config.paths_per_endpoint * max(self.block_sizes)
                              if bound is None else bound))
                  if isinstance(beam, CartesianBeamFamily) else beam.record())
        self.beam_stopped = True
        record = dict(reason='beam_materialization_bound',
                      source_decision=f'beam/{source}', status='unresolved_beam_materialization',
                      unresolved_search=True, panels=[], beam_domain=domain,
                      representative_prior='retain_incumbent',
                      incumbent_panel=tie_rules.panel_key(self.selected))
        self.release_families.append(record)
        self.events.append(dict(stage='beam_materialization_bound', source=source,
                                beam_domain=domain, incumbent_founders=len(self.selected)))

    # -- tie classes -----------------------------------------------------------------------------

    def _quantum(self, scale=None) -> float:
        """The join's tie quantum at the scale of ``scale`` (default: the current log likelihood)."""
        return tie_rules.quantum('join_ll', self.likelihood if scale is None else scale)

    def _census(self, kind, key, values):
        if self.label is not None and ties.census_enabled():
            with self._tie_lock:
                ties.census(kind, f"{self.label}/{key}", values)

    def _tie_add(self, kind, result, decided_inside=None):
        with self._tie_lock:
            self.tie_counter.add(kind, result, decided_inside=decided_inside)

    def _panel_link(self, panel):
        """The linker score of a panel (sum over its paths), or None without a linkage function."""
        if self.linkage is None:
            return None
        return float(sum(self.linkage(path) for path in panel))

    def _class_order(self, kind, items, value, content, *, census_key=None, scale=None,
                     quantum_kind='join_ll', link=None, panel=None):
        """``items`` in tie-class order of value(item), descending (log likelihoods and gains: higher is
        better), content(item) deciding inside a class; returns (ordered items, TieClasses). The quantum
        is quantum_kind's at ``scale`` (join_ll: default the current log likelihood; join_proxy: default
        the largest |value|)."""
        items = list(items)
        values = [float(value(item)) for item in items]
        if scale is None and quantum_kind == 'join_ll':
            scale = getattr(self, 'likelihood', None)
        if scale is None:
            finite = [abs(v) for v in values if np.isfinite(v)]
            scale = max(finite) if finite else 0.0
        quantum = tie_rules.quantum(quantum_kind, scale)
        motion = tie_rules.motion(quantum_kind, scale)
        contents = [content(item) for item in items]
        result = ties.tie_classes(values, quantum, content=contents, descending=True)
        tie_rules.add_decision(self.tie_counter, kind, result, values, quantum, motion, descending=True)
        if census_key is not None:
            # Census values: smaller is better (negated log likelihoods), the decision's candidates.
            self._census(kind, census_key, [-v for v in values])
        links = None
        if link is not None and self.linkage is not None and result.tied_winner:
            links = [link(item) for item in items]
        if result.tied_winner:
            # Fully symmetric for the data (log likelihood and linker score tied): decided by content, counted,
            # and the tied alternatives released in the search events.
            symmetric = tie_rules.symmetric_members(result, links)
            with self._tie_lock:
                self.tie_counter.note(kind + ".symmetric", inside_tie=bool(symmetric))
                if symmetric and panel is None:
                    self.events.append(dict(stage='symmetric_tie', kind=kind,
                                            alternatives=sorted(contents[m] for m in symmetric)))
        order = tie_rules.linkage_order(result, links, contents)
        symmetric = tie_rules.symmetric_members(result, links)
        if panel is not None and symmetric:
            winner = self._prefer_panels(kind, [panel(items[i]) for i in symmetric],
                                        [values[i] for i in symmetric], census_key)
            chosen = symmetric[winner]
            order = [chosen] + [i for i in order if i != chosen]
        return [items[i] for i in order], result

    def _class_winner(self, kind, items, value, content, *, census_key=None, scale=None,
                      quantum_kind='join_ll', link=None, panel=None):
        ordered, _ = self._class_order(kind, items, value, content, census_key=census_key, scale=scale,
                                       quantum_kind=quantum_kind, link=link, panel=panel)
        return ordered[0] if ordered else None

    def _prefer_panels(self, kind, panels, scores, source=None):
        """Name a data-tied representative by the stated priors; retain the whole family.

        These are decision-point alternatives. Unrun continuations remain explicit rather
        than being claimed equal to the eventual final objective.
        """
        keys = [tie_rules.panel_key(panel) for panel in panels]
        distinct = sorted(set(keys))
        if len(distinct) == 1:
            return 0
        counts = {}
        smallest = min(map(len, distinct))
        for key in distinct:
            if len(key) != smallest:
                continue
            with self._tie_lock:
                found = self._switch_counts.get(key)
            if found is None:
                self._claim()
                found = int(objective.minimum_switches(*self.layout.arrays(), self._rows(key),
                                                       self.penalty, self.samples).sum())
                with self._tie_lock:
                    self._switch_counts[key] = found
            counts[key] = found
        chosen = min(counts, key=lambda key: (counts[key], key))
        if self.choice_plan is not None:
            chosen = self.choice_plan.choose(kind, source, distinct, chosen)
        family = dict(panels=distinct, reason=kind, source_decision=source,
                      objective='join_ll', values=[float(scores[keys.index(key)]) for key in distinct],
                      representative_prior='smaller_K_then_exact_minimum_switches',
                      minimum_switches=[counts.get(key) for key in distinct],
                      status='unresolved_search', unresolved_search=True)
        with self._tie_lock:
            self.events.append(dict(stage='symmetric_tie', kind=kind, alternatives=distinct,
                                    representative_prior=family['representative_prior']))
            if source is not None and re.search(r'/seed[0-9]+(?:/|$)', source) is not None:
                self._seed_families.setdefault(source.split('/p')[0].split('/pure')[0].split('/death')[0], []).append(family)
            else:
                self.release_families.append(family)
        return keys.index(chosen)

    def _record_count_family(self, family):
        """A count outcome left unresolved by equally near shadow rows."""
        source = family.get('source_decision')
        with self._tie_lock:
            self.events.append(dict(stage='count_shadow_ambiguity', **family))
            if source is not None and re.search(r'/seed[0-9]+(?:/|$)', source) is not None:
                key = source.split('/p')[0].split('/pure')[0].split('/death')[0]
                self._seed_families.setdefault(key, []).append(family)
            else:
                self.release_families.append(family)

    def _set_panel(self, panel, scores, likelihood):
        """The search's panel, in canonical order (paths sorted), with its per-sample scores."""
        self.selected = tie_rules.sort_panel(panel)
        self.baseline_scores = scores
        self.likelihood = float(likelihood)
        # Flat neighbors belong to this exact nominal search state. A subsequent
        # count or same-count move needs its own terminal certification evidence.
        self._plateau_candidates = {}
        # Genuine accepted same-count/count moves end the uninterrupted flat
        # episode. A terminal continuation explicitly restores its visited set.
        self._plateau_certified = set()

    # -- threads -------------------------------------------------------------------------------

    def _budget(self):
        """The calling thread's budget: a concurrent refit's share, else the search's."""
        return getattr(self._local, "budget", None) or self.budget

    def _claim(self) -> int:
        """Boundary: set the calling thread's Numba team from its budget."""
        threads = max(1, int(self._budget().threads()))
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
        return threads

    def _concurrent(self, function, items):
        """[function(item) for item in items] as concurrent in-process tasks of the budget, each
        on its share (re-read at its boundaries); results in item order."""
        def bound(item):
            def run(child):
                previous = getattr(self._local, "budget", None)
                self._local.budget = child
                try:
                    return function(item)
                finally:
                    self._local.budget = previous
            return run
        # A task's transient arrays: one panel's exclusion forwards, one chunk of per-sample
        # replacement bounds and one chunk of structural bounds.
        task_bytes = (8 * self.samples * self.total_bins * (max(self.block_sizes) + 2)
                      + (1 << 26) + 8 * self.samples * MOVES_PER_CHUNK)
        results = [None] * len(items)
        for index, value in self._budget().concurrent([bound(item) for item in items], task_bytes,
                                                      label="join concurrent"):
            results[index] = value
        return results

    # -- objective -----------------------------------------------------------------------------

    def _rows(self, paths) -> np.ndarray:
        """int64[len(paths), n] local rows of a list of paths (beam rows are local rows)."""
        return np.asarray(paths, dtype=np.int64).reshape(-1, self.blocks)

    def _score(self, paths) -> np.ndarray:
        """Exact per-sample join scores of a panel (counted as one full score)."""
        with self._evaluations_lock:
            self.evaluations += 1
        self._claim()
        return objective.panel_scores(self.layout, self._rows(paths), self.penalty)

    def _paint(self, paths, penalty, reverse=None) -> np.ndarray:
        """Viterbi painting int32[S, total bins] of a panel at ``penalty`` (reverse: the traceback tie
        convention; None = maths.ties.reverse_conventions())."""
        self._claim()
        return objective.painting(self.layout, self._rows(paths), penalty,
                                  self.config.tensor_budget_mb, self._claim, reverse=reverse)

    def _exact(self, trial):
        trial_scores = self._score(trial)
        trial_ll = float(trial_scores.sum())
        return -2.0 * trial_ll, (trial_ll, trial_scores)

    # -- pool, beam re-runs and caches -----------------------------------------------------------

    def _set_pool(self, pool):
        """Tier-1 workspaces of the current candidate pool."""
        self.candidates = pool
        self.candidate_array = np.asarray(pool, dtype=np.int64)
        self.candidate_workspace = prepare_candidate_scores(self.layout, self.candidate_array)
        self.capacity = len(pool)

    def _extension_pool(self, quota, panel):
        """C0 plus the beam with base quota and the panel's row multiplicity."""
        counts = {}
        for path in panel:
            for block, row in enumerate(path):
                counts[(block, row)] = counts.get((block, row), 0) + 1
        overrides = {endpoint: quota * count for endpoint, count in counts.items() if count >= 2}
        if quota == self.config.paths_per_endpoint and not overrides:
            return self.original_pool
        key = (quota, tuple(sorted(overrides.items())))
        if key not in self.beam_cache:
            if len(self.beam_cache) >= BEAM_RUNS_KEPT:
                self.beam_cache.clear()
            started = time.perf_counter()
            beam = self.beam_runner(quota, overrides)
            if isinstance(beam, (CartesianBeamFamily, UnresolvedBeam)):
                self._record_beam_bound(beam, f'quota{quota}',
                    max(2 * sum(overrides.get((block, row), quota) for row in range(count))
                        for block, count in enumerate(self.block_sizes)))
                self.beam_seconds += time.perf_counter() - started
                return self.candidates
            paths = dict.fromkeys(_paths(beam))
            self.beam_cache[key] = self.original_pool + [path for path in paths
                                                         if path not in self.original_set]
            self.beam_seconds += time.perf_counter() - started
        return self.beam_cache[key]

    def _panel_forwards(self, panel):
        """Exclusion forwards of a panel (row order matters), kept for the last few panels.

        Birth bounds depend only on the set of distinct panel paths: the forward maxima and the
        maxima over the new path's states are exact over the same values, which duplicate rows
        and row order do not change. Panels with one path set therefore share them (kept for
        the last birth_sets sets), above all the duplicate-seed refits of a count step, whose
        panels start as P plus a copy and mostly continue through the same accepted moves."""
        key = tuple(map(tuple, panel))
        with self._cache_lock:
            found = self.forwards_cache.get(key)
        if found is None:
            self._claim()
            computed = PanelForwards(self.layout, self._rows(panel), self.penalty, self.total_bins)
            with self._cache_lock:
                found = self.forwards_cache.get(key)
                if found is None:
                    found = computed
                    if len(self.forwards_cache) >= FORWARDS_KEPT:
                        self.forwards_cache.pop(next(iter(self.forwards_cache)))
                    self.forwards_cache[key] = found
        path_set = tuple(sorted(set(key)))
        with self._cache_lock:
            shared = self.births_by_set.pop(path_set, None)
            if shared is None:
                shared = found.births
            elif shared is not found.births:
                shared.update(found.births)
                found.births = shared
            self.births_by_set[path_set] = shared
            while len(self.births_by_set) > self.birth_sets:
                self.births_by_set.pop(next(iter(self.births_by_set)))
        return found

    # -- tier 2 and the saturation rungs ----------------------------------------------------------

    def _certify(self, *args, **kwargs):
        # A lazily built splice table belongs to this pass, including seed refits.
        # Reserve its bytes until the pass ends; return only copied sample vectors.
        with ExitStack() as held:
            return self._certify_moves(*args, **kwargs, held=held)

    def _certify_moves(self, panel, pool, pool_offset, structural, pool_capacity, panel_ll=None,
                       slots=None, refit_scores=None, key=None, *, held):
        """Bound every same-count move of N(P, pool) (or only replacements by pool paths).

        slots: a seed refit; only moves changing these rows, which are exempt from the
        novelty rule of same-count moves. refit_scores: the exact one-path scores shared by
        the seed refits of one count step. key: the pass's census key. The best move is chosen
        by tie classes of its exact change (the moved panel's content inside a class). Returns
        (best move or None, record, forwards)."""
        started = time.perf_counter()
        panel_ll = self.likelihood if panel_ll is None else panel_ll
        rows = np.asarray(panel, dtype=np.int64)
        moves = Moves(rows, pool_capacity)
        added = moves.add_pool(np.asarray(pool, dtype=np.int64).reshape(-1, self.blocks),
                               pool_offset)
        if structural:
            moves.add_structural(self.block_sizes, slots)
        moves.finish()
        forwards = self._panel_forwards(panel)
        self._claim()
        bounds, sample_bounds = lower_bounds(moves, forwards, panel_ll, self._rows,
                                             self.block_offsets, replaced=slots)
        # A seed refit moves the seed and its copy freely (no novelty rule for any path while the
        # copies separate); its panel is settled by the count rule afterwards.
        exempt = frozenset() if slots is None else frozenset(range(len(panel)))
        if slots is not None:
            replacing = moves.kind == REPLACE
            bounds[replacing & ~np.isin(moves.first_removed, list(slots))] = np.inf
        combined = self.candidates if not pool_offset else self.candidates + list(pool)
        cache = {}
        sparse = dict(scores=0, samples=0)
        splice_work = dict(tables=0, scores=0, seconds=0.0)

        scored_changes = []

        def exact_move(trial, move):
            result = _exact_move(trial, move)
            if key is not None:
                scored_changes.append(result[1][0] - panel_ll)
            return result

        def _exact_move(trial, move):
            if moves.kind[move] == SPLICE:
                from ...execution import reserve_memory
                from .splice import pair_column, splice_scores, workspace_bytes
                if "splice" not in cache:
                    threads = self._claim()
                    held.enter_context(reserve_memory(
                        workspace_bytes(self.samples, self.blocks, len(panel),
                                        max(self.layout.bins), threads),
                        label="join suffix-exchange scores"))
                    table_started = time.perf_counter()
                    cache["splice"] = splice_scores(self.layout, rows, self.penalty)
                    splice_work["seconds"] += time.perf_counter() - table_started
                    splice_work["tables"] += 1
                _, first, second, boundary, _ = moves.canonical[move]
                column = pair_column(len(panel), int(first), int(second))
                values = cache["splice"][:, int(boundary) - 1, column].copy()
                splice_work["scores"] += 1
                trial_ll = float(values.sum())
                return -2.0 * trial_ll, (trial_ll, values, 'splice')
            # A survivor replacing one path is scored exactly per sample where its bound exceeds
            # V_s(P minus f) (PanelForwards.replacement_scores); other moves by the full score.
            from collections import Counter
            removed = list((Counter(panel) - Counter(trial)).elements())
            added = list((Counter(trial) - Counter(panel)).elements())
            canonical = tie_rules.sort_panel(trial)
            if len(removed) != 1 or len(added) != 1:
                return self._exact(canonical)
            row = panel.index(removed[0])
            shared = None
            if refit_scores is not None:
                shared = (tuple(sorted(path for i, path in enumerate(panel) if i != row)), added[0])
                found = refit_scores.get(shared)
                if found is not None:
                    return found
            if 'removal' not in cache:
                cache['removal'] = forwards.removal_scores()
            self._claim()
            # The bound's original slot is useful only for the same unique replacement.
            bound = sample_bounds.get(move) if replaced_row(moves, move) == row else None
            values, scored_samples = forwards.replacement_scores(
                row, self._rows(added)[0], self._rows(canonical), cache['removal'], bound)
            sparse['scores'] += 1
            sparse['samples'] += scored_samples
            trial_ll = float(values.sum())
            result = -2.0 * trial_ll, (trial_ll, values)
            if shared is not None:
                refit_scores[shared] = result
            return result

        q_ll = self._quantum(panel_ll)
        flat = {}

        def observe(trial, exact_score, payload):
            trial_ll = -0.5 * exact_score
            if abs(trial_ll - panel_ll) <= ties.WIDE_CLASS_QUANTA * q_ll:
                flat[tie_rules.panel_key(trial)] = (float(trial_ll), key)

        def choose(entries):
            # Changes are in units of the join score -2 LL: the class value is the LL gain.
            return self._class_winner('assembly.join_tier2', entries, lambda entry: -0.5 * entry[0],
                                      lambda entry: tie_rules.panel_key(entry[3]),
                                      census_key=key, scale=panel_ll,
                                      link=lambda entry: self._panel_link(entry[3]), panel=lambda entry: entry[3])

        best_move, survivors, exact_scores = best_improvement(
            bounds, moves.canonical,
            lambda move: _materialize(moves.description(move), panel, combined),
            exact_move, -2.0 * panel_ll, signature=lambda trial: tuple(sorted(trial)),
            admissible=lambda trial: _admissible(panel, trial, self.geometry, self.near_identity,
                                                 exempt),
            reach=2.0 * ties.WIDE_CLASS_QUANTA * q_ll, choose=choose,
            observe=observe if slots is None else None)
        if (best_move is not None and len(best_move[4]) == 3
                and best_move[4][2] == 'splice'):
            # Prefix/suffix scores rank all exchanges cheaply. Count support needs a
            # consistent forward-scored baseline: tiny cancellation noise at zero
            # otherwise changes bootstrap effective-carrier ratios. Score only this
            # selected panel before any continuation, including duplicate-seed fits.
            # The local payload tag survives a duplicate move replacing its witness.
            score, payload = self._exact(tie_rules.sort_panel(best_move[3]))
            best_move = (score + 2.0 * panel_ll, *best_move[1:4], payload)
        if best_move is None and slots is None:
            # Keep only known flat neighbors. Seed-local proposals are not final
            # panels, and bounds are deliberately not widened for this fallback.
            self._plateau_candidates.update(flat)
        if scored_changes:
            self._census('assembly.join_improve', f"{key}/t2",
                         [_improvement_distance(panel_ll + gain, panel_ll) for gain in scored_changes])
        record = dict(panel_size=len(panel), pool=len(combined), pool_paths_tested=added,
                      neighbourhood=moves.counts(), moves=len(moves), survivors=int(survivors),
                      exact_scores=int(exact_scores), sparse_exact_scores=sparse['scores'],
                      sparse_exact_samples=sparse['samples'],
                      splice_exact_scores=splice_work["scores"],
                      splice_tables=splice_work["tables"], splice_seconds=splice_work["seconds"],
                      observed_plateau_neighbours=len(flat) if best_move is None else 0,
                      recomputed_exclusions_per_sample=float(forwards.recomputed_per_sample),
                      accepted=None if best_move is None else str(moves.description(best_move[2])),
                      accepted_likelihood_change=(None if best_move is None
                                                  else -0.5 * float(best_move[0])),
                      seconds=time.perf_counter() - started)
        return best_move, record, forwards

    # -- tier 1 ----------------------------------------------------------------------------------

    def _apply_same_count(self, new_panel, new_scores):
        # This is a cycle guard, not a claim about founder ancestry. Retain tags on
        # unchanged paths; if a tagged path is replaced in a multi-path edit, every new
        # path is a possible descendant. Slot correspondence is not identifiable from
        # the target panel and must not make the guard depend on its edit witness.
        old, new = set(self.selected), set(new_panel)
        inherited = bool((old - new) & self.born)
        self.born.intersection_update(new)
        if inherited:
            self.born.update(new - old)
        self._set_panel(new_panel, new_scores, float(new_scores.sum()))

    def _proposal_scores(self, panel, trial):
        """Exact per-sample scores of a tier-1 proposal: one that replaces one path is scored per
        sample where its two-region bound exceeds V_s(P minus f) (sparse; returns (scores, True));
        others in full (returns (scores, False))."""
        # A one-path replacement is identified by multiset difference, independent of slots.
        # A suffix exchange can have the same target as a local edit after row permutation.
        from collections import Counter
        removed = list((Counter(panel) - Counter(trial)).elements())
        added = list((Counter(trial) - Counter(panel)).elements())
        if len(removed) != 1 or len(added) != 1:
            return self._score(tie_rules.sort_panel(trial)), False
        founder = panel.index(removed[0])
        if self.sweep_forwards is None or self.sweep_forwards[0] != tuple(panel):
            forwards = self._panel_forwards(panel)
            self.sweep_forwards = (tuple(panel), forwards, forwards.removal_scores())
        self._claim()
        values, _ = self.sweep_forwards[1].replacement_scores(
            founder, self._rows(added)[0], self._rows(tie_rules.sort_panel(trial)), self.sweep_forwards[2])
        return values, True

    def _painting_proposals(self, panel, local, eligible, wide_quota, convention):
        """The proposals of the painting-dependent generators (replace, local, one-sided, splices,
        switch splices, cycles) from the panel's Viterbi paintings with the traceback tie convention
        ``convention`` (release 2.2: _tier1_sweep merges both conventions). Returns (proposals by kind,
        local edits screened)."""
        k = len(panel)
        blocks = self.blocks
        offsets = self.layout.offsets
        candidates = self.candidates
        candidate_array = self.candidate_array
        proposals = {kind: [] for kind in ('replace', 'local', 'onesided', 'splice', 'switch_splice',
                                           'switch_cycle', 'cycle')}
        painted = self._paint(panel, self.penalty, reverse=convention)
        fields = []
        corrections = []
        for b in range(blocks):
            painting = np.ascontiguousarray(painted[:, offsets[b]:offsets[b + 1]])
            unary, pair = conditional_fields(self.layout.block(b), painting, local[:, b], k,
                                              int(self.layout.founders[b]))
            fields.append(unary)
            corrections.append(pair)
        gains = np.zeros((k, len(candidates)))
        for b in range(blocks):
            gains += fields[b][:, candidate_array[:, b]]
        proposals['replace'] = replacement_candidates(gains, eligible)
        proposals['local'] = local_candidates(fields, local)
        local_screened = len(proposals['local'])
        exploratory = self._paint(panel, min(self.penalty, EXPLORATORY_PENALTY), reverse=convention)
        preference = np.zeros((k, k))
        pair_gain = np.zeros((k, k))
        for boundary in range(blocks - 1, 0, -1):
            preference += fields[boundary][:, local[:, boundary]]
            pair_gain += corrections[boundary]
            proposals['onesided'].extend(boundary_proposals(preference, boundary, None,
                                                            'onesided'))
            delta = preference + preference.T + pair_gain
            proposals['splice'].extend(boundary_proposals(delta, boundary, None, 'splice'))
            votes = boundary_vote_matrix(exploratory, int(offsets[boundary]), k)
            switch_gain = votes + votes.T - np.diag(votes)[:, None] - np.diag(votes)[None, :]
            proposals['switch_splice'].extend(boundary_proposals(switch_gain, boundary, None,
                                                                 'splice'))
            # Both tie conventions of the greedy and cycle permutations (release 2.2: an order-invariant union).
            for permutation in greedy_permutations(votes):
                if np.any(permutation != np.arange(k)):
                    gain = float(votes[np.arange(k), permutation].sum() - np.trace(votes))
                    proposals['switch_cycle'].append(
                        (gain, ('cycle', boundary, tuple(map(int, permutation)))))
            for permutation in cycle_permutations(preference):
                if np.any(permutation != np.arange(k)):
                    new = local.copy()
                    new[:, boundary:] = local[permutation, boundary:]
                    gain = 0.
                    for b in range(boundary, blocks):
                        gain += fixed_paint_delta(self.layout.block(b), painted[:, offsets[b]:offsets[b + 1]],
                                                  local[:, b], new[:, b], k)
                    proposals['cycle'].append(
                        (float(gain), ('cycle', boundary, tuple(map(int, permutation)))))
        del candidates
        return proposals, local_screened

    def _tier1_sweep(self, panel, panel_scores, panel_ll):
        """One bounded sweep at ``panel``: the best improving admissible move among the top
        proposals of every same-count kind. Returns (best or None, record)."""
        k = len(panel)
        blocks = self.blocks
        config = self.config
        quota = config.full_scores_per_kind
        started = time.perf_counter()
        sparse_scores = 0
        candidates = self.candidates
        candidate_array = self.candidate_array
        offsets = self.layout.offsets
        local = np.asarray(panel, dtype=np.int64)
        panel_set = set(panel)
        proposals = {kind: [] for kind in SAME_COUNT_KINDS}
        eligible = np.asarray([c for c, path in enumerate(candidates) if path not in panel_set],
                              dtype=np.int64)
        proposal_counts = {}
        eligible_mask = np.zeros(len(candidates), dtype=bool)
        eligible_mask[eligible] = True
        novelty, novelty_scored = candidate_novelty(local, candidate_array, self.penalty,
                                                    panel_scores, eligible_mask,
                                                    workspace=self.candidate_workspace)
        self._sweeps_run += 1
        sweep_key = f"s{self._sweeps_run}"
        # Novel candidates: novelty beyond the band at its exact zero (a rounding-level positive novelty,
        # equal scores reached by another summation order, is no novelty; band = the join_proxy floor),
        # in tie-class order of their novelty (content: the candidate path), the class the budget cuts
        # kept whole (at most twice the budget).
        band = tie_rules.quantum('join_proxy')
        novel_items = [c for c in np.flatnonzero(eligible_mask).tolist()
                       if candidates[c] not in panel_set and novelty[c] > band]
        near = sorted(float(novelty[c]) for c in np.flatnonzero(eligible_mask).tolist()
                      if 0.0 < novelty[c] <= 1e3 * band)
        with self._tie_lock:
            # The band at the exact zero of a non-novel candidate's novelty (owner rule A-7).
            self.tie_counter.note('assembly.join_novelty.band', inside_tie=bool(near and near[0] <= band))
        if near:
            self._census('assembly.join_novelty.band', f"{sweep_key}", near[:16])
        novel = []
        if novel_items:
            scale = float(max(novelty[c] for c in novel_items))
            values = [float(novelty[c]) for c in novel_items]
            quantum = tie_rules.quantum('join_proxy', scale)
            result = ties.tie_classes(values, quantum, content=[candidates[c] for c in novel_items],
                                      descending=True)
            tie_rules.add_decision(self.tie_counter, 'assembly.join_novelty', result, values, quantum,
                                   tie_rules.motion('join_proxy', scale), descending=True, cut=quota)
            novel = tie_rules.truncate(novel_items, result, quota)
            self._census('assembly.join_novelty', f"{sweep_key}",
                         sorted(-float(novelty[c]) for c in novel_items)[:4 * quota])
        for c in novel:
            similarity = np.mean(local == candidate_array[c], axis=1)
            # the two most similar founders, with every founder tied with the second (release 2.2: order-invariant)
            if len(similarity) > 2:
                second = np.sort(similarity)[::-1][1]
                ranked = [int(f) for f in np.flatnonzero(similarity >= second)]
            else:
                ranked = list(range(len(similarity)))
            for founder in ranked:
                proposals['novel_replace'].append((float(novelty[c] * (.5 + similarity[founder])),
                                                   ('replace', int(founder), int(c))))
        static_gains = static_replacement_gains(local, candidate_array, totals=self.static_totals,
                                                workspace=self.candidate_workspace)
        wide_quota = 2 * quota
        proposals['static_replace'] = replacement_candidates(static_gains, eligible)
        proposal_counts['replace'] = proposal_counts['static_replace'] = k * len(eligible)
        proposal_counts['onesided'] = (blocks - 1) * k * (k - 1)
        proposal_counts['splice'] = (blocks - 1) * k * (k - 1) // 2
        proposal_counts['switch_splice'] = proposal_counts['splice']
        # Painting-dependent generators under both traceback tie conventions (release 2.2): the Viterbi
        # painting's choice among equal-score states is a convention, so the proposals of both paintings are
        # merged, the larger proxy of a repeated edit kept: the union does not depend on the order.
        merged = {}
        for convention in (False, True):
            generated, local_screened = self._painting_proposals(panel, local, eligible, wide_quota, convention)
            proposal_counts['local'] = local_screened
            for kind, items in generated.items():
                if isinstance(items, ScoredEdits):
                    if kind in merged:
                        merged[kind].merge(items)
                    else:
                        merged[kind] = items
                    continue
                bucket = merged.setdefault(kind, {})
                for gain, description in items:
                    if description not in bucket or gain > bucket[description]:
                        bucket[description] = gain
        for kind, bucket in merged.items():
            if isinstance(bucket, ScoredEdits):
                proposals[kind] = bucket
            else:
                proposals[kind].extend((gain, description) for description, gain in bucket.items())
        start_evaluations = self.evaluations
        best = None
        evaluated = []
        evaluated_by_kind = {}
        inadmissible = 0
        boundary_acceptances = []
        targets = {}
        current_signature = tuple(sorted(panel))
        improving_trials = []
        threshold_distances = []
        for kind, items in proposals.items():
            # The top q proposals of the kind before de-duplication: tie classes of their proxy gains
            # (content: the edit description, slots and pool indices being content ranks), the class
            # the budget cuts kept whole (at most 2q).
            ranked = []
            if isinstance(items, ScoredEdits):
                ranked = items.ranked(quota, self.tie_counter)
            elif items:
                scale = max(abs(float(item[0])) for item in items)
                values = [float(item[0]) for item in items]
                quantum = tie_rules.quantum('join_proxy', scale)
                result = ties.tie_classes(values, quantum, content=[item[1] for item in items], descending=True)
                tie_rules.add_decision(self.tie_counter, 'assembly.join_proposals', result, values, quantum,
                                       tie_rules.motion('join_proxy', scale), descending=True, cut=quota)
                ranked = tie_rules.truncate(items, result, quota)
                if len(items) > quota:
                    self._census('assembly.join_proposals', f"{sweep_key}/{kind}",
                                 sorted(-float(item[0]) for item in items)[:wide_quota])
            evaluated_by_kind[kind] = 0
            for proposal_rank, (gain, description) in enumerate(ranked, 1):
                trial = _materialize(description, panel, candidates)
                signature = tuple(sorted(trial))
                if signature == current_signature:
                    continue
                if not _admissible(panel, trial, self.geometry, self.near_identity):
                    inadmissible += 1
                    continue
                # Admissibility belongs to the edit witness. One inadmissible representation
                # must not suppress a different admissible route to the identical target.
                targets.setdefault(signature, []).append((kind, proposal_rank, gain, description, trial))
        for signature, witnesses in sorted(targets.items()):
            # The descriptor names a witness, not the target's biological identity. Choose it
            # consistently for diagnostics; score the canonical target independent of the witness.
            kind, proposal_rank, gain, description, trial = min(witnesses, key=lambda w: w[3])
            trial_scores, sparse = self._proposal_scores(panel, list(signature))
            sparse_scores += sparse
            trial_ll = float(trial_scores.sum())
            evaluated_by_kind[kind] += 1
            evaluated.append((float(trial_ll), str(description)))
            improving = _improves(trial_ll, panel_ll)
            threshold_distances.append(_improvement_distance(trial_ll, panel_ll))
            if improving:
                boundary_acceptances.extend(w[0] for w in witnesses if w[1] == quota)
                improving_trials.append((trial_ll, trial, description, gain, trial_scores, signature))
        # The winner: tie classes of the improving trials' log likelihoods, their content (sorted
        # paths) inside a class (base: the first evaluated of the maximum).
        if improving_trials:
            best = self._class_winner('assembly.join_tier1', improving_trials, lambda item: item[0],
                                      lambda item: item[5], census_key=f"{sweep_key}/winner",
                                      scale=panel_ll, link=lambda item: self._panel_link(item[1]),
                                      panel=lambda item: item[1])[:5]
        if threshold_distances:
            # Valley census of the improvement threshold (a continuous threshold: the distances of the
            # evaluated trials' gains to it, in nats; positive = improving).
            self._census('assembly.join_improve', f"{sweep_key}/t1", threshold_distances)
        leading = sorted([(float(panel_ll), "current panel"), *evaluated], reverse=True)[:2]
        record = dict(
            panel_size=k, candidate_pool=len(candidates),
            proposals={kind: proposal_counts.get(kind, len(items)) for kind, items in proposals.items()},
            full_scores=self.evaluations - start_evaluations, sparse_scores=sparse_scores,
            novelty_scored=novelty_scored, likelihood=float(panel_ll),
            accepted=None if best is None else str(best[2]), full_scores_per_kind=quota,
            evaluated_by_kind=evaluated_by_kind, inadmissible=inadmissible,
            truncated_kinds=[kind for kind, items in proposals.items()
                             if proposal_counts.get(kind, len(items)) > quota],
            improving_at_budget_boundary=boundary_acceptances,
            leading_evaluated_alternatives=leading,
            scope="this sweep only; proxy-truncated and ungenerated edits are untested",
            seconds=time.perf_counter() - started)
        return best, record

    # -- the count step ---------------------------------------------------------------------------

    def _refit_seeds(self, seeds, refit_scores):
        """The duplicate-seed refits: for each seed, certified best-improvement from P plus a copy
        of the seed over the same-count moves that change the seed or its copy (both exempt from
        the novelty rule meanwhile). Returns (panel, scores, likelihood, pass records) per seed.

        The refits are independent searches, advanced together one certify pass at a time: the
        passes of one round run concurrently (each refit with a share of the threads); before
        them, the pool's birth bounds of every path set the round meets are computed once with
        all threads (they depend on the path set alone, and the refits mostly reach the same
        sets), and the caches keep all of them through the round."""
        selected = self.selected
        states = [[selected + [selected[seed]], self.baseline_scores, self.likelihood,
                   frozenset((seed, len(selected))), []] for seed in seeds]
        active = list(range(len(states)))
        candidates = self.candidates

        step = self._count_steps

        def certify_pass(index):
            panel, _, panel_ll, slots, passes = states[index]
            return self._certify(panel, candidates, 0, True, len(candidates) + 1,
                                 panel_ll=panel_ll, slots=slots, refit_scores=refit_scores,
                                 key=f"c{step}/seed{seeds[index]}/p{len(passes)}")[:2]
        for _ in range(self.sweep_backstop):
            if not active:
                break
            panels = {}
            for index in active:
                panel = states[index][0]
                panels.setdefault(tuple(sorted(set(map(tuple, panel)))), panel)
            self.birth_sets = max(BIRTH_SETS, len(panels))
            for path_set, panel in panels.items():
                members = set(path_set)
                outside = [path for path in candidates if path not in members]
                forwards = self._panel_forwards(panel)
                if outside:
                    self._claim()
                    forwards.birth_values(self._rows(outside), self.block_offsets)
            moved = []
            outcomes = self._concurrent(certify_pass, active)
            for index, (best_move, record) in zip(active, outcomes):
                state = states[index]
                state[4].append(dict(accepted=record['accepted'], survivors=record['survivors'],
                                     exact=record['exact_scores'],
                                     change=record['accepted_likelihood_change']))
                if best_move is not None:
                    state[0], state[2], state[1] = (list(best_move[3]), best_move[4][0],
                                                    best_move[4][1])
                    moved.append(index)
            active = moved
        self.birth_sets = BIRTH_SETS
        return [(state[0], state[1], state[2], state[4]) for state in states]

    def _deaths(self, forwards) -> bool:
        """Deaths by the count rule; True if any path died.

        Paths without sequence of their own are deleted unconditionally, one after another
        without re-certifying in between (largest V(P minus f) first); otherwise the best-F
        candidate (largest V(P minus f)) is decided by the rule, and a deletion returns the
        search to tier 1 (the next death is decided at the re-optimized panel). Deleting a
        path that a birth here added closes a count cycle and freezes births; the deletion
        stands. It is recorded as undecided when it is ambiguous (its bootstrap band contains
        zero) and the cycle began with a decided deletion (count_rule.pre_cycle_count;
        deleting a path without sequence of its own is not a decision)."""
        count = self.count
        changed = False
        deaths = 0
        while True:
            outcome = self.counter.deletion(self.selected, forwards, self.baseline_scores,
                                            key=f"c{self._count_steps}/death{deaths}")
            deaths += 1
            if outcome is None:
                break
            record, rest, values = outcome
            removed = self.selected[record['founder']]
            record['tagged'] = bool(removed in self.born)
            if rest is not None and record['tagged']:
                count['cycles'] += 1
                if not record['pure_mosaic']:
                    origin = count_rule.pre_cycle_count(self.sizes, len(self.selected),
                                                        self.cleanup_steps)
                    record['cycle_start_size'] = origin
                    if origin >= len(self.selected) and record['ambiguous']:
                        record['cycle_undecided'] = True
                        count['cycles_undecided'] += 1
            self.events.append(dict(stage='count_death', **record))
            if rest is None:
                break
            if removed in self.born:
                self.births_frozen = True       # count cycle: no further births here
                self.born.discard(removed)
            count['decisions'] += 1
            count['deaths'] += 1
            count['pure_mosaic_deaths'] += record['pure_mosaic']
            count['ambiguous'] += record['ambiguous']
            self._set_panel(rest, values, float(values.sum()))
            self._note_count_state('death')
            self.sizes.append(len(self.selected))
            if record['pure_mosaic']:
                self.cleanup_steps.add(len(self.sizes) - 1)
            changed = True
            if not record['pure_mosaic']:
                break
            forwards = self._panel_forwards(self.selected)
        return changed

    def _count_step(self, forwards, pool=None, seeds=False) -> bool:
        started = time.perf_counter()
        self._count_steps += 1
        try:
            return self._count_decision(forwards, pool, seeds)
        finally:
            self.count['seconds'] += time.perf_counter() - started

    def _count_decision(self, forwards, pool, seeds) -> bool:
        """One count decision; True if the panel changed. pool None: deaths, then births from the
        current pool, then (seeds=True, at a certified panel) duplicate-seed births; otherwise
        births by the given (saturation) paths only."""
        count = self.count
        if pool is None:
            if self._deaths(forwards):
                return True
            pool_paths = self.candidates
        else:
            pool_paths = pool
        if not self.births_frozen and count['decisions'] + count['seed_swaps'] >= self.count_backstop:
            # Reported backstop: births and seed swaps stop (deaths still decided, panel moves continue).
            self.births_frozen = True
            count['count_backstop_reached'] = True
            self.events.append(dict(stage='count_backstop', changes=count['decisions'] + count['seed_swaps'],
                                    panel_size=len(self.selected)))
            print(f"  [panel search] count backstop {self.count_backstop} reached (panel "
                  f"{len(self.selected)}): births and seed swaps stop", flush=True)
        if self.births_frozen:
            return False
        record, larger_panel, values, indicators = self.counter.birth(
            self.selected, pool_paths, forwards, self.baseline_scores, key=f"c{self._count_steps}")
        count['trap_indicators'] = max(count['trap_indicators'], record['trap_indicators'])
        best = None if larger_panel is None else (float(values.sum()), larger_panel, values, record,
                                                  'pool')
        if seeds and best is None and indicators:
            # Duplicate-seed births: pure-mosaic pool births that pass the rule on their total
            # gain mark a panel path covering a founder's sequence (the chimera trap). Copy the
            # main coverer of the strongest ones, refit the two copies, and keep the refit panel
            # if it survives its own death test and LL rises.
            if len(self.selected) <= self.seed_refits:
                chosen = list(range(len(self.selected)))      # every seed when k <= r
            else:
                # The strongest indicators (tie classes of their bound gains, the class the budget
                # cuts kept whole), then their exact gains in tie-class order (content inside).
                screened = indicators[:3 * self.seed_refits]
                if len(indicators) > len(screened):
                    scale = max(abs(item[0]) for item in indicators)
                    bound_gains = [item[0] for item in indicators]
                    quantum = tie_rules.quantum('join_proxy', scale)
                    result = ties.tie_classes(bound_gains, quantum, content=[item[1] for item in indicators],
                                              descending=True)
                    tie_rules.add_decision(self.tie_counter, 'assembly.join_seed_screen', result, bound_gains,
                                           quantum, tie_rules.motion('join_proxy', scale), descending=True,
                                           cut=3 * self.seed_refits)
                    screened = tie_rules.truncate(indicators, result, 3 * self.seed_refits)
                strongest = []
                for _, path in screened:
                    gain = self.counter.total_gain(self.selected, path, forwards,
                                                   self.baseline_scores)
                    if gain is not None:
                        strongest.append((gain, path))
                if strongest:
                    strongest, _ = self._class_order('assembly.join_seed_rank', strongest,
                                                     lambda item: item[0], lambda item: item[1],
                                                     link=lambda item: self._panel_link([item[1]]))
                chosen = []
                for _, path in strongest:
                    seed = main_coverer(self.selected, path)
                    if seed not in chosen:
                        chosen.append(seed)
                    if len(chosen) >= self.seed_refits:
                        break
            seed_started = time.perf_counter()
            # Every refit starts from P plus a copy, so the refits of one count step reach many of
            # the same one-path moves (above all replace(seed, c) = P + c): each is scored once
            # (its value does not depend on the refit that scored it).
            refits = self._refit_seeds(chosen, {})
            step = self._count_steps
            settles = self._concurrent(
                lambda item: self.counter.settle(item[1][0], item[1][1], key=f"c{step}/seed{item[0]}"),
                list(zip(chosen, refits)))
            candidates = []
            for seed, (_, _, _, used), (settled, settled_scores, settle_record, survives) in zip(
                    chosen, refits, settles):
                count['seed_refits'] += 1
                settled_ll = float(settled_scores.sum())
                seed_record = dict(kind='seed_birth', seed=seed, refit_passes=len(used), refit=used,
                                   settled_size=len(settled), survives=bool(survives),
                                   likelihood_change=settled_ll - self.likelihood,
                                   settle=settle_record)
                self.events.append(dict(stage='count_seed', **seed_record))
                if (survives and len(settled) >= len(self.selected)
                        and _improves(settled_ll, self.likelihood)):
                    candidates.append((settled_ll, settled, settled_scores, seed_record, 'seed'))
            if candidates:
                # The best surviving refit: tie classes of its log likelihood, the settled panel's
                # content inside a class (base: the first seed of the maximum).
                best = self._class_winner('assembly.join_seed', candidates, lambda item: item[0],
                                          lambda item: tie_rules.panel_key(item[1]),
                                          census_key=f"c{step}/seed",
                                          link=lambda item: self._panel_link(item[1]), panel=lambda item: item[1])
            count['seed_seconds'] += time.perf_counter() - seed_started
        self.events.append(dict(stage='count_birth', **record))
        if best is None:
            return False
        new_ll, new_panel, new_scores, new_record, source = best
        if tie_rules.panel_key(new_panel) in self.count_states:
            # The step would return to a panel this search already held after a count change: a count
            # cycle (births, deaths and seed swaps chasing each other; seed swaps never tag born paths, so
            # the born-path freeze misses it). Births and seed swaps stop; the panel stays.
            self.births_frozen = True
            count['state_revisits'] += 1
            self.events.append(dict(stage='count_revisit', source=source, panel_size=len(new_panel)))
            return False
        if source == 'seed':
            # Equivalent settled endpoints can have different unresolved source histories.
            # Keep every accepted witness of this exact panel: selecting one seed's
            # diagnostic record must not select which ambiguity reaches release.
            selected_key = tie_rules.panel_key(new_panel)
            accepted_seeds = sorted(
                {candidate[3]['seed'] for candidate in candidates
                 if tie_rules.panel_key(candidate[1]) == selected_key},
                key=lambda seed: (self.selected[seed], seed))
            # Seed histories arrive concurrently; release them in founder-path content order.
            self.release_families.extend(
                family for seed in accepted_seeds
                for family in self._seed_families.get(f"c{self._count_steps}/seed{seed}", ()))
        old = set(self.selected)
        if len(new_panel) > len(self.selected):
            count['decisions'] += 1
            count['births'] += 1
            count['seed_births' if source == 'seed' else 'pool_births'] += 1
            count['ambiguous'] += bool(new_record.get('ambiguous'))
            self.born.update(path for path in new_panel if path not in old)
        else:
            count['seed_swaps'] += 1
        self._set_panel(new_panel, new_scores, new_ll)
        self.count_states.add(tie_rules.panel_key(self.selected))
        if len(self.selected) != self.sizes[-1]:
            self.sizes.append(len(self.selected))
        return True

    def _note_count_state(self, source):
        """Record the panel held after a death; a death returning to a panel this search already held after
        a count change stands but stops births and seed swaps (a count cycle)."""
        key = tie_rules.panel_key(self.selected)
        if key in self.count_states and not self.births_frozen:
            self.births_frozen = True
            self.count['state_revisits'] += 1
            self.events.append(dict(stage='count_revisit', source=source, panel_size=len(self.selected)))
        self.count_states.add(key)

    def _final_census(self, forwards) -> int:
        """Final paths that would fail their own death test (reported; nothing changes)."""
        selected = self.selected
        k = len(selected)
        if k < 2:
            return 0
        failing = 0
        removal = forwards.removal_scores()
        for founder in range(k):
            rest = [path for i, path in enumerate(selected) if i != founder]
            cover = count_rule.covered_rows(np.asarray(rest, dtype=np.int64))
            if not count_rule.has_novelty(selected[founder], rest, cover, self.geometry,
                                          self.near_identity):
                failing += 1
                continue
            decision, _, _ = self.counter._shadow_decision(
                selected, rest, founder, self.baseline_scores, removal[:, founder],
                'deletion', None)
            failing += not decision.accept_larger
        return failing

    # -- the search --------------------------------------------------------------------------------

    def minimum_switch_count(self, panel=None) -> int:
        """Exact minimum switches of a requested finalist, cached on its canonical panel."""
        key = tie_rules.panel_key(self.selected if panel is None else panel)
        with self._tie_lock:
            found = self._switch_counts.get(key)
        if found is None:
            self._claim()
            found = int(objective.minimum_switches(*self.layout.arrays(), self._rows(key),
                                                   self.penalty, self.samples).sum())
            with self._tie_lock:
                self._switch_counts[key] = found
        return found

    def _continue_terminal_family(self) -> bool:
        """Choose an observed flat neighbor, then run its ordinary continuation.

        The current panel has passed this history's count and saturation checks.
        Earlier certificates in the same uninterrupted flat episode are not
        revisited: their stay-here choice is already an earlier replay leaf. A
        genuine accepted move resets the episode, so its new count/history state
        is not mistaken for an already certified flat continuation.
        """
        if not self._plateau_candidates:
            return False
        from .alternatives import observed_plateau_family

        family = observed_plateau_family(self.selected, self.likelihood,
                                         self._plateau_candidates, linkage=self.linkage)
        if family is None:
            return False
        started = time.perf_counter()
        nominal = tie_rules.panel_key(self.selected)
        panels = family['panels']
        eligible = [panel for panel in panels
                    if panel == nominal or panel not in self._plateau_certified]
        excluded = [panel for panel in panels if panel not in eligible]
        self._plateau_certified.add(nominal)
        if len(eligible) < 2:
            # Every alternative is covered by an earlier source choice. Do not
            # create A->B->A replay cycles or erase those source families.
            self.events.append(dict(stage='terminal_cycle_covered', panel=nominal,
                                    observed_panels=panels, cycle_excluded_panels=excluded))
            return False
        self._terminal_choices += 1
        source = f"terminal_certification/{self._terminal_choices}"
        values = [family['values'][panels.index(panel)] for panel in eligible]
        selected = eligible[self._prefer_panels(family['reason'], eligible, values, source)]
        # The replay event omits already certified predecessors solely to prevent
        # flat cycles. The release record retains the COMPLETE observed family;
        # its eligible identity is consumed only when the whole tree completes.
        record = self.release_families[-1]
        record.update(family, source_decision=source, replay_panels=eligible,
                      cycle_excluded_panels=excluded,
                      minimum_switches=[self.minimum_switch_count(panel) for panel in panels],
                      selected_panel=selected)
        self.events.append(dict(stage='terminal_choice', source_decision=source,
                                observed_panels=len(panels), replay_choices=len(eligible),
                                cycle_excluded=len(excluded), continued=selected != nominal,
                                representative_prior=record['representative_prior'],
                                minimum_switches=record['minimum_switches']))
        if selected == nominal:
            self.events[-1]['seconds'] = time.perf_counter() - started
            return False
        visited = self._plateau_certified
        # This is a branch transition, not an improving move or count decision.
        # Reuse ordinary born-tag inheritance; keep count states, freeze flags,
        # accepted-change counters and all loop/backstop/quota state intact.
        scores = self._score(selected)
        self._apply_same_count(selected, scores)
        self._plateau_certified = visited
        self.events[-1]['seconds'] = time.perf_counter() - started
        return True

    def run(self, *, report_switches=False) -> JoinOutcome:
        """Search to a certified panel; optionally report the exact finalist switch prior."""
        with ExitStack() as held:       # the ladder extension's memory reservation, if it climbs on
            outcome = self._search(held)
            if report_switches:
                outcome = replace(outcome, minimum_switches=self.minimum_switch_count())
            return outcome

    def _search(self, held: ExitStack) -> JoinOutcome:
        config = self.config
        quota = config.paths_per_endpoint
        ladder_top = config.paths_per_endpoint * 2 ** POOL_QUOTA_DOUBLINGS
        quota_backstop = config.paths_per_endpoint * 2 ** POOL_EXTENDED_DOUBLINGS
        sweeps = phase_sweeps = tier1_sweeps = tier2_passes = tier2_accepted = 0
        saturation_steps = saturation_accepted = 0
        certified = backstop_reached = pool_backstop = False
        largest_pool = len(self.candidates)
        final_forwards = None
        while True:
            if self.beam_stopped:
                backstop_reached = pool_backstop = True
                break
            if phase_sweeps >= self.sweep_backstop:
                backstop_reached = True
                print(f"  [panel search] sweep backstop {self.sweep_backstop} reached "
                      f"(panel {len(self.selected)}, pool {len(self.candidates)}): not certified",
                      flush=True)
                break
            best, record = self._tier1_sweep(self.selected, self.baseline_scores, self.likelihood)
            record.update(sweep=tier1_sweeps + 1)
            self.events.append(record)
            tier1_sweeps += 1
            sweeps += 1
            phase_sweeps += 1
            if best is not None:
                self._apply_same_count(best[1], best[4])
                continue
            # A tier-1 stall: count decisions first (re-certification only once they are quiescent).
            stall_forwards = self._panel_forwards(self.selected)
            if self._count_step(stall_forwards):
                phase_sweeps = 0
                continue
            # Tier 2 at the stall, on C0 plus the beam with the panel's multiplicity.
            pool_before = len(self.candidates)
            beam_before = self.beam_seconds
            pool = self._extension_pool(quota, self.selected)
            if self.beam_stopped:
                backstop_reached = pool_backstop = True
                break
            if pool is not self.candidates:
                self._set_pool(pool)
                largest_pool = max(largest_pool, len(self.candidates))
                # The pool grew: births by its new paths are count moves of this stall too.
                if self._count_step(stall_forwards):
                    phase_sweeps = 0
                    continue
            tier2_passes += 1
            best_move, record, forwards = self._certify(self.selected, self.candidates, 0, True,
                                                        self.capacity, key=f"t2/{tier2_passes}")
            record.update(tier=2, tier2_pass=tier2_passes, quota=quota, pool_before=pool_before,
                          beam_seconds=self.beam_seconds - beam_before)
            self.events.append(record)
            if best_move is not None:
                tier2_accepted += 1
                sweeps += 1
                phase_sweeps += 1
                self._apply_same_count(best_move[3], best_move[4][1])
                continue
            # Same-count certified and count-quiescent: duplicate-seed births (the chimera trap).
            if self._count_step(forwards, seeds=True):
                phase_sweeps = 0
                continue
            # Pool saturation: certify only when no doubled quota up to the ladder top adds an
            # improving replacement or accepted birth. The top is 256 for every join; once a
            # move was accepted at the 256 rung (the quota reached it) the ladder climbs on to
            # the backstop.
            if quota >= ladder_top:
                if ladder_top != quota_backstop:
                    from ...execution import reserve_memory
                    held.enter_context(reserve_memory(
                        ladder_extension_bytes(self.blocks, max(self.block_sizes), config.paths_per_endpoint),
                        label="join pool extension"))
                ladder_top = quota_backstop
            if 2 * quota > ladder_top:
                pool_backstop = True
                print(f"  [panel search] pool backstop: a move was accepted at quota "
                      f"{quota} = backstop (panel {len(self.selected)}): not certified", flush=True)
                final_forwards = forwards
                break
            checked = set(self.candidates)
            level = quota
            accepted = False
            while not accepted and 2 * level <= ladder_top:
                level *= 2
                saturation_steps += 1
                beam_before = self.beam_seconds
                doubled = [path for path in self._extension_pool(level, self.selected)
                           if path not in checked]
                if self.beam_stopped:
                    break
                best_move, record, level_forwards = self._certify(
                    self.selected, doubled, len(self.candidates), False,
                    len(self.candidates) + len(doubled), key=f"sat{saturation_steps}")
                checked.update(doubled)
                record.update(stage='pool_saturation', quota=quota, tested_quota=level,
                              new_paths=len(doubled), beam_seconds=self.beam_seconds - beam_before)
                self.events.append(record)
                if best_move is not None:
                    self._apply_same_count(best_move[3], best_move[4][1])
                    accepted = True
                elif doubled and self._count_step(level_forwards, pool=doubled):
                    self.count['ladder_births'] += 1
                    accepted = True
            if self.beam_stopped:
                backstop_reached = pool_backstop = True
                break
            if not accepted:
                if self._continue_terminal_family():
                    # No fresh sweep/backstop/quota budget and no improvement
                    # counter. The chosen neighbor must pass its own continuation.
                    continue
                certified = True
                final_forwards = forwards
                break
            saturation_accepted += 1
            sweeps += 1
            phase_sweeps = 0
            quota = level
            self._set_pool(self._extension_pool(quota, self.selected))
            largest_pool = max(largest_pool, len(self.candidates))
        self.count['births_frozen'] = self.births_frozen
        if final_forwards is not None:
            self.count['final_paths_failing_death_test'] = self._final_census(final_forwards)
        summary = JoinSummary(
            final_panel_size=len(self.selected), final_likelihood=float(self.likelihood),
            total_full_scores=self.evaluations, initialization=tuple(self.initialization),
            selected_initial_anchor=self.selected_anchor, sweeps=sweeps, tier1_sweeps=tier1_sweeps,
            sweep_backstop=self.sweep_backstop, tier2_passes=tier2_passes,
            tier2_accepted=tier2_accepted, saturation_steps=saturation_steps,
            saturation_accepted=saturation_accepted, quota=quota, ladder_top=ladder_top,
            quota_backstop=quota_backstop, beam_seconds=self.beam_seconds,
            count=CountCensus(**self.count),
            count_exact_scores=self.counter.exact_scores,
            count_exact_sample_scores=self.counter.exact_sample_scores,
            pool_original=len(self.original_pool), pool_final=len(self.candidates),
            pool_largest=largest_pool, certified=certified, backstop_reached=backstop_reached,
            pool_backstop=pool_backstop, events=tuple(_plain(event) for event in self.events),
            tie_events=self.tie_counter.record(),
            phase_seconds={"count_step": float(self.count["seconds"])},
            work_counters={"beam_materialization_bounds": sum(
                event.get("stage") == "beam_materialization_bound" for event in self.events)})
        return JoinOutcome(paths=np.asarray(self.selected, dtype=np.int32).reshape(-1, self.blocks),
                           log_likelihood=float(self.likelihood), summary=summary,
                           release_alternatives=tuple(self.release_families))


def search_panel(inputs, beam, beam_runner: Callable[[int, Mapping | None], list], *, config,
                 count_rule, cost_scale: float, budget, label: str | None = None,
                 counter: ties.TieCounter | None = None, linkage=None, report_switches=False,
                 max_branches=4) -> JoinOutcome:
    """The certified panel of one joined batch.

    inputs: JoinInputs (n >= 2 input components, the informative samples' float32 evidence).
    beam: the batch's endpoint beam at quota ``config.paths_per_endpoint``, [(path, score)] with
        paths of local rows (one per input component). beam_runner(quota, overrides): the
        batch's endpoint beam re-run with a base quota and per-endpoint quotas
        ({(block, row): quota}), same format (pool extension and saturation rungs).
    config: JoinConfig; count_rule: CountRuleConfig; cost_scale: the per-founder cost scale
        (count_rule.founder_cost); budget: the batch's CpuBudget (re-read at the search's
        boundaries; concurrent seed refits share it). label: the batch's valley-census label (None:
        no census); counter: the batch's maths.ties.TieCounter (shared with its beam; recorded as the
        summary's tie_events). max_branches: complete original-input executions, at
        most four including the primary; use one when an outer joint input family
        already consumes the branch budget. report_switches: return the exact final
        minimum-switch prior even if the terminal panel is unique.
    """
    from .alternatives import run_bounded_replays, resolve_replays

    shared_layout = None
    last_search = None
    shared_counter = counter if counter is not None else tie_rules.SharedCounter()

    def run(plan):
        nonlocal shared_layout, last_search
        # Only emissions are immutable shared state. Born paths, count cycles,
        # candidate pools, histories and caches start from the original inputs.
        last_search = None
        last_search = PanelSearch(inputs, beam, beam_runner, config=config,
                                  count_rule_config=count_rule, cost_scale=cost_scale,
                                  budget=budget, label=label, counter=shared_counter,
                                  linkage=linkage, choice_plan=plan,
                                  shared_layout=shared_layout)
        shared_layout = last_search.layout
        try:
            return last_search.run()
        finally:
            # The finished CountStep owns bound callbacks to this search. Break
            # that cycle so sequential branches release their large caches now.
            last_search.counter = None

    replay = run_bounded_replays(run, max_branches=max_branches)
    # A failed forced prefix may have raised during construction. Reuse the
    # immutable layout directly for finalist priors rather than depending on that
    # partially constructed mutable search.
    def minimum_switches(panel):
        if last_search is not None:
            return last_search.minimum_switch_count(panel)
        threads = max(1, int(budget.threads()))
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
        site_counts = [len(block.positions) for block in inputs.blocks]
        return int(objective.minimum_switches(
            *shared_layout.arrays(), np.ascontiguousarray(panel, dtype=np.int64),
            emissions.switch_penalty(site_counts), int(inputs.sample_count)).sum())

    return resolve_replays(replay, minimum_switches, report_switches=report_switches,
                           tie_events=shared_counter.record())
