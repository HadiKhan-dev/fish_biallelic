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
duplicate-seed births run (a copy of every selected founder when chimera-trap indicators
are present, refitted concurrently) and then a pool-saturation step re-runs the endpoint beam with doubled quotas
before the panel is certified (up to 256 paths per endpoint, up to 1024 for a join whose 256 rung
accepted a move). The panel size follows from the count rule; the pool, not a cap, bounds it; a
reported backstop never decides a normal result.

Costs: each sweep O(N m K^2 + K^2 B log K) (candidate scores only where a block-restart bound
leaves a candidate's novelty open); a tier-2 pass O(N m K (|C| + K^2)) (structural moves share
forward prefixes and backward suffixes); a count step O(N m K |C|) plus sparse exact scores.

Determinism: per-sample values are computed by one thread each and summed in sample order;
concurrent refits are evaluated in seed order; caches never change results; so the selected
panel does not depend on thread counts. The Numba team is bound by the caller callback at the
boundaries where the search changes phase (each full score or painting chunk, each panel's
forwards, each certify pass); the caller binds each concurrent refit to its assigned team.

Ties: the panel is kept in canonical order (its paths sorted, tie_rules.sort_panel) after
every change, so every slot tie-break (painting ties to the lower slot, proposal descriptions, the death candidate,
duplicate-seed enumeration) is a content rule; the pool comes from the beam in tie-class order. Every choice
between candidates on a continuous value (start panel, tier-1 winner, the top proposals of each kind,
tier-2 winner, death candidate, birth and seed choices) uses anchored tie classes of the join log likelihood (quantum
ties.quantum("join_ll", |LL|)) with the candidate panel's content (its sorted paths) deciding inside a class;
a truncation keeps every proposal of the class it cuts (at most twice the budget).

Owns: initial panel selection and the certified same-count/pool loop.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: docs/methods/assembly.md#aj12.
Invariant: Only changed ordered pool content triggers additional count evidence; cache identity does not.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numba
import numpy as np

from haplotype_reconstruction.core import ties

from .. import orders as tie_rules
from ..count_rule import criterion as count_criterion
from ..count_rule import geometry as count_geometry
from ..count_rule.parameters import BIRTH_SETS, COUNT_BACKSTOP_BASE, COUNT_BACKSTOP_PER_ROW
from ..parameters import (
    EXPLORATORY_PENALTY,
    IMPROVEMENT_ABSOLUTE,
    IMPROVEMENT_RELATIVE,
    MAX_BRANCHES,
    POOL_EXTENDED_DOUBLINGS,
    POOL_QUOTA_DOUBLINGS,
    SWEEP_BACKSTOP_BASE,
    SWEEP_BACKSTOP_PER_ROW,
)
from . import emissions, objective
from .beam import CartesianBeamFamily
from .certify import PanelForwards
from .count_step import (
    CountStep,
    apply_same_count,
    count_step,
    final_census,
    record_count_family,
)
from .diagnostics import (
    CountCensus,
    InitialPanel,
    JoinOutcome,
    JoinSummary,
    _plain,
    append_event,
)
from .moves import MOVES_PER_CHUNK, REPLACE, SPLICE, Moves, best_improvement, lower_bounds, replaced_row
from .moves import improves as _improves
from .proposals import (
    ScoredEdits,
    boundary_proposals,
    boundary_vote_matrix,
    candidate_novelty,
    conditional_fields,
    cycle_permutations,
    fixed_paint_delta,
    greedy_permutations,
    local_candidates,
    prepare_candidate_scores,
    replacement_candidates,
    static_replacement_gains,
)
from .state import DEFAULT_TIE_SETTINGS, Panel, PanelKey, Path, search_tie_policy

if TYPE_CHECKING:
    from haplotype_reconstruction.contracts.assembly import JoinInputs, ReleaseFamily

    from ..count_rule.parameters import CountRuleParameters
    from ..parameters import JoinParameters
    from .emissions import EmissionLayout

Beam = list[tuple[Path, float]] | CartesianBeamFamily
BeamRunner = Callable[[int, Mapping[tuple[int, int], int] | None], Beam]

# Reported backstops, never expected to bind: total sweeps (tier-1 sweeps plus accepted tier-2
# moves) <= base + per_row x max_b K_b, and the pool-saturation ladder. Every join doubles the
# endpoint quota up to POOL_QUOTA_DOUBLINGS times (16 x 2**4 = 256 paths per endpoint); a join
# whose 256 rung accepted a move climbs on, up to POOL_EXTENDED_DOUBLINGS (16 x 2**6 = 1024). A
# panel is certified only when no rung up to its ladder top adds a move; a move accepted at 1024
# is the reported pool backstop.
# The count-change backstop guards cycles returning through seed swaps, which the born-path
# freeze cannot see. Normal searches make at most a few count changes.


# Panels whose exclusion forwards are kept (a sweep, its stall, tier 2 and the saturation ladder
# often share one panel) and beam re-runs kept per search.
FORWARDS_KEPT = 3
BEAM_RUNS_KEPT = 4
# Same-count proposal kinds of a tier-1 sweep, in evaluation order.
SAME_COUNT_KINDS = (
    "replace",
    "static_replace",
    "novel_replace",
    "local",
    "onesided",
    "splice",
    "cycle",
    "switch_splice",
    "switch_cycle",
)

__all__ = [
    "BEAM_RUNS_KEPT",
    "BIRTH_SETS",
    "CountCensus",
    "InitialPanel",
    "JoinOutcome",
    "JoinSummary",
    "MOVES_PER_CHUNK",
    "POOL_EXTENDED_DOUBLINGS",
    "POOL_QUOTA_DOUBLINGS",
    "PanelSearch",
    "search_panel",
]


def _materialize(description, selected, candidates):
    """Panel after a same-count move (paths keep their slots)."""
    kind, *args = description
    result = [list(path) for path in selected]
    if kind == "replace":
        founder, candidate = args
        result[founder] = list(candidates[candidate])
    elif kind == "local":
        founder, block, hap = args
        result[founder][block] = hap
    elif kind == "onesided":
        first, second, boundary = args
        result[first][boundary:] = selected[second][boundary:]
    elif kind == "splice":
        first, second, boundary = args
        result[first][boundary:], result[second][boundary:] = (result[second][boundary:], result[first][boundary:])
    elif kind == "cycle":
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
    checked = [i for i, (a, b) in enumerate(zip(old, new, strict=False)) if a != b and i not in exempt]
    if not exempt and len(set(new)) != len(new):
        return False
    if not checked:
        return True
    private = count_geometry.private_mask(np.asarray(new, dtype=np.int64))
    for i in checked:
        if not private[i].any():
            return False
        others = [path for j, path in enumerate(new) if j != i]
        if new[i] in others or geometry.near_identical(new[i], others, near_identity):
            return False
    return True


def _improvement_distance(new_ll, old_ll) -> float:
    """Signed distance (nats of LL) of a rise to the improvement threshold of :func:`_improves`."""
    return (new_ll - old_ll) - 0.5 * max(IMPROVEMENT_ABSOLUTE, IMPROVEMENT_RELATIVE * (1.0 + 2.0 * abs(old_ll)))


def _paths(beam) -> list[tuple[int, ...]]:
    """Distinct beam paths (tuples of Python ints) in beam order."""
    return list(dict.fromkeys(tuple(int(row) for row in path) for path, _ in beam))


@dataclasses.dataclass
class _Loop:
    """The search loop's own state (part of the complete state at a sweep boundary, AJ29).

    quota: the endpoint quota of the pool; ladder_top / quota_backstop: the saturation ladder's top and its
    backstop; largest_pool: the largest candidate pool so far; sweeps (tier-1 sweeps plus accepted tier-2 moves and
    saturation rungs), phase_sweeps (since the last count change: the sweep backstop's count), tier1_sweeps,
    tier2_passes, tier2_accepted, saturation_steps, saturation_accepted; certified / backstop_reached /
    pool_backstop: how the loop ended. Every field enters PanelSearch.state_key.
    """

    quota: int
    ladder_top: int
    quota_backstop: int
    largest_pool: int
    sweeps: int = 0
    phase_sweeps: int = 0
    tier1_sweeps: int = 0
    tier2_passes: int = 0
    tier2_accepted: int = 0
    saturation_steps: int = 0
    saturation_accepted: int = 0
    certified: bool = False
    backstop_reached: bool = False
    pool_backstop: bool = False


def _canonical(value):
    """A plain, order-defined form of a state value for the state key: ints and floats as Python numbers (repr is
    exact), sets sorted, dicts in insertion order, arrays as (dtype, shape, bytes)."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return ("array", contiguous.dtype.str, contiguous.shape, contiguous.tobytes())
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(_canonical(item) for item in value)))
    if isinstance(value, Mapping):
        return ("map", tuple((_canonical(key), _canonical(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    raise TypeError(f"no state-key form for {type(value).__name__}")


class PanelSearch:
    """One join's certified panel search (see the module docstring); :meth:`run` executes it.

    The search state: ``selected`` (the panel, a list of path tuples), its per-sample scores
    ``baseline_scores`` and log likelihood ``likelihood``; the candidate pool (``candidates``,
    with its array, sorted workspace and capacity); the paths the search's births added
    (``born``, followed through same-count moves); the panel sizes after every count change
    (``sizes``, with the changes that deleted a path without sequence of its own in
    ``cleanup_steps``) and the count tallies; the caches (exclusion forwards of the last panels,
    birth bounds by path set, beam re-runs), which never change results.

    STATE_FIELDS (with the loop's _Loop) is the complete state at a sweep boundary: everything the rest of the
    search reads that is neither a constant of the join, a pure cache, a record the search only appends to, nor a
    work or tie counter. state_key digests it; a replay whose key equals an earlier execution's adopts that
    execution's continuation (alternatives.ReplayMerge, AJ29).
    """

    STATE_FIELDS = (
        "selected",
        "baseline_scores",
        "likelihood",
        "candidates",
        "capacity",
        "born",
        "births_frozen",
        "birth_sets",
        "count_states",
        "sizes",
        "cleanup_steps",
        "count",
        "count_steps",
        "_sweeps_run",
        "_certify_runs",
        "_terminal_choices",
        "_plateau_candidates",
        "_plateau_certified",
        "refit_events",
    )
    # The bounded replays' merge index (alternatives.ReplayMerge; None: no merging).
    merge = None

    def __init__(
        self,
        inputs: JoinInputs,
        beam: Beam,
        beam_runner: BeamRunner,
        *,
        config: JoinParameters,
        count_rule_config: CountRuleParameters,
        cost_scale: float,
        claim_threads: Callable[[], int] | None = None,
        map_tasks: Callable[[Callable, list, int], list] | None = None,
        label: str | None = None,
        counter: ties.TieCounter | None = None,
        linkage: Callable[[Path], float] | None = None,
        choice_plan: Any = None,
        shared_layout: EmissionLayout | None = None,
        tie_settings: ties.TieSettings = DEFAULT_TIE_SETTINGS,
        merge: Any = None,
    ) -> None:
        self.tie_settings = tie_settings
        self.choice_plan = choice_plan
        self.merge = merge
        self.config = config
        # The batch linker's path score (join/beam.path_linkage): the data criterion inside
        # log-likelihood tie classes (None: panel content alone decides inside a class).
        self.linkage = linkage
        self._claim_threads = claim_threads or numba.get_num_threads
        self._map_tasks = map_tasks
        # The batch's census label (None: no valley census) and its tie-class decision counter.
        self.label = label
        self.tie_counter = counter if counter is not None else ties.SharedCounter()
        self.tie_lock = threading.Lock()  # concurrent seed refits record decisions too
        self._sweeps_run = 0
        self._certify_runs = 0
        self.count_steps = 0
        self.beam_runner = beam_runner
        self._local = threading.local()
        self.blocks = len(inputs.blocks)
        self.samples = int(inputs.sample_count)
        self.initial_family = beam if isinstance(beam, CartesianBeamFamily) else None
        if self.initial_family is not None:
            # The constant family's bounded first-covering proposal pool. Other
            # Cartesian combinations are outside this declared search approximation.
            # self.initial_family is beam itself (the narrowed name for the type checker).
            candidates = list(
                dict.fromkeys(
                    self.initial_family.first_covering(block, row)
                    for block, count in enumerate(self.initial_family.founders)
                    for row in range(count)
                )
            )
        else:
            candidates = _paths(beam)
        if not candidates:
            raise ValueError("a join needs at least one beam path")
        site_counts = [len(block.positions) for block in inputs.blocks]
        self.penalty = emissions.switch_penalty(site_counts)
        self.sites_per_bin = emissions.sites_per_bin(site_counts, config.max_bins)
        self.cost = count_criterion.founder_cost(site_counts, self.samples, cost_scale)
        self.layout = (
            emissions.binned_emissions(inputs, self.sites_per_bin, self.claim)
            if shared_layout is None
            else shared_layout
        )
        self.total_bins = self.layout.total_bins
        self.block_offsets = np.ascontiguousarray(self.layout.offsets[:-1], dtype=np.int64)
        self.block_sizes = [int(k) for k in self.layout.founders]
        self.geometry = count_geometry.RowGeometry.from_panels(inputs.blocks, reverse=self.tie_settings.reverse)
        self.near_identity = float(count_rule_config.near_identity)
        self.release_families: list[dict[str, Any]] = []
        # Observed flat neighbours of the current panel: canonical panel -> (log likelihood, certify key).
        self._plateau_candidates: dict[PanelKey, tuple[float, str | None]] = {}
        self._plateau_certified: set[PanelKey] = set()
        self._terminal_choices = 0
        self.seed_families: dict[str, list[dict[str, Any]]] = {}
        self._switch_counts: dict[PanelKey, int] = {}
        self.counter = CountStep(
            layout=self.layout,
            penalty=self.penalty,
            cost=self.cost,
            samples=self.samples,
            geometry=self.geometry,
            config=count_rule_config,
            score=self._score,
            tie_counter=self.tie_counter,
            label=label,
            linkage=linkage,
            choose_panel=self.prefer_panels,
            record_count_family=lambda family: record_count_family(self, family),
            tie_settings=self.tie_settings,
        )
        self.evaluations = 0
        self._evaluations_lock = threading.Lock()
        self.original_pool = candidates
        self.original_set = set(candidates)
        # None until _set_pool installs the original pool at the end of __init__ (nothing reads it before).
        self.candidates: Panel = cast("Panel", None)
        self.candidate_array = self.candidate_workspace = None
        self.capacity = 0
        self.beam_cache: dict[tuple[int, tuple[tuple[tuple[int, int], int], ...]], Panel] = {}
        self.forwards_cache: dict[PanelKey, PanelForwards] = {}
        self.births_by_set: dict[PanelKey, dict[bytes, np.ndarray]] = {}
        self.birth_sets = BIRTH_SETS
        self._cache_lock = threading.Lock()
        self.sweep_forwards = None  # (panel, exclusion forwards, removal scores) of the last sweep
        self.born: set[Path] = set()
        self.births_frozen = False
        self.sweep_backstop = SWEEP_BACKSTOP_BASE + SWEEP_BACKSTOP_PER_ROW * max(self.block_sizes)
        self.count_backstop = COUNT_BACKSTOP_BASE + COUNT_BACKSTOP_PER_ROW * max(self.block_sizes)
        # Canonical panel states held after the start and after every count change (count-cycle guard).
        self.count_states: set[PanelKey] = set()
        self.events: list[dict[str, Any]] = []
        self.refit_events: dict[str, list[dict[str, Any]]] | None = None
        # The count tallies (CountCensus fields; ints, flags and an optional census).
        self.count: dict[str, Any] = dict(
            decisions=0,
            births=0,
            pool_births=0,
            seed_births=0,
            seed_swaps=0,
            deaths=0,
            pure_mosaic_deaths=0,
            ambiguous=0,
            births_frozen=False,
            start_size=0,
            cycles=0,
            cycles_undecided=0,
            trap_indicators=0,
            seed_refits=0,
            ladder_births=0,
            final_paths_failing_death_test=None,
            state_revisits=0,
            count_backstop_reached=False,
        )
        # The start: the anchor-covering panel of largest likelihood (tie classes of the join log
        # likelihood, the panel's content inside a class). It only seeds the search; the count rule
        # decides the panel size.
        self.initialization = []
        # None until set_panel installs the start panel below (nothing reads it before).
        self.selected: Panel = cast("Panel", None)
        initial_panels = set()
        trials = []
        for anchor in range(self.blocks):
            seen = set()
            trial = []
            anchor_candidates = (
                candidates
                if self.initial_family is None
                else [
                    self.initial_family.first_covering(anchor, row)
                    for row in (
                        range(self.initial_family.founders[anchor] - 1, -1, -1)
                        if self.initial_family.reverse
                        else range(self.initial_family.founders[anchor])
                    )
                ]
            )
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
        best = self.class_winner(
            "assembly.join_start",
            trials,
            lambda item: item[0],
            lambda item: item[1],
            census_key="start",
            quantum_kind="join_ll_wide",
            link=lambda item: self.panel_link(item[1]),
            panel=lambda item: item[1],
        )
        self.set_panel(best[1], best[3], best[0])
        self.selected_anchor = best[2]
        self.count_states.add(tie_rules.panel_key(self.selected))
        self.start_size = len(self.selected)
        self.count["start_size"] = self.start_size
        # Panel size at the start and after every count change, and the changes that deleted a
        # path without sequence of its own (count_criterion.pre_cycle_count).
        self.sizes = [self.start_size]
        self.cleanup_steps: set[int] = set()
        self.static_totals = [self.layout.block(j).sum(axis=2) for j in range(self.blocks)]
        self._set_pool(self.original_pool)
        if self.initial_family is not None:
            self.events.append(
                dict(
                    stage="beam_covering_proposals",
                    represented_paths=self.initial_family.cardinality,
                    proposed_paths=len(candidates),
                    complete=False,
                )
            )

    # -- tie classes -----------------------------------------------------------------------------

    def _quantum(self, scale=None) -> float:
        """The join's tie quantum at the scale of ``scale`` (default: the current log likelihood)."""
        return ties.quantum("join_ll", self.likelihood if scale is None else scale)

    def _census(self, kind, key, values):
        if self.label is not None and ties.census_enabled(sink=self.tie_settings.census):
            with self.tie_lock:
                ties.census(kind, f"{self.label}/{key}", values, sink=self.tie_settings.census)

    def panel_link(self, panel):
        """The linker score of a panel (sum over its paths), or None without a linkage function."""
        if self.linkage is None:
            return None
        return float(sum(self.linkage(path) for path in panel))

    def class_order(
        self, kind, items, value, content, *, census_key=None, scale=None, quantum_kind="join_ll", link=None, panel=None
    ):
        """``items`` in tie-class order of value(item), descending (log likelihoods and gains: higher is
        better), content(item) deciding inside a class; returns (ordered items, TieClasses). The quantum
        is quantum_kind's at ``scale`` (join_ll: default the current log likelihood; join_proxy: default
        the largest |value|)."""
        items = list(items)
        values = [float(value(item)) for item in items]
        if scale is None and quantum_kind == "join_ll":
            scale = getattr(self, "likelihood", None)
        if scale is None:
            finite = [abs(v) for v in values if np.isfinite(v)]
            scale = max(finite) if finite else 0.0
        policy = search_tie_policy(kind, quantum_kind, self.tie_settings)
        quantum = policy.quantum(scale)
        motion = ties.motion(quantum_kind, scale)
        contents = [content(item) for item in items]
        result = ties.tie_classes(values, quantum, content=contents, descending=True, reverse=self.tie_settings.reverse)
        ties.add_decision(self.tie_counter, kind, result, values, quantum, motion, descending=True)
        if census_key is not None:
            # Census values: smaller is better (negated log likelihoods), the decision's candidates.
            self._census(kind, census_key, [-v for v in values])
        links = None
        if link is not None and self.linkage is not None and result.tied_winner:
            links = [link(item) for item in items]
        if result.tied_winner:
            # Fully symmetric for the data (log likelihood and linker score tied): decided by content, counted,
            # and the tied alternatives released in the search events.
            symmetric = tie_rules.symmetric_members(result, links, reverse=self.tie_settings.reverse)
            with self.tie_lock:
                self.tie_counter.note(kind + ".symmetric", inside_tie=bool(symmetric))
                if symmetric and panel is None:
                    append_event(
                        self,
                        dict(stage="symmetric_tie", kind=kind, alternatives=sorted(contents[m] for m in symmetric)),
                        census_key,
                    )
        order = tie_rules.linkage_order(result, links, contents, reverse=self.tie_settings.reverse)
        symmetric = tie_rules.symmetric_members(result, links, reverse=self.tie_settings.reverse)
        if panel is not None and symmetric:
            winner = self.prefer_panels(
                kind, [panel(items[i]) for i in symmetric], [values[i] for i in symmetric], census_key
            )
            chosen = symmetric[winner]
            order = [chosen] + [i for i in order if i != chosen]
        return [items[i] for i in order], result

    def class_winner(
        self, kind, items, value, content, *, census_key=None, scale=None, quantum_kind="join_ll", link=None, panel=None
    ):
        ordered, _ = self.class_order(
            kind,
            items,
            value,
            content,
            census_key=census_key,
            scale=scale,
            quantum_kind=quantum_kind,
            link=link,
            panel=panel,
        )
        return ordered[0] if ordered else None

    def prefer_panels(self, kind, panels, scores, source=None):
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
            with self.tie_lock:
                found = self._switch_counts.get(key)
            if found is None:
                self.claim()
                found = int(
                    objective.minimum_switches(*self.layout.arrays(), self.rows(key), self.penalty, self.samples).sum()
                )
                with self.tie_lock:
                    self._switch_counts[key] = found
            counts[key] = found
        chosen = min(counts, key=lambda key: (counts[key], key))
        if self.choice_plan is not None:
            chosen = self.choice_plan.choose(kind, source, distinct, chosen)
        family = dict(
            panels=distinct,
            reason=kind,
            source_decision=source,
            objective="join_ll",
            values=[float(scores[keys.index(key)]) for key in distinct],
            representative_prior="smaller_K_then_exact_minimum_switches",
            minimum_switches=[counts.get(key) for key in distinct],
            status="unresolved_search",
            unresolved_search=True,
        )
        with self.tie_lock:
            append_event(
                self,
                dict(
                    stage="symmetric_tie",
                    kind=kind,
                    alternatives=distinct,
                    representative_prior=family["representative_prior"],
                ),
                source,
            )
            if source is not None and re.search(r"/seed[0-9]+(?:/|$)", source) is not None:
                self.seed_families.setdefault(source.split("/p")[0].split("/pure")[0].split("/death")[0], []).append(
                    family
                )
            else:
                self.release_families.append(family)
        return keys.index(chosen)

    def set_panel(self, panel, scores, likelihood):
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

    def state_key(self, loop: _Loop) -> bytes:
        """SHA-256 of the complete search state at a sweep boundary: STATE_FIELDS and the loop's fields.

        The rest of the search is a deterministic function of this state (with its later serialized choices at
        their defaults): two executions with one key continue identically. Floats enter by their exact value, the
        pool by its ordered paths (digested once per pool list; pool lists are never changed in place).
        """
        digest = hashlib.sha256()
        for name in self.STATE_FIELDS:
            value = getattr(self, name)
            if name == "candidates":
                cached = getattr(self, "_pool_key", None)
                if cached is None or cached[0] is not value:
                    pool = np.asarray(value, dtype=np.int64).reshape(len(value), -1)
                    cached = (value, hashlib.sha256(pool.tobytes() + repr(pool.shape).encode()).digest())
                    self._pool_key = cached
                encoded = cached[1]
            else:
                encoded = repr(_canonical(value)).encode()
            digest.update(name.encode() + b"\0" + encoded + b"\0")
        digest.update(b"loop\0" + repr(dataclasses.astuple(loop)).encode())
        return digest.digest()

    # -- threads -------------------------------------------------------------------------------

    def claim(self) -> int:
        """Ask the caller for the current team at an existing kernel boundary."""
        return self._claim_threads()

    def concurrent(self, function, items):
        """Return independent seed results in input order using the caller's mapper.

        The default evaluates in order on the caller's current team. A stage may
        supply an in-process map that binds a child team for each item. This
        declared working set never changes a scientific candidate or decision.
        """
        task_bytes = (
            8 * self.samples * self.total_bins * (max(self.block_sizes) + 2)
            + (1 << 26)
            + 8 * self.samples * MOVES_PER_CHUNK
        )
        if self._map_tasks is None:
            return [function(item) for item in items]
        return list(self._map_tasks(function, items, task_bytes))

    # -- objective -----------------------------------------------------------------------------

    def rows(self, paths) -> np.ndarray:
        """int64[len(paths), n] local rows of a list of paths (beam rows are local rows)."""
        return np.asarray(paths, dtype=np.int64).reshape(-1, self.blocks)

    def _score(self, paths) -> np.ndarray:
        """Exact per-sample join scores of a panel (counted as one full score)."""
        with self._evaluations_lock:
            self.evaluations += 1
        self.claim()
        return objective.panel_scores(self.layout, self.rows(paths), self.penalty)

    def _paint(self, paths, penalty, reverse=None) -> np.ndarray:
        """Viterbi painting int32[S, total bins] of a panel at ``penalty`` (reverse: the traceback tie
        convention; None uses the caller-provided TieSettings)."""
        self.claim()
        return objective.painting(
            self.layout,
            self.rows(paths),
            penalty,
            self.config.tensor_budget_mb,
            self.claim,
            reverse=self.tie_settings.reverse if reverse is None else reverse,
        )

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

            beam = self.beam_runner(quota, overrides)
            if isinstance(beam, CartesianBeamFamily):
                beam = beam.covering()
            paths = dict.fromkeys(_paths(beam))
            self.beam_cache[key] = self.original_pool + [path for path in paths if path not in self.original_set]
        return self.beam_cache[key]

    def panel_forwards(self, panel):
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
            self.claim()
            computed = PanelForwards(self.layout, self.rows(panel), self.penalty, self.total_bins)
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

    def certify(self, *args, **kwargs):
        """A splice table is task-local; the stage declares its working set."""
        return self._certify_moves(*args, **kwargs)

    def _certify_moves(
        self,
        panel,
        pool,
        pool_offset,
        structural,
        pool_capacity,
        panel_ll=None,
        slots=None,
        refit_scores=None,
        key=None,
    ):
        """Bound every same-count move of N(P, pool) (or only replacements by pool paths).

        slots: a seed refit; only moves changing these rows, which are exempt from the
        novelty rule of same-count moves. refit_scores: the exact one-path scores shared by
        the seed refits of one count step. key: the pass's census key. The best move is chosen
        by tie classes of its exact change (the moved panel's content inside a class). Returns
        (best move or None, record, forwards)."""

        panel_ll = self.likelihood if panel_ll is None else panel_ll
        rows = np.asarray(panel, dtype=np.int64)
        moves = Moves(rows, pool_capacity)
        added = moves.add_pool(np.asarray(pool, dtype=np.int64).reshape(-1, self.blocks), pool_offset)
        if structural:
            moves.add_structural(self.block_sizes, slots)
        moves.finish()
        forwards = self.panel_forwards(panel)
        self.claim()
        bounds, sample_bounds = lower_bounds(moves, forwards, panel_ll, self.rows, self.block_offsets, replaced=slots)
        # A seed refit moves the seed and its copy freely (no novelty rule for any path while the
        # copies separate); its panel is settled by the count rule afterwards.
        exempt = frozenset() if slots is None else frozenset(range(len(panel)))
        if slots is not None:
            replacing = moves.kind == REPLACE
            bounds[replacing & ~np.isin(moves.first_removed, list(slots))] = np.inf
        combined = self.candidates if not pool_offset else self.candidates + list(pool)
        cache = {}
        sparse = dict(scores=0, samples=0)
        splice_work = dict(tables=0, scores=0)

        scored_changes = []

        def exact_move(trial, move):
            result = _exact_move(trial, move)
            if key is not None:
                scored_changes.append(result[1][0] - panel_ll)
            return result

        def _exact_move(trial, move):
            if moves.kind[move] == SPLICE:
                from .splice import pair_column, splice_scores

                if "splice" not in cache:
                    self.claim()

                    cache["splice"] = splice_scores(self.layout, rows, self.penalty)
                    splice_work["tables"] += 1
                _, first, second, boundary, _ = moves.canonical[move]
                column = pair_column(len(panel), int(first), int(second))
                values = cache["splice"][:, int(boundary) - 1, column].copy()
                splice_work["scores"] += 1
                trial_ll = float(values.sum())
                return -2.0 * trial_ll, (trial_ll, values, "splice")
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
            if "removal" not in cache:
                cache["removal"] = forwards.removal_scores()
            self.claim()
            # The bound's original slot is useful only for the same unique replacement.
            bound = sample_bounds.get(move) if replaced_row(moves, move) == row else None
            values, scored_samples = forwards.replacement_scores(
                row, self.rows(added)[0], self.rows(canonical), cache["removal"], bound
            )
            sparse["scores"] += 1
            sparse["samples"] += scored_samples
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
            return self.class_winner(
                "assembly.join_tier2",
                entries,
                lambda entry: -0.5 * entry[0],
                lambda entry: tie_rules.panel_key(entry[3]),
                census_key=key,
                scale=panel_ll,
                link=lambda entry: self.panel_link(entry[3]),
                panel=lambda entry: entry[3],
            )

        best_move, survivors, exact_scores = best_improvement(
            bounds,
            moves.canonical,
            lambda move: _materialize(moves.description(move), panel, combined),
            exact_move,
            -2.0 * panel_ll,
            signature=lambda trial: tuple(sorted(trial)),
            admissible=lambda trial: _admissible(panel, trial, self.geometry, self.near_identity, exempt),
            reach=2.0 * ties.WIDE_CLASS_QUANTA * q_ll,
            choose=choose,
            observe=observe if slots is None else None,
        )
        if best_move is not None and len(best_move[4]) == 3 and best_move[4][2] == "splice":
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
            self._census(
                "assembly.join_improve",
                f"{key}/t2",
                [_improvement_distance(panel_ll + gain, panel_ll) for gain in scored_changes],
            )
        record = dict(
            panel_size=len(panel),
            pool=len(combined),
            pool_paths_tested=added,
            neighbourhood=moves.counts(),
            moves=len(moves),
            survivors=int(survivors),
            exact_scores=int(exact_scores),
            sparse_exact_scores=sparse["scores"],
            sparse_exact_samples=sparse["samples"],
            splice_exact_scores=splice_work["scores"],
            splice_tables=splice_work["tables"],
            observed_plateau_neighbours=len(flat) if best_move is None else 0,
            recomputed_exclusions_per_sample=float(forwards.recomputed_per_sample),
            accepted=None if best_move is None else str(moves.description(best_move[2])),
            accepted_likelihood_change=(None if best_move is None else -0.5 * float(best_move[0])),
        )
        return best_move, record, forwards

    # -- tier 1 ----------------------------------------------------------------------------------

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
            forwards = self.panel_forwards(panel)
            self.sweep_forwards = (tuple(panel), forwards, forwards.removal_scores())
        self.claim()
        values, _ = self.sweep_forwards[1].replacement_scores(
            founder, self.rows(added)[0], self.rows(tie_rules.sort_panel(trial)), self.sweep_forwards[2]
        )
        return values, True

    def _painting_proposals(self, panel, local, eligible, convention):
        """The proposals of the painting-dependent generators (replace, local, one-sided, splices,
        switch splices, cycles) from the panel's Viterbi paintings with the traceback tie convention
        ``convention`` (_tier1_sweep merges both conventions). Returns (proposals by kind,
        local edits screened)."""
        k = len(panel)
        blocks = self.blocks
        candidates = self.candidates
        candidate_array = self.candidate_array
        proposals = {
            kind: [] for kind in ("replace", "local", "onesided", "splice", "switch_splice", "switch_cycle", "cycle")
        }
        offsets = self.layout.offsets
        painted = self._paint(panel, self.penalty, reverse=convention)
        fields = []
        corrections = []
        for b in range(blocks):
            painting = np.ascontiguousarray(painted[:, offsets[b] : offsets[b + 1]])
            unary, pair = conditional_fields(
                self.layout.block(b), painting, local[:, b], k, int(self.layout.founders[b])
            )
            fields.append(unary)
            corrections.append(pair)
        gains = np.zeros((k, len(candidates)))
        for b in range(blocks):
            gains += fields[b][:, candidate_array[:, b]]
        proposals["replace"] = replacement_candidates(gains, eligible)
        proposals["local"] = local_candidates(fields, local)
        local_screened = len(proposals["local"])
        exploratory = self._paint(panel, min(self.penalty, EXPLORATORY_PENALTY), reverse=convention)
        preference = np.zeros((k, k))
        pair_gain = np.zeros((k, k))
        for boundary in range(blocks - 1, 0, -1):
            preference += fields[boundary][:, local[:, boundary]]
            pair_gain += corrections[boundary]
            proposals["onesided"].extend(
                boundary_proposals(preference, boundary, None, "onesided", reverse=self.tie_settings.reverse)
            )
            delta = preference + preference.T + pair_gain
            proposals["splice"].extend(
                boundary_proposals(delta, boundary, None, "splice", reverse=self.tie_settings.reverse)
            )
            votes = boundary_vote_matrix(exploratory, int(offsets[boundary]), k)
            switch_gain = votes + votes.T - np.diag(votes)[:, None] - np.diag(votes)[None, :]
            proposals["switch_splice"].extend(
                boundary_proposals(switch_gain, boundary, None, "splice", reverse=self.tie_settings.reverse)
            )
            # Both tie conventions of the greedy and cycle permutations (an order-invariant union).
            for permutation in greedy_permutations(votes):
                if np.any(permutation != np.arange(k)):
                    gain = float(votes[np.arange(k), permutation].sum() - np.trace(votes))
                    proposals["switch_cycle"].append((gain, ("cycle", boundary, tuple(map(int, permutation)))))
            for permutation in cycle_permutations(preference):
                if np.any(permutation != np.arange(k)):
                    new = local.copy()
                    new[:, boundary:] = local[permutation, boundary:]
                    gain = 0.0
                    for b in range(boundary, blocks):
                        gain += fixed_paint_delta(
                            self.layout.block(b), painted[:, offsets[b] : offsets[b + 1]], local[:, b], new[:, b], k
                        )
                    proposals["cycle"].append((float(gain), ("cycle", boundary, tuple(map(int, permutation)))))
        del candidates
        return proposals, local_screened

    def _tier1_sweep(self, panel, panel_scores, panel_ll):
        """One bounded sweep at ``panel``: the best improving admissible move among the top
        proposals of every same-count kind. Returns (best or None, record)."""
        k = len(panel)
        blocks = self.blocks
        config = self.config
        quota = config.full_scores_per_kind

        sparse_scores = 0
        candidates = self.candidates
        candidate_array = self.candidate_array
        local = np.asarray(panel, dtype=np.int64)
        panel_set = set(panel)
        proposals = {kind: [] for kind in SAME_COUNT_KINDS}
        eligible = np.asarray([c for c, path in enumerate(candidates) if path not in panel_set], dtype=np.int64)
        proposal_counts = {}
        eligible_mask = np.zeros(len(candidates), dtype=bool)
        eligible_mask[eligible] = True
        novelty, novelty_scored = candidate_novelty(
            local, candidate_array, self.penalty, panel_scores, eligible_mask, workspace=self.candidate_workspace
        )
        self._sweeps_run += 1
        sweep_key = f"s{self._sweeps_run}"
        # Novel candidates: novelty beyond the band at its exact zero (a rounding-level positive novelty,
        # equal scores reached by another summation order, is no novelty; band = the join_proxy floor),
        # in tie-class order of their novelty (content: the candidate path), the class the budget cuts
        # kept whole (at most twice the budget).
        band = ties.quantum("join_proxy")
        # Operand replays observe the comprehension line events; keep these boundaries explicit.
        # fmt: off
        novel_items = [c for c in np.flatnonzero(eligible_mask).tolist()
                       if candidates[c] not in panel_set and novelty[c] > band]
        # fmt: on
        near = sorted(
            float(novelty[c]) for c in np.flatnonzero(eligible_mask).tolist() if 0.0 < novelty[c] <= 1e3 * band
        )
        with self.tie_lock:
            # The band at the exact zero of a non-novel candidate's novelty (symmetric candidates have exactly zero
            # novelty).
            self.tie_counter.note("assembly.join_novelty.band", inside_tie=bool(near and near[0] <= band))
        if near:
            self._census("assembly.join_novelty.band", f"{sweep_key}", near[:16])
        novel = []
        if novel_items:
            scale = float(max(novelty[c] for c in novel_items))
            values = [float(novelty[c]) for c in novel_items]
            quantum = ties.quantum("join_proxy", scale)
            result = ties.tie_classes(
                values,
                quantum,
                content=[candidates[c] for c in novel_items],
                descending=True,
                reverse=self.tie_settings.reverse,
            )
            ties.add_decision(
                self.tie_counter,
                "assembly.join_novelty",
                result,
                values,
                quantum,
                ties.motion("join_proxy", scale),
                descending=True,
                cut=quota,
            )
            novel = ties.truncate(novel_items, result, quota)
            self._census(
                "assembly.join_novelty", f"{sweep_key}", sorted(-float(novelty[c]) for c in novel_items)[: 4 * quota]
            )
        for c in novel:
            similarity = np.mean(local == candidate_array[c], axis=1)
            # the two most similar founders, with every founder tied with the second (an order-invariant set)
            if len(similarity) > 2:
                second = np.sort(similarity)[::-1][1]
                ranked = [int(f) for f in np.flatnonzero(similarity >= second)]
            else:
                ranked = list(range(len(similarity)))
            for founder in ranked:
                proposals["novel_replace"].append(
                    (float(novelty[c] * (0.5 + similarity[founder])), ("replace", int(founder), int(c)))
                )
        static_gains = static_replacement_gains(
            local, candidate_array, totals=self.static_totals, workspace=self.candidate_workspace
        )
        wide_quota = 2 * quota
        proposals["static_replace"] = replacement_candidates(static_gains, eligible)
        proposal_counts["replace"] = proposal_counts["static_replace"] = k * len(eligible)
        proposal_counts["onesided"] = (blocks - 1) * k * (k - 1)
        proposal_counts["splice"] = (blocks - 1) * k * (k - 1) // 2
        proposal_counts["switch_splice"] = proposal_counts["splice"]
        # Painting-dependent generators under both traceback tie conventions: the Viterbi
        # painting's choice among equal-score states is a convention, so the proposals of both paintings are
        # merged, the larger proxy of a repeated edit kept: the union does not depend on the order.
        merged = {}
        for convention in (False, True):
            generated, local_screened = self._painting_proposals(panel, local, eligible, convention)
            proposal_counts["local"] = local_screened
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
                ranked = items.ranked(quota, self.tie_counter, reverse=self.tie_settings.reverse)
            elif items:
                scale = max(abs(float(item[0])) for item in items)
                values = [float(item[0]) for item in items]
                quantum = ties.quantum("join_proxy", scale)
                result = ties.tie_classes(
                    values,
                    quantum,
                    content=[item[1] for item in items],
                    descending=True,
                    reverse=self.tie_settings.reverse,
                )
                ties.add_decision(
                    self.tie_counter,
                    "assembly.join_proposals",
                    result,
                    values,
                    quantum,
                    ties.motion("join_proxy", scale),
                    descending=True,
                    cut=quota,
                )
                ranked = ties.truncate(items, result, quota)
                if len(items) > quota:
                    self._census(
                        "assembly.join_proposals",
                        f"{sweep_key}/{kind}",
                        sorted(-float(item[0]) for item in items)[:wide_quota],
                    )
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
        # paths) inside a class.
        if improving_trials:
            best = self.class_winner(
                "assembly.join_tier1",
                improving_trials,
                lambda item: item[0],
                lambda item: item[5],
                census_key=f"{sweep_key}/winner",
                scale=panel_ll,
                link=lambda item: self.panel_link(item[1]),
                panel=lambda item: item[1],
            )[:5]
        if threshold_distances:
            # Valley census of the improvement threshold (a continuous threshold: the distances of the
            # evaluated trials' gains to it, in nats; positive = improving).
            self._census("assembly.join_improve", f"{sweep_key}/t1", threshold_distances)
        leading = sorted([(float(panel_ll), "current panel"), *evaluated], reverse=True)[:2]
        record = dict(
            panel_size=k,
            candidate_pool=len(candidates),
            proposals={kind: proposal_counts.get(kind, len(items)) for kind, items in proposals.items()},
            full_scores=self.evaluations - start_evaluations,
            sparse_scores=sparse_scores,
            novelty_scored=novelty_scored,
            likelihood=float(panel_ll),
            accepted=None if best is None else str(best[2]),
            full_scores_per_kind=quota,
            evaluated_by_kind=evaluated_by_kind,
            inadmissible=inadmissible,
            truncated_kinds=[
                kind for kind, items in proposals.items() if proposal_counts.get(kind, len(items)) > quota
            ],
            improving_at_budget_boundary=boundary_acceptances,
            leading_evaluated_alternatives=leading,
            scope="this sweep only; proxy-truncated and ungenerated edits are untested",
        )
        return best, record

    # -- the count step ---------------------------------------------------------------------------

    # -- the search --------------------------------------------------------------------------------

    def minimum_switch_count(self, panel=None) -> int:
        """Exact minimum switches of a requested finalist, cached on its canonical panel."""
        key = tie_rules.panel_key(self.selected if panel is None else panel)
        with self.tie_lock:
            found = self._switch_counts.get(key)
        if found is None:
            self.claim()
            found = int(
                objective.minimum_switches(*self.layout.arrays(), self.rows(key), self.penalty, self.samples).sum()
            )
            with self.tie_lock:
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

        family = observed_plateau_family(
            self.selected,
            self.likelihood,
            self._plateau_candidates,
            linkage=self.linkage,
            tie_settings=self.tie_settings,
        )
        if family is None:
            return False

        nominal = tie_rules.panel_key(self.selected)
        panels = family["panels"]
        eligible = [panel for panel in panels if panel == nominal or panel not in self._plateau_certified]
        excluded = [panel for panel in panels if panel not in eligible]
        self._plateau_certified.add(nominal)
        if len(eligible) < 2:
            # Every alternative is covered by an earlier source choice. Do not
            # create A->B->A replay cycles or erase those source families.
            self.events.append(
                dict(
                    stage="terminal_cycle_covered",
                    panel=nominal,
                    observed_panels=panels,
                    cycle_excluded_panels=excluded,
                )
            )
            return False
        self._terminal_choices += 1
        source = f"terminal_certification/{self._terminal_choices}"
        values = [family["values"][panels.index(panel)] for panel in eligible]
        selected = eligible[self.prefer_panels(family["reason"], eligible, values, source)]
        # The replay event omits already certified predecessors solely to prevent
        # flat cycles. The release record retains the COMPLETE observed family;
        # its eligible identity is consumed only when the whole tree completes.
        record = self.release_families[-1]
        record.update(
            family,
            source_decision=source,
            replay_panels=eligible,
            cycle_excluded_panels=excluded,
            minimum_switches=[self.minimum_switch_count(panel) for panel in panels],
            selected_panel=selected,
        )
        self.events.append(
            dict(
                stage="terminal_choice",
                source_decision=source,
                observed_panels=len(panels),
                replay_choices=len(eligible),
                cycle_excluded=len(excluded),
                continued=selected != nominal,
                representative_prior=record["representative_prior"],
                minimum_switches=record["minimum_switches"],
            )
        )
        if selected == nominal:
            return False
        visited = self._plateau_certified
        # This is a branch transition, not an improving move or count decision.
        # Reuse ordinary born-tag inheritance; keep count states, freeze flags,
        # accepted-change counters and all loop/backstop/quota state intact.
        scores = self._score(selected)
        apply_same_count(self, selected, scores)
        self._plateau_certified = visited
        return True

    def run(self, *, report_switches=False) -> JoinOutcome:
        """Search to a certified panel; optionally report the exact finalist switch prior."""
        outcome = self._search()
        if report_switches:
            # The outcome's panel: a merged replay's own panel is the one it held at the merge.
            outcome = replace(outcome, minimum_switches=self.minimum_switch_count(outcome.paths))
        return outcome

    def _search(self) -> JoinOutcome:
        config = self.config
        loop = _Loop(
            quota=config.paths_per_endpoint,
            ladder_top=config.paths_per_endpoint * 2**POOL_QUOTA_DOUBLINGS,
            quota_backstop=config.paths_per_endpoint * 2**POOL_EXTENDED_DOUBLINGS,
            largest_pool=len(self.candidates),
        )
        final_forwards = None
        while True:
            # A sweep boundary: the rest of the search reads only the state STATE_FIELDS and loop hold (AJ29).
            if self.merge is not None:
                merged = self.merge.boundary(self, loop)
                if merged is not None:
                    return merged
            if loop.phase_sweeps >= self.sweep_backstop:
                loop.backstop_reached = True
                break
            best, record = self._tier1_sweep(self.selected, self.baseline_scores, self.likelihood)
            record.update(sweep=loop.tier1_sweeps + 1)
            self.events.append(record)
            loop.tier1_sweeps += 1
            loop.sweeps += 1
            loop.phase_sweeps += 1
            if best is not None:
                apply_same_count(self, best[1], best[4])
                continue
            # A tier-1 stall: count decisions first (re-certification only once they are quiescent).
            stall_forwards = self.panel_forwards(self.selected)
            if count_step(self, stall_forwards):
                loop.phase_sweeps = 0
                continue
            # Tier 2 at the stall, on C0 plus the beam with the panel's multiplicity.
            pool_before = len(self.candidates)
            pool = self._extension_pool(loop.quota, self.selected)
            # A fresh list with identical ordered content adds no count evidence.
            if pool != self.candidates:
                self._set_pool(pool)
                loop.largest_pool = max(loop.largest_pool, len(self.candidates))
                # The pool grew: births by its new paths are count moves of this stall too.
                if count_step(self, stall_forwards):
                    loop.phase_sweeps = 0
                    continue
            loop.tier2_passes += 1
            best_move, record, forwards = self.certify(
                self.selected, self.candidates, 0, True, self.capacity, key=f"t2/{loop.tier2_passes}"
            )
            record.update(
                tier=2,
                tier2_pass=loop.tier2_passes,
                quota=loop.quota,
                pool_before=pool_before,
            )
            self.events.append(record)
            if best_move is not None:
                loop.tier2_accepted += 1
                loop.sweeps += 1
                loop.phase_sweeps += 1
                apply_same_count(self, best_move[3], best_move[4][1])
                continue
            # Same-count certified and count-quiescent: duplicate-seed births (the chimera trap).
            if count_step(self, forwards, seeds=True):
                loop.phase_sweeps = 0
                continue
            # Pool saturation: certify only when no doubled quota up to the ladder top adds an
            # improving replacement or accepted birth. The top is 256 for every join; once a
            # move was accepted at the 256 rung (the quota reached it) the ladder climbs on to
            # the backstop.
            if loop.quota >= loop.ladder_top:
                loop.ladder_top = loop.quota_backstop
            if 2 * loop.quota > loop.ladder_top:
                loop.pool_backstop = True
                final_forwards = forwards
                break
            checked = set(self.candidates)
            level = loop.quota
            accepted = False
            while not accepted and 2 * level <= loop.ladder_top:
                level *= 2
                loop.saturation_steps += 1
                doubled = [path for path in self._extension_pool(level, self.selected) if path not in checked]
                best_move, record, level_forwards = self.certify(
                    self.selected,
                    doubled,
                    len(self.candidates),
                    False,
                    len(self.candidates) + len(doubled),
                    key=f"sat{loop.saturation_steps}",
                )
                checked.update(doubled)
                record.update(
                    stage="pool_saturation",
                    quota=loop.quota,
                    tested_quota=level,
                    new_paths=len(doubled),
                )
                self.events.append(record)
                if best_move is not None:
                    apply_same_count(self, best_move[3], best_move[4][1])
                    accepted = True
                elif doubled and count_step(self, level_forwards, pool=doubled):
                    self.count["ladder_births"] += 1
                    accepted = True
            if not accepted:
                if self._continue_terminal_family():
                    # No fresh sweep/backstop/quota budget and no improvement
                    # counter. The chosen neighbor must pass its own continuation.
                    continue
                loop.certified = True
                final_forwards = forwards
                break
            loop.saturation_accepted += 1
            loop.sweeps += 1
            loop.phase_sweeps = 0
            loop.quota = level
            self._set_pool(self._extension_pool(loop.quota, self.selected))
            loop.largest_pool = max(loop.largest_pool, len(self.candidates))
        self.count["births_frozen"] = self.births_frozen
        if final_forwards is not None:
            self.count["final_paths_failing_death_test"] = final_census(self, final_forwards)
        summary = JoinSummary(
            final_panel_size=len(self.selected),
            final_likelihood=float(self.likelihood),
            total_full_scores=self.evaluations,
            initialization=tuple(self.initialization),
            selected_initial_anchor=self.selected_anchor,
            sweeps=loop.sweeps,
            tier1_sweeps=loop.tier1_sweeps,
            sweep_backstop=self.sweep_backstop,
            tier2_passes=loop.tier2_passes,
            tier2_accepted=loop.tier2_accepted,
            saturation_steps=loop.saturation_steps,
            saturation_accepted=loop.saturation_accepted,
            quota=loop.quota,
            ladder_top=loop.ladder_top,
            quota_backstop=loop.quota_backstop,
            count=CountCensus(**self.count),
            count_exact_scores=self.counter.exact_scores,
            count_exact_sample_scores=self.counter.exact_sample_scores,
            pool_original=len(self.original_pool),
            pool_final=len(self.candidates),
            pool_largest=loop.largest_pool,
            certified=loop.certified,
            backstop_reached=loop.backstop_reached,
            pool_backstop=loop.pool_backstop,
            events=tuple(_plain(event) for event in self.events),
            tie_events=self.tie_counter.record(),
            work_counters={"beam_materialization_bounds": 0},
        )
        return JoinOutcome(
            paths=np.asarray(self.selected, dtype=np.int32).reshape(-1, self.blocks),
            log_likelihood=float(self.likelihood),
            summary=summary,
            # JoinOutcome.__post_init__ types these family mappings (diagnostics.join_family).
            release_alternatives=cast("tuple[ReleaseFamily, ...]", tuple(self.release_families)),
        )


def search_panel(
    inputs: JoinInputs,
    beam: Beam,
    beam_runner: BeamRunner,
    *,
    config: JoinParameters,
    count_rule: CountRuleParameters,
    cost_scale: float,
    claim_threads: Callable[[], int] | None = None,
    map_tasks: Callable[[Callable, list, int], list] | None = None,
    label: str | None = None,
    counter: ties.TieCounter | None = None,
    linkage: Callable[[Path], float] | None = None,
    report_switches: bool = False,
    max_branches: int = MAX_BRANCHES,
    tie_settings: ties.TieSettings = DEFAULT_TIE_SETTINGS,
    replay_merge: str = "on",
) -> JoinOutcome:
    """The certified panel of one joined batch.

    inputs: JoinInputs (n >= 2 input components, the informative samples' float32 evidence).
    beam: the batch's endpoint beam at quota ``config.paths_per_endpoint``, [(path, score)] with
        paths of local rows (one per input component). beam_runner(quota, overrides): the
        batch's endpoint beam re-run with a base quota and per-endpoint quotas
        ({(block, row): quota}), same format (pool extension and saturation rungs).
    config: JoinParameters; count_rule: CountRuleParameters; cost_scale: the per-founder cost scale
        (count_criterion.founder_cost). claim_threads binds the current team at kernel boundaries;
        map_tasks binds independent seed teams and returns results in input order. label: the census label (None:
        no census); counter: the batch's core.ties.TieCounter (shared with its beam; recorded as the
        summary's tie_events). max_branches: complete original-input executions, at
        most four including the primary; use one when an outer joint input family
        already consumes the branch budget. report_switches: return the exact final
        minimum-switch prior even if the terminal panel is unique. replay_merge: "on" (a replay whose
        complete search state equals an earlier execution's at a sweep boundary adopts that execution's
        continuation; its records are those of the full replay, its work and tie counters count only what ran),
        "off" (every execution runs to its end), or "verify" (every execution runs to its end and each state
        coincidence's adopted continuation is checked against it; alternatives.ReplayMergeMismatch otherwise).
    """
    from .alternatives import ReplayMerge, resolve_replays, run_bounded_replays

    shared_layout = None
    last_search = None
    shared_counter = counter if counter is not None else ties.SharedCounter()
    if replay_merge not in ("on", "off", "verify"):
        raise ValueError(f"unknown replay merge mode: {replay_merge}")
    # One execution has nothing to merge with.
    merge = ReplayMerge(verify=replay_merge == "verify") if replay_merge != "off" and max_branches > 1 else None

    def run(plan):
        nonlocal shared_layout, last_search
        # Only emissions are immutable shared state. Born paths, count cycles,
        # candidate pools, histories and caches start from the original inputs.
        last_search = None
        last_search = PanelSearch(
            inputs,
            beam,
            beam_runner,
            config=config,
            count_rule_config=count_rule,
            cost_scale=cost_scale,
            claim_threads=claim_threads,
            map_tasks=map_tasks,
            label=label,
            counter=shared_counter,
            linkage=linkage,
            choice_plan=plan,
            shared_layout=shared_layout,
            tie_settings=tie_settings,
            merge=merge,
        )
        shared_layout = last_search.layout
        try:
            outcome = last_search.run()
            if merge is not None:
                merge.complete(last_search, outcome)
            return outcome
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
        if claim_threads is not None:
            claim_threads()
        site_counts = [len(block.positions) for block in inputs.blocks]
        return int(
            objective.minimum_switches(
                *shared_layout.arrays(),
                np.ascontiguousarray(panel, dtype=np.int64),
                emissions.switch_penalty(site_counts),
                int(inputs.sample_count),
            ).sum()
        )

    return resolve_replays(
        replay,
        minimum_switches,
        report_switches=report_switches,
        tie_events=shared_counter.record(),
        tie_settings=tie_settings,
    )
