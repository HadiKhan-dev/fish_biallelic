"""Bounded original-input replays of encountered join choices.

A replay forces a complete prefix of serialized decision-point choices, preserving the
ordinary search's count, born-path and cycle semantics. At most four executions include
the primary run. Inner concurrent seed choices remain source ambiguity; this planner
continues observed certification neighbors, without enumerating entire plateaus or
nearest-shadow count outcomes.

A replay whose complete search state (PanelSearch.state_key) equals an earlier execution's at a sweep
boundary, both past their forced prefixes, continues exactly as that execution continued from there: the
search is deterministic and every later serialized choice takes its default. ReplayMerge stops it there and
adopts that continuation, so its records are those the full replay would have produced; only its work and tie
counters (what actually ran) and the merge diagnostics differ.

Owns: bounded original-input replay plans, their state merges and their releases.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: docs/methods/assembly.md#aj29.
Invariant: Replay is FIFO within MAX_BRANCHES; seed-refit choices are never independently forced; a merged
replay's records equal its full replay's.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from haplotype_reconstruction.contracts.roles import record_field
from haplotype_reconstruction.core import ties

from .. import orders as tie_rules
from ..parameters import MAX_BRANCHES
from .state import DEFAULT_TIE_SETTINGS


@dataclass(frozen=True)
class PanelChoice:
    kind: str = record_field("state", axes=(), units="1", dtype="str")
    source: str | None = record_field("state", axes=(), units="1", dtype="str | None")
    panels: tuple = record_field("state", axes=(), units="1", dtype="tuple")
    selected: tuple = record_field("state", axes=(), units="1", dtype="tuple")

    @property
    def family(self):
        return self.kind, self.source, self.panels

    def record(self):
        return dict(kind=self.kind, source=self.source, panels=self.panels, selected=self.selected)


class ReplayMismatch(RuntimeError):
    """An original-input replay did not encounter its expected decision prefix."""


class ReplayChoices:
    """One execution's forced prefix and actual serial choice transcript."""

    def __init__(self, forced=()):
        self.forced = tuple(forced)
        self.observed = []
        self.excluded = []

    def choose(self, kind, source, panels, default):
        panels = tuple(sorted({tie_rules.panel_key(panel) for panel in panels}))
        default = tie_rules.panel_key(default)
        if source is not None and re.search(r"/seed[0-9]+(?:/|$)", source) is not None:
            # These choices occur within independent concurrent refits; forcing an
            # execution-order prefix there would depend on thread scheduling.
            self.excluded.append(PanelChoice(kind, source, panels, default))
            return default
        at = len(self.observed)
        selected = default
        if at < len(self.forced):
            expected = self.forced[at]
            if expected.family != (kind, source, panels):
                raise ReplayMismatch(f"join replay prefix differs at {kind}:{source}")
            selected = expected.selected
        choice = PanelChoice(kind, source, panels, selected)
        self.observed.append(choice)
        return selected

    def finish(self):
        if len(self.observed) < len(self.forced):
            raise ReplayMismatch("join replay ended before its forced prefix was encountered")


@dataclass(frozen=True)
class ReplayBranch:
    outcome: object = record_field("state", axes=(), units="1", dtype="object")
    choices: tuple[PanelChoice, ...] = record_field("state", axes=(), units="1", dtype="tuple[PanelChoice, ...]")
    forced_choices: int = record_field("state", axes=(), units="1", dtype="int64")


@dataclass(frozen=True)
class ReplayResult:
    branches: tuple[ReplayBranch, ...] = record_field("state", axes=(), units="1", dtype="tuple[ReplayBranch, ...]")
    unresolved_families: tuple[dict, ...] = record_field("state", axes=(), units="1", dtype="tuple[dict, ...]")
    excluded_families: tuple[dict, ...] = record_field("state", axes=(), units="1", dtype="tuple[dict, ...]")
    complete: bool = record_field("state", axes=(), units="1", dtype="bool")
    attempted: int = record_field("state", axes=(), units="1", dtype="int64")
    failures: tuple[str, ...] = record_field("state", axes=(), units="1", dtype="tuple[str, ...]")

    def diagnostics(self):
        return dict(
            scope="encountered serialized join panel choices",
            complete=self.complete,
            attempted=self.attempted,
            completed=len(self.branches),
            failures=list(self.failures),
            unresolved_families=len(self.unresolved_families),
            excluded_seed_families=len(self.excluded_families),
        )


def _family_records(choices, status):
    distinct = {}
    for choice in choices:
        distinct.setdefault(choice.family, choice)
    return tuple(
        dict(
            reason=choice.kind,
            source_decision=choice.source,
            panels=choice.panels,
            status=status,
            unresolved_search=True,
            representative_prior="smaller_K_then_exact_minimum_switches",
        )
        for choice in sorted(distinct.values(), key=lambda item: (item.kind, str(item.source), item.panels))
    )


def run_bounded_replays(run, *, max_branches=MAX_BRANCHES):
    """Run ``run(ReplayChoices) -> outcome`` from original inputs for each explored prefix.

    ``run`` may reuse the first execution's immutable emission layout. It must construct
    fresh semantic search state; no mutable search object or candidate history is reused.
    Return completed outcomes without selecting among different K by raw likelihood.
    ``complete`` concerns the encountered serialized choice tree only; excluded inner
    seed families and other source ambiguities remain explicitly outside this claim.
    """
    if not 1 <= max_branches <= MAX_BRANCHES:
        raise ValueError(f"join replay budget must be between 1 and {MAX_BRANCHES}")

    pending = [()]
    queued = {()}
    visited = set()
    branches, excluded, failed_choices, failures = [], [], [], []
    attempted = 0
    while pending and attempted < max_branches:
        # Do not evaluate a convenient subset of a family already known not to fit.
        if branches and len(branches) + len(pending) > max_branches:
            break
        forced = pending.pop(0)
        queued.discard(forced)
        if forced in visited:
            continue
        plan = ReplayChoices(forced)

        attempted += 1
        try:
            outcome = run(plan)
            plan.finish()
        except ReplayMismatch as error:
            failures.append(str(error))
            failed_choices.extend(forced[-1:])
            # A changed prefix invalidates this family comparison. Preserve all
            # unrun choices, rather than trying more mismatched replays.
            break
        choices = tuple(plan.observed)
        branches.append(ReplayBranch(outcome, choices, len(forced)))
        excluded.extend(plan.excluded)
        visited.update(choices[:stop] for stop in range(len(choices) + 1))
        for at, choice in enumerate(choices):
            for panel in choice.panels:
                if panel == choice.selected:
                    continue
                alternate = PanelChoice(choice.kind, choice.source, choice.panels, panel)
                prefix = (*choices[:at], alternate)
                if prefix not in visited and prefix not in queued:
                    pending.append(prefix)
                    queued.add(prefix)
    remaining = [prefix[-1] for prefix in pending if prefix] + failed_choices
    return ReplayResult(
        tuple(branches),
        _family_records(remaining, "unresolved_replay_budget"),
        _family_records(excluded, "unresolved_seed_continuation"),
        not pending and not failures,
        attempted,
        tuple(failures),
    )


class ReplayMergeMismatch(RuntimeError):
    """Verify mode: a continuation adopted at a state coincidence differs from the full replay's."""


@dataclass(frozen=True)
class _Boundary:
    """Where one execution stood at a sweep boundary: the lengths of its append-only records (search events,
    release families, observed and excluded serialized choices) and its sweeps."""

    events: int
    families: int
    observed: int
    excluded: int
    sweeps: int


@dataclass(frozen=True)
class _Execution:
    """A completed execution: its outcome and its whole serialized choice transcript."""

    outcome: object
    observed: tuple
    excluded: tuple


# The JoinSummary fields of a merged replay by origin. continuation: the adopted execution's final values (the
# rest of the search is a function of the boundary state, which holds every counter these report); own: fixed
# at the replay's construction; concatenated: the replay's records up to the boundary, then the adopted
# execution's records after it; diagnostic: work and tie counters of what actually ran (and merge counters).
MERGED_SUMMARY_FIELDS = dict(
    continuation=(
        "final_panel_size",
        "final_likelihood",
        "sweeps",
        "tier1_sweeps",
        "sweep_backstop",
        "tier2_passes",
        "tier2_accepted",
        "saturation_steps",
        "saturation_accepted",
        "quota",
        "ladder_top",
        "quota_backstop",
        "count",
        "pool_original",
        "pool_final",
        "pool_largest",
        "certified",
        "backstop_reached",
        "pool_backstop",
    ),
    own=("initialization", "selected_initial_anchor"),
    concatenated=("events",),
    diagnostic=("total_full_scores", "count_exact_scores", "count_exact_sample_scores", "tie_events", "work_counters"),
)


class ReplayMerge:
    """Merge bounded replays into earlier executions at equal complete search states (AJ29).

    PanelSearch calls boundary(search, loop) at every sweep boundary and search_panel calls complete(search,
    outcome) when an execution ends. Boundaries are keyed only once the execution has made a serialized choice
    and consumed its forced prefix (before that it can adopt no continuation, and none of its continuations can
    be adopted: a merging replay's later choices are all defaults). A replay reaching a key that a completed
    execution held stops there: its transcript continues with that execution's later choices, its records with
    that execution's later records, its final panel, likelihood, counters and families are that execution's.
    verify: run every execution to its end and check, at complete, each coincidence's adopted continuation
    against the actual one (ReplayMergeMismatch otherwise; ``verified`` counts the checks).
    """

    def __init__(self, *, verify: bool = False) -> None:
        self.verify = verify
        self.verified = 0
        self._known: dict[bytes, tuple[int, _Boundary]] = {}
        self._executions: list[_Execution] = []
        self._search = None
        self._pending: list[tuple[bytes, _Boundary]] = []
        self._matches: list[tuple[_Boundary, int, _Boundary]] = []

    def boundary(self, search, loop):
        """The merged outcome when this boundary's state was an earlier execution's, else None (continue)."""
        plan = search.choice_plan
        if plan is None:
            return None
        if search is not self._search:
            self._search, self._pending, self._matches = search, [], []
        if len(plan.observed) < max(1, len(plan.forced)):
            return None
        key = search.state_key(loop)
        here = _Boundary(
            len(search.events), len(search.release_families), len(plan.observed), len(plan.excluded), loop.sweeps
        )
        self._pending.append((key, here))
        found = self._known.get(key)
        if found is None:
            return None
        if self.verify:
            self._matches.append((here, *found))
            return None
        index, there = found
        target = self._executions[index]
        plan.observed.extend(target.observed[there.observed :])
        plan.excluded.extend(target.excluded[there.excluded :])
        return _merged_outcome(search, target.outcome, there)

    def complete(self, search, outcome) -> None:
        """Record a finished (or merged) execution's boundaries; in verify mode check its coincidences first."""
        if search is not self._search:
            self._pending, self._matches = [], []
        plan = search.choice_plan
        execution = _Execution(outcome, tuple(plan.observed), tuple(plan.excluded))
        for here, index, there in self._matches:
            _verify_merge(execution, here, self._executions[index], there)
            self.verified += 1
        number = len(self._executions)
        self._executions.append(execution)
        for key, here in self._pending:
            self._known.setdefault(key, (number, here))
        self._search, self._pending, self._matches = None, [], []


def _merged_outcome(search, target, there):
    """The outcome of a replay that holds ``target``'s state at boundary ``there``: target's continuation after
    the replay's own records; work and tie counters of what ran, with the merge counted."""
    from .diagnostics import JoinOutcome, _plain

    summary = target.summary
    work = {name: value for name, value in summary.work_counters.items() if not name.startswith("replay_merge")}
    summary = replace(
        summary,
        initialization=tuple(search.initialization),
        selected_initial_anchor=search.selected_anchor,
        events=(*(_plain(event) for event in search.events), *summary.events[there.events :]),
        total_full_scores=search.evaluations,
        count_exact_scores=search.counter.exact_scores,
        count_exact_sample_scores=search.counter.exact_sample_scores,
        tie_events=search.tie_counter.record(),
        work_counters=dict(work, replay_merges=1, replay_merged_sweeps=summary.sweeps - there.sweeps),
    )
    return JoinOutcome(
        paths=np.array(target.paths, copy=True),
        log_likelihood=float(target.log_likelihood),
        summary=summary,
        release_alternatives=(*search.release_families, *target.release_alternatives[there.families :]),
        minimum_switches=target.minimum_switches,
    )


def _comparable(value):
    """An exactly comparable form of a record value (floats by their bits, arrays with dtype and shape)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        named = tuple((field.name, _comparable(getattr(value, field.name))) for field in dataclasses.fields(value))
        return (type(value).__name__, named)
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, _comparable(value.tolist()))
    if isinstance(value, Mapping):
        return ("map", tuple((key, _comparable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_comparable(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return ("float", float(value).hex())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _verify_merge(execution, here, target, there):
    """The continuation ``target`` showed from ``there`` against the one ``execution`` actually ran from ``here``."""
    actual, adopted = execution.outcome, target.outcome
    pairs = dict(
        paths=(actual.paths, adopted.paths),
        log_likelihood=(actual.log_likelihood, adopted.log_likelihood),
        minimum_switches=(actual.minimum_switches, adopted.minimum_switches),
        events=(actual.summary.events[here.events :], adopted.summary.events[there.events :]),
        release_alternatives=(
            actual.release_alternatives[here.families :],
            adopted.release_alternatives[there.families :],
        ),
        choices=(execution.observed[here.observed :], target.observed[there.observed :]),
        excluded_choices=(execution.excluded[here.excluded :], target.excluded[there.excluded :]),
        **{
            f"summary.{name}": (getattr(actual.summary, name), getattr(adopted.summary, name))
            for name in MERGED_SUMMARY_FIELDS["continuation"]
        },
    )
    differ = [name for name, (a, b) in pairs.items() if _comparable(a) != _comparable(b)]
    if differ:
        raise ReplayMergeMismatch(
            f"adopted continuation differs from the full replay in {differ} (sweep {here.sweeps})"
        )


def _family_key(family):
    return (
        family.get("reason"),
        family.get("source_decision"),
        tuple(sorted({tie_rules.panel_key(panel) for panel in family.get("replay_panels", family.get("panels", ()))})),
    )


def resolve_replays(
    replay,
    minimum_switches,
    *,
    report_switches=False,
    tie_events=None,
    tie_settings: ties.TieSettings = DEFAULT_TIE_SETTINGS,
):
    """Compare terminal join panels while retaining unexamined source histories.

    ``minimum_switches(paths)`` evaluates the exact prior on this join's shared layout.
    Terminal likelihoods are freshly classified on a common scale within each K; raw
    likelihood cannot choose between different counts. A complete replay consumes only
    the serialized choice families it actually continued. Candidate-internal seed
    records are promoted by PanelSearch only for the selected endpoint's complete
    class of accepted seed witnesses, so the planner's full
    excluded-seed diagnostics must not themselves become released ambiguity.
    """
    from ..alternatives import terminal_winners

    outcomes = [branch.outcome for branch in replay.branches]
    survivors = terminal_winners(outcomes, rows=lambda outcome: outcome.paths, reverse=tie_settings.reverse)
    survivor_ids = {id(outcome) for outcome in survivors}
    if len(survivors) > 1 or report_switches:
        survivors = tuple(
            replace(
                outcome,
                minimum_switches=(
                    outcome.minimum_switches
                    if outcome.minimum_switches is not None
                    else int(minimum_switches(outcome.paths))
                ),
            )
            for outcome in survivors
        )
    chosen = min(
        survivors,
        key=lambda outcome: (
            len(outcome.paths),
            outcome.minimum_switches if outcome.minimum_switches is not None else 0,
            tie_rules.panel_key(outcome.paths),
        ),
    )

    certified = all(outcome.summary.certified for outcome in outcomes)
    consumed = {choice.family for branch in replay.branches for choice in branch.choices}
    families = []
    unverified = []
    for outcome in outcomes:
        for family in outcome.release_alternatives:
            if family.get("count_unverified") or family.get("status") == "count_unverified":
                # Computational work left undone is neither a panel choice nor
                # evidence for a count verdict. Preserve each branch's metadata
                # verbatim; never consume it using a biological-choice identity.
                unverified.append(family)
                continue
            if id(outcome) not in survivor_ids and not family.get("unresolved_search", False):
                continue
            if replay.complete and certified and _family_key(family) in consumed:
                continue
            families.append(family)
    if not replay.complete:
        # Preserve the complete domain, including alternatives not executed because
        # the joint tree exceeded the bound or its forced prefix did not reproduce.
        families.extend(replay.unresolved_families)
    distinct: dict[tuple, dict] = {}
    for family in families:
        key = (_family_key(family), bool(family.get("unresolved_search", False)), family.get("status"))
        distinct.setdefault(key, family)
    families = [*distinct.values(), *unverified]
    if len(survivors) > 1:
        families.append(
            dict(
                reason="terminal_join_continuations",
                source_decision="join_replay",
                panels=[tie_rules.panel_key(outcome.paths) for outcome in survivors],
                objective="join_ll",
                values=[float(outcome.log_likelihood) for outcome in survivors],
                minimum_switches=[outcome.minimum_switches for outcome in survivors],
                status=(
                    "unresolved_join_continuation"
                    if not certified
                    else "ambiguous_k"
                    if len({len(outcome.paths) for outcome in survivors}) > 1
                    else "same_k_equal_LL"
                ),
                unresolved_search=not certified,
                representative_prior="smaller_K_then_exact_minimum_switches",
                source_replay_complete=replay.complete,
                source_search_certified=certified,
                source_choices=[[choice.record() for choice in branch.choices] for branch in replay.branches],
            )
        )

    diagnostic = dict(
        stage="join_replay",
        **replay.diagnostics(),
        branch_K=[len(outcome.paths) for outcome in outcomes],
        branch_LL=[float(outcome.log_likelihood) for outcome in outcomes],
        survivors=len(survivors),
        certified=certified,
        resolved=replay.complete and certified and len(survivors) == 1 and not families,
    )
    merged = [outcome.summary.work_counters.get("replay_merges", 0) for outcome in outcomes]
    if any(merged):
        # Replays that adopted an earlier execution's continuation at an equal search state (ReplayMerge).
        diagnostic.update(
            merged_branches=merged,
            merged_sweeps=[outcome.summary.work_counters.get("replay_merged_sweeps", 0) for outcome in outcomes],
        )
    # Biological census/acceptance events describe the selected complete history.
    # Work totals account for every executed history, and the replay event reports
    # branch work separately without mixing their count-state semantics.
    summary = replace(
        chosen.summary,
        total_full_scores=sum(outcome.summary.total_full_scores for outcome in outcomes),
        count_exact_scores=sum(outcome.summary.count_exact_scores for outcome in outcomes),
        count_exact_sample_scores=sum(outcome.summary.count_exact_sample_scores for outcome in outcomes),
        work_counters={
            name: sum(outcome.summary.work_counters.get(name, 0) for outcome in outcomes)
            for name in {key for outcome in outcomes for key in outcome.summary.work_counters}
        },
        tie_events=chosen.summary.tie_events if tie_events is None else tie_events,
        events=(*chosen.summary.events, diagnostic),
    )
    return replace(chosen, summary=summary, release_alternatives=tuple(families))


def observed_plateau_family(
    panel, likelihood, observed, *, linkage=None, tie_settings: ties.TieSettings = DEFAULT_TIE_SETTINGS
):
    """Uncontinued equal-data neighbors actually scored at a final certification.

    This does not enumerate a plateau or assert that neighbors passed their own
    count decisions. PanelSearch applies the exact prior to the eligible choices
    and continues the selected panel before returning. Unrun choices remain source
    ambiguity until the complete bounded replay tree has evaluated them.
    """
    from haplotype_reconstruction.core import ties

    nominal = tie_rules.panel_key(panel)
    candidates = dict(observed)
    candidates[nominal] = (float(likelihood), "terminal_representative")
    if len(candidates) < 2:
        return None
    panels = sorted(candidates)
    values = [float(candidates[key][0]) for key in panels]
    scale = max(abs(value) for value in values)
    classes = ties.tie_classes(
        values,
        ties.quantum("join_ll", scale),
        motion=ties.motion("join_ll", scale),
        content=panels,
        descending=True,
        reverse=tie_settings.reverse,
    )
    links = None if linkage is None else [sum(linkage(path) for path in key) for key in panels]
    members = sorted(tie_rules.symmetric_members(classes, links, reverse=tie_settings.reverse))
    if panels.index(nominal) not in members:
        return None
    return dict(
        reason="terminal_certification_plateau",
        source_decision="terminal_certification",
        panels=[panels[index] for index in members],
        objective="join_ll",
        values=[values[index] for index in members],
        source_passes=[candidates[panels[index]][1] for index in members],
        status="unresolved_plateau_continuation",
        unresolved_search=True,
        enumeration_scope="exact_scored_certification_neighbours",
        representative_prior="smaller_K_then_exact_minimum_switches",
    )


__all__ = [
    "MERGED_SUMMARY_FIELDS",
    "PanelChoice",
    "ReplayMerge",
    "ReplayMergeMismatch",
    "ReplayMismatch",
    "ReplayChoices",
    "ReplayBranch",
    "ReplayResult",
    "run_bounded_replays",
    "resolve_replays",
    "observed_plateau_family",
]
