"""Bounded original-input replays of encountered join choices.

A replay forces a complete prefix of serialized decision-point choices, preserving the
ordinary search's count, born-path and cycle semantics. At most four executions include
the primary run. Inner concurrent seed choices remain source ambiguity; this planner
continues observed certification neighbors, without enumerating entire plateaus or
nearest-shadow count outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
import time

from .. import tie_rules

# Approved joint-branch budget, including the primary search.
MAX_BRANCHES = 4


@dataclass(frozen=True)
class PanelChoice:
    kind: str
    source: str | None
    panels: tuple
    selected: tuple

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
    outcome: object
    choices: tuple[PanelChoice, ...]
    seconds: float
    forced_choices: int


@dataclass(frozen=True)
class ReplayResult:
    branches: tuple[ReplayBranch, ...]
    unresolved_families: tuple[dict, ...]
    excluded_families: tuple[dict, ...]
    complete: bool
    attempted: int
    seconds: float
    failures: tuple[str, ...]

    def diagnostics(self):
        return dict(scope="encountered serialized join panel choices", complete=self.complete,
                    attempted=self.attempted, completed=len(self.branches),
                    branch_seconds=[branch.seconds for branch in self.branches],
                    seconds=self.seconds, failures=list(self.failures),
                    unresolved_families=len(self.unresolved_families),
                    excluded_seed_families=len(self.excluded_families))


def _family_records(choices, status):
    distinct = {}
    for choice in choices:
        distinct.setdefault(choice.family, choice)
    return tuple(dict(reason=choice.kind, source_decision=choice.source, panels=choice.panels,
                      status=status, unresolved_search=True,
                      representative_prior="smaller_K_then_exact_minimum_switches")
                 for choice in sorted(distinct.values(), key=lambda item: (item.kind, str(item.source), item.panels)))


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
    started = time.perf_counter()
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
        branch_started = time.perf_counter()
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
        elapsed = time.perf_counter() - branch_started
        choices = tuple(plan.observed)
        branches.append(ReplayBranch(outcome, choices, elapsed, len(forced)))
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
    return ReplayResult(tuple(branches), _family_records(remaining, "unresolved_replay_budget"),
                        _family_records(excluded, "unresolved_seed_continuation"),
                        not pending and not failures, attempted,
                        time.perf_counter() - started, tuple(failures))


def _family_key(family):
    return (family.get("reason"), family.get("source_decision"),
            tuple(sorted({tie_rules.panel_key(panel) for panel in
                          family.get("replay_panels", family.get("panels", ()))})))


def resolve_replays(replay, minimum_switches, *, report_switches=False, tie_events=None):
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

    resolution_started = time.perf_counter()
    outcomes = [branch.outcome for branch in replay.branches]
    survivors = terminal_winners(outcomes, rows=lambda outcome: outcome.paths)
    survivor_ids = {id(outcome) for outcome in survivors}
    if len(survivors) > 1 or report_switches:
        survivors = tuple(replace(outcome, minimum_switches=(
            outcome.minimum_switches if outcome.minimum_switches is not None
            else int(minimum_switches(outcome.paths)))) for outcome in survivors)
    chosen = min(survivors, key=lambda outcome: (
        len(outcome.paths), outcome.minimum_switches if outcome.minimum_switches is not None else 0,
        tie_rules.panel_key(outcome.paths)))

    certified = all(outcome.summary.certified for outcome in outcomes)
    consumed = {choice.family for branch in replay.branches for choice in branch.choices}
    families = []
    unverified = []
    for outcome in outcomes:
        for family in outcome.release_alternatives:
            if (family.get("count_unverified") or family.get("status") == "count_unverified"
                    or "beam_domain" in family):
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
    distinct = {}
    for family in families:
        key = (_family_key(family), bool(family.get("unresolved_search", False)),
               family.get("status"))
        distinct.setdefault(key, family)
    families = [*distinct.values(), *unverified]
    if len(survivors) > 1:
        families.append(dict(
            reason="terminal_join_continuations", source_decision="join_replay",
            panels=[tie_rules.panel_key(outcome.paths) for outcome in survivors],
            objective="join_ll", values=[float(outcome.log_likelihood) for outcome in survivors],
            minimum_switches=[outcome.minimum_switches for outcome in survivors],
            status=("unresolved_join_continuation" if not certified else
                    "ambiguous_k" if len({len(outcome.paths) for outcome in survivors}) > 1
                    else "same_k_equal_LL"), unresolved_search=not certified,
            representative_prior="smaller_K_then_exact_minimum_switches",
            source_replay_complete=replay.complete, source_search_certified=certified,
            source_choices=[[choice.record() for choice in branch.choices]
                            for branch in replay.branches]))

    diagnostic = dict(stage="join_replay", **replay.diagnostics(),
                      branch_K=[len(outcome.paths) for outcome in outcomes],
                      branch_LL=[float(outcome.log_likelihood) for outcome in outcomes],
                      survivors=len(survivors),
                      certified=certified,
                      resolved=replay.complete and certified and len(survivors) == 1 and not families,
                      additional_seconds=max(0.0, replay.seconds - replay.branches[0].seconds),
                      resolution_seconds=time.perf_counter() - resolution_started)
    # Biological census/acceptance events describe the selected complete history.
    # Work totals account for every executed history, and the replay event reports
    # branch costs separately without mixing their count-state semantics.
    summary = replace(chosen.summary,
                      total_full_scores=sum(outcome.summary.total_full_scores for outcome in outcomes),
                      count_exact_scores=sum(outcome.summary.count_exact_scores for outcome in outcomes),
                      count_exact_sample_scores=sum(outcome.summary.count_exact_sample_scores for outcome in outcomes),
                      beam_seconds=sum(outcome.summary.beam_seconds for outcome in outcomes),
                      phase_seconds={name: sum(outcome.summary.phase_seconds.get(name, 0.0)
                                               for outcome in outcomes)
                                     for name in {key for outcome in outcomes
                                                  for key in outcome.summary.phase_seconds}},
                      work_counters={name: sum(outcome.summary.work_counters.get(name, 0)
                                               for outcome in outcomes)
                                     for name in {key for outcome in outcomes
                                                  for key in outcome.summary.work_counters}},
                      tie_events=chosen.summary.tie_events if tie_events is None else tie_events,
                      events=(*chosen.summary.events, diagnostic))
    return replace(chosen, summary=summary, release_alternatives=tuple(families))


def observed_plateau_family(panel, likelihood, observed, *, linkage=None):
    """Uncontinued equal-data neighbors actually scored at a final certification.

    This does not enumerate a plateau or assert that neighbors passed their own
    count decisions. PanelSearch applies the exact prior to the eligible choices
    and continues the selected panel before returning. Unrun choices remain source
    ambiguity until the complete bounded replay tree has evaluated them.
    """
    from ...maths import ties

    nominal = tie_rules.panel_key(panel)
    candidates = dict(observed)
    candidates[nominal] = (float(likelihood), "terminal_representative")
    if len(candidates) < 2:
        return None
    panels = sorted(candidates)
    values = [float(candidates[key][0]) for key in panels]
    scale = max(abs(value) for value in values)
    classes = ties.tie_classes(values, tie_rules.quantum("join_ll", scale),
                               motion=tie_rules.motion("join_ll", scale),
                               content=panels, descending=True)
    links = None if linkage is None else [sum(linkage(path) for path in key) for key in panels]
    members = sorted(tie_rules.symmetric_members(classes, links))
    if panels.index(nominal) not in members:
        return None
    return dict(reason="terminal_certification_plateau",
                source_decision="terminal_certification",
                panels=[panels[index] for index in members],
                objective="join_ll", values=[values[index] for index in members],
                source_passes=[candidates[panels[index]][1] for index in members],
                status="unresolved_plateau_continuation", unresolved_search=True,
                enumeration_scope="exact_scored_certification_neighbours",
                representative_prior="smaller_K_then_exact_minimum_switches")
