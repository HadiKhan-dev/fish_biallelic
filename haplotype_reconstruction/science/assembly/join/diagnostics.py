"""Typed join outcomes and ordered diagnostic records (AJ14, AJ21–25).

Owns immutable snapshots of one search. Arrays retain the original sample and
canonical founder order. Pure record encoders preserve the payload schema;
The stage measures whole-join duration; the phase timing fields here are unmeasured zero placeholders.

Owns: typed outcomes, count census and ordered per-refit event snapshots.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: docs/methods/assembly.md#aj14.
Invariant: Only cross-refit scheduling interleaving is removed; every within-refit event remains ordered.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from dataclasses import fields as dataclass_fields

import numpy as np

from haplotype_reconstruction.contracts.assembly import ReleaseFamily
from haplotype_reconstruction.contracts.roles import record_field, schema_field
from haplotype_reconstruction.core.ties import TieCounts


def _plain(value):
    """A JSON-able copy (dict, list, str, int, float, bool, None) of a search record value."""
    if isinstance(value, TieCounts):
        return {
            entry.kind: {name: getattr(entry, name) for name in ("decisions", "inside_tie", "wide", "boundary")}
            for entry in value.entries
        }
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
    backstop).
    state_revisits: count steps that would have returned to a panel this search already
    held after a count change (a count cycle through births, deaths and seed swaps; births and seed
    swaps then stop); count_backstop_reached: the search reached COUNT_BACKSTOP count changes and
    stopped births and seed swaps (reported, never expected).
    """

    decisions: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    births: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    pool_births: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    seed_births: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    seed_swaps: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    deaths: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    pure_mosaic_deaths: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    ambiguous: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    births_frozen: bool = record_field("state", axes=(), units="1", dtype="bool", default=False)
    start_size: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    cycles: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    cycles_undecided: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    trap_indicators: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    seed_refits: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    ladder_births: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    final_paths_failing_death_test: int | None = record_field(
        "state", axes=(), units="1", dtype="int | None", default=None
    )
    state_revisits: int = record_field("state", axes=(), units="1", dtype="int64", default=0)
    count_backstop_reached: bool = record_field("state", axes=(), units="1", dtype="bool", default=False)


@dataclass(frozen=True)
class InitialPanel:
    """One anchor panel of the search's start: anchor block, its size and log likelihood."""

    anchor: int = record_field("state", axes=(), units="1", dtype="int64")
    panel_size: int = record_field("state", axes=(), units="1", dtype="int64")
    likelihood: float = record_field("state", axes=(), units="nats", dtype="float64")


@dataclass(frozen=True)
class JoinSummary:
    """What one joined batch's certified panel search did (a small typed record).

    final_panel_size, final_likelihood: K and the join log likelihood (nats) of the result;
    total_full_scores: full-panel exact scores; initialization: the distinct anchor panels
    scored at the start, selected_initial_anchor the one the search started from; sweeps
    (tier-1 sweeps + accepted tier-2 moves), tier1_sweeps, sweep_backstop, tier2_passes,
    tier2_accepted, saturation_steps, saturation_accepted: work and moves by phase; quota: the
    endpoint quota reached, ladder_top / quota_backstop: the saturation ladder's top and its
    backstop (ladder_extended: the ladder climbed beyond 256);
    count: the count step's tallies; count_exact_scores / count_exact_sample_scores:
    exact scores the count step made; pool_original / pool_final / pool_largest: candidate pool
    sizes; certified: no move improves the panel and the count rule is quiescent at every rung;
    backstop_reached / pool_backstop: the search stopped at a backstop (not certified).
    tie_events: core.ties.TieCounter record of the join's tie-class decisions (beam cuts and search choices):
    {kind: {decisions, inside_tie, wide, boundary}}.
    events: the search's records in order (tier-1 sweeps, count deaths / seeds / births, tier-2
    passes, pool-saturation rungs), plain JSON dicts as the reports table lists them.
    """

    final_panel_size: int = record_field("state", axes=(), units="1", dtype="int64")
    final_likelihood: float = record_field("state", axes=(), units="nats", dtype="float64")
    total_full_scores: int = record_field("diagnostic", axes=(), units="1", dtype="int64")
    initialization: tuple[InitialPanel, ...] = record_field(
        "state", axes=(), units="1", dtype="tuple[InitialPanel, ...]"
    )
    selected_initial_anchor: int = record_field("state", axes=(), units="1", dtype="int64")
    sweeps: int = record_field("state", axes=(), units="1", dtype="int64")
    tier1_sweeps: int = record_field("state", axes=(), units="1", dtype="int64")
    sweep_backstop: int = record_field("state", axes=(), units="1", dtype="int64")
    tier2_passes: int = record_field("state", axes=(), units="1", dtype="int64")
    tier2_accepted: int = record_field("state", axes=(), units="1", dtype="int64")
    saturation_steps: int = record_field("state", axes=(), units="1", dtype="int64")
    saturation_accepted: int = record_field("state", axes=(), units="1", dtype="int64")
    quota: int = record_field("state", axes=(), units="1", dtype="int64")
    ladder_top: int = record_field("state", axes=(), units="1", dtype="int64")
    quota_backstop: int = record_field("state", axes=(), units="1", dtype="int64")
    count: CountCensus = record_field("state", axes=(), units="1", dtype="CountCensus")
    count_exact_scores: int = record_field("diagnostic", axes=(), units="1", dtype="int64")
    count_exact_sample_scores: int = record_field("diagnostic", axes=(), units="1", dtype="int64")
    pool_original: int = record_field("state", axes=(), units="1", dtype="int64")
    pool_final: int = record_field("state", axes=(), units="1", dtype="int64")
    pool_largest: int = record_field("state", axes=(), units="1", dtype="int64")
    certified: bool = record_field("state", axes=(), units="1", dtype="bool")
    backstop_reached: bool = record_field("state", axes=(), units="1", dtype="bool")
    pool_backstop: bool = record_field("state", axes=(), units="1", dtype="bool")
    events: tuple[dict, ...] = record_field("state", axes=(), units="1", dtype="tuple[dict, ...]", default=())
    # A fresh search holds its TieCounter record (TieCounts); a decoded summary holds its plain dict.
    tie_events: TieCounts | dict = record_field("diagnostic", axes=(), units="1", dtype="dict", default_factory=dict)
    work_counters: dict = record_field("diagnostic", axes=(), units="1", dtype="dict", default_factory=dict)

    @property
    def ladder_extended(self) -> bool:
        return self.ladder_top == self.quota_backstop

    @property
    def ambiguous_count_calls(self) -> int:
        """Count decisions (deaths and births) of this search whose band contains zero."""
        return sum(
            bool(event.get("ambiguous"))
            for event in self.events
            if event.get("stage") in ("count_death", "count_birth")
        )

    def final_record(self) -> dict:
        """The search's closing record (the last row of its reports table entries)."""
        return dict(
            total_full_scores=self.total_full_scores,
            final_panel_size=self.final_panel_size,
            final_likelihood=self.final_likelihood,
            initialization=[
                dict(anchor=item.anchor, panel_size=item.panel_size, likelihood=item.likelihood)
                for item in self.initialization
            ],
            selected_initial_anchor=self.selected_initial_anchor,
            sweeps=self.sweeps,
            tier1_sweeps=self.tier1_sweeps,
            sweep_backstop=self.sweep_backstop,
            tier2_passes=self.tier2_passes,
            tier2_accepted=self.tier2_accepted,
            saturation_steps=self.saturation_steps,
            saturation_accepted=self.saturation_accepted,
            quota=self.quota,
            ladder_top=self.ladder_top,
            ladder_extended=self.ladder_extended,
            quota_backstop=self.quota_backstop,
            count={item.name: getattr(self.count, item.name) for item in dataclass_fields(CountCensus)},
            count_exact_scores=self.count_exact_scores,
            count_exact_sample_scores=self.count_exact_sample_scores,
            pool_sizes=dict(original=self.pool_original, final=self.pool_final, largest=self.pool_largest),
            certified=self.certified,
            backstop_reached=self.backstop_reached,
            pool_backstop=self.pool_backstop,
            tie_events=_plain(self.tie_events),
            work_counters=_plain(self.work_counters),
        )

    def records(self) -> list[dict]:
        """Every record of the search in order, the closing record last (one reports-table row
        each); plain dicts (copies)."""
        return [_plain(event) for event in self.events] + [self.final_record()]

    def record(self) -> dict:
        """The summary as one plain JSON dict (the closing record plus ``events``)."""
        return dict(self.final_record(), events=[_plain(event) for event in self.events])

    @classmethod
    def from_record(cls, record: Mapping) -> JoinSummary:
        """The summary :meth:`record` encoded."""
        pools = record["pool_sizes"]
        return cls(
            final_panel_size=int(record["final_panel_size"]),
            final_likelihood=float(record["final_likelihood"]),
            total_full_scores=int(record["total_full_scores"]),
            initialization=tuple(
                InitialPanel(int(item["anchor"]), int(item["panel_size"]), float(item["likelihood"]))
                for item in record["initialization"]
            ),
            selected_initial_anchor=int(record["selected_initial_anchor"]),
            sweeps=int(record["sweeps"]),
            tier1_sweeps=int(record["tier1_sweeps"]),
            sweep_backstop=int(record["sweep_backstop"]),
            tier2_passes=int(record["tier2_passes"]),
            tier2_accepted=int(record["tier2_accepted"]),
            saturation_steps=int(record["saturation_steps"]),
            saturation_accepted=int(record["saturation_accepted"]),
            quota=int(record["quota"]),
            ladder_top=int(record["ladder_top"]),
            quota_backstop=int(record["quota_backstop"]),
            count=CountCensus(
                **{
                    item.name: record["count"][item.name]
                    for item in dataclass_fields(CountCensus)
                    if item.name in record["count"]
                }
            ),
            count_exact_scores=int(record["count_exact_scores"]),
            count_exact_sample_scores=int(record["count_exact_sample_scores"]),
            pool_original=int(pools["original"]),
            pool_final=int(pools["final"]),
            pool_largest=int(pools["largest"]),
            certified=bool(record["certified"]),
            backstop_reached=bool(record["backstop_reached"]),
            pool_backstop=bool(record["pool_backstop"]),
            events=tuple(_plain(event) for event in record.get("events", ())),
            tie_events=_plain(record.get("tie_events", {})),
            work_counters=_plain(record.get("work_counters", {})),
        )


@dataclass(frozen=True)
class JoinOutcome:
    """The selected panel of one join.

    paths: int32[K_out, n]; output founder f copies founder (local row) paths[f, j] of input
        block j, in canonical order (the hierarchy publishes the prepared-row form in canonical order).
    log_likelihood: the join objective of the panel (sum over the batch's samples of V_s, nats).
    summary: what the search did.
    """

    schema: int = schema_field(1)
    paths: np.ndarray = record_field("state", axes=("K", "B"), units="1", dtype="int32")
    log_likelihood: float = record_field("state", axes=(), units="nats", dtype="float64")
    summary: JoinSummary = record_field("state", axes=(), units="1", dtype="JoinSummary")
    release_alternatives: tuple[ReleaseFamily, ...] = record_field(
        "state", axes=(), units="1", dtype="tuple", default=()
    )
    minimum_switches: int | None = record_field("state", axes=(), units="1", dtype="int | None", default=None)

    def __post_init__(self) -> None:
        """Freeze family contracts at the numerical output boundary."""
        object.__setattr__(
            self,
            "release_alternatives",
            tuple(join_family(family, self.paths.shape[1]) for family in self.release_alternatives),
        )


def join_family(value: Mapping | ReleaseFamily, blocks: int) -> ReleaseFamily:
    """Type a sparse join-local family while retaining its supplied field presence.

    The hierarchy supplies the final component span. Local records own [0,B);
    absent span keys remain absent from their source-provenance mapping view.
    """
    if isinstance(value, ReleaseFamily):
        return value
    result = ReleaseFamily.from_mapping({"start": 0, "stop": blocks, **value})
    return replace(result, _present=tuple(value))


def append_event(search, record: dict, source: str | None = None) -> None:
    """Append to a seed's ordered trace while that count step is collecting refits.

    Callers hold the search lock. The source is routing information, not an extra
    scientific record field; the event payload contains only the decision record.
    """
    buffers = getattr(search, "refit_events", None)
    match = None if source is None else re.match(r"(c[0-9]+/seed[0-9]+)(?:/|$)", source)
    if buffers is not None and match is not None and match[1] in buffers:
        buffers[match[1]].append(record)
    else:
        search.events.append(record)


@contextmanager
def ordered_refit_events(search, seeds):
    """Collect concurrent refit records, then append complete traces in seed order."""
    buffers = {f"c{search.count_steps}/seed{seed}": [] for seed in seeds}
    search.refit_events = buffers
    try:
        yield
    finally:
        search.refit_events = None
        for records in buffers.values():
            search.events.extend(records)


__all__ = [
    "CountCensus",
    "InitialPanel",
    "JoinSummary",
    "JoinOutcome",
    "join_family",
    "append_event",
    "ordered_refit_events",
]
