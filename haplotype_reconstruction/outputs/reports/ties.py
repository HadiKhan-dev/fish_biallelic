"""OUTPUT/ties/: the decisions a run took inside a tie, and the ambiguity its stages recorded (release 2.1).

Where the data do not separate alternatives (equal values, or values closer than the decision kind's tie quantum),
release 2.1 decides by fixed deterministic conventions; docs/methods/ties.md lists them per stage, as conventions
under review for release 2.2. The stages count those decisions in the ``tie_events`` fields of their records
(``maths.ties.TieCounter``: per decision kind, decisions made, ``inside_tie`` decisions taken by the convention,
``wide`` and ``boundary`` guard events). Kinds named ``...band`` count threshold comparisons instead: a value inside
the band around a threshold is given the threshold's fixed side. This report sums the counts when the run completes,
decoding only the summary entries of the records; it never feeds back into inference.

  ties/decisions.csv   one row per contig (``run`` for run-level records), record group and decision kind
  ties/summary.json    per stage the tie-class decisions and band comparisons, the counts per kind, and the
                       recorded ambiguity flags: discovery blocks whose selection was taken inside a tie
                       (``ambiguous``, ``tied_alternatives``) and the margin of each block's founder-count choice
                       (``score_margin``); final founder-count calls marked ambiguous or closing a count cycle, and
                       refinement count decisions whose tied refits disagree
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from ...core.ties import TieCounts
from .tables import write_json, write_rows

FIELDS = ("decisions", "inside_tie", "wide", "boundary")
STAGE_ORDER = ("evidence", "discovery", "local_selection", "assembly", "painting", "pedigree")
# Record groups that hold tie_events: (key pattern, stage, top-level entries decoded). assembly/final repeats the
# join summaries of assembly/l*/join and is read only for the final count calls; refinement fit checkpoints and the
# digest-keyed joins/ records are not read (no events, or the same joins again).
GROUPS = (
    (re.compile(r"evidence"), "evidence", ("report",)),
    (re.compile(r"discovery/blocks/\d+"), "discovery", ("blocks",)),
    (re.compile(r"local_selection/[^/]+/blocks/\d+"), "local_selection", ("summaries",)),
    (re.compile(r"local_selection/[^/]+/proposals"), "local_selection", ("summary",)),
    (re.compile(r"local_selection/[^/]+/assembly/completion"), "local_selection", ("summary",)),
    (re.compile(r"local_selection/[^/]+/assembly/l\d+/join"), "local_selection", ("record",)),
    (re.compile(r"assembly/completion"), "assembly", ("summary",)),
    (re.compile(r"assembly/l\d+/join"), "assembly", ("record",)),
    (re.compile(r"assembly/l\d+/refine/\d+"), "assembly", ("refinement",)),
    (re.compile(r"painting"), "painting", ("label_continuity",)),
)
FINAL_ASSEMBLY = "assembly/final"
PEDIGREE = "pedigree"


def is_band(kind: str) -> bool:
    """A threshold-band kind (a value inside the band is given the threshold's fixed side), not a tie class."""
    return kind.endswith("band")


def add_events(value: TieCounts, sink: dict) -> None:
    """Add one immutable counter set, never recursively traverse arbitrary records."""
    for count in value.entries:
        entry = sink.setdefault(count.kind, dict.fromkeys(FIELDS, 0))
        for name in FIELDS:
            entry[name] += int(getattr(count, name))


def _group_of(key: str):
    for pattern, stage, items in GROUPS:
        if pattern.fullmatch(key):
            return re.sub(r"/\d+$", "", key), stage, items
    return None


class _Ambiguity:
    """The recorded ambiguity flags (summed over the contigs merged into it)."""

    def __init__(self):
        self.blocks = self.ambiguous_blocks = self.tied_alternatives = 0
        self.margins: list[float] = []
        self.search_incomplete_blocks = self.unresolved_blocks = 0
        self.search_omissions: list[tuple] = []
        self.backstops = Counter()
        self.mean_field_measured_blocks = 0
        self.selected_mean_field_failures = self.other_mean_field_failures = 0
        self.count_calls = self.ambiguous_calls = self.count_cycles = self.undecided_cycles = 0
        self.tied_refits = self.disagreeing_refits = 0

    def discovery(self, payload) -> None:
        for block in payload:
            if block is None:  # uninformative block: no search
                continue
            search = block
            self.blocks += 1
            self.backstops.update(dict(search.backstops))
            if search.selected_mean_field_failures is not None:
                self.mean_field_measured_blocks += 1
                self.selected_mean_field_failures += int(search.selected_mean_field_failures)
                self.other_mean_field_failures += int(search.other_mean_field_failures)
            self.search_incomplete_blocks += int(bool(search.search_incomplete))
            self.unresolved_blocks += int(bool(search.unresolved))
            self.search_omissions.extend(
                (str(item.cause), str(item.scope), str(item.count_kind), str(item.kind), int(item.count))
                for item in search.omissions
            )
            self.ambiguous_blocks += int(bool(search.ambiguous))
            self.tied_alternatives += int(search.tied_alternatives)
            margin = float(search.score_margin)
            if math.isfinite(margin):
                self.margins.append(margin)

    def refinement(self, payload) -> None:
        for agree in payload:
            if agree is not None:
                self.tied_refits += 1
                self.disagreeing_refits += int(not agree)

    def final_calls(self, payload) -> None:
        for call in payload:
            if call is None:
                continue
            self.count_calls += 1
            self.ambiguous_calls += int(bool(call.ambiguous))
            self.count_cycles += int(bool(call.count_cycle))
            self.undecided_cycles += int(bool(call.cycle_undecided))

    def merge(self, other: _Ambiguity) -> None:
        for name, value in vars(other).items():
            setattr(self, name, getattr(self, name) + value)

    def record(self) -> dict:
        margins = sorted(self.margins)
        omissions: Counter[tuple[str, str, str, str]] = Counter()
        for cause, scope, count_kind, kind, count in self.search_omissions:
            omissions[cause, scope, count_kind, kind] += count
        return dict(
            discovery=dict(
                blocks=self.blocks,
                ambiguous=self.ambiguous_blocks,
                tied_alternatives=self.tied_alternatives,
                search_incomplete=self.search_incomplete_blocks,
                unresolved=self.unresolved_blocks,
                backstops={name: self.backstops[name] for name in BACKSTOP_COLUMNS},
                mean_field_failures=dict(
                    measured_blocks=self.mean_field_measured_blocks,
                    selected=self.selected_mean_field_failures,
                    other=self.other_mean_field_failures,
                ),
                search_incomplete_fraction=self.search_incomplete_blocks / self.blocks if self.blocks else 0.0,
                omissions=[
                    dict(cause=key[0], scope=key[1], count_kind=key[2], kind=key[3], count=count)
                    for key, count in sorted(omissions.items())
                ],
                score_margin=dict(
                    blocks_with_other_k=len(margins),
                    below_one=sum(1 for m in margins if m < 1.0),
                    minimum=margins[0] if margins else None,
                    median=margins[len(margins) // 2] if margins else None,
                ),
            ),
            assembly=dict(
                count_calls=self.count_calls,
                ambiguous=self.ambiguous_calls,
                count_cycles=self.count_cycles,
                undecided_count_cycles=self.undecided_cycles,
                tied_refit_decisions=self.tied_refits,
                tied_refits_disagree=self.disagreeing_refits,
            ),
        )


def _rows(contig, group, stage, events) -> list[dict]:
    return [
        dict(contig=contig, stage=stage, records=group, kind=kind, **{name: int(events[kind][name]) for name in FIELDS})
        for kind in sorted(events)
    ]


def _contig(results, contig: str):
    """Read only the original record groups, retaining their counter attribution."""
    groups: dict = {}
    ambiguity = _Ambiguity()
    for source in results.tie_sources(contig):
        key = source.record_group
        found = _group_of(key)
        if found is None and (key.endswith("/blocks") or key.endswith("/refine")):
            found = _group_of(key + "/0")
        if found is None:
            if key == FINAL_ASSEMBLY:
                ambiguity.final_calls(source.count_calls)
            continue
        group, stage, _ = found
        sink = groups.setdefault((stage, group), {})
        for counters in source.counters:
            add_events(counters, sink)
        if stage == "discovery":
            ambiguity.discovery(source.discovery)
        elif group.startswith("assembly/") and group.endswith("/refine"):
            ambiguity.refinement(source.refinement_agreement)
    rows = []
    for (stage, group), events in sorted(groups.items(), key=lambda item: (STAGE_ORDER.index(item[0][0]), item[0][1])):
        rows += _rows(contig, group, stage, events)
    return rows, ambiguity


def collect(results, contigs: Sequence[str], pedigree_events: TieCounts | None = None):
    """Aggregate original source groups in contig order, followed by the pedigree."""
    rows, ambiguity = [], _Ambiguity()
    for contig in contigs:
        contig_rows, contig_ambiguity = _contig(results, contig)
        rows += contig_rows
        ambiguity.merge(contig_ambiguity)
    if pedigree_events is None:
        pedigree_events = results.pedigree_ties()
    if pedigree_events is not None:
        sink: dict = {}
        add_events(pedigree_events, sink)
        rows += _rows("run", PEDIGREE, "pedigree", sink)
    return rows, ambiguity.record()


def totals(rows) -> dict:
    """The counts summed over contigs and record groups: {stage: {kind: {field: count}}}, stages in pipeline order."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        entry = out.setdefault(row["stage"], {}).setdefault(row["kind"], dict.fromkeys(FIELDS, 0))
        for name in FIELDS:
            entry[name] += row[name]
    return {stage: dict(sorted(out[stage].items())) for stage in STAGE_ORDER if stage in out}


def stage_totals(kinds: dict) -> dict:
    """Per stage: tie-class decisions (decisions, inside_tie, wide, boundary) and band comparisons (comparisons,
    inside_band)."""
    out = {}
    for stage, entries in kinds.items():
        classes = {name: sum(entry[name] for kind, entry in entries.items() if not is_band(kind)) for name in FIELDS}
        bands = [entry for kind, entry in entries.items() if is_band(kind)]
        out[stage] = dict(
            tie_classes=classes,
            bands=dict(
                comparisons=sum(entry["decisions"] for entry in bands),
                inside_band=sum(entry["inside_tie"] for entry in bands),
            ),
        )
    return out


def write_ties(
    output,
    results,
    contigs: Sequence[str],
    pedigree_events: TieCounts | None = None,
    *,
    reverse_conventions: bool = False,
) -> list[Path]:
    """Publish completed-run counters with the resolved explicit convention setting."""
    rows, ambiguity = collect(results, contigs, pedigree_events)
    kinds = totals(rows)
    directory = Path(output) / "ties"
    written = [write_rows(directory / "decisions.csv", rows, columns=DECISION_COLUMNS)]
    written.append(
        write_json(
            directory / "summary.json",
            dict(
                conventions="docs/methods/ties.md",
                reverse_conventions=reverse_conventions,
                contigs=list(contigs),
                stages=stage_totals(kinds),
                kinds=kinds,
                ambiguity=ambiguity,
            ),
        )
    )
    return written


# Provenance: B discovery/search.py:119-122, fixed order of the published backstop counts.
BACKSTOP_COLUMNS = (
    "expansion_backstop",
    "exact_score_backstop",
    "expansion_ranked_truncation",
    "expansion_tie_class_at_budget",
    "exact_score_ranked_truncation",
    "exact_score_tie_class_at_budget",
)


class TieSource(Protocol):
    """One typed results source retaining its original 3.5 record-group label."""

    record_group: str
    counters: tuple[TieCounts, ...]
    discovery: tuple[object, ...]
    refinement_agreement: tuple[bool | None, ...]
    count_calls: tuple[object, ...]


# Published column schemas: B reports and P0 outputs inventory D6 / OU15.
# Row-dict tables retain B's headerless empty constructor; these constants do not
# impose columns on those tables. Scientific values remain upstream contract fields.
DECISION_COLUMNS = ("contig", "stage", "records", "kind", "decisions", "inside_tie", "wide", "boundary")
"""Columns in published order.

    contig: contig identifier (text).
    stage: scientific stage attribution (text).
    records: whitelisted record group, with repeated refinement indices grouped (text).
    kind: recorded tie or threshold-band kind (text).
    decisions: number of decisions recorded (1).
    inside_tie: number of recorded decisions inside the tie class or threshold band (1).
    wide: number of wide guard events (1).
    boundary: number of boundary guard events (1).
"""
