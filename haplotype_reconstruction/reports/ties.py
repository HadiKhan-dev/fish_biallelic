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
                       incomplete join count verification
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import math
import re
from pathlib import Path

from .tables import write_json, write_rows

FIELDS = ("decisions", "inside_tie", "wide", "boundary")
STAGE_ORDER = ("discovery", "local_selection", "assembly", "painting", "pedigree")
# Record groups that hold tie_events: (key pattern, stage, top-level entries decoded). assembly/final repeats the
# join summaries of assembly/l*/join and is read only for the final count calls; the
# digest-keyed joins/ records are not read (no events, or the same joins again).
GROUPS = (
    (re.compile(r"discovery/blocks/\d+"), "discovery", ("blocks",)),
    (re.compile(r"local_selection/[^/]+/blocks/\d+"), "local_selection", ("summaries",)),
    (re.compile(r"local_selection/[^/]+/proposals"), "local_selection", ("summary",)),
    (re.compile(r"local_selection/[^/]+/assembly/completion"), "local_selection", ("summary",)),
    (re.compile(r"local_selection/[^/]+/assembly/l\d+/join"), "local_selection", ("record",)),
    (re.compile(r"assembly/completion"), "assembly", ("summary",)),
    (re.compile(r"assembly/l\d+/join"), "assembly", ("record",)),
    (re.compile(r"painting"), "painting", ("label_continuity",)),
)
FINAL_ASSEMBLY = "assembly/final"
PEDIGREE = "pedigree"
READ_THREADS = 8        # contigs read concurrently (small decodes; the reads are filesystem-bound)


def is_band(kind: str) -> bool:
    """A threshold-band kind (a value inside the band is given the threshold's fixed side), not a tie class."""
    return kind.endswith("band")


def add_events(value, sink: dict) -> None:
    """Add every ``tie_events`` record found in a decoded payload (dicts, lists, tuples) to ``sink``."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "tie_events" and isinstance(item, dict):
                for kind, counts in item.items():
                    entry = sink.setdefault(str(kind), dict.fromkeys(FIELDS, 0))
                    for name in FIELDS:
                        entry[name] += int(counts.get(name, 0))
            else:
                add_events(item, sink)
    elif isinstance(value, (list, tuple)):
        for item in value:
            add_events(item, sink)


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
        self.unverified_calls = 0

    def discovery(self, payload) -> None:
        from ..discovery.search import backstop_counts
        for block in payload.get("blocks") or ():
            if block is None:                     # uninformative block: no search
                continue
            search = block["search"]
            self.blocks += 1
            self.backstops.update(backstop_counts(search))
            if search.get("selected_mean_field_failures") is not None:
                self.mean_field_measured_blocks += 1
                self.selected_mean_field_failures += int(search["selected_mean_field_failures"])
                self.other_mean_field_failures += int(search["other_mean_field_failures"])
            self.search_incomplete_blocks += int(bool(search.get("search_incomplete", False)))
            self.unresolved_blocks += int(bool(search.get("unresolved", False)))
            self.search_omissions.extend((str(item["cause"]), str(item["scope"]),
                                          str(item.get("count_kind", "exact")),
                                          str(item.get("kind", "tie_class_at_budget")), int(item["count"]))
                                         for item in search.get("omissions", ()))
            self.ambiguous_blocks += int(bool(search.get("ambiguous", False)))
            self.tied_alternatives += int(search.get("tied_alternatives", 0))
            margin = float(search.get("score_margin", math.inf))
            if math.isfinite(margin):
                self.margins.append(margin)

    def final_calls(self, payload) -> None:
        for call in (payload.get("components") or {}).get("count_call") or ():
            if call is None:
                continue
            self.count_calls += 1
            self.unverified_calls += int(call.get("reason") == "join_search_unverified")
            self.ambiguous_calls += int(bool(call.get("ambiguous", False)))
            self.count_cycles += int(bool(call.get("count_cycle", False)))
            self.undecided_cycles += int(bool(call.get("cycle_undecided", False)))

    def merge(self, other: "_Ambiguity") -> None:
        for name, value in vars(other).items():
            setattr(self, name, getattr(self, name) + value)

    def record(self) -> dict:
        from ..discovery.search import backstop_counts
        margins = sorted(self.margins)
        omissions = Counter()
        for cause, scope, count_kind, kind, count in self.search_omissions:
            omissions[cause, scope, count_kind, kind] += count
        return dict(
            discovery=dict(
                blocks=self.blocks, ambiguous=self.ambiguous_blocks, tied_alternatives=self.tied_alternatives,
                search_incomplete=self.search_incomplete_blocks, unresolved=self.unresolved_blocks,
                backstops={name: self.backstops[name] for name in backstop_counts({})},
                mean_field_failures=dict(measured_blocks=self.mean_field_measured_blocks,
                                        selected=self.selected_mean_field_failures,
                                        other=self.other_mean_field_failures),
                search_incomplete_fraction=self.search_incomplete_blocks / self.blocks if self.blocks else 0.0,
                omissions=[dict(cause=key[0], scope=key[1], count_kind=key[2], kind=key[3], count=count)
                           for key, count in sorted(omissions.items())],
                score_margin=dict(
                    blocks_with_other_k=len(margins), below_one=sum(1 for m in margins if m < 1.0),
                    minimum=margins[0] if margins else None,
                    median=margins[len(margins) // 2] if margins else None)),
            assembly=dict(
                count_calls=self.count_calls, ambiguous=self.ambiguous_calls, count_cycles=self.count_cycles,
                undecided_count_cycles=self.undecided_cycles, unverified_count_calls=self.unverified_calls))


def _rows(contig, group, stage, events) -> list[dict]:
    return [dict(contig=contig, stage=stage, records=group, kind=kind,
                 **{name: int(events[kind][name]) for name in FIELDS}) for kind in sorted(events)]


def _contig(store, contig):
    """The tie-event rows and ambiguity flags of one contig's records."""
    file = store.contig(contig)
    groups: dict[tuple[str, str], dict] = {}
    ambiguity = _Ambiguity()
    for key in sorted(file.keys()):
        found = _group_of(key)
        if found is None:
            if key == FINAL_ASSEMBLY:
                ambiguity.final_calls(file.get(key, items=("components",), nthreads=1))
            continue
        group, stage, items = found
        payload = file.get(key, items=items, nthreads=1)
        add_events(payload, groups.setdefault((stage, group), {}))
        if stage == "discovery":
            ambiguity.discovery(payload)
    rows = []
    for (stage, group), events in sorted(groups.items(),
                                         key=lambda item: (STAGE_ORDER.index(item[0][0]), item[0][1])):
        rows += _rows(contig, group, stage, events)
    return rows, ambiguity


def collect(store, contigs, pedigree_events: dict | None = None):
    """Per-contig tie-event rows (contig order) and the summed ambiguity flags of a run's records."""
    contigs = list(contigs)
    with ThreadPoolExecutor(max_workers=max(1, min(READ_THREADS, len(contigs)))) as pool:
        results = list(pool.map(lambda contig: _contig(store, contig), contigs))
    rows, ambiguity = [], _Ambiguity()
    for contig_rows, contig_ambiguity in results:
        rows += contig_rows
        ambiguity.merge(contig_ambiguity)
    if pedigree_events is None and store.run.has(PEDIGREE):
        pedigree_events = store.run.get(PEDIGREE, items=("tie_events",)).get("tie_events") or {}
    if pedigree_events:
        sink: dict = {}
        add_events(dict(tie_events=pedigree_events), sink)
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
        out[stage] = dict(tie_classes=classes, bands=dict(comparisons=sum(entry["decisions"] for entry in bands),
                                                          inside_band=sum(entry["inside_tie"] for entry in bands)))
    return out


def write_ties(output, store, contigs, pedigree_events: dict | None = None) -> list[Path]:
    """Write ties/decisions.csv and ties/summary.json for a completed run; returns the paths written.
    ``pedigree_events``: the pedigree product's tie_events when the caller holds it (else read from the record)."""
    rows, ambiguity = collect(store, contigs, pedigree_events)
    kinds = totals(rows)
    stages = stage_totals(kinds)
    directory = Path(output) / "ties"
    written = [write_rows(directory / "decisions.csv", rows,
                          columns=("contig", "stage", "records", "kind") + FIELDS)]
    from ..maths.ties import reverse_conventions
    reversed_run = reverse_conventions()
    written.append(write_json(directory / "summary.json", dict(
        conventions="docs/methods/ties.md", reverse_conventions=reversed_run,
        contigs=list(contigs), stages=stages, kinds=kinds, ambiguity=ambiguity)))
    discovery, assembly = ambiguity["discovery"], ambiguity["assembly"]
    classes = "; ".join(f"{stage} {value['tie_classes']['inside_tie']:,} of {value['tie_classes']['decisions']:,}"
                        for stage, value in stages.items())
    inside_band = sum(value["bands"]["inside_band"] for value in stages.values())
    comparisons = sum(value["bands"]["comparisons"] for value in stages.values())
    print(f"[Ties] tie-class decisions taken by the fixed conventions (docs/methods/ties.md"
          + ("; conventions REVERSED, HAPLOTYPES_TIE_REVERSE" if reversed_run else "") + f"): {classes}; "
          f"threshold comparisons inside a band: {inside_band:,} of {comparisons:,}. Discovery blocks selected "
          f"inside a tie: {discovery['ambiguous']:,} of {discovery['blocks']:,}; computational search limits: "
          f"{discovery['search_incomplete']:,} blocks ({discovery['unresolved']:,} with no incumbent); "
          f"founder-count calls ambiguous: "
          f"{assembly['ambiguous']:,} of {assembly['count_calls']:,}; unverified count calls: "
          f"{assembly['unverified_count_calls']:,}. Tables: {directory}",
          flush=True)
    return written
