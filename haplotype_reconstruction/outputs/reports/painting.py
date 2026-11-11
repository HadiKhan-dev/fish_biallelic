"""OUTPUT/painting/: read-only founder callability, carrier support, assembly-search, founder-count
and sample-switch tables of one painted chromosome.

  founder_support.csv          per component and founder row: called/missing alleles, observed named carriers
                               (samples whose released painting names the row at a site with read depth),
                               pooled candidates (tracks the reads cannot separate between identical rows)
  founder_support_windows.csv  the same per 200-site window
  assembly_search.csv          every record of the final assembly's join searches (level, span, record fields)
  assembly_search_shadow_alternatives.csv  ordered shadow alternatives keyed by contig and assembly_search_row
  assembly_search_items.csv    unbudgeted symmetric/plateau collection items with parent, column and ordinal
  founder_counts.csv           per component: how firmly its founder count is determined (final count step;
                               count cycles and band-straddling decisions of the final and feedback join searches)
  sample_switches.csv          painted switches per sample relative to the cohort median
  label_continuity.csv         every accepted label-continuity move with its winners' switch directions and
                               whether the founder identity across its link is identified
  sample_segments.csv          the released painting per sample: runs of bins with one release (status, named
                               founder row, candidate rows), switch intervals included
  painting_model.csv           per component: the estimated copying model (ancestry-change rate, panel-error
                               entry and exit, the emissions' site correlation)

Carrier support uses released named painting tracks intersected with positive read depth: it is
painting-dependent, not independent lineage evidence or a phasing probability, and samples are correlated.
Unknown/pooled ancestry does not count as a named carrier. Labels are component-local. Diagnostics only: nothing
here is filtered, reweighted or fed back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...results.reader import Results

import numpy as np

from ...contracts.painting import Painting
from ...core.errors import InvariantError
from .tables import contig_part, write_contig_tables

WINDOW_SITES = 200
SUPPORT_BASIS = "released named painting ancestry AND positive depth; correlated samples"
TABLES = ("founder_support", "founder_support_windows", "assembly_search", "sample_switches", "founder_counts")


# ------------------------------------------------------------------------------------- assembly searches
def feedback_searches(results, contig: str):
    """Read the typed local-selection context join summaries for this contig."""
    return results.painting_report(contig).feedback_searches


def _count_calls(searches, first: int, last: int):
    """(count cycles, undecided cycles, band-straddling count decisions) of the join searches overlapping
    [first, last]; (None, None, None) when no join search covers the span. Each search counts once,
    even when a pass-through component lists that search at several levels."""
    cycles = undecided = ambiguous = 0
    recorded = False
    for _, start, end, summary in searches:
        if end < first or start > last:
            continue
        recorded = True
        cycles += summary.count.cycles
        undecided += summary.count.cycles_undecided
        ambiguous += summary.ambiguous_count_calls
    return (cycles, undecided, ambiguous) if recorded else (None, None, None)


def assembly_search_tables(rows):
    """Separate alternatives without changing records or the other event fields.

    A span/level can occur more than once, so the contig-local row ordinal is the
    join key. The alternative ordinal preserves list order. Each value is one
    original dict, including absent keys and JSON nulls; batch_size, K and the
    shadow_combinations budget bound its internal arrays (see the output guide).
    """
    parents, alternatives, items = [], [], []
    for index, original in enumerate(rows):
        row = {name: original[name] for name in ("Level", "FirstVariant1Based", "LastVariant1Based")}
        row["assembly_search_row"] = index
        row.update(original)
        values = row.get("shadow_alternatives")
        if values is not None:
            row["shadow_alternatives"] = len(values)
            for alternative, value in enumerate(values):
                alternatives.append(
                    dict(
                        assembly_search_row=index,
                        Level=row["Level"],
                        FirstVariant1Based=row["FirstVariant1Based"],
                        LastVariant1Based=row["LastVariant1Based"],
                        alternative=alternative,
                        value=value,
                    )
                )
        for column in SEARCH_COLLECTION_FIELDS:
            value = row.get(column)
            if isinstance(value, (list, tuple)):
                # observed_panels is a scalar census in some events and a list
                # in others. The fixed-size descriptor keeps that distinction,
                # including for an empty list with no item rows.
                row[column] = {"items": len(value)}
                items.extend(
                    dict(assembly_search_row=index, column=column, item=i, value=json.dumps(item, sort_keys=True))
                    for i, item in enumerate(value)
                )
        parents.append(row)
    return parents, alternatives, items


# Symmetric/plateau families retain all observed alternatives, not just the
# bounded number of executed continuations. They have no cardinality budget.
SEARCH_COLLECTION_FIELDS = (
    "panels",
    "alternatives",
    "observed_panels",
    "replay_panels",
    "cycle_excluded_panels",
    "values",
    "minimum_switches",
    "source_passes",
)

SEARCH_ITEM_COLUMNS = ("assembly_search_row", "column", "item", "value")
"""Columns in published order.

    assembly_search_row: zero-based parent assembly-search row within the contig (1).
    column: original collection column name (text).
    item: zero-based position within the original collection (1).
    value: one original alternative, panel, score or source as JSON; panel shape at most K by batch_size (text).
"""


SHADOW_ALTERNATIVE_COLUMNS = (
    "assembly_search_row",
    "Level",
    "FirstVariant1Based",
    "LastVariant1Based",
    "alternative",
    "value",
)
"""Columns in published order.

    assembly_search_row: zero-based parent row within the contig's assembly_search table (1).
    Level: parent's assembly level (1).
    FirstVariant1Based: parent's first variant coordinate, inclusive (bp).
    LastVariant1Based: parent's last variant coordinate, inclusive (bp).
    alternative: zero-based index in the original shadow_alternatives list (1).
    value: original alternative dictionary as sorted-key JSON; bounded by batch size, K and shadow_combinations (text).
"""


# ------------------------------------------------------------------------------------- tables
def _support_statistics(called, named, pooled, *, carriers=None, candidates=None):
    """Shared support columns, with whole-founder sample counts in published column order."""
    values, pooled_values = named[called], pooled[called]
    result = dict(CalledAlleles=int(called.sum()), MissingAlleles=int((~called).sum()))
    if carriers is not None:
        result["ObservedCarrierSamples"] = int(carriers.sum())
    result["CalledSitesWithoutObservedNamedCarrier"] = int(np.sum(values == 0))
    if carriers is not None:
        result["MinimumObservedCarriers"] = None if not len(values) else int(values.min())
    result["MeanObservedCarriers"] = None if not len(values) else float(values.mean())
    if candidates is not None:
        result["ObservedPooledCandidateSamples"] = int(candidates.sum())
    result["MeanObservedPooledCandidates"] = None if not len(values) else float(pooled_values.mean())
    result["CalledSitesWithoutObservedNamedOrPooledCandidate"] = int(np.sum(values + pooled_values == 0))
    return result


def founder_support_tables(painting, sites, observed, searches, *, window_sites=WINDOW_SITES):
    """(founder_support, founder_support_windows, assembly_search) row lists of one chromosome.

    ``painting``: ChromosomePainting with panels; ``sites``/``observed``: the evidence positions and its
    positive-depth mask bool [N, L]; ``searches``: the final assembly's join searches.
    """
    sites = np.asarray(sites)
    observed = np.asarray(observed, dtype=bool)
    samples = len(painting.sample_ids)
    if observed.shape != (samples, len(sites)):
        raise InvariantError("founder support and observation axes do not match")
    summaries, windows = [], []
    for component_index, component in enumerate(painting.components):
        panel = component.panel
        positions = np.asarray(panel.positions)
        index = np.searchsorted(sites, positions)
        if np.any(index >= len(sites)) or not np.array_equal(sites[index], positions):
            raise InvariantError("founder support and observation axes do not match")
        calls = np.asarray(panel.released)
        # One count per sample/site, even if both homologues have this label (or this candidate row).
        support = np.zeros(calls.shape, dtype=np.int32)
        pooled = np.zeros(calls.shape, dtype=np.int32)
        carriers = np.zeros((len(calls), samples), dtype=bool)
        candidates = np.zeros_like(carriers)
        for sample in range(samples):
            for chunk in component.chunks(sample):
                lo, hi = np.searchsorted(positions, [chunk.start, chunk.end])
                depth = observed[sample, index[lo:hi]]
                named = {rows[0] for rows in chunk.members if len(rows) == 1}
                pooled_rows = {row for rows in chunk.members if len(rows) >= 2 for row in rows}
                for rows, counts, flags in ((named, support, carriers), (pooled_rows, pooled, candidates)):
                    for founder in rows:
                        usable = depth & (calls[founder, lo:hi] >= 0)
                        counts[founder, lo:hi] += usable
                        flags[founder, sample] |= np.any(usable)
        for founder in range(len(calls)):
            called = calls[founder] >= 0
            base = dict(
                Component=component_index,
                Founder=str(founder),
                FirstVariant1Based=int(positions[0]),
                LastVariant1Based=int(positions[-1]),
                BreakBefore=panel.break_before is not None,
                BreakAfter=panel.break_after is not None,
            )
            summaries.append(
                dict(
                    **base,
                    MarkerCount=len(positions),
                    **_support_statistics(
                        called,
                        support[founder],
                        pooled[founder],
                        carriers=carriers[founder],
                        candidates=candidates[founder],
                    ),
                    SupportBasis=SUPPORT_BASIS,
                )
            )
            for start in range(0, len(positions), window_sites):
                stop = min(len(positions), start + window_sites)
                windows.append(
                    dict(
                        Component=component_index,
                        Founder=str(founder),
                        FirstVariant1Based=int(positions[start]),
                        LastVariant1Based=int(positions[stop - 1]),
                        MarkerCount=stop - start,
                        **_support_statistics(
                            called[start:stop], support[founder, start:stop], pooled[founder, start:stop]
                        ),
                    )
                )
    search_rows = [
        dict(Level=level, FirstVariant1Based=first, LastVariant1Based=last, **record)
        for level, first, last, summary in searches
        for record in summary.records()
    ]
    return summaries, windows, search_rows


def founder_count_table(painting, searches, feedback) -> list[dict]:
    """One row per component: how firmly its founder count is determined.

    FounderCountAmbiguous is the final count step's call (the converged count pass of the last assembly level,
    which decides the reported count): True if any of its decisions has a bootstrap band containing zero, or if a
    restored best-F panel fails its own deletion test. The final count step only deletes founders, so it cannot
    cycle; count cycles occur in join searches. A join count cycle (a row that an addition of the same join search
    created is deleted again) keeps the deletion; it is undecided when that deletion is ambiguous and the cycle
    began with a decided deletion: the data could not tell the two counts apart. Join* columns count the final
    assembly's join cycles, undecided cycles and count decisions whose band contains zero over the component;
    Feedback* the same for the local-selection context assemblies, whose panels shape the final assembly's input.
    UndecidedCountCycles is JoinUndecidedCycles plus FeedbackJoinUndecidedCycles (an uncovered side counts zero).
    Empty values: no count step or no join search covers the component; UndecidedCountCycles is empty only when
    the component has neither.
    """
    rows = []
    for component_index, component in enumerate(painting.components):
        panel = component.panel
        first, last = int(panel.positions[0]), int(panel.positions[-1])
        call = panel.count_call
        join = _count_calls(searches, first, last)
        fed = _count_calls(feedback, first, last)
        decisions = None if call is None else call.decisions
        undecided = [join[1], fed[1]]
        rows.append(
            dict(
                Component=component_index,
                FirstVariant1Based=first,
                LastVariant1Based=last,
                FounderRows=int(panel.released.shape[0]),
                FounderCountAmbiguous=None if call is None else call.ambiguous,
                UndecidedCountCycles=(
                    None
                    if call is None and all(part is None for part in undecided)
                    else sum(part or 0 for part in undecided)
                ),
                CountStepFoundersBefore=None if call is None else call.founders_before,
                CountStepAmbiguousDecisions=(
                    None if decisions is None else sum(bool(decision.ambiguous) for decision in decisions)
                ),
                CountStepDecisions=None
                if decisions is None
                else [
                    dict(
                        kind=decision.kind,
                        founders=decision.founders,
                        accept_larger=decision.accept_larger,
                        skipped_by_bound=decision.skipped_by_bound,
                        margin=decision.margin,
                        band=None if decision.band is None else list(decision.band),
                        ambiguous=decision.ambiguous,
                    )
                    for decision in decisions
                ],
                JoinCountCycles=join[0],
                JoinUndecidedCycles=join[1],
                JoinAmbiguousCountCalls=join[2],
                FeedbackJoinCountCycles=fed[0],
                FeedbackJoinUndecidedCycles=fed[1],
                FeedbackJoinAmbiguousCountCalls=fed[2],
            )
        )
    return rows


def sample_switch_table(painting, input_ids=None):
    """Painted switches per sample on one chromosome, relative to the cohort median.

    Counts are the painter's own biological switches (minimum homologue label changes of the Viterbi state
    path; a pure track swap is gauge), summed over the chromosome's components. The median is taken over samples
    painted in at least one component and is not split by generation. A sample far above it is one the founder
    panel does not explain as a founder mosaic (for example foreign reads, or a haplotype missing from the panel),
    or it marks a panel error. Diagnostic only: nothing is filtered or reweighted.
    """
    import pandas as pd

    count = len(painting.sample_ids)
    switches = np.zeros(count, dtype=np.int64)
    painted = np.zeros(count, dtype=np.int64)
    for component in painting.components:
        switches += np.asarray(component.switch_counts, dtype=np.int64)
        painted += np.asarray(component.eligible, dtype=bool)
    median = float(np.median(switches[painted > 0])) if np.any(painted) else float("nan")
    ratio = np.full(count, np.nan)
    if median > 0:
        ratio[painted > 0] = switches[painted > 0] / median
    frame = pd.DataFrame(
        dict(
            SampleIndex=np.arange(count),
            SampleID=list(painting.sample_ids),
            PaintedComponents=painted,
            PaintedSwitches=switches,
            CohortMedianSwitches=median,
            SwitchesOverCohortMedian=ratio,
        )
    )
    if input_ids is not None:
        # the records hold the canonical sample order; SampleIndex is the input (VCF) position.
        from .tables import frame_in_input_order

        position = {str(name): index for index, name in enumerate(input_ids)}
        frame["SampleIndex"] = [position.get(str(name), -1) for name in frame["SampleID"]]
        frame = frame_in_input_order(frame, input_ids, column="SampleID")
    return frame


def contig_painting(output: str | Path, results: Results, contig: str):
    """Publish all painting tables from the typed results report view."""
    view = results.painting_report(contig)
    assert view is not None  # the writer runs after the contig's painting task published it
    return contig_part(
        output,
        "painting",
        contig,
        painting_frames(
            view.painting,
            view.positions,
            view.observed,
            view.searches,
            view.feedback_searches,
            results.input_sample_ids,
        ),
    )


def painting_frames(painting, positions, observed, searches=(), feedback=(), input_ids=None):
    """Format the eight painting tables; all inputs are recorded views."""
    import pandas as pd

    support, windows, search_rows = founder_support_tables(painting, positions, observed, searches)
    switches = sample_switch_table(painting, input_ids)
    counts = founder_count_table(painting, searches, feedback)
    search_rows, alternatives, search_items = assembly_search_tables(search_rows)
    frames = dict(
        zip(
            TABLES,
            (
                pd.DataFrame(support, columns=None if support else FOUNDER_SUPPORT_COLUMNS),
                pd.DataFrame(windows, columns=None if windows else FOUNDER_SUPPORT_WINDOW_COLUMNS),
                pd.DataFrame(search_rows, columns=None if search_rows else ASSEMBLY_SEARCH_PREFIX_COLUMNS),
                switches,
                pd.DataFrame(counts, columns=None if counts else FOUNDER_COUNT_COLUMNS),
            ),
            strict=True,
        )
    )
    frames["assembly_search_shadow_alternatives"] = pd.DataFrame(alternatives, columns=SHADOW_ALTERNATIVE_COLUMNS)
    frames["assembly_search_items"] = pd.DataFrame(search_items, columns=SEARCH_ITEM_COLUMNS)
    frames["sample_segments"] = pd.DataFrame(segment_rows(painting, input_ids), columns=SEGMENT_COLUMNS)
    frames["label_continuity"] = pd.DataFrame(label_continuity_rows(painting), columns=CONTINUITY_COLUMNS)
    frames["painting_model"] = pd.DataFrame(painting_model_rows(painting), columns=MODEL_COLUMNS)
    return frames


def write_painting(output, parts) -> list[Path]:
    """Publish all run-level painting tables in configured contig order."""
    return write_contig_tables(output, "painting", parts)


SEGMENT_COLUMNS = ("SampleID", "Component", "Track", "Start", "End", "Status", "Label", "Rows")
"""Columns in published order.

    SampleID: sample identifier (text).
    Component: zero-based component index (1).
    Track: homologue gauge, 0 or 1 (1).
    Start: inclusive segment start (bp).
    End: exclusive segment end (bp).
    Status: TrackStatus enum name (text).
    Label: named founder row or empty for an unresolved label (text).
    Rows: semicolon-separated candidate founder rows (text).
"""
CONTINUITY_COLUMNS = (
    "Component",
    "AtomicSpan",
    "Position",
    "RowA",
    "RowB",
    "ScoreGainSwitches",
    "PaintedSwitchChange",
    "Winners",
    "Losers",
    "WinnersAtoB",
    "WinnersBtoA",
    "Identity",
)
"""Columns in published order.

    Component: zero-based component index (1).
    AtomicSpan: atomic span (1).
    Position: position (bp).
    RowA: row a (1).
    RowB: row b (1).
    ScoreGainSwitches: score gain switches (1).
    PaintedSwitchChange: painted switch change (1).
    Winners: winners (1).
    Losers: losers (1).
    WinnersAtoB: winners ato b (1).
    WinnersBtoA: winners bto a (1).
    Identity: identity (1).
"""
MODEL_COLUMNS = (
    "Component",
    "FirstVariant1Based",
    "LastVariant1Based",
    "Bins",
    "States",
    "ErrorCapableClasses",
    "SwitchRatePerMorgan",
    "ErrorEntry",
    "ErrorExit",
    "ErrorRunBinsKeepingLabel",
    "SiteCorrelation",
    "EligibleSamples",
    "ContinuityBudgetReached",
)
"""Columns in published order.

    Component: zero-based component index (1).
    FirstVariant1Based: first variant coordinate, inclusive (bp).
    LastVariant1Based: last variant coordinate, inclusive (bp).
    Bins: bins (1).
    States: states (1).
    ErrorCapableClasses: error capable classes (1).
    SwitchRatePerMorgan: biological switch rate per Morgan (Morgan^-1).
    ErrorEntry: entry probability of the copying-model error chain (1).
    ErrorExit: exit probability of the copying-model error chain (1).
    ErrorRunBinsKeepingLabel: 1 / ErrorExit; empty when ErrorExit is zero (bins).
    SiteCorrelation: fitted within-site emission correlation (1).
    EligibleSamples: eligible samples (1).
    ContinuityBudgetReached: label continuity's search ended at its per-iteration check budget:
        candidates ranked below the budget were not checked (bool).
"""


def painting_model_rows(painting) -> list[tuple]:
    """One row per component with bins: the copying model painting estimated from all its samples
    (painting.fidelity.estimate_chain): rho, the panel-error chain's entry a and exit b (ErrorRunBinsKeepingLabel =
    1 / b, the mean length of a run that keeps its label; a at its floor 1e-6 is a boundary fit: no support for error
    runs) and the site correlation r of the emissions' design effect."""
    rows = []
    continuity = getattr(painting, "label_continuity", ())
    for index, component in enumerate(painting.components):
        if component.bins.n_bins == 0:
            continue
        transitions = component.transitions
        capable = int(sum(1 for value in component.states.anchored if value))
        rows.append(
            (
                index,
                int(component.states.positions[0]),
                int(component.states.positions[-1]),
                int(component.bins.n_bins),
                int(component.states.n_states),
                capable,
                float(transitions.switch_rate),
                float(transitions.error_entry),
                float(transitions.error_exit),
                (1.0 / transitions.error_exit) if transitions.error_exit > 0 else None,
                float(transitions.site_correlation),
                int(component.eligible.sum()),
                bool(getattr(continuity[index], "check_budget_reached", False)) if index < len(continuity) else False,
            )
        )
    return rows


def label_continuity_rows(painting) -> list[tuple]:
    """One row per accepted label-continuity move: Identity is 'two_way' when winners switch in both
    directions (a label exchange of two founders' carriers) and 'ambiguous' otherwise (consistent with one shared
    recombinant lineage: the relabelling is kept, the founder identity across the link is not identified)."""
    rows = []
    for component, outcome in enumerate(painting.label_continuity):
        if not outcome.accepted:
            continue
        for move in outcome.moves:
            rows.append(
                (
                    component,
                    move.atomic_span,
                    move.position,
                    move.rows[0],
                    move.rows[1],
                    round(move.score_gain_switches, 3),
                    move.painted_switch_change,
                    len(move.winners),
                    len(move.losers),
                    move.winners_forward,
                    move.winners_reverse,
                    "two_way" if move.identified else "ambiguous",
                )
            )
    return rows


def segment_rows(painting: Painting, input_ids=None) -> list[dict]:
    """The released painting as segments: one row per sample, component, track and run of bins with
    one release (status, label and candidate rows), in sample, component, track and position order.

    Columns: SampleID, Component, Track (0/1, the painting's gauge: which homologue is track 0 carries no meaning),
    Start, End (bp, [Start, End)), Status (TrackStatus name), Label (the founder row of a SINGLETON_NAMED segment,
    else empty), Rows (candidate founder rows, ';'-joined; component-local). A SWITCH_INTERVAL segment is the bp
    interval where the track switches between its candidate rows; the segments before and after it name the two
    sides. Ineligible samples have no rows. ``input_ids`` (the run's input sample order) orders the samples; default
    the records' order."""
    from ...contracts.painting import TrackStatus

    names = {int(status.value): status.name for status in TrackStatus}
    order = list(range(len(painting.sample_ids)))
    if input_ids is not None:
        position = {str(name): index for index, name in enumerate(input_ids)}
        order.sort(key=lambda sample: position.get(str(painting.sample_ids[sample]), len(position)))
    rows = []
    for sample in order:
        sample_id = painting.sample_ids[sample]
        for index, component in enumerate(painting.components):
            if not component.eligible[sample] or component.bins.n_bins == 0:
                continue
            edges = component.bins.edges
            n_bins = component.bins.n_bins
            for track in (0, 1):
                status = component.status[sample, track]
                labels = component.labels[sample, track]
                ids = component.member_ids[sample, track]
                changed = np.ones(n_bins, dtype=np.bool_)
                changed[1:] = (status[1:] != status[:-1]) | (labels[1:] != labels[:-1]) | (ids[1:] != ids[:-1])
                starts = np.flatnonzero(changed)
                stops = np.r_[starts[1:], n_bins]
                for start, stop in zip(starts.tolist(), stops.tolist(), strict=False):
                    label = int(labels[start])
                    rows.append(
                        dict(
                            SampleID=str(sample_id),
                            Component=index,
                            Track=track,
                            Start=int(edges[start]),
                            End=int(edges[stop]),
                            Status=names[int(status[start])],
                            Label="" if label < 0 else str(label),
                            Rows=";".join(map(str, component.member_sets[ids[start]])),
                        )
                    )
    return rows


# Published table schemas, including contigs that contribute no rows.
# Header spelling is normalized by the shared CSV writer.
FOUNDER_SUPPORT_COLUMNS = (
    "Component",
    "Founder",
    "FirstVariant1Based",
    "LastVariant1Based",
    "BreakBefore",
    "BreakAfter",
    "MarkerCount",
    "CalledAlleles",
    "MissingAlleles",
    "ObservedCarrierSamples",
    "CalledSitesWithoutObservedNamedCarrier",
    "MinimumObservedCarriers",
    "MeanObservedCarriers",
    "ObservedPooledCandidateSamples",
    "MeanObservedPooledCandidates",
    "CalledSitesWithoutObservedNamedOrPooledCandidate",
    "SupportBasis",
)
"""Columns in published order.

    Component: zero-based component index (1).
    Founder: founder row label within the component (1).
    FirstVariant1Based: first variant coordinate, inclusive (bp).
    LastVariant1Based: last variant coordinate, inclusive (bp).
    BreakBefore: break before (1).
    BreakAfter: break after (1).
    MarkerCount: marker count (1).
    CalledAlleles: called alleles (1).
    MissingAlleles: missing alleles (1).
    ObservedCarrierSamples: observed carrier samples (1).
    CalledSitesWithoutObservedNamedCarrier: called sites without observed named carrier (1).
    MinimumObservedCarriers: minimum named observed carriers over called alleles; empty without calls (1).
    MeanObservedCarriers: mean named observed carriers over called alleles; empty without calls (1).
    ObservedPooledCandidateSamples: observed pooled candidate samples (1).
    MeanObservedPooledCandidates: mean pooled observed candidates over called alleles; empty without calls (1).
    CalledSitesWithoutObservedNamedOrPooledCandidate: called sites without observed named or pooled candidate (1).
    SupportBasis: recorded description of named and pooled observed-carrier support (text).
"""

FOUNDER_SUPPORT_WINDOW_COLUMNS = (
    "Component",
    "Founder",
    "FirstVariant1Based",
    "LastVariant1Based",
    "MarkerCount",
    "CalledAlleles",
    "MissingAlleles",
    "CalledSitesWithoutObservedNamedCarrier",
    "MeanObservedCarriers",
    "MeanObservedPooledCandidates",
    "CalledSitesWithoutObservedNamedOrPooledCandidate",
)
"""Columns in published order.

    Component: zero-based component index (1).
    Founder: founder row label within the component (1).
    FirstVariant1Based: first variant coordinate, inclusive (bp).
    LastVariant1Based: last variant coordinate, inclusive (bp).
    MarkerCount: marker count (1).
    CalledAlleles: called alleles (1).
    MissingAlleles: missing alleles (1).
    CalledSitesWithoutObservedNamedCarrier: called sites without observed named carrier (1).
    MeanObservedCarriers: mean named observed carriers over called alleles; empty without calls (1).
    MeanObservedPooledCandidates: mean pooled observed candidates over called alleles; empty without calls (1).
    CalledSitesWithoutObservedNamedOrPooledCandidate: called sites without observed named or pooled candidate (1).
"""

ASSEMBLY_SEARCH_PREFIX_COLUMNS = ("Level", "FirstVariant1Based", "LastVariant1Based", "assembly_search_row")
"""Columns in published order.

    assembly_search_row: zero-based assembly-search row within this contig, including closing records (1).
    Level: level (1).
    FirstVariant1Based: first variant coordinate, inclusive (bp).
    LastVariant1Based: last variant coordinate, inclusive (bp).
"""

FOUNDER_COUNT_COLUMNS = (
    "Component",
    "FirstVariant1Based",
    "LastVariant1Based",
    "FounderRows",
    "FounderCountAmbiguous",
    "UndecidedCountCycles",
    "CountStepFoundersBefore",
    "CountStepAmbiguousDecisions",
    "CountStepDecisions",
    "JoinCountCycles",
    "JoinUndecidedCycles",
    "JoinAmbiguousCountCalls",
    "FeedbackJoinCountCycles",
    "FeedbackJoinUndecidedCycles",
    "FeedbackJoinAmbiguousCountCalls",
)
"""Columns in published order.

    Component: zero-based component index (1).
    FirstVariant1Based: first variant coordinate, inclusive (bp).
    LastVariant1Based: last variant coordinate, inclusive (bp).
    FounderRows: founder rows (1).
    FounderCountAmbiguous: founder count ambiguous (1).
    UndecidedCountCycles: undecided count cycles of the final and feedback join searches (1).
    CountStepFoundersBefore: count step founders before (1).
    CountStepAmbiguousDecisions: count step ambiguous decisions (1).
    CountStepDecisions: JSON list of the recorded count-step decisions (text).
    JoinCountCycles: join count cycles (1).
    JoinUndecidedCycles: join undecided cycles (1).
    JoinAmbiguousCountCalls: join ambiguous count calls (1).
    FeedbackJoinCountCycles: feedback join count cycles (1).
    FeedbackJoinUndecidedCycles: feedback join undecided cycles (1).
    FeedbackJoinAmbiguousCountCalls: feedback join ambiguous count calls (1).
"""

SAMPLE_SWITCH_COLUMNS = (
    "SampleIndex",
    "SampleID",
    "PaintedComponents",
    "PaintedSwitches",
    "CohortMedianSwitches",
    "SwitchesOverCohortMedian",
)
"""Columns in published order.

    SampleIndex: zero-based input sample index (1).
    SampleID: sample identifier (text).
    PaintedComponents: painted components (1).
    PaintedSwitches: painted switches (1).
    CohortMedianSwitches: cohort median switches (1).
    SwitchesOverCohortMedian: painted switches divided by the cohort median; NaN if median is not positive (1).
"""
