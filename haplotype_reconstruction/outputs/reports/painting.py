"""OUTPUT/painting/<contig>/: read-only founder callability, carrier support, assembly-search, founder-count
and sample-switch tables of one painted chromosome.

  founder_support.tsv          per component and founder row: called/missing alleles, observed named carriers
                               (samples whose released painting names the row at a site with read depth),
                               pooled candidates (tracks the reads cannot separate between identical rows)
  founder_support_windows.tsv  the same per 200-site window
  assembly_search.tsv          every record of the final assembly's join searches (level, span, record fields)
  founder_counts.tsv           per component: how firmly its founder count is determined (final count step,
                               count cycles and band-straddling decisions of the final and feedback join searches)
  sample_switches.tsv          painted switches per sample relative to the cohort median
  label_continuity.tsv         every accepted label-continuity move with its winners' switch directions and
                               whether the founder identity across its link is identified (release 2.14)
  sample_segments.tsv          the released painting per sample: runs of bins with one release (status, named
                               founder row, candidate rows), switch intervals included (release 2.2)
  painting_model.tsv           per component: the estimated copying model (ancestry-change rate, panel-error
                               entry and exit, the emissions' site correlation; release 2.14)

Carrier support uses released named painting tracks intersected with positive read depth: it is
painting-dependent, not independent lineage evidence or a phasing probability, and samples are correlated.
Unknown/pooled ancestry does not count as a named carrier. Labels are component-local. Diagnostics only: nothing
here is filtered, reweighted or fed back.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...results.reader import Results

import numpy as np

from ...contracts.painting import Painting
from ...core.errors import InvariantError
from .tables import json_cells, write_frame

WINDOW_SITES = 200
SUPPORT_BASIS = "released named painting ancestry AND positive depth; correlated samples"
TABLES = ("founder_support", "founder_support_windows", "assembly_search", "sample_switches", "founder_counts")


# ------------------------------------------------------------------------------------- assembly searches
def feedback_searches(results, contig: str):
    """Read the typed local-selection context join summaries for this contig."""
    return results.painting_report(contig).feedback_searches


def _count_calls(searches, first: int, last: int):
    """(count cycles, undecided cycles, band-straddling count decisions) of the join searches overlapping
    [first, last]; (None, None, None) when no join search covers the span. Each search counts once (base summed
    its per-level listings, which repeat a pass-through component's earlier search at every later level)."""
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


# ------------------------------------------------------------------------------------- tables
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
            selected = support[founder, called]
            selected_pooled = pooled[founder, called]
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
                    CalledAlleles=int(called.sum()),
                    MissingAlleles=int((~called).sum()),
                    ObservedCarrierSamples=int(carriers[founder].sum()),
                    CalledSitesWithoutObservedNamedCarrier=int(np.sum(selected == 0)),
                    MinimumObservedCarriers=None if not len(selected) else int(selected.min()),
                    MeanObservedCarriers=None if not len(selected) else float(selected.mean()),
                    ObservedPooledCandidateSamples=int(candidates[founder].sum()),
                    MeanObservedPooledCandidates=None if not len(selected) else float(selected_pooled.mean()),
                    CalledSitesWithoutObservedNamedOrPooledCandidate=int(np.sum(selected + selected_pooled == 0)),
                    SupportBasis=SUPPORT_BASIS,
                )
            )
            for start in range(0, len(positions), window_sites):
                stop = min(len(positions), start + window_sites)
                mask = called[start:stop]
                values = support[founder, start:stop][mask]
                pooled_values = pooled[founder, start:stop][mask]
                windows.append(
                    dict(
                        Component=component_index,
                        Founder=str(founder),
                        FirstVariant1Based=int(positions[start]),
                        LastVariant1Based=int(positions[stop - 1]),
                        MarkerCount=stop - start,
                        CalledAlleles=int(mask.sum()),
                        MissingAlleles=int((~mask).sum()),
                        CalledSitesWithoutObservedNamedCarrier=int(np.sum(values == 0)),
                        MeanObservedCarriers=None if not len(values) else float(values.mean()),
                        MeanObservedPooledCandidates=(None if not len(values) else float(pooled_values.mean())),
                        CalledSitesWithoutObservedNamedOrPooledCandidate=int(np.sum(values + pooled_values == 0)),
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
    which decides the reported count): True if any of its decisions has a bootstrap band containing zero, if it
    cycled (an addition whose deletion round then deleted), or if a restored best-F panel fails its own deletion
    test. Count cycles (a row that an addition of the same search created is deleted again) keep the deletion; a
    cycle is undecided when that deletion is ambiguous and the cycle began with a decided deletion: the data could
    not tell the two counts apart. Join* columns count the final assembly's join cycles, undecided cycles and
    count decisions whose band contains zero over the component; Feedback* the same for the local-selection
    context assemblies, whose panels shape the final assembly's input. UndecidedCountCycles sums the undecided
    cycles of all three. Empty values: no count step or no join search covers the component.
    """
    rows = []
    for component_index, component in enumerate(painting.components):
        panel = component.panel
        first, last = int(panel.positions[0]), int(panel.positions[-1])
        call = panel.count_call
        join = _count_calls(searches, first, last)
        fed = _count_calls(feedback, first, last)
        decisions = None if call is None else call.decisions
        parts = [None if call is None else int(call.cycle_undecided), join[1], fed[1]]
        rows.append(
            dict(
                Component=component_index,
                FirstVariant1Based=first,
                LastVariant1Based=last,
                FounderRows=int(panel.released.shape[0]),
                FounderCountAmbiguous=None if call is None else call.ambiguous,
                UndecidedCountCycles=(
                    None if all(part is None for part in parts) else sum(part or 0 for part in parts)
                ),
                CountStepFoundersBefore=None if call is None else call.founders_before,
                CountStepCycle=None if call is None else call.count_cycle,
                CountStepCycleUndecided=None if call is None else call.cycle_undecided,
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
        # Release 2.1: the records hold the canonical sample order; SampleIndex is the input (VCF) position.
        from .tables import frame_in_input_order

        position = {str(name): index for index, name in enumerate(input_ids)}
        frame["SampleIndex"] = [position.get(str(name), -1) for name in frame["SampleID"]]
        frame = frame_in_input_order(frame, input_ids, column="SampleID")
    return frame


def report_bytes(decoded_bytes: int, samples: int, sites: int) -> int:
    """Plan decoded records and carrier counts with the original allocator slack."""
    return (decoded_bytes + samples * sites + 16 * sites) * 5 // 4


def write_painting(output: str | Path, results: Results, contig: str) -> list[Path]:
    """Publish all painting tables from the typed results report view."""
    view = results.painting_report(contig)
    assert view is not None  # the writer runs after the contig's painting task published it
    return write_tables(
        output,
        contig,
        view.painting,
        view.positions,
        view.observed,
        view.searches,
        view.feedback_searches,
        results.input_sample_ids,
    )


def write_tables(
    output, contig: str, painting, positions, observed, searches=(), feedback=(), input_ids=None
) -> list[Path]:
    """Format the eight painting tables; all inputs are recorded views."""
    import pandas as pd

    support, windows, search_rows = founder_support_tables(painting, positions, observed, searches)
    switches = sample_switch_table(painting, input_ids)
    directory = Path(output) / "painting" / contig
    counts = founder_count_table(painting, searches, feedback)
    frames = (pd.DataFrame(support), pd.DataFrame(windows), pd.DataFrame(search_rows), switches, pd.DataFrame(counts))
    written = [
        write_frame(directory / f"{name}.tsv", json_cells(frame), sep="\t")
        for name, frame in zip(TABLES, frames, strict=False)
    ]
    segments = pd.DataFrame(segment_rows(painting, input_ids), columns=SEGMENT_COLUMNS)
    written.append(write_frame(directory / "sample_segments.tsv", segments, sep="\t"))
    written.append(
        write_frame(
            directory / "label_continuity.tsv",
            pd.DataFrame(label_continuity_rows(painting), columns=CONTINUITY_COLUMNS),
            sep="\t",
        )
    )
    written.append(
        write_frame(
            directory / "painting_model.tsv",
            pd.DataFrame(painting_model_rows(painting), columns=MODEL_COLUMNS),
            sep="\t",
        )
    )
    return written


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
"""


def painting_model_rows(painting) -> list[tuple]:
    """One row per component with bins (release 2.14): the copying model painting estimated from all its samples
    (painting.fidelity.estimate_chain): rho, the panel-error chain's entry a and exit b (ErrorRunBinsKeepingLabel =
    1 / b, the mean length of a run that keeps its label; a at its floor 1e-6 is a boundary fit: no support for error
    runs) and the site correlation r of the emissions' design effect."""
    rows = []
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
            )
        )
    return rows


def label_continuity_rows(painting) -> list[tuple]:
    """One row per accepted label-continuity move (release 2.14): Identity is 'two_way' when winners switch in both
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
    """The released painting as segments (release 2.2): one row per sample, component, track and run of bins with
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


# Published column schemas: B reports and P0 outputs inventory D6 / OU15.
# Row-dict tables retain B's headerless empty constructor; these constants do not
# impose columns on those tables. Scientific values remain upstream contract fields.
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

ASSEMBLY_SEARCH_PREFIX_COLUMNS = ("Level", "FirstVariant1Based", "LastVariant1Based")
"""Columns in published order.

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
    "CountStepCycle",
    "CountStepCycleUndecided",
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
    UndecidedCountCycles: undecided count cycles (1).
    CountStepFoundersBefore: count step founders before (1).
    CountStepCycle: count step cycle (1).
    CountStepCycleUndecided: count step cycle undecided (1).
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
