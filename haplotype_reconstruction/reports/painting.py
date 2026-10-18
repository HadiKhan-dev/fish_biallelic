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

Carrier support uses released named painting tracks intersected with positive read depth: it is
painting-dependent, not independent lineage evidence or a phasing probability, and samples are correlated.
Unknown/pooled ancestry does not count as a named carrier. Labels are component-local. Diagnostics only: nothing
here is filtered, reweighted or fed back.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .tables import json_cells, write_frame

WINDOW_SITES = 200
SUPPORT_BASIS = "released named painting ancestry AND positive depth; correlated samples"
TABLES = ("founder_support", "founder_support_windows", "assembly_search", "sample_switches", "founder_counts")


# ------------------------------------------------------------------------------------- assembly searches
def feedback_searches(file) -> list[tuple[int, int, int, object]]:
    """The join searches of the local-selection rounds' context assemblies (base: the feedback assemblies), whose
    panels shape the final assembly's input; rounds without a context assembly contribute none."""
    from ..assembly import records as assembly_records
    from ..local_selection import records as selection_records
    return [search for name in selection_records.ROUNDS
            for search in assembly_records.join_searches(file, selection_records.context_assembly_prefix(name))]


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
        raise ValueError("founder support and observation axes do not match")
    summaries, windows = [], []
    for component_index, component in enumerate(painting.components):
        panel = component.panel
        positions = np.asarray(panel.positions)
        index = np.searchsorted(sites, positions)
        if np.any(index >= len(sites)) or not np.array_equal(sites[index], positions):
            raise ValueError("founder support and observation axes do not match")
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
            base = dict(Component=component_index, Founder=str(founder),
                        FirstVariant1Based=int(positions[0]), LastVariant1Based=int(positions[-1]),
                        BreakBefore=panel.break_before is not None, BreakAfter=panel.break_after is not None)
            summaries.append(dict(
                **base, MarkerCount=len(positions), CalledAlleles=int(called.sum()),
                MissingAlleles=int((~called).sum()), ObservedCarrierSamples=int(carriers[founder].sum()),
                CalledSitesWithoutObservedNamedCarrier=int(np.sum(selected == 0)),
                MinimumObservedCarriers=None if not len(selected) else int(selected.min()),
                MeanObservedCarriers=None if not len(selected) else float(selected.mean()),
                ObservedPooledCandidateSamples=int(candidates[founder].sum()),
                MeanObservedPooledCandidates=None if not len(selected) else float(selected_pooled.mean()),
                CalledSitesWithoutObservedNamedOrPooledCandidate=int(np.sum(selected + selected_pooled == 0)),
                SupportBasis=SUPPORT_BASIS))
            for start in range(0, len(positions), window_sites):
                stop = min(len(positions), start + window_sites)
                mask = called[start:stop]
                values = support[founder, start:stop][mask]
                pooled_values = pooled[founder, start:stop][mask]
                windows.append(dict(Component=component_index, Founder=str(founder),
                                    FirstVariant1Based=int(positions[start]),
                                    LastVariant1Based=int(positions[stop - 1]), MarkerCount=stop - start,
                                    CalledAlleles=int(mask.sum()), MissingAlleles=int((~mask).sum()),
                                    CalledSitesWithoutObservedNamedCarrier=int(np.sum(values == 0)),
                                    MeanObservedCarriers=None if not len(values) else float(values.mean()),
                                    MeanObservedPooledCandidates=(
                                        None if not len(values) else float(pooled_values.mean())),
                                    CalledSitesWithoutObservedNamedOrPooledCandidate=int(
                                        np.sum(values + pooled_values == 0))))
    search_rows = [dict(Level=level, FirstVariant1Based=first, LastVariant1Based=last, **record)
                   for level, first, last, summary in searches for record in summary.records()]
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
        rows.append(dict(
            Component=component_index, FirstVariant1Based=first, LastVariant1Based=last,
            FounderRows=int(panel.released.shape[0]),
            FounderCountAmbiguous=None if call is None else call.ambiguous,
            UndecidedCountCycles=(None if all(part is None for part in parts)
                                  else sum(part or 0 for part in parts)),
            CountStepFoundersBefore=None if call is None else call.founders_before,
            CountStepCycle=None if call is None else call.count_cycle,
            CountStepCycleUndecided=None if call is None else call.cycle_undecided,
            CountStepAmbiguousDecisions=(None if decisions is None
                                         else sum(bool(decision.ambiguous) for decision in decisions)),
            CountStepDecisions=None if decisions is None else [
                dict(kind=decision.kind, founders=decision.founders, accept_larger=decision.accept_larger,
                     skipped_by_bound=decision.skipped_by_bound, margin=decision.margin,
                     band=None if decision.band is None else list(decision.band), ambiguous=decision.ambiguous)
                for decision in decisions],
            JoinCountCycles=join[0], JoinUndecidedCycles=join[1], JoinAmbiguousCountCalls=join[2],
            FeedbackJoinCountCycles=fed[0], FeedbackJoinUndecidedCycles=fed[1],
            FeedbackJoinAmbiguousCountCalls=fed[2]))
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
    frame = pd.DataFrame(dict(SampleIndex=np.arange(count), SampleID=list(painting.sample_ids),
                              PaintedComponents=painted, PaintedSwitches=switches,
                              CohortMedianSwitches=median, SwitchesOverCohortMedian=ratio))
    if input_ids is not None:
        # Release 2.1: the records hold the canonical sample order; SampleIndex is the input (VCF) position.
        from ..evidence.sample_order import frame_in_input_order
        position = {str(name): index for index, name in enumerate(input_ids)}
        frame["SampleIndex"] = [position.get(str(name), -1) for name in frame["SampleID"]]
        frame = frame_in_input_order(frame, input_ids, column="SampleID")
    return frame


def report_bytes(file, samples: int, sites: int) -> int:
    """Planned peak bytes of one contig's painting tables: the decoded painting record and, twice, the final
    assembly's prepared panels (the painted panels are materialised from them; the per-component carrier counts
    int32 [K, L_c] are of the same order), the evidence positions and observation mask bool [N, L]; 5/4 for
    allocator slack. Measured on std_n80 chr1 (N 80, L 329,391): 142 MiB for an estimate of 214 MiB."""
    from ..assembly import records as assembly_records
    from ..painting import records as painting_records
    completion = file.info(f"{assembly_records.FINAL_PREFIX}/{assembly_records.COMPLETION}")
    decoded = (file.info(painting_records.KEY).decoded_bytes
               + 2 * (0 if completion is None else completion.decoded_bytes))
    return (decoded + samples * sites + 16 * sites) * 5 // 4


def write_painting(output, store, contig: str) -> list[Path]:
    """Write the contig's painting tables (module docstring) and log their highlights; returns the paths. The
    decoding holds a reservation of the run's memory (report_bytes; waits while it does not fit)."""
    from ..evidence import records as evidence_records
    from ..execution.memory import reserve_memory
    from ..painting import records as painting_records
    file = store.contig(contig)
    positions = evidence_records.load_arrays(file, ("positions",))["positions"]
    samples = len(file.get(painting_records.KEY, items=("sample_ids",))["sample_ids"])
    with reserve_memory(report_bytes(file, samples, len(positions)), label="painting report"):
        return _write_tables(output, file, contig, positions, store.manifest("run"))


def _write_tables(output, file, contig: str, positions, store_manifest=None) -> list[Path]:
    import pandas as pd
    from ..assembly import records as assembly_records
    from ..evidence import records as evidence_records
    from ..painting import records as painting_records
    painting = painting_records.load(file, panels=True, emissions=False)
    observed = evidence_records.load_arrays(file, ("observed",))["observed"]
    searches = assembly_records.join_searches(file)
    support, windows, search_rows = founder_support_tables(painting, positions, observed, searches)
    del observed
    from ..evidence.sample_order import recorded_input_order
    switches = sample_switch_table(painting, recorded_input_order(store_manifest))
    directory = Path(output) / "painting" / contig
    highest = switches[switches.PaintedComponents > 0].nlargest(3, "PaintedSwitches", keep="first")
    listed = ", ".join(f"{row.SampleIndex} ({row.SampleID}) {row.PaintedSwitches} "
                       f"({row.SwitchesOverCohortMedian:.1f}x)" for row in highest.itertuples())
    print(f"[Painting] {contig}: sample switches, cohort median "
          f"{switches.CohortMedianSwitches.iloc[0]:g}; highest {listed}", flush=True)
    counts = founder_count_table(painting, searches, feedback_searches(file))
    flagged = [row["Component"] for row in counts
               if row["FounderCountAmbiguous"] not in (None, False) or (row["UndecidedCountCycles"] or 0) > 0]
    if flagged:
        print(f"[Painting] {contig}: founder count ambiguous or undecided count cycles in component(s) "
              f"{', '.join(map(str, flagged))} (founder_counts.tsv)", flush=True)
    frames = (pd.DataFrame(support), pd.DataFrame(windows), pd.DataFrame(search_rows), switches,
              pd.DataFrame(counts))
    return [write_frame(directory / f"{name}.tsv", json_cells(frame), sep="\t")
            for name, frame in zip(TABLES, frames)]
