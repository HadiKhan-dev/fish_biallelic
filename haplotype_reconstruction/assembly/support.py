"""Read-only founder callability, observed-carrier, sample-switch, assembly-search and
founder-count reports.

Carrier support uses released named painting tracks intersected with positive read
depth. It is painting-dependent, not independent lineage evidence or a phasing
probability. Unknown/pooled ancestry does not count as a named carrier. Pooled
tracks (rows the sample's reads cannot separate where they are identical) are
listed separately: every candidate row of such a track counts as a pooled
candidate. Labels are component-local and must not be joined across a
structural break.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from ..core import runtime


def founder_support_tables(checkpoint, sites, observed, *, window_sites=200):
    sites = np.asarray(sites)
    observed = np.asarray(observed, dtype=bool)
    summaries, windows, search = [], [], []
    records = checkpoint.component_manifest[runtime.PHASE_COMPONENTS_KEY]
    for record, component in zip(records, checkpoint.painting_bundle.components):
        block = record[runtime.FOUNDER_BLOCK_KEY]
        positions = np.asarray(block.positions)
        index = np.searchsorted(sites, positions)
        if (np.any(index >= len(sites)) or not np.array_equal(sites[index], positions)
                or observed.shape != (len(checkpoint.sample_ids), len(sites))):
            raise ValueError("founder support and observation axes do not match")
        calls = np.asarray(block.discrete_haps)
        # One count per sample/site, even if both homologues have this label
        # (or this candidate row).
        support = np.zeros(calls.shape, dtype=np.int32)
        pooled = np.zeros(calls.shape, dtype=np.int32)
        carriers = np.zeros((len(calls), len(checkpoint.sample_ids)), dtype=bool)
        candidates = np.zeros_like(carriers)
        for sample in component.painting.samples:
            for chunk in sample.chunks:
                lo, hi = np.searchsorted(positions, [chunk.start, chunk.end])
                obs = observed[sample.sample_index, index[lo:hi]]
                tracks = (chunk.members(0), chunk.members(1))
                named = {rows[0] for rows in tracks if len(rows) == 1}
                pooled_rows = {row for rows in tracks if len(rows) >= 2 for row in rows}
                for rows, counts, sample_flags in ((named, support, carriers),
                                                   (pooled_rows, pooled, candidates)):
                    for founder in rows:
                        usable = obs & (calls[founder, lo:hi] >= 0)
                        counts[founder, lo:hi] += usable
                        sample_flags[founder, sample.sample_index] |= np.any(usable)
        for founder, key in enumerate(component.founder_keys):
            called = calls[founder] >= 0
            selected = support[founder, called]
            selected_pooled = pooled[founder, called]
            base = dict(Component=component.component_index, Founder=str(key),
                        FirstVariant1Based=int(positions[0]), LastVariant1Based=int(positions[-1]),
                        BreakBefore=record["break_before"], BreakAfter=record["break_after"])
            summaries.append(dict(
                **base, MarkerCount=len(positions), CalledAlleles=int(called.sum()),
                MissingAlleles=int((~called).sum()), ObservedCarrierSamples=int(carriers[founder].sum()),
                CalledSitesWithoutObservedNamedCarrier=int(np.sum(selected == 0)),
                MinimumObservedCarriers=None if not len(selected) else int(selected.min()),
                MeanObservedCarriers=None if not len(selected) else float(selected.mean()),
                ObservedPooledCandidateSamples=int(candidates[founder].sum()),
                MeanObservedPooledCandidates=(
                    None if not len(selected) else float(selected_pooled.mean())),
                CalledSitesWithoutObservedNamedOrPooledCandidate=int(
                    np.sum(selected + selected_pooled == 0)),
                SupportBasis="released named painting ancestry AND positive depth; correlated samples"))
            for start in range(0, len(positions), window_sites):
                stop = min(len(positions), start + window_sites)
                mask = called[start:stop]
                values = support[founder, start:stop][mask]
                pooled_values = pooled[founder, start:stop][mask]
                windows.append(dict(Component=component.component_index, Founder=str(key),
                                    FirstVariant1Based=int(positions[start]),
                                    LastVariant1Based=int(positions[stop-1]), MarkerCount=stop-start,
                                    CalledAlleles=int(mask.sum()), MissingAlleles=int((~mask).sum()),
                                    CalledSitesWithoutObservedNamedCarrier=int(np.sum(values == 0)),
                                    MeanObservedCarriers=None if not len(values) else float(values.mean()),
                                    MeanObservedPooledCandidates=(
                                        None if not len(values) else float(pooled_values.mean())),
                                    CalledSitesWithoutObservedNamedOrPooledCandidate=int(
                                        np.sum(values + pooled_values == 0))))
        for level in getattr(block, "assembly_search_history", ()):
            for panel in level["panels"]:
                for diagnostic in panel["records"]:
                    search.append(dict(Level=level["level"],
                                       FirstVariant1Based=panel["position_start"],
                                       LastVariant1Based=panel["position_end"], **diagnostic))
    return pd.DataFrame(summaries), pd.DataFrame(windows), pd.DataFrame(search)


def _count_calls(panels, first, last):
    """Count cycles, undecided cycles and band-straddling count decisions of the join searches
    over [first, last]; None when the records predate the cycle classification."""
    cycles = undecided = ambiguous = 0
    recorded = False
    for panel in panels:
        if panel["position_end"] < first or panel["position_start"] > last:
            continue
        for item in panel["records"] or ():
            if "certified" in item:
                recorded |= "cycles" in item["count"]
                cycles += item["count"].get("cycles", 0)
                undecided += item["count"].get("cycles_undecided", 0)
            elif item.get("stage") in ("count_death", "count_birth"):
                ambiguous += bool(item.get("ambiguous"))
    return (cycles, undecided, ambiguous) if recorded else (None, None, None)


def founder_count_table(checkpoint, feedback=()):
    """One row per component: how firmly its founder count is determined.

    FounderCountAmbiguous is the final count step's call (the converged count pass of the
    last assembly level, which decides the reported count): True if any of its decisions has
    a bootstrap band containing zero, if it cycled (an addition whose deletion round then
    deleted), or if a restored best-F panel fails its own deletion test. Count cycles (a row
    that an addition of the same search created is deleted again) keep the deletion; a cycle
    is undecided when that deletion is ambiguous and the cycle began with a decided deletion
    (count_criterion.pre_cycle_count): the data could not tell the two counts apart.
    Join* columns count the final assembly's join cycles, undecided cycles and count
    decisions whose band contains zero over the component; Feedback* the same for the
    feedback assemblies (``feedback``: their join search records), whose panels shape the
    final assembly's input. UndecidedCountCycles sums the undecided cycles of all three.
    Empty values: the records predate these fields. Diagnostics only; nothing is filtered.
    """
    records = checkpoint.component_manifest[runtime.PHASE_COMPONENTS_KEY]
    blocks = [record[runtime.FOUNDER_BLOCK_KEY] for record in records]
    history = next((block.assembly_search_history for block in blocks
                    if getattr(block, "assembly_search_history", None)), ())
    final_panels = [panel for level in history for panel in level["panels"]]
    rows = []
    for record, component in zip(records, checkpoint.painting_bundle.components):
        block = record[runtime.FOUNDER_BLOCK_KEY]
        first, last = int(block.positions[0]), int(block.positions[-1])
        call = getattr(block, "founder_count_call", None)
        join = _count_calls(final_panels, first, last)
        fed = _count_calls(feedback, first, last)
        decisions = None if call is None else call["decisions"]
        parts = [None if call is None else int(call["cycle_undecided"]), join[1], fed[1]]
        rows.append(dict(
            Component=component.component_index, FirstVariant1Based=first, LastVariant1Based=last,
            FounderRows=len(component.founder_keys),
            FounderCountAmbiguous=None if call is None else call["ambiguous"],
            UndecidedCountCycles=(None if all(part is None for part in parts)
                                  else sum(part or 0 for part in parts)),
            CountStepFoundersBefore=None if call is None else call["founders_before"],
            CountStepCycle=None if call is None else call["count_cycle"],
            CountStepCycleUndecided=None if call is None else call["cycle_undecided"],
            CountStepAmbiguousDecisions=(None if decisions is None
                                         else sum(bool(d["ambiguous"]) for d in decisions)),
            CountStepDecisions=decisions,
            JoinCountCycles=join[0], JoinUndecidedCycles=join[1], JoinAmbiguousCountCalls=join[2],
            FeedbackJoinCountCycles=fed[0], FeedbackJoinUndecidedCycles=fed[1],
            FeedbackJoinAmbiguousCountCalls=fed[2]))
    return pd.DataFrame(rows)


def sample_switch_table(checkpoint):
    """Painted switches per sample on one chromosome, relative to the cohort median.

    Counts are the painter's own biological switches (minimum homologue label
    changes of the Viterbi state path; a pure track swap is gauge), summed over
    the chromosome's components. The median is taken over samples painted in at
    least one component and is not split by generation. A sample far above it is
    one the founder panel does not explain as a founder mosaic (for example
    foreign reads, or a haplotype missing from the panel), or it marks a panel
    error. Diagnostic only: nothing is filtered or reweighted.
    """
    count = len(checkpoint.sample_ids)
    switches = np.zeros(count, dtype=np.int64)
    painted = np.zeros(count, dtype=np.int64)
    for component in checkpoint.painting_bundle.components:
        if component.ragged_diagnostics is not None:
            switches += np.asarray(component.ragged_diagnostics.biological_switch_counts, dtype=np.int64)
        painted += np.asarray(component.evidence_eligible_sample_mask, dtype=bool)
    median = float(np.median(switches[painted > 0])) if np.any(painted) else float("nan")
    ratio = np.full(count, np.nan)
    if median > 0:
        ratio[painted > 0] = switches[painted > 0] / median
    return pd.DataFrame(dict(SampleIndex=np.arange(count), SampleID=list(checkpoint.sample_ids),
                             PaintedComponents=painted, PaintedSwitches=switches,
                             CohortMedianSwitches=median, SwitchesOverCohortMedian=ratio))


def write_founder_support(checkpoint, sites, observed, output, feedback=()):
    """Export small TSVs atomically without changing the scientific checkpoint (``feedback``:
    the feedback assemblies' join search records, for founder_counts.tsv)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = founder_support_tables(checkpoint, sites, observed)
    switches = sample_switch_table(checkpoint)
    highest = switches[switches.PaintedComponents > 0].nlargest(3, "PaintedSwitches", keep="first")
    listed = ", ".join(f"{row.SampleIndex} ({row.SampleID}) {row.PaintedSwitches} "
                       f"({row.SwitchesOverCohortMedian:.1f}x)" for row in highest.itertuples())
    print(f"[Painting] {output.name}: sample switches, cohort median "
          f"{switches.CohortMedianSwitches.iloc[0]:g}; highest {listed}", flush=True)
    counts = founder_count_table(checkpoint, feedback)
    flagged = [row.Component for row in counts.itertuples()
               if row.FounderCountAmbiguous == True or (row.UndecidedCountCycles or 0) > 0]  # noqa: E712
    if flagged:
        print(f"[Painting] {output.name}: founder count ambiguous or undecided count cycles in "
              f"component(s) {', '.join(map(str, flagged))} (founder_counts.tsv)", flush=True)
    for name, frame in zip(("founder_support", "founder_support_windows", "assembly_search",
                            "sample_switches", "founder_counts"), (*tables, switches, counts)):
        frame = frame.copy()
        for column in frame:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(
                    lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value)
        temporary = output / f".{name}.tsv.tmp"
        frame.to_csv(temporary, sep="\t", index=False)
        temporary.replace(output / f"{name}.tsv")
    return tables
