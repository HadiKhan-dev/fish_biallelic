"""Read-only founder callability, observed-carrier and assembly-search reports.

Carrier support uses released named painting tracks intersected with positive read
depth. It is painting-dependent, not independent lineage evidence or a phasing
probability. Unknown/pooled ancestry does not count as a named carrier. Labels
are component-local and must not be joined across a structural break.
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
        # One count per sample/site, even if both homologues have this label.
        support = np.zeros(calls.shape, dtype=np.int32)
        carriers = np.zeros((len(calls), len(checkpoint.sample_ids)), dtype=bool)
        for sample in component.painting.samples:
            for chunk in sample.chunks:
                lo, hi = np.searchsorted(positions, [chunk.start, chunk.end])
                obs = observed[sample.sample_index, index[lo:hi]]
                for founder in set((chunk.hap1, chunk.hap2)):
                    if founder < 0:
                        continue
                    usable = obs & (calls[founder, lo:hi] >= 0)
                    support[founder, lo:hi] += usable
                    carriers[founder, sample.sample_index] |= np.any(usable)
        for founder, key in enumerate(component.founder_keys):
            called = calls[founder] >= 0
            selected = support[founder, called]
            base = dict(Component=component.component_index, Founder=str(key),
                        FirstVariant1Based=int(positions[0]), LastVariant1Based=int(positions[-1]),
                        BreakBefore=record["break_before"], BreakAfter=record["break_after"])
            summaries.append(dict(
                **base, MarkerCount=len(positions), CalledAlleles=int(called.sum()),
                MissingAlleles=int((~called).sum()), ObservedCarrierSamples=int(carriers[founder].sum()),
                CalledSitesWithoutObservedNamedCarrier=int(np.sum(selected == 0)),
                MinimumObservedCarriers=None if not len(selected) else int(selected.min()),
                MeanObservedCarriers=None if not len(selected) else float(selected.mean()),
                SupportBasis="released named painting ancestry AND positive depth; correlated samples"))
            for start in range(0, len(positions), window_sites):
                stop = min(len(positions), start + window_sites)
                mask = called[start:stop]
                values = support[founder, start:stop][mask]
                windows.append(dict(Component=component.component_index, Founder=str(key),
                                    FirstVariant1Based=int(positions[start]),
                                    LastVariant1Based=int(positions[stop-1]), MarkerCount=stop-start,
                                    CalledAlleles=int(mask.sum()), MissingAlleles=int((~mask).sum()),
                                    CalledSitesWithoutObservedNamedCarrier=int(np.sum(values == 0)),
                                    MeanObservedCarriers=None if not len(values) else float(values.mean())))
        for level in getattr(block, "assembly_search_history", ()):
            for panel in level["panels"]:
                for diagnostic in panel["records"]:
                    search.append(dict(Level=level["level"],
                                       FirstVariant1Based=panel["position_start"],
                                       LastVariant1Based=panel["position_end"], **diagnostic))
    return pd.DataFrame(summaries), pd.DataFrame(windows), pd.DataFrame(search)


def write_founder_support(checkpoint, sites, observed, output):
    """Export small TSVs atomically without changing the scientific checkpoint."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = founder_support_tables(checkpoint, sites, observed)
    for name, frame in zip(("founder_support", "founder_support_windows", "assembly_search"), tables):
        frame = frame.copy()
        for column in frame:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(
                    lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value)
        temporary = output / f".{name}.tsv.tmp"
        frame.to_csv(temporary, sep="\t", index=False)
        temporary.replace(output / f"{name}.tsv")
    return tables
