"""Truth-only founder evaluation with fixed component-wide label matching.

Nearest-founder error counts assess precision of all called reconstructed rows.
One-to-one matching additionally exposes duplicated/missing founders. Its objective
counts mismatches AND unknown alleles; a missing row costs the entire component.
Neither metric rematches founder labels at individual markers.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from numba.typed import List
from scipy.optimize import linear_sum_assignment

from ..core.products import panel_alleles
from ..core.chromosome_parallel import current_threads


# Bound temporary counting storage while exposing long panels to site-level
# parallelism. Small panels share a single batch launch across components.
_SITE_TILE = 65536


@njit(parallel=True, cache=True)
def _ancestry_mask(events, counts, offsets, founder_count, sites):
    result = np.empty((founder_count, sites), dtype=np.bool_)
    tiles = (sites + _SITE_TILE - 1) // _SITE_TILE
    for task in prange(founder_count * tiles):
        founder, tile = task // tiles, task % tiles
        left, right = tile * _SITE_TILE, min(sites, (tile + 1) * _SITE_TILE)
        first, last = offsets[founder], offsets[founder + 1]
        cursor = first + np.searchsorted(events[first:last], left, side="right")
        active = counts[cursor - 1] if cursor > first else 0
        for site in range(left, right):
            while cursor < last and events[cursor] <= site:
                active = counts[cursor]
                cursor += 1
            result[founder, site] = active > 0
    return result


def represented_ancestry(truth_painting, positions, founder_count):
    """Presence in any sampled homologue, not independent-lineage identifiability."""
    positions = np.asarray(positions)
    endpoints = [[] for _ in range(founder_count)]
    changes = [[] for _ in range(founder_count)]
    chunks = [chunk for sample in truth_painting.samples for chunk in sample.chunks]
    bounds = np.asarray([(chunk.start, chunk.end) for chunk in chunks])
    indices = np.searchsorted(positions, bounds) if chunks else np.empty((0, 2), dtype=np.int64)
    for chunk, (left, right) in zip(chunks, indices):
        # The terminal endpoint includes the last marker; internal boundaries
        # remain half-open, exactly as in the truth producer.
        if chunk.end == positions[-1]:
            right = len(positions)
        for founder in set((chunk.hap1, chunk.hap2)):
            if 0 <= founder < founder_count:
                endpoints[founder].extend((left, right))
                changes[founder].extend((1, -1))
    offsets = np.zeros(founder_count + 1, dtype=np.int64)
    event_parts, count_parts = [], []
    for founder in range(founder_count):
        event = np.asarray(endpoints[founder], dtype=np.int64)
        change = np.asarray(changes[founder], dtype=np.int64)
        order = np.argsort(event, kind="stable")
        event_parts.append(event[order])
        count_parts.append(np.cumsum(change[order]))
        offsets[founder + 1] = offsets[founder] + len(event)
    events = np.concatenate(event_parts) if event_parts else np.empty(0, dtype=np.int64)
    counts = np.concatenate(count_parts) if count_parts else np.empty(0, dtype=np.int64)
    current_threads()
    return _ancestry_mask(events, counts, offsets, founder_count, len(positions))


@njit(parallel=True, cache=True)
def _count_tiles(panels, indices, truth, represented, tasks):
    # Columns: called errors, represented called, represented errors,
    # represented missing. Absent counts follow by exact integer subtraction.
    parts = np.zeros((len(tasks), len(truth), 4), dtype=np.int64)
    missing = np.zeros(len(tasks), dtype=np.int64)
    for task in prange(len(tasks)):
        component, row, left, right = tasks[task]
        calls, index = panels[component], indices[component]
        unknown = 0
        for site in range(left, right):
            unknown += calls[row, site] < 0
        missing[task] = unknown
        for founder in range(len(truth)):
            errors = rep_called = rep_errors = rep_missing = 0
            for site in range(left, right):
                actual_site = index[site]
                call = calls[row, site]
                active = represented[founder, actual_site]
                if call >= 0:
                    error = call != truth[founder, actual_site]
                    errors += error
                    rep_called += active
                    rep_errors += active and error
                else:
                    rep_missing += active
            parts[task, founder, 0] = errors
            parts[task, founder, 1] = rep_called
            parts[task, founder, 2] = rep_errors
            parts[task, founder, 3] = rep_missing
    return parts, missing


@njit(parallel=True, cache=True)
def _reduce_tiles(parts, missing_parts, row_offsets):
    counts = np.zeros((len(row_offsets) - 1, parts.shape[1], 4), dtype=np.int64)
    missing = np.zeros(len(row_offsets) - 1, dtype=np.int64)
    for row in prange(len(missing)):
        for task in range(row_offsets[row], row_offsets[row + 1]):
            missing[row] += missing_parts[task]
            for founder in range(parts.shape[1]):
                for column in range(4):
                    counts[row, founder, column] += parts[task, founder, column]
    return counts, missing


@njit(parallel=True, cache=True)
def _represented_counts(indices, represented):
    counts = np.zeros((len(indices), len(represented)), dtype=np.int64)
    for task in prange(len(indices) * len(represented)):
        component, founder = task // len(represented), task % len(represented)
        total = 0
        for site in indices[component]:
            total += represented[founder, site]
        counts[component, founder] = total
    return counts


def _count_panels(panels, indices, truth, represented):
    if not panels:
        return [], [], []
    packed_panels, packed_indices = List(), List()
    tasks, row_offsets, component_offsets = [], [0], [0]
    for component, (calls, index) in enumerate(zip(panels, indices)):
        packed_panels.append(np.ascontiguousarray(calls, dtype=np.int8))
        packed_indices.append(np.ascontiguousarray(index, dtype=np.int64))
        for row in range(len(calls)):
            tasks.extend((component, row, left, min(calls.shape[1], left + _SITE_TILE))
                         for left in range(0, calls.shape[1], _SITE_TILE))
            row_offsets.append(len(tasks))
        component_offsets.append(component_offsets[-1] + len(calls))
    current_threads()
    parts, missing_parts = _count_tiles(
        packed_panels, packed_indices, truth, represented,
        np.asarray(tasks, dtype=np.int64).reshape(-1, 4))
    counts, missing = _reduce_tiles(parts, missing_parts, np.asarray(row_offsets, dtype=np.int64))
    rep_counts = _represented_counts(packed_indices, represented)
    return ([counts[left:right] for left, right in zip(component_offsets[:-1], component_offsets[1:])],
            [missing[left:right] for left, right in zip(component_offsets[:-1], component_offsets[1:])],
            rep_counts)


def _compare_counts(counts, missing, represented_counts, sites):
    k, f = counts.shape[:2]
    errors = counts[:, :, 0]
    called = sites - missing
    predicted, actual = linear_sum_assignment(errors + missing[:, None])
    nearest = np.argmin(errors, axis=1) if k else np.empty(0, dtype=int)
    result = dict(founder_rows=k, truth_founders=f, markers=sites,
                  called_alleles=int(called.sum()), missing_alleles=int(missing.sum()),
                  called_allele_errors=int(errors[np.arange(k), nearest].sum()),
                  one_to_one_called_allele_errors=int(errors[predicted, actual].sum()),
                  matched_called_alleles=int(called[predicted].sum()),
                  unmatched_founder_rows=k-len(predicted),
                  unmatched_truth_founders=f-len(actual),
                  unmatched_called_alleles=int(called.sum()-called[predicted].sum()),
                  truth_alleles=f*sites,
                  truth_to_panel_errors_missing=int(
                      errors[predicted, actual].sum() + missing[predicted].sum()
                      + (f-len(actual))*sites))
    result["represented_truth_alleles"] = int(represented_counts.sum())
    result["absent_truth_alleles"] = f*sites - result["represented_truth_alleles"]
    rep_called = int(counts[predicted, actual, 1].sum())
    rep_errors = int(counts[predicted, actual, 2].sum())
    rep_missing = int(counts[predicted, actual, 3].sum()
                      + represented_counts.sum() - represented_counts[actual].sum())
    for label, called_count, error_count, missing_count in (
            ("represented", rep_called, rep_errors, rep_missing),
            ("absent", result["matched_called_alleles"] - rep_called,
             result["one_to_one_called_allele_errors"] - rep_errors,
             int(missing[predicted].sum() + (f-len(actual))*sites) - rep_missing)):
        result[f"{label}_matched_called_alleles"] = called_count
        result[f"{label}_matched_called_errors"] = error_count
        result[f"{label}_truth_to_panel_errors_missing"] = error_count + missing_count
    matches = [dict(predicted_row=int(row), truth_founder=int(founder),
                    called_alleles=int(called[row]),
                    errors=int(errors[row, founder]), missing=int(missing[row]))
               for row, founder in zip(predicted, actual)]
    return result, matches


def compare_panel(calls, truth, represented=None):
    """Compare a K-by-L hard-call panel to F complete truth haplotypes."""
    calls, truth = np.asarray(calls), np.asarray(truth)
    if calls.ndim != 2 or truth.ndim != 2 or calls.shape[1] != truth.shape[1]:
        raise ValueError("predicted and true founder site axes differ")
    if np.any(~np.isin(calls, (-1, 0, 1))) or np.any(~np.isin(truth, (0, 1))):
        raise ValueError("require -1/0/1 calls and complete 0/1 founder truth")
    if len(truth) == 0:
        raise ValueError("founder truth must not be empty")
    represented = (np.ones(truth.shape, dtype=bool) if represented is None
                   else np.asarray(represented, dtype=bool))
    if represented.shape != truth.shape:
        raise ValueError("ancestry representation mask has the wrong axes")
    counts, missing, rep_counts = _count_panels(
        [calls], [np.arange(calls.shape[1])], truth, represented)
    return _compare_counts(counts[0], missing[0], rep_counts[0], calls.shape[1])


def evaluate_founders(blocks, positions, truth, represented, *, contig, stage):
    """Count every truth marker, including markers outside reconstructed spans."""
    positions, truth = np.asarray(positions), np.asarray(truth)
    covered = np.zeros(len(positions), dtype=bool)
    rows, matches = [], []
    panels, indices, metadata = [], [], []
    if len(truth) == 0:
        raise ValueError("founder truth must not be empty")
    if np.any(~np.isin(truth, (0, 1))):
        raise ValueError("require -1/0/1 calls and complete 0/1 founder truth")
    if represented.shape != truth.shape:
        raise ValueError("ancestry representation mask has the wrong axes")
    for component, block in enumerate(blocks):
        sites, keys, calls = panel_alleles(block)
        index = np.searchsorted(positions, sites)
        if np.any(index >= len(positions)) or not np.array_equal(positions[index], sites):
            raise ValueError(f"{contig}: founder truth and reconstruction positions differ")
        if np.any(covered[index]):
            raise ValueError("evaluated founder components must not overlap")
        covered[index] = True
        panels.append(calls)
        indices.append(index)
        metadata.append((sites, keys))
    counts, missing, rep_counts = _count_panels(panels, indices, truth, represented)
    for component, (sites, keys) in enumerate(metadata):
        metrics, component_matches = _compare_counts(
            counts[component], missing[component], rep_counts[component], len(sites))
        rows.append(dict(contig=contig, stage=stage, component=component,
                         first_position=int(sites[0]), last_position=int(sites[-1]), **metrics))
        matches.extend(dict(contig=contig, stage=stage, component=component,
                            founder_key=str(keys[match["predicted_row"]]), **match)
                       for match in component_matches)
    sum_keys = ("called_alleles", "missing_alleles", "called_allele_errors",
                "one_to_one_called_allele_errors", "matched_called_alleles",
                "unmatched_founder_rows", "unmatched_truth_founders", "unmatched_called_alleles",
                "truth_to_panel_errors_missing", "represented_matched_called_alleles",
                "represented_matched_called_errors", "absent_matched_called_alleles",
                "absent_matched_called_errors", "represented_truth_to_panel_errors_missing",
                "absent_truth_to_panel_errors_missing")
    summary = dict(contig=contig, stage=stage, components=len(rows),
                   markers=len(positions), reconstructed_markers=int(covered.sum()),
                   uncovered_markers=int((~covered).sum()),
                   min_founder_rows=min((r["founder_rows"] for r in rows), default=0),
                   max_founder_rows=max((r["founder_rows"] for r in rows), default=0),
                   truth_founders=len(truth), truth_alleles=int(truth.size),
                   represented_truth_alleles=int(represented.sum()),
                   absent_truth_alleles=int((~represented).sum()),
                   **{key: sum(row[key] for row in rows) for key in sum_keys})
    summary["truth_to_panel_errors_missing"] += len(truth)*int((~covered).sum())
    summary["represented_truth_to_panel_errors_missing"] += int(represented[:, ~covered].sum())
    summary["absent_truth_to_panel_errors_missing"] += int((~represented[:, ~covered]).sum())
    summary["called_errors_per_million"] = (
        1e6*summary["called_allele_errors"]/summary["called_alleles"]
        if summary["called_alleles"] else None)
    summary["one_to_one_errors_per_million"] = (
        1e6*summary["one_to_one_called_allele_errors"]/summary["matched_called_alleles"]
        if summary["matched_called_alleles"] else None)
    return summary, rows, matches
