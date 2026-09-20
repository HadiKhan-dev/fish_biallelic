"""Truth-only founder evaluation with fixed component-wide label matching.

Nearest-founder error counts assess precision of all called reconstructed rows.
One-to-one matching additionally exposes duplicated/missing founders. Its objective
counts mismatches AND unknown alleles; a missing row costs the entire component.
Neither metric rematches founder labels at individual markers.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core.products import panel_alleles


def represented_ancestry(truth_painting, positions, founder_count):
    """Presence in any sampled homologue, not independent-lineage identifiability."""
    positions = np.asarray(positions)
    delta = np.zeros((founder_count, len(positions) + 1), dtype=np.int32)
    for sample in truth_painting.samples:
        for chunk in sample.chunks:
            left, right = np.searchsorted(positions, (chunk.start, chunk.end))
            # Simulation terminal ancestry coordinates use the final marker as
            # their endpoint. Internal crossover boundaries remain half-open.
            if chunk.end == positions[-1]:
                right = len(positions)
            for founder in set((chunk.hap1, chunk.hap2)):
                if 0 <= founder < founder_count:
                    delta[founder, left] += 1
                    delta[founder, right] -= 1
    return np.cumsum(delta[:, :-1], axis=1) > 0


def compare_panel(calls, truth, represented=None):
    """Compare a K-by-L hard-call panel to F complete truth haplotypes."""
    calls, truth = np.asarray(calls), np.asarray(truth)
    if calls.ndim != 2 or truth.ndim != 2 or calls.shape[1] != truth.shape[1]:
        raise ValueError("predicted and true founder site axes differ")
    if np.any(~np.isin(calls, (-1, 0, 1))) or np.any(~np.isin(truth, (0, 1))):
        raise ValueError("require -1/0/1 calls and complete 0/1 founder truth")
    k, sites = calls.shape
    f = len(truth)
    if f == 0:
        raise ValueError("founder truth must not be empty")
    called = calls >= 0
    missing = np.count_nonzero(~called, axis=1)
    errors = np.empty((k, f), dtype=np.int64)
    for row in range(k):
        errors[row] = np.count_nonzero(
            called[row][None, :] & (calls[row][None, :] != truth), axis=1)
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
    represented = (np.ones(truth.shape, dtype=bool) if represented is None
                   else np.asarray(represented, dtype=bool))
    if represented.shape != truth.shape:
        raise ValueError("ancestry representation mask has the wrong axes")
    result["represented_truth_alleles"] = int(represented.sum())
    result["absent_truth_alleles"] = int((~represented).sum())
    for label, mask in (("represented", represented), ("absent", ~represented)):
        error_count = called_count = missing_count = 0
        for row, founder in zip(predicted, actual):
            active = mask[founder]
            called_count += int(np.count_nonzero(active & called[row]))
            error_count += int(np.count_nonzero(active & called[row] & (calls[row] != truth[founder])))
            missing_count += int(np.count_nonzero(active & ~called[row]))
        for founder in set(range(f)) - set(map(int, actual)):
            missing_count += int(mask[founder].sum())
        result[f"{label}_matched_called_alleles"] = called_count
        result[f"{label}_matched_called_errors"] = error_count
        result[f"{label}_truth_to_panel_errors_missing"] = error_count + missing_count
    matches = [dict(predicted_row=int(row), truth_founder=int(founder),
                    called_alleles=int(called[row].sum()),
                    errors=int(errors[row, founder]), missing=int(missing[row]))
               for row, founder in zip(predicted, actual)]
    return result, matches


def evaluate_founders(blocks, positions, truth, represented, *, contig, stage):
    """Count every truth marker, including markers outside reconstructed spans."""
    positions, truth = np.asarray(positions), np.asarray(truth)
    covered = np.zeros(len(positions), dtype=bool)
    rows, matches = [], []
    for component, block in enumerate(blocks):
        sites, keys, calls = panel_alleles(block)
        index = np.searchsorted(positions, sites)
        if np.any(index >= len(positions)) or not np.array_equal(positions[index], sites):
            raise ValueError(f"{contig}: founder truth and reconstruction positions differ")
        if np.any(covered[index]):
            raise ValueError("evaluated founder components must not overlap")
        covered[index] = True
        metrics, component_matches = compare_panel(calls, truth[:, index], represented[:, index])
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
