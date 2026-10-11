"""Reference-order junctions in the recombination map.

The pipeline assumes that reference order is meiotic order. Where it is not
(misplaced or misassembled contigs, inversions), about half of all meioses
change transmitted homologue at the junction, and the map reports these real
transmission changes as crossover pile-ups. This module adds three things to
the recombination stage; without a breakpoint file no call, exposure or map
value changes.

* Supplied order breakpoints (optional TSV ``chrom start end reason``, 1-based
  inclusive bp; ``#`` comments). The reference-order discontinuity lies
  somewhere inside each interval. Painting and phase are rigid inside a ragged
  painting bin, so a change at the discontinuity is placed at an edge of the
  bin that contains it: the sites of every painting bin overlapping the
  interval leave the map axis, and the assembly component is split there.
  Every reset rule of the map keys on component changes, so no
  shared-orientation tract, exposure or call crosses the interval and the
  decoder restarts beyond it. A row whose reason starts with
  ``reversed_segment`` declares an inverted reference segment: its two ends are
  breakpoints, and calls inside it are flagged. Positions stay in reference
  coordinates throughout.
* Bin-honest call intervals. A call whose flanking supported markers straddle
  a painting-bin edge is reported with the two bins that share the edge (a
  change inside either bin is placed at that edge). Calls, counts and the map
  are unchanged; only the reported interval widens.
* A truth-free junction detector, reported as a table; it never masks.
  Calls are grouped into loci by overlapping reported intervals. For each
  meiosis that is decidable across a locus, y = parity of its calls there.
  The model is y = s XOR f: s ~ Bernoulli(q) is a real change of transmitted
  homologue across the locus, f (prior 1/2, one per parent) is a frame flip of
  the parent shared by all its children (crossed panel rows, scaffold or
  decoder frame errors). Ordinary recombination, hotspots included, bounds q
  by a ceiling h (a point hotspot plus the background map rate across the
  locus); a junction does not (q = 1/2 for unlinked sides, 1/2(1 - e^-2d) at
  the ends of an inverted segment of genetic length d). The mixing statistic is
  log sup_q L(q) - log sup_{q<=h} L(q): all-or-none patterns are explained by
  f and give no evidence, nor do hotspot-like single calls below h. The two
  ends of a displaced or inverted segment switch in the same meioses: the
  pairing statistic compares y at a second locus predicted from the first
  (up to a discordance probability e) with independent switching there, takes
  the smaller of the two directions and subtracts log(number of locus pairs
  tested on the chromosome). Loci are assumed independent across meioses
  given the parents' flips; likelihood ratios are conditional on the calls,
  not calibrated posterior odds.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

# Profile grids. The flip symmetry makes q and 1 - q equivalent, so q <= 1/2.
# The discordance floor (0.5%) keeps one discordant meiosis (a call error or a
# crossover inside a displaced segment) from erasing pairing evidence.
_SWITCH_GRID = np.linspace(.001, .5, 500)
_DISCORDANCE_GRID = np.linspace(.005, .25, 50)
REVERSED_SEGMENT = "reversed_segment"
LOCUS_COLUMNS = (
    "contig", "locus", "left_bp", "right_bp", "calls_left_bp", "calls_right_bp", "calls",
    "parents_with_calls", "odd_meioses", "decidable_meioses", "switch_fraction",
    "flip_adjusted_switch_fraction", "hotspot_ceiling", "mixed_parents", "all_or_none_parents",
    "parents", "children", "mixing_log_lr", "paired_locus", "paired_odd_meioses",
    "pairing_log_lr", "pairing_tests", "junction_log_lr", "flagged", "interpretation",
    "nearest_supplied_breakpoint", "breakpoint_distance_bp")


@dataclass(frozen=True)
class JunctionReportConfig:
    """Interval reporting and junction-detector controls (bp, probabilities, nats)."""
    # Report the two bins around a straddled bin edge when they span at least
    # this many bp; 0 widens every straddled call.
    snapped_bin_minimum_bp: int = 0
    minimum_parents: int = 3
    # Point-hotspot part of the ceiling: no single locus gives a real
    # crossover in more than 5% of meioses (5 cM) beyond the background rate.
    hotspot_switch_probability: float = .05
    # Flag at 100:1. A whole genome has about 30 candidate loci; a likelihood
    # ratio reaches c under its null with probability about 1/c or less, so
    # fewer than one false flag per genome is expected.
    flag_log_likelihood_ratio: float = math.log(100.)

    def validated(self):
        if self.snapped_bin_minimum_bp < 0 or self.minimum_parents < 2:
            raise ValueError("invalid junction interval/parent thresholds")
        if not 0 <= self.hotspot_switch_probability < .5 or not self.flag_log_likelihood_ratio > 0:
            raise ValueError("invalid junction hotspot ceiling or flag threshold")
        return self


def read_order_breakpoints(path):
    """{contig: (breakpoints (k, 2), reversed segments (j, 2), reasons)}; 1-based inclusive bp."""
    table = pd.read_csv(path, sep="\t", comment="#", dtype={"chrom": str, "reason": str})
    if list(table.columns[:4]) != ["chrom", "start", "end", "reason"]:
        raise ValueError("order breakpoint file needs the columns chrom, start, end, reason")
    starts = table["start"].to_numpy(dtype=np.int64)
    ends = table["end"].to_numpy(dtype=np.int64)
    if np.any(starts < 1) or np.any(ends < starts):
        raise ValueError("order breakpoint intervals need 1 <= start <= end")
    reasons = table["reason"].fillna("").astype(str).str.strip()
    result = {}
    for contig in dict.fromkeys(table["chrom"].astype(str)):
        rows = np.flatnonzero(table["chrom"].astype(str).to_numpy() == contig)
        reversed_rows = rows[reasons.iloc[rows].str.startswith(REVERSED_SEGMENT).to_numpy()]
        order_rows = np.setdiff1d(rows, reversed_rows)
        result[contig] = (np.stack((starts[order_rows], ends[order_rows]), axis=1),
                          np.stack((starts[reversed_rows], ends[reversed_rows]), axis=1),
                          tuple(reasons.iloc[rows]))
    return result


def component_bin_edges(painting_checkpoint):
    """Ragged painting/phase bin edges per component, as family phase uses them.

    Same rule as ``refinement.conditioning.prepare_phase_scaffold``: the ragged
    diagnostics' half-open bp edges, or 100-site bins for complete or empty
    components.
    """
    blocks = core_runtime.validate_phase_component_manifest(painting_checkpoint.component_manifest)
    edges = []
    for block, component in zip(blocks, painting_checkpoint.painting_bundle.components):
        diagnostic = component.ragged_diagnostics
        if diagnostic is not None:
            edges.append(np.asarray(diagnostic.bin_edges, dtype=np.int64))
        else:
            source = np.asarray(block.positions, dtype=np.int64)
            edges.append(np.unique(np.r_[source[::100], source[-1] + 1]))
    return tuple(edges)


def _breakpoint_intervals(breakpoints):
    """Supplied breakpoint intervals plus both ends of every inverted segment."""
    order, reversed_segments, _ = breakpoints
    intervals = [(int(start), int(end)) for start, end in order]
    for start, end in reversed_segments:
        intervals += [(int(start), int(start)), (int(end), int(end))]
    return intervals


def apply_order_breakpoints(product, breakpoints, bin_edges):
    """Return a final-phase view split at supplied breakpoints and its segment origins.

    Sites inside a breakpoint interval, or inside a painting bin overlapping it,
    get component -1; a new segment starts after each excluded zone. Segment ids
    are consecutive in axis order; ``origin[segment]`` is the assembly component.
    """
    positions = np.asarray(product["positions"], dtype=np.int64)
    components = np.asarray(product["component_ids"], dtype=np.int32)
    intervals = _breakpoint_intervals(breakpoints)
    excluded = np.zeros(len(positions), dtype=bool)
    cuts = []
    for start, end in intervals:
        lo, hi = int(start), int(end) + 1
        for edges in bin_edges:
            if edges[0] < hi and edges[-1] > lo:
                first = max(np.searchsorted(edges, lo, side="right") - 1, 0)
                last = min(np.searchsorted(edges, hi, side="left"), len(edges) - 1)
                lo, hi = min(lo, int(edges[first])), max(hi, int(edges[last]))
        excluded |= (positions >= lo) & (positions < hi)
        cuts.append(hi)
    counter = np.searchsorted(np.sort(np.asarray(cuts, dtype=np.int64)), positions, side="right")
    valid = (components >= 0) & ~excluded
    key = components.astype(np.int64) * (len(cuts) + 1) + counter
    segments = np.full(len(positions), -1, dtype=np.int32)
    origin = []
    previous = None
    for index in np.flatnonzero(valid):
        if key[index] != previous:
            previous = key[index]
            origin.append(int(components[index]))
        segments[index] = len(origin) - 1
    view = dict(product)
    view["component_ids"] = segments
    return view, np.asarray(origin, dtype=np.int32), int(excluded.sum())


def restore_component_ids(result, origin):
    """Report assembly components again on calls and spans decoded per segment."""
    for key, column in (("crossover_intervals", 4), ("callable_spans", 2), ("informative_spans", 2)):
        for rows in result[key]:
            if len(rows):
                rows[:, column] = origin[rows[:, column].astype(np.int64)]


def _segment_limits(positions, segments):
    """First and last site position of each segment."""
    valid = np.flatnonzero(segments >= 0)
    labels = segments[valid]
    starts = np.r_[0, np.flatnonzero(np.diff(labels)) + 1]
    stops = np.r_[starts[1:], len(labels)] - 1
    limits = np.zeros((labels.max() + 1 if len(labels) else 0, 2), dtype=np.int64)
    limits[labels[starts], 0] = positions[valid[starts]]
    limits[labels[starts], 1] = positions[valid[stops]]
    return limits


def reported_intervals(result, positions, segments, bin_edges, config):
    """Per edge, (n, 2) int64 reported call intervals in bp (inclusive).

    Where the decoder's flanking markers straddle painting-bin edges, the
    interval also covers the bins on both sides of those edges, clipped to the
    call's segment of the decoded axis ``segments`` (the assembly components
    when no breakpoints were supplied). Call rows carry assembly components.
    """
    positions = np.asarray(positions, dtype=np.int64)
    limits = _segment_limits(positions, np.asarray(segments))
    output = []
    for rows in result["crossover_intervals"]:
        intervals = np.empty((len(rows), 2), dtype=np.int64)
        for index, (left, right, _, _, component) in enumerate(rows):
            lo, hi = int(left), int(right)
            edges = bin_edges[int(component)]
            inside = np.flatnonzero((edges > left) & (edges <= right))
            if len(inside):
                first = int(edges[max(inside[0] - 1, 0)])
                last = int(edges[min(inside[-1] + 1, len(edges) - 1)]) - 1
                if last - first + 1 >= config.snapped_bin_minimum_bp:
                    segment = segments[np.searchsorted(positions, left)]
                    lo = max(min(lo, first), int(limits[segment, 0]))
                    hi = min(max(hi, last), int(limits[segment, 1]))
            intervals[index] = lo, hi
        output.append(intervals)
    return output


def reversed_segment_calls(result, breakpoints):
    """Per edge, whether each call lies inside a supplied reversed segment."""
    reversed_segments = np.zeros((0, 2), dtype=np.int64) if breakpoints is None else breakpoints[1]
    return [np.asarray([bool(np.any((reversed_segments[:, 0] <= left) & (right <= reversed_segments[:, 1])))
                        for left, right, *_ in rows], dtype=bool)
            for rows in result["crossover_intervals"]]


def _flip_mixture(counts, grid):
    """Grid of sum_p log(1/2 p^k (1-p)^(n-k) + 1/2 p^(n-k) (1-p)^k) for (k, n) rows."""
    counts = np.asarray(counts, dtype=np.float64).reshape(-1, 2)
    p = np.asarray(grid, dtype=np.float64)[:, None]
    k, n = counts[:, 0][None, :], counts[:, 1][None, :]
    same = k * np.log(p) + (n - k) * np.log1p(-p)
    flipped = (n - k) * np.log(p) + k * np.log1p(-p)
    return (np.logaddexp(same, flipped) - math.log(2.)).sum(axis=1)


def _hotspot_ceiling(left, right, config, rate, chromosome_map):
    """Point hotspot XOR background parity across the locus (Haldane)."""
    if chromosome_map is not None and chromosome_map.has_map:
        morgans = float(np.diff(chromosome_map.cumulative_morgans(np.asarray([left, right], dtype=float)))[0])
    else:
        morgans = rate * (right - left)
    background = -.5 * math.expm1(-2. * max(morgans, 0.))
    point = config.hotspot_switch_probability
    return point + background - 2. * point * background


def detect_junctions(result, reported, *, config, rate, chromosome_map=None,
                     breakpoints=None):
    """Junction loci of one chromosome as a list of table rows (see module docstring)."""
    names = result["sample_ids"]
    parents = np.asarray(result["edge_parent"], dtype=np.int64)
    children = np.asarray(result["edge_child"], dtype=np.int64)
    calls = sorted((int(lo), int(hi), edge, index)
                   for edge, rows in enumerate(reported) for index, (lo, hi) in enumerate(rows))
    groups = []
    for lo, hi, edge, index in calls:
        if groups and lo <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], hi)
            groups[-1][2].append((edge, index))
        else:
            groups.append([lo, hi, [(edge, index)]])
    loci = []
    for left, right, members in groups:
        if len({int(parents[edge]) for edge, _ in members}) < config.minimum_parents:
            continue
        decoded = np.asarray([result["crossover_intervals"][edge][index, :2] for edge, index in members])
        first, last = float(decoded[:, 0].min()), float(decoded[:, 1].max())
        odd = np.zeros(len(parents), dtype=np.int64)
        for edge, _ in members:
            odd[edge] ^= 1
        with_calls = {edge for edge, _ in members}
        decidable = np.asarray([edge in with_calls or any(
            span_left <= first and span_right >= last
            for span_left, span_right, _ in result["callable_spans"][edge]) for edge in range(len(parents))])
        loci.append(dict(left=left, right=right, first=first, last=last, members=members,
                         odd=odd, decidable=decidable))
    rows = []
    for number, locus in enumerate(loci):
        counts, parent_text, children_text = [], [], []
        mixed = all_or_none = 0
        for parent in np.unique(parents):
            edges = np.flatnonzero((parents == parent) & locus["decidable"])
            if not len(edges):
                continue
            k, n = int(locus["odd"][edges].sum()), len(edges)
            counts.append((k, n))
            if k:
                parent_text.append(f"{names[parent]}:{k}/{n}")
                children_text.extend(names[child] for child in children[edges[locus["odd"][edges] == 1]])
                mixed += k < n
                all_or_none += k == n and n >= 2
        counts = np.asarray(counts, dtype=np.int64).reshape(-1, 2)
        ceiling = _hotspot_ceiling(locus["left"], locus["right"], config, rate, chromosome_map)
        mixture = _flip_mixture(counts, _SWITCH_GRID)
        bounded = _flip_mixture(counts, np.r_[_SWITCH_GRID[_SWITCH_GRID < ceiling],
                                              max(ceiling, _SWITCH_GRID[0])])
        nearest, distance = "", np.nan
        if breakpoints is not None:
            for start, end in _breakpoint_intervals(breakpoints):
                gap = max(int(start) - locus["right"], locus["left"] - int(end), 0)
                if not gap >= distance:
                    nearest, distance = f"{start}-{end}", gap
        rows.append({
            "contig": result["contig"], "locus": f"{result['contig']}:{number}",
            "left_bp": int(locus["left"]), "right_bp": int(locus["right"]),
            "calls_left_bp": int(locus["first"]), "calls_right_bp": int(locus["last"]),
            "calls": len(locus["members"]), "parents_with_calls": len(parent_text),
            "odd_meioses": int(counts[:, 0].sum()), "decidable_meioses": int(counts[:, 1].sum()),
            "switch_fraction": float(counts[:, 0].sum() / max(counts[:, 1].sum(), 1)),
            "flip_adjusted_switch_fraction": float(_SWITCH_GRID[np.argmax(mixture)]),
            "hotspot_ceiling": ceiling, "mixed_parents": mixed, "all_or_none_parents": all_or_none,
            "parents": ";".join(parent_text), "children": ";".join(children_text),
            "mixing_log_lr": max(float(mixture.max() - bounded.max()), 0.),
            "paired_locus": "", "paired_odd_meioses": 0, "pairing_log_lr": 0., "pairing_tests": 0,
            "nearest_supplied_breakpoint": nearest, "breakpoint_distance_bp": distance})
    tests = len(loci) * (len(loci) - 1) // 2
    for i, locus in enumerate(loci):
        best = (0., None, 0)
        for j, other in enumerate(loci):
            if i == j:
                continue
            both = locus["decidable"] & other["decidable"]
            discordance, here, there = [], [], []
            for parent in np.unique(parents[both]):
                edges = np.flatnonzero((parents == parent) & both)
                a, b = locus["odd"][edges], other["odd"][edges]
                discordance.append((int((a != b).sum()), len(edges)))
                here.append((int(a.sum()), len(edges)))
                there.append((int(b.sum()), len(edges)))
            if not discordance:
                continue
            paired = _flip_mixture(discordance, _DISCORDANCE_GRID).max()
            score = paired - max(_flip_mixture(here, _SWITCH_GRID).max(),
                                 _flip_mixture(there, _SWITCH_GRID).max())
            if score > best[0]:
                shared = int(((locus["odd"] == 1) & (other["odd"] == 1) & both).sum())
                best = (float(score), j, shared)
        if best[1] is not None:
            rows[i].update(paired_locus=rows[best[1]]["locus"], paired_odd_meioses=best[2],
                           pairing_log_lr=best[0])
        rows[i]["pairing_tests"] = tests
    for row in rows:
        adjusted = row["pairing_log_lr"] - (math.log(row["pairing_tests"]) if row["pairing_tests"] else 0.)
        row["junction_log_lr"] = max(row["mixing_log_lr"], adjusted, 0.)
        row["flagged"] = bool(row["junction_log_lr"] >= config.flag_log_likelihood_ratio)
        if row["flagged"]:
            row["interpretation"] = "junction"
        elif row["all_or_none_parents"] and row["flip_adjusted_switch_fraction"] <= row["hotspot_ceiling"]:
            row["interpretation"] = "parent_frame_flips"
        else:
            row["interpretation"] = "recombination_or_unresolved"
    return [{column: row[column] for column in LOCUS_COLUMNS} for row in rows]


import haplotype_reconstruction.core.runtime as core_runtime
