"""Reference-order junctions in the recombination map.

The pipeline assumes that reference order is meiotic order. Where it is not (misplaced or misassembled contigs,
inversions), about half of all meioses change transmitted homologue at the junction, and the map reports these
real transmission changes as crossover pile-ups. Without a breakpoint table no call, exposure or map value
changes; this module adds:

* Supplied order breakpoints (family_phase.breakpoints; 1-based inclusive bp). The discontinuity lies somewhere
  inside each interval. Painting and phase are rigid inside a painting bin, so the sites of every painting bin
  overlapping the interval leave the map axis and the component is split there: every reset rule keys on
  component changes, so no shared-orientation tract, exposure or call crosses the interval. A reversed segment's
  two ends are breakpoints and calls inside it are flagged. Positions stay in reference coordinates.
* Bin-honest call intervals: a call whose flanking supported markers straddle a painting-bin edge is reported
  with the two bins that share the edge (a change inside either bin is placed at that edge). Calls, counts and
  the map are unchanged.
* A truth-free junction detector, reported as a table; it never masks. Calls are grouped into loci by
  overlapping reported intervals. For each meiosis decidable across a locus, y = parity of its calls there.
  y = s XOR f: s ~ Bernoulli(q) is a real change of transmitted homologue across the locus, f (prior 1/2, one per
  parent) a frame flip of the parent shared by all its children. Ordinary recombination, hotspots included,
  bounds q by a ceiling h (a point hotspot plus the background map rate across the locus); a junction does not.
  Mixing statistic: log sup_q L(q) - log sup_{q<=h} L(q) (all-or-none patterns are explained by f). The two
  ends of a displaced or inverted segment switch in the same meioses: the pairing statistic compares y at a
  second locus predicted from the first (up to a discordance probability e) with independent switching there,
  takes the smaller of the two directions and subtracts log(number of locus pairs tested on the chromosome).
  Likelihood ratios are conditional on the calls, not calibrated posterior odds.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numba import njit

from ..model.genetic_map import haldane_fraction


# Profile grids. The flip symmetry makes q and 1 - q equivalent, so q <= 1/2. The discordance floor (0.5%) keeps
# one discordant meiosis (a call error or a crossover inside a displaced segment) from erasing pairing evidence.
SWITCH_GRID = np.linspace(.001, .5, 500)
DISCORDANCE_GRID = np.linspace(.005, .25, 50)


@dataclass(frozen=True)
class JunctionLocus:
    """One row of the junction table (a locus of overlapping reported call intervals)."""

    contig: str
    locus: str
    left_bp: int
    right_bp: int
    calls_left_bp: int
    calls_right_bp: int
    calls: int
    parents_with_calls: int
    odd_meioses: int
    decidable_meioses: int
    switch_fraction: float
    flip_adjusted_switch_fraction: float
    hotspot_ceiling: float
    mixed_parents: int
    all_or_none_parents: int
    parents: str
    children: str
    mixing_log_lr: float
    paired_locus: str
    paired_odd_meioses: int
    pairing_log_lr: float
    pairing_tests: int
    junction_log_lr: float
    flagged: bool
    interpretation: str
    nearest_supplied_breakpoint: str
    breakpoint_distance_bp: float


def breakpoint_intervals(breakpoints):
    """Supplied breakpoint intervals plus both ends of every reversed segment."""
    order, reversed_segments, _ = breakpoints
    intervals = [(int(start), int(end)) for start, end in order]
    for start, end in reversed_segments:
        intervals += [(int(start), int(start)), (int(end), int(end))]
    return intervals


def split_at_breakpoints(positions, components, breakpoints, bin_edges):
    """(segment ids int32 [L], segment -> component int32, masked sites) of the axis split at breakpoints.

    Sites inside a breakpoint interval, or inside a painting bin overlapping it, get segment -1; a new segment
    starts after each excluded zone. Segment ids are consecutive in axis order.
    """
    intervals = breakpoint_intervals(breakpoints)
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
    return segments, np.asarray(origin, dtype=np.int32), int(excluded.sum())


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


def reported_intervals(crossovers, positions, segments, bin_edges, config):
    """int64 (n, 2) reported intervals in bp (inclusive) of called crossovers (n, 5) carrying assembly components.

    Where the decoder's flanking markers straddle painting-bin edges, the interval also covers the bins on both
    sides of those edges, clipped to the call's segment of the decoded axis ``segments``.
    """
    limits = _segment_limits(positions, segments)
    intervals = np.empty((len(crossovers), 2), dtype=np.int64)
    for index, (left, right, _, _, component) in enumerate(crossovers):
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
    return intervals


def in_reversed_segment(crossovers, breakpoints):
    """bool (n,): whether each called crossover lies inside a supplied reversed segment."""
    segments = np.zeros((0, 2), dtype=np.int64) if breakpoints is None else breakpoints[1]
    return np.asarray([bool(np.any((segments[:, 0] <= left) & (right <= segments[:, 1])))
                       for left, right, *_ in crossovers], dtype=bool)


def mixture_terms(counts, grid):
    """(grid, rows) terms log(1/2 p^k (1-p)^(n-k) + 1/2 p^(n-k) (1-p)^k) for (k, n) rows."""
    counts = np.asarray(counts, dtype=np.float64).reshape(-1, 2)
    p = np.asarray(grid, dtype=np.float64)[:, None]
    k, n = counts[:, 0][None, :], counts[:, 1][None, :]
    same = k * np.log(p) + (n - k) * np.log1p(-p)
    flipped = (n - k) * np.log(p) + k * np.log1p(-p)
    return np.logaddexp(same, flipped) - math.log(2.)


# The pairing statistic of every locus pair: each parent decidable at both loci contributes the term of its
# (k, n) count to three flip mixtures. A (k, n) count is coded n (n + 1) / 2 + k and its terms are tabulated once.
@njit(cache=True)
def _pair_codes(decidable, odd, parent_offsets, parent_edges, first, second, here, there, discordance):
    """Count codes per parent with edges decidable at both loci; returns the number of parents."""
    rows = 0
    for parent in range(len(parent_offsets) - 1):
        n = k_here = k_there = k_discordant = 0
        for slot in range(parent_offsets[parent], parent_offsets[parent + 1]):
            edge = parent_edges[slot]
            if decidable[first, edge] and decidable[second, edge]:
                n += 1
                k_here += odd[first, edge]
                k_there += odd[second, edge]
                if odd[first, edge] != odd[second, edge]:
                    k_discordant += 1
        if n:
            offset = n * (n + 1) // 2
            here[rows] = offset + k_here
            there[rows] = offset + k_there
            discordance[rows] = offset + k_discordant
            rows += 1
    return rows


@njit(cache=True)
def _used_pair_codes(decidable, odd, parent_offsets, parent_edges, codes):
    """Mark the switch (here/there) and discordance count codes that occur in any locus pair."""
    loci, parents = decidable.shape[0], len(parent_offsets) - 1
    switch, discordant = np.zeros(codes, dtype=np.bool_), np.zeros(codes, dtype=np.bool_)
    here, there = np.empty(parents, dtype=np.int64), np.empty(parents, dtype=np.int64)
    discordance = np.empty(parents, dtype=np.int64)
    for first in range(loci):
        for second in range(first + 1, loci):
            rows = _pair_codes(decidable, odd, parent_offsets, parent_edges, first, second,
                               here, there, discordance)
            for row in range(rows):
                switch[here[row]] = True
                switch[there[row]] = True
                discordant[discordance[row]] = True
    return switch, discordant


@njit(cache=True)
def _mixture_maximum(table, rows, codes, count, sums):
    """Maximum over the grid of the parents' summed mixture terms (summed in parent order)."""
    sums[:] = 0.
    for index in range(count):
        row = rows[codes[index]]
        for g in range(len(sums)):
            sums[g] += table[row, g]
    return np.max(sums)


@njit(cache=True)
def _pairing_scores(decidable, odd, parent_offsets, parent_edges, switch_table, switch_rows,
                    discordance_table, discordance_rows):
    """Symmetric pairing log-LR of every locus pair; -inf where no parent is decidable at both loci."""
    loci, parents = decidable.shape[0], len(parent_offsets) - 1
    scores = np.full((loci, loci), -np.inf)
    here, there = np.empty(parents, dtype=np.int64), np.empty(parents, dtype=np.int64)
    discordance = np.empty(parents, dtype=np.int64)
    switch_sums = np.empty(switch_table.shape[1])
    discordance_sums = np.empty(discordance_table.shape[1])
    for first in range(loci):
        for second in range(first + 1, loci):
            rows = _pair_codes(decidable, odd, parent_offsets, parent_edges, first, second, here, there, discordance)
            if rows:
                paired = _mixture_maximum(discordance_table, discordance_rows, discordance, rows, discordance_sums)
                independent = max(_mixture_maximum(switch_table, switch_rows, here, rows, switch_sums),
                                  _mixture_maximum(switch_table, switch_rows, there, rows, switch_sums))
                scores[first, second] = scores[second, first] = paired - independent
    return scores


def _pairing_log_lr(loci, parents):
    """Pairing statistic of every locus pair."""
    decidable = np.ascontiguousarray([locus["decidable"] for locus in loci], dtype=np.bool_)
    odd = np.ascontiguousarray([locus["odd"] for locus in loci], dtype=np.int64)
    order = np.argsort(parents, kind="stable")
    parent_offsets = np.r_[0, np.cumsum(np.unique(parents, return_counts=True)[1])].astype(np.int64)
    largest = int(np.diff(parent_offsets).max())
    count_n = np.repeat(np.arange(largest + 1), np.arange(1, largest + 2))
    count_k = np.arange(len(count_n)) - count_n * (count_n + 1) // 2
    tables = []
    for used, grid in zip(_used_pair_codes(decidable, odd, parent_offsets, order, len(count_n)),
                          (SWITCH_GRID, DISCORDANCE_GRID)):
        rows = np.full(len(count_n), -1, dtype=np.int64)
        rows[used] = np.arange(int(used.sum()))
        terms = mixture_terms(np.stack((count_k[used], count_n[used]), axis=1), grid)
        tables += [np.ascontiguousarray(terms.T), rows]
    return _pairing_scores(decidable, odd, parent_offsets, order, *tables)


def _hotspot_ceiling(left, right, config, chromosome_map):
    """Point hotspot XOR background parity across the locus (Haldane)."""
    morgans = float(np.diff(chromosome_map.cumulative_morgans(np.asarray([left, right], dtype=float)))[0]) \
        if chromosome_map.has_map else chromosome_map.fallback_rate_per_bp * (right - left)
    background = haldane_fraction(max(morgans, 0.))
    point = config.hotspot_switch_probability
    return point + background - 2. * point * background


def detect_junctions(contig, names, parents, children, crossovers, crossover_offsets, reported, callable_spans,
                     callable_offsets, config, chromosome_map, breakpoints=None):
    """Junction loci of one chromosome (module docstring); per-edge tables are packed with offsets."""
    edges = len(parents)
    members_of = sorted((int(reported[index, 0]), int(reported[index, 1]), edge, index)
                        for edge in range(edges) for index in range(crossover_offsets[edge], crossover_offsets[edge + 1]))
    groups = []
    for lo, hi, edge, index in members_of:
        if groups and lo <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], hi)
            groups[-1][2].append((edge, index))
        else:
            groups.append([lo, hi, [(edge, index)]])
    loci = []
    for left, right, members in groups:
        if len({int(parents[edge]) for edge, _ in members}) < config.minimum_parents:
            continue
        decoded = np.asarray([crossovers[index, :2] for _, index in members])
        first, last = float(decoded[:, 0].min()), float(decoded[:, 1].max())
        odd = np.zeros(edges, dtype=np.int64)
        for edge, _ in members:
            odd[edge] ^= 1
        with_calls = {edge for edge, _ in members}
        decidable = np.asarray([edge in with_calls or any(
            span_left <= first and span_right >= last
            for span_left, span_right, _ in callable_spans[callable_offsets[edge]:callable_offsets[edge + 1]])
            for edge in range(edges)])
        loci.append(dict(left=left, right=right, first=first, last=last, members=members,
                         odd=odd, decidable=decidable))
    rows = []
    for number, locus in enumerate(loci):
        counts, parent_text, children_text = [], [], []
        mixed = all_or_none = 0
        for parent in np.unique(parents):
            parent_edges = np.flatnonzero((parents == parent) & locus["decidable"])
            if not len(parent_edges):
                continue
            k, n = int(locus["odd"][parent_edges].sum()), len(parent_edges)
            counts.append((k, n))
            if k:
                parent_text.append(f"{names[parent]}:{k}/{n}")
                children_text.extend(names[child] for child in children[parent_edges[locus["odd"][parent_edges] == 1]])
                mixed += k < n
                all_or_none += k == n and n >= 2
        counts = np.asarray(counts, dtype=np.int64).reshape(-1, 2)
        ceiling = _hotspot_ceiling(locus["left"], locus["right"], config, chromosome_map)
        mixture = mixture_terms(counts, SWITCH_GRID).sum(axis=1)
        bounded = mixture_terms(counts, np.r_[SWITCH_GRID[SWITCH_GRID < ceiling],
                                              max(ceiling, SWITCH_GRID[0])]).sum(axis=1)
        nearest, distance = "", np.nan
        if breakpoints is not None:
            for start, end in breakpoint_intervals(breakpoints):
                gap = max(int(start) - locus["right"], locus["left"] - int(end), 0)
                if not gap >= distance:
                    nearest, distance = f"{start}-{end}", gap
        rows.append(dict(
            contig=contig, locus=f"{contig}:{number}", left_bp=int(locus["left"]), right_bp=int(locus["right"]),
            calls_left_bp=int(locus["first"]), calls_right_bp=int(locus["last"]), calls=len(locus["members"]),
            parents_with_calls=len(parent_text), odd_meioses=int(counts[:, 0].sum()),
            decidable_meioses=int(counts[:, 1].sum()),
            switch_fraction=float(counts[:, 0].sum() / max(counts[:, 1].sum(), 1)),
            flip_adjusted_switch_fraction=float(SWITCH_GRID[np.argmax(mixture)]),
            hotspot_ceiling=ceiling, mixed_parents=mixed, all_or_none_parents=all_or_none,
            parents=";".join(parent_text), children=";".join(children_text),
            mixing_log_lr=max(float(mixture.max() - bounded.max()), 0.),
            paired_locus="", paired_odd_meioses=0, pairing_log_lr=0., pairing_tests=0,
            nearest_supplied_breakpoint=nearest, breakpoint_distance_bp=float(distance)))
    tests = len(loci) * (len(loci) - 1) // 2
    scores = _pairing_log_lr(loci, parents) if len(loci) > 1 else None
    for i, locus in enumerate(loci):
        # The best pairing is the first locus with the largest positive score.
        j = -1 if scores is None else int(np.argmax(scores[i]))
        if j >= 0 and scores[i, j] > 0.:
            other = loci[j]
            both = locus["decidable"] & other["decidable"]
            rows[i].update(paired_locus=rows[j]["locus"],
                           paired_odd_meioses=int(((locus["odd"] == 1) & (other["odd"] == 1) & both).sum()),
                           pairing_log_lr=float(scores[i, j]))
        rows[i]["pairing_tests"] = tests
    result = []
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
        result.append(JunctionLocus(**row))
    return tuple(result)
