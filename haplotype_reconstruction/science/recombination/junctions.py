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
  Likelihood ratios are conditional on the calls, not calibrated posterior odds. Under the explicit convention
  reversal (the ``reverse`` argument; core.ties) the report's exact ties (best paired locus, profile
  maximum on the switch grid, nearest supplied breakpoint) go to the last candidate instead of the first; the
  positive-score and flag thresholds are kept.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numba import types
from numpy.typing import NDArray

from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.recombination import JunctionLocus
from ...core.kernels import kernel
from ...core.ties import conv_argmax
from ..shared.recombination import haldane_fraction
from .parameters import DISCORDANCE_GRID_SPEC, SWITCH_GRID_SPEC, JunctionParameters

if TYPE_CHECKING:
    from numbers import Real
    from typing import TypedDict

    class _Locus(TypedDict):
        left: int
        right: int
        first: float
        last: float
        members: list[tuple[int, int]]
        odd: NDArray[np.int64]
        decidable: NDArray[np.bool_]


REVERSED_SEGMENT = "reversed_segment"

# Instantiate the fixed inclusive profile specifications from the parameter module.
SWITCH_GRID = np.linspace(*SWITCH_GRID_SPEC)
DISCORDANCE_GRID = np.linspace(*DISCORDANCE_GRID_SPEC)


_OVERLOADS = {
    "_mixture_maximum": (
        (
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_pair_codes": (
        (
            types.Array(types.bool, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_pairing_scores": (
        (
            types.Array(types.bool, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_used_pair_codes": (
        (
            types.Array(types.bool, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
        ),
    ),
}


def breakpoint_intervals(breakpoints):
    """Supplied breakpoint intervals plus both ends of every reversed segment."""
    order, reversed_segments, _ = breakpoints
    intervals = [(int(start), int(end)) for start, end in order]
    for start, end in reversed_segments:
        intervals += [(int(start), int(start)), (int(end), int(end))]
    return intervals


def split_at_breakpoints(
    positions: NDArray[np.int64],
    components: NDArray[np.int32],
    breakpoints: tuple[NDArray[np.int64], NDArray[np.int64], tuple[str, ...]],
    bin_edges: Sequence[NDArray[np.int64]],
) -> tuple[np.ndarray, np.ndarray, int]:
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
            overlap = edges[0] < hi and edges[-1] > lo
            if overlap:
                first = cast("int | np.integer", max(np.searchsorted(edges, lo, side="right") - 1, 0))
                last = cast("int | np.integer", min(np.searchsorted(edges, hi, side="left"), len(edges) - 1))
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
    result = segments, np.asarray(origin, dtype=np.int32), int(excluded.sum())
    return result


def in_reversed_segment(crossovers, breakpoints):
    """bool (n,): whether each called crossover lies inside a supplied reversed segment."""
    segments = np.zeros((0, 2), dtype=np.int64) if breakpoints is None else breakpoints[1]
    result = []
    for left, right, *_ in crossovers:
        inside = bool(np.any((segments[:, 0] <= left) & (right <= segments[:, 1])))
        result.append(inside)
    value = np.asarray(result, dtype=bool)
    return value


def mixture_terms(counts, grid):
    """(grid, rows) terms log(1/2 p^k (1-p)^(n-k) + 1/2 p^(n-k) (1-p)^k) for (k, n) rows."""
    counts = np.asarray(counts, dtype=np.float64).reshape(-1, 2)
    p = np.asarray(grid, dtype=np.float64)[:, None]
    k, n = counts[:, 0][None, :], counts[:, 1][None, :]
    same = k * np.log(p) + (n - k) * np.log1p(-p)
    flipped = (n - k) * np.log(p) + k * np.log1p(-p)
    return np.logaddexp(same, flipped) - math.log(2.0)


# The pairing statistic of every locus pair: each parent decidable at both loci contributes the term of its
# (k, n) count to three flip mixtures. A (k, n) count is coded n (n + 1) / 2 + k and its terms are tabulated once.
@kernel(overloads=_OVERLOADS["_pair_codes"], cache=True, nogil=True)
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


@kernel(overloads=_OVERLOADS["_used_pair_codes"], cache=True, nogil=True)
def _used_pair_codes(decidable, odd, parent_offsets, parent_edges, codes):
    """Mark the switch (here/there) and discordance count codes that occur in any locus pair."""
    loci, parents = decidable.shape[0], len(parent_offsets) - 1
    switch, discordant = np.zeros(codes, dtype=np.bool_), np.zeros(codes, dtype=np.bool_)
    here, there = np.empty(parents, dtype=np.int64), np.empty(parents, dtype=np.int64)
    discordance = np.empty(parents, dtype=np.int64)
    for first in range(loci):
        for second in range(first + 1, loci):
            rows = _pair_codes(decidable, odd, parent_offsets, parent_edges, first, second, here, there, discordance)
            for row in range(rows):
                switch[here[row]] = True
                switch[there[row]] = True
                discordant[discordance[row]] = True
    return switch, discordant


@kernel(overloads=_OVERLOADS["_mixture_maximum"], cache=True, nogil=True)
def _mixture_maximum(table, rows, codes, count, sums):
    """Maximum over the grid of the parents' summed mixture terms (summed in parent order)."""
    sums[:] = 0.0
    for index in range(count):
        row = rows[codes[index]]
        for g in range(len(sums)):
            sums[g] += table[row, g]
    return np.max(sums)


@kernel(overloads=_OVERLOADS["_pairing_scores"], cache=True, nogil=True)
def _pairing_scores(
    decidable, odd, parent_offsets, parent_edges, switch_table, switch_rows, discordance_table, discordance_rows
):
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
                independent = max(
                    _mixture_maximum(switch_table, switch_rows, here, rows, switch_sums),
                    _mixture_maximum(switch_table, switch_rows, there, rows, switch_sums),
                )
                scores[first, second] = scores[second, first] = paired - independent
    return scores


def _pairing_log_lr(loci, parents, *, parent_offsets=None, parent_edges=None):
    """Pairing statistic of every locus pair."""
    decidable = np.ascontiguousarray([locus["decidable"] for locus in loci], dtype=np.bool_)
    odd = np.ascontiguousarray([locus["odd"] for locus in loci], dtype=np.int64)
    if parent_offsets is None:
        parent_edges = np.argsort(parents, kind="stable")
        parent_offsets = np.r_[0, np.cumsum(np.unique(parents, return_counts=True)[1])].astype(np.int64)
    largest = int(np.diff(parent_offsets).max())
    count_n = np.repeat(np.arange(largest + 1), np.arange(1, largest + 2))
    count_k = np.arange(len(count_n)) - count_n * (count_n + 1) // 2
    tables = []
    for used, grid in zip(
        _used_pair_codes(decidable, odd, parent_offsets, parent_edges, len(count_n)),
        (SWITCH_GRID, DISCORDANCE_GRID),
        strict=False,
    ):
        rows = np.full(len(count_n), -1, dtype=np.int64)
        rows[used] = np.arange(int(used.sum()))
        terms = mixture_terms(np.stack((count_k[used], count_n[used]), axis=1), grid)
        tables += [np.ascontiguousarray(terms.T), rows]
    return _pairing_scores(decidable, odd, parent_offsets, parent_edges, *tables)


def _hotspot_ceiling(left, right, config, chromosome_map):
    """Point hotspot XOR background parity across the locus (Haldane)."""
    morgans = (
        float(np.diff(chromosome_map.cumulative_morgans(np.asarray([left, right], dtype=float)))[0])
        if chromosome_map.has_map
        else chromosome_map.fallback_rate_per_bp * (right - left)
    )
    background = haldane_fraction(max(morgans, 0.0))
    point = config.hotspot_switch_probability
    return point + background - 2.0 * point * background


def detect_junctions(
    contig: str,
    names: Sequence[str],
    parents: np.ndarray,
    children: np.ndarray,
    crossovers: np.ndarray,
    crossover_offsets: np.ndarray,
    reported: np.ndarray,
    callable_spans: np.ndarray,
    callable_offsets: np.ndarray,
    config: JunctionParameters,
    chromosome_map: ChromosomeGeneticMap,
    breakpoints: tuple[NDArray[np.int64], NDArray[np.int64], tuple[str, ...]] | None = None,
    *,
    reverse: bool = False,
) -> tuple[JunctionLocus, ...]:
    """Junction loci of one chromosome (module docstring); per-edge tables are packed with offsets. Exact ties of
    the report go to the last candidate under the convention reversal (module docstring)."""
    rev = bool(reverse)
    edges = len(parents)
    members_of = sorted(
        (int(reported[index, 0]), int(reported[index, 1]), edge, index)
        for edge in range(edges)
        for index in range(crossover_offsets[edge], crossover_offsets[edge + 1])
    )
    groups: list[list[Any]] = []
    index: int | np.integer
    for lo, hi, edge, index in members_of:
        overlaps = bool(groups and lo <= groups[-1][1])
        if overlaps:
            groups[-1][1] = max(groups[-1][1], hi)
            groups[-1][2].append((edge, index))
        else:
            groups.append([lo, hi, [(edge, index)]])
    loci: list[_Locus] = []
    for left, right, members in groups:
        parent_count = len({int(parents[edge]) for edge, _ in members})
        if parent_count < config.minimum_parents:
            continue
        decoded = np.asarray([crossovers[index, :2] for _, index in members])
        first, last = (
            float(cast(Callable[[], np.float64], decoded[:, 0].min)()),
            float(cast(Callable[[], np.float64], decoded[:, 1].max)()),
        )
        odd = np.zeros(edges, dtype=np.int64)
        for edge, _ in members:
            odd[edge] ^= 1
        with_calls = {edge for edge, _ in members}
        decidable = cast(
            NDArray[np.bool_],
            np.asarray(
                [
                    edge in with_calls
                    or any(
                        span_left <= first and span_right >= last
                        for span_left, span_right, _ in callable_spans[
                            callable_offsets[edge] : callable_offsets[edge + 1]
                        ]
                    )
                    for edge in range(edges)
                ]
            ),
        )
        loci.append(dict(left=left, right=right, first=first, last=last, members=members, odd=odd, decidable=decidable))
    rows: list[dict[str, Any]] = []
    # Integer counts per parent need one pass over edges per locus. Stable CSR
    # preserves the parent and child report order and the likelihood sum order.
    unique_parents, parent_index, parent_sizes = np.unique(parents, return_inverse=True, return_counts=True)
    parent_edges = np.argsort(parents, kind="stable")
    parent_offsets = np.r_[0, np.cumsum(parent_sizes)].astype(np.int64)
    for number, locus in enumerate(loci):
        counts: list[tuple[int, int]] | NDArray[np.int64]
        children_text: list[str]
        counts, parent_text, children_text = [], [], []
        mixed = all_or_none = 0
        odd_decidable = locus["decidable"] & (locus["odd"] == 1)
        n_by_parent = np.bincount(parent_index[locus["decidable"]], minlength=len(unique_parents))
        k_by_parent = np.bincount(parent_index[odd_decidable], minlength=len(unique_parents))
        for index in np.flatnonzero(n_by_parent):
            parent = unique_parents[index]
            k, n = int(k_by_parent[index]), int(n_by_parent[index])
            counts.append((k, n))
            if k:
                parent_text.append(f"{names[parent]}:{k}/{n}")
                edges_here = parent_edges[parent_offsets[index] : parent_offsets[index + 1]]
                children_text.extend(names[child] for child in children[edges_here[odd_decidable[edges_here]]])
                mixed += k < n
                whole = k == n and n >= 2
                all_or_none += whole
        counts = np.asarray(counts, dtype=np.int64).reshape(-1, 2)
        ceiling = _hotspot_ceiling(locus["left"], locus["right"], config, chromosome_map)
        mixture = mixture_terms(counts, SWITCH_GRID).sum(axis=1)
        bounded = mixture_terms(counts, np.r_[SWITCH_GRID[SWITCH_GRID < ceiling], max(ceiling, SWITCH_GRID[0])]).sum(
            axis=1
        )
        nearest, distance = "", np.nan
        if breakpoints is not None:
            for start, end in breakpoint_intervals(breakpoints):
                gap = max(int(start) - locus["right"], locus["left"] - int(end), 0)
                # The first nearest interval (the last under the reversal); distance starts as NaN.
                closer = (not gap > distance) if rev else (not gap >= distance)
                if closer:
                    nearest, distance = f"{start}-{end}", gap
        fraction_index = conv_argmax(mixture, reverse=rev)
        rows.append(
            dict(
                contig=contig,
                locus=f"{contig}:{number}",
                left_bp=int(locus["left"]),
                right_bp=int(locus["right"]),
                calls_left_bp=int(locus["first"]),
                calls_right_bp=int(locus["last"]),
                calls=len(locus["members"]),
                parents_with_calls=len(parent_text),
                odd_meioses=int(counts[:, 0].sum()),
                decidable_meioses=int(counts[:, 1].sum()),
                # NumPy integer counts use real arithmetic and yield a float64 quotient.
                switch_fraction=float(
                    cast(np.float64, cast("Real", counts[:, 0].sum()) / cast("Real", max(counts[:, 1].sum(), 1)))
                ),
                flip_adjusted_switch_fraction=float(SWITCH_GRID[fraction_index]),
                hotspot_ceiling=ceiling,
                mixed_parents=mixed,
                all_or_none_parents=all_or_none,
                parents=";".join(parent_text),
                children=";".join(children_text),
                mixing_log_lr=max(float(mixture.max() - bounded.max()), 0.0),
                paired_locus="",
                paired_odd_meioses=0,
                pairing_log_lr=0.0,
                pairing_tests=0,
                nearest_supplied_breakpoint=nearest,
                breakpoint_distance_bp=float(distance),
            )
        )
    tests = len(loci) * (len(loci) - 1) // 2
    scores = (
        _pairing_log_lr(loci, parents, parent_offsets=parent_offsets, parent_edges=parent_edges)
        if len(loci) > 1
        else None
    )
    for i, locus in enumerate(loci):
        # The best pairing is the first locus with the largest positive score (the last under the reversal).
        j = -1 if scores is None else conv_argmax(scores[i], reverse=rev)
        positive = j >= 0 and cast(np.ndarray, scores)[i, j] > 0.0
        if positive:
            other = loci[j]
            both = locus["decidable"] & other["decidable"]
            rows[i].update(
                paired_locus=rows[j]["locus"],
                paired_odd_meioses=int(((locus["odd"] == 1) & (other["odd"] == 1) & both).sum()),
                pairing_log_lr=float(cast(np.ndarray, scores)[i, j]),
            )
        rows[i]["pairing_tests"] = tests
    result = []
    for row in rows:
        adjusted = row["pairing_log_lr"] - (math.log(row["pairing_tests"]) if row["pairing_tests"] else 0.0)
        row["junction_log_lr"] = max(row["mixing_log_lr"], adjusted, 0.0)
        row["flagged"] = bool(row["junction_log_lr"] >= config.flag_log_likelihood_ratio)
        if row["flagged"]:
            row["interpretation"] = "junction"
        elif row["all_or_none_parents"] and row["flip_adjusted_switch_fraction"] <= row["hotspot_ceiling"]:
            row["interpretation"] = "parent_frame_flips"
        else:
            row["interpretation"] = "recombination_or_unresolved"
        result.append(JunctionLocus(**row))
    value = tuple(result)
    return value


def contig_breakpoints(
    rows: Sequence[tuple[str, int, int, str]], contig: str
) -> tuple[NDArray[np.int64], NDArray[np.int64], tuple[str, ...]] | None:
    """Ordinary intervals int64[K,2], reversed segments int64[J,2], all reasons.

    Returns None when the contig has no rows. Each array keeps its input order;
    the reason tuple interleaves both kinds in the original row order.
    """
    selected = [row for row in rows if row[0] == contig]
    if not selected:
        return None
    order = [(start, end) for _, start, end, reason in selected if not reason.startswith(REVERSED_SEGMENT)]
    reversed_segments = [(start, end) for _, start, end, reason in selected if reason.startswith(REVERSED_SEGMENT)]
    ordinary = np.asarray(order, dtype=np.int64).reshape(-1, 2)
    inverted = np.asarray(reversed_segments, dtype=np.int64).reshape(-1, 2)
    ordinary.flags.writeable = inverted.flags.writeable = False
    return ordinary, inverted, tuple(reason for *_, reason in selected)
