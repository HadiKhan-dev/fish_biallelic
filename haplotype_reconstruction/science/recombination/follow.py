"""Follow released meiosis chains and construct conditional recombination reports.

The chain supplies parity only; expected counts condition on released endpoints,
not the full read posterior. No exposure crosses a reset or excluded order zone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np

from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.pedigree import ParentEdges
from ...contracts.phase import PhasedSamples
from ...contracts.recombination import AppliedBreakpoints, Recombination, RecombinationSummary
from ...core.errors import InvariantError
from . import junctions as junction_report
from .diagnostics import RecombinationTrace, Traced
from .junctions import contig_breakpoints
from .map import aggregate_map, genetic_coordinates, map_bins, posterior_intervals
from .parameters import RecombinationParameters

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ...contracts.recombination import RecombinationMap

    _Breakpoints = tuple[NDArray[np.int64], NDArray[np.int64], tuple[str, ...]]
    # schema_field is keyword-only at runtime; mypy cannot see through its helper.
    _RecombinationFactory = Callable[
        [
            str,
            tuple[str, ...],
            NDArray[np.int64],
            NDArray[np.int64],
            NDArray[np.int64],
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.bool_],
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.float64],
            RecombinationMap,
            tuple[junction_report.JunctionLocus, ...],
            AppliedBreakpoints | None,
            RecombinationSummary,
        ],
        Recombination,
    ]


def packed(rows, columns):
    """(offsets int64 [E+1], float64 [n, columns]) of per-edge row arrays."""
    offsets = np.r_[0, np.cumsum([len(item) for item in rows])].astype(np.int64)
    values = (
        np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1, columns) for item in rows])
        if len(rows)
        else np.zeros((0, columns))
    )
    return offsets, values.reshape(-1, columns)


def chain_bins(phase, segments, *, trace=False, audit=None):
    """(left, right, centre bp float64 [M], segment int64 [M], reset bool [M]) of the chain arrays' painting bins:
    each bin's half-open bp range and centre (family phase's bin_centers: the mean position of its sites, which the
    meiosis transitions use), the segment of its sites on the decoded axis ``segments`` (-1 when its sites fall in
    two segments or in an excluded breakpoint zone), and where a new segment starts."""
    audit = RecombinationTrace() if trace and audit is None else audit
    edges = [phase.component_bin_edges(c) for c in range(len(phase.bin_edge_offsets) - 1)]
    left = np.concatenate([e[:-1] for e in edges]).astype(np.float64) if edges else np.zeros(0)
    right = np.concatenate([e[1:] for e in edges]).astype(np.float64) if edges else np.zeros(0)
    positions = np.asarray(phase.positions, dtype=np.int64)
    first = np.searchsorted(positions, left, side="left")
    last = np.searchsorted(positions, right, side="left") - 1
    segment = np.full(len(left), -1, dtype=np.int64)
    for b in range(len(left)):
        if last[b] >= first[b]:
            values = segments[first[b] : last[b] + 1]
            same = values[0] >= 0 and np.all(values == values[0])
            if audit is not None:
                audit.emit("bin.segment", same, bin=int(b), segments=values.copy())
            if same:
                segment[b] = values[0]
    reset = np.ones(len(left), dtype=bool)
    reset[1:] = segment[1:] != segment[:-1]
    result = left, right, np.asarray(phase.bin_centers, dtype=np.float64), segment, reset
    if audit is not None:
        audit.bin_axis = result
    return Traced(result, audit) if trace else result


def follow_chain(source, support, reset, left, right, centres, segment, *, trace=False, audit=None):
    """One meiosis from its chain arrays: (called events (n, 5): left bp, right bp, released-homologue support left
    and right, segment; reported intervals (n, 2), inclusive bp; callable spans (m, 3); marker counts (released bins,
    bins of followed stretches, of them without a released homologue); the map markers for
    map.posterior_intervals: the released bins, whether each starts a stretch, and the released parity of the
    interval ending at each)."""
    audit = RecombinationTrace() if trace and audit is None else audit
    released = (source >= 0) & (segment >= 0)
    stretch = np.cumsum(reset)
    events, reported, spans = [], [], []
    followed = np.zeros(len(source), dtype=bool)
    previous = -1
    for b in np.flatnonzero(released):
        if previous >= 0 and stretch[b] == stretch[previous]:
            followed[previous + 1 : b + 1] = True
            part = float(segment[b])
            merge = bool(spans and spans[-1][1] == centres[previous] and spans[-1][2] == part)
            if audit is not None and spans:
                audit.emit(
                    "span.merge",
                    merge,
                    last_right=spans[-1][1],
                    last_segment=spans[-1][2],
                    left=float(centres[previous]),
                    segment=part,
                )
            if merge:
                spans[-1] = (spans[-1][0], float(centres[b]), part)
            else:
                spans.append((float(centres[previous]), float(centres[b]), part))
            changed = source[b] != source[previous]
            if audit is not None:
                audit.emit(
                    "event",
                    changed,
                    previous=int(source[previous]),
                    current=int(source[b]),
                    previous_bin=int(previous),
                    current_bin=int(b),
                )
            if changed:
                events.append(
                    (float(centres[previous]), float(centres[b]), float(support[previous]), float(support[b]), part)
                )
                reported.append((int(left[previous]), int(right[b]) - 1))
        previous = b
    inside = followed.copy()
    starts = np.flatnonzero(followed & ~np.r_[False, followed[:-1]])
    inside[np.clip(starts - 1, 0, None)] = True
    counts = (int(released.sum()), int(inside.sum()), int(np.sum(inside & ~released)))
    markers = np.flatnonzero(released)
    marker_reset = np.ones(len(markers), dtype=bool)
    marker_reset[1:] = stretch[markers[1:]] != stretch[markers[:-1]]
    parity = np.zeros(len(markers), dtype=np.float64)
    parity[1:] = source[markers[1:]] != source[markers[:-1]]
    result = (
        np.asarray(events, dtype=np.float64).reshape(-1, 5),
        np.asarray(reported, dtype=np.int64).reshape(-1, 2),
        np.asarray(spans, dtype=np.float64).reshape(-1, 3),
        counts,
        (markers, marker_reset, parity),
    )

    if audit is not None:
        audit.followed.append(result)
    return Traced(result, audit) if trace else result


@overload
def map_chromosome(
    phase: PhasedSamples,
    edges: ParentEdges,
    config: RecombinationParameters,
    chromosome_map: ChromosomeGeneticMap,
    *,
    reverse: bool = False,
    trace: Literal[False] = False,
) -> Recombination: ...


@overload
def map_chromosome(
    phase: PhasedSamples,
    edges: ParentEdges,
    config: RecombinationParameters,
    chromosome_map: ChromosomeGeneticMap,
    *,
    reverse: bool = False,
    trace: Literal[True],
) -> Traced[Recombination]: ...


@overload
def map_chromosome(
    phase: PhasedSamples,
    edges: ParentEdges,
    config: RecombinationParameters,
    chromosome_map: ChromosomeGeneticMap,
    *,
    reverse: bool = False,
    trace: bool,
) -> Recombination | Traced[Recombination]: ...


def map_chromosome(
    phase: PhasedSamples,
    edges: ParentEdges,
    config: RecombinationParameters,
    chromosome_map: ChromosomeGeneticMap,
    *,
    reverse: bool = False,
    trace: bool = False,
) -> Recombination | Traced[Recombination]:
    """Recombination of one chromosome, or its producer trace when requested.

    ``phase`` PhasedSamples with meiosis chains; ``edges`` ParentEdges (the edges the phase was
    conditioned on, whose order the chain rows follow).
    """
    audit = RecombinationTrace() if trace else None
    if phase.edges_identity != edges.identity:
        raise InvariantError(f"{phase.contig}: family phase was conditioned on a different pedigree")
    if phase.chain_source is None:
        raise InvariantError(
            f"{phase.contig}: the family-phase product carries no meiosis chains; recompute family phase"
        )
    names = phase.sample_ids
    positions = np.asarray(phase.positions, dtype=np.int64)
    components = np.asarray(phase.component_ids, dtype=np.int32)
    bin_edges = tuple(phase.component_bin_edges(c) for c in range(len(phase.bin_edge_offsets) - 1))
    breakpoints = contig_breakpoints(config.order_breakpoints, phase.contig)
    if breakpoints is None:
        segments, origin, masked = components, None, 0
    else:
        segments, origin, masked = junction_report.split_at_breakpoints(
            positions, components, breakpoints, bin_edges, audit=audit
        )
    parents, children, slots = (np.asarray(x, dtype=np.int64) for x in (edges.parents, edges.children, edges.slots))
    left, right, centres, segment, segment_reset = chain_bins(phase, segments, audit=audit)
    bins = map_bins(positions, segments, config.bin_bp)
    genetic, knots, knot_genetic = genetic_coordinates(positions, chromosome_map)
    centre_genetic = None if knots is None else np.interp(centres, knots, knot_genetic)
    rate = chromosome_map.fallback_rate_per_bp
    events, reported, spans = [], [], []
    marker_counts = np.zeros((len(parents), 3), dtype=np.int64)
    expected = np.zeros((len(parents), len(bins) - 1))
    exposure = np.zeros_like(expected)
    for edge in range(len(parents)):
        if audit is not None:
            audit.edge = edge
        called, report, callable_spans, counts, (markers, marker_reset, parity) = follow_chain(
            np.asarray(phase.chain_source[edge]),
            np.asarray(phase.chain_support[edge]),
            np.asarray(phase.chain_reset[edge]) | segment_reset,
            left,
            right,
            centres,
            segment,
            audit=audit,
        )
        if centre_genetic is None:
            soft = posterior_intervals(centres[markers], marker_reset, parity, bins, rate, audit=audit)
        else:
            soft = posterior_intervals(
                centres[markers],
                marker_reset,
                parity,
                bins,
                rate,
                centre_genetic[markers],
                knots,
                knot_genetic,
                audit=audit,
            )
        events.append(called)
        reported.append(report)
        spans.append(callable_spans)
        expected[edge], exposure[edge] = soft[0], soft[1]
        marker_counts[edge] = counts
    crossover_offsets, crossover_rows = packed(events, 5)
    callable_offsets, callable_rows = packed(spans, 3)
    reported_rows = np.concatenate(reported).astype(np.int64) if reported else np.zeros((0, 2), dtype=np.int64)
    curve = aggregate_map(bins, crossover_rows, callable_rows, expected, exposure, knots, knot_genetic)
    applied = None
    if origin is not None:
        # Calls and spans were followed per segment; report assembly components again.
        for rows, column in ((crossover_rows, 4), (callable_rows, 2)):
            if len(rows):
                rows[:, column] = origin[rows[:, column].astype(np.int64)]
        applied = AppliedBreakpoints(
            cast("_Breakpoints", breakpoints)[0],
            cast("_Breakpoints", breakpoints)[1],
            cast("_Breakpoints", breakpoints)[2],
            masked,
            origin,
        )
    if audit is not None:
        audit.edge = -1
    reversed_calls = junction_report.in_reversed_segment(crossover_rows, breakpoints, audit=audit)
    loci = junction_report.detect_junctions(
        phase.contig,
        names,
        parents,
        children,
        crossover_rows,
        crossover_offsets,
        reported_rows,
        callable_rows,
        callable_offsets,
        config.junctions,
        chromosome_map,
        breakpoints,
        reverse=reverse,
        audit=audit,
    )
    summary = RecombinationSummary(
        phase.contig,
        len(parents),
        int(np.count_nonzero(np.diff(callable_offsets))),
        len(crossover_rows),
        float(curve.expected_crossovers.sum()),
        int(marker_counts[:, 2].sum()),
        float(curve.exposure_meiosis_bp.sum()),
        float(curve.covered_bp.sum()),
        float(curve.cumulative_observed_cM[-1]),
        int(np.sum(curve.exposure_meiosis_bp == 0)),
        0 if breakpoints is None else len(breakpoints[0]) + len(breakpoints[1]),
        masked,
        len(loci),
        sum(locus.flagged for locus in loci),
        bool(phase.diagnostics.converged),
    )
    product = cast("_RecombinationFactory", Recombination)(
        phase.contig,
        names,
        parents,
        children,
        slots,
        crossover_offsets,
        crossover_rows,
        reported_rows,
        reversed_calls,
        callable_offsets,
        callable_rows,
        marker_counts,
        expected,
        exposure,
        curve,
        loci,
        applied,
        summary,
    )

    return Traced(product, cast(RecombinationTrace, audit)) if trace else product
