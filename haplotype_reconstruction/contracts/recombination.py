"""Plain-payload recombination calls and reports, conditional on fixed phase chains.

Products store these typed fields via the shared payload codec. Crossovers and
exposure keep their working-value meanings,
not posterior means. Elapsed time belongs in run records, outside this product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .roles import freeze_record_arrays, record_field, schema_field


@dataclass(frozen=True)
class RecombinationMap:
    """Per map bin (edges_bp float64 [B+1]): crossover_count (called events, spread over their intervals),
    exposure_meiosis_bp (observable meiosis-bp), expected_crossovers (given the marker intervals' switch
    probabilities), called_exposure_meiosis_bp,
    called_rate_cM_per_Mb, covered_bp (union of observable sequence), effective_meioses, rate_cM_per_Mb,
    cM_on_covered_bp; cumulative_observed_cM float64 [B+1] (cumulative over covered sequence; gaps omitted)."""

    edges_bp: NDArray[np.float64] = record_field("report", axes=("Bm+1",), units="bp", dtype="float64")
    crossover_count: NDArray[np.float64] = record_field("report", axes=("Bm",), units="count", dtype="float64")
    exposure_meiosis_bp: NDArray[np.float64] = record_field("report", axes=("Bm",), units="meiosis-bp", dtype="float64")
    expected_crossovers: NDArray[np.float64] = record_field("report", axes=("Bm",), units="count", dtype="float64")
    called_exposure_meiosis_bp: NDArray[np.float64] = record_field(
        "report", axes=("Bm",), units="meiosis-bp", dtype="float64"
    )
    called_rate_cM_per_Mb: NDArray[np.float64] = record_field("report", axes=("Bm",), units="cM/Mb", dtype="float64")
    covered_bp: NDArray[np.float64] = record_field("report", axes=("Bm",), units="bp", dtype="float64")
    effective_meioses: NDArray[np.float64] = record_field("report", axes=("Bm",), units="count", dtype="float64")
    rate_cM_per_Mb: NDArray[np.float64] = record_field("report", axes=("Bm",), units="cM/Mb", dtype="float64")
    cM_on_covered_bp: NDArray[np.float64] = record_field("report", axes=("Bm",), units="cM", dtype="float64")
    cumulative_observed_cM: NDArray[np.float64] = record_field("report", axes=("Bm+1",), units="cM", dtype="float64")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class JunctionLocus:
    """One row of the junction table (a locus of overlapping reported call intervals)."""

    contig: str = record_field("report", axes=(), units="1", dtype="str")
    locus: str = record_field("report", axes=(), units="1", dtype="str")
    left_bp: int = record_field("report", axes=(), units="bp", dtype="int")
    right_bp: int = record_field("report", axes=(), units="bp", dtype="int")
    calls_left_bp: int = record_field("report", axes=(), units="bp", dtype="int")
    calls_right_bp: int = record_field("report", axes=(), units="bp", dtype="int")
    calls: int = record_field("report", axes=(), units="1", dtype="int")
    parents_with_calls: int = record_field("report", axes=(), units="1", dtype="int")
    odd_meioses: int = record_field("report", axes=(), units="1", dtype="int")
    decidable_meioses: int = record_field("report", axes=(), units="1", dtype="int")
    switch_fraction: float = record_field("report", axes=(), units="1", dtype="float")
    flip_adjusted_switch_fraction: float = record_field("report", axes=(), units="1", dtype="float")
    hotspot_ceiling: float = record_field("report", axes=(), units="1", dtype="float")
    mixed_parents: int = record_field("report", axes=(), units="1", dtype="int")
    all_or_none_parents: int = record_field("report", axes=(), units="1", dtype="int")
    parents: str = record_field("report", axes=(), units="1", dtype="str")
    children: str = record_field("report", axes=(), units="1", dtype="str")
    mixing_log_lr: float = record_field("report", axes=(), units="nats", dtype="float")
    paired_locus: str = record_field("call", axes=(), units="1", dtype="str")
    paired_odd_meioses: int = record_field("report", axes=(), units="1", dtype="int")
    pairing_log_lr: float = record_field("report", axes=(), units="nats", dtype="float")
    pairing_tests: int = record_field("report", axes=(), units="1", dtype="int")
    junction_log_lr: float = record_field("report", axes=(), units="nats", dtype="float")
    flagged: bool = record_field("call", axes=(), units="1", dtype="bool")
    interpretation: str = record_field("call", axes=(), units="1", dtype="str")
    nearest_supplied_breakpoint: str = record_field("report", axes=(), units="1", dtype="str")
    breakpoint_distance_bp: float = record_field("report", axes=(), units="bp", dtype="float")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class AppliedBreakpoints:
    """Supplied order breakpoints of the contig: breakpoints int64 (k, 2) and reversed segments int64 (j, 2)
    (1-based inclusive bp), their reasons, masked sites, and the assembly component of every decoded segment."""

    breakpoints: NDArray[np.int64] = record_field("obs", axes=("Q", "2"), units="bp", dtype="int64")
    reversed_segments: NDArray[np.int64] = record_field("obs", axes=("R", "2"), units="bp", dtype="int64")
    reasons: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="str")
    masked_sites: int = record_field("report", axes=(), units="site", dtype="int")
    segment_components: NDArray[np.int32] = record_field("state", axes=("C",), units="1", dtype="int32")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class RecombinationSummary:
    """One row of the recombination summary table."""

    contig: str = record_field("report", axes=(), units="1", dtype="str")
    meioses: int = record_field("report", axes=(), units="1", dtype="int")
    observable_meioses: int = record_field("report", axes=(), units="1", dtype="int")
    crossovers: int = record_field("report", axes=(), units="1", dtype="int")
    expected_crossovers: float = record_field("report", axes=(), units="1", dtype="float")
    unresolved_bins: int = record_field("report", axes=(), units="1", dtype="int")
    exposure_meiosis_bp: float = record_field("report", axes=(), units="meiosis-bp", dtype="float")
    covered_bp: float = record_field("report", axes=(), units="bp", dtype="float")
    cumulative_observed_cM: float = record_field("report", axes=(), units="cM", dtype="float")
    uncovered_bins: int = record_field("report", axes=(), units="1", dtype="int")
    order_breakpoints: int = record_field("report", axes=(), units="1", dtype="int")
    order_breakpoint_masked_sites: int = record_field("report", axes=(), units="1", dtype="int")
    junction_loci: int = record_field("report", axes=(), units="1", dtype="int")
    junction_loci_flagged: int = record_field("report", axes=(), units="1", dtype="int")
    family_phase_converged: bool = record_field("state", axes=(), units="1", dtype="bool")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class Recombination:
    """Recombination of one chromosome, conditional on family phase's meiosis chains and the fixed pedigree.

    edge_parent/edge_child/edge_slot int64 [E]; per-edge tables packed with offsets int64 [E+1]:
    crossovers float64 [n, 5] (left bp, right bp, released-homologue support left/right, assembly component: changes
    of parental homologue between the centres of consecutive released chain bins; even crossovers within one
    interval are not identifiable) with reported_intervals int64 [n, 2] (both bins, inclusive bp) and
    in_reversed_segment bool [n]; callable_spans float64 [m, 3] (the followed
    stretches between released chain bins, as (left bp, right bp, component)); marker_counts int64 [E, 3] (released
    chain bins, bins of followed stretches, of them without a released homologue); expected_crossovers / exposure
    float64 [E, B] per map bin (working expected crossovers: Haldane interpolations conditional on the called
    endpoint states of the released chain bins, not posterior means; followed meiosis-bp).
    """

    schema: int = schema_field(2)
    contig: str = record_field("obs", axes=(), units="1", dtype="str")
    sample_ids: tuple[str, ...] = record_field("obs", axes=("N",), units="1", dtype="str")
    edge_parent: NDArray[np.int64] = record_field("call", axes=("E",), units="1", dtype="int64")
    edge_child: NDArray[np.int64] = record_field("call", axes=("E",), units="1", dtype="int64")
    edge_slot: NDArray[np.int64] = record_field("call", axes=("E",), units="1", dtype="int64")
    crossover_offsets: NDArray[np.int64] = record_field("call", axes=("E+1",), units="1", dtype="int64")
    crossovers: NDArray[np.float64] = record_field(
        "call", axes=("Q", "5"), units="bp,bp,probability,probability,index", dtype="float64"
    )
    reported_intervals: NDArray[np.int64] = record_field("call", axes=("Q", "2"), units="bp", dtype="int64")
    in_reversed_segment: NDArray[np.bool_] = record_field("call", axes=("Q",), units="1", dtype="bool")
    callable_offsets: NDArray[np.int64] = record_field("call", axes=("E+1",), units="1", dtype="int64")
    callable_spans: NDArray[np.float64] = record_field("call", axes=("R", "3"), units="bp,bp,index", dtype="float64")
    marker_counts: NDArray[np.int64] = record_field("report", axes=("E", "3"), units="1", dtype="int64")
    expected_crossovers: NDArray[np.float64] = record_field("report", axes=("E", "Bm"), units="1", dtype="float64")
    exposure: NDArray[np.float64] = record_field("report", axes=("E", "Bm"), units="meiosis-bp", dtype="float64")
    map: RecombinationMap = record_field("report", axes=(), units="1", dtype="RecombinationMap")
    junctions: tuple[JunctionLocus, ...] = record_field("report", axes=(), units="1", dtype="JunctionLocus")
    order_breakpoints: AppliedBreakpoints | None = record_field(
        "state", axes=(), units="1", dtype="AppliedBreakpoints | None"
    )
    summary: RecombinationSummary = record_field("report", axes=(), units="1", dtype="RecombinationSummary")

    def edge_crossovers(self, edge: int) -> NDArray[np.float64]:
        return self.crossovers[self.crossover_offsets[edge] : self.crossover_offsets[edge + 1]]

    def edge_reported_intervals(self, edge: int) -> NDArray[np.int64]:
        return self.reported_intervals[self.crossover_offsets[edge] : self.crossover_offsets[edge + 1]]

    def edge_callable_spans(self, edge: int) -> NDArray[np.float64]:
        return self.callable_spans[self.callable_offsets[edge] : self.callable_offsets[edge + 1]]

    def __post_init__(self) -> None:
        freeze_record_arrays(self)
