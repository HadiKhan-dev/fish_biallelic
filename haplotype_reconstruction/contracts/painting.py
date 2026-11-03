"""Painting records, supplied release views and explicit selective payload layout.

Owns immutable state/bin/transition records and released calls, not raw Viterbi
paths or posterior buffers. Panel row paths and optional packed emissions travel
as independent top-level payload items; adapters supply assembly for panel views.
Pure views are relocated from painting.release (PA2 handoff, PR19); no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any, ClassVar

import numpy as np
from numba import prange, types
from numpy.typing import NDArray

from ..core.errors import InvariantError
from ..core.kernels import kernel
from ..core.ties import TieCounts, noise_quantum
from .panels import ComponentPanel, FounderComponent, PreparedPanel, materialise
from .payload import from_payload, to_payload
from .roles import freeze_record_arrays, record_field, require_schema, schema_field


class TrackStatus(IntEnum):
    """Scientific meaning of one released track in one bin (stored as uint8)."""

    INELIGIBLE_NO_EVIDENCE = 0
    LOW_POSTERIOR_ABSTENTION = 1
    SINGLETON_NAMED = 2
    POOLED_EQUIVALENCE = 3
    BACKGROUND = 4
    UNANCHORED_TRAJECTORY = 5
    LOCAL_ALLELE_EQUIVALENCE = 6
    SWITCH_INTERVAL = 7


@dataclass(frozen=True)
class PaintedChunk:
    __payload_view__: ClassVar[bool] = True
    "Derived release of both tracks on [start, end) bp; never stored."
    start: int = record_field("call", axes=(), units="bp", dtype="int", stored=False)
    end: int = record_field("call", axes=(), units="bp", dtype="int", stored=False)
    labels: tuple[int, int] = record_field("call", axes=("2",), units="1", dtype="int", stored=False)
    members: tuple[tuple[int, ...], tuple[int, ...]] = record_field(
        "call", axes=("2",), units="1", dtype="tuple[int]", stored=False
    )

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


def coalesce_chunks(
    edges: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    status: np.ndarray,
    members: np.ndarray,
    member_sets: tuple[tuple[int, ...], ...],
) -> list[PaintedChunk]:
    """Coalesce adjacent bins with equal labels, classes, status and members on both tracks."""
    if labels.shape[1] == 0:
        return []
    n_bins = labels.shape[1]
    changed = np.zeros(n_bins, dtype=np.bool_)
    changed[0] = True
    for grid in (labels, classes, status, members):
        changed[1:] |= (grid[0, 1:] != grid[0, :-1]) | (grid[1, 1:] != grid[1, :-1])
    starts = np.flatnonzero(changed)
    stops = np.r_[starts[1:], n_bins]
    return [
        PaintedChunk(
            int(edges[start]),
            int(edges[stop]),
            (int(labels[0, start]), int(labels[1, start])),
            (member_sets[members[0, start]], member_sets[members[1, start]]),
        )
        for start, stop in zip(starts, stops, strict=True)
    ]


def state_alt_probabilities(
    named_alleles: NDArray[np.int8], background_frequency: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Called class alleles, otherwise the recorded background frequency (P0 PR01).

    Shared by PaintingStates and the emissions producer; no clipping or prior.
    """
    return np.where(named_alleles >= 0, named_alleles, background_frequency[None, :]).astype(np.float64)


@dataclass(frozen=True)
class PaintingStates:
    """State alphabet of one component: C trajectory classes plus BACKGROUND (state C).

    site_indices: int64[L_s] selected sites as indices into the component
        panel's positions (kept sites with at least one inference call).
    positions: int64[L_s] their bp positions (strictly increasing).
    named_alleles: int8[C, L_s] each class's allele at the selected sites,
        -1 where the class is uncalled. No two classes are equal.
    classes: tuple of C tuples of component-local founder rows (ascending);
        classes are ordered by their first row.
    background_frequency: float64[L_s] BACKGROUND's ALT probability,
        (1 + ALT calls) / (2 + calls) over the C classes (Beta(1, 1) posterior
        mean, see :func:`background_alt_frequency`).
    """

    site_indices: NDArray[np.int64] = record_field("state", axes=("L_s",), units="site", dtype="int64")
    positions: NDArray[np.int64] = record_field("obs", axes=("L_s",), units="bp", dtype="int64")
    named_alleles: NDArray[np.int8] = record_field("state", axes=("C", "L_s"), units="1", dtype="int8")
    classes: tuple[tuple[int, ...], ...] = record_field("state", axes=("C",), units="1", dtype="tuple[int]")
    background_frequency: NDArray[np.float64] = record_field(
        "state", axes=("L_s",), units="probability", dtype="float64"
    )

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_states(self) -> int:
        """S = C + 1 (BACKGROUND is the last state)."""
        return len(self.classes) + 1

    @property
    def n_sites(self) -> int:
        return len(self.positions)

    @property
    def anchored(self) -> np.ndarray:
        """bool[C]: the class is called at some selected site (else UNANCHORED)."""
        return np.any(self.named_alleles >= 0, axis=1)

    def alt_probability(self) -> np.ndarray:
        """float64[C, L_s]: each class's ALT probability (its call, else BACKGROUND's)."""
        return state_alt_probabilities(self.named_alleles, self.background_frequency)

    def public_state_ids(self) -> np.ndarray:
        """int32[S]: anchored classes keep their index; unanchored classes and
        BACKGROUND map to the public UNKNOWN index C."""
        result = np.arange(self.n_states, dtype=np.int32)
        result[:-1][~self.anchored] = self.n_classes
        return result

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class PaintingBins:
    """Consecutive runs of selected sites (B bins, B >= 1 when L_s >= 1).

    starts, stops: int64[B] half-open ranges into the selected sites; the
        selected sites are split into max(1, L_s // snps_per_bin) near-equal
        runs (``np.array_split`` sizes: the first L_s mod B runs are one longer).
    centers: float64[B] mean bp position of each bin's sites (transition distances).
    edges: int64[B + 1] bp boundaries: bin b owns [edges[b], edges[b + 1]);
        edges[0] = first site, edges[B] = last site + 1, and an inner edge is
        floor((last site of b - 1 + first site of b + 1) / 2).
    """

    starts: NDArray[np.int64] = record_field("state", axes=("B",), units="site", dtype="int64")
    stops: NDArray[np.int64] = record_field("state", axes=("B",), units="site", dtype="int64")
    centers: NDArray[np.float64] = record_field("state", axes=("B",), units="bp", dtype="float64")
    edges: NDArray[np.int64] = record_field("state", axes=("B+1",), units="bp", dtype="int64")

    @property
    def n_bins(self) -> int:
        return len(self.starts)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class PaintingTransitions:
    """Normalised ordered-diplotype Hamming transition of one component, with its copying-process parameters.

    n_states: S (classes + BACKGROUND).
    log_same, log_one, log_two: float64[B - 1], per bin edge e (between bins
        e and e + 1), the log probability of moving to the same ordered pair,
        to one particular pair that changes exactly one track, and to one
        particular pair that changes both tracks:
        exp(log_same) + 2 (S - 1) exp(log_one) + (S - 1)^2 exp(log_two) = 1.
    switch_rate: rho, ancestry changes per Morgan per track.
    error_entry, error_exit: a and b of the panel-error chain (painting.fidelity; 0 when it has no error state).
    site_correlation: r of the design effect 1 + (n - 1) r that divided each sample's bin log emission
        (PaintingConfig.emission_site_correlation; 0 = untempered); consumers of the emissions (pedigree's source
        chain) apply the same.
    """

    n_states: int = record_field("state", axes=(), units="1", dtype="int")
    log_same: NDArray[np.float64] = record_field("state", axes=("B-1",), units="nats", dtype="float64")
    log_one: NDArray[np.float64] = record_field("state", axes=("B-1",), units="nats", dtype="float64")
    log_two: NDArray[np.float64] = record_field("state", axes=("B-1",), units="nats", dtype="float64")
    switch_rate: float = record_field("state", axes=(), units="Morgan^-1", dtype="float64", default=0.0)
    error_entry: float = record_field("state", axes=(), units="probability", dtype="float64", default=0.0)
    error_exit: float = record_field("state", axes=(), units="probability", dtype="float64", default=0.0)
    site_correlation: float = record_field("state", axes=(), units="1", dtype="float", default=0.0)

    def probabilities(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(same, one, two) per edge, the exponentials of the log table."""
        return (np.exp(self.log_same), np.exp(self.log_one), np.exp(self.log_two))

    def switch_unit(self) -> float:
        """The median over edges of the log-probability cost of one track change against staying (nats)."""
        if len(self.log_same) == 0:
            return 0.0
        return float(np.median(self.log_same - self.log_one))

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class ComponentPainting:
    """The released painting of one component (N samples, B bins, S states).

    panel: the founder panel painted (assembly's component after label
        continuity); rows of every grid below are its component-local rows.
        None when the product was loaded without panels (records.load(..., assembly=None)).
    states, bins, transitions: the HMM's state alphabet, bins and transition
        table (pedigree rebuilds its source model from exactly these).
    local_keys: int32[B, S] per-bin allele-equivalence keys of the states.
    eligible: bool[N] the sample has an observed informative GL row at a
        selected site of the component (else it abstains everywhere).
    labels: int32[N, 2, B] released founder row of a SINGLETON_NAMED track, else -1.
    classes: int32[N, 2, B] released class (stage 1) or local key
        (LOCAL_ALLELE_EQUIVALENCE), -1 when abstaining or SWITCH_INTERVAL.
    status: uint8[N, 2, B] :class:`TrackStatus`.
    direct: bool[N, 2, B] the released class is called and read-observed at an
        informative selected site of the bin (direct callability).
    member_ids: int32[N, 2, B] indices into ``member_sets`` (0 = empty set).
    member_sets: interned ascending tuples of candidate founder rows; [0] = ().
    switch_counts: int64[N] minimum unordered track changes of the Viterbi
        path (a pure track swap is gauge, not a switch); 0 if ineligible.
    packed_emissions: float64[N, S (S + 1) / 2, B] upper-triangle binned log
        emissions (the ordered-pair emissions are symmetric), kept for
        pedigree preparation when they fit the chromosome's cache budget,
        else None.
    """

    panel: ComponentPanel | None = record_field(
        "state", axes=(), units="1", dtype="ComponentPanel | None", stored=False, default=None, kw_only=True
    )
    states: PaintingStates = record_field("state", axes=(), units="1", dtype="PaintingStates")
    bins: PaintingBins = record_field("state", axes=(), units="1", dtype="PaintingBins")
    transitions: PaintingTransitions = record_field("state", axes=(), units="1", dtype="PaintingTransitions")
    local_keys: NDArray[np.int32] = record_field("state", axes=("B", "S"), units="1", dtype="int32")
    eligible: NDArray[np.bool_] = record_field("state", axes=("N",), units="1", dtype="bool")
    labels: NDArray[np.int32] = record_field("call", axes=("N", "2", "B"), units="1", dtype="int32")
    classes: NDArray[np.int32] = record_field("call", axes=("N", "2", "B"), units="1", dtype="int32")
    status: NDArray[np.uint8] = record_field("call", axes=("N", "2", "B"), units="1", dtype="uint8")
    direct: NDArray[np.bool_] = record_field("call", axes=("N", "2", "B"), units="1", dtype="bool")
    member_ids: NDArray[np.int32] = record_field("call", axes=("N", "2", "B"), units="1", dtype="int32")
    member_sets: tuple[tuple[int, ...], ...] = record_field("call", axes=(), units="1", dtype="tuple[int]")
    switch_counts: NDArray[np.int64] = record_field("call", axes=("N",), units="1", dtype="int64")
    packed_emissions: NDArray[np.float64] | None = record_field(
        "state", axes=("N", "S*(S+1)/2", "B"), units="nats", dtype="float64", stored=False, default=None, kw_only=True
    )

    @property
    def interval(self) -> tuple[int, int]:
        """First and last bp position of the component panel (needs the panel)."""
        if self.panel is None:
            raise InvariantError("painting panel was not reconstructed")
        positions = self.panel.positions
        return (int(positions[0]), int(positions[-1]))

    def chunks(self, sample: int) -> list[PaintedChunk]:
        """Released chunks of one sample: consecutive bins with equal (label, class,
        status, members) on both tracks coalesce; ineligible samples have none."""
        if not self.eligible[sample] or self.bins.n_bins == 0:
            return []
        return coalesce_chunks(
            self.bins.edges,
            self.labels[sample],
            self.classes[sample],
            self.status[sample],
            self.member_ids[sample],
            self.member_sets,
        )

    def carrier_observed(
        self, positions: np.ndarray, columns: np.ndarray, observed: np.ndarray, n_rows: int
    ) -> np.ndarray:
        """bool[K, L]: founder row k at bp ``positions[j]`` is observed by a painted carrier (release 2.2): some
        eligible sample whose released track at that site names k in its member set (named, pooled, local
        equivalence or switch interval) has read depth there (``observed`` bool [N, L_e] at evidence columns
        ``columns`` int64 [L]). False outside the painting bins."""
        positions = np.asarray(positions, dtype=np.int64)
        if self.bins.n_bins == 0:
            return np.zeros((n_rows, len(positions)), dtype=np.bool_)
        bins = np.searchsorted(self.bins.edges, positions, side="right") - 1
        bins[(bins < 0) | (bins >= self.bins.n_bins)] = -1
        offsets = np.zeros(len(self.member_sets) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(rows) for rows in self.member_sets])
        rows = np.asarray([row for members in self.member_sets for row in members], dtype=np.int64)
        return _carrier_observed_kernel(
            np.ascontiguousarray(self.member_ids, dtype=np.int32),
            offsets,
            rows,
            np.ascontiguousarray(self.eligible, dtype=np.bool_),
            bins.astype(np.int64),
            np.asarray(columns, dtype=np.int64),
            np.ascontiguousarray(observed, dtype=np.bool_),
            int(n_rows),
        )

    def site_labels(self, positions: np.ndarray) -> np.ndarray:
        """int16[N, 2, L] released labels at bp ``positions`` (sorted); -1 outside the bins."""
        positions = np.asarray(positions, dtype=np.int64)
        n_samples = len(self.eligible)
        result = np.full((n_samples, 2, len(positions)), -1, dtype=np.int16)
        if self.bins.n_bins == 0:
            return result
        bins = np.searchsorted(self.bins.edges, positions, side="right") - 1
        inside = (bins >= 0) & (bins < self.bins.n_bins)
        result[:, :, inside] = self.labels[:, :, bins[inside]]
        return result

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class RelabelMove:
    """One accepted row-suffix relabelling of label continuity.

    atomic_span: index of the atomic span from which rows exchange labels;
    position: bp position of its first site; rows: the two founder rows;
    score_gain_switches: summed Viterbi score gain in units of one track change;
    painted_switch_change: change of the painted switch total (negative);
    winners, losers: samples whose painted switches fall / rise;
    winners_forward, winners_reverse (release 2.14): winners whose decoded path, under the labels before the move,
        switches from rows[0] to rows[1] at the link, or from rows[1] to rows[0]. A label exchange of two founders'
        rows makes carriers of either founder switch, so both directions are expected; one shared recombinant
        lineage switches one way only, and the panel's founder identity across the link is then not identified
        (``identified`` False: the move is released as identity-ambiguous).
    """

    atomic_span: int = record_field("state", axes=(), units="1", dtype="int")
    position: int = record_field("call", axes=(), units="bp", dtype="int")
    rows: tuple[int, int] = record_field("call", axes=("2",), units="1", dtype="int")
    score_gain_switches: float = record_field("state", axes=(), units="switch", dtype="float64")
    painted_switch_change: int = record_field("state", axes=(), units="switch", dtype="int")
    winners: tuple[int, ...] = record_field("call", axes=(), units="1", dtype="tuple[int]")
    losers: tuple[int, ...] = record_field("call", axes=(), units="1", dtype="tuple[int]")
    winners_forward: int = record_field("state", axes=(), units="1", dtype="int", default=0)
    winners_reverse: int = record_field("state", axes=(), units="1", dtype="int", default=0)

    @property
    def identified(self) -> bool:
        """Winners switch in both directions (a label exchange of two founders' carriers)."""
        return self.winners_forward > 0 and self.winners_reverse > 0

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class LabelContinuity:
    """Outcome of label continuity for one component.

    accepted: the painted panel is the relabelled one.
    reason: 'relabelled', 'no_link_or_movable_row', 'no_evidence', 'no_move',
        'trajectory_classes_changed' or 'painted_switches_not_lowered'.
    link_points: atomic-span boundaries inside the selected sites;
    movable_rows: singleton anchored classes (only these exchange labels);
    switch_unit_nats: median cost of one track change over the bin edges;
    switches_before / switches_after: painted switch totals of the eligible
        samples before and after (after = before unless accepted);
    candidate_checks: candidates re-decoded during the search;
    moves: accepted moves in search order (also when the final check rejects them);
    row_maps: int64[A, K] applied relabelling (span a, new row k carries old
        row row_maps[a, k]), None unless accepted.
    tie_events: candidate rankings decided inside a tie class (``maths.ties.TieCounter``
        record; release 2.1).
    """

    accepted: bool = record_field("call", axes=(), units="1", dtype="bool")
    reason: str = record_field("state", axes=(), units="1", dtype="str")
    link_points: int = record_field("state", axes=(), units="1", dtype="int")
    movable_rows: int = record_field("state", axes=(), units="1", dtype="int")
    switch_unit_nats: float | None = record_field("state", axes=(), units="nats", dtype="float64 | None")
    switches_before: int | None = record_field("state", axes=(), units="1", dtype="int | None")
    switches_after: int | None = record_field("state", axes=(), units="1", dtype="int | None")
    candidate_checks: int = record_field("state", axes=(), units="1", dtype="int")
    moves: tuple[RelabelMove, ...] = record_field("state", axes=(), units="1", dtype="tuple[RelabelMove, ...]")
    row_maps: NDArray[np.int64] | None = record_field("call", axes=("A", "K"), units="1", dtype="int64")
    tie_events: TieCounts = record_field(
        "diagnostic", axes=(), units="1", dtype="TieCounts", default_factory=lambda: TieCounts(())
    )

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class PaintingSettings:
    """Captured parameters from 3.5 painting/config.py:47–59; science owns defaults."""

    emission_site_correlation: float = record_field("state", axes=(), units="1", dtype="float")
    initial_switch_rate: float = record_field("state", axes=(), units="Morgan^-1", dtype="float")
    initial_error_entry: float = record_field("state", axes=(), units="probability", dtype="float")
    initial_error_exit: float = record_field("state", axes=(), units="probability", dtype="float")
    estimation_iterations: int = record_field("state", axes=(), units="count", dtype="int")
    estimate_grid: float = record_field("state", axes=(), units="nats", dtype="float")
    robustness_epsilon: float = record_field("state", axes=(), units="probability", dtype="float")
    snps_per_bin: int = record_field("state", axes=(), units="sites", dtype="int")
    emission_log_floor: float = record_field("state", axes=(), units="nats", dtype="float")
    minimum_release_posterior: float = record_field("state", axes=(), units="probability", dtype="float")
    label_continuity_minimum_switch_gain: float = record_field("state", axes=(), units="count", dtype="float")
    label_continuity_maximum_loser_ratio: float = record_field("state", axes=(), units="share", dtype="float")

    # 3.5 painting/hmm.py:47, P0 PR09, WP_PA2 §7. Captured in the payload.
    viterbi_tie: float = record_field("state", units="nats", dtype="float64", default=noise_quantum(1.9e-6))

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class Painting:
    __payload_custom__: ClassVar[bool] = True
    """Painting product with explicit hoisted row paths and optional emission caches."""
    schema: int = schema_field(1)
    contig: str = record_field("obs", axes=(), units="1", dtype="str")
    sample_ids: tuple[str, ...] = record_field("obs", axes=("N",), units="1", dtype="str")
    components: tuple[ComponentPainting, ...] = record_field("call", axes=("C",), units="1", dtype="ComponentPainting")
    label_continuity: tuple[LabelContinuity, ...] = record_field(
        "call", axes=("C",), units="1", dtype="LabelContinuity"
    )
    parameters: PaintingSettings = record_field("state", axes=(), units="1", dtype="PaintingSettings")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class ComponentShape:
    """Sizes of one painted component: founders K, classes C, states S = C + 1,
    selected sites, bins B, and whether packed emissions are stored."""

    founders: int = record_field("report", axes=(), units="1", dtype="int")
    classes: int = record_field("report", axes=(), units="1", dtype="int")
    states: int = record_field("report", axes=(), units="1", dtype="int")
    sites: int = record_field("report", axes=(), units="1", dtype="int")
    bins: int = record_field("report", axes=(), units="1", dtype="int")
    packed_emissions: bool = record_field("report", axes=(), units="1", dtype="bool")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class PaintingSummary:
    """Record ``painting/summary``: shapes, eligible sample-component pairs and label continuity;
    released switch intervals (runs of SWITCH_INTERVAL bins per sample track) and their track bins."""

    schema: int = schema_field(1)
    contig: str = record_field("report", axes=(), units="1", dtype="str")
    samples: int = record_field("report", axes=(), units="1", dtype="int")
    components: tuple[ComponentShape, ...] = record_field(
        "report", axes=(), units="1", dtype="tuple[ComponentShape, ...]"
    )
    eligible_pairs: int = record_field("report", axes=(), units="1", dtype="int")
    relabel_moves: int = record_field("report", axes=(), units="1", dtype="int")
    switches_before: int = record_field("report", axes=(), units="1", dtype="int")
    switches_after: int = record_field("report", axes=(), units="1", dtype="int")
    switch_intervals: int = record_field("report", axes=(), units="1", dtype="int", default=0)
    interval_track_bins: int = record_field("report", axes=(), units="1", dtype="int", default=0)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


def segment_rows(painting: Painting, input_ids: Sequence[str] | None = None) -> list[dict[str, str | int]]:
    """The released painting as segments (release 2.2): one row per sample, component, track and run of bins with
    one release (status, label and candidate rows), in sample, component, track and position order.

    Columns: SampleID, Component, Track (0/1, the painting's gauge: which homologue is track 0 carries no meaning),
    Start, End (bp, [Start, End)), Status (TrackStatus name), Label (the founder row of a SINGLETON_NAMED segment,
    else empty), Rows (candidate founder rows, ';'-joined; component-local). A SWITCH_INTERVAL segment is the bp
    interval where the track switches between its candidate rows; the segments before and after it name the two
    sides. Ineligible samples have no rows. ``input_ids`` (the run's input sample order) orders the samples; default
    the records' order."""
    names = {int(status.value): status.name for status in TrackStatus}
    order = list(range(len(painting.sample_ids)))
    if input_ids is not None:
        position = {str(name): index for index, name in enumerate(input_ids)}
        order.sort(key=lambda sample: position.get(str(painting.sample_ids[sample]), len(position)))
    rows: list[dict[str, str | int]] = []
    for sample in order:
        sample_id = painting.sample_ids[sample]
        for index, component in enumerate(painting.components):
            if not component.eligible[sample] or component.bins.n_bins == 0:
                continue
            edges = component.bins.edges
            n_bins = component.bins.n_bins
            for track in (0, 1):
                status = component.status[sample, track]
                labels = component.labels[sample, track]
                ids = component.member_ids[sample, track]
                changed = np.ones(n_bins, dtype=np.bool_)
                changed[1:] = (status[1:] != status[:-1]) | (labels[1:] != labels[:-1]) | (ids[1:] != ids[:-1])
                starts = np.flatnonzero(changed)
                stops = np.r_[starts[1:], n_bins]
                for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
                    label = int(labels[start])
                    rows.append(
                        dict(
                            SampleID=str(sample_id),
                            Component=index,
                            Track=track,
                            Start=int(edges[start]),
                            End=int(edges[stop]),
                            Status=names[int(status[start])],
                            Label="" if label < 0 else str(label),
                            Rows=";".join(map(str, component.member_sets[ids[start]])),
                        )
                    )
    return rows


@kernel(
    overloads=(
        (
            types.Array(types.int32, 3, "C", readonly=True),
            types.int64[::1],
            types.int64[::1],
            types.Array(types.boolean, 1, "C", readonly=True),
            types.int64[::1],
            types.int64[::1],
            types.Array(types.boolean, 2, "C", readonly=True),
            types.int64,
        ),
    ),
    cache=True,
    parallel=True,
)
def _carrier_observed_kernel(
    member_ids: NDArray[np.int32],
    set_offsets: NDArray[np.int64],
    set_rows: NDArray[np.int64],
    eligible: NDArray[np.bool_],
    site_bins: NDArray[np.int64],
    site_columns: NDArray[np.int64],
    observed: NDArray[np.bool_],
    n_rows: int,
) -> NDArray[np.bool_]:
    n_samples = member_ids.shape[0]
    n_sites = len(site_bins)
    result = np.zeros((n_rows, n_sites), dtype=np.bool_)
    for site in prange(n_sites):
        bin_index = site_bins[site]
        if bin_index < 0:
            continue
        column = site_columns[site]
        for sample in range(n_samples):
            if not eligible[sample] or not observed[sample, column]:
                continue
            for track in range(2):
                identifier = member_ids[sample, track, bin_index]
                for member in range(set_offsets[identifier], set_offsets[identifier + 1]):
                    row = set_rows[member]
                    if row < n_rows:
                        result[row, site] = True
    return result


def painting_to_payload(painting: Painting) -> dict[str, Any]:
    """Encode stored component fields plus separately readable rows and emissions.

    A writer needs every panel reconstructed. Large panel arrays never enter the
    payload; only its painted int32 row path does. No storage calls or mutation.
    """
    rows = []
    for component in painting.components:
        if component.panel is None:
            raise InvariantError("writing painting requires every derived panel")
        rows.append(component.panel.rows)
    return {
        "schema": painting.schema,
        "contig": painting.contig,
        "sample_ids": list(painting.sample_ids),
        "parameters": to_payload(painting.parameters),
        "label_continuity": [to_payload(item) for item in painting.label_continuity],
        "components": [to_payload(item) for item in painting.components],
        "rows": rows,
        "packed_emissions": [item.packed_emissions for item in painting.components],
    }


def painting_from_payload(
    payload: Mapping[str, Any],
    *,
    assembly: tuple[tuple[PreparedPanel, ...], tuple[FounderComponent, ...]] | None = None,
) -> Painting:
    """Decode any of four independent panel/emission routes without I/O.

    The adapter omits rows for the no-panel route and packed_emissions for the
    no-cache route. Supplying assembly requires rows. Component counts must agree.
    Arrays borrow immutable payload buffers for the lifetime of the decoded record.
    """
    require_schema(payload, 1)
    required = {"schema", "contig", "sample_ids", "parameters", "components", "label_continuity"}
    extra = payload.keys() - required - {"rows", "packed_emissions"}
    missing = required - payload.keys()
    if missing or extra:
        raise InvariantError(f"painting fields: missing {sorted(missing)}, unknown {sorted(extra)}")
    count = len(payload["components"])
    if len(payload["label_continuity"]) != count:
        raise InvariantError("painting label-continuity count disagrees with components")
    for name in ("rows", "packed_emissions"):
        if name in payload and len(payload[name]) != count:
            raise InvariantError(f"painting {name} count disagrees with components")
    panels: tuple[ComponentPanel | None, ...] = (None,) * count
    if assembly is not None:
        if "rows" not in payload:
            raise InvariantError("reconstructing painting panels requires rows")
        prepared, components = assembly
        if len(components) != count:
            raise InvariantError("assembly component count disagrees with painting")
        panels = materialise(
            tuple(replace(component, rows=rows) for component, rows in zip(components, payload["rows"], strict=True)),
            prepared,
        )
    packed = payload.get("packed_emissions", (None,) * count)
    return Painting(
        contig=payload["contig"],
        sample_ids=tuple(payload["sample_ids"]),
        parameters=from_payload(PaintingSettings, payload["parameters"]),
        label_continuity=tuple(from_payload(LabelContinuity, item) for item in payload["label_continuity"]),
        components=tuple(
            from_payload(ComponentPainting, item, derived={"panel": panel, "packed_emissions": cache})
            for item, panel, cache in zip(payload["components"], panels, packed, strict=True)
        ),
    )
