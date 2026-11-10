"""Describe estimator trajectories, released intervals and explicit tie censuses.

These records explain scientific decisions and never affect search or release. Interval runs split at
member-id changes. No I/O or clocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...contracts.painting import ComponentPainting, ComponentShape, Painting, PaintingSummary, TrackStatus
from ...contracts.roles import record_field, schema_field
from ...core.ties import TieCensus

__all__ = ["ModelTrajectory", "PaintingDiagnostics", "interval_counts", "summarise"]


@dataclass(frozen=True)
class ModelTrajectory:
    """Unrounded rate, entry, exit and pre-update log likelihood after each EM update."""

    component: int = record_field("diagnostic", dtype="int64", units="1")
    updates: tuple[tuple[float, float, float, float], ...] = record_field(
        "diagnostic", axes=("updates", "4"), dtype="float64", units="rate, probability, probability, nats"
    )


@dataclass(frozen=True)
class PaintingDiagnostics:
    """Optional model/census receipts; computation and scientific identity ignore them."""

    schema: int = schema_field(1)
    trajectories: tuple[ModelTrajectory, ...] = record_field("diagnostic", dtype="tuple[ModelTrajectory]")
    census: tuple[TieCensus, ...] = record_field("diagnostic", dtype="tuple[TieCensus]")


def interval_counts(component: ComponentPainting) -> tuple[int, int]:
    """Count released interval runs and track bins, splitting a run when members change."""
    flagged = np.asarray(component.status) == int(TrackStatus.SWITCH_INTERVAL)
    if flagged.size == 0:
        return 0, 0
    ids = np.asarray(component.member_ids)
    starts = flagged.copy()
    starts[..., 1:] &= ~flagged[..., :-1] | (ids[..., 1:] != ids[..., :-1])
    return int(np.sum(starts)), int(np.sum(flagged))


def summarise(painting: Painting) -> PaintingSummary:
    """The component sizes, eligibility, moves and released interval counts."""
    outcomes = painting.label_continuity
    counts = [interval_counts(component) for component in painting.components]
    return PaintingSummary(
        contig=painting.contig,
        samples=len(painting.sample_ids),
        components=tuple(
            ComponentShape(
                founders=sum(map(len, component.states.classes)),
                classes=component.states.n_classes,
                states=component.states.n_states,
                sites=component.states.n_sites,
                bins=component.bins.n_bins,
                packed_emissions=component.packed_emissions is not None,
            )
            for component in painting.components
        ),
        eligible_pairs=sum(int(np.sum(component.eligible)) for component in painting.components),
        relabel_moves=sum(len(outcome.moves) for outcome in outcomes if outcome.accepted),
        switches_before=sum(outcome.switches_before or 0 for outcome in outcomes),
        switches_after=sum(outcome.switches_after or 0 for outcome in outcomes),
        switch_intervals=sum(count[0] for count in counts),
        interval_track_bins=sum(count[1] for count in counts),
    )
