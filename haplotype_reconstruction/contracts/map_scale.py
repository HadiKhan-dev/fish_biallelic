"""Record the painting-based genome-wide recombination rate and the pedigree material decision it informs (4.19).

``MapScaleEstimate`` is the run record ``pedigree/map_scale``: the late (painting) estimate of the fallback rate, its
interval, gate and route. ``MaterialChoice`` is the pedigree's statement of which material regime it applied and
why. Method: docs/methods/pedigree.md#map-scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .roles import record_field

__all__ = ["MapScaleEstimate", "MaterialChoice", "ROUTES"]

# working_map: the regime reads the working map alone; both_maps: a call is released only when the working map's and
# the estimated map's regimes both release it; estimate: the estimated map's regime alone (adopt mode only).
ROUTES = ("working_map", "both_maps", "estimate")


@dataclass(frozen=True)
class MapScaleEstimate:
    """The late (painting) estimate of the working map's fallback rate and its route (DESIGN_RELEASE).

    Family rows (F) cover every relatedness family that gave frames on the map-free contigs, in family order.
    ``estimate_cm_per_mb`` is NaN when the map-free contigs gave no frame; the file-map ratio (``file_ratio``: painted
    crossover rate over the file map's, on frames inside the files' knots) is NaN without file contigs or frames.
    """

    mode: str = record_field("obs", dtype="str")
    working_rate_cm_per_mb: float = record_field("obs", units="cM/Mb", dtype="float64")
    map_free_contigs: tuple[str, ...] = record_field("obs", dtype="tuple[str, ...]")
    file_contigs: tuple[str, ...] = record_field("obs", dtype="tuple[str, ...]")
    estimate_cm_per_mb: float = record_field("call", units="cM/Mb", dtype="float64")
    interval_low_cm_per_mb: float = record_field("call", units="cM/Mb", dtype="float64")
    interval_high_cm_per_mb: float = record_field("call", units="cM/Mb", dtype="float64")
    at_grid_edge: bool = record_field("diagnostic", dtype="bool")
    expected_events: float = record_field("diagnostic", units="crossovers", dtype="float64")
    material_units: int = record_field("diagnostic", units="count", dtype="int64")
    frame_morgans: float = record_field("diagnostic", units="Morgans", dtype="float64")
    frames: int = record_field("diagnostic", units="count", dtype="int64")
    targets: int = record_field("diagnostic", units="count", dtype="int64")
    consistent: bool = record_field("diagnostic", dtype="bool")
    maximum_z: float = record_field("diagnostic", dtype="float64")
    critical_z: float = record_field("diagnostic", dtype="float64")
    dispersion: float = record_field("diagnostic", dtype="float64")
    adopted: bool = record_field("call", dtype="bool")
    gate_reason: str = record_field("diagnostic", dtype="str")
    usable: bool = record_field("call", dtype="bool")
    usable_reason: str = record_field("diagnostic", dtype="str")
    route: str = record_field("call", dtype="str")
    route_reason: str = record_field("diagnostic", dtype="str")
    family: NDArray[np.int64] = record_field("diagnostic", axes=("F",), dtype="int64")
    family_estimate_cm_per_mb: NDArray[np.float64] = record_field(
        "diagnostic", axes=("F",), units="cM/Mb", dtype="float64"
    )
    family_events: NDArray[np.float64] = record_field("diagnostic", axes=("F",), units="crossovers", dtype="float64")
    family_z: NDArray[np.float64] = record_field("diagnostic", axes=("F",), dtype="float64")
    family_status: tuple[str, ...] = record_field("diagnostic", axes=("F",), dtype="str")
    guard_pair_excluded: int = record_field("diagnostic", units="count", dtype="int64")
    guard_target_removed: int = record_field("diagnostic", units="count", dtype="int64")
    file_ratio: float = record_field("diagnostic", units="1", dtype="float64")
    file_ratio_low: float = record_field("diagnostic", units="1", dtype="float64")
    file_ratio_high: float = record_field("diagnostic", units="1", dtype="float64")
    file_ratio_events: float = record_field("diagnostic", units="crossovers", dtype="float64")
    warning: str = record_field("report", dtype="str")

    @property
    def estimate_interval(self) -> tuple[float, float]:
        return (self.interval_low_cm_per_mb, self.interval_high_cm_per_mb)


@dataclass(frozen=True)
class MaterialChoice:
    """The pedigree's material decision under the run's maps (DESIGN_RELEASE section 6).

    working_morgans: U of the working map (the configured map, unless rate_estimate = adopt let the relatedness
    estimate replace its fallback rate); estimate_morgans / estimate_low_morgans / estimate_high_morgans: U with the
    fallback stretches at the estimate and at its interval ends (NaN without a usable estimate).
    working_regime: the regime under the working map alone; estimate_regime: the regime under the estimated map ("" on
    the working_map route); applied_regime: the regime whose full rule produced the product (the lower of the two
    under both_maps); veto_regime: the other regime under both_maps, whose released calls a call must also be in, or
    "".
    withheld_calls / withheld_edges / withheld_tier_a: Tier B calls, Tier B edges and Tier A calls that the working map
    alone releases and the product does not release identically, whichever rule withheld them (the applied regime's
    own rule or the veto); added_calls / added_edges / added_tier_a: those the product releases and the working map
    alone does not (0 except on the estimate route). A call released differently counts in both. -1: not computed,
    when the product is not assessable under the applied regime but the working map alone is assessable (its chain
    does not run). vetoed_calls / vetoed_edges / vetoed_tier_a: the part withheld by the veto pass itself (status
    tier_b_material_veto / tier_a_material_veto). straddles: the regime cut-offs (Morgans) inside
    [estimate_low_morgans, estimate_high_morgans]. statement: the decision and the counts in words (the run log
    prints it)."""

    route: str = record_field("call", dtype="str")
    mode: str = record_field("obs", dtype="str")
    working_morgans: float = record_field("obs", units="Morgans", dtype="float64")
    estimate_morgans: float = record_field("obs", units="Morgans", dtype="float64")
    estimate_low_morgans: float = record_field("obs", units="Morgans", dtype="float64")
    estimate_high_morgans: float = record_field("obs", units="Morgans", dtype="float64")
    working_regime: str = record_field("report", dtype="str")
    estimate_regime: str = record_field("report", dtype="str")
    applied_regime: str = record_field("call", dtype="str")
    veto_regime: str = record_field("call", dtype="str")
    withheld_calls: int = record_field("report", units="count", dtype="int64", default=0)
    withheld_edges: int = record_field("report", units="count", dtype="int64", default=0)
    withheld_tier_a: int = record_field("report", units="count", dtype="int64", default=0)
    added_calls: int = record_field("report", units="count", dtype="int64", default=0)
    added_edges: int = record_field("report", units="count", dtype="int64", default=0)
    added_tier_a: int = record_field("report", units="count", dtype="int64", default=0)
    vetoed_calls: int = record_field("report", units="count", dtype="int64", default=0)
    vetoed_edges: int = record_field("report", units="count", dtype="int64", default=0)
    vetoed_tier_a: int = record_field("report", units="count", dtype="int64", default=0)
    straddles: tuple[float, ...] = record_field("report", units="Morgans", dtype="tuple[float, ...]", default=())
    statement: str = record_field("report", dtype="str", default="")
