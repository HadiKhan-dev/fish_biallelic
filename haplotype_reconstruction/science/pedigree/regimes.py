"""The run's material and its regime, from the prepared layout alone, before score tasks (DESIGN_417 rev 3).

One function serves every entry point (I5): the screen barrier, the bound-input reader, the decision task and
the stacking all read U, the offsets and the regime from ``axis`` and ``classify``. U is the working Morgans of
the informative components (configured contig order, components in map order, per-bin Haldane coordinates of the
selector switch probabilities); the offsets are its running sums, so every contig's score task and the stacking
see bitwise the same coordinates. No fit, per-child winner, successful resample or count of chromosomes,
contigs or components enters the regime: U < assessable_morgans is not assessable; below standard_morgans the
low-material band (Tier B only, MaterialParameters.low_*); from it the released thresholds, with Tier A from
tier_a_morgans.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from ...contracts.map_scale import MapScaleEstimate, MaterialChoice
from ...contracts.pedigree import PreparedChromosome, TierReach
from ...core.errors import InvariantError
from ..shared.material import unit_count
from .blocks import component_length
from .parameters import SELECTOR_SWITCH_FLOOR, MaterialParameters, PedigreeParameters

__all__ = ["Layout", "MaterialAxis", "Regime", "layout", "axis", "classify", "effective", "reach", "material_choice",
           "rescaled", "reference_regime"]

_RANK = {"not_assessable": 0, "low_material": 1, "standard": 2}


@dataclass(frozen=True)
class Layout:
    """Small prepared summary of one informative contig: its components' selector switch probabilities and the
    fallback-rate length (bp) of each boundary (PreparedComponent.selector_fallback_bp; empty for fixtures)."""

    contig: str
    selectors: tuple[np.ndarray, ...]
    fallback: tuple[np.ndarray | None, ...] = ()


@dataclass(frozen=True)
class MaterialAxis:
    """The genome axis of the informative contigs (configured order).

    offsets: genome coordinate of each contig's first scored bin; ends: the coordinate after its last component;
    morgans: U (= the last end); units: the nominal unit count of U (MaterialParameters tail rule)."""

    contigs: tuple[str, ...]
    offsets: tuple[float, ...]
    ends: tuple[float, ...]
    morgans: float
    units: int


@dataclass(frozen=True)
class Regime:
    """``name``: "not_assessable", "low_material" or "standard"; ``tier_a``: Tier A is assessed."""

    name: str
    tier_a: bool
    reach: TierReach


def layout(prepared: PreparedChromosome) -> Layout:
    return Layout(prepared.contig, tuple(c.selector_switch_probability for c in prepared.components),
                  tuple(c.selector_fallback_bp for c in prepared.components))


def rescaled(layouts: Sequence[Layout], rate_change: float) -> tuple[Layout, ...]:
    """The layouts under a working map whose fallback rate differs by ``rate_change`` cM/Mb: each boundary's working
    Morgans d (the Haldane inverse of its switch probability) gains rate_change 1e-8 x its fallback length, and the
    switch probability is recomputed as preparation computes it (Haldane, kept within [floor, 1/2]). Distances inside
    a map file's knots are unchanged."""
    out = []
    for value in layouts:
        if len(value.fallback) != len(value.selectors) or any(f is None for f in value.fallback):
            raise InvariantError(f"{value.contig}: the prepared layout lacks its fallback lengths")
        selectors = []
        for theta, fallback in zip(value.selectors, value.fallback, strict=True):
            theta = np.asarray(theta, dtype=np.float64)
            with np.errstate(divide="ignore"):
                d = -0.5 * np.log1p(-2.0 * theta)
            d = np.maximum(d + rate_change * 1e-8 * np.asarray(fallback, dtype=np.float64), 0.0)
            selectors.append(np.clip(0.5 * (1.0 - np.exp(-2.0 * d)), SELECTOR_SWITCH_FLOOR, 0.5))
        out.append(Layout(value.contig, tuple(selectors), value.fallback))
    return tuple(out)


def contig_end(selectors: Sequence[np.ndarray], offset: float, material: MaterialParameters) -> float:
    """The running genome coordinate after a contig's components, starting from its ``offset``."""
    for selector in selectors:
        offset += component_length(selector, material)
    return offset


def axis(layouts: Sequence[Layout], material: MaterialParameters) -> MaterialAxis:
    """Offsets and U by one running sum over the informative contigs' components in configured order."""
    offsets, ends = [], []
    offset = 0.0
    for value in layouts:
        offsets.append(offset)
        offset = contig_end(value.selectors, offset, material)
        ends.append(offset)
    return MaterialAxis(
        tuple(v.contig for v in layouts), tuple(offsets), tuple(ends), float(offset),
        unit_count(float(offset), material.unit_morgans, material.tail_morgans) if layouts else 0,
    )


def reach(morgans: float, units: int, material: MaterialParameters) -> TierReach:
    """The material statement of a run of U = ``morgans`` working Morgans in ``units`` units."""
    if morgans < material.assessable_morgans:
        name, tier_b, tier_a = "not_assessable", False, False
        verdict = (f"not assessable (requires at least {material.assessable_morgans:g} Morgans); "
                   "Tier A and Tier B not assessed")
    elif morgans < material.standard_morgans:
        name, tier_b, tier_a = "low_material", True, False
        verdict = (f"Tier B assessable with low-material resampling (below {material.standard_morgans:g} Morgans); "
                   f"Tier A not assessed (requires {material.tier_a_morgans:g} Morgans)")
    else:
        name, tier_b, tier_a = "standard", True, morgans >= material.tier_a_morgans
        verdict = "Tier B assessable; " + (
            "Tier A assessable" if tier_a else f"Tier A not assessed (requires {material.tier_a_morgans:g} Morgans)")
    statement = (f"{morgans:.2f} working Morgans of informative material in {units} units of "
                 f"{material.unit_morgans:g} Morgan: {verdict}")
    return TierReach(float(morgans), int(units), name, tier_a, tier_b, material.unit_morgans,
                     material.assessable_morgans, material.standard_morgans, material.tier_a_morgans, statement)


def classify(morgans: float, units: int, parameters: PedigreeParameters) -> Regime:
    """The regime of U = ``morgans`` alone; ``units`` (the resampled units) only enters the statement."""
    value = reach(morgans, units, parameters.material)
    return Regime(value.regime, value.tier_a_reachable, value)


def classify_layouts(layouts: Sequence[Layout], parameters: PedigreeParameters) -> tuple[Regime, MaterialAxis]:
    """The regime and axis of the prepared informative contigs (nominal unit count in the statement)."""
    value = axis(layouts, parameters.material)
    return classify(value.morgans, value.units, parameters), value


def effective(parameters: PedigreeParameters, regime: str) -> PedigreeParameters:
    """The release rule of a regime: the low-material band's minimums and Tier B thresholds, else as released."""
    if regime != "low_material":
        return parameters
    material = parameters.material
    return replace(parameters,
        exposure=replace(parameters.exposure, minimum_exposed_units=1.0),
        release=replace(parameters.release, minimum_informative_units=1.0,
            tier_b=replace(parameters.release.tier_b,
                configuration_bootstrap=material.low_bootstrap, parent_bootstrap=material.low_bootstrap,
                lobo_fraction=material.low_leave_one_out)))


def _key(regime: Regime) -> tuple[int, bool]:
    return (_RANK[regime.name], bool(regime.tier_a))


def material_choice(
    layouts: Sequence[Layout], parameters: PedigreeParameters, estimate: MapScaleEstimate | None
) -> tuple[MaterialChoice, Regime, float, MaterialAxis]:
    """The regime the pedigree applies (DESIGN_RELEASE section 6) and the Morgans it reads.

    U of the working map decides alone on the working_map route. With an estimate, U is recomputed with the fallback
    stretches at the estimate (and at its interval ends: the straddle diagnostic, reported on every route). On
    both_maps the lower regime of the two maps is applied and the higher one, when it differs in name and both are
    assessable, vetoes: a call is released only when both release it. On estimate the estimated map's regime alone is
    applied. Returns (choice, applied regime, the Morgans it reads, the working axis); units, offsets and every other
    material quantity keep the working axis."""
    material = parameters.material
    working_axis = axis(layouts, material)
    working = classify(working_axis.morgans, working_axis.units, parameters)
    nan = float("nan")
    route = "working_map" if estimate is None else estimate.route
    mode = "ignore" if estimate is None else estimate.mode
    finite = estimate is not None and np.isfinite(estimate.estimate_cm_per_mb)
    est_u = low_u = high_u = nan
    straddles: tuple[float, ...] = ()
    estimated = None
    if finite:  # with no informative layout both maps hold 0 Morgans (not assessable)
        rate = estimate.working_rate_cm_per_mb
        est_u = axis(rescaled(layouts, estimate.estimate_cm_per_mb - rate), material).morgans
        estimated = classify(est_u, unit_count(est_u, material.unit_morgans, material.tail_morgans), parameters)
        if np.isfinite(estimate.interval_low_cm_per_mb):
            low_u = axis(rescaled(layouts, estimate.interval_low_cm_per_mb - rate), material).morgans
            high_u = axis(rescaled(layouts, estimate.interval_high_cm_per_mb - rate), material).morgans
            cuts = sorted({material.assessable_morgans, material.standard_morgans, material.tier_a_morgans})
            straddles = tuple(float(c) for c in cuts if low_u < c < high_u)
    applied, morgans, veto = working, working_axis.morgans, ""
    if route == "estimate":
        if estimated is None:
            raise InvariantError("the estimate route needs a finite estimate")
        applied, morgans = estimated, est_u
    elif route == "both_maps":
        if estimated is None:
            raise InvariantError("the both-maps route needs a finite estimate")
        if _key(estimated) < _key(working):
            applied, morgans = estimated, est_u
        higher = working if applied is estimated else estimated
        if applied.name != "not_assessable" and higher.name != applied.name:
            veto = higher.name
    applied = classify(morgans, working_axis.units, parameters)
    words = {
        "working_map": "the working map's regime",
        "both_maps": "only what both the working map's and the estimated map's regimes release",
        "estimate": "the estimated map's regime",
    }[route]
    statement = (
        f"material: working map {working_axis.morgans:.2f} Morgans ({working.name})"
        + (f", estimated map {est_u:.2f} Morgans ({estimated.name}; interval {low_u:.2f}-{high_u:.2f})"
           if estimated is not None else "")
        + f"; the pedigree releases {words}: {applied.name}" + (f" with the {veto} rule's veto" if veto else "")
        + ("" if not straddles else
           "; WARNING: the estimate's interval brackets the cut-off" + ("s " if len(straddles) > 1 else " ")
           + ", ".join(f"{c:g}" for c in straddles) + " Morgans")
    )
    choice = MaterialChoice(
        route=route,
        mode=mode,
        working_morgans=float(working_axis.morgans),
        estimate_morgans=float(est_u),
        estimate_low_morgans=float(low_u),
        estimate_high_morgans=float(high_u),
        working_regime=working.name,
        estimate_regime="" if estimated is None else estimated.name,
        applied_regime=applied.name,
        veto_regime=veto,
        straddles=straddles,
        statement=statement,
    )
    return choice, applied, float(morgans), working_axis


def reference_regime(choice: MaterialChoice) -> str:
    """The working map's regime when its chain must run only to count what the product withholds or adds relative to
    the working map alone (``MaterialChoice.withheld_calls``), else "": on the estimate route when the two regimes
    differ and both are assessable. Under both_maps the working map's release is already computed (the applied chain
    before the veto, or the veto chain itself); when the applied regime is not assessable no scoring runs, so the
    working map's release is not computed (the counts are -1)."""
    applied, working = choice.applied_regime, choice.working_regime
    if choice.veto_regime or applied == working or "not_assessable" in (applied, working):
        return ""
    return working
