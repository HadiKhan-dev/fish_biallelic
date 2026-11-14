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

from ...contracts.pedigree import PreparedChromosome, TierReach
from ..shared.material import unit_count
from .blocks import component_length
from .parameters import MaterialParameters, PedigreeParameters

__all__ = ["Layout", "MaterialAxis", "Regime", "layout", "axis", "classify", "effective", "reach"]


@dataclass(frozen=True)
class Layout:
    """Small prepared summary of one informative contig: its components' selector switch probabilities."""

    contig: str
    selectors: tuple[np.ndarray, ...]


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
    return Layout(prepared.contig, tuple(c.selector_switch_probability for c in prepared.components))


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
