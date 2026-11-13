"""Choose one pedigree regime from prepared layout alone, before score tasks.

No fit, per-child winner or successful resample enters this decision (A1).
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from collections.abc import Sequence
import numpy as np
from ...contracts.pedigree import PreparedChromosome, TierReach
from ...core.random import bootstrap_multiplicities
from . import blocks, resampling, selection
from .parameters import (PedigreeParameters, LOW_MATERIAL_MINIMUM_BLOCKS,
                         LOW_MATERIAL_BOOTSTRAP, LOW_MATERIAL_LEAVE_ONE_OUT)

@dataclass(frozen=True)
class Layout:
    """Small prepared summary; selectors retain component order, without block computation."""
    contig: str
    scored_morgans: float
    selectors: tuple[np.ndarray, ...]

@dataclass(frozen=True)
class Regime:
    name: str
    reach: TierReach
    block_count: int | None


def layout(prepared: PreparedChromosome, parameters: PedigreeParameters) -> Layout:
    return Layout(prepared.contig,
                  float(sum(np.minimum(c.measured_morgans, parameters.exposure.unit_morgans).sum()
                            for c in prepared.components)),
                  tuple(c.selector_switch_probability for c in prepared.components))


def effective(parameters: PedigreeParameters, regime: str) -> PedigreeParameters:
    if regime != "low_material":
        return parameters
    return replace(parameters,
        exposure=replace(parameters.exposure, minimum_exposed_units=1.0),
        release=replace(parameters.release, minimum_informative_units=1.0,
            tier_b=replace(parameters.release.tier_b,
                configuration_bootstrap=LOW_MATERIAL_BOOTSTRAP, parent_bootstrap=LOW_MATERIAL_BOOTSTRAP,
                loco_fraction=LOW_MATERIAL_LEAVE_ONE_OUT)))


def classify(prepared: Sequence[Layout], parameters: PedigreeParameters) -> Regime:
    """A1: released count eligibility AND seeded chromosome reach, else occupied blocks."""
    count = len(prepared)
    minimum = int(np.ceil(parameters.exposure.minimum_exposed_units))
    if count >= minimum:
        units = selection.map_units(np.asarray([c.scored_morgans for c in prepared]), parameters.exposure)
        draws = bootstrap_multiplicities(count, parameters.resampling.bootstrap_replicates,
                                        parameters.resampling.bootstrap_seed)
        reach = resampling.reach(draws, units, parameters)
        if reach.tier_b_reachable:
            return Regime("standard", reach, None)
    count = sum(len(blocks.partition(theta)[1]) for chromosome in prepared for theta in chromosome.selectors)
    return for_blocks(count, parameters)


def for_blocks(count: int, parameters: PedigreeParameters) -> Regime:
    """Reach and the four-block mask for an already classified low-material axis."""
    low = effective(parameters, "low_material")
    draws = bootstrap_multiplicities(count, low.resampling.bootstrap_replicates,
                                    low.resampling.bootstrap_seed) if count else np.empty((0, 0))
    reach = resampling.reach(draws, np.ones(count), low)
    assessable = count >= LOW_MATERIAL_MINIMUM_BLOCKS
    statement = (f"{count} informative blocks: Tier B assessable with low-material resampling; Tier A not assessed"
                 if assessable else f"{count} informative blocks: not assessable (requires at least {LOW_MATERIAL_MINIMUM_BLOCKS}); Tier A and Tier B not assessed")
    reach = replace(reach, tier_a_reachable=False, tier_b_reachable=assessable,
                    minimum_contigs_tier_a=0, minimum_contigs_tier_b=LOW_MATERIAL_MINIMUM_BLOCKS,
                    statement=statement, unit_morgans=float("nan"))
    return Regime("low_material" if assessable else "not_assessable", reach, count)
