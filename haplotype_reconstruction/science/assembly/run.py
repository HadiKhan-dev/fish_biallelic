"""Assembly hierarchy stop rules, level records and context transitions.

The stage level loop applies these transitions to both final and context assemblies.
Stages own publication, resume, clocks and refinement scheduling.
Method: docs/methods/assembly.md#u02. Inputs are borrowed without mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from haplotype_reconstruction.contracts.assembly import JoinSettings, LevelRecord, ReleaseFamily
from haplotype_reconstruction.contracts.genetic_map import ChromosomeGeneticMap
from haplotype_reconstruction.contracts.panels import FounderComponent, LocalPanel, PreparedPanel
from haplotype_reconstruction.core.ties import TieSettings

from .alternatives import (
    COMPONENT_COUNT_NOT_REDUCED,
    CONTEXT_LEVELS_REACHED,
    ONLY_PHASE_BREAKS_REMAIN,
    SINGLE_COMPONENT,
)
from .evidence import AssemblyEvidence
from .parameters import AssemblyParameters


class Role(Enum):
    """The final hierarchy or a bounded local-selection context."""

    FINAL = "final"
    CONTEXT = "context"


@dataclass(frozen=True)
class AssemblyInputs:
    """Selected local panels, founder evidence and map on the same contig axes."""

    panels: tuple[LocalPanel, ...]
    evidence: AssemblyEvidence
    genetic_map: ChromosomeGeneticMap
    panels_identity: str | None = None


@dataclass(frozen=True)
class AssemblyResult:
    """Prepared panels and components, with all executed level decisions."""

    prepared: tuple[PreparedPanel, ...]
    components: tuple[FounderComponent, ...]
    levels: tuple[LevelRecord, ...]
    release_alternatives: tuple[ReleaseFamily, ...] = ()


def stop_reason(role: Role, levels: int | None, level: int, before: int, components, minimum_levels: int) -> str | None:
    """First matching stop reason, evaluated on joined components before refinement."""
    after = len(components)
    if after == 1:
        return SINGLE_COMPONENT
    if level >= 2 and after >= before:
        return COMPONENT_COUNT_NOT_REDUCED
    if role is Role.CONTEXT:
        assert levels is not None
        return CONTEXT_LEVELS_REACHED if level >= levels else None
    if level >= minimum_levels and all(c.break_after is not None for c in components[:-1]):
        return ONLY_PHASE_BREAKS_REMAIN
    return None


def initial_components(prepared) -> tuple[FounderComponent, ...]:
    """One identity row-path component per prepared panel, in genomic order."""
    return tuple(FounderComponent(i, i + 1, np.arange(p.rows, dtype=np.int32)[:, None]) for i, p in enumerate(prepared))


def next_context(level: int, refined, joined, previous):
    """Retain refined then unrefined L1 paths; later levels cannot reorder them."""
    return (*refined, *joined) if level == 1 else previous


def join_settings(
    parameters: AssemblyParameters, genetic_map: ChromosomeGeneticMap, level: int, ties: TieSettings
) -> JoinSettings:
    """Capture the owned parameter records and the batch context."""

    return JoinSettings(
        parameters.linker,
        parameters.join,
        parameters.count_rule,
        parameters.cost_scale,
        parameters.gap_horizon_scale,
        parameters.recombination_tolerance,
        level == 1,
        genetic_map,
        reverse=ties.reverse,
        tie_keep=ties.keep,
        census=ties.census,
    )


def level_record(
    level: int, before: int, result: Any, parameters: AssemblyParameters, role: Role, levels: int | None
) -> LevelRecord:
    """Record the hierarchy decision independently of subsequent refinement."""
    return LevelRecord(
        level=level,
        batch_size=parameters.batch_size,
        components_before=before,
        components_after=len(result.components),
        stop_reason=stop_reason(role, levels, level, before, result.components, parameters.minimum_levels),
        plan=result.plan,
        joins=result.joins,
        log_likelihoods=result.log_likelihoods,
        reused_joins=result.reused,
        boundary_joint=tuple(int(x) for x in result.partition.joint),
        shadows=result.shadows,
    )
