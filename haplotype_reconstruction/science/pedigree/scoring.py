"""Per-chromosome M2 scoring, exposure and junction evidence for the fixed panel.

Scoring reconstructs resident projections and M0 after the selective screen. Its
component scratch scores contain raw M1 only for fallback dyads where one parent
is unavailable; the published M1 profile is supplied separately. Chromosome
scores sum components in their prepared order.

Method: docs/methods/pedigree.md#ragged-quadratic-scoring-of-m0-m1-and-m2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...contracts.pedigree import ChromosomeScores, PreparedChromosome
from . import exposure, forward, junctions
from .eligibility import ParentEligibility
from .parameters import PedigreeParameters
from .resident import ResidentComponent, resident


@dataclass(frozen=True)
class ComponentScreen:
    """M0 and partial fallback raw M1 composite log likelihoods of one component.

    zero: float64 [N], -inf for samples that are not eligible children.
    one:  float64 [N, N] [child, parent], -inf except for the scored fallback dyads.
    """

    zero: np.ndarray
    one: np.ndarray


@dataclass(frozen=True)
class ChromosomeScreen:
    """Transient component scratch scores for the fixed-panel scorer, in prepared order."""

    components: tuple[ComponentScreen, ...]


def interval_morgans(component, interval_cap: float) -> np.ndarray:
    """float64 [B-1]: the component's measured map length between adjacent bins (PreparedComponent.measured_morgans),
    each interval at most ``interval_cap`` (one interval is at most one map unit, however long a gap)."""
    return np.ascontiguousarray(np.minimum(np.asarray(component.measured_morgans, dtype=np.float64), interval_cap))


def scored_morgans(prepared: PreparedChromosome, interval_cap: float) -> float:
    """Measured map length (Morgans) of the scored components: the capped measured lengths between adjacent bins of
    each component (``interval_morgans``; gaps between components add nothing, and 0 without a measured map)."""
    return float(sum(np.sum(interval_morgans(component, interval_cap)) for component in prepared.components))


def score_chromosome(
    prepared: PreparedChromosome,
    screen: ChromosomeScreen,
    states: list[ResidentComponent | None],
    trios: np.ndarray,
    eligibility: ParentEligibility,
    config: PedigreeParameters,
    claim_threads=lambda: None,
    *,
    one_values: np.ndarray,
    reverse: bool | None = None,
) -> ChromosomeScores:
    """M2, exposure and junction evidence for ``trios``; M0 and fallback M1 from ``screen``.

    ``states`` supplies resident components, consumed in order. ``one_values`` supplies the fixed
    chromosome M1 profile for publication; fallback scratch cells are never published as that profile."""
    reverse = config.reverse_conventions if reverse is None else reverse
    required = eligibility.parents | eligibility.parents.T
    np.fill_diagonal(required, True)
    parts = []
    schedules, thetas = [], []
    for index, (component, component_screen) in enumerate(zip(prepared.components, screen.components, strict=True)):
        state = states[index]
        assert state is not None  # each supplied state is read once, before its slot is released below
        two = forward.component_m2(
            state.projection, state.terms, trios, component_screen.zero, component_screen.one, claim_threads
        )
        claim_threads()
        counts, callable_bins = junctions.junction_counts(
            component.state_classes, component.named_alleles, component.bin_sizes
        )
        claim_threads()
        schedules.append(
            junctions.schedules(component.state_classes, component.named_alleles, component.bin_sizes, reverse=reverse)
        )
        thetas.append(np.ascontiguousarray(component.selector_switch_probability, dtype=np.float64))
        assert state.pair_masses is not None  # states scored here are resident with their pair masses (the default)
        (matched, exposed, explained, pair_exposed, exposed_morgans, pair_morgans) = exposure.expected_structure(
            state.pair_masses,
            component.anchored_states,
            required,
            trios,
            component.factors.available,
            claim_threads,
            interval_morgans(component, config.exposure.unit_morgans),
        )
        parts.append(
            (
                component_screen.zero,
                component_screen.one,
                two,
                matched,
                exposed,
                explained,
                pair_exposed,
                counts,
                callable_bins,
                exposed_morgans,
                pair_morgans,
            )
        )
        states[index] = None  # release the component's resident arrays
        del state

    def total(position):
        return np.sum(np.stack([part[position] for part in parts]), axis=0)

    # Linked junction transmission for every dyad the decision orients (both directions of each
    # eligible edge): the child's chromosome-wide callable fraction thins the inherited junctions.
    own = total(7).astype(np.int64)
    total_bins = float(sum(component.n_bins for component in prepared.components))
    detection = np.minimum(1.0, total(8) / (2.0 * total_bins)) if total_bins > 0 else np.zeros(len(own))
    oriented = required.copy()
    np.fill_diagonal(oriented, False)
    dyads = np.ascontiguousarray(np.argwhere(oriented), dtype=np.int64)
    junction_log_cdf = np.zeros(len(dyads))
    if len(dyads):
        claim_threads()
        junction_log_cdf = junctions.linked_log_cdf(dyads, schedules, thetas, own, detection, own)

    return ChromosomeScores(
        contig=prepared.contig,
        zero=total(0),
        one_floored=one_values,
        two=total(2),
        edge_matched=total(3),
        edge_exposed=total(4),
        pair_explained=total(5),
        pair_exposed=total(6),
        total_bins=float(sum(component.n_bins for component in prepared.components)),
        junctions=total(7),
        callable_bins=total(8),
        informative_markers=prepared.informative_markers,
        component_count=prepared.component_count,
        scored_component_count=len(prepared.components),
        junction_dyads=dyads,
        junction_log_cdf=junction_log_cdf,
        scored_morgans=scored_morgans(prepared, config.exposure.unit_morgans),
        edge_exposed_morgans=total(9),
        pair_exposed_morgans=total(10),
    )


def score_chromosome_selective(
    prepared: PreparedChromosome,
    one: np.ndarray,
    trios: np.ndarray,
    eligibility: ParentEligibility,
    config: PedigreeParameters,
    claim_threads=lambda: None,
    *,
    reverse: bool | None = None,
) -> ChromosomeScores:
    """M2/exposure with an already fixed chromosome M1 profile.

    M2 reduces to raw component M1 when exactly one parent is unavailable.
    Those dyads are scored exactly here. The remaining partial component screen
    cells are never consumed, checkpointed or returned as complete raw scores.
    """

    states: list[ResidentComponent | None]
    screens, states = [], []
    for component in prepared.components:
        state = resident(component, config, claim_threads, reverse=reverse)
        projected, terms = state.projection, state.terms
        zero = forward.component_m0(projected, terms, eligibility.children, claim_threads)
        partial = np.full((len(zero), len(zero)), -np.inf)
        first_available = projected.available[trios[:, 1]]
        second_available = projected.available[trios[:, 2]]
        fallback = trios[first_available ^ second_available]
        if len(fallback):
            parents = np.where(projected.available[fallback[:, 1]], fallback[:, 1], fallback[:, 2])
            dyads = np.unique(np.column_stack((fallback[:, 0], parents)), axis=0)
            for first in range(0, len(zero), terms.batch):
                stop = min(len(zero), first + terms.batch)
                pairs = np.ascontiguousarray(dyads[(dyads[:, 0] >= first) & (dyads[:, 0] < stop)])
                if not len(pairs):
                    continue
                claim_threads()
                products, powered = terms.common(projected, first, stop)
                values = forward.component_m1(projected, terms, pairs, zero, first, products, powered, claim_threads)
                partial[pairs[:, 0], pairs[:, 1]] = values
                del products, powered
        screens.append(ComponentScreen(zero, partial))
        states.append(state)
    screen = ChromosomeScreen(tuple(screens))
    return score_chromosome(
        prepared, screen, states, trios, eligibility, config, claim_threads, one_values=one, reverse=reverse
    )
