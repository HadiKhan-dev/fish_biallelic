"""Per-chromosome screen (M0 + all-pairs M1) and score (M2 for the fixed panel, exposure, junctions).

The screen and the score of a chromosome share each component's exact source pair masses,
projection and child terms: a chromosome worker keeps them resident across the genome-wide panel barrier (they
are rebuilt from the prepared record only after an interruption). The score phase reuses the
screen's M0/M1 values, so both phases agree exactly. A chromosome's evidence is the sum over its
components (in component order).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import exposure, forward, junctions, projection, sources
from .config import PedigreeConfig
from .eligibility import ParentEligibility
from .records import (ChromosomeScores, ChromosomeScreen, ComponentScreen, PreparedChromosome,
                      PreparedComponent)


@dataclass
class ResidentComponent:
    """Worker-resident scoring state of one prepared component (not a record)."""

    pair_masses: np.ndarray          # float64 [N, B, S (S + 1) / 2] unordered source pair masses
    projection: projection.Projection
    terms: forward.ChildTerms


def resident(component: PreparedComponent, config: PedigreeConfig,
             claim_threads=lambda: None) -> ResidentComponent:
    """Exact source pair masses, projection and child terms of one component."""
    claim_threads()
    marginals = sources.marginals(component.factors, component.transition)
    claim_threads()
    # Exposure needs only the unordered pair masses (half the size of the ordered marginals); the
    # projection then consumes the marginals in place.
    masses = exposure.pair_masses(marginals)
    claim_threads()
    projected = projection.project(
        component, marginals, source_gap_support_sites=config.scoring.source_gap_support_sites,
        bridge_tolerance=config.scoring.bridge_tolerance,
        bridge_max_iterations=config.scoring.bridge_max_iterations)
    del marginals
    transition = component.transition
    terms = forward.ChildTerms(projected, component.genotype_likelihoods, component.observed,
                               component.information_exponent,
                               component.selector_switch_probability, transition.same,
                               transition.one_change, transition.two_changes,
                               config.scoring.mismatch_probability, config.scoring.null_prior_samples)
    return ResidentComponent(masses, projected, terms)


def screen_chromosome(prepared: PreparedChromosome, eligibility: ParentEligibility,
                      config: PedigreeConfig, claim_threads=lambda: None,
                      ) -> tuple[ChromosomeScreen, list[ResidentComponent]]:
    """M0 and all-pairs M1 of every component, plus the resident state for the score phase."""
    screens, states = [], []
    for component in prepared.components:
        state = resident(component, config, claim_threads)
        zero, one = forward.screen(state.projection, state.terms, eligibility.children,
                                   eligibility.parents, claim_threads)
        screens.append(ComponentScreen(zero, one))
        states.append(state)
    return ChromosomeScreen(prepared.contig, tuple(screens), prepared.informative_markers), states


def score_chromosome(prepared: PreparedChromosome, screen: ChromosomeScreen,
                     states: list[ResidentComponent] | None, trios: np.ndarray,
                     eligibility: ParentEligibility, config: PedigreeConfig,
                     claim_threads=lambda: None, *, one_values=None) -> ChromosomeScores:
    """M2, exposure and junction evidence for the panel ``trios``; M0/M1 from ``screen``."""
    required = eligibility.parents | eligibility.parents.T
    np.fill_diagonal(required, True)
    parts = []
    schedules, thetas = [], []
    for index, (component, component_screen) in enumerate(zip(prepared.components,
                                                              screen.components)):
        state = states[index] if states is not None else resident(component, config, claim_threads)
        two = forward.score_pairs(state.projection, state.terms, trios, component_screen.zero,
                                  component_screen.one, claim_threads)
        claim_threads()
        counts, callable_bins = junctions.junction_counts(component.state_classes,
                                                          component.named_alleles,
                                                          component.bin_sizes)
        claim_threads()
        schedules.append(junctions.schedules(component.state_classes, component.named_alleles,
                                             component.bin_sizes))
        thetas.append(np.ascontiguousarray(component.selector_switch_probability, dtype=np.float64))
        structure = exposure.expected_structure(state.pair_masses, component.anchored_states,
                                                required, trios, component.factors.available,
                                                claim_threads)
        parts.append((component_screen.zero, component_screen.one, two, *structure, counts,
                      callable_bins))
        if states is not None:
            states[index] = None    # release the component's resident arrays
        del state

    def total(position):
        return np.sum(np.stack([part[position] for part in parts]), axis=0)

    # Linked junction transmission (release 2.16) for every dyad the decision orients (both directions of each
    # eligible edge): the child's chromosome-wide callable fraction thins the inherited junctions (as before).
    own = total(7).astype(np.int64)
    total_bins = float(sum(component.n_bins for component in prepared.components))
    detection = np.minimum(1.0, total(8) / (2.0 * total_bins)) if total_bins > 0 else np.zeros(len(own))
    oriented = required.copy()
    np.fill_diagonal(oriented, False)
    dyads = np.argwhere(oriented).astype(np.int64)
    junction_log_cdf = np.zeros(len(dyads))
    if len(dyads):
        claim_threads()
        junction_log_cdf = junctions.linked_log_cdf(dyads, schedules, thetas, own, detection, own)

    return ChromosomeScores(
        contig=prepared.contig, zero=total(0),
        one=total(1) if one_values is None else one_values,
        two=total(2), edge_matched=total(3),
        edge_exposed=total(4), pair_explained=total(5), pair_exposed=total(6),
        total_bins=float(sum(component.n_bins for component in prepared.components)),
        junctions=total(7), callable_bins=total(8),
        informative_markers=prepared.informative_markers,
        component_count=prepared.component_count,
        scored_component_count=len(prepared.components),
        junction_dyads=dyads, junction_log_cdf=junction_log_cdf)


def score_chromosome_selective(prepared: PreparedChromosome, one: np.ndarray,
                               trios: np.ndarray, eligibility: ParentEligibility,
                               config: PedigreeConfig, claim_threads=lambda: None) -> ChromosomeScores:
    """M2/exposure with an already fixed chromosome M1 profile.

    M2 reduces to raw component M1 when exactly one parent is unavailable.
    Those dyads are scored exactly here. The remaining partial component screen
    cells are never consumed, checkpointed or returned as complete raw scores.
    """
    from . import selective_m1

    screens, states = [], []
    for component in prepared.components:
        state = resident(component, config, claim_threads)
        projected, terms = state.projection, state.terms
        zero = selective_m1.zero(projected, terms, eligibility.children, claim_threads)
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
                values = selective_m1.exact_pairs(projected, terms, pairs, zero, first,
                                                   products, powered, claim_threads)
                partial[pairs[:, 0], pairs[:, 1]] = values
                del products, powered
        screens.append(ComponentScreen(zero, partial))
        states.append(state)
    screen = ChromosomeScreen(prepared.contig, tuple(screens), prepared.informative_markers)
    return score_chromosome(prepared, screen, states, trios, eligibility, config,
                            claim_threads, one_values=one)
