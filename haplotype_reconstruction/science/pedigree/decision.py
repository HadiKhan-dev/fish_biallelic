"""Genome-wide decision from the chromosome scores (before the full-marker count release).

Steps: configuration rows of the eligible universe -> the full-data
fit (orientation, exposure gate, family terms, state, identity, graph) -> prior sensitivity of M0 ->
chromosome bootstrap and LOCO refits -> per-child support -> Tier A / Tier B / complete views,
marginal edges and candidate sets.

Method: docs/methods/pedigree.md#one-weighted-fit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, fields
from typing import Any

import numpy as np

from ...contracts.pedigree import (
    Decision,
    DecisionTrace,
    FamilySolveRecord,
    TierCalls,
    PartialEdges,
    GenomeScores,
    Orientation,
    Resampling,
    Selection,
)
from ...core.numerics import ordered_sum
from ...core.ties import TieCounter
from . import alternatives as alternatives_module
from . import release, resampling, selection
from .alternatives import Alternatives
from .eligibility import ParentEligibility
from .orientation import junction_evidence
from .parameters import PedigreeParameters
from .selection import DecisionInputs
from .ties import unique_finite_winner

__all__ = [
    "informative_chromosomes",
    "child_units",
    "FullFit",
    "start_fit",
    "decide_resampled",
]


def informative_chromosomes(alternatives: Alternatives, *, counter=None) -> np.ndarray:
    """bool [C, N]: fit rows with a unique parent-count winner (blocks in low material)."""
    winners = np.zeros((len(alternatives.log_likelihoods), alternatives.n_samples), dtype=np.bool_)
    states = np.arange(3, dtype=np.int64)
    for contig, likelihoods in enumerate(alternatives.log_likelihoods):
        evidence = alternatives_module.state_log_evidence(likelihoods, alternatives)
        for child in range(alternatives.n_samples):
            winner, _ = unique_finite_winner(
                states, evidence[child], key=f"informative/k{contig}/c{child}", counter=counter
            )
            winners[contig, child] = winner is not None
    return winners


def child_units(edges: np.ndarray, edge_units: np.ndarray, n_samples: int) -> np.ndarray:
    """float64 [C, N]: the map units a child's own evidence can claim on each chromosome: the most any of its eligible
    edges earns there (ExposureEvidence.edge_units; 1 for a child without eligible edges), so a whole long
    chromosome counts for a child only where some candidate edge is exposed over it."""
    units = np.ones((edge_units.shape[0], n_samples))
    if len(edges):
        np.maximum.at(units.T, edges[:, 0], edge_units.T)
    return units


@dataclass(frozen=True)
class FullFit:
    """The decision state before resampling: stacked rows, fit inputs and the full-data fit.

    ``counter`` is the decision's one tie counter: the full fit's events, then the informative-chromosome
    events after resampling (decide_resampled), in that order.
    """

    scores: GenomeScores
    eligibility: ParentEligibility
    inputs: DecisionInputs
    full: Selection
    orientation: Orientation
    adjusted: tuple[np.ndarray, np.ndarray]
    trace: dict
    sensitivity: tuple
    counter: TieCounter


def start_fit(scores: GenomeScores, eligibility: ParentEligibility, config: PedigreeParameters) -> FullFit:
    """Build the full-data fit from the alternatives' composite log likelihoods as supplied."""
    # np.argwhere's Fortran memory order must not choose downstream reductions.
    # Any: each value replaces an ndarray field by name, which mypy cannot match from a **mapping to the fields.
    contiguous: dict[str, Any] = {
        f.name: np.ascontiguousarray(getattr(scores, f.name))
        for f in fields(scores)
        if isinstance(getattr(scores, f.name), np.ndarray)
    }
    scores = replace(scores, **contiguous)
    alternatives = alternatives_module.build(scores, eligibility)
    inputs = DecisionInputs(
        alternatives,
        selection.build_exposure(scores, alternatives, config.exposure),
        junction_evidence(
            scores.junctions,
            scores.callable_bins,
            scores.total_bins,
            config.direction.chromosome_error_probability,
            scores.junction_dyads,
            scores.transmission_log_likelihood,
        ),
        regime=scores.regime,
    )
    if scores.regime == "low_material":
        inputs = replace(inputs, omission_totals=selection.omission_totals(inputs))
    trace: dict = {}
    counter = TieCounter()
    full, orientation, adjusted = selection.fit(np.ones(len(scores.contigs)), inputs, config, trace, counter=counter)
    sensitivity = tuple(
        (
            tuple(priors),
            full
            if tuple(priors) == tuple(config.selection.state_priors)
            else selection.select(*adjusted, alternatives, priors, orientation, config),
        )
        for priors in config.selection.prior_sensitivity
    )
    return FullFit(
        scores, eligibility, inputs, full, orientation, adjusted, trace, sensitivity, counter
    )


def decide_resampled(state: FullFit, resampled: Resampling, config: PedigreeParameters) -> Decision:
    """Support, tiers, views and candidate sets from the full fit and its resampling."""
    scores, eligibility, inputs = state.scores, state.eligibility, state.inputs
    alternatives, full, trace, counter = inputs.alternatives, state.full, state.trace, state.counter
    if scores.regime == "low_material":
        # PD15 block decisions are a different diagnostic scope from the genome
        # solve; the winner/tie algorithm itself stays exactly the released one.
        block_counter = TieCounter()
        winners = informative_chromosomes(alternatives, counter=block_counter)
        for event in block_counter.record().entries:
            counter.add_counts(event.kind.replace("pedigree.winner", "pedigree.informative_block.winner"),
                               event.decisions, event.inside_tie, event.wide, event.boundary)
    else:
        winners = informative_chromosomes(alternatives, counter=counter)
    informative = winners.sum(axis=0).astype(np.int64)
    informative_units = ordered_sum(
        winners * (np.ones(winners.shape) if scores.regime == "low_material" else
                   child_units(scores.edges, inputs.exposure.edge_units, alternatives.n_samples)), axis=0
    )
    tie_events = counter.record()
    log_odds = trace["orientation_log_odds"]
    evidence = release.child_evidence(
        full, list(state.sensitivity), resampled, alternatives, informative_units, log_odds, config
    )
    n = alternatives.n_samples
    low = scores.regime == "low_material"
    tier_a = (TierCalls(np.full(n, -1, np.int8), np.full(n, -1, np.int64), ("not_assessed",) * n,
                         np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool), np.zeros((n, 2), bool))
              if low else release.tier_calls("tier_a", config.release.tier_a, full, evidence, eligibility, config))
    tier_b = release.tier_calls("tier_b", config.release.tier_b, full, evidence, eligibility, config)
    complete = release.complete_calls(full, eligibility, alternatives)
    partial_a = (PartialEdges(np.full((n, 2), -1, np.int64), np.zeros((n, 2), bool), np.zeros(n, bool),
                  np.zeros(n, np.int64), np.full(n, 2, np.int64), np.full(n, -1, np.int8),
                  ("not_assessed",) * n, ("",) * n) if low else
                 release.partial_edges("tier_a", config.release.tier_a, tier_a, evidence, alternatives, config))
    partial_b = release.partial_edges("tier_b", config.release.tier_b, tier_b, evidence, alternatives, config)
    candidates = release.candidate_sets(tier_b, partial_b, full, alternatives, resampled, log_odds, config)
    direction = config.direction
    family_solve = FamilySolveRecord(
        iterations=int(trace["final_family_iterations"]),
        residual=float(trace["final_family_residual"]),
        tolerance=direction.final_tolerance,
        maximum_iterations=direction.final_max_iterations,
        retry_iterations=int(trace["final_family_retry_iterations"]),
        retry_limit=direction.retry_iterations,
        retry_damping=direction.retry_damping,
        frozen_path_preparation_passes=direction.message_passes,
        # Resamples that abstained for every sample (lower support, never a call).
        nonconverged_bootstrap_replicates=int(resampled.bootstrap.nonconverged),
        nonconverged_loco_replicates=int(resampled.loco.nonconverged),
    )
    reach = resampling.reach(resampled.multiplicities, inputs.exposure.units, config)
    if low:
        reach = replace(reach, tier_a_reachable=False, minimum_contigs_tier_a=0,
            statement=f"{len(scores.contigs)} informative blocks: Tier B assessable with low-material resampling; Tier A not assessed",
            unit_morgans=float("nan"))
    return Decision(
        scores,
        eligibility,
        inputs,
        full,
        state.orientation,
        state.adjusted,
        DecisionTrace(**trace),
        state.sensitivity,
        resampled,
        informative,
        evidence,
        tier_a,
        tier_b,
        complete,
        partial_a,
        partial_b,
        candidates,
        reach,
        family_solve,
        tie_events,
        informative_units,
    )
