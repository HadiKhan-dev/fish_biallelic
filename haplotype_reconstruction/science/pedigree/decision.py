"""Genome-wide decision from the chromosome scores (before the full-marker count release).

Steps: configuration rows of the eligible universe -> predictive evidence scale -> the full-data
fit (orientation, exposure gate, family terms, state, identity, graph) -> prior sensitivity of M0 ->
chromosome bootstrap and LOCO refits -> per-child support -> Tier A / Tier B / complete views,
marginal edges and candidate sets.

Method: docs/methods/pedigree.md#one-weighted-fit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from ...contracts.pedigree import (
    CalibrationRecord,
    Decision,
    DecisionTrace,
    FamilySolveRecord,
    GenomeScores,
    Orientation,
    Resampling,
    Selection,
)
from ...core.numerics import chromosome_sum
from ...core.ties import TieCounter
from . import alternatives as alternatives_module
from . import calibration, release, resampling, selection
from .alternatives import Alternatives
from .eligibility import ParentEligibility
from .orientation import junction_evidence
from .parameters import PedigreeConfig
from .selection import DecisionInputs
from .ties import unique_finite_winner

__all__ = [
    "informative_chromosomes",
    "child_units",
    "FitStart",
    "FullFit",
    "fit_full",
    "start_fit",
    "finish_fit",
    "decide_resampled",
    "decide_floored",
]


def informative_chromosomes(alternatives: Alternatives, *, counter=None) -> np.ndarray:
    """bool [C, N]: chromosomes with any unique parent-count evidence winner, regardless of the genome-wide call."""
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
    """The decision state before resampling: stacked rows, evidence scale, fit inputs and the full-data fit.

    ``counter`` is the decision's one tie counter: the full fit's events, then the informative-chromosome
    events after resampling (decide_resampled), in that order.
    """

    scores: GenomeScores
    eligibility: ParentEligibility
    inputs: DecisionInputs
    calibration: CalibrationRecord
    full: Selection
    orientation: Orientation
    adjusted: tuple[np.ndarray, np.ndarray]
    trace: dict
    sensitivity: tuple
    counter: TieCounter


@dataclass(frozen=True)
class FitStart:
    """fit_full before its evidence scale: contiguous scores, alternatives and the prepared PD03 problem."""

    scores: GenomeScores
    eligibility: ParentEligibility
    alternatives: Alternatives
    calibration: calibration.CalibrationProblem


def fit_full(scores: GenomeScores, eligibility: ParentEligibility, config: PedigreeConfig, executor=None) -> FullFit:
    """Rows, evidence scale, fit inputs, the full-data fit and its prior sensitivity (decide_floored's first part).

    ``executor``: the stage's calibration term executor (None: serial).
    """
    started = start_fit(scores, eligibility, config)
    return finish_fit(started, calibration.solve(started.calibration, executor), config)


def start_fit(scores: GenomeScores, eligibility: ParentEligibility, config: PedigreeConfig) -> FitStart:
    """fit_full's first part: rows and the calibration problem; the stage may run its scale search in rounds."""
    # L2: np.argwhere's Fortran memory order must not choose downstream reductions.
    # Any: each value replaces an ndarray field by name, which mypy cannot match from a **mapping to the fields.
    contiguous: dict[str, Any] = {
        f.name: np.ascontiguousarray(getattr(scores, f.name))
        for f in fields(scores)
        if isinstance(getattr(scores, f.name), np.ndarray)
    }
    scores = replace(scores, **contiguous)
    alternatives = alternatives_module.build(scores, eligibility)
    return FitStart(scores, eligibility, alternatives, calibration.problem(scores, alternatives, config))


def finish_fit(started: FitStart, fitted: tuple[float, CalibrationRecord], config: PedigreeConfig) -> FullFit:
    """fit_full's last part: the scaled fit inputs, the full-data fit and its prior sensitivity."""
    scores, eligibility, alternatives = started.scores, started.eligibility, started.alternatives
    scale, calibration_report = fitted
    if scale != 1.0:
        alternatives = alternatives.scaled(scale)
    inputs = DecisionInputs(
        alternatives,
        selection.build_exposure(scores, alternatives, config.exposure),
        junction_evidence(
            scores.junctions,
            scores.callable_bins,
            scores.total_bins,
            config.direction.chromosome_error_probability,
            scores.junction_dyads,
            scores.junction_log_cdf,
        ),
    )
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
        scores, eligibility, inputs, calibration_report, full, orientation, adjusted, trace, sensitivity, counter
    )


def decide_resampled(state: FullFit, resampled: Resampling, config: PedigreeConfig) -> Decision:
    """Support, tiers, views and candidate sets from the full fit and its resampling (decide_floored's last part)."""
    scores, eligibility, inputs = state.scores, state.eligibility, state.inputs
    alternatives, full, trace, counter = inputs.alternatives, state.full, state.trace, state.counter
    winners = informative_chromosomes(alternatives, counter=counter)
    informative = winners.sum(axis=0).astype(np.int64)
    informative_units = chromosome_sum(
        winners * child_units(scores.edges, inputs.exposure.edge_units, alternatives.n_samples)
    )
    tie_events = counter.record()
    log_odds = trace["orientation_log_odds"]
    evidence = release.child_evidence(
        full, list(state.sensitivity), resampled, alternatives, informative_units, log_odds, config
    )
    tier_a = release.tier_calls("tier_a", config.release.tier_a, full, evidence, eligibility, config)
    tier_b = release.tier_calls("tier_b", config.release.tier_b, full, evidence, eligibility, config)
    complete = release.complete_calls(full, eligibility, alternatives)
    partial_a = release.partial_edges("tier_a", config.release.tier_a, tier_a, evidence, alternatives, config)
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
        state.calibration,
        resampling.reach(resampled.multiplicities, inputs.exposure.units, config),
        family_solve,
        tie_events,
        informative_units,
    )


def decide_floored(
    scores: GenomeScores, eligibility: ParentEligibility, config: PedigreeConfig, executor=None
) -> Decision:
    """Decide from the already fixed full-genome M1 profile, resampling serially in this process.

    Used by the certified screen. The genome floor is not idempotent: chromosome M1 inputs
    must already have been floored exactly once using the fixed genome profile.
    ``eligibility`` must be the one ``scores`` was stacked with.
    ``executor``: the stage's calibration term executor (None: serial). The stage runs the same resampling
    tasks as independent engine subtasks between fit_full and decide_resampled.
    """
    state = fit_full(scores, eligibility, config, executor)
    return decide_resampled(state, resampling.run(state.inputs, config), config)
