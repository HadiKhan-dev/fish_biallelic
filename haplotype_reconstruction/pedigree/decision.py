"""Genome-wide decision from the chromosome scores (before the full-marker count release).

Steps: configuration rows of the eligible universe -> predictive evidence scale -> the full-data
fit (orientation, exposure gate, family terms, state, identity, graph) -> prior sensitivity of M0 ->
chromosome bootstrap and LOCO refits -> per-child support -> Tier A / Tier B / complete views,
marginal edges and candidate sets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import alternatives as alternatives_module
from . import calibration, m1, release, resampling, selection
from .alternatives import Alternatives
from .config import PedigreeConfig
from .eligibility import ParentEligibility
from .orientation import Orientation, junction_evidence
from .records import GenomeScores, TierReach
from .release import CandidateSets, ChildEvidence, PartialEdges, TierCalls
from .resampling import Resampling
from .selection import DecisionInputs, ExposureEvidence, Selection
from .ties import unique_finite_winner


@dataclass(frozen=True)
class Decision:
    """Everything the release, exclusion and report steps read.

    scores: genome-floor transformed chromosome scores (unscaled); inputs: the fit inputs (alternatives
    carry the evidence scale); full / orientation / adjusted (state, identity scores) / trace: the
    full-data fit and its per-row terms; sensitivity: (priors, selection) per alternative prior;
    informative: int64 [N] chromosomes whose own evidence picks a unique state.
    """

    scores: GenomeScores
    eligibility: ParentEligibility
    inputs: DecisionInputs
    full: Selection
    orientation: Orientation
    adjusted: tuple[np.ndarray, np.ndarray]
    trace: dict
    sensitivity: tuple[tuple[tuple[float, float, float], Selection], ...]
    resampling: Resampling
    informative: np.ndarray
    evidence: ChildEvidence
    tier_a: TierCalls
    tier_b: TierCalls
    complete: TierCalls
    partial_a: PartialEdges
    partial_b: PartialEdges
    candidates: CandidateSets
    calibration: dict
    reach: TierReach
    family_solve: dict
    tie_events: dict

    @property
    def alternatives(self) -> Alternatives:
        return self.inputs.alternatives


def informative_contigs(alternatives: Alternatives) -> np.ndarray:
    """int64 [N]: chromosomes on which the child's own state evidence has a unique winner."""
    counts = np.zeros(alternatives.n_samples, dtype=np.int64)
    states = np.arange(3, dtype=np.int64)
    for contig, likelihoods in enumerate(alternatives.log_likelihoods):
        evidence = alternatives_module.state_log_evidence(likelihoods, alternatives)
        for child in range(alternatives.n_samples):
            winner, _ = unique_finite_winner(states, evidence[child], key=f"informative/k{contig}/c{child}")
            counts[child] += winner is not None
    return counts


def decide(scores: GenomeScores, eligibility: ParentEligibility, config: PedigreeConfig,
           budget) -> Decision:
    """Full-data fit, resampling support and the tier views of the informative chromosomes."""
    # Public input is RAW chromosome aggregates. Freeze the full-genome gate
    # and chromosome profile before calibration and every resampling fit.
    scores = m1.floor_genome(scores, eligibility, config)
    return decide_floored(scores, eligibility, config, budget)


def decide_floored(scores: GenomeScores, eligibility: ParentEligibility,
                   config: PedigreeConfig, budget) -> Decision:
    """Decide from the already fixed full-genome M1 profile.

    Used by the certified screen. The genome floor is not idempotent: raw
    chromosome scores must use ``decide`` instead. ``eligibility`` must be the one ``scores`` was stacked with.
    """
    alternatives = alternatives_module.build(scores, eligibility)
    scale, calibration_report = calibration.fit_scale(scores, alternatives, config)
    if scale != 1.0:
        alternatives = alternatives.scaled(scale)
    inputs = DecisionInputs(
        alternatives,
        ExposureEvidence.build(scores.edges, scores.edge_exposed, scores.pair_exposed, scores.total_bins,
                               alternatives, config.exposure),
        junction_evidence(scores.junctions, scores.callable_bins, scores.total_bins,
                          config.direction.chromosome_error_probability, scores.junction_dyads,
                          scores.junction_log_cdf),
        eligibility.chronology)
    trace: dict = {}
    from ..maths import ties as tie_rules
    from . import ties as pedigree_ties
    pedigree_ties.FULL_FIT_COUNTER = tie_rules.TieCounter()
    pedigree_ties.CENSUS_CONTEXT = "full"         # census keys and tie counts for the full-data fit only
    try:
        full, orientation, adjusted = selection.fit(np.ones(len(scores.contigs)), inputs, config, trace)
    finally:
        pedigree_ties.CENSUS_CONTEXT = None
    sensitivity = tuple(
        (tuple(priors), full if tuple(priors) == tuple(config.selection.state_priors)
         else selection.select(*adjusted, alternatives, priors, orientation, config))
        for priors in config.selection.prior_sensitivity)
    resampled = resampling.run(inputs, config, budget)
    pedigree_ties.CENSUS_CONTEXT = "full"
    try:
        informative = informative_contigs(alternatives)
    finally:
        pedigree_ties.CENSUS_CONTEXT = None
    tie_events = pedigree_ties.FULL_FIT_COUNTER.record()
    pedigree_ties.FULL_FIT_COUNTER = None
    log_odds = trace["orientation_log_odds"]
    evidence = release.child_evidence(full, list(sensitivity), resampled, alternatives, informative,
                                      log_odds, eligibility.chronology, config)
    tier_a = release.tier_calls("tier_a", config.release.tier_a, full, evidence, eligibility, config)
    tier_b = release.tier_calls("tier_b", config.release.tier_b, full, evidence, eligibility, config)
    complete = release.complete_calls(full, eligibility, alternatives)
    partial_a = release.partial_edges("tier_a", config.release.tier_a, tier_a, evidence,
                                      alternatives, config)
    partial_b = release.partial_edges("tier_b", config.release.tier_b, tier_b, evidence,
                                      alternatives, config)
    candidates = release.candidate_sets(tier_b, partial_b, full, alternatives, resampled, log_odds,
                                        eligibility.chronology, config)
    direction = config.direction
    family_solve = dict(
        iterations=int(trace["final_family_iterations"]),
        residual=float(trace["final_family_residual"]), tolerance=direction.final_tolerance,
        maximum_iterations=direction.final_max_iterations,
        retry_iterations=int(trace["final_family_retry_iterations"]),
        retry_limit=direction.retry_iterations, retry_damping=direction.retry_damping,
        frozen_path_preparation_passes=direction.message_passes,
        # Resamples that abstained for every sample (lower support, never a call).
        nonconverged_bootstrap_replicates=int(resampled.bootstrap.nonconverged),
        nonconverged_loco_replicates=int(resampled.loco.nonconverged))
    return Decision(scores, eligibility, inputs, full, orientation, adjusted, trace, sensitivity,
                    resampled, informative, evidence, tier_a, tier_b, complete, partial_a, partial_b,
                    candidates, calibration_report, resampling.reach(resampled.multiplicities, config),
                    family_solve, tie_events)
