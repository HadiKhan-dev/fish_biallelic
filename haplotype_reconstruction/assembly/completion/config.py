"""Settings of founder completion (base production values)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CrossFitConfig:
    """The two-fold cross-fitted variational EM over diplotypes and missing founder alleles.

    max_iterations / minimum_iterations: EM iteration bounds; elbo_absolute_tolerance,
    elbo_relative_tolerance: convergence of the evidence lower bound; monotonic_tolerance: allowed
    ELBO decrease before an iteration counts as non-monotone. uniform_mix, log_floor: the robust
    genotype emission max(log((1 - mix) P + mix / 3), floor). fold_seed: seed of the deterministic
    two-fold sample split. minimum_call_probability: an unknown allele is released when its
    held-out posterior reaches it; minimum_carriers: and at least this many samples carry it.
    """

    max_iterations: int = 100
    minimum_iterations: int = 2
    elbo_absolute_tolerance: float = 1e-8
    elbo_relative_tolerance: float = 1e-8
    monotonic_tolerance: float = 1e-8
    uniform_mix: float = 0.01
    log_floor: float = -2.0
    fold_seed: int = 1729
    minimum_call_probability: float = 0.99
    minimum_carriers: int = 2


@dataclass(frozen=True)
class LinkConfig:
    """Class-level boundary correspondences from cross-fitted carrier profiles.

    Costs are mean squared discrepancies on class-level expected copy number and its bounded
    presence proxy; a match must be a reciprocal unique optimum in pooled samples and in every
    fold. maximum_mean_cost = 0.099 is base's frozen release threshold (chr8/chr13 fixtures).
    """

    minimum_pooled_informative_samples: int = 6
    minimum_fold_informative_samples: int = 3
    minimum_informative_sites_per_sample: int = 1
    minimum_effective_copies: float = 1.0
    minimum_fold_effective_copies: float = 0.5
    maximum_mean_cost: float = 0.099
    minimum_pooled_margin: float = 0.03
    minimum_fold_margin: float = 0.01
    minimum_original_hard_anchor_sites: int = 1
    tolerance: float = 1e-12


@dataclass(frozen=True)
class CavityFillConfig:
    """Whole-bin leave-out ("cavity") fill of unknown founder alleles.

    snps_per_bin: bin width of the cavity; minimum_supporters: carrier support of a fill;
    minimum_posterior_probability: posterior of the filled allele; uniform_mix: emission mixture;
    tie_tolerance, founder_identifiability_tolerance: numerical ties; minimum_observable_founder_
    separation: each founder must differ observably from every other founder outside the bin.
    """

    snps_per_bin: int = 50
    minimum_supporters: int = 2
    minimum_posterior_probability: float = 0.99
    uniform_mix: float = 0.01
    tie_tolerance: float = 1e-12
    founder_identifiability_tolerance: float = 1e-12
    minimum_observable_founder_separation: int = 1


@dataclass(frozen=True)
class OccupancyConfig:
    """The variable-K occupancy gate: abundant and rare-continuity routes that release fills."""

    minimum_abundant_sum_squared_probability: float = 8.0
    abundant_high_confidence_probability: float = 0.8
    minimum_abundant_high_confidence_carriers: int = 5
    minimum_rare_local_carrier_probability: float = 0.9
    minimum_rare_neighbor_carrier_probability: float = 0.8
    minimum_same_sample_continuity: float = 0.72
    minimum_rare_continuity_samples: int = 2


@dataclass(frozen=True)
class CompletionConfig:
    """Founder completion before the hierarchy.

    maximum_discovery_unknown_mass: a block whose local selection left more unknown-copy mass
        (``LocalPanel.unknown_mass``) than this releases no fill (base 0.0: any unexplained copy
        vetoes fills in that block).
    """

    crossfit: CrossFitConfig = field(default_factory=CrossFitConfig)
    links: LinkConfig = field(default_factory=LinkConfig)
    cavity: CavityFillConfig = field(default_factory=CavityFillConfig)
    occupancy: OccupancyConfig = field(default_factory=OccupancyConfig)
    maximum_discovery_unknown_mass: float = 0.0
