"""Name founder-completion parameters with their preserved scientific provenance.

Owns U03-U07 fold, EM, cavity, boundary-link and occupancy settings.
Must not read user files or select a computational route.
Method: docs/methods/assembly.md#completion.
"""

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

    max_iterations: int = field(
        default=100,
        metadata={
            "provenance": "P0 assembly-core U03-U07; baseline completion/config.py CrossFitConfig.max_iterations"
        },
    )
    minimum_iterations: int = field(
        default=2,
        metadata={
            "provenance": "P0 assembly-core U03-U07; baseline completion/config.py CrossFitConfig.minimum_iterations"
        },
    )
    elbo_absolute_tolerance: float = field(
        default=1e-08,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; baseline completion/config.py CrossFitConfig.elbo_absolute_tolerance"
            ),
        },
    )
    elbo_relative_tolerance: float = field(
        default=1e-08,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; baseline completion/config.py CrossFitConfig.elbo_relative_tolerance"
            ),
        },
    )
    monotonic_tolerance: float = field(
        default=1e-08,
        metadata={
            "provenance": "P0 assembly-core U03-U07; baseline completion/config.py CrossFitConfig.monotonic_tolerance"
        },
    )
    uniform_mix: float = field(
        default=0.01,
        metadata={"provenance": "P0 assembly-core U03-U07; baseline completion/config.py CrossFitConfig.uniform_mix"},
    )
    log_floor: float = field(
        default=-2.0,
        metadata={"provenance": "P0 assembly-core U03-U07; baseline completion/config.py CrossFitConfig.log_floor"},
    )
    fold_seed: int = field(
        default=1729,
        metadata={"provenance": "P0 assembly-core U03-U07; baseline completion/config.py CrossFitConfig.fold_seed"},
    )
    minimum_call_probability: float = field(
        default=0.99,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; baseline completion/config.py CrossFitConfig.minimum_call_probability"
            ),
        },
    )
    minimum_carriers: int = field(
        default=2,
        metadata={
            "provenance": "P0 assembly-core U03-U07; baseline completion/config.py CrossFitConfig.minimum_carriers"
        },
    )


@dataclass(frozen=True)
class LinkConfig:
    """Class-level boundary correspondences from cross-fitted carrier profiles.

    Costs are mean squared discrepancies on class-level expected copy number and its bounded
    presence proxy; a match must be a reciprocal unique optimum in pooled samples and in every
    fold. maximum_mean_cost = 0.099 is base's frozen release threshold (chr8/chr13 fixtures).
    """

    minimum_pooled_informative_samples: int = field(
        default=6,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; baseline completion/config.py LinkConfig.minimum_pooled_informative_samples"
            ),
        },
    )
    minimum_fold_informative_samples: int = field(
        default=3,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; baseline completion/config.py LinkConfig.minimum_fold_informative_samples"
            ),
        },
    )
    minimum_informative_sites_per_sample: int = field(
        default=1,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; "
                "baseline completion/config.py LinkConfig.minimum_informative_sites_per_sample"
            ),
        },
    )
    minimum_effective_copies: float = field(
        default=1.0,
        metadata={
            "provenance": "P0 assembly-core U03-U07; baseline completion/config.py LinkConfig.minimum_effective_copies"
        },
    )
    minimum_fold_effective_copies: float = field(
        default=0.5,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; baseline completion/config.py LinkConfig.minimum_fold_effective_copies"
            ),
        },
    )
    maximum_mean_cost: float = field(
        default=0.099,
        metadata={"provenance": "P0 assembly-core U03-U07; baseline completion/config.py LinkConfig.maximum_mean_cost"},
    )
    minimum_pooled_margin: float = field(
        default=0.03,
        metadata={
            "provenance": "P0 assembly-core U03-U07; baseline completion/config.py LinkConfig.minimum_pooled_margin"
        },
    )
    minimum_fold_margin: float = field(
        default=0.01,
        metadata={
            "provenance": "P0 assembly-core U03-U07; baseline completion/config.py LinkConfig.minimum_fold_margin"
        },
    )
    minimum_original_hard_anchor_sites: int = field(
        default=1,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; baseline completion/config.py LinkConfig.minimum_original_hard_anchor_sites"
            ),
        },
    )
    tolerance: float = field(
        default=1e-12,
        metadata={"provenance": "P0 assembly-core U03-U07; baseline completion/config.py LinkConfig.tolerance"},
    )


@dataclass(frozen=True)
class CavityFillConfig:
    """Whole-bin leave-out ("cavity") fill of unknown founder alleles.

    snps_per_bin: bin width of the cavity; minimum_supporters: carrier support of a fill;
    minimum_posterior_probability: posterior of the filled allele; uniform_mix: emission mixture;
    tie_tolerance, founder_identifiability_tolerance: numerical ties; minimum_observable_founder_
    separation: each founder must differ observably from every other founder outside the bin.
    """

    snps_per_bin: int = field(
        default=50,
        metadata={
            "provenance": "P0 assembly-core U03-U07; baseline completion/config.py CavityFillConfig.snps_per_bin"
        },
    )
    minimum_supporters: int = field(
        default=2,
        metadata={
            "provenance": "P0 assembly-core U03-U07; baseline completion/config.py CavityFillConfig.minimum_supporters"
        },
    )
    minimum_posterior_probability: float = field(
        default=0.99,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; baseline completion/config.py CavityFillConfig.minimum_posterior_probability"
            ),
        },
    )
    uniform_mix: float = field(
        default=0.01,
        metadata={"provenance": "P0 assembly-core U03-U07; baseline completion/config.py CavityFillConfig.uniform_mix"},
    )
    tie_tolerance: float = field(
        default=1e-12,
        metadata={
            "provenance": "P0 assembly-core U03-U07; baseline completion/config.py CavityFillConfig.tie_tolerance"
        },
    )
    founder_identifiability_tolerance: float = field(
        default=1e-12,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; "
                "baseline completion/config.py CavityFillConfig.founder_identifiability_tolerance"
            ),
        },
    )
    minimum_observable_founder_separation: int = field(
        default=1,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; "
                "baseline completion/config.py CavityFillConfig.minimum_observable_founder_separation"
            ),
        },
    )


@dataclass(frozen=True)
class OccupancyConfig:
    """The variable-K occupancy gate: abundant and rare-continuity routes that release fills."""

    minimum_abundant_sum_squared_probability: float = field(
        default=8.0,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; "
                "baseline completion/config.py OccupancyConfig.minimum_abundant_sum_squared_probability"
            ),
        },
    )
    abundant_high_confidence_probability: float = field(
        default=0.8,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; "
                "baseline completion/config.py OccupancyConfig.abundant_high_confidence_probability"
            ),
        },
    )
    minimum_abundant_high_confidence_carriers: int = field(
        default=5,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; "
                "baseline completion/config.py OccupancyConfig.minimum_abundant_high_confidence_carriers"
            ),
        },
    )
    minimum_rare_local_carrier_probability: float = field(
        default=0.9,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; "
                "baseline completion/config.py OccupancyConfig.minimum_rare_local_carrier_probability"
            ),
        },
    )
    minimum_rare_neighbor_carrier_probability: float = field(
        default=0.8,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; "
                "baseline completion/config.py OccupancyConfig.minimum_rare_neighbor_carrier_probability"
            ),
        },
    )
    minimum_same_sample_continuity: float = field(
        default=0.72,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; baseline completion/config.py OccupancyConfig.minimum_same_sample_continuity"
            ),
        },
    )
    minimum_rare_continuity_samples: int = field(
        default=2,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; "
                "baseline completion/config.py OccupancyConfig.minimum_rare_continuity_samples"
            ),
        },
    )


@dataclass(frozen=True)
class CompletionParameters:
    """Founder completion before the hierarchy.

    maximum_discovery_unknown_mass: a block whose local selection left more unknown-copy mass
        (``LocalPanel.unknown_mass``) than this releases no fill (base 0.0: any unexplained copy
        vetoes fills in that block).
    """

    crossfit: CrossFitConfig = field(
        default_factory=CrossFitConfig,
        metadata={
            "provenance": "P0 assembly-core U03-U07; baseline completion/config.py CompletionParameters.crossfit"
        },
    )
    links: LinkConfig = field(
        default_factory=LinkConfig,
        metadata={"provenance": "P0 assembly-core U03-U07; baseline completion/config.py CompletionParameters.links"},
    )
    cavity: CavityFillConfig = field(
        default_factory=CavityFillConfig,
        metadata={"provenance": "P0 assembly-core U03-U07; baseline completion/config.py CompletionParameters.cavity"},
    )
    occupancy: OccupancyConfig = field(
        default_factory=OccupancyConfig,
        metadata={
            "provenance": "P0 assembly-core U03-U07; baseline completion/config.py CompletionParameters.occupancy"
        },
    )
    maximum_discovery_unknown_mass: float = field(
        default=0.0,
        metadata={
            "provenance": (
                "P0 assembly-core U03-U07; "
                "baseline completion/config.py CompletionParameters.maximum_discovery_unknown_mass"
            ),
        },
    )


# U04/U05/U07: normalized genotype evidence range; observation.py:15.
UNIFORM_TOLERANCE = 1e-12
# U05 bin emissions use this fixed floor independently of cross-fit configuration.
CAVITY_LOG_FLOOR = -2.0

__all__ = [
    "CompletionParameters",
    "CrossFitConfig",
    "CavityFillConfig",
    "LinkConfig",
    "OccupancyConfig",
    "UNIFORM_TOLERANCE",
    "CAVITY_LOG_FLOOR",
]
