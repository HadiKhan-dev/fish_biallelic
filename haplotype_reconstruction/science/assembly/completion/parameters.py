"""Name founder-completion parameters with their meaning, units and rationale.

Owns U03-U07 fold, EM, cavity, boundary-link and occupancy settings.
Must not read user files or select a computational route.
Method: docs/methods/assembly.md#completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CrossFitParameters:
    """The two-fold cross-fitted variational EM over diplotypes and missing founder alleles.

    max_iterations / minimum_iterations: EM iteration bounds; elbo_absolute_tolerance,
    elbo_relative_tolerance: convergence of the evidence lower bound; monotonic_tolerance: allowed
    ELBO decrease before an iteration counts as non-monotone. uniform_mix, log_floor: the robust
    genotype emission max(log((1 - mix) P + mix / 3), floor). fold_seed: seed of the deterministic
    two-fold sample split. minimum_call_probability: an unknown allele is released when its
    posterior reaches it; minimum_carriers: support by carrier count or expected carrier mass.
    """

    max_iterations: int = field(
        default=100,
        metadata={"units": "iterations", "provenance": "Maximum 100 variational EM iterations per training fold."},
    )
    minimum_iterations: int = field(
        default=2,
        metadata={"units": "iterations", "provenance": "Require 2 iterations before testing ELBO convergence."},
    )
    elbo_absolute_tolerance: float = field(
        default=1e-08,
        metadata={
            "units": "nats",
            "provenance": ("Absolute ELBO convergence tolerance 1e-8."),
        },
    )
    elbo_relative_tolerance: float = field(
        default=1e-08,
        metadata={
            "units": "dimensionless",
            "provenance": ("Relative ELBO convergence tolerance 1e-8."),
        },
    )
    monotonic_tolerance: float = field(
        default=1e-08,
        metadata={
            "units": "nats",
            "provenance": "Allow an ELBO decrease of at most 1e-8 before flagging a non-monotone iteration.",
        },
    )
    uniform_mix: float = field(
        default=0.01,
        metadata={
            "units": "probability",
            "provenance": "Uniform genotype emission mixture weight 0.01 for robust cross-fit evidence.",
        },
    )
    log_floor: float = field(
        default=-2.0,
        metadata={"units": "nats", "provenance": "Robust genotype log-emission floor -2 in variational EM."},
    )
    fold_seed: int = field(
        default=1729,
        metadata={
            "units": "dimensionless seed",
            "provenance": "Seed 1729 makes the content-keyed two-fold assignment reproducible.",
        },
    )
    minimum_call_probability: float = field(
        default=0.99,
        metadata={
            "units": "probability",
            "provenance": ("Release an unknown allele only at posterior probability at least 0.99."),
        },
    )
    minimum_carriers: int = field(
        default=2,
        metadata={"units": "samples", "provenance": "Require at least 2 carriers for a cross-fitted allele release."},
    )


@dataclass(frozen=True)
class LinkParameters:
    """Class-level boundary correspondences from cross-fitted carrier profiles.

    Costs are mean squared discrepancies on class-level expected copy number and its bounded
    presence proxy; a match must be a reciprocal unique optimum in pooled samples and in every
    fold. maximum_mean_cost = 0.099 bounds the mean squared copy/presence discrepancy.
    """

    minimum_pooled_informative_samples: int = field(
        default=6,
        metadata={
            "units": "samples",
            "provenance": ("Require 6 jointly informative samples for a pooled boundary correspondence."),
        },
    )
    minimum_fold_informative_samples: int = field(
        default=3,
        metadata={
            "units": "samples/fold",
            "provenance": ("Require 3 jointly informative samples in each cross-fit fold."),
        },
    )
    minimum_informative_sites_per_sample: int = field(
        default=1,
        metadata={
            "units": "sites/sample",
            "provenance": ("Require at least 1 informative site on each side of a boundary."),
        },
    )
    minimum_effective_copies: float = field(
        default=1.0,
        metadata={
            "units": "expected copies",
            "provenance": "Require pooled expected copy support at least 1 for a class correspondence.",
        },
    )
    minimum_fold_effective_copies: float = field(
        default=0.5,
        metadata={
            "units": "expected copies/fold",
            "provenance": ("Require expected copy support at least 0.5 in each fold."),
        },
    )
    maximum_mean_cost: float = field(
        default=0.099,
        metadata={
            "units": "squared copy/presence discrepancy",
            "provenance": "Accept a class link only when its mean copy/presence discrepancy is at most 0.099.",
        },
    )
    minimum_pooled_margin: float = field(
        default=0.03,
        metadata={
            "units": "mean cost difference",
            "provenance": "Require a pooled best-versus-runner-up cost margin of 0.03.",
        },
    )
    minimum_fold_margin: float = field(
        default=0.01,
        metadata={
            "units": "mean cost difference",
            "provenance": "Require a best-versus-runner-up cost margin of 0.01 in each fold.",
        },
    )
    minimum_original_hard_anchor_sites: int = field(
        default=1,
        metadata={
            "units": "sites",
            "provenance": ("Require at least 1 original hard-call anchor site for a boundary class."),
        },
    )
    tolerance: float = field(
        default=1e-12,
        metadata={
            "units": "cost units",
            "provenance": "Numerical comparison allowance 1e-12 in reciprocal class matching.",
        },
    )


@dataclass(frozen=True)
class CavityFillParameters:
    """Whole-bin leave-out ("cavity") fill of unknown founder alleles.

    snps_per_bin: bin width of the cavity; minimum_supporters: carrier support of a fill;
    minimum_posterior_probability: posterior of the filled allele; uniform_mix: emission mixture;
    tie_tolerance, founder_identifiability_tolerance: numerical ties; minimum_observable_founder_
    separation: each founder must differ observably from every other founder outside the bin.
    """

    snps_per_bin: int = field(
        default=50,
        metadata={
            "units": "sites/bin",
            "provenance": "Leave out 50-site bins to obtain outside-bin evidence for candidate fills.",
        },
    )
    minimum_supporters: int = field(
        default=2,
        metadata={
            "units": "samples or expected carrier mass",
            "provenance": "Require carrier count or mass at least 2 for a candidate cavity fill.",
        },
    )
    minimum_posterior_probability: float = field(
        default=0.99,
        metadata={
            "units": "probability",
            "provenance": ("Require filled-allele posterior probability at least 0.99."),
        },
    )
    uniform_mix: float = field(
        default=0.01,
        metadata={
            "units": "probability",
            "provenance": "Uniform genotype mixture weight 0.01 for robust cavity emissions.",
        },
    )
    tie_tolerance: float = field(
        default=1e-12,
        metadata={"units": "log-score units", "provenance": "Numerical tie tolerance 1e-12 in cavity calculations."},
    )
    founder_identifiability_tolerance: float = field(
        default=1e-12,
        metadata={
            "units": "allele probability",
            "provenance": ("Treat outside-bin founder probabilities within 1e-12 as indistinguishable."),
        },
    )
    minimum_observable_founder_separation: int = field(
        default=1,
        metadata={
            "units": "sites",
            "provenance": (
                "Require each founder to differ from every other at 1 or more observable outside-bin sites."
            ),
        },
    )


@dataclass(frozen=True)
class OccupancyParameters:
    """The variable-K occupancy gate: abundant and rare-continuity routes that release fills."""

    minimum_abundant_sum_squared_probability: float = field(
        default=8.0,
        metadata={
            "units": "sum of squared carrier probabilities",
            "provenance": ("Abundant occupancy needs a squared-carrier-probability sum above the band around 8."),
        },
    )
    abundant_high_confidence_probability: float = field(
        default=0.8,
        metadata={
            "units": "probability",
            "provenance": ("A high-confidence abundant-route carrier has probability at least 0.8."),
        },
    )
    minimum_abundant_high_confidence_carriers: int = field(
        default=5,
        metadata={
            "units": "samples",
            "provenance": ("Require 5 high-confidence carriers for the abundant route."),
        },
    )
    minimum_rare_local_carrier_probability: float = field(
        default=0.9,
        metadata={
            "units": "probability",
            "provenance": ("Rare continuity needs local carrier probability at least 0.9."),
        },
    )
    minimum_rare_neighbor_carrier_probability: float = field(
        default=0.8,
        metadata={
            "units": "probability",
            "provenance": ("Rare continuity needs neighbour carrier probability at least 0.8."),
        },
    )
    minimum_same_sample_continuity: float = field(
        default=0.72,
        metadata={
            "units": "probability product",
            "provenance": ("The same sample must have local-times-neighbour carrier probability at least 0.72."),
        },
    )
    minimum_rare_continuity_samples: int = field(
        default=2,
        metadata={
            "units": "samples",
            "provenance": ("Require 2 samples meeting the rare continuity conditions."),
        },
    )


@dataclass(frozen=True)
class CompletionParameters:
    """Founder completion before the hierarchy.

    maximum_discovery_unknown_mass: a block whose local selection left more unknown-copy mass
        (``LocalPanel.unknown_mass``) than this releases no fill. At the default 0.0, any
        unexplained copy vetoes fills in that block.
    """

    crossfit: CrossFitParameters = field(
        default_factory=CrossFitParameters,
        metadata={
            "units": "parameter record",
            "provenance": "Default two-fold variational EM and held-out release settings.",
        },
    )
    links: LinkParameters = field(
        default_factory=LinkParameters,
        metadata={
            "units": "parameter record",
            "provenance": "Default reciprocal boundary-class correspondence settings.",
        },
    )
    cavity: CavityFillParameters = field(
        default_factory=CavityFillParameters,
        metadata={"units": "parameter record", "provenance": "Default leave-bin-out candidate-fill settings."},
    )
    occupancy: OccupancyParameters = field(
        default_factory=OccupancyParameters,
        metadata={"units": "parameter record", "provenance": "Default abundant and rare-continuity occupancy gates."},
    )
    maximum_discovery_unknown_mass: float = field(
        default=0.0,
        metadata={
            "units": "unknown-copy fraction",
            "provenance": (
                "Maximum spatial MAP unknown-copy fraction 0; any unexplained copy vetoes completion fills."
            ),
        },
    )


# U04/U05/U07: normalized genotype evidence range; observation.py:15.
UNIFORM_TOLERANCE = 1e-12
# U05 bin emissions use this fixed floor independently of cross-fit configuration.
CAVITY_LOG_FLOOR = -2.0

__all__ = [
    "CompletionParameters",
    "CrossFitParameters",
    "CavityFillParameters",
    "LinkParameters",
    "OccupancyParameters",
    "UNIFORM_TOLERANCE",
    "CAVITY_LOG_FLOOR",
]
