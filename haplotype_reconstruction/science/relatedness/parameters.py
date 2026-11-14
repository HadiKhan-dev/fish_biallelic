"""Name the family model and search parameters.

Likelihood thresholds are in nats, support intervals in Morgans, and probability controls are dimensionless.
Method: docs/methods/relatedness.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlleleFrequencyParameters:
    """Sitewise Hardy-Weinberg EM schedule and probability bounds."""

    iterations: int = field(
        default=6,
        metadata={
            "provenance": (
                "Hardy-Weinberg EM updates per site (iterations); fixes the allele-frequency "
                "estimation schedule."
            ),
        },
    )
    initial: float = field(
        default=0.5,
        metadata={
            "provenance": (
                "Initial ALT frequency (probability); starts the sitewise EM at equal REF and ALT "
                "mass."
            ),
        },
    )
    clip: float = field(
        default=1e-4,
        metadata={
            "provenance": (
                "Lower ALT-frequency bound and distance from one (probability); keeps genotype "
                "priors away from zero."
            ),
        },
    )
    total_floor: float = field(
        default=1e-300,
        metadata={
            "provenance": (
                "Minimum genotype-weight sum (probability); prevents division by zero in the EM "
                "posterior."
            ),
        },
    )


@dataclass(frozen=True)
class FamilyParameters:
    """Family evidence gates, objective tolerances, support units and workspace bounds."""

    minor_allele_frequency: float = field(
        default=0.05,
        metadata={
            "provenance": (
                "Minimum minor allele frequency (fraction); selects informative sites for family "
                "fitting."
            ),
        },
    )
    minimum_observed: float = field(
        default=0.5,
        metadata={
            "provenance": (
                "Minimum observed-sample fraction (fraction); excludes poorly observed sites from "
                "family fitting."
            ),
        },
    )
    family_site_step: int = field(
        default=20,
        metadata={
            "provenance": (
                "Stride through eligible family sites (sites); thins the family likelihood "
                "evidence."
            ),
        },
    )
    max_sweeps: int = field(
        default=30,
        metadata={
            "provenance": (
                "Maximum sweeps of each family search (sweeps); bounds search iterations."
            ),
        },
    )
    support_unit_morgans: float = field(
        default=1.0,
        metadata={
            "provenance": (
                "Length of the support units (Morgans of the configured working map, DESIGN_417 D5): the family "
                "sites' axis pooled over contigs in configured order is cut into consecutive units of this "
                "length, each weighted by its Morgans in the member and parent support shares."
            ),
        },
    )
    support_tail_morgans: float = field(
        default=0.5,
        metadata={
            "provenance": (
                "Final-piece length (Morgans) below which the last piece of the pooled support axis joins the "
                "unit before it (DESIGN_417 D5, the pedigree material rule)."
            ),
        },
    )
    infer_rate: bool = field(
        default=False,
        metadata={
            "provenance": (
                "Use the estimated genome-wide recombination rate as the working fallback rate of contigs "
                "without a map file when its gate passes, the supplied rate lies outside its 95 % interval and "
                "at least minimum_adoption_families informative families were used (boolean); off: the "
                "estimate is a diagnostic and nothing downstream reads it."
            ),
        },
    )
    gain_tolerance: float = field(
        default=1e-6,
        metadata={
            "provenance": (
                "Family objective and support tie tolerance (nats); prevents numerical noise from "
                "choosing a placement or counting support."
            ),
        },
    )
    screen_margin: float = field(
        default=0.1,
        metadata={
            "provenance": (
                "Fixed screening-error allowance (nats); triggers exact evaluation near a "
                "screened gain boundary."
            ),
        },
    )
    screen_margin_per_site: float = field(
        default=1e-6,
        metadata={
            "provenance": (
                "Additional screening allowance per family site (nats per site); scales the error "
                "allowance with likelihood length."
            ),
        },
    )
    screen_rounding: float = field(
        default=2.0**-53,
        metadata={
            "provenance": (
                "Relative rounding factor in the screening-error bound (dimensionless); "
                "multiplied by twice the padded site count and gain magnitude."
            ),
        },
    )
    screen_padding: int = field(
        default=16,
        metadata={
            "provenance": (
                "Extra site count in the screening-rounding bound (sites); provides a fixed "
                "rounding allowance for short inputs."
            ),
        },
    )
    screen_tile: int = field(
        default=16,
        metadata={
            "provenance": (
                "Family-pair screen tile width (families); bounds workspace without selecting "
                "candidates."
            ),
        },
    )
    sum_bytes: int = field(
        default=32 * 1024**2,
        metadata={
            "provenance": (
                "Target member-sum temporary budget (bytes); bounds the site chunk for "
                "accumulating family terms."
            ),
        },
    )
    pair_batch: int = field(
        default=262144,
        metadata={
            "provenance": (
                "Packed pair gains evaluated per batch (pairs); bounds exact-evaluation "
                "workspace."
            ),
        },
    )
    prefetch_batch: int = field(
        default=64,
        metadata={
            "provenance": (
                "Family fits prefetched per batch (families); bounds cached evaluation work."
            ),
        },
    )
    term_floor: float = field(
        default=1e-300,
        metadata={
            "provenance": (
                "Minimum genotype term before logarithms (probability); keeps family likelihood "
                "terms finite."
            ),
        },
    )


@dataclass(frozen=True)
class RateParameters:
    """Genome-wide recombination-rate estimate from relatives and its adoption gate (rl20).

    Measured on 85 simulated runs and the TroMau cross (estimator version 5 of the 4.18 rate study); not user
    settings. Rates are in cM/Mb, likelihood thresholds in nats, probabilities dimensionless.
    """

    site_posterior: float = field(
        default=0.99,
        metadata={
            "provenance": (
                "Minimum leave-pair-out posterior of a family's parental genotype pair for a site to enter a "
                "pair's inheritance chain (probability); the pair's chain conditions on that pair exactly."
            ),
        },
    )
    segregation_posterior: float = field(
        default=0.9,
        metadata={
            "provenance": (
                "Minimum posterior of the same unordered parental pair from the other members alone, both slots "
                "latent (probability); a sequenced slot's confident false heterozygous calls cannot select sites "
                "its children do not segregate at."
            ),
        },
    )
    minimum_information: int = field(
        default=2,
        metadata={
            "provenance": (
                "Minimum family information, members - 2 + 2 x sequenced slots (count); a family below it cannot "
                "determine its parents' genotypes without the pair."
            ),
        },
    )
    rate_low: float = field(
        default=0.2,
        metadata={"provenance": "Lowest rate of the likelihood grid (cM/Mb)."},
    )
    rate_high: float = field(
        default=50.0,
        metadata={"provenance": "Highest rate of the likelihood grid (cM/Mb)."},
    )
    rate_points: int = field(
        default=48,
        metadata={
            "provenance": "Log-spaced rate grid points (count); the maximiser is refined by a parabola on log rate."
        },
    )
    artefacts: tuple[tuple[float, float], ...] = field(
        default=((0.0, 20.0), (0.1, 20.0), (0.1, 200.0), (1.0, 20.0), (1.0, 200.0), (10.0, 20.0), (10.0, 200.0)),
        metadata={
            "provenance": (
                "Artefact-stretch nuisance grid: (start rate per Mb, mean length kb) pairs, fitted with the "
                "rate by maximum likelihood and shared by the data set; simulations select 0-0.1 per Mb, TroMau "
                "0.1-1 per Mb."
            ),
        },
    )
    error_rates: tuple[float, ...] = field(
        default=(0.0, 0.001, 0.003, 0.01, 0.03, 0.1),
        metadata={
            "provenance": (
                "Isolated genotype-error grid (probability per site): each selected site's state emissions are "
                "mixed with their state mean, e' = (1 - eps) e + eps mean(e), and eps is fitted by maximum likelihood "
                "with the rate; errors the genotype likelihoods do not represent otherwise read as inheritance "
                "changes (simulations select 0-0.001, TroMau 0.03, a gain of 712 nats)."
            ),
        },
    )
    inconsistency_nats: float = field(
        default=10.0,
        metadata={
            "provenance": (
                "Family exclusion threshold (nats): a family whose own composite maximum exceeds its value at "
                "the leave-one-family-out maximiser by more than this is dropped, one family at a time; read on "
                "the chi-square(1) scale (p about 1e-5)."
            ),
        },
    )
    bootstrap_replicates: int = field(
        default=1000,
        metadata={"provenance": "Percentile-bootstrap replicates of the interval (count)."},
    )
    bootstrap_seed: int = field(
        default=417,
        metadata={"provenance": "Fixed seed of the bootstrap draws (integer); the interval is reproducible."},
    )
    minimum_interval_units: int = field(
        default=5,
        metadata={
            "provenance": (
                "Minimum contigs for the two-level bootstrap (families, then each drawn family's chromosomes), "
                "else minimum families for a family bootstrap (count); a bootstrap over fewer units is degenerate "
                "and the estimate falls back."
            ),
        },
    )
    minimum_determined_sites: float = field(
        default=1000.0,
        metadata={
            "provenance": (
                "Minimum meiosis-weighted mean of the families' median determined sites per pair (sites); the "
                "depth and design guard: 1x runs score 2-200, 5x runs 3,000-20,000."
            ),
        },
    )
    maximum_half_width: float = field(
        default=0.25,
        metadata={
            "provenance": (
                "Maximum 95 % interval half-width relative to the estimate (fraction); about 60 expected "
                "crossovers (informative meioses x genome Morgans) reach it."
            ),
        },
    )
    minimum_adoption_families: int = field(
        default=5,
        metadata={
            "provenance": (
                "Minimum informative families for infer_rate to adopt the estimate (count); structural: with fewer, "
                "between-family variation in the rate (sex, background) cannot be measured, whatever the interval."
            ),
        },
    )
    edge_low: float = field(
        default=0.21,
        metadata={"provenance": "An estimate at or below this lies at the low grid edge and falls back (cM/Mb)."},
    )
    edge_high: float = field(
        default=49.0,
        metadata={"provenance": "An estimate at or above this lies at the high grid edge and falls back (cM/Mb)."},
    )
    working_digits: int = field(
        default=3,
        metadata={
            "provenance": (
                "Significant digits of an adopted working rate (digits); far below the interval's width, so the "
                "working map's identity does not depend on last-bit arithmetic."
            ),
        },
    )


# Fixed numerical and workspace controls; not user settings.
LOG_ZERO = -1e300  # Log-domain maximum initializer (nats).
SCREEN_FLOOR = 1e-30  # Float32 site-product fallback threshold (probability).
SCALE_EXPONENT = 500  # Power-of-two product-rescaling exponent (binary exponent).
EVIDENCE_COMPARE_CHUNK = 256  # Early-exit evidence-comparison chunk width (values).
SCREEN_SITE_TILE = 256  # Storage-only site tile width (sites).

__all__ = ["AlleleFrequencyParameters", "FamilyParameters", "RateParameters"]
