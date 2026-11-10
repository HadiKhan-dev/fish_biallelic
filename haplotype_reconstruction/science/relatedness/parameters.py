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
                "Target genetic length of support units (Morgans); partitions chromosomes for "
                "member and parent stability summaries."
            ),
        },
    )
    rate_is_measured: bool = field(
        default=False,
        metadata={
            "provenance": (
                "Treat fallback map rates as measured support distance (boolean); otherwise clip "
                "to measured map bounds and use zero length when no map exists."
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


# Fixed numerical and workspace controls; not user settings.
LOG_ZERO = -1e300  # Log-domain maximum initializer (nats).
SCREEN_FLOOR = 1e-30  # Float32 site-product fallback threshold (probability).
SCALE_EXPONENT = 500  # Power-of-two product-rescaling exponent (binary exponent).
EVIDENCE_COMPARE_CHUNK = 256  # Early-exit evidence-comparison chunk width (values).
SCREEN_SITE_TILE = 256  # Storage-only site tile width (sites).

__all__ = ["AlleleFrequencyParameters", "FamilyParameters"]
