"""Fixed numerical tolerances for pedigree inference."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WinnerNumerics:
    """Exact winner and graph-confidence tolerances; no user-facing tuning (PD11-PD13)."""

    contrast_ulp_factor: float = field(
        default=512.0,
        metadata={
            "provenance": "PD11; 512 machine-epsilon units times score magnitude define the rounding tie band.",
        },
    )
    noise_floor: float = field(
        default=2.0**-15,
        metadata={
            "provenance": (
                "PD11; absolute tie floor of 2^-15 nats prevents small score perturbations selecting a winner."
            ),
        },
    )
    minimum_graph_quantum: float = field(
        default=1e-300,
        metadata={
            "provenance": "PD13; positive graph-confidence tie quantum floor of 1e-300 nats.",
        },
    )


# Fixed model numerics are versioned by source identity, not additional user settings.
WINNER_NUMERICS = WinnerNumerics()

@dataclass(frozen=True)
class ScoringNumerics:
    """Fixed preparation and scoring numerical cutoffs, named independently of user-facing settings."""

    informative_gl_spread: float = field(
        default=1e-12,
        metadata={
            "provenance": (
                "PS01; genotype-likelihood max-minus-min spread must exceed 1e-12 to count a marker as "
                "informative."
            ),
        },
    )
    transmission_switch_floor: float = field(
        default=1e-12,
        metadata={
            "provenance": (
                "PS19; dimensionless probability floor 1e-12 keeps the scheduled joint transmission "
                "and founder-mosaic log transitions finite at equal map positions."
            ),
        },
    )
    selector_switch_floor: float = field(
        default=1e-15,
        metadata={
            "provenance": "PS04; probability floor 1e-15 keeps selector switch costs finite at equal map positions.",
        },
    )
    hall_ulps: float = field(
        default=64.0,
        metadata={
            "provenance": "PS06; 64 scaled machine-epsilon units identify bridge Hall equality up to rounding.",
        },
    )
    hub_mass_tie: float = field(
        default=2.0**-20,
        metadata={
            "provenance": "PS07; absolute probability-mass tie band 2^-20 selects a stable Hall hub among tied loads.",
        },
    )
    site_polynomial_threshold: float = field(
        default=1e-8,
        metadata={
            "provenance": (
                "PS09; dimensionless likelihood threshold 1e-8 selects direct emission evaluation to avoid "
                "polynomial cancellation."
            ),
        },
    )
    log_floor: float = field(
        default=-60.0,
        metadata={
            "provenance": "PS11; log-relative M1 score floor of -60 nats for nonretained parent identities.",
        },
    )

    tiny: float = field(
        default=1e-280,
        metadata={
            "provenance": (
                "PS15; positive mass threshold 1e-280 selects probability-domain rather than log-domain "
                "pair-bound absorption."
            ),
        },
    )
    family_exp_underflow: float = field(
        default=750.0,
        metadata={
            "provenance": "PD09; magnitude cutoff 750 nats skips exponentials whose float64 contributions vanish.",
        },
    )
    order_minimum_unit_weight: float = field(
        default=2.0,
        metadata={
            "provenance": (
                "PD07; require more than 2 callability-weighted unit comparisons (one per material unit since "
                "4.18) for graph-ordering support."
            ),
        },
    )
    order_variance_offset: float = field(
        default=1.0,
        metadata={
            "provenance": (
                "PD07; additive variance offset 1 in squared junction-count units for graph-ordering support."
            ),
        },
    )
    direction_reliability_grid: int = field(
        default=40,
        metadata={
            "provenance": (
                "PD06; number of midpoints of (0, 1) on which the posterior of the run's direction reliability is "
                "summed (spacing 0.025; the posterior mean moves by less than 0.005 against 80 midpoints on the "
                "calibration layouts). Numerical resolution, not a model constant."
            ),
        },
    )
    direction_coin_share_grid: int = field(
        default=100,
        metadata={
            "provenance": (
                "PD06; number of midpoints of (0, 1) on which the posterior of the run's coin share is summed "
                "(spacing 0.01, so the smallest representable share is 0.005). Numerical resolution, not a model "
                "constant."
            ),
        },
    )
    junction_unreachable_state: int = field(
        default=1 << 30,
        metadata={
            "provenance": "PS18; integer cost sentinel 2^30 marks an unreachable minimum-switch trajectory state.",
        },
    )
    junction_unreachable_count: int = field(
        default=1 << 28,
        metadata={
            "provenance": "PS18; integer switch-count sentinel 2^28 marks unavailable junction counts.",
        },
    )

    null_rounding_ulps: float = field(
        default=64.0,
        metadata={
            "provenance": "PS09; 64 scaled machine-epsilon units tolerate rounding in null-source persistence.",
        },
    )

    bridge_feasibility_ulps: float = field(
        default=64.0,
        metadata={
            "provenance": "PS07; 64 scaled machine-epsilon units tolerate bridge flow feasibility rounding.",
        },
    )

    bridge_post_tolerance_factor: float = field(
        default=8.0,
        metadata={
            "provenance": "PS07; dimensionless multiplier 8 expands the configured bridge post-check tolerance.",
        },
    )

    bridge_post_ulps: float = field(
        default=256.0,
        metadata={
            "provenance": "PS07; 256 machine-epsilon units set the minimum bridge post-check tolerance.",
        },
    )

    bridge_bisection_limit: int = field(
        default=128,
        metadata={
            "provenance": "PS07; hard ceiling of 128 bisection halvings in the maximum-entropy bridge solve.",
        },
    )

    bridge_fold_discriminant: float = field(
        default=2.0**-26,
        metadata={
            "provenance": (
                "PS07; sqrt(machine epsilon): an all-small root whose fold state has 1 - q L at or below it takes "
                "that state's diagonal product from the root equation, not from its square-root-folded quadratic."
            ),
        },
    )

    certificate_gl_ulps: int = field(
        default=64,
        metadata={
            "provenance": (
                "PS22; 64 scaled machine-epsilon units pad common genotype-likelihood logs outward in "
                "certificates."
            ),
        },
    )

    selective_upper_ulps: int = field(
        default=4096,
        metadata={
            "provenance": "PS22; 4096 scaled machine-epsilon units pad selective M1 upper bounds outward.",
        },
    )

    selective_upper_state_factor: int = field(
        default=64,
        metadata={
            "provenance": (
                "PS22; 64 times the state count supplies additional machine-epsilon padding for selective upper "
                "bounds."
            ),
        },
    )

    selective_bound_ulps: int = field(
        default=128,
        metadata={
            "provenance": "PS22; 128 scaled machine-epsilon units guard selective-bound coefficients outward.",
        },
    )

    selective_sum_ulps: int = field(
        default=8,
        metadata={
            "provenance": (
                "PS22; 8 machine-epsilon units per ordered accumulation term guard the selective chromosome sum."
            ),
        },
    )

    selective_frontier_ulps: int = field(
        default=32,
        metadata={
            "provenance": "PS22; 32 scaled machine-epsilon units guard best and floor frontiers in selective search.",
        },
    )

    pair_null_relative_guard: float = field(
        default=1e-12,
        metadata={
            "provenance": (
                "PS15; dimensionless relative padding 1e-12 inflates null-row maxima for conservative pair "
                "bounds."
            ),
        },
    )

    exclusion_tiny: float = field(
        default=1e-300,
        metadata={
            "provenance": (
                "PS20; positive probability floor 1e-300 protects exclusion ratio denominators and bet "
                "logarithms."
            ),
        },
    )


SCORING_NUMERICS = ScoringNumerics()
INFORMATIVE_GL_SPREAD = SCORING_NUMERICS.informative_gl_spread
SELECTOR_SWITCH_FLOOR = SCORING_NUMERICS.selector_switch_floor
TRANSMISSION_SWITCH_FLOOR = SCORING_NUMERICS.transmission_switch_floor
HALL_ULPS = SCORING_NUMERICS.hall_ulps
HUB_MASS_TIE = SCORING_NUMERICS.hub_mass_tie
SITE_POLYNOMIAL_THRESHOLD = SCORING_NUMERICS.site_polynomial_threshold
LOG_FLOOR = SCORING_NUMERICS.log_floor

TINY = SCORING_NUMERICS.tiny
FAMILY_EXP_UNDERFLOW = SCORING_NUMERICS.family_exp_underflow
ORDER_MINIMUM_UNIT_WEIGHT = SCORING_NUMERICS.order_minimum_unit_weight
ORDER_VARIANCE_OFFSET = SCORING_NUMERICS.order_variance_offset
DIRECTION_RELIABILITY_GRID = SCORING_NUMERICS.direction_reliability_grid
DIRECTION_COIN_SHARE_GRID = SCORING_NUMERICS.direction_coin_share_grid
JUNCTION_UNREACHABLE_STATE = SCORING_NUMERICS.junction_unreachable_state
JUNCTION_UNREACHABLE_COUNT = SCORING_NUMERICS.junction_unreachable_count

NULL_ROUNDING_ULPS = SCORING_NUMERICS.null_rounding_ulps
BRIDGE_FEASIBILITY_ULPS = SCORING_NUMERICS.bridge_feasibility_ulps
BRIDGE_POST_TOLERANCE_FACTOR = SCORING_NUMERICS.bridge_post_tolerance_factor
BRIDGE_POST_ULPS = SCORING_NUMERICS.bridge_post_ulps
BRIDGE_BISECTION_LIMIT = SCORING_NUMERICS.bridge_bisection_limit
BRIDGE_FOLD_DISCRIMINANT = SCORING_NUMERICS.bridge_fold_discriminant
CERTIFICATE_GL_ULPS = SCORING_NUMERICS.certificate_gl_ulps
SELECTIVE_UPPER_ULPS = SCORING_NUMERICS.selective_upper_ulps
SELECTIVE_UPPER_STATE_FACTOR = SCORING_NUMERICS.selective_upper_state_factor
SELECTIVE_BOUND_ULPS = SCORING_NUMERICS.selective_bound_ulps
SELECTIVE_SUM_ULPS = SCORING_NUMERICS.selective_sum_ulps
SELECTIVE_FRONTIER_ULPS = SCORING_NUMERICS.selective_frontier_ulps
PAIR_NULL_RELATIVE_GUARD = SCORING_NUMERICS.pair_null_relative_guard
EXCLUSION_TINY = SCORING_NUMERICS.exclusion_tiny
