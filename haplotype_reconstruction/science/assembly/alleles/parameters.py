"""Name the founder-allele search, release and numerical constants.

Owns founder-allele settings and their typed kernel-vector adapter (U08-U10).
Must not choose a search route, read an environment or perform I/O.
Method: docs/methods/assembly.md#founder-allele-posteriors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from haplotype_reconstruction.core.ties import fixed_policy

MAX_EXACT_BITS = 10
SITE_CERTAIN_NATS = 20.0
SITE_BALL_CONFIGURATIONS = 1024
SITE_BALL_MINIMUM_RADIUS = 3
SITE_SHELL_GUARD = 3.0
SITE_BP_DAMPING = 0.5
SITE_BP_TOLERANCE = 1e-09
SITE_BP_ITERATIONS = 500
SITE_ASCENT_BACKSTOP = 50
SITE_RECENTRE_BACKSTOP = 50
SITE_IMPROVEMENT = 2.0 ** (-10)
SITE_HALF_BAND = 2.0 ** (-20)

# Direct positive-sum fallback when polynomial cancellation exceeds this relative guard.
POLYNOMIAL_CANCELLATION = 1e-10
# Positive likelihood floor, preventing log(0) in site posteriors.
TINY_LIKELIHOOD = 1e-300
# Robust evidence mixture weights and strict uniform-range tolerance for context feedback.
FEEDBACK_EVIDENCE_KEEP = 0.99
FEEDBACK_EVIDENCE_MIX = 0.01
FEEDBACK_UNIFORM_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SitePosteriorParameters:
    """Frozen U09/U10 settings, with no mutation or implicit convention choice.

    vector returns a fresh float64[13] in SETTING_* index order.
    The explicit reverse flag changes only the three arbitrary conventions.
    """

    max_exact_bits: int = field(
        default=MAX_EXACT_BITS,
        metadata={
            "units": "free founder alleles",
            "provenance": "Enumerate the complete cube through 10 unresolved bits; larger sites use core support.",
        },
    )
    certain_nats: float = field(
        default=SITE_CERTAIN_NATS,
        metadata={
            "units": "nats",
            "provenance": "A certain founder must resist single and pair flips by more than 20 log-posterior units.",
        },
    )
    ball_configurations: int = field(
        default=SITE_BALL_CONFIGURATIONS,
        metadata={
            "units": "configurations",
            "provenance": "Target at most 1024 configurations when sizing the uncertain-core Hamming ball.",
        },
    )
    ball_minimum_radius: int = field(
        default=SITE_BALL_MINIMUM_RADIUS,
        metadata={
            "units": "flipped alleles",
            "provenance": "Use radius at least 3 even when the target configuration count would suggest less.",
        },
    )
    shell_guard: float = field(
        default=SITE_SHELL_GUARD,
        metadata={
            "units": "dimensionless",
            "provenance": "Multiply outer-shell flip mass by 3 to guard releases near their posterior threshold.",
        },
    )
    bp_damping: float = field(
        default=SITE_BP_DAMPING,
        metadata={"units": "dimensionless", "provenance": "Belief-propagation update damping weight 0.5."},
    )
    bp_tolerance: float = field(
        default=SITE_BP_TOLERANCE,
        metadata={
            "units": "nats of message log odds",
            "provenance": "Stop belief propagation when message changes reach tolerance 1e-9.",
        },
    )
    bp_iterations: int = field(
        default=SITE_BP_ITERATIONS,
        metadata={
            "units": "iterations",
            "provenance": "Maximum 500 damped belief-propagation iterations for the mode start.",
        },
    )
    ascent_backstop: int = field(
        default=SITE_ASCENT_BACKSTOP,
        metadata={"units": "steps", "provenance": "Report a backstop after 50 single/pair-flip ascent steps."},
    )
    recentre_backstop: int = field(
        default=SITE_RECENTRE_BACKSTOP,
        metadata={"units": "steps", "provenance": "Report a backstop after 50 uncertain-core ball recentrings."},
    )
    improvement: float = field(
        default=SITE_IMPROVEMENT,
        metadata={
            "units": "nats",
            "provenance": "Require log-posterior gain above 2^-10; the same band groups nearly equal ascent moves.",
        },
    )
    half_band: float = field(
        default=SITE_HALF_BAND,
        metadata={
            "units": "absolute probability",
            "provenance": "Exclude carrier probabilities within 2^-20 of one half from the confident carrier count.",
        },
    )

    def vector(self, reverse: bool) -> NDArray[np.float64]:
        """Return a fresh kernel vector; inputs are unchanged and no state is read."""
        convention = fixed_policy("assembly.alleles.ascent", self.improvement, reverse=reverse)
        return np.array(
            [
                self.max_exact_bits,
                self.certain_nats,
                self.ball_configurations,
                self.ball_minimum_radius,
                self.shell_guard,
                self.bp_damping,
                self.bp_tolerance,
                self.bp_iterations,
                self.ascent_backstop,
                self.recentre_backstop,
                self.improvement,
                self.half_band,
                1.0 if convention.reverse else 0.0,
            ],
            dtype=np.float64,
        )


__all__ = [
    "SitePosteriorParameters",
    "MAX_EXACT_BITS",
    "SITE_CERTAIN_NATS",
    "SITE_BALL_CONFIGURATIONS",
    "SITE_BALL_MINIMUM_RADIUS",
    "SITE_SHELL_GUARD",
    "SITE_BP_DAMPING",
    "SITE_BP_TOLERANCE",
    "SITE_BP_ITERATIONS",
    "SITE_ASCENT_BACKSTOP",
    "SITE_RECENTRE_BACKSTOP",
    "SITE_IMPROVEMENT",
    "SITE_HALF_BAND",
    "POLYNOMIAL_CANCELLATION",
    "TINY_LIKELIHOOD",
    "FEEDBACK_EVIDENCE_KEEP",
    "FEEDBACK_EVIDENCE_MIX",
    "FEEDBACK_UNIFORM_TOLERANCE",
]
