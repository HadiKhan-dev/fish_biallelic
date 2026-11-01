"""Name the preserved founder-allele search, release and numerical constants.

Owns the typed settings-to-kernel adapter (P0 U08-U10; WP_AS2 sections 5 and 7).
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

# U08 direct positive-sum cancellation guard; baseline polynomials.py:145.
POLYNOMIAL_CANCELLATION = 1e-10
# U09 positive likelihood floor; baseline site_posterior.py:125.
TINY_LIKELIHOOD = 1e-300
# U10 robust evidence mixture and strict uniform-range test; baseline :163-164.
FEEDBACK_EVIDENCE_KEEP = 0.99
FEEDBACK_EVIDENCE_MIX = 0.01
FEEDBACK_UNIFORM_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SitePosteriorParameters:
    """Frozen U09/U10 settings, with no mutation or implicit convention choice.

    vector returns a fresh float64[13] in the baseline SETTING_* index order.
    The explicit reverse flag changes only the three arbitrary conventions.
    """

    max_exact_bits: int = field(
        default=MAX_EXACT_BITS,
        metadata={"provenance": "P0 assembly-core U09/U10; baseline founder_alleles/site_posterior.py:86-104"},
    )
    certain_nats: float = field(
        default=SITE_CERTAIN_NATS,
        metadata={"provenance": "P0 assembly-core U09/U10; baseline founder_alleles/site_posterior.py:86-104"},
    )
    ball_configurations: int = field(
        default=SITE_BALL_CONFIGURATIONS,
        metadata={"provenance": "P0 assembly-core U09/U10; baseline founder_alleles/site_posterior.py:86-104"},
    )
    ball_minimum_radius: int = field(
        default=SITE_BALL_MINIMUM_RADIUS,
        metadata={"provenance": "P0 assembly-core U09/U10; baseline founder_alleles/site_posterior.py:86-104"},
    )
    shell_guard: float = field(
        default=SITE_SHELL_GUARD,
        metadata={"provenance": "P0 assembly-core U09/U10; baseline founder_alleles/site_posterior.py:86-104"},
    )
    bp_damping: float = field(
        default=SITE_BP_DAMPING,
        metadata={"provenance": "P0 assembly-core U09/U10; baseline founder_alleles/site_posterior.py:86-104"},
    )
    bp_tolerance: float = field(
        default=SITE_BP_TOLERANCE,
        metadata={"provenance": "P0 assembly-core U09/U10; baseline founder_alleles/site_posterior.py:86-104"},
    )
    bp_iterations: int = field(
        default=SITE_BP_ITERATIONS,
        metadata={"provenance": "P0 assembly-core U09/U10; baseline founder_alleles/site_posterior.py:86-104"},
    )
    ascent_backstop: int = field(
        default=SITE_ASCENT_BACKSTOP,
        metadata={"provenance": "P0 assembly-core U09/U10; baseline founder_alleles/site_posterior.py:86-104"},
    )
    recentre_backstop: int = field(
        default=SITE_RECENTRE_BACKSTOP,
        metadata={"provenance": "P0 assembly-core U09/U10; baseline founder_alleles/site_posterior.py:86-104"},
    )
    improvement: float = field(
        default=SITE_IMPROVEMENT,
        metadata={"provenance": "P0 assembly-core U09/U10; baseline founder_alleles/site_posterior.py:86-104"},
    )
    half_band: float = field(
        default=SITE_HALF_BAND,
        metadata={"provenance": "P0 assembly-core U09/U10; baseline founder_alleles/site_posterior.py:86-104"},
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
