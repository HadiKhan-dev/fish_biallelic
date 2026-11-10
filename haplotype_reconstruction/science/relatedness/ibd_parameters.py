"""Name the optional pairwise IBD chain parameters.

Likelihood thresholds are in nats, support intervals in Morgans, and probability controls are dimensionless.
Method: docs/methods/relatedness.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PairwiseIbdParameters:
    """Site thinning and genetic jump rate of the optional pair report."""

    ibd_site_step: int = field(
        default=10,
        metadata={
            "provenance": (
                "Unconditional site stride for the optional pairwise IBD report (sites); thins "
                "report evidence independently of family fitting."
            ),
        },
    )
    ibd_switch_rate: float = field(
        default=4.0,
        metadata={
            "provenance": (
                "IBD-chain jump rate along the genetic map (per Morgan); determines exp(-rate "
                "times distance) stay weights."
            ),
        },
    )
