"""Settings of the founder refit (one conditional M-step of the founder alleles)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FounderRefitConfig:
    """Defaults are the production values.

    minimum_gain_nats: prior per allele: a called allele changes only if the
        painted carriers' robust log likelihood improves by at least this much
        (5 nats: the assembled allele keeps prior odds of about 150 to 1); a
        swap of two rows' alleles changes two cells and needs twice this.
    coordinate_sweeps: sweeps of single-row flips and two-row swaps per site.
    The genotype-error mixture is the painting's (``PaintingConfig.robustness_epsilon``):
    the refit maximises the painting's own emission given the painted rows.
    """

    minimum_gain_nats: float = 5.0
    coordinate_sweeps: int = 3
