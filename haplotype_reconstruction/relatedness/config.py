"""Relatedness stage settings.

Internal model constants (no user setting yet), documented here and in docs/configuration.md; the whole object is
bound into the stage's record identity. The stage reads only genotype likelihoods (no painting, pedigree or phase),
so its product can serve any later stage without a feedback loop.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelatednessConfig:
    """Pairwise identity by descent and full-sib families from genotype likelihoods.

    Attributes:
      minor_allele_frequency: sites enter only where the EM allele frequency (Hardy-Weinberg, all samples) gives
        a minor allele frequency of at least this value.
      minimum_observed: and where at least this share of samples has reads.
      family_site_step: the family model uses every this-many-th eligible site (sites are linked; the composite
        likelihood over thinned sites ranks family arrangements, and per-chromosome support measures stability).
      ibd_site_step: the pairwise IBD model uses every this-many-th evidence site (any frequency).
      ibd_switch_rate: the pairwise IBD chain leaves a state at this rate per Morgan, to the other allowed states
        equally.
      max_sweeps: at most this many search sweeps (parent assignment, single-sample moves, one merge) after the
        agglomeration.
      support_unit_morgans: support is measured on each map unit alone: a chromosome with a measured map length (a
        supplied genetic map, or a stated rate) is cut into max(1, floor(measured length / this)) consecutive units
        of equal measured length (an unlinked chromosome is one unit however short; a long one holds nearly
        independent stretches, as the pedigree's exposure gate counts); a chromosome with neither (fallback rate) is
        one unit.
      rate_is_measured: [genetic_map] rate_is_measured: the fallback rate is the organism's stated genome-wide
        average, so support units count map length from it where no supplied map covers a chromosome.
    """

    minor_allele_frequency: float = 0.05
    minimum_observed: float = 0.5
    family_site_step: int = 20
    ibd_site_step: int = 10
    ibd_switch_rate: float = 4.0
    max_sweeps: int = 30
    support_unit_morgans: float = 1.0
    rate_is_measured: bool = False

    def __post_init__(self):
        if not 0.0 <= self.minor_allele_frequency < 0.5:
            raise ValueError("relatedness.minor_allele_frequency must lie in [0, 0.5)")
        if not 0.0 <= self.minimum_observed <= 1.0:
            raise ValueError("relatedness.minimum_observed must lie in [0, 1]")
        for name in ("family_site_step", "ibd_site_step", "max_sweeps"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"relatedness.{name} must be a positive integer")
        if not self.ibd_switch_rate > 0:
            raise ValueError("relatedness.ibd_switch_rate must be positive")
        if not self.support_unit_morgans > 0:
            raise ValueError("relatedness.support_unit_morgans must be positive")
