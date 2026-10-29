"""Settings of the founder refit (a conditional correction of the founder alleles given the painting)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FounderRefitConfig:
    """Defaults are the production values.

    minimum_gain_nats: minimum gain per changed cell of each accepted move: a called
        allele changes only if the painted carriers' robust log likelihood improves by
        at least this much (5 nats: the assembled allele keeps prior odds of about 150
        to 1 against a single move); a two-row move (a swap of two rows' different
        alleles or a pair flip of their equal alleles) changes two cells and needs twice this.
    coordinate_sweeps: sweeps of single-row flips and two-row moves (swaps and pair flips) per site.
    fill_min_supporters: release 2.18: an allele the painted panel leaves unknown is called when one
        allele beats the other by ``minimum_gain_nats`` and at least this many painted carriers'
        own reads favour it (the same minimum as local selection's and discovery's ``min_supporters``).
    The genotype-error mixture is the painting's (``PaintingConfig.robustness_epsilon``), untempered:
    the refit is a per-site conditional correction given the painting's released labels, not the
    M-step of painting's tempered copying-model likelihood.
    """

    minimum_gain_nats: float = 5.0
    coordinate_sweeps: int = 3
    fill_min_supporters: int = 2
