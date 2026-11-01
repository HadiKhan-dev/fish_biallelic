"""Settings of the founder refit (a conditional correction of the founder alleles given the painting)."""

from __future__ import annotations

from dataclasses import dataclass, field

# 3.5 founder_refit/refit.py:59–70; carrier-scaled gain relation (P0 FR03).
REFIT_TIE_QUANTUM = 2.0**-16
REFIT_TIE_QUANTUM_PER_CARRIER = 2.0**-18
WIDE_QUANTA = 4.0
# 3.5 founder_refit/refit.py:57; sites per independent parallel task (P0 FR01).
SITE_TILE = 1024


@dataclass(frozen=True)
class RefitParameters:
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

    minimum_gain_nats: float = field(
        default=5.0,
        metadata={
            "provenance": "3.5 founder_refit/config.py: minimum_gain_nats; P0 FR02/FR05; WP_RF1 §7",
            "role": "state",
            "axes": (),
            "units": "nats",
            "dtype": "float64",
            "stored": True,
        },
    )
    coordinate_sweeps: int = field(
        default=3,
        metadata={
            "provenance": "3.5 founder_refit/config.py: coordinate_sweeps; P0 FR02/FR05; WP_RF1 §7",
            "role": "state",
            "axes": (),
            "units": "1",
            "dtype": "int64",
            "stored": True,
        },
    )
    fill_min_supporters: int = field(
        default=2,
        metadata={
            "provenance": "3.5 founder_refit/config.py: fill_min_supporters; P0 FR02/FR05; WP_RF1 §7",
            "role": "state",
            "axes": (),
            "units": "1",
            "dtype": "int64",
            "stored": True,
        },
    )


__all__ = ["RefitParameters", "REFIT_TIE_QUANTUM", "REFIT_TIE_QUANTUM_PER_CARRIER", "WIDE_QUANTA", "SITE_TILE"]
