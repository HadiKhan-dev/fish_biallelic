"""Settings of the founder refit (a conditional correction of the founder alleles given the painting)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.ties import WIDE_CLASS_QUANTA as WIDE_QUANTA

# Move-gain tie quantum (nats), carrier-scaled quantum (nats per carrier) and class width (quanta).
REFIT_TIE_QUANTUM = 2.0**-16
REFIT_TIE_QUANTUM_PER_CARRIER = 2.0**-18
# Sites per independent parallel task; each task owns one founder-pair table.
SITE_TILE = 1024
# Maximum keys for stable comparison sort (keys); depth <=12 beats eight radix passes here.
# Larger arrays use fixed-width radix sorting to keep round ordering linear in candidate count.
STABLE_SORT_COMPARISON_LIMIT = 4096


@dataclass(frozen=True)
class RefitParameters:
    """Conditional founder correction given the released painting.

    minimum_gain_nats: minimum gain per changed cell of each accepted move: a called
        allele changes only if the painted carriers' robust log likelihood improves by
        at least this much (a per-move hysteresis threshold); a two-row move (a swap of two rows' different
        alleles or a pair flip of their equal alleles) changes two cells and needs twice this.
    coordinate_sweeps: sweeps of single-row flips and two-row moves (swaps and pair flips) per site.
    fill_min_supporters: an allele the painted panel leaves unknown is called when one
        allele beats the other by ``minimum_gain_nats`` and at least this many painted carriers'
        own reads favour it (the same minimum as local selection's and discovery's ``min_supporters``).
    The genotype-error mixture is the painting's (``PaintingParameters.robustness_epsilon``), untempered:
    the refit is a per-site conditional correction given the painting's released labels, not the
    M-step of painting's tempered copying-model likelihood.
    """

    minimum_gain_nats: float = field(
        default=5.0,
        metadata={
            "provenance": (
                "Minimum robust likelihood gain per changed allele cell (nats per cell); provides "
                "move hysteresis, with twice the gain required for a two-row move."
            ),
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
            "provenance": (
                "Maximum single-row and two-row coordinate sweeps per site (sweeps); ends sooner "
                "if a sweep changes nothing."
            ),
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
            "provenance": (
                "Minimum painted carriers whose own reads favour a filled allele (samples); "
                "supplements the likelihood-gain requirement for an unknown cell."
            ),
            "role": "state",
            "axes": (),
            "units": "1",
            "dtype": "int64",
            "stored": True,
        },
    )


__all__ = [
    "RefitParameters",
    "REFIT_TIE_QUANTUM",
    "REFIT_TIE_QUANTUM_PER_CARRIER",
    "WIDE_QUANTA",
    "SITE_TILE",
    "STABLE_SORT_COMPARISON_LIMIT",
]
