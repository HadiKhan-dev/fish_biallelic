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
# 4.21 acceptance (ADVICE2 section 12): a called cell's prior error p_c is estimated per contig and per carrier class
# (scoring carriers at the painted state); these declare the classes and the estimator's grid, not tuned values.
CARRIER_CLASS_EDGES = (2, 3, 5, 9, 17, 33)
"""Lower edges of the carrier classes 2, 3-4, 5-8, 9-16, 17-32, 33+ (one carrier is class 0: never changed, since
one carrier cannot separate a wrong allele from a sample that does not carry the row there)."""
MINIMUM_CLASS_CELLS = 4096
"""A carrier class with fewer start-state cells pools with the next class of more carriers (the last with the one
before), so every coefficient rests on thousands of cells; a declared constant, not an estimate."""
GAIN_BIN_WIDTH = 0.05
"""Width (nats) of the start-state gain histogram the prior is estimated from: integer counts, the same on any
thread count; 0.05 nats moves a threshold of 5 to 15 nats by far less than its sampling error."""
GAIN_LIMIT = 60.0
"""Gains beyond +-60 nats fall in the end bins (ADVICE2 15.1 item 2): the likelihood of c is flat in G there (exp(-60)
is negligible, and for a large positive G the term is G + log c whatever G)."""


@dataclass(frozen=True)
class RefitParameters:
    """Conditional founder correction given the released painting.

    The acceptance rule (4.21, ADVICE2 section 12): a called cell of carrier class c changes when the move's
    likelihood gain exceeds log((1 - p_c) / p_c), p_c the estimated per-contig prior that a called cell of that class
    is wrong (science.refit.refit.class_priors; the sum of two thresholds for a two-row move), and an unknown cell is
    filled when its posterior log odds against the cohort's allele frequency reach the same threshold.
    coordinate_sweeps: sweeps of single-row flips and two-row moves (swaps and pair flips) per site (a guard: the
        rule raises a fixed log posterior, so it cannot cycle; sites still moving at the last sweep are reported).
    min_supporters: every change (flip, two-row move, fill) needs at least this many painted carriers whose own
        reads favour it (the same minimum as local selection's and discovery's ``min_supporters``).
    The genotype-error mixture is the painting's (``Painting.robustness_epsilon``), untempered:
    the refit is a per-site conditional correction given the painting's released labels, not the
    M-step of painting's tempered copying-model likelihood.
    """

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
    min_supporters: int = field(
        default=2,
        metadata={
            "provenance": (
                "Minimum painted carriers whose own reads favour a change (samples): a declared identification "
                "rule (ADVICE2 12.2): with one carrier a wrong allele and a sample that does not carry the row are "
                "the same observation."
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
    "CARRIER_CLASS_EDGES",
    "MINIMUM_CLASS_CELLS",
    "GAIN_BIN_WIDTH",
    "GAIN_LIMIT",
]
