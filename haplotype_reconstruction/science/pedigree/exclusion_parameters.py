"""Parameters for the full-marker pedigree exclusion test."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExclusionParameters:
    """Full-marker Mendelian exclusion releasing M0/M1 counts without new edges.

    4.22 A11: the per-site ratio is scored under the observation model at each sample's own outlier rate from the
    painting calibration (science.pedigree.exclusion._ratio); there is no genotype-replacement constant."""

    alpha: float = field(
        default=0.01,
        metadata={
            "provenance": "PD22; dimensionless exclusion level 0.01 sets the dyad log-e threshold log(2N/alpha).",
        },
    )
    bet_sizes: tuple[float, ...] = field(
        default=(1 / 1024, 1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0),
        metadata={
            "provenance": (
                "PD22; dimensionless bet fractions from 1/1024 through 1 average exclusion evidence at several "
                "strengths."
            ),
        },
    )
    # Half-overlapping physical windows (bp), plus the whole chromosome.
    window_widths_bp: tuple[int, ...] = field(
        default=(1000000, 5000000, 20000000),
        metadata={
            "provenance": (
                "PD22; physical window widths 1000000, 5000000 and 20000000 bp, half-overlapping plus the whole "
                "chromosome."
            ),
        },
    )
