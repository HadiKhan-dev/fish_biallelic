"""Scientific settings of the run's observation pilot (science.observation): which blocks are fitted, the start of
the strict fits' rate and the estimator's grid.

Method: docs/methods/observation.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservationParameters:
    """The pilot that estimates the run's genotype outlier rate before discovery.

    pilot_share: the share of each analysed contig's blocks fitted by the pilot (chosen by a hash of the contig and
        block index, so the choice is reproducible and independent of run order).
    minimum_contig_blocks, minimum_run_blocks, maximum_run_blocks: bounds on the pilot's blocks (every block when a
        contig or run has fewer).
    strict_outlier_rate: the outlier rate of the pilot's strict fits, from which the run's rate is estimated (the
        numerical lower bound: a fit at a tolerant rate leaves its own discordance in the cells and reads it back;
        the estimate is never fed into the fits that make it; ADVICE2 section 8, addendum of 16 November 2026).
    minimum_unit_cells: a (sample, block) unit enters the two-level estimate with at least this many observed cells of
        known founder states.
    grid_points, grid_low, grid_high: the log grid of outlier rates on which unit likelihoods are evaluated.
    """

    pilot_share: float = field(
        default=0.01,
        metadata={"units": "share of blocks", "provenance": "About 1 % of every analysed contig's blocks."},
    )
    minimum_contig_blocks: int = field(
        default=4, metadata={"units": "blocks", "provenance": "At least 4 pilot blocks per analysed contig."}
    )
    minimum_run_blocks: int = field(
        default=256,
        metadata={
            "units": "blocks",
            "provenance": (
                "At least 256 pilot blocks per run where it has them (ADVICE2 section 8, note of 09:12; sized for the "
                "removed scale test and kept for the rate: its strict fits are a small share of discovery's cost)."
            ),
        },
    )
    maximum_run_blocks: int = field(
        default=512, metadata={"units": "blocks", "provenance": "At most 512 pilot blocks per run (cost bound)."}
    )
    strict_outlier_rate: float = field(
        default=1e-6,
        metadata={"units": "probability", "provenance": "Strict fits (ADVICE2 section 8, addendum: M17, M18)."},
    )
    minimum_unit_cells: int = field(
        default=50,
        metadata={"units": "cells", "provenance": "A (sample, block) unit needs 50 cells of known states."},
    )
    grid_points: int = field(
        default=31, metadata={"units": "points", "provenance": "31 log-spaced rates from grid_low to grid_high."}
    )
    grid_low: float = field(default=1e-7, metadata={"units": "probability", "provenance": "Lowest rate of the grid."})
    grid_high: float = field(default=0.3, metadata={"units": "probability", "provenance": "Highest rate of the grid."})


__all__ = ["ObservationParameters"]
