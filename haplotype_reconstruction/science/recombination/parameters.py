"""Recombination map and junction-report constants.

User-visible: ``order_breakpoints`` (``[recombination] order_breakpoints``, a breakpoint TSV read by
stages.inputs.breakpoints.read_breakpoint_table). Recombination rates come from the chromosome's genetic map; crossovers
come from family phase's meiosis chains, whose confidence threshold is family phase's
``phase_posterior``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# RC06: (first probability, last probability, point count), inclusive and evenly spaced.
# q and 1-q have the same flip-adjusted likelihood, so the switch profile ends at 1/2.
SWITCH_GRID_SPEC = (0.001, 0.5, 500)
# A 0.5% discordance floor allows an isolated call error or crossover inside a displaced segment
# without erasing the pairing evidence; the upper endpoint and resolution define the fitted profile.
DISCORDANCE_GRID_SPEC = (0.005, 0.25, 50)


@dataclass(frozen=True)
class JunctionParameters:
    """The reference-order junction detector (probabilities, nats).

    minimum_parents: parents with calls at a locus;
    hotspot_switch_probability: point-hotspot part of the recombination ceiling (no single locus gives a real
    crossover in more than 5% of meioses beyond the background rate); flag_log_likelihood_ratio: flag at 100:1
    (about 30 candidate loci per genome; a ratio reaches c under its null with probability about 1/c).
    """

    minimum_parents: int = field(
        default=3,
        metadata={
            "provenance": "RC06; require at least 3 distinct called parents per diagnostic junction locus.",
            "units": "count",
        },
    )
    hotspot_switch_probability: float = field(
        default=0.05,
        metadata={
            "provenance": "RC06; point-hotspot crossover probability ceiling 0.05 beyond the background rate.",
            "units": "probability",
        },
    )
    flag_log_likelihood_ratio: float = field(
        default=math.log(100.0),
        metadata={
            "provenance": "RC06; diagnostic flag threshold log(100) nats represents a 100:1 likelihood ratio.",
            "units": "nats",
        },
    )

    def __post_init__(self):
        if self.minimum_parents < 2:
            raise ValueError("invalid junction parent threshold")
        if not 0 <= self.hotspot_switch_probability < 0.5 or not self.flag_log_likelihood_ratio > 0:
            raise ValueError("invalid junction hotspot ceiling or flag threshold")


@dataclass(frozen=True)
class RecombinationParameters:
    """Map aggregation of the meiosis chains.

    bin_bp: map bin width; order_breakpoints: (contig, start, end, reason) rows of the reference-order breakpoint
    table (user setting).
    """

    bin_bp: int = field(
        default=1_000_000,
        metadata={
            "provenance": "RC04-RC05; map-bin width 1000000 bp sets the physical aggregation resolution.",
            "units": "bp",
        },
    )
    order_breakpoints: tuple[tuple[str, int, int, str], ...] = field(
        default=(),
        metadata={
            "provenance": (
                "RC01-RC02; supplied reference-order breakpoint rows with closed bp endpoints; empty means no "
                "breakpoints."
            ),
            "units": "bp",
        },
    )
    junctions: JunctionParameters = field(
        default_factory=JunctionParameters,
        metadata={
            "provenance": (
                "RC06; internal diagnostic-junction parameter group; counts, probabilities and nats are declared "
                "by its fields."
            ),
            "units": "1",
        },
    )

    def __post_init__(self):
        if self.bin_bp <= 0:
            raise ValueError("the map bin width must be positive")
