"""Recombination map and junction-report constants.

User-visible: ``order_breakpoints`` (``[recombination] order_breakpoints``, a breakpoint TSV read by
stages.inputs.breakpoints.read_breakpoint_table). Recombination rates come from the chromosome's genetic map; crossovers
come from family phase's meiosis chains (release 3.0), whose confidence threshold is family phase's
``phase_posterior``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class JunctionParameters:
    """The reference-order junction detector (probabilities, nats).

    minimum_parents: parents with calls at a locus;
    hotspot_switch_probability: point-hotspot part of the recombination ceiling (no single locus gives a real
    crossover in more than 5% of meioses beyond the background rate); flag_log_likelihood_ratio: flag at 100:1
    (about 30 candidate loci per genome; a ratio reaches c under its null with probability about 1/c).
    """

    minimum_parents: int = field(
        default=3, metadata={"provenance": "RC03; 3.5 config.py minimum distinct parents", "units": "count"}
    )
    hotspot_switch_probability: float = field(
        default=0.05, metadata={"provenance": "RC03; 3.5 point-hotspot ceiling", "units": "probability"}
    )
    flag_log_likelihood_ratio: float = field(
        default=math.log(100.0), metadata={"provenance": "RC03; 3.5 100:1 flag odds", "units": "nats"}
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

    bin_bp: int = field(default=1_000_000, metadata={"provenance": "RC05; 3.5 fixed map-bin width", "units": "bp"})
    order_breakpoints: tuple[tuple[str, int, int, str], ...] = field(
        default=(), metadata={"provenance": "RC06; user-supplied breakpoint rows", "units": "bp"}
    )
    junctions: JunctionParameters = field(
        default_factory=JunctionParameters,
        metadata={"provenance": "RC03; junction hypothesis parameters", "units": "1"},
    )

    def __post_init__(self):
        if self.bin_bp <= 0:
            raise ValueError("the map bin width must be positive")
