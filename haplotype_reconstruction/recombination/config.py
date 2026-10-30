"""Recombination map and junction-report constants.

User-visible: ``order_breakpoints`` (``[recombination] order_breakpoints``, a breakpoint TSV read by
family_phase.breakpoints.read_breakpoint_table). Recombination rates come from the chromosome's genetic map; crossovers
come from family phase's meiosis chains (release 3.0), whose confidence threshold is family phase's
``phase_posterior``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class JunctionConfig:
    """The reference-order junction detector (probabilities, nats).

    minimum_parents: parents with calls at a locus;
    hotspot_switch_probability: point-hotspot part of the recombination ceiling (no single locus gives a real
    crossover in more than 5% of meioses beyond the background rate); flag_log_likelihood_ratio: flag at 100:1
    (about 30 candidate loci per genome; a ratio reaches c under its null with probability about 1/c).
    """

    minimum_parents: int = 3
    hotspot_switch_probability: float = .05
    flag_log_likelihood_ratio: float = math.log(100.)

    def __post_init__(self):
        if self.minimum_parents < 2:
            raise ValueError("invalid junction parent threshold")
        if not 0 <= self.hotspot_switch_probability < .5 or not self.flag_log_likelihood_ratio > 0:
            raise ValueError("invalid junction hotspot ceiling or flag threshold")


@dataclass(frozen=True)
class RecombinationConfig:
    """Map aggregation of the meiosis chains.

    bin_bp: map bin width; order_breakpoints: (contig, start, end, reason) rows of the reference-order breakpoint
    table (user setting).
    """

    bin_bp: int = 1_000_000
    order_breakpoints: tuple[tuple[str, int, int, str], ...] = ()
    junctions: JunctionConfig = field(default_factory=JunctionConfig)

    def __post_init__(self):
        if self.bin_bp <= 0:
            raise ValueError("the map bin width must be positive")
