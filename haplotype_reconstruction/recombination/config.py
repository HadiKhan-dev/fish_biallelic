"""Recombination model, shared-orientation and junction-report constants (defaults = base 92b3daf production).

User-visible: ``shared_family_evidence`` (``[recombination] shared_family_evidence``) and ``order_breakpoints``
(``[recombination] order_breakpoints``, a breakpoint TSV read by family_phase.breakpoints.read_breakpoint_table).
Recombination rates come from the chromosome's genetic map.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class SharedOrientationConfig:
    """Screened coordinate ascent of shared sample-orientation corrections (rates per physical bp).

    error_rate / error_mean_bp: asymmetric orientation-error CTMC prior of a sample's track orientation (entry
    rate per bp, mean tract length); maximum_candidate_bp: longest candidate tract; minimum_incident_edges:
    meioses a sample must share for its tracts to be candidates; maximum_sweeps: coordinate-ascent sweeps;
    minimum_objective_gain: nats a flip must gain; candidate_support: screening posterior that marks a
    supported run.
    """

    error_rate: float = 1e-8
    error_mean_bp: float = 50_000.
    maximum_candidate_bp: int = 1_000_000
    minimum_incident_edges: int = 2
    maximum_sweeps: int = 4
    minimum_objective_gain: float = 2.
    candidate_support: float = .9

    def __post_init__(self):
        if not (math.isfinite(self.error_rate) and self.error_rate > 0
                and math.isfinite(self.error_mean_bp) and self.error_mean_bp > 0):
            raise ValueError("shared orientation error rate and mean must be finite and positive")
        if (self.maximum_candidate_bp <= 0 or self.minimum_incident_edges < 2 or self.maximum_sweeps < 1
                or self.minimum_objective_gain < 0 or not .5 < self.candidate_support < 1):
            raise ValueError("invalid shared orientation screening controls")


@dataclass(frozen=True)
class JunctionConfig:
    """Bin-honest call intervals and the reference-order junction detector (bp, probabilities, nats).

    snapped_bin_minimum_bp: report the two bins around a straddled painting-bin edge when they span at least
    this many bp (0 widens every straddled call); minimum_parents: parents with calls at a locus;
    hotspot_switch_probability: point-hotspot part of the recombination ceiling (no single locus gives a real
    crossover in more than 5% of meioses beyond the background rate); flag_log_likelihood_ratio: flag at 100:1
    (about 30 candidate loci per genome; a ratio reaches c under its null with probability about 1/c).
    """

    snapped_bin_minimum_bp: int = 0
    minimum_parents: int = 3
    hotspot_switch_probability: float = .05
    flag_log_likelihood_ratio: float = math.log(100.)

    def __post_init__(self):
        if self.snapped_bin_minimum_bp < 0 or self.minimum_parents < 2:
            raise ValueError("invalid junction interval/parent thresholds")
        if not 0 <= self.hotspot_switch_probability < .5 or not self.flag_log_likelihood_ratio > 0:
            raise ValueError("invalid junction hotspot ceiling or flag threshold")


@dataclass(frozen=True)
class RecombinationConfig:
    """Per-meiosis decoding and map aggregation.

    copy_error: probability that a transmitted allele reads as the other parental homologue; artifact_rate /
    artifact_mean_bp: correlated orientation-artifact process of a meiosis (entry rate per bp, mean length);
    minimum_origin_probability: posterior support of a marker's parental homologue (and of no artifact) for a
    called crossover; maximum_gap_bp: informative markers further apart restart the decoder; bin_bp: map bin
    width; read_free_tract_sites: runs of at least this many zero-depth sites of a sample are uncovered for it
    (its calls there are imputed; 1000 sites = ten 100-SNP painting bins; chance runs are <= 15 sites in the 5x
    simulations, <= 604 in real tromau3 data; 0 disables); shared_family_evidence: fit shared sample-orientation
    corrections before decoding (user setting); order_breakpoints: (contig, start, end, reason) rows of the
    reference-order breakpoint table (user setting).
    """

    copy_error: float = .01
    artifact_rate: float = 1e-8
    artifact_mean_bp: float = 50_000.
    minimum_origin_probability: float = .98
    maximum_gap_bp: int = 1_000_000
    bin_bp: int = 1_000_000
    read_free_tract_sites: int = 1000
    shared_family_evidence: bool = True
    order_breakpoints: tuple[tuple[str, int, int, str], ...] = ()
    shared_orientation: SharedOrientationConfig = field(default_factory=SharedOrientationConfig)
    junctions: JunctionConfig = field(default_factory=JunctionConfig)

    def __post_init__(self):
        if not 0 < self.copy_error < .5 or not .5 < self.minimum_origin_probability < 1:
            raise ValueError("invalid copy-error/origin-support probability")
        if (not math.isfinite(self.artifact_rate) or self.artifact_rate < 0
                or not math.isfinite(self.artifact_mean_bp) or self.artifact_mean_bp <= 0):
            raise ValueError("invalid correlated orientation-artifact process")
        if self.maximum_gap_bp <= 0 or self.bin_bp <= 0 or self.read_free_tract_sites < 0:
            raise ValueError("gap and bin widths must be positive and read_free_tract_sites nonnegative")
