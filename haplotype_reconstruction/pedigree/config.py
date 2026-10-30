"""Pedigree stage settings.

Every default is the production value of 92b3daf. Only ``evidence_calibration`` is a user setting
(TOML ``[pedigree] evidence_calibration``); the groups below are internal model constants, documented
here and in docs/configuration.md. Each group is bound into the identity of exactly the records it
affects: ``preparation`` into prepared factors, ``scoring`` into screens and scores, ``panel`` into the
M2 candidate panel, the rest into the genome-wide decision.

Thresholds label evidence tiers; they are not biological acceptance probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field

EVIDENCE_CALIBRATION_MODES = ("off", "predictive")


@dataclass(frozen=True)
class PreparationConfig:
    """Marker subsampling and information tempering of the prepared painting components."""

    # At most this many evenly spaced selected markers per painting bin enter the transmission model.
    markers_per_bin: int = 10
    # Chromosome-wide groups of at least this many selected markers share one information exponent.
    markers_per_information_block: int = 100
    # The tempered view raises a bin's emission to min(this, n) / n for n informative markers in its group.
    effective_markers_per_information_block: float = 3.0


@dataclass(frozen=True)
class ScoringConfig:
    """Projected transmission model and forward scoring of M0/M1/M2 configurations."""

    # Probability that an inherited allele is read as the other allele (child emission).
    mismatch_probability: float = 0.01
    # A candidate-parent bridge mixes in the uniform selector kernel with weight
    # switch probability x (1 - min(1, n / N)), n = the smaller informative selected-marker count of
    # the two adjacent parent bins, N = this value; 0 keeps the painter-interpolated chain.
    source_gap_support_sites: int = 1
    # Maximum-entropy bridge: root-finding tolerance and bisection budget.
    bridge_tolerance: float = 2e-13
    bridge_max_iterations: int = 128
    # Null source of an unobserved parent's slot: it starts from, and redraws from, the founder-state frequencies
    # transmitted by the other samples with an informative painting site (leave the child out), with a uniform
    # pseudo-count of this many samples (keeps a state no other sample carries possible).
    null_prior_samples: float = 1.0


@dataclass(frozen=True)
class PanelConfig:
    """Genome-wide retained-parent M2 panel; legacy screen constants remain for compatibility."""

    # Legacy reference only; production includes every eligible pair of genome-retained M1 parents.
    parents_per_child: int = 20
    # Per-chromosome softmax temperature = (ceil(markers / markers_per_information_block)) ** power.
    information_tempering_power: float = 0.5
    # Chromosome weights sqrt(markers) / median are clipped to [1 / ratio, ratio].
    maximum_contig_weight_ratio: float = 4.0
    # Utility = rank_weight x tied rank evidence + (1 - rank_weight) x contaminated softmax.
    rank_weight: float = 0.35
    # Softmax mixed with the uniform distribution over eligible parents at this weight.
    chromosome_contamination: float = 0.02


@dataclass(frozen=True)
class ExposureConfig:
    """Hard exposure gate: a configuration is testable only on enough exposed evidence."""

    minimum_edge_exposed_bins: float = 1.0
    minimum_pair_exposed_bins: float = 1.0
    # Exposed bins / total painting bins of the selected chromosomes.
    minimum_exposed_fraction: float = 0.10
    # Working map units a fit's exposed chromosomes must hold (release 3.2; bootstrap draws count once). A
    # chromosome counts max(1, its scored map length / unit_morgans) units, and an edge (or panel pair) there
    # max(1, the map length it is exposed over / unit_morgans): an unlinked chromosome is an independent replicate
    # however short, and a long one holds several nearly independent stretches (one meiosis's transmission
    # correlation between loci d Morgans apart is exp(-2 d) under Haldane, 0.14 at 1 Morgan). With the resampling
    # thresholds Tier B needs 5 chromosomes shorter than unit_morgans, fewer when they are longer (resampling.reach).
    # Lengths are measured by a supplied genetic map, over the span its knots cover, or by a stated rate
    # (rate_is_measured); the fallback rate, and a map's extrapolated tails, are placeholders that measure no
    # recombination (pedigree.prepare.measured_coordinates), so a chromosome with neither counts one unit.
    minimum_exposed_units: float = 3.0
    unit_morgans: float = 1.0
    # [genetic_map] rate_is_measured: the fallback rate is the organism's stated genome-wide average, so map lengths
    # count from it where no supplied map covers them (pedigree.prepare.measured_coordinates).
    rate_is_measured: bool = False
    # Report-only thresholds of the C/X compatibility diagnostics (never gate a decision).
    report_edge_coverage: float = 0.95
    report_pair_explainability: float = 0.95


@dataclass(frozen=True)
class DirectionConfig:
    """Junction-count orientation and the reciprocal / short-ancestry family factors."""

    # Probability that one chromosome's junction comparison of a pair is painting failure.
    chromosome_error_probability: float = 0.05
    # Pair-level painting-failure mixture bounding the orientation correction to
    # [log(epsilon), log(2 - epsilon)]. Base shared this value with the M0 contamination
    # (parent_state_contamination_probability); the two roles are now separate settings.
    pair_failure_probability: float = 0.02
    # Synchronous cavity-message passes preparing the frozen short-ancestry path factor.
    message_passes: int = 4
    # Final reciprocal solve: undamped, then a damped retry; residual measured before damping.
    final_max_iterations: int = 8192
    # Undamped progress window: after two windows, an undamped solve that made no progress over the last window,
    # or whose geometric residual rate there needs more than final_pause_budgets x final_max_iterations
    # iterations in all to reach final_tolerance, pauses for the damped retry, and resumes where it paused if
    # the retry fails (so no fit that the full undamped budget would converge is lost). The forecast is a
    # heuristic: slow modes that speed up later can still pause; resumption is the safeguard.
    final_progress_window: int = 512
    final_pause_budgets: int = 4
    final_tolerance: float = 1e-8
    retry_iterations: int = 4096
    retry_damping: float = 0.5
    # Top joint configurations per sample in the reverse two-edge path factor.
    ancestry_path_budget: int = 16


@dataclass(frozen=True)
class SelectionConfig:
    """Marginal parent-state priors, M0 contamination and acyclic graph search."""

    state_priors: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    # Alternative fixed priors; an M0 call must survive all of them to be released.
    prior_sensitivity: tuple[tuple[float, float, float], ...] = (
        (0.50, 0.30, 0.20),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (0.20, 0.30, 0.50),
    )
    # Each aggregated M1/M2 likelihood is mixed with the child's M0 likelihood at this weight.
    m0_contamination_probability: float = 0.02
    graph_local_search_passes: int = 3


@dataclass(frozen=True)
class ResamplingConfig:
    """Chromosome bootstrap (seeded) and leave-one-chromosome-out refits."""

    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 20260725


@dataclass(frozen=True)
class TierThresholds:
    """Release cut-offs of one evidence tier (fractions of resampled fits)."""

    configuration_bootstrap: float
    parent_bootstrap: float
    loco_fraction: float


@dataclass(frozen=True)
class ReleaseConfig:
    """Tier A / Tier B release rules (Tier B is the primary product)."""

    # A tier call needs the child's own per-chromosome state evidence to pick a unique winner on chromosomes holding
    # at least this many map units (ExposureConfig.unit_morgans; release 3.2: three chromosomes shorter than one unit
    # or without a genetic map).
    minimum_informative_units: float = 3.0
    tier_a: TierThresholds = field(default_factory=lambda: TierThresholds(0.95, 0.95, 0.95))
    tier_b: TierThresholds = field(default_factory=lambda: TierThresholds(0.70, 0.70, 0.80))
    # A released inferred edge needs full-data orientation log odds >= log((1 - p) / p) (3.9 nats);
    # caller chronology is exempt.
    orientation_reversal_probability: float = 0.02
    # Candidate sets carry this much configuration probability mass.
    support_set_coverage: float = 0.95


@dataclass(frozen=True)
class ExclusionConfig:
    """Full-marker Mendelian exclusion releasing M0/M1 counts without new edges."""

    alpha: float = 0.01
    genotype_replacement_probability: float = 0.01
    bet_sizes: tuple[float, ...] = (1 / 1024, 1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0)
    # Half-overlapping physical windows (bp), plus the whole chromosome.
    window_widths_bp: tuple[int, ...] = (1_000_000, 5_000_000, 20_000_000)


@dataclass(frozen=True)
class SiblingGuardConfig:
    """The sibling-as-parent guard on released edges (release 3.2, pedigree.guard).

    A released edge p -> c is tested when c's full-sib family in the relatedness product has enough supported
    members, and withheld when the ratio r of p's IBD0 evidence with that family to the family's own sib-sib level
    lies in [lower, upper] (a sib is exchangeable with the other sibs, r near 1; a parent is never IBD0 with its
    children, r near 0).

    Attributes:
      minimum_family: an edge is tested only when c's family without p has at least this many supported members.
      member_support: a family member counts when its family is its best placement on at least this share of
        chromosomes (relatedness member support).
      lower: the edge is withheld when r is at least this...
      upper: ...and at most this (a more distant or unrelated candidate has r above 1).
    """

    minimum_family: int = 3
    member_support: float = 0.75
    lower: float = 0.75
    upper: float = 1.33

    def __post_init__(self) -> None:
        if int(self.minimum_family) != self.minimum_family or self.minimum_family < 2:
            raise ValueError("pedigree guard minimum_family must be an integer of at least 2")
        if not 0.5 <= self.member_support <= 1.0:
            raise ValueError("pedigree guard member_support must lie in [0.5, 1]")
        if not 0.0 < self.lower < 1.0 < self.upper:
            raise ValueError("pedigree guard needs 0 < lower < 1 < upper")


@dataclass(frozen=True)
class PedigreeConfig:
    """All pedigree settings; ``evidence_calibration`` is the only user-visible field."""

    # "predictive": one cross-chromosome evidence scale fitted on held-out chromosomes (<= 1);
    # "off": unscaled composite evidence.
    evidence_calibration: str = "predictive"
    preparation: PreparationConfig = field(default_factory=PreparationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    panel: PanelConfig = field(default_factory=PanelConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    resampling: ResamplingConfig = field(default_factory=ResamplingConfig)
    release: ReleaseConfig = field(default_factory=ReleaseConfig)
    exclusion: ExclusionConfig = field(default_factory=ExclusionConfig)
    guard: SiblingGuardConfig = field(default_factory=SiblingGuardConfig)

    def __post_init__(self) -> None:
        if self.evidence_calibration not in EVIDENCE_CALIBRATION_MODES:
            raise ValueError(
                f"pedigree evidence_calibration must be one of {EVIDENCE_CALIBRATION_MODES}, "
                f"not {self.evidence_calibration!r}")
