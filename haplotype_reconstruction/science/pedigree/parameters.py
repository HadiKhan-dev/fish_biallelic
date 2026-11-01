"""Bind the pedigree decision rules to explicit, immutable parameters.

Every default is the production value of 92b3daf. Only ``evidence_calibration`` is a user setting
(TOML ``[pedigree] evidence_calibration``); the groups below are internal model constants, documented
here and in docs/configuration.md. Each group is bound into the identity of exactly the records it
affects: ``preparation`` into prepared factors, ``scoring`` into screens and scores, ``panel`` into the
M2 candidate panel, the rest into the genome-wide decision.

Thresholds label evidence tiers; they are not biological acceptance probabilities.

Method: docs/methods/pedigree.md#inputs-and-outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...core.errors import SettingsError

__all__ = [
    "CalibrationNumerics",
    "WinnerNumerics",
    "CALIBRATION_NUMERICS",
    "WINNER_NUMERICS",
    "PreparationConfig",
    "ScoringConfig",
    "PanelConfig",
    "ExposureConfig",
    "DirectionConfig",
    "SelectionConfig",
    "ResamplingConfig",
    "TierThresholds",
    "ReleaseConfig",
    "ExclusionConfig",
    "SiblingGuardConfig",
    "PedigreeConfig",
    "ScoringNumerics",
    "PedigreeWorkspace",
    "SCORING_NUMERICS",
    "PEDIGREE_WORKSPACE",
    "INFORMATIVE_GL_SPREAD",
    "SELECTOR_SWITCH_FLOOR",
    "HALL_ULPS",
    "HUB_MASS_TIE",
    "SITE_POLYNOMIAL_THRESHOLD",
    "LOG_FLOOR",
    "COMMON_EMISSION_CACHE_BYTES",
    "TINY",
    "FAMILY_EXP_UNDERFLOW",
    "ORDER_MINIMUM_WEIGHT",
    "ORDER_VARIANCE_OFFSET",
    "JUNCTION_UNREACHABLE_STATE",
    "JUNCTION_UNREACHABLE_COUNT",
    "BOUND_WORKSPACE_BYTES",
    "BOUND_BATCH_ROWS",
    "EXCLUSION_PREFIX_SITES",
    "EXCLUSION_CHUNK_SITES",
    "EXCLUSION_PATTERN_LIMIT",
    "FAMILY_PARALLEL_MINIMUM_WORK",
    "NULL_ROUNDING_ULPS",
    "BRIDGE_FEASIBILITY_ULPS",
    "BRIDGE_POST_TOLERANCE_FACTOR",
    "BRIDGE_POST_ULPS",
    "BRIDGE_BISECTION_LIMIT",
    "CERTIFICATE_GL_ULPS",
    "SELECTIVE_UPPER_ULPS",
    "SELECTIVE_UPPER_STATE_FACTOR",
    "SELECTIVE_BOUND_ULPS",
    "SELECTIVE_SUM_ULPS",
    "SELECTIVE_FRONTIER_ULPS",
    "PAIR_NULL_RELATIVE_GUARD",
    "EXCLUSION_TINY",
    "EMISSION_SAMPLE_BLOCK",
    "FAMILY_LANES_PER_THREAD",
    "FAMILY_CLAIM_WORK",
    "FAMILY_TRANSPOSE_BLOCK",
    "SELECTIVE_BOUND_GROUP_BINS",
    "EXCLUSION_DYAD_BATCH",
]

EVIDENCE_CALIBRATION_MODES = ("off", "predictive")


@dataclass(frozen=True)
class CalibrationNumerics:
    """Fixed search bounds, accuracy and budgets of predictive calibration (PD03)."""

    log_scale_bounds: tuple[float, float] = field(
        default=(float(np.log(0.001)), float(np.log(100.0))),
        metadata={"provenance": "3.5 pedigree/calibration.py:LOG_SCALE_BOUNDS; P0 PD03"},
    )
    grid_points: int = field(default=33, metadata={"provenance": "3.5 calibration.py:GRID_POINTS; P0 PD03"})
    optimizer_xatol: float = field(default=1e-5, metadata={"provenance": "3.5 calibration.py:OPTIMIZER_XATOL; P0 PD03"})
    minimum_gain: float = field(default=1e-8, metadata={"provenance": "3.5 calibration.py:MINIMUM_GAIN; P0 PD03"})
    boundary_margin: float = field(default=0.01, metadata={"provenance": "3.5 calibration.py:BOUNDARY_MARGIN; P0 PD03"})
    refine_tolerance: float = field(
        default=1e-14, metadata={"provenance": "3.5 calibration.py:REFINE_TOLERANCE; P0 PD03"}
    )
    refine_iterations: int = field(default=60, metadata={"provenance": "3.5 calibration.py:REFINE_ITERATIONS; P0 PD03"})
    bracket_expansions: int = field(default=30, metadata={"provenance": "3.5 calibration.py:_refine; P0 PD03"})
    parallel_minimum_terms: int = field(
        default=200_000, metadata={"provenance": "3.5 calibration.py:POOL_MINIMUM_TERMS; scheduling only"}
    )


@dataclass(frozen=True)
class WinnerNumerics:
    """Exact winner and graph-confidence tolerances; no user-facing tuning (PD11-PD13)."""

    contrast_ulp_factor: float = field(default=512.0, metadata={"provenance": "3.5 pedigree/ties.py; P0 PD11"})
    noise_floor: float = field(default=2.0**-15, metadata={"provenance": "3.5 pedigree/ties.py:NOISE_FLOOR; P0 PD11"})
    minimum_graph_quantum: float = field(default=1e-300, metadata={"provenance": "3.5 graph.py:_visit_order; P0 PD13"})


# Fixed model numerics are versioned by source identity, not additional user settings.
CALIBRATION_NUMERICS = CalibrationNumerics()
WINNER_NUMERICS = WinnerNumerics()


@dataclass(frozen=True)
class PreparationConfig:
    """Marker subsampling and information tempering of the prepared painting components."""

    # At most this many evenly spaced selected markers per painting bin enter the transmission model.
    markers_per_bin: int = field(
        default=10, metadata={"provenance": "3.5 pedigree/config.py:PreparationConfig.markers_per_bin; P0 PS01"}
    )
    # Chromosome-wide groups of at least this many selected markers share one information exponent.
    markers_per_information_block: int = field(
        default=100,
        metadata={"provenance": "3.5 pedigree/config.py:PreparationConfig.markers_per_information_block; P0 PS01"},
    )
    # The tempered view raises a bin's emission to min(this, n) / n for n informative markers in its group.
    effective_markers_per_information_block: float = field(
        default=3.0,
        metadata={
            "provenance": "3.5 pedigree/config.py:PreparationConfig.effective_markers_per_information_block; P0 PS01"
        },
    )


@dataclass(frozen=True)
class ScoringConfig:
    """Projected transmission model and forward scoring of M0/M1/M2 configurations."""

    # Probability that an inherited allele is read as the other allele (child emission).
    mismatch_probability: float = field(
        default=0.01,
        metadata={"provenance": "3.5 pedigree/config.py:ScoringConfig.mismatch_probability; P0 PS04"},
    )
    # A candidate-parent bridge mixes in the uniform selector kernel with weight
    # switch probability x (1 - min(1, n / N)), n = the smaller informative selected-marker count of
    # the two adjacent parent bins, N = this value; 0 keeps the painter-interpolated chain.
    source_gap_support_sites: int = field(
        default=1,
        metadata={"provenance": "3.5 pedigree/config.py:ScoringConfig.source_gap_support_sites; P0 PS04"},
    )
    # Maximum-entropy bridge: post-check tolerance and independent bisection budget.
    bridge_postcheck_tolerance: float = field(
        default=2e-13, metadata={"provenance": "3.5 pedigree/config.py:ScoringConfig.bridge_tolerance; P0 PS04"}
    )
    bridge_max_iterations: int = field(
        default=128,
        metadata={"provenance": "3.5 pedigree/config.py:ScoringConfig.bridge_max_iterations; P0 PS04"},
    )
    # Null source of an unobserved parent's slot: it starts from, and redraws from, the founder-state frequencies
    # transmitted by the other samples with an informative painting site (leave the child out), with a uniform
    # pseudo-count of this many samples (keeps a state no other sample carries possible).
    null_prior_samples: float = field(
        default=1.0, metadata={"provenance": "3.5 pedigree/config.py:ScoringConfig.null_prior_samples; P0 PS04"}
    )


@dataclass(frozen=True)
class PanelConfig:
    """Genome-wide M2 panel of bounded retained-parent pairs."""

    # Release 3.4: a pair (p, q) of genome-retained M1 parents enters the M2 panel when, in the full-data fit or in
    # some leave-one-chromosome-out fit, a valid upper bound on its joint M2 likelihood (pedigree.pair_bound: both
    # parents' transmission chains against the child's genotypes, bin by bin, summed over the fit's chromosomes)
    # comes within this many raw nats of the child's best known configuration there (M0 or a retained M1). A pair
    # left out therefore has an M2 more than this far below the best configuration in every such fit. 150 raw nats
    # are 33 nats at the 0.22 evidence scale fitted on bc8. Pairs outside contribute nothing to the M2 evidence (a
    # lower bound).
    pair_relevance_margin: float = field(
        default=150.0,
        metadata={"provenance": "3.5 pedigree/config.py:PanelConfig.pair_relevance_margin; P0 PS07"},
    )
    # States with at least this transmitted-state mass keep their own group in the pair bound (a parent's other named
    # states are aggregated). Any value keeps the bound valid; it only sets its tightness and cost.
    pair_bound_high_mass: float = field(
        default=0.01, metadata={"provenance": "3.5 pedigree/config.py:PanelConfig.pair_bound_high_mass; P0 PS07"}
    )
    # Per-chromosome softmax temperature = (ceil(markers / markers_per_information_block)) ** power.
    information_tempering_power: float = field(
        default=0.5,
        metadata={"provenance": "3.5 pedigree/config.py:PanelConfig.information_tempering_power; P0 PS07"},
    )
    # Chromosome weights sqrt(markers) / median are clipped to [1 / ratio, ratio].
    maximum_contig_weight_ratio: float = field(
        default=4.0,
        metadata={"provenance": "3.5 pedigree/config.py:PanelConfig.maximum_contig_weight_ratio; P0 PS07"},
    )
    # Utility = rank_weight x tied rank evidence + (1 - rank_weight) x contaminated softmax.
    rank_weight: float = field(
        default=0.35, metadata={"provenance": "3.5 pedigree/config.py:PanelConfig.rank_weight; P0 PS07"}
    )
    # Softmax mixed with the uniform distribution over eligible parents at this weight.
    chromosome_contamination: float = field(
        default=0.02,
        metadata={"provenance": "3.5 pedigree/config.py:PanelConfig.chromosome_contamination; P0 PS07"},
    )


@dataclass(frozen=True)
class ExposureConfig:
    """Hard exposure gate: a configuration is testable only on enough exposed evidence."""

    minimum_edge_exposed_bins: float = field(
        default=1.0,
        metadata={"provenance": "3.5 pedigree/config.py:ExposureConfig.minimum_edge_exposed_bins; P0 PD05"},
    )
    minimum_pair_exposed_bins: float = field(
        default=1.0,
        metadata={"provenance": "3.5 pedigree/config.py:ExposureConfig.minimum_pair_exposed_bins; P0 PD05"},
    )
    # Exposed bins / total painting bins of the selected chromosomes.
    minimum_exposed_fraction: float = field(
        default=0.1,
        metadata={"provenance": "3.5 pedigree/config.py:ExposureConfig.minimum_exposed_fraction; P0 PD05"},
    )
    # Working map units a fit's exposed chromosomes must hold (release 3.2; bootstrap draws count once). A
    # chromosome counts max(1, its scored map length / unit_morgans) units, and an edge (or panel pair) there
    # max(1, the map length it is exposed over / unit_morgans): an unlinked chromosome is an independent replicate
    # however short, and a long one holds several nearly independent stretches (one meiosis's transmission
    # correlation between loci d Morgans apart is exp(-2 d) under Haldane, 0.14 at 1 Morgan). With the resampling
    # thresholds Tier B needs 5 chromosomes shorter than unit_morgans, fewer when they are longer (resampling.reach).
    # Lengths are measured by a supplied genetic map, over the span its knots cover, or by a stated rate
    # (rate_is_measured); the fallback rate, and a map's extrapolated tails, are placeholders that measure no
    # recombination (pedigree.prepare.measured_coordinates), so a chromosome with neither counts one unit.
    minimum_exposed_units: float = field(
        default=3.0,
        metadata={"provenance": "3.5 pedigree/config.py:ExposureConfig.minimum_exposed_units; P0 PD05"},
    )
    unit_morgans: float = field(
        default=1.0, metadata={"provenance": "3.5 pedigree/config.py:ExposureConfig.unit_morgans; P0 PD05"}
    )
    # [genetic_map] rate_is_measured: the fallback rate is the organism's stated genome-wide average, so map lengths
    # count from it where no supplied map covers them (pedigree.prepare.measured_coordinates).
    rate_is_measured: bool = field(
        default=False, metadata={"provenance": "3.5 pedigree/config.py:ExposureConfig.rate_is_measured; P0 PD05"}
    )
    # Report-only thresholds of the C/X compatibility diagnostics (never gate a decision).
    report_edge_coverage: float = field(
        default=0.95,
        metadata={"provenance": "3.5 pedigree/config.py:ExposureConfig.report_edge_coverage; P0 PD05"},
    )
    report_pair_explainability: float = field(
        default=0.95,
        metadata={"provenance": "3.5 pedigree/config.py:ExposureConfig.report_pair_explainability; P0 PD05"},
    )


@dataclass(frozen=True)
class DirectionConfig:
    """Junction-count orientation and the reciprocal / short-ancestry family factors."""

    # Probability that one chromosome's junction comparison of a pair is painting failure.
    chromosome_error_probability: float = field(
        default=0.05,
        metadata={
            "provenance": "3.5 pedigree/config.py:DirectionConfig.chromosome_error_probability; P0 PD06/PD08/PD09/PD10"
        },
    )
    # Pair-level painting-failure mixture bounding the orientation correction to
    # [log(epsilon), log(2 - epsilon)]. Base shared this value with the M0 contamination
    # (parent_state_contamination_probability); the two roles are now separate settings.
    pair_failure_probability: float = field(
        default=0.02,
        metadata={
            "provenance": "3.5 pedigree/config.py:DirectionConfig.pair_failure_probability; P0 PD06/PD08/PD09/PD10"
        },
    )
    # Synchronous cavity-message passes preparing the frozen short-ancestry path factor.
    message_passes: int = field(
        default=4,
        metadata={"provenance": "3.5 pedigree/config.py:DirectionConfig.message_passes; P0 PD06/PD08/PD09/PD10"},
    )
    # Final reciprocal solve: undamped, then a damped retry; residual measured before damping.
    final_max_iterations: int = field(
        default=8192,
        metadata={"provenance": "3.5 pedigree/config.py:DirectionConfig.final_max_iterations; P0 PD06/PD08/PD09/PD10"},
    )
    # Undamped progress window: after two windows, an undamped solve that made no progress over the last window,
    # or whose geometric residual rate there needs more than final_pause_budgets x final_max_iterations
    # iterations in all to reach final_tolerance, pauses for the damped retry, and resumes where it paused if
    # the retry fails (so no fit that the full undamped budget would converge is lost). The forecast is a
    # heuristic: slow modes that speed up later can still pause; resumption is the safeguard.
    final_progress_window: int = field(
        default=512,
        metadata={"provenance": "3.5 pedigree/config.py:DirectionConfig.final_progress_window; P0 PD06/PD08/PD09/PD10"},
    )
    final_pause_budgets: int = field(
        default=4,
        metadata={"provenance": "3.5 pedigree/config.py:DirectionConfig.final_pause_budgets; P0 PD06/PD08/PD09/PD10"},
    )
    final_tolerance: float = field(
        default=1e-08,
        metadata={"provenance": "3.5 pedigree/config.py:DirectionConfig.final_tolerance; P0 PD06/PD08/PD09/PD10"},
    )
    retry_iterations: int = field(
        default=4096,
        metadata={"provenance": "3.5 pedigree/config.py:DirectionConfig.retry_iterations; P0 PD06/PD08/PD09/PD10"},
    )
    retry_damping: float = field(
        default=0.5,
        metadata={"provenance": "3.5 pedigree/config.py:DirectionConfig.retry_damping; P0 PD06/PD08/PD09/PD10"},
    )
    # Top joint configurations per sample in the reverse two-edge path factor.
    ancestry_path_budget: int = field(
        default=16,
        metadata={"provenance": "3.5 pedigree/config.py:DirectionConfig.ancestry_path_budget; P0 PD06/PD08/PD09/PD10"},
    )


@dataclass(frozen=True)
class SelectionConfig:
    """Marginal parent-state priors, M0 contamination and acyclic graph search."""

    state_priors: tuple[float, float, float] = field(
        default=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        metadata={"provenance": "3.5 pedigree/config.py:SelectionConfig.state_priors; P0 PD04/PD12/PD13/PD14"},
    )
    # Alternative fixed priors; an M0 call must survive all of them to be released.
    prior_sensitivity: tuple[tuple[float, float, float], ...] = field(
        default=((0.5, 0.3, 0.2), (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), (0.2, 0.3, 0.5)),
        metadata={"provenance": "3.5 pedigree/config.py:SelectionConfig.prior_sensitivity; P0 PD04/PD12/PD13/PD14"},
    )
    # Each aggregated M1/M2 likelihood is mixed with the child's M0 likelihood at this weight.
    m0_contamination_probability: float = field(
        default=0.02,
        metadata={
            "provenance": "3.5 pedigree/config.py:SelectionConfig.m0_contamination_probability; P0 PD04/PD12/PD13/PD14"
        },
    )
    graph_local_search_passes: int = field(
        default=3,
        metadata={
            "provenance": "3.5 pedigree/config.py:SelectionConfig.graph_local_search_passes; P0 PD04/PD12/PD13/PD14"
        },
    )


@dataclass(frozen=True)
class ResamplingConfig:
    """Chromosome bootstrap (seeded) and leave-one-chromosome-out refits."""

    bootstrap_replicates: int = field(
        default=1000,
        metadata={"provenance": "3.5 pedigree/config.py:ResamplingConfig.bootstrap_replicates; P0 PD16"},
    )
    bootstrap_seed: int = field(
        default=20260725,
        metadata={"provenance": "3.5 pedigree/config.py:ResamplingConfig.bootstrap_seed; P0 PD16"},
    )


@dataclass(frozen=True)
class TierThresholds:
    """Release cut-offs of one evidence tier (fractions of resampled fits)."""

    configuration_bootstrap: float = field(
        metadata={"provenance": "3.5 pedigree/config.py:TierThresholds.configuration_bootstrap; P0 PD18"}
    )
    parent_bootstrap: float = field(
        metadata={"provenance": "3.5 pedigree/config.py:TierThresholds.parent_bootstrap; P0 PD18"}
    )
    loco_fraction: float = field(
        metadata={"provenance": "3.5 pedigree/config.py:TierThresholds.loco_fraction; P0 PD18"}
    )


@dataclass(frozen=True)
class ReleaseConfig:
    """Tier A / Tier B release rules (Tier B is the primary product)."""

    # A tier call needs the child's own per-chromosome state evidence to pick a unique winner on chromosomes holding
    # at least this many map units (ExposureConfig.unit_morgans; release 3.2: three chromosomes shorter than one unit
    # or without a genetic map).
    minimum_informative_units: float = field(
        default=3.0,
        metadata={"provenance": "3.5 pedigree/config.py:ReleaseConfig.minimum_informative_units; P0 PD18/PD20/PD21"},
    )
    tier_a: TierThresholds = field(
        default_factory=lambda: TierThresholds(0.95, 0.95, 0.95),
        metadata={"provenance": "3.5 pedigree/config.py:ReleaseConfig.tier_a; P0 PD18/PD20/PD21"},
    )
    tier_b: TierThresholds = field(
        default_factory=lambda: TierThresholds(0.7, 0.7, 0.8),
        metadata={"provenance": "3.5 pedigree/config.py:ReleaseConfig.tier_b; P0 PD18/PD20/PD21"},
    )
    # A released edge needs full-data orientation log odds >= log((1 - p) / p) (3.9 nats).
    orientation_reversal_probability: float = field(
        default=0.02,
        metadata={
            "provenance": "3.5 pedigree/config.py:ReleaseConfig.orientation_reversal_probability; P0 PD18/PD20/PD21"
        },
    )
    # Candidate sets carry this much configuration probability mass.
    support_set_coverage: float = field(
        default=0.95,
        metadata={"provenance": "3.5 pedigree/config.py:ReleaseConfig.support_set_coverage; P0 PD18/PD20/PD21"},
    )


@dataclass(frozen=True)
class ExclusionConfig:
    """Full-marker Mendelian exclusion releasing M0/M1 counts without new edges."""

    alpha: float = field(default=0.01, metadata={"provenance": "3.5 pedigree/config.py:ExclusionConfig.alpha; P0 PD22"})
    genotype_replacement_probability: float = field(
        default=0.01,
        metadata={"provenance": "3.5 pedigree/config.py:ExclusionConfig.genotype_replacement_probability; P0 PD22"},
    )
    bet_sizes: tuple[float, ...] = field(
        default=(1 / 1024, 1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0),
        metadata={"provenance": "3.5 pedigree/config.py:ExclusionConfig.bet_sizes; P0 PD22"},
    )
    # Half-overlapping physical windows (bp), plus the whole chromosome.
    window_widths_bp: tuple[int, ...] = field(
        default=(1000000, 5000000, 20000000),
        metadata={"provenance": "3.5 pedigree/config.py:ExclusionConfig.window_widths_bp; P0 PD22"},
    )


@dataclass(frozen=True)
class SiblingGuardConfig:
    """The sibling-as-parent guard on released edges (release 3.4: family consistency, pedigree.guard).

    A released edge p -> c is tested when c's full-sib family in the relatedness product, without p, without members
    released as a parent or child of c or of another member, and without members with a released parent other than p
    and c's released parents, has enough supported members; it is withheld when a strict majority of the tested
    members (c excluded) are Mendelian-excluded from p at the exclusion release's threshold (a parent of c is a parent
    of every full sib of c).

    Attributes:
      minimum_family: an edge is tested only when at least this many supported members remain, counting c.
      member_support: a family member counts when its family is its best placement on at least this share of
        chromosomes (relatedness member support); c itself must reach it.
    """

    minimum_family: int = field(
        default=3, metadata={"provenance": "3.5 pedigree/config.py:SiblingGuardConfig.minimum_family; P0 PD23"}
    )
    member_support: float = field(
        default=0.75,
        metadata={"provenance": "3.5 pedigree/config.py:SiblingGuardConfig.member_support; P0 PD23"},
    )

    def __post_init__(self) -> None:
        if int(self.minimum_family) != self.minimum_family or self.minimum_family < 2:
            raise SettingsError(
                f"pedigree guard minimum_family={self.minimum_family!r} must be an integer of at least 2"
            )
        if not 0.5 <= self.member_support <= 1.0:
            raise SettingsError(f"pedigree guard member_support={self.member_support!r} must lie in [0.5, 1]")


@dataclass(frozen=True)
class PedigreeConfig:
    """All pedigree settings; ``evidence_calibration`` is the only user-visible field."""

    # "predictive": one cross-chromosome evidence scale fitted on held-out chromosomes (<= 1);
    # "off": unscaled composite evidence.
    evidence_calibration: str = field(
        default="predictive",
        metadata={"provenance": "3.5 pedigree/config.py:PedigreeConfig.evidence_calibration; P0 PD27"},
    )
    # L8: explicit arbitrary convention, never an environment read.
    reverse_conventions: bool = field(
        default=False,
        metadata={"provenance": "3.5 pedigree/config.py:PedigreeConfig.reverse_conventions; P0 PD27"},
    )
    preparation: PreparationConfig = field(
        default_factory=PreparationConfig,
        metadata={"provenance": "3.5 pedigree/config.py:PedigreeConfig.preparation; P0 PD27"},
    )
    scoring: ScoringConfig = field(
        default_factory=ScoringConfig,
        metadata={"provenance": "3.5 pedigree/config.py:PedigreeConfig.scoring; P0 PD27"},
    )
    panel: PanelConfig = field(
        default_factory=PanelConfig,
        metadata={"provenance": "3.5 pedigree/config.py:PedigreeConfig.panel; P0 PD27"},
    )
    exposure: ExposureConfig = field(
        default_factory=ExposureConfig,
        metadata={"provenance": "3.5 pedigree/config.py:PedigreeConfig.exposure; P0 PD27"},
    )
    direction: DirectionConfig = field(
        default_factory=DirectionConfig,
        metadata={"provenance": "3.5 pedigree/config.py:PedigreeConfig.direction; P0 PD27"},
    )
    selection: SelectionConfig = field(
        default_factory=SelectionConfig,
        metadata={"provenance": "3.5 pedigree/config.py:PedigreeConfig.selection; P0 PD27"},
    )
    resampling: ResamplingConfig = field(
        default_factory=ResamplingConfig,
        metadata={"provenance": "3.5 pedigree/config.py:PedigreeConfig.resampling; P0 PD27"},
    )
    release: ReleaseConfig = field(
        default_factory=ReleaseConfig,
        metadata={"provenance": "3.5 pedigree/config.py:PedigreeConfig.release; P0 PD27"},
    )
    exclusion: ExclusionConfig = field(
        default_factory=ExclusionConfig,
        metadata={"provenance": "3.5 pedigree/config.py:PedigreeConfig.exclusion; P0 PD27"},
    )
    guard: SiblingGuardConfig = field(
        default_factory=SiblingGuardConfig,
        metadata={"provenance": "3.5 pedigree/config.py:PedigreeConfig.guard; P0 PD27"},
    )

    def __post_init__(self) -> None:
        if self.evidence_calibration not in EVIDENCE_CALIBRATION_MODES:
            raise SettingsError(
                f"pedigree evidence_calibration must be one of {EVIDENCE_CALIBRATION_MODES}, "
                f"not {self.evidence_calibration!r}"
            )


@dataclass(frozen=True)
class ScoringNumerics:
    """Fixed PE1/PE2 numerical cutoffs, named independently of user-facing settings."""

    informative_gl_spread: float = field(
        default=1e-12, metadata={"provenance": "3.5 prepare.py:informative_components; PS01"}
    )
    selector_switch_floor: float = field(
        default=1e-15, metadata={"provenance": "3.5 prepare.py:selector_switch_probability; PS04"}
    )
    hall_ulps: float = field(default=64.0, metadata={"provenance": "3.5 projection.py:_hall; PS06"})
    hub_mass_tie: float = field(default=2.0**-20, metadata={"provenance": "3.5 projection.py:hub selection; PS07"})
    site_polynomial_threshold: float = field(
        default=1e-8, metadata={"provenance": "3.5 forward.py:site_likelihood; PS09"}
    )
    log_floor: float = field(default=-60.0, metadata={"provenance": "3.5 m1.py:LOG_FLOOR; PS11"})

    tiny: float = field(default=1e-280, metadata={"provenance": "3.5 pair_bound.py:_TINY; PS15"})
    family_exp_underflow: float = field(default=750.0, metadata={"provenance": "3.5 family.py:_EXP_UNDERFLOW; PD09"})
    order_minimum_weight: float = field(
        default=2.0, metadata={"provenance": "3.5 orientation.py:order_probability; PD07"}
    )
    order_variance_offset: float = field(
        default=1.0, metadata={"provenance": "3.5 orientation.py:order_probability; PD07"}
    )
    junction_unreachable_state: int = field(default=1 << 30, metadata={"provenance": "3.5 junctions.py:102; PS18"})
    junction_unreachable_count: int = field(default=1 << 28, metadata={"provenance": "3.5 junctions.py:189; PS18"})

    null_rounding_ulps: float = field(
        default=64.0, metadata={"provenance": "3.5 forward.py:ChildTerms.__init__ null persistence rounding; PS09"}
    )

    bridge_feasibility_ulps: float = field(
        default=64.0, metadata={"provenance": "3.5 projection.py:_bridges_kernel feasibility_tolerance; PS07"}
    )

    bridge_post_tolerance_factor: float = field(
        default=8.0, metadata={"provenance": "3.5 projection.py:_bridges_kernel post_tolerance; PS07"}
    )

    bridge_post_ulps: float = field(
        default=256.0, metadata={"provenance": "3.5 projection.py:_bridges_kernel post_tolerance; PS07"}
    )

    bridge_bisection_limit: int = field(
        default=128, metadata={"provenance": "3.5 projection.py:_bisect_root fixed iteration cap; PS07"}
    )

    certificate_gl_ulps: int = field(
        default=64, metadata={"provenance": "3.5 block_certificate_fast.py:global_common_logs GL outward guard; PS22"}
    )

    selective_upper_ulps: int = field(
        default=4096, metadata={"provenance": "3.5 selective_bounds.py:Geometry.upper outward padding; PS22"}
    )

    selective_upper_state_factor: int = field(
        default=64, metadata={"provenance": "3.5 selective_bounds.py:Geometry.upper state-count padding factor; PS22"}
    )

    selective_bound_ulps: int = field(
        default=128,
        metadata={"provenance": "3.5 selective_bounds.py:Geometry.__init__ coefficient outward guards; PS22"},
    )

    selective_sum_ulps: int = field(
        default=8, metadata={"provenance": "3.5 selective_m1.py:outward_sum ordered accumulation guard; PS22"}
    )

    selective_frontier_ulps: int = field(
        default=32, metadata={"provenance": "3.5 selective_m1.py:search_batch best/floor frontier guards; PS22"}
    )

    pair_null_relative_guard: float = field(
        default=1e-12, metadata={"provenance": "3.5 pair_bound.py:component_context null_row_max inflation; PS15"}
    )

    exclusion_tiny: float = field(
        default=1e-300,
        metadata={"provenance": "3.5 exclusion.py:_ratio and _log_bet positive denominator/log floor; PS20"},
    )


@dataclass(frozen=True)
class PedigreeWorkspace:
    """Storage dimensions only; estimates never reserve memory or limit concurrency."""

    common_emission_cache_bytes: int = field(
        default=256 << 20, metadata={"provenance": "3.5 forward.py:common cache; PS09"}
    )

    bound_workspace_bytes: int = field(
        default=512 << 20, metadata={"provenance": "3.5 pair_bound.py:WORKSPACE_BYTES; PS15"}
    )
    bound_batch_rows: int = field(default=4_000_000, metadata={"provenance": "3.5 pair_bound.py:BATCH_ROWS; PS15"})
    exclusion_prefix_sites: int = field(
        default=64, metadata={"provenance": "3.5 exclusion_bound.py:PREFIX_SITES; PS21"}
    )
    exclusion_chunk_sites: int = field(
        default=4096, metadata={"provenance": "3.5 exclusion_bound.py:CHUNK_SITES; PS21"}
    )
    exclusion_pattern_limit: int = field(default=1024, metadata={"provenance": "3.5 exclusion.py:MAX_PATTERNS; PS20"})
    family_parallel_minimum_work: int = field(
        default=200_000, metadata={"provenance": "3.5 family.py:_PARALLEL_MINIMUM_WORK; PD09"}
    )

    emission_sample_block: int = field(
        default=16, metadata={"provenance": "3.5 prepare.py:_EMISSION_SAMPLE_BLOCK in _prepare_component; PS01"}
    )

    family_lanes_per_thread: int = field(default=8, metadata={"provenance": "3.5 family.py:_LANES_PER_THREAD; PD09"})

    family_claim_work: int = field(default=20_000_000, metadata={"provenance": "3.5 family.py:_CLAIM_WORK; PD09"})

    family_transpose_block: int = field(
        default=64, metadata={"provenance": "3.5 family.py:_cavity_iterations transpose block; PD09"}
    )

    selective_bound_group_bins: int = field(
        default=8, metadata={"provenance": "3.5 selective_bounds.py:Geometry.__init__ grouping bin edges; PS22"}
    )

    exclusion_dyad_batch: int = field(default=256, metadata={"provenance": "3.5 exclusion.py:DYAD_BATCH; PS20"})


SCORING_NUMERICS = ScoringNumerics()
PEDIGREE_WORKSPACE = PedigreeWorkspace()
INFORMATIVE_GL_SPREAD = SCORING_NUMERICS.informative_gl_spread
SELECTOR_SWITCH_FLOOR = SCORING_NUMERICS.selector_switch_floor
HALL_ULPS = SCORING_NUMERICS.hall_ulps
HUB_MASS_TIE = SCORING_NUMERICS.hub_mass_tie
SITE_POLYNOMIAL_THRESHOLD = SCORING_NUMERICS.site_polynomial_threshold
LOG_FLOOR = SCORING_NUMERICS.log_floor
COMMON_EMISSION_CACHE_BYTES = PEDIGREE_WORKSPACE.common_emission_cache_bytes

TINY = SCORING_NUMERICS.tiny
FAMILY_EXP_UNDERFLOW = SCORING_NUMERICS.family_exp_underflow
ORDER_MINIMUM_WEIGHT = SCORING_NUMERICS.order_minimum_weight
ORDER_VARIANCE_OFFSET = SCORING_NUMERICS.order_variance_offset
JUNCTION_UNREACHABLE_STATE = SCORING_NUMERICS.junction_unreachable_state
JUNCTION_UNREACHABLE_COUNT = SCORING_NUMERICS.junction_unreachable_count
BOUND_WORKSPACE_BYTES = PEDIGREE_WORKSPACE.bound_workspace_bytes
BOUND_BATCH_ROWS = PEDIGREE_WORKSPACE.bound_batch_rows
EXCLUSION_PREFIX_SITES = PEDIGREE_WORKSPACE.exclusion_prefix_sites
EXCLUSION_CHUNK_SITES = PEDIGREE_WORKSPACE.exclusion_chunk_sites
EXCLUSION_PATTERN_LIMIT = PEDIGREE_WORKSPACE.exclusion_pattern_limit
FAMILY_PARALLEL_MINIMUM_WORK = PEDIGREE_WORKSPACE.family_parallel_minimum_work

NULL_ROUNDING_ULPS = SCORING_NUMERICS.null_rounding_ulps
BRIDGE_FEASIBILITY_ULPS = SCORING_NUMERICS.bridge_feasibility_ulps
BRIDGE_POST_TOLERANCE_FACTOR = SCORING_NUMERICS.bridge_post_tolerance_factor
BRIDGE_POST_ULPS = SCORING_NUMERICS.bridge_post_ulps
BRIDGE_BISECTION_LIMIT = SCORING_NUMERICS.bridge_bisection_limit
CERTIFICATE_GL_ULPS = SCORING_NUMERICS.certificate_gl_ulps
SELECTIVE_UPPER_ULPS = SCORING_NUMERICS.selective_upper_ulps
SELECTIVE_UPPER_STATE_FACTOR = SCORING_NUMERICS.selective_upper_state_factor
SELECTIVE_BOUND_ULPS = SCORING_NUMERICS.selective_bound_ulps
SELECTIVE_SUM_ULPS = SCORING_NUMERICS.selective_sum_ulps
SELECTIVE_FRONTIER_ULPS = SCORING_NUMERICS.selective_frontier_ulps
PAIR_NULL_RELATIVE_GUARD = SCORING_NUMERICS.pair_null_relative_guard
EXCLUSION_TINY = SCORING_NUMERICS.exclusion_tiny
EMISSION_SAMPLE_BLOCK = PEDIGREE_WORKSPACE.emission_sample_block
FAMILY_LANES_PER_THREAD = PEDIGREE_WORKSPACE.family_lanes_per_thread
FAMILY_CLAIM_WORK = PEDIGREE_WORKSPACE.family_claim_work
FAMILY_TRANSPOSE_BLOCK = PEDIGREE_WORKSPACE.family_transpose_block
SELECTIVE_BOUND_GROUP_BINS = PEDIGREE_WORKSPACE.selective_bound_group_bins
EXCLUSION_DYAD_BATCH = PEDIGREE_WORKSPACE.exclusion_dyad_batch
