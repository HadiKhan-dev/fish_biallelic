"""Bind the pedigree decision rules to explicit, immutable parameters.

Only ``evidence_calibration`` is a user setting
(TOML ``[pedigree] evidence_calibration``); the groups below are internal model constants, documented
here and in docs/configuration.md. Each group is bound into the identity of exactly the records it
affects: ``preparation`` into prepared factors, ``scoring`` into screens and scores, ``panel`` into the
M2 candidate panel, direction mismatch probability into scores, and the remaining decision parameters
into the genome-wide decision.

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
    "PreparationParameters",
    "ScoringParameters",
    "PanelParameters",
    "ExposureParameters",
    "DirectionParameters",
    "SelectionParameters",
    "ResamplingParameters",
    "TierThresholds",
    "ReleaseParameters",
    "ExclusionParameters",
    "RelationshipParameters",
    "PedigreeParameters",
    "ScoringNumerics",
    "PedigreeWorkspace",
    "SCORING_NUMERICS",
    "PEDIGREE_WORKSPACE",
    "INFORMATIVE_GL_SPREAD",
    "SELECTOR_SWITCH_FLOOR",
    "TRANSMISSION_SWITCH_FLOOR",
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
    "EXCLUSION_PATTERN_LIMIT",
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
        metadata={
            "provenance": (
                "PD03; dimensionless log-scale search interval, corresponding to evidence weights 0.001 through "
                "100."
            ),
        },
    )
    grid_points: int = field(
        default=33,
        metadata={
            "provenance": "PD03; 33 grid points spanning the log-scale interval before interior-maximum refinement.",
        },
    )
    optimizer_xatol: float = field(
        default=1e-5,
        metadata={
            "provenance": "PD03; absolute optimizer tolerance of 1e-5 in dimensionless log scale.",
        },
    )
    minimum_gain: float = field(
        default=1e-8,
        metadata={
            "provenance": "PD03; predictive gain must exceed 1e-8 nats to apply the fitted evidence scale.",
        },
    )
    boundary_margin: float = field(
        default=0.01,
        metadata={
            "provenance": "PD03; exclude optima within 0.01 dimensionless log-scale units of a search endpoint.",
        },
    )
    refine_tolerance: float = field(
        default=1e-14,
        metadata={
            "provenance": "PD03; relative log-scale refinement tolerance 1e-14, scaled by max(1, abs(log scale)).",
        },
    )
    refine_iterations: int = field(
        default=60,
        metadata={
            "provenance": "PD03; at most 60 safeguarded refinement iterations per interior maximum.",
        },
    )
    bracket_expansions: int = field(
        default=30,
        metadata={
            "provenance": "PD03; at most 30 expansions to bracket the derivative root during refinement.",
        },
    )
    parallel_minimum_terms: int = field(
        default=200_000,
        metadata={
            "provenance": "PD03; 200000 likelihood terms trigger executor use; an execution work threshold.",
        },
    )


@dataclass(frozen=True)
class WinnerNumerics:
    """Exact winner and graph-confidence tolerances; no user-facing tuning (PD11-PD13)."""

    contrast_ulp_factor: float = field(
        default=512.0,
        metadata={
            "provenance": "PD11; 512 machine-epsilon units times score magnitude define the rounding tie band.",
        },
    )
    noise_floor: float = field(
        default=2.0**-15,
        metadata={
            "provenance": (
                "PD11; absolute tie floor of 2^-15 nats prevents small score perturbations selecting a winner."
            ),
        },
    )
    minimum_graph_quantum: float = field(
        default=1e-300,
        metadata={
            "provenance": "PD13; positive graph-confidence tie quantum floor of 1e-300 nats.",
        },
    )


# Fixed model numerics are versioned by source identity, not additional user settings.
CALIBRATION_NUMERICS = CalibrationNumerics()
WINNER_NUMERICS = WinnerNumerics()


@dataclass(frozen=True)
class PreparationParameters:
    """Marker subsampling and information tempering of the prepared painting components."""

    # At most this many evenly spaced selected markers per painting bin enter the transmission model.
    markers_per_bin: int = field(
        default=10,
        metadata={
            "provenance": "PS01; select at most 10 evenly spaced markers per painting bin for transmission scoring.",
        },
    )
    # Chromosome-wide groups of at least this many selected markers share one information exponent.
    markers_per_information_block: int = field(
        default=100,
        metadata={
            "provenance": "PS01; group at least 100 selected markers per information block for tempering.",
        },
    )
    # The tempered view raises a bin's emission to min(this, n) / n for n informative markers in its group.
    effective_markers_per_information_block: float = field(
        default=3.0,
        metadata={
            "provenance": "PS01; cap each information block at 3 effective markers through its emission exponent.",
        },
    )


@dataclass(frozen=True)
class ScoringParameters:
    """Projected transmission model and forward scoring of M0/M1/M2 configurations."""

    # Probability that an inherited allele is read as the other allele (child emission).
    mismatch_probability: float = field(
        default=0.01,
        metadata={
            "provenance": (
                "PS04; probability 0.01 of reading the inherited allele as the other allele in child emissions."
            ),
        },
    )
    # A candidate-parent bridge mixes in the uniform selector kernel with weight
    # switch probability x (1 - min(1, n / N)), n = the smaller informative selected-marker count of
    # the two adjacent parent bins, N = this value; 0 keeps the painter-interpolated chain.
    source_gap_support_sites: int = field(
        default=1,
        metadata={
            "provenance": (
                "PS04; 1 informative selected marker supplies full adjacent-bin support; less support mixes a "
                "uniform selector excursion."
            ),
        },
    )
    # Maximum-entropy bridge: post-check tolerance and independent bisection budget.
    bridge_postcheck_tolerance: float = field(
        default=2e-13,
        metadata={
            "provenance": (
                "PS04; dimensionless base tolerance 2e-13 for bridge probability and marginal residual checks."
            ),
        },
    )
    bridge_max_iterations: int = field(
        default=128,
        metadata={
            "provenance": "PS04; at most 128 bisection iterations for the maximum-entropy bridge root.",
        },
    )
    # Null source of an unobserved parent's slot: it starts from, and redraws from, the founder-state frequencies
    # transmitted by the other samples with an informative painting site (leave the child out), with a uniform
    # pseudo-count of this many samples (keeps a state no other sample carries possible).
    null_prior_samples: float = field(
        default=1.0,
        metadata={
            "provenance": (
                "PS04; 1 uniform pseudo-sample keeps every founder state possible in the leave-child-out null "
                "source."
            ),
        },
    )


@dataclass(frozen=True)
class PanelParameters:
    """Genome-wide M2 panel of bounded retained-parent pairs."""

    # A pair (p, q) of genome-retained M1 parents passes relevance screening when, in the full-data fit or in
    # some leave-one-chromosome-out fit, a valid upper bound on its joint M2 likelihood (pedigree.pair_bound: both
    # parents' transmission chains against the child's genotypes, bin by bin, summed over the fit's chromosomes)
    # comes within this many raw nats of the child's best known configuration there (M0 or a retained M1). A pair
    # left out therefore has an M2 more than this far below the best configuration in every such fit. 150 raw nats
    # are 33 nats at the 0.22 evidence scale fitted on bc8. Pairs outside contribute nothing to the M2 evidence (a
    # lower bound).
    pair_relevance_margin: float = field(
        default=150.0,
        metadata={
            "provenance": (
                "PS07; admit a retained-parent pair when its M2 upper bound is within 150 raw nats of the best "
                "known score in any relevance fit."
            ),
        },
    )
    # At most this many admitted pairs of a child enter the M2 panel (an owner decision), the best
    # by their largest bound gap over the relevance fits. Pairs tied with the last kept one (an equal gap) all enter,
    # and a pair whose gap is not a number is always kept. A child with more admitted pairs is flagged
    # (CandidatePanel.capped_children, the search diagnostics' m2_panel_capped): its M2 evidence then covers fewer
    # pairs, so its calls rest on an incomplete panel. The cap binds only at very low depth, where admitted pairs grow
    # with the square of the eligible parents (about 50,000 per child at 0.05x and N = 320, against about 1.4 at
    # 0.35x), making the M2 scoring cubic in N.
    max_pairs_per_child: int = field(
        default=256,
        metadata={
            "provenance": (
                "PS07; nominal cap of 256 admitted M2 pairs per child; boundary ties and unknown gaps remain, "
                "and binding is flagged."
            ),
        },
    )
    # States with at least this transmitted-state mass keep their own group in the pair bound (a parent's other named
    # states are aggregated). Any value keeps the bound valid; it only sets its tightness and cost.
    pair_bound_high_mass: float = field(
        default=0.01,
        metadata={
            "provenance": (
                "PS07; probability-mass cutoff 0.01 for separate named-state groups; controls bound tightness "
                "and cost."
            ),
        },
    )
    # Per-chromosome softmax temperature = (ceil(markers / markers_per_information_block)) ** power.
    information_tempering_power: float = field(
        default=0.5,
        metadata={
            "provenance": (
                "PS07; dimensionless exponent 0.5 on information-block count sets chromosome softmax "
                "temperature."
            ),
        },
    )
    # Chromosome weights sqrt(markers) / median are clipped to [1 / ratio, ratio].
    maximum_contig_weight_ratio: float = field(
        default=4.0,
        metadata={
            "provenance": (
                "PS07; dimensionless ratio 4 clips sqrt(marker count) relative to its median before normalizing "
                "weights."
            ),
        },
    )
    # Utility = rank_weight x tied rank evidence + (1 - rank_weight) x contaminated softmax.
    rank_weight: float = field(
        default=0.35,
        metadata={
            "provenance": (
                "PS07; probability weight 0.35 for tied-rank evidence in diagnostic utility; the remainder uses "
                "contaminated softmax."
            ),
        },
    )
    # Softmax mixed with the uniform distribution over eligible parents at this weight.
    chromosome_contamination: float = field(
        default=0.02,
        metadata={
            "provenance": (
                "PS07; probability weight 0.02 of the uniform eligible-parent distribution in chromosome "
                "diagnostic utility."
            ),
        },
    )


@dataclass(frozen=True)
class ExposureParameters:
    """Hard exposure gate: a configuration is testable only on enough exposed evidence."""

    minimum_edge_exposed_bins: float = field(
        default=1.0,
        metadata={
            "provenance": "PD05; require at least 1 posterior-expected exposed painting bin for an edge.",
        },
    )
    minimum_pair_exposed_bins: float = field(
        default=1.0,
        metadata={
            "provenance": "PD05; require at least 1 posterior-expected exposed painting bin for a parent pair.",
        },
    )
    # Exposed bins / total painting bins of the selected chromosomes.
    minimum_exposed_fraction: float = field(
        default=0.1,
        metadata={
            "provenance": "PD05; require an exposed-bin fraction of at least 0.1 across selected chromosomes.",
        },
    )
    # Working map units a fit's exposed chromosomes must hold (bootstrap draws count once). A
    # chromosome counts max(1, its scored map length / unit_morgans) units, and an edge (or panel pair) there
    # max(1, the map length it is exposed over / unit_morgans): an unlinked chromosome is an independent replicate
    # however short, and a long one holds several nearly independent stretches (one meiosis's transmission
    # correlation between loci d Morgans apart is exp(-2 d) under Haldane, 0.14 at 1 Morgan). With the resampling
    # thresholds Tier B needs 5 chromosomes shorter than unit_morgans, fewer when they are longer (resampling.reach).
    # Lengths are measured by a supplied genetic map, over the span its knots cover, or by a stated rate
    # (rate_is_measured); the fallback rate, and a map's extrapolated tails, are placeholders that measure no
    # recombination (contracts.genetic_map.measured_coordinates), so a chromosome with neither counts one unit.
    minimum_exposed_units: float = field(
        default=3.0,
        metadata={
            "provenance": (
                "PD05; require 3 working map units of exposed chromosomes; each distinct chromosome supplies at "
                "least one unit."
            ),
        },
    )
    unit_morgans: float = field(
        default=1.0,
        metadata={
            "provenance": (
                "PD05; 1 Morgan per working map unit measures approximately independent transmission stretches."
            ),
        },
    )
    # [genetic_map] rate_is_measured: the fallback rate is the organism's stated genome-wide average, so map lengths
    # count from it where no supplied map covers them (contracts.genetic_map.measured_coordinates).
    rate_is_measured: bool = field(
        default=False,
        metadata={
            "provenance": "PD05; boolean false leaves fallback genetic-map distances unmeasured for exposure units.",
        },
    )
    # Report-only thresholds of the C/X compatibility diagnostics (never gate a decision).
    report_edge_coverage: float = field(
        default=0.95,
        metadata={
            "provenance": "PD05; fraction 0.95 labels the edge-coverage diagnostic; does not gate decisions.",
        },
    )
    report_pair_explainability: float = field(
        default=0.95,
        metadata={
            "provenance": "PD05; fraction 0.95 labels the pair-explainability diagnostic; does not gate decisions.",
        },
    )


@dataclass(frozen=True)
class DirectionParameters:
    """Joint-transmission orientation and the reciprocal / short-ancestry family factors."""

    mismatch_probability: float = field(
        default=0.01,
        metadata={
            "provenance": (
                "PS19/PD06; dimensionless per-bin probability 0.01 that a meiotic copy has a different painted "
                "local class. Allows painting disagreement without making a direction impossible. Decision "
                "replays at 0.001, 0.01 and 0.1 agreed at Tier B for a real multi-generation cross, the TroMau "
                "cross and large known-truth simulations; 0.1 added false full-fit edges in a "
                "30-founder known-truth simulation."
            ),
        },
    )

    # Probability that one chromosome's junction comparison of a pair is painting failure.
    chromosome_error_probability: float = field(
        default=0.05,
        metadata={
            "provenance": (
                "PD06/PD08/PD09/PD10; probability 0.05 of painting failure in a chromosome junction comparison."
            ),
        },
    )
    # Pair-level painting-failure mixture bounding the orientation correction to
    # [log(epsilon), log(2 - epsilon)]. Independent of SelectionParameters.m0_contamination_probability, which has the
    # same default but a different role.
    pair_failure_probability: float = field(
        default=0.02,
        metadata={
            "provenance": "PD06/PD08/PD09/PD10; pair-level failure probability 0.02 bounds the orientation correction.",
        },
    )
    # Synchronous cavity-message passes preparing the frozen short-ancestry path factor.
    message_passes: int = field(
        default=4,
        metadata={
            "provenance": "PD06/PD08/PD09/PD10; 4 synchronous cavity passes prepare the frozen short-ancestry factor.",
        },
    )
    # Final reciprocal solve: undamped, then a damped retry; residual measured before damping.
    final_max_iterations: int = field(
        default=8192,
        metadata={
            "provenance": "PD06/PD08/PD09/PD10; budget of 8192 iterations for the undamped reciprocal solve.",
        },
    )
    # Undamped progress window: after two windows, an undamped solve that made no progress over the last window,
    # or whose geometric residual rate there needs more than final_pause_budgets x final_max_iterations
    # iterations in all to reach final_tolerance, pauses for the damped retry, and resumes where it paused if
    # the retry fails (so no fit that the full undamped budget would converge is lost). The forecast is a
    # heuristic: slow modes that speed up later can still pause; resumption is the safeguard.
    final_progress_window: int = field(
        default=512,
        metadata={
            "provenance": (
                "PD06/PD08/PD09/PD10; 512 residual observations per progress window before forecasting a damped "
                "retry."
            ),
        },
    )
    final_pause_budgets: int = field(
        default=4,
        metadata={
            "provenance": (
                "PD06/PD08/PD09/PD10; dimensionless multiplier 4 of the final iteration budget for the pause "
                "forecast."
            ),
        },
    )
    final_tolerance: float = field(
        default=1e-08,
        metadata={
            "provenance": (
                "PD06/PD08/PD09/PD10; maximum cavity-message change tolerance 1e-8 nats, measured before "
                "damping."
            ),
        },
    )
    retry_iterations: int = field(
        default=4096,
        metadata={
            "provenance": "PD06/PD08/PD09/PD10; at most 4096 iterations in the damped reciprocal retry.",
        },
    )
    retry_damping: float = field(
        default=0.5,
        metadata={
            "provenance": "PD06/PD08/PD09/PD10; dimensionless mixing weight 0.5 in the reciprocal retry.",
        },
    )
    # Top joint configurations per sample in the reverse two-edge path factor.
    ancestry_path_budget: int = field(
        default=16,
        metadata={
            "provenance": (
                "PD06/PD08/PD09/PD10; retain at most 16 joint configurations per sample in the reverse two-edge "
                "ancestry factor."
            ),
        },
    )


@dataclass(frozen=True)
class SelectionParameters:
    """Marginal parent-state priors, M0 contamination and acyclic graph search."""

    state_priors: tuple[float, float, float] = field(
        default=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        metadata={
            "provenance": (
                "PD04/PD12/PD13/PD14; dimensionless M0/M1/M2 probabilities of one third each before evidence."
            ),
        },
    )
    # Alternative fixed priors; an M0 call must survive all of them to be released.
    prior_sensitivity: tuple[tuple[float, float, float], ...] = field(
        default=((0.5, 0.3, 0.2), (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), (0.2, 0.3, 0.5)),
        metadata={
            "provenance": (
                "PD04/PD12/PD13/PD14; three fixed dimensionless M0/M1/M2 prior vectors; released M0 must survive "
                "all."
            ),
        },
    )
    # Each aggregated M1/M2 likelihood is mixed with the child's M0 likelihood at this weight.
    m0_contamination_probability: float = field(
        default=0.02,
        metadata={
            "provenance": (
                "PD04/PD12/PD13/PD14; probability weight 0.02 mixes child M0 into aggregated M1/M2 likelihoods."
            ),
        },
    )
    graph_local_search_passes: int = field(
        default=3,
        metadata={
            "provenance": "PD04/PD12/PD13/PD14; at most 3 coordinate passes of acyclic graph repair.",
        },
    )


@dataclass(frozen=True)
class ResamplingParameters:
    """Chromosome bootstrap (seeded) and leave-one-chromosome-out refits."""

    bootstrap_replicates: int = field(
        default=1000,
        metadata={
            "provenance": "PD16; 1000 chromosome-bootstrap draws for configuration and parent support fractions.",
        },
    )
    bootstrap_seed: int = field(
        default=20260725,
        metadata={
            "provenance": "PD16; integer seed 20260725 fixes the chromosome-bootstrap random stream.",
        },
    )


@dataclass(frozen=True)
class TierThresholds:
    """Release cut-offs of one evidence tier (fractions of resampled fits)."""

    configuration_bootstrap: float = field(
        metadata={
            "provenance": "PD18; minimum fraction of bootstrap fits selecting the configuration for this release tier.",
        },
    )
    parent_bootstrap: float = field(
        metadata={
            "provenance": (
                "PD18; minimum fraction of bootstrap fits selecting the parent identity for this release tier."
            ),
        },
    )
    loco_fraction: float = field(
        metadata={
            "provenance": "PD18; minimum fraction of leave-one-chromosome-out fits supporting this release tier.",
        },
    )


@dataclass(frozen=True)
class ReleaseParameters:
    """Tier A / Tier B release rules (Tier B is the primary product)."""

    # A tier call needs the child's own per-chromosome state evidence to pick a unique winner on chromosomes holding
    # at least this many map units (ExposureParameters.unit_morgans: three chromosomes shorter than one unit,
    # or without a genetic map, suffice).
    minimum_informative_units: float = field(
        default=3.0,
        metadata={
            "provenance": (
                "PD18/PD20/PD21; require 3 map units of chromosomes with a unique child-evidence state winner."
            ),
        },
    )
    tier_a: TierThresholds = field(
        default_factory=lambda: TierThresholds(0.95, 0.95, 0.95),
        metadata={
            "provenance": (
                "PD18/PD20/PD21; dimensionless configuration-bootstrap, parent-bootstrap and leave-one-out "
                "cutoffs (0.95, 0.95, 0.95)."
            ),
        },
    )
    tier_b: TierThresholds = field(
        default_factory=lambda: TierThresholds(0.7, 0.7, 0.8),
        metadata={
            "provenance": (
                "PD18/PD20/PD21; primary dimensionless configuration-bootstrap, parent-bootstrap and "
                "leave-one-out cutoffs (0.7, 0.7, 0.8)."
            ),
        },
    )
    # A released edge needs full-data orientation log odds >= log((1 - p) / p) (3.9 nats).
    orientation_reversal_probability: float = field(
        default=0.02,
        metadata={
            "provenance": (
                "PD18/PD20/PD21; reversal probability 0.02 gives the edge orientation cutoff log((1-p)/p) in "
                "nats."
            ),
        },
    )
    # Candidate sets carry this much configuration probability mass.
    support_set_coverage: float = field(
        default=0.95,
        metadata={
            "provenance": (
                "PD18/PD20/PD21; probability mass 0.95 retained by configuration candidate sets, including "
                "boundary ties."
            ),
        },
    )


@dataclass(frozen=True)
class ExclusionParameters:
    """Full-marker Mendelian exclusion releasing M0/M1 counts without new edges."""

    alpha: float = field(
        default=0.01,
        metadata={
            "provenance": "PD22; dimensionless exclusion level 0.01 sets the dyad log-e threshold log(2N/alpha).",
        },
    )
    genotype_replacement_probability: float = field(
        default=0.01,
        metadata={
            "provenance": (
                "PD22; probability 0.01 of uniform genotype replacement in the Mendelian compatibility ratio."
            ),
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


@dataclass(frozen=True)
class RelationshipParameters:
    """Painting relationship process and score (PD28).

    Rates are per Morgan, lengths in Morgans, isolated_error scales chance-infeasibility per known bin,
    withholding_score in nats, sibship_prior probabilities in increasing
    support bands, support_bounds dimensionless shares, reference_triples and
    sibship_check_partners counts.
    """

    failure_region_rate: float = field(
        default=0.1,
        metadata={
            "units": "1/Morgan",
            "provenance": "PD28; one observed painting failure in fourteen Morgans motivates 0.1 regions per Morgan.",
        },
    )
    failure_region_length: float = field(
        default=0.05,
        metadata={
            "units": "Morgan",
            "provenance": "PD28; observed painting failures span 4–8 cM, shorter than full-sib IBD0 segments.",
        },
    )
    isolated_error: float = field(
        default=1e-3,
        metadata={
            "units": "probability/bin",
            "provenance": "PD28; scale 1-q per bin; at most 0.03% wrong bins; no uninformative-bin penalty.",
        },
    )
    withholding_score: float = field(
        default=float(np.log(1000.0)),
        metadata={
            "units": "nats",
            "provenance": "PD23; log(1000) survives higher-prior sensitivity; two top-band witnesses suffice.",
        },
    )
    sibship_prior: tuple[float, ...] = field(
        default=(0.46, 0.87, 0.98),
        metadata={
            "units": "probability",
            "provenance": "PD28; measured full-sib shares among 3556 same-family pairs in three support bands.",
        },
    )
    support_bounds: tuple[float, ...] = field(
        default=(0.5, 0.75),
        metadata={
            "units": "share",
            "provenance": "PD28; support bands of the measured same-family sibship calibration.",
        },
    )
    sibship_check_partners: int = field(
        default=16,
        metadata={
            "units": "count",
            "provenance": "PD28; bounded ring check; 8 and 16 partners match all-partner validation scores.",
        },
    )
    reference_triples: int = field(
        default=400,
        metadata={
            "units": "count",
            "provenance": "PD28; 400 reference triples give about 0.02 sampling error in bin infeasibility.",
        },
    )

    def __post_init__(self) -> None:
        import math

        for name in ("failure_region_rate", "failure_region_length"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise SettingsError(f"pedigree relationship {name} must be finite and positive")
        if not 0 < self.isolated_error < 1:
            raise SettingsError("pedigree relationship isolated_error must lie in (0, 1)")
        if not math.isfinite(self.withholding_score) or self.withholding_score <= 0:
            raise SettingsError("pedigree relationship withholding_score must be finite and positive")
        if (
            len(self.sibship_prior) != 3
            or any(not 0 < p < 1 for p in self.sibship_prior)
            or tuple(sorted(self.sibship_prior)) != self.sibship_prior
        ):
            raise SettingsError("pedigree relationship sibship_prior must contain three increasing probabilities")
        if len(self.support_bounds) != 2 or not 0 <= self.support_bounds[0] < self.support_bounds[1] <= 1:
            raise SettingsError("pedigree relationship support_bounds must contain two increasing shares")
        if (
            isinstance(self.sibship_check_partners, bool)
            or not isinstance(self.sibship_check_partners, int)
            or self.sibship_check_partners < 1
        ):
            raise SettingsError("pedigree relationship sibship_check_partners must be a positive integer")
        if (
            isinstance(self.reference_triples, bool)
            or not isinstance(self.reference_triples, int)
            or self.reference_triples < 1
        ):
            raise SettingsError("pedigree relationship reference_triples must be a positive integer")


@dataclass(frozen=True)
class PedigreeParameters:
    """All pedigree settings; ``evidence_calibration`` is the only user-visible field."""

    # "predictive": one cross-chromosome evidence scale fitted on held-out chromosomes (<= 1);
    # "off": unscaled composite evidence.
    evidence_calibration: str = field(
        default="predictive",
        metadata={
            "provenance": (
                "PD27; categorical mode predictive fits a held-out-chromosome evidence weight; off uses unit "
                "weight."
            ),
        },
    )
    # The run's explicit tie convention (reverses arbitrary orders); never read from the environment.
    reverse_conventions: bool = field(
        default=False,
        metadata={
            "provenance": "PD27; boolean false selects the default explicit arbitrary tie convention.",
        },
    )
    preparation: PreparationParameters = field(
        default_factory=PreparationParameters,
        metadata={
            "provenance": (
                "PD27; internal marker-selection and tempering parameter group; units are declared by its "
                "fields."
            ),
        },
    )
    scoring: ScoringParameters = field(
        default_factory=ScoringParameters,
        metadata={
            "provenance": "PD27; internal projected-transmission parameter group; units are declared by its fields.",
        },
    )
    panel: PanelParameters = field(
        default_factory=PanelParameters,
        metadata={
            "provenance": "PD27; internal bounded-M2-search parameter group; units are declared by its fields.",
        },
    )
    exposure: ExposureParameters = field(
        default_factory=ExposureParameters,
        metadata={
            "provenance": "PD27; internal exposure-gate parameter group; units are declared by its fields.",
        },
    )
    direction: DirectionParameters = field(
        default_factory=DirectionParameters,
        metadata={
            "provenance": (
                "PD27; internal orientation and family-factor parameter group; units are declared by its fields."
            ),
        },
    )
    selection: SelectionParameters = field(
        default_factory=SelectionParameters,
        metadata={
            "provenance": (
                "PD27; internal state-prior and acyclic-search parameter group; units are declared by its "
                "fields."
            ),
        },
    )
    resampling: ResamplingParameters = field(
        default_factory=ResamplingParameters,
        metadata={
            "provenance": "PD27; internal chromosome-bootstrap parameter group; units are declared by its fields.",
        },
    )
    release: ReleaseParameters = field(
        default_factory=ReleaseParameters,
        metadata={
            "provenance": "PD27; internal evidence-tier parameter group; units are declared by its fields.",
        },
    )
    exclusion: ExclusionParameters = field(
        default_factory=ExclusionParameters,
        metadata={
            "provenance": "PD27; internal full-marker exclusion parameter group; units are declared by its fields.",
        },
    )
    relationship: RelationshipParameters = field(
        default_factory=RelationshipParameters,
        metadata={"provenance": "PD27; painting relationship evidence parameters, with units declared by each field."},
    )

    def __post_init__(self) -> None:
        if self.evidence_calibration not in EVIDENCE_CALIBRATION_MODES:
            raise SettingsError(
                f"pedigree evidence_calibration must be one of {EVIDENCE_CALIBRATION_MODES}, "
                f"not {self.evidence_calibration!r}"
            )


@dataclass(frozen=True)
class ScoringNumerics:
    """Fixed preparation and scoring numerical cutoffs, named independently of user-facing settings."""

    informative_gl_spread: float = field(
        default=1e-12,
        metadata={
            "provenance": (
                "PS01; genotype-likelihood max-minus-min spread must exceed 1e-12 to count a marker as "
                "informative."
            ),
        },
    )
    transmission_switch_floor: float = field(
        default=1e-12,
        metadata={
            "provenance": (
                "PS19; dimensionless probability floor 1e-12 keeps the scheduled joint transmission "
                "and founder-mosaic log transitions finite at equal map positions."
            ),
        },
    )
    selector_switch_floor: float = field(
        default=1e-15,
        metadata={
            "provenance": "PS04; probability floor 1e-15 keeps selector switch costs finite at equal map positions.",
        },
    )
    hall_ulps: float = field(
        default=64.0,
        metadata={
            "provenance": "PS06; 64 scaled machine-epsilon units identify bridge Hall equality up to rounding.",
        },
    )
    hub_mass_tie: float = field(
        default=2.0**-20,
        metadata={
            "provenance": "PS07; absolute probability-mass tie band 2^-20 selects a stable Hall hub among tied loads.",
        },
    )
    site_polynomial_threshold: float = field(
        default=1e-8,
        metadata={
            "provenance": (
                "PS09; dimensionless likelihood threshold 1e-8 selects direct emission evaluation to avoid "
                "polynomial cancellation."
            ),
        },
    )
    log_floor: float = field(
        default=-60.0,
        metadata={
            "provenance": "PS11; log-relative M1 score floor of -60 nats for nonretained parent identities.",
        },
    )

    tiny: float = field(
        default=1e-280,
        metadata={
            "provenance": (
                "PS15; positive mass threshold 1e-280 selects probability-domain rather than log-domain "
                "pair-bound absorption."
            ),
        },
    )
    family_exp_underflow: float = field(
        default=750.0,
        metadata={
            "provenance": "PD09; magnitude cutoff 750 nats skips exponentials whose float64 contributions vanish.",
        },
    )
    order_minimum_weight: float = field(
        default=2.0,
        metadata={
            "provenance": "PD07; require more than 2 weighted chromosome comparisons for graph-ordering support.",
        },
    )
    order_variance_offset: float = field(
        default=1.0,
        metadata={
            "provenance": (
                "PD07; additive variance offset 1 in squared junction-count units for graph-ordering support."
            ),
        },
    )
    junction_unreachable_state: int = field(
        default=1 << 30,
        metadata={
            "provenance": "PS18; integer cost sentinel 2^30 marks an unreachable minimum-switch trajectory state.",
        },
    )
    junction_unreachable_count: int = field(
        default=1 << 28,
        metadata={
            "provenance": "PS18; integer switch-count sentinel 2^28 marks unavailable junction counts.",
        },
    )

    null_rounding_ulps: float = field(
        default=64.0,
        metadata={
            "provenance": "PS09; 64 scaled machine-epsilon units tolerate rounding in null-source persistence.",
        },
    )

    bridge_feasibility_ulps: float = field(
        default=64.0,
        metadata={
            "provenance": "PS07; 64 scaled machine-epsilon units tolerate bridge flow feasibility rounding.",
        },
    )

    bridge_post_tolerance_factor: float = field(
        default=8.0,
        metadata={
            "provenance": "PS07; dimensionless multiplier 8 expands the configured bridge post-check tolerance.",
        },
    )

    bridge_post_ulps: float = field(
        default=256.0,
        metadata={
            "provenance": "PS07; 256 machine-epsilon units set the minimum bridge post-check tolerance.",
        },
    )

    bridge_bisection_limit: int = field(
        default=128,
        metadata={
            "provenance": "PS07; hard ceiling of 128 bisection halvings in the maximum-entropy bridge solve.",
        },
    )

    certificate_gl_ulps: int = field(
        default=64,
        metadata={
            "provenance": (
                "PS22; 64 scaled machine-epsilon units pad common genotype-likelihood logs outward in "
                "certificates."
            ),
        },
    )

    selective_upper_ulps: int = field(
        default=4096,
        metadata={
            "provenance": "PS22; 4096 scaled machine-epsilon units pad selective M1 upper bounds outward.",
        },
    )

    selective_upper_state_factor: int = field(
        default=64,
        metadata={
            "provenance": (
                "PS22; 64 times the state count supplies additional machine-epsilon padding for selective upper "
                "bounds."
            ),
        },
    )

    selective_bound_ulps: int = field(
        default=128,
        metadata={
            "provenance": "PS22; 128 scaled machine-epsilon units guard selective-bound coefficients outward.",
        },
    )

    selective_sum_ulps: int = field(
        default=8,
        metadata={
            "provenance": (
                "PS22; 8 machine-epsilon units per ordered accumulation term guard the selective chromosome sum."
            ),
        },
    )

    selective_frontier_ulps: int = field(
        default=32,
        metadata={
            "provenance": "PS22; 32 scaled machine-epsilon units guard best and floor frontiers in selective search.",
        },
    )

    pair_null_relative_guard: float = field(
        default=1e-12,
        metadata={
            "provenance": (
                "PS15; dimensionless relative padding 1e-12 inflates null-row maxima for conservative pair "
                "bounds."
            ),
        },
    )

    exclusion_tiny: float = field(
        default=1e-300,
        metadata={
            "provenance": (
                "PS20; positive probability floor 1e-300 protects exclusion ratio denominators and bet "
                "logarithms."
            ),
        },
    )


@dataclass(frozen=True)
class PedigreeWorkspace:
    """Storage dimensions only; estimates never reserve memory or limit concurrency."""

    relationship_sequence_chunk: int = field(
        default=32_768,
        metadata={
            "units": "count",
            "provenance": "PD28; bound binary sequence workspaces per contig task; execution grain only.",
        },
    )

    common_emission_cache_bytes: int = field(
        default=256 << 20,
        metadata={
            "provenance": "PS09; 268435456 bytes is the cutoff for retaining common child-emission caches.",
        },
    )

    bound_workspace_bytes: int = field(
        default=512 << 20,
        metadata={
            "provenance": (
                "PS15; workspace target 536870912 bytes sizes pair-bound batches without limiting concurrency."
            ),
        },
    )
    bound_batch_rows: int = field(
        default=4_000_000,
        metadata={
            "provenance": "PS15; at most 4000000 pair-bound workspace rows per batch.",
        },
    )
    exclusion_pattern_limit: int = field(
        default=1024,
        metadata={
            "provenance": (
                "PS20; cache at most 1024 distinct genotype-likelihood patterns per sample before direct "
                "fallback."
            ),
        },
    )

    emission_sample_block: int = field(
        default=16,
        metadata={
            "provenance": "PS01; batches of 16 samples bound live recomputed-emission and genotype-likelihood storage.",
        },
    )

    family_lanes_per_thread: int = field(
        default=8,
        metadata={
            "provenance": (
                "PD09; 8 interleaved independent-item lanes per supplied thread count distribute unequal family "
                "row counts."
            ),
        },
    )

    family_claim_work: int = field(
        default=20_000_000,
        metadata={
            "provenance": "PD09; target 20000000 row-scan work terms per cavity iteration chunk.",
        },
    )

    family_transpose_block: int = field(
        default=64,
        metadata={
            "provenance": "PD09; transpose cavity messages in blocks of 64 samples for contiguous child reads.",
        },
    )

    selective_bound_group_bins: int = field(
        default=8,
        metadata={
            "provenance": "PS22; group 8 consecutive bins when constructing selective-bound geometry.",
        },
    )

    exclusion_dyad_batch: int = field(
        default=256,
        metadata={
            "provenance": "PS20; batches of 256 dyads bound exclusion scanner workspaces.",
        },
    )


SCORING_NUMERICS = ScoringNumerics()
PEDIGREE_WORKSPACE = PedigreeWorkspace()
INFORMATIVE_GL_SPREAD = SCORING_NUMERICS.informative_gl_spread
SELECTOR_SWITCH_FLOOR = SCORING_NUMERICS.selector_switch_floor
TRANSMISSION_SWITCH_FLOOR = SCORING_NUMERICS.transmission_switch_floor
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
EXCLUSION_PATTERN_LIMIT = PEDIGREE_WORKSPACE.exclusion_pattern_limit

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
