"""Bind the pedigree decision rules to explicit, immutable parameters.

The groups below are internal model constants, documented
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
    "WinnerNumerics",
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
    "FamilyTransmissionParameters",
    "MaterialParameters",
    "MapScaleParameters",
    "RATE_ESTIMATE_MODES",
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
    "ORDER_MINIMUM_UNIT_WEIGHT",
    "ORDER_VARIANCE_OFFSET",
    "DIRECTION_RELIABILITY_GRID",
    "DIRECTION_COIN_SHARE_GRID",
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
    "BRIDGE_FOLD_DISCRIMINANT",
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

    # Every eligible pair (p, q) of a child's genome-retained M1 parents enters the M2 panel: there is no relevance
    # screen (DESIGN_417 S8, 14 Nov 2026). A screen's exclusion guarantee covers only the fits it screens, and both
    # screens measured (full data alone; full data plus leave-one-chromosome-out fits) dropped pairs that bootstrap
    # and leave-one-unit-out fits select, changing released calls at low material. At most this many pairs of a
    # child enter (an owner decision): above it, the pairs with the largest full-data bound on their joint M2
    # likelihood (pedigree.pair_bound: both parents' transmission chains against the child's genotypes, bin by bin,
    # summed over the genome). Pairs tied with the last kept one (an equal bound) all enter, and a pair whose bound is
    # not a number is always kept. A child with more pairs is flagged (CandidatePanel.capped_children, the search
    # diagnostics' m2_panel_capped): its M2 evidence then covers fewer pairs, a lower bound, so its calls rest on an
    # incomplete panel. The cap binds when a child retains more than 23 parents (24 give 276 pairs): at low depth and
    # in large sib-rich cohorts. It bounds the panel at cap x N rows, so M2 scoring stays linear in N per child.
    max_pairs_per_child: int = field(
        default=256,
        metadata={
            "provenance": (
                "PS07; nominal cap of 256 M2 pairs per child, every retained eligible pair below it (no relevance "
                "screen, DESIGN_417 S8, owner 14 Nov 2026), the largest full-data bounds above it; boundary ties "
                "and unknown bounds remain, and binding is flagged."
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
    # Exposed bins / total painting bins of the fit's units.
    minimum_exposed_fraction: float = field(
        default=0.1,
        metadata={
            "provenance": "PD05; require an exposed-bin fraction of at least 0.1 across the fit's units.",
        },
    )
    # Material units (MaterialParameters: 1-Morgan units of the working map) a fit's exposed units must number
    # (bootstrap draws count once). Every unit counts one, whatever chromosome or component it lies on; the
    # low-material band (U < MaterialParameters.standard_morgans) needs one.
    minimum_exposed_units: float = field(
        default=3.0,
        metadata={
            "provenance": (
                "PD05 / DESIGN_417 D2; require 3 exposed material units (1-Morgan units of the working map) in the "
                "standard regime; 1 in the low-material band."
            ),
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

    # The probability that one unit's junction comparison of a pair is painting failure is not a parameter here:
    # it follows from the relationship model's two-state failure process (RelationshipParameters
    # failure_region_rate and failure_region_length) and the unit's working length (orientation.unit_failure,
    # DESIGN_417 D8). It protects against short failures only; the window cap below bounds long ones.
    # The reliability of a decisive unit contrast and the share of linked pairs whose units are fair coins are not
    # parameters either: each run estimates them from its own linked pairs (orientation.direction_model: posterior
    # means under uniform priors; no constant of this record enters except pair_failure_probability, which bounds
    # the structural orientation used as the orientation prior).
    window_morgans: float = field(
        default=0.0,
        metadata={
            "units": "Morgan",
            "provenance": (
                "DESIGN_417 rev 4; width W of the direction windows (consecutive windows of W working Morgans of "
                "the unit axis) whose summed direction log odds are capped at window_cap; 0 (default) turns the "
                "cap off. A count-free bound on a long (chromosome-wide) painting failure's votes, measured in V5 "
                "with a positive control before the owner chooses W."
            ),
        },
    )
    window_cap: float = field(
        default=float(np.log(39.0)),
        metadata={
            "units": "nats",
            "provenance": (
                "DESIGN_417 rev 4; ln 39 = 3.66 nats, 4.16.1's per-chromosome cap (failure probability 0.05), "
                "below the release cut-off ln 49, so one window cannot orient an edge alone."
            ),
        },
    )
    # Pair-level painting-failure mixture bounding the orientation correction to
    # [log(epsilon), log(2 - epsilon)]; the same probability bounds the structural orientation that serves as the
    # orientation prior of the run's direction model (orientation.structural_prior). Independent of
    # SelectionParameters.m0_contamination_probability, which has the same default but a different role.
    pair_failure_probability: float = field(
        default=0.02,
        metadata={
            "provenance": (
                "PD06/PD08/PD09/PD10; pair-level failure probability 0.02 bounds the orientation correction, and "
                "the orientation prior that the run's direction model reads from the direction-free solve "
                "(within [0.01, 0.99])."
            ),
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
    """Material-unit bootstrap (seeded) and leave-one-unit-out (LOBO) fits."""

    bootstrap_replicates: int = field(
        default=1000,
        metadata={
            "provenance": "PD16; 1000 unit-bootstrap draws for configuration and parent support fractions.",
        },
    )
    bootstrap_seed: int = field(
        default=20260725,
        metadata={
            "provenance": "PD16; integer seed 20260725 fixes the unit-bootstrap random stream.",
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
    lobo_fraction: float = field(
        metadata={
            "provenance": "PD18; minimum fraction of leave-one-unit-out (LOBO) fits supporting this release tier.",
        },
    )


@dataclass(frozen=True)
class ReleaseParameters:
    """Tier A / Tier B release rules (Tier B is the primary product)."""

    # A tier call needs the child's own per-unit state evidence to pick a unique winner on at least this many
    # material units (1 in the low-material band).
    minimum_informative_units: float = field(
        default=3.0,
        metadata={
            "provenance": (
                "PD18/PD20/PD21 / DESIGN_417 D2; require 3 material units with a unique child-evidence state winner."
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
            "provenance": (
                "PD23; log(1000) survives higher-prior sensitivity; with the 4.18 priors one top-band witness "
                "withholds only with a pair margin below -2.31 nats."
            ),
        },
    )
    sibship_prior: tuple[float, ...] = field(
        default=(0.67, 0.96, 0.99),
        metadata={
            "units": "probability",
            "provenance": (
                "PD28 / DESIGN_417 D5; full-sib shares of the guard's witness occurrences in three support bands, "
                "support on pooled Morgan-weighted 1-M units, pooled over three known-truth sets (4,159 "
                "occurrences, 73 runs; run-bootstrap intervals 0.54-0.79, 0.91-0.98, 0.97-1.00); every held-out "
                "fit changed no guard decision (RELCAL)."
            ),
        },
    )
    support_bounds: tuple[float, ...] = field(
        default=(0.5, 0.75),
        metadata={
            "units": "share",
            "provenance": "PD28; support bands of the sibship calibration, unchanged in 4.18 (DESIGN_417 D5).",
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
class FamilyTransmissionParameters:
    """PD29 family transmission test of retained Tier B parent edges (after the sibling guard's fixed point)."""

    role_prior: float = field(
        default=0.01,
        metadata={
            "units": "probability",
            "provenance": (
                "PD29; prior of each non-child role of a witness (a half child through either latent parent, a "
                "direct observation of a latent parent, or an outsider explained by its own mosaic); refitted per "
                "1-Morgan segment, so the prior alone charges at most ln((1 - 5 rho) / rho) = 4.55 nats per segment "
                "for a witness that is not a full sib (the decision rests on PAR - ALT). Fixed before the held-out "
                "runs (sibpar_tx VALIDATION_PLAN; half-child role: VALIDATION_PLAN_419b amendment 3)."
            ),
        },
    )
    minimum_witnesses: int = field(
        default=2,
        metadata={
            "units": "count",
            "provenance": (
                "PD29; edges with fewer retained witnesses (families of fewer than three children) are untested: "
                "with one witness the evidence is near-pairwise and a single non-sib witness dominates it (sibpar_tx "
                "R1 withheld 2 of 38,601 correct Tier A instances there; released sibs scored +10 to +18)."
            ),
        },
    )
    maximum_witnesses: int = field(
        default=12,
        metadata={
            "units": "count",
            "provenance": (
                "PD29; witness cap bounding the per-edge cost (fits are linear to quadratic in the witness count); "
                "witnesses are taken by relatedness member support, then observed-site counts in configured contig "
                "order (the PD24 convention), names last."
            ),
        },
    )
    maximum_iterations: int = field(
        default=30,
        metadata={"units": "count", "provenance": "PD29; MAP-EM iteration bound per fit (each step is monotone)."},
    )
    start_rounds: int = field(
        default=3,
        metadata={
            "units": "count",
            "provenance": "PD29; rounds of cross-initialisation between the three models (stops when none improves).",
        },
    )
    random_starts: int = field(
        default=3,
        metadata={
            "units": "count",
            "provenance": (
                "PD29; data-seeded block perturbations of each model's best paths per round (the review's restarts "
                "raised segment likelihoods by up to 32 nats without them)."
            ),
        },
    )

    def __post_init__(self) -> None:
        import math

        if not math.isfinite(self.role_prior) or not 0 < self.role_prior < 1 / 5:
            raise SettingsError("pedigree family_transmission role_prior must lie in (0, 1/5)")
        for name in ("minimum_witnesses", "maximum_witnesses", "maximum_iterations", "start_rounds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise SettingsError(f"pedigree family_transmission {name} must be a positive integer")
        if isinstance(self.random_starts, bool) or not isinstance(self.random_starts, int) or self.random_starts < 0:
            raise SettingsError("pedigree family_transmission random_starts must be a nonnegative integer")
        if self.maximum_witnesses < self.minimum_witnesses:
            raise SettingsError("pedigree family_transmission maximum_witnesses must be at least minimum_witnesses")


@dataclass(frozen=True)
class MaterialParameters:
    """The genome's material: equal-material units of the working map and the regime cut-offs (4.18, DESIGN_417).

    Every run resamples consecutive units of ``unit_morgans`` of the genome-ordered working map (configured contig
    order, then painting components in map order; the per-bin Haldane coordinates of the selector switch
    probabilities, the configured map whether or not it is declared measured, gaps between components contributing
    nothing). A final piece shorter than ``tail_morgans`` joins the unit before it, so a run holds round(U) units,
    where U is the working Morgans of its informative components. The regime reads U alone; no count of chromosomes,
    contigs or components enters it. Distances are Morgans of the configured working map: a wrong configured rate
    scales U, and with it every cut below, by the same factor.
    """

    unit_morgans: float = field(
        default=1.0,
        metadata={
            "units": "Morgan",
            "provenance": (
                "DESIGN_417 D1; one unit per working Morgan, about one meiosis's correlation length (Haldane "
                "exp(-2d) = 0.14 at 1 M); adjacent 1-M units correlate 0.12-0.22 for relatives on one chromosome, "
                "lag 2 at most 0.04 (CALIB2 section 5)."
            ),
        },
    )
    tail_morgans: float = field(
        default=0.5,
        metadata={
            "units": "Morgan",
            "provenance": (
                "DESIGN_4154 A2 / DESIGN_417 D1; a final piece shorter than half a unit joins the unit before it."
            ),
        },
    )
    switch_clip: float = field(
        default=0.499999,
        metadata={
            "units": "probability",
            "provenance": "DESIGN_4154 A2; switch probabilities are clipped below 1/2 before the Haldane distance.",
        },
    )
    assessable_morgans: float = field(
        default=5.0,
        metadata={
            "units": "Morgan",
            "provenance": (
                "MCAL section 4.1 (60 simulated genomes, configured map = generating map, so U in true Morgans; "
                "single chromosomes 0.5-15 M; pre-registered one-sided exact tests, clustered by genome); below 5 "
                "Morgans nothing is assessed: the low-material rule released non-parents in mixed-family designs "
                "at 3.5-5 M (12 of 971 against 0 of 600 at 5-7 M) and in small families; from 5 M every design "
                "passes the tested criterion (8 M on its literal point reading). V5 (64 end-to-end cells, "
                "g4c5_12): no hold criterion fires below 12 M. Replaces DESIGN_417 rev 3's 3.5 (MCAL: 3.5-5 M in excess)."
            ),
        },
    )
    standard_morgans: float = field(
        default=12.0,
        metadata={
            "units": "Morgan",
            "provenance": (
                "MCAL section 4.2; the released Tier B thresholds apply from 12 working Morgans: below it they "
                "released more non-parents than their own reference (multi-chromosome genomes of >= 15 M) in "
                "small families (11-12 M: 10 of 486 against 10 of 1,379, p = 0.035) and in parent-free sib-rich "
                "designs (10-11 M: 10 releases per 3,808 parentless children against 2 per 6,720, p = 0.008); "
                "from 12 M no excess in any design (10 M on the cumulative test; per-design power 0.1-0.4 near "
                "the cut, so an empirical calibration, not demonstrated equivalence). V5: rev 3's 9.5 released "
                "4.46 edges per 100 parentless children on sib-rich cells read on a 2.5x long map (hold H2); "
                "12 returns the 9.5-12 M band to the low-material rule, as 4.16.1 (0.11 per 100), and passes the "
                "direction positive control (SCORE_g4c5_12, CONTROL_SCORE_g4c5_12)."
            ),
        },
    )
    tier_a_morgans: float = field(
        default=12.0,
        metadata={
            "units": "Morgan",
            "provenance": (
                "MCAL section 4.3; no Tier A non-parent in designs with sequenced parents from 8 M (74,337 Tier A "
                "edges at 8-15+ M), but Tier A is assessed only with the released Tier B thresholds, so its "
                "effective minimum is standard_morgans (12 M). Parent-free sib-rich designs release Tier A "
                "non-parents at 12-13, 13-15 and >= 15 M (2, 2, 1; their >= 15-M reference holds 1), so no "
                "higher cut removes them (documented limitation; V5 g4c5_12: 1 such edge in 448 parentless "
                "children at 13.7 M)."
            ),
        },
    )
    low_bootstrap: float = field(
        default=0.95,
        metadata={
            "units": "fraction",
            "provenance": (
                "DESIGN_4154 A4; Tier B configuration and parent bootstrap cut-off of the low-material band "
                "(assessable_morgans <= U < standard_morgans), where adjacent units of one chromosome are dependent."
            ),
        },
    )
    low_leave_one_out: float = field(
        default=1.0,
        metadata={
            "units": "fraction",
            "provenance": "DESIGN_4154 A4; every leave-one-unit-out fit must agree in the low-material band.",
        },
    )

    def __post_init__(self) -> None:
        if not 0 < self.tail_morgans <= self.unit_morgans:
            raise SettingsError("pedigree material tail must lie in (0, unit]")
        if not 0 < self.switch_clip < 0.5:
            raise SettingsError("pedigree material switch clip must lie in (0, 0.5)")
        if not 0 < self.assessable_morgans <= self.standard_morgans <= self.tier_a_morgans:
            raise SettingsError("pedigree material cuts must increase: assessable <= standard <= Tier A")


# [genetic_map] rate_estimate: what the estimated recombination rates may change (DESIGN_RELEASE section 1).
RATE_ESTIMATE_MODES = ("veto", "ignore", "adopt")


@dataclass(frozen=True)
class MapScaleParameters:
    """The painting-based genome-wide recombination rate and its use by the material rule (4.19, DESIGN_MAPSCALE
    rev 5, DESIGN_RELEASE).

    One rate in cM/Mb (Morgans per bp = rate 1e-8) for the fallback stretches of the working map (contigs without a
    map file, stretches beyond a file's knots), estimated after painting from per-sib transmission frames of the
    relatedness families (error-run chain on the painting's own emissions). It never changes the working map of any
    stage; it decides only which material regime the pedigree applies (``mode``). Grids and guard constants are the
    rev-5 frozen estimator's (md5 404a44b6); the gate replaces rev 5's contig rule by information and material
    criteria (owner, 13 Nov: no count of chromosomes decides).
    """

    mode: str = field(
        default="veto",
        metadata={
            "provenance": (
                "Owner, 14 Nov 2026 ([genetic_map] rate_estimate): veto (default) releases only what both the working "
                "map's and the estimated map's regimes release; ignore reports the estimate only; adopt uses the "
                "estimated map's regime alone when its interval excludes the working rate (test-agree, review3 "
                "MUST-FIX 2), else both."
            ),
        },
    )
    guard_nats: float = field(
        default=10.0,
        metadata={
            "units": "nats",
            "provenance": (
                "DESIGN_MAPSCALE rev 4 G4: both relationship tests (pair, target) need a log-likelihood ratio above "
                "10 nats (p about 1e-5 on the chi-square(1) scale) to keep a pair or remove a target."
            ),
        },
    )
    target_share: float = field(
        default=0.2,
        metadata={
            "units": "fraction",
            "provenance": (
                "Rev 4 G4 (ii): a target is removed only when at least 20 % of its frames are posterior non-full-sib; "
                "the share criterion prevents selection on crossover counts (review R4 note 1)."
            ),
        },
    )
    target_iterations: int = field(
        default=12,
        metadata={"provenance": "Rev 4: at most 12 target-test rounds (one removal per family and round)."},
    )
    rate_grid: tuple[float, float, int] = field(
        default=(0.1, 50.0, 141),
        metadata={
            "units": "cM/Mb",
            "provenance": (
                "Rev 2-5: log-spaced rate grid (low, high, points) of the misfit-weight fit; the error-run chain "
                "uses every second point (71, log step 0.089). An estimate at either end is at the grid edge."
            ),
        },
    )
    coarse_rate_grid_spec: tuple[float, float, int] = field(
        default=(0.2, 50.0, 29),
        metadata={
            "units": "cM/Mb",
            "provenance": "Rev 4: the coarse log grid (low, high, points) of the relationship tests' nuisance rate.",
        },
    )
    misfit_weights: tuple[float, ...] = field(
        default=(0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 0.03, 0.1, 0.3),
        metadata={
            "units": "probability",
            "provenance": (
                "Rev 3: weight delta of the misfit column (mean over all state pairs: panel or painting misfit) "
                "mixed into each frame emission; profiled by maximum likelihood."
            ),
        },
    )
    pair_error_rates: tuple[float, ...] = field(
        default=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
        metadata={"units": "probability", "provenance": "Rev 2: observation error grid of the pair-test chains."},
    )
    target_entry_rates: tuple[float, ...] = field(
        default=(0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1),
        metadata={
            "units": "probability per frame",
            "provenance": "Rev 4 G4 (ii): entry grid of a track leaving its parents' classes (0 = the full-sib chain).",
        },
    )
    target_exit_rates: tuple[float, ...] = field(
        default=(3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3),
        metadata={"units": "probability per frame", "provenance": "Rev 4 G4 (ii): exit grid of such an excursion."},
    )
    error_run_starts: tuple[float, ...] = field(
        default=(0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
        metadata={
            "units": "probability per frame",
            "provenance": (
                "Rev 4 er: start probability eps of an error run (the shown state leaves the true transmission "
                "state); eps = 0 is the 4-state chain; profiled by maximum likelihood."
            ),
        },
    )
    error_run_ends: tuple[float, ...] = field(
        default=(0.02, 0.1, 0.4),
        metadata={"units": "probability per frame", "provenance": "Rev 4 er: end probability xi of an error run."},
    )
    consistency_alpha: float = field(
        default=0.01,
        metadata={
            "units": "probability",
            "provenance": (
                "Rev 5 R5.2: family-wise level of the family-consistency test (Bonferroni over the active families, "
                "t(df) reference of the quasi-likelihood z); null calibration 1 of 65 genomes (upper bound 7 %)."
            ),
        },
    )
    drop_margin: float = field(
        default=1.25,
        metadata={
            "provenance": (
                "Rev 4/5: with three or more families the most deviant family is dropped when |z| exceeds 1.25 "
                "z_crit (an identifiable outlier, e.g. a guard failure), and the test repeats."
            ),
        },
    )
    bootstrap_replicates: int = field(
        default=2000,
        metadata={"provenance": "Rev 5 B_J: percentile-bootstrap replicates of the joint interval (count)."},
    )
    minimum_interval_units: int = field(
        default=5,
        metadata={
            "provenance": (
                "The joint interval needs at least five equal-material units (working Morgans of frame axis) and two "
                "families: a percentile bootstrap over fewer units is degenerate. The same minimum as the relatedness "
                "estimate's (RateParameters.minimum_interval_units) and the pedigree's assessable material; it "
                "replaces rev 5's >= 5 contigs (owner, 13 Nov 2026). On the 195 lab genomes no adopted or usable "
                "estimate had fewer than 5 units (matrix f3: 3 units, refused by R2 and the usable floor)."
            ),
        },
    )
    bootstrap_seed: int = field(
        default=20261114,
        metadata={"provenance": "Fixed seed of the joint bootstrap (integer, rev 5's); the interval is reproducible."},
    )
    minimum_events: float = field(
        default=60.0,
        metadata={
            "units": "expected crossovers",
            "provenance": (
                "Rev 4/5 R2: adoption needs 60 expected crossovers on the kept families (2 x frame span x rate): "
                "about a 25 % half-width (R3) at Poisson precision; the information criterion that replaces rev 5's "
                ">= 5 contigs together with the material-unit interval (DESIGN_RELEASE section 4)."
            ),
        },
    )
    maximum_half_width: float = field(
        default=0.25,
        metadata={
            "units": "fraction",
            "provenance": "Rev 4/5 R3: adoption needs a joint 95 % interval half-width of at most 25 % of the estimate.",
        },
    )
    usable_minimum_events: float = field(
        default=20.0,
        metadata={
            "units": "expected crossovers",
            "provenance": (
                "Review3 SHOULD-FIX 4: a non-adopted estimate may still take part in the veto only with 20 expected "
                "crossovers and an interval whose lower end lies above the grid floor; every dev and held-out "
                "fallback had >= 46, the two V5 scaffold cells that decided on 19 and 4 events (interval lower end on "
                "the floor) are refused."
            ),
        },
    )

    def __post_init__(self) -> None:
        if self.mode not in RATE_ESTIMATE_MODES:
            raise SettingsError(f"rate_estimate must be one of {', '.join(RATE_ESTIMATE_MODES)}")

    def rate_grid_values(self) -> np.ndarray:
        """The fine log grid (cM/Mb)."""
        low, high, points = self.rate_grid
        return np.exp(np.linspace(np.log(low), np.log(high), int(points)))

    def error_run_grid(self) -> np.ndarray:
        """The error-run chain's rate grid: every second fine point (cM/Mb), contiguous."""
        return np.ascontiguousarray(self.rate_grid_values()[::2])

    def coarse_rate_grid(self) -> np.ndarray:
        """The relationship tests' coarse log grid (cM/Mb)."""
        low, high, points = self.coarse_rate_grid_spec
        return np.exp(np.linspace(np.log(low), np.log(high), int(points)))


@dataclass(frozen=True)
class PedigreeParameters:
    """Internal pedigree model and execution parameters."""

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
            "provenance": "PD27; internal unit-bootstrap parameter group; units are declared by its fields.",
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
    family_transmission: FamilyTransmissionParameters = field(
        default_factory=FamilyTransmissionParameters,
        metadata={"provenance": "PD29; family transmission test parameters, with units declared by each field."},
    )
    material: MaterialParameters = field(
        default_factory=MaterialParameters,
        metadata={"provenance": "DESIGN_417; material units and regime cut-offs, with units declared by each field."},
    )
    map_scale: MapScaleParameters = field(
        default_factory=MapScaleParameters,
        metadata={
            "provenance": (
                "4.19 (DESIGN_RELEASE); the painting-based rate estimate and the mode of its use by the material "
                "regime, with units declared by each field."
            ),
        },
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
    order_minimum_unit_weight: float = field(
        default=2.0,
        metadata={
            "provenance": (
                "PD07; require more than 2 callability-weighted unit comparisons (one per material unit since "
                "4.18) for graph-ordering support."
            ),
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
    direction_reliability_grid: int = field(
        default=40,
        metadata={
            "provenance": (
                "PD06; number of midpoints of (0, 1) on which the posterior of the run's direction reliability is "
                "summed (spacing 0.025; the posterior mean moves by less than 0.005 against 80 midpoints on the "
                "calibration layouts). Numerical resolution, not a model constant."
            ),
        },
    )
    direction_coin_share_grid: int = field(
        default=100,
        metadata={
            "provenance": (
                "PD06; number of midpoints of (0, 1) on which the posterior of the run's coin share is summed "
                "(spacing 0.01, so the smallest representable share is 0.005). Numerical resolution, not a model "
                "constant."
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

    bridge_fold_discriminant: float = field(
        default=2.0**-26,
        metadata={
            "provenance": (
                "PS07; sqrt(machine epsilon): an all-small root whose fold state has 1 - q L at or below it takes "
                "that state's diagonal product from the root equation, not from its square-root-folded quadratic."
            ),
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
ORDER_MINIMUM_UNIT_WEIGHT = SCORING_NUMERICS.order_minimum_unit_weight
ORDER_VARIANCE_OFFSET = SCORING_NUMERICS.order_variance_offset
DIRECTION_RELIABILITY_GRID = SCORING_NUMERICS.direction_reliability_grid
DIRECTION_COIN_SHARE_GRID = SCORING_NUMERICS.direction_coin_share_grid
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
BRIDGE_FOLD_DISCRIMINANT = SCORING_NUMERICS.bridge_fold_discriminant
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
