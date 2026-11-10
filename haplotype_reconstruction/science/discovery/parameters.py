"""Settings of raw block discovery (internal model constants).

Every internal control enters the discovery record identity. Backstops are K-scaled reported limits (``base + per_founder * K``), not
scientific bounds on the number of founders; generator sizes bound how many proposals of one
kind a search step fits, and their truncation is reported.

Terminology: an *unknown copy* is a sample chromosome copy that no panel row explains
(assignment sentinel K); its allele is chosen freely per site at a fixed penalty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...core.ties import tie_quantum


@dataclass(frozen=True)
class FixedKModelParameters:
    """The hard fixed-K factorisation H (K binary rows) x A (each sample's unordered row pair).

    unknown_copy_penalty: nats charged per unknown copy at every observed site (lambda).
    snps_per_bin: sites per bin of the switching Viterbi cost (one bin per site when a block has
        at most this many discovery sites).
    switch_penalty: nats per switch of a sample's diploid state between adjacent bins.
    max_iterations: coordinate-descent iterations (A step + H step) per fit.
    """

    unknown_copy_penalty: float = field(
        default=0.5,
        metadata={
            "units": "nats per unknown copy per observed site",
            "provenance": (
                "Penalty for an unexplained sample copy at an observed site (nats per copy per "
                "site); allows evidence unsupported by a panel row."
            ),
        },
    )
    snps_per_bin: int = field(
        default=10,
        metadata={
            "units": "sites per bin",
            "provenance": (
                "Target sites per switching-cost bin (sites); uses one bin per site for short "
                "blocks."
            ),
        },
    )
    switch_penalty: float = field(
        default=10.0,
        metadata={
            "units": "nats per state switch between bins",
            "provenance": (
                "Cost of a sample diploid-state change between bins (nats per switch); "
                "discourages within-block switching."
            ),
        },
    )
    max_iterations: int = field(
        default=50,
        metadata={
            "units": "A+H iterations per fit",
            "provenance": (
                "Maximum assignment-plus-founder coordinate iterations per fit (iterations); "
                "bounds a fixed-K fit."
            ),
        },
    )


@dataclass(frozen=True)
class CavityScoreParameters:
    """Fixed-assignment leave-one-sample-out mean-field cavity score of one panel.

    state_concentration: Dirichlet concentration of the diploid-state prior.
    unknown_prior_mass: prior mass of the states with an unknown copy (row-unknown and
        unknown-unknown).
    double_unknown_fraction: share of that mass on the unknown-unknown state.
    mean_field_max_iterations, mean_field_tolerance: per held-out site solve (three starts).
    likelihood_floor: lower bound of a genotype likelihood before its logarithm.
    """

    state_concentration: float = field(
        default=0.5,
        metadata={
            "units": "Dirichlet concentration alpha",
            "provenance": (
                "Diploid-state Dirichlet concentration alpha (dimensionless); smooths "
                "fixed-assignment state frequencies."
            ),
        },
    )
    unknown_prior_mass: float = field(
        default=0.02,
        metadata={
            "units": "prior mass omega of states with an unknown copy",
            "provenance": (
                "Prior mass on row-unknown and unknown-unknown states (probability); permits "
                "copies unexplained by the fitted rows."
            ),
        },
    )
    double_unknown_fraction: float = field(
        default=0.25,
        metadata={
            "units": "share delta of omega on (U,U)",
            "provenance": (
                "Share of unknown-state mass placed on the unknown-unknown state (fraction); "
                "divides that prior mass between one and two unknown copies."
            ),
        },
    )
    mean_field_max_iterations: int = field(
        default=100,
        metadata={
            "units": "sweeps per held-out site solve and start",
            "provenance": (
                "Maximum sweeps per held-out-site solve and start (sweeps); bounds each "
                "mean-field optimization."
            ),
        },
    )
    mean_field_tolerance: float = field(
        default=1e-10,
        metadata={
            "units": "max probability change; also endpoint-equivalence scale",
            "provenance": (
                "Maximum probability-change stopping tolerance and endpoint-equivalence scale "
                "(probability); resolves both solver convergence and predictive-mode classes at "
                "the same scale."
            ),
        },
    )
    likelihood_floor: float = field(
        default=1e-300,
        metadata={
            "units": "probability floor before log",
            "provenance": (
                "Minimum predictive likelihood before logarithms (probability); keeps cavity "
                "scores finite."
            ),
        },
    )


@dataclass(frozen=True)
class ResidualBirthParameters:
    """Soft-residual birth proposals (residual_births.py). Fractions are of the block's sites.

    residual_call_probability: residual strand allele called when its probability reaches this.
    min_residual_joint_known_fraction: residual pairs need this joint-known fraction for a
        finite Hamming distance (else distance 1).
    max_cluster_hamming: maximum pairwise Hamming distance inside a residual cluster.
    candidate_call_probability: consensus allele counted as known at this probability.
    min_candidate_known_fraction: a consensus needs this known fraction to be emitted.
    min_dedup_joint_known_fraction, min_dedup_bidirectional_coverage, dedup_hamming_fraction:
        a consensus within dedup_hamming_fraction of an existing row, with that joint-known
        fraction and coverage both ways, is a duplicate.
    min_soft_responsibility: minimum expected copies of a subtracted row for a residual record.
    min_unique_sample_support: distinct samples behind an emitted consensus.
    min_effective_sample_support: responsibility-weighted samples behind an emitted consensus.
    max_alternative_groups: soft count budget of alternative residual groups (edge-cover cliques),
        ranked by whole size classes. Admit a straddling class through twice this budget;
        omit it whole beyond that. Strictly smaller classes are ranked truncation.
    """

    # Bounded proposal geometry; zero explicitly selects the dense reference path.
    landmark_budget: int = field(
        default=256,
        metadata={
            "units": "anchors A",
            "provenance": (
                "Maximum residual geometry anchors (records); bounds proposal geometry, with zero "
                "selecting the dense reference path."
            ),
        },
    )
    residual_call_probability: float = field(
        default=0.80,
        metadata={
            "units": "probability",
            "provenance": (
                "Minimum residual allele posterior for a hard call (probability); leaves weaker "
                "strand evidence unknown."
            ),
        },
    )
    min_residual_joint_known_fraction: float = field(
        default=0.10,
        metadata={
            "units": "fraction of block sites",
            "provenance": (
                "Minimum jointly called block fraction for residual distance (fraction of sites); "
                "assigns distance one below that evidence level."
            ),
        },
    )
    max_cluster_hamming: float = field(
        default=0.10,
        metadata={
            "units": "fraction of jointly called sites",
            "provenance": (
                "Maximum within-cluster residual disagreement (fraction of jointly called sites); "
                "bounds residual-group diameter."
            ),
        },
    )
    candidate_call_probability: float = field(
        default=0.90,
        metadata={
            "units": "probability",
            "provenance": (
                "Minimum consensus allele probability counted as known (probability); defines "
                "consensus call coverage."
            ),
        },
    )
    min_candidate_known_fraction: float = field(
        default=0.80,
        metadata={
            "units": "fraction of block sites",
            "provenance": (
                "Minimum called block fraction of an emitted consensus (fraction of sites); "
                "excludes incomplete birth rows."
            ),
        },
    )
    min_dedup_joint_known_fraction: float = field(
        default=0.60,
        metadata={
            "units": "fraction of block sites",
            "provenance": (
                "Minimum shared called block fraction for duplicate comparison (fraction of "
                "sites); requires enough overlap to deduplicate rows."
            ),
        },
    )
    min_dedup_bidirectional_coverage: float = field(
        default=0.95,
        metadata={
            "units": "fraction",
            "provenance": (
                "Minimum call coverage in both duplicate-comparison directions (fraction); avoids "
                "merging rows on a small overlap."
            ),
        },
    )
    dedup_hamming_fraction: float = field(
        default=0.005,
        metadata={
            "units": "fraction of jointly called sites",
            "provenance": (
                "Maximum disagreement for residual-row deduplication (fraction of jointly called "
                "sites); treats sufficiently similar well-covered rows as duplicates."
            ),
        },
    )
    min_soft_responsibility: float = field(
        default=0.25,
        metadata={
            "units": "expected copies",
            "provenance": (
                "Minimum expected copies of a subtracted row (copies per sample); selects "
                "soft-residual records."
            ),
        },
    )
    min_unique_sample_support: int = field(
        default=2,
        metadata={
            "units": "distinct samples",
            "provenance": (
                "Minimum distinct samples behind a residual consensus (samples); prevents a "
                "single sample from seeding a birth."
            ),
        },
    )
    min_effective_sample_support: float = field(
        default=1.50,
        metadata={
            "units": "responsibility-weighted samples",
            "provenance": (
                "Minimum responsibility-weighted consensus support (samples); requires effective "
                "evidence beyond distinct-sample counts."
            ),
        },
    )
    max_alternative_groups: int = field(
        default=64,
        metadata={
            "units": "groups per connected component (straddle 128)",
            "provenance": (
                "Alternative residual-group budget per connected component (groups); admits a "
                "whole straddling size class through twice the budget and reports larger "
                "omissions."
            ),
        },
    )


@dataclass(frozen=True)
class ReleaseRuleParameters:
    """Raw-discovery allele release (release.py): a founder allele is released when its capped
    fixed-assignment pseudo-probability reaches ``min_call_probability`` and at least
    ``min_supporters`` informative samples (of either direction) carry the founder there."""

    min_call_probability: float = field(
        default=0.85,
        metadata={
            "units": "pseudo-probability",
            "provenance": (
                "Minimum capped fixed-assignment allele pseudo-probability (probability); with "
                "carrier support, gates raw-discovery allele release."
            ),
        },
    )
    min_supporters: int = field(
        default=2,
        metadata={
            "units": "informative carriers (either direction)",
            "provenance": (
                "Minimum informative founder carriers in either direction (samples); supplements "
                "the allele pseudo-probability gate."
            ),
        },
    )


@dataclass(frozen=True)
class SearchParameters:
    """Adaptive reversible search over complete panels (search.py, moves.py).

    beam_width: low-NLL basins expanded at the selected K.
    expansion_backstop_base/_per_founder, exact_score_backstop_base/_per_founder: reported
        backstops, K = the largest K represented in the search archive.
    max_proposals_per_expansion: generator size of deaths + gauge rewires, and of soft births.
    data_start_beam_width: K=1 data-seed fits kept as search starts.
    data_seed_count: soft-cluster seeds tried for K=1.
    seed_min_cluster_size: HDBSCAN minimum cluster size of the K=1 seeds.
    max_replacement_children: generator size of birth-delete replacements per expansion.
    exact_cut_max_k: gauge rewires use exact maximum cuts up to this K, a degree-ordered local
        search above (reported separately from search limits).
    max_cut_ties: nominal maximum-cut multiplicity; the effective family budget is four times
        this value, with whole-family admission through twice that effective budget.
    data_birth_attempts_factor: data births fit at most max(16, 2(K + 1)) x this many starts (the
        attempt budget).
    data_birth_children_factor: data births stop once (K + 1) x this many distinct complete
        children exist (3).
    data_birth_tie_budget_factor: data-birth tie groups (samples tied on excess NLL and observed
        sites) union distinct seeds before admission; a whole group is omitted (reported as
        "data_birth_tie_class_at_budget") when its starts would exceed this multiple of the attempt
        budget (the hard cap).

    Ranked search budgets generally admit a straddling class through twice the nominal budget; an
    overlarge class is omitted whole and recorded separately from ordinary ranked truncation.
    Landmark geometry uses its own non-straddling anchor budget.

    Ties: every order, winner and improvement test of the search uses the block's tie
    quanta (core.ties.BlockQuanta from the block's null NLL): NLL orders and the selection by tie
    classes of kind ``nll`` with the canonical key inside a class, improvement tests by the
    ``improvement`` quantum.
    """

    beam_width: int = field(
        default=6,
        metadata={
            "units": "panels per K (archive and parents; straddle 12)",
            "provenance": (
                "Archive and parent beam per founder count (panels); admits whole NLL classes "
                "through twice the nominal width."
            ),
        },
    )
    expansion_backstop_base: int = field(
        default=32,
        metadata={
            "units": "expansions",
            "provenance": (
                "Base search expansion allowance (expansions); combined with the per-founder "
                "allowance and reported when binding."
            ),
        },
    )
    expansion_backstop_per_founder: int = field(
        default=8,
        metadata={
            "units": "expansions per founder of K_max",
            "provenance": (
                "Extra expansion allowance per largest archived founder count (expansions per "
                "founder); scales the reported backstop with K."
            ),
        },
    )
    exact_score_backstop_base: int = field(
        default=96,
        metadata={
            "units": "cavity-scored panels",
            "provenance": (
                "Base cavity-scoring allowance (panels); combined with the per-founder allowance "
                "and reported when binding."
            ),
        },
    )
    exact_score_backstop_per_founder: int = field(
        default=12,
        metadata={
            "units": "cavity-scored panels per founder of K_max",
            "provenance": (
                "Extra cavity-scoring allowance per largest archived founder count (panels per "
                "founder); scales the reported backstop with K."
            ),
        },
    )
    max_proposals_per_expansion: int = field(
        default=128,
        metadata={
            "units": "starts (death+gauge) / soft candidates (straddle 256)",
            "provenance": (
                "Death-plus-gauge starts or soft-birth candidates per expansion (starts or "
                "candidates); admits a whole straddling class through twice the nominal budget."
            ),
        },
    )
    data_start_beam_width: int = field(
        default=6,
        metadata={
            "units": "K=1 fitted starts (straddle 12)",
            "provenance": (
                "K=1 fitted-start beam (panels); retains whole leading NLL classes with the "
                "straddling allowance."
            ),
        },
    )
    data_seed_count: int = field(
        default=8,
        metadata={
            "units": "K=1 seed rows (straddle 16)",
            "provenance": (
                "K=1 seed-row budget (rows); retains whole ranked seed classes with the "
                "straddling allowance."
            ),
        },
    )
    seed_landmark_budget: int = field(
        default=256,
        metadata={
            "units": "anchors for K=1 seed pools",
            "provenance": (
                "Maximum geometry anchors for K=1 seed pools (samples); bounds proposal geometry."
            ),
        },
    )
    seed_min_cluster_size: int = field(
        default=2,
        metadata={
            "units": "samples (HDBSCAN min cluster size and min samples)",
            "provenance": (
                "HDBSCAN minimum cluster size and minimum samples for K=1 seeds (samples); sets "
                "the smallest supported seed group."
            ),
        },
    )
    max_replacement_children: int = field(
        default=48,
        metadata={
            "units": "replacement starts (straddle 96)",
            "provenance": (
                "Birth-delete replacement-start budget (starts); admits a whole straddling class "
                "through twice the nominal budget."
            ),
        },
    )
    exact_cut_max_k: int = field(
        default=12,
        metadata={
            "units": "rows",
            "provenance": (
                "Largest founder count using exhaustive maximum cuts (rows); larger panels use "
                "degree-ordered local-search proposals."
            ),
        },
    )
    max_cut_ties: int = field(
        default=4,
        metadata={
            "units": "cuts (effective cap 4x = 16; family limit 32)",
            "provenance": (
                "Quarter of the nominal gauge-cut family budget (cuts); four times this value "
                "gives 16 nominal cuts, with whole-family admission through 32 at the default."
            ),
        },
    )
    data_birth_attempts_factor: int = field(
        default=3,
        metadata={
            "units": "multiplier of max(16, 2(K+1)) starts",
            "provenance": (
                "Multiplier of max(16, 2(K+1)) data-birth starts (dimensionless); scales the "
                "per-call attempt budget."
            ),
        },
    )
    data_birth_children_factor: int = field(
        default=3,
        metadata={
            "units": "multiplier of (K+1) children",
            "provenance": (
                "Multiplier of K+1 distinct data-birth children (dimensionless); stops a call "
                "after enough complete children exist."
            ),
        },
    )
    data_birth_tie_budget_factor: int = field(
        default=2,
        metadata={
            "units": "multiple of the attempt budget (hard cap)",
            "provenance": (
                "Hard-cap multiplier of the data-birth attempt budget (dimensionless); omits an "
                "overlarge tied start group whole."
            ),
        },
    )

    def data_birth_attempts(self, k: int) -> int:
        """Attempt budget of one data-birth call from a K-row panel."""
        return max(16, 2 * (int(k) + 1)) * max(1, int(self.data_birth_attempts_factor))

    def data_birth_children(self, k: int) -> int:
        """Distinct complete children after which one data-birth call from a K-row panel stops."""
        return (int(k) + 1) * max(1, int(self.data_birth_children_factor))

    def expansion_backstop(self, k: int) -> int:
        return self.expansion_backstop_base + self.expansion_backstop_per_founder * k

    def exact_score_backstop(self, k: int) -> int:
        return self.exact_score_backstop_base + self.exact_score_backstop_per_founder * k


@dataclass(frozen=True)
class DiscoveryParameters:
    """All raw-discovery settings (one record identity)."""

    model: FixedKModelParameters = field(
        default_factory=FixedKModelParameters,
        metadata={
            "provenance": (
                "Fixed-K assignment and allele-fit controls (parameter group); defines the block "
                "factorization objective."
            ),
        },
    )
    cavity: CavityScoreParameters = field(
        default_factory=CavityScoreParameters,
        metadata={
            "provenance": (
                "Held-out predictive-scoring controls (parameter group); compares complete fitted "
                "panels."
            ),
        },
    )
    search: SearchParameters = field(
        default_factory=SearchParameters,
        metadata={
            "provenance": (
                "Panel-search and reported work-budget controls (parameter group); governs "
                "candidate exploration."
            ),
        },
    )
    residual_births: ResidualBirthParameters = field(
        default_factory=ResidualBirthParameters,
        metadata={
            "provenance": (
                "Soft-residual birth controls (parameter group); governs strand grouping and "
                "consensus admission."
            ),
        },
    )
    release: ReleaseRuleParameters = field(
        default_factory=ReleaseRuleParameters,
        metadata={
            "provenance": (
                "Raw-discovery allele-release controls (parameter group); retains only "
                "probability- and carrier-supported calls."
            ),
        },
    )


# Numerical probability bounds and rounding allowances.
EMISSION_FLOOR = 1e-12  # Minimum fixed-K emission probability before logarithms.
RESIDUAL_GEOMETRY_SLACK = 1e-12  # Fractional slack for residual geometry boundary comparisons.
SIGMOID_SATURATION = 37.0  # Logit where exp(-x) is below 2**-53.
LOG_ODDS_MAX = math.log((1.0 - np.finfo(np.float64).eps) / np.finfo(np.float64).eps)  # Maximum residual log odds (nats), finite at one minus float64 epsilon.
LOG_ODDS_MIN = math.log(np.finfo(np.float64).tiny) - math.log1p(
    -np.finfo(np.float64).tiny
)  # Minimum residual log odds (nats), finite at float64 tiny.
PROBABILITY_BAND = tie_quantum(0.0, "probability")  # Probability comparison band from the common GL-noise policy.
H_SUM_ERROR_FACTOR = 16.0  # Error-enclosure multiplier (dimensionless).
H_SUM_ERROR_CUTOFF = 0.0625  # Maximum count times float64 epsilon for the error enclosure.
SET_COUNT_ROUNDING_FLOOR = -1e-12  # Allowed roundoff below zero description length (nats).

# Operational memo/tile bounds, outside the resolved scientific configuration.
MAX_CUT_CACHE_ENTRIES = 256  # Maximum memoized exact max-cut problems (entries).
COUNT_CHUNK = 256  # Samples per omitted-start counting batch.
PREDICTION_TILE_BYTES = 8 * 1024 * 1024  # Predictive contraction workspace target (bytes).
PREDICTIVE_CHUNK_BYTES = 32 * 1024 * 1024  # Residual predictive workspace target (bytes).

__all__ = [
    "MAX_CUT_CACHE_ENTRIES",
    "COUNT_CHUNK",
    "PREDICTION_TILE_BYTES",
    "PREDICTIVE_CHUNK_BYTES",
    "DiscoveryParameters",
    "FixedKModelParameters",
    "CavityScoreParameters",
    "ResidualBirthParameters",
    "ReleaseRuleParameters",
    "SearchParameters",
    "EMISSION_FLOOR",
    "RESIDUAL_GEOMETRY_SLACK",
    "SIGMOID_SATURATION",
    "LOG_ODDS_MAX",
    "LOG_ODDS_MIN",
    "PROBABILITY_BAND",
    "H_SUM_ERROR_FACTOR",
    "H_SUM_ERROR_CUTOFF",
    "SET_COUNT_ROUNDING_FLOOR",
]
