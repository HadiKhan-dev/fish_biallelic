"""Settings of raw block discovery (internal model constants; base production values).

None of these is user-visible: base never overrode a field. Every field enters the discovery
record identity. Backstops are K-scaled reported limits (``base + per_founder * K``), not
scientific bounds on the number of founders; generator sizes bound how many proposals of one
kind a search step fits, and their truncation is reported.

Terminology: an *unknown copy* is a sample chromosome copy that no panel row explains (base
"wildcard", assignment sentinel K); its allele is chosen freely per site at a fixed penalty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...core.ties import tie_quantum


@dataclass(frozen=True)
class FixedKModelConfig:
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
            "provenance": "discovery/config.py:27 (stage config); base production value (config docstring)",
        },
    )
    snps_per_bin: int = field(
        default=10, metadata={"units": "sites per bin", "provenance": "discovery/config.py:28; base production value"}
    )
    switch_penalty: float = field(
        default=10.0,
        metadata={
            "units": "nats per state switch between bins",
            "provenance": "discovery/config.py:29; base production value",
        },
    )
    max_iterations: int = field(
        default=50,
        metadata={"units": "A+H iterations per fit", "provenance": "discovery/config.py:30; base production value"},
    )


@dataclass(frozen=True)
class CavityScoreConfig:
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
            "provenance": "discovery/config.py:45; base production value",
        },
    )
    unknown_prior_mass: float = field(
        default=0.02,
        metadata={
            "units": "prior mass omega of states with an unknown copy",
            "provenance": "discovery/config.py:46; base production value",
        },
    )
    double_unknown_fraction: float = field(
        default=0.25,
        metadata={
            "units": "share delta of omega on (U,U)",
            "provenance": "discovery/config.py:47; base production value",
        },
    )
    mean_field_max_iterations: int = field(
        default=100,
        metadata={
            "units": "sweeps per held-out site solve and start",
            "provenance": "discovery/config.py:48; base production value",
        },
    )
    mean_field_tolerance: float = field(
        default=1e-10,
        metadata={
            "units": "max probability change; also endpoint-equivalence scale",
            "provenance": "discovery/config.py:49; base value; endpoint-class reuse release 2.3 (docs cavity section)",
        },
    )
    likelihood_floor: float = field(
        default=1e-300,
        metadata={
            "units": "probability floor before log",
            "provenance": "discovery/config.py:50; base production value",
        },
    )


@dataclass(frozen=True)
class ResidualBirthConfig:
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
            "provenance": "discovery/config.py:76; release 2.3 A256 campaign (docs Bounded proposal geometry)",
        },
    )
    residual_call_probability: float = field(
        default=0.80, metadata={"units": "probability", "provenance": "discovery/config.py:78; base production value"}
    )
    min_residual_joint_known_fraction: float = field(
        default=0.10,
        metadata={"units": "fraction of block sites", "provenance": "discovery/config.py:79; base production value"},
    )
    max_cluster_hamming: float = field(
        default=0.10,
        metadata={
            "units": "fraction of jointly called sites",
            "provenance": "discovery/config.py:80; base production value",
        },
    )
    candidate_call_probability: float = field(
        default=0.90, metadata={"units": "probability", "provenance": "discovery/config.py:81; base production value"}
    )
    min_candidate_known_fraction: float = field(
        default=0.80,
        metadata={"units": "fraction of block sites", "provenance": "discovery/config.py:82; base production value"},
    )
    min_dedup_joint_known_fraction: float = field(
        default=0.60,
        metadata={"units": "fraction of block sites", "provenance": "discovery/config.py:83; base production value"},
    )
    min_dedup_bidirectional_coverage: float = field(
        default=0.95, metadata={"units": "fraction", "provenance": "discovery/config.py:84; base production value"}
    )
    dedup_hamming_fraction: float = field(
        default=0.005,
        metadata={
            "units": "fraction of jointly called sites",
            "provenance": "discovery/config.py:85; base production value",
        },
    )
    min_soft_responsibility: float = field(
        default=0.25,
        metadata={"units": "expected copies", "provenance": "discovery/config.py:86; base production value"},
    )
    min_unique_sample_support: int = field(
        default=2, metadata={"units": "distinct samples", "provenance": "discovery/config.py:87; base production value"}
    )
    min_effective_sample_support: float = field(
        default=1.50,
        metadata={
            "units": "responsibility-weighted samples",
            "provenance": "discovery/config.py:88; base production value",
        },
    )
    max_alternative_groups: int = field(
        default=64,
        metadata={
            "units": "groups per connected component (straddle 128)",
            "provenance": "discovery/config.py:89; 2.3/2.4 defaults; per-component scope release 2.8 (docs)",
        },
    )


@dataclass(frozen=True)
class ReleaseRuleConfig:
    """Raw-discovery allele release (release.py): a founder allele is released when its capped
    fixed-assignment pseudo-probability reaches ``min_call_probability`` and at least
    ``min_supporters`` informative samples (of either direction) carry the founder there."""

    min_call_probability: float = field(
        default=0.85,
        metadata={"units": "pseudo-probability", "provenance": "discovery/config.py:98; base production value"},
    )
    min_supporters: int = field(
        default=2,
        metadata={
            "units": "informative carriers (either direction)",
            "provenance": "discovery/config.py:99; base production value",
        },
    )


@dataclass(frozen=True)
class SearchConfig:
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
        attempt budget; release 2.1 broader births, 3).
    data_birth_children_factor: data births stop once (K + 1) x this many distinct complete
        children exist (3).
    data_birth_tie_budget_factor: data-birth tie groups (samples tied on excess NLL and observed
        sites) union distinct seeds before admission; a whole group is omitted (reported as
        "data_birth_tie_class_at_budget") when its starts would exceed this multiple of the attempt
        budget (the hard cap).

    Release 2.3 budgets admit every straddling class through twice the nominal budget; an
    overlarge class is omitted whole and recorded separately from ordinary ranked truncation.

    Ties (release 2.1): every order, winner and improvement test of the search uses the block's tie
    quanta (maths.ties.BlockQuanta from the block's null NLL): NLL orders and the selection by tie
    classes of kind ``nll`` with the canonical key inside a class, improvement tests by the
    ``improvement`` quantum (replacing ``score_tolerance`` = 1e-9, which sat at the 1e-9-noise scale).
    """

    beam_width: int = field(
        default=6,
        metadata={
            "units": "panels per K (archive and parents; straddle 12)",
            "provenance": "discovery/config.py:136; 2.3/2.4 defaults (docs Current policy)",
        },
    )
    expansion_backstop_base: int = field(
        default=32, metadata={"units": "expansions", "provenance": "discovery/config.py:137; base; K-scaled backstop"}
    )
    expansion_backstop_per_founder: int = field(
        default=8,
        metadata={
            "units": "expansions per founder of K_max",
            "provenance": "discovery/config.py:138; base; K-scaled backstop",
        },
    )
    exact_score_backstop_base: int = field(
        default=96,
        metadata={"units": "cavity-scored panels", "provenance": "discovery/config.py:139; base; K-scaled backstop"},
    )
    exact_score_backstop_per_founder: int = field(
        default=12,
        metadata={
            "units": "cavity-scored panels per founder of K_max",
            "provenance": "discovery/config.py:140; base; K-scaled backstop",
        },
    )
    max_proposals_per_expansion: int = field(
        default=128,
        metadata={
            "units": "starts (death+gauge) / soft candidates (straddle 256)",
            "provenance": "discovery/config.py:141; 2.3/2.4 defaults",
        },
    )
    data_start_beam_width: int = field(
        default=6,
        metadata={
            "units": "K=1 fitted starts (straddle 12)",
            "provenance": "discovery/config.py:142; 2.3/2.4 defaults",
        },
    )
    data_seed_count: int = field(
        default=8,
        metadata={"units": "K=1 seed rows (straddle 16)", "provenance": "discovery/config.py:143; 2.3/2.4 defaults"},
    )
    seed_landmark_budget: int = field(
        default=256,
        metadata={"units": "anchors for K=1 seed pools", "provenance": "discovery/config.py:144; release 2.3 A256"},
    )
    seed_min_cluster_size: int = field(
        default=2,
        metadata={
            "units": "samples (HDBSCAN min cluster size and min samples)",
            "provenance": "discovery/config.py:145; base production value",
        },
    )
    max_replacement_children: int = field(
        default=48,
        metadata={
            "units": "replacement starts (straddle 96)",
            "provenance": (
                "discovery/config.py:146; adopted doubled-budget policy (docs Replacement admission; K30 trial)"
            ),
        },
    )
    exact_cut_max_k: int = field(
        default=12,
        metadata={"units": "rows", "provenance": "discovery/config.py:147; base; D9 closure release 2.5 (docs)"},
    )
    max_cut_ties: int = field(
        default=4,
        metadata={
            "units": "cuts (effective cap 4x = 16; family limit 32)",
            "provenance": "discovery/config.py:148; historical effective budget 4 x max_cut_ties (moves.py:1045)",
        },
    )
    data_birth_attempts_factor: int = field(
        default=3,
        metadata={
            "units": "multiplier of max(16, 2(K+1)) starts",
            "provenance": "discovery/config.py:149; release 2.1 broader births (config docstring)",
        },
    )
    data_birth_children_factor: int = field(
        default=3,
        metadata={"units": "multiplier of (K+1) children", "provenance": "discovery/config.py:150; release 2.1"},
    )
    data_birth_tie_budget_factor: int = field(
        default=2,
        metadata={
            "units": "multiple of the attempt budget (hard cap)",
            "provenance": "discovery/config.py:151; release 2.3 whole-class rule",
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
class DiscoveryConfig:
    """All raw-discovery settings (one record identity)."""

    model: FixedKModelConfig = field(
        default_factory=FixedKModelConfig, metadata={"provenance": "3.5 discovery/config.py:169-176; P0 DI01"}
    )
    cavity: CavityScoreConfig = field(
        default_factory=CavityScoreConfig, metadata={"provenance": "3.5 discovery/config.py:169-176; P0 DI01"}
    )
    search: SearchConfig = field(
        default_factory=SearchConfig, metadata={"provenance": "3.5 discovery/config.py:169-176; P0 DI01"}
    )
    residual_births: ResidualBirthConfig = field(
        default_factory=ResidualBirthConfig, metadata={"provenance": "3.5 discovery/config.py:169-176; P0 DI01"}
    )
    release: ReleaseRuleConfig = field(
        default_factory=ReleaseRuleConfig, metadata={"provenance": "3.5 discovery/config.py:169-176; P0 DI01"}
    )


# Numerical safety constants; values and evaluation forms are the 3.5 inventory's.
EMISSION_FLOOR = 1e-12  # DI03/DI26: discovery/fixed_k/workspace.py:42, release.py:149; probability.
RESIDUAL_GEOMETRY_SLACK = 1e-12  # DI18/DI19: residual_births.py:693,859-900,1127; fraction.
SIGMOID_SATURATION = 37.0  # DI10: cavity_kernels.py:45; logit where exp(-x) < 2**-53.
LOG_ODDS_MAX = math.log((1.0 - np.finfo(np.float64).eps) / np.finfo(np.float64).eps)  # DI17: residual_births.py:472.
LOG_ODDS_MIN = math.log(np.finfo(np.float64).tiny) - math.log1p(
    -np.finfo(np.float64).tiny
)  # DI17: residual_births.py:473.
PROBABILITY_BAND = tie_quantum(0.0, "probability")  # DI13/DI17/DI19; release 2.13 noise policy.
H_SUM_ERROR_FACTOR = 16.0  # DI05: fixed_k/update.py:332; certified error enclosure.
H_SUM_ERROR_CUTOFF = 0.0625  # DI05: fixed_k/update.py:330; count * float64 epsilon.
SET_COUNT_ROUNDING_FLOOR = -1e-12  # DI09: panel.py:478; roundoff below zero nats.

# Operational memo/tile bounds, outside the resolved scientific configuration.
MAX_CUT_CACHE_ENTRIES = 256  # discovery/moves.py:48; cached exact max-cut problems.
COUNT_CHUNK = 256  # discovery/moves.py:641; sample batch for omitted-start counts.
PREDICTION_TILE_BYTES = 8 * 1024 * 1024  # discovery/cavity_kernels.py:38; predictive workspace bytes.
PREDICTIVE_CHUNK_BYTES = 32 * 1024 * 1024  # discovery/residual_births.py:475; predictive workspace bytes.

# Public resolved-parameter name; the old spelling is retained for internal callers.
DiscoveryParameters = DiscoveryConfig

__all__ = [
    "MAX_CUT_CACHE_ENTRIES",
    "COUNT_CHUNK",
    "PREDICTION_TILE_BYTES",
    "PREDICTIVE_CHUNK_BYTES",
    "DiscoveryParameters",
    "DiscoveryConfig",
    "FixedKModelConfig",
    "CavityScoreConfig",
    "ResidualBirthConfig",
    "ReleaseRuleConfig",
    "SearchConfig",
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
