"""Settings of raw block discovery (internal model constants; base production values).

None of these is user-visible: base never overrode a field. Every field enters the discovery
record identity. Backstops are K-scaled reported limits (``base + per_founder * K``), not
scientific bounds on the number of founders; generator sizes bound how many proposals of one
kind a search step fits, and their truncation is reported.

Terminology: an *unknown copy* is a sample chromosome copy that no panel row explains (base
"wildcard", assignment sentinel K); its allele is chosen freely per site at a fixed penalty.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FixedKModelConfig:
    """The hard fixed-K factorisation H (K binary rows) x A (each sample's unordered row pair).

    unknown_copy_penalty: nats charged per unknown copy at every observed site (lambda).
    snps_per_bin: sites per bin of the switching Viterbi cost (one bin per site when a block has
        at most this many discovery sites).
    switch_penalty: nats per switch of a sample's diploid state between adjacent bins.
    max_iterations: coordinate-descent iterations (A step + H step) per fit.
    """

    unknown_copy_penalty: float = 0.5
    snps_per_bin: int = 10
    switch_penalty: float = 10.0
    max_iterations: int = 50


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

    state_concentration: float = 0.5
    unknown_prior_mass: float = 0.02
    double_unknown_fraction: float = 0.25
    mean_field_max_iterations: int = 100
    mean_field_tolerance: float = 1e-10
    likelihood_floor: float = 1e-300


@dataclass(frozen=True)
class ResidualBirthConfig:
    """Soft-residual birth proposals (residual_births.py). Fractions are of the block's sites.

    usable_founder_known_fraction: rows must be known at this fraction of sites to subtract.
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
    landmark_budget: int = 256
    usable_founder_known_fraction: float = 0.80
    residual_call_probability: float = 0.80
    min_residual_joint_known_fraction: float = 0.10
    max_cluster_hamming: float = 0.10
    candidate_call_probability: float = 0.90
    min_candidate_known_fraction: float = 0.80
    min_dedup_joint_known_fraction: float = 0.60
    min_dedup_bidirectional_coverage: float = 0.95
    dedup_hamming_fraction: float = 0.005
    min_soft_responsibility: float = 0.25
    min_unique_sample_support: int = 2
    min_effective_sample_support: float = 1.50
    max_alternative_groups: int = 64


@dataclass(frozen=True)
class ReleaseRuleConfig:
    """Raw-discovery allele release (release.py): a founder allele is released when its capped
    fixed-assignment pseudo-probability reaches ``min_call_probability`` and at least
    ``min_supporters`` informative samples (of either direction) carry the founder there."""

    min_call_probability: float = 0.85
    min_supporters: int = 2


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

    beam_width: int = 6
    expansion_backstop_base: int = 32
    expansion_backstop_per_founder: int = 8
    exact_score_backstop_base: int = 96
    exact_score_backstop_per_founder: int = 12
    max_proposals_per_expansion: int = 128
    data_start_beam_width: int = 6
    data_seed_count: int = 8
    seed_landmark_budget: int = 256
    seed_min_cluster_size: int = 2
    max_replacement_children: int = 48
    exact_cut_max_k: int = 12
    max_cut_ties: int = 4
    data_birth_attempts_factor: int = 3
    data_birth_children_factor: int = 3
    data_birth_tie_budget_factor: int = 2

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

    model: FixedKModelConfig = field(default_factory=FixedKModelConfig)
    cavity: CavityScoreConfig = field(default_factory=CavityScoreConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    residual_births: ResidualBirthConfig = field(default_factory=ResidualBirthConfig)
    release: ReleaseRuleConfig = field(default_factory=ReleaseRuleConfig)
