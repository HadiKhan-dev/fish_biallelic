"""Settings of local selection (internal model constants; base production values).

Local selection refits every discovery block under a normalised diploid copying model (K
founder rows plus a fixed-mass unknown state whose copies draw Bernoulli(1/2) alleles) and
selects the panel by a drop / merge / add search, first from the raw discovery panel, then after
each context round (assembly through L1, then L1+L2) with the context's refitted founders as
candidates. None of these fields is user-visible (base never overrode one).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.ties import NOISE_SCALE

__all__ = [
    "PathModelConfig",
    "PathSearchConfig",
    "CandidateBankConfig",
    "PathReleaseConfig",
    "ContextRefitConfig",
    "SelectionParameters",
    "FLIP_TIE",
    "EXACT_FIRST_UPDATE_MAX_K",
    "DESCRIPTION_TIE",
    "MATCHING_SLACK",
    "LOG_RATIO_NEAR",
    "LANE_WIDTH",
    "LANE_ALIGN",
    "RELEASE_SAMPLE_BATCH",
    "COMBINE_TILE",
    "CONSENSUS_TIE",
    "PROPOSAL_FREQUENCY_TIE",
    "DECISION_MOTION",
    "CONSENSUS_CENSUS_WINDOW",
    "KIND_ORDER",
    "BACKGROUND_COUNT_BAND",
    "HALF_CENSUS_WINDOW",
    "FAVOURABLE_CENSUS_WINDOW",
    "UNKNOWN_CENSUS_WINDOW",
    "TASK_WORKSPACE_BASE_BYTES",
    "SEARCH_GROWTH",
    "PARAMETER_PROVENANCE",
]


@dataclass(frozen=True)
class PathModelConfig:
    """The copying model of a block.

    unknown_mass: fixed prior mass of the unknown state (base ``background_mass``; also the
        background-state mass of the context refit's bin-level HMM).

    redraw_rate_per_morgan: founder-copy redraw intensity per Morgan for both the block
        path HMM and the context-refit HMM. The redraw probability is
        r = 1 - exp(-redraw_rate_per_morgan x Morgans), using the contig's genetic map.
        A redraw can choose the same state; this is not the observed founder-switch rate
        or a sample's generation label. Independent of assembly's gap horizon.
    """

    unknown_mass: float = field(
        default=0.01,
        metadata={"provenance": "3.5 local_selection/config.py: PathModelConfig.unknown_mass; SE2 section 7"},
    )
    redraw_rate_per_morgan: float = field(
        default=3.0,
        metadata={"provenance": "3.5 local_selection/config.py: PathModelConfig.redraw_rate_per_morgan; SE2 section 7"},
    )


@dataclass(frozen=True)
class PathSearchConfig:
    """Drop / merge / add search with GEM refits (search.py, fit.py).

    refits_per_kind: top proposals per kind (drop, merge, add) refitted each round.
    round_backstop_base/_per_founder: reported round backstop, K of the current panel.
    update_backstop_base/_per_founder/_per_site: reported backstop of fixed-K GEM updates,
        base + per_founder K + per_site L.
    refits_per_kind_cap: release 2.2: the proposal cut takes whole tie classes of the screened
        score (at decision_tolerance) until at least refits_per_kind; a class that would take the
        refits of a kind past this cap is left out whole (counted as an overflow; no member of a
        tie class is preferred to another).
    tolerance: the GEM fit's own tolerance (frequency-step acceptance, flip gains, convergence; nats).
    decision_tolerance: tie quantum of every decision between EM-fitted or screened objectives
        (release 2.1): the proposal cut per kind, the round winner and the stop of the climb, the
        start-or-incumbent choice and the keep-incumbent rule between rounds; values chained within
        it are tied and decided by content (fewer founders, then the canonical panel bytes). A power
        of two far above EM convergence residuals and at least 16 times the typical motion of a
        compared contrast under 1e-6 relative GL noise (release 2.13: a search round's gain moves by
        about 5e-5 nats at the 99.9th percentile in the matched-history census of 1e-6 jitter runs;
        2.1 placed it against 1e-9 noise), far below
        genuine drop / merge / add differences (nats).
    """

    refits_per_kind: int = field(
        default=8,
        metadata={"provenance": "3.5 local_selection/config.py: PathSearchConfig.refits_per_kind; SE2 section 7"},
    )
    refits_per_kind_cap: int = field(
        default=16,
        metadata={"provenance": "3.5 local_selection/config.py: PathSearchConfig.refits_per_kind_cap; SE2 section 7"},
    )
    round_backstop_base: int = field(
        default=10,
        metadata={"provenance": "3.5 local_selection/config.py: PathSearchConfig.round_backstop_base; SE2 section 7"},
    )
    round_backstop_per_founder: int = field(
        default=2,
        metadata={
            "provenance": "3.5 local_selection/config.py: PathSearchConfig.round_backstop_per_founder; SE2 section 7"
        },
    )
    update_backstop_base: int = field(
        default=20,
        metadata={"provenance": "3.5 local_selection/config.py: PathSearchConfig.update_backstop_base; SE2 section 7"},
    )
    update_backstop_per_founder: int = field(
        default=4,
        metadata={
            "provenance": "3.5 local_selection/config.py: PathSearchConfig.update_backstop_per_founder; SE2 section 7"
        },
    )
    update_backstop_per_site: int = field(
        default=2,
        metadata={
            "provenance": "3.5 local_selection/config.py: PathSearchConfig.update_backstop_per_site; SE2 section 7"
        },
    )
    tolerance: float = field(
        default=1e-6,
        metadata={"provenance": "3.5 local_selection/config.py: PathSearchConfig.tolerance; SE2 section 7"},
    )
    decision_tolerance: float = field(
        default=2.0**-10,
        metadata={"provenance": "3.5 local_selection/config.py: PathSearchConfig.decision_tolerance; SE2 section 7"},
    )

    def round_backstop(self, k: int) -> int:
        return self.round_backstop_base + self.round_backstop_per_founder * k

    def update_backstop(self, k: int, length: int) -> int:
        return self.update_backstop_base + self.update_backstop_per_founder * k + self.update_backstop_per_site * length


@dataclass(frozen=True)
class CandidateBankConfig:
    """Candidate-bank search (bank.py): refits of the context proposals with discovery's
    fixed-K model, ranked by its binned-Viterbi BIC, K * complexity_scale * (L / 200) * N_active
    - 2 log L; K-scaled reported backstops with K the current panel's founder count: at most
    start_backstop(K) novel starts per call, score_backstop(K) admitted panels,
    round_backstop(K) rounds. Release 2.1: the block's discovery tie quanta (maths.ties
    BlockQuanta of the fixed-K workspace) order the BIC selection and the cavity-best endpoints
    (kind ``nll``) and stop the rounds (an improvement must exceed the ``improvement`` quantum).

    completion_variant_cap: release 2.2: a proposal panel whose partial rows have tied completions
        becomes one source per combination of them, up to this many; above it the panel's source
        keeps only its rows with a unique completion (counted as an overflow,
        "ls.bank.completion.overflow")."""

    round_backstop_base: int = field(
        default=8,
        metadata={
            "provenance": "3.5 local_selection/config.py: CandidateBankConfig.round_backstop_base; SE2 section 7"
        },
    )
    completion_variant_cap: int = field(
        default=8,
        metadata={
            "provenance": "3.5 local_selection/config.py: CandidateBankConfig.completion_variant_cap; SE2 section 7"
        },
    )
    score_backstop_base: int = field(
        default=1024,
        metadata={
            "provenance": "3.5 local_selection/config.py: CandidateBankConfig.score_backstop_base; SE2 section 7"
        },
    )
    score_backstop_per_founder: int = field(
        default=128,
        metadata={
            "provenance": "3.5 local_selection/config.py: CandidateBankConfig.score_backstop_per_founder; SE2 section 7"
        },
    )
    start_backstop_base: int = field(
        default=512,
        metadata={
            "provenance": "3.5 local_selection/config.py: CandidateBankConfig.start_backstop_base; SE2 section 7"
        },
    )
    start_backstop_per_founder: int = field(
        default=64,
        metadata={
            "provenance": "3.5 local_selection/config.py: CandidateBankConfig.start_backstop_per_founder; SE2 section 7"
        },
    )
    complexity_scale: float = field(
        default=0.5,
        metadata={"provenance": "3.5 local_selection/config.py: CandidateBankConfig.complexity_scale; SE2 section 7"},
    )

    def score_backstop(self, k: int) -> int:
        return self.score_backstop_base + self.score_backstop_per_founder * k

    def start_backstop(self, k: int) -> int:
        return self.start_backstop_base + self.start_backstop_per_founder * k

    def round_backstop(self, k: int) -> int:
        return self.round_backstop_base + -(-k // 2)


@dataclass(frozen=True)
class PathReleaseConfig:
    """Release of a selected panel's alleles (release.py): the one-bit conditional allele
    probability reaches ``min_call_probability`` and the favourable samples (whose own evidence
    favours the allele by more than ``favourable_margin`` nats) include ``min_supporters``
    carriers by count (posterior carrier probability above 1/2) or by expected carrier mass.
    Release 2.1: ``favourable_margin`` is the ``cell`` tie quantum of maths.ties (2^-14 nats since
    2.13, placed against 1e-6 relative GL noise; 2^-23 in 2.1-2.12; the 1e-9 of 2.0 sat at the motion
    of 1e-9 GL noise, so a symmetric-evidence cell, whose gain is exactly 0, straddled it).
    ``unknown_count_band``: a site's MAP number of unknown copies (0, 1, 2; it feeds assembly
    completion's all-or-nothing veto) must win by more than this relative posterior margin, else the
    smaller count (the ``probability`` tie quantum of maths.ties, 2^-15 since 2.13, 2^-20 before;
    design A-7: an integer count whose posterior masses can tie exactly)."""

    min_call_probability: float = field(
        default=0.85,
        metadata={"provenance": "3.5 local_selection/config.py: PathReleaseConfig.min_call_probability; SE2 section 7"},
    )
    min_supporters: float = field(
        default=2.0,
        metadata={"provenance": "3.5 local_selection/config.py: PathReleaseConfig.min_supporters; SE2 section 7"},
    )
    favourable_margin: float = field(
        default=2.0**-14,
        metadata={"provenance": "3.5 local_selection/config.py: PathReleaseConfig.favourable_margin; SE2 section 7"},
    )
    unknown_count_band: float = field(
        default=2.0**-15,
        metadata={"provenance": "3.5 local_selection/config.py: PathReleaseConfig.unknown_count_band; SE2 section 7"},
    )


@dataclass(frozen=True)
class ContextRefitConfig:
    """Refit of a context component's founders from the block's own evidence (context_refit.py).

    posterior_threshold: a refitted allele is released at this site-posterior probability.
    min_support: and with this carrier support (count or mass).
    emission_mix: mixing weight of a uniform genotype in the bin-level carrier emission.
    """

    posterior_threshold: float = field(
        default=0.99,
        metadata={"provenance": "3.5 local_selection/config.py: ContextRefitConfig.posterior_threshold; SE2 section 7"},
    )
    min_support: float = field(
        default=2.0,
        metadata={"provenance": "3.5 local_selection/config.py: ContextRefitConfig.min_support; SE2 section 7"},
    )
    emission_mix: float = field(
        default=0.01,
        metadata={"provenance": "3.5 local_selection/config.py: ContextRefitConfig.emission_mix; SE2 section 7"},
    )


@dataclass(frozen=True)
class SelectionParameters:
    """All local-selection settings (one record identity per round).

    context_levels: assembly levels of the context rounds, in order (rounds "l1", "l2").
    """

    model: PathModelConfig = field(
        default_factory=PathModelConfig,
        metadata={"provenance": "3.5 local_selection/config.py: LocalSelectionConfig.model; SE2 section 7"},
    )
    search: PathSearchConfig = field(
        default_factory=PathSearchConfig,
        metadata={"provenance": "3.5 local_selection/config.py: LocalSelectionConfig.search; SE2 section 7"},
    )
    bank: CandidateBankConfig = field(
        default_factory=CandidateBankConfig,
        metadata={"provenance": "3.5 local_selection/config.py: LocalSelectionConfig.bank; SE2 section 7"},
    )
    release: PathReleaseConfig = field(
        default_factory=PathReleaseConfig,
        metadata={"provenance": "3.5 local_selection/config.py: LocalSelectionConfig.release; SE2 section 7"},
    )
    context_refit: ContextRefitConfig = field(
        default_factory=ContextRefitConfig,
        metadata={"provenance": "3.5 local_selection/config.py: LocalSelectionConfig.context_refit; SE2 section 7"},
    )
    context_levels: tuple[int, ...] = field(
        default=(1, 2),
        metadata={"provenance": "3.5 local_selection/config.py: LocalSelectionConfig.context_levels; SE2 section 7"},
    )


# Scientific constants preserve the exact baseline expressions. Provenance is kept
# alongside resource and diagnostic constants, which do not alter decisions.
FLIP_TIE = 2.0**-20 * NOISE_SCALE
EXACT_FIRST_UPDATE_MAX_K = 16
DESCRIPTION_TIE = 1e-9
MATCHING_SLACK = 1e-12
LOG_RATIO_NEAR = 0.5
LANE_WIDTH = 64
LANE_ALIGN = 8
RELEASE_SAMPLE_BATCH = 128
COMBINE_TILE = 4096
CONSENSUS_TIE = 2.0**-20
PROPOSAL_FREQUENCY_TIE = 1e-12
DECISION_MOTION = 5e-5
CONSENSUS_CENSUS_WINDOW = 1e-3
KIND_ORDER = {"drop": 0, "merge": 1, "add": 2}
BACKGROUND_COUNT_BAND = 2.0**-15
HALF_CENSUS_WINDOW = 1e-3
FAVOURABLE_CENSUS_WINDOW = 1e-3
UNKNOWN_CENSUS_WINDOW = 1e-3
TASK_WORKSPACE_BASE_BYTES = 2048 << 20
SEARCH_GROWTH = 8

PARAMETER_PROVENANCE = {
    "FLIP_TIE": "3.5 fit.py:38; release 2.13 relative flip band scaled to 1e-6 GL noise",
    "EXACT_FIRST_UPDATE_MAX_K": "3.5 fit.py:41; first-update exact-enumeration cap",
    "DESCRIPTION_TIE": "3.5 objective.py shortest-description strict preference slack",
    "MATCHING_SLACK": "3.5 objective.py reduced-cost matching rounding allowance",
    "LOG_RATIO_NEAR": "3.5 path_model.py:50 _log_ratio close-ratio branch; not a decision tolerance",
    "LANE_WIDTH": "3.5 path_model.py:35; resource layout, 64 lanes",
    "LANE_ALIGN": "3.5 path_model.py:39; resource layout, align to 8 lanes",
    "RELEASE_SAMPLE_BATCH": "3.5 path_model.py:41; resource-only release batch",
    "COMBINE_TILE": "3.5 path_model.py:42; resource-only parallel reduction tile",
    "CONSENSUS_TIE": "3.5 search.py:51; noise_quantum(3.9e-8) == 2**-20",
    "PROPOSAL_FREQUENCY_TIE": "3.5 search.py:44; absolute duplicate-start frequency band",
    "DECISION_MOTION": "3.5 search.py:55; diagnostic boundary motion measured at p99.9",
    "CONSENSUS_CENSUS_WINDOW": "3.5 search.py:77; diagnostic census window",
    "KIND_ORDER": "3.5 search.py:79; fixed same-panel representative preference",
    "BACKGROUND_COUNT_BAND": "3.5 context_refit.py:43; probability quantum 2**-15",
    "HALF_CENSUS_WINDOW": "3.5 context_refit.py:45; diagnostic census window",
    "FAVOURABLE_CENSUS_WINDOW": "3.5 release.py:46; diagnostic census window",
    "UNKNOWN_CENSUS_WINDOW": "3.5 release.py:49; diagnostic census window",
    "TASK_WORKSPACE_BASE_BYTES": "3.5 rounds.py:259; resource-only workspace allowance",
    "SEARCH_GROWTH": "3.5 rounds.py:261; resource-only founder growth allowance",
}
