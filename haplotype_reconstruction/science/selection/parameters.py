"""Settings of local selection (internal model constants).

Local selection refits every discovery block under a normalised diploid copying model (K
founder rows plus a fixed-mass unknown state whose copies draw Bernoulli(1/2) alleles) and
selects the panel by a drop / merge / add search, first from the raw discovery panel, then after
each context round (assembly through L1, then L1+L2) with the context's refitted founders as
candidates. These settings are internal to the selection model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.ties import NOISE_SCALE

__all__ = [
    "CHAIN_LOW",
    "CHAIN_HIGH",
    "PathModelParameters",
    "PathSearchParameters",
    "CandidateBankParameters",
    "PathReleaseParameters",
    "ContextRefitParameters",
    "SelectionParameters",
    "FLIP_TIE",
    "EXACT_FIRST_UPDATE_MAX_K",
    "DESCRIPTION_TIE",
    "DESCRIPTION_MEMO_ENTRIES",
    "MERGE_EXACT_CODES_PER_ROUND",
    "MATCHING_SLACK",
    "LOG_RATIO_NEAR",
    "LANE_WIDTH",
    "LANE_ALIGN",
    "LANE_MIN_WIDTH",
    "CHUNK_BLOCKS",
    "CHUNK_WAVES",
    "CHUNK_FLOOR",
    "CHUNK_TASKS_PER_CPU",
    "SHARE_REREAD_SECONDS",
    "SMALL_COHORT_TASKS_PER_CPU",
    "SMALL_COHORT_CHUNK_BLOCKS",
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
    "RACE_MIN_STARTS",
    "RACE_PROBE_ITERATIONS",
    "RACE_KEEP",
]


@dataclass(frozen=True)
class PathModelParameters:
    """The copying model of a block.

    unknown_mass: fixed prior mass of the unknown state (also the background-state mass of the context refit's
    bin-level HMM).

    redraw_rate_per_morgan: founder-copy redraw intensity per Morgan for both the block
        path HMM and the context-refit HMM. The redraw probability is
        r = 1 - exp(-redraw_rate_per_morgan x Morgans), using the contig's genetic map.
        A redraw can choose the same state; this is not the observed founder-switch rate
        or a sample's generation label. Independent of assembly's gap horizon.
    """

    unknown_mass: float = field(
        default=0.01,
        metadata={
            "units": "probability",
            "provenance": "Fixed unknown-state prior mass 0.01 permits unexplained founder copies.",
        },
    )
    redraw_rate_per_morgan: float = field(
        default=3.0,
        metadata={
            "units": "1/Morgan",
            "provenance": "Copy redraw intensity 3 gives redraw probability 1-exp(-3 times interval Morgans).",
        },
    )


@dataclass(frozen=True)
class PathSearchParameters:
    """Drop / merge / add search with GEM refits (search.py, fit.py).

    refits_per_kind: top proposals per kind (drop, merge, add) refitted each round.
    round_backstop_base/_per_founder: reported round backstop, K of the current panel.
    update_backstop_base/_per_founder/_per_site: reported backstop of fixed-K GEM updates,
        base + per_founder K + per_site L.
    refits_per_kind_cap: the proposal cut takes whole tie classes of the screened
        score (at decision_tolerance) until at least refits_per_kind; a class that would take the
        refits of a kind past this cap is left out whole (counted as an overflow; no member of a
        tie class is preferred to another).
    tolerance: the GEM fit's own tolerance (frequency-step acceptance, flip gains, convergence; nats).
    decision_tolerance: tie quantum of every decision between EM-fitted or screened objectives
        for the proposal cut per kind, the round winner and the stop of the climb, the
        start-or-incumbent choice and the keep-incumbent rule between rounds; values chained within
        it are tied and decided by content (fewer founders, then the canonical panel bytes). A power
        of two far above EM convergence residuals and at least 16 times the typical motion of a
        compared contrast under 1e-6 relative GL noise: a search round's gain moves by about
        5e-5 nats at the measured 99.9th percentile. This keeps noise-scale contrasts inside
        the band while separating larger drop / merge / add differences (nats).
    """

    refits_per_kind: int = field(
        default=8,
        metadata={
            "units": "proposals/kind/round",
            "provenance": "Target 8 GEM refits per proposal kind, admitting whole score tie classes.",
        },
    )
    refits_per_kind_cap: int = field(
        default=16,
        metadata={
            "units": "proposals/kind/round",
            "provenance": "Whole-class admission cap 16; classes that would exceed it are omitted and flagged.",
        },
    )
    round_backstop_base: int = field(
        default=10,
        metadata={"units": "rounds", "provenance": "Base 10 in the reported round backstop 10+2K."},
    )
    round_backstop_per_founder: int = field(
        default=2,
        metadata={
            "units": "rounds/founder",
            "provenance": "Add 2 rounds per current founder to the reported search backstop.",
        },
    )
    update_backstop_base: int = field(
        default=20,
        metadata={"units": "updates", "provenance": "Base 20 in the fixed-K GEM backstop 20+4K+2L."},
    )
    update_backstop_per_founder: int = field(
        default=4,
        metadata={
            "units": "updates/founder",
            "provenance": "Add 4 updates per founder to the reported fixed-K GEM backstop.",
        },
    )
    update_backstop_per_site: int = field(
        default=2,
        metadata={
            "units": "updates/site",
            "provenance": "Add 2 updates per block site to the reported fixed-K GEM backstop.",
        },
    )
    tolerance: float = field(
        default=1e-6,
        metadata={"units": "nats", "provenance": "Fit acceptance, flip-gain and convergence tolerance 1e-6."},
    )
    decision_tolerance: float = field(
        default=2.0**-10,
        metadata={
            "units": "nats",
            ("provenance"): (
                "Decision quantum 2^-10 exceeds fit residuals and measured 5e-5-nat contrast motion under "
                "1e-6 GL noise."
            ),
        },
    )

    def round_backstop(self, k: int) -> int:
        return self.round_backstop_base + self.round_backstop_per_founder * k

    def update_backstop(self, k: int, length: int) -> int:
        return self.update_backstop_base + self.update_backstop_per_founder * k + self.update_backstop_per_site * length


@dataclass(frozen=True)
class CandidateBankParameters:
    """Candidate-bank search (bank.py): refits of the context proposals with discovery's
    fixed-K model, ranked by its binned-Viterbi BIC, K * complexity_scale * (L / 200) * N_active
    - 2 log L; K-scaled reported backstops with K the current panel's founder count: at most
    start_backstop(K) novel starts per call, score_backstop(K) admitted panels,
    round_backstop(K) rounds. The block's discovery tie quanta (core.ties
    BlockQuanta of the fixed-K workspace) order the BIC selection and the cavity-best endpoints
    (kind ``nll``) and stop the rounds (an improvement must exceed the ``improvement`` quantum).

    completion_variant_cap: a proposal panel whose partial rows have tied completions
        becomes one source per combination of them, up to this many; above it the panel's source
        keeps only its rows with a unique completion (counted as an overflow,
        "ls.bank.completion.overflow")."""

    round_backstop_base: int = field(
        default=8,
        metadata={"units": "rounds", "provenance": "Base 8 in the reported bank-round backstop 8+ceil(K/2)."},
    )
    completion_variant_cap: int = field(
        default=8,
        metadata={
            "units": "panels/source",
            ("provenance"): (
                "Keep whole tied-completion families through 8 variants; larger sources retain only "
                "uniquely completed rows."
            ),
        },
    )
    score_backstop_base: int = field(
        default=1024,
        metadata={"units": "panels", "provenance": "Base 1024 in the reported scored-panel backstop 1024+128K."},
    )
    score_backstop_per_founder: int = field(
        default=128,
        metadata={
            "units": "panels/founder",
            "provenance": "Add 128 admitted panels per current founder to the scored-panel backstop.",
        },
    )
    start_backstop_base: int = field(
        default=512,
        metadata={"units": "starts", "provenance": "Base 512 in the reported novel-start backstop 512+64K."},
    )
    start_backstop_per_founder: int = field(
        default=64,
        metadata={
            "units": "starts/founder",
            "provenance": "Add 64 novel starts per current founder to the start backstop.",
        },
    )
    complexity_scale: float = field(
        default=0.5,
        metadata={
            "units": "dimensionless",
            "provenance": "BIC founder-cost multiplier 0.5 times block sites/200 times active samples.",
        },
    )

    def score_backstop(self, k: int) -> int:
        return self.score_backstop_base + self.score_backstop_per_founder * k

    def start_backstop(self, k: int) -> int:
        return self.start_backstop_base + self.start_backstop_per_founder * k

    def round_backstop(self, k: int) -> int:
        return self.round_backstop_base + -(-k // 2)


@dataclass(frozen=True)
class PathReleaseParameters:
    """Release of a selected panel's alleles (release.py): the one-bit conditional allele
    probability reaches ``min_call_probability`` and the favourable samples (whose own evidence
    favours the allele by more than ``favourable_margin`` nats) include ``min_supporters``
    carriers by count (posterior carrier probability above 1/2) or by expected carrier mass.
    ``favourable_margin`` is the ``cell`` tie quantum of core.ties (2^-14 nats), separating
    symmetric-evidence cells with exactly zero gain from favourable evidence under GL noise.
    ``unknown_count_band``: a site's MAP number of unknown copies (0, 1, 2; it feeds assembly
    completion's all-or-nothing veto) must win by more than this relative posterior margin, else the
    smaller count. The ``probability`` tie quantum of core.ties is 2^-15: posterior masses of
    different integer counts can tie exactly."""

    min_call_probability: float = field(
        default=0.85,
        metadata={
            "units": "probability",
            "provenance": "Release an allele only when its one-bit conditional posterior reaches 0.85.",
        },
    )
    min_supporters: float = field(
        default=2.0,
        metadata={
            "units": "samples or expected carrier mass",
            ("provenance"): (
                "Require favourable carrier count or mass at least 2 unless an explicit context witness rescues it."
            ),
        },
    )
    favourable_margin: float = field(
        default=2.0**-14,
        metadata={
            "units": "nats",
            ("provenance"): (
                "Favourable evidence must exceed the cell quantum 2^-14, separating symmetric evidence from support."
            ),
        },
    )
    unknown_count_band: float = field(
        default=2.0**-15,
        metadata={
            "units": "relative probability",
            "provenance": "Prefer the smaller unknown-copy count inside a relative posterior band of 2^-15.",
        },
    )


@dataclass(frozen=True)
class ContextRefitParameters:
    """Refit of a context component's founders from the block's own evidence (context_refit.py).

    posterior_threshold: a refitted allele is released at this site-posterior probability.
    min_support: and with this carrier support (count or mass).
    emission_mix: mixing weight of a uniform genotype in the bin-level carrier emission.
    """

    posterior_threshold: float = field(
        default=0.99,
        metadata={
            "units": "probability",
            "provenance": "Release a context-refitted allele only when its site posterior reaches 0.99.",
        },
    )
    min_support: float = field(
        default=2.0,
        metadata={
            "units": "samples or expected carrier mass",
            "provenance": "Require context carrier count or mass support of 2.",
        },
    )
    emission_mix: float = field(
        default=0.01,
        metadata={
            "units": "probability",
            "provenance": "Uniform genotype mixture weight 0.01 in bin-level carrier emissions.",
        },
    )


@dataclass(frozen=True)
class SelectionParameters:
    """All local-selection settings (one record identity per round).

    context_levels: assembly levels of the context rounds, in order (rounds "l1", "l2").
    """

    model: PathModelParameters = field(
        default_factory=PathModelParameters,
        metadata={"units": "parameter record", "provenance": "Default normalised diploid copying-model settings."},
    )
    search: PathSearchParameters = field(
        default_factory=PathSearchParameters,
        metadata={"units": "parameter record", "provenance": "Default drop, merge and add search and GEM settings."},
    )
    bank: CandidateBankParameters = field(
        default_factory=CandidateBankParameters,
        metadata={"units": "parameter record", "provenance": "Default discovery-model candidate-bank settings."},
    )
    release: PathReleaseParameters = field(
        default_factory=PathReleaseParameters,
        metadata={
            "units": "parameter record",
            "provenance": "Default conditional-probability and favourable-carrier release gates.",
        },
    )
    context_refit: ContextRefitParameters = field(
        default_factory=ContextRefitParameters,
        metadata={
            "units": "parameter record",
            "provenance": "Default block-evidence refit settings for assembled context founders.",
        },
    )
    context_levels: tuple[int, ...] = field(
        default=(1, 2),
        metadata={
            "units": "levels/round",
            "provenance": "Context rounds assemble through levels 1 then 2, in that order.",
        },
    )


# Scientific values, resource layout and diagnostic windows are named here;
# PARAMETER_PROVENANCE describes their meaning, units and rationale.
# Running products of one-bit likelihood ratios move into the log total outside these
# dimensionless bounds; the same applies to a single ratio, preventing overflow and underflow.
CHAIN_LOW = 2.0**-300
CHAIN_HIGH = 2.0**300
FLIP_TIE = 2.0**-20 * NOISE_SCALE
EXACT_FIRST_UPDATE_MAX_K = 16
DESCRIPTION_TIE = 1e-9
DESCRIPTION_MEMO_ENTRIES = 4096
MERGE_EXACT_CODES_PER_ROUND = 128
MATCHING_SLACK = 1e-12
LOG_RATIO_NEAR = 0.5
LANE_WIDTH = 64
LANE_ALIGN = 8
LANE_MIN_WIDTH = 16
# Execution grain and live-share boundaries; provenance below gives units and rationale.
CHUNK_BLOCKS = 8
CHUNK_WAVES = 2
CHUNK_FLOOR = 8
CHUNK_TASKS_PER_CPU = 8
SHARE_REREAD_SECONDS = 0.1
# Small-cohort chunks amortise repeated evidence metadata and coordinator turnaround.
SMALL_COHORT_TASKS_PER_CPU = 2
SMALL_COHORT_CHUNK_BLOCKS = 4
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

# Candidate-bank neighbour rounds use short fits to admit a bounded family of starts.
RACE_MIN_STARTS = 24
RACE_PROBE_ITERATIONS = 3
RACE_KEEP = 16

PARAMETER_PROVENANCE = {
    "CHAIN_LOW": "Lower dimensionless bound 2^-300 of the exact likelihood-ratio running product.",
    "CHAIN_HIGH": "Upper dimensionless bound 2^300 of the exact likelihood-ratio running product.",
    "RACE_MIN_STARTS": (
        "In non-cavity neighbour rounds, race starts only above 24 novel starts; otherwise fit every start."
    ),
    "RACE_PROBE_ITERATIONS": "Probe each raced start for 3 fixed-K iterations before ranking its BIC endpoint.",
    "RACE_KEEP": "Retain the top 16 short-fit starts and every tie at the cut for full fitting.",
    "FLIP_TIE": "Relative flip-gain band scaled to 1e-6 relative genotype-likelihood noise.",
    "EXACT_FIRST_UPDATE_MAX_K": (
        "Founder-count threshold 16 for eagerly computing exact conditional statistics on the "
        "first frequency update; controls work only."
    ),
    "DESCRIPTION_TIE": "Relative shortest-description preference slack 1e-9, scaled by max(1, absolute code in nats).",
    "DESCRIPTION_MEMO_ENTRIES": (
        "Resource-only limit of 4096 panel descriptions per context; clearing recomputes exact values."
    ),
    "MERGE_EXACT_CODES_PER_ROUND": (
        "At most 128 exact context-code requests per round for uncertified merge starts, including K-2 panels "
        "and memo hits; certified codes are free. Unsolved starts rank optimistically. This declared screening "
        "approximation bounds cubic matching work at a constant allowance; smaller allowances lost more "
        "objective on constructed cases. It is not an error bound on selected panels."
    ),
    "MATCHING_SLACK": "Reduced-cost matching rounding allowance 1e-12 nats.",
    "LOG_RATIO_NEAR": "Dimensionless close-ratio boundary 0.5 for stable log-ratio evaluation.",
    "LANE_WIDTH": "Resource layout: 64 sample lanes per likelihood block.",
    "LANE_ALIGN": "Resource layout: multiples of 8 sample lanes retain vector work within each likelihood worker.",
    "LANE_MIN_WIDTH": "Resource layout: 16 sample lanes, the narrowest block a likelihood worker takes; bounds a small cohort's useful selection width.",
    "CHUNK_BLOCKS": (
        "Execution grain: average 8 blocks' cost per chunk, amortising coordinator turnaround; "
        "individual block results and publication remain independent."
    ),
    "CHUNK_WAVES": (
        "Execution grain: 2 waves per CPU in the guided remaining-work target, keeping tail chunks small."
    ),
    "CHUNK_FLOOR": (
        "Execution-only ratio 8: guided chunks have at least one eighth of the average target cost, "
        "bounding scheduling overhead as a round drains."
    ),
    "CHUNK_TASKS_PER_CPU": (
        "Execution-only minimum 8 chunks per declared CPU for larger cohorts; extra ready chunks "
        "limit straggler tails when refit block costs are poorly predicted."
    ),
    "SHARE_REREAD_SECONDS": (
        "Execution interval 0.1 seconds between live CPU-share reads; amortises boundary calls "
        "across frequent lane evaluations while allowing a block to grow as peers finish."
    ),
    "SMALL_COHORT_TASKS_PER_CPU": (
        "Resource-only minimum 2 chunks per declared CPU for cohorts fitting in one 64-lane block; "
        "amortises coordinator messages while retaining more ready tasks than CPU seats."
    ),
    "SMALL_COHORT_CHUNK_BLOCKS": (
        "Resource-only ceiling 4 blocks per small-cohort chunk bounds a cost-model underprediction's tail "
        "while sharing repeated evidence metadata across short blocks."
    ),
    "RELEASE_SAMPLE_BATCH": "Resource layout: process 128 samples per release batch.",
    "COMBINE_TILE": "Resource layout: 4096 elements per parallel reduction tile.",
    "CONSENSUS_TIE": "Absolute allele-probability band 2^-20 around one half for merge consensus.",
    "PROPOSAL_FREQUENCY_TIE": "Absolute frequency band 1e-12 for duplicate-start detection.",
    "DECISION_MOTION": "Diagnostic contrast motion 5e-5 nats at the measured 99.9th percentile under 1e-6 GL noise.",
    "CONSENSUS_CENSUS_WINDOW": "Diagnostic probability window 1e-3 around a merge-consensus boundary.",
    "KIND_ORDER": "Dimensionless same-panel representative priority: drop, merge, then add.",
    "BACKGROUND_COUNT_BAND": "Relative probability quantum 2^-15 for conservative background-copy counts.",
    "HALF_CENSUS_WINDOW": "Diagnostic carrier-probability window 1e-3 around one half.",
    "FAVOURABLE_CENSUS_WINDOW": "Diagnostic favourable-evidence window 1e-3 nats around zero.",
    "UNKNOWN_CENSUS_WINDOW": "Diagnostic relative posterior-gap window 1e-3 for unknown-copy counts.",
    "TASK_WORKSPACE_BASE_BYTES": "Resource-only base workspace allowance 2048 MiB per block task.",
    "SEARCH_GROWTH": "Resource-only founder growth allowance of 8 rows when estimating search storage.",
}
