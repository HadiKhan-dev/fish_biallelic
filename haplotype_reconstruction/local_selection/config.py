"""Settings of local selection (internal model constants; base production values).

Local selection refits every discovery block under a normalised diploid copying model (K
founder rows plus a fixed-mass unknown state whose copies draw Bernoulli(1/2) alleles) and
selects the panel by a drop / merge / add search, first from the raw discovery panel, then after
each context round (assembly through L1, then L1+L2) with the context's refitted founders as
candidates. None of these fields is user-visible (base never overrode one).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PathModelConfig:
    """The copying model of a block.

    unknown_mass: fixed prior mass of the unknown state (base ``background_mass``; also the
        background-state mass of the context refit's bin-level HMM).

    The copying hazard r = 1 - exp(-g x Morgans) uses g = ``AssemblyConfig.n_generations`` (one
    value shared with assembly's linker gap, as base) and the contig's genetic map.
    """

    unknown_mass: float = 0.01


@dataclass(frozen=True)
class PathSearchConfig:
    """Drop / merge / add search with GEM refits (search.py, fit.py).

    refits_per_kind: top proposals per kind (drop, merge, add) refitted each round.
    round_backstop_base/_per_founder: reported round backstop, K of the current panel.
    update_backstop_base/_per_founder/_per_site: reported backstop of fixed-K GEM updates,
        base + per_founder K + per_site L.
    refits_per_kind_cap: release 2.1: the proposals tied with the last of the top refits_per_kind
        (its tie class of the screened score at decision_tolerance) join the refits, up to this many
        per kind; a cut still inside the class goes by panel bytes, then generation order.
    tolerance: the GEM fit's own tolerance (frequency-step acceptance, flip gains, convergence; nats).
    decision_tolerance: tie quantum of every decision between EM-fitted or screened objectives
        (release 2.1): the proposal cut per kind, the round winner and the stop of the climb, the
        start-or-incumbent choice and the keep-incumbent rule between rounds; values chained within
        it are tied and decided by content (fewer founders, then the canonical panel bytes). A power
        of two placed by the valley census far above EM convergence residuals and the motion of 1e-9
        relative GL noise, far below genuine drop / merge / add differences (nats).
    """

    refits_per_kind: int = 8
    refits_per_kind_cap: int = 12
    round_backstop_base: int = 10
    round_backstop_per_founder: int = 2
    update_backstop_base: int = 20
    update_backstop_per_founder: int = 4
    update_backstop_per_site: int = 2
    tolerance: float = 1e-6
    decision_tolerance: float = 2.0 ** -10

    def round_backstop(self, k: int) -> int:
        return self.round_backstop_base + self.round_backstop_per_founder * k

    def update_backstop(self, k: int, length: int) -> int:
        return (self.update_backstop_base + self.update_backstop_per_founder * k
                + self.update_backstop_per_site * length)


@dataclass(frozen=True)
class CandidateBankConfig:
    """Candidate-bank search (bank.py): refits of the context proposals with discovery's
    fixed-K model, ranked by its binned-Viterbi BIC, K * complexity_scale * (L / 200) * N_active
    - 2 log L; K-scaled reported backstops with K the current panel's founder count: at most
    start_backstop(K) novel starts per call, score_backstop(K) admitted panels,
    round_backstop(K) rounds. Release 2.1: the block's discovery tie quanta (maths.ties
    BlockQuanta of the fixed-K workspace) order the BIC selection and the cavity-best endpoints
    (kind ``nll``) and stop the rounds (an improvement must exceed the ``improvement`` quantum)."""

    round_backstop_base: int = 8
    score_backstop_base: int = 1024
    score_backstop_per_founder: int = 128
    start_backstop_base: int = 512
    start_backstop_per_founder: int = 64
    complexity_scale: float = 0.5

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
    Release 2.1: ``favourable_margin`` is the ``cell`` tie quantum of maths.ties (2^-23 nats; the
    1e-9 of 2.0 sat at the motion of 1e-9 GL noise, so a symmetric-evidence cell, whose gain is
    exactly 0, straddled it). ``unknown_count_band``: a site's MAP number of unknown copies (0, 1,
    2; it feeds assembly completion's all-or-nothing veto) must win by more than this relative
    posterior margin, else the smaller count (the ``probability`` tie quantum of maths.ties, 2^-20;
    design A-7: an integer count whose posterior masses can tie exactly)."""

    min_call_probability: float = 0.85
    min_supporters: float = 2.0
    favourable_margin: float = 2.0 ** -23
    unknown_count_band: float = 2.0 ** -20


@dataclass(frozen=True)
class ContextRefitConfig:
    """Refit of a context component's founders from the block's own evidence (context_refit.py).

    posterior_threshold: a refitted allele is released at this site-posterior probability.
    min_support: and with this carrier support (count or mass).
    emission_mix: mixing weight of a uniform genotype in the bin-level carrier emission.
    """

    posterior_threshold: float = 0.99
    min_support: float = 2.0
    emission_mix: float = 0.01


@dataclass(frozen=True)
class LocalSelectionConfig:
    """All local-selection settings (one record identity per round).

    context_levels: assembly levels of the context rounds, in order (rounds "l1", "l2").
    """

    model: PathModelConfig = field(default_factory=PathModelConfig)
    search: PathSearchConfig = field(default_factory=PathSearchConfig)
    bank: CandidateBankConfig = field(default_factory=CandidateBankConfig)
    release: PathReleaseConfig = field(default_factory=PathReleaseConfig)
    context_refit: ContextRefitConfig = field(default_factory=ContextRefitConfig)
    context_levels: tuple[int, ...] = (1, 2)
