"""Scientific tolerances, search domains and fixed refinement controls.

Owns parameter meaning and provenance, not configuration loading.
Method: docs/methods/founder_refinement.md#fixed-k-search (RS01, RS05–RS11).
The frontier switch and work debits are scientific state. Live-memory admission
may delay a solve but never changes its route, limits or search order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from haplotype_reconstruction.core.ties import policy

__all__ = ["RefinementParameters"]

# Strict acceptance gain floor (nats), supplied by the calibrated acceptance policy.
ACCEPTANCE_GAIN_FLOOR = policy("assembly.refine.accept").quantum()
# Minimum genotype-fit loss tested by the coarse-move guard (nats), not a score tie band.
GUARD_TOLERANCE = 1e-6
# Closed-message dominance tolerance (nats): a dropped path cannot beat a retained path
# by more than this for any completion. Restricted programmes choose the canonical answer.
DOMINANCE_TOLERANCE = 1e-6
# Canonical-answer absolute tolerance (nats): 16 times the measured ~1e-6 motion of a
# window contrast under 1e-6 relative genotype-likelihood noise.
CANONICAL_TOL = 2.0**-14
# Relative arithmetic allowance between the box bound and programme sums (dimensionless).
BOX_ARITHMETIC_MARGIN = 5e-10
# Per-junction caps on full content rerankings and suffix exchanges (rounds).
CONTENT_FULL_ROUNDS = 4
EXCHANGE_ROUNDS = 4
# Exact-gain certificate: relative score allowance plus nats per accepted move.
EXACT_GAIN_RELATIVE = 1e-9
EXACT_GAIN_ABSOLUTE = 1e-6
# Pair-code table byte cap; above it the same recurrence scores directly.
PAIR_TABLE_BYTES = 1 << 31
# Minimum per-sample gain for inclusion in the shared-switch guard (nats).
AFFECTED_GAIN_TOLERANCE = 1e-9
# Roundoff allowances for programme, dominance and canonical-skip comparisons.
# Absolute margins are nats; relative margins and slack coefficients are dimensionless.
PROGRAMME_RELATIVE_MARGIN = 1e-9
DOMINANCE_ABSOLUTE_MARGIN = 1e-9
DOMINANCE_RELATIVE_MARGIN = 1e-13
CANONICAL_SKIP_ABSOLUTE_MARGIN = 1e-9
CANONICAL_SKIP_RELATIVE_MARGIN = 1e-12

# Fixed frontier and work limits for a complete component fit.
DEFAULT_WINDOW_WORK_COEFFICIENT = 1 << 25

# Uniform diplotype state-change cost of the binned surrogate (nats); exact
# acceptance uses map-derived per-homologue costs instead.
SWITCH_PENALTY = 20.0
# Exact-programme frontier (kept paths) above which a template takes the seeded
# search. In component replays of two known-truth simulations (6 and 30
# founders) 1.05% and 3.4% of templates exceeded it, and every released panel
# row equalled the exact programme's; the templates of a real chromosome with
# 27 to 30 local founders all exceeded it.
HYBRID_FRONTIER = 64
# Ascent starts (paths), whole wishes and restrictions: six plus six match exact
# gain on all six captured K13-K28 templates; three plus three reach 95.5% at K28.
SEED_STARTS = 6
# Restriction starts (paths): six plus six match exact gain on all six captured
# K13-K28 templates; three plus three reach 95.5% at K28.
INTERVAL_SEED_STARTS = 6
# Relative arithmetic allowance (dimensionless) between a sample's relaxed bound
# and background recurrence; suppresses last-bit relevance differences.
RELEVANCE_MARGIN = 1e-9
# Maximum improving single-unit ascent sweeps per start (iterations). Bounds work
# even when a discrete row path has a long sequence of small improvements.
LOCAL_ITERATIONS = 32
# Fine templates cover prepared blocks; coarse templates cover L1 spans.
# The final window is also included when its end falls between regular strides.
FINE_WINDOW = 8
FINE_STRIDE = 4
COARSE_WINDOW = 4
COARSE_STRIDE = 2


# Fixed robust-emission mixture weight (probability), recorded for model validation.
DEFAULT_GENOTYPE_ERROR = 0.01
# Ancestry-reset opportunities per Morgan per homologue.
DEFAULT_RESET_RATE = 1.0
# Per-edge reset probability floor, keeping equal-position switch costs finite.
MAP_PROBABILITY_FLOOR = 1e-15


@dataclass(frozen=True)
class RefinementParameters:
    """Founder refinement settings.

    proposal_max_bins: surrogate bin-count target per component; bin size is at
        least ceil(component sites / proposal_max_bins).
    proposal_min_sites_per_bin: smallest surrogate bin in sites. Levels 1 and 2
        share chromosome resolution, max(this, ceil(chromosome sites / proposal_max_bins)),
        so small early components do not each receive a chromosome's bin budget.
        The surrogate proposes moves; acceptance scores every site exactly.
    sweep_limit: sweeps per fixed-K search; reaching it records nonconvergence.
    window_work_coefficient: positive dimensionless C in the per-template charged-work
        bound C * N * W * b^2 * max(K, R)^3. N is full sample count, W template unit
        count, b largest actual bin count, K current founder count, and R largest
        actual alphabet. One account covers every focal, its exact attempt and its seeded search. Exhaustion in
        the count pass discards the entire component refinement/count attempt,
        preserves input rows and families, and releases unevaluated source domains.
        The value 2^25 is the next power of two above twice the largest measured
        normalized charge in a 98-component census. It is neither a biological
        threshold nor a FLOP count or a bound on component fits/windows.
    alternative_branches: complete same-level count continuations, including the
        ordinary path and incoming panels. The shared 16-leaf budget retains a whole
        unresolved class when its full expansion would exceed the budget. Branches
        reuse labelled fixed-K fits on identical inputs and run sequentially.
    genotype_error: robust emission's uniform-genotype mixture probability (0.01):
        max(log(0.99 P + 0.01/3), -2). Recorded, not varied.
    reset_rate: ancestry-reset opportunities per Morgan per homologue (1.0); an
        edge of d Morgans has hazard reset_rate x d (component_workspace.map_hazards).
    map_probability_floor: smallest per-edge reset probability before gap
        composition, keeping switch costs finite at equal map positions.
    polish: after count selection, refit the decided panel and its same-K equal-F
        alternatives by one forward-first search. At every ordinary fixed point,
        offer beam escapes (beam.WindowBeam), long one-founder moves with one private
        correction (LONG_WINDOW), and fine-window/suffix bridges. Count-pass fits use
        the ordinary window programmes. Polish never changes K or re-evaluates count
        decisions. Work exhaustion keeps the count panel unpolished and is recorded.
    """

    proposal_max_bins: int = field(
        default=2000,
        metadata={"provenance": "Surrogate bin-count target per component (bins); sets proposal resolution."},
    )
    proposal_min_sites_per_bin: int = field(
        default=200,
        metadata={"provenance": "Minimum surrogate bin width (sites); bounds proposal resolution."},
    )
    sweep_limit: int = field(
        default=100,
        metadata={"provenance": "Maximum sweeps per fixed-K search (sweeps); reaching it records nonconvergence."},
    )
    window_work_coefficient: int = field(
        default=DEFAULT_WINDOW_WORK_COEFFICIENT,
        metadata={
            "provenance": (
                "Dimensionless work coefficient; next power of two above twice the 98-component "
                "normalized-charge maximum."
            )
        },
    )
    alternative_branches: int = field(
        default=16,
        metadata={"provenance": ("Count-continuation leaf budget (histories); overflow retains the unresolved class.")},
    )
    genotype_error: float = field(
        default=DEFAULT_GENOTYPE_ERROR,
        metadata={
            "provenance": (
                "Uniform-genotype mixture probability (0.01) in the fixed robust emission; recorded for validation."
            )
        },
    )
    reset_rate: float = field(
        default=DEFAULT_RESET_RATE,
        metadata={
            "provenance": (
                "Ancestry-reset opportunities per Morgan per homologue; multiplies each physical-edge map distance."
            )
        },
    )
    map_probability_floor: float = field(
        default=MAP_PROBABILITY_FLOOR,
        metadata={"provenance": "Per-edge reset probability floor; keeps equal-map-position switch costs finite."},
    )
    polish: bool = field(
        default=True,
        metadata={
            "provenance": ("Boolean post-count fixed-K polish switch; offers extra move families without changing K.")
        },
    )


# Long-repair window/stride/correction widths in prepared blocks; donor/beam widths in paths.
LONG_WINDOW = 64
LONG_STRIDE = 16
LONG_CORRECTION = 8
LONG_DONORS = 2
LONG_BEAM_WIDTH = 16
POLISH_BEAM_WIDTH = 64
# Outward deletion-bound allowance relative to the total score scale.
COUNT_BOUND_RELATIVE = 1e-8
# Cost-only storage/planning constants: bytes unless named ENTRIES.
BRIDGE_BATCH_BYTES = 256 << 20
WINDOW_STORE_MIN_BYTES = 256 << 20
WINDOW_STORE_MAX_BYTES = 8 << 30
WINDOW_STORE_TEMPLATE_BYTES = 8 << 10
WINDOW_STORE_ENTRY_OVERHEAD = 512
WINDOW_STORE_TRACK_BYTES = 64
WINDOW_STORE_TRACKED_FITS = 128
POLISH_SHARE_MIN_BYTES = 64 << 20
POLISH_SHARE_MAX_BYTES = 2 << 30
POLISH_SHARE_TEMPLATE_BYTES = 2 << 10
# Standing-offer gain tables of a search (one per pass orientation): slots per window template and the smallest
# table (rounded up to a power of two). In replays of captured refinement components the tables evicted under 0.1 %
# of their inserts at K 15-30 on simulated crosses and 11-16 % on a hybrid cross at K 27-28. Runtime only: a missing
# entry recomputes the same gains.
GAIN_CACHE_TEMPLATE_SLOTS = 32
GAIN_CACHE_MIN_SLOTS = 1024
# Cumulative exact score miss admissions per component, shared by continuation branches.
# Both limits bound storage only; saturation recomputes the same score. The byte
# allowance keeps long panel keys from dominating small score vectors.
SCORE_LEDGER_ENTRIES = 4096
# Retained miss allowance (bytes), reserved once per component. A fixed 64 MiB
# bounds multiplied component residence; large keys reduce the admitted count.
SCORE_LEDGER_BYTES = 64 << 20
# Conservative bytes per miss for dictionary storage, admission counters
# and allocator slack; tuple/int keys and ndarray storage are charged separately.
SCORE_LEDGER_ENTRY_OVERHEAD = 512
FAMILY_BYTES = 1 << 30
COUNT_FANOUT_BYTES = 4 << 30

PARAMETER_PROVENANCE = {
    "HYBRID_FRONTIER": (
        "Kept paths. In component replays of two known-truth simulations (6 and 30 founders) 1.05% and "
        "3.4% of templates exceeded it, and every released panel row equalled the exact programme's; the "
        "templates of a real chromosome with 27 to 30 local founders all exceeded it."
    ),
    "SEED_STARTS": (
        "Whole-wish starts (paths); six whole plus six restriction starts match exact gain on all six "
        "captured K13-K28 templates; three plus three reach 95.5% at K28."
    ),
    "INTERVAL_SEED_STARTS": (
        "Restriction starts (paths); six whole plus six restriction starts match exact gain on all six "
        "captured K13-K28 templates; three plus three reach 95.5% at K28."
    ),
    "RELEVANCE_MARGIN": "Relative arithmetic allowance (dimensionless) for background relevance comparisons.",
    "LOCAL_ITERATIONS": "Single-unit ascent limit per start (iterations); bounds improvement-chain work.",
    "BOX_ARITHMETIC_MARGIN": (
        "Dimensionless relative margin between box-bound and programme sums; certifies only clear bounds."
    ),
    "CONTENT_FULL_ROUNDS": (
        "Full content reranking cap per junction (rounds); remaining candidates follow the last full ranking."
    ),
    "EXCHANGE_ROUNDS": "Suffix-exchange acceptance cap per junction (rounds).",
    "DEFAULT_WINDOW_WORK_COEFFICIENT": (
        "Dimensionless per-template work coefficient, above twice the measured normalized-charge maximum."
    ),
    "SWITCH_PENALTY": "Uniform diplotype state-change penalty in the binned proposal surrogate (nats).",
    "FINE_WINDOW": "Fine proposal window width (prepared blocks).",
    "FINE_STRIDE": "Fine proposal window stride (prepared blocks).",
    "COARSE_WINDOW": "Coarse proposal window width (L1 spans).",
    "COARSE_STRIDE": "Coarse proposal window stride (L1 spans).",
    "LONG_WINDOW": "Long-repair proposal window width (prepared blocks).",
    "LONG_STRIDE": "Long-repair window stride (prepared blocks).",
    "LONG_CORRECTION": "Private correction width for a long offer (prepared blocks).",
    "LONG_DONORS": "Screened correction donors per long offer (founder paths).",
    "LONG_BEAM_WIDTH": "Retained partial paths in the long-window beam (paths).",
    "POLISH_BEAM_WIDTH": "Retained partial paths in the polish escape beam (paths).",
    "COUNT_BOUND_RELATIVE": "Dimensionless relative outward allowance on aggregate count bounds.",
    "BRIDGE_BATCH_BYTES": "Message-byte budget per batch of bridge precursor scoring (bytes).",
    "WINDOW_STORE_MIN_BYTES": "Smallest window-result store budget of a component (bytes).",
    "WINDOW_STORE_MAX_BYTES": "Largest window-result store budget of a component (bytes).",
    "WINDOW_STORE_TEMPLATE_BYTES": (
        "Window-result store budget per window template and squared founder count (bytes); replays of captured "
        "components with incoming-root, continuation and polish alternatives stored 3.5-9.4 KiB per template and K^2 "
        "for every distinct solve of the whole component."
    ),
    "WINDOW_STORE_ENTRY_OVERHEAD": "Bookkeeping charge per stored window result beyond its payload (bytes).",
    "WINDOW_STORE_TRACK_BYTES": "Bookkeeping charge per window-store key tracked for a count fit (bytes).",
    "WINDOW_STORE_TRACKED_FITS": (
        "Most recent count fits whose window-store keys are tracked for pinning (fits); covers the refits of a "
        "few concurrent K = 30 deletion rounds."
    ),
    "POLISH_SHARE_MIN_BYTES": "Smallest budget of the polish answers shared by a component's polish fits (bytes).",
    "POLISH_SHARE_MAX_BYTES": "Largest budget of the polish answers shared by a component's polish fits (bytes).",
    "POLISH_SHARE_TEMPLATE_BYTES": (
        "Shared polish-answer budget per window template and squared founder count (bytes)."
    ),
    "SCORE_LEDGER_ENTRIES": (
        "Cumulative per-sample/shadow score miss admissions per component (entries); fitted publications "
        "remain unconditional."
    ),
    "SCORE_LEDGER_BYTES": (
        "Retained miss-score allowance per component (bytes), including vectors, panel keys and metadata; "
        "reserved once across branches."
    ),
    "SCORE_LEDGER_ENTRY_OVERHEAD": (
        "Conservative miss-entry bookkeeping allowance (bytes) for dictionary storage, admission counters "
        "and allocator slack."
    ),
    "FAMILY_BYTES": "Shadow-family scoring array budget (bytes); larger families are scored member by member.",
    "COUNT_FANOUT_BYTES": (
        "Concurrent count-scoring and shadow-decision transient budget (bytes); larger items run alone."
    ),
}
