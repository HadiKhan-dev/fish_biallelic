"""Name the scientific caps, tolerances and sweep controls carried from 3.5.

Owns parameter provenance, not storage policy or configuration loading.
Method: docs/methods/founder_refinement.md#fixed-k-search (RS01, RS05–RS11).
Accretion (RS06/RS07): frontier retries and work debits are scientific state because
binding stops a search. They are never replaced by a live-memory admission rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from haplotype_reconstruction.core.ties import TiePolicy, policy

__all__ = ["SweepParameters", "SweepControl"]

# RS09: strict gain floor, taken from the named acceptance policy's calibrated floor.
ACCEPTANCE_GAIN_FLOOR = policy("assembly.refine.accept").quantum()
# RS10: dE guard; retained scientific cutoff, not a score tie band (3.5 acceptance.py).
GUARD_TOLERANCE = 1e-6
# RS05: closed-message dominance and canonical-answer bands (3.5 proposals.py).
DOMINANCE_TOLERANCE = 1e-6
CANONICAL_TOL = 2.0**-14
# RS05 box arithmetic margin; 3.5 proposals.py and refinement-search.parameters.csv.
BOX_ARITHMETIC_MARGIN = 5e-10
# RS08 fixed acceptance round caps; 3.5 acceptance.py and refinement-search.parameters.csv.
CONTENT_FULL_ROUNDS = 4
EXCHANGE_ROUNDS = 4
PARAMETER_PROVENANCE = {
    "BOX_ARITHMETIC_MARGIN": "RS05; base40 proposals.py box certificate; refinement-search.parameters.csv",
    "CONTENT_FULL_ROUNDS": "RS08; base40 acceptance.py full re-ranking cap; refinement-search.parameters.csv",
    "EXCHANGE_ROUNDS": "RS08; base40 acceptance.py exchange cap; refinement-search.parameters.csv",
}
# RC05: uniform predictive contamination and per-observation log floor.
ROBUST_CONTAMINATION = 0.01
ROBUST_LOG_FLOOR = -2.0
# RS11: original exactness certificate; relative allowance plus allowance per accepted move, in nats.
EXACT_GAIN_RELATIVE = 1e-9
EXACT_GAIN_ABSOLUTE = 1e-6
# RS01: direct scoring fallback uses the same recurrence; this is a cost-only dimension cap.
PAIR_TABLE_BYTES = 1 << 31
# RS10: the affected-sample predicate of the shared-switch guard, in nats (3.5 acceptance.py:444).
AFFECTED_GAIN_TOLERANCE = 1e-9
# RS05: roundoff allowances of the canonical/dominance/dual programme, inherited unchanged.
PROGRAMME_RELATIVE_MARGIN = 1e-9
DOMINANCE_ABSOLUTE_MARGIN = 1e-9
DOMINANCE_RELATIVE_MARGIN = 1e-13
DUAL_RELATIVE_SLACK = 1e-12
CANONICAL_SKIP_ABSOLUTE_MARGIN = 1e-9
CANONICAL_SKIP_RELATIVE_MARGIN = 1e-12


@dataclass(frozen=True)
class SweepParameters:
    """Computational policies which can change the scientific search outcome.

    frontier_bytes: RS06's fixed cap, with the original retry schedule; bytes.
    window_work_coefficient: RS07 coefficient for N*W*b²*max(K,R)³; the next power
    of two above twice the largest normalized charge in the 98-component census.
    Inputs are immutable; invalid work coefficients raise SettingsError at entry.
    """

    frontier_bytes: int = field(
        default=128 << 30, metadata={"provenance": "RS06: original 3.5 128 GiB fixed frontier cap and retry schedule"}
    )
    window_work_coefficient: int = field(
        default=1 << 25,
        metadata={
            "provenance": "RS07: next power of two above twice the 98-component normalized-charge census maximum"
        },
    )


# RS09: 3.5 gave every fixed-K acceptance pass the convention reverse=(first == "rev") (founder_refinement/
# search.py:1306), whatever the run's tie reversal; a sweep's default convention is the forward-first one.
DEFAULT_SWEEP_CONVENTION_REVERSED = False


@dataclass(frozen=True)
class SweepControl:
    """Independent traversal direction, tie convention and starting message mode.

    lexicographic=True is the production first-sweep choice when coarse context
    exists. False exercises RS11's optimistic retry and is also the next-sweep
    choice after a guarded sweep with zero lex_items. No global flag is read.
    """

    direction: Literal["fwd", "rev"] = field(
        default="fwd", metadata={"provenance": "RS01: explicit original junction traversal direction"}
    )
    convention: TiePolicy = field(
        default_factory=lambda: policy("assembly.refine.accept", reverse=DEFAULT_SWEEP_CONVENTION_REVERSED),
        metadata={"provenance": "RS09: original assembly.refine.accept policy, independent of direction"},
    )
    lexicographic: bool = field(
        default=True,
        metadata={
            "provenance": "RS11: original coarse-context first-sweep guard mode; false exercises optimistic retry"
        },
    )


DEFAULT_GENOTYPE_ERROR = 0.01
DEFAULT_RESET_RATE = 1.0
MAP_PROBABILITY_FLOOR = 1e-15


@dataclass(frozen=True)
class RefinementParameters:
    """Founder refinement settings.

    proposal_max_bins: the binned window surrogate of a component has at most this many bins
        per component: bin size >= ceil(component sites / proposal_max_bins).
    proposal_min_sites_per_bin: smallest surrogate bin (sites). Levels 1 and 2 share the
        chromosome's resolution, max(this, ceil(chromosome sites / proposal_max_bins)), so the
        many small early components do not each get a chromosome's bin budget. The surrogate
        decides which moves are proposed; acceptance always scores every site exactly.
    sweep_limit: sweeps per fixed-K search. A search that reaches it is reported (and warned)
        as not converged; it never stops silently.
    frontier_bytes: the largest frontier one window programme may hold (bytes, counted as the
        frontier_limit formula: four copies of S x (Q + K + bins + 2) float64 per kept path); a
        programme that would exceed it keeps its incumbent (a reported fallback). A fixed constant,
        the same on every machine, so results never depend on free memory: 128 GiB is above base's
        own memory-derived cap on every reference node (at most a quarter of the available memory,
        about 120 GB), where base never fell back (largest frontier 76,776 paths, sim_s10001 chr3
        L1); K = 30 synthetic and AulStu peaks (257 and 357 paths) are four orders of magnitude
        below it. The memory of large programmes is admitted by the run's memory governor
        (proposals.PROGRAMME_BUDGETS).
    window_work_coefficient: positive integer C in the per-template charged-work bound
        C * N * W * b^2 * max(K, R)^3. N is the full sample count, W the template's unit
        count, b its largest actual bin count, K the current working founder count, and R
        its largest actual alphabet. One account covers every focal and retry. Exhaustion
        during the count pass discards the entire component refinement/count attempt, preserves its original rows
        and incoming families, and releases computationally unevaluated source domains.
        The value 2^25 is the next power of two strictly above twice the largest
        measured normalized charge in 98 realistic original component replays. The
        factor-two margin was declared before those inference-only measurements. This
        is not a biological threshold, FLOP count, or bound on component fits/windows.
    alternative_branches: complete same-level count continuations, including the
        ordinary path and all incoming panels. The shared 16-leaf budget retains a whole unresolved class
        when its complete expansion would exceed the budget. Branches reuse
        labelled fixed-K fits on identical inputs and run sequentially.
    genotype_error: the robust emission's uniform-genotype mixture (0.01: max(log(0.99 P + 0.01/3),
        -2)); recorded, not varied.
    reset_rate: ancestry-reset opportunities per Morgan per homologue (1.0): a physical edge of
        d Morgans has hazard reset_rate x d (objective.py; workspace.map_hazards).
    map_probability_floor: smallest per-edge reset probability before gap composition (1e-15), so
        sites at equal map positions still have a finite switch cost.
    polish: after the count pass, refit the decided panel (and its same-K equal-F alternatives)
        once more, by one forward-first search, with three extra move families at every ordinary fixed
        point: a beam escape (the
        window-beam proposer on the map-aware binned surrogate, beam.WindowBeam, then one exact
        acceptance pass over its bank), long one-founder moves with one private correction
        (search.LONG_WINDOW) and fine-window/suffix bridges. Count-pass fits use 2.6's exact window
        programmes only. K is never changed by the polish, and the count decisions are not re-evaluated on
        the polished panel. A polish that exhausts the window work account (window_work_coefficient) keeps
        the count pass's panel unpolished (warned; recorded as the refinement record's polish status).
    """

    proposal_max_bins: int = field(
        default=2000, metadata={"provenance": "base40 founder_refinement/config.py; AS7/AS8 proposal_max_bins"}
    )
    proposal_min_sites_per_bin: int = field(
        default=200, metadata={"provenance": "base40 founder_refinement/config.py; AS7/AS8 proposal_min_sites_per_bin"}
    )
    sweep_limit: int = field(
        default=100, metadata={"provenance": "base40 founder_refinement/config.py; AS7/AS8 sweep_limit"}
    )
    frontier_bytes: int = field(
        default=128 << 30, metadata={"provenance": "base40 founder_refinement/config.py; AS7/AS8 frontier_bytes"}
    )
    window_work_coefficient: int = field(
        default=1 << 25, metadata={"provenance": "base40 founder_refinement/config.py; AS7/AS8 window_work_coefficient"}
    )
    alternative_branches: int = field(
        default=16, metadata={"provenance": "base40 founder_refinement/config.py; AS7/AS8 alternative_branches"}
    )
    genotype_error: float = field(
        default=DEFAULT_GENOTYPE_ERROR,
        metadata={"provenance": "base40 founder_refinement/config.py; AS7/AS8 genotype_error"},
    )
    reset_rate: float = field(
        default=DEFAULT_RESET_RATE, metadata={"provenance": "base40 founder_refinement/config.py; AS7/AS8 reset_rate"}
    )
    map_probability_floor: float = field(
        default=MAP_PROBABILITY_FLOOR,
        metadata={"provenance": "base40 founder_refinement/config.py; AS7/AS8 map_probability_floor"},
    )
    polish: bool = field(default=True, metadata={"provenance": "base40 founder_refinement/config.py; AS7/AS8 polish"})


__all__ += ["RefinementParameters"]

# RS14-RS17 polish search domains; unchanged base40 search.py/beam.py.
LONG_WINDOW = 64
LONG_STRIDE = 16
LONG_CORRECTION = 8
LONG_DONORS = 2
LONG_BEAM_WIDTH = 16
POLISH_BEAM_WIDTH = 64
WEAK_JUNCTION_MARGIN = 20.0
# RC09/RC10 carried count policy and outward bound roundoff allowances.
COUNT_LIMIT_MULTIPLE = 1
COUNT_BOUND_RELATIVE = 1e-8
SAMPLE_BOUND_RELATIVE = 1e-9
# Cost-only storage/planning constants; never scientific admission results.
BRIDGE_BATCH_BYTES = 256 << 20
WINDOW_MEMO_BYTES = 256 << 20
WINDOW_MEMO_OVERHEAD = 1 << 20
WINDOW_MEMO_ENTRIES = 4096
FAMILY_BYTES = 1 << 30
COUNT_FANOUT_BYTES = 4 << 30
PARAMETER_PROVENANCE.update(
    {
        name: "base40 founder_refinement/search.py, beam.py or count.py; AS7 §7 / AS8 §7; same value and scope"
        for name in (
            "LONG_WINDOW",
            "LONG_STRIDE",
            "LONG_CORRECTION",
            "LONG_DONORS",
            "LONG_BEAM_WIDTH",
            "POLISH_BEAM_WIDTH",
            "WEAK_JUNCTION_MARGIN",
            "COUNT_LIMIT_MULTIPLE",
            "COUNT_BOUND_RELATIVE",
            "SAMPLE_BOUND_RELATIVE",
            "BRIDGE_BATCH_BYTES",
            "WINDOW_MEMO_BYTES",
            "WINDOW_MEMO_OVERHEAD",
            "WINDOW_MEMO_ENTRIES",
            "FAMILY_BYTES",
            "COUNT_FANOUT_BYTES",
        )
    }
)
