"""Settings of the final founder-path refinement (base production values).

These are internal model constants of the assembly stage (documented in docs/configuration.md),
not user TOML keys: base never let a run override them. The count rule's constants (bootstrap
resamples and level, ranked refits) belong to the shared count rule (assembly/count_rule).
"""
from __future__ import annotations

from dataclasses import dataclass

# Release 2.8 switch model: map-derived per-homologue reset probabilities.
DEFAULT_GENOTYPE_ERROR = 0.01
DEFAULT_RESET_RATE = 1.0
MAP_PROBABILITY_FLOOR = 1e-15


@dataclass(frozen=True)
class FounderRefinementConfig:
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

    proposal_max_bins: int = 2000
    proposal_min_sites_per_bin: int = 200
    sweep_limit: int = 100
    frontier_bytes: int = 128 << 30
    window_work_coefficient: int = 1 << 25
    alternative_branches: int = 16
    genotype_error: float = DEFAULT_GENOTYPE_ERROR
    reset_rate: float = DEFAULT_RESET_RATE
    map_probability_floor: float = MAP_PROBABILITY_FLOOR
    polish: bool = True
