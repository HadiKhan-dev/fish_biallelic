"""Settings of the final founder-path refinement (base production values).

These are internal model constants of the assembly stage (documented in docs/configuration.md),
not user TOML keys: base never let a run override them. The count rule's constants (bootstrap
resamples and level, ranked refits) belong to the shared count rule (assembly/count_rule).
"""
from __future__ import annotations

from dataclasses import dataclass


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
    """

    proposal_max_bins: int = 2000
    proposal_min_sites_per_bin: int = 200
    sweep_limit: int = 100
    frontier_bytes: int = 128 << 30
