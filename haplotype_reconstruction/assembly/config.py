"""Settings of the assembly stage (base production values; internal model constants).

None of these is a user TOML key: base never let a run override them (docs/configuration.md lists
them). The per-base-pair recombination rate is not here: assembly reads it from the contig's
genetic map (its fallback rate when no map is given).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .completion.config import CompletionConfig
from .count_rule.config import CountRuleConfig
from .founder_refinement.config import FounderRefinementConfig


@dataclass(frozen=True)
class LinkerConfig:
    """The distance-aware sum-product linker that fits dense haploid transitions between blocks.

    max_sites: linking proxies keep at most this many sites per input component (a stride over
        its usable sites); emissions marginalise unknown alleles at partial sites.
    max_iterations: maximum damped updates per gap fit (convergence may stop earlier).
    robustness: genotype emission mixture P = (1 - r) P(g | pair) + r / 3 (no log floor).
    error_fraction, error_tract_bp: the within-block error-tract model of the micro HMM (a
        fraction of copying in error tracts of this mean length).
    min_log_probability, pseudo_count, uniform_mix: M-step regularisation (log floor of a
        transition before renormalising, pseudocount added to every edge count, robust uniform
        mixture of each fitted row).
    min_change: stop when max(abs(M(T) - T)) is at most this value at the returned transitions,
        where M is the undamped regularised forward M-step. A capped fit may remain unconverged.
    learning_rate_decay, minimum_learning_rate: damping of the M-step towards the previous
        transitions, rate = max(decay^iteration, minimum).
    """

    max_sites: int = 2000
    max_iterations: int = 50
    robustness: float = 0.01
    error_fraction: float = 0.01
    error_tract_bp: float = 20000.0
    min_log_probability: float = -10.0
    pseudo_count: float = 0.1
    uniform_mix: float = 0.01
    min_change: float = 0.001
    learning_rate_decay: float = 0.9
    minimum_learning_rate: float = 0.1


@dataclass(frozen=True)
class JoinConfig:
    """The endpoint beam and the certified panel search of one joined batch.

    paths_per_endpoint: beam quota of candidate chromosome paths per endpoint haplotype (16); the
        pool-saturation ladder doubles it up to quota 256, and up to 1024 once a move was
        accepted at the 256 rung.
    full_scores_per_kind: exact scores per proposal kind per tier-1 sweep.
    max_bins: the join's binned emissions have at most this many bins per batch.
    tensor_budget_mb: memory of one chunk of painting back pointers (MiB).
    """

    paths_per_endpoint: int = 16
    full_scores_per_kind: int = 16
    max_bins: int = 2000
    tensor_budget_mb: int = 256


@dataclass(frozen=True)
class AssemblyConfig:
    """Assembly of local panels into chromosome-scale founder components.

    batch_size: input components per joined batch at every level (base L1 and L2-L4 both 10).
    minimum_levels: hierarchy levels that run (while they reduce the component count) before the
        loop may stop because only phase breaks separate the remaining components; base ran
        exactly four levels, so at today's sizes (<= 10^4 blocks per break-free segment) the
        level schedule is base's. Beyond it a level runs only while a boundary that is not a
        phase break remains (base stopped at level 4 regardless: B4).
    min_boundary_informative_samples: a boundary between adjacent components becomes a phase
        break (never joined) when fewer samples are informative on both sides.
    gap_horizon_scale: inverse-Morgan scale of the linker and beam's maximum transition
        gap at levels >= 2: max(1, 1 + floor(recombination_tolerance /
        (mean component span in Morgans x gap_horizon_scale))). Level 1 uses every gap.
        This search-horizon control is independent of local selection's copying intensity.
    recombination_tolerance: dimensionless span budget in the maximum-gap rule; divided
        by the mean component span in Morgans times gap_horizon_scale.
    cost_scale: scale of the per-founder cost c = cost_scale x (mean sites per block / 200) x S x B
        (count_rule.founder_cost) of the join and founder refinement.
    """

    batch_size: int = 10
    minimum_levels: int = 4
    min_boundary_informative_samples: int = 4
    gap_horizon_scale: int = 3
    recombination_tolerance: float = 0.5
    cost_scale: float = 0.5
    linker: LinkerConfig = field(default_factory=LinkerConfig)
    join: JoinConfig = field(default_factory=JoinConfig)
    completion: CompletionConfig = field(default_factory=CompletionConfig)
    count_rule: CountRuleConfig = field(default_factory=CountRuleConfig)
    refinement: FounderRefinementConfig = field(default_factory=FounderRefinementConfig)
