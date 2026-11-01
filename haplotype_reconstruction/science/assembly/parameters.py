"""Name the scientific hierarchy, linker and join parameters with provenance.

Owns the AS6 hierarchy schedule and the AS3/AS4 shared parameter records.
No settings files, environment flags or computational routing are read here.
Method: docs/methods/assembly.md; P0 assembly-core and assembly-joins inventories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .completion.parameters import CompletionParameters
from .count_rule.parameters import CountRuleParameters
from .refinement.parameters import RefinementParameters


def _count_defaults() -> CountRuleParameters:
    from .count_rule.parameters import CountRuleParameters

    return CountRuleParameters()


def _refinement_defaults() -> RefinementParameters:
    from .refinement.parameters import RefinementParameters

    return RefinementParameters()


@dataclass(frozen=True)
class LinkerParameters:
    """The distance-aware sum-product linker that fits dense haploid transitions between blocks.

    max_sites: linking proxies keep at most this many sites per input component (a stride over
        its usable sites); emissions marginalise unknown alleles at partial sites.
    max_iterations: maximum updates per gap fit (convergence may stop earlier).
    robustness: genotype emission mixture P = (1 - r) P(g | pair) + r / 3 (no log floor).
    error_fraction, error_tract_bp: the within-block error-tract model of the micro HMM (a
        fraction of copying in error tracts of this mean length).
    min_log_probability, pseudo_count, uniform_mix: M-step regularisation (log floor of a
        transition before renormalising, pseudocount added to every edge count, robust uniform
        mixture of each fitted row).
    min_change: stop when max(abs(M(T) - T)) is at most this value at the returned transitions,
        where M is the undamped regularised forward M-step. A capped fit may remain unconverged.
    learning_rate_decay, minimum_learning_rate: damping of the M-step towards the previous
        transitions, rate = max(decay^iteration, minimum). The default decay of 1 takes full
        M-steps; smaller values retain damped iteration for numerical comparisons.
    """

    max_sites: int = field(
        default=2000, metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py LinkerConfig.max_sites"}
    )
    max_iterations: int = field(
        default=50, metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py LinkerConfig.max_iterations"}
    )
    robustness: float = field(
        default=0.01, metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py LinkerConfig.robustness"}
    )
    error_fraction: float = field(
        default=0.01, metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py LinkerConfig.error_fraction"}
    )
    error_tract_bp: float = field(
        default=20000.0,
        metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py LinkerConfig.error_tract_bp"},
    )
    min_log_probability: float = field(
        default=-10.0,
        metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py LinkerConfig.min_log_probability"},
    )
    pseudo_count: float = field(
        default=0.1, metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py LinkerConfig.pseudo_count"}
    )
    uniform_mix: float = field(
        default=0.01, metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py LinkerConfig.uniform_mix"}
    )
    min_change: float = field(
        default=0.001, metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py LinkerConfig.min_change"}
    )
    learning_rate_decay: float = field(
        default=1.0,
        metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py LinkerConfig.learning_rate_decay"},
    )
    minimum_learning_rate: float = field(
        default=0.1,
        metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py LinkerConfig.minimum_learning_rate"},
    )


@dataclass(frozen=True)
class JoinParameters:
    """The endpoint beam and the certified panel search of one joined batch.

    paths_per_endpoint: beam quota of candidate chromosome paths per endpoint haplotype (16); the
        pool-saturation ladder doubles it up to quota 256, and up to 1024 once a move was
        accepted at the 256 rung.
    full_scores_per_kind: exact scores per proposal kind per tier-1 sweep.
    max_bins: the join's binned emissions have at most this many bins per batch.
    tensor_budget_mb: memory of one chunk of painting back pointers (MiB).
    """

    paths_per_endpoint: int = field(
        default=16, metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py JoinConfig.paths_per_endpoint"}
    )
    full_scores_per_kind: int = field(
        default=16, metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py JoinConfig.full_scores_per_kind"}
    )
    max_bins: int = field(
        default=2000, metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py JoinConfig.max_bins"}
    )
    tensor_budget_mb: int = field(
        default=256, metadata={"provenance": "P0 assembly-joins; 3.5 assembly/config.py JoinConfig.tensor_budget_mb"}
    )


@dataclass(frozen=True)
class AssemblyParameters:
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

    batch_size: int = field(
        default=10, metadata={"provenance": "P0 assembly-core; 3.5 assembly/config.py AssemblyConfig.batch_size"}
    )
    minimum_levels: int = field(
        default=4, metadata={"provenance": "P0 assembly-core; 3.5 assembly/config.py AssemblyConfig.minimum_levels"}
    )
    min_boundary_informative_samples: int = field(
        default=4,
        metadata={
            "provenance": "P0 assembly-core; 3.5 assembly/config.py AssemblyConfig.min_boundary_informative_samples"
        },
    )
    gap_horizon_scale: int = field(
        default=3, metadata={"provenance": "P0 assembly-core; 3.5 assembly/config.py AssemblyConfig.gap_horizon_scale"}
    )
    recombination_tolerance: float = field(
        default=0.5,
        metadata={"provenance": "P0 assembly-core; 3.5 assembly/config.py AssemblyConfig.recombination_tolerance"},
    )
    cost_scale: float = field(
        default=0.5, metadata={"provenance": "P0 assembly-core; 3.5 assembly/config.py AssemblyConfig.cost_scale"}
    )
    linker: LinkerParameters = field(
        default_factory=LinkerParameters,
        metadata={"provenance": "P0 assembly-core; 3.5 assembly/config.py AssemblyConfig.linker"},
    )
    join: JoinParameters = field(
        default_factory=JoinParameters,
        metadata={"provenance": "P0 assembly-core; 3.5 assembly/config.py AssemblyConfig.join"},
    )
    completion: CompletionParameters = field(
        default_factory=CompletionParameters,
        metadata={"provenance": "P0 assembly-core; 3.5 assembly/config.py AssemblyConfig.completion"},
    )
    count_rule: CountRuleParameters = field(
        default_factory=_count_defaults,
        metadata={"provenance": "P0 assembly-core; 3.5 assembly/config.py AssemblyConfig.count_rule"},
    )
    refinement: RefinementParameters = field(
        default_factory=_refinement_defaults,
        metadata={"provenance": "P0 assembly-core; 3.5 assembly/config.py AssemblyConfig.refinement"},
    )


# AS4, P0 assembly-joins.parameters.csv. Values retain the 3.5 search mechanism.
EXPLORATORY_PENALTY = 10.0
POOL_QUOTA_DOUBLINGS = 4
POOL_EXTENDED_DOUBLINGS = 6
SWEEP_BACKSTOP_BASE = 100
SWEEP_BACKSTOP_PER_ROW = 20
PENALTY = 20.0
PENALTY_FLAT_SITES = 5000
PENALTY_REFERENCE_SITES = 2000
BINS_PER_BLOCK = 20
MIN_SITES_PER_BIN = 10
MAX_SITES_PER_BIN = 100
MAX_BRANCHES = 4
IMPROVEMENT_ABSOLUTE = 1e-8
IMPROVEMENT_RELATIVE = 1e-9
SPARSE_RELATIVE = 1e-9
SPARSE_ABSOLUTE = 1e-6
NOVELTY_TOLERANCE = 1e-9
NOVELTY_FLOOR = 2.0**-19
SWAP_PASSES = 20
SEGMENT_MAXIMUM = 3
SEGMENT_MINIMUM_SPAN = 2
SEGMENT_WINDOW = 8
# AS6, U13 alternatives.py: bounded Cartesian continuation, not the AS4 beam quota.
BRANCH_LIMIT = 16
# AS6, U13 terminal_winners: single-linkage class width in join_ll quanta.
WIDE_CLASS_QUANTA = 4

# AS3, P0 assembly-joins.parameters.csv and 3.5 linker/{em,kernels}.py.
LINKER_THETA_CAP = 0.5
LINKER_THETA_FLOOR = 1e-15
LINKER_EMISSION_FLOOR = 1e-300
LINKER_MIN_SCALED_MASS = 1e-250
LINKER_MAX_LOG_RANGE = 500.0
# Cost/route only: existing execution lanes and block-product routing.
LINKER_LANE_WIDTH = 32
LINKER_MIN_ROUND_CELLS = 50000
LINKER_MAX_PRODUCT_LOG_RANGE = 500.0
LINKER_PROPAGATION_PRODUCT_FOUNDERS = 16
LINKER_EDGE_PRODUCT_FOUNDERS = 32
