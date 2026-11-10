"""Name the scientific hierarchy, linker and join parameters with provenance.

Owns the hierarchy schedule and shared linker and join parameter records.
No settings files, environment flags or computational routing are read here.
Method: docs/methods/assembly.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from haplotype_reconstruction.core.ties import WIDE_CLASS_QUANTA as WIDE_CLASS_QUANTA

from .completion.parameters import CompletionParameters
from .count_rule.parameters import CountRuleParameters
from .refinement.parameters import RefinementParameters


@dataclass(frozen=True)
class LinkerParameters:
    """The distance-aware sum-product linker that fits dense haploid transitions between blocks.

    max_sites: linking proxies keep at most this many sites per input component (a stride over
        its usable sites); emissions marginalise unknown alleles at partial sites.
    max_iterations: maximum updates per gap fit (convergence may stop earlier).
    robustness: genotype emission mixture P = (1 - r) P(g | pair) + r / 3, with a numerical likelihood floor.
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
        default=2000,
        metadata={
            "role": "state", "axes": (), "dtype": "int64", "stored": True,
            "units": "sites/component",
            "provenance": "At most 2000 usable proxy sites per component, bounding each gap fit scan.",
        },
    )
    max_iterations: int = field(
        default=50,
        metadata={
            "role": "state", "axes": (), "dtype": "int64", "stored": True,
            "units": "updates/gap fit",
            "provenance": "At most 50 EM updates per gap fit; convergence may stop earlier.",
        },
    )
    robustness: float = field(
        default=0.01,
        metadata={
            "role": "state", "axes": (), "dtype": "float64", "stored": True,
            "units": "probability",
            "provenance": "Uniform genotype mixture weight 0.01, limiting sensitivity to incompatible evidence.",
        },
    )
    error_fraction: float = field(
        default=0.01,
        metadata={
            "role": "state", "axes": (), "dtype": "float64", "stored": True,
            "units": "probability",
            "provenance": "Stationary error-tract copying fraction 0.01 in the within-block model.",
        },
    )
    error_tract_bp: float = field(
        default=20000.0,
        metadata={
            'role': 'state',
            'axes': (),
            'dtype': 'float64',
            'stored': True,
            'units': 'bp',
            'provenance': 'Mean error-tract length 20000 bp in the within-block quality process.',
        },
    )
    min_log_probability: float = field(
        default=-10.0,
        metadata={
            "role": "state", "axes": (), "dtype": "float64", "stored": True,
            "units": "nats",
            "provenance": "Transition log floor -10 before row renormalisation, retaining support for every edge.",
        },
    )
    pseudo_count: float = field(
        default=0.1,
        metadata={
            "role": "state", "axes": (), "dtype": "float64", "stored": True,
            "units": "expected edge counts",
            "provenance": "Add 0.1 to each edge count before normalising the transition row.",
        },
    )
    uniform_mix: float = field(
        default=0.01,
        metadata={
            "role": "state", "axes": (), "dtype": "float64", "stored": True,
            "units": "probability",
            "provenance": "Mix each fitted transition row with uniform weight 0.01 to retain robust support.",
        },
    )
    min_change: float = field(
        default=0.001,
        metadata={
            "role": "state", "axes": (), "dtype": "float64", "stored": True,
            "units": "absolute probability",
            "provenance": "Stop when the maximum undamped M-step residual is at most 0.001.",
        },
    )
    learning_rate_decay: float = field(
        default=1.0,
        metadata={
            "role": "state", "axes": (), "dtype": "float64", "stored": True,
            "units": "dimensionless",
            "provenance": "Per-update damping decay 1 takes full M-steps at the default.",
        },
    )
    minimum_learning_rate: float = field(
        default=0.1,
        metadata={
            "role": "state", "axes": (), "dtype": "float64", "stored": True,
            "units": "dimensionless",
            "provenance": "Lower bound 0.1 on the M-step damping rate when decay is configured.",
        },
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
        default=16,
        metadata={
            "role": "state", "axes": (), "dtype": "int64", "stored": True,
            "units": "paths/endpoint",
            "provenance": "Initial endpoint quota 16; saturation doubles it to test pool adequacy.",
        },
    )
    full_scores_per_kind: int = field(
        default=16,
        metadata={
            "role": "state", "axes": (), "dtype": "int64", "stored": True,
            "units": "proposals/kind/sweep",
            "provenance": "Exactly score the top 16 proposals per kind in a tier-one sweep.",
        },
    )
    max_bins: int = field(
        default=2000,
        metadata={
            "role": "state", "axes": (), "dtype": "int64", "stored": True,
            "units": "bins/batch",
            "provenance": "Target at most 2000 emission bins per batch when choosing the bin width.",
        },
    )
    tensor_budget_mb: int = field(
        default=256,
        metadata={
            'role': 'diagnostic',
            'axes': (),
            'dtype': 'int64',
            'stored': True,
            'units': 'MiB',
            'provenance': 'Painting back-pointer chunks have a 256 MiB memory allowance.',
        },
    )


@dataclass(frozen=True)
class AssemblyParameters:
    """Assembly of local panels into chromosome-scale founder components.

    batch_size: maximum adjacent input components per joined batch at every level.
    minimum_levels: reducing hierarchy levels required before the loop may stop because
        only phase breaks separate the remaining components. The loop also stops whenever
        a level cannot reduce the component count.
    min_boundary_informative_samples: a boundary between adjacent components becomes a phase
        break (never joined) when fewer samples are informative on both sides.
    gap_horizon_scale: inverse-Morgan scale of the linker and beam's maximum transition
        gap at levels >= 2: max(1, 1 + floor(recombination_tolerance /
        (mean component span in Morgans x gap_horizon_scale))). Level 1 uses every gap.
        This search-horizon control is independent of local selection's copying intensity.
    recombination_tolerance: dimensionless span budget in the maximum-gap rule; divided
        by the mean component span in Morgans times gap_horizon_scale.
    cost_scale: scale of the per-founder cost c = cost_scale x (mean sites per block / 200) x S x B
        (count_rule.criterion.founder_cost) of the join and founder refinement.
    """

    batch_size: int = field(
        default=10,
        metadata={
            "units": "components/batch",
            "provenance": "Join at most 10 adjacent components in each symmetric hierarchy batch.",
        },
    )
    minimum_levels: int = field(
        default=4,
        metadata={
            "units": "levels",
            "provenance": "Run at least 4 reducing levels before the all-phase-breaks stopping condition applies.",
        },
    )
    min_boundary_informative_samples: int = field(
        default=4,
        metadata={
            "units": "samples",
            "provenance": "A boundary with fewer than 4 informative samples on both sides is a phase break.",
        },
    )
    gap_horizon_scale: int = field(
        default=3,
        metadata={
            "units": "1/Morgan",
            "provenance": "Inverse-Morgan scale 3 converts mean component span to the higher-level transition horizon.",
        },
    )
    recombination_tolerance: float = field(
        default=0.5,
        metadata={
            'units': 'dimensionless',
            'provenance': 'Span budget 0.5 in the higher-level maximum-gap rule.',
        },
    )
    cost_scale: float = field(
        default=0.5,
        metadata={
            "units": "dimensionless",
            ("provenance"): (
                "Multiplier 0.5 in founder cost: scale times mean sites/200 times informative samples times blocks."
            ),
        },
    )
    linker: LinkerParameters = field(
        default_factory=LinkerParameters,
        metadata={
            "units": "parameter record",
            "provenance": "Default transition-model and EM settings for the linking proxies.",
        },
    )
    join: JoinParameters = field(
        default_factory=JoinParameters,
        metadata={
            "units": "parameter record",
            "provenance": "Default endpoint-beam and certified panel-search settings.",
        },
    )
    completion: CompletionParameters = field(
        default_factory=CompletionParameters,
        metadata={
            "units": "parameter record",
            "provenance": "Default cross-fit, cavity, link and occupancy settings before assembly.",
        },
    )
    count_rule: CountRuleParameters = field(
        default_factory=CountRuleParameters,
        metadata={
            "units": "parameter record",
            "provenance": "Default effective-carrier and sequence-shadow founder-count settings.",
        },
    )
    refinement: RefinementParameters = field(
        default_factory=RefinementParameters,
        metadata={
            "units": "parameter record",
            "provenance": "Default founder-refinement settings used between hierarchy levels.",
        },
    )


# Join search settings (docs/methods/assembly.md#aj12 through #aj29).
# The exploratory painting caps its switch penalty to reveal coordinated switches;
# it generates proposals only, whose acceptance uses the ordinary join penalty.
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
# Complete joint input continuations (docs/methods/assembly.md#u13), separate from endpoint beam quotas.
BRANCH_LIMIT = 16

# Linker probability floors and log-range bounds (docs/methods/assembly.md#aj07).
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
