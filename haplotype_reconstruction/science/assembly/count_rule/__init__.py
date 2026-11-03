"""The pipeline's one founder-count rule, shared by the assembly join and founder refinement.

criterion: the effective-carrier test on the total gain and its sequence part, the bootstrap
ambiguity band, count cycles, optimistic-bound tests and the per-founder cost.
geometry: private cells, nearest rows, shadow paths and novelty (over the pre-fill inference calls).
config: CountRuleConfig (bootstrap, shadow families, the join's near-identity threshold).

Owns: the shared count-rule public API.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: WP_AS5 §6; AJ30–AJ35.
Identity: Join and refinement consume one criterion; CountRuleConfig is only a compatibility spelling.
"""

from .criterion import (
    CRITERION,
    CountDecision,
    MarginTest,
    ShadowDecisionFamily,
    addition_bound_excludes,
    addition_per_sample_bound_excludes,
    bootstrap_seed,
    decide,
    deletion_bound_excludes,
    effective_carriers,
    founder_cost,
    margin,
    passes,
    pre_cycle_count,
    smallest_threshold,
)
from .geometry import RowGeometry, covered_rows, has_novelty, novel_paths, private_mask, shadow_choices, shadow_path
from .parameters import CountRuleParameters
from .shadow_bounds import FactorPlan, attempt_certificate, factor_plan, factor_sizes, total_dominance_certificate

__all__ = [
    "CountRuleConfig",
    "CountRuleParameters",
    "CRITERION",
    "CountDecision",
    "MarginTest",
    "ShadowDecisionFamily",
    "addition_bound_excludes",
    "addition_per_sample_bound_excludes",
    "bootstrap_seed",
    "decide",
    "deletion_bound_excludes",
    "effective_carriers",
    "founder_cost",
    "margin",
    "passes",
    "pre_cycle_count",
    "smallest_threshold",
    "RowGeometry",
    "covered_rows",
    "has_novelty",
    "novel_paths",
    "private_mask",
    "shadow_path",
    "shadow_choices",
    "FactorPlan",
    "factor_sizes",
    "factor_plan",
    "total_dominance_certificate",
    "attempt_certificate",
]

# Transitional spelling used by the neighbouring relocated adapters.
CountRuleConfig = CountRuleParameters
