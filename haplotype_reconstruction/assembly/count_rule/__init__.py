"""The pipeline's one founder-count rule, shared by the assembly join and founder refinement.

criterion: the effective-carrier test on the total gain and its sequence part, the bootstrap
ambiguity band, count cycles, optimistic-bound tests and the per-founder cost.
geometry: private cells, nearest rows, shadow paths and novelty (over the pre-fill inference calls).
config: CountRuleConfig (bootstrap, ranked refits, the join's near-identity threshold).
"""
from .config import CountRuleConfig
from .criterion import (CRITERION, CountDecision, MarginTest, addition_bound_excludes,
                        addition_per_sample_bound_excludes, bootstrap_seed, decide,
                        deletion_bound_excludes, effective_carriers, founder_cost, margin,
                        passes, pre_cycle_count, smallest_threshold)
from .geometry import RowGeometry, covered_rows, has_novelty, novel_paths, private_mask, shadow_path

__all__ = [
    "CountRuleConfig", "CRITERION", "CountDecision", "MarginTest", "addition_bound_excludes",
    "addition_per_sample_bound_excludes", "bootstrap_seed", "decide", "deletion_bound_excludes",
    "effective_carriers", "founder_cost", "margin", "passes", "pre_cycle_count",
    "smallest_threshold", "RowGeometry", "covered_rows", "has_novelty", "novel_paths",
    "private_mask", "shadow_path",
]
