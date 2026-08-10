"""Shared outer founder-count model-selection primitives.

These helpers implement only the block-founder complexity convention used by
K growth and recovery.  Candidate-space mixture BIC and cavity predictive
scores are different criteria and deliberately remain in their owning
subsystems.
"""

import math


def compute_founder_complexity_cost(
    cc_scale,
    n_samples,
    n_sites,
    use_log_bic=False,
    n_blocks=1,
):
    """Return the outer per-founder complexity cost.

    The default linear convention is
    ``cc_scale * (n_sites / 200) * n_samples * n_blocks``.  The optional
    historical log convention is preserved exactly for a single block and
    extended linearly over explicitly combined independent blocks.
    """

    site_growth = n_sites / 200.0
    if use_log_bic:
        log_n = math.log(max(n_samples * n_sites, 2))
        return cc_scale * log_n * site_growth * n_blocks
    return cc_scale * site_growth * n_samples * n_blocks


def compute_outer_bic(k, nll, complexity_cost):
    """Return ``k * complexity_cost + 2 * nll`` (lower is better)."""

    return k * complexity_cost + 2.0 * nll


__all__ = ["compute_founder_complexity_cost", "compute_outer_bic"]
