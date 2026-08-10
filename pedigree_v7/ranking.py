"""Stable, equally spaced rank utilities shared by V7 aggregation stages."""

import numpy as np


def descending_rank_votes(scores):
    """Convert candidate scores to stable descending rank votes in [0, 1]."""
    values = np.asarray(scores, dtype=np.float64)
    n_candidates = values.shape[-1]
    if n_candidates < 2:
        return np.ones_like(values)
    order = np.argsort(-values, axis=-1, kind="stable")
    ranks = np.argsort(order, axis=-1, kind="stable")
    return 1.0 - ranks / float(n_candidates - 1)
