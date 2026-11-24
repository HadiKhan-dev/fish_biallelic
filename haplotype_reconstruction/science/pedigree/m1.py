"""Genome-wide consistent M1 floor, fixed before decision and resampling.

Retain identities with total raw M1 >= best eligible total - 60*T(total markers).
Retained identities keep raw chromosome scores. Every excluded identity takes
its child's common chromosome profile, whose sum is the genome cutoff:
 f_cj = b_cj - T_j/sum(T) * (sum(b_c) - cutoff_c).
All identity multiplicities and contamination are preserved downstream.

Method: docs/methods/pedigree.md#candidate-panel
"""

from __future__ import annotations

import numpy as np

# The screen's softmax cutoff, shared by ranks and every M1 consumer.
from .numerical_parameters import LOG_FLOOR

MODEL = "genome-relative-m1-floor-retained-m2-v1"


def retention_units(prepared) -> np.ndarray:
    """float64 [C]: the retention band's units of each prepared chromosome, every selected marker with an empty bin
    counted as one (4.21's effective units at r_T = 0). The band is a compute budget of the latent-order decision
    (4.22 R4): with the pair cap, it bounds the panel without deciding calls (wide band and no cap gave the same
    decisions on every measured root)."""
    return np.asarray(
        [float(sum(int(np.maximum(np.asarray(c.bin_sizes, dtype=np.int64), 1).sum()) for c in p.components))
         for p in prepared],
        dtype=np.float64,
    )


def temperature(units, config):
    """The retention temperature T = units ** information_tempering_power, minimum 1 (units: retention_units, or
    their genome sum for the genome band)."""
    return np.maximum(np.asarray(units, dtype=np.float64), 1.0) ** config.panel.information_tempering_power


def identity(config) -> dict:
    """Only model settings affecting aggregate M1 and therefore derived caches."""
    return dict(
        model=MODEL,
        chromosome_score_storage="raw",
        m2_panel="all-retained-eligible-pairs",
        centered_log_floor=LOG_FLOOR,
        retention_units="selected-markers",
        information_tempering_power=config.panel.information_tempering_power,
    )
