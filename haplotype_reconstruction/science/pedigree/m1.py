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
from .parameters import LOG_FLOOR

MODEL = "genome-relative-m1-floor-retained-m2-v1"


def temperature(markers, config):
    blocks = np.maximum(
        np.ceil(np.asarray(markers, dtype=np.float64) / config.preparation.markers_per_information_block), 1.0
    )
    return blocks**config.panel.information_tempering_power


def identity(config) -> dict:
    """Only model settings affecting aggregate M1 and therefore derived caches."""
    return dict(
        model=MODEL,
        chromosome_score_storage="raw",
        m2_panel="all-retained-eligible-pairs",
        centered_log_floor=LOG_FLOOR,
        markers_per_information_block=config.preparation.markers_per_information_block,
        information_tempering_power=config.panel.information_tempering_power,
    )
