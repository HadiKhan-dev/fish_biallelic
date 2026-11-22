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


def temperature(markers, config, units=None):
    """The retention temperature T: 4.20's ceil(markers / markers_per_information_block) ** power, one effective unit
    per information group; with ``units`` (4.21 B2, ADVICE2 11.3 and 14.1 item 8) the effective-count retention
    temperature (sum over information groups of n_g / (1 + (n_g - 1) r_T)) ** power, minimum 1, under the transferred
    working temperature r_T of B1. 4.20's rule is its r_T = 1 limit."""
    if units is not None:
        return np.maximum(np.asarray(units, dtype=np.float64), 1.0) ** config.panel.information_tempering_power
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
        information_temperature=config.preparation.information_temperature,
    )
