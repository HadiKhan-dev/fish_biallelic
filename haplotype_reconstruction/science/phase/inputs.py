"""Build component rates and refit-state alleles (P0 FP02, FP13)."""

import numpy as np

from ...contracts.phase import PhaseMap, PhasePainting, PhaseRates
from ..shared.recombination import haldane_fraction
from .parameters import (
    LABEL_SWITCH_MINIMUM,
    MEIOSIS_SWITCH_MINIMUM,
    NOT_COPY_ENTRY_MINIMUM,
    NOT_COPY_EXIT_MINIMUM,
    PhaseParameters,
)


def _state_alleles(states, panel_inference, n_states):
    """int8 [S, Lc]: each painting state's allele at every panel site: the common refit allele of its rows (all
    called and agreeing), else -1; BACKGROUND -1."""
    out = np.full((n_states, panel_inference.shape[1]), -1, dtype=np.int8)
    for c, rows in enumerate(states.classes):
        rows = list(rows)
        allele = panel_inference[rows[0]].astype(np.int16)
        ok = allele >= 0
        for r in rows[1:]:
            ok &= panel_inference[r] == allele
        out[c] = np.where(ok, allele, -1)
    return out


def edge_rates(painted: PhasePainting, genetic_map: PhaseMap, config: PhaseParameters) -> PhaseRates:
    """(theta_l, theta_m, alpha, beta) per bin edge: the painting's per-track label switch (its pair stay
    probability is the square of a track's; at most (S - 1) / S, the uniform-destination limit), the Haldane meiosis
    switch between bin centres (1/2 across protected intervals), and the not-a-copy entry and exit."""
    centres = np.asarray(painted.bins.centers, dtype=np.float64)
    dbp = np.diff(centres)
    S = painted.states.n_states
    theta_l = np.clip(
        1.0 - np.sqrt(np.exp(np.asarray(painted.transitions.log_same, dtype=np.float64))),
        LABEL_SWITCH_MINIMUM,
        (S - 1) / S,
    )
    morgans = np.asarray(genetic_map.distance_morgans, dtype=np.float64)
    theta_m = np.clip(haldane_fraction(morgans), MEIOSIS_SWITCH_MINIMUM, 0.5)
    for start, end in genetic_map.protected:  # transmission resets at order breakpoints
        theta_m[(centres[1:] >= start) & (centres[:-1] <= end)] = 0.5
    alpha = np.clip(1.0 - np.exp(-config.not_copy_rate_per_bp * dbp), NOT_COPY_ENTRY_MINIMUM, 0.5)
    beta = np.clip(1.0 - np.exp(-dbp / config.not_copy_mean_bp), NOT_COPY_EXIT_MINIMUM, 1.0 - NOT_COPY_EXIT_MINIMUM)
    return PhaseRates(theta_l, theta_m, alpha, beta)


__all__ = ["edge_rates"]
