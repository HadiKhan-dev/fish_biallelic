"""The pilot's strict fit of one block (ADVICE2 sections 1 and 8): the real discovery block search
(science.discovery.run.discover_block) at the strict outlier rate, all cells; its decoded known-state cells give the
unit log likelihoods on the rate grid (the one-rate estimate sums them; the two-level fit is a diagnostic). The rate is
estimated only from strict fits and never fed back into them (ADVICE2 section 8, addendum). Fits of different blocks
are independent.

Method: docs/methods/observation.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...contracts.likelihoods import gather_likelihoods
from ...contracts.observations import dense_observed
from ...core.concurrency import ThreadSource
from ..discovery.api import BlockEvidence, discover_block
from ..discovery.parameters import DiscoveryParameters
from .estimate import rate_grid
from .parameters import ObservationParameters
from .pilot import decode_units

__all__ = ["PilotFit", "pilot_fit"]


@dataclass(frozen=True)
class PilotFit:
    """One strict pilot fit of one block: ``units`` float64[M, G] (the unit log likelihoods of the samples with enough
    cells, for the two-level diagnostic), ``grid`` float64[G] (the log likelihood of every known-state cell of the block
    on the rate grid), ``cells`` and ``rows`` (the fitted panel's rows; 0: no panel)."""

    block: int
    units: np.ndarray
    grid: np.ndarray
    cells: int
    rows: int = 0


def pilot_fit(
    evidence: BlockEvidence,
    discovery: DiscoveryParameters,
    observation: ObservationParameters,
    *,
    threads: ThreadSource = 1,
    map_work=map,
) -> PilotFit:
    """The strict fit of one block at ``observation.strict_outlier_rate``, decoded under discovery's binned switching
    model (science.observation.pilot.decode_units)."""
    grid = rate_grid(observation.grid_low, observation.grid_high, observation.grid_points)
    empty = PilotFit(evidence.block, np.zeros((0, len(grid))), np.zeros(len(grid)), 0)
    if not evidence.informative or evidence.cohort is None:
        return empty
    rate = float(observation.strict_outlier_rate)
    fitted = discover_block(evidence, discovery, outlier_rate=rate, threads=threads, map_work=map_work)
    rows = 0 if fitted is None else int(fitted.latent.shape[0])
    if rows == 0:
        return empty
    kept = evidence.kept
    observed = dense_observed(evidence.observed)[:, kept]
    likelihoods = gather_likelihoods(evidence.likelihoods, slice(None))[:, kept]
    units, cells = decode_units(
        likelihoods,
        evidence.cohort[kept],
        observed,
        np.asarray(fitted.latent, np.int8),
        snps_per_bin=discovery.model.snps_per_bin,
        unknown_penalty=discovery.model.unknown_copy_penalty,
        switch_penalty=discovery.model.switch_penalty,
        decode_rate=rate,
        grid=grid,
    )
    enough = cells >= observation.minimum_unit_cells
    return PilotFit(evidence.block, units[enough], units.sum(axis=0), int(cells.sum()), rows)
