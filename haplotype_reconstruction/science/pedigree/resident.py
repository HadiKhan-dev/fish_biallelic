"""Resident source masses, transmitted projection and child terms, in dependency order.
Method: docs/methods/pedigree.md#preparation-markers-and-source-factors
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...contracts.pedigree import PreparedComponent
from . import forward, projection
from .parameters import PedigreeParameters


@dataclass
class ResidentComponent:
    """Worker-resident scoring state of one prepared component (not a record)."""

    pair_masses: np.ndarray | None  # float64 [N, B, S (S + 1) / 2] unordered source pair masses
    projection: projection.Projection
    terms: forward.ChildTerms


def resident(
    component: PreparedComponent,
    config: PedigreeParameters,
    claim_threads: Callable[[], int | None] = lambda: None,
    *,
    reverse: bool | None = None,
    include_pair_masses: bool = True,
) -> ResidentComponent:
    """Source pair masses, projection and child terms of one component; the source laws are the prepared ones
    (PreparedComponent.sources), the bridges and child terms are built here."""
    reverse = config.reverse_conventions if reverse is None else reverse
    claim_threads()
    # a writable copy (the stored laws are read-only views; exposure's kernels take writable masses)
    masses = np.array(component.sources.pair_mass, dtype=np.float64) if include_pair_masses else None
    claim_threads()
    projected = projection.project(
        component,
        source_gap_support_sites=config.scoring.source_gap_support_sites,
        bridge_tolerance=config.scoring.bridge_postcheck_tolerance,
        bridge_max_iterations=config.scoring.bridge_max_iterations,
        reverse=reverse,
    )
    transition = component.transition
    terms = forward.ChildTerms(
        projected,
        component.genotype_likelihoods,
        component.observed,
        component.selector_switch_probability,
        transition.same,
        transition.one_change,
        transition.two_changes,
        config.scoring.null_prior_samples,
        reads=component.child_reads,
    )
    return ResidentComponent(masses, projected, terms)
