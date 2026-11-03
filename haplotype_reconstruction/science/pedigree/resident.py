"""Resident source masses, transmitted projection and child terms, in dependency order.
Method: docs/methods/pedigree.md#preparation-markers-source-factors-and-information-blocks
PE1–PE3 preserve this 3.5 method.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...contracts.pedigree import PreparedComponent
from . import forward, projection, sources
from .parameters import PedigreeConfig


@dataclass
class ResidentComponent:
    """Worker-resident scoring state of one prepared component (not a record)."""

    pair_masses: np.ndarray | None  # float64 [N, B, S (S + 1) / 2] unordered source pair masses
    projection: projection.Projection
    terms: forward.ChildTerms


def resident(
    component: PreparedComponent,
    config: PedigreeConfig,
    claim_threads: Callable[[], int | None] = lambda: None,
    *,
    reverse: bool | None = None,
    include_pair_masses: bool = True,
) -> ResidentComponent:
    """Exact source pair masses, projection and child terms of one component."""
    reverse = config.reverse_conventions if reverse is None else reverse
    claim_threads()
    marginals = sources.marginals(component.factors, component.transition)
    claim_threads()
    # Exposure needs only the unordered pair masses (half the size of the ordered marginals); the
    # projection then consumes the marginals in place.
    masses = sources.pair_masses(marginals) if include_pair_masses else None
    claim_threads()
    projected = projection.project(
        component,
        marginals,
        source_gap_support_sites=config.scoring.source_gap_support_sites,
        bridge_tolerance=config.scoring.bridge_postcheck_tolerance,
        bridge_max_iterations=config.scoring.bridge_max_iterations,
        reverse=reverse,
    )
    del marginals
    transition = component.transition
    terms = forward.ChildTerms(
        projected,
        component.genotype_likelihoods,
        component.observed,
        component.information_exponent,
        component.selector_switch_probability,
        transition.same,
        transition.one_change,
        transition.two_changes,
        config.scoring.mismatch_probability,
        config.scoring.null_prior_samples,
    )
    return ResidentComponent(masses, projected, terms)
