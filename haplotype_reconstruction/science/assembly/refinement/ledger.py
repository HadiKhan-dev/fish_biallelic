"""Keep completed fits and branch-local count choices with explicit lifetimes.

Owns mutable in-component search state, never persistence or scientific ordering.
Method: docs/methods/founder_refinement.md#count-continuation.
Shared caches retain exact values for the immutable component. Fit scores are
consumed on read by SweepEngine; each continuation has fresh selection state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from haplotype_reconstruction.contracts.assembly import Fit

PanelKey = tuple[tuple[int, ...], ...]
FitToken = tuple[str, PanelKey]

__all__ = ["FitLedger", "PanelKey", "FitToken"]


class FitLedger:
    """Cache ownership for one engine branch; count-pass state resets explicitly."""

    def __init__(self) -> None:
        self.fits: dict[FitToken, tuple[Fit, ...]] = {}
        self.scores: dict[PanelKey, NDArray[np.float64]] = {}
        self.minimum_switches: dict[PanelKey, int] = {}
        self.death_classes: dict[tuple[tuple[int, ...], bytes], list[int]] = {}
        self.invocations: dict[str, PanelKey] = {}
        self.fit_scores: dict[str, NDArray[np.float64]] = {}
        self.chosen_panels: dict[FitToken, PanelKey] = {}
        self.forced_panels: dict[FitToken, PanelKey] = {}
        self.forced_count_choices: dict[str, PanelKey] = {}
        self.count_families: dict[str, dict[str, Any]] = {}
        self.chosen_count_panels: dict[str, PanelKey] = {}

    def branch(self, choices: dict[FitToken, PanelKey]) -> FitLedger:
        """Share completed results; bind only the new branch's forced fit choices."""
        child = FitLedger()
        child.fits = self.fits
        child.scores = self.scores
        child.minimum_switches = self.minimum_switches
        child.death_classes = self.death_classes
        child.forced_panels = dict(choices)
        return child

    def start_count_pass(self) -> None:
        """Reset the pass-local caches, preserving forced branch decisions."""
        self.count_families = {}
        self.chosen_count_panels = {}
