"""Keep completed fits and branch-local count choices with explicit lifetimes.

Owns mutable in-component search state, never persistence or scientific ordering.
Method: docs/methods/founder_refinement.md#count-continuation.
Shared caches retain exact values for the immutable component. Fit scores are
consumed on read by SweepEngine; each continuation has fresh selection state.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from haplotype_reconstruction.contracts.assembly import Fit
from haplotype_reconstruction.core.sizes import shallow_bytes

from .parameters import SCORE_LEDGER_BYTES, SCORE_LEDGER_ENTRIES, SCORE_LEDGER_ENTRY_OVERHEAD

PanelKey = tuple[tuple[int, ...], ...]
FitToken = tuple[str, PanelKey]

__all__ = ["FitLedger", "PanelKey", "FitToken"]


@dataclass
class _ScoreAdmission:
    """Shared residence accounting for optional scores, separate from fitted ones."""

    entries: int = 0
    used: int = 0
    lock: Any = field(default_factory=threading.Lock)


class FitLedger:
    """Cache ownership for one engine branch; count-pass state resets explicitly."""

    def __init__(self) -> None:
        self.fits: dict[FitToken, tuple[Fit, ...]] = {}
        self.scores: dict[PanelKey, NDArray[np.float64]] = {}
        self._score_admission = _ScoreAdmission()
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
        child._score_admission = self._score_admission
        child.minimum_switches = self.minimum_switches
        child.death_classes = self.death_classes
        child.forced_panels = dict(choices)
        return child

    def remember_score(self, key: PanelKey, scores: NDArray[np.float64]) -> None:
        """Retain an owned exact miss vector if its key and storage fit the allowance.

        The immutable workspace fixes the score function. Count endpoints keep
        their own decisions and bootstrap bands; only panel score vectors share.
        Reads may race with identical computations; admission and fit publication
        share a lock so neither the limits nor fitted-vector precedence can race.
        """
        admission = self._score_admission
        with admission.lock:
            if key in self.scores or admission.entries >= SCORE_LEDGER_ENTRIES:
                return
            # Count every referenced integer, even when Python interns or shares
            # it. Scorers return owned ndarrays, whose sizeof includes the payload.
            size = (
                shallow_bytes(scores)
                + shallow_bytes(key)
                + sum(shallow_bytes(row) + sum(map(shallow_bytes, row)) for row in key)
                + SCORE_LEDGER_ENTRY_OVERHEAD
            )
            if admission.used + size > SCORE_LEDGER_BYTES:
                return
            self.scores[key] = scores
            admission.entries += 1
            admission.used += size

    def publish_score(self, key: PanelKey, scores: NDArray[np.float64]) -> None:
        """Publish a completed fit unconditionally, taking precedence over a miss.

        Fitted vectors have the completed-fit lifetime. Miss admission is
        cumulative for the component, including misses later published by fits:
        their keys and dictionary capacity may remain resident. The allowance
        bounds additional misses, not the whole completed-fit ledger.
        """
        admission = self._score_admission
        with admission.lock:
            self.scores[key] = scores

    def start_count_pass(self) -> None:
        """Reset the pass-local caches, preserving forced branch decisions."""
        self.count_families = {}
        self.chosen_count_panels = {}
