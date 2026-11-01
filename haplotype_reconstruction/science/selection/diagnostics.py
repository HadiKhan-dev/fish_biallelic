"""Collect task-local selection census records without files, clocks or environment.

Persistent result summaries belong to contracts.selection. This module owns the
explicit diagnostic sink only; no scientific choice reads its collected values.
Method: docs/methods/local_selection.md (LS08-LS31, L8 convention/census boundary).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ...core.ties import TieCensus

__all__ = ["CensusCollector", "CensusSink"]

CensusSink = Callable[[TieCensus], None]


@dataclass
class CensusCollector:
    """Append a science call's census to a private list and return an immutable receipt."""

    events: list[TieCensus] = field(default_factory=list)

    def __call__(self, event: TieCensus) -> None:
        self.events.append(event)

    def record(self) -> tuple[TieCensus, ...]:
        return tuple(self.events)
