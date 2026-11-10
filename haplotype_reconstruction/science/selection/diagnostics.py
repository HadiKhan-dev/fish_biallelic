"""Collect task-local selection census records without files, clocks or environment.

Persistent result summaries belong to contracts.selection. This module owns the
explicit diagnostic sink only; no scientific choice reads its collected values.
Method: docs/methods/selection.md#ls08.
"""

from __future__ import annotations

from collections.abc import Callable

from ...core.ties import TieCensus

__all__ = ["CensusSink"]

CensusSink = Callable[[TieCensus], None]
