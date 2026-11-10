"""Expose versioned run readers while keeping package import free of numerical initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .reader import Results

__all__ = ["Results", "StagePanel", "RelatednessView", "recorded_input_order", "open_results"]


def __getattr__(name: str):
    if name in ("Results", "RelatednessView"):
        from . import reader

        return getattr(reader, name)
    if name == "StagePanel":
        from .panels import StagePanel

        return StagePanel
    if name == "recorded_input_order":
        from . import order

        return getattr(order, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def open_results(path: str | Path, *, checkpoints: str | Path | None = None) -> Results:
    """Open the versioned reader without routing through the application layer."""
    from .reader import Results

    return Results(path, checkpoints=checkpoints)
