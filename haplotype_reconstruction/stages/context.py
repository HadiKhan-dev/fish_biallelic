"""Carry the bound inputs of one invocation into stage task factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StageRun:
    """Explicit run capabilities; scientific settings are already resolved."""

    root: str
    output: str
    resolved: Any
    maps: Any
    manifest: Any
    cores: int
    ties: Any = None
    source: Any = None
    scratch: str | None = None


def default_scratch(checkpoints):
    """Keep disposable work beside the immutable checkpoint layout on its filesystem."""
    return checkpoints.parent / ("." + checkpoints.name + ".scratch")
