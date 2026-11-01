"""Locate a run's checkpoint root without reopening its original observations.

The resolved config is routing metadata only (P0 ENG-02/ENG-30). Reading results
must continue to work when an input VCF or a template directory has moved.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..core.errors import SettingsError

__all__ = ["RunPaths"]


@dataclass(frozen=True)
class RunPaths:
    """Absolute output, checkpoint and log paths; construction never creates them."""

    output: Path
    checkpoints: Path
    logs: Path

    @classmethod
    def from_directory(cls, run: str | Path) -> RunPaths:
        """Read only checkpoint routing from RUN/config.toml, defaulting to RUN/checkpoints."""
        output = Path(run).expanduser().resolve()
        checkpoints = output / "checkpoints"
        resolved = output / "config.toml"
        if resolved.is_file():
            try:
                value = tomllib.loads(resolved.read_text()).get("run", {}).get("checkpoints")
            except (OSError, tomllib.TOMLDecodeError) as error:
                raise SettingsError(f"{resolved}: {error}") from None
            if value:
                checkpoints = Path(value).expanduser().resolve()
        return cls(output, checkpoints, output / "logs")
