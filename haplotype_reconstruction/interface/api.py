"""Provide the lazy Python entry points with an explicit numerical runtime boundary.

Settings loading uses only the standard library. Execution is delegated to the
runtime and the same graph composition used by the CLI.
Results are opened lazily and never reopen original observations.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core import errors
from .settings.load import _assign, _merge
from .settings.load import load_settings as _load_settings
from .settings.schema import Settings

if TYPE_CHECKING:
    from ..results import Results
    from .run import RunResult

__all__ = ["load_settings", "run", "open_results", "errors"]


def load_settings(path: str | Path | None = None, **overrides: Any) -> Settings:
    """Resolve a TOML file with optional section mappings or dotted-key keyword overrides.

    The loader's preset/options/overrides/cwd arguments are also accepted. Paths are
    relative to cwd, never to the TOML file. Invalid values raise errors.SettingsError.
    No input mapping or file is modified and no numerical library is imported.
    """
    arguments = dict(overrides)
    preset = arguments.pop("preset", None)
    cwd = arguments.pop("cwd", None)
    sets = arguments.pop("overrides", ())
    options = arguments.pop("options", {})
    mapping: dict[str, Any] = {}
    for key, value in arguments.items():
        _assign(mapping, key.split("."), value)
    return _load_settings(path, preset=preset, options=_merge(options, mapping), overrides=sets, cwd=cwd)


def _run_configured(settings: Settings, cores: int, *, in_process: bool = False) -> RunResult:
    """Enter a numerical runtime before importing graph or scientific dependencies."""
    from ..engine.runtime import configure_worker

    configure_worker(cores)
    from .run import execute

    return execute(settings, command="run", in_process=in_process)


def run(
    settings: Settings | str | Path,
    output: str | Path,
    *,
    cores: int | str | None = None,
    in_process: bool = False,
) -> RunResult:
    """Initialize if needed and execute a run in isolated runtime processes by default.

    in_process=True first checks any live BLAS/Numba policy and refuses a mismatch.
    Returns RunResult; typed input/settings/resource errors are preserved. The
    supplied frozen settings and the caller's numerical policy are not rewritten.
    """
    if not isinstance(settings, Settings):
        settings = load_settings(settings)
    settings = replace(settings, run=replace(settings.run, output=Path(output).expanduser().resolve()))
    if cores is not None:
        settings = replace(settings, execution=replace(settings.execution, cores=cores))
    from .run import resolve_cores

    count = resolve_cores(settings)
    if in_process:
        return _run_configured(settings, count, in_process=True)
    from ..engine.runtime import isolated

    return isolated(_run_configured, settings, count, cores=count)


def open_results(path: str | Path, *, checkpoints: str | Path | None = None) -> Results:
    """Open typed readers for a finished or partial initialized run, without loading products."""
    from ..results import Results

    return Results(path, checkpoints=checkpoints)
