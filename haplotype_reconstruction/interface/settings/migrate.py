"""Translate every 3.5 setting explicitly into the versioned 4.0 user schema.

Owns migration reporting and atomic TOML output (P0 ENG-09; WP_I1 section 4.2).
Never opens observations, runs science or silently drops an input key. The old
metadata sections remain a refusal because inference reads no sample metadata.
"""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.errors import SettingsError
from ...core.files import atomic_path
from .load import _check_removed, _convert, _toml_value, load_settings
from .schema import Settings

__all__ = ["MigrationReport", "migrate_mapping", "migrate_config"]

_PAIR_NOTE = (
    "pairs.csv is no longer written by default; set [relatedness] pairwise_ibd = true to keep it (cost O(N^2 L))"
)


@dataclass(frozen=True)
class MigrationReport:
    """A loadable TOML translation, its complete audit lines, and validated settings."""

    text: str
    translations: tuple[str, ...]
    settings: Settings


def _leaves(mapping: dict[str, Any], prefix: tuple[str, ...] = ()):
    for key, value in mapping.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            yield from _leaves(value, path)
        else:
            yield path, value


def _mapping_text(mapping: dict[str, Any]) -> str:
    lines: list[str] = []

    def emit(table: dict[str, Any], prefix: tuple[str, ...]) -> None:
        if prefix:
            lines.extend(("", f"[{'.'.join(prefix)}]"))
        for key, value in table.items():
            if not isinstance(value, dict):
                lines.append(f"{key} = {_toml_value(value)}")
        for key, value in table.items():
            if isinstance(value, dict):
                emit(value, (*prefix, key))

    emit(mapping, ())
    return "\n".join(lines) + "\n"


def migrate_mapping(mapping: dict[str, Any], *, cwd: str | Path | None = None) -> MigrationReport:
    """Validate a 3.5 mapping and preserve every carried value, including relative paths.

    Returns a frozen report without mutating the mapping or writing files.
    Unsupported presets, removed non-default options and unknown keys raise SettingsError.
    """
    _check_removed(mapping)
    if mapping.get("preset") == "aulstu":
        raise SettingsError("the aulstu preset is not part of 4.0; list its contigs explicitly with [input] contigs")
    result = copy.deepcopy(mapping)
    lines: list[str] = []
    execution = result.get("execution", {})
    if isinstance(execution, dict):
        if "process_contigs" in execution:
            value = _convert("strings", execution["process_contigs"], "[execution] process_contigs", (), Path.cwd())
            if value:
                raise SettingsError("shards are workflow-mode steps in 4.0: haplotypes reconstruct CONFIG --contig C")
            del execution["process_contigs"]
            lines.append("dropped [execution] process_contigs at its empty default; use workflow-mode steps")
        if "batch_queue" in execution:
            value = _convert("path", execution["batch_queue"], "[execution] batch_queue", (), Path.cwd())
            if value is not None:
                raise SettingsError("helper queues are replaced by workflow mode")
            del execution["batch_queue"]
            lines.append(
                "dropped [execution] batch_queue at its empty default; helper queues are replaced by workflow mode"
            )
    if "version" in result and (type(result["version"]) is not int or result["version"] not in (3, 4)):
        raise SettingsError(f"cannot migrate settings version {result['version']!r}; expected 3 or 4")
    for path, _ in _leaves(result):
        if path != ("version",):
            lines.append(f"kept {'.'.join(path)}: same key, same value")
    result = {"version": 4, **{key: value for key, value in result.items() if key != "version"}}
    lines.append("set version = 4")
    relatedness = result.setdefault("relatedness", {})
    if isinstance(relatedness, dict) and "pairwise_ibd" not in relatedness:
        relatedness["pairwise_ibd"] = False
        lines.append("added [relatedness] pairwise_ibd = false: " + _PAIR_NOTE)
    settings = load_settings(options=result, cwd=cwd)
    header = "# Migrated to haplotypes 4.0\n" + "".join(f"# {line}\n" for line in lines)
    # Version remains the first TOML assignment; all translation comments precede it.
    return MigrationReport(header + _mapping_text(result), tuple(lines), settings)


def migrate_config(
    path: str | Path, output: str | Path | None = None, *, cwd: str | Path | None = None
) -> MigrationReport:
    """Translate a TOML file and optionally atomically write it after all validation succeeds."""
    try:
        with Path(path).open("rb") as handle:
            mapping = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SettingsError(f"{path}: {error}") from None
    report = migrate_mapping(mapping, cwd=cwd)
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(destination) as temporary:
            temporary.write_text(report.text)
    return report
