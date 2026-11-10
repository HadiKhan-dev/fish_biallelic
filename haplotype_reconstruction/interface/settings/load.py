"""Loading the run configuration: preset, TOML file, command-line options and `--set` overrides, strictly.

Precedence (later wins): schema defaults < preset (interface/settings/presets/<name>.toml) < TOML file < command-line
options < `--set section.key=value`. Unknown sections and keys are errors naming the valid ones (a section removed
from the schema names its removal: REMOVED_SECTIONS); values are type-checked by their field kind; relative paths
resolve against the working directory. The resolved configuration is written back as TOML (`settings_text`) to
OUTPUT/config.toml and into the run record. The nested simulation.reads preset is expanded after
merging these layers, with every explicit read key taking precedence over its values.
"""

from __future__ import annotations

import json
import math
import re
import tomllib
from collections.abc import Sequence, Set
from importlib import resources
from pathlib import Path
from typing import Any

from ...core.errors import SettingsError
from ...core.files import atomic_path
from .schema import Settings, schema_fields

PRESETS = ("tromau", "generic")

# The pipeline uses no sample metadata. Inference reads only the observations (VCF/BCF, or the simulated
# reads) and the genetic map, and sample names are identifiers; every sequenced sample is a candidate parent of every
# other. A configuration (file, preset, option or --set) that still sets one of these sections fails at load.
REMOVED_SECTIONS = {
    ("samples",): "the [samples] workbook and its generation and sex labels",
    ("pedigree", "eligibility"): "parent eligibility is every sequenced sample: no cohort or constraints policy and "
    "no opposite-sex pair rule",
    ("reports",): "the reference-consistency report ([reports] reference_consistency), which read generation labels",
}
_MEMORY_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(?:I?B)?\s*$", re.IGNORECASE)
_MEMORY_UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def _section_classes(cls=Settings, prefix=()) -> dict:
    """{section path: section class} of every (nested) section."""
    result = {}
    for item in schema_fields(cls):
        if item.metadata["kind"] == "section":
            path = (*prefix, item.name)
            section_class = type(item.default_factory())
            result[path] = section_class
            result.update(_section_classes(section_class, path))
    return result


_SECTIONS = _section_classes()


def preset_mapping(name: str) -> dict:
    """The TOML mapping of a built-in preset."""
    if name not in PRESETS:
        raise SettingsError(f"unknown preset {name!r}; presets: {', '.join(PRESETS)}")
    text = (
        resources.files("haplotype_reconstruction.interface.settings").joinpath("presets", f"{name}.toml").read_text()
    )
    return tomllib.loads(text)


def _merge(base: dict, update: dict, prefix=()) -> dict:
    """Merge section tables key by key; any other value replaces the earlier one."""
    result = dict(base)
    for key, value in update.items():
        path = (*prefix, key)
        if path in _SECTIONS and isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value, path)
        else:
            result[key] = value
    return result


def _given_keys(mapping: dict, prefix=()) -> set:
    """Paths of the settings a mapping gives (sections descended into)."""
    keys = set()
    for key, value in mapping.items():
        path = (*prefix, key)
        if path in _SECTIONS and isinstance(value, dict):
            keys |= _given_keys(value, path)
        else:
            keys.add(path)
    return keys


def _parse_value(text: str):
    """A `--set` value: TOML syntax if it parses (numbers, booleans, arrays, tables, quoted strings), else text."""
    try:
        return tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        return text


def parse_override(text: str) -> tuple[list[str], object]:
    """``section[.sub].key=value`` -> (key path, value)."""
    name, separator, value = text.partition("=")
    if not separator or not name.strip():
        raise SettingsError(f"--set expects section.key=value, got {text!r}")
    return [part.strip() for part in name.strip().split(".")], _parse_value(value.strip())


def _assign(mapping: dict, path: list[str], value) -> None:
    for part in path[:-1]:
        mapping = mapping.setdefault(part, {})
        if not isinstance(mapping, dict):
            raise SettingsError(f"{'.'.join(path)}: {part} is not a section")
    mapping[path[-1]] = value


def _convert(kind: str, value, where: str, choices: tuple, cwd: Path):
    def fail(expected):
        raise SettingsError(f"{where}: expected {expected}, got {value!r}")

    if kind == "bool":
        if not isinstance(value, bool):
            fail("true or false")
        return value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            fail("a nonnegative integer")
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            fail("a finite number")
        return float(value)
    if kind == "str":
        if not isinstance(value, str):
            fail("a string")
        if choices and value not in choices:
            raise SettingsError(f"{where}: {value!r} is not one of {', '.join(repr(c) for c in choices)}")
        return value
    if kind == "path":
        if not isinstance(value, str):
            fail("a path string")
        return None if value == "" else (cwd / Path(value).expanduser()).resolve()
    if kind == "strings":
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            fail("an array of nonempty strings")
        return tuple(value)
    if kind == "ints":
        if not isinstance(value, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        ):
            fail("an array of integers")
        return tuple(value)
    if kind == "cores":
        if value != "auto" and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            fail('"auto" or a positive integer')
        return value
    if kind == "memory":
        if value == "auto" or (isinstance(value, int) and not isinstance(value, bool) and value > 0):
            return value
        match = _MEMORY_SIZE.match(value) if isinstance(value, str) else None
        if match and float(match.group(1)) * _MEMORY_UNITS[match.group(2).upper()] >= 1:
            return int(float(match.group(1)) * _MEMORY_UNITS[match.group(2).upper()])
        fail('"auto", a positive number of bytes or a size such as "200G" (binary units K, M, G, T)')
    if kind == "seed":
        if value != "random" and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            fail('a nonnegative integer or "random"')
        return value
    raise AssertionError(kind)


def _build(cls, mapping, prefix: tuple, cwd: Path):
    where = f"[{'.'.join(prefix)}]" if prefix else "top level"
    if not isinstance(mapping, dict):
        raise SettingsError(f"{where}: expected a table")
    known = {item.name: item for item in schema_fields(cls)}
    unknown = sorted(set(mapping) - set(known))
    if unknown:
        raise SettingsError(f"{where}: unknown key(s) {', '.join(unknown)}; valid: {', '.join(known)}")
    values = {}
    for name, value in mapping.items():
        item = known[name]
        kind = item.metadata["kind"]
        if kind == "section":
            values[name] = _build(type(item.default_factory()), value, (*prefix, name), cwd)
        else:
            label = f"{where} {name}" if prefix else name
            values[name] = _convert(kind, value, label, item.metadata["choices"], cwd)
    return cls(**values)


def _check_removed(mapping: dict) -> None:
    """Refuse a mapping that sets any of REMOVED_SECTIONS, naming all of them at once."""
    found = []
    for path, reason in REMOVED_SECTIONS.items():
        table: Any = mapping
        for part in path[:-1]:
            table = table.get(part) if isinstance(table, dict) else None
        if isinstance(table, dict) and path[-1] in table:
            found.append((".".join(path), reason))
    if found:
        names = ", ".join(f"[{name}]" for name, _ in found)
        reasons = "; ".join(reason for _, reason in found)
        raise SettingsError(
            f"{names}: removed in 3.4: the pipeline uses no sample metadata ({reasons}); delete "
            f"{'it' if len(found) == 1 else 'them'} from the configuration"
        )


def build_settings(mapping: dict, *, cwd: str | Path | None = None) -> Settings:
    """A typed Settings from a merged mapping (paths resolved against ``cwd``); no cross-key validation."""
    _check_removed(mapping)
    simulation = mapping.get("simulation", {})
    reads = simulation.get("reads", {}) if isinstance(simulation, dict) else {}
    if isinstance(reads, dict) and reads.get("preset") == "realistic":
        text = (
            resources.files("haplotype_reconstruction.interface.settings")
            .joinpath("presets", "realistic_reads.toml")
            .read_text()
        )
        # Expand after all user/CLI layers have merged, so every explicit key wins,
        # including values equal to schema defaults. Resolved TOML records all values.
        mapping = _merge(tomllib.loads(text), mapping)
    settings = _build(Settings, mapping, (), Path.cwd() if cwd is None else Path(cwd))
    if settings.version != 4:
        raise SettingsError(
            f"settings version {settings.version!r} is not 4; current TOML settings require version = 4"
        )
    return settings


def load_settings(
    path: str | Path | None = None,
    *,
    preset: str | None = None,
    options: dict | None = None,
    overrides: Sequence[str] = (),
    cwd: str | Path | None = None,
    fallback_contigs: Sequence[str] | None = None,
) -> Settings:
    """Resolve and validate a run configuration (see the module docstring).

    ``path``: TOML file or None; ``preset``: overrides the file's ``preset``; ``options``: {section: {key: value}}
    from command-line flags (TOML-typed values); ``overrides``: `--set` strings.
    ``fallback_contigs`` supplies an initial list only when the merged configuration has none;
    run otherwise uses its CLI contigs as a graph cut over the configured list.
    """
    user = {}
    if path is not None:
        try:
            with open(path, "rb") as handle:
                user = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise SettingsError(f"{path}: {error}") from None
    sets: dict[str, Any] = {}
    for text in overrides:
        name, value = parse_override(text)
        _assign(sets, name, value)
    layers = (user, options or {}, sets)
    chosen = preset or next((layer["preset"] for layer in reversed(layers) if layer.get("preset")), "")
    merged = preset_mapping(chosen) if chosen else {}
    given = set()
    for layer in layers:
        merged = _merge(merged, layer)
        given |= _given_keys(layer)
    merged["preset"] = chosen
    config = build_settings(merged, cwd=cwd)
    if fallback_contigs is not None and not config.input.contigs:
        merged.setdefault("input", {})["contigs"] = list(fallback_contigs)
        config = build_settings(merged, cwd=cwd)
    validate(config, given)
    return config


def validate(config: Settings, given: Set[tuple[str, ...]] = frozenset()) -> None:
    """Cross-key rules that need no input files (input-dependent checks run in the driver before computation).

    ``given``: key paths set by the user (not by the preset or the defaults).
    """
    execution, inputs = config.execution, config.input
    if execution.max_contigs and ("execution", "serial") in given and not execution.serial:
        raise SettingsError("[execution] max_contigs runs contigs serially; it cannot be combined with serial = false")
    if config.serial and config.run.stop_after == "discovery":
        raise SettingsError(
            "serial contigs are painted before the next is discovered; they cannot stop after discovery"
        )
    if not inputs.contigs:
        raise SettingsError("[input] contigs is required (no preset lists them)")
    if len(set(inputs.contigs)) != len(inputs.contigs):
        raise SettingsError("[input] contigs must be unique")
    if inputs.kind == "vcf":
        if inputs.vcf is None:
            raise SettingsError('[input] kind = "vcf" needs [input] vcf (or --vcf)')
        stray = sorted(
            ".".join(key) for key in given if key[0] == "simulation" and _value(config, key) != _value(Settings(), key)
        )
        if stray:
            raise SettingsError(f'[simulation] settings need [input] kind = "simulation": {", ".join(stray)}')
    else:
        if config.simulation.templates is None:
            raise SettingsError('[input] kind = "simulation" needs [simulation] templates (or --templates)')
        if inputs.vcf is not None:
            raise SettingsError(
                "[input] vcf is not read by simulations (templates are built with the templates command)"
            )
    if not config.simulation.cohorts or min(config.simulation.cohorts) < 1:
        raise SettingsError("[simulation] cohorts must be positive sizes")
    for name in ("site_depth_cv", "count_dispersion"):
        value = getattr(config.simulation.reads, name)
        if not math.isfinite(value) or value < 0:
            raise SettingsError(f"[simulation.reads] {name} must be finite and nonnegative")


def _value(config, path):
    value = config
    for part in path:
        value = getattr(value, part, None)
    return value


def default_output(config: Settings) -> Path:
    """work/runs/<preset or input kind>[_seed<seed>] under the working directory."""
    name = config.preset or config.input.kind
    if config.input.kind == "simulation":
        name += f"_seed{config.simulation.seed}"
    return (Path.cwd() / "work" / "runs" / name).resolve()


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        return text if any(c in text for c in ".e") else text + ".0"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"no TOML form for {type(value).__name__}")


def _plain(kind: str, value):
    if kind == "path":
        return "" if value is None else str(value)
    return value


def settings_text(config: Settings) -> str:
    """The complete resolved configuration as TOML (loadable again by load_settings)."""
    lines = [f"version = {config.version}", f"preset = {_toml_value(config.preset)}"]

    def emit(name, section):
        lines.extend(("", f"[{name}]"))
        nested = []
        for item in schema_fields(type(section)):
            value = getattr(section, item.name)
            if item.metadata["kind"] == "section":
                nested.append((f"{name}.{item.name}", value))
            else:
                lines.append(f"{item.name} = {_toml_value(_plain(item.metadata['kind'], value))}")
        for sub, value in nested:
            emit(sub, value)

    for item in schema_fields(Settings):
        if item.metadata["kind"] == "section":
            emit(item.name, getattr(config, item.name))
    return "\n".join(lines) + "\n"


def write_settings(config: Settings, path: str | Path) -> Path:
    """Write the resolved configuration atomically; return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temporary:
        temporary.write_text(settings_text(config))
    return path


__all__ = [
    "SettingsError",
    "PRESETS",
    "preset_mapping",
    "parse_override",
    "build_settings",
    "load_settings",
    "validate",
    "default_output",
    "settings_text",
    "write_settings",
]
