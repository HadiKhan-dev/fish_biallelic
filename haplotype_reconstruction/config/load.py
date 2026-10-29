"""Loading the run configuration: preset, TOML file, command-line options and `--set` overrides, strictly.

Precedence (later wins): schema defaults < preset (config/presets/<name>.toml) < the TOML file < command-line
options < `--set section.key=value`. Unknown sections and keys are errors naming the valid ones; values are
type-checked by their field kind; relative paths resolve against the working directory. The resolved
configuration is written back as TOML (`config_text`) to OUTPUT/config.toml and into the run record.
"""
from __future__ import annotations

from importlib import resources
import json
import math
import os
from pathlib import Path
import re
import tomllib

from .schema import RunConfig, schema_fields


class ConfigError(ValueError):
    """An invalid configuration (unknown key, wrong type or value, inconsistent settings)."""


PRESETS = ("tromau", "aulstu")

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_MEMORY_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(?:I?B)?\s*$", re.IGNORECASE)
_MEMORY_UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def _section_classes(cls=RunConfig, prefix=()) -> dict:
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
        raise ConfigError(f"unknown preset {name!r}; presets: {', '.join(PRESETS)}")
    text = resources.files("haplotype_reconstruction.config").joinpath("presets", f"{name}.toml").read_text()
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
        raise ConfigError(f"--set expects section.key=value, got {text!r}")
    return [part.strip() for part in name.strip().split(".")], _parse_value(value.strip())


def _assign(mapping: dict, path: list[str], value) -> None:
    for part in path[:-1]:
        mapping = mapping.setdefault(part, {})
        if not isinstance(mapping, dict):
            raise ConfigError(f"{'.'.join(path)}: {part} is not a section")
    mapping[path[-1]] = value


def _convert(kind: str, value, where: str, choices: tuple, cwd: Path):
    def fail(expected):
        raise ConfigError(f"{where}: expected {expected}, got {value!r}")

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
            raise ConfigError(f"{where}: {value!r} is not one of {', '.join(repr(c) for c in choices)}")
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
                isinstance(item, int) and not isinstance(item, bool) for item in value):
            fail("an array of integers")
        return tuple(value)
    if kind == "table":
        if not isinstance(value, dict) or not all(isinstance(item, str) for item in value.values()):
            fail("a table of strings")
        return tuple(sorted(value.items()))
    if kind == "cores":
        if value != "auto" and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            fail("\"auto\" or a positive integer")
        return value
    if kind == "memory":
        if value == "auto" or (isinstance(value, int) and not isinstance(value, bool) and value > 0):
            return value
        match = _MEMORY_SIZE.match(value) if isinstance(value, str) else None
        if match and float(match.group(1)) * _MEMORY_UNITS[match.group(2).upper()] >= 1:
            return int(float(match.group(1)) * _MEMORY_UNITS[match.group(2).upper()])
        fail("\"auto\", a positive number of bytes or a size such as \"200G\" (binary units K, M, G, T)")
    if kind == "seed":
        if value != "random" and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            fail("a nonnegative integer or \"random\"")
        return value
    raise AssertionError(kind)


def _build(cls, mapping, prefix: tuple, cwd: Path):
    where = f"[{'.'.join(prefix)}]" if prefix else "top level"
    if not isinstance(mapping, dict):
        raise ConfigError(f"{where}: expected a table")
    known = {item.name: item for item in schema_fields(cls)}
    unknown = sorted(set(mapping) - set(known))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {', '.join(unknown)}; valid: {', '.join(known)}")
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


def build_config(mapping: dict, *, cwd=None) -> RunConfig:
    """A typed RunConfig from a merged mapping (paths resolved against ``cwd``); no cross-key validation."""
    return _build(RunConfig, mapping, (), Path.cwd() if cwd is None else Path(cwd))


def load_config(path=None, *, preset: str | None = None, options: dict | None = None, overrides=(),
                cwd=None) -> RunConfig:
    """Resolve and validate a run configuration (see the module docstring).

    ``path``: TOML file or None; ``preset``: overrides the file's ``preset``; ``options``: {section: {key: value}}
    from command-line flags (TOML-typed values); ``overrides``: `--set` strings.
    """
    user = {}
    if path is not None:
        try:
            with open(path, "rb") as handle:
                user = tomllib.load(handle)
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"{path}: {error}") from None
    sets = {}
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
    config = build_config(merged, cwd=cwd)
    validate(config, given)
    return config


def validate(config: RunConfig, given: set = frozenset()) -> None:
    """Cross-key rules that need no input files (input-dependent checks run in the driver before computation).

    ``given``: key paths set by the user (not by the preset or the defaults).
    """
    execution, inputs = config.execution, config.input
    if execution.max_contigs and ("execution", "serial") in given and not execution.serial:
        raise ConfigError("[execution] max_contigs runs contigs serially; it cannot be combined with "
                          "serial = false")
    if config.serial and config.run.stop_after == "discovery":
        raise ConfigError("serial contigs are painted before the next is discovered; they cannot stop "
                          "after discovery")
    if not inputs.contigs:
        raise ConfigError("[input] contigs is required (no preset lists them)")
    if len(set(inputs.contigs)) != len(inputs.contigs):
        raise ConfigError("[input] contigs must be unique")
    if len(set(execution.process_contigs)) != len(execution.process_contigs):
        raise ConfigError("[execution] process_contigs must be unique")
    if set(execution.process_contigs) - set(inputs.contigs):
        raise ConfigError("[execution] process_contigs must be a subset of [input] contigs")
    if execution.process_contigs and config.serial:
        raise ConfigError("[execution] process_contigs (a shard) and serial cannot be combined")
    if inputs.kind == "vcf":
        if inputs.vcf is None:
            raise ConfigError("[input] kind = \"vcf\" needs [input] vcf (or --vcf)")
        stray = sorted(".".join(key) for key in given
                       if key[0] == "simulation" and _value(config, key) != _value(RunConfig(), key))
        if stray:
            raise ConfigError(f"[simulation] settings need [input] kind = \"simulation\": {', '.join(stray)}")
    else:
        if config.simulation.templates is None:
            raise ConfigError("[input] kind = \"simulation\" needs [simulation] templates (or --templates)")
        if inputs.vcf is not None:
            raise ConfigError("[input] vcf is not read by simulations (templates are built with the templates "
                              "command)")
        if (config.samples.metadata is not None or config.pedigree.eligibility.policy != "none") and \
                config.simulation.pedigree is None:
            # 2.17: an explicit pedigree fixes the sequenced samples before simulating, so its design may carry
            # metadata and an eligibility policy; the random design's samples exist only after simulating.
            raise ConfigError("simulations take [samples] metadata or a parent-eligibility policy only with an "
                              "explicit [simulation] pedigree")
    eligibility = config.pedigree.eligibility
    if eligibility.policy == "cohorts":
        if config.samples.metadata is None:
            raise ConfigError("[pedigree.eligibility] policy = \"cohorts\" needs [samples] metadata")
        if not eligibility.parent_generations:
            raise ConfigError("[pedigree.eligibility] policy = \"cohorts\" needs parent_generations")
    elif eligibility.parent_generations:
        raise ConfigError("[pedigree.eligibility] parent_generations needs policy = \"cohorts\"")
    if (eligibility.policy == "constraints") != (eligibility.constraints is not None):
        raise ConfigError("[pedigree.eligibility] policy = \"constraints\" and a constraints file go together")
    if config.reports.reference_consistency and not config.samples.reference_generation:
        raise ConfigError("[reports] reference_consistency needs [samples] reference_generation")
    if config.samples.reference_generation and config.samples.metadata is None:
        raise ConfigError("[samples] reference_generation needs [samples] metadata")
    if not config.simulation.cohorts or min(config.simulation.cohorts) < 1:
        raise ConfigError("[simulation] cohorts must be positive sizes")


def _value(config, path):
    value = config
    for part in path:
        value = getattr(value, part, None)
    return value


def default_output(config: RunConfig) -> Path:
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
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_toml_key(key)} = {_toml_value(item)}" for key, item in value.items()) + "}"
    raise TypeError(f"no TOML form for {type(value).__name__}")


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(key)


def _plain(kind: str, value):
    if kind == "path":
        return "" if value is None else str(value)
    if kind == "table":
        return dict(value)
    return value


def config_text(config: RunConfig) -> str:
    """The complete resolved configuration as TOML (loadable again by load_config)."""
    lines = [f"preset = {_toml_value(config.preset)}"]

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

    for item in schema_fields(RunConfig):
        if item.metadata["kind"] == "section":
            emit(item.name, getattr(config, item.name))
    return "\n".join(lines) + "\n"


def write_config(config: RunConfig, path) -> Path:
    """Write the resolved configuration atomically; return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(config_text(config))
    temporary.replace(path)
    return path
