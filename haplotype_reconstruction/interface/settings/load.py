"""Loading the run configuration: preset, TOML file, command-line options and `--set` overrides, strictly.

Precedence (later wins): schema defaults < preset (interface/settings/presets/<name>.toml) < TOML file < command-line
options < `--set section.key=value`. Unknown sections and keys are errors naming the valid ones; values are
type-checked by their field kind; relative paths resolve against the working directory. The resolved configuration
is written back as TOML (`settings_text`) to OUTPUT/config.toml and into the run record. The nested simulation.reads
preset is expanded after merging these layers, with every explicit read key taking precedence over its values.
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
from .schema import NAMED_SECTIONS, Settings, schema_fields

PRESETS = ("tromau", "generic")

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
        if isinstance(value, dict) and isinstance(result.get(key), dict):
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

    if kind in ("optional_int", "optional_float"):
        if value == "":
            return None
        return _convert(kind.removeprefix("optional_"), value, where, choices, cwd)
    if kind == "floats":
        if not isinstance(value, list) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value
        ):
            fail("an array of finite numbers")
        return tuple(float(v) for v in value)
    if kind in (*NAMED_SECTIONS, "counts", "lengths"):
        if kind == "lengths" and isinstance(value, int) and not isinstance(value, bool):
            return value
        if not isinstance(value, dict) or any(not isinstance(k, str) or not k for k in value):
            fail("a table with nonempty names")
        if kind in NAMED_SECTIONS:
            return tuple((k, _build(NAMED_SECTIONS[kind], v, (where, k), cwd)) for k, v in value.items())
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in value.values()):
            fail("a table of positive integers")
        return tuple(value.items())
    if kind == "regions":
        if not isinstance(value, list):
            fail("an array of interval tables")
        regions = []
        for r in value:
            if not isinstance(r, dict) or set(r) != {"contig", "start", "end", "factor"}:
                fail("interval tables with contig, start, end, factor")
            regions.append(
                (
                    _convert("str", r["contig"], where, (), cwd),
                    _convert("int", r["start"], where, (), cwd),
                    _convert("int", r["end"], where, (), cwd),
                    _convert("float", r["factor"], where, (), cwd),
                )
            )
        return tuple(regions)
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


def build_settings(mapping: dict, *, cwd: str | Path | None = None) -> Settings:
    """A typed Settings from a merged mapping (paths resolved against ``cwd``); no cross-key validation."""
    simulation = mapping.get("simulation", {})
    artefacts = simulation.get("artefacts", {}) if isinstance(simulation, dict) else {}
    if isinstance(artefacts, dict) and artefacts.get("preset") == "reference_blocks":
        mapping = _merge({"simulation": {"artefacts": {"enabled": True}}}, mapping)
    if isinstance(simulation, dict):
        genome = simulation.get("genome", {})
        inputs = mapping.get("input", {})
        if (
            isinstance(genome, dict)
            and genome.get("contigs")
            and isinstance(inputs, dict)
            and not inputs.get("contigs")
        ):
            mapping = _merge(mapping, {"input": {"contigs": genome["contigs"]}})
    reads = simulation.get("reads", {}) if isinstance(simulation, dict) else {}
    if isinstance(reads, dict) and reads.get("preset") in ("realistic", "short_read_cohort"):
        text = (
            resources.files("haplotype_reconstruction.interface.settings")
            .joinpath("presets", "realistic_reads.toml" if reads["preset"] == "realistic" else "short_read_cohort.toml")
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
        merged = _merge(merged, {"input": {"contigs": list(fallback_contigs)}})  # never mutates a caller's layer
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
        if config.simulation.source.kind == "file" and config.simulation.source.file is None:
            raise SettingsError(
                '[input] kind = "simulation" needs [simulation.source] kind = "file", file = DIR or kind = "coalescent"'
            )
        if inputs.vcf is not None:
            raise SettingsError("[input] vcf is not read by simulations; configure simulation.source")
    if config.simulation.pedigree is None and (not config.simulation.cohorts or min(config.simulation.cohorts) < 1):
        raise SettingsError("[simulation] cohorts must be positive sizes")
    if inputs.kind == "simulation":
        _validate_simulation(config)
    for name in ("site_depth_cv", "count_dispersion", "region_depth_cv"):
        value = getattr(config.simulation.reads, name)
        if not math.isfinite(value) or value < 0:
            raise SettingsError(f"[simulation.reads] {name} must be finite and nonnegative")


def _validate_simulation(config: Settings) -> None:
    """Structural generating constraints without importing numerical libraries.

    Input-dependent file panels/pedigrees are checked by the source preflight;
    typed generating records recheck their own constraints at the stage boundary.
    No seed is realised by either validation pass.
    """
    sim = config.simulation
    source, sites, genome, reads, artefacts = sim.source, sim.sites, sim.genome, sim.reads, sim.artefacts
    stocks, populations = dict(sim.stocks), dict(source.populations)

    def require(condition, message):
        if not condition:
            raise SettingsError(message)

    require(0 <= sites.spurious_fraction < 1, "spurious_fraction must be in [0, 1)")
    require(sim.depth > 0, "simulation depth must be positive")
    require(sim.generating_rate_cm_per_mb >= 0, "generating_rate_cm_per_mb must be nonnegative")
    require(0 <= sim.backcross_fraction <= 1, "backcross_fraction must be between zero and one")
    require(sim.backcross_start_cohort >= 2, "backcrosses need a generated individual and one actual parent")
    require(0 < sim.observed_fraction <= 1, "observed_fraction must be in (0, 1]")
    require(
        len(set(sim.observe_cohorts)) == len(sim.observe_cohorts) and all(c >= 1 for c in sim.observe_cohorts),
        "observe_cohorts must be unique positive cohort numbers",
    )
    if sim.pedigree is not None:
        require(not sim.founders, "explicit pedigree sets its own founder stocks; omit founders")
        require(
            not sim.backcross_fraction and not sim.observe_cohorts and sim.observed_fraction == 1,
            "an explicit pedigree sets matings and observation itself; leave observation options at defaults",
        )
    else:
        require(
            sim.cohorts and all(n >= 2 for n in sim.cohorts), "each generated cohort must have at least two individuals"
        )
        require(
            not sim.observe_cohorts or max(sim.observe_cohorts) <= len(sim.cohorts),
            "an observed cohort was not generated",
        )
        eligible = sum(n for c, n in enumerate(sim.cohorts, 1) if not sim.observe_cohorts or c in sim.observe_cohorts)
        require(
            int(eligible * sim.observed_fraction) >= 3, "observation selection retains fewer than three individuals"
        )

    require(
        not genome.contigs or genome.contigs == config.input.contigs,
        "simulation.genome.contigs and input.contigs must have the same order",
    )
    if source.kind == "file":
        require(
            not populations and not source.reference_population, "file source cannot configure coalescent populations"
        )
        require(
            not genome.contigs and not genome.length_bp and sites == type(sites)(),
            "file sources fix their sites and extent; genome and sites settings do not apply",
        )
    else:
        require(
            source.file is None and source.unknown == "refuse", "coalescent source cannot configure file or unknown"
        )
        require(bool(populations), "coalescent populations must have unique nonempty names")
        require(
            not source.reference_population or source.reference_population in populations,
            "unknown source reference_population",
        )
        require(sum(not p.parent for p in populations.values()) == 1, "population tree requires exactly one root")
        for name, pop in populations.items():
            require(pop.size > 0, "population sizes must be positive diploid individuals")
            if pop.parent:
                require(pop.parent in populations, "unknown source population parent")
                require(
                    pop.split_generations is not None and pop.split_generations >= 0,
                    "each non-root population needs finite nonnegative split_generations",
                )
                parent = populations[pop.parent]
                if parent.parent:
                    require(
                        parent.split_generations is not None and pop.split_generations <= parent.split_generations,
                        "child split must not exceed parent split",
                    )
            else:
                require(pop.split_generations is None, "the root population has no split_generations")
            visited = set()
            current = name
            while current:
                require(current not in visited, "population tree contains a cycle")
                require(current in populations, "unknown source population parent")
                visited.add(current)
                current = populations[current].parent
        require(
            bool(stocks) and all(stock.population in populations for stock in stocks.values()),
            "coalescent stocks must name known source populations",
        )
        if isinstance(genome.length_bp, int):
            lengths = [genome.length_bp]
        else:
            lengths_by_name = dict(genome.length_bp)
            require(
                set(lengths_by_name) == set(config.input.contigs),
                "genome length_bp must name exactly the configured contigs",
            )
            lengths = list(lengths_by_name.values())
        require(all(length >= 1 for length in lengths), "genome length_bp must be a positive integer")
        require(
            sum(v is not None for v in (sites.count, sites.per_mb, sites.positions)) == 1,
            "exactly one of count, per_mb or positions is required",
        )
        if sites.count is not None:
            require(
                1 <= sites.count <= min(lengths), "count must be positive and cannot exceed integer genome positions"
            )
        if sites.per_mb is not None:
            require(0 < sites.per_mb < 1e6, "per_mb target must be positive and below integer genome capacity")
        require(
            len(sites.sigmas) == 5 and all(v >= 0 for v in sites.sigmas),
            "profile sigmas must be five finite nonnegative log-intensity standard deviations",
        )
        require(0 <= sites.masked_fraction < 1, "masked_fraction must be in [0, 1)")
        require(sites.masked_mean_bp > 0, "masked_mean_bp must be positive")
        if sites.positions is not None:
            require(sites.profile == "none", "positions requires profile = none")
            try:
                positions = [int(v) for v in sites.positions.read_text().split()]
            except (OSError, ValueError) as error:
                raise SettingsError(f"cannot read integer site positions: {error}") from error
            require(
                all(1 <= value <= min(lengths) for value in positions),
                "site positions must lie in the generated genome extent",
            )
            require(
                all(a < b for a, b in zip(positions, positions[1:], strict=False)),
                "positions must be sorted unique integer bp values",
            )

    for name, stock in stocks.items():
        require(
            stock.founders >= 2 and stock.size >= 2, f"stock {name}: founders and size must be at least two individuals"
        )
    if sim.pedigree is None and stocks:
        requests = dict(sim.founders)
        require(
            bool(requests) and set(requests) <= set(stocks),
            "random cohorts require founders = {STOCK = positive integer} for known stocks",
        )
        require(sum(requests.values()) >= 2, "the cross requires at least two founder individuals")
        for name, count in requests.items():
            stock = stocks[name]
            available = stock.size if stock.generations else stock.founders
            require(count <= available, f"stock {name}: requested {count} individuals, available {available}")
    elif not stocks:
        require(not sim.founders, "founders requires configured stocks")

    require(0 < reads.error_rate < 0.5, "generating error_rate must be between zero and 0.5")
    require(0 < reads.heterozygote_alt_probability < 1, "heterozygote_alt_probability must be in (0, 1)")
    require(0 <= reads.dropout_fraction < 1, "dropout_fraction must be in [0, 1)")
    for name in ("depth_cv", "site_depth_cv", "count_dispersion"):
        require(getattr(reads, name) >= 0, f"[simulation.reads] {name} must be finite and nonnegative")
    if reads.read_length_bp:
        require(
            reads.fragment_mean_bp > 0 and reads.fragment_sd_bp >= 0,
            "fragment length mean must be positive and its sd nonnegative",
        )
    for name in (
        "hard_sites",
        "hard_miss",
        "stretch_cover",
        "stretch_miss",
        "miss",
        "call_min_alt_fraction",
        "systematic_sites",
        "systematic_lo",
        "systematic_hi",
    ):
        require(0 <= getattr(artefacts, name) <= 1, f"artefact {name} must be between zero and one")
    require(0 <= reads.site_loss_fraction < 1, "site_loss_fraction must be in [0, 1)")
    require(0 <= reads.site_loss_fine_fraction < 1, "site_loss_fine_fraction must be in [0, 1)")
    require(not reads.site_loss_fine_fraction or reads.site_loss_fine_window_bp > 0,
            "positive site_loss_fine_fraction requires positive site_loss_fine_window_bp")
    require(reads.region_depth_cv >= 0, "region_depth_cv must be nonnegative")
    require(not reads.region_depth_cv or reads.region_depth_window_bp > 0,
            "positive region_depth_cv requires positive region_depth_window_bp")
    require(0 <= artefacts.balance_dispersion < 1, "artefact balance_dispersion must be in [0, 1)")
    require(artefacts.miss_sample_cv >= 0, "artefact miss_sample_cv must be nonnegative")
    require(
        artefacts.block_band > 0 and artefacts.stretch_sites >= 1,
        "artefact block_band must be positive and stretch_sites at least one",
    )
    if artefacts.stretch_cover:
        require(
            artefacts.stretch_cover < 1
            and artefacts.stretch_sites * (1 - artefacts.stretch_cover) / artefacts.stretch_cover >= 1,
            "artefact stretch_cover and stretch_sites require a geometric gap mean >= 1",
        )
    require(artefacts.systematic_lo <= artefacts.systematic_hi, "artefact systematic_lo must not exceed systematic_hi")
    regions = sorted(artefacts.regions)
    for contig, start, end, factor in regions:
        require(
            bool(contig) and start < end and factor >= 0,
            "artefact region needs a contig, start < end and nonnegative factor",
        )
    require(
        all(a[0] != b[0] or a[2] <= b[1] for a, b in zip(regions, regions[1:], strict=False)),
        "artefact regions on the same contig must not overlap",
    )


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
    if value is None:
        return '""'
    if isinstance(value, dict):
        return "{ " + ", ".join(json.dumps(str(k)) + " = " + _toml_value(v) for k, v in value.items()) + " }"
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
                kind = item.metadata["kind"]
                if kind in NAMED_SECTIONS:
                    from .schema import to_dict

                    value = {k: to_dict(v) for k, v in value}
                elif kind == "counts" or (kind == "lengths" and isinstance(value, tuple)):
                    value = dict(value)
                elif kind == "regions":
                    value = [dict(zip(("contig", "start", "end", "factor"), r, strict=True)) for r in value]
                lines.append(f"{item.name} = {_toml_value(_plain(kind, value))}")
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
