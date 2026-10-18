#!/usr/bin/env python3
"""Generate docs/configuration.md from the code, so that the reference cannot drift from it.

Sources:
- config/schema.py: every TOML key with its kind, default, choices and description;
- cli.py: every command and command-line option;
- config/load.py: the consistency rules (the messages of ``validate``);
- config/presets/*.toml: the presets, verbatim;
- config/stages.py and the stage configs: the internal model constants with their defaults and descriptions
  (from each dataclass's docstring entries ``name: text`` or the comments above its fields), and which of them a
  user setting changes (found by changing each setting and comparing the stage configs);
- ENVIRONMENT below: the environment variables the package reads or sets (docs/check_docs.py checks that every
  variable the code reads directly is listed).

    python docs/generate_configuration.py           # rewrite docs/configuration.md
    python docs/generate_configuration.py --check   # exit 1 when docs/configuration.md is out of date

The source tree is the parent of this file's directory; it is put first on sys.path.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib.resources
import inspect
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import textwrap
import tokenize

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS.parent
TARGET = DOCS / "configuration.md"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
# argparse wraps usage lines to the terminal width; fix it so the output does not depend on the terminal.
os.environ["COLUMNS"] = "110"

KIND_TEXT = {
    "bool": "boolean",
    "int": "integer >= 0",
    "float": "number",
    "str": "string",
    "path": "path",
    "strings": "array of strings",
    "ints": "array of integers",
    "table": "table of strings",
    "cores": "\"auto\" or integer >= 1",
    "seed": "integer >= 0 or \"random\"",
    "memory": "\"auto\", integer bytes, or a size with a binary unit (K, M, G, T), e.g. \"200G\"",
}

# Environment variables: (name, how the package uses it). Settings never travel through the environment; these
# are facts about the environment (scheduler, scratch and cache locations) and the variables the package sets for
# itself and its child processes.
ENVIRONMENT_READ = (
    ("NUMBA_CACHE_DIR", "Base directory for compiled kernels; the package uses a subdirectory per code version "
     "(package source, Python/NumPy/Numba/llvmlite versions, code-generation settings, CPU model). Default: a "
     "node-local scratch directory per scheduler job (or run) and host."),
    ("XDG_CACHE_HOME", "Persistent kernel store `$XDG_CACHE_HOME/haplotype-reconstruction/numba` (default "
     "`~/.cache/haplotype-reconstruction/numba`); the six most recently used code versions are kept."),
    ("SLURM_JOB_ID", "Scheduler job identity: scopes the automatic kernel cache directory; recorded in batch-queue "
     "claims."),
    ("SLURM_ARRAY_TASK_ID", "Array task identity (automatic kernel cache directory)."),
    ("SLURM_TMPDIR", "Scratch directory of the automatic kernel cache inside a Slurm job."),
    ("SLURM_MEM_PER_NODE", "The job's memory: one of the bounds of the available memory the memory governor "
     "starts from."),
    ("SLURM_JOB_END_TIME", "Batch queue: the allocation end of a coordinator or helper (expired owners stop "
     "offering work; expired claims can be taken over)."),
    ("PBS_JOBID", "PBS job identity (automatic kernel cache directory); also `PBS_ARRAY_INDEX`, `PBS_ARRAYID`."),
    ("LSB_JOBID", "LSF job identity (automatic kernel cache directory); also `LSB_JOBINDEX`."),
    ("SGE_ROOT", "Marks a Grid Engine site, whose `JOB_ID` and `SGE_TASK_ID` then scope the automatic kernel "
     "cache directory."),
    ("HAPLOTYPES_TIE_CENSUS", "Diagnostic, unset in normal use: a directory into which every tie decision appends "
     "its compared values (`DIR/census.<pid>.jsonl`, one JSON line per decision) for the valley census that places "
     "the tie quanta (maths/ties.py). Results do not depend on it."),
)
ENVIRONMENT_SET = (
    ("OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, VECLIB_MAXIMUM_THREADS, NUMEXPR_NUM_THREADS",
     "Set to 1 at package import: numerical libraries never start their own thread pools; all parallelism is "
     "explicit (Numba teams and process pools)."),
    ("NUMBA_THREADING_LAYER, OMP_WAIT_POLICY", "Set to `omp` and `PASSIVE` at package import."),
    ("NUMBA_DISABLE_INTEL_SVML", "Set to 1 at package import (release 2.1): Numba's math functions use scalar libm, "
     "never Intel SVML's vector kernels, so results do not depend on the machine."),
    ("NUMBA_NUM_THREADS", "Set to the run's cores at run start, so worker processes size their Numba pools from "
     "it."),
    ("NUMBA_CACHE_DIR", "Set to the code-version cache directory at package import (inherited by workers)."),
)
# Variables read only to learn the scheduler's array/task identity (listed in the descriptions above).
ENVIRONMENT_ALSO_READ = ("PBS_ARRAY_INDEX", "PBS_ARRAYID", "LSB_JOBINDEX", "JOB_ID", "SGE_TASK_ID")


def cell(text) -> str:
    """Text for one Markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


def toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        return text if any(c in text for c in ".einf") else text + ".0"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key} = {toml_value(item)}" for key, item in value.items()) + "}"
    return repr(value)


def anchor(text: str) -> str:
    """GitHub's heading anchor."""
    slug = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return slug.replace(" ", "-")


# ----------------------------------------------------------------------------------------------- command line
def _subcommands(parser):
    action = next(item for item in parser._actions if isinstance(item, argparse._SubParsersAction))
    helps = {item.dest: item.help for item in action._choices_actions}
    return [(name, command, helps.get(name, "")) for name, command in action.choices.items()]


def _option_value(action) -> str:
    if action.nargs == 0:
        return ""
    if action.choices:
        text = "{" + ",".join(str(choice) for choice in action.choices) + "}"
    elif action.metavar:
        text = str(action.metavar)
    else:
        text = action.dest.upper()
    if action.nargs in ("+", "*"):
        text += " ..."
    return text


def _option_default(action) -> str:
    default = action.default
    if action.nargs == 0 or default is None or default is argparse.SUPPRESS or default == []:
        return ""
    if isinstance(default, (list, tuple)):
        return " ".join(str(item) for item in default)
    return str(default)


def command_line_section() -> list[str]:
    from haplotype_reconstruction.cli import build_parser
    parser = build_parser()
    lines = ["## Command-line options", "",
             "`haplotypes COMMAND --help` prints the same options. `python -m haplotype_reconstruction` is the same "
             "program. Options of `run` override the configuration file; their default is the configuration's "
             "value.", ""]
    for name, command, summary in _subcommands(parser):
        lines += [f"### haplotypes {name}", "", summary.rstrip(".") + ".", "", "```text",
                  command.format_usage().strip(), "```", "",
                  "| option | value | default | meaning |", "| --- | --- | --- | --- |"]
        for action in command._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            if action.option_strings:
                option = ", ".join(f"`{item}`" for item in action.option_strings)
            else:
                option = f"`{action.metavar or action.dest}`" + (" (optional)" if action.nargs == "?" else "")
            value = _option_value(action) if action.option_strings else ""
            lines.append(f"| {option} | {cell(f'`{value}`' if value else '')} | {cell(_option_default(action))} | "
                         f"{cell(action.help or '')} |")
        lines.append("")
    return lines


# ----------------------------------------------------------------------------------------------- TOML settings
def toml_section() -> list[str]:
    from haplotype_reconstruction.config.schema import RunConfig, reference_rows, schema_fields
    rows = reference_rows()
    docs = {}

    def collect(cls, prefix=()):
        for item in schema_fields(cls):
            if item.metadata["kind"] == "section":
                path = (*prefix, item.name)
                docs[".".join(path)] = item.metadata["doc"]
                collect(type(item.default_factory()), path)

    collect(RunConfig)
    lines = ["## TOML settings", "",
             "A configuration file holds the sections below; every key is optional. A key or section that is not "
             "listed here is an error naming the valid keys. Values are type-checked; paths are resolved against "
             "the working directory when the file is loaded; an empty path string means unset.", ""]
    top = [row for row in rows if not row["section"]]
    sections = []
    for row in rows:
        if row["section"] and row["section"] not in sections:
            sections.append(row["section"])
    for name in [""] + sections:
        selected = top if not name else [row for row in rows if row["section"] == name]
        title = "Top level" if not name else f"[{name}]"
        lines += [f"### {title}", ""]
        if name:
            lines += [docs.get(name, ""), ""]
        lines += ["| key | type | default | meaning |", "| --- | --- | --- | --- |"]
        for row in selected:
            kind = KIND_TEXT.get(row["kind"], row["kind"])
            if row["choices"]:
                kind += ": " + ", ".join(f"`{toml_value(choice)}`" for choice in row["choices"])
            default = row["default"]
            shown = f"`{toml_value(default)}`"
            if row["kind"] == "path" and default == "":
                shown += " (unset)"
            lines.append(f"| `{row['key']}` | {cell(kind)} | {cell(shown)} | {cell(row['doc'])} |")
        lines.append("")
    return lines


# ----------------------------------------------------------------------------------------- consistency rules
def _message(node) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(part.value if isinstance(part, ast.Constant) else "..." for part in node.values)
    return ast.unparse(node)


def rules_section() -> list[str]:
    from haplotype_reconstruction.config import load
    tree = ast.parse(inspect.getsource(load))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "validate")
    rules = []
    for node in ast.walk(function):
        if (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                and getattr(node.exc.func, "id", "") == "ConfigError" and node.exc.args):
            rules.append((node.lineno, _message(node.exc.args[0])))
    lines = ["## Consistency rules", "",
             "Checked when the configuration is loaded, before any input is read (the error messages of "
             "`config/load.py`). Checks that need the inputs (VCF header, workbook, contig counts) run next, "
             "still before any computation; see [usage](usage.md#checks-before-computation).", ""]
    lines += [f"- {cell(text)}" for _, text in sorted(rules)]
    return lines + [""]


# ---------------------------------------------------------------------------------------------------- presets
def presets_section() -> list[str]:
    from haplotype_reconstruction.config.load import PRESETS
    folder = importlib.resources.files("haplotype_reconstruction.config").joinpath("presets")
    lines = ["## Presets", "",
             "`preset = \"NAME\"` (or `--preset NAME`) starts from a built-in configuration; the file, command-line "
             "options and `--set` override it key by key. Presets hold no data paths.", ""]
    for name in PRESETS:
        text = folder.joinpath(f"{name}.toml").read_text().rstrip()
        lines += [f"### {name}", "", "```toml", text, "```", ""]
    return lines


# ------------------------------------------------------------------------------------ internal model constants
def _comments(source: str) -> dict[int, str]:
    """{line number: comment text} of a source snippet."""
    found = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            found[token.start[0]] = token.string.lstrip("#").strip()
    return found


def _docstring_entries(cls, names: set) -> dict[str, str]:
    """``name: text`` entries of a class docstring: at the start of a line (any indentation) or after "; ", also
    ``a, b: text``; an entry runs to the next entry, a blank line or a section line ending in a colon."""
    doc = inspect.getdoc(cls) or ""
    if not names:
        return {}
    alternatives = "|".join(sorted(map(re.escape, names), key=len, reverse=True))
    pattern = re.compile(rf"\b((?:{alternatives})(?:\s*[,/]\s*(?:{alternatives}))*)\s*:(?=\s)")
    starts = []
    for match in pattern.finditer(doc):
        before = doc[:match.start()].rstrip(" \t")
        if not before or before.endswith(("\n", ";")):
            starts.append(match)
    docs = {}
    for index, match in enumerate(starts):
        stop = starts[index + 1].start() if index + 1 < len(starts) else len(doc)
        body = doc[match.end():stop]
        body = re.split(r"\n\s*\n", body)[0]
        kept = []
        for number, line in enumerate(body.split("\n")):
            if number and line.strip().endswith(":") and not pattern.search(line):
                break                                   # a section line such as "Belief-propagation schedule:"
            kept.append(line.strip())
        entry = " ".join(" ".join(kept).split()).rstrip(";").strip()
        for name in re.split(r"[,/]", match.group(1)):
            docs.setdefault(name.strip(), entry)
    return docs


def field_docs(cls) -> dict[str, str]:
    """Descriptions of a dataclass's fields: docstring entries (``_docstring_entries``), else the comment block
    directly above the field, else its trailing comment. A comment above one field is not extended to the fields
    after it (it may or may not describe them)."""
    names = {item.name for item in dataclasses.fields(cls)}
    docs = _docstring_entries(cls, names)
    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError):
        return docs
    lines = source.splitlines()
    comments = _comments(source)
    for node in ast.parse(source).body[0].body:
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        name, number = node.target.id, node.lineno
        if docs.get(name):
            continue
        block, cursor = [], number - 1
        while cursor >= 1 and lines[cursor - 1].strip().startswith("#"):
            block.insert(0, comments.get(cursor, ""))
            cursor -= 1
        text = " ".join(block) or comments.get(number, "")
        if text:
            docs[name] = text
    return docs


def _first_sentence(cls) -> str:
    doc = " ".join((inspect.getdoc(cls) or "").split("\n\n")[0].split())
    return doc


def _leaves(value, path: tuple, out: dict, groups: list | None = None) -> None:
    for item in dataclasses.fields(value):
        entry = getattr(value, item.name)
        if dataclasses.is_dataclass(entry) and not isinstance(entry, type):
            if groups is not None:
                groups.append((path + (item.name,), entry))
            _leaves(entry, path + (item.name,), out, groups)
        else:
            out[".".join(path + (item.name,))] = entry


def _stage_values(config) -> dict:
    from haplotype_reconstruction.config import stages
    values = {}
    configs = stages.stage_configs(config)
    for item in dataclasses.fields(configs):
        root = getattr(configs, item.name)
        _leaves(root, (type(root).__name__,), values)
    simulation = stages.simulation(config)
    _leaves(simulation, (type(simulation).__name__,), values)
    return values


def _replace(config, path, key, value):
    if not path:
        return dataclasses.replace(config, **{key: value})
    head = getattr(config, path[0])
    return dataclasses.replace(config, **{path[0]: _replace(head, path[1:], key, value)})


def _probe_value(row, table: Path):
    from haplotype_reconstruction.config.schema import RunConfig
    kind, default = row["kind"], row["default"]
    if kind == "bool":
        return not default
    if kind == "str":
        others = [choice for choice in row["choices"] if choice != default]
        return others[0] if others else None
    if kind == "float":
        return float(default) / 2 if default > 0 else 0.25
    if kind == "int":
        return int(default) + 1
    if kind == "path":
        return table
    if kind == "ints":
        return (7, 8)
    if kind == "strings":
        return ("probe",)
    if kind == "seed":
        return 401 if default != 401 else 402
    return None


def user_settings_of_constants(base) -> dict[str, list[str]]:
    """{stage-config leaf: [user settings that change it]}, by changing each scientific setting in turn."""
    from haplotype_reconstruction.config.schema import reference_rows
    baseline = _stage_values(base)
    found: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory() as folder:
        table = Path(folder) / "breakpoints.tsv"
        table.write_text("chrom\tstart\tend\treason\nchrProbe\t10\t20\tprobe\n")
        for row in reference_rows():
            section = row["section"]
            if not section or section.split(".")[0] in ("run", "execution", "input", "samples", "reports",
                                                         "genetic_map"):
                continue
            value = _probe_value(row, table)
            if value is None:
                continue
            try:
                changed = _stage_values(_replace(base, section.split("."), row["key"], value))
            except Exception as error:     # a probe the stage configs reject: report, do not guess
                print(f"warning: probing [{section}] {row['key']}: {error}", file=sys.stderr)
                continue
            for leaf, entry in changed.items():
                try:
                    differs = bool(baseline.get(leaf) != entry)
                except ValueError:
                    differs = True
                if differs:
                    found.setdefault(leaf, []).append(f"[{section}] {row['key']}")
    return found


def constants_section() -> list[str]:
    from haplotype_reconstruction.config import stages
    from haplotype_reconstruction.config.schema import RunConfig
    # A simulation configuration builds every stage config, the simulation's included (the input kind only
    # resolves the "auto" quality settings, which the probes change explicitly).
    defaults = RunConfig()
    base = RunConfig(input=dataclasses.replace(defaults.input, kind="simulation", contigs=("chrProbe",)),
                     simulation=dataclasses.replace(defaults.simulation, templates=Path("/probe/templates")))
    mapped = user_settings_of_constants(base)
    configs = stages.stage_configs(base)
    roots = []
    for item in dataclasses.fields(configs):
        built = getattr(configs, item.name)
        try:
            roots.append((item.name, type(built)()))      # the dataclass defaults, not this run's values
        except TypeError:
            roots.append((item.name, built))
    roots.append(("simulation", stages.simulation(base)))
    lines = ["## Internal model constants", "",
             "Each stage has a frozen configuration whose defaults are the production values. They are model "
             "constants, not settings: no TOML key or option changes them, except the fields marked *set by* a "
             "TOML key. They enter the stage's checkpoint identity, so results always record them. The "
             "[methods](methods/README.md) explain them; a methods page cites a constant as "
             "`RootConfig.path = value`, which `docs/check_docs.py` checks against these defaults.", ""]
    for stage, root in roots:
        cls = type(root)
        module = cls.__module__.removeprefix("haplotype_reconstruction.").replace(".", "/") + ".py"
        lines += [f"### {cls.__name__}", "",
                  f"Stage `{stage}`, `{module}`. {_first_sentence(cls)}", "",
                  "| constant | default | meaning |", "| --- | --- | --- |"]
        values, groups = {}, []
        _leaves(root, (), values, groups)
        group_docs = {(): field_docs(cls)}
        for path, entry in groups:
            group_docs[path] = field_docs(type(entry))
        group_rows = {path: entry for path, entry in groups}
        emitted_groups = set()
        for leaf, value in values.items():
            parts = tuple(leaf.split("."))
            full = f"{cls.__name__}.{leaf}"
            if stage == "simulation" and (full in mapped or parts[0] in ("contigs",)):
                continue
            for depth in range(1, len(parts)):
                group = parts[:depth]
                if group in group_rows and group not in emitted_groups:
                    emitted_groups.add(group)
                    entry = group_rows[group]
                    text = group_docs.get(group[:-1], {}).get(group[-1]) or _first_sentence(type(entry))
                    lines.append(f"| **{'.'.join(group)}** ({type(entry).__name__}) |  | {cell(text)} |")
            text = group_docs.get(parts[:-1], {}).get(parts[-1], "")
            if full in mapped:
                text = f"*Set by* {', '.join(f'`{key}`' for key in mapped[full])}. {text}".strip()
            lines.append(f"| `{leaf}` | `{cell(repr(value))}` | {cell(text)} |")
        lines.append("")
    return lines


# --------------------------------------------------------------------------------------------- environment
def environment_section() -> list[str]:
    lines = ["## Environment", "",
             "No setting is read from the environment: every setting is in the configuration. The package reads "
             "these facts about its environment:", "",
             "| variable | use |", "| --- | --- |"]
    lines += [f"| `{name}` | {cell(text)} |" for name, text in ENVIRONMENT_READ]
    lines += ["", "and sets these for itself and its worker processes:", "", "| variable | value |", "| --- | --- |"]
    lines += [f"| {', '.join(f'`{name.strip()}`' for name in names.split(','))} | {cell(text)} |"
              for names, text in ENVIRONMENT_SET]
    return lines + [""]


# --------------------------------------------------------------------------------------------------- page
def render() -> str:
    lines = ["# Configuration reference", "",
             "<!-- Generated by docs/generate_configuration.py from the code: do not edit by hand. Regenerate it "
             "after changing config/schema.py, cli.py, config/load.py, a preset or a stage config; "
             "docs/check_docs.py fails while it is out of date. -->", "",
             "A run is configured by one TOML file, optionally based on a built-in preset, with command-line "
             "options and `--set SECTION.KEY=VALUE` overrides on top. Later sources win, key by key: schema "
             "defaults < preset < TOML file < command-line options < `--set`. The resolved configuration is "
             "written to `OUTPUT/config.toml` and into every run record; it loads again unchanged. Only the "
             "scientific sections (all but `[run]`, `[execution]` and `[reports]`) can change results; "
             "[usage](usage.md) explains the commands and the run directory.", "",
             "`--set` takes TOML values: `--set quality.read_calibration=false`, `--set execution.cores=32`, "
             "`--set 'input.contigs=[\"chr1\", \"chr2\"]'`; a value that is not valid TOML is read as a string.",
             "", "Contents:", ""]
    sections = [command_line_section(), toml_section(), rules_section(), presets_section(),
                constants_section(), environment_section()]
    for section in sections:
        title = section[0].removeprefix("## ")
        lines.append(f"- [{title}](#{anchor(title)})")
    lines.append("")
    for section in sections:
        lines += section
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Only check that docs/configuration.md is current.")
    parser.add_argument("--output", type=Path, default=TARGET, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current != text:
            print(f"{args.output} is out of date: run python docs/generate_configuration.py", file=sys.stderr)
            return 1
        return 0
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
