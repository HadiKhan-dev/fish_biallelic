#!/usr/bin/env python3
"""Check the documentation against the code it describes (STYLE_40 §9).

Page checks: local links and anchors; TOML blocks and `[section] key = value`
spans against the settings schema; command lines and options against the CLI;
constants cited as `Owner.field = value` against the parameter modules; no
site-specific paths; no historical phrasing outside the changelog and the
migration guide.

Whole-set checks: every generated reference equals what `docs/generate_docs.py`
writes now; every environment variable the package reads is documented, and
only the runtime (`engine/`) touches the process environment; every carried
semantic unit of `docs/methods/coverage.json` has exactly one methods section
with its anchor, implementation links, named parameters and cost; every
documentation link in a package docstring resolves; every file the writers
publish has an output-dictionary entry; the Snakemake example's commands exist,
and its integration receipt is current and passed.

Warnings fail the build. `--page` checks only the named pages. `--draft` lists
the release-evidence gates (the workflow receipt) as pending instead of failing
on them; a release build never uses it.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import json
import math
import re
import shlex
import sys
import tempfile
from collections import Counter
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
DOCS = SOURCE / "docs"
PACKAGE = SOURCE / "haplotype_reconstruction"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))

HISTORY_PAGES = ("CHANGELOG.md",)
MIGRATION_PAGES = ("docs/user/migration.md",)
HISTORY_PHRASES = re.compile(
    r"\b(previously|no longer|as before|used to|formerly|restores the earlier|was removed|were removed|"
    r"has been removed|now replaced|earlier (?:release|version)s?|in (?:release |version )?3\.\d+)\b",
    re.IGNORECASE,
)
# Methods pages describe the current model only (WP_IN4 §3): any release number there is history.
METHODS_VERSION = re.compile(r"\b(?:release|version) \d+(?:\.\d+)*\b|\b[34]\.\d+(?:\.\d+)?(?= (?:did|used|had|kept))")
SITE_PATH = re.compile(r"/(?:rds|home)/|\b(?:CSD3|Cambridge)\b")
TODO = re.compile(r"TODO\(([^)]*)\)")
CODE_SPAN = re.compile(r"(`+)(.+?)\1")
LINK = re.compile(r"!?\[(?:[^\]\[]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)")
EXPLICIT_ANCHOR = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"")
DIRECTIVE = re.compile(r"<!--\s*check_docs:\s*(\w+)\s*-->")
TOML_KEY_SPAN = re.compile(r"^\[([a-z_][\w.]*)\](?:\s+([a-z_]\w*)(?:\s*=\s*(.+))?)?$")
CONSTANT_SPAN = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?:\s*=\s*(.+))?$")
MODULE_FILE_SPAN = re.compile(r"^(?:haplotype_reconstruction/)?([\w/]+\.py)$")
DOCS_REFERENCE = re.compile(r"\bdocs/([\w/.-]+?\.md)(?:#([\w.-]+))?")
OUTPUT_NAME = re.compile(r"\.(?:csv|tsv|json|png|tsv\.gz|npz|vcf\.gz|bcf|csi|tbi)$")

# Package subtrees allowed to touch the process environment: the runtime only (ARCHITECTURE_40 §1, WP_IN4 §4.6).
ENVIRONMENT_OWNERS = ("engine",)
ENVIRONMENT_ATTRIBUTES = frozenset({"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"})

# Writers and the modules whose published file names the output dictionary must cover.
OUTPUT_WRITERS = ("outputs", "evaluation/evaluate.py", "interface/example.py", "interface/templates/builder.py")
OUTPUT_DICTIONARY = ("docs/user/outputs.md", "docs/reference/output-columns.md")
# Variables holding a contig name or its percent-encoded file stem in writers' f-strings.
CONTIG_NAMES = frozenset({"contig", "stem"})
# Names built from another variable: (writer module, name pattern) -> where the concrete names come from. A pattern
# whose source changes makes its entry stale, which is an error: the table is maintained with the writers.
DYNAMIC_OUTPUTS = {
    ("outputs/reports/painting.py", "*.tsv"): ("constant", "outputs/reports/painting.py", "TABLES", "{}.tsv"),
    ("outputs/reports/pedigree.py", "*.csv"): ("keys", "outputs/reports/pedigree.py", "COLUMNS", "{}.csv"),
    ("outputs/reports/quality.py", "CONTIG.*.tsv"): ("names", ("CONTIG.samples.tsv", "CONTIG.segments.tsv")),
    ("outputs/export/run.py", "CONTIG.component*.founders.tsv.gz"): (
        "names",
        ("CONTIG.componentNNNN.founders.tsv.gz",),
    ),
    ("interface/example.py", "*.npz"): ("names", ("CONTIG.npz",)),
}
# Export file names are concatenated from a stem, a role and a format suffix (outputs/export/{run,vcf}.py).
EXPORT_NAMES = {
    ".founders": ("CONTIG.founders.vcf.gz", "CONTIG.founders.bcf"),
    ".samples": ("CONTIG.samples.vcf.gz", "CONTIG.samples.bcf"),
    ".csi": ("CONTIG.founders.vcf.gz.csi",),
    ".tbi": ("CONTIG.founders.vcf.gz.tbi",),
}


# Evaluation rows are dictionaries built in these modules; their literal keys and named key tuples must be defined in
# the output dictionary's D7 section. Working keys that are never published are listed with the reason.
EVALUATION_MODULES = ("evaluate.py", "founders.py", "identity.py", "metrics.py")
EVALUATION_KEY_TUPLES = (
    "PHASE_COLUMNS",
    "SAMPLE_COLUMNS",
    "_SUMMED",
    "REPORTED_INTERVAL_COLUMNS",
    "PARENT_STATES",
    "PARENT_GROUPS",
    "STRATA",
)
EVALUATION_INTERNAL_KEYS = {
    "haplotypes": "truth grid argument",
    "positions": "truth grid argument",
    "scored": "span accumulator",
    "released": "span accumulator",
    "phase_set_spans_bp": "popped and replaced by its N50 before writing",
    "released_phase_set_spans_bp": "popped and replaced by its N50 before writing",
}


def evaluation_keys(package: Path = PACKAGE) -> dict[str, str]:
    """Literal keys of evaluation row dictionaries and of their named key tuples, with a source location."""
    keys: dict[str, str] = {}
    for name in EVALUATION_MODULES:
        path = package / "evaluation" / name
        for node in ast.walk(ast.parse(path.read_text())):
            found: list[str] = []
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
                found = [keyword.arg for keyword in node.keywords if keyword.arg]
            elif isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                        found += [target.slice.value] if isinstance(target.slice.value, str) else []
                    if isinstance(target, ast.Name) and target.id in EVALUATION_KEY_TUPLES:
                        found += [item.value for item in ast.walk(node.value) if isinstance(item, ast.Constant)]
            for key in found:
                keys.setdefault(key, f"evaluation/{name}:{node.lineno}")
    return {key: where for key, where in keys.items() if key not in EVALUATION_INTERNAL_KEYS}


def undefined_evaluation_keys(dictionary: str, package: Path = PACKAGE) -> list[tuple[str, str]]:
    """Evaluation keys absent from the D7 section's code spans (a family member matches by suffix)."""
    start = dictionary.find("## D7")
    section = dictionary[start : dictionary.find("\n## ", start + 1)] if start >= 0 else ""
    tokens = {span[1] for span in CODE_SPAN.findall(section)}
    return [
        (key, where)
        for key, where in sorted(evaluation_keys(package).items())
        if not any(token == key or token.endswith("_" + key) for token in tokens)
    ]


@dataclasses.dataclass
class Page:
    path: Path
    name: str  # path relative to SOURCE
    lines: list[str]


def slug(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)  # inline HTML
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links keep their text
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


def markdown_anchors(text: str) -> set[str]:
    """Explicit `<a id>` anchors and GitHub-style heading slugs, ignoring fenced code."""
    found, counts = set(), Counter()
    fence = None
    for line in text.splitlines():
        stripped = line.strip()
        if fence:
            if stripped.startswith(fence):
                fence = None
            continue
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            continue
        found.update(EXPLICIT_ANCHOR.findall(line))
        match = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if match:
            base = slug(match.group(1))
            found.add(base if counts[base] == 0 else f"{base}-{counts[base]}")
            counts[base] += 1
    return found


def environment_access(package: Path = PACKAGE, owners: tuple[str, ...] = ENVIRONMENT_OWNERS) -> list[str]:
    """Every use of the process environment outside the runtime subtrees, as `path:line: name`."""
    found = []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        if relative.parts and relative.parts[0] in owners:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            name = None
            if isinstance(node, ast.Attribute) and node.attr in ENVIRONMENT_ATTRIBUTES:
                if isinstance(node.value, ast.Name) and node.value.id == "os":
                    name = f"os.{node.attr}"
            elif isinstance(node, ast.ImportFrom) and node.module == "os":
                names = [alias.name for alias in node.names if alias.name in ENVIRONMENT_ATTRIBUTES]
                name = ", ".join(f"os.{item}" for item in names) or None
            if name:
                found.append(f"{relative.as_posix()}:{node.lineno}: {name}")
    return found


def module_constant(path: Path, name: str):
    for node in ast.parse(path.read_text()).body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(target, ast.Name) and target.id == name for target in targets) and node.value is not None:
            return node.value
    raise KeyError(f"{path.relative_to(SOURCE)} defines no {name}")


def name_patterns(path: Path) -> list[tuple[int, str]]:
    """File-name patterns written by one module: string literals and f-strings with an output suffix."""
    found = []
    tree = ast.parse(path.read_text())
    parts = {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for part in node.values}
    for node in ast.walk(tree):
        if id(node) in parts:
            continue  # a literal piece of an f-string: the f-string itself is the name
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(
                part.value
                if isinstance(part, ast.Constant)
                else "CONTIG"
                if isinstance(part.value, ast.Name) and part.value.id in CONTIG_NAMES
                else "*"
                for part in node.values
            )
        else:
            continue
        if text.endswith(".tmp") or "/" in text or " " in text or not OUTPUT_NAME.search(text):
            continue
        if text.startswith("*") or not text.startswith("."):
            found.append((node.lineno, text))
        else:
            found.append((node.lineno, "*" + text))
    return found


def published_names(package: Path = PACKAGE) -> tuple[dict[str, str], list[str]]:
    """Concrete published file names (wildcards as CONTIG) mapped to their writer; and resolution problems."""
    names: dict[str, str] = {}
    problems: list[str] = []
    used = set()
    paths = []
    for entry in OUTPUT_WRITERS:
        target = package / entry
        if not target.exists():
            problems.append(f"{entry}: OUTPUT_WRITERS names a writer that does not exist")
            continue
        paths += sorted(target.rglob("*.py")) if target.is_dir() else [target]
    for path in paths:
        relative = path.relative_to(package).as_posix()
        for line, pattern in name_patterns(path):
            if pattern.startswith(".") or pattern in ("*.csi", "*.tbi", "*.bcf", "*.vcf.gz"):
                continue  # format suffixes: EXPORT_NAMES resolves the composed names
            key = (relative, pattern)
            rest = pattern.removeprefix("*")
            if key in DYNAMIC_OUTPUTS:
                used.add(key)
                rule = DYNAMIC_OUTPUTS[key]
                if rule[0] == "names":
                    concrete = list(rule[1])
                else:
                    node = module_constant(package / rule[1], rule[2])
                    values = [item.value for item in node.keys] if rule[0] == "keys" else ast.literal_eval(node)
                    concrete = [rule[3].format(value) for value in values]
                for name in concrete:
                    names[name] = f"{relative}:{line}"
            elif "*" in rest or (pattern.startswith("*") and OUTPUT_NAME.fullmatch(rest)):
                problems.append(f"{relative}:{line}: dynamic output name {pattern!r} has no DYNAMIC_OUTPUTS entry")
            else:
                names[pattern.replace("*", "CONTIG", 1)] = f"{relative}:{line}"
    for key in DYNAMIC_OUTPUTS:
        if key not in used:
            problems.append(f"{key[0]}: DYNAMIC_OUTPUTS entry {key[1]!r} no longer matches the writer")
    export = (package / "outputs/export/run.py").read_text() + (package / "outputs/export/vcf.py").read_text()
    for stem, concrete in EXPORT_NAMES.items():
        if f'"{stem}' not in export:
            problems.append(f"outputs/export: EXPORT_NAMES entry {stem!r} no longer matches the writer")
        for name in concrete:
            names[name] = "outputs/export"
    return names, problems


def documented_output(name: str, text: str) -> bool:
    """A published name is documented when it appears literally, with CONTIG for a contig stem."""
    if name in text:
        return True
    bare = name.removeprefix("CONTIG")
    return bare != name and re.search(r"(?:CONTIG|`)" + re.escape(bare) + r"\b", text) is not None


class Checker:
    def __init__(self, draft: bool = False):
        self.draft = draft
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.todos: list[str] = []
        self.pending: list[str] = []
        self._anchors: dict[Path, set[str]] = {}
        self._load_code()

    # ------------------------------------------------------------------------------------------- code facts
    def _load_code(self):
        import generate_references

        from haplotype_reconstruction.interface.cli import build_parser
        from haplotype_reconstruction.interface.settings import load, schema

        self.load, self.schema = load, schema
        parser = build_parser()
        sub = next(item for item in parser._actions if isinstance(item, argparse._SubParsersAction))
        self.commands = {
            name: {flag: action for action in command._actions for flag in action.option_strings}
            for name, command in sub.choices.items()
        }
        self.all_flags = {flag for options in self.commands.values() for flag in options}
        self.evaluation_stages = evaluation_stages()
        rows = schema.reference_rows()
        self.keys = {(row["section"], row["key"]): row for row in rows}
        self.sections = {row["section"] for row in rows if row["section"]}
        self.parameter_rows = generate_references.parameter_catalog()
        self.roots = {row["owner"]: row["module"] for row in self.parameter_rows if row["owner"]}
        self.parameter_names = {row["name"] for row in self.parameter_rows} | {
            row["owner"] for row in self.parameter_rows
        }
        self.parameter_names |= {key for _, key in self.keys} | {section.split(".")[-1] for section in self.sections}

    # ------------------------------------------------------------------------------------------------ helpers
    def error(self, page: Page, number: int, message: str):
        self.errors.append(f"{page.name}:{number}: {message}")

    def warn(self, page: Page, number: int, message: str):
        self.warnings.append(f"{page.name}:{number}: {message}")

    def anchors(self, path: Path) -> set[str]:
        if path not in self._anchors:
            self._anchors[path] = markdown_anchors(path.read_text())
        return self._anchors[path]

    # --------------------------------------------------------------------------------------------- checks
    def check_link(self, page: Page, number: int, target: str):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target):  # http:, https:, mailto:
            return
        path_part, _, fragment = target.partition("#")
        destination = page.path if not path_part else (page.path.parent / path_part).resolve()
        if path_part and not destination.exists():
            self.error(page, number, f"link target does not exist: {target}")
            return
        if fragment:
            if destination.is_dir() or destination.suffix != ".md":
                self.error(page, number, f"anchor on a non-Markdown target: {target}")
            elif fragment not in self.anchors(destination):
                self.error(page, number, f"anchor #{fragment} not found in {path_part or page.name}")

    def check_toml_mapping(self, page: Page, number: int, mapping: dict, where: str):
        try:
            self.load.build_settings(mapping, cwd=tempfile.gettempdir())
        except Exception as error:
            self.error(page, number, f"{where}: {error}")
            return False
        return True

    def check_key_span(self, page: Page, number: int, section: str, key: str | None, value: str | None):
        if key is None:
            if section not in self.sections:
                self.error(page, number, f"unknown configuration section [{section}]")
            return
        if (section, key) not in self.keys:
            self.error(page, number, f"unknown setting [{section}] {key}")
            return
        if value is not None and not self.placeholder(value.strip().strip('"')):
            mapping: dict = {}
            cursor = mapping
            for part in section.split("."):
                cursor = cursor.setdefault(part, {})
            try:
                import tomllib

                cursor[key] = tomllib.loads(f"value = {value}")["value"]
            except Exception:
                cursor[key] = value
            self.check_toml_mapping(page, number, mapping, f"[{section}] {key} = {value}")

    def check_override(self, page: Page, number: int, text: str):
        try:
            path, value = self.load.parse_override(text)
        except Exception as error:
            self.error(page, number, f"--set {text}: {error}")
            return
        mapping: dict = {}
        cursor = mapping
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = value
        if isinstance(value, str) and (value.isupper() or "..." in value):
            section, key = ".".join(path[:-1]), path[-1]
            if (section, key) not in self.keys and key.upper() != key:
                self.error(page, number, f"--set {text}: unknown setting")
            return
        self.check_toml_mapping(page, number, mapping, f"--set {text}")

    def check_toml_block(self, page: Page, number: int, text: str, directive: str | None):
        import tomllib

        try:
            mapping = tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            self.error(page, number, f"TOML block does not parse: {error}")
            return
        if not self.check_toml_mapping(page, number, mapping, "TOML block"):
            return
        if directive == "config":
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "config.toml"
                path.write_text(text)
                try:
                    self.load.load_settings(path, cwd=folder)
                except Exception as error:
                    self.error(page, number, f"TOML block is not a complete configuration: {error}")
        elif directive == "defaults":
            config = self.load.build_settings(mapping, cwd=tempfile.gettempdir())
            defaults = self.schema.Settings()

            def walk(node, prefix):
                for key, value in node.items():
                    path = (*prefix, key)
                    if isinstance(value, dict) and (".".join(path) in self.sections):
                        walk(value, path)
                        continue
                    stated, default = config, defaults
                    for part in path:
                        stated, default = getattr(stated, part), getattr(default, part)
                    if stated != default:
                        self.error(
                            page,
                            number,
                            f"{'.'.join(path)} = {value!r} is not the default ({self.schema.to_dict(default)!r})",
                        )

            walk(mapping, ())

    def check_command(self, page: Page, number: int, line: str):
        text = line.strip().removeprefix("$ ")
        try:
            tokens = shlex.split(text, comments=True)
        except ValueError:
            tokens = text.split()
        tokens = [token.strip("[]") for token in tokens]
        if tokens[:1] == ["haplotypes"]:
            rest = tokens[1:]
        elif tokens[:3] == ["python", "-m", "haplotype_reconstruction"]:
            rest = tokens[3:]
        else:
            return
        if not rest or rest[0].startswith("-"):
            return
        command, arguments = rest[0], rest[1:]
        if self.placeholder(command):
            return
        if command not in self.commands:
            self.error(page, number, f"unknown command: haplotypes {command}")
            return
        options = self.commands[command]
        index = 0
        while index < len(arguments):
            token = arguments[index]
            index += 1
            if not token.startswith("--"):
                continue
            flag, _, inline = token.partition("=")
            action = options.get(flag)
            if action is None:
                self.error(page, number, f"haplotypes {command} has no option {flag}")
                continue
            values = []
            if inline:
                values = [inline]
            elif action.nargs != 0:
                while index < len(arguments) and not arguments[index].startswith("-"):
                    values.append(arguments[index])
                    index += 1
                    if action.nargs is None:
                        break
            literal = [value for value in values if not self.placeholder(value)]
            if action.choices:
                for value in literal:
                    for alternative in value.split("|"):
                        if alternative not in action.choices:
                            choices = ", ".join(map(str, action.choices))
                            self.error(
                                page, number, f"haplotypes {command} {flag}: {alternative!r} is not one of {choices}"
                            )
            if command == "evaluate" and flag == "--stages":
                for value in literal:
                    if value not in self.evaluation_stages:
                        self.error(page, number, f"evaluate --stages: unknown stage {value!r}")
            if flag == "--set":
                for value in values:
                    self.check_override(page, number, value)

    @staticmethod
    def placeholder(value: str) -> bool:
        return (
            value.isupper()
            or "<" in value
            or "..." in value
            or value.startswith("$")
            or re.fullmatch(r"[A-Z][A-Z0-9_]*(\.[A-Z]+)?", value) is not None
        )

    def resolve_constant(self, dotted: str):
        """Resolve only parameter declarations, never arbitrary science imports."""
        head, _, tail = dotted.partition(".")
        if head in self.roots:
            module = importlib.import_module("haplotype_reconstruction." + self.roots[head])
            value = getattr(module, head)()
            for part in tail.split("."):
                if not hasattr(value, part):
                    return False, None
                value = getattr(value, part)
            return True, value
        for row in self.parameter_rows:
            qualified = ".".join(part for part in (row["module"], row["owner"], row["name"]) if part)
            if dotted == qualified:
                value = importlib.import_module("haplotype_reconstruction." + row["module"])
                if row["owner"]:
                    value = getattr(value, row["owner"])()
                return True, getattr(value, row["name"])
        return False, None

    @staticmethod
    def same(stated, actual) -> bool:
        if hasattr(actual, "tolist"):
            actual = actual.tolist()
        if isinstance(stated, bool) or isinstance(actual, bool):
            return type(stated) is type(actual) and stated == actual
        if isinstance(stated, (int, float)) and isinstance(actual, (int, float)):
            return math.isclose(stated, actual, rel_tol=1e-12, abs_tol=0.0) or stated == actual
        if isinstance(stated, (tuple, list)) and isinstance(actual, (tuple, list)):
            return len(stated) == len(actual) and all(Checker.same(a, b) for a, b in zip(stated, actual, strict=True))
        return stated == actual

    def check_span(self, page: Page, number: int, span: str, checks: bool):
        span = span.strip()
        if not checks:
            return
        migration = page.name in MIGRATION_PAGES
        if span.startswith("haplotypes ") or span.startswith("python -m haplotype_reconstruction "):
            self.check_command(page, number, span)
            return
        match = TOML_KEY_SPAN.match(span)
        if match:
            self.check_key_span(page, number, match.group(1), match.group(2), match.group(3))
            return
        if span.startswith("--set ") or span.startswith("--set="):
            try:
                tokens = shlex.split(span)
            except ValueError:
                tokens = span.split()
            for index, token in enumerate(tokens):
                if token == "--set" and index + 1 < len(tokens):
                    self.check_override(page, number, tokens[index + 1])
                elif token.startswith("--set="):
                    self.check_override(page, number, token.removeprefix("--set="))
            return
        flag = re.fullmatch(r"(--[a-z][\w-]*)(?:[ =].*)?", span)
        if flag:
            if migration and flag.group(1) in {"--process-contigs", "--batch-queue"}:
                return
            if flag.group(1) not in self.all_flags:
                self.error(page, number, f"no command has the option {flag.group(1)}")
            return
        match = MODULE_FILE_SPAN.match(span)
        if match and not migration:
            relative = match.group(1)
            if not ((PACKAGE / relative).exists() or (SOURCE / relative).exists()):
                self.error(page, number, f"no such file: {span} (module paths are relative to the package)")
            return
        match = CONSTANT_SPAN.match(span)
        if match:
            dotted, value_text = match.group(1), match.group(2)
            head = dotted.split(".")[0]
            if any(part.isupper() for part in dotted.split(".")[:-1]) or head == "RootConfig":
                return  # placeholders such as SECTION.KEY
            if (
                value_text is not None
                and re.fullmatch(r"[A-Za-z_]\w*", value_text.strip())
                and value_text.strip() not in ("True", "False", "None")
            ):
                value_text = None  # a placeholder value: check only that the name exists
            if value_text is None and head not in self.roots:
                return  # module paths, attribute names in prose: not checked
            if head not in self.roots:
                section, _, key = dotted.rpartition(".")
                if (section, key) in self.keys:
                    self.check_key_span(page, number, section, key, value_text)
                    return
            found, actual = self.resolve_constant(dotted)
            if not found:
                if head in self.roots or value_text is not None:
                    self.error(page, number, f"no such constant: {dotted}")
                return
            if value_text is None:
                return
            try:
                stated = ast.literal_eval(value_text.strip())
            except (ValueError, SyntaxError):
                self.error(page, number, f"{dotted}: value {value_text!r} is not a Python literal")
                return
            if dataclasses.is_dataclass(actual):
                self.error(page, number, f"{dotted} is a group, not a constant")
            elif not self.same(stated, actual):
                self.error(page, number, f"{dotted} = {value_text} but the code's default is {actual!r}")
            elif page.name.startswith("docs/methods/"):
                self.warn(page, number, f"{dotted}: methods prose names parameters; values belong to the reference")

    # ---------------------------------------------------------------------------------------------- pages
    def check_page(self, page: Page):
        history = page.name in HISTORY_PAGES
        migration = page.name in MIGRATION_PAGES
        methods = page.name.startswith("docs/methods/")
        fence = None
        language = ""
        block: list[str] = []
        block_start = 0
        directive = None
        pending_directive = None
        in_comment = False
        headers: list[str] = []
        for number, line in enumerate(page.lines, 1):
            stripped = line.strip()
            if SITE_PATH.search(line) and fence:
                self.error(page, number, "site-specific text in code block")
            if fence:
                if stripped.startswith(fence):
                    self.finish_block(page, block_start, language, block, directive, not history)
                    fence, block = None, []
                else:
                    block.append(line)
                continue
            if in_comment:
                if "-->" in line:
                    in_comment = False
                continue
            found = DIRECTIVE.search(line)
            if found:
                pending_directive = found.group(1)
                continue
            if stripped.startswith("<!--"):
                if "-->" not in stripped:
                    in_comment = True
                continue
            if stripped.startswith(("```", "~~~")):
                fence, language = stripped[:3], stripped[3:].strip().lower()
                block_start, directive, pending_directive = number, pending_directive, None
                continue
            if stripped:
                pending_directive = None  # a directive applies only to the code block right after it
            for match in TODO.finditer(line):
                self.todos.append(f"{page.name}:{number}: TODO({match.group(1)})")
                self.warn(page, number, f"open TODO({match.group(1)})")
            site = SITE_PATH.search(line)
            if site:
                self.error(page, number, f"site-specific path: {site.group(0)}")
            # Table columns: cells under an "old" or "3.5" header of the migration guide are not checked.
            cells = None
            if stripped.startswith("|"):
                cells = [cell.strip() for cell in stripped.strip("|").split("|")]
                following = page.lines[number] if number < len(page.lines) else ""
                if re.match(r"^\|?\s*:?-{3,}", following.strip()):
                    headers = [cell.lower() for cell in cells]
            else:
                headers = []
            segments = [(line, True)]
            if cells is not None and migration and headers:
                segments = [
                    (cell, not (headers[i].startswith(("old", "3.5"))) if i < len(headers) else True)
                    for i, cell in enumerate(cells)
                ]
            for text, checked in segments:
                checks = checked and not history
                for span_match in CODE_SPAN.finditer(text):
                    self.check_span(page, number, span_match.group(2), checks)
                without_code = CODE_SPAN.sub("", text)
                if checks and not migration:
                    phrase = HISTORY_PHRASES.search(without_code)
                    if phrase:
                        self.warn(page, number, f"history phrasing: {phrase.group(0)!r}")
                    if methods and METHODS_VERSION.search(without_code):
                        self.warn(
                            page,
                            number,
                            f"release history in a methods page: {METHODS_VERSION.search(without_code).group(0)!r}",
                        )
                for target in LINK.findall(without_code):
                    self.check_link(page, number, target)
            reference = REFERENCE_LINK.match(line)
            if reference:
                self.check_link(page, number, reference.group(1))
        if fence:
            self.error(page, block_start, "unclosed code block")

    def finish_block(self, page: Page, number: int, language: str, block: list[str], directive, checks: bool):
        if not checks or directive == "skip":
            return
        if language == "toml":
            self.check_toml_block(page, number, "\n".join(block), directive)
            return
        if language in ("", "bash", "sh", "shell", "console", "text"):
            joined, start = "", number
            for offset, line in enumerate(block, 1):
                if not joined:
                    start = number + offset
                if line.rstrip().endswith("\\"):
                    joined += line.rstrip()[:-1] + " "
                    continue
                joined += line
                if joined.strip():
                    self.check_command(page, start, joined)
                joined = ""

    # ------------------------------------------------------------------------------------------ whole set
    def check_whole_set(self):
        self.check_generated()
        self.check_environment()
        self.check_methods()
        self.check_ledger()
        self.check_docstring_links()
        self.check_outputs()
        self.check_examples()
        self.check_workflow()

    def check_generated(self, pages: dict[str, str] | None = None):
        """Every committed generated reference equals the generators' current output."""
        if pages is None:
            try:
                import generate_docs

                pages = generate_docs.generated()
            except Exception as error:  # a generator refusing the source is a documentation failure
                self.errors.append(f"docs/generate_docs.py: generators failed: {type(error).__name__}: {error}")
                return
        for name, expected in pages.items():
            path = SOURCE / name
            if not path.exists() or path.read_text() != expected:
                self.errors.append(f"{name}: stale generated reference; run python docs/generate_docs.py")

    def check_environment(self, package: Path = PACKAGE):
        """Every runtime environment read is in the generated environment reference; nothing else reads it."""
        import generate_environment

        try:
            reads = generate_environment.environment_reads(package)
        except ValueError as error:
            self.errors.append(f"environment: {error}")
            reads = []
        reference = DOCS / "reference/environment.md"
        documented = reference.read_text() if reference.exists() else ""
        for row in reads:
            if f"`{row['name']}`" not in documented:
                self.errors.append(f"{row['source']}:{row['line']}: environment read {row['name']} is undocumented")
        for access in environment_access(package):
            self.errors.append(f"haplotype_reconstruction/{access}: only the runtime may use the process environment")

    def check_methods(self, coverage: dict | None = None, manifest: list[dict] | None = None):
        """Each carried P0 unit has exactly one methods section, with its anchor and required parts."""
        if coverage is None:
            coverage = json.loads((DOCS / "methods/coverage.json").read_text())
        if manifest is None:
            manifest = json.loads((DOCS / "methods/units.json").read_text())["units"]
        carried = {(row["area"], row["unit"]): row for row in coverage["carried"]}
        removed = {(row["area"], row["unit"]) for row in coverage.get("removed", [])}
        counts = Counter((row["area"], row["unit"]) for row in manifest)
        for key in sorted(carried):
            if counts[key] != 1:
                self.errors.append(f"docs/methods/units.json: carried unit {'/'.join(key)} has {counts[key]} sections")
        for key in sorted(set(counts) - set(carried)):
            reason = "a removed unit" if key in removed else "not a carried unit"
            self.errors.append(f"docs/methods/units.json: {'/'.join(key)} is {reason}")
        accretions = (DOCS / "developer/accretions.md").read_text()
        for row in manifest:
            key = (row["area"], row["unit"])
            page_path = DOCS / row["page"]
            label = f"docs/{row['page']}#{row['anchor']}"
            if not page_path.exists():
                self.errors.append(f"{label}: page missing for unit {row['unit']}")
                continue
            if row["classification"] != carried.get(key, row)["classification"]:
                self.errors.append(f"{label}: classification differs from coverage.json")
            section = self.section(page_path, row["anchor"])
            if section is None:
                self.errors.append(f"{label}: no section anchored for unit {row['unit']}")
                continue
            heading = next((line for line in section if line.startswith("#")), "")
            if row["unit"].lower() not in heading.lower():
                self.errors.append(f"{label}: section heading does not name unit {row['unit']}")
            text = "\n".join(section)
            for part in ("Implementation:", "Parameters", "Cost and scope:"):
                if part not in text:
                    self.errors.append(f"{label}: section lacks {part!r}")
            for source in row["sources"]:
                if not (PACKAGE / source).exists():
                    self.errors.append(f"{label}: implementation source {source} does not exist")
            for name in row["parameters"]:
                if name not in self.parameter_names:
                    self.errors.append(f"{label}: parameter {name!r} is not a declared parameter or setting")
            named = re.search(r"^Parameters by name: (.*)$", text, re.M)
            for name in CODE_SPAN.findall(named.group(1)) if named else ():
                if name[1].split(".")[-1] not in self.parameter_names:
                    self.errors.append(f"{label}: parameter `{name[1]}` is not a declared parameter or setting")
            if row["classification"].startswith("Accretion") and "**Accretion:**" not in text:
                self.errors.append(f"{label}: accretion unit is not marked as such")
            if "Accretion" in row["classification"] and f"#{row['anchor']})" not in accretions:
                self.errors.append(f"{label}: accretion (or accretion part) missing from docs/developer/accretions.md")
        for _area, unit in sorted(removed):
            for page_path in sorted((DOCS / "methods").glob("*.md")):
                if unit.lower() in self.anchors(page_path):
                    self.errors.append(f"{page_path.relative_to(SOURCE)}: removed unit {unit} has a methods section")

    def check_ledger(self, ledger: Path = SOURCE / "tests/oracle/ledger.toml"):
        """Each numerical-ledger entry has a row in the tie guide whose linked methods sections name it."""
        page = DOCS / "developer/ties.md"
        rows = {}
        for line in page.read_text().splitlines():
            match = re.match(r"^\| (L\d+) \|", line)
            if match:
                rows[match.group(1)] = line
        if ledger.exists():  # the oracle ships with the source tree, not necessarily with a distribution
            import tomllib

            for entry in tomllib.loads(ledger.read_text())["entry"]:
                top = re.match(r"L\d+", entry["id"]).group(0)
                if top not in rows:
                    self.errors.append(f"docs/developer/ties.md: ledger entry {entry['id']} has no ledger row {top}")
        for entry, line in rows.items():
            links = re.findall(r"\]\(\.\./methods/([\w]+\.md)#([\w-]+)\)", line)
            if not links:
                self.errors.append(f"docs/developer/ties.md: ledger row {entry} links no methods section")
            for target, anchor in links:
                section = self.section(DOCS / "methods" / target, anchor)
                if section is None or not re.search(rf"\b{entry}\b", "\n".join(section)):
                    self.errors.append(
                        f"docs/developer/ties.md: {entry} links methods/{target}#{anchor}, "
                        f"which does not describe {entry}"
                    )

    def section(self, path: Path, anchor: str) -> list[str] | None:
        """Lines from an explicit anchor up to the next explicit anchor or same-or-higher heading."""
        lines = path.read_text().splitlines()
        marker = f'<a id="{anchor}"></a>'
        try:
            start = lines.index(marker)
        except ValueError:
            return None
        body = []
        for line in lines[start + 1 :]:
            if EXPLICIT_ANCHOR.search(line) or (body and re.match(r"^#{1,2}\s", line)):
                break
            body.append(line)
        return body

    def check_docstring_links(self):
        """Documentation paths cited in package source resolve to a page and, when given, an anchor."""
        for path in sorted(PACKAGE.rglob("*.py")):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                for page, anchor in DOCS_REFERENCE.findall(line):
                    target = DOCS / page
                    where = f"{path.relative_to(SOURCE)}:{number}"
                    if not target.exists():
                        self.errors.append(f"{where}: cites missing page docs/{page}")
                    elif anchor and anchor.rstrip(".") not in self.anchors(target):
                        self.errors.append(f"{where}: cites missing anchor docs/{page}#{anchor.rstrip('.')}")

    def check_outputs(self, package: Path = PACKAGE):
        """Every file name a writer publishes appears in the output dictionary."""
        names, problems = published_names(package)
        self.errors += [f"haplotype_reconstruction/{problem}" for problem in problems]
        text = "\n".join((SOURCE / page).read_text() for page in OUTPUT_DICTIONARY)
        for name, writer in sorted(names.items()):
            if not documented_output(name, text):
                self.errors.append(f"haplotype_reconstruction/{writer}: published {name} has no dictionary entry")
        for key, where in undefined_evaluation_keys((SOURCE / OUTPUT_DICTIONARY[0]).read_text(), package):
            self.errors.append(f"haplotype_reconstruction/{where}: evaluation field {key} is not defined in D7")

    def check_examples(self, receipt: Path = DOCS / "examples/vcf_headers.json"):
        """Captured VCF headers are the files their receipt hashes (observed output, never hand-edited)."""
        import hashlib

        entries = json.loads(receipt.read_text())["files"]
        listed = {entry["file"] for entry in entries}
        for entry in entries:
            path = receipt.parent / entry["file"]
            if not path.exists():
                self.errors.append(f"docs/examples/{entry['file']}: listed in vcf_headers.json but missing")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                self.errors.append(f"docs/examples/{entry['file']}: differs from its recorded capture hash")
        for path in sorted(receipt.parent.glob("*.header.vcf")):
            if path.name not in listed:
                self.errors.append(f"docs/examples/{path.name}: header capture without a receipt entry")

    def check_workflow(self, receipt_path: Path | None = None):
        """The Snakemake example's commands exist; its integration receipt is current and passed."""
        import workflow_receipt

        path = DOCS / "examples/Snakefile"
        if not path.exists():
            self.errors.append("docs/examples/Snakefile: missing example")
            return
        text = path.read_text()
        page = Page(path, str(path.relative_to(SOURCE)), text.splitlines())
        templates = workflow_receipt.rule_commands(text)
        if tuple(templates) != workflow_receipt.RULES:
            self.errors.append(f"{page.name}: rules are {tuple(templates)}, expected {workflow_receipt.RULES}")
        for number, line in enumerate(page.lines, 1):
            match = re.search(r'shell:\s*"(.*)"', line)
            if match:
                self.check_command(page, number, re.sub(r"\{[^}]+\}", "PLACEHOLDER", match.group(1)))
        receipt_path = receipt_path or workflow_receipt.RECEIPT
        name = receipt_path.relative_to(SOURCE) if receipt_path.is_relative_to(SOURCE) else receipt_path
        if not receipt_path.exists():
            problems = [f"missing; the integration test {workflow_receipt.WORKFLOW_TEST} must pass and be projected"]
        else:
            problems = workflow_receipt.validate(json.loads(receipt_path.read_text()), path)
        for problem in problems:
            if self.draft:
                self.pending.append(f"{name}: {problem}")
            else:
                self.errors.append(f"{name}: {problem}")


def evaluation_stages(path: Path = PACKAGE / "evaluation/evaluate.py") -> set[str]:
    """`evaluate --stages` names from the module's literal stage tuples (no numerical import)."""
    tuples: dict[str, list[str]] = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple):
            names: list[str] = []
            for item in node.value.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    names.append(item.value)
                elif isinstance(item, ast.Starred) and isinstance(item.value, ast.Name):
                    names += tuples[item.value.id]
            for target in node.targets:
                if isinstance(target, ast.Name):
                    tuples[target.id] = names
    return {*tuples["EVALUATION_STAGES"], "available", "all"}


def default_pages() -> list[Path]:
    pages = [SOURCE / "README.md", SOURCE / "CHANGELOG.md"]
    pages += sorted(DOCS.rglob("*.md"))
    return [page for page in pages if page.exists()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check the documentation against the code.")
    parser.add_argument(
        "--page",
        nargs="+",
        type=Path,
        help="Check only these pages (relative to the source tree or absolute); skips the whole-set checks.",
    )
    parser.add_argument("--todo", action="store_true", help="List every TODO marker.")
    parser.add_argument(
        "--draft", action="store_true", help="Report release-evidence gates as pending (never for a release)."
    )
    args = parser.parse_args(argv)
    checker = Checker(draft=args.draft)
    paths = (
        default_pages()
        if not args.page
        else [(page if page.is_absolute() else (SOURCE / page)).resolve() for page in args.page]
    )
    for path in paths:
        if not path.exists():
            checker.errors.append(f"{path}: no such page")
            continue
        name = str(path.relative_to(SOURCE)) if path.is_relative_to(SOURCE) else str(path)
        checker.check_page(Page(path, name, path.read_text().splitlines()))
    if not args.page:
        checker.check_whole_set()
    out = sys.stdout
    for warning in checker.warnings:
        out.write(f"warning: {warning}\n")
    if args.todo:
        for todo in checker.todos:
            out.write(f"todo: {todo}\n")
    for error in checker.errors:
        out.write(f"error: {error}\n")
    for item in checker.pending:
        out.write(f"pending: {item}\n")
    scope = "selected pages only" if args.page else "whole set"
    out.write(f"{len(paths)} pages ({scope}): {len(checker.errors)} errors, {len(checker.warnings)} warnings")
    out.write(f"; DRAFT: {len(checker.pending)} release gates pending\n" if args.draft else "\n")
    return 1 if checker.errors or checker.warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
