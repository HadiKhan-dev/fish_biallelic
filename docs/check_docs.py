#!/usr/bin/env python3
"""Check the documentation against the code.

    python docs/check_docs.py                         # every page: README.md, CHANGELOG.md, docs/**/*.md
    python docs/check_docs.py --page docs/usage.md    # only these pages (no whole-set checks)
    python docs/check_docs.py --todo                  # also list the TODO markers

Checks (errors make the exit status 1):
- links: relative link targets exist; `#anchors` exist in the target page (GitHub heading slugs or explicit
  `<a id>` anchors);
- configuration: `[section] key` and `[section] key = value` code spans, `--set section.key=value`, and TOML code
  blocks name existing settings with valid values (the schema's own parser); a TOML block after
  `<!-- check_docs: config -->` must load as a complete configuration, one after `<!-- check_docs: defaults -->`
  may hold only default values, one after `<!-- check_docs: skip -->` is not checked;
- command lines: `haplotypes COMMAND ...` lines (code blocks and code spans) use options that COMMAND has, with
  valid choices; `--flag` code spans exist in some command; `evaluate --stages` names are evaluation stages;
- constants: a code span `RootConfig.path = value` (or `<module>.<NAME> = value`, module relative to the package)
  states the code's default; a span `RootConfig.path` names an existing field; `path/to/module.py` exists;
- docs/configuration.md is what docs/generate_configuration.py writes now;
- every environment variable the package reads directly is listed in docs/configuration.md;
- no site-specific absolute paths (/rds/, /home/).
Not checked in CHANGELOG.md (history) and in the "old" column of docs/migration.md tables: configuration, command
lines and constants. Warnings (exit status unaffected): history phrasing ("previously", "no longer", ...) outside
CHANGELOG.md and docs/migration.md, and TODO markers other than TODO(acceptance).
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import math
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS.parent
PACKAGE = SOURCE / "haplotype_reconstruction"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

HISTORY_PAGES = ("CHANGELOG.md",)
MIGRATION_PAGES = ("docs/migration.md",)
HISTORY_PHRASES = re.compile(r"\b(previously|no longer|as before|used to|formerly|restores the earlier|"
                             r"was removed|were removed|has been removed|now replaced)\b", re.IGNORECASE)
SITE_PATH = re.compile(r"(?<![\w.])/(rds|home)/[\w./-]+")
TODO = re.compile(r"TODO\(([^)]*)\)")
CODE_SPAN = re.compile(r"(`+)(.+?)\1")
LINK = re.compile(r"!?\[(?:[^\]\[]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)")
EXPLICIT_ANCHOR = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"")
DIRECTIVE = re.compile(r"<!--\s*check_docs:\s*(\w+)\s*-->")
TOML_KEY_SPAN = re.compile(r"^\[([a-z_][\w.]*)\](?:\s+([a-z_]\w*)(?:\s*=\s*(.+))?)?$")
SET_OVERRIDE = re.compile(r"--set[ =]+['\"]?([a-z_][\w.]*=[^'\"\s]*(?:\[[^\]]*\])?)['\"]?")
CONSTANT_SPAN = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?:\s*=\s*(.+))?$")
MODULE_FILE_SPAN = re.compile(r"^(?:haplotype_reconstruction/)?([\w/]+\.py)$")
ENV_READ = re.compile(r"os\.environ\.get\(\s*[\"']([A-Z][A-Z0-9_]+)[\"']"
                      r"|os\.environ\[\s*[\"']([A-Z][A-Z0-9_]+)[\"']\s*\](?!\s*=[^=])"
                      r"|os\.getenv\(\s*[\"']([A-Z][A-Z0-9_]+)[\"']")


@dataclasses.dataclass
class Page:
    path: Path
    name: str                      # path relative to SOURCE
    lines: list[str]


class Checker:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.todos: list[str] = []
        self._anchors: dict[Path, set[str]] = {}
        self._load_code()

    # ------------------------------------------------------------------------------------------- code facts
    def _load_code(self):
        from haplotype_reconstruction.cli import build_parser
        from haplotype_reconstruction.config import load, schema, stages
        from haplotype_reconstruction.evaluation import EVALUATION_STAGES
        self.load, self.schema = load, schema
        parser = build_parser()
        sub = next(item for item in parser._actions if isinstance(item, argparse._SubParsersAction))
        self.commands = {}
        for name, command in sub.choices.items():
            options = {}
            for action in command._actions:
                for option in action.option_strings:
                    options[option] = action
            self.commands[name] = options
        self.all_flags = {flag for options in self.commands.values() for flag in options}
        self.evaluation_stages = set(EVALUATION_STAGES) | {"available", "all"}
        rows = schema.reference_rows()
        self.keys = {(row["section"], row["key"]): row for row in rows}
        self.sections = {row["section"] for row in rows if row["section"]}
        defaults = schema.RunConfig()
        base = schema.RunConfig(input=dataclasses.replace(defaults.input, kind="simulation", contigs=("chrX",)),
                                simulation=dataclasses.replace(defaults.simulation, templates=Path("/probe")))
        configs = stages.stage_configs(base)
        self.roots = {}
        for item in dataclasses.fields(configs):
            built = getattr(configs, item.name)
            try:
                self.roots[type(built).__name__] = type(built)()
            except TypeError:
                self.roots[type(built).__name__] = built
        simulation = stages.simulation(base)
        self.roots[type(simulation).__name__] = simulation

    # ------------------------------------------------------------------------------------------------ helpers
    def error(self, page: Page, number: int, message: str):
        self.errors.append(f"{page.name}:{number}: {message}")

    def warn(self, page: Page, number: int, message: str):
        self.warnings.append(f"{page.name}:{number}: {message}")

    @staticmethod
    def slug(text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)                       # inline HTML
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)      # links keep their text
        slug = re.sub(r"[^\w\- ]", "", text.strip().lower())
        return slug.replace(" ", "-")

    def anchors(self, path: Path) -> set[str]:
        if path not in self._anchors:
            found, counts = set(), {}
            fence = None
            for line in path.read_text().splitlines():
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
                    base = self.slug(match.group(1))
                    count = counts.get(base, 0)
                    counts[base] = count + 1
                    found.add(base if count == 0 else f"{base}-{count}")
            self._anchors[path] = found
        return self._anchors[path]

    # --------------------------------------------------------------------------------------------- checks
    def check_link(self, page: Page, number: int, target: str):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target):          # http:, https:, mailto:
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
            self.load.build_config(mapping, cwd=tempfile.gettempdir())
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
            mapping = {}
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
        mapping = {}
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
                    self.load.load_config(path, cwd=folder)
                except Exception as error:
                    self.error(page, number, f"TOML block is not a complete configuration: {error}")
        elif directive == "defaults":
            config = self.load.build_config(mapping, cwd=tempfile.gettempdir())
            defaults = self.schema.RunConfig()

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
                        self.error(page, number, f"{'.'.join(path)} = {value!r} is not the default "
                                                 f"({self.schema.to_dict(default)!r})")

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
                            self.error(page, number, f"haplotypes {command} {flag}: {alternative!r} is not one of "
                                                     f"{', '.join(map(str, action.choices))}")
            if command == "evaluate" and flag == "--stages":
                for value in literal:
                    if value not in self.evaluation_stages:
                        self.error(page, number, f"evaluate --stages: unknown stage {value!r}")
            if flag == "--set":
                for value in values:
                    self.check_override(page, number, value)

    @staticmethod
    def placeholder(value: str) -> bool:
        return (value.isupper() or "<" in value or "..." in value or value.startswith("$")
                or re.fullmatch(r"[A-Z][A-Z0-9_]*(\.[A-Z]+)?", value) is not None)

    def resolve_constant(self, dotted: str):
        """(found, value) of `Root.path` or `<module>.<NAME>`; found is False when nothing resolves."""
        head, *rest = dotted.split(".")
        if head in self.roots:
            value = self.roots[head]
            for part in rest:
                if not hasattr(value, part):
                    return False, None
                value = getattr(value, part)
            return True, value
        parts = dotted.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            module_name = "haplotype_reconstruction." + ".".join(parts[:cut])
            try:
                value = importlib.import_module(module_name)
            except ImportError:
                continue
            for part in parts[cut:]:
                if not hasattr(value, part):
                    return False, None
                value = getattr(value, part)
            return True, value
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
            return len(stated) == len(actual) and all(Checker.same(a, b) for a, b in zip(stated, actual))
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
                return                          # placeholders such as SECTION.KEY
            if value_text is not None and re.fullmatch(r"[A-Za-z_]\w*", value_text.strip()) and \
                    value_text.strip() not in ("True", "False", "None"):
                value_text = None               # a placeholder value: check only that the name exists
            if value_text is None and head not in self.roots:
                return                          # module paths, attribute names in prose: not checked
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

    # ---------------------------------------------------------------------------------------------- pages
    def check_page(self, page: Page):
        history = page.name in HISTORY_PAGES
        migration = page.name in MIGRATION_PAGES
        fence = None
        language = ""
        block: list[str] = []
        block_start = 0
        directive = None
        pending_directive = None
        in_comment = False
        headers: list[str] = []
        continued = ""
        for number, line in enumerate(page.lines, 1):
            stripped = line.strip()
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
                pending_directive = None        # a directive applies only to the code block right after it
            for match in TODO.finditer(line):
                self.todos.append(f"{page.name}:{number}: TODO({match.group(1)})")
                if match.group(1) != "acceptance":
                    self.warn(page, number, f"open TODO({match.group(1)})")
            if SITE_PATH.search(line):
                self.error(page, number, f"site-specific path: {SITE_PATH.search(line).group(0)}")
            # Table columns: cells under an "old" header of docs/migration.md are not checked.
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
                segments = [(cell, not headers[i].startswith("old") if i < len(headers) else True)
                            for i, cell in enumerate(cells)]
            for text, checked in segments:
                checks = checked and not history
                for span_match in CODE_SPAN.finditer(text):
                    self.check_span(page, number, span_match.group(2), checks)
                without_code = CODE_SPAN.sub("", text)
                if checks and not migration and HISTORY_PHRASES.search(without_code):
                    self.warn(page, number, f"history phrasing: {HISTORY_PHRASES.search(without_code).group(0)!r}")
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
    def check_generated(self):
        sys.path.insert(0, str(DOCS))
        import generate_configuration
        current = (DOCS / "configuration.md").read_text() if (DOCS / "configuration.md").exists() else ""
        if generate_configuration.render() != current:
            self.errors.append("docs/configuration.md: out of date; run python docs/generate_configuration.py")
        documented = (DOCS / "configuration.md").read_text() if (DOCS / "configuration.md").exists() else ""
        for path in sorted(PACKAGE.rglob("*.py")):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                for groups in ENV_READ.findall(line):
                    name = next(group for group in groups if group)
                    if f"`{name}`" not in documented:
                        self.errors.append(f"{path.relative_to(SOURCE)}:{number}: environment variable {name} is "
                                           "read but not listed in docs/configuration.md (generate_configuration.py "
                                           "ENVIRONMENT_READ)")


def default_pages() -> list[Path]:
    pages = [SOURCE / "README.md", SOURCE / "CHANGELOG.md"]
    pages += sorted(DOCS.rglob("*.md"))
    return [page for page in pages if page.exists()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check the documentation against the code.")
    parser.add_argument("--page", nargs="+", type=Path, help="Check only these pages (paths relative to the "
                        "source tree or absolute); skips the whole-set checks.")
    parser.add_argument("--todo", action="store_true", help="List every TODO marker.")
    args = parser.parse_args(argv)
    checker = Checker()
    paths = default_pages() if not args.page else [
        (page if page.is_absolute() else (SOURCE / page)).resolve() for page in args.page]
    for path in paths:
        if not path.exists():
            checker.errors.append(f"{path}: no such page")
            continue
        name = str(path.relative_to(SOURCE)) if path.is_relative_to(SOURCE) else str(path)
        checker.check_page(Page(path, name, path.read_text().splitlines()))
    if not args.page:
        checker.check_generated()
    for warning in checker.warnings:
        print(f"warning: {warning}")
    if args.todo:
        for todo in checker.todos:
            print(f"todo: {todo}")
    for error in checker.errors:
        print(f"error: {error}")
    acceptance = sum(1 for todo in checker.todos if todo.endswith("TODO(acceptance)"))
    print(f"{len(paths)} pages: {len(checker.errors)} errors, {len(checker.warnings)} warnings, "
          f"{acceptance} TODO(acceptance) placeholders")
    return 1 if checker.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
