#!/usr/bin/env python3
"""Generate portable settings, parameter and command references from this source tree.

No science is executed. Parameter defaults are transcribed as Python expressions
from the AST, retaining named factories and their provenance. Run with --check to
verify committed pages without rewriting them.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from haplotype_reconstruction.core.files import atomic_path  # noqa: E402
DOCS = ROOT / "docs"
PACKAGE = ROOT / "haplotype_reconstruction"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cell(value) -> str:
    return " ".join(str(value).split()).replace("|", "&#124;")


def literal(value) -> str:
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(literal(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def parameter_catalog() -> list[dict]:
    """All declared parameter fields and public constants, in source order."""
    rows = []
    for path in sorted([*PACKAGE.rglob("*parameters.py"), PACKAGE / "core/ties.py"]):
        source = path.read_text()
        lines = source.splitlines()
        tree = ast.parse(source)
        module = ".".join(path.relative_to(PACKAGE).with_suffix("").parts)
        module_doc = ast.get_docstring(tree) or ""

        def declarations(body, owner="", description="", lines=lines, module=module, module_doc=module_doc, path=path):
            for node in body:
                if isinstance(node, ast.ClassDef):
                    if path.name.endswith("parameters.py"):
                        declarations(node.body, node.name, ast.get_docstring(node) or "")
                    continue
                if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                    continue
                target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
                if not isinstance(target, ast.Name) or target.id.startswith("_"):
                    continue
                name = target.id
                if not owner and not name.isupper():
                    continue
                value = node.value
                provenance = ""
                units = ""
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "field":
                    keywords = {part.arg: part.value for part in value.keywords}
                    if "default" in keywords:
                        value = keywords["default"]
                    elif "default_factory" in keywords:
                        value = ast.Call(func=keywords["default_factory"], args=[], keywords=[])
                    else:
                        continue
                    metadata = keywords.get("metadata")
                    if isinstance(metadata, ast.Dict):
                        for key, item in zip(metadata.keys, metadata.values, strict=True):
                            if isinstance(key, ast.Constant) and key.value in ("provenance", "units"):
                                try:
                                    text = ast.literal_eval(item)
                                except (ValueError, TypeError):
                                    text = ast.unparse(item)
                                if key.value == "provenance":
                                    provenance = str(text)
                                else:
                                    units = str(text)
                comments = []
                before = node.lineno - 2
                while before >= 0 and lines[before].lstrip().startswith("#"):
                    comments.insert(0, lines[before].lstrip().lstrip("#").strip())
                    before -= 1
                inline = (
                    lines[node.end_lineno - 1].split("#", 1)[1].strip() if "#" in lines[node.end_lineno - 1] else ""
                )
                explanation = " ".join([*comments, inline]).strip()
                if not provenance:
                    provenance = explanation or description.split("\n\n")[0] or module_doc.split("\n\n")[0]
                rows.append(
                    dict(
                        module=module,
                        owner=owner,
                        name=name,
                        expression=ast.unparse(value),
                        provenance=provenance,
                        units=units,
                        description=explanation,
                        source=path.relative_to(ROOT).as_posix(),
                        line=node.lineno,
                    )
                )

        declarations(tree.body)
    return rows


def render_settings() -> str:
    from haplotype_reconstruction.interface.settings.load import PRESETS
    from haplotype_reconstruction.interface.settings.schema import reference_rows

    out = [
        "# Settings reference",
        "",
        "Generated by `python docs/generate_references.py`; edit the schema, then regenerate.",
        "",
        (
            "Defaults < preset < TOML < command-line options < repeated `--set` overrides. Paths "
            "resolve against the invocation's working directory. Unknown settings and invalid values "
            "fail before execution."
        ),
        "",
        (
            "Empty strings represent an unset optional value. A default is not necessarily a complete "
            "runnable configuration. Scientific settings enter the relevant derivation; operational "
            "fields are identified below."
        ),
        "",
    ]
    section = None
    for row in reference_rows():
        if row["section"] != section:
            section = row["section"]
            out += [
                f"## {section or 'Top level'}",
                "",
                "| Key | Kind | Default | Choices | Affects results | Meaning and provenance |",
                "|---|---|---|---|---|---|",
            ]
        out.append(
            "| "
            + " | ".join(
                (
                    f"`{row['key']}`",
                    cell(row["kind"]),
                    f"`{literal(row['default'])}`",
                    cell(", ".join(map(str, row["choices"])) or "—"),
                    "yes" if row["affects_results"] else "no",
                    cell(row["doc"] + " Provenance: " + str(row["provenance"])),
                )
            )
            + " |"
        )
    out += ["", "## Presets", "", "A preset supplies settings; it does not read pedigree metadata.", ""]
    for name in PRESETS:
        path = PACKAGE / "interface/settings/presets" / f"{name}.toml"
        out += [f"### {name}", "", "```toml", path.read_text().strip(), "```", ""]
    out += ["Runtime environment variables are listed in the [environment reference](environment.md).", ""]
    return "\n".join(out)


def render_parameters() -> str:
    out = [
        "# Parameter reference",
        "",
        (
            "Generated by `python docs/generate_references.py` from every `*parameters.py` module. "
            "Defaults below are **Python source expressions**; named factories refer to their own "
            "tables. Expressions preserve exact arithmetic and provenance without importing science or "
            "running kernels."
        ),
        "",
        (
            "These internal model constants are not extra TOML settings. Only keys in the [settings "
            "reference](settings.md) are user settings. Module constants may also express "
            "dimension-only workspace limits; their source provenance distinguishes them from "
            "scientific parameters."
        ),
        "",
    ]
    group = None
    for row in parameter_catalog():
        current = (row["module"], row["owner"])
        if current != group:
            group = current
            out += [
                f"## {row['module']}" + (f" — {row['owner']}" if row["owner"] else ""),
                "",
                f"Source: [{row['source']}](../../{row['source']}).",
                "",
                "| Name | Default expression | Units | Provenance |",
                "|---|---|---|---|",
            ]
        out.append(
            f"| `{row['name']}` | `{cell(row['expression'])}` | {cell(row['units'] or 'See model')} | "
            f"{cell(row['provenance'])} |"
        )
    return "\n".join(out) + "\n"


def render_commands() -> str:
    from haplotype_reconstruction.interface.cli import build_parser

    parser = build_parser()
    sub = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    helps = {action.dest: action.help for action in sub._choices_actions}
    out = [
        "# Command reference",
        "",
        (
            "Generated by `python docs/generate_references.py` from the CLI parser. Options are checked "
            "against that parser by the documentation build."
        ),
        "",
        (
            "`run CONFIG` initialises a run when needed. Explicit `init` is required before "
            "workflow-step commands. Standalone export uses published records and has no source-VCF "
            "option. See [workflow mode](../user/workflow.md) for dependencies and markers."
        ),
        "",
    ]
    for name, command in sub.choices.items():
        out += [
            f"## {name}",
            "",
            str(helps.get(name, "")),
            "",
            "| Argument | Arity | Default | Choices | Help |",
            "|---|---|---|---|---|",
        ]
        for action in command._actions:
            options = ", ".join(action.option_strings) or action.dest.upper()
            default = "—" if action.default is argparse.SUPPRESS or action.default is None else repr(action.default)
            choices = ", ".join(map(str, action.choices)) if action.choices is not None else "—"
            out.append(
                f"| `{options}` | {action.nargs if action.nargs is not None else '1'} | `{cell(default)}` | "
                f"{cell(choices)} | {cell(action.help or '')} |"
            )
        out.append("")
    return "\n".join(out)


def generated() -> dict[str, str]:
    return {"settings.md": render_settings(), "parameters.md": render_parameters(), "commands.md": render_commands()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    stale = []
    for name, text in generated().items():
        path = DOCS / "reference" / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with atomic_path(path) as temporary:
                temporary.write_text(text)
    if stale:
        sys.stdout.write("Stale references: " + ", ".join(stale) + "\n")
    return int(bool(stale))


if __name__ == "__main__":
    raise SystemExit(main())
