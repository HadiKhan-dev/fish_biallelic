#!/usr/bin/env python3
"""Generate tie constants, policy call sites and named random streams from the AST."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from haplotype_reconstruction.core.files import atomic_path  # noqa: E402
PACKAGE = ROOT / "haplotype_reconstruction"
TARGET = ROOT / "docs/reference/conventions.md"


def cell(value: str) -> str:
    return " ".join(value.split()).replace("|", "&#124;")


def render() -> str:
    source = PACKAGE / "core/ties.py"
    tree = ast.parse(source.read_text())
    lines = [
        "# Tie and random conventions",
        "",
        (
            "Generated from `core/ties.py`, its policy call sites and `core/random.py` without "
            "importing numerical modules. Expressions retain the code's arithmetic. See the [tie "
            "guide](../developer/ties.md) for their meaning."
        ),
        "",
        "## Shared quantities",
        "",
        "| Name | Source expression |",
        "|---|---|",
    ]
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
        if isinstance(target, ast.Name) and target.id.isupper():
            lines.append(f"| `{target.id}` | `{cell(ast.unparse(node.value))}` |")
    lines += [
        "",
        "## Named policy call sites",
        "",
        (
            "The name identifies a decision. The kind selects its scale rule; a fixed width is an "
            "explicit expression from that decision's parameter record. A dynamic expression is "
            "displayed as source rather than guessed. Secondary scientific preferences are documented "
            "in the owning method, not encoded by reversing this name."
        ),
        "",
        "| Policy name or expression | Constructor and arguments | Source |",
        "|---|---|---|",
    ]
    for path in sorted(PACKAGE.rglob("*.py")):
        parsed = ast.parse(path.read_text())
        for node in ast.walk(parsed):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func).split(".")[-1]
            if name not in {"policy", "fixed_policy", "fixed_band"} or not node.args:
                continue
            rel = path.relative_to(ROOT).as_posix()
            lines.append(
                f"| `{cell(ast.unparse(node.args[0]))}` | `{cell(ast.unparse(node))}` | "
                f"[{path.name}:{node.lineno}](../../{rel}) |"
            )
    lines += [
        "",
        "## Random streams",
        "",
        (
            "The stream name preserves generator construction and draw order, not just a seed. "
            "`simulation.structure` uses Python's MT generator through `structure_stream`; the other "
            "registered streams use NumPy's default PCG64 generator. See "
            "[simulation](../methods/simulation.md) and [pedigree "
            "resampling](../methods/pedigree.md#pd16)."
        ),
        "",
        "| Stream |",
        "|---|",
    ]
    tree = ast.parse((PACKAGE / "core/random.py").read_text())
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
            if isinstance(target, ast.Name) and target.id == "STREAMS":
                for name in ast.literal_eval(node.value):
                    lines.append(f"| `{name}` |")
    lines += [
        "",
        (
            "Simulation realises an unspecified seed once during initialisation and persists it. Resume "
            "reuses that realised seed. Founder streams use configured contig order; read seeds are "
            "consumed for every contig, including already completed ones. Bootstrap draws preserve "
            "their original vector shapes and replicate order. Renaming a stream or replacing a loop "
            "with a bulk draw requires explicit identity proof."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    expected = render()
    if args.check:
        return int(not TARGET.exists() or TARGET.read_text() != expected)
    with atomic_path(TARGET) as temp:
        temp.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
