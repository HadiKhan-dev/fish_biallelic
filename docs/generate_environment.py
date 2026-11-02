#!/usr/bin/env python3
"""Generate tested versions and actual runtime environment reads without imports."""

from __future__ import annotations

import argparse
import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "haplotype_reconstruction"
TARGET = ROOT / "docs/reference/environment.md"


def environment_reads(package: Path = PACKAGE) -> list[dict]:
    """Include literal reads and the runtime's explicit indirect-name families."""
    found = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            name = None
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) in ("os.getenv", "os.environ.get", "environ.get")
                and node.args
            ):
                name = node.args[0]
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Load)
                and ast.unparse(node.value) in ("os.environ", "environ")
            ):
                name = node.slice
            if name is None:
                continue
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                names = [name.value]
            elif path.name == "kernels.py" and path.parent.name == "engine":
                # Indirect reads enumerate this local scheduler table and NUMBA_*.
                function = next(
                    (
                        n
                        for n in tree.body
                        if isinstance(n, ast.FunctionDef) and n.lineno <= node.lineno <= n.end_lineno
                    ),
                    None,
                )
                if function is None:
                    raise ValueError(f"Unscoped environment read at {path.name}:{node.lineno}")
                names = sorted(
                    {
                        n.value
                        for n in ast.walk(function)
                        if isinstance(n, ast.Constant)
                        and isinstance(n.value, str)
                        and re.fullmatch(r"[A-Z][A-Z0-9_]+", n.value)
                    }
                )
                if function.name == "cache_key":
                    names = ["NUMBA_*"]
            else:
                raise ValueError(f"Unresolved environment read at {path.relative_to(package)}:{node.lineno}")
            for value in names:
                row = dict(name=value, source=path.relative_to(ROOT).as_posix(), line=node.lineno)
                if row not in found:
                    found.append(row)
    return found


def render() -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    pins = [
        line.strip()
        for line in (ROOT / "requirements-tested.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    header = (ROOT / "requirements-tested.txt").read_text().splitlines()[0].lstrip("# ")
    lines = [
        "# Tested environment and runtime variables",
        "",
        (
            "Generated from package metadata, `requirements-tested.txt` and runtime source reads. "
            "Regenerate with `python docs/generate_docs.py`."
        ),
        "",
        f"Packaging requires Python `{project['requires-python']}`. Reference capture: {header}",
        "",
        (
            "The reference platform is Linux x86_64 with CPU execution. These are tested runtime "
            "constraints, not a complete transitive lockfile or proof for every later dependency "
            "version. No GPU is required."
        ),
        "",
        "| Distribution | Tested version |",
        "|---|---|",
    ]
    for pin in pins:
        name, version = pin.split("==", 1)
        lines.append(f"| `{name}` | `{version}` |")
    lines += [
        "",
        (
            "Install a supplied wheel or source distribution with these constraints; see "
            "[installation](../user/installation.md). Build tools and optional developer/test tools are "
            "separate from runtime dependencies."
        ),
        "",
        "## Environment reads",
        "",
        (
            "Science reads no environment variables. The application establishes the numerical policy "
            "before numerical imports. Runtime reads below choose process limits, cache paths or cache "
            "identity; they are not scientific settings. Names are listed even when a caller never "
            "supplies them."
        ),
        "",
        "| Variable | Runtime use | Readers |",
        "|---|---|---|",
    ]
    uses = {
        "SLURM_MEM_PER_NODE": "Optional memory-limit input in MiB, subordinate to validated process limits.",
        "NUMBA_CACHE_DIR": "Explicit/inherited private compilation cache directory.",
        "NUMBA_NUM_THREADS": "Inherited runtime team capacity.",
        "XDG_CACHE_HOME": "Persistent user cache base when supplied.",
        "SLURM_TMPDIR": "Optional temporary cache directory when a run identifier is present.",
        "NUMBA_*": (
            "Nonoperational code-generation values enter the compilation cache key; NUMBA_CACHE_DIR, "
            "NUMBA_NUM_THREADS and NUMBA_THREADING_LAYER are excluded from that key."
        ),
    }
    grouped = {}
    for row in environment_reads():
        grouped.setdefault(row["name"], set()).add(row["source"])
    for name, sources in sorted(grouped.items()):
        use = uses.get(
            name, "Optional external execution-scope identifier for private cache naming; no scientific effect."
        )
        links = ", ".join(f"[{Path(source).name}](../../{source})" for source in sorted(sources))
        lines.append(f"| `{name}` | {use} | {links} |")
    lines += [
        "",
        (
            "The runtime sets numerical-library thread limits and fixes `NUMBA_DISABLE_INTEL_SVML`, "
            "`NUMBA_THREADING_LAYER` and `OMP_WAIT_POLICY`. These assignments are not additional "
            "user-controlled scientific options. Tie reversal and census use explicit settings "
            "described in [ties](../developer/ties.md)."
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
    temp = TARGET.with_suffix(".tmp")
    temp.write_text(expected)
    temp.replace(TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
