#!/usr/bin/env python3
"""Regenerate all committed code-derived documentation references atomically."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import generate_conventions
import generate_environment
import generate_outputs
import generate_records
import generate_references

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from haplotype_reconstruction.core.files import atomic_path  # noqa: E402


def kernel_table() -> str:
    """The declared-kernel reference of the roots its page names, rendered in a fresh interpreter.

    A fresh process imports only those roots, so the table never lists kernels that the other
    generators happened to import.
    """
    command = [sys.executable, str(ROOT / "tools/generate_kernel_table.py"), "--page-roots", "--stdout"]
    environment = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run(command, cwd=ROOT, env=environment, check=True, capture_output=True, text=True).stdout


def generated() -> dict[str, str]:
    pages = {f"docs/reference/{name}": value for name, value in generate_references.generated().items()}
    for module in (generate_conventions, generate_environment, generate_outputs, generate_records):
        pages[module.TARGET.relative_to(ROOT).as_posix()] = module.render()
    pages["docs/developer/kernels.md"] = kernel_table()
    return pages


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail on stale references without changing them.")
    args = parser.parse_args(argv)
    stale = []
    for name, content in generated().items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with atomic_path(path) as temporary:
                temporary.write_text(content)
    if stale:
        sys.stdout.write("Stale references: " + ", ".join(stale) + "\n")
    return int(bool(stale))


if __name__ == "__main__":
    raise SystemExit(main())
