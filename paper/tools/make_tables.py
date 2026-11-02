"""Write the supplementary tables from the repository's own documentation.

Table S1 is the "Kernel ladders" table of docs/developer/costs.md and Table S2 the
"Numerical ledger" of docs/developer/ties.md. Rows are copied, with Markdown links
reduced to their text, so the supplement always shows what those pages record.

Run from anywhere: python3 paper/tools/make_tables.py
"""

from __future__ import annotations

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
DOCS = PAPER.parent / "docs" / "developer"
OUT = PAPER / "supplement" / "S7_tables.md"


def section_rows(path: Path, heading: str) -> list[list[str]]:
    rows, inside = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.strip() == heading
            continue
        if inside and line.startswith("|") and not re.match(r"^\|\s*-", line):
            cells = [re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", c).strip() for c in line.strip().strip("|").split("|")]
            rows.append(cells)
    if len(rows) < 2:
        raise SystemExit(f"{path}: no table under {heading!r}")
    return rows[1:]  # drop the header row


def table(header: list[str], rows: list[list[str]], caption: str, widths: list[int]) -> str:
    """A pipe table; `widths` are relative column widths (pandoc reads them from the dashes)."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("-" * w for w in widths) + "|"]
    out += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
    return "\n".join(out) + f"\n\nTable: {caption}\n"


def slope(cell: str) -> str:
    """'1.57 ± 0.08 over 5–30 (≤ cubic)' -> '1.57 ± 0.08 (≤ cubic)': the ladder range is in the header."""
    return re.sub(r" over [^(]*\(", " (", cell)


def main() -> None:
    ladders = section_rows(DOCS / "costs.md", "## Kernel ladders")
    ledger = section_rows(DOCS / "ties.md", "## Numerical ledger")
    s1 = table(
        ["Stage", "Probe", "What the probe covers", "K slope (K = 5–30, N = 200)", "N slope (N = 250–2,500, K = 15)"],
        [[r[0], r[1].strip("`"), r[2], slope(r[3]), slope(r[4])] for r in ladders],
        "Kernel ladders of release 4.0. Each probe times one kernel on direct synthetic inputs, on one "
        "thread, after a discarded first call; slopes are fitted log–log exponents with standard errors. "
        "A K slope at or below 3.3 is accepted as at most cubic; “exempt” marks a stage under a decided "
        "cost exemption, and “size proxy” an operational kernel whose size stands in for K and N. "
        "Source: docs/developer/costs.md.",
        [13, 19, 38, 15, 15],
    )
    s2 = table(
        ["Entry", "Category", "Form", "Where it acts"],
        ledger,
        "The numerical ledger of release 4.0: declared normalisations in which one arithmetic form "
        "replaces several, or a declared order replaces one chosen by workers, memory, object identity or "
        "the environment. No entry changes a model, objective, threshold, candidate set, search order or "
        "budget. Source: docs/developer/ties.md.",
        [7, 17, 46, 30],
    )
    OUT.write_text(
        "# S7. Supplementary tables\n\n"
        f"{s1}\n"
        f"{s2}",
        encoding="utf-8",
    )
    print(f"tables: {OUT.relative_to(PAPER)} ({len(ladders)} ladder rows, {len(ledger)} ledger entries)")


if __name__ == "__main__":
    main()
