"""Writing published tables, and the tables that are a stage's summary rows.

Every file is written to a hidden temporary next to its destination and renamed into place, so a reader never sees
a partial table. Cells holding lists or dicts are JSON (sorted keys); other values are written as pandas writes
them.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def json_cells(frame):
    """A copy of ``frame`` whose list, tuple and dict cells are JSON text (sorted keys)."""
    frame = frame.copy()
    for column in frame:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(
                lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value)
    return frame


def write_frame(path, frame, *, sep: str = ",") -> Path:
    """Write a DataFrame (no index) atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    frame.to_csv(temporary, sep=sep, index=False)
    temporary.replace(path)
    return path


def write_rows(path, rows, *, columns=None, sep: str = ",") -> Path:
    """Write row dicts (or dataclasses) as a table; ``columns`` fixes the header of an empty table."""
    import pandas as pd
    rows = [asdict(row) if is_dataclass(row) else dict(row) for row in rows]
    frame = pd.DataFrame(rows, columns=None if rows or columns is None else list(columns))
    return write_frame(path, json_cells(frame), sep=sep)


def write_text_table(path, header, rows, *, sep: str = "\t") -> Path:
    """Write a table whose cells are already text (header + rows of strings), one line per row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    temporary.write_text("\n".join(sep.join(map(str, line)) for line in (header, *rows)) + "\n")
    temporary.replace(path)
    return path


def write_json(path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
    return path


# --------------------------------------------------------------------------------------- stage summary tables
def write_founder_refit(output, summaries) -> Path:
    """OUTPUT/founder_refit/summary.csv: one row per contig (founder_refit.run's summaries, contig order)."""
    return write_rows(Path(output) / "founder_refit" / "summary.csv", summaries)


def write_family_phase(output, summaries) -> Path:
    """OUTPUT/family_phase/summary.csv: one row per contig (family_phase.run's FamilyPhaseSummary rows)."""
    return write_rows(Path(output) / "family_phase" / "summary.csv", summaries)


def write_truth_pedigree(output, manifest) -> Path:
    """OUTPUT/simulation/truth_pedigree.csv: the analysed simulated samples and their true parents."""
    from ..simulation.truth import truth_pedigree_table
    header, rows = truth_pedigree_table(manifest)
    return write_rows(Path(output) / "simulation" / "truth_pedigree.csv",
                      [dict(zip(header, row)) for row in rows], columns=header)
