"""Writing published tables, and the tables that are a stage's summary rows.

Every file is written to a hidden temporary next to its destination and renamed into place, so a reader never sees
a partial table. Cells holding lists or dicts are JSON (sorted keys); other values are written as pandas writes
them.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ...core.columns import column_name


def temporary_path(path: Path) -> Path:
    """A hidden temporary beside ``path``, unique per writer and call, for every published table and export file.

    Duplicate workflow steps (two scatter jobs for one contig) may write the same file at once; a shared temporary
    name would let one writer rename the other's file away (ARCHITECTURE_40 3.3). Final names, contents and the
    atomic rename are unchanged.
    """
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def json_cells(frame):
    """A copy of ``frame`` whose list, tuple and dict cells are JSON text (sorted keys)."""
    frame = frame.copy()
    for column in frame:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(
                lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
            )
    return frame


def write_frame(path, frame, *, sep: str = ",") -> Path:
    """Write a DataFrame (no index) atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    frame.rename(columns=column_name).to_csv(temporary, sep=sep, index=False)
    temporary.replace(path)
    return path


def write_rows(path, rows, *, columns=None, sep: str = ",") -> Path:
    """Write row dicts (or dataclasses) as a table; ``columns`` fixes the header of an empty table."""
    import pandas as pd

    rows = [asdict(cast(Any, row)) if is_dataclass(row) else dict(row) for row in rows]
    frame = pd.DataFrame(rows, columns=None if rows or columns is None else list(columns))
    return write_frame(path, json_cells(frame), sep=sep)


def write_text_table(path, header, rows, *, sep: str = ",") -> Path:
    """Write a table whose cells are already text (header + rows of strings), one line per row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter=sep, lineterminator="\n")
        writer.writerow([column_name(name) for name in header])
        writer.writerows(rows)
    temporary.replace(path)
    return path


def contig_part(output, family, contig, frames):
    """Publish private CSV parts without retaining all chromosomes' report frames in memory."""
    if not frames:
        return contig, {}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f".{family}-", dir=output))
    paths = {}
    for name, frame in frames.items():
        frame = frame.copy()
        if "contig" not in frame:
            frame.insert(0, "contig", contig)
        else:
            frame = frame[["contig", *(name for name in frame if name != "contig")]]
        paths[name] = write_frame(directory / f"{name}.csv", json_cells(frame))
    return contig, paths


def write_contig_tables(output, family, parts):
    """Concatenate private CSV cells in supplied contig order and publish each table atomically."""
    directory = Path(output) / family
    directory.mkdir(parents=True, exist_ok=True)
    names = dict.fromkeys(name for _, paths in parts for name in paths)
    written = []
    try:
        for name in names:
            paths = [paths[name] for _, paths in parts if name in paths]
            columns = {}
            for path in paths:
                with Path(path).open(newline="") as handle:
                    columns.update(dict.fromkeys(next(csv.reader(handle), ())))
            path = directory / f"{name}.csv"
            temporary = temporary_path(path)
            with temporary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
                writer.writeheader()
                for part in paths:
                    with Path(part).open(newline="") as source:
                        writer.writerows(csv.DictReader(source))
            temporary.replace(path)
            written.append(path)
    finally:
        for parent in {Path(path).parent for _, paths in parts for path in paths.values()}:
            shutil.rmtree(parent)
    return written


def write_json(path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
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
    """Write the simulated observed pedigree supplied by results; never read inference."""
    table = manifest.observed
    header = TRUTH_PEDIGREE_COLUMNS
    rows = [
        dict(zip(header, (sample, f"F{cohort}", first, second), strict=False))
        for sample, cohort, first, second in zip(
            table.samples, table.cohorts, table.parent1, table.parent2, strict=False
        )
    ]
    return write_rows(Path(output) / "simulation" / "truth_pedigree.csv", rows, columns=header)


def frame_in_input_order(frame, input_ids: Sequence[str] | None, column: str = "Sample"):
    """A per-sample table with its rows stably sorted by the input position of ``column`` (published tables list
    samples in the input order, as release 2.0 did; the records keep the canonical order). Rows whose sample is not
    in ``input_ids`` keep their relative order after the others. The frame is returned unchanged when
    ``input_ids`` is None or it has no ``column``."""
    if input_ids is None or column not in getattr(frame, "columns", ()):
        return frame
    position = {str(name): index for index, name in enumerate(input_ids)}
    keys: np.ndarray = np.asarray([position.get(str(name), len(position)) for name in frame[column]], dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    if np.array_equal(order, np.arange(len(order))):
        return frame
    return frame.iloc[order].reset_index(drop=True)


def record_mapping(value):
    """Format an explicit typed summary as JSON fields, preserving its field order."""
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {key: record_mapping(item) for key, item in value.items()}
    if is_dataclass(value):
        from dataclasses import fields

        return {
            field.name: record_mapping(getattr(value, field.name)) for field in fields(value) if field.name != "schema"
        }
    if isinstance(value, (list, tuple)):
        return [record_mapping(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# Published column schemas: B reports and P0 outputs inventory D6 / OU15.
# Row-dict tables retain B's headerless empty constructor; these constants do not
# impose columns on those tables. Scientific values remain upstream contract fields.
TRUTH_PEDIGREE_COLUMNS = ("sample", "cohort", "parent1", "parent2")
"""Columns in published order.

    sample: sample identifier (text).
    cohort: simulation generation label F<k>; F0 denotes an explicit-pedigree founder (text).
    parent1: first parent identifier; empty for an unobserved parent (text).
    parent2: second parent identifier; empty for an unobserved parent (text).
"""

REFIT_SUMMARY_COLUMNS = (
    "contig",
    "components",
    "founder_rows",
    "sites",
    "called_alleles",
    "changed_alleles",
    "ambiguous_alleles",
    "filled_alleles",
    "changed_inference_alleles",
    "changed_sites",
    "median_changed_gain_nats",
    "excluded_samples",
)
"""Columns in published order.

    contig: contig identifier (text).
    components: components (1).
    founder_rows: founder rows (1).
    sites: sites (1).
    called_alleles: called alleles (count).
    changed_alleles: changed alleles (1).
    ambiguous_alleles: ambiguous alleles (1).
    filled_alleles: filled alleles (1).
    changed_inference_alleles: changed inference alleles (1).
    changed_sites: changed sites (site).
    median_changed_gain_nats: median changed gain nats (nats).
    excluded_samples: excluded samples (1).
"""

PHASE_SUMMARY_COLUMNS = (
    "contig",
    "samples",
    "sites",
    "components",
    "sweeps",
    "settled",
    "objective",
    "junction_candidates",
    "withheld_joins",
    "tie_breaks",
    "called_alleles",
    "phased_heterozygotes",
    "unphased_heterozygotes",
    "phase_sets",
    "elapsed_seconds",
)
"""Columns in published order.

    contig: contig identifier (text).
    samples: samples (1).
    sites: sites (1).
    components: components (1).
    sweeps: sweeps (count).
    settled: settled (1).
    objective: objective (nats).
    junction_candidates: junction candidates (count).
    withheld_joins: withheld joins (count).
    tie_breaks: tie breaks (count).
    called_alleles: called alleles (count).
    phased_heterozygotes: phased heterozygotes (count).
    unphased_heterozygotes: unphased heterozygotes (count).
    phase_sets: phase sets (count).
    elapsed_seconds: recorded runtime, empty when the run record does not supply it (s).
"""
