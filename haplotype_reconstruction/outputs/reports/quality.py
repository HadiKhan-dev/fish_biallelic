"""OUTPUT/quality/samples.csv and segments.csv: the per-contig sample quality exclusion.

The rows are the evidence stage's (evidence.reports.sample_table / segment_table): per sample the contamination
statistics and whether its reads were kept out of founder construction on this contig, and the inherited-segment
regions shared by pairs of samples. Nothing is written for a contig on which the sample QC did not run.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...results.reader import Results

from .tables import contig_part, write_contig_tables


def contig_quality(
    output: str | Path,
    results: Results,
    contig: str,
    sample_ids: Sequence[str] | None = None,
    input_ids: Sequence[str] | None = None,
):
    """Collect one contig's recorded QC tables; absent QC contributes no rows or files."""
    import pandas as pd

    from ...contracts.samples import input_rows
    from . import evidence as evidence_reports

    report = results.evidence_summary(contig)
    if report is None:
        return contig, {}
    sample_ids = tuple(map(str, results.manifest["sample_ids"] if sample_ids is None else sample_ids))
    if input_ids is None:
        input_ids = results.input_sample_ids
    frames = {}
    for suffix, table in (("samples", evidence_reports.sample_table), ("segments", evidence_reports.segment_table)):
        value = table(report, sample_ids)
        if value is None:
            break
        header, rows = value
        if suffix == "samples":
            order = input_rows(sample_ids, input_ids)
            if order is not None and len(rows) == len(sample_ids):
                rows = [rows[index] for index in order]
        frames[suffix] = pd.DataFrame(rows, columns=header)
    return contig_part(output, "quality", contig, frames)


def write_quality(output, parts) -> list[Path]:
    """Publish QC CSVs once, with contigs in configured order."""
    return write_contig_tables(output, "quality", parts)
