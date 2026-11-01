"""OUTPUT/quality/<contig>.samples.tsv and <contig>.segments.tsv: the per-contig sample quality exclusion.

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

from .tables import write_text_table


def write_quality(
    output: str | Path,
    results: Results,
    contig: str,
    sample_ids: Sequence[str] | None = None,
    input_ids: Sequence[str] | None = None,
) -> list[Path]:
    """Write recorded QC tables through results; no sample table means no files."""
    from ...contracts.samples import input_rows
    from . import evidence as evidence_reports

    report = results.evidence_summary(contig)
    if report is None:
        return []
    sample_ids = tuple(map(str, results.manifest["sample_ids"] if sample_ids is None else sample_ids))
    if input_ids is None:
        input_ids = results.input_sample_ids
    directory = Path(output) / "quality"
    written: list[Path] = []
    for suffix, table in (("samples", evidence_reports.sample_table), ("segments", evidence_reports.segment_table)):
        value = table(report, sample_ids)
        if value is None:
            return written
        header, rows = value
        if suffix == "samples":
            order = input_rows(sample_ids, input_ids)
            if order is not None and len(rows) == len(sample_ids):
                rows = [rows[index] for index in order]
        written.append(write_text_table(directory / f"{contig}.{suffix}.tsv", header, rows))
    return written
