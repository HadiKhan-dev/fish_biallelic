"""OUTPUT/quality/<contig>.samples.tsv and <contig>.segments.tsv: the per-contig library-contamination QC.

The rows are the evidence stage's (evidence.reports.sample_table / segment_table): per sample the contamination
statistics and whether its reads were kept out of founder construction on this contig, and the inherited-segment
regions shared by pairs of samples. Nothing is written for a contig on which the sample QC did not run.
"""
from __future__ import annotations

from pathlib import Path

from .tables import write_text_table


def write_quality(output, store, contig: str, sample_ids) -> list[Path]:
    """Write the contig's QC tables from its evidence record; returns the paths written (none without QC)."""
    from ..evidence import records as evidence_records
    from ..evidence import reports as evidence_reports
    from ..evidence.sample_order import input_rows, store_input_order
    report = evidence_records.load_report(store.contig(contig))
    sample_ids = tuple(map(str, sample_ids))
    directory = Path(output) / "quality"
    written = []
    for suffix, table in (("samples", evidence_reports.sample_table), ("segments", evidence_reports.segment_table)):
        result = table(report, sample_ids)
        if result is None:
            return written
        header, rows = result
        if suffix == "samples":
            # One row per sample in axis order: list them in the run's input order (release 2.1 records are canonical).
            order = input_rows(sample_ids, store_input_order(store))
            if order is not None and len(rows) == len(sample_ids):
                rows = [rows[index] for index in order]
        written.append(write_text_table(directory / f"{contig}.{suffix}.tsv", header, rows))
    return written
