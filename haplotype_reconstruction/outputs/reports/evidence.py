"""Format recorded sample QC diagnostics as the original fixed-decimal cells.

Owns P0 OU16 text only; no quality decisions or source reads occur here.
Method: docs/methods/evidence.md#sample-quality.
"""

from __future__ import annotations

import numpy as np

from ...contracts.evidence import EvidenceDecisions


def _cell(value, digits):
    return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def sample_table(report: EvidenceDecisions, sample_ids) -> tuple[list[str], list[list[str]]] | None:
    """Rows of OUTPUT/quality/<contig>.samples.tsv (the sample quality exclusion per sample), or None when
    the QC did not run on this contig."""
    samples = report.sample_quality
    if samples is None:
        return None
    header = list(SAMPLE_COLUMNS)
    rows = [
        [
            str(name),
            str(int(samples.observed_cells[i])),
            str(int(samples.eligible_cells[i])),
            _cell(samples.eligible_windows[i], 1),
            str(int(samples.minor_cells[i])),
            str(int(samples.near_private_cells[i])),
            _cell(samples.lrt[i], 2),
            _cell(samples.foreign_fraction[i], 4),
            _cell(samples.foreign_share[i], 5),
            _cell(samples.artefact_share[i], 5),
            _cell(samples.artefact_peak[i], 4),
            _cell(samples.heterozygote_fit[i], 3),
            _cell(samples.heterozygote_dispersion[i], 4),
            _cell(samples.error_fit[i], 6),
            str(int(samples.flagged[i])),
        ]
        for i, name in enumerate(sample_ids)
    ]
    return header, rows


def segment_table(report: EvidenceDecisions, sample_ids) -> tuple[list[str], list[list[str]]] | None:
    """Rows of OUTPUT/quality/<contig>.segments.tsv (inherited-segment regions), or None when the QC
    did not run on this contig."""
    samples = report.sample_quality
    if samples is None:
        return None
    header = list(SEGMENT_COLUMNS)
    rows = [
        [
            str(sample_ids[row.sample]),
            "-" if row.partner < 0 else str(sample_ids[row.partner]),
            str(row.start),
            str(row.end),
            f"{row.events:.1f}",
            str(row.cells),
            f"{row.minor_read_fraction:.3f}",
            _cell(row.partner_minor_read_fraction, 3),
        ]
        for row in samples.segments
    ]
    return header, rows


# Published column schemas: B reports and P0 outputs inventory D6 / OU15.
# Row-dict tables retain B's headerless empty constructor; these constants do not
# impose columns on those tables. Scientific values remain upstream contract fields.
SAMPLE_COLUMNS = (
    "sample",
    "observed_cells",
    "eligible_cells",
    "eligible_windows",
    "minor_cells",
    "near_private_cells",
    "lrt",
    "foreign_fraction",
    "foreign_share",
    "artefact_share",
    "artefact_peak",
    "heterozygote_fit",
    "heterozygote_dispersion",
    "error_fit",
    "flagged",
)
"""Columns in published order.

    sample: sample identifier (text).
    observed_cells: observed cells (count).
    eligible_cells: eligible cells (count).
    eligible_windows: eligible windows (1).
    minor_cells: minor cells (count).
    near_private_cells: near private cells (count).
    lrt: lrt (nats).
    foreign_fraction: foreign fraction (1).
    foreign_share: foreign share (1).
    artefact_share: artefact share (1).
    artefact_peak: artefact peak (1).
    heterozygote_fit: heterozygote fit (1).
    heterozygote_dispersion: heterozygote dispersion (1).
    error_fit: error fit (1).
    flagged: flagged (1).
"""

SEGMENT_COLUMNS = (
    "sample",
    "partner",
    "start",
    "end",
    "events",
    "cells",
    "minor_read_fraction",
    "partner_minor_read_fraction",
)
"""Columns in published order.

    sample: sample identifier (text).
    partner: QC partner identifier; - when absent (text).
    start: inclusive segment start (bp).
    end: exclusive segment end (bp).
    events: events (1).
    cells: cells (1).
    minor_read_fraction: minor read fraction (1).
    partner_minor_read_fraction: partner minor read fraction (1).
"""
