"""What the evidence stage decided on a contig, in the forms its readers need.

``EvidenceReport`` is stored with the evidence (record ``evidence``, item ``report``);
``EvidenceSummary`` is the small JSON-compatible outcome for logs and the run record; the table
functions give the rows of the published QC tables (written by ``reports/``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .read_model import calibrate as calibration
from .site_quality import SiteQualityResult


@dataclass(frozen=True)
class EvidenceReport:
    """What the stage decided on one contig (stored with the evidence; QC reports read it).

    input_positions: int64[L_in] the ContigReads sites (``site_quality.rule_bits`` indexes them).
    dropped: loader records dropped by reason. timings: wall seconds of the steps.
    """

    contig: str
    input_positions: np.ndarray
    dropped: Mapping[str, int]
    site_quality: SiteQualityResult
    calibration: calibration.CalibrationReport
    timings: Mapping[str, float]

    def record(self) -> dict:
        return dict(contig=self.contig, input_positions=self.input_positions,
                    dropped=dict(self.dropped), site_quality=self.site_quality.record(),
                    calibration=self.calibration.record(), timings=dict(self.timings))

    @classmethod
    def from_record(cls, record: Mapping, kept: np.ndarray) -> "EvidenceReport":
        return cls(contig=record["contig"], input_positions=record["input_positions"],
                   dropped=record["dropped"],
                   site_quality=SiteQualityResult.from_record(record["site_quality"], kept),
                   calibration=calibration.CalibrationReport.from_record(record["calibration"]),
                   timings=record["timings"])


@dataclass(frozen=True)
class EvidenceSummary:
    """Small JSON-compatible outcome of one contig (logs, run record)."""

    contig: str
    resumed: bool
    input_sites: int
    sites: int
    dropped: Mapping[str, int]
    site_filter: str
    flagged_samples: tuple[str, ...]
    read_model: str
    timings: Mapping[str, float]
    # In-memory evidence for the next stage (run_contig(..., keep=True)); never stored.
    prepared: object = None
    read_model_warning: str | None = None

    def record(self) -> dict:
        return dict(contig=self.contig, resumed=self.resumed, input_sites=self.input_sites,
                    sites=self.sites, dropped=dict(self.dropped), site_filter=self.site_filter,
                    flagged_samples=list(self.flagged_samples), read_model=self.read_model,
                    timings=dict(self.timings),
                    **({} if self.read_model_warning is None else dict(
                        read_model_warning=self.read_model_warning)))


def summarise(report: EvidenceReport, evidence_sites: int, sample_ids, *, resumed: bool) -> EvidenceSummary:
    site = report.site_quality
    if not site.enabled:
        site_filter = "off"
    elif not site.applied:
        site_filter = f"not applied ({site.reason})"
    else:
        site_filter = "applied"
    flagged = () if site.sample_quality is None else tuple(
        sample_ids[index] for index in np.flatnonzero(site.sample_quality.flagged))
    calibration_report = report.calibration
    read_model = (calibration_report.selected if calibration_report.applied
                  else f"fixed ({calibration_report.fallback_reason})")
    return EvidenceSummary(contig=report.contig, resumed=resumed,
                           input_sites=int(len(report.input_positions)), sites=int(evidence_sites),
                           dropped=dict(report.dropped), site_filter=site_filter,
                           flagged_samples=flagged, read_model=read_model,
                           timings=dict(report.timings),
                           read_model_warning=calibration_report.shared_fit_warning)


def log_lines(summary: EvidenceSummary, report: EvidenceReport) -> list[str]:
    """Human-readable lines of one contig's evidence (printed by run_contig)."""
    dropped = summary.dropped
    lines = [f"    [Evidence] {summary.contig}: {summary.input_sites:,} biallelic SNPs"
             + (f" ({', '.join(f'{count:,} {reason}' for reason, count in dropped.items() if count)}"
                " records dropped)" if any(dropped.values()) else "")
             + f"; site filter {summary.site_filter}, {summary.sites:,} sites kept"]
    site = report.site_quality
    if site.applied:
        parameters = site.parameters
        error, balance = parameters['error'], parameters['balance']
        # A rule with no eligible sites records why instead of fitted parameters.
        error_text = (f"clean error {error['eps0']:.2e} (r3 applied {error['applied']})" if 'eps0' in error
                      else f"r3 not fitted ({error['reason']})")
        balance_text = (f"clean balance {balance['h0']:.4f} (r4 applied {balance['applied']})"
                        if 'reason' not in balance
                        else f"r4 not fitted ({balance['reason']}; QC balance {balance['h0']:.4f})")
        lines.append(
            f"    [Site quality] {summary.contig}: "
            + ", ".join(f"{rule} {count:,}" for rule, count in site.flagged.items())
            + f"; depth window [{parameters['depth']['low']:.0f}, {parameters['depth']['high']:.0f}], "
            f"{error_text}, {balance_text}")
    samples = site.sample_quality
    if samples is not None:
        lines.append(f"    [Sample quality] {summary.contig}: contamination flagged "
                     f"{len(summary.flagged_samples)} (z > {samples.config['flag_robust_z']})"
                     + (f": {', '.join(summary.flagged_samples)}" if summary.flagged_samples else "")
                     + ("" if samples.applied else f" (not applied: {samples.reason})")
                     + f"; {len(samples.segments)} inherited-segment regions reported")
    calibration_report = report.calibration
    if calibration_report.applied:
        lines.append(f"    [Read model] {summary.contig}: {calibration_report.selected}; selection gain "
                     f"over shared {calibration_report.selection_gain_over_shared:.3f}; diagnostic gain "
                     + ("unavailable (no admitted diagnostic-fold observations)"
                        if calibration_report.diagnostic_gain_over_shared is None
                        else f"{calibration_report.diagnostic_gain_over_shared:.3f}"))
    else:
        lines.append(f"    [Read model] {summary.contig}: fixed model ({calibration_report.fallback_reason})")
    if calibration_report.shared_fit_warning is not None:
        lines.append(f"    [Read model WARNING] {summary.contig}: "
                     f"{calibration_report.shared_fit_warning}; endpoints and distances in evidence report")
    return lines


def _cell(value, digits):
    return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def sample_table(report: EvidenceReport, sample_ids) -> tuple[list[str], list[list[str]]] | None:
    """Rows of OUTPUT/quality/<contig>.samples.tsv (the contamination QC per sample), or None when
    the QC did not run on this contig."""
    samples = report.site_quality.sample_quality
    if samples is None:
        return None
    header = ["sample", "observed_cells", "near_private_cells", "private_cells", "private_events",
              "low_fraction_events", "fitted_low_fraction", "private_minor_read_fraction",
              "heterozygous_like_events", "statistic", "robust_z", "flagged"]
    rows = [[str(name), str(int(samples.observed_cells[i])), str(int(samples.near_private_cells[i])),
             str(int(samples.private_cells[i])), _cell(samples.private_events[i], 1),
             _cell(samples.low_fraction_events[i], 2), _cell(samples.fitted_low_fraction[i], 4),
             _cell(samples.private_minor_read_fraction[i], 3),
             _cell(samples.heterozygous_like_events[i], 1), _cell(samples.statistic[i], 5),
             _cell(samples.robust_z[i], 3), str(int(samples.flagged[i]))]
            for i, name in enumerate(sample_ids)]
    return header, rows


def segment_table(report: EvidenceReport, sample_ids) -> tuple[list[str], list[list[str]]] | None:
    """Rows of OUTPUT/quality/<contig>.segments.tsv (inherited-segment regions), or None when the QC
    did not run on this contig."""
    samples = report.site_quality.sample_quality
    if samples is None:
        return None
    header = ["sample", "partner", "start", "end", "events", "cells", "minor_read_fraction",
              "partner_minor_read_fraction"]
    rows = [[str(sample_ids[row.sample]), "-" if row.partner < 0 else str(sample_ids[row.partner]),
             str(row.start), str(row.end), f"{row.events:.1f}", str(row.cells),
             f"{row.minor_read_fraction:.3f}", _cell(row.partner_minor_read_fraction, 3)]
            for row in samples.segments]
    return header, rows
