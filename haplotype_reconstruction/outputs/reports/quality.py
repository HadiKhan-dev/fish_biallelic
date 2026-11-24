"""OUTPUT/quality/samples.csv, segments.csv and flagged_libraries.csv: the per-contig sample quality exclusion.

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


FLAGGED_COLUMNS = (
    "sample",
    "contigs_flagged",
    "contigs_assessed",
    "max_foreign_fraction",
    "at_ceiling",
)
"""Columns of OUTPUT/quality/flagged_libraries.csv, one row per library flagged on at least one contig.

    sample: sample identifier (text).
    contigs_flagged: contigs on which the sample QC flagged the library (count).
    contigs_assessed: contigs on which the sample QC ran for the library (count).
    max_foreign_fraction: largest fitted foreign fraction over the flagged contigs (1).
    at_ceiling: 1 when that fraction sits at the model ceiling, so the library's contamination is at least it (1).

A sample QC flag keeps the library out of founder construction. The separate contamination
decision governs likelihood correction and downstream tiers from the repaint onward.
"""


def flagged_libraries(samples):
    """Rows of flagged_libraries.csv from the concatenated samples table (a DataFrame), or None when nothing is
    flagged."""
    if samples is None or samples.empty or "flagged" not in samples:
        return None
    flags = samples.assign(flagged=samples["flagged"].astype(str) == "1")
    hit = flags[flags["flagged"]]
    if hit.empty:
        return None
    rows = []
    for name, group in hit.groupby("sample", sort=True):
        assessed = int((flags["sample"] == name).sum())
        fractions = group["foreign_fraction"].astype(float)
        top = int(fractions.idxmax())
        ceiling = str(group.loc[top, "foreign_fraction_at_ceiling"]) if "foreign_fraction_at_ceiling" in group else "NA"
        rows.append([str(name), str(len(group)), str(assessed), f"{fractions.max():.4f}", ceiling])
    return rows


def write_quality(output, parts) -> list[Path]:
    """Publish QC CSVs once, with contigs in configured order, and the per-library flag summary."""
    import pandas as pd

    from .tables import write_rows

    paths = write_contig_tables(output, "quality", parts)
    samples_path = next((Path(path) for path in paths if Path(path).name == "samples.csv"), None)
    if samples_path is not None and samples_path.exists():
        rows = flagged_libraries(pd.read_csv(samples_path, dtype=str))
        if rows:
            target = samples_path.with_name("flagged_libraries.csv")
            write_rows(target, [dict(zip(FLAGGED_COLUMNS, row, strict=True)) for row in rows], columns=FLAGGED_COLUMNS)
            paths = [*paths, target]
    return paths


CONTAMINATION_COLUMNS = (
    "sample",
    "alpha",
    "own_error",
    "alpha_lower_bound",
    "z",
    "corrected",
    "donor",
    "correction_alpha",
    "donor_residual",
    "tier",
)
"""Columns of OUTPUT/quality/contamination.csv, one row per library (4.22 QC A).

    sample: sample identifier (text).
    alpha: estimated read fraction of foreign DNA under the cohort-frequency law (1).
    own_error: the library's own read error fitted with alpha (1).
    alpha_lower_bound: 1 when no donor is named and own_error sits at its grid top: foreign reads were absorbed as
        error and alpha understates the contamination (the library is marked interpret_with_care) (1).
    z: robust z of alpha over the cohort (1).
    corrected: 1 when the library's likelihoods are corrected from the repaint on (alpha and z both above their lines).
    donor: the sequenced library named as the source of the foreign reads, empty when none (text).
    correction_alpha: the foreign read fraction the correction uses: the donor law's when a donor is named (1).
    donor_residual: the best candidate donor's same-homozygote residual; a donor is named at or below its line (1).
    tier: clean, corrected, or interpret_with_care (corrected above the care line, or alpha a lower bound) (text).
"""


def contamination_rows(decision) -> list[list[str]]:
    """Rows of contamination.csv from the run's decision (results.Results.contamination)."""
    sample_ids = [str(s) for s in decision["sample_ids"]]
    donor = [int(x) for x in decision["donor"]]
    rows = []
    for i, name in enumerate(sample_ids):
        flagged = bool(decision["flagged"][i])
        alpha = float(decision["donor_alpha"][i] if donor[i] >= 0 else decision["alpha"][i])
        residual = float(decision["residual"][i])
        care = bool(decision["interpret_with_care"][i])
        bound = bool(decision["alpha_lower_bound"][i]) if "alpha_lower_bound" in decision else False
        rows.append([
            name, f"{float(decision['alpha'][i]):.5f}", f"{float(decision['error'][i]):.2e}", "1" if bound else "0",
            f"{float(decision['z'][i]):.2f}", "1" if flagged else "0",
            sample_ids[donor[i]] if donor[i] >= 0 else "", f"{alpha:.5f}" if flagged else "",
            "" if residual != residual else f"{residual:.4f}",
            "interpret_with_care" if care else "corrected" if flagged else "clean",
        ])
    return rows


def write_contamination(output, results) -> list[Path]:
    """Publish quality/contamination.csv from the run's contamination decision (nothing before it exists)."""
    from .tables import write_rows

    decision = results.contamination()
    if decision is None:
        return []
    target = Path(output) / "quality" / "contamination.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = contamination_rows(decision)
    write_rows(target, [dict(zip(CONTAMINATION_COLUMNS, row, strict=True)) for row in rows],
               columns=CONTAMINATION_COLUMNS)
    return [target]
