"""Run-level recombination CSVs and maps.pdf from checked per-contig products.

Tables retain each contig's recorded bin/edge/call/locus order. Coverage is the
single callable-span representation: consecutive released markers in a followed
stretch. Figures retain the conditional rates and effective-meiosis staircase.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...contracts.recombination import Recombination, RecombinationMap
    from ...results.reader import Results

import numpy as np

from ...core.files import atomic_path
from .tables import contig_part, record_mapping, write_contig_tables

SPAN_COLUMNS = ("parent", "child", "parent_slot", "left_bp", "right_bp", "component_id")
"""Columns in published order.

    parent: sequenced parent identifier (text).
    child: child identifier (text).
    parent_slot: parental transmission slot, 0 or 1 (1).
    left_bp: left coordinate of this map bin, span or crossover interval; see the table definition (bp).
    right_bp: right coordinate of this map bin, span or crossover interval; see the table definition (bp).
    component_id: zero-based component index (1).
"""
CROSSOVER_COLUMNS = (
    "parent",
    "child",
    "parent_slot",
    "left_bp",
    "right_bp",
    "left_origin_probability",
    "right_origin_probability",
    "joint_endpoint_probability_lower_bound",
    "component_id",
    "reported_left_bp",
    "reported_right_bp",
    "in_reversed_segment",
)
"""Columns in published order.

    parent: sequenced parent identifier (text).
    child: child identifier (text).
    parent_slot: parental transmission slot, 0 or 1 (1).
    left_bp: left coordinate of this map bin, span or crossover interval; see the table definition (bp).
    right_bp: right coordinate of this map bin, span or crossover interval; see the table definition (bp).
    left_origin_probability: left origin probability (1).
    right_origin_probability: right origin probability (1).
    joint_endpoint_probability_lower_bound: max(0, left_origin_probability + right_origin_probability - 1) (1).
    component_id: zero-based component index (1).
    reported_left_bp: reported crossover interval left edge including the preceding chain bin (bp).
    reported_right_bp: reported crossover interval right edge including the following chain bin (bp).
    in_reversed_segment: in reversed segment (1).
"""


def map_table(curve: RecombinationMap) -> dict:
    """Columns of map.csv from a RecombinationMap."""
    bins = np.asarray(curve.edges_bp)
    table = {"left_bp": bins[:-1], "right_bp": bins[1:]}
    table.update(
        {
            field.name: getattr(curve, field.name)
            for field in fields(curve)
            if field.name not in ("edges_bp", "cumulative_observed_cM")
        }
    )
    table["cumulative_observed_cM_at_right"] = np.asarray(curve.cumulative_observed_cM)[1:]
    return table


def meiosis_tables(product: Recombination) -> tuple[list, list, list]:
    """(crossovers, coverage, meioses) rows of one chromosome's RecombinationProduct."""
    names = product.sample_ids
    crossovers: list[dict] = []
    coverage: list[dict] = []
    meioses: list[dict] = []
    for edge, (parent, child, slot) in enumerate(
        zip(product.edge_parent.tolist(), product.edge_child.tolist(), product.edge_slot.tolist(), strict=False)
    ):
        labels = {"parent": names[parent], "child": names[child], "parent_slot": int(slot)}
        calls = product.edge_crossovers(edge)
        first = product.crossover_offsets[edge]
        for index, (left, right, left_support, right_support, component) in enumerate(calls.tolist()):
            reported_left, reported_right = product.reported_intervals[first + index].tolist()
            crossovers.append(
                {
                    **labels,
                    "left_bp": int(left),
                    "right_bp": int(right),
                    "left_origin_probability": left_support,
                    "right_origin_probability": right_support,
                    "joint_endpoint_probability_lower_bound": max(0.0, left_support + right_support - 1),
                    "component_id": int(component),
                    "reported_left_bp": int(reported_left),
                    "reported_right_bp": int(reported_right),
                    "in_reversed_segment": bool(product.in_reversed_segment[first + index]),
                }
            )
        for left, right, component in product.edge_callable_spans(edge).tolist():
            coverage.append({**labels, "left_bp": int(left), "right_bp": int(right), "component_id": int(component)})
        counts = product.marker_counts[edge]
        meioses.append(
            {
                **labels,
                "released_chain_bins": int(counts[0]),
                "followed_chain_bins": int(counts[1]),
                "unresolved_chain_bins": int(counts[2]),
                "called_crossovers": len(calls),
                "expected_crossovers": float(product.expected_crossovers[edge].sum()),
                "exposure_meiosis_bp": float(product.exposure[edge].sum()),
            }
        )
    return crossovers, coverage, meioses


def map_figure(contig: str, curve: RecombinationMap):
    """One chromosome's unchanged rate/exposure figure, for a page of maps.pdf."""
    from matplotlib.figure import Figure

    bins = np.asarray(curve.edges_bp)
    figure = Figure(figsize=(9, 6))
    axes = figure.subplots(2, 1, sharex=True)
    centres = (bins[:-1] + bins[1:]) / 2e6
    axes[0].plot(centres, curve.rate_cM_per_Mb, color="#24557b", label="Expected (called endpoints)")
    axes[0].plot(centres, curve.called_rate_cM_per_Mb, color="#777777", linestyle="--", label="High-confidence calls")
    axes[0].legend()
    axes[0].set_ylabel("Conditional rate (cM/Mb)")
    axes[1].stairs(curve.effective_meioses, bins / 1e6, color="#3d7e55")
    axes[1].set_ylabel("Effective meioses")
    axes[1].set_xlabel("Position (Mb)")
    axes[0].set_title(f"{contig}: missing-aware recombination and exposure")
    figure.tight_layout()
    return figure


def contig_recombination(output: str | Path, results: Results, contig: str, *, elapsed_seconds: float | None = None):
    """Collect one checked chromosome's cells in parallel with the other chromosomes."""
    import pandas as pd

    product = results.recombination(contig)
    assert product is not None
    crossovers, coverage, meioses = meiosis_tables(product)
    junctions, members = junction_tables(product.junctions)
    frames = {
        "map": pd.DataFrame(map_table(product.map)),
        "crossovers": pd.DataFrame(crossovers, columns=CROSSOVER_COLUMNS),
        "meioses": pd.DataFrame(meioses, columns=MEIOSIS_COLUMNS),
        "coverage": pd.DataFrame(coverage, columns=SPAN_COLUMNS),
        "junctions": pd.DataFrame(junctions, columns=JUNCTION_COLUMNS),
        "junction_members": pd.DataFrame(members, columns=JUNCTION_MEMBER_COLUMNS),
        "summary": pd.DataFrame([{**record_mapping(product.summary), "elapsed_seconds": elapsed_seconds}]),
    }
    return contig_part(output, "recombination", contig, frames), product.map


def write_recombination(output, parts) -> list[Path]:
    """Publish seven run-level CSVs and one PDF page per contig, in configured order."""
    from matplotlib.backends.backend_pdf import PdfPages

    written = write_contig_tables(output, "recombination", [part for part, _ in parts])
    if parts:
        path = Path(output) / "recombination" / "maps.pdf"
        with atomic_path(path) as temporary:
            with PdfPages(temporary, metadata={"CreationDate": None}) as pages:
                for (contig, _), curve in parts:
                    pages.savefig(map_figure(contig, curve), dpi=150)
        written.append(path)
    return written


JUNCTION_COLUMNS = (
    "contig",
    "locus",
    "left_bp",
    "right_bp",
    "calls_left_bp",
    "calls_right_bp",
    "calls",
    "parents_with_calls",
    "odd_meioses",
    "decidable_meioses",
    "switch_fraction",
    "flip_adjusted_switch_fraction",
    "hotspot_ceiling",
    "mixed_parents",
    "all_or_none_parents",
    "parents",
    "children",
    "mixing_log_lr",
    "paired_locus",
    "paired_odd_meioses",
    "pairing_log_lr",
    "pairing_tests",
    "junction_log_lr",
    "flagged",
    "interpretation",
    "nearest_supplied_breakpoint",
    "breakpoint_distance_bp",
)
"""Columns in published order.

    contig: contig identifier (text).
    locus: locus (1).
    left_bp: left coordinate of this map bin, span or crossover interval; see the table definition (bp).
    right_bp: right coordinate of this map bin, span or crossover interval; see the table definition (bp).
    calls_left_bp: calls left bp (bp).
    calls_right_bp: calls right bp (bp).
    calls: calls (1).
    parents_with_calls: parents with calls (1).
    odd_meioses: odd meioses (1).
    decidable_meioses: decidable meioses (1).
    switch_fraction: switch fraction (1).
    flip_adjusted_switch_fraction: flip adjusted switch fraction (1).
    hotspot_ceiling: hotspot ceiling (1).
    mixed_parents: mixed parents (1).
    all_or_none_parents: all or none parents (1).
    parents: number of parent text entries in junction_members (count).
    children: number of child text entries in junction_members (count).
    mixing_log_lr: mixing log lr (nats).
    paired_locus: paired locus (1).
    paired_odd_meioses: paired odd meioses (1).
    pairing_log_lr: pairing log lr (nats).
    pairing_tests: pairing tests (1).
    junction_log_lr: junction log lr (nats).
    flagged: flagged (1).
    interpretation: interpretation (1).
    nearest_supplied_breakpoint: nearest supplied breakpoint (1).
    breakpoint_distance_bp: breakpoint distance bp (bp).
"""


# Published table schemas, including contigs that contribute no rows.
# Header spelling is normalized by the shared CSV writer.
MAP_COLUMNS = (
    "left_bp",
    "right_bp",
    "crossover_count",
    "exposure_meiosis_bp",
    "expected_crossovers",
    "called_exposure_meiosis_bp",
    "called_rate_cM_per_Mb",
    "covered_bp",
    "effective_meioses",
    "rate_cM_per_Mb",
    "cM_on_covered_bp",
    "cumulative_observed_cM_at_right",
)
"""Columns in published order.

    left_bp: left coordinate of this map bin, span or crossover interval; see the table definition (bp).
    right_bp: right coordinate of this map bin, span or crossover interval; see the table definition (bp).
    crossover_count: crossover count (count).
    exposure_meiosis_bp: informative transmission exposure (meiosis bp).
    expected_crossovers: expected crossovers (1).
    called_exposure_meiosis_bp: callable transmission exposure (meiosis bp).
    called_rate_cM_per_Mb: called rate c m per mb (1).
    covered_bp: covered bp (bp).
    effective_meioses: observable transmission exposure divided by covered bases (count).
    rate_cM_per_Mb: rate c m per mb (1).
    cM_on_covered_bp: c m on covered bp (bp).
    cumulative_observed_cM_at_right: cumulative observed genetic distance at the right bin edge (cM).
"""

MEIOSIS_COLUMNS = (
    "parent",
    "child",
    "parent_slot",
    "released_chain_bins",
    "followed_chain_bins",
    "unresolved_chain_bins",
    "called_crossovers",
    "expected_crossovers",
    "exposure_meiosis_bp",
)
"""Columns in published order.

    parent: sequenced parent identifier (text).
    child: child identifier (text).
    parent_slot: parental transmission slot, 0 or 1 (1).
    released_chain_bins: released chain bins (1).
    followed_chain_bins: followed chain bins (1).
    unresolved_chain_bins: unresolved chain bins (1).
    called_crossovers: called crossovers (1).
    expected_crossovers: expected crossovers (1).
    exposure_meiosis_bp: informative transmission exposure (meiosis bp).
"""

SUMMARY_COLUMNS = (
    "contig",
    "meioses",
    "observable_meioses",
    "crossovers",
    "expected_crossovers",
    "unresolved_bins",
    "exposure_meiosis_bp",
    "covered_bp",
    "cumulative_observed_cM",
    "uncovered_bins",
    "order_breakpoints",
    "order_breakpoint_masked_sites",
    "junction_loci",
    "junction_loci_flagged",
    "family_phase_converged",
    "elapsed_seconds",
)
"""Columns in published order.

    contig: contig identifier (text).
    meioses: meioses (1).
    observable_meioses: observable meioses (1).
    crossovers: crossovers (bp,bp,probability,probability,index).
    expected_crossovers: expected crossovers (1).
    unresolved_bins: unresolved bins (1).
    exposure_meiosis_bp: informative transmission exposure (meiosis bp).
    covered_bp: covered bp (bp).
    cumulative_observed_cM: cumulative observed c m (cM).
    uncovered_bins: uncovered bins (1).
    order_breakpoints: order breakpoints (1).
    order_breakpoint_masked_sites: order breakpoint masked sites (1).
    junction_loci: junction loci (1).
    junction_loci_flagged: junction loci flagged (1).
    family_phase_converged: family phase converged (1).
    elapsed_seconds: recorded runtime, empty when the run record does not supply it (s).
"""


JUNCTION_MEMBER_COLUMNS = ("contig", "locus", "column", "item", "value")
"""Columns in published order.

    contig: contig identifier (text).
    locus: parent junction locus key (text).
    column: parents or children, the original junction-table column (text).
    item: zero-based semicolon-delimited item position (1).
    value: original parent k/n entry or child identifier; join with semicolons to recover the cell (text).
"""


def junction_tables(loci):
    """Separate the two cohort-sized text lists; retain their exact order and text."""
    rows, members = [], []
    for locus in loci:
        row = record_mapping(locus)
        for column in ("parents", "children"):
            values = row[column].split(";") if row[column] else []
            row[column] = len(values)
            members.extend(
                dict(contig=row["contig"], locus=row["locus"], column=column, item=i, value=value)
                for i, value in enumerate(values)
            )
        rows.append(row)
    return rows, members
