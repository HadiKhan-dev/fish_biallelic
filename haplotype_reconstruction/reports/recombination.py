"""OUTPUT/recombination/: readable recombination maps, calls, exposure and junction loci.

Per contig (from its recombination record):
  <contig>.map.csv              per map bin [left_bp, right_bp): called and expected crossovers, observable
                                exposure, rates (cM/Mb), effective meioses, cumulative observed cM
  <contig>.crossovers.csv       called crossover intervals per meiosis (parent, child, parent slot): (left_bp,
                                right_bp] between flanking informative markers, origin probabilities, assembly
                                component, the bin-honest reported interval, reversed-segment flag
  <contig>.coverage.csv         informative spans per meiosis; <contig>.called_coverage.csv: spans between
                                supported markers
  <contig>.meioses.csv          per meiosis: marker counts, calls, expected crossovers, exposure
  <contig>.junctions.csv        junction loci (overlapping reported calls) and their flags
  <contig>.png                  conditional rate and effective meioses along the chromosome
Genome-wide: junctions.csv (every contig's loci) and summary.csv (one row per contig). Maps are conditional on
the final phase and the fixed inferred pedigree.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from .tables import write_rows

SPAN_COLUMNS = ("parent", "child", "parent_slot", "left_bp", "right_bp", "component_id")
CROSSOVER_COLUMNS = ("parent", "child", "parent_slot", "left_bp", "right_bp", "left_origin_probability",
                     "right_origin_probability", "joint_endpoint_probability_lower_bound", "component_id",
                     "reported_left_bp", "reported_right_bp", "in_reversed_segment")
PLOT_BYTES = 128 << 20        # one figure (and matplotlib on first use)


def _locus_columns() -> tuple[str, ...]:
    from ..recombination.junctions import JunctionLocus
    return tuple(field.name for field in fields(JunctionLocus))


def map_table(curve) -> dict:
    """Columns of <contig>.map.csv from a RecombinationMap."""
    bins = np.asarray(curve.edges_bp)
    table = {"left_bp": bins[:-1], "right_bp": bins[1:]}
    table.update({field.name: getattr(curve, field.name) for field in fields(curve)
                  if field.name not in ("edges_bp", "cumulative_observed_cM")})
    table["cumulative_observed_cM_at_right"] = np.asarray(curve.cumulative_observed_cM)[1:]
    return table


def meiosis_tables(product) -> tuple[list, list, list, list]:
    """(crossovers, coverage, called_coverage, meioses) rows of one chromosome's RecombinationProduct."""
    names = product.sample_ids
    crossovers, coverage, called_coverage, meioses = [], [], [], []
    for edge, (parent, child, slot) in enumerate(zip(product.edge_parent.tolist(), product.edge_child.tolist(),
                                                     product.edge_slot.tolist())):
        labels = {"parent": names[parent], "child": names[child], "parent_slot": int(slot)}
        calls = product.edge_crossovers(edge)
        first = product.crossover_offsets[edge]
        for index, (left, right, left_support, right_support, component) in enumerate(calls.tolist()):
            reported_left, reported_right = product.reported_intervals[first + index].tolist()
            crossovers.append({**labels, "left_bp": int(left), "right_bp": int(right),
                               "left_origin_probability": left_support,
                               "right_origin_probability": right_support,
                               "joint_endpoint_probability_lower_bound": max(0., left_support + right_support - 1),
                               "component_id": int(component), "reported_left_bp": int(reported_left),
                               "reported_right_bp": int(reported_right),
                               "in_reversed_segment": bool(product.in_reversed_segment[first + index])})
        for rows, spans in ((coverage, product.edge_informative_spans(edge)),
                            (called_coverage, product.edge_callable_spans(edge))):
            for left, right, component in spans.tolist():
                rows.append({**labels, "left_bp": int(left), "right_bp": int(right), "component_id": int(component)})
        counts = product.marker_counts[edge]
        meioses.append({**labels, "informative_markers": int(counts[0]), "supported_origin_markers": int(counts[1]),
                        "orientation_artifact_uncertain_markers": int(counts[2]), "called_crossovers": len(calls),
                        "expected_crossovers": float(product.expected_crossovers[edge].sum()),
                        "exposure_meiosis_bp": float(product.exposure[edge].sum())})
    return crossovers, coverage, called_coverage, meioses


def write_plot(path: Path, contig: str, curve) -> Path:
    """<contig>.png: posterior-mean and called rates, and effective meioses (no global plotting state)."""
    from matplotlib.figure import Figure
    bins = np.asarray(curve.edges_bp)
    figure = Figure(figsize=(9, 6))
    axes = figure.subplots(2, 1, sharex=True)
    centres = (bins[:-1] + bins[1:]) / 2e6
    axes[0].plot(centres, curve.rate_cM_per_Mb, color="#24557b", label="Posterior mean")
    axes[0].plot(centres, curve.called_rate_cM_per_Mb, color="#777777", linestyle="--",
                 label="High-confidence calls")
    axes[0].legend()
    axes[0].set_ylabel("Conditional rate (cM/Mb)")
    axes[1].stairs(curve.effective_meioses, bins / 1e6, color="#3d7e55")
    axes[1].set_ylabel("Effective meioses")
    axes[1].set_xlabel("Position (Mb)")
    axes[0].set_title(f"{contig}: missing-aware recombination and exposure")
    figure.tight_layout()
    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(temporary, format="png", dpi=150)
    temporary.replace(path)
    return path


def write_chromosome(directory: Path, product) -> list[Path]:
    """Every per-contig file of one RecombinationProduct."""
    import pandas as pd
    from .tables import write_frame
    contig = product.contig
    crossovers, coverage, called_coverage, meioses = meiosis_tables(product)
    written = [write_frame(directory / f"{contig}.map.csv", pd.DataFrame(map_table(product.map))),
               write_rows(directory / f"{contig}.meioses.csv", meioses),
               write_rows(directory / f"{contig}.crossovers.csv", crossovers, columns=CROSSOVER_COLUMNS),
               write_rows(directory / f"{contig}.junctions.csv", product.junctions, columns=_locus_columns()),
               write_rows(directory / f"{contig}.coverage.csv", coverage, columns=SPAN_COLUMNS),
               write_rows(directory / f"{contig}.called_coverage.csv", called_coverage, columns=SPAN_COLUMNS),
               write_plot(directory / f"{contig}.png", contig, product.map)]
    return written


def contig_bytes(file) -> int:
    """Planned peak bytes of one contig's files: the decoded record three times (product, row tables, frames)
    and one figure (measured 34 MiB in all, matplotlib's import included, on std_n80 chr1)."""
    from ..recombination import records as recombination_records
    return 3 * file.info(recombination_records.KEY).decoded_bytes + PLOT_BYTES


def write_recombination(output, store, contigs) -> list[Path]:
    """Write every contig's files and the genome-wide junction and summary tables (from the records); each
    contig holds a reservation of the run's memory while it is written (contig_bytes)."""
    from ..execution.memory import reserve_memory
    from ..recombination import records as recombination_records
    directory = Path(output) / "recombination"
    directory.mkdir(parents=True, exist_ok=True)
    written, loci, summaries = [], [], []
    for contig in contigs:
        file = store.contig(contig)
        with reserve_memory(contig_bytes(file), label="recombination report"):
            product = recombination_records.load(file)
            written.extend(write_chromosome(directory, product))
            loci.extend(asdict(locus) for locus in product.junctions)
            summaries.append(product.summary)
            del product
    written.append(write_rows(directory / "junctions.csv", loci, columns=_locus_columns()))
    written.append(write_rows(directory / "summary.csv", summaries))
    print(f"[recombination] maps written to {directory}", flush=True)
    return written
