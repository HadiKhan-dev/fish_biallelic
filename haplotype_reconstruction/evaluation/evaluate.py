"""Evaluate a simulation run's products against its known truth (`haplotypes evaluate RUN`).

Truth is used only here, after inference; nothing is fed back. Every product is read through its stage's records
API. Outputs, RUN/evaluation/:

  summary.json             seed, samples, pedigree counts, per-chromosome, phase-group and founder rows, totals, notes
  chromosomes.csv          per chromosome: genotype and released-phase counts (family phase, all samples) and map
                           (recombination) counts
  phase_groups.csv         per chromosome and parent group (all, parents0/1/2: released pedigree edges into the
                           sample): genotype and released-phase counts (evaluation.metrics)
  founders.csv             per chromosome and founder stage: panel-vs-truth counts
  founder_components.csv   per component (or block) of every founder stage
  founder_matches.csv      the one-to-one row-to-founder matches behind founder_components

``evaluate_run(..., destination=D)`` writes these files to D instead (the run directory is then only read).

Founder stages (pipeline order, the stage vocabulary): discovery blocks, the three local-selection rounds, the
final assembly's levels 1-4, founder refinement (the final assembly product), painting (relabelled panels) and
founder refit. ``available`` evaluates the latest founder product of each contig plus whatever downstream
products exist; ``all`` every founder product present; explicit stages require their products.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.pedigree import ParentEdges, PedigreeCalls
from ..contracts.recombination import Recombination
from ..core.errors import InputError
from ..results.panels import FOUNDER_STAGES, StagePanel
from ..simulation.truth import ContigTruth, PedigreeTable, TruthCrossovers
from .metrics import SAMPLE_COLUMNS
from .parameters import PHASE_METRIC
from .tasks import ChromosomeEvaluation, ChromosomeInput, EvaluationReader, EvaluationTask, MetricRow, PhaseArrays

EVALUATION_STAGES = (*FOUNDER_STAGES, "pedigree", "family_phase", "recombination")
# Released-phase counts (release 3.2, evaluation.metrics): summed over samples per chromosome (all samples) and
# per parent group; the N50 columns and phased_share are derived from them.
PHASE_COLUMNS = SAMPLE_COLUMNS

# Parent groups: the number of released (Tier-B supported) pedigree edges into the sample, 0, 1 or 2.
PARENT_GROUPS = ("parents0", "parents1", "parents2")

# Reported-interval map columns, written after every earlier column.
REPORTED_INTERVAL_COLUMNS = (
    "reported_intervals_containing_true_crossover",
    "reported_interval_hit_rate",
    "called_interval_median_width_bp",
    "reported_interval_median_width_bp",
)
PARENT_STATES = ("zero_observed_parents", "one_observed_parent", "two_observed_parents")
INTERPRETATION = (
    "Truth is used only for evaluation. Founder labels match over whole components, "
    "not marker-wise; nearest-row errors and one-to-one completeness are separate. "
    "Represented ancestry means present in a sampled homologue, not independently identifiable. "
    "Phase is scored per sample and released phase set (a component split at the sample's released phase "
    "breaks) over correctly called true heterozygotes released phased; genotype errors are counted "
    "separately, never as phase errors. Map truth uses correct edges "
    "within inferred observable spans. Unexecuted stages are not evaluated."
)
SEGMENT_INTERPRETATION = (
    "set_aligned_errors are sites on a set's minority orientation, switch_errors orientation changes "
    "between consecutive scored sites of a set, long_switches orientation changes between consecutive "
    "orientation runs of >= 20 scored sites. segment_allele_errors "
    "label founder rows segment-wise (a label switch costs 20 mismatches); "
    "chimeric_segment_charges = called_allele_errors - segment_allele_errors. "
    "Component-wise founder counts can fall when a component is cut (labels re-match on each piece); "
    "split_charged_segment_cost charges each component boundary as one label switch per row and never falls. "
    "called_interval_hit_rate is the fraction of called crossover intervals on correct "
    "edges that contain a true crossover. reported_interval_hit_rate applies the same "
    "rule to the bin-honest reported intervals (reported_left_bp, reported_right_bp]; "
    "the median widths (right - left) of both kinds of interval give its context."
)
# Per-task allowance of the chromosome estimate besides the decoded records: decode buffers and the kernels a
# worker loads in its first task. Calibrated on std_n80 a1: measured peak per chromosome at most 0.34 GiB for
# estimates of 0.40-0.45 GiB.


def pedigree_counts(
    sample_ids: Sequence[str], calls: PedigreeCalls, truth: PedigreeTable, names: Sequence[str]
) -> MetricRow:
    """Tier-B calls vs the true pedigree of the analysed samples.

    ``sample_ids``/``calls``: pedigree.records.load_calls (PedigreeCalls: state -1 unresolved, parents -1 none);
    ``truth``: the simulation manifest's PedigreeTable of the analysed samples (parents outside the analysed
    samples are not observable edges); ``names``: the analysed sample axis. The calls' axis may be any order of
    ``names`` (release 2.1 records hold the run's canonical sample order); edges are matched by sample name. Every
    analysed sample is an inference target (release 3.4: no metadata eligibility), so every sample is counted.
    """
    sample_ids = tuple(map(str, sample_ids))
    if len(set(sample_ids)) != len(sample_ids) or set(sample_ids) != set(names) or len(sample_ids) != len(names):
        raise InputError("inferred and true sample axes differ")
    true_parents = {
        sample: (first, second)
        for sample, first, second in zip(truth.samples, truth.parent1, truth.parent2, strict=False)
    }
    known = set(names)
    exact = correct_states = true_edges = inferred_edges = shared_edges = 0
    expected_states = {state: 0 for state in PARENT_STATES}
    for sample, state, parents in zip(
        sample_ids, np.asarray(calls.state).tolist(), np.asarray(calls.parents).tolist(), strict=False
    ):
        expected = set(true_parents[sample]) & known
        actual = {sample_ids[parent] for parent in parents if parent >= 0}
        called_state = PARENT_STATES[state] if state >= 0 else None
        expected_state = PARENT_STATES[len(expected)]
        expected_states[expected_state] += 1
        exact += int(expected == actual and called_state == expected_state)
        correct_states += int(called_state == expected_state)
        true_edges += len(expected)
        inferred_edges += len(actual)
        shared_edges += len(expected & actual)
    result: MetricRow = dict(
        exact_configurations=exact,
        correct_parent_states=correct_states,
        expected_states=expected_states,
        true_edges=true_edges,
        correct_edges=shared_edges,
        extra_edges=inferred_edges - shared_edges,
        missing_edges=true_edges - shared_edges,
    )
    return result


def map_counts(product: Recombination, events: dict[tuple[str, str], np.ndarray], names: Sequence[str]) -> MetricRow:
    """Recombination calls of one chromosome vs the true crossovers of its meioses.

    ``product``: recombination.records RecombinationProduct; ``events``: {(parent, child): sorted true crossover
    positions (bp)}. Only edges of the true pedigree count; an interval (left, right] localizes a true crossover
    when it contains one. Edge indices refer to the product's own sample axis (any order of ``names``).
    """
    if getattr(product, "sample_ids", None) is not None:
        if set(map(str, product.sample_ids)) != set(names):
            raise InputError("recombination and true sample axes differ")
        names = tuple(map(str, product.sample_ids))
    true_observed = unmatched = 0
    expected = exposure = 0.0
    called = called_hits = reported_hits = 0
    called_widths, reported_widths = [], []
    for edge, (parent, child) in enumerate(zip(product.edge_parent, product.edge_child, strict=False)):
        raw = events.get((names[int(parent)], names[int(child)]))
        if raw is None:
            unmatched += 1
            continue
        for span in product.edge_callable_spans(edge):
            left, right = span[:2]
            true_observed += int(np.searchsorted(raw, right, side="right") - np.searchsorted(raw, left, side="right"))
        expected += float(product.expected_crossovers[edge].sum())
        exposure += float(product.exposure[edge].sum())
        reported = product.edge_reported_intervals(edge)
        for index, interval in enumerate(product.edge_crossovers(edge)):
            left, right = interval[:2]
            called += 1
            called_hits += int(np.searchsorted(raw, right, side="right") > np.searchsorted(raw, left, side="right"))
            called_widths.append(float(right - left))
            # Same rule on the reported interval (low, high]: a change placed at a
            # painting-bin edge can lie anywhere in the bins around that edge.
            low, high = reported[index]
            reported_hits += int(np.searchsorted(raw, high, side="right") > np.searchsorted(raw, low, side="right"))
            reported_widths.append(float(high - low))
    return dict(
        true_crossovers_in_inferred_exposure=true_observed,
        expected_crossovers_on_correct_edges=expected,
        correct_edge_exposure_meiosis_bp=exposure,
        inferred_edges_absent_from_truth=unmatched,
        called_crossovers_on_correct_edges=called,
        called_intervals_containing_true_crossover=called_hits,
        called_interval_hit_rate=called_hits / called if called else None,
        reported_intervals_containing_true_crossover=reported_hits,
        reported_interval_hit_rate=reported_hits / called if called else None,
        called_interval_median_width_bp=float(np.median(called_widths)) if called_widths else None,
        reported_interval_median_width_bp=(float(np.median(reported_widths)) if reported_widths else None),
    )


def true_crossover_events(crossovers: TruthCrossovers) -> dict[tuple[str, str], np.ndarray]:
    """{(parent, child): true crossover positions} of one chromosome's gametes (TruthCrossovers)."""
    offsets = np.asarray(crossovers.offsets)
    positions = np.asarray(crossovers.positions)
    return {
        (parent, child): positions[offsets[gamete] : offsets[gamete + 1]]
        for gamete, (parent, child) in enumerate(zip(crossovers.parent, crossovers.child, strict=False))
    }


def _phase_shares(row: dict) -> None:
    """Add phased_share (scored sites / correctly called true heterozygotes) to a row of PHASE_COLUMNS."""
    row["phased_share"] = (
        row["phased_true_heterozygotes"] / row["correct_true_heterozygotes"]
        if row.get("correct_true_heterozygotes")
        else None
    )


def _write_table(destination: Path, filename: str, rows) -> None:
    import pandas as pd

    from ..core.columns import column_name

    temporary = destination / f".{filename}.tmp"

    pd.DataFrame(rows).rename(columns=column_name).to_csv(temporary, index=False)
    temporary.replace(destination / filename)


def new_report(
    seed: int, names: Sequence[str], requested: Sequence[str], checkpoint_root: str | Path
) -> dict[str, Any]:
    """The summary skeleton (keys in their published order)."""
    return dict(
        seed=seed,
        samples=len(names),
        pedigree=None,
        chromosomes=[],
        phase_groups=[],
        founders=[],
        requested_stages=list(requested),
        checkpoint_root=str(Path(checkpoint_root).resolve()),
        phase_set_metric=PHASE_METRIC,
        skipped=[],
    )


def collect(report: dict, results) -> tuple[list, list]:
    """Add the chromosomes' rows (evaluate_chromosome results, contig order) to ``report`` and complete it with
    totals and notes; returns (component rows, match rows)."""
    component_rows, matching_rows = [], []
    for row, founders, component_part, matching_part, phase_groups, skipped in results:
        report["chromosomes"].append(row)
        report["phase_groups"].extend(phase_groups)
        report["founders"].extend(founders)
        report["skipped"].extend(skipped)
        component_rows.extend(component_part)
        matching_rows.extend(matching_part)
    from .metrics import n50

    # Phase-set N50s (bp) per chromosome and group, then genome-wide per group; the spans themselves are not kept.
    spans: dict[str, dict[str, list[int]]] = {}
    totals: dict[str, Any] = {}
    for group_row in report["phase_groups"]:
        group = group_row["parent_group"]
        part = spans.setdefault(group, dict(scored=[], released=[]))
        total = totals.setdefault(group, dict(samples=group_row["samples"], **{key: 0 for key in PHASE_COLUMNS}))
        for key in PHASE_COLUMNS:
            total[key] += group_row[key]
        _phase_shares(group_row)
        for key, name in (("phase_set_spans_bp", "scored"), ("released_phase_set_spans_bp", "released")):
            values = group_row.pop(key)
            group_row[key.replace("_spans_bp", "_n50_bp")] = n50(values)
            part[name].extend(values)
        if group == "all":
            row = next(row for row in report["chromosomes"] if row["contig"] == group_row["contig"])
            row["phase_set_n50_bp"] = group_row["phase_set_n50_bp"]
            row["released_phase_set_n50_bp"] = group_row["released_phase_set_n50_bp"]
    for group, total in totals.items():
        _phase_shares(total)
        total["phase_set_n50_bp"] = n50(spans[group]["scored"])
        total["released_phase_set_n50_bp"] = n50(spans[group]["released"])
    report["phase_totals"] = totals
    report["totals"] = {
        key: sum(row.get(key, 0) for row in report["chromosomes"])
        for key in ("total_alleles", "called_alleles", *PHASE_COLUMNS)
    }
    _phase_shares(report["totals"])
    report["totals"]["phase_set_n50_bp"] = totals.get("all", {}).get("phase_set_n50_bp", 0)
    report["totals"]["released_phase_set_n50_bp"] = totals.get("all", {}).get("released_phase_set_n50_bp", 0)
    report["interpretation"] = INTERPRETATION
    report["segment_and_long_range_interpretation"] = SEGMENT_INTERPRETATION
    return component_rows, matching_rows


def write_evaluation(
    destination: str | Path,
    report: dict[str, Any],
    component_rows: Sequence[MetricRow],
    matching_rows: Sequence[MetricRow],
) -> None:
    """Write summary.json and the five tables atomically into ``destination``."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination / "summary.json")
    for filename, rows in (
        ("chromosomes.csv", report["chromosomes"]),
        ("phase_groups.csv", report["phase_groups"]),
        ("founders.csv", report["founders"]),
        ("founder_components.csv", component_rows),
        ("founder_matches.csv", matching_rows),
    ):
        _write_table(destination, filename, rows)


def phase_scores(
    product: PhaseArrays, truth: ContigTruth, names: Sequence[str], parent_counts: np.ndarray | None, *, contig: str
) -> tuple[MetricRow, list[MetricRow]]:
    """One chromosome's released family phase scored against the truth (evaluation.metrics).

    ``truth``: the contig's simulation truth (alleles int8 [N, L, 2] in ``names`` order, positions); ``parent_counts``:
    parent_counts_of (None: only the all-sample group). Returns (chromosome columns, one row per parent group with the
    sample count and PHASE_COLUMNS, plus the groups' scored and released set spans under ``phase_set_spans_bp`` and
    ``released_phase_set_spans_bp`` for collect)."""
    from .metrics import SAMPLE_COLUMNS, released_phase_counts

    sample_ids = tuple(map(str, product["sample_ids"]))
    if len(set(sample_ids)) != len(sample_ids) or sorted(sample_ids) != sorted(names):
        raise InputError(f"{contig}: family-phase and true sample axes differ")
    # Release 2.1 records hold the run's canonical sample order: rows back to the truth's order.
    index = {name: row for row, name in enumerate(sample_ids)}
    order = np.asarray([index[name] for name in names], dtype=np.int64)
    calls = np.asarray(product["calls"])
    true_alleles = np.asarray(truth.alleles, dtype=np.int8)
    positions = np.asarray(truth.positions)
    if true_alleles.shape != calls.shape:
        raise InputError(f"{contig}: incompatible true/final allele axes")
    if not np.array_equal(np.asarray(product["positions"]), positions):
        raise InputError(f"{contig}: final positions differ from truth")
    undetermined = product.get("frame_undetermined")
    breaks, offsets = product.get("phase_breaks"), product.get("phase_break_offsets")
    sample_breaks = (
        None
        if breaks is None or offsets is None
        else [np.asarray(breaks)[offsets[i] : offsets[i + 1]] for i in range(len(names))]
    )
    counts, spans, released = released_phase_counts(
        calls,
        true_alleles,
        np.asarray(product["component_ids"]),
        undetermined,
        positions,
        sample_breaks,
        sample_order=order,
        undetermined_bit=2,
    )
    total_alleles, called_alleles = int(calls.size), int(np.count_nonzero(calls >= 0))
    columns: MetricRow = dict(total_alleles=total_alleles, called_alleles=called_alleles)
    columns["called_fraction"] = called_alleles / total_alleles if calls.size else None
    del calls, true_alleles
    members = {"all": np.ones(len(names), dtype=bool)}
    if parent_counts is not None:
        members.update((group, np.asarray(parent_counts) == count) for count, group in enumerate(PARENT_GROUPS))
    groups: list[MetricRow] = []
    for group, mask in members.items():
        totals = counts[mask].sum(axis=0) if mask.any() else np.zeros(len(SAMPLE_COLUMNS), dtype=np.int64)
        row: MetricRow = dict(
            contig=contig,
            parent_group=group,
            samples=int(mask.sum()),
            **dict(zip(SAMPLE_COLUMNS, map(int, totals), strict=False)),
        )
        chosen = np.flatnonzero(mask).tolist()
        row["phase_set_spans_bp"] = [int(v) for i in chosen for v in spans[i]]
        row["released_phase_set_spans_bp"] = [int(v) for i in chosen for v in released[i]]
        groups.append(row)
    columns.update((key, groups[0][key]) for key in PHASE_COLUMNS)
    return columns, groups


def identity_flags(panels: Sequence[StagePanel]) -> dict[int, dict[int, np.ndarray]] | None:
    """Derive bp flags from the original row paths, including founder_refit panels."""
    from ..contracts.panels import partner_changes

    result = {}
    has_paths = False
    for component, panel in enumerate(panels):
        if panel.rows is None or panel.atomic_offsets is None:
            continue
        has_paths = True
        offsets, positions = np.asarray(panel.atomic_offsets), np.asarray(panel.positions)
        rows: dict[int, list[int]] = {}
        for flag in partner_changes(panel.rows):
            rows.setdefault(flag.row, []).append(int(positions[offsets[flag.block]]))
        result[component] = {row: np.asarray(sorted(marks), dtype=np.int64) for row, marks in rows.items()}
    return result if has_paths else None


def evaluate_chromosome(
    inputs: ChromosomeInput, names: Sequence[str], requested: Sequence[str], *, parent_counts: np.ndarray | None = None
) -> ChromosomeEvaluation:
    """Score selected original-axis products of one contig and return sparse typed rows.

    No I/O, clock, logging or thread-state changes occur here. All inputs are borrowed;
    output row dictionaries are new. Invalid sample/position axes raise InputError.
    """
    from .founders import evaluate_founders, represented_ancestry
    from .identity import copy_counts, identity_counts, stratum_counts
    from .tasks import ChromosomeEvaluation

    contig, truth = inputs.contig, inputs.truth
    automatic = tuple(requested) in (("available",), ("all",))
    founders: list[MetricRow] = []
    component_rows: list[MetricRow] = []
    matching_rows: list[MetricRow] = []
    phase_groups: list[MetricRow] = []
    skipped: list[str] = []
    if tuple(truth.sample_ids) != tuple(names):
        raise InputError(f"{contig}: simulation truth sample axis differs")
    positions = np.asarray(truth.positions)
    if inputs.founders:
        founder_truth = np.asarray(truth.founder_haplotypes, dtype=np.int8)
        represented = represented_ancestry(truth.painting, positions, len(founder_truth))
        counts = copy_counts(truth.painting, positions, len(founder_truth))
        grid = dict(haplotypes=founder_truth, positions=positions)
        for stage, stage_panels in inputs.founders.items():
            panels = [(panel.positions, panel.calls) for panel in stage_panels]
            summary, components, matches = evaluate_founders(
                panels, positions, founder_truth, represented, contig=contig, stage=stage
            )
            for name, values in stratum_counts(panels, matches, grid, counts).items():
                summary.update({f"{name}_copy_{key}": value for key, value in values.items()})
            summary.update(
                {
                    f"identity_{key}": value
                    for key, value in identity_counts(panels, grid, counts, identity_flags(stage_panels)).items()
                }
            )
            founders.append(summary)
            component_rows.extend(components)
            matching_rows.extend(matches)
    row: MetricRow = dict(contig=contig, sites=len(positions))
    reported = {}
    if automatic or "family_phase" in requested:
        if inputs.phase is not None:
            columns, phase_groups = phase_scores(inputs.phase, truth, names, parent_counts, contig=contig)
            row.update(columns)
            _phase_shares(row)
        elif not automatic:
            raise InputError(f"{contig}: no completed family phase")
        else:
            skipped.append(f"{contig}/family_phase")
    if automatic or "recombination" in requested:
        if inputs.recombination is not None:
            map_metrics = map_counts(inputs.recombination, true_crossover_events(truth.crossovers), names)
            reported = {key: map_metrics.pop(key) for key in REPORTED_INTERVAL_COLUMNS}
            row.update(map_metrics)
        elif not automatic:
            raise InputError(f"{contig}: no completed recombination map")
        else:
            skipped.append(f"{contig}/recombination")
    row.update(reported)
    return ChromosomeEvaluation(row, founders, component_rows, matching_rows, phase_groups, skipped)


def parent_counts_of(edges: ParentEdges | None, names: Sequence[str]) -> np.ndarray | None:
    """Count released Tier-B edges per requested sample, not truth parents or cohorts."""
    if edges is None:
        return None
    sample_ids = tuple(map(str, edges.sample_ids))
    if sorted(sample_ids) != sorted(names):
        raise InputError("pedigree and true sample axes differ")
    counts = np.bincount(np.asarray(edges.children, dtype=np.int64), minlength=len(sample_ids))
    by_name = dict(zip(sample_ids, counts.tolist(), strict=False))
    return np.asarray([by_name[name] for name in names], dtype=np.int64)


def plan_evaluation(
    reader: EvaluationReader, *, stages: Sequence[str] = ("available",), contigs: Sequence[str] | None = None
) -> tuple[dict[str, Any], tuple[EvaluationTask, ...]]:
    """Validate every requested product before returning ordered chromosome tasks."""
    from ..simulation.truth import COMPLETE
    from .tasks import EvaluationTask

    manifest = reader.simulation_manifest()
    if manifest is None:
        raise InputError(f"{reader.root}: no simulation manifest")
    if manifest.state != COMPLETE:
        raise InputError("evaluation requires complete simulated truth")
    if manifest.observed is None:
        raise InputError("complete simulation manifest has no observed pedigree")
    names, requested = tuple(manifest.sample_ids), tuple(stages)
    if not requested or set(requested) - set((*EVALUATION_STAGES, "available", "all")):
        raise InputError("unrecognized evaluation stages")
    automatic = requested in (("available",), ("all",))
    if not automatic and {"all", "available"} & set(requested):
        raise InputError("all/available cannot be mixed with explicit stages")
    selected = tuple(manifest.contigs)
    if contigs is not None:
        if len(set(contigs)) != len(contigs) or set(contigs) - set(selected):
            raise InputError("evaluation contigs must be unique names in this simulation")
        selected = tuple(c for c in selected if c in contigs)
    report = new_report(manifest.seed, names, requested, reader.checkpoints)
    pedigree = reader.pedigree()
    if automatic or "pedigree" in requested:
        if pedigree is not None:
            report["pedigree"] = pedigree_counts(pedigree.sample_ids, pedigree.calls, manifest.observed, names)
        elif not automatic:
            raise InputError("no completed genome-wide pedigree")
    parent_counts = parent_counts_of(None if pedigree is None else pedigree.edges, names)
    tasks = []
    for contig in selected:
        found = [stage for stage in FOUNDER_STAGES if reader.stage_present(contig, stage)]
        chosen = (
            found[-1:]
            if requested == ("available",)
            else found
            if requested == ("all",)
            else [stage for stage in requested if stage in FOUNDER_STAGES]
        )
        for stage in chosen:
            if stage not in found:
                raise InputError(f"{contig}: no completed {stage} product")
        if not automatic:
            for stage in ("family_phase", "recombination"):
                if stage in requested and not reader.stage_present(contig, stage):
                    raise InputError(f"{contig}: no completed {stage} product")
        tasks.append(EvaluationTask(contig, names, tuple(chosen), requested, parent_counts))
    return report, tuple(tasks)


def evaluate_run(
    run_directory,
    *,
    checkpoints=None,
    cores=None,
    stages=("available",),
    contigs=None,
    destination=None,
    reader=None,
    task_runner=None,
):
    """Evaluate a simulation through results readers, with caller-supplied task execution.

    The application owns parallel scheduling: task_runner(tasks, reader) returns
    results in task order. Direct calls execute sequentially without hidden pools.
    Explicit-stage absence is checked before computation or output publication.
    """
    from .tasks import evaluate_task

    if reader is None:
        from ..results import open_results

        reader = open_results(run_directory, checkpoints=checkpoints)
    report, tasks = plan_evaluation(reader, stages=stages, contigs=contigs)
    results = [evaluate_task(task, reader) for task in tasks] if task_runner is None else task_runner(tasks, reader)
    components, matches = collect(report, [result.parts() for result in results])
    target = Path(run_directory) / "evaluation" if destination is None else Path(destination)
    write_evaluation(target, report, components, matches)
    return report
