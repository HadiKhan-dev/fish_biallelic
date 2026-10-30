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
from pathlib import Path
import time

import numpy as np

FOUNDER_STAGES = ("discovery", "local_selection_initial", "local_selection_l1", "local_selection_l2",
                  "assembly_l1", "assembly_l2", "assembly_l3", "assembly_l4", "founder_refinement",
                  "painting", "founder_refit")
EVALUATION_STAGES = (*FOUNDER_STAGES, "pedigree", "family_phase", "recombination")
# Released-phase counts (release 3.2, evaluation.metrics): summed over samples per chromosome (all samples) and
# per parent group; the N50 columns and phased_share are derived from them.
PHASE_COLUMNS = ("called_genotypes", "genotype_errors", "correct_true_heterozygotes", "phased_true_heterozygotes",
                 "phase_sets", "set_aligned_errors", "switch_errors", "long_switches", "released_phase_sets")
# Parent groups: the number of released (Tier-B supported) pedigree edges into the sample, 0, 1 or 2.
PARENT_GROUPS = ("parents0", "parents1", "parents2")
PHASE_METRIC = "released_phase_v3"
# Reported-interval map columns, written after every earlier column.
REPORTED_INTERVAL_COLUMNS = ("reported_intervals_containing_true_crossover",
                             "reported_interval_hit_rate", "called_interval_median_width_bp",
                             "reported_interval_median_width_bp")
PARENT_STATES = ("zero_observed_parents", "one_observed_parent", "two_observed_parents")
INTERPRETATION = (
    "Truth is used only for evaluation. Founder labels match over whole components, "
    "not marker-wise; nearest-row errors and one-to-one completeness are separate. "
    "Represented ancestry means present in a sampled homologue, not independently identifiable. "
    "Phase is scored per sample and released phase set (a component split at the sample's released phase "
    "breaks) over correctly called true heterozygotes released phased; genotype errors are counted "
    "separately, never as phase errors. Map truth uses correct edges "
    "within inferred observable spans. Unexecuted stages are not evaluated.")
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
    "the median widths (right - left) of both kinds of interval give its context.")
# Per-task allowance of the chromosome estimate besides the decoded records: decode buffers and the kernels a
# worker loads in its first task. Calibrated on std_n80 a1: measured peak per chromosome at most 0.34 GiB for
# estimates of 0.40-0.45 GiB.
TASK_ALLOWANCE_BYTES = 256 << 20


# ------------------------------------------------------------------------------------------- founder stages
def _local_round(stage: str) -> str:
    return stage.removeprefix("local_selection_")


def _assembly_level(stage: str) -> int:
    return int(stage.removeprefix("assembly_l"))


def founder_stage_present(file, stage: str) -> bool:
    """Whether a contig holds a founder stage's product (record keys only, nothing decoded)."""
    from ..assembly import records as assembly_records
    from ..discovery import records as discovery_records
    from ..founder_refit import records as refit_records
    from ..local_selection import records as selection_records
    from ..painting import records as painting_records
    if stage == "discovery":
        return discovery_records.complete(file)
    if stage.startswith("local_selection_"):
        return selection_records.round_complete(file, _local_round(stage))
    if stage.startswith("assembly_l"):
        progress = assembly_records.progress(file)
        return progress["completion"] and _assembly_level(stage) in progress["levels"]
    if stage == "founder_refinement":
        return assembly_records.complete(file)
    if stage == "painting":
        return painting_records.complete(file)
    if stage == "founder_refit":
        return refit_records.complete(file)
    raise ValueError(f"unknown founder stage: {stage}")


def identity_flags(file, stage: str):
    """Per component {row: sorted int64 positions} of the stage's founder identity flags (export.identity: the
    first marker of the block right of each flagged junction), or None for a stage without row paths (discovery,
    local selection). Release 2.17."""
    from ..assembly import records as assembly_records
    from ..founder_refit import records as refit_records
    from ..export.identity import partner_changes
    from ..painting import records as painting_records
    if stage.startswith("assembly_l") or stage == "founder_refinement":
        panels = assembly_records.founder_panels(file, level=None if stage == "founder_refinement"
                                                 else _assembly_level(stage))
    elif stage == "painting":
        panels = painting_records.founder_panels(file)
    elif stage == "founder_refit":
        panels = refit_records.founder_panels(file)
    else:
        return None
    result = {}
    for component, panel in enumerate(panels or ()):
        offsets, positions = np.asarray(panel.atomic_offsets), np.asarray(panel.positions)
        rows = {}
        for flag in partner_changes(panel.rows):
            rows.setdefault(flag.row, []).append(int(positions[offsets[flag.block]]))
        result[component] = {row: np.asarray(sorted(marks), dtype=np.int64) for row, marks in rows.items()}
    return result


def founder_panels(file, stage: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """One contig's panels of a founder stage, in chromosome order: (positions int64 [L], calls int8 [K, L]).

    Calls are the released hard calls (-1 = no call); rows are founder labels 0..K-1.
    """
    from ..assembly import records as assembly_records
    from ..discovery import records as discovery_records
    from ..founder_refit import records as refit_records
    from ..local_selection import records as selection_records
    from ..painting import records as painting_records
    if stage == "discovery":
        return [(positions, calls) for positions, calls, _ in discovery_records.founder_panels(file)]
    if stage.startswith("local_selection_"):
        return [(positions, calls) for positions, calls, _ in
                selection_records.founder_panels(file, _local_round(stage))]
    if stage.startswith("assembly_l") or stage == "founder_refinement":
        level = None if stage == "founder_refinement" else _assembly_level(stage)
        panels = assembly_records.founder_panels(file, level=level)
    elif stage == "painting":
        panels = painting_records.founder_panels(file)
    elif stage == "founder_refit":
        panels = refit_records.founder_panels(file)
    else:
        raise ValueError(f"unknown founder stage: {stage}")
    if panels is None:
        raise FileNotFoundError(f"{file.label}: no completed {stage} product")
    return [(panel.positions, panel.released) for panel in panels]


def _stage_bytes(file, stage: str) -> int:
    """Decoded bytes of the records a founder stage's reader decodes (record metadata only): the block records of
    discovery and of a local-selection round; for the assembly-based stages (assembly levels, founder refinement,
    painting, founder refit) the prepared panels every component panel is materialised over."""
    from ..assembly import records as assembly_records
    if stage == "discovery":
        prefix = "discovery/blocks/"
    elif stage.startswith("local_selection_"):
        prefix = f"local_selection/{_local_round(stage)}/blocks/"
    else:
        prefix = f"{assembly_records.FINAL_PREFIX}/{assembly_records.COMPLETION}"
    return sum(info.decoded_bytes for info in file.infos(prefix))


# ------------------------------------------------------------------------------------------- downstream counts
def pedigree_counts(sample_ids, calls, truth, names, targets=None) -> dict:
    """Tier-B calls vs the true pedigree of the analysed samples.

    ``sample_ids``/``calls``: pedigree.records.load_calls (PedigreeCalls: state -1 unresolved, parents -1 none);
    ``truth``: the simulation manifest's PedigreeTable of the analysed samples (parents outside the analysed
    samples are not observable edges); ``names``: the analysed sample axis. The calls' axis may be any order of
    ``names`` (release 2.1 records hold the run's canonical sample order); edges are matched by sample name.
    ``targets`` (release 2.17): the eligibility's inference targets by name (a simulated design with metadata);
    only they are counted, and the count is reported. None: every sample is a target.
    """
    sample_ids = tuple(map(str, sample_ids))
    if len(set(sample_ids)) != len(sample_ids) or set(sample_ids) != set(names) or len(sample_ids) != len(names):
        raise ValueError("inferred and true sample axes differ")
    true_parents = {sample: (first, second)
                    for sample, first, second in zip(truth.samples, truth.parent1, truth.parent2)}
    known = set(names)
    exact = correct_states = true_edges = inferred_edges = shared_edges = 0
    expected_states = {state: 0 for state in PARENT_STATES}
    for sample, state, parents in zip(sample_ids, np.asarray(calls.state).tolist(),
                                      np.asarray(calls.parents).tolist()):
        if targets is not None and sample not in targets:
            continue
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
    result = dict(exact_configurations=exact, correct_parent_states=correct_states,
                  expected_states=expected_states, true_edges=true_edges,
                  correct_edges=shared_edges, extra_edges=inferred_edges-shared_edges,
                  missing_edges=true_edges-shared_edges)
    if targets is not None:
        result["targets"] = len(set(targets) & set(sample_ids))
    return result


def pedigree_targets(run_directory, names):
    """Inference targets of a simulated design with metadata eligibility (release 2.17), by name; None when the
    run's configuration supplies no [samples] metadata (every sample a target)."""
    from ..config.load import load_config
    from ..pipeline.samples import build_eligibility, read_sample_table
    path = Path(run_directory) / "config.toml"
    if not path.is_file():
        return None
    config = load_config(path)
    if config.samples.metadata is None:
        return None
    eligibility = build_eligibility(config, read_sample_table(config.samples, names))
    if eligibility is None:
        return None
    return {name for name, child in zip(eligibility.sample_ids, eligibility.children) if child}


def map_counts(product, events, names) -> dict:
    """Recombination calls of one chromosome vs the true crossovers of its meioses.

    ``product``: recombination.records RecombinationProduct; ``events``: {(parent, child): sorted true crossover
    positions (bp)}. Only edges of the true pedigree count; an interval (left, right] localizes a true crossover
    when it contains one. Edge indices refer to the product's own sample axis (any order of ``names``).
    """
    if getattr(product, "sample_ids", None) is not None:
        if set(map(str, product.sample_ids)) != set(names):
            raise ValueError("recombination and true sample axes differ")
        names = tuple(map(str, product.sample_ids))
    true_observed = unmatched = 0
    expected = exposure = 0.0
    called = called_hits = reported_hits = 0
    called_widths, reported_widths = [], []
    for edge, (parent, child) in enumerate(zip(product.edge_parent, product.edge_child)):
        raw = events.get((names[int(parent)], names[int(child)]))
        if raw is None:
            unmatched += 1
            continue
        for span in product.edge_informative_spans(edge):
            left, right = span[:2]
            true_observed += int(np.searchsorted(raw, right, side="right")
                                 - np.searchsorted(raw, left, side="right"))
        expected += float(product.expected_crossovers[edge].sum())
        exposure += float(product.exposure[edge].sum())
        reported = product.edge_reported_intervals(edge)
        for index, interval in enumerate(product.edge_crossovers(edge)):
            left, right = interval[:2]
            called += 1
            called_hits += int(np.searchsorted(raw, right, side="right")
                               > np.searchsorted(raw, left, side="right"))
            called_widths.append(float(right - left))
            # Same rule on the reported interval (low, high]: a change placed at a
            # painting-bin edge can lie anywhere in the bins around that edge.
            low, high = reported[index]
            reported_hits += int(np.searchsorted(raw, high, side="right")
                                 > np.searchsorted(raw, low, side="right"))
            reported_widths.append(float(high - low))
    return dict(true_crossovers_in_inferred_exposure=true_observed,
                expected_crossovers_on_correct_edges=expected,
                correct_edge_exposure_meiosis_bp=exposure,
                inferred_edges_absent_from_truth=unmatched,
                called_crossovers_on_correct_edges=called,
                called_intervals_containing_true_crossover=called_hits,
                called_interval_hit_rate=called_hits/called if called else None,
                reported_intervals_containing_true_crossover=reported_hits,
                reported_interval_hit_rate=reported_hits/called if called else None,
                called_interval_median_width_bp=float(np.median(called_widths)) if called_widths else None,
                reported_interval_median_width_bp=(float(np.median(reported_widths))
                                                   if reported_widths else None))


def true_crossover_events(crossovers) -> dict:
    """{(parent, child): true crossover positions} of one chromosome's gametes (TruthCrossovers)."""
    offsets = np.asarray(crossovers.offsets)
    positions = np.asarray(crossovers.positions)
    return {(parent, child): positions[offsets[gamete]:offsets[gamete + 1]]
            for gamete, (parent, child) in enumerate(zip(crossovers.parent, crossovers.child))}


# ------------------------------------------------------------------------------------------- released phase
def parent_counts_of(store, names):
    """int64 [N] in ``names`` order: the released (Tier-B supported) pedigree edges into each sample, the edges family
    phase was conditioned on (pedigree.records.load_edges); None when the run has no pedigree."""
    from ..pedigree import records as pedigree_records
    if not pedigree_records.complete(store):
        return None
    edges = pedigree_records.load_edges(store)
    sample_ids = tuple(map(str, edges.sample_ids))
    if sorted(sample_ids) != sorted(names):
        raise ValueError("pedigree and true sample axes differ")
    counts = np.bincount(np.asarray(edges.children, dtype=np.int64), minlength=len(sample_ids))
    by_name = dict(zip(sample_ids, counts.tolist()))
    return np.asarray([by_name[name] for name in names], dtype=np.int64)


def phase_scores(file, truth, names, parent_counts, *, nthreads=1):
    """One chromosome's released family phase scored against the truth (evaluation.metrics).

    ``truth``: the contig's simulation truth (alleles int8 [N, L, 2] in ``names`` order, positions); ``parent_counts``:
    parent_counts_of (None: only the all-sample group). Returns (chromosome columns, one row per parent group with the
    sample count and PHASE_COLUMNS, plus the groups' scored and released set spans under ``phase_set_spans_bp`` and
    ``released_phase_set_spans_bp`` for collect)."""
    from ..family_phase import records as phase_records
    from .metrics import SAMPLE_COLUMNS, released_phase_counts
    product = phase_records.load_arrays(file, ("sample_ids", "positions", "component_ids", "calls",
                                               "frame_undetermined", "phase_breaks", "phase_break_offsets"),
                                        nthreads=nthreads)
    sample_ids = tuple(map(str, product["sample_ids"]))
    if len(set(sample_ids)) != len(sample_ids) or sorted(sample_ids) != sorted(names):
        raise ValueError(f"{file.name}: family-phase and true sample axes differ")
    # Release 2.1 records hold the run's canonical sample order: rows back to the truth's order.
    index = {name: row for row, name in enumerate(sample_ids)}
    order = np.asarray([index[name] for name in names], dtype=np.int64)
    calls = np.asarray(product["calls"])[order]
    true_alleles = np.asarray(truth.alleles, dtype=np.int8)
    positions = np.asarray(truth.positions)
    if true_alleles.shape != calls.shape:
        raise ValueError(f"{file.name}: incompatible true/final allele axes")
    if not np.array_equal(np.asarray(product["positions"]), positions):
        raise ValueError(f"{file.name}: final positions differ from truth")
    undetermined = product.get("frame_undetermined")
    if undetermined is not None:
        undetermined = (np.asarray(undetermined)[order] & 2) != 0       # bit 1: displayed phase undetermined
    breaks, offsets = product.get("phase_breaks"), product.get("phase_break_offsets")
    sample_breaks = (None if breaks is None or offsets is None else
                     [np.asarray(breaks)[offsets[i]:offsets[i + 1]] for i in order.tolist()])
    counts, spans, released = released_phase_counts(calls, true_alleles, np.asarray(product["component_ids"]),
                                                    undetermined, positions, sample_breaks)
    columns = dict(total_alleles=int(calls.size), called_alleles=int(np.count_nonzero(calls >= 0)))
    columns["called_fraction"] = columns["called_alleles"] / columns["total_alleles"] if calls.size else None
    del calls, true_alleles
    members = {"all": np.ones(len(names), dtype=bool)}
    if parent_counts is not None:
        members.update((group, np.asarray(parent_counts) == count) for count, group in enumerate(PARENT_GROUPS))
    groups = []
    for group, mask in members.items():
        totals = counts[mask].sum(axis=0) if mask.any() else np.zeros(len(SAMPLE_COLUMNS), dtype=np.int64)
        row = dict(contig=file.name, parent_group=group, samples=int(mask.sum()), **dict(zip(
            SAMPLE_COLUMNS, map(int, totals))))
        chosen = np.flatnonzero(mask).tolist()
        row["phase_set_spans_bp"] = [int(v) for i in chosen for v in spans[i]]
        row["released_phase_set_spans_bp"] = [int(v) for i in chosen for v in released[i]]
        groups.append(row)
    columns.update((key, groups[0][key]) for key in PHASE_COLUMNS)
    return columns, groups


def _phase_shares(row: dict) -> None:
    """Add phased_share (scored sites / correctly called true heterozygotes) to a row of PHASE_COLUMNS."""
    row["phased_share"] = (row["phased_true_heterozygotes"] / row["correct_true_heterozygotes"]
                           if row.get("correct_true_heterozygotes") else None)


# ------------------------------------------------------------------------------------------- one chromosome
def evaluate_chromosome(file, names, founder_stages, requested, *, nthreads=1, parent_counts=None):
    """One chromosome's rows: (chromosome row, founder rows, component rows, match rows, phase-group rows,
    skipped). ``parent_counts``: parent_counts_of (released edges into each sample, ``names`` order) or None."""
    from .. import execution
    from ..family_phase import records as phase_records
    from ..recombination import records as recombination_records
    from ..simulation import truth as truth_records
    from .founders import evaluate_founders, represented_ancestry
    contig = file.name
    automatic = requested in (("available",), ("all",))
    founders, component_rows, matching_rows, phase_groups, skipped = [], [], [], [], []
    truth = truth_records.load(file, nthreads=nthreads)
    if tuple(truth.sample_ids) != tuple(names):
        raise ValueError(f"{file.label}: simulation truth sample axis differs")
    positions = np.asarray(truth.positions)
    if founder_stages:
        from .identity import copy_counts, identity_counts, stratum_counts
        founder_truth = np.asarray(truth.founder_haplotypes, dtype=np.int8)
        execution.claim_threads()
        represented = represented_ancestry(truth.painting, positions, len(founder_truth))
        counts = copy_counts(truth.painting, positions, len(founder_truth))
        grid = dict(haplotypes=founder_truth, positions=positions)
        for stage in founder_stages:
            panels = founder_panels(file, stage)
            execution.claim_threads()
            summary, components, matches = evaluate_founders(
                panels, positions, founder_truth, represented, contig=contig, stage=stage)
            # Release 2.17: copy-count strata and continuity assertions (evaluation.identity), after every
            # existing column.
            for name, values in stratum_counts(panels, matches, grid, counts).items():
                summary.update({f"{name}_copy_{key}": value for key, value in values.items()})
            summary.update({f"identity_{key}": value for key, value in
                            identity_counts(panels, grid, counts, identity_flags(file, stage)).items()})
            founders.append(summary)
            component_rows.extend(components)
            matching_rows.extend(matches)
            del panels
        del founder_truth, represented
    row = dict(contig=contig, sites=len(positions))
    reported = {}  # the reported-interval map columns follow every other column
    if automatic or "family_phase" in requested:
        if phase_records.complete(file):
            execution.claim_threads()
            columns, phase_groups = phase_scores(file, truth, names, parent_counts, nthreads=nthreads)
            row.update(columns)
            _phase_shares(row)
        elif not automatic:
            raise FileNotFoundError(f"{contig}: no completed family phase")
        else:
            skipped.append(f"{contig}/family_phase")
    if automatic or "recombination" in requested:
        if recombination_records.complete(file):
            counts = map_counts(recombination_records.load(file, nthreads=nthreads),
                                true_crossover_events(truth.crossovers), names)
            reported = {key: counts.pop(key) for key in REPORTED_INTERVAL_COLUMNS}
            row.update(counts)
        elif not automatic:
            raise FileNotFoundError(f"{contig}: no completed recombination map")
        else:
            skipped.append(f"{contig}/recombination")
    row.update(reported)
    return row, founders, component_rows, matching_rows, phase_groups, skipped


def _evaluate_task(task, state, phase, common, budget):
    """Chromosome-worker handler: load one chromosome's truth and products locally; return only rows."""
    from .. import execution
    from ..store import CheckpointStore
    contig, founder_stages = task
    started = time.monotonic()
    threads = execution.claim_threads()
    print(f"EVALUATION {contig}: start; threads={threads}", flush=True)
    file = CheckpointStore(common["checkpoints"], nthreads=budget).contig(contig)        # live lease
    result = evaluate_chromosome(file, common["names"], founder_stages, common["requested"], nthreads=budget,
                                 parent_counts=common.get("parent_counts"))
    print(f"EVALUATION {contig}: complete in {time.monotonic() - started:.2f}s; "
          f"threads={execution.claim_threads()}", flush=True)
    return state, result


def _memory_estimates(store, tasks, requested) -> list[int]:
    """Planned peak bytes per chromosome task: decoded truth plus the largest selected product, 25% for
    numerical copies, and TASK_ALLOWANCE_BYTES."""
    from ..family_phase import records as phase_records
    from ..recombination import records as recombination_records
    from ..simulation import truth as truth_records
    automatic = requested in (("available",), ("all",))
    estimates = []
    for contig, founder_stages in tasks:
        file = store.contig(contig)
        sizes = [_stage_bytes(file, stage) for stage in founder_stages]
        for stage, key in (("family_phase", phase_records.KEY), ("recombination", recombination_records.KEY)):
            if (automatic or stage in requested) and file.has(key):
                sizes.append(file.info(key).decoded_bytes)
        estimates.append((file.info(truth_records.TRUTH).decoded_bytes + max(sizes, default=0)) * 5 // 4
                         + TASK_ALLOWANCE_BYTES)
    return estimates


# ------------------------------------------------------------------------------------------- one run
def _write_table(destination: Path, filename: str, rows) -> None:
    import pandas as pd
    temporary = destination / f".{filename}.tmp"
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(destination / filename)


def new_report(seed, names, requested, checkpoint_root) -> dict:
    """The summary skeleton (keys in their published order)."""
    return dict(seed=seed, samples=len(names), pedigree=None, chromosomes=[], phase_groups=[], founders=[],
                requested_stages=list(requested), checkpoint_root=str(Path(checkpoint_root).resolve()),
                phase_set_metric=PHASE_METRIC, skipped=[])


def evaluate_run(run_directory, *, checkpoints=None, cores=None, stages=("available",), contigs=None,
                 destination=None) -> dict:
    """Evaluate completed contigs/stages of a simulation run, without requiring downstream completion.

    Writes RUN/evaluation/ (module docstring), or ``destination`` when given (the run is then only read), and
    returns the summary. ``checkpoints`` defaults to RUN/checkpoints; ``cores`` (default: the CPU affinity) bounds
    the chromosome workers and their threads; ``contigs`` (default: every simulated contig) never changes the
    genome-wide pedigree counts.
    """
    from .. import execution
    from ..pedigree import records as pedigree_records
    from ..simulation import truth as truth_records
    from ..store import CheckpointStore
    output = Path(run_directory).resolve()
    available = execution.affinity_cpus()
    workers = available if cores is None else int(cores)
    if not 1 <= workers <= available:
        raise ValueError("evaluation cores must fit the current affinity")
    store = CheckpointStore(checkpoints or output / "checkpoints", nthreads=workers)
    manifest = truth_records.load_manifest(store)
    if manifest is None:
        raise FileNotFoundError(f"{store.root}: no simulation manifest")
    if manifest.state != truth_records.COMPLETE:
        raise ValueError("evaluation requires complete simulated truth")
    names = tuple(manifest.sample_ids)
    requested = tuple(stages)
    if not requested or set(requested) - set((*EVALUATION_STAGES, "available", "all")):
        raise ValueError("unrecognized evaluation stages")
    automatic = requested in (("available",), ("all",))
    if not automatic and ({"all", "available"} & set(requested)):
        raise ValueError("all/available cannot be mixed with explicit stages")
    selected = tuple(manifest.contigs)
    if contigs is not None:
        if len(set(contigs)) != len(contigs) or set(contigs) - set(selected):
            raise ValueError("evaluation contigs must be unique names in this simulation")
        selected = tuple(contig for contig in selected if contig in contigs)
    report = new_report(manifest.seed, names, requested, store.root)
    if automatic or "pedigree" in requested:
        if pedigree_records.complete(store):
            sample_ids, calls = pedigree_records.load_calls(store)
            report["pedigree"] = pedigree_counts(sample_ids, calls, manifest.observed, names,
                                                 targets=pedigree_targets(output, names))
        elif not automatic:
            raise FileNotFoundError("no completed genome-wide pedigree")
    tasks = []
    for contig in selected:
        file = store.contig(contig)
        found = [stage for stage in FOUNDER_STAGES if founder_stage_present(file, stage)]
        founder_stages = (found[-1:] if requested == ("available",) else
                          found if requested == ("all",) else
                          [stage for stage in requested if stage in FOUNDER_STAGES])
        # Resolve requested-product absence before launching workers (outputs untouched).
        for stage in founder_stages:
            if stage not in found:
                raise FileNotFoundError(f"{contig}: no completed {stage} product")
        if not automatic:
            from ..family_phase import records as phase_records
            from ..recombination import records as recombination_records
            for stage, complete, description in (
                    ("family_phase", phase_records.complete, "family phase"),
                    ("recombination", recombination_records.complete, "recombination map")):
                if stage in requested and not complete(file):
                    raise FileNotFoundError(f"{contig}: no completed {description}")
        tasks.append((contig, tuple(founder_stages)))
    # Each chromosome task holds its planned peak of the run's memory while it runs (it waits until it fits).
    weights = _memory_estimates(store, tasks, requested)
    limit = execution.memory_worker_limit(weights, workers, label="EVALUATION")
    # Parent groups of the phase metrics: the released edges family phase was conditioned on (genome-wide).
    parent_counts = parent_counts_of(store, names) if automatic or "family_phase" in requested else None
    common = dict(checkpoints=str(store.root), names=names, requested=requested, parent_counts=parent_counts)
    with execution.ChromosomeExecutor(tasks, _evaluate_task, budget=execution.CpuBudget.fixed(workers),
                                      weights=weights, max_workers=limit, label="EVALUATION",
                                      task_bytes=weights) as executor:
        results = executor.run("evaluate", common)
    component_rows, matching_rows = collect(report, results)
    write_evaluation(output / "evaluation" if destination is None else Path(destination), report,
                     component_rows, matching_rows)
    return report


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
        for summary in founders:
            print(f"[EVALUATE founders] {row['contig']}/{summary['stage']}: "
                  f"{summary['called_allele_errors']}/{summary['called_alleles']} "
                  f"called errors; {summary['truth_to_panel_errors_missing']} truth-to-panel errors/missing",
                  flush=True)
    from .metrics import n50
    # Phase-set N50s (bp) per chromosome and group, then genome-wide per group; the spans themselves are not kept.
    spans, totals = {}, {}
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
    report["totals"] = {key: sum(row.get(key, 0) for row in report["chromosomes"])
                        for key in ("total_alleles", "called_alleles", *PHASE_COLUMNS)}
    _phase_shares(report["totals"])
    report["totals"]["phase_set_n50_bp"] = totals.get("all", {}).get("phase_set_n50_bp", 0)
    report["totals"]["released_phase_set_n50_bp"] = totals.get("all", {}).get("released_phase_set_n50_bp", 0)
    report["interpretation"] = INTERPRETATION
    report["segment_and_long_range_interpretation"] = SEGMENT_INTERPRETATION
    return component_rows, matching_rows


def write_evaluation(destination, report: dict, component_rows, matching_rows) -> None:
    """Write summary.json and the five tables atomically into ``destination``."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination / "summary.json")
    for filename, rows in (("chromosomes.csv", report["chromosomes"]),
                           ("phase_groups.csv", report["phase_groups"]),
                           ("founders.csv", report["founders"]),
                           ("founder_components.csv", component_rows),
                           ("founder_matches.csv", matching_rows)):
        _write_table(destination, filename, rows)


def evaluate_command(args) -> int:
    """`haplotypes evaluate RUN [--stages ...] [--contigs ...] [--cores N]`."""
    from .. import execution
    available = execution.affinity_cpus()
    cores = available if args.cores is None else int(args.cores)
    if not 1 <= cores <= available:
        raise SystemExit(f"evaluate: --cores must be between 1 and the CPU affinity ({available})")
    from ..pipeline.driver import RunPaths
    started = time.monotonic()
    execution.start_run(cores, preload=(__name__,))
    # Evaluation compiles only the kernels it uses. Warming the saved manifest
    # here can compile hundreds of reconstruction kernels before a short report.
    evaluate_run(args.run_directory, checkpoints=RunPaths.from_directory(args.run_directory).checkpoints,
                 cores=cores, stages=tuple(args.stages),
                 contigs=None if args.contigs is None else tuple(args.contigs))
    print(f"Evaluation written to {Path(args.run_directory).resolve() / 'evaluation'} in "
          f"{time.monotonic() - started:.1f} s", flush=True)
    for line in execution.memory_governor().summary_lines():
        print(line, flush=True)
    return 0
