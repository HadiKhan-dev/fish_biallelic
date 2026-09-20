"""Evaluate available simulated products without feeding truth into inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.runtime import CheckpointStore, available_cpu_count
from ..core.products import FOUNDER_STAGES, load_panels, panel_location
from ..core.parallel import numba_thread_scope
from .founder_metrics import evaluate_founders, represented_ancestry
from .metrics import phase_counts

EVALUATION_STAGES = (*FOUNDER_STAGES, "pedigree", "family_phase", "recombination")
PHASE_COLUMNS = ("called_alleles", "called_genotypes", "genotype_errors",
                 "eligible_heterozygotes", "phase_comparisons", "phase_switch_errors",
                 "component_aligned_allele_errors")


def pedigree_counts(inferred, truth_pedigree, names):
    if tuple(inferred.Sample) != names:
        raise ValueError("inferred and true sample axes differ")
    truth_pedigree = truth_pedigree.set_index("Sample")
    known = set(names)
    states = ("zero_observed_parents", "one_observed_parent", "two_observed_parents")
    exact = correct_states = true_edges = inferred_edges = shared_edges = 0
    expected_states = {state: 0 for state in states}
    for row in inferred.itertuples(index=False):
        truth_row = truth_pedigree.loc[row.Sample]
        expected = {truth_row.Parent1, truth_row.Parent2} & known
        actual = {parent for parent in (row.Parent1, row.Parent2) if pd.notna(parent)}
        state = states[len(expected)]
        expected_states[state] += 1
        exact += int(expected == actual and row.ParentState == state)
        correct_states += int(row.ParentState == state)
        true_edges += len(expected)
        inferred_edges += len(actual)
        shared_edges += len(expected & actual)
    return dict(exact_configurations=exact, correct_parent_states=correct_states,
                expected_states=expected_states, true_edges=true_edges,
                correct_edges=shared_edges, extra_edges=inferred_edges-shared_edges,
                missing_edges=true_edges-shared_edges)


def map_counts(mapping, events, names):
    true_observed = unmatched = 0
    expected = exposure = 0.0
    for edge, (parent, child) in enumerate(zip(mapping["edge_parent"], mapping["edge_child"])):
        raw = events.get((names[int(parent)], names[int(child)]))
        if raw is None:
            unmatched += 1
            continue
        for span in mapping["informative_spans"][edge]:
            left, right = span[:2]
            true_observed += int(np.searchsorted(raw, right, side="right")
                                 - np.searchsorted(raw, left, side="right"))
        expected += float(mapping["expected_crossovers_by_edge_bin"][edge].sum())
        exposure += float(mapping["exposure_by_edge_bin"][edge].sum())
    return dict(true_crossovers_in_inferred_exposure=true_observed,
                expected_crossovers_on_correct_edges=expected,
                correct_edge_exposure_meiosis_bp=exposure,
                inferred_edges_absent_from_truth=unmatched)


def _write_table(destination, filename, rows):
    temporary = destination/f".{filename}.tmp"
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(destination/filename)


def evaluate_run(output_dir, *, checkpoints=None, cores=None, stages=("available",),
                 contigs=None, feedback_selection="balanced"):
    """Evaluate completed contigs/stages, without requiring downstream completion.

    available: latest existing founder product per contig, plus available
    downstream results. all: every saved discovery/feedback/hierarchy product.
    Explicit stages require that product; unexecuted hierarchy levels are not
    invented. Caller-selected contigs do not change genome-wide pedigree scoring.
    """
    output = Path(output_dir).resolve()
    workers = available_cpu_count() if cores is None else int(cores)
    if not 1 <= workers <= available_cpu_count():
        raise ValueError("evaluation cores must fit the current affinity")
    store = CheckpointStore(checkpoints or output/"checkpoints", nthreads=workers)
    simulation = store.load_global("simulated_reads")
    if simulation.get("simulation_state") != "complete":
        raise ValueError("evaluation requires complete simulated truth")
    names = tuple(simulation["sample_names"])
    requested = tuple(stages)
    if not requested or set(requested) - set((*EVALUATION_STAGES, "available", "all")):
        raise ValueError("unrecognized evaluation stages")
    automatic = requested in (("available",), ("all",))
    if not automatic and ({"all", "available"} & set(requested)):
        raise ValueError("all/available cannot be mixed with explicit stages")
    selected = tuple(simulation["region_keys"])
    if contigs is not None:
        if len(set(contigs)) != len(contigs) or set(contigs)-set(selected):
            raise ValueError("evaluation contigs must be unique names in this simulation")
        selected = tuple(c for c in selected if c in contigs)
    report = dict(seed=simulation["simulation_seed"], samples=len(names), pedigree=None,
                  chromosomes=[], founders=[], requested_stages=list(requested),
                  checkpoint_root=str(Path(store.root).resolve()), skipped=[])
    if automatic or "pedigree" in requested:
        if store.global_done("pedigree"):
            inferred = store.load_global("pedigree")["tier_b_relationships"]
            report["pedigree"] = pedigree_counts(inferred, simulation["truth_pedigree"], names)
        elif not automatic:
            raise FileNotFoundError("no completed genome-wide pedigree")
    component_rows, matching_rows = [], []
    with numba_thread_scope(workers):
        for contig in selected:
            source = store.load_contig("simulated_reads", contig, nthreads=workers)
            positions = np.asarray(store.load_contig("founder_templates", contig)["naive_long_haps"][0])
            found = [s for s in FOUNDER_STAGES if panel_location(store, contig, s, feedback_selection)]
            founder_stages = (found[-1:] if requested == ("available",) else
                              found if requested == ("all",) else
                              [s for s in requested if s in FOUNDER_STAGES])
            if founder_stages:
                truth = np.asarray(source["truth_founder_haplotypes"], dtype=np.int8)
                represented = represented_ancestry(source["truth_painting"], positions, len(truth))
                for stage in founder_stages:
                    blocks = load_panels(store, contig, stage, feedback_selection=feedback_selection)
                    summary, components, matches = evaluate_founders(
                        blocks, positions, truth, represented, contig=contig, stage=stage)
                    report["founders"].append(summary)
                    component_rows.extend(components)
                    matching_rows.extend(matches)
                    print(f"[EVALUATE founders] {contig}/{stage}: "
                          f"{summary['called_allele_errors']}/{summary['called_alleles']} "
                          f"called errors; {summary['truth_to_panel_errors_missing']} truth-to-panel errors/missing",
                          flush=True)
                del truth, represented, blocks
            row = dict(contig=contig, sites=len(positions))
            if automatic or "family_phase" in requested:
                if store.contig_done("family_phase", contig):
                    final = store.load_contig("family_phase", contig, nthreads=workers)
                    calls, truth = final["phase"].allele_calls, np.asarray(source["truth_alleles"], dtype=np.int8)
                    if tuple(final["sample_ids"]) != names or truth.shape != calls.shape:
                        raise ValueError(f"{contig}: incompatible true/final allele axes")
                    if not np.array_equal(final["positions"], positions):
                        raise ValueError(f"{contig}: final positions differ from truth")
                    values = phase_counts(calls, truth, final["component_ids"])
                    row.update(total_alleles=int(calls.size), **dict(zip(PHASE_COLUMNS, map(int, values))))
                    row["called_fraction"] = row["called_alleles"]/row["total_alleles"]
                    del final, calls, truth
                elif not automatic:
                    raise FileNotFoundError(f"{contig}: no completed family phase")
                else:
                    report["skipped"].append(f"{contig}/family_phase")
            if automatic or "recombination" in requested:
                if store.contig_done("recombination", contig):
                    events = {(r["parent"], r["child"]): r["crossover_positions_bp"]
                              for r in source["truth_crossovers"]}
                    row.update(map_counts(store.load_contig("recombination", contig), events, names))
                elif not automatic:
                    raise FileNotFoundError(f"{contig}: no completed recombination map")
                else:
                    report["skipped"].append(f"{contig}/recombination")
            report["chromosomes"].append(row)
            del source
    report["totals"] = {key: sum(row.get(key, 0) for row in report["chromosomes"])
                        for key in (*PHASE_COLUMNS, "total_alleles")}
    report["interpretation"] = (
        "Truth is used only for evaluation. Founder labels match over whole components, "
        "not marker-wise; nearest-row errors and one-to-one completeness are separate. "
        "Represented ancestry means present in a sampled homologue, not independently identifiable. "
        "Phase switches break at gaps/incorrect heterozygotes. Map truth uses correct edges "
        "within inferred observable spans. Unexecuted stages are not evaluated.")
    destination = output/"evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination/"summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(destination/"summary.json")
    for filename, rows in (("chromosomes.csv", report["chromosomes"]),
                           ("founders.csv", report["founders"]),
                           ("founder_components.csv", component_rows),
                           ("founder_matches.csv", matching_rows)):
        _write_table(destination, filename, rows)
    return report
