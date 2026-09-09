"""Simulation-template matching and post-inference evaluation against known truth."""
from __future__ import annotations

import json
from pathlib import Path
from numba import njit, prange
import pandas as pd

import numpy as np


np.seterr(divide='ignore',invalid="ignore")


def match_best_vectorised(haps_dict, diploids, keep_flags=None):
    """
    Vectorized matching of diploid samples to haplotype pairs.
    Uses Matrix Multiplication for high performance.
    """
    diploids = np.array(diploids)
    num_samples, total_sites, _ = diploids.shape

    if keep_flags is None:
        keep_flags = slice(None)
    elif keep_flags.dtype != bool:
        keep_flags = np.array(keep_flags, dtype=bool)

    diploids_masked = diploids[:, keep_flags, :]
    masked_sites = diploids_masked.shape[1]

    if masked_sites == 0:
        return ([], {}, np.zeros(num_samples))

    diploids_flat = diploids_masked.reshape(num_samples, -1)

    hap_keys = list(haps_dict.keys())
    num_haps = len(hap_keys)

    if num_haps == 0:
        return ([], {}, np.zeros(num_samples))

    # Stack haps: (Num_Haps, Masked_Sites, 2)
    hap_tensor = np.array([haps_dict[k][keep_flags] for k in hap_keys])

    p0 = hap_tensor[:, :, 0]
    p1 = hap_tensor[:, :, 1]

    # Broadcasting: (Num_Haps, Num_Haps, Masked_Sites)
    # This generates the probability for every possible combination (i, j)
    prob_00 = p0[:, None, :] * p0[None, :, :]
    prob_11 = p1[:, None, :] * p1[None, :, :]
    prob_01 = (p0[:, None, :] * p1[None, :, :]) + (p1[:, None, :] * p0[None, :, :])

    combinations_4d = np.stack([prob_00, prob_01, prob_11], axis=-1)
    # Reshape to (N*N, Sites, 3)
    combinations_list = combinations_4d.reshape(-1, masked_sites, 3)

    # Calculate expected distance for each pair against [0,1,2] states
    dist_weights = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    combinations_weighted = combinations_list @ dist_weights
    combinations_weighted_flat = combinations_weighted.reshape(-1, masked_sites * 3)

    # Matrix Mult: (Samples, Features) @ (Combinations, Features).T
    dists = diploids_flat @ combinations_weighted_flat.T
    dists *= (100.0 / masked_sites)

    best_indices_flat = np.argmin(dists, axis=1)
    best_errors = dists[np.arange(num_samples), best_indices_flat]

    # Map flat index back to (i, j)
    idx_grid_i, idx_grid_j = np.indices((num_haps, num_haps))
    idx_grid_i = idx_grid_i.flatten()
    idx_grid_j = idx_grid_j.flatten()

    best_parents_i = idx_grid_i[best_indices_flat]
    best_parents_j = idx_grid_j[best_indices_flat]

    all_used = np.concatenate([best_parents_i, best_parents_j])
    unique_idx, counts = np.unique(all_used, return_counts=True)

    haps_usage = {k: 0 for k in hap_keys}
    for idx, count in zip(unique_idx, counts):
        haps_usage[hap_keys[idx]] = count

    dips_matches = [
        ((hap_keys[p1], hap_keys[p2]), err)
        for p1, p2, err in zip(best_parents_i, best_parents_j, best_errors)
    ]

    return (dips_matches, haps_usage, best_errors)


def combined_best_hap_matches(block_result):
    if hasattr(block_result, 'haplotypes'):
        reads_array = block_result.reads_count_matrix
        haps = block_result.haplotypes
        keep_flags = getattr(block_result, 'keep_flags', None)
        probs_array = getattr(block_result, 'probs_array', None)
    else:
        # Assuming tuple structure (pos, keep_flags, reads, haps)
        keep_flags = block_result[1]
        reads_array = block_result[2]
        haps = block_result[3]
        probs_array = None

    # Handle Empty Block Case
    if len(haps) == 0:
        return ([], {}, [])

    # Determine which probability source to use
    if reads_array is not None and reads_array.size > 0:
        # Prefer computing from reads (original behavior)
        (site_priors, actual_probs) = core_numerics.reads_to_probabilities(
            reads_array,
            use_hwe_prior=False,
        )
    elif probs_array is not None and probs_array.size > 0:
        # Fallback to pre-computed probs_array (when reads discarded for memory)
        actual_probs = probs_array
    else:
        # No probability data available
        return ([], {}, [])

    matches = match_best_vectorised(haps, actual_probs, keep_flags=keep_flags)
    return matches


def relative_haplotype_usage(first_hap, first_matches, second_matches):
    """
    Calculates usages of haplotypes in the second block for samples that
    used 'first_hap' in the first block.

    Includes bounds checking to handle cases where one block is empty/invalid.
    """
    use_indices = []

    # 1. Validate Inputs
    if not first_matches or len(first_matches) < 1: return {}
    if not second_matches or len(second_matches) < 1: return {}

    match_list_1 = first_matches[0]
    match_list_2 = second_matches[0]

    if not match_list_1 or not match_list_2: return {}

    len_1 = len(match_list_1)
    len_2 = len(match_list_2)

    # 2. Collect Indices from First Block
    for sample_idx, (parents, _) in enumerate(match_list_1):
        if first_hap in parents:
            # Only add if this sample ALSO exists in the second block
            if sample_idx < len_2:
                use_indices.append(sample_idx)

    second_usages = {}

    # 3. Aggregate Usage in Second Block
    for sample_idx in use_indices:
        # Tuple unpacking safety
        entry = match_list_2[sample_idx]
        if entry:
            parents_2, _ = entry
            for parent in parents_2:
                second_usages[parent] = second_usages.get(parent, 0) + 1

    return dict(sorted(second_usages.items(), key=lambda item: item[1]))


def hap_matching_comparison(haps_data, matches_data, first_block_index, second_block_index):
    forward_scores = {}
    backward_scores = {}

    b1 = haps_data[first_block_index]
    b2 = haps_data[second_block_index]

    first_haps_dict = b1.haplotypes if hasattr(b1, 'haplotypes') else b1[3]
    second_haps_dict = b2.haplotypes if hasattr(b2, 'haplotypes') else b2[3]

    # Handle empty blocks
    if not first_haps_dict or not second_haps_dict:
        return ({}, {})

    first_matches = matches_data[first_block_index]
    second_matches = matches_data[second_block_index]

    # Validate match data structure
    if not first_matches or not second_matches:
        return ({}, {})

    for hap in first_haps_dict.keys():
        hap_usages = relative_haplotype_usage(hap, first_matches, second_matches)
        total_matches = sum(hap_usages.values())
        if total_matches == 0: continue

        hap_percs = {x: 100 * count / total_matches for x, count in hap_usages.items()}

        for other_hap in second_haps_dict.keys():
            perc = hap_percs.get(other_hap, 0)
            scaled_val = 100 * (min(1, 2 * perc / 100))**2
            key = ((first_block_index, hap), (second_block_index, other_hap))
            forward_scores[key] = scaled_val

    for hap in second_haps_dict.keys():
        hap_usages = relative_haplotype_usage(hap, second_matches, first_matches)
        total_matches = sum(hap_usages.values())
        if total_matches == 0: continue

        hap_percs = {x: 100 * count / total_matches for x, count in hap_usages.items()}

        for other_hap in first_haps_dict.keys():
            perc = hap_percs.get(other_hap, 0)
            scaled_val = 100 * (min(1, 2 * perc / 100))**2
            key = ((first_block_index, other_hap), (second_block_index, hap))
            backward_scores[key] = scaled_val

    return (forward_scores, backward_scores)


def get_block_hap_similarities(block_result):
    scores = []

    if hasattr(block_result, 'haplotypes'):
        hap_vals = block_result.haplotypes
        flags = getattr(block_result, 'keep_flags', None)
    else:
        hap_vals = block_result[3]
        flags = block_result[1]

    if not hap_vals:
        return np.array([])

    if flags is None:
        any_key = next(iter(hap_vals))
        flags = np.ones(len(hap_vals[any_key]), dtype=bool)
    else:
        flags = np.array(flags, dtype=bool)

    keys = sorted(hap_vals.keys())

    for i in keys:
        row_scores = []
        for j in keys:
            if j < i:
                row_scores.append(0)
            else:
                first_hap = hap_vals[i][flags]
                second_hap = hap_vals[j][flags]
                hap_len = len(first_hap)

                if hap_len == 0:
                    similarity = 0
                else:
                    dist = core_numerics.calc_distance(first_hap, second_hap, calc_type="haploid")
                    scoring = 2.0 * dist / hap_len
                    similarity = 1.0 - min(1.0, scoring)

                row_scores.append(similarity)
        scores.append(row_scores)

    scores = np.array(scores)
    scores = scores + scores.T - np.diag(scores.diagonal())

    scr_diag = np.sqrt(scores.diagonal())
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = scores / scr_diag
        scores = scores / scr_diag.reshape(1, -1).T

    return scores

import haplotype_reconstruction.core.numerics as core_numerics


@njit(cache=True, parallel=True)
def phase_counts(calls, truth, components):
    """Count called coverage and phase errors without crossing unsupported links.

    Compare consecutive true heterozygotes only when both calls have the right
    genotype and belong to the same supported component. An uncalled/incorrect
    intervening heterozygote breaks the comparison. Allele mismatch counts are
    aligned by one arbitrary strand swap per sample and component.
    """
    samples, sites, _ = calls.shape
    result = np.zeros((samples, 7), dtype=np.int64)
    component_count = max(0, int(components.max()) + 1)
    for sample in prange(samples):
        direct = np.zeros(component_count, dtype=np.int64)
        swapped = np.zeros(component_count, dtype=np.int64)
        previous_valid = False
        previous_component = -1
        previous_flip = False
        for site in range(sites):
            a, b = calls[sample, site]
            x, y = truth[sample, site]
            component = components[site]
            result[sample, 0] += int(a >= 0) + int(b >= 0)
            complete = a >= 0 and b >= 0
            result[sample, 1] += int(complete)
            correct = complete and a + b == x + y
            result[sample, 2] += int(complete and not correct)
            if component >= 0:
                direct[component] += int(a >= 0 and a != x) + int(b >= 0 and b != y)
                swapped[component] += int(a >= 0 and a != y) + int(b >= 0 and b != x)
            if x != y:
                valid = correct and component >= 0
                result[sample, 3] += int(valid)
                flip = a != x
                if valid and previous_valid and component == previous_component:
                    result[sample, 4] += 1
                    result[sample, 5] += int(flip != previous_flip)
                previous_valid = valid
                previous_component = component
                previous_flip = flip
        for component in range(component_count):
            result[sample, 6] += min(direct[component], swapped[component])
    return result.sum(axis=0)


def evaluate_run(output_dir, *, cores=None):
    """Evaluate completed simulated outputs; never provide truth to inference."""
    from haplotype_reconstruction.core.runtime import CheckpointStore, available_cpu_count
    from haplotype_reconstruction.core.parallel import numba_thread_scope
    from haplotype_reconstruction.pedigree.pipeline import PEDIGREE_STAGE
    from haplotype_reconstruction.refinement.pipeline import FINAL_PHASE_STAGE
    from haplotype_reconstruction.recombination.pipeline import RECOMBINATION_STAGE

    output = Path(output_dir).resolve()
    store = CheckpointStore(output / "checkpoints")
    workers = available_cpu_count() if cores is None else int(cores)
    if not 1 <= workers <= available_cpu_count():
        raise ValueError("evaluation cores must fit the current affinity")
    for stage in ("00_simulated_reads", PEDIGREE_STAGE, FINAL_PHASE_STAGE, RECOMBINATION_STAGE):
        if not store.stage_complete(stage):
            raise ValueError(f"evaluation requires completed {stage}")
    simulation = store.load_global("00_simulated_reads")
    inferred = store.load_global(PEDIGREE_STAGE)["tier_b_relationships"]
    truth_pedigree = simulation["truth_pedigree"].set_index("Sample")
    names = tuple(simulation["sample_names"])
    if tuple(inferred.Sample) != names:
        raise ValueError("inferred and true sample axes differ")
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
    report = {"seed": simulation["simulation_seed"], "samples": len(names),
        "pedigree": {"exact_configurations": exact, "correct_parent_states": correct_states,
            "expected_states": expected_states, "true_edges": true_edges,
            "correct_edges": shared_edges, "extra_edges": inferred_edges - shared_edges,
            "missing_edges": true_edges - shared_edges}, "chromosomes": []}
    columns = ("called_alleles", "called_genotypes", "genotype_errors", "eligible_heterozygotes",
               "phase_comparisons", "phase_switch_errors", "component_aligned_allele_errors")
    with numba_thread_scope(workers):
        for contig in simulation["region_keys"]:
            source = store.load_contig("00_simulated_reads", contig, nthreads=workers)
            truth = np.asarray(source["truth_alleles"], dtype=np.int8)
            events = {(r["parent"], r["child"]): r["crossover_positions_bp"] for r in source["truth_crossovers"]}
            del source
            final = store.load_contig(FINAL_PHASE_STAGE, contig, nthreads=workers)
            calls = final["phase"].allele_calls
            if tuple(final["sample_ids"]) != names or truth.shape != calls.shape:
                raise ValueError(f"{contig}: incompatible true/final allele axes")
            values = phase_counts(calls, truth, final["component_ids"])
            row = {"contig": contig, "sites": int(calls.shape[1]), "total_alleles": int(calls.size),
                   **dict(zip(columns, map(int, values)))}
            row["called_fraction"] = row["called_alleles"] / row["total_alleles"]
            del final, calls, truth
            mapping = store.load_contig(RECOMBINATION_STAGE, contig, nthreads=workers)
            true_observed = 0
            expected = exposure = 0.0
            unmatched = 0
            for edge, (parent, child) in enumerate(zip(mapping["edge_parent"], mapping["edge_child"])):
                raw = events.get((names[int(parent)], names[int(child)]))
                if raw is None:
                    unmatched += 1
                    continue
                for span in mapping["informative_spans"][edge]:
                    left, right = span[:2]
                    true_observed += int(np.searchsorted(raw, right, side="right") - np.searchsorted(raw, left, side="right"))
                expected += float(mapping["expected_crossovers_by_edge_bin"][edge].sum())
                exposure += float(mapping["exposure_by_edge_bin"][edge].sum())
            row.update(true_crossovers_in_inferred_exposure=true_observed,
                expected_crossovers_on_correct_edges=expected, correct_edge_exposure_meiosis_bp=exposure,
                inferred_edges_absent_from_truth=unmatched)
            report["chromosomes"].append(row)
            print(f"[EVALUATE] {contig}: {row['phase_switch_errors']} phase switches; {row['called_fraction']:.4%} called", flush=True)
            del mapping, events
    report["totals"] = {key: sum(row[key] for row in report["chromosomes"]) for key in (*columns, "total_alleles")}
    report["interpretation"] = "Known-truth simulation only. Component-aligned phase; gaps/incorrect heterozygotes break switch comparisons. Map truth uses only correctly inferred edges within inferred observable spans."
    destination = output / "evaluation"
    destination.mkdir(exist_ok=True)
    temporary = destination / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination / "summary.json")
    pd.DataFrame(report["chromosomes"]).to_csv(destination / "chromosomes.csv", index=False)
    return report
