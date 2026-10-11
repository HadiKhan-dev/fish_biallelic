"""Mesh path search, source provenance and haplotype reconstruction."""
from __future__ import annotations


import numpy as np
import math


class FastMesh:
    """
    Optimized container for the Transition Mesh.

    Converts the sparse dictionary-based probabilities from the HMM step into
    dense NumPy log-probability matrices. This allows for O(1) lookups and
    vectorized broadcasting during the Beam Search.

    Stores the forward transition matrices P(later_hap | earlier_hap) that the
    endpoint beam scores paths with.

    Attributes:
        num_blocks (int): Total number of genomic blocks.
        mappings (list): List of dicts mapping {real_hap_key: dense_index} for each block.
        reverse_mappings (list): List of lists mapping [dense_index] -> real_hap_key.
        registry (dict): Forward matrices. registry[from_block][to_block] = Log_Prob_Matrix (N_from x N_to).
    """
    def __init__(self, block_results, transition_mesh):
        """
        Initialize the FastMesh.

        Args:
            block_results (BlockResults): List of BlockResult objects containing local haplotypes.
            transition_mesh (TransitionMesh): The sparse mesh calculated by block_linking_em.
        """
        self.num_blocks = len(block_results)

        # 1. Build Index Mappings (Dense ID <-> Real Key)
        self.mappings = []
        self.reverse_mappings = []

        for i in range(self.num_blocks):
            keys = sorted(list(block_results[i].haplotypes.keys()))
            self.mappings.append({k: idx for idx, k in enumerate(keys)})
            self.reverse_mappings.append(keys)

        # 2. Build Forward Dense Matrices Registry
        self.registry = {}

        for gap in transition_mesh.keys():
            # Forward Dictionary: P(Next | Curr)
            fwd_dict = transition_mesh[gap][0]

            if fwd_dict is None:
                continue

            for i_idx, transitions in fwd_dict.items():
                j_idx = i_idx + gap

                if j_idx >= self.num_blocks:
                    continue

                n_from = len(self.mappings[i_idx])
                n_to = len(self.mappings[j_idx])

                mat = np.full((n_from, n_to), -np.inf, dtype=np.float32)

                for (key_from, key_to), prob in transitions.items():
                    u_key = key_from[1]
                    v_key = key_to[1]

                    if u_key in self.mappings[i_idx] and v_key in self.mappings[j_idx]:
                        r = self.mappings[i_idx][u_key]
                        c = self.mappings[j_idx][v_key]
                        mat[r, c] = math.log(prob)

                if i_idx not in self.registry:
                    self.registry[i_idx] = {}
                self.registry[i_idx][j_idx] = mat

    def get_transition_matrix(self, from_block, to_block):
        """
        Returns the forward dense log-probability matrix P(to_block | from_block).
        from_block < to_block.
        Returns None if no transition data exists for this pair.
        """
        if from_block in self.registry and to_block in self.registry[from_block]:
            return self.registry[from_block][to_block]
        return None

    def get_key_from_dense(self, block_idx, dense_idx):
        """Converts internal dense integer index back to original haplotype key."""
        return self.reverse_mappings[block_idx][dense_idx]

    def get_num_haps(self, block_idx):
        """Returns the number of haplotypes in a specific block."""
        return len(self.reverse_mappings[block_idx])


def _compute_step_scores(path, fast_mesh, eff_max_gap):
    """
    Compute the per-step mean-transition scores for a complete path.

    Step k (for k=1..N-1) is the mean of forward transition log-probs
    from all earlier blocks within max_gap to block k.
    Step 0 is always 0.0 (no incoming transitions).

    Args:
        path: List of dense hap indices, one per block.
        fast_mesh: FastMesh object.
        eff_max_gap: Effective maximum gap for transitions.

    Returns:
        np.ndarray of shape (num_blocks,) with per-step scores.
    """
    num_blocks = len(path)
    step_scores = np.zeros(num_blocks, dtype=np.float64)

    for curr_block in range(1, num_blocks):
        earliest_past = max(0, curr_block - eff_max_gap)
        step_total = 0.0
        n_transitions = 0

        for past_idx in range(earliest_past, curr_block):
            mat = fast_mesh.get_transition_matrix(past_idx, curr_block)
            if mat is not None:
                step_total += mat[path[past_idx], path[curr_block]]
                n_transitions += 1

        if n_transitions > 0:
            step_scores[curr_block] = step_total / n_transitions

    return step_scores


def _affected_steps(paths, step_scores, changed_block, n_choices, fast_mesh, eff_max_gap):
    """
    Per-step scores of every path with every haplotype choice at one block.

    When block k changes, the affected steps are step k itself (transitions
    from past blocks INTO k) and steps k+1 .. min(k + max_gap, N-1)
    (transitions FROM k into future blocks). Entry [i, h] holds the step
    scores of path i with haplotype h at block k. Each element follows the
    scalar recursion exactly: float32 accumulation in past-block order, float32
    division by the transition count, float64 storage.

    Returns:
        (n_paths, n_choices, num_blocks) step scores.
    """
    n_paths, num_blocks = paths.shape
    steps = np.repeat(step_scores[:, None,:], n_choices, axis=1)
    recompute_start = max(1, changed_block)  # step 0 is always 0
    recompute_end = min(changed_block + eff_max_gap, num_blocks - 1)

    for curr_block in range(recompute_start, recompute_end + 1):
        earliest_past = max(0, curr_block - eff_max_gap)
        step_total = np.zeros((n_paths, n_choices), dtype=np.float32)
        n_transitions = 0

        for past_idx in range(earliest_past, curr_block):
            mat = fast_mesh.get_transition_matrix(past_idx, curr_block)
            if mat is not None:
                if curr_block == changed_block:
                    step_total += mat[paths[:, past_idx],:]
                elif past_idx == changed_block:
                    step_total += mat[:, paths[:, curr_block]].T
                else:
                    step_total += mat[paths[:, past_idx], paths[:, curr_block]][:, None]
                n_transitions += 1

        if n_transitions > 0:
            steps[:,:, curr_block] = step_total / n_transitions
        else:
            steps[:,:, curr_block] = 0.0

    return steps


def _endpoint_order(paths, scores, block, quota, overrides):
    """Indices kept by the endpoint quota, in selection order.

    Order: descending score, then lexicographic path (the scores' float64
    values); identical paths are kept once; at most quota (or its override)
    paths per haplotype at `block`.
    """
    order = np.lexsort(tuple(paths[:, column] for column in range(paths.shape[1] - 1, -1, -1))
                       + (-np.asarray(scores, dtype=np.float64),))
    ordered = paths[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    order = order[keep]
    endpoint = paths[order, block]
    # Rank of every candidate among those with the same endpoint so far.
    by_endpoint = np.argsort(endpoint, kind='stable')
    sorted_endpoint = endpoint[by_endpoint]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_endpoint)) + 1]
    sizes = np.diff(np.r_[starts, len(sorted_endpoint)])
    rank = np.empty(len(order), dtype=np.int64)
    rank[by_endpoint] = np.arange(len(order)) - np.repeat(starts, sizes)
    limit = np.full(len(order), quota, dtype=np.int64)
    if overrides:
        for (override_block, haplotype), value in overrides.items():
            if override_block == block:
                limit[endpoint == haplotype] = value
    return order[rank < limit]


def run_endpoint_beam(fast_mesh, *, endpoint_quota, endpoint_quota_overrides=None,
                      max_gap=None, verbose=True):
    """
    Bidirectional endpoint beam with scaffold-based backward refinement.

    Pass 1 (Forward): builds full paths left to right; at every block the
    candidates are ranked by score (ties by path) and at most a quota of them
    is kept per endpoint haplotype.

    Pass 2 (Backward Refinement): uses each forward path as a scaffold. At each
    block k (from N-1 down to 0) it tries every haplotype for block k with the
    rest of the scaffold fixed, scores the full path from cached per-step scores
    (recomputing only the steps the change affects), pools the candidates across
    scaffolds and selects them with the same endpoint quota. Every selected path
    is kept in an archive, so full paths representing interior states survive
    (at most B times the largest per-block sum of endpoint quotas).

    Candidates are scored and selected as arrays; every score follows the same
    float32/float64 operations as the per-path recursion.

    Args:
        fast_mesh: FastMesh of the batch.
        endpoint_quota: Paths kept per endpoint haplotype at every block.
        endpoint_quota_overrides: Optional {(block, haplotype): quota} for
                 endpoints whose quota differs (the panel search's pool
                 extension and saturation re-runs).
        max_gap: Maximum gap between blocks to consider transitions for.
                 If None, use all available transitions (no limit).
        verbose: If True, print progress.

    Returns:
        List of (path_indices, score) sorted by score descending.
    """
    num_blocks = fast_mesh.num_blocks

    if num_blocks < 2:
        n_0 = fast_mesh.get_num_haps(0)
        return [([h], 0.0) for h in range(n_0)]

    # Effective max_gap: if None, use all blocks
    eff_max_gap = max_gap if max_gap is not None else num_blocks

    if verbose:
        print(f"  max_gap={max_gap} (effective: {eff_max_gap})")

    # ========== FORWARD PASS ==========
    # MEAN scoring: add the mean of the transitions into each new block.
    if verbose:
        print(f"Pass 1: Forward Beam (blocks 0 to {num_blocks-1})...")

    n_0 = fast_mesh.get_num_haps(0)
    beam_paths = np.arange(n_0, dtype=np.int64)[:, None]
    beam_scores = np.zeros(n_0, dtype=np.float32)

    for curr_block in range(1, num_blocks):
        n_curr = fast_mesh.get_num_haps(curr_block)
        earliest_past = max(0, curr_block - eff_max_gap)
        transition_to_curr = np.zeros((len(beam_paths), n_curr), dtype=np.float32)
        n_transitions = 0
        for past_idx in range(earliest_past, curr_block):
            mat = fast_mesh.get_transition_matrix(past_idx, curr_block)
            if mat is not None:
                transition_to_curr += mat[beam_paths[:, past_idx],:]
                n_transitions += 1
        if n_transitions > 0:
            mean_transition = transition_to_curr / n_transitions
        else:
            mean_transition = transition_to_curr
        scores = (beam_scores[:, None] + mean_transition).ravel()
        paths = np.empty((len(beam_paths) * n_curr, curr_block + 1), dtype=np.int64)
        paths[:,:curr_block] = np.repeat(beam_paths, n_curr, axis=0)
        paths[:, curr_block] = np.tile(np.arange(n_curr, dtype=np.int64), len(beam_paths))
        kept = _endpoint_order(paths, scores, curr_block, endpoint_quota,
                               endpoint_quota_overrides)
        beam_paths, beam_scores = paths[kept], scores[kept]

    if verbose:
        print(f"  Forward beam has {len(beam_paths)} full paths")

    # ========== BACKWARD PASS (Scaffold Refinement with Cached Scores) ==========
    if verbose:
        print(f"Pass 2: Backward Refinement (blocks {num_blocks-1} to 0)...")

    scaffold_paths = beam_paths
    scaffold_steps = np.stack([_compute_step_scores(list(path), fast_mesh, eff_max_gap)
                               for path in beam_paths.tolist()])
    scaffold_totals = np.array([np.sum(steps) for steps in scaffold_steps])
    # Preserve full paths representing interior states, not only the final
    # first-block quota (insertion order = first selection).
    archive = {}
    for path, total in zip(map(tuple, scaffold_paths.tolist()), scaffold_totals):
        archive.setdefault(path, total)

    for refine_block in range(num_blocks - 1, -1, -1):
        n_choices = fast_mesh.get_num_haps(refine_block)
        n_scaffolds = len(scaffold_paths)
        steps = _affected_steps(scaffold_paths, scaffold_steps, refine_block, n_choices,
                                fast_mesh, eff_max_gap)
        totals = steps.reshape(-1, num_blocks).sum(axis=1).reshape(n_scaffolds, n_choices)
        # An unchanged scaffold keeps its cached scores.
        original = scaffold_paths[:, refine_block]
        rows = np.arange(n_scaffolds)
        steps[rows, original] = scaffold_steps
        totals[rows, original] = scaffold_totals
        paths = np.repeat(scaffold_paths, n_choices, axis=0)
        paths[:, refine_block] = np.tile(np.arange(n_choices, dtype=np.int64), n_scaffolds)
        totals = totals.ravel()
        kept = _endpoint_order(paths, totals, refine_block, endpoint_quota,
                               endpoint_quota_overrides)
        scaffold_paths = paths[kept]
        scaffold_steps = steps.reshape(-1, num_blocks)[kept]
        scaffold_totals = totals[kept]
        for path, total in zip(map(tuple, scaffold_paths.tolist()), scaffold_totals):
            archive[path] = total

    if verbose:
        print(f"  Backward refinement complete. {len(scaffold_paths)} paths.")

    # ========== FINAL OUTPUT ==========
    # The archive as (path, score), sorted by score descending (stable).
    final_results = [(list(path), total) for path, total in archive.items()]
    final_results.sort(key=lambda x: x[1], reverse=True)

    return final_results


_ATOMIC_SOURCE_PROVENANCE_ATTRIBUTES = (
    'missing_aware_atomic_source_row_paths',
    'missing_aware_atomic_position_offsets',
    'missing_aware_atomic_position_counts',
    'missing_aware_atomic_source_row_counts',
)


def missing_aware_atomic_source_provenance(block):
    """Return validated atomic row/span provenance for a block.

    block discovery blocks do not carry hierarchy provenance, so their rows are the
    atomic sources: row ordinals follow the sorted haplotype-key order already
    used by FastMesh. Hierarchical blocks must carry the complete set of
    arrays, whose span offsets/counts exactly partition their position axis.
    """
    present = tuple(
        hasattr(block, attribute)
        for attribute in _ATOMIC_SOURCE_PROVENANCE_ATTRIBUTES
    )
    founder_count = len(block.haplotypes)
    position_count = len(block.positions)
    if not any(present):
        return (
            np.arange(founder_count, dtype=np.int64)[:, None],
            np.asarray([0], dtype=np.int64),
            np.asarray([position_count], dtype=np.int64),
            np.asarray([founder_count], dtype=np.int64),
        )
    if not all(present):
        raise ValueError(
            "block has incomplete missing-aware atomic source provenance"
        )

    arrays = tuple(
        np.asarray(getattr(block, attribute))
        for attribute in _ATOMIC_SOURCE_PROVENANCE_ATTRIBUTES
    )
    if any(
            array.dtype == np.bool_
            or not np.issubdtype(array.dtype, np.integer)
            for array in arrays):
        raise ValueError("atomic source provenance arrays must be integer-valued")
    paths, offsets, counts, row_counts = (
        np.asarray(array, dtype=np.int64) for array in arrays
    )
    if paths.ndim != 2 or paths.shape[0] != founder_count:
        raise ValueError(
            "atomic source row paths must be founder-by-atomic-span aligned"
        )
    atomic_count = paths.shape[1]
    if atomic_count < 1 or any(
            array.ndim != 1 or array.shape != (atomic_count,)
            for array in (offsets, counts, row_counts)):
        raise ValueError("atomic source span arrays are not path-column aligned")
    if np.any(counts < 0) or np.any(row_counts < 1):
        raise ValueError("atomic position counts and source row counts are invalid")
    expected_offsets = np.concatenate((
        np.asarray([0], dtype=np.int64), np.cumsum(counts[:-1])
    ))
    if (
            not np.array_equal(offsets, expected_offsets)
            or int(np.sum(counts)) != position_count):
        raise ValueError(
            "atomic position offsets/counts must exactly partition positions"
        )
    if np.any(paths < 0) or np.any(paths >= row_counts[None,:]):
        raise ValueError("atomic source row path contains an out-of-range row")
    return paths, offsets, counts, row_counts


def reconstruct_haplotypes_from_beam(beam_results, fast_mesh, haps_data):
    """Reconstruct selected paths while preserving founder uncertainty metadata."""
    reconstructed = []
    source_paths_by_block = tuple(
        missing_aware_atomic_source_provenance(block)[0]
        for block in haps_data
    )

    for path_indices, score in beam_results:
        pos_parts = []
        hap_parts = []
        discrete_parts = []
        inference_discrete_parts = []
        probability_parts = []
        support_parts = []
        atomic_source_row_path_parts = []
        has_complete_support = True
        for block_idx, dense_idx in enumerate(path_indices):
            key = fast_mesh.get_key_from_dense(block_idx, dense_idx)
            block_obj = haps_data[block_idx]
            pos_parts.append(np.asarray(block_obj.positions))
            haplotype = np.asarray(block_obj.haplotypes[key])
            hap_parts.append(haplotype)

            keys = sorted(block_obj.haplotypes)
            founder_index = keys.index(key)
            n_sites = len(block_obj.positions)
            atomic_source_row_path_parts.append(
                source_paths_by_block[block_idx][founder_index]
            )

            discrete = getattr(block_obj, 'discrete_haps', None)
            if discrete is not None:
                discrete_part = np.asarray(discrete)[founder_index]
            elif haplotype.ndim == 1:
                discrete_part = np.where(
                    np.isin(haplotype, (0, 1)), haplotype, -1
                )
            else:
                explicit_ref = (
                    (haplotype[:, 0] == 1.0)
                    & (haplotype[:, 1] == 0.0)
                )
                explicit_alt = (
                    (haplotype[:, 0] == 0.0)
                    & (haplotype[:, 1] == 1.0)
                )
                discrete_part = np.full(n_sites, -1, dtype=np.int8)
                discrete_part[explicit_ref] = 0
                discrete_part[explicit_alt] = 1
            discrete_parts.append(np.asarray(discrete_part, dtype=np.int8))

            inference_discrete = getattr(
                block_obj, 'missing_aware_inference_discrete_haps', None
            )
            if inference_discrete is None:
                inference_discrete_part = discrete_part
            else:
                inference_discrete_part = np.asarray(
                    inference_discrete
                )[founder_index]
            inference_discrete_parts.append(
                np.asarray(inference_discrete_part, dtype=np.int8)
            )

            probabilities = getattr(
                block_obj, 'founder_alt_pseudo_probability', None
            )
            if probabilities is not None:
                probability_part = np.asarray(probabilities)[founder_index]
            elif haplotype.ndim == 1:
                probability_part = haplotype
            else:
                probability_part = haplotype[:, 1]
            probability_parts.append(
                np.asarray(probability_part, dtype=np.float64)
            )

            support = getattr(
                block_obj, 'n_directional_site_supporters', None
            )
            if support is None:
                has_complete_support = False
            else:
                support_parts.append(np.asarray(support)[founder_index])

        result = {
            "score": score,
            "positions": np.concatenate(pos_parts) if pos_parts else np.array([]),
            "haplotype": np.concatenate(hap_parts) if hap_parts else np.array([]),
            "path_indices": path_indices,
            "discrete_haplotype": np.concatenate(discrete_parts),
            "missing_aware_inference_discrete_haplotype": np.concatenate(
                inference_discrete_parts
            ),
            "founder_alt_pseudo_probability": np.concatenate(
                probability_parts
            ),
            "missing_aware_atomic_source_row_path": np.concatenate(
                atomic_source_row_path_parts
            ),
        }
        if has_complete_support:
            result["n_directional_site_supporters"] = np.concatenate(
                support_parts
            )
        reconstructed.append(result)

    return reconstructed
