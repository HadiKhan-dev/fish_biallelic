"""Full-panel Viterbi scores, emissions and founder-complexity penalties."""
from __future__ import annotations


import numpy as np
import math
from . import partial_emissions


def compute_penalty(batch_blocks):
    """Compute switching penalty based on block sizes.

    L1 (200 SNP input, 2k output):   pen=10
    L2 (2k SNP input, 20k output):   pen~63
    L3 (20k SNP input, 200k output): pen~200
    """
    avg_input_sites = np.mean([len(b.positions) for b in batch_blocks])
    avg_output_sites = avg_input_sites * len(batch_blocks)
    if avg_output_sites <= 5000:
        return 20.0
    return 20.0 * math.sqrt(avg_output_sites / 2000.0)


def compute_spb(batch_blocks):
    """Compute SNPs per bin. Targets ~20 bins per input block, clamped [10, 100]."""
    avg_sites = np.mean([len(b.positions) for b in batch_blocks])
    return int(min(100, max(10, avg_sites // 20)))


def compute_cc(batch_blocks, num_samples, cc_scale=0.5):
    """Compute complexity cost per founder using actual block sizes.

    CC = cc_scale * (avg_snps / 200) * num_samples * num_blocks
    """
    avg_snps = np.mean([len(b.positions) for b in batch_blocks])
    return discovery_objectives.compute_founder_complexity_cost(
        cc_scale, num_samples, avg_snps, n_blocks=len(batch_blocks)
    )


def compute_subblock_emissions(input_blocks, global_probs, global_sites,
                               snps_per_bin, num_threads=None):
    """Compute binned diploid emission log-likelihoods for each block.

    Uses numba kernel with prange over samples when available — 10x faster
    than numpy, uses no large temporary arrays, and parallelism is controlled
    by numba_thread_scope (set in the caller).

    num_threads parameter is accepted for API compatibility but unused —
    parallelism comes from numba_thread_scope instead.

    NOTE: when num_threads is callable (e.g. _get_dynamic_threads from
    hierarchical_assembly), _resolve_threads is invoked once per block
    iteration so the per-block _compute_bin_emissions_numba kernel picks
    up the latest total_cores // active_workers value as peers finish.
    Default None preserves the original behaviour (no thread resync) for
    any existing callers that don't pass the argument.

    Partial founder observations use the same predictive distributions as the
    linker. All-unknown sites and uniform sample observations remain neutral.

    Returns list of dicts with keys: 'hap_keys', 'bin_emissions', 'n_bins'.
    bin_emissions shape: (num_samples, n_haps, n_haps, n_bins)
    """
    all_emissions = []
    for block in input_blocks:
        positions = block.positions
        n_sites = len(positions)
        n_bins = math.ceil(n_sites / snps_per_bin)
        indices = np.searchsorted(global_sites, positions)
        # Dynamic thread resync (gated on num_threads being provided, so
        # any external caller that doesn't pass it sees the original
        # "no thread management" behaviour).  Done BEFORE the gather so the
        # (multi-GB at L4) block_samples copy is parallelised as well.
        if num_threads is not None:
            assembly_chimera_kernels._resolve_threads(num_threads)
        # block_samples = np.ascontiguousarray(global_probs[:, indices, :]), but
        # via a parallel kernel.  numpy's advanced-index gather of the
        # (N, n_global, 3) probs runs single-threaded and costs ~10-20s at L4,
        # where it dominated the emission phase; _gather_samples_numba is a
        # bit-identical pure copy spread over the inner-core pool.
        block_samples = assembly_chimera_kernels._gather_samples_numba(global_probs, indices)
        hap_keys = sorted(block.haplotypes.keys())
        n_haps = len(hap_keys)

        panel = assembly_observations.founder_inference_panel_from_block_result(block)
        if panel.keys != tuple(hap_keys):
            raise ValueError("founder panel keys disagree with haplotype keys")
        block_keep = getattr(block, "keep_flags", None)
        if block_keep is None:
            block_keep = np.ones(n_sites, dtype=np.bool_)
        else:
            block_keep = np.asarray(block_keep, dtype=np.bool_)
        usable = np.ascontiguousarray(block_keep & np.any(panel.called, axis=0))
        if np.all(panel.called[:, block_keep]):
            # Preserve the optimized fully-called path exactly.
            bin_emissions = assembly_chimera_kernels._compute_bin_emissions_numba(
                block_samples, np.ascontiguousarray(panel.q), n_haps, n_bins,
                snps_per_bin, n_sites, usable)
        else:
            alleles = np.ascontiguousarray(
                np.where(panel.called, panel.q, -1), dtype=np.int8)
            bin_emissions = partial_emissions.binned_log_emissions(
                block_samples, alleles, partial_emissions.source_row_ids(block),
                usable, snps_per_bin)
        del block_samples, usable, panel
        # Precompute a {hap_key: local_index_in_hap_keys} dict so downstream
        # call sites that map path entries → local hap indices can use an
        # O(1) lookup instead of repeating list.index() (O(n_haps) per call);
        # panel_scoring.panel_indices does K such lookups per block for every
        # scored or painted panel.  Built once per block here; never mutated
        # downstream.
        key_to_local_idx = {k: i for i, k in enumerate(hap_keys)}
        all_emissions.append({
            'hap_keys': hap_keys,
            'bin_emissions': bin_emissions,
            'n_bins': n_bins,
            'key_to_local_idx': key_to_local_idx,
        })
    core_parallel.malloc_trim()
    return all_emissions


def boundary_vote_matrix(sample_paths, boundary_bin, K):
    """Vote matrix at given absolute bin index.  W[i, j] counts strand-
    units that were on path i at bin (boundary_bin - 1) and on path j
    at bin boundary_bin.  Each sample contributes 2 strand votes (one
    per diploid strand) per boundary.
    """
    W = np.zeros((K, K), dtype=np.float64)
    n_samples_, _ = sample_paths.shape
    for s in range(n_samples_):
        pre = int(sample_paths[s, boundary_bin - 1])
        post = int(sample_paths[s, boundary_bin])
        pre_a, pre_b = pre // K, pre % K
        post_a, post_b = post // K, post % K
        W[pre_a, post_a] += 1
        W[pre_b, post_b] += 1
    return W


import haplotype_reconstruction.assembly.chimera_kernels as assembly_chimera_kernels
import haplotype_reconstruction.assembly.observations as assembly_observations
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.discovery.objectives as discovery_objectives
