"""
Residual Discovery Module

Discovers missing founder haplotypes that HDBSCAN failed to find,
typically due to high IBS with an existing founder in that region.

Algorithm:
  1. Paint samples against K known founders per block
  2. Identify high-residual samples (carrying the missing founder)
  3. Subtract the known partner to extract the missing founder's alleles
  4. Iteratively refine via re-painting and re-extraction
  5. Filter via consistency (neighbour overlap), residual reduction,
     and structural chimera pruning

Runs post-refinement, pre-L1 assembly. Adds discovered haplotypes
to blocks and lets existing dedup + chimera pruning handle cleanup.

Main entry point: discover_missing_haplotypes()
"""

import numpy as np
import math
import time
import multiprocessing as mp
import multiprocessing.pool
from multiprocessing.shared_memory import SharedMemory

import block_haplotypes
import block_haplotype_refinement


# =============================================================================
# MULTIPROCESSING INFRASTRUCTURE
# =============================================================================

try:
    _forkserver_ctx = mp.get_context('forkserver')
except ValueError:
    _forkserver_ctx = mp.get_context('fork')


class _ForkserverPool(multiprocessing.pool.Pool):
    """A Pool using forkserver context."""
    def __init__(self, *args, **kwargs):
        kwargs['context'] = _forkserver_ctx
        super().__init__(*args, **kwargs)


# Worker globals — set by _init_pass1_worker, used by _pass1_worker
_RD_SHM_REF = None          # SharedMemory handle (kept alive for worker lifetime)
_RD_GLOBAL_PROBS = None     # numpy view into shared memory
_RD_NUM_SAMPLES = None


def _init_pass1_worker(probs_shm_name, probs_shape, probs_dtype, num_samples):
    """Initializer for Pass 1 pool workers.
    
    Attaches to shared memory containing global_probs once per worker
    (not per task), avoiding repeated attach/detach overhead.
    Also caps numba threads to 1 to prevent oversubscription from
    imported modules.
    """
    global _RD_SHM_REF, _RD_GLOBAL_PROBS, _RD_NUM_SAMPLES
    _RD_SHM_REF = SharedMemory(name=probs_shm_name, create=False)
    _RD_GLOBAL_PROBS = np.ndarray(probs_shape, dtype=probs_dtype, buffer=_RD_SHM_REF.buf)
    _RD_NUM_SAMPLES = num_samples
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass


def _pass1_worker(args):
    """Process a single block for Pass 1 residual discovery.
    
    Slices block_probs from shared global_probs and runs the core
    residual discovery algorithm. Returns (bi, new_hap, n_assigned,
    assigned_set) or None if no discovery.
    """
    bi, hap_array, site_indices, min_assigned = args
    block_probs = _RD_GLOBAL_PROBS[:, site_indices, :]
    
    new_hap, n_assigned, assigned_set = _residual_discover_core(
        hap_array, block_probs, _RD_NUM_SAMPLES)
    
    if new_hap is not None and n_assigned >= min_assigned:
        return (bi, new_hap, n_assigned, assigned_set)
    return None


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _get_block_hap_array(block):
    """Extract concretised haplotype array from a BlockResult.
    
    Returns:
        (hap_array, hap_keys) where hap_array is (K, n_sites) int8.
    """
    known_haps = {}
    for h_key, hap in block.haplotypes.items():
        if hap.ndim > 1:
            known_haps[h_key] = np.argmax(hap, axis=1)
        else:
            known_haps[h_key] = hap.copy()
    hap_keys = sorted(known_haps.keys())
    return np.array([known_haps[k] for k in hap_keys]), hap_keys


def _paint_samples_fast(hap_array, block_probs, num_samples):
    """Paint samples against known haplotypes using diploid likelihood.
    
    For each sample, finds the best diploid pair (i, j) and computes
    a per-site residual measuring unexplained signal.
    
    Args:
        hap_array: (K, n_sites) concretised haplotypes.
        block_probs: (num_samples, n_sites, 3) genotype probabilities.
        num_samples: Number of samples.
    
    Returns:
        (best_pairs_decoded, best_ll, residuals)
        best_pairs_decoded: List of (i, j) tuples per sample.
        best_ll: (num_samples,) log-likelihood of best pair.
        residuals: (num_samples,) mean per-site residual.
    """
    K, n_sites = hap_array.shape
    
    pairs = []
    for i in range(K):
        for j in range(i, K):
            pairs.append((i, j))
    
    best_pair = np.zeros(num_samples, dtype=int)
    best_ll = np.full(num_samples, -np.inf)
    
    for pi, (i, j) in enumerate(pairs):
        expected = hap_array[i] + hap_array[j]
        ll = np.zeros(num_samples)
        for s in range(n_sites):
            ll += np.log(np.maximum(block_probs[:, s, expected[s]], 1e-10))
        better = ll > best_ll
        best_ll[better] = ll[better]
        best_pair[better] = pi
    
    residuals = np.zeros(num_samples)
    for si in range(num_samples):
        pi = best_pair[si]
        i, j = pairs[pi]
        expected = hap_array[i] + hap_array[j]
        for s in range(n_sites):
            residuals[si] += 1.0 - block_probs[si, s, expected[s]]
    residuals /= n_sites
    
    best_pairs_decoded = [pairs[best_pair[si]] for si in range(num_samples)]
    return best_pairs_decoded, best_ll, residuals


def _extract_missing_hap(hap_array, block_probs, sample_indices, partner_indices):
    """Extract a missing haplotype by subtracting known partners.
    
    For each high-residual sample, we know it carries the missing founder
    paired with a known partner. We subtract the partner's contribution
    to recover the missing founder's alleles via soft voting.
    
    Args:
        hap_array: (K, n_sites) known haplotypes.
        block_probs: (num_samples, n_sites, 3) genotype probabilities.
        sample_indices: Indices of samples carrying the missing founder.
        partner_indices: Index into hap_array of each sample's known partner.
    
    Returns:
        (n_sites,) int8 array of the extracted haplotype.
    """
    K, n_sites = hap_array.shape
    allele_votes_0 = np.zeros(n_sites, dtype=np.float64)
    allele_votes_1 = np.zeros(n_sites, dtype=np.float64)
    
    for idx, si in enumerate(sample_indices):
        partner = hap_array[partner_indices[idx]]
        for s in range(n_sites):
            a = partner[s]
            p0 = block_probs[si, s, a]
            p1 = block_probs[si, s, min(a + 1, 2)]
            total = p0 + p1
            if total > 1e-10:
                allele_votes_0[s] += p0 / total
                allele_votes_1[s] += p1 / total
    
    new_hap = (allele_votes_1 > allele_votes_0).astype(np.int8)
    return new_hap


def _iterative_refine(base_hap_array, new_hap, block_probs, num_samples, max_rounds=10):
    """Iteratively refine a discovered haplotype.
    
    After initial extraction, re-paint samples with the expanded set
    (known + new), re-identify which samples use the new hap, and
    re-extract. Converges in 2-5 rounds typically.
    
    Returns:
        (refined_hap, n_assigned, assigned_set)
    """
    K_base = base_hap_array.shape[0]
    assigned_set = set()
    
    for round_i in range(max_rounds):
        expanded = np.vstack([base_hap_array, new_hap.reshape(1, -1)])
        new_idx = K_base
        
        best_pairs, _, _ = _paint_samples_fast(expanded, block_probs, num_samples)
        
        assigned_samples = []
        assigned_partners = []
        for si in range(num_samples):
            i, j = best_pairs[si]
            if i == new_idx or j == new_idx:
                partner = j if i == new_idx else i
                assigned_samples.append(si)
                assigned_partners.append(partner)
        
        if len(assigned_samples) < 3:
            break
        
        prev_hap = new_hap.copy()
        new_hap = _extract_missing_hap(
            expanded, block_probs, assigned_samples, assigned_partners)
        assigned_set = set(assigned_samples)
        
        if np.sum(new_hap != prev_hap) == 0:
            break
    
    return new_hap, len(assigned_set), assigned_set


def _residual_discover_core(base_hap_array, block_probs, num_samples):
    """Core residual discovery logic operating on pre-concretised arrays.
    
    Identifies samples with unusually high residuals (MAD-based outlier
    detection), finds their best-fit known partner, subtracts to extract
    the missing haplotype, then iteratively refines.
    
    Args:
        base_hap_array: (K, n_sites) int8 concretised haplotypes.
        block_probs: (num_samples, n_sites, 3) genotype probabilities.
        num_samples: Number of samples.
    
    Returns:
        (new_hap, n_assigned, assigned_set) or (None, 0, set()) if no signal.
    """
    K_base = base_hap_array.shape[0]
    n_sites = block_probs.shape[1]
    
    best_pairs, best_ll, residuals = _paint_samples_fast(
        base_hap_array, block_probs, num_samples)
    
    # MAD-based outlier detection for high-residual samples
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))
    threshold = median_res + 3 * max(mad, 0.01)
    high_indices = np.where(residuals > threshold)[0]
    
    if len(high_indices) < 5:
        return None, 0, set()
    
    # Find best partner for each high-residual sample
    partner_indices = []
    for si in high_indices:
        best_ki = -1
        best_ki_ll = -np.inf
        for ki in range(K_base):
            partner = base_hap_array[ki]
            ll = 0.0
            for s in range(n_sites):
                a = partner[s]
                p0 = block_probs[si, s, a]
                p1 = block_probs[si, s, min(a + 1, 2)]
                ll += np.log(max(p0 + p1, 1e-10))
            if ll > best_ki_ll:
                best_ki_ll = ll
                best_ki = ki
        partner_indices.append(best_ki)
    
    new_hap = _extract_missing_hap(
        base_hap_array, block_probs, high_indices, partner_indices)
    new_hap, n_assigned, assigned_set = _iterative_refine(
        base_hap_array, new_hap, block_probs, num_samples)
    return new_hap, n_assigned, assigned_set


def _residual_discover(block, block_probs, num_samples):
    """Attempt to discover a missing haplotype via residual analysis.
    
    Wrapper that extracts concretised hap array from the block,
    then delegates to _residual_discover_core.
    
    Returns:
        (new_hap, n_assigned, assigned_set) or (None, 0, set()) if no signal.
    """
    base_hap_array, hap_keys = _get_block_hap_array(block)
    return _residual_discover_core(base_hap_array, block_probs, num_samples)


def _seeded_discover(block, block_probs, seed_samples, num_samples):
    """Discover a missing haplotype using seed samples from a neighbour block.
    
    Instead of MAD-based outlier detection, uses a pre-identified set of
    samples known to carry the missing founder in adjacent blocks.
    
    Returns:
        (new_hap, n_assigned, assigned_set) or (None, 0, set()) if too few seeds.
    """
    base_hap_array, hap_keys = _get_block_hap_array(block)
    K_base = len(hap_keys)
    n_sites = block_probs.shape[1]
    
    seed_list = sorted(seed_samples)
    if len(seed_list) < 3:
        return None, 0, set()
    
    partner_indices = []
    for si in seed_list:
        best_ki = -1
        best_ki_ll = -np.inf
        for ki in range(K_base):
            partner = base_hap_array[ki]
            ll = 0.0
            for s in range(n_sites):
                a = partner[s]
                p0 = block_probs[si, s, a]
                p1 = block_probs[si, s, min(a + 1, 2)]
                ll += np.log(max(p0 + p1, 1e-10))
            if ll > best_ki_ll:
                best_ki_ll = ll
                best_ki = ki
        partner_indices.append(best_ki)
    
    new_hap = _extract_missing_hap(
        base_hap_array, block_probs, seed_list, partner_indices)
    new_hap, n_assigned, assigned_set = _iterative_refine(
        base_hap_array, new_hap, block_probs, num_samples)
    return new_hap, n_assigned, assigned_set


def _sample_overlap(set_a, set_b):
    """Jaccard-like overlap: intersection / min(|A|, |B|)."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def _quality_check(block, block_probs, new_hap, num_samples, min_rr):
    """Check whether a discovered haplotype passes quality filters.
    
    Two checks:
      1. Residual reduction ≥ min_rr (adding the hap must meaningfully
         reduce total sample residuals)
      2. prune_chimeras survival (the hap must not be explainable as
         a recombination of existing haps)
    
    Returns:
        (passed, residual_reduction, reason_if_failed)
    """
    base_hap_array, hap_keys = _get_block_hap_array(block)
    n_sites = block_probs.shape[1]
    
    # Residual reduction check
    _, _, base_residuals = _paint_samples_fast(base_hap_array, block_probs, num_samples)
    base_total = np.sum(base_residuals)
    
    expanded_array = np.vstack([base_hap_array, new_hap.reshape(1, -1)])
    _, _, new_residuals = _paint_samples_fast(expanded_array, block_probs, num_samples)
    new_total = np.sum(new_residuals)
    
    rr = base_total - new_total
    
    if rr < min_rr:
        return False, rr, 'low_residual_reduction'
    
    # Structural chimera pruning check
    expanded_hap_dict = {}
    for ki, k in enumerate(hap_keys):
        expanded_hap_dict[ki] = block.haplotypes[k]
    
    new_hap_2d = np.zeros((n_sites, 2), dtype=np.float32)
    new_hap_2d[np.arange(n_sites), new_hap] = 1.0
    expanded_hap_dict[len(hap_keys)] = new_hap_2d
    
    pruned = block_haplotypes.prune_chimeras(
        expanded_hap_dict, block_probs,
        max_recombs=1, max_mismatch_percent=0.5,
        min_mean_delta_to_protect=0.25
    )
    
    survived = len(pruned) > len(hap_keys)
    
    if not survived:
        return False, rr, 'pruned'
    
    return True, rr, None


def _add_hap_to_block(block, new_hap):
    """Add a discovered haplotype to a block, returning a new BlockResult.
    
    The new hap gets the next available integer key. The haplotype is
    stored in probabilistic (n_sites, 2) format to match existing haps.
    """
    n_sites = len(block.positions)
    existing_keys = sorted(block.haplotypes.keys())
    new_key = max(existing_keys) + 1 if existing_keys else 0
    
    new_hap_2d = np.zeros((n_sites, 2), dtype=np.float32)
    new_hap_2d[np.arange(n_sites), new_hap] = 1.0
    
    new_haps = dict(block.haplotypes)
    new_haps[new_key] = new_hap_2d
    
    return block_haplotypes.BlockResult(
        positions=block.positions,
        haplotypes=new_haps,
        keep_flags=block.keep_flags,
        reads_count_matrix=getattr(block, 'reads_count_matrix', None),
        probs_array=block.probs_array,
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def discover_missing_haplotypes(blocks, global_probs, global_sites,
                                 min_residual_reduction=0.10,
                                 overlap_threshold=0.50,
                                 neighbour_range=5,
                                 min_assigned=5,
                                 max_propagation_rounds=5,
                                 num_processes=1,
                                 verbose=True):
    """
    Discover missing founder haplotypes and add them to blocks.
    
    Runs a 5-pass pipeline:
      Pass 1: Residual discovery on all blocks (MAD outlier detection)
      Pass 2: Consistency filter (assigned samples must overlap with neighbours)
      Pass 3: Quality check (residual reduction + chimera pruning)
      Pass 4: Neighbour-seeded recovery (use accepted neighbours' samples as seeds)
      Pass 5: Propagation (iterate seeded recovery until no new discoveries)
    
    Discovered haplotypes are added to their blocks. Existing dedup and
    chimera pruning in the pipeline handle any duplicates or noise.
    
    Args:
        blocks: BlockResults from refinement (or HDBSCAN if no refinement).
        global_probs: (num_samples, num_sites, 3) genotype probability array.
        global_sites: Array of all genomic site positions.
        min_residual_reduction: Minimum total residual decrease to accept a
            discovered haplotype. Default 0.10, validated on chr7/chr23.
        overlap_threshold: Minimum sample overlap with neighbours for
            consistency check. Default 0.50.
        neighbour_range: How many blocks to search for neighbours. Default 5.
        min_assigned: Minimum samples assigned to the new hap. Default 5.
        max_propagation_rounds: Maximum propagation iterations. Default 5.
        num_processes: Number of parallel workers for Pass 1. Default 1
            (sequential). Set to n_cores for full parallelism.
        verbose: Print progress and summary.
    
    Returns:
        BlockResults with discovered haplotypes added to relevant blocks.
    """
    t0 = time.time()
    n_blocks = len(blocks)
    num_samples = global_probs.shape[0]
    site_to_idx = {s: idx for idx, s in enumerate(global_sites)}
    
    if verbose:
        print(f"  Residual discovery: {n_blocks} blocks, {num_samples} samples, "
              f"{num_processes} workers")
    
    # =====================================================================
    # Pass 1: Residual discovery on all blocks (parallel)
    # =====================================================================
    raw_discoveries = {}
    
    if num_processes > 1:
        # Pre-compute per-block task arguments (lightweight: ~3KB each)
        task_args = []
        for bi in range(n_blocks):
            block = blocks[bi]
            hap_array, _ = _get_block_hap_array(block)
            site_indices = np.array([site_to_idx[p] for p in block.positions])
            task_args.append((bi, hap_array, site_indices, min_assigned))
        
        # Create shared memory for global_probs (~2 GB)
        shm = SharedMemory(create=True, size=global_probs.nbytes)
        shm_probs = np.ndarray(global_probs.shape, dtype=global_probs.dtype,
                               buffer=shm.buf)
        shm_probs[:] = global_probs
        
        if verbose:
            shm_mb = global_probs.nbytes / (1024 * 1024)
            print(f"    Shared memory: {shm_mb:.0f} MB")
        
        try:
            with _ForkserverPool(
                    processes=num_processes,
                    initializer=_init_pass1_worker,
                    initargs=(shm.name, global_probs.shape,
                              global_probs.dtype, num_samples)) as pool:
                
                n_done = 0
                for result in pool.imap_unordered(_pass1_worker, task_args,
                                                   chunksize=4):
                    n_done += 1
                    if result is not None:
                        bi, new_hap, n_assigned, assigned_set = result
                        raw_discoveries[bi] = (new_hap, n_assigned, assigned_set)
                    if verbose and n_done % 500 == 0:
                        print(f"    Pass 1: {n_done}/{n_blocks} blocks...")
        finally:
            shm.close()
            shm.unlink()
    else:
        # Sequential fallback
        for bi in range(n_blocks):
            block = blocks[bi]
            site_indices = np.array([site_to_idx[p] for p in block.positions])
            block_probs = global_probs[:, site_indices, :]
            
            new_hap, n_assigned, assigned_set = _residual_discover(
                block, block_probs, num_samples)
            
            if new_hap is not None and n_assigned >= min_assigned:
                raw_discoveries[bi] = (new_hap, n_assigned, assigned_set)
            
            if verbose and (bi + 1) % 500 == 0:
                print(f"    Pass 1: {bi+1}/{n_blocks} blocks...")
    
    if verbose:
        print(f"    Pass 1: {len(raw_discoveries)} raw discoveries")
    
    if not raw_discoveries:
        if verbose:
            print(f"    No discoveries — returning blocks unchanged ({time.time()-t0:.1f}s)")
        return blocks
    
    # =====================================================================
    # Pass 2: Consistency filter
    # =====================================================================
    consistent = set()
    for bi in raw_discoveries:
        _, _, assigned_a = raw_discoveries[bi]
        for dist in range(1, neighbour_range + 1):
            for nbi in [bi - dist, bi + dist]:
                if nbi in raw_discoveries:
                    _, _, assigned_b = raw_discoveries[nbi]
                    if _sample_overlap(assigned_a, assigned_b) >= overlap_threshold:
                        consistent.add(bi)
                        break
            if bi in consistent:
                break
    
    if verbose:
        print(f"    Pass 2: {len(consistent)} consistent (filtered {len(raw_discoveries) - len(consistent)})")
    
    # =====================================================================
    # Pass 3: Quality check (residual reduction + chimera pruning)
    # =====================================================================
    accepted = {}
    reject_counts = {'low_residual_reduction': 0, 'pruned': 0}
    
    for bi in consistent:
        block = blocks[bi]
        site_indices = np.array([site_to_idx[p] for p in block.positions])
        block_probs = global_probs[:, site_indices, :]
        new_hap = raw_discoveries[bi][0]
        
        passed, rr, reason = _quality_check(
            block, block_probs, new_hap, num_samples, min_residual_reduction)
        
        if passed:
            accepted[bi] = {
                'new_hap': new_hap,
                'assigned_samples': raw_discoveries[bi][2],
            }
        elif reason in reject_counts:
            reject_counts[reason] += 1
    
    if verbose:
        print(f"    Pass 3: {len(accepted)} passed quality "
              f"(rr_reject={reject_counts['low_residual_reduction']}, "
              f"pruned={reject_counts['pruned']})")
    
    # =====================================================================
    # Pass 4: Neighbour-seeded recovery
    # =====================================================================
    pass4_count = 0
    for bi in range(n_blocks):
        if bi in accepted:
            continue
        
        block = blocks[bi]
        site_indices = np.array([site_to_idx[p] for p in block.positions])
        block_probs = global_probs[:, site_indices, :]
        
        # Find seed samples from nearest accepted neighbour
        seed_samples = None
        for dist in range(1, neighbour_range + 1):
            for nbi in [bi - dist, bi + dist]:
                if nbi in accepted:
                    seed_samples = accepted[nbi]['assigned_samples']
                    break
            if seed_samples is not None:
                break
        
        if seed_samples is None or len(seed_samples) < min_assigned:
            continue
        
        new_hap, n_assigned, assigned_set = _seeded_discover(
            block, block_probs, seed_samples, num_samples)
        
        if new_hap is None or n_assigned < min_assigned:
            continue
        
        if _sample_overlap(assigned_set, seed_samples) < overlap_threshold:
            continue
        
        passed, rr, reason = _quality_check(
            block, block_probs, new_hap, num_samples, min_residual_reduction)
        if not passed:
            continue
        
        accepted[bi] = {
            'new_hap': new_hap,
            'assigned_samples': assigned_set,
        }
        pass4_count += 1
    
    if verbose:
        print(f"    Pass 4: {pass4_count} neighbour-seeded recoveries")
    
    # =====================================================================
    # Pass 5: Propagation
    # =====================================================================
    total_propagated = 0
    for prop_round in range(max_propagation_rounds):
        new_recoveries = 0
        
        for bi in range(n_blocks):
            if bi in accepted:
                continue
            
            block = blocks[bi]
            site_indices = np.array([site_to_idx[p] for p in block.positions])
            block_probs = global_probs[:, site_indices, :]
            
            seed_samples = None
            for dist in range(1, neighbour_range + 1):
                for nbi in [bi - dist, bi + dist]:
                    if nbi in accepted:
                        seed_samples = accepted[nbi]['assigned_samples']
                        break
                if seed_samples is not None:
                    break
            
            if seed_samples is None or len(seed_samples) < min_assigned:
                continue
            
            new_hap, n_assigned, assigned_set = _seeded_discover(
                block, block_probs, seed_samples, num_samples)
            
            if new_hap is None or n_assigned < min_assigned:
                continue
            
            if _sample_overlap(assigned_set, seed_samples) < overlap_threshold:
                continue
            
            passed, rr, reason = _quality_check(
                block, block_probs, new_hap, num_samples, min_residual_reduction)
            if not passed:
                continue
            
            accepted[bi] = {
                'new_hap': new_hap,
                'assigned_samples': assigned_set,
            }
            new_recoveries += 1
        
        total_propagated += new_recoveries
        if verbose and new_recoveries > 0:
            print(f"    Pass 5 round {prop_round+1}: {new_recoveries} new")
        if new_recoveries == 0:
            break
    
    if verbose:
        print(f"    Pass 5: {total_propagated} propagated total")
    
    # =====================================================================
    # Build output: add discovered haps, then dedup + chimera prune
    # =====================================================================
    if not accepted:
        if verbose:
            print(f"    No discoveries accepted — returning blocks unchanged ({time.time()-t0:.1f}s)")
        return blocks
    
    output_blocks = list(blocks)
    for bi, info in accepted.items():
        output_blocks[bi] = _add_hap_to_block(blocks[bi], info['new_hap'])
    
    if verbose:
        print(f"    Added haplotypes to {len(accepted)} blocks")
    
    # Dedup: merge near-identical haplotypes (catches duplicate FPs at 0% error)
    output_br = block_haplotype_refinement.dedup_blocks(
        block_haplotypes.BlockResults(output_blocks),
        threshold_pct=1.0,
        verbose=verbose
    )
    
    # Chimera prune: remove haps explainable as recombinations of others.
    # Only process blocks that received new haps (the rest are unchanged).
    n_pruned = 0
    output_list = list(output_br)
    for bi in accepted:
        block = output_list[bi]
        if block.probs_array is None or len(block.haplotypes) < 3:
            continue
        pruned_haps = block_haplotypes.prune_chimeras(
            block.haplotypes, block.probs_array,
            max_recombs=1,
            max_mismatch_percent=0.25,
            min_mean_delta_to_protect=0.25
        )
        if len(pruned_haps) < len(block.haplotypes):
            n_pruned += len(block.haplotypes) - len(pruned_haps)
            block.haplotypes = {i: v for i, v in enumerate(pruned_haps.values())}
    
    if verbose:
        print(f"    Chimera prune: removed {n_pruned} haps from modified blocks")
        print(f"    Residual discovery complete ({time.time()-t0:.1f}s)")
    
    return block_haplotypes.BlockResults(output_list)