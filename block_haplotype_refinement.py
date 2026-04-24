"""
haplotype_refinement.py

Iterative refinement of raw block haplotypes via Viterbi painting
and deconvolution. Supports L1 and L2 refinement levels.

Pipeline:
  HDBSCAN raw blocks
    → L1 assembly → Viterbi paint → deconvolve → L1 refined raw blocks
    → dedup → L1 v2 → L2 → Viterbi paint → deconvolve → L2 refined raw blocks
"""
import thread_config

import numpy as np
import math
import time
import warnings
import block_haplotypes


# =============================================================================
# NUMBA JIT KERNELS (parallelize over samples)
# =============================================================================
# paint_viterbi and deconvolve both have hot Python loops over samples/carriers
# that were running single-threaded despite being embarrassingly parallel.
# These numba kernels move those loops into prange-parallel code so they use
# all available cores.  Fallback to pure Python if numba is unavailable --
# matches the style of block_haplotypes.py's numba guard.

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    warnings.warn("Numba not found. Refinement will be extremely slow.",
                  ImportWarning)
    # Dummy decorators so the @njit-decorated defs below still import cleanly
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range


@njit(parallel=True, fastmath=False)
def _paint_viterbi_samples_numba(block_emissions, switch_penalties):
    """Per-sample Viterbi over all samples in parallel.

    Mathematically equivalent to the original pure-Python per-sample loop in
    paint_viterbi.  The only algorithmic change is computing
    max/argmax(scores[:, t-1]) ONCE per t rather than inside the inner-p loop
    (the original recomputes it n_pairs times per t, which is wasteful but
    not incorrect since scores[:, t-1] is not modified during the inner
    loop).  Tie-breaking matches np.argmax's first-occurrence semantics via
    strict `>` comparison.

    Args:
        block_emissions: (num_samples, n_pairs, n_raw_local) float64 --
            per-sample, per-pair, per-raw-block log-emissions.
        switch_penalties: (n_raw_local - 1,) float64 -- distance-based
            switching penalty between consecutive raw blocks.

    Returns:
        sample_paths: (num_samples, n_raw_local) int32 -- best pair-index
            (into the pair_indices list) at each raw-block position per sample.
    """
    num_samples, n_pairs, n_raw_local = block_emissions.shape
    sample_paths = np.zeros((num_samples, n_raw_local), dtype=np.int32)

    for s in prange(num_samples):
        # Thread-local working arrays (allocated per-sample inside prange)
        scores = np.full((n_pairs, n_raw_local), -np.inf)
        backptr = np.zeros((n_pairs, n_raw_local), dtype=np.int32)

        # Initialize t=0
        for p in range(n_pairs):
            scores[p, 0] = block_emissions[s, p, 0]

        # Forward pass
        for t in range(1, n_raw_local):
            pen_t = switch_penalties[t - 1]

            # Compute max and first-argmax of scores[:, t-1] once per t
            # (scores[:, t-1] is not modified during the inner p loop)
            best_prev = -np.inf
            best_prev_idx = 0
            for p in range(n_pairs):
                if scores[p, t - 1] > best_prev:
                    best_prev = scores[p, t - 1]
                    best_prev_idx = p

            for p in range(n_pairs):
                stay = scores[p, t - 1] + block_emissions[s, p, t]
                best_switch = best_prev - pen_t + block_emissions[s, p, t]
                if stay >= best_switch:
                    scores[p, t] = stay
                    backptr[p, t] = p
                else:
                    scores[p, t] = best_switch
                    backptr[p, t] = best_prev_idx

        # Traceback (first-argmax over final column)
        best_final = -np.inf
        best_final_idx = 0
        for p in range(n_pairs):
            if scores[p, n_raw_local - 1] > best_final:
                best_final = scores[p, n_raw_local - 1]
                best_final_idx = p
        sample_paths[s, n_raw_local - 1] = best_final_idx

        for t in range(n_raw_local - 1, 0, -1):
            sample_paths[s, t - 1] = backptr[sample_paths[s, t], t]

    return sample_paths


@njit(parallel=True, fastmath=False)
def _deconvolve_one_block_numba(sample_probs_raw, haps_raw_alleles,
                                 sample_pair_a, sample_pair_b):
    """Deconvolve a single raw block -- parallelized over haps.

    Mathematically equivalent to the original per-hap/per-carrier Python
    loop in deconvolve.  The original builds a `carriers[ki]` list per hap
    then iterates.  Here we loop over all samples for each hap (parallel
    across haps via prange) and conditionally accumulate; since carriers
    are determined by sample_pair_a/sample_pair_b equality with ki, the
    effective set of accumulated samples is identical.

    The 1e-300 floor -> math.log pattern preserves the NaN-fix behaviour of
    casting to float64 before clamping (sample_probs_raw is passed in as
    float64 from the caller, which handles the cast once rather than
    per-site as the original did).

    Args:
        sample_probs_raw: (num_samples, n_raw_sites, 3) float64 -- genotype
            probabilities for this raw block's sites only.  Caller MUST pass
            this already cast to float64 (critical for the 1e-300 floor to
            not underflow as it would in float32).
        haps_raw_alleles: (K, n_raw_sites) int8 -- the super-block's concrete
            allele (0 or 1) at each raw-block site, indexed by hap.
        sample_pair_a: (num_samples,) int32 -- hap index "a" of painted pair
            per sample at this raw block.
        sample_pair_b: (num_samples,) int32 -- hap index "b" of painted pair
            per sample.  For homozygous paintings, a == b.

    Returns:
        new_haps: (K, n_raw_sites, 2) float64 -- posterior (P[allele=0],
            P[allele=1]) per hap per site after carrier-based deconvolution.
            Normalizes to sum 1.0 along axis=2.
    """
    num_samples = sample_probs_raw.shape[0]
    n_raw_sites = sample_probs_raw.shape[1]
    K = haps_raw_alleles.shape[0]

    new_haps = np.zeros((K, n_raw_sites, 2), dtype=np.float64)

    # Count carriers per hap (matches original: each sample contributes to
    # one hap if homozygous, two haps if heterozygous)
    carrier_count = np.zeros(K, dtype=np.int64)
    for s in range(num_samples):
        a = sample_pair_a[s]
        b = sample_pair_b[s]
        if a == b:
            carrier_count[a] += 1
        else:
            carrier_count[a] += 1
            carrier_count[b] += 1

    # Parallelize over haps: each hap's log_p0/log_p1 accumulation is
    # independent of other haps', so no race conditions.
    for ki in prange(K):
        if carrier_count[ki] == 0:
            # No carriers: use super-block alleles as fallback (matches original)
            for j in range(n_raw_sites):
                allele = haps_raw_alleles[ki, j]
                new_haps[ki, j, 0] = 1.0 - allele
                new_haps[ki, j, 1] = float(allele)
            continue

        # Thread-local log-prob accumulators for this hap
        log_p0 = np.zeros(n_raw_sites, dtype=np.float64)
        log_p1 = np.zeros(n_raw_sites, dtype=np.float64)

        # Iterate all samples; accumulate if sample is a carrier of hap ki.
        # This mirrors the original carrier-list-iterating behavior: the
        # selection condition replaces the carriers[ki] membership test.
        for s in range(num_samples):
            a = sample_pair_a[s]
            b = sample_pair_b[s]

            if a == b:
                if a == ki:
                    # Homozygous carrier of hap ki -- use p[0] and p[2]
                    # (matches original `is_hom=True` branch)
                    for j in range(n_raw_sites):
                        p0_val = sample_probs_raw[s, j, 0]
                        if p0_val < 1e-300:
                            p0_val = 1e-300
                        p2_val = sample_probs_raw[s, j, 2]
                        if p2_val < 1e-300:
                            p2_val = 1e-300
                        log_p0[j] += math.log(p0_val)
                        log_p1[j] += math.log(p2_val)
                # else: this sample is homozygous for a different hap, skip
            else:
                # Heterozygous: sample s carries both haps a and b
                if a == ki:
                    # Partner hap is b -- use haps_raw_alleles[b, :] as template
                    # (matches original `carriers[a].append((s, b, False))`)
                    for j in range(n_raw_sites):
                        pa = haps_raw_alleles[b, j]
                        p0_val = sample_probs_raw[s, j, pa]
                        if p0_val < 1e-300:
                            p0_val = 1e-300
                        p1_val = sample_probs_raw[s, j, pa + 1]
                        if p1_val < 1e-300:
                            p1_val = 1e-300
                        log_p0[j] += math.log(p0_val)
                        log_p1[j] += math.log(p1_val)
                elif b == ki:
                    # Partner hap is a -- use haps_raw_alleles[a, :] as template
                    # (matches original `carriers[b].append((s, a, False))`)
                    for j in range(n_raw_sites):
                        pa = haps_raw_alleles[a, j]
                        p0_val = sample_probs_raw[s, j, pa]
                        if p0_val < 1e-300:
                            p0_val = 1e-300
                        p1_val = sample_probs_raw[s, j, pa + 1]
                        if p1_val < 1e-300:
                            p1_val = 1e-300
                        log_p0[j] += math.log(p0_val)
                        log_p1[j] += math.log(p1_val)
                # else: neither a nor b equals ki, skip

        # Normalize to posterior probabilities for this hap
        # (log-sum-exp with max-subtraction for numerical stability)
        for j in range(n_raw_sites):
            lp0 = log_p0[j]
            lp1 = log_p1[j]
            if lp0 > lp1:
                ml = lp0
            else:
                ml = lp1
            p0 = math.exp(lp0 - ml)
            p1 = math.exp(lp1 - ml)
            total = p0 + p1
            new_haps[ki, j, 0] = p0 / total
            new_haps[ki, j, 1] = p1 / total

    return new_haps


@njit(parallel=True, fastmath=False)
def _deconvolve_super_block_numba(sample_probs_sb_f64, haps_sb,
                                   paintings_arr, raw_block_starts):
    """Deconvolve ALL raw blocks of a super-block in a single kernel call.

    Supersedes _deconvolve_one_block_numba for the common case where we
    have a full super-block's paintings in hand.  The math per (raw_block,
    hap) pair is identical to _deconvolve_one_block_numba -- the only
    difference is parallelism topology:
      - _deconvolve_one_block_numba: prange over K haps (K~7 parallel units)
      - _deconvolve_super_block_numba: prange over (n_raw_local * K) pairs
        (70-700 parallel units), filling 112 cores much more completely.

    Also hoists the sample_probs_sb float64 cast out of the per-raw-block
    loop -- caller does it once per super-block.

    Args:
        sample_probs_sb_f64: (num_samples, n_sb_sites, 3) float64 -- super-
            block-level genotype probabilities, already cast to float64 by
            the caller (critical for the 1e-300 floor not to underflow).
        haps_sb: (K, n_sb_sites) int8 -- super-block's concrete alleles at
            every super-block site.
        paintings_arr: (num_samples, n_raw_local, 2) int32 -- per-sample
            painted (hap_a, hap_b) pair at each raw-block position.  Comes
            directly from paint_viterbi's new array output.
        raw_block_starts: (n_raw_local + 1,) int32 -- cumulative site offsets.
            raw_block_starts[ri] is the first super-block-site index of raw
            block ri; raw_block_starts[n_raw_local] == n_sb_sites.  This
            encoding handles variable raw-block sizes naturally.

    Returns:
        new_haps_flat: (K, n_sb_sites, 2) float64 -- concatenated posterior
            for every (hap, super-block-site) pair.  Caller slices by
            raw_block_starts to recover per-raw-block (K, n_raw_sites, 2).
    """
    num_samples = sample_probs_sb_f64.shape[0]
    n_sb_sites = sample_probs_sb_f64.shape[1]
    K = haps_sb.shape[0]
    n_raw_local = raw_block_starts.shape[0] - 1

    new_haps_flat = np.zeros((K, n_sb_sites, 2), dtype=np.float64)

    # Parallelize over (ri_local, ki) pairs -- flat index idx = ri_local * K + ki
    # This gives n_raw_local * K parallel units, much more than the K-only
    # parallelism of the per-raw-block kernel.
    total_units = n_raw_local * K
    for idx in prange(total_units):
        ri_local = idx // K
        ki = idx % K

        site_start = raw_block_starts[ri_local]
        site_end = raw_block_starts[ri_local + 1]
        n_raw_sites = site_end - site_start

        # Count carriers of ki at this raw block
        carrier_count = 0
        for s in range(num_samples):
            a = paintings_arr[s, ri_local, 0]
            b = paintings_arr[s, ri_local, 1]
            if a == ki or b == ki:
                carrier_count += 1

        if carrier_count == 0:
            # No carriers: fallback to super-block's allele (matches original)
            for j in range(n_raw_sites):
                allele = haps_sb[ki, site_start + j]
                new_haps_flat[ki, site_start + j, 0] = 1.0 - allele
                new_haps_flat[ki, site_start + j, 1] = float(allele)
            continue

        # Thread-local log-prob accumulators for this (ri_local, ki) pair
        log_p0 = np.zeros(n_raw_sites, dtype=np.float64)
        log_p1 = np.zeros(n_raw_sites, dtype=np.float64)

        # Accumulate contributions from all carriers.  Same three branches
        # as _deconvolve_one_block_numba: homozygous / het-as-a / het-as-b.
        for s in range(num_samples):
            a = paintings_arr[s, ri_local, 0]
            b = paintings_arr[s, ri_local, 1]

            if a == b:
                if a == ki:
                    # Homozygous carrier of hap ki -- p[0] and p[2]
                    for j in range(n_raw_sites):
                        p0_val = sample_probs_sb_f64[s, site_start + j, 0]
                        if p0_val < 1e-300:
                            p0_val = 1e-300
                        p2_val = sample_probs_sb_f64[s, site_start + j, 2]
                        if p2_val < 1e-300:
                            p2_val = 1e-300
                        log_p0[j] += math.log(p0_val)
                        log_p1[j] += math.log(p2_val)
                # else: homozygous for a different hap, skip
            else:
                if a == ki:
                    # Partner hap is b -- template against haps_sb[b, ...]
                    for j in range(n_raw_sites):
                        pa = haps_sb[b, site_start + j]
                        p0_val = sample_probs_sb_f64[s, site_start + j, pa]
                        if p0_val < 1e-300:
                            p0_val = 1e-300
                        p1_val = sample_probs_sb_f64[s, site_start + j, pa + 1]
                        if p1_val < 1e-300:
                            p1_val = 1e-300
                        log_p0[j] += math.log(p0_val)
                        log_p1[j] += math.log(p1_val)
                elif b == ki:
                    # Partner hap is a -- template against haps_sb[a, ...]
                    for j in range(n_raw_sites):
                        pa = haps_sb[a, site_start + j]
                        p0_val = sample_probs_sb_f64[s, site_start + j, pa]
                        if p0_val < 1e-300:
                            p0_val = 1e-300
                        p1_val = sample_probs_sb_f64[s, site_start + j, pa + 1]
                        if p1_val < 1e-300:
                            p1_val = 1e-300
                        log_p0[j] += math.log(p0_val)
                        log_p1[j] += math.log(p1_val)
                # else: neither a nor b is ki, skip

        # Normalize to posterior probabilities via log-sum-exp
        for j in range(n_raw_sites):
            lp0 = log_p0[j]
            lp1 = log_p1[j]
            if lp0 > lp1:
                ml = lp0
            else:
                ml = lp1
            p0 = math.exp(lp0 - ml)
            p1 = math.exp(lp1 - ml)
            total = p0 + p1
            new_haps_flat[ki, site_start + j, 0] = p0 / total
            new_haps_flat[ki, site_start + j, 1] = p1 / total

    return new_haps_flat


# =============================================================================
# VITERBI PAINTING
# =============================================================================

def paint_viterbi(super_blocks, raw_blocks, raw_per_super,
                  global_probs, global_sites, num_samples,
                  penalty_scale=20.0, recomb_rate=5e-8, n_generations=3):
    """
    Paint samples with Viterbi switching at raw block boundaries.
    
    For each super-block, compute per-raw-block emissions for all diplotype
    pairs, then run Viterbi per sample allowing pair-switches with a
    recombination-based penalty.
    
    Args:
        super_blocks: BlockResults from L1 or L2 assembly.
        raw_blocks: Original 200-SNP raw blocks.
        raw_per_super: Number of raw blocks per super-block
            (BATCH_SIZE for L1, BATCH_SIZE^2 for L2).
        global_probs: (n_samples, n_sites, 3) genotype probability array.
        global_sites: Array of all genomic site positions.
        num_samples: Number of samples.
        penalty_scale: Multiplier on the base switching penalty.
            Higher = fewer switches. Default 20.0.
        recomb_rate: Per-bp recombination rate.
        n_generations: Average meioses between founders and samples.
    
    Returns:
        List of dicts, one per super-block:
            {'raw_start': int, 'raw_end': int,
             'paintings': list[list[tuple(int,int)]]}
        paintings[sample][raw_block_local] = (hap_a, hap_b)
    """
    n_raw = len(raw_blocks)
    all_paintings = []
    
    for sb_idx, sb in enumerate(super_blocks):
        hap_keys = sorted(sb.haplotypes.keys())
        K = len(hap_keys)
        sb_positions = sb.positions
        n_sb_sites = len(sb_positions)
        
        # Concretify haplotypes
        haps = np.zeros((K, n_sb_sites), dtype=np.int8)
        for ki, k in enumerate(hap_keys):
            h = sb.haplotypes[k]
            haps[ki] = np.argmax(h, axis=1) if h.ndim > 1 else h
        
        # Sample probabilities at super-block sites
        indices = np.searchsorted(global_sites, sb_positions)
        sample_probs = global_probs[:, indices, :]
        
        # Build pair list (upper triangle including diagonal)
        pair_indices = []
        for a in range(K):
            for b in range(a, K):
                pair_indices.append((a, b))
        n_pairs = len(pair_indices)
        
        # Pair genotypes
        pair_genos = np.zeros((n_pairs, n_sb_sites), dtype=np.int8)
        for pi, (a, b) in enumerate(pair_indices):
            pair_genos[pi] = haps[a] + haps[b]
        
        # Raw block boundaries within this super-block
        raw_start_idx = sb_idx * raw_per_super
        raw_end_idx = min(raw_start_idx + raw_per_super, n_raw)
        n_raw_local = raw_end_idx - raw_start_idx
        
        raw_block_boundaries = []
        site_offset = 0
        for ri in range(raw_start_idx, raw_end_idx):
            n_sites_ri = len(raw_blocks[ri].positions)
            raw_block_boundaries.append((site_offset, site_offset + n_sites_ri))
            site_offset += n_sites_ri
        
        # Per-raw-block emissions: (n_samples, n_pairs, n_raw_local)
        block_emissions = np.zeros((num_samples, n_pairs, n_raw_local),
                                   dtype=np.float64)
        for ri_local, (s_start, s_end) in enumerate(raw_block_boundaries):
            for pi in range(n_pairs):
                geno = pair_genos[pi, s_start:s_end]
                site_idx = np.arange(s_start, s_end)
                probs = sample_probs[:, site_idx, geno]
                # Cast to float64 before the 1e-300 floor: global_probs is
                # float32 in the pipeline (downcast in hierarchical_assembly),
                # and 1e-300 underflows to 0.0 in float32, defeating the floor
                # and producing -inf log values that cascade to NaN downstream.
                block_emissions[:, pi, ri_local] = np.sum(
                    np.log(np.maximum(probs.astype(np.float64), 1e-300)), axis=1)
        
        # Switching penalties based on physical distance
        switch_penalties = np.zeros(n_raw_local - 1)
        for ri_local in range(n_raw_local - 1):
            ri_abs = raw_start_idx + ri_local
            if ri_abs + 1 < n_raw:
                pos_a = raw_blocks[ri_abs].positions
                pos_b = raw_blocks[ri_abs + 1].positions
                dist = pos_b[len(pos_b) // 2] - pos_a[len(pos_a) // 2]
                expected_recombs = dist * recomb_rate * n_generations
                if expected_recombs > 0:
                    base_pen = min(20.0, -np.log(max(expected_recombs, 1e-10)))
                else:
                    base_pen = 20.0
                switch_penalties[ri_local] = base_pen * penalty_scale
        
        # Viterbi per sample -- parallelized via numba prange over samples.
        # Mathematically equivalent to the original per-sample loop (see the
        # _paint_viterbi_samples_numba kernel docstring for the one
        # optimization: computing max/argmax of scores[:, t-1] once per t
        # rather than recomputing inside the p loop, which is equivalent
        # because scores[:, t-1] isn't modified during the inner p loop).
        #
        # Output format: (num_samples, n_raw_local, 2) int32 array.  This
        # replaces the earlier list-of-lists-of-tuples format (which forced
        # a 320*n_raw_local Python tuple-unpack loop in deconvolve per raw
        # block).  All internal consumers -- deconvolve and refine_at_level's
        # switch counter -- use the array directly.  pair_indices_arr is
        # built once here so the fancy-index conversion is a single numpy op.
        pair_indices_arr = np.array(pair_indices, dtype=np.int32)  # (n_pairs, 2)
        if _HAS_NUMBA:
            sample_paths_int = _paint_viterbi_samples_numba(
                block_emissions, switch_penalties)
            # Fancy index: pair_indices_arr[sample_paths_int] gives
            # (num_samples, n_raw_local, 2) -- one numpy op, no Python loop.
            sample_paintings = pair_indices_arr[sample_paths_int]
        else:
            # Pure-Python fallback (original code path, with array output)
            sample_paintings = np.empty((num_samples, n_raw_local, 2),
                                        dtype=np.int32)
            for s in range(num_samples):
                scores = np.full((n_pairs, n_raw_local), -np.inf)
                backptr = np.zeros((n_pairs, n_raw_local), dtype=np.int32)
                
                scores[:, 0] = block_emissions[s, :, 0]
                
                for t in range(1, n_raw_local):
                    pen_t = switch_penalties[t - 1]
                    for p in range(n_pairs):
                        stay = scores[p, t - 1] + block_emissions[s, p, t]
                        best_switch = (np.max(scores[:, t - 1]) - pen_t
                                       + block_emissions[s, p, t])
                        if stay >= best_switch:
                            scores[p, t] = stay
                            backptr[p, t] = p
                        else:
                            scores[p, t] = best_switch
                            backptr[p, t] = np.argmax(scores[:, t - 1])
                
                # Traceback
                path = np.zeros(n_raw_local, dtype=np.int32)
                path[-1] = np.argmax(scores[:, -1])
                for t in range(n_raw_local - 1, 0, -1):
                    path[t - 1] = backptr[path[t], t]
                
                # Write into the (num_samples, n_raw_local, 2) array rather
                # than appending tuples to a list (array is the new format)
                for t in range(n_raw_local):
                    a, b = pair_indices[path[t]]
                    sample_paintings[s, t, 0] = a
                    sample_paintings[s, t, 1] = b
        
        all_paintings.append({
            'raw_start': raw_start_idx,
            'raw_end': raw_end_idx,
            'paintings': sample_paintings
        })
    
    return all_paintings


# =============================================================================
# DECONVOLUTION
# =============================================================================

def deconvolve(super_blocks, viterbi_paintings, raw_blocks,
               global_probs, global_sites, num_samples):
    """
    Deconvolve raw blocks using per-raw-block Viterbi painting.
    
    For each haplotype in the super-block, collect all carrier samples
    (from the painting), then re-estimate the allele at each site using
    the carrier's read data and the partner haplotype as a template.
    
    Args:
        super_blocks: BlockResults from L1 or L2 assembly.
        viterbi_paintings: Output of paint_viterbi.
        raw_blocks: Original 200-SNP raw blocks.
        global_probs: (n_samples, n_sites, 3) genotype probability array.
        global_sites: Array of all genomic site positions.
        num_samples: Number of samples.
    
    Returns:
        BlockResults with refined raw blocks (same count as raw_blocks).
    """
    n_raw = len(raw_blocks)
    refined_blocks = [None] * n_raw
    
    for paint_data in viterbi_paintings:
        raw_start = paint_data['raw_start']
        raw_end = paint_data['raw_end']
        sample_paintings = paint_data['paintings']
        
        # Find corresponding super-block by position match
        sb_idx = None
        for li, sb in enumerate(super_blocks):
            if len(sb.positions) > 0:
                if sb.positions[0] == raw_blocks[raw_start].positions[0]:
                    sb_idx = li
                    break
        if sb_idx is None:
            for li, pd in enumerate(viterbi_paintings):
                if pd['raw_start'] == raw_start:
                    sb_idx = li
                    break
        
        sb = super_blocks[sb_idx]
        hap_keys = sorted(sb.haplotypes.keys())
        K = len(hap_keys)
        sb_positions = sb.positions
        n_sb_sites = len(sb_positions)
        
        # Concretify super-block haplotypes
        haps_sb = np.zeros((K, n_sb_sites), dtype=np.int8)
        for ki, k in enumerate(hap_keys):
            h = sb.haplotypes[k]
            haps_sb[ki] = np.argmax(h, axis=1) if h.ndim > 1 else h
        
        # Sample probabilities at super-block sites
        indices = np.searchsorted(global_sites, sb_positions)
        sample_probs_sb = global_probs[:, indices, :]
        
        # Build raw_block_starts array -- cumulative site offsets for each
        # raw block in this super-block.  raw_block_starts[ri_local] is the
        # first super-block-site index of raw block ri_local; the sentinel
        # raw_block_starts[n_raw_local] == total_sb_sites_in_paint_data.
        # This encoding handles variable raw-block sizes naturally (raw
        # blocks at chromosome ends may be shorter than the standard 200
        # SNPs) and is shared by both the numba and fallback deconvolve
        # paths so per-raw-block BlockResult construction can slice it
        # cheaply at the end.
        n_raw_local = raw_end - raw_start
        raw_block_starts = np.zeros(n_raw_local + 1, dtype=np.int32)
        _offset = 0
        for _ri in range(raw_start, raw_end):
            raw_block_starts[_ri - raw_start] = _offset
            _offset += len(raw_blocks[_ri].positions)
        raw_block_starts[n_raw_local] = _offset
        n_sb_sites_total = int(raw_block_starts[n_raw_local])

        # Deconvolve all raw blocks in this super-block -- parallelized via
        # numba prange over (raw_block, hap) pairs in a single kernel call.
        # Previously deconvolve made n_raw_local separate kernel calls (one
        # per raw block), each with only K-way (~7) parallelism; now it
        # makes one call with n_raw_local * K (~70 for L1, ~700 for L2)
        # parallelism, filling the 112-core pool much more completely.
        # Preserves the NaN-fix by casting sample_probs_sb to float64 ONCE
        # per super-block (previously done n_raw_local times per super-block).
        if _HAS_NUMBA:
            sample_probs_sb_f64 = sample_probs_sb.astype(np.float64)
            new_haps_flat = _deconvolve_super_block_numba(
                sample_probs_sb_f64, haps_sb,
                sample_paintings, raw_block_starts)
        else:
            # Pure-Python fallback (original code path, per-raw-block).
            # Writes into the same new_haps_flat buffer the numba branch
            # produces, so the downstream BlockResult construction below is
            # shared between both branches.
            new_haps_flat = np.zeros((K, n_sb_sites_total, 2), dtype=np.float64)
            for ri in range(raw_start, raw_end):
                ri_local = ri - raw_start
                site_start = int(raw_block_starts[ri_local])
                site_end = int(raw_block_starts[ri_local + 1])
                n_raw_sites = site_end - site_start
                site_slice = slice(site_start, site_end)
                site_idx_local = np.arange(site_start, site_end)
                
                # Build per-haplotype carrier list for this raw block
                carriers = {ki: [] for ki in range(K)}
                for s in range(num_samples):
                    # sample_paintings is now an int32 array of shape
                    # (num_samples, n_raw_local, 2); explicit int() casts
                    # ensure numpy scalar dict keys interoperate cleanly
                    # with the Python-int ki keys in the carriers dict.
                    a = int(sample_paintings[s, ri_local, 0])
                    b = int(sample_paintings[s, ri_local, 1])
                    if a == b:
                        carriers[a].append((s, a, True))
                    else:
                        carriers[a].append((s, b, False))
                        carriers[b].append((s, a, False))
                
                # Deconvolve each haplotype
                new_haps_block = np.zeros((K, n_raw_sites, 2), dtype=np.float64)
                
                for ki in range(K):
                    if not carriers[ki]:
                        # No carriers: use super-block alleles as fallback
                        alleles = haps_sb[ki, site_slice]
                        new_haps_block[ki, :, 0] = 1.0 - alleles
                        new_haps_block[ki, :, 1] = alleles.astype(np.float64)
                        continue
                    
                    log_p0 = np.zeros(n_raw_sites, dtype=np.float64)
                    log_p1 = np.zeros(n_raw_sites, dtype=np.float64)
                    
                    for s, partner_ki, is_hom in carriers[ki]:
                        # Cast to float64 before the 1e-300 floor: global_probs
                        # (and thus sample_probs_sb) is float32 in the pipeline,
                        # and 1e-300 underflows to 0.0 in float32.  Without the
                        # cast, sites where a sample's genotype prob rounds to
                        # float32-zero produce -inf log values, which accumulate
                        # to -inf log_p0/log_p1 and cascade to NaN downstream via
                        # max_log = -inf → log_p - max_log = -inf - -inf = NaN.
                        if is_hom:
                            log_p0 += np.log(np.maximum(
                                sample_probs_sb[s, site_idx_local, 0].astype(np.float64),
                                1e-300))
                            log_p1 += np.log(np.maximum(
                                sample_probs_sb[s, site_idx_local, 2].astype(np.float64),
                                1e-300))
                        else:
                            pa = haps_sb[partner_ki, site_slice]
                            log_p0 += np.log(np.maximum(
                                sample_probs_sb[s, site_idx_local, pa].astype(np.float64),
                                1e-300))
                            log_p1 += np.log(np.maximum(
                                sample_probs_sb[s, site_idx_local, pa + 1].astype(np.float64),
                                1e-300))
                    
                    max_log = np.maximum(log_p0, log_p1)
                    p0 = np.exp(log_p0 - max_log)
                    p1 = np.exp(log_p1 - max_log)
                    total = p0 + p1
                    new_haps_block[ki, :, 0] = p0 / total
                    new_haps_block[ki, :, 1] = p1 / total
                
                # Write this raw block's result into the flat output so the
                # downstream BlockResult construction below handles it the
                # same way as the numba branch.
                new_haps_flat[:, site_start:site_end, :] = new_haps_block
        
        # Split new_haps_flat into per-raw-block BlockResults (common path
        # for both numba and fallback branches).  .copy() ensures each
        # BlockResult owns an independent (K, n_raw_sites, 2) array rather
        # than a view into new_haps_flat (which would keep the big array
        # alive and share storage between unrelated raw blocks).
        for ri in range(raw_start, raw_end):
            ri_local = ri - raw_start
            raw_block = raw_blocks[ri]
            site_start = int(raw_block_starts[ri_local])
            site_end = int(raw_block_starts[ri_local + 1])
            
            block_new_haps = new_haps_flat[:, site_start:site_end, :]
            refined_haps = {ki: block_new_haps[ki].copy() for ki in range(K)}
            refined_block = block_haplotypes.BlockResult(
                positions=raw_block.positions.copy(),
                haplotypes=refined_haps,
                keep_flags=raw_block.keep_flags,
                probs_array=raw_block.probs_array
            )
            refined_blocks[ri] = refined_block
    
    # Fill any gaps (safety net for rounding in last super-block)
    for ri in range(n_raw):
        if refined_blocks[ri] is None:
            refined_blocks[ri] = raw_blocks[ri]
    
    return block_haplotypes.BlockResults(refined_blocks)


# =============================================================================
# DEDUPLICATION
# =============================================================================

def dedup_blocks(blocks, threshold_pct=1.0, verbose=True):
    """
    Merge near-identical haplotypes within each block.
    
    Prevents degenerate EM meshes in subsequent assembly steps by
    removing haplotypes that differ by less than threshold_pct.
    When merging, keeps the haplotype with higher sharpness
    (mean max probability across sites).
    
    Args:
        blocks: BlockResults to deduplicate.
        threshold_pct: Merge haplotypes differing by less than this %.
        verbose: If True, print summary.
    
    Returns:
        Deduplicated BlockResults.
    """
    deduped = []
    total_before = 0
    total_after = 0
    n_collapsed = 0
    
    for block in blocks:
        hap_keys = sorted(block.haplotypes.keys())
        K = len(hap_keys)
        total_before += K
        
        if K <= 1:
            deduped.append(block)
            total_after += K
            continue
        
        # Concretify and compute sharpness
        concrete = {}
        sharpness = {}
        for k in hap_keys:
            h = block.haplotypes[k]
            if h.ndim > 1:
                concrete[k] = np.argmax(h, axis=1)
                sharpness[k] = np.mean(np.max(h, axis=1))
            else:
                concrete[k] = np.array(h)
                sharpness[k] = 1.0
        
        # Greedy dedup
        kept = [hap_keys[0]]
        for k in hap_keys[1:]:
            is_dup = False
            for kk in kept:
                if np.mean(concrete[k] != concrete[kk]) * 100 < threshold_pct:
                    is_dup = True
                    if sharpness[k] > sharpness[kk]:
                        kept[kept.index(kk)] = k
                    break
            if not is_dup:
                kept.append(k)
        
        if len(kept) < K:
            n_collapsed += 1
        
        new_haps = {new_idx: block.haplotypes[old_key]
                    for new_idx, old_key in enumerate(kept)}
        new_block = block_haplotypes.BlockResult(
            positions=block.positions.copy(),
            haplotypes=new_haps,
            keep_flags=block.keep_flags,
            probs_array=block.probs_array
        )
        deduped.append(new_block)
        total_after += len(kept)
    
    if verbose:
        print(f"  Dedup: {total_before} -> {total_after} total haps "
              f"({n_collapsed} blocks had merges)")
    
    return block_haplotypes.BlockResults(deduped)


# =============================================================================
# HIGH-LEVEL REFINEMENT
# =============================================================================

def refine_at_level(assembly_blocks, raw_blocks,
                    global_probs, global_sites, num_samples,
                    raw_per_super,
                    penalty_scale=20.0,
                    recomb_rate=5e-8, n_generations=3,
                    verbose=True):
    """
    Run one round of refinement: Viterbi paint → deconvolve.
    
    Args:
        assembly_blocks: Super-blocks from L1 or L2 assembly.
        raw_blocks: Original 200-SNP raw blocks.
        global_probs: (n_samples, n_sites, 3) array.
        global_sites: Site position array.
        num_samples: Number of samples.
        raw_per_super: Raw blocks per super-block
            (batch_size for L1, batch_size^2 for L2).
        penalty_scale: Viterbi switching penalty scale.
        recomb_rate: Per-bp recombination rate.
        n_generations: Average meioses.
        verbose: If True, print progress and switching stats.
    
    Returns:
        BlockResults with refined raw blocks.
    """
    t0 = time.time()
    if verbose:
        print(f"  Viterbi painting (penalty_scale={penalty_scale})...")
    
    paintings = paint_viterbi(
        assembly_blocks, raw_blocks, raw_per_super,
        global_probs, global_sites, num_samples,
        penalty_scale=penalty_scale,
        recomb_rate=recomb_rate,
        n_generations=n_generations
    )
    
    if verbose:
        # Count painting switches (consecutive-position pair changes) for
        # the progress stat.  paintings now stores arrays of shape
        # (num_samples, n_raw_local, 2) int32 rather than list-of-lists of
        # tuples, so this reduces to a vectorized numpy comparison.  A
        # switch at position t for sample s is any change in the (a, b)
        # pair relative to position t-1 -- np.any over axis=2 captures
        # "either hap_a or hap_b changed", matching the original
        # `if path[t] != path[t - 1]` tuple-inequality semantics.
        total_switches = 0
        total_positions = 0
        for pd in paintings:
            paintings_arr = pd['paintings']  # (num_samples, n_raw_local, 2) int32
            if paintings_arr.shape[1] > 1:
                diffs = np.any(
                    paintings_arr[:, 1:, :] != paintings_arr[:, :-1, :],
                    axis=2)  # (num_samples, n_raw_local - 1) bool
                total_switches += int(np.sum(diffs))
                total_positions += (paintings_arr.shape[1] - 1) * paintings_arr.shape[0]
        pct = 100 * total_switches / max(total_positions, 1)
        print(f"  Switches: {total_switches}/{total_positions} ({pct:.2f}%)")
        print(f"  Painting done in {time.time() - t0:.1f}s")
    
    t0 = time.time()
    if verbose:
        print(f"  Deconvolving...")
    
    refined = deconvolve(
        assembly_blocks, paintings, raw_blocks,
        global_probs, global_sites, num_samples
    )
    
    if verbose:
        print(f"  Deconvolution done in {time.time() - t0:.1f}s")
    
    return refined


def run_refinement_pipeline(raw_blocks,
                            global_probs, global_sites, num_samples,
                            run_l1_assembly_fn,
                            run_l2_assembly_fn,
                            batch_size=10,
                            penalty_scale=20.0,
                            recomb_rate=5e-8, n_generations=3,
                            verbose=True):
    """
    Run the full L1 + L2 refinement pipeline.
    
    Steps:
      1. L1 assembly on raw blocks
      2. L1 refinement (Viterbi paint + deconvolve)
      3. Dedup
      4. L1 v2 assembly on refined blocks
      5. L2 assembly
      6. L2 refinement (Viterbi paint + deconvolve)
    
    Args:
        raw_blocks: HDBSCAN-discovered raw blocks.
        global_probs: (n_samples, n_sites, 3) array.
        global_sites: Site position array.
        num_samples: Number of samples.
        run_l1_assembly_fn: Callable that takes BlockResults and returns
            L1 super-blocks. Should handle its own parallelization.
        run_l2_assembly_fn: Callable that takes BlockResults and returns
            L2 super-blocks (with HMM linking).
        batch_size: Number of blocks per batch (default 10).
        penalty_scale: Viterbi switching penalty scale.
        recomb_rate: Per-bp recombination rate.
        n_generations: Average meioses.
        verbose: If True, print progress.
    
    Returns:
        dict with keys:
            'l1_refined': L1-refined raw blocks
            'l2_refined': L2-refined raw blocks
            'l1_assembly': L1 super-blocks (from refined input)
            'l2_assembly': L2 super-blocks (from refined input)
    """
    results = {}
    
    # ── Step 1: L1 Assembly ──
    if verbose:
        print(f"\n  Step 1: L1 assembly on raw blocks...")
    t0 = time.time()
    l1_blocks = run_l1_assembly_fn(raw_blocks)
    if verbose:
        print(f"  L1 assembly done in {time.time() - t0:.0f}s")
    
    # ── Step 2: L1 Refinement ──
    if verbose:
        print(f"\n  Step 2: L1 refinement...")
    l1_refined = refine_at_level(
        l1_blocks, raw_blocks,
        global_probs, global_sites, num_samples,
        raw_per_super=batch_size,
        penalty_scale=penalty_scale,
        recomb_rate=recomb_rate,
        n_generations=n_generations,
        verbose=verbose
    )
    results['l1_refined'] = l1_refined
    
    # ── Step 3: Dedup ──
    if verbose:
        print(f"\n  Step 3: Dedup L1 refined blocks...")
    l1_refined_dd = dedup_blocks(l1_refined, verbose=verbose)
    
    # ── Step 4: L1 v2 Assembly ──
    if verbose:
        print(f"\n  Step 4: L1 v2 assembly on refined blocks...")
    t0 = time.time()
    l1_v2 = run_l1_assembly_fn(l1_refined_dd)
    if verbose:
        print(f"  L1 v2 assembly done in {time.time() - t0:.0f}s")
    results['l1_assembly'] = l1_v2
    
    # ── Step 5: L2 Assembly ──
    if verbose:
        print(f"\n  Step 5: L2 assembly...")
    t0 = time.time()
    l2_blocks = run_l2_assembly_fn(l1_v2)
    if verbose:
        print(f"  L2 assembly done in {time.time() - t0:.0f}s")
    results['l2_assembly'] = l2_blocks
    
    # ── Step 6: L2 Refinement ──
    if verbose:
        print(f"\n  Step 6: L2 refinement...")
    l2_refined = refine_at_level(
        l2_blocks, raw_blocks,
        global_probs, global_sites, num_samples,
        raw_per_super=batch_size * batch_size,
        penalty_scale=penalty_scale,
        recomb_rate=recomb_rate,
        n_generations=n_generations,
        verbose=verbose
    )
    results['l2_refined'] = l2_refined
    
    return results