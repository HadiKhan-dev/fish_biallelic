"""
paint_samples.py (IBS-AWARE VITERBI PAINTING)

A robust module for reconstructing diploid haplotypes (painting) from probabilistic 
genotype data using a Viterbi algorithm with IBS-aware pedigree support.

BINNED VERSION: Aggregates SNP-level emissions into bins (~100 SNPs/bin) to reduce
memory usage by ~100x while preserving accuracy.

Key Features:
1.  **Binned Emissions:** Sums per-SNP log-likelihoods into bins for memory efficiency.
2.  **Single Viterbi Path:** Fast, deterministic best-path reconstruction per sample.
3.  **IBS-Aware Pedigree Support:** Instead of enumerating tolerance paths for IBS
    ambiguity, the pedigree inference stage handles IBS equivalence directly via
    allele-level comparison. This eliminates the path explosion / straggler
    problem entirely.
4.  **Double-Recomb Discount:** Prefers simultaneous switches to prevent 2^N path explosion.
5.  **Visualization:** Plots individual paintings AND whole-population.
6.  **Parallel Execution:** Uses multiprocessing with forkserver and dynamic thread scaling.
7.  **Deterministic Emissions:** Converts probabilistic haplotypes to deterministic via argmax
    to prevent epistemic uncertainty from biasing founder selection.

IMPORTANT: Uses float64 for alpha arrays to prevent precision loss over long chromosomes.

IMPORTANT FIX (v2): Emission calculations now use DETERMINISTIC haplotypes (via argmax).
This fixes the "founder aliasing" bug where epistemic uncertainty in founder haplotypes
(e.g., 50/50 probability at a site) caused systematic bias toward uncertain founders.
The probabilistic approach gave uncertain founders "moderate" scores everywhere, making
them appear better than founders who were certain but had occasional mismatches. By
converting to deterministic alleles, we treat "I don't know" as a coin flip (unbiased
noise) rather than as evidence (systematic bias).
"""

import thread_config
from thread_config import numba_thread_scope

import os
import numpy as np
import math
import pandas as pd
import warnings
from typing import List, Tuple, Dict, NamedTuple, Set, DefaultDict, Counter, Optional, Union
from collections import defaultdict
import copy
import multiprocessing as _mp
try:
    _forkserver_ctx = _mp.get_context('forkserver')
except ValueError:
    _forkserver_ctx = _mp.get_context('fork')
from tqdm import tqdm
from functools import partial

import analysis_utils

# --- VISUALIZATION IMPORTS ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    import networkx as nx
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Suppress warnings
warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Painting will be slow.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================

class PaintedChunk(NamedTuple):
    start: int
    end: int
    hap1: int
    hap2: int

class SamplePainting:
    def __init__(self, sample_index: int, chunks: List[PaintedChunk],
                 raw_chunks: Optional[List[PaintedChunk]] = None):
        self.sample_index = sample_index
        self.chunks = chunks 
        # `raw_chunks`: the painter's UNCLEANED Viterbi output (with W
        # chunks and short chunks intact).  When emission-aware cleanup
        # is applied inside _worker_paint_batch_binned, the worker
        # stashes the raw chunks here and replaces `chunks` with the
        # cleaned version.  Downstream stages (pedigree inference,
        # phase correction, allele-level reconstruction) consume
        # `chunks`; the topology validator and any diagnostic that
        # wants to see what the painter actually emitted before
        # cleanup should consult `raw_chunks`.  Defaults to the same
        # list object as `chunks` when no cleanup was applied, so call
        # sites that don't distinguish between raw and cleaned work
        # unchanged.
        self.raw_chunks = chunks if raw_chunks is None else raw_chunks
        self.num_recombinations = max(0, len(self.chunks) - 1)

    def __repr__(self):
        return f"<SamplePainting ID {self.sample_index}: {len(self.chunks)} chunks>"

    def __iter__(self):
        return iter(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]

# =============================================================================
# 2. PAINTING CONTAINERS
# =============================================================================

class BlockPainting:
    def __init__(self, block_position_range: Tuple[int, int], samples: List[SamplePainting]):
        self.start_pos = block_position_range[0]
        self.end_pos = block_position_range[1]
        self.samples = samples
        self.num_samples = len(samples)

    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return self.samples[idx]
    def __iter__(self): return iter(self.samples)

# =============================================================================
# 3. CHUNK POST-PROCESSING (SHORT-CHUNK & WILDCARD MERGE)
# =============================================================================
# Paint_samples' Viterbi reconstruction can produce two kinds of chunks that
# downstream stages (phase correction, allele-level reconstruction) generally
# would rather not see in their input:
#
#   (a) SHORT CHUNKS that survived the Viterbi switch penalty.  With
#       switch_penalty_per_snp = alpha and snps_per_bin = B, the per-event
#       cost of a round-trip excursion to a different state is roughly
#       2 * alpha * B * (round_trip_n_bins).  At alpha=1, B=100, that means
#       spurious round trips of at least ~30 bins (~3000 SNPs) need to
#       overcome the penalty.  Anything shorter that *did* survive
#       therefore had strong emission support; but a small handful of
#       short residual chunks still slip through, e.g. near the boundary
#       of a true recombination event where allele-level evidence is
#       genuinely ambiguous for a kilobase or two.
#
#   (b) WILDCARD CHUNKS (W).  When `wildcard_per_snp_penalty` is enabled
#       (see calculate_binned_emissions), the painter has a W slot it can
#       emit in regions where no real founder fits well.  W is a
#       PLACEHOLDER -- it carries no founder identity and therefore
#       cannot be consumed by phase correction or pedigree allele-level
#       scoring directly (pedigree_inference handles W as missing data
#       via the -1 sentinel route, but that is the exception).
#       Downstream code wanting a real founder identity at every position
#       needs W chunks resolved to a real-founder neighbour BEFORE it
#       runs.
#
# Two cleaner implementations are provided here:
#
#   * `merge_short_chunks`  /  `clean_block_painting`
#     The GEOMETRIC version.  Removes removable chunks using only chunk
#     boundary positions (longer-bp neighbour absorbs; same-ordered-tuple
#     neighbours collapse across).  Self-contained: only needs the chunk
#     list.  No threshold gating.  Provided as a quick fallback when
#     the per-bin emission LL is not available (e.g. operating on a
#     painting reloaded from a checkpoint after the LL data has been
#     discarded).
#
#   * `merge_short_chunks_emission_aware`  /
#     `clean_block_painting_emission_aware`
#     The LL-AWARE version (PREFERRED).  Uses the per-bin emission
#     log-likelihoods (the same LL table the Viterbi forward pass
#     consumed) to (i) find the OPTIMAL bp coordinate at which one
#     neighbour should extend over the removed chunk -- a dynamic-
#     programming "best split" over bin positions, replacing the naive
#     longer-neighbour heuristic -- and (ii) GATE every merge by a
#     per-SNP LL-worsening threshold so that chunks the data strongly
#     supports as distinct (genuine short recombinations,
#     unresolvable-into-real-founder wildcard regions) are KEPT in
#     spite of being short or wildcard.  Requires the per-bin LL
#     array; intended call site is inside the painting worker
#     (where LL is naturally available) or anywhere the LL has been
#     persisted alongside the painting.
#
# DESIGN CHOICES (shared by both implementations):
#   - Threshold for "short" is a parameter (caller picks); default usage
#     is 10 kbp (matches the existing N_chunks_lt_10kb diagnostic).
#   - Wildcard chunks are removed regardless of length, controlled by a
#     separate flag (`merge_wildcard`, default True).
#   - When deciding HOW two neighbours' tuples interact for collapse
#     purposes, the comparison is ORDERED -- (a,b) and (b,a) are
#     considered DIFFERENT.  The Viterbi's K = num_haps^2 state space
#     distinguishes (a,b) from (b,a) as different states; an
#     a,b -> short -> b,a sequence is a Viterbi-asserted PHASE FLIP,
#     and collapsing via unordered equality would erase that flip and
#     confuse downstream phase correction.
#   - When the ENTIRE painting is removable (e.g. a single-chunk W
#     painting), the cleaner leaves it alone -- there's no real-founder
#     neighbour to extend from, and erasing it would lose position
#     information entirely.
#   - Neither cleaner is called automatically from inside
#     `paint_chromosome`.  The topology validator wants to see RAW W
#     chunks (preserves the "where did the painter punt" diagnostic),
#     so cleaning should happen in pipeline.py after Stage 10
#     validation -- OR be wired into the painting worker explicitly
#     for the LL-aware version that needs the in-worker LL access.

def merge_short_chunks(chunks, min_chunk_bp, n_real_founders,
                       merge_wildcard=True):
    """
    Iteratively remove short and/or wildcard chunks by extending neighbours.
    Returns a NEW list of PaintedChunk (does not mutate input).

    Parameters
    ----------
    chunks : list[PaintedChunk]
        Input chunks, ordered by .start, tiling the chromosome (i.e. each
        chunk's .end == next chunk's .start).  The cleaner preserves
        this tiling invariant.
    min_chunk_bp : int
        Chunks with (end - start) < min_chunk_bp are removable as
        "short".  Set to 0 to disable short-chunk merging (then only
        wildcards are merged, if merge_wildcard=True).
    n_real_founders : int
        Number of REAL founder slots in the painter's state space.
        A chunk is a wildcard chunk iff hap1 >= n_real_founders OR
        hap2 >= n_real_founders (W_idx is appended one past the real
        range by calculate_binned_emissions).
    merge_wildcard : bool
        If True (default), wildcard chunks are removed regardless of
        length.  If False, wildcards are only removed when they fall
        below min_chunk_bp.

    Returns
    -------
    list[PaintedChunk]
        Cleaned chunks list.  May be shorter than the input.  Tiles the
        same [chunks[0].start, chunks[-1].end] interval as the input.
    """
    if not chunks:
        return list(chunks)

    def is_wildcard(c):
        return (c.hap1 >= n_real_founders) or (c.hap2 >= n_real_founders)

    def is_short(c):
        return (c.end - c.start) < min_chunk_bp

    def should_remove(c):
        return is_short(c) or (merge_wildcard and is_wildcard(c))

    def same_ordered_tuple(c1, c2):
        # Ordered (not unordered!) -- see "DESIGN CHOICES" note above
        # re: preserving Viterbi-asserted phase flips like (a,b)->(b,a).
        return (c1.hap1 == c2.hap1) and (c1.hap2 == c2.hap2)

    work = list(chunks)

    # Iterate to fixed point.  Each pass finds the FIRST removable chunk
    # and merges it; restart the scan after each modification so we
    # always see fresh-state neighbours (a merge can create a chunk that
    # itself becomes a new neighbour of a previously-unhandled removable
    # chunk).  Worst case O(N^2) for a painting that's entirely small
    # chunks, but typical chunk counts (<= ~50 per (sample, contig)
    # after the wildcard fix) make this trivial.
    while True:
        target = -1
        for i, c in enumerate(work):
            if should_remove(c):
                target = i
                break

        if target < 0:
            break  # No removable chunks remain -> done

        if len(work) == 1:
            # Single-chunk painting that's somehow short/W; nothing to
            # extend from.  Leave it.  Better to keep the position
            # information than erase the chunk entirely.
            break

        c = work[target]
        left  = work[target - 1] if target > 0           else None
        right = work[target + 1] if target < len(work)-1 else None

        # --- Boundary case 1: chunk at chromosome start, only right ---
        if left is None:
            new_right = PaintedChunk(start=c.start, end=right.end,
                                     hap1=right.hap1, hap2=right.hap2)
            # work[target] = c, work[target+1] = right both removed;
            # new_right replaces them
            work = [new_right] + work[target + 2:]
            continue

        # --- Boundary case 2: chunk at chromosome end, only left ---
        if right is None:
            new_left = PaintedChunk(start=left.start, end=c.end,
                                    hap1=left.hap1, hap2=left.hap2)
            work = work[:target - 1] + [new_left]
            continue

        # --- Case A: both neighbours, same ORDERED tuple -> collapse ---
        if same_ordered_tuple(left, right):
            new_chunk = PaintedChunk(start=left.start, end=right.end,
                                     hap1=left.hap1, hap2=left.hap2)
            # Replace [left, c, right] with [new_chunk]
            work = work[:target - 1] + [new_chunk] + work[target + 2:]
            continue

        # --- Case B: both neighbours, different tuples -> longer wins ---
        # Tie-breaker: when left_len == right_len, the >= test below
        # picks LEFT.  Arbitrary but deterministic.
        left_len  = left.end  - left.start
        right_len = right.end - right.start
        if left_len >= right_len:
            new_left = PaintedChunk(start=left.start, end=c.end,
                                    hap1=left.hap1, hap2=left.hap2)
            # Replace [left, c] with [new_left]
            work = work[:target - 1] + [new_left] + work[target + 1:]
        else:
            new_right = PaintedChunk(start=c.start, end=right.end,
                                     hap1=right.hap1, hap2=right.hap2)
            # Replace [c, right] with [new_right]
            work = work[:target] + [new_right] + work[target + 2:]
        # continue implicit -- top of while loop

    return work


def clean_block_painting(painting, min_chunk_bp, n_real_founders,
                         merge_wildcard=True):
    """
    Apply `merge_short_chunks` to every sample in a BlockPainting.

    Returns a NEW BlockPainting (input is not mutated).  Each
    SamplePainting's chunks list is replaced by the cleaned list; the
    sample_index and the BlockPainting's position range are preserved.

    Intended call site: pipeline.py, AFTER the Stage-10 topology
    validation (which wants to see raw W chunks for diagnostics) and
    BEFORE any stage that needs a real founder ID at every painted
    position.

    Example
    -------
    >>> cleaned = clean_block_painting(
    ...     raw_painting,
    ...     min_chunk_bp=10000,
    ...     n_real_founders=len(discovered_block.haplotypes),
    ...     merge_wildcard=True,
    ... )
    """
    new_samples = []
    for sample in painting.samples:
        new_chunks = merge_short_chunks(
            sample.chunks, min_chunk_bp=min_chunk_bp,
            n_real_founders=n_real_founders,
            merge_wildcard=merge_wildcard,
        )
        new_samples.append(SamplePainting(sample.sample_index, new_chunks))
    return BlockPainting(
        (painting.start_pos, painting.end_pos),
        new_samples,
    )


def merge_short_chunks_emission_aware(
    chunks,
    *,
    min_chunk_bp,
    n_real_founders,
    bin_edges,
    binned_log_likelihoods,
    state_definitions,
    hap_keys,
    snps_per_bin,
    threshold_short_nats_per_snp=0.02,
    threshold_wildcard_nats_per_snp=0.20,
    merge_wildcard=True,
):
    """
    LL-aware short-chunk and wildcard cleaner.  Removes short and/or
    wildcard chunks by extending neighbours, using the per-bin emission
    log-likelihoods (the SAME LL table the Viterbi forward pass
    consumed) to (i) choose the OPTIMAL bp coordinate at which to make
    the extension, and (ii) GATE the merge by a per-SNP LL-worsening
    threshold so the Viterbi's choice is only overridden when the chunk
    was emission-ambiguous.

    ALGORITHM (per removable chunk C with bin span [b_start, b_end)):

      Baseline:
        LL_baseline = sum over bins [b_start, b_end) of
                      LL[bin, state_of(C)]
        (i.e. the data fit under the Viterbi's choice for these bins)

      Case A -- both neighbours, SAME ordered tuple (collapse):
        replacement state = left's state
        LL_after = sum over bins [b_start, b_end) of
                   LL[bin, state_of(left)]
        Build new chunks: [..., left.start -> right.end as left's
                           state, ...].

      Case B -- both neighbours, DIFFERENT tuples (optimal split via DP):
        For each candidate split k in [b_start, b_end], compute
            LL_at_split(k) = sum_{b in [b_start, k)} LL[b, left_state]
                           + sum_{b in [k, b_end)}   LL[b, right_state]
        Maximise over k via prefix sums.  Build new chunks:
            left extends from left.start to bin_edges[k*]
            right starts from bin_edges[k*] to right.end
        with k* boundary cases (k* == b_start: right absorbs all;
        k* == b_end: left absorbs all) handled explicitly.

      Case C -- only one neighbour (chromosome boundary):
        Forced: that neighbour absorbs the chunk.
        LL_after = sum over bins [b_start, b_end) of LL[bin, that_neighbour_state]

      Threshold check (applies to all three cases):
        drop_per_snp = (LL_baseline - LL_after) / (n_bins_in_chunk * snps_per_bin)
        if drop_per_snp <= threshold:  do the merge
        else:                          KEEP the chunk

      Threshold used = threshold_wildcard_nats_per_snp when the chunk is
      a wildcard (hap1 >= n_real_founders OR hap2 >= n_real_founders),
      otherwise threshold_short_nats_per_snp.

      Iteration:
        After each successful merge, the chunks list changes and
        previously-rejected merges may now be reachable from new
        neighbours.  Restart the scan from index 0 and re-attempt.
        Terminate when a full pass yields no successful merges.
        Worst case O(N_chunks^2); typical case <= 50 chunks so this
        is trivial.

    DESIGN NOTES (in addition to the shared notes at the top of this
    section):

      * Per-SNP units for the threshold.  Total LL drop scales with
        chunk size; per-SNP normalisation makes the threshold
        scale-invariant.  Per-SNP threshold of 0.02 nats means "the
        merge degrades the per-SNP likelihood by at most ~2 percent."
        Per-SNP threshold of 0.20 nats means "by at most ~20 percent."

      * Two thresholds, not one.  Wildcards have a different
        per-SNP-LL profile from real chunks: W's per-SNP LL is the
        constant -wildcard_per_snp_penalty (e.g. -0.05) regardless of
        the data, while a real founder's per-SNP LL ranges from very
        good (e.g. -0.007 in well-fitting regions) to very poor
        (e.g. -0.5+ in regions where the disc-hap is locally
        chimeric).  In problem regions, W's constant LL is BETTER
        than any real founder's LL, which is precisely why the
        painter chose W there.  Resolving W to the locally-best real
        founder therefore has a per-SNP LL DROP of (real_LL - W_LL)
        > 0.  Diagnostic from chr3 problem regions gives that drop
        as ~0.037 nats/SNP for the best-alt real founder, so the
        default threshold_wildcard_nats_per_snp = 0.20 comfortably
        passes wildcard resolution in those regions while remaining
        a sanity cap for cases where no real founder fits at all.

      * Threshold gates EVEN SAME-TUPLE COLLAPSES.  The data within
        a short chunk might be strongly explained by the chunk's own
        state -- e.g. a genuine recombination from (0,1) to (2,3) back
        to (0,1) where (2,3) emits well at the chunk's bins.
        Collapsing across erases that signal.  The threshold check
        will reject the collapse if the interior bins strongly
        prefer the chunk's state.

      * State lookup.  state_definitions stores HAP INDICES (positions
        in hap_keys); PaintedChunk.hap1/.hap2 store FOUNDER IDs (the
        values in hap_keys).  An inverse map (fid1, fid2) -> state_idx
        is built once per call.  Chunks whose tuple is not in the
        state space (shouldn't happen for Viterbi output, but defended
        against) are treated as having LL_baseline = -inf, which makes
        ANY merge pass the threshold (i.e. "if we can't even score the
        chunk's own state, we have no reason to keep it").

    Parameters
    ----------
    chunks : list[PaintedChunk]
        Input chunks.  Must tile the bp range [chunks[0].start,
        chunks[-1].end) and each chunk's .start / .end MUST coincide
        with a value in `bin_edges` (i.e. chunks should come from
        `reconstruct_single_best_path_binned`).
    min_chunk_bp : int
        Real chunks shorter than this are candidates for removal.
        Set to 0 to disable short-chunk merging (then only wildcards
        are merged, if merge_wildcard=True).
    n_real_founders : int
        Number of real founder slots.  Chunks with hap1 OR hap2 >=
        this are wildcard chunks.
    bin_edges : (n_bins+1,) np.ndarray
        Bin boundaries in bp coordinates.  Bin b covers
        [bin_edges[b], bin_edges[b+1]).
    binned_log_likelihoods : (n_bins, K) np.ndarray
        Per-bin per-state log-likelihoods.  This is the SAME array
        the Viterbi forward pass consumed for this sample.
    state_definitions : (K, 2) np.ndarray
        State index -> (hap_idx_1, hap_idx_2) where hap_idx is a
        position in `hap_keys`.
    hap_keys : list[int]
        Ordered list of founder IDs (the values referenced by
        state_definitions indices).  For 6 real founders + W:
        typically [0, 1, 2, 3, 4, 5, 6] with hap_keys[6] = W_idx.
    snps_per_bin : int
        Number of SNPs aggregated into each bin (used for per-SNP
        normalisation of the LL drop).
    threshold_short_nats_per_snp : float, default 0.02
        Maximum acceptable per-SNP LL worsening for merging a SHORT
        REAL chunk.  Smaller = more conservative (closer to Viterbi's
        original choice).
    threshold_wildcard_nats_per_snp : float, default 0.20
        Maximum acceptable per-SNP LL worsening for resolving a
        WILDCARD chunk.  Set to np.inf for unconditional resolution.
    merge_wildcard : bool, default True
        If True, wildcards are candidates for removal regardless of
        length.

    Returns
    -------
    list[PaintedChunk]
        Cleaned chunks list.  Tiles the same bp range as the input.
    """
    if not chunks:
        return list(chunks)

    bin_edges_arr = np.asarray(bin_edges)
    ll_arr = np.asarray(binned_log_likelihoods)
    sd_arr = np.asarray(state_definitions)
    n_bins_total = int(ll_arr.shape[0])

    # ---- Build state lookup: (founder_id_1, founder_id_2) -> state_idx ----
    # state_definitions stores HAP INDICES (positions in hap_keys); chunks
    # store FOUNDER IDs (values in hap_keys).  Cache the inverse map once
    # so per-chunk lookups are O(1).
    state_idx_for_pair = {}
    for k in range(sd_arr.shape[0]):
        h1_idx = int(sd_arr[k, 0])
        h2_idx = int(sd_arr[k, 1])
        if 0 <= h1_idx < len(hap_keys) and 0 <= h2_idx < len(hap_keys):
            fid1 = int(hap_keys[h1_idx])
            fid2 = int(hap_keys[h2_idx])
            state_idx_for_pair[(fid1, fid2)] = k

    def state_idx(fid1, fid2):
        return state_idx_for_pair.get((int(fid1), int(fid2)), None)

    # ---- Map a chunk's bp range to its bin span [b_start, b_end_excl) ----
    def chunk_bin_span(c):
        b_start = int(np.searchsorted(bin_edges_arr, c.start))
        b_end_excl = int(np.searchsorted(bin_edges_arr, c.end))
        b_start = max(0, min(b_start, n_bins_total))
        b_end_excl = max(b_start, min(b_end_excl, n_bins_total))
        return b_start, b_end_excl

    # ---- Sum of LL[b, s] over a bin range, with None state -> -inf ----
    def total_ll(b_start, b_end_excl, s_idx):
        if b_end_excl <= b_start:
            return 0.0
        if s_idx is None:
            return float(-np.inf)
        return float(ll_arr[b_start:b_end_excl, s_idx].sum())

    # ---- Removability predicates ----
    def is_wildcard(c):
        return (c.hap1 >= n_real_founders) or (c.hap2 >= n_real_founders)

    def is_short(c):
        return (c.end - c.start) < min_chunk_bp

    def should_remove(c):
        if is_wildcard(c):
            return merge_wildcard
        return is_short(c)

    def same_ordered_tuple(c1, c2):
        # Ordered comparison preserves Viterbi-asserted phase flips
        # (a,b) -> short -> (b,a); see section docstring.
        return (c1.hap1 == c2.hap1) and (c1.hap2 == c2.hap2)

    work = list(chunks)

    # ---- Main loop: iterate to fixed point ----
    while True:
        made_progress = False

        for i in range(len(work)):
            c = work[i]
            if not should_remove(c):
                continue

            b_start, b_end_excl = chunk_bin_span(c)
            n_bins_in_chunk = b_end_excl - b_start
            if n_bins_in_chunk <= 0:
                # Degenerate chunk (zero bins between its start and end);
                # we can't reason about its LL.  Skip.
                continue
            n_snps_in_chunk = n_bins_in_chunk * snps_per_bin

            # Baseline LL: chunk in its Viterbi-painted state
            chunk_s = state_idx(c.hap1, c.hap2)
            ll_baseline = total_ll(b_start, b_end_excl, chunk_s)
            # If chunk_s is None, ll_baseline = -inf and any candidate
            # merge automatically passes the threshold.  This handles
            # malformed inputs gracefully but should never trigger from
            # well-formed Viterbi output.

            threshold = (threshold_wildcard_nats_per_snp if is_wildcard(c)
                         else threshold_short_nats_per_snp)

            left  = work[i - 1] if i > 0               else None
            right = work[i + 1] if i < len(work) - 1   else None

            if left is None and right is None:
                # Single-chunk painting; nothing to extend from
                break  # exit for-loop; no merges possible this pass

            # --- Decide best merge plan + its LL_after ---
            # `merge_plan` is (new_chunks_list, ll_after).
            merge_plan = None

            if left is not None and right is not None:
                if same_ordered_tuple(left, right):
                    # Case A: collapse.  Replacement state = left's
                    # state (== right's state by ordered equality).
                    s_replace = state_idx(left.hap1, left.hap2)
                    ll_after = total_ll(b_start, b_end_excl, s_replace)
                    new_chunk = PaintedChunk(
                        start=left.start, end=right.end,
                        hap1=left.hap1, hap2=left.hap2)
                    new_chunks = work[:i - 1] + [new_chunk] + work[i + 2:]
                    merge_plan = (new_chunks, ll_after)
                else:
                    # Case B: DP optimal split between left's state and
                    # right's state across the chunk's bins.
                    s_left  = state_idx(left.hap1,  left.hap2)
                    s_right = state_idx(right.hap1, right.hap2)

                    # Per-bin LL for each candidate state; None -> -inf
                    if s_left is None:
                        left_ll = np.full(n_bins_in_chunk, -np.inf)
                    else:
                        left_ll = ll_arr[b_start:b_end_excl, s_left].astype(np.float64)
                    if s_right is None:
                        right_ll = np.full(n_bins_in_chunk, -np.inf)
                    else:
                        right_ll = ll_arr[b_start:b_end_excl, s_right].astype(np.float64)

                    # Prefix sums (length n+1, indexed by local-k)
                    prefix_left  = np.concatenate(([0.0], np.cumsum(left_ll)))
                    prefix_right = np.concatenate(([0.0], np.cumsum(right_ll)))
                    total_right_ll = prefix_right[n_bins_in_chunk]
                    # LL_at_split(local_k) = prefix_left[local_k]
                    #                       + (total_right_ll - prefix_right[local_k])
                    ll_per_split = prefix_left + (total_right_ll - prefix_right)
                    best_local_k = int(np.argmax(ll_per_split))
                    ll_after = float(ll_per_split[best_local_k])
                    k_star_bin = b_start + best_local_k  # absolute bin index

                    if k_star_bin <= b_start:
                        # Right absorbs the entire chunk
                        new_right_chunk = PaintedChunk(
                            start=c.start, end=right.end,
                            hap1=right.hap1, hap2=right.hap2)
                        new_chunks = work[:i] + [new_right_chunk] + work[i + 2:]
                    elif k_star_bin >= b_end_excl:
                        # Left absorbs the entire chunk
                        new_left_chunk = PaintedChunk(
                            start=left.start, end=c.end,
                            hap1=left.hap1, hap2=left.hap2)
                        new_chunks = work[:i - 1] + [new_left_chunk] + work[i + 1:]
                    else:
                        # Partial split at bin boundary k_star_bin
                        split_bp = int(bin_edges_arr[k_star_bin])
                        new_left_chunk = PaintedChunk(
                            start=left.start, end=split_bp,
                            hap1=left.hap1, hap2=left.hap2)
                        new_right_chunk = PaintedChunk(
                            start=split_bp, end=right.end,
                            hap1=right.hap1, hap2=right.hap2)
                        new_chunks = (work[:i - 1] + [new_left_chunk, new_right_chunk]
                                       + work[i + 2:])
                    merge_plan = (new_chunks, ll_after)

            elif left is None:
                # Boundary case: chromosome start, only right neighbour
                s_right = state_idx(right.hap1, right.hap2)
                ll_after = total_ll(b_start, b_end_excl, s_right)
                new_right_chunk = PaintedChunk(
                    start=c.start, end=right.end,
                    hap1=right.hap1, hap2=right.hap2)
                new_chunks = [new_right_chunk] + work[i + 2:]
                merge_plan = (new_chunks, ll_after)

            else:  # right is None
                # Boundary case: chromosome end, only left neighbour
                s_left = state_idx(left.hap1, left.hap2)
                ll_after = total_ll(b_start, b_end_excl, s_left)
                new_left_chunk = PaintedChunk(
                    start=left.start, end=c.end,
                    hap1=left.hap1, hap2=left.hap2)
                new_chunks = work[:i - 1] + [new_left_chunk]
                merge_plan = (new_chunks, ll_after)

            # ---- Threshold check ----
            if merge_plan is None:
                continue
            new_chunks, ll_after = merge_plan
            drop = ll_baseline - ll_after  # >0 means merge worsens the fit
            drop_per_snp = (drop / n_snps_in_chunk) if n_snps_in_chunk > 0 else 0.0
            if drop_per_snp <= threshold:
                work = new_chunks
                made_progress = True
                break  # restart scan from index 0

            # else: keep this chunk, try next removable chunk in this pass

        if not made_progress:
            break

    return work


def clean_block_painting_emission_aware(
    painting,
    *,
    min_chunk_bp,
    n_real_founders,
    bin_edges,
    per_sample_binned_log_likelihoods,
    state_definitions,
    hap_keys,
    snps_per_bin,
    threshold_short_nats_per_snp=0.02,
    threshold_wildcard_nats_per_snp=0.20,
    merge_wildcard=True,
):
    """
    Apply `merge_short_chunks_emission_aware` to every sample in a
    BlockPainting.  Returns a NEW BlockPainting (input not mutated);
    sample_indexes and the BlockPainting's position range are preserved.

    Parameters
    ----------
    painting : BlockPainting
        The raw painting from paint_chromosome.
    min_chunk_bp, n_real_founders, bin_edges, state_definitions,
    hap_keys, snps_per_bin, threshold_*, merge_wildcard
        Forwarded to merge_short_chunks_emission_aware (see its
        docstring).
    per_sample_binned_log_likelihoods : (n_samples, n_bins, K) np.ndarray
        The Viterbi forward pass's per-bin per-state log-likelihoods,
        one (n_bins, K) slice per sample, indexed in the same order
        as painting.samples.

    Note
    ----
    The per-sample LL arrays are NOT persisted in the BlockPainting
    checkpoint by default.  Callers therefore usually invoke this
    cleaner from inside the painting worker (immediately after Viterbi
    reconstruction, where the LL is in scope) rather than from
    pipeline.py after the painting has been written to disk.
    """
    ll_arr = np.asarray(per_sample_binned_log_likelihoods)
    if ll_arr.ndim != 3:
        raise ValueError(
            f"per_sample_binned_log_likelihoods must be 3-D "
            f"(n_samples, n_bins, K); got shape {ll_arr.shape}")
    if ll_arr.shape[0] != len(painting.samples):
        raise ValueError(
            f"per_sample_binned_log_likelihoods has {ll_arr.shape[0]} samples "
            f"but painting has {len(painting.samples)}")

    new_samples = []
    for i, sample in enumerate(painting.samples):
        new_chunks = merge_short_chunks_emission_aware(
            sample.chunks,
            min_chunk_bp=min_chunk_bp,
            n_real_founders=n_real_founders,
            bin_edges=bin_edges,
            binned_log_likelihoods=ll_arr[i],
            state_definitions=state_definitions,
            hap_keys=hap_keys,
            snps_per_bin=snps_per_bin,
            threshold_short_nats_per_snp=threshold_short_nats_per_snp,
            threshold_wildcard_nats_per_snp=threshold_wildcard_nats_per_snp,
            merge_wildcard=merge_wildcard,
        )
        new_samples.append(SamplePainting(sample.sample_index, new_chunks))
    return BlockPainting(
        (painting.start_pos, painting.end_pos),
        new_samples,
    )

# =============================================================================
# 4. HELPER: DENSE MATRIX CONVERSION
# =============================================================================

def founder_block_to_dense(block_result):
    """Convert probabilistic haplotypes to dense integer matrix via argmax."""
    positions = np.array(block_result.positions, dtype=np.int64)
    hap_dict = block_result.haplotypes
    if not hap_dict: return np.zeros((0, 0), dtype=np.int8), positions
    max_id = max(hap_dict.keys())
    n_sites = len(positions)
    dense_haps = np.full((max_id + 1, n_sites), -1, dtype=np.int8)
    for fid, hap_arr in hap_dict.items():
        if hap_arr.ndim == 2: 
            concrete = np.argmax(hap_arr, axis=1)
        else: 
            concrete = hap_arr
        dense_haps[fid, :] = concrete.astype(np.int8)
    return dense_haps, positions

# =============================================================================
# 5. BINNED EMISSION CALCULATOR (NEW - MEMORY EFFICIENT)
# =============================================================================

def calculate_binned_emissions(sample_probs_matrix, hap_dict, positions, 
                               snps_per_bin=100, robustness_epsilon=1e-2,
                               wildcard_per_snp_penalty=None):
    """
    Calculate emission log-likelihoods aggregated into bins.
    
    MEMORY EFFICIENT: Instead of (n_samples, K, n_sites), produces (n_samples, K, n_bins)
    by summing log-likelihoods within each bin. For 50,000 SNPs with 100 SNPs/bin,
    this reduces memory by ~100x.
    
    IMPORTANT: Haplotypes are converted to DETERMINISTIC (via argmax) before emission
    calculation. This fixes a bug where epistemic uncertainty in founder haplotypes
    caused a systematic bias toward uncertain founders. When a founder has 50/50
    probability at a site, the old probabilistic approach would give it "moderate"
    emissions everywhere, making it appear better than founders who are certain but
    happen to mismatch at a few sites. The argmax approach introduces small unbiased
    noise at uncertain sites (<1%), which is preferable to systematic bias.

    WILDCARD STATE (optional).  If `wildcard_per_snp_penalty` is not
    None, the state space is extended by one extra "founder" slot W at
    index num_real_haps (i.e. immediately past the real range).  W is
    a *virtual* founder that has NO underlying allele sequence; instead,
    every state involving W has a CONSTANT per-SNP log-likelihood of
    -wildcard_per_snp_penalty (independent of the sample data and of
    the robustness epsilon).  Summed over a bin of m SNPs, this gives
    a per-bin LL of -m * wildcard_per_snp_penalty.

    The motivation is to give the Viterbi a coherent "I don't know"
    state that beats all real states only in regions where the
    discovered haplotypes are LOCALLY chimeric for the truth founder
    (mean per-SNP truth-LL substantially more negative than
    -wildcard_per_snp_penalty), while LOSING to the best real state in
    regions where the painting is correct.  Without W, the painter
    flickers through wrong-but-locally-better founders in chimeric
    regions, e.g. (0,1) -> (0,3) -> (0,1) -> (0,4) -> (0,1) over a
    single bad stretch.  With W, the bad stretch is painted as
    (0,W) instead, both robustifying the painting and flagging the
    chimeric region for follow-up.

    Calibration of `wildcard_per_snp_penalty` is empirical -- it must
    sit BELOW the per-SNP LL of the *best non-truth real state* in
    problem regions (so the painter prefers W to switching to a wrong
    founder) and ABOVE the per-SNP LL of the truth state in normal
    regions (so W doesn't displace correct painting).  See
    diagnose_per_snp_ll_for_wildcard.py for measuring these from
    actual data; on the dataset this code was developed on, c = 0.05
    nats/SNP was the calibrated value.

    The W slot is appended to hap_keys as the integer num_real_haps
    (one past the real range), so that downstream code which bounds-
    checks founder IDs against the real founder count (e.g. the
    topology validator's `0 <= h < M_len` test) automatically routes
    W chunks to its "unmappable" branch.  Set wildcard_per_snp_penalty
    to None (default) to preserve original behaviour with no W slot
    in the state space at all.
    
    Args:
        sample_probs_matrix: (n_samples, n_sites, 3) genotype probabilities
        hap_dict: Dict mapping founder ID -> (n_sites,) or (n_sites, 2) haplotypes
                  If (n_sites, 2), columns are P(allele=0), P(allele=1) - converted via argmax
                  If (n_sites,), values are deterministic alleles (0 or 1)
        positions: (n_sites,) array of SNP positions
        snps_per_bin: Number of SNPs to aggregate per bin
        robustness_epsilon: Numerical stability term
        wildcard_per_snp_penalty: float or None.  If not None, enables the
                                  W slot as described above; the value is the
                                  per-SNP cost (in nats) of being in any
                                  W-involving state.  None (default) means
                                  no W slot is added.
    
    Returns:
        binned_ll: (n_samples, K, n_bins) aggregated log-likelihoods
        state_defs: (K, 2) state definitions
        hap_keys: List of founder IDs (with W_idx appended as the last
                  element when wildcard is enabled)
        bin_centers: (n_bins,) physical positions of bin centers
        bin_edges: (n_bins + 1,) bin boundary positions
    """
    hap_keys = sorted(list(hap_dict.keys()))
    num_real_haps = len(hap_keys)
    num_samples, num_sites, _ = sample_probs_matrix.shape

    # WILDCARD: if enabled, extend hap_keys with one extra slot W at
    # index num_real_haps.  W's "founder ID" is the integer num_real_haps
    # itself, chosen so that downstream code which bounds-checks founder
    # IDs against the real founder count (e.g. `0 <= h < M_len` in the
    # topology validator) automatically routes W-containing chunks to
    # its "unmappable" branch.  num_haps below is therefore the SIZE OF
    # THE STATE-SPACE SLOT INDEXING (real + W), not the number of real
    # founders -- the Viterbi forward pass and reconstruction use this
    # same num_haps for their log_N_minus_1 transition prior, so the
    # prior probability of switching is uniform over (num_real_haps + 1)
    # alternatives when W is enabled, vs num_real_haps - 1 alternatives
    # when W is disabled.  This is a deliberate modelling choice: W is
    # an admissible target for any single-position switch on the same
    # footing as any real founder.
    use_wildcard = (wildcard_per_snp_penalty is not None)
    if use_wildcard:
        W_idx = num_real_haps  # int just past the real range
        hap_keys = hap_keys + [W_idx]
        num_haps = num_real_haps + 1
    else:
        num_haps = num_real_haps

    # Create state definitions
    idx_i, idx_j = np.unravel_index(np.arange(num_haps**2), (num_haps, num_haps))
    state_defs = np.stack([idx_i, idx_j], axis=1).astype(np.int32)
    K = num_haps ** 2
    
    if not hap_keys or num_sites == 0:
        return np.zeros((num_samples, K, 0), dtype=np.float64), state_defs, hap_keys, np.array([]), np.array([])
    
    # Compute bin structure
    n_bins = max(1, num_sites // snps_per_bin)
    if n_bins == 0:
        n_bins = 1
    
    # Split SNP indices into approximately equal bins
    bin_snp_indices = np.array_split(np.arange(num_sites), n_bins)
    
    # Compute bin centers (average physical position of SNPs in each bin)
    bin_centers = np.array([positions[idx].mean() for idx in bin_snp_indices], dtype=np.float64)
    
    # Compute bin edges for chunk reconstruction
    bin_edges = np.zeros(n_bins + 1, dtype=np.int64)
    bin_edges[0] = positions[0]
    for i in range(n_bins - 1):
        # Edge is midpoint between last SNP of this bin and first SNP of next bin
        last_pos = positions[bin_snp_indices[i][-1]]
        next_first = positions[bin_snp_indices[i+1][0]]
        bin_edges[i+1] = (last_pos + next_first) // 2
    bin_edges[-1] = positions[-1] + 1  # Final edge past last SNP
    
    # =========================================================================
    # CRITICAL FIX: Convert haplotypes to DETERMINISTIC using argmax
    # =========================================================================
    # This fixes the "founder aliasing" bug where epistemic uncertainty in founder
    # haplotypes caused biased emissions. When founder A is 50/50 uncertain at many
    # sites but founder B is certain, the old probabilistic approach would give A
    # "moderate" scores everywhere while B would get perfect scores at matches but
    # harsh penalties at mismatches. This made uncertain founders appear artificially
    # better than they should be.
    #
    # By converting to deterministic alleles via argmax, we treat "I don't know" as
    # a coin flip rather than as evidence. This introduces small unbiased noise at
    # uncertain sites (<1% of sites typically), which is far preferable to the
    # systematic bias of the probabilistic approach.
    # =========================================================================
    
    deterministic_alleles = np.zeros((num_haps, num_sites), dtype=np.int8)
    
    # NOTE: when wildcard is enabled, hap_keys ends with W_idx for which
    # there is no entry in hap_dict.  We iterate over the REAL founders
    # only (hap_keys[:num_real_haps]) and leave deterministic_alleles
    # row W_idx as its initial zeros.  W's "genotype" contributions to
    # the per-bin LL are meaningless and are OVERWRITTEN with a
    # constant value after the bin loop -- the sentinel zeros are just
    # to keep the subsequent state_genotypes broadcast well-defined.
    for i, k in enumerate(hap_keys[:num_real_haps]):
        hap = hap_dict[k]
        if hap.ndim == 2 and hap.shape[1] == 2:
            # Probabilistic: (n_sites, 2) with P(allele=0), P(allele=1)
            # Use argmax to get deterministic allele (0 or 1)
            deterministic_alleles[i] = np.argmax(hap, axis=1).astype(np.int8)
        else:
            # Already deterministic: (n_sites,) with values 0 or 1
            deterministic_alleles[i] = hap.astype(np.int8)
    
    # Compute deterministic genotypes for all state pairs
    # genotype = allele_i + allele_j (values: 0, 1, or 2)
    # Shape: (num_haps, num_haps, num_sites)
    state_genotypes = (deterministic_alleles[:, None, :] + 
                       deterministic_alleles[None, :, :])
    
    # Reshape to (K, num_sites) to match state indexing
    state_genotypes_flat = state_genotypes.reshape(K, num_sites)
    
    # Allocate binned emissions
    binned_ll = np.zeros((num_samples, K, n_bins), dtype=np.float64)
    
    # Process each bin - aggregate SNP emissions
    uniform_prob = 1.0 / 3.0
    
    for bin_idx, snp_indices in enumerate(bin_snp_indices):
        if len(snp_indices) == 0:
            continue
        
        n_snps_bin = len(snp_indices)
        
        # Extract sample probabilities for SNPs in this bin
        # Shape: (n_samples, n_snps_in_bin, 3)
        sample_probs_bin = sample_probs_matrix[:, snp_indices, :]
        
        # Extract state genotypes for this bin
        # Shape: (K, n_snps_in_bin)
        geno_bin = state_genotypes_flat[:, snp_indices]
        
        # For each state, look up P(observed | genotype) using the deterministic genotype
        # We need to gather: sample_probs_bin[sample, snp, geno[state, snp]]
        # for all (sample, state, snp) combinations
        
        # Efficient approach: compute for each genotype value separately and combine
        model_probs = np.zeros((num_samples, K, n_snps_bin), dtype=np.float64)
        
        for geno_val in range(3):
            # Mask where this genotype applies: (K, n_snps_bin)
            mask = (geno_bin == geno_val)
            # Get sample probability for this genotype: (n_samples, n_snps_bin)
            s_prob = sample_probs_bin[:, :, geno_val]
            # Expand dimensions for broadcasting: (n_samples, 1, n_snps_bin)
            s_prob_expanded = s_prob[:, np.newaxis, :]
            # mask broadcasts from (1, K, n_snps_bin) to (n_samples, K, n_snps_bin)
            # Apply where mask is True
            model_probs += mask[np.newaxis, :, :] * s_prob_expanded
        
        # Apply robustness epsilon
        final_probs = model_probs * (1.0 - robustness_epsilon) + robustness_epsilon * uniform_prob
        
        # Compute log-likelihoods
        final_probs = np.maximum(final_probs, 1e-300)
        ll_snps = np.log(final_probs)
        ll_snps = np.maximum(ll_snps, -50.0)
        
        # Sum log-likelihoods within bin
        binned_ll[:, :, bin_idx] = ll_snps.sum(axis=2)

    # =========================================================================
    # WILDCARD: overwrite W-involving states with a constant per-bin LL.
    # =========================================================================
    # The bin loop above computed binned_ll for ALL K states, including
    # the W-involving ones, but the values for W-involving states are
    # meaningless (they come from looking up sample_probs at the
    # sentinel-zero row of deterministic_alleles).  We now overwrite
    # them with the correct constant: each W-involving state has
    # per-SNP LL = -wildcard_per_snp_penalty, summed over the number of
    # SNPs in the bin.  This is independent of the sample data and of
    # robustness_epsilon, by design -- W is a no-information state.
    #
    # A state k is W-involving iff at least one of (state_defs[k, 0],
    # state_defs[k, 1]) equals W_idx.  For num_real_haps real founders
    # and 1 W slot, there are 2*num_real_haps + 1 W-involving states
    # out of (num_real_haps + 1)**2 total states (e.g. 13 of 49 for 6
    # real founders).
    if use_wildcard:
        w_state_mask = ((state_defs[:, 0] == W_idx) |
                        (state_defs[:, 1] == W_idx))
        # snps_in_each_bin[bin_idx] is the number of SNPs in that bin;
        # np.array_split allocates ceil/floor, so this varies by at most
        # +/-1 across bins (negligible for snps_per_bin >> 1).
        snps_in_each_bin = np.array(
            [len(idx) for idx in bin_snp_indices], dtype=np.int64)
        w_ll_per_bin = (-float(wildcard_per_snp_penalty)
                         * snps_in_each_bin)  # shape (n_bins,)
        # Broadcast assign: LHS is (n_samples, n_w_states, n_bins),
        # RHS is (1, 1, n_bins) -> identical per (sample, W-state) for
        # each bin.
        binned_ll[:, w_state_mask, :] = w_ll_per_bin[None, None, :]

    return binned_ll, state_defs, hap_keys, bin_centers, bin_edges

# =============================================================================
# 6. TOLERANCE VITERBI KERNELS (BINNED VERSION)
# =============================================================================

@njit(parallel=True, fastmath=True)
def run_forward_pass_max_sum_binned(ll_tensor, bin_centers, recomb_rate, state_definitions, 
                                     n_haps, switch_penalty_per_snp, snps_per_bin,
                                     double_recomb_factor=1.5):
    """
    Forward pass of max-sum Viterbi algorithm on BINNED data.
    
    Uses physical distance between bin centers for transition probabilities.

    The transition prior at bin i is the recombination probability
    theta = dist_bp * recomb_rate, plus an extra anti-switch damping that
    SCALES WITH BIN SIZE (number of SNPs aggregated into the bin):

        log_switch = log(theta) - switch_penalty_per_snp * snps_per_bin

    Rationale: with snps_per_bin SNPs aggregated per bin, the emission
    likelihood difference between competing states scales linearly with
    snps_per_bin in the worst case.  A fixed (bin-size-independent)
    switch_penalty therefore implicitly depends on the binning choice
    -- halving snps_per_bin doubles the model's tolerance for spurious
    short excursions.  Making the damping per-SNP keeps the effective
    minimum-chunk-length scale invariant under rebinning.

    Bin sizes are nearly uniform (np.array_split varies by +/-1 SNP),
    so passing a single scalar snps_per_bin (the nominal aggregation
    factor) is accurate to <1% for typical settings.
    """
    n_samples, K, n_bins = ll_tensor.shape
    
    alpha = np.full((n_samples, n_bins, K), -np.inf, dtype=np.float64)
    
    if n_haps > 1: 
        log_N_minus_1 = math.log(float(n_haps - 1))
    else: 
        log_N_minus_1 = 0.0

    for s in prange(n_samples):
        for k in range(K): 
            alpha[s, 0, k] = ll_tensor[s, k, 0]

        for i in range(1, n_bins):
            # Distance between bin centers
            dist_bp = bin_centers[i] - bin_centers[i-1]
            if dist_bp < 1: dist_bp = 1
            theta = float(dist_bp) * recomb_rate
            if theta > 0.5: theta = 0.5
            if theta < 1e-15: theta = 1e-15
            
            log_switch = math.log(theta) - switch_penalty_per_snp * snps_per_bin
            log_stay = math.log(1.0 - theta)
            
            cost_0 = 2.0 * log_stay                                    # Neither switches
            cost_1 = log_switch + log_stay - log_N_minus_1             # One switches
            cost_2 = double_recomb_factor * log_switch - 2.0 * log_N_minus_1  # Both switch
            
            for k_curr in range(K):
                h1_curr = state_definitions[k_curr, 0]
                h2_curr = state_definitions[k_curr, 1]
                best_score = -np.inf
                
                for k_prev in range(K):
                    h1_prev = state_definitions[k_prev, 0]
                    h2_prev = state_definitions[k_prev, 1]
                    
                    dist = 0
                    if h1_curr != h1_prev: dist += 1
                    if h2_curr != h2_prev: dist += 1
                    
                    if dist == 0: trans = cost_0
                    elif dist == 1: trans = cost_1
                    else: trans = cost_2
                    
                    score = alpha[s, i-1, k_prev] + trans
                    if score > best_score: best_score = score
                    
                alpha[s, i, k_curr] = best_score + ll_tensor[s, k_curr, i]
                
    return alpha


# =============================================================================
# 7. VITERBI TRACEBACK (BINNED VERSION)
# =============================================================================

def reconstruct_single_best_path_binned(alpha, ll_tensor, bin_centers, bin_edges,
                                         recomb_rate, state_definitions, n_haps, 
                                         switch_penalty_per_snp, snps_per_bin,
                                         hap_keys, double_recomb_factor=1.5):
    """Reconstruct the single best path via standard Viterbi traceback - BINNED.

    Uses the SAME bin-size-scaled switch cost as run_forward_pass_max_sum_binned
    so the traceback is consistent with the forward pass:
        log_switch = log(theta) - switch_penalty_per_snp * snps_per_bin
    """
    n_bins, K = alpha.shape
    if n_haps > 1: 
        log_N_minus_1 = math.log(float(n_haps - 1))
    else: 
        log_N_minus_1 = 0.0
    
    curr_k = np.argmax(alpha[n_bins-1])
    h1_idx, h2_idx = state_definitions[curr_k]
    t1, t2 = hap_keys[h1_idx], hap_keys[h2_idx]
    
    # Use bin edges for chunk positions
    chunks = [PaintedChunk(start=int(bin_edges[n_bins-1]), end=int(bin_edges[n_bins]), hap1=t1, hap2=t2)]
    
    for t in range(n_bins - 1, 0, -1):
        prev_t = t - 1
        curr_h1, curr_h2 = state_definitions[curr_k]
        
        dist_bp = bin_centers[t] - bin_centers[prev_t]
        if dist_bp < 1: dist_bp = 1
        theta = float(dist_bp) * recomb_rate
        if theta > 0.5: theta = 0.5
        if theta < 1e-15: theta = 1e-15
        
        log_switch = math.log(theta) - switch_penalty_per_snp * snps_per_bin
        log_stay = math.log(1.0 - theta)
        
        cost_0 = 2.0 * log_stay
        cost_1 = log_switch + log_stay - log_N_minus_1
        cost_2 = double_recomb_factor * log_switch - 2.0 * log_N_minus_1
        
        best_prev = -1
        best_score = -np.inf
        
        for prev_k in range(K):
            if alpha[prev_t, prev_k] == -np.inf: continue
            prev_h1, prev_h2 = state_definitions[prev_k]
            dist = 0
            if curr_h1 != prev_h1: dist += 1
            if curr_h2 != prev_h2: dist += 1
            
            trans = cost_0 if dist == 0 else (cost_1 if dist == 1 else cost_2)
            score = alpha[prev_t, prev_k] + trans
            if score > best_score:
                best_score = score
                best_prev = prev_k
                
        curr_k = best_prev
        prev_h1, prev_h2 = state_definitions[curr_k]
        pt1, pt2 = hap_keys[prev_h1], hap_keys[prev_h2]
        old_chunk = chunks[0]
        
        is_extension = False
        if (pt1 == old_chunk.hap1 and pt2 == old_chunk.hap2): is_extension = True
        elif (pt1 == old_chunk.hap2 and pt2 == old_chunk.hap1): is_extension = True
        
        if is_extension:
            chunks[0] = PaintedChunk(start=int(bin_edges[prev_t]), end=old_chunk.end, 
                                     hap1=old_chunk.hap1, hap2=old_chunk.hap2)
        else:
            chunks.insert(0, PaintedChunk(start=int(bin_edges[prev_t]), end=int(bin_edges[t]), 
                                          hap1=pt1, hap2=pt2))
            
    return [SamplePainting(0, chunks)]

# =============================================================================
# 9. MULTIPROCESSING DRIVER (SharedMemory + Persistent Pool)
# =============================================================================

from multiprocessing import shared_memory as _shm

# Worker-local cache for SharedMemory arrays
_PAINT_SHARED = {}
_PAINT_SHM_REFS = []
_PAINT_CHROM_ID = None

# Dynamic thread scaling globals (set by _init_persistent_paint_worker)
_PAINT_ACTIVE_COUNTER = None
_PAINT_TOTAL_CORES = None


def _paint_get_dynamic_threads():
    """Recheck active worker count and return optimal numba thread count."""
    if _PAINT_ACTIVE_COUNTER is None or _PAINT_TOTAL_CORES is None:
        return 1
    active = max(_PAINT_ACTIVE_COUNTER.value, 1)
    return max(1, _PAINT_TOTAL_CORES // active)

def _create_shm_from_array(arr):
    """Create a SharedMemory block from a numpy array. Returns (shm, name, shape, dtype_str)."""
    shm = _shm.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr
    return shm, shm.name, arr.shape, str(arr.dtype)

def _array_from_shm(name, shape, dtype_str):
    """Reconstruct a numpy array from SharedMemory. Returns (shm_ref, array)."""
    shm = _shm.SharedMemory(name=name, create=False)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    return shm, arr


def _worker_paint_batch_binned(args):
    """Worker function for parallel painting — single Viterbi path per sample."""
    indices, start_idx, end_idx = args
    
    # Read from SharedMemory (zero-copy)
    sample_probs_slice = _PAINT_SHARED['block_samples_data'][start_idx:end_idx]
    positions = _PAINT_SHARED['positions']
    hap_dict = _PAINT_SHARED['hap_dict']
    params = _PAINT_SHARED['params']
    
    recomb_rate = params['recomb_rate']
    switch_penalty_per_snp = params['switch_penalty_per_snp']
    robustness_epsilon = params['robustness_epsilon']
    double_recomb_factor = params.get('double_recomb_factor', 1.5)
    snps_per_bin = params.get('snps_per_bin', 100)
    numba_threads = params.get('numba_threads', 1)
    # When wildcard is enabled, calculate_binned_emissions extends the
    # state space by one extra W slot (per-SNP LL = -wildcard_per_snp_penalty);
    # see its docstring.  num_haps below picks this up automatically via
    # len(hap_keys) since calculate_binned_emissions returns the W-extended
    # hap_keys when this param is not None.
    wildcard_per_snp_penalty = params.get('wildcard_per_snp_penalty', None)
    # Post-processing knobs (see merge_short_chunks_emission_aware for
    # the algorithm; called per-sample below).  Defaults match the
    # values exposed via paint_chromosome's kwargs so that running
    # the worker through that entry point is equivalent.
    enable_emission_aware_cleanup = params.get(
        'enable_emission_aware_cleanup', True)
    cleanup_min_chunk_bp = params.get('cleanup_min_chunk_bp', 10000)
    cleanup_threshold_short_nats_per_snp = params.get(
        'cleanup_threshold_short_nats_per_snp', 0.02)
    cleanup_threshold_wildcard_nats_per_snp = params.get(
        'cleanup_threshold_wildcard_nats_per_snp', 0.20)
    cleanup_merge_wildcard = params.get('cleanup_merge_wildcard', True)
    
    # Calculate BINNED emissions
    ll_tensor, state_defs, hap_keys, bin_centers, bin_edges = calculate_binned_emissions(
        sample_probs_slice, hap_dict, positions,
        snps_per_bin=snps_per_bin,
        robustness_epsilon=robustness_epsilon,
        wildcard_per_snp_penalty=wildcard_per_snp_penalty,
    )
    num_haps = len(hap_keys)
    n_bins = len(bin_centers)
    
    if n_bins == 0:
        # No bins - return empty results
        results = []
        for global_idx in indices:
            results.append(SamplePainting(global_idx, []))
        return results
    
    # `n_real_founders` for the cleaner: when wildcard is enabled
    # calculate_binned_emissions appends ONE extra W slot to hap_keys,
    # so the W slot index is len(hap_keys) - 1.  When wildcard is
    # disabled, all hap_keys are real founders.
    if wildcard_per_snp_penalty is not None:
        n_real_founders_for_cleanup = num_haps - 1
    else:
        n_real_founders_for_cleanup = num_haps
    
    # Control Numba thread count for prange loops in the forward kernel.
    # Dynamic scaling: use more threads when fewer workers are active (tail).
    dyn_threads = _paint_get_dynamic_threads()
    effective_threads = max(numba_threads, dyn_threads)
    with numba_thread_scope(effective_threads):
        # Run forward pass on BINNED data (no backward pass needed for single Viterbi)
        alpha = run_forward_pass_max_sum_binned(ll_tensor, bin_centers, recomb_rate, state_defs, 
                                                 num_haps, float(switch_penalty_per_snp),
                                                 int(snps_per_bin), double_recomb_factor)
    
    results = []
    for i, global_idx in enumerate(indices):
        # Single Viterbi best path — fast, deterministic, no beam pruning
        viterbi_path = reconstruct_single_best_path_binned(
            alpha[i], ll_tensor[i], bin_centers, bin_edges,
            recomb_rate, state_defs, num_haps, switch_penalty_per_snp,
            snps_per_bin, hap_keys,
            double_recomb_factor=double_recomb_factor
        )
        
        painting = viterbi_path[0]  # reconstruct returns [SamplePainting(...)]
        painting.sample_index = global_idx
        
        # ----------------------------------------------------------------
        # Emission-aware short-chunk & wildcard cleanup (post-processing)
        # ----------------------------------------------------------------
        # Run AFTER Viterbi reconstruction, using the same per-bin LL
        # table the forward pass consumed.  The raw painter output is
        # stashed in `raw_chunks` so that the topology validator (which
        # wants to count W chunks and short residual chunks as
        # diagnostics) can still see what the painter emitted before
        # cleanup; downstream consumers (pedigree inference, phase
        # correction) read `chunks`, which is the cleaned list.
        #
        # The cleaner is a NO-OP on paintings with zero chunks (n_bins=0
        # edge case already returned above with empty chunks) or when
        # both min_chunk_bp == 0 and merge_wildcard is False / no W
        # chunks exist.
        if enable_emission_aware_cleanup and len(painting.chunks) > 0:
            raw_chunks_list = painting.chunks
            cleaned_chunks = merge_short_chunks_emission_aware(
                raw_chunks_list,
                min_chunk_bp=cleanup_min_chunk_bp,
                n_real_founders=n_real_founders_for_cleanup,
                bin_edges=bin_edges,
                binned_log_likelihoods=ll_tensor[i],
                state_definitions=state_defs,
                hap_keys=hap_keys,
                snps_per_bin=snps_per_bin,
                threshold_short_nats_per_snp=cleanup_threshold_short_nats_per_snp,
                threshold_wildcard_nats_per_snp=cleanup_threshold_wildcard_nats_per_snp,
                merge_wildcard=cleanup_merge_wildcard,
            )
            painting = SamplePainting(
                global_idx, cleaned_chunks, raw_chunks=raw_chunks_list)
        # If cleanup is disabled, the painting flows through untouched
        # and SamplePainting.raw_chunks defaults to the same list as
        # .chunks via the SamplePainting.__init__ default.
        
        results.append(painting)
        
    return results


def _init_persistent_paint_worker(total_cores=None, active_counter=None):
    """Initializer for persistent pool — sets up globals and dynamic threading."""
    global _PAINT_SHARED, _PAINT_SHM_REFS, _PAINT_CHROM_ID
    global _PAINT_ACTIVE_COUNTER, _PAINT_TOTAL_CORES
    _PAINT_SHARED = {}
    _PAINT_SHM_REFS = []
    _PAINT_CHROM_ID = None
    _PAINT_ACTIVE_COUNTER = active_counter
    _PAINT_TOTAL_CORES = total_cores
    # Cap numba threads to 1 initially — workers scale up dynamically
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass

def _load_shm_for_chromosome(chrom_id, meta):
    """
    Lazy-load SharedMemory for a new chromosome. Only re-loads when chrom_id changes.
    Called by workers on first task for each chromosome.
    """
    global _PAINT_SHARED, _PAINT_SHM_REFS, _PAINT_CHROM_ID
    
    if _PAINT_CHROM_ID == chrom_id:
        return  # Already loaded
    
    # Close old SharedMemory refs (from previous chromosome)
    for shm_ref in _PAINT_SHM_REFS:
        try:
            shm_ref.close()
        except Exception:
            pass
    _PAINT_SHM_REFS = []
    _PAINT_SHARED = {}
    
    # Open new SharedMemory blocks
    shm, arr = _array_from_shm(meta['samples_name'], meta['samples_shape'], meta['samples_dtype'])
    _PAINT_SHM_REFS.append(shm)
    _PAINT_SHARED['block_samples_data'] = arr
    
    shm, arr = _array_from_shm(meta['positions_name'], meta['positions_shape'], meta['positions_dtype'])
    _PAINT_SHM_REFS.append(shm)
    _PAINT_SHARED['positions'] = arr
    
    shm, arr = _array_from_shm(meta['haps_name'], meta['haps_shape'], meta['haps_dtype'])
    _PAINT_SHM_REFS.append(shm)
    hap_keys = meta['hap_keys']
    _PAINT_SHARED['hap_dict'] = {k: arr[i] for i, k in enumerate(hap_keys)}
    
    _PAINT_SHARED['params'] = meta['params']
    _PAINT_CHROM_ID = chrom_id


def _worker_paint_persistent(args):
    """
    Worker for persistent pool. Accepts (chrom_id, meta, indices, start_idx, end_idx).
    Lazy-loads SharedMemory when chromosome changes — meta is tiny (~500 bytes),
    so including it in each task adds negligible pickle overhead.
    
    Tracks active workers for dynamic thread scaling: straggler samples
    get more numba threads as peers finish.
    """
    chrom_id, meta, indices, start_idx, end_idx = args
    _load_shm_for_chromosome(chrom_id, meta)
    
    if _PAINT_ACTIVE_COUNTER is not None:
        with _PAINT_ACTIVE_COUNTER.get_lock():
            _PAINT_ACTIVE_COUNTER.value += 1
    
    try:
        return _worker_paint_batch_binned((indices, start_idx, end_idx))
    finally:
        if _PAINT_ACTIVE_COUNTER is not None:
            with _PAINT_ACTIVE_COUNTER.get_lock():
                _PAINT_ACTIVE_COUNTER.value -= 1


class PaintingPoolManager:
    """
    Persistent pool manager for painting multiple chromosomes efficiently.
    
    Creates the multiprocessing Pool ONCE and reuses it across chromosomes.
    SharedMemory is created per chromosome; workers lazy-initialize when they
    detect a new chromosome ID.
    
    Usage:
        with paint_samples.PaintingPoolManager(num_processes=112) as painter:
            for r_name in region_keys:
                result = painter.paint_chromosome(
                    block_result, sample_probs_matrix, sample_sites, ...
                )
    
    Saves ~10s per chromosome by avoiding repeated Pool creation/teardown.
    """
    
    def __init__(self, num_processes=16):
        self.num_processes = num_processes
        self._active_counter = _forkserver_ctx.Value('i', 0)
        self.pool = _forkserver_ctx.Pool(
            num_processes,
            initializer=_init_persistent_paint_worker,
            initargs=(num_processes, self._active_counter)
        )
        self._chrom_counter = 0
    
    def paint_chromosome(self, block_result, sample_probs_matrix, sample_sites,
                         recomb_rate=1e-8, switch_penalty_per_snp=1.0,
                         robustness_epsilon=1e-2, absolute_margin=5.0,
                         margin_per_snp=0.0, batch_size=1, 
                         max_active_paths=2000, double_recomb_factor=1.5,
                         snps_per_bin=100,
                         wildcard_per_snp_penalty=None,
                         enable_emission_aware_cleanup=True,
                         cleanup_min_chunk_bp=10000,
                         cleanup_threshold_short_nats_per_snp=0.02,
                         cleanup_threshold_wildcard_nats_per_snp=0.20,
                         cleanup_merge_wildcard=True):
        """
        Paint one chromosome using the persistent pool.
        
        Uses single Viterbi best path per sample (fast, deterministic).
        IBS ambiguity is handled downstream by ibs_painting.py at the
        pedigree inference stage, not by enumerating tolerance paths.

        switch_penalty_per_snp is the per-SNP anti-switch damping in nats.
        Per-bin switch cost is `switch_penalty_per_snp * snps_per_bin`
        (so at snps_per_bin=100, switch_penalty_per_snp=1.0 means
        100 nats per transition on top of the distance-based recomb
        prior).  This replaces the old fixed-per-bin switch_penalty,
        which made minimum-chunk-length depend on snps_per_bin.

        wildcard_per_snp_penalty (optional): if not None, extends the
        painting state space by one extra "wildcard" founder W (at
        index num_real_haps).  W has a CONSTANT per-SNP log-likelihood
        of -wildcard_per_snp_penalty, independent of sample data.  In
        regions where every real state's average per-SNP truth-LL is
        more negative than -wildcard_per_snp_penalty, the painter
        prefers to land on W rather than flicker between mediocre
        real states; this both robustifies the painting in chimeric
        regions and flags them via W-containing chunks for follow-up.
        Recommended starting value: 0.05 nats/SNP (calibrated for the
        chr3 chimera regime via diagnose_per_snp_ll_for_wildcard.py).
        None (default) disables the W slot entirely and preserves the
        previous behaviour.

        EMISSION-AWARE CLEANUP (post-Viterbi short-chunk and wildcard
        resolution).  After Viterbi reconstruction, each sample's
        painted chunks are passed through
        `merge_short_chunks_emission_aware` using the same per-bin LL
        table the forward pass consumed.  Short chunks below
        `cleanup_min_chunk_bp` and wildcard chunks (if
        `cleanup_merge_wildcard=True`) are candidates for removal;
        removal is gated by a per-SNP LL-worsening threshold
        (`cleanup_threshold_short_nats_per_snp` for short real chunks,
        `cleanup_threshold_wildcard_nats_per_snp` for wildcards).
        When two neighbours' ordered tuples differ, the optimal
        boundary position is chosen via dynamic programming over bin
        positions (replacing the geometric longer-neighbour heuristic
        in merge_short_chunks).

        The raw Viterbi-painted chunks are preserved in
        SamplePainting.raw_chunks so that the topology validator and
        any W-footprint diagnostic still has access to the
        pre-cleanup painting; downstream stages (pedigree inference,
        phase correction, allele-level reconstruction) consume
        SamplePainting.chunks, which is the cleaned list.

        Set `enable_emission_aware_cleanup=False` to disable cleanup
        entirely; in that case .chunks and .raw_chunks are the same
        list.

        Returns:
            BlockPainting with all samples' single Viterbi paths
            (cleaned chunks in .chunks, raw chunks in .raw_chunks).
        """
        self._chrom_counter += 1
        chrom_id = self._chrom_counter
        
        positions = block_result.positions
        n_sites_block = len(positions)
        
        n_bins = max(1, n_sites_block // snps_per_bin)
        
        block_samples_data = analysis_utils.get_sample_data_at_sites_multiple(
            sample_probs_matrix, sample_sites, positions
        )
        num_samples = block_samples_data.shape[0]
        
        num_tasks = math.ceil(num_samples / batch_size)
        actual_pool_size = min(num_tasks, self.num_processes)
        numba_threads = max(1, self.num_processes // max(actual_pool_size, 1))
        
        print(f"Viterbi Painting (BINNED) {num_samples} samples ({n_sites_block} SNPs → {n_bins} bins) "
              f"using {self.num_processes} workers...")
        if enable_emission_aware_cleanup:
            print(f"  Emission-aware cleanup ENABLED: "
                  f"min_chunk_bp={cleanup_min_chunk_bp}, "
                  f"thresh_short={cleanup_threshold_short_nats_per_snp} nats/SNP, "
                  f"thresh_W={cleanup_threshold_wildcard_nats_per_snp} nats/SNP, "
                  f"merge_W={cleanup_merge_wildcard}")
        else:
            print(f"  Emission-aware cleanup DISABLED (raw Viterbi chunks)")
        
        params = {
            'recomb_rate': recomb_rate,
            'switch_penalty_per_snp': switch_penalty_per_snp,
            'robustness_epsilon': robustness_epsilon,
            'double_recomb_factor': double_recomb_factor,
            'snps_per_bin': snps_per_bin,
            'numba_threads': numba_threads,
            'wildcard_per_snp_penalty': wildcard_per_snp_penalty,
            # Emission-aware cleanup config (consumed by
            # _worker_paint_batch_binned).  ON by default; downstream
            # stages assume .chunks is wildcard-free unless this is
            # disabled explicitly.
            'enable_emission_aware_cleanup': enable_emission_aware_cleanup,
            'cleanup_min_chunk_bp': cleanup_min_chunk_bp,
            'cleanup_threshold_short_nats_per_snp':
                cleanup_threshold_short_nats_per_snp,
            'cleanup_threshold_wildcard_nats_per_snp':
                cleanup_threshold_wildcard_nats_per_snp,
            'cleanup_merge_wildcard': cleanup_merge_wildcard,
        }
        
        # Create SharedMemory for this chromosome
        shm_blocks = []
        
        try:
            samples_c = np.ascontiguousarray(block_samples_data)
            shm_s, s_name, s_shape, s_dtype = _create_shm_from_array(samples_c)
            shm_blocks.append(shm_s)
            
            positions_arr = np.ascontiguousarray(np.array(positions, dtype=np.int64))
            shm_p, p_name, p_shape, p_dtype = _create_shm_from_array(positions_arr)
            shm_blocks.append(shm_p)
            
            hap_keys_list = sorted(block_result.haplotypes.keys())
            hap_arrays = [block_result.haplotypes[k] for k in hap_keys_list]
            hap_stack = np.ascontiguousarray(np.stack(hap_arrays))
            shm_h, h_name, h_shape, h_dtype = _create_shm_from_array(hap_stack)
            shm_blocks.append(shm_h)
            
            meta = {
                'samples_name': s_name, 'samples_shape': s_shape, 'samples_dtype': s_dtype,
                'positions_name': p_name, 'positions_shape': p_shape, 'positions_dtype': p_dtype,
                'haps_name': h_name, 'haps_shape': h_shape, 'haps_dtype': h_dtype,
                'hap_keys': hap_keys_list,
                'params': params,
            }
            
            # Tasks carry chrom_id + meta (meta is ~500 bytes, negligible)
            tasks = []
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                indices = list(range(start_idx, end_idx))
                tasks.append((chrom_id, meta, indices, start_idx, end_idx))
            
            all_sample_paintings = []
            for batch_result in tqdm(
                self.pool.imap_unordered(_worker_paint_persistent, tasks),
                total=len(tasks)
            ):
                all_sample_paintings.extend(batch_result)
        
        finally:
            for shm in shm_blocks:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
        
        all_sample_paintings.sort(key=lambda x: x.sample_index)
        range_tuple = (int(positions[0]), int(positions[-1]))
        return BlockPainting(range_tuple, all_sample_paintings)
    
    def close(self):
        """Terminate and join the persistent pool."""
        try:
            self.pool.terminate()
            self.pool.join()
        except Exception:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# =============================================================================
# 10. VISUALIZATIONS (unchanged)
# =============================================================================

def plot_painting_topology(block_painting, sample_idx=0, output_file=None):
    """Plot the topology of a sample's painting as a graph."""
    if not HAS_PLOTTING: return
    sample_obj = block_painting[sample_idx]
    chunks = sample_obj.chunks if hasattr(sample_obj, 'chunks') else []
    if not chunks: return

    G = nx.DiGraph()
    unique_pairs = set()
    for c in chunks: 
        unique_pairs.add(tuple(sorted((c.hap1, c.hap2))))
            
    sorted_pairs = sorted(list(unique_pairs))
    pair_to_y = {p: i for i, p in enumerate(sorted_pairs)}
    
    pos = {}
    
    for i, chunk in enumerate(chunks):
        pair = tuple(sorted((chunk.hap1, chunk.hap2)))
        pos_x = (chunk.start + chunk.end) / 2
        node_id = (chunk.start, pair[0], pair[1])
        G.add_node(node_id, label=f"{pair[0]}/{pair[1]}")
        pos[node_id] = (pos_x, pair_to_y[pair])
        if i > 0:
            prev_c = chunks[i-1]
            prev_p = tuple(sorted((prev_c.hap1, prev_c.hap2)))
            prev_node = (prev_c.start, prev_p[0], prev_p[1])
            G.add_edge(prev_node, node_id)

    fig, ax = plt.subplots(figsize=(15, max(4, len(sorted_pairs))))
    for pair, y in pair_to_y.items():
        ax.axhline(y, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.text(chunks[0].start, y, f" {pair[0]}/{pair[1]}", va='center', fontsize=9, fontweight='bold')

    nx.draw_networkx_nodes(G, pos, node_color='#aaccff', node_size=100, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.0, ax=ax)
    
    ax.set_title(f"Painting Topology — Sample {sample_idx} ({len(chunks)} chunks)", fontsize=14)
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_yticks([])
    if output_file: plt.savefig(output_file, bbox_inches='tight')
    else: plt.show()
    plt.close()

def plot_viable_paintings(block_painting, sample_idx=0, max_paths=50, output_file=None):
    """Plot the painting for a sample."""
    if not HAS_PLOTTING: return
    sample_obj = block_painting[sample_idx]
    chunks = sample_obj.chunks if hasattr(sample_obj, 'chunks') else []
    if not chunks: return

    unique_haps = set()
    for c in chunks: 
        unique_haps.add(c.hap1)
        unique_haps.add(c.hap2)
    sorted_haps = sorted(list(unique_haps))
    hap_to_idx = {h: i for i, h in enumerate(sorted_haps)}
    
    if len(sorted_haps) <= 10: palette = sns.color_palette("tab10", len(sorted_haps))
    else: palette = sns.color_palette("husl", len(sorted_haps))

    row_height = 0.5
    y_height = 0.4 
    fig, ax = plt.subplots(figsize=(20, 2.0))
    
    # Draw painting (two tracks: hap1 bottom, hap2 top)
    y_base = 0
    for chunk in chunks:
        width = chunk.end - chunk.start
        if width <= 0: continue
        c1 = palette[hap_to_idx[chunk.hap1]]
        rect1 = mpatches.Rectangle((chunk.start, y_base), width, y_height/2, facecolor=c1, edgecolor='none')
        ax.add_patch(rect1)
        c2 = palette[hap_to_idx[chunk.hap2]]
        rect2 = mpatches.Rectangle((chunk.start, y_base + y_height/2), width, y_height/2, facecolor=c2, edgecolor='none')
        ax.add_patch(rect2)

    ax.set_xlim(block_painting.start_pos, block_painting.end_pos)
    ax.set_ylim(-0.1, row_height + 0.1)
    ax.set_yticks([row_height/2])
    ax.set_yticklabels([f"Sample {sample_idx}"], fontsize=8)
    ax.set_xlabel("Genomic Position (bp)")
    
    patches = [mpatches.Patch(color=palette[hap_to_idx[h]], label=f"H{h}") for h in sorted_haps]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    if output_file: plt.savefig(output_file)
    else: plt.show()
    plt.close()

def plot_population_painting(block_painting, output_file=None, 
                             title="Population Painting", 
                             figsize_width=20, 
                             row_height_per_sample=0.25,
                             show_labels=True,
                             sample_names=None):
    """Plot paintings for all samples in a population view."""
    if not HAS_PLOTTING:
        print("Error: Matplotlib/Seaborn not installed.")
        return

    unique_haps = set()
    for sample in block_painting:
        for chunk in sample:
            if chunk.hap1 != -1: unique_haps.add(chunk.hap1)
            if chunk.hap2 != -1: unique_haps.add(chunk.hap2)
    
    sorted_haps = sorted(list(unique_haps))
    hap_to_idx = {h: i for i, h in enumerate(sorted_haps)}
    
    if len(sorted_haps) <= 10: palette = sns.color_palette("tab10", len(sorted_haps))
    elif len(sorted_haps) <= 20: palette = sns.color_palette("tab20", len(sorted_haps))
    else: palette = sns.color_palette("husl", len(sorted_haps))
        
    num_samples = len(block_painting)
    header_space = 2.0 
    calc_height = (num_samples * row_height_per_sample) + header_space
    if calc_height < 6: calc_height = 6
    if calc_height > 300: calc_height = 300
    
    figsize = (figsize_width, calc_height)
    fig, ax = plt.subplots(figsize=figsize)
    
    y_height = 0.8 
    
    for i, sample in enumerate(block_painting):
        y_base = i 
        for chunk in sample:
            width = chunk.end - chunk.start
            if width <= 0: continue
            
            if chunk.hap1 != -1:
                color1 = palette[hap_to_idx[chunk.hap1]]
                rect1 = mpatches.Rectangle((chunk.start, y_base), width, y_height/2, facecolor=color1, edgecolor='none')
                ax.add_patch(rect1)
            
            if chunk.hap2 != -1:
                color2 = palette[hap_to_idx[chunk.hap2]]
                rect2 = mpatches.Rectangle((chunk.start, y_base + y_height/2), width, y_height/2, facecolor=color2, edgecolor='none')
                ax.add_patch(rect2)
            
    ax.set_xlim(block_painting.start_pos, block_painting.end_pos)
    ax.set_ylim(-0.5, len(block_painting) + 0.5)
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    
    if show_labels:
        if sample_names and len(sample_names) == len(block_painting): labels = sample_names
        else: labels = [f"S{s.sample_index}" for s in block_painting]
        ax.set_yticks(np.arange(len(block_painting)) + y_height/2)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        ax.set_yticks([])

    legend_patches = []
    for h_key in sorted_haps:
        c = palette[hap_to_idx[h_key]]
        legend_patches.append(mpatches.Patch(color=c, label=f"Founder {h_key}"))
        
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    if output_file:
        dpi = 100 if calc_height > 100 else 150
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


# =============================================================================
# 11. IBS-AWARE PEDIGREE SUPPORT
# =============================================================================
#
# These functions convert SamplePainting objects into allele grids with
# IBS-aware homozygosity masks for the pedigree HMM in pedigree_inference.py.
#
# The pedigree HMM compares allele values (0/1), not founder IDs. When two
# founders are IBS (identical-by-state) at a genomic region, they carry the
# same alleles — the HMM literally cannot distinguish them. The hom mask
# captures this: if alleles on track 1 == alleles on track 2 at a bin,
# the HMM allows free phase switches there (correct, since the data cannot
# resolve which parental chromosome contributed which track).
#
# This replaces the old multi-consensus tolerance painting approach which
# enumerated hundreds of paths differing only in IBS regions — all producing
# identical allele grids — then evaluated M² HMM runs per parent-child pair.
# =============================================================================

def convert_id_grid_to_allele_grid_multisnp(id_grid, bin_centers, founder_block,
                                             bin_width_bp=None, max_snps_per_bin=10):
    """
    Convert founder ID grid to allele grid with multiple SNPs per bin.
    
    For each bin, samples up to max_snps_per_bin SNPs and looks up the
    actual allele (0/1) for the assigned founder.
    
    Returns:
        allele_grid: (num_bins, 2, max_snps_per_bin) int8
    """
    num_bins = id_grid.shape[0]
    snp_positions = founder_block.positions
    n_snps = len(snp_positions)
    allele_grid = np.full((num_bins, 2, max_snps_per_bin), -1, dtype=np.int8)
    
    if n_snps == 0:
        return allele_grid
    
    if bin_width_bp is None:
        if len(bin_centers) > 1:
            bin_width_bp = bin_centers[1] - bin_centers[0]
        else:
            bin_width_bp = 10000
    
    half_width = bin_width_bp / 2.0
    
    # Build founder allele lookup
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    max_id = max(hap_keys) if hap_keys else 0
    founder_alleles = np.full((max_id + 1, n_snps), -1, dtype=np.int8)
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2:
            founder_alleles[fid, :] = np.argmax(h_arr, axis=1).astype(np.int8)
        else:
            founder_alleles[fid, :] = h_arr.astype(np.int8)
    
    # Assign alleles to bins
    bin_starts = bin_centers - half_width
    bin_ends = bin_centers + half_width
    start_indices = np.searchsorted(snp_positions, bin_starts, side='left')
    end_indices = np.searchsorted(snp_positions, bin_ends, side='right')
    
    for b in range(num_bins):
        s_start = start_indices[b]
        s_end = end_indices[b]
        bin_n_snps = s_end - s_start
        if bin_n_snps == 0:
            continue
        
        if bin_n_snps <= max_snps_per_bin:
            sampled_indices = list(range(s_start, s_end))
        else:
            step = bin_n_snps / max_snps_per_bin
            sampled_indices = [s_start + int(i * step) for i in range(max_snps_per_bin)]
        
        f0 = id_grid[b, 0]
        f1 = id_grid[b, 1]
        # NOTE: founder_alleles has shape (num_real_haps, n_snps) -- it does
        # NOT include the wildcard W slot.  When the painter has assigned
        # W to a track (id = num_real_haps = W_idx, one past the real
        # range), the existing `if f0 >= 0` check would let the lookup
        # proceed and IndexError on founder_alleles[W_idx, ...].  The
        # bounds check below treats BOTH negative ids (unfilled bins,
        # the historical "no painted founder here" sentinel) AND
        # too-large ids (W) as missing-data positions, leaving the
        # allele_grid entry at -1.  Downstream HMM scoring functions
        # (run_phase_agnostic_hmm{_multisnp}, run_trio_phase_aware_hmm
        # {_multisnp} in pedigree_inference.py) already skip -1 entries
        # as missing, so the W positions naturally drop out of the
        # parentage scoring on each contig -- which is the correct
        # treatment, since W denotes "no real founder fits here" and
        # therefore carries NO information about parentage.
        n_real_founders = founder_alleles.shape[0]
        for k_idx, snp_idx in enumerate(sampled_indices):
            if k_idx >= max_snps_per_bin:
                break
            if 0 <= f0 < n_real_founders:
                allele_grid[b, 0, k_idx] = founder_alleles[f0, snp_idx]
            if 0 <= f1 < n_real_founders:
                allele_grid[b, 1, k_idx] = founder_alleles[f1, snp_idx]
    
    return allele_grid


def convert_id_grid_to_allele_grid(id_grid, bin_centers, founder_block, bin_width_bp=None):
    """
    Convert founder ID grid to single-SNP allele grid.
    
    Returns:
        allele_grid: (num_bins, 2) int8
    """
    num_bins = id_grid.shape[0]
    bin_indices = np.searchsorted(founder_block.positions, bin_centers)
    bin_indices = np.clip(bin_indices, 0, len(founder_block.positions) - 1)
    
    if bin_width_bp is None:
        if len(bin_centers) > 1:
            bin_width_bp = bin_centers[1] - bin_centers[0]
        else:
            bin_width_bp = 10000
    
    found_snps_pos = founder_block.positions[bin_indices]
    dist_to_center = np.abs(found_snps_pos - bin_centers)
    valid_snp_mask = dist_to_center <= (bin_width_bp / 2.0)
    
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    max_id = max(hap_keys) if hap_keys else 0
    allele_lookup = np.full((max_id + 1, num_bins), -1, dtype=np.int8)
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2:
            raw_alleles = np.argmax(h_arr, axis=1)
        else:
            raw_alleles = h_arr
        extracted = raw_alleles[bin_indices]
        extracted[~valid_snp_mask] = -1
        allele_lookup[fid, :] = extracted
    
    allele_grid = np.full_like(id_grid, -1, dtype=np.int8)
    b_indices = np.arange(num_bins)
    # NOTE: allele_lookup has shape (num_real_haps, num_bins).  Bin-painted
    # ids in {0, ..., num_real_haps - 1} are real founders.  The painter
    # may additionally emit W_idx = num_real_haps when the wildcard slot
    # is enabled (see calculate_binned_emissions); these IDs lie OUTSIDE
    # the allele_lookup index range and must be treated as missing data
    # (the same way the historical -1 sentinel for "no painted founder"
    # is treated).  See the matching note in
    # convert_id_grid_to_allele_grid_multisnp for the full rationale
    # and the downstream scoring code's -1 handling.
    n_real_founders = allele_lookup.shape[0]
    for chrom in [0, 1]:
        ids = id_grid[:, chrom]
        valid_mask = (ids >= 0) & (ids < n_real_founders)
        safe_ids = ids.copy()
        safe_ids[~valid_mask] = 0
        alleles = allele_lookup[safe_ids, b_indices]
        alleles[~valid_mask] = -1
        allele_grid[:, chrom] = alleles
    
    return allele_grid


def compute_ibs_hom_mask(allele_grid):
    """
    Derive homozygosity mask from allele identity across tracks.
    
    A bin is marked as potentially homozygous (phase-ambiguous) if the
    alleles on track 1 and track 2 are identical at ALL SNPs in the bin.
    This captures both:
      - True homozygosity (same founder on both tracks)
      - Effective homozygosity from IBS (different founders, same alleles)
    
    When the HMM sees a homozygous bin, it allows free phase switches,
    which is correct because the data cannot resolve which parental
    chromosome contributed which track.
    
    Args:
        allele_grid: (num_bins, 2, max_snps_per_bin) int8 for multi-SNP,
                     or (num_bins, 2) int8 for single-SNP
    
    Returns:
        hom_mask: (num_bins,) bool
    """
    if allele_grid.ndim == 3:
        # Multi-SNP: check all SNPs in each bin
        num_bins = allele_grid.shape[0]
        hom_mask = np.ones(num_bins, dtype=np.bool_)
        for b in range(num_bins):
            a0 = allele_grid[b, 0, :]  # track 1 alleles
            a1 = allele_grid[b, 1, :]  # track 2 alleles
            # Valid SNPs: both tracks have data
            valid = (a0 != -1) & (a1 != -1)
            if not np.any(valid):
                # No valid SNPs → treat as homozygous (no information)
                hom_mask[b] = True
            else:
                # Homozygous if ALL valid SNPs match
                hom_mask[b] = np.all(a0[valid] == a1[valid])
    else:
        # Single-SNP
        hom_mask = ((allele_grid[:, 0] == allele_grid[:, 1]) |
                    (allele_grid[:, 0] == -1) |
                    (allele_grid[:, 1] == -1))
    
    return hom_mask


def process_contig_for_pedigree(contig_idx, sample_start, sample_end,
                                 painting, founder_block,
                                 snps_per_bin, recomb_rate,
                                 max_snps_per_bin=10):
    """
    Process a contig's painting into allele grids with IBS-aware hom masks
    for pedigree inference.
    
    Called by _process_contig_batch in pedigree_inference.py.
    Accesses SamplePainting.chunks directly, derives the homozygosity mask
    from allele identity instead of consensus set overlap.
    
    Returns:
        dict with keys: contig_idx, sample_start, sample_end,
             sample_allele_grids, sw_costs, st_costs, num_bins, switch_counts
    """
    start_pos = painting.start_pos
    end_pos = painting.end_pos
    total_len = end_pos - start_pos
    approx_bp_per_bin = snps_per_bin * 100
    num_bins = int(total_len / approx_bp_per_bin)
    if num_bins < 100:
        num_bins = 100
    
    bin_edges = np.linspace(start_pos, end_pos, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_width = 10000.0
    if num_bins > 1:
        bin_width = bin_centers[1] - bin_centers[0]
    
    sample_allele_grids = []
    switch_counts = []
    
    for i in range(sample_start, sample_end):
        sample_obj = painting[i]
        
        # Access chunks directly — painting returns SamplePainting objects
        chunks = sample_obj.chunks if sample_obj.chunks else None
        
        # Discretize to founder ID grid
        id_grid = np.full((num_bins, 2), -1, dtype=np.int32)
        if chunks:
            c_ends = np.array([c.end for c in chunks])
            c_h1 = np.array([c.hap1 for c in chunks])
            c_h2 = np.array([c.hap2 for c in chunks])
            c_starts = np.array([c.start for c in chunks])
            indices = np.searchsorted(c_ends, bin_centers)
            indices = np.clip(indices, 0, len(chunks) - 1)
            valid_mask = bin_centers >= c_starts[indices]
            id_grid[:, 0] = np.where(valid_mask, c_h1[indices], -1)
            id_grid[:, 1] = np.where(valid_mask, c_h2[indices], -1)
        
        # Convert to allele grid
        if max_snps_per_bin > 1:
            allele_grid = convert_id_grid_to_allele_grid_multisnp(
                id_grid, bin_centers, founder_block, bin_width, max_snps_per_bin
            )
        else:
            allele_grid = convert_id_grid_to_allele_grid(
                id_grid, bin_centers, founder_block, bin_width
            )
        
        # Derive hom mask from allele identity (IBS-aware)
        hom_mask = compute_ibs_hom_mask(allele_grid)
        
        # Single entry per sample (no consensus combinations needed)
        sample_allele_grids.append([(allele_grid, hom_mask, 1.0)])
        
        # Switch counts
        switches = ((id_grid[:-1, :] != id_grid[1:, :]) &
                    (id_grid[:-1, :] != -1) & (id_grid[1:, :] != -1))
        switch_counts.append(np.sum(switches))
    
    # Transition costs
    dists = np.zeros(num_bins)
    dists[1:] = np.diff(bin_centers)
    theta = np.clip(1.0 - np.exp(-dists * recomb_rate), 1e-15, 0.5)
    sw_costs = np.log(theta)
    st_costs = np.log(1.0 - theta)
    
    return {
        'contig_idx': contig_idx,
        'sample_start': sample_start,
        'sample_end': sample_end,
        'sample_allele_grids': sample_allele_grids,
        'sw_costs': sw_costs,
        'st_costs': st_costs,
        'num_bins': num_bins,
        'switch_counts': np.array(switch_counts),
    }


def precompute_founder_ibs(founder_block, snps_per_bin=100):
    """
    Precompute pairwise IBS between all founder haplotypes, per bin.
    
    This is provided for diagnostic/visualization purposes. The main
    pipeline doesn't need it — the allele-level hom mask in compute_ibs_hom_mask
    implicitly captures IBS.
    
    Returns:
        ibs_matrix: (n_haps, n_haps, n_bins) bool — True if founders i,j
                     have identical alleles at all SNPs in bin b
        bin_centers: (n_bins,) float64
        hap_keys: sorted list of founder IDs
    """
    positions = founder_block.positions
    n_snps = len(positions)
    hap_keys = sorted(list(founder_block.haplotypes.keys()))
    n_haps = len(hap_keys)
    
    # Build allele matrix (n_haps, n_snps)
    max_id = max(hap_keys) if hap_keys else 0
    alleles = np.full((max_id + 1, n_snps), -1, dtype=np.int8)
    for fid, h_arr in founder_block.haplotypes.items():
        if h_arr.ndim == 2:
            alleles[fid, :] = np.argmax(h_arr, axis=1).astype(np.int8)
        else:
            alleles[fid, :] = h_arr.astype(np.int8)
    
    # Bin structure
    num_bins = max(1, n_snps // snps_per_bin)
    if num_bins < 100 and n_snps >= 100:
        num_bins = 100
    
    bin_edges = np.linspace(positions[0], positions[-1], num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Per-bin IBS check
    ibs_matrix = np.ones((max_id + 1, max_id + 1, num_bins), dtype=np.bool_)
    
    snp_bin_idx = np.searchsorted(bin_edges[1:], positions)
    snp_bin_idx = np.clip(snp_bin_idx, 0, num_bins - 1)
    
    for b in range(num_bins):
        snp_mask = snp_bin_idx == b
        if not np.any(snp_mask):
            continue
        bin_alleles = alleles[:, snp_mask]  # (n_haps, n_snps_in_bin)
        for i in range(n_haps):
            fi = hap_keys[i]
            for j in range(i + 1, n_haps):
                fj = hap_keys[j]
                is_ibs = np.all(bin_alleles[fi] == bin_alleles[fj])
                ibs_matrix[fi, fj, b] = is_ibs
                ibs_matrix[fj, fi, b] = is_ibs
    
    return ibs_matrix, bin_centers, hap_keys


def summarize_ibs_regions(ibs_matrix, bin_centers, hap_keys):
    """
    Print a summary of IBS regions between all founder pairs.
    Useful for understanding how much IBS-driven ambiguity exists.
    """
    n_bins = len(bin_centers)
    genome_len = bin_centers[-1] - bin_centers[0] if n_bins > 1 else 0
    bin_width = genome_len / max(n_bins - 1, 1)
    
    print(f"IBS Summary: {len(hap_keys)} founders, {n_bins} bins, "
          f"{genome_len/1e6:.1f} Mb")
    print(f"{'Pair':>10s}  {'IBS bins':>10s}  {'IBS fraction':>12s}  {'IBS Mb':>8s}")
    
    for i, fi in enumerate(hap_keys):
        for j, fj in enumerate(hap_keys):
            if j <= i:
                continue
            n_ibs = np.sum(ibs_matrix[fi, fj, :])
            frac = n_ibs / n_bins if n_bins > 0 else 0
            ibs_mb = n_ibs * bin_width / 1e6
            print(f"  ({fi},{fj}):  {n_ibs:10d}  {frac:12.3f}  {ibs_mb:8.1f}")