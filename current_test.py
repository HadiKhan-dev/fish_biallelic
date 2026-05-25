"""
chr1 L1 -> L2 hierarchical assembly: timing + ground-truth validation.

This is the L2 analogue of current_test.py, focused on the pipeline step
that takes L1 super-blocks (the output of stage 6: 06_assembly_L1) and
links them into L2 super-blocks via `hierarchical_assembly.run_-
hierarchical_step` with HMM linking enabled — exactly the call made by
pipeline.py's STAGE_7.

Loads from existing pipeline checkpoints:
  .pipeline_checkpoints/06_assembly_L1/chr1.pkl
        -> super_blocks_L1  (a block_haplotypes.BlockResults)
        Input to the L2 assembly step.

  .pipeline_checkpoints/02_simulation/chr1.pkl
        -> simd_probs       (global_probs: contig-wide (N, n_total_sites, 3))
        Sample-level genotype posteriors that the L2 step reads from
        shared memory.

  .pipeline_checkpoints/01_vcf_discovery/chr1.pkl
        -> naive_long_haps  ((global_sites, haps_data))
        global_sites is the contig-wide SNP position array; haps_data is
        the K=6 probabilistic founder haps from which the simulation
        sampled — these are the GROUND TRUTH for validation.

For each L2 super-block produced, we compare its discovered haplotypes
against the K=6 truth founders' bit patterns at the super-block's SNP
positions, computing hamming-% over positions that are (a) kept by the
super-block (super_block.keep_flags, which is the concatenation of the
constituent L1 blocks' keep_flags) and (b) NOT wildcard in the
discovered hap (the (0.5, 0.5) prob-pair convention).

Reports (mirroring current_test.py's format so the two are directly
comparable):
  - Assembly timing (STAGE_7 equivalent wall-clock)
  - haps/L2-super-block summary
  - Total RECALL: fraction of (super-block, truth-founder) pairs whose
    best discovered hap is within <= RECOVERY_THRESHOLD_PCT hamming.
  - Total PRECISION: fraction of (super-block, disc-hap) pairs whose
    best truth founder is within the same threshold.
  - Per-block recall distribution (100%, 80-99%, 50-79%, <50%).
  - Best-disc-match hamming-% distribution across all (block, truth)
    pairs.
  - Worst-recall L2 super-blocks for inspection.

The entry-script-name caveat from hierarchical_assembly.py applies:
this file is intentionally NOT named main.py so that forkserver workers
do not re-execute it.  See `run_hierarchical_step`'s docstring for the
rationale.
"""

import os
import time
import pickle
import numpy as np

import hierarchical_assembly
# Import block_haplotypes so the unpickled BlockResults/BlockResult
# classes resolve.  Side effect of import: registers the class names
# in sys.modules so pickle.load can find them.
import block_haplotypes  # noqa: F401

# ---------------------------------------------------------------------------
# Config — matches pipeline.py STAGE_7's run_hierarchical_step call.
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = '.pipeline_checkpoints'
CONTIG = 'chr1'

# Total cores available to the L2 step.  Mirrors pipeline.py's n_processes
# (which is typically os.cpu_count() / the scheduler's allocation).
N_PROCESSES = 112

# Worker-recycling cadence from pipeline.py (WORKER_MAXTASKS = 1).  Set
# to 1 to prevent glibc fragmentation memory accumulation across batches.
WORKER_MAXTASKS = 1

# Recovery threshold for the validation step — same convention as
# current_test.py.  A discovered L2 hap "matches" a truth founder iff
# their hamming-% over (kept AND non-wildcard) positions is <= this.
RECOVERY_THRESHOLD_PCT = 1.0

# STAGE_7's exact run_hierarchical_step parameters (verbatim from
# pipeline.py:904-918).  Keep these in sync with the pipeline if it
# changes upstream.
L2_PARAMS = dict(
    batch_size=10,
    use_hmm_linking=True,
    recomb_rate=5e-8,
    beam_width=200,
    max_founders=12,
    cc_scale=0.5,
    n_generations=3,
    verbose=False,
)


# ---------------------------------------------------------------------------
# Truth helpers (copied verbatim from current_test.py so the two scripts
# share the same validation semantics without coupling via import — this
# script is self-contained for portability).
# ---------------------------------------------------------------------------

def _concretify_haps(haps_list):
    """Mirror of simulate_sequences.concretify_haps without the heavy
    import.  Each prob-hap (n_sites, 2) -> (n_sites,) int8 bit array
    via argmax along the allele axis."""
    return [np.argmax(h, axis=1).astype(np.int8) for h in haps_list]


def load_truth(checkpoint_dir, contig):
    """Returns (contig_sites, truth_haploid_bits) where truth_haploid_-
    bits is a list of K (n_total_sites,) int8 arrays — the founder bit
    patterns over the full contig."""
    path = os.path.join(checkpoint_dir, '01_vcf_discovery', f'{contig}.pkl')
    with open(path, 'rb') as f:
        d = pickle.load(f)
    sites, haps_data = d['naive_long_haps']
    return np.asarray(sites), _concretify_haps(haps_data)


def load_global_probs(checkpoint_dir, contig):
    """Returns the contig-wide (N, n_total_sites, 3) genotype-posterior
    tensor used by the L2 assembly step (placed in shared memory by
    run_hierarchical_step)."""
    path = os.path.join(checkpoint_dir, '02_simulation', f'{contig}.pkl')
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['simd_probs']


def load_super_blocks_L1(checkpoint_dir, contig):
    """Returns the L1 BlockResults (output of stage 6)."""
    path = os.path.join(checkpoint_dir, '06_assembly_L1', f'{contig}.pkl')
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['super_blocks_L1']


# ---------------------------------------------------------------------------
# Per-block validation (lifted from current_test.py — same logic, applies
# unchanged to L2 super-blocks because BlockResult exposes the same
# (positions, haplotypes, keep_flags) interface for L1 200-SNP blocks
# and L2 multi-thousand-SNP super-blocks).
# ---------------------------------------------------------------------------

def _block_true_haps(positions, contig_sites, truth_haploid_bits):
    """At the block's SNP positions, gather each founder's truth bits.

    positions: (n_block_sites,) — block's SNP positions (sorted)
    contig_sites: (n_total_sites,) — full-contig SNP positions (sorted)
    truth_haploid_bits: list of K (n_total_sites,) int arrays

    Returns: (K, n_block_sites) int8.  Raises if any block position
    fails to map exactly into contig_sites — that would indicate the
    truth and the block data are from different VCF runs.
    """
    idx = np.searchsorted(contig_sites, positions)
    if idx.max() >= len(contig_sites) or not np.array_equal(
            contig_sites[idx], positions):
        raise ValueError(
            "Block positions do not all map into contig_sites; the "
            "ground-truth file and the simulation checkpoint appear to "
            "be from different runs.")
    K = len(truth_haploid_bits)
    out = np.empty((K, len(positions)), dtype=np.int8)
    for k in range(K):
        out[k] = truth_haploid_bits[k][idx]
    return out


def _disc_haps_bits(block_haplotypes_dict, n_block_sites):
    """Extract (bits, wildcard_mask) from a block's haplotypes dict.

    The dict has values of shape (n_block_sites, 2): (1.0, 0.0) -> bit 0,
    (0.0, 1.0) -> bit 1, (0.5, 0.5) -> wildcard (no information).
    """
    hap_ids = list(block_haplotypes_dict.keys())
    K = len(hap_ids)
    bits = np.empty((K, n_block_sites), dtype=np.int8)
    wc = np.empty((K, n_block_sites), dtype=bool)
    for i, hid in enumerate(hap_ids):
        prob = block_haplotypes_dict[hid]
        bits[i] = np.argmax(prob, axis=1)
        # Wildcard: both probabilities are exactly 0.5 (the legacy
        # convention).  Any other case is informative.
        wc[i] = (prob[:, 0] == 0.5) & (prob[:, 1] == 0.5)
    return bits, wc


def _hamming_pct(disc_bit, disc_wc, true_bit, kept_mask):
    """Hamming% over (kept AND non-wildcard) positions; -1 if no
    comparable sites."""
    valid = kept_mask & (~disc_wc)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return -1.0
    return 100.0 * float(np.sum(disc_bit[valid] != true_bit[valid])) / n_valid


def validate_block(block, contig_sites, truth_haploid_bits, threshold_pct):
    """Per-block validation stats.

    Returns a dict with:
        K_true         — number of truth founders (= len(truth_haploid_-
                          bits); same for every block)
        K_disc         — number of discovered haps
        n_kept_sites   — block.keep_flags.sum()
        n_truth_recovered — count of truth founders whose best-discovered
                            match has hamming-% <= threshold_pct
        n_disc_matched    — count of discovered haps whose best-truth
                            match has hamming-% <= threshold_pct
        truth_min_hammings — list of len K_true (best disc hamming per truth)
        disc_min_hammings  — list of len K_disc (best truth hamming per disc)
    """
    positions = np.asarray(block.positions)
    n_block_sites = len(positions)
    if block.keep_flags is not None:
        kept_mask = np.asarray(block.keep_flags, dtype=bool)
    else:
        kept_mask = np.ones(n_block_sites, dtype=bool)
    n_kept = int(kept_mask.sum())

    K_true = len(truth_haploid_bits)
    K_disc = len(block.haplotypes)

    # Empty discoveries / empty kept-mask -> nothing comparable
    if K_disc == 0 or n_kept == 0:
        return {
            'K_true': K_true,
            'K_disc': K_disc,
            'n_kept_sites': n_kept,
            'n_block_sites': n_block_sites,
            'n_truth_recovered': 0,
            'n_disc_matched': 0,
            'truth_min_hammings': [float('inf')] * K_true,
            'disc_min_hammings': [],
        }

    true_haps = _block_true_haps(positions, contig_sites, truth_haploid_bits)
    disc_bits, disc_wc = _disc_haps_bits(block.haplotypes, n_block_sites)

    # Pairwise hamming-% matrix
    ham = np.empty((K_true, K_disc), dtype=np.float64)
    for ti in range(K_true):
        for di in range(K_disc):
            ham[ti, di] = _hamming_pct(
                disc_bits[di], disc_wc[di], true_haps[ti], kept_mask)
    # Sentinel -1.0 (no comparable sites) -> +inf so min reductions ignore it
    ham_clean = np.where(ham < 0.0, np.inf, ham)
    truth_min = ham_clean.min(axis=1)
    disc_min = ham_clean.min(axis=0)

    n_truth_recovered = int(np.sum(truth_min <= threshold_pct))
    n_disc_matched = int(np.sum(disc_min <= threshold_pct))

    return {
        'K_true': K_true,
        'K_disc': K_disc,
        'n_kept_sites': n_kept,
        'n_block_sites': n_block_sites,
        'n_truth_recovered': n_truth_recovered,
        'n_disc_matched': n_disc_matched,
        'truth_min_hammings': truth_min.tolist(),
        'disc_min_hammings': disc_min.tolist(),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print(f"L1 -> L2 hierarchical assembly + validation on {CONTIG}")
    print(f"  checkpoint_dir:      {CHECKPOINT_DIR}")
    print(f"  n_processes:         {N_PROCESSES}")
    print(f"  WORKER_MAXTASKS:     {WORKER_MAXTASKS}")
    print(f"  recovery threshold:  <= {RECOVERY_THRESHOLD_PCT:.2f}% hamming")
    print(f"  L2 params:           {L2_PARAMS}")
    print("=" * 78)

    # -----------------------------------------------------------------------
    # Load inputs
    # -----------------------------------------------------------------------
    t0 = time.time()
    super_blocks_L1 = load_super_blocks_L1(CHECKPOINT_DIR, CONTIG)
    global_probs = load_global_probs(CHECKPOINT_DIR, CONTIG)
    contig_sites, truth_haploid = load_truth(CHECKPOINT_DIR, CONTIG)
    load_time = time.time() - t0

    # global_sites for the assembly step (the same array as contig_sites
    # — they both come from naive_long_haps[0]).
    global_sites = contig_sites

    n_L1 = len(super_blocks_L1)
    n_L1_haps = [len(b.haplotypes) for b in super_blocks_L1]
    print(f"\nLoaded checkpoints in {load_time:.1f}s")
    print(f"  L1 super-blocks (input):  {n_L1}")
    print(f"  L1 haps/block:            min={min(n_L1_haps)}, "
          f"max={max(n_L1_haps)}, mean={np.mean(n_L1_haps):.2f}")
    print(f"  global_probs shape:       {global_probs.shape}, dtype={global_probs.dtype}")
    print(f"  K_truth:                  {len(truth_haploid)}")
    print(f"  n_contig_sites:           {len(contig_sites)}")

    # -----------------------------------------------------------------------
    # Run L1 -> L2 hierarchical assembly (STAGE_7 equivalent)
    # -----------------------------------------------------------------------
    print(f"\nRunning hierarchical_assembly.run_hierarchical_step "
          f"(L1 -> L2, n_processes={N_PROCESSES}) ...")
    t0 = time.time()
    super_blocks_L2 = hierarchical_assembly.run_hierarchical_step(
        super_blocks_L1,
        global_probs,
        global_sites,
        num_processes=N_PROCESSES,
        maxtasksperchild=WORKER_MAXTASKS,
        **L2_PARAMS,
    )
    assembly_time = time.time() - t0

    n_L2 = len(super_blocks_L2)
    n_L2_haps = [len(b.haplotypes) for b in super_blocks_L2]
    n_L2_sites = [len(b.positions) for b in super_blocks_L2]
    print()
    print("=" * 78)
    print("L2 ASSEMBLY RESULT")
    print("=" * 78)
    print(f"  contig:                  {CONTIG}")
    print(f"  n_L1_super_blocks:       {n_L1}")
    print(f"  n_L2_super_blocks:       {n_L2}")
    print(f"  L2 haps/super-block:     min={min(n_L2_haps)}, "
          f"max={max(n_L2_haps)}, mean={np.mean(n_L2_haps):.2f}")
    print(f"  L2 sites/super-block:    min={min(n_L2_sites)}, "
          f"max={max(n_L2_sites)}, mean={np.mean(n_L2_sites):.0f}")
    print(f"  total sites covered:     {sum(n_L2_sites)}")
    print(f"  assembly time:           {assembly_time:.2f} s "
          f"({assembly_time/60:.2f} min)")

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    print()
    print("=" * 78)
    print("VALIDATION (L2 discovered haps vs ground-truth founders)")
    print("=" * 78)

    t0 = time.time()
    stats = []
    for b in super_blocks_L2:
        stats.append(validate_block(
            b, contig_sites, truth_haploid, RECOVERY_THRESHOLD_PCT))
    val_time = time.time() - t0
    print(f"Validation took {val_time:.1f}s")

    # Aggregates
    n_blocks = len(stats)
    total_truth = sum(s['K_true'] for s in stats)
    total_recov = sum(s['n_truth_recovered'] for s in stats)
    total_disc  = sum(s['K_disc'] for s in stats)
    total_matched = sum(s['n_disc_matched'] for s in stats)

    print()
    print(f"  RECALL    (truth founders matched by some disc hap):")
    print(f"             {total_recov} / {total_truth} "
          f"= {100.0 * total_recov / max(total_truth, 1):.3f}%")
    print(f"  PRECISION (disc haps that match some truth founder):")
    print(f"             {total_matched} / {total_disc} "
          f"= {100.0 * total_matched / max(total_disc, 1):.3f}%")

    # Per-block recall distribution
    per_block_recall = np.array([
        s['n_truth_recovered'] / s['K_true'] if s['K_true'] else 1.0
        for s in stats])
    perfect = int(np.sum(per_block_recall == 1.0))
    near    = int(np.sum((per_block_recall >= 0.8) & (per_block_recall < 1.0)))
    mid     = int(np.sum((per_block_recall >= 0.5) & (per_block_recall < 0.8)))
    low     = int(np.sum(per_block_recall < 0.5))
    print()
    print(f"  Per-block recall distribution (out of {n_blocks} L2 super-blocks):")
    print(f"     100% (all truth founders recovered): "
          f"{perfect:>5d} ({100*perfect/n_blocks:5.1f}%)")
    print(f"     80-99%:                              "
          f"{near:>5d} ({100*near/n_blocks:5.1f}%)")
    print(f"     50-79%:                              "
          f"{mid:>5d} ({100*mid/n_blocks:5.1f}%)")
    print(f"     <50%:                                "
          f"{low:>5d} ({100*low/n_blocks:5.1f}%)")

    # Histogram of best-truth-to-disc hamming
    all_truth_min = np.concatenate([
        np.array(s['truth_min_hammings']) for s in stats
        if s['truth_min_hammings']])
    if len(all_truth_min) > 0:
        finite = all_truth_min[np.isfinite(all_truth_min)]
        print()
        print(f"  Best-disc-match hamming-% across all (L2-block, truth) pairs "
              f"(n={len(finite)} finite):")
        bins = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        for lo, hi in zip(bins[:-1], bins[1:]):
            cnt = int(np.sum((finite >= lo) & (finite < hi)))
            print(f"    [{lo:>5.1f}%, {hi:>5.1f}%): "
                  f"{cnt:>6d} ({100*cnt/len(finite):5.2f}%)")
        cnt_over = int(np.sum(finite >= bins[-1]))
        if cnt_over > 0:
            print(f"    [{bins[-1]:>5.1f}%, +inf):  "
                  f"{cnt_over:>6d} ({100*cnt_over/len(finite):5.2f}%)")
        n_inf = int(np.sum(~np.isfinite(all_truth_min)))
        if n_inf > 0:
            print(f"    (uncomparable: {n_inf})")
        print(f"    median: {np.median(finite):.3f}%   "
              f"mean: {np.mean(finite):.3f}%   "
              f"p95: {np.percentile(finite, 95):.3f}%   "
              f"p99: {np.percentile(finite, 99):.3f}%")

    # Worst blocks by missing-founder count
    block_recall_with_idx = [
        (i, s['K_true'], s['K_disc'], s['n_truth_recovered'],
         s['K_true'] - s['n_truth_recovered'], s['n_block_sites'],
         s['n_kept_sites'])
        for i, s in enumerate(stats)]
    block_recall_with_idx.sort(key=lambda x: (-x[4], -x[1]))
    n_show = min(10, sum(1 for e in block_recall_with_idx if e[4] > 0))
    if n_show > 0:
        print()
        print(f"  Worst {n_show} L2 super-blocks by missing-truth count:")
        for entry in block_recall_with_idx[:n_show]:
            idx, kt, kd, nr, missing, nsites, nkept = entry
            h = stats[idx]['truth_min_hammings']
            h_str = ', '.join(
                f'{x:5.2f}' if np.isfinite(x) else '  inf' for x in h)
            print(f"    L2-block {idx:>4d}: K_true={kt}, K_disc={kd}, "
                  f"n_recov={nr} ({missing} missing), "
                  f"n_sites={nsites}, n_kept={nkept}.")
            print(f"        per-truth min-disc-hamming-%: [{h_str}]")
    else:
        print()
        print(f"  No L2 super-blocks with missing truth founders — all "
              f"{n_blocks} super-blocks recovered all "
              f"K_truth={len(truth_haploid)} founders.")

    print()
    print("=" * 78)


if __name__ == '__main__':
    main()