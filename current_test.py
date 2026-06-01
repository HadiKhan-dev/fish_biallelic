"""
diagnose_block154_L0 — determine whether L0->L1 chimera resolution could have fixed
block 154's F2 tail, by inspecting block 154's constituent L0 blocks.

The question: block 154 (L1) is built from ~10 L0 blocks via beam search +
chimera_resolution + reconstruction.  Its F2 hap is correct in the head, garbled in the
tail [34102448..34157989].  Could chimera_resolution (which routes each founder through
L0 block-haplotypes) have fixed the tail at the L0->L1 build?

Two scenarios, distinguished by the L0 data:
  SCENARIO 1 (routing-fixable): a clean true-F2 L0 hap EXISTS in the tail L0 blocks but
    the path selected a different L0 hap.  chimera_resolution could route to it.
  SCENARIO 2 (not in L0 vocabulary): no L0 hap is a clean true-F2 segment in the tail
    (L0 reconstruction itself merged F2/F5).  Nothing to route to.

PLUS the flat-objective check: over each tail L0 block, what is true-F2 vs true-F5
Hamming?  If ~0, F2/F5 coincide at L0-block scale -> routing between them is cost-free
-> chimera_resolution has no signal to prefer correct routing (the twin-coincidence
wall at the routing level), regardless of scenario.

Outputs:
  1. Which L0 blocks span block 154's window; which lie in the tail span.
  2. For each tail L0 block: F2 vs F5 truth distance over that block (the flat-objective
     test), and each L0 hap's distance to true F2 and true F5 (the vocabulary test).
  3. Verdict: scenario 1 vs 2, and whether the objective is flat in the tail.

Falls back gracefully if simd_block_results (L0) is pruned from the checkpoint.

Run:  python diagnose_block154_L0.py                 # chr4 154 F2 (vs F5)
      python diagnose_block154_L0.py chr4 154 F2 F5
"""
import os
import sys
import pickle

import thread_config  # noqa: F401

import numpy as np


CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', '.pipeline_checkpoints')


def _load(stage, contig):
    with open(os.path.join(CHECKPOINT_DIR, stage, f'{contig}.pkl'), 'rb') as f:
        return pickle.load(f)


def _collapse(hap):
    arr = np.asarray(hap)
    if arr.ndim > 1:
        return np.argmax(arr, axis=1).astype(np.int8)
    return arr.astype(np.int8)


def _ham(a, b):
    if len(a) == 0:
        return float('nan')
    return 100.0 * float(np.mean(a != b))


def _build(contig):
    import simulate_sequences
    nlh = _load('01_vcf_discovery', contig)['naive_long_haps']
    orig_sites = np.asarray(nlh[0])
    truth = [np.asarray(t).astype(np.int8) for t in simulate_sequences.concretify_haps(nlh[1])]
    l1 = _load('06_assembly_L1', contig)['super_blocks_L1']
    site_to_idx = {int(s): i for i, s in enumerate(orig_sites)}
    return orig_sites, truth, l1, site_to_idx


def _try_load_L0(contig):
    """Try several plausible keys/stages for the L0 block haplotypes."""
    candidates = [
        ('03_block_haplotypes', 'simd_block_results'),
        ('03_block_haplotypes', 'block_results'),
        ('03_block_haplotypes', 'L0'),
    ]
    for stage, key in candidates:
        try:
            d = _load(stage, contig)
            if key in d and d[key] is not None:
                return d[key], f"{stage}:{key}"
        except Exception:
            continue
    # also try: maybe the whole checkpoint IS the list
    try:
        d = _load('03_block_haplotypes', contig)
        for k, v in d.items():
            if isinstance(v, (list, tuple)) and len(v) > 0 and hasattr(v[0], 'haplotypes'):
                return v, f"03_block_haplotypes:{k}"
    except Exception:
        pass
    return None, None


def main(contig, j_l1, fa, fb):
    orig_sites, truth, l1, site_to_idx = _build(contig)
    K = len(truth)
    a = int(fa.lstrip('F')); b = int(fb.lstrip('F'))

    # locate block 154's tail span (where its F_a hap is wrong)
    sb = l1[j_l1]
    pos = np.asarray(sb.positions); L = len(pos)
    idx = np.array([site_to_idx[int(p)] for p in pos])
    seqs = np.stack([_collapse(sb.haplotypes[k]) for k in sorted(sb.haplotypes.keys())], axis=0)
    tb1 = np.stack([truth[f][idx] for f in range(K)]).astype(np.int8)
    D1 = np.array([[_ham(seqs[h], tb1[f]) for f in range(K)] for h in range(seqs.shape[0])])
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(D1)
        h1 = {int(c): int(r) for r, c in zip(rows, cols)}[a]
    except Exception:
        h1 = int(np.argmin(D1[:, a]))
    diff_full = (seqs[h1] != tb1[a]).astype(float)
    win = max(20, L // 40)
    local = np.convolve(diff_full, np.ones(win) / win, mode='same')
    hot = np.nonzero(local > 0.10)[0]
    split = int(hot[0]) if hot.size else int(L * 0.75)
    tail_lo_pos, tail_hi_pos = int(pos[split]), int(pos[-1])

    print("=" * 92)
    print(f"diagnose_block154_L0 — {contig} L1 block {j_l1}, founder F{a} (twin F{b})")
    print(f"  L1 window [{int(pos[0])}..{int(pos[-1])}], tail span [{tail_lo_pos}..{tail_hi_pos}]")
    print(f"  L1 F{a} hap tail-dist to truth: {_ham(seqs[h1][split:], tb1[a][split:]):.3f}%")
    print("=" * 92)

    # ---- the flat-objective test does NOT need L0 haps; do it from truth first ----
    # Partition the L1 window into ~200-SNP L0-sized chunks and report F_a vs F_b per chunk.
    print(f"\n  FLAT-OBJECTIVE TEST (truth-only): F{a} vs F{b} per ~200-SNP L0-sized chunk")
    print(f"  across the L1 window.  (~0% in the tail => routing F{a}<->F{b} is cost-free")
    print(f"  there, so chimera_resolution has NO signal to route correctly.)")
    chunk = 200
    nchunks = int(np.ceil(L / chunk))
    print("    chunk  SNP-range            genomic-range                    F%da-F%db" % (a, b))
    for c in range(nchunks):
        lo = c * chunk; hi = min(lo + chunk, L)
        d = _ham(tb1[a][lo:hi], tb1[b][lo:hi])
        in_tail = lo >= split
        marker = "  <-- TAIL" if in_tail else ""
        print(f"    {c:<5}  [{lo:5}:{hi:5}]  [{int(pos[lo]):>9}..{int(pos[hi-1]):>9}]   "
              f"{d:6.2f}%{marker}")

    # ---- vocabulary test: needs the L0 blocks ----
    print("\n  VOCABULARY TEST: do block 154's tail L0 blocks contain a clean true-F%d" % a)
    print("  segment (scenario 1, routable) or is F%d merged into F%d at L0 (scenario 2)?" % (a, b))
    L0, src = _try_load_L0(contig)
    if L0 is None:
        print("\n    [L0 block haplotypes NOT available in checkpoints -- simd_block_results")
        print("     was pruned after L1 (known pipeline behaviour).  Cannot inspect L0 hap")
        print("     vocabulary directly.  The flat-objective test above still stands and is")
        print("     the decisive one: see verdict.]")
        L0 = None
    else:
        print(f"    [loaded L0 from {src}: {len(L0)} L0 blocks]")
        # find L0 blocks overlapping the tail span
        tail_L0 = []
        for bi in range(len(L0)):
            p0 = np.asarray(L0[bi].positions)
            if p0[-1] >= tail_lo_pos and p0[0] <= tail_hi_pos:
                ov = int(np.sum((p0 >= tail_lo_pos) & (p0 <= tail_hi_pos)))
                if ov > 0:
                    tail_L0.append(bi)
        print(f"    L0 blocks overlapping the tail span: {tail_L0}")
        for bi in tail_L0:
            p0 = np.asarray(L0[bi].positions)
            in_t = (p0 >= tail_lo_pos) & (p0 <= tail_hi_pos)
            idx0 = np.array([site_to_idx[int(pp)] for pp in p0[in_t]])
            if len(idx0) == 0:
                continue
            t_a = truth[a][idx0]; t_b = truth[b][idx0]
            seqs0 = np.stack([_collapse(L0[bi].haplotypes[k])[in_t]
                              for k in sorted(L0[bi].haplotypes.keys())], axis=0)
            da = np.array([_ham(seqs0[h], t_a) for h in range(seqs0.shape[0])])
            db = np.array([_ham(seqs0[h], t_b) for h in range(seqs0.shape[0])])
            best_a = float(da.min()); best_b = float(db.min())
            print(f"      L0 block {bi} (tail part, {in_t.sum()} SNPs, {seqs0.shape[0]} haps): "
                  f"F{a}/F{b} truth-dist over this part = {_ham(t_a, t_b):.2f}%")
            print(f"         best L0 hap match to true F{a}: {best_a:.2f}%   "
                  f"to true F{b}: {best_b:.2f}%")
            if best_a < 2.0 and _ham(t_a, t_b) > 2.0:
                print(f"         -> a clean true-F{a} L0 hap EXISTS here and F{a}!=F{b} "
                      f"(SCENARIO 1: routable)")
            elif _ham(t_a, t_b) < 1.0:
                print(f"         -> F{a}==F{b} here (twin-coincidence): routing cost-free, "
                      f"no signal (flat objective)")
            else:
                print(f"         -> no clean true-F{a} L0 hap (SCENARIO 2: not in vocabulary)")

    # ---- verdict ----
    tail_d = _ham(tb1[a][split:], tb1[b][split:])
    print("\n" + "=" * 92)
    print("  VERDICT")
    print("=" * 92)
    if tail_d < 1.0:
        print(f"  F{a} and F{b} coincide over the tail ({tail_d:.2f}%).  Whether or not a")
        print(f"  clean L0 hap exists, chimera_resolution's objective is FLAT there: routing a")
        print(f"  sample onto F{a}'s hap vs F{b}'s hap costs the same (identical sequences), so")
        print(f"  it has NO basis to prefer the correct routing.  => L0->L1 chimera resolution")
        print(f"  CANNOT fix this tail -- not a tuning issue, a fundamental lack of signal where")
        print(f"  the two founders are identical.  Same twin-coincidence wall, routing-level.")
        print(f"  The error is only resolvable using the FLANKS where F{a}!=F{b} to carry the")
        print(f"  distinction THROUGH the coincident region -- which neither boundary-swap")
        print(f"  chimera resolution nor carrier re-derivation does.")
    else:
        print(f"  F{a} and F{b} DIFFER over the tail ({tail_d:.2f}%) -- NOT a pure twin-")
        print(f"  coincidence.  If a clean true-F{a} L0 hap exists (scenario 1), then")
        print(f"  chimera_resolution SHOULD have been able to route to it, and its failure to")
        print(f"  is a detection/threshold gap worth investigating in find_hotspots.")


if __name__ == '__main__':
    args = sys.argv[1:]
    contig = args[0] if len(args) >= 1 else 'chr4'
    j = int(args[1]) if len(args) >= 2 else 154
    fa = args[2] if len(args) >= 3 else 'F2'
    fb = args[3] if len(args) >= 4 else 'F5'
    main(contig, j, fa, fb)