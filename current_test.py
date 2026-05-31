"""
test_level_refine — validate level_refine.refine_level (SELF-SCORING) on saved L1
checkpoints.

WHAT THIS CHECKS
----------------
level_refine now scores each super-block AGAINST ITSELF (no L0 blocks): each sample is
painted as a mosaic of the super-block's own haps over its window, directly from
global_probs.  This is a DIFFERENT objective than the earlier L0-keypath scoring used
by the L1->L2 prototype, so the prototype's results (and the prior L0-keypath
"31 improved, 0 lost") do NOT carry over -- this objective must be validated fresh.
The retired strict math-identity test (which compared to the L0-keypath core) no
longer applies, by design.

The success criteria (the gate for pipeline integration):
  (1) CONVERGENCE: every block converges (max passes < cap, no DID NOT CONVERGE).
      The re-paint fixed-point loop + min_dll margin guarantee termination.
  (2) AGREEMENT: truth-guided (mode a) and truth-free (mode b) counts are close
      (a genuine fixed point should not depend much on rep ordering).
  (3) TRUTH ACCURACY: each replaced hap should move CLOSER to its true founder, or at
      worst negligibly worse -- never lost.  dLL>min_dll optimises the SELF-BLOCK
      PAINTING LL (truth-free), which does NOT by itself guarantee closer-to-truth: on
      a contaminated carrier set a replace can raise LL while nudging a hap toward a
      near-twin.  So we measure, per replaced hap, the Hamming distance to the nearest
      true founder BEFORE vs AFTER, and report improved / worse / unchanged, the worst
      degradation, and any hap that crossed from <2% ("matches a founder") to >=2%
      ("lost").

Truth is used ONLY for these validation labels; the refinement itself is truth-free
(mode b is the production path; mode a's truth-guided rep selector exists only to
compare rep-ordering sensitivity).

Run:
  python test_level_refine.py chr3 chr4 chr6 chr7 chr8
  python test_level_refine.py                       # default set
"""
import os
import sys
import pickle

import thread_config  # noqa: F401

import numpy as np
import level_refine as lr


CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', '.pipeline_checkpoints')
DEFAULT_CONTIGS = ['chr3', 'chr4', 'chr6', 'chr7', 'chr8']
EPS_PRESENT = float(os.environ.get('EPS_PRESENT', '2.0'))  # the "matches a founder" band


def _load(stage, contig):
    with open(os.path.join(CHECKPOINT_DIR, stage, f'{contig}.pkl'), 'rb') as f:
        return pickle.load(f)


def _collapse(hap):
    arr = np.asarray(hap)
    if arr.ndim > 1:
        return np.argmax(arr, axis=1).astype(np.int8)
    return arr.astype(np.int8)


def _hamming_pct(a, b):
    return 100.0 * float(np.mean(a != b))


def _build(contig):
    import simulate_sequences
    nlh = _load('01_vcf_discovery', contig)['naive_long_haps']
    orig_sites, orig_haps = nlh
    orig_sites = np.asarray(orig_sites)
    truth = [np.asarray(t).astype(np.int8) for t in simulate_sequences.concretify_haps(orig_haps)]
    sim = _load('02_simulation', contig)
    global_probs = np.ascontiguousarray(sim['simd_probs'])
    l1 = _load('06_assembly_L1', contig)['super_blocks_L1']
    site_to_idx = {int(s): i for i, s in enumerate(orig_sites)}
    return orig_sites, truth, global_probs, l1, site_to_idx


def _block_seqs(sb):
    return np.stack([_collapse(sb.haplotypes[k]) for k in sorted(sb.haplotypes.keys())], axis=0)


def _nearest_truth_dist(hap, truth_block):
    """min Hamming(%) of hap to any true founder over this block's positions."""
    return min(_hamming_pct(hap, truth_block[f]) for f in range(truth_block.shape[0]))


def _truth_accuracy_delta(orig_blocks, refined_blocks, truth, site_to_idx):
    """For every block that changed, compare each hap's distance-to-nearest-true-founder
    before vs after.  Returns aggregate stats over all CHANGED haps."""
    K = len(truth)
    improved = worse = unchanged = 0
    worst_degradation = 0.0          # max positive (after-before) over changed haps (%)
    best_improvement = 0.0           # min (after-before) i.e. most negative (%)
    lost = 0                         # crossed <EPS -> >=EPS (matched a founder, now doesn't)
    gained = 0                       # crossed >=EPS -> <EPS (now matches a founder)
    per_change = []
    for j in range(len(orig_blocks)):
        ob, rb = orig_blocks[j], refined_blocks[j]
        if ob is rb:
            continue  # unchanged block (same object)
        opos = np.asarray(ob.positions)
        idx = np.array([site_to_idx[int(p)] for p in opos])
        tb = np.stack([truth[f][idx] for f in range(K)]).astype(np.int8)
        oseq = _block_seqs(ob)
        rseq = _block_seqs(rb)
        # haps are positionally aligned (replace-only, same count/order)
        n = min(oseq.shape[0], rseq.shape[0])
        for h in range(n):
            if np.array_equal(oseq[h], rseq[h]):
                continue  # this hap didn't change
            d_before = _nearest_truth_dist(oseq[h], tb)
            d_after = _nearest_truth_dist(rseq[h], tb)
            delta = d_after - d_before
            per_change.append((j, h, d_before, d_after, delta))
            if delta < -1e-12:
                improved += 1
            elif delta > 1e-12:
                worse += 1
            else:
                unchanged += 1
            worst_degradation = max(worst_degradation, delta)
            best_improvement = min(best_improvement, delta)
            if d_before < EPS_PRESENT <= d_after:
                lost += 1
            if d_before >= EPS_PRESENT > d_after:
                gained += 1
    return dict(improved=improved, worse=worse, unchanged=unchanged,
                worst_degradation=worst_degradation, best_improvement=best_improvement,
                lost=lost, gained=gained, per_change=per_change)


def main(contig):
    orig_sites, truth, global_probs, l1, site_to_idx = _build(contig)
    K = len(truth)
    n_l1 = len(l1)
    print("=" * 88)
    print(f"test_level_refine — {contig}   (L1 blocks: {n_l1}, K={K})")
    print("=" * 88)

    # truth-guided rep selector: rep for founder f = nearest current hap to truth[f].
    # Used ONLY to compare rep-ordering sensitivity vs the truth-free path.
    def truth_guided_reps(j, cur_seqs, positions):
        idx = np.array([site_to_idx[int(p)] for p in positions])
        reps = []
        for f in range(K):
            tf = truth[f][idx]
            dists = np.array([_hamming_pct(cur_seqs[h], tf) for h in range(cur_seqs.shape[0])])
            reps.append(int(np.argmin(dists)))
        return reps

    # ---- mode (a): truth-guided reps (rep-ordering sensitivity check) ----
    print("\n  [mode a] truth-guided reps ...")
    refined_a, actions_a = lr.refine_level(
        l1, global_probs, orig_sites,
        rep_selector=truth_guided_reps, return_actions=True, verbose=True)
    replace_a = sum(len(v) for v in actions_a.values())
    iters_a = [x.get('block_iter', 1) for v in actions_a.values() for x in v]
    max_iter_a = max(iters_a) if iters_a else 0

    # ---- mode (b): truth-free reps (PRODUCTION path) ----
    print("\n  [mode b] truth-free reps (production path) ...")
    refined_b, actions_b = lr.refine_level(
        l1, global_probs, orig_sites,
        rep_selector=None, return_actions=True, verbose=True)
    replace_b = sum(len(v) for v in actions_b.values())
    iters_b = [x.get('block_iter', 1) for v in actions_b.values() for x in v]
    max_iter_b = max(iters_b) if iters_b else 0

    # ---- (3) TRUTH ACCURACY on the PRODUCTION (mode-b) refinement ----
    print("\n  [truth accuracy] mode-b refined haps vs nearest true founder (before/after) ...")
    acc = _truth_accuracy_delta(l1, refined_b, truth, site_to_idx)

    # ---- criteria ----
    # convergence: max pass strictly below cap (cap = 3*K). also: refine_level prints
    # a loud DID NOT CONVERGE line itself; here we infer from max pass vs cap.
    cap = 3 * K
    converged_a = (max_iter_a < cap)
    converged_b = (max_iter_b < cap)
    agree = abs(replace_a - replace_b) <= max(2, int(0.1 * max(replace_a, replace_b)))
    no_founder_lost = (acc['lost'] == 0)

    print("\n  SUMMARY — " + contig)
    print(f"    replacements:   mode-a {replace_a} (max pass {max_iter_a}), "
          f"mode-b {replace_b} (max pass {max_iter_b})   [cap={cap}]")
    print(f"    (1) converged:  mode-a {'YES' if converged_a else '*** NO (hit cap)'}, "
          f"mode-b {'YES' if converged_b else '*** NO (hit cap)'}")
    print(f"    (2) agreement:  |{replace_a}-{replace_b}|={abs(replace_a-replace_b)}  "
          f"{'OK' if agree else '*** modes disagree'}")
    print(f"    (3) truth acc:  improved {acc['improved']}, worse {acc['worse']}, "
          f"unchanged {acc['unchanged']}  | worst degradation {acc['worst_degradation']:+.3f}%, "
          f"best improvement {acc['best_improvement']:+.3f}%")
    print(f"                    founders LOST (<2%->>=2%): {acc['lost']}   "
          f"founders GAINED (>=2%-><2%): {acc['gained']}")
    if acc['worse'] > 0:
        # show the worse ones (small list)
        worse_list = [(j, h, f"{b:.3f}->{a:.3f}%") for (j, h, b, a, d) in acc['per_change'] if d > 1e-12]
        print(f"                    worse haps ({len(worse_list)}): {worse_list[:10]}"
              f"{' ...' if len(worse_list) > 10 else ''}")

    overall_ok = converged_a and converged_b and agree and no_founder_lost
    print(f"    => {'PASS' if overall_ok else '*** CHECK'} "
          f"(converged + modes agree + no founder lost)")
    return overall_ok, acc


if __name__ == '__main__':
    import hierarchical_assembly as _ha
    _w = _ha.NoDaemonPool(1); _w.terminate(); _w.join(); del _w
    print("Forkserver started (lightweight, pre-data).")
    all_ok = True
    total_improved = total_worse = total_lost = 0
    worst_deg = 0.0
    for c in (sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CONTIGS):
        ok, acc = main(c)
        all_ok = all_ok and ok
        total_improved += acc['improved']; total_worse += acc['worse']; total_lost += acc['lost']
        worst_deg = max(worst_deg, acc['worst_degradation'])
        print()
    print("=" * 88)
    print(f"OVERALL: improved {total_improved}, worse {total_worse}, lost {total_lost}, "
          f"worst degradation {worst_deg:+.3f}%")
    print(f"  criteria (converge + modes agree + no founder lost): "
          f"{'ALL PASS' if all_ok else '*** SOME FAILED — investigate'}")
    print("=" * 88)