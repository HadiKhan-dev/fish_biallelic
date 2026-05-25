"""
Consolidated parameter sweep for bhd_trio.

Runs one-at-a-time (OAT) sweeps over all 5 tunable trio parameters in
a single script invocation, sharing the loaded contig data + test
block set across sweeps for ~5x speedup vs running 5 separate scripts.

Parameters swept (in this order):

  1. cluster_fraction       — clustering threshold = cluster_fraction*D
                              (controls how aggressively samples merge
                              into pair-type groups before trio
                              enumeration).  KEY LEVER for chr6:1006
                              because F0/F1 and F1/F5 differ by only
                              (F0+F5) mod 2 = 12 bits, so a loose
                              cluster_fraction merges them.

  2. match_fraction         — triangle match threshold = match_fraction*D
                              (controls how loose the
                              X(g1) XOR X(g2) ≈ X(g3) check is).
                              Tighter values reject more candidate
                              trios including some tetragons.

  3. distinct_fraction      — pairwise distinct threshold = distinct_fraction*D
                              (rejects trios where any two of the
                              three group centroids are too close).
                              Tighter values reject more spurious
                              same-pair-type trios.

  4. hap_dedup_pct          — Hamming-% threshold below which two
                              candidate haplotypes are considered the
                              same founder for output clustering.

  5. min_hap_cluster_size   — minimum cluster size to emit (drop noise).
                              AFTER the canonical-enumeration fix in
                              bhd_trio, this maps directly to "minimum
                              number of supporting underlying
                              triangles" per emitted founder.

For each parameter:
  - Focal block (chr6:1006 F4 by default) status at each swept value
  - Aggregate stats across all 253+ test blocks
  - Fixes/breaks vs default with per-founder block listings

Then a cross-parameter summary highlighting the value(s) that
recover the focal founder + their cost in regressions.

Run as:
  python sweep_trio_all_params.py
  python sweep_trio_all_params.py --focal chr6:1006:4
  python sweep_trio_all_params.py --skip distinct_fraction,hap_dedup_pct
  python sweep_trio_all_params.py --healthy-sample 100   # fewer regression blocks

Env vars:
  CHECKPOINT_DIR=...
  RECOVERY_THRESH=1.0
  STAGE=03_block_haplotypes
"""
import os
import sys
import pickle
import argparse
import random
import time
import numpy as np

import simulate_sequences
import bhd_trio


CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', '.pipeline_checkpoints')
STAGE          = os.environ.get('STAGE', '03_block_haplotypes')

_KEY_SOURCE = {
    'naive_long_haps':    ['01_vcf_discovery'],
    'simd_probs':         ['02_simulation'],
    'simd_block_results': ['05_residual_discovery', '04_refinement', '03_block_haplotypes'],
}

# Order in which we sweep — putting the one most likely to fix chr6:1006
# (cluster_fraction) first so the most actionable result lands at the
# top of the report rather than buried at the bottom.
SWEEP_ORDER = [
    'cluster_fraction',
    'match_fraction',
    'distinct_fraction',
    'hap_dedup_pct',
    'min_hap_cluster_size',
]


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def _resolve_key(key, contig, stage_override=None):
    stages = ([stage_override] if stage_override else _KEY_SOURCE[key])
    for stage in stages:
        path = os.path.join(CHECKPOINT_DIR, stage, f'{contig}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                blob = pickle.load(f)
            if key in blob:
                return blob[key]
    return None


def _discover_contigs():
    vcf_dir = os.path.join(CHECKPOINT_DIR, '01_vcf_discovery')
    return sorted(f[:-4] for f in os.listdir(vcf_dir) if f.endswith('.pkl'))


class ContigCache:
    """Hold the per-contig data we need to score blocks."""
    def __init__(self, contig):
        self.contig = contig
        nlh = _resolve_key('naive_long_haps', contig)
        orig_sites, orig_haps = nlh
        self.orig_sites = np.asarray(orig_sites)
        self.site_to_idx = {int(s): i for i, s in enumerate(self.orig_sites)}
        self.truth_haps = simulate_sequences.concretify_haps(orig_haps)
        self.K_truth = len(self.truth_haps)

        # Block results from the requested inspection stage (default
        # 03_block_haplotypes — that's what we want to compare against
        # for the "did trio recovery have the chance to find it"
        # question).
        self.simd_block_results = _resolve_key(
            'simd_block_results', contig, stage_override=STAGE)
        if self.simd_block_results is None:
            # Fall through to the normal fallback chain
            self.simd_block_results = _resolve_key(
                'simd_block_results', contig)
        if self.simd_block_results is None:
            raise FileNotFoundError(
                f"No simd_block_results for {contig} at any stage.")

        self.simd_probs = _resolve_key('simd_probs', contig)
        if self.simd_probs is None:
            raise FileNotFoundError(f"No simd_probs for {contig}.")

    def get_block_input(self, block_idx):
        """Return (probs_k, truth_bits) for one L0 block, in kept-site space."""
        block = self.simd_block_results[block_idx]
        positions = np.asarray(block.positions)
        keep_flags = (np.asarray(block.keep_flags, dtype=bool)
                      if block.keep_flags is not None
                      else np.ones(len(positions), dtype=bool))
        kept_positions = positions[keep_flags]
        if len(kept_positions) == 0:
            return None, None
        try:
            site_idx_kept = np.array(
                [self.site_to_idx[int(p)] for p in kept_positions])
        except KeyError:
            return None, None
        probs_k = self.simd_probs[:, site_idx_kept, :].astype(np.float64)
        truth_bits = np.stack(
            [self.truth_haps[f][site_idx_kept] for f in range(self.K_truth)]
        ).astype(np.int64)
        return probs_k, truth_bits


def _build_all_contig_caches():
    """Load every contig's cache once, return dict[contig] -> ContigCache.
    Shared across find_failure_blocks, sample_healthy_blocks, and the
    sweep itself — avoids ~3x redundant disk I/O.
    """
    contigs = _discover_contigs()
    print(f"Loading {len(contigs)} contig caches ...", flush=True)
    t0 = time.time()
    out = {}
    for c in contigs:
        out[c] = ContigCache(c)
    print(f"  loaded in {time.time() - t0:.1f}s")
    return out


# -----------------------------------------------------------------------------
# Block list construction (using pre-loaded caches)
# -----------------------------------------------------------------------------
def _block_failure_check(block, truth_haps, site_to_idx, threshold_pct):
    """Return list of truth-founder indices whose best disc-hap Hamming-%
    is >= threshold_pct.  Empty list = perfect block.  Uses argmax-based
    Hamming (matches L1 test + the previous diagnostic scripts).
    """
    positions = np.asarray(block.positions)
    K_truth = len(truth_haps)
    try:
        site_idx = np.array([site_to_idx[int(p)] for p in positions])
    except KeyError:
        return list(range(K_truth))   # treat as all-missing
    truth_at_block = [truth_haps[f][site_idx] for f in range(K_truth)]
    disc_concrete = []
    for h in block.haplotypes.values():
        disc_concrete.append(np.argmax(h, axis=1) if h.ndim > 1 else h)
    if not disc_concrete:
        return list(range(K_truth))
    missing = []
    for f in range(K_truth):
        best = min(np.mean(d != truth_at_block[f]) * 100.0
                   for d in disc_concrete)
        if best >= threshold_pct:
            missing.append(f)
    return missing


def find_failure_blocks(contig_caches, threshold_pct):
    """Discover all (contig, block_idx) where some truth founder is
    missing from the discovery output.
    """
    out = []
    print(f"Scanning failure blocks across {len(contig_caches)} contigs ...",
          flush=True)
    for contig, cache in contig_caches.items():
        n_blocks = len(cache.simd_block_results)
        for bi in range(n_blocks):
            missing = _block_failure_check(
                cache.simd_block_results[bi],
                cache.truth_haps,
                cache.site_to_idx,
                threshold_pct,
            )
            if missing:
                out.append((contig, bi, missing))
    print(f"  Found {len(out)} failure blocks across all contigs.")
    return out


def sample_healthy_blocks(contig_caches, failure_set, n_total, seed=42):
    """Random sample of (contig, block_idx) NOT in failure_set, spread
    across contigs proportional to block count.
    """
    failure_keys = {(c, b) for (c, b, _) in failure_set}
    rng = random.Random(seed)
    per_contig = max(1, n_total // len(contig_caches))
    out = []
    for contig, cache in contig_caches.items():
        n_blocks = len(cache.simd_block_results)
        candidates = [bi for bi in range(n_blocks)
                      if (contig, bi) not in failure_keys]
        if not candidates:
            continue
        k = min(per_contig, len(candidates))
        sampled = rng.sample(candidates, k)
        for bi in sampled:
            out.append((contig, bi, []))
    print(f"  Sampled {len(out)} healthy blocks (target {n_total}).")
    return out


# -----------------------------------------------------------------------------
# Scoring trio output
# -----------------------------------------------------------------------------
def _score_trio_output(candidate_haps, truth_bits, threshold_pct):
    """Return (recovered_set, n_candidates, n_spurious).

    recovered_set: set of truth-founder indices f for which some
        candidate has Hamming-% to truth_bits[f] below threshold_pct.
    n_candidates: total candidate haps produced.
    n_spurious: candidates whose best-truth Hamming-% is >= threshold_pct.
    """
    K_truth = truth_bits.shape[0]
    if candidate_haps.shape[0] == 0:
        return set(), 0, 0
    n_cand = candidate_haps.shape[0]
    ham = np.zeros((n_cand, K_truth), dtype=np.float64)
    for c in range(n_cand):
        for f in range(K_truth):
            ham[c, f] = 100.0 * float((candidate_haps[c] != truth_bits[f]).mean())
    truth_best = ham.min(axis=0)
    recovered = {int(f) for f in range(K_truth)
                 if truth_best[f] < threshold_pct}
    cand_best = ham.min(axis=1)
    n_spurious = int((cand_best >= threshold_pct).sum())
    return recovered, n_cand, n_spurious


# -----------------------------------------------------------------------------
# One-parameter sweep
# -----------------------------------------------------------------------------
def run_sweep(test_blocks, param_name, param_values,
              contig_caches, threshold_pct, focal=None):
    """Run trio recovery for each block × param value combination."""
    results = {}
    n_total = len(test_blocks) * len(param_values)
    n_done = 0
    print()
    print(f"Running {param_name} sweep: "
          f"{len(test_blocks)} blocks × {len(param_values)} values "
          f"= {n_total} runs",
          flush=True)
    t_start = time.time()

    for (contig, block_idx, _missing) in test_blocks:
        cache = contig_caches[contig]
        probs_k, truth_bits = cache.get_block_input(block_idx)
        if probs_k is None:
            n_done += len(param_values)
            continue
        per_pv = {}
        for pv in param_values:
            kwargs = {param_name: pv}
            try:
                candidate_haps = bhd_trio._trio_recovery_candidate_haps(
                    probs_k, verbose=False, **kwargs)
            except Exception as e:
                # Print so we can see real errors, but don't crash the whole sweep
                print(f"    WARNING: {param_name}={pv} at "
                      f"{contig}:{block_idx} raised {type(e).__name__}: {e}")
                per_pv[pv] = (set(), 0, 0)
                n_done += 1
                continue
            recovered, n_cand, n_spur = _score_trio_output(
                candidate_haps, truth_bits, threshold_pct)
            per_pv[pv] = (recovered, n_cand, n_spur)
            n_done += 1
        results[(contig, block_idx)] = per_pv
        if (n_done % max(1, n_total // 10) < len(param_values)
                or n_done == n_total):
            elapsed = time.time() - t_start
            print(f"  {n_done}/{n_total} runs "
                  f"({100.0*n_done/n_total:.0f}%), {elapsed:.0f}s elapsed",
                  flush=True)
    return results


# -----------------------------------------------------------------------------
# Per-parameter report
# -----------------------------------------------------------------------------
def report_parameter(results, test_blocks, param_values, default_value,
                     param_name, focal, threshold_pct):
    print()
    print("=" * 78)
    print(f"  RESULTS:  {param_name}")
    print("=" * 78)

    # Focal block status per param
    if focal is not None:
        focal_contig, focal_bi, focal_f = focal
        print()
        print(f"FOCAL BLOCK: {focal_contig}:{focal_bi}, founder F{focal_f}")
        print("-" * 78)
        key = (focal_contig, focal_bi)
        if key not in results:
            print(f"  (no results for focal block — block_idx may be "
                  f"out of range)")
        else:
            print(f"  {param_name:<22s}  F{focal_f} recov?   "
                  f"n_cand   n_spur")
            for pv in param_values:
                rec, n_cand, n_spur = results[key][pv]
                marker = ""
                if pv == default_value:
                    marker = " (default)"
                status = "YES" if focal_f in rec else "NO"
                print(f"  {pv:<22.4f}  {status:<8s}  "
                      f"{n_cand:>6d}   {n_spur:>5d}{marker}")

    # Aggregate stats per param value
    print()
    print(f"AGGREGATE across {len(test_blocks)} test blocks")
    print("-" * 78)
    if focal is not None:
        print(f"  {param_name:<22s}  recov    cand    spur   "
              f"blocks_w_F{focal[2]}")
    else:
        print(f"  {param_name:<22s}  recov    cand    spur")

    total_recov = {pv: 0 for pv in param_values}
    total_cand  = {pv: 0 for pv in param_values}
    total_spur  = {pv: 0 for pv in param_values}
    focal_recov_count = {pv: 0 for pv in param_values}

    for key, per_pv in results.items():
        for pv in param_values:
            rec, n_cand, n_spur = per_pv[pv]
            total_recov[pv] += len(rec)
            total_cand[pv]  += n_cand
            total_spur[pv]  += n_spur
            if focal is not None and focal[2] in rec:
                focal_recov_count[pv] += 1

    for pv in param_values:
        marker = " (default)" if pv == default_value else ""
        focal_str = ""
        if focal is not None:
            focal_str = f"   {focal_recov_count[pv]:>4d}"
        print(f"  {pv:<22.4f}  {total_recov[pv]:>5d}   "
              f"{total_cand[pv]:>5d}   {total_spur[pv]:>5d}"
              f"{focal_str}{marker}")

    # Fixes / breaks per non-default param value
    print()
    print(f"FIXES/BREAKS vs default {param_name}={default_value}")
    print("-" * 78)
    print(f"  {param_name:<22s}  net   fixes   breaks")

    fixes_by_pv = {}
    breaks_by_pv = {}
    for pv in param_values:
        if pv == default_value:
            continue
        fixes  = []
        breaks = []
        for (contig, bi), per_pv in results.items():
            rec_default = per_pv[default_value][0]
            rec_pv      = per_pv[pv][0]
            for f in rec_pv - rec_default:
                fixes.append((contig, bi, f))
            for f in rec_default - rec_pv:
                breaks.append((contig, bi, f))
        net = len(fixes) - len(breaks)
        sign = "+" if net >= 0 else ""
        print(f"  {pv:<22.4f}  {sign}{net:>3d}   "
              f"{len(fixes):>4d}    {len(breaks):>4d}")
        fixes_by_pv[pv] = fixes
        breaks_by_pv[pv] = breaks

    # Detail at each non-default PV (founder x block listings)
    for pv in param_values:
        if pv == default_value:
            continue
        if not fixes_by_pv[pv] and not breaks_by_pv[pv]:
            continue
        print()
        print(f"  --- detail at {param_name}={pv} ---")
        if fixes_by_pv[pv]:
            print(f"  FIXES ({len(fixes_by_pv[pv])}):")
            by_founder = {}
            for c, b, f in fixes_by_pv[pv]:
                by_founder.setdefault(f, []).append((c, b))
            for f in sorted(by_founder):
                items = by_founder[f][:8]
                more = len(by_founder[f]) - len(items)
                items_str = ", ".join(f"{c}:{b}" for c, b in items)
                if more:
                    items_str += f", ...(+{more} more)"
                print(f"    F{f}: {items_str}")
        if breaks_by_pv[pv]:
            print(f"  BREAKS ({len(breaks_by_pv[pv])}):")
            by_founder = {}
            for c, b, f in breaks_by_pv[pv]:
                by_founder.setdefault(f, []).append((c, b))
            for f in sorted(by_founder):
                items = by_founder[f][:8]
                more = len(by_founder[f]) - len(items)
                items_str = ", ".join(f"{c}:{b}" for c, b in items)
                if more:
                    items_str += f", ...(+{more} more)"
                print(f"    F{f}: {items_str}")

    return total_recov, total_cand, total_spur, focal_recov_count


# -----------------------------------------------------------------------------
# Cross-parameter summary
# -----------------------------------------------------------------------------
def print_cross_summary(all_results, focal):
    """Across all swept parameters, identify which (parameter, value)
    combinations recover the focal founder + their aggregate cost.
    """
    print()
    print("=" * 78)
    print("  CROSS-PARAMETER SUMMARY")
    print("=" * 78)

    if focal is None:
        print("  (no focal block specified)")
        return

    focal_contig, focal_bi, focal_f = focal
    focal_key = (focal_contig, focal_bi)

    print()
    print(f"FOCAL: {focal_contig}:{focal_bi} F{focal_f}")
    print(f"Values that recover the focal founder, sorted by net "
          f"impact on aggregate recovery:")
    print()
    print(f"  {'parameter':<22s}  {'value':>8s}  "
          f"{'focal':<6s}  {'agg_recov':>9s}  "
          f"{'spur':>5s}  {'vs_def':>7s}")
    print("  " + "-" * 70)

    rows = []
    for param_name, info in all_results.items():
        per_pv_totals = info['totals']
        per_pv_focal  = info['focal']
        default_value = info['default']
        for pv in info['values']:
            focal_ok = per_pv_focal.get(pv, 0) > 0 if False else None
            # Get focal status for THIS block from raw results
            raw = info['raw_results']
            if focal_key in raw:
                focal_rec_pv = raw[focal_key][pv][0]
                focal_ok = focal_f in focal_rec_pv
            else:
                focal_ok = None
            total_recov, _, _ = per_pv_totals[pv]
            total_spur = info['totals_spur'][pv]
            default_recov = per_pv_totals[default_value][0]
            delta = total_recov - default_recov
            rows.append({
                'param': param_name,
                'value': pv,
                'focal_ok': focal_ok,
                'agg_recov': total_recov,
                'spur': total_spur,
                'delta_vs_default': delta,
                'is_default': pv == default_value,
            })

    # Filter to those that recover focal, sort by delta desc
    focal_recovered_rows = [r for r in rows if r['focal_ok'] is True]
    focal_recovered_rows.sort(key=lambda r: -r['delta_vs_default'])

    if focal_recovered_rows:
        for r in focal_recovered_rows:
            tag = "YES"
            sign = "+" if r['delta_vs_default'] >= 0 else ""
            default_marker = " (default)" if r['is_default'] else ""
            print(f"  {r['param']:<22s}  {r['value']:>8.4f}  "
                  f"{tag:<6s}  {r['agg_recov']:>9d}  "
                  f"{r['spur']:>5d}  {sign}{r['delta_vs_default']:>5d}"
                  f"{default_marker}")
    else:
        print(f"  NONE — no swept parameter value recovers F{focal_f} at "
              f"{focal_contig}:{focal_bi} alone.  May require combining "
              f"parameter changes (grid sweep), or a structural change "
              f"to the algorithm.")

    # Defaults for comparison
    print()
    print("DEFAULTS (for comparison):")
    print(f"  {'parameter':<22s}  {'value':>8s}  "
          f"{'focal':<6s}  {'agg_recov':>9s}  {'spur':>5s}")
    print("  " + "-" * 60)
    for r in rows:
        if not r['is_default']:
            continue
        tag = "YES" if r['focal_ok'] else "NO"
        print(f"  {r['param']:<22s}  {r['value']:>8.4f}  "
              f"{tag:<6s}  {r['agg_recov']:>9d}  {r['spur']:>5d}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Consolidated bhd_trio parameter sweep (all params, OAT)")
    ap.add_argument('--focal', default='chr6:1006:4',
                    help="Focal block:founder, e.g. chr6:1006:4 or none")
    ap.add_argument('--healthy-sample', type=int, default=200,
                    help="Random healthy blocks for regression test")
    ap.add_argument('--threshold', type=float,
                    default=float(os.environ.get('RECOVERY_THRESH', '1.0')),
                    help="Hamming%% threshold for 'recovered'")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip', default='',
                    help="Comma-separated parameters to skip")
    # Per-parameter overrides for the value list
    ap.add_argument('--values-cluster_fraction', default=None,
                    help="Override default value list for cluster_fraction")
    ap.add_argument('--values-match_fraction', default=None,
                    help="Override default value list for match_fraction")
    ap.add_argument('--values-distinct_fraction', default=None,
                    help="Override default value list for distinct_fraction")
    ap.add_argument('--values-hap_dedup_pct', default=None,
                    help="Override default value list for hap_dedup_pct")
    ap.add_argument('--values-min_hap_cluster_size', default=None,
                    help="Override default value list for min_hap_cluster_size")
    args = ap.parse_args()

    # Defaults pulled from the bhd_trio module (so they reflect any
    # post-patch defaults, e.g. TRIO_MIN_HAP_CLUSTER_SIZE = 1 after the
    # canonical-enumeration fix).
    defaults = {
        'cluster_fraction':       bhd_trio.TRIO_CLUSTER_FRACTION,
        'match_fraction':         bhd_trio.TRIO_MATCH_FRACTION,
        'distinct_fraction':      bhd_trio.TRIO_DISTINCT_FRACTION,
        'hap_dedup_pct':          bhd_trio.TRIO_HAP_DEDUP_PCT,
        'min_hap_cluster_size':   bhd_trio.TRIO_MIN_HAP_CLUSTER_SIZE,
    }

    default_value_sets = {
        'cluster_fraction':       [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60],
        'match_fraction':         [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        'distinct_fraction':      [0.30, 0.40, 0.50, 0.60, 0.70],
        'hap_dedup_pct':          [1.0, 2.0, 3.0, 5.0],
        'min_hap_cluster_size':   [1, 2, 3, 4, 5],
    }

    # Apply per-parameter overrides
    overrides = {
        'cluster_fraction':       args.values_cluster_fraction,
        'match_fraction':         args.values_match_fraction,
        'distinct_fraction':      args.values_distinct_fraction,
        'hap_dedup_pct':          args.values_hap_dedup_pct,
        'min_hap_cluster_size':   args.values_min_hap_cluster_size,
    }
    for pname, override in overrides.items():
        if override:
            if pname == 'min_hap_cluster_size':
                default_value_sets[pname] = [int(v) for v in override.split(',')]
            else:
                default_value_sets[pname] = [float(v) for v in override.split(',')]

    # Ensure default is in each sweep set
    for pname in default_value_sets:
        if defaults[pname] not in default_value_sets[pname]:
            default_value_sets[pname] = sorted(
                default_value_sets[pname] + [defaults[pname]])

    # Skipped params
    skip_set = {s.strip() for s in args.skip.split(',') if s.strip()}
    sweep_params = [p for p in SWEEP_ORDER if p not in skip_set]

    # Parse focal
    focal = None
    if args.focal.lower() not in ('', 'none'):
        parts = args.focal.split(':')
        if len(parts) != 3:
            print(f"ERROR: --focal must be CONTIG:BLOCK:FOUNDER, got {args.focal}")
            sys.exit(1)
        focal = (parts[0], int(parts[1]), int(parts[2]))

    # ---- Header ------------------------------------------------------------
    print("=" * 78)
    print("Consolidated trio parameter sweep")
    print(f"  stage:            {STAGE}")
    print(f"  recovery thresh:  < {args.threshold}% Hamming")
    print(f"  healthy sample:   {args.healthy_sample} blocks")
    if focal:
        print(f"  focal:            {focal[0]}:{focal[1]} F{focal[2]}")
    print(f"  params to sweep:  {', '.join(sweep_params)}")
    if skip_set:
        print(f"  skipped:          {', '.join(sorted(skip_set))}")
    print(f"  bhd_trio defaults (read from module):")
    for p in SWEEP_ORDER:
        marker = " (will sweep)" if p in sweep_params else " (skipped)"
        print(f"    {p:<22s} = {defaults[p]}{marker}")
    print("=" * 78)

    # ---- Load data once ----------------------------------------------------
    print()
    contig_caches = _build_all_contig_caches()
    failure_blocks = find_failure_blocks(contig_caches, args.threshold)
    healthy_blocks = sample_healthy_blocks(
        contig_caches, failure_blocks, args.healthy_sample, seed=args.seed)
    test_blocks = list(failure_blocks) + list(healthy_blocks)
    if focal is not None:
        focal_key = (focal[0], focal[1])
        already = any((c, b) == focal_key for (c, b, _) in test_blocks)
        if not already:
            test_blocks.append((focal[0], focal[1], [focal[2]]))
            print(f"  Added focal block to test set: {focal[0]}:{focal[1]}")
    print(f"  Total test set: {len(test_blocks)} blocks")

    # ---- Run each sweep ----------------------------------------------------
    all_results = {}
    t_grand = time.time()
    for pname in sweep_params:
        param_values = default_value_sets[pname]
        default_value = defaults[pname]
        results = run_sweep(
            test_blocks, pname, param_values,
            contig_caches, args.threshold, focal=focal)
        total_recov, total_cand, total_spur, focal_recov_count = \
            report_parameter(
                results, test_blocks, param_values, default_value,
                pname, focal, args.threshold)
        # Pack per-pv totals into 3-tuples for the cross-summary
        per_pv_totals = {pv: (total_recov[pv], total_cand[pv], total_spur[pv])
                         for pv in param_values}
        all_results[pname] = {
            'values':       param_values,
            'default':      default_value,
            'totals':       per_pv_totals,
            'totals_spur':  total_spur,
            'focal':        focal_recov_count,
            'raw_results':  results,
        }
    print()
    print(f"Total sweep time: {time.time() - t_grand:.1f}s")

    # ---- Cross-parameter summary ------------------------------------------
    print_cross_summary(all_results, focal)


if __name__ == '__main__':
    main()