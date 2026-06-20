#!/usr/bin/env python
"""compare_sim_across_seeds.py -- are the stage-2 simulated inputs identical across seeds?

The pedigree inference came out byte-identical across seeds at a given depth, even
though the read RNG is seeded per-seed.  This decides WHY:

  * if the stage-2 simulated arrays (simulated_reads / simd_probs / simd_genomic_data)
    are IDENTICAL across seeds  -> the inference input is seed-invariant: an UPSTREAM
    simulation / checkpoint-isolation problem (the per-seed reads aren't really
    per-seed), so re-running won't help until that's fixed;

  * if those arrays DIFFER across seeds but the inferred pedigree is identical
    -> the INFERENCE is ignoring the genotype data at this depth (a degenerate
    low-coverage fixed point), i.e. a genuine "no signal below 2x" result.

It loads <sweep-root>/depth<D>/seed<S>/checkpoints/02_simulation/<contig>.pkl for each
seed and md5s each numpy array it can find, then reports whether they match across seeds.
Read-only; prints to stdout and (optionally) --out.

    python compare_sim_across_seeds.py --sweep-root /path/to/pedigree_depth_sweep --depth 1
    python compare_sim_across_seeds.py --sweep-root ... --depth 0.2 --out cmp_d02.txt
"""
import argparse, glob, hashlib, os, pickle
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--sweep-root", required=True)
ap.add_argument("--depth", default="1", help="depth label as on disk, e.g. 1 or 0.2")
ap.add_argument("--ckpt-name", default="checkpoints",
                help="per-seed checkpoint dir name (default: checkpoints)")
ap.add_argument("--stage", default="02_simulation",
                help="checkpoint stage subdir to compare (default: 02_simulation)")
ap.add_argument("--keys", default="simulated_reads,simd_probs,simd_priors,simd_genomic_data",
                help="comma-separated checkpoint keys to compare (first found per file wins "
                     "for the headline verdict)")
ap.add_argument("--out", default=None)
args = ap.parse_args()

keys = [k.strip() for k in args.keys.split(",") if k.strip()]
ddir = os.path.join(args.sweep_root, "depth" + args.depth)
seed_dirs = sorted(glob.glob(os.path.join(ddir, "seed*")),
                   key=lambda p: (int(os.path.basename(p)[4:])
                                  if os.path.basename(p)[4:].isdigit() else 10**9))
if not seed_dirs:
    raise SystemExit("no seed*/ under %s" % ddir)

out_path = args.out or ("cmp_sim_depth%s.txt" % args.depth)
_fh = open(out_path, "w")
def emit(line=""):
    print(line); _fh.write(line + "\n")


def arr_md5(obj):
    """md5 of a numpy array's raw bytes; None if obj isn't array-like."""
    try:
        a = np.ascontiguousarray(obj)
        if a.dtype == object:
            return None
        return hashlib.md5(a.tobytes()).hexdigest()[:12] + " shape=%s dtype=%s" % (a.shape, a.dtype)
    except Exception:
        return None


emit("# compare_sim_across_seeds.py   depth=%s   stage=%s" % (args.depth, args.stage))
emit("# sweep-root: %s" % os.path.abspath(args.sweep_root))

# contig pkl basenames present under seed0's stage dir (use as the comparison set)
first_stage_dir = os.path.join(seed_dirs[0], args.ckpt_name, args.stage)
contigs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(first_stage_dir, "*.pkl")))
if not contigs:
    emit("\nNo %s/*.pkl under %s" % (args.stage, first_stage_dir))
    emit("  -> the stage-2 arrays may have been removed from disk, or the per-seed checkpoint")
    emit("     dir name differs (try --ckpt-name).  Listing what IS there:")
    for sdir in seed_dirs:
        cd = os.path.join(sdir, args.ckpt_name)
        sub = sorted(glob.glob(os.path.join(cd, "*"))) if os.path.isdir(cd) else []
        emit("    %s/%s : %s" % (os.path.basename(sdir), args.ckpt_name,
                                 [os.path.basename(x) for x in sub] or "(absent)"))
    _fh.close(); raise SystemExit(0)

verdict_any_key = {}
for contig in contigs:
    emit("\n" + "=" * 72)
    emit("=== contig pkl: %s ===" % contig)
    # per key: {seed_name: md5str}
    per_key = {k: {} for k in keys}
    for sdir in seed_dirs:
        sname = os.path.basename(sdir)
        path = os.path.join(sdir, args.ckpt_name, args.stage, contig)
        if not os.path.isfile(path):
            emit("  %s: MISSING %s" % (sname, path)); continue
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
        except Exception as exc:                       # noqa: BLE001
            emit("  %s: could NOT unpickle (%s: %s)" % (sname, type(exc).__name__, exc))
            emit("       (likely a class moved in a refactor; run this from the project dir "
                 "in bio-env so the original modules import)")
            continue
        present = [k for k in keys if isinstance(d, dict) and k in d]
        if not present:
            emit("  %s: none of the requested keys present; keys on disk = %s"
                 % (sname, list(d.keys()) if isinstance(d, dict) else type(d)))
            continue
        for k in present:
            h = arr_md5(d[k])
            per_key[k][sname] = h if h is not None else "(non-array)"
    # report per key
    for k in keys:
        hashes = per_key[k]
        if not hashes:
            continue
        vals = [v for v in hashes.values() if v and v != "(non-array)"]
        emit("  key '%s':" % k)
        for sname, h in hashes.items():
            emit("    %-8s %s" % (sname, h))
        if len(vals) > 1:
            same = len(set(v.split(" ")[0] for v in vals)) == 1
            emit("    -> %s across seeds" % ("IDENTICAL" if same else "DIFFERENT"))
            verdict_any_key.setdefault(k, []).append(same)

emit("\n" + "=" * 72)
emit("=== VERDICT (depth %s) ===" % args.depth)
if not verdict_any_key:
    emit("  No comparable arrays were loaded -- see messages above.")
else:
    for k, sames in verdict_any_key.items():
        allsame = all(sames)
        emit("  '%s': %s across seeds on all %d contig(s)"
             % (k, "IDENTICAL" if allsame else "DIFFERENT", len(sames)))
    any_identical = any(all(s) for s in verdict_any_key.values())
    emit("")
    if any_identical:
        emit("  => Stage-2 simulated input is IDENTICAL across seeds. The per-seed reads are")
        emit("     not actually per-seed (UPSTREAM simulation / checkpoint-isolation issue).")
        emit("     The identical inference is a consequence of identical input, not the method.")
    else:
        emit("  => Stage-2 simulated input DIFFERS across seeds, yet the inferred pedigree was")
        emit("     identical -> the INFERENCE is not using the genotype data at this depth")
        emit("     (degenerate low-coverage solution). That's a real 'no signal' regime.")
_fh.close()
print("\nWrote: %s" % os.path.abspath(out_path))