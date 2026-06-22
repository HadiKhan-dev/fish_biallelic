#!/usr/bin/env python
"""
archive_and_clean_sweep.py -- archive the small, irreplaceable sweep outputs
(the per-combo ground-truth / inferred pedigree CSVs and the depth-accuracy
figure) into a new folder, then delete the large per-combo checkpoint trees to
reclaim disk.

Layout it expects (as written by pedigree_depth_sweep.py):

    <sweep_root>/depth<D>/seed<S>/results/ground_truth_pedigree.csv
    <sweep_root>/depth<D>/seed<S>/results/pedigree_inference_discovered.csv
    <sweep_root>/depth<D>/seed<S>/checkpoints/        <- the multi-TB part, deleted
    <sweep_root>/depth<D>/seed<S>/run_*.log           <- copied too (tiny run record)
    <sweep_root>/depth_accuracy.{png,pdf}             <- the figure, copied
    <sweep_root>/depth_accuracy_summary.csv

The archive mirrors that tree minus the checkpoints, so it stays a valid input
to plot_depth_accuracy.py: `python plot_depth_accuracy.py --sweep-root <archive>`
will rebuild the figure straight from the archived CSVs.

SAFETY
  * DRY RUN BY DEFAULT. It prints exactly what it would copy and delete, with
    sizes, and removes nothing until you re-run with --apply.
  * A combo's checkpoints are deleted ONLY after both its CSVs are copied AND
    byte-size-verified in the archive. A combo whose CSVs are missing is left
    untouched (use --force-delete-unarchived to override).
  * The stage-1 checkpoint inside each combo is a SYMLINK to the shared
    .pipeline_checkpoints/01_vcf_discovery. Deletion unlinks it and never
    follows it, so the shared founders are preserved. The shared checkpoint
    directory itself is left alone unless you pass --also-shared.
  * It only ever removes directories literally named 'checkpoints' located
    inside the sweep root (plus, optionally, the one shared dir you name).

Usage (from the sweep project dir):

    python archive_and_clean_sweep.py                 # DRY RUN: show the plan
    python archive_and_clean_sweep.py --apply         # archive, verify, then delete
    python archive_and_clean_sweep.py --apply --also-shared
    python archive_and_clean_sweep.py --archive /path/to/archive --apply
"""
import argparse
import glob
import os
import shutil

RESULT_FILES = ("ground_truth_pedigree.csv", "pedigree_inference_discovered.csv")


def _dir_size(path):
    """Sum of real file sizes under path; does NOT follow symlinks."""
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.islink(fp):
                continue
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _fmt(nbytes):
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0


def discover_combos(sweep_root):
    """Return [(depth_dirname, seed_dirname, seed_path), ...] for every depth*/seed*."""
    combos = []
    for ddir in sorted(glob.glob(os.path.join(sweep_root, "depth*"))):
        if not os.path.isdir(ddir):
            continue
        for sdir in sorted(glob.glob(os.path.join(ddir, "seed*"))):
            if os.path.isdir(sdir):
                combos.append((os.path.basename(ddir), os.path.basename(sdir), sdir))
    return combos


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-root", default=os.path.join(here, "pedigree_depth_sweep"),
                    help="sweep tree with depth*/seed*/{results,checkpoints} "
                         "(default: ./pedigree_depth_sweep next to this script)")
    ap.add_argument("--archive", default=None,
                    help="destination folder for the kept CSVs + figure "
                         "(default: <sweep-root>_archive, a sibling of the sweep root)")
    ap.add_argument("--apply", action="store_true",
                    help="actually copy and DELETE; without it this is a dry run")
    ap.add_argument("--also-shared", action="store_true",
                    help="ALSO delete the shared .pipeline_checkpoints dir (the "
                         "reusable stage-1 founders); off by default so future "
                         "sweeps can still reuse it")
    ap.add_argument("--shared-dir", default=os.path.join(here, ".pipeline_checkpoints"),
                    help="path of the shared checkpoint dir (only deleted with "
                         "--also-shared)")
    ap.add_argument("--force-delete-unarchived", action="store_true",
                    help="delete a combo's checkpoints even if its result CSVs were "
                         "not found/archived (NOT recommended)")
    args = ap.parse_args()

    sweep_root = os.path.abspath(args.sweep_root)
    archive = os.path.abspath(args.archive) if args.archive else sweep_root + "_archive"

    if not os.path.isdir(sweep_root):
        raise SystemExit("[clean] sweep root not found: %s" % sweep_root)
    # The archive must live OUTSIDE the sweep tree (else we'd copy into a tree we
    # are about to prune).
    if os.path.commonpath([archive, sweep_root]) == sweep_root:
        raise SystemExit(
            "[clean] --archive must be outside --sweep-root.\n"
            "        sweep-root: %s\n        archive:    %s" % (sweep_root, archive))

    combos = discover_combos(sweep_root)
    if not combos:
        raise SystemExit("[clean] no depth*/seed* combos under %s" % sweep_root)

    print("[clean] %s" % ("APPLY" if args.apply else "DRY RUN"))
    print("[clean] sweep root: %s" % sweep_root)
    print("[clean] archive:    %s" % archive)
    print("[clean] %d combo(s) found\n" % len(combos))

    # ---- 1. archive CSVs (+ run logs), recording which combos are safe to wipe
    plan = []                 # (seed_path, ckpt_dir, ckpt_bytes, archived_ok, n_csv)
    total_ckpt_bytes = 0
    for ddir, sdir, seed_path in combos:
        res_src = os.path.join(seed_path, "results")
        arch_res = os.path.join(archive, ddir, sdir, "results")
        csvs = [f for f in RESULT_FILES if os.path.isfile(os.path.join(res_src, f))]
        ok = len(csvs) == len(RESULT_FILES)

        if csvs and args.apply:
            os.makedirs(arch_res, exist_ok=True)
            for f in csvs:
                src, dst = os.path.join(res_src, f), os.path.join(arch_res, f)
                shutil.copy2(src, dst)
                if not (os.path.isfile(dst) and os.path.getsize(dst) > 0 and
                        os.path.getsize(dst) == os.path.getsize(src)):
                    raise SystemExit(
                        "[clean] ARCHIVE VERIFY FAILED for %s -- aborting before any "
                        "deletion." % dst)
            # tiny run logs: the only record of the runs once checkpoints are gone
            for lg in sorted(glob.glob(os.path.join(seed_path, "run_*.log"))):
                shutil.copy2(lg, os.path.join(archive, ddir, sdir, os.path.basename(lg)))

        ckpt_dir = os.path.join(seed_path, "checkpoints")
        ckpt_bytes = _dir_size(ckpt_dir) if os.path.isdir(ckpt_dir) else 0
        total_ckpt_bytes += ckpt_bytes
        plan.append((seed_path, ckpt_dir, ckpt_bytes, ok, len(csvs)))
        print("  %s/%s: %d/%d CSV(s), checkpoints %s%s"
              % (ddir, sdir, len(csvs), len(RESULT_FILES), _fmt(ckpt_bytes),
                 "" if ok else "   <-- results MISSING"))

    # ---- 2. copy the figure(s) from the sweep-root top level
    graphs = [g for g in sorted(glob.glob(os.path.join(sweep_root, "depth_accuracy*")))
              if os.path.isfile(g)]
    print("\n[clean] figure files: %s"
          % (", ".join(os.path.basename(g) for g in graphs) or "(none found)"))
    if args.apply and graphs:
        os.makedirs(archive, exist_ok=True)
        for g in graphs:
            shutil.copy2(g, os.path.join(archive, os.path.basename(g)))

    # ---- 3. decide + report deletions
    deletable = [(sp, cd, b) for (sp, cd, b, ok, n) in plan
                 if os.path.isdir(cd) and (ok or args.force_delete_unarchived)]
    skipped = [(cd, b) for (sp, cd, b, ok, n) in plan
               if os.path.isdir(cd) and not ok and not args.force_delete_unarchived]

    print("\n[clean] total checkpoint size: %s" % _fmt(total_ckpt_bytes))
    if skipped:
        print("[clean] WARNING: %d combo(s) have missing CSVs and will be KEPT "
              "(use --force-delete-unarchived to remove):" % len(skipped))
        for cd, b in skipped:
            print("          %s  (%s)" % (cd, _fmt(b)))

    if not args.apply:
        n_arch = sum(1 for p in plan if p[3])
        print("\n[clean] DRY RUN -- nothing copied or deleted.")
        print("[clean] would archive %d combo(s)' CSVs + %d figure file(s) to %s"
              % (n_arch, len(graphs), archive))
        print("[clean] would delete %d checkpoint tree(s), freeing ~%s"
              % (len(deletable), _fmt(sum(b for _, _, b in deletable))))
        if args.also_shared:
            print("[clean] would ALSO delete shared %s" % os.path.abspath(args.shared_dir))
        print("[clean] re-run with --apply to do it.")
        return

    # ---- 4. delete (archive is complete + verified by here)
    freed = 0
    for sp, cd, b in deletable:
        assert os.path.basename(cd) == "checkpoints"                       # only 'checkpoints'
        assert os.path.commonpath([os.path.abspath(cd), sweep_root]) == sweep_root
        assert not os.path.islink(cd)
        for entry in os.scandir(cd):          # unlink the stage-1 symlink first; never follow it
            if entry.is_symlink():
                os.unlink(entry.path)
        shutil.rmtree(cd)
        freed += b
        print("  deleted %s  (freed %s)" % (cd, _fmt(b)))

    if args.also_shared:
        shared = os.path.abspath(args.shared_dir)
        if os.path.isdir(shared) and not os.path.islink(shared):
            sb = _dir_size(shared)
            shutil.rmtree(shared)
            freed += sb
            print("  deleted shared %s  (freed %s)" % (shared, _fmt(sb)))
        else:
            print("  [also-shared] not a directory, skipped: %s" % shared)

    print("\n[clean] DONE. archived to %s; freed ~%s." % (archive, _fmt(freed)))
    if skipped:
        print("[clean] kept %d unarchived combo(s)' checkpoints." % len(skipped))


if __name__ == "__main__":
    main()