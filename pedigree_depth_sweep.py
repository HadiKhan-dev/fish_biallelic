#!/usr/bin/env python3
# =============================================================================
# pedigree_depth_sweep.py
# =============================================================================
# Driver for the read-depth x seed sweep used to make the pedigree-inference
# accuracy figure.  Runs the SELF-CONTAINED pipeline snapshot
# `pedigree_sim_pipeline.py` (stages 2-11) once per seed, in isolated
# per-(depth, seed) directories.  Neither this file nor the snapshot has any
# runtime dependency on pipeline.py -- you can delete pipeline.py and both
# still run.
#
# WHAT IT DOES (one cluster node == one depth)
#   For the depth set below, for each seed in SEEDS:
#     * make isolated checkpoint + results dirs under
#       SWEEP_ROOT/depth<D>/seed<S>/,
#     * symlink the existing .pipeline_checkpoints/01_vcf_discovery in
#       read-only so stage 1 (founders) is reused, never recomputed,
#     * run pedigree_sim_pipeline.py with that combo's seed/depth/dirs via
#       environment variables, as a fresh subprocess (so memory is released
#       between seeds),
#     * skip a seed whose stage 11 is already complete; a half-finished seed
#       resumes from its last completed stage (the pipeline's own checkpointing).
#   Each run leaves ground_truth_pedigree.csv and pedigree_inference_discovered.csv
#   in that combo's results/ -- all the figure script will need.
#
# ISOLATION / PARALLELISM
#   Different depths (different nodes) write to different subtrees and never
#   collide.  The only shared path is .pipeline_checkpoints/01_vcf_discovery,
#   which is read-only here, so concurrent nodes are safe.
#
# USAGE (next to pedigree_sim_pipeline.py and the project modules):
#     python pedigree_depth_sweep.py                 # uses READ_DEPTH below
#     python pedigree_depth_sweep.py --depth 3       # CLI override (e.g. SLURM)
#     python pedigree_depth_sweep.py --depth 0.2
#     python pedigree_depth_sweep.py --dry-run       # set up dirs, don't run
#   Re-running is safe: finished seeds are skipped, partial seeds resume.
# =============================================================================

import os
import sys
import shutil
import argparse
import subprocess
from datetime import datetime

# =============================================================================
# CONFIG  -- set the depth for THIS node here (one node == one depth).
# =============================================================================
READ_DEPTH = 5.0                      # <-- CHANGE PER NODE: 5, 3, 2, 1, 0.5, 0.2
SEEDS = [0, 1, 2, 3, 4]               # five replicate simulations per depth

SWEEP_ROOT_NAME = "pedigree_depth_sweep"   # new top-level dir for ALL sweep data
SOURCE_CHECKPOINTS = ".pipeline_checkpoints"   # reuse stage 1 (read-only) from here
STAGE1_NAME = "01_vcf_discovery"           # the seed/depth-independent stage to reuse
STAGE11_NAME = "11_pedigree_inference"     # completion marker = this stage's _done
SIM_PIPELINE_FILE = "pedigree_sim_pipeline.py"   # the self-contained stages-2-11 file

CONTINUE_ON_FAILURE = True            # if one seed fails, still attempt the rest

# --- storage: prune throwaway intermediate checkpoints once a combo is COMPLETE ---
# Verified safe against the pipeline's _KEY_SOURCE map: block haps live in stage 5
# (the [05,04,03] fallback makes 5 supersede 4 and 3), and each assembly level only
# feeds the next (L1->L2->L3->L4), so once L4 (stage 9) exists L1/L2/L3 (06/07/08)
# are dead and the stage-10 painting has already been consumed by stage 11.
# Pruning happens ONLY after a combo's stage 11 is done -- i.e. after the post-stage-9
# block-haps-vs-truth validations (which reload L1..L4) have already run -- so it is
# an end-state cleanup, never mid-run. Kept: 02 (simulation), 05 (block haps),
# 09 (L4), plus 11 (tiny; carries the completion marker) and the results/ CSVs.
PRUNE_INTERMEDIATES = True             # set False to keep every stage
PRUNE_STAGES = (
    "03_block_haplotypes",
    "04_refinement",
    "06_assembly_L1",
    "07_assembly_L2",
    "08_assembly_L3",
    "10_viterbi_painting",
)

# Environment-variable names the snapshot pipeline reads.
ENV_CKPT = "BHD_SWEEP_CKPT_DIR"
ENV_OUT = "BHD_SWEEP_OUTPUT_DIR"
ENV_SEED = "BHD_SWEEP_SEED"
ENV_DEPTH = "BHD_SWEEP_DEPTH"

# Absolute project dir = directory this file lives in (with the snapshot + modules).
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Helpers
# =============================================================================
def depth_label(depth):
    """Filesystem-safe depth tag: 5.0->'5', 3.0->'3', 0.5->'0.5', 0.2->'0.2'."""
    return "%g" % depth


def combo_paths(depth, seed):
    """Return (combo_ckpt_dir, combo_out_dir) absolute paths for one combo."""
    base = os.path.join(PROJECT_DIR, SWEEP_ROOT_NAME,
                        "depth" + depth_label(depth), "seed%d" % seed)
    return os.path.join(base, "checkpoints"), os.path.join(base, "results")


def source_stage1_dir():
    return os.path.join(PROJECT_DIR, SOURCE_CHECKPOINTS, STAGE1_NAME)


def verify_prereqs():
    """Abort early if the snapshot pipeline or the stage-1 checkpoint is missing."""
    sim = os.path.join(PROJECT_DIR, SIM_PIPELINE_FILE)
    if not os.path.isfile(sim):
        raise SystemExit(
            f"[BHD-SWEEP] cannot find {sim}. Keep pedigree_sim_pipeline.py next "
            f"to this driver.")
    s1 = source_stage1_dir()
    if not os.path.isdir(s1) or not os.path.exists(os.path.join(s1, "_done")):
        raise SystemExit(
            f"[BHD-SWEEP] existing stage-1 checkpoint not found at {s1} "
            f"(expected a '_done' file). Run the pipeline at least through "
            f"stage 1 once, then re-run this sweep.")
    return sim


def link_stage1(combo_ckpt):
    """Symlink the existing stage-1 checkpoint into this combo's checkpoint dir.

    Read-only reuse: the combo never writes to stage 1 (it is marked done), so
    pointing at the shared copy is safe even across concurrent nodes.
    """
    dst = os.path.join(combo_ckpt, STAGE1_NAME)
    if os.path.lexists(dst):
        return  # already linked (resume case)
    os.symlink(source_stage1_dir(), dst)


def combo_is_complete(combo_ckpt):
    return os.path.exists(os.path.join(combo_ckpt, STAGE11_NAME, "_done"))


def _dir_size_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def prune_intermediates(combo_ckpt, dry_run=False):
    """Delete throwaway intermediate stage dirs for a COMPLETED combo only.

    Safety: returns immediately unless the combo's stage 11 is done (so every
    stage, including the post-stage-9 validations that reload L1..L4, has
    finished), never follows the stage-1 symlink, and only ever touches the
    explicit PRUNE_STAGES list.
    """
    if not PRUNE_INTERMEDIATES:
        return
    if not combo_is_complete(combo_ckpt):
        return  # never prune an incomplete combo
    freed = 0
    for name in PRUNE_STAGES:
        path = os.path.join(combo_ckpt, name)
        if os.path.islink(path) or not os.path.isdir(path):
            continue
        sz = _dir_size_bytes(path)
        if dry_run:
            print(f"             [prune/dry-run] would remove {name} (~{sz/1e9:.1f} GB)")
            continue
        shutil.rmtree(path)
        freed += sz
        print(f"             [prune] removed {name} (~{sz/1e9:.1f} GB)")
    if freed and not dry_run:
        print(f"             [prune] freed ~{freed/1e9:.1f} GB "
              f"(kept 02/05/09 + 11 + results/)")


# =============================================================================
# Run one combo
# =============================================================================
def run_one_combo(depth, seed, sim_pipeline, dry_run=False):
    """Run (or resume) stages 2..11 for one (depth, seed). Returns True on success."""
    combo_ckpt, combo_out = combo_paths(depth, seed)

    if combo_is_complete(combo_ckpt):
        print(f"[BHD-SWEEP] depth {depth_label(depth)} seed {seed}: "
              f"already complete (stage 11 done) -- skipping.")
        prune_intermediates(combo_ckpt, dry_run=dry_run)
        return True

    os.makedirs(combo_ckpt, exist_ok=True)
    os.makedirs(combo_out, exist_ok=True)
    link_stage1(combo_ckpt)

    env = os.environ.copy()
    env[ENV_CKPT] = combo_ckpt
    env[ENV_OUT] = combo_out
    env[ENV_SEED] = str(seed)
    env[ENV_DEPTH] = repr(float(depth))
    # Make the project modules importable regardless of how sys.path[0] resolves.
    env["PYTHONPATH"] = PROJECT_DIR + os.pathsep + env.get("PYTHONPATH", "")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(os.path.dirname(combo_ckpt), f"run_{ts}.log")

    print(f"[BHD-SWEEP] depth {depth_label(depth)} seed {seed}: launching stages 2-11")
    print(f"             checkpoints: {combo_ckpt}")
    print(f"             results:     {combo_out}")
    print(f"             log:         {log_path}")

    if dry_run:
        print(f"             [dry-run] would run: {sys.executable} "
              f"{sim_pipeline}  (cwd={PROJECT_DIR})")
        return True

    with open(log_path, "w") as logf:
        logf.write(f"[BHD-SWEEP] depth={depth} seed={seed} started {ts}\n")
        logf.flush()
        proc = subprocess.run(
            [sys.executable, sim_pipeline],
            cwd=PROJECT_DIR,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )

    if proc.returncode == 0 and combo_is_complete(combo_ckpt):
        print(f"[BHD-SWEEP] depth {depth_label(depth)} seed {seed}: DONE.")
        prune_intermediates(combo_ckpt, dry_run=dry_run)
        return True

    print(f"[BHD-SWEEP] depth {depth_label(depth)} seed {seed}: FAILED "
          f"(exit code {proc.returncode}; stage-11 marker "
          f"{'present' if combo_is_complete(combo_ckpt) else 'absent'}). "
          f"See log: {log_path}")
    return False


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Run the pedigree-inference depth sweep for ONE depth "
                    "across seeds 0-4 (stages 2-11 only).")
    ap.add_argument("--depth", type=float, default=None,
                    help=f"read depth for this node (default {READ_DEPTH})")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help=f"seeds to run (default {SEEDS})")
    ap.add_argument("--dry-run", action="store_true",
                    help="set up dirs + stage-1 symlinks, but launch nothing")
    args = ap.parse_args()

    depth = args.depth if args.depth is not None else READ_DEPTH
    seeds = args.seeds if args.seeds is not None else SEEDS

    print("=" * 70)
    print(f"[BHD-SWEEP] read depth = {depth_label(depth)}x   seeds = {seeds}")
    print(f"[BHD-SWEEP] project dir = {PROJECT_DIR}")
    print(f"[BHD-SWEEP] sweep root  = {os.path.join(PROJECT_DIR, SWEEP_ROOT_NAME)}")
    print("=" * 70)

    sim_pipeline = verify_prereqs()

    results = {}
    for seed in seeds:
        ok = run_one_combo(depth, seed, sim_pipeline, dry_run=args.dry_run)
        results[seed] = ok
        if not ok and not CONTINUE_ON_FAILURE:
            break

    print("\n" + "=" * 70)
    print(f"[BHD-SWEEP] summary for depth {depth_label(depth)}x:")
    for seed in seeds:
        status = results.get(seed)
        tag = "ok" if status else ("FAILED" if status is False else "skipped")
        print(f"             seed {seed}: {tag}")
    print("=" * 70)

    n_failed = sum(1 for s in results.values() if s is False)
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()