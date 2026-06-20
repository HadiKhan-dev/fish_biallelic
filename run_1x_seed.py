#!/usr/bin/env python
"""
run_1x_seed.py -- run ONE 1x-depth sweep seed in this interactive session.

Open one interactive session per node, give each a different seed, and run:

    python run_1x_seed.py            # uses the SEED set below
    python run_1x_seed.py 3          # or override on the command line
    SEED=3 python run_1x_seed.py     # or via the environment

So for "one of each seed 1-4", run `python run_1x_seed.py 1` in session 1,
`... 2` in session 2, and so on -- no need to edit the file on each node.

This is a thin wrapper around the existing sweep driver: it calls
pedigree_depth_sweep.run_one_combo(depth=1.0, seed=SEED), so the run is exactly
the same as `python pedigree_depth_sweep.py --depth 1 --seeds <SEED>` -- same
pedigree_depth_sweep/depth1/seed<N>/ layout, stage-1 symlink, resume/skip and
pruning -- and the depth-accuracy figure script picks it up with no special
handling. Re-running a seed is safe: a finished seed is skipped, a partial one
resumes from its checkpoint.

Place this next to pedigree_depth_sweep.py (the sweep project dir) and run it
from there.

COLLISION WARNING: the original 1x sweep node was launched with the full
SEEDS=[0,1,2,3,4] and will keep marching through 2,3,4 after its current seed.
Two processes writing the same depth1/seed<N>/ at once will corrupt it (the
skip check only runs at startup). Since you're taking over seeds 1-4 across
sessions, cancel that original 1x job FIRST -- its in-progress seed (e.g. seed
1) then resumes cleanly from its checkpoint on whichever session you give it.
Keep each session's seed distinct.
"""
import os
import sys

# ============================================================================
# THE ONE KNOB: which seed THIS session runs.  Change per session (1, 2, 3, 4),
# or pass it as `python run_1x_seed.py <seed>` / `SEED=<seed> python ...`.
# ============================================================================
SEED = 1
DEPTH = 1.0          # 1x depth (fixed for this launcher)

# Optional overrides so you don't have to edit the file on every node.
if len(sys.argv) > 1:
    SEED = int(sys.argv[1])
elif os.environ.get("SEED"):
    SEED = int(os.environ["SEED"])

# Make the sweep driver importable when run from its own directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pedigree_depth_sweep as sweep

# Node-local numba cache: stops the intermittent "can't unbox array from
# PyObject" glitch from racing on the shared filesystem, and avoids a cold-cache
# compile storm.  Set BEFORE run_one_combo so it reaches the pipeline subprocess
# (run_one_combo forwards os.environ).  Delete these three lines to use your
# usual numba cache location instead.
_cache = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                      "numba_cache_d%s_s%d" % (sweep.depth_label(DEPTH), SEED))
os.makedirs(_cache, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = _cache


def main():
    sim_pipeline = sweep.verify_prereqs()    # checks sim pipeline + shared stage-1
    ckpt, out = sweep.combo_paths(DEPTH, SEED)
    print("=" * 70)
    print("[run_1x_seed] depth=%sx  seed=%d" % (sweep.depth_label(DEPTH), SEED))
    print("[run_1x_seed] checkpoints: %s" % ckpt)
    print("[run_1x_seed] results:     %s" % out)
    print("[run_1x_seed] numba cache: %s" % _cache)
    print("=" * 70)

    ok = sweep.run_one_combo(DEPTH, SEED, sim_pipeline, dry_run=False)

    print("[run_1x_seed] depth=%sx seed=%d: %s"
          % (sweep.depth_label(DEPTH), SEED, "DONE" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()