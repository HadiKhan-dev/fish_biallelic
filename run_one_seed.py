#!/usr/bin/env python
"""
run_one_seed.py -- run ONE (depth, seed) sweep combo in this interactive session.

Generalises run_1x_seed.py to ANY depth.  Open one interactive session per node,
give each a distinct (depth, seed), and run:

    python run_one_seed.py 1.7 4         # depth 1.7x, seed 4  (depth first, then seed)
    python run_one_seed.py               # uses the DEPTH/SEED defaults set below
    DEPTH=1.7 SEED=4 python run_one_seed.py   # or via the environment

This is a thin wrapper around the sweep driver: it calls
pedigree_depth_sweep.run_one_combo(depth, seed), so the run is exactly the same
as `python pedigree_depth_sweep.py --depth <DEPTH> --seeds <SEED>` -- same
pedigree_depth_sweep/depth<DEPTH>/seed<SEED>/ layout, shared stage-1 symlink,
resume/skip and intermediate pruning -- and the depth-accuracy figure script
picks it up with no special handling.  Re-running a combo is safe: a finished
combo is skipped, a partial one resumes from its checkpoint.

Place this next to pedigree_depth_sweep.py (the sweep project dir) and run it
from there.

ONE COMBO PER SESSION: each session must own a DISTINCT (depth, seed).  Two
processes writing the same depth<DEPTH>/seed<SEED>/ at once will corrupt it (the
skip check only runs at startup).  A partial combo resumes cleanly from its
checkpoint on whichever single session you hand it -- just keep each session's
(depth, seed) distinct.
"""
import os
import sys

# ============================================================================
# THE TWO KNOBS for THIS session.  Override positionally
# (`run_one_seed.py <depth> <seed>`) or via the DEPTH/SEED environment vars.
# ============================================================================
DEPTH = 1.7          # read depth, e.g. 1.7
SEED  = 4            # replicate seed, e.g. 4

# Optional overrides so you don't have to edit the file on every node.
# Positional form is `<depth> <seed>`; either may be omitted to fall back to the
# default above (or to the DEPTH / SEED environment variable).
_argv = sys.argv[1:]
if len(_argv) >= 1:
    DEPTH = float(_argv[0])
elif os.environ.get("DEPTH"):
    DEPTH = float(os.environ["DEPTH"])
if len(_argv) >= 2:
    SEED = int(_argv[1])
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
    print("[run_one_seed] depth=%sx  seed=%d" % (sweep.depth_label(DEPTH), SEED))
    print("[run_one_seed] checkpoints: %s" % ckpt)
    print("[run_one_seed] results:     %s" % out)
    print("[run_one_seed] numba cache: %s" % _cache)
    print("=" * 70)

    ok = sweep.run_one_combo(DEPTH, SEED, sim_pipeline, dry_run=False)

    print("[run_one_seed] depth=%sx seed=%d: %s"
          % (sweep.depth_label(DEPTH), SEED, "DONE" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()