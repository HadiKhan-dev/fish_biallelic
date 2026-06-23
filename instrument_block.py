#!/usr/bin/env python3
"""
instrument_block.py -- single-block serialization profiler for BHD discovery.

Loads ONE heavy 100 kbp block exactly as pipeline.py stage 1 loads it, pins
numba to all cores (the lone-straggler-on-a-full-node scenario), runs the REAL
generate_haplotypes_block_robust IN THIS PROCESS, and reports, per Python phase,
how much wall time it costs and how parallel it actually was.

The parallelism signal
----------------------
For each wrapped Python function we record inclusive + exclusive (self) WALL
time (perf_counter) and CPU time (process_time). process_time() is the whole
process's CPU summed across ALL threads -- including numba's native workqueue
threads, which py-spy / cProfile cannot see without --native. So:

    parallelism = cpu_time / wall_time
      ~1   -> that phase ran serially (one core busy)        <<< the bottleneck
      ~N   -> that phase kept N cores busy (good parallelism)

A phase with large SELF wall and self-parallelism ~1 is a serial section that
threads cannot help -- that is exactly what we are hunting.

The orchestration inside a block is single-threaded Python (parallelism comes
only from numba kernels), so the enter/exit CPU deltas partition cleanly. If
the harness ever sees concurrent Python threads it says so (the CPU split would
then be unreliable).

Run
---
    # straight profile (uses the in-process phase profiler):
    python instrument_block.py --block 59

    # clean numbers (drops one-time numba compile cost; ~2x runtime):
    python instrument_block.py --block 59 --warmup

    # also expose the process for a py-spy --native deep dive:
    python instrument_block.py --block 59 --pause 20
    #   ...then, in another shell, using the PID it prints:
    #   py-spy record --pid <PID> --native --threads --idle \
    #       --rate 50 --duration 450 --format speedscope -o block_solo.json
"""

import os
import sys
import time
import types
import inspect
import functools
import argparse
import threading
import multiprocessing


# --------------------------------------------------------------------------
# Args + environment (NUMBA_NUM_THREADS must be set BEFORE numba is imported)
# --------------------------------------------------------------------------
def _detect_cores():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def parse_args():
    ap = argparse.ArgumentParser(
        description="Single-block BHD serialization profiler")
    ap.add_argument("--vcf",
                    default="./fish_vcf_restriped/AsAc.AulStuGenome.biallelic.bcf.gz",
                    help="path to the VCF/BCF (default: the pipeline.py path)")
    ap.add_argument("--contig", default="chr1")
    ap.add_argument("--block", type=int, default=59,
                    help="block index -- matches the [block-time] id= value")
    ap.add_argument("--threads", type=int, default=None,
                    help="numba threads to pin (default: all available cores)")
    ap.add_argument("--block-size", type=int, default=100000)
    ap.add_argument("--shift-size", type=int, default=50000)
    ap.add_argument("--warmup", action="store_true",
                    help="run once untimed first to drop numba compile time")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds to pause before the timed run (attach py-spy)")
    ap.add_argument("--top", type=int, default=30,
                    help="number of rows in the phase table")
    ap.add_argument("--code-dir", default=None,
                    help="dir holding block_haplotypes.py etc "
                         "(default: cwd + this script's dir)")
    return ap.parse_args()


ARGS = parse_args()
N_THREADS = ARGS.threads or _detect_cores()
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ["NUMBA_NUM_THREADS"] = str(N_THREADS)   # cap; set before import numba

for _d in [ARGS.code_dir, os.getcwd(),
           os.path.dirname(os.path.abspath(__file__))]:
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)

import numba                     # noqa: E402
numba.set_num_threads(N_THREADS)

import vcf_data_loader           # noqa: E402
import block_haplotypes          # noqa: E402
import dynamic_threads           # noqa: E402


# --------------------------------------------------------------------------
# Phase profiler: inclusive + exclusive wall AND cpu per wrapped function
# --------------------------------------------------------------------------
class PhaseProfiler:
    def __init__(self):
        self._local = threading.local()
        self._lock = threading.Lock()
        self.stats = {}              # label -> [n, self_w, self_c, inc_w, inc_c]
        self._active_threads = set()
        self.concurrent = False

    def _stack(self):
        st = getattr(self._local, "stack", None)
        if st is None:
            st = []
            self._local.stack = st
        return st

    def enter(self, label):
        st = self._stack()
        if not st:                                   # entering top-level
            tid = threading.get_ident()
            with self._lock:
                self._active_threads.add(tid)
                if len(self._active_threads) > 1:
                    self.concurrent = True
        # [label, wall0, cpu0, child_wall, child_cpu]
        st.append([label, time.perf_counter(), time.process_time(), 0.0, 0.0])

    def exit(self):
        st = self._stack()
        label, w0, c0, cw, cc = st.pop()
        iw = time.perf_counter() - w0               # inclusive wall
        ic = time.process_time() - c0               # inclusive cpu (all threads)
        sw = iw - cw                                 # self wall
        sc = ic - cc                                 # self cpu
        with self._lock:
            rec = self.stats.get(label)
            if rec is None:
                rec = [0, 0.0, 0.0, 0.0, 0.0]
                self.stats[label] = rec
            rec[0] += 1
            rec[1] += sw
            rec[2] += sc
            rec[3] += iw
            rec[4] += ic
        if st:                                       # fold into parent
            st[-1][3] += iw
            st[-1][4] += ic
        else:
            with self._lock:
                self._active_threads.discard(threading.get_ident())


def _make_wrapper(orig, label, prof):
    @functools.wraps(orig)
    def wrapper(*a, **k):
        prof.enter(label)
        try:
            return orig(*a, **k)
        finally:
            prof.exit()
    return wrapper


def _collect_bhd_modules():
    mods = []
    for name, m in list(sys.modules.items()):
        if m is None:
            continue
        if name == "block_haplotypes" or name.startswith("bhd"):
            mods.append(m)
    return mods


def install_profiler(prof, modules):
    """Wrap every pure-Python function/method DEFINED in the given modules.

    numba dispatchers are not FunctionType, so njit kernels are skipped
    automatically -- their time rolls up into the self-time of the Python
    function that called them (and that self-parallelism is what reveals
    whether the kernel actually used the cores). Each function is wrapped
    once and ALL references to it across the modules are repointed at the
    wrapper, so `from x import y` aliases are intercepted too.
    """
    # 1. module-level functions (dedup by identity; keep defining module)
    funcs = {}                                   # id -> (func, label)
    for mod in modules:
        modname = getattr(mod, "__name__", "")
        for attr, val in list(vars(mod).items()):
            if attr.startswith("__"):
                continue
            if (isinstance(val, types.FunctionType)
                    and getattr(val, "__module__", None) == modname):
                funcs.setdefault(id(val),
                                 (val, "%s.%s" % (modname.split(".")[-1], attr)))
    # 2. class methods
    methods = {}                                 # id -> (func, cls, name, label)
    for mod in modules:
        modname = getattr(mod, "__name__", "")
        for cattr, cval in list(vars(mod).items()):
            if not isinstance(cval, type):
                continue
            if getattr(cval, "__module__", None) != modname:
                continue
            for mattr, mval in list(vars(cval).items()):
                if mattr.startswith("__"):
                    continue
                if (isinstance(mval, types.FunctionType)
                        and getattr(mval, "__module__", None) == modname):
                    methods.setdefault(id(mval),
                                       (mval, cval, mattr,
                                        "%s.%s" % (cval.__name__, mattr)))

    wrapped = 0
    func_ids = set(funcs)
    for fid, (func, label) in funcs.items():
        w = _make_wrapper(func, label, prof)
        for mod in modules:                      # repoint every reference
            for attr, val in list(vars(mod).items()):
                if val is func:
                    setattr(mod, attr, w)
        wrapped += 1
    for mid, (func, cls, name, label) in methods.items():
        if mid in func_ids:                      # already wrapped as a function
            continue
        setattr(cls, name, _make_wrapper(func, label, prof))
        wrapped += 1
    return wrapped


# --------------------------------------------------------------------------
# Faithful block load + kwargs (exactly what pipeline.py stage 1 uses)
# --------------------------------------------------------------------------
DISCOVERY_KEYS = [
    "lambda_wildcard_penalty", "wildcard_mass_threshold",
    "min_wildcard_relative_improvement", "K_max", "coord_descent_max_iter",
    "min_supporters_for_confidence", "recovery_max_K", "recovery_mixture_K_max",
    "recovery_mixture_patience", "diff_threshold_percent", "penalty_strength",
    "chimera_max_recombs", "chimera_max_mismatch_pct",
    "chimera_min_delta_to_protect", "uniqueness_threshold_percent",
    "wrongness_threshold", "max_intermediate_haps",
]


def build_discovery_kwargs():
    """Pull the discovery defaults straight from generate_all_block_haplotypes'
    signature -- the pipeline calls it with only num_processes set, so every
    discovery parameter takes its default. Introspecting keeps us faithful
    even if the defaults change."""
    sig = inspect.signature(block_haplotypes.generate_all_block_haplotypes)
    kw = {}
    for k in DISCOVERY_KEYS:
        p = sig.parameters.get(k)
        if p is not None and p.default is not inspect.Parameter.empty:
            kw[k] = p.default
    return kw


def load_block(args):
    gd = vcf_data_loader.cleanup_block_reads_list(
        args.vcf, args.contig,
        start_block_idx=args.block, end_block_idx=args.block + 1,
        block_size=args.block_size, shift_size=args.shift_size,
        num_processes=1,
    )
    if len(gd) != 1:
        raise SystemExit("expected 1 block, loader returned %d" % len(gd))
    positions, reads, flags = gd[0]          # GenomicData -> (pos, reads, flags)
    return positions, reads, flags


def wire_dynamic_threads(n_threads):
    """Wire dynamic_threads with active=1 so get_dynamic_threads() = total/1 =
    n_threads -- i.e. the block runs on all cores the whole time, like the last
    straggler block in a draining pool. Without this, the un-wired sequential
    path forces every kernel to 1 thread."""
    active = multiprocessing.Value("i", 0)
    extra = multiprocessing.Value("i", 0)
    dynamic_threads.set_dynamic_thread_state(n_threads, active, extra)
    dynamic_threads.increment_active()       # active -> 1
    return active, extra


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def report(prof, top, n_cores, total_wall, total_cpu):
    with prof._lock:
        rows = [(lab, n, sw, sc, iw, ic)
                for lab, (n, sw, sc, iw, ic) in prof.stats.items()]
    rows.sort(key=lambda r: r[2], reverse=True)        # by self_wall desc

    print()
    print("=" * 100)
    print("PER-PHASE BREAKDOWN  (self = time in this function's own code, "
          "excluding wrapped callees)")
    print("parallelism = cpu/wall:  ~1.0 == serial (one core)   ...   ~%d "
          "== all cores busy" % n_cores)
    if prof.concurrent:
        print("WARNING: concurrent Python threads were detected -- the cpu/wall "
              "split is unreliable here.")
    print("=" * 100)
    hdr = ("%-44s %7s %9s %9s %9s %9s"
           % ("phase", "calls", "self_s", "self_par", "incl_s", "incl_par"))
    print(hdr)
    print("-" * len(hdr))

    serial_flag = max(2.0, 0.03 * total_wall)
    shown = 0
    for lab, n, sw, sc, iw, ic in rows:
        if shown >= top:
            break
        spar = sc / sw if sw > 1e-9 else 0.0
        ipar = ic / iw if iw > 1e-9 else 0.0
        flag = "  <<< SERIAL" if (sw >= serial_flag and spar < 1.8) else ""
        print("%-44s %7d %9.2f %9.1f %9.2f %9.1f%s"
              % (lab[:44], n, sw, spar, iw, ipar, flag))
        shown += 1
    print("-" * len(hdr))

    tot_self = sum(r[2] for r in rows)
    print("sum(self_s) = %.1f s   (should ~= total wall %.1f s; any gap is "
          "time in unwrapped / njit-only code)" % (tot_self, total_wall))
    print("overall: wall=%.1f s  cpu=%.1f s  parallelism=%.1fx of %d cores"
          % (total_wall, total_cpu, total_cpu / max(total_wall, 1e-9), n_cores))
    print()
    print("How to read it:")
    print("  * Largest self_s with self_par ~1  ->  serial bottleneck threads "
          "can't fix (the target).")
    print("  * Largest self_s with self_par ~%d ->  already parallel; leave it "
          "alone." % n_cores)
    print("  * incl_par on a high-level phase shows how parallel its whole "
          "subtree was.")
    if not ARGS.warmup:
        print("  * NOTE: no --warmup, so a phase called very few times may look "
              "serial purely")
        print("    because its first call paid one-time numba compilation. "
              "Re-run with --warmup")
        print("    (≈2x runtime) to confirm any single-call SERIAL flag.")


# --------------------------------------------------------------------------
def main():
    args = ARGS
    print("[harness] pid=%d  contig=%s  block=%d  threads=%d"
          % (os.getpid(), args.contig, args.block, N_THREADS))
    print("[harness] NUMBA_THREADING_LAYER=%s  NUMBA_NUM_THREADS=%s"
          % (os.environ.get("NUMBA_THREADING_LAYER"),
             os.environ.get("NUMBA_NUM_THREADS")))

    t = time.perf_counter()
    positions, reads, flags = load_block(args)
    n_sites = len(positions)
    n_reads = reads.shape[0] if hasattr(reads, "shape") else len(reads)
    print("[harness] loaded block in %.2f s  reads=%s sites=%s  reads.shape=%s"
          % (time.perf_counter() - t, n_reads, n_sites,
             getattr(reads, "shape", None)))
    if n_sites == 0:
        raise SystemExit("block has no sites -- pick another --block")

    kwargs = build_discovery_kwargs()
    print("[harness] discovery kwargs: %s" % (kwargs,))

    wire_dynamic_threads(N_THREADS)
    applied = dynamic_threads.apply_dynamic_threads()
    print("[harness] dynamic_threads applied=%d  numba.get_num_threads()=%d "
          "(expect %d)" % (applied, numba.get_num_threads(), N_THREADS))

    # Warmup BEFORE installing the profiler so compile time isn't attributed.
    if args.warmup:
        print("[harness] warmup run (untimed, drops numba compile cost) ...")
        tw = time.perf_counter()
        block_haplotypes.generate_haplotypes_block_robust(
            positions, reads, keep_flags=flags, **kwargs)
        print("[harness] warmup done in %.1f s" % (time.perf_counter() - tw))

    prof = PhaseProfiler()
    n_wrapped = install_profiler(prof, _collect_bhd_modules())
    print("[harness] wrapped %d python functions/methods across bhd modules"
          % n_wrapped)

    if args.pause > 0:
        dur = int(max(60, 1.25 * (400 if args.block == 59 else 300)))
        print("[harness] pausing %.0f s -- attach py-spy now (PID %d):"
              % (args.pause, os.getpid()))
        print("  py-spy record --pid %d --native --threads --idle --rate 50 "
              "--duration %d --format speedscope -o block_solo.json"
              % (os.getpid(), dur))
        time.sleep(args.pause)

    print("[harness] running generate_haplotypes_block_robust (timed) ...")
    w0 = time.perf_counter()
    c0 = time.process_time()
    block_haplotypes.generate_haplotypes_block_robust(
        positions, reads, keep_flags=flags, **kwargs)
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0
    print("[harness] DONE  wall=%.2f s  cpu=%.2f s  overall_parallelism=%.1fx"
          % (wall, cpu, cpu / max(wall, 1e-9)))

    report(prof, args.top, N_THREADS, wall, cpu)


if __name__ == "__main__":
    main()