"""Process set-up: numerical-library configuration at package import, run settings at run start.

Package import (`configure_imports`, called once from ``haplotype_reconstruction/__init__.py``): BLAS/OpenMP
library pools get one thread each (all parallelism is explicit: Numba teams and process pools); Numba's math stays on
scalar libm (no Intel SVML, release 2.1: machine-independent results); Numba's threading
layer is omp with passive waiting (the measured production baseline; workqueue threads spin and tbb was never the
production layer); Numba gets a cache directory that holds only kernels of this exact package source (so the
kernel cache can persist them between runs). These settings must precede the first NumPy/Numba import to take
effect, which is why they are the package's one sanctioned import-time effect. Child processes inherit them
through the environment.

Run start (`start_run`, called by the driver before any pool exists): Numba's pool size = the run's cores
(bounded by affinity), the forkserver preload list, the profiling flag and the run's memory governor
(execution.memory). Pools and executors forward the run settings and the governor's handle to their workers
(`worker_settings` / `apply_worker_settings`).
"""
from __future__ import annotations

from dataclasses import dataclass
import getpass
import hashlib
import multiprocessing
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import uuid

from ..store.identity import PACKAGE_ROOT, add_sources
from .resources import affinity_cpus

NUMERIC_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

THREADING_LAYER = "omp"

# Modules every forkserver worker imports before it is forked (copy-on-write); the driver adds the stages'.
BASE_PRELOAD = ("numpy", "numba", "scipy")

# NUMBA_* settings that do not change compiled machine code (where it is cached, how many threads run it and
# which threading layer runs it): excluded from the code key.
_RUNTIME_NUMBA_SETTINGS = frozenset((
    "NUMBA_CACHE_DIR", "NUMBA_NUM_THREADS", "NUMBA_THREADING_LAYER",
))

# Private markers through which child processes inherit this process tree's choices.
_CODE_KEY_MARKER = "_HAPLOTYPES_NUMBA_CODE_KEY"
_CACHE_BASE_MARKER = "_HAPLOTYPES_NUMBA_CACHE_BASE"
_CACHE_DIR_MARKER = "_HAPLOTYPES_NUMBA_CACHE_DIR"
_CACHE_RUN_MARKER = "_HAPLOTYPES_NUMBA_CACHE_RUN"

def configure_imports() -> None:
    """Configure the numerical libraries before any numerical module is imported (package import)."""
    for variable in NUMERIC_THREAD_VARIABLES:
        os.environ[variable] = "1"
    os.environ["NUMBA_THREADING_LAYER"] = THREADING_LAYER
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    # Release 2.1: Numba's math functions stay on scalar libm even where an Intel SVML runtime is installed (SVML's
    # vector kernels can differ from libm by an ulp and by CPU), so results do not depend on the machine.
    os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
    _configure_numba_cache()
    numba_config = sys.modules.get("numba.core.config")
    if numba_config is not None:
        # Numba was imported first (a library caller): apply the settings to its loaded configuration;
        # a threading layer that already runs is left alone (it cannot be replaced).
        numba_config.reload_config()
        if not _threading_layer_started():
            numba_config.THREADING_LAYER = THREADING_LAYER


def _threading_layer_started() -> bool:
    numba = sys.modules.get("numba")
    if numba is None:
        return False
    try:
        numba.threading_layer()
    except ValueError:
        return False
    return True


def _batch_cache_scope() -> str | None:
    """A scheduler job (and array task) identifier for the automatic cache directory, if any."""
    schedulers = (
        ("slurm", "SLURM_JOB_ID", ("SLURM_ARRAY_TASK_ID",)),
        ("pbs", "PBS_JOBID", ("PBS_ARRAY_INDEX", "PBS_ARRAYID")),
        ("lsf", "LSB_JOBID", ("LSB_JOBINDEX",)),
    )
    if os.environ.get("SGE_ROOT"):
        schedulers += (("sge", "JOB_ID", ("SGE_TASK_ID",)),)
    for name, variable, task_variables in schedulers:
        job = os.environ.get(variable)
        if job:
            task = next((os.environ[key] for key in task_variables if os.environ.get(key)), None)
            return f"{name}-{job}" + (f"-task-{task}" if task else "")
    return None


def _host_cpu_model() -> str:
    """CPU vendor, model and instruction-set flags (Linux), else the platform name."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            first = handle.read(1 << 16).split("\n\n", 1)[0]
    except OSError:
        import platform
        return f"{platform.machine()} {platform.processor()}"
    fields = {name.strip(): value.strip() for name, value in
              (line.split(":", 1) for line in first.splitlines() if ":" in line)}
    return " | ".join(fields.get(name, "") for name in ("vendor_id", "model name", "flags"))


def numba_code_key() -> str:
    """Key of every input of this package's compiled kernels (24 hex digits).

    Numba's own disk cache checks only each kernel's source-file timestamp, so a kernel whose callee or imported
    constant changed in another file would load stale machine code. The key covers the whole package source and
    its location (Numba's cache layout depends on it), the Python, NumPy, Numba and llvmlite versions, the
    code-generation NUMBA_* settings and the host CPU model. Computed once per process tree; children inherit it.
    """
    root = str(PACKAGE_ROOT)
    inherited_root, _, inherited_key = os.environ.get(_CODE_KEY_MARKER, "").rpartition("|")
    if inherited_key and inherited_root == root:
        return inherited_key
    from importlib import metadata
    digest = hashlib.sha256()

    def add(*values):
        for value in values:
            digest.update(str(value).encode("utf-8", "surrogateescape") + b"\0")

    add(root, sys.version, sys.implementation.cache_tag, getattr(sys, "abiflags", ""))
    for package in ("numba", "llvmlite", "numpy"):
        try:
            add(package, metadata.version(package))
        except metadata.PackageNotFoundError:
            add(package, None)
    for name in sorted(os.environ):
        if name.startswith("NUMBA_") and name not in _RUNTIME_NUMBA_SETTINGS:
            add(name, os.environ[name])
    add(_host_cpu_model())
    add_sources(digest, PACKAGE_ROOT)
    key = digest.hexdigest()[:24]
    os.environ[_CODE_KEY_MARKER] = f"{root}|{key}"
    return key


def _configure_numba_cache() -> str:
    """Give Numba a cache directory that only ever holds kernels of this package source.

    The directory is <base>/<numba_code_key()>: base is the user's NUMBA_CACHE_DIR, otherwise a scratch
    directory isolated by scheduler job (or run) and host. Markers distinguish our inherited setting from a new
    user one. These are disposable compiled kernels, not scientific checkpoints.
    """
    explicit = os.environ.get("NUMBA_CACHE_DIR")
    if explicit and explicit != os.environ.get(_CACHE_DIR_MARKER):
        os.environ[_CACHE_BASE_MARKER] = explicit
    base = os.environ.get(_CACHE_BASE_MARKER)
    if base:
        directory = Path(base)
    else:
        run_id = os.environ.get(_CACHE_RUN_MARKER)
        if not run_id:
            run_id = uuid.uuid4().hex
            os.environ[_CACHE_RUN_MARKER] = run_id
        scope = _batch_cache_scope() or f"run-{run_id}"
        user = str(os.getuid()) if hasattr(os, "getuid") else getpass.getuser()
        user = re.sub(r"[^A-Za-z0-9_.-]", "_", user)
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{scope}-{socket.gethostname()}")
        scratch = Path((os.environ.get("SLURM_TMPDIR") if os.environ.get("SLURM_JOB_ID") else None)
                       or tempfile.gettempdir())
        directory = scratch / f"haplotype-reconstruction-numba-{user}" / name
    directory = directory / numba_code_key()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache = str(directory)
    os.environ["NUMBA_CACHE_DIR"] = cache
    os.environ[_CACHE_DIR_MARKER] = cache
    return cache


@dataclass(frozen=True)
class RunSettings:
    """Execution settings of one run, forwarded to every worker process."""

    cores: int
    profile: bool = False


_RUN: RunSettings | None = None


def start_run(cores: int, *, preload: tuple[str, ...] = (), profile: bool = False,
              memory_limit: int | None = None) -> RunSettings:
    """Apply a run's execution settings before any pool or executor exists.

    ``cores`` (<= affinity) becomes Numba's pool size here and, through the environment, in every child
    process, and the calling thread's Numba team. ``preload`` adds stage modules to the forkserver preload list
    (effective only before the forkserver starts, i.e. before the first pool). ``memory_limit`` (bytes; None =
    the memory available now minus headroom) caps the run's memory governor, created here.
    """
    global _RUN
    cores = int(cores)
    available = affinity_cpus()
    if not 1 <= cores <= available:
        raise ValueError(f"cores must be between 1 and the CPU affinity ({available}), got {cores}")
    os.environ["NUMBA_NUM_THREADS"] = str(cores)
    numba = sys.modules.get("numba")
    if numba is not None and not _threading_layer_started():
        numba.config.NUMBA_NUM_THREADS = cores
    try:
        multiprocessing.set_forkserver_preload(list(dict.fromkeys((*BASE_PRELOAD, *preload))))
    except (AttributeError, RuntimeError, ValueError):
        pass
    _RUN = RunSettings(cores=cores, profile=bool(profile))
    from .memory import start_governor
    start_governor(memory_limit)
    import numba as numba_module
    numba_module.set_num_threads(min(cores, numba_module.config.NUMBA_NUM_THREADS))
    return _RUN


def worker_settings() -> tuple[RunSettings | None, str]:
    """What a pool or executor passes to its workers: the run settings (None outside a run) and the handle of
    the memory governor of this process tree (see apply_worker_settings)."""
    from .memory import governor_handle
    return _RUN, governor_handle()


def apply_worker_settings(settings: tuple[RunSettings | None, str | None]) -> None:
    """In a freshly started worker process: adopt the parent's run settings and memory governor."""
    global _RUN
    _RUN, memory_handle = settings
    from .memory import attach_governor
    attach_governor(memory_handle)


def profiling() -> bool:
    """Whether the run asked for diagnostic timing/thread traces ([execution] profile)."""
    return _RUN is not None and _RUN.profile
