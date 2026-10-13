"""Import-time numerical-library limits and supported environment choices."""
from __future__ import annotations


import os
import json
import getpass
import hashlib
import re
from pathlib import Path
import socket
import sys
import tempfile
import uuid


NUMERIC_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def boolean_setting(value, name):
    """Parse the same boolean spellings in CLI, TOML and environment adapters."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        setting = value.strip().lower()
        if setting in ('1', 'true', 'yes', 'on'):
            return True
        if setting in ('0', 'false', 'no', 'off'):
            return False
    raise ValueError(f'{name} must be boolean (1/0, true/false, yes/no or on/off)')


# Removed alternatives (one join route, one discovery search, path feedback).
# Each entry: setting name -> (the single remaining value, why the others went).
OBSOLETE_SETTINGS = {
    "assembly_model": ("dense", "the structured transition model was removed; the dense linker is "
                                "the only assembly transition model"),
    "assembly_search": ("bounded", "the broad assembly search was removed: it capped panels at 12 "
                                   "founders"),
    "discovery_search": ("standard", "the batched discovery search was removed"),
    "feedback_selection": ("path", "the balanced/strict feedback selections were removed; path "
                                   "selection is the only feedback selection"),
    "feedback_segment_exchange": (False, "feedback segment exchange was removed"),
}
_WARNED_OBSOLETE = set()


def obsolete_setting(name, value, source):
    """Accept a removed option only at its remaining value (warn once); reject anything else."""
    import warnings
    remaining, reason = OBSOLETE_SETTINGS[name]
    if isinstance(remaining, bool):
        accepted = boolean_setting(value, name) is remaining
    else:
        accepted = str(value).strip().lower() == remaining
    if not accepted:
        raise ValueError(f"{source} {name} = {value!r}: {reason}; delete this setting")
    if (source, name) not in _WARNED_OBSOLETE:
        _WARNED_OBSOLETE.add((source, name))
        warnings.warn(f"{source} {name} is obsolete and ignored ({reason}); delete this setting",
                      UserWarning, stacklevel=3)


def check_obsolete_environment():
    """Validate removed options still present in the environment (warn once / clear error)."""
    for name in OBSOLETE_SETTINGS:
        variable = "HAPLOTYPES_" + name.upper()
        if variable in os.environ:
            obsolete_setting(name, os.environ[variable], f"environment variable {variable}:")


def assembly_founder_refinement():
    """Reopen local row choices after each final L1-L4 level by default."""
    value = os.environ.get("HAPLOTYPES_FOUNDER_REFINEMENT", "on")
    if value not in ("on", "off"):
        raise ValueError("HAPLOTYPES_FOUNDER_REFINEMENT must be on or off")
    return value == "on"


def coarse_move_guard():
    """Coarse-move guard of the converged founder search; "shared_junction" by default."""
    value = os.environ.get("HAPREC_COARSE_MOVE_GUARD", "shared_junction")
    if value not in ("shared_junction", "genotype_fit", "none"):
        raise ValueError("HAPREC_COARSE_MOVE_GUARD must be shared_junction, genotype_fit or none")
    return value


def configured_regions(default, *, template_regions=False):
    requested = os.environ.get("HAPLOTYPES_CONTIGS")
    if requested is None:
        return default
    names = json.loads(requested)
    if not names or len(names) != len(set(names)):
        raise ValueError("contigs must be a nonempty unique ordered list")
    return [dict(contig=str(name), **({'start': 0, 'end': 3000} if template_regions else {}))
            for name in names]



def force_single_threaded_numeric_libraries():
    """Force BLAS/OpenMP libraries to one thread before importing them."""

    for variable in NUMERIC_THREAD_ENV_VARS:
        os.environ[variable] = "1"


def _batch_cache_scope():
    """Recognize common batch jobs; unknown launchers use an inherited run ID."""
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
            task = next((os.environ[key] for key in task_variables
                         if os.environ.get(key)), None)
            return f"{name}-{job}" + (f"-task-{task}" if task else "")
    return None


# NUMBA_* settings that do not change compiled machine code: where it is
# cached, how many threads run it and which threading layer runs it.
_RUNTIME_NUMBA_SETTINGS = frozenset((
    "NUMBA_CACHE_DIR", "NUMBA_NUM_THREADS", "NUMBA_THREADING_LAYER",
    "NUMBA_THREADING_LAYER_OVERRIDE",
))


def _host_cpu_model():
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


def numba_code_key():
    """Key of every input of this package's compiled Numba kernels.

    Numba's own disk cache checks only the timestamp of each kernel's source
    file, so a kernel whose callee or imported constant changed in another file
    would load stale machine code. This key covers the whole package source and
    its location (Numba's cache layout depends on it), the Python, NumPy, Numba
    and llvmlite versions, the code-generation NUMBA_* settings and the host CPU
    model: any edit gives a new key. It is computed once per process tree; child
    processes inherit it together with the package location.
    """
    root = str(Path(__file__).resolve().parents[1])
    inherited_root, _, inherited_key = os.environ.get(
        "_HAPLOTYPES_NUMBA_CODE_KEY", "").rpartition("|")
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
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = sorted(name for name in subdirectories if name != "__pycache__")
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(directory, name)
                add(os.path.relpath(path, root))
                with open(path, "rb") as handle:
                    digest.update(handle.read())
    key = digest.hexdigest()[:24]
    os.environ["_HAPLOTYPES_NUMBA_CODE_KEY"] = f"{root}|{key}"
    return key


def configure_numba_cache():
    """Give Numba a run-local cache directory for exactly this compiled code.

    The directory is <base>/<numba_code_key()>: base is the user's
    NUMBA_CACHE_DIR, otherwise a directory isolated by run/job and host. A
    directory therefore only ever holds kernels of one package source, so
    core.numba_warmup can seed it from, and publish it to, a persistent store.
    Markers distinguish our inherited setting (and the user's base, kept for
    child processes) from a new user override. Spawned processes reuse the run
    ID, but recalculate the automatic path so a remote host or a different batch
    allocation cannot inherit our old cache. These are disposable compiled
    kernels, not scientific checkpoints.
    """
    explicit = os.environ.get("NUMBA_CACHE_DIR")
    if explicit and explicit != os.environ.get("_HAPLOTYPES_NUMBA_CACHE_DIR"):
        os.environ["_HAPLOTYPES_NUMBA_CACHE_BASE"] = explicit
    base = os.environ.get("_HAPLOTYPES_NUMBA_CACHE_BASE")
    if base:
        directory = Path(base)
    else:
        run_id = os.environ.get("_HAPLOTYPES_NUMBA_CACHE_RUN")
        if not run_id:
            run_id = uuid.uuid4().hex
            os.environ["_HAPLOTYPES_NUMBA_CACHE_RUN"] = run_id
        job_id = os.environ.get("SLURM_JOB_ID")
        scope = _batch_cache_scope() or f"run-{run_id}"
        user = str(os.getuid()) if hasattr(os, "getuid") else getpass.getuser()
        # Host/user identifiers are path components, including on non-POSIX OSes.
        user = re.sub(r"[^A-Za-z0-9_.-]", "_", user)
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{scope}-{socket.gethostname()}")
        scratch = Path((os.environ.get("SLURM_TMPDIR") if job_id else None)
                       or tempfile.gettempdir())
        directory = scratch / f"haplotype-reconstruction-numba-{user}" / name
    directory = directory / numba_code_key()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache = str(directory)
    os.environ["NUMBA_CACHE_DIR"] = cache
    os.environ["_HAPLOTYPES_NUMBA_CACHE_DIR"] = cache

    # Library callers and forkserver preloads may import Numba first. Apply
    # our setting before our numerical modules create cached dispatchers,
    # without eagerly importing Numba for ordinary CLI/help startup.
    numba_config = sys.modules.get("numba.core.config")
    if numba_config is not None:
        numba_config.reload_config()
    return cache
