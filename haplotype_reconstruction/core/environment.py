"""Import-time numerical-library limits and supported environment choices."""
from __future__ import annotations


import os
import json
import getpass
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


def assembly_transition_model():
    """Assembly choice is independent of discovery; dense is the default."""
    value = os.environ.get("HAPLOTYPES_ASSEMBLY_MODEL", "dense")
    if value not in ("dense", "structured"):
        raise ValueError("HAPLOTYPES_ASSEMBLY_MODEL must be dense or structured")
    return value


def assembly_panel_search():
    """Panel-search breadth is independent of the assembly transition model."""
    value = os.environ.get("HAPLOTYPES_ASSEMBLY_SEARCH", "bounded")
    if value not in ("bounded", "broad"):
        raise ValueError("HAPLOTYPES_ASSEMBLY_SEARCH must be bounded or broad")
    return value


def assembly_founder_refinement():
    """Reopen local row choices after each final L1-L4 level by default."""
    value = os.environ.get("HAPLOTYPES_FOUNDER_REFINEMENT", "on")
    if value not in ("on", "off"):
        raise ValueError("HAPLOTYPES_FOUNDER_REFINEMENT must be on or off")
    return value == "on"


def block_feedback_selection():
    """Normalized path selection is the default for local founder panels."""
    value = os.environ.get("HAPLOTYPES_FEEDBACK_SELECTION", "path")
    if value not in ("path", "balanced", "strict"):
        raise ValueError("HAPLOTYPES_FEEDBACK_SELECTION must be path, balanced or strict")
    return value


def block_feedback_segment_exchange():
    """Optional same-count segment proposals after both feedback rounds."""
    return boolean_setting(os.environ.get("HAPLOTYPES_FEEDBACK_SEGMENT_EXCHANGE", "off"),
                           "feedback_segment_exchange")


def batched_discovery_enabled():
    """The experimental discovery search requires its own explicit choice."""
    value = os.environ.get("HAPLOTYPES_DISCOVERY_SEARCH", "standard")
    if value not in ("standard", "batched"):
        raise ValueError("HAPLOTYPES_DISCOVERY_SEARCH must be standard or batched")
    return value == "batched"


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


def configure_numba_cache():
    """Isolate native caches by run/job and host, without splitting its workers.

    The automatic path marker distinguishes our inherited setting from a user
    override. Spawned processes reuse the run ID, but recalculate the path so
    a remote host or a different batch allocation cannot inherit our old cache.
    These are disposable compiled kernels, not scientific checkpoints.
    """
    explicit = os.environ.get("NUMBA_CACHE_DIR")
    automatic = os.environ.get("_HAPLOTYPES_NUMBA_CACHE_DIR")
    if explicit and explicit != automatic:
        cache = explicit
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
