"""Run logging, the checkpoint store and lightweight workflow utilities."""
from __future__ import annotations


from contextlib import contextmanager
from functools import wraps
import sys

import os
from pathlib import Path

from datetime import datetime


FOUNDER_BLOCK_KEY = "founder_block"


PHASE_COMPONENTS_KEY = "phase_components"


PHASE_COMPONENT_MANIFEST_SCHEMA = "ordered_phase_components_v1"


def available_cpu_count():
    """Return CPUs available to this process, respecting Slurm affinity."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    return max(1, int(count))


def _inactive_file_bytes(directory, version1):
    """Reclaimable inactive file cache of a cgroup (hierarchical total in v1)."""
    key = "total_inactive_file" if version1 else "inactive_file"
    try:
        for line in (directory / "memory.stat").read_text().splitlines():
            name, _, value = line.partition(" ")
            if name == key:
                return int(value)
    except (OSError, ValueError):
        pass
    return 0


def available_memory_bytes():
    """Respect the current nested Slurm cgroup and every enclosing memory cap."""
    candidates = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    try:
        entries = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        entries = []
    for entry in entries:
        _, controllers, relative = entry.split(":", 2)
        if "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup/memory")
            maximum_name, current_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
        elif not controllers:
            root = Path("/sys/fs/cgroup")
            maximum_name, current_name = "memory.max", "memory.current"
        else:
            continue
        directory = root / relative.lstrip("/")
        while True:
            try:
                maximum = int((directory / maximum_name).read_text().strip())
                current = int((directory / current_name).read_text().strip())
                if 0 < maximum < (1 << 60):
                    # Inactive file cache is charged to the cgroup but reclaimed under
                    # pressure: count only the working set as used, so plans do not
                    # depend on how much recently read checkpoint data is still cached.
                    current -= _inactive_file_bytes(directory, bool(controllers))
                    candidates.append(max(0, maximum - current))
            except (OSError, ValueError):
                pass
            if directory == root:
                break
            directory = directory.parent
    slurm_mib = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mib and int(slurm_mib) > 0:
        candidates.append(int(slurm_mib) << 20)
    return min(candidates) if candidates else None


class TeeOutput:
    """Mirror stdout to a line-buffered log while proxying stream attributes."""

    def __init__(self, log_path, original_stdout):
        object.__setattr__(
            self, "_log_file", open(log_path, "a", buffering=1)
        )
        object.__setattr__(self, "_original", original_stdout)

    def write(self, message):
        self._original.write(message)
        try:
            self._log_file.write(message)
        except (ValueError, OSError):
            pass
        return None

    def flush(self):
        self._original.flush()
        try:
            self._log_file.flush()
        except (ValueError, OSError):
            pass

    def close(self):
        self._log_file.close()

    def __getattr__(self, name):
        return getattr(self._original, name)


@contextmanager
def run_log(log_path):
    """Mirror a run's stdout and restore it even on interruption or failure.

    The caller owns the surrounding run directory. Stderr remains on the
    terminal (including progress bars and native crash diagnostics).
    """
    original = sys.stdout
    stream = TeeOutput(log_path, original)
    sys.stdout = stream
    try:
        yield stream
    finally:
        sys.stdout = original
        stream.flush()
        stream.close()


def logged_workflow(label):
    """Give each workflow invocation one closed, uniquely named stdout log.

    The workflow runs with stored and warmed-up Numba kernels
    (core.numba_warmup), compiled before it starts any pool.
    """
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            directory = os.environ.get("HAPLOTYPES_LOG_DIR", "work/logs")
            os.makedirs(directory, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(directory, f"{label}_{stamp}_{os.getpid()}.log")
            from .numba_warmup import persistent_numba_cache
            from .run_record import record_run
            output = os.environ.get('HAPLOTYPES_OUTPUT_DIR', os.path.dirname(directory))
            cores = min(available_cpu_count(),
                        int(os.environ.get("BHD_NUM_PROCESSES") or available_cpu_count()))
            with run_log(path), record_run(output, label, path):
                print(f"Logging to: {path}")
                with persistent_numba_cache(cores):
                    return function(*args, **kwargs)
        return wrapped
    return decorate


CONTIG_DIRECTORY = "contigs"


WORK_DIRECTORY = "work"


RUN_FILE = "run.ckpt"


CONTIG_SUFFIX = ".ckpt"


def _path_token(value, name):
    result = str(value)
    if (not result or result in {".", ".."} or os.sep in result
            or (os.altsep is not None and os.altsep in result)):
        raise ValueError(f"{name} must be a nonempty path-component token")
    return result


class CheckpointStore:
    """A run's checkpoint directory.

    At rest it holds ``run.ckpt`` (genome-wide records), one append-only
    ``contigs/<contig>.ckpt`` per chromosome, and ``work/`` only while
    transient resume-only state is live (core.checkpoints describes the
    container format and its concurrency rules). Cheap and picklable.
    """

    def __init__(self, root, *, nthreads=1, clevel=None):
        self.root = os.fspath(root)
        self.nthreads = max(1, int(nthreads))
        self.clevel = core_checkpoints.CLEVEL if clevel is None else int(clevel)
        os.makedirs(self.root, exist_ok=True)
        for entry in os.scandir(self.root):
            if entry.is_dir() and entry.name not in (CONTIG_DIRECTORY, WORK_DIRECTORY):
                raise RuntimeError(
                    f"{self.root} holds '{entry.name}/', which is not part of this "
                    "release's checkpoint layout (run.ckpt, contigs/, work/): use a "
                    "new output root, or the release that wrote these checkpoints")

    @property
    def run(self):
        """The genome-wide container, run.ckpt."""
        return core_checkpoints.CheckpointFile(
            os.path.join(self.root, RUN_FILE), kind="run",
            nthreads=self.nthreads, clevel=self.clevel)

    def contig(self, name):
        """The container of one chromosome (created by its first put)."""
        name = _path_token(name, "contig")
        return core_checkpoints.CheckpointFile(
            os.path.join(self.root, CONTIG_DIRECTORY, name + CONTIG_SUFFIX),
            kind="contig", name=name, nthreads=self.nthreads, clevel=self.clevel)

    def contig_names(self):
        """Chromosomes that have a container, sorted by name."""
        try:
            names = os.listdir(os.path.join(self.root, CONTIG_DIRECTORY))
        except FileNotFoundError:
            return ()
        return tuple(sorted(name[:-len(CONTIG_SUFFIX)] for name in names
                            if name.endswith(CONTIG_SUFFIX) and not name.startswith(".")))

    def bind_identity(self, file, section, identity):
        """Bind a section of a container to one exact model/configuration record.

        The first binder records ``identity/<section>``; later runs must match
        it exactly (canonical JSON), else a clear error names the section.
        """
        try:
            file.put_once(f"identity/{section}", identity)
        except core_checkpoints.CheckpointMismatch:
            raise RuntimeError(
                f"{file.label}: checkpoint identity '{section}' does not match the "
                "current model/configuration; use a new output root") from None

    def bind_manifest(self, key, payload):
        """Publish one JSON-compatible run manifest exactly once (manifest/<key>)."""
        if not isinstance(payload, dict) or not payload:
            raise ValueError("a run manifest must be a nonempty dictionary")
        try:
            self.run.put_once(f"manifest/{key}", payload)
        except core_checkpoints.CheckpointMismatch:
            raise RuntimeError(
                f"manifest '{key}' does not match the current run; use a new "
                "output root") from None

    def stage_complete(self, stage):
        return self.run.has(f"complete/{stage}")

    def mark_stage_complete(self, stage):
        if not self.stage_complete(stage):
            self.run.put_once(f"complete/{stage}", True)
            print(f"  [Checkpoint] Stage '{stage}' marked complete")

    def work(self, contig, name):
        """Transient resume-only state of one contig step (see WorkState)."""
        return WorkState(self.root, contig, name, nthreads=self.nthreads)


class WorkState:
    """Resume-only state of one contig step: ``work/<contig>/<name>.p5.b2``.

    Saved by atomic, durable replacement; cleared (with any directories it
    leaves empty) once its step's product is recorded, so a completed run has
    no ``work/`` directory.
    """

    def __init__(self, root, contig, name, *, nthreads=1):
        self.root = os.fspath(root)
        self.contig = _path_token(contig, "contig")
        self.name = _path_token(name, "work state name")
        self.nthreads = max(1, int(nthreads))

    @property
    def path(self):
        return os.path.join(self.root, WORK_DIRECTORY, self.contig,
                            self.name + core_checkpoints.PAYLOAD_SUFFIX)

    def exists(self):
        return os.path.exists(self.path)

    def save(self, payload):
        """Replace the state atomically; return its size in bytes."""
        return core_checkpoints.write(self.path, payload, nthreads=self.nthreads,
                                      clevel=core_checkpoints.FAST_CLEVEL)

    def load(self):
        """The saved state, or None when there is none."""
        try:
            return core_checkpoints.read(self.path, nthreads=self.nthreads)
        except FileNotFoundError:
            return None

    def clear(self):
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        directory = os.path.dirname(self.path)
        for path in (directory, os.path.dirname(directory)):
            try:
                os.rmdir(path)
            except OSError:
                return


def require_records(store, key, contigs):
    """Require ``key`` in every listed contig's container (e.g. before completion)."""
    missing = [str(contig) for contig in contigs if not store.contig(contig).has(key)]
    if missing:
        raise OSError(f"Failed to checkpoint {key}: " + ", ".join(missing))


def strip_block_evidence(blocks):
    """Drop block-local evidence unused after supported checkpoint boundaries."""
    for block in blocks:
        if hasattr(block, "probs_array"):
            block.probs_array = None
        if hasattr(block, "reads_count_matrix"):
            block.reads_count_matrix = None
    return blocks


def _phase_component_record(component_id, block):
    """Build the verifiable metadata for one component-local founder panel."""
    import numpy as np

    positions = np.asarray(getattr(block, "positions", None))
    haplotypes = getattr(block, "haplotypes", None)
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("phase components must contain positions")
    if positions.size > 1 and np.any(positions[1:] <= positions[:-1]):
        raise ValueError("phase-component positions must be strictly increasing")
    if not haplotypes:
        raise ValueError("phase components must contain founder haplotypes")
    founder_ids = tuple(sorted(haplotypes))
    return {
        "component_id": int(component_id),
        FOUNDER_BLOCK_KEY: block,
        "component_local_founder_ids": founder_ids,
        "position_start": int(positions[0]),
        "position_end": int(positions[-1]),
        "position_count": int(positions.size),
        "break_before": bool(
            getattr(block, "missing_aware_break_before", False)
        ),
        "break_after": bool(
            getattr(block, "missing_aware_break_after", False)
        ),
        "break_reason_before": getattr(
            block, "missing_aware_break_reason_before", None
        ),
        "break_reason_after": getattr(
            block, "missing_aware_break_reason_after", None
        ),
        "joint_informative_samples_before": getattr(
            block, "missing_aware_joint_informative_samples_before", None
        ),
        "joint_informative_samples_after": getattr(
            block, "missing_aware_joint_informative_samples_after", None
        ),
    }


def create_phase_component_manifest(blocks):
    """Create an ordered, self-contained manifest from nonempty components."""
    components = tuple(blocks)
    if not components:
        raise ValueError("phase-component collections must be nonempty")
    manifest = {
        "schema": PHASE_COMPONENT_MANIFEST_SCHEMA,
        PHASE_COMPONENTS_KEY: tuple(
            _phase_component_record(index, block)
            for index, block in enumerate(components)
        ),
    }
    validate_phase_component_manifest(manifest)
    return manifest


def validate_phase_component_manifest(manifest):
    """Validate ordering and component-local identities; return the blocks."""
    if not isinstance(manifest, dict):
        raise TypeError("phase-component manifest must be a dictionary")
    if manifest.get("schema") != PHASE_COMPONENT_MANIFEST_SCHEMA:
        raise ValueError("phase-component manifest schema mismatch")
    records = manifest.get(PHASE_COMPONENTS_KEY)
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("phase-component manifest must contain components")

    blocks = []
    previous_end = None
    for expected_id, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError("phase-component records must be dictionaries")
        if record.get("component_id") != expected_id:
            raise ValueError("phase-component IDs must be consecutive and ordered")
        if FOUNDER_BLOCK_KEY not in record:
            raise KeyError("phase-component record lacks founder_block")
        block = record[FOUNDER_BLOCK_KEY]
        expected = _phase_component_record(expected_id, block)
        for key, value in expected.items():
            if key == FOUNDER_BLOCK_KEY:
                continue
            if record.get(key) != value:
                raise ValueError(
                    f"phase-component record disagrees with block field {key}"
                )
        if previous_end is not None and expected["position_start"] <= previous_end:
            raise ValueError("phase components must be strictly position ordered")
        previous_end = expected["position_end"]
        blocks.append(block)
    return tuple(blocks)

import haplotype_reconstruction.core.checkpoints as core_checkpoints
