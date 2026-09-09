"""core / runtime for the canonical reconstruction pipeline."""
from __future__ import annotations


import copy

import json
import os
import tempfile

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


class CheckpointStore:
    """Facade for atomic pipeline checkpoints with optional stage binding.

    Writes are atomic, and filesystem ``OSError`` exceptions are reported then
    re-raised so a stage cannot be marked complete after losing a checkpoint.
    """

    def __init__(self, root, *, nthreads=1, global_log_indent="  "):
        self.root = os.fspath(root)
        self.nthreads = max(1, int(nthreads))
        self.global_log_indent = global_log_indent
        os.makedirs(self.root, exist_ok=True)

    def stage_dir(self, stage):
        path = os.path.join(self.root, stage)
        os.makedirs(path, exist_ok=True)
        return path

    def bind_stage_identity(self, stage, identity):
        """Bind a stage directory to one exact model/configuration record.

        Existing outputs without an identity are rejected rather than being
        retroactively treated as current. Concurrent first binders publish a
        fully written sidecar with an atomic hard-link claim.
        """

        stage_path = os.path.join(self.root, stage)
        identity_path = os.path.join(stage_path, "_identity.json")
        expected_text = json.dumps(
            copy.deepcopy(identity),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        def verify_existing():
            try:
                with open(identity_path, "r", encoding="utf-8") as handle:
                    observed = json.load(handle)
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    f"{stage}: checkpoint identity is unreadable"
                ) from error
            observed_text = json.dumps(
                observed,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if observed_text != expected_text:
                raise RuntimeError(
                    f"{stage}: checkpoint identity does not match the "
                    "current model/configuration"
                )

        if os.path.exists(identity_path):
            verify_existing()
            return

        os.makedirs(stage_path, exist_ok=True)
        existing = [
            name for name in os.listdir(stage_path)
            if name != "_identity.json"
            and not (
                name.startswith("_identity.")
                and name.endswith(".tmp")
            )
        ]
        if existing:
            if os.path.exists(identity_path):
                verify_existing()
                return
            raise RuntimeError(
                f"{stage}: existing checkpoints lack the required "
                "model/configuration identity"
            )

        descriptor, temporary_path = tempfile.mkstemp(
            prefix="_identity.", suffix=".tmp", dir=stage_path
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(expected_text)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_path, identity_path)
            except FileExistsError:
                pass
        finally:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        verify_existing()

    def stage_complete(self, stage):
        return os.path.exists(
            os.path.join(self.stage_dir(stage), core_checkpoints.DONE_MARKER)
        )

    def mark_stage_complete(self, stage):
        marker = os.path.join(self.stage_dir(stage), core_checkpoints.DONE_MARKER)
        with open(marker, "w") as handle:
            handle.write(datetime.now().isoformat())
        print(f"  [Checkpoint] Stage '{stage}' marked complete")

    def contig_done(self, stage, contig):
        return os.path.exists(core_checkpoints.contig_path(self.root, stage, contig))

    def global_done(self, stage):
        return os.path.exists(core_checkpoints.global_path(self.root, stage))

    def save_contig(self, stage, contig, payload):
        self.stage_dir(stage)
        try:
            written = core_checkpoints.write(
                core_checkpoints.contig_path(self.root, stage, contig),
                payload,
                nthreads=self.nthreads,
            )
            size_mb = written / (1024 * 1024)
            print(f"    [Checkpoint] {stage}/{contig} ({size_mb:.1f} MB)")
        except OSError as error:
            print(f"    [Checkpoint] WARNING: {stage}/{contig}: {error}")
            raise

    def load_contig(self, stage, contig, *, nthreads=None):
        read_threads = (
            self.nthreads if nthreads is None else max(1, int(nthreads))
        )
        return core_checkpoints.read(
            core_checkpoints.contig_path(self.root, stage, contig),
            nthreads=read_threads,
        )

    def save_global(self, stage, payload):
        self.stage_dir(stage)
        try:
            written = core_checkpoints.write(
                core_checkpoints.global_path(self.root, stage),
                payload,
                nthreads=self.nthreads,
            )
            size_mb = written / (1024 * 1024)
            print(
                f"{self.global_log_indent}[Checkpoint] "
                f"{stage}/_global ({size_mb:.1f} MB)"
            )
        except OSError as error:
            print(
                f"{self.global_log_indent}[Checkpoint] WARNING: "
                f"{stage}/_global: {error}"
            )
            raise

    def load_global(self, stage):
        return core_checkpoints.read(
            core_checkpoints.global_path(self.root, stage),
            nthreads=self.nthreads,
        )


    def bind_global_manifest(self, stage, payload):
        """Publish one JSON-compatible global manifest exactly once.

        Concurrent chromosome shards write fully formed unique candidates and
        use an atomic hard-link claim for the shared target. Every contender
        then verifies that the published manifest is scientifically identical.
        """

        if not isinstance(payload, dict) or not payload:
            raise ValueError("global manifest must be a nonempty dictionary")

        def canonical(value):
            return json.dumps(
                copy.deepcopy(value),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )

        expected = canonical(payload)
        target = core_checkpoints.global_path(self.root, stage)

        def verify_existing():
            try:
                observed = self.load_global(stage)
                observed_text = canonical(observed)
            except (OSError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"{stage}: global manifest is unreadable"
                ) from error
            if observed_text != expected:
                raise RuntimeError(
                    f"{stage}: global manifest does not match the current run"
                )

        if os.path.exists(target):
            verify_existing()
            return

        stage_path = self.stage_dir(stage)
        descriptor, candidate = tempfile.mkstemp(
            prefix="_global.", suffix=".candidate", dir=stage_path
        )
        os.close(descriptor)
        os.unlink(candidate)
        try:
            core_checkpoints.write(candidate, payload, nthreads=self.nthreads)
            try:
                os.link(candidate, target)
            except FileExistsError:
                pass
        finally:
            for temporary in (candidate, candidate + ".tmp"):
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        verify_existing()


def require_contig_checkpoints(store, stage, contigs):
    """Require every expected per-contig payload before marking a stage done."""
    missing = [str(contig) for contig in contigs
               if not store.contig_done(stage, contig)]
    if missing:
        raise OSError(
            f"Failed to checkpoint {stage}: " + ", ".join(missing)
        )


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
