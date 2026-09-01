"""Shared operational helpers for the repository's pipeline entry points.

This module contains only behaviour-neutral runtime mechanics.  Dataset paths,
stage names, scientific settings, and Pearly's provenance-bound resume policy
remain in their owning entry points.
"""

from __future__ import annotations

import copy
import gc
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import checkpoint_io
from memory_utils import malloc_trim


FOUNDER_BLOCK_KEY = "founder_block"
SAMPLE_IDS_KEY = "sample_ids"


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
            os.path.join(self.stage_dir(stage), checkpoint_io.DONE_MARKER)
        )

    def mark_stage_complete(self, stage):
        marker = os.path.join(self.stage_dir(stage), checkpoint_io.DONE_MARKER)
        with open(marker, "w") as handle:
            handle.write(datetime.now().isoformat())
        print(f"  [Checkpoint] Stage '{stage}' marked complete")

    def contig_done(self, stage, contig):
        return os.path.exists(checkpoint_io.contig_path(self.root, stage, contig))

    def global_done(self, stage):
        return os.path.exists(checkpoint_io.global_path(self.root, stage))

    def save_contig(self, stage, contig, payload):
        self.stage_dir(stage)
        try:
            written = checkpoint_io.write(
                checkpoint_io.contig_path(self.root, stage, contig),
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
        return checkpoint_io.read(
            checkpoint_io.contig_path(self.root, stage, contig),
            nthreads=read_threads,
        )

    def save_global(self, stage, payload):
        self.stage_dir(stage)
        try:
            written = checkpoint_io.write(
                checkpoint_io.global_path(self.root, stage),
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
        return checkpoint_io.read(
            checkpoint_io.global_path(self.root, stage),
            nthreads=self.nthreads,
        )


def require_contig_checkpoints(store, stage, contigs):
    """Require every expected per-contig payload before marking a stage done."""
    missing = [str(contig) for contig in contigs
               if not store.contig_done(stage, contig)]
    if missing:
        raise OSError(
            f"Failed to checkpoint {stage}: " + ", ".join(missing)
        )


def strip_block_probs(blocks):
    """Drop probability arrays that supported stages reload globally."""
    for block in blocks:
        if hasattr(block, "probs_array") and block.probs_array is not None:
            block.probs_array = None
    return blocks


def strip_block_evidence(blocks):
    """Drop block-local evidence unused after supported checkpoint boundaries."""
    for block in blocks:
        if hasattr(block, "probs_array"):
            block.probs_array = None
        if hasattr(block, "reads_count_matrix"):
            block.reads_count_matrix = None
    return blocks


def compact_founder_block(block):
    """Return a shallow final-panel snapshot containing downstream inputs.

    The block-local probability and read-count tensors are not consumed after painting.
    The copied object retains positions, founder haplotypes and IDs, keep flags,
    evidence-mode metadata, and any lightweight dynamic attributes needed by
    pedigree or phase processing.
    """
    compact = copy.copy(block)
    strip_block_evidence((compact,))
    return compact


def load_founder_blocks_parallel(
    store,
    contigs,
    stage_keys,
    *,
    max_workers,
    strip_probs=True,
    require_all=False,
):
    """Load the preferred available founder block for each contig.

    ``stage_keys`` contains ordered ``(stage, payload_key)`` pairs. A payload
    value may be a founder block directly or a non-empty block collection.
    Checkpoints are considered in that order independently for each contig,
    falling back when a checkpoint is absent or its value is empty. The
    returned mapping contains only contigs for which a founder block was found.
    With ``require_all=True``, raise instead if any requested contig has no
    usable block; this preserves strict production stages that cannot skip one.
    """
    contigs = tuple(contigs)
    stage_keys = tuple(stage_keys)
    if not contigs or not stage_keys:
        return {}

    requested_workers = int(max_workers)
    if requested_workers < 1:
        raise ValueError("max_workers must be greater than zero")
    store_threads = max(1, int(getattr(store, "nthreads", requested_workers)))
    total_read_threads = min(requested_workers, store_threads)
    effective_workers = min(total_read_threads, len(contigs))
    threads_per_read = max(1, total_read_threads // effective_workers)
    extra_thread_reads = total_read_threads % effective_workers
    missing = object()

    def load_one(indexed_contig):
        index, contig = indexed_contig
        read_threads = threads_per_read + int(index < extra_thread_reads)
        for stage, payload_key in stage_keys:
            if not store.contig_done(stage, contig):
                continue
            payload = store.load_contig(
                stage, contig, nthreads=read_threads
            )
            try:
                if payload_key not in payload:
                    continue
                value = payload[payload_key]
                if value is None:
                    continue
                if hasattr(value, "haplotypes"):
                    founder_block = value
                else:
                    if not value:
                        continue
                    founder_block = value[0]
                if strip_probs:
                    strip_block_probs((founder_block,))
                return contig, founder_block
            finally:
                del payload
        return contig, missing

    found = {}
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        for contig, founder_block in executor.map(load_one, enumerate(contigs)):
            if founder_block is not missing:
                found[contig] = founder_block
    if require_all and len(found) != len(contigs):
        missing_contigs = [
            str(contig) for contig in contigs if contig not in found
        ]
        raise RuntimeError(
            "No non-empty founder block found for required contigs: "
            + ", ".join(missing_contigs)
        )
    return found


def load_global_arrays(store, discovery_stage, contig):
    """Load global probabilities/sites and promptly release the full payload."""
    import numpy as np

    load_contig = getattr(store, "load_contig", store)
    payload = load_contig(discovery_stage, contig)
    global_probs = payload["global_probs"]
    global_sites = payload["global_sites"]
    del payload
    gc.collect()
    malloc_trim()
    if global_probs.dtype == np.float64:
        global_probs = global_probs.astype(np.float32)
    return global_probs, global_sites


def validate_painting_bundle(
    payload, *, expected_sample_ids=None, context="painting checkpoint"
):
    """Validate the ordered sample identity of an atomic painting bundle."""
    required = ("tolerance_result", FOUNDER_BLOCK_KEY, SAMPLE_IDS_KEY)
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(
            f"{context} lacks required keys: " + ", ".join(missing)
        )

    sample_ids = tuple(str(value) for value in payload[SAMPLE_IDS_KEY])
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{context} contains duplicate sample IDs")
    painting = payload["tolerance_result"]
    samples = (
        painting
        if isinstance(painting, (list, tuple))
        else getattr(painting, "samples", None)
    )
    if samples is None:
        raise TypeError(f"{context} has an unrecognized painting container")
    if len(samples) != len(sample_ids):
        raise ValueError(
            f"{context} has {len(samples)} painted samples but "
            f"{len(sample_ids)} sample IDs"
        )

    if expected_sample_ids is not None:
        expected = tuple(str(value) for value in expected_sample_ids)
        if sample_ids != expected:
            mismatch = next(
                (index for index, pair in enumerate(zip(sample_ids, expected))
                 if pair[0] != pair[1]),
                min(len(sample_ids), len(expected)),
            )
            raise ValueError(
                f"{context} sample order does not match the expected order "
                f"(observed={len(sample_ids)}, expected={len(expected)}, "
                f"first_mismatch_index={mismatch})"
            )
    return sample_ids


def load_phase_correction_inputs(
    checkpoint_dir,
    contig,
    *,
    tolerance_stage,
    checkpoint_available=os.path.exists,
    checkpoint_reader=checkpoint_io.read,
    strip_founder_probs=False,
):
    """Load an atomic final-panel painting bundle for phase correction.

    Painting, founder block, and ordered sample IDs must come from the same
    checkpoint payload; no assembly fallback is permitted. Callbacks remain
    picklable under forkserver.
    """
    tolerance_path = checkpoint_io.contig_path(
        checkpoint_dir, tolerance_stage, contig
    )
    if not checkpoint_available(tolerance_path):
        raise FileNotFoundError(
            f"Missing required painting checkpoint for {contig}: "
            f"{tolerance_path}"
        )
    payload = checkpoint_reader(tolerance_path)
    sample_ids = validate_painting_bundle(
        payload, context=f"Painting checkpoint {tolerance_path}"
    )
    founder_block = payload[FOUNDER_BLOCK_KEY]
    if strip_founder_probs:
        strip_block_probs((founder_block,))
    data = {
        "tolerance_result": payload["tolerance_result"],
        "founder_block": founder_block,
        SAMPLE_IDS_KEY: sample_ids,
    }
    del payload
    return data


def make_refinement_assembly_functions(
    run_hierarchical_step,
    global_probs,
    global_sites,
    *,
    batch_size,
    recomb_rate,
    n_generations,
    beam_width,
    max_founders,
    cc_scale,
    num_processes,
    maxtasksperchild,
):
    """Build the established L1/L2 callbacks used by block refinement."""

    def run_l1(input_blocks):
        return run_hierarchical_step(
            input_blocks=input_blocks,
            global_probs=global_probs,
            global_sites=global_sites,
            batch_size=batch_size,
            use_hmm_linking=False,
            beam_width=beam_width,
            max_founders=max_founders,
            max_sites_for_linking=2000,
            cc_scale=cc_scale,
            num_processes=num_processes,
            maxtasksperchild=maxtasksperchild,
            refine_after_stitch=False,
        )

    def run_l2(input_blocks):
        return run_hierarchical_step(
            input_blocks=input_blocks,
            global_probs=global_probs,
            global_sites=global_sites,
            batch_size=batch_size,
            use_hmm_linking=True,
            recomb_rate=recomb_rate,
            beam_width=beam_width,
            max_founders=max_founders,
            cc_scale=cc_scale,
            num_processes=num_processes,
            n_generations=n_generations,
            verbose=False,
            maxtasksperchild=maxtasksperchild,
            refine_after_stitch=False,
        )

    return run_l1, run_l2
