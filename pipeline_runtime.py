"""Shared operational helpers for the repository's pipeline entry points.

This module contains only behaviour-neutral runtime mechanics.  Dataset paths,
stage names, scientific settings, and Pearly's provenance-bound resume policy
remain in their owning entry points.
"""

from __future__ import annotations

import gc
import os
from datetime import datetime

import checkpoint_io
from memory_utils import malloc_trim


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
    """Facade for the standard, non-provenance-bound pipeline checkpoints.

    Save failures retain the established standard-pipeline policy: filesystem
    ``OSError`` exceptions are reported and the pipeline continues.  Pearly's
    fatal, provenance-bound checkpoint policy deliberately does not use this
    class.
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

    def stage_complete(self, stage):
        return os.path.exists(os.path.join(self.stage_dir(stage), "_done"))

    def mark_stage_complete(self, stage):
        with open(os.path.join(self.stage_dir(stage), "_done"), "w") as handle:
            handle.write(datetime.now().isoformat())
        print(f"  [Checkpoint] Stage '{stage}' marked complete")

    def contig_done(self, stage, contig):
        return os.path.exists(checkpoint_io.contig_path(self.root, stage, contig))

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

    def load_contig(self, stage, contig):
        return checkpoint_io.read(
            checkpoint_io.contig_path(self.root, stage, contig),
            nthreads=self.nthreads,
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

    def load_global(self, stage):
        return checkpoint_io.read(
            checkpoint_io.global_path(self.root, stage),
            nthreads=self.nthreads,
        )


def strip_block_probs(blocks):
    """Drop reconstructible per-block probability arrays in place."""
    for block in blocks:
        if hasattr(block, "probs_array") and block.probs_array is not None:
            block.probs_array = None
    return blocks


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


def load_phase_correction_inputs(
    checkpoint_dir,
    contig,
    *,
    tolerance_stage,
    founder_stage_keys,
    checkpoint_available=os.path.exists,
    checkpoint_reader=checkpoint_io.read,
    strip_founder_probs=False,
):
    """Traverse painting and assembly checkpoints for phase correction.

    ``founder_stage_keys`` is ordered from preferred to fallback checkpoint.
    Callers retain module-level wrappers so callbacks remain importable and
    picklable under the project's forkserver multiprocessing model.
    """
    data = {}
    tolerance_path = checkpoint_io.contig_path(
        checkpoint_dir, tolerance_stage, contig
    )
    if checkpoint_available(tolerance_path):
        payload = checkpoint_reader(tolerance_path)
        if "tolerance_result" in payload:
            data["tolerance_result"] = payload["tolerance_result"]
        del payload

    for stage, list_key in founder_stage_keys:
        path = checkpoint_io.contig_path(checkpoint_dir, stage, contig)
        if not checkpoint_available(path):
            continue
        payload = checkpoint_reader(path)
        if list_key in payload and payload[list_key]:
            founder_block = payload[list_key][0]
            if strip_founder_probs:
                founder_block.probs_array = None
            data["founder_block"] = founder_block
            del payload
            break
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
