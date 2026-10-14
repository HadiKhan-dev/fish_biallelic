"""Component-local evidence reuse for the final founder-refinement passes.

The original prepared panels, observation mask and binned evidence are fixed
across count proposals and path passes. Keep them once per invocation, not once
per competing fit.
"""
import math

import numpy as np
from numba.typed import List

from .. import chimera_scoring
from . import scoring as founder_scoring
from .. import observations
from ...core import parallel
from .evidence import gather_evidence, build_models


# A search's messages and per-sample scores run over consecutive sample ranges and re-read
# its thread share before each range (sample_ranges), so a search whose share grows while it
# computes them (its peers finished) takes the freed threads within the phase. At most
# SAMPLE_RANGES ranges per phase (each range builds the phase's shared pair-dosage tables
# again) and at least SAMPLES_PER_THREAD samples per thread of a range (the static OpenMP
# schedule gives every thread one contiguous slice), so a wide share runs one range.
SAMPLE_RANGES = 8
SAMPLES_PER_THREAD = 8


def resolve_threads(budget):
    return budget() if callable(budget) else budget


def sample_ranges(samples, threads):
    """Consecutive ranges (s0, s1, n) covering samples 0 .. samples - 1; n is the share
    ``threads`` (an int or a callable budget) read when the range starts."""
    start = 0
    while start < samples:
        n = max(1, int(resolve_threads(threads)))
        stop = min(samples, start + max(-(-samples // SAMPLE_RANGES), SAMPLES_PER_THREAD * n))
        yield start, stop, n
        start = stop


class ComponentWorkspace:
    def __init__(self, batch, neutral, sites, max_bins, threads, minimum_bin_size=1):
        self.positions = np.concatenate([b.positions for b in batch])
        indices = np.searchsorted(sites, self.positions)
        if not np.array_equal(sites[indices], self.positions):
            raise ValueError("founder refinement evidence positions do not match")
        self.evidence = gather_evidence(neutral, indices)
        self.leaves = List([np.ascontiguousarray(getattr(
            b, "missing_aware_inference_discrete_haps", b.discrete_haps), np.int8)
            for b in batch])
        # Every block's pre-fill inference panel, built once (the block models read them too).
        self.panels = [observations.founder_inference_panel_from_block_result(b) for b in batch]
        self.complete = np.concatenate([
            (np.ones(len(b.positions), np.bool_) if b.keep_flags is None
             else np.asarray(b.keep_flags, np.bool_))
            & np.all(panel.called, axis=0)
            for b, panel in zip(batch, self.panels)])
        self.offsets = np.asarray([0, *np.cumsum([len(b.positions) for b in batch])], np.int64)
        self.penalty = founder_scoring.SWITCH_PENALTY
        self.bin_size = max(minimum_bin_size, chimera_scoring.compute_spb(batch),
                            math.ceil(len(self.positions) / max_bins))
        self.logs = founder_scoring.prepare_log_evidence(self.evidence, self.complete)
        self.batch = batch
        self.submodels = None
        self.threads = threads

    def models(self):
        if self.submodels is None:
            with parallel.numba_thread_scope(resolve_threads(self.threads)):
                self.submodels = build_models(self)
        return self.submodels


def component_workspace(cache, batch, neutral, sites, max_bins, threads, minimum_bin_size=1):
    """Cache belongs to one final-refinement invocation with fixed inputs."""
    key = (int(batch[0].positions[0]), int(batch[-1].positions[-1]))
    if key not in cache:
        cache[key] = ComponentWorkspace(batch, neutral, sites, max_bins, threads,
                                        minimum_bin_size)
    cache[key].threads = threads
    return cache[key]
