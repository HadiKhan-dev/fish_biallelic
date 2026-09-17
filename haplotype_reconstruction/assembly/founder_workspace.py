"""Component-local evidence reuse for the final founder-refinement passes.

The original prepared panels, observation mask and binned evidence are fixed
across count proposals and path passes. Keep them once per invocation, not once
per competing fit. Score caches retain ordered rows (including founder labels)
and are bounded; they never change proposal ordering or acceptance rules.
"""
from collections import OrderedDict
import math

import numpy as np
from numba.typed import List

from . import chimera_scoring, founder_scoring, founder_delta, observations
from ..core import parallel
from .founder_evidence import gather_evidence, build_models


def resolve_threads(budget):
    return budget() if callable(budget) else budget


class ComponentWorkspace:
    def __init__(self, batch, neutral, sites, max_bins, threads, prepared_arrays=None,
                 minimum_bin_size=1):
        self.positions = np.concatenate([b.positions for b in batch])
        indices = np.searchsorted(sites, self.positions)
        if not np.array_equal(sites[indices], self.positions):
            raise ValueError("founder refinement evidence positions do not match")
        self.evidence = (gather_evidence(neutral, indices)
                         if prepared_arrays is None else prepared_arrays[0])
        self.leaves = List([np.ascontiguousarray(getattr(
            b, "missing_aware_inference_discrete_haps", b.discrete_haps), np.int8)
            for b in batch])
        self.complete = (np.concatenate([
            (np.ones(len(b.positions), np.bool_) if b.keep_flags is None
             else np.asarray(b.keep_flags, np.bool_))
            & np.all(observations.founder_inference_panel_from_block_result(b).called, axis=0)
            for b in batch]) if prepared_arrays is None else prepared_arrays[1])
        self.offsets = np.asarray([0, *np.cumsum([len(b.positions) for b in batch])], np.int64)
        self.penalty = chimera_scoring.compute_penalty(batch)
        self.bin_size = max(minimum_bin_size, chimera_scoring.compute_spb(batch),
                            math.ceil(len(self.positions) / max_bins))
        self.logs = (founder_scoring.prepare_log_evidence(self.evidence, self.complete)
                     if prepared_arrays is None else prepared_arrays[2])
        self.batch = batch
        self.submodels = None
        self.scores = OrderedDict()
        self.local_keys = set()
        self.painted_key = None
        self.painting = None
        self.threads = threads
        self.reference = None
        self.flanks = None
        self.local_scores = 0
        # The existing identical-order window reuse must also work when small
        # components retain only a completed checkpoint, not proposal files.
        # Entries carry their exact panel signature and live for this fit only.
        self.path_proposals = {}
        self.predictive_evidence = None
        self.predictive_scores = OrderedDict()

    def models(self):
        if self.submodels is None:
            with parallel.numba_thread_scope(resolve_threads(self.threads)):
                self.submodels = build_models(self)
        return self.submodels

    def set_reference(self, panel):
        if self.reference is None or not np.array_equal(panel, self.reference):
            self.reference = panel.copy()
            self.flanks = None

    def _remember(self, key, value, local=False):
        self.scores[key] = value
        if local:
            self.local_keys.add(key)
        else:
            self.local_keys.discard(key)
        if len(self.scores) > 64:
            expired, _ = self.scores.popitem(last=False)
            self.local_keys.discard(expired)

    def canonical(self, panel):
        key = (panel.shape, panel.tobytes())
        if key in self.scores and key not in self.local_keys:
            return self.scores[key]
        with parallel.numba_thread_scope(resolve_threads(self.threads)):
            alleles = founder_scoring.selected_alleles(self.leaves, self.offsets, panel)
            value = float(founder_scoring.score_panel(
                alleles, self.evidence, self.complete, self.penalty, self.logs).sum())
        self._remember(key, value)
        return value

    def predictive(self, panel):
        """Secondary partial-founder score, never a replacement primary score."""
        key = (panel.shape, panel.tobytes())
        if key in self.predictive_scores:
            self.predictive_scores.move_to_end(key)
            return self.predictive_scores[key]
        with parallel.numba_thread_scope(resolve_threads(self.threads)):
            if self.predictive_evidence is None:
                from .founder_predictive import PredictiveEvidence
                self.predictive_evidence = PredictiveEvidence(self)
            value = self.predictive_evidence.score(self, panel)
        self.predictive_scores[key] = value
        if len(self.predictive_scores) > 64:
            self.predictive_scores.popitem(last=False)
        return value

    def primary_preserving_choices(self, panel, branch_cap):
        if self.predictive_evidence is None:
            from .founder_predictive import PredictiveEvidence
            self.predictive_evidence = PredictiveEvidence(self)
        return self.predictive_evidence.primary_preserving_choices(
            self, panel, branch_cap)

    def _local_score(self, panel):
        if self.reference is None or panel.shape != self.reference.shape:
            return None
        changed = np.flatnonzero(np.any(panel != self.reference, axis=0))
        if not len(changed):
            return None
        start, stop = int(changed[0]), int(changed[-1])+1
        # Whole/long edits retain the linear full scan; cache only short edits.
        if self.offsets[stop]-self.offsets[start] > len(self.positions)//2:
            return None
        if self.flanks is None:
            from ..painting.model import available_process_memory_bytes
            available = available_process_memory_bytes()
            states = len(panel)*(len(panel)+1)//2
            required = 16*(len(self.batch)+1)*len(self.evidence)*states
            if available is not None and required > available//8:
                return None
            alleles = founder_scoring.selected_alleles(self.leaves, self.offsets, self.reference)
            self.flanks = founder_delta.messages(alleles, self.logs, self.offsets, self.penalty)
        prefix, suffix = self.flanks
        value = founder_delta.score_interval(self.leaves, self.offsets, panel, self.logs,
            self.penalty, prefix[start], suffix[stop], start, stop)
        self.local_scores += 1
        return float(value.sum())

    def evaluate(self, panel, paint=False):
        key = (panel.shape, panel.tobytes())
        if paint and key == self.painted_key:
            return self.painting
        if not paint and key in self.scores:
            self.scores.move_to_end(key)
            return self.scores[key]
        with parallel.numba_thread_scope(resolve_threads(self.threads)):
            value = None if paint else self._local_score(panel)
            if value is not None:
                self._remember(key, value, local=True)
                return value
            alleles = founder_scoring.selected_alleles(self.leaves, self.offsets, panel)
            if paint:
                self.painting = founder_scoring.paint_panel(
                    alleles, self.evidence, self.complete, self.penalty, self.logs)[0]
                self.painted_key = key
                return self.painting
            value = float(founder_scoring.score_panel(
                alleles, self.evidence, self.complete, self.penalty, self.logs).sum())
        self._remember(key, value)
        return value


def component_workspace(cache, batch, neutral, sites, max_bins, threads, prepared_arrays=None,
                        minimum_bin_size=1):
    """Cache belongs to one final-refinement invocation with fixed inputs."""
    key = (int(batch[0].positions[0]), int(batch[-1].positions[-1]))
    if key not in cache:
        cache[key] = ComponentWorkspace(batch, neutral, sites, max_bins, threads,
                                        prepared_arrays, minimum_bin_size)
    cache[key].threads = threads
    return cache[key]
