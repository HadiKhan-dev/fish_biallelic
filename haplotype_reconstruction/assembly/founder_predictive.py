"""Full-site partial-founder evidence for exact primary-objective ties.

The primary complete-site score remains authoritative. This secondary score
uses the same seven predictive genotype distributions as proposal generation,
but permits sample-state changes at every SNP rather than only bin boundaries.
Unknown alleles are not imputed or released. Shared local source rows share
their unknown allele; different source rows use the existing independent
Bernoulli(1/2) predictive approximation.

Only partial-site log emissions are additional to the existing complete-site
workspace. Scoring costs O(N L K²), with O(N P + L K²) extra storage for P
partially observed sites, not an N L K² floating-point tensor.
"""
import math

import numpy as np
from numba import njit, prange
from numba.typed import List

from . import observations, partial_emissions


@njit(cache=True, parallel=True, nogil=True)
def _partial_logs(evidence, partial_sites):
    out = np.zeros((len(evidence), len(partial_sites), 7), np.float64)
    center = math.log(1.0 / 3.0)
    for sample in prange(len(evidence)):
        for index in range(len(partial_sites)):
            site = partial_sites[index]
            p0, p1, p2 = evidence[sample, site]
            total = p0 + p1 + p2
            if total <= 0.0 or (p0 == p1 and p1 == p2):
                continue
            p0, p1, p2 = p0 / total, p1 / total, p2 / total
            for code in range(7):
                value = .99 * partial_emissions.predictive_likelihood(
                    p0, p1, p2, code) + .01 / 3.0
                out[sample, index, code] = max(math.log(value), -2.0) - center
    return out


@njit(cache=True, parallel=True, nogil=True)
def _selected_codes(leaves, source, offsets, selected, partial_index):
    founders = len(selected)
    states = founders * (founders + 1) // 2
    out = np.empty((offsets[-1], states), np.uint8)
    for task in prange(len(leaves)):
        block = np.int64(task)
        local, rows = leaves[block], source[block]
        for site in range(local.shape[1]):
            index = offsets[block] + site
            state = 0
            for first in range(founders):
                a = selected[first, block]
                for second in range(first, founders):
                    b = selected[second, block]
                    # Source identity is needed only at usable partial sites.
                    same = a == b
                    if partial_index[index] >= 0:
                        same = rows[a, site] == rows[b, site]
                    out[index, state] = partial_emissions.pair_code(
                        local[a, site], local[b, site], same)
                    state += 1
    return out


@njit(cache=True, parallel=True, nogil=True)
def _score(codes, logs, extra, partial_index, penalty):
    samples, sites = logs.shape[:2]
    states = codes.shape[1]
    out = np.empty(samples, np.float64)
    center = math.log(1.0 / 3.0)
    for sample in prange(samples):
        scores = np.zeros(states)
        for site in range(sites):
            index = partial_index[site]
            if index >= 0:
                # Code 0/1/2 zero together means a neutral observation.
                if (extra[sample, index, 0] == 0.0
                        and extra[sample, index, 1] == 0.0
                        and extra[sample, index, 2] == 0.0):
                    continue
                switched = np.max(scores) - penalty
                for state in range(states):
                    scores[state] = max(scores[state], switched) + extra[
                        sample, index, codes[site, state]]
            else:
                p0, p1, p2 = logs[sample, site]
                if p0 == 0.0 and p1 == 0.0 and p2 == 0.0:
                    continue
                switched = np.max(scores) - penalty
                for state in range(states):
                    scores[state] = max(scores[state], switched) + (
                        logs[sample, site, codes[site, state]] - center)
        out[sample] = np.max(scores)
    return out


class PredictiveEvidence:
    """Lazy component-local preparation; all masks are fixed by the inputs."""

    def __init__(self, workspace):
        usable = np.concatenate([
            (np.ones(len(block.positions), bool) if block.keep_flags is None
             else np.asarray(block.keep_flags, bool))
            & np.any(observations.founder_inference_panel_from_block_result(
                block).called, axis=0)
            for block in workspace.batch])
        sites = np.flatnonzero(usable & ~workspace.complete)
        self.index = np.full(len(usable), -1, np.int32)
        self.index[sites] = np.arange(len(sites), dtype=np.int32)
        self.extra = _partial_logs(workspace.evidence, sites)
        self.equivalence = None
        self.source = List([
            partial_emissions.source_row_ids(block)
            if np.any(self.index[start:end] >= 0)
            else np.empty((0, 0), np.int32)
            for block, start, end in zip(
                workspace.batch, workspace.offsets[:-1], workspace.offsets[1:])])

    def score(self, workspace, selected):
        if not self.extra.shape[1]:
            return workspace.canonical(selected)
        codes = _selected_codes(workspace.leaves, self.source, workspace.offsets,
                                selected, self.index)
        return float(_score(codes, workspace.logs, self.extra, self.index,
                            workspace.penalty).sum())

    def primary_preserving_choices(self, workspace, selected, branch_cap):
        """Restrict each focal row to its complete-site allele equivalence class.

        This is an explicit search restriction, not a relaxed acceptance mask.
        All allowed paths have exactly the same primary emissions as the
        incumbent. The existing bounded dual solver can therefore target
        partial evidence without sacrificing a complete-site call elsewhere.
        """
        if not self.extra.shape[1]:
            return {}
        if self.equivalence is None:
            self.equivalence = []
            for local, start, end in zip(
                    workspace.leaves, workspace.offsets[:-1], workspace.offsets[1:]):
                groups = {}
                keys = []
                for row in local:
                    key = row[workspace.complete[start:end]].tobytes()
                    keys.append(key)
                    groups.setdefault(key, []).append(len(keys) - 1)
                self.equivalence.append([
                    np.asarray(groups[key], np.int64) for key in keys])
        answers = {}
        ranks = {}
        for focal, path in enumerate(selected):
            offsets, choices = [0], []
            ambiguous = False
            for block, incumbent in enumerate(path):
                local = self.equivalence[block][incumbent]
                if len(local) > branch_cap:
                    if block not in ranks:
                        emissions = workspace.models()[block]["bin_emissions"]
                        ranks[block] = emissions.max(axis=2).sum(axis=(0, 2))
                    order = np.argsort(-ranks[block][local], kind="stable")
                    local = local[order[:branch_cap]].copy()
                    if incumbent not in local:
                        local[-1] = incumbent
                ambiguous |= len(local) > 1
                choices.extend(local)
                offsets.append(len(choices))
            if ambiguous:
                answers[focal] = (np.asarray(choices, np.int64),
                                  np.asarray(offsets, np.int64))
        return answers
