"""Inputs of one component's refinement, built once per level and shared by every fit.

The prepared panels (alleles and evidence masks), the complete-site mask, the robust log evidence
and the binned surrogate emissions are fixed across count rounds and fits. The component's evidence
is a view of the chromosome's float32 neutral evidence (the joins' copy; a gathered copy only when
the component's prepared panels are not adjacent on the site axis).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from numba.typed import List

from ...maths.emissions import (CENTER, normalised_likelihoods, pair_code, predictive_likelihood,
                                robust_log_likelihood)
from .objective import SWITCH_PENALTY, is_skipped, prepare_log_evidence

# A search's messages and per-sample scores run over consecutive sample ranges and re-read its
# thread share before each range (sample_ranges), so a search whose share grows while it computes
# them (its peers finished) takes the freed threads within the phase. At most SAMPLE_RANGES ranges
# per phase (each range rebuilds the phase's shared pair-dosage tables) and at least
# SAMPLES_PER_THREAD samples per thread of a range (the static OpenMP schedule gives every thread
# one contiguous slice), so a wide share runs one range.
SAMPLE_RANGES = 8
SAMPLES_PER_THREAD = 8


def sample_ranges(samples, budget):
    """Consecutive ranges (s0, s1, n) covering samples 0 .. samples - 1; n is the budget's live
    share read when the range starts."""
    start = 0
    while start < samples:
        n = max(1, int(budget.threads()))
        stop = min(samples, start + max(-(-samples // SAMPLE_RANGES), SAMPLES_PER_THREAD * n))
        yield start, stop, n
        start = stop


@njit(cache=True, parallel=True, nogil=True)
def binned_emissions(evidence, offsets, alleles, keeps, full, bin_size, outputs):
    """Binned surrogate emissions of every block and pair of its local rows, into the zeroed
    outputs[b] float64 (S, R_b, R_b, bins_b) (symmetric in the two rows).

    The model of the join's binned scorer: per kept site with informative evidence the robust
    emission max(log(0.99 P + 0.01/3), -2) - log(1/3), P the pair's genotype likelihood (hard
    dosage when every row of the block is called at its kept sites, else the predictive pair codes
    with unknown alleles marginalised; prepared panels are atomic, so only a row paired with itself
    shares an unknown allele), summed over the sites of each bin of bin_size block sites."""
    samples = len(evidence)
    for task in prange(len(alleles) * samples):
        block, sample = (task // samples, task % samples)
        local = alleles[block]
        kept = keeps[block]
        out = outputs[block]
        logs = np.empty(7, np.float64)
        for site in range(local.shape[1]):
            if not kept[site]:
                continue
            index = offsets[block] + site
            informative, p0, p1, p2 = normalised_likelihoods(
                evidence[sample, index, 0], evidence[sample, index, 1], evidence[sample, index, 2])
            if not informative:
                continue
            if full[block]:
                logs[0] = robust_log_likelihood(p0) - CENTER
                logs[1] = robust_log_likelihood(p1) - CENTER
                logs[2] = robust_log_likelihood(p2) - CENTER
            else:
                for code in range(7):
                    logs[code] = robust_log_likelihood(predictive_likelihood(p0, p1, p2, code)) - CENTER
            for first in range(len(local)):
                for second in range(first, len(local)):
                    if full[block]:
                        code = local[first, site] + local[second, site]
                    else:
                        code = pair_code(local[first, site], local[second, site], first == second)
                    value = logs[code]
                    bin_index = site // bin_size
                    out[sample, first, second, bin_index] += value
                    if first != second:
                        out[sample, second, first, bin_index] += value


@njit(cache=True, nogil=True)
def informative_samples(logs):
    """bool (S,): samples with evidence at a complete site of the component (a site that is not
    skipped). Every other sample's F_s is exactly 0 for every panel (no site step ever runs), so it
    adds an exact 0 to every per-sample difference of every count decision. A sample's scan stops
    at its first such site."""
    result = np.zeros(logs.shape[0], np.bool_)
    for sample in range(logs.shape[0]):
        for site in range(logs.shape[1]):
            if not is_skipped(logs, sample, site):
                result[sample] = True
                break
    return result


# Bookkeeping and transients of one component task beyond its arrays (bytes); measured on std_n80 L1-L2 components
# (N = 80): whole-task peaks of 62-310 MiB above the worker's starting RSS, fits and window solves included.
COMPONENT_BASE_BYTES = 64 << 20


def component_bytes(panels, samples, *, min_sites_per_bin, max_bins) -> int:
    """Peak bytes of one component's workspace (the component task's memory reservation, execution.memory): the
    robust log evidence and the reversed orientation's copy (2 x 24 S L float64), a gathered evidence copy when the
    panels are not adjacent (12 S L float32), the binned emissions (8 S R_b^2 bins_b per block) and the leaves
    (R_b x L_b int8), plus COMPONENT_BASE_BYTES. The fits' message arrays, the window solves and the score tables
    reserve their own bytes (additive estimates); calibrated on K30's map (founder_refinement.md 3.2-3.3)."""
    sites = sum(panel.sites for panel in panels)
    bin_size = max(int(min_sites_per_bin), math.ceil(sites / max_bins))
    starts = [panel.site_start for panel in panels]
    adjacent = all(b == a + panel.sites for a, b, panel in zip(starts, starts[1:], panels))
    emissions = sum(8 * samples * panel.rows * panel.rows * ((panel.sites + bin_size - 1) // bin_size)
                    for panel in panels)
    leaves = sum(panel.rows * panel.sites for panel in panels)
    return (48 * samples * sites + (0 if adjacent else 12 * samples * sites) + emissions + leaves
            + COMPONENT_BASE_BYTES)


def component_evidence(evidence, panels):
    """The component's (S, L_c, 3) float32 evidence: a view of the chromosome's when its prepared
    panels are adjacent on the site axis, else a gathered copy."""
    starts = np.asarray([panel.site_start for panel in panels], np.int64)
    lengths = np.asarray([panel.sites for panel in panels], np.int64)
    if np.all(starts[1:] == starts[:-1] + lengths[:-1]):
        return evidence[:, starts[0]:starts[-1] + lengths[-1]]
    indices = np.concatenate([np.arange(start, start + length) for start, length in zip(starts, lengths)])
    return np.ascontiguousarray(evidence[:, indices])


class ComponentWorkspace:
    """One component's fixed inputs.

    panels: the prepared panels of the component's blocks (model.components.PreparedPanel).
    evidence: float32 (S, L_c, 3) component evidence (component_evidence).
    leaves: typed List of the blocks' inference calls int8 (R_b, L_b) (-1 uncalled): every
        likelihood reads them. offsets: int64 (B + 1) block site offsets. complete: bool (L_c),
        kept sites where every local row of the block is called (the objective's site mask,
        candidate-independent). logs: float64 (S, L_c, 3) robust log evidence of complete sites.
    count_samples: int64 evidence rows of the informative samples (informative_samples), the
        samples the count rule reads (count_rule.founder_cost's S); None when every sample is
        informative (or, degenerate, none is: every score is then 0 and every row is kept).
    bin_size: surrogate bin (sites) = max(min_sites_per_bin, ceil(L_c / max_bins)).
    emissions: per block float64 (S, R_b, R_b, bins_b) binned surrogate emissions.
    """

    def __init__(self, panels, evidence, *, min_sites_per_bin, max_bins):
        self.panels = tuple(panels)
        self.evidence = evidence
        self.leaves = List([np.ascontiguousarray(panel.inference, np.int8) for panel in self.panels])
        self.offsets = np.asarray([0, *np.cumsum([panel.sites for panel in self.panels])], np.int64)
        self.complete = np.concatenate([
            np.asarray(panel.kept, np.bool_) & np.all(panel.inference >= 0, axis=0)
            for panel in self.panels])
        self.penalty = SWITCH_PENALTY
        self.samples = int(evidence.shape[0])
        self.bin_size = max(int(min_sites_per_bin), math.ceil(int(self.offsets[-1]) / max_bins))
        self.logs = prepare_log_evidence(evidence, self.complete)
        informative = informative_samples(self.logs)
        self.count_samples = (None if informative.all() or not informative.any()
                              else np.flatnonzero(informative))
        self.emissions = self._binned_emissions()

    @property
    def blocks(self):
        return len(self.panels)

    @property
    def count_sample_count(self) -> int:
        """S of the count rule: the informative samples (every sample when count_samples is None)."""
        return self.samples if self.count_samples is None else len(self.count_samples)

    def _binned_emissions(self):
        alleles, keeps, outputs, full = List(), List(), List(), []
        for panel in self.panels:
            kept = np.asarray(panel.kept, np.bool_)
            called = panel.inference >= 0
            full.append(bool(np.all(called[:, kept])))
            keeps.append(np.ascontiguousarray(kept & np.any(called, axis=0)))
            alleles.append(np.ascontiguousarray(panel.inference, np.int8))
            bins = (panel.sites + self.bin_size - 1) // self.bin_size
            outputs.append(np.zeros((self.samples, panel.rows, panel.rows, bins), np.float64))
        binned_emissions(self.evidence, self.offsets, alleles, keeps, np.asarray(full, np.bool_),
                         self.bin_size, outputs)
        return list(outputs)
