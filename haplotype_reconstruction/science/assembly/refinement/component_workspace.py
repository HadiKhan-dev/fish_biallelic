"""Inputs of one component's refinement, built once per level and shared by every fit.

Owns immutable component dimensions and surrogate emissions; storage comes from an explicit factory.
Method: docs/methods/founder_refinement.md#workspace (RC01–RC03).

The prepared panels (alleles and evidence masks), the objective's site mask and the binned surrogate
emissions are fixed across count rounds and fits. The component's evidence is a view of the
chromosome's float32 neutral evidence (the joins' copy; a gathered copy only when the component's
prepared panels are not adjacent on the site axis). Release 3.4: the robust log evidence and the switch
costs are not held for the whole component; kernels read them through windows (window.py, `windows`),
resident for a small component and streamed per consumer for a large one.
"""

from __future__ import annotations

import math
import threading

import numpy as np
from numba import prange
from numba.typed import List

from haplotype_reconstruction.core.kernels import kernel
from haplotype_reconstruction.science.shared.emissions import (
    CENTER,
    normalised_likelihoods,
    pair_code,
    predictive_likelihood,
    robust_log_likelihood,
)

from ..pairs import emission_pair_index as pair_index
from .configuration import DEFAULT_GENOTYPE_ERROR, DEFAULT_RESET_RATE, MAP_PROBABILITY_FLOOR
from .objective import SWITCH_PENALTY
from .overloads import overloads_for
from .window import (
    informative_samples,
    map_normalisers,
)


@kernel(overloads=overloads_for(__name__, "binned_emissions"), cache=True, parallel=True, nogil=True)
def binned_emissions(evidence, offsets, alleles, keeps, full, bin_size, outputs):
    """Binned surrogate emissions of every block and pair of its local rows, into the zeroed
    outputs[b] float64 (S, R_b(R_b + 1)/2, bins_b) (symmetric in the two rows).

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
                evidence[sample, index, 0], evidence[sample, index, 1], evidence[sample, index, 2]
            )
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
                    out[sample, pair_index(first, second), bin_index] += value


# ------------------------------------------------------------------ map-derived switch costs
def map_hazards(positions, genetic_map, reset_rate, probability_floor):
    """float64 (L,) -log(no reset) of every physical edge (site j - 1 to j; edge 0 has none):
    reset_rate x Morgans, at least -log(1 - probability_floor) (equal map positions included)."""
    positions = np.asarray(positions, np.int64)
    result = np.zeros(len(positions), np.float64)
    if len(positions) > 1:
        distance = np.asarray(genetic_map.interval_morgans(positions[:-1], positions[1:]), np.float64)
        result[1:] = np.maximum(float(reset_rate) * distance, -math.log1p(-float(probability_floor)))
    return result


# The stage consumes dimension working-set models from costs.py.


def frozen(array, dtype):
    """``array`` as one C-contiguous, read-only ndarray of ``dtype``: a read-only view when it already is contiguous
    (the caller's array keeps its own flags), else a copy. Refinement's shared inputs take this one representation,
    the argument type the warm-up compiles: a strided or writable variant is a kernel type of its own, which every
    worker would compile on first use."""
    part = np.ascontiguousarray(array, dtype).view()
    part.flags.writeable = False
    return part


def component_evidence(evidence, panels):
    """The component's (S, L_c, 3) float32 evidence, frozen: the chromosome's own array when the component spans it
    (or a view of a contiguous span), else a copy of the component's sites."""
    starts = np.asarray([panel.site_start for panel in panels], np.int64)
    lengths = np.asarray([panel.sites for panel in panels], np.int64)
    if np.all(starts[1:] == starts[:-1] + lengths[:-1]):
        return frozen(evidence[:, starts[0] : starts[-1] + lengths[-1]], np.float32)
    sites = np.concatenate([np.arange(start, start + n) for start, n in zip(starts, lengths, strict=True)])
    return frozen(evidence[:, sites], np.float32)


class ComponentWorkspace:
    """One component's fixed inputs.

    panels: the prepared panels of the component's blocks (model.components.PreparedPanel).
    evidence: float32 (S, L_c, 3) component evidence (component_evidence), frozen (C-contiguous, read-only).
    leaves: typed List of the blocks' coded calls int16 (R_b, L_b): the called allele, or -2 - local
        row where uncalled (objective.pair_code7); every likelihood reads them. offsets: int64 (B + 1)
        block site offsets. kept: bool (L_c), the objective's sites: kept sites with at least one
        called local row (candidate-independent). partial: bool (L_c), kept sites where a local row is
        uncalled (the only sites where pair codes 3-6 occur). windows: window.Windows over them, the
        robust log evidence of the seven pair codes and the switch costs (resident or streamed).
    count_samples: int64 evidence rows of the informative samples (informative_samples), the
        samples the count rule reads (count_rule.founder_cost's S); None when every sample is
        informative (or, degenerate, none is: every score is then 0 and every row is kept).
    bin_size: surrogate bin (sites) = max(min_sites_per_bin, ceil(L_c / max_bins)).
    emissions: per block float64 (S, R_b(R_b + 1)/2, bins_b) binned surrogate emissions.
    hazards: float64 (L_c,) physical-edge reset hazards of the genetic map (map_hazards); the windows
        and normalisers(K) derive the objective's switch costs and normalisers from them.
    """

    def __init__(
        self,
        panels,
        evidence,
        *,
        min_sites_per_bin,
        max_bins,
        genetic_map=None,
        genotype_error=DEFAULT_GENOTYPE_ERROR,
        reset_rate=DEFAULT_RESET_RATE,
        map_probability_floor=MAP_PROBABILITY_FLOOR,
        windows_factory=None,
        message_step=None,
    ):
        if genetic_map is None:
            raise ValueError("founder refinement needs the contig's genetic map (map-derived switch costs)")
        if genotype_error != DEFAULT_GENOTYPE_ERROR:
            raise ValueError("the robust emission's genotype error is fixed at 0.01")
        self.panels = tuple(panels)
        # One frozen representation before the first consumer (setup kernels, windows, normalisers), whatever the
        # caller passed (a direct replay's own array): a no-copy view of the stage's component_evidence.
        evidence = frozen(evidence, np.float32)
        self.evidence = evidence
        # Coded leaves: the called allele or -2 - local row where uncalled (objective.pair_code7).
        self.leaves = List(
            [
                np.ascontiguousarray(
                    np.where(
                        panel.inference >= 0, panel.inference, -2 - np.arange(panel.rows, dtype=np.int16)[:, None]
                    ),
                    np.int16,
                )
                for panel in self.panels
            ]
        )
        self.offsets = np.asarray([0, *np.cumsum([panel.sites for panel in self.panels])], np.int64)
        self.penalty = SWITCH_PENALTY
        self.samples = int(evidence.shape[0])
        self.bin_size = max(int(min_sites_per_bin), math.ceil(int(self.offsets[-1]) / max_bins))
        # The objective's sites are the kept sites with at least one called
        # prepared row (the assembly's usable sites, join/emissions.py; the binned surrogate's mask). A site
        # where every prepared row is uncalled is excluded: there every panel's pairs are unknown-allele codes and
        # the site would only weigh same-row against different-row pairs through the uniform allele prior.
        self.kept = np.concatenate(
            [np.asarray(panel.kept, np.bool_) & np.any(panel.inference >= 0, axis=0) for panel in self.panels]
        )
        self.partial = self.kept & np.concatenate([np.any(panel.inference < 0, axis=0) for panel in self.panels])
        informative = informative_samples(evidence, self.kept)
        self.count_samples = None if informative.all() or not informative.any() else np.flatnonzero(informative)
        self.emissions = self._binned_emissions()
        positions = np.concatenate([np.asarray(panel.positions, np.int64) for panel in self.panels])
        self.hazards = map_hazards(positions, genetic_map, reset_rate, map_probability_floor)
        self.genetic_map = genetic_map
        self.reset_rate = float(reset_rate)
        self.map_probability_floor = float(map_probability_floor)
        self.windows = windows_factory(evidence, self.kept, self.partial, self.hazards)
        self.message_step = message_step
        self._normalisers = {}
        self._cost_lock = threading.Lock()

    def normalisers(self, founders):
        """float64 (S,) per-sample normalisers of K = founders (window.map_normalisers), cached."""
        key = int(founders)
        with self._cost_lock:
            found = self._normalisers.get(key)
        if found is None:
            found = map_normalisers(self.evidence, self.kept, self.hazards, float(founders))
            with self._cost_lock:
                found = self._normalisers.setdefault(key, found)
        return found

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
            outputs.append(np.zeros((self.samples, panel.rows * (panel.rows + 1) // 2, bins), np.float64))
        binned_emissions(
            self.evidence, self.offsets, alleles, keeps, np.asarray(full, np.bool_), self.bin_size, outputs
        )
        return list(outputs)


__all__ = ["binned_emissions", "map_hazards", "frozen", "component_evidence", "ComponentWorkspace"]
