"""Binned emissions of one joined batch, and the join's switch penalty and bin size.

The join scores a panel per sample by a Viterbi programme over unordered pairs of panel paths
(:mod:`.objective`) on *binned* emissions: block j's sites are cut into bins of ``sites_per_bin``
consecutive sites, and the emission of the founder pair (a, c) (local rows of block j) at bin t is
the sum over the bin's usable sites of the robust, floored, centred log likelihood

    e = max(log(0.99 P(g | a, c) + 0.01 / 3), -2) - log(1 / 3),

where P(g | a, c) is the sample's genotype likelihood (normalised to sum to one) under the pair's
predictive genotype distribution: the dosage a + c when both alleles are called; when an allele
is unknown, the seven predictive pair codes of the linker (an unknown allele is Bernoulli(1/2);
two founders copying the same prepared row share one latent allele, so a pair of copies of one
unknown allele is homozygous). Every allele is the block's pre-fill inference call.

Usable sites are kept (discovery) sites at which at least one founder of the block is called; a
sample whose likelihoods are uniform or sum to zero at a site contributes nothing there. Values
are float64, accumulated per (sample, pair, bin) in site order.
"""
from __future__ import annotations

import math
from typing import Sequence

import numba
import numpy as np
from numba import njit, prange

from ...maths.emissions import (CENTER, normalised_likelihoods, pair_code, predictive_likelihood,
                                robust_log_likelihood)

# The join's switch penalty below this many output sites (L1: 10 x 200-site blocks); above it the
# penalty grows as sqrt(sites / 2000).
_PENALTY = 20.0
_PENALTY_FLAT_SITES = 5000
_PENALTY_REFERENCE_SITES = 2000.0
# Bin size bounds: about 20 bins per input block, between 10 and 100 sites per bin.
_BINS_PER_BLOCK = 20
_MIN_SITES_PER_BIN = 10
_MAX_SITES_PER_BIN = 100


def switch_penalty(site_counts: Sequence[int]) -> float:
    """The join's penalty per change of the unordered state (nats).

    20 while the batch's mean block length times its block count (the output length) is at most
    5,000 sites, else 20 sqrt(output sites / 2,000). ``site_counts``: sites of each input block.
    """
    output_sites = float(np.mean(site_counts)) * len(site_counts)
    if output_sites <= _PENALTY_FLAT_SITES:
        return _PENALTY
    return _PENALTY * math.sqrt(output_sites / _PENALTY_REFERENCE_SITES)


def bin_size(site_counts: Sequence[int]) -> int:
    """Sites per bin for blocks of these lengths: mean sites // 20, clamped to [10, 100]."""
    mean_sites = np.mean(site_counts)
    return int(min(_MAX_SITES_PER_BIN, max(_MIN_SITES_PER_BIN, mean_sites // _BINS_PER_BLOCK)))


def sites_per_bin(site_counts: Sequence[int], max_bins: int) -> int:
    """The join's bin size for one batch: :func:`bin_size`, raised so that the batch has at most
    about ``max_bins`` bins (max(bin_size, ceil(total sites / max_bins)))."""
    return max(bin_size(site_counts), math.ceil(sum(int(n) for n in site_counts) / max_bins))


@njit(cache=True, inline="always")
def pair_index(first, second):
    """Packed index of an unordered local-row pair, independent of the block's row count."""
    high = max(first, second)
    return high * (high + 1) // 2 + min(first, second)


class EmissionLayout:
    """The binned emissions of one join in one flat float64 array.

    Element (s, a, c, t) of block j (sample s of the batch, local rows a and c of block j, bin t) is
    ``flat[start[j] + (s * P_j + pair_index(a, c)) * bins_j + t]``, where
    P_j = K_j(K_j + 1)/2; :meth:`block` gives block j's (samples, P_j, bins_j) view.
    The search's kernels read the flat array (taking a block's
    array out of a typed List inside a kernel updated its reference count, an atomic shared by all
    threads, which stopped the kernels scaling with threads).

    samples: S; founders: int64[n] K_j; bins: int64[n] bins of each block; start: int64[n] offset of
    each block in ``flat``; offsets: int64[n + 1] global bin index of each block's first bin (the
    last entry is the total bin count).
    """

    def __init__(self, samples: int, founders, bins):
        self.samples = int(samples)
        self.founders = np.asarray(founders, dtype=np.int64)
        self.bins = np.asarray(bins, dtype=np.int64)
        sizes = self.samples * (self.founders * (self.founders + 1) // 2) * self.bins
        self.start = np.zeros(len(sizes), dtype=np.int64)
        self.start[1:] = np.cumsum(sizes)[:-1]
        self.offsets = np.zeros(len(self.bins) + 1, dtype=np.int64)
        np.cumsum(self.bins, out=self.offsets[1:])
        self.flat = np.empty(int(sizes.sum()))

    @property
    def total_bins(self) -> int:
        return int(self.offsets[-1])

    def block(self, j: int) -> np.ndarray:
        """Block j's (samples, K_j(K_j + 1)/2, bins_j) packed-pair view."""
        founders, bins = int(self.founders[j]), int(self.bins[j])
        start = int(self.start[j])
        pairs = founders * (founders + 1) // 2
        return self.flat[start:start + self.samples * pairs * bins].reshape(
            (self.samples, pairs, bins))

    def arrays(self):
        """(flat, start, founders, bins): the arrays every join kernel takes first."""
        return self.flat, self.start, self.founders, self.bins


@njit(cache=True, parallel=True, nogil=True)
def _binned_log_emissions(flat, start, likelihoods, offset, alleles, source_rows, usable, complete,
                          sites_per_bin):
    """Fill one block's binned emissions into ``flat`` (EmissionLayout order, from ``start``).

    likelihoods: float[S, L_total, 3] the batch's evidence; the block's sites are the columns
    offset .. offset + L - 1. alleles: int8[K, L] inference calls; source_rows: int32[K, L];
    usable: kept sites with some called founder; complete: every founder called at the site.

    At a complete site only the three dosage codes occur, so only their logs are evaluated; the
    values are the same as with all seven (each code's log is computed alone).
    """
    founders, sites = alleles.shape
    bins = (sites + sites_per_bin - 1) // sites_per_bin
    width = (founders * (founders + 1) // 2) * bins
    for sample in prange(likelihoods.shape[0]):
        base = start + sample * width
        for index in range(width):
            flat[base + index] = 0.0
        logs = np.empty(7, np.float64)
        for site in range(sites):
            if not usable[site]:
                continue
            p0, p1, p2 = likelihoods[sample, offset + site]
            informative, p0, p1, p2 = normalised_likelihoods(p0, p1, p2)
            if not informative:
                continue
            column = site // sites_per_bin
            if complete[site]:
                for code in range(3):
                    logs[code] = robust_log_likelihood(predictive_likelihood(p0, p1, p2, code)) - CENTER
                for first in range(founders):
                    allele = alleles[first, site]
                    for second in range(first, founders):
                        value = logs[allele + alleles[second, site]]
                        flat[base + pair_index(first, second) * bins + column] += value
            else:
                for code in range(7):
                    logs[code] = robust_log_likelihood(predictive_likelihood(p0, p1, p2, code)) - CENTER
                for first in range(founders):
                    for second in range(first, founders):
                        value = logs[pair_code(alleles[first, site], alleles[second, site],
                                               source_rows[first, site] == source_rows[second, site])]
                        flat[base + pair_index(first, second) * bins + column] += value


def binned_emissions(inputs, sites_per_bin: int, budget) -> EmissionLayout:
    """The binned emissions of every block of a join (``inputs``: JoinInputs).

    Each block reads its pre-fill inference calls, its kept mask and its source rows (founders
    copying one prepared row share an unknown allele); the evidence is ``inputs.gl``. The Numba
    team is re-read from ``budget`` before each block.
    """
    blocks = inputs.blocks
    likelihoods = inputs.gl
    layout = EmissionLayout(inputs.sample_count, [block.founders for block in blocks],
                            [math.ceil(len(block.positions) / sites_per_bin) for block in blocks])
    for j, block in enumerate(blocks):
        alleles = np.ascontiguousarray(block.inference, dtype=np.int8)
        called = alleles >= 0
        usable = np.ascontiguousarray(np.asarray(block.kept, dtype=np.bool_) & called.any(axis=0))
        complete = np.ascontiguousarray(called.all(axis=0))
        numba.set_num_threads(min(max(1, int(budget.threads())), numba.config.NUMBA_NUM_THREADS))
        _binned_log_emissions(layout.flat, int(layout.start[j]), likelihoods, int(inputs.offsets[j]),
                              alleles, np.ascontiguousarray(block.source_rows, dtype=np.int32), usable,
                              complete, int(sites_per_bin))
    return layout
