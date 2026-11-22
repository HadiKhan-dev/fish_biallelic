"""One pilot block of the observation pilot: discovery's own strict fit, decoded under discovery's binned switching
model.

A pilot block is fitted by the real discovery block search (science.discovery.run.discover_block) at the strict rate.
Its founders (latent alleles, unknown allele -1) plus one unknown copy (allele unknown, its own latent row) define the
states of a sample: unordered pairs of rows. Under discovery's model a state costs, per observed site,

    -log[(1 - eps0) P + eps0 B] + lambda (number of unknown copies),

P the state's predictive likelihood (science.shared.emissions: an unknown allele is Bernoulli(1/2), two copies of one
row share it), B the cohort likelihood and lambda discovery's unknown-copy cost, and a change of state between
adjacent bins costs discovery's switch cost. decode_units takes each sample's Viterbi path and returns, per sample,
the log likelihood log[(1 - e) P + e B] summed over its cells of known states (no unknown copy) at every rate e of a
grid, and the count of those cells: one (sample, block) unit of the outlier-rate estimate
(science.observation.estimate).

Samples are decoded in parallel; a sample's path and sums do not depend on the others or on the thread count.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from numba import prange, types

from ...core.kernels import kernel
from ..discovery.api import bin_layout
from ..shared.emissions import pair_code, predictive_likelihood

__all__ = ["pilot_block_indices", "decode_units"]


def pilot_block_indices(contig: str, n_blocks: int, count: int) -> np.ndarray:
    """int64[count] ascending: the ``count`` blocks of ``contig`` (of ``n_blocks``) with the smallest keyed hashes,
    a choice that depends only on the contig's name and the block indices."""
    count = min(int(count), int(n_blocks))
    keys = [hashlib.sha256(f"observation-pilot/{contig}/{index}".encode()).digest() for index in range(n_blocks)]
    chosen = sorted(range(n_blocks), key=lambda index: keys[index])[:count]
    return np.asarray(sorted(chosen), dtype=np.int64)


_B1 = types.Array(types.boolean, 2, "C", readonly=True)
_F3 = types.Array(types.float64, 3, "C", readonly=True)
_F2 = types.Array(types.float64, 2, "C", readonly=True)
_F1 = types.Array(types.float64, 1, "C", readonly=True)


@kernel(
    overloads=(
        (
            _F3, _F2, _B1, types.Array(types.int8, 2, "C", readonly=True),
            types.Array(types.int64, 1, "C", readonly=True), types.int64,
            types.float64, types.float64, types.float64, _F1,
            types.Array(types.float64, 2, "C"), types.Array(types.int64, 1, "C"),
        ),
    ),
    cache=True,
    parallel=True,
    nogil=True,
)
def _decode(
    likelihoods, cohort, observed, alleles, bin_edges, unknown_row,
    decode_rate, unknown_penalty, switch_penalty, grid,
    unit_ll, unit_cells,
):
    n_samples, n_sites, _ = likelihoods.shape
    rows = alleles.shape[0]
    n_states = rows * (rows + 1) // 2
    n_bins = bin_edges.shape[0] - 1
    first_row = np.empty(n_states, dtype=np.int64)
    second_row = np.empty(n_states, dtype=np.int64)
    unknown = np.empty(n_states, dtype=np.int64)
    state = 0
    for a in range(rows):
        for b in range(a, rows):
            first_row[state] = a
            second_row[state] = b
            unknown[state] = int(a == unknown_row) + int(b == unknown_row)
            state += 1
    for n in prange(n_samples):
        emission = np.zeros((n_states, n_bins))
        for st in range(n_states):
            a = first_row[st]
            b = second_row[st]
            same = a == b and a != unknown_row
            for k in range(n_bins):
                total = 0.0
                for site in range(bin_edges[k], bin_edges[k + 1]):
                    if not observed[n, site]:
                        continue
                    l0 = likelihoods[n, site, 0]
                    l1 = likelihoods[n, site, 1]
                    l2 = likelihoods[n, site, 2]
                    code = pair_code(alleles[a, site], alleles[b, site], same)
                    p = predictive_likelihood(l0, l1, l2, code)
                    background = cohort[site, 0] * l0 + cohort[site, 1] * l1 + cohort[site, 2] * l2
                    total += math.log((1.0 - decode_rate) * p + decode_rate * background)
                    total -= unknown_penalty * unknown[st]
                emission[st, k] = total
        # Viterbi with one switch cost between any two different states (ties keep the earlier state).
        score = emission[:, 0].copy()
        back = np.zeros((n_bins, n_states), dtype=np.int64)
        for k in range(1, n_bins):
            best = 0
            for st in range(1, n_states):
                if score[st] > score[best]:
                    best = st
            switched = score[best] - switch_penalty
            for st in range(n_states):
                if score[st] >= switched:
                    back[k, st] = st
                    score[st] = score[st] + emission[st, k]
                else:
                    back[k, st] = best
                    score[st] = switched + emission[st, k]
        path = np.empty(n_bins, dtype=np.int64)
        last = 0
        for st in range(1, n_states):
            if score[st] > score[last]:
                last = st
        path[n_bins - 1] = last
        for k in range(n_bins - 1, 0, -1):
            path[k - 1] = back[k, path[k]]
        cells = 0
        for k in range(n_bins):
            st = path[k]
            if unknown[st] != 0:
                continue
            a = first_row[st]
            b = second_row[st]
            same = a == b and a != unknown_row
            for site in range(bin_edges[k], bin_edges[k + 1]):
                if not observed[n, site]:
                    continue
                l0 = likelihoods[n, site, 0]
                l1 = likelihoods[n, site, 1]
                l2 = likelihoods[n, site, 2]
                p = predictive_likelihood(l0, l1, l2, pair_code(alleles[a, site], alleles[b, site], same))
                background = cohort[site, 0] * l0 + cohort[site, 1] * l1 + cohort[site, 2] * l2
                cells += 1
                for g in range(grid.shape[0]):
                    unit_ll[n, g] += math.log((1.0 - grid[g]) * p + grid[g] * background)
        unit_cells[n] = cells


def decode_units(
    likelihoods: np.ndarray,
    cohort: np.ndarray,
    observed: np.ndarray,
    latent: np.ndarray,
    *,
    snps_per_bin: int,
    unknown_penalty: float,
    switch_penalty: float,
    decode_rate: float,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(unit log likelihoods float64[N, G], unit cells int64[N]) of one pilot block (module docstring).
    ``likelihoods`` float64[N, L, 3] normalised genotype likelihoods at the block's kept sites, ``cohort`` float64[L, 3]
    their cohort genotype frequencies, ``observed`` bool[N, L] the cells the paths read and score, ``latent``
    int8[K, L] the fitted founders' alleles (-1 unknown). Bins are discovery's own
    (science.discovery.fixed_k.workspace.bin_layout)."""
    n_sites = likelihoods.shape[1]
    width, n_bins = bin_layout(n_sites, int(snps_per_bin))
    edges = np.minimum(np.arange(n_bins + 1, dtype=np.int64) * width, n_sites)
    rows = np.full((latent.shape[0] + 1, n_sites), -1, dtype=np.int8)
    rows[:-1] = latent
    n = likelihoods.shape[0]
    unit_ll = np.zeros((n, len(grid)))
    unit_cells = np.zeros(n, dtype=np.int64)

    def frozen(array, dtype):
        out = np.ascontiguousarray(array, dtype=dtype)
        out = out.copy() if out.flags.writeable else out
        out.flags.writeable = False
        return out

    _decode(
        frozen(likelihoods, np.float64), frozen(cohort, np.float64), frozen(observed, np.bool_),
        frozen(rows, np.int8), frozen(edges, np.int64), int(latent.shape[0]),
        float(decode_rate), float(unknown_penalty), float(switch_penalty),
        frozen(grid, np.float64), unit_ll, unit_cells,
    )
    return unit_ll, unit_cells
