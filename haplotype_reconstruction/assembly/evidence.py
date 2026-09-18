"""Chromosome-lifetime raw evidence shared by feedback and final assembly.

No candidate results are cached here. Raw arrays are read-only inputs throughout
one reconstruction call; keep-flag changes rebuild the neutral view. This
avoids repeated full-array validation, hashing, masking and per-level casts.
"""
from dataclasses import dataclass, field
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def finite_nonnegative(probabilities):
    valid = np.ones(probabilities.shape[0], np.bool_)
    for sample in prange(probabilities.shape[0]):
        for site in range(probabilities.shape[1]):
            for genotype in range(3):
                value = probabilities[sample, site, genotype]
                if not np.isfinite(value) or value < 0.0:
                    valid[sample] = False
    return np.all(valid)


@njit(parallel=True, cache=True)
def neutral_evidence(probabilities, observed, kept):
    samples, sites, _ = probabilities.shape
    result = np.empty((samples, sites, 3), np.float64)
    mask = np.empty((samples, sites), np.bool_)
    for sample in prange(samples):
        for site in range(sites):
            seen = observed[sample, site] and kept[site]
            mask[sample, site] = seen
            for genotype in range(3):
                result[sample, site, genotype] = (
                    probabilities[sample, site, genotype] if seen else 1.0 / 3.0)
    return result, mask


@njit(parallel=True, cache=True)
def float32_evidence(probabilities):
    result = np.empty(probabilities.shape, np.float32)
    for sample in prange(probabilities.shape[0]):
        result[sample] = probabilities[sample]
    return result


@dataclass
class ChromosomeEvidence:
    probabilities: np.ndarray
    sites: np.ndarray
    observed: np.ndarray
    digests: dict
    _kept: np.ndarray | None = field(default=None, init=False, repr=False)
    _neutral: np.ndarray | None = field(default=None, init=False, repr=False)
    _observed: np.ndarray | None = field(default=None, init=False, repr=False)

    def check_arrays(self, probabilities, sites, observed):
        if (probabilities is not self.probabilities or sites is not self.sites
                or observed is not self.observed):
            raise ValueError("chromosome evidence belongs to different raw arrays")

    def prepare(self, blocks):
        indices, kept = block_indices_and_keep(blocks, self.sites)
        if self._kept is None or not np.array_equal(kept, self._kept):
            self._neutral, self._observed = neutral_evidence(
                self.probabilities, self.observed, kept)
            self._kept = kept
        return block_views(self._neutral, self._observed, indices)


def block_indices_and_keep(blocks, sites):
    indices = []
    kept = np.ones(len(sites), np.bool_)
    for block in blocks:
        index = np.searchsorted(sites, block.positions)
        flags = getattr(block, "keep_flags", None)
        if flags is not None:
            flags = np.asarray(flags) > 0
            if flags.shape != index.shape:
                raise ValueError("keep_flags must match block positions")
            kept[index] = flags
        indices.append(index)
    return indices, kept


def block_views(neutral, observed, indices):
    evidence, masks = [], []
    for index in indices:
        # The normal 200-SNP layout is contiguous. Preprocessing owns its
        # block-local copies; do not allocate an extra chromosome of copies.
        selection = (slice(int(index[0]), int(index[-1]) + 1)
                     if len(index) == index[-1] - index[0] + 1 else index)
        evidence.append(neutral[:, selection,:])
        masks.append(observed[:, selection])
    return neutral, tuple(evidence), tuple(masks)
