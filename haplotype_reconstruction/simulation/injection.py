"""Injected outliers (gate G1i, ADVICE2 11.4): replace ordered genotypes before any read is drawn.

For sample n and site s, with probability r, the ORDERED genotype (h0, h1) of the sample at the site is replaced by a
draw from the ordered cohort law (q0; q1 / 2, q1 / 2; q2) of the simulated cohort at that site (q the genotype shares of
the analysed samples' original haplotypes). Both read samplers then read the replaced alleles: the independent-site
sampler draws its counts from the replaced genotype; the fragment sampler keeps its fragments, their homologue
assignment and depths, and every fragment covering s reads the replaced allele of its own homologue (cross-site allele
noise preserved). A redraw that returns the same genotype is part of the model.

Patterns (``ReadModel.injected_outlier_pattern``) at the setting r:
* uniform: every sample and site at r;
* heterogeneous: each sample is heavy with probability about 1 / 10 (a hash of its id, not a fixed quota), heavy samples
  at 10 r_low and the rest at r_low, r_low = r / 1.9. The cohort mean is r only when exactly a tenth is heavy; with H
  heavy samples of N it is r (1 + 9 H / N) / 1.9. The heavy probability 10 r / 1.9 must not exceed 1 (r <= 0.19);
* clustered: runs of five consecutive sites start with probability r / 5 per sample and site; each site of a run is
  replaced by its own independent draw. Away from the contig ends a site is replaced with probability
  1 - (1 - r / 5) ** 5, slightly below r;
* misspecified: as uniform, but half of the replacements are forced heterozygous ((0, 1) or (1, 0) at 1 / 2 each).

Every sample draws from its own stream (simulation.injected_outliers, the contig seed and the sample's index in the
call), and the law is that of the samples passed. The whole analysed cohort is therefore injected in one call (as
simulation/source.py does): a call on a slice would restart the indices and use the slice's law. A rate of zero draws
nothing (the clean simulation is unchanged).
Truth keeps the original alleles (ancestry), the replaced ones and the mask (simulation.truth.ContigTruth).
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from ..core.random import stream

PATTERNS = ("uniform", "heterogeneous", "clustered", "misspecified")
RUN_SITES = 5  # the clustered pattern's run length (sites)
HETEROGENEOUS_SHARE = 10  # one sample in this many is at ten times the low rate
HETEROGENEOUS_FACTOR = 10.0
HETEROGENEOUS_DIVISOR = 1.9  # r_low = r / 1.9: mean (9 r_low + 10 r_low) / 10 = r when exactly a tenth is heavy


def _heavy(sample_id: str) -> bool:
    """The heterogeneous pattern's tenth: a stable hash of the sample id."""
    digest = hashlib.sha256(f"injected-outliers/{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % HETEROGENEOUS_SHARE == 0


def ordered_law(hap0: NDArray[np.int8], hap1: NDArray[np.int8]) -> NDArray[np.float64]:
    """float64 [L, 4]: the cumulative ordered cohort law over (0, 0), (0, 1), (1, 0), (1, 1) at every site."""
    dosage = hap0.astype(np.int64) + hap1.astype(np.int64)
    n = max(dosage.shape[0], 1)
    q0 = (dosage == 0).sum(axis=0) / n
    q1 = (dosage == 1).sum(axis=0) / n
    law = np.stack((q0, q0 + q1 / 2.0, q0 + q1, np.ones_like(q0)), axis=1)
    return law


def inject(
    hap0: NDArray[np.int8],
    hap1: NDArray[np.int8],
    rate: float,
    pattern: str,
    seed: int,
    sample_ids: tuple[str, ...],
) -> tuple[NDArray[np.int8], NDArray[np.int8], NDArray[np.bool_]]:
    """(replaced hap0, replaced hap1, mask bool [N, L]) of the analysed samples' haplotypes int8 [N, L]."""
    if pattern not in PATTERNS:
        raise ValueError(f"unknown injected-outlier pattern {pattern!r}")
    if pattern == "heterogeneous" and rate * HETEROGENEOUS_FACTOR / HETEROGENEOUS_DIVISOR > 1.0:
        raise ValueError("heterogeneous injected outliers need rate <= 0.19 (heavy samples at 10 r / 1.9)")
    out0, out1 = hap0.copy(), hap1.copy()
    samples, sites = hap0.shape
    mask = np.zeros((samples, sites), dtype=np.bool_)
    if rate <= 0 or sites == 0:
        return out0, out1, mask
    law = ordered_law(hap0, hap1)
    low = rate / HETEROGENEOUS_DIVISOR
    for n in range(samples):
        rng = stream("simulation.injected_outliers", seed=seed, index=n)
        if pattern == "clustered":
            starts = np.flatnonzero(rng.random(sites) < rate / RUN_SITES)
            hit = np.zeros(sites, dtype=np.bool_)
            for s in starts:
                hit[s : s + RUN_SITES] = True
        else:
            r = rate
            if pattern == "heterogeneous":
                r = low * (HETEROGENEOUS_FACTOR if _heavy(sample_ids[n]) else 1.0)
            hit = rng.random(sites) < r
        at = np.flatnonzero(hit)
        u = rng.random(len(at))
        state = (u[:, None] >= law[at, :3]).sum(axis=1)  # 0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)
        if pattern == "misspecified":
            forced = rng.random(len(at)) < 0.5
            state = np.where(forced, 1 + (rng.random(len(at)) < 0.5), state)
        out0[n, at] = (state >= 2).astype(np.int8)
        out1[n, at] = ((state == 1) | (state == 3)).astype(np.int8)
        mask[n, at] = True
    return out0, out1, mask
