"""Sequencing reads of simulated individuals (the generating read model, never the inference one).

Two samplers, chosen by ``ReadModel.read_length_bp``:

* independent sites (0): per cell, depth ~ Poisson(depth x sample factor / (1 - dropout share))
  and ALT reads ~ Binomial(depth, p_g), p = (error_rate, heterozygote_alt_probability,
  1 - error_rate); a lognormal sample factor (mean one, coefficient of variation depth_cv) and one
  contiguous dropout tract of dropout_fraction of the sites per sample are drawn only when
  configured. At the defaults this is the historical per-site sampler draw for draw;
* paired-end fragments (> 0): per sample, fragment starts form a Poisson process along the contig;
  a fragment has insert length F ~ Normal(fragment_mean_bp, fragment_sd_bp) rounded and clipped to
  [50 bp, mean + 6 sd], comes from one of the sample's two haplotypes with probability 1/2 and is
  read over read_length_bp at each end (bases of overlapping mates counted once). The start rate is
  depth / E[bases covered per fragment], so every site keeps the Poisson(depth) marginal while
  neighbouring sites become dependent. depth_cv and dropout_fraction act as above. An allelic bias
  p = heterozygote_alt_probability != 1/2 is applied per read at heterozygous sites (a read of the
  disfavoured allele reports the other one with probability |2p - 1|); the error rate acts on
  homozygous cells only.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri

from .design import ReadModel

# Fragment-length quantiles used to compute the expected bases covered per fragment.
_FRAGMENT_QUANTILES = 100_000
_MINIMUM_FRAGMENT_BP = 50


def sample_reads(hap0, hap1, positions, depth: float, model: ReadModel, rng: np.random.Generator) -> np.ndarray:
    """REF/ALT read counts int64[N, L, 2] of individuals with haplotypes hap0, hap1 (int8[N, L])."""
    if model.read_length_bp:
        return _fragment_reads(hap0, hap1, positions, depth, model, rng)
    return _site_reads(hap0, hap1, depth, model, rng)


def _sample_factors(samples, model, rng):
    factor = np.ones(samples)
    if model.depth_cv:
        sigma = np.sqrt(np.log1p(model.depth_cv ** 2))
        factor = rng.lognormal(-sigma * sigma / 2, sigma, size=samples)
        factor /= factor.mean()
    return factor


def _site_reads(hap0, hap1, depth, model, rng):
    samples, sites = hap0.shape
    factor = _sample_factors(samples, model, rng)
    width = int(sites * model.dropout_fraction)
    lam = depth * factor[:, None] / (1 - width / sites)
    counts = rng.poisson(lam=lam, size=(samples, sites))
    if width:
        starts = rng.integers(0, sites - width + 1, size=samples)
        for sample, start in enumerate(starts):
            counts[sample, start:start + width] = 0
    dosage = hap0 + hap1
    alt = np.zeros_like(counts)
    for genotype, probability in enumerate(
            (model.error_rate, model.heterozygote_alt_probability, 1 - model.error_rate)):
        mask = dosage == genotype
        if np.any(mask):
            alt[mask] = rng.binomial(counts[mask], probability)
    return np.stack((counts - alt, alt), axis=-1).astype(np.int64, copy=False)


def _fragment_reads(hap0, hap1, positions, depth, model, rng):
    samples, sites = hap0.shape
    positions = np.asarray(positions, dtype=np.int64)
    if positions.shape != (sites,) or np.any(np.diff(positions) < 0):
        raise ValueError("fragment sampling needs sorted positions aligned with the haplotypes")
    r = int(model.read_length_bp)
    mean, sd = float(model.fragment_mean_bp), float(model.fragment_sd_bp)
    f_min, f_max = _MINIMUM_FRAGMENT_BP, int(math.ceil(mean + 6 * sd)) + 1
    # Expected bases covered per fragment under the sampled length law (normal, rounded, clipped)
    # evaluated at fixed quantiles.
    quantiles = (np.arange(_FRAGMENT_QUANTILES) + 0.5) / _FRAGMENT_QUANTILES
    lengths = np.clip(np.rint(mean + sd * ndtri(quantiles)), f_min, f_max)
    covered_per_fragment = float(np.mean(np.minimum(lengths, 2 * r)))
    factor = _sample_factors(samples, model, rng)
    width = int(sites * model.dropout_fraction)
    lo = int(positions[0]) - f_max
    span = int(positions[-1]) + 1 - lo
    counts = np.zeros((samples, sites), dtype=np.int64)
    alt = np.zeros((samples, sites), dtype=np.int64)
    haplotypes = (np.asarray(hap0), np.asarray(hap1))
    bias = model.heterozygote_alt_probability - 0.5
    for sample in range(samples):
        rate = depth * factor[sample] / (1 - width / sites) / covered_per_fragment
        n = int(rng.poisson(rate * span))
        start = lo + rng.integers(0, span, size=n)
        length = np.clip(np.rint(rng.normal(mean, sd, size=n)), f_min, f_max).astype(np.int64)
        which = rng.integers(0, 2, size=n)
        first_end = start + np.minimum(length, r)
        second_start = np.maximum(start + length - r, first_end)
        second_end = start + length
        intervals = (
            (np.searchsorted(positions, start), np.searchsorted(positions, first_end)),
            (np.searchsorted(positions, second_start), np.searchsorted(positions, second_end)),
        )
        site_parts, hap_parts = [], []
        for first, stop in intervals:
            size = stop - first
            keep = size > 0
            if not np.any(keep):
                continue
            first, size, owner = first[keep], size[keep], which[keep]
            offsets = np.arange(int(size.sum())) - np.repeat(np.cumsum(size) - size, size)
            site_parts.append(np.repeat(first, size) + offsets)
            hap_parts.append(np.repeat(owner, size))
        if not site_parts:
            continue
        site = np.concatenate(site_parts)
        owner = np.concatenate(hap_parts)
        allele = np.where(owner == 0, haplotypes[0][sample, site], haplotypes[1][sample, site]).astype(np.int64)
        het = haplotypes[0][sample, site] != haplotypes[1][sample, site]
        if bias:
            favoured = 1 if bias > 0 else 0
            convert = het & (allele != favoured) & (rng.random(len(site)) < 2 * abs(bias))
            allele[convert] = favoured
        allele ^= (~het) & (rng.random(len(site)) < model.error_rate)
        counts[sample] = np.bincount(site, minlength=sites)
        alt[sample] = np.bincount(site, weights=allele, minlength=sites).astype(np.int64)
    if width:
        starts = rng.integers(0, sites - width + 1, size=samples)
        for sample, first in enumerate(starts):
            counts[sample, first:first + width] = 0
            alt[sample, first:first + width] = 0
    return np.stack((counts - alt, alt), axis=-1).astype(np.int64, copy=False)
