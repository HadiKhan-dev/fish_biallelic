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

Optional site factors are mean-one lognormal draws shared by all samples (one per site,
or per occupied physical window). Optional independent gamma cell factors have mean one
and variance count_dispersion. Conditional depth is Poisson in both samplers, hence gamma
mixing gives negative-binomial marginals, Var(D | sample, site) = mu + alpha * mu**2.
Fragments are drawn at the maximum target factor per sample and site-hits thinned to their
target intensity, retaining the shared fragment's homologue. No extra draws or arithmetic
on the historical rates occur when these options are off.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri

from ..core.errors import SettingsError
from .design import ReadModel
from .parameters import FRAGMENT_QUANTILES, FRAGMENT_SD_LIMIT, MINIMUM_FRAGMENT_BP

# Fragment-length quantiles used to compute the expected bases covered per fragment.


def sample_reads(
    hap0: NDArray[np.int8],
    hap1: NDArray[np.int8],
    positions: NDArray[np.int64],
    depth: float,
    model: ReadModel,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """REF/ALT read counts int64[N, L, 2] of individuals with haplotypes hap0, hap1 (int8[N, L])."""
    if model.read_length_bp:
        return _fragment_reads(hap0, hap1, positions, depth, model, rng)
    return _site_reads(hap0, hap1, positions, depth, model, rng)


def _sample_factors(samples, model, rng):
    factor = np.ones(samples)
    if model.depth_cv:
        sigma = np.sqrt(np.log1p(model.depth_cv**2))
        factor = rng.lognormal(-sigma * sigma / 2, sigma, size=samples)
        factor /= factor.mean()
    return factor


def _site_factors(positions, model, rng):
    """Population mean one, without finite-contig renormalisation (which changes the law).

    Physical windows are anchored at bp zero; only occupied windows consume draws.
    Window zero uses independent sites. Spatial sharing is separate from fragment sharing.
    """
    if not model.site_depth_cv:
        return np.ones(len(positions))
    sigma = np.sqrt(np.log1p(model.site_depth_cv**2))
    if model.site_depth_window_bp:
        windows, inverse = np.unique(np.asarray(positions) // model.site_depth_window_bp, return_inverse=True)
        return rng.lognormal(-sigma * sigma / 2, sigma, size=len(windows))[inverse]
    return rng.lognormal(-sigma * sigma / 2, sigma, size=len(positions))


def _fragment_envelopes(positions, target, lo, span, f_max):
    """Start windows of ``f_max`` bp over [lo, lo + span), with each window's thinning envelope.

    A fragment starting in window k covers sites only in [start, start + f_max), inside windows
    k and k + 1, so max(block maximum k, block maximum k + 1) bounds every target such a
    fragment can reach. Returns window starts, widths (the last may be shorter) and envelopes;
    a window that reaches no site has envelope zero and draws no fragment.
    """
    edges = lo + np.arange(0, span, f_max, dtype=np.int64)
    widths = np.minimum(f_max, lo + span - edges)
    first = np.searchsorted(positions, edges)
    stop = np.searchsorted(positions, edges + widths)
    block = np.zeros(len(edges))
    nonempty = stop > first
    if np.any(nonempty):
        # Blocks are disjoint and contiguous: each nonempty block runs to the next one's first site.
        block[nonempty] = np.maximum.reduceat(target, first[nonempty])
    local = block.copy()
    local[:-1] = np.maximum(block[:-1], block[1:])
    return edges, widths, local


def _site_reads(hap0, hap1, positions, depth, model, rng):
    samples, sites = hap0.shape
    factor = _sample_factors(samples, model, rng)
    width = int(sites * model.dropout_fraction)
    lam = depth * factor[:, None] / (1 - width / sites)
    if model.site_depth_cv:
        lam = lam * _site_factors(positions, model, rng)[None, :]
    if model.count_dispersion:
        lam = lam * rng.gamma(1 / model.count_dispersion, model.count_dispersion, size=(samples, sites))
    counts = rng.poisson(lam=lam, size=(samples, sites))
    if width:
        starts = rng.integers(0, sites - width + 1, size=samples)
        for sample, start in enumerate(starts):
            counts[sample, start : start + width] = 0
    dosage = hap0 + hap1
    alt = np.zeros_like(counts)
    for genotype, probability in enumerate(
        (model.error_rate, model.heterozygote_alt_probability, 1 - model.error_rate)
    ):
        mask = dosage == genotype
        if np.any(mask):
            alt[mask] = rng.binomial(counts[mask], probability)
    return np.stack((counts - alt, alt), axis=-1).astype(np.int64, copy=False)


def _fragment_reads(hap0, hap1, positions, depth, model, rng):
    samples, sites = hap0.shape
    positions = np.asarray(positions, dtype=np.int64)
    if positions.shape != (sites,) or np.any(np.diff(positions) < 0):
        raise SettingsError("fragment sampling needs sorted positions aligned with the haplotypes")
    r = int(model.read_length_bp)
    mean, sd = float(model.fragment_mean_bp), float(model.fragment_sd_bp)
    f_min, f_max = MINIMUM_FRAGMENT_BP, int(math.ceil(mean + FRAGMENT_SD_LIMIT * sd)) + 1
    # Expected bases covered per fragment under the sampled length law (normal, rounded, clipped)
    # evaluated at fixed quantiles.
    quantiles = (np.arange(FRAGMENT_QUANTILES) + 0.5) / FRAGMENT_QUANTILES
    lengths = np.clip(np.rint(mean + sd * ndtri(quantiles)), f_min, f_max)
    covered_per_fragment = float(np.mean(np.minimum(lengths, 2 * r)))
    factor = _sample_factors(samples, model, rng)
    site_factor = _site_factors(positions, model, rng) if model.site_depth_cv else None
    width = int(sites * model.dropout_fraction)
    lo = int(positions[0]) - f_max
    span = int(positions[-1]) + 1 - lo
    counts = np.zeros((samples, sites), dtype=np.int64)
    alt = np.zeros((samples, sites), dtype=np.int64)
    haplotypes = (np.asarray(hap0), np.asarray(hap1))
    bias = model.heterozygote_alt_probability - 0.5
    for sample in range(samples):
        rate = depth * factor[sample] / (1 - width / sites) / covered_per_fragment
        target = site_factor
        if model.count_dispersion:
            target = rng.gamma(1 / model.count_dispersion, model.count_dispersion, size=sites)
            if site_factor is not None:
                target *= site_factor
        if target is None:
            n = int(rng.poisson(rate * span))
            start = lo + rng.integers(0, span, size=n)
        else:
            # Conditional Poisson thinning gives each cell its target rate while retaining
            # common fragment starts/lengths/homologues across sites. Gamma mixing then
            # gives exact NB marginals, unlike duplicating already sampled read counts.
            # Starts are drawn per window of f_max bp at the window's own envelope (the
            # largest target any fragment starting there can reach), so candidate
            # fragments scale with local, not contig-wide, maxima of the target.
            edges, widths, local = _fragment_envelopes(positions, target, lo, span, f_max)
            per_window = rng.poisson(rate * local * widths)
            n = int(per_window.sum())
            start = np.repeat(edges, per_window) + rng.integers(0, np.repeat(widths, per_window))
            envelope = np.repeat(local, per_window)
        length = np.clip(np.rint(rng.normal(mean, sd, size=n)), f_min, f_max).astype(np.int64)
        which = rng.integers(0, 2, size=n)
        first_end = start + np.minimum(length, r)
        second_start = np.maximum(start + length - r, first_end)
        second_end = start + length
        intervals = (
            (np.searchsorted(positions, start), np.searchsorted(positions, first_end)),
            (np.searchsorted(positions, second_start), np.searchsorted(positions, second_end)),
        )
        site_parts, hap_parts, fragment_parts = [], [], []
        for first, stop in intervals:
            size = stop - first
            keep = size > 0
            if not np.any(keep):
                continue
            first, size, owner = first[keep], size[keep], which[keep]
            offsets = np.arange(int(size.sum())) - np.repeat(np.cumsum(size) - size, size)
            site_parts.append(np.repeat(first, size) + offsets)
            hap_parts.append(np.repeat(owner, size))
            if target is not None:
                fragment_parts.append(np.repeat(np.flatnonzero(keep), size))
        if not site_parts:
            continue
        site = np.concatenate(site_parts)
        owner = np.concatenate(hap_parts)
        if target is not None:
            # One uniform mark per fragment couples retention across all its hits.
            # Shared-hit intensity is proportional to min(target_i, target_j), so an
            # unrelated high-depth site cannot change local dependence via the envelope.
            mark = rng.random(n)
            fragment = np.concatenate(fragment_parts)
            keep = mark[fragment] < target[site] / envelope[fragment]
            site, owner = site[keep], owner[keep]
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
            counts[sample, first : first + width] = 0
            alt[sample, first : first + width] = 0
    return np.stack((counts - alt, alt), axis=-1).astype(np.int64, copy=False)
