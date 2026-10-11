"""Explicit generating designs and observation selection, never inference metadata."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np
from scipy.special import ndtri


@dataclass(frozen=True)
class SimulationDesign:
    backcross_fraction: float = 0.0
    backcross_start_generation: int = 3
    observe_generations: tuple[int, ...] = ()
    observed_fraction: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, 'observe_generations', tuple(self.observe_generations))
        if not 0 <= self.backcross_fraction <= 1:
            raise ValueError("backcross_fraction must be between zero and one")
        if self.backcross_start_generation < 2:
            raise ValueError("backcrosses need a generated individual and one actual parent")
        if not 0 < self.observed_fraction <= 1:
            raise ValueError("observed_fraction must be in (0,1]")
        if any(g < 1 for g in self.observe_generations) or len(set(self.observe_generations)) != len(self.observe_generations):
            raise ValueError("observe_generations must be unique positive cohort numbers")

    def record(self):
        return asdict(self)


_FRAGMENT_FIELDS = ("read_length_bp", "fragment_mean_bp", "fragment_sd_bp")


@dataclass(frozen=True)
class ReadModel:
    """Generator-only perturbations. Inference retains its own 2% read model.

    ``read_length_bp = 0`` keeps the historical sampler (independent reads at
    every site). A positive value draws paired-end fragments instead, so sites
    closer than a fragment share reads (see :func:`fragment_read_sampling`).
    """
    error_rate: float = 0.02
    depth_cv: float = 0.0
    heterozygote_alt_probability: float = 0.5
    dropout_fraction: float = 0.0
    read_length_bp: int = 0
    fragment_mean_bp: float = 375.0
    fragment_sd_bp: float = 150.0

    def __post_init__(self):
        if not 0 < self.error_rate < 0.5:
            raise ValueError("generating error_rate must be between zero and 0.5")
        if not math.isfinite(self.depth_cv) or self.depth_cv < 0:
            raise ValueError("depth_cv must be finite and nonnegative")
        if not 0 < self.heterozygote_alt_probability < 1:
            raise ValueError("heterozygote_alt_probability must be in (0,1)")
        if not 0 <= self.dropout_fraction < 1:
            raise ValueError("dropout_fraction must be in [0,1)")
        if int(self.read_length_bp) != self.read_length_bp or self.read_length_bp < 0:
            raise ValueError("read_length_bp must be a nonnegative integer")
        if self.read_length_bp:
            if not (math.isfinite(self.fragment_mean_bp) and self.fragment_mean_bp > 0
                    and math.isfinite(self.fragment_sd_bp) and self.fragment_sd_bp >= 0):
                raise ValueError("fragment length mean must be positive and its sd nonnegative")

    def record(self):
        # Legacy (per-site) records keep their historical keys, so existing
        # simulation run specifications and checkpoints stay compatible.
        record = asdict(self)
        if not self.read_length_bp:
            for key in _FRAGMENT_FIELDS:
                record.pop(key)
        return record


def observed_indices(pedigree, design, seed):
    """Select rows only after the entire biological pedigree has been generated."""
    cohort = pedigree.Generation.str.removeprefix("F").astype(int).to_numpy()
    if design.observe_generations and set(design.observe_generations)-set(cohort):
        raise ValueError("an observed cohort was not generated")
    keep = np.arange(len(pedigree))
    if design.observe_generations:
        keep = keep[np.isin(cohort, design.observe_generations)]
    if design.observed_fraction < 1:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 3_000_000]))
        keep = np.sort(rng.choice(keep, size=int(len(keep)*design.observed_fraction), replace=False))
    if len(keep) < 3:
        raise ValueError("observation selection retains fewer than three individuals")
    return keep


def perturb_read_sampling(hap0, hap1, read_depth, model, rng):
    """Depth heterogeneity/one contiguous dropout tract per sample.

    The expected retained depth stays at read_depth: increase the non-dropout
    rate by 1/(1-realized marker dropout fraction), and normalize sample factors
    to mean one. This is a robustness generator, not a sequencing likelihood.
    """
    samples, sites = hap0.shape
    factor = np.ones(samples)
    if model.depth_cv:
        sigma = np.sqrt(np.log1p(model.depth_cv**2))
        factor = rng.lognormal(-sigma*sigma/2, sigma, size=samples)
        factor /= factor.mean()
    width = int(sites*model.dropout_fraction)
    lam = read_depth*factor[:, None]/(1-width/sites)
    counts = rng.poisson(lam=lam, size=(samples, sites))
    if width:
        starts = rng.integers(0, sites-width+1, size=samples)
        for sample, start in enumerate(starts):
            counts[sample, start:start+width] = 0
    dosage = hap0+hap1
    alt = np.zeros_like(counts)
    for genotype, probability in enumerate(
            (model.error_rate, model.heterozygote_alt_probability, 1-model.error_rate)):
        mask = dosage == genotype
        if np.any(mask):
            alt[mask] = rng.binomial(counts[mask], probability)
    return np.stack((counts-alt, alt), axis=-1).astype(int, copy=False)


def fragment_read_sampling(hap0, hap1, positions, read_depth, model, rng):
    """Paired-end fragments whose reads are shared by neighbouring sites.

    Per sample, fragment starts form a Poisson process along the contig. A
    fragment has insert length F ~ Normal(fragment_mean_bp, fragment_sd_bp),
    truncated to [50 bp, mean + 6 sd]; it comes from one of the sample's two
    haplotypes with probability 1/2 and is read over read_length_bp at each
    end (bases of overlapping mates counted once). Every site a fragment
    covers reports that haplotype's allele, flipped with error_rate. The start
    rate is depth / E[bases covered per fragment], so every site keeps the
    legacy Poisson(depth) marginal; only neighbouring sites become dependent.
    depth_cv and dropout_fraction act as in :func:`perturb_read_sampling`. An
    allelic bias p = heterozygote_alt_probability != 0.5 is applied per read at
    heterozygous sites: a read carrying the disfavoured allele reports the
    other one with probability |2p - 1|, so the per-cell law stays Binomial(n, p).
    """
    samples, sites = hap0.shape
    positions = np.asarray(positions, dtype=np.int64)
    if positions.shape != (sites,) or np.any(np.diff(positions) < 0):
        raise ValueError("fragment sampling needs sorted positions aligned with the haplotypes")
    r = int(model.read_length_bp)
    mean, sd = float(model.fragment_mean_bp), float(model.fragment_sd_bp)
    f_min, f_max = 50, int(math.ceil(mean + 6 * sd)) + 1
    # Expected bases covered per fragment under the exact sampled length law
    # (normal, rounded, clipped to [f_min, f_max]) evaluated at fixed quantiles.
    quantiles = (np.arange(100_000) + 0.5) / 100_000
    lengths = np.clip(np.rint(mean + sd * ndtri(quantiles)), f_min, f_max)
    covered_per_fragment = float(np.mean(np.minimum(lengths, 2 * r)))
    factor = np.ones(samples)
    if model.depth_cv:
        sigma = np.sqrt(np.log1p(model.depth_cv**2))
        factor = rng.lognormal(-sigma*sigma/2, sigma, size=samples)
        factor /= factor.mean()
    width = int(sites*model.dropout_fraction)
    lo = int(positions[0]) - f_max
    span = int(positions[-1]) + 1 - lo
    counts = np.zeros((samples, sites), dtype=np.int64)
    alt = np.zeros((samples, sites), dtype=np.int64)
    haplotypes = (np.asarray(hap0), np.asarray(hap1))
    for sample in range(samples):
        rate = read_depth * factor[sample] / (1 - width/sites) / covered_per_fragment
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
        bias = model.heterozygote_alt_probability - 0.5
        if bias:
            favoured = 1 if bias > 0 else 0
            convert = het & (allele != favoured) & (rng.random(len(site)) < 2 * abs(bias))
            allele[convert] = favoured
        # As in the legacy sampler, the error rate acts on homozygous cells;
        # at a heterozygous cell a symmetric error leaves Binomial(n, p) intact
        # only for p = 1/2, so it is not applied there.
        allele ^= (~het) & (rng.random(len(site)) < model.error_rate)
        counts[sample] = np.bincount(site, minlength=sites)
        alt[sample] = np.bincount(site, weights=allele, minlength=sites).astype(np.int64)
    if width:
        starts = rng.integers(0, sites-width+1, size=samples)
        for sample, first in enumerate(starts):
            counts[sample, first:first+width] = 0
            alt[sample, first:first+width] = 0
    return np.stack((counts-alt, alt), axis=-1).astype(int, copy=False)
