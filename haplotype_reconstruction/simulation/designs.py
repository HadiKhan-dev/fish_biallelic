"""Explicit generating designs and observation selection, never inference metadata."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np


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


@dataclass(frozen=True)
class ReadModel:
    """Generator-only perturbations. Inference retains its own 2% read model."""
    error_rate: float = 0.02
    depth_cv: float = 0.0
    heterozygote_alt_probability: float = 0.5
    dropout_fraction: float = 0.0

    def __post_init__(self):
        if not 0 < self.error_rate < 0.5:
            raise ValueError("generating error_rate must be between zero and 0.5")
        if not math.isfinite(self.depth_cv) or self.depth_cv < 0:
            raise ValueError("depth_cv must be finite and nonnegative")
        if not 0 < self.heterozygote_alt_probability < 1:
            raise ValueError("heterozygote_alt_probability must be in (0,1)")
        if not 0 <= self.dropout_fraction < 1:
            raise ValueError("dropout_fraction must be in [0,1)")

    def record(self):
        return asdict(self)


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
