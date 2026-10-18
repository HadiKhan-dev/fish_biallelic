"""Release of a selected panel's alleles and its unknown-copy summary.

A founder allele is released when its one-bit conditional probability (the exact marginal
likelihood ratio of flipping it, every other bit and pi fixed, with equal prior odds) reaches
``min_call_probability`` and the samples whose own evidence favours it (their per-sample exact
gain of flipping it is below -``favourable_margin``) include ``min_supporters`` carriers: as many
samples each more likely than not carrying the founder at the site, or that much expected carrier
mass (the mass alone sits just below 2 for a founder with exactly two near-certain carriers; the
count alone would withhold diffusely supported sites). Released alleles are the panel's own.

The unknown-copy summary for assembly: per sample the largest marginal MAP number of unknown
copies over its observed sites (2 without depth) and their fraction over samples with depth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from ..model.local_panels import LocalPanel, unknown_copy_mass
from .path_model import release_statistics


@dataclass(frozen=True)
class PathRelease:
    """calls int8[K, L_kept] (-1 withheld); probability float64[K, L_kept] of the current
    allele; supporters int64[K, L_kept] favourable carriers by count; unknown_copy_fraction:
    posterior unknown-state occupancy over all copies (diagnostic); has_depth bool[N];
    unknown_slots int8[N]; unknown_mass float."""

    calls: np.ndarray
    probability: np.ndarray
    supporters: np.ndarray
    unknown_copy_fraction: float
    has_depth: np.ndarray
    unknown_slots: np.ndarray
    unknown_mass: float


def release(fit, prepared, config) -> PathRelease:
    statistics = release_statistics(fit.panel, fit.frequencies, prepared, config.favourable_margin)
    probability = expit(-statistics["conditional_flip_gains"])
    supporters, mass = statistics["supporters"], statistics["carrier_mass"]
    confident = ((probability >= config.min_call_probability)
                 & (np.maximum(supporters, mass) >= config.min_supporters))
    occupancy = statistics["occupancy"]
    has_depth = statistics["has_depth"]
    slots = statistics["unknown_slots"]
    return PathRelease(
        calls=np.where(confident, fit.panel, -1).astype(np.int8), probability=probability,
        supporters=supporters, unknown_copy_fraction=float(occupancy[-1].sum()) / float(occupancy.sum()),
        has_depth=has_depth, unknown_slots=slots, unknown_mass=unknown_copy_mass(slots, has_depth))


def local_panel(block: int, site_start: int, positions, kept, fit, path_release: PathRelease
                ) -> LocalPanel:
    """The LocalPanel of a selected fit: released calls, the fit's allele probabilities and the
    favourable carrier counts at the discovery sites; -1 / 0.5 / 0 elsewhere."""
    k = len(fit.panel)
    length = len(positions)
    calls = np.full((k, length), -1, dtype=np.int8)
    calls[:, kept] = path_release.calls
    alt_probability = np.full((k, length), .5)
    alt_probability[:, kept] = fit.allele_probability
    support = np.zeros((k, length))
    support[:, kept] = path_release.supporters
    return LocalPanel(block=int(block), site_start=int(site_start),
                      positions=np.ascontiguousarray(positions, dtype=np.int64),
                      kept=np.ascontiguousarray(kept, dtype=np.bool_), calls=calls,
                      alt_probability=alt_probability, support=support,
                      has_depth=path_release.has_depth, unknown_slots=path_release.unknown_slots,
                      unknown_mass=path_release.unknown_mass)
