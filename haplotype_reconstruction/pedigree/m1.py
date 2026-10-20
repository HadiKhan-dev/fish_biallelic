"""Genome-wide consistent M1 floor, fixed before calibration and resampling.

Retain identities with total raw M1 >= best eligible total - 60*T(total markers).
Retained identities keep raw chromosome scores. Every excluded identity takes
its child's common chromosome profile, whose sum is the genome cutoff:
 f_cj = b_cj - T_j/sum(T) * (sum(b_c) - cutoff_c).
All identity multiplicities and contamination are preserved downstream.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

# The existing screen softmax cutoff, now shared by ranks and all M1 consumers.
LOG_FLOOR = -60.0
MODEL = "genome-relative-m1-floor-retained-m2-v1"


def temperature(markers, config):
    blocks = np.maximum(np.ceil(np.asarray(markers, dtype=np.float64)
                                / config.preparation.markers_per_information_block), 1.0)
    return blocks ** config.panel.information_tempering_power


def floor_chromosome(one: np.ndarray, markers: int | float, eligibility, config) -> np.ndarray:
    """Return floored [child,parent] values; preserve ineligible entries.

    Component factors must be summed before this call. Empty eligible rows are
    unchanged. The transformation is idempotent and does not modify its input.
    """
    output = one.copy()
    shift = LOG_FLOOR * float(temperature(markers, config))
    for child in np.flatnonzero(eligibility.children):
        parents = np.flatnonzero(eligibility.parents[child])
        if not len(parents):
            continue
        values = one[child, parents]
        best = float(np.max(values))
        if np.isfinite(best):
            floor = best + shift
            output[child, parents] = np.maximum(values, floor)
    return output


def genome_profile(one, markers, eligibility, config):
    """Exact genome gate and exact-sum chromosome profile from RAW M1 scores.

    Gate/profile are computed once from raw full-genome scores, before resampling.
    The returned mask includes cutoff ties. Ineligible cells are never replaced.
    """
    temperatures = temperature(markers, config)
    genome_temperature = float(temperature(np.sum(markers), config))
    keep = np.zeros(one.shape[1:], dtype=bool)
    floors = np.full(one.shape[:2], -np.inf)
    best = np.full(one.shape[1], -np.inf)
    cutoff = np.full(one.shape[1], -np.inf)
    output = one.copy()
    for child in np.flatnonzero(eligibility.children):
        parents = np.flatnonzero(eligibility.parents[child])
        if not len(parents):
            continue
        values = one[:, child, parents]
        totals = np.sum(values, axis=0)
        best[child] = np.max(totals)
        cutoff[child] = best[child] + LOG_FLOOR * genome_temperature
        retained = totals >= cutoff[child]
        keep[child, parents] = retained
        chromosome_best = np.max(values, axis=1)
        floors[:, child] = chromosome_best - temperatures / np.sum(temperatures) * (
            np.sum(chromosome_best) - cutoff[child])
        output[:, child, parents[~retained]] = floors[:, child, None]
    return output, keep, floors, best, cutoff


def floor_genome(scores, eligibility, config):
    """Transform RAW chromosome aggregate scores once at the decision boundary."""
    one, _, _, _, _ = genome_profile(scores.one, scores.informative_markers, eligibility, config)
    return replace(scores, one=one)


def identity(config) -> dict:
    """Only model settings affecting aggregate M1 and therefore derived caches."""
    return dict(model=MODEL, chromosome_score_storage="raw", m2_panel="all-retained-eligible-pairs",
                centered_log_floor=LOG_FLOOR,
                markers_per_information_block=config.preparation.markers_per_information_block,
                information_tempering_power=config.panel.information_tempering_power)
