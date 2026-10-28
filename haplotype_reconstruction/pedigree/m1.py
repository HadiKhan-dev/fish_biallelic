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


def _child_profile(values, temperatures, genome_temperature):
    """(retained bool [P], floors [C], best, cutoff) of one child from the RAW M1 scores [C, P] of its eligible
    parents."""
    # Each parent's chromosome scores contiguous (the layout of the dense reference's [C, child, parents] gather),
    # so the genome totals are the same pairwise sums on the dense and the per-edge layout.
    values = np.asfortranarray(values)
    totals = np.sum(values, axis=0)
    best = np.max(totals)
    cutoff = best + LOG_FLOOR * genome_temperature
    retained = totals >= cutoff
    chromosome_best = np.max(values, axis=1)
    floors = chromosome_best - temperatures / np.sum(temperatures) * (np.sum(chromosome_best) - cutoff)
    return retained, floors, best, cutoff


def genome_profile(one, markers, eligibility, config):
    """Exact genome gate and exact-sum chromosome profile from RAW M1 scores [C, N, N] (reference paths).

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
        retained, floors[:, child], best[child], cutoff[child] = _child_profile(
            one[:, child, parents], temperatures, genome_temperature)
        keep[child, parents] = retained
        output[:, child, parents[~retained]] = floors[:, child, None]
    return output, keep, floors, best, cutoff


def floor_genome(scores, eligibility, config):
    """Transform RAW chromosome aggregate scores once at the decision boundary (GenomeScores, M1 per eligible
    edge [C, E]); the same profile as ``genome_profile``."""
    temperatures = temperature(scores.informative_markers, config)
    genome_temperature = float(temperature(np.sum(scores.informative_markers), config))
    starts = scores.edge_starts()
    one = scores.one.copy()
    for child in np.flatnonzero(eligibility.children):
        start, stop = starts[child], starts[child + 1]
        if start == stop:
            continue
        retained, floors, _, _ = _child_profile(scores.one[:, start:stop], temperatures, genome_temperature)
        one[:, start + np.flatnonzero(~retained)] = floors[:, None]
    return replace(scores, one=one)


def identity(config) -> dict:
    """Only model settings affecting aggregate M1 and therefore derived caches."""
    return dict(model=MODEL, chromosome_score_storage="raw", m2_panel="all-retained-eligible-pairs",
                centered_log_floor=LOG_FLOOR,
                markers_per_information_block=config.preparation.markers_per_information_block,
                information_tempering_power=config.panel.information_tempering_power)
