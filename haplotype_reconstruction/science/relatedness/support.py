"""Measure member and parent support on pooled Morgan-weighted material units (DESIGN_417 D5).

The units are consecutive 1-Morgan pieces of the family sites' working-map axis pooled over the contigs in
configured order (run.concatenate_terms); each unit's hit counts its working Morgans.

Method: docs/methods/relatedness.md.
"""

from __future__ import annotations

import numpy as np

from ...core.concurrency import thread_count
from .class_kernels import FAMILY, FAMILY_SAMPLE, SAMPLE
from .family_kernels import family_logliks, latent_alternatives, unit_terms
from .partition import _Partition


def _support(part: _Partition, ranges, weights=None):
    """Per support unit (site range ``ranges[u]``, weight ``weights[u]`` its working Morgans): whether each member's
    family is its best placement, and whether each sequenced parent beats a latent one; the shares are the
    Morgan-weighted means over the units. A margin within the tolerance is a tie, not a win: a latent
    one-member family is exactly its member alone, and a parent without reads on a unit exactly a latent one, so
    float noise must not decide those (it follows the sample order)."""
    weights = [1.0] * len(ranges) if weights is None else weights
    kept = [(lo, hi, float(w)) for (lo, hi), w in zip(ranges, weights, strict=True) if hi > lo]
    total = sum(w for _, _, w in kept)
    live = part.live()
    N, F = part.N, len(live)
    hits = np.zeros(N)
    parent_hits = np.zeros((len(part.members), 2))
    position = np.full(len(part.members), -1, dtype=np.int64)
    position[live] = np.arange(F)
    family_of = position[part.of]
    single = np.asarray([len(part.members[f]) == 1 for f in live], dtype=np.bool_)[family_of]
    # no placement in the sample's own family, nor in a family it is a parent of
    row_parts, col_parts = [family_of], [part.everyone]
    for j, f in enumerate(live):
        for p in part.slots[f]:
            if p >= 0:
                row_parts.append(np.asarray([j], dtype=np.int64))
                col_parts.append(np.asarray([p], dtype=np.int64))
    excluded_rows, excluded_cols = np.concatenate(row_parts), np.concatenate(col_parts)
    named = part.slots[live] >= 0
    joins, rest, alone, latent = np.empty((F, N)), np.empty(N), np.empty(N), np.empty((F, 2))
    for lo, hi, weight in kept:
        thread_count(part.threads)
        llk = np.empty(F)
        classed = getattr(part, "class_prior", None) is not None  # rl21: two source populations
        if classed:
            _class_unit(part, live, family_of, named, lo, hi, llk, joins, rest, alone, latent)
        else:
            family_logliks(part.S, part.slots, live, part.logpi, part.logpost, lo, hi, llk)
            unit_terms(
                part.S, part.C, part.slots, live, llk, family_of, part.logpi, part.logpost, lo, hi, joins, rest, alone
            )
        rest[single] = 0.0
        stay = llk[family_of] - rest
        joins[excluded_rows, excluded_cols] = -np.inf
        elsewhere = np.maximum(joins.max(axis=0) if F else np.full(N, -np.inf), alone)
        hits += weight * (stay > elsewhere + part.parameters.gain_tolerance)
        if not classed:
            latent_alternatives(part.S, part.slots, live, part.logpi, part.logpost, lo, hi, latent)
        parent_hits[live] += weight * (named & (llk[:, None] > latent + part.parameters.gain_tolerance))
    total = total if total > 0 else 1.0
    return hits / total, parent_hits / total


def _class_unit(part: _Partition, live, family_of, named, lo, hi, llk, joins, rest, alone, latent):
    """One support unit with two source populations (rl21): family_logliks, unit_terms and latent_alternatives as
    class marginals over the unit's own sites (the unit is scored as if it were the genome)."""
    F, N = len(live), part.N
    llk[:] = part.class_values(FAMILY, live, slots=part.slots[live], lo=lo, hi=hi)
    families = np.repeat(live, N)
    joins[:] = part.class_values(
        FAMILY_SAMPLE, families, np.tile(part.everyone, F), 1.0, part.slots[families], lo, hi
    ).reshape(F, N) - llk[:, None]
    own = live[family_of]
    rest[:] = part.class_values(FAMILY_SAMPLE, own, part.everyone, -1.0, part.slots[own], lo, hi)
    alone[:] = part.class_values(SAMPLE, part.everyone, lo=lo, hi=hi)
    latent[:] = np.nan
    for k in (0, 1):
        rows = np.flatnonzero(named[:, k])
        if len(rows):
            slots = part.slots[live[rows]].copy()
            slots[:, k] = -1
            latent[rows, k] = part.class_values(FAMILY, live[rows], slots=slots, lo=lo, hi=hi)


__all__: list[str] = []
