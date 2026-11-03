"""Measure member and parent support on map units.

Owns the relocated support units. No storage, environment or runtime policy.
Method: docs/methods/relatedness.md; WP_RL1. Legacy arithmetic retained for relocation.
"""

from __future__ import annotations

import numpy as np

from .family_kernels import family_logliks, latent_alternatives, unit_terms
from .partition import _Partition


def support_units(morgans: np.ndarray, unit_morgans: float) -> np.ndarray:
    """int64 [k + 1]: boundaries (site offsets) of a chromosome's support units: max(1, floor(length / unit))
    consecutive pieces of equal measured map length over its family sites (``morgans``: their measured map
    coordinates, nondecreasing; one piece without a measured length)."""
    n_sites = len(morgans)
    if n_sites == 0:
        return np.asarray([0, 0], dtype=np.int64)
    length = float(morgans[-1] - morgans[0])
    pieces = max(1, int(np.floor(length / unit_morgans)))
    cuts = morgans[0] + length * np.arange(1, pieces) / pieces
    return np.concatenate(([0], np.searchsorted(morgans, cuts, side="left"), [n_sites])).astype(np.int64)


def _support(part: _Partition, ranges):
    """Per chromosome (with family sites): whether each member's family is its best placement, and whether each
    sequenced parent beats a latent one. A margin within the tolerance is a tie, not a win: a latent
    one-member family is exactly its member alone, and a parent without reads on a unit exactly a latent one, so
    float noise must not decide those (it follows the sample order)."""
    ranges = [(lo, hi) for lo, hi in ranges if hi > lo]
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
    for lo, hi in ranges:
        llk = np.empty(F)
        family_logliks(part.S, part.slots, live, part.logpi, part.logpost, lo, hi, llk)
        unit_terms(
            part.S, part.C, part.slots, live, llk, family_of, part.logpi, part.logpost, lo, hi, joins, rest, alone
        )
        rest[single] = 0.0
        stay = llk[family_of] - rest
        joins[excluded_rows, excluded_cols] = -np.inf
        elsewhere = np.maximum(joins.max(axis=0) if F else np.full(N, -np.inf), alone)
        hits += stay > elsewhere + part.parameters.gain_tolerance
        latent_alternatives(part.S, part.slots, live, part.logpi, part.logpost, lo, hi, latent)
        parent_hits[live] += named & (llk[:, None] > latent + part.parameters.gain_tolerance)
    return hits / max(len(ranges), 1), parent_hits / max(len(ranges), 1)


__all__ = ["support_units"]
