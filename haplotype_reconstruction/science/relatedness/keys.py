"""Resolve family search ties by evidence content.

Method: docs/methods/relatedness.md.
"""

from __future__ import annotations

from functools import cmp_to_key

import numpy as np

from .parameters import EVIDENCE_COMPARE_CHUNK


def _evidence_ranks(C):
    """Exact lexicographic order of the sample's own retained child-term vector, computed once per fit.

    Terms depend only on that sample's normalised GLs and observed mask, never the cohort allele frequency.
    Compare numeric values in site/parent-genotype order, in bounded chunks to stop at the first difference.
    Unlike a digest order, this does not scramble unrelated samples when an upstream fit changes a last bit.
    Equal vectors share a rank: identities within that evidence class are interchangeable, so outputs are
    defined up to their permutation. No name or axis index is appended, and no likelihood tolerance or
    rounding enters the evidence comparison. Canonical member order also fixes restoration's reduction order.
    """
    vectors = C.reshape(len(C), -1)

    def compare(a, b):
        for lo in range(0, vectors.shape[1], EVIDENCE_COMPARE_CHUNK):
            left, right = vectors[a, lo : lo + EVIDENCE_COMPARE_CHUNK], vectors[b, lo : lo + EVIDENCE_COMPARE_CHUNK]
            different = np.flatnonzero(left != right)
            if len(different):
                j = different[0]
                return 1 if left[j] > right[j] else -1
        return 0

    order = sorted(range(len(C)), key=cmp_to_key(compare))
    ranks = np.empty(len(C), dtype=np.int64)
    rank = 0
    for j, c in enumerate(order):
        if j and compare(order[j - 1], c):
            rank += 1
        ranks[c] = rank
    return ranks


def _top(gains, keys, tolerance: float) -> int:
    """Index of the largest gain (NaN: not a candidate). Gains within the tolerance of it tie, and the tie goes to the
    largest content key, so the choice follows the data, not sample names or input order."""
    tied = np.flatnonzero(gains >= np.nanmax(gains) - tolerance)
    return int(max(tied, key=lambda j: keys[j]))


def _top_keyed(gains, key, tolerance: float) -> int:
    """_top with the content keys evaluated for the tied gains only (key(j): candidate j's key)."""
    tied = np.flatnonzero(gains >= np.nanmax(gains) - tolerance)
    return int(tied[0]) if len(tied) == 1 else int(max(tied, key=key))


def _ranked(offers, key, tolerance: float):
    """``offers`` (gain first) in decreasing gain. Gains within the tolerance of a run's first gain tie, and are
    ordered by ``key`` (content), not by sample or family index."""
    offers = sorted(offers, key=lambda offer: -offer[0])
    out, start = [], 0
    while start < len(offers):
        stop = start + 1
        while stop < len(offers) and offers[stop][0] >= offers[start][0] - tolerance:
            stop += 1
        out.extend(sorted(offers[start:stop], key=key, reverse=True))
        start = stop
    return out


__all__ = []
