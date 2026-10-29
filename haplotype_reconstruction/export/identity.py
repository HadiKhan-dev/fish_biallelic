"""Founder identity diagnostics of founder row paths (release 2.17): local duplicate-partner changes.

A founder row of an assembled component is a path through the local rows of its prepared blocks (one local row
per block; ``FounderComponent.rows``). On a block, row k's local partners are the other founder rows on the same
prepared local row. At the junction between blocks b - 1 and b, when row k has partners on both sides and the two
partner sets are disjoint, no local content of its own carries the row across the junction: it copies one set of
founders on the left and another on the right (reason ``local_duplicate_partner_change``).

The flag is structural and diagnostic only. It does not show that the row's long-range correspondence is wrong or
unidentifiable (informative flanks or carriers can still link a true founder through stretches it shares with other
founders), that a competing panel scores as well, or that a crossover happened there; its absence is no identity
certificate. A flag names the focal row only: an exchange at that junction could involve its partners too, whose
identities are not thereby certified. It changes no routing, call, founder column or release break.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REASON = "local_duplicate_partner_change"
METHOD = "duplicate-partner-v1"


@dataclass(frozen=True)
class PartnerChange:
    """Row ``row`` at the junction before block ``block`` (component-local, 1 .. B-1): its local rows on blocks
    block - 1 and block, and its partner founder rows there (disjoint, both non-empty)."""

    row: int
    block: int
    left_local_row: int
    right_local_row: int
    left_partners: tuple[int, ...]
    right_partners: tuple[int, ...]


def partner_changes(rows) -> list[PartnerChange]:
    """The duplicate-partner changes of a component's row paths ``rows`` int [K, B] (local row per founder and
    block), in (block, row) order. One pass over the blocks; O(B K^2) time, O(K^2) memory."""
    rows = np.asarray(rows)
    if rows.ndim != 2:
        raise ValueError("row paths must be a [founders, blocks] matrix")
    founders, blocks = rows.shape
    result = []
    previous = None
    for block in range(blocks):
        column = rows[:, block]
        partners = [tuple(int(j) for j in np.flatnonzero(column == column[k]) if j != k) for k in range(founders)]
        if previous is not None:
            for k in range(founders):
                left, right = previous[k], partners[k]
                if left and right and not set(left) & set(right):
                    result.append(PartnerChange(k, block, int(rows[k, block - 1]), int(column[k]), left, right))
        previous = partners
    return result
