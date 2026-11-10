"""The variable-K occupancy gate: which candidate cavity fills are released.

A founder row's fills are released only when the row is demonstrably occupied by the block's
samples (pre-fill, held-out carrier profiles):
* abundant route: on the block's informative samples, the sum of squared carrier probabilities
  reaches ``minimum_abundant_sum_squared_probability`` and at least
  ``minimum_abundant_high_confidence_carriers`` samples carry the row with probability >=
  ``abundant_high_confidence_probability``; all of the row's fills are released;
* rare-continuity route: through an accepted one-to-one boundary correspondence (``links``) with a
  row of an adjacent block, at least ``minimum_rare_continuity_samples`` samples informative on
  both sides carry both rows (local >= ``minimum_rare_local_carrier_probability``, neighbour >=
  ``minimum_rare_neighbor_carrier_probability``, product >= ``minimum_same_sample_continuity``);
  the best such neighbour must be unique; a fill is released where one of those samples is
  focal-informative at its site.
Blocks whose local selection left unexplained copies (unknown-mass veto) and single-founder
blocks release nothing. The released fills are a subset of the candidates.

Carrier probabilities can saturate at exactly 1.0, putting the abundant sum of squares on its
integer threshold and making neighbours' continuity scores equal. Use the ``core.ties``
probability band, relative to the threshold for the sum: a sum inside the threshold band takes
the conservative side (not abundant). Neighbour scores in one class are tied; tied neighbours
abstain only when their qualifying sample sets differ, because only those samples affect release.


Owns the completion occupancy boundary (U07). Must not read files, clocks or
environment, allocate execution resources or publish diagnostics.
Method: docs/methods/assembly.md#completion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from haplotype_reconstruction.contracts.roles import record_field
from haplotype_reconstruction.core import ties

from .crossfit import CarrierProfiles
from .links import BoundaryLink
from .parameters import OccupancyParameters

# U07 uses only the policy quantum: the conservative abundant rule and tied
# neighbor abstention are scientific decisions, not reversible conventions.
_PROBABILITY_QUANTUM = ties.policy("assembly.completion.abundant", kind="probability").quantum(0.0)
_NEIGHBOR_QUANTUM = ties.policy("assembly.completion.neighbor", kind="probability").quantum(0.0)


@dataclass(frozen=True)
class GateResult:
    """released: bool[K_b, L_b] released fills per block; reason_counts: (reason, founder rows) sorted by
    reason; rare_without_focal_observation: candidate fills of rare-continuity rows withheld because
    no qualifying sample is focal-informative at the site."""

    released: tuple[np.ndarray, ...] = record_field("state", dtype="tuple[bool[K,L],...]", axes=())
    reason_counts: tuple[tuple[str, int], ...] = record_field("report", dtype="tuple[tuple[str,int64],...]", axes=())
    rare_without_focal_observation: int = record_field("report", dtype="int64", axes=())
    tie_events: ties.TieCounts | None = record_field("diagnostic", dtype="TieCounts | None", axes=(), default=None)


@dataclass(frozen=True)
class _Neighbor:
    block: int = record_field("state", dtype="int64", axes=())
    row: int = record_field("state", dtype="int64", axes=())
    boundary: int = record_field("state", dtype="int64", axes=())


def _neighbor_table(links: tuple[BoundaryLink, ...], row_counts):
    """Each row's atomic neighbours (in boundary order, then link order)."""
    neighbors: list = [[[] for _ in range(row_count)] for row_count in row_counts]
    for boundary, link in enumerate(links):
        for left, right in link.pairs.tolist():
            neighbors[boundary][left].append(_Neighbor(boundary + 1, right, boundary))
            neighbors[boundary + 1][right].append(_Neighbor(boundary, left, boundary))
    return neighbors


def _decision(block, row, profiles, neighbors, links, rule, unknown_mass_veto, counter=None, key="", census=None):
    """(route "abundant" | "rare_continuity" | "abstain", reason, qualifying samples) of one row."""
    profile = profiles[block]
    local_probability = profile.carrier_probability[:, row]
    local_informative = profile.informative_samples
    informative_probability = local_probability[local_informative]
    if unknown_mass_veto:
        return "abstain", "discovery_wildcard_panel_underfit", ()
    if profile.carrier_probability.shape[1] == 1:
        return "abstain", "single_founder_panel_underfit_unverifiable", ()
    sum_squared = float(np.sum(informative_probability**2))
    high_confidence_count = int(np.sum(informative_probability >= rule.abundant_high_confidence_probability))
    # Saturated probabilities put the sum exactly on its (integer) threshold: inside the band the conservative side
    # (not abundant) is taken.
    threshold = rule.minimum_abundant_sum_squared_probability
    band = ties.threshold_band(sum_squared, threshold, _PROBABILITY_QUANTUM * max(1.0, abs(threshold)))
    if counter is not None:
        counter.note("completion.abundant.band", inside_tie=band == 0)
        ties.census("completion.abundant.band", f"{key}/r{row}", [sum_squared - threshold], sink=census)
    if band > 0 and high_confidence_count >= rule.minimum_abundant_high_confidence_carriers:
        return "abundant", "abundant_occupancy", ()

    atomic_neighbors = neighbors[block][row]
    qualifying_neighbor_probability = False
    candidates = []
    for neighbor in atomic_neighbors:
        neighbor_profile = profiles[neighbor.block]
        neighbor_probability = neighbor_profile.carrier_probability[:, neighbor.row]
        eligible = local_informative & neighbor_profile.informative_samples
        eligible &= links[neighbor.boundary].informative
        eligible &= local_probability >= rule.minimum_rare_local_carrier_probability
        eligible &= neighbor_probability >= rule.minimum_rare_neighbor_carrier_probability
        qualifying_neighbor_probability |= bool(np.any(eligible))
        if not np.any(eligible):
            continue
        sample_indices = np.flatnonzero(eligible)
        products = local_probability[sample_indices] * neighbor_probability[sample_indices]
        boundary_samples = tuple(
            int(value) for value in sample_indices[products >= rule.minimum_same_sample_continuity]
        )
        candidates.append(
            (len(boundary_samples), float(np.max(products)), (neighbor.block, neighbor.row), boundary_samples)
        )

    # The best neighbour: most qualifying samples, then the highest continuity score; scores in one tie class
    # (probability quantum) are tied. The release depends only on the qualifying samples, so tied neighbours are
    # Abstain only when tied neighbours have different qualifying samples. Carrier probabilities
    # can saturate at 1.0, so an equal score alone does not establish ambiguity.
    qualifying_samples: tuple[int, ...] = ()
    ambiguous_best_neighbor = False
    if candidates:
        most = max(candidate[0] for candidate in candidates)
        top = [candidate for candidate in candidates if candidate[0] == most]
        classes = ties.tie_classes([candidate[1] for candidate in top], _NEIGHBOR_QUANTUM, descending=True)
        winners = [top[index] for index in classes.members(0)]
        if counter is not None:
            counter.add("completion.neighbor", classes)
        if len(top) > 1:
            ties.census("completion.neighbor", f"{key}/r{row}", [candidate[1] for candidate in top], sink=census)
        if any(other[3] != winners[0][3] for other in winners[1:]):
            ambiguous_best_neighbor = True
        qualifying_samples = winners[0][3]

    if ambiguous_best_neighbor:
        return "abstain", "ambiguous_tied_atomic_neighbors", ()
    if len(qualifying_samples) >= rule.minimum_rare_continuity_samples:
        return "rare_continuity", "rare_atomic_same_sample_continuity", qualifying_samples
    maximum_local = float(np.max(informative_probability)) if informative_probability.size else 0.0
    if informative_probability.size == 0:
        reason = "no_informative_samples"
    elif not atomic_neighbors:
        reason = "no_atomic_identifiable_neighbor"
    elif maximum_local < rule.minimum_rare_local_carrier_probability:
        reason = "insufficient_local_carrier_probability"
    elif not qualifying_neighbor_probability:
        reason = "insufficient_neighbor_carrier_probability"
    elif qualifying_samples:
        reason = "insufficient_rare_continuity_samples"
    else:
        reason = "insufficient_same_sample_continuity"
    return "abstain", reason, ()


def occupancy_gate(
    candidates,
    profiles: tuple[CarrierProfiles, ...],
    focal,
    links: tuple[BoundaryLink, ...],
    rule: OccupancyParameters,
    unknown_mass_veto,
    keys=None,
    *,
    census=None,
) -> GateResult:
    """Release the candidate fills of demonstrably occupied founder rows.

    candidates: bool[K_b, L_b] per block (cavity fills); profiles: per block; focal: bool[N, L_b] per
    block (observed discovery sites with non-uniform evidence); links: per boundary (B - 1);
    rule: CompletionParameters.occupancy; unknown_mass_veto: bool per block; keys: optional stable decision keys
    per block for the valley census (default: the block index).
    """
    row_counts = tuple(profile.carrier_probability.shape[1] for profile in profiles)
    neighbors = _neighbor_table(links, row_counts)
    released_fills = []
    reason_counts: dict[str, int] = {}
    rare_without_focal = 0
    counter = ties.TieCounter()
    for block, (candidate, focal_mask) in enumerate(zip(candidates, focal, strict=False)):
        released = np.zeros(candidate.shape, dtype=np.bool_)
        key = f"b{block}" if keys is None else str(keys[block])
        for row in range(row_counts[block]):
            route, reason, samples = _decision(
                block, row, profiles, neighbors, links, rule, bool(unknown_mass_veto[block]), counter, key, census
            )
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if route == "abundant":
                released[row] = candidate[row]
            elif route == "rare_continuity":
                focal_observed = np.any(focal_mask[np.asarray(samples, dtype=np.int64)], axis=0)
                released[row] = candidate[row] & focal_observed
                rare_without_focal += int(np.sum(candidate[row] & ~focal_observed))
        released_fills.append(released)
    return GateResult(
        released=tuple(released_fills),
        reason_counts=tuple(sorted(reason_counts.items())),
        rare_without_focal_observation=rare_without_focal,
        tie_events=counter.record(),
    )


__all__ = ["GateResult", "occupancy_gate"]
