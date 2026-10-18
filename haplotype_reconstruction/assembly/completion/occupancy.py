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
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crossfit import CarrierProfiles
from .links import BoundaryLink


@dataclass(frozen=True)
class GateResult:
    """released: bool[K_b, L_b] released fills per block; reason_counts: (reason, founder rows) sorted by
    reason; rare_without_focal_observation: candidate fills of rare-continuity rows withheld because
    no qualifying sample is focal-informative at the site."""

    released: tuple[np.ndarray, ...]
    reason_counts: tuple[tuple[str, int], ...]
    rare_without_focal_observation: int


@dataclass(frozen=True)
class _Neighbor:
    block: int
    row: int
    boundary: int


def _neighbor_table(links: tuple[BoundaryLink, ...], row_counts):
    """Each row's atomic neighbours (in boundary order, then link order)."""
    neighbors = [[[] for _ in range(row_count)] for row_count in row_counts]
    for boundary, link in enumerate(links):
        for left, right in link.pairs.tolist():
            neighbors[boundary][left].append(_Neighbor(boundary + 1, right, boundary))
            neighbors[boundary + 1][right].append(_Neighbor(boundary, left, boundary))
    return neighbors


def _decision(block, row, profiles, neighbors, links, rule, unknown_mass_veto):
    """(route "abundant" | "rare_continuity" | "abstain", reason, qualifying samples) of one row."""
    profile = profiles[block]
    local_probability = profile.carrier_probability[:, row]
    local_informative = profile.informative_samples
    informative_probability = local_probability[local_informative]
    if unknown_mass_veto:
        return "abstain", "discovery_wildcard_panel_underfit", ()
    if profile.carrier_probability.shape[1] == 1:
        return "abstain", "single_founder_panel_underfit_unverifiable", ()
    sum_squared = float(np.sum(informative_probability ** 2))
    high_confidence_count = int(np.sum(
        informative_probability >= rule.abundant_high_confidence_probability))
    if (sum_squared >= rule.minimum_abundant_sum_squared_probability
            and high_confidence_count >= rule.minimum_abundant_high_confidence_carriers):
        return "abundant", "abundant_occupancy", ()

    atomic_neighbors = neighbors[block][row]
    best_score = 0.0
    best_neighbor = None
    qualifying_neighbor_probability = False
    qualifying_samples: tuple[int, ...] = ()
    ambiguous_best_neighbor = False
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
        boundary_samples = tuple(int(value) for value in sample_indices[
            products >= rule.minimum_same_sample_continuity])
        candidate_score = float(products[int(np.argmax(products))])
        candidate_rank = (len(boundary_samples), candidate_score)
        best_rank = (len(qualifying_samples), best_score)
        if candidate_rank > best_rank:
            best_score = candidate_score
            best_neighbor = (neighbor.block, neighbor.row)
            qualifying_samples = boundary_samples
            ambiguous_best_neighbor = False
        elif candidate_rank == best_rank and best_neighbor is not None:
            if ((neighbor.block, neighbor.row) != best_neighbor
                    or boundary_samples != qualifying_samples):
                ambiguous_best_neighbor = True

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


def occupancy_gate(candidates, profiles: tuple[CarrierProfiles, ...], focal, links: tuple[BoundaryLink, ...],
                   rule, unknown_mass_veto) -> GateResult:
    """Release the candidate fills of demonstrably occupied founder rows.

    candidates: bool[K_b, L_b] per block (cavity fills); profiles: per block; focal: bool[N, L_b] per
    block (observed discovery sites with non-uniform evidence); links: per boundary (B - 1);
    rule: CompletionConfig.occupancy; unknown_mass_veto: bool per block.
    """
    row_counts = tuple(profile.carrier_probability.shape[1] for profile in profiles)
    neighbors = _neighbor_table(links, row_counts)
    released_fills = []
    reason_counts: dict[str, int] = {}
    rare_without_focal = 0
    for block, (candidate, focal_mask) in enumerate(zip(candidates, focal)):
        released = np.zeros(candidate.shape, dtype=np.bool_)
        for row in range(row_counts[block]):
            route, reason, samples = _decision(block, row, profiles, neighbors, links, rule,
                                               bool(unknown_mass_veto[block]))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if route == "abundant":
                released[row] = candidate[row]
            elif route == "rare_continuity":
                focal_observed = np.any(focal_mask[np.asarray(samples, dtype=np.int64)], axis=0)
                released[row] = candidate[row] & focal_observed
                rare_without_focal += int(np.sum(candidate[row] & ~focal_observed))
        released_fills.append(released)
    return GateResult(released=tuple(released_fills), reason_counts=tuple(sorted(reason_counts.items())),
                      rare_without_focal_observation=rare_without_focal)
