"""One-to-one founder-row correspondences at the boundary of two adjacent blocks.

Rows of a block are grouped into exact classes (identical pre-fill allele vectors). A left class
and a right class match when, on the samples informative on both sides, their held-out carrier
profiles agree: the mean squared discrepancy of class expected copy number and of its bounded
presence proxy min(copies, 1) is at most ``maximum_mean_cost``, the pair is a reciprocal unique
minimum (by more than the configured margins) in the pooled samples and in each fold separately,
both classes carry enough effective copies (pooled and per fold), there are enough informative
samples (pooled and per fold), each class is separated from every other class of its block by
at least ``minimum_original_hard_anchor_sites`` jointly called discordant sites, and both blocks
have at least two classes. The occupancy gate uses the accepted matches of singleton classes
(one row each) as atomic neighbours.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .crossfit import CarrierProfiles
from .observation import row_classes


@dataclass(frozen=True)
class BoundaryLink:
    """The boundary between blocks b and b + 1 as the occupancy gate reads it.

    informative: bool[N] samples informative in both blocks with at least
        ``minimum_informative_sites_per_sample`` usable sites on each side.
    pairs: int64[P, 2] (left row, right row) of every accepted match of two singleton classes, in
        class order (left class, then right class).
    """

    informative: np.ndarray
    pairs: np.ndarray


def _class_profiles(profiles: CarrierProfiles, classes):
    copies = np.asarray(profiles.expected_copies, dtype=np.float64)
    class_copies = np.column_stack([np.sum(copies[:, rows], axis=1) for rows in classes])
    # Marginal row-carrier probabilities are not jointly sufficient for an exact class-carrier
    # probability and their sum changes with duplicate multiplicity. This bounded presence proxy
    # is derived from the invariant class copy expectation instead; it is not a probability.
    return class_copies, np.minimum(class_copies, 1.0)


def _class_anchor_counts(alleles, classes) -> np.ndarray:
    """Minimum hard discordance of each class to any competing class of its block."""
    result = np.zeros(len(classes), dtype=np.int64)
    if len(classes) <= 1:
        return result
    for index, rows in enumerate(classes):
        representative = rows[0]
        minimum = alleles.shape[1]
        for other_index, other in enumerate(classes):
            if other_index == index:
                continue
            competitor = other[0]
            jointly_called = (alleles[representative] >= 0) & (alleles[competitor] >= 0)
            separated = jointly_called & (alleles[representative] != alleles[competitor])
            minimum = min(minimum, int(np.sum(separated)))
        result[index] = minimum
    return result


def _cost_matrix(left_copies, left_carriers, right_copies, right_carriers, samples):
    if not np.any(samples):
        return np.full((left_copies.shape[1], right_copies.shape[1]), math.inf, dtype=np.float64)
    copy_difference = left_copies[samples, :, None] - right_copies[samples, None, :]
    carrier_difference = left_carriers[samples, :, None] - right_carriers[samples, None, :]
    return np.mean(0.5 * (copy_difference ** 2 + carrier_difference ** 2), axis=0)


def _effective_copies(values, samples):
    if not np.any(samples):
        return np.zeros(values.shape[1], dtype=np.float64)
    return np.sum(values[samples], axis=0)


def _reciprocal_ranks(cost, tolerance):
    """Reciprocal unique minima (bool) and the row/column runner-up margins of a cost matrix."""
    mutual = np.zeros(cost.shape, dtype=np.bool_)
    row_margin = np.full(cost.shape[0], -math.inf, dtype=np.float64)
    column_margin = np.full(cost.shape[1], -math.inf, dtype=np.float64)
    row_best = np.full(cost.shape[0], -1, dtype=np.int64)
    column_best = np.full(cost.shape[1], -1, dtype=np.int64)
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        return mutual, row_margin, column_margin
    for left_index in range(cost.shape[0]):
        values = cost[left_index]
        order = np.argsort(values, kind="stable")
        if not np.isfinite(values[order[0]]):
            continue
        row_best[left_index] = int(order[0])
        row_margin[left_index] = (math.inf if order.size == 1
                                  else float(values[order[1]] - values[order[0]]))
        if row_margin[left_index] <= tolerance:
            row_best[left_index] = -1
    for right_index in range(cost.shape[1]):
        values = cost[:, right_index]
        order = np.argsort(values, kind="stable")
        if not np.isfinite(values[order[0]]):
            continue
        column_best[right_index] = int(order[0])
        column_margin[right_index] = (math.inf if order.size == 1
                                      else float(values[order[1]] - values[order[0]]))
        if column_margin[right_index] <= tolerance:
            column_best[right_index] = -1
    for left_index in range(cost.shape[0]):
        right_index = row_best[left_index]
        if right_index >= 0 and column_best[right_index] == left_index:
            mutual[left_index, right_index] = True
    return mutual, row_margin, column_margin


def link_boundary(left_alleles, right_alleles, left: CarrierProfiles, right: CarrierProfiles,
                  folds, config) -> BoundaryLink:
    """Accepted singleton-class correspondences at one boundary.

    left_alleles / right_alleles: int8[K, L] pre-fill calls of the two blocks; left / right: their
    cross-fitted carrier profiles; folds: int8[N] fold labels; config: CompletionConfig.links.
    """
    left_classes = row_classes(left_alleles)
    right_classes = row_classes(right_alleles)
    left_copies, left_carriers = _class_profiles(left, left_classes)
    right_copies, right_carriers = _class_profiles(right, right_classes)
    left_anchors = _class_anchor_counts(left_alleles, left_classes)
    right_anchors = _class_anchor_counts(right_alleles, right_classes)

    informative = (left.informative_samples & right.informative_samples
                   & (left.informative_site_count >= config.minimum_informative_sites_per_sample)
                   & (right.informative_site_count >= config.minimum_informative_sites_per_sample))
    pairs = np.empty((0, 2), dtype=np.int64)
    if len(left_classes) == 1 or len(right_classes) == 1:
        # One class on a side is not identifiable from carrier profiles alone.
        return BoundaryLink(informative=informative, pairs=pairs)
    pooled_count = int(np.sum(informative))
    if pooled_count < config.minimum_pooled_informative_samples:
        return BoundaryLink(informative=informative, pairs=pairs)
    pooled_cost = _cost_matrix(left_copies, left_carriers, right_copies, right_carriers, informative)
    pooled_left = _effective_copies(left_copies, informative)
    pooled_right = _effective_copies(right_copies, informative)
    pooled_mutual, pooled_row_margin, pooled_column_margin = _reciprocal_ranks(
        pooled_cost, config.tolerance)
    folds_ranked = []
    for label in np.unique(folds).tolist():
        selected = informative & (folds == label)
        cost = _cost_matrix(left_copies, left_carriers, right_copies, right_carriers, selected)
        mutual, row_margin, column_margin = _reciprocal_ranks(cost, config.tolerance)
        folds_ranked.append((int(np.sum(selected)), _effective_copies(left_copies, selected),
                             _effective_copies(right_copies, selected), cost, mutual, row_margin,
                             column_margin))

    accepted = []
    for left_index, right_index in zip(*np.nonzero(pooled_mutual)):
        if not (pooled_left[left_index] >= config.minimum_effective_copies
                and pooled_right[right_index] >= config.minimum_effective_copies
                and pooled_cost[left_index, right_index] <= config.maximum_mean_cost
                and pooled_row_margin[left_index] > config.minimum_pooled_margin
                and pooled_column_margin[right_index] > config.minimum_pooled_margin
                and left_anchors[left_index] >= config.minimum_original_hard_anchor_sites
                and right_anchors[right_index] >= config.minimum_original_hard_anchor_sites):
            continue
        if all(count >= config.minimum_fold_informative_samples
               and fold_left[left_index] >= config.minimum_fold_effective_copies
               and fold_right[right_index] >= config.minimum_fold_effective_copies
               and cost[left_index, right_index] <= config.maximum_mean_cost
               and mutual[left_index, right_index]
               and row_margin[left_index] > config.minimum_fold_margin
               and column_margin[right_index] > config.minimum_fold_margin
               for count, fold_left, fold_right, cost, mutual, row_margin, column_margin
               in folds_ranked):
            if len(left_classes[left_index]) == 1 and len(right_classes[right_index]) == 1:
                accepted.append((left_classes[left_index][0], right_classes[right_index][0]))
    if accepted:
        pairs = np.asarray(accepted, dtype=np.int64)
    return BoundaryLink(informative=informative, pairs=pairs)
