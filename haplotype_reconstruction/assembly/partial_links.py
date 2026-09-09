"""assembly / partial links for the canonical reconstruction pipeline."""
from __future__ import annotations


from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class PartialLinkConfig:
    """Immutable conservative acceptance thresholds.

    Costs are mean squared discrepancies on class-level expected copy number
    and its bounded presence proxy.  A match must be a reciprocal unique
    optimum in pooled samples and in every fold.
    ``minimum_original_hard_anchor_sites`` is the minimum number of jointly
    called, discordant sites separating a class from every other class in its
    block.  ``maximum_mean_cost=0.099`` is the frozen simple global release
    threshold: it cleared every primary gate on chr8 no-missing, MCAR, and tract
    fixtures and on the independent chr13 contig, while retaining more chr13
    coverage and valid full stitches than 0.075.
    """

    minimum_pooled_informative_samples: int = 6
    minimum_fold_informative_samples: int = 3
    minimum_informative_sites_per_sample: int = 1
    minimum_effective_copies: float = 1.0
    minimum_fold_effective_copies: float = 0.5
    maximum_mean_cost: float = 0.099
    minimum_pooled_margin: float = 0.03
    minimum_fold_margin: float = 0.01
    minimum_original_hard_anchor_sites: int = 1
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        integer_fields = (
            "minimum_pooled_informative_samples",
            "minimum_fold_informative_samples",
            "minimum_informative_sites_per_sample",
            "minimum_original_hard_anchor_sites",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.minimum_pooled_informative_samples < 1:
            raise ValueError("minimum_pooled_informative_samples must be positive")
        if self.minimum_fold_informative_samples < 1:
            raise ValueError("minimum_fold_informative_samples must be positive")
        nonnegative = (
            "minimum_effective_copies",
            "minimum_fold_effective_copies",
            "maximum_mean_cost",
            "minimum_pooled_margin",
            "minimum_fold_margin",
            "tolerance",
        )
        for name in nonnegative:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ExchangeableRowClass:
    index: int
    rows: tuple[int, ...]


@dataclass(frozen=True)
class ClassPairEvidence:
    """Pooled and fold-specific evidence for one possible class edge."""

    left_class: int
    right_class: int
    pooled_cost: float
    pooled_row_margin: float
    pooled_column_margin: float
    pooled_informative_samples: int
    left_effective_copies: float
    right_effective_copies: float
    fold_labels: tuple[int, ...]
    fold_costs: tuple[float, ...]
    fold_row_margins: tuple[float, ...]
    fold_column_margins: tuple[float, ...]
    fold_informative_samples: tuple[int, ...]
    fold_left_effective_copies: tuple[float, ...]
    fold_right_effective_copies: tuple[float, ...]
    left_original_hard_anchors: int
    right_original_hard_anchors: int
    external_identity: bool
    accepted: bool
    reason: str


@dataclass(frozen=True)
class ClassMatch:
    left_class: int
    right_class: int
    left_rows: tuple[int, ...]
    right_rows: tuple[int, ...]
    evidence: ClassPairEvidence


@dataclass(frozen=True)
class BoundaryEvent:
    """A supported local appearance/disappearance, not a biological claim."""

    side: str
    class_index: int
    rows: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class UnresolvedClass:
    side: str
    class_index: int
    rows: tuple[int, ...]
    reason: str
    compatible_classes: tuple[int, ...]


@dataclass(frozen=True)
class SplitMergeDiagnostic:
    """Local representation change or non-injective compatibility component."""

    kind: str
    left_classes: tuple[int, ...]
    right_classes: tuple[int, ...]
    left_rows: tuple[int, ...]
    right_rows: tuple[int, ...]


@dataclass(frozen=True)
class PartialBoundaryLink:
    """Partial class correspondence and an optional complete atomic stitch."""

    reason: str
    left_classes: tuple[ExchangeableRowClass, ...]
    right_classes: tuple[ExchangeableRowClass, ...]
    matched: tuple[ClassMatch, ...]
    unmatched_left: tuple[int, ...]
    unmatched_right: tuple[int, ...]
    births: tuple[BoundaryEvent, ...]
    deaths: tuple[BoundaryEvent, ...]
    unresolved: tuple[UnresolvedClass, ...]
    split_merge: tuple[SplitMergeDiagnostic, ...]
    pair_evidence: tuple[ClassPairEvidence, ...]
    informative_samples: np.ndarray
    fold_labels: tuple[int, ...]
    full_stitch: bool
    left_to_right: np.ndarray | None


def exact_exchangeable_row_classes(
    panel: assembly_joint_completion.HardFounderPanel,
) -> tuple[ExchangeableRowClass, ...]:
    """Partition rows by exact original ``{-1, 0, 1}`` hard-call vectors."""

    grouped: list[list[int]] = []
    for row in range(panel.n_founders):
        for members in grouped:
            if np.array_equal(panel.alleles[row], panel.alleles[members[0]]):
                members.append(row)
                break
        else:
            grouped.append([row])
    return tuple(
        ExchangeableRowClass(index, tuple(rows))
        for index, rows in enumerate(grouped)
    )


def _validate_profiles(profiles: assembly_joint_completion.CarrierProfiles, n_rows: int, name: str) -> int:
    copies = np.asarray(profiles.expected_copies, dtype=np.float64)
    carriers = np.asarray(profiles.carrier_probability, dtype=np.float64)
    informative = np.asarray(profiles.informative_samples, dtype=np.bool_)
    site_count = np.asarray(profiles.informative_site_count)
    if copies.ndim != 2 or copies.shape[1] != n_rows:
        raise ValueError(f"{name} expected_copies must have shape (samples, rows)")
    if carriers.shape != copies.shape:
        raise ValueError(f"{name} carrier_probability must match expected_copies")
    if informative.shape != (copies.shape[0],):
        raise ValueError(f"{name} informative_samples must match samples")
    if site_count.shape != (copies.shape[0],):
        raise ValueError(f"{name} informative_site_count must match samples")
    if np.any(~np.isfinite(copies)) or np.any(~np.isfinite(carriers)):
        raise ValueError(f"{name} profiles must be finite")
    if np.any(site_count < 0):
        raise ValueError(f"{name} informative_site_count must be non-negative")
    return copies.shape[0]


def _class_profiles(
    profiles: assembly_joint_completion.CarrierProfiles,
    classes: tuple[ExchangeableRowClass, ...],
) -> tuple[np.ndarray, np.ndarray]:
    copies = np.asarray(profiles.expected_copies, dtype=np.float64)
    class_copies = np.column_stack(
        [np.sum(copies[:, item.rows], axis=1) for item in classes]
    )
    # Marginal row-carrier probabilities are not jointly sufficient for an
    # exact class-carrier probability and their sum changes with duplicate
    # multiplicity.  This bounded presence proxy is derived from the invariant
    # class copy expectation instead; it is not interpreted as a probability.
    class_carriers = np.minimum(class_copies, 1.0)
    return class_copies, class_carriers


def _class_anchor_counts(
    panel: assembly_joint_completion.HardFounderPanel,
    classes: tuple[ExchangeableRowClass, ...],
) -> np.ndarray:
    """Minimum hard discordance to any competing exact row class."""

    result = np.zeros(len(classes), dtype=np.int64)
    if len(classes) <= 1:
        return result
    for item in classes:
        representative = item.rows[0]
        minimum = panel.n_sites
        for other in classes:
            if other.index == item.index:
                continue
            competitor = other.rows[0]
            jointly_called = (
                panel.alleles[representative] >= 0
            ) & (panel.alleles[competitor] >= 0)
            separated = jointly_called & (
                panel.alleles[representative] != panel.alleles[competitor]
            )
            minimum = min(minimum, int(np.sum(separated)))
        result[item.index] = minimum
    return result


def _cost_matrix(
    left_copies: np.ndarray,
    left_carriers: np.ndarray,
    right_copies: np.ndarray,
    right_carriers: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    if not np.any(samples):
        return np.full(
            (left_copies.shape[1], right_copies.shape[1]), math.inf,
            dtype=np.float64,
        )
    copy_difference = (
        left_copies[samples, :, None] - right_copies[samples, None, :]
    )
    carrier_difference = (
        left_carriers[samples, :, None] - right_carriers[samples, None, :]
    )
    return np.mean(
        0.5 * (copy_difference**2 + carrier_difference**2), axis=0
    )


def _effective_copies(values: np.ndarray, samples: np.ndarray) -> np.ndarray:
    if not np.any(samples):
        return np.zeros(values.shape[1], dtype=np.float64)
    return np.sum(values[samples], axis=0)


def _reciprocal_ranks(
    cost: np.ndarray,
    active_left: np.ndarray,
    active_right: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return reciprocal unique minima and row/column runner-up margins."""

    mutual = np.zeros(cost.shape, dtype=np.bool_)
    row_margin = np.full(cost.shape[0], -math.inf, dtype=np.float64)
    column_margin = np.full(cost.shape[1], -math.inf, dtype=np.float64)
    row_best = np.full(cost.shape[0], -1, dtype=np.int64)
    column_best = np.full(cost.shape[1], -1, dtype=np.int64)
    left_indices = np.flatnonzero(active_left)
    right_indices = np.flatnonzero(active_right)
    if left_indices.size == 0 or right_indices.size == 0:
        return mutual, row_margin, column_margin, row_best, column_best

    for left_index in left_indices:
        values = cost[left_index, right_indices]
        order = np.argsort(values, kind="stable")
        if not np.isfinite(values[order[0]]):
            continue
        best = int(right_indices[order[0]])
        row_best[left_index] = best
        row_margin[left_index] = (
            math.inf if order.size == 1
            else float(values[order[1]] - values[order[0]])
        )
        if row_margin[left_index] <= tolerance:
            row_best[left_index] = -1
    for right_index in right_indices:
        values = cost[left_indices, right_index]
        order = np.argsort(values, kind="stable")
        if not np.isfinite(values[order[0]]):
            continue
        best = int(left_indices[order[0]])
        column_best[right_index] = best
        column_margin[right_index] = (
            math.inf if order.size == 1
            else float(values[order[1]] - values[order[0]])
        )
        if column_margin[right_index] <= tolerance:
            column_best[right_index] = -1
    for left_index in left_indices:
        right_index = row_best[left_index]
        if right_index >= 0 and column_best[right_index] == left_index:
            mutual[left_index, right_index] = True
    return mutual, row_margin, column_margin, row_best, column_best


def _external_class_pairs(
    anchors: Sequence[tuple[int, int]],
    left_classes: tuple[ExchangeableRowClass, ...],
    right_classes: tuple[ExchangeableRowClass, ...],
    n_left: int,
    n_right: int,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    left_class_by_row = {
        row: item.index for item in left_classes for row in item.rows
    }
    right_class_by_row = {
        row: item.index for item in right_classes for row in item.rows
    }
    class_forward: dict[int, int] = {}
    class_reverse: dict[int, int] = {}
    atomic: dict[int, int] = {}
    used_right_rows: set[int] = set()
    for raw_left, raw_right in anchors:
        if isinstance(raw_left, bool) or isinstance(raw_right, bool):
            raise ValueError("external identity row indices must be integers")
        left_row, right_row = int(raw_left), int(raw_right)
        if left_row != raw_left or right_row != raw_right:
            raise ValueError("external identity row indices must be integers")
        if not 0 <= left_row < n_left or not 0 <= right_row < n_right:
            raise ValueError("external identity row index is outside its panel")
        if left_row in atomic and atomic[left_row] != right_row:
            raise ValueError("one left row has conflicting external identities")
        if right_row in used_right_rows and atomic.get(left_row) != right_row:
            raise ValueError("external row identities must be injective")
        left_class = left_class_by_row[left_row]
        right_class = right_class_by_row[right_row]
        if left_class in class_forward and class_forward[left_class] != right_class:
            raise ValueError("one left class has conflicting external identities")
        if right_class in class_reverse and class_reverse[right_class] != left_class:
            raise ValueError("one right class has conflicting external identities")
        class_forward[left_class] = right_class
        class_reverse[right_class] = left_class
        atomic[left_row] = right_row
        used_right_rows.add(right_row)
    return class_forward, class_reverse, atomic


def _pair_reason(
    left_index: int,
    right_index: int,
    *,
    external: bool,
    pooled_cost: np.ndarray,
    pooled_mutual: np.ndarray,
    pooled_row_margin: np.ndarray,
    pooled_column_margin: np.ndarray,
    pooled_count: int,
    pooled_left_support: np.ndarray,
    pooled_right_support: np.ndarray,
    fold_costs: tuple[np.ndarray, ...],
    fold_mutual: tuple[np.ndarray, ...],
    fold_row_margins: tuple[np.ndarray, ...],
    fold_column_margins: tuple[np.ndarray, ...],
    fold_counts: tuple[int, ...],
    fold_left_support: tuple[np.ndarray, ...],
    fold_right_support: tuple[np.ndarray, ...],
    left_anchors: np.ndarray,
    right_anchors: np.ndarray,
    config: PartialLinkConfig,
) -> str:
    if external:
        return "external_identity"
    if len(left_anchors) == 1 or len(right_anchors) == 1:
        return "single_class_requires_external_identity"
    if pooled_count < config.minimum_pooled_informative_samples:
        return "insufficient_pooled_informative_samples"
    if (
        pooled_left_support[left_index] < config.minimum_effective_copies
        or pooled_right_support[right_index] < config.minimum_effective_copies
    ):
        return "absent_or_unsupported_carriers"
    if pooled_cost[left_index, right_index] > config.maximum_mean_cost:
        return "pooled_cost_too_high"
    if not pooled_mutual[left_index, right_index]:
        return "pooled_not_reciprocal_unique"
    if (
        pooled_row_margin[left_index] <= config.minimum_pooled_margin
        or pooled_column_margin[right_index] <= config.minimum_pooled_margin
    ):
        return "pooled_margin_too_small"
    if (
        left_anchors[left_index] < config.minimum_original_hard_anchor_sites
        or right_anchors[right_index] < config.minimum_original_hard_anchor_sites
    ):
        return "insufficient_original_hard_identity_anchors"
    for fold_index, count in enumerate(fold_counts):
        if count < config.minimum_fold_informative_samples:
            return "insufficient_fold_informative_samples"
        if (
            fold_left_support[fold_index][left_index]
            < config.minimum_fold_effective_copies
            or fold_right_support[fold_index][right_index]
            < config.minimum_fold_effective_copies
        ):
            return "fold_absent_or_unsupported_carriers"
        if fold_costs[fold_index][left_index, right_index] > config.maximum_mean_cost:
            return "fold_cost_too_high"
        if not fold_mutual[fold_index][left_index, right_index]:
            return "fold_disagreement"
        if (
            fold_row_margins[fold_index][left_index] <= config.minimum_fold_margin
            or fold_column_margins[fold_index][right_index]
            <= config.minimum_fold_margin
        ):
            return "fold_margin_too_small"
    return "matched"


def link_partial_profiles(
    left_panel: assembly_joint_completion.HardFounderPanel,
    right_panel: assembly_joint_completion.HardFounderPanel,
    left: assembly_joint_completion.CarrierProfiles,
    right: assembly_joint_completion.CarrierProfiles,
    *,
    fold_assignments: np.ndarray,
    config: PartialLinkConfig = PartialLinkConfig(),
    external_identity_anchors: Sequence[tuple[int, int]] = (),
) -> PartialBoundaryLink:
    """Infer an injective *partial* class correspondence at one boundary.

    ``external_identity_anchors`` are independently established row-index
    correspondences, not simulated truth.  They constrain class matching and
    may resolve an otherwise unidentifiable one-class boundary.
    """

    n_samples = _validate_profiles(left, left_panel.n_founders, "left")
    if _validate_profiles(right, right_panel.n_founders, "right") != n_samples:
        raise ValueError("adjacent profiles must contain the same samples")
    folds = np.asarray(fold_assignments)
    if folds.shape != (n_samples,):
        raise ValueError("fold_assignments must have one label per sample")
    fold_labels = tuple(np.unique(folds).tolist())
    if len(fold_labels) < 2:
        raise ValueError("fold-specific agreement requires at least two folds")

    left_classes = exact_exchangeable_row_classes(left_panel)
    right_classes = exact_exchangeable_row_classes(right_panel)
    left_copies, left_carriers = _class_profiles(left, left_classes)
    right_copies, right_carriers = _class_profiles(right, right_classes)
    left_anchors = _class_anchor_counts(left_panel, left_classes)
    right_anchors = _class_anchor_counts(right_panel, right_classes)
    external_forward, external_reverse, atomic_external = _external_class_pairs(
        external_identity_anchors,
        left_classes,
        right_classes,
        left_panel.n_founders,
        right_panel.n_founders,
    )

    informative = (
        np.asarray(left.informative_samples, dtype=np.bool_)
        & np.asarray(right.informative_samples, dtype=np.bool_)
        & (
            np.asarray(left.informative_site_count)
            >= config.minimum_informative_sites_per_sample
        )
        & (
            np.asarray(right.informative_site_count)
            >= config.minimum_informative_sites_per_sample
        )
    )
    pooled_cost = _cost_matrix(
        left_copies, left_carriers, right_copies, right_carriers, informative
    )
    pooled_left_support = _effective_copies(left_copies, informative)
    pooled_right_support = _effective_copies(right_copies, informative)

    active_left = np.ones(len(left_classes), dtype=np.bool_)
    active_right = np.ones(len(right_classes), dtype=np.bool_)
    if external_forward:
        active_left[np.fromiter(external_forward, dtype=np.int64)] = False
        active_right[np.fromiter(external_reverse, dtype=np.int64)] = False
    pooled_rank = _reciprocal_ranks(
        pooled_cost, active_left, active_right, config.tolerance
    )

    fold_costs_list: list[np.ndarray] = []
    fold_mutual_list: list[np.ndarray] = []
    fold_row_margin_list: list[np.ndarray] = []
    fold_column_margin_list: list[np.ndarray] = []
    fold_counts: list[int] = []
    fold_left_support: list[np.ndarray] = []
    fold_right_support: list[np.ndarray] = []
    for label in fold_labels:
        selected = informative & (folds == label)
        cost = _cost_matrix(
            left_copies, left_carriers, right_copies, right_carriers, selected
        )
        mutual, row_margin, column_margin, _, _ = _reciprocal_ranks(
            cost, active_left, active_right, config.tolerance
        )
        fold_costs_list.append(cost)
        fold_mutual_list.append(mutual)
        fold_row_margin_list.append(row_margin)
        fold_column_margin_list.append(column_margin)
        fold_counts.append(int(np.sum(selected)))
        fold_left_support.append(_effective_copies(left_copies, selected))
        fold_right_support.append(_effective_copies(right_copies, selected))

    fold_costs = tuple(fold_costs_list)
    fold_mutual = tuple(fold_mutual_list)
    fold_row_margins = tuple(fold_row_margin_list)
    fold_column_margins = tuple(fold_column_margin_list)
    count = int(np.sum(informative))
    evidence: list[ClassPairEvidence] = []
    accepted_pairs: list[tuple[int, int]] = []
    for left_index in range(len(left_classes)):
        for right_index in range(len(right_classes)):
            external = external_forward.get(left_index) == right_index
            if (
                (left_index in external_forward and not external)
                or (right_index in external_reverse and not external)
            ):
                reason = "reserved_by_external_identity"
            else:
                reason = _pair_reason(
                    left_index,
                    right_index,
                    external=external,
                    pooled_cost=pooled_cost,
                    pooled_mutual=pooled_rank[0],
                    pooled_row_margin=pooled_rank[1],
                    pooled_column_margin=pooled_rank[2],
                    pooled_count=count,
                    pooled_left_support=pooled_left_support,
                    pooled_right_support=pooled_right_support,
                    fold_costs=fold_costs,
                    fold_mutual=fold_mutual,
                    fold_row_margins=fold_row_margins,
                    fold_column_margins=fold_column_margins,
                    fold_counts=tuple(fold_counts),
                    fold_left_support=tuple(fold_left_support),
                    fold_right_support=tuple(fold_right_support),
                    left_anchors=left_anchors,
                    right_anchors=right_anchors,
                    config=config,
                )
            accepted = reason in {"matched", "external_identity"}
            item = ClassPairEvidence(
                left_class=left_index,
                right_class=right_index,
                pooled_cost=float(pooled_cost[left_index, right_index]),
                pooled_row_margin=float(pooled_rank[1][left_index]),
                pooled_column_margin=float(pooled_rank[2][right_index]),
                pooled_informative_samples=count,
                left_effective_copies=float(pooled_left_support[left_index]),
                right_effective_copies=float(pooled_right_support[right_index]),
                fold_labels=fold_labels,
                fold_costs=tuple(float(values[left_index, right_index]) for values in fold_costs),
                fold_row_margins=tuple(float(values[left_index]) for values in fold_row_margins),
                fold_column_margins=tuple(float(values[right_index]) for values in fold_column_margins),
                fold_informative_samples=tuple(fold_counts),
                fold_left_effective_copies=tuple(float(values[left_index]) for values in fold_left_support),
                fold_right_effective_copies=tuple(float(values[right_index]) for values in fold_right_support),
                left_original_hard_anchors=int(left_anchors[left_index]),
                right_original_hard_anchors=int(right_anchors[right_index]),
                external_identity=external,
                accepted=accepted,
                reason=reason,
            )
            evidence.append(item)
            if accepted:
                accepted_pairs.append((left_index, right_index))

    evidence_by_pair = {
        (item.left_class, item.right_class): item for item in evidence
    }
    accepted_pairs.sort()
    matched = tuple(
        ClassMatch(
            left_index,
            right_index,
            left_classes[left_index].rows,
            right_classes[right_index].rows,
            evidence_by_pair[(left_index, right_index)],
        )
        for left_index, right_index in accepted_pairs
    )
    matched_left = {pair[0] for pair in accepted_pairs}
    matched_right = {pair[1] for pair in accepted_pairs}
    unmatched_left = tuple(
        index for index in range(len(left_classes)) if index not in matched_left
    )
    unmatched_right = tuple(
        index for index in range(len(right_classes)) if index not in matched_right
    )

    plausible = np.zeros(pooled_cost.shape, dtype=np.bool_)
    if count >= config.minimum_pooled_informative_samples:
        plausible = (
            (pooled_cost <= config.maximum_mean_cost)
            & (pooled_left_support[:, None] >= config.minimum_effective_copies)
            & (pooled_right_support[None, :] >= config.minimum_effective_copies)
        )
    deaths: list[BoundaryEvent] = []
    births: list[BoundaryEvent] = []
    unresolved: list[UnresolvedClass] = []
    for left_index in unmatched_left:
        compatible = tuple(np.flatnonzero(plausible[left_index]).tolist())
        if (
            count >= config.minimum_pooled_informative_samples
            and pooled_left_support[left_index] >= config.minimum_effective_copies
            and not compatible
        ):
            deaths.append(BoundaryEvent(
                "left", left_index, left_classes[left_index].rows,
                "no_compatible_right_class",
            ))
        else:
            best_reason = (
                "insufficient_pooled_informative_samples"
                if count < config.minimum_pooled_informative_samples
                else "absent_or_unsupported_carriers"
                if pooled_left_support[left_index] < config.minimum_effective_copies
                else evidence_by_pair[(
                    left_index,
                    min(compatible, key=lambda value: pooled_cost[left_index, value]),
                )].reason
            )
            unresolved.append(UnresolvedClass(
                "left", left_index, left_classes[left_index].rows,
                best_reason, compatible,
            ))
    for right_index in unmatched_right:
        compatible = tuple(np.flatnonzero(plausible[:, right_index]).tolist())
        if (
            count >= config.minimum_pooled_informative_samples
            and pooled_right_support[right_index] >= config.minimum_effective_copies
            and not compatible
        ):
            births.append(BoundaryEvent(
                "right", right_index, right_classes[right_index].rows,
                "no_compatible_left_class",
            ))
        else:
            best_reason = (
                "insufficient_pooled_informative_samples"
                if count < config.minimum_pooled_informative_samples
                else "absent_or_unsupported_carriers"
                if pooled_right_support[right_index] < config.minimum_effective_copies
                else evidence_by_pair[(
                    min(compatible, key=lambda value: pooled_cost[value, right_index]),
                    right_index,
                )].reason
            )
            unresolved.append(UnresolvedClass(
                "right", right_index, right_classes[right_index].rows,
                best_reason, compatible,
            ))

    split_merge: list[SplitMergeDiagnostic] = []
    for item in matched:
        if len(item.left_rows) != len(item.right_rows):
            kind = (
                "right_row_expansion"
                if len(item.right_rows) > len(item.left_rows)
                else "right_row_contraction"
            )
            split_merge.append(SplitMergeDiagnostic(
                kind,
                (item.left_class,),
                (item.right_class,),
                item.left_rows,
                item.right_rows,
            ))
    for right_index in range(len(right_classes)):
        possible_left = tuple(np.flatnonzero(plausible[:, right_index]).tolist())
        if len(possible_left) > 1:
            split_merge.append(SplitMergeDiagnostic(
                "multiple_left_classes_compatible",
                possible_left,
                (right_index,),
                tuple(row for index in possible_left for row in left_classes[index].rows),
                right_classes[right_index].rows,
            ))
    for left_index in range(len(left_classes)):
        possible_right = tuple(np.flatnonzero(plausible[left_index]).tolist())
        if len(possible_right) > 1:
            split_merge.append(SplitMergeDiagnostic(
                "multiple_right_classes_compatible",
                (left_index,),
                possible_right,
                left_classes[left_index].rows,
                tuple(row for index in possible_right for row in right_classes[index].rows),
            ))

    atomic_mapping = dict(atomic_external)
    for item in matched:
        if len(item.left_rows) == len(item.right_rows) == 1:
            left_row, right_row = item.left_rows[0], item.right_rows[0]
            if left_row in atomic_mapping and atomic_mapping[left_row] != right_row:
                raise RuntimeError("class and external atomic identities disagree")
            atomic_mapping[left_row] = right_row
    full = (
        not unmatched_left
        and not unmatched_right
        and left_panel.n_founders == right_panel.n_founders
        and len(atomic_mapping) == left_panel.n_founders
        and len(set(atomic_mapping.values())) == right_panel.n_founders
    )
    left_to_right: np.ndarray | None = None
    if full:
        left_to_right = np.asarray(
            [atomic_mapping[row] for row in range(left_panel.n_founders)],
            dtype=np.int64,
        )
    reason = "full_unique_correspondence" if full else (
        "partial_correspondence" if matched else "no_resolved_correspondence"
    )
    return PartialBoundaryLink(
        reason=reason,
        left_classes=left_classes,
        right_classes=right_classes,
        matched=matched,
        unmatched_left=unmatched_left,
        unmatched_right=unmatched_right,
        births=tuple(births),
        deaths=tuple(deaths),
        unresolved=tuple(unresolved),
        split_merge=tuple(split_merge),
        pair_evidence=tuple(evidence),
        informative_samples=informative.copy(),
        fold_labels=fold_labels,
        full_stitch=full,
        left_to_right=left_to_right,
    )

import haplotype_reconstruction.assembly.joint_completion as assembly_joint_completion
