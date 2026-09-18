"""Cross-block founder occupancy and variable-count boundary eligibility."""
from __future__ import annotations


from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numba import njit, types, prange
from numba.typed import List
import haplotype_reconstruction.assembly.joint_completion as assembly_joint_completion

@dataclass(frozen=True)
class CrossBlockOccupancyRule:
    """Thresholds for abundant and rare founder release routes."""

    minimum_abundant_sum_squared_probability: float = 8.0
    abundant_high_confidence_probability: float = 0.8
    minimum_abundant_high_confidence_carriers: int = 5
    minimum_rare_local_carrier_probability: float = 0.9
    minimum_rare_neighbor_carrier_probability: float = 0.8
    minimum_same_sample_continuity: float = 0.72
    minimum_rare_continuity_samples: int = 2

    def __post_init__(self) -> None:
        if self.minimum_abundant_sum_squared_probability < 0.0:
            raise ValueError(
                "minimum_abundant_sum_squared_probability must be non-negative"
            )
        for name in (
            "abundant_high_confidence_probability",
            "minimum_rare_local_carrier_probability",
            "minimum_rare_neighbor_carrier_probability",
            "minimum_same_sample_continuity",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in (
            "minimum_abundant_high_confidence_carriers",
            "minimum_rare_continuity_samples",
        ):
            count = getattr(self, name)
            if isinstance(count, bool) or int(count) != count or count < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class FounderOccupancyDecision:
    """One row-level occupancy decision and its release diagnostics."""

    block_index: int
    founder_index: int
    keep: bool
    route: str
    reason: str
    sum_squared_carrier_probability: float
    high_confidence_carrier_count: int
    maximum_local_carrier_probability: float
    accepted_neighbor_count: int
    best_neighbor_block: int | None
    best_neighbor_founder: int | None
    best_sample: int | None
    best_local_carrier_probability: float
    best_neighbor_carrier_probability: float
    best_same_sample_continuity: float
    qualifying_continuity_sample_count: int
    qualifying_continuity_samples: tuple[int, ...]


@dataclass(frozen=True)
class VariableKOccupancyGateResult:
    """Immutable row decisions and strict-subset cavity fill masks."""

    profiles: tuple[assembly_joint_completion.CarrierProfiles, ...]
    block_underfit_flags: tuple[bool, ...]
    decisions: tuple[tuple[FounderOccupancyDecision, ...], ...]
    kept_fills: tuple[np.ndarray, ...]
    current_fill_count: int
    kept_fill_count: int
    abstained_fill_count: int
    rare_no_focal_observation_count: int
    decision_reason_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _AtomicNeighbor:
    block: int
    row: int
    boundary: int


def _normalised_evidence(genotype_likelihoods: np.ndarray) -> np.ndarray:
    evidence = np.asarray(genotype_likelihoods, dtype=np.float64)
    if evidence.ndim != 3 or evidence.shape[2] != 3:
        raise ValueError("genotype_likelihoods must have shape (samples, sites, 3)")
    if np.any(~np.isfinite(evidence)) or np.any(evidence < 0.0):
        raise ValueError("genotype_likelihoods must be finite and non-negative")
    total = np.sum(evidence, axis=2, keepdims=True)
    result = np.full(evidence.shape, 1.0 / 3.0, dtype=np.float64)
    np.divide(evidence, total, out=result, where=total > 0.0)
    return result


@njit(cache=True, parallel=True, fastmath=False)
def _focal_mask_rows(evidence, observed, output, offsets, tolerance):
    """Independent sample rows; divide before subtracting as in NumPy."""
    valid = np.ones(offsets[-1], np.bool_)
    for task in prange(offsets[-1]):
        block = np.searchsorted(offsets, task, side="right") - 1
        sample = task - offsets[block]
        values = evidence[block]
        mask = observed[block]
        result = output[block]
        for site in range(values.shape[1]):
            a, b, c = values[sample, site]
            if (not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c)
                    or a < 0.0 or b < 0.0 or c < 0.0):
                valid[task] = False
                result[sample, site] = False
                continue
            total = (a + b) + c
            if total > 0.0:
                a, b, c = a / total, b / total, c / total
                result[sample, site] = (
                    mask[sample, site] and max(a, b, c) - min(a, b, c) > tolerance)
            else:
                result[sample, site] = False
    return valid


def focal_informative_masks(
    genotype_likelihoods: Sequence[np.ndarray],
    observed: Sequence[np.ndarray],
    *,
    uniform_tolerance: float=1e-12,
) -> tuple[np.ndarray, ...]:
    """Return immutable observed, non-uniform sample-by-site masks."""

    if uniform_tolerance < 0.0:
        raise ValueError("uniform_tolerance must be non-negative")
    likelihoods = tuple(genotype_likelihoods)
    observed_masks = tuple(observed)
    if not likelihoods or len(likelihoods) != len(observed_masks):
        raise ValueError("genotype likelihoods and observed masks must align")
    # Keep one representation for writable/read-only and contiguous/strided
    # callers. The temporary views do not change the caller's write flags.
    values = List.empty_list(types.Array(types.float64, 3, "A", readonly=True))
    masks = List.empty_list(types.Array(types.boolean, 2, "A", readonly=True))
    output = List.empty_list(types.Array(types.boolean, 2, "C"))
    offsets = [0]
    for evidence, observed_mask in zip(likelihoods, observed_masks):
        evidence = np.asarray(evidence, dtype=np.float64)
        observed_mask = np.asarray(observed_mask, dtype=np.bool_)
        if evidence.ndim != 3 or evidence.shape[2] != 3:
            raise ValueError("genotype_likelihoods must have shape (samples, sites, 3)")
        if observed_mask.shape != evidence.shape[:2]:
            raise ValueError("observed masks must have shape (samples, sites)")
        frozen_evidence, frozen_mask = evidence.view(), observed_mask.view()
        frozen_evidence.setflags(write=False)
        frozen_mask.setflags(write=False)
        values.append(frozen_evidence)
        masks.append(frozen_mask)
        output.append(np.empty(evidence.shape[:2], dtype=np.bool_))
        offsets.append(offsets[-1] + evidence.shape[0])
    valid = _focal_mask_rows(values, masks, output,
                             np.asarray(offsets, dtype=np.int64), uniform_tolerance)
    if not np.all(valid):
        raise ValueError("genotype_likelihoods must be finite and non-negative")
    result = tuple(output)
    for value in result:
        value.setflags(write=False)
    return result


def _freeze_profile(profile: assembly_joint_completion.CarrierProfiles, block: int) -> assembly_joint_completion.CarrierProfiles:
    copies = np.asarray(profile.expected_copies, dtype=np.float64)
    carriers = np.asarray(profile.carrier_probability, dtype=np.float64)
    informative = np.asarray(profile.informative_samples, dtype=np.bool_)
    site_count = np.asarray(profile.informative_site_count, dtype=np.int64)
    if copies.ndim != 2 or copies.shape[1] < 1:
        raise ValueError(
            f"block {block} expected_copies must have shape (samples, rows)"
        )
    if carriers.shape != copies.shape:
        raise ValueError(
            f"block {block} carrier_probability must match expected_copies"
        )
    if informative.shape != (copies.shape[0],):
        raise ValueError(
            f"block {block} informative_samples must match samples"
        )
    if site_count.shape != (copies.shape[0],):
        raise ValueError(
            f"block {block} informative_site_count must match samples"
        )
    probability_tolerance = 32.0 * np.finfo(np.float64).eps
    if np.any(~np.isfinite(copies)) or np.any(
        (copies < -probability_tolerance)
        | (copies > 2.0 + probability_tolerance)
    ):
        raise ValueError("expected copy profiles must be finite and in [0, 2]")
    if np.any(~np.isfinite(carriers)) or np.any(
        (carriers < -probability_tolerance)
        | (carriers > 1.0 + probability_tolerance)
    ):
        raise ValueError("carrier probabilities must be finite and in [0, 1]")
    if np.any(site_count < 0):
        raise ValueError("informative site counts must be non-negative")
    copies = np.clip(copies, 0.0, 2.0)
    carriers = np.clip(carriers, 0.0, 1.0)

    values = []
    for array in (copies, carriers, informative, site_count):
        frozen = np.ascontiguousarray(array).copy()
        frozen.setflags(write=False)
        values.append(frozen)
    return assembly_joint_completion.CarrierProfiles(*values)


def _validate_class_partition(classes, n_rows: int, side: str) -> None:
    observed = []
    for expected_index, item in enumerate(classes):
        if item.index != expected_index or not item.rows:
            raise ValueError(f"{side} partial-link classes are not ordered")
        observed.extend(int(row) for row in item.rows)
    if sorted(observed) != list(range(n_rows)):
        raise ValueError(f"{side} partial-link classes do not partition profile rows")


def _atomic_pairs(link: assembly_partial_links.PartialBoundaryLink, boundary: int, n_samples: int):
    """Return only reciprocal unique accepted singleton-to-singleton edges."""

    if not isinstance(link, assembly_partial_links.PartialBoundaryLink):
        raise TypeError("every boundary link must be a PartialBoundaryLink")
    informative = np.asarray(link.informative_samples, dtype=np.bool_)
    if informative.shape != (n_samples,):
        raise ValueError("partial-link informative samples are not aligned")

    candidates = []
    for match in link.matched:
        if not bool(getattr(match.evidence, "accepted", False)):
            continue
        if not (len(match.left_rows) == len(match.right_rows) == 1):
            continue
        if not (0 <= match.left_class < len(link.left_classes)):
            raise ValueError("partial link has an invalid left class index")
        if not (0 <= match.right_class < len(link.right_classes)):
            raise ValueError("partial link has an invalid right class index")
        left_class = link.left_classes[match.left_class]
        right_class = link.right_classes[match.right_class]
        if (
            tuple(left_class.rows) != tuple(match.left_rows)
            or tuple(right_class.rows) != tuple(match.right_rows)
        ):
            raise ValueError("partial-link match rows disagree with their classes")
        candidates.append((int(match.left_rows[0]), int(match.right_rows[0])))

    left_counts = {}
    right_counts = {}
    for left, right in candidates:
        left_counts[left] = left_counts.get(left, 0) + 1
        right_counts[right] = right_counts.get(right, 0) + 1
    return tuple(
        (left, right)
        for left, right in candidates
        if left_counts[left] == 1 and right_counts[right] == 1
    )


def _neighbor_table(
    links: Sequence[assembly_partial_links.PartialBoundaryLink],
    row_counts: Sequence[int],
    n_samples: int,
):
    neighbors = [
        [[] for _ in range(row_count)] for row_count in row_counts
    ]
    boundary_informative = []
    for boundary, link in enumerate(links):
        _validate_class_partition(
            link.left_classes, row_counts[boundary], "left"
        )
        _validate_class_partition(
            link.right_classes, row_counts[boundary + 1], "right"
        )
        informative = np.asarray(link.informative_samples, dtype=np.bool_)
        boundary_informative.append(informative)
        for left, right in _atomic_pairs(link, boundary, n_samples):
            if not (0 <= left < row_counts[boundary]):
                raise ValueError("atomic left row is outside its carrier profile")
            if not (0 <= right < row_counts[boundary + 1]):
                raise ValueError("atomic right row is outside its carrier profile")
            neighbors[boundary][left].append(
                _AtomicNeighbor(boundary + 1, right, boundary)
            )
            neighbors[boundary + 1][right].append(
                _AtomicNeighbor(boundary, left, boundary)
            )
    return neighbors, tuple(boundary_informative)


def _decision_for_row(
    block_index: int,
    row_index: int,
    profiles: Sequence[assembly_joint_completion.CarrierProfiles],
    neighbors,
    boundary_informative,
    rule: CrossBlockOccupancyRule,
    block_underfit: bool,
) -> FounderOccupancyDecision:
    profile = profiles[block_index]
    local_probability = profile.carrier_probability[:, row_index]
    local_informative = profile.informative_samples
    informative_probability = local_probability[local_informative]
    sum_squared = float(np.sum(informative_probability ** 2))
    high_confidence_count = int(np.sum(
        informative_probability >= rule.abundant_high_confidence_probability
    ))
    maximum_local = (
        float(np.max(informative_probability))
        if informative_probability.size else 0.0
    )
    atomic_neighbors = neighbors[block_index][row_index]

    if block_underfit:
        return FounderOccupancyDecision(
            block_index, row_index, False, "abstain",
            "stage1_wildcard_panel_underfit",
            sum_squared, high_confidence_count, maximum_local,
            len(atomic_neighbors), None, None, None, 0.0, 0.0, 0.0, 0, (),
        )

    if profile.carrier_probability.shape[1] == 1:
        return FounderOccupancyDecision(
            block_index, row_index, False, "abstain",
            "single_founder_panel_underfit_unverifiable",
            sum_squared, high_confidence_count, maximum_local,
            len(atomic_neighbors), None, None, None, 0.0, 0.0, 0.0, 0, (),
        )

    if (
        sum_squared >= rule.minimum_abundant_sum_squared_probability
        and high_confidence_count
        >= rule.minimum_abundant_high_confidence_carriers
    ):
        return FounderOccupancyDecision(
            block_index, row_index, True, "abundant", "abundant_occupancy",
            sum_squared, high_confidence_count, maximum_local,
            len(atomic_neighbors), None, None, None, 0.0, 0.0, 0.0, 0, (),
        )

    best_score = 0.0
    best_neighbor_block = None
    best_neighbor_row = None
    best_sample = None
    best_local_probability = 0.0
    best_neighbor_probability = 0.0
    qualifying_neighbor_probability = False
    qualifying_samples: tuple[int, ...] = ()
    ambiguous_best_neighbor = False
    for neighbor in atomic_neighbors:
        neighbor_profile = profiles[neighbor.block]
        neighbor_probability = neighbor_profile.carrier_probability[:, neighbor.row]
        eligible = local_informative & neighbor_profile.informative_samples
        eligible &= boundary_informative[neighbor.boundary]
        eligible &= (
            local_probability >= rule.minimum_rare_local_carrier_probability
        )
        eligible &= (
            neighbor_probability
            >= rule.minimum_rare_neighbor_carrier_probability
        )
        qualifying_neighbor_probability |= bool(np.any(eligible))
        if not np.any(eligible):
            continue
        sample_indices = np.flatnonzero(eligible)
        products = local_probability[sample_indices] * neighbor_probability[sample_indices]
        boundary_samples = tuple(
            int(value) for value in sample_indices[
                products >= rule.minimum_same_sample_continuity
            ]
        )
        candidate_offset = int(np.argmax(products))
        candidate_score = float(products[candidate_offset])
        candidate_rank = (len(boundary_samples), candidate_score)
        best_rank = (len(qualifying_samples), best_score)
        if candidate_rank > best_rank:
            sample = int(sample_indices[candidate_offset])
            best_score = candidate_score
            best_neighbor_block = neighbor.block
            best_neighbor_row = neighbor.row
            best_sample = sample
            best_local_probability = float(local_probability[sample])
            best_neighbor_probability = float(neighbor_probability[sample])
            qualifying_samples = boundary_samples
            ambiguous_best_neighbor = False
        elif candidate_rank == best_rank and best_neighbor_block is not None:
            if (
                    neighbor.block != best_neighbor_block
                    or neighbor.row != best_neighbor_row
                    or boundary_samples != qualifying_samples):
                ambiguous_best_neighbor = True

    qualifying_count = len(qualifying_samples)
    if ambiguous_best_neighbor:
        keep = False
        route = "abstain"
        reason = "ambiguous_tied_atomic_neighbors"
    elif qualifying_count >= rule.minimum_rare_continuity_samples:
        keep = True
        route = "rare_continuity"
        reason = "rare_atomic_same_sample_continuity"
    else:
        keep = False
        route = "abstain"
        if informative_probability.size == 0:
            reason = "no_informative_samples"
        elif not atomic_neighbors:
            reason = "no_atomic_identifiable_neighbor"
        elif maximum_local < rule.minimum_rare_local_carrier_probability:
            reason = "insufficient_local_carrier_probability"
        elif not qualifying_neighbor_probability:
            reason = "insufficient_neighbor_carrier_probability"
        elif qualifying_count > 0:
            reason = "insufficient_rare_continuity_samples"
        else:
            reason = "insufficient_same_sample_continuity"

    return FounderOccupancyDecision(
        block_index=block_index,
        founder_index=row_index,
        keep=keep,
        route=route,
        reason=reason,
        sum_squared_carrier_probability=sum_squared,
        high_confidence_carrier_count=high_confidence_count,
        maximum_local_carrier_probability=maximum_local,
        accepted_neighbor_count=len(atomic_neighbors),
        best_neighbor_block=best_neighbor_block,
        best_neighbor_founder=best_neighbor_row,
        best_sample=best_sample,
        best_local_carrier_probability=best_local_probability,
        best_neighbor_carrier_probability=best_neighbor_probability,
        best_same_sample_continuity=best_score,
        qualifying_continuity_sample_count=qualifying_count,
        qualifying_continuity_samples=qualifying_samples,
    )


def apply_variable_k_occupancy_gate(
    cavity_fill_masks: Sequence[np.ndarray],
    profiles: Sequence[assembly_joint_completion.CarrierProfiles],
    focal_informative: Sequence[np.ndarray],
    links: Sequence[assembly_partial_links.PartialBoundaryLink],
    rule: CrossBlockOccupancyRule=CrossBlockOccupancyRule(),
    *,
    block_underfit_flags: Sequence[bool] | None=None,
) -> VariableKOccupancyGateResult:
    """Gate fills using pre-cavity occupancy and Stage-1 underfit flags."""

    masks = tuple(np.asarray(value, dtype=np.bool_) for value in cavity_fill_masks)
    if not masks:
        raise ValueError("at least one cavity-fill mask is required")
    if len(profiles) != len(masks) or len(focal_informative) != len(masks):
        raise ValueError("fill masks, carrier profiles, and focal masks must align")
    if len(links) != len(masks) - 1:
        raise ValueError("one partial link is required per adjacent boundary")
    if block_underfit_flags is None:
        underfit_flags = (False,) * len(masks)
    else:
        raw_underfit = np.asarray(block_underfit_flags)
        if raw_underfit.shape != (len(masks),):
            raise ValueError(
                "block_underfit_flags must contain one flag per block"
            )
        if np.any(~np.isin(raw_underfit, (False, True))):
            raise ValueError("block_underfit_flags must be boolean")
        underfit_flags = tuple(bool(value) for value in raw_underfit)
    frozen_profiles = tuple(
        _freeze_profile(profile, block)
        for block, profile in enumerate(profiles)
    )
    n_samples = frozen_profiles[0].carrier_probability.shape[0]
    if any(
        profile.carrier_probability.shape[0] != n_samples
        for profile in frozen_profiles
    ):
        raise ValueError("all carrier profiles must share the sample axis")

    row_counts = tuple(profile.carrier_probability.shape[1] for profile in frozen_profiles)
    focal_masks = tuple(
        np.asarray(value, dtype=np.bool_) for value in focal_informative
    )
    for block, (mask, focal, row_count) in enumerate(
        zip(masks, focal_masks, row_counts)
    ):
        if mask.ndim != 2 or mask.shape[0] != row_count:
            raise ValueError(
                f"block {block} cavity-fill mask must have shape (rows, sites)"
            )
        if focal.shape != (n_samples, mask.shape[1]):
            raise ValueError(
                f"block {block} focal mask must have shape (samples, sites)"
            )

    neighbors, boundary_informative = _neighbor_table(
        links, row_counts, n_samples
    )
    decisions = []
    kept_fills = []
    current_count = 0
    kept_count = 0
    rare_no_focal_count = 0
    reason_counts = {}
    for block, (fill_mask, focal_mask) in enumerate(zip(masks, focal_masks)):
        block_decisions = tuple(
            _decision_for_row(
                block, row, frozen_profiles, neighbors,
                boundary_informative, rule, underfit_flags[block],
            )
            for row in range(row_counts[block])
        )
        kept = np.zeros_like(fill_mask, dtype=np.bool_)
        for row, decision in enumerate(block_decisions):
            reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1
            if decision.route == "abundant":
                kept[row] = fill_mask[row]
            elif decision.route == "rare_continuity":
                samples = np.asarray(
                    decision.qualifying_continuity_samples, dtype=np.int64
                )
                focal_observed = np.any(focal_mask[samples], axis=0)
                kept[row] = fill_mask[row] & focal_observed
                rare_no_focal_count += int(np.sum(fill_mask[row] & ~focal_observed))
        if kept.shape != fill_mask.shape or np.any(kept & ~fill_mask):
            raise AssertionError("occupancy output is not a strict input subset")
        kept = np.ascontiguousarray(kept)
        kept.setflags(write=False)
        decisions.append(block_decisions)
        kept_fills.append(kept)
        current_count += int(np.sum(fill_mask))
        kept_count += int(np.sum(kept))

    abstained_count = current_count - kept_count
    if current_count != kept_count + abstained_count:
        raise AssertionError("occupancy fill counts are not conserved")
    return VariableKOccupancyGateResult(
        profiles=frozen_profiles,
        block_underfit_flags=underfit_flags,
        decisions=tuple(decisions),
        kept_fills=tuple(kept_fills),
        current_fill_count=current_count,
        kept_fill_count=kept_count,
        abstained_fill_count=abstained_count,
        rare_no_focal_observation_count=rare_no_focal_count,
        decision_reason_counts=tuple(sorted(reason_counts.items())),
    )

import haplotype_reconstruction.assembly.partial_links as assembly_partial_links
