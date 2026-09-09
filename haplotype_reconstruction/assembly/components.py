"""assembly / components for the canonical reconstruction pipeline."""
from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np


@dataclass(frozen=True)
class BoundaryMapping:
    """One adjacent-boundary decision.

    ``left_to_right`` uses sorted local founder-key indices.  ``None`` means
    that no complete founder correspondence was identified.  A non-``None``
    map must be a complete permutation; partial maps belong in diagnostics,
    not in this materialiser.
    """

    left_to_right: np.ndarray | None
    reason: str

    def __post_init__(self) -> None:
        mapping = self.left_to_right
        if mapping is not None:
            values = np.asarray(mapping, dtype=np.int64)
            if values.ndim != 1:
                raise ValueError("left_to_right must be one-dimensional")
            object.__setattr__(self, "left_to_right", values.copy())
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("boundary reason must be non-empty")


def _full_permutation(mapping: np.ndarray | None, left_k: int, right_k: int) -> bool:
    if mapping is None or left_k != right_k:
        return False
    values = np.asarray(mapping, dtype=np.int64)
    return (
        values.shape == (left_k,)
        and np.array_equal(np.sort(values), np.arange(right_k, dtype=np.int64))
    )


def _ordered_haplotypes(block: core_haplotypes.BlockResult) -> tuple[tuple[Any, ...], list[np.ndarray]]:
    keys = tuple(sorted(block.haplotypes))
    arrays = [np.asarray(block.haplotypes[key]) for key in keys]
    n_sites = len(block.positions)
    for array in arrays:
        if array.shape not in ((n_sites,), (n_sites, 2)):
            raise ValueError("founder haplotype has the wrong site dimension")
    if arrays and any(array.ndim != arrays[0].ndim for array in arrays):
        raise ValueError("founder haplotype representations must agree within a block")
    return keys, arrays


def _keep_flags(block: core_haplotypes.BlockResult) -> np.ndarray:
    flags = getattr(block, "keep_flags", None)
    if flags is None:
        return np.ones(len(block.positions), dtype=np.int8)
    values = np.asarray(flags)
    if values.shape != (len(block.positions),):
        raise ValueError("keep_flags are not aligned with positions")
    return values.copy()


def _support(block: core_haplotypes.BlockResult, shape: tuple[int, int]) -> np.ndarray | None:
    values = getattr(block, "n_directional_site_supporters", None)
    if values is None:
        return None
    result = np.asarray(values)
    if result.shape != shape:
        raise ValueError("founder support is not aligned with the block")
    return result.copy()


def _probability(block: core_haplotypes.BlockResult, fallback: np.ndarray) -> np.ndarray:
    values = getattr(block, "founder_alt_pseudo_probability", None)
    if values is None:
        return np.asarray(fallback, dtype=np.float64).copy()
    result = np.asarray(values, dtype=np.float64)
    if result.shape != fallback.shape:
        raise ValueError("founder pseudo-probability is not aligned with the block")
    if np.any(~np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("founder pseudo-probability must lie in [0, 1]")
    return result.copy()


def _copy_single_component(block: core_haplotypes.BlockResult, block_index: int) -> dict[str, Any]:
    keys, haplotypes = _ordered_haplotypes(block)
    released = assembly_observations.founder_panel_from_block_result(block)
    inference = assembly_observations.founder_inference_panel_from_block_result(block)
    if released.keys != keys or inference.keys != keys:
        raise ValueError("founder panel key order disagrees with haplotypes")
    support = _support(block, released.q.shape)
    return {
        "positions": np.asarray(block.positions).copy(),
        "haplotypes": [value.copy() for value in haplotypes],
        "discrete": np.where(released.called, released.q, -1).astype(np.int8),
        "inference_discrete": np.where(
            inference.called, inference.q, -1
        ).astype(np.int8),
        "probability": _probability(block, released.q),
        "support": support,
        "keep_flags": _keep_flags(block),
        "source_block_indices": [int(block_index)],
        "source_founder_keys": [keys],
        # Component rows initially use the first block's local row order.
        # After every append this records component-root row -> most recent
        # block-local row, so the next adjacent mapping can be composed.
        "terminal_local_order": np.arange(len(keys), dtype=np.int64),
        "break_before": bool(getattr(block, "missing_aware_break_before", False)),
        "break_reason_before": getattr(
            block, "missing_aware_break_reason_before", None
        ),
        "break_after": bool(getattr(block, "missing_aware_break_after", False)),
        "break_reason_after": getattr(
            block, "missing_aware_break_reason_after", None
        ),
        "genotype_evidence_mode": getattr(block, "genotype_evidence_mode", None),
    }


def _append_block(
    state: dict[str, Any], block: core_haplotypes.BlockResult, block_index: int, mapping: np.ndarray
) -> None:
    keys, haplotypes = _ordered_haplotypes(block)
    released = assembly_observations.founder_panel_from_block_result(block)
    inference = assembly_observations.founder_inference_panel_from_block_result(block)
    if released.keys != keys or inference.keys != keys:
        raise ValueError("founder panel key order disagrees with haplotypes")
    k = len(state["haplotypes"])
    if not _full_permutation(mapping, k, len(keys)):
        raise ValueError("component append requires a complete permutation")
    if np.asarray(block.positions)[0] <= state["positions"][-1]:
        raise ValueError("component blocks must have increasing, non-overlapping positions")

    adjacent_mapping = np.asarray(mapping, dtype=np.int64)
    terminal_order = np.asarray(state["terminal_local_order"], dtype=np.int64)
    if terminal_order.shape != (k,) or not np.array_equal(
            np.sort(terminal_order), np.arange(k, dtype=np.int64)):
        raise ValueError("component terminal local order is not a permutation")
    # Compose previous-local -> right-local into component-root row order.
    order = adjacent_mapping[terminal_order]
    for row in range(k):
        state["haplotypes"][row] = np.concatenate(
            (state["haplotypes"][row], haplotypes[int(order[row])]), axis=0
        )
    state["positions"] = np.concatenate((state["positions"], block.positions))
    state["discrete"] = np.concatenate(
        (
            state["discrete"],
            np.where(released.called, released.q, -1).astype(np.int8)[order],
        ),
        axis=1,
    )
    state["inference_discrete"] = np.concatenate(
        (
            state["inference_discrete"],
            np.where(inference.called, inference.q, -1).astype(np.int8)[order],
        ),
        axis=1,
    )
    state["probability"] = np.concatenate(
        (state["probability"], _probability(block, released.q)[order]), axis=1
    )
    right_support = _support(block, released.q.shape)
    if state["support"] is None or right_support is None:
        state["support"] = None
    else:
        state["support"] = np.concatenate(
            (state["support"], right_support[order]), axis=1
        )
    state["keep_flags"] = np.concatenate(
        (state["keep_flags"], _keep_flags(block))
    )
    state["source_block_indices"].append(int(block_index))
    state["source_founder_keys"].append(tuple(keys[int(value)] for value in order))
    state["terminal_local_order"] = order.copy()
    state["break_after"] = bool(
        getattr(block, "missing_aware_break_after", False)
    )
    state["break_reason_after"] = getattr(
        block, "missing_aware_break_reason_after", None
    )


def _materialise(state: dict[str, Any], component_id: int) -> core_haplotypes.BlockResult:
    block = core_haplotypes.BlockResult(
        positions=np.ascontiguousarray(state["positions"]),
        haplotypes={
            row: np.ascontiguousarray(value)
            for row, value in enumerate(state["haplotypes"])
        },
        keep_flags=np.ascontiguousarray(state["keep_flags"]),
        genotype_evidence_mode=state["genotype_evidence_mode"],
    )
    block.discrete_haps = np.ascontiguousarray(state["discrete"])
    block.missing_aware_inference_discrete_haps = np.ascontiguousarray(
        state["inference_discrete"]
    )
    block.founder_alt_pseudo_probability = np.ascontiguousarray(
        state["probability"]
    )
    if state["support"] is not None:
        block.n_directional_site_supporters = np.ascontiguousarray(
            state["support"]
        )
    block.missing_aware_phase_component_id = int(component_id)
    block.missing_aware_source_block_indices = tuple(
        state["source_block_indices"]
    )
    block.missing_aware_source_founder_keys = tuple(
        state["source_founder_keys"]
    )
    block.missing_aware_break_before = bool(state["break_before"])
    block.missing_aware_break_after = bool(state["break_after"])
    if state["break_reason_before"] is not None:
        block.missing_aware_break_reason_before = state["break_reason_before"]
    if state["break_reason_after"] is not None:
        block.missing_aware_break_reason_after = state["break_reason_after"]
    return block


def assemble_phase_components(
    blocks: Sequence[core_haplotypes.BlockResult], boundaries: Sequence[BoundaryMapping]
) -> core_haplotypes.BlockResults:
    """Return ordered components, crossing only supplied full permutations."""

    source = list(blocks)
    decisions = list(boundaries)
    if not source:
        raise ValueError("at least one input block is required")
    if len(decisions) != len(source) - 1:
        raise ValueError("one boundary decision is required per adjacent pair")
    for left, right in zip(source, source[1:]):
        if np.asarray(right.positions)[0] <= np.asarray(left.positions)[-1]:
            raise ValueError("blocks must be ordered and non-overlapping")

    components: list[core_haplotypes.BlockResult] = []
    state = _copy_single_component(source[0], 0)
    for boundary_index, (decision, right) in enumerate(
        zip(decisions, source[1:])
    ):
        left_k = len(state["haplotypes"])
        right_k = len(right.haplotypes)
        if _full_permutation(decision.left_to_right, left_k, right_k):
            _append_block(
                state, right, boundary_index + 1, decision.left_to_right
            )
            continue

        state["break_after"] = True
        state["break_reason_after"] = decision.reason
        components.append(_materialise(state, len(components)))
        state = _copy_single_component(right, boundary_index + 1)
        state["break_before"] = True
        state["break_reason_before"] = decision.reason
    components.append(_materialise(state, len(components)))
    return core_haplotypes.BlockResults(components)

import haplotype_reconstruction.assembly.observations as assembly_observations
import haplotype_reconstruction.core.haplotypes as core_haplotypes
