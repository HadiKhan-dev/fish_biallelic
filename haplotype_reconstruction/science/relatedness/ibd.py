"""Accumulate the optional pairwise IBD report independently of family fitting.

Accretion (P0 RL04): report only, quadratic in samples and linear in thinned sites.
Owns pair state and final means; no storage, family model, environment or clocks.
Method: docs/methods/relatedness.md#pairwise-ibd. Contigs stay in bound order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.likelihoods import CodedLikelihoods, gather_likelihoods
from ...contracts.observations import PairArray
from ...contracts.relatedness import RelatednessPairs
from .frequency import allele_frequency
from .ibd_kernels import IBD_MASKS, pair_ibd, pair_indices
from .ibd_parameters import PairwiseIbdParameters
from .parameters import AlleleFrequencyParameters

_DEFAULT_FREQUENCY = AlleleFrequencyParameters()
_DEFAULT_IBD = PairwiseIbdParameters()
_NORMALIZE_BATCH = 262144  # 3.5 stage.py:145; storage batching only.

__all__ = ["PairwiseIbdState", "new_pairwise_ibd", "accumulate_pairwise_ibd", "finish_pairwise_ibd", "RelatednessPairs"]


class PairwiseEvidence(Protocol):
    """Read-only evidence view required by the report chain."""

    @property
    def sample_ids(self) -> tuple[str, ...]: ...

    @property
    def positions(self) -> np.ndarray: ...

    @property
    def likelihoods(self) -> np.ndarray | CodedLikelihoods: ...

    @property
    def observed(self) -> np.ndarray | PairArray: ...


@dataclass
class PairwiseIbdState:
    """Mutable pair accumulators: int64[P,2], float64[P,4] posterior, float64[P,3] log likelihood."""

    pairs: np.ndarray
    posterior: np.ndarray
    loglik: np.ndarray


def new_pairwise_ibd(samples: int) -> PairwiseIbdState:
    """Allocate zeroed accumulators in row-major upper-triangle pair order."""
    pairs = pair_indices(samples, 0, samples * (samples - 1) // 2)
    return PairwiseIbdState(pairs, np.zeros((len(pairs), 4), order="F"), np.zeros((len(pairs), len(IBD_MASKS))))


def accumulate_pairwise_ibd(
    state: PairwiseIbdState,
    evidence: PairwiseEvidence,
    genetic_map: ChromosomeGeneticMap,
    ibd_parameters: PairwiseIbdParameters = _DEFAULT_IBD,
    frequency_parameters: AlleleFrequencyParameters = _DEFAULT_FREQUENCY,
) -> None:
    """Add one contig at the unconditional site stride, without the family frequency/observation gates.

    Keep reverse site accumulation and forward masked-chain totals.
    """
    gl, observed, positions = evidence.likelihoods, evidence.observed, evidence.positions
    if not len(positions):
        return
    p = allele_frequency(gl, observed, frequency_parameters)
    step = ibd_parameters.ibd_site_step
    morgans = np.asarray(genetic_map.cumulative_morgans(positions[::step]), dtype=np.float64)
    stay = np.ones(len(morgans))
    stay[1:] = np.exp(-ibd_parameters.ibd_switch_rate * np.diff(morgans))
    pair_ibd(
        gather_likelihoods(gl, slice(None, None, step)),
        np.ascontiguousarray(observed[:, ::step]),
        np.ascontiguousarray(p[::step]),
        stay,
        state.pairs,
        IBD_MASKS,
        state.posterior,
        state.loglik,
    )


def finish_pairwise_ibd(state: PairwiseIbdState) -> RelatednessPairs:
    """Normalize state in place exactly once, returning the five pair report arrays."""
    for lo in range(0, len(state.pairs), _NORMALIZE_BATCH):
        block = state.posterior[lo : lo + _NORMALIZE_BATCH]
        block[:, :3] /= np.maximum(block[:, 3:4], 1.0)
    return RelatednessPairs(
        pairs=state.pairs,
        ibd=state.posterior[:, :3],
        ibd_sites=state.posterior[:, 3],
        lr_ibd0=state.loglik[:, 0] - state.loglik[:, 1],
        lr_ibd2=state.loglik[:, 0] - state.loglik[:, 2],
    )
