"""Shared parent-state evidence types, model identifiers and scoring contracts."""
from __future__ import annotations


import numpy as np

from dataclasses import dataclass


import haplotype_reconstruction.core.parallel as core_parallel

DEFAULT_MISMATCH_PENALTY = -4.605170


core_parallel.ensure_numba_registry_warmup()


njit = core_parallel.original_njit


class PedigreeEvidenceError(ValueError):
    """Raised when explicit pedigree evidence is absent or internally invalid."""


_PARENT_STATE_NAMES = (
    "zero_observed_parents",
    "one_observed_parent",
    "two_observed_parents",
)


_ZERO_OBSERVED = 0


_ONE_OBSERVED = 1


_TWO_OBSERVED = 2


_EXTERNAL_PARENT = -1


PARENT_ELIGIBILITY_FORMAT_VERSION = 1


_PARENT_STATE_METHOD = "combined_v1"


_PARENT_STATE_LIKELIHOOD = "b1"


T09_RAGGED_POSTERIOR_MODE = "t09_ragged_posterior_v1"


RAGGED_QUADRATIC_MODEL = (
    "t09_ragged_projected_quadratic_v1"
)


T09_RAGGED_POSTERIOR_SOURCE_MODES = frozenset((
    T09_RAGGED_POSTERIOR_MODE,
    RAGGED_QUADRATIC_MODEL,
))


@dataclass(frozen=True)
class ParentStateEvidence:
    """Comparable forward evidence for 0/1/2 observed-parent models.

    The zero-parent vector is indexed by child, the one-parent matrix by
    ``[child, observed_parent]`` (with a ``-inf`` diagonal), and the two-parent
    vector by ``trios``. Scores must be normalized forward log likelihoods
    from the same observation model, not ranks, Viterbi maxima, or unrelated
    score scales. The two-parent panel may be screened; aggregation always
    uses the full eligible pair count as its identity-prior denominator.
    """

    contig: str
    trios: np.ndarray
    zero_parent_log_likelihoods: np.ndarray
    one_parent_log_likelihoods: np.ndarray
    two_parent_log_likelihoods: np.ndarray
    informative_markers: int
    edge_matched_bins: np.ndarray | None = None
    edge_exposed_bins: np.ndarray | None = None
    pair_explained_bins: np.ndarray | None = None
    pair_exposed_bins: np.ndarray | None = None
    structure_total_bins: float | None = None


@dataclass(frozen=True)
class ComponentEvidenceArrays:
    """Allele-grid representation derived from one historical contig input."""

    contig: str
    stacked_alleles: np.ndarray
    stacked_hom_mask: np.ndarray
    switch_costs: np.ndarray
    stay_costs: np.ndarray
    informative_markers: int
    stacked_labels: np.ndarray
    founder_alleles: np.ndarray
    selected_markers_per_bin: np.ndarray
    switch_probabilities: np.ndarray
    genotype_likelihoods: np.ndarray | None = None
    selected_positions: np.ndarray | None = None
    state_evidence_mode: str = "hard_allele"
