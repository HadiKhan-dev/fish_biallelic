"""Metadata-free, stability-aware pedigree inference.

This module is import-compatible with :mod:`pedigree_inference`. Historical
``tolerance_painting`` plus ``founder_block`` inputs automatically activate
comparable normalized forward models for zero, one, and two observed parents.
Explicit parent-state evidence is also supported; historical pair-only smart
and raw genotype-likelihood evidence remain compatibility paths. Only an
unknown nonstandard
input schema is delegated unchanged to the historical implementation. No
smart path reads spreadsheets, parses sample names, or invents cohort or sex
eligibility rules.

The smart model separates four questions which the historical single MAP
pedigree conflated:

* whether the data support zero, one, or two observed parents;
* which candidate identity is supported conditional on that parent-count state;
* whether state and identity are stable to chromosome bootstrap and leave-one-
  chromosome-out (LOCO) analyses; and
* which jointly selected variable-edge configurations form a directed acyclic
  graph.

On the historical pair-only compatibility path, a chromosome is informative
when it contains a genuine scale-aware candidate-score range, even if two
leading identities remain tied. On the parent-state path used by standard
painting inputs, state-tier informativeness instead requires a unique marginal
M0/M1/M2 winner on that chromosome. Full-data, bootstrap, and LOCO identity
selection always requires a unique winner and never resolves a tied top score
by candidate or sample-array order.

For direct APIs, ``relationships`` is the configured scientific reporting view
(Tier B by default). Every scientific/status frame retains every sample; a row
with no reported parents can mean either a top-level individual or unresolved
evidence and is not itself evidence of founder status. Smart results also
expose ``tier_a_relationships``, ``tier_b_relationships``,
``complete_relationships`` and ``smart_diagnostics``. The exact-signature
legacy pipeline wrapper additionally preserves this clean table as
``scientific_relationships`` and places a visibly labelled internal
``pipeline_control_adapter`` in ``relationships``. That adapter is not an
export pedigree. The complete scientific view is a leading acyclic hypothesis
for identifiable children, not a claim that every edge is established
parentage.

Scientific limitation
---------------------
With no breeding records, generation labels, or sex information, parent roles
are unordered and direction can be weakly identifiable. ``Parent1`` and
``Parent2`` are therefore ordered only by sample-array index. Bootstrap/LOCO
fractions measure internal stability, not calibrated posterior probability.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import operator
import os
import warnings
from typing import Any, Mapping, Optional, Sequence

import thread_config  # must precede NumPy/Numba imports
import dynamic_threads

# thread_config wraps numba.njit so project kernels default to disk caching.
# Numba also decorates some package-internal helpers lazily, and those helpers
# do not always have a cache locator. Temporarily restore Numba's real
# decorator only while forcing those helpers to import, then put the project's
# wrapper back. Smart kernels bind the real decorator locally and explicitly
# opt into disk caching; importing this module must not change the caching
# policy for modules imported later in the pipeline.

try:
    import numba
    from numba import prange

    _project_njit_wrapper = numba.njit
    _real_njit = getattr(thread_config, "_original_njit", _project_njit_wrapper)
    try:
        numba.njit = _real_njit

        # A tiny compile forces all lazily imported CPU registries (including
        # numba.typed dict/list helpers) to bind the real decorator now.
        @_real_njit(cache=False)
        def _smart_numba_registry_warmup(value):
            return value + 1

        _smart_numba_registry_warmup(0)
    finally:
        numba.njit = _project_njit_wrapper
    njit = _real_njit
    SMART_HAS_NUMBA = True
except ImportError:
    numba = None
    SMART_HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(function):
            return function
        return decorator

    prange = range

import numpy as np
import pandas as pd

import pedigree_inference as _legacy


# Re-export the historical public surface. Definitions below deliberately
# override only the main inference function; callers using numerical helpers,
# PedigreeResult, or draw_pedigree_tree continue to see the legacy objects.
for _public_name in dir(_legacy):
    if not _public_name.startswith("_") and _public_name not in globals():
        globals()[_public_name] = getattr(_legacy, _public_name)

PedigreeResult = _legacy.PedigreeResult
draw_pedigree_tree = _legacy.draw_pedigree_tree
DEFAULT_MISMATCH_PENALTY = _legacy.DEFAULT_MISMATCH_PENALTY


class SmartEvidenceError(ValueError):
    """Raised when explicit smart evidence is absent or internally invalid."""


_PARENT_STATE_NAMES = (
    "zero_observed_parents",
    "one_observed_parent",
    "two_observed_parents",
)
_ZERO_OBSERVED = 0
_ONE_OBSERVED = 1
_TWO_OBSERVED = 2
_EXTERNAL_PARENT = -1


@dataclass(frozen=True)
class SmartPedigreeConfig:
    """Numerical and reporting policy for metadata-free inference.

    Thresholds label evidence tiers; they are not biological acceptance
    probabilities. Defaults deliberately require evidence from at least
    three independently aggregated contigs.
    """

    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 20260725
    markers_per_information_block: int = 100
    information_tempering_power: float = 0.5
    maximum_contig_weight_ratio: float = 4.0
    rank_weight: float = 0.35
    chromosome_contamination: float = 0.02
    linked_evidence_weight: float = 0.65
    genotype_evidence_weight: float = 0.35
    minimum_informative_contigs: int = 3
    tier_a_pair_bootstrap: float = 0.95
    tier_a_parent_bootstrap: float = 0.95
    tier_a_loco_fraction: float = 0.95
    tier_b_pair_bootstrap: float = 0.70
    tier_b_parent_bootstrap: float = 0.70
    tier_b_loco_fraction: float = 0.80
    support_set_coverage: float = 0.95
    primary_view: str = "tier_b"
    dag_local_search_passes: int = 3
    parent_state_mismatch_probability: float = 0.01
    parent_state_phase_switch_probability: float = 0.01
    parent_state_contamination_probability: float = 0.02
    parent_state_effective_markers_per_information_block: float = 1.0
    parent_state_external_state_pseudocount: float = 1.0
    parent_state_external_transition_pseudocount: float = 20.0
    parent_state_priors: tuple[float, float, float] = (
        1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    )
    parent_state_prior_sensitivity: tuple[
        tuple[float, float, float], ...
    ] = (
        (0.50, 0.30, 0.20),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (0.20, 0.30, 0.50),
    )
    parent_state_prior_strength: float = 3.0
    parent_state_prior_max_iterations: int = 100
    parent_state_prior_tolerance: float = 1e-10

    def validated(self) -> "SmartPedigreeConfig":
        def require_integer(name: str, minimum: int) -> None:
            raw_value = getattr(self, name)
            if isinstance(raw_value, (bool, np.bool_)):
                raise SmartEvidenceError(
                    f"{name} must be an integer of at least {minimum}"
                )
            try:
                value = operator.index(raw_value)
            except TypeError as exc:
                raise SmartEvidenceError(
                    f"{name} must be an integer of at least {minimum}"
                ) from exc
            if value < minimum:
                raise SmartEvidenceError(
                    f"{name} must be an integer of at least {minimum}"
                )

        for integer_name, minimum in (
            ("bootstrap_replicates", 1),
            ("bootstrap_seed", 0),
            ("markers_per_information_block", 1),
            ("minimum_informative_contigs", 1),
            ("dag_local_search_passes", 0),
            ("parent_state_prior_max_iterations", 1),
        ):
            require_integer(integer_name, minimum)
        if (
            not np.isfinite(self.maximum_contig_weight_ratio)
            or self.maximum_contig_weight_ratio < 1.0
        ):
            raise SmartEvidenceError(
                "maximum_contig_weight_ratio must be finite and at least one"
            )
        for name in (
            "information_tempering_power",
            "rank_weight",
            "chromosome_contamination",
            "linked_evidence_weight",
            "genotype_evidence_weight",
            "tier_a_pair_bootstrap",
            "tier_a_parent_bootstrap",
            "tier_a_loco_fraction",
            "tier_b_pair_bootstrap",
            "tier_b_parent_bootstrap",
            "tier_b_loco_fraction",
            "support_set_coverage",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise SmartEvidenceError(f"{name} must be finite")
        if not 0.0 <= self.rank_weight <= 1.0:
            raise SmartEvidenceError("rank_weight must lie in [0, 1]")
        if not 0.0 <= self.chromosome_contamination < 1.0:
            raise SmartEvidenceError(
                "chromosome_contamination must lie in [0, 1)"
            )
        for name in (
            "tier_a_pair_bootstrap",
            "tier_a_parent_bootstrap",
            "tier_a_loco_fraction",
            "tier_b_pair_bootstrap",
            "tier_b_parent_bootstrap",
            "tier_b_loco_fraction",
            "support_set_coverage",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise SmartEvidenceError(f"{name} must lie in [0, 1]")
        if self.linked_evidence_weight < 0.0 or self.genotype_evidence_weight < 0.0:
            raise SmartEvidenceError("evidence weights must be non-negative")
        if self.linked_evidence_weight + self.genotype_evidence_weight <= 0.0:
            raise SmartEvidenceError("at least one evidence weight must be positive")
        if self.primary_view not in {"tier_a", "tier_b", "complete"}:
            raise SmartEvidenceError(
                "primary_view must be 'tier_a', 'tier_b', or 'complete'"
            )
        if (
            not np.isfinite(self.parent_state_mismatch_probability)
            or not 0.0 < self.parent_state_mismatch_probability < 0.5
        ):
            raise SmartEvidenceError(
                "parent_state_mismatch_probability must lie in (0, 0.5)"
            )
        if (
            not np.isfinite(self.parent_state_phase_switch_probability)
            or not 0.0 <= self.parent_state_phase_switch_probability <= 0.5
        ):
            raise SmartEvidenceError(
                "parent_state_phase_switch_probability must lie in [0, 0.5]"
            )
        if (
            not np.isfinite(self.parent_state_contamination_probability)
            or not 0.0 <= self.parent_state_contamination_probability < 1.0
        ):
            raise SmartEvidenceError(
                "parent_state_contamination_probability must lie in [0, 1)"
            )
        for name in (
            "parent_state_effective_markers_per_information_block",
            "parent_state_external_state_pseudocount",
            "parent_state_external_transition_pseudocount",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise SmartEvidenceError(f"{name} must be finite and positive")
        priors = np.asarray(self.parent_state_priors, dtype=np.float64)
        if (
            priors.shape != (3,)
            or np.any(~np.isfinite(priors))
            or np.any(priors <= 0.0)
            or not np.isclose(np.sum(priors), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise SmartEvidenceError(
                "parent_state_priors must be three positive probabilities "
                "summing to one"
            )
        if (
            not np.isfinite(self.parent_state_prior_strength)
            or self.parent_state_prior_strength <= 0.0
        ):
            raise SmartEvidenceError(
                "parent_state_prior_strength must be finite and positive"
            )
        if (
            not np.isfinite(self.parent_state_prior_tolerance)
            or self.parent_state_prior_tolerance <= 0.0
        ):
            raise SmartEvidenceError(
                "parent_state_prior_tolerance must be finite and positive"
            )
        sensitivity = np.asarray(
            self.parent_state_prior_sensitivity, dtype=np.float64
        )
        if (
            sensitivity.ndim != 2
            or sensitivity.shape[1] != 3
            or len(sensitivity) == 0
            or np.any(~np.isfinite(sensitivity))
            or np.any(sensitivity <= 0.0)
            or np.any(~np.isclose(
                np.sum(sensitivity, axis=1), 1.0, rtol=0.0, atol=1e-12
            ))
        ):
            raise SmartEvidenceError(
                "parent_state_prior_sensitivity must contain positive "
                "probability triples summing to one"
            )
        return self


@dataclass(frozen=True)
class SmartContigEvidence:
    """Explicit chromosome evidence for a fixed candidate-trio panel.

    ``trios`` is an integer ``(n_rows, 3)`` array with columns
    ``child, parent1, parent2``. Every contig must contain the same canonical
    trio keys, although row order may differ. Scores are relative log-like
    evidence and may be on different scales: each source is robustly
    transformed within chromosome and child before aggregation.
    """

    contig: str
    trios: np.ndarray
    linked_log_likelihoods: np.ndarray
    genotype_log_likelihoods: np.ndarray
    informative_markers: int


@dataclass(frozen=True)
class SmartParentStateEvidence:
    """Comparable forward evidence for 0/1/2 observed-parent models.

    The zero-parent vector is indexed by child, the one-parent matrix by
    ``[child, observed_parent]`` (with a ``-inf`` diagonal), and the two-parent
    vector by ``trios``. Scores must be normalized forward log likelihoods
    from the same observation model, not ranks, Viterbi maxima, or unrelated
    score scales. The two-parent panel may be screened; aggregation always
    uses the full unrestricted pair count as its identity-prior denominator.
    """

    contig: str
    trios: np.ndarray
    zero_parent_log_likelihoods: np.ndarray
    one_parent_log_likelihoods: np.ndarray
    two_parent_log_likelihoods: np.ndarray
    informative_markers: int


@dataclass(frozen=True)
class _StandardContigCache:
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



_TINY = np.finfo(np.float64).tiny


def _apply_smart_dynamic_threads() -> int:
    """Settle the shared remainder allocation before a parallel kernel.

    ``dynamic_threads`` intentionally reads its active counter lock-free. If
    several workers release stale remainder claims together, the first pass
    can temporarily leave a few remainder cores idle. A second public-API
    pass immediately lets non-holders reclaim those cores without changing
    the foundational allocator or the total thread budget.
    """
    dynamic_threads.apply_dynamic_threads()
    return dynamic_threads.apply_dynamic_threads()


@dataclass(frozen=True)
class _ParentStateContigScores:
    """Comparable forward likelihoods and ancestry depth for one contig."""

    zero_observed: np.ndarray
    one_observed: np.ndarray
    two_observed: np.ndarray
    ancestry_junction_counts: np.ndarray
    ancestry_callable_haplotype_bins: np.ndarray


@njit(cache=True, inline="always")
def _source_alt_probability(hard_allele, background_alt):
    if hard_allele < 0:
        return background_alt
    return float(hard_allele)


@njit(cache=True, inline="always")
def _observed_allele_log_probability(observed, source_alt, mismatch_probability):
    if observed < 0:
        return 0.0
    if observed == 1:
        probability = (
            (1.0 - mismatch_probability) * source_alt
            + mismatch_probability * (1.0 - source_alt)
        )
    else:
        probability = (
            (1.0 - mismatch_probability) * (1.0 - source_alt)
            + mismatch_probability * source_alt
        )
    return math.log(max(probability, _TINY))


@njit(cache=True, inline="always")
def _phase_switch_probability(
    previous_homozygous,
    current_homozygous,
    phase_switch_probability,
):
    # Reconstructed homolog orientation is unidentified across a homozygous or
    # wholly missing block. Resetting to 1/2 is normalized and avoids treating
    # arbitrary painting phase as biological transmission evidence.
    if previous_homozygous or current_homozygous:
        return 0.5
    return phase_switch_probability


@njit(cache=True, parallel=True)
def _local_ibs_class_kernel(founders):
    """Assign first-occurrence local classes independently by cache bin."""
    n_states, n_bins, n_snps = founders.shape
    mapping = np.empty((n_bins, n_states), dtype=np.int16)
    class_counts = np.empty(n_bins, dtype=np.int16)
    pooled_founders = np.full(
        (n_states, n_bins, n_snps), -1, dtype=np.int8
    )
    active = np.zeros((n_bins, n_states), dtype=np.bool_)
    for block in prange(n_bins):
        number_of_classes = 0
        for state in range(n_states):
            local_class = -1
            for previous_state in range(state):
                equivalent = True
                for snp in range(n_snps):
                    if (
                        founders[state, block, snp]
                        != founders[previous_state, block, snp]
                    ):
                        equivalent = False
                        break
                if equivalent:
                    local_class = int(mapping[block, previous_state])
                    break
            if local_class < 0:
                local_class = number_of_classes
                number_of_classes += 1
                active[block, local_class] = True
                for snp in range(n_snps):
                    pooled_founders[local_class, block, snp] = (
                        founders[state, block, snp]
                    )
            mapping[block, state] = local_class
        class_counts[block] = number_of_classes
    return mapping, class_counts, pooled_founders, active


@njit(cache=True, parallel=True)
def _pool_local_label_kernel(labels, mapping):
    n_samples, n_bins, _ = labels.shape
    pooled_labels = np.full_like(labels, -1, dtype=np.int16)
    for flat_index in prange(n_samples * n_bins):
        sample = flat_index // n_bins
        block = flat_index - sample * n_bins
        for track in range(2):
            label = int(labels[sample, block, track])
            if label >= 0:
                pooled_labels[sample, block, track] = mapping[block, label]
    return pooled_labels


@njit(cache=True)
def _unique_trajectory_state_kernel(mapping):
    """Deduplicate complete founder trajectories in first-occurrence order."""
    n_bins, n_states = mapping.shape
    unique_states = np.empty(n_states, dtype=np.int64)
    number_of_unique_states = 0
    for state in range(n_states):
        duplicate = False
        for unique_index in range(number_of_unique_states):
            representative = int(unique_states[unique_index])
            equivalent = True
            for block in range(n_bins):
                if mapping[block, state] != mapping[block, representative]:
                    equivalent = False
                    break
            if equivalent:
                duplicate = True
                break
        if not duplicate:
            unique_states[number_of_unique_states] = state
            number_of_unique_states += 1
    return unique_states[:number_of_unique_states]


@njit(cache=True, parallel=True)
def _continuation_bridge_kernel(
    mapping,
    class_counts,
    unique_trajectory_states,
    maximum_classes,
):
    n_bins = mapping.shape[0]
    destination_counts = np.zeros(
        (n_bins, maximum_classes, maximum_classes), dtype=np.int64
    )
    continuation_bridge = np.zeros(
        (n_bins, maximum_classes, maximum_classes), dtype=np.float64
    )
    for block in prange(1, n_bins):
        for unique_index in range(len(unique_trajectory_states)):
            state = int(unique_trajectory_states[unique_index])
            previous = int(mapping[block - 1, state])
            current = int(mapping[block, state])
            destination_counts[block, previous, current] += 1
        for previous in range(int(class_counts[block - 1])):
            denominator = 0
            for current in range(int(class_counts[block])):
                denominator += destination_counts[block, previous, current]
            for current in range(int(class_counts[block])):
                count = destination_counts[block, previous, current]
                if count > 0:
                    continuation_bridge[block, previous, current] = (
                        float(count) / float(denominator)
                    )
    return continuation_bridge


def _pool_local_ibs_states(
    stacked_labels: np.ndarray,
    founder_alleles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool numeric labels that are locally IBS-equivalent.

    Reconstructed founder-label numbers are not biological identities.  On
    every cache bin, founder rows with the same observed allele vector are
    collapsed into one local state before external-parent frequencies and
    transitions are estimated.  The transition model is then estimated
    between these pooled local classes, so splitting one IBS haplotype into
    duplicate numeric labels cannot manufacture ancestry switches.
    """
    labels = np.asarray(stacked_labels, dtype=np.int16)
    founders = np.asarray(founder_alleles, dtype=np.int8)
    mapping, class_counts, pooled_founders, active = (
        _local_ibs_class_kernel(founders)
    )
    maximum_classes = int(np.max(class_counts))
    pooled_founders = np.ascontiguousarray(
        pooled_founders[:maximum_classes]
    )
    active = np.ascontiguousarray(active[:, :maximum_classes])
    pooled_labels = _pool_local_label_kernel(labels, mapping)

    # Class numbers are local to a bin.  Preserve physical ancestry continuity
    # by following each distinct chromosome-wide reconstructed founder path
    # into its next-bin pooled class.  Globally duplicate trajectories receive
    # one representative, so duplicating an arbitrary numeric label cannot
    # alter merge/split probabilities.
    unique_trajectory_states = _unique_trajectory_state_kernel(mapping)
    trajectory_classes = np.ascontiguousarray(
        mapping[:, unique_trajectory_states].T, dtype=np.int16
    )
    continuation_bridge = _continuation_bridge_kernel(
        mapping,
        class_counts,
        unique_trajectory_states,
        maximum_classes,
    )
    return (
        np.ascontiguousarray(pooled_labels),
        pooled_founders,
        active,
        np.ascontiguousarray(continuation_bridge),
        trajectory_classes,
    )


@njit(cache=True, parallel=True)
def _ancestry_junction_count_kernel(pooled_labels, trajectory_classes):
    """Minimum chromosome-wide founder switches for each diploid painting.

    The hidden state is an unordered pair drawn from deduplicated whole-
    chromosome founder trajectories (the two entries may be equal). Local IBS
    equivalence is an
    emission ambiguity, not permission to splice two different trajectories
    together for free. A transition costs zero for the same unordered pair,
    one when the pairs share one trajectory, and two otherwise. Per-founder
    and global minima reduce each dynamic-programming update from quartic to
    quadratic in the number of unique trajectories.

    Missing painted labels are uninformative. The second returned vector is
    the number of observed haplotype-bin labels and is carried into the depth
    model so incomplete paintings cannot masquerade as shallow ancestry merely
    because fewer switches were detectable.
    """
    n_samples, n_bins, _ = pooled_labels.shape
    n_trajectories = trajectory_classes.shape[0]
    n_pairs = n_trajectories * (n_trajectories + 1) // 2
    pair_first = np.empty(n_pairs, dtype=np.int64)
    pair_second = np.empty(n_pairs, dtype=np.int64)
    pair_index = 0
    for first in range(n_trajectories):
        for second in range(first, n_trajectories):
            pair_first[pair_index] = first
            pair_second[pair_index] = second
            pair_index += 1

    output = np.zeros(n_samples, dtype=np.int64)
    callable_bins = np.zeros(n_samples, dtype=np.int64)
    unreachable = 1 << 30
    for sample in prange(n_samples):
        previous_cost = np.zeros(n_pairs, dtype=np.int64)
        current_cost = np.empty(n_pairs, dtype=np.int64)
        per_trajectory_minimum = np.empty(n_trajectories, dtype=np.int64)
        observed_count = 0
        for block in range(n_bins):
            observed0 = int(pooled_labels[sample, block, 0])
            observed1 = int(pooled_labels[sample, block, 1])
            observed_count += int(observed0 >= 0) + int(observed1 >= 0)

            if block > 0:
                global_minimum = unreachable
                for trajectory in range(n_trajectories):
                    per_trajectory_minimum[trajectory] = unreachable
                for state in range(n_pairs):
                    value = int(previous_cost[state])
                    if value < global_minimum:
                        global_minimum = value
                    first = int(pair_first[state])
                    second = int(pair_second[state])
                    if value < per_trajectory_minimum[first]:
                        per_trajectory_minimum[first] = value
                    if value < per_trajectory_minimum[second]:
                        per_trajectory_minimum[second] = value

            for state in range(n_pairs):
                first = int(pair_first[state])
                second = int(pair_second[state])
                first_class = int(trajectory_classes[first, block])
                second_class = int(trajectory_classes[second, block])
                if observed0 < 0 and observed1 < 0:
                    compatible = True
                elif observed0 < 0:
                    compatible = (
                        first_class == observed1 or second_class == observed1
                    )
                elif observed1 < 0:
                    compatible = (
                        first_class == observed0 or second_class == observed0
                    )
                else:
                    compatible = (
                        first_class == observed0 and second_class == observed1
                    ) or (
                        first_class == observed1 and second_class == observed0
                    )
                if not compatible:
                    current_cost[state] = unreachable
                elif block == 0:
                    current_cost[state] = 0
                else:
                    shared = min(
                        int(per_trajectory_minimum[first]),
                        int(per_trajectory_minimum[second]),
                    ) + 1
                    unrelated = int(global_minimum) + 2
                    current_cost[state] = min(
                        int(previous_cost[state]), shared, unrelated
                    )
            swap = previous_cost
            previous_cost = current_cost
            current_cost = swap
        output[sample] = int(np.min(previous_cost))
        callable_bins[sample] = observed_count
    return output, callable_bins


@njit(cache=True, parallel=True)
def _background_totals(
    stacked_alleles,
    stacked_labels,
    n_states,
):
    n_samples, n_bins, _, n_snps = stacked_alleles.shape
    state_counts = np.zeros((n_bins, n_states), dtype=np.int64)
    alt_counts = np.zeros((n_bins, n_snps), dtype=np.int64)
    allele_counts = np.zeros((n_bins, n_snps), dtype=np.int64)
    transition_counts = np.zeros(
        (n_bins, n_states, n_states), dtype=np.int64
    )
    for block in prange(n_bins):
        for sample in range(n_samples):
            for track in range(2):
                label = int(stacked_labels[sample, block, track])
                if label >= 0:
                    state_counts[block, label] += 1
                for snp in range(n_snps):
                    allele = int(stacked_alleles[sample, block, track, snp])
                    if allele >= 0:
                        allele_counts[block, snp] += 1
                        alt_counts[block, snp] += allele
                if block > 0:
                    previous = int(stacked_labels[sample, block - 1, track])
                    if previous >= 0 and label >= 0:
                        transition_counts[block, previous, label] += 1
    return state_counts, alt_counts, allele_counts, transition_counts


@njit(cache=True, parallel=True)
def _prepare_leave_child_out_background(
    stacked_alleles,
    stacked_labels,
    selected_markers_per_bin,
    active_states,
    continuation_bridge,
    switch_probability,
    state_counts,
    alt_counts,
    allele_counts,
    transition_counts,
    external_state_pseudocount,
    external_transition_pseudocount,
    markers_per_information_block,
    effective_markers_per_information_block,
):
    """Create child-left-out pooled external-parent distributions.

    Transition rows use empirical pooled local-IBS transitions from all other
    sampled homologs.  Their Dirichlet target follows the same reconstructed
    ancestry label across adjacent local IBS classes with probability
    ``1 - theta`` and redraws from next-bin occupancy with probability
    ``theta``.  Shrinking directly to occupancy at every bin would create a
    bin-resolution-dependent ancestry switch process and systematically
    penalize each unobserved parental gamete.
    """
    n_samples, n_bins, _, n_snps = stacked_alleles.shape
    n_states = state_counts.shape[1]
    state_probability = np.zeros(
        (n_samples, n_bins, n_states), dtype=np.float64
    )
    external_transition_probability = np.zeros(
        (n_samples, n_bins, n_states, n_states), dtype=np.float64
    )
    background_alt_probability = np.empty(
        (n_samples, n_bins, n_snps), dtype=np.float64
    )

    information_block = np.empty(n_bins, dtype=np.int64)
    group = 0
    markers_in_group = 0
    for block in range(n_bins):
        if block > 0 and markers_in_group >= markers_per_information_block:
            group += 1
            markers_in_group = 0
        information_block[block] = group
        markers_in_group += max(int(selected_markers_per_bin[block]), 1)
    n_information_blocks = group + 1
    information_exponent = np.zeros((n_samples, n_bins), dtype=np.float64)

    for child in prange(n_samples):
        valid_sites_per_group = np.zeros(
            n_information_blocks, dtype=np.int64
        )
        for block in range(n_bins):
            count_sum = 0
            active_count = 0
            for state in range(n_states):
                if active_states[block, state]:
                    active_count += 1
                count = int(state_counts[block, state])
                for track in range(2):
                    count -= int(
                        int(stacked_labels[child, block, track]) == state
                    )
                count = max(count, 0)
                state_probability[child, block, state] = count
                count_sum += count
            active_count = max(active_count, 1)
            denominator = count_sum + external_state_pseudocount
            for state in range(n_states):
                if active_states[block, state]:
                    state_probability[child, block, state] = (
                        state_probability[child, block, state]
                        + external_state_pseudocount / active_count
                    ) / denominator
                else:
                    state_probability[child, block, state] = 0.0

            for snp in range(n_snps):
                count = int(allele_counts[block, snp])
                alts = int(alt_counts[block, snp])
                for track in range(2):
                    allele = int(stacked_alleles[child, block, track, snp])
                    if allele >= 0:
                        count -= 1
                        alts -= allele
                background_alt_probability[child, block, snp] = (
                    alts + 0.5
                ) / max(count + 1.0, 1.0)
                if (
                    int(stacked_alleles[child, block, 0, snp]) >= 0
                    or int(stacked_alleles[child, block, 1, snp]) >= 0
                ):
                    valid_sites_per_group[information_block[block]] += 1

            if block > 0:
                for previous in range(n_states):
                    row_total = 0
                    for current in range(n_states):
                        count = int(
                            transition_counts[block, previous, current]
                        )
                        for track in range(2):
                            child_previous = int(
                                stacked_labels[child, block - 1, track]
                            )
                            child_current = int(
                                stacked_labels[child, block, track]
                            )
                            if (
                                child_previous == previous
                                and child_current == current
                            ):
                                count -= 1
                        count = max(count, 0)
                        external_transition_probability[
                            child, block, previous, current
                        ] = count
                        row_total += count
                    transition_denominator = (
                        row_total + external_transition_pseudocount
                    )
                    theta = switch_probability[block]
                    for current in range(n_states):
                        linked_homolog_transition = (
                            external_transition_probability[
                                child, block, previous, current
                            ]
                            + external_transition_pseudocount
                            * continuation_bridge[block, previous, current]
                        ) / transition_denominator
                        external_transition_probability[
                            child, block, previous, current
                        ] = (
                            (1.0 - theta) * linked_homolog_transition
                            + theta
                            * state_probability[child, block, current]
                        )

        for block in range(n_bins):
            valid_sites = valid_sites_per_group[information_block[block]]
            if valid_sites > 0:
                information_exponent[child, block] = (
                    min(
                        effective_markers_per_information_block,
                        float(valid_sites),
                    )
                    / float(valid_sites)
                )

    return (
        state_probability,
        external_transition_probability,
        background_alt_probability,
        information_exponent,
    )


@njit(cache=True, inline="always")
def _external_transition_vector(source, destination, transition):
    """Apply one row-normalized pooled external-state transition."""
    n_states = source.shape[0]
    for current in range(n_states):
        value = 0.0
        for previous in range(n_states):
            value += source[previous] * transition[previous, current]
        destination[current] = value


@njit(cache=True, inline="always")
def _log_emission_hard_sources(
    child_alleles,
    first_source,
    second_source,
    background_alt,
    orientation,
    mismatch_probability,
    exponent,
):
    n_snps = child_alleles.shape[1]
    value = 0.0
    first_child_track = orientation
    second_child_track = 1 - orientation
    for snp in range(n_snps):
        first_alt = _source_alt_probability(
            int(first_source[snp]), background_alt[snp]
        )
        second_alt = _source_alt_probability(
            int(second_source[snp]), background_alt[snp]
        )
        value += _observed_allele_log_probability(
            int(child_alleles[first_child_track, snp]),
            first_alt,
            mismatch_probability,
        )
        value += _observed_allele_log_probability(
            int(child_alleles[second_child_track, snp]),
            second_alt,
            mismatch_probability,
        )
    return exponent * value


@njit(cache=True, parallel=True)
def _score_zero_parent_forward_kernel(
    stacked_alleles,
    stacked_hom_mask,
    founder_alleles,
    state_probability,
    external_transition_probability,
    background_alt_probability,
    information_exponent,
    phase_switch_probability,
    mismatch_probability,
):
    n_samples, n_bins, _, _ = stacked_alleles.shape
    n_states = founder_alleles.shape[0]
    output = np.empty(n_samples, dtype=np.float64)

    for child in prange(n_samples):
        forward = np.empty((2, n_states, n_states), dtype=np.float64)
        work1 = np.empty_like(forward)
        work2 = np.empty_like(forward)
        source = np.empty(n_states, dtype=np.float64)
        destination = np.empty(n_states, dtype=np.float64)
        total_log_likelihood = 0.0
        for orientation in range(2):
            for first in range(n_states):
                for second in range(n_states):
                    forward[orientation, first, second] = (
                        0.5
                        * state_probability[child, 0, first]
                        * state_probability[child, 0, second]
                    )

        for block in range(n_bins):
            if block > 0:
                transition = external_transition_probability[child, block]
                for orientation in range(2):
                    for second in range(n_states):
                        for first in range(n_states):
                            source[first] = forward[orientation, first, second]
                        _external_transition_vector(
                            source, destination, transition
                        )
                        for first in range(n_states):
                            work1[orientation, first, second] = destination[first]
                    for first in range(n_states):
                        for second in range(n_states):
                            source[second] = work1[orientation, first, second]
                        _external_transition_vector(
                            source, destination, transition
                        )
                        for second in range(n_states):
                            work2[orientation, first, second] = destination[second]
                phase_rho = _phase_switch_probability(
                    bool(stacked_hom_mask[child, block - 1]),
                    bool(stacked_hom_mask[child, block]),
                    phase_switch_probability,
                )
                for first in range(n_states):
                    for second in range(n_states):
                        value0 = work2[0, first, second]
                        value1 = work2[1, first, second]
                        forward[0, first, second] = (
                            (1.0 - phase_rho) * value0 + phase_rho * value1
                        )
                        forward[1, first, second] = (
                            (1.0 - phase_rho) * value1 + phase_rho * value0
                        )

            maximum = -math.inf
            for orientation in range(2):
                for first in range(n_states):
                    for second in range(n_states):
                        log_emission = _log_emission_hard_sources(
                            stacked_alleles[child, block],
                            founder_alleles[first, block],
                            founder_alleles[second, block],
                            background_alt_probability[child, block],
                            orientation,
                            mismatch_probability,
                            information_exponent[child, block],
                        )
                        work1[orientation, first, second] = log_emission
                        maximum = max(maximum, log_emission)
            scale = 0.0
            for orientation in range(2):
                for first in range(n_states):
                    for second in range(n_states):
                        forward[orientation, first, second] *= math.exp(
                            work1[orientation, first, second] - maximum
                        )
                        scale += forward[orientation, first, second]
            scale = max(scale, _TINY)
            total_log_likelihood += maximum + math.log(scale)
            for orientation in range(2):
                for first in range(n_states):
                    for second in range(n_states):
                        forward[orientation, first, second] /= scale
        output[child] = total_log_likelihood
    return output


@njit(cache=True, parallel=True)
def _score_one_parent_forward_kernel(
    stacked_alleles,
    stacked_hom_mask,
    founder_alleles,
    state_probability,
    external_transition_probability,
    background_alt_probability,
    information_exponent,
    switch_probability,
    phase_switch_probability,
    mismatch_probability,
):
    n_samples, n_bins, _, _ = stacked_alleles.shape
    n_states = founder_alleles.shape[0]
    output = np.full((n_samples, n_samples), -math.inf, dtype=np.float64)

    for flat_index in prange(n_samples * n_samples):
        child = flat_index // n_samples
        parent = flat_index - child * n_samples
        if parent == child:
            continue
        forward = np.empty((2, 2, n_states), dtype=np.float64)
        work1 = np.empty_like(forward)
        work2 = np.empty_like(forward)
        source = np.empty(n_states, dtype=np.float64)
        destination = np.empty(n_states, dtype=np.float64)
        total_log_likelihood = 0.0
        for orientation in range(2):
            for track in range(2):
                for state in range(n_states):
                    forward[orientation, track, state] = (
                        0.25 * state_probability[child, 0, state]
                    )

        for block in range(n_bins):
            if block > 0:
                transition = external_transition_probability[child, block]
                for orientation in range(2):
                    for track in range(2):
                        for state in range(n_states):
                            source[state] = forward[orientation, track, state]
                        _external_transition_vector(
                            source, destination, transition
                        )
                        for state in range(n_states):
                            work1[orientation, track, state] = destination[state]
                theta = _phase_switch_probability(
                    bool(stacked_hom_mask[parent, block - 1]),
                    bool(stacked_hom_mask[parent, block]),
                    switch_probability[block],
                )
                for orientation in range(2):
                    for state in range(n_states):
                        value0 = work1[orientation, 0, state]
                        value1 = work1[orientation, 1, state]
                        work2[orientation, 0, state] = (
                            (1.0 - theta) * value0 + theta * value1
                        )
                        work2[orientation, 1, state] = (
                            (1.0 - theta) * value1 + theta * value0
                        )
                phase_rho = _phase_switch_probability(
                    bool(stacked_hom_mask[child, block - 1]),
                    bool(stacked_hom_mask[child, block]),
                    phase_switch_probability,
                )
                for track in range(2):
                    for state in range(n_states):
                        value0 = work2[0, track, state]
                        value1 = work2[1, track, state]
                        forward[0, track, state] = (
                            (1.0 - phase_rho) * value0 + phase_rho * value1
                        )
                        forward[1, track, state] = (
                            (1.0 - phase_rho) * value1 + phase_rho * value0
                        )

            maximum = -math.inf
            for orientation in range(2):
                for track in range(2):
                    for state in range(n_states):
                        log_emission = _log_emission_hard_sources(
                            stacked_alleles[child, block],
                            stacked_alleles[parent, block, track],
                            founder_alleles[state, block],
                            background_alt_probability[child, block],
                            orientation,
                            mismatch_probability,
                            information_exponent[child, block],
                        )
                        work1[orientation, track, state] = log_emission
                        maximum = max(maximum, log_emission)
            scale = 0.0
            for orientation in range(2):
                for track in range(2):
                    for state in range(n_states):
                        forward[orientation, track, state] *= math.exp(
                            work1[orientation, track, state] - maximum
                        )
                        scale += forward[orientation, track, state]
            scale = max(scale, _TINY)
            total_log_likelihood += maximum + math.log(scale)
            for orientation in range(2):
                for track in range(2):
                    for state in range(n_states):
                        forward[orientation, track, state] /= scale
        output[child, parent] = total_log_likelihood
    return output


@njit(cache=True, parallel=True)
def _score_two_parent_forward_kernel(
    stacked_alleles,
    stacked_hom_mask,
    trios,
    background_alt_probability,
    information_exponent,
    switch_probability,
    phase_switch_probability,
    mismatch_probability,
):
    n_samples, n_bins, _, _ = stacked_alleles.shape
    output = np.full(len(trios), -math.inf, dtype=np.float64)
    for row in prange(len(trios)):
        child = int(trios[row, 0])
        parent1 = int(trios[row, 1])
        parent2 = int(trios[row, 2])
        if (
            child < 0
            or child >= n_samples
            or parent1 < 0
            or parent1 >= n_samples
            or parent2 < 0
            or parent2 >= n_samples
            or parent1 == child
            or parent2 == child
            or parent1 == parent2
        ):
            continue
        forward = np.full((2, 2, 2), 0.125, dtype=np.float64)
        work1 = np.empty_like(forward)
        work2 = np.empty_like(forward)
        total_log_likelihood = 0.0
        for block in range(n_bins):
            if block > 0:
                theta1 = _phase_switch_probability(
                    bool(stacked_hom_mask[parent1, block - 1]),
                    bool(stacked_hom_mask[parent1, block]),
                    switch_probability[block],
                )
                theta2 = _phase_switch_probability(
                    bool(stacked_hom_mask[parent2, block - 1]),
                    bool(stacked_hom_mask[parent2, block]),
                    switch_probability[block],
                )
                stay1 = 1.0 - theta1
                stay2 = 1.0 - theta2
                for orientation in range(2):
                    for second in range(2):
                        value0 = forward[orientation, 0, second]
                        value1 = forward[orientation, 1, second]
                        work1[orientation, 0, second] = (
                            stay1 * value0 + theta1 * value1
                        )
                        work1[orientation, 1, second] = (
                            stay1 * value1 + theta1 * value0
                        )
                    for first in range(2):
                        value0 = work1[orientation, first, 0]
                        value1 = work1[orientation, first, 1]
                        work2[orientation, first, 0] = (
                            stay2 * value0 + theta2 * value1
                        )
                        work2[orientation, first, 1] = (
                            stay2 * value1 + theta2 * value0
                        )
                phase_rho = _phase_switch_probability(
                    bool(stacked_hom_mask[child, block - 1]),
                    bool(stacked_hom_mask[child, block]),
                    phase_switch_probability,
                )
                for first in range(2):
                    for second in range(2):
                        value0 = work2[0, first, second]
                        value1 = work2[1, first, second]
                        forward[0, first, second] = (
                            (1.0 - phase_rho) * value0 + phase_rho * value1
                        )
                        forward[1, first, second] = (
                            (1.0 - phase_rho) * value1 + phase_rho * value0
                        )

            maximum = -math.inf
            for orientation in range(2):
                for first in range(2):
                    for second in range(2):
                        log_emission = _log_emission_hard_sources(
                            stacked_alleles[child, block],
                            stacked_alleles[parent1, block, first],
                            stacked_alleles[parent2, block, second],
                            background_alt_probability[child, block],
                            orientation,
                            mismatch_probability,
                            information_exponent[child, block],
                        )
                        work1[orientation, first, second] = log_emission
                        maximum = max(maximum, log_emission)
            scale = 0.0
            for orientation in range(2):
                for first in range(2):
                    for second in range(2):
                        forward[orientation, first, second] *= math.exp(
                            work1[orientation, first, second] - maximum
                        )
                        scale += forward[orientation, first, second]
            scale = max(scale, _TINY)
            total_log_likelihood += maximum + math.log(scale)
            for orientation in range(2):
                for first in range(2):
                    for second in range(2):
                        forward[orientation, first, second] /= scale
        output[row] = total_log_likelihood
    return output


@njit(cache=True, inline="always")
def _single_source_log_emission(
    child_track_alleles,
    source_alleles,
    background_alt,
    mismatch_probability,
    hard_match_log_probability,
    hard_mismatch_log_probability,
):
    """Log emission contributed by one parental source.

    Hard reconstructed alleles have only two possible probabilities, so their
    logarithms are computed once per kernel rather than once per
    source/state/SNP/candidate. Missing source alleles retain the exact
    background-mixture calculation used by
    :func:`_observed_allele_log_probability`.
    """
    value = 0.0
    for snp in range(child_track_alleles.shape[0]):
        observed = int(child_track_alleles[snp])
        if observed < 0:
            continue
        source = int(source_alleles[snp])
        if source >= 0:
            if source == observed:
                value += hard_match_log_probability
            else:
                value += hard_mismatch_log_probability
        else:
            source_alt = background_alt[snp]
            if observed == 1:
                probability = (
                    (1.0 - mismatch_probability) * source_alt
                    + mismatch_probability * (1.0 - source_alt)
                )
            else:
                probability = (
                    (1.0 - mismatch_probability) * (1.0 - source_alt)
                    + mismatch_probability * source_alt
                )
            value += math.log(max(probability, _TINY))
    return value


@njit(cache=True, parallel=True)
def _effective_switch_probability_kernel(
    stacked_hom_mask,
    switch_probability,
    phase_switch_probability,
):
    """Precompute transmission and child-orientation switch probabilities."""
    n_samples, n_bins = stacked_hom_mask.shape
    transmission = np.empty((n_samples, n_bins), dtype=np.float64)
    phase = np.empty((n_samples, n_bins), dtype=np.float64)
    for sample in prange(n_samples):
        transmission[sample, 0] = 0.0
        phase[sample, 0] = 0.0
        for block in range(1, n_bins):
            reset = (
                bool(stacked_hom_mask[sample, block - 1])
                or bool(stacked_hom_mask[sample, block])
            )
            if reset:
                transmission[sample, block] = 0.5
                phase[sample, block] = 0.5
            else:
                transmission[sample, block] = switch_probability[block]
                phase[sample, block] = phase_switch_probability
    return transmission, phase


@njit(cache=True, parallel=True)
def _score_one_parent_forward_kernel_grouped(
    stacked_alleles,
    founder_alleles,
    state_probability,
    external_transition_probability,
    background_alt_probability,
    information_exponent,
    effective_transmission_probability,
    effective_phase_probability,
    mismatch_probability,
    child_start,
    child_end,
):
    """Score M1 while reusing emissions across every parent of one child.

    This is the same normalized forward model as
    :func:`_score_one_parent_forward_kernel`, reorganised from
    ``(child,parent)->blocks`` to ``child->blocks->parents``. A known-parent
    source emission is computed once per block and reused for every external
    founder state; an external-founder emission is computed once and reused
    for every candidate parent.
    """
    n_samples, n_bins, _, _ = stacked_alleles.shape
    n_states = founder_alleles.shape[0]
    n_children = child_end - child_start
    output = np.full((n_children, n_samples), -math.inf, dtype=np.float64)
    hard_match_log_probability = math.log(1.0 - mismatch_probability)
    hard_mismatch_log_probability = math.log(mismatch_probability)

    for local_child in prange(n_children):
        child = child_start + local_child
        forward = np.empty(
            (n_samples, 2, 2, n_states), dtype=np.float64
        )
        work1 = np.empty_like(forward)
        work2 = np.empty_like(forward)
        totals = np.zeros(n_samples, dtype=np.float64)
        known_emission = np.empty((n_samples, 2, 2), dtype=np.float64)
        external_emission = np.empty((n_states, 2), dtype=np.float64)

        for parent in range(n_samples):
            if parent == child:
                continue
            for orientation in range(2):
                for track in range(2):
                    for state in range(n_states):
                        forward[parent, orientation, track, state] = (
                            0.25 * state_probability[child, 0, state]
                        )

        for block in range(n_bins):
            if block > 0:
                transition = external_transition_probability[child, block]
                phase_rho = effective_phase_probability[child, block]
                phase_stay = 1.0 - phase_rho
                for parent in range(n_samples):
                    if parent == child:
                        continue
                    for orientation in range(2):
                        for track in range(2):
                            for current in range(n_states):
                                value = 0.0
                                for previous in range(n_states):
                                    value += (
                                        forward[
                                            parent,
                                            orientation,
                                            track,
                                            previous,
                                        ]
                                        * transition[previous, current]
                                    )
                                work1[
                                    parent, orientation, track, current
                                ] = value
                    theta = effective_transmission_probability[parent, block]
                    stay = 1.0 - theta
                    for orientation in range(2):
                        for state in range(n_states):
                            value0 = work1[parent, orientation, 0, state]
                            value1 = work1[parent, orientation, 1, state]
                            work2[parent, orientation, 0, state] = (
                                stay * value0 + theta * value1
                            )
                            work2[parent, orientation, 1, state] = (
                                stay * value1 + theta * value0
                            )
                    for track in range(2):
                        for state in range(n_states):
                            value0 = work2[parent, 0, track, state]
                            value1 = work2[parent, 1, track, state]
                            forward[parent, 0, track, state] = (
                                phase_stay * value0 + phase_rho * value1
                            )
                            forward[parent, 1, track, state] = (
                                phase_stay * value1 + phase_rho * value0
                            )

            child_block = stacked_alleles[child, block]
            background_block = background_alt_probability[child, block]
            for parent in range(n_samples):
                for track in range(2):
                    for child_track in range(2):
                        known_emission[parent, track, child_track] = (
                            _single_source_log_emission(
                                child_block[child_track],
                                stacked_alleles[parent, block, track],
                                background_block,
                                mismatch_probability,
                                hard_match_log_probability,
                                hard_mismatch_log_probability,
                            )
                        )
            for state in range(n_states):
                for child_track in range(2):
                    external_emission[state, child_track] = (
                        _single_source_log_emission(
                            child_block[child_track],
                            founder_alleles[state, block],
                            background_block,
                            mismatch_probability,
                            hard_match_log_probability,
                            hard_mismatch_log_probability,
                        )
                    )

            exponent = information_exponent[child, block]
            for parent in range(n_samples):
                if parent == child:
                    continue
                maximum = -math.inf
                for orientation in range(2):
                    for track in range(2):
                        for state in range(n_states):
                            log_emission = exponent * (
                                known_emission[parent, track, orientation]
                                + external_emission[state, 1 - orientation]
                            )
                            work1[
                                parent, orientation, track, state
                            ] = log_emission
                            maximum = max(maximum, log_emission)
                scale = 0.0
                for orientation in range(2):
                    for track in range(2):
                        for state in range(n_states):
                            forward[
                                parent, orientation, track, state
                            ] *= math.exp(
                                work1[
                                    parent, orientation, track, state
                                ] - maximum
                            )
                            scale += forward[
                                parent, orientation, track, state
                            ]
                scale = max(scale, _TINY)
                totals[parent] += maximum + math.log(scale)
                inverse_scale = 1.0 / scale
                for orientation in range(2):
                    for track in range(2):
                        for state in range(n_states):
                            forward[
                                parent, orientation, track, state
                            ] *= inverse_scale

        for parent in range(n_samples):
            if parent != child:
                output[local_child, parent] = totals[parent]
    return output


@njit(cache=True, parallel=True)
def _score_two_parent_forward_kernel_grouped(
    stacked_alleles,
    sorted_trios,
    child_row_starts,
    background_alt_probability,
    information_exponent,
    effective_transmission_probability,
    effective_phase_probability,
    mismatch_probability,
    child_start,
    child_end,
):
    """Score M2 by child while reusing each parent-source emission."""
    n_samples, n_bins, _, _ = stacked_alleles.shape
    output_row_start = int(child_row_starts[child_start])
    output_row_end = int(child_row_starts[child_end])
    output = np.full(
        output_row_end - output_row_start, -math.inf, dtype=np.float64
    )
    hard_match_log_probability = math.log(1.0 - mismatch_probability)
    hard_mismatch_log_probability = math.log(mismatch_probability)

    for local_child in prange(child_end - child_start):
        child = child_start + local_child
        start = int(child_row_starts[child])
        end = int(child_row_starts[child + 1])
        n_pairs = end - start
        if n_pairs <= 0:
            continue
        forward = np.full((n_pairs, 2, 2, 2), 0.125, dtype=np.float64)
        work1 = np.empty_like(forward)
        work2 = np.empty_like(forward)
        totals = np.zeros(n_pairs, dtype=np.float64)
        source_emission = np.empty(
            (n_samples, 2, 2), dtype=np.float64
        )

        for block in range(n_bins):
            if block > 0:
                phase_rho = effective_phase_probability[child, block]
                phase_stay = 1.0 - phase_rho
                for local_row in range(n_pairs):
                    row = start + local_row
                    parent1 = int(sorted_trios[row, 1])
                    parent2 = int(sorted_trios[row, 2])
                    theta1 = effective_transmission_probability[
                        parent1, block
                    ]
                    theta2 = effective_transmission_probability[
                        parent2, block
                    ]
                    stay1 = 1.0 - theta1
                    stay2 = 1.0 - theta2
                    for orientation in range(2):
                        for second in range(2):
                            value0 = forward[
                                local_row, orientation, 0, second
                            ]
                            value1 = forward[
                                local_row, orientation, 1, second
                            ]
                            work1[
                                local_row, orientation, 0, second
                            ] = stay1 * value0 + theta1 * value1
                            work1[
                                local_row, orientation, 1, second
                            ] = stay1 * value1 + theta1 * value0
                        for first in range(2):
                            value0 = work1[
                                local_row, orientation, first, 0
                            ]
                            value1 = work1[
                                local_row, orientation, first, 1
                            ]
                            work2[
                                local_row, orientation, first, 0
                            ] = stay2 * value0 + theta2 * value1
                            work2[
                                local_row, orientation, first, 1
                            ] = stay2 * value1 + theta2 * value0
                    for first in range(2):
                        for second in range(2):
                            value0 = work2[local_row, 0, first, second]
                            value1 = work2[local_row, 1, first, second]
                            forward[local_row, 0, first, second] = (
                                phase_stay * value0 + phase_rho * value1
                            )
                            forward[local_row, 1, first, second] = (
                                phase_stay * value1 + phase_rho * value0
                            )

            child_block = stacked_alleles[child, block]
            background_block = background_alt_probability[child, block]
            for parent in range(n_samples):
                for track in range(2):
                    for child_track in range(2):
                        source_emission[parent, track, child_track] = (
                            _single_source_log_emission(
                                child_block[child_track],
                                stacked_alleles[parent, block, track],
                                background_block,
                                mismatch_probability,
                                hard_match_log_probability,
                                hard_mismatch_log_probability,
                            )
                        )

            exponent = information_exponent[child, block]
            for local_row in range(n_pairs):
                row = start + local_row
                parent1 = int(sorted_trios[row, 1])
                parent2 = int(sorted_trios[row, 2])
                maximum = -math.inf
                for orientation in range(2):
                    for first in range(2):
                        for second in range(2):
                            log_emission = exponent * (
                                source_emission[
                                    parent1, first, orientation
                                ]
                                + source_emission[
                                    parent2, second, 1 - orientation
                                ]
                            )
                            work1[
                                local_row, orientation, first, second
                            ] = log_emission
                            maximum = max(maximum, log_emission)
                scale = 0.0
                for orientation in range(2):
                    for first in range(2):
                        for second in range(2):
                            forward[
                                local_row, orientation, first, second
                            ] *= math.exp(
                                work1[
                                    local_row, orientation, first, second
                                ] - maximum
                            )
                            scale += forward[
                                local_row, orientation, first, second
                            ]
                scale = max(scale, _TINY)
                totals[local_row] += maximum + math.log(scale)
                inverse_scale = 1.0 / scale
                for orientation in range(2):
                    for first in range(2):
                        for second in range(2):
                            forward[
                                local_row, orientation, first, second
                            ] *= inverse_scale

        for local_row in range(n_pairs):
            output[start + local_row - output_row_start] = totals[local_row]
    return output


def score_parent_state_hmms(
    stacked_alleles,
    stacked_labels,
    stacked_hom_mask,
    founder_alleles,
    selected_markers_per_bin,
    switch_probability,
    trios,
    *,
    mismatch_probability=0.01,
    phase_switch_probability=0.01,
    markers_per_information_block=100,
    effective_markers_per_information_block=1.0,
    external_state_pseudocount=1.0,
    external_transition_pseudocount=20.0,
    _dynamic_rebalance=False,
    _dynamic_child_chunk_floor=32,
    _dynamic_child_chunk_scale=4,
):
    """Score M0, M1 and M2 with one comparable normalized forward model."""
    raw_alleles = np.asarray(stacked_alleles)
    if raw_alleles.ndim == 3:
        raw_alleles = raw_alleles[..., None]
    if (
        raw_alleles.ndim != 4
        or min(raw_alleles.shape) < 1
        or np.any(~np.isin(raw_alleles, (-1, 0, 1)))
    ):
        raise SmartEvidenceError(
            "stacked_alleles must be non-empty hard alleles in {-1, 0, 1}"
        )
    alleles = np.ascontiguousarray(raw_alleles, dtype=np.int8)

    raw_founders = np.asarray(founder_alleles)
    if raw_founders.ndim == 2:
        raw_founders = raw_founders[..., None]
    if (
        raw_founders.ndim != 3
        or min(raw_founders.shape) < 1
        or np.any(~np.isin(raw_founders, (-1, 0, 1)))
    ):
        raise SmartEvidenceError(
            "founder_alleles must be non-empty hard alleles in {-1, 0, 1}"
        )
    founders = np.ascontiguousarray(raw_founders, dtype=np.int8)
    if founders.shape[1:] != (alleles.shape[1], alleles.shape[3]):
        raise SmartEvidenceError(
            "founder alleles must have shape (states, bins, SNPs)"
        )

    raw_labels = np.asarray(stacked_labels)
    if (
        raw_labels.shape != alleles.shape[:3]
        or np.any(~np.isfinite(raw_labels))
        or np.any(raw_labels != np.floor(raw_labels))
        or np.any(raw_labels < -1)
        or np.any(raw_labels >= founders.shape[0])
    ):
        raise SmartEvidenceError(
            "stacked_labels must have shape (samples, bins, 2) and values "
            "in {-1, 0, ..., states-1}"
        )
    labels = np.ascontiguousarray(raw_labels, dtype=np.int16)
    hom = np.ascontiguousarray(stacked_hom_mask, dtype=np.bool_)
    if hom.shape != alleles.shape[:2]:
        raise SmartEvidenceError(
            "stacked_hom_mask must have shape (samples, bins)"
        )

    raw_marker_counts = np.asarray(selected_markers_per_bin)
    if (
        raw_marker_counts.shape != (alleles.shape[1],)
        or np.any(~np.isfinite(raw_marker_counts))
        or np.any(raw_marker_counts != np.floor(raw_marker_counts))
        or np.any(raw_marker_counts < 0)
        or np.any(raw_marker_counts > alleles.shape[3])
        or np.sum(raw_marker_counts) < 1
    ):
        raise SmartEvidenceError(
            "selected_markers_per_bin must be integer counts between zero and "
            "the SNP-slot count, with at least one selected marker"
        )
    marker_counts = np.ascontiguousarray(
        raw_marker_counts, dtype=np.int64
    )
    theta = np.ascontiguousarray(switch_probability, dtype=np.float64)
    if (
        theta.shape != (alleles.shape[1],)
        or np.any(~np.isfinite(theta))
        or np.any((theta < 0.0) | (theta > 0.5))
    ):
        raise SmartEvidenceError(
            "switch probabilities must be finite with shape (bins,) and lie "
            "in [0, 0.5]"
        )

    raw_trios = np.asarray(trios)
    if (
        raw_trios.ndim != 2
        or raw_trios.shape[1] != 3
        or np.any(~np.isfinite(raw_trios))
        or np.any(raw_trios != np.floor(raw_trios))
    ):
        raise SmartEvidenceError("trios must be an integer array of shape (rows, 3)")
    trio_array = np.ascontiguousarray(raw_trios, dtype=np.int64)
    if len(trio_array):
        if np.any(trio_array < 0) or np.any(trio_array >= alleles.shape[0]):
            raise SmartEvidenceError("trio index outside the sample array")
        if (
            np.any(trio_array[:, 0] == trio_array[:, 1])
            or np.any(trio_array[:, 0] == trio_array[:, 2])
            or np.any(trio_array[:, 1] == trio_array[:, 2])
        ):
            raise SmartEvidenceError(
                "trios cannot contain self-parents or duplicate parents"
            )

    if (
        not np.isfinite(mismatch_probability)
        or not 0.0 < mismatch_probability < 0.5
    ):
        raise SmartEvidenceError("mismatch_probability must lie in (0, 0.5)")
    if (
        not np.isfinite(phase_switch_probability)
        or not 0.0 <= phase_switch_probability <= 0.5
    ):
        raise SmartEvidenceError("phase_switch_probability must lie in [0, 0.5]")
    try:
        marker_block_size = float(markers_per_information_block)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SmartEvidenceError(
            "markers_per_information_block must be a positive integer"
        ) from exc
    if (
        isinstance(markers_per_information_block, (bool, np.bool_))
        or not np.isfinite(marker_block_size)
        or marker_block_size != math.floor(marker_block_size)
        or marker_block_size < 1
    ):
        raise SmartEvidenceError(
            "markers_per_information_block must be a positive integer"
        )
    for name, value in (
        (
            "effective_markers_per_information_block",
            effective_markers_per_information_block,
        ),
        ("external_state_pseudocount", external_state_pseudocount),
        ("external_transition_pseudocount", external_transition_pseudocount),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise SmartEvidenceError(f"{name} must be finite and positive")

    try:
        child_chunk_floor = operator.index(_dynamic_child_chunk_floor)
    except TypeError as exc:
        raise SmartEvidenceError(
            "_dynamic_child_chunk_floor must be a positive integer"
        ) from exc
    if child_chunk_floor < 1:
        raise SmartEvidenceError(
            "_dynamic_child_chunk_floor must be a positive integer"
        )
    try:
        child_chunk_scale = operator.index(_dynamic_child_chunk_scale)
    except TypeError as exc:
        raise SmartEvidenceError(
            "_dynamic_child_chunk_scale must be a non-negative integer"
        ) from exc
    if child_chunk_scale < 0:
        raise SmartEvidenceError(
            "_dynamic_child_chunk_scale must be a non-negative integer"
        )

    (
        labels,
        founders,
        active_states,
        continuation_bridge,
        trajectory_classes,
    ) = _pool_local_ibs_states(labels, founders)
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    totals = _background_totals(alleles, labels, founders.shape[0])
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    background = _prepare_leave_child_out_background(
        alleles,
        labels,
        marker_counts,
        active_states,
        continuation_bridge,
        theta,
        *totals,
        float(external_state_pseudocount),
        float(external_transition_pseudocount),
        int(marker_block_size),
        float(effective_markers_per_information_block),
    )
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    zero = _score_zero_parent_forward_kernel(
        alleles,
        hom,
        founders,
        *background,
        float(phase_switch_probability),
        float(mismatch_probability),
    )
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    effective_transmission, effective_phase = (
        _effective_switch_probability_kernel(
            hom,
            theta,
            float(phase_switch_probability),
        )
    )
    one = np.full(
        (alleles.shape[0], alleles.shape[0]), -math.inf, dtype=np.float64
    )
    child_start = 0
    while child_start < alleles.shape[0]:
        applied_threads = (
            _apply_smart_dynamic_threads()
            if _dynamic_rebalance
            else alleles.shape[0]
        )
        child_chunk_size = max(child_chunk_floor, applied_threads)
        if child_chunk_scale:
            child_chunk_size = max(
                child_chunk_size,
                min(64, child_chunk_scale * applied_threads),
            )
        child_end = min(
            child_start + child_chunk_size,
            alleles.shape[0],
        )
        one[child_start:child_end] = (
            _score_one_parent_forward_kernel_grouped(
                alleles,
                founders,
                *background,
                effective_transmission,
                effective_phase,
                float(mismatch_probability),
                child_start,
                child_end,
            )
        )
        child_start = child_end
    if len(trio_array):
        if np.all(trio_array[1:, 0] >= trio_array[:-1, 0]):
            trio_order = None
            sorted_trios = trio_array
        else:
            trio_order = np.argsort(
                trio_array[:, 0], kind="stable"
            ).astype(np.int64)
            sorted_trios = np.ascontiguousarray(
                trio_array[trio_order], dtype=np.int64
            )
        child_row_starts = np.searchsorted(
            sorted_trios[:, 0],
            np.arange(alleles.shape[0] + 1, dtype=np.int64),
        ).astype(np.int64)
    else:
        trio_order = None
        sorted_trios = trio_array
        child_row_starts = np.zeros(
            alleles.shape[0] + 1, dtype=np.int64
        )
    sorted_two = np.full(len(sorted_trios), -math.inf, dtype=np.float64)
    child_start = 0
    while child_start < alleles.shape[0]:
        applied_threads = (
            _apply_smart_dynamic_threads()
            if _dynamic_rebalance
            else alleles.shape[0]
        )
        child_chunk_size = max(child_chunk_floor, applied_threads)
        if child_chunk_scale:
            child_chunk_size = max(
                child_chunk_size,
                min(64, child_chunk_scale * applied_threads),
            )
        child_end = min(
            child_start + child_chunk_size,
            alleles.shape[0],
        )
        row_start = int(child_row_starts[child_start])
        row_end = int(child_row_starts[child_end])
        sorted_two[row_start:row_end] = (
            _score_two_parent_forward_kernel_grouped(
                alleles,
                sorted_trios,
                child_row_starts,
                background[2],
                background[3],
                effective_transmission,
                effective_phase,
                float(mismatch_probability),
                child_start,
                child_end,
            )
        )
        child_start = child_end
    if trio_order is None:
        two = sorted_two
    else:
        two = np.empty_like(sorted_two)
        two[trio_order] = sorted_two
    if _dynamic_rebalance:
        _apply_smart_dynamic_threads()
    junctions, callable_bins = _ancestry_junction_count_kernel(
        labels, trajectory_classes
    )
    return _ParentStateContigScores(
        zero, one, two, junctions, callable_bins
    )


def _normalise_genotype_likelihoods(values: np.ndarray, kind: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise SmartEvidenceError(
            "genotype likelihoods must have shape (samples, markers, 3)"
        )
    key = str(kind).lower()
    if key in {"pl", "phred"}:
        invalid_rows = np.any(~np.isfinite(array) | (array < 0.0), axis=2)
        sanitized = array.copy()
        sanitized[invalid_rows] = 0.0
        shifted = sanitized - np.min(sanitized, axis=2, keepdims=True)
        likelihoods = np.power(10.0, -shifted / 10.0)
    elif key in {"log", "ln", "log_likelihood"}:
        if np.any(~np.isfinite(array)):
            raise SmartEvidenceError("log genotype likelihoods must be finite")
        shifted = array - np.max(array, axis=2, keepdims=True)
        likelihoods = np.exp(shifted)
    elif key in {"probability", "probabilities", "linear"}:
        if np.any(~np.isfinite(array)):
            raise SmartEvidenceError("linear genotype likelihoods must be finite")
        if np.any(array < 0.0):
            raise SmartEvidenceError("linear genotype likelihoods cannot be negative")
        likelihoods = array.copy()
    else:
        raise SmartEvidenceError(
            "likelihood_kind must be 'PL', 'log', or 'probability'"
        )
    totals = np.sum(likelihoods, axis=2, keepdims=True)
    if np.any(totals <= 0.0):
        raise SmartEvidenceError("every sample-marker likelihood vector needs mass")
    likelihoods /= totals
    return likelihoods


def _empirical_allele_frequencies(
    likelihoods: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate HWE allele frequencies and genotype posteriors by EM.

    Normalized PL/GL vectors are relative likelihoods, not genotype
    posteriors. This empirical-Bayes step supplies the required genotype prior
    before parental genotype uncertainty is integrated over transmissions.
    """
    if iterations < 1:
        raise SmartEvidenceError("frequency_em_iterations must be positive")
    n_samples = likelihoods.shape[0]
    frequencies = np.full(likelihoods.shape[1], 0.5, dtype=np.float64)
    posterior = np.empty_like(likelihoods)
    for _ in range(iterations):
        prior = np.stack((
            (1.0 - frequencies) ** 2,
            2.0 * frequencies * (1.0 - frequencies),
            frequencies ** 2,
        ), axis=1)
        posterior[:] = likelihoods * prior[None, :, :]
        normalizer = np.sum(posterior, axis=2, keepdims=True)
        posterior /= np.maximum(normalizer, np.finfo(np.float64).tiny)
        updated = np.sum(
            posterior[:, :, 1] + 2.0 * posterior[:, :, 2], axis=0
        ) / (2.0 * n_samples)
        frequencies = np.clip(updated, 1e-6, 1.0 - 1e-6)
    return frequencies, posterior


def _mendelian_table() -> np.ndarray:
    """P(child genotype | two unphased diploid parent genotypes)."""
    table = np.zeros((3, 3, 3), dtype=np.float64)
    gametes = np.asarray(((1.0, 0.0), (0.5, 0.5), (0.0, 1.0)))
    for first in range(3):
        for second in range(3):
            for first_alt in range(2):
                for second_alt in range(2):
                    child = first_alt + second_alt
                    table[first, second, child] += (
                        gametes[first, first_alt] * gametes[second, second_alt]
                    )
    return table


_MENDELIAN = _mendelian_table()


@njit(cache=True, parallel=True, fastmath=False)
def _score_genotype_likelihood_kernel(
    likelihoods: np.ndarray,
    genotype_posteriors: np.ndarray,
    null_child: np.ndarray,
    trios: np.ndarray,
    mendelian_table: np.ndarray,
) -> np.ndarray:
    """Parallel two-parent minus population-null composite likelihood."""
    scores = np.empty(len(trios), dtype=np.float64)
    epsilon = 2.2250738585072014e-308
    for row in prange(len(trios)):
        child = trios[row, 0]
        first = trios[row, 1]
        second = trios[row, 2]
        score = 0.0
        for marker in range(likelihoods.shape[1]):
            mendelian = 0.0
            for first_genotype in range(3):
                first_probability = genotype_posteriors[
                    first, marker, first_genotype
                ]
                for second_genotype in range(3):
                    parental_probability = (
                        first_probability
                        * genotype_posteriors[second, marker, second_genotype]
                    )
                    for child_genotype in range(3):
                        mendelian += (
                            parental_probability
                            * mendelian_table[
                                first_genotype, second_genotype, child_genotype
                            ]
                            * likelihoods[child, marker, child_genotype]
                        )
            null = 0.0
            for child_genotype in range(3):
                null += (
                    null_child[marker, child_genotype]
                    * likelihoods[child, marker, child_genotype]
                )
            score += math.log(max(mendelian, epsilon))
            score -= math.log(max(null, epsilon))
        scores[row] = score
    return scores


def _score_genotype_likelihood_reference(
    likelihoods: np.ndarray,
    genotype_posteriors: np.ndarray,
    null_child: np.ndarray,
    trios: np.ndarray,
) -> np.ndarray:
    """Independent vectorized oracle for kernel regression tests."""
    scores = np.empty(len(trios), dtype=np.float64)
    epsilon = np.finfo(np.float64).tiny
    for row, (child, first, second) in enumerate(trios):
        expected_child = np.einsum(
            "mg,mh,ghc->mc",
            genotype_posteriors[int(first)],
            genotype_posteriors[int(second)],
            _MENDELIAN,
            optimize=True,
        )
        mendelian = np.sum(expected_child * likelihoods[int(child)], axis=1)
        null = np.sum(null_child * likelihoods[int(child)], axis=1)
        scores[row] = float(
            np.sum(
                np.log(np.maximum(mendelian, epsilon))
                - np.log(np.maximum(null, epsilon))
            )
        )
    return scores


def smart_numba_thread_capacity() -> int:
    """Return the active Numba thread capacity without changing it."""
    if not SMART_HAS_NUMBA:
        return 1
    return int(numba.get_num_threads())


def score_genotype_likelihood_evidence(
    contig: str,
    genotype_likelihoods: np.ndarray,
    candidate_trios: np.ndarray,
    *,
    likelihood_kind: str = "PL",
    positions: Optional[np.ndarray] = None,
    linked_log_likelihoods: Optional[np.ndarray] = None,
    allele_frequencies: Optional[np.ndarray] = None,
    frequency_em_iterations: int = 12,
) -> SmartContigEvidence:
    """Score explicit candidate trios from raw genotype likelihoods.

    The score is the marker-summed log likelihood ratio between Mendelian
    transmission from the candidate pair and an empirical-population child
    genotype distribution. It is intentionally an unlinked complement to a
    painting/HMM score, not a substitute for recombination-aware evidence.
    Candidate eligibility is supplied numerically by the caller and is never
    inferred from identifiers. Unless ``allele_frequencies`` is supplied, the
    genotype prior is an empirical HWE estimate; closed crosses can violate
    that approximation, so linked painting/HMM evidence should remain primary.
    Candidate trios are evaluated by a memory-bounded Numba ``prange`` kernel
    using the currently configured thread capacity; this function does not
    force a thread count or create nested parallelism.
    """
    likelihoods = _normalise_genotype_likelihoods(
        genotype_likelihoods, likelihood_kind
    )
    trios = np.asarray(candidate_trios, dtype=np.int64)
    if trios.ndim != 2 or trios.shape[1] != 3 or len(trios) == 0:
        raise SmartEvidenceError("candidate_trios must have shape (n, 3)")
    n_samples, n_markers, _ = likelihoods.shape
    if np.any(trios < 0) or np.any(trios >= n_samples):
        raise SmartEvidenceError("candidate_trios contains an invalid sample index")
    if np.any(trios[:, 0] == trios[:, 1]) or np.any(trios[:, 0] == trios[:, 2]):
        raise SmartEvidenceError("a child cannot be its own parent")
    if np.any(trios[:, 1] == trios[:, 2]):
        raise SmartEvidenceError("the two candidate parents must differ")
    if positions is not None:
        marker_positions = np.asarray(positions)
        if marker_positions.ndim != 1 or len(marker_positions) != n_markers:
            raise SmartEvidenceError("positions must have one value per marker")
        if len(marker_positions) > 1 and np.any(np.diff(marker_positions) <= 0):
            raise SmartEvidenceError("positions must be strictly increasing")

    if allele_frequencies is None:
        allele_frequency, genotype_posteriors = _empirical_allele_frequencies(
            likelihoods, frequency_em_iterations
        )
    else:
        allele_frequency = np.asarray(allele_frequencies, dtype=np.float64)
        if allele_frequency.shape != (n_markers,):
            raise SmartEvidenceError(
                "allele_frequencies must contain one value per marker"
            )
        if np.any(~np.isfinite(allele_frequency)) or np.any(
            (allele_frequency <= 0.0) | (allele_frequency >= 1.0)
        ):
            raise SmartEvidenceError(
                "allele_frequencies must be finite and strictly inside (0, 1)"
            )
        prior = np.stack((
            (1.0 - allele_frequency) ** 2,
            2.0 * allele_frequency * (1.0 - allele_frequency),
            allele_frequency ** 2,
        ), axis=1)
        genotype_posteriors = likelihoods * prior[None, :, :]
        genotype_posteriors /= np.maximum(
            np.sum(genotype_posteriors, axis=2, keepdims=True),
            np.finfo(np.float64).tiny,
        )
    null_child = np.column_stack((
        (1.0 - allele_frequency) ** 2,
        2.0 * allele_frequency * (1.0 - allele_frequency),
        allele_frequency ** 2,
    ))

    # Parallelism is over candidate trios. The function respects the active
    # Numba thread pool configured by thread_config/the caller and never opens
    # a nested process pool or changes the requested thread count.
    genotype_scores = _score_genotype_likelihood_kernel(
        np.ascontiguousarray(likelihoods),
        np.ascontiguousarray(genotype_posteriors),
        np.ascontiguousarray(null_child),
        np.ascontiguousarray(trios),
        _MENDELIAN,
    )

    if linked_log_likelihoods is None:
        linked = np.zeros(len(trios), dtype=np.float64)
    else:
        linked = np.asarray(linked_log_likelihoods, dtype=np.float64)
        if linked.shape != (len(trios),) or np.any(~np.isfinite(linked)):
            raise SmartEvidenceError(
                "linked_log_likelihoods must be one finite score per trio"
            )
    informative_marker_mask = np.any(
        np.ptp(likelihoods, axis=2) > 1e-12, axis=0
    )
    informative_markers = int(np.sum(informative_marker_mask))
    if informative_markers == 0:
        raise SmartEvidenceError(
            "genotype likelihoods contain no informative marker evidence"
        )
    return SmartContigEvidence(
        contig=str(contig),
        trios=trios.copy(),
        linked_log_likelihoods=linked.copy(),
        genotype_log_likelihoods=genotype_scores,
        informative_markers=informative_markers,
    )


def load_bcf_genotype_likelihoods(
    bcf_path: Any,
    contig: str,
    sample_ids: Sequence[Any],
    *,
    selected_positions: Optional[np.ndarray] = None,
    threads: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Load PL evidence from one explicitly named BCF/VCF contig.

    The header sample order must exactly equal ``sample_ids``. No path,
    sample, contig, or candidate discovery is attempted. When positions are
    supplied every requested position must be present with a valid three-state
    PL vector, preventing silent panel changes across chromosomes.
    """
    try:
        from cyvcf2 import VCF
    except ImportError as error:
        raise SmartEvidenceError(
            "cyvcf2 is required to load an explicitly supplied BCF/VCF"
        ) from error
    if int(threads) != threads or threads < 1:
        raise SmartEvidenceError("threads must be a positive integer")
    requested = None
    if selected_positions is not None:
        requested_array = np.asarray(selected_positions, dtype=np.int64)
        if requested_array.ndim != 1 or len(requested_array) == 0:
            raise SmartEvidenceError(
                "selected_positions must be a non-empty one-dimensional array"
            )
        if np.any(np.diff(requested_array) <= 0):
            raise SmartEvidenceError(
                "selected_positions must be unique and strictly increasing"
            )
        requested = set(int(value) for value in requested_array)

    reader = VCF(str(bcf_path), threads=int(threads))
    try:
        if list(reader.samples) != list(sample_ids):
            raise SmartEvidenceError(
                "BCF/VCF header sample order does not exactly match sample_ids"
            )
        values_by_position = {}
        for variant in reader(str(contig)):
            position = int(variant.POS)
            if requested is not None and position not in requested:
                continue
            values = variant.format("PL")
            if values is None:
                continue
            values = np.asarray(values, dtype=np.float64)
            if values.ndim != 2 or values.shape != (len(sample_ids), 3):
                continue
            # cyvcf2 uses negative sentinels for a missing sample call. Keep
            # the marker and make only that sample's PL row uninformative;
            # dropping the whole marker would let one missing sample erase
            # evidence for every other trio.
            invalid_rows = np.any(~np.isfinite(values) | (values < 0), axis=1)
            values = values.copy()
            values[invalid_rows] = 0.0
            if position in values_by_position:
                raise SmartEvidenceError(
                    f"duplicate PL-bearing position {position} on {contig}"
                )
            values_by_position[position] = values
    finally:
        reader.close()
    if requested is not None:
        missing = requested.difference(values_by_position)
        if missing:
            raise SmartEvidenceError(
                f"{len(missing)} requested PL positions are absent on {contig}"
            )
    if not values_by_position:
        raise SmartEvidenceError(f"no valid three-state PL evidence on {contig}")
    positions = np.asarray(sorted(values_by_position), dtype=np.int64)
    likelihoods = np.stack(
        [values_by_position[int(position)] for position in positions], axis=1
    )
    return likelihoods, positions


def _as_contig_evidence(value: Any) -> SmartContigEvidence:
    if isinstance(value, SmartContigEvidence):
        evidence = value
    elif isinstance(value, Mapping):
        try:
            trios = np.asarray(value["trios"], dtype=np.int64)
            evidence = SmartContigEvidence(
                contig=str(value["contig"]),
                trios=trios,
                linked_log_likelihoods=np.asarray(
                    value.get("linked_log_likelihoods", np.zeros(len(trios))),
                    dtype=np.float64,
                ),
                genotype_log_likelihoods=np.asarray(
                    value.get("genotype_log_likelihoods", np.zeros(len(trios))),
                    dtype=np.float64,
                ),
                informative_markers=int(value["informative_markers"]),
            )
        except KeyError as error:
            raise SmartEvidenceError(
                f"smart evidence is missing required field {error.args[0]!r}"
            ) from error
    else:
        raise SmartEvidenceError(
            "smart_evidence must be SmartContigEvidence or an explicit mapping"
        )
    trios = np.asarray(evidence.trios, dtype=np.int64)
    if trios.ndim != 2 or trios.shape[1] != 3 or len(trios) == 0:
        raise SmartEvidenceError("each contig needs a non-empty (n, 3) trio array")
    linked = np.asarray(evidence.linked_log_likelihoods, dtype=np.float64)
    genotype = np.asarray(evidence.genotype_log_likelihoods, dtype=np.float64)
    if linked.shape != (len(trios),) or genotype.shape != (len(trios),):
        raise SmartEvidenceError("each score array must have one value per trio")
    if np.any(~np.isfinite(linked)) or np.any(~np.isfinite(genotype)):
        raise SmartEvidenceError("smart evidence scores must be finite")
    if evidence.informative_markers < 1:
        raise SmartEvidenceError("informative_markers must be positive")
    return replace(
        evidence,
        trios=trios.copy(),
        linked_log_likelihoods=linked.copy(),
        genotype_log_likelihoods=genotype.copy(),
    )


def _canonical_evidence(
    evidence: Sequence[SmartContigEvidence], n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    keys_reference: Optional[list[tuple[int, int, int]]] = None
    linked_rows = []
    genotype_rows = []
    markers = []
    names = []
    seen_contigs = set()
    for raw in evidence:
        item = _as_contig_evidence(raw)
        if item.contig in seen_contigs:
            raise SmartEvidenceError(f"duplicate contig identifier {item.contig!r}")
        seen_contigs.add(item.contig)
        keys = []
        score_lookup = {}
        for row, triple in enumerate(item.trios):
            child, first, second = (int(value) for value in triple)
            if not (0 <= child < n_samples and 0 <= first < n_samples
                    and 0 <= second < n_samples):
                raise SmartEvidenceError("trio index outside sample array")
            if child in (first, second) or first == second:
                raise SmartEvidenceError("invalid self-parent or duplicate-parent trio")
            if second < first:
                first, second = second, first
            key = (child, first, second)
            if key in score_lookup:
                raise SmartEvidenceError(f"duplicate trio key {key} on {item.contig}")
            score_lookup[key] = (
                float(item.linked_log_likelihoods[row]),
                float(item.genotype_log_likelihoods[row]),
            )
            keys.append(key)
        keys = sorted(keys)
        if keys_reference is None:
            keys_reference = keys
        elif keys != keys_reference:
            raise SmartEvidenceError(
                "every contig must score the same canonical candidate-trio panel"
            )
        linked_rows.append([score_lookup[key][0] for key in keys])
        genotype_rows.append([score_lookup[key][1] for key in keys])
        markers.append(item.informative_markers)
        names.append(item.contig)
    if not linked_rows or keys_reference is None:
        raise SmartEvidenceError("at least one explicit contig is required")
    return (
        np.asarray(keys_reference, dtype=np.int64),
        np.asarray(linked_rows, dtype=np.float64),
        np.asarray(genotype_rows, dtype=np.float64),
        np.asarray(markers, dtype=np.float64),
        names,
    )


_CONTRAST_ULP_FACTOR = 512.0


def _contrast_tolerance(values: np.ndarray) -> float:
    """Tolerance covering scale-dependent floating evaluation-order noise."""
    finite = np.asarray(values, dtype=np.float64)
    if finite.ndim != 1 or len(finite) == 0 or np.any(~np.isfinite(finite)):
        raise SmartEvidenceError("candidate utilities must be finite vectors")
    scale = max(1.0, float(np.max(np.abs(finite))))
    return _CONTRAST_ULP_FACTOR * np.finfo(np.float64).eps * scale


def _has_genuine_contrast(values: np.ndarray) -> bool:
    """Return whether scores contain any non-numerical candidate range."""
    scores = np.asarray(values, dtype=np.float64)
    if len(scores) < 2:
        return False
    return float(np.ptp(scores)) > _contrast_tolerance(scores)


def _unique_winner_contrast(values: np.ndarray) -> tuple[bool, float]:
    """Return unique-winner identifiability and its top-two score margin."""
    scores = np.asarray(values, dtype=np.float64)
    if len(scores) < 2:
        return False, 0.0
    ordered = np.sort(scores)
    margin = float(ordered[-1] - ordered[-2])
    return margin > _contrast_tolerance(scores), margin


def _tied_rank_probabilities(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    n_values = len(values)
    if n_values == 1:
        return np.ones(1, dtype=np.float64)
    tolerance = _contrast_tolerance(values)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(n_values, dtype=np.float64)
    start = 0
    while start < n_values:
        end = start + 1
        while (
            end < n_values
            and abs(values[order[end]] - values[order[start]]) <= tolerance
        ):
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    evidence = (n_values - ranks) / n_values
    evidence /= np.sum(evidence)
    return evidence


def _source_utilities(
    scores: np.ndarray,
    trios: np.ndarray,
    markers: np.ndarray,
    config: SmartPedigreeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform one source and mark any genuine child-contig score range."""
    utilities = np.zeros_like(scores, dtype=np.float64)
    children = np.unique(trios[:, 0])
    contrast = np.zeros(
        (scores.shape[0], int(np.max(children)) + 1), dtype=np.bool_
    )
    n_blocks = np.maximum(
        np.ceil(markers / config.markers_per_information_block), 1.0
    )
    temper = n_blocks ** config.information_tempering_power
    for contig_index in range(scores.shape[0]):
        for child in children:
            rows = np.flatnonzero(trios[:, 0] == child)
            values = scores[contig_index, rows]
            has_contrast = _has_genuine_contrast(values)
            if not has_contrast:
                utilities[contig_index, rows] = 1.0 / len(rows)
                continue
            contrast[contig_index, int(child)] = True
            centered = (values - np.max(values)) / temper[contig_index]
            centered = np.clip(centered, -60.0, 0.0)
            soft = np.exp(centered)
            soft /= np.sum(soft)
            soft = (
                (1.0 - config.chromosome_contamination) * soft
                + config.chromosome_contamination / len(rows)
            )
            ranks = _tied_rank_probabilities(values)
            utilities[contig_index, rows] = (
                config.rank_weight * ranks + (1.0 - config.rank_weight) * soft
            )
    return utilities, contrast


def _information_weights(
    markers: np.ndarray, config: SmartPedigreeConfig
) -> np.ndarray:
    weights = np.sqrt(np.asarray(markers, dtype=np.float64))
    median = float(np.median(weights))
    if median <= 0.0:
        raise SmartEvidenceError("marker information is empty")
    ratio = config.maximum_contig_weight_ratio
    weights = np.clip(weights / median, 1.0 / ratio, ratio)
    return weights / np.sum(weights)


def _combined_utilities(
    linked: np.ndarray,
    genotype: np.ndarray,
    trios: np.ndarray,
    markers: np.ndarray,
    config: SmartPedigreeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine sources with child-contig-specific identifiability masks."""
    source_arrays = []
    source_contrasts = []
    source_weights = []
    if config.linked_evidence_weight > 0.0:
        values, contrast = _source_utilities(
            linked, trios, markers, config
        )
        source_arrays.append(values)
        source_contrasts.append(contrast)
        source_weights.append(config.linked_evidence_weight)
    if config.genotype_evidence_weight > 0.0:
        values, contrast = _source_utilities(
            genotype, trios, markers, config
        )
        source_arrays.append(values)
        source_contrasts.append(contrast)
        source_weights.append(config.genotype_evidence_weight)

    weights = np.asarray(source_weights, dtype=np.float64)
    utilities = np.zeros_like(source_arrays[0])
    combined_contrast = np.logical_or.reduce(source_contrasts)
    children = np.unique(trios[:, 0])
    for contig_index in range(linked.shape[0]):
        for child in children:
            rows = np.flatnonzero(trios[:, 0] == child)
            active = np.asarray([
                contrast[contig_index, int(child)]
                for contrast in source_contrasts
            ], dtype=np.bool_)
            if not np.any(active):
                utilities[contig_index, rows] = 1.0 / len(rows)
                continue
            active_weights = weights[active]
            active_weights /= np.sum(active_weights)
            active_indices = np.flatnonzero(active)
            for source_weight, source_index in zip(
                active_weights, active_indices
            ):
                utilities[contig_index, rows] += (
                    source_weight
                    * source_arrays[int(source_index)][contig_index, rows]
                )
    return utilities, _information_weights(markers, config), combined_contrast


def _path_exists(adjacency: list[set[int]], start: int, target: int) -> bool:
    if start == target:
        return True
    stack = [start]
    seen = {start}
    while stack:
        node = stack.pop()
        for neighbour in adjacency[node]:
            if neighbour == target:
                return True
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return False


def _acyclic_selection(
    trios: np.ndarray,
    scores: np.ndarray,
    n_samples: int,
    local_search_passes: int,
) -> dict[int, int]:
    """Select a DAG without resolving score ties by canonical row order."""
    by_child = {}
    margins = {}
    for child in np.unique(trios[:, 0]):
        rows = np.flatnonzero(trios[:, 0] == child)
        identifiable, margin = _unique_winner_contrast(scores[rows])
        if identifiable:
            by_child[int(child)] = rows
            margins[int(child)] = margin
    child_order = sorted(by_child, key=lambda value: (-margins[value], value))
    adjacency = [set() for _ in range(n_samples)]
    selected: dict[int, int] = {}

    def can_add(row: int) -> bool:
        child, first, second = (int(value) for value in trios[row])
        return not (
            _path_exists(adjacency, child, first)
            or _path_exists(adjacency, child, second)
        )

    def add(row: int) -> None:
        child, first, second = (int(value) for value in trios[row])
        adjacency[first].add(child)
        adjacency[second].add(child)
        selected[child] = row

    def remove(child: int) -> Optional[int]:
        row = selected.pop(child, None)
        if row is not None:
            _, first, second = (int(value) for value in trios[row])
            adjacency[first].discard(child)
            adjacency[second].discard(child)
        return row

    def best_feasible(child: int) -> Optional[int]:
        rows = by_child[child]
        ordered = rows[np.argsort(-scores[rows], kind="stable")]
        feasible = np.asarray(
            [int(row) for row in ordered if can_add(int(row))],
            dtype=np.int64,
        )
        if len(feasible) == 0:
            return None
        if len(feasible) == 1:
            return int(feasible[0])
        identifiable, _ = _unique_winner_contrast(scores[feasible])
        if not identifiable:
            return None
        return int(feasible[0])

    for child in child_order:
        row = best_feasible(child)
        if row is not None:
            add(row)

    for _ in range(local_search_passes):
        changed = False
        for child in child_order:
            previous = remove(child)
            replacement = best_feasible(child)
            if replacement is not None:
                add(replacement)
            changed |= replacement != previous
        if not changed:
            break
    return selected


def _aggregate_scores(
    utilities: np.ndarray, weights: np.ndarray, selected_contigs: np.ndarray
) -> np.ndarray:
    chosen_weights = weights[selected_contigs]
    chosen_weights = chosen_weights / np.sum(chosen_weights)
    return np.sum(utilities[selected_contigs] * chosen_weights[:, None], axis=0)


def _generation_frame(
    sample_ids: Sequence[Any], selected: Mapping[int, tuple[Optional[int], Optional[int]]]
) -> pd.DataFrame:
    """Build an all-sample relationship frame without invented cohorts.

    A both-null row can be an unscored root or an unresolved descendant. With
    no metadata those states are not distinguishable, and graph depth cannot
    establish biological G0/F1/F2 labels. Generation therefore remains
    ``Unknown`` for every row; selected parent edges carry the inferred
    topology.
    """
    rows = []
    for child, sample in enumerate(sample_ids):
        first, second = selected.get(child, (None, None))
        rows.append({
            "Sample": sample,
            "Generation": "Unknown",
            "Parent1": None if first is None else sample_ids[first],
            "Parent2": None if second is None else sample_ids[second],
        })
    return pd.DataFrame(rows)


def _support_text(
    rows: np.ndarray,
    probabilities: np.ndarray,
    trios: np.ndarray,
    sample_ids: Sequence[Any],
    coverage: float,
) -> str:
    order = rows[np.argsort(-probabilities[rows], kind="stable")]
    pieces = []
    cumulative = 0.0
    denominator = float(np.sum(probabilities[rows]))
    if denominator <= 0.0:
        return ""
    for row in order:
        _, first, second = (int(value) for value in trios[row])
        probability = float(probabilities[row] / denominator)
        pieces.append(f"{sample_ids[first]}+{sample_ids[second]}:{probability:.3f}")
        cumulative += probability
        if cumulative >= coverage:
            break
    return ";".join(pieces)


def infer_from_contig_evidence(
    evidence: Sequence[SmartContigEvidence],
    sample_ids: Sequence[Any],
    config: Optional[SmartPedigreeConfig] = None,
) -> PedigreeResult:
    """Infer a tiered acyclic pedigree from explicit chromosome evidence."""
    settings = (config or SmartPedigreeConfig()).validated()
    sample_ids = list(sample_ids)
    if len(sample_ids) < 3 or len(set(sample_ids)) != len(sample_ids):
        raise SmartEvidenceError("sample_ids must contain at least three unique IDs")
    trios, linked, genotype, markers, contig_names = _canonical_evidence(
        evidence, len(sample_ids)
    )
    utilities, contig_weights, contig_contrast = _combined_utilities(
        linked, genotype, trios, markers, settings
    )
    all_contigs = np.arange(len(contig_names), dtype=np.int64)
    aggregate = _aggregate_scores(utilities, contig_weights, all_contigs)
    complete_rows = _acyclic_selection(
        trios, aggregate, len(sample_ids), settings.dag_local_search_passes
    )

    pair_counts = np.zeros(len(trios), dtype=np.int64)
    parent_counts = np.zeros((len(sample_ids), len(sample_ids)), dtype=np.int64)
    rng = np.random.default_rng(settings.bootstrap_seed)
    for _ in range(settings.bootstrap_replicates):
        selected_contigs = rng.integers(0, len(contig_names), len(contig_names))
        scores = _aggregate_scores(utilities, contig_weights, selected_contigs)
        chosen = _acyclic_selection(
            trios, scores, len(sample_ids), settings.dag_local_search_passes
        )
        for child, row in chosen.items():
            pair_counts[row] += 1
            _, first, second = (int(value) for value in trios[row])
            parent_counts[child, first] += 1
            parent_counts[child, second] += 1

    loco_choices = []
    if len(contig_names) > 1:
        for omitted in range(len(contig_names)):
            kept = all_contigs[all_contigs != omitted]
            scores = _aggregate_scores(utilities, contig_weights, kept)
            loco_choices.append(_acyclic_selection(
                trios, scores, len(sample_ids), settings.dag_local_search_passes
            ))

    children = sorted(int(value) for value in np.unique(trios[:, 0]))
    tier_a: dict[int, tuple[Optional[int], Optional[int]]] = {}
    tier_b: dict[int, tuple[Optional[int], Optional[int]]] = {}
    tier_a_exact: dict[int, tuple[Optional[int], Optional[int]]] = {}
    tier_b_exact: dict[int, tuple[Optional[int], Optional[int]]] = {}
    complete: dict[int, tuple[Optional[int], Optional[int]]] = {}
    diagnostics = []
    trio_candidates = {}
    parent_candidates = {}
    trio_scores = {}
    probability_vector = pair_counts.astype(np.float64)
    for child in children:
        child_rows = np.flatnonzero(trios[:, 0] == child)
        ordered = child_rows[np.argsort(-aggregate[child_rows], kind="stable")]
        aggregate_identifiable, margin = _unique_winner_contrast(
            aggregate[child_rows]
        )
        unconstrained_best = float(np.max(aggregate[child_rows]))
        informative_contig_count = int(np.count_nonzero(
            contig_contrast[:, child]
        ))
        enough_contigs = (
            informative_contig_count >= settings.minimum_informative_contigs
        )
        selected_row = complete_rows.get(child)
        identifiable = bool(
            aggregate_identifiable and selected_row is not None
        )
        if not aggregate_identifiable:
            unresolved_reason = "no_unique_full_aggregate_winner"
        elif selected_row is None:
            unresolved_reason = "no_unique_acyclic_feasible_pair"
        else:
            unresolved_reason = None

        if identifiable:
            row = int(selected_row)
            _, first, second = (int(value) for value in trios[row])
            complete[child] = (first, second)
            pair_bootstrap = pair_counts[row] / settings.bootstrap_replicates
            first_bootstrap = (
                parent_counts[child, first] / settings.bootstrap_replicates
            )
            second_bootstrap = (
                parent_counts[child, second] / settings.bootstrap_replicates
            )
            if loco_choices:
                pair_loco = np.mean([
                    choice.get(child) == row for choice in loco_choices
                ])
                first_loco = np.mean([
                    child in choice and first in trios[choice[child], 1:]
                    for choice in loco_choices
                ])
                second_loco = np.mean([
                    child in choice and second in trios[choice[child], 1:]
                    for choice in loco_choices
                ])
            else:
                pair_loco = first_loco = second_loco = 0.0
        else:
            row = None
            first = second = None
            complete[child] = (None, None)
            pair_bootstrap = np.nan
            first_bootstrap = np.nan
            second_bootstrap = np.nan
            pair_loco = np.nan
            first_loco = np.nan
            second_loco = np.nan

        selected_aggregate_utility = (
            np.nan if row is None else float(aggregate[row])
        )
        selected_minus_unconstrained_best = (
            np.nan
            if row is None
            else selected_aggregate_utility - unconstrained_best
        )

        tier_a_pair = bool(
            identifiable
            and enough_contigs
            and pair_bootstrap >= settings.tier_a_pair_bootstrap
            and pair_loco >= settings.tier_a_loco_fraction
        )
        tier_b_pair = bool(
            identifiable
            and enough_contigs
            and pair_bootstrap >= settings.tier_b_pair_bootstrap
            and pair_loco >= settings.tier_b_loco_fraction
        )
        tier_a_first = bool(
            identifiable
            and enough_contigs
            and first_bootstrap >= settings.tier_a_parent_bootstrap
            and first_loco >= settings.tier_a_loco_fraction
        )
        tier_a_second = bool(
            identifiable
            and enough_contigs
            and second_bootstrap >= settings.tier_a_parent_bootstrap
            and second_loco >= settings.tier_a_loco_fraction
        )
        tier_b_first = bool(
            identifiable
            and enough_contigs
            and first_bootstrap >= settings.tier_b_parent_bootstrap
            and first_loco >= settings.tier_b_loco_fraction
        )
        tier_b_second = bool(
            identifiable
            and enough_contigs
            and second_bootstrap >= settings.tier_b_parent_bootstrap
            and second_loco >= settings.tier_b_loco_fraction
        )
        if tier_a_pair:
            tier_a[child] = (first, second)
        else:
            tier_a[child] = (
                first if tier_a_first else None,
                second if tier_a_second else None,
            )
        if tier_b_pair:
            tier_b[child] = (first, second)
        else:
            tier_b[child] = (
                first if tier_b_first else None,
                second if tier_b_second else None,
            )
        tier_a_exact[child] = (
            (first, second) if tier_a_pair else (None, None)
        )
        tier_b_exact[child] = (
            (first, second) if tier_b_pair else (None, None)
        )

        diagnostics.append({
            "Sample": sample_ids[child],
            "CompleteParent1": (
                None if first is None else sample_ids[first]
            ),
            "CompleteParent2": (
                None if second is None else sample_ids[second]
            ),
            "ParentOrderMeaning": "unordered_sample_array_index",
            "Identifiable": identifiable,
            "InformativeContigCount": informative_contig_count,
            "PairBootstrapFraction": pair_bootstrap,
            "Parent1BootstrapFraction": first_bootstrap,
            "Parent2BootstrapFraction": second_bootstrap,
            "PairLOCOFraction": pair_loco,
            "Parent1LOCOFraction": first_loco,
            "Parent2LOCOFraction": second_loco,
            "UnconstrainedWinnerMargin": margin,
            "SelectedAggregateUtility": selected_aggregate_utility,
            "SelectedMinusUnconstrainedBest": (
                selected_minus_unconstrained_best
            ),
            "CandidatePairCount": len(child_rows),
            "UnresolvedReason": unresolved_reason,
            "TierAExactPair": tier_a_pair,
            "TierBExactPair": tier_b_pair,
            "TierAParent1": tier_a_first,
            "TierAParent2": tier_a_second,
            "TierBParent1": tier_b_first,
            "TierBParent2": tier_b_second,
            "PairSupportSet": _support_text(
                child_rows, probability_vector, trios, sample_ids,
                settings.support_set_coverage,
            ),
            "Interpretation": (
                "internal chromosome-resampling stability; not a calibrated "
                "biological posterior"
            ),
        })
        trio_candidates[sample_ids[child]] = [
            (
                sample_ids[int(trios[index, 1])],
                sample_ids[int(trios[index, 2])],
                float(aggregate[index]),
            )
            for index in ordered
        ]
        parent_score = {}
        for index in child_rows:
            for parent in trios[index, 1:]:
                parent = int(parent)
                parent_score[parent] = max(
                    parent_score.get(parent, -np.inf), float(aggregate[index])
                )
        parent_candidates[sample_ids[child]] = [
            (sample_ids[parent], score)
            for parent, score in sorted(
                parent_score.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        trio_scores[sample_ids[child]] = selected_aggregate_utility

    complete_frame = _generation_frame(sample_ids, complete)
    tier_a_partial_frame = _generation_frame(sample_ids, tier_a)
    tier_b_partial_frame = _generation_frame(sample_ids, tier_b)
    tier_a_frame = _generation_frame(sample_ids, tier_a_exact)
    tier_b_frame = _generation_frame(sample_ids, tier_b_exact)
    primary = {
        "tier_a": tier_a_frame,
        "tier_b": tier_b_frame,
        "complete": complete_frame,
    }[settings.primary_view].copy()
    effective_blocks = int(np.sum(np.ceil(
        markers / settings.markers_per_information_block
    )))
    # Construct against both HEAD and the current working implementation. HEAD
    # does not accept trio_candidate_scores in __init__, so the diagnostic
    # attribute is assigned after construction.
    result = PedigreeResult(
        sample_ids,
        primary,
        parent_candidates,
        None,
        [],
        None,
        None,
        trio_scores=trio_scores,
        total_bins=effective_blocks,
    )
    result.trio_candidate_scores = trio_candidates
    result.smart_mode = True
    result.smart_config = settings
    result.tier_a_partial_relationships = tier_a_partial_frame
    result.tier_b_partial_relationships = tier_b_partial_frame
    result.tier_a_relationships = tier_a_frame
    result.tier_b_relationships = tier_b_frame
    result.complete_relationships = complete_frame
    result.smart_diagnostics = pd.DataFrame(diagnostics)
    result.smart_evidence_summary = pd.DataFrame({
        "Contig": contig_names,
        "InformativeMarkers": markers.astype(np.int64),
        "AggregationWeight": contig_weights,
    })
    result.smart_selection_method = (
        "deterministic confidence-ordered greedy DAG with coordinate local "
        "search; not an exact MILP optimum"
    )
    result.smart_candidate_screening_scope = (
        "caller-supplied fixed candidate panel; chromosome bootstrap and LOCO "
        "stability are conditional on that panel"
    )
    result.smart_missing_parent_model = (
        "No calibrated zero-parent or one-parent likelihood is fitted. Stable "
        "individual-parent identities are diagnostics only; reported pedigree "
        "rows contain either an exact pair or no parents."
    )
    result.smart_limitations = (
        "No cohort, sex, or breeding-record information was used; parent order "
        "is arbitrary. The complete view assumes two observed candidates for "
        "each scored child, cannot identify a zero-parent state, and may contain "
        "weakly identified edges. Empirical HWE genotype priors can be "
        "misspecified in a closed cross; explicit allele frequencies or linked "
        "painting/HMM evidence are preferable when available. Stability is "
        "conditional on the fixed candidate panel. For API compatibility, "
        "PedigreeResult.trio_scores stores selected aggregate utilities and "
        "total_bins stores an effective information-block count; those units "
        "differ from the legacy raw HMM score and bin-count fields and must "
        "not be compared across implementations."
    )
    return result



def _as_parent_state_evidence(
    value: Any,
    n_samples: int,
) -> SmartParentStateEvidence:
    """Validate one comparable parent-state evidence object."""
    if isinstance(value, SmartParentStateEvidence):
        item = value
    elif isinstance(value, Mapping):
        try:
            item = SmartParentStateEvidence(
                contig=str(value["contig"]),
                trios=np.asarray(value["trios"], dtype=np.int64),
                zero_parent_log_likelihoods=np.asarray(
                    value["zero_parent_log_likelihoods"], dtype=np.float64
                ),
                one_parent_log_likelihoods=np.asarray(
                    value["one_parent_log_likelihoods"], dtype=np.float64
                ),
                two_parent_log_likelihoods=np.asarray(
                    value["two_parent_log_likelihoods"], dtype=np.float64
                ),
                informative_markers=int(value["informative_markers"]),
            )
        except KeyError as error:
            raise SmartEvidenceError(
                f"parent-state evidence is missing {error.args[0]!r}"
            ) from error
    else:
        raise SmartEvidenceError(
            "parent-state evidence must be SmartParentStateEvidence or a mapping"
        )

    trios = np.asarray(item.trios, dtype=np.int64)
    zero = np.asarray(item.zero_parent_log_likelihoods, dtype=np.float64)
    one = np.asarray(item.one_parent_log_likelihoods, dtype=np.float64)
    two = np.asarray(item.two_parent_log_likelihoods, dtype=np.float64)
    if trios.ndim != 2 or trios.shape[1] != 3:
        raise SmartEvidenceError("parent-state trios must have shape (rows, 3)")
    if zero.shape != (n_samples,) or np.any(~np.isfinite(zero)):
        raise SmartEvidenceError(
            "zero-parent evidence must be one finite log likelihood per child"
        )
    if one.shape != (n_samples, n_samples):
        raise SmartEvidenceError(
            "one-parent evidence must have shape (samples, samples)"
        )
    off_diagonal = ~np.eye(n_samples, dtype=np.bool_)
    if np.any(~np.isfinite(one[off_diagonal])):
        raise SmartEvidenceError(
            "off-diagonal one-parent log likelihoods must be finite"
        )
    if not np.all(np.isneginf(np.diag(one))):
        raise SmartEvidenceError(
            "the one-parent self-parent diagonal must be negative infinity"
        )
    if two.shape != (len(trios),) or np.any(~np.isfinite(two)):
        raise SmartEvidenceError(
            "two-parent evidence must be one finite log likelihood per trio"
        )
    if item.informative_markers < 1:
        raise SmartEvidenceError("informative_markers must be positive")
    return SmartParentStateEvidence(
        contig=str(item.contig),
        trios=trios.copy(),
        zero_parent_log_likelihoods=zero.copy(),
        one_parent_log_likelihoods=one.copy(),
        two_parent_log_likelihoods=two.copy(),
        informative_markers=int(item.informative_markers),
    )


def _canonical_parent_state_evidence(
    evidence: Sequence[SmartParentStateEvidence],
    n_samples: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]:
    """Align a fixed two-parent screen while retaining full M0/M1 evidence.

    Internally generated panels are already canonical and identically ordered.
    Validate that common case with vector operations and append its score array
    directly. Permuted caller-supplied panels are canonically sorted once,
    preserving the historical support for arbitrary external row order without
    rebuilding a 539,200-entry Python dictionary for every contig.
    """
    reference_trios: Optional[np.ndarray] = None
    zero_rows = []
    one_rows = []
    two_rows = []
    markers = []
    names = []
    seen_contigs = set()
    for raw in evidence:
        item = _as_parent_state_evidence(raw, n_samples)
        if item.contig in seen_contigs:
            raise SmartEvidenceError(f"duplicate contig identifier {item.contig!r}")
        seen_contigs.add(item.contig)

        canonical = np.asarray(item.trios, dtype=np.int64).copy()
        scores = np.asarray(
            item.two_parent_log_likelihoods, dtype=np.float64
        )
        if len(canonical):
            child = canonical[:, 0]
            first = canonical[:, 1]
            second = canonical[:, 2]
            if (
                np.any(child < 0)
                or np.any(child >= n_samples)
                or np.any(first < 0)
                or np.any(first >= n_samples)
                or np.any(second < 0)
                or np.any(second >= n_samples)
            ):
                raise SmartEvidenceError("trio index outside sample array")
            if np.any(child == first) or np.any(child == second):
                raise SmartEvidenceError(
                    "invalid self-parent or duplicate-parent trio"
                )
            swap = second < first
            if np.any(swap):
                temporary = canonical[swap, 1].copy()
                canonical[swap, 1] = canonical[swap, 2]
                canonical[swap, 2] = temporary
            if np.any(canonical[:, 1] == canonical[:, 2]):
                raise SmartEvidenceError(
                    "invalid self-parent or duplicate-parent trio"
                )

            previous = canonical[:-1]
            current = canonical[1:]
            ordered = np.all(
                (current[:, 0] > previous[:, 0])
                | (
                    (current[:, 0] == previous[:, 0])
                    & (
                        (current[:, 1] > previous[:, 1])
                        | (
                            (current[:, 1] == previous[:, 1])
                            & (current[:, 2] >= previous[:, 2])
                        )
                    )
                )
            )
            if not ordered:
                order = np.lexsort((
                    canonical[:, 2], canonical[:, 1], canonical[:, 0]
                ))
                canonical = np.ascontiguousarray(canonical[order])
                scores = np.ascontiguousarray(scores[order])
            duplicate = np.all(canonical[1:] == canonical[:-1], axis=1)
            if np.any(duplicate):
                key = tuple(
                    int(value) for value in canonical[1:][duplicate][0]
                )
                raise SmartEvidenceError(
                    f"duplicate trio key {key} on {item.contig}"
                )

        if reference_trios is None:
            reference_trios = canonical.copy()
        elif not np.array_equal(canonical, reference_trios):
            raise SmartEvidenceError(
                "every contig must score the same canonical two-parent panel"
            )
        zero_rows.append(item.zero_parent_log_likelihoods)
        one_rows.append(item.one_parent_log_likelihoods)
        two_rows.append(scores)
        markers.append(item.informative_markers)
        names.append(item.contig)
    if not zero_rows or reference_trios is None:
        raise SmartEvidenceError("at least one parent-state contig is required")
    return (
        np.asarray(reference_trios, dtype=np.int64).reshape((-1, 3)),
        np.asarray(zero_rows, dtype=np.float64),
        np.asarray(one_rows, dtype=np.float64),
        np.asarray(two_rows, dtype=np.float64),
        np.asarray(markers, dtype=np.float64),
        names,
    )

def _parent_state_alternatives(
    trios: np.ndarray,
    zero: np.ndarray,
    one: np.ndarray,
    two: np.ndarray,
    contamination: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """Build canonical M0/M1/M2 configurations and comparable likelihoods."""
    n_contigs, n_samples = zero.shape
    two_by_child = [
        np.flatnonzero(trios[:, 0] == child) for child in range(n_samples)
    ]
    n_rows = n_samples * n_samples + len(trios)
    alternatives = np.empty((n_rows, 3), dtype=np.int64)
    states = np.empty(n_rows, dtype=np.int8)
    log_likelihoods = np.empty((n_contigs, n_rows), dtype=np.float64)
    by_child = []
    scored_counts = np.zeros((n_samples, 3), dtype=np.int64)
    offset = 0
    for child in range(n_samples):
        start = offset
        alternatives[offset] = (child, _EXTERNAL_PARENT, _EXTERNAL_PARENT)
        states[offset] = _ZERO_OBSERVED
        log_likelihoods[:, offset] = zero[:, child]
        offset += 1

        for parent in range(n_samples):
            if parent == child:
                continue
            alternatives[offset] = (child, parent, _EXTERNAL_PARENT)
            states[offset] = _ONE_OBSERVED
            log_likelihoods[:, offset] = one[:, child, parent]
            offset += 1

        child_two_rows = two_by_child[child]
        for two_row in child_two_rows:
            alternatives[offset] = trios[int(two_row)]
            states[offset] = _TWO_OBSERVED
            log_likelihoods[:, offset] = two[:, int(two_row)]
            offset += 1
        rows = np.arange(start, offset, dtype=np.int64)
        by_child.append(rows)
        scored_counts[child] = (
            1,
            n_samples - 1,
            len(child_two_rows),
        )
    if offset != n_rows:
        raise AssertionError("internal parent-state alternative count mismatch")

    if contamination > 0.0:
        log_primary = math.log1p(-contamination)
        log_null = math.log(contamination)
        for child, rows in enumerate(by_child):
            alternative_rows = rows[states[rows] != _ZERO_OBSERVED]
            if len(alternative_rows):
                log_likelihoods[:, alternative_rows] = np.logaddexp(
                    log_primary * np.ones((n_contigs, 1))
                    + log_likelihoods[:, alternative_rows],
                    log_null * np.ones((n_contigs, 1))
                    + zero[:, child, None],
                )

    full_counts = np.empty((n_samples, 3), dtype=np.int64)
    full_counts[:, _ZERO_OBSERVED] = 1
    full_counts[:, _ONE_OBSERVED] = n_samples - 1
    full_counts[:, _TWO_OBSERVED] = math.comb(n_samples - 1, 2)
    return (
        alternatives,
        states,
        log_likelihoods,
        by_child,
        full_counts,
        scored_counts,
    )


def _logsumexp_finite(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return -np.inf
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def _softmax_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.zeros_like(values)
    finite = np.isfinite(values)
    if not np.any(finite):
        return output
    maximum = float(np.max(values[finite]))
    output[finite] = np.exp(values[finite] - maximum)
    output /= np.sum(output)
    return output


@njit(cache=True)
def _linear_top_two_finite(rows, scores):
    """Stable O(n) top-two scan used by repeated bootstrap selections."""
    best_row = -1
    best_value = -math.inf
    second_value = -math.inf
    finite_count = 0
    maximum_absolute_value = 1.0
    for position in range(len(rows)):
        row = int(rows[position])
        value = float(scores[row])
        if not math.isfinite(value):
            continue
        finite_count += 1
        absolute_value = abs(value)
        if absolute_value > maximum_absolute_value:
            maximum_absolute_value = absolute_value
        if best_row < 0 or value > best_value:
            if best_row >= 0:
                second_value = best_value
            best_row = row
            best_value = value
        elif value > second_value:
            # Strict comparison retains the first occurrence as the stable
            # winner while still making an equal later value the runner-up.
            second_value = value
    return (
        best_row,
        best_value,
        second_value,
        finite_count,
        maximum_absolute_value,
    )


def _unique_finite_winner(
    rows: np.ndarray,
    scores: np.ndarray,
) -> tuple[Optional[int], float]:
    candidate_rows = np.asarray(rows, dtype=np.int64)
    candidate_scores = np.asarray(scores, dtype=np.float64)
    (
        best_row,
        best_value,
        second_value,
        finite_count,
        maximum_absolute_value,
    ) = _linear_top_two_finite(candidate_rows, candidate_scores)
    if finite_count == 0:
        return None, 0.0
    if finite_count == 1:
        return int(best_row), np.inf
    margin = float(best_value - second_value)
    tolerance = (
        _CONTRAST_ULP_FACTOR
        * np.finfo(np.float64).eps
        * maximum_absolute_value
    )
    if margin <= tolerance:
        return None, margin
    return int(best_row), margin


_ANCESTRY_DEPTH_MAX_COMPONENTS = 6
_ANCESTRY_DEPTH_GMM_N_INIT = 10
_ANCESTRY_DEPTH_GMM_MAX_ITERATIONS = 500
_ANCESTRY_DEPTH_GMM_REGULARIZATION = 1e-3


@dataclass(frozen=True)
class _AncestryDepthModel:
    """Unsupervised relative ancestry-depth model from reconstructed paintings."""

    adjusted_junction_burden: np.ndarray
    callability_fraction: np.ndarray
    posterior: np.ndarray
    component_means: np.ndarray
    component_standard_deviations: np.ndarray
    component_weights: np.ndarray
    selected_bic: float
    tested_bics: tuple[float, ...]


@dataclass(frozen=True)
class _GraphParentStateSelection:
    rows: dict[int, int]
    direction_resolved_children: frozenset[int]
    selected_parent_role_probabilities: dict[int, float]


def _fit_ancestry_depth_model(
    junction_counts: np.ndarray,
    callable_haplotype_bins: np.ndarray,
    seed: int,
) -> _AncestryDepthModel:
    """Fit a deterministic BIC-selected mixture of relative ancestry burdens.

    Junction counts are scaled to the maximum observed painting callability.
    Component likelihoods are then tempered by each sample's relative
    callability, so incomplete paintings move toward the mixture prevalence
    rather than becoming spuriously certain shallow samples. Components are
    ordered by increasing burden and represent relative ancestry depth only;
    they are not generation labels.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture

    counts = np.asarray(junction_counts, dtype=np.float64)
    callable_bins = np.asarray(callable_haplotype_bins, dtype=np.float64)
    if (
        counts.ndim != 1
        or callable_bins.shape != counts.shape
        or np.any(~np.isfinite(counts))
        or np.any(~np.isfinite(callable_bins))
        or np.any(counts < 0.0)
        or np.any(callable_bins < 0.0)
    ):
        raise SmartEvidenceError(
            "ancestry junction counts and callability must be finite, "
            "non-negative sample vectors"
        )

    adjusted = np.full(len(counts), np.nan, dtype=np.float64)
    callability = np.zeros(len(counts), dtype=np.float64)
    valid = callable_bins > 0.0
    if not np.any(valid):
        return _AncestryDepthModel(
            adjusted,
            callability,
            np.zeros((len(counts), 1), dtype=np.float64),
            np.asarray((np.nan,)),
            np.asarray((np.nan,)),
            np.asarray((1.0,)),
            np.nan,
            (np.nan,),
        )

    reference_callability = float(np.max(callable_bins[valid]))
    callability[valid] = callable_bins[valid] / reference_callability
    adjusted[valid] = (
        counts[valid] * reference_callability / callable_bins[valid]
    )
    observed = adjusted[valid]
    center = float(np.mean(observed))
    scale = float(np.std(observed))
    distinct = np.unique(observed)
    if (
        len(observed) < 4
        or len(distinct) < 2
        or scale <= _contrast_tolerance(np.asarray((center,)))
    ):
        posterior = np.zeros((len(counts), 1), dtype=np.float64)
        posterior[valid, 0] = 1.0
        return _AncestryDepthModel(
            adjusted,
            callability,
            posterior,
            np.asarray((center,)),
            np.asarray((scale,)),
            np.asarray((1.0,)),
            np.nan,
            (np.nan,),
        )

    standardized = np.sort((observed - center) / scale)[:, None]
    maximum_components = min(
        _ANCESTRY_DEPTH_MAX_COMPONENTS,
        len(distinct),
        max(1, len(observed) // 2),
    )
    fitted = []
    bics = []
    for component_count in range(1, maximum_components + 1):
        model = GaussianMixture(
            n_components=component_count,
            covariance_type="full",
            reg_covar=_ANCESTRY_DEPTH_GMM_REGULARIZATION,
            n_init=_ANCESTRY_DEPTH_GMM_N_INIT,
            max_iter=_ANCESTRY_DEPTH_GMM_MAX_ITERATIONS,
            random_state=int(seed),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(standardized)
        if not model.converged_:
            fitted.append(None)
            bics.append(np.inf)
            continue
        fitted.append(model)
        bics.append(float(model.bic(standardized)))

    finite = np.flatnonzero(np.isfinite(bics))
    if not len(finite):
        posterior = np.zeros((len(counts), 1), dtype=np.float64)
        posterior[valid, 0] = 1.0
        return _AncestryDepthModel(
            adjusted,
            callability,
            posterior,
            np.asarray((center,)),
            np.asarray((scale,)),
            np.asarray((1.0,)),
            np.nan,
            tuple(float(value) for value in bics),
        )
    selected_index = int(finite[np.argmin(np.asarray(bics)[finite])])
    model = fitted[selected_index]
    component_order = np.argsort(model.means_[:, 0], kind="stable")
    standardized_means = model.means_[component_order, 0]
    standardized_variances = model.covariances_[component_order, 0, 0]
    weights = model.weights_[component_order]

    posterior = np.zeros((len(counts), len(component_order)), dtype=np.float64)
    valid_standardized = (adjusted[valid] - center) / scale
    log_density = (
        -0.5 * np.log(2.0 * math.pi * standardized_variances)[None, :]
        -0.5 * (
            valid_standardized[:, None] - standardized_means[None, :]
        ) ** 2 / standardized_variances[None, :]
    )
    log_scores = (
        np.log(weights)[None, :]
        + callability[valid, None] * log_density
    )
    log_scores -= np.max(log_scores, axis=1, keepdims=True)
    probabilities = np.exp(log_scores)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    posterior[valid] = probabilities
    return _AncestryDepthModel(
        adjusted,
        callability,
        posterior,
        center + scale * standardized_means,
        scale * np.sqrt(standardized_variances),
        weights,
        float(bics[selected_index]),
        tuple(float(value) for value in bics),
    )


def _parent_role_probability(
    row: int,
    alternatives: np.ndarray,
    depth_posterior: Optional[np.ndarray],
) -> float:
    """Probability all observed parents occupy an earlier latent depth."""
    parents = tuple(
        int(parent)
        for parent in alternatives[row, 1:]
        if int(parent) >= 0
    )
    if not parents:
        return 1.0
    if depth_posterior is None or depth_posterior.shape[1] < 2:
        return 0.0
    child = int(alternatives[row, 0])
    child_posterior = depth_posterior[child]
    if float(np.sum(child_posterior)) <= 0.0:
        return 0.0
    lower_depth_probability = (
        np.cumsum(depth_posterior, axis=1) - depth_posterior
    )
    probability = 0.0
    for depth, child_probability in enumerate(child_posterior):
        term = float(child_probability)
        for parent in parents:
            term *= float(lower_depth_probability[parent, depth])
        probability += term
    return float(np.clip(probability, 0.0, 1.0))


@dataclass(frozen=True)
class _ParentStateSelection:
    state_log_evidence: np.ndarray
    state_scores: np.ndarray
    state_support: np.ndarray
    decision_scores: np.ndarray
    fitted_prior_parameters: np.ndarray
    loo_state_priors: np.ndarray
    local_states: dict[int, int]
    local_rows: dict[int, int]
    graph_rows: dict[int, int]
    graph_tie_conflicts: frozenset[int]
    graph_direction_resolved_children: frozenset[int]
    graph_parent_role_probabilities: dict[int, float]
    ancestry_depth_model: Optional[_AncestryDepthModel]
    state_margins: np.ndarray
    identity_margins: np.ndarray
    unresolved_reasons: tuple[Optional[str], ...]


def _integrated_parent_state_log_evidence(
    aggregate_log_likelihoods: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
) -> np.ndarray:
    """Log-mean identity evidence, using the full eligible multiplicity."""
    evidence = np.full((len(by_child), 3), -np.inf, dtype=np.float64)
    for child, child_rows in enumerate(by_child):
        for state in range(3):
            rows = child_rows[states[child_rows] == state]
            if len(rows):
                evidence[child, state] = (
                    _logsumexp_finite(aggregate_log_likelihoods[rows])
                    - math.log(float(full_counts[child, state]))
                )
    return evidence


def _fit_hierarchical_parent_state_prior(
    state_log_evidence: np.ndarray,
    base_probabilities: Sequence[float],
    strength: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a weak Dirichlet mixture and return leave-one-child-out priors.

    A child's own responsibility is subtracted before its predictive state
    prior is formed. This prevents the evidence being evaluated from directly
    setting its own prior while allowing the cohort to share information about
    the prevalence of zero-, one-, and two-observed-parent states.
    """
    evidence = np.asarray(state_log_evidence, dtype=np.float64)
    base = np.asarray(base_probabilities, dtype=np.float64)
    alpha = strength * base
    active = np.any(np.isfinite(evidence), axis=1)
    responsibilities = np.zeros_like(evidence)
    mixture = base.copy()
    for _ in range(max_iterations):
        previous = mixture.copy()
        log_mixture = np.log(mixture)
        for child in np.flatnonzero(active):
            responsibilities[child] = _softmax_finite(
                evidence[child] + log_mixture
            )
        posterior = alpha + np.sum(responsibilities[active], axis=0)
        mixture = posterior / np.sum(posterior)
        if float(np.max(np.abs(mixture - previous))) <= tolerance:
            break

    posterior = alpha + np.sum(responsibilities[active], axis=0)
    loo_priors = np.tile(base, (len(evidence), 1))
    for child in np.flatnonzero(active):
        parameters = posterior - responsibilities[child]
        loo_priors[child] = parameters / np.sum(parameters)
    return posterior, loo_priors


def _parent_state_score_components(
    aggregate_log_likelihoods: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
    child_state_priors: np.ndarray,
    state_log_evidence: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build marginal state and conditional-identity decision evidence."""
    n_samples = len(by_child)
    if state_log_evidence is None:
        state_log_evidence = _integrated_parent_state_log_evidence(
            aggregate_log_likelihoods, states, by_child, full_counts
        )
    priors = np.asarray(child_state_priors, dtype=np.float64)
    if priors.shape == (3,):
        priors = np.tile(priors, (n_samples, 1))
    if (
        priors.shape != (n_samples, 3)
        or np.any(~np.isfinite(priors))
        or np.any(priors <= 0.0)
    ):
        raise SmartEvidenceError(
            "child parent-state priors must be positive with shape (samples, 3)"
        )
    state_scores = state_log_evidence + np.log(priors)
    state_support = np.zeros((n_samples, 3), dtype=np.float64)
    decision_scores = np.full(
        len(aggregate_log_likelihoods), -np.inf, dtype=np.float64
    )
    for child, child_rows in enumerate(by_child):
        for state in range(3):
            rows = child_rows[states[child_rows] == state]
            if not len(rows) or not np.isfinite(state_scores[child, state]):
                continue
            values = aggregate_log_likelihoods[rows]
            maximum = float(np.max(values))
            decision_scores[rows] = (
                state_scores[child, state] + values - maximum
            )
        state_support[child] = _softmax_finite(state_scores[child])
    return state_log_evidence, state_scores, state_support, decision_scores


def _local_parent_state_winners(
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    state_scores: np.ndarray,
    decision_scores: np.ndarray,
) -> tuple[
    dict[int, int],
    dict[int, int],
    np.ndarray,
    np.ndarray,
    tuple[Optional[str], ...],
]:
    """Choose the marginal state first, then an identity within that state."""
    n_samples = len(by_child)
    local_states = {}
    local_rows = {}
    state_margins = np.full(n_samples, np.nan, dtype=np.float64)
    identity_margins = np.full(n_samples, np.nan, dtype=np.float64)
    reasons: list[Optional[str]] = [None] * n_samples
    state_indices = np.arange(3, dtype=np.int64)
    for child, rows in enumerate(by_child):
        state, state_margin = _unique_finite_winner(
            state_indices, state_scores[child]
        )
        state_margins[child] = state_margin
        if state is None:
            reasons[child] = "no_unique_marginal_parent_state"
            continue
        state = int(state)
        local_states[child] = state
        state_rows = rows[states[rows] == state]
        row, identity_margin = _unique_finite_winner(
            state_rows, decision_scores
        )
        identity_margins[child] = identity_margin
        if row is None:
            reasons[child] = "parent_state_resolved_identity_unresolved"
            continue
        local_rows[child] = int(row)
    return (
        local_states,
        local_rows,
        state_margins,
        identity_margins,
        tuple(reasons),
    )


def _acyclic_parent_state_selection(
    alternatives: np.ndarray,
    states: np.ndarray,
    decision_scores: np.ndarray,
    by_child: Sequence[np.ndarray],
    local_rows: Mapping[int, int],
    state_margins: np.ndarray,
    identity_margins: np.ndarray,
    n_samples: int,
    local_search_passes: int,
    depth_posterior: Optional[np.ndarray] = None,
) -> _GraphParentStateSelection:
    """Choose a DAG without converting graph feasibility into parent evidence.

    A unique locally preferred row is retained whenever it is acyclic. Only
    when that row conflicts with an already supported direction may the graph
    consider another identity in the *same* parent-count state, and then only
    when the unsupervised painting-depth model gives MAP support that every
    proposed parent is earlier than the child. If no unique same-state row
    meets both conditions, the child falls back to explicit M0. Thus graph
    constraints can orient supported relatedness but cannot manufacture M1
    from M2, M2 from M1, or resolve a locally tied identity.
    """
    confidence = {}
    local_role_probability = {}
    for child, row in local_rows.items():
        margins = (state_margins[child], identity_margins[child])
        confidence[child] = min(
            value for value in margins if not np.isnan(value)
        )
        local_role_probability[child] = _parent_role_probability(
            row, alternatives, depth_posterior
        )
    child_order = sorted(
        local_rows,
        key=lambda child: (
            -local_role_probability[child],
            -confidence[child],
            child,
        ),
    )
    adjacency = [set() for _ in range(n_samples)]
    selected: dict[int, int] = {}
    direction_resolved: set[int] = set()
    role_probabilities: dict[int, float] = {}

    def observed_parents(row: int) -> tuple[int, ...]:
        return tuple(
            int(parent)
            for parent in alternatives[row, 1:]
            if int(parent) >= 0
        )

    def can_add(row: int) -> bool:
        child = int(alternatives[row, 0])
        return not any(
            _path_exists(adjacency, child, parent)
            for parent in observed_parents(row)
        )

    def add(row: int, displaced_local: bool = False) -> None:
        child = int(alternatives[row, 0])
        for parent in observed_parents(row):
            adjacency[parent].add(child)
        selected[child] = row
        role_probabilities[child] = _parent_role_probability(
            row, alternatives, depth_posterior
        )
        if displaced_local and int(states[row]) != _ZERO_OBSERVED:
            direction_resolved.add(child)
        else:
            direction_resolved.discard(child)

    def remove(child: int) -> Optional[int]:
        row = selected.pop(child, None)
        role_probabilities.pop(child, None)
        direction_resolved.discard(child)
        if row is not None:
            for parent in observed_parents(row):
                adjacency[parent].discard(child)
        return row

    def best_feasible(child: int) -> tuple[Optional[int], bool]:
        local = local_rows.get(child)
        if local is not None and can_add(local):
            return local, False
        if local is not None and depth_posterior is not None:
            local_state = int(states[local])
            eligible = []
            for row in by_child[child]:
                row = int(row)
                if (
                    row != local
                    and int(states[row]) == local_state
                    and np.isfinite(decision_scores[row])
                    and can_add(row)
                ):
                    role_probability = _parent_role_probability(
                        row, alternatives, depth_posterior
                    )
                    tolerance = _contrast_tolerance(np.asarray(
                        (role_probability, 0.5), dtype=np.float64
                    ))
                    if role_probability > 0.5 + tolerance:
                        eligible.append(row)
            if eligible:
                winner, _ = _unique_finite_winner(
                    np.asarray(eligible, dtype=np.int64), decision_scores
                )
                if winner is not None:
                    return winner, True
        zero_rows = [
            int(row)
            for row in by_child[child]
            if int(states[row]) == _ZERO_OBSERVED
        ]
        if len(zero_rows) != 1:
            raise SmartEvidenceError(
                "each child must have exactly one zero-observed-parent row"
            )
        zero = zero_rows[0]
        return (
            (zero, local is not None and zero != local)
            if np.isfinite(decision_scores[zero])
            else (None, False)
        )

    for child in child_order:
        row, displaced = best_feasible(child)
        if row is not None:
            add(row, displaced)
    for _ in range(local_search_passes):
        changed = False
        for child in child_order:
            previous = remove(child)
            replacement, displaced = best_feasible(child)
            if replacement is not None:
                add(replacement, displaced)
            changed |= replacement != previous
        if not changed:
            break
    return _GraphParentStateSelection(
        selected,
        frozenset(direction_resolved),
        role_probabilities,
    )


def _graph_tie_conflict_children(
    alternatives: np.ndarray,
    local_rows: Mapping[int, int],
    state_margins: np.ndarray,
    identity_margins: np.ndarray,
    n_samples: int,
) -> frozenset[int]:
    """Find tied local rows that a DAG cannot resolve without arbitrariness.

    Cycles are peeled at their least-supported child configuration.  A unique
    weakest row can be left for the normal DAG optimizer to displace.  If two
    or more weakest rows are numerically tied, all are marked ambiguous and
    excluded from graph selection; selecting one by sample-array order would
    manufacture exact confidence from a graph constraint.
    """
    active = {int(child): int(row) for child, row in local_rows.items()}
    ambiguous: set[int] = set()

    def observed_parents(row: int) -> tuple[int, ...]:
        return tuple(
            int(parent)
            for parent in alternatives[row, 1:]
            if int(parent) >= 0
        )

    def confidence(child: int) -> float:
        values = (
            float(state_margins[child]),
            float(identity_margins[child]),
        )
        return min(value for value in values if not np.isnan(value))

    def cyclic_components() -> list[frozenset[int]]:
        adjacency = [set() for _ in range(n_samples)]
        reverse = [set() for _ in range(n_samples)]
        for child, row in active.items():
            for parent in observed_parents(row):
                adjacency[parent].add(child)
                reverse[child].add(parent)

        def reachable(graph: Sequence[set[int]], start: int) -> set[int]:
            seen = {start}
            stack = [start]
            while stack:
                node = stack.pop()
                for neighbour in graph[node]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            return seen

        remaining = set(range(n_samples))
        components = []
        while remaining:
            start = min(remaining)
            component = reachable(adjacency, start) & reachable(reverse, start)
            remaining.difference_update(component)
            if len(component) > 1:
                components.append(frozenset(component))
        return components

    while True:
        components = cyclic_components()
        if not components:
            break
        removed = set()
        for component in components:
            implicated = [
                child
                for child, row in active.items()
                if child in component
                and any(parent in component for parent in observed_parents(row))
            ]
            if not implicated:
                continue
            values = np.asarray(
                [confidence(child) for child in implicated], dtype=np.float64
            )
            minimum = float(np.min(values))
            if np.isfinite(minimum):
                finite = values[np.isfinite(values)]
                tolerance = _contrast_tolerance(finite)
                tied = [
                    child for child, value in zip(implicated, values)
                    if np.isfinite(value) and abs(float(value) - minimum) <= tolerance
                ]
            else:
                tied = [
                    child for child, value in zip(implicated, values)
                    if float(value) == minimum
                ]
            if len(tied) > 1:
                ambiguous.update(tied)
                removed.update(tied)
            else:
                removed.add(tied[0])
        if not removed:
            raise SmartEvidenceError("failed to peel a cyclic local pedigree")
        for child in removed:
            active.pop(child, None)
    return frozenset(ambiguous)



def _evaluate_parent_state_aggregate(
    aggregate_log_likelihoods: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
    base_priors: Sequence[float],
    prior_strength: float,
    prior_max_iterations: int,
    prior_tolerance: float,
    n_samples: int,
    local_search_passes: int,
    ancestry_depth_model: Optional[_AncestryDepthModel] = None,
) -> _ParentStateSelection:
    state_log_evidence = _integrated_parent_state_log_evidence(
        aggregate_log_likelihoods, states, by_child, full_counts
    )
    fitted_parameters, loo_state_priors = (
        _fit_hierarchical_parent_state_prior(
            state_log_evidence,
            base_priors,
            prior_strength,
            prior_max_iterations,
            prior_tolerance,
        )
    )
    components = _parent_state_score_components(
        aggregate_log_likelihoods,
        states,
        by_child,
        full_counts,
        loo_state_priors,
        state_log_evidence,
    )
    (
        state_log_evidence,
        state_scores,
        state_support,
        decision_scores,
    ) = components
    (
        local_states,
        local_rows,
        state_margins,
        identity_margins,
        unresolved_reasons,
    ) = _local_parent_state_winners(
        states, by_child, state_scores, decision_scores
    )
    graph_tie_conflicts = _graph_tie_conflict_children(
        alternatives,
        local_rows,
        state_margins,
        identity_margins,
        n_samples,
    )
    graph_eligible_rows = {
        child: row
        for child, row in local_rows.items()
        if child not in graph_tie_conflicts
    }
    graph_selection = _acyclic_parent_state_selection(
        alternatives,
        states,
        decision_scores,
        by_child,
        graph_eligible_rows,
        state_margins,
        identity_margins,
        n_samples,
        local_search_passes,
        (
            None
            if ancestry_depth_model is None
            else ancestry_depth_model.posterior
        ),
    )
    return _ParentStateSelection(
        state_log_evidence=state_log_evidence,
        state_scores=state_scores,
        state_support=state_support,
        decision_scores=decision_scores,
        fitted_prior_parameters=fitted_parameters,
        loo_state_priors=loo_state_priors,
        local_states=local_states,
        local_rows=local_rows,
        graph_rows=graph_selection.rows,
        graph_tie_conflicts=graph_tie_conflicts,
        graph_direction_resolved_children=(
            graph_selection.direction_resolved_children
        ),
        graph_parent_role_probabilities=(
            graph_selection.selected_parent_role_probabilities
        ),
        ancestry_depth_model=ancestry_depth_model,
        state_margins=state_margins,
        identity_margins=identity_margins,
        unresolved_reasons=unresolved_reasons,
    )


def _parent_state_frame(
    sample_ids: Sequence[Any],
    alternatives: np.ndarray,
    states: np.ndarray,
    rows_by_child: Mapping[int, Optional[int]],
    state_by_child: Mapping[int, int],
    status_by_child: Mapping[int, str],
) -> pd.DataFrame:
    rows = []
    for child, sample in enumerate(sample_ids):
        selected = rows_by_child.get(child)
        state = state_by_child.get(child)
        first = second = None
        if selected is not None:
            first_index = int(alternatives[selected, 1])
            second_index = int(alternatives[selected, 2])
            first = None if first_index < 0 else sample_ids[first_index]
            second = None if second_index < 0 else sample_ids[second_index]
        rows.append({
            "Sample": sample,
            "Generation": "Unknown",
            "Parent1": first,
            "Parent2": second,
            "ParentState": (
                "unresolved" if state is None else _PARENT_STATE_NAMES[state]
            ),
            "ObservedParentCount": (
                np.nan if state is None else int(state)
            ),
            "InferenceStatus": status_by_child.get(child, "unresolved"),
        })
    return pd.DataFrame(rows)


def _configuration_support_text(
    child_rows: np.ndarray,
    counts: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    sample_ids: Sequence[Any],
    coverage: float,
) -> str:
    total = float(np.sum(counts[child_rows]))
    if total <= 0.0:
        return ""
    ordered = child_rows[np.argsort(-counts[child_rows], kind="stable")]
    pieces = []
    cumulative = 0.0
    for row in ordered:
        probability = float(counts[row] / total)
        state = int(states[row])
        parents = [
            str(sample_ids[int(parent)])
            for parent in alternatives[row, 1:]
            if int(parent) >= 0
        ]
        label = "+".join(parents) if parents else "external+external"
        pieces.append(
            f"{_PARENT_STATE_NAMES[state]}:{label}:{probability:.3f}"
        )
        cumulative += probability
        if cumulative >= coverage:
            break
    return ";".join(pieces)


_SMART_BOOTSTRAP_SHARED: dict[str, Any] = {}
_SMART_BOOTSTRAP_SHM_REFS: list[Any] = []
# More forkserver processes were slower on the representative 1,000-replicate
# workload because process/import overhead dominates these small GMM fits.
# The 1,000-replicate full simulated panel is memory/IPC-bound: a bounded
# 112-core sweep was fastest and repeatable at 42 one-thread workers, while
# both fewer workers and an all-core process pool were slower. Smaller CPU
# allocations are still clipped below this cap by ``_run_parent_state_bootstraps``.
_SMART_BOOTSTRAP_MAX_PROCESSES = 42


def _init_smart_bootstrap_worker(shared: Mapping[str, Any]) -> None:
    """Attach read-only bootstrap arrays once in each forkserver worker."""
    global _SMART_BOOTSTRAP_SHARED, _SMART_BOOTSTRAP_SHM_REFS
    for handle in _SMART_BOOTSTRAP_SHM_REFS:
        try:
            handle.close()
        except Exception:
            pass
    _SMART_BOOTSTRAP_SHM_REFS = []
    _SMART_BOOTSTRAP_SHARED = {}
    for key, value in shared.items():
        if isinstance(value, Mapping) and "shm_name" in value:
            handle, array = _legacy._attach_shm_view(value)
            _SMART_BOOTSTRAP_SHM_REFS.append(handle)
            _SMART_BOOTSTRAP_SHARED[key] = array
        else:
            _SMART_BOOTSTRAP_SHARED[key] = value


def _bootstrap_selection_requires_depth(
    selection: _ParentStateSelection,
) -> bool:
    """Whether ancestry depth can change this replicate's selected rows.

    When every non-tied local row is already jointly acyclic, adding those
    edges in any order produces the same graph. The depth model can then alter
    only unused ordering diagnostics, not any bootstrap count accumulated by
    this function.
    """
    expected = {
        child: row
        for child, row in selection.local_rows.items()
        if child not in selection.graph_tie_conflicts
    }
    return selection.graph_rows != expected


def _evaluate_smart_bootstrap_chunk(
    shared: Mapping[str, Any],
    multiplicities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Evaluate a deterministic chunk of chromosome bootstrap replicates."""
    alternatives = shared["alternatives"]
    states = shared["states"]
    contig_log_likelihoods = shared["contig_log_likelihoods"]
    by_child = shared["by_child"]
    full_counts = shared["full_counts"]
    junction_matrix = shared["junction_matrix"]
    callable_matrix = shared["callable_matrix"]
    n_samples = int(shared["n_samples"])
    n_replicates = len(multiplicities)
    local_rows = np.full((n_replicates, n_samples), -1, dtype=np.int64)
    graph_rows = np.full((n_replicates, n_samples), -1, dtype=np.int64)
    local_states = np.full((n_replicates, n_samples), -1, dtype=np.int8)
    graph_states = np.full((n_replicates, n_samples), -1, dtype=np.int8)
    depth_refits = 0

    def evaluate(
        aggregate: np.ndarray,
        depth_model: Optional[_AncestryDepthModel],
    ) -> _ParentStateSelection:
        return _evaluate_parent_state_aggregate(
            aggregate,
            alternatives,
            states,
            by_child,
            full_counts,
            shared["base_priors"],
            float(shared["prior_strength"]),
            int(shared["prior_max_iterations"]),
            float(shared["prior_tolerance"]),
            n_samples,
            int(shared["local_search_passes"]),
            depth_model,
        )

    for replicate, multiplicity in enumerate(multiplicities):
        aggregate = multiplicity @ contig_log_likelihoods
        selection = evaluate(aggregate, None)
        if (
            junction_matrix is not None
            and _bootstrap_selection_requires_depth(selection)
        ):
            depth_model = _fit_ancestry_depth_model(
                multiplicity @ junction_matrix,
                multiplicity @ callable_matrix,
                int(shared["bootstrap_seed"]),
            )
            selection = evaluate(aggregate, depth_model)
            depth_refits += 1
        for child, state in selection.local_states.items():
            local_states[replicate, child] = state
        for child, row in selection.local_rows.items():
            local_rows[replicate, child] = row
        for child, row in selection.graph_rows.items():
            graph_rows[replicate, child] = row
            graph_states[replicate, child] = states[row]
    return (
        local_rows,
        graph_rows,
        local_states,
        graph_states,
        depth_refits,
    )


def _smart_bootstrap_worker(
    multiplicities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Module-scope forkserver callback for one bootstrap chunk."""
    return _evaluate_smart_bootstrap_chunk(
        _SMART_BOOTSTRAP_SHARED, multiplicities
    )


def _accumulate_smart_bootstrap_chunk(
    chunk: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int],
    alternatives: np.ndarray,
    local_configuration_counts: np.ndarray,
    graph_configuration_counts: np.ndarray,
    local_state_counts: np.ndarray,
    graph_state_counts: np.ndarray,
    local_parent_counts: np.ndarray,
    graph_parent_counts: np.ndarray,
) -> int:
    """Reduce one worker result with order-independent integer additions."""
    local_rows, graph_rows, local_states, graph_states, depth_refits = chunk
    n_replicates, n_samples = local_rows.shape
    children = np.broadcast_to(
        np.arange(n_samples, dtype=np.int64),
        (n_replicates, n_samples),
    )

    for values, counts in (
        (local_states, local_state_counts),
        (graph_states, graph_state_counts),
    ):
        valid = values >= 0
        np.add.at(counts, (children[valid], values[valid]), 1)

    for rows, configuration_counts, parent_counts in (
        (local_rows, local_configuration_counts, local_parent_counts),
        (graph_rows, graph_configuration_counts, graph_parent_counts),
    ):
        valid = rows >= 0
        selected_rows = rows[valid]
        selected_children = children[valid]
        np.add.at(configuration_counts, selected_rows, 1)
        selected_alternatives = alternatives[selected_rows]
        for slot in (1, 2):
            parents = selected_alternatives[:, slot]
            observed = parents >= 0
            np.add.at(
                parent_counts,
                (selected_children[observed], parents[observed]),
                1,
            )
    return int(depth_refits)


def _run_parent_state_bootstraps(
    contig_log_likelihoods: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
    junction_matrix: Optional[np.ndarray],
    callable_matrix: Optional[np.ndarray],
    settings: SmartPedigreeConfig,
    n_workers: Optional[int],
    local_configuration_counts: np.ndarray,
    graph_configuration_counts: np.ndarray,
    local_state_counts: np.ndarray,
    graph_state_counts: np.ndarray,
    local_parent_counts: np.ndarray,
    graph_parent_counts: np.ndarray,
) -> tuple[int, int]:
    """Run fixed-seed bootstraps serially or in a shared-memory pool."""
    n_contigs = contig_log_likelihoods.shape[0]
    rng = np.random.default_rng(settings.bootstrap_seed)
    multiplicities = np.empty(
        (settings.bootstrap_replicates, n_contigs), dtype=np.float64
    )
    for replicate in range(settings.bootstrap_replicates):
        draws = rng.integers(0, n_contigs, size=n_contigs)
        multiplicities[replicate] = np.bincount(
            draws, minlength=n_contigs
        ).astype(np.float64)

    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = os.cpu_count() or 1
    capacity = (
        int(numba.config.NUMBA_NUM_THREADS) if SMART_HAS_NUMBA else 1
    )
    if n_workers is None:
        cpu_budget = min(available_cpus, capacity)
    else:
        if int(n_workers) != n_workers or n_workers < 1:
            raise SmartEvidenceError("n_workers must be a positive integer")
        cpu_budget = min(int(n_workers), available_cpus, capacity)
    use_pool = bool(
        cpu_budget > 1
        and settings.bootstrap_replicates >= 32
        and len(alternatives) >= 100_000
    )
    worker_count = (
        min(
            cpu_budget,
            settings.bootstrap_replicates,
            _SMART_BOOTSTRAP_MAX_PROCESSES,
        )
        if use_pool
        else 1
    )

    ordinary_shared = {
        "by_child": tuple(np.asarray(rows, dtype=np.int64) for rows in by_child),
        "base_priors": tuple(settings.parent_state_priors),
        "prior_strength": settings.parent_state_prior_strength,
        "prior_max_iterations": settings.parent_state_prior_max_iterations,
        "prior_tolerance": settings.parent_state_prior_tolerance,
        "n_samples": len(by_child),
        "local_search_passes": settings.dag_local_search_passes,
        "bootstrap_seed": settings.bootstrap_seed,
    }
    depth_refits = 0
    if worker_count == 1:
        shared = {
            **ordinary_shared,
            "contig_log_likelihoods": contig_log_likelihoods,
            "alternatives": alternatives,
            "states": states,
            "full_counts": full_counts,
            "junction_matrix": junction_matrix,
            "callable_matrix": callable_matrix,
        }
        results = (_evaluate_smart_bootstrap_chunk(shared, multiplicities),)
        for chunk in results:
            depth_refits += _accumulate_smart_bootstrap_chunk(
                chunk,
                alternatives,
                local_configuration_counts,
                graph_configuration_counts,
                local_state_counts,
                graph_state_counts,
                local_parent_counts,
                graph_parent_counts,
            )
        return worker_count, depth_refits

    handles = []
    shared = dict(ordinary_shared)
    for key, array in (
        ("contig_log_likelihoods", contig_log_likelihoods),
        ("alternatives", alternatives),
        ("states", states),
        ("full_counts", full_counts),
        ("junction_matrix", junction_matrix),
        ("callable_matrix", callable_matrix),
    ):
        if array is None:
            shared[key] = None
        else:
            try:
                handle, metadata = _legacy._create_shm_array(array)
            except BaseException:
                with _legacy._shm_cleanup(handles):
                    pass
                raise
            handles.append(handle)
            shared[key] = metadata

    chunk_size = max(
        1,
        int(math.ceil(
            settings.bootstrap_replicates / float(worker_count * 4)
        )),
    )
    tasks = [
        np.ascontiguousarray(multiplicities[start:start + chunk_size])
        for start in range(0, settings.bootstrap_replicates, chunk_size)
    ]
    with _legacy._shm_cleanup(handles), _legacy._safe_forkserver_pool(
        worker_count,
        initializer=_init_smart_bootstrap_worker,
        initargs=(shared,),
    ) as pool:
        for chunk in pool.imap_unordered(
            _smart_bootstrap_worker, tasks, chunksize=1
        ):
            depth_refits += _accumulate_smart_bootstrap_chunk(
                chunk,
                alternatives,
                local_configuration_counts,
                graph_configuration_counts,
                local_state_counts,
                graph_state_counts,
                local_parent_counts,
                graph_parent_counts,
            )
    return worker_count, depth_refits


def infer_from_parent_state_evidence(
    evidence: Sequence[SmartParentStateEvidence],
    sample_ids: Sequence[Any],
    config: Optional[SmartPedigreeConfig] = None,
    *,
    ancestry_junction_counts: Optional[np.ndarray] = None,
    ancestry_callable_haplotype_bins: Optional[np.ndarray] = None,
    n_workers: Optional[int] = None,
) -> PedigreeResult:
    """Infer a DAG from comparable 0/1/2-observed-parent likelihoods.

    State evidence is integrated over candidate identities only after contig
    log likelihoods have been summed. A weak Dirichlet cohort prior is fitted
    leave-one-child-out. Parent-count, conditional identity, and graph support
    are resampled separately so the DAG cannot create biological confidence.
    Optional per-contig ancestry-junction and callability matrices activate the
    metadata-free relative-depth model; they must have shape
    ``(len(evidence), len(sample_ids))`` and be supplied together.
    """
    settings = (config or SmartPedigreeConfig()).validated()
    samples = list(sample_ids)
    n_samples = len(samples)
    if n_samples < 3 or len(set(samples)) != n_samples:
        raise SmartEvidenceError(
            "sample_ids must contain at least three unique IDs"
        )
    (
        trios,
        zero,
        one,
        two,
        markers,
        contig_names,
    ) = _canonical_parent_state_evidence(evidence, n_samples)
    if (
        ancestry_junction_counts is None
        or ancestry_callable_haplotype_bins is None
    ):
        if not (
            ancestry_junction_counts is None
            and ancestry_callable_haplotype_bins is None
        ):
            raise SmartEvidenceError(
                "ancestry junction counts and callable haplotype bins must "
                "be supplied together"
            )
        junction_matrix = callable_matrix = None
    else:
        junction_matrix = np.asarray(
            ancestry_junction_counts, dtype=np.float64
        )
        callable_matrix = np.asarray(
            ancestry_callable_haplotype_bins, dtype=np.float64
        )
        expected_shape = (len(contig_names), n_samples)
        if (
            junction_matrix.shape != expected_shape
            or callable_matrix.shape != expected_shape
            or np.any(~np.isfinite(junction_matrix))
            or np.any(~np.isfinite(callable_matrix))
            or np.any(junction_matrix < 0.0)
            or np.any(callable_matrix < 0.0)
        ):
            raise SmartEvidenceError(
                "per-contig ancestry junction counts and callability must be "
                f"finite, non-negative arrays with shape {expected_shape}"
            )
    (
        alternatives,
        states,
        contig_log_likelihoods,
        by_child,
        full_counts,
        scored_counts,
    ) = _parent_state_alternatives(
        trios,
        zero,
        one,
        two,
        settings.parent_state_contamination_probability,
    )

    def evaluate(
        aggregate: np.ndarray,
        base_priors: Sequence[float] = settings.parent_state_priors,
        depth_model: Optional[_AncestryDepthModel] = None,
    ) -> _ParentStateSelection:
        return _evaluate_parent_state_aggregate(
            aggregate,
            alternatives,
            states,
            by_child,
            full_counts,
            base_priors,
            settings.parent_state_prior_strength,
            settings.parent_state_prior_max_iterations,
            settings.parent_state_prior_tolerance,
            n_samples,
            settings.dag_local_search_passes,
            depth_model,
        )

    full_aggregate = np.sum(contig_log_likelihoods, axis=0)
    full_depth_model = (
        None
        if junction_matrix is None
        else _fit_ancestry_depth_model(
            np.sum(junction_matrix, axis=0),
            np.sum(callable_matrix, axis=0),
            settings.bootstrap_seed,
        )
    )
    full_selection = evaluate(
        full_aggregate, depth_model=full_depth_model
    )

    informative = np.zeros(
        (len(contig_names), n_samples), dtype=np.bool_
    )
    base_log_prior = np.log(
        np.asarray(settings.parent_state_priors, dtype=np.float64)
    )
    state_indices = np.arange(3, dtype=np.int64)
    for contig_index in range(len(contig_names)):
        contig_state_evidence = _integrated_parent_state_log_evidence(
            contig_log_likelihoods[contig_index],
            states,
            by_child,
            full_counts,
        ) + base_log_prior
        for child in range(n_samples):
            winner, _ = _unique_finite_winner(
                state_indices, contig_state_evidence[child]
            )
            informative[contig_index, child] = winner is not None

    n_alternatives = len(alternatives)
    local_configuration_counts = np.zeros(
        n_alternatives, dtype=np.int64
    )
    graph_configuration_counts = np.zeros(
        n_alternatives, dtype=np.int64
    )
    local_state_counts = np.zeros((n_samples, 3), dtype=np.int64)
    graph_state_counts = np.zeros((n_samples, 3), dtype=np.int64)
    local_parent_counts = np.zeros(
        (n_samples, n_samples), dtype=np.int64
    )
    graph_parent_counts = np.zeros(
        (n_samples, n_samples), dtype=np.int64
    )

    def accumulate(
        selection: _ParentStateSelection,
        configuration_counts: np.ndarray,
        state_counts: np.ndarray,
        parent_counts: np.ndarray,
        graph: bool,
    ) -> None:
        selected_rows = selection.graph_rows if graph else selection.local_rows
        selected_states = (
            {
                child: int(states[row])
                for child, row in selection.graph_rows.items()
            }
            if graph
            else selection.local_states
        )
        for child, state in selected_states.items():
            state_counts[child, state] += 1
        for child, row in selected_rows.items():
            configuration_counts[row] += 1
            for parent in alternatives[row, 1:]:
                parent = int(parent)
                if parent >= 0:
                    parent_counts[child, parent] += 1

    bootstrap_worker_count, bootstrap_depth_refits = (
        _run_parent_state_bootstraps(
            contig_log_likelihoods,
            alternatives,
            states,
            by_child,
            full_counts,
            junction_matrix,
            callable_matrix,
            settings,
            n_workers,
            local_configuration_counts,
            graph_configuration_counts,
            local_state_counts,
            graph_state_counts,
            local_parent_counts,
            graph_parent_counts,
        )
    )

    loco_local_configuration_counts = np.zeros(
        n_alternatives, dtype=np.int64
    )
    loco_graph_configuration_counts = np.zeros(
        n_alternatives, dtype=np.int64
    )
    loco_local_state_counts = np.zeros(
        (n_samples, 3), dtype=np.int64
    )
    loco_graph_state_counts = np.zeros(
        (n_samples, 3), dtype=np.int64
    )
    loco_local_parent_counts = np.zeros(
        (n_samples, n_samples), dtype=np.int64
    )
    loco_graph_parent_counts = np.zeros(
        (n_samples, n_samples), dtype=np.int64
    )
    n_loco = 0
    if len(contig_names) > 1:
        for omitted in range(len(contig_names)):
            loco_depth_model = (
                None
                if junction_matrix is None
                else _fit_ancestry_depth_model(
                    np.sum(junction_matrix, axis=0)
                    - junction_matrix[omitted],
                    np.sum(callable_matrix, axis=0)
                    - callable_matrix[omitted],
                    settings.bootstrap_seed,
                )
            )
            selection = evaluate(
                full_aggregate - contig_log_likelihoods[omitted],
                depth_model=loco_depth_model,
            )
            n_loco += 1
            accumulate(
                selection,
                loco_local_configuration_counts,
                loco_local_state_counts,
                loco_local_parent_counts,
                False,
            )
            accumulate(
                selection,
                loco_graph_configuration_counts,
                loco_graph_state_counts,
                loco_graph_parent_counts,
                True,
            )

    sensitivity_runs = []
    sensitivity_summary_rows = []
    for base_priors in settings.parent_state_prior_sensitivity:
        selection = evaluate(
            full_aggregate, base_priors, full_depth_model
        )
        sensitivity_runs.append((base_priors, selection))
        state_call_counts = [
            sum(
                int(state == target)
                for state in selection.local_states.values()
            )
            for target in range(3)
        ]
        sensitivity_summary_rows.append({
            "BasePrior0": base_priors[0],
            "BasePrior1": base_priors[1],
            "BasePrior2": base_priors[2],
            "FittedDirichletParameter0": (
                selection.fitted_prior_parameters[0]
            ),
            "FittedDirichletParameter1": (
                selection.fitted_prior_parameters[1]
            ),
            "FittedDirichletParameter2": (
                selection.fitted_prior_parameters[2]
            ),
            "LocalZeroObservedCalls": state_call_counts[0],
            "LocalOneObservedCalls": state_call_counts[1],
            "LocalTwoObservedCalls": state_call_counts[2],
            "LocalIdentityResolvedCalls": len(selection.local_rows),
            "GraphConfigurationCalls": len(selection.graph_rows),
        })

    bootstrap_denominator = float(settings.bootstrap_replicates)
    loco_denominator = float(n_loco) if n_loco else np.nan
    complete_rows: dict[int, Optional[int]] = {}
    complete_states: dict[int, int] = {}
    complete_status = {}
    tier_a_rows: dict[int, Optional[int]] = {}
    tier_b_rows: dict[int, Optional[int]] = {}
    tier_a_states: dict[int, int] = {}
    tier_b_states: dict[int, int] = {}
    tier_a_status = {}
    tier_b_status = {}
    tier_a_parent_flags: dict[int, tuple[bool, bool]] = {}
    tier_b_parent_flags: dict[int, tuple[bool, bool]] = {}
    diagnostics = []
    state_call_rows = []
    trio_candidates = {}
    parent_candidates = {}
    trio_scores = {}

    def stable_fraction(count: int, denominator: float) -> float:
        if not np.isfinite(denominator) or denominator <= 0.0:
            return 0.0
        return float(count / denominator)

    for child, sample in enumerate(samples):
        child_rows = by_child[child]
        local_state = full_selection.local_states.get(child)
        local_row = full_selection.local_rows.get(child)
        graph_row = full_selection.graph_rows.get(child)
        graph_tie_conflict = child in full_selection.graph_tie_conflicts
        graph_direction_resolved = (
            child in full_selection.graph_direction_resolved_children
        )
        selected_parent_role_probability = (
            full_selection.graph_parent_role_probabilities.get(child, np.nan)
        )
        local_parent_role_probability = (
            np.nan
            if local_row is None
            else _parent_role_probability(
                local_row,
                alternatives,
                (
                    None
                    if full_depth_model is None
                    else full_depth_model.posterior
                ),
            )
        )
        graph_conflict = bool(
            graph_tie_conflict
            or (local_row is not None and graph_row != local_row)
        )
        graph_displaced = (
            local_row is not None
            and graph_row is not None
            and graph_row != local_row
        )
        graph_state = (
            None if graph_row is None else int(states[graph_row])
        )
        informative_count = int(np.count_nonzero(informative[:, child]))
        enough_contigs = (
            informative_count >= settings.minimum_informative_contigs
        )

        if local_state is None:
            state_bootstrap = state_loco = np.nan
        else:
            state_bootstrap = stable_fraction(
                local_state_counts[child, local_state],
                bootstrap_denominator,
            )
            state_loco = stable_fraction(
                loco_local_state_counts[child, local_state],
                loco_denominator,
            )
        if local_row is None:
            local_configuration_bootstrap = np.nan
            graph_configuration_bootstrap = np.nan
            local_configuration_loco = np.nan
            graph_configuration_loco = np.nan
            selected_parents: tuple[int, ...] = ()
        else:
            local_configuration_bootstrap = stable_fraction(
                local_configuration_counts[local_row],
                bootstrap_denominator,
            )
            graph_configuration_bootstrap = stable_fraction(
                graph_configuration_counts[local_row],
                bootstrap_denominator,
            )
            local_configuration_loco = stable_fraction(
                loco_local_configuration_counts[local_row],
                loco_denominator,
            )
            graph_configuration_loco = stable_fraction(
                loco_graph_configuration_counts[local_row],
                loco_denominator,
            )
            selected_parents = tuple(
                int(parent)
                for parent in alternatives[local_row, 1:]
                if int(parent) >= 0
            )

        parent_bootstrap = [
            stable_fraction(
                local_parent_counts[child, parent], bootstrap_denominator
            )
            for parent in selected_parents
        ]
        parent_loco = [
            stable_fraction(
                loco_local_parent_counts[child, parent], loco_denominator
            )
            for parent in selected_parents
        ]
        first_bootstrap = (
            parent_bootstrap[0] if parent_bootstrap else np.nan
        )
        second_bootstrap = (
            parent_bootstrap[1] if len(parent_bootstrap) > 1 else np.nan
        )
        first_loco = parent_loco[0] if parent_loco else np.nan
        second_loco = parent_loco[1] if len(parent_loco) > 1 else np.nan

        def tier_decision(
            state_bootstrap_cutoff: float,
            parent_bootstrap_cutoff: float,
            loco_cutoff: float,
        ) -> tuple[bool, bool, tuple[bool, bool]]:
            state_pass = bool(
                local_state is not None
                and enough_contigs
                and state_bootstrap >= state_bootstrap_cutoff
                and state_loco >= loco_cutoff
            )
            parent_flags = tuple(
                bool(
                    state_pass
                    and parent_bootstrap[index]
                    >= parent_bootstrap_cutoff
                    and parent_loco[index] >= loco_cutoff
                )
                for index in range(len(selected_parents))
            )
            identity_pass = bool(
                state_pass
                and local_row is not None
                and graph_row == local_row
                and not graph_tie_conflict
                and local_configuration_bootstrap
                >= state_bootstrap_cutoff
                and graph_configuration_bootstrap
                >= state_bootstrap_cutoff
                and local_configuration_loco >= loco_cutoff
                and graph_configuration_loco >= loco_cutoff
                and all(parent_flags)
            )
            padded_flags = (
                parent_flags[0] if len(parent_flags) > 0 else False,
                parent_flags[1] if len(parent_flags) > 1 else False,
            )
            return state_pass, identity_pass, padded_flags

        tier_a_state_pass, tier_a_identity_pass, tier_a_flags = tier_decision(
            settings.tier_a_pair_bootstrap,
            settings.tier_a_parent_bootstrap,
            settings.tier_a_loco_fraction,
        )
        tier_b_state_pass, tier_b_identity_pass, tier_b_flags = tier_decision(
            settings.tier_b_pair_bootstrap,
            settings.tier_b_parent_bootstrap,
            settings.tier_b_loco_fraction,
        )
        tier_a_parent_flags[child] = tier_a_flags
        tier_b_parent_flags[child] = tier_b_flags

        if graph_tie_conflict:
            complete_rows[child] = None
            complete_states[child] = local_state
            complete_status[child] = (
                "graph_tie_conflict_parent_identity_unresolved"
            )
        elif graph_row is not None:
            complete_rows[child] = graph_row
            complete_states[child] = graph_state
            if graph_row == local_row:
                complete_status[child] = (
                    f"selected_{_PARENT_STATE_NAMES[graph_state]}"
                )
            elif graph_direction_resolved:
                complete_status[child] = (
                    "graph_displaced_direction_resolved_same_state_"
                    "hypothesis_not_tier_eligible"
                )
            else:
                complete_status[child] = (
                    "graph_displaced_hypothesis_not_tier_eligible"
                )
        elif local_state is not None:
            complete_rows[child] = None
            complete_states[child] = local_state
            complete_status[child] = (
                full_selection.unresolved_reasons[child]
                or "parent_state_resolved_identity_unresolved"
            )
        else:
            complete_rows[child] = None
            complete_status[child] = (
                full_selection.unresolved_reasons[child]
                or "unresolved_parent_state"
            )

        def populate_tier(
            state_pass: bool,
            identity_pass: bool,
            tier_rows: dict[int, Optional[int]],
            tier_states: dict[int, int],
            tier_status: dict[int, str],
            label: str,
        ) -> None:
            if not state_pass or local_state is None:
                tier_rows[child] = None
                tier_status[child] = f"unresolved_below_{label}_state_support"
                return
            tier_states[child] = local_state
            if identity_pass:
                tier_rows[child] = local_row
                tier_status[child] = (
                    f"{label}_supported_{_PARENT_STATE_NAMES[local_state]}"
                )
            else:
                tier_rows[child] = None
                suffix = (
                    "graph_tie_conflict"
                    if graph_tie_conflict
                    else (
                        "graph_conflict"
                        if graph_conflict
                        else "identity_unresolved"
                    )
                )
                tier_status[child] = (
                    f"{label}_parent_state_supported_{suffix}"
                )

        populate_tier(
            tier_a_state_pass,
            tier_a_identity_pass,
            tier_a_rows,
            tier_a_states,
            tier_a_status,
            "tier_a",
        )
        populate_tier(
            tier_b_state_pass,
            tier_b_identity_pass,
            tier_b_rows,
            tier_b_states,
            tier_b_status,
            "tier_b",
        )

        sensitivity_state_agreement = []
        sensitivity_identity_agreement = []
        sensitivity_graph_agreement = []
        sensitivity_selected_support = []
        for _, selection in sensitivity_runs:
            sensitivity_state_agreement.append(
                selection.local_states.get(child) == local_state
            )
            sensitivity_identity_agreement.append(
                selection.local_rows.get(child) == local_row
            )
            sensitivity_graph_agreement.append(
                selection.graph_rows.get(child) == graph_row
            )
            if local_state is not None:
                sensitivity_selected_support.append(
                    selection.state_support[child, local_state]
                )

        graph_parents = (
            ()
            if graph_row is None
            else tuple(
                int(parent)
                for parent in alternatives[graph_row, 1:]
                if int(parent) >= 0
            )
        )
        selected_utility = (
            np.nan
            if graph_row is None
            else float(full_selection.decision_scores[graph_row])
        )
        selected_graph_configuration_bootstrap = (
            np.nan
            if graph_row is None
            else stable_fraction(
                graph_configuration_counts[graph_row],
                bootstrap_denominator,
            )
        )
        selected_graph_configuration_loco = (
            np.nan
            if graph_row is None
            else stable_fraction(
                loco_graph_configuration_counts[graph_row],
                loco_denominator,
            )
        )
        selected_graph_parent_bootstrap = [
            stable_fraction(
                graph_parent_counts[child, parent], bootstrap_denominator
            )
            for parent in graph_parents
        ]
        selected_graph_parent_loco = [
            stable_fraction(
                loco_graph_parent_counts[child, parent], loco_denominator
            )
            for parent in graph_parents
        ]
        unconstrained_best = float(np.max(
            full_selection.decision_scores[child_rows]
        ))
        selected_minus_best = (
            np.nan
            if graph_row is None
            else selected_utility - unconstrained_best
        )
        local_margin = (
            np.nan
            if local_state is None
            else float(full_selection.state_margins[child])
        )
        identity_margin = (
            np.nan
            if local_row is None
            else float(full_selection.identity_margins[child])
        )
        unresolved_reason = complete_status[child]
        support_text = _configuration_support_text(
            child_rows,
            local_configuration_counts,
            alternatives,
            states,
            samples,
            settings.support_set_coverage,
        )
        raw_junction_count = (
            np.nan
            if junction_matrix is None
            else float(np.sum(junction_matrix[:, child]))
        )
        callable_haplotype_bin_count = (
            np.nan
            if callable_matrix is None
            else float(np.sum(callable_matrix[:, child]))
        )
        if full_depth_model is None:
            adjusted_junction_burden = np.nan
            ancestry_callability = np.nan
            depth_posterior_text = ""
            depth_map = np.nan
            depth_component_count = 0
            depth_component_means = ""
            depth_component_standard_deviations = ""
            depth_component_weights = ""
            depth_selected_bic = np.nan
            depth_tested_bics = ""
        else:
            adjusted_junction_burden = float(
                full_depth_model.adjusted_junction_burden[child]
            )
            ancestry_callability = float(
                full_depth_model.callability_fraction[child]
            )
            child_depth_posterior = full_depth_model.posterior[child]
            depth_posterior_text = ";".join(
                f"{value:.8g}" for value in child_depth_posterior
            )
            depth_map = (
                np.nan
                if float(np.sum(child_depth_posterior)) <= 0.0
                else int(np.argmax(child_depth_posterior))
            )
            depth_component_count = len(
                full_depth_model.component_means
            )
            depth_component_means = ";".join(
                f"{value:.8g}"
                for value in full_depth_model.component_means
            )
            depth_component_standard_deviations = ";".join(
                f"{value:.8g}"
                for value in full_depth_model.component_standard_deviations
            )
            depth_component_weights = ";".join(
                f"{value:.8g}"
                for value in full_depth_model.component_weights
            )
            depth_selected_bic = full_depth_model.selected_bic
            depth_tested_bics = ";".join(
                f"{value:.8g}" for value in full_depth_model.tested_bics
            )

        diagnostics.append({
            "Sample": sample,
            "CompleteParent1": (
                None if len(graph_parents) < 1 else samples[graph_parents[0]]
            ),
            "CompleteParent2": (
                None if len(graph_parents) < 2 else samples[graph_parents[1]]
            ),
            "ParentOrderMeaning": "unordered_sample_array_index",
            "SelectedParentState": (
                None if graph_state is None else _PARENT_STATE_NAMES[graph_state]
            ),
            "LocalWinnerParentState": (
                None if local_state is None else _PARENT_STATE_NAMES[local_state]
            ),
            "ObservedParentCount": (
                np.nan if graph_state is None else graph_state
            ),
            "LocalObservedParentCount": (
                np.nan if local_state is None else local_state
            ),
            "Identifiable": bool(
                local_row is not None and graph_row == local_row
            ),
            "DAGDisplaced": graph_displaced,
            "GraphConflict": graph_conflict,
            "GraphTieConflict": graph_tie_conflict,
            "GraphDirectionResolvedAlternative": graph_direction_resolved,
            "GraphFallbackPolicy": (
                "local_if_feasible_else_same_state_depth_MAP_or_M0"
            ),
            "LocalParentRoleProbability": local_parent_role_probability,
            "SelectedParentRoleProbability": (
                selected_parent_role_probability
            ),
            "RawAncestryJunctionCount": raw_junction_count,
            "CallableAncestryHaplotypeBinCount": (
                callable_haplotype_bin_count
            ),
            "AdjustedAncestryJunctionBurden": adjusted_junction_burden,
            "AncestryPaintingCallabilityFraction": ancestry_callability,
            "LatentAncestryDepthMAP": depth_map,
            "LatentAncestryDepthPosterior": depth_posterior_text,
            "LatentAncestryDepthComponentCount": depth_component_count,
            "LatentAncestryDepthComponentMeans": depth_component_means,
            "LatentAncestryDepthComponentStandardDeviations": (
                depth_component_standard_deviations
            ),
            "LatentAncestryDepthComponentWeights": depth_component_weights,
            "LatentAncestryDepthSelectedBIC": depth_selected_bic,
            "LatentAncestryDepthTestedBICs": depth_tested_bics,
            "InformativeContigCount": informative_count,
            "LocalStateBootstrapFraction": state_bootstrap,
            "LocalConfigurationBootstrapFraction": (
                local_configuration_bootstrap
            ),
            "GraphConfigurationBootstrapFraction": (
                graph_configuration_bootstrap
            ),
            "SelectedGraphConfigurationBootstrapFraction": (
                selected_graph_configuration_bootstrap
            ),
            "SelectedGraphParent1BootstrapFraction": (
                selected_graph_parent_bootstrap[0]
                if len(selected_graph_parent_bootstrap) > 0 else np.nan
            ),
            "SelectedGraphParent2BootstrapFraction": (
                selected_graph_parent_bootstrap[1]
                if len(selected_graph_parent_bootstrap) > 1 else np.nan
            ),
            "PairBootstrapFraction": local_configuration_bootstrap,
            "Parent1BootstrapFraction": first_bootstrap,
            "Parent2BootstrapFraction": second_bootstrap,
            "LocalStateLOCOFraction": state_loco,
            "LocalConfigurationLOCOFraction": local_configuration_loco,
            "GraphConfigurationLOCOFraction": graph_configuration_loco,
            "SelectedGraphConfigurationLOCOFraction": (
                selected_graph_configuration_loco
            ),
            "SelectedGraphParent1LOCOFraction": (
                selected_graph_parent_loco[0]
                if len(selected_graph_parent_loco) > 0 else np.nan
            ),
            "SelectedGraphParent2LOCOFraction": (
                selected_graph_parent_loco[1]
                if len(selected_graph_parent_loco) > 1 else np.nan
            ),
            "PairLOCOFraction": local_configuration_loco,
            "Parent1LOCOFraction": first_loco,
            "Parent2LOCOFraction": second_loco,
            "StateWinnerMargin": local_margin,
            "ConditionalIdentityMargin": identity_margin,
            "UnconstrainedWinnerMargin": (
                min(local_margin, identity_margin)
                if np.isfinite(identity_margin)
                else local_margin
            ),
            "SelectedAggregateUtility": selected_utility,
            "SelectedMinusUnconstrainedBest": selected_minus_best,
            "StateLogEvidence0": full_selection.state_log_evidence[child, 0],
            "StateLogEvidence1": full_selection.state_log_evidence[child, 1],
            "StateLogEvidence2": full_selection.state_log_evidence[child, 2],
            "StateSupport0": full_selection.state_support[child, 0],
            "StateSupport1": full_selection.state_support[child, 1],
            "StateSupport2": full_selection.state_support[child, 2],
            "LOOStatePrior0": full_selection.loo_state_priors[child, 0],
            "LOOStatePrior1": full_selection.loo_state_priors[child, 1],
            "LOOStatePrior2": full_selection.loo_state_priors[child, 2],
            "ScoredCandidateCount0": scored_counts[child, 0],
            "ScoredCandidateCount1": scored_counts[child, 1],
            "ScoredCandidateCount2": scored_counts[child, 2],
            "FullCandidateCount0": full_counts[child, 0],
            "FullCandidateCount1": full_counts[child, 1],
            "FullCandidateCount2": full_counts[child, 2],
            "CandidatePairCount": scored_counts[child, 2],
            "FullCandidatePairCount": full_counts[child, 2],
            "M2StateEvidenceIsLowerBound": bool(
                scored_counts[child, 2] < full_counts[child, 2]
            ),
            "PriorSensitivityLocalStateAgreementFraction": float(np.mean(
                sensitivity_state_agreement
            )),
            "PriorSensitivityLocalIdentityAgreementFraction": float(np.mean(
                sensitivity_identity_agreement
            )),
            "PriorSensitivityGraphAgreementFraction": float(np.mean(
                sensitivity_graph_agreement
            )),
            "PriorSensitivitySelectedStateMinimumSupport": (
                np.nan
                if not sensitivity_selected_support
                else float(np.min(sensitivity_selected_support))
            ),
            "UnresolvedReason": unresolved_reason,
            "InferenceStatus": complete_status[child],
            "TierAStateCall": tier_a_state_pass,
            "TierBStateCall": tier_b_state_pass,
            "TierAExactConfiguration": tier_a_identity_pass,
            "TierBExactConfiguration": tier_b_identity_pass,
            "TierAExactPair": bool(
                tier_a_identity_pass and local_state == _TWO_OBSERVED
            ),
            "TierBExactPair": bool(
                tier_b_identity_pass and local_state == _TWO_OBSERVED
            ),
            "TierAParent1": tier_a_flags[0],
            "TierAParent2": tier_a_flags[1],
            "TierBParent1": tier_b_flags[0],
            "TierBParent2": tier_b_flags[1],
            "PairSupportSet": support_text,
            "ConfigurationSupportSet": support_text,
            "Interpretation": (
                "composite forward-likelihood model support and internal "
                "chromosome-resampling stability; not calibrated biological "
                "posterior probability"
            ),
        })
        state_call_rows.append({
            "Sample": sample,
            "SelectedParentState": (
                None if graph_state is None else _PARENT_STATE_NAMES[graph_state]
            ),
            "LocalWinnerParentState": (
                None if local_state is None else _PARENT_STATE_NAMES[local_state]
            ),
            "ObservedParentCount": (
                np.nan if graph_state is None else graph_state
            ),
            "LocalObservedParentCount": (
                np.nan if local_state is None else local_state
            ),
            "InferenceStatus": complete_status[child],
            "DAGDisplaced": graph_displaced,
            "GraphConflict": graph_conflict,
            "GraphTieConflict": graph_tie_conflict,
            "GraphDirectionResolvedAlternative": graph_direction_resolved,
            "SelectedParentRoleProbability": (
                selected_parent_role_probability
            ),
            "TierAStateCall": tier_a_state_pass,
            "TierBStateCall": tier_b_state_pass,
        })

        two_rows = child_rows[states[child_rows] == _TWO_OBSERVED]
        ordered_two = two_rows[np.argsort(
            -full_aggregate[two_rows], kind="stable"
        )]
        trio_candidates[sample] = [
            (
                samples[int(alternatives[row, 1])],
                samples[int(alternatives[row, 2])],
                float(full_aggregate[row]),
            )
            for row in ordered_two
        ]
        parent_score = {}
        for row in child_rows[states[child_rows] != _ZERO_OBSERVED]:
            for parent in alternatives[row, 1:]:
                parent = int(parent)
                if parent >= 0:
                    parent_score[parent] = max(
                        parent_score.get(parent, -np.inf),
                        float(full_aggregate[row]),
                    )
        parent_candidates[sample] = [
            (samples[parent], score)
            for parent, score in sorted(
                parent_score.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        trio_scores[sample] = selected_utility

    complete_frame = _parent_state_frame(
        samples,
        alternatives,
        states,
        complete_rows,
        complete_states,
        complete_status,
    )
    tier_a_frame = _parent_state_frame(
        samples,
        alternatives,
        states,
        tier_a_rows,
        tier_a_states,
        tier_a_status,
    )
    tier_b_frame = _parent_state_frame(
        samples,
        alternatives,
        states,
        tier_b_rows,
        tier_b_states,
        tier_b_status,
    )

    def partial_frame(
        exact: pd.DataFrame,
        tier_states: Mapping[int, int],
        tier_rows: Mapping[int, Optional[int]],
        parent_flags: Mapping[int, tuple[bool, bool]],
        label: str,
    ) -> pd.DataFrame:
        partial = exact.copy(deep=True)
        for child, state in tier_states.items():
            if tier_rows.get(child) is not None:
                continue
            local_row = full_selection.local_rows.get(child)
            if local_row is None or full_selection.graph_rows.get(child) != local_row:
                continue
            flags = parent_flags[child]
            parent_indices = [
                int(parent)
                for parent in alternatives[local_row, 1:]
                if int(parent) >= 0
            ]
            retained = 0
            for index, parent in enumerate(parent_indices):
                if index < len(flags) and flags[index]:
                    partial.at[child, f"Parent{index + 1}"] = samples[parent]
                    retained += 1
            if retained:
                partial.at[child, "InferenceStatus"] = (
                    f"{label}_partial_parent_support"
                )
        return partial

    tier_a_partial_frame = partial_frame(
        tier_a_frame,
        tier_a_states,
        tier_a_rows,
        tier_a_parent_flags,
        "tier_a",
    )
    tier_b_partial_frame = partial_frame(
        tier_b_frame,
        tier_b_states,
        tier_b_rows,
        tier_b_parent_flags,
        "tier_b",
    )
    primary = {
        "tier_a": tier_a_frame,
        "tier_b": tier_b_frame,
        "complete": complete_frame,
    }[settings.primary_view].copy(deep=True)
    effective_blocks = int(np.sum(np.ceil(
        markers / settings.markers_per_information_block
    )))
    result = PedigreeResult(
        samples,
        primary,
        parent_candidates,
        None,
        [],
        None,
        None,
        trio_scores=trio_scores,
        total_bins=effective_blocks,
    )
    result.trio_candidate_scores = trio_candidates
    result.smart_mode = True
    result.smart_parent_state_model = True
    result.smart_pair_only_compatibility_mode = False
    result.smart_bootstrap_worker_count = bootstrap_worker_count
    result.smart_bootstrap_depth_refit_count = bootstrap_depth_refits
    result.smart_config = settings
    result.tier_a_partial_relationships = tier_a_partial_frame
    result.tier_b_partial_relationships = tier_b_partial_frame
    result.tier_a_relationships = tier_a_frame
    result.tier_b_relationships = tier_b_frame
    result.complete_relationships = complete_frame
    result.smart_parent_state_calls = pd.DataFrame(state_call_rows)
    result.smart_diagnostics = pd.DataFrame(diagnostics)
    result.smart_prior_sensitivity_summary = pd.DataFrame(
        sensitivity_summary_rows
    )
    result.smart_fitted_parent_state_prior_parameters = (
        full_selection.fitted_prior_parameters.copy()
    )
    evidence_summary = {
        "Contig": contig_names,
        "InformativeMarkers": markers.astype(np.int64),
        "AggregationWeight": np.ones(len(contig_names), dtype=np.float64),
    }
    if junction_matrix is not None:
        evidence_summary["MeanMinimumAncestryJunctions"] = np.mean(
            junction_matrix, axis=1
        )
        evidence_summary["MeanCallableHaplotypeBins"] = np.mean(
            callable_matrix, axis=1
        )
    result.smart_evidence_summary = pd.DataFrame(evidence_summary)
    result.smart_ancestry_depth_model = full_depth_model
    result.smart_ancestry_depth_model_available = bool(
        full_depth_model is not None
        and full_depth_model.posterior.shape[1] >= 2
    )
    result.smart_ancestry_depth_model_parameters = {
        "maximum_components": _ANCESTRY_DEPTH_MAX_COMPONENTS,
        "gmm_initializations": _ANCESTRY_DEPTH_GMM_N_INIT,
        "gmm_max_iterations": _ANCESTRY_DEPTH_GMM_MAX_ITERATIONS,
        "standardized_covariance_regularization": (
            _ANCESTRY_DEPTH_GMM_REGULARIZATION
        ),
        "component_selection": "minimum_BIC",
        "direction_decision": "strict_MAP_probability_above_0.5",
    }
    result.smart_ancestry_depth_model_specification = (
        "BIC-selected one-dimensional Gaussian mixture over callability-"
        "adjusted, chromosome-wide phase-invariant minimum founder-trajectory "
        "switch burden; components are ordered relative ancestry layers, not "
        "generation labels. Parent-role probability orders mutually "
        "graph-conflicting local rows and, when a local row becomes "
        "infeasible, gates same-state alternatives. It never changes local "
        "likelihood winners or Tier A/B support."
    )
    result.smart_selection_method = (
        "marginal parent-state selection followed by conditional identity; "
        "deterministic ancestry-direction-then-confidence-ordered variable-edge "
        "DAG with coordinate local search. A graph-conflicted local row may "
        "use a unique same-state alternative only when all proposed parents "
        "are more likely earlier than the child; otherwise it falls to M0. "
        "Local support is measured before the DAG."
    )
    result.smart_candidate_screening_scope = (
        "M0 and every M1 identity are scored; M2 uses a fixed candidate panel. "
        "The M2 identity prior denominator is the full unrestricted pair "
        "space, so integrated M2 evidence is an explicit lower bound when the "
        "screen is incomplete."
    )
    result.smart_missing_parent_model = (
        "Normalized forward HMMs compare zero, one, and two observed parents. "
        "An external parent is a linked child-left-out mixture of locally "
        "IBS-pooled reconstructed founder haplotypes; zero observed parents "
        "does not assert biological founder status."
    )
    result.smart_limitations = (
        "No cohort, sex, breeding record, or sample-name eligibility was used; "
        "parent order is arbitrary. State support is based on a tempered "
        "composite likelihood and a weak leave-one-child-out hierarchical "
        "prior, not a calibrated posterior probability. M2 state evidence is "
        "a conservative lower bound wherever the candidate-pair screen is "
        "incomplete. Relative ancestry depth is unsupervised and painting-"
        "dependent: it is inferred from a conservative minimum-switch burden, "
        "not from known generation, age, or breeding metadata. Callability "
        "tempers each sample's component posterior, but callability-adjusted "
        "burdens still enter mixture fitting and BIC equally; highly incomplete "
        "paintings therefore remain a validation risk. Depth is used only to "
        "order graph conflicts and orient same-state complete-view hypotheses; "
        "those "
        "alternatives remain ineligible for Tier A/B and for the pipeline "
        "control adapter. Reconstructed paintings and hard founder alleles "
        "inherit upstream errors; raw genotype likelihoods are not double-"
        "counted as an independent source. Zero observed parents may mean a "
        "top-level individual or unsequenced biological parents and is not by "
        "itself safe founder-recolour eligibility. For API compatibility, trio_scores "
        "contains hierarchical decision utilities and total_bins an effective "
        "information-block count; neither is comparable to legacy Viterbi "
        "scores or raw bin counts."
    )
    return result


def _standard_input_schema(
    contig_data_list: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether every item has the historical numerical input schema."""
    if not contig_data_list:
        return False
    status = []
    for item in contig_data_list:
        if not isinstance(item, Mapping):
            return False
        has_painting = "tolerance_painting" in item
        has_founder = "founder_block" in item
        if has_painting != has_founder:
            raise SmartEvidenceError(
                "each standard contig needs both tolerance_painting and "
                "founder_block"
            )
        status.append(has_painting)
    if any(status) and not all(status):
        raise SmartEvidenceError(
            "standard tolerance-painting evidence must be supplied for every "
            "contig or for none"
        )
    return all(status)


def _hard_founder_haplotypes(founder_block: Any) -> tuple[list[int], np.ndarray]:
    positions = np.asarray(getattr(founder_block, "positions", None))
    haplotypes = getattr(founder_block, "haplotypes", None)
    if positions.ndim != 1 or len(positions) == 0:
        raise SmartEvidenceError("founder_block.positions must be non-empty")
    if len(positions) > 1 and np.any(np.diff(positions) <= 0):
        raise SmartEvidenceError("founder positions must be strictly increasing")
    if not isinstance(haplotypes, Mapping) or not haplotypes:
        raise SmartEvidenceError("founder_block.haplotypes must be non-empty")
    keys = sorted(int(value) for value in haplotypes)
    if any(value < 0 for value in keys) or len(set(keys)) != len(keys):
        raise SmartEvidenceError("founder labels must be unique non-negative integers")
    hard = np.empty((len(keys), len(positions)), dtype=np.int8)
    for local, key in enumerate(keys):
        values = np.asarray(haplotypes[key])
        if values.ndim == 2:
            if values.shape != (len(positions), 2):
                raise SmartEvidenceError(
                    "probabilistic founder haplotypes need shape (markers, 2)"
                )
            if np.any(~np.isfinite(values)):
                raise SmartEvidenceError("founder haplotype probabilities must be finite")
            values = np.argmax(values, axis=1)
        elif values.ndim != 1 or len(values) != len(positions):
            raise SmartEvidenceError(
                "hard founder haplotypes need one allele per founder position"
            )
        if np.any(~np.isin(values, (-1, 0, 1))):
            raise SmartEvidenceError("hard founder alleles must be -1, 0, or 1")
        hard[local] = values.astype(np.int8)
    return keys, hard


def _build_standard_contig_cache(
    item: Mapping[str, Any],
    contig_index: int,
    n_samples: int,
    snps_per_bin: int,
    recombination_rate: float,
    max_snps_per_bin: int,
) -> _StandardContigCache:
    """Convert historical painting inputs without rebuilding founders per sample."""
    if int(snps_per_bin) != snps_per_bin or snps_per_bin < 1:
        raise SmartEvidenceError("snps_per_bin must be a positive integer")
    if int(max_snps_per_bin) != max_snps_per_bin or max_snps_per_bin < 1:
        raise SmartEvidenceError("max_snps_per_bin must be a positive integer")
    if not np.isfinite(recombination_rate) or recombination_rate <= 0.0:
        raise SmartEvidenceError("recomb_rate must be finite and positive")
    painting = item["tolerance_painting"]
    founder_block = item["founder_block"]
    try:
        painting_samples = len(painting)
        start_position = float(painting.start_pos)
        end_position = float(painting.end_pos)
    except (AttributeError, TypeError) as error:
        raise SmartEvidenceError("invalid tolerance_painting container") from error
    if painting_samples != n_samples:
        raise SmartEvidenceError(
            "tolerance painting sample count does not match sample_ids"
        )
    if not (np.isfinite(start_position) and np.isfinite(end_position)):
        raise SmartEvidenceError("painting coordinates must be finite")
    if end_position <= start_position:
        raise SmartEvidenceError("painting end must be greater than start")

    positions = np.asarray(founder_block.positions, dtype=np.int64)
    founder_keys, hard_founders = _hard_founder_haplotypes(founder_block)
    approximate_bp_per_bin = int(snps_per_bin) * 100
    n_bins = max(
        100,
        int((end_position - start_position) / approximate_bp_per_bin),
    )
    bin_edges = np.linspace(start_position, end_position, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = (
        float(bin_centers[1] - bin_centers[0]) if n_bins > 1 else 10000.0
    )

    id_grids = np.full((n_samples, n_bins, 2), -1, dtype=np.int32)
    for sample_index in range(n_samples):
        sample = painting[sample_index]
        chunks = list(getattr(sample, "chunks", ()) or ())
        if not chunks:
            continue
        starts = np.asarray([chunk.start for chunk in chunks], dtype=np.float64)
        ends = np.asarray([chunk.end for chunk in chunks], dtype=np.float64)
        first_ids = np.asarray([chunk.hap1 for chunk in chunks], dtype=np.int64)
        second_ids = np.asarray([chunk.hap2 for chunk in chunks], dtype=np.int64)
        if (
            np.any(~np.isfinite(starts))
            or np.any(~np.isfinite(ends))
            or np.any(ends <= starts)
            or np.any(starts[1:] < starts[:-1])
            or np.any(ends[1:] < ends[:-1])
        ):
            raise SmartEvidenceError("painting chunks must be sorted valid intervals")
        chunk_indices = np.searchsorted(ends, bin_centers)
        chunk_indices = np.clip(chunk_indices, 0, len(chunks) - 1)
        valid = bin_centers >= starts[chunk_indices]
        id_grids[sample_index, :, 0] = np.where(
            valid, first_ids[chunk_indices], -1
        )
        id_grids[sample_index, :, 1] = np.where(
            valid, second_ids[chunk_indices], -1
        )

    local_ids = np.full_like(id_grids, -1)
    for local, founder_id in enumerate(founder_keys):
        local_ids[id_grids == founder_id] = local

    if max_snps_per_bin > 1:
        start_indices = np.searchsorted(
            positions, bin_centers - bin_width / 2.0, side="left"
        )
        end_indices = np.searchsorted(
            positions, bin_centers + bin_width / 2.0, side="right"
        )
        selected = np.full((n_bins, max_snps_per_bin), -1, dtype=np.int64)
        for bin_index in range(n_bins):
            first = int(start_indices[bin_index])
            last = int(end_indices[bin_index])
            count = last - first
            if count <= 0:
                continue
            if count <= max_snps_per_bin:
                chosen = np.arange(first, last, dtype=np.int64)
            else:
                step = count / float(max_snps_per_bin)
                chosen = np.asarray([
                    first + int(slot * step)
                    for slot in range(max_snps_per_bin)
                ], dtype=np.int64)
            selected[bin_index, :len(chosen)] = chosen
        informative_markers = int(np.sum(selected >= 0))
        selected_markers_per_bin = np.sum(selected >= 0, axis=1).astype(
            np.int64
        )
        lookup = np.full(
            (len(founder_keys), n_bins, max_snps_per_bin),
            -1,
            dtype=np.int8,
        )
        for bin_index in range(n_bins):
            chosen = selected[bin_index]
            valid_slots = chosen >= 0
            if np.any(valid_slots):
                lookup[:, bin_index, valid_slots] = hard_founders[
                    :, chosen[valid_slots]
                ]
        safe_ids = np.maximum(local_ids, 0)
        bin_axis = np.arange(n_bins)[None, :]
        first_track = lookup[safe_ids[:, :, 0], bin_axis].copy()
        second_track = lookup[safe_ids[:, :, 1], bin_axis].copy()
        first_track[local_ids[:, :, 0] < 0] = -1
        second_track[local_ids[:, :, 1] < 0] = -1
        stacked = np.stack((first_track, second_track), axis=2)
        jointly_valid = (first_track != -1) & (second_track != -1)
        hom_mask = (
            ~np.any(jointly_valid, axis=2)
            | np.all(~jointly_valid | (first_track == second_track), axis=2)
        )
    else:
        selected = np.searchsorted(positions, bin_centers)
        selected = np.clip(selected, 0, len(positions) - 1)
        valid_marker = (
            np.abs(positions[selected] - bin_centers) <= bin_width / 2.0
        )
        informative_markers = int(np.sum(valid_marker))
        lookup = hard_founders[:, selected].copy()
        lookup[:, ~valid_marker] = -1
        selected_markers_per_bin = valid_marker.astype(np.int64)
        safe_ids = np.maximum(local_ids, 0)
        bin_axis = np.arange(n_bins)[None, :]
        first_track = lookup[safe_ids[:, :, 0], bin_axis].copy()
        second_track = lookup[safe_ids[:, :, 1], bin_axis].copy()
        first_track[local_ids[:, :, 0] < 0] = -1
        second_track[local_ids[:, :, 1] < 0] = -1
        stacked = np.stack((first_track, second_track), axis=2)
        hom_mask = (
            (first_track == second_track)
            | (first_track == -1)
            | (second_track == -1)
        )
    if informative_markers < 1:
        raise SmartEvidenceError("a standard contig contains no sampled markers")

    distances = np.zeros(n_bins, dtype=np.float64)
    distances[1:] = np.diff(bin_centers)
    theta = np.clip(
        1.0 - np.exp(-distances * recombination_rate), 1e-15, 0.5
    )
    name = str(item.get("contig", f"contig_{contig_index + 1}"))
    return _StandardContigCache(
        contig=name,
        stacked_alleles=np.ascontiguousarray(stacked, dtype=np.int8),
        stacked_hom_mask=np.ascontiguousarray(hom_mask, dtype=np.bool_),
        switch_costs=np.log(theta),
        stay_costs=np.log(1.0 - theta),
        informative_markers=informative_markers,
        stacked_labels=np.ascontiguousarray(local_ids, dtype=np.int16),
        founder_alleles=np.ascontiguousarray(
            lookup if lookup.ndim == 3 else lookup[:, :, None],
            dtype=np.int8,
        ),
        selected_markers_per_bin=np.ascontiguousarray(
            selected_markers_per_bin, dtype=np.int64
        ),
        switch_probabilities=np.ascontiguousarray(theta, dtype=np.float64),
    )


_RUN_PAIR_HMM = _legacy.run_phase_agnostic_hmm
_RUN_PAIR_HMM_MULTISNP = _legacy.run_phase_agnostic_hmm_multisnp


@njit(fastmath=True, cache=True, parallel=True)
def _score_all_pairs_kernel_multisnp(
    stacked_alleles,
    stacked_hom_mask,
    switch_costs,
    stay_costs,
    error_penalty,
    phase_penalty,
    mismatch_penalty,
):
    """Score the full child-parent matrix in one parallel launch."""
    n_samples = stacked_alleles.shape[0]
    output = np.empty((n_samples, n_samples), dtype=np.float64)
    for flat_index in prange(n_samples * n_samples):
        child = flat_index // n_samples
        parent = flat_index - child * n_samples
        if child == parent:
            output[child, parent] = -math.inf
        else:
            output[child, parent] = _RUN_PAIR_HMM_MULTISNP(
                stacked_alleles[child],
                stacked_hom_mask[child],
                stacked_alleles[parent],
                switch_costs,
                stay_costs,
                error_penalty,
                phase_penalty,
                mismatch_penalty,
            )
    return output


@njit(fastmath=True, cache=True, parallel=True)
def _score_all_pairs_kernel(
    stacked_alleles,
    stacked_hom_mask,
    switch_costs,
    stay_costs,
    error_penalty,
    phase_penalty,
    mismatch_penalty,
):
    """Non-multisnp counterpart of the flattened pair-screen kernel."""
    n_samples = stacked_alleles.shape[0]
    output = np.empty((n_samples, n_samples), dtype=np.float64)
    for flat_index in prange(n_samples * n_samples):
        child = flat_index // n_samples
        parent = flat_index - child * n_samples
        if child == parent:
            output[child, parent] = -math.inf
        else:
            output[child, parent] = _RUN_PAIR_HMM(
                stacked_alleles[child],
                stacked_hom_mask[child],
                stacked_alleles[parent],
                switch_costs,
                stay_costs,
                error_penalty,
                phase_penalty,
                mismatch_penalty,
            )
    return output


def _score_pair_hmm_contig(
    cache: _StandardContigCache,
    mismatch_penalty: float,
) -> np.ndarray:
    error_penalty = -math.log(1e-2)
    phase_penalty = 50.0
    if cache.stacked_alleles.ndim == 4:
        return _score_all_pairs_kernel_multisnp(
            cache.stacked_alleles,
            cache.stacked_hom_mask,
            cache.switch_costs,
            cache.stay_costs,
            error_penalty,
            phase_penalty,
            mismatch_penalty,
        )
    return _score_all_pairs_kernel(
        cache.stacked_alleles,
        cache.stacked_hom_mask,
        cache.switch_costs,
        cache.stay_costs,
        error_penalty,
        phase_penalty,
        mismatch_penalty,
    )


def _robust_parent_screen(
    pair_scores: np.ndarray,
    marker_counts: np.ndarray,
    config: SmartPedigreeConfig,
) -> np.ndarray:
    """Information-weighted parent utilities used only to fix the trio panel."""
    n_contigs, n_samples, _ = pair_scores.shape
    output = np.full((n_samples, n_samples), -np.inf, dtype=np.float64)
    contig_weights = _information_weights(marker_counts, config)
    blocks = np.maximum(
        np.ceil(marker_counts / config.markers_per_information_block), 1.0
    )
    tempering = blocks ** config.information_tempering_power
    for child in range(n_samples):
        parents = np.asarray(
            [index for index in range(n_samples) if index != child],
            dtype=np.int64,
        )
        utility = np.empty((n_contigs, len(parents)), dtype=np.float64)
        for contig_index in range(n_contigs):
            values = pair_scores[contig_index, child, parents]
            if np.any(~np.isfinite(values)):
                raise SmartEvidenceError("standard pair-HMM scores must be finite")
            centered = (values - np.max(values)) / tempering[contig_index]
            soft = np.exp(np.clip(centered, -60.0, 0.0))
            soft /= np.sum(soft)
            soft = (
                (1.0 - config.chromosome_contamination) * soft
                + config.chromosome_contamination / len(parents)
            )
            ranks = _tied_rank_probabilities(values)
            utility[contig_index] = (
                config.rank_weight * ranks
                + (1.0 - config.rank_weight) * soft
            )
        output[child, parents] = np.sum(
            utility * contig_weights[:, None], axis=0
        )
    return output


def _fixed_trio_panel(
    parent_scores: np.ndarray,
    top_k: int,
    anchor_k: int,
    use_anchor_union: bool,
) -> np.ndarray:
    n_samples = parent_scores.shape[0]
    if n_samples < 3:
        raise SmartEvidenceError("at least three samples are required")
    if int(top_k) != top_k or top_k < 1:
        raise SmartEvidenceError("top_k must be a positive integer")
    if int(anchor_k) != anchor_k or anchor_k < 0:
        raise SmartEvidenceError("anchor_k must be a non-negative integer")
    rows = []
    for child in range(n_samples):
        parents = np.asarray(
            [index for index in range(n_samples) if index != child],
            dtype=np.int64,
        )
        order = np.lexsort((parents, -parent_scores[child, parents]))
        leading = parents[order[:min(int(top_k), len(parents))]].tolist()
        pairs = {
            tuple(sorted((int(leading[first]), int(leading[second]))))
            for first in range(len(leading))
            for second in range(first + 1, len(leading))
        }
        if use_anchor_union:
            for anchor in leading[:min(int(anchor_k), len(leading))]:
                for other in parents:
                    if int(other) != int(anchor):
                        pairs.add(tuple(sorted((int(anchor), int(other)))))
        if not pairs:
            raise SmartEvidenceError(
                "candidate policy produced no two-parent pair for a child"
            )
        rows.extend((child, first, second) for first, second in sorted(pairs))
    return np.asarray(rows, dtype=np.int64)


def _score_trio_hmm_contig(
    cache: _StandardContigCache,
    trios: np.ndarray,
    mismatch_penalty: float,
) -> np.ndarray:
    output = np.empty(len(trios), dtype=np.float64)
    error_penalty = -math.log(1e-2)
    phase_penalty = 50.0
    children, starts = np.unique(trios[:, 0], return_index=True)
    ends = np.r_[starts[1:], len(trios)]
    for child, start, end in zip(children, starts, ends):
        first = np.ascontiguousarray(trios[start:end, 1], dtype=np.int64)
        second = np.ascontiguousarray(trios[start:end, 2], dtype=np.int64)
        if cache.stacked_alleles.ndim == 4:
            values = _legacy.score_trio_batch_kernel_multisnp(
                cache.stacked_alleles[int(child)],
                cache.stacked_hom_mask[int(child)],
                cache.stacked_alleles,
                first,
                second,
                cache.switch_costs,
                cache.stay_costs,
                error_penalty,
                phase_penalty,
                mismatch_penalty,
            )
        else:
            values = _legacy.score_trio_batch_kernel(
                cache.stacked_alleles[int(child)],
                cache.stacked_hom_mask[int(child)],
                cache.stacked_alleles,
                first,
                second,
                cache.switch_costs,
                cache.stay_costs,
                error_penalty,
                phase_penalty,
                mismatch_penalty,
            )
        output[start:end] = values
    return output



_SMART_STATE_TRIOS = None
_SMART_STATE_TRIO_SHM = None
_SMART_STATE_CONFIG = None
_SMART_STATE_START_BARRIER = None
_SMART_STATE_TARGET_THREADS_PER_PROCESS = 8
_SMART_STATE_CHILD_CHUNK_FLOOR = 32
_SMART_STATE_CHILD_CHUNK_SCALE = 4


def _init_smart_state_worker(
    trio_meta,
    config,
    active_counter,
    total_cores,
    extra_counter,
    start_barrier,
    child_chunk_floor,
    child_chunk_scale,
):
    """Attach the fixed trio panel and initialize bounded dynamic threads."""
    global _SMART_STATE_TRIOS, _SMART_STATE_TRIO_SHM
    global _SMART_STATE_CONFIG, _SMART_STATE_START_BARRIER
    global _SMART_STATE_CHILD_CHUNK_FLOOR
    global _SMART_STATE_CHILD_CHUNK_SCALE
    _SMART_STATE_TRIO_SHM, _SMART_STATE_TRIOS = _legacy._attach_shm_view(
        trio_meta
    )
    _SMART_STATE_CONFIG = config
    _SMART_STATE_START_BARRIER = start_barrier
    _SMART_STATE_CHILD_CHUNK_FLOOR = child_chunk_floor
    _SMART_STATE_CHILD_CHUNK_SCALE = child_chunk_scale
    dynamic_threads.set_dynamic_thread_state(
        total_cores, active_counter, extra_counter
    )
    if SMART_HAS_NUMBA:
        numba.set_num_threads(1)


def _score_standard_contig_state_worker(bundle):
    """Score one static contig bundle with dynamic intra-process threads."""
    dynamic_threads.increment_active()
    try:
        _SMART_STATE_START_BARRIER.wait(timeout=180.0)
        settings = _SMART_STATE_CONFIG
        indexed = []
        for contig_index, cache in bundle:
            _apply_smart_dynamic_threads()
            scores = score_parent_state_hmms(
                cache.stacked_alleles,
                cache.stacked_labels,
                cache.stacked_hom_mask,
                cache.founder_alleles,
                cache.selected_markers_per_bin,
                cache.switch_probabilities,
                _SMART_STATE_TRIOS,
                mismatch_probability=(
                    settings.parent_state_mismatch_probability
                ),
                phase_switch_probability=(
                    settings.parent_state_phase_switch_probability
                ),
                markers_per_information_block=(
                    settings.markers_per_information_block
                ),
                effective_markers_per_information_block=(
                    settings.parent_state_effective_markers_per_information_block
                ),
                external_state_pseudocount=(
                    settings.parent_state_external_state_pseudocount
                ),
                external_transition_pseudocount=(
                    settings.parent_state_external_transition_pseudocount
                ),
                _dynamic_rebalance=True,
                _dynamic_child_chunk_scale=(
                    _SMART_STATE_CHILD_CHUNK_SCALE
                ),
                _dynamic_child_chunk_floor=(
                    _SMART_STATE_CHILD_CHUNK_FLOOR
                ),
            )
            indexed.append((contig_index, scores))
        return indexed
    finally:
        dynamic_threads.release_dynamic_extra()
        dynamic_threads.decrement_active()


def _warm_smart_parent_state_kernels():
    """Compile each standard-input smart signature once before forkserver."""
    if not SMART_HAS_NUMBA:
        return
    previous_threads = int(numba.get_num_threads())
    numba.set_num_threads(1)
    try:
        founders = np.asarray(
            (((0,), (1,)), ((1,), (0,))), dtype=np.int8
        )
        labels = np.asarray(
            (
                ((0, 1), (0, 1)),
                ((0, 0), (1, 1)),
                ((1, 0), (1, 0)),
                ((1, 1), (0, 0)),
            ),
            dtype=np.int16,
        )
        alleles = np.empty((4, 2, 2, 1), dtype=np.int8)
        for sample in range(4):
            for block in range(2):
                for track in range(2):
                    alleles[sample, block, track, 0] = founders[
                        labels[sample, block, track], block, 0
                    ]
        trios = []
        for child in range(4):
            parents = [value for value in range(4) if value != child]
            for first_index in range(len(parents)):
                for second_index in range(first_index + 1, len(parents)):
                    trios.append(
                        (child, parents[first_index], parents[second_index])
                    )
        score_parent_state_hmms(
            alleles,
            labels,
            np.zeros((4, 2), dtype=np.bool_),
            founders,
            np.ones(2, dtype=np.int64),
            np.asarray((0.0, 0.01), dtype=np.float64),
            np.asarray(trios, dtype=np.int64),
        )
    finally:
        numba.set_num_threads(previous_threads)


def _balanced_state_contig_bundles(
    caches,
    worker_count,
):
    """Assign whole contigs to deterministic least-loaded static bundles."""
    if int(worker_count) != worker_count or worker_count < 1:
        raise SmartEvidenceError("state_worker_count must be a positive integer")
    bundle_count = min(int(worker_count), len(caches))
    bundles = [[] for _ in range(bundle_count)]
    loads = [0 for _ in range(bundle_count)]
    weighted = sorted(
        (
            (
                int(cache.stacked_alleles.shape[1]),
                index,
                cache,
            )
            for index, cache in enumerate(caches)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for weight, index, cache in weighted:
        target = min(range(bundle_count), key=lambda item: (loads[item], item))
        bundles[target].append((index, cache))
        loads[target] += weight
    return [tuple(bundle) for bundle in bundles]


def _score_standard_state_contigs(
    caches,
    trios,
    config,
    requested_threads,
    state_worker_count=None,
    dynamic_child_chunk_floor=32,
    dynamic_child_chunk_scale=4,
):
    """Score contigs serially or with a fixed, dynamically threaded pool."""
    n_contigs = len(caches)
    try:
        child_chunk_floor = operator.index(dynamic_child_chunk_floor)
    except TypeError as exc:
        raise SmartEvidenceError(
            "dynamic_child_chunk_floor must be a positive integer"
        ) from exc
    if child_chunk_floor < 1:
        raise SmartEvidenceError(
            "dynamic_child_chunk_floor must be a positive integer"
        )
    try:
        child_chunk_scale = operator.index(dynamic_child_chunk_scale)
    except TypeError as exc:
        raise SmartEvidenceError(
            "dynamic_child_chunk_scale must be a non-negative integer"
        ) from exc
    if child_chunk_scale < 0:
        raise SmartEvidenceError(
            "dynamic_child_chunk_scale must be a non-negative integer"
        )
    if state_worker_count is None:
        target = _SMART_STATE_TARGET_THREADS_PER_PROCESS
        if n_contigs < target:
            worker_count = min(n_contigs, requested_threads)
        else:
            worker_count = min(
                n_contigs,
                max(1, (requested_threads + target - 1) // target),
            )
    else:
        try:
            worker_count = operator.index(state_worker_count)
        except TypeError as exc:
            raise SmartEvidenceError(
                "state_worker_count must be a positive integer"
            ) from exc
        if worker_count < 1:
            raise SmartEvidenceError("state_worker_count must be positive")
        worker_count = min(worker_count, n_contigs)
    use_pool = (
        SMART_HAS_NUMBA
        and worker_count > 1
        and requested_threads >= worker_count
    )
    if not use_pool:
        if SMART_HAS_NUMBA:
            numba.set_num_threads(requested_threads)
        return [
            score_parent_state_hmms(
                cache.stacked_alleles,
                cache.stacked_labels,
                cache.stacked_hom_mask,
                cache.founder_alleles,
                cache.selected_markers_per_bin,
                cache.switch_probabilities,
                trios,
                mismatch_probability=(
                    config.parent_state_mismatch_probability
                ),
                phase_switch_probability=(
                    config.parent_state_phase_switch_probability
                ),
                markers_per_information_block=(
                    config.markers_per_information_block
                ),
                effective_markers_per_information_block=(
                    config.parent_state_effective_markers_per_information_block
                ),
                external_state_pseudocount=(
                    config.parent_state_external_state_pseudocount
                ),
                external_transition_pseudocount=(
                    config.parent_state_external_transition_pseudocount
                ),
            )
            for cache in caches
        ], 1

    _warm_smart_parent_state_kernels()
    active_counter = _legacy._forkserver_ctx.Value("i", 0)
    extra_counter = _legacy._forkserver_ctx.Value("i", 0)
    start_barrier = _legacy._forkserver_ctx.Barrier(worker_count)
    trio_shm, trio_meta = _legacy._create_shm_array(trios)
    tasks = _balanced_state_contig_bundles(caches, worker_count)
    with _legacy._shm_cleanup([trio_shm]), _legacy._safe_forkserver_pool(
        worker_count,
        initializer=_init_smart_state_worker,
        initargs=(
            trio_meta,
            config,
            active_counter,
            requested_threads,
            extra_counter,
            start_barrier,
            child_chunk_floor,
            child_chunk_scale,
        ),
    ) as pool:
        nested = list(pool.imap_unordered(
            _score_standard_contig_state_worker, tasks, chunksize=1
        ))
    indexed = [
        item for bundle in nested for item in bundle
    ]
    indexed.sort(key=operator.itemgetter(0))
    return [scores for _, scores in indexed], worker_count


def _standard_contig_evidence(
    contig_data_list: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[Any],
    config: SmartPedigreeConfig,
    *,
    top_k: int,
    snps_per_bin: int,
    recomb_rate: float,
    mismatch_penalty: float,
    max_snps_per_bin: int,
    n_workers: Optional[int],
    anchor_k: int,
    use_anchor_union: bool,
) -> tuple[
    list[SmartParentStateEvidence],
    np.ndarray,
    int,
    int,
    np.ndarray,
    np.ndarray,
]:
    if not np.isfinite(mismatch_penalty) or mismatch_penalty >= 0.0:
        raise SmartEvidenceError("mismatch_penalty must be finite and negative")
    n_samples = len(sample_ids)
    if n_samples < 3 or len(set(sample_ids)) != n_samples:
        raise SmartEvidenceError(
            "sample_ids must contain at least three unique IDs"
        )
    caches = [
        _build_standard_contig_cache(
            item,
            contig_index,
            n_samples,
            snps_per_bin,
            recomb_rate,
            max_snps_per_bin,
        )
        for contig_index, item in enumerate(contig_data_list)
    ]
    names = [cache.contig for cache in caches]
    if len(set(names)) != len(names):
        raise SmartEvidenceError("standard contig identifiers must be unique")
    marker_counts = np.asarray(
        [cache.informative_markers for cache in caches], dtype=np.float64
    )

    previous_threads = None
    if SMART_HAS_NUMBA:
        capacity = int(numba.config.NUMBA_NUM_THREADS)
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available_cpus = os.cpu_count() or 1
        if n_workers is None:
            requested_threads = min(
                capacity, available_cpus
            )
        else:
            if int(n_workers) != n_workers or n_workers < 1:
                raise SmartEvidenceError("n_workers must be a positive integer")
            requested_threads = min(
                int(n_workers), available_cpus, capacity
            )
        previous_threads = int(numba.get_num_threads())
        numba.set_num_threads(requested_threads)
    else:
        requested_threads = 1
    try:
        pair_scores = np.asarray([
            _score_pair_hmm_contig(cache, mismatch_penalty)
            for cache in caches
        ])
        parent_scores = _robust_parent_screen(
            pair_scores, marker_counts, config
        )
        trios = _fixed_trio_panel(
            parent_scores, top_k, anchor_k, use_anchor_union
        )
        state_scores, state_processes = _score_standard_state_contigs(
            caches, trios, config, requested_threads
        )
    finally:
        if previous_threads is not None:
            numba.set_num_threads(previous_threads)

    evidence = [
        SmartParentStateEvidence(
            contig=cache.contig,
            trios=trios,
            zero_parent_log_likelihoods=state_scores[index].zero_observed,
            one_parent_log_likelihoods=state_scores[index].one_observed,
            two_parent_log_likelihoods=state_scores[index].two_observed,
            informative_markers=cache.informative_markers,
        )
        for index, cache in enumerate(caches)
    ]
    junction_counts = np.stack([
        scores.ancestry_junction_counts for scores in state_scores
    ])
    callable_haplotype_bins = np.stack([
        scores.ancestry_callable_haplotype_bins for scores in state_scores
    ])
    return (
        evidence,
        trios,
        requested_threads,
        state_processes,
        junction_counts,
        callable_haplotype_bins,
    )


def _config_from_contig_inputs(
    contig_data_list: Sequence[Mapping[str, Any]],
    explicit_config: Optional[SmartPedigreeConfig] = None,
) -> SmartPedigreeConfig:
    if explicit_config is not None:
        if not isinstance(explicit_config, SmartPedigreeConfig):
            raise SmartEvidenceError("config must be a SmartPedigreeConfig")
        return explicit_config.validated()
    config: Any = None
    if contig_data_list and "smart_config" in contig_data_list[0]:
        config = contig_data_list[0]["smart_config"]
        if isinstance(config, Mapping):
            config = SmartPedigreeConfig(**dict(config))
        if not isinstance(config, SmartPedigreeConfig):
            raise SmartEvidenceError(
                "smart_config must be SmartPedigreeConfig or a field mapping"
            )
    return (config or SmartPedigreeConfig()).validated()


def _infer_standard_inputs(
    contig_data_list: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[Any],
    config: SmartPedigreeConfig,
    *,
    top_k: int,
    snps_per_bin: int,
    recomb_rate: float,
    mismatch_penalty: float,
    max_snps_per_bin: int,
    n_workers: Optional[int],
    anchor_k: int,
    use_anchor_union: bool,
    apply_consistency_cutoff: bool,
) -> PedigreeResult:
    (
        evidence,
        trios,
        used_threads,
        state_processes,
        junction_counts,
        callable_haplotype_bins,
    ) = _standard_contig_evidence(
        contig_data_list,
        sample_ids,
        config,
        top_k=top_k,
        snps_per_bin=snps_per_bin,
        recomb_rate=recomb_rate,
        mismatch_penalty=mismatch_penalty,
        max_snps_per_bin=max_snps_per_bin,
        n_workers=n_workers,
        anchor_k=anchor_k,
        use_anchor_union=use_anchor_union,
    )
    result = infer_from_parent_state_evidence(
        evidence,
        sample_ids,
        config=config,
        ancestry_junction_counts=junction_counts,
        ancestry_callable_haplotype_bins=callable_haplotype_bins,
        n_workers=used_threads,
    )
    result.smart_evidence_source = (
        "normalized per-contig zero/one/two-observed-parent forward HMMs "
        "from tolerance_painting + reconstructed founder_block; locally "
        "IBS-equivalent founder labels are pooled; no metadata and no raw "
        "genotype likelihoods; parent direction is resolved only for "
        "graph-conflicted rows using a phase-invariant chromosome-wide "
        "minimum-switch model over unique founder trajectories"
    )
    result.smart_candidate_screening_scope = (
        f"M0 plus every M1 identity and a fixed {len(trios):,}-trio M2 panel; "
        "the M2 panel was selected once from robust, information-weighted "
        "full-data pair-HMM scores. Bootstrap and LOCO are conditional on the "
        "screen, while the M2 identity prior still uses the full unrestricted "
        "pair count, making incomplete-screen M2 state evidence a labelled "
        "lower bound."
    )
    result.smart_standard_input_threads = used_threads
    result.smart_standard_input_processes = state_processes
    result.smart_legacy_consistency_cutoff_requested = bool(
        apply_consistency_cutoff
    )
    result.smart_legacy_consistency_cutoff_applied = False
    result.smart_missing_parent_model = (
        "Comparable normalized forward likelihoods explicitly model zero, one, "
        "and two observed parents. Missing biological parents are linked "
        "external ancestry paths over locally IBS-pooled reconstructed "
        "haplotypes; zero observed parents does not assert founder status."
    )
    result.smart_limitations += (
        " Standard-input evidence comes from reconstructed paintings rather "
        "than raw VCF genotype likelihoods. The historical 0.90 label "
        "consistency cutoff is recorded but is neither used nor tuned by the "
        "new model."
    )
    return result


def _explicit_evidence_from_inputs(
    contig_data_list: Sequence[Mapping[str, Any]],
    sample_ids: Optional[Sequence[Any]] = None,
) -> Optional[list[SmartContigEvidence | SmartParentStateEvidence]]:
    if not contig_data_list:
        return None
    state_present = [
        "smart_parent_state_evidence" in item for item in contig_data_list
    ]
    if any(state_present) and not all(state_present):
        raise SmartEvidenceError(
            "smart_parent_state_evidence must be supplied for every contig "
            "or for none"
        )
    if all(state_present):
        if sample_ids is None:
            raise SmartEvidenceError(
                "sample_ids are required for parent-state evidence"
            )
        return [
            _as_parent_state_evidence(
                item["smart_parent_state_evidence"], len(sample_ids)
            )
            for item in contig_data_list
        ]

    present = ["smart_evidence" in item for item in contig_data_list]
    if any(present) and not all(present):
        raise SmartEvidenceError(
            "smart_evidence must be supplied for every contig or for none"
        )
    if all(present):
        return [
            _as_contig_evidence(item["smart_evidence"])
            for item in contig_data_list
        ]

    direct = [
        all(key in item for key in ("genotype_likelihoods", "candidate_trios"))
        for item in contig_data_list
    ]
    if any(direct) and not all(direct):
        raise SmartEvidenceError(
            "direct genotype evidence must be supplied for every contig or none"
        )
    if all(direct):
        evidence = []
        for index, item in enumerate(contig_data_list):
            evidence.append(score_genotype_likelihood_evidence(
                str(item.get("contig", index)),
                item["genotype_likelihoods"],
                item["candidate_trios"],
                likelihood_kind=str(item.get("likelihood_kind", "PL")),
                positions=item.get("positions"),
                linked_log_likelihoods=item.get("linked_log_likelihoods"),
                allele_frequencies=item.get("allele_frequencies"),
                frequency_em_iterations=int(item.get("frequency_em_iterations", 12)),
            ))
        return evidence

    bcf_direct = [
        all(key in item for key in ("vcf_path", "contig", "candidate_trios"))
        for item in contig_data_list
    ]
    if any(bcf_direct) and not all(bcf_direct):
        raise SmartEvidenceError(
            "explicit BCF/VCF evidence must be supplied for every contig or none"
        )
    if all(bcf_direct):
        if sample_ids is None:
            raise SmartEvidenceError("sample_ids are required for BCF/VCF loading")
        evidence = []
        for item in contig_data_list:
            likelihoods, positions = load_bcf_genotype_likelihoods(
                item["vcf_path"],
                str(item["contig"]),
                sample_ids,
                selected_positions=item.get("positions"),
                threads=int(item.get("bcf_threads", 1)),
            )
            evidence.append(score_genotype_likelihood_evidence(
                str(item["contig"]),
                likelihoods,
                item["candidate_trios"],
                likelihood_kind="PL",
                positions=positions,
                linked_log_likelihoods=item.get("linked_log_likelihoods"),
                allele_frequencies=item.get("allele_frequencies"),
                frequency_em_iterations=int(item.get("frequency_em_iterations", 12)),
            ))
        return evidence
    return None


def _infer_recognized_smart_evidence(
    evidence: Sequence[SmartContigEvidence | SmartParentStateEvidence],
    sample_ids: Sequence[Any],
    settings: SmartPedigreeConfig,
) -> PedigreeResult:
    """Dispatch explicit evidence without overstating pair-only capabilities."""
    if evidence and all(
        isinstance(item, SmartParentStateEvidence) for item in evidence
    ):
        return infer_from_parent_state_evidence(
            evidence, sample_ids, config=settings
        )
    if any(isinstance(item, SmartParentStateEvidence) for item in evidence):
        raise SmartEvidenceError(
            "cannot mix parent-state and pair-only evidence across contigs"
        )
    result = infer_from_contig_evidence(
        evidence, sample_ids, config=settings
    )
    result.smart_parent_state_model = False
    result.smart_pair_only_compatibility_mode = True
    result.smart_missing_parent_model = (
        "Explicit SmartContigEvidence/raw-GL inputs contain only M2 pair "
        "scores. This compatibility path remains forced-pair and cannot infer "
        "zero or one observed parent; use SmartParentStateEvidence for the "
        "new missing-parent model."
    )
    result.smart_limitations += (
        " This explicit input used pair-only compatibility mode; conclusions "
        "about zero/one observed parents are unavailable."
    )
    return result


def infer_pedigree_smart(
    contig_data_list: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[Any],
    *,
    config: Optional[SmartPedigreeConfig] = None,
    require_smart: bool = False,
    legacy_kwargs: Optional[Mapping[str, Any]] = None,
) -> PedigreeResult:
    """Run explicit evidence, standard painting evidence, or legacy fallback."""
    evidence = _explicit_evidence_from_inputs(contig_data_list, sample_ids)
    settings = _config_from_contig_inputs(contig_data_list, config)
    if evidence is not None:
        return _infer_recognized_smart_evidence(
            evidence, sample_ids, settings
        )
    if _standard_input_schema(contig_data_list):
        kwargs = dict(legacy_kwargs or {})
        result = _infer_standard_inputs(
            contig_data_list,
            sample_ids,
            settings,
            top_k=kwargs.pop("top_k", 20),
            snps_per_bin=kwargs.pop("snps_per_bin", 100),
            recomb_rate=kwargs.pop("recomb_rate", 5e-8),
            mismatch_penalty=kwargs.pop(
                "mismatch_penalty", DEFAULT_MISMATCH_PENALTY
            ),
            max_snps_per_bin=kwargs.pop("max_snps_per_bin", 10),
            n_workers=kwargs.pop("n_workers", None),
            anchor_k=kwargs.pop("anchor_k", 5),
            use_anchor_union=kwargs.pop("use_anchor_union", True),
            apply_consistency_cutoff=kwargs.pop(
                "apply_consistency_cutoff", True
            ),
        )
        kwargs.pop("collect_trio_candidates", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected legacy_kwargs: {unexpected}")
        return result
    if require_smart:
        raise SmartEvidenceError(
            "smart inference requires standard tolerance-painting inputs or "
            "explicit numerical evidence"
        )
    kwargs = dict(legacy_kwargs or {})
    return _legacy.infer_pedigree_multi_contig_tolerance(
        contig_data_list, sample_ids, **kwargs
    )


_PIPELINE_CONTROL_SENTINEL = "0"
_PIPELINE_CONTROL_WARNING = (
    "pipeline_control_adapter: result.relationships is an internal control "
    "table for the unchanged phase-correction pipeline, not an export "
    "pedigree. Use result.scientific_relationships for scientific reporting."
)
_CONDITIONAL_SCREEN_WARNING = (
    "M2 identity stability is conditional on the fixed candidate panel; "
    "integrated M2 evidence is a lower bound when that panel is incomplete. "
    "State support and resampling fractions are not calibrated biological "
    "posterior probabilities."
)
_PIPELINE_DIAGNOSTIC_COLUMNS = (
    "Identifiable",
    "InformativeContigCount",
    "PairBootstrapFraction",
    "Parent1BootstrapFraction",
    "Parent2BootstrapFraction",
    "PairLOCOFraction",
    "Parent1LOCOFraction",
    "Parent2LOCOFraction",
    "UnconstrainedWinnerMargin",
    "SelectedAggregateUtility",
    "SelectedMinusUnconstrainedBest",
    "CandidatePairCount",
    "UnresolvedReason",
    "PairSupportSet",
    "TierAExactPair",
    "TierBExactPair",
    "SelectedParentState",
    "LocalWinnerParentState",
    "GraphTieConflict",
    "GraphDirectionResolvedAlternative",
    "SelectedParentRoleProbability",
    "AdjustedAncestryJunctionBurden",
    "AncestryPaintingCallabilityFraction",
    "LatentAncestryDepthMAP",
    "LatentAncestryDepthPosterior",
    "LocalStateBootstrapFraction",
    "LocalConfigurationBootstrapFraction",
    "GraphConfigurationBootstrapFraction",
    "LocalStateLOCOFraction",
    "M2StateEvidenceIsLowerBound",
    "TierAStateCall",
    "TierBStateCall",
)


def _require_pipeline_sentinel_available(sample_ids: Sequence[Any]) -> None:
    if any(str(sample) == _PIPELINE_CONTROL_SENTINEL for sample in sample_ids):
        raise SmartEvidenceError(
            "sample ID string '0' is reserved by the exact-signature "
            "pipeline_control_adapter"
        )


def _attach_pipeline_control_adapter(result: PedigreeResult) -> PedigreeResult:
    """Guard the unchanged downstream pipeline from unsupported parent rows.

    Only a Tier-B-supported, resolved, non-displaced M2 call is representable
    by the historical phase-correction interface. M0, M1, unresolved, weak
    complete-view, and graph-displaced rows
    receive the control sentinel.  In particular M0 is *not* converted to a
    both-null F1 row: zero observed parents does not establish founder-recolour
    eligibility when biological parents may simply be unsequenced.
    """
    scientific = result.relationships.copy(deep=True)
    expected_samples = list(result.samples)
    if (
        len(scientific) != len(expected_samples)
        or scientific["Sample"].tolist() != expected_samples
    ):
        raise SmartEvidenceError(
            "smart scientific relationships must retain every sample in order"
        )
    has_first = scientific["Parent1"].notna()
    has_second = scientific["Parent2"].notna()
    has_state_model = "ParentState" in scientific.columns
    diagnostics = result.smart_diagnostics
    if diagnostics["Sample"].duplicated().any():
        raise SmartEvidenceError("smart diagnostics contain duplicate samples")
    diagnostic_lookup = diagnostics.set_index("Sample")
    tier_column = (
        "TierBExactConfiguration" if has_state_model else "TierBExactPair"
    )
    if tier_column not in diagnostic_lookup.columns:
        raise SmartEvidenceError(
            f"smart diagnostics lack required adapter gate {tier_column!r}"
        )
    tier_b_exact = scientific["Sample"].map(
        diagnostic_lookup[tier_column]
    )
    if tier_b_exact.isna().any():
        raise SmartEvidenceError(
            "smart diagnostics do not cover every scientific sample"
        )
    tier_b_exact = tier_b_exact.astype(bool)
    if has_state_model:
        parent_state = scientific["ParentState"].astype(object)
        zero = parent_state.eq(_PARENT_STATE_NAMES[_ZERO_OBSERVED])
        one = parent_state.eq(_PARENT_STATE_NAMES[_ONE_OBSERVED])
        two = parent_state.eq(_PARENT_STATE_NAMES[_TWO_OBSERVED])
        unresolved_state = parent_state.eq("unresolved") | parent_state.isna()
        if np.any(zero & (has_first | has_second)):
            raise SmartEvidenceError("M0 scientific rows cannot contain parents")
        if np.any(one & has_first & has_second):
            raise SmartEvidenceError(
                "M1 scientific rows cannot contain two observed parents"
            )
        if np.any(two & (has_first ^ has_second)):
            raise SmartEvidenceError(
                "M2 scientific rows need an exact pair or unresolved identity"
            )
        if np.any(unresolved_state & (has_first | has_second)):
            raise SmartEvidenceError(
                "unresolved scientific rows cannot contain observed parents"
            )
        scientific_status = scientific["InferenceStatus"].astype(str)
        disallowed = scientific_status.str.contains(
            "graph_displaced|graph_conflict|unresolved", regex=True
        )
        pass_to_pipeline = (
            two & has_first & has_second & ~disallowed & tier_b_exact
        )
    else:
        if not np.array_equal(has_first.to_numpy(), has_second.to_numpy()):
            raise SmartEvidenceError(
                "pair-only scientific relationships need exact pairs or no parents"
            )
        parent_state = pd.Series(
            np.where(has_first, "pair_only_m2", "pair_only_unresolved"),
            index=scientific.index,
            dtype=object,
        )
        scientific_status = pd.Series(
            np.where(has_first, "resolved_exact_pair", "unresolved_or_unscored"),
            index=scientific.index,
            dtype=object,
        )
        pass_to_pipeline = has_first & has_second & tier_b_exact

    scientific.attrs["table_role"] = "scientific_relationships"
    scientific.attrs["warning"] = (
        "ParentState distinguishes zero observed parents from unresolved "
        "identity where the parent-state model is available. Zero observed "
        "parents does not establish biological founder status."
    )
    adapter = scientific.copy(deep=True)
    adapter["ScientificParent1"] = scientific["Parent1"].astype(object)
    adapter["ScientificParent2"] = scientific["Parent2"].astype(object)
    adapter["ScientificParentState"] = parent_state
    adapter["ScientificInferenceStatus"] = scientific_status
    adapter["InferenceStatus"] = np.where(
        pass_to_pipeline,
        "resolved_exact_pair",
        "pipeline_control_sentinel_" + parent_state.astype(str),
    )

    for column in _PIPELINE_DIAGNOSTIC_COLUMNS:
        if column in diagnostic_lookup.columns:
            adapter[column] = adapter["Sample"].map(
                diagnostic_lookup[column]
            )
    adapter["ConditionalScreenWarning"] = _CONDITIONAL_SCREEN_WARNING
    adapter["ParentFieldRole"] = "pipeline_control_adapter_not_export"
    adapter["PipelineAdapterWarning"] = _PIPELINE_CONTROL_WARNING

    sentinel_rows = ~pass_to_pipeline
    adapter["Parent1"] = adapter["Parent1"].astype(object)
    adapter["Parent2"] = adapter["Parent2"].astype(object)
    adapter.loc[sentinel_rows, "Parent1"] = None
    adapter.loc[sentinel_rows, "Parent2"] = _PIPELINE_CONTROL_SENTINEL
    adapter.loc[sentinel_rows, "Generation"] = "Unknown"
    adapter.attrs["table_role"] = "pipeline_control_adapter"
    adapter.attrs["warning"] = _PIPELINE_CONTROL_WARNING

    result.scientific_relationships = scientific
    result.relationships = adapter
    result.pipeline_control_adapter = True
    result.pipeline_control_adapter_name = "pipeline_control_adapter"
    result.pipeline_control_adapter_warning = _PIPELINE_CONTROL_WARNING
    result.pipeline_control_sentinel = _PIPELINE_CONTROL_SENTINEL
    result.relationships_is_export_pedigree = False
    result.pipeline_control_status_summary = {
        str(name): int(count)
        for name, count in adapter["InferenceStatus"].value_counts().items()
    }
    warnings.warn(_PIPELINE_CONTROL_WARNING, RuntimeWarning, stacklevel=2)
    return result


def infer_pedigree_multi_contig_tolerance(
    contig_data_list,
    sample_ids,
    top_k=20,
    snps_per_bin=100,
    recomb_rate=5e-8,
    mismatch_penalty=DEFAULT_MISMATCH_PENALTY,
    max_snps_per_bin=10,
    n_workers=None,
    anchor_k=5,
    use_anchor_union=True,
):
    """Exact-signature replacement for the historical pipeline entry point.

    Recognised smart inputs return a guarded ``pipeline_control_adapter`` in
    ``relationships`` so unchanged phase-correction callers treat unresolved
    samples as neither trios nor inferred F1s. The clean all-sample pedigree is
    retained in ``scientific_relationships`` and must be used for export.
    Historical tolerance-painting inputs automatically activate a fixed-panel,
    per-contig linked-HMM analysis followed by chromosome bootstrap, LOCO, and
    joint acyclic selection. Unknown nonstandard schemas alone retain the
    compatibility fallback to :mod:`pedigree_inference`.
    """
    explicit = _explicit_evidence_from_inputs(contig_data_list, sample_ids)
    settings = _config_from_contig_inputs(contig_data_list)
    if explicit is not None:
        _require_pipeline_sentinel_available(sample_ids)
        result = _infer_recognized_smart_evidence(
            explicit, sample_ids, settings
        )
        return _attach_pipeline_control_adapter(result)
    if _standard_input_schema(contig_data_list):
        _require_pipeline_sentinel_available(sample_ids)
        result = _infer_standard_inputs(
            contig_data_list,
            sample_ids,
            settings,
            top_k=top_k,
            snps_per_bin=snps_per_bin,
            recomb_rate=recomb_rate,
            mismatch_penalty=mismatch_penalty,
            max_snps_per_bin=max_snps_per_bin,
            n_workers=n_workers,
            anchor_k=anchor_k,
            use_anchor_union=use_anchor_union,
            apply_consistency_cutoff=True,
        )
        return _attach_pipeline_control_adapter(result)
    return _legacy.infer_pedigree_multi_contig_tolerance(
        contig_data_list,
        sample_ids,
        top_k=top_k,
        snps_per_bin=snps_per_bin,
        recomb_rate=recomb_rate,
        mismatch_penalty=mismatch_penalty,
        max_snps_per_bin=max_snps_per_bin,
        n_workers=n_workers,
        anchor_k=anchor_k,
        use_anchor_union=use_anchor_union,
    )


def draw_pedigree_tree(relationships_df, output_file="pedigree_tree.png"):
    """Draw scientific parent fields, never pipeline-control sentinels."""
    has_first = "ScientificParent1" in relationships_df.columns
    has_second = "ScientificParent2" in relationships_df.columns
    if has_first != has_second:
        raise SmartEvidenceError(
            "adapter drawing requires both ScientificParent columns"
        )
    if not has_first:
        return _legacy.draw_pedigree_tree(relationships_df, output_file)

    sanitized = relationships_df.copy(deep=True)
    sanitized["Parent1"] = sanitized["ScientificParent1"]
    sanitized["Parent2"] = sanitized["ScientificParent2"]
    parent_values = pd.concat(
        (sanitized["Parent1"], sanitized["Parent2"]), ignore_index=True
    )
    if parent_values.astype(object).eq(_PIPELINE_CONTROL_SENTINEL).any():
        raise SmartEvidenceError(
            "reserved pipeline-control sentinel present in scientific parents"
        )
    return _legacy.draw_pedigree_tree(sanitized, output_file)


def __getattr__(name: str) -> Any:
    """Delegate non-overridden attributes to the historical module."""
    return getattr(_legacy, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_legacy)))


__all__ = sorted({
    name for name in dir(_legacy) if not name.startswith("_")
} | {
    "SmartContigEvidence",
    "SmartParentStateEvidence",
    "SmartEvidenceError",
    "SmartPedigreeConfig",
    "infer_from_contig_evidence",
    "infer_from_parent_state_evidence",
    "infer_pedigree_smart",
    "load_bcf_genotype_likelihoods",
    "score_genotype_likelihood_evidence",
    "score_parent_state_hmms",
    "smart_numba_thread_capacity",
})
