"""Complete-mode proposals and fitting for block-haplotype discovery.

This module addresses a specific local optimum in block-haplotype discovery:
when most observed diplotypes form a bipartite family graph, independently
updating one founder at a time cannot cross a site-wise gauge transformation
that must change several founders simultaneously.  The implementation here
keeps *complete* fixed-K factorisations together, applies the joint
bipartite-column move, and carries a bounded beam of those complete modes.
Rows from incompatible modes are never pooled into a synthetic haplotype set.

The production reversible-cavity search uses these utilities to create,
canonicalize, refit, and deduplicate coherent local optima.  Complete modes
are kept intact throughout: rows from incompatible modes are never pooled
into a synthetic haplotype set.

No API accepts sample metadata, founder identities, cohort labels, pedigree
labels, simulated truth, or an expected K.  ``max_k`` is only a computational
safety cap on represented modes.

The inference evidence is expected to be normalized *raw genotype
likelihoods* with shape ``(samples, sites, 3)``.  Population/HWE posteriors
must not be supplied as though they were independent read evidence.

The exact maximum-cut search is intentionally bounded.  For larger K a
deterministic multi-start single-vertex local search is used; the optimized
implementation retains a reference path and bounded regression self-test.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
from numba import njit

from bhd_genotype_evidence import validate_normalized_genotype_evidence
from bhd_mode_canonicalization import (
    canonicalize_binary_panel,
    exact_unique_binary_rows,
)

from bhd_config import (
    DEFAULT_LAMBDA,
    FIXED_K_FIT_MAX_THREADS,
    DEFAULT_DATA_SEED_MODES,
    DEFAULT_SOFT_SEED_MIN_CLUSTER_SIZE,
)


DEFAULT_EXACT_CUT_MAX_K = 12
DEFAULT_MAX_CUT_TIES = 4
_MAX_CUT_CACHE_SIZE = 256
_TINY = np.finfo(np.float64).tiny
_NLL_TOLERANCE = 1e-10

# Cached values are immutable byte strings rather than arrays so callers can
# freely mutate the fresh arrays returned by ``maximum_cut_partitions``.
_MAX_CUT_CACHE: OrderedDict[
    tuple[int, int, int, bytes], tuple[bytes, ...]
] = OrderedDict()


@dataclass(frozen=True)
class _CanonicalMode:
    """Canonical probability-panel identity used by scientific scorers."""

    k: int
    haplotypes: np.ndarray
    key: bytes
    digest: str


def _extract_mode_haplotypes(mode: Any) -> np.ndarray:
    """Return a normalized probability matrix from a supported mode value."""

    if isinstance(mode, Mapping):
        if set(mode) != {"haplotypes"}:
            raise ValueError(
                "mapping mode inputs may contain only the 'haplotypes' key"
            )
        value = mode["haplotypes"]
    elif isinstance(mode, np.ndarray):
        value = mode
    elif hasattr(mode, "haplotypes"):
        value = mode.haplotypes
    else:
        raise TypeError(
            "a mode must be an array, {'haplotypes': array}, or an object "
            "with a haplotypes attribute"
        )

    array = np.asarray(value)
    if array.ndim == 3 and array.shape[2] == 2:
        allele_weights = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(allele_weights)):
            raise ValueError("allele-mass mode values must be finite")
        if np.any(allele_weights < 0.0):
            raise ValueError("allele-mass mode values must be non-negative")
        allele_mass = np.sum(allele_weights, axis=2)
        if np.any(allele_mass <= 0.0):
            raise ValueError(
                "every haplotype/site must have positive allele mass"
            )
        q = allele_weights[..., 1] / allele_mass
    elif array.ndim == 2:
        q = np.asarray(array, dtype=np.float64)
    else:
        raise ValueError(
            "mode haplotypes must have shape (K, sites) or (K, sites, 2)"
        )
    if q.shape[0] < 1 or q.shape[1] < 1:
        raise ValueError("a mode must contain haplotypes and sites")
    if not np.all(np.isfinite(q)):
        raise ValueError("haplotype probabilities must be finite")
    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("haplotype probabilities must lie in [0, 1]")
    # Canonicalization mutates negative zero; copy because diagnostic mode
    # arrays are intentionally read-only and np.ascontiguousarray may alias.
    q = np.array(q, dtype=np.float64, order="C", copy=True)
    q[q == 0.0] = 0.0
    return q


def _canonicalize_mode(mode: Any, expected_k: int) -> _CanonicalMode:
    """Canonicalize a hard or soft panel without changing digest semantics."""

    haplotypes = _extract_mode_haplotypes(mode)
    if len(haplotypes) != expected_k:
        raise ValueError(
            f"mode stored under K={expected_k} contains "
            f"{len(haplotypes)} haplotypes"
        )
    order = sorted(
        range(expected_k),
        key=lambda index: tuple(float(x) for x in haplotypes[index]),
    )
    canonical = np.ascontiguousarray(haplotypes[order], dtype=np.float64)
    if expected_k > 1 and any(
        np.array_equal(canonical[index - 1], canonical[index])
        for index in range(1, expected_k)
    ):
        raise ValueError(
            "a complete K-mode cannot contain duplicate haplotype rows"
        )
    shape_prefix = np.asarray(canonical.shape, dtype="<i8").tobytes()
    key = shape_prefix + canonical.astype("<f8", copy=False).tobytes()
    digest = hashlib.sha256(key).hexdigest()[:20]
    canonical.setflags(write=False)
    return _CanonicalMode(expected_k, canonical, key, digest)


@dataclass(frozen=True)
class FixedKPanelFitConfig:
    """Settings needed to refit reversible-search panel proposals."""

    lambda_wildcard_penalty: float = 0.5
    coordinate_descent_max_iter: int = 50

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.lambda_wildcard_penalty)
            or self.lambda_wildcard_penalty < 0.0
        ):
            raise ValueError(
                "lambda_wildcard_penalty must be finite and non-negative"
            )
        if (
            isinstance(self.coordinate_descent_max_iter, bool)
            or int(self.coordinate_descent_max_iter) < 1
        ):
            raise ValueError(
                "coordinate_descent_max_iter must be a positive integer"
            )


@dataclass(frozen=True)
class FactorizationMode:
    """One coherent fixed-K factorisation.

    ``haplotypes`` are stored in deterministic lexicographic row order and
    ``assignments`` are remapped to that order.  The wildcard sentinel is K.
    Arrays are private copies marked read-only so a mode cannot be silently
    mutated after its canonical key has been used for deduplication.
    """

    haplotypes: np.ndarray
    assignments: np.ndarray
    per_sample_cost: np.ndarray
    wildcard_slots: np.ndarray
    n_iter: int
    total_nll: float
    fixed_point_certified: bool = field(
        default=False, repr=False, compare=False
    )
    _canonical_key: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._initialize(
            np.array(
                self.haplotypes, dtype=np.int64, order="C", copy=True
            ),
            np.array(
                self.assignments, dtype=np.int64, order="C", copy=True
            ),
            np.array(
                self.per_sample_cost,
                dtype=np.float64,
                order="C",
                copy=True,
            ),
            np.array(
                self.wildcard_slots,
                dtype=np.int64,
                order="C",
                copy=True,
            ),
            int(self.n_iter),
            float(self.total_nll),
            bool(self.fixed_point_certified),
        )

    def _initialize(
        self,
        haplotypes: np.ndarray,
        assignments: np.ndarray,
        costs: np.ndarray,
        wildcard_slots: np.ndarray,
        n_iter: int,
        total_nll: float,
        fixed_point_certified: bool,
        *,
        canonical_key: bytes | None = None,
    ) -> None:
        if haplotypes.ndim != 2 or len(haplotypes) < 1:
            raise ValueError("haplotypes must have shape (K, sites), K >= 1")
        if haplotypes.size and (
            int(np.min(haplotypes)) < 0 or int(np.max(haplotypes)) > 1
        ):
            raise ValueError("haplotypes must be binary")
        if assignments.ndim != 2 or assignments.shape[1] != 2:
            raise ValueError("assignments must have shape (samples, 2)")
        if len(costs) != len(assignments) or len(wildcard_slots) != len(
            assignments
        ):
            raise ValueError("sample-level arrays have inconsistent lengths")
        if assignments.size and (
            int(np.min(assignments)) < 0
            or int(np.max(assignments)) > len(haplotypes)
        ):
            raise ValueError("assignment index lies outside [0, K]")
        if np.any(assignments[:, 0] > assignments[:, 1]):
            raise ValueError("assignment pairs must be sorted")
        if wildcard_slots.size and (
            int(np.min(wildcard_slots)) < 0
            or int(np.max(wildcard_slots)) > 2
        ):
            raise ValueError("wildcard slot counts must lie in [0, 2]")
        expected_wildcard_slots = np.count_nonzero(
            assignments == len(haplotypes), axis=1
        )
        if not np.array_equal(wildcard_slots, expected_wildcard_slots):
            raise ValueError("wildcard_slots disagree with assignment sentinels")
        if n_iter < 0:
            raise ValueError("n_iter must be non-negative")
        if not math.isfinite(total_nll):
            raise ValueError("total_nll must be finite")

        for value in (haplotypes, assignments, costs, wildcard_slots):
            value.setflags(write=False)
        object.__setattr__(self, "haplotypes", haplotypes)
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "per_sample_cost", costs)
        object.__setattr__(self, "wildcard_slots", wildcard_slots)
        object.__setattr__(self, "n_iter", n_iter)
        object.__setattr__(self, "total_nll", total_nll)
        object.__setattr__(
            self, "fixed_point_certified", fixed_point_certified
        )
        object.__setattr__(
            self,
            "_canonical_key",
            canonical_key
            if canonical_key is not None
            else haplotypes.astype(np.int8, copy=False).tobytes(),
        )

    @classmethod
    def _from_owned_arrays(
        cls,
        haplotypes: np.ndarray,
        assignments: np.ndarray,
        per_sample_cost: np.ndarray,
        wildcard_slots: np.ndarray,
        n_iter: int,
        total_nll: float,
        fixed_point_certified: bool,
        canonical_key: bytes,
    ) -> FactorizationMode:
        """Build from fresh private arrays while retaining full validation."""

        mode = object.__new__(cls)
        mode._initialize(
            haplotypes,
            assignments,
            per_sample_cost,
            wildcard_slots,
            n_iter,
            total_nll,
            fixed_point_certified,
            canonical_key=canonical_key,
        )
        return mode

    @property
    def k(self) -> int:
        return int(self.haplotypes.shape[0])

    @property
    def n_sites(self) -> int:
        return int(self.haplotypes.shape[1])

    @property
    def canonical_key(self) -> bytes:
        return self._canonical_key


@dataclass(frozen=True)
class GaugeRewireProposal:
    """One coherent joint-column start produced from a fitted mode."""

    haplotypes: np.ndarray
    partition: np.ndarray
    gauge_sites: np.ndarray
    proposal_kind: str
    n_flipped_sites: int
    cross_assignment_weight: int
    within_assignment_weight: int
    conditional_nll_before: float
    conditional_nll_after: float

    def __post_init__(self) -> None:
        haplotypes = np.array(
            self.haplotypes, dtype=np.int64, order="C", copy=True
        )
        partition = np.array(
            self.partition, dtype=bool, order="C", copy=True
        )
        gauge_sites = np.array(
            self.gauge_sites, dtype=bool, order="C", copy=True
        )
        if haplotypes.ndim != 2:
            raise ValueError("proposal haplotypes must be two-dimensional")
        if partition.shape != (len(haplotypes),):
            raise ValueError("partition length must equal K")
        if gauge_sites.shape != (haplotypes.shape[1],):
            raise ValueError("gauge-site mask length must equal site count")
        if self.proposal_kind not in {"conditional_sitewise", "opposite_endpoint"}:
            raise ValueError("unknown gauge proposal kind")
        for value in (haplotypes, partition, gauge_sites):
            value.setflags(write=False)
        object.__setattr__(self, "haplotypes", haplotypes)
        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "gauge_sites", gauge_sites)


@dataclass(frozen=True)
class CompleteModeBeam:
    """Complete modes retained at every represented K."""

    modes_by_k: tuple[tuple[FactorizationMode, ...], ...]
    beam_width: int
    max_k_cap: int
    raw_trial_counts: tuple[int, ...]
    gauge_trial_counts: tuple[int, ...]
    growth_seed_attempt_counts: tuple[int, ...]
    synchronized_fallback_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.beam_width < 1 or self.max_k_cap < 1:
            raise ValueError("beam width and max-K cap must be positive")
        if not self.modes_by_k:
            raise ValueError("a beam must contain at least K=1")
        if len(self.raw_trial_counts) != len(self.modes_by_k):
            raise ValueError("raw trial diagnostics have the wrong length")
        if len(self.gauge_trial_counts) != len(self.modes_by_k):
            raise ValueError("gauge trial diagnostics have the wrong length")
        if len(self.growth_seed_attempt_counts) != len(self.modes_by_k):
            raise ValueError(
                "growth-seed attempt diagnostics have the wrong length"
            )
        if len(self.synchronized_fallback_counts) != len(self.modes_by_k):
            raise ValueError(
                "synchronized-fallback diagnostics have the wrong length"
            )
        for expected_k, modes in enumerate(self.modes_by_k, start=1):
            if not modes:
                raise ValueError(f"beam is empty at K={expected_k}")
            if len(modes) > self.beam_width:
                raise ValueError("beam exceeds configured width")
            if any(mode.k != expected_k for mode in modes):
                raise ValueError("mode stored under the wrong K")

    @property
    def k_values(self) -> tuple[int, ...]:
        return tuple(range(1, len(self.modes_by_k) + 1))

    def modes(self, k: int) -> tuple[FactorizationMode, ...]:
        if k < 1 or k > len(self.modes_by_k):
            raise KeyError(k)
        return self.modes_by_k[k - 1]


def _validate_evidence(
    evidence: np.ndarray,
    *,
    n_sites: int | None = None,
    n_samples: int | None = None,
) -> np.ndarray:
    return validate_normalized_genotype_evidence(
        evidence, n_sites=n_sites, n_samples=n_samples
    )


def _canonicalize_fit(
    fit: Sequence[Any],
    *,
    fit_workspace: Any | None = None,
) -> FactorizationMode:
    if len(fit) != 6:
        raise ValueError("fixed-K fit must contain six fields")
    haplotypes = np.asarray(fit[0], dtype=np.int64)
    assignments = np.asarray(fit[1], dtype=np.int64)
    k = len(haplotypes)
    valid_assignment_indices = (
        assignments.ndim == 2
        and assignments.shape[1] == 2
        and (
            assignments.size == 0
            or (
                int(np.min(assignments)) >= 0
                and int(np.max(assignments)) <= k
            )
        )
    )
    if haplotypes.ndim == 2 and valid_assignment_indices:
        (
            canonical_haplotypes,
            canonical_assignments,
            _order,
            _inverse,
            canonical_key,
        ) = canonicalize_binary_panel(haplotypes, assignments)
    else:
        # Retain the historical validation path for malformed fits.
        byte_rows = np.asarray(haplotypes, dtype=np.int8)
        order = np.asarray(
            sorted(
                range(k),
                key=lambda index: (byte_rows[index].tobytes(), index),
            ),
            dtype=np.int64,
        )
        inverse = np.empty(k, dtype=np.int64)
        inverse[order] = np.arange(k, dtype=np.int64)
        canonical_assignments = assignments.copy()
        real = canonical_assignments < k
        canonical_assignments[real] = inverse[canonical_assignments[real]]
        canonical_assignments.sort(axis=1)
        canonical_haplotypes = haplotypes[order]
        canonical_key = np.ascontiguousarray(byte_rows[order]).tobytes()
    fixed_point_certified = bool(
        fit_workspace is not None
        and fit_workspace.certifies_fixed_point(canonical_haplotypes.copy())
    )
    return FactorizationMode._from_owned_arrays(
        canonical_haplotypes,
        canonical_assignments,
        np.array(fit[2], dtype=np.float64, order="C", copy=True),
        np.array(fit[3], dtype=np.int64, order="C", copy=True),
        int(fit[4]),
        float(fit[5]),
        fixed_point_certified,
        canonical_key,
    )


def _has_distinct_rows(mode: FactorizationMode) -> bool:
    row_bytes = mode.n_sites
    key = mode.canonical_key
    if row_bytes == 0:
        return mode.k == 1
    return len({
        key[start:start + row_bytes]
        for start in range(0, len(key), row_bytes)
    }) == mode.k


def _deduplicate_modes(
    modes: Sequence[FactorizationMode],
    beam_width: int | None = None,
) -> tuple[FactorizationMode, ...]:
    unique: dict[bytes, FactorizationMode] = {}
    for mode in modes:
        if not _has_distinct_rows(mode):
            continue
        previous = unique.get(mode.canonical_key)
        if previous is None or mode.total_nll < previous.total_nll - _NLL_TOLERANCE:
            unique[mode.canonical_key] = mode
    ordered = sorted(
        unique.values(),
        key=lambda mode: (mode.total_nll, mode.canonical_key),
    )
    if beam_width is not None:
        ordered = ordered[:beam_width]
    return tuple(ordered)


def _unique_binary_rows(matrix: np.ndarray) -> np.ndarray:
    """Return distinct binary rows in exact NumPy lexicographic order."""

    return exact_unique_binary_rows(matrix)


def _fit_starts_with_synchronized_endpoints(
    genotype_likelihoods: np.ndarray,
    starts: Sequence[np.ndarray],
    config: FixedKPanelFitConfig,
    workspace: Any,
    *,
    max_iter: int | None = None,
) -> tuple[tuple[FactorizationMode, ...], tuple[FactorizationMode, ...]]:
    """Fit starts once, returning exact raw and refitted mode collections.

    For each original distinct-row start, the first assignment update is the
    synchronized ``max_iter=0`` endpoint. Coordinate descent then continues
    from that same update. Panels collapsed by the refit follow the ordinary
    lower-K recursion and contribute only to the final collection.
    """

    from bhd_fit import (
        _fit_at_fixed_K_many,
        _fit_at_fixed_K_many_with_initial,
    )

    iterations = (
        config.coordinate_descent_max_iter
        if max_iter is None
        else int(max_iter)
    )
    pending: list[np.ndarray] = []
    seen_starts: set[tuple[tuple[int, int], bytes]] = set()
    expected_raw_keys: set[bytes] = set()
    initially_certified: dict[tuple[tuple[int, int], bytes], bool] = {}
    for start in starts:
        h = np.ascontiguousarray(np.asarray(start), dtype=np.int64)
        if h.ndim != 2 or h.shape[1] != genotype_likelihoods.shape[1]:
            raise ValueError("all starts must match the evidence sites")
        if np.any((h != 0) & (h != 1)):
            raise ValueError("all starts must be hard binary matrices")
        h = np.ascontiguousarray(_unique_binary_rows(h), dtype=np.int64)
        key = (tuple(int(value) for value in h.shape), h.tobytes())
        if key in seen_starts:
            continue
        seen_starts.add(key)
        pending.append(h)
        expected_raw_keys.add(h.astype(np.int8, copy=False).tobytes())
        # Snapshot the pre-call state so raw endpoints retain their historical
        # fixed-point certification even when the final fit populates the cache.
        initially_certified[key] = workspace.certifies_fixed_point(h)

    raw_modes: list[FactorizationMode] = []
    fitted: list[FactorizationMode] = []
    capture_initial = True
    while pending:
        grouped: dict[int, list[np.ndarray]] = {}
        for h in pending:
            grouped.setdefault(len(h), []).append(h)
        pending = []
        for k in sorted(grouped):
            group_starts = grouped[k]
            if capture_initial:
                fit_records = _fit_at_fixed_K_many_with_initial(
                    genotype_likelihoods,
                    group_starts,
                    config.lambda_wildcard_penalty,
                    max_iter=iterations,
                    workspace=workspace,
                )
            else:
                final_fits = _fit_at_fixed_K_many(
                    genotype_likelihoods,
                    group_starts,
                    config.lambda_wildcard_penalty,
                    max_iter=iterations,
                    workspace=workspace,
                )
                fit_records = tuple((None, fit) for fit in final_fits)

            for start, (initial_fit, final_fit) in zip(
                group_starts, fit_records
            ):
                if initial_fit is not None:
                    start_key = (
                        tuple(int(value) for value in start.shape),
                        start.tobytes(),
                    )
                    raw_workspace = (
                        workspace if initially_certified[start_key] else None
                    )
                    raw_modes.append(
                        _canonicalize_fit(
                            initial_fit, fit_workspace=raw_workspace
                        )
                    )

                mode = _canonicalize_fit(
                    final_fit, fit_workspace=workspace
                )
                distinct = np.ascontiguousarray(
                    _unique_binary_rows(mode.haplotypes), dtype=np.int64
                )
                if len(distinct) == mode.k:
                    fitted.append(mode)
                    continue
                key = (
                    tuple(int(value) for value in distinct.shape),
                    distinct.tobytes(),
                )
                if key not in seen_starts:
                    seen_starts.add(key)
                    pending.append(distinct)
        capture_initial = False

    synchronized = _deduplicate_modes(raw_modes)
    observed_raw_keys = {mode.canonical_key for mode in synchronized}
    if any(mode.n_iter != 0 for mode in synchronized):
        raise AssertionError(
            "captured fixed-K synchronization executed a coordinate iteration"
        )
    if observed_raw_keys != expected_raw_keys:
        raise AssertionError(
            "captured fixed-K synchronization changed a raw haplotype panel"
        )
    return synchronized, _deduplicate_modes(fitted)


def _death_starts(mode: FactorizationMode) -> tuple[np.ndarray, ...]:
    """Return every one-row deletion from a fitted complete panel."""

    if mode.k <= 1:
        return ()
    return tuple(
        np.ascontiguousarray(np.delete(mode.haplotypes, index, axis=0))
        for index in range(mode.k)
    )


@njit(cache=True, nogil=True)
def _assignment_graph_kernel(
    assignments: np.ndarray,
    k: int,
) -> np.ndarray:
    weights = np.zeros((k, k), dtype=np.int64)
    for sample_index in range(len(assignments)):
        first = assignments[sample_index, 0]
        second = assignments[sample_index, 1]
        if first < k and second < k and first != second:
            weights[first, second] += 1
            weights[second, first] += 1
    return weights


def assignment_graph(mode: FactorizationMode) -> np.ndarray:
    """Count real-real, non-self assignments between haplotype rows."""

    return _assignment_graph_kernel(mode.assignments, mode.k)


@njit(cache=True, nogil=True)
def _exact_cut_score_table(
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score exact cuts in binary-candidate order via Gray-code updates.

    Vertex zero remains fixed on the first side.  Gray-code traversal changes
    one other vertex at a time, so its exact integer cut-score delta costs
    O(K), rather than rescoring all O(K^2) edges for every candidate.  Scores
    are written at ``bits - 1`` so callers observe the historical binary
    candidate order regardless of traversal order.
    """

    k = len(weights)
    n_candidates = (1 << (k - 1)) - 1
    cross_scores = np.empty(n_candidates, dtype=np.int64)
    within_scores = np.empty(n_candidates, dtype=np.int64)
    side = np.zeros(k, dtype=np.bool_)
    total_weight = 0
    for first in range(k - 1):
        for second in range(first + 1, k):
            total_weight += weights[first, second]

    cross = 0
    previous_gray = 0
    for traversal_index in range(1, 1 << (k - 1)):
        gray = traversal_index ^ (traversal_index >> 1)
        changed = gray ^ previous_gray
        bit_index = 0
        while (changed >> bit_index) != 1:
            bit_index += 1
        vertex = bit_index + 1

        delta = 0
        old_side = side[vertex]
        for other in range(k):
            if other == vertex:
                continue
            value = weights[vertex, other]
            if old_side == side[other]:
                delta += value
            else:
                delta -= value
        side[vertex] = not old_side
        cross += delta
        cross_scores[gray - 1] = cross
        within_scores[gray - 1] = total_weight - cross
        previous_gray = gray
    return cross_scores, within_scores

@njit(cache=True, nogil=True)
def _cut_score_kernel(
    weights: np.ndarray,
    side: np.ndarray,
) -> tuple[int, int]:
    cross = 0
    within = 0
    for first in range(len(weights) - 1):
        for second in range(first + 1, len(weights)):
            value = weights[first, second]
            if side[first] != side[second]:
                cross += value
            else:
                within += value
    return cross, within


@njit(cache=True, nogil=True)
def _locally_improve_cut_kernel(
    weights: np.ndarray,
    initial: np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    side = initial.copy()
    current_cross = _cut_score_kernel(weights, side)[0]
    while True:
        changed = False
        for order_index in range(len(order)):
            index = order[order_index]
            old_side = side[index]
            delta = 0
            for other in range(len(weights)):
                if other == index:
                    continue
                value = weights[index, other]
                if old_side == side[other]:
                    delta += value
                else:
                    delta -= value
            if delta > 0:
                side[index] = not old_side
                current_cross += delta
                changed = True
        if not changed:
            return side


def _cut_score(weights: np.ndarray, side: np.ndarray) -> tuple[int, int]:
    cross, within = _cut_score_kernel(weights, side)
    return int(cross), int(within)


def _locally_improve_cut(
    weights: np.ndarray,
    initial: np.ndarray,
    order: np.ndarray | None = None,
) -> np.ndarray:
    if order is None:
        order = np.asarray(sorted(
            range(1, len(weights)),
            key=lambda index: (-int(np.sum(weights[index])), index),
        ), dtype=np.int64)
    return _locally_improve_cut_kernel(weights, initial, order)


def _maximum_cut_cache_get(
    key: tuple[int, int, int, bytes],
) -> tuple[np.ndarray, ...] | None:
    encoded = _MAX_CUT_CACHE.get(key)
    if encoded is None:
        return None
    _MAX_CUT_CACHE.move_to_end(key)
    k = key[0]
    return tuple(
        np.frombuffer(value, dtype=np.bool_, count=k).copy()
        for value in encoded
    )


def _maximum_cut_cache_put(
    key: tuple[int, int, int, bytes],
    partitions: tuple[np.ndarray, ...],
) -> None:
    _MAX_CUT_CACHE[key] = tuple(side.tobytes() for side in partitions)
    _MAX_CUT_CACHE.move_to_end(key)
    while len(_MAX_CUT_CACHE) > _MAX_CUT_CACHE_SIZE:
        _MAX_CUT_CACHE.popitem(last=False)


def maximum_cut_partitions(
    weights: np.ndarray,
    *,
    exact_max_k: int = DEFAULT_EXACT_CUT_MAX_K,
    max_ties: int = DEFAULT_MAX_CUT_TIES,
) -> tuple[np.ndarray, ...]:
    """Return deterministic maximum-weight bipartitions.

    Complement-equivalent cuts are represented once by fixing vertex zero on
    side ``False``.  Exact enumeration is used through ``exact_max_k``.
    Larger graphs use deterministic degree-ordered local searches from
    singleton and alternating starts.  At most ``max_ties`` equal optima are
    returned, in lexicographic bit order.
    """

    matrix = np.asarray(weights, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("weights must be a square matrix")
    if len(matrix) < 2:
        return ()
    if np.any(matrix < 0) or not np.array_equal(matrix, matrix.T):
        raise ValueError("weights must be symmetric and non-negative")
    if np.any(np.diag(matrix) != 0):
        raise ValueError("cut graph must not contain self-loop weights")
    if exact_max_k < 2 or max_ties < 1:
        raise ValueError("invalid maximum-cut limits")

    matrix = np.ascontiguousarray(matrix)
    cache_key = (
        len(matrix), int(exact_max_k), int(max_ties), matrix.tobytes()
    )
    cached = _maximum_cut_cache_get(cache_key)
    if cached is not None:
        return cached

    candidates: list[np.ndarray] = []
    k = len(matrix)
    if k <= exact_max_k:
        cross_scores, within_scores = _exact_cut_score_table(matrix)
        best_cross = int(np.max(cross_scores))
        best_within = int(np.min(
            within_scores[cross_scores == best_cross]
        ))
        winner_indices = np.flatnonzero(
            (cross_scores == best_cross)
            & (within_scores == best_within)
        )
        winners = []
        for candidate_index in winner_indices:
            bits = int(candidate_index) + 1
            side = np.zeros(k, dtype=bool)
            for index in range(1, k):
                side[index] = bool(bits & (1 << (index - 1)))
            winners.append(side)
        winners.sort(key=lambda side: side.tobytes())
        result = tuple(side.copy() for side in winners[:max_ties])
        _maximum_cut_cache_put(cache_key, result)
        return result

    degree_order = np.asarray(sorted(
        range(1, k),
        key=lambda index: (-int(np.sum(matrix[index])), index),
    ), dtype=np.int64)
    for index in range(1, k):
        side = np.zeros(k, dtype=bool)
        side[index] = True
        candidates.append(_locally_improve_cut(
            matrix, side, degree_order
        ))
    alternating = np.zeros(k, dtype=bool)
    alternating[degree_order[::2]] = True
    candidates.append(_locally_improve_cut(
        matrix, alternating, degree_order
    ))

    unique: dict[bytes, np.ndarray] = {}
    for side in candidates:
        if not np.any(side):
            continue
        unique.setdefault(side.tobytes(), side)
    if not unique:
        result: tuple[np.ndarray, ...] = ()
        _maximum_cut_cache_put(cache_key, result)
        return result
    scored = [
        (_cut_score(matrix, side), side.tobytes(), side)
        for side in unique.values()
    ]
    best_cross = max(item[0][0] for item in scored)
    best_within = min(
        item[0][1] for item in scored if item[0][0] == best_cross
    )
    winners = [
        item
        for item in scored
        if item[0] == (best_cross, best_within)
    ]
    winners.sort(key=lambda item: item[1])
    result = tuple(item[2].copy() for item in winners[:max_ties])
    _maximum_cut_cache_put(cache_key, result)
    return result


def _locally_improve_cut_reference(
    weights: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    """Pre-optimization local search retained as a self-test oracle."""

    side = initial.copy()
    order = sorted(
        range(1, len(weights)),
        key=lambda index: (-int(np.sum(weights[index])), index),
    )
    while True:
        changed = False
        current_cross = _cut_score(weights, side)[0]
        for index in order:
            side[index] = not side[index]
            proposed_cross = _cut_score(weights, side)[0]
            if proposed_cross > current_cross:
                current_cross = proposed_cross
                changed = True
            else:
                side[index] = not side[index]
        if not changed:
            return side


def _maximum_cut_partitions_reference(
    weights: np.ndarray,
    *,
    exact_max_k: int,
    max_ties: int,
) -> tuple[np.ndarray, ...]:
    """Pre-optimization algorithm retained only as a self-test oracle."""

    matrix = np.asarray(weights, dtype=np.int64)
    candidates: list[np.ndarray] = []
    k = len(matrix)
    if k <= exact_max_k:
        for bits in range(1, 1 << (k - 1)):
            side = np.zeros(k, dtype=bool)
            for index in range(1, k):
                side[index] = bool(bits & (1 << (index - 1)))
            candidates.append(side)
    else:
        for index in range(1, k):
            side = np.zeros(k, dtype=bool)
            side[index] = True
            candidates.append(_locally_improve_cut_reference(matrix, side))
        degree_order = sorted(
            range(1, k),
            key=lambda index: (-int(np.sum(matrix[index])), index),
        )
        alternating = np.zeros(k, dtype=bool)
        alternating[degree_order[::2]] = True
        candidates.append(_locally_improve_cut_reference(
            matrix, alternating
        ))

    unique: dict[bytes, np.ndarray] = {}
    for side in candidates:
        if np.any(side):
            unique.setdefault(side.tobytes(), side)
    scored = [
        (_cut_score(matrix, side), side.tobytes(), side)
        for side in unique.values()
    ]
    best_cross = max(item[0][0] for item in scored)
    best_within = min(
        item[0][1] for item in scored if item[0][0] == best_cross
    )
    winners = [
        item
        for item in scored
        if item[0] == (best_cross, best_within)
    ]
    winners.sort(key=lambda item: item[1])
    return tuple(item[2].copy() for item in winners[:max_ties])


def _self_test_maximum_cut_optimizations() -> Mapping[str, int]:
    """Check exact output/order against the retained implementation."""

    comparison_count = 0

    def compare(matrix: np.ndarray, exact_max_k: int, max_ties: int) -> None:
        nonlocal comparison_count
        expected = _maximum_cut_partitions_reference(
            matrix,
            exact_max_k=exact_max_k,
            max_ties=max_ties,
        )
        observed = maximum_cut_partitions(
            matrix,
            exact_max_k=exact_max_k,
            max_ties=max_ties,
        )
        if tuple(side.tobytes() for side in observed) != tuple(
            side.tobytes() for side in expected
        ):
            raise AssertionError(
                f"maximum-cut mismatch at K={len(matrix)}"
            )
        comparison_count += 1

    # Exhaust every unweighted simple graph through K=5.  This includes many
    # exact ties and therefore checks deterministic byte-order truncation.
    for k in range(2, 6):
        edges = [
            (first, second)
            for first in range(k - 1)
            for second in range(first + 1, k)
        ]
        for graph_bits in range(1 << len(edges)):
            matrix = np.zeros((k, k), dtype=np.int64)
            for bit_index, (first, second) in enumerate(edges):
                value = (graph_bits >> bit_index) & 1
                matrix[first, second] = value
                matrix[second, first] = value
            compare(matrix, 12, 16)

    # Exercise exact and heuristic branches, tie caps, and integer weights at
    # every supported K through 16 without introducing stochastic test state.
    generator = np.random.default_rng(20260801)
    for k in range(2, 17):
        for _ in range(8):
            upper = np.triu(
                generator.integers(
                    0, 20, size=(k, k), dtype=np.int64
                ),
                1,
            )
            matrix = upper + upper.T
            for exact_max_k in (2, 4, 8, 12):
                for max_ties in (1, 4):
                    compare(matrix, exact_max_k, max_ties)

        assignments = generator.integers(
            0, k + 1, size=(256, 2), dtype=np.int64
        )
        expected_graph = np.zeros((k, k), dtype=np.int64)
        for first, second in assignments:
            if first < k and second < k and first != second:
                expected_graph[first, second] += 1
                expected_graph[second, first] += 1
        observed_graph = _assignment_graph_kernel(assignments, k)
        np.testing.assert_array_equal(observed_graph, expected_graph)

    # A cache hit must return an independent mutable copy, not its stored
    # result or another caller's array.
    matrix = np.ones((8, 8), dtype=np.int64)
    np.fill_diagonal(matrix, 0)
    first = maximum_cut_partitions(matrix, max_ties=4)
    encoded = tuple(side.tobytes() for side in first)
    first[0][:] = False
    second = maximum_cut_partitions(matrix, max_ties=4)
    if tuple(side.tobytes() for side in second) != encoded:
        raise AssertionError("maximum-cut cache leaked caller mutation")
    if np.shares_memory(first[0], second[0]):
        raise AssertionError("maximum-cut cache returned aliased arrays")

    _MAX_CUT_CACHE.clear()
    return {
        "reference_comparisons": comparison_count,
        "exhaustive_graphs_through_k": 5,
        "random_weighted_max_k": 16,
    }


def _gauge_site_mask(haplotypes: np.ndarray, side: np.ndarray) -> np.ndarray:
    first_side = np.flatnonzero(~side)
    second_side = np.flatnonzero(side)
    if len(first_side) == 0 or len(second_side) == 0:
        return np.zeros(haplotypes.shape[1], dtype=bool)
    first_anchor = haplotypes[first_side[0]]
    second_anchor = haplotypes[second_side[0]]
    first_constant = np.all(
        haplotypes[first_side] == first_anchor[None, :], axis=0
    )
    second_constant = np.all(
        haplotypes[second_side] == second_anchor[None, :], axis=0
    )
    return first_constant & second_constant & (first_anchor != second_anchor)


def _same_side_conditional_nll(
    haplotypes: np.ndarray,
    assignments: np.ndarray,
    evidence: np.ndarray,
    side: np.ndarray,
    site_mask: np.ndarray,
) -> float:
    k = len(haplotypes)
    first = assignments[:, 0]
    second = assignments[:, 1]
    usable = (first < k) & (second < k)
    usable &= side[first.clip(max=k - 1)] == side[second.clip(max=k - 1)]
    sample_indices = np.flatnonzero(usable)
    if len(sample_indices) == 0:
        return 0.0
    total = 0.0
    for site in np.flatnonzero(site_mask):
        dosage = (
            haplotypes[first[sample_indices], site]
            + haplotypes[second[sample_indices], site]
        )
        values = evidence[sample_indices, site, dosage]
        total -= float(np.sum(np.log(np.maximum(values, _TINY))))
    return total


def _propose_bipartite_gauge_starts(
    mode: FactorizationMode,
    evidence: np.ndarray,
    *,
    exact_cut_max_k: int,
    max_cut_ties: int,
    assignment_weights: np.ndarray | None = None,
    cut_partitions: Sequence[np.ndarray] | None = None,
) -> tuple[GaugeRewireProposal, ...]:
    weights = (
        assignment_graph(mode)
        if assignment_weights is None
        else np.asarray(assignment_weights, dtype=np.int64)
    )
    if weights.shape != (mode.k, mode.k):
        raise ValueError("assignment_weights must have shape (K, K)")
    partitions = (
        maximum_cut_partitions(
            weights,
            exact_max_k=exact_cut_max_k,
            max_ties=max_cut_ties,
        )
        if cut_partitions is None
        else tuple(cut_partitions)[:max_cut_ties]
    )
    proposals: list[GaugeRewireProposal] = []
    original_key = mode.canonical_key
    seen: set[bytes] = {original_key}
    for raw_side in partitions:
        side = np.asarray(raw_side, dtype=bool)
        if side.shape != (mode.k,):
            raise ValueError("cut partitions must each have length K")
        gauge_sites = _gauge_site_mask(mode.haplotypes, side)
        if not np.any(gauge_sites):
            continue
        cross_weight, within_weight = _cut_score(weights, side)
        before = _same_side_conditional_nll(
            mode.haplotypes,
            mode.assignments,
            evidence,
            side,
            gauge_sites,
        )

        sitewise = mode.haplotypes.copy()
        first = mode.assignments[:, 0]
        second = mode.assignments[:, 1]
        real = (first < mode.k) & (second < mode.k)
        same_side = real.copy()
        real_indices = np.flatnonzero(real)
        same_side[real_indices] = (
            side[first[real_indices]] == side[second[real_indices]]
        )
        resolving_samples = np.flatnonzero(same_side)
        for site in np.flatnonzero(gauge_sites):
            if len(resolving_samples) == 0:
                continue
            dosage = (
                mode.haplotypes[first[resolving_samples], site]
                + mode.haplotypes[second[resolving_samples], site]
            )
            current_nll = -float(np.sum(np.log(np.maximum(
                evidence[resolving_samples, site, dosage], _TINY
            ))))
            opposite_nll = -float(np.sum(np.log(np.maximum(
                evidence[resolving_samples, site, 2 - dosage], _TINY
            ))))
            if opposite_nll < current_nll - _NLL_TOLERANCE:
                sitewise[:, site] = 1 - sitewise[:, site]
        sitewise_key = _canonical_haplotype_key(sitewise)
        if sitewise_key not in seen:
            seen.add(sitewise_key)
            changed = np.any(sitewise != mode.haplotypes, axis=0)
            proposals.append(GaugeRewireProposal(
                haplotypes=sitewise,
                partition=side,
                gauge_sites=gauge_sites,
                proposal_kind="conditional_sitewise",
                n_flipped_sites=int(np.sum(changed)),
                cross_assignment_weight=cross_weight,
                within_assignment_weight=within_weight,
                conditional_nll_before=before,
                conditional_nll_after=_same_side_conditional_nll(
                    sitewise,
                    mode.assignments,
                    evidence,
                    side,
                    gauge_sites,
                ),
            ))

        opposite = mode.haplotypes.copy()
        opposite[:, gauge_sites] = 1 - opposite[:, gauge_sites]
        opposite_key = _canonical_haplotype_key(opposite)
        if opposite_key not in seen:
            seen.add(opposite_key)
            proposals.append(GaugeRewireProposal(
                haplotypes=opposite,
                partition=side,
                gauge_sites=gauge_sites,
                proposal_kind="opposite_endpoint",
                n_flipped_sites=int(np.sum(gauge_sites)),
                cross_assignment_weight=cross_weight,
                within_assignment_weight=within_weight,
                conditional_nll_before=before,
                conditional_nll_after=_same_side_conditional_nll(
                    opposite,
                    mode.assignments,
                    evidence,
                    side,
                    gauge_sites,
                ),
            ))
    return tuple(proposals)


def propose_bipartite_gauge_starts(
    mode: FactorizationMode,
    evidence: np.ndarray,
    *,
    exact_cut_max_k: int = DEFAULT_EXACT_CUT_MAX_K,
    max_cut_ties: int = DEFAULT_MAX_CUT_TIES,
) -> tuple[GaugeRewireProposal, ...]:
    """Propose coherent joint-column starts using training evidence only.

    The maximum-cut partition identifies the dominant bipartite assignment
    backbone.  A site is gauge-compatible when every haplotype on either side
    has one common allele and the two sides are opposite.  Complementing all
    K alleles at such a site preserves dosage one for every cross-cut pair.

    The conditional proposal orients each compatible column using only
    same-side real-real assignments.  A second proposal complements the whole
    compatible endpoint; after refitting this can expose a same-side parent
    hidden by the current assignment state.  No held-out evidence is used.
    """

    checked = _validate_evidence(
        evidence,
        n_sites=mode.n_sites,
        n_samples=len(mode.assignments),
    )
    return _propose_bipartite_gauge_starts(
        mode,
        checked,
        exact_cut_max_k=exact_cut_max_k,
        max_cut_ties=max_cut_ties,
    )


def _canonical_haplotype_key(haplotypes: np.ndarray) -> bytes:
    matrix = np.asarray(haplotypes, dtype=np.int8)
    order = sorted(
        range(len(matrix)),
        key=lambda index: (matrix[index].tobytes(), index),
    )
    return np.ascontiguousarray(matrix[order]).tobytes()


def _refit_gauge_modes(
    mode: FactorizationMode,
    evidence: np.ndarray,
    lambda_wildcard_penalty: float,
    max_iter_per_k: int,
    exact_cut_max_k: int,
    max_cut_ties: int,
    *,
    fit_workspace: Any | None = None,
) -> tuple[FactorizationMode, ...]:
    from bhd_fit import _fit_at_fixed_K_many

    proposals = _propose_bipartite_gauge_starts(
        mode,
        evidence,
        exact_cut_max_k=exact_cut_max_k,
        max_cut_ties=max_cut_ties,
    )
    fits = _fit_at_fixed_K_many(
        evidence,
        (proposal.haplotypes for proposal in proposals),
        lambda_wildcard_penalty,
        max_iter=max_iter_per_k,
        workspace=fit_workspace,
    )
    return tuple(
        _canonicalize_fit(fit, fit_workspace=fit_workspace)
        for fit in fits
    )


def _refit_gauge_modes_many(
    modes: Sequence[FactorizationMode],
    evidence: np.ndarray,
    lambda_wildcard_penalty: float,
    max_iter_per_k: int,
    exact_cut_max_k: int,
    max_cut_ties: int,
    *,
    fit_workspace: Any | None = None,
) -> tuple[FactorizationMode, ...]:
    """Refit proposals from same-K modes in one deterministic fit batch.

    Proposal generation and result consumption retain mode order and each
    mode's proposal order.  Only the scheduling of independent fixed-K fits is
    changed.
    """

    from bhd_fit import _fit_at_fixed_K_many

    starts = []
    for mode in modes:
        proposals = _propose_bipartite_gauge_starts(
            mode,
            evidence,
            exact_cut_max_k=exact_cut_max_k,
            max_cut_ties=max_cut_ties,
        )
        starts.extend(proposal.haplotypes for proposal in proposals)
    if not starts:
        return ()
    fits = _fit_at_fixed_K_many(
        evidence,
        starts,
        lambda_wildcard_penalty,
        max_iter=max_iter_per_k,
        workspace=fit_workspace,
    )
    return tuple(
        _canonicalize_fit(fit, fit_workspace=fit_workspace)
        for fit in fits
    )


def refit_bipartite_gauge_modes(
    mode: FactorizationMode,
    evidence: np.ndarray,
    *,
    lambda_wildcard_penalty: float = DEFAULT_LAMBDA,
    max_iter_per_k: int = 50,
    exact_cut_max_k: int = DEFAULT_EXACT_CUT_MAX_K,
    max_cut_ties: int = DEFAULT_MAX_CUT_TIES,
) -> tuple[FactorizationMode, ...]:
    """Refit all novel gauge starts and return coherent fixed-K modes."""

    checked = _validate_evidence(
        evidence,
        n_sites=mode.n_sites,
        n_samples=len(mode.assignments),
    )
    return _refit_gauge_modes(
        mode,
        checked,
        lambda_wildcard_penalty,
        max_iter_per_k,
        exact_cut_max_k,
        max_cut_ties,
    )


def _initial_complete_modes(
    evidence: np.ndarray,
    beam_width: int,
    n_seed_modes: int,
    soft_seed_min_cluster_size: int,
    lambda_wildcard_penalty: float,
    max_iter_per_k: int,
    *,
    fit_workspace: Any | None = None,
    seed_sample_mask: np.ndarray | None = None,
) -> tuple[FactorizationMode, ...]:
    from bhd_fit import _fit_at_fixed_K_many
    from bhd_kernels import _init_hap_from_sample_dosage, _select_initial_seed
    from bhd_soft_seeding import soft_cluster_seed_haplotypes

    seed_evidence = evidence
    if seed_sample_mask is not None:
        mask = np.asarray(seed_sample_mask, dtype=np.bool_)
        if mask.shape != (len(evidence),) or not np.any(mask):
            raise ValueError(
                "seed_sample_mask must retain an evidence sample"
            )
        if not np.all(mask):
            seed_evidence = np.ascontiguousarray(evidence[mask])

    seeds = soft_cluster_seed_haplotypes(
        seed_evidence,
        n_seed_modes,
        min_cluster_size=soft_seed_min_cluster_size,
    )
    if not seeds:
        sample = _select_initial_seed(seed_evidence, kept_mask=None)
        seeds = [
            _init_hap_from_sample_dosage(
                seed_evidence, sample, kept_mask=None
            )
        ]
    starts = [
        np.asarray(seed, dtype=np.int64)[None, :]
        for seed in seeds
    ]
    fits = _fit_at_fixed_K_many(
        evidence,
        starts,
        lambda_wildcard_penalty,
        max_iter=max_iter_per_k,
        workspace=fit_workspace,
    )
    fitted = [
        _canonicalize_fit(fit, fit_workspace=fit_workspace)
        for fit in fits
    ]
    return _deduplicate_modes(fitted, beam_width)


def _expand_one_complete_mode(
    mode: FactorizationMode,
    evidence: np.ndarray,
    lambda_wildcard_penalty: float,
    max_iter_per_k: int,
    *,
    fit_workspace: Any | None = None,
    oracle_nll: np.ndarray | None = None,
    decisiveness: np.ndarray | None = None,
    dosage_by_sample: np.ndarray | None = None,
    seed_haplotypes_by_sample: np.ndarray | None = None,
    active_sample_mask: np.ndarray | None = None,
) -> tuple[tuple[FactorizationMode, ...], int, int]:
    """Generate data-derived K+1 children without rewarding diffuse reads.

    Raw per-sample NLL is not a residual-misfit score: a low-depth sample
    with nearly uniform genotype likelihoods has a large irreducible NLL and
    can outrank an informative, genuinely misfit sample. Growth therefore
    ranks samples by excess NLL above their sitewise unconstrained genotype
    oracle. Decisiveness and sample index provide deterministic tie-breaks.

    A bounded search tries novel subtraction/dosage seeds from that ranking.
    Independent candidates are fitted as one ordered batch, then consumed in
    the historical prefix order.  Thus stopping, retained modes, diagnostics,
    and mode ordering are unchanged even though work may execute concurrently.
    If every optimized child collapses two rows, the highest-priority
    *training-derived* distinct initialization is synchronized with
    ``max_iter=0`` and retained as a transparent fallback. This represents
    the requested K without manufacturing evidence: assignments and NLL are
    recomputed, but the unsupported row is not moved into a duplicate. The
    posterior occupancy model and held-out prediction can then penalize it.

    Returns ``(children, seed_attempts, synchronized_fallback_count)``.
    """

    import dynamic_threads
    from bhd_fit import _fit_at_fixed_K, _fit_at_fixed_K_many
    from bhd_kernels import _init_hap_from_sample_dosage

    dynamic_threads.apply_dynamic_threads(
        max_threads=FIXED_K_FIT_MAX_THREADS
    )
    # Match the existing K-growth transition: refit the parent before using
    # its assignments and per-sample costs to construct K+1 proposals.  A mode
    # may bypass this only when this exact evidence workspace certified that
    # its haplotypes are already an A(H), H(A) fixed point.
    parent_is_certified = bool(
        mode.fixed_point_certified
        and fit_workspace is not None
        and fit_workspace.certifies_fixed_point(mode.haplotypes)
    )
    if parent_is_certified:
        parent = mode
    else:
        parent = _canonicalize_fit(
            _fit_at_fixed_K(
                evidence,
                mode.haplotypes,
                lambda_wildcard_penalty,
                max_iter=max_iter_per_k,
                workspace=fit_workspace,
            ),
            fit_workspace=fit_workspace,
        )
    if oracle_nll is None:
        oracle_nll = -np.sum(
            np.log(np.maximum(np.max(evidence, axis=2), _TINY)), axis=1
        )
    excess_nll = np.maximum(parent.per_sample_cost - oracle_nll, 0.0)
    if decisiveness is None:
        decisiveness = np.sum(np.max(evidence, axis=2), axis=1)
    if dosage_by_sample is None:
        dosage_by_sample = np.argmax(evidence, axis=2)
    if seed_haplotypes_by_sample is None:
        seed_haplotypes_by_sample = np.stack([
            _init_hap_from_sample_dosage(
                evidence, sample, kept_mask=None
            )
            for sample in range(len(evidence))
        ])
    if active_sample_mask is None:
        proposal_samples = np.arange(len(evidence), dtype=np.int64)
    else:
        active = np.asarray(active_sample_mask, dtype=np.bool_)
        if active.shape != (len(evidence),):
            raise ValueError(
                "active_sample_mask must match the evidence sample axis"
            )
        proposal_samples = np.flatnonzero(active)
        if len(proposal_samples) == 0:
            raise ValueError("active_sample_mask must retain an evidence sample")
    sample_order = sorted(
        proposal_samples.tolist(),
        key=lambda sample: (
            -float(excess_nll[sample]),
            -float(decisiveness[sample]),
            int(sample),
        ),
    )

    # The historical transition fitted at most K+1 seeds from one sample.
    # Permit a second sample-worth only when seeds collapse.
    target_children = parent.k + 1
    maximum_attempts = max(16, 2 * target_children)
    existing_keys = {
        np.ascontiguousarray(row, dtype=np.int8).tobytes()
        for row in parent.haplotypes
    }
    attempted_seed_keys: set[bytes] = set()
    candidate_initials: list[np.ndarray] = []

    # Candidate construction is independent of fit outcomes.  Materialize the
    # same deterministic seed prefix up to the historical attempt cap, fit it
    # as one batch, and below consume only through the historical stopping
    # point.
    for sample in sample_order:
        dosage = dosage_by_sample[sample]
        seed_haplotypes = [
            np.clip(
                dosage - parent.haplotypes[index], 0, 1
            ).astype(np.int64)
            for index in range(parent.k)
        ]
        seed_haplotypes.append(seed_haplotypes_by_sample[sample])
        for seed in seed_haplotypes:
            contiguous = np.ascontiguousarray(seed, dtype=np.int64)
            seed_key = contiguous.astype(np.int8, copy=False).tobytes()
            if seed_key in existing_keys or seed_key in attempted_seed_keys:
                continue
            attempted_seed_keys.add(seed_key)
            candidate_initials.append(np.vstack(
                [parent.haplotypes, contiguous[None, :]]
            ))
            if len(candidate_initials) >= maximum_attempts:
                break
        if len(candidate_initials) >= maximum_attempts:
            break

    children: list[FactorizationMode] = []
    child_keys: set[bytes] = set()
    seed_attempts = 0
    # A target of T distinct children cannot be reached in fewer than T fit
    # results.  T-sized waves therefore expose useful fit-level parallelism
    # without doing avoidable work before the earliest possible stop.  Results
    # inside and across waves are still consumed in the exact historical order.
    for wave_start in range(0, len(candidate_initials), target_children):
        wave = candidate_initials[
            wave_start:wave_start + target_children
        ]
        fits = _fit_at_fixed_K_many(
            evidence,
            wave,
            lambda_wildcard_penalty,
            max_iter=max_iter_per_k,
            workspace=fit_workspace,
        )
        stop = False
        for fit in fits:
            fitted = _canonicalize_fit(
                fit, fit_workspace=fit_workspace
            )
            seed_attempts += 1
            if _has_distinct_rows(fitted):
                children.append(fitted)
                child_keys.add(fitted.canonical_key)
            if (
                len(child_keys) >= target_children
                or seed_attempts >= maximum_attempts
            ):
                stop = True
                break
        if stop:
            break

    if children:
        return tuple(children), seed_attempts, 0

    fallback_fits = _fit_at_fixed_K_many(
        evidence,
        candidate_initials[:seed_attempts],
        lambda_wildcard_penalty,
        max_iter=0,
        workspace=fit_workspace,
    )
    synchronized_fallbacks = [
        _canonicalize_fit(fit, fit_workspace=fit_workspace)
        for fit in fallback_fits
    ]
    synchronized_fallbacks = [
        fitted
        for fitted in synchronized_fallbacks
        if _has_distinct_rows(fitted)
    ]
    fallback_modes = _deduplicate_modes(
        synchronized_fallbacks, target_children
    )
    if fallback_modes:
        return fallback_modes, seed_attempts, len(fallback_modes)
    return (), seed_attempts, 0


def enumerate_complete_modes(
    training_evidence: np.ndarray,
    *,
    max_k: int,
    beam_width: int,
    lambda_wildcard_penalty: float = DEFAULT_LAMBDA,
    n_seed_modes: int = DEFAULT_DATA_SEED_MODES,
    soft_seed_min_cluster_size: int = DEFAULT_SOFT_SEED_MIN_CLUSTER_SIZE,
    max_iter_per_k: int = 50,
    apply_gauge_rewire: bool = True,
    exact_cut_max_k: int = DEFAULT_EXACT_CUT_MAX_K,
    max_cut_ties: int = DEFAULT_MAX_CUT_TIES,
    _fit_workspace: Any | None = None,
) -> CompleteModeBeam:
    """Enumerate a bounded beam of coherent modes for every K up to a cap.

    ``max_k`` is not an expected or target founder count.  It is a safety cap
    on the set of K values that a later held-out comparison may consider.
    Every retained object is a complete K-row model with synchronized
    assignments.  Gauge-refitted modes are fed into the next K expansion;
    haplotype rows from separate modes are never unioned.
    """

    evidence = _validate_evidence(training_evidence)
    if max_k < 1 or max_k > 2 * len(evidence):
        raise ValueError("max_k must lie in [1, 2 * n_samples]")
    if beam_width < 1 or n_seed_modes < 1:
        raise ValueError("beam_width and n_seed_modes must be positive")
    if soft_seed_min_cluster_size < 2 or max_iter_per_k < 1:
        raise ValueError("invalid seed-cluster or iteration limit")
    if not math.isfinite(lambda_wildcard_penalty):
        raise ValueError("lambda_wildcard_penalty must be finite")

    from bhd_fit import _prepare_fixed_k_fit_workspace

    fit_workspace = _fit_workspace
    if fit_workspace is None:
        fit_workspace = _prepare_fixed_k_fit_workspace(
            evidence, lambda_wildcard_penalty
        )
    max_genotype_probability = np.max(evidence, axis=2)
    oracle_nll = -np.sum(
        np.log(np.maximum(max_genotype_probability, _TINY)), axis=1
    )
    decisiveness = np.sum(max_genotype_probability, axis=1)
    dosage_by_sample = np.argmax(evidence, axis=2)
    population_alt_frequency = (
        evidence[..., 1].mean(axis=0) * 0.5
        + evidence[..., 2].mean(axis=0)
    )
    seed_haplotypes_by_sample = np.zeros_like(dosage_by_sample)
    seed_haplotypes_by_sample[dosage_by_sample == 2] = 1
    heterozygous = dosage_by_sample == 1
    seed_haplotypes_by_sample[
        heterozygous & (population_alt_frequency[None, :] > 0.5)
    ] = 1

    modes = _initial_complete_modes(
        evidence,
        beam_width,
        n_seed_modes,
        soft_seed_min_cluster_size,
        lambda_wildcard_penalty,
        max_iter_per_k,
        fit_workspace=fit_workspace,
    )
    if not modes:
        raise RuntimeError("could not initialize a distinct K=1 mode")
    paths: list[tuple[FactorizationMode, ...]] = [modes]
    raw_counts = [len(modes)]
    gauge_counts = [0]
    seed_attempt_counts = [0]
    synchronized_fallback_counts = [0]
    for _target_k in range(2, max_k + 1):
        raw_children: list[FactorizationMode] = []
        target_seed_attempts = 0
        target_synchronized_fallbacks = 0
        for mode in modes:
            children, seed_attempts, fallback_count = (
                _expand_one_complete_mode(
                    mode,
                    evidence,
                    lambda_wildcard_penalty,
                    max_iter_per_k,
                    fit_workspace=fit_workspace,
                    oracle_nll=oracle_nll,
                    decisiveness=decisiveness,
                    dosage_by_sample=dosage_by_sample,
                    seed_haplotypes_by_sample=seed_haplotypes_by_sample,
                )
            )
            raw_children.extend(children)
            target_seed_attempts += seed_attempts
            target_synchronized_fallbacks += fallback_count
        raw_counts.append(len(raw_children))
        seed_attempt_counts.append(target_seed_attempts)
        synchronized_fallback_counts.append(
            target_synchronized_fallbacks
        )
        # Gauge-rewire every distinct raw child before applying the beam cap.
        # A poor pre-gauge basin can be exactly the mode whose coherent joint
        # move reveals the best fixed-K factorisation.
        raw_modes = _deduplicate_modes(raw_children)
        if not raw_modes:
            break

        gauge_children: tuple[FactorizationMode, ...] = ()
        if apply_gauge_rewire:
            gauge_children = _refit_gauge_modes_many(
                raw_modes,
                evidence,
                lambda_wildcard_penalty,
                max_iter_per_k,
                exact_cut_max_k,
                max_cut_ties,
                fit_workspace=fit_workspace,
            )
        gauge_counts.append(len(gauge_children))
        modes = _deduplicate_modes(
            (*raw_modes, *gauge_children),
            beam_width,
        )
        if not modes:
            break
        paths.append(modes)

    if len(paths) != max_k:
        raise RuntimeError(
            "complete fixed-K mode enumeration stopped before the declared "
            f"cap: represented=1..{len(paths)}, requested=1..{max_k}"
        )

    # A failed transition has one diagnostic entry beyond the represented
    # paths.  Trim it so every tuple is indexed by K-1.
    raw_counts = raw_counts[:len(paths)]
    gauge_counts = gauge_counts[:len(paths)]
    seed_attempt_counts = seed_attempt_counts[:len(paths)]
    synchronized_fallback_counts = synchronized_fallback_counts[:len(paths)]
    return CompleteModeBeam(
        modes_by_k=tuple(paths),
        beam_width=beam_width,
        max_k_cap=max_k,
        raw_trial_counts=tuple(raw_counts),
        gauge_trial_counts=tuple(gauge_counts),
        growth_seed_attempt_counts=tuple(seed_attempt_counts),
        synchronized_fallback_counts=tuple(
            synchronized_fallback_counts
        ),
    )


def self_test() -> Mapping[str, Any]:
    """Run bounded algebraic and exact max-cut regression tests."""

    cut_validation = _self_test_maximum_cut_optimizations()

    # The first three columns are bipartite gauge columns.  The final column
    # is not: rows within each side differ there.
    haplotypes = np.asarray(
        [
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
        ],
        dtype=np.int64,
    )
    cross_pairs = np.asarray(
        [(0, 2), (0, 3), (1, 2), (1, 3)] * 3,
        dtype=np.int64,
    )
    within_pairs = np.asarray([(0, 1), (2, 3)], dtype=np.int64)
    assignments = np.vstack([cross_pairs, within_pairs])
    evidence = np.full((len(assignments), 4, 3), 1.0 / 3.0)
    # At site zero, both observed within-side pairs favour the opposite gauge.
    evidence[-2, 0] = (0.001, 0.001, 0.998)
    evidence[-1, 0] = (0.998, 0.001, 0.001)
    # At site one they favour the current orientation.  Site two is a tie.
    evidence[-2, 1] = (0.001, 0.001, 0.998)
    evidence[-1, 1] = (0.998, 0.001, 0.001)
    evidence /= np.sum(evidence, axis=2, keepdims=True)
    mode = FactorizationMode(
        haplotypes=haplotypes,
        assignments=assignments,
        per_sample_cost=np.zeros(len(assignments)),
        wildcard_slots=np.zeros(len(assignments), dtype=np.int64),
        n_iter=0,
        total_nll=1.0,
    )
    weights = assignment_graph(mode)
    partitions = maximum_cut_partitions(weights, exact_max_k=4, max_ties=4)
    if len(partitions) != 1:
        raise AssertionError("the synthetic assignment graph needs one max cut")
    side = partitions[0]
    if not np.array_equal(side, np.asarray([False, False, True, True])):
        raise AssertionError("maximum cut did not recover the bipartition")
    gauge_sites = _gauge_site_mask(mode.haplotypes, side)
    np.testing.assert_array_equal(
        gauge_sites, np.asarray([True, True, True, False])
    )

    proposals = propose_bipartite_gauge_starts(
        mode, evidence, exact_cut_max_k=4, max_cut_ties=4
    )
    sitewise = next(
        proposal
        for proposal in proposals
        if proposal.proposal_kind == "conditional_sitewise"
    )
    if sitewise.n_flipped_sites != 1:
        raise AssertionError("conditional rewire should flip exactly one site")
    changed = np.any(sitewise.haplotypes != mode.haplotypes, axis=0)
    np.testing.assert_array_equal(
        changed, np.asarray([True, False, False, False])
    )
    if not sitewise.conditional_nll_after < sitewise.conditional_nll_before:
        raise AssertionError("conditional gauge rewire did not improve NLL")

    cross_samples = np.arange(len(cross_pairs))
    before = (
        mode.haplotypes[assignments[cross_samples, 0]]
        + mode.haplotypes[assignments[cross_samples, 1]]
    )
    after = (
        sitewise.haplotypes[assignments[cross_samples, 0]]
        + sitewise.haplotypes[assignments[cross_samples, 1]]
    )
    np.testing.assert_array_equal(before[:, gauge_sites], 1)
    np.testing.assert_array_equal(after[:, gauge_sites], before[:, gauge_sites])
    return {
        "status": "ok",
        "k": mode.k,
        "n_samples": len(assignments),
        "n_sites": mode.n_sites,
        "cross_weight": int(_cut_score(weights, side)[0]),
        "within_weight": int(_cut_score(weights, side)[1]),
        "gauge_sites": int(np.sum(gauge_sites)),
        "conditionally_flipped_sites": sitewise.n_flipped_sites,
        "cross_dosage_invariant": True,
        "maximum_cut_validation": cut_validation,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run bounded gauge and exact max-cut regression tests",
    )
    arguments = parser.parse_args()
    if not arguments.self_test:
        parser.error("no production CLI; pass --self-test")
    print(json.dumps(self_test(), indent=2, sort_keys=True))
