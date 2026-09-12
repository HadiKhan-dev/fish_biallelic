"""discovery / modes for the canonical reconstruction pipeline."""
from __future__ import annotations


from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Mapping, Sequence
import numpy as np
from numba import njit


DEFAULT_EXACT_CUT_MAX_K = 12


DEFAULT_MAX_CUT_TIES = 4


_MAX_CUT_CACHE_SIZE = 256


_TINY = np.finfo(np.float64).tiny


_NLL_TOLERANCE = 1e-10


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


def _validate_evidence(
    evidence: np.ndarray,
    *,
    n_sites: int | None = None,
    n_samples: int | None = None,
) -> np.ndarray:
    return core_genotypes.validate_normalized_genotype_evidence(
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
        ) = discovery_objectives.canonicalize_binary_panel(haplotypes, assignments)
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
        and fit_workspace.certifies_fixed_point(canonical_haplotypes)
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

    return discovery_objectives.exact_unique_binary_rows(matrix)


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
                fit_records = discovery_fitting._fit_at_fixed_K_many_with_initial(
                    genotype_likelihoods,
                    group_starts,
                    config.lambda_wildcard_penalty,
                    max_iter=iterations,
                    workspace=workspace,
                )
            else:
                final_fits = discovery_fitting._fit_at_fixed_K_many(
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
def _component_exact_cuts(weights, max_ties):
    """Factor independent graph components; solve bipartite pieces by coloring.

    Integer codes use vertex order as most-to-least significant bits, so
    retaining the smallest codes preserves the public byte-lexicographic tie
    rule. Non-bipartite components retain exhaustive weighted maximum cuts.
    """
    k=len(weights)
    component=np.full(k,-1,dtype=np.int64)
    color=np.zeros(k,dtype=np.bool_)
    vertices=np.empty((k,k),dtype=np.int64)
    sizes=np.zeros(k,dtype=np.int64)
    bipartite=np.ones(k,dtype=np.bool_)
    count=0
    for root in range(k):
        if component[root]>=0:continue
        component[root]=count
        vertices[count,0]=root;sizes[count]=1
        head=0
        while head<sizes[count]:
            vertex=vertices[count,head];head+=1
            for other in range(k):
                if weights[vertex,other]==0:continue
                if component[other]<0:
                    component[other]=count;color[other]=not color[vertex]
                    vertices[count,sizes[count]]=other;sizes[count]+=1
                elif color[other]==color[vertex]:
                    bipartite[count]=False
        count+=1
    # One extra permits omission of the all-False empty cut at the end.
    limit=max_ties+1
    combined=np.zeros(1,dtype=np.int64)
    for c in range(count):
        members=np.sort(vertices[c,:sizes[c]])
        mask=np.int64(0)
        for vertex in members:mask|=np.int64(1)<<(k-1-vertex)
        if bipartite[c]:
            code=np.int64(0)
            for vertex in members:
                if color[vertex]:code|=np.int64(1)<<(k-1-vertex)
            choices=np.empty(1 if c==0 else 2,dtype=np.int64)
            choices[0]=code
            if c>0:choices[1]=code^mask
        else:
            m=len(members)
            local=np.empty((m,m),dtype=np.int64)
            for i in range(m):
                for j in range(m):local[i,j]=weights[members[i],members[j]]
            cross,_=_exact_cut_score_table(local)
            best=np.max(cross)
            winners=np.flatnonzero(cross==best)
            choices=np.empty(len(winners)*(1 if c==0 else 2),dtype=np.int64)
            cursor=0
            for index in winners:
                bits=index+1;code=np.int64(0)
                for j in range(1,m):
                    if bits & (1<<(j-1)):code|=np.int64(1)<<(k-1-members[j])
                choices[cursor]=code;cursor+=1
                if c>0:choices[cursor]=code^mask;cursor+=1
        choices=np.sort(choices)[:limit]
        joined=np.empty(len(combined)*len(choices),dtype=np.int64)
        cursor=0
        for previous in combined:
            for choice in choices:joined[cursor]=previous|choice;cursor+=1
        combined=np.sort(joined)[:limit]
    combined=combined[combined!=0][:max_ties]
    result=np.zeros((len(combined),k),dtype=np.bool_)
    for row in range(len(combined)):
        for vertex in range(k):
            result[row,vertex]=bool(combined[row] & (np.int64(1)<<(k-1-vertex)))
    return result


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
    side ``False``.  Exact component-wise inference is used through ``exact_max_k``;
    bipartite components need no enumeration.
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
        result = tuple(side.copy() for side in _component_exact_cuts(matrix, max_ties))
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


def _canonical_haplotype_key(haplotypes: np.ndarray) -> bytes:
    matrix = np.asarray(haplotypes, dtype=np.int8)
    order = sorted(
        range(len(matrix)),
        key=lambda index: (matrix[index].tobytes(), index),
    )
    return np.ascontiguousarray(matrix[order]).tobytes()


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


    seed_evidence = evidence
    if seed_sample_mask is not None:
        mask = np.asarray(seed_sample_mask, dtype=np.bool_)
        if mask.shape != (len(evidence),) or not np.any(mask):
            raise ValueError(
                "seed_sample_mask must retain an evidence sample"
            )
        if not np.all(mask):
            seed_evidence = np.ascontiguousarray(evidence[mask])

    seeds = discovery_objectives.soft_cluster_seed_haplotypes(
        seed_evidence,
        n_seed_modes,
        min_cluster_size=soft_seed_min_cluster_size,
    )
    if not seeds:
        sample = discovery_objectives._select_initial_seed(seed_evidence, kept_mask=None)
        seeds = [
            discovery_objectives._init_hap_from_sample_dosage(
                seed_evidence, sample, kept_mask=None
            )
        ]
    starts = [
        np.asarray(seed, dtype=np.int64)[None, :]
        for seed in seeds
    ]
    fits = discovery_fitting._fit_at_fixed_K_many(
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


    core_parallel.apply_dynamic_threads(
        max_threads=core_config.FIXED_K_FIT_MAX_THREADS
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
            discovery_fitting._fit_at_fixed_K(
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
            discovery_objectives._init_hap_from_sample_dosage(
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
        fits = discovery_fitting._fit_at_fixed_K_many(
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

    fallback_fits = discovery_fitting._fit_at_fixed_K_many(
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

import haplotype_reconstruction.core.config as core_config
import haplotype_reconstruction.core.genotypes as core_genotypes
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.discovery.fitting as discovery_fitting
import haplotype_reconstruction.discovery.objectives as discovery_objectives
