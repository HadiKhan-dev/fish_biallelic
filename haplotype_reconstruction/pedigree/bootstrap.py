"""pedigree / bootstrap for the canonical reconstruction pipeline."""
from __future__ import annotations


import numpy as np
from numba import njit

import math

import os

from typing import Any, Mapping, Optional, Sequence
import numba


@njit(cache=True, fastmath=False, nogil=True)
def _pack_positive_presence_2d(values: np.ndarray) -> np.ndarray:
    n_contigs, n_items = values.shape
    n_words = (n_contigs + 63) // 64
    packed = np.zeros((n_words, n_items), dtype=np.uint64)
    one = np.uint64(1)
    for contig in range(n_contigs):
        word = contig // 64
        bit = one << np.uint64(contig % 64)
        for item in range(n_items):
            if values[contig, item] > 0:
                packed[word, item] |= bit
    return packed


_BOOTSTRAP_SHARED: dict[str, Any] = {}


def pack_contig_presence(values_by_contig: np.ndarray) -> np.ndarray:
    """Pack ``values > 0`` along the leading contig axis into uint64 words.

    The returned shape is ``(ceil(n_contigs / 64), *item_shape)``.  Padding
    bits in the final word are always zero.  The representation therefore
    supports any positive number of contigs, including more than 64.
    """
    values = np.asarray(values_by_contig)
    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("values_by_contig must have a non-empty contig axis")
    if values.dtype.kind not in "bui f":
        raise ValueError("contig exposure values must be numeric or boolean")
    contiguous = np.ascontiguousarray(values.reshape(values.shape[0], -1))
    packed = _pack_positive_presence_2d(contiguous)
    return packed.reshape((packed.shape[0],) + values.shape[1:])


_BOOTSTRAP_SHM_REFS: list[Any] = []


@njit(cache=True, fastmath=False, nogil=True)
def _active_contig_words(weights: np.ndarray) -> np.ndarray:
    n_contigs = weights.shape[0]
    words = np.zeros((n_contigs + 63) // 64, dtype=np.uint64)
    one = np.uint64(1)
    for contig in range(n_contigs):
        if weights[contig] > 0:
            words[contig // 64] |= one << np.uint64(contig % 64)
    return words


_BOOTSTRAP_MIN_WORK_ITEMS = 100_000


def active_contig_words(contig_weights: np.ndarray) -> np.ndarray:
    """Return packed bits for contigs with strictly positive multiplicity."""
    weights = np.asarray(contig_weights)
    if weights.ndim != 1 or weights.shape[0] < 1:
        raise ValueError("contig_weights must be a non-empty one-dimensional array")
    if weights.dtype.kind not in "bui f":
        raise ValueError("contig weights must be numeric or boolean")
    return _active_contig_words(np.ascontiguousarray(weights))


def _bootstrap_worker_count(
    n_alternatives: int,
    bootstrap_replicates: int,
    cpu_budget: int,
) -> int:
    """Choose one process per usable CPU for a substantial bootstrap.

    Replicates are independent tasks, and their dominant selection work scans
    the candidate alternatives.  Gate pool startup on that combined work
    rather than on candidate count alone, then consume the complete caller-
    bounded CPU budget when enough replicates are available.
    """
    work_items = int(n_alternatives) * int(bootstrap_replicates)
    if (
        cpu_budget <= 1
        or bootstrap_replicates < 32
        or work_items < _BOOTSTRAP_MIN_WORK_ITEMS
    ):
        return 1
    return min(int(cpu_budget), int(bootstrap_replicates))


@njit(cache=True, fastmath=False, nogil=True, inline="always")
def _popcount_u64(value: np.uint64) -> int:
    count = 0
    one = np.uint64(1)
    while value != 0:
        value &= value - one
        count += 1
    return count


def _init_bootstrap_worker(shared: Mapping[str, Any]) -> None:
    """Attach read-only bootstrap arrays once in each forkserver worker."""
    global _BOOTSTRAP_SHARED, _BOOTSTRAP_SHM_REFS
    for handle in _BOOTSTRAP_SHM_REFS:
        try:
            handle.close()
        except Exception:
            pass
    _BOOTSTRAP_SHM_REFS = []
    _BOOTSTRAP_SHARED = {}
    for key, value in shared.items():
        if isinstance(value, Mapping) and "shm_name" in value:
            handle, array = core_parallel.attach_shared_array(value)
            _BOOTSTRAP_SHM_REFS.append(handle)
            _BOOTSTRAP_SHARED[key] = array
        else:
            _BOOTSTRAP_SHARED[key] = value
    _BOOTSTRAP_SHARED["scaffold_prepared"] = None


@njit(cache=True, fastmath=False, nogil=True)
def _count_active_presence_2d(
    packed_presence: np.ndarray,
    active_words: np.ndarray,
) -> np.ndarray:
    n_words, n_items = packed_presence.shape
    counts = np.zeros(n_items, dtype=np.int32)
    for item in range(n_items):
        total = 0
        for word in range(n_words):
            total += _popcount_u64(
                packed_presence[word, item] & active_words[word]
            )
        counts[item] = total
    return counts


def _evaluate_bootstrap_chunk(
    shared: Mapping[str, Any],
    multiplicities: np.ndarray,
) -> tuple:
    """Return baseline counts plus optional opt-in M1 direction-state counts."""
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
    m1_direction_counts = (
        None)
    depth_refits = 0

    def evaluate(
        weights: np.ndarray,
        depth_model: Optional[pedigree_direction._AncestryDepthModel],
    ) -> pedigree_states._ParentStateSelection:
        return pedigree_states._evaluate_parent_state_weighted_contigs(
            contig_log_likelihoods,
            weights,
            shared["contig_information_weights"],
            alternatives,
            states,
            by_child,
            full_counts,
            shared["settings"],
            n_samples,
            depth_model,
            structure_pair_indices=shared["structure_pair_indices"],
            edge_matched_by_contig=shared["edge_matched_by_contig"],
            edge_exposed_by_contig=shared["edge_exposed_by_contig"],
            pair_explained_by_contig=shared["pair_explained_by_contig"],
            pair_exposed_by_contig=shared["pair_exposed_by_contig"],
            structure_total_bins_by_contig=shared[
                "structure_total_bins_by_contig"
            ],
            edge_exposure_presence_words=shared.get(
                "edge_exposure_presence_words"
            ),
            pair_exposure_presence_words=shared.get(
                "pair_exposure_presence_words"
            ),
            direction_supported_parents=shared.get(
                "direction_supported_parents"
            ),
            scaffold_prepared=shared.get("scaffold_prepared"),
        )

    for replicate, multiplicity in enumerate(multiplicities):
        depth_model = pedigree_direction._fit_ancestry_depth_model(
            multiplicity @ junction_matrix,
            multiplicity @ callable_matrix,
            int(shared["bootstrap_seed"]),
        )
        selection = evaluate(multiplicity, depth_model)
        if m1_direction_counts is not None:
            m1_direction_counts += selection.m1_direction_state_supported
        depth_refits += 1
        for child, state in selection.local_states.items():
            local_states[replicate, child] = state
        for child, row in selection.local_rows.items():
            local_rows[replicate, child] = row
        for child, row in selection.graph_rows.items():
            graph_rows[replicate, child] = row
    result = (local_rows, graph_rows, local_states, depth_refits)
    return result if m1_direction_counts is None else result + (m1_direction_counts,)


def count_exposed_contigs(
    packed_presence: np.ndarray,
    contig_weights: np.ndarray,
) -> np.ndarray:
    """Count exposed contigs selected by arbitrary bootstrap multiplicities.

    Multiplicity affects this count only through presence (``weight > 0``),
    exactly matching the combined-v1 exposure rule.  The result has the item
    shape of ``packed_presence`` and dtype int32.
    """
    packed = np.asarray(packed_presence)
    weights = np.asarray(contig_weights)
    if packed.dtype != np.uint64 or packed.ndim < 1:
        raise ValueError("packed_presence must be a uint64 array")
    if weights.ndim != 1 or weights.shape[0] < 1:
        raise ValueError("contig_weights must be a non-empty one-dimensional array")
    expected_words = (weights.shape[0] + 63) // 64
    if packed.shape[0] != expected_words:
        raise ValueError("packed presence word count does not match contig weights")
    flat = np.ascontiguousarray(packed.reshape(packed.shape[0], -1))
    words = active_contig_words(weights)
    counts = _count_active_presence_2d(flat, words)
    return counts.reshape(packed.shape[1:])


def _bootstrap_worker(
    multiplicities: np.ndarray,
) -> tuple:
    """Module-scope forkserver callback for one bootstrap chunk."""
    return _evaluate_bootstrap_chunk(
        _BOOTSTRAP_SHARED, multiplicities
    )


@njit(cache=True, fastmath=False, nogil=True)
def _accumulate_bootstrap_counts_into(
    local_rows: np.ndarray,
    graph_rows: np.ndarray,
    local_states: np.ndarray,
    alternatives: np.ndarray,
    alternative_states: np.ndarray | None,
    local_configuration_counts: np.ndarray,
    graph_configuration_counts: np.ndarray,
    local_state_counts: np.ndarray,
    graph_state_counts: np.ndarray | None,
    local_parent_counts: np.ndarray,
    graph_parent_counts: np.ndarray,
) -> None:
    n_replicates, n_samples = local_rows.shape
    for replicate in range(n_replicates):
        for child in range(n_samples):
            state = local_states[replicate, child]
            if state >= 0:
                local_state_counts[child, state] += 1

            local_row = local_rows[replicate, child]
            if local_row >= 0:
                local_configuration_counts[local_row] += 1
                for slot in range(1, 3):
                    parent = alternatives[local_row, slot]
                    if parent >= 0:
                        local_parent_counts[child, parent] += 1

            graph_row = graph_rows[replicate, child]
            if graph_row >= 0:
                graph_configuration_counts[graph_row] += 1
                if graph_state_counts is not None:
                    graph_state = alternative_states[graph_row]
                    if graph_state >= 0:
                        graph_state_counts[child, graph_state] += 1
                for slot in range(1, 3):
                    parent = alternatives[graph_row, slot]
                    if parent >= 0:
                        graph_parent_counts[child, parent] += 1


def _accumulate_bootstrap_chunk(
    chunk: tuple,
    alternatives: np.ndarray,
    local_configuration_counts: np.ndarray,
    graph_configuration_counts: np.ndarray,
    local_state_counts: np.ndarray,
    _graph_state_counts: Optional[np.ndarray],
    local_parent_counts: np.ndarray,
    graph_parent_counts: np.ndarray,
    m1_direction_state_counts: Optional[np.ndarray] = None,
) -> int:
    """Reduce one worker result with order-independent integer additions."""
    local_rows, graph_rows, local_states, depth_refits = chunk[:4]
    if m1_direction_state_counts is not None:
        m1_direction_state_counts += chunk[4]
    # Graph-state counts have always been an unused compatibility argument;
    # leave them untouched while compiling every count that is consumed.
    accumulate_bootstrap_counts_into(
        local_rows,
        graph_rows,
        local_states,
        alternatives,
        None,
        local_configuration_counts,
        graph_configuration_counts,
        local_state_counts,
        None,
        local_parent_counts,
        graph_parent_counts,
    )
    return int(depth_refits)


def _validate_bootstrap_selection_inputs(
    local_rows: np.ndarray,
    graph_rows: np.ndarray,
    local_states: np.ndarray,
    alternatives: np.ndarray,
    alternative_states: np.ndarray | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    local = np.ascontiguousarray(np.asarray(local_rows, dtype=np.int64))
    graph = np.ascontiguousarray(np.asarray(graph_rows, dtype=np.int64))
    selected_states = np.ascontiguousarray(np.asarray(local_states, dtype=np.int8))
    candidate_rows = np.ascontiguousarray(np.asarray(alternatives, dtype=np.int64))
    candidate_states = (
        None
        if alternative_states is None
        else np.ascontiguousarray(
            np.asarray(alternative_states, dtype=np.int8)
        )
    )
    if local.ndim != 2 or graph.shape != local.shape or selected_states.shape != local.shape:
        raise ValueError("local rows, graph rows, and local states must share shape (replicates, samples)")
    if candidate_rows.ndim != 2 or candidate_rows.shape[1] != 3:
        raise ValueError("alternatives must have shape (alternatives, 3)")
    if (
        candidate_states is not None
        and candidate_states.shape != (len(candidate_rows),)
    ):
        raise ValueError(
            "alternative_states must have one entry per alternative"
        )
    n_alternatives = len(candidate_rows)
    for name, rows in (("local", local), ("graph", graph)):
        if np.any(rows < -1) or np.any(rows >= n_alternatives):
            raise ValueError(f"{name} selected rows are outside [-1, n_alternatives)")
    if np.any(selected_states < -1) or np.any(selected_states > 2):
        raise ValueError("local states must lie in {-1, 0, 1, 2}")
    selected = np.concatenate((local[local >= 0], graph[graph >= 0]))
    if len(selected):
        if candidate_states is not None:
            selected_candidate_states = candidate_states[selected]
            if (
                np.any(selected_candidate_states < 0)
                or np.any(selected_candidate_states > 2)
            ):
                raise ValueError(
                    "selected alternative states must lie in {0, 1, 2}"
                )
        selected_parents = candidate_rows[selected, 1:]
        if (
            np.any(selected_parents < -1)
            or np.any(selected_parents >= local.shape[1])
        ):
            raise ValueError(
                "selected alternatives contain invalid parent indices"
            )
    return local, graph, selected_states, candidate_rows, candidate_states


def _run_parent_state_bootstraps(
    contig_log_likelihoods: np.ndarray,
    alternatives: np.ndarray,
    states: np.ndarray,
    by_child: Sequence[np.ndarray],
    full_counts: np.ndarray,
    junction_matrix: Optional[np.ndarray],
    callable_matrix: Optional[np.ndarray],
    settings: module_pedigree_config.PedigreeConfig,
    n_workers: Optional[int],
    local_configuration_counts: np.ndarray,
    graph_configuration_counts: np.ndarray,
    local_state_counts: np.ndarray,
    graph_state_counts: np.ndarray,
    local_parent_counts: np.ndarray,
    graph_parent_counts: np.ndarray,
    *,
    contig_information_weights: Optional[np.ndarray] = None,
    structure_pair_indices: Optional[np.ndarray] = None,
    edge_matched_by_contig: Optional[np.ndarray] = None,
    edge_exposed_by_contig: Optional[np.ndarray] = None,
    pair_explained_by_contig: Optional[np.ndarray] = None,
    pair_exposed_by_contig: Optional[np.ndarray] = None,
    structure_total_bins_by_contig: Optional[np.ndarray] = None,
    edge_exposure_presence_words: Optional[np.ndarray] = None,
    pair_exposure_presence_words: Optional[np.ndarray] = None,
    direction_supported_parents: Optional[np.ndarray] = None,
    scaffold_data: Optional[Mapping[str, Any]] = None,
    m1_direction_state_counts: Optional[np.ndarray] = None,
) -> tuple[int, int]:
    """Run fixed-seed bootstraps serially or in a shared-memory pool."""
    n_contigs = contig_log_likelihoods.shape[0]
    if contig_information_weights is None:
        information_weights = np.ones(n_contigs, dtype=np.float64)
    else:
        information_weights = np.asarray(
            contig_information_weights, dtype=np.float64
        )
        if information_weights.shape != (n_contigs,):
            raise pedigree_models.PedigreeEvidenceError("contig information weights must match contigs")
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
    capacity = int(numba.config.NUMBA_NUM_THREADS)
    if n_workers is None:
        cpu_budget = min(available_cpus, capacity)
    else:
        if int(n_workers) != n_workers or n_workers < 1:
            raise pedigree_models.PedigreeEvidenceError("n_workers must be a positive integer")
        cpu_budget = min(int(n_workers), available_cpus, capacity)
    worker_count = _bootstrap_worker_count(
        len(alternatives),
        settings.bootstrap_replicates,
        cpu_budget,
    )

    scaffold_arrays = {}
    scaffold_metadata = None
    ordinary_shared = {
        "scaffold_metadata": scaffold_metadata,
        "scaffold_array_names": tuple(scaffold_arrays),
        "by_child": tuple(np.asarray(rows, dtype=np.int64) for rows in by_child),
        "settings": settings,
        "n_samples": len(by_child),
        "bootstrap_seed": settings.bootstrap_seed,
        "contig_information_weights": information_weights,
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
            "structure_pair_indices": structure_pair_indices,
            "edge_matched_by_contig": edge_matched_by_contig,
            "edge_exposed_by_contig": edge_exposed_by_contig,
            "pair_explained_by_contig": pair_explained_by_contig,
            "pair_exposed_by_contig": pair_exposed_by_contig,
            "structure_total_bins_by_contig": structure_total_bins_by_contig,
            "edge_exposure_presence_words": edge_exposure_presence_words,
            "direction_supported_parents": direction_supported_parents,
            "pair_exposure_presence_words": pair_exposure_presence_words,
        }
        shared.update({"scaffold_" + key: value for key, value in scaffold_arrays.items()})
        shared["scaffold_prepared"] = None
        results = (_evaluate_bootstrap_chunk(shared, multiplicities),)
        for chunk in results:
            depth_refits += _accumulate_bootstrap_chunk(
                chunk,
                alternatives,
                local_configuration_counts,
                graph_configuration_counts,
                local_state_counts,
                graph_state_counts,
                local_parent_counts,
                graph_parent_counts,
                m1_direction_state_counts,
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
        ("structure_pair_indices", structure_pair_indices),
        ("edge_matched_by_contig", edge_matched_by_contig),
        ("edge_exposed_by_contig", edge_exposed_by_contig),
        ("pair_explained_by_contig", pair_explained_by_contig),
        ("pair_exposed_by_contig", pair_exposed_by_contig),
        ("structure_total_bins_by_contig", structure_total_bins_by_contig),
        ("direction_supported_parents", direction_supported_parents),
        ("edge_exposure_presence_words", edge_exposure_presence_words),
        ("pair_exposure_presence_words", pair_exposure_presence_words),
    ) + tuple(("scaffold_" + key, value) for key, value in scaffold_arrays.items()):
        if array is None:
            shared[key] = None
        else:
            try:
                handle, metadata = core_parallel.create_shared_array(array)
            except BaseException:
                with core_parallel.shared_memory_cleanup(handles):
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
    with core_parallel.shared_memory_cleanup(handles), core_parallel.safe_forkserver_pool(
        worker_count,
        initializer=_init_bootstrap_worker,
        initargs=(shared,),
    ) as pool:
        for chunk in pool.imap_unordered(
            _bootstrap_worker, tasks, chunksize=1
        ):
            depth_refits += _accumulate_bootstrap_chunk(
                chunk,
                alternatives,
                local_configuration_counts,
                graph_configuration_counts,
                local_state_counts,
                graph_state_counts,
                local_parent_counts,
                graph_parent_counts,
                m1_direction_state_counts,
            )
    return worker_count, depth_refits


def accumulate_bootstrap_counts_into(
    local_rows: np.ndarray,
    graph_rows: np.ndarray,
    local_states: np.ndarray,
    alternatives: np.ndarray,
    alternative_states: np.ndarray | None,
    local_configuration_counts: np.ndarray,
    graph_configuration_counts: np.ndarray,
    local_state_counts: np.ndarray,
    graph_state_counts: np.ndarray | None,
    local_parent_counts: np.ndarray,
    graph_parent_counts: np.ndarray,
) -> None:
    """Add a bootstrap selection chunk to caller-owned int64 count arrays."""
    local, graph, selected_states, candidate_rows, candidate_states = (
        _validate_bootstrap_selection_inputs(
            local_rows, graph_rows, local_states, alternatives, alternative_states
        )
    )
    n_samples = local.shape[1]
    n_alternatives = len(candidate_rows)
    expected = (
        (local_configuration_counts, (n_alternatives,), "local configuration"),
        (graph_configuration_counts, (n_alternatives,), "graph configuration"),
        (local_state_counts, (n_samples, 3), "local state"),
        (local_parent_counts, (n_samples, n_samples), "local parent"),
        (graph_parent_counts, (n_samples, n_samples), "graph parent"),
    )
    normalized = []
    for values, shape, name in expected:
        array = np.asarray(values)
        if array.dtype != np.int64 or array.shape != shape or not array.flags.c_contiguous:
            raise ValueError(f"{name} counts must be C-contiguous int64 with shape {shape}")
        normalized.append(array)
    normalized_graph_states = None
    if graph_state_counts is not None:
        normalized_graph_states = np.asarray(graph_state_counts)
        expected_shape = (n_samples, 3)
        if (
            normalized_graph_states.dtype != np.int64
            or normalized_graph_states.shape != expected_shape
            or not normalized_graph_states.flags.c_contiguous
        ):
            raise ValueError(
                "graph state counts must be C-contiguous int64 with shape "
                f"{expected_shape}"
            )
        if candidate_states is None:
            raise ValueError(
                "alternative_states are required when graph states are counted"
            )
    _accumulate_bootstrap_counts_into(
        local,
        graph,
        selected_states,
        candidate_rows,
        candidate_states,
        normalized[0],
        normalized[1],
        normalized[2],
        normalized_graph_states,
        normalized[3],
        normalized[4],
    )


@njit(cache=True, fastmath=False, nogil=True)
def _is_acyclic_parent_rows(
    selected_rows: np.ndarray,
    alternatives: np.ndarray,
) -> bool:
    n_samples = selected_rows.shape[0]
    indegree = np.zeros(n_samples, dtype=np.int64)
    outdegree = np.zeros(n_samples, dtype=np.int64)
    n_edges = 0

    for child in range(n_samples):
        row = selected_rows[child]
        if row < 0:
            continue
        first = alternatives[row, 1]
        second = alternatives[row, 2]
        if first >= 0:
            indegree[child] += 1
            outdegree[first] += 1
            n_edges += 1
        if second >= 0 and second != first:
            indegree[child] += 1
            outdegree[second] += 1
            n_edges += 1

    offsets = np.empty(n_samples + 1, dtype=np.int64)
    offsets[0] = 0
    for node in range(n_samples):
        offsets[node + 1] = offsets[node] + outdegree[node]
    cursor = offsets[:-1].copy()
    outgoing_children = np.empty(n_edges, dtype=np.int64)
    for child in range(n_samples):
        row = selected_rows[child]
        if row < 0:
            continue
        first = alternatives[row, 1]
        second = alternatives[row, 2]
        if first >= 0:
            outgoing_children[cursor[first]] = child
            cursor[first] += 1
        if second >= 0 and second != first:
            outgoing_children[cursor[second]] = child
            cursor[second] += 1

    queue = np.empty(n_samples, dtype=np.int64)
    tail = 0
    for node in range(n_samples):
        if indegree[node] == 0:
            queue[tail] = node
            tail += 1
    head = 0
    visited = 0
    while head < tail:
        node = queue[head]
        head += 1
        visited += 1
        for edge in range(offsets[node], offsets[node + 1]):
            child = outgoing_children[edge]
            indegree[child] -= 1
            if indegree[child] == 0:
                queue[tail] = child
                tail += 1
    return visited == n_samples


def is_acyclic_parent_rows(
    selected_rows: np.ndarray,
    alternatives: np.ndarray,
) -> bool:
    """Return whether parent-to-child edges in selected rows form a DAG.

    ``selected_rows[child]`` is either ``-1`` or the alternative row selected
    for that child.  Duplicate parent slots represent one edge, matching the
    set-based graph used by the exact pedigree selector.
    """
    rows = np.ascontiguousarray(np.asarray(selected_rows, dtype=np.int64))
    candidate_rows = np.ascontiguousarray(np.asarray(alternatives, dtype=np.int64))
    if rows.ndim != 1:
        raise ValueError("selected_rows must be one-dimensional")
    if candidate_rows.ndim != 2 or candidate_rows.shape[1] != 3:
        raise ValueError("alternatives must have shape (alternatives, 3)")
    if np.any(rows < -1) or np.any(rows >= len(candidate_rows)):
        raise ValueError("selected rows are outside [-1, n_alternatives)")
    selected_children = np.flatnonzero(rows >= 0)
    if len(selected_children):
        chosen = rows[selected_children]
        if np.any(candidate_rows[chosen, 0] != selected_children):
            raise ValueError("each selected alternative must belong to its child")
        parents = candidate_rows[chosen, 1:]
        if np.any(parents < -1) or np.any(parents >= len(rows)):
            raise ValueError("selected alternatives contain invalid parent indices")
    return bool(_is_acyclic_parent_rows(rows, candidate_rows))

import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.pedigree.config as module_pedigree_config
import haplotype_reconstruction.pedigree.direction as pedigree_direction
import haplotype_reconstruction.pedigree.models as pedigree_models
import haplotype_reconstruction.pedigree.states as pedigree_states
