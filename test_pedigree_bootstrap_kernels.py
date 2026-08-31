"""Independent equivalence tests for exact pedigree bootstrap kernels."""

from __future__ import annotations

import os
import time
import unittest
from unittest import mock

import numpy as np

import pedigree_inference as pedigree
from pedigree_bootstrap_kernels import (
    accumulate_bootstrap_counts,
    count_exposed_contigs,
    is_acyclic_parent_rows,
    ordered_weighted_contig_sums,
    pack_contig_presence,
)


def _reference_exposed_counts(values, weights):
    values = np.asarray(values)
    weights = np.asarray(weights)
    return np.sum(
        (values > 0) & (weights > 0).reshape((-1,) + (1,) * (values.ndim - 1)),
        axis=0,
        dtype=np.int64,
    )


def _reference_bootstrap_counts(
    local_rows,
    graph_rows,
    local_states,
    alternatives,
    alternative_states,
):
    n_replicates, n_samples = local_rows.shape
    n_alternatives = len(alternatives)
    local_configurations = np.zeros(n_alternatives, dtype=np.int64)
    graph_configurations = np.zeros(n_alternatives, dtype=np.int64)
    local_state_counts = np.zeros((n_samples, 3), dtype=np.int64)
    graph_state_counts = np.zeros((n_samples, 3), dtype=np.int64)
    local_parent_counts = np.zeros((n_samples, n_samples), dtype=np.int64)
    graph_parent_counts = np.zeros((n_samples, n_samples), dtype=np.int64)
    for replicate in range(n_replicates):
        for child in range(n_samples):
            state = int(local_states[replicate, child])
            if state >= 0:
                local_state_counts[child, state] += 1
            for rows, configurations, state_counts, parent_counts, is_graph in (
                (
                    local_rows,
                    local_configurations,
                    local_state_counts,
                    local_parent_counts,
                    False,
                ),
                (
                    graph_rows,
                    graph_configurations,
                    graph_state_counts,
                    graph_parent_counts,
                    True,
                ),
            ):
                row = int(rows[replicate, child])
                if row < 0:
                    continue
                configurations[row] += 1
                if is_graph:
                    state_counts[child, int(alternative_states[row])] += 1
                for parent in alternatives[row, 1:]:
                    if int(parent) >= 0:
                        parent_counts[child, int(parent)] += 1
    return (
        local_configurations,
        graph_configurations,
        local_state_counts,
        graph_state_counts,
        local_parent_counts,
        graph_parent_counts,
    )


def _reference_is_acyclic(selected_rows, alternatives):
    n_samples = len(selected_rows)
    adjacency = [set() for _ in range(n_samples)]
    indegree = np.zeros(n_samples, dtype=np.int64)
    for child, row in enumerate(selected_rows):
        if int(row) < 0:
            continue
        for parent in alternatives[int(row), 1:]:
            parent = int(parent)
            if parent >= 0 and child not in adjacency[parent]:
                adjacency[parent].add(child)
                indegree[child] += 1
    queue = [node for node in range(n_samples) if indegree[node] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in adjacency[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return visited == n_samples


def _reference_ordered_sums(weights, values):
    weights = np.asarray(weights, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    vector = weights.ndim == 1
    matrix = weights[None, :] if vector else weights
    flat = values.reshape(values.shape[0], -1)
    output = np.empty((len(matrix), flat.shape[1]), dtype=np.float64)
    for replicate in range(len(matrix)):
        for item in range(flat.shape[1]):
            total = 0.0
            for contig in range(values.shape[0]):
                total += matrix[replicate, contig] * flat[contig, item]
            output[replicate, item] = total
    shaped = output.reshape((len(matrix),) + values.shape[1:])
    return shaped[0] if vector else shaped


def _selection_fixture(n_samples, n_alternatives, n_replicates, seed):
    if n_alternatives < 3 * n_samples:
        raise ValueError("fixture requires at least three rows per child")
    rng = np.random.default_rng(seed)
    alternatives = np.full((n_alternatives, 3), -1, dtype=np.int64)
    states = np.zeros(n_alternatives, dtype=np.int8)
    for child in range(n_samples):
        base = 3 * child
        alternatives[base, 0] = child
        alternatives[base + 1] = (
            child,
            (child + 1) % n_samples,
            -1,
        )
        alternatives[base + 2] = (
            child,
            (child + 1) % n_samples,
            (child + 2) % n_samples,
        )
        states[base:base + 3] = (0, 1, 2)
    if n_alternatives > 3 * n_samples:
        remaining = np.arange(3 * n_samples, n_alternatives, dtype=np.int64)
        children = remaining % n_samples
        state = (remaining // max(n_samples, 1)) % 3
        alternatives[remaining, 0] = children
        states[remaining] = state.astype(np.int8)
        one_or_two = state >= 1
        alternatives[remaining[one_or_two], 1] = (
            children[one_or_two] + 1
        ) % n_samples
        two = state == 2
        alternatives[remaining[two], 2] = (children[two] + 2) % n_samples

    choices = rng.integers(0, 3, size=(n_replicates, n_samples))
    children = np.arange(n_samples, dtype=np.int64)[None, :]
    local_rows = children * 3 + choices
    graph_rows = children * 3 + rng.integers(
        0, 3, size=(n_replicates, n_samples)
    )
    local_states = states[local_rows]
    local_rows[rng.random(local_rows.shape) < 0.12] = -1
    graph_rows[rng.random(graph_rows.shape) < 0.17] = -1
    local_states[rng.random(local_states.shape) < 0.08] = -1
    # Exercise the upper alternative bound without making a large fixture dense.
    alternatives[-1] = (0, 1, 2 if n_samples > 2 else -1)
    states[-1] = 2 if n_samples > 2 else 1
    local_rows[0, 0] = n_alternatives - 1
    local_states[0, 0] = states[-1]
    return local_rows, graph_rows, local_states, alternatives, states


class ExposurePresenceTests(unittest.TestCase):
    def test_randomized_counts_across_word_boundaries(self):
        rng = np.random.default_rng(20260831)
        for n_contigs in (1, 22, 64, 65, 130):
            with self.subTest(n_contigs=n_contigs):
                values = rng.integers(0, 4, size=(n_contigs, 14, 9), dtype=np.uint8)
                weights = rng.integers(0, 5, size=n_contigs).astype(np.float64)
                packed = pack_contig_presence(values)
                self.assertEqual(packed.shape[0], (n_contigs + 63) // 64)
                np.testing.assert_array_equal(
                    count_exposed_contigs(packed, weights),
                    _reference_exposed_counts(values, weights),
                )

    def test_sample_dimensions_14_84_320(self):
        rng = np.random.default_rng(402)
        for n_samples, n_contigs in ((14, 70), (84, 65), (320, 22)):
            with self.subTest(n_samples=n_samples):
                values = rng.integers(
                    0, 2, size=(n_contigs, n_samples, n_samples), dtype=np.uint8
                )
                weights = rng.poisson(1.0, size=n_contigs).astype(np.float64)
                np.testing.assert_array_equal(
                    count_exposed_contigs(pack_contig_presence(values), weights),
                    _reference_exposed_counts(values, weights),
                )

    def test_539200_presence_rows_and_more_than_64_contigs(self):
        n_contigs = 70
        n_rows = 539_200
        row = np.arange(n_rows, dtype=np.int64)
        # Broadcasting creates the only large fixture; uint8 bounds it at 38 MB.
        values = (
            (row[None, :] + np.arange(n_contigs)[:, None]) % 11 == 0
        ).astype(np.uint8)
        weights = np.zeros(n_contigs, dtype=np.float64)
        weights[[0, 1, 63, 64, 69]] = (2, 1, 5, 3, 1)
        observed = count_exposed_contigs(pack_contig_presence(values), weights)
        expected = _reference_exposed_counts(values, weights)
        np.testing.assert_array_equal(observed, expected)


class BootstrapAccumulatorTests(unittest.TestCase):
    def _assert_fixture(self, n_samples, n_alternatives, n_replicates, seed):
        fixture = _selection_fixture(
            n_samples, n_alternatives, n_replicates, seed
        )
        observed = accumulate_bootstrap_counts(*fixture)
        expected = _reference_bootstrap_counts(*fixture)
        for actual, reference in zip(
            (
                observed.local_configurations,
                observed.graph_configurations,
                observed.local_states,
                observed.graph_states,
                observed.local_parents,
                observed.graph_parents,
            ),
            expected,
        ):
            np.testing.assert_array_equal(actual, reference)

    def test_randomized_sample_scales(self):
        for fixture in (
            (14, 1_000, 7, 11),
            (84, 20_000, 4, 12),
            (320, 539_200, 3, 13),
        ):
            with self.subTest(n_samples=fixture[0]):
                self._assert_fixture(*fixture)


class AcyclicityTests(unittest.TestCase):
    @staticmethod
    def _rows(n_samples, parents):
        alternatives = np.full((n_samples, 3), -1, dtype=np.int64)
        alternatives[:, 0] = np.arange(n_samples)
        for child, values in parents.items():
            alternatives[child, 1:1 + len(values)] = values
        return np.arange(n_samples, dtype=np.int64), alternatives

    def test_explicit_cycles_and_duplicate_edges(self):
        cases = (
            ({1: (0,), 2: (1,)}, True),
            ({0: (1,), 1: (0,)}, False),
            ({0: (2,), 1: (0,), 2: (1,)}, False),
            ({3: (3,)}, False),
            ({2: (0, 0)}, True),
        )
        for parents, expected in cases:
            rows, alternatives = self._rows(4, parents)
            with self.subTest(parents=parents):
                self.assertEqual(
                    is_acyclic_parent_rows(rows, alternatives), expected
                )
                self.assertEqual(
                    is_acyclic_parent_rows(rows, alternatives),
                    _reference_is_acyclic(rows, alternatives),
                )

    def test_randomized_dags_and_injected_cycles_at_sample_scales(self):
        rng = np.random.default_rng(994)
        for n_samples in (14, 84, 320):
            alternatives = np.full((n_samples, 3), -1, dtype=np.int64)
            alternatives[:, 0] = np.arange(n_samples)
            for child in range(1, n_samples):
                parents = rng.choice(
                    child, size=min(2, child), replace=False
                )
                alternatives[child, 1:1 + len(parents)] = parents
            rows = np.arange(n_samples, dtype=np.int64)
            self.assertTrue(is_acyclic_parent_rows(rows, alternatives))
            self.assertEqual(
                is_acyclic_parent_rows(rows, alternatives),
                _reference_is_acyclic(rows, alternatives),
            )
            cyclic = alternatives.copy()
            cyclic[0, 1] = n_samples - 1
            self.assertFalse(is_acyclic_parent_rows(rows, cyclic))
            self.assertEqual(
                is_acyclic_parent_rows(rows, cyclic),
                _reference_is_acyclic(rows, cyclic),
            )


class OrderedWeightedReductionTests(unittest.TestCase):
    def test_exact_left_to_right_reference(self):
        rng = np.random.default_rng(8871)
        for n_contigs, item_shape in ((22, (14,)), (65, (84, 3)), (130, (320,))):
            with self.subTest(n_contigs=n_contigs, item_shape=item_shape):
                weights = rng.integers(0, 5, size=(4, n_contigs)).astype(np.float64)
                values = -rng.exponential(size=(n_contigs,) + item_shape)
                observed = ordered_weighted_contig_sums(weights, values)
                expected = _reference_ordered_sums(weights, values)
                np.testing.assert_array_equal(observed, expected)
                np.testing.assert_array_equal(
                    ordered_weighted_contig_sums(weights[0], values), expected[0]
                )

    def test_numpy_matmul_difference_is_bounded_for_model_like_values(self):
        rng = np.random.default_rng(1041)
        weights = rng.integers(0, 5, size=(5, 130)).astype(np.float64)
        likelihoods = -rng.exponential(size=(130, 4096))
        observed = ordered_weighted_contig_sums(weights, likelihoods)
        numpy_batched = np.vstack([weight @ likelihoods for weight in weights])
        # BLAS may use a different reduction tree; repository policy permits a
        # few hundred ULP for mathematically identical evaluation order changes.
        np.testing.assert_array_max_ulp(observed, numpy_batched, maxulp=256)


class PedigreeIntegrationTests(unittest.TestCase):
    @staticmethod
    def _aggregate_fixture(cyclic=False):
        if cyclic:
            alternatives = np.asarray((
                (0, -1, -1), (0, 1, -1),
                (1, -1, -1), (1, 0, -1),
                (2, -1, -1),
            ), dtype=np.int64)
            aggregate = np.asarray((0.0, 10.0, 0.0, 10.0, 5.0))
        else:
            alternatives = np.asarray((
                (0, -1, -1),
                (1, -1, -1), (1, 0, -1),
                (2, -1, -1), (2, 0, -1), (2, 0, 1),
            ), dtype=np.int64)
            aggregate = np.asarray((5.0, 0.0, 9.0, 0.0, 2.0, 10.0))
        states = np.count_nonzero(alternatives[:, 1:] >= 0, axis=1).astype(
            np.int8
        )
        n_samples = 3
        by_child = tuple(
            np.flatnonzero(alternatives[:, 0] == child).astype(np.int64)
            for child in range(n_samples)
        )
        full_counts = np.zeros((n_samples, 3), dtype=np.int64)
        for child, rows in enumerate(by_child):
            full_counts[child] = np.bincount(states[rows], minlength=3)
        return aggregate, alternatives, states, by_child, full_counts

    @staticmethod
    def _aggregate_selection(fixture):
        aggregate, alternatives, states, by_child, full_counts = fixture
        return pedigree._evaluate_parent_state_aggregate(
            aggregate,
            alternatives,
            states,
            by_child,
            full_counts,
            (1.0 / 3.0,) * 3,
            3.0,
            100,
            1e-10,
            len(by_child),
            3,
            use_cohort_prior=False,
            use_fixed_base_priors=True,
        )

    def test_packed_structure_exposure_is_exact_fallback_equivalent(self):
        rng = np.random.default_rng(7301)
        n_contigs = 70
        alternatives = np.asarray((
            (0, -1, -1), (1, -1, -1), (2, -1, -1),
            (2, 0, -1), (2, 1, -1), (2, 0, 1),
        ), dtype=np.int64)
        states = np.asarray((0, 0, 0, 1, 1, 2), dtype=np.int8)
        pair_indices = np.asarray((-1, -1, -1, -1, -1, 0), dtype=np.int64)
        edge_exposed = rng.integers(
            0, 8, size=(n_contigs, 3, 3)
        ).astype(np.float64)
        edge_matched = edge_exposed * rng.uniform(
            0.94, 1.0, size=edge_exposed.shape
        )
        pair_exposed = rng.integers(
            0, 8, size=(n_contigs, 1)
        ).astype(np.float64)
        pair_explained = pair_exposed * rng.uniform(
            0.94, 1.0, size=pair_exposed.shape
        )
        total_bins = np.full(n_contigs, 10.0)
        depth = np.asarray(((1.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
        settings = pedigree.PedigreeConfig(
            parent_state_minimum_exposed_contigs=3,
        ).validated()
        weights = rng.integers(0, 4, size=n_contigs).astype(np.float64)
        args = (
            weights,
            alternatives,
            states,
            pair_indices,
            edge_matched,
            edge_exposed,
            pair_explained,
            pair_exposed,
            total_bins,
            depth,
            settings,
        )
        fallback = pedigree._parent_state_structure_mask(*args)
        packed = pedigree._parent_state_structure_mask(
            *args,
            pack_contig_presence(edge_exposed),
            pack_contig_presence(pair_exposed),
        )
        for observed, expected in zip(packed, fallback):
            np.testing.assert_array_equal(observed, expected)

    def test_compiled_reducer_keeps_unused_graph_state_array_untouched(self):
        fixture = _selection_fixture(14, 1_000, 7, 981)
        expected = _reference_bootstrap_counts(*fixture)
        graph_state_sentinel = np.full((14, 3), 17, dtype=np.int64)
        arrays = (
            np.zeros(1_000, dtype=np.int64),
            np.zeros(1_000, dtype=np.int64),
            np.zeros((14, 3), dtype=np.int64),
            graph_state_sentinel,
            np.zeros((14, 14), dtype=np.int64),
            np.zeros((14, 14), dtype=np.int64),
        )
        pedigree._accumulate_smart_bootstrap_chunk(
            (*fixture[:3], 7), fixture[3], *arrays
        )
        np.testing.assert_array_equal(arrays[0], expected[0])
        np.testing.assert_array_equal(arrays[1], expected[1])
        np.testing.assert_array_equal(arrays[2], expected[2])
        np.testing.assert_array_equal(arrays[3], 17)
        np.testing.assert_array_equal(arrays[4], expected[4])
        np.testing.assert_array_equal(arrays[5], expected[5])

    def test_acyclic_aggregate_shortcut_matches_full_graph_selector(self):
        fixture = self._aggregate_fixture(cyclic=False)
        shortcut = self._aggregate_selection(fixture)
        with mock.patch.object(
            pedigree, "is_acyclic_parent_rows", return_value=False
        ):
            full = self._aggregate_selection(fixture)
        self.assertEqual(shortcut.graph_rows, full.graph_rows)
        self.assertEqual(shortcut.graph_tie_conflicts, full.graph_tie_conflicts)
        self.assertEqual(
            shortcut.graph_direction_resolved_children,
            full.graph_direction_resolved_children,
        )
        self.assertEqual(
            shortcut.graph_parent_role_probabilities,
            full.graph_parent_role_probabilities,
        )

    def test_cyclic_aggregate_retains_full_fallback(self):
        fixture = self._aggregate_fixture(cyclic=True)
        original = pedigree._graph_tie_conflict_children
        with mock.patch.object(
            pedigree,
            "_graph_tie_conflict_children",
            wraps=original,
        ) as fallback:
            selection = self._aggregate_selection(fixture)
        self.assertTrue(fallback.called)
        self.assertTrue(selection.graph_tie_conflicts)


@unittest.skipUnless(
    os.environ.get("PEDIGREE_KERNEL_BENCHMARKS") == "1",
    "set PEDIGREE_KERNEL_BENCHMARKS=1 to run warm microbenchmarks",
)
class WarmKernelBenchmarks(unittest.TestCase):
    def test_warm_representative_operations(self):
        rng = np.random.default_rng(73)
        n_samples = 320
        n_rows = 539_200
        n_contigs = 22
        fixture = _selection_fixture(n_samples, n_rows, 4, 81)

        # Warm every specialization before measuring.
        accumulate_bootstrap_counts(*fixture)
        start = time.perf_counter()
        accumulate_bootstrap_counts(*fixture)
        accumulator_seconds = time.perf_counter() - start

        exposure = rng.integers(
            0, 2, size=(n_contigs, n_samples, n_samples), dtype=np.uint8
        )
        packed = pack_contig_presence(exposure)
        weights = rng.poisson(1.0, size=n_contigs).astype(np.float64)
        count_exposed_contigs(packed, weights)
        start = time.perf_counter()
        count_exposed_contigs(packed, weights)
        exposure_seconds = time.perf_counter() - start

        selected = np.arange(n_samples, dtype=np.int64) * 3
        is_acyclic_parent_rows(selected, fixture[3])
        start = time.perf_counter()
        for _ in range(1_000):
            is_acyclic_parent_rows(selected, fixture[3])
        dag_seconds = time.perf_counter() - start

        likelihoods = -rng.exponential(size=(n_contigs, n_rows))
        batch_weights = rng.poisson(1.0, size=(4, n_contigs)).astype(np.float64)
        ordered_weighted_contig_sums(batch_weights, likelihoods)
        start = time.perf_counter()
        ordered_weighted_contig_sums(batch_weights, likelihoods)
        reduction_seconds = time.perf_counter() - start

        print(
            "warm pedigree kernels:",
            f"accumulator={accumulator_seconds:.6f}s",
            f"exposure={exposure_seconds:.6f}s",
            f"1000_dag_checks={dag_seconds:.6f}s",
            f"weighted_4x22x539200={reduction_seconds:.6f}s",
        )
        for elapsed in (
            accumulator_seconds,
            exposure_seconds,
            dag_seconds,
            reduction_seconds,
        ):
            self.assertGreaterEqual(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
