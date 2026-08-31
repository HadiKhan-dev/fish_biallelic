"""Focused scientific tests for candidate founder-source posteriors."""

from dataclasses import replace
from itertools import product
import unittest

import numpy as np

import pedigree_candidate_source_posterior as source_model

from pedigree_candidate_source_posterior import (
    CandidateSourceTrajectoryDraws,
    infer_candidate_source_posterior,
    reference_conditional_child_likelihood_mc,
    score_conditional_child_tensor_exact,
    score_candidate_source_batch_exact,
    score_candidate_source_batch_matched_null_exact,
    sample_candidate_source_trajectories,
)


def _gl(*values):
    return np.asarray(values, dtype=np.float64)


def _infer_debug(*args, **kwargs):
    kwargs["return_state_posterior"] = True
    return infer_candidate_source_posterior(*args, **kwargs)


def _genotype_distribution(first_alt, second_alt):
    return np.asarray(
        (
            (1.0 - first_alt) * (1.0 - second_alt),
            first_alt * (1.0 - second_alt)
            + (1.0 - first_alt) * second_alt,
            first_alt * second_alt,
        ),
        dtype=np.float64,
    )


def _enumerated_two_bin_posterior(gl, founders, rho, root_prior, eta=1.0):
    """Independent enumeration of every F=2, B=2 ordered-state path."""
    n_founders = founders.shape[0]
    transition = np.full(
        (n_founders, n_founders), rho / (n_founders - 1), dtype=np.float64
    )
    np.fill_diagonal(transition, 1.0 - rho)
    states = list(product(range(n_founders), repeat=2))
    weights = []
    paths = []
    for first_state in states:
        for second_state in states:
            weight = root_prior[first_state]
            weight *= transition[first_state[0], second_state[0]]
            weight *= transition[first_state[1], second_state[1]]
            for block, state in enumerate((first_state, second_state)):
                q = _genotype_distribution(
                    founders[state[0], block, 0],
                    founders[state[1], block, 0],
                )
                weight *= (3.0 * np.dot(gl[block, 0], q)) ** eta
            weights.append(weight)
            paths.append((first_state, second_state))
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    posterior = np.zeros((2, n_founders, n_founders), dtype=np.float64)
    for path_weight, path in zip(weights, paths):
        for block, state in enumerate(path):
            posterior[block, state[0], state[1]] += path_weight
    return posterior


def _linked_path_distribution(result, sample):
    initial = np.exp(
        result.linked_initial_log_probability[sample].astype(np.float64)
    )
    initial /= initial.sum()
    n_bins = result.linked_next_log_weight.shape[1] + 1
    n_founders = initial.shape[0]
    states = list(product(range(n_founders), repeat=2))
    paths = [((first, second),) for first, second in states]
    probabilities = [initial[first, second] for first, second in states]
    for block in range(1, n_bins):
        transition = result.linked_transition_probability[
            block - 1
        ].astype(np.float64)
        right = np.exp(
            result.linked_next_log_weight[sample, block - 1].astype(np.float64)
        )
        next_paths = []
        next_probabilities = []
        for path, path_probability in zip(paths, probabilities):
            previous = path[-1]
            weights = np.empty((n_founders, n_founders), dtype=np.float64)
            for first, second in states:
                weights[first, second] = (
                    transition[previous[0], first]
                    * transition[previous[1], second]
                    * right[first, second]
                )
            weights /= weights.sum()
            for first, second in states:
                next_paths.append(path + ((first, second),))
                next_probabilities.append(
                    path_probability * weights[first, second]
                )
        paths = next_paths
        probabilities = next_probabilities
    return paths, np.asarray(probabilities)


def _track_path_distribution(n_bins, switch_probability):
    paths = list(product(range(2), repeat=n_bins))
    probabilities = []
    for path in paths:
        value = 0.5
        for block in range(1, n_bins):
            value *= (
                1.0 - switch_probability
                if path[block] == path[block - 1]
                else switch_probability
            )
        probabilities.append(value)
    return paths, np.asarray(probabilities)


def _exact_called_child_log_likelihood(
    child_gl,
    founders,
    first_source,
    *,
    second_source=None,
    track_switch_probability=0.1,
):
    n_bins, n_slots, _ = child_gl.shape
    first_paths, first_probability = _linked_path_distribution(first_source, 0)
    track_paths, track_probability = _track_path_distribution(
        n_bins, track_switch_probability
    )
    if second_source is None:
        second_paths = [None]
        second_probability = np.ones(1)
        second_track_paths = [None]
        second_track_probability = np.ones(1)
    else:
        second_paths, second_probability = _linked_path_distribution(
            second_source, 1
        )
        second_track_paths = track_paths
        second_track_probability = track_probability
    likelihood = 0.0
    for first_index, first_path in enumerate(first_paths):
        for first_track_index, first_track_path in enumerate(track_paths):
            for second_index, second_path in enumerate(second_paths):
                for second_track_index, second_track_path in enumerate(
                    second_track_paths
                ):
                    value = 1.0
                    for block in range(n_bins):
                        first_founder = first_path[block][
                            first_track_path[block]
                        ]
                        for snp in range(n_slots):
                            first_allele = founders[
                                first_founder, block, snp
                            ]
                            if second_path is None:
                                second_allele = 0
                            else:
                                second_founder = second_path[block][
                                    second_track_path[block]
                                ]
                                second_allele = founders[
                                    second_founder, block, snp
                                ]
                            value *= 3.0 * child_gl[
                                block, snp, first_allele + second_allele
                            ]
                    likelihood += (
                        first_probability[first_index]
                        * track_probability[first_track_index]
                        * second_probability[second_index]
                        * second_track_probability[second_track_index]
                        * value
                    )
    return np.log(likelihood)


def _lumped_path_distribution(result, sample):
    initial = np.exp(
        result.lumped_initial_log_probability[sample].astype(np.float64)
    )
    initial /= initial.sum()
    n_bins = result.lumped_next_log_weight.shape[1] + 1
    n_founders = initial.shape[0]
    states = list(product(range(n_founders), repeat=2))
    paths = [((first, second),) for first, second in states]
    probabilities = [initial[first, second] for first, second in states]
    for boundary in range(n_bins - 1):
        transition = result.lumped_transition_probability[
            boundary
        ].astype(np.float64)
        right = np.exp(
            result.lumped_next_log_weight[sample, boundary].astype(np.float64)
        )
        expanded_paths = []
        expanded_probabilities = []
        for path, probability in zip(paths, probabilities):
            previous = path[-1]
            weights = (
                transition[previous[0], :, None]
                * transition[previous[1], None, :]
                * right
            )
            weights /= weights.sum()
            for first, second in states:
                expanded_paths.append(path + ((first, second),))
                expanded_probabilities.append(
                    probability * weights[first, second]
                )
        paths = expanded_paths
        probabilities = expanded_probabilities
    return paths, np.asarray(probabilities)


def _external_path_distribution(initial, transitions):
    n_bins = transitions.shape[0] + 1
    n_states = len(initial)
    paths = list(product(range(n_states), repeat=n_bins))
    probabilities = []
    for path in paths:
        value = initial[path[0]]
        for boundary in range(n_bins - 1):
            value *= transitions[boundary, path[boundary], path[boundary + 1]]
        probabilities.append(value)
    return paths, np.asarray(probabilities)


def _exact_full_source_track_likelihood(child, founders, source, samples):
    n_bins, n_slots, _ = child.shape
    track_paths, track_probability = _track_path_distribution(n_bins, 0.13)
    source_distributions = [
        _lumped_path_distribution(source, sample) for sample in samples
    ]
    external_initial = np.full(founders.shape[0], 1.0 / founders.shape[0])
    external_paths, external_probability = _external_path_distribution(
        external_initial,
        source.lumped_transition_probability.astype(np.float64),
    )
    if len(samples) == 0:
        first_paths, first_probability = external_paths, external_probability
        second_paths, second_probability = external_paths, external_probability
        first_tracks = second_tracks = [None]
        first_track_probability = second_track_probability = np.ones(1)
    elif len(samples) == 1:
        first_paths, first_probability = source_distributions[0]
        second_paths, second_probability = external_paths, external_probability
        first_tracks, first_track_probability = track_paths, track_probability
        second_tracks = [None]
        second_track_probability = np.ones(1)
    else:
        first_paths, first_probability = source_distributions[0]
        second_paths, second_probability = source_distributions[1]
        first_tracks = second_tracks = track_paths
        first_track_probability = second_track_probability = track_probability
    total = 0.0
    for first_index, first_path in enumerate(first_paths):
        for first_track_index, first_track in enumerate(first_tracks):
            for second_index, second_path in enumerate(second_paths):
                for second_track_index, second_track in enumerate(second_tracks):
                    value = 1.0
                    for block in range(n_bins):
                        if first_track is None:
                            first_founder = first_path[block]
                        else:
                            first_founder = first_path[block][first_track[block]]
                        if second_track is None:
                            second_founder = second_path[block]
                        else:
                            second_founder = second_path[block][second_track[block]]
                        for snp in range(n_slots):
                            genotype = (
                                founders[first_founder, block, snp]
                                + founders[second_founder, block, snp]
                            )
                            value *= 3.0 * child[block, snp, genotype]
                    total += (
                        first_probability[first_index]
                        * first_track_probability[first_track_index]
                        * second_probability[second_index]
                        * second_track_probability[second_track_index]
                        * value
                    )
    return np.log(total)
def _null_gamete_path_distribution(
    n_bins: int,
    n_founders: int,
    source_switch_probability: float,
    transmission_switch_probability: float,
) -> dict[tuple[int, ...], float]:
    transition = source_model._transition_matrix(
        n_founders, source_switch_probability
    )
    homolog_paths = list(product(range(n_founders), repeat=n_bins))
    homolog_probability = {}
    for path in homolog_paths:
        value = 1.0 / float(n_founders)
        for boundary in range(n_bins - 1):
            value *= transition[
                path[boundary], path[boundary + 1]
            ]
        homolog_probability[path] = value
    selector_paths, selector_probability = _track_path_distribution(
        n_bins, transmission_switch_probability
    )
    output: dict[tuple[int, ...], float] = {}
    for first in homolog_paths:
        for second in homolog_paths:
            diploid_probability = (
                homolog_probability[first] * homolog_probability[second]
            )
            for selector, selector_mass in zip(
                selector_paths, selector_probability
            ):
                transmitted = tuple(
                    (first, second)[selector[block]][block]
                    for block in range(n_bins)
                )
                output[transmitted] = output.get(transmitted, 0.0) + (
                    diploid_probability * selector_mass
                )
    return output


def _enumerated_matched_null_log_likelihood(
    child_gl: np.ndarray,
    founders: np.ndarray,
    source_switch_probability: float,
    transmission_switch_probability: float,
) -> float:
    gametes = _null_gamete_path_distribution(
        child_gl.shape[0],
        founders.shape[0],
        source_switch_probability,
        transmission_switch_probability,
    )
    likelihood = 0.0
    for first_path, first_probability in gametes.items():
        for second_path, second_probability in gametes.items():
            value = 1.0
            for block in range(child_gl.shape[0]):
                for snp in range(child_gl.shape[1]):
                    genotype = (
                        founders[first_path[block], block, snp]
                        + founders[second_path[block], block, snp]
                    )
                    value *= 3.0 * child_gl[block, snp, genotype]
            likelihood += first_probability * second_probability * value
    return float(np.log(likelihood))


class CandidateSourcePosteriorTests(unittest.TestCase):
    def test_normalized_gl_scale_invariance(self):
        founders = np.asarray(
            [
                [[0, 0], [1, 0]],
                [[1, 1], [0, 1]],
                [[0, 1], [1, 1]],
            ],
            dtype=np.int8,
        )
        gl = np.asarray(
            [[[[0.8, 0.15, 0.05], [0.1, 0.8, 0.1]],
              [[0.05, 0.25, 0.7], [0.2, 0.7, 0.1]]]],
            dtype=np.float64,
        )
        scale = np.asarray([[[[7.0], [0.13]], [[31.0], [2.5]]]])
        first = _infer_debug(
            gl, founders, np.asarray([2, 2]), 0.08, eta=0.7
        )
        second = _infer_debug(
            gl * scale, founders, np.asarray([2, 2]), 0.08, eta=0.7
        )
        np.testing.assert_allclose(
            first.source_posterior, second.source_posterior, atol=2e-15
        )
        np.testing.assert_allclose(
            first.track_alt_probability,
            second.track_alt_probability,
            atol=2e-15,
        )

    def test_default_retained_output_is_bounded_float32(self):
        founders = np.asarray([[[0], [0]], [[1], [1]]], dtype=np.int8)
        gl = np.asarray(
            [[[[0.1, 0.8, 0.1]], [[0.2, 0.7, 0.1]]]], dtype=np.float64
        )
        result = infer_candidate_source_posterior(
            gl, founders, np.asarray([1, 1]), 0.05
        )
        self.assertIsNone(result.source_posterior)
        self.assertIsNone(result.linked_initial_log_probability)
        self.assertIsNone(result.linked_next_log_weight)
        self.assertIsNone(result.linked_transition_probability)
        self.assertIsNone(result.lumped_initial_log_probability)
        self.assertIsNone(result.lumped_next_log_weight)
        self.assertIsNone(result.lumped_transition_probability)
        self.assertFalse(result.lumped_available[0]) if not result.available[0] else None
        self.assertEqual(result.track_alt_probability.dtype, np.float32)
        self.assertEqual(result.track_alt_probability.nbytes, 1 * 2 * 2 * 1 * 4)

    def test_uniform_candidate_is_explicitly_unavailable(self):
        gl = np.full((2, 3, 2, 3), 1.0 / 3.0)
        founders = np.asarray(
            [
                [[0, 1], [0, 1], [1, 0]],
                [[1, 0], [1, 0], [0, 1]],
            ],
            dtype=np.int8,
        )
        anchor = np.zeros((2, 3, 2), dtype=np.int64)
        result = _infer_debug(
            gl,
            founders,
            np.asarray([2, 2, 2]),
            0.03,
            painted_track_labels=anchor,
        )
        np.testing.assert_array_equal(result.available, [False, False])
        np.testing.assert_array_equal(result.informative_site_count, [0, 0])
        self.assertTrue(np.all(np.isnan(result.source_posterior)))
        self.assertTrue(np.all(np.isnan(result.track_alt_probability)))
        self.assertFalse(np.any(result.inconsistent))

    def test_founder_id_permutation_with_permuted_anchor(self):
        founders = np.asarray(
            [
                [[0, 0], [0, 1]],
                [[1, 1], [1, 0]],
                [[0, 1], [1, 1]],
            ],
            dtype=np.int8,
        )
        gl = np.asarray(
            [[[[0.05, 0.9, 0.05], [0.8, 0.15, 0.05]],
              [[0.1, 0.8, 0.1], [0.05, 0.2, 0.75]]]],
            dtype=np.float64,
        )
        anchor = np.asarray([[[2, 1], [-1, -1]]], dtype=np.int64)
        original = _infer_debug(
            gl,
            founders,
            np.asarray([2, 2]),
            0.07,
            painted_track_labels=anchor,
        )

        permutation = np.asarray([2, 0, 1])
        inverse = np.argsort(permutation)
        permuted_anchor = anchor.copy()
        for old_id in range(3):
            permuted_anchor[anchor == old_id] = inverse[old_id]
        permuted = _infer_debug(
            gl,
            founders[permutation],
            np.asarray([2, 2]),
            0.07,
            painted_track_labels=permuted_anchor,
        )
        np.testing.assert_allclose(
            original.track_alt_probability,
            permuted.track_alt_probability,
            atol=2e-14,
        )
        restored = permuted.source_posterior[:, :, inverse][:, :, :, inverse]
        np.testing.assert_allclose(
            original.source_posterior, restored, atol=2e-14
        )

    def test_unanchored_gauge_is_flagged_and_founder_order_independent(self):
        founders = np.asarray(
            [
                [[1, 0], [0, 1]],
                [[0, 0], [1, 1]],
                [[0, 1], [0, 0]],
            ],
            dtype=np.int8,
        )
        gl = np.asarray(
            [[[[0.1, 0.8, 0.1], [0.75, 0.2, 0.05]],
              [[0.2, 0.7, 0.1], [0.8, 0.15, 0.05]]]],
            dtype=np.float64,
        )
        original = _infer_debug(
            gl, founders, np.asarray([2, 2]), 0.09
        )
        permutation = np.asarray([1, 2, 0])
        inverse = np.argsort(permutation)
        permuted = _infer_debug(
            gl, founders[permutation], np.asarray([2, 2]), 0.09
        )
        self.assertFalse(original.gauge_anchored[0])
        np.testing.assert_allclose(
            original.track_alt_probability,
            permuted.track_alt_probability,
            atol=2e-7,
        )
        restored = permuted.source_posterior[:, :, inverse][:, :, :, inverse]
        np.testing.assert_allclose(
            original.source_posterior, restored, atol=2e-14
        )

    def test_swapping_gauge_anchor_only_swaps_tracks(self):
        founders = np.asarray(
            [
                [[0, 0], [0, 1], [1, 0]],
                [[1, 1], [1, 0], [0, 1]],
                [[0, 1], [1, 1], [1, 1]],
            ],
            dtype=np.int8,
        )
        gl = np.asarray(
            [[[[0.05, 0.9, 0.05], [0.1, 0.8, 0.1]],
              [[0.1, 0.8, 0.1], [0.2, 0.7, 0.1]],
              [[0.05, 0.85, 0.1], [0.1, 0.8, 0.1]]]],
            dtype=np.float64,
        )
        anchor = np.full((1, 3, 2), -1, dtype=np.int64)
        anchor[0, 0] = (0, 1)
        swapped_anchor = anchor[..., ::-1].copy()
        direct = _infer_debug(
            gl,
            founders,
            np.asarray([2, 2, 2]),
            np.asarray([0.03, 0.12]),
            painted_track_labels=anchor,
        )
        swapped = _infer_debug(
            gl,
            founders,
            np.asarray([2, 2, 2]),
            np.asarray([0.03, 0.12]),
            painted_track_labels=swapped_anchor,
        )
        np.testing.assert_allclose(
            direct.track_alt_probability,
            swapped.track_alt_probability[:, :, ::-1],
            atol=2e-14,
        )
        np.testing.assert_allclose(
            direct.source_posterior,
            swapped.source_posterior.swapaxes(-1, -2),
            atol=2e-14,
        )
        # Any downstream genotype distribution is invariant to this gauge.
        for first, second in zip(
            direct.track_alt_probability.ravel(),
            swapped.track_alt_probability[:, :, ::-1].ravel(),
        ):
            self.assertAlmostEqual(first, second, places=13)

    def test_f2_b2_matches_exact_path_enumeration(self):
        founders = np.asarray([[[0], [1]], [[1], [0]]], dtype=np.int8)
        gl = np.asarray(
            [[[[0.1, 0.7, 0.2]], [[0.65, 0.3, 0.05]]]], dtype=np.float64
        )
        anchor = np.asarray([[[0, 1], [-1, -1]]])
        rho = 0.17
        eta = 0.8
        result = _infer_debug(
            gl,
            founders,
            np.asarray([1, 1]),
            rho,
            eta=eta,
            painted_track_labels=anchor,
        )
        root_prior = np.asarray([[1.0 / 3.0, 1.0 / 3.0], [0.0, 1.0 / 3.0]])
        expected = _enumerated_two_bin_posterior(
            gl[0], founders, rho, root_prior, eta=eta
        )
        np.testing.assert_allclose(
            result.source_posterior[0], expected, rtol=2e-14, atol=2e-14
        )

    def test_anchor_at_first_informative_nonzero_bin_roots_gauge(self):
        founders = np.asarray(
            [[[0], [0], [0]], [[1], [1], [1]]], dtype=np.int8
        )
        gl = np.asarray(
            [[
                [[1.0, 1.0, 1.0]],
                [[0.001, 0.998, 0.001]],
                [[0.001, 0.998, 0.001]],
            ]],
            dtype=np.float64,
        )
        anchor = np.full((1, 3, 2), -1, dtype=np.int64)
        anchor[0, 1] = (0, 1)
        result = _infer_debug(
            gl,
            founders,
            np.asarray([1, 1, 1]),
            0.0,
            eta=2.0,
            painted_track_labels=anchor,
        )
        self.assertEqual(result.gauge_anchor_bin[0], 1)
        self.assertTrue(result.gauge_anchored[0])
        self.assertFalse(result.informative_bins[0, 0])
        self.assertLess(result.track_alt_probability[0, 0, 0, 0], 0.01)
        self.assertGreater(result.track_alt_probability[0, 0, 1, 0], 0.99)

    def test_linkage_imputes_an_internal_uniform_gap(self):
        founders = np.asarray(
            [
                [[0, 0], [0, 0], [0, 0]],
                [[1, 1], [1, 1], [1, 1]],
            ],
            dtype=np.int8,
        )
        gl = np.asarray(
            [[
                [[0.001, 0.998, 0.001], [0.001, 0.998, 0.001]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.001, 0.998, 0.001], [0.001, 0.998, 0.001]],
            ]],
            dtype=np.float64,
        )
        anchor = np.full((1, 3, 2), -1, dtype=np.int64)
        anchor[0, 0] = (0, 1)
        result = _infer_debug(
            gl,
            founders,
            np.asarray([2, 2, 2]),
            0.01,
            eta=2.0,
            painted_track_labels=anchor,
        )
        self.assertFalse(result.informative_bins[0, 1])
        self.assertTrue(result.available[0])
        self.assertLess(result.track_alt_probability[0, 1, 0, 0], 0.03)
        self.assertGreater(result.track_alt_probability[0, 1, 1, 0], 0.97)

    def test_output_normalization_bounds_and_founder_only_missingness(self):
        founders = np.asarray(
            [
                [[0, -1, -1]],
                [[1, -1, -1]],
                [[-1, -1, -1]],
            ],
            dtype=np.int8,
        )
        gl = np.asarray(
            [[[[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.05, 0.05, 0.9]]]],
            dtype=np.float64,
        )
        result = _infer_debug(
            gl, founders, np.asarray([3]), 0.0
        )
        self.assertTrue(result.available[0])
        self.assertEqual(result.informative_site_count[0], 1)
        np.testing.assert_allclose(
            result.source_posterior.sum(axis=(-1, -2)), 1.0, atol=2e-15
        )
        self.assertEqual(result.founder_alt_frequency[0, 0], 0.5)
        self.assertTrue(np.isnan(result.founder_alt_frequency[0, 1]))
        self.assertTrue(np.isnan(result.track_alt_probability[0, 0, :, 1:]).all())
        finite = result.track_alt_probability[np.isfinite(result.track_alt_probability)]
        self.assertTrue(np.all((finite >= 0.0) & (finite <= 1.0)))
        self.assertTrue(np.all(result.posterior_entropy >= 0.0))
        self.assertTrue(
            np.all(
                (result.max_state_posterior >= 0.0)
                & (result.max_state_posterior <= 1.0)
            )
        )

    def test_linked_factors_reconstruct_exact_two_bin_joint(self):
        founders = np.asarray([[[0], [1]], [[1], [0]]], dtype=np.int8)
        gl = np.asarray(
            [[[[0.1, 0.7, 0.2]], [[0.65, 0.3, 0.05]]]], dtype=np.float64
        )
        anchor = np.asarray([[[0, 1], [-1, -1]]])
        rho = 0.17
        result = infer_candidate_source_posterior(
            gl,
            founders,
            np.asarray([1, 1]),
            rho,
            eta=0.8,
            painted_track_labels=anchor,
            return_state_posterior=True,
            return_linked_posterior=True,
        )
        initial = np.exp(result.linked_initial_log_probability[0].astype(float))
        transition = result.linked_transition_probability[0].astype(float)
        right_weight = np.exp(
            result.linked_next_log_weight[0, 0].astype(float)
        )
        denominator = transition @ right_weight @ transition.T
        reconstructed = np.empty((2, 2, 2, 2), dtype=np.float64)
        for first_i, first_j, second_i, second_j in product(range(2), repeat=4):
            reconstructed[first_i, first_j, second_i, second_j] = (
                initial[first_i, first_j]
                * transition[first_i, second_i]
                * transition[first_j, second_j]
                * right_weight[second_i, second_j]
                / denominator[first_i, first_j]
            )
        expected = np.empty_like(reconstructed)
        root_prior = np.asarray(
            [[1.0 / 3.0, 1.0 / 3.0], [0.0, 1.0 / 3.0]]
        )
        for first_i, first_j, second_i, second_j in product(range(2), repeat=4):
            first_q = _genotype_distribution(
                founders[first_i, 0, 0], founders[first_j, 0, 0]
            )
            second_q = _genotype_distribution(
                founders[second_i, 1, 0], founders[second_j, 1, 0]
            )
            expected[first_i, first_j, second_i, second_j] = (
                root_prior[first_i, first_j]
                * transition[first_i, second_i]
                * transition[first_j, second_j]
                * (3.0 * np.dot(gl[0, 0, 0], first_q)) ** 0.8
                * (3.0 * np.dot(gl[0, 1, 0], second_q)) ** 0.8
            )
        expected /= expected.sum()
        np.testing.assert_allclose(reconstructed, expected, atol=2e-7)
        np.testing.assert_allclose(reconstructed.sum(), 1.0, atol=2e-7)
        np.testing.assert_allclose(
            reconstructed.sum(axis=(0, 1)),
            result.source_posterior[0, 1],
            atol=2e-7,
        )

    def test_linked_state_not_site_marginals_preserves_multisite_likelihood(self):
        founders = np.asarray([[[0, 0]], [[1, 1]]], dtype=np.int8)
        candidate_gl = np.asarray(
            [[[[0.5, 0.0, 0.5], [0.5, 0.0, 0.5]]]], dtype=np.float64
        )
        result = infer_candidate_source_posterior(
            candidate_gl,
            founders,
            np.asarray([2]),
            0.0,
            return_linked_posterior=True,
        )
        initial = np.exp(result.linked_initial_log_probability[0].astype(float))
        initial /= initial.sum()
        child_ref_likelihood = np.asarray([0.9, 0.9])
        child_alt_likelihood = np.asarray([0.1, 0.1])
        exact = 0.0
        for first in range(2):
            for second in range(2):
                allele = founders[first, 0]
                site_likelihood = np.where(
                    allele == 0, child_ref_likelihood, child_alt_likelihood
                )
                exact += initial[first, second] * np.prod(site_likelihood)
        marginal = result.track_alt_probability[0, 0, 0].astype(float)
        plugin = np.prod(
            (1.0 - marginal) * child_ref_likelihood
            + marginal * child_alt_likelihood
        )
        self.assertAlmostEqual(exact, 0.41, places=7)
        self.assertAlmostEqual(plugin, 0.25, places=7)
        self.assertNotAlmostEqual(exact, plugin, places=3)

    def test_same_missing_founder_is_one_shared_latent_allele(self):
        founders = np.asarray([[[-1]], [[0]], [[1]]], dtype=np.int8)
        gl = np.asarray([[[[0.0, 1.0, 0.0]]]], dtype=np.float64)
        result = infer_candidate_source_posterior(
            gl,
            founders,
            np.asarray([1]),
            0.0,
            return_state_posterior=True,
        )
        self.assertEqual(result.founder_alt_frequency[0, 0], 0.5)
        self.assertEqual(result.source_posterior[0, 0, 0, 0], 0.0)
        self.assertTrue(result.available[0])
        self.assertGreater(result.source_posterior[0, 0, 0, 1], 0.0)

    def test_high_confidence_limit_selects_matching_diplotype(self):
        founders = np.asarray(
            [
                [[0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1]],
                [[0, 1, 0, 1, 0, 1]],
            ],
            dtype=np.int8,
        )
        gl = np.tile(_gl(1e-5, 1.0, 1e-5), (1, 1, 6, 1))
        anchor = np.asarray([[[0, 1]]])
        result = _infer_debug(
            gl,
            founders,
            np.asarray([6]),
            0.0,
            eta=2.0,
            painted_track_labels=anchor,
        )
        self.assertGreater(result.source_posterior[0, 0, 0, 1], 0.999)
        np.testing.assert_allclose(
            result.track_alt_probability[0, 0, 0], 0.0, atol=2e-3
        )
        np.testing.assert_allclose(
            result.track_alt_probability[0, 0, 1], 1.0, atol=2e-3
        )

    def test_coherent_draws_recover_multisite_041_counterexample(self):
        founders = np.asarray([[[0, 0]], [[1, 1]]], dtype=np.int8)
        candidate_gl = np.asarray(
            [[[[0.5, 0.0, 0.5], [0.5, 0.0, 0.5]]]], dtype=np.float64
        )
        posterior = infer_candidate_source_posterior(
            candidate_gl,
            founders,
            np.asarray([2]),
            0.0,
            return_linked_posterior=True,
        )
        draws = sample_candidate_source_trajectories(
            posterior, n_draws=8192, seed=91
        )
        child_gl = np.asarray([[[0.9, 0.1, 0.0], [0.9, 0.1, 0.0]]])
        observed = reference_conditional_child_likelihood_mc(
            child_gl,
            founders,
            np.asarray([2]),
            first_draws=draws,
            external_second_alt_probability=np.zeros((1, 2)),
            mismatch_probability=0.0,
            track_switch_probability=0.0,
        )
        likelihood_without_factor_three = np.exp(observed.log_likelihood) / 9.0
        self.assertAlmostEqual(likelihood_without_factor_three, 0.41, delta=0.02)
        self.assertLess(observed.log_likelihood_standard_error, 0.02)
        self.assertLess(abs(observed.doubling_delta_log_likelihood), 0.02)

    def test_trajectory_draws_are_track_and_founder_order_equivariant(self):
        founders = np.asarray(
            [[[0], [1]], [[1], [0]], [[0], [0]]], dtype=np.int8
        )
        gl = np.asarray([[[[0.1, 0.8, 0.1]], [[0.2, 0.7, 0.1]]]])
        anchor = np.full((1, 2, 2), -1, dtype=np.int64)
        anchor[0, 0] = (0, 1)
        direct = infer_candidate_source_posterior(
            gl,
            founders,
            np.asarray([1, 1]),
            0.1,
            painted_track_labels=anchor,
            return_linked_posterior=True,
        )
        direct_draws = sample_candidate_source_trajectories(
            direct, n_draws=512, seed=3
        )
        swapped = infer_candidate_source_posterior(
            gl,
            founders,
            np.asarray([1, 1]),
            0.1,
            painted_track_labels=anchor[..., ::-1].copy(),
            return_linked_posterior=True,
        )
        swapped_draws = sample_candidate_source_trajectories(
            swapped, n_draws=512, seed=3
        )
        np.testing.assert_array_equal(
            direct_draws.founder_tracks,
            swapped_draws.founder_tracks[..., ::-1],
        )

        permutation = np.asarray([2, 0, 1])
        inverse = np.argsort(permutation)
        permuted_anchor = anchor.copy()
        for old_id in range(3):
            permuted_anchor[anchor == old_id] = inverse[old_id]
        permuted = infer_candidate_source_posterior(
            gl,
            founders[permutation],
            np.asarray([1, 1]),
            0.1,
            painted_track_labels=permuted_anchor,
            return_linked_posterior=True,
        )
        permuted_draws = sample_candidate_source_trajectories(
            permuted, n_draws=512, seed=3
        )
        np.testing.assert_array_equal(
            direct_draws.founder_tracks,
            permutation[permuted_draws.founder_tracks],
        )

    def test_unavailable_draws_reduce_m1_and_m2_canonically(self):
        founders = np.asarray([[[0]], [[1]]], dtype=np.int8)
        gl = np.asarray(
            [
                [[[1.0, 1.0, 1.0]]],
                [[[0.05, 0.9, 0.05]]],
            ],
            dtype=np.float64,
        )
        posterior = infer_candidate_source_posterior(
            gl,
            founders,
            np.asarray([1]),
            0.0,
            return_linked_posterior=True,
        )
        draws = sample_candidate_source_trajectories(
            posterior, n_draws=1024, seed=12
        )
        child = np.asarray([[[0.2, 0.7, 0.1]]])
        external_first = np.full((1, 1), 0.2)
        external_second = np.full((1, 1), 0.7)
        common = dict(
            child_genotype_likelihoods=child,
            founder_alleles=founders,
            selected_markers_per_bin=np.asarray([1]),
            external_first_alt_probability=external_first,
            external_second_alt_probability=external_second,
        )
        m0 = reference_conditional_child_likelihood_mc(**common)
        m1_unavailable = reference_conditional_child_likelihood_mc(
            **common, first_draws=draws, first_sample=0
        )
        m2_unavailable = reference_conditional_child_likelihood_mc(
            **common,
            first_draws=draws,
            first_sample=0,
            second_draws=draws,
            second_sample=0,
        )
        self.assertEqual(m0.log_likelihood, m1_unavailable.log_likelihood)
        self.assertEqual(m0.log_likelihood, m2_unavailable.log_likelihood)

        one_available = reference_conditional_child_likelihood_mc(
            **common, second_draws=draws, second_sample=1
        )
        one_of_two_available = reference_conditional_child_likelihood_mc(
            **common,
            first_draws=draws,
            first_sample=0,
            second_draws=draws,
            second_sample=1,
        )
        self.assertEqual(
            one_available.log_likelihood,
            one_of_two_available.log_likelihood,
        )
        self.assertEqual(one_of_two_available.n_available_parents, 1)

    def test_shared_missing_founder_draw_cannot_create_heterozygote(self):
        founders = np.asarray([[[-1]], [[0]], [[1]]], dtype=np.int8)
        tracks = np.zeros((2, 256, 1, 2), dtype=np.int16)
        draws = CandidateSourceTrajectoryDraws(
            founder_tracks=tracks,
            available=np.asarray([True, True]),
            n_draws=256,
            seed=44,
            canonical_founder_order=np.asarray([0, 1, 2]),
        )
        observed = reference_conditional_child_likelihood_mc(
            np.asarray([[[0.0, 1.0, 0.0]]]),
            founders,
            np.asarray([1]),
            first_draws=draws,
            first_sample=0,
            second_draws=draws,
            second_sample=1,
            mismatch_probability=0.0,
        )
        self.assertLess(observed.log_likelihood, -500.0)
        self.assertGreater(observed.missing_founder_draws_used, 0)

    def test_uniform_child_is_neutral_for_coherent_draws(self):
        founders = np.asarray([[[0], [0]], [[1], [1]]], dtype=np.int8)
        candidate = np.asarray(
            [[[[0.1, 0.8, 0.1]], [[0.2, 0.7, 0.1]]]]
        )
        source = infer_candidate_source_posterior(
            candidate,
            founders,
            np.asarray([1, 1]),
            0.1,
            return_linked_posterior=True,
        )
        draws = sample_candidate_source_trajectories(
            source, n_draws=1024, seed=77
        )
        observed = reference_conditional_child_likelihood_mc(
            np.ones((2, 1, 3)),
            founders,
            np.asarray([1, 1]),
            first_draws=draws,
            mismatch_probability=0.01,
        )
        self.assertAlmostEqual(observed.log_likelihood, 0.0, places=14)
        self.assertAlmostEqual(
            observed.log_likelihood_standard_error, 0.0, places=14
        )

    def test_m1_coherent_mc_converges_to_f2_b3_enumeration(self):
        founders = np.asarray(
            [[[0], [0], [1]], [[1], [1], [0]]], dtype=np.int8
        )
        candidate_gl = np.asarray(
            [[
                [[0.2, 0.7, 0.1]],
                [[0.6, 0.3, 0.1]],
                [[0.1, 0.8, 0.1]],
            ]]
        )
        source = infer_candidate_source_posterior(
            candidate_gl,
            founders,
            np.ones(3, dtype=np.int64),
            np.asarray([0.13, 0.21]),
            return_linked_posterior=True,
        )
        child = np.asarray(
            [
                [[0.7, 0.25, 0.05]],
                [[0.25, 0.65, 0.1]],
                [[0.1, 0.75, 0.15]],
            ]
        )
        exact = _exact_called_child_log_likelihood(
            child, founders, source, track_switch_probability=0.17
        )
        draws = sample_candidate_source_trajectories(
            source, n_draws=8192, seed=2026
        )
        observed = reference_conditional_child_likelihood_mc(
            child,
            founders,
            np.ones(3, dtype=np.int64),
            first_draws=draws,
            external_second_alt_probability=np.zeros((3, 1)),
            mismatch_probability=0.0,
            track_switch_probability=0.17,
        )
        self.assertLess(
            abs(observed.log_likelihood - exact),
            max(0.035, 4.0 * observed.log_likelihood_standard_error),
        )
        self.assertLess(abs(observed.doubling_delta_log_likelihood), 0.04)

    def test_m2_common_draws_converge_and_are_parent_swap_invariant(self):
        founders = np.asarray([[[0], [1]], [[1], [0]]], dtype=np.int8)
        candidate_gl = np.asarray(
            [
                [[[0.15, 0.75, 0.1]], [[0.65, 0.3, 0.05]]],
                [[[0.7, 0.25, 0.05]], [[0.1, 0.8, 0.1]]],
            ]
        )
        source = infer_candidate_source_posterior(
            candidate_gl,
            founders,
            np.ones(2, dtype=np.int64),
            0.19,
            return_linked_posterior=True,
        )
        child = np.asarray(
            [[[0.55, 0.4, 0.05]], [[0.1, 0.7, 0.2]]]
        )
        exact = _exact_called_child_log_likelihood(
            child,
            founders,
            source,
            second_source=source,
            track_switch_probability=0.11,
        )
        draws = sample_candidate_source_trajectories(
            source, n_draws=16384, seed=808
        )
        direct = reference_conditional_child_likelihood_mc(
            child,
            founders,
            np.ones(2, dtype=np.int64),
            first_draws=draws,
            first_sample=0,
            second_draws=draws,
            second_sample=1,
            mismatch_probability=0.0,
            track_switch_probability=0.11,
        )
        swapped = reference_conditional_child_likelihood_mc(
            child,
            founders,
            np.ones(2, dtype=np.int64),
            first_draws=draws,
            first_sample=1,
            second_draws=draws,
            second_sample=0,
            mismatch_probability=0.0,
            track_switch_probability=0.11,
        )
        self.assertAlmostEqual(
            direct.log_likelihood, swapped.log_likelihood, places=13
        )
        self.assertLess(
            abs(direct.log_likelihood - exact),
            max(0.04, 4.0 * direct.log_likelihood_standard_error),
        )
        self.assertLess(abs(direct.doubling_delta_log_likelihood), 0.05)



    def test_compound_axis_update_matches_dense_reference(self):
        rng = np.random.default_rng(991)
        for n_founders in (2, 3, 5):
            transition = source_model._transition_matrix(n_founders, 0.17)
            values = rng.random((n_founders,) * 4)
            for axis in range(4):
                observed = source_model._compound_transition_axis(
                    values, transition, axis
                )
                expected = source_model._dense_transition_axis(
                    values, transition, axis
                )
                np.testing.assert_allclose(
                    observed, expected, rtol=2e-15, atol=2e-15
                )

    def test_exact_lumped_m1_matches_full_f2_b3_enumeration(self):
        founders = np.asarray(
            [[[0], [0], [1]], [[1], [1], [0]]], dtype=np.int8
        )
        candidate = np.asarray(
            [[
                [[0.2, 0.7, 0.1]],
                [[0.6, 0.3, 0.1]],
                [[0.1, 0.8, 0.1]],
            ]]
        )
        source = infer_candidate_source_posterior(
            candidate,
            founders,
            np.ones(3, dtype=np.int64),
            np.asarray([0.13, 0.21]),
            return_lumped_posterior=True,
            posterior_factor_dtype=np.float64,
        )
        child = np.asarray(
            [
                [[0.7, 0.25, 0.05]],
                [[0.25, 0.65, 0.1]],
                [[0.1, 0.75, 0.15]],
            ]
        )
        expected = _exact_full_source_track_likelihood(
            child, founders, source, [0]
        )
        observed = score_conditional_child_tensor_exact(
            child,
            founders,
            np.ones(3, dtype=np.int64),
            source,
            first_sample=0,
            mismatch_probability=0.0,
            track_switch_probability=0.13,
        )
        self.assertEqual(observed.mode, "m1")
        self.assertEqual(observed.hidden_state_count, 8)
        self.assertAlmostEqual(observed.log_likelihood, expected, places=13)

    def test_exact_lumped_m2_matches_full_f2_b2_enumeration(self):
        founders = np.asarray([[[0], [1]], [[1], [0]]], dtype=np.int8)
        candidate = np.asarray(
            [
                [[[0.15, 0.75, 0.1]], [[0.65, 0.3, 0.05]]],
                [[[0.7, 0.25, 0.05]], [[0.1, 0.8, 0.1]]],
            ]
        )
        source = infer_candidate_source_posterior(
            candidate,
            founders,
            np.ones(2, dtype=np.int64),
            0.19,
            return_lumped_posterior=True,
            posterior_factor_dtype=np.float64,
        )
        child = np.asarray(
            [[[0.55, 0.4, 0.05]], [[0.1, 0.7, 0.2]]]
        )
        expected = _exact_full_source_track_likelihood(
            child, founders, source, [0, 1]
        )
        observed = score_conditional_child_tensor_exact(
            child,
            founders,
            np.ones(2, dtype=np.int64),
            source,
            first_sample=0,
            second_sample=1,
            mismatch_probability=0.0,
            track_switch_probability=0.13,
        )
        self.assertEqual(observed.mode, "m2")
        self.assertEqual(observed.hidden_state_count, 16)
        self.assertAlmostEqual(observed.log_likelihood, expected, places=13)

    def test_compiled_m2_kernel_matches_exact_tensor_scorer(self):
        founders = np.asarray([[[0], [1]], [[1], [0]]], dtype=np.int8)
        candidate = np.asarray(
            [
                [[[0.15, 0.75, 0.1]], [[0.65, 0.3, 0.05]]],
                [[[0.7, 0.25, 0.05]], [[0.1, 0.8, 0.1]]],
            ]
        )
        source = infer_candidate_source_posterior(
            candidate,
            founders,
            np.ones(2, dtype=np.int64),
            0.19,
            return_lumped_posterior=True,
            posterior_factor_dtype=np.float64,
        )
        child = np.asarray(
            [[[0.55, 0.4, 0.05]], [[0.1, 0.7, 0.2]]]
        )
        reference = score_conditional_child_tensor_exact(
            child,
            founders,
            np.ones(2, dtype=np.int64),
            source,
            first_sample=0,
            second_sample=1,
            mismatch_probability=0.0,
            track_switch_probability=0.13,
        )
        initial = []
        for sample in (0, 1):
            value = np.exp(source.lumped_initial_log_probability[sample])
            value /= value.sum()
            initial.append(0.5 * (value + value.T))
        initial_pair = (
            initial[0][:, :, None, None]
            * initial[1][None, None, :, :]
        )[None]
        transition = source.lumped_transition_probability.astype(float)
        right = np.empty((1, 1, 2, 2, 2))
        denominator = np.empty_like(right)
        for parent, sample in enumerate((0, 1)):
            right[0, 0, parent] = np.exp(
                source.lumped_next_log_weight[sample, 0]
            )
            denominator[0, 0, parent] = (
                transition[0] @ right[0, 0, parent] @ transition[0].T
            )
        frequency = source_model._reference_founder_frequency(founders)
        emission = np.stack([
            source_model._founder_pair_bin_emission(
                child, founders, frequency, 1, block, 0.0, 1.0
            )
            for block in range(2)
        ])
        compiled = source_model._m2_compound_forward_kernel(
            initial_pair,
            right,
            denominator,
            np.asarray([transition[0, 0, 0]]),
            np.asarray([transition[0, 0, 1]]),
            np.asarray([0.13]),
            emission,
        )
        self.assertAlmostEqual(compiled[0], reference.log_likelihood, places=13)

    def test_exact_uniform_child_neutral_and_unavailable_reductions(self):
        founders = np.asarray([[[0], [0]], [[1], [1]]], dtype=np.int8)
        candidate = np.asarray(
            [
                [[[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]],
                [[[0.1, 0.8, 0.1]], [[0.2, 0.7, 0.1]]],
            ]
        )
        source = infer_candidate_source_posterior(
            candidate,
            founders,
            np.ones(2, dtype=np.int64),
            0.1,
            return_lumped_posterior=True,
        )
        uniform_child = np.ones((2, 1, 3))
        for first_sample, second_sample in (
            (None, None),
            (1, None),
            (0, None),
            (0, 0),
            (0, 1),
        ):
            result = score_conditional_child_tensor_exact(
                uniform_child,
                founders,
                np.ones(2, dtype=np.int64),
                source,
                first_sample=first_sample,
                second_sample=second_sample,
            )
            self.assertAlmostEqual(result.log_likelihood, 0.0, places=13)
        child = np.asarray([[[0.7, 0.2, 0.1]], [[0.1, 0.8, 0.1]]])
        m0 = score_conditional_child_tensor_exact(
            child, founders, np.ones(2, dtype=np.int64), source
        )
        unavailable_m1 = score_conditional_child_tensor_exact(
            child,
            founders,
            np.ones(2, dtype=np.int64),
            source,
            first_sample=0,
        )
        unavailable_m2 = score_conditional_child_tensor_exact(
            child,
            founders,
            np.ones(2, dtype=np.int64),
            source,
            first_sample=0,
            second_sample=0,
        )
        available_m1 = score_conditional_child_tensor_exact(
            child,
            founders,
            np.ones(2, dtype=np.int64),
            source,
            first_sample=1,
        )
        one_available_m2 = score_conditional_child_tensor_exact(
            child,
            founders,
            np.ones(2, dtype=np.int64),
            source,
            first_sample=0,
            second_sample=1,
        )
        self.assertEqual(m0.log_likelihood, unavailable_m1.log_likelihood)
        self.assertEqual(m0.log_likelihood, unavailable_m2.log_likelihood)
        self.assertEqual(
            available_m1.log_likelihood, one_available_m2.log_likelihood
        )

    def test_exact_parent_track_and_founder_invariance(self):
        founders = np.asarray(
            [[[0], [1]], [[1], [0]], [[0], [0]]], dtype=np.int8
        )
        candidate = np.asarray(
            [
                [[[0.15, 0.75, 0.1]], [[0.65, 0.3, 0.05]]],
                [[[0.7, 0.25, 0.05]], [[0.1, 0.8, 0.1]]],
            ]
        )
        anchor = np.full((2, 2, 2), -1, dtype=np.int64)
        anchor[:, 0] = (0, 1)
        source = infer_candidate_source_posterior(
            candidate,
            founders,
            np.ones(2, dtype=np.int64),
            0.16,
            painted_track_labels=anchor,
            return_lumped_posterior=True,
        )
        swapped_gauge = infer_candidate_source_posterior(
            candidate,
            founders,
            np.ones(2, dtype=np.int64),
            0.16,
            painted_track_labels=anchor[..., ::-1].copy(),
            return_lumped_posterior=True,
        )
        np.testing.assert_array_equal(
            source.lumped_initial_log_probability,
            swapped_gauge.lumped_initial_log_probability,
        )
        child = np.asarray([[[0.5, 0.4, 0.1]], [[0.1, 0.7, 0.2]]])
        direct = score_conditional_child_tensor_exact(
            child,
            founders,
            np.ones(2, dtype=np.int64),
            source,
            first_sample=0,
            second_sample=1,
        )
        parent_swapped = score_conditional_child_tensor_exact(
            child,
            founders,
            np.ones(2, dtype=np.int64),
            source,
            first_sample=1,
            second_sample=0,
        )
        self.assertAlmostEqual(
            direct.log_likelihood, parent_swapped.log_likelihood, places=13
        )
        permutation = np.asarray([2, 0, 1])
        inverse = np.argsort(permutation)
        permuted_anchor = anchor.copy()
        for old in range(3):
            permuted_anchor[anchor == old] = inverse[old]
        permuted = infer_candidate_source_posterior(
            candidate,
            founders[permutation],
            np.ones(2, dtype=np.int64),
            0.16,
            painted_track_labels=permuted_anchor,
            return_lumped_posterior=True,
        )
        reordered = score_conditional_child_tensor_exact(
            child,
            founders[permutation],
            np.ones(2, dtype=np.int64),
            permuted,
            first_sample=0,
            second_sample=1,
        )
        self.assertAlmostEqual(
            direct.log_likelihood, reordered.log_likelihood, places=12
        )

    def test_exact_point_mass_recovers_hard_transmission_limit(self):
        founders = np.asarray([[[0]], [[1]]], dtype=np.int8)
        candidate = np.asarray([[[[0.05, 0.9, 0.05]]]])
        source = infer_candidate_source_posterior(
            candidate,
            founders,
            np.asarray([1]),
            0.0,
            return_lumped_posterior=True,
            posterior_factor_dtype=np.float64,
        )
        point = np.full((1, 2, 2), -np.inf)
        point[0, 0, 1] = 0.0
        source = replace(source, lumped_initial_log_probability=point)
        child = np.asarray([[[0.7, 0.25, 0.05]]])
        observed = score_conditional_child_tensor_exact(
            child,
            founders,
            np.asarray([1]),
            source,
            first_sample=0,
            mismatch_probability=0.0,
            external_initial_probability=np.asarray([1.0, 0.0]),
        )
        expected = np.log(3.0 * 0.5 * (child[0, 0, 0] + child[0, 0, 1]))
        self.assertAlmostEqual(observed.log_likelihood, expected, places=14)

    def test_exact_missing_founder_site_is_excluded_for_fallback(self):
        founders = np.asarray([[[-1]], [[0]], [[1]]], dtype=np.int8)
        candidate = np.asarray(
            [[[[0.08716244, 0.06217411, 0.85066344]]]]
        )
        source = infer_candidate_source_posterior(
            candidate,
            founders,
            np.asarray([1]),
            0.0,
            return_lumped_posterior=True,
            posterior_factor_dtype=np.float64,
        )
        self.assertFalse(source.lumped_site_available[0, 0])
        self.assertFalse(source.lumped_available[0])
        self.assertEqual(source.lumped_informative_site_count[0], 0)
        observed = score_conditional_child_tensor_exact(
            np.asarray([[[0.47114697, 0.09299659, 0.43585644]]]),
            founders,
            np.asarray([1]),
            source,
            first_sample=0,
            mismatch_probability=0.0,
            external_initial_probability=np.asarray([0.0, 1.0, 0.0]),
        )
        self.assertEqual(observed.mode, "m0")
        self.assertEqual(observed.excluded_marker_count, 1)
        self.assertEqual(observed.log_likelihood, 0.0)

    def test_float32_lumped_factors_match_float64_reference(self):
        rng = np.random.default_rng(181)
        founders = rng.integers(0, 2, size=(4, 4, 2), dtype=np.int8)
        candidate = rng.random((2, 4, 2, 3))
        child = rng.random((4, 2, 3))
        common = dict(
            genotype_likelihoods=candidate,
            founder_alleles=founders,
            selected_markers_per_bin=np.full(4, 2),
            switch_probability=np.asarray([0.04, 0.13, 0.21]),
            return_lumped_posterior=True,
        )
        source32 = infer_candidate_source_posterior(**common)
        source64 = infer_candidate_source_posterior(
            **common, posterior_factor_dtype=np.float64
        )
        score32 = score_conditional_child_tensor_exact(
            child,
            founders,
            np.full(4, 2),
            source32,
            first_sample=0,
            second_sample=1,
        )
        score64 = score_conditional_child_tensor_exact(
            child,
            founders,
            np.full(4, 2),
            source64,
            first_sample=0,
            second_sample=1,
        )
        self.assertLess(abs(score32.log_likelihood - score64.log_likelihood), 2e-6)

    def test_physical_lumped_factors_use_nonzero_informative_root(self):
        founders = np.asarray(
            [[[0], [0], [1]], [[1], [1], [0]]], dtype=np.int8
        )
        gl = np.asarray(
            [[
                [[1.0, 1.0, 1.0]],
                [[0.68, 0.27, 0.05]],
                [[0.08, 0.74, 0.18]],
            ]]
        )
        source = infer_candidate_source_posterior(
            gl,
            founders,
            np.ones(3, dtype=np.int64),
            np.asarray([0.23, 0.11]),
            return_linked_posterior=True,
            return_lumped_posterior=True,
            posterior_factor_dtype=np.float64,
        )
        self.assertEqual(source.gauge_anchor_bin[0], 1)
        gauge_paths, gauge_probability = _linked_path_distribution(source, 0)
        physical_paths, physical_probability = _lumped_path_distribution(
            source, 0
        )
        gauge = dict(zip(gauge_paths, gauge_probability))
        expected = []
        for path in physical_paths:
            swapped = tuple((state[1], state[0]) for state in path)
            expected.append(0.5 * (gauge[path] + gauge[swapped]))
        np.testing.assert_allclose(
            physical_probability, expected, rtol=2e-15, atol=2e-15
        )

    def test_exact_zero_support_rho_zero_and_one(self):
        founders = np.asarray([[[0], [0]], [[1], [1]]], dtype=np.int8)
        child = np.asarray([[[0.8, 0.15, 0.05]], [[0.7, 0.25, 0.05]]])
        for rho in (0.0, 1.0):
            candidate = np.asarray(
                [[
                    [[1.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0]] if rho == 0.0 else [[0.0, 0.0, 1.0]],
                ]]
            )
            source = infer_candidate_source_posterior(
                candidate,
                founders,
                np.ones(2, dtype=np.int64),
                rho,
                return_lumped_posterior=True,
                posterior_factor_dtype=np.float64,
            )
            observed = score_conditional_child_tensor_exact(
                child,
                founders,
                np.ones(2, dtype=np.int64),
                source,
                first_sample=0,
                mismatch_probability=0.01,
                track_switch_probability=0.0,
                external_initial_probability=np.asarray([1.0, 0.0]),
                external_transition_probability=np.eye(2)[None, :, :],
            )
            expected = 0.0
            for block in range(2):
                source_alt = 0.01 if block == 0 or rho == 0.0 else 0.99
                q = _genotype_distribution(source_alt, 0.01)
                expected += np.log(3.0 * np.dot(child[block, 0], q))
            self.assertTrue(np.isfinite(observed.log_likelihood))
            self.assertAlmostEqual(observed.log_likelihood, expected, places=14)

    def test_excluded_site_gl_cannot_move_exact_physical_root_or_score(self):
        founders = np.asarray(
            [
                [[-1], [0], [1]],
                [[0], [1], [0]],
                [[1], [0], [0]],
            ],
            dtype=np.int8,
        )
        common_gl = np.asarray(
            [
                [[0.18, 0.72, 0.10]],
                [[0.62, 0.28, 0.10]],
                [[0.08, 0.74, 0.18]],
            ]
        )
        first_gl = common_gl.copy()
        first_gl[0] = 1.0 / 3.0
        second_gl = common_gl.copy()
        second_gl[0] = np.asarray([0.91, 0.07, 0.02])
        first = infer_candidate_source_posterior(
            first_gl[None],
            founders,
            np.ones(3, dtype=np.int64),
            np.asarray([0.23, 0.11]),
            return_lumped_posterior=True,
            posterior_factor_dtype=np.float64,
        )
        second = infer_candidate_source_posterior(
            second_gl[None],
            founders,
            np.ones(3, dtype=np.int64),
            np.asarray([0.23, 0.11]),
            return_lumped_posterior=True,
            posterior_factor_dtype=np.float64,
        )
        self.assertEqual(first.gauge_anchor_bin[0], 1)
        self.assertEqual(second.gauge_anchor_bin[0], 0)
        self.assertEqual(first.lumped_anchor_bin[0], 1)
        self.assertEqual(second.lumped_anchor_bin[0], 1)
        np.testing.assert_array_equal(
            first.lumped_site_available, second.lumped_site_available
        )
        np.testing.assert_allclose(
            first.lumped_initial_log_probability,
            second.lumped_initial_log_probability,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            first.lumped_next_log_weight,
            second.lumped_next_log_weight,
            rtol=0.0,
            atol=0.0,
        )
        child = np.asarray(
            [
                [[0.51, 0.39, 0.10]],
                [[0.16, 0.72, 0.12]],
                [[0.63, 0.27, 0.10]],
            ]
        )
        first_score = score_conditional_child_tensor_exact(
            child,
            founders,
            np.ones(3, dtype=np.int64),
            first,
            first_sample=0,
        )
        second_score = score_conditional_child_tensor_exact(
            child,
            founders,
            np.ones(3, dtype=np.int64),
            second,
            first_sample=0,
        )
        self.assertEqual(first_score.excluded_marker_count, 1)
        self.assertEqual(first_score.log_likelihood, second_score.log_likelihood)

    def test_batch_scores_match_per_row_exact_reference(self):
        rng = np.random.default_rng(717)
        founders = np.asarray(
            [[[0], [0], [1]], [[1], [1], [0]]], dtype=np.int8
        )
        n_children = n_candidates = 3
        candidate_gl = rng.random((n_candidates, 3, 1, 3))
        source = infer_candidate_source_posterior(
            candidate_gl,
            founders,
            np.ones(3, dtype=np.int64),
            np.asarray([0.1, 0.2]),
            return_lumped_posterior=True,
            posterior_factor_dtype=np.float64,
        )
        child_gl = rng.random((n_children, 3, 1, 3))
        exponent = rng.uniform(0.5, 1.0, size=(n_children, 3))
        track_switch = np.full((n_candidates, 2), 0.13)
        external_initial = rng.random((n_children, 2))
        external_transition = rng.random((n_children, 2, 2, 2))
        trios = np.asarray(
            [[0, 0, 1], [0, 1, 0], [1, 1, 2], [2, 0, 2]],
            dtype=np.int64,
        )
        batch = score_candidate_source_batch_exact(
            source,
            child_gl,
            founders,
            np.ones(3, dtype=np.int64),
            exponent,
            track_switch,
            external_initial,
            external_transition,
            trios,
            mismatch_probability=0.03,
        )
        for child in range(n_children):
            zero = score_conditional_child_tensor_exact(
                child_gl[child],
                founders,
                np.ones(3, dtype=np.int64),
                source,
                mismatch_probability=0.03,
                track_switch_probability=0.13,
                eta=exponent[child],
                external_initial_probability=external_initial[child],
                external_transition_probability=external_transition[child],
            )
            self.assertAlmostEqual(
                batch.zero_observed[child], zero.log_likelihood, places=13
            )
            for parent in range(n_candidates):
                one = score_conditional_child_tensor_exact(
                    child_gl[child],
                    founders,
                    np.ones(3, dtype=np.int64),
                    source,
                    first_sample=parent,
                    mismatch_probability=0.03,
                    track_switch_probability=0.13,
                    eta=exponent[child],
                    external_initial_probability=external_initial[child],
                    external_transition_probability=external_transition[child],
                )
                self.assertAlmostEqual(
                    batch.one_observed[child, parent],
                    one.log_likelihood,
                    places=13,
                )
        for row, (child, first, second) in enumerate(trios):
            two = score_conditional_child_tensor_exact(
                child_gl[child],
                founders,
                np.ones(3, dtype=np.int64),
                source,
                first_sample=int(first),
                second_sample=int(second),
                mismatch_probability=0.03,
                track_switch_probability=0.13,
                eta=exponent[child],
                external_initial_probability=external_initial[child],
                external_transition_probability=external_transition[child],
            )
            self.assertAlmostEqual(
                batch.two_observed[row], two.log_likelihood, places=12
            )
        self.assertEqual(batch.two_observed[0], batch.two_observed[1])
        self.assertEqual(batch.complete_founder_marker_count, 3)
        self.assertEqual(batch.excluded_founder_marker_count, 0)
        self.assertTrue(np.all(batch.one_parent_identity_information > 0.0))
        self.assertTrue(np.all(batch.two_parent_edge_information > 0.0))

    def test_batch_unavailable_reductions_and_excluded_counts(self):
        founders = np.asarray(
            [
                [[-1], [0]],
                [[0], [1]],
                [[1], [0]],
            ],
            dtype=np.int8,
        )
        candidate_gl = np.asarray(
            [
                [[[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]],
                [[[0.1, 0.8, 0.1]], [[0.2, 0.7, 0.1]]],
            ]
        )
        source = infer_candidate_source_posterior(
            candidate_gl,
            founders,
            np.ones(2, dtype=np.int64),
            0.1,
            return_lumped_posterior=True,
        )
        child = np.asarray(
            [[[[0.3, 0.6, 0.1]], [[0.1, 0.8, 0.1]]]]
        )
        trios = np.asarray([[0, 0, 1]], dtype=np.int64)
        batch = score_candidate_source_batch_exact(
            source,
            child,
            founders,
            np.ones(2, dtype=np.int64),
            np.ones((1, 2)),
            np.full((2, 1), 0.1),
            np.full((1, 3), 1.0 / 3.0),
            np.broadcast_to(np.eye(3), (1, 1, 3, 3)).copy(),
            trios,
        )
        self.assertEqual(
            batch.one_observed[0, 0], batch.zero_observed[0]
        )
        self.assertEqual(
            batch.two_observed[0], batch.one_observed[0, 1]
        )
        self.assertEqual(batch.excluded_founder_marker_count, 1)
        self.assertEqual(batch.complete_founder_marker_count, 1)
        np.testing.assert_array_equal(
            batch.excluded_founder_marker_count_per_child, [1]
        )
        self.assertEqual(batch.one_parent_identity_information[0, 0], 0.0)
        self.assertGreater(batch.one_parent_identity_information[0, 1], 0.0)

    def test_batch_point_mass_diagnostic_matches_hard_limit_seam(self):
        founders = np.asarray([[[0]], [[1]]], dtype=np.int8)
        candidate = np.asarray([[[[0.05, 0.9, 0.05]]]])
        source = infer_candidate_source_posterior(
            candidate,
            founders,
            np.asarray([1]),
            0.0,
            return_lumped_posterior=True,
            posterior_factor_dtype=np.float64,
        )
        point = np.full((1, 2, 2), -np.inf)
        point[0, 0, 1] = 0.0
        source = replace(source, lumped_initial_log_probability=point)
        child = np.asarray([[[[0.7, 0.25, 0.05]]]])
        batch = score_candidate_source_batch_exact(
            source,
            child,
            founders,
            np.asarray([1]),
            np.ones((1, 1)),
            np.empty((1, 0)),
            np.asarray([[1.0, 0.0]]),
            np.empty((1, 0, 2, 2)),
            np.empty((0, 3), dtype=np.int64),
            mismatch_probability=0.0,
        )
        expected = np.log(3.0 * 0.5 * (0.7 + 0.25))
        self.assertTrue(batch.candidate_initial_point_mass[0])
        self.assertEqual(batch.candidate_initial_max_probability[0], 1.0)
        self.assertAlmostEqual(batch.one_observed[0, 0], expected, places=14)
    def test_ordered_independent_lumped_root_and_default_b4_unchanged(self):
        founders = np.asarray([[[0]], [[1]], [[0]]], dtype=np.int8)
        epsilon = 1e-12
        candidate = np.asarray(
            [[[[1.0 / 3.0 + epsilon, 1.0 / 3.0, 1.0 / 3.0 - epsilon]]]]
        )
        common = dict(
            genotype_likelihoods=candidate,
            founder_alleles=founders,
            selected_markers_per_bin=np.asarray([1]),
            switch_probability=0.06,
            uniform_tolerance=0.0,
            return_lumped_posterior=True,
            posterior_factor_dtype=np.float64,
        )
        default = infer_candidate_source_posterior(**common)
        explicit = infer_candidate_source_posterior(
            **common, lumped_root_prior_mode="uniform_unordered"
        )
        ordered = infer_candidate_source_posterior(
            **common, lumped_root_prior_mode="ordered_independent_uniform"
        )
        np.testing.assert_array_equal(
            default.lumped_initial_log_probability,
            explicit.lumped_initial_log_probability,
        )
        self.assertEqual(default.lumped_root_prior_mode, "uniform_unordered")
        self.assertEqual(
            ordered.lumped_root_prior_mode, "ordered_independent_uniform"
        )
        self.assertAlmostEqual(
            float(np.trace(np.exp(default.lumped_initial_log_probability[0]))),
            0.5,
            places=10,
        )
        self.assertAlmostEqual(
            float(np.trace(np.exp(ordered.lumped_initial_log_probability[0]))),
            1.0 / 3.0,
            places=10,
        )

    def test_matched_null_exact_nesting_and_epsilon_continuity(self):
        rng = np.random.default_rng(501)
        founders = np.asarray(
            [[[0], [0], [1]], [[1], [1], [0]]], dtype=np.int8
        )
        source = infer_candidate_source_posterior(
            rng.random((3, 3, 1, 3)),
            founders,
            np.ones(3, dtype=np.int64),
            0.06,
            return_lumped_posterior=True,
            lumped_root_prior_mode="ordered_independent_uniform",
            posterior_factor_dtype=np.float64,
        )
        null_source = replace(
            source,
            lumped_initial_log_probability=np.full(
                (3, 2, 2), np.log(0.25)
            ),
            lumped_next_log_weight=np.zeros((3, 2, 2, 2)),
            lumped_available=np.ones(3, dtype=np.bool_),
        )
        child = rng.random((2, 3, 1, 3))
        trios = np.asarray(
            ((0, 0, 1), (0, 1, 2), (1, 0, 2)), dtype=np.int64
        )
        nested = score_candidate_source_batch_matched_null_exact(
            null_source,
            child,
            founders,
            np.ones(3, dtype=np.int64),
            np.ones((2, 3)),
            0.01960528042383841,
            trios,
        )
        np.testing.assert_allclose(
            nested.one_observed,
            np.broadcast_to(
                nested.zero_observed[:, None], nested.one_observed.shape
            ),
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            nested.two_observed,
            nested.zero_observed[trios[:, 0]],
            rtol=0.0,
            atol=1e-14,
        )
        self.assertEqual(nested.null_parent_count, 2)
        self.assertEqual(nested.matched_pair_evaluation_count, 11)

        small_founders = np.asarray(
            [[[0], [1]], [[1], [0]]], dtype=np.int8
        )
        small_child = np.asarray(
            [[[[0.71, 0.24, 0.05]], [[0.08, 0.76, 0.16]]]]
        )
        gains = []
        availability = []
        for epsilon in (0.0, 2e-12, 1e-9):
            candidate = np.full((1, 2, 1, 3), 1.0 / 3.0)
            candidate[0, 0, 0] = (
                1.0 / 3.0 + epsilon,
                1.0 / 3.0,
                1.0 / 3.0 - epsilon,
            )
            epsilon_source = infer_candidate_source_posterior(
                candidate,
                small_founders,
                np.ones(2, dtype=np.int64),
                0.06,
                return_lumped_posterior=True,
                lumped_root_prior_mode="ordered_independent_uniform",
                posterior_factor_dtype=np.float64,
            )
            observed = score_candidate_source_batch_matched_null_exact(
                epsilon_source,
                small_child,
                small_founders,
                np.ones(2, dtype=np.int64),
                np.ones((1, 2)),
                0.02,
                np.empty((0, 3), dtype=np.int64),
            )
            availability.append(bool(epsilon_source.lumped_available[0]))
            gains.append(
                float(observed.one_observed[0, 0] - observed.zero_observed[0])
            )
        self.assertEqual(availability, [False, True, True])
        self.assertEqual(gains[0], 0.0)
        self.assertLess(abs(gains[1]), 1e-10)
        self.assertLess(abs(gains[2]), 1e-7)

    def test_matched_null_enumerates_rho_and_transmission_extremes(self):
        founders = np.asarray(
            [[[0], [1]], [[1], [0]]], dtype=np.int8
        )
        child = np.asarray(
            [[[[0.63, 0.31, 0.06]], [[0.09, 0.74, 0.17]]]]
        )
        candidate = np.full((2, 2, 1, 3), 1.0 / 3.0)
        trios = np.asarray(((0, 0, 1),), dtype=np.int64)
        for rho in (0.01960528042383841, 0.06):
            source = infer_candidate_source_posterior(
                candidate,
                founders,
                np.ones(2, dtype=np.int64),
                rho,
                return_lumped_posterior=True,
                lumped_root_prior_mode="ordered_independent_uniform",
                posterior_factor_dtype=np.float64,
            )
            for theta in (0.0, 0.5):
                observed = score_candidate_source_batch_matched_null_exact(
                    source,
                    child,
                    founders,
                    np.ones(2, dtype=np.int64),
                    np.ones((1, 2)),
                    theta,
                    trios,
                    mismatch_probability=0.0,
                )
                expected = _enumerated_matched_null_log_likelihood(
                    child[0], founders, rho, theta
                )
                self.assertAlmostEqual(
                    observed.zero_observed[0], expected, places=13
                )
                self.assertEqual(
                    observed.zero_observed[0], observed.one_observed[0, 0]
                )
                self.assertEqual(
                    observed.zero_observed[0], observed.two_observed[0]
                )
                np.testing.assert_allclose(
                    observed.source_path_switch_probability, rho, atol=2e-15
                )
                np.testing.assert_allclose(
                    observed.transmission_switch_probability, theta
                )


    def test_matched_null_duplicate_padding_and_founder_permutation_invariance(self):
        rng = np.random.default_rng(911)
        founders = np.asarray(
            [
                [[0, 0], [0, 1]],
                [[1, 1], [1, 0]],
                [[0, 1], [1, 1]],
            ],
            dtype=np.int8,
        )
        marker_counts = np.asarray((1, 1), dtype=np.int64)
        candidate = rng.random((3, 2, 2, 3))
        candidate[:, :, 1] = 1.0 / 3.0
        candidate[1] = candidate[0]
        child = rng.random((2, 2, 2, 3))
        child[:, :, 1] = 1.0 / 3.0
        trios = np.asarray(
            ((0, 0, 2), (0, 1, 2), (1, 0, 2)), dtype=np.int64
        )

        def score(panel, candidate_gl, child_gl):
            posterior = infer_candidate_source_posterior(
                candidate_gl,
                panel,
                marker_counts,
                0.06,
                return_lumped_posterior=True,
                lumped_root_prior_mode="ordered_independent_uniform",
                posterior_factor_dtype=np.float64,
            )
            return score_candidate_source_batch_matched_null_exact(
                posterior,
                child_gl,
                panel,
                marker_counts,
                np.ones((2, 2)),
                0.02,
                trios,
            )

        direct = score(founders, candidate, child)
        np.testing.assert_array_equal(
            direct.one_observed[:, 0], direct.one_observed[:, 1]
        )
        self.assertEqual(direct.two_observed[0], direct.two_observed[1])

        padding_founders = founders.copy()
        padding_founders[:, :, 1] = 1 - padding_founders[:, :, 1]
        padding_candidate = candidate.copy()
        padding_candidate[:, :, 1] = rng.random((3, 2, 3))
        padding_child = child.copy()
        padding_child[:, :, 1] = rng.random((2, 2, 3))
        padded = score(padding_founders, padding_candidate, padding_child)
        for name in ("zero_observed", "one_observed", "two_observed"):
            np.testing.assert_allclose(
                getattr(direct, name), getattr(padded, name),
                rtol=0.0, atol=1e-14,
            )

        permutation = np.asarray((2, 0, 1))
        permuted = score(founders[permutation], candidate, child)
        for name in ("zero_observed", "one_observed", "two_observed"):
            np.testing.assert_allclose(
                getattr(direct, name), getattr(permuted, name),
                rtol=0.0, atol=2e-14,
            )


if __name__ == "__main__":
    unittest.main()
