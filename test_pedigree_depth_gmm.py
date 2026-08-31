"""Reference-parity tests for the specialized ancestry-depth 1-D GMM."""

import math
import unittest
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from pedigree_depth_gmm import (
    _kmeans_labels_1d,
    fit_bic_selected_gaussian_mixture_1d,
)


_SEED = 20260725
_REGULARIZATION = 1e-3
_N_INIT = 10
_MAX_ITERATIONS = 500
_DIRECTION_THRESHOLD = 0.01


def _representative_depth_fixture(sample_count):
    """Synthetic relative-depth burdens spanning the intended cohort sizes."""
    rng = np.random.default_rng(981 + sample_count)
    component = np.arange(sample_count) % 3
    rng.shuffle(component)
    component_means = np.asarray((1.0, 8.0, 18.0))
    component_scales = np.asarray((0.8, 1.4, 2.1))
    burden = component_means[component] + rng.normal(
        0.0, component_scales[component]
    )
    standardized = (burden - np.mean(burden)) / np.std(burden)
    # In production these fractions temper component likelihoods for samples
    # whose ancestry paintings are incompletely callable.
    callability = rng.uniform(0.25, 1.0, size=sample_count)
    callability[np.argmax(callability)] = 1.0
    return standardized, callability


def _sklearn_reference(values, maximum_components):
    observations = np.sort(values)[:, None]
    models = []
    bics = []
    for component_count in range(1, maximum_components + 1):
        model = GaussianMixture(
            n_components=component_count,
            covariance_type="full",
            reg_covar=_REGULARIZATION,
            n_init=_N_INIT,
            max_iter=_MAX_ITERATIONS,
            random_state=_SEED,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(observations)
        if model.converged_:
            models.append(model)
            bics.append(float(model.bic(observations)))
        else:
            models.append(None)
            bics.append(np.inf)
    finite = np.flatnonzero(np.isfinite(bics))
    selected_index = int(finite[np.argmin(np.asarray(bics)[finite])])
    selected = models[selected_index]
    order = np.argsort(selected.means_[:, 0], kind="stable")
    return (
        selected_index + 1,
        np.asarray(bics),
        selected.means_[order, 0],
        selected.covariances_[order, 0, 0],
        selected.weights_[order],
    )


def _tempered_posterior(values, callability, means, variances, weights):
    log_density = (
        -0.5 * np.log(2.0 * math.pi * variances)[None, :]
        - 0.5
        * (values[:, None] - means[None, :]) ** 2
        / variances[None, :]
    )
    log_scores = np.log(weights)[None, :] + callability[:, None] * log_density
    log_scores -= np.max(log_scores, axis=1, keepdims=True)
    posterior = np.exp(log_scores)
    posterior /= np.sum(posterior, axis=1, keepdims=True)
    return posterior


def _direction_matrix(posterior):
    lower_depth_probability = np.cumsum(posterior, axis=1) - posterior
    return np.clip(posterior @ lower_depth_probability.T, 0.0, 1.0)


class GaussianMixture1DParityTests(unittest.TestCase):
    def test_representative_component_bic_posterior_and_d_mask_parity(self):
        for sample_count in (14, 84, 320):
            with self.subTest(sample_count=sample_count):
                values, callability = _representative_depth_fixture(
                    sample_count
                )
                maximum_components = min(
                    6,
                    len(np.unique(values)),
                    max(1, sample_count // 2),
                )
                reference = _sklearn_reference(values, maximum_components)
                fitted = fit_bic_selected_gaussian_mixture_1d(
                    np.sort(values),
                    maximum_components,
                    _SEED,
                    n_init=_N_INIT,
                    max_iter=_MAX_ITERATIONS,
                    reg_covar=_REGULARIZATION,
                )

                self.assertEqual(
                    fitted.selected_component_count, reference[0]
                )
                np.testing.assert_allclose(
                    fitted.tested_bics,
                    reference[1],
                    rtol=0.0,
                    atol=2e-10,
                )
                np.testing.assert_allclose(
                    fitted.means, reference[2], rtol=0.0, atol=2e-12
                )
                np.testing.assert_allclose(
                    fitted.variances, reference[3], rtol=0.0, atol=2e-12
                )
                np.testing.assert_allclose(
                    fitted.weights, reference[4], rtol=0.0, atol=2e-12
                )

                reference_posterior = _tempered_posterior(
                    values,
                    callability,
                    reference[2],
                    reference[3],
                    reference[4],
                )
                fitted_posterior = _tempered_posterior(
                    values,
                    callability,
                    fitted.means,
                    fitted.variances,
                    fitted.weights,
                )
                np.testing.assert_allclose(
                    fitted_posterior,
                    reference_posterior,
                    rtol=0.0,
                    atol=2e-12,
                )
                reference_direction = _direction_matrix(reference_posterior)
                fitted_direction = _direction_matrix(fitted_posterior)
                np.testing.assert_array_equal(
                    fitted_direction >= _DIRECTION_THRESHOLD,
                    reference_direction >= _DIRECTION_THRESHOLD,
                )

    def test_discrete_tied_burdens_match_sklearn_partition_ties(self):
        burden = np.asarray(
            (0, 0, 1, 1, 2, 2, 8, 8, 9, 9, 18, 18, 20, 20),
            dtype=np.float64,
        )
        values = (burden - np.mean(burden)) / np.std(burden)
        callability = np.linspace(0.25, 1.0, len(values))
        reference = _sklearn_reference(values, 6)
        fitted = fit_bic_selected_gaussian_mixture_1d(
            np.sort(values), 6, _SEED
        )

        self.assertEqual(fitted.selected_component_count, reference[0])
        np.testing.assert_allclose(
            fitted.tested_bics, reference[1], rtol=0.0, atol=2e-10
        )
        np.testing.assert_allclose(
            fitted.means, reference[2], rtol=0.0, atol=2e-12
        )
        np.testing.assert_allclose(
            fitted.variances, reference[3], rtol=0.0, atol=2e-12
        )
        np.testing.assert_allclose(
            fitted.weights, reference[4], rtol=0.0, atol=2e-12
        )
        reference_posterior = _tempered_posterior(
            values, callability, reference[2], reference[3], reference[4]
        )
        fitted_posterior = _tempered_posterior(
            values,
            callability,
            fitted.means,
            fitted.variances,
            fitted.weights,
        )
        np.testing.assert_array_equal(
            _direction_matrix(fitted_posterior) >= _DIRECTION_THRESHOLD,
            _direction_matrix(reference_posterior) >= _DIRECTION_THRESHOLD,
        )

    def test_seeded_kmeans_partitions_match_sklearn(self):
        continuous, _ = _representative_depth_fixture(84)
        discrete = np.asarray(
            (0, 0, 1, 1, 2, 2, 8, 8, 9, 9, 18, 18, 20, 20),
            dtype=np.float64,
        )
        discrete = (discrete - np.mean(discrete)) / np.std(discrete)
        for values in (continuous, discrete):
            observations = np.sort(values)
            for component_count in range(1, 7):
                with self.subTest(
                    sample_count=len(values),
                    component_count=component_count,
                ):
                    reference_rng = np.random.RandomState(_SEED)
                    specialized_rng = np.random.RandomState(_SEED)
                    for _ in range(_N_INIT):
                        reference_labels = KMeans(
                            n_clusters=component_count,
                            n_init=1,
                            random_state=reference_rng,
                        ).fit(observations[:, None]).labels_
                        specialized_labels = _kmeans_labels_1d(
                            observations,
                            component_count,
                            specialized_rng,
                        )
                        np.testing.assert_array_equal(
                            specialized_labels[:, None]
                            == specialized_labels[None, :],
                            reference_labels[:, None]
                            == reference_labels[None, :],
                        )

    def test_fixed_seed_is_bitwise_deterministic(self):
        values, _ = _representative_depth_fixture(84)
        first = fit_bic_selected_gaussian_mixture_1d(
            np.sort(values), 6, _SEED
        )
        second = fit_bic_selected_gaussian_mixture_1d(
            np.sort(values), 6, _SEED
        )
        self.assertEqual(first.tested_bics, second.tested_bics)
        np.testing.assert_array_equal(first.means, second.means)
        np.testing.assert_array_equal(first.variances, second.variances)
        np.testing.assert_array_equal(first.weights, second.weights)

    def test_all_nonconverged_candidates_have_explicit_failure_result(self):
        values, _ = _representative_depth_fixture(14)
        fitted = fit_bic_selected_gaussian_mixture_1d(
            np.sort(values), 3, _SEED, max_iter=1
        )
        self.assertFalse(fitted.converged)
        self.assertEqual(fitted.selected_component_count, 0)
        self.assertTrue(all(np.isinf(value) for value in fitted.tested_bics))
        self.assertEqual(fitted.means.size, 0)


if __name__ == "__main__":
    unittest.main()
