"""Founder-site pseudo-evidence materialization invariants."""

import math

import numpy as np

from bhd_results import _materialize_founder_site_pseudo_evidence


def _run(probs, haps, assignments, observed, lam=1.0,
         min_directional_supporters=1, min_pseudo_probability=0.5):
    return _materialize_founder_site_pseudo_evidence(
        np.asarray(probs, dtype=np.float64),
        np.asarray(haps, dtype=np.int64),
        np.asarray(assignments, dtype=np.int64),
        np.asarray(observed, dtype=np.bool_),
        lam, min_directional_supporters, min_pseudo_probability)


def test_all_carrier_blackout_is_unknown():
    q, support, pseudo_odds, mask, calls = _run(
        [[[0.98, 0.01, 0.01]], [[0.01, 0.01, 0.98]]],
        [[1]], [[0, 0], [0, 0]], [[False], [False]],
        min_pseudo_probability=0.9)
    np.testing.assert_array_equal(q, [[0.5]])
    np.testing.assert_array_equal(support, [[0]])
    np.testing.assert_array_equal(pseudo_odds, [[0.0]])
    np.testing.assert_array_equal(mask, [[False]])
    np.testing.assert_array_equal(calls, [[-1]])


def test_observed_same_founder_carriers_call_both_alleles():
    ref = [0.98, 0.01, 0.01]
    alt = ref[::-1]
    probs = np.asarray([[ref, alt], [ref, alt], [ref, alt]])
    q, support, pseudo_odds, mask, calls = _run(
        probs, [[1, 0]], [[0, 0], [0, 0], [0, 0]],
        np.ones((3, 2), dtype=np.bool_),
        min_directional_supporters=2, min_pseudo_probability=0.99)
    np.testing.assert_array_equal(support, [[3, 3]])
    assert pseudo_odds[0, 0] < 0.0 < pseudo_odds[0, 1]
    assert q[0, 0] < 0.01 and q[0, 1] > 0.99
    np.testing.assert_array_equal(mask, [[True, True]])
    np.testing.assert_array_equal(calls, [[0, 1]])


def test_ref_alt_symmetry_with_real_founder_partner():
    probs = np.asarray([[[0.80, 0.19, 0.01]],
                        [[0.75, 0.24, 0.01]]])
    haps = np.asarray([[0], [0]])
    assignments = np.asarray([[0, 1], [0, 1]])
    observed = np.ones((2, 1), dtype=np.bool_)
    direct = _run(probs, haps, assignments, observed, lam=0.7)
    swapped = _run(
        probs[:, :, ::-1], 1 - haps, assignments, observed, lam=0.7)
    q0, support0, odds0, mask0, calls0 = direct
    q1, support1, odds1, mask1, calls1 = swapped
    np.testing.assert_allclose(q1, 1.0 - q0, atol=1e-15)
    np.testing.assert_allclose(odds1, -odds0, atol=1e-15)
    np.testing.assert_array_equal(support1, support0)
    np.testing.assert_array_equal(mask1, mask0)
    np.testing.assert_array_equal(calls1, 1 - calls0)


def test_real_wildcard_bucket_uses_wildcard_cap():
    q, support, pseudo_odds, mask, calls = _run(
        [[[0.90, 0.09, 0.01]]], [[1]], [[0, 1]], [[True]],
        lam=0.5, min_pseudo_probability=0.6)
    np.testing.assert_allclose(pseudo_odds, [[-0.5]], atol=1e-15)
    np.testing.assert_allclose(
        q, [[1.0 / (1.0 + math.exp(0.5))]], atol=1e-15)
    np.testing.assert_array_equal(support, [[1]])
    np.testing.assert_array_equal(mask, [[True]])
    np.testing.assert_array_equal(calls, [[0]])


def test_uniform_and_masked_cells_cannot_add_support():
    probs = np.asarray([[[0.98, 0.01, 0.01]],
                        [[1 / 3, 1 / 3, 1 / 3]],
                        [[0.01, 0.01, 0.98]]])
    q, support, pseudo_odds, mask, calls = _run(
        probs, [[1]], [[0, 0], [0, 0], [0, 0]],
        [[True], [True], [False]], lam=0.5, min_pseudo_probability=0.7)
    np.testing.assert_array_equal(support, [[1]])
    np.testing.assert_allclose(pseudo_odds, [[-1.0]], atol=1e-15)
    np.testing.assert_allclose(
        q, [[1.0 / (1.0 + math.exp(1.0))]], atol=1e-15)
    np.testing.assert_array_equal(mask, [[True]])
    np.testing.assert_array_equal(calls, [[0]])
