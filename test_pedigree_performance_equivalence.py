"""Portable numerical and behavioural equivalence tests for pedigree speedups.

Every test in this module is self-contained: equivalence is checked against an
independent reference calculation, across execution modes, or across chunk
boundaries.  Historical snapshot comparisons belong in one-off benchmark
artifacts rather than the durable test suite.

Tropheops-sized fixtures reproduce only the observed eligibility dimensions
(116 samples: 4 G0, 16 F1 split 8/8 by recorded sex, and 96 F2).  They do not
assert individual parentage or use cohort labels as biological truth.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pedigree-performance-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/pedigree-performance-cache")

import numpy as np
import pandas as pd

import pedigree_inference as pedigree


def _assert_float_array_parity(
    case: unittest.TestCase,
    observed,
    expected,
    *,
    maxulp: int = 0,
    label: str = "array",
) -> None:
    observed_array = np.asarray(observed)
    expected_array = np.asarray(expected)
    case.assertEqual(observed_array.shape, expected_array.shape, label)
    case.assertTrue(
        np.array_equal(np.isnan(observed_array), np.isnan(expected_array)),
        f"{label}: NaN masks differ",
    )
    case.assertTrue(
        np.array_equal(np.isposinf(observed_array), np.isposinf(expected_array)),
        f"{label}: +inf masks differ",
    )
    case.assertTrue(
        np.array_equal(np.isneginf(observed_array), np.isneginf(expected_array)),
        f"{label}: -inf masks differ",
    )
    finite = np.isfinite(observed_array) & np.isfinite(expected_array)
    if maxulp == 0:
        np.testing.assert_array_equal(
            observed_array[finite], expected_array[finite], err_msg=label
        )
    else:
        np.testing.assert_array_max_ulp(
            observed_array[finite], expected_array[finite], maxulp=maxulp
        )


def _assert_frame_parity(
    case: unittest.TestCase,
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    float_rtol: float = 5e-8,
    float_atol: float = 5e-10,
) -> None:
    case.assertEqual(observed.columns.tolist(), expected.columns.tolist())
    case.assertEqual(observed.index.tolist(), expected.index.tolist())
    for column in observed.columns:
        left = observed[column]
        right = expected[column]
        if pd.api.types.is_float_dtype(left.dtype) and pd.api.types.is_float_dtype(
            right.dtype
        ):
            np.testing.assert_allclose(
                left.to_numpy(),
                right.to_numpy(),
                rtol=float_rtol,
                atol=float_atol,
                equal_nan=True,
                err_msg=column,
            )
        else:
            pd.testing.assert_series_equal(
                left, right, check_dtype=False, check_names=True
            )


def _cohort_policy(scenario: str) -> tuple[tuple[str, ...], dict[str, object]]:
    """Return synthetic eligibility with real cohort dimensions only."""
    if scenario == "tropheops116":
        generation_sizes = (("G0", 4), ("F1", 16), ("F2", 96))
        parent_generation = {"F2": "F1"}
    elif scenario == "synthetic320":
        generation_sizes = (("F1", 20), ("F2", 100), ("F3", 200))
        parent_generation = {"F2": "F1", "F3": "F2"}
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    sample_ids: list[str] = []
    generations: list[str] = []
    sexes: list[str] = []
    for generation, size in generation_sizes:
        for index in range(size):
            sample_ids.append(f"{generation}_{index}")
            generations.append(generation)
            # Balanced recorded-sex strata reproduce the pair-mask dimensions.
            sexes.append("F" if index < size // 2 else "M")

    samples = tuple(sample_ids)
    generation_array = np.asarray(generations, dtype=object)
    sex_array = np.asarray(sexes, dtype=object)
    n_samples = len(samples)
    children = np.isin(generation_array, tuple(parent_generation))
    parents = np.zeros((n_samples, n_samples), dtype=np.bool_)
    pairs = np.zeros((n_samples, n_samples, n_samples), dtype=np.bool_)
    for child in np.flatnonzero(children):
        parent_mask = generation_array == parent_generation[generation_array[child]]
        parents[child] = parent_mask
        females = np.flatnonzero(parent_mask & (sex_array == "F"))
        males = np.flatnonzero(parent_mask & (sex_array == "M"))
        pairs[child][np.ix_(females, males)] = True
        pairs[child][np.ix_(males, females)] = True
    record = {
        "format_version": 1,
        "sample_ids": samples,
        "eligible_children": np.ascontiguousarray(children),
        "eligible_parents": np.ascontiguousarray(parents),
        "eligible_parent_pairs": np.ascontiguousarray(pairs),
        "policy_name": f"performance_{scenario}_opposite_sex_v1",
        "source_fields": ("synthetic_generation", "synthetic_recorded_sex"),
        "assumptions": (
            "Cohort labels constrain the synthetic candidate universe only.",
        ),
        "individual_parentage_ground_truth": False,
    }
    return samples, record


def _canonical_trios(policy: dict[str, object]) -> np.ndarray:
    children = np.asarray(policy["eligible_children"], dtype=np.bool_)
    pairs = np.asarray(policy["eligible_parent_pairs"], dtype=np.bool_)
    rows: list[tuple[int, int, int]] = []
    for child in np.flatnonzero(children):
        first, second = np.nonzero(np.triu(pairs[child], k=1))
        rows.extend(
            (int(child), int(parent1), int(parent2))
            for parent1, parent2 in zip(first, second)
        )
    return np.asarray(rows, dtype=np.int64).reshape((-1, 3))


def _small_evidence_fixture() -> tuple[tuple[str, ...], list[dict], dict]:
    sample_ids = ("s0", "s1", "s2", "s3")
    trios = []
    for child in range(4):
        candidate_parents = [index for index in range(4) if index != child]
        for first_index, first in enumerate(candidate_parents):
            for second in candidate_parents[first_index + 1 :]:
                trios.append((child, first, second))
    trio_array = np.asarray(trios, dtype=np.int64)
    evidence: list[dict] = []
    edge_exposed = np.full((4, 4), 100.0)
    edge_matched = edge_exposed.copy()
    pair_exposed = np.full(len(trio_array), 100.0)
    pair_explained = pair_exposed.copy()
    for contig_index in range(3):
        zero = np.asarray((0.0, 8.0, 8.0, 8.0), dtype=np.float64)
        one = np.full((4, 4), -8.0, dtype=np.float64)
        np.fill_diagonal(one, -np.inf)
        one[0, 1] = 30.0
        one[0, 2] = 20.0
        one[0, 3] = 50.0
        two = np.full(len(trio_array), -8.0, dtype=np.float64)
        for row, (child, first, second) in enumerate(trio_array):
            if child == 0 and (first, second) == (1, 2):
                two[row] = 25.0
            elif child == 0 and (first, second) == (1, 3):
                two[row] = 60.0
        evidence.append(
            {
                "contig": f"ctg{contig_index}",
                "trios": trio_array,
                "zero_parent_log_likelihoods": zero,
                "one_parent_log_likelihoods": one,
                "two_parent_log_likelihoods": two,
                "informative_markers": 100,
                "edge_matched_bins": edge_matched,
                "edge_exposed_bins": edge_exposed,
                "pair_explained_bins": pair_explained,
                "pair_exposed_bins": pair_exposed,
                "structure_total_bins": 100.0,
            }
        )
    children = np.asarray((True, False, False, False), dtype=np.bool_)
    parents = np.zeros((4, 4), dtype=np.bool_)
    parents[0, (1, 2)] = True
    pairs = np.zeros((4, 4, 4), dtype=np.bool_)
    pairs[0, 1, 2] = pairs[0, 2, 1] = True
    policy = {
        "format_version": 1,
        "sample_ids": sample_ids,
        "eligible_children": children,
        "eligible_parents": parents,
        "eligible_parent_pairs": pairs,
        "policy_name": "performance_small_restrictive_v1",
        "source_fields": ("synthetic_policy",),
        "assumptions": ("Synthetic numerical fixture only.",),
        "individual_parentage_ground_truth": False,
    }
    return sample_ids, evidence, policy


def _ancestry_fixture(n_contigs: int, n_samples: int):
    junctions = np.tile(
        np.arange(n_samples, 0, -1, dtype=np.float64), (n_contigs, 1)
    )
    callable_bins = np.full_like(junctions, 100.0)
    return junctions, callable_bins


class EligibilityAndAlternativeEquivalenceTests(unittest.TestCase):
    def test_tropheops_candidate_dimensions_are_reproduced_without_truth_labels(self):
        samples, policy = _cohort_policy("tropheops116")
        self.assertEqual(len(samples), 116)
        self.assertEqual(np.count_nonzero(policy["eligible_children"]), 96)
        np.testing.assert_array_equal(
            np.count_nonzero(policy["eligible_parents"], axis=1)[20:], 16
        )
        self.assertEqual(len(_canonical_trios(policy)), 96 * 8 * 8)

    def test_robust_screen_respects_eligibility_masks(self):
        samples, policy = _cohort_policy("tropheops116")
        eligibility = pedigree._resolve_parent_eligibility(policy, samples)
        rng = np.random.default_rng(7721)
        scores = rng.normal(size=(5, 116, 116))
        marker_counts = np.asarray((90, 100, 250, 400, 900), dtype=np.float64)
        config = pedigree.PedigreeConfig(bootstrap_replicates=1).validated()
        observed = pedigree._robust_parent_screen(
            scores, marker_counts, config, eligibility
        )
        self.assertTrue(np.all(np.isneginf(observed[:20])))
        self.assertTrue(
            np.all(np.isneginf(observed[20:, :4]))
            and np.all(np.isneginf(observed[20:, 20:]))
        )
        self.assertTrue(np.all(np.isfinite(observed[20:, 4:20])))
        observed_trios = pedigree._fixed_trio_panel(
            observed, 20, 5, True, eligibility
        )
        np.testing.assert_array_equal(observed_trios, _canonical_trios(policy))


class StructureAndDepthEquivalenceTests(unittest.TestCase):
    @staticmethod
    def _reference_structure_counts(labels: np.ndarray, trios: np.ndarray):
        n_samples, n_bins, _ = labels.shape
        matched = np.zeros((n_samples, n_samples), dtype=np.float64)
        exposed = np.zeros_like(matched)
        for first in range(n_samples):
            for second in range(n_samples):
                for block in range(n_bins):
                    first_labels = labels[first, block]
                    second_labels = labels[second, block]
                    if np.any(first_labels < 0) or np.any(second_labels < 0):
                        continue
                    exposed[first, second] += 1.0
                    if np.intersect1d(first_labels, second_labels).size:
                        matched[first, second] += 1.0
        pair_explained = np.zeros(len(trios), dtype=np.float64)
        pair_exposed = np.zeros(len(trios), dtype=np.float64)
        for row, (child, first, second) in enumerate(trios):
            for block in range(n_bins):
                child_labels = labels[child, block]
                first_labels = labels[first, block]
                second_labels = labels[second, block]
                if (
                    np.any(child_labels < 0)
                    or np.any(first_labels < 0)
                    or np.any(second_labels < 0)
                ):
                    continue
                pair_exposed[row] += 1.0
                direct = (
                    child_labels[0] in first_labels
                    and child_labels[1] in second_labels
                )
                swapped = (
                    child_labels[1] in first_labels
                    and child_labels[0] in second_labels
                )
                pair_explained[row] += float(direct or swapped)
        return matched, exposed, pair_explained, pair_exposed

    def test_structure_counts_match_independent_missing_aware_reference(self):
        rng = np.random.default_rng(918)
        labels = rng.integers(0, 4, size=(7, 6, 2), dtype=np.int16)
        labels[0, 1, 0] = -1
        labels[4, 3, :] = -1
        trios = np.asarray(
            ((0, 1, 2), (0, 2, 3), (4, 1, 5), (6, 0, 5)),
            dtype=np.int64,
        )
        required_edges = np.ones((7, 7), dtype=np.bool_)
        observed = pedigree._parenthood_structure_count_kernel(labels, trios, required_edges)
        expected = self._reference_structure_counts(labels, trios)
        for left, right in zip(observed, expected):
            np.testing.assert_array_equal(left, right)




class BootstrapAndResultEquivalenceTests(unittest.TestCase):
    @staticmethod
    def _result(module, n_workers: int, *, bootstrap_replicates: int = 8):
        sample_ids, evidence, policy = _small_evidence_fixture()
        junctions, callable_bins = _ancestry_fixture(len(evidence), len(sample_ids))
        config = module.PedigreeConfig(
            bootstrap_replicates=bootstrap_replicates,
            minimum_informative_contigs=1,
            parent_state_minimum_exposed_contigs=1,
        ).validated()
        return module.infer_from_parent_state_evidence(
            evidence,
            sample_ids,
            config=config,
            parent_eligibility=policy,
            ancestry_junction_counts=junctions,
            ancestry_callable_haplotype_bins=callable_bins,
            n_workers=n_workers,
        )

    def test_bootstrap_multiplicity_chunks_and_integer_reductions_are_exact(self):
        sample_ids, evidence, policy = _small_evidence_fixture()
        settings = pedigree.PedigreeConfig(
            bootstrap_replicates=7,
            minimum_informative_contigs=1,
            parent_state_minimum_exposed_contigs=1,
        ).validated()
        eligibility = pedigree._resolve_parent_eligibility(policy, sample_ids)
        canonical = pedigree._canonical_parent_state_evidence(evidence, 4)
        (
            trios,
            zero,
            one,
            two,
            markers,
            edge_matched,
            edge_exposed,
            pair_explained,
            pair_exposed,
            total_bins,
            _,
        ) = canonical
        alternatives = pedigree._parent_state_alternatives(
            trios,
            zero,
            one,
            two,
            settings.parent_state_contamination_probability,
            eligibility,
        )
        rows, states, contig_likelihoods, by_child, full_counts = alternatives[:5]
        junctions, callable_bins = _ancestry_fixture(3, 4)
        shared = {
            "alternatives": rows,
            "states": states,
            "contig_log_likelihoods": contig_likelihoods,
            "by_child": tuple(by_child),
            "full_counts": full_counts,
            "junction_matrix": junctions,
            "callable_matrix": callable_bins,
            "n_samples": 4,
            "bootstrap_seed": settings.bootstrap_seed,
            "contig_information_weights": np.ceil(
                markers / settings.markers_per_information_block
            ).astype(float),
            "settings": settings,
            "structure_pair_indices": pedigree._structure_pair_indices(
                rows, states, trios
            ),
            "edge_matched_by_contig": edge_matched,
            "edge_exposed_by_contig": edge_exposed,
            "pair_explained_by_contig": pair_explained,
            "pair_exposed_by_contig": pair_exposed,
            "structure_total_bins_by_contig": total_bins,
        }
        rng = np.random.default_rng(settings.bootstrap_seed)
        multiplicities = np.stack(
            [
                np.bincount(rng.integers(0, 3, size=3), minlength=3).astype(float)
                for _ in range(settings.bootstrap_replicates)
            ]
        )
        whole = pedigree._evaluate_smart_bootstrap_chunk(shared, multiplicities)
        split = [
            pedigree._evaluate_smart_bootstrap_chunk(shared, chunk)
            for chunk in np.array_split(multiplicities, (2, 5))
        ]
        for index in range(len(whole) - 1):
            np.testing.assert_array_equal(
                whole[index], np.concatenate([chunk[index] for chunk in split])
            )
        self.assertEqual(whole[-1], sum(chunk[-1] for chunk in split))

        def empty_counts():
            return (
                np.zeros(len(rows), dtype=np.int64),
                np.zeros(len(rows), dtype=np.int64),
                np.zeros((4, 3), dtype=np.int64),
                np.zeros((4, 3), dtype=np.int64),
                np.zeros((4, 4), dtype=np.int64),
                np.zeros((4, 4), dtype=np.int64),
            )

        whole_counts = empty_counts()
        split_counts = empty_counts()
        whole_refits = pedigree._accumulate_smart_bootstrap_chunk(
            whole, rows, *whole_counts
        )
        split_refits = sum(
            pedigree._accumulate_smart_bootstrap_chunk(
                chunk, rows, *split_counts
            )
            for chunk in split
        )
        self.assertEqual(whole_refits, split_refits)
        for left, right in zip(whole_counts, split_counts):
            np.testing.assert_array_equal(left, right)

    def test_serial_and_forced_parallel_bootstrap_results_are_identical(self):
        previous = pedigree._SMART_BOOTSTRAP_MIN_WORK_ITEMS
        pedigree._SMART_BOOTSTRAP_MIN_WORK_ITEMS = 0
        try:
            serial = self._result(pedigree, 1, bootstrap_replicates=32)
            parallel = self._result(pedigree, 2, bootstrap_replicates=32)
        finally:
            pedigree._SMART_BOOTSTRAP_MIN_WORK_ITEMS = previous
        self.assertEqual(serial.smart_bootstrap_worker_count, 1)
        self.assertEqual(parallel.smart_bootstrap_worker_count, 2)
        for attribute in (
            "complete_relationships",
            "tier_a_relationships",
            "tier_b_relationships",
            "smart_parent_state_calls",
            "smart_diagnostics",
        ):
            _assert_frame_parity(
                self, getattr(serial, attribute), getattr(parallel, attribute)
            )


class HMMKernelEquivalenceTests(unittest.TestCase):
    @staticmethod
    def _hmm_inputs(n_samples: int = 5, n_bins: int = 4, n_snps: int = 2):
        rng = np.random.default_rng(30011)
        founders = rng.integers(0, 2, size=(4, n_bins, n_snps), dtype=np.int8)
        labels = rng.integers(0, 4, size=(n_samples, n_bins, 2), dtype=np.int16)
        alleles = np.empty((n_samples, n_bins, 2, n_snps), dtype=np.int8)
        for sample in range(n_samples):
            for block in range(n_bins):
                for track in range(2):
                    alleles[sample, block, track] = founders[
                        labels[sample, block, track], block
                    ]
        # Candidate 1 stays fully called (fast path); candidates 2 and 3 each
        # contain a missing-called-missing pattern (linked partial path).
        alleles[2, 0, 0, 0] = -1
        alleles[2, 2, 1, 1] = -1
        alleles[3, 1, 0, :] = -1
        hom = labels[:, :, 0] == labels[:, :, 1]
        marker_counts = np.full(n_bins, n_snps, dtype=np.int64)
        theta = np.linspace(0.0, 0.03, n_bins)
        trios = []
        for child in range(n_samples):
            candidates = [index for index in range(n_samples) if index != child]
            for first_index, first in enumerate(candidates):
                for second in candidates[first_index + 1 :]:
                    trios.append((child, first, second))
        trios = np.asarray(trios, dtype=np.int64)
        genotype = np.empty((n_samples, n_bins, n_snps, 3), dtype=np.float64)
        for sample in range(n_samples):
            for block in range(n_bins):
                for snp in range(n_snps):
                    dosage = sum(
                        int(alleles[sample, block, track, snp])
                        for track in range(2)
                        if alleles[sample, block, track, snp] >= 0
                    )
                    genotype[sample, block, snp] = np.asarray(
                        (0.82, 0.14, 0.04)
                        if dosage == 0
                        else (
                            (0.08, 0.84, 0.08)
                            if dosage == 1
                            else (0.04, 0.14, 0.82)
                        ),
                        dtype=np.float64,
                    )
        return genotype, alleles, labels, hom, founders, marker_counts, theta, trios

    def _compare_score_records(self, observed, expected, *, maxulp: int = 0):
        for field in (
            "zero_observed",
            "one_observed",
            "two_observed",
            "ancestry_junction_counts",
            "ancestry_callable_haplotype_bins",
            "one_parent_identity_information",
            "two_parent_edge_information",
            "edge_matched_bins",
            "edge_exposed_bins",
            "pair_explained_bins",
            "pair_exposed_bins",
        ):
            left = getattr(observed, field)
            right = getattr(expected, field)
            if left is None or right is None:
                self.assertIs(left, right, field)
            elif np.asarray(left).dtype.kind == "f":
                _assert_float_array_parity(
                    self, left, right, maxulp=maxulp, label=field
                )
            else:
                np.testing.assert_array_equal(left, right)

    def test_hard_hmm_dynamic_chunk_boundaries_are_bit_identical(self):
        inputs = self._hmm_inputs()[1:]
        small_chunks = pedigree.score_parent_state_hmms(
            *inputs,
            _dynamic_child_chunk_floor=1,
            _dynamic_child_chunk_scale=0,
        )
        one_chunk = pedigree.score_parent_state_hmms(
            *inputs,
            _dynamic_child_chunk_floor=64,
            _dynamic_child_chunk_scale=0,
        )
        self._compare_score_records(small_chunks, one_chunk)



if __name__ == "__main__":
    unittest.main(verbosity=2)
