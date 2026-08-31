"""Equivalence and lightweight performance checks for eligibility policies."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import unittest

import numpy as np
import pandas as pd

import pedigree_pipeline as integration


def _reference_masks(
    generation: np.ndarray,
    sex: np.ndarray,
    *,
    policy: str,
    require_opposite_sex_pair: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the original loop formulas as an independent oracle."""
    n_samples = len(generation)
    if policy == "tropheops":
        eligible_children = generation == "F2"
        eligible_parent_samples = generation == "F1"
        eligible_parents = (
            eligible_children[:, None] & eligible_parent_samples[None, :]
        )
        np.fill_diagonal(eligible_parents, False)
    elif policy == "asac":
        eligible_children = np.isin(generation, ("F2", "F3"))
        eligible_parents = np.zeros((n_samples, n_samples), dtype=np.bool_)
        for child in np.flatnonzero(eligible_children):
            parent_generation = "F1" if generation[child] == "F2" else "F2"
            eligible_parents[child] = generation == parent_generation
            eligible_parents[child, child] = False
    else:
        raise AssertionError(policy)

    eligible_pairs = np.zeros(
        (n_samples, n_samples, n_samples), dtype=np.bool_
    )
    for child in np.flatnonzero(eligible_children):
        parents = np.flatnonzero(eligible_parents[child])
        for first_offset, first in enumerate(parents):
            for second in parents[first_offset + 1:]:
                if (
                    require_opposite_sex_pair
                    and {sex[int(first)], sex[int(second)]} != {"F", "M"}
                ):
                    continue
                eligible_pairs[child, first, second] = True
                eligible_pairs[child, second, first] = True
    return eligible_children, eligible_parents, eligible_pairs


def _reference_identity(record: dict) -> str:
    scalar = {
        "format_version": int(record["format_version"]),
        "policy_name": str(record["policy_name"]),
        "sample_ids": [str(value) for value in record["sample_ids"]],
        "source_fields": list(record.get("source_fields", ())),
        "assumptions": list(record.get("assumptions", ())),
        "individual_parentage_ground_truth": bool(
            record.get("individual_parentage_ground_truth", False)
        ),
    }
    digest = hashlib.sha256(json.dumps(
        scalar,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8"))
    for field in (
        "eligible_children",
        "eligible_parents",
        "eligible_parent_pairs",
    ):
        values = np.ascontiguousarray(record[field], dtype=np.bool_)
        digest.update(field.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _reference_summary(record: dict) -> dict:
    children = np.asarray(record["eligible_children"], dtype=np.bool_)
    parents = np.asarray(record["eligible_parents"], dtype=np.bool_)
    pairs = np.asarray(record["eligible_parent_pairs"], dtype=np.bool_)
    pair_counts = np.count_nonzero(np.triu(pairs, k=1), axis=(1, 2))
    parent_counts = np.count_nonzero(parents, axis=1)
    targets = np.flatnonzero(children)
    return {
        "format_version": int(record["format_version"]),
        "policy_name": str(record["policy_name"]),
        "sample_count": int(len(children)),
        "eligible_child_count": int(np.count_nonzero(children)),
        "candidate_parent_sample_count": int(
            np.count_nonzero(np.any(parents, axis=0))
        ),
        "minimum_parent_candidates_per_target": int(
            np.min(parent_counts[targets]) if len(targets) else 0
        ),
        "maximum_parent_candidates_per_target": int(
            np.max(parent_counts[targets]) if len(targets) else 0
        ),
        "minimum_parent_pairs_per_target": int(
            np.min(pair_counts[targets]) if len(targets) else 0
        ),
        "maximum_parent_pairs_per_target": int(
            np.max(pair_counts[targets]) if len(targets) else 0
        ),
        "individual_parentage_ground_truth": bool(
            record.get("individual_parentage_ground_truth", False)
        ),
        "assumptions": list(record.get("assumptions", ())),
    }


def _metadata(generation: np.ndarray, sex: np.ndarray) -> tuple[list[str], pd.DataFrame]:
    sample_ids = [f"sample_{index}" for index in range(len(generation))]
    metadata = pd.DataFrame({
        "primary_ID": sample_ids,
        "generation": generation,
        "sex": sex,
    })
    return sample_ids, metadata


def _assert_masks_equal(
    testcase: unittest.TestCase,
    record: dict,
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    for field, values in zip(
        ("eligible_children", "eligible_parents", "eligible_parent_pairs"),
        expected,
    ):
        observed = record[field]
        testcase.assertEqual(observed.dtype, np.dtype(np.bool_))
        testcase.assertTrue(observed.flags.c_contiguous)
        testcase.assertEqual(observed.shape, values.shape)
        testcase.assertEqual(observed.tobytes(), values.tobytes())
    testcase.assertEqual(
        integration.parent_eligibility_identity(record),
        _reference_identity(record),
    )
    testcase.assertEqual(
        integration.summarize_parent_eligibility(record),
        _reference_summary(record),
    )


class EligibilityBuilderEquivalenceTests(unittest.TestCase):
    def test_randomized_small_metadata_matches_loop_formulas(self):
        rng = np.random.default_rng(20260831)
        for policy, builder in (
            ("tropheops", integration.build_tropheops_parent_eligibility),
            ("asac", integration.build_asac_parent_eligibility),
        ):
            for replicate in range(20):
                n_samples = int(rng.integers(3, 30))
                if policy == "tropheops":
                    generation = rng.choice(
                        np.asarray(("G0", "F1", "F2", "F3", "")),
                        n_samples,
                    )
                    sex = rng.choice(
                        np.asarray(("F", "M", "U", "")), n_samples
                    )
                else:
                    generation = rng.choice(
                        np.asarray(("G0", "F1", "F2", "F3")), n_samples
                    )
                    sex = rng.choice(np.asarray(("F", "M")), n_samples)
                sample_ids, metadata = _metadata(generation, sex)
                metadata = metadata.sample(
                    frac=1.0, random_state=replicate
                ).reset_index(drop=True)
                for require_opposite_sex_pair in (False, True):
                    with self.subTest(
                        policy=policy,
                        replicate=replicate,
                        opposite_sex=require_opposite_sex_pair,
                    ):
                        record = builder(
                            metadata,
                            sample_ids,
                            require_opposite_sex_pair=(
                                require_opposite_sex_pair
                            ),
                        )
                        expected = _reference_masks(
                            generation,
                            sex,
                            policy=policy,
                            require_opposite_sex_pair=(
                                require_opposite_sex_pair
                            ),
                        )
                        _assert_masks_equal(self, record, expected)

    def test_representative_dimensions_match_loop_formulas(self):
        fixtures = (
            (
                "tropheops",
                integration.build_tropheops_parent_eligibility,
                np.asarray(["G0"] * 4 + ["F1"] * 16 + ["F2"] * 96),
            ),
            (
                "asac",
                integration.build_asac_parent_eligibility,
                np.asarray(
                    ["G0"] * 4 + ["F1"] * 40
                    + ["F2"] * 200 + ["F3"] * 46
                ),
            ),
        )
        for policy, builder, generation in fixtures:
            sex = np.where(np.arange(len(generation)) % 2, "M", "F")
            sample_ids, metadata = _metadata(generation, sex)
            with self.subTest(policy=policy, n_samples=len(generation)):
                record = builder(metadata, sample_ids)
                expected = _reference_masks(
                    generation,
                    sex,
                    policy=policy,
                    require_opposite_sex_pair=True,
                )
                _assert_masks_equal(self, record, expected)

    def test_summary_matches_reference_for_asymmetric_pair_mask(self):
        rng = np.random.default_rng(5541)
        n_samples = 17
        record = {
            "format_version": 1,
            "policy_name": "test",
            "eligible_children": rng.random(n_samples) < 0.6,
            "eligible_parents": rng.random((n_samples, n_samples)) < 0.3,
            "eligible_parent_pairs": (
                rng.random((n_samples, n_samples, n_samples)) < 0.2
            ),
        }
        self.assertEqual(
            integration.summarize_parent_eligibility(record),
            _reference_summary(record),
        )


def _run_lightweight_benchmark() -> None:
    fixtures = (
        ("tropheops", integration.build_tropheops_parent_eligibility,
         np.asarray(["G0"] * 4 + ["F1"] * 16 + ["F2"] * 96)),
        ("asac", integration.build_asac_parent_eligibility,
         np.asarray(["G0"] * 4 + ["F1"] * 40 + ["F2"] * 200 + ["F3"] * 46)),
    )
    for policy, builder, generation in fixtures:
        sex = np.where(np.arange(len(generation)) % 2, "M", "F")
        sample_ids, metadata = _metadata(generation, sex)
        start = time.perf_counter()
        expected = _reference_masks(
            generation, sex, policy=policy, require_opposite_sex_pair=True
        )
        reference_seconds = time.perf_counter() - start
        start = time.perf_counter()
        record = builder(metadata, sample_ids)
        optimized_seconds = time.perf_counter() - start
        np.testing.assert_array_equal(record["eligible_parent_pairs"], expected[2])
        print(
            f"{policy}: reference_core={reference_seconds:.6f}s "
            f"optimized_full_builder={optimized_seconds:.6f}s "
            f"speedup={reference_seconds / optimized_seconds:.2f}x"
        )


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        _run_lightweight_benchmark()
    else:
        unittest.main()
