"""Independent dense-reference tests for compact parent-pair eligibility.

The production default means that every unordered pair of individually
eligible parents is M2-eligible.  These tests intentionally construct the old
dense ``(child, parent1, parent2)`` mask only inside bounded reference fixtures;
the resolved production object must keep that policy implicit.

Run the fresh-process resource benchmark explicitly with::

    PEDIGREE_ELIGIBILITY_BENCHMARK=1 \
        python -m unittest test_pedigree_eligibility_quadratic
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import numpy as np

import pedigree_inference as pedigree


def _dense_derived_pairs(parents: np.ndarray) -> np.ndarray:
    """Historical dense reference for all pairs of eligible parents."""
    parents = np.asarray(parents, dtype=np.bool_)
    pairs = parents[:, :, None] & parents[:, None, :]
    diagonal = np.arange(parents.shape[1])
    pairs[:, diagonal, diagonal] = False
    return np.ascontiguousarray(pairs)


def _random_parent_policy(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    children = rng.random(n_samples) < 0.75
    parents = rng.random((n_samples, n_samples)) < 0.45
    np.fill_diagonal(parents, False)
    parents[~children] = False
    return np.ascontiguousarray(children), np.ascontiguousarray(parents)


def _implicit_record(children: np.ndarray, parents: np.ndarray) -> dict:
    n_samples = len(children)
    return {
        "format_version": pedigree.PARENT_ELIGIBILITY_FORMAT_VERSION,
        "sample_ids": tuple(range(n_samples)),
        "eligible_children": np.ascontiguousarray(children),
        "eligible_parents": np.ascontiguousarray(parents),
        "eligible_parent_pairs": None,
        "policy_name": "quadratic_test_implicit_pairs_v1",
        "source_fields": ("synthetic",),
        "assumptions": ("all unordered pairs of eligible parents",),
        "individual_parentage_ground_truth": False,
    }


def _explicit_record(children: np.ndarray, parents: np.ndarray) -> dict:
    record = _implicit_record(children, parents)
    record["policy_name"] = "quadratic_test_explicit_dense_pairs_v1"
    record["eligible_parent_pairs"] = _dense_derived_pairs(parents)
    return record


def _reference_pair_count(parents: np.ndarray) -> np.ndarray:
    degrees = np.count_nonzero(parents, axis=1).astype(np.int64)
    return degrees * (degrees - 1) // 2


def _reference_pair_membership(
    parents: np.ndarray,
    child: np.ndarray | int,
    first: np.ndarray | int,
    second: np.ndarray | int,
) -> np.ndarray:
    child_array, first_array, second_array = np.broadcast_arrays(
        np.asarray(child), np.asarray(first), np.asarray(second)
    )
    return (
        (first_array != second_array)
        & parents[child_array, first_array]
        & parents[child_array, second_array]
    )


def _reference_fixed_panel(
    scores: np.ndarray,
    children: np.ndarray,
    parents: np.ndarray,
    pairs: np.ndarray,
    top_k: int,
    anchor_k: int,
    use_anchor_union: bool,
) -> np.ndarray:
    rows: list[tuple[int, int, int]] = []
    for child in range(len(children)):
        if not children[child]:
            continue
        candidates = np.flatnonzero(parents[child])
        order = np.lexsort((candidates, -scores[child, candidates]))
        leading = candidates[order[: min(top_k, len(candidates))]].tolist()
        selected = {
            tuple(sorted((int(leading[i]), int(leading[j]))))
            for i in range(len(leading))
            for j in range(i + 1, len(leading))
            if pairs[child, leading[i], leading[j]]
        }
        if use_anchor_union:
            for anchor in leading[: min(anchor_k, len(leading))]:
                for other in candidates:
                    if other != anchor and pairs[child, anchor, other]:
                        selected.add(tuple(sorted((int(anchor), int(other)))))
        rows.extend((child, first, second) for first, second in sorted(selected))
    return np.asarray(rows, dtype=np.int64).reshape((-1, 3))


def _screened_trios(
    scores: np.ndarray,
    children: np.ndarray,
    parents: np.ndarray,
    *,
    top_k: int,
    anchor_k: int,
    use_anchor_union: bool,
) -> np.ndarray:
    return _reference_fixed_panel(
        scores,
        children,
        parents,
        _dense_derived_pairs(parents),
        top_k,
        anchor_k,
        use_anchor_union,
    )


def _assert_resolved_has_no_cube(test: unittest.TestCase, resolved) -> None:
    for field in dataclasses.fields(resolved):
        value = getattr(resolved, field.name)
        test.assertFalse(
            isinstance(value, np.ndarray) and value.ndim == 3,
            f"resolved eligibility field {field.name!r} materialized a cube",
        )


def _helper_candidates(*names: str):
    for name in names:
        helper = getattr(pedigree, name, None)
        if helper is not None:
            return helper
    return None


class CompactResolutionTests(unittest.TestCase):
    def test_default_and_implicit_records_resolve_without_any_cube(self):
        for n_samples in range(3, 9):
            default = pedigree._resolve_parent_eligibility(None, range(n_samples))
            _assert_resolved_has_no_cube(self, default)
            children, parents = _random_parent_policy(n_samples, 1000 + n_samples)
            implicit = pedigree._resolve_parent_eligibility(
                _implicit_record(children, parents), range(n_samples)
            )
            _assert_resolved_has_no_cube(self, implicit)

    def test_exhaustive_small_membership_and_counts_match_dense_reference(self):
        pair_helper = _helper_candidates(
            "_eligible_parent_pair_mask",
            "_parent_pair_eligibility_mask",
            "_eligible_pair_mask",
        )
        count_helper = _helper_candidates(
            "_eligible_parent_pair_counts",
            "_parent_pair_eligibility_counts",
            "_eligible_pair_counts",
        )
        if pair_helper is None or count_helper is None:
            self.fail("compact eligibility membership/count helpers are not available")

        # Exhaust every off-diagonal parent mask for N=3 and every child target.
        n_samples = 3
        off_diagonal = [(i, j) for i in range(n_samples) for j in range(n_samples) if i != j]
        for bits in range(1 << len(off_diagonal)):
            parents = np.zeros((n_samples, n_samples), dtype=np.bool_)
            for bit, (child, parent) in enumerate(off_diagonal):
                parents[child, parent] = bool(bits & (1 << bit))
            children = np.any(parents, axis=1)
            parents[~children] = False
            resolved = pedigree._resolve_parent_eligibility(
                _implicit_record(children, parents), range(n_samples)
            )
            dense = _dense_derived_pairs(parents)
            child, first, second = np.indices(dense.shape)
            observed = pair_helper(resolved, child.ravel(), first.ravel(), second.ravel())
            np.testing.assert_array_equal(observed.reshape(dense.shape), dense)
            np.testing.assert_array_equal(count_helper(resolved), _reference_pair_count(parents))

    def test_randomized_membership_and_counts_at_n14_and_n84(self):
        pair_helper = _helper_candidates(
            "_eligible_parent_pair_mask",
            "_parent_pair_eligibility_mask",
            "_eligible_pair_mask",
        )
        count_helper = _helper_candidates(
            "_eligible_parent_pair_counts",
            "_parent_pair_eligibility_counts",
            "_eligible_pair_counts",
        )
        if pair_helper is None or count_helper is None:
            self.fail("compact eligibility membership/count helpers are not available")
        for n_samples, seed in ((14, 441), (84, 442)):
            with self.subTest(n_samples=n_samples):
                children, parents = _random_parent_policy(n_samples, seed)
                resolved = pedigree._resolve_parent_eligibility(
                    _implicit_record(children, parents), range(n_samples)
                )
                dense = _dense_derived_pairs(parents)
                rng = np.random.default_rng(seed + 100)
                child = rng.integers(0, n_samples, size=20_000)
                first = rng.integers(0, n_samples, size=20_000)
                second = rng.integers(0, n_samples, size=20_000)
                np.testing.assert_array_equal(
                    pair_helper(resolved, child, first, second),
                    dense[child, first, second],
                )
                np.testing.assert_array_equal(
                    count_helper(resolved), _reference_pair_count(parents)
                )

    def test_n320_default_is_compact_and_has_exact_analytic_counts(self):
        count_helper = _helper_candidates(
            "_eligible_parent_pair_counts",
            "_parent_pair_eligibility_counts",
            "_eligible_pair_counts",
        )
        if count_helper is None:
            self.fail("compact eligibility pair-count helper is not available")
        resolved = pedigree._resolve_parent_eligibility(None, range(320))
        _assert_resolved_has_no_cube(self, resolved)
        np.testing.assert_array_equal(
            count_helper(resolved), np.full(320, 319 * 318 // 2, dtype=np.int64)
        )


class TrioPanelTests(unittest.TestCase):
    def _case(self, n_samples: int, seed: int, top_k: int, anchor_k: int):
        children, parents = _random_parent_policy(n_samples, seed)
        rng = np.random.default_rng(seed + 1)
        scores = rng.integers(-2, 4, size=(n_samples, n_samples)).astype(np.float64)
        scores[~parents] = -np.inf
        implicit = pedigree._resolve_parent_eligibility(
            _implicit_record(children, parents), range(n_samples)
        )
        explicit = pedigree._resolve_parent_eligibility(
            _explicit_record(children, parents), range(n_samples)
        )
        dense = _dense_derived_pairs(parents)
        for use_anchor_union in (False, True):
            expected = _reference_fixed_panel(
                scores, children, parents, dense,
                top_k, anchor_k, use_anchor_union,
            )
            observed = pedigree._fixed_trio_panel(
                scores, top_k, anchor_k, use_anchor_union, implicit
            )
            explicit_observed = pedigree._fixed_trio_panel(
                scores, top_k, anchor_k, use_anchor_union, explicit
            )
            np.testing.assert_array_equal(observed, expected)
            np.testing.assert_array_equal(explicit_observed, expected)

    def test_exhaustive_tied_small_panels_preserve_canonical_row_order(self):
        for n_samples in range(3, 8):
            with self.subTest(n_samples=n_samples):
                children = np.ones(n_samples, dtype=np.bool_)
                parents = np.ones((n_samples, n_samples), dtype=np.bool_)
                np.fill_diagonal(parents, False)
                scores = np.zeros((n_samples, n_samples), dtype=np.float64)
                np.fill_diagonal(scores, -np.inf)
                resolved = pedigree._resolve_parent_eligibility(
                    _implicit_record(children, parents), range(n_samples)
                )
                expected = _reference_fixed_panel(
                    scores, children, parents, _dense_derived_pairs(parents),
                    min(4, n_samples - 1), min(2, n_samples - 1), True,
                )
                observed = pedigree._fixed_trio_panel(
                    scores, min(4, n_samples - 1), min(2, n_samples - 1), True,
                    resolved,
                )
                np.testing.assert_array_equal(observed, expected)

    def test_randomized_n14_n84_and_bounded_n320(self):
        for case in ((14, 710, 7, 3), (84, 711, 20, 5), (320, 712, 20, 5)):
            with self.subTest(n_samples=case[0]):
                self._case(*case)


class ParentStateAlternativeTests(unittest.TestCase):
    def _assert_case(self, n_samples: int, seed: int, n_contigs: int = 3):
        children, parents = _random_parent_policy(n_samples, seed)
        rng = np.random.default_rng(seed + 20)
        scores = rng.integers(-3, 5, size=(n_samples, n_samples)).astype(np.float64)
        scores[~parents] = -np.inf
        trios = _screened_trios(
            scores, children, parents, top_k=min(20, n_samples - 1),
            anchor_k=min(5, n_samples - 1), use_anchor_union=True,
        )
        zero = rng.normal(size=(n_contigs, n_samples))
        one = rng.normal(size=(n_contigs, n_samples, n_samples))
        for contig in range(n_contigs):
            np.fill_diagonal(one[contig], -np.inf)
        two = rng.normal(size=(n_contigs, len(trios)))
        implicit = pedigree._resolve_parent_eligibility(
            _implicit_record(children, parents), range(n_samples)
        )
        explicit = pedigree._resolve_parent_eligibility(
            _explicit_record(children, parents), range(n_samples)
        )
        for contamination in (0.0, 0.02):
            observed = pedigree._parent_state_alternatives(
                trios, zero, one, two, contamination, implicit
            )
            expected = pedigree._parent_state_alternatives(
                trios, zero, one, two, contamination, explicit
            )
            for index in (0, 1, 2, 4, 5):
                np.testing.assert_array_equal(observed[index], expected[index])
            self.assertEqual(len(observed[3]), len(expected[3]))
            for left, right in zip(observed[3], expected[3]):
                np.testing.assert_array_equal(left, right)
            np.testing.assert_array_equal(
                observed[4][:, 2], _reference_pair_count(parents)
            )
            np.testing.assert_array_equal(
                observed[5][:, 2], np.bincount(
                    trios[:, 0], minlength=n_samples
                ) if len(trios) else np.zeros(n_samples, dtype=np.int64)
            )

    def test_exact_implicit_vs_dense_for_n14_and_n84(self):
        self._assert_case(14, 901)
        self._assert_case(84, 902)

    def test_bounded_n320_alternatives_without_dense_reference_cube(self):
        n_samples = 320
        rng = np.random.default_rng(903)
        children = np.ones(n_samples, dtype=np.bool_)
        parents = np.ones((n_samples, n_samples), dtype=np.bool_)
        np.fill_diagonal(parents, False)
        scores = rng.normal(size=(n_samples, n_samples))
        np.fill_diagonal(scores, -np.inf)
        resolved = pedigree._resolve_parent_eligibility(
            _implicit_record(children, parents), range(n_samples)
        )
        trios = pedigree._fixed_trio_panel(scores, 20, 5, True, resolved)
        zero = rng.normal(size=(2, n_samples))
        one = rng.normal(size=(2, n_samples, n_samples))
        for contig in range(2):
            np.fill_diagonal(one[contig], -np.inf)
        two = rng.normal(size=(2, len(trios)))
        output = pedigree._parent_state_alternatives(
            trios, zero, one, two, 0.02, resolved
        )
        np.testing.assert_array_equal(
            output[4][:, 2], np.full(n_samples, 319 * 318 // 2, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            output[5][:, 2], np.bincount(trios[:, 0], minlength=n_samples)
        )
        _assert_resolved_has_no_cube(self, resolved)


class CompactResultRecordTests(unittest.TestCase):
    def test_default_result_record_does_not_expand_pairs(self):
        from test_pedigree_inference import (
            _eligibility_test_ancestry,
            _eligibility_test_evidence,
        )

        sample_ids, evidence = _eligibility_test_evidence()
        result = pedigree.infer_from_parent_state_evidence(
            evidence,
            sample_ids,
            config=pedigree.PedigreeConfig(
                bootstrap_replicates=1,
                minimum_informative_contigs=1,
            ),
            n_workers=1,
            **_eligibility_test_ancestry(sample_ids, evidence),
        )
        record = result.smart_parent_eligibility_record
        self.assertIn("eligible_parent_pairs", record)
        self.assertIsNone(record["eligible_parent_pairs"])
        self.assertEqual(
            record["pair_policy"], "all_unordered_pairs_of_eligible_parents"
        )
        for value in record.values():
            self.assertFalse(isinstance(value, np.ndarray) and value.ndim == 3)


@unittest.skipUnless(
    os.environ.get("PEDIGREE_ELIGIBILITY_BENCHMARK") == "1",
    "set PEDIGREE_ELIGIBILITY_BENCHMARK=1 for fresh-process wall/RSS checks",
)
class FreshProcessScalingBenchmark(unittest.TestCase):
    def test_resolver_panel_and_alternatives_n84_n320(self):
        script = textwrap.dedent(
            """
            import json, resource, sys, time
            import numpy as np
            import pedigree_inference as p

            n = int(sys.argv[1])
            rng = np.random.default_rng(601 + n)
            t0 = time.perf_counter()
            e = p._resolve_parent_eligibility(None, range(n))
            t1 = time.perf_counter()
            scores = rng.normal(size=(n, n))
            np.fill_diagonal(scores, -np.inf)
            trios = p._fixed_trio_panel(scores, 20, 5, True, e)
            t2 = time.perf_counter()
            zero = rng.normal(size=(2, n))
            one = rng.normal(size=(2, n, n))
            for c in range(2): np.fill_diagonal(one[c], -np.inf)
            two = rng.normal(size=(2, len(trios)))
            out = p._parent_state_alternatives(trios, zero, one, two, 0.02, e)
            t3 = time.perf_counter()
            has_cube = any(
                isinstance(getattr(e, f.name), np.ndarray)
                and getattr(e, f.name).ndim == 3
                for f in __import__('dataclasses').fields(e)
            )
            print(json.dumps({
                'n': n, 'resolve_s': t1-t0, 'panel_s': t2-t1,
                'alternatives_s': t3-t2, 'trios': len(trios),
                'alternatives': len(out[0]), 'has_cube': has_cube,
                'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }))
            """
        )
        results = []
        for n_samples in (84, 320):
            completed = subprocess.run(
                [sys.executable, "-c", script, str(n_samples)],
                cwd=Path(__file__).resolve().parent,
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
            results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
        print("eligibility_quadratic_benchmark=" + json.dumps(results, sort_keys=True))
        for result in results:
            self.assertFalse(result["has_cube"])
            self.assertLess(result["resolve_s"], 10.0)
            self.assertLess(result["panel_s"], 30.0)
            self.assertLess(result["alternatives_s"], 30.0)
        # This is a bounded guard, not a noisy microbenchmark ratio assertion.
        self.assertLess(results[1]["maxrss_kib"], 1_500_000)


if __name__ == "__main__":
    unittest.main()
