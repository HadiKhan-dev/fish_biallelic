"""Regression tests for current pedigree truth evaluation and consumers."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pedigree_evaluation import compare_relationships_to_truth
from plot_depth_accuracy import collect
from recombination_map import read_inferred_pedigree


def _truth_and_inferred():
    truth = pd.DataFrame({
        "Sample": ["a", "b", "c", "d"],
        "Generation": ["F1", "F1", "F2", "F2"],
        "Parent1": ["Founder0", "Founder2", "a", "a"],
        "Parent2": ["Founder1", "Founder3", "Founder4", "b"],
    })
    inferred = pd.DataFrame({
        "Sample": ["a", "b", "c", "d"],
        "Generation": ["Unknown"] * 4,
        "ParentState": [
            "zero_observed_parents", "zero_observed_parents",
            "one_observed_parent", "two_observed_parents",
        ],
        "Parent1": [None, None, "a", "b"],
        "Parent2": [None, None, None, "a"],
    })
    return truth, inferred


class CurrentPedigreeEvaluationTests(unittest.TestCase):
    def test_truth_states_and_observed_parent_sets_include_m1(self):
        truth, inferred = _truth_and_inferred()
        comparison = compare_relationships_to_truth(truth, inferred)
        self.assertEqual(
            comparison["ParentState_True"].tolist(),
            [
                "zero_observed_parents", "zero_observed_parents",
                "one_observed_parent", "two_observed_parents",
            ],
        )
        self.assertTrue(comparison["ParentState_Match"].all())
        self.assertTrue(comparison["Parents_Match"].all())

    def test_depth_collection_uses_only_current_scientific_filename(self):
        truth, inferred = _truth_and_inferred()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "depth3" / "seed7" / "results"
            results.mkdir(parents=True)
            truth.to_csv(results / "ground_truth_pedigree.csv", index=False)
            inferred.to_csv(
                results / "pedigree_inference_current_scientific.csv",
                index=False,
            )
            pd.DataFrame({"retired": [True]}).to_csv(
                results / "pedigree_inference_discovered.csv", index=False
            )
            observed = collect(root)
        self.assertEqual(observed["parentage"].tolist(), [100.0])
        self.assertEqual(observed["parent_state"].tolist(), [100.0])

    def test_recombination_map_uses_only_current_scientific_filename(self):
        _, inferred = _truth_and_inferred()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_root = root / "checkpoints"
            checkpoint_root.mkdir()
            results = root / "results_simulation"
            results.mkdir()
            inferred.to_csv(
                results / "pedigree_inference_current_scientific.csv",
                index=False,
            )
            pd.DataFrame({
                "Sample": ["retired"], "Parent1": [None], "Parent2": [None]
            }).to_csv(
                results / "pedigree_inference_discovered.csv", index=False
            )
            observed = read_inferred_pedigree(checkpoint_root)
        self.assertEqual(observed["Sample"].tolist(), inferred["Sample"].tolist())


if __name__ == "__main__":
    unittest.main()
