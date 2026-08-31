"""Focused contracts for real-data pedigree integration policies."""

import dataclasses
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import pedigree_pipeline as integration


class TropheopsParentEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.sample_ids = (
            "g0f", "g0m", "f1f0", "f1f1", "f1m0", "f1m1",
            "f2a", "f2b",
        )
        self.metadata = pd.DataFrame({
            "primary_ID": self.sample_ids,
            "generation": ("G0", "G0", "F1", "F1", "F1", "F1", "F2", "F2"),
            "sex": ("F", "M", "F", "F", "M", "M", "M", "M"),
        })

    def test_f2_targets_f1_parents_and_opposite_sex_pairs(self):
        record = integration.build_tropheops_parent_eligibility(
            self.metadata, self.sample_ids
        )
        children = record["eligible_children"]
        parents = record["eligible_parents"]
        pairs = record["eligible_parent_pairs"]

        np.testing.assert_array_equal(
            np.flatnonzero(children), np.asarray((6, 7))
        )
        for child in (6, 7):
            np.testing.assert_array_equal(
                np.flatnonzero(parents[child]), np.asarray((2, 3, 4, 5))
            )
            observed = {
                (first, second)
                for first, second in zip(*np.nonzero(np.triu(pairs[child], 1)))
            }
            self.assertEqual(observed, {(2, 4), (2, 5), (3, 4), (3, 5)})
        self.assertFalse(np.any(parents[:, :2]))
        self.assertFalse(np.any(parents[:, 6:]))

        summary = integration.summarize_parent_eligibility(record)
        self.assertEqual(summary["eligible_child_count"], 2)
        self.assertEqual(summary["candidate_parent_sample_count"], 4)
        self.assertEqual(summary["minimum_parent_candidates_per_target"], 4)
        self.assertEqual(summary["maximum_parent_pairs_per_target"], 4)
        self.assertFalse(summary["individual_parentage_ground_truth"])

    def test_optional_pair_policy_allows_every_unordered_f1_pair(self):
        record = integration.build_tropheops_parent_eligibility(
            self.metadata,
            self.sample_ids,
            require_opposite_sex_pair=False,
        )
        for child in (6, 7):
            self.assertEqual(
                int(np.count_nonzero(np.triu(
                    record["eligible_parent_pairs"][child], 1
                ))),
                6,
            )

    def test_eligibility_identity_changes_with_scientific_policy(self):
        record = integration.build_tropheops_parent_eligibility(
            self.metadata, self.sample_ids
        )
        observed = integration.parent_eligibility_identity(record)
        copied = dict(record)
        copied["eligible_parent_pairs"] = record[
            "eligible_parent_pairs"
        ].copy()
        self.assertEqual(
            observed, integration.parent_eligibility_identity(copied)
        )
        copied["eligible_parent_pairs"][6, 2, 4] = False
        copied["eligible_parent_pairs"][6, 4, 2] = False
        self.assertNotEqual(
            observed, integration.parent_eligibility_identity(copied)
        )

    def test_metadata_alignment_is_exact(self):
        duplicated = pd.concat(
            (self.metadata, self.metadata.iloc[[0]]), ignore_index=True
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            integration.build_tropheops_parent_eligibility(
                duplicated, self.sample_ids
            )

        missing = self.metadata.iloc[:-1]
        with self.assertRaisesRegex(ValueError, "missing"):
            integration.build_tropheops_parent_eligibility(
                missing, self.sample_ids
            )


class AsAcParentEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.sample_ids = (
            "g0f", "g0m", "f1f", "f1m",
            "f2a", "f2b", "f3a", "f3b",
        )
        self.metadata = pd.DataFrame({
            "primary_ID": self.sample_ids,
            "generation": (
                "G0; species A", "G0; species B", "F1", "F1",
                "F2", "F2", "F3", "F3",
            ),
            "sex": ("F", "M", "F", "M", "M", "M", "M", "M"),
        })

    def test_generation_step_candidates_and_sex_constrained_pairs(self):
        record = integration.build_asac_parent_eligibility(
            self.metadata, self.sample_ids
        )
        children = record["eligible_children"]
        parents = record["eligible_parents"]
        pairs = record["eligible_parent_pairs"]
        np.testing.assert_array_equal(
            np.flatnonzero(children), np.asarray((4, 5, 6, 7))
        )
        for child in (4, 5):
            np.testing.assert_array_equal(
                np.flatnonzero(parents[child]), np.asarray((2, 3))
            )
            self.assertEqual(
                int(np.count_nonzero(np.triu(pairs[child], 1))), 1
            )
        for child in (6, 7):
            np.testing.assert_array_equal(
                np.flatnonzero(parents[child]), np.asarray((4, 5))
            )
            self.assertEqual(
                int(np.count_nonzero(np.triu(pairs[child], 1))), 0
            )
        self.assertFalse(np.any(parents[:, :2]))
        self.assertEqual(
            record["policy_name"], integration.ASAC_GENERATION_STEP_POLICY
        )
        self.assertFalse(record["individual_parentage_ground_truth"])

    def test_no_sex_policy_is_explicit_sensitivity_universe(self):
        record = integration.build_asac_parent_eligibility(
            self.metadata, self.sample_ids,
            require_opposite_sex_pair=False,
        )
        for child in (4, 5, 6, 7):
            self.assertEqual(
                int(np.count_nonzero(np.triu(
                    record["eligible_parent_pairs"][child], 1
                ))),
                1,
            )
        self.assertEqual(
            record["policy_name"],
            integration.ASAC_GENERATION_STEP_NO_SEX_POLICY,
        )

    def test_unknown_generation_or_sex_is_rejected(self):
        invalid_generation = self.metadata.copy()
        invalid_generation.loc[7, "generation"] = "Unknown"
        with self.assertRaisesRegex(ValueError, "generation"):
            integration.build_asac_parent_eligibility(
                invalid_generation, self.sample_ids
            )
        invalid_sex = self.metadata.copy()
        invalid_sex.loc[7, "sex"] = ""
        with self.assertRaisesRegex(ValueError, "sex"):
            integration.build_asac_parent_eligibility(
                invalid_sex, self.sample_ids
            )


class ImplicitParentPairPolicyTests(unittest.TestCase):
    def _record(self):
        return {
            "format_version": 1,
            "policy_name": "unrestricted_test_v1",
            "sample_ids": ("a", "b", "c", "d", "e"),
            "eligible_children": np.asarray(
                (False, False, True, True, False), dtype=np.bool_
            ),
            "eligible_parents": np.asarray(
                (
                    (False, False, False, False, False),
                    (False, False, False, False, False),
                    (True, True, False, True, False),
                    (True, True, True, False, True),
                    (True, True, True, True, False),
                ),
                dtype=np.bool_,
            ),
            "eligible_parent_pairs": None,
            "source_fields": (),
            "assumptions": (
                "Every eligible unordered pair is admitted.",
            ),
            "individual_parentage_ground_truth": False,
        }

    def test_summary_uses_each_child_specific_parent_row(self):
        summary = integration.summarize_parent_eligibility(self._record())
        self.assertEqual(summary["eligible_child_count"], 2)
        self.assertEqual(summary["candidate_parent_sample_count"], 5)
        self.assertEqual(summary["minimum_parent_candidates_per_target"], 3)
        self.assertEqual(summary["maximum_parent_candidates_per_target"], 4)
        self.assertEqual(summary["minimum_parent_pairs_per_target"], 3)
        self.assertEqual(summary["maximum_parent_pairs_per_target"], 6)

    def test_excluded_children_do_not_affect_target_minima(self):
        record = self._record()
        # The first two excluded samples have no candidate parents. They must
        # not lower target-only minima to zero.
        self.assertFalse(record["eligible_children"][0])
        self.assertEqual(
            integration.summarize_parent_eligibility(record)[
                "minimum_parent_pairs_per_target"
            ],
            3,
        )

    def test_identity_binds_rule_masks_and_sample_order(self):
        record = self._record()
        observed = integration.parent_eligibility_identity(record)
        self.assertEqual(
            observed, integration.parent_eligibility_identity(dict(record))
        )

        changed_row = dict(record)
        changed_row["eligible_parents"] = record[
            "eligible_parents"
        ].copy()
        changed_row["eligible_parents"][2, 4] = True
        self.assertNotEqual(
            observed, integration.parent_eligibility_identity(changed_row)
        )

        reordered = dict(record)
        reordered["sample_ids"] = ("b", "a", "c", "d", "e")
        self.assertNotEqual(
            observed, integration.parent_eligibility_identity(reordered)
        )

    def test_none_policy_does_not_enter_dense_cube_counter(self):
        record = self._record()
        with mock.patch.object(
            integration,
            "_upper_triangle_counts",
            side_effect=AssertionError("dense pair cube path used"),
        ):
            summary = integration.summarize_parent_eligibility(record)
            identity = integration.parent_eligibility_identity(record)
        self.assertEqual(summary["maximum_parent_pairs_per_target"], 6)
        self.assertEqual(len(identity), 64)

    def test_dense_identity_and_summary_remain_exact(self):
        sample_ids = (
            "g0f", "g0m", "f1f0", "f1f1", "f1m0", "f1m1", "f2a", "f2b",
        )
        metadata = pd.DataFrame({
            "primary_ID": sample_ids,
            "generation": (
                "G0", "G0", "F1", "F1", "F1", "F1", "F2", "F2",
            ),
            "sex": ("F", "M", "F", "F", "M", "M", "M", "M"),
        })
        record = integration.build_tropheops_parent_eligibility(
            metadata, sample_ids
        )
        self.assertEqual(
            integration.parent_eligibility_identity(record),
            "c9b9d0088a6a813f5a1450ace6be7dd8a11b7d10c6783f59ae15b65e6566cc27",
        )
        summary = integration.summarize_parent_eligibility(record)
        self.assertEqual(summary["eligible_child_count"], 2)
        self.assertEqual(summary["candidate_parent_sample_count"], 4)
        self.assertEqual(summary["minimum_parent_candidates_per_target"], 4)
        self.assertEqual(summary["maximum_parent_candidates_per_target"], 4)
        self.assertEqual(summary["minimum_parent_pairs_per_target"], 4)
        self.assertEqual(summary["maximum_parent_pairs_per_target"], 4)


class CurrentPedigreeReleaseTests(unittest.TestCase):
    def test_current_config_has_every_locked_calibrated_field(self):
        config = integration.build_current_pedigree_config(
            bootstrap_replicates=7
        )
        locked = {
            "bootstrap_replicates": 7,
            "primary_view": "tier_b",
            "parent_state_algorithm_mode": "b1",
            "parent_state_structure_mode": "combined_v1",
            "parent_state_candidate_source_mode": "hard_painted",
            "parent_state_effective_markers_per_information_block": 3.0,
            "parent_state_minimum_edge_coverage": 0.95,
            "parent_state_minimum_pair_explainability": 0.95,
            "parent_state_minimum_edge_exposed_bins": 1.0,
            "parent_state_minimum_pair_exposed_bins": 1.0,
            "parent_state_minimum_exposed_fraction": 0.10,
            "parent_state_minimum_exposed_contigs": 3,
            "parent_state_minimum_direction_probability": 0.01,
        }
        for field, expected in locked.items():
            with self.subTest(field=field):
                self.assertEqual(getattr(config, field), expected)

    def test_checkpoint_and_schema_identities_are_exact(self):
        raw_input, inference, downstream = integration.pedigree_stage_names()
        self.assertEqual(raw_input, "T10a_smart_raw_gl_inputs_v1")
        self.assertEqual(inference, "T10b_smart_raw_gl_hard_painted_b1_combined_v1_calibrated_v1")
        self.assertEqual(downstream, "T11_smart_raw_gl_hard_painted_b1_combined_v1_calibrated_v1_phase_correction_v1")
        self.assertEqual(integration.PEDIGREE_BACKEND, "smart_raw_gl_parent_state_v1")
        self.assertEqual(integration.PARENT_ELIGIBILITY_FORMAT_VERSION, 1)
        self.assertEqual(integration.PEDIGREE_RAW_GL_INPUT_SCHEMA_VERSION, 1)
        self.assertEqual(integration.PEDIGREE_T10_PAYLOAD_SCHEMA_VERSION, 2)
        self.assertEqual(integration.PEDIGREE_REAL_PAYLOAD_SCHEMA_VERSION, 2)
        self.assertEqual(integration.TROPHEOPS_PEDIGREE_BASELINE, "hard_painted_b1_combined_v1_calibrated_v1")
        self.assertEqual(integration.ASAC_PEDIGREE_ENGINE_ID, "parent_state_hard_painted_b1_combined_v1_calibrated_asac_policy_v1")

    def test_method_identities_are_fixed_and_not_selectable(self):
        config = integration.build_current_pedigree_config()
        config_fields = {field.name: field for field in dataclasses.fields(config)}
        self.assertEqual(config.parent_state_algorithm_mode, "b1")
        self.assertEqual(config.parent_state_structure_mode, "combined_v1")
        self.assertFalse(config_fields["parent_state_algorithm_mode"].init)
        self.assertFalse(config_fields["parent_state_structure_mode"].init)
        retired_builder = "build_tropheops_" + "hard_b1_config"
        self.assertFalse(hasattr(integration, retired_builder))

    def test_all_pipeline_stage_identities_and_current_builder_usages(self):
        root = Path(__file__).resolve().parent
        expected = {
            "pipeline_tropheops.py": (
                'PEDIGREE_BASELINE = "hard_painted_b1_combined_v1_calibrated_v1"',
                "pedigree_pipeline.build_current_pedigree_config()",
            ),
            "pipeline_real.py": (
                'STAGE_R10 = "R10_pedigree_inference_current_b1_combined_v1_calibrated_asac_policy_v1"',
                'STAGE_R11 = "R11_phase_correction_current_b1_combined_v1_calibrated_asac_policy_v1"',
                "pedigree_pipeline.build_current_pedigree_config()",
            ),
            "pipeline.py": (
                'STAGE_12 = "12_pedigree_inference_current_b1_combined_v1_calibrated_v1"',
                'STAGE_13 = "13_phase_correction_current_b1_combined_v1_calibrated_v1"',
                "pedigree_pipeline.build_current_pedigree_config()",
            ),
            "pedigree_sim_pipeline.py": (
                'STAGE_11 = "11_pedigree_inference_current_b1_combined_v1_calibrated_v1"',
                "pedigree_pipeline.build_current_pedigree_config()",
            ),
            "pedigree_depth_sweep.py": (
                'STAGE11_NAME = "11_pedigree_inference_current_b1_combined_v1_calibrated_v1"',
            ),
        }
        for filename, tokens in expected.items():
            source = (root / filename).read_text(encoding="utf-8")
            for token in tokens:
                with self.subTest(filename=filename, token=token):
                    self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
