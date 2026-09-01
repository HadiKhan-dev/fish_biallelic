"""Focused known-truth checks for rare-founder Stage-1 validation."""

from dataclasses import replace

import numpy as np
import numpy.testing as npt

from stage1_validation import (
    SimulationConfig,
    TaskSpec,
    _argument_parser,
    _build_tasks,
    _evaluate,
    _make_diplotypes,
    _run_inference,
    _simulate_task,
    _simulation_id,
)


def _rare_config(carriers=3, depth=8.0):
    return SimulationConfig(
        n_samples=30,
        n_founders=4,
        mean_depth=depth,
        rare_founder_carriers=carriers,
    )


def test_rare_diplotypes_have_exact_carrier_count_and_common_coverage():
    config = _rare_config(carriers=3)
    first = _make_diplotypes(np.random.default_rng(123), config)
    second = _make_diplotypes(np.random.default_rng(123), config)

    npt.assert_array_equal(first, second)
    assert int(np.sum(np.any(first == 0, axis=1))) == 3
    assert not np.any(np.all(first == 0, axis=1))
    assert set(np.unique(first)) == {0, 1, 2, 3}


def test_targeted_blackout_only_masks_the_rare_carriers_and_pairs_inputs():
    task = TaskSpec(
        pattern="rare_founder_blackout",
        missing_rate=0.30,
        seed=41,
        simulation=_rare_config(carriers=3),
    )
    alternate_search = replace(
        task, search_overrides=(("max_expansions", 8),)
    )
    first = _simulate_task(task)
    second = _simulate_task(alternate_search)

    metadata = first["metadata"]
    start = int(metadata["tract_start"])
    length = int(metadata["tract_length"])
    expected = np.zeros_like(first["deliberate_missing"])
    expected[first["rare_carriers"], start : start + length] = True
    npt.assert_array_equal(first["deliberate_missing"], expected)
    assert int(np.sum(first["rare_carriers"])) == 3
    assert _simulation_id(task) == _simulation_id(alternate_search)
    assert (
        first["metadata"]["observable_input_sha256"]
        == second["metadata"]["observable_input_sha256"]
    )
    npt.assert_array_equal(first["reads"], second["reads"])


def test_rare_target_metrics_are_post_inference_and_support_stratified():
    overrides = tuple(sorted({
        "beam_width": 1,
        "max_expansions": 4,
        "max_exact_scores": 16,
        "max_proposals_per_expansion": 24,
        "data_start_beam_width": 2,
        "n_data_seed_modes": 2,
        "max_candidate_start_rows": 8,
        "max_replacement_children_per_mode": 4,
        "min_directional_supporters": 2,
        "min_hard_call_pseudo_probability": 0.85,
    }.items()))
    task = TaskSpec(
        pattern="none",
        missing_rate=0.0,
        seed=52,
        simulation=_rare_config(carriers=3, depth=10.0),
        search_overrides=overrides,
    )
    simulation = _simulate_task(task)
    result = _run_inference(
        task,
        simulation["positions"],
        simulation["reads"],
        simulation["keep_flags"],
    )
    evaluated = _evaluate(task, simulation, result)
    target = evaluated["rare_target"]

    assert target is not None
    assert target["truth_index"] == 0
    assert target["carrier_sample_count"] == 3
    assert sum(
        stratum["eligible"]
        for stratum in evaluated["rare_target_support_strata"].values()
    ) == 200
    assert "rare_target_recovered" in evaluated["metrics"]
    assert "rare_target_noncarrier_false_target_assignment_rate" in evaluated["metrics"]


def test_cli_wires_rare_founder_and_calibrated_release_default():
    arguments = _argument_parser().parse_args([
        "--output-root",
        "/tmp/stage1-rare-cli-test",
        "--patterns",
        "rare_founder_blackout",
        "--rates",
        "0.3",
        "--seeds",
        "0:1",
        "--rare-founder-carriers",
        "2",
    ])
    tasks = _build_tasks(arguments)

    assert arguments.min_hard_call_pseudo_probability == 0.85
    assert len(tasks) == 1
    assert all(task.simulation.rare_founder_carriers == 2 for task in tasks)
    assert all(task.pattern == "rare_founder_blackout" for task in tasks)
