"""Focused invariants for canonical Stage-1 founder discovery."""

import ast
import inspect
from pathlib import Path

import numpy as np
import numpy.testing as npt
import block_haplotypes
import stage1_validation
from bhd_fit import _fit_at_fixed_K, _prepare_fixed_k_fit_workspace
from bhd_genotype_evidence import allele_depths_to_raw_genotype_likelihoods
from bhd_reversible_cavity import (
    ReversibleCavitySearchConfig,
    search_reversible_cavity,
)
from bhd_reversible_discovery import discover_block_reversible_cavity


def _assert_fit_equal(left, right):
    for index in range(4):
        npt.assert_array_equal(left[index], right[index])
    assert left[4] == right[4]
    assert left[5] == right[5]


def _two_founder_reads(repeats=3):
    haplotypes = np.asarray(
        [[0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 0]],
        dtype=np.int64,
    )
    pairs = ((0, 0), (0, 1), (1, 1)) * repeats
    reads = np.zeros((len(pairs), haplotypes.shape[1], 2), dtype=np.int64)
    for sample, (first, second) in enumerate(pairs):
        dosage = haplotypes[first] + haplotypes[second]
        reads[sample, :, 0] = np.where(
            dosage == 0, 20, np.where(dosage == 1, 10, 0)
        )
        reads[sample, :, 1] = np.where(
            dosage == 2, 20, np.where(dosage == 1, 10, 0)
        )
    return haplotypes, reads


def _small_search_config():
    return ReversibleCavitySearchConfig(
        beam_width=1,
        max_expansions=4,
        max_exact_scores=32,
        max_proposals_per_expansion=16,
        data_start_beam_width=2,
        n_data_seed_modes=2,
        max_candidate_start_rows=4,
        max_replacement_children_per_mode=4,
    )


def test_all_observed_mask_matches_unmasked_fixed_k_reference():
    rng = np.random.default_rng(20260901)
    likelihood = rng.random((14, 37, 3))
    likelihood /= likelihood.sum(axis=2, keepdims=True)
    observed = np.ones((14, 37), dtype=np.bool_)

    for k in (1, 2, 3):
        initial = rng.integers(0, 2, size=(k, 37), dtype=np.int64)
        reference = _fit_at_fixed_K(
            likelihood,
            initial,
            0.5,
            workspace=_prepare_fixed_k_fit_workspace(likelihood, 0.5),
        )
        masked = _fit_at_fixed_K(
            likelihood,
            initial,
            0.5,
            workspace=_prepare_fixed_k_fit_workspace(
                likelihood, 0.5, observed_mask=observed
            ),
        )
        _assert_fit_equal(reference, masked)


def test_wholly_missing_samples_are_ww_and_do_not_move_founders():
    reads = np.zeros((8, 40, 2), dtype=np.int64)
    reads[:3, :, 0] = 12
    reads[3:6, :, 1] = 12
    likelihood = allele_depths_to_raw_genotype_likelihoods(reads)
    observed = reads.sum(axis=2) > 0
    initial = np.vstack((
        np.zeros(40, dtype=np.int64),
        np.ones(40, dtype=np.int64),
    ))

    full = _fit_at_fixed_K(
        likelihood,
        initial,
        0.5,
        workspace=_prepare_fixed_k_fit_workspace(
            likelihood, 0.5, observed_mask=observed
        ),
    )
    active_likelihood = np.ascontiguousarray(likelihood[:6])
    active = _fit_at_fixed_K(
        active_likelihood,
        initial,
        0.5,
        workspace=_prepare_fixed_k_fit_workspace(
            active_likelihood,
            0.5,
            observed_mask=np.ones((6, 40), dtype=np.bool_),
        ),
    )

    npt.assert_array_equal(full[0], active[0])
    npt.assert_array_equal(full[1][:6], active[1])
    npt.assert_array_equal(full[1][6:], np.full((2, 2), 2, dtype=np.int64))
    npt.assert_array_equal(full[2][6:], np.zeros(2))
    npt.assert_array_equal(full[3][6:], np.full(2, 2, dtype=np.int64))
    assert full[5] == active[5]


def test_missing_cells_have_zero_fit_emission_and_cost():
    likelihood = np.full((2, 11, 3), 1.0 / 3.0)
    observed = np.zeros((2, 11), dtype=np.bool_)
    workspace = _prepare_fixed_k_fit_workspace(
        likelihood, 0.5, observed_mask=observed
    )
    npt.assert_array_equal(workspace.log_probs, np.zeros((2, 11, 3)))
    npt.assert_array_equal(workspace.cost_WW, np.zeros((2, 11)))
    npt.assert_array_equal(workspace.WW_total_cost, np.zeros(2))


def test_zero_depth_likelihood_values_cannot_change_full_search():
    haplotypes, reads = _two_founder_reads(repeats=3)
    missing = np.random.default_rng(123).random(reads.shape[:2]) < 0.35
    reads[missing] = 0
    evidence = allele_depths_to_raw_genotype_likelihoods(reads)
    adversarial = evidence.copy()
    missing_cells = np.argwhere(reads.sum(axis=2) == 0)
    for index, (sample, site) in enumerate(missing_cells):
        row = np.full(3, 0.0001)
        row[index % 3] = 0.9998
        adversarial[sample, site] = row

    config = _small_search_config()
    direct = search_reversible_cavity(
        evidence,
        (haplotypes,),
        candidate_haplotypes=haplotypes,
        allele_depths=reads,
        config=config,
    )
    changed = search_reversible_cavity(
        adversarial,
        (haplotypes,),
        candidate_haplotypes=haplotypes,
        allele_depths=reads,
        config=config,
    )

    npt.assert_array_equal(
        direct.selected.mode.haplotypes,
        changed.selected.mode.haplotypes,
    )
    npt.assert_array_equal(
        direct.selected.mode.assignments,
        changed.selected.mode.assignments,
    )
    npt.assert_array_equal(
        direct.selected.mode.wildcard_slots,
        changed.selected.mode.wildcard_slots,
    )
    assert direct.best_score_by_k == changed.best_score_by_k
    assert direct.pseudo_probability_by_k == changed.pseudo_probability_by_k
    assert direct.stop_reason == changed.stop_reason


def test_wholly_missing_samples_cannot_change_reversible_search():
    _haplotypes, active_reads = _two_founder_reads(repeats=3)
    full_reads = np.concatenate((
        active_reads,
        np.zeros((5, active_reads.shape[1], 2), dtype=np.int64),
    ))
    positions = np.arange(active_reads.shape[1], dtype=np.int64)
    config = _small_search_config()

    active_discovery = discover_block_reversible_cavity(
        positions, active_reads, config=config
    )
    full_discovery = discover_block_reversible_cavity(
        positions, full_reads, config=config
    )
    assert active_discovery.selection.n_samples == len(active_reads)
    assert full_discovery.selection.n_samples == len(active_reads)
    active = active_discovery.to_block_result()
    full = full_discovery.to_block_result()

    assert active.K_final == full.K_final
    npt.assert_array_equal(
        active.cavity_selected_mode.haplotypes,
        full.cavity_selected_mode.haplotypes,
    )
    npt.assert_array_equal(
        active.pair_assignments,
        full.pair_assignments[: len(active_reads)],
    )
    npt.assert_array_equal(active.discrete_haps, full.discrete_haps)
    npt.assert_array_equal(
        full.sample_has_observed_kept_depth[-5:],
        np.zeros(5, dtype=np.bool_),
    )



def test_one_informative_sample_uses_nonclustered_seed_fallback():
    reads = np.zeros((8, 24, 2), dtype=np.int64)
    alleles = np.arange(24) % 2
    reads[0, :, 0] = np.where(alleles == 0, 12, 0)
    reads[0, :, 1] = np.where(alleles == 1, 12, 0)

    discovery = discover_block_reversible_cavity(
        np.arange(24, dtype=np.int64),
        reads,
        config=_small_search_config(),
    )
    result = discovery.to_block_result()
    assert discovery.selection.n_samples == 1
    assert result.K_final >= 1
    npt.assert_array_equal(
        result.sample_has_observed_kept_depth,
        [True, False, False, False, False, False, False, False],
    )


def test_validation_identity_covers_local_stage1_dependency_closure():
    root = Path(__file__).resolve().parent
    pending = {
        "block_haplotypes",
        "bhd_reversible_cavity",
        "bhd_reversible_discovery",
    }
    closure = set()
    while pending:
        module = pending.pop()
        if module in closure:
            continue
        path = root / f"{module}.py"
        if not path.is_file():
            continue
        closure.add(module)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(
                    alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        pending.update(
            name for name in imports if (root / f"{name}.py").is_file()
        )

    identity_modules = {
        Path(filename).stem
        for filename in stage1_validation.PRODUCTION_FILES
    }
    assert closure <= identity_modules


def test_canonical_configuration_exposes_the_calibrated_model():
    config = ReversibleCavitySearchConfig()
    assert config.min_directional_supporters == 2
    assert config.min_hard_call_pseudo_probability == 0.85
    assert set(config.__dataclass_fields__) == {
        "beam_width",
        "max_expansions",
        "max_exact_scores",
        "max_proposals_per_expansion",
        "data_start_beam_width",
        "n_data_seed_modes",
        "max_candidate_start_rows",
        "max_replacement_children_per_mode",
        "lambda_wildcard_penalty",
        "read_error_probability",
        "min_soft_unique_sample_support",
        "min_directional_supporters",
        "min_hard_call_pseudo_probability",
        "coordinate_descent_max_iter",
        "soft_seed_min_cluster_size",
        "apply_gauge_rewire",
        "exact_cut_max_k",
        "max_cut_ties",
        "score_tolerance",
        "cavity",
    }


def test_public_orchestrator_has_only_canonical_stage1_dispatch():
    parameters = inspect.signature(
        block_haplotypes.generate_all_block_haplotypes
    ).parameters
    assert tuple(parameters) == (
        "genomic_data",
        "num_processes",
        "discard_reads_after",
        "total_numba_threads",
        "block_pool",
        "discovery_config",
    )
    assert set(block_haplotypes.__all__) == {
        "BlockDiscoveryPool",
        "BlockResult",
        "BlockResults",
        "ReversibleCavitySearchConfig",
        "STAGE1_BACKEND",
        "generate_all_block_haplotypes",
    }
    assert block_haplotypes._selftest_stage1_orchestration() == {
        "canonical_dispatch": "pass",
        "degenerate_blocks_omitted": "pass",
        "exact_materialization_forwarding": "pass",
    }
