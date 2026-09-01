"""Stage-1 checkpoint identity and unknown-allele round-trip tests."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

import block_haplotypes
import pipeline_runtime
import pipeline_tropheops


def test_tropheops_uses_one_fixed_stage1_identity():
    assert pipeline_tropheops.STAGE1_BACKEND == block_haplotypes.STAGE1_BACKEND
    assert pipeline_tropheops.CHECKPOINT_DIR == (
        ".pipeline_checkpoints_tropheops_withFounders_"
        "reversible_cavity_depth_observation_v1"
    )
    config = block_haplotypes.ReversibleCavitySearchConfig()
    assert config.min_directional_supporters == 2
    assert config.min_hard_call_pseudo_probability == 0.85



def test_entry_points_stop_before_unknown_alleles_are_hardened():
    root = Path(__file__).resolve().parent
    for filename in (
        "pipeline.py",
        "pipeline_real.py",
        "pipeline_tropheops.py",
        "pedigree_sim_pipeline.py",
    ):
        source = (root / filename).read_text(encoding="utf-8")
        assert "preserve unknown founder alleles" in source
        assert "raise SystemExit(0)" in source

    sweep_source = (root / "pedigree_depth_sweep.py").read_text(
        encoding="utf-8"
    )
    assert "unavailable in the current Stage-1-only release" in sweep_source


def test_checkpoint_identity_rejects_mismatch_and_unbound_outputs(tmp_path):
    identity = {
        "backend": block_haplotypes.STAGE1_BACKEND,
        "config": asdict(block_haplotypes.ReversibleCavitySearchConfig()),
    }
    stage = "01_founder_discovery"
    bound = pipeline_runtime.CheckpointStore(tmp_path / "bound", nthreads=1)
    bound.bind_stage_identity(stage, identity)
    bound.save_contig(stage, "chr1", {"partial": True})
    bound.bind_stage_identity(stage, identity)
    bound.mark_stage_complete(stage)
    bound.bind_stage_identity(stage, identity)

    concurrent = pipeline_runtime.CheckpointStore(
        tmp_path / "concurrent", nthreads=1
    )
    with ThreadPoolExecutor(max_workers=8) as workers:
        futures = [
            workers.submit(concurrent.bind_stage_identity, stage, identity)
            for _ in range(16)
        ]
        for future in futures:
            future.result()

    linked = pipeline_runtime.CheckpointStore(tmp_path / "linked", nthreads=1)
    linked_stage = Path(linked.root) / stage
    linked_stage.symlink_to(Path(bound.root) / stage, target_is_directory=True)
    linked.bind_stage_identity(stage, identity)

    try:
        bound.bind_stage_identity(
            "01_founder_discovery",
            {**identity, "backend": "different"},
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("mismatched Stage-1 identity was accepted")

    unbound = pipeline_runtime.CheckpointStore(
        tmp_path / "unbound", nthreads=1
    )
    unbound.save_contig(
        "01_founder_discovery", "chr1", {"unbound": True}
    )
    try:
        unbound.bind_stage_identity("01_founder_discovery", identity)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unbound existing Stage-1 output was accepted")

def test_checkpoint_root_isolation_and_unknown_round_trip(tmp_path):
    unrelated = pipeline_runtime.CheckpointStore(
        tmp_path / "unrelated", nthreads=1
    )
    current = pipeline_runtime.CheckpointStore(
        tmp_path / "current", nthreads=1
    )
    stage = "01_founder_discovery"
    contig = "chr1"
    unrelated.save_contig(stage, contig, {"marker": "unrelated"})
    assert unrelated.contig_done(stage, contig)
    assert not current.contig_done(stage, contig)

    positions = np.asarray([10, 20, 30], dtype=np.int64)
    haplotypes = {
        0: np.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
    }
    block = block_haplotypes.BlockResult(
        positions,
        haplotypes,
        keep_flags=np.ones(3, dtype=np.int64),
    )
    block.discrete_haps = np.asarray([[0, -1, 1]], dtype=np.int64)
    config_record = asdict(block_haplotypes.ReversibleCavitySearchConfig())
    current.save_contig(stage, contig, {
        "stage1_backend": block_haplotypes.STAGE1_BACKEND,
        "stage1_config": config_record,
        "block_results": block_haplotypes.BlockResults([block]),
    })
    restored = current.load_contig(stage, contig)
    assert restored["stage1_backend"] == block_haplotypes.STAGE1_BACKEND
    assert restored["stage1_config"] == config_record
    np.testing.assert_array_equal(
        restored["block_results"][0].discrete_haps,
        [[0, -1, 1]],
    )
    np.testing.assert_array_equal(
        restored["block_results"][0].haplotypes[0][1],
        [0.5, 0.5],
    )
