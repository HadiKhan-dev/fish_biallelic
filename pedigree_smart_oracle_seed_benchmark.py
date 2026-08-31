#!/usr/bin/env python3
"""Truth-painted combined-v1 regression for the retained seed-100/101 pedigrees.

This is deliberately an *oracle* pedigree-engine benchmark.  It recreates the
simulated ancestry from the retained six-founder Stage-1 panel and gives Smart
the true founder-label paintings, thereby isolating pedigree inference from
founder discovery and sample-painting errors.  The retained artifacts prove
topology equivalence, not bitwise identity of old physical breakpoints or
ordered tracks.  This script does not regenerate reads or any downstream
pipeline stage and is not a production-data pathway.
"""

from __future__ import annotations

import thread_config  # noqa: F401  (must precede NumPy/Numba imports)

import argparse
import ast
import dataclasses
import gc
import hashlib
import importlib
import json
import os
import pickle
import resource
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import blosc2
import numpy as np
import pandas as pd

import checkpoint_io
import pedigree_inference as smart
import pipeline_runtime
import simulate_sequences as current_simulate_sequences
from bhd_results import BlockResult
from pedigree_evaluation import parent_columns_match


CONTIGS = tuple(
    [f"chr{index}" for index in range(1, 21)] + ["chr22", "chr23"]
)
TARGET_SEEDS = (100, 101)
LEGACY_MAGIC = b"BHDB2CK1"
SCHEMA_VERSION = 2
GENERATION_SIZES = (20, 100, 200)
RECOMBINATION_RATE = 5e-8
MUTATION_RATE = 1e-10
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260725

DEFAULT_STAGE1_ROOT = Path(
    "results_simulation/hmm_joint_fresh_20260813/"
    "seed73_k6_depth5_constant/checkpoints/01_vcf_discovery"
)
DEFAULT_OUTPUT_ROOT = Path(
    "results_simulation/pedigree_smart_oracle_seed100_101_hard_b1_combined_v1_v2"
)


def _read_legacy_checkpoint(path: Path, nthreads: int = 1) -> Any:
    """Read one bounded historical BHDB2CK1 payload locally."""
    with path.open("rb") as handle:
        if handle.read(8) != LEGACY_MAGIC:
            raise ValueError(f"{path}: not a BHDB2CK1 checkpoint")
        raw_count = handle.read(8)
        if len(raw_count) != 8:
            raise ValueError(f"{path}: truncated chunk count")
        count, = struct.unpack("<Q", raw_count)
        parts = []
        for _ in range(count):
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                raise ValueError(f"{path}: truncated segment size")
            size, = struct.unpack("<Q", raw_size)
            blob = handle.read(size)
            if len(blob) != size:
                raise ValueError(f"{path}: truncated segment")
            parts.append(blosc2.decompress2(
                blob,
                dparams=blosc2.DParams(nthreads=max(1, int(nthreads))),
            ))
        if handle.read(1):
            raise ValueError(f"{path}: trailing bytes")
    if not parts:
        raise ValueError(f"{path}: empty checkpoint")
    return pickle.loads(parts[0] if len(parts) == 1 else b"".join(parts))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _file_identity(path: Path, *, content_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()
    result = {
        "path": os.fspath(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash:
        result["sha256"] = _sha256(path)
    return result


def _legacy_stage1_path(root: Path, contig: str) -> Path:
    return root / f"{contig}.pkl.b2"


def _load_simulator(source: Path | None) -> Any:
    if source is None:
        return current_simulate_sequences
    resolved = source.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if not resolved.stem.isidentifier():
        raise ValueError("simulator source filename must be a Python identifier")
    source_directory = os.fspath(resolved.parent)
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)
    module = importlib.import_module(resolved.stem)
    if Path(module.__file__).resolve() != resolved:
        raise ValueError("simulator module resolved to the wrong source file")
    required = (
        "concretify_haps", "pairup_haps", "simulate_pedigree",
        "convert_truth_to_painting_objects",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ValueError("simulator source lacks: " + ", ".join(missing))
    return module


def _validate_stage1_payload(
    payload: Any, contig: str
) -> tuple[np.ndarray, list[np.ndarray]]:
    if not isinstance(payload, dict) or set(payload) != {"naive_long_haps"}:
        raise ValueError(f"{contig}: unexpected Stage-1 payload schema")
    value = payload["naive_long_haps"]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{contig}: malformed naive_long_haps")
    positions = np.asarray(value[0])
    probability_haplotypes = list(value[1])
    if (
        positions.ndim != 1
        or len(positions) == 0
        or not np.issubdtype(positions.dtype, np.integer)
        or (len(positions) > 1 and np.any(np.diff(positions) <= 0))
    ):
        raise ValueError(f"{contig}: invalid marker coordinates")
    if len(probability_haplotypes) != 6:
        raise ValueError(f"{contig}: expected six founder haplotypes")
    checked = []
    for index, raw in enumerate(probability_haplotypes):
        haplotype = np.asarray(raw)
        if (
            haplotype.shape != (len(positions), 2)
            or np.any(~np.isfinite(haplotype))
            or np.any(haplotype < 0.0)
            or not np.allclose(
                haplotype.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
            )
        ):
            raise ValueError(f"{contig}: invalid founder haplotype {index}")
        checked.append(np.ascontiguousarray(haplotype, dtype=np.float64))
    return (
        np.ascontiguousarray(positions, dtype=np.int64),
        checked,
    )


def _founder_block(
    positions: np.ndarray, probability_haplotypes: list[np.ndarray]
) -> BlockResult:
    return pipeline_runtime.compact_founder_block(BlockResult(
        positions=positions,
        haplotypes={
            index: haplotype
            for index, haplotype in enumerate(probability_haplotypes)
        },
        keep_flags=np.ones(len(positions), dtype=np.bool_),
        reads_count_matrix=None,
        probs_array=None,
        genotype_evidence_mode=None,
    ))


def _topology_from_chunks(chunks: Iterable[Any]) -> list[tuple[int, int]]:
    topology: list[tuple[int, int]] = []
    for chunk in chunks or ():
        first, second = int(chunk.hap1), int(chunk.hap2)
        pair = (first, second) if first <= second else (second, first)
        if not topology or topology[-1] != pair:
            topology.append(pair)
    return topology


def _expected_truth(seed: int, historical_root: Path) -> pd.DataFrame:
    path = historical_root / f"seed{seed}_linear_h1" / "ground_truth_pedigree.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _expected_topology(seed: int, historical_root: Path) -> pd.DataFrame:
    path = (
        historical_root / f"seed{seed}_linear_h1"
        / "paint_samples_topology_evaluation.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "Sample", "Contig", "N_chunks_truth", "N_topology_truth",
        "Topology_truth",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: missing replay-validation columns")
    if len(frame) != 320 * len(CONTIGS):
        raise ValueError(f"{path}: unexpected row count {len(frame)}")
    if frame.duplicated(["Sample", "Contig"]).any():
        raise ValueError(f"{path}: duplicate Sample/Contig rows")
    return frame.set_index(["Sample", "Contig"], verify_integrity=True)


def _require_exact_truth_pedigree(
    replayed: pd.DataFrame, expected: pd.DataFrame, seed: int
) -> None:
    columns = ["Sample", "Generation", "Parent1", "Parent2"]
    if list(replayed.columns) != columns or list(expected.columns) != columns:
        raise ValueError(f"seed {seed}: unexpected truth-pedigree schema")
    if not replayed.reset_index(drop=True).equals(expected.reset_index(drop=True)):
        merged = replayed.merge(
            expected, on="Sample", how="outer", suffixes=("_Replay", "_Saved"),
            indicator=True,
        )
        mismatch = merged.loc[
            (merged["_merge"] != "both")
            | (merged["Generation_Replay"] != merged["Generation_Saved"])
            | (merged["Parent1_Replay"] != merged["Parent1_Saved"])
            | (merged["Parent2_Replay"] != merged["Parent2_Saved"])
        ]
        raise ValueError(
            f"seed {seed}: replayed truth pedigree differs in "
            f"{len(mismatch)} rows"
        )


def _validate_replayed_painting(
    painting: Any,
    contig: str,
    sample_ids: tuple[str, ...],
    expected: pd.DataFrame,
) -> dict[str, Any]:
    samples = tuple(getattr(painting, "samples", ()))
    if len(samples) != len(sample_ids):
        raise ValueError(f"{contig}: truth painting/sample length mismatch")
    topology_mismatches = 0
    chunk_count_mismatches = 0
    topology_count_mismatches = 0
    for sample_index, (sample, sample_id) in enumerate(zip(samples, sample_ids)):
        if int(sample.sample_index) != sample_index:
            raise ValueError(f"{contig}: truth-painting sample order mismatch")
        row = expected.loc[(sample_id, contig)]
        observed_topology = _topology_from_chunks(sample.chunks)
        saved_topology = [
            tuple(map(int, pair)) for pair in ast.literal_eval(row["Topology_truth"])
        ]
        topology_mismatches += int(observed_topology != saved_topology)
        chunk_count_mismatches += int(
            len(sample.chunks) != int(row["N_chunks_truth"])
        )
        topology_count_mismatches += int(
            len(observed_topology) != int(row["N_topology_truth"])
        )
    summary = {
        "contig": contig,
        "rows": len(sample_ids),
        "topology_sequence_mismatches": topology_mismatches,
        "truth_chunk_count_mismatches": chunk_count_mismatches,
        "truth_topology_count_mismatches": topology_count_mismatches,
    }
    if any(summary[key] for key in (
        "topology_sequence_mismatches",
        "truth_chunk_count_mismatches",
        "truth_topology_count_mismatches",
    )):
        raise ValueError(f"{contig}: retained truth replay mismatch: {summary}")
    return summary


def _scientific_config() -> smart.PedigreeConfig:
    return smart.PedigreeConfig(
        parent_state_candidate_source_mode="hard_painted",
        parent_state_effective_markers_per_information_block=3.0,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
        primary_view="tier_b",
    ).validated()


def _manifest(
    stage1_root: Path,
    historical_root: Path,
    config: smart.PedigreeConfig,
    simulator_source: Path | None,
) -> dict[str, Any]:
    source_files = {
        f"stage1_{contig}": _legacy_stage1_path(stage1_root, contig)
        for contig in CONTIGS
    }
    for seed in TARGET_SEEDS:
        seed_root = historical_root / f"seed{seed}_linear_h1"
        source_files[f"seed{seed}_pedigree"] = seed_root / "ground_truth_pedigree.csv"
        source_files[f"seed{seed}_topology"] = (
            seed_root / "paint_samples_topology_evaluation.csv"
        )
    missing = [os.fspath(path) for path in source_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing benchmark inputs: " + ", ".join(missing))

    root = Path(__file__).resolve().parent
    code_files = (
        root / Path(__file__).name,
        root / "simulate_sequences.py",
        root / "pedigree_inference.py",
        root / "pedigree_hard_painting.py",
        root / "pedigree_result.py",
        root / "pedigree_evaluation.py",
        root / "pedigree_hmm.py",
        root / "pedigree_candidate_source_posterior.py",
        root / "painting_grid_utils.py",
        root / "paint_samples.py",
        root / "pipeline_runtime.py",
        root / "bhd_results.py",
        root / "bhd_config.py",
        root / "bhd_genotype_evidence.py",
        root / "checkpoint_io.py",
        root / "thread_config.py",
        root / "thread_env.py",
        root / "dynamic_threads.py",
        root / "multiprocessing_runtime.py",
        root / "shared_array.py",
    )
    if simulator_source is not None:
        code_files += (simulator_source.resolve(),)
    scientific_identity = {
        "schema_version": SCHEMA_VERSION,
        "interpretation": (
            "oracle truth-founder/truth-painting benchmark of the pedigree "
            "engine; topology-equivalent to retained seed artifacts but not "
            "proven bitwise-identical in physical breakpoints or track order; "
            "not a production reconstruction accuracy estimate"
        ),
        "contigs": list(CONTIGS),
        "target_seeds": list(TARGET_SEEDS),
        "generation_sizes": list(GENERATION_SIZES),
        "recombination_rate_per_bp": RECOMBINATION_RATE,
        "mutation_rate_per_bp": MUTATION_RATE,
        "simulator_mode": (
            "current" if simulator_source is None else "explicit_source"
        ),
        "smart_config": dataclasses.asdict(config),
        "scoring_kwargs": {
            "top_k": 20,
            "anchor_k": 5,
            "use_anchor_union": True,
            "snps_per_bin": 100,
            "max_snps_per_bin": 10,
            "recomb_rate": RECOMBINATION_RATE,
        },
        "inputs": {
            name: _file_identity(path, content_hash=True)
            for name, path in source_files.items()
        },
        "code": {
            path.name: _file_identity(path, content_hash=True)
            for path in code_files
        },
    }
    return {
        **scientific_identity,
        "scientific_identity_sha256": _json_digest(scientific_identity),
    }


def _initialize_output(root: Path, manifest: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("scientific_identity_sha256") != manifest[
            "scientific_identity_sha256"
        ]:
            raise RuntimeError(
                f"{root}: existing output has a different scientific identity"
            )
    else:
        _atomic_json(manifest_path, manifest)


def _prepare_founder_checkpoints(
    store: pipeline_runtime.CheckpointStore,
    stage1_root: Path,
    workers: int,
) -> None:
    stage = "01_oracle_founders"
    if store.stage_complete(stage):
        pipeline_runtime.require_contig_checkpoints(store, stage, CONTIGS)
        for contig in CONTIGS:
            source = _legacy_stage1_path(stage1_root, contig)
            payload = store.load_contig(stage, contig)
            if (
                payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("contig") != contig
                or payload.get("source_sha256") != _sha256(source)
            ):
                raise ValueError(f"{contig}: founder checkpoint identity mismatch")
        return
    for contig in CONTIGS:
        source = _legacy_stage1_path(stage1_root, contig)
        payload = _read_legacy_checkpoint(source, nthreads=workers)
        positions, probability_haplotypes = _validate_stage1_payload(
            payload, contig
        )
        store.save_contig(stage, contig, {
            "schema_version": SCHEMA_VERSION,
            "contig": contig,
            "source_sha256": _sha256(source),
            "founder_block": _founder_block(positions, probability_haplotypes),
        })
    pipeline_runtime.require_contig_checkpoints(store, stage, CONTIGS)
    store.mark_stage_complete(stage)


def _load_oracle_founders(
    store: pipeline_runtime.CheckpointStore,
    simulator: Any,
) -> tuple[list[Any], list[np.ndarray]]:
    founders_by_contig = []
    sites_by_contig = []
    for contig in CONTIGS:
        payload = store.load_contig("01_oracle_founders", contig)
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("contig") != contig
        ):
            raise ValueError(f"{contig}: founder checkpoint identity mismatch")
        block = payload["founder_block"]
        keys = sorted(int(key) for key in block.haplotypes)
        if keys != list(range(6)):
            raise ValueError(f"{contig}: founder labels are not exactly 0..5")
        concrete = simulator.concretify_haps(
            [block.haplotypes[index] for index in keys]
        )
        founders_by_contig.append(
            simulator.pairup_haps(concrete, shuffle=False)
        )
        sites_by_contig.append(np.asarray(block.positions, dtype=np.int64))
    return founders_by_contig, sites_by_contig


def _prepare_seed_inputs(
    store: pipeline_runtime.CheckpointStore,
    seed: int,
    historical_root: Path,
    workers: int,
    simulator: Any,
) -> tuple[tuple[str, ...], pd.DataFrame, dict[str, Any]]:
    stage = f"02_seed{seed}_oracle_inputs"
    if store.stage_complete(stage):
        pipeline_runtime.require_contig_checkpoints(store, stage, CONTIGS)
        if not store.global_done(stage):
            raise ValueError(f"{stage}: complete marker lacks global checkpoint")
        global_payload = store.load_global(stage)
        sample_ids = tuple(global_payload.get("sample_ids", ()))
        replay = global_payload.get("replay_validation", {})
        if (
            global_payload.get("schema_version") != SCHEMA_VERSION
            or global_payload.get("seed") != seed
            or len(sample_ids) != 320
            or replay.get("topology_sequence_mismatches") != 0
            or replay.get("truth_chunk_count_mismatches") != 0
            or replay.get("truth_topology_count_mismatches") != 0
        ):
            raise ValueError(f"{stage}: global checkpoint identity mismatch")
        for contig in CONTIGS:
            payload = store.load_contig(stage, contig)
            if (
                payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("seed") != seed
                or payload.get("contig") != contig
                or tuple(payload.get("sample_ids", ())) != sample_ids
            ):
                raise ValueError(f"{stage}/{contig}: checkpoint identity mismatch")
        return sample_ids, global_payload["truth_pedigree"], replay

    expected_truth = _expected_truth(seed, historical_root)
    expected_topology = _expected_topology(seed, historical_root)
    founders, sites = _load_oracle_founders(store, simulator)
    started = time.perf_counter()
    offspring, truth, raw_ancestry = simulator.simulate_pedigree(
        founders,
        sites,
        GENERATION_SIZES,
        recomb_rate=RECOMBINATION_RATE,
        mutate_rate=MUTATION_RATE,
        output_plot=None,
        parallel=True,
        num_processes=workers,
        seed=seed,
    )
    simulation_seconds = time.perf_counter() - started
    _require_exact_truth_pedigree(truth, expected_truth, seed)
    sample_ids = tuple(str(value) for value in truth["Sample"])

    replay_rows = []
    conversion_workers = workers
    for contig_index, contig in enumerate(CONTIGS):
        painting = simulator.convert_truth_to_painting_objects(
            raw_ancestry[contig_index], num_workers=conversion_workers
        )
        replay_rows.append(_validate_replayed_painting(
            painting, contig, sample_ids, expected_topology
        ))
        store.save_contig(stage, contig, {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "contig": contig,
            "sample_ids": sample_ids,
            "tolerance_painting": painting,
        })
        offspring[contig_index] = None
        raw_ancestry[contig_index] = None
        founders[contig_index] = None
        sites[contig_index] = None
        gc.collect()

    replay_validation = {
        "seed": seed,
        "pedigree_rows_exact": len(truth),
        "topology_equivalent_rows": int(sum(row["rows"] for row in replay_rows)),
        "topology_sequence_mismatches": int(sum(
            row["topology_sequence_mismatches"] for row in replay_rows
        )),
        "truth_chunk_count_mismatches": int(sum(
            row["truth_chunk_count_mismatches"] for row in replay_rows
        )),
        "truth_topology_count_mismatches": int(sum(
            row["truth_topology_count_mismatches"] for row in replay_rows
        )),
        "simulation_seconds": simulation_seconds,
        "simulation_process_thread_budget": workers,
        "simulation_outer_workers": min(len(CONTIGS), workers),
        "simulation_inner_workers": max(1, workers // len(CONTIGS)),
    }
    store.save_global(stage, {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "sample_ids": sample_ids,
        "truth_pedigree": truth,
        "replay_validation": replay_validation,
        "per_contig_replay_validation": replay_rows,
    })
    pipeline_runtime.require_contig_checkpoints(store, stage, CONTIGS)
    if not store.global_done(stage):
        raise OSError(f"failed to checkpoint {stage}/_global")
    store.mark_stage_complete(stage)
    return sample_ids, truth, replay_validation


def _load_contig_inputs(
    store: pipeline_runtime.CheckpointStore,
    seed: int,
    sample_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    inputs = []
    for contig in CONTIGS:
        founder_payload = store.load_contig("01_oracle_founders", contig)
        painting_payload = store.load_contig(
            f"02_seed{seed}_oracle_inputs", contig
        )
        if (
            painting_payload.get("seed") != seed
            or painting_payload.get("contig") != contig
            or tuple(painting_payload.get("sample_ids", ())) != sample_ids
        ):
            raise ValueError(f"seed {seed}/{contig}: input identity mismatch")
        inputs.append({
            "contig": contig,
            "tolerance_painting": painting_payload["tolerance_painting"],
            "founder_block": pipeline_runtime.compact_founder_block(
                founder_payload["founder_block"]
            ),
        })
    return inputs


def _evaluate_relationships(
    truth: pd.DataFrame,
    inferred: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    tier_b: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {"Sample", "Parent1", "Parent2"}
    if not required.issubset(truth) or not required.issubset(inferred):
        raise ValueError("truth and inference need Sample/Parent1/Parent2")
    if truth["Sample"].duplicated().any() or inferred["Sample"].duplicated().any():
        raise ValueError("duplicate pedigree sample rows")
    if set(truth["Sample"]) != set(inferred["Sample"]):
        raise ValueError("truth and inference sample sets differ")
    merged = truth[["Sample", "Generation", "Parent1", "Parent2"]].merge(
        inferred[[
            "Sample", "Parent1", "Parent2", "ParentState",
            "InferenceStatus",
        ]],
        on="Sample",
        validate="one_to_one",
        suffixes=("_True", "_Inf"),
    )
    merged["TopologyExact"] = merged.apply(parent_columns_match, axis=1)
    diagnostic_fields = diagnostics[[
        "Sample", "TierBStateCall", "TierBExactConfiguration",
    ]]
    merged = merged.merge(diagnostic_fields, on="Sample", validate="one_to_one")
    founder_truth = merged[["Parent1_True", "Parent2_True"]].apply(
        lambda column: column.astype(str).str.contains("Founder", regex=False)
    ).any(axis=1)
    merged["ExpectedParentState"] = np.where(
        founder_truth, "zero_observed_parents", "two_observed_parents"
    )
    merged["StateExact"] = (
        merged["ParentState"] == merged["ExpectedParentState"]
    )

    def observed_truth_edges(row: Any) -> set[str]:
        return {
            str(value) for value in (row.Parent1_True, row.Parent2_True)
            if not pd.isna(value) and "Founder" not in str(value)
        }

    def inferred_edges(row: Any) -> set[str]:
        return {
            str(value) for value in (row.Parent1_Inf, row.Parent2_Inf)
            if not pd.isna(value)
        }

    correct_edges = sum(
        len(observed_truth_edges(row) & inferred_edges(row))
        for row in merged.itertuples()
    )
    inferred_edge_count = int(
        merged[["Parent1_Inf", "Parent2_Inf"]].notna().sum().sum()
    )
    false_root_edges = int(
        merged.loc[founder_truth, ["Parent1_Inf", "Parent2_Inf"]]
        .notna().sum().sum()
    )
    released = (
        merged["TierBExactConfiguration"].astype(bool)
        if tier_b else pd.Series(True, index=merged.index)
    )
    released_count = int(released.sum())
    state_released = (
        merged["TierBStateCall"].astype(bool)
        if tier_b else pd.Series(True, index=merged.index)
    )
    state_released_count = int(state_released.sum())
    summary = {
        "rows": len(merged),
        "correct_topology_rows": int(merged["TopologyExact"].sum()),
        "correct_state_rows": int(merged["StateExact"].sum()),
        "released_state_rows": state_released_count,
        "released_state_correct_rows": int(
            merged.loc[state_released, "StateExact"].sum()
        ),
        "correct_edges": int(correct_edges),
        "inferred_edges": inferred_edge_count,
        "false_root_edges": false_root_edges,
        "expected_m0_rows": int(founder_truth.sum()),
        "expected_m2_rows": int((~founder_truth).sum()),
        "released_exact_configuration_rows": released_count,
        "exact_configuration_coverage": released_count / len(merged),
        "released_exact_configuration_correct_rows": int(
            merged.loc[released, "TopologyExact"].sum()
        ),
        "released_exact_configuration_selective_accuracy": (
            None if released_count == 0
            else float(merged.loc[released, "TopologyExact"].mean())
        ),
        "tier_b_state_coverage": float(
            merged["TierBStateCall"].astype(bool).mean()
        ),
    }
    return summary, merged


def _persist_result_tables(result: Any, output: Path) -> None:
    table_attributes = (
        "complete_relationships",
        "tier_a_relationships",
        "tier_b_relationships",
        "tier_a_partial_relationships",
        "tier_b_partial_relationships",
        "tier_b_candidate_sets",
        "smart_parent_state_calls",
        "smart_diagnostics",
        "smart_prior_sensitivity_summary",
        "smart_evidence_summary",
        "smart_candidate_source_diagnostics",
        "smart_predictive_folds",
    )
    for attribute in table_attributes:
        value = getattr(result, attribute, None)
        if isinstance(value, pd.DataFrame):
            _atomic_csv(output / f"{attribute}.csv", value)


def _run_inference(
    store: pipeline_runtime.CheckpointStore,
    output_root: Path,
    seed: int,
    sample_ids: tuple[str, ...],
    truth: pd.DataFrame,
    replay_validation: dict[str, Any],
    config: smart.PedigreeConfig,
    workers: int,
) -> dict[str, Any]:
    seed_output = output_root / f"seed{seed}"
    complete_marker = seed_output / "inference_complete.json"
    if complete_marker.is_file():
        with complete_marker.open(encoding="utf-8") as handle:
            completed = json.load(handle)
        if (
            completed.get("schema_version") != SCHEMA_VERSION
            or completed.get("seed") != seed
            or _json_digest(completed.get("smart_config"))
            != _json_digest(dataclasses.asdict(config))
        ):
            raise ValueError(f"seed {seed}: inference marker identity mismatch")
        return completed

    inputs = _load_contig_inputs(store, seed, sample_ids)
    started_wall = time.perf_counter()
    started_usage = resource.getrusage(resource.RUSAGE_SELF)
    started_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = smart.infer_pedigree(
        inputs,
        sample_ids,
        parent_eligibility=None,
        config=config,
        scoring_kwargs={
            "top_k": 20,
            "anchor_k": 5,
            "use_anchor_union": True,
            "snps_per_bin": 100,
            "max_snps_per_bin": 10,
            "recomb_rate": RECOMBINATION_RATE,
            "n_workers": workers,
        },
    )
    elapsed = time.perf_counter() - started_wall
    ended_usage = resource.getrusage(resource.RUSAGE_SELF)
    ended_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    diagnostics = result.smart_diagnostics
    complete_summary, complete_comparison = _evaluate_relationships(
        truth, result.complete_relationships, diagnostics, tier_b=False
    )
    tier_b_summary, tier_b_comparison = _evaluate_relationships(
        truth, result.tier_b_relationships, diagnostics, tier_b=True
    )
    accepted = (
        complete_summary["rows"] == 320
        and complete_summary["correct_topology_rows"] == 320
        and complete_summary["correct_state_rows"] == 320
        and complete_summary["correct_edges"] == 600
        and complete_summary["inferred_edges"] == 600
        and complete_summary["false_root_edges"] == 0
        and complete_summary["expected_m0_rows"] == 20
        and complete_summary["expected_m2_rows"] == 300
    )
    _persist_result_tables(result, seed_output)
    _atomic_csv(seed_output / "complete_comparison_to_truth.csv", complete_comparison)
    _atomic_csv(seed_output / "tier_b_comparison_to_truth.csv", tier_b_comparison)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "complete_320_gate_passed": accepted,
        "complete": complete_summary,
        "tier_b": tier_b_summary,
        "replay_validation": replay_validation,
        "runtime": {
            "workers": workers,
            "elapsed_seconds": elapsed,
            "self_user_cpu_seconds": ended_usage.ru_utime - started_usage.ru_utime,
            "self_system_cpu_seconds": ended_usage.ru_stime - started_usage.ru_stime,
            "self_peak_rss_kib": int(ended_usage.ru_maxrss),
            "children_user_cpu_seconds": (
                ended_children.ru_utime - started_children.ru_utime
            ),
            "children_system_cpu_seconds": (
                ended_children.ru_stime - started_children.ru_stime
            ),
            "children_peak_rss_kib": int(ended_children.ru_maxrss),
            "smart_standard_input_threads": int(
                getattr(result, "smart_standard_input_threads", 0)
            ),
            "smart_standard_input_processes": int(
                getattr(result, "smart_standard_input_processes", 0)
            ),
            "smart_bootstrap_worker_count": int(
                getattr(result, "smart_bootstrap_worker_count", 0)
            ),
        },
        "smart_config": dataclasses.asdict(config),
        "legacy_consistency_cutoff_requested": bool(getattr(
            result, "smart_legacy_consistency_cutoff_requested", False
        )),
        "legacy_consistency_cutoff_applied": bool(getattr(
            result, "smart_legacy_consistency_cutoff_applied", False
        )),
    }
    _atomic_json(seed_output / "evaluation_summary.json", summary)
    _atomic_json(complete_marker, summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument(
        "--historical-results-root", type=Path, default=Path("results_simulation")
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--simulator-source", type=Path, default=None)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(TARGET_SEEDS),
        choices=TARGET_SEEDS,
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--infer-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.prepare_only and args.infer_only:
        raise ValueError("--prepare-only and --infer-only are mutually exclusive")
    affinity = len(os.sched_getaffinity(0))
    workers = affinity if args.workers is None else int(args.workers)
    if not 1 <= workers <= affinity:
        raise ValueError(f"workers must lie in [1, {affinity}]")
    seeds = tuple(dict.fromkeys(args.seeds))
    config = _scientific_config()
    simulator = _load_simulator(args.simulator_source)
    manifest = _manifest(
        args.stage1_root, args.historical_results_root, config,
        args.simulator_source,
    )
    _initialize_output(args.output_root, manifest)
    checkpoint_root = args.output_root / "checkpoints"
    store = pipeline_runtime.CheckpointStore(checkpoint_root, nthreads=workers)

    print(
        f"Oracle combined-v1 benchmark: affinity={affinity}, workers={workers}, "
        f"seeds={','.join(map(str, seeds))}",
        flush=True,
    )
    _prepare_founder_checkpoints(store, args.stage1_root, workers)
    summaries = []
    for seed in seeds:
        if args.infer_only:
            stage = f"02_seed{seed}_oracle_inputs"
            if not store.stage_complete(stage):
                raise RuntimeError(f"{stage} is not complete")
        sample_ids, truth, replay_validation = _prepare_seed_inputs(
            store, seed, args.historical_results_root, workers, simulator
        )
        print(
            f"seed {seed}: topology-equivalence replay gate passed for "
            f"{replay_validation['topology_equivalent_rows']:,} sample-contigs",
            flush=True,
        )
        if args.prepare_only:
            continue
        summary = _run_inference(
            store,
            args.output_root,
            seed,
            sample_ids,
            truth,
            replay_validation,
            config,
            workers,
        )
        summaries.append(summary)
        print(
            f"seed {seed}: complete topology "
            f"{summary['complete']['correct_topology_rows']}/320; "
            f"gate={'PASS' if summary['complete_320_gate_passed'] else 'FAIL'}",
            flush=True,
        )

    if summaries:
        combined = {
            "schema_version": SCHEMA_VERSION,
            "seeds_completed": [summary["seed"] for summary in summaries],
            "all_complete_320_gates_passed": all(
                summary["complete_320_gate_passed"] for summary in summaries
            ),
            "correct_topology_rows": int(sum(
                summary["complete"]["correct_topology_rows"]
                for summary in summaries
            )),
            "total_topology_rows": int(sum(
                summary["complete"]["rows"] for summary in summaries
            )),
            "correct_edges": int(sum(
                summary["complete"]["correct_edges"] for summary in summaries
            )),
            "expected_edges": 600 * len(summaries),
        }
        _atomic_json(args.output_root / "combined_summary.json", combined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
