"""Bounded performance harness for combined-v1 pedigree inference seams.

Each seam runs in a fresh Python process so peak RSS is interpretable.  JIT
compilation and fixture construction precede the timed repetitions.  Results
are JSON and may be written only below ``/tmp``.  This driver measures speed
and resource use; equivalence is enforced by
``test_pedigree_performance_equivalence.py``.

The ``tropheops116`` fixture reproduces the observed candidate-universe sizes
(4 G0, 16 F1 split 8/8 by recorded sex, 96 F2) without asserting individual
parentage. ``synthetic320`` is a bounded F1/F2/F3 scaling workload.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import resource
import statistics
import struct
import subprocess
import sys
import threading
import time
import tracemalloc
from typing import Any, Callable
import warnings

import numpy as np
import pandas as pd
import psutil


SEAMS = (
    "alternatives",
    "screen",
    "structure",
    "pair_hmm",
    "raw_gl",
    "gmm",
    "bootstrap_chunk",
)


def _load_module(value: str):
    path = Path(value)
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        name = (
            "_pedigree_performance_baseline"
            if path.name == "pedigree_inference_combined_v1_baseline.py"
            else f"_pedigree_benchmark_target_{digest}"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module, str(path.resolve()), digest
    module = importlib.import_module(value)
    module_path = Path(module.__file__).resolve()
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()[:16]
    return module, str(module_path), digest


def _policy(scenario: str):
    # Keep every inference target in a leading contiguous block.  The frozen
    # baseline infers its internal child-axis length from the maximum target
    # index; this ordering lets the same scientific candidate dimensions run
    # on both modules without changing any eligibility counts.
    if scenario == "tropheops116":
        sizes = (("F2", 96), ("F1", 16), ("G0", 4))
        parent_generation = {"F2": "F1"}
    elif scenario == "synthetic320":
        sizes = (("F2", 100), ("F3", 200), ("F1", 20))
        parent_generation = {"F2": "F1", "F3": "F2"}
    else:
        raise ValueError(scenario)
    samples: list[str] = []
    generations: list[str] = []
    sexes: list[str] = []
    for generation, size in sizes:
        for index in range(size):
            samples.append(f"{generation}_{index}")
            generations.append(generation)
            sexes.append("F" if index < size // 2 else "M")
    generations_array = np.asarray(generations, dtype=object)
    sexes_array = np.asarray(sexes, dtype=object)
    n_samples = len(samples)
    children = np.isin(generations_array, tuple(parent_generation))
    parents = np.zeros((n_samples, n_samples), dtype=np.bool_)
    pairs = np.zeros((n_samples, n_samples, n_samples), dtype=np.bool_)
    for child in np.flatnonzero(children):
        parent_mask = (
            generations_array == parent_generation[generations_array[child]]
        )
        parents[child] = parent_mask
        female = np.flatnonzero(parent_mask & (sexes_array == "F"))
        male = np.flatnonzero(parent_mask & (sexes_array == "M"))
        pairs[child][np.ix_(female, male)] = True
        pairs[child][np.ix_(male, female)] = True
    return tuple(samples), {
        "format_version": 1,
        "sample_ids": tuple(samples),
        "eligible_children": np.ascontiguousarray(children),
        "eligible_parents": np.ascontiguousarray(parents),
        "eligible_parent_pairs": np.ascontiguousarray(pairs),
        "policy_name": f"benchmark_{scenario}_v1",
        "source_fields": ("synthetic_generation", "synthetic_recorded_sex"),
        "assumptions": ("Candidate-universe scaling fixture only.",),
        "individual_parentage_ground_truth": False,
    }


def _trios(policy: dict[str, Any]) -> np.ndarray:
    rows: list[tuple[int, int, int]] = []
    children = np.asarray(policy["eligible_children"], dtype=np.bool_)
    pairs = np.asarray(policy["eligible_parent_pairs"], dtype=np.bool_)
    for child in np.flatnonzero(children):
        pair_mask = np.triu(pairs[child], k=1)
        if len(children) == 320:
            # Reproduce the production top-k/anchor-union scale without
            # pretending the deterministic synthetic ordering is a ranking.
            parent_indices = np.flatnonzero(policy["eligible_parents"][child])
            leading = parent_indices[:20]
            anchors = leading[:5]
            bounded = np.zeros_like(pair_mask)
            bounded[np.ix_(leading, leading)] = pair_mask[np.ix_(leading, leading)]
            bounded[anchors] |= pair_mask[anchors]
            pair_mask = bounded
        first, second = np.nonzero(pair_mask)
        rows.extend(
            (int(child), int(parent1), int(parent2))
            for parent1, parent2 in zip(first, second)
        )
    return np.asarray(rows, dtype=np.int64).reshape((-1, 3))


def _config(module, bootstrap_replicates: int = 8):
    return module.PedigreeConfig(
        bootstrap_replicates=bootstrap_replicates,
        minimum_informative_contigs=1,
        parent_state_minimum_exposed_contigs=1,
    ).validated()


def _alternative_fixture(module, scenario: str, *, n_contigs: int = 8):
    samples, policy = _policy(scenario)
    eligibility = module._resolve_parent_eligibility(policy, samples)
    trios = _trios(policy)
    n_samples = len(samples)
    rng = np.random.default_rng(20260831)
    zero = rng.normal(size=(n_contigs, n_samples))
    one = rng.normal(size=(n_contigs, n_samples, n_samples))
    for contig in range(n_contigs):
        np.fill_diagonal(one[contig], -np.inf)
    two = rng.normal(size=(n_contigs, len(trios)))
    return samples, policy, eligibility, trios, zero, one, two


def _alternatives_seam(module, scenario: str):
    fixture = _alternative_fixture(module, scenario)
    _, _, eligibility, trios, zero, one, two = fixture

    def run():
        return module._parent_state_alternatives(
            trios, zero, one, two, 0.02, eligibility
        )

    return run, {
        "samples": len(fixture[0]),
        "trios": len(trios),
        "contigs": zero.shape[0],
    }


def _screen_seam(module, scenario: str):
    samples, _, eligibility, _, _, _, _ = _alternative_fixture(
        module, scenario, n_contigs=1
    )
    rng = np.random.default_rng(882)
    marker_counts = np.asarray(
        (80, 100, 150, 200, 300, 500, 800, 1300), dtype=np.float64
    )
    scores = rng.normal(size=(len(marker_counts), len(samples), len(samples)))
    config = _config(module, 1)

    def run():
        return module._robust_parent_screen(
            scores, marker_counts, config, eligibility
        )

    return run, {
        "samples": len(samples),
        "contigs": len(marker_counts),
        "eligible_directed_edges": int(
            np.count_nonzero(eligibility.eligible_parents)
        ),
    }


def _structure_seam(module, scenario: str):
    samples, policy = _policy(scenario)
    trios = _trios(policy)
    rng = np.random.default_rng(9981)
    n_bins = 48 if scenario == "tropheops116" else 16
    labels = rng.integers(0, 8, size=(len(samples), n_bins, 2), dtype=np.int16)
    labels[rng.random(labels.shape) < 0.015] = -1
    required_edges = np.ascontiguousarray(
        policy["eligible_parents"] | np.asarray(policy["eligible_parents"]).T
    )
    signature = inspect.signature(module._parenthood_structure_count_kernel.py_func)

    def run():
        if "required_edges" in signature.parameters:
            return module._parenthood_structure_count_kernel(
                labels, trios, required_edges
            )
        return module._parenthood_structure_count_kernel(labels, trios)

    return run, {
        "samples": len(samples),
        "bins": n_bins,
        "trios": len(trios),
        "required_undirected_edge_entries": int(np.count_nonzero(required_edges)),
    }


def _pair_cache(module, scenario: str):
    samples, policy = _policy(scenario)
    eligibility = module._resolve_parent_eligibility(policy, samples)
    rng = np.random.default_rng(665)
    n_bins = 24 if scenario == "tropheops116" else 12
    alleles = rng.integers(0, 2, size=(len(samples), n_bins, 2), dtype=np.int8)
    hom = rng.random((len(samples), n_bins)) < 0.2
    theta = np.linspace(0.001, 0.03, n_bins)
    switch_costs = -np.log(theta)
    stay_costs = -np.log1p(-theta)
    labels = rng.integers(0, 4, size=(len(samples), n_bins, 2), dtype=np.int16)
    founders = rng.integers(0, 2, size=(4, n_bins), dtype=np.int8)
    cache = module._StandardContigCache(
        "benchmark",
        alleles,
        hom,
        switch_costs,
        stay_costs,
        n_bins,
        labels,
        founders,
        np.ones(n_bins, dtype=np.int64),
        theta,
    )
    return cache, eligibility


def _pair_hmm_seam(module, scenario: str):
    cache, eligibility = _pair_cache(module, scenario)
    signature = inspect.signature(module._score_pair_hmm_contig)

    def run():
        if "eligibility" in signature.parameters:
            return module._score_pair_hmm_contig(cache, -3.0, eligibility)
        return module._score_pair_hmm_contig(cache, -3.0)

    return run, {
        "samples": cache.stacked_alleles.shape[0],
        "bins": cache.stacked_alleles.shape[1],
        "eligible_directed_edges": int(
            np.count_nonzero(eligibility.eligible_parents)
        ),
    }


def _raw_gl_fixture(module, scenario: str):
    samples, policy = _policy(scenario)
    eligibility = module._resolve_parent_eligibility(policy, samples)
    trios = _trios(policy)
    rng = np.random.default_rng(731)
    n_samples = len(samples)
    n_bins = 5 if scenario == "tropheops116" else 3
    n_states = 4
    founders = rng.integers(0, 2, size=(n_states, n_bins, 1), dtype=np.int8)
    labels = rng.integers(
        0, n_states, size=(n_samples, n_bins, 2), dtype=np.int16
    )
    alleles = np.empty((n_samples, n_bins, 2, 1), dtype=np.int8)
    for sample in range(n_samples):
        for block in range(n_bins):
            for track in range(2):
                alleles[sample, block, track, 0] = founders[
                    labels[sample, block, track], block, 0
                ]
    # Hit both complete fast and linked-partial candidate paths.
    eligible_parent_indices = np.flatnonzero(np.any(policy["eligible_parents"], axis=0))
    if len(eligible_parent_indices) >= 2:
        alleles[eligible_parent_indices[1::3], 1, 0, 0] = -1
    genotype = np.empty((n_samples, n_bins, 1, 3), dtype=np.float64)
    dosage = np.clip(np.sum(np.maximum(alleles[..., 0], 0), axis=2), 0, 2)
    likelihoods = np.asarray(
        ((0.82, 0.14, 0.04), (0.08, 0.84, 0.08), (0.04, 0.14, 0.82))
    )
    genotype[:, :, 0] = likelihoods[dosage]
    hom = labels[:, :, 0] == labels[:, :, 1]
    marker_counts = np.ones(n_bins, dtype=np.int64)
    theta = np.linspace(0.0, 0.03, n_bins)
    return (
        eligibility,
        genotype,
        alleles,
        labels,
        hom,
        founders,
        marker_counts,
        theta,
        trios,
    )


def _raw_gl_seam(module, scenario: str):
    fixture = _raw_gl_fixture(module, scenario)
    eligibility, *args = fixture
    signature = inspect.signature(module.score_parent_state_gl_hmms)
    kwargs = {}
    if "_eligible_children" in signature.parameters:
        kwargs["_eligible_children"] = eligibility.eligible_children
        kwargs["_eligible_parents"] = eligibility.eligible_parents

    def run():
        return module.score_parent_state_gl_hmms(*args, **kwargs)

    return run, {
        "samples": len(eligibility.sample_ids),
        "bins": args[0].shape[1],
        "trios": len(args[-1]),
        "eligible_directed_edges": int(
            np.count_nonzero(eligibility.eligible_parents)
        ),
    }


def _gmm_seam(module, scenario: str):
    n_samples = 116 if scenario == "tropheops116" else 320
    counts = np.concatenate(
        (
            np.linspace(2.0, 10.0, n_samples // 3),
            np.linspace(25.0, 42.0, n_samples // 3),
            np.linspace(80.0, 115.0, n_samples - 2 * (n_samples // 3)),
        )
    )
    callable_bins = np.linspace(70.0, 100.0, n_samples)
    callable_bins[::97] = 0.0

    def run():
        return module._fit_ancestry_depth_model(counts, callable_bins, 991)

    return run, {"samples": n_samples, "maximum_components": 6}


def _bootstrap_seam(module, scenario: str):
    n_contigs = 8 if scenario == "tropheops116" else 4
    fixture = _alternative_fixture(module, scenario, n_contigs=n_contigs)
    samples, _, eligibility, trios, zero, one, two = fixture
    settings = _config(module, 4)
    alternatives = module._parent_state_alternatives(
        trios,
        zero,
        one,
        two,
        settings.parent_state_contamination_probability,
        eligibility,
    )
    rows, states, contig_likelihoods, by_child, full_counts = alternatives[:5]
    pair_indices = module._structure_pair_indices(rows, states, trios)
    n_samples = len(samples)
    required_edges = eligibility.eligible_parents | eligibility.eligible_parents.T
    edge_exposed = np.zeros((n_contigs, n_samples, n_samples), dtype=np.float64)
    edge_exposed[:, required_edges] = 100.0
    edge_matched = edge_exposed * 0.98
    pair_exposed = np.full((n_contigs, len(trios)), 100.0)
    pair_explained = pair_exposed * 0.98
    burden = np.linspace(3.0, 100.0, n_samples)
    junctions = np.stack(
        [burden * (1.0 + 0.01 * index) for index in range(n_contigs)]
    )
    callable_bins = np.full_like(junctions, 100.0)
    information = np.ones(n_contigs, dtype=np.float64)
    shared = {
        "alternatives": rows,
        "states": states,
        "contig_log_likelihoods": contig_likelihoods,
        "by_child": tuple(by_child),
        "full_counts": full_counts,
        "junction_matrix": junctions,
        "callable_matrix": callable_bins,
        "n_samples": n_samples,
        "bootstrap_seed": settings.bootstrap_seed,
        "contig_information_weights": information,
        "settings": settings,
        "structure_pair_indices": pair_indices,
        "edge_matched_by_contig": edge_matched,
        "edge_exposed_by_contig": edge_exposed,
        "pair_explained_by_contig": pair_explained,
        "pair_exposed_by_contig": pair_exposed,
        "structure_total_bins_by_contig": np.full(n_contigs, 100.0),
    }
    rng = np.random.default_rng(settings.bootstrap_seed)
    replicate_count = 4 if scenario == "tropheops116" else 1
    multiplicities = np.stack(
        [
            np.bincount(
                rng.integers(0, n_contigs, size=n_contigs),
                minlength=n_contigs,
            ).astype(float)
            for _ in range(replicate_count)
        ]
    )

    def run():
        return module._evaluate_smart_bootstrap_chunk(shared, multiplicities)

    return run, {
        "samples": n_samples,
        "contigs": n_contigs,
        "alternatives": len(rows),
        "replicates": replicate_count,
    }


_BUILDERS: dict[str, Callable] = {
    "alternatives": _alternatives_seam,
    "screen": _screen_seam,
    "structure": _structure_seam,
    "pair_hmm": _pair_hmm_seam,
    "raw_gl": _raw_gl_seam,
    "gmm": _gmm_seam,
    "bootstrap_chunk": _bootstrap_seam,
}


def _checksum(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, np.ndarray):
        finite = np.asarray(value)[np.isfinite(value)]
        return float(np.sum(finite, dtype=np.float64)) + float(value.size)
    if isinstance(value, (tuple, list)):
        return sum(_checksum(item) for item in value)
    if hasattr(value, "__dataclass_fields__"):
        return sum(
            _checksum(getattr(value, field)) for field in value.__dataclass_fields__
        )
    if isinstance(value, dict):
        return sum(_checksum(item) for item in value.values())
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if np.isfinite(value) else 0.0
    return 0.0


def _measure(run: Callable, repeats: int):
    warm = run()
    warm_checksum = _checksum(warm)
    del warm
    gc.collect()
    tracemalloc.start()
    timings = []
    checksum = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        value = run()
        timings.append(time.perf_counter() - start)
        checksum += _checksum(value)
        del value
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "wall_seconds": timings,
        "median_wall_seconds": statistics.median(timings),
        "minimum_wall_seconds": min(timings),
        "peak_rss_kib": int(usage.ru_maxrss),
        "traced_peak_bytes": int(traced_peak),
        "warm_checksum": warm_checksum,
        "timed_checksum": checksum,
    }


def _worker(args) -> int:
    module, module_path, digest = _load_module(args.module)
    import numba

    available = len(os.sched_getaffinity(0))
    threads = min(args.threads, available, int(numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(threads)
    run, dimensions = _BUILDERS[args.seam](module, args.scenario)
    result = {
        "seam": args.seam,
        "scenario": args.scenario,
        "module_path": module_path,
        "module_sha256_prefix": digest,
        "threads": threads,
        "available_affinity_cpus": available,
        "dimensions": dimensions,
        **_measure(run, args.repeats),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _safe_output(path_text: str) -> Path:
    path = Path(path_text).resolve()
    try:
        path.relative_to("/tmp")
    except ValueError as exc:
        raise ValueError("benchmark output must be below /tmp") from exc
    return path


def _controller(args) -> int:
    output = _safe_output(args.output)
    selected = SEAMS if args.seams == "all" else tuple(args.seams.split(","))
    invalid = sorted(set(selected).difference(SEAMS))
    if invalid:
        raise ValueError(f"unknown seams: {invalid}")
    records = []
    for seam in selected:
        command = (
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--module",
            args.module,
            "--scenario",
            args.scenario,
            "--seam",
            seam,
            "--threads",
            str(args.threads),
            "--repeats",
            str(args.repeats),
        )
        completed = subprocess.run(
            command, check=True, text=True, capture_output=True
        )
        records.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    payload = {
        "format": "pedigree_combined_v1_benchmark_v1",
        "created_unix_seconds": time.time(),
        "python": sys.version,
        "module": args.module,
        "scenario": args.scenario,
        "records": records,
    }
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(output)
    return 0


_SEED72_SIMULATION_STAGE = "02_simulation"
_SEED72_PAINTING_STAGE = "11_viterbi_painting"
_SEED72_BASELINE_STAGE = "12_pedigree_inference_current_b1_v1"


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _usage_snapshot() -> dict[str, float]:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "wall": time.perf_counter(),
        "own_user": own.ru_utime,
        "own_system": own.ru_stime,
        "children_user": children.ru_utime,
        "children_system": children.ru_stime,
        "own_peak_rss_kib": int(own.ru_maxrss),
        "children_peak_rss_kib": int(children.ru_maxrss),
        "input_blocks": int(own.ru_inblock + children.ru_inblock),
        "output_blocks": int(own.ru_oublock + children.ru_oublock),
    }


def _phase_delta(
    start: dict[str, float],
    end: dict[str, float],
    allocated_cpus: int,
) -> dict[str, float]:
    wall = float(end["wall"] - start["wall"])
    cpu = float(
        end["own_user"] - start["own_user"]
        + end["own_system"] - start["own_system"]
        + end["children_user"] - start["children_user"]
        + end["children_system"] - start["children_system"]
    )
    return {
        "elapsed_seconds": wall,
        "cpu_seconds": cpu,
        "allocation_cpu_fraction": (
            0.0 if wall <= 0.0 else cpu / (wall * allocated_cpus)
        ),
        "input_blocks": int(end["input_blocks"] - start["input_blocks"]),
        "output_blocks": int(end["output_blocks"] - start["output_blocks"]),
    }


def _start_recursive_monitor(phase_state: list[str]):
    root = psutil.Process(os.getpid())
    stop = threading.Event()
    samples: list[dict[str, Any]] = []

    def monitor() -> None:
        previous_cpu: dict[int, float] = {}
        previous_wall = time.perf_counter()
        while not stop.wait(1.0):
            now = time.perf_counter()
            try:
                processes = [root, *root.children(recursive=True)]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                processes = []
            current_cpu: dict[int, float] = {}
            rss = 0
            threads = 0
            descendants = 0
            for process in processes:
                try:
                    cpu = process.cpu_times()
                    current_cpu[process.pid] = float(cpu.user + cpu.system)
                    rss += int(process.memory_info().rss)
                    threads += int(process.num_threads())
                    descendants += int(process.pid != root.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            cpu_delta = sum(
                max(0.0, value - previous_cpu.get(pid, value))
                for pid, value in current_cpu.items()
            )
            wall_delta = now - previous_wall
            samples.append({
                "phase": phase_state[0],
                "wall_seconds": now,
                "recursive_rss_bytes": rss,
                "recursive_threads": threads,
                "active_descendants": descendants,
                "logical_cpu_equivalent": (
                    0.0 if wall_delta <= 0.0 else cpu_delta / wall_delta
                ),
            })
            previous_cpu = current_cpu
            previous_wall = now

    thread = threading.Thread(
        target=monitor, name="seed72-resource-monitor", daemon=True
    )
    thread.start()
    return stop, thread, samples


def _summarize_monitor(samples: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for phase in sorted({sample["phase"] for sample in samples}):
        rows = [sample for sample in samples if sample["phase"] == phase]
        output[phase] = {
            "samples": len(rows),
            "peak_recursive_rss_bytes": max(
                row["recursive_rss_bytes"] for row in rows
            ),
            "peak_recursive_threads": max(row["recursive_threads"] for row in rows),
            "peak_active_descendants": max(
                row["active_descendants"] for row in rows
            ),
            "peak_logical_cpu_equivalent": max(
                row["logical_cpu_equivalent"] for row in rows
            ),
            "mean_logical_cpu_equivalent": float(np.mean([
                row["logical_cpu_equivalent"] for row in rows
            ])),
        }
    return output


def _ordered_float_key(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    sign = 1 << 63
    return ((~bits) & ((1 << 64) - 1)) if bits & sign else bits | sign


def _diagnostic_float_differences(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
) -> dict[str, Any]:
    differences: dict[str, Any] = {}
    for column in observed.columns:
        left = observed[column]
        right = expected[column]
        if not (
            pd.api.types.is_float_dtype(left.dtype)
            and pd.api.types.is_float_dtype(right.dtype)
        ):
            pd.testing.assert_series_equal(left, right, check_exact=True)
            continue
        left_values = left.to_numpy(dtype=np.float64)
        right_values = right.to_numpy(dtype=np.float64)
        if not np.array_equal(np.isnan(left_values), np.isnan(right_values)):
            raise AssertionError(f"{column}: NaN masks differ")
        if not np.array_equal(np.isposinf(left_values), np.isposinf(right_values)):
            raise AssertionError(f"{column}: positive-infinity masks differ")
        if not np.array_equal(np.isneginf(left_values), np.isneginf(right_values)):
            raise AssertionError(f"{column}: negative-infinity masks differ")
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        if np.any(finite):
            left_finite = left_values[finite]
            right_finite = right_values[finite]
            max_absolute = float(np.max(np.abs(left_finite - right_finite)))
            max_ulp = max(
                abs(_ordered_float_key(left_value) - _ordered_float_key(right_value))
                for left_value, right_value in zip(left_finite, right_finite)
            )
        else:
            max_absolute = 0.0
            max_ulp = 0
        differences[column] = {
            "finite_values": int(np.count_nonzero(finite)),
            "max_absolute_difference": max_absolute,
            "max_ulp_difference": int(max_ulp),
        }
    return differences


def _frame_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(repr(tuple(frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _load_seed72_shadow_inputs(
    checkpoint_root: Path,
    current_module,
) -> tuple[list[dict[str, Any]], tuple[str, ...], pd.DataFrame, dict[str, Any], int]:
    import checkpoint_io
    import pipeline_runtime

    simulation_path = Path(checkpoint_io.global_path(
        checkpoint_root, _SEED72_SIMULATION_STAGE
    ))
    simulation = checkpoint_io.read(simulation_path, nthreads=1)
    sample_ids = tuple(map(str, simulation["sample_names"]))
    contigs = tuple(map(str, simulation["region_keys"]))
    truth = simulation["truth_pedigree"].copy()
    if len(sample_ids) != 320 or len(contigs) != 22 or len(set(contigs)) != 22:
        raise ValueError("seed-72 simulation checkpoint dimensions differ")
    if truth["Sample"].astype(str).tolist() != list(sample_ids):
        raise ValueError("seed-72 truth/sample order differs")

    inputs = []
    bytes_read = simulation_path.stat().st_size
    for contig in contigs:
        path = Path(checkpoint_io.contig_path(
            checkpoint_root, _SEED72_PAINTING_STAGE, contig
        ))
        payload = checkpoint_io.read(path, nthreads=1)
        bytes_read += path.stat().st_size
        if tuple(map(str, payload["sample_ids"])) != sample_ids:
            raise ValueError(f"{contig}: painting sample order differs")
        inputs.append({
            "contig": contig,
            "tolerance_painting": payload["tolerance_result"],
            "founder_block": pipeline_runtime.compact_founder_block(
                payload["founder_block"]
            ),
        })

    baseline_path = Path(checkpoint_io.global_path(
        checkpoint_root, _SEED72_BASELINE_STAGE
    ))
    baseline = checkpoint_io.read(baseline_path, nthreads=1)
    bytes_read += baseline_path.stat().st_size
    required = {
        "scientific_relationships",
        "complete_relationships",
        "tier_b_relationships",
        "smart_diagnostics",
        "smart_config",
    }
    if not required.issubset(baseline):
        raise ValueError("seed-72 baseline payload schema differs")
    return inputs, sample_ids, truth, baseline, bytes_read


def _seed72_shadow(args) -> int:
    if args.module != "pedigree_inference":
        raise ValueError("seed72 shadow requires the current pedigree_inference module")
    output = _safe_output(args.output)
    checkpoint_root = Path(args.seed72_root).resolve(strict=True)
    module, module_path_text, source_prefix = _load_module(args.module)
    source_path = Path(module_path_text)
    expected_hash = str(args.expected_source_hash)
    if len(expected_hash) != 64:
        raise ValueError("--expected-source-hash must be a full SHA-256 digest")

    def require_source(label: str) -> str:
        observed_hash = _source_sha256(source_path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"{label}: pedigree_inference source changed "
                f"({observed_hash} != {expected_hash})"
            )
        return observed_hash

    available_cpus = len(os.sched_getaffinity(0))
    if args.threads != 112 or available_cpus != 112:
        raise RuntimeError(
            "seed72 shadow requires the verified 112-CPU allocation and --threads 112"
        )
    require_source("launch")
    inputs, sample_ids, truth, baseline, compressed_input_bytes = (
        _load_seed72_shadow_inputs(checkpoint_root, module)
    )
    require_source("after_input_load")

    import pedigree_pipeline
    import pedigree_smart_oracle_seed_benchmark as oracle

    config = pedigree_pipeline.build_current_pedigree_config()
    if baseline["smart_config"] != config:
        raise AssertionError("stored seed-72 config differs from the current locked config")
    if config.bootstrap_replicates != 1000:
        raise AssertionError("seed-72 shadow requires bootstrap_replicates=1000")

    events: dict[str, dict[str, float]] = {}
    state_processes = None
    bootstrap_workers = None
    original_state = module._score_standard_state_contigs
    original_bootstrap = module._run_parent_state_bootstraps

    def state_wrapper(*positional, **keywords):
        nonlocal state_processes
        require_source("state_begin")
        events["state_begin"] = _usage_snapshot()
        print(
            "SEED72_BOUNDARY state_begin planned_processes=14 "
            "dynamic_total_threads=112",
            flush=True,
        )
        value = original_state(*positional, **keywords)
        state_processes = int(value[1])
        events["state_end"] = _usage_snapshot()
        require_source("state_end")
        print(
            f"SEED72_BOUNDARY state_end actual_processes={state_processes}",
            flush=True,
        )
        return value

    def bootstrap_wrapper(*positional, **keywords):
        nonlocal bootstrap_workers
        require_source("bootstrap_begin")
        events["bootstrap_begin"] = _usage_snapshot()
        print(
            "SEED72_BOUNDARY bootstrap_begin planned_processes=112 "
            "threads_per_process=1",
            flush=True,
        )
        value = original_bootstrap(*positional, **keywords)
        bootstrap_workers = int(value[0])
        events["bootstrap_end"] = _usage_snapshot()
        require_source("bootstrap_end")
        print(
            f"SEED72_BOUNDARY bootstrap_end actual_processes={bootstrap_workers}",
            flush=True,
        )
        return value

    module._score_standard_state_contigs = state_wrapper
    module._run_parent_state_bootstraps = bootstrap_wrapper
    events["launch"] = _usage_snapshot()
    print(
        f"SEED72_LAUNCH pid={os.getpid()} affinity={available_cpus} "
        f"source={source_prefix} inputs={len(inputs)} bootstrap=1000",
        flush=True,
    )
    recorded_warnings = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = module.infer_pedigree(
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
                    "recomb_rate": 5e-8,
                    "n_workers": 112,
                },
            )
            recorded_warnings = [
                {
                    "category": item.category.__name__,
                    "message": str(item.message),
                }
                for item in caught
            ]
    finally:
        module._score_standard_state_contigs = original_state
        module._run_parent_state_bootstraps = original_bootstrap
    events["inference_end"] = _usage_snapshot()
    require_source("after_inference")

    if args.frames_output:
        required_events = {
            "launch", "state_begin", "state_end",
            "bootstrap_begin", "bootstrap_end", "inference_end",
        }
        if set(events) != required_events:
            raise AssertionError(
                f"missing utilization boundary: "
                f"{sorted(required_events - set(events))}"
            )
        phases = {
            "pre_state": _phase_delta(
                events["launch"], events["state_begin"], 112
            ),
            "state_workers": _phase_delta(
                events["state_begin"], events["state_end"], 112
            ),
            "pre_bootstrap": _phase_delta(
                events["state_end"], events["bootstrap_begin"], 112
            ),
            "bootstrap_workers": _phase_delta(
                events["bootstrap_begin"], events["bootstrap_end"], 112
            ),
            "tail": _phase_delta(
                events["bootstrap_end"], events["inference_end"], 112
            ),
            "total": _phase_delta(
                events["launch"], events["inference_end"], 112
            ),
        }
        frames = {
            "scientific_relationships": result.relationships,
            "complete_relationships": result.complete_relationships,
            "tier_b_relationships": result.tier_b_relationships,
        }
        diagnostics = result.smart_diagnostics
        nonfloat = diagnostics.select_dtypes(exclude=["float"])
        fraction_columns = [
            column for column in diagnostics.columns
            if "BootstrapFraction" in column or "LOCOFraction" in column
        ]
        float_columns = diagnostics.select_dtypes(include=["float"]).columns
        masks = pd.DataFrame(index=diagnostics.index)
        for column in float_columns:
            values = diagnostics[column].to_numpy(dtype=np.float64)
            masks[column] = np.select(
                (
                    np.isnan(values),
                    np.isposinf(values),
                    np.isneginf(values),
                ),
                (1, 2, 3),
                default=0,
            ).astype(np.int8)
        first_twenty = result.complete_relationships.iloc[:20]
        historical = baseline["complete_relationships"]
        payload = {
            "format": "pedigree_seed72_minimal_shadow_v1",
            "source_sha256": require_source("minimal_before_output"),
            "driver_sha256": _source_sha256(Path(__file__).resolve()),
            "frame_digests": {
                name: _frame_digest(frame) for name, frame in frames.items()
            },
            "diagnostic_nonfloat_digest": _frame_digest(nonfloat),
            "diagnostic_fraction_digest": _frame_digest(
                diagnostics[fraction_columns]
            ),
            "diagnostic_float_mask_digest": _frame_digest(masks),
            "first_20_parent_state_counts": {
                str(key): int(value)
                for key, value in first_twenty["ParentState"]
                .value_counts(dropna=False).items()
            },
            "first_20_all_m0": bool(
                first_twenty["ParentState"].eq(
                    "zero_observed_parents"
                ).all()
            ),
            "all_parent_state_counts": {
                str(key): int(value)
                for key, value in result.complete_relationships[
                    "ParentState"
                ].value_counts(dropna=False).items()
            },
            "historical_parent_state_differences": int(np.count_nonzero(
                result.complete_relationships["ParentState"].to_numpy()
                != historical["ParentState"].to_numpy()
            )),
            "runtime": {
                "allocated_cpus": 112,
                "state_processes": state_processes,
                "bootstrap_workers": bootstrap_workers,
                "own_peak_rss_kib": int(
                    events["inference_end"]["own_peak_rss_kib"]
                ),
                "children_peak_rss_kib": int(
                    events["inference_end"]["children_peak_rss_kib"]
                ),
                "phases": phases,
            },
            "warnings": recorded_warnings,
            "publication_marker_written": False,
            "repository_outputs_written": False,
        }
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        require_source("minimal_after_temp")
        temporary.replace(output)
        print(
            f"SEED72_MINIMAL_COMPLETE output={output} "
            f"first_20_all_m0={payload['first_20_all_m0']} "
            f"elapsed={phases['total']['elapsed_seconds']:.3f}s",
            flush=True,
        )
        return 0


    frame_pairs = {
        "scientific_relationships": (
            result.relationships, baseline["scientific_relationships"]
        ),
        "complete_relationships": (
            result.complete_relationships, baseline["complete_relationships"]
        ),
        "tier_b_relationships": (
            result.tier_b_relationships, baseline["tier_b_relationships"]
        ),
    }
    frame_digests = {}
    for name, (observed, expected) in frame_pairs.items():
        pd.testing.assert_frame_equal(observed, expected, check_exact=True)
        frame_digests[name] = {
            "current_sha256": _frame_digest(observed),
            "baseline_sha256": _frame_digest(expected),
            "rows": len(observed),
        }

    observed_diagnostics = result.smart_diagnostics
    expected_diagnostics = baseline["smart_diagnostics"]
    if observed_diagnostics.columns.tolist() != expected_diagnostics.columns.tolist():
        raise AssertionError("seed-72 diagnostic schema differs")
    if observed_diagnostics["Sample"].tolist() != expected_diagnostics["Sample"].tolist():
        raise AssertionError("seed-72 diagnostic sample order differs")
    float_differences = _diagnostic_float_differences(
        observed_diagnostics, expected_diagnostics
    )
    exact_fraction_columns = [
        column for column in observed_diagnostics.columns
        if "BootstrapFraction" in column or "LOCOFraction" in column
    ]
    for column in exact_fraction_columns:
        np.testing.assert_array_equal(
            observed_diagnostics[column].to_numpy(),
            expected_diagnostics[column].to_numpy(),
            err_msg=column,
        )
    if result.smart_config != baseline["smart_config"]:
        raise AssertionError("result smart_config differs from seed-72 baseline")

    complete_truth, _ = oracle._evaluate_relationships(
        truth,
        result.complete_relationships,
        result.smart_diagnostics,
        tier_b=False,
    )
    tier_b_truth, _ = oracle._evaluate_relationships(
        truth,
        result.tier_b_relationships,
        result.smart_diagnostics,
        tier_b=True,
    )

    required_events = {
        "launch", "state_begin", "state_end",
        "bootstrap_begin", "bootstrap_end", "inference_end",
    }
    if set(events) != required_events:
        raise AssertionError(f"missing utilization boundary: {sorted(required_events - set(events))}")
    phases = {
        "pre_state": _phase_delta(events["launch"], events["state_begin"], 112),
        "state_workers": _phase_delta(
            events["state_begin"], events["state_end"], 112
        ),
        "pre_bootstrap": _phase_delta(
            events["state_end"], events["bootstrap_begin"], 112
        ),
        "bootstrap_workers": _phase_delta(
            events["bootstrap_begin"], events["bootstrap_end"], 112
        ),
        "tail": _phase_delta(
            events["bootstrap_end"], events["inference_end"], 112
        ),
        "total": _phase_delta(events["launch"], events["inference_end"], 112),
    }
    payload = {
        "format": "pedigree_seed72_read_only_shadow_v1",
        "created_unix_seconds": time.time(),
        "source_sha256": require_source("before_output"),
        "checkpoint_root": str(checkpoint_root),
        "compressed_input_bytes": int(compressed_input_bytes),
        "samples": len(sample_ids),
        "contigs": len(inputs),
        "config": {
            "bootstrap_replicates": config.bootstrap_replicates,
            "candidate_source_mode": config.parent_state_candidate_source_mode,
            "top_k": 20,
            "anchor_k": 5,
        },
        "comparisons": {
            "frames_exact": True,
            "frame_digests": frame_digests,
            "diagnostic_schema_and_nonfloat_values_exact": True,
            "bootstrap_and_loco_fractions_exact": exact_fraction_columns,
            "float_differences": float_differences,
            "config_exact": True,
        },
        "truth_summary": {
            "complete": complete_truth,
            "tier_b": tier_b_truth,
        },
        "runtime": {
            "allocated_cpus": 112,
            "state_processes": state_processes,
            "bootstrap_workers": bootstrap_workers,
            "smart_standard_input_threads": int(
                result.smart_standard_input_threads
            ),
            "smart_standard_input_processes": int(
                result.smart_standard_input_processes
            ),
            "smart_bootstrap_worker_count": int(
                result.smart_bootstrap_worker_count
            ),
            "own_peak_rss_kib": int(events["inference_end"]["own_peak_rss_kib"]),
            "children_peak_rss_kib": int(
                events["inference_end"]["children_peak_rss_kib"]
            ),
            "phases": phases,
        },
        "warnings": recorded_warnings,
        "publication_marker_written": False,
        "repository_outputs_written": False,
    }
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    require_source("after_temp_output")
    temporary.replace(output)
    print(
        f"SEED72_COMPLETE output={output} elapsed={phases['total']['elapsed_seconds']:.3f}s "
        f"own_peak_rss_kib={payload['runtime']['own_peak_rss_kib']} "
        f"children_peak_rss_kib={payload['runtime']['children_peak_rss_kib']}",
        flush=True,
    )
    return 0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="pedigree_inference")
    parser.add_argument(
        "--scenario", choices=("tropheops116", "synthetic320")
    )
    parser.add_argument("--seams", default="all")
    parser.add_argument("--output")
    parser.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seam", choices=SEAMS)
    parser.add_argument("--seed72-shadow", action="store_true")
    parser.add_argument("--seed72-root")
    parser.add_argument("--expected-source-hash")
    parser.add_argument("--reference-frames")
    parser.add_argument("--frames-output")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.repeats < 1:
        parser.error("--threads and --repeats must be positive")
    if args.seed72_shadow:
        if args.worker or args.seam is not None or args.scenario is not None:
            parser.error("seed72 shadow cannot be combined with seam modes")
        if not args.seed72_root or not args.expected_source_hash:
            parser.error(
                "seed72 shadow requires --seed72-root and --expected-source-hash"
            )
        if not args.output:
            parser.error("seed72 shadow requires --output below /tmp")
        return args
    if args.scenario is None:
        parser.error("benchmark mode requires --scenario")
    if args.worker and args.seam is None:
        parser.error("--worker requires --seam")
    if not args.worker and not args.output:
        parser.error("controller mode requires --output below /tmp")
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.seed72_shadow:
        return _seed72_shadow(args)
    return _worker(args) if args.worker else _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
