#!/usr/bin/env python3
"""Known-truth validation for canonical Stage-1 founder discovery.

The harness exercises the sole reversible-cavity model on simulated 200-SNP
blocks with random, tract, whole-site, founder-targeted, wholly missing-sample,
and genotype-dependent missingness. Simulation truth is used only after
inference; inference receives positions, allele depths, retained-site flags,
and its explicit numerical configuration.

Each task is checkpointed atomically by pattern, rate, and seed. Task identity
binds the simulation configuration, seed, harness, and directly exercised
production modules.

Examples
--------
Run a small integration smoke test::

    python stage1_validation.py --selftest --output-root /tmp/stage1-smoke

Run the default matrix on the current CPU affinity::

    python stage1_validation.py --output-root results_stage1_validation

The production scientific imports are lazy. Process-pool workers constrain
numerical libraries to one thread before importing Stage-1 modules.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime as _datetime
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import socket
import tempfile
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "stage1-validation-v1"
PATTERN_NAMES = (
    "none",
    "random",
    "tract",
    "founder_blackout",
    "rare_founder_blackout",
    "whole_site",
    "all_missing_samples",
    "mnar",
)
PRODUCTION_FILES = (
    "bhd_candidate_pool.py",
    "bhd_cavity_selection.py",
    "bhd_config.py",
    "bhd_factorization_modes.py",
    "bhd_fit.py",
    "bhd_genotype_evidence.py",
    "bhd_haplotype_mdl.py",
    "bhd_kernels.py",
    "bhd_mode_canonicalization.py",
    "bhd_model_selection.py",
    "bhd_results.py",
    "bhd_reversible_cavity.py",
    "bhd_reversible_discovery.py",
    "bhd_soft_seeding.py",
    "block_haplotypes.py",
    "dynamic_threads.py",
    "multiprocessing_runtime.py",
)


@dataclass(frozen=True)
class SimulationConfig:
    """Scientific inputs for one known-truth Stage-1 task."""

    n_sites: int = 200
    n_samples: int = 30
    n_founders: int = 4
    mean_depth: float = 12.0
    read_error_probability: float = 0.02
    min_total_site_reads: int = 5
    min_founder_hamming_fraction: float = 0.10
    founder_recovery_max_error: float = 0.02
    founder_recovery_min_identifiable_coverage: float = 0.80
    mnar_log_weight_per_alt_copy: float = 0.8
    rare_founder_carriers: int | None = None

    def validate(self) -> None:
        if self.n_sites != 200:
            raise ValueError("Stage-1 validation blocks must contain 200 SNPs")
        if self.n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        if self.n_founders < 1 or self.n_founders > 2 * self.n_samples:
            raise ValueError("n_founders must be in [1, 2*n_samples]")
        if self.rare_founder_carriers is not None:
            if self.n_founders < 2:
                raise ValueError(
                    "rare_founder_carriers requires at least two founders"
                )
            if (
                isinstance(self.rare_founder_carriers, bool)
                or self.rare_founder_carriers < 1
                or self.rare_founder_carriers > self.n_samples
            ):
                raise ValueError("rare_founder_carriers must be in [1, n_samples]")
        if self.mean_depth <= 0.0 or not math.isfinite(self.mean_depth):
            raise ValueError("mean_depth must be finite and positive")
        if not 0.0 < self.read_error_probability < 0.5:
            raise ValueError("read_error_probability must be in (0, 0.5)")
        if self.min_total_site_reads < 1:
            raise ValueError("min_total_site_reads must be positive")
        if not 0.0 <= self.min_founder_hamming_fraction <= 1.0:
            raise ValueError("min_founder_hamming_fraction must be in [0, 1]")
        if not 0.0 <= self.founder_recovery_max_error <= 1.0:
            raise ValueError("founder_recovery_max_error must be in [0, 1]")
        if not 0.0 <= self.founder_recovery_min_identifiable_coverage <= 1.0:
            raise ValueError(
                "founder_recovery_min_identifiable_coverage must be in [0, 1]"
            )


@dataclass(frozen=True)
class TaskSpec:
    pattern: str
    missing_rate: float
    seed: int
    simulation: SimulationConfig
    search_overrides: tuple[tuple[str, Any], ...] = ()

    def validate(self) -> None:
        if self.pattern not in PATTERN_NAMES:
            raise ValueError(f"unknown pattern {self.pattern!r}")
        if not 0.0 <= self.missing_rate < 1.0:
            raise ValueError("missing_rate must be in [0, 1)")
        if self.pattern == "none" and self.missing_rate != 0.0:
            raise ValueError("the none pattern requires missing_rate=0")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            self.pattern == "rare_founder_blackout"
            and self.simulation.rare_founder_carriers is None
        ):
            raise ValueError(
                "rare_founder_blackout requires rare_founder_carriers"
            )
        self.simulation.validate()

    def as_record(self) -> dict[str, Any]:
        result = asdict(self)
        result["search_overrides"] = dict(self.search_overrides)
        return result


@dataclass(frozen=True)
class TaskEnvelope:
    task: TaskSpec
    output_root: str
    code_identity: Mapping[str, Any]
    force: bool = False


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_identity(repo_root: Path) -> dict[str, Any]:
    paths = (Path(__file__).resolve(),) + tuple(
        (repo_root / name).resolve() for name in PRODUCTION_FILES
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"code-identity files are absent: {missing}")
    return {
        "schema_version": SCHEMA_VERSION,
        "files": {
            str(path.relative_to(repo_root.resolve()))
            if path.is_relative_to(repo_root.resolve())
            else str(path): _file_sha256(path)
            for path in paths
        },
    }


def _simulation_id(task: TaskSpec) -> str:
    simulation = task.as_record()
    simulation.pop("search_overrides")
    return _sha256_bytes(_canonical_json(simulation).encode("utf-8"))[:20]


def _task_id(task: TaskSpec, code_identity: Mapping[str, Any]) -> str:
    binding = {
        "schema_version": SCHEMA_VERSION,
        "task": task.as_record(),
        "code_identity": code_identity,
    }
    return _sha256_bytes(_canonical_json(binding).encode("utf-8"))[:24]


def _atomic_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_complete_checkpoint(
    path: Path, task_id: str, code_identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return None
    if (
        record.get("status") == "complete"
        and record.get("task_id") == task_id
        and record.get("code_identity") == code_identity
    ):
        return record
    return None


def _configure_one_numerical_thread() -> None:
    # These assignments occur before importing numerical libraries in a
    # spawned worker.  They prevent process x BLAS/OpenMP/Numba oversubscription.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _make_founders(rng: Any, config: SimulationConfig) -> Any:
    import numpy as np

    minimum_distance = max(
        1, int(round(config.min_founder_hamming_fraction * config.n_sites))
    )
    founders: list[Any] = []
    attempts = 0
    max_attempts = 10000
    while len(founders) < config.n_founders and attempts < max_attempts:
        candidate = rng.integers(0, 2, size=config.n_sites, dtype=np.int8)
        if all(int(np.sum(candidate != prior)) >= minimum_distance for prior in founders):
            founders.append(candidate)
        attempts += 1
    if len(founders) != config.n_founders:
        raise RuntimeError("could not generate sufficiently separated truth founders")
    return np.ascontiguousarray(np.stack(founders), dtype=np.int8)


def _balanced_diplotypes(rng: Any, config: SimulationConfig) -> Any:
    import numpy as np

    pairs = [
        (left, right)
        for left in range(config.n_founders)
        for right in range(left, config.n_founders)
    ]
    order = rng.permutation(len(pairs))
    shuffled_pairs = [pairs[int(index)] for index in order]
    result = np.empty((config.n_samples, 2), dtype=np.int64)
    for sample in range(config.n_samples):
        result[sample] = shuffled_pairs[sample % len(shuffled_pairs)]
    rng.shuffle(result, axis=0)
    return np.ascontiguousarray(result)

def _rare_founder_diplotypes(rng: Any, config: SimulationConfig) -> Any:
    """Construct diplotypes with founder 0 confined to exactly c carriers."""

    import numpy as np

    carrier_count = int(config.rare_founder_carriers)
    common_founders = np.arange(1, config.n_founders, dtype=np.int64)
    partner_order = common_founders[rng.permutation(len(common_founders))]
    rare_rows = [
        (0, int(partner_order[index % len(partner_order)]))
        for index in range(carrier_count)
    ]

    remaining_count = config.n_samples - carrier_count
    represented_common = {right for _, right in rare_rows}
    uncovered_common = set(int(value) for value in common_founders) - represented_common
    if len(uncovered_common) > 2 * remaining_count:
        raise ValueError(
            "rare-founder diplotypes cannot represent every common founder with "
            "the requested sample and carrier counts"
        )

    # Each candidate is a distinct unordered diplotype among common founders.
    # Shuffle before greedy coverage so repeated seeds are deterministic while
    # seeds still explore different balanced panels.
    common_pairs = [
        (left, right)
        for left in range(1, config.n_founders)
        for right in range(left, config.n_founders)
    ]
    common_pairs = [
        common_pairs[int(index)]
        for index in rng.permutation(len(common_pairs))
    ]
    common_rows: list[tuple[int, int]] = []
    available = list(common_pairs)
    while uncovered_common:
        best_index = max(
            range(len(available)),
            key=lambda index: len(uncovered_common.intersection(available[index])),
        )
        pair = available.pop(best_index)
        common_rows.append(pair)
        uncovered_common.difference_update(pair)
    if len(common_rows) > remaining_count:
        raise AssertionError("common-founder coverage exceeded available samples")

    next_pair = 0
    while len(common_rows) < remaining_count:
        if next_pair == len(available):
            available = [
                common_pairs[int(index)]
                for index in rng.permutation(len(common_pairs))
            ]
            next_pair = 0
        common_rows.append(available[next_pair])
        next_pair += 1

    result = np.asarray(rare_rows + common_rows, dtype=np.int64)
    rng.shuffle(result, axis=0)
    rare_carriers = np.any(result == 0, axis=1)
    if int(np.sum(rare_carriers)) != carrier_count:
        raise AssertionError("rare founder carrier count changed during construction")
    if np.any(np.all(result == 0, axis=1)):
        raise AssertionError("rare founder must never occur as a 0/0 diplotype")
    represented = set(int(value) for value in np.unique(result))
    if not set(int(value) for value in common_founders).issubset(represented):
        raise AssertionError("every common founder must be represented")
    return np.ascontiguousarray(result)


def _make_diplotypes(rng: Any, config: SimulationConfig) -> Any:
    if config.rare_founder_carriers is None:
        return _balanced_diplotypes(rng, config)
    return _rare_founder_diplotypes(rng, config)


def _solve_mnar_scale(weights: Any, target_rate: float) -> float:
    import numpy as np

    if target_rate <= 0.0:
        return 0.0
    lower, upper = 0.0, 1.0
    while float(np.mean(np.minimum(0.98, upper * weights))) < target_rate:
        upper *= 2.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        achieved = float(np.mean(np.minimum(0.98, midpoint * weights)))
        if achieved < target_rate:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _apply_missingness(
    rng: Any,
    reads: Any,
    truth_dosage: Any,
    truth_diplotypes: Any,
    task: TaskSpec,
) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    n_samples, n_sites = truth_dosage.shape
    missing = np.zeros((n_samples, n_sites), dtype=bool)
    rate = float(task.missing_rate)
    metadata: dict[str, Any] = {"requested_rate": rate}

    if task.pattern == "none":
        pass
    elif task.pattern == "random":
        missing = rng.random((n_samples, n_sites)) < rate
    elif task.pattern == "tract":
        tract_length = min(n_sites, max(1, int(round(rate * n_sites))))
        starts = rng.integers(0, n_sites - tract_length + 1, size=n_samples)
        for sample, start in enumerate(starts):
            missing[sample, int(start) : int(start) + tract_length] = True
        metadata.update(
            tract_length=int(tract_length),
            tract_starts=[int(value) for value in starts],
        )
    elif task.pattern == "founder_blackout":
        founder = int(rng.integers(0, task.simulation.n_founders))
        tract_length = min(n_sites, max(1, int(round(rate * n_sites))))
        start = int(rng.integers(0, n_sites - tract_length + 1))
        carriers = np.any(truth_diplotypes == founder, axis=1)
        missing[carriers, start : start + tract_length] = True
        metadata.update(
            target_founder=founder,
            tract_start=start,
            tract_length=int(tract_length),
            carrier_samples=np.flatnonzero(carriers).astype(int).tolist(),
        )
    elif task.pattern == "rare_founder_blackout":
        founder = 0
        tract_length = min(n_sites, max(1, int(round(rate * n_sites))))
        start = int(rng.integers(0, n_sites - tract_length + 1))
        carriers = np.any(truth_diplotypes == founder, axis=1)
        missing[carriers, start : start + tract_length] = True
        metadata.update(
            target_founder=founder,
            tract_start=start,
            tract_length=int(tract_length),
            carrier_samples=np.flatnonzero(carriers).astype(int).tolist(),
            rare_founder_carrier_count=int(np.sum(carriers)),
        )
    elif task.pattern == "whole_site":
        count = min(n_sites, max(1, int(round(rate * n_sites))))
        sites = np.sort(rng.choice(n_sites, size=count, replace=False))
        missing[:, sites] = True
        metadata["whole_cohort_sites"] = sites.astype(int).tolist()
    elif task.pattern == "all_missing_samples":
        count = min(n_samples, max(1, int(round(rate * n_samples))))
        samples = np.sort(rng.choice(n_samples, size=count, replace=False))
        missing[samples, :] = True
        metadata["all_missing_samples"] = samples.astype(int).tolist()
    elif task.pattern == "mnar":
        beta = float(task.simulation.mnar_log_weight_per_alt_copy)
        weights = np.exp(beta * (truth_dosage.astype(np.float64) - 1.0))
        scale = _solve_mnar_scale(weights, rate)
        probability = np.minimum(0.98, scale * weights)
        missing = rng.random((n_samples, n_sites)) < probability
        metadata.update(
            mnar_log_weight_per_alt_copy=beta,
            dropout_probability_by_truth_dosage={
                str(dosage): float(np.mean(probability[truth_dosage == dosage]))
                for dosage in (0, 1, 2)
                if np.any(truth_dosage == dosage)
            },
        )
    else:  # guarded by TaskSpec.validate
        raise AssertionError(f"unhandled pattern {task.pattern}")

    result = np.array(reads, copy=True)
    result[missing, :] = 0
    metadata["realized_deliberate_missing_rate"] = float(np.mean(missing))
    return np.ascontiguousarray(result), missing, metadata


def _array_bundle_sha256(arrays: Sequence[Any]) -> str:
    """Hash array dtype, shape, and canonical contiguous bytes in order."""
    import numpy as np

    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        descriptor = {"dtype": value.dtype.str, "shape": list(value.shape)}
        digest.update(_canonical_json(descriptor).encode("utf-8"))
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _simulate_task(task: TaskSpec) -> dict[str, Any]:
    import numpy as np

    # The same seed and simulation record always produce byte-identical
    # observable inputs and known truth.
    rng = np.random.default_rng(task.seed)
    config = task.simulation
    founders = _make_founders(rng, config)
    diplotypes = _make_diplotypes(rng, config)
    dosage = (
        founders[diplotypes[:, 0], :] + founders[diplotypes[:, 1], :]
    ).astype(np.int8)
    depth = rng.poisson(config.mean_depth, size=dosage.shape).astype(np.int64)
    alt_probability = np.asarray(
        [
            config.read_error_probability,
            0.5,
            1.0 - config.read_error_probability,
        ],
        dtype=np.float64,
    )[dosage]
    alt = rng.binomial(depth, alt_probability).astype(np.int64)
    reads = np.stack((depth - alt, alt), axis=2)
    reads, deliberate_missing, missing_metadata = _apply_missingness(
        rng, reads, dosage, diplotypes, task
    )
    total_site_reads = np.sum(reads, axis=(0, 2), dtype=np.int64)
    keep_threshold = max(
        config.min_total_site_reads,
        config.read_error_probability * config.n_samples,
    )
    keep_flags = (total_site_reads >= keep_threshold).astype(np.int64)
    if not np.any(keep_flags):
        raise RuntimeError("simulation retained no sites for Stage-1 inference")
    positions = np.arange(1, config.n_sites + 1, dtype=np.int64)
    information = np.sum(reads, axis=2) > 0
    rare_target_founder = 0 if config.rare_founder_carriers is not None else None
    rare_carriers = (
        np.any(diplotypes == rare_target_founder, axis=1)
        if rare_target_founder is not None
        else np.zeros(config.n_samples, dtype=bool)
    )
    observable_input_sha256 = _array_bundle_sha256(
        (positions, reads, keep_flags)
    )
    truth_sha256 = _array_bundle_sha256(
        (founders, diplotypes, dosage, deliberate_missing)
    )
    return {
        "positions": positions,
        "reads": np.ascontiguousarray(reads),
        "keep_flags": np.ascontiguousarray(keep_flags),
        "truth_founders": founders,
        "truth_diplotypes": diplotypes,
        "truth_dosage": dosage,
        "deliberate_missing": deliberate_missing,
        "information": information,
        "rare_target_founder": rare_target_founder,
        "rare_carriers": rare_carriers,
        "rare_founder_carrier_count": int(np.sum(rare_carriers)),
        "metadata": {
            **missing_metadata,
            "observable_input_sha256": observable_input_sha256,
            "truth_sha256": truth_sha256,
            "realized_zero_depth_rate": float(np.mean(~information)),
            "kept_site_count": int(np.sum(keep_flags)),
            "filtered_site_count": int(np.sum(keep_flags == 0)),
            "mean_depth_after_missing": float(np.mean(np.sum(reads, axis=2))),
            "rare_target_founder": rare_target_founder,
            "rare_founder_carrier_count": int(np.sum(rare_carriers)),
            "rare_founder_carrier_samples": (
                np.flatnonzero(rare_carriers).astype(int).tolist()
            ),
        },
    }


def _run_inference(
    task: TaskSpec,
    positions: Any,
    reads: Any,
    keep_flags: Any,
) -> Any:
    """Run Stage 1 without exposing any truth object to the inference call."""

    from bhd_reversible_cavity import ReversibleCavitySearchConfig
    from bhd_reversible_discovery import discover_block_reversible_cavity

    settings = ReversibleCavitySearchConfig(
        **dict(task.search_overrides),
    )
    return discover_block_reversible_cavity(
        positions,
        reads,
        keep_flags,
        config=settings,
    )


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _carrier_support(truth_diplotypes: Any, information: Any, n_founders: int) -> Any:
    import numpy as np

    support = np.zeros((n_founders, information.shape[1]), dtype=np.int64)
    for founder in range(n_founders):
        carriers = np.any(truth_diplotypes == founder, axis=1)
        support[founder, :] = np.sum(information[carriers, :], axis=0)
    return support


def _matching(truth_founders: Any, inferred_haps: Any) -> list[dict[str, Any]]:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n_truth, n_sites = truth_founders.shape
    n_inferred = inferred_haps.shape[0]
    if n_truth == 0 or n_inferred == 0:
        return []
    cost = np.empty((n_truth, n_inferred), dtype=np.float64)
    for truth_index in range(n_truth):
        for inferred_index in range(n_inferred):
            called = inferred_haps[inferred_index] >= 0
            mismatches = int(
                np.sum(
                    inferred_haps[inferred_index, called]
                    != truth_founders[truth_index, called]
                )
            )
            # An abstention has the expected loss of an uninformed binary call.
            # This aligns partially masked rows without rewarding blanket
            # abstention, while downstream metrics retain calls and errors
            # separately rather than collapsing them into this matching cost.
            cost[truth_index, inferred_index] = (
                mismatches + 0.5 * int(n_sites - np.sum(called))
            ) / n_sites
    truth_indices, inferred_indices = linear_sum_assignment(cost)
    return [
        {
            "truth_index": int(truth_index),
            "inferred_index": int(inferred_index),
            "matching_cost": float(cost[truth_index, inferred_index]),
        }
        for truth_index, inferred_index in zip(truth_indices, inferred_indices)
    ]


def _support_stratum(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 5:
        return "3-5"
    return "6+"
def _evaluate_rare_target(
    simulation: Mapping[str, Any],
    truth_founders: Any,
    inferred_haps: Any,
    support: Any,
    matched_records: Sequence[Mapping[str, Any]],
    translated_pairs: Any,
    assignment_resolved: Any,
    assignment_correct: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Evaluate the designated rare founder only after label alignment."""

    import numpy as np

    target_value = simulation.get("rare_target_founder")
    if target_value is None:
        return None, {}
    target = int(target_value)
    target_match = next(
        (
            record
            for record in matched_records
            if int(record["truth_index"]) == target
        ),
        None,
    )
    truth = np.asarray(truth_founders[target])
    called = np.zeros(len(truth), dtype=bool)
    errors = np.zeros(len(truth), dtype=bool)
    if target_match is not None:
        inferred_index = int(target_match["inferred_index"])
        inferred = np.asarray(inferred_haps[inferred_index])
        called = inferred >= 0
        errors = called & (inferred != truth)

    target_support = np.asarray(support[target], dtype=np.int64)
    target_strata: dict[str, Any] = {}
    for label in ("0", "1", "2", "3-5", "6+"):
        eligible = np.asarray(
            [_support_stratum(int(value)) == label for value in target_support],
            dtype=bool,
        )
        eligible_count = int(np.sum(eligible))
        called_count = int(np.sum(eligible & called))
        error_count = int(np.sum(eligible & errors))
        correct_count = called_count - error_count
        target_strata[label] = {
            "eligible": eligible_count,
            "called": called_count,
            "correct": correct_count,
            "errors": error_count,
            "hard_call_coverage": _ratio(called_count, eligible_count),
            "hard_call_error_rate": _ratio(error_count, called_count),
            "correct_hard_call_recall": _ratio(correct_count, eligible_count),
        }

    rare_carriers = np.asarray(simulation["rare_carriers"], dtype=bool)
    predicted_target = assignment_resolved & np.any(
        translated_pairs == target, axis=1
    )
    carrier_count = int(np.sum(rare_carriers))
    noncarrier_count = int(np.sum(~rare_carriers))
    carrier_target_assignments = int(np.sum(predicted_target & rare_carriers))
    noncarrier_target_assignments = int(np.sum(predicted_target & ~rare_carriers))
    resolved_carriers = int(np.sum(assignment_resolved & rare_carriers))
    exact_carrier_diplotypes = int(np.sum(assignment_correct & rare_carriers))
    hard_call_count = int(np.sum(called))
    hard_call_errors = int(np.sum(errors))
    hard_call_correct = hard_call_count - hard_call_errors
    identifiable = target_support > 0
    identifiable_count = int(np.sum(identifiable))
    identifiable_called = int(np.sum(identifiable & called))
    identifiable_errors = int(np.sum(identifiable & errors))

    record = {
        "truth_index": target,
        "matched": target_match is not None,
        "inferred_index": (
            None if target_match is None else int(target_match["inferred_index"])
        ),
        "recovered": (
            False if target_match is None else bool(target_match["recovered"])
        ),
        "exact_on_identifiable_sites": (
            False
            if target_match is None
            else bool(target_match["exact_on_identifiable_sites"])
        ),
        "hard_call_count": hard_call_count,
        "hard_call_coverage": _ratio(hard_call_count, len(truth)),
        "hard_call_errors": hard_call_errors,
        "hard_call_error_rate": _ratio(hard_call_errors, hard_call_count),
        "correct_hard_call_recall": _ratio(hard_call_correct, len(truth)),
        "identifiable_site_count": identifiable_count,
        "identifiable_hard_call_coverage": _ratio(
            identifiable_called, identifiable_count
        ),
        "identifiable_hard_call_error_rate": _ratio(
            identifiable_errors, identifiable_called
        ),
        "carrier_sample_count": carrier_count,
        "carrier_target_assignment_count": carrier_target_assignments,
        "carrier_assignment_recall": _ratio(
            carrier_target_assignments, carrier_count
        ),
        "resolved_carrier_count": resolved_carriers,
        "exact_carrier_diplotype_count": exact_carrier_diplotypes,
        "exact_carrier_diplotype_call_rate": _ratio(
            exact_carrier_diplotypes, carrier_count
        ),
        "noncarrier_sample_count": noncarrier_count,
        "noncarrier_false_target_assignment_count": noncarrier_target_assignments,
        "noncarrier_false_target_assignment_rate": _ratio(
            noncarrier_target_assignments, noncarrier_count
        ),
    }
    return record, target_strata


def _evaluate(task: TaskSpec, simulation: Mapping[str, Any], result: Any) -> dict[str, Any]:
    import numpy as np

    materialized = result.materialize()
    truth_founders = simulation["truth_founders"]
    truth_diplotypes = simulation["truth_diplotypes"]
    truth_dosage = simulation["truth_dosage"]
    information = simulation["information"]
    inferred_haps = np.asarray(materialized.discrete_haps, dtype=np.int64)
    inferred_pairs = np.asarray(materialized.pair_assignments, dtype=np.int64)
    wildcard_slots = np.asarray(materialized.wildcard_slots)
    n_samples = len(truth_diplotypes)
    if wildcard_slots.shape == (n_samples,):
        wildcard_sample = wildcard_slots > 0
    elif wildcard_slots.shape == (n_samples, 2):
        wildcard_sample = np.any(wildcard_slots.astype(bool), axis=1)
    else:
        raise ValueError("materialized wildcard_slots has an unexpected shape")
    assignment_policy_resolved = np.asarray(
        materialized.sample_has_observed_kept_depth,
        dtype=bool,
    )
    if assignment_policy_resolved.shape != (n_samples,):
        raise ValueError(
            "materialized sample depth mask has the wrong shape"
        )
    n_truth, n_sites = truth_founders.shape
    n_inferred = inferred_haps.shape[0]

    support = _carrier_support(truth_diplotypes, information, n_truth)
    matches = _matching(truth_founders, inferred_haps)
    stratum_counts = {
        label: {"eligible": 0, "called": 0, "correct": 0, "errors": 0}
        for label in ("0", "1", "2", "3-5", "6+")
    }
    matched_records: list[dict[str, Any]] = []
    recovered = 0
    exact_on_identifiable = 0

    for match in matches:
        truth_index = match["truth_index"]
        inferred_index = match["inferred_index"]
        inferred = inferred_haps[inferred_index]
        truth = truth_founders[truth_index]
        called = inferred >= 0
        errors = called & (inferred != truth)
        identifiable = support[truth_index] > 0
        identifiable_called = called & identifiable
        identifiable_errors = errors & identifiable
        identifiable_count = int(np.sum(identifiable))
        identifiable_coverage = _ratio(
            int(np.sum(identifiable_called)), identifiable_count
        )
        identifiable_error = _ratio(
            int(np.sum(identifiable_errors)), int(np.sum(identifiable_called))
        )
        is_recovered = bool(
            identifiable_count > 0
            and identifiable_coverage is not None
            and identifiable_coverage
            >= task.simulation.founder_recovery_min_identifiable_coverage
            and identifiable_error is not None
            and identifiable_error <= task.simulation.founder_recovery_max_error
        )
        is_exact_identifiable = bool(
            identifiable_count > 0
            and int(np.sum(identifiable_called)) == identifiable_count
            and int(np.sum(identifiable_errors)) == 0
        )
        recovered += int(is_recovered)
        exact_on_identifiable += int(is_exact_identifiable)
        matched_records.append(
            {
                **match,
                "hard_call_count": int(np.sum(called)),
                "hard_call_coverage": _ratio(int(np.sum(called)), n_sites),
                "hard_call_errors": int(np.sum(errors)),
                "hard_call_error_rate": _ratio(
                    int(np.sum(errors)), int(np.sum(called))
                ),
                "identifiable_site_count": identifiable_count,
                "identifiable_hard_call_coverage": identifiable_coverage,
                "identifiable_hard_call_error_rate": identifiable_error,
                "recovered": is_recovered,
                "exact_on_identifiable_sites": is_exact_identifiable,
            }
        )
        for site in range(n_sites):
            label = _support_stratum(int(support[truth_index, site]))
            counts = stratum_counts[label]
            counts["eligible"] += 1
            if called[site]:
                counts["called"] += 1
                if errors[site]:
                    counts["errors"] += 1
                else:
                    counts["correct"] += 1

    # A truth founder omitted by inference contributes no calls, but its sites
    # remain in panel-level coverage and carrier-support denominators.
    matched_truth_indices = {item["truth_index"] for item in matches}
    for truth_index in set(range(n_truth)) - matched_truth_indices:
        for site in range(n_sites):
            label = _support_stratum(int(support[truth_index, site]))
            stratum_counts[label]["eligible"] += 1

    stratum_metrics: dict[str, Any] = {}
    for label, counts in stratum_counts.items():
        stratum_metrics[label] = {
            **counts,
            "hard_call_coverage": _ratio(counts["called"], counts["eligible"]),
            "hard_call_error_rate": _ratio(counts["errors"], counts["called"]),
        }

    # Translate inferred founder labels into truth labels using only the
    # post-inference Hungarian match.  This evaluates assignments without
    # exposing truth to fitting.
    inferred_to_truth = {
        item["inferred_index"]: item["truth_index"] for item in matches
    }
    assignment_resolved = np.zeros(len(truth_diplotypes), dtype=bool)
    assignment_correct = np.zeros(len(truth_diplotypes), dtype=bool)
    predicted_dosage = np.full(truth_dosage.shape, -1, dtype=np.int8)
    translated_pairs = np.full(
        (len(truth_diplotypes), 2), -1, dtype=np.int64
    )
    for sample in range(len(truth_diplotypes)):
        left, right = (int(value) for value in inferred_pairs[sample])
        has_wildcard = bool(wildcard_sample[sample])
        labels_exist = left in inferred_to_truth and right in inferred_to_truth
        if assignment_policy_resolved[sample] and not has_wildcard and labels_exist and left < n_inferred and right < n_inferred:
            assignment_resolved[sample] = True
            translated = sorted((inferred_to_truth[left], inferred_to_truth[right]))
            translated_pairs[sample] = translated
            assignment_correct[sample] = translated == sorted(
                int(value) for value in truth_diplotypes[sample]
            )
            callable_sites = (inferred_haps[left] >= 0) & (inferred_haps[right] >= 0)
            predicted_dosage[sample, callable_sites] = (
                inferred_haps[left, callable_sites] + inferred_haps[right, callable_sites]
            )

    rare_target, rare_target_strata = _evaluate_rare_target(
        simulation,
        truth_founders,
        inferred_haps,
        support,
        matched_records,
        translated_pairs,
        assignment_resolved,
        assignment_correct,
    )

    hidden = ~information
    observed = information
    predicted = predicted_dosage >= 0
    hidden_called = hidden & predicted
    observed_called = observed & predicted
    hidden_errors = hidden_called & (predicted_dosage != truth_dosage)
    observed_errors = observed_called & (predicted_dosage != truth_dosage)

    all_missing_samples = np.all(~information, axis=1)
    all_missing_count = int(np.sum(all_missing_samples))
    all_missing_unresolved = int(
        np.sum(all_missing_samples & ~assignment_resolved)
    )

    total_panel_eligible = sum(
        item["eligible"] for item in stratum_counts.values()
    )
    total_matched_called = sum(item["called"] for item in stratum_counts.values())
    total_matched_errors = sum(item["errors"] for item in stratum_counts.values())
    total_matched_eligible = len(matches) * n_sites
    unsupported = stratum_counts["0"]
    metrics = {
        "true_k": int(n_truth),
        "selected_k": int(n_inferred),
        "k_error": int(n_inferred - n_truth),
        "exact_k": bool(n_inferred == n_truth),
        "matched_founder_count": len(matches),
        "founder_recovered_count": int(recovered),
        "founder_recall": _ratio(recovered, n_truth),
        "founder_precision": _ratio(recovered, n_inferred),
        "founder_exact_on_identifiable_count": int(exact_on_identifiable),
        "founder_exact_on_identifiable_recall": _ratio(
            exact_on_identifiable, n_truth
        ),
        "matched_hard_call_coverage": _ratio(
            total_matched_called, total_matched_eligible
        ),
        "panel_hard_call_coverage": _ratio(
            total_matched_called, total_panel_eligible
        ),
        "panel_correct_hard_call_recall": _ratio(
            total_matched_called - total_matched_errors, total_panel_eligible
        ),
        "matched_hard_call_error_rate": _ratio(
            total_matched_errors, total_matched_called
        ),
        "unsupported_site_count": int(unsupported["eligible"]),
        "unsupported_false_hard_call_count": int(unsupported["called"]),
        "unsupported_false_hard_call_rate": _ratio(
            unsupported["called"], unsupported["eligible"]
        ),
        "unsupported_hard_call_error_rate": _ratio(
            unsupported["errors"], unsupported["called"]
        ),
        "hidden_genotype_count": int(np.sum(hidden)),
        "hidden_genotype_hard_call_coverage": _ratio(
            int(np.sum(hidden_called)), int(np.sum(hidden))
        ),
        "hidden_genotype_hard_call_error_rate": _ratio(
            int(np.sum(hidden_errors)), int(np.sum(hidden_called))
        ),
        "observed_genotype_count": int(np.sum(observed)),
        "observed_genotype_hard_call_coverage": _ratio(
            int(np.sum(observed_called)), int(np.sum(observed))
        ),
        "observed_genotype_hard_call_error_rate": _ratio(
            int(np.sum(observed_errors)), int(np.sum(observed_called))
        ),
        "assignment_resolved_rate": _ratio(
            int(np.sum(assignment_resolved)), len(assignment_resolved)
        ),
        "assignment_accuracy_when_resolved": _ratio(
            int(np.sum(assignment_correct)), int(np.sum(assignment_resolved))
        ),
        "wildcard_sample_rate": float(np.mean(wildcard_sample)),
        "all_missing_sample_count": all_missing_count,
        "all_missing_sample_unresolved_count": all_missing_unresolved,
        "all_missing_sample_unresolved_rate": _ratio(
            all_missing_unresolved, all_missing_count
        ),
        "search_limited": bool(
            getattr(result.diagnostics.candidate_search, "search_limited", False)
        ),
    }
    if rare_target is not None:
        metrics.update({
            "rare_target_matched": rare_target["matched"],
            "rare_target_recovered": rare_target["recovered"],
            "rare_target_exact_on_identifiable_sites": (
                rare_target["exact_on_identifiable_sites"]
            ),
            "rare_target_hard_call_coverage": rare_target["hard_call_coverage"],
            "rare_target_hard_call_error_rate": (
                rare_target["hard_call_error_rate"]
            ),
            "rare_target_correct_hard_call_recall": (
                rare_target["correct_hard_call_recall"]
            ),
            "rare_target_carrier_assignment_recall": (
                rare_target["carrier_assignment_recall"]
            ),
            "rare_target_exact_carrier_diplotype_call_rate": (
                rare_target["exact_carrier_diplotype_call_rate"]
            ),
            "rare_target_noncarrier_false_target_assignment_rate": (
                rare_target["noncarrier_false_target_assignment_rate"]
            ),
            "rare_target_unsupported_false_hard_call_rate": (
                rare_target_strata["0"]["hard_call_coverage"]
            ),
        })
    return {
        "metrics": metrics,
        "rare_target": rare_target,
        "rare_target_support_strata": rare_target_strata,
        "carrier_support_strata": stratum_metrics,
        "founder_matches": matched_records,
    }


def _peak_rss_mib() -> float:
    import resource

    # Linux reports KiB; this repository runs on CSD3 Linux.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _run_task(envelope: TaskEnvelope) -> dict[str, Any]:
    _configure_one_numerical_thread()
    task = envelope.task
    task.validate()
    repo_root = Path(__file__).resolve().parent
    current_identity = _code_identity(repo_root)
    if current_identity != envelope.code_identity:
        raise RuntimeError(
            "source files changed after task dispatch; refusing an unbound run"
        )
    task_id = _task_id(task, current_identity)
    output_path = Path(envelope.output_root) / "tasks" / f"task-{task_id}.json"
    if not envelope.force:
        checkpoint = _read_complete_checkpoint(
            output_path, task_id, current_identity
        )
        if checkpoint is not None:
            return {
                "task_id": task_id,
                "path": str(output_path),
                "status": "skipped_complete",
            }

    started = _utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    base_record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "simulation_id": _simulation_id(task),
        "task": task.as_record(),
        "code_identity": current_identity,
        "started_utc": started,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "numerical_threads": 1,
    }
    try:
        simulation = _simulate_task(task)
        inference_start = time.perf_counter()
        # Pass only observable inputs across the inference boundary.  Truth
        # founders, diplotypes, dosages, and missingness labels stay here and
        # are first used by _evaluate after discovery has returned.
        result = _run_inference(
            task,
            simulation["positions"],
            simulation["reads"],
            simulation["keep_flags"],
        )
        inference_seconds = time.perf_counter() - inference_start
        evaluation = _evaluate(task, simulation, result)
        record = {
            **base_record,
            "status": "complete",
            "completed_utc": _utc_now(),
            "simulation_summary": simulation["metadata"],
            **evaluation,
            "performance": {
                "inference_wall_seconds": float(inference_seconds),
                "total_wall_seconds": float(time.perf_counter() - wall_start),
                "process_cpu_seconds": float(time.process_time() - cpu_start),
                "peak_rss_mib": _peak_rss_mib(),
            },
        }
    except BaseException as error:
        record = {
            **base_record,
            "status": "error",
            "completed_utc": _utc_now(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "performance": {
                "total_wall_seconds": float(time.perf_counter() - wall_start),
                "process_cpu_seconds": float(time.process_time() - cpu_start),
                "peak_rss_mib": _peak_rss_mib(),
            },
        }
    _atomic_json(output_path, record)
    return {
        "task_id": task_id,
        "path": str(output_path),
        "status": record["status"],
        "error": record.get("error", {}).get("message"),
    }


def _parse_seeds(value: str) -> list[int]:
    if ":" in value:
        fields = value.split(":")
        if len(fields) not in (2, 3):
            raise argparse.ArgumentTypeError("seed range must be START:STOP[:STEP]")
        start, stop = int(fields[0]), int(fields[1])
        step = int(fields[2]) if len(fields) == 3 else 1
        if start < 0 or stop < 0 or step <= 0:
            raise argparse.ArgumentTypeError("seed ranges must be non-negative")
        result = list(range(start, stop, step))
    else:
        result = [int(field) for field in value.split(",") if field]
    if not result or any(seed < 0 for seed in result):
        raise argparse.ArgumentTypeError("at least one non-negative seed is required")
    return result


def _parse_float_list(value: str) -> list[float]:
    result = [float(field) for field in value.split(",") if field]
    if not result or any(not 0.0 <= item < 1.0 for item in result):
        raise argparse.ArgumentTypeError("rates must be comma-separated values in [0,1)")
    return result


DEFAULT_RATES: Mapping[str, tuple[float, ...]] = {
    "none": (0.0,),
    "random": (0.10, 0.25, 0.40, 0.60, 0.80),
    "tract": (0.10, 0.25, 0.50),
    "founder_blackout": (0.10, 0.30),
    "whole_site": (0.10, 0.25),
    "rare_founder_blackout": (0.30,),
    "all_missing_samples": (0.10, 0.25),
    "mnar": (0.10, 0.25, 0.40),
}


def _build_tasks(arguments: argparse.Namespace) -> list[TaskSpec]:
    if arguments.selftest:
        simulation = SimulationConfig(
            n_sites=200,
            n_samples=12,
            n_founders=3,
            mean_depth=10.0,
            read_error_probability=arguments.read_error_probability,
        )
        # Operationally reduced search checks end-to-end integration only.
        overrides = tuple(sorted({
            "beam_width": 1,
            "max_expansions": 4,
            "max_exact_scores": 12,
            "max_proposals_per_expansion": 24,
            "data_start_beam_width": 2,
            "n_data_seed_modes": 2,
            "max_candidate_start_rows": 8,
            "max_replacement_children_per_mode": 4,
            "read_error_probability": arguments.read_error_probability,
        }.items()))
        tasks = [
            TaskSpec(
                pattern=pattern,
                missing_rate=rate,
                seed=arguments.seeds[0],
                simulation=simulation,
                search_overrides=overrides,
            )
            for pattern, rate in (
                ("founder_blackout", 0.25),
                ("all_missing_samples", 0.25),
            )
        ]
    else:
        simulation = SimulationConfig(
            n_sites=200,
            n_samples=arguments.n_samples,
            n_founders=arguments.n_founders,
            mean_depth=arguments.mean_depth,
            read_error_probability=arguments.read_error_probability,
            rare_founder_carriers=arguments.rare_founder_carriers,
        )
        overrides = tuple(sorted({
            "min_directional_supporters": (
                arguments.min_directional_supporters
            ),
            "min_hard_call_pseudo_probability": (
                arguments.min_hard_call_pseudo_probability
            ),
            "read_error_probability": arguments.read_error_probability,
        }.items()))
        tasks = []
        for pattern in arguments.patterns:
            rates = arguments.rates or list(DEFAULT_RATES[pattern])
            if pattern == "none":
                rates = [0.0]
            for rate in rates:
                for seed in arguments.seeds:
                    tasks.append(TaskSpec(
                        pattern=pattern,
                        missing_rate=rate,
                        seed=seed,
                        search_overrides=overrides,
                        simulation=simulation,
                    ))
    for task in tasks:
        task.validate()
    unique_tasks = {
        _canonical_json(task.as_record()): task for task in tasks
    }
    return list(unique_tasks.values())


SUMMARY_METRICS = (
    "selected_k",
    "k_error",
    "exact_k",
    "founder_recall",
    "founder_precision",
    "founder_exact_on_identifiable_recall",
    "matched_hard_call_coverage",
    "panel_hard_call_coverage",
    "panel_correct_hard_call_recall",
    "matched_hard_call_error_rate",
    "unsupported_false_hard_call_rate",
    "unsupported_hard_call_error_rate",
    "hidden_genotype_hard_call_coverage",
    "hidden_genotype_hard_call_error_rate",
    "assignment_resolved_rate",
    "assignment_accuracy_when_resolved",
    "all_missing_sample_unresolved_rate",
    "rare_target_matched",
    "rare_target_recovered",
    "rare_target_exact_on_identifiable_sites",
    "rare_target_hard_call_coverage",
    "rare_target_hard_call_error_rate",
    "rare_target_correct_hard_call_recall",
    "rare_target_carrier_assignment_recall",
    "rare_target_exact_carrier_diplotype_call_rate",
    "rare_target_noncarrier_false_target_assignment_rate",
    "rare_target_unsupported_false_hard_call_rate",
)


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return None if not numeric else float(sum(numeric) / len(numeric))


def _write_summary(
    output_root: Path,
    tasks: Sequence[TaskSpec],
    code_identity: Mapping[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        task_id = _task_id(task, code_identity)
        path = output_root / "tasks" / f"task-{task_id}.json"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    complete = [
        record for record in records if record.get("status") == "complete"
    ]
    errors = [
        record for record in records if record.get("status") == "error"
    ]

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for record in complete:
        task = record["task"]
        key = (task["pattern"], float(task["missing_rate"]))
        grouped.setdefault(key, []).append(record)
    group_records = []
    for (pattern, rate), group in sorted(grouped.items()):
        group_records.append({
            "pattern": pattern,
            "missing_rate": rate,
            "n": len(group),
            "mean_metrics": {
                name: _mean(item["metrics"].get(name) for item in group)
                for name in SUMMARY_METRICS
            },
            "mean_performance": {
                name: _mean(item["performance"].get(name) for item in group)
                for name in (
                    "inference_wall_seconds",
                    "total_wall_seconds",
                    "process_cpu_seconds",
                    "peak_rss_mib",
                )
            },
        })

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "code_identity": code_identity,
        "requested_task_count": len(tasks),
        "complete_task_count": len(complete),
        "error_task_count": len(errors),
        "missing_task_count": len(tasks) - len(records),
        "groups": group_records,
        "errors": [
            {
                "task_id": record["task_id"],
                "task": record["task"],
                "error": record.get("error"),
            }
            for record in errors
        ],
        "metric_interpretation": {
            "matching": (
                "Hungarian one-to-one founder matching with binary mismatch "
                "loss 1 and unknown-call loss 0.5; metrics report hard-call "
                "coverage and error separately"
            ),
            "founder_recovery": (
                "matched truth founder with configured hard-call coverage on "
                "sites having an informative true carrier and configured "
                "maximum hard-call error"
            ),
            "unsupported_false_hard_call": (
                "a hard allele emitted where no true carrier has read depth"
            ),
        },
    }
    _atomic_json(output_root / "summary.json", summary)
    return summary


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="isolated root for atomic task JSON files and summary.json",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run two reduced-budget 200-SNP integration tasks",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=PATTERN_NAMES,
        default=[
            "none",
            "random",
            "tract",
            "founder_blackout",
            "whole_site",
            "all_missing_samples",
            "mnar",
        ],
    )
    parser.add_argument(
        "--rates",
        type=_parse_float_list,
        default=None,
        help="override pattern-specific rates with a comma-separated list",
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=list(range(8)))
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--n-founders", type=int, default=4)
    parser.add_argument("--mean-depth", type=float, default=12.0)
    parser.add_argument(
        "--min-directional-supporters",
        type=int,
        default=2,
        help="minimum directional carriers for a founder-site hard call",
    )
    parser.add_argument(
        "--rare-founder-carriers",
        type=int,
        default=None,
        help="place truth founder 0 heterozygously in this many samples",
    )
    parser.add_argument(
        "--min-hard-call-pseudo-probability",
        type=float,
        default=0.85,
        help=(
            "minimum capped fixed-assignment pseudo-probability for a "
            "founder-site hard call"
        ),
    )
    parser.add_argument("--read-error-probability", type=float, default=0.02)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="one-thread workers (default: min(tasks, CPU affinity))",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace compatible completed task files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    code_identity = _code_identity(repo_root)
    tasks = _build_tasks(arguments)
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    affinity = int(affinity or 1)
    workers = min(len(tasks), affinity) if arguments.workers is None else arguments.workers
    if workers < 1 or workers > affinity:
        raise ValueError(f"workers must be in [1, current CPU affinity {affinity}]")
    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "code_identity": code_identity,
        "task_count": len(tasks),
        "simulation_count": len({_simulation_id(task) for task in tasks}),
        "patterns": sorted({task.pattern for task in tasks}),
        "cpu_affinity": affinity,
        "workers": workers,
        "numerical_threads_per_worker": 1,
        "aggregate_process_thread_budget": workers,
        "selftest": bool(arguments.selftest),
        "tasks": [
            {
                "task_id": _task_id(task, code_identity),
                "simulation_id": _simulation_id(task),
                "task": task.as_record(),
            }
            for task in tasks
        ],
    }
    _atomic_json(output_root / "manifest.json", manifest)
    print(
        f"Stage-1 validation: {len(tasks)} tasks, {workers} spawned "
        f"workers x 1 numerical thread, affinity={affinity}"
    )
    print(f"Output root: {output_root}")

    envelopes = [
        TaskEnvelope(
            task=task,
            output_root=str(output_root),
            code_identity=code_identity,
            force=bool(arguments.force),
        )
        for task in tasks
    ]
    statuses: dict[str, int] = {}
    if workers == 1:
        iterator = map(_run_task, envelopes)
        pool = None
    else:
        # The project-standard forkserver avoids inheriting a live numerical
        # runtime. Its minimal preload shares scientific import pages across
        # workers, lowering startup time and memory for a many-task matrix.
        _configure_one_numerical_thread()
        try:
            context = multiprocessing.get_context("forkserver")
            multiprocessing.set_forkserver_preload(
                ["thread_config", "bhd_reversible_discovery"]
            )
        except (ValueError, AttributeError):
            context = multiprocessing.get_context("spawn")
        pool = context.Pool(processes=workers)
        iterator = pool.imap_unordered(_run_task, envelopes, chunksize=1)
    try:
        for index, outcome in enumerate(iterator, start=1):
            status = str(outcome["status"])
            statuses[status] = statuses.get(status, 0) + 1
            detail = f": {outcome['error']}" if outcome.get("error") else ""
            print(f"[{index}/{len(tasks)}] {outcome['task_id']} {status}{detail}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    summary = _write_summary(output_root, tasks, code_identity)
    print(
        "Completed: "
        f"{summary['complete_task_count']} complete, "
        f"{summary['error_task_count']} error, "
        f"{summary['missing_task_count']} missing"
    )
    print(f"Summary: {output_root / 'summary.json'}")
    return 0 if summary["error_task_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
