"""Predeclared analysis for Smart pedigree release calibration and holdout.

The protocol in :data:`ANALYSIS_PROTOCOL` was fixed before reading the
calibration outcomes.  Biological truth counts every simulated parent present
in the candidate panel.  Observable truth is a secondary diagnostic that drops
a biological edge only when its simulated identity information is exactly
zero.  Model selection uses biological truth only.

Calibration selects the smallest tested effective-marker value satisfying all
predeclared constraints.  Holdout requires the resulting frozen selection JSON
and never compares or selects among the other effective-marker settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
PROTOCOL_ID = "pedigree_smart_release_analysis_v1_20260829"
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_REPLICATES = 2_000
CI_LEVEL = 0.95
ECE_EDGES = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))

# These are known relationships created by the simulation, not relationships
# inferred from cohort labels.  A substitution requires omission of the true
# anchor and release of at least one of its simulated related decoys.
RELATED_DECOYS = {
    "candidate_00": ("candidate_01", "candidate_02", "candidate_03"),
    "candidate_08": ("candidate_09", "candidate_10", "candidate_11"),
}

V1_RELATIONSHIP_LABELS = {
    "0": "female_anchor",
    "1": "female_full_sibling_decoy",
    "2": "female_half_sibling_decoy",
    "3": "female_near_duplicate_decoy",
    "8": "male_anchor",
    "9": "male_full_sibling_decoy",
    "10": "male_half_sibling_decoy",
    "11": "male_near_duplicate_decoy",
}

BOOTSTRAP_THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 1.0)
LOCO_THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 1.0)
SUPPORT_THRESHOLDS = (0.50, 2.0 / 3.0, 0.80, 0.90, 0.95, 0.99)
JOINT_THRESHOLDS = (
    (0.50, 0.50, 0.50),
    (0.70, 0.80, 2.0 / 3.0),
    (0.80, 0.80, 0.80),
    (0.90, 0.90, 0.80),
    (0.95, 0.95, 0.90),
    (0.975, 0.975, 0.95),
    (0.99, 1.0, 0.99),
    (1.0, 1.0, 0.99),
)

ANALYSIS_PROTOCOL = {
    "schema_version": SCHEMA_VERSION,
    "protocol_id": PROTOCOL_ID,
    "frozen_before_outcome_review": True,
    "truth_definitions": {
        "biological": (
            "simulated biological parents that are present in the eligible "
            "candidate panel; this is the sole calibration-selection target"
        ),
        "observable": (
            "biological candidate-panel parents with strictly positive "
            "simulated parent-child identity information; diagnostic only"
        ),
    },
    "uncertainty": {
        "cluster": "replicate",
        "method": "replicate-cluster percentile bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": CI_LEVEL,
    },
    "input_validation": {
        "runner_v1_integer_key_normalization": (
            "The v1 runner hashed integer candidate_relationships keys before "
            "JSON persistence. Analysis reconstructs only those digit-only "
            "keys and still requires the completion-marker hash to match."
        ),
    },
    "calibration": {
        "ece_bins": list(ECE_EDGES),
        "state_support_target": "biological truth (observable also reported)",
    },
    "risk_coverage": {
        "bootstrap_thresholds": list(BOOTSTRAP_THRESHOLDS),
        "loco_thresholds": list(LOCO_THRESHOLDS),
        "minimum_state_support_thresholds": list(SUPPORT_THRESHOLDS),
        "joint_thresholds": [list(values) for values in JOINT_THRESHOLDS],
        "interpretation": (
            "predeclared sensitivity curves; thresholds are not tuned and do "
            "not replace the runner's Tier A/B release policy"
        ),
    },
    "selection": {
        "unit": "one global effective-marker value across all scenarios",
        "ordering": "smallest numeric value satisfying every constraint",
        "minimum_replicate_clusters_per_scenario_state": 40,
        "biological_raw_state_accuracy": {
            "applies_to": "every scenario separately and each of M0/M1/M2",
            "minimum_point_estimate": 0.90,
            "minimum_ci_lower": 0.80,
        },
        "tier_a_m0_false_parent_release": {
            "definition": (
                "Tier A releases a nonzero state or any Tier A partial parent "
                "edge for a biological M0 child"
            ),
            "maximum_point_estimate": 0.02,
            "maximum_ci_upper": 0.05,
        },
        "tier_a_m1_to_m2_error": {
            "definition": (
                "Tier A releases state M2 for a biological M1 child; denominator "
                "is all biological M1 children"
            ),
            "maximum_point_estimate": 0.02,
            "maximum_ci_upper": 0.05,
        },
        "tier_a_related_decoy_substitution": {
            "definition": (
                "Tier A partial output omits a true simulated anchor and releases "
                "one of that anchor's predeclared related decoys; denominator is "
                "all at-risk true anchor edges"
            ),
            "maximum_point_estimate": 0.02,
            "maximum_ci_upper": 0.05,
        },
        "release_threshold_tuning": False,
        "release_policy": "runner Tier A/B thresholds unchanged",
        "failure_policy": (
            "write a frozen failed selection with no selected value and exit 2"
        ),
    },
}

REQUIRED_FIELDS = {
    "replicate",
    "scenario",
    "effective_markers",
    "sample",
    "stratum",
    "truth_state",
    "truth_parents",
    "observable_truth_state",
    "observable_truth_parents",
    "complete_observed_parent_count",
    "complete_parent1",
    "complete_parent2",
    "tier_a_observed_parent_count",
    "tier_a_parent1",
    "tier_a_parent2",
    "tier_b_observed_parent_count",
    "tier_b_parent1",
    "tier_b_parent2",
    "tier_a_partial_parent1",
    "tier_a_partial_parent2",
    "tier_b_partial_parent1",
    "tier_b_partial_parent2",
    "diagnostic_LocalObservedParentCount",
    "diagnostic_InformativeContigCount",
    "diagnostic_StateSupport0",
    "diagnostic_StateSupport1",
    "diagnostic_StateSupport2",
    "diagnostic_LocalStateBootstrapFraction",
    "diagnostic_LocalConfigurationBootstrapFraction",
    "diagnostic_GraphConfigurationBootstrapFraction",
    "diagnostic_Parent1BootstrapFraction",
    "diagnostic_Parent2BootstrapFraction",
    "diagnostic_LocalStateLOCOFraction",
    "diagnostic_LocalConfigurationLOCOFraction",
    "diagnostic_GraphConfigurationLOCOFraction",
    "diagnostic_Parent1LOCOFraction",
    "diagnostic_Parent2LOCOFraction",
    "diagnostic_TierAStateCall",
    "diagnostic_TierBStateCall",
    "diagnostic_TierAExactConfiguration",
    "diagnostic_TierBExactConfiguration",
    "diagnostic_GraphConflict",
    "diagnostic_GraphTieConflict",
}


class AnalysisError(RuntimeError):
    """Input or analysis failure with a concise CLI message."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("calibration", "holdout", "holdout-comparison"),
        required=True,
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        help="required in holdout mode; ignored nowhere and forbidden in calibration",
    )
    return parser.parse_args(argv)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    fields = sorted({field for row in materialized for field in row})
    if not fields:
        raise AnalysisError(f"refusing to write empty table: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _output_paths(prefix: Path, mode: str) -> dict[str, Path]:
    paths = {
        "protocol": Path(f"{prefix}.protocol.json"),
        "metrics": Path(f"{prefix}.metrics.csv"),
        "confusion": Path(f"{prefix}.raw_state_confusion.csv"),
        "calibration": Path(f"{prefix}.state_support_calibration.csv"),
        "risk_coverage": Path(f"{prefix}.risk_coverage.csv"),
        "summary": Path(f"{prefix}.summary.json"),
    }
    if mode == "calibration":
        paths["selection"] = Path(f"{prefix}.selection.json")
    return paths


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validated_completed_input(root: Path, mode: str) -> tuple[dict, dict, Path, str]:
    root = root.resolve()
    status_path = root / "status.computational_complete.json"
    if not status_path.is_file():
        raise AnalysisError(f"completion marker is absent: {status_path}")
    status = _load_json(status_path)
    if not isinstance(status, dict) or status.get("computational_complete") is not True:
        raise AnalysisError("completion marker does not assert computational completion")
    outputs = status.get("outputs")
    if not isinstance(outputs, dict):
        raise AnalysisError("completion marker has no output identity mapping")
    validated = {}
    for name, identity in outputs.items():
        if not isinstance(identity, dict):
            raise AnalysisError(f"invalid completed-output identity for {name}")
        path = (root / str(identity.get("path", ""))).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise AnalysisError(f"completed output escapes input root: {path}") from exc
        if not path.is_file():
            raise AnalysisError(f"completed output is missing: {path}")
        if path.stat().st_size != int(identity.get("size_bytes", -1)):
            raise AnalysisError(f"completed output size changed: {path}")
        if _sha256(path) != identity.get("sha256"):
            raise AnalysisError(f"completed output digest changed: {path}")
        validated[name] = path
    manifest_path = validated.get("manifest.json", root / "manifest.json")
    records_path = validated.get(
        "per_child_release_metrics.csv", root / "per_child_release_metrics.csv"
    )
    if not manifest_path.is_file() or not records_path.is_file():
        raise AnalysisError("manifest or per-child metrics are absent from completed input")
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise AnalysisError("manifest is not a JSON object")
    split = manifest.get("parameters", {}).get("split")
    if split != mode:
        raise AnalysisError(f"input split {split!r} does not match analysis mode {mode!r}")
    persisted_hash = _canonical_hash(manifest)
    identity_normalization = None
    if status.get("manifest_sha256") != persisted_hash:
        relationships = manifest.get("fixed_design", {}).get(
            "candidate_relationships"
        )
        if not (
            manifest.get("attempt_identity")
            == "pedigree_smart_release_known_truth_v1"
            and isinstance(relationships, dict)
            and relationships
            and all(str(key).isdigit() for key in relationships)
        ):
            raise AnalysisError(
                "manifest canonical identity does not match completion marker"
            )
        reconstructed = json.loads(json.dumps(manifest))
        reconstructed["fixed_design"]["candidate_relationships"] = {
            int(key): value for key, value in relationships.items()
        }
        if status.get("manifest_sha256") != _canonical_hash(reconstructed):
            raise AnalysisError(
                "v1 integer-key reconstruction does not match completion marker"
            )
        identity_normalization = "runner_v1_candidate_relationship_integer_keys"
    manifest["_analysis_input_identity_normalization"] = identity_normalization
    return status, manifest, records_path, _sha256(status_path)


def _read_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_FIELDS - fields)
        if missing:
            raise AnalysisError(f"per-child metrics lack required fields: {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise AnalysisError("per-child metrics are empty")
    return rows


def _none(value: object) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "nan"}


def _as_int(row: Mapping[str, str], field: str) -> int | None:
    value = row[field]
    if _none(value):
        return None
    return int(float(value))


def _as_float(row: Mapping[str, str], field: str) -> float:
    value = row[field]
    if _none(value):
        return math.nan
    return float(value)


def _as_bool(row: Mapping[str, str], field: str) -> bool:
    value = row[field].strip().lower()
    if value in {"true", "1", "1.0"}:
        return True
    if value in {"false", "0", "0.0", "", "none", "nan"}:
        return False
    raise AnalysisError(f"invalid Boolean {field}={row[field]!r}")


def _parent_set(row: Mapping[str, str], prefix: str) -> frozenset[str]:
    return frozenset(
        value
        for value in (row[f"{prefix}_parent1"], row[f"{prefix}_parent2"])
        if not _none(value)
    )


def _truth_state(row: Mapping[str, str], basis: str) -> int:
    field = "truth_state" if basis == "biological" else "observable_truth_state"
    value = _as_int(row, field)
    if value not in (0, 1, 2):
        raise AnalysisError(f"invalid {field}={value!r}")
    return value


def _truth_set(row: Mapping[str, str], basis: str) -> frozenset[str]:
    field = "truth_parents" if basis == "biological" else "observable_truth_parents"
    return frozenset(value for value in row[field].split(";") if value)


def _raw_state(row: Mapping[str, str]) -> int | None:
    return _as_int(row, "diagnostic_LocalObservedParentCount")


def _tier_label(tier: str) -> str:
    return "TierA" if tier == "tier_a" else "TierB"


def _state_support(row: Mapping[str, str]) -> np.ndarray:
    support = np.asarray(
        [_as_float(row, f"diagnostic_StateSupport{state}") for state in range(3)],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(support)) or np.any(support < -1e-10):
        raise AnalysisError("StateSupport contains non-finite or negative values")
    total = float(np.sum(support))
    if not math.isfinite(total) or total <= 0.0 or abs(total - 1.0) > 1e-5:
        raise AnalysisError(f"StateSupport does not sum to one: {support.tolist()}")
    return np.clip(support / total, 0.0, 1.0)


def _seed_for(*identity: object) -> int:
    digest = hashlib.blake2b(
        "|".join(map(str, (BOOTSTRAP_SEED, *identity))).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little")


def _quantiles(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    alpha = (1.0 - CI_LEVEL) / 2.0
    lower, upper = np.quantile(finite, (alpha, 1.0 - alpha))
    return float(lower), float(upper)


def _cluster_ratio(
    rows: Sequence[Mapping[str, str]],
    contribution: Callable[[Mapping[str, str]], tuple[float, float]],
    identity: Sequence[object],
) -> dict[str, object]:
    clusters = sorted({_as_int(row, "replicate") for row in rows})
    if any(cluster is None for cluster in clusters):
        raise AnalysisError("replicate identifiers may not be missing")
    index = {cluster: position for position, cluster in enumerate(clusters)}
    numerators = np.zeros(len(clusters), dtype=np.float64)
    denominators = np.zeros(len(clusters), dtype=np.float64)
    for row in rows:
        numerator, denominator = contribution(row)
        position = index[_as_int(row, "replicate")]
        numerators[position] += numerator
        denominators[position] += denominator
    numerator = float(np.sum(numerators))
    denominator = float(np.sum(denominators))
    estimate = None if denominator <= 0.0 else numerator / denominator
    lower = upper = None
    if denominator > 0.0 and clusters:
        rng = np.random.default_rng(_seed_for(*identity))
        weights = rng.multinomial(
            len(clusters), np.full(len(clusters), 1.0 / len(clusters)),
            size=BOOTSTRAP_REPLICATES,
        )
        boot_num = weights @ numerators
        boot_den = weights @ denominators
        estimates = np.divide(
            boot_num,
            boot_den,
            out=np.full(BOOTSTRAP_REPLICATES, np.nan),
            where=boot_den > 0.0,
        )
        lower, upper = _quantiles(estimates)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "estimate": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "replicate_clusters": len(clusters),
        "ci_method": "replicate_cluster_percentile_bootstrap",
    }


def _balanced_accuracy(
    rows: Sequence[Mapping[str, str]], basis: str, identity: Sequence[object]
) -> dict[str, object]:
    clusters = sorted({_as_int(row, "replicate") for row in rows})
    index = {cluster: position for position, cluster in enumerate(clusters)}
    correct = np.zeros((len(clusters), 3), dtype=np.float64)
    total = np.zeros((len(clusters), 3), dtype=np.float64)
    for row in rows:
        cluster = index[_as_int(row, "replicate")]
        state = _truth_state(row, basis)
        total[cluster, state] += 1.0
        correct[cluster, state] += float(_raw_state(row) == state)

    def summarize(c: np.ndarray, n: np.ndarray) -> float:
        state_total = np.sum(n, axis=0)
        state_correct = np.sum(c, axis=0)
        present = state_total > 0.0
        if not np.all(present):
            return math.nan
        return float(np.mean(state_correct[present] / state_total[present]))

    estimate = summarize(correct, total)
    rng = np.random.default_rng(_seed_for(*identity))
    weights = rng.multinomial(
        len(clusters), np.full(len(clusters), 1.0 / len(clusters)),
        size=BOOTSTRAP_REPLICATES,
    )
    boot_correct = np.einsum("bc,cs->bs", weights, correct)
    boot_total = np.einsum("bc,cs->bs", weights, total)
    boot = np.mean(
        np.divide(
            boot_correct,
            boot_total,
            out=np.full_like(boot_correct, np.nan),
            where=boot_total > 0.0,
        ),
        axis=1,
    )
    lower, upper = _quantiles(boot)
    return {
        "numerator": None,
        "denominator": len(rows),
        "estimate": None if not math.isfinite(estimate) else estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "replicate_clusters": len(clusters),
        "ci_method": "replicate_cluster_percentile_bootstrap",
    }


def _metric_record(
    scenario: str,
    effective: float,
    basis: str,
    scope_type: str,
    scope: str,
    metric: str,
    value: Mapping[str, object],
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "effective_markers": effective,
        "truth_basis": basis,
        "scope_type": scope_type,
        "scope": scope,
        "metric": metric,
        **value,
    }


def _false_parent_release(row: Mapping[str, str], tier: str, basis: str) -> bool:
    if _truth_state(row, basis) != 0:
        return False
    label = _tier_label(tier)
    state_release = _as_bool(row, f"diagnostic_{label}StateCall")
    selected_state = _as_int(row, f"{tier}_observed_parent_count")
    partial_release = bool(_parent_set(row, f"{tier}_partial"))
    return bool((state_release and selected_state not in (None, 0)) or partial_release)


def _related_decoy_counts(
    row: Mapping[str, str], prefix: str, basis: str
) -> tuple[float, float]:
    truth = _truth_set(row, basis)
    predicted = _parent_set(row, prefix)
    substitutions = 0
    at_risk = 0
    for anchor, decoys in RELATED_DECOYS.items():
        if anchor not in truth:
            continue
        at_risk += 1
        substitutions += int(
            anchor not in predicted and any(decoy in predicted for decoy in decoys)
        )
    return float(substitutions), float(at_risk)


def _core_metrics(
    records: Sequence[Mapping[str, str]],
    scenarios: Sequence[str],
    effective_values: Sequence[float],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    strata = sorted({row["stratum"] for row in records})
    for scenario in scenarios:
        for effective in effective_values:
            cell = [
                row
                for row in records
                if row["scenario"] == scenario
                and _as_float(row, "effective_markers") == effective
            ]
            if not cell:
                raise AnalysisError(f"empty scenario/marker cell: {scenario}, {effective}")
            for basis in ("biological", "observable"):
                scopes = [("all", "ALL", cell)]
                scopes.extend(
                    (
                        "truth_state",
                        f"M{state}",
                        [row for row in cell if _truth_state(row, basis) == state],
                    )
                    for state in range(3)
                )
                scopes.extend(
                    (
                        "stratum",
                        stratum,
                        [row for row in cell if row["stratum"] == stratum],
                    )
                    for stratum in strata
                )
                for scope_type, scope, subset in scopes:
                    if not subset:
                        continue
                    base_identity = (
                        "metric", scenario, effective, basis, scope_type, scope
                    )

                    def add(metric: str, contribution) -> None:
                        value = _cluster_ratio(
                            subset, contribution, (*base_identity, metric)
                        )
                        output.append(
                            _metric_record(
                                scenario, effective, basis, scope_type, scope,
                                metric, value,
                            )
                        )

                    add(
                        "raw_state_accuracy",
                        lambda row, b=basis: (
                            float(_raw_state(row) == _truth_state(row, b)), 1.0
                        ),
                    )
                    if scope_type == "all":
                        output.append(
                            _metric_record(
                                scenario,
                                effective,
                                basis,
                                scope_type,
                                scope,
                                "raw_state_balanced_accuracy",
                                _balanced_accuracy(
                                    subset, basis, (*base_identity, "balanced")
                                ),
                            )
                        )
                    add(
                        "complete_state_accuracy",
                        lambda row, b=basis: (
                            float(
                                _as_int(row, "complete_observed_parent_count")
                                == _truth_state(row, b)
                            ),
                            1.0,
                        ),
                    )
                    add(
                        "complete_joint_accuracy",
                        lambda row, b=basis: (
                            float(
                                _as_int(row, "complete_observed_parent_count")
                                == _truth_state(row, b)
                                and _parent_set(row, "complete") == _truth_set(row, b)
                            ),
                            1.0,
                        ),
                    )
                    for tier in ("tier_a", "tier_b"):
                        label = _tier_label(tier)
                        state_flag = f"diagnostic_{label}StateCall"
                        exact_flag = f"diagnostic_{label}ExactConfiguration"
                        add(
                            f"{tier}_state_coverage",
                            lambda row, flag=state_flag: (
                                float(_as_bool(row, flag)), 1.0
                            ),
                        )
                        add(
                            f"{tier}_state_selective_accuracy",
                            lambda row, b=basis, flag=state_flag, t=tier: (
                                float(
                                    _as_bool(row, flag)
                                    and _as_int(row, f"{t}_observed_parent_count")
                                    == _truth_state(row, b)
                                ),
                                float(_as_bool(row, flag)),
                            ),
                        )
                        add(
                            f"{tier}_exact_coverage",
                            lambda row, flag=exact_flag: (
                                float(_as_bool(row, flag)), 1.0
                            ),
                        )
                        add(
                            f"{tier}_exact_selective_accuracy",
                            lambda row, b=basis, flag=exact_flag, t=tier: (
                                float(
                                    _as_bool(row, flag)
                                    and _as_int(row, f"{t}_observed_parent_count")
                                    == _truth_state(row, b)
                                    and _parent_set(row, t) == _truth_set(row, b)
                                ),
                                float(_as_bool(row, flag)),
                            ),
                        )
                        partial = f"{tier}_partial"
                        add(
                            f"{tier}_partial_edge_precision",
                            lambda row, b=basis, p=partial: (
                                float(len(_parent_set(row, p) & _truth_set(row, b))),
                                float(len(_parent_set(row, p))),
                            ),
                        )
                        add(
                            f"{tier}_partial_edge_recall",
                            lambda row, b=basis, p=partial: (
                                float(len(_parent_set(row, p) & _truth_set(row, b))),
                                float(len(_truth_set(row, b))),
                            ),
                        )
                        add(
                            f"{tier}_m0_false_parent_release",
                            lambda row, b=basis, t=tier: (
                                float(_false_parent_release(row, t, b)),
                                float(_truth_state(row, b) == 0),
                            ),
                        )
                        add(
                            f"{tier}_m1_to_m2_error",
                            lambda row, b=basis, t=tier, flag=state_flag: (
                                float(
                                    _truth_state(row, b) == 1
                                    and _as_bool(row, flag)
                                    and _as_int(row, f"{t}_observed_parent_count") == 2
                                ),
                                float(_truth_state(row, b) == 1),
                            ),
                        )
                        add(
                            f"{tier}_m1_to_m2_error_among_state_releases",
                            lambda row, b=basis, t=tier, flag=state_flag: (
                                float(
                                    _truth_state(row, b) == 1
                                    and _as_bool(row, flag)
                                    and _as_int(row, f"{t}_observed_parent_count") == 2
                                ),
                                float(
                                    _truth_state(row, b) == 1
                                    and _as_bool(row, flag)
                                ),
                            ),
                        )
                        add(
                            f"{tier}_related_decoy_substitution_exact",
                            lambda row, b=basis, t=tier: _related_decoy_counts(
                                row, t, b
                            ),
                        )
                        add(
                            f"{tier}_related_decoy_substitution_partial",
                            lambda row, b=basis, p=partial: _related_decoy_counts(
                                row, p, b
                            ),
                        )
    return output


def _confusion_rows(
    records: Sequence[Mapping[str, str]],
    scenarios: Sequence[str],
    effective_values: Sequence[float],
) -> list[dict[str, object]]:
    output = []
    for scenario in scenarios:
        for effective in effective_values:
            cell = [
                row for row in records
                if row["scenario"] == scenario
                and _as_float(row, "effective_markers") == effective
            ]
            for basis in ("biological", "observable"):
                for true_state in range(3):
                    subset = [
                        row for row in cell if _truth_state(row, basis) == true_state
                    ]
                    for selected in (0, 1, 2, None):
                        value = _cluster_ratio(
                            subset,
                            lambda row, selected=selected: (
                                float(_raw_state(row) == selected), 1.0
                            ),
                            (
                                "confusion", scenario, effective, basis,
                                true_state, selected,
                            ),
                        )
                        output.append({
                            "scenario": scenario,
                            "effective_markers": effective,
                            "truth_basis": basis,
                            "truth_state": true_state,
                            "raw_selected_state": (
                                "unresolved" if selected is None else selected
                            ),
                            **value,
                        })
    return output


def _calibration_values(
    rows: Sequence[Mapping[str, str]], basis: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.stack([_state_support(row) for row in rows])
    targets = np.asarray([_truth_state(row, basis) for row in rows], dtype=np.int64)
    predicted = np.argmax(probabilities, axis=1)
    return probabilities, targets, predicted


def _ece(probabilities: np.ndarray, targets: np.ndarray) -> float:
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predicted == targets
    total = len(targets)
    if total == 0:
        return math.nan
    value = 0.0
    for index, (lower, upper) in enumerate(zip(ECE_EDGES[:-1], ECE_EDGES[1:])):
        if index + 1 == len(ECE_EDGES) - 1:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence >= lower) & (confidence < upper)
        count = int(np.sum(selected))
        if count:
            value += count / total * abs(
                float(np.mean(correct[selected])) - float(np.mean(confidence[selected]))
            )
    return value


def _cluster_ece(
    rows: Sequence[Mapping[str, str]], basis: str, identity: Sequence[object]
) -> dict[str, object]:
    probabilities, targets, _ = _calibration_values(rows, basis)
    clusters = sorted({_as_int(row, "replicate") for row in rows})
    cluster_rows = {
        cluster: np.asarray(
            [index for index, row in enumerate(rows) if _as_int(row, "replicate") == cluster],
            dtype=np.int64,
        )
        for cluster in clusters
    }
    estimate = _ece(probabilities, targets)
    rng = np.random.default_rng(_seed_for(*identity))
    boot = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for iteration in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([cluster_rows[cluster] for cluster in sampled])
        boot[iteration] = _ece(probabilities[indices], targets[indices])
    lower, upper = _quantiles(boot)
    return {
        "numerator": None,
        "denominator": len(rows),
        "estimate": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "replicate_clusters": len(clusters),
        "ci_method": "replicate_cluster_percentile_bootstrap",
    }


def _state_support_metrics(
    records: Sequence[Mapping[str, str]],
    scenarios: Sequence[str],
    effective_values: Sequence[float],
) -> list[dict[str, object]]:
    output = []
    for scenario in scenarios:
        for effective in effective_values:
            cell = [
                row for row in records
                if row["scenario"] == scenario
                and _as_float(row, "effective_markers") == effective
            ]
            for basis in ("biological", "observable"):
                base = ("calibration", scenario, effective, basis)
                brier = _cluster_ratio(
                    cell,
                    lambda row, b=basis: (
                        float(
                            np.sum(
                                (
                                    _state_support(row)
                                    - np.eye(3)[_truth_state(row, b)]
                                ) ** 2
                            )
                        ),
                        1.0,
                    ),
                    (*base, "brier"),
                )
                log_loss = _cluster_ratio(
                    cell,
                    lambda row, b=basis: (
                        -math.log(
                            max(_state_support(row)[_truth_state(row, b)], 1e-300)
                        ),
                        1.0,
                    ),
                    (*base, "log_loss"),
                )
                ece = _cluster_ece(cell, basis, (*base, "ece"))
                for metric, value in (
                    ("multiclass_brier", brier),
                    ("log_loss", log_loss),
                    ("top_label_ece_10_equal_width_bins", ece),
                ):
                    output.append({
                        "scenario": scenario,
                        "effective_markers": effective,
                        "truth_basis": basis,
                        "metric": metric,
                        **value,
                    })
    return output


def _finite_min(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else math.nan


def _risk_scores(
    row: Mapping[str, str], endpoint: str
) -> tuple[bool, float, float, float]:
    state = _raw_state(row)
    if state not in (0, 1, 2):
        return False, math.nan, math.nan, math.nan
    support = float(_state_support(row)[state])
    informative = (_as_int(row, "diagnostic_InformativeContigCount") or 0) >= 3
    state_bootstrap = _as_float(row, "diagnostic_LocalStateBootstrapFraction")
    state_loco = _as_float(row, "diagnostic_LocalStateLOCOFraction")
    if endpoint == "state":
        return informative, state_bootstrap, state_loco, support

    complete_state = _as_int(row, "complete_observed_parent_count")
    eligible = bool(
        informative
        and complete_state == state
        and not _as_bool(row, "diagnostic_GraphConflict")
        and not _as_bool(row, "diagnostic_GraphTieConflict")
    )
    bootstrap_values = [
        state_bootstrap,
        _as_float(row, "diagnostic_LocalConfigurationBootstrapFraction"),
        _as_float(row, "diagnostic_GraphConfigurationBootstrapFraction"),
    ]
    loco_values = [
        state_loco,
        _as_float(row, "diagnostic_LocalConfigurationLOCOFraction"),
        _as_float(row, "diagnostic_GraphConfigurationLOCOFraction"),
    ]
    if state >= 1:
        bootstrap_values.append(
            _as_float(row, "diagnostic_Parent1BootstrapFraction")
        )
        loco_values.append(_as_float(row, "diagnostic_Parent1LOCOFraction"))
    if state >= 2:
        bootstrap_values.append(
            _as_float(row, "diagnostic_Parent2BootstrapFraction")
        )
        loco_values.append(_as_float(row, "diagnostic_Parent2LOCOFraction"))
    return (
        eligible,
        _finite_min(bootstrap_values),
        _finite_min(loco_values),
        support,
    )


def _risk_grid() -> list[tuple[str, float, float, float]]:
    values = [("baseline", 0.0, 0.0, 0.0)]
    values.extend(("bootstrap", threshold, 0.0, 0.0) for threshold in BOOTSTRAP_THRESHOLDS)
    values.extend(("loco", 0.0, threshold, 0.0) for threshold in LOCO_THRESHOLDS)
    values.extend(("min_support", 0.0, 0.0, threshold) for threshold in SUPPORT_THRESHOLDS)
    values.extend(("joint", *thresholds) for thresholds in JOINT_THRESHOLDS)
    deduplicated = []
    seen = set()
    for value in values:
        if value not in seen:
            deduplicated.append(value)
            seen.add(value)
    return deduplicated


def _risk_coverage_rows(
    records: Sequence[Mapping[str, str]],
    scenarios: Sequence[str],
    effective_values: Sequence[float],
) -> list[dict[str, object]]:
    output = []
    grid = _risk_grid()
    for scenario in scenarios:
        for effective in effective_values:
            cell = [
                row for row in records
                if row["scenario"] == scenario
                and _as_float(row, "effective_markers") == effective
            ]
            clusters = sorted({_as_int(row, "replicate") for row in cell})
            cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
            row_counts = np.zeros(len(clusters), dtype=np.float64)
            for row in cell:
                row_counts[cluster_index[_as_int(row, "replicate")]] += 1.0
            for basis in ("biological", "observable"):
                for endpoint in ("state", "exact"):
                    pass_by_cluster = np.zeros((len(clusters), len(grid)), dtype=np.float64)
                    correct_by_cluster = np.zeros_like(pass_by_cluster)
                    for row in cell:
                        eligible, bootstrap, loco, support = _risk_scores(row, endpoint)
                        if endpoint == "state":
                            correct = _raw_state(row) == _truth_state(row, basis)
                        else:
                            correct = bool(
                                _as_int(row, "complete_observed_parent_count")
                                == _truth_state(row, basis)
                                and _parent_set(row, "complete") == _truth_set(row, basis)
                            )
                        cluster = cluster_index[_as_int(row, "replicate")]
                        for gate_index, (_, b_cut, l_cut, s_cut) in enumerate(grid):
                            released = bool(
                                eligible
                                and math.isfinite(bootstrap)
                                and math.isfinite(loco)
                                and bootstrap >= b_cut
                                and loco >= l_cut
                                and support >= s_cut
                            )
                            pass_by_cluster[cluster, gate_index] += float(released)
                            correct_by_cluster[cluster, gate_index] += float(
                                released and correct
                            )
                    rng = np.random.default_rng(
                        _seed_for("risk", scenario, effective, basis, endpoint)
                    )
                    weights = rng.multinomial(
                        len(clusters),
                        np.full(len(clusters), 1.0 / len(clusters)),
                        size=BOOTSTRAP_REPLICATES,
                    )
                    boot_released = weights @ pass_by_cluster
                    boot_correct = weights @ correct_by_cluster
                    boot_total = weights @ row_counts
                    boot_coverage = np.divide(
                        boot_released,
                        boot_total[:, None],
                        out=np.full_like(boot_released, np.nan),
                        where=boot_total[:, None] > 0.0,
                    )
                    boot_accuracy = np.divide(
                        boot_correct,
                        boot_released,
                        out=np.full_like(boot_correct, np.nan),
                        where=boot_released > 0.0,
                    )
                    released = np.sum(pass_by_cluster, axis=0)
                    correct = np.sum(correct_by_cluster, axis=0)
                    total = float(np.sum(row_counts))
                    for gate_index, (family, b_cut, l_cut, s_cut) in enumerate(grid):
                        coverage_lower, coverage_upper = _quantiles(
                            boot_coverage[:, gate_index]
                        )
                        accuracy_lower, accuracy_upper = _quantiles(
                            boot_accuracy[:, gate_index]
                        )
                        accuracy = (
                            None
                            if released[gate_index] == 0.0
                            else float(correct[gate_index] / released[gate_index])
                        )
                        output.append({
                            "scenario": scenario,
                            "effective_markers": effective,
                            "truth_basis": basis,
                            "endpoint": endpoint,
                            "curve_family": family,
                            "minimum_bootstrap": b_cut,
                            "minimum_loco": l_cut,
                            "minimum_state_support": s_cut,
                            "released": float(released[gate_index]),
                            "total": total,
                            "coverage": float(released[gate_index] / total),
                            "coverage_ci_lower": coverage_lower,
                            "coverage_ci_upper": coverage_upper,
                            "selective_accuracy": accuracy,
                            "selective_accuracy_ci_lower": accuracy_lower,
                            "selective_accuracy_ci_upper": accuracy_upper,
                            "selective_risk": (
                                None if accuracy is None else 1.0 - accuracy
                            ),
                            "replicate_clusters": len(clusters),
                            "ci_method": "replicate_cluster_percentile_bootstrap",
                        })
    return output


def _validate_design(
    records: Sequence[Mapping[str, str]], manifest: Mapping[str, object]
) -> tuple[list[str], list[float]]:
    manifest_parameters = manifest.get("parameters", {})
    scenarios = list(manifest_parameters.get("scenarios", []))
    effective_values = sorted(
        float(value) for value in manifest_parameters.get("effective_markers", [])
    )
    if not scenarios or not effective_values:
        raise AnalysisError("manifest has no scenarios or effective-marker values")
    record_scenarios = {row["scenario"] for row in records}
    record_effective = {_as_float(row, "effective_markers") for row in records}
    if record_scenarios != set(scenarios):
        raise AnalysisError("record scenarios do not match manifest")
    if record_effective != set(effective_values):
        raise AnalysisError("record effective-marker values do not match manifest")
    keys = set()
    for row in records:
        key = (
            _as_int(row, "replicate"), row["scenario"],
            _as_float(row, "effective_markers"), row["sample"],
        )
        if key in keys:
            raise AnalysisError(f"duplicate per-child record: {key}")
        keys.add(key)
        if len(_truth_set(row, "biological")) != _truth_state(row, "biological"):
            raise AnalysisError("biological truth state/parent-set mismatch")
        if len(_truth_set(row, "observable")) != _truth_state(row, "observable"):
            raise AnalysisError("observable truth state/parent-set mismatch")
        _state_support(row)
    return scenarios, effective_values


def _selection_stat(
    rows: Sequence[Mapping[str, str]],
    contribution,
    identity: Sequence[object],
) -> dict[str, object]:
    return _cluster_ratio(rows, contribution, ("selection", *identity))


def _select_effective_markers(
    records: Sequence[Mapping[str, str]],
    scenarios: Sequence[str],
    effective_values: Sequence[float],
) -> dict[str, object]:
    policy = ANALYSIS_PROTOCOL["selection"]
    evaluations = []
    selected = None
    for effective in sorted(effective_values):
        checks = []
        for scenario in scenarios:
            cell = [
                row for row in records
                if row["scenario"] == scenario
                and _as_float(row, "effective_markers") == effective
            ]
            for state in range(3):
                subset = [row for row in cell if _truth_state(row, "biological") == state]
                value = _selection_stat(
                    subset,
                    lambda row, state=state: (
                        float(_raw_state(row) == state), 1.0
                    ),
                    (effective, scenario, f"M{state}", "raw_state_accuracy"),
                )
                minimum_clusters = policy[
                    "minimum_replicate_clusters_per_scenario_state"
                ]
                accuracy = policy["biological_raw_state_accuracy"]
                passed = bool(
                    value["replicate_clusters"] >= minimum_clusters
                    and value["estimate"] is not None
                    and value["estimate"] >= accuracy["minimum_point_estimate"]
                    and value["ci_lower"] is not None
                    and value["ci_lower"] >= accuracy["minimum_ci_lower"]
                )
                checks.append({
                    "scenario": scenario,
                    "scope": f"M{state}",
                    "constraint": "biological_raw_state_accuracy",
                    "passed": passed,
                    **value,
                })

            false_specs = (
                (
                    "tier_a_m0_false_parent_release",
                    lambda row: (
                        float(_false_parent_release(row, "tier_a", "biological")),
                        float(_truth_state(row, "biological") == 0),
                    ),
                ),
                (
                    "tier_a_m1_to_m2_error",
                    lambda row: (
                        float(
                            _truth_state(row, "biological") == 1
                            and _as_bool(row, "diagnostic_TierAStateCall")
                            and _as_int(row, "tier_a_observed_parent_count") == 2
                        ),
                        float(_truth_state(row, "biological") == 1),
                    ),
                ),
                (
                    "tier_a_related_decoy_substitution",
                    lambda row: _related_decoy_counts(
                        row, "tier_a_partial", "biological"
                    ),
                ),
            )
            for constraint, contribution in false_specs:
                value = _selection_stat(
                    cell, contribution, (effective, scenario, constraint)
                )
                limits = policy[constraint]
                passed = bool(
                    value["estimate"] is not None
                    and value["estimate"] <= limits["maximum_point_estimate"]
                    and value["ci_upper"] is not None
                    and value["ci_upper"] <= limits["maximum_ci_upper"]
                )
                checks.append({
                    "scenario": scenario,
                    "scope": "applicable biological truth subset",
                    "constraint": constraint,
                    "passed": passed,
                    **value,
                })
        passed = all(check["passed"] for check in checks)
        evaluations.append({
            "effective_markers": effective,
            "passed": passed,
            "checks": checks,
        })
        if selected is None and passed:
            selected = effective
    return {
        "selection_passed": selected is not None,
        "selected_effective_markers": selected,
        "evaluations": evaluations,
    }


def _load_frozen_selection(path: Path) -> dict[str, object]:
    value = _load_json(path)
    if not isinstance(value, dict):
        raise AnalysisError("selection JSON is not an object")
    if value.get("protocol_sha256") != _canonical_hash(ANALYSIS_PROTOCOL):
        raise AnalysisError("selection JSON uses a different analysis protocol")
    if value.get("mode") != "calibration" or value.get("selection_passed") is not True:
        raise AnalysisError("selection JSON is not a successful calibration selection")
    selected = value.get("selected_effective_markers")
    if selected is None or not math.isfinite(float(selected)):
        raise AnalysisError("selection JSON has no finite selected marker value")
    return value


def run(args: argparse.Namespace) -> bool:
    if args.mode == "calibration" and args.selection_json is not None:
        raise AnalysisError("--selection-json is forbidden in calibration mode")
    if args.mode == "holdout" and args.selection_json is None:
        raise AnalysisError("--selection-json is required in holdout mode")
    if args.mode == "holdout-comparison" and args.selection_json is not None:
        raise AnalysisError(
            "--selection-json is forbidden in descriptive holdout comparison mode"
        )

    paths = _output_paths(args.output_prefix, args.mode)
    paths["protocol"].parent.mkdir(parents=True, exist_ok=True)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise AnalysisError(f"refusing to overwrite analysis outputs: {existing}")

    input_mode = "holdout" if args.mode.startswith("holdout") else args.mode
    status, manifest, records_path, status_sha256 = _validated_completed_input(
        args.input_root, input_mode
    )
    protocol_hash = _canonical_hash(ANALYSIS_PROTOCOL)
    _atomic_json(
        paths["protocol"],
        {"protocol_sha256": protocol_hash, "protocol": ANALYSIS_PROTOCOL},
    )

    frozen_selection = None
    if args.mode == "holdout":
        frozen_selection = _load_frozen_selection(args.selection_json.resolve())

    records = _read_records(records_path)
    scenarios, effective_values = _validate_design(records, manifest)
    if frozen_selection is not None:
        selected = float(frozen_selection["selected_effective_markers"])
        if selected not in effective_values:
            raise AnalysisError(
                f"holdout lacks frozen effective-marker value {selected:g}"
            )
        # Do not evaluate or compare the unselected holdout marker settings.
        records = [
            row for row in records
            if _as_float(row, "effective_markers") == selected
        ]
        effective_values = [selected]

    metrics = _core_metrics(records, scenarios, effective_values)
    confusion = _confusion_rows(records, scenarios, effective_values)
    calibration = _state_support_metrics(records, scenarios, effective_values)
    risk_coverage = _risk_coverage_rows(records, scenarios, effective_values)
    _atomic_csv(paths["metrics"], metrics)
    _atomic_csv(paths["confusion"], confusion)
    _atomic_csv(paths["calibration"], calibration)
    _atomic_csv(paths["risk_coverage"], risk_coverage)

    selection = None
    success = True
    if args.mode == "calibration":
        selection = _select_effective_markers(
            records, scenarios, effective_values
        )
        frozen = {
            "schema_version": SCHEMA_VERSION,
            "mode": "calibration",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": protocol_hash,
            "input_root": str(args.input_root.resolve()),
            "input_completion_status_sha256": status_sha256,
            "input_manifest_sha256": status.get("manifest_sha256"),
            "release_policy": ANALYSIS_PROTOCOL["selection"]["release_policy"],
            "release_thresholds_tuned": False,
            **selection,
        }
        _atomic_json(paths["selection"], frozen)
        success = bool(selection["selection_passed"])

    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_hash,
        "input_root": str(args.input_root.resolve()),
        "input_completion_status_sha256": status_sha256,
        "input_manifest_sha256": status.get("manifest_sha256"),
        "scenarios": scenarios,
        "evaluated_effective_markers": effective_values,
        "child_records_evaluated": len(records),
        "truth_interpretation": {
            "selection": "biological",
            "parallel_diagnostic": "observable",
        },
        "input_manifest_identity_normalization": manifest.get(
            "_analysis_input_identity_normalization"
        ),
        "selection": selection,
        "frozen_selection_source": (
            None if args.selection_json is None else str(args.selection_json.resolve())
        ),
        "frozen_selection_sha256": (
            None if args.selection_json is None else _sha256(args.selection_json.resolve())
        ),
        "holdout_tuning_performed": False,
        "output_files": {
            name: str(path) for name, path in paths.items() if name != "summary"
        },
    }
    _atomic_json(paths["summary"], summary)
    return success


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        success = run(args)
    except (AnalysisError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 2
    if not success:
        print(
            "calibration selection failed: no effective-marker value satisfied "
            "all frozen constraints",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
