"""Predeclared descriptive analysis of paired Smart candidate-panel runs.

The input must be a completed ``pedigree_smart_panel_simulation.py`` output
root.  Every child is paired across nested eligible-candidate panels K=8, 16,
and 32 while split, replicate, scenario, effective-marker value, sample,
state-profile, and candidate-relatedness profile remain fixed.

This analysis never selects or tunes an effective-marker value.  Frozen gates
are reported descriptively, including coverage-floor diagnostics that are not
part of the safety gate.  An unresolved raw state is incorrect for every truth
state, including M0.  Tier state/exact correctness is evaluated only when the
corresponding explicit Tier release flag is true; otherwise the outcome is an
abstention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
PROTOCOL_ID = "pedigree_smart_panel_analysis_v2_primary_aligned_20260829"
BOOTSTRAP_SEED = 20260829
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
MINIMUM_BOOTSTRAP_REPLICATES = 2_000
CI_LEVEL = 0.95
EXPECTED_PANEL_SIZES = (8, 16, 32)
PANEL_TRANSITIONS = ((8, 16), (16, 32), (8, 32))
RELATEDNESS_PROFILES = ("unrelated", "full_half", "full_half_near")
CONFIDENCE_STATES = ("correct_confident", "wrong_confident", "abstain")
TIERS = ("tier_a", "tier_b")
ENDPOINTS = ("state", "exact")

RAW_STATE_MINIMUM = 0.90
RAW_STATE_CI_LOWER_MINIMUM = 0.80
SAFETY_EVENT_MAXIMUM = 0.02
SAFETY_EVENT_CI_UPPER_MAXIMUM = 0.05
PANEL_REGRESSION_MAXIMUM = 0.01
PANEL_REGRESSION_CI_UPPER_MAXIMUM = 0.02
TIER_A_STATE_COVERAGE_FLOOR = 0.05
TIER_A_EXACT_COVERAGE_FLOOR = 0.01

BASE_REQUIRED_FIELDS = {
    "split",
    "replicate",
    "scenario",
    "effective_markers",
    "sample",
    "candidate_relatedness",
    "eligible_candidate_count",
    "stratum",
    "truth_state",
    "truth_parents",
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
    "diagnostic_TierAStateCall",
    "diagnostic_TierBStateCall",
    "diagnostic_TierAExactConfiguration",
    "diagnostic_TierBExactConfiguration",
}
PROFILE_FIELDS = ("state_profile", "profile")
OBSERVABLE_FIELDS = {"observable_truth_state", "observable_truth_parents"}
PAIRED_HASH_FIELDS = {
    "truth_sha256",
    "child_genetics_sha256",
    "child_observations_sha256",
    "candidate_universe_sha256",
}


class AnalysisError(RuntimeError):
    """Input or analysis failure with a concise CLI message."""


@dataclass(frozen=True, slots=True)
class Record:
    split: str
    replicate: int
    scenario: str
    effective_markers: float
    sample: str
    state_profile: str
    candidate_relatedness: str
    eligible_candidate_count: int
    stratum: str
    truth_state: int
    truth_parents: frozenset[str]
    observable_truth_state: int
    observable_truth_parents: frozenset[str]
    raw_state: int | None
    complete_state: int | None
    complete_parents: frozenset[str]
    tier_a_state: int | None
    tier_a_parents: frozenset[str]
    tier_b_state: int | None
    tier_b_parents: frozenset[str]
    tier_a_partial_parents: frozenset[str]
    tier_b_partial_parents: frozenset[str]
    tier_a_state_call: bool
    tier_b_state_call: bool
    tier_a_exact_call: bool
    tier_b_exact_call: bool
    truth_sha256: str | None = None
    child_genetics_sha256: str | None = None
    child_observations_sha256: str | None = None
    candidate_universe_sha256: str | None = None


def _analysis_protocol(bootstrap_replicates: int) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "frozen_before_panel_outcome_review": False,
        "protocol_amendment": {
            "reason": (
                "align safety gates with Smart primary_view=tier_b and report "
                "selective exact-call error explicitly"
            ),
            "timing": (
                "added after balanced-panel outcome review and before the "
                "predeclared cohort-prevalence follow-up outcomes"
            ),
        },
        "selection_or_tuning": False,
        "paired_design": {
            "panel_sizes": list(EXPECTED_PANEL_SIZES),
            "transitions": [list(values) for values in PANEL_TRANSITIONS],
            "pairing_key": [
                "split",
                "replicate",
                "scenario",
                "effective_markers",
                "sample",
                "state_profile",
                "candidate_relatedness",
            ],
            "transition_states": list(CONFIDENCE_STATES),
            "conditional_transition_denominator": (
                "children in the smaller panel's origin confidence state"
            ),
            "all_pairs_rate_also_reported": True,
        },
        "truth": {
            "primary": "biological candidate-panel parent truth",
            "observable": (
                "reported only when the completed runner provides observable truth"
            ),
            "unresolved_m0": (
                "unresolved is never correct for M0; a Tier M0 result is correct "
                "only when the explicit Tier state/exact release flag is true"
            ),
        },
        "uncertainty": {
            "cluster": "replicate",
            "method": "replicate-cluster percentile bootstrap",
            "replicates": bootstrap_replicates,
            "minimum_replicates": MINIMUM_BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": CI_LEVEL,
            "resampling": "shared bootstrap weights within each analysis cell",
        },
        "descriptive_gates": {
            "raw_state_accuracy": {
                "applies_to": "each scenario/profile/relatedness/K and M0/M1/M2",
                "minimum_point_estimate": RAW_STATE_MINIMUM,
                "minimum_ci_lower": RAW_STATE_CI_LOWER_MINIMUM,
            },
            "tier_a_and_primary_tier_b_safety_events": {
                "metrics": [
                    "tier_a_m0_false_parent_release",
                    "tier_a_m1_to_m2_error",
                    "tier_a_related_decoy_substitution_partial",
                    "tier_a_exact_error_release",
                    "tier_a_exact_selective_error",
                    "tier_b_m0_false_parent_release",
                    "tier_b_m1_to_m2_error",
                    "tier_b_related_decoy_substitution_partial",
                    "tier_b_exact_error_release",
                    "tier_b_exact_selective_error",
                ],
                "maximum_point_estimate": SAFETY_EVENT_MAXIMUM,
                "maximum_ci_upper": SAFETY_EVENT_CI_UPPER_MAXIMUM,
            },
            "correct_confident_to_wrong_confident": {
                "applies_to": "every panel transition, Tier A/B, and state/exact endpoint",
                "maximum_conditional_point_estimate": PANEL_REGRESSION_MAXIMUM,
                "maximum_conditional_ci_upper": PANEL_REGRESSION_CI_UPPER_MAXIMUM,
            },
            "coverage_floor_diagnostics": {
                "used_for_safety_gate_or_selection": False,
                "tier_a_state_minimum_point_estimate": TIER_A_STATE_COVERAGE_FLOOR,
                "tier_a_exact_minimum_point_estimate": TIER_A_EXACT_COVERAGE_FLOOR,
                "tier_b_state_minimum_point_estimate": TIER_A_STATE_COVERAGE_FLOOR,
                "tier_b_exact_minimum_point_estimate": TIER_A_EXACT_COVERAGE_FLOOR,
            },
        },
        "related_decoys": {
            "source": "completed manifest candidate_relationships_by_profile",
            "unrelated": "no decoys",
            "full_half": "full- and half-sibling decoys only",
            "full_half_near": "full-, half-sibling, and near-duplicate decoys",
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
        help="replicate-cluster bootstrap draws (minimum 2000)",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run a tiny in-memory and completed-input smoke test",
    )
    args = parser.parse_args(argv)
    if args.selftest:
        if args.input_root is not None or args.output_prefix is not None:
            parser.error("--selftest cannot be combined with input/output paths")
    elif args.input_root is None or args.output_prefix is None:
        parser.error("--input-root and --output-prefix are required")
    if args.bootstrap_replicates < MINIMUM_BOOTSTRAP_REPLICATES:
        parser.error(
            f"--bootstrap-replicates must be >= {MINIMUM_BOOTSTRAP_REPLICATES}"
        )
    return args


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


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _output_paths(prefix: Path) -> dict[str, Path]:
    return {
        "protocol": Path(f"{prefix}.protocol.json"),
        "metrics": Path(f"{prefix}.metrics.csv"),
        "confusion": Path(f"{prefix}.raw_state_confusion.csv"),
        "paired_transitions": Path(f"{prefix}.paired_transitions.csv"),
        "identity_changes": Path(f"{prefix}.paired_identity_changes.csv"),
        "gates": Path(f"{prefix}.descriptive_gates.csv"),
        "summary": Path(f"{prefix}.summary.json"),
    }


def _validated_completed_input(
    root: Path,
) -> tuple[dict[str, object], dict[str, object], Path, str]:
    root = root.resolve()
    status_path = root / "status.computational_complete.json"
    if not status_path.is_file():
        raise AnalysisError(f"completion marker is absent: {status_path}")
    status = _load_json(status_path)
    if not isinstance(status, dict) or status.get("computational_complete") is not True:
        raise AnalysisError("completion marker does not assert computational completion")
    identities = status.get("outputs")
    if not isinstance(identities, dict) or not identities:
        raise AnalysisError("completion marker has no completed-output identities")
    validated: dict[str, Path] = {}
    for name, identity in identities.items():
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
        validated[str(name)] = path

    manifest_path = validated.get("manifest.json", root / "manifest.json")
    records_path = validated.get(
        "per_child_release_metrics.csv", root / "per_child_release_metrics.csv"
    )
    if not manifest_path.is_file() or not records_path.is_file():
        raise AnalysisError("completed manifest or per-child metrics are absent")
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise AnalysisError("manifest is not a JSON object")

    persisted_hash = _canonical_hash(manifest)
    normalization = None
    if status.get("manifest_sha256") != persisted_hash:
        relationships = manifest.get("fixed_design", {}).get(
            "candidate_relationships"
        )
        if not (
            isinstance(relationships, dict)
            and relationships
            and all(str(key).isdigit() for key in relationships)
        ):
            raise AnalysisError("manifest canonical identity does not match completion marker")
        reconstructed = json.loads(json.dumps(manifest))
        reconstructed["fixed_design"]["candidate_relationships"] = {
            int(key): value for key, value in relationships.items()
        }
        if status.get("manifest_sha256") != _canonical_hash(reconstructed):
            raise AnalysisError("manifest identity mismatch after integer-key reconstruction")
        normalization = "candidate_relationship_integer_keys"
    manifest["_analysis_input_identity_normalization"] = normalization
    return status, manifest, records_path, _sha256(status_path)


def _is_none(value: object) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "null", "nan"}


def _as_int(value: object, field: str, *, optional: bool = False) -> int | None:
    if _is_none(value):
        if optional:
            return None
        raise AnalysisError(f"missing integer field {field}")
    try:
        number = float(str(value))
    except ValueError as exc:
        raise AnalysisError(f"invalid integer field {field}: {value!r}") from exc
    if not math.isfinite(number) or number != math.floor(number):
        raise AnalysisError(f"invalid integer field {field}: {value!r}")
    return int(number)


def _as_float(value: object, field: str) -> float:
    try:
        number = float(str(value))
    except ValueError as exc:
        raise AnalysisError(f"invalid numeric field {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise AnalysisError(f"non-finite numeric field {field}: {value!r}")
    return number


def _as_bool(value: object, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "1.0"}:
        return True
    if normalized in {"false", "0", "0.0"}:
        return False
    raise AnalysisError(f"invalid Boolean field {field}: {value!r}")


def _truth_parent_set(value: object) -> frozenset[str]:
    if _is_none(value):
        return frozenset()
    return frozenset(part.strip() for part in str(value).split(";") if part.strip())


def _parent_set(row: Mapping[str, str], prefix: str) -> frozenset[str]:
    values = []
    for suffix in ("parent1", "parent2"):
        value = row.get(f"{prefix}_{suffix}")
        if not _is_none(value):
            values.append(str(value).strip())
    return frozenset(values)


def _profile_field(fields: set[str]) -> str:
    present = [field for field in PROFILE_FIELDS if field in fields]
    if not present:
        raise AnalysisError("per-child metrics require state_profile or profile")
    return "state_profile" if "state_profile" in present else present[0]


def _read_records(path: Path) -> tuple[list[Record], tuple[str, ...], str]:
    output: list[Record] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        profile_field = _profile_field(fields)
        missing = sorted(BASE_REQUIRED_FIELDS - fields)
        if missing:
            raise AnalysisError(f"per-child metrics lack required fields: {missing}")
        observable_present = OBSERVABLE_FIELDS <= fields
        partial_observable = bool(OBSERVABLE_FIELDS & fields) and not observable_present
        if partial_observable:
            raise AnalysisError("observable truth fields must be present together")
        paired_hashes_present = PAIRED_HASH_FIELDS <= fields
        if bool(PAIRED_HASH_FIELDS & fields) and not paired_hashes_present:
            raise AnalysisError("paired provenance hash fields must be present together")

        for row_number, row in enumerate(reader, start=2):
            try:
                truth_state = _as_int(row["truth_state"], "truth_state")
                assert truth_state is not None
                truth_parents = _truth_parent_set(row["truth_parents"])
                if observable_present:
                    observable_state = _as_int(
                        row["observable_truth_state"], "observable_truth_state"
                    )
                    assert observable_state is not None
                    observable_parents = _truth_parent_set(
                        row["observable_truth_parents"]
                    )
                else:
                    observable_state = truth_state
                    observable_parents = truth_parents
                record = Record(
                    split=str(row["split"]),
                    replicate=int(_as_int(row["replicate"], "replicate")),
                    scenario=str(row["scenario"]),
                    effective_markers=_as_float(
                        row["effective_markers"], "effective_markers"
                    ),
                    sample=str(row["sample"]),
                    state_profile=str(row[profile_field]),
                    candidate_relatedness=str(row["candidate_relatedness"]),
                    eligible_candidate_count=int(
                        _as_int(
                            row["eligible_candidate_count"],
                            "eligible_candidate_count",
                        )
                    ),
                    stratum=str(row["stratum"]),
                    truth_state=truth_state,
                    truth_parents=truth_parents,
                    observable_truth_state=int(observable_state),
                    observable_truth_parents=observable_parents,
                    raw_state=_as_int(
                        row["diagnostic_LocalObservedParentCount"],
                        "diagnostic_LocalObservedParentCount",
                        optional=True,
                    ),
                    complete_state=_as_int(
                        row["complete_observed_parent_count"],
                        "complete_observed_parent_count",
                        optional=True,
                    ),
                    complete_parents=_parent_set(row, "complete"),
                    tier_a_state=_as_int(
                        row["tier_a_observed_parent_count"],
                        "tier_a_observed_parent_count",
                        optional=True,
                    ),
                    tier_a_parents=_parent_set(row, "tier_a"),
                    tier_b_state=_as_int(
                        row["tier_b_observed_parent_count"],
                        "tier_b_observed_parent_count",
                        optional=True,
                    ),
                    tier_b_parents=_parent_set(row, "tier_b"),
                    tier_a_partial_parents=_parent_set(row, "tier_a_partial"),
                    tier_b_partial_parents=_parent_set(row, "tier_b_partial"),
                    tier_a_state_call=_as_bool(
                        row["diagnostic_TierAStateCall"],
                        "diagnostic_TierAStateCall",
                    ),
                    tier_b_state_call=_as_bool(
                        row["diagnostic_TierBStateCall"],
                        "diagnostic_TierBStateCall",
                    ),
                    tier_a_exact_call=_as_bool(
                        row["diagnostic_TierAExactConfiguration"],
                        "diagnostic_TierAExactConfiguration",
                    ),
                    tier_b_exact_call=_as_bool(
                        row["diagnostic_TierBExactConfiguration"],
                        "diagnostic_TierBExactConfiguration",
                    ),
                    truth_sha256=(
                        str(row["truth_sha256"]) if paired_hashes_present else None
                    ),
                    child_genetics_sha256=(
                        str(row["child_genetics_sha256"])
                        if paired_hashes_present else None
                    ),
                    child_observations_sha256=(
                        str(row["child_observations_sha256"])
                        if paired_hashes_present else None
                    ),
                    candidate_universe_sha256=(
                        str(row["candidate_universe_sha256"])
                        if paired_hashes_present else None
                    ),
                )
            except (KeyError, TypeError, AssertionError) as exc:
                raise AnalysisError(
                    f"invalid per-child record at CSV row {row_number}"
                ) from exc
            if record.truth_state not in (0, 1, 2):
                raise AnalysisError(f"truth state outside M0/M1/M2 at row {row_number}")
            if len(record.truth_parents) != record.truth_state:
                raise AnalysisError(f"biological truth state/set mismatch at row {row_number}")
            if len(record.observable_truth_parents) != record.observable_truth_state:
                raise AnalysisError(f"observable truth state/set mismatch at row {row_number}")
            output.append(record)
    if not output:
        raise AnalysisError("per-child metrics are empty")
    bases = ("biological", "observable") if observable_present else ("biological",)
    return output, bases, profile_field


def _base_pair_key(record: Record) -> tuple[object, ...]:
    return (
        record.split,
        record.replicate,
        record.scenario,
        record.effective_markers,
        record.sample,
        record.state_profile,
        record.candidate_relatedness,
    )


def _validate_design(records: Sequence[Record], manifest: Mapping[str, object]) -> None:
    sizes = {record.eligible_candidate_count for record in records}
    if sizes != set(EXPECTED_PANEL_SIZES):
        raise AnalysisError(
            f"eligible candidate counts must be {EXPECTED_PANEL_SIZES}; observed {sorted(sizes)}"
        )
    unknown_relatedness = sorted(
        {record.candidate_relatedness for record in records} - set(RELATEDNESS_PROFILES)
    )
    if unknown_relatedness:
        raise AnalysisError(f"unknown candidate-relatedness profiles: {unknown_relatedness}")

    parameters = manifest.get("parameters", {})
    if isinstance(parameters, dict):
        expected_scenarios = set(parameters.get("scenarios", []))
        expected_effective = {
            float(value) for value in parameters.get("effective_markers", [])
        }
        if expected_scenarios and expected_scenarios != {
            record.scenario for record in records
        }:
            raise AnalysisError("record scenarios do not match manifest")
        if expected_effective and expected_effective != {
            record.effective_markers for record in records
        }:
            raise AnalysisError("record effective-marker values do not match manifest")
        expected_split = parameters.get("split")
        if expected_split is not None and {record.split for record in records} != {
            str(expected_split)
        }:
            raise AnalysisError("record split does not match manifest")
        expected_sizes = {
            int(value) for value in parameters.get("eligible_candidate_counts", [])
        }
        if expected_sizes and expected_sizes != sizes:
            raise AnalysisError("record panel sizes do not match manifest")
        expected_relatedness = set(
            parameters.get("candidate_relatedness_profiles", [])
        )
        observed_relatedness = {
            record.candidate_relatedness for record in records
        }
        if expected_relatedness and expected_relatedness != observed_relatedness:
            raise AnalysisError(
                "record candidate-relatedness profiles do not match manifest"
            )

    paired: dict[tuple[object, ...], dict[int, Record]] = defaultdict(dict)
    for record in records:
        key = _base_pair_key(record)
        if record.eligible_candidate_count in paired[key]:
            raise AnalysisError(
                "duplicate per-child panel record: "
                f"{key}, K={record.eligible_candidate_count}"
            )
        paired[key][record.eligible_candidate_count] = record
    for key, by_size in paired.items():
        if set(by_size) != set(EXPECTED_PANEL_SIZES):
            raise AnalysisError(f"incomplete nested-panel pairing for {key}: {sorted(by_size)}")
        reference = by_size[EXPECTED_PANEL_SIZES[0]]
        invariants = (
            reference.stratum,
            reference.truth_state,
            reference.truth_parents,
            reference.observable_truth_state,
            reference.observable_truth_parents,
        )
        for size in EXPECTED_PANEL_SIZES[1:]:
            candidate = by_size[size]
            if (
                candidate.stratum,
                candidate.truth_state,
                candidate.truth_parents,
                candidate.observable_truth_state,
                candidate.observable_truth_parents,
            ) != invariants:
                raise AnalysisError(f"truth changed across K for paired child {key}")
    _validate_paired_provenance(records)


def _validate_paired_provenance(records: Sequence[Record]) -> None:
    """Validate runner-supplied physical pairing hashes when available."""
    present = [record.truth_sha256 is not None for record in records]
    if any(present) and not all(present):
        raise AnalysisError("paired provenance hashes are missing from some records")
    if not any(present):
        return

    relatedness = {record.candidate_relatedness for record in records}
    expected_arms = {
        (size, profile)
        for size in EXPECTED_PANEL_SIZES
        for profile in relatedness
    }
    grouped: dict[
        tuple[object, ...],
        dict[tuple[int, str], tuple[str | None, str | None, str | None, str | None]],
    ] = defaultdict(dict)
    for record in records:
        key = (
            record.split,
            record.replicate,
            record.scenario,
            record.effective_markers,
            record.state_profile,
        )
        arm = (record.eligible_candidate_count, record.candidate_relatedness)
        hashes = (
            record.truth_sha256,
            record.child_genetics_sha256,
            record.child_observations_sha256,
            record.candidate_universe_sha256,
        )
        previous = grouped[key].setdefault(arm, hashes)
        if previous != hashes:
            raise AnalysisError(f"paired provenance hashes vary within arm {key}, {arm}")

    for key, arms in grouped.items():
        if set(arms) != expected_arms:
            raise AnalysisError(
                f"paired provenance arm set incomplete for {key}: {sorted(arms)}"
            )
        shared_physical = {values[:3] for values in arms.values()}
        if len(shared_physical) != 1:
            raise AnalysisError(
                f"truth/child genetics/observations differ across paired arms for {key}"
            )
        for profile in relatedness:
            candidate_hashes = {
                values[3]
                for (size, arm_profile), values in arms.items()
                if arm_profile == profile
            }
            if len(candidate_hashes) != 1:
                raise AnalysisError(
                    f"candidate universe changes across K for {key}, {profile}"
                )
def _candidate_id(value: object) -> str:
    text = str(value)
    if text.isdigit():
        return f"candidate_{int(text):02d}"
    return text


def _related_decoys_by_profile(
    manifest: Mapping[str, object], present_profiles: Iterable[str]
) -> dict[str, dict[str, tuple[str, ...]]]:
    fixed = manifest.get("fixed_design", {})
    if not isinstance(fixed, dict):
        fixed = {}
    by_profile = fixed.get("candidate_relationships_by_profile", {})
    flat_relationships = fixed.get("candidate_relationships", {})
    if not isinstance(by_profile, dict):
        by_profile = {}
    if not isinstance(flat_relationships, dict):
        flat_relationships = {}

    output: dict[str, dict[str, tuple[str, ...]]] = {}
    for profile in present_profiles:
        relationships = by_profile.get(profile, flat_relationships)
        if not isinstance(relationships, dict):
            relationships = {}
        labels = {
            _candidate_id(key): str(label)
            for key, label in relationships.items()
        }
        anchors: dict[str, str] = {}
        decoys: list[tuple[str, str, str]] = []
        for candidate, label in labels.items():
            group = label.split("_", 1)[0]
            if label.endswith("_anchor"):
                anchors[group] = candidate
            elif "decoy" in label:
                decoys.append((candidate, group, label))

        mapping: dict[str, tuple[str, ...]] = {}
        if profile != "unrelated":
            for group, anchor in anchors.items():
                candidates = []
                for candidate, decoy_group, label in decoys:
                    if decoy_group != group:
                        continue
                    if profile == "full_half" and "near" in label:
                        continue
                    candidates.append(candidate)
                if candidates:
                    mapping[anchor] = tuple(sorted(candidates))
        output[profile] = mapping

    requiring_decoys = set(present_profiles) - {"unrelated"}
    if requiring_decoys and any(not output[profile] for profile in requiring_decoys):
        raise AnalysisError(
            "manifest relationships cannot define required profile-aware decoys"
        )
    return output


def _truth_state(record: Record, basis: str) -> int:
    return record.truth_state if basis == "biological" else record.observable_truth_state


def _truth_set(record: Record, basis: str) -> frozenset[str]:
    return (
        record.truth_parents
        if basis == "biological"
        else record.observable_truth_parents
    )


def _tier_state(record: Record, tier: str) -> int | None:
    return record.tier_a_state if tier == "tier_a" else record.tier_b_state


def _tier_parents(record: Record, tier: str) -> frozenset[str]:
    return record.tier_a_parents if tier == "tier_a" else record.tier_b_parents


def _partial_parents(record: Record, tier: str) -> frozenset[str]:
    return (
        record.tier_a_partial_parents
        if tier == "tier_a"
        else record.tier_b_partial_parents
    )


def _state_call(record: Record, tier: str) -> bool:
    return record.tier_a_state_call if tier == "tier_a" else record.tier_b_state_call


def _exact_call(record: Record, tier: str) -> bool:
    return record.tier_a_exact_call if tier == "tier_a" else record.tier_b_exact_call


def _exact_correct(record: Record, tier: str, basis: str) -> bool:
    return bool(
        _exact_call(record, tier)
        and _tier_state(record, tier) == _truth_state(record, basis)
        and _tier_parents(record, tier) == _truth_set(record, basis)
    )


def _false_parent_release(record: Record, tier: str, basis: str) -> bool:
    if _truth_state(record, basis) != 0:
        return False
    state_false_release = bool(
        _state_call(record, tier) and _tier_state(record, tier) not in (None, 0)
    )
    return state_false_release or bool(_partial_parents(record, tier))


def _related_decoy_counts(
    record: Record,
    predicted: frozenset[str],
    basis: str,
    related_decoys: Mapping[str, Mapping[str, Sequence[str]]],
) -> tuple[float, float]:
    truth = _truth_set(record, basis)
    substitutions = 0
    at_risk = 0
    mapping = related_decoys.get(record.candidate_relatedness, {})
    for anchor, decoys in mapping.items():
        if anchor not in truth:
            continue
        at_risk += 1
        substitutions += int(
            anchor not in predicted and any(decoy in predicted for decoy in decoys)
        )
    return float(substitutions), float(at_risk)


def _seed_for(*identity: object) -> int:
    digest = hashlib.blake2b(
        "|".join(map(str, identity)).encode("utf-8"), digest_size=8
    ).digest()
    return BOOTSTRAP_SEED ^ int.from_bytes(digest, "little")


def _quantiles(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None, None
    tail = 0.5 * (1.0 - CI_LEVEL)
    lower, upper = np.quantile(finite, (tail, 1.0 - tail))
    return float(lower), float(upper)


def _cluster_ratios(
    items: Sequence[object],
    contributions: Mapping[str, Callable[[object], tuple[float, float]]],
    identity: Sequence[object],
    bootstrap_replicates: int,
) -> dict[str, dict[str, object]]:
    clusters = sorted(
        {
            item.replicate if isinstance(item, Record) else item[0].replicate
            for item in items
        }
    )
    if not clusters:
        raise AnalysisError(f"empty clustered analysis cell: {identity}")
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    names = list(contributions)
    numerators = np.zeros((len(clusters), len(names)), dtype=np.float64)
    denominators = np.zeros_like(numerators)
    for item in items:
        replicate = item.replicate if isinstance(item, Record) else item[0].replicate
        cluster = cluster_index[replicate]
        for column, name in enumerate(names):
            numerator, denominator = contributions[name](item)
            numerators[cluster, column] += float(numerator)
            denominators[cluster, column] += float(denominator)

    rng = np.random.default_rng(_seed_for(*identity))
    weights = rng.multinomial(
        len(clusters),
        np.full(len(clusters), 1.0 / len(clusters)),
        size=bootstrap_replicates,
    )
    boot_numerators = weights @ numerators
    boot_denominators = weights @ denominators
    boot_estimates = np.divide(
        boot_numerators,
        boot_denominators,
        out=np.full_like(boot_numerators, np.nan),
        where=boot_denominators > 0.0,
    )
    total_numerators = np.sum(numerators, axis=0)
    total_denominators = np.sum(denominators, axis=0)
    output: dict[str, dict[str, object]] = {}
    for column, name in enumerate(names):
        lower, upper = _quantiles(boot_estimates[:, column])
        denominator = float(total_denominators[column])
        output[name] = {
            "estimate": (
                None
                if denominator <= 0.0
                else float(total_numerators[column] / denominator)
            ),
            "numerator": float(total_numerators[column]),
            "denominator": denominator,
            "ci_lower": lower,
            "ci_upper": upper,
            "replicate_clusters": len(clusters),
            "ci_method": "replicate_cluster_percentile_bootstrap",
            "bootstrap_replicates": bootstrap_replicates,
        }
    return output


def _cell_fields(record: Record) -> dict[str, object]:
    return {
        "split": record.split,
        "scenario": record.scenario,
        "effective_markers": record.effective_markers,
        "state_profile": record.state_profile,
        "candidate_relatedness": record.candidate_relatedness,
        "eligible_candidate_count": record.eligible_candidate_count,
    }


def _record_cells(records: Sequence[Record]) -> list[list[Record]]:
    grouped: dict[tuple[object, ...], list[Record]] = defaultdict(list)
    for record in records:
        grouped[
            (
                record.split,
                record.scenario,
                record.effective_markers,
                record.state_profile,
                record.candidate_relatedness,
                record.eligible_candidate_count,
            )
        ].append(record)
    return [grouped[key] for key in sorted(grouped, key=lambda value: tuple(map(str, value)))]


def _record_scopes(
    records: Sequence[Record], basis: str, *, include_strata: bool = True
) -> list[tuple[str, str, list[Record]]]:
    scopes: list[tuple[str, str, list[Record]]] = [("all", "ALL", list(records))]
    for state in range(3):
        subset = [record for record in records if _truth_state(record, basis) == state]
        if subset:
            scopes.append(("truth_state", f"M{state}", subset))
    if include_strata:
        for stratum in sorted({record.stratum for record in records}):
            scopes.append(
                (
                    "stratum",
                    stratum,
                    [record for record in records if record.stratum == stratum],
                )
            )
    return scopes


def _core_metrics(
    records: Sequence[Record],
    truth_bases: Sequence[str],
    related_decoys: Mapping[str, Mapping[str, Sequence[str]]],
    bootstrap_replicates: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for cell in _record_cells(records):
        fields = _cell_fields(cell[0])
        for basis in truth_bases:
            for scope_type, scope, subset in _record_scopes(cell, basis):
                contributions: dict[
                    str, Callable[[object], tuple[float, float]]
                ] = {
                    "raw_state_accuracy": lambda item, b=basis: (
                        float(item.raw_state == _truth_state(item, b)),
                        1.0,
                    ),
                    "complete_state_accuracy": lambda item, b=basis: (
                        float(item.complete_state == _truth_state(item, b)),
                        1.0,
                    ),
                    "complete_joint_accuracy": lambda item, b=basis: (
                        float(
                            item.complete_state == _truth_state(item, b)
                            and item.complete_parents == _truth_set(item, b)
                        ),
                        1.0,
                    ),
                }
                for tier in TIERS:
                    contributions.update(
                        {
                            f"{tier}_state_coverage": lambda item, t=tier: (
                                float(_state_call(item, t)),
                                1.0,
                            ),
                            f"{tier}_state_selective_accuracy": lambda item, t=tier, b=basis: (
                                float(
                                    _state_call(item, t)
                                    and _tier_state(item, t) == _truth_state(item, b)
                                ),
                                float(_state_call(item, t)),
                            ),
                            f"{tier}_exact_coverage": lambda item, t=tier: (
                                float(_exact_call(item, t)),
                                1.0,
                            ),
                            f"{tier}_exact_selective_accuracy": lambda item, t=tier, b=basis: (
                                float(_exact_correct(item, t, b)),
                                float(_exact_call(item, t)),
                            ),
                            f"{tier}_exact_error_release": lambda item, t=tier, b=basis: (
                                float(_exact_call(item, t) and not _exact_correct(item, t, b)),
                                1.0,
                            ),
                            f"{tier}_exact_selective_error": lambda item, t=tier, b=basis: (
                                float(_exact_call(item, t) and not _exact_correct(item, t, b)),
                                float(_exact_call(item, t)),
                            ),
                            f"{tier}_partial_edge_precision": lambda item, t=tier, b=basis: (
                                float(len(_partial_parents(item, t) & _truth_set(item, b))),
                                float(len(_partial_parents(item, t))),
                            ),
                            f"{tier}_partial_edge_recall": lambda item, t=tier, b=basis: (
                                float(len(_partial_parents(item, t) & _truth_set(item, b))),
                                float(len(_truth_set(item, b))),
                            ),
                            f"{tier}_m0_false_parent_release": lambda item, t=tier, b=basis: (
                                float(_false_parent_release(item, t, b)),
                                float(_truth_state(item, b) == 0),
                            ),
                            f"{tier}_m1_to_m2_error": lambda item, t=tier, b=basis: (
                                float(
                                    _truth_state(item, b) == 1
                                    and _state_call(item, t)
                                    and _tier_state(item, t) == 2
                                ),
                                float(_truth_state(item, b) == 1),
                            ),
                            f"{tier}_related_decoy_substitution_exact": lambda item, t=tier, b=basis: _related_decoy_counts(
                                item,
                                _tier_parents(item, t) if _exact_call(item, t) else frozenset(),
                                b,
                                related_decoys,
                            ),
                            f"{tier}_related_decoy_substitution_partial": lambda item, t=tier, b=basis: _related_decoy_counts(
                                item,
                                _partial_parents(item, t),
                                b,
                                related_decoys,
                            ),
                        }
                    )
                values = _cluster_ratios(
                    subset,
                    contributions,
                    (
                        "metrics",
                        *fields.values(),
                        basis,
                        scope_type,
                        scope,
                    ),
                    bootstrap_replicates,
                )
                for metric, value in values.items():
                    output.append(
                        {
                            **fields,
                            "truth_basis": basis,
                            "scope_type": scope_type,
                            "scope": scope,
                            "metric": metric,
                            **value,
                        }
                    )
    return output


def _raw_confusion(
    records: Sequence[Record],
    truth_bases: Sequence[str],
    bootstrap_replicates: int,
) -> list[dict[str, object]]:
    output = []
    for cell in _record_cells(records):
        fields = _cell_fields(cell[0])
        for basis in truth_bases:
            for truth_state in range(3):
                subset = [
                    record for record in cell if _truth_state(record, basis) == truth_state
                ]
                if not subset:
                    continue
                contributions = {
                    str(selected) if selected is not None else "unresolved": (
                        lambda item, selected=selected: (
                            float(item.raw_state == selected),
                            1.0,
                        )
                    )
                    for selected in (0, 1, 2, None)
                }
                values = _cluster_ratios(
                    subset,
                    contributions,
                    ("confusion", *fields.values(), basis, truth_state),
                    bootstrap_replicates,
                )
                for selected, value in values.items():
                    output.append(
                        {
                            **fields,
                            "truth_basis": basis,
                            "truth_state": truth_state,
                            "raw_selected_state": selected,
                            **value,
                        }
                    )
    return output


def _paired_cells(
    records: Sequence[Record], k_from: int, k_to: int
) -> list[list[tuple[Record, Record]]]:
    by_child: dict[tuple[object, ...], dict[int, Record]] = defaultdict(dict)
    for record in records:
        by_child[_base_pair_key(record)][record.eligible_candidate_count] = record
    grouped: dict[tuple[object, ...], list[tuple[Record, Record]]] = defaultdict(list)
    for by_size in by_child.values():
        smaller = by_size[k_from]
        larger = by_size[k_to]
        key = (
            smaller.split,
            smaller.scenario,
            smaller.effective_markers,
            smaller.state_profile,
            smaller.candidate_relatedness,
        )
        grouped[key].append((smaller, larger))
    return [grouped[key] for key in sorted(grouped, key=lambda value: tuple(map(str, value)))]


def _pair_scopes(
    pairs: Sequence[tuple[Record, Record]], basis: str
) -> list[tuple[str, str, list[tuple[Record, Record]]]]:
    scopes: list[tuple[str, str, list[tuple[Record, Record]]]] = [
        ("all", "ALL", list(pairs))
    ]
    for state in range(3):
        subset = [pair for pair in pairs if _truth_state(pair[0], basis) == state]
        if subset:
            scopes.append(("truth_state", f"M{state}", subset))
    return scopes


def _confidence_state(record: Record, tier: str, endpoint: str, basis: str) -> str:
    if endpoint == "state":
        if not _state_call(record, tier):
            return "abstain"
        correct = _tier_state(record, tier) == _truth_state(record, basis)
    else:
        if not _exact_call(record, tier):
            return "abstain"
        correct = _exact_correct(record, tier, basis)
    return "correct_confident" if correct else "wrong_confident"


def _paired_transition_rows(
    records: Sequence[Record],
    truth_bases: Sequence[str],
    bootstrap_replicates: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for k_from, k_to in PANEL_TRANSITIONS:
        for pairs in _paired_cells(records, k_from, k_to):
            first = pairs[0][0]
            fields = {
                "split": first.split,
                "scenario": first.scenario,
                "effective_markers": first.effective_markers,
                "state_profile": first.state_profile,
                "candidate_relatedness": first.candidate_relatedness,
                "k_from": k_from,
                "k_to": k_to,
            }
            for basis in truth_bases:
                for scope_type, scope, subset in _pair_scopes(pairs, basis):
                    contributions: dict[
                        str, Callable[[object], tuple[float, float]]
                    ] = {}
                    for tier in TIERS:
                        for endpoint in ENDPOINTS:
                            for origin in CONFIDENCE_STATES:
                                for destination in CONFIDENCE_STATES:
                                    prefix = f"{tier}|{endpoint}|{origin}|{destination}"
                                    contributions[f"{prefix}|conditional"] = (
                                        lambda pair, t=tier, e=endpoint, b=basis, o=origin, d=destination: (
                                            float(
                                                _confidence_state(pair[0], t, e, b) == o
                                                and _confidence_state(pair[1], t, e, b) == d
                                            ),
                                            float(_confidence_state(pair[0], t, e, b) == o),
                                        )
                                    )
                                    contributions[f"{prefix}|all"] = (
                                        lambda pair, t=tier, e=endpoint, b=basis, o=origin, d=destination: (
                                            float(
                                                _confidence_state(pair[0], t, e, b) == o
                                                and _confidence_state(pair[1], t, e, b) == d
                                            ),
                                            1.0,
                                        )
                                    )
                    values = _cluster_ratios(
                        subset,
                        contributions,
                        (
                            "paired_transitions",
                            *fields.values(),
                            basis,
                            scope_type,
                            scope,
                        ),
                        bootstrap_replicates,
                    )
                    for tier in TIERS:
                        for endpoint in ENDPOINTS:
                            for origin in CONFIDENCE_STATES:
                                for destination in CONFIDENCE_STATES:
                                    prefix = f"{tier}|{endpoint}|{origin}|{destination}"
                                    conditional = values[f"{prefix}|conditional"]
                                    all_pairs = values[f"{prefix}|all"]
                                    output.append(
                                        {
                                            **fields,
                                            "truth_basis": basis,
                                            "scope_type": scope_type,
                                            "scope": scope,
                                            "tier": tier,
                                            "endpoint": endpoint,
                                            "origin": origin,
                                            "destination": destination,
                                            "transition": f"{origin}_to_{destination}",
                                            **conditional,
                                            "all_pairs_estimate": all_pairs["estimate"],
                                            "all_pairs_numerator": all_pairs["numerator"],
                                            "all_pairs_denominator": all_pairs["denominator"],
                                            "all_pairs_ci_lower": all_pairs["ci_lower"],
                                            "all_pairs_ci_upper": all_pairs["ci_upper"],
                                        }
                                    )
    return output


def _view_parents(record: Record, view: str) -> frozenset[str]:
    if view == "complete":
        return record.complete_parents
    if view == "tier_a_exact":
        return record.tier_a_parents if record.tier_a_exact_call else frozenset()
    if view == "tier_b_exact":
        return record.tier_b_parents if record.tier_b_exact_call else frozenset()
    if view == "tier_a_partial":
        return record.tier_a_partial_parents
    if view == "tier_b_partial":
        return record.tier_b_partial_parents
    raise AssertionError(view)

def _view_truth_match(record: Record, view: str, basis: str) -> bool:
    """Return truth identity without treating an exact abstention as M0-correct."""
    if view == "complete":
        return bool(
            record.complete_state == _truth_state(record, basis)
            and record.complete_parents == _truth_set(record, basis)
        )
    if view == "tier_a_exact":
        return _exact_correct(record, "tier_a", basis)
    if view == "tier_b_exact":
        return _exact_correct(record, "tier_b", basis)
    # Partial outputs have no state-confidence flag. Here equality describes
    # only the released edge set; it is not a confident state call.
    return _view_parents(record, view) == _truth_set(record, basis)



def _paired_identity_rows(
    records: Sequence[Record],
    truth_bases: Sequence[str],
    related_decoys: Mapping[str, Mapping[str, Sequence[str]]],
    bootstrap_replicates: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    views = ("complete", "tier_a_exact", "tier_b_exact", "tier_a_partial", "tier_b_partial")
    for k_from, k_to in PANEL_TRANSITIONS:
        for pairs in _paired_cells(records, k_from, k_to):
            first = pairs[0][0]
            fields = {
                "split": first.split,
                "scenario": first.scenario,
                "effective_markers": first.effective_markers,
                "state_profile": first.state_profile,
                "candidate_relatedness": first.candidate_relatedness,
                "k_from": k_from,
                "k_to": k_to,
            }
            for basis in truth_bases:
                for scope_type, scope, subset in _pair_scopes(pairs, basis):
                    contributions: dict[
                        str, Callable[[object], tuple[float, float]]
                    ] = {}
                    for view in views:
                        contributions.update(
                            {
                                f"{view}|identity_changed": lambda pair, v=view: (
                                    float(_view_parents(pair[0], v) != _view_parents(pair[1], v)),
                                    1.0,
                                ),
                                f"{view}|empty_to_nonempty": lambda pair, v=view: (
                                    float(not _view_parents(pair[0], v) and bool(_view_parents(pair[1], v))),
                                    1.0,
                                ),
                                f"{view}|nonempty_to_empty": lambda pair, v=view: (
                                    float(bool(_view_parents(pair[0], v)) and not _view_parents(pair[1], v)),
                                    1.0,
                                ),
                                f"{view}|changed_nonempty": lambda pair, v=view: (
                                    float(
                                        bool(_view_parents(pair[0], v))
                                        and bool(_view_parents(pair[1], v))
                                        and _view_parents(pair[0], v) != _view_parents(pair[1], v)
                                    ),
                                    1.0,
                                ),
                                f"{view}|truth_match_to_mismatch": lambda pair, v=view, b=basis: (
                                    float(
                                        _view_truth_match(pair[0], v, b)
                                        and not _view_truth_match(pair[1], v, b)
                                    ),
                                    1.0,
                                ),
                                f"{view}|mismatch_to_truth_match": lambda pair, v=view, b=basis: (
                                    float(
                                        not _view_truth_match(pair[0], v, b)
                                        and _view_truth_match(pair[1], v, b)
                                    ),
                                    1.0,
                                ),
                                f"{view}|true_edge_lost": lambda pair, v=view, b=basis: (
                                    float(
                                        bool(
                                            (_view_parents(pair[0], v) & _truth_set(pair[0], b))
                                            - _view_parents(pair[1], v)
                                        )
                                    ),
                                    1.0,
                                ),
                                f"{view}|false_edge_gained": lambda pair, v=view, b=basis: (
                                    float(
                                        bool(
                                            (_view_parents(pair[1], v) - _truth_set(pair[1], b))
                                            - _view_parents(pair[0], v)
                                        )
                                    ),
                                    1.0,
                                ),
                                f"{view}|related_decoy_substitution_gained": lambda pair, v=view, b=basis: (
                                    float(
                                        _related_decoy_counts(
                                            pair[1], _view_parents(pair[1], v), b, related_decoys
                                        )[0]
                                        > _related_decoy_counts(
                                            pair[0], _view_parents(pair[0], v), b, related_decoys
                                        )[0]
                                    ),
                                    1.0,
                                ),
                            }
                        )
                    values = _cluster_ratios(
                        subset,
                        contributions,
                        (
                            "paired_identity",
                            *fields.values(),
                            basis,
                            scope_type,
                            scope,
                        ),
                        bootstrap_replicates,
                    )
                    for key, value in values.items():
                        view, metric = key.split("|", 1)
                        output.append(
                            {
                                **fields,
                                "truth_basis": basis,
                                "scope_type": scope_type,
                                "scope": scope,
                                "view": view,
                                "metric": metric,
                                **value,
                            }
                        )
    return output


def _gate_record(
    source: Mapping[str, object],
    *,
    gate: str,
    comparison: str,
    point_threshold: float,
    ci_threshold: float | None,
    diagnostic_only: bool = False,
) -> dict[str, object]:
    estimate = source.get("estimate")
    denominator = float(source.get("denominator", 0.0) or 0.0)
    applicable = bool(denominator > 0.0 and estimate is not None)
    if not applicable:
        passed: bool | None = None
    elif comparison == "minimum":
        passed = bool(
            float(estimate) >= point_threshold
            and (
                ci_threshold is None
                or (
                    source.get("ci_lower") is not None
                    and float(source["ci_lower"]) >= ci_threshold
                )
            )
        )
    elif comparison == "maximum":
        passed = bool(
            float(estimate) <= point_threshold
            and (
                ci_threshold is None
                or (
                    source.get("ci_upper") is not None
                    and float(source["ci_upper"]) <= ci_threshold
                )
            )
        )
    else:
        raise AssertionError(comparison)
    keep = {
        key: value
        for key, value in source.items()
        if key
        not in {
            "metric",
            "estimate",
            "numerator",
            "denominator",
            "ci_lower",
            "ci_upper",
            "ci_method",
            "bootstrap_replicates",
            "replicate_clusters",
        }
    }
    return {
        **keep,
        "gate": gate,
        "comparison": comparison,
        "point_threshold": point_threshold,
        "ci_threshold": ci_threshold,
        "diagnostic_only": diagnostic_only,
        "applicable": applicable,
        "passed": passed,
        "estimate": estimate,
        "numerator": source.get("numerator"),
        "denominator": source.get("denominator"),
        "ci_lower": source.get("ci_lower"),
        "ci_upper": source.get("ci_upper"),
        "replicate_clusters": source.get("replicate_clusters"),
        "ci_method": source.get("ci_method"),
        "bootstrap_replicates": source.get("bootstrap_replicates"),
    }


def _descriptive_gates(
    metrics: Sequence[Mapping[str, object]],
    transitions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    biological = [row for row in metrics if row["truth_basis"] == "biological"]
    for row in biological:
        metric = row["metric"]
        if row["scope_type"] == "truth_state" and metric == "raw_state_accuracy":
            output.append(
                _gate_record(
                    row,
                    gate="raw_state_accuracy",
                    comparison="minimum",
                    point_threshold=RAW_STATE_MINIMUM,
                    ci_threshold=RAW_STATE_CI_LOWER_MINIMUM,
                )
            )
        if row["scope_type"] == "all" and metric in {
            "tier_a_m0_false_parent_release",
            "tier_a_m1_to_m2_error",
            "tier_a_related_decoy_substitution_partial",
            "tier_a_exact_error_release",
            "tier_a_exact_selective_error",
            "tier_b_m0_false_parent_release",
            "tier_b_m1_to_m2_error",
            "tier_b_related_decoy_substitution_partial",
            "tier_b_exact_error_release",
            "tier_b_exact_selective_error",
        }:
            output.append(
                _gate_record(
                    row,
                    gate=str(metric),
                    comparison="maximum",
                    point_threshold=SAFETY_EVENT_MAXIMUM,
                    ci_threshold=SAFETY_EVENT_CI_UPPER_MAXIMUM,
                )
            )
        if row["scope_type"] == "all" and metric in {
            "tier_a_state_coverage",
            "tier_a_exact_coverage",
            "tier_b_state_coverage",
            "tier_b_exact_coverage",
        }:
            floor = (
                TIER_A_STATE_COVERAGE_FLOOR
                if metric.endswith("_state_coverage")
                else TIER_A_EXACT_COVERAGE_FLOOR
            )
            output.append(
                _gate_record(
                    row,
                    gate=f"{metric}_floor_diagnostic",
                    comparison="minimum",
                    point_threshold=floor,
                    ci_threshold=None,
                    diagnostic_only=True,
                )
            )

    # Some predeclared state profiles deliberately contain no M0 children.
    # Preserve the frozen M0/M1/M2 gate inventory and mark absent strata as
    # not applicable instead of silently omitting their gate.
    cell_keys = (
        "split",
        "scenario",
        "effective_markers",
        "state_profile",
        "candidate_relatedness",
        "eligible_candidate_count",
    )
    cells = {
        tuple(row[key] for key in cell_keys)
        for row in biological
        if row["scope_type"] == "all"
    }
    raw_gate_keys = {
        tuple(row[key] for key in cell_keys) + (row["scope"],)
        for row in biological
        if row["scope_type"] == "truth_state"
        and row["metric"] == "raw_state_accuracy"
    }
    for cell in sorted(cells, key=lambda value: tuple(map(str, value))):
        for state in range(3):
            scope = f"M{state}"
            if cell + (scope,) in raw_gate_keys:
                continue
            source = {
                **dict(zip(cell_keys, cell)),
                "truth_basis": "biological",
                "scope_type": "truth_state",
                "scope": scope,
                "metric": "raw_state_accuracy",
                "estimate": None,
                "numerator": 0.0,
                "denominator": 0.0,
                "ci_lower": None,
                "ci_upper": None,
                "replicate_clusters": 0,
                "ci_method": "replicate_cluster_percentile_bootstrap",
                "bootstrap_replicates": (
                    biological[0]["bootstrap_replicates"] if biological else None
                ),
            }
            output.append(
                _gate_record(
                    source,
                    gate="raw_state_accuracy",
                    comparison="minimum",
                    point_threshold=RAW_STATE_MINIMUM,
                    ci_threshold=RAW_STATE_CI_LOWER_MINIMUM,
                )
            )

    for row in transitions:
        if (
            row["truth_basis"] == "biological"
            and row["scope_type"] == "all"
            and row["origin"] == "correct_confident"
            and row["destination"] == "wrong_confident"
        ):
            output.append(
                _gate_record(
                    row,
                    gate="correct_confident_to_wrong_confident",
                    comparison="maximum",
                    point_threshold=PANEL_REGRESSION_MAXIMUM,
                    ci_threshold=PANEL_REGRESSION_CI_UPPER_MAXIMUM,
                )
            )
    return output


def _summarize_gates(gates: Sequence[Mapping[str, object]]) -> dict[str, int]:
    substantive = [row for row in gates if not row["diagnostic_only"]]
    diagnostics = [row for row in gates if row["diagnostic_only"]]
    return {
        "substantive_applicable": sum(bool(row["applicable"]) for row in substantive),
        "substantive_passed": sum(row["passed"] is True for row in substantive),
        "substantive_failed": sum(row["passed"] is False for row in substantive),
        "substantive_not_applicable": sum(not row["applicable"] for row in substantive),
        "coverage_diagnostics_met": sum(row["passed"] is True for row in diagnostics),
        "coverage_diagnostics_not_met": sum(row["passed"] is False for row in diagnostics),
    }


def run(
    input_root: Path,
    output_prefix: Path,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    if bootstrap_replicates < MINIMUM_BOOTSTRAP_REPLICATES:
        raise AnalysisError(
            f"bootstrap_replicates must be >= {MINIMUM_BOOTSTRAP_REPLICATES}"
        )
    paths = _output_paths(output_prefix)
    paths["protocol"].parent.mkdir(parents=True, exist_ok=True)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise AnalysisError(f"refusing to overwrite analysis outputs: {existing}")

    status, manifest, records_path, status_sha256 = _validated_completed_input(
        input_root
    )
    records, truth_bases, profile_field = _read_records(records_path)
    _validate_design(records, manifest)
    related_decoys = _related_decoys_by_profile(
        manifest, {record.candidate_relatedness for record in records}
    )

    metrics = _core_metrics(
        records, truth_bases, related_decoys, bootstrap_replicates
    )
    confusion = _raw_confusion(records, truth_bases, bootstrap_replicates)
    transitions = _paired_transition_rows(
        records, truth_bases, bootstrap_replicates
    )
    identity = _paired_identity_rows(
        records, truth_bases, related_decoys, bootstrap_replicates
    )
    gates = _descriptive_gates(metrics, transitions)
    protocol = _analysis_protocol(bootstrap_replicates)
    protocol_hash = _canonical_hash(protocol)

    _atomic_json(
        paths["protocol"], {"protocol_sha256": protocol_hash, "protocol": protocol}
    )
    _atomic_csv(paths["metrics"], metrics)
    _atomic_csv(paths["confusion"], confusion)
    _atomic_csv(paths["paired_transitions"], transitions)
    _atomic_csv(paths["identity_changes"], identity)
    _atomic_csv(paths["gates"], gates)

    output_identities = {
        name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
        if name != "summary"
    }
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_hash,
        "analysis_source_sha256": _sha256(Path(__file__).resolve()),
        "input_root": str(input_root.resolve()),
        "input_completion_status_sha256": status_sha256,
        "input_manifest_sha256": status.get("manifest_sha256"),
        "input_manifest_identity_normalization": manifest.get(
            "_analysis_input_identity_normalization"
        ),
        "input_profile_field": profile_field,
        "paired_provenance_hashes_validated": all(
            record.truth_sha256 is not None for record in records
        ),
        "child_panel_records_evaluated": len(records),
        "paired_children": len(records) // len(EXPECTED_PANEL_SIZES),
        "truth_bases": list(truth_bases),
        "splits": sorted({record.split for record in records}),
        "scenarios": sorted({record.scenario for record in records}),
        "effective_markers": sorted({record.effective_markers for record in records}),
        "state_profiles": sorted({record.state_profile for record in records}),
        "candidate_relatedness": sorted(
            {record.candidate_relatedness for record in records}
        ),
        "eligible_candidate_counts": list(EXPECTED_PANEL_SIZES),
        "selection_or_tuning_performed": False,
        "selected_effective_markers": None,
        "production_eligible": False,
        "gate_summary": _summarize_gates(gates),
        "output_files": output_identities,
    }
    _atomic_json(paths["summary"], summary)
    return summary


def _synthetic_record(
    replicate: int,
    size: int,
    truth_state: int,
    *,
    wrong_at_16: bool = False,
) -> Record:
    truth = (
        frozenset()
        if truth_state == 0
        else frozenset({"candidate_00"})
        if truth_state == 1
        else frozenset({"candidate_00", "candidate_16"})
    )
    wrong = wrong_at_16 and size == 16
    state = 2 if wrong else truth_state
    parents = (
        frozenset({"candidate_02", "candidate_16"}) if wrong else truth
    )
    return Record(
        split="smoke",
        replicate=replicate,
        scenario="clean",
        effective_markers=1.0,
        sample=f"child_m{truth_state}",
        state_profile="mixed",
        candidate_relatedness="full_half_near",
        eligible_candidate_count=size,
        stratum=f"m{truth_state}",
        truth_state=truth_state,
        truth_parents=truth,
        observable_truth_state=truth_state,
        observable_truth_parents=truth,
        raw_state=state,
        complete_state=state,
        complete_parents=parents,
        tier_a_state=state,
        tier_a_parents=parents,
        tier_b_state=state,
        tier_b_parents=parents,
        tier_a_partial_parents=parents,
        tier_b_partial_parents=parents,
        tier_a_state_call=True,
        tier_b_state_call=True,
        tier_a_exact_call=True,
        tier_b_exact_call=True,
        truth_sha256="truth",
        child_genetics_sha256="child-genetics",
        child_observations_sha256="child-observations",
        candidate_universe_sha256="candidate-universe-full-half-near",
    )


def _selftest() -> None:
    if _profile_field({"profile"}) != "profile":
        raise AssertionError("legacy profile-field normalization failed")
    records = [
        _synthetic_record(
            replicate,
            size,
            truth_state,
            wrong_at_16=(replicate == 0 and truth_state == 1),
        )
        for replicate in range(3)
        for truth_state in range(3)
        for size in EXPECTED_PANEL_SIZES
    ]
    manifest = {
        "parameters": {
            "split": "smoke",
            "scenarios": ["clean"],
            "effective_markers": [1.0],
            "eligible_candidate_counts": list(EXPECTED_PANEL_SIZES),
            "candidate_relatedness_profiles": ["full_half_near"],
        },
        "fixed_design": {
            "candidate_relationships_by_profile": {
                "full_half_near": {
                    "0": "female_normal_truth_core_anchor",
                    "2": "female_full_sibling_decoy",
                    "3": "female_half_sibling_decoy",
                    "4": "female_near_duplicate_decoy",
                    "16": "male_normal_truth_core_anchor",
                    "18": "male_full_sibling_decoy",
                    "19": "male_half_sibling_decoy",
                    "20": "male_near_duplicate_decoy",
                }
            }
        },
    }
    _validate_design(records, manifest)
    decoys = _related_decoys_by_profile(manifest, {"full_half_near"})
    # Exercise the requested minimum bootstrap size on a bounded ratio.
    ratio = _cluster_ratios(
        [record for record in records if record.eligible_candidate_count == 8],
        {"raw": lambda item: (float(item.raw_state == item.truth_state), 1.0)},
        ("selftest", "ratio"),
        MINIMUM_BOOTSTRAP_REPLICATES,
    )["raw"]
    if ratio["estimate"] != 1.0 or ratio["ci_lower"] != 1.0:
        raise AssertionError("clustered ratio self-test failed")
    # Use ten draws for structural table smoke; scientific CLI output enforces >=2000.
    metrics = _core_metrics(records, ("biological",), decoys, 10)
    transitions = _paired_transition_rows(records, ("biological",), 10)
    identity = _paired_identity_rows(records, ("biological",), decoys, 10)
    regressions = [
        row
        for row in transitions
        if row["k_from"] == 8
        and row["k_to"] == 16
        and row["scope"] == "ALL"
        and row["tier"] == "tier_a"
        and row["endpoint"] == "state"
        and row["origin"] == "correct_confident"
        and row["destination"] == "wrong_confident"
    ]
    if len(regressions) != 1 or regressions[0]["numerator"] != 1.0:
        raise AssertionError("paired confidence-transition self-test failed")
    if not metrics or not identity or not _descriptive_gates(metrics, transitions):
        raise AssertionError("panel analysis table smoke failed")

    # Hash-validation smoke using the same completed-output convention as the runner.
    with tempfile.TemporaryDirectory(prefix="smart-panel-analysis-") as temporary:
        root = Path(temporary)
        manifest_path = root / "manifest.json"
        records_path = root / "per_child_release_metrics.csv"
        _atomic_json(manifest_path, manifest)
        _atomic_csv(
            records_path,
            [
                {
                    "split": "smoke",
                    "replicate": 0,
                    "scenario": "clean",
                    "effective_markers": 1.0,
                    "sample": "child_m1",
                    "state_profile": "mixed",
                    "candidate_relatedness": "full_half_near",
                    "eligible_candidate_count": size,
                    "stratum": "m1",
                    "truth_state": 1,
                    "truth_parents": "candidate_00",
                    "observable_truth_state": 1,
                    "observable_truth_parents": "candidate_00",
                    "complete_observed_parent_count": 1,
                    "complete_parent1": "candidate_00",
                    "complete_parent2": None,
                    "tier_a_observed_parent_count": 1,
                    "tier_a_parent1": "candidate_00",
                    "tier_a_parent2": None,
                    "tier_b_observed_parent_count": 1,
                    "tier_b_parent1": "candidate_00",
                    "tier_b_parent2": None,
                    "tier_a_partial_parent1": "candidate_00",
                    "tier_a_partial_parent2": None,
                    "tier_b_partial_parent1": "candidate_00",
                    "tier_b_partial_parent2": None,
                    "diagnostic_LocalObservedParentCount": 1,
                    "diagnostic_TierAStateCall": True,
                    "diagnostic_TierBStateCall": True,
                    "diagnostic_TierAExactConfiguration": True,
                    "diagnostic_TierBExactConfiguration": True,
                }
                for size in EXPECTED_PANEL_SIZES
            ],
        )
        status = {
            "computational_complete": True,
            "manifest_sha256": _canonical_hash(manifest),
            "outputs": {
                path.name: {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in (manifest_path, records_path)
            },
        }
        _atomic_json(root / "status.computational_complete.json", status)
        _, loaded_manifest, loaded_records_path, _ = _validated_completed_input(root)
        loaded, bases, profile_field = _read_records(loaded_records_path)
        _validate_design(loaded, loaded_manifest)
        if len(loaded) != 3 or bases != ("biological", "observable"):
            raise AssertionError("completed-input reader self-test failed")
        if profile_field != "state_profile":
            raise AssertionError("state-profile field normalization failed")
        summary = run(
            root,
            root / "analysis" / "panel",
            bootstrap_replicates=MINIMUM_BOOTSTRAP_REPLICATES,
        )
        if summary["paired_children"] != 1:
            raise AssertionError("end-to-end analysis smoke failed")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if args.selftest:
            _selftest()
            print("panel analysis self-test passed")
        else:
            summary = run(
                args.input_root,
                args.output_prefix,
                bootstrap_replicates=args.bootstrap_replicates,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
    except (AnalysisError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
