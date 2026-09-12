"""painting / checkpoints for the canonical reconstruction pipeline."""
from __future__ import annotations


import copy
from dataclasses import dataclass
import json
from typing import Any, Mapping
import numpy as np
import haplotype_reconstruction.painting.components as painting_components

T09_COMPONENT_CHECKPOINT_SCHEMA = "t09-component-painting-checkpoint-v4"


COMPONENT_PAINTING_PRODUCT_SCHEMA = "stage2-component-painting-product-v3"


COMPONENT_PAINTING_PRODUCT_BACKEND = (
    "unified-open-set-frozen-background-component-viterbi-posterior-v3"
)


def _validate_identity_record(record, label="Stage-2 identity"):
    if not isinstance(record, Mapping) or not record:
        raise ValueError(f"{label} must be a nonempty mapping")
    schema = record.get("schema")
    backend = record.get("backend")
    if not isinstance(schema, str) or not schema:
        raise ValueError(f"{label} requires a nonempty schema")
    if not isinstance(backend, str) or not backend:
        raise ValueError(f"{label} requires a nonempty backend")
    config = record.get(
        "config", record.get("scientific_and_runtime_config")
    )
    if not isinstance(config, Mapping) or not config:
        raise ValueError(f"{label} requires a nonempty config envelope")


def component_painting_product_identity(
        scientific_config: Mapping[str, Any],
        code_identity_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build the canonical scientific identity for a component painting."""

    if not isinstance(scientific_config, Mapping) or not scientific_config:
        raise ValueError("painting scientific config must be a nonempty mapping")
    if not isinstance(code_identity_sha256, Mapping) or not code_identity_sha256:
        raise ValueError("painting code identity must be a nonempty mapping")
    record = {
        "schema": COMPONENT_PAINTING_PRODUCT_SCHEMA,
        "backend": COMPONENT_PAINTING_PRODUCT_BACKEND,
        "config": copy.deepcopy(dict(scientific_config)),
        "code_identity_sha256": copy.deepcopy(dict(code_identity_sha256)),
        "component_checkpoint_schema": T09_COMPONENT_CHECKPOINT_SCHEMA,
    }
    _validate_identity_record(record, "painting-product identity")
    return json.loads(json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ))


@dataclass(frozen=True)
class ScientificIdentity:
    """Canonical immutable copy of the Stage-2 backend/configuration identity."""

    canonical_json: str

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "FrozenStage2Identity":
        _validate_identity_record(record)
        canonical = json.dumps(
            copy.deepcopy(dict(record)),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return cls(canonical)

    def record(self) -> dict[str, Any]:
        try:
            value = json.loads(self.canonical_json)
        except (TypeError, ValueError) as error:
            raise ValueError("Stage-2 identity is not valid canonical JSON") from error
        _validate_identity_record(value)
        return value


@dataclass(frozen=True)
class RuntimeProvenance:
    """Canonical runtime record that is not part of product identity."""

    canonical_json: str

    @classmethod
    def from_record(
            cls, record: Mapping[str, Any]) -> "FrozenRuntimeProvenance":
        if not isinstance(record, Mapping) or not record:
            raise ValueError("runtime provenance must be a nonempty mapping")
        canonical = json.dumps(
            copy.deepcopy(dict(record)),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return cls(canonical)

    def record(self) -> dict[str, Any]:
        try:
            value = json.loads(self.canonical_json)
        except (TypeError, ValueError) as error:
            raise ValueError("runtime provenance is not valid canonical JSON") from error
        if not isinstance(value, dict) or not value:
            raise ValueError("runtime provenance must decode to a nonempty mapping")
        return value


@dataclass(frozen=True)
class PaintingCheckpoint:
    """Atomic release-bound component painting product.

    ``release_identity`` binds the exact canonical Stage-2 components and
    their inputs. ``painting_product_identity`` independently binds the
    scientific painting model and its code. Scheduling choices live only in
    ``runtime_provenance`` and therefore are not scientific product identity.
    """

    schema: str
    component_manifest: dict[str, Any]
    painting_bundle: painting_components.ComponentPaintingBundle
    sample_ids: tuple[str, ...]
    release_identity: ScientificIdentity
    painting_product_identity: ScientificIdentity
    runtime_provenance: RuntimeProvenance


def _freeze_stage2_identity(value) -> ScientificIdentity:
    if isinstance(value, ScientificIdentity):
        value.record()
        return value
    if not isinstance(value, Mapping):
        raise ValueError("Stage-2 identity is missing")
    return ScientificIdentity.from_record(value)


def _freeze_runtime_provenance(value) -> RuntimeProvenance:
    if isinstance(value, RuntimeProvenance):
        value.record()
        return value
    if not isinstance(value, Mapping):
        raise ValueError("runtime provenance is missing")
    return RuntimeProvenance.from_record(value)


def _canonical_sample_ids(sample_ids) -> tuple[str, ...]:
    try:
        result = tuple(str(value) for value in sample_ids)
    except TypeError as error:
        raise TypeError("sample_ids must be an ordered collection") from error
    if not result:
        raise ValueError("sample_ids must be nonempty")
    if len(result) != len(set(result)):
        raise ValueError("sample_ids must be unique and ordered")
    return result


def _snapshot_component_manifest(manifest) -> dict[str, Any]:
    core_runtime.validate_phase_component_manifest(manifest)
    records = manifest[core_runtime.PHASE_COMPONENTS_KEY]
    return {
        "schema": manifest["schema"],
        core_runtime.PHASE_COMPONENTS_KEY: tuple(
            dict(record) for record in records
        ),
    }


def build_t09_component_checkpoint(
        component_manifest,
        painting_bundle,
        sample_ids,
        release_identity,
        painting_product_identity,
        runtime_provenance,
) -> PaintingCheckpoint:
    """Build and validate one schema-versioned T09 checkpoint payload."""

    payload = PaintingCheckpoint(
        schema=T09_COMPONENT_CHECKPOINT_SCHEMA,
        component_manifest=_snapshot_component_manifest(component_manifest),
        painting_bundle=painting_bundle,
        sample_ids=_canonical_sample_ids(sample_ids),
        release_identity=_freeze_stage2_identity(release_identity),
        painting_product_identity=_freeze_stage2_identity(painting_product_identity),
        runtime_provenance=_freeze_runtime_provenance(runtime_provenance),
    )
    return validate_t09_component_checkpoint(payload)


def _validate_ragged_diagnostics(
        component, block, sample_count, painting_config):
    """Validate the posterior-qualified open-set release contract."""

    diagnostics = component.ragged_diagnostics
    if component.informative_site_count == 0:
        if diagnostics is not None:
            raise ValueError("empty ragged painting must not carry diagnostics")
        return
    if not isinstance(diagnostics, module_painting_model.RaggedPaintingDiagnostics):
        raise TypeError("ragged painting lacks typed diagnostics")

    selected = np.asarray(diagnostics.selected_site_indices)
    positions = np.asarray(diagnostics.selected_positions)
    block_positions = np.asarray(block.positions)
    if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("ragged selected site indices must be an integer vector")
    if (len(selected) != component.informative_site_count
            or np.any(selected < 0) or np.any(selected >= len(block_positions))
            or not np.array_equal(positions, block_positions[selected])):
        raise ValueError("ragged selected sites disagree with the component")

    classes = diagnostics.equivalence_classes
    if (not isinstance(classes, tuple)
            or any(not isinstance(value, tuple) or not value
                   for value in classes)):
        raise TypeError("ragged equivalence classes must be nonempty tuples")
    flattened = []
    for members in classes:
        for value in members:
            if (isinstance(value, bool) or int(value) != value or value < 0):
                raise ValueError("ragged equivalence classes contain an invalid row")
            flattened.append(int(value))
    if (len(set(flattened)) != len(flattened)
            or sorted(flattened) != list(range(len(component.founder_keys)))):
        raise ValueError("ragged equivalence classes do not partition founder rows")

    alleles = np.asarray(diagnostics.named_alleles)
    if (alleles.shape != (len(classes), len(selected))
            or np.any(~np.isin(alleles, (-1, 0, 1)))):
        raise ValueError("ragged named allele grid is invalid")
    class_is_anchored = np.any(alleles >= 0, axis=1)
    snapshot = np.asarray(
        getattr(block, "missing_aware_inference_discrete_haps", None)
    )
    expected_snapshot_shape = (len(component.founder_keys), len(block_positions))
    if (snapshot.shape != expected_snapshot_shape
            or np.any(~np.isin(snapshot, (-1, 0, 1)))):
        raise ValueError("ragged component lacks its frozen founder snapshot")
    for class_index, members in enumerate(classes):
        expected = snapshot[int(members[0]), selected]
        if not np.array_equal(alleles[class_index], expected):
            raise ValueError("ragged named alleles disagree with frozen founders")
        for member in members[1:]:
            if not np.array_equal(snapshot[int(member), selected], expected):
                raise ValueError("ragged equivalence class mixes founder trajectories")
    signatures = [alleles[index].tobytes() for index in range(len(classes))]
    if len(set(signatures)) != len(signatures):
        raise ValueError("ragged duplicate founder trajectories were not pooled")

    centers = np.asarray(diagnostics.bin_centers)
    edges = np.asarray(diagnostics.bin_edges)
    if (centers.ndim != 1 or edges.shape != (len(centers) + 1,)
            or np.any(np.diff(edges) <= 0)
            or np.any((centers < edges[:-1]) | (centers >= edges[1:]))):
        raise ValueError("ragged bin coordinates are invalid")
    expected_grid = (sample_count, 2, len(centers))
    labels = np.asarray(diagnostics.map_label_grid)
    class_grid = np.asarray(diagnostics.map_state_class_grid)
    viterbi_grid = np.asarray(diagnostics.viterbi_state_class_grid)
    statuses = np.asarray(diagnostics.track_status_grid)
    direct = np.asarray(diagnostics.map_direct_callability)
    background_index = len(classes)
    for name, value, minimum in (
        ("raw Viterbi state-class", viterbi_grid, 0),
        ("released state-class", class_grid, -1),
        ("released label", labels, -1),
    ):
        if (value.shape != expected_grid or np.any(~np.isfinite(value))
                or np.any(value != np.floor(value)) or np.any(value < minimum)):
            raise ValueError(f"ragged {name} grid is invalid")
    if np.any(viterbi_grid > background_index):
        raise ValueError("ragged raw Viterbi grid has an invalid class")
    if np.any(class_grid > background_index):
        raise ValueError("ragged released grid has an invalid class")
    if np.any(labels >= len(component.founder_keys)):
        raise ValueError("ragged released label grid has an invalid founder")
    if direct.shape != expected_grid or direct.dtype != np.bool_:
        raise ValueError("ragged direct-callability grid has the wrong shape")
    if statuses.shape != expected_grid or statuses.dtype != np.uint8:
        raise ValueError("ragged track-status grid is invalid")
    valid_statuses = np.asarray(
        [int(value) for value in module_painting_model.PaintingTrackStatus], dtype=np.uint8
    )
    if np.any(~np.isin(statuses, valid_statuses)):
        raise ValueError("ragged track-status grid has an unknown status")

    eligible = np.asarray(component.evidence_eligible_sample_mask)
    posterior_names = (
        "posterior_max_class_mass",
        "posterior_viterbi_public_class_mass",
        "posterior_class_entropy",
        "posterior_background_mass",
        "posterior_public_unknown_mass",
    )
    for name in posterior_names:
        value = np.asarray(getattr(diagnostics, name))
        if value.shape != (sample_count, len(centers)):
            raise ValueError(f"ragged {name} has the wrong shape")
        if np.any(~np.isfinite(value[eligible])):
            raise ValueError(f"ragged {name} is not finite for eligible samples")
        if np.any(~np.isnan(value[~eligible])):
            raise ValueError(f"ragged {name} must abstain for ineligible samples")
        if name != "posterior_class_entropy" and np.any(
                (value[eligible] < 0.0) | (value[eligible] > 1.0)):
            raise ValueError(f"ragged {name} lies outside [0, 1]")
        if name == "posterior_class_entropy" and np.any(value[eligible] < 0.0):
            raise ValueError("ragged posterior entropy is negative")

    try:
        threshold = float(
            painting_config["minimum_viterbi_public_class_posterior"]
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "painting identity lacks the public posterior release threshold"
        ) from error
    if (not np.isfinite(threshold) or not 0.5 < threshold <= 1.0
            or diagnostics.minimum_viterbi_public_class_posterior != threshold):
        raise ValueError("ragged posterior threshold disagrees with product identity")

    qv = np.asarray(diagnostics.posterior_viterbi_public_class_mass)
    maximum = np.asarray(diagnostics.posterior_max_class_mass)
    background_mass = np.asarray(diagnostics.posterior_background_mass)
    unknown_mass = np.asarray(diagnostics.posterior_public_unknown_mass)
    tolerance = 32 * np.finfo(np.float64).eps
    if np.any(qv[eligible] > maximum[eligible] + tolerance):
        raise ValueError("ragged Viterbi public posterior exceeds public maximum")
    if np.any(background_mass[eligible] > unknown_mass[eligible] + tolerance):
        raise ValueError("ragged BACKGROUND mass exceeds public UNKNOWN mass")
    status_type = module_painting_model.PaintingTrackStatus
    class_status = np.full(len(classes) + 1, int(status_type.BACKGROUND),
                           dtype=np.uint8)
    class_labels = np.full(len(classes) + 1, -1, dtype=np.int64)
    for class_index, members in enumerate(classes):
        if not class_is_anchored[class_index]:
            class_status[class_index] = int(status_type.UNANCHORED_TRAJECTORY)
        elif len(members) > 1:
            class_status[class_index] = int(status_type.POOLED_EQUIVALENCE)
        else:
            class_status[class_index] = int(status_type.SINGLETON_NAMED)
            class_labels[class_index] = members[0]
    # The grids have already been checked to contain integer-valued classes.
    state_indices = viterbi_grid.astype(np.intp, copy=False)
    expected_status = class_status[state_indices]
    expected_labels = class_labels[state_indices]
    expected_classes = viterbi_grid.copy()
    low = np.broadcast_to((qv < threshold)[:, None, :], expected_grid)
    expected_status[low] = int(status_type.LOW_POSTERIOR_ABSTENTION)
    expected_labels[low] = -1
    expected_classes[low] = -1
    expected_status[~eligible] = int(status_type.INELIGIBLE_NO_EVIDENCE)
    expected_labels[~eligible] = -1
    expected_classes[~eligible] = -1
    if not np.array_equal(statuses, expected_status):
        raise ValueError("ragged track status disagrees with qV/public state")
    if not np.array_equal(class_grid, expected_classes):
        raise ValueError("ragged released classes disagree with qV/raw Viterbi")
    if not np.array_equal(labels, expected_labels):
        raise ValueError("ragged released labels disagree with class semantics")
    source_status = (
        (statuses == int(module_painting_model.PaintingTrackStatus.SINGLETON_NAMED))
        | (statuses == int(module_painting_model.PaintingTrackStatus.POOLED_EQUIVALENCE))
    )
    if np.any(direct & ~source_status):
        raise ValueError("ragged direct callability requires represented ancestry")

    for name in ("biological_switch_counts", "structural_handoff_counts"):
        value = np.asarray(getattr(diagnostics, name))
        if value.shape != (sample_count,) or np.any(value < 0):
            raise ValueError(f"ragged {name} is invalid")
    for name in (
        "hmm_batch_size", "hmm_thread_count",
        "hmm_working_memory_budget_bytes", "hmm_estimated_bytes_per_sample",
    ):
        value = getattr(diagnostics, name)
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"ragged {name} is invalid")
    if diagnostics.hmm_batch_size > sample_count:
        raise ValueError("ragged HMM batch exceeds the sample count")


def _chunks_match_label_bins(chunks, centers, labels):
    """Check half-open chunk coverage in O(C log M + M), without a C-by-M scan.

    Difference arrays count covering intervals and sum their one-based IDs.
    Exactly one covering interval makes that sum its unique owner. This also
    preserves overlap/gap detection and does not require chunks to be sorted.
    """
    if not len(centers):
        return True
    starts = np.asarray([chunk.start for chunk in chunks])
    ends = np.asarray([chunk.end for chunk in chunks])
    if np.any(starts >= ends):
        return False
    left = np.searchsorted(centers, starts, side="left")
    right = np.searchsorted(centers, ends, side="left")
    coverage = np.zeros(len(centers) + 1, dtype=np.int64)
    owner = np.zeros_like(coverage)
    ids = np.arange(1, len(chunks) + 1, dtype=np.int64)
    np.add.at(coverage, left, 1)
    np.add.at(coverage, right, -1)
    if np.any(np.cumsum(coverage[:-1]) != 1):
        return False
    np.add.at(owner, left, ids)
    np.add.at(owner, right, -ids)
    selected = np.cumsum(owner[:-1]) - 1
    hap1 = np.asarray([chunk.hap1 for chunk in chunks])[selected]
    hap2 = np.asarray([chunk.hap2 for chunk in chunks])[selected]
    return bool(np.all(
        ((hap1 == labels[0]) & (hap2 == labels[1]))
        | ((hap1 == labels[1]) & (hap2 == labels[0]))
    ))


def validate_t09_component_checkpoint(
        payload,
        *,
        expected_sample_ids=None,
        expected_release_identity=None,
        expected_painting_product_identity=None,
) -> PaintingCheckpoint:
    """Validate an atomic component checkpoint and return the typed payload."""

    if not isinstance(payload, PaintingCheckpoint):
        raise TypeError(
            "T09 requires a bound typed component checkpoint"
        )
    if payload.schema != T09_COMPONENT_CHECKPOINT_SCHEMA:
        raise ValueError("unknown T09 component checkpoint schema")
    if not isinstance(payload.release_identity, ScientificIdentity):
        raise ValueError("T09 component checkpoint lacks release identity")
    release_identity = payload.release_identity.record()
    if not isinstance(payload.painting_product_identity, ScientificIdentity):
        raise ValueError("T09 component checkpoint lacks painting-product identity")
    painting_identity = payload.painting_product_identity.record()
    painting_config = painting_identity["config"]
    if not isinstance(payload.runtime_provenance, RuntimeProvenance):
        raise ValueError("T09 component checkpoint lacks runtime provenance")
    payload.runtime_provenance.record()

    if not isinstance(payload.sample_ids, tuple):
        raise TypeError("checkpoint sample_ids must be an immutable tuple")
    sample_ids = _canonical_sample_ids(payload.sample_ids)
    if sample_ids != payload.sample_ids:
        raise ValueError("checkpoint sample IDs must use their canonical strings")
    if expected_sample_ids is not None:
        expected = _canonical_sample_ids(expected_sample_ids)
        if sample_ids != expected:
            raise ValueError("checkpoint sample order does not match expected order")

    if expected_release_identity is not None:
        expected_identity = _freeze_stage2_identity(expected_release_identity)
        if payload.release_identity != expected_identity:
            raise ValueError("checkpoint release identity mismatch")
    if expected_painting_product_identity is not None:
        expected_painting = _freeze_stage2_identity(
            expected_painting_product_identity
        )
        if payload.painting_product_identity != expected_painting:
            raise ValueError("checkpoint painting-product identity mismatch")

    if not isinstance(payload.painting_bundle, painting_components.ComponentPaintingBundle):
        raise TypeError("checkpoint lacks a component painting bundle")
    bundle = payload.painting_bundle
    if (
            isinstance(bundle.num_samples, bool)
            or int(bundle.num_samples) != bundle.num_samples
            or bundle.num_samples < 0):
        raise ValueError("painting bundle has an invalid sample count")
    if bundle.num_samples != len(sample_ids):
        raise ValueError(
            "painting bundle sample count does not match ordered sample IDs"
        )
    if not isinstance(bundle.components, tuple):
        raise TypeError("painting bundle components must be an immutable tuple")

    blocks = core_runtime.validate_phase_component_manifest(
        payload.component_manifest
    )
    records = payload.component_manifest[core_runtime.PHASE_COMPONENTS_KEY]
    if len(bundle.components) != len(records):
        raise ValueError(
            "painting and phase manifests have different component counts"
        )

    for component_id, (record, block, component) in enumerate(
            zip(records, blocks, bundle.components)
    ):
        if not isinstance(component, painting_components.ComponentPaintingResult):
            raise TypeError("painting bundle contains an unrecognized component")
        if component.component_index != component_id:
            raise ValueError("painting component IDs must be consecutive and ordered")
        if component.painting_model != painting_components.PAINTING_MODEL_RAGGED:
            raise ValueError(
                "T09 requires the unified open-set ragged painting model"
            )

        expected_interval = (
            int(record["position_start"]),
            int(record["position_end"]),
        )
        if tuple(component.interval) != expected_interval:
            raise ValueError(
                f"painting component {component_id} interval disagrees with manifest"
            )
        expected_founder_keys = tuple(record["component_local_founder_ids"])
        if tuple(component.founder_keys) != expected_founder_keys:
            raise ValueError(
                f"painting component {component_id} founder keys disagree with manifest"
            )

        block_identity = getattr(block, "stage2_component_identity", None)
        if not isinstance(block_identity, Mapping):
            raise ValueError(
                f"phase component {component_id} lacks release identity"
            )
        if _freeze_stage2_identity(block_identity).record() != release_identity:
            raise ValueError(
                f"phase component {component_id} release identity mismatch"
            )

        eligible = np.asarray(component.evidence_eligible_sample_mask)
        if eligible.shape != (len(sample_ids),) or eligible.dtype != np.bool_:
            raise ValueError(
                f"painting component {component_id} evidence-eligible mask "
                "does not match sample IDs"
            )
        if len(component.sample_break_reasons) != len(sample_ids):
            raise ValueError(
                f"painting component {component_id} sample break reasons "
                "do not match sample IDs"
            )

        if component.painting_model == painting_components.PAINTING_MODEL_RAGGED:
            _validate_ragged_diagnostics(
                component, block, len(sample_ids), painting_config
            )

        samples = getattr(component.painting, "samples", None)
        if samples is None or len(samples) != len(sample_ids):
            raise ValueError(
                f"painting component {component_id} sample count mismatch"
            )
        for sample_index, sample in enumerate(samples):
            if getattr(sample, "sample_index", None) != sample_index:
                raise ValueError(
                    f"painting component {component_id} sample order mismatch"
                )
            chunks = tuple(getattr(sample, "chunks", ()))
            diagnostics = component.ragged_diagnostics
            if diagnostics is not None:
                if not eligible[sample_index]:
                    if chunks:
                        raise ValueError(
                            f"painting component {component_id} gives chunks "
                            "to an ineligible sample"
                        )
                else:
                    centers = np.asarray(diagnostics.bin_centers)
                    expected_grid = np.asarray(diagnostics.map_label_grid)
                    if not _chunks_match_label_bins(
                            chunks, centers, expected_grid[sample_index]):
                        raise ValueError(
                            f"painting component {component_id} chunks disagree "
                            "with released label bins"
                        )
            for chunk in chunks:
                if chunk.start >= chunk.end:
                    raise ValueError(
                        f"painting component {component_id} has a non-positive chunk"
                    )
                if not (
                    -1 <= chunk.hap1 < len(expected_founder_keys)
                    and -1 <= chunk.hap2 < len(expected_founder_keys)
                ):
                    raise ValueError(
                        f"painting component {component_id} uses an invalid "
                        "founder label for its painting model"
                    )

    return payload

import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.painting.model as module_painting_model
