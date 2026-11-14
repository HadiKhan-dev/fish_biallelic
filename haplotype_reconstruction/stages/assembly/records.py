"""Publish assembly products and partial results under derivation identities.

Owns assembly keys, prepared/component codecs and the scoped final refinement record capability.
Every read verifies the expected identity. Joins are contig-wide and normalize
transport axes; output publication is atomic and never uses physical record IDs.
Method: docs/methods/assembly.md (derivations and publication records).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from haplotype_reconstruction.contracts.assembly import (
    AssemblyProduct,
    BestFReplacement,
    CompletionProduct,
    ComponentRefinement,
    CountDecisionRecord,
    FitAcceptanceTotals,
    FitOrdinalBinding,
    FitSummary,
    FitWorkCounters,
    JoinedLevel,
    JoinRecord,
    JoinTask,
    LevelRecord,
    LogicalWindowReceipt,
    PhysicalWindowWork,
    RefinedLevel,
    ReleaseFamily,
    WeakJunction,
    WindowAccount,
    WindowDomain,
    WindowOrdinal,
    WindowPathAlphabet,
    WindowRangeAlphabet,
    WindowStop,
    WindowWorkCounters,
    WindowWorkShape,
)
from haplotype_reconstruction.contracts.panels import CountCall, CountStepDecision, FounderComponent, PreparedPanel
from haplotype_reconstruction.contracts.payload import to_payload
from haplotype_reconstruction.contracts.roles import require_schema
from haplotype_reconstruction.core.digest import array_digest, json_digest
from haplotype_reconstruction.core.ties import TieCount, TieCounts
from haplotype_reconstruction.store.container import CheckpointFile, RecordInfo
from haplotype_reconstruction.store.identity import CodeIdentity, code_dependency_identity
from haplotype_reconstruction.store.protocol import (
    Derivation,
    NumericalPolicy,
    bound_payload_identity,
    is_complete,
    publish,
    read_items,
    resume,
)
from haplotype_reconstruction.store.store import CheckpointStore

if TYPE_CHECKING:
    from haplotype_reconstruction.science.assembly.join.diagnostics import JoinSummary


FINAL_PREFIX = "assembly"
JOINS = "assembly/joins/"
COMPLETION = "completion"
FINAL = "final"
SCHEMA = 1


@dataclass(frozen=True)
class RecordCapability:
    """Identity-checked load/save/scoped capability supplied to refinement stages.

    identity binds all inputs of the enclosing operation. A child record's key
    is part of its derivation, so records from other iterations cannot collide.
    Component-scoped capabilities must be created with component_derivation.
    """

    file: CheckpointFile
    prefix: str
    identity: str

    def key(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def derivation(self, name: str) -> str:
        return json_digest(dict(parent=self.identity, record=name))

    def scoped(self, sub: str, *, nthreads: int | None = None) -> RecordCapability:
        return RecordCapability(self.file, self.key(sub), self.derivation(sub))

    def load(self, name: str) -> dict[str, Any] | None:
        found = resume(self.file, self.key(name), self.derivation(name))
        return None if found is None else found.payload

    def load_items(self, name: str, items: tuple[str, ...]) -> dict[str, Any] | None:
        """Check the derivation before decoding only the requested top-level fields."""
        found = read_items(self.file, self.key(name), self.derivation(name), items)
        return None if found is None else found.payload

    def has(self, name: str) -> bool:
        return is_complete(self.file, {self.key(name): self.derivation(name)})

    def save(self, name: str, payload: Mapping[str, Any]) -> RecordInfo:
        return publish(self.file, self.key(name), self.derivation(name), payload)


def identity_value(value: Any) -> Any:
    """Hash array leaves while preserving scalar/container structure and presence."""
    if isinstance(value, np.ndarray):
        return {"array": array_digest(value)}
    if isinstance(value, ReleaseFamily):
        return {key: identity_value(item) for key, item in value.items()}
    if is_dataclass(value):
        return {
            f.name: identity_value(getattr(value, f.name))
            for f in fields(value)
            if f.name not in ("schema",) and f.metadata.get("stored", True)
        }
    if isinstance(value, Mapping):
        return {key: identity_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [identity_value(item) for item in value]
    return value


def operation_derivation(
    parameters: Any,
    upstream: Mapping[str, str],
    *,
    reverse: bool = False,
    roots: tuple[str, ...] = ("haplotype_reconstruction.stages.assembly.tasks", "haplotype_reconstruction.stages.assembly.final"),
    numerical_policy: NumericalPolicy | None = None,
    code: CodeIdentity | None = None,
) -> Derivation:
    """Bind declared code closure, parameter values, upstream payloads and convention. ``code``: the closure of
    ``roots`` already hashed by a caller deriving many operations at once."""
    return Derivation(
        code_dependency_identity(roots) if code is None else code,
        identity_value(parameters),
        numerical_policy or NumericalPolicy.current(),
        dict(upstream),
        tie_reverse=reverse,
        schema=SCHEMA,
    )


def component_derivation(
    parent: str, component: FounderComponent, *, level: int, context: Any, parameters: Any
) -> str:
    """final refinement component identity includes its ordered coarse alphabet."""
    return json_digest(
        dict(
            parent=parent,
            level=level,
            component=identity_value(component),
            context=identity_value(context),
            parameters=identity_value(parameters),
        )
    )


def normalise_join(task: JoinTask, scoring: np.ndarray) -> tuple[JoinTask, np.ndarray]:
    """Rebase sample/site transport axes without changing gathered values or bp positions."""
    first = task.prepared[0].site_start
    stop = task.prepared[-1].site_start + task.prepared[-1].sites
    local_scoring = np.ascontiguousarray(scoring[task.samples, first:stop])
    prepared = tuple(replace(p, block=index, site_start=p.site_start - first) for index, p in enumerate(task.prepared))
    return replace(
        task, index=0, prepared=prepared, samples=np.arange(len(task.samples), dtype=np.int64)
    ), local_scoring


JOIN_CODE_ROOTS = (
    "haplotype_reconstruction.science.assembly.hierarchy.batch",
    "haplotype_reconstruction.stages.assembly.records",
)


def join_code() -> CodeIdentity:
    """The code closure every join derivation binds: hash it once per level, not once per batch (reading and parsing
    its sources dominated planning a level)."""
    return code_dependency_identity(JOIN_CODE_ROOTS)


def join_derivation(
    task: JoinTask,
    scoring: np.ndarray,
    *,
    evidence_identity: str,
    numerical_policy: NumericalPolicy | None = None,
    code: CodeIdentity | None = None,
) -> Derivation:
    """Identity of precisely the normalized batch data, masks, branches and sources.

    Cached PreparedPanel.digest is never read. Absolute bp positions, inference
    and release arrays are bound directly. L1 omits unused horizon controls.
    ``code``: ``join_code()``, when the caller derives a level's batches together.
    """
    local, data = normalise_join(task, scoring)
    settings = identity_value(local.settings)
    if local.settings.level_one:
        settings["gap_horizon_scale"] = None
        settings["recombination_tolerance"] = None
    parameters = dict(
        settings=settings,
        prepared=identity_value(local.prepared),
        components=identity_value(local.components),
        branches=identity_value(local.branches),
        branch_sources=identity_value(local.branch_sources),
        common_usable=identity_value(local.common_usable),
        common_mask_lost=local.common_mask_lost,
        common_usable_lost=local.common_usable_lost,
        overflow=local.overflow,
        samples=identity_value(local.samples),
    )
    return operation_derivation(
        parameters,
        dict(evidence=evidence_identity, scoring=array_digest(data)),
        reverse=local.settings.reverse,
        roots=JOIN_CODE_ROOTS,
        numerical_policy=numerical_policy,
        code=code,
    )


def load_join(file: CheckpointFile, identity: str, *, index: int) -> JoinRecord | None:
    stored = resume(file, JOINS + identity, identity)
    return None if stored is None else decode_join(stored.payload, index=index)


def save_join(file: CheckpointFile, identity: str, result: JoinRecord) -> None:
    publish(file, JOINS + identity, identity, encode_join(result))


def encode_join(record: JoinRecord) -> dict:
    payload = to_payload(record)
    payload["release_alternatives"] = [dict(f) for f in record.release_alternatives]
    payload["summary"] = None if record.summary is None else record.summary.record()
    return payload


def decode_join(payload: Mapping[str, Any], *, index: int) -> JoinRecord:
    require_schema(payload, SCHEMA)
    from haplotype_reconstruction.science.assembly.join.diagnostics import JoinSummary

    payload = dict(payload, index=index)
    payload["summary"] = None if payload["summary"] is None else JoinSummary.from_record(payload["summary"])
    payload["release_alternatives"] = tuple(ReleaseFamily.from_mapping(f) for f in payload["release_alternatives"])
    return JoinRecord(**payload)


def encode_level(record: LevelRecord) -> dict:
    payload = to_payload(record)
    payload["joins"] = [s.record() for s in record.joins]
    return payload


def decode_level(payload: Mapping[str, Any]) -> LevelRecord:
    require_schema(payload, SCHEMA)
    from haplotype_reconstruction.science.assembly.join.diagnostics import JoinSummary

    payload = dict(payload)
    payload["joins"] = tuple(JoinSummary.from_record(s) for s in payload["joins"])
    for name in ("plan", "log_likelihoods", "boundary_joint", "shadows"):
        payload[name] = tuple(tuple(x) if isinstance(x, list) else x for x in payload[name])
    return LevelRecord(**payload)


def encode_families(families: Sequence[Mapping[str, Any] | ReleaseFamily]) -> tuple[dict, ...]:
    return tuple(dict(ReleaseFamily.from_mapping(f)) for f in families)


def decode_families(payload: Sequence[Mapping[str, Any]]) -> tuple[ReleaseFamily, ...]:
    return tuple(ReleaseFamily.from_mapping(f) for f in payload)


def level_key(level: int) -> str:
    return f"l{level}/join"


def refine_scope(level: int) -> str:
    return f"l{level}/refine"


def refined_key(level: int) -> str:
    return f"l{level}/refined"


def _concatenate(arrays: Sequence[np.ndarray], dtype: Any) -> np.ndarray:
    """Preserve array dtypes, including an empty prepared/component record."""
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=dtype)


def encode_prepared(prepared: Sequence[PreparedPanel]) -> dict:
    """Prepared panels as a few concatenated arrays (one record, no per-block objects)."""
    return dict(
        schema=SCHEMA,
        block=np.asarray([p.block for p in prepared], np.int64),
        site_start=np.asarray([p.site_start for p in prepared], np.int64),
        sites=np.asarray([p.sites for p in prepared], np.int64),
        rows=np.asarray([p.rows for p in prepared], np.int64),
        positions=_concatenate([p.positions for p in prepared], np.int64),
        kept=_concatenate([p.kept for p in prepared], np.bool_),
        released=_concatenate([p.released.ravel() for p in prepared], np.int8),
        inference=_concatenate([p.inference.ravel() for p in prepared], np.int8),
        alt_probability=_concatenate([p.alt_probability.ravel() for p in prepared], np.float64),
        support=_concatenate([p.support.ravel() for p in prepared], np.float64),
    )


def decode_prepared(record: Mapping[str, Any]) -> tuple[PreparedPanel, ...]:
    """Inverse of :func:`encode_prepared` (read-only views of the record's arrays)."""
    require_schema(record, SCHEMA)
    for name in ("positions", "kept", "released", "inference", "alt_probability", "support"):
        record[name].setflags(write=False)
    site = cell = 0
    panels = []
    for block, start, sites, rows in zip(
        record["block"].tolist(),
        record["site_start"].tolist(),
        record["sites"].tolist(),
        record["rows"].tolist(),
        strict=False,
    ):
        cells = rows * sites

        def matrix(
            name: str, *, cell: int = cell, cells: int = cells, rows: int = rows, sites: int = sites
        ) -> np.ndarray:
            return record[name][cell : cell + cells].reshape(rows, sites)

        panels.append(
            PreparedPanel(
                block=block,
                site_start=start,
                positions=record["positions"][site : site + sites],
                kept=record["kept"][site : site + sites],
                released=matrix("released"),
                inference=matrix("inference"),
                alt_probability=matrix("alt_probability"),
                support=matrix("support"),
            )
        )
        site += sites
        cell += cells
    return tuple(panels)


def _encode_count_call(call: CountCall | None) -> dict[str, Any] | None:
    if call is None:
        return None
    return dict(
        founders=call.founders,
        founders_before=call.founders_before,
        ambiguous=call.ambiguous,
        reason=call.reason,
        decisions=[
            dict(
                kind=d.kind,
                founders=d.founders,
                accept_larger=d.accept_larger,
                skipped_by_bound=d.skipped_by_bound,
                margin=d.margin,
                band=None if d.band is None else list(d.band),
                ambiguous=d.ambiguous,
            )
            for d in call.decisions
        ],
    )


def _decode_count_call(value: Mapping[str, Any] | None) -> CountCall | None:
    if value is None:
        return None
    decisions = tuple(
        CountStepDecision(
            kind=d["kind"],
            founders=d["founders"],
            accept_larger=d["accept_larger"],
            skipped_by_bound=d["skipped_by_bound"],
            margin=d["margin"],
            band=None if d["band"] is None else tuple(d["band"]),
            ambiguous=d["ambiguous"],
        )
        for d in value["decisions"]
    )
    return CountCall(
        founders=value["founders"],
        founders_before=value["founders_before"],
        ambiguous=value["ambiguous"],
        decisions=decisions,
        reason=value["reason"],
    )


def encode_components(components: Sequence[FounderComponent]) -> dict:
    """Founder components as concatenated row paths plus small per-component metadata."""
    return dict(
        schema=SCHEMA,
        start=np.asarray([c.start for c in components], np.int64),
        stop=np.asarray([c.stop for c in components], np.int64),
        founders=np.asarray([c.founders for c in components], np.int64),
        rows=_concatenate([np.asarray(c.rows, np.int32).ravel() for c in components], np.int32),
        break_before=[c.break_before for c in components],
        break_after=[c.break_after for c in components],
        count_call=[_encode_count_call(c.count_call) for c in components],
    )


def decode_components(record: Mapping[str, Any]) -> tuple[FounderComponent, ...]:
    """Inverse of :func:`encode_components`."""
    require_schema(record, SCHEMA)
    components = []
    cell = 0
    for index, (start, stop, founders) in enumerate(
        zip(record["start"].tolist(), record["stop"].tolist(), record["founders"].tolist(), strict=False)
    ):
        cells = founders * (stop - start)
        rows = record["rows"][cell : cell + cells].reshape(founders, stop - start)
        components.append(
            FounderComponent(
                start=start,
                stop=stop,
                rows=rows,
                break_before=record["break_before"][index],
                break_after=record["break_after"][index],
                count_call=_decode_count_call(record["count_call"][index]),
            )
        )
        cell += cells
    return tuple(components)


def encode_completion(product: CompletionProduct) -> dict:
    return dict(schema=SCHEMA, prepared=encode_prepared(product.prepared), summary=product.summary)


def decode_completion(payload: Mapping[str, Any]) -> CompletionProduct:
    require_schema(payload, SCHEMA)
    from haplotype_reconstruction.contracts.assembly import CompletionProduct

    return CompletionProduct(decode_prepared(payload["prepared"]), payload["summary"])


def encode_joined(product: JoinedLevel) -> dict:
    return dict(
        schema=SCHEMA,
        components=encode_components(product.components),
        record=encode_level(product.record),
        # Scalar control permits availability checks without decoding scientific arrays.
        stopped=product.record.stop_reason is not None,
        release_alternatives=encode_families(product.release_alternatives),
    )


def decode_joined(payload: Mapping[str, Any]) -> JoinedLevel:
    require_schema(payload, SCHEMA)
    from haplotype_reconstruction.contracts.assembly import JoinedLevel

    return JoinedLevel(
        decode_components(payload["components"]),
        decode_level(payload["record"]),
        decode_families(payload["release_alternatives"]),
    )


def encode_refined(product: RefinedLevel) -> dict:
    return dict(
        schema=SCHEMA,
        components=encode_components(product.components),
        summary=product.summary,
        release_alternatives=encode_families(product.release_alternatives),
    )


def decode_refined(payload: Mapping[str, Any]) -> RefinedLevel:
    require_schema(payload, SCHEMA)
    from haplotype_reconstruction.contracts.assembly import RefinedLevel

    return RefinedLevel(
        decode_components(payload["components"]), payload["summary"], decode_families(payload["release_alternatives"])
    )


def encode_product(product: AssemblyProduct) -> dict:
    return dict(
        schema=SCHEMA,
        components=encode_components(product.components),
        levels=[encode_level(item) for item in product.levels],
        release_alternatives=encode_families(product.release_alternatives),
    )


def decode_product(payload: Mapping[str, Any], *, prepared: Sequence[PreparedPanel] = ()) -> AssemblyProduct:
    require_schema(payload, SCHEMA)
    from haplotype_reconstruction.contracts.assembly import AssemblyProduct

    return AssemblyProduct(
        decode_components(payload["components"]),
        tuple(decode_level(r) for r in payload["levels"]),
        decode_families(payload["release_alternatives"]),
        tuple(prepared),
    )


def assembly_identity(
    parameters: Any,
    upstream: Mapping[str, str],
    *,
    role: str,
    levels: int | None,
    reverse: bool = False,
    tie_keep: tuple[str, ...] = (),
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind a role's complete inputs and only the parameters that role consumes.

    Upstream names are selected panels, founder evidence, source payload and map.
    Context assembly never consumes refinement settings or the final level floor.
    RecordCapability derives each completed product from this operation identity.
    """
    from haplotype_reconstruction.core.errors import InvariantError

    required = {"panels", "evidence", "payload", "map"}
    if set(upstream) != required:
        raise InvariantError(f"assembly identity needs {sorted(required)}")
    settings = identity_value(parameters)
    if role == "context":
        settings.pop("refinement", None)
        settings.pop("minimum_levels", None)
    elif role != "final":
        raise InvariantError(f"unknown assembly role {role!r}")
    derivation = operation_derivation(
        dict(role=role, levels=levels, parameters=settings),
        upstream,
        reverse=reverse,
        numerical_policy=numerical_policy,
    )
    return replace(derivation, tie_keep=tie_keep).identity


@dataclass(frozen=True)
class Reader:
    """Structural results request with an independently recomputed derivation."""

    key: str
    identity: str
    product: str
    prepared: tuple[PreparedPanel, ...] = ()
    schema: int = SCHEMA
    records: RecordCapability | None = None

    def decode(self, payload: Mapping[str, Any]) -> Any:
        from haplotype_reconstruction.contracts.panels import materialise

        require_schema(payload, self.schema)
        prepared = self.prepared if self.records is None else _read_prepared(self.records)
        if self.product in ("founders", "assembly"):
            result = decode_product(payload, prepared=prepared)
            return result if self.product == "assembly" else materialise(result.components, result.prepared)
        components = decode_components(payload["components"])
        return materialise(components, prepared)


def _read_prepared(records: RecordCapability) -> tuple[PreparedPanel, ...]:
    from haplotype_reconstruction.core.errors import InputError

    completion = records.load_items(COMPLETION, ("schema", "prepared"))
    if completion is None:
        raise InputError("assembly panels need their completed preparation record")
    require_schema(completion, SCHEMA)
    return decode_prepared(completion["prepared"])


def expected(store, manifest, *, contig, round_name=None, upstream=None, levels=None):
    """Bind this role's input construction and complete record namespace once."""
    from ...contracts.payload import from_payload
    from ...core.errors import InputError
    from ...science.assembly.parameters import AssemblyParameters
    from ..evidence import records as evidence_records
    from ..relatedness.working_map import map_identity
    from ..selection import records as selection_records

    file = store.contig(contig)
    prefix = FINAL_PREFIX if round_name is None else f"local_selection/{round_name}/assembly"
    ties = manifest["ties"]
    policy = NumericalPolicy(**manifest["numerical_policy"])
    if upstream is None:
        evidence = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
        selected = selection_records.result_reader(
            store,
            manifest,
            product="selected",
            contig=contig,
            stage=None if round_name is None else "local_selection_" + ("initial" if round_name == "l1" else "l1"),
        )
        if evidence is None or selected is None:
            raise InputError("assembly reader needs published evidence and selected panels")
        # Verify each independently derived upstream before using its identity.
        if (
            read_items(file, evidence.key, evidence.identity, ("schema",)) is None
            or read_items(file, selected.key, selected.identity, ("schema",)) is None
        ):
            raise InputError("assembly upstream publication is incomplete")
        upstream = dict(
            panels=selected.identity,
            evidence=evidence.identity,
            payload=bound_payload_identity(file),
            map=map_identity(store, manifest, contig),
        )
    parameters = from_payload(AssemblyParameters, manifest["parameters"]["assembly"])
    root = assembly_identity(
        parameters,
        upstream,
        role="final" if round_name is None else "context",
        levels=None if round_name is None else int(round_name[1:]) if levels is None else levels,
        reverse=bool(ties["reverse"]),
        tie_keep=tuple(ties["keep"]),
        numerical_policy=policy,
    )
    capability = RecordCapability(file, prefix, root)
    return capability


def result_reader(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
    stage: str | None = None,
    upstream: Mapping[str, str] | None = None,
    round_name: str | None = None,
) -> Reader | TieReader | ContextReader | JoinSearchReader | None:
    """Rebuild final/release requests from bound settings and checked upstream readers.

    Composition may supply already verified upstream identities. Otherwise the
    selection/evidence/painting/refit adapters derive them from the same manifest.
    No expected identity is read from an assembly publication envelope.
    """
    from haplotype_reconstruction.core.errors import InputError

    if contig is None or contig not in manifest["contigs"]:
        raise InputError("assembly reader needs a bound contig")
    if product not in (
        "founders",
        "assembly",
        "stage_panels",
        "tie_sources",
        "join_searches",
    ):
        raise InputError(f"unsupported assembly reader product {product!r}")
    if round_name is not None and (
        round_name not in ("l1", "l2") or product not in ("assembly", "join_searches", "tie_sources")
    ):
        raise InputError("context reader needs assembly, join_searches or tie_sources and round_name l1 or l2")
    prefix = FINAL_PREFIX if round_name is None else f"local_selection/{round_name}/assembly"
    file = store.contig(contig)
    requested_key = COMPLETION if product in ("tie_sources", "join_searches") else FINAL
    if product == "stage_panels" and stage != "founder_refinement":
        if stage is None or not stage.startswith("assembly_l"):
            raise InputError(f"unsupported assembly panel stage {stage!r}")
        try:
            ordinal = int(stage.removeprefix("assembly_l"))
        except ValueError as error:
            raise InputError(f"invalid assembly panel stage {stage!r}") from error
        if ordinal < 1:
            raise InputError("assembly panel level must be positive")
        requested_key = level_key(ordinal)
    if round_name is not None:
        requested_key = COMPLETION
    physical_key = prefix + "/" + requested_key
    if not file.has(physical_key):
        return None
    capability = expected(store, manifest, contig=contig, round_name=round_name, upstream=upstream)
    if product == "tie_sources":
        return TieReader(capability, context=round_name is not None)
    if product == "join_searches":
        return JoinSearchReader(capability)
    completion = capability.load_items(COMPLETION, ("schema",))
    if completion is None:
        raise InputError("assembly panels need their completed preparation record")
    require_schema(completion, SCHEMA)
    if round_name is not None:
        for ordinal in range(1, int(round_name[1:]) + 1):
            saved = capability.load_items(level_key(ordinal), ("schema", "stopped"))
            if saved is None:
                return None
            require_schema(saved, SCHEMA)
            if not isinstance(saved.get("stopped"), bool):
                raise InputError("context join lacks its terminal control field")
            if saved["stopped"]:
                return ContextReader(capability, ordinal)
        return None
    key = requested_key
    if product == "stage_panels" and stage != "founder_refinement" and capability.has(refined_key(ordinal)):
        key = refined_key(ordinal)
    if not capability.has(key):
        return None
    return Reader(capability.key(key), capability.derivation(key), product, records=capability)


def encode_refinement(refinement: ComponentRefinement) -> dict[str, Any]:
    value = asdict(refinement)
    if value.get("window_stop") is None:
        value.pop("window_stop", None)
    return value


def _decode_window_account(value: Mapping[str, Any] | None) -> WindowAccount | None:
    if value is None:
        return None
    value = dict(value)
    value["work_shape"] = WindowWorkShape(**value["work_shape"])
    value["phase_work"] = tuple(value["phase_work"])
    value["phase_work_units"] = tuple(value["phase_work_units"])
    return WindowAccount(**value)


def _decode_window_domain(value: Mapping[str, Any]) -> WindowDomain:
    value = dict(value)
    alphabet = dict(value["alphabet"])
    if alphabet["encoding"] == "prepared_row_range":
        alphabet["sizes"] = tuple(alphabet["sizes"])
        value["alphabet"] = WindowRangeAlphabet(**alphabet)
    elif alphabet["encoding"] == "prepared_row_paths":
        alphabet["paths"] = tuple(tuple(tuple(row) for row in rows) for rows in alphabet["paths"])
        value["alphabet"] = WindowPathAlphabet(**alphabet)
    else:
        raise ValueError("unknown refinement window alphabet")
    value["account"] = _decode_window_account(value["account"])
    value["source_panel"] = tuple(tuple(row) for row in value["source_panel"])
    value["unit_spans"] = tuple(tuple(span) for span in value["unit_spans"])
    value["focals"] = tuple(value["focals"])
    return WindowDomain(**value)


def _decode_window_stop(value: Mapping[str, Any]) -> WindowStop:
    value = dict(value)
    value["first"] = WindowOrdinal(**value["first"])
    value["fits"] = tuple(
        FitOrdinalBinding(**dict(item, input_panel=tuple(tuple(row) for row in item["input_panel"])))
        for item in value["fits"]
    )
    value["prefix"] = tuple(
        LogicalWindowReceipt(WindowOrdinal(**item["ordinal"]), _decode_window_account(item["account"]), item["reused"])
        for item in value["prefix"]
    )
    value["domain"] = _decode_window_domain(value["domain"])
    value["work"] = WindowWorkCounters(**value["work"])
    value["physical_work"] = PhysicalWindowWork(**value["physical_work"])
    return WindowStop(**value)


def decode_fit_summary(value: Mapping[str, Any]) -> FitSummary:
    """Restore complete native fit counters; sparse outer diagnostics have their own codecs."""
    value = dict(value)
    for name, record in (("accepted", FitAcceptanceTotals), ("work_counters", FitWorkCounters)):
        if not isinstance(value[name], record):
            value[name] = record(**value[name])
    return FitSummary(**value)


def decode_refinement(value: Mapping[str, Any]) -> ComponentRefinement:
    value = dict(value)
    for item in value["decisions"]:
        require_schema(item, 2)
    value["decisions"] = tuple(CountDecisionRecord(**item) for item in value["decisions"])
    value["fits"] = tuple(decode_fit_summary(item) for item in value["fits"])
    value["weak_label_junctions"] = tuple(WeakJunction(**item) for item in value["weak_label_junctions"])
    if value["best_f_replacement"] is not None:
        value["best_f_replacement"] = BestFReplacement(**value["best_f_replacement"])
    if value.get("window_stop") is not None:
        value["window_stop"] = _decode_window_stop(value["window_stop"])
    return ComponentRefinement(**value)


@dataclass(frozen=True)
class TieSource:
    """OU19's explicit native group, containing counters from that group only."""

    record_group: str
    counters: tuple[TieCounts, ...]
    discovery: tuple = ()
    refinement_agreement: tuple[bool | None, ...] = ()
    count_calls: tuple[CountCall, ...] = ()


def _tie_counts(value: Any) -> TieCounts:
    """Decode the owning counter's typed payload or its unchanged native key map."""
    if isinstance(value, TieCounts):
        return value
    if "entries" in value:
        return TieCounts(tuple(TieCount(**entry) for entry in value["entries"]))
    return TieCounts(tuple(TieCount(kind, **counts) for kind, counts in sorted(value.items())))


def context_tie_sources(records: RecordCapability) -> tuple[TieSource, ...]:
    """Whitelist context completion and joined levels, with their native group names.

    The caller has bound the context capability independently of its envelopes.
    Digest join-cache entries and any refinement/fit records are never traversed.
    """
    completion = records.load_items(COMPLETION, ("schema", "summary"))
    if completion is None:
        return ()
    sources = [TieSource(records.key(COMPLETION), (_tie_counts(completion["summary"]["tie_events"]),))]
    level = 1
    while (joined := records.load_items(level_key(level), ("schema", "record"))) is not None:
        summaries = decode_level(joined["record"]).joins
        sources.append(TieSource(records.key(level_key(level)), tuple(_tie_counts(s.tie_events) for s in summaries)))
        level += 1
    return tuple(sources)


@dataclass(frozen=True)
class TieReader:
    """Read only assembly/final refinement's OU19 whitelist under independently reconstructed scopes."""

    records: RecordCapability
    context: bool = False
    schema: int = SCHEMA
    items: tuple[str, ...] = ("schema", "summary")

    @property
    def key(self) -> str:
        return self.records.key(COMPLETION)

    @property
    def identity(self) -> str:
        return self.records.derivation(COMPLETION)

    def decode(self, payload: Mapping[str, Any]) -> tuple[TieSource, ...]:
        from haplotype_reconstruction.core.errors import InputError

        require_schema(payload, SCHEMA)
        sources = list(context_tie_sources(self.records))
        if self.context:
            return tuple(sources)
        level = 1
        while self.records.has(level_key(level)):
            index = self.records.load(refinement_index_key(level))
            counters: list[TieCounts] = []
            agreement: list[bool | None] = []
            if index is None:
                if self.records.has(refined_key(level)):
                    raise InputError("refined level lacks its bound component index")
                break
            require_schema(index, SCHEMA)
            for entry in index["components"]:
                item = RecordCapability(self.records.file, entry["prefix"], entry["identity"]).load_items(
                    entry["name"], ("refinement",)
                )
                if item is None:
                    continue
                refinement = decode_refinement(item["refinement"])
                counters.append(_tie_counts(refinement.tie_events))
                agreement.extend(d.tie_decisions_agree for d in refinement.decisions)
            sources.append(
                TieSource(self.records.key(refine_scope(level)), tuple(counters), refinement_agreement=tuple(agreement))
            )
            level += 1
        final = self.records.load_items(FINAL, ("schema", "components"))
        if final is not None:
            calls = tuple(c.count_call for c in decode_components(final["components"]) if c.count_call is not None)
            sources.append(TieSource(self.records.key(FINAL), (), count_calls=calls))
        return tuple(sources)


def context_join_searches(records: RecordCapability) -> tuple[tuple[int, int, int, JoinSummary], ...]:
    """Each joined batch once, in level/batch order, under a caller-bound scope.

    B/assembly/records.py::join_searches. Pass-through batches advance over their
    unchanged output components and never repeat a prior level's search.
    """
    completion = records.load_items(COMPLETION, ("schema", "prepared"))
    if completion is None:
        return ()
    return JoinSearchReader(records).decode(completion)


@dataclass(frozen=True)
class JoinSearchReader:
    """Identity-checked final or feedback search summaries for painting reports."""

    records: RecordCapability
    schema: int = SCHEMA
    items: tuple[str, ...] = ("schema", "prepared")

    @property
    def key(self) -> str:
        return self.records.key(COMPLETION)

    @property
    def identity(self) -> str:
        return self.records.derivation(COMPLETION)

    def decode(self, payload: Mapping[str, Any]) -> tuple[tuple[int, int, int, JoinSummary], ...]:
        from haplotype_reconstruction.core.errors import InputError

        require_schema(payload, SCHEMA)
        prepared = payload["prepared"]
        require_schema(prepared, SCHEMA)
        block_end: np.ndarray = np.cumsum(prepared["sites"])
        block_first = block_end - prepared["sites"]
        positions = prepared["positions"]
        result = []
        level = 1
        while (stored := self.records.load_items(level_key(level), ("schema", "components", "record"))) is not None:
            require_schema(stored, SCHEMA)
            components = stored["components"]
            require_schema(components, SCHEMA)
            starts, stops = components["start"], components["stop"]
            record = decode_level(stored["record"])
            output = joined = 0
            for first, end, digest in record.plan:
                if digest is None:
                    output += end - first
                    continue
                if output >= len(starts) or joined >= len(record.joins):
                    raise InputError("assembly search plan does not match its joined components")
                start, stop = int(starts[output]), int(stops[output])
                result.append(
                    (
                        level,
                        int(positions[block_first[start]]),
                        int(positions[block_end[stop - 1] - 1]),
                        # decode_level decodes every join summary as a JoinSummary.
                        cast("JoinSummary", record.joins[joined]),
                    )
                )
                output += 1
                joined += 1
            if output != len(starts) or joined != len(record.joins):
                raise InputError("assembly search plan has unmatched components or joins")
            level += 1
        return tuple(result)


def refinement_index_key(level: int) -> str:
    """Parent-bound index permits checked report reads without rerunning science."""
    return f"l{level}/refinement_index"


def progress(file: CheckpointFile) -> dict[str, Any]:
    """Report committed container-index entries without opening any payload.

    Presence and byte counts describe durable work. Schema and compatibility with
    current settings remain the independently bound result reader's responsibility.
    Physical storage tokens are never used as scientific derivation identities.
    """
    selected = []
    for key in file.keys("assembly/"):
        tail = key.removeprefix("assembly/")
        parts = tail.split("/")
        if tail in (COMPLETION, FINAL) or (
            len(parts) == 2 and parts[0].startswith("l") and parts[0][1:].isdigit() and parts[1] in ("join", "refined")
        ):
            selected.append(key)
    if file.has("release"):
        selected.append("release")
    records = []
    for key in sorted(selected):
        info = file.info(key)
        if info is not None:
            records.append(dict(key=key, decoded_bytes=info.decoded_bytes, stored_bytes=info.stored_bytes))
    return dict(
        published=records,
        final_published=any(r["key"] == "assembly/final" for r in records),
        release_published=any(r["key"] == "release" for r in records),
    )


@dataclass(frozen=True)
class ContextReader:
    """A checked terminal context join plus the preceding ordered level records."""

    records: RecordCapability
    terminal_level: int
    schema: int = SCHEMA

    @property
    def key(self) -> str:
        return self.records.key(level_key(self.terminal_level))

    @property
    def identity(self) -> str:
        return self.records.derivation(level_key(self.terminal_level))

    def decode(self, payload: Mapping[str, Any]) -> AssemblyProduct:
        from haplotype_reconstruction.core.errors import InputError

        joined = decode_joined(payload)
        levels = []
        for ordinal in range(1, self.terminal_level):
            saved = self.records.load_items(level_key(ordinal), ("schema", "record"))
            if saved is None:
                raise InputError("context assembly lacks a preceding level")
            require_schema(saved, SCHEMA)
            levels.append(decode_level(saved["record"]))
        levels.append(joined.record)
        return AssemblyProduct(
            joined.components, tuple(levels), joined.release_alternatives, _read_prepared(self.records)
        )
