"""Publish painting and its optional packed-emission cache independently.

Reads verify caller-supplied derivations. Four load routes independently select
reconstructed panels and emissions. Raw paths and posteriors are never stored.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ...contracts.assembly import AssemblyProduct, JoinSummaryView
from ...contracts.painting import Painting, PaintingSummary, painting_from_payload, painting_to_payload
from ...contracts.panels import FounderComponent, PreparedPanel
from ...contracts.payload import from_payload, to_payload
from ...contracts.roles import require_schema
from ...core.errors import InputError, InvariantError
from ...core.ties import TieCounts, TieSettings
from ...store.container import CheckpointFile
from ...store.protocol import NumericalPolicy, bound_payload_identity, publish, read_items, resume
from ...store.store import CheckpointStore

__all__ = [
    "PAINTING",
    "REFERENCE",
    "reference_key",
    "reference_derivation",
    "SUMMARY",
    "CACHE",
    "DIAGNOSTICS",
    "IDENTITY",
    "SCHEMA",
    "Adapter",
    "write",
    "load",
]
PAINTING = "painting"
REFERENCE = "painting/reference"
SUMMARY = "painting/summary"
CACHE = "painting/cache"
DIAGNOSTICS = "painting/diagnostics"
IDENTITY = "identity/painting"
SCHEMA = 1
Assembly = tuple[tuple[PreparedPanel, ...], tuple[FounderComponent, ...]]


from ...science.painting.parameters import PaintingParameters
from ...store.identity import code_dependency_identity, pin_generation
from ...store.protocol import Derivation


def derivation(
    parameters: PaintingParameters,
    *,
    contig: str,
    sample_ids: tuple[str, ...],
    assembly_identity: str,
    evidence_identity: str,
    payload_identity: str,
    map_identity: str,
    calibration_identity: str,
    ties: TieSettings = TieSettings(),
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind scientific inputs and convention; exclude cache, batches and thread counts."""
    return Derivation(
        code_dependency_identity(("science.painting", "stages.painting.tasks", "stages.painting.records")),
        dict(parameters=parameters, contig=contig, sample_ids=sample_ids),
        numerical_policy or NumericalPolicy.current(),
        dict(assembly_final=assembly_identity, evidence=evidence_identity, payload=payload_identity, map=map_identity, calibration=calibration_identity),
        tie_reverse=ties.reverse,
        tie_keep=ties.keep,
    ).identity


def reference_key(parameters: PaintingParameters) -> str:
    """No second product when fine and reference grids coincide."""
    return PAINTING if parameters.snps_per_bin == parameters.reference_snps_per_bin else REFERENCE


def reference_derivation(fine_identity: str, *, numerical_policy: NumericalPolicy | None = None) -> str:
    """The reference decode shares every fine input and its published label-continuity outcome."""
    return Derivation(
        code_dependency_identity(("stages.painting.records", "stages.painting.tasks")),
        dict(product=REFERENCE), numerical_policy or NumericalPolicy.current(), dict(painting=fine_identity),
    ).identity


def progress(file: CheckpointFile) -> dict[str, Any]:
    """Report durable publication from the container index without decoding arrays.

    This is work status only; the result reader checks schema and compatibility
    with the independently bound request before exposing scientific products.
    """
    info = file.info(PAINTING)
    return dict(
        stage=PAINTING,
        complete=info is not None,
        **({} if info is None else dict(decoded_bytes=info.decoded_bytes)),
    )


class Adapter:
    """Product codec; cache buffers have their own publication and lifetime."""

    schema = SCHEMA

    def encode(self, product: Painting) -> dict[str, Any]:
        payload = painting_to_payload(product)
        payload.pop("packed_emissions")
        return payload

    def decode(self, payload: Mapping[str, Any], *, assembly: Assembly | None = None) -> Painting:
        return painting_from_payload(payload, assembly=assembly)


def write(
    file: CheckpointFile,
    identity: str,
    painting: Painting,
    summary: PaintingSummary,
    *,
    evidence_identity: str | None = None,
    key: str = PAINTING,
) -> None:
    """Publish cache and summary first, scientific product last as the commit record."""
    publish(
        file,
        key + "/cache",
        identity,
        {"schema": SCHEMA, "packed_emissions": [component.packed_emissions for component in painting.components]},
    )
    publish(file, key + "/summary", identity, to_payload(summary))
    publish(file, "identity/" + key, identity, {"schema": SCHEMA, "identity": identity, "evidence_identity": evidence_identity})
    publish(file, key, identity, Adapter().encode(painting))


def load(
    file: CheckpointFile, identity: str, *, assembly: Assembly | None = None, emissions: bool = True,
    key: str = PAINTING,
) -> Painting | None:
    """Read a matching product; assembly reconstructs panels, emissions adds the cache."""
    names = ("schema", "contig", "sample_ids", "parameters", "components", "label_continuity", "robustness_epsilon", "emission_divisor", "path_divisor")
    record = read_items(file, key, identity, names + (("rows",) if assembly is not None else ()))
    if record is None:
        return None
    payload = dict(record.payload)
    if emissions:
        cache = resume(file, key + "/cache", identity)
        if cache is not None:
            require_schema(cache.payload, SCHEMA)
            payload["packed_emissions"] = cache.payload["packed_emissions"]
    return Adapter().decode(payload, assembly=assembly)


@dataclass(frozen=True)
class TieSource:
    """The explicit painting group of the shared diagnostic report protocol."""

    record_group: str
    counters: tuple[TieCounts, ...]
    discovery: tuple[object, ...] = ()
    refinement_agreement: tuple[bool | None, ...] = ()
    count_calls: tuple[object, ...] = ()


@dataclass(frozen=True)
class PaintingReport:
    """Read-only report inputs, including each recorded final/feedback search once."""

    painting: Painting
    positions: np.ndarray
    observed: np.ndarray
    searches: tuple[tuple[int, int, int, JoinSummaryView], ...]
    feedback_searches: tuple[tuple[int, int, int, JoinSummaryView], ...]


@dataclass(frozen=True)
class Reader:
    """Results request with identity derived independently from the bound manifest."""

    identity: str
    product: str
    assembly: Assembly | None = None
    packed_emissions: tuple[np.ndarray | None, ...] | None = None
    positions: np.ndarray | None = None
    observed: np.ndarray | None = None
    searches: tuple[tuple[int, int, int, JoinSummaryView], ...] = ()
    feedback_searches: tuple[tuple[int, int, int, JoinSummaryView], ...] = ()
    key: str = PAINTING
    schema: int = SCHEMA

    @property
    def items(self) -> tuple[str, ...] | None:
        if self.product == "summary":
            return None
        names = ("schema", "contig", "sample_ids", "parameters", "components", "label_continuity", "robustness_epsilon", "emission_divisor", "path_divisor")
        return names + (("rows",) if self.assembly is not None else ())

    def decode(self, payload: Mapping[str, Any]) -> Painting | PaintingSummary | PaintingReport | tuple[TieSource, ...]:
        if self.product == "summary":
            return from_payload(PaintingSummary, payload)
        data = dict(payload)
        if self.packed_emissions is not None:
            data["packed_emissions"] = self.packed_emissions
        painting = painting_from_payload(data, assembly=self.assembly)
        if self.product == "painting_report":
            if self.positions is None or self.observed is None:
                raise InvariantError("painting report requires checked evidence arrays")
            return PaintingReport(painting, self.positions, self.observed, self.searches, self.feedback_searches)
        if self.product == "tie_sources":
            return (TieSource("painting", tuple(outcome.tie_events for outcome in painting.label_continuity)),)
        return painting


# One contig's bound upstream per process: (run directory, contig, record-reuse pin generation) -> (manifest,
# bound_upstream's value), and the lock its first binding holds. Published records never change, the manifest binds
# every setting the identities derive from, and the pin generation the code identities (store.identity). Every contig's painting identity includes the run calibration's, which derives from every analysed contig's
# assembly: unshared, each painting reader bound every contig again (contigs squared bindings, each a chain of header
# reads back to discovery, in every process that reads paintings: a pedigree step, a report, an evaluation).
_UPSTREAM: dict[tuple[str, str, int], tuple[Any, tuple[str, str, str, str, str]]] = {}
_UPSTREAM_LOCKS: dict[tuple[str, str, int], threading.Lock] = {}
_UPSTREAM_GUARD = threading.Lock()


def bound_upstream(
    store: CheckpointStore, manifest: Mapping[str, Any], contig: str
) -> tuple[str, str, str, str, str] | None:
    """(assembly key, assembly identity, evidence key, evidence identity, working-map identity) of one contig, from
    its checked final-assembly and evidence readers; None while either is unpublished (never remembered).

    A thread binding a contig holds that contig's lock, so concurrent readers of several contigs (threaded_each)
    bind each contig once: the binding reads only its own contig, hence no thread waits while holding a lock that
    another needs."""
    from ..assembly import records as assembly_records
    from ..evidence import records as evidence_records
    from ..relatedness.working_map import map_identity

    key = (os.path.realpath(store.root), contig, pin_generation())
    with _UPSTREAM_GUARD:
        lock = _UPSTREAM_LOCKS.setdefault(key, threading.Lock())
    with lock:
        cached = _UPSTREAM.get(key)
        if cached is not None and (cached[0] is manifest or cached[0] == manifest):
            return cached[1]
        assembled = assembly_records.result_reader(store, manifest, product="assembly", contig=contig)
        evidence = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
        if assembled is None or evidence is None:
            return None
        bound = (assembled.key, assembled.identity, evidence.key, evidence.identity, map_identity(store, manifest, contig))
        _UPSTREAM[key] = (manifest, bound)
        return bound


def run_binding(store: CheckpointStore) -> tuple[str, int, int]:
    """The per-process cache key of a run folder's derived identities: (run directory, its run container's inode,
    record-reuse pin generation). Published records never change and the pin generation fixes the code identities the
    derivations bind; the inode separates a run folder made again at the same path. Callers also check the manifest."""
    run = store.run.path
    return os.path.realpath(store.root), os.stat(run).st_ino if os.path.exists(run) else -1, pin_generation()


# Painting requests this process derived: run_binding + (contig, product) -> (manifest, Reader). The identity is a
# pure function of the manifest, the checked upstream headers (bound_upstream, and the assembly, evidence and run
# calibration publications, which never change once published) and the contig's bound payload; a failed check
# raises and is never remembered. The contamination decision, the refit priors and the run calibration bind every
# contig's painting, so a pedigree task derived these requests 138 times on TroMau (each re-reading headers).
_EXPECTED: dict[tuple, tuple[Any, Reader]] = {}
_EXPECTED_GUARD = threading.Lock()


def expected(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
) -> Reader | None:
    """Rebuild painting identity from checked assembly/evidence requests and run settings.

    Never read an expected identity from this product's own envelope. Physical
    cache presence is independent of the scientific identity and its schema. Derived once per process and run
    folder (_EXPECTED)."""
    key = (*run_binding(store), contig, product)
    with _EXPECTED_GUARD:
        cached = _EXPECTED.get(key)
    if cached is not None and (cached[0] is manifest or cached[0] == manifest):
        return cached[1]
    request = _derive_expected(store, manifest, product=product, contig=contig)
    if request is not None:
        with _EXPECTED_GUARD:
            _EXPECTED[key] = (manifest, request)
    return request


def _derive_expected(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
) -> Reader | None:
    """expected, derived afresh."""
    from ...science.painting.parameters import PaintingParameters

    if contig is None or contig not in manifest["contigs"]:
        raise InputError("painting reader requires a bound contig")
    if product not in ("painting", "reference", "summary", "tie_sources", "painting_report"):
        raise InputError(f"unsupported painting reader product {product!r}")
    file = store.contig(contig)
    bound = bound_upstream(store, manifest, contig)
    if bound is None:
        raise InvariantError("painting requires published assembly and evidence")
    assembly_key, assembly_identity, evidence_key, evidence_identity, working_map = bound
    # Identity checks read headers only; the assembly payload is decoded only when panels are wanted.
    checked_assembly = read_items(file, assembly_key, assembly_identity, ("schema",))
    checked_evidence = read_items(file, evidence_key, evidence_identity, ("schema",))
    if checked_assembly is None or checked_evidence is None:
        raise InvariantError("painting upstream publication is incomplete")
    parameters = from_payload(PaintingParameters, manifest["parameters"]["painting"])
    settings = manifest["ties"]
    from . import calibration_records
    calibrated = calibration_records.result_reader(store, manifest)
    if calibrated is None:
        raise InvariantError("painting requires run calibration; run haplotypes calibrate after all assemblies")
    expected_identity = derivation(
        parameters,
        contig=contig,
        sample_ids=tuple(manifest["sample_ids"]),
        assembly_identity=assembly_identity,
        evidence_identity=evidence_identity,
        payload_identity=bound_payload_identity(file),
        map_identity=working_map,
        calibration_identity=calibrated.identity,
        ties=TieSettings(reverse=bool(settings["reverse"]), keep=tuple(settings["keep"])),
        numerical_policy=NumericalPolicy(**manifest["numerical_policy"]),
    )
    key = SUMMARY if product == "summary" else PAINTING
    if product == "reference":
        key = reference_key(parameters)
        if key == REFERENCE:
            expected_identity = reference_derivation(
                expected_identity, numerical_policy=NumericalPolicy(**manifest["numerical_policy"])
            )
    return Reader(expected_identity, product, key=key)


def result_reader(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
    panels: bool = True,
    emissions: bool = True,
) -> Reader | None:
    """Attach requested panel/cache views to the shared expected publication."""
    from ..assembly import records as assembly_records
    from ..evidence import records as evidence_records

    file = store.contig(contig)
    if not file.has(PAINTING):
        return None
    request = expected(store, manifest, product=product, contig=contig)
    expected_identity = request.identity
    if not file.has(request.key):
        return None
    if product == "painting":
        # A fine publication may precede an interrupted reference decode. Workflow
        # receipts must not skip this contig until both required products exist.
        reference = expected(store, manifest, product="reference", contig=contig)
        if reference.key == REFERENCE and read_items(
            file, reference.key, reference.identity, ("schema",)
        ) is None:
            return None
    if product == "summary":
        return request
    assembly = None
    if (panels and product in ("painting", "reference")) or product == "painting_report":
        # expected checked the assembly and evidence headers; their readers are needed only for these views
        assembled = assembly_records.result_reader(store, manifest, product="assembly", contig=contig)
        assembly_record = resume(file, assembled.key, assembled.identity)
        if assembly_record is None:
            raise InvariantError("painting upstream publication is incomplete")
        value = cast(AssemblyProduct, assembled.decode(assembly_record.payload))
        assembly = (tuple(value.prepared), tuple(value.components))
    packed = None
    if emissions and product in ("painting", "reference"):
        cache = resume(file, request.key + "/cache", expected_identity)
        if cache is not None:
            require_schema(cache.payload, SCHEMA)
            packed = tuple(cache.payload["packed_emissions"])
    if product == "painting_report":
        evidence = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
        arrays = evidence_records.load_arrays(file, ("positions", "observed"), expected_identity=evidence.identity)
        arrays["observed"].flags.writeable = False
        histories: list[tuple[tuple[int, int, int, JoinSummaryView], ...]] = []
        for round_name in (None, "l1", "l2"):
            request = assembly_records.result_reader(
                store, manifest, product="join_searches", contig=contig, round_name=round_name
            )
            if request is None:
                if round_name is None:
                    raise InvariantError("painting report requires final assembly search history")
                histories.append(())
                continue
            request = cast(assembly_records.JoinSearchReader, request)
            saved = read_items(file, request.key, request.identity, request.items)
            if saved is None:
                raise InvariantError("painting report search history is incomplete")
            histories.append(cast(tuple[tuple[int, int, int, JoinSummaryView], ...], request.decode(saved.payload)))
        return Reader(
            expected_identity,
            product,
            assembly,
            positions=arrays["positions"],
            observed=arrays["observed"],
            searches=histories[0],
            feedback_searches=histories[1] + histories[2],
        )
    return Reader(expected_identity, product, assembly, packed, key=request.key)


__all__ += ["Reader", "TieSource", "PaintingReport", "result_reader"]


def require_evidence_binding(file: CheckpointFile, identity: str, evidence_identity: str) -> None:
    """Reject a painting made from another evidence product before conditional refit."""
    binding = resume(file, IDENTITY, identity)
    if binding is None or binding.payload.get("evidence_identity") != evidence_identity:
        raise InvariantError("painting is not bound to the current evidence identity")


__all__ += ["require_evidence_binding"]


__all__ += ["progress"]
