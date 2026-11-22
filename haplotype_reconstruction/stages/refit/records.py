"""Store schema-3 founder refits and reconstruct panels from painting views.

The envelope contains the versioned record codec product. Derived panels never enter storage;
reads check the caller's expected derivation, never a physical record identifier.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any
from weakref import WeakKeyDictionary

from ...contracts.contigs import AnalysisContigs
from ...contracts.painting import Painting
from ...contracts.payload import from_payload, to_payload
from ...contracts.refit import FounderRefit
from ...contracts.roles import require_schema
from ...core.concurrency import threaded_each
from ...core.errors import InputError, InvariantError
from ...core.ties import TieSettings
from ...science.refit.parameters import RefitParameters
from ...science.refit.run import RefitSummary
from ...store import identity as source_identity
from ...store.container import CheckpointFile
from ...store.protocol import NumericalPolicy, bound_payload_identity, publish, read_items, resume
from ...store.store import CheckpointStore

__all__ = ["FOUNDER_REFIT", "SUMMARY", "IDENTITY", "HISTOGRAM", "PRIORS", "SCHEMA", "Adapter", "write", "load"]
FOUNDER_REFIT = "founder_refit"
SUMMARY = "founder_refit/summary"
IDENTITY = "identity/founder_refit"
HISTOGRAM = "founder_refit/histogram"  # per contig: the start-state gain histogram by carrier class
PRIORS = "founder_refit/priors"  # run: the class priors pooled over every analysed contig (N0run, ADVICE2 16.2)
SCHEMA = 4  # 4.21: run-level class priors (G7' froze N0run)
_PRODUCT_KEYS = {
    "refit": FOUNDER_REFIT,
    "summary": SUMMARY,
    "tie_sources": FOUNDER_REFIT,
    "histogram": HISTOGRAM,
    "priors": PRIORS,
}


from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation


def derivation(
    parameters: RefitParameters,
    *,
    contig: str,
    sample_ids: tuple[str, ...],
    painting_identity: str,
    evidence_identity: str,
    payload_identity: str,
    ties: TieSettings = TieSettings(),
    numerical_policy: NumericalPolicy | None = None,
    priors_identity: str = "",
    product: str = "refit",
) -> str:
    """A contig's refit (bound to the run's priors) or, with ``product`` = histogram, its gain histogram."""
    upstream = dict(painting=painting_identity, evidence=evidence_identity, payload=payload_identity)
    if product == "refit":
        upstream["priors"] = priors_identity
    return Derivation(
        code_dependency_identity(("science.refit", "stages.refit.tasks", "stages.refit.records")),
        dict(parameters=parameters, contig=contig, sample_ids=sample_ids, product=product),
        numerical_policy or NumericalPolicy.current(),
        upstream,
        tie_reverse=ties.reverse,
        tie_keep=ties.keep,
        schema=SCHEMA,
    ).identity


def priors_derivation(parameters: RefitParameters, histograms: Mapping[str, str], *, ties: TieSettings = TieSettings(),
                      numerical_policy: NumericalPolicy | None = None) -> str:
    """The run's class priors over the analysed contigs' histogram identities."""
    return Derivation(
        code_dependency_identity(("science.refit", "stages.refit.tasks", "stages.refit.records")),
        dict(parameters=parameters, product="priors"),
        numerical_policy or NumericalPolicy.current(),
        dict(histograms),
        tie_reverse=ties.reverse,
        tie_keep=ties.keep,
        schema=SCHEMA,
    ).identity


def progress(file: CheckpointFile) -> dict[str, Any]:
    """Report durable publication from the container index without decoding arrays.

    This is work status only; the result reader checks schema and compatibility
    with the independently bound request before exposing scientific products.
    """
    info = file.info(FOUNDER_REFIT)
    return dict(
        stage=FOUNDER_REFIT,
        complete=info is not None,
        **({} if info is None else dict(decoded_bytes=info.decoded_bytes)),
    )


class Adapter:
    schema = SCHEMA

    def encode(self, product: FounderRefit) -> dict[str, Any]:
        return {"schema": SCHEMA, "refit": to_payload(product)}

    def decode(self, payload: Mapping[str, Any], *, painting: Painting | None = None) -> FounderRefit:
        require_schema(payload, SCHEMA)
        if set(payload) != {"schema", "refit"}:
            raise InvariantError("unexpected founder-refit envelope fields")
        result = from_payload(FounderRefit, payload["refit"])
        if painting is None:
            return result
        if result.contig != painting.contig or result.sample_ids != painting.sample_ids:
            raise InvariantError("refit and painting contig/sample axes differ")
        if len(result.components) != len(painting.components):
            raise InvariantError("refit and painting component counts differ")
        components = []
        for value, painted in zip(result.components, painting.components, strict=True):
            if painted.panel is None:
                raise InvariantError("refit reconstruction requires painted panels")
            if value.released.shape != painted.panel.released.shape:
                raise InvariantError("refit and painted panel axes differ")
            components.append(
                replace(value, panel=replace(painted.panel, released=value.released, inference=value.inference))
            )
        return replace(result, components=tuple(components))


def write(file: CheckpointFile, identity: str, product: FounderRefit, summary: RefitSummary) -> None:
    publish(file, SUMMARY, identity, {"schema": SCHEMA, **to_payload(summary)})
    publish(file, IDENTITY, identity, {"schema": SCHEMA, "identity": identity})
    publish(file, FOUNDER_REFIT, identity, Adapter().encode(product))


def load(file: CheckpointFile, identity: str, *, painting: Painting | None = None) -> FounderRefit | None:
    record = resume(file, FOUNDER_REFIT, identity)
    return None if record is None else Adapter().decode(record.payload, painting=painting)


@dataclass(frozen=True)
class Reader:
    """Manifest-derived request, with independently checked painted panels."""

    identity: str
    product: str
    painting: Painting | None = None
    key: str = FOUNDER_REFIT
    schema: int = SCHEMA

    def decode(self, payload: Mapping[str, Any]) -> FounderRefit | RefitSummary | tuple[()]:
        if self.product == "tie_sources":
            # founder refit has no shared tie census (its carrier-specific relation is a local rule).
            require_schema(payload, SCHEMA)
            return ()
        if self.product == "summary":
            require_schema(payload, SCHEMA)
            return from_payload(RefitSummary, {key: value for key, value in payload.items() if key != "schema"})
        return Adapter().decode(payload, painting=self.painting)


def _settings(manifest):
    parameters = from_payload(RefitParameters, manifest["parameters"]["founder_refit"])
    settings = manifest["ties"]
    ties = TieSettings(reverse=bool(settings["reverse"]), keep=tuple(settings["keep"]))
    return parameters, ties, NumericalPolicy(**manifest["numerical_policy"])


@dataclass
class _Bindings:
    """Checked dependencies for one store, manifest and fixed code snapshot."""

    manifest: Mapping[str, Any]
    pins: int
    priors: str | None = None
    priors_lock: Lock = field(default_factory=Lock)
    upstream: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    contig_locks: dict[str, Lock] = field(default_factory=dict)


_BINDINGS: WeakKeyDictionary[CheckpointStore, _Bindings] = WeakKeyDictionary()
_BINDINGS_LOCK = Lock()


def _bindings(store: CheckpointStore, manifest: Mapping[str, Any]) -> _Bindings | None:
    # A standalone reader follows editable source changes. Engine workers fix the source snapshot and
    # validate immutable upstream publications once per store. Reuse pins can change between invocations.
    if not source_identity._SOURCES_FIXED:
        return None
    pins = source_identity.pin_generation()
    with _BINDINGS_LOCK:
        bound = _BINDINGS.get(store)
        if bound is None or bound.pins != pins or bound.manifest != manifest:
            bound = _BINDINGS[store] = _Bindings(deepcopy(dict(manifest)), pins)
        return bound


def _derive_priors(store, manifest) -> str:
    parameters, ties, policy = _settings(manifest)
    contigs = AnalysisContigs.from_manifest(manifest).included
    identities = threaded_each(lambda c: expected(store, manifest, product="histogram", contig=c).identity, contigs)
    histograms = dict(zip(contigs, identities, strict=True))
    return priors_derivation(parameters, histograms, ties=ties, numerical_policy=policy)


def _priors_identity(store, manifest) -> str:
    """Bind the common priors once across the contigs that request refit."""
    # On TroMau, 44 cold decision-binding requests used to decode 34,848 upstream records here.
    # The contig checks run independently. The run identity needs all their checked histogram identities.
    bound = _bindings(store, manifest)
    if bound is None:
        return _derive_priors(store, manifest)
    with bound.priors_lock:
        if bound.priors is None:
            bound.priors = _derive_priors(store, manifest)
        return bound.priors


def _checked_upstream(store, manifest, contig: str) -> tuple[str, str, str]:
    from ..evidence import records as evidence_records
    from ..painting import records as painting_records

    file = store.contig(contig)
    # Identity construction checks headers; result_reader attaches panels only when requested.
    painted = painting_records.result_reader(
        store, manifest, product="painting", contig=contig, panels=False, emissions=False
    )
    evidence = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
    if painted is None or evidence is None:
        raise InvariantError("refit requires published painting and evidence")
    painting_record = read_items(file, painted.key, painted.identity, ("schema",))
    checked_evidence = read_items(file, evidence.key, evidence.identity, ("schema",))
    if painting_record is None or checked_evidence is None:
        raise InvariantError("refit upstream publication is incomplete")
    painting_records.require_evidence_binding(file, painted.identity, evidence.identity)
    return painted.identity, evidence.identity, bound_payload_identity(file)


def _upstream(store, manifest, contig: str) -> tuple[str, str, str]:
    bound = _bindings(store, manifest)
    if bound is None:
        return _checked_upstream(store, manifest, contig)
    with _BINDINGS_LOCK:
        lock = bound.contig_locks.get(contig)
        if lock is None:
            lock = bound.contig_locks[contig] = Lock()
    # The histogram and refit share these checks. Fetching a contig never acquires the priors lock.
    with lock:
        checked = bound.upstream.get(contig)
        if checked is None:
            checked = bound.upstream[contig] = _checked_upstream(store, manifest, contig)
        return checked


def expected(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
) -> Reader:
    """Derive the refit request from manifest settings and checked upstream products: a contig's refit, summary,
    tie sources or gain histogram, or (contig None) the run's priors."""
    if product == "priors":
        return Reader(_priors_identity(store, manifest), product, key=PRIORS)
    if contig is None or contig not in manifest["contigs"]:
        raise InputError("refit reader requires a bound contig")
    if product not in _PRODUCT_KEYS:
        raise InputError(f"unsupported refit reader product {product!r}")
    painting_identity, evidence_identity, payload_identity = _upstream(store, manifest, contig)
    parameters, ties, policy = _settings(manifest)
    expected_identity = derivation(
        parameters,
        contig=contig,
        sample_ids=tuple(manifest["sample_ids"]),
        painting_identity=painting_identity,
        evidence_identity=evidence_identity,
        payload_identity=payload_identity,
        ties=ties,
        numerical_policy=policy,
        priors_identity="" if product == "histogram" else _priors_identity(store, manifest),
        product="histogram" if product == "histogram" else "refit",
    )
    return Reader(expected_identity, product, key=_PRODUCT_KEYS[product])


def result_reader(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
    panels: bool = True,
) -> Reader | None:
    """Request the published product, attaching painted panels to a complete refit when requested."""
    if product not in _PRODUCT_KEYS:
        raise InputError(f"unsupported refit reader product {product!r}")
    file = store.run if product == "priors" else store.contig(contig)
    if not file.has(_PRODUCT_KEYS[product]):
        return None
    request = expected(store, manifest, product=product, contig=contig)
    if not panels or product != "refit":
        return request

    from ..painting import records as painting_records

    painted = painting_records.result_reader(store, manifest, product="painting", contig=contig,
                                             panels=True, emissions=False)
    saved = read_items(file, painted.key, painted.identity, painted.items)
    if saved is None:
        raise InvariantError("refit upstream publication is incomplete")
    return replace(request, painting=painted.decode(saved.payload))


__all__ += ["Reader", "result_reader"]


__all__ += ["progress"]
