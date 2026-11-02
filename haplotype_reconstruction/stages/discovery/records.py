"""Translate discovery blocks to their recorded field representation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ...contracts.discovery import DiscoveredBlock, DiscoveryBlockResult, DiscoveryPlan, DiscoverySearchStatus
from ...contracts.payload import from_payload, to_payload
from ...contracts.roles import require_schema
from ...core.errors import InputError, InvariantError
from ...core.ties import TieCounts, TieSettings
from ...science.discovery.diagnostics import DiscoveryDiagnostics
from ...science.discovery.parameters import DiscoveryParameters
from ...science.discovery.release import _search_status
from ...science.discovery.search import SearchSummary
from ...store.container import CheckpointFile, CheckpointMismatch
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, is_complete, publication_metadata, read_items, resume
from ...store.store import CheckpointStore


def encode(block: DiscoveredBlock | None, diagnostics: DiscoveryDiagnostics | None = None) -> dict[str, Any] | None:
    if block is None:
        return None
    summary = block.search
    return dict(
        block=int(block.block),
        site_start=int(block.site_start),
        positions=block.positions,
        kept=block.kept,
        latent=block.latent,
        calls=block.calls,
        alt_probability=block.alt_probability,
        support=block.support,
        unknown_mass=float(block.unknown_mass),
        nll=float(block.nll),
        latent_unknown=block.latent_unknown,
        tied_panels=(
            np.concatenate(block.tied_panels)
            if block.tied_panels
            else np.zeros((0, block.latent.shape[1]), dtype=np.int8)
        ),
        tied_panel_k=np.asarray([len(panel) for panel in block.tied_panels], dtype=np.int32),
        tied_panel_nll=np.asarray(block.tied_panel_nll, dtype=np.float64),
        tied_unknown=(
            np.concatenate(block.tied_unknown)
            if block.tied_unknown
            else np.zeros((0, block.latent.shape[1]), dtype=np.bool_)
        ),
        search=dict(
            k_values=summary.k_values,
            k_log_scores=summary.k_log_scores,
            natural_k_ceiling=int(summary.natural_k_ceiling),
            expansions=int(summary.expansions),
            data_starts=int(summary.data_starts),
            exact_scores=int(summary.exact_scores),
            exact_score_cache_hits=int(summary.exact_score_cache_hits),
            residual_calls=int(summary.residual_calls),
            residual_rows=int(summary.residual_rows),
            limits=list(summary.limits),
            heuristic_gauge_cuts=bool(summary.heuristic_gauge_cuts),
            stop_reason=str(summary.stop_reason),
            certified=bool(summary.certified),
            tie_quantum=float(summary.tie_quantum),
            escalated=False,
            conventions=1,
            ambiguous=bool(summary.ambiguous),
            tied_alternatives=int(summary.tied_alternatives),
            score_margin=float(summary.score_margin),
            search_incomplete=bool(summary.search_incomplete),
            unresolved=bool(summary.unresolved),
            omissions=[asdict(item) for item in summary.omissions],
            **diagnostic_fields(summary, diagnostics),
            tie_events=(
                {
                    item.kind: {name: getattr(item, name) for name in ("decisions", "inside_tie", "wide", "boundary")}
                    for item in diagnostics.tie_events.entries
                }
                if diagnostics is not None
                else {}
            ),
        ),
    )


def diagnostic_fields(summary: DiscoverySearchStatus, diagnostics: DiscoveryDiagnostics | None = None) -> dict:
    """Fixed-size measured fields; None/empty buckets mean unavailable in old data."""
    return dict(
        archive_max_k=summary.archive_max_k,
        wall_seconds=None if diagnostics is None else diagnostics.wall_seconds,
        cpu_seconds=None if diagnostics is None else diagnostics.cpu_seconds,
        phase_seconds={},
        selected_mean_field_failures=summary.selected_mean_field_failures,
        other_mean_field_failures=summary.other_mean_field_failures,
    )


def decode(payload: Mapping[str, Any] | None) -> DiscoveredBlock | None:
    """A recorded block (2.0 payloads without the 2.1 summary fields decode with their defaults)."""
    if payload is None:
        return None
    search = dict(payload["search"])
    search["limits"] = tuple(search["limits"])
    if "omissions" in search:
        search["omissions"] = tuple(dict(item) for item in search["omissions"])
    if "tie_events" in search:
        search["tie_events"] = {
            str(kind): {str(name): int(value) for name, value in dict(counts).items()}
            for kind, counts in dict(search["tie_events"]).items()
        }
    fields = dict(payload)
    fields["search"] = _search_status(SearchSummary(**search))
    counts = fields.pop("tied_panel_k", None)
    if counts is not None:
        bounds = np.concatenate(([0], np.cumsum(np.asarray(counts, dtype=np.int64))))
        stacked, unknown = fields["tied_panels"], fields["tied_unknown"]
        fields["tied_panels"] = tuple(
            np.ascontiguousarray(stacked[a:b]) for a, b in zip(bounds[:-1], bounds[1:], strict=False)
        )
        fields["tied_unknown"] = tuple(
            np.ascontiguousarray(unknown[a:b]) for a, b in zip(bounds[:-1], bounds[1:], strict=False)
        )
        fields["tied_panel_nll"] = tuple(float(value) for value in fields.get("tied_panel_nll", ()))
    else:
        for name in ("tied_panels", "tied_panel_nll", "tied_unknown"):
            fields.pop(name, None)
    return DiscoveredBlock(**fields)


STAGE = "discovery"
PLAN = "discovery/plan"
BLOCK_PREFIX = "discovery/blocks/"
DIAGNOSTIC_PREFIX = "discovery/diagnostics/"


def block_key(block: int) -> str:
    """The semantic per-block record key, independent of dispatch or completion order."""
    return f"{BLOCK_PREFIX}{block}"


def diagnostic_key(block: int) -> str:
    """The corresponding diagnostic record; never used as evidence of completion."""
    return f"{DIAGNOSTIC_PREFIX}{block}"


class BlockAdapter:
    """Versioned discriminated block completion, decoded into D1's frozen contract."""

    schema = 1

    def encode(self, product: DiscoveryBlockResult) -> dict:
        return to_payload(product)

    def decode(self, payload: Mapping[str, Any]) -> DiscoveryBlockResult:
        return from_payload(DiscoveryBlockResult, payload)


class PlanAdapter:
    """The ordered block domain, stored through its canonical contract."""

    schema = 1

    def encode(self, product: DiscoveryPlan) -> dict:
        from ...core.errors import InvariantError

        if type(product.blocks) is not int or product.blocks < 0:
            raise InvariantError("discovery block count must be a nonnegative integer")
        return to_payload(product)

    def decode(self, payload: Mapping[str, Any]) -> DiscoveryPlan:
        return from_payload(DiscoveryPlan, payload)


def identities(
    parameters: DiscoveryParameters,
    *,
    contig: str,
    blocks: int,
    evidence_identity: str,
    numerical_policy: NumericalPolicy,
    ties: TieSettings,
) -> dict[str, str]:
    """Bind the ordered block domain and each block to independent scientific inputs."""
    code = code_dependency_identity(("science.discovery", "stages.discovery.tasks", "stages.discovery.records"))

    def derive(unit: tuple[str, int]) -> str:
        return Derivation(
            code,
            dict(discovery=parameters, contig=contig, unit=unit),
            numerical_policy,
            {"evidence": evidence_identity},
            ties.reverse,
            ties.keep,
        ).identity

    return {PLAN: derive(("plan", blocks)), **{block_key(i): derive(("block", i)) for i in range(blocks)}}


@dataclass(frozen=True)
class TieSource:
    """The discovery portion of the diagnostic report protocol, in block order."""

    record_group: str
    counters: tuple[TieCounts, ...]
    discovery: tuple[DiscoverySearchStatus, ...]
    refinement_agreement: tuple[bool | None, ...] = ()
    count_calls: tuple[object, ...] = ()


@dataclass(frozen=True)
class Reader:
    """Read a verified complete block domain without trusting its completion marker."""

    file: CheckpointFile
    identities: Mapping[str, str]
    product: str
    blocks: int
    key: str = PLAN
    schema: int = 1
    items: tuple[str, ...] = ("schema", "blocks")

    @property
    def identity(self) -> str:
        return self.identities[PLAN]

    def decode(self, payload: Mapping[str, Any]) -> tuple:
        if PlanAdapter().decode(payload).blocks != self.blocks:
            raise InvariantError("discovery plan block count differs from evidence")
        products = []
        for index in range(self.blocks):
            key = block_key(index)
            saved = resume(self.file, key, self.identities[key])
            if saved is None:
                raise InvariantError("discovery block domain is incomplete")
            products.append(BlockAdapter().decode(saved.payload).result)
        if self.product == "stage_panels":
            return tuple(product for product in products if product is not None)
        if self.product != "tie_sources":
            return tuple(products)
        counters = []
        for index in range(self.blocks):
            saved = resume(self.file, diagnostic_key(index), self.identities[block_key(index)])
            if saved is not None:
                require_schema(saved.payload, 1)
                counters.append(from_payload(TieCounts, saved.payload["tie_events"]))
        return (TieSource("discovery/blocks", tuple(counters), tuple(p.search for p in products if p is not None)),)


# Discovery derivations whose every block this process has checked: (container path, its inode, plan identity).
# Published records never change (containers are append-only; identities fix their content), so later readers of the
# same derivation skip the per-block checks: every downstream product's reader rebuilds discovery's identity, and
# re-checking ~10^3 blocks per read made a stage's input reads take tens of seconds.
_CHECKED: set[tuple[str, int, str]] = set()


def result_reader(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
    stage: str | None = None,
) -> Reader | None:
    """Derive expected identities from the manifest and checked evidence publication.

    The evidence's small block-span array fixes the complete block domain. Every
    block header is checked before returning a request, including None results.
    Neither stored discovery identities nor a completion marker supply an input.
    """
    from ..evidence import records as evidence_records

    if contig is None or contig not in manifest["contigs"]:
        raise InputError("discovery reader requires a bound contig")
    if product not in ("discovery", "discovered", "stage_panels", "tie_sources"):
        raise InputError(f"unsupported discovery reader product {product!r}")
    if stage not in (None, "discovery"):
        raise InputError(f"unsupported discovery stage {stage!r}")
    file = store.contig(contig)
    if not file.has(PLAN):
        return None
    request = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
    if request is None:
        raise InvariantError("discovery requires published evidence")
    saved = read_items(file, request.key, request.identity, ("schema", "block_spans"))
    if saved is None:
        raise InvariantError("discovery evidence publication is incomplete")
    require_schema(saved.payload, request.schema)
    count = len(saved.payload["block_spans"])
    expected = identities(
        from_payload(DiscoveryParameters, manifest["parameters"]["discovery"]),
        contig=contig,
        blocks=count,
        evidence_identity=request.identity,
        numerical_policy=NumericalPolicy(**manifest["numerical_policy"]),
        ties=TieSettings(reverse=bool(manifest["ties"]["reverse"]), keep=tuple(manifest["ties"].get("keep", ()))),
    )
    checked = (file.path, os.stat(file.path).st_ino, expected[PLAN])
    if checked in _CHECKED:
        return Reader(file, expected, product, count)
    if not is_complete(file, expected):
        return None
    plan = read_items(file, PLAN, expected[PLAN], ("schema", "blocks"))
    if plan is None or PlanAdapter().decode(plan.payload).blocks != count:
        raise InvariantError("discovery plan block count differs from evidence")
    for index in range(count):
        block = read_items(file, block_key(index), expected[block_key(index)], ("schema", "block", "kind"))
        if block is None:
            return None
        require_schema(block.payload, BlockAdapter.schema)
        if block.payload["block"] != index or block.payload["kind"] not in ("discovered", "uninformative"):
            raise InvariantError("discovery block completion has invalid metadata")
    _CHECKED.add(checked)
    return Reader(file, expected, product, count)


def progress(file: CheckpointFile) -> dict[str, Any]:
    """Report checked small metadata; compatibility with settings belongs to readers."""
    records = []
    for key in (PLAN, *file.keys(BLOCK_PREFIX)):
        names = ("schema", "blocks") if key == PLAN else ("schema", "block", "kind")
        # The checked status boundary: one committed snapshot, publication header validated, arrays refused.
        try:
            metadata = publication_metadata(file, key, names)
        except CheckpointMismatch as error:
            raise InvariantError("malformed discovery publication header") from error
        if metadata is None:
            continue
        payload = dict(metadata.fields)
        require_schema(payload, 1)
        records.append(dict(key=key, derivation=metadata.recorded_derivation, **payload))
    return dict(published=tuple(records), completed_blocks=sum(row["key"] != PLAN for row in records))
