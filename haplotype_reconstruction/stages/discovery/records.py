"""Translate discovery blocks to their recorded field representation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...contracts.discovery import DiscoveryBlockResult, DiscoveryPlan, DiscoverySearchStatus
from ...contracts.payload import from_payload, to_payload
from ...contracts.roles import require_schema
from ...core.errors import InputError, InvariantError
from ...core.ties import TieCounts, TieSettings
from ...science.discovery.parameters import DiscoveryParameters
from ...store.container import CheckpointFile
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, read_items, resume
from ...store.store import CheckpointStore

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
    """Versioned discriminated block completion, decoded into its frozen record contract."""

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
    """Bind the ordered block domain and each block to independent scientific inputs.

    Cache the result per process, keyed by every derivation input, so downstream readers do not
    repeatedly traverse thousands of block identities. Shared parts are converted to plain JSON data
    once for the key; ``jsonable`` returns that data unchanged."""
    from ...core.digest import json_digest, jsonable

    code = jsonable(
        code_dependency_identity(("science.discovery", "stages.discovery.tasks", "stages.discovery.records"))
    )
    plain = jsonable(parameters)
    policy = jsonable(numerical_policy)
    key = (json_digest([code, plain, policy]), contig, int(blocks), evidence_identity, ties.reverse, ties.keep)
    cached = _IDENTITIES.get(key)
    if cached is not None:
        return dict(cached)

    def derive(unit: tuple[str, int]) -> str:
        return Derivation(
            code,
            dict(discovery=plain, contig=contig, unit=unit),
            policy,
            {"evidence": evidence_identity},
            ties.reverse,
            ties.keep,
        ).identity

    result = {PLAN: derive(("plan", blocks)), **{block_key(i): derive(("block", i)) for i in range(blocks)}}
    _IDENTITIES[key] = result
    return dict(result)


# Discovery identities per process, keyed by everything they derive from.
_IDENTITIES: dict[tuple, dict[str, str]] = {}


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
        with self.file.batch() as reader:  # one index snapshot and handle for every block record
            for index in range(self.blocks):
                key = block_key(index)
                saved = resume(self.file, key, self.identities[key], reader=reader)
                if saved is None:
                    raise InvariantError("discovery block domain is incomplete")
                products.append(BlockAdapter().decode(saved.payload).result)
        if self.product == "stage_panels":
            return tuple(product for product in products if product is not None)
        if self.product != "tie_sources":
            return tuple(products)
        counters = []
        with self.file.batch() as reader:
            for index in range(self.blocks):
                saved = resume(self.file, diagnostic_key(index), self.identities[block_key(index)], reader=reader)
                if saved is not None:
                    require_schema(saved.payload, 1)
                    counters.append(from_payload(TieCounts, saved.payload["tie_events"]))
        return (TieSource("discovery/blocks", tuple(counters), tuple(p.search for p in products if p is not None)),)


# Discovery derivations whose every block this process has checked: (container path, its inode, plan identity).
# Published records never change (containers are append-only; identities fix their content), so later readers of the
# same derivation skip the per-block checks: every downstream product's reader rebuilds discovery's identity, and
# re-checking ~10^3 blocks per read made a stage's input reads take tens of seconds.
_CHECKED: set[tuple[str, int, str]] = set()


def expected(
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
        ties=TieSettings(reverse=bool(manifest["ties"]["reverse"]), keep=tuple(manifest["ties"]["keep"])),
    )
    return Reader(file, expected, product, count)


def result_reader(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
    stage: str | None = None,
) -> Reader | None:
    """Check the expected plan and every block before exposing discovery."""
    if contig is None or not store.contig(contig).has(PLAN):
        return None
    request = expected(store, manifest, product=product, contig=contig, stage=stage)
    file, count = store.contig(contig), request.blocks
    expected_identities = request.identities
    checked = (file.path, os.stat(file.path).st_ino, expected_identities[PLAN])
    if checked in _CHECKED:
        return Reader(file, expected_identities, product, count)
    # One pass over the plan and every block: each record's envelope is checked once, with its small metadata. Every
    # record is read before deciding, so a present record whose identity differs raises CheckpointMismatch even when
    # another is missing; a missing record leaves the domain incomplete. Use one index snapshot and
    # one handle (CheckpointFile.batch); per block, an index refresh and an open cost a network filesystem far more
    # than the small decode (chr3 at 0.5x: 7,533 blocks, read again by every process that checks the domain).
    with file.batch() as reader:
        plan = read_items(file, PLAN, expected_identities[PLAN], ("schema", "blocks"), reader=reader)
        blocks = [
            read_items(file, block_key(i), expected_identities[block_key(i)], ("schema", "block", "kind"), reader=reader)
            for i in range(count)
        ]
    present = [block for block in blocks if block is not None]
    if plan is None or len(present) < count:
        return None
    if PlanAdapter().decode(plan.payload).blocks != count:
        raise InvariantError("discovery plan block count differs from evidence")
    for index, block in enumerate(present):
        require_schema(block.payload, BlockAdapter.schema)
        if block.payload["block"] != index or block.payload["kind"] not in ("discovered", "uninformative"):
            raise InvariantError("discovery block completion has invalid metadata")
    _CHECKED.add(checked)
    return Reader(file, expected_identities, product, count)


def progress(file: CheckpointFile) -> dict[str, Any]:
    """Count committed block headers without opening every block's payload metadata.

    Schema and scientific compatibility are checked by result_reader, not status.
    """
    keys = file.keys(BLOCK_PREFIX)
    return dict(published=keys, completed_blocks=len(keys), plan_published=file.has(PLAN))
