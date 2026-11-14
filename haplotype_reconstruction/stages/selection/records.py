"""Publish each selection block and its memo under a deterministic round derivation.

Owns selection/block, memo, proposals and selected adapters (LS30). Memos are state:
initial/l1 memo records are published before their block, and missing memo state
cannot silently become a cold search. Completion verifies all promised records.
Must not fit, choose candidates or use physical container tokens as input identity.
Method: docs/methods/local_selection.md#records.14.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any

from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.panels import LocalPanel
from ...contracts.payload import from_payload, to_payload
from ...contracts.selection import (
    ContextProposal,
    PathRelease,
    RefitSummary,
    SearchOutcome,
    SelectionMemo,
    SelectionSummary,
)
from ...core.digest import json_digest
from ...core.errors import InvariantError
from ...science.assembly.parameters import AssemblyParameters
from ...science.discovery.parameters import DiscoveryParameters
from ...science.selection.parameters import SelectionParameters
from ...science.selection.rounds import BlockSelection
from ...store.container import CheckpointFile
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, is_complete, publish, read_items, resume
from ...store.store import CheckpointStore

__all__ = [
    "ROUNDS",
    "SelectionRecords",
    "round_identity",
    "encode_panel",
    "decode_panel",
    "encode_outcome",
    "decode_outcome",
    "encode_release",
    "decode_release",
    "encode_proposals",
    "decode_proposals",
    "context_assembly_prefix",
    "result_reader",
    "progress",
]

if TYPE_CHECKING:
    from ...science.selection.context_refit import RefitFinished


ROUNDS = ("initial", "l1", "l2")
SCHEMA = 1


def context_assembly_prefix(round_name: str) -> str:
    """Return assembly's context scope; its role and level enter its own derivation."""
    return f"local_selection/{round_name}/assembly"


def round_identity(
    round_name: str,
    parameters: SelectionParameters,
    discovery_parameters: DiscoveryParameters,
    genetic_map: ChromosomeGeneticMap | str,
    upstream: dict[str, str],
    assembly_parameters: AssemblyParameters | Mapping[str, Any] | None = None,
    *,
    reverse: bool = False,
    tie_keep: tuple[str, ...] = (),
    numerical_policy: NumericalPolicy | None = None,
) -> str:
    """Bind model inputs and code, excluding timing, routing and physical record IDs.

    Upstream values must already be semantic derivation/payload identities. Context
    assembly parameters must be the CONTEXT projection, with refinement removed.
    Arrays/map shape failures and identity encoding errors propagate.
    """
    roots = [
        "haplotype_reconstruction.science.selection.rounds",
        "haplotype_reconstruction.science.selection.run",
        "haplotype_reconstruction.stages.selection.records",
        "haplotype_reconstruction.stages.selection.tasks",
    ]
    settings = dict(
        round=round_name,
        selection=parameters,
        discovery=discovery_parameters,
        genetic_map=(genetic_map if isinstance(genetic_map, str) else json_digest(genetic_map.identity())),
    )
    if round_name != "initial":
        assert assembly_parameters is not None
        settings["assembly"] = (
            {
                item.name: getattr(assembly_parameters, item.name)
                for item in fields(assembly_parameters)
                if item.name != "refinement"
            }
            if is_dataclass(assembly_parameters)
            else {key: value for key, value in assembly_parameters.items() if key != "refinement"}
        )
        roots.append("haplotype_reconstruction.science.selection.context_refit")
    return Derivation(
        code_dependency_identity(roots),
        settings,
        numerical_policy or NumericalPolicy.current(),
        upstream,
        tie_reverse=reverse,
        tie_keep=tie_keep,
    ).identity


def encode_panel(panel: LocalPanel) -> dict:
    return to_payload(panel)


def decode_panel(payload: dict) -> LocalPanel:
    return from_payload(LocalPanel, payload)


def encode_outcome(outcome: SearchOutcome) -> dict:
    return to_payload(outcome)


def decode_outcome(payload: dict) -> SearchOutcome:
    return from_payload(SearchOutcome, payload)


def encode_release(release: PathRelease) -> dict:
    return to_payload(release)


def decode_release(payload: dict) -> PathRelease:
    return from_payload(PathRelease, payload)


def encode_proposals(proposals: Sequence[ContextProposal], summary: RefitSummary) -> dict:
    return dict(schema=SCHEMA, proposals=[to_payload(item) for item in proposals], summary=to_payload(summary))


def decode_proposals(payload: dict) -> tuple[ContextProposal, ...]:
    if payload["schema"] != SCHEMA:
        raise InvariantError(f"selection proposals schema {payload['schema']} != {SCHEMA}")
    return tuple(from_payload(ContextProposal, item) for item in payload["proposals"])


@dataclass(frozen=True)
class SelectionRecords:
    """One contig-round's explicit record capability and semantic derivation.

    file is a writable contig container. Identity checks happen before decoding;
    publications are append-only and idempotent under the store protocol.
    """

    file: CheckpointFile
    contig: str
    round_name: str
    identity: str

    def key(self, kind: str, index: int | None = None) -> str:
        unit = self.contig if index is None else f"{self.contig}:{index:06d}"
        return f"selection/{kind}/{unit}/{self.round_name}"

    def derivation(self, kind: str, index: int | None = None) -> str:
        return json_digest(dict(round=self.identity, record=kind, index=index, schema=SCHEMA))

    def load_refit(self, index: int) -> RefitFinished | None:
        """Restore one context component's exact worker reply under the round binding."""
        saved = resume(self.file, self.key("refit", index), self.derivation("refit", index))
        if saved is None:
            return None
        value = saved.payload
        if value["schema"] != SCHEMA:
            raise InvariantError("selection context refit schema differs")
        changes, fills, sites, founders = value["counts"]
        return (value["index"], [from_payload(ContextProposal, item) for item in value["proposals"]],
                changes, fills, sites, founders, value["report"])

    def save_refit(self, result: RefitFinished) -> None:
        """Publish only after the whole context refit component has finished."""
        index, proposals, changes, fills, sites, founders, report = result
        payload = dict(schema=SCHEMA, index=index, proposals=[to_payload(item) for item in proposals],
                       counts=(changes, fills, sites, founders), report=report)
        publish(self.file, self.key("refit", index), self.derivation("refit", index), payload)

    def load_proposals(self) -> tuple[ContextProposal, ...] | None:
        saved = resume(self.file, self.key("proposals"), self.derivation("proposals"))
        return None if saved is None else decode_proposals(saved.payload)

    def save_proposals(self, proposals: Sequence[ContextProposal], summary: RefitSummary) -> None:
        publish(self.file, self.key("proposals"), self.derivation("proposals"), encode_proposals(proposals, summary))

    def save_block(self, selection: BlockSelection) -> None:
        index = selection.index
        if self.round_name != "l2":
            if selection.memo is None:
                raise InvariantError(f"selection {self.contig}/{self.round_name}/{index} has no required memo")
            publish(
                self.file,
                self.key("memo", index),
                self.derivation("memo", index),
                to_payload(selection.memo),
            )
        payload = dict(
            schema=SCHEMA,
            index=index,
            panel=encode_panel(selection.panel),
            outcome=encode_outcome(selection.outcome),
            release=encode_release(selection.release),
            summary=to_payload(selection.summary),
        )
        publish(self.file, self.key("block", index), self.derivation("block", index), payload)

    def load_memo(self, index: int) -> SelectionMemo:
        """Read one prior block's exact state at its consumer, without the other blocks' memos."""
        stored = resume(self.file, self.key("memo", index), self.derivation("memo", index))
        if stored is None:
            raise InvariantError(f"selection block {index} exists without its {self.round_name} memo")
        return from_payload(SelectionMemo, stored.payload)

    def load_block(self, index: int, *, memos: bool = True) -> BlockSelection | None:
        saved = resume(self.file, self.key("block", index), self.derivation("block", index))
        if saved is None:
            return None
        payload = saved.payload
        if payload["schema"] != SCHEMA:
            raise InvariantError(f"selection block schema {payload['schema']} != {SCHEMA}")
        memo = None
        if memos and self.round_name != "l2":
            memo = self.load_memo(index)
        return BlockSelection(
            index=index,
            panel=decode_panel(payload["panel"]),
            outcome=decode_outcome(payload["outcome"]),
            release=decode_release(payload["release"]),
            memo=memo,
            summary=from_payload(SelectionSummary, payload["summary"]),
        )

    def complete(self, indices: tuple[int, ...]) -> None:
        expected = {self.key("block", index): self.derivation("block", index) for index in indices}
        if self.round_name != "l2":
            expected.update({self.key("memo", index): self.derivation("memo", index) for index in indices})
        if self.round_name != "initial":
            expected[self.key("proposals")] = self.derivation("proposals")
        if expected and not is_complete(self.file, expected):
            raise InvariantError(f"selection {self.contig}/{self.round_name} has missing records")
        publish(
            self.file,
            self.key("selected"),
            self.derivation("selected"),
            dict(schema=SCHEMA, indices=indices, records=expected),
        )

    def selected(self, *, verify: bool = True) -> dict[str, Any] | None:
        """The round's completion payload once every record it promises is present with its identity; ``verify=False``
        skips that record-by-record check, for readers that check each record's identity as they read it."""
        selected = resume(self.file, self.key("selected"), self.derivation("selected"))
        if selected is None:
            return None
        payload = selected.payload
        if payload["schema"] != SCHEMA:
            raise InvariantError("selection completion schema mismatch")
        expected = {self.key("block", int(i)): self.derivation("block", int(i)) for i in payload["indices"]}
        if self.round_name != "l2":
            expected.update({self.key("memo", int(i)): self.derivation("memo", int(i)) for i in payload["indices"]})
        if self.round_name != "initial":
            expected[self.key("proposals")] = self.derivation("proposals")
        if payload["records"] != expected:
            raise InvariantError("selection completion promises a different derivation")
        if verify and payload["records"] and not is_complete(self.file, payload["records"]):
            raise InvariantError(f"selection {self.contig}/{self.round_name} completion has missing records")
        return payload

    def load_selections(self, *, memos: bool = True) -> tuple[BlockSelection, ...]:
        """The completed round's block selections (with their memos unless ``memos=False``), each checked against its
        record identity as it is read. A projection: records the completion promises but this load does not read (the
        proposals; the memos when not loaded) are not checked here; selected() checks the whole completion."""
        selected = self.selected(verify=False)
        if selected is None:
            raise InvariantError(f"selection {self.contig}/{self.round_name} is incomplete")
        values = []
        for index in selected["indices"]:
            value = self.load_block(int(index), memos=memos)
            if value is None:
                raise InvariantError(f"selection {self.contig}/{self.round_name} completion has missing records")
            values.append(value)
        return tuple(values)

    def load_panels(self) -> tuple[LocalPanel, ...]:
        """Decode only panels from completed records: no fits, releases or memo arrays. A projection: each block record
        read is checked against its identity; the completion's other records are not (selected() checks them)."""
        selected = self.selected(verify=False)  # each panel read checks its record's identity
        if selected is None:
            raise InvariantError(f"selection {self.contig}/{self.round_name} is incomplete")
        result = []
        with self.file.batch() as reader:  # one index snapshot and handle for the round's block records
            for index in selected["indices"]:
                key, identity = self.key("block", index), self.derivation("block", index)
                saved = read_items(self.file, key, identity, ("panel",), reader=reader)
                if saved is None:
                    raise InvariantError(f"selection {self.contig}/{self.round_name} completion has missing records")
                result.append(decode_panel(saved.payload["panel"]))
        return tuple(result)


@dataclass(frozen=True)
class TieSource:
    """Selection's explicit group in the shared diagnostic reporting protocol."""

    record_group: str
    counters: tuple
    discovery: tuple = ()
    refinement_agreement: tuple = ()
    count_calls: tuple = ()


@dataclass(frozen=True)
class Reader:
    """A checked selected-record request whose decoder reads only its promised view."""

    scopes: tuple[SelectionRecords, ...]
    round_index: int
    product: str
    schema: int = SCHEMA
    context_scopes: tuple[Any, ...] = ()

    @property
    def key(self) -> str:
        return self.scopes[self.round_index].key("selected")

    @property
    def identity(self) -> str:
        return self.scopes[self.round_index].derivation("selected")

    def decode(self, payload: dict[str, Any]) -> Any:
        from ...contracts.roles import require_schema
        from ...contracts.selection import RefitSummary

        require_schema(payload, SCHEMA)
        scope = self.scopes[self.round_index]
        if self.product in ("selected", "selection", "stage_panels"):
            return scope.load_panels()
        if self.product == "refit_summary":
            return tuple(
                from_payload(RefitSummary, saved.payload["summary"])
                for item in self.scopes[1:]
                if (saved := read_items(item.file, item.key("proposals"), item.derivation("proposals"), ("summary",)))
                is not None
            )
        if self.product == "join_searches":
            from ..assembly.records import context_join_searches

            return tuple(row for context in self.context_scopes for row in context_join_searches(context))
        sources = []
        for item in self.scopes:
            selected = item.selected()
            if selected is None:
                continue
            summaries = []
            with item.file.batch() as reader:
                for index in selected["indices"]:
                    key, identity = item.key("block", index), item.derivation("block", index)
                    saved = read_items(item.file, key, identity, ("summary",), reader=reader)
                    assert saved is not None
                    summaries.append(from_payload(SelectionSummary, saved.payload["summary"]))
            # The record groups the ties report reads: a round's blocks, then its proposals (refit rounds).
            blocks = tuple(summary.tie_events for summary in summaries)
            sources.append(TieSource(f"local_selection/{item.round_name}/blocks", blocks))
            if item.round_name != "initial":
                saved = read_items(item.file, item.key("proposals"), item.derivation("proposals"), ("summary",))
                assert saved is not None
                summary = from_payload(RefitSummary, saved.payload["summary"])
                if summary.tie_events is not None:
                    sources.append(TieSource(f"local_selection/{item.round_name}/proposals", (summary.tie_events,)))
        from ..assembly.records import context_tie_sources

        for context in self.context_scopes:
            sources.extend(
                TieSource(item.record_group, item.counters, item.discovery, item.refinement_agreement, item.count_calls)
                for item in context_tie_sources(context)
            )
        return tuple(sources)


def expected(
    store: CheckpointStore,
    manifest: Mapping[str, Any],
    *,
    product: str,
    contig: str | None = None,
    stage: str | None = None,
    upstream: Mapping[str, str] | None = None,
) -> Reader | None:
    """Derive selection identities from bound parameters and independently checked upstreams.

    graph/workflow may supply already verified evidence/discovery identities. Otherwise their
    stage readers verify them from the same manifest. No selection envelope supplies
    an expected identity. Finished panel views do not decode outcomes or memo fits.
    """
    from ...contracts.roles import require_schema
    from ...core.errors import InputError
    from ...science.assembly.parameters import AssemblyParameters
    from ...science.discovery.parameters import DiscoveryParameters
    from ...science.selection.parameters import SelectionParameters
    from ..discovery import records as discovery_records
    from ..evidence import records as evidence_records
    from ..relatedness.working_map import map_identity

    if contig is None or contig not in manifest["contigs"]:
        raise InputError("selection reader requires a bound contig")
    if product not in ("selected", "selection", "stage_panels", "tie_sources", "refit_summary", "join_searches"):
        raise InputError(f"unsupported selection reader product {product!r}")
    name = "l2" if stage is None else stage.removeprefix("local_selection_")
    if name not in ROUNDS:
        raise InputError(f"unsupported selection round {name!r}")
    file = store.contig(contig)
    if upstream is None:
        evidence = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
        discovery = discovery_records.result_reader(store, manifest, product="discovery", contig=contig)
        if evidence is None or discovery is None:
            raise InvariantError("selection requires published evidence and discovery")
        for request in (evidence, discovery):
            saved = read_items(file, request.key, request.identity, ("schema",))
            if saved is None:
                raise InvariantError("selection upstream publication is incomplete")
            require_schema(saved.payload, request.schema)
        upstream = dict(evidence=evidence.identity, discovery=discovery.identity)
    parameters = from_payload(SelectionParameters, manifest["parameters"]["local_selection"])
    discovery_parameters = from_payload(DiscoveryParameters, manifest["parameters"]["discovery"])
    assembly_parameters = from_payload(AssemblyParameters, manifest["parameters"]["assembly"])
    settings = manifest["ties"]
    genetic_map = map_identity(store, manifest, contig)
    identities, scopes = dict(upstream), []
    for round_name in ROUNDS:
        expected = round_identity(
            round_name,
            parameters,
            discovery_parameters,
            genetic_map,
            identities,
            assembly_parameters if round_name != "initial" else None,
            reverse=bool(settings["reverse"]),
            tie_keep=tuple(settings["keep"]),
            numerical_policy=NumericalPolicy(**manifest["numerical_policy"]),
        )
        scopes.append(SelectionRecords(file, contig, round_name, expected))
        identities = dict(identities, previous_round=expected)
    context_scopes = []
    if product in ("tie_sources", "join_searches"):
        from ...store.protocol import bound_payload_identity
        from ..assembly.records import COMPLETION
        from ..assembly import records as assembly_records

        for index, round_name in enumerate(ROUNDS[1:], start=1):
            prefix = context_assembly_prefix(round_name)
            if not file.has(prefix + "/" + COMPLETION):
                continue
            context_scopes.append(assembly_records.expected(
                store, manifest, contig=contig, round_name=round_name, levels=parameters.context_levels[index - 1]
            ))
    return Reader(tuple(scopes), ROUNDS.index(name), product, context_scopes=tuple(context_scopes))


def result_reader(store, manifest, *, product, contig=None, stage=None, upstream=None):
    """Expose a selected publication through the same expected round scopes."""
    name = "l2" if stage is None else stage.removeprefix("local_selection_")
    if contig is None or not store.contig(contig).has(f"selection/selected/{contig}/{name}"):
        return None
    return expected(store, manifest, product=product, contig=contig, stage=stage, upstream=upstream)


def progress(file: CheckpointFile) -> dict[str, Any]:
    """Index-only publication progress, without opening any scientific payload.

    These committed-key counts describe work present on disk. Compatibility with
    the requested settings is established separately by result_reader.
    """
    keys = tuple(sorted(file.keys("selection/")))
    rounds = {key.rsplit("/", 1)[1] for key in keys if key.startswith("selection/selected/")}
    return dict(
        published=keys,
        rounds=tuple(name for name in ROUNDS if name in rounds),
        blocks=sum(key.startswith("selection/block/") for key in keys),
        memos=sum(key.startswith("selection/memo/") for key in keys),
    )
