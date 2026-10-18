"""Local-selection records in a contig's checkpoint file.

Per round R in ROUNDS, under ``local_selection/R/`` (a RecordScope bound to the round identity):
``blocks/<n:06d>``  chunk n (write order) of up to CHUNK_BLOCKS completed blocks in completion
                    order: {indices int64[m], panels, outcomes, releases, summaries} (encoded
                    LocalPanel, SearchOutcome (fit + search), PathRelease, summary dict)
``memos/<n:06d>``   the same blocks' likelihood memos for the next round (rounds "initial",
                    "l1"; written before the chunk, so a recorded chunk always has its memos)
``proposals``       rounds "l1", "l2": {proposals: encoded ContextProposal per block, summary}
``selected``        round completion {chunks: count, blocks: number of informative blocks}
``assembly/...``    rounds "l1", "l2": the context assembly's records (assembly's API)

Identities (``identity/local_selection/R``): the initial round binds the local-selection and
discovery configs, the copying generations, the genetic map and the evidence and discovery
record identities, with the code of local_selection + discovery only (assembly changes leave
the initial fits valid, as base); the context rounds add the assembly config and the previous
round's identity record, with the assembly code too.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ..model.local_panels import LocalPanel
from ..store import RecordScope
from ..store.identity import stage_identity
from .fit import PathFit
from .release import PathRelease
from .search import RoundTrace, SearchOutcome

STAGE = "local_selection"
SCHEMA = 1
ROUNDS = ("initial", "l1", "l2")
CHUNK_BLOCKS = 64


def round_prefix(round_name: str) -> str:
    return f"{STAGE}/{round_name}"


def context_assembly_prefix(round_name: str) -> str:
    """Record prefix of a context round's assembly (assembly's own records, its API)."""
    return f"{round_prefix(round_name)}/assembly"


def round_identity(round_name, config, discovery_config, generations, genetic_map, upstream,
                   assembly_config=None) -> dict:
    """The identity of one round (see the module docstring)."""
    settings = dict(round=round_name, local_selection=config, discovery=discovery_config,
                    copying_generations=float(generations), genetic_map=genetic_map.identity())
    if round_name == ROUNDS[0]:
        return stage_identity(STAGE, schema=SCHEMA, config=settings, upstream=upstream,
                              packages=("local_selection", "discovery"))
    settings["assembly"] = assembly_config
    return stage_identity(STAGE, schema=SCHEMA, config=settings, upstream=upstream)


def round_scope(file, round_name) -> RecordScope:
    # Small per-block writes stay single-threaded while numerical workers run.
    return RecordScope(file, round_prefix(round_name), nthreads=1)


# ---------------------------------------------------------------------------------------------
# Encoding (plain dicts of arrays and scalars)
# ---------------------------------------------------------------------------------------------
def encode_panel(panel: LocalPanel) -> dict:
    return asdict(panel)


def decode_panel(payload) -> LocalPanel:
    return LocalPanel(**payload)


def encode_outcome(outcome: SearchOutcome) -> dict:
    fit = outcome.fit
    return dict(fit=asdict(fit), selection_objective=float(outcome.selection_objective),
                trace=[asdict(item) for item in outcome.trace], search_start=outcome.search_start,
                round_backstop=bool(outcome.round_backstop),
                context_founders=outcome.context_founders, free_rows=int(outcome.free_rows),
                input_key=outcome.input_key, work=list(outcome.work),
                start_was_incumbent_panel=bool(outcome.start_was_incumbent_panel))


def decode_outcome(payload) -> SearchOutcome:
    return SearchOutcome(
        fit=PathFit(**payload["fit"]), selection_objective=payload["selection_objective"],
        trace=tuple(RoundTrace(**dict(item, proposals=tuple(map(tuple, item["proposals"]))))
                    for item in payload["trace"]),
        search_start=payload["search_start"], round_backstop=payload["round_backstop"],
        context_founders=payload["context_founders"], free_rows=payload["free_rows"],
        input_key=payload["input_key"], work=tuple(map(tuple, payload["work"])),
        start_was_incumbent_panel=bool(payload.get("start_was_incumbent_panel", False)))


def encode_release(release: PathRelease) -> dict:
    return asdict(release)


def decode_release(payload) -> PathRelease:
    return PathRelease(**payload)


class SelectionChunks:
    """A round's completed blocks, recorded CHUNK_BLOCKS at a time; chunk numbers are contiguous
    across resumes and advance only after a successful put."""

    def __init__(self, scope: RecordScope):
        self.scope = scope
        self.ordinal = len(scope.keys("blocks/"))
        self.pending: list = []

    def load(self, memos: bool = True) -> list:
        """The recorded BlockSelections of every chunk (memo None unless ``memos``)."""
        from .rounds import BlockSelection
        selections = []
        for ordinal in range(self.ordinal):
            chunk = self.scope.load(f"blocks/{ordinal:06d}")
            saved = self.scope.load(f"memos/{ordinal:06d}") if memos else None
            memo_by_index = {} if saved is None else dict(zip(saved["indices"].tolist(), saved["memos"]))
            for position, index in enumerate(chunk["indices"].tolist()):
                selections.append(BlockSelection(
                    index=index, panel=decode_panel(chunk["panels"][position]),
                    outcome=decode_outcome(chunk["outcomes"][position]),
                    release=decode_release(chunk["releases"][position]),
                    memo=memo_by_index.get(index), summary=chunk["summaries"][position]))
        return selections

    def add(self, selection) -> None:
        self.pending.append(selection)
        if len(self.pending) == CHUNK_BLOCKS:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        indices = np.array([selection.index for selection in self.pending], dtype=np.int64)
        name = f"{self.ordinal:06d}"
        if any(selection.memo is not None for selection in self.pending):
            self.scope.save(f"memos/{name}", dict(
                indices=indices, memos=[selection.memo for selection in self.pending]))
        self.scope.save(f"blocks/{name}", dict(
            indices=indices, panels=[encode_panel(selection.panel) for selection in self.pending],
            outcomes=[encode_outcome(selection.outcome) for selection in self.pending],
            releases=[encode_release(selection.release) for selection in self.pending],
            summaries=[selection.summary for selection in self.pending]))
        self.ordinal += 1
        self.pending = []

    def complete(self, blocks: int) -> None:
        if not self.scope.has("selected"):
            self.scope.save("selected", dict(chunks=self.ordinal, blocks=int(blocks)))


def encode_proposals(proposals, summary) -> dict:
    return dict(proposals=[asdict(proposal) for proposal in proposals], summary=asdict(summary))


def decode_proposals(payload):
    from .context_refit import ContextProposal
    return [ContextProposal(**item) for item in payload["proposals"]]


# ---------------------------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------------------------
def round_complete(file, round_name) -> bool:
    return file.has(f"{round_prefix(round_name)}/selected")


def complete(file) -> bool:
    """The contig's last round is recorded."""
    return round_complete(file, ROUNDS[-1])


def progress(file) -> dict:
    return dict(rounds=[name for name in ROUNDS if round_complete(file, name)])


def panels_identity(file, round_name=ROUNDS[-1]) -> str:
    """The record identity of the round's completion record (``selected``): the upstream identity
    of whatever reads the round's panels (the final assembly binds it)."""
    return file.info(f"{round_prefix(round_name)}/selected").identity


def _completed(file, round_name):
    scope = round_scope(file, round_name)
    selected = scope.load("selected")
    if selected is None:
        raise RuntimeError(f"{file.label}: local-selection round {round_name} is incomplete")
    return scope, selected


def _check_count(file, round_name, recorded, selected):
    if recorded != selected["blocks"]:
        raise RuntimeError(f"{file.label}: round {round_name} records {recorded} of "
                           f"{selected['blocks']} blocks")


def load_selections(file, round_name, memos: bool = True) -> list:
    """The BlockSelections of a completed round, in block order (memo None unless ``memos``)."""
    scope, selected = _completed(file, round_name)
    selections = SelectionChunks(scope).load(memos)
    _check_count(file, round_name, len(selections), selected)
    return sorted(selections, key=lambda selection: selection.index)


def load_panels(file, round_name=ROUNDS[-1]) -> tuple[LocalPanel, ...]:
    """The LocalPanels of a completed round (default the last), in block order. Decodes only the
    chunks' panels (not the fits, releases or memos)."""
    scope, selected = _completed(file, round_name)
    panels = []
    for ordinal in range(len(scope.keys("blocks/"))):
        chunk = file.get(scope.key(f"blocks/{ordinal:06d}"), nthreads=scope.nthreads,
                         items=("indices", "panels"))
        panels.extend(zip(chunk["indices"].tolist(), chunk["panels"]))
    _check_count(file, round_name, len(panels), selected)
    panels.sort(key=lambda item: item[0])
    return tuple(decode_panel(payload) for _, payload in panels)


def founder_panels(file, round_name) -> tuple[tuple[np.ndarray, np.ndarray, int], ...]:
    """(positions int64[L], released calls int8[K, L], block) of a round's selected panels, in
    block order (the evaluation reader)."""
    return tuple((panel.positions, panel.calls, panel.block)
                 for panel in load_panels(file, round_name))
