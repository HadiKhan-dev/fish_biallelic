"""Discovery records in a contig's checkpoint file (keys, encoding, completeness, readers).

``identity/discovery``       stage identity: DiscoveryConfig, the evidence record identity, code
``discovery/plan``           {blocks: number of blocks B of the evidence partition}
``discovery/blocks/<n:06d>`` chunk n (n = 0, 1, ... in write order) of up to CHUNK_BLOCKS
                             completed blocks in completion order: {indices int64[m], blocks:
                             [encoded DiscoveredBlock, or None for an uninformative block]}

A contig is complete when every block index 0 .. B-1 is in some chunk. No genotype likelihoods
are stored (blocks reference the evidence record by site span).

Release 2.3: search summaries also hold search_incomplete, omissions and unresolved (defaults decode older records).
Release 2.2: blocks also hold latent_unknown (founder alleles no assigned carrier observes) and the tied alternatives
(tied_panels, tied_panel_nll, tied_unknown; recorded stacked with their founder counts tied_panel_k); 2.1 payloads
decode with None / empty tuples.
Release 2.1: the encoded search summary also holds tie_quantum, escalated, conventions, ambiguous,
tied_alternatives, score_margin and tie_events (search.SearchSummary); 2.0 payloads without them decode with the
defaults. The SCHEMA stays 1 (the stage identity carries the code digest, so 2.1 never resumes 2.0 records).
"""
from __future__ import annotations

import numpy as np

from ..evidence import records as evidence_records
from ..store.identity import stage_identity
from .release import DiscoveredBlock
from .search import SearchSummary

STAGE = "discovery"
SCHEMA = 1
PLAN = "discovery/plan"
CHUNK_PREFIX = "discovery/blocks/"
# Completed blocks per chunk record (a kill loses at most the unwritten blocks of one chunk).
CHUNK_BLOCKS = 64


def chunk_key(ordinal: int) -> str:
    return f"{CHUNK_PREFIX}{ordinal:06d}"


def identity(config, evidence_identity: str) -> dict:
    """The discovery record identity of a contig."""
    return stage_identity(STAGE, schema=SCHEMA, config=config,
                          upstream={"evidence": evidence_identity})


def bind(store, file, config) -> None:
    """Bind the contig's discovery records to ``config`` and its evidence record."""
    store.bind_identity(file, STAGE, identity(config, file.info(evidence_records.EVIDENCE).identity))


def encode(block: DiscoveredBlock | None):
    if block is None:
        return None
    summary = block.search
    return dict(
        block=int(block.block), site_start=int(block.site_start), positions=block.positions,
        kept=block.kept, latent=block.latent, calls=block.calls,
        alt_probability=block.alt_probability, support=block.support,
        unknown_mass=float(block.unknown_mass), nll=float(block.nll),
        latent_unknown=block.latent_unknown,
        tied_panels=(np.concatenate(block.tied_panels) if block.tied_panels
                     else np.zeros((0, block.latent.shape[1]), dtype=np.int8)),
        tied_panel_k=np.asarray([len(panel) for panel in block.tied_panels], dtype=np.int32),
        tied_panel_nll=np.asarray(block.tied_panel_nll, dtype=np.float64),
        tied_unknown=(np.concatenate(block.tied_unknown) if block.tied_unknown
                      else np.zeros((0, block.latent.shape[1]), dtype=np.bool_)),
        search=dict(
            k_values=summary.k_values, k_log_scores=summary.k_log_scores,
            natural_k_ceiling=int(summary.natural_k_ceiling), expansions=int(summary.expansions),
            data_starts=int(summary.data_starts), exact_scores=int(summary.exact_scores),
            exact_score_cache_hits=int(summary.exact_score_cache_hits),
            residual_calls=int(summary.residual_calls), residual_rows=int(summary.residual_rows),
            limits=list(summary.limits), heuristic_gauge_cuts=bool(summary.heuristic_gauge_cuts),
            stop_reason=str(summary.stop_reason), certified=bool(summary.certified),
            tie_quantum=float(summary.tie_quantum), escalated=bool(summary.escalated),
            conventions=int(summary.conventions), ambiguous=bool(summary.ambiguous),
            tied_alternatives=int(summary.tied_alternatives),
            score_margin=float(summary.score_margin),
            search_incomplete=bool(summary.search_incomplete), unresolved=bool(summary.unresolved),
            omissions=[dict(item) for item in summary.omissions],
            tie_events={str(kind): {str(name): int(value) for name, value in counts.items()}
                        for kind, counts in summary.tie_events.items()}))


def decode(payload) -> DiscoveredBlock | None:
    """A recorded block (2.0 payloads without the 2.1 summary fields decode with their defaults)."""
    if payload is None:
        return None
    search = dict(payload["search"])
    search["limits"] = tuple(search["limits"])
    if "omissions" in search:
        search["omissions"] = tuple(dict(item) for item in search["omissions"])
    if "tie_events" in search:
        search["tie_events"] = {str(kind): {str(name): int(value) for name, value in dict(counts).items()}
                                for kind, counts in dict(search["tie_events"]).items()}
    fields = dict(payload)
    fields["search"] = SearchSummary(**search)
    counts = fields.pop("tied_panel_k", None)
    if counts is not None:
        bounds = np.concatenate(([0], np.cumsum(np.asarray(counts, dtype=np.int64))))
        stacked, unknown = fields.get("tied_panels"), fields.get("tied_unknown")
        fields["tied_panels"] = tuple(np.ascontiguousarray(stacked[a:b]) for a, b in zip(bounds[:-1], bounds[1:]))
        fields["tied_unknown"] = tuple(np.ascontiguousarray(unknown[a:b]) for a, b in zip(bounds[:-1], bounds[1:]))
        fields["tied_panel_nll"] = tuple(float(value) for value in fields.get("tied_panel_nll", ()))
    else:
        for name in ("tied_panels", "tied_panel_nll", "tied_unknown"):
            fields.pop(name, None)
    return DiscoveredBlock(**fields)


class ChunkWriter:
    """Completed blocks recorded CHUNK_BLOCKS at a time; chunk numbers stay contiguous across
    resumes and advance only after a successful put (discovery B1)."""

    def __init__(self, file):
        self.file = file
        self.next_ordinal = len(file.keys(CHUNK_PREFIX))
        self.pending: list = []

    def add(self, index: int, block: DiscoveredBlock | None) -> None:
        self.pending.append((int(index), block))
        if len(self.pending) == CHUNK_BLOCKS:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        self.file.put(chunk_key(self.next_ordinal), dict(
            indices=np.array([index for index, _ in self.pending], dtype=np.int64),
            blocks=[encode(block) for _, block in self.pending]), nthreads=1)
        self.next_ordinal += 1
        self.pending = []


def planned_blocks(file) -> int | None:
    return int(file.get(PLAN)["blocks"]) if file.has(PLAN) else None


def recorded_indices(file) -> set[int]:
    """Block indices already recorded (decodes only the chunks' index arrays)."""
    recorded: set[int] = set()
    for key in file.keys(CHUNK_PREFIX):
        recorded.update(file.get(key, items=("indices",))["indices"].tolist())
    return recorded


def complete(file) -> bool:
    """Every planned block of the contig is recorded."""
    planned = planned_blocks(file)
    return planned is not None and recorded_indices(file) == set(range(planned))


def progress(file) -> dict:
    planned = planned_blocks(file)
    return dict(blocks=planned, recorded=len(recorded_indices(file)) if planned is not None else 0)


def load(file, *, nthreads=None) -> tuple[DiscoveredBlock | None, ...]:
    """Every block's result by block index (None = uninformative); the contig must be complete."""
    planned = planned_blocks(file)
    by_index: dict[int, DiscoveredBlock | None] = {}
    for key in file.keys(CHUNK_PREFIX):
        chunk = file.get(key, nthreads=nthreads)
        for index, payload in zip(chunk["indices"].tolist(), chunk["blocks"]):
            by_index[index] = decode(payload)
    if planned is None or set(by_index) != set(range(planned)):
        raise RuntimeError(f"{file.label}: block discovery is incomplete "
                           f"({len(by_index)} of {planned} blocks recorded)")
    return tuple(by_index[index] for index in range(planned))


def founder_panels(file) -> tuple[tuple[np.ndarray, np.ndarray, int], ...]:
    """(positions int64[L], released calls int8[K, L], block) of every informative block, in
    block order (the evaluation reader of raw discovery)."""
    return tuple((block.positions, block.calls, block.block)
                 for block in load(file) if block is not None)
