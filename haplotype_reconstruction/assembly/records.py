"""Assembly records in a contig's checkpoint file.

Every assembly of a contig writes through a ``store.RecordScope`` under its section prefix P (the final
assembly: ``assembly``; a local-selection context assembly: the scope its caller gives, e.g.
``local_selection/l1/assembly``):

``identity/P``                 the assembly's identity (bound once, before anything is written)
``P/completion``               prepared panels + completion summary
``P/l<k>/plan``                level k at its start: batches [first input, end input, digest|None]
``joins/<digest>``             one joined batch (shared by every assembly of the contig), keyed by
                               the SHA-256 of its explicit inputs: rows over prepared panels,
                               panel log likelihood, search summary
``P/l<k>/join``                level k's hierarchy output: components (row paths) + level record
``P/final``                    the stage product (final role): the last level's joined components,
                               with the count calls from each component's latest join

Prepared panels are decoded from ``P/completion``. Records are plain dicts of numpy arrays and
builtins (encoded and decoded here), not pickled classes. Readers for other stages and the
evaluator: :func:`load_product`, :func:`founder_panels`, :func:`complete`, :func:`progress`.
"""
from __future__ import annotations

import numpy as np

from ..model.components import CountCall, CountStepDecision, FounderComponent, PreparedPanel
from ..store.codec import FAST_CLEVEL

FINAL_PREFIX = "assembly"     # the final assembly's section; context sections are given by their caller
JOINS = "joins/"              # contig-wide joined batches, shared by every assembly of the contig
COMPLETION = "completion"
FINAL = "final"
SCHEMA = 3


def level_key(level: int) -> str:
    return f"l{level}/join"


def load_joins(file, digests) -> dict:
    """{digest: join record} of the given batch digests that some assembly of the contig recorded."""
    present = set(file.keys(JOINS))
    return {digest: file.get(JOINS + digest) for digest in digests if JOINS + digest in present}


def save_join(file, digest: str, payload) -> None:
    """Record one joined batch (by the coordinating process, as it completes)."""
    file.put(JOINS + digest, payload, nthreads=1, clevel=FAST_CLEVEL)


def complete(file, prefix: str = FINAL_PREFIX) -> bool:
    """True when the contig's final assembly product is recorded."""
    return file.has(f"{prefix}/{FINAL}")


def progress(file, prefix: str = FINAL_PREFIX) -> dict:
    """Small status of an assembly section (record keys only, nothing decoded)."""
    names = [key[len(prefix) + 1:] for key in file.keys(prefix + "/")]
    levels = sorted({int(name.split("/")[0][1:]) for name in names
                     if name.startswith("l") and name.endswith("/join")})
    return dict(completion=COMPLETION in names, levels=levels, final=FINAL in names)



# ----------------------------------------------------------------------------- prepared panels
def encode_prepared(prepared) -> dict:
    """Prepared panels as a few concatenated arrays (one record, no per-block objects)."""
    return dict(
        schema=SCHEMA,
        block=np.asarray([p.block for p in prepared], np.int64),
        site_start=np.asarray([p.site_start for p in prepared], np.int64),
        sites=np.asarray([p.sites for p in prepared], np.int64),
        rows=np.asarray([p.rows for p in prepared], np.int64),
        positions=np.concatenate([p.positions for p in prepared]),
        kept=np.concatenate([p.kept for p in prepared]),
        released=np.concatenate([p.released.ravel() for p in prepared]),
        inference=np.concatenate([p.inference.ravel() for p in prepared]),
        alt_probability=np.concatenate([p.alt_probability.ravel() for p in prepared]),
        support=np.concatenate([p.support.ravel() for p in prepared]))


def decode_prepared(record) -> tuple[PreparedPanel, ...]:
    """Inverse of :func:`encode_prepared` (read-only views of the record's arrays)."""
    for name in ("positions", "kept", "released", "inference", "alt_probability", "support"):
        record[name].setflags(write=False)
    site = cell = 0
    panels = []
    for block, start, sites, rows in zip(record["block"].tolist(), record["site_start"].tolist(),
                                         record["sites"].tolist(), record["rows"].tolist()):
        cells = rows * sites
        def matrix(name):
            return record[name][cell:cell + cells].reshape(rows, sites)
        panels.append(PreparedPanel(
            block=block, site_start=start, positions=record["positions"][site:site + sites],
            kept=record["kept"][site:site + sites], released=matrix("released"),
            inference=matrix("inference"), alt_probability=matrix("alt_probability"),
            support=matrix("support")))
        site += sites
        cell += cells
    return tuple(panels)


# ----------------------------------------------------------------------------- components
def _encode_count_call(call: CountCall | None):
    if call is None:
        return None
    return dict(founders=call.founders, founders_before=call.founders_before,
                ambiguous=call.ambiguous, count_cycle=call.count_cycle,
                cycle_undecided=call.cycle_undecided, reason=call.reason,
                decisions=[dict(kind=d.kind, founders=d.founders, accept_larger=d.accept_larger,
                                skipped_by_bound=d.skipped_by_bound, margin=d.margin,
                                band=None if d.band is None else list(d.band),
                                ambiguous=d.ambiguous) for d in call.decisions])


def _decode_count_call(value) -> CountCall | None:
    if value is None:
        return None
    decisions = tuple(CountStepDecision(
        kind=d["kind"], founders=d["founders"], accept_larger=d["accept_larger"],
        skipped_by_bound=d["skipped_by_bound"], margin=d["margin"],
        band=None if d["band"] is None else tuple(d["band"]), ambiguous=d["ambiguous"])
        for d in value["decisions"])
    return CountCall(founders=value["founders"], founders_before=value["founders_before"],
                     ambiguous=value["ambiguous"], count_cycle=value["count_cycle"],
                     cycle_undecided=value["cycle_undecided"], decisions=decisions,
                     reason=value["reason"])


def encode_components(components) -> dict:
    """Founder components as concatenated row paths plus small per-component metadata."""
    return dict(
        schema=SCHEMA,
        start=np.asarray([c.start for c in components], np.int64),
        stop=np.asarray([c.stop for c in components], np.int64),
        founders=np.asarray([c.founders for c in components], np.int64),
        rows=np.concatenate([np.asarray(c.rows, np.int32).ravel() for c in components]),
        break_before=[c.break_before for c in components],
        break_after=[c.break_after for c in components],
        count_call=[_encode_count_call(c.count_call) for c in components])


def decode_components(record) -> tuple[FounderComponent, ...]:
    """Inverse of :func:`encode_components`."""
    components = []
    cell = 0
    for index, (start, stop, founders) in enumerate(zip(
            record["start"].tolist(), record["stop"].tolist(), record["founders"].tolist())):
        cells = founders * (stop - start)
        rows = record["rows"][cell:cell + cells].reshape(founders, stop - start)
        components.append(FounderComponent(
            start=start, stop=stop, rows=rows, break_before=record["break_before"][index],
            break_after=record["break_after"][index],
            count_call=_decode_count_call(record["count_call"][index])))
        cell += cells
    return tuple(components)


# ----------------------------------------------------------------------------- products
def load_product(file, prefix: str = FINAL_PREFIX):
    """(prepared panels, final components) of a contig's assembly, or None when it is absent.

    The consumers' entry (painting, export, evaluation): the components of ``P/final`` over the
    prepared panels of ``P/completion``.
    """
    if not file.has(f"{prefix}/{FINAL}"):
        return None
    prepared = decode_prepared(file.get(f"{prefix}/{COMPLETION}")["prepared"])
    components = decode_components(file.get(f"{prefix}/{FINAL}")["components"])
    return prepared, components


def founder_panels(file, level: int | None = None, prefix: str = FINAL_PREFIX):
    """One contig's founder panels of an assembly product, as ComponentPanels (positions, released
    calls, founder labels 0..K-1 = row indices), or None while the product is absent.

    level k: that level's joined product (``assembly_l<k>``); None: the final product (``assembly``).
    """
    from ..model.components import materialise
    if level is None:
        key = FINAL
    else:
        key = level_key(level)
    if not file.has(f"{prefix}/{key}") or not file.has(f"{prefix}/{COMPLETION}"):
        return None
    prepared = decode_prepared(file.get(f"{prefix}/{COMPLETION}", items=("prepared",))["prepared"])
    components = decode_components(file.get(f"{prefix}/{key}", items=("components",))["components"])
    return materialise(components, prepared)


def join_searches(file, prefix: str = FINAL_PREFIX) -> tuple:
    """Every panel search of an assembly section, for reports (assembly_search.tsv): one
    ``(level, first_position, last_position, JoinSummary)`` per joined batch, in level order, then
    batch order. The span is the first and last bp position of the batch's output component (base
    panel_search position_start / position_end). Each search is listed once, at the level it ran
    (base also re-listed a pass-through component's earlier search under every later level)."""
    from .stage import LevelRecord
    if not file.has(f"{prefix}/{COMPLETION}"):
        return ()
    prepared = file.get(f"{prefix}/{COMPLETION}", items=("prepared",))["prepared"]
    block_end = np.cumsum(prepared["sites"])
    block_first = block_end - prepared["sites"]
    positions = prepared["positions"]
    result = []
    level = 1
    while file.has(f"{prefix}/{level_key(level)}"):
        stored = file.get(f"{prefix}/{level_key(level)}", items=("components", "record"))
        starts = stored["components"]["start"].tolist()
        stops = stored["components"]["stop"].tolist()
        record = LevelRecord.from_record(stored["record"])
        joins = iter(record.joins)
        output = 0              # a joined batch emits one component, a pass-through batch its inputs
        for first, end, digest in record.plan:
            if digest is None:
                output += end - first
                continue
            start, stop = starts[output], stops[output]
            result.append((level, int(positions[block_first[start]]),
                           int(positions[block_end[stop - 1] - 1]), next(joins)))
            output += 1
        level += 1
    return tuple(result)


def level_records(file, prefix: str = FINAL_PREFIX) -> tuple:
    """The executed levels of an assembly section as ``stage.LevelRecord`` (plan, join summaries in
    the order of the joined batches, log likelihoods, stop reason), with each
    joined batch's span recoverable from the plan and the level's input components; for reports
    (assembly_search.tsv, founder_counts.tsv join and feedback columns)."""
    from .stage import LevelRecord
    result = []
    level = 1
    while file.has(f"{prefix}/{level_key(level)}"):
        record = LevelRecord.from_record(file.get(f"{prefix}/{level_key(level)}", items=("record",))["record"])
        result.append(record)
        level += 1
    return tuple(result)
