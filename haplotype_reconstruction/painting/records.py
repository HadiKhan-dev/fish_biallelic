"""Painting records in a contig's checkpoint file.

``identity/painting``  stage identity: painting config + identities of the assembly and
                       evidence records painted + the contig's genetic map (bound once)
``painting/summary``   small per-component shapes and label-continuity totals (status,
                       pedigree memory planning) -- written before the product
``painting``           the ChromosomePainting (presence = contig complete)

Records are plain dicts of numpy arrays and builtins. The painted founder panels are
stored as row paths over the assembly's prepared panels (label continuity permutes rows
span by span) and are materialised on load from the assembly records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from ..assembly import records as assembly_records
from ..evidence import records as evidence_records
from ..model.components import ComponentPanel, materialise
from ..store.identity import bound_identity, json_digest, stage_identity as build_identity
from .config import PaintingConfig
from .release import ChromosomePainting, ComponentPainting, LabelContinuity, RelabelMove, TrackStatus
from .states import PaintingBins, PaintingStates
from .transitions import PaintingTransitions

STAGE = "painting"
KEY = "painting"
SUMMARY_KEY = "painting/summary"
# 2: release 2.2 switch intervals (TrackStatus.SWITCH_INTERVAL, candidate-set posteriors).
SCHEMA = 2
EVIDENCE_KEY = evidence_records.EVIDENCE
ASSEMBLY_KEY = f"{assembly_records.FINAL_PREFIX}/final"


@dataclass(frozen=True)
class ComponentShape:
    """Sizes of one painted component: founders K, classes C, states S = C + 1,
    selected sites, bins B, and whether packed emissions are stored."""

    founders: int
    classes: int
    states: int
    sites: int
    bins: int
    packed_emissions: bool


@dataclass(frozen=True)
class PaintingSummary:
    """Record ``painting/summary``: shapes, eligible sample-component pairs and label continuity;
    released switch intervals (runs of SWITCH_INTERVAL bins per sample track) and their track bins."""

    contig: str
    samples: int
    components: tuple[ComponentShape, ...]
    eligible_pairs: int
    relabel_moves: int
    switches_before: int
    switches_after: int
    switch_intervals: int = 0
    interval_track_bins: int = 0


def stage_identity(file, config: PaintingConfig, genetic_map) -> dict:
    """The painting identity of a contig: config, upstream record identities, genetic map."""
    return build_identity(STAGE, schema=SCHEMA, config=config, upstream={
        "assembly": file.info(ASSEMBLY_KEY).identity,
        "evidence": file.info(EVIDENCE_KEY).identity,
        "genetic_map": json_digest(genetic_map.identity()),
    })


def bind_identity(store, file, config: PaintingConfig, genetic_map) -> None:
    """Bind ``identity/painting`` of the contig (an error if a different one is bound)."""
    store.bind_identity(file, STAGE, stage_identity(file, config, genetic_map))


def require_evidence_binding(file) -> None:
    """Raise unless the contig's painting was computed from its current evidence record.

    The new form of base's release-array identity check (pedigree/components.py
    _validate_release_array_identity, used by refit and family phase): the evidence
    record's identity is compared with the one the painting bound, without hashing arrays.
    """
    bound = bound_identity(file, STAGE)
    current = file.info(EVIDENCE_KEY)
    if bound is None or current is None or bound["upstream"]["evidence"] != current.identity:
        raise ValueError(f"{file.label}: the painting was not computed from this evidence record")


def complete(file) -> bool:
    return file.has(KEY)


def progress(file) -> dict:
    """Small status record (no product decoding)."""
    if not file.has(SUMMARY_KEY):
        return {"stage": STAGE, "complete": False}
    summary = load_summary(file)
    return {"stage": STAGE, "complete": file.has(KEY), "components": len(summary.components),
            "eligible_pairs": summary.eligible_pairs, "relabel_moves": summary.relabel_moves,
            "switches_before": summary.switches_before, "switches_after": summary.switches_after,
            "switch_intervals": summary.switch_intervals, "interval_track_bins": summary.interval_track_bins}


# ------------------------------------------------------------------------------- encoding
def _encode_component(component: ComponentPainting) -> dict:
    states, bins, transitions = component.states, component.bins, component.transitions
    return {
        "site_indices": states.site_indices, "positions": states.positions,
        "named_alleles": states.named_alleles,
        "classes": [list(members) for members in states.classes],
        "background_frequency": states.background_frequency,
        "bin_starts": bins.starts, "bin_stops": bins.stops, "bin_centers": bins.centers,
        "bin_edges": bins.edges, "log_same": transitions.log_same,
        "log_one": transitions.log_one, "log_two": transitions.log_two,
        "local_keys": component.local_keys, "eligible": component.eligible,
        "labels": component.labels, "track_classes": component.classes,
        "status": component.status, "direct": component.direct,
        "member_ids": component.member_ids,
        "member_sets": [list(members) for members in component.member_sets],
        "switch_counts": component.switch_counts,
    }


def _encode_outcome(outcome: LabelContinuity) -> dict:
    record = asdict(outcome)
    record["moves"] = [asdict(move) for move in outcome.moves]
    return record


def encode(painting: ChromosomePainting) -> dict:
    return {
        "schema": SCHEMA,
        "contig": painting.contig,
        "sample_ids": list(painting.sample_ids),
        "config": asdict(painting.config),
        "rows": [np.ascontiguousarray(component.panel.rows, dtype=np.int32)
                 for component in painting.components],
        "components": [_encode_component(component) for component in painting.components],
        "packed_emissions": [component.packed_emissions for component in painting.components],
        "label_continuity": [_encode_outcome(outcome) for outcome in painting.label_continuity],
    }


def interval_counts(component: ComponentPainting) -> tuple[int, int]:
    """(released switch intervals, SWITCH_INTERVAL track bins) of one component."""
    flagged = np.asarray(component.status) == int(TrackStatus.SWITCH_INTERVAL)
    if flagged.size == 0:
        return 0, 0
    ids = np.asarray(component.member_ids)
    starts = flagged.copy()
    starts[..., 1:] &= ~flagged[..., :-1] | (ids[..., 1:] != ids[..., :-1])
    return int(np.sum(starts)), int(np.sum(flagged))


def summarise(painting: ChromosomePainting) -> PaintingSummary:
    outcomes = painting.label_continuity
    counts = [interval_counts(component) for component in painting.components]
    return PaintingSummary(
        contig=painting.contig, samples=len(painting.sample_ids),
        components=tuple(ComponentShape(
            founders=int(component.panel.founders), classes=component.states.n_classes,
            states=component.states.n_states, sites=component.states.n_sites,
            bins=component.bins.n_bins,
            packed_emissions=component.packed_emissions is not None)
            for component in painting.components),
        eligible_pairs=int(sum(int(np.sum(component.eligible))
                               for component in painting.components)),
        relabel_moves=sum(len(outcome.moves) for outcome in outcomes if outcome.accepted),
        switches_before=sum(outcome.switches_before or 0 for outcome in outcomes),
        switches_after=sum(outcome.switches_after or 0 for outcome in outcomes),
        switch_intervals=sum(count[0] for count in counts),
        interval_track_bins=sum(count[1] for count in counts))


def write(file, painting: ChromosomePainting):
    """Write the summary, then the product (its presence marks the contig complete)."""
    summary = summarise(painting)
    file.put(SUMMARY_KEY, {"schema": SCHEMA, **asdict(summary)})
    return file.put(KEY, encode(painting))


def load_summary(file) -> PaintingSummary:
    record = dict(file.get(SUMMARY_KEY))
    record.pop("schema")
    record["components"] = tuple(ComponentShape(**shape) for shape in record["components"])
    return PaintingSummary(**record)


def _decode_outcome(record: dict) -> LabelContinuity:
    record = dict(record)
    record["moves"] = tuple(RelabelMove(**{**move, "rows": tuple(move["rows"]),
                                           "winners": tuple(move["winners"]),
                                           "losers": tuple(move["losers"])})
                            for move in record["moves"])
    return LabelContinuity(**record)


def _decode_component(record: dict, panel: ComponentPanel | None,
                      packed: np.ndarray | None, n_states: int) -> ComponentPainting:
    return ComponentPainting(
        panel=panel,
        states=PaintingStates(
            site_indices=record["site_indices"], positions=record["positions"],
            named_alleles=record["named_alleles"],
            classes=tuple(tuple(members) for members in record["classes"]),
            background_frequency=record["background_frequency"]),
        bins=PaintingBins(record["bin_starts"], record["bin_stops"], record["bin_centers"],
                          record["bin_edges"]),
        transitions=PaintingTransitions(n_states, record["log_same"], record["log_one"],
                                        record["log_two"]),
        local_keys=record["local_keys"], eligible=record["eligible"], labels=record["labels"],
        classes=record["track_classes"], status=record["status"], direct=record["direct"],
        member_ids=record["member_ids"],
        member_sets=tuple(tuple(members) for members in record["member_sets"]),
        switch_counts=record["switch_counts"], packed_emissions=packed)


def _painted_panels(file, rows) -> tuple[ComponentPanel, ...]:
    prepared, components = assembly_records.load_product(file)
    return materialise([replace(component, rows=np.asarray(path, dtype=np.int32))
                        for component, path in zip(components, rows)], prepared)


def founder_panels(file) -> tuple[ComponentPanel, ...]:
    """The painted founder panels (the assembly's components after label continuity) in
    chromosome order, materialised from the assembly records: founder stage ``painting``
    of evaluation and export (positions, released calls; rows = founder labels)."""
    return _painted_panels(file, file.get(KEY, items=("rows",))["rows"])


def load(file, *, panels: bool = True, emissions: bool = True) -> ChromosomePainting:
    """The contig's ChromosomePainting.

    ``panels=False`` leaves ``component.panel`` None (no assembly records are read);
    ``emissions=False`` leaves ``packed_emissions`` None (pedigree recomputes them).
    """
    items = ["schema", "contig", "sample_ids", "config", "components", "label_continuity"]
    if panels:
        items.append("rows")
    if emissions:
        items.append("packed_emissions")
    record = file.get(KEY, items=tuple(items))
    encoded = record["components"]
    painted = _painted_panels(file, record["rows"]) if panels else (None,) * len(encoded)
    packed = record["packed_emissions"] if emissions else (None,) * len(encoded)
    return ChromosomePainting(
        contig=record["contig"], sample_ids=tuple(record["sample_ids"]),
        components=tuple(
            _decode_component(component, panel, emissions_upper,
                              len(component["classes"]) + 1)
            for component, panel, emissions_upper in zip(encoded, painted, packed)),
        label_continuity=tuple(_decode_outcome(outcome) for outcome in record["label_continuity"]),
        config=PaintingConfig(**record["config"]))
