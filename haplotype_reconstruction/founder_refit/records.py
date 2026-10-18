"""Founder refit records in a contig's checkpoint file.

``identity/founder_refit``  stage identity: refit config + identities of the painting and
                            evidence records read (bound once)
``founder_refit``           per component the refit released alleles and the changed cells

The refit panels are rebuilt on load from the painted panels (painting.records.founder_panels):
released = the stored refit alleles, inference = them on the painted snapshot's called cells.
"""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np

from ..painting import records as painting_records
from ..store.identity import stage_identity as build_identity
from .config import FounderRefitConfig
from .refit import ComponentRefit, FounderRefit

STAGE = "founder_refit"
KEY = "founder_refit"
SCHEMA = 1


def stage_identity(file, config: FounderRefitConfig) -> dict:
    return build_identity(STAGE, schema=SCHEMA, config=config, upstream={
        "painting": file.info(painting_records.KEY).identity,
        "evidence": file.info(painting_records.EVIDENCE_KEY).identity,
    })


def bind_identity(store, file, config: FounderRefitConfig) -> None:
    store.bind_identity(file, STAGE, stage_identity(file, config))


def complete(file) -> bool:
    return file.has(KEY)


def progress(file) -> dict:
    return {"stage": STAGE, "complete": file.has(KEY)}


def encode(refit: FounderRefit) -> dict:
    return {
        "schema": SCHEMA, "contig": refit.contig, "sample_ids": list(refit.sample_ids),
        "excluded_samples": list(refit.excluded_samples),
        "components": [{"released": component.panel.released,
                        "changed_rows": component.changed_rows,
                        "changed_sites": component.changed_sites,
                        "gain_nats": component.gain_nats} for component in refit.components],
    }


def write(file, refit: FounderRefit):
    return file.put(KEY, encode(refit))


def load(file) -> FounderRefit:
    """The contig's refit product with its refit ComponentPanels."""
    record = file.get(KEY)
    panels = painting_records.founder_panels(file)
    components = []
    for panel, component in zip(panels, record["components"]):
        released = np.asarray(component["released"], dtype=np.int8)
        components.append(ComponentRefit(
            panel=replace(panel, released=released,
                          inference=np.where(panel.inference >= 0, released, -1).astype(np.int8)),
            changed_rows=component["changed_rows"], changed_sites=component["changed_sites"],
            gain_nats=component["gain_nats"]))
    return FounderRefit(contig=record["contig"], sample_ids=tuple(record["sample_ids"]),
                        components=tuple(components),
                        excluded_samples=tuple(record["excluded_samples"]))


def founder_panels(file):
    """The refit ComponentPanels in chromosome order: founder stage ``founder_refit`` of
    evaluation and export, and the scaffold alleles of family phase."""
    return load(file).panels
