"""Record of the evidence stage: one contig record ``evidence`` and its identity.

Payload (a dict; arrays decode selectively with ``items=``): ``schema``, ``contig``,
``sample_ids``, the ContigEvidence arrays (``positions``, ``counts``, ``likelihoods``,
``observed``, ``discovery_sites``, ``founder_excluded``, ``block_spans``) and ``report``
(EvidenceReport.record(): loader counts, site/sample QC, read-model calibration, timings).
Identity ``identity/evidence`` (per contig) = stage_identity("evidence", config, upstream =
{"source": digest of the observation source identity}). Downstream stages bind
``file.info(EVIDENCE).identity``.
"""
from __future__ import annotations

import numpy as np

from ..model.evidence import ContigEvidence
from ..execution import reserve_memory
from ..store.identity import json_digest, stage_identity
from .config import EvidenceConfig
from .reports import EvidenceReport

STAGE = "evidence"
EVIDENCE = "evidence"          # contig record key
SCHEMA = 1
ARRAYS = ("positions", "counts", "likelihoods", "observed", "discovery_sites", "founder_excluded",
          "block_spans")


def identity(config: EvidenceConfig, source_identity: dict) -> dict:
    """The identity of a contig's evidence: stage config + the observation source."""
    return stage_identity(STAGE, schema=SCHEMA, config=config,
                          upstream={"source": json_digest(source_identity)})


def write(file, evidence: ContigEvidence, report: EvidenceReport):
    """Record one contig's evidence and the stage's report; returns the RecordInfo."""
    if evidence.counts is None:
        raise ValueError("the evidence record keeps the counts")
    payload = dict(schema=SCHEMA, contig=evidence.contig, sample_ids=evidence.sample_ids,
                   report=report.record())
    payload.update({name: getattr(evidence, name) for name in ARRAYS})
    # The codec streams existing contiguous buffers, never a second full evidence payload.
    # Its rolling chunks need <=2 GiB raw/compressed space plus grouping and metadata;
    # smaller records need at most twice their raw arrays plus that bounded overhead.
    raw_bytes = sum(getattr(evidence, name).nbytes for name in ARRAYS)
    raw_bytes += report.input_positions.nbytes
    with reserve_memory(min(2 * raw_bytes, 2 << 30) + (256 << 20), label="evidence record write"):
        return file.put(EVIDENCE, payload)


def complete(file) -> bool:
    return file.has(EVIDENCE)


def progress(file) -> dict:
    """Status of one contig without decoding payloads."""
    info = file.info(EVIDENCE)
    return dict(evidence=info is not None,
                **({} if info is None else dict(decoded_bytes=info.decoded_bytes)))


def load(file, *, counts: bool = False, nthreads: int | None = None) -> ContigEvidence:
    """A contig's evidence; the counts are decoded only when asked (most stages do not need them)."""
    names = tuple(name for name in ARRAYS if counts or name != "counts")
    payload = file.get(EVIDENCE, items=("contig", "sample_ids", *names), nthreads=nthreads)
    return ContigEvidence(contig=payload["contig"], sample_ids=payload["sample_ids"],
                          counts=payload.get("counts"),
                          **{name: payload[name] for name in names if name != "counts"})


def load_arrays(file, names, *, nthreads: int | None = None) -> dict:
    """Selected arrays of a contig's evidence ({name: array}), decoding only those (e.g. positions, observed)."""
    names = tuple(names)
    unknown = sorted(set(names) - set(ARRAYS))
    if unknown:
        raise KeyError(f"not evidence arrays: {unknown}")
    payload = file.get(EVIDENCE, items=names, nthreads=nthreads)
    return {name: payload[name] for name in names}


def load_report(file, *, nthreads: int | None = None) -> EvidenceReport:
    """The stage's report of a contig (QC tables, calibration, loader counts)."""
    payload = file.get(EVIDENCE, items=("report", "positions"), nthreads=nthreads)
    report = payload["report"]
    kept = np.isin(report["input_positions"], payload["positions"], assume_unique=True)
    return EvidenceReport.from_record(report, kept)
