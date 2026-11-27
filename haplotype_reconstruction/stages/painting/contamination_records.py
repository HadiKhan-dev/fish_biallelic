"""Records of the per-library contamination stages (4.22 QC A): keys, identities, the reader and the run's decision.

stages.painting.contamination owns the actions (the contig's cells, the run's decision, corrected evidence for the
consumers); this module is what the results layer and the consumers' identities read. Cells bind the painting's and
the evidence's published identities; the decision binds every analysed contig's cells.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...contracts.contigs import AnalysisContigs
from ...contracts.roles import require_schema
from ...core.concurrency import threaded_each
from ...core.errors import InputError, InvariantError
from ...science.painting.parameters import (
    CONTAMINATION_ALPHA_FLOOR,
    CONTAMINATION_CARE_ALPHA,
    CONTAMINATION_DONOR_RESIDUAL,
    CONTAMINATION_Z_FLOOR,
)
from ...store.identity import code_dependency_identity
from ...store.protocol import Derivation, NumericalPolicy, resume
from ...store.store import CheckpointStore

CELLS = "painting/contamination_cells"
CONTAMINATION = "painting/contamination"
SCHEMA = 1
_CODE = ("science.painting.contamination", "stages.painting.contamination", "stages.painting.contamination_records")

__all__ = [
    "CELLS", "CONTAMINATION", "SCHEMA", "Reader", "cells_derivation", "decision", "derivation", "expected",
    "progress", "result_reader",
]


def _settings() -> dict[str, float]:
    return dict(alpha_floor=CONTAMINATION_ALPHA_FLOOR, z_floor=CONTAMINATION_Z_FLOOR,
                donor_residual=CONTAMINATION_DONOR_RESIDUAL, care_alpha=CONTAMINATION_CARE_ALPHA)


def cells_derivation(painting_identity: str, evidence_identity: str, *, numerical_policy=None) -> str:
    """A contig's contamination cells: the painting's and the evidence's identities."""
    return Derivation(
        code_dependency_identity(_CODE), dict(product=CELLS, settings=_settings()),
        numerical_policy or NumericalPolicy.current(),
        dict(painting=painting_identity, evidence=evidence_identity),
    ).identity


def derivation(cells: Mapping[str, str], *, numerical_policy=None) -> str:
    """The run's contamination decision: every analysed contig's cells identity."""
    return Derivation(
        code_dependency_identity(_CODE), dict(product=CONTAMINATION, settings=_settings()),
        numerical_policy or NumericalPolicy.current(), dict(cells),
    ).identity


@dataclass(frozen=True)
class Reader:
    """A checked request for one contamination product (stages.inputs.protocol.ProductReader)."""

    identity: str
    product: str
    key: str
    schema: int = SCHEMA

    def decode(self, payload: dict[str, Any]) -> dict[str, Any]:
        require_schema(payload, SCHEMA)
        return {name: value for name, value in payload.items() if name != "schema"}


# Contamination requests this process derived: records.run_binding + (product, contig) -> (manifest, Reader). Pure
# functions of the manifest and the published painting and evidence identities (which never change); a failed check
# raises and is never remembered. Every pedigree contig task reads the run's decision, which binds every contig's
# cells: re-derived per call, one TroMau preparation re-read every contig's painting and evidence headers each time.
_EXPECTED: dict[tuple, tuple[Any, Reader]] = {}
_EXPECTED_GUARD = threading.Lock()


def expected(store: CheckpointStore, manifest: Mapping[str, Any], *, product: str, contig: str | None = None) -> Reader:
    """Rebuild the identity from the checked published painting and evidence requests (cells) or every analysed
    contig's cells (the decision); never read it from the product's own envelope. Derived once per process and run
    folder (_EXPECTED)."""
    from .records import run_binding

    key = (*run_binding(store), product, contig)
    with _EXPECTED_GUARD:
        cached = _EXPECTED.get(key)
    if cached is not None and (cached[0] is manifest or cached[0] == manifest):
        return cached[1]
    request = _derive_expected(store, manifest, product=product, contig=contig)
    with _EXPECTED_GUARD:
        _EXPECTED[key] = (manifest, request)
    return request


def _derive_expected(store: CheckpointStore, manifest: Mapping[str, Any], *, product: str,
                     contig: str | None = None) -> Reader:
    """expected, derived afresh."""
    from ..evidence import records as evidence_records
    from . import records as painting_records

    policy = NumericalPolicy(**manifest["numerical_policy"])
    if product == "cells":
        if contig is None or contig not in manifest["contigs"]:
            raise InputError("contamination cells require a bound contig")
        painting = painting_records.result_reader(store, manifest, product="painting", contig=contig, panels=False,
                                                  emissions=False)
        evidence = evidence_records.result_reader(store, manifest, product="evidence", contig=contig)
        if painting is None or evidence is None:
            raise InvariantError(f"{contig}: contamination cells require the published painting and evidence")
        return Reader(cells_derivation(painting.identity, evidence.identity, numerical_policy=policy), product, CELLS)
    if product == "contamination":
        contigs = AnalysisContigs.from_manifest(manifest).included
        identities = threaded_each(lambda c: expected(store, manifest, product="cells", contig=c).identity, contigs)
        return Reader(derivation(dict(zip(contigs, identities, strict=True)), numerical_policy=policy), product,
                      CONTAMINATION)
    raise InputError(f"unsupported contamination product {product!r}")


def result_reader(store, manifest, *, product: str, contig: str | None = None, **options) -> Reader | None:
    """The published product's request, None before it exists."""
    del options
    file = store.run if product == "contamination" else store.contig(contig)
    key = CONTAMINATION if product == "contamination" else CELLS
    if not file.has(key):
        return None
    return expected(store, manifest, product=product, contig=contig)


def progress(file) -> dict[str, Any]:
    """Status rows: whether the contig's cells are published."""
    return {"cells": bool(file.has(CELLS))}


def decision(store: CheckpointStore, manifest: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """The run's published decision and its identity (consumers bind the identity)."""
    request = expected(store, manifest, product="contamination")
    saved = resume(store.run, CONTAMINATION, request.identity)
    if saved is None:
        raise InvariantError("the run's contamination decision is not published")
    return request.identity, request.decode(saved.payload)
