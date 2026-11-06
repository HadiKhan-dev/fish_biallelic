"""Read bound contigs with one prefetched successor, sharing no optional-report policy.

At most two evidence blocks are resident. The caller's consumer keeps only its
small result (family-site terms or None), never an evidence block. Reading and
sample-axis checks belong here; numerical work belongs in science/relatedness.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from ...contracts.genetic_map import ChromosomeGeneticMap
from ...contracts.likelihoods import CodedLikelihoods
from ...contracts.observations import PairArray
from ...contracts.roles import freeze_record_arrays, require_schema
from ...core.digest import json_digest
from ...core.errors import InputError
from ...store.protocol import read_items
from ...store.store import CheckpointStore
from ..evidence.records import SUPPORTED_SCHEMAS, load_likelihoods, load_observed
from ..evidence.records import result_reader as evidence_reader

__all__ = ["EvidenceInput", "EvidenceArrays", "evidence_pass", "upstream_identities", "bound_upstream"]
T = TypeVar("T")
_EVIDENCE_KEY = "evidence"  # EV3/D1 published record. Adapter is an explicit code-identity root.


@dataclass(frozen=True)
class EvidenceInput:
    """One bound contig, its evidence derivation and its run map; order is significant."""

    contig: str
    evidence_identity: str
    genetic_map: ChromosomeGeneticMap

    @property
    def map_identity(self) -> str:
        return json_digest(self.genetic_map.identity())


@dataclass(frozen=True)
class EvidenceArrays:
    """The four read-only evidence fields consumed by relatedness, without founder views."""

    sample_ids: tuple[str, ...]
    positions: np.ndarray
    likelihoods: np.ndarray | CodedLikelihoods
    observed: np.ndarray | PairArray

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


def upstream_identities(inputs: tuple[EvidenceInput, ...]) -> dict[str, str]:
    """Name every consumed evidence and map derivation in bound contig order."""
    return {
        key: value
        for item in inputs
        for key, value in (
            (f"evidence/{item.contig}", item.evidence_identity),
            (f"genetic_map/{item.contig}", item.map_identity),
        )
    }


def _read(store: CheckpointStore, item: EvidenceInput, sample_ids: tuple[str, ...]) -> EvidenceArrays:
    file = store.contig(item.contig)
    record = read_items(file, _EVIDENCE_KEY, item.evidence_identity, ("schema", "sample_ids", "positions"))
    if record is None:
        raise InputError(f"relatedness needs evidence for {item.contig}")
    require_schema(record.payload, SUPPORTED_SCHEMAS)
    names = tuple(record.payload["sample_ids"])
    if names != sample_ids:
        raise InputError(f"relatedness sample axis differs on {item.contig}")
    gl = load_likelihoods(file, expected_identity=item.evidence_identity)
    observed = load_observed(file, expected_identity=item.evidence_identity, likelihoods=gl)
    return EvidenceArrays(names, record.payload["positions"], gl, observed)


def evidence_pass(
    root: str | Path,
    inputs: tuple[EvidenceInput, ...],
    sample_ids: tuple[str, ...],
    consume: Callable[[EvidenceInput, EvidenceArrays], T],
    *,
    team: int = 1,
) -> tuple[T, ...]:
    """Read/consume each contig exactly once in bound order, prefetching one successor."""
    if not inputs:
        raise InputError("relatedness requires at least one bound contig")
    if len({item.contig for item in inputs}) != len(inputs):
        raise InputError("relatedness contigs must be unique")
    store = CheckpointStore(root, nthreads=team)
    result = []
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="relatedness-evidence") as pool:
        pending = pool.submit(_read, store, inputs[0], sample_ids)
        for index, item in enumerate(inputs):
            evidence = pending.result()
            if index + 1 < len(inputs):
                pending = pool.submit(_read, store, inputs[index + 1], sample_ids)
            result.append(consume(item, evidence))
            del evidence
    return tuple(result)


def bound_upstream(store: CheckpointStore, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Verify upstream evidence against EV3's independently derived bound-input identity."""
    upstream = {}
    for name in manifest["contigs"]:
        reader = evidence_reader(store, manifest, product="evidence", contig=name)
        if reader is None:
            raise InputError(f"relatedness needs upstream evidence for {name}")
        record = read_items(store.contig(name), reader.key, reader.identity, ("schema",))
        if record is None:
            raise InputError(f"missing evidence publication for {name}")
        require_schema(record.payload, reader.schema)
        upstream[f"evidence/{name}"] = reader.identity
        upstream[f"genetic_map/{name}"] = manifest["map_identities"][name]
    return upstream
