"""Stage founder completion and construct the shared assembly evidence view.

Owns record decoding, bounded execution and completion publication; AS1 owns
all completion decisions. No scientific result depends on worker allocation.
Method: WP_AS6 §4.1/4.6; WP_AS1 §6.2.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from haplotype_reconstruction.contracts.assembly import CompletionProduct
from haplotype_reconstruction.contracts.evidence import ContigEvidence
from haplotype_reconstruction.contracts.payload import to_payload
from haplotype_reconstruction.core.ties import TieSettings
from haplotype_reconstruction.engine.runtime import thread_team
from haplotype_reconstruction.engine.workers import (
    SharedArrayRef,
    attach_shared_array,
    close_shared_memory,
    create_shared_array,
)
from haplotype_reconstruction.science.assembly.completion.parameters import CompletionParameters
from haplotype_reconstruction.science.assembly.completion.run import CompletionWorkspace, complete
from haplotype_reconstruction.science.assembly.evidence import AssemblyEvidence
from haplotype_reconstruction.science.assembly.run import AssemblyInputs

from .records import COMPLETION, RecordCapability, decode_completion, encode_completion


def build_evidence(evidence: ContigEvidence, identity: str, *, cores: int = 1) -> AssemblyEvidence:
    """Construct one exact contig view after evidence has been identity-checked."""
    with thread_team(cores):
        return AssemblyEvidence.from_contig(evidence, identity)


def complete_stage(
    inputs: AssemblyInputs,
    parameters: CompletionParameters,
    records: RecordCapability | None,
    *,
    ties: TieSettings | None = None,
    cores: int = 1,
    map_tasks: Callable | None = None,
) -> CompletionProduct:
    """Resume completed preparation, or execute AS1 and publish once."""
    ties = TieSettings() if ties is None else ties
    stored = None if records is None else records.load(COMPLETION)
    if stored is not None:
        return decode_completion(stored)
    with thread_team(cores):
        prepared, summary = complete(
            inputs.panels, inputs.evidence, parameters, CompletionWorkspace(ties=ties, map_tasks=map_tasks)
        )
    product = CompletionProduct(tuple(prepared), to_payload(summary))
    if records is not None:
        records.save(COMPLETION, encode_completion(product))
    return product


@dataclass
class SharedEvidence:
    """Stage-owned lifetime of engine shared arrays and the plain science view.

    Child batches receive only scoring_ref and attach without copying. The owner
    keeps the segments alive until their last consumer and can release neutral
    independently before final refinement. No handle enters AssemblyEvidence.
    """

    view: AssemblyEvidence
    references: dict[str, SharedArrayRef]
    handles: dict[str, SharedMemory]

    @property
    def scoring_ref(self) -> SharedArrayRef:
        return self.references["scoring"]

    def release_neutral(self) -> None:
        self.view.release_neutral()
        handle = self.handles.pop("neutral", None)
        self.references.pop("neutral", None)
        if handle is not None:
            close_shared_memory((handle,), unlink=True)

    def close(self) -> None:
        """Close only after all science consumers have completed."""
        close_shared_memory(tuple(self.handles.values()), unlink=True)
        self.handles.clear()
        self.references.clear()

    def __enter__(self) -> SharedEvidence:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def share_evidence(evidence: AssemblyEvidence) -> SharedEvidence:
    """Copy a completed view once into engine segments, then publish read-only views.

    IN1 owns this context across selection and assembly. bind_session accepts the
    owner explicitly; a plain AssemblyEvidence remains valid for bounded replays.
    """
    handles: dict[str, SharedMemory] = {}
    references: dict[str, SharedArrayRef] = {}
    arrays: dict[str, Any] = {}
    try:
        for name in ("positions", "neutral", "scoring", "observed", "kept"):
            source = getattr(evidence, name)
            if source is None:
                arrays[name] = None
                continue
            handle, reference = create_shared_array(source)
            handles[name], references[name] = handle, reference
            array = np.ndarray(reference.shape, dtype=np.dtype(reference.dtype), buffer=handle.buf)
            array.flags.writeable = False
            arrays[name] = array
        view = AssemblyEvidence(**arrays, identity=evidence.identity, depth_keys=evidence.depth_keys)
        return SharedEvidence(view, references, handles)
    except BaseException:
        close_shared_memory(tuple(handles.values()), unlink=True)
        raise


@contextmanager
def attached_scoring(scoring: np.ndarray | SharedArrayRef) -> Iterator[np.ndarray]:
    """Borrow a child worker's scoring view without a transport copy."""
    if isinstance(scoring, np.ndarray):
        yield scoring
        return
    handle, array = attach_shared_array(scoring)
    array.flags.writeable = False
    try:
        yield array
    finally:
        close_shared_memory((handle,))
