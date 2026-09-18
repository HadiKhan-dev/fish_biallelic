"""Atomic assembly work checkpoints and scientific resume identities."""
from __future__ import annotations


import copy
from dataclasses import dataclass
import json
import os
from typing import Any, Mapping, Protocol


STAGE2_RELEASE_CHECKPOINT_SCHEMA = "stage2-release-work-checkpoint-v1"


class AssemblyCheckpointIO(Protocol):
    """Minimal checkpoint callback contract consumed by the release runner."""

    def bind(self, identity: Mapping[str, Any]) -> None:
        ...

    def load(self, phase: str) -> Any | None:
        ...

    def save(self, phase: str, payload: Any) -> None:
        ...


def _canonical_identity(identity: Mapping[str, Any]) -> str:
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("Stage-2 checkpoint identity must be a nonempty mapping")
    return json.dumps(
        copy.deepcopy(dict(identity)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_token(value: str, name: str) -> str:
    result = str(value)
    if (
            not result
            or result in {".", ".."}
            or os.sep in result
            or (os.altsep is not None and os.altsep in result)):
        raise ValueError(f"{name} must be a nonempty path-component token")
    return result


@dataclass(frozen=True)
class Stage2ReleaseWorkCheckpoint:
    """One atomic phase payload bound to an exact release identity."""

    schema: str
    phase: str
    identity_json: str
    payload: Any


class AssemblyCheckpointStore:
    """CheckpointStore-backed adapter for one contig's internal phases.

    The adapter writes into a separate work stage and never creates its done
    marker. Pipeline-level completion remains exclusively the responsibility
    of the final T09 publisher.
    """

    def __init__(self, checkpoint_store, *, work_stage: str, contig: str):
        if not isinstance(checkpoint_store, core_runtime.CheckpointStore):
            raise TypeError("checkpoint_store must be a CheckpointStore")
        self.checkpoint_store = checkpoint_store
        self.work_stage = _validate_token(work_stage, "work_stage")
        self.contig = _validate_token(contig, "contig")
        self._identity_json: str | None = None

    def _artifact(self, phase: str) -> str:
        return f"{self.contig}.__stage2_release__.{_validate_token(phase, 'phase')}"

    def bind(self, identity: Mapping[str, Any]) -> None:
        canonical = _canonical_identity(identity)
        if self._identity_json is not None and self._identity_json != canonical:
            raise RuntimeError("checkpoint adapter was rebound to another release")
        self._identity_json = canonical

    def load(self, phase: str) -> Any | None:
        if self._identity_json is None:
            raise RuntimeError("checkpoint adapter must be bound before loading")
        artifact = self._artifact(phase)
        if not self.checkpoint_store.contig_done(self.work_stage, artifact):
            return None
        checkpoint = self.checkpoint_store.load_contig(
            self.work_stage, artifact
        )
        if not isinstance(checkpoint, Stage2ReleaseWorkCheckpoint):
            raise TypeError("unrecognized Stage-2 release work checkpoint")
        if checkpoint.schema != STAGE2_RELEASE_CHECKPOINT_SCHEMA:
            raise ValueError("unknown Stage-2 release checkpoint schema")
        if checkpoint.phase != str(phase):
            raise ValueError("Stage-2 release checkpoint phase mismatch")
        if checkpoint.identity_json != self._identity_json:
            raise RuntimeError(
                "Stage-2 release checkpoint scientific identity mismatch"
            )
        return checkpoint.payload

    def save(self, phase: str, payload: Any) -> None:
        if self._identity_json is None:
            raise RuntimeError("checkpoint adapter must be bound before saving")
        artifact = self._artifact(phase)
        checkpoint = Stage2ReleaseWorkCheckpoint(
            schema=STAGE2_RELEASE_CHECKPOINT_SCHEMA,
            phase=str(phase),
            identity_json=self._identity_json,
            payload=payload,
        )
        self.checkpoint_store.save_contig(
            self.work_stage, artifact, checkpoint
        )

import haplotype_reconstruction.core.runtime as core_runtime
