"""Evaluation facade over a VCF-route run's checked products, with its evidence axes for truth binding.

No release guard, store layout, product schema or derivation check is bypassed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts.roles import freeze_record_arrays, record_field
from ..core.errors import InputError
from ..store.protocol import read_items
from .reader import Results


@dataclass(frozen=True)
class ObservationAxes:
    """Retained evidence axes and the independently bound consumed-source digest."""

    sample_ids: tuple[str, ...] = record_field("obs", axes=("N",), dtype="str")
    positions: np.ndarray = record_field("obs", axes=("L",), units="bp", dtype="int64")
    payload_identity: str = record_field("diagnostic", dtype="str")

    def __post_init__(self):
        freeze_record_arrays(self)


class RunProducts:
    """Evaluation facade over the current release's checked Results."""

    def __init__(self, run, *, checkpoints=None):
        self.reader = run if isinstance(run, Results) else Results(run, checkpoints=checkpoints)
        self.root, self.checkpoints = self.reader.root, self.reader.checkpoints

    def __getattr__(self, name):
        reader = self.__dict__.get("reader")
        if reader is None:
            raise AttributeError(name)
        return getattr(reader, name)

    def observation_axes(self, contig):
        from ..stages.evidence import records

        request = self.reader.request(records, "evidence", contig)
        if request is None:
            raise InputError(f"{contig}: no completed evidence to bind to observation truth")
        record = read_items(
            self.reader._file(contig),
            request.key,
            request.identity,
            ("schema", "sample_ids", "positions", "payload_identity"),
        )
        if record is None:
            raise InputError(f"{contig}: no completed evidence")
        from ..contracts.roles import require_schema

        require_schema(record.payload, request.schema)
        return ObservationAxes(
            tuple(record.payload["sample_ids"]), record.payload["positions"], record.payload["payload_identity"]
        )
