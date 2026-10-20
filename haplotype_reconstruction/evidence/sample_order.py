"""The run's canonical sample order (release 2.1, design A-11 / review R11).

Every stage after the evidence stage has decisions whose schedule or slot assignment follows the sample index
(family-phase sweeps and colouring, pedigree pair slots and candidate cuts, recombination's greedy order, sums in
sample order). Tolerances cannot make those order-free, so the run applies one permutation, once, at the entry of
the evidence stage: the samples sorted by name (exact, noise-free, unique, and independent of the input order).
Every record of the run then holds its sample axis in that order, and a run on permuted input is bitwise identical
to the run on the original input. Outputs whose order users rely on (the export's sample columns) are returned to the
input order, which the run manifest records (``source.input_sample_ids``).

:class:`CanonicalOrderSource` wraps an observation source (``kind``, ``contigs()``, ``sample_ids()``,
``identity()``, ``materialise(store, contigs, budget)``, ``reads(contig, budget)``; pipeline.sources). The driver
applies it to the source it builds (``check_run``), after any wrapper a test harness may add, so that the canonical
order is taken from whatever order the source delivers.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from ..maths.ties import run_sample_order

# The order rule, recorded in the source identity (it is part of every record's provenance).
SAMPLE_ORDER_RULE = "sample-name-v1"
# Key of the input order in the source identity (and therefore in the run manifest's ``source``).
INPUT_ORDER_KEY = "input_sample_ids"


class CanonicalOrderSource:
    """An observation source whose sample axis is in canonical order (samples sorted by name).

    ``sample_ids()`` and ``reads()`` give the canonical order; ``input_sample_ids()`` the wrapped source's order;
    ``identity()`` the wrapped source's identity plus the order rule and the input order. Every other attribute is
    the wrapped source's.
    """

    def __init__(self, inner):
        self._inner = inner
        self._input = None
        self._order = None

    def __getattr__(self, name):
        # Only called for attributes this class does not define (kind, contigs, check, materialise, ...).
        return getattr(self._inner, name)

    @property
    def inner(self):
        return self._inner

    def _resolve(self):
        if self._order is None:
            ids = tuple(str(name) for name in self._inner.sample_ids())
            self._order = run_sample_order(ids)
            self._input = ids
        return self._input, self._order

    def input_sample_ids(self) -> tuple[str, ...]:
        """The wrapped source's sample order (the order the export restores)."""
        return self._resolve()[0]

    def order(self) -> np.ndarray:
        """int64[N] permutation: canonical position i holds input sample order()[i]."""
        return self._resolve()[1].copy()

    def sample_ids(self) -> tuple[str, ...]:
        ids, order = self._resolve()
        return tuple(ids[index] for index in order)

    def identity(self) -> dict:
        ids, _ = self._resolve()
        value = dict(self._inner.identity())
        value["sample_order"] = SAMPLE_ORDER_RULE
        value[INPUT_ORDER_KEY] = list(ids)
        return value

    def reads(self, contig, budget):
        """The wrapped source's reads of ``contig`` with the sample axis in canonical order."""
        ids, order = self._resolve()
        reads = self._inner.reads(contig, budget)
        if tuple(reads.sample_ids) != ids:
            raise RuntimeError(f"{contig}: the reads' sample axis differs from the source's sample ids")
        if np.array_equal(order, np.arange(len(order))):
            return reads
        # One permuted copy of the counts (int32 [N, L, 2]); the input-order reads are freed on return.
        from ..execution import reserve_memory
        with reserve_memory(reads.counts.nbytes, label="evidence sample ordering"):
            return replace(reads, sample_ids=tuple(ids[index] for index in order), counts=reads.counts[order])


def canonical_order_source(source):
    """``source`` with its sample axis in canonical order (idempotent)."""
    return source if isinstance(source, CanonicalOrderSource) else CanonicalOrderSource(source)


def recorded_input_order(manifest: dict | None) -> tuple[str, ...] | None:
    """The input sample order a run manifest records (``manifest["source"]["input_sample_ids"]``), or None for
    runs without the canonical order (release 2.0: their records are in input order)."""
    source = (manifest or {}).get("source") or {}
    ids = source.get(INPUT_ORDER_KEY)
    return None if ids is None else tuple(str(name) for name in ids)


def input_rows(sample_ids: Sequence[str], input_ids: Sequence[str] | None) -> np.ndarray | None:
    """Rows of a sample axis ``sample_ids`` in the order ``input_ids`` (int64[N]: output row i is input_ids[i]),
    or None when the orders already agree (or no input order is recorded)."""
    if input_ids is None:
        return None
    sample_ids, input_ids = tuple(map(str, sample_ids)), tuple(map(str, input_ids))
    if sample_ids == input_ids:
        return None
    index = {name: row for row, name in enumerate(sample_ids)}
    if len(index) != len(sample_ids) or len(input_ids) != len(sample_ids) or set(input_ids) != set(index):
        raise ValueError("the recorded input sample order is not a permutation of the run's samples")
    return np.asarray([index[name] for name in input_ids], dtype=np.int64)


def store_input_order(store) -> tuple[str, ...] | None:
    """The input sample order recorded in a run's manifest (None for runs without the canonical order)."""
    return recorded_input_order(store.manifest("run"))


def frame_in_input_order(frame, input_ids: Sequence[str] | None, column: str = "Sample"):
    """A per-sample table with its rows stably sorted by the input position of ``column`` (published tables list
    samples in the input order, as release 2.0 did; the records keep the canonical order). Rows whose sample is not
    in ``input_ids`` keep their relative order after the others. The frame is returned unchanged when
    ``input_ids`` is None or it has no ``column``."""
    if input_ids is None or column not in getattr(frame, "columns", ()):
        return frame
    position = {str(name): index for index, name in enumerate(input_ids)}
    keys = np.asarray([position.get(str(name), len(position)) for name in frame[column]], dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    if np.array_equal(order, np.arange(len(order))):
        return frame
    return frame.iloc[order].reset_index(drop=True)
