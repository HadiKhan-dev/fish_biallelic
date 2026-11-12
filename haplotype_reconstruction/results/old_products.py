"""Adapt checked VCF-route products, including release 4.14 in its own interpreter.

An older run is decoded by that release's unmodified Results reader. No native
release guard, store layout, product schema or derivation check is bypassed.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

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


# This program uses only APIs present in 4.14.0. PYTHONPATH selects that release;
# it never imports the caller's modified simulation or evaluation implementations.
_SNAPSHOT = r"""
import pickle, sys
from pathlib import Path
from haplotype_reconstruction import __version__
from haplotype_reconstruction.results.reader import Results
from haplotype_reconstruction.results.panels import FOUNDER_STAGES
from haplotype_reconstruction.stages.evidence import records
from haplotype_reconstruction.store.protocol import read_items
from haplotype_reconstruction.contracts.roles import require_schema
if __version__ != "4.14.0":
    raise RuntimeError("the old-product adapter requires release 4.14.0")
r = Results(sys.argv[1])
data = dict(root=r.root, checkpoints=r.checkpoints, manifest=r.manifest,
            contigs=r.contigs, pedigree=r.pedigree(), axes={}, panels={}, phase={}, recombination={})
for c in r.contigs:
    request = r.request(records, "evidence", c)
    if request is None:
        raise RuntimeError(c + ": no completed evidence")
    item = read_items(r._file(c), request.key, request.identity,
                      ("schema", "sample_ids", "positions", "payload_identity"))
    if item is None:
        raise RuntimeError(c + ": no completed evidence")
    require_schema(item.payload, request.schema)
    data["axes"][c] = item.payload
    data["panels"][c] = {s: tuple(r.stage_panels(c, s)) for s in FOUNDER_STAGES if r.stage_present(c, s)}
    data["phase"][c] = r.phase(c)
    data["recombination"][c] = r.recombination(c)
with open(sys.argv[2], "wb") as handle:
    pickle.dump(data, handle, protocol=5)
"""


def _freeze_transfer(value):
    """Restore read-only array views after the private pickle transfer."""
    if isinstance(value, np.ndarray):
        result = value.view()
        result.flags.writeable = False
        return result
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            object.__setattr__(value, item.name, _freeze_transfer(getattr(value, item.name)))
        return value
    if isinstance(value, dict):
        return {key: _freeze_transfer(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_freeze_transfer(item) for item in value)
    if isinstance(value, list):
        return [_freeze_transfer(item) for item in value]
    return value


class OldProducts:
    """Read-only snapshot of products checked by an unmodified 4.14.0 reader.

    ``release_tree`` is a source/package root supplied by the caller. It is never
    inferred from machine-specific directories or substituted into sys.modules.
    """

    def __init__(self, run, release_tree, *, python=None, timeout=600):
        run, release_tree = Path(run).resolve(), Path(release_tree).resolve()
        environment = dict(
            PYTHONPATH=str(release_tree),
            PYTHONDONTWRITEBYTECODE="1",
            NUMBA_DISABLE_INTEL_SVML="1",
            NUMBA_THREADING_LAYER="omp",
            NUMBA_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        with tempfile.TemporaryDirectory(prefix="observation-products-") as temporary:
            snapshot = Path(temporary) / "products.pkl"
            environment["NUMBA_CACHE_DIR"] = str(Path(temporary) / "numba")
            result = subprocess.run(
                [str(python or sys.executable), "-P", "-c", _SNAPSHOT, str(run), str(snapshot)],
                env=environment,
                cwd=temporary,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode:
                raise InputError(f"old-product reader failed (exit {result.returncode}): {result.stderr.strip()}")
            # This is a private, locally generated transfer file, after the old
            # reader has validated all provenance and publication envelopes.
            with snapshot.open("rb") as handle:
                self._data = _freeze_transfer(pickle.load(handle))
        self.root, self.checkpoints = self._data["root"], self._data["checkpoints"]
        self.manifest = self._data["manifest"]
        self.contigs = tuple(self._data["contigs"])

    def simulation_manifest(self):
        return None

    def pedigree(self):
        return self._data["pedigree"]

    def observation_axes(self, contig):
        axes = self._data["axes"][contig]
        return ObservationAxes(tuple(axes["sample_ids"]), axes["positions"], axes["payload_identity"])

    def stage_present(self, contig, stage):
        if stage == "family_phase":
            return self._data["phase"][contig] is not None
        if stage == "recombination":
            return self._data["recombination"][contig] is not None
        return stage in self._data["panels"][contig]

    def stage_panels(self, contig, stage):
        return self._data["panels"][contig][stage]

    def phase(self, contig):
        return self._data["phase"][contig]

    def recombination(self, contig):
        return self._data["recombination"][contig]
