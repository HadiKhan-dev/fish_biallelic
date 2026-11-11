"""Check expected product requests without importing stage orchestration or science."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Protocol, TypeVar

from ..contracts.roles import require_schema
from ..core.errors import SettingsError
from ..store.protocol import read_items, resume
from ..store.store import CheckpointStore

Product = TypeVar("Product", covariant=True)


class ProductReader(Protocol[Product]):
    """A stage-supplied read request; constructing it verifies manifest dependencies.

    The expected identity must be computed from the bound configuration and inputs,
    never copied from the product's own publication envelope. Stage adapters may
    implement this structural protocol with their own ordinary frozen dataclasses.
    """

    @property
    def key(self) -> str: ...

    @property
    def identity(self) -> str: ...

    @property
    def schema(self) -> int: ...

    def decode(self, payload: dict[str, Any]) -> Product: ...


class ProductInputs:
    """Read checked inputs; writes require a separate explicit store capability."""

    def __init__(self, checkpoints, *, nthreads=1):
        self.checkpoints = Path(checkpoints)
        self._store = CheckpointStore(checkpoints, nthreads=nthreads)
        self._store.check_layout()
        manifest = self._store.manifest("run")
        if manifest is None:
            raise SettingsError(f"{self.checkpoints}: run has not been initialized; use haplotypes init")
        require_schema(manifest, 1)
        self._manifest = manifest

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a private copy of the bound run manifest; callers cannot mutate reader state."""
        return copy.deepcopy(self._manifest)


    @property
    def contigs(self) -> tuple[str, ...]:
        """The manifest's bound processing order, including contigs with no products yet."""
        return tuple(self._manifest.get("contigs", ()))


    def _file(self, contig: str | None) -> Any:
        if contig is None:
            return self._store.run
        if contig not in self.contigs:
            raise SettingsError(f"contig {contig!r} is not in the run manifest: {', '.join(self.contigs)}")
        return self._store.contig(contig)


    def read(self, request: ProductReader[Product], contig: str | None = None) -> Product | None:
        """Check a stage adapter's expected identity and schema, then decode its immutable product."""
        items = getattr(request, "items", None)
        record = (
            resume(self._file(contig), request.key, request.identity)
            if items is None
            else read_items(self._file(contig), request.key, request.identity, tuple(dict.fromkeys(("schema", *items))))
        )
        if record is None:
            return None
        require_schema(record.payload, request.schema)
        return request.decode(record.payload)


    def request(self, adapter: Any, product: str, contig: str | None = None, **options: Any) -> Any:
        """Construct an owner's request without exposing the reader's store."""
        return adapter.result_reader(self._store, self.manifest, product=product, contig=contig, **options)


    def receipt(self, request: ProductReader[Any] | None, contig: str | None = None) -> Any | None:
        """Check one publication's expected identity and schema using its small header."""
        return check_receipt(self._file(contig), request)

    def output_receipt(self, key, identity, schema, contig):
        """Check a supersedable output publication against its selected source products."""
        from ..store.protocol import read_output

        record = read_output(self._file(contig), key, identity=lambda payload: payload["identity"])
        if record is None or record.identity != identity:
            return None
        require_schema(record.payload, schema)
        return record

    def require_receipt(self, request: ProductReader[Any] | None, contig: str | None = None) -> Any:
        """Require a published product without decoding its arrays."""
        record = self.receipt(request, contig)
        if record is None:
            raise SettingsError(f"{contig}: required publication is missing")
        return record


    def require(self, request: ProductReader[Product] | None, contig: str | None = None) -> Product:
        """Decode a required upstream product through its owning adapter."""
        value = None if request is None else self.read(request, contig)
        if value is None:
            raise SettingsError(f"{contig}: required publication is missing")
        return value


    @property
    def synthetic(self) -> bool:
        """Whether the bound observation source is explicitly simulated."""
        return self._manifest.get("source", {}).get("kind") == "simulation"


def check_receipt(file, request):
    """Check one expected scientific publication's identity and schema without its arrays."""
    if request is None:
        return None
    record = read_items(file, request.key, request.identity, ("schema",))
    if record is not None:
        require_schema(record.payload, request.schema)
    return record
