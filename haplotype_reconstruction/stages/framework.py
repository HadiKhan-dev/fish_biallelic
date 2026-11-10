"""Give stages a small task and record-adapter protocol for graph execution.

Stages declare units, dependencies and stored products; engine.graph executes them.
Adapters alone translate frozen domain records to codec payloads. There is no stage
registry, inheritance framework or implicit settings lookup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from ..contracts.roles import require_schema
from ..engine.graph import ALL_CORES, Expansion, RecordKey, ResourceRequest, ResultRef, Task, TaskKey
from ..store.container import CheckpointFile, RecordInfo
from ..store.protocol import publish, resume

__all__ = [
    "Expansion",
    "Task",
    "TaskKey",
    "RecordKey",
    "ALL_CORES",
    "ResourceRequest",
    "ResultRef",
    "Stage",
    "RecordAdapter",
    "publish_product",
    "load_product",
]

if TYPE_CHECKING:
    from ..store.store import CheckpointStore, ScatterView

Product = TypeVar("Product")
Configuration = TypeVar("Configuration", contravariant=True)


class Stage(Protocol[Configuration]):
    """A stage supplies concrete tasks; the engine owns their execution order and resources."""

    def tasks(self, resolved: Configuration, store_view: CheckpointStore | ScatterView) -> Sequence[Task]:
        """Declare the stage's units, dependencies, input/output record keys and resources."""
        ...


class RecordAdapter(Protocol[Product]):
    """Translate one product schema between domain records and plain codec payloads."""

    @property
    def schema(self) -> int:
        """The integer payload schema (read-only: frozen adapters satisfy it too)."""
        ...

    def encode(self, product: Product) -> Mapping[str, Any]:
        """Return a payload carrying the integer schema and documented product fields."""
        ...

    def decode(self, payload: Mapping[str, Any]) -> Product:
        """Reconstruct the frozen product with its declared axes, dtypes and ownership."""
        ...


def publish_product(
    file: CheckpointFile, key: str, identity: str, product: Product, adapter: RecordAdapter[Product]
) -> RecordInfo:
    """Validate the adapter's schema and atomically publish a product by derivation identity."""
    payload = adapter.encode(product)
    require_schema(payload, adapter.schema)
    return publish(file, key, identity, payload)


def load_product(file: CheckpointFile, key: str, identity: str, adapter: RecordAdapter[Product]) -> Product | None:
    """Resume a matching product through its versioned adapter, or return None if absent."""
    record = resume(file, key, identity)
    if record is None:
        return None
    require_schema(record.payload, adapter.schema)
    return adapter.decode(record.payload)
