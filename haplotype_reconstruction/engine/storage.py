"""Choose equivalent array storage from dimensions, with explicit borrowed lifetimes.

Routes change cost only. Pressure never chooses a route or a scientific work
allowance. A consumer must provide forced-route equality receipts before using
more than one route. All array views are read-only and borrowed until close.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..core.errors import InvariantError, SettingsError

__all__ = [
    "StorageRoute",
    "StorageDimensions",
    "StoragePolicy",
    "StoragePlan",
    "ArrayCapability",
    "WindowCapability",
    "select_route",
    "open_array",
]


class StorageRoute(StrEnum):
    RESIDENT = "resident"
    SHARED = "shared"
    MAPPED = "mapped"
    WINDOWED = "windowed"


@dataclass(frozen=True)
class StorageDimensions:
    shape: tuple[int, ...]
    dtype: str
    consumers: int = 1
    window_axis: int = 0

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * np.dtype(self.dtype).itemsize

    def __post_init__(self) -> None:
        if any(n < 0 for n in self.shape) or self.consumers < 1:
            raise SettingsError("invalid storage dimensions")
        if np.dtype(self.dtype).hasobject:
            raise SettingsError("storage capabilities do not transport object arrays")
        if self.shape and not 0 <= self.window_axis < len(self.shape):
            raise SettingsError("window axis is outside the array")


@dataclass(frozen=True)
class StoragePolicy:
    """Cost-only thresholds supplied by a consumer's validated route plan.

    There are no model defaults: consumers explicitly supply their dimension
    limits and window width. None means the corresponding route is unavailable.
    """

    resident_bytes: int | None
    shared_bytes: int | None
    mapped_bytes: int | None
    window_width: int

    def __post_init__(self) -> None:
        if self.window_width < 1 or any(
            x is not None and x < 0 for x in (self.resident_bytes, self.shared_bytes, self.mapped_bytes)
        ):
            raise SettingsError("invalid cost-only storage thresholds")


@dataclass(frozen=True)
class StoragePlan:
    dimensions: StorageDimensions
    route: StorageRoute
    window_width: int


def select_route(
    dimensions: StorageDimensions, policy: StoragePolicy, *, force: StorageRoute | None = None
) -> StoragePlan:
    """Choose using byte dimensions/consumer count only; force is an equality-test seam."""
    if force is not None:
        return StoragePlan(dimensions, StorageRoute(force), policy.window_width)
    nbytes = dimensions.nbytes
    if policy.resident_bytes is not None and nbytes <= policy.resident_bytes and dimensions.consumers == 1:
        route = StorageRoute.RESIDENT
    elif policy.shared_bytes is not None and nbytes <= policy.shared_bytes:
        route = StorageRoute.SHARED
    elif policy.mapped_bytes is not None and nbytes <= policy.mapped_bytes:
        route = StorageRoute.MAPPED
    else:
        route = StorageRoute.WINDOWED
    return StoragePlan(dimensions, route, policy.window_width)


def _record_route(plan: StoragePlan) -> None:
    from .records import record_storage_route

    dimensions = plan.dimensions
    record_storage_route(
        route=plan.route.value,
        shape=dimensions.shape,
        dtype=dimensions.dtype,
        consumers=dimensions.consumers,
        window_axis=dimensions.window_axis,
        window_width=plan.window_width,
    )


class ArrayCapability:
    """A read-only borrowed array; each access checks that its owner is still open."""

    def __init__(self, array: NDArray[Any], plan: StoragePlan):
        self._array: NDArray[Any] | None = array.view()
        self._array.flags.writeable = False
        self.plan = plan
        _record_route(plan)

    @property
    def view(self) -> NDArray[Any]:
        if self._array is None:
            raise InvariantError("storage capability is closed")
        return self._array

    def windows(self) -> Iterator[tuple[slice, NDArray[Any]]]:
        array = self.view
        axis = self.plan.dimensions.window_axis
        for start in range(0, array.shape[axis], self.plan.window_width):
            section = slice(start, min(array.shape[axis], start + self.plan.window_width))
            selection = [slice(None)] * array.ndim
            selection[axis] = section
            # Validate on every yield; a generator cannot extend its owner's life.
            yield section, self.view[tuple(selection)]

    def close(self) -> None:
        self._array = None


class WindowCapability:
    """Identity-checked external windows supplied by a protocol-owning stage.

    The reader is called only inside this lifetime. It must verify its expected
    record identity (store.protocol.open_windows/read_items) and return the exact
    requested axis slice. Each returned array is borrowed until the next read.
    """

    def __init__(self, plan: StoragePlan, read: Callable[[slice], NDArray[Any]]):
        self.plan, self._read, self._open = plan, read, True
        _record_route(plan)

    def windows(self) -> Iterator[tuple[slice, NDArray[Any]]]:
        dimensions = self.plan.dimensions
        axis = dimensions.window_axis
        for start in range(0, dimensions.shape[axis], self.plan.window_width):
            if not self._open:
                raise InvariantError("window capability is closed")
            section = slice(start, min(dimensions.shape[axis], start + self.plan.window_width))
            value = np.asarray(self._read(section))
            expected = list(dimensions.shape)
            expected[axis] = section.stop - section.start
            if value.shape != tuple(expected) or value.dtype != np.dtype(dimensions.dtype):
                raise InvariantError("window reader returned an incompatible shape or dtype")
            value = value.view()
            value.flags.writeable = False
            yield section, value

    def close(self) -> None:
        self._open = False


@contextmanager
def open_array(
    array: NDArray[Any], plan: StoragePlan, *, directory: str | Path | None = None, copy_threads: int = 1
) -> Iterator[ArrayCapability]:
    """Open one equivalent route; cleanup also runs after consumer failure.

    A windowed in-memory source is useful for forced-route controls. Production
    sources use WindowCapability with selective protocol readers, avoiding a full
    materialization. Resident affinity lifetimes are retained by engine.workers.
    """
    array = np.asarray(array)
    if array.shape != plan.dimensions.shape or array.dtype != np.dtype(plan.dimensions.dtype):
        raise SettingsError("array dimensions disagree with the storage plan")
    handle = mapped = temporary = capability = None
    try:
        if plan.route == StorageRoute.SHARED:
            from .workers import create_shared_array

            handle, _ = create_shared_array(array, copy_threads=copy_threads)
            value = np.ndarray(array.shape, dtype=array.dtype, buffer=handle.buf)
        elif plan.route == StorageRoute.MAPPED and array.nbytes:
            temporary = tempfile.NamedTemporaryFile(
                prefix="haplotypes-array-", suffix=".mmap", dir=directory, delete=False
            )
            temporary.close()
            mapped = np.memmap(temporary.name, dtype=array.dtype, mode="w+", shape=array.shape)
            np.copyto(mapped, array)
            mapped.flush()
            cast(Any, mapped)._mmap.close()
            mapped = np.memmap(temporary.name, dtype=array.dtype, mode="r", shape=array.shape)
            value = mapped
        else:
            value = array
        capability = ArrayCapability(value, plan)
        yield capability
    finally:
        if capability is not None:
            capability.close()
        if mapped is not None:
            cast(Any, mapped)._mmap.close()
        if temporary is not None:
            Path(temporary.name).unlink(missing_ok=True)
        if handle is not None:
            handle.close()
            handle.unlink()
