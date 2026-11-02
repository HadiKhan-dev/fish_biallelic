"""Arrays declared immutable at birth, which the engine's task transport may share by identity.

Read-only flags alone cannot show that an array will not change: a view made before the freeze stays writable, and
a read-only view over a mutable buffer changes with it. A producer that builds an array that owns its data, freezes
it before any view of it exists and never changes it again declares it here; the transport then shares it once
instead of pickling it into every task.
"""

from __future__ import annotations

import weakref

import numpy as np

_DECLARED: dict[int, weakref.ref] = {}


def declare_immutable(array: np.ndarray) -> np.ndarray:
    """Freeze ``array`` (which owns its data and has no other view yet) and declare that it never changes."""
    if array.base is not None:
        raise ValueError("only an array that owns its data can be declared immutable")
    array.flags.writeable = False
    key = id(array)

    def forget(_: weakref.ref, key: int = key) -> None:
        _DECLARED.pop(key, None)

    _DECLARED[key] = weakref.ref(array, forget)
    return array


def declared_immutable(array: np.ndarray) -> bool:
    """Whether ``array``'s producer declared it immutable (declare_immutable)."""
    ref = _DECLARED.get(id(array))
    return ref is not None and ref() is array
