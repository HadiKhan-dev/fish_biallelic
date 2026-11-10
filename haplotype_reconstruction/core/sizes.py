"""Shallow Python object sizes for memory accounting, so science can size its caches without importing ``sys``."""

from __future__ import annotations

import sys

__all__ = ["shallow_bytes"]


def shallow_bytes(value: object) -> int:
    """Bytes of ``value``'s own allocation (sys.getsizeof), excluding objects it references; an ndarray that owns its
    data includes the buffer."""
    return sys.getsizeof(value)
