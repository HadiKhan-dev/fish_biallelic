"""Give parameters and arrays one canonical content identity (ENG-06).

Canonical JSON and dtype/shape/C-order array bytes define the content encoding.
These are content and resume keys, not a cryptographic trust boundary. Arrays are
explicitly digested rather than silently expanded into JSON.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from collections.abc import Mapping
from pathlib import PurePath

__all__ = ["canonical_json", "jsonable", "json_digest", "array_digest"]


def canonical_json(value) -> str:
    """The canonical JSON text of a JSON-compatible value (sorted keys, no spaces, no NaN)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def jsonable(value):
    """Plain JSON data for a config-like value.

    Dataclasses become dicts of their fields, Enums their values, tuples/lists lists, sets sorted lists, paths
    strings, NumPy scalars Python scalars. Arrays are rejected (digest them with array_digest), as are NaN and
    infinite floats (they have no canonical JSON form).
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return jsonable(value.value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, bool)):
                raise TypeError(f"identity mapping keys must be strings or integers, not {type(key).__name__}")
            result[str(key)] = jsonable(item)
        return result
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((jsonable(item) for item in value), key=canonical_json)
    if isinstance(value, PurePath):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("identity values must be finite numbers")
        return value
    if type(value).__module__ == "numpy":
        if getattr(value, "ndim", 0):
            raise TypeError("arrays have no identity form; digest them with array_digest")
        return jsonable(value.item())
    raise TypeError(f"no identity form for {type(value).__name__}")


def json_digest(value) -> str:
    """SHA-256 (hex) of the canonical JSON of ``jsonable(value)``."""
    return hashlib.sha256(canonical_json(jsonable(value)).encode()).hexdigest()


def array_digest(array) -> str:
    """SHA-256 (hex) of an array's dtype, shape and C-order bytes."""
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode() + b"\0" + repr(contiguous.shape).encode() + b"\0")
    if contiguous.size:  # an empty array (e.g. an empty M2 panel) has no bytes to add
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()
