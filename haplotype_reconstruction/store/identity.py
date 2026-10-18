"""Identities: canonical JSON, digests, code identity and the stage identity helper (the only hasher).

A stage record's identity says exactly what produced it:

    {"stage", "schema", "config", "upstream", "code"}

``config`` is the stage's frozen config (all fields scientific); ``upstream`` maps a name to the
``RecordInfo.identity`` of every upstream record the stage read (unique per committed record, so a rewritten
input invalidates its consumers); ``code`` is the SHA-256 of the source files of the packages the stage executes
plus ``model/`` and ``maths/`` (computed once per process). Execution, store, pipeline and report code never
enter an identity: results do not depend on them. The whole-package digest (`source_digest`) is recorded per
run for provenance only.

This module imports nothing heavy at module level: the package bootstrap uses its source walker at import time.
"""
from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import enum
import hashlib
import json
import os
from pathlib import Path, PurePath

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Packages every stage's code identity includes (shared data model and numerics).
SHARED_CODE = ("model", "maths")


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
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def add_sources(digest, directory: Path, root: Path = PACKAGE_ROOT) -> None:
    """Feed every .py file under ``directory`` (path relative to ``root``, then bytes) into ``digest``."""
    for current, subdirectories, files in os.walk(directory):
        subdirectories[:] = sorted(name for name in subdirectories if name != "__pycache__")
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(current, name)
                digest.update(str(os.path.relpath(path, root)).encode("utf-8", "surrogateescape") + b"\0")
                with open(path, "rb") as handle:
                    digest.update(handle.read())


_CODE_IDENTITIES: dict[tuple[str, ...], str] = {}


def code_identity(*packages: str) -> str:
    """SHA-256 (hex) of the sources of ``haplotype_reconstruction/<package>/`` for each package, plus model/ and
    maths/ (cached per process)."""
    names = tuple(sorted(set(packages) | set(SHARED_CODE)))
    cached = _CODE_IDENTITIES.get(names)
    if cached is None:
        digest = hashlib.sha256()
        for name in names:
            directory = PACKAGE_ROOT / name
            if not directory.is_dir():
                raise ValueError(f"no package haplotype_reconstruction.{name}")
            add_sources(digest, directory)
        cached = _CODE_IDENTITIES[names] = digest.hexdigest()
    return cached


def source_digest() -> str:
    """SHA-256 (hex) of the whole package source (run provenance)."""
    cached = _CODE_IDENTITIES.get(("*",))
    if cached is None:
        digest = hashlib.sha256()
        add_sources(digest, PACKAGE_ROOT)
        cached = _CODE_IDENTITIES[("*",)] = digest.hexdigest()
    return cached


def stage_identity(stage: str, *, schema: int, config, upstream: Mapping[str, str] | None = None,
                   packages: tuple[str, ...] | None = None) -> dict:
    """The identity of one stage record (see the module docstring).

    ``packages`` defaults to the packages the registry lists for ``stage``; a stage that runs another stage's
    code (e.g. local selection running trial assemblies) is listed with both.
    """
    from .registry import stage as registered
    if packages is None:
        packages = registered(stage).packages
    return dict(
        stage=str(stage),
        schema=int(schema),
        config=jsonable(config),
        upstream={str(name): str(identity) for name, identity in sorted((upstream or {}).items())},
        code=code_identity(*packages),
    )


def bound_identity(file, section: str) -> dict | None:
    """The identity bound to ``identity/<section>`` of a container, or None if none is bound."""
    key = f"identity/{section}"
    return file.get(key) if file.has(key) else None
