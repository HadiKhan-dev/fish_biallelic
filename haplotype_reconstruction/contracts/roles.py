"""Make field meaning inspectable by comparators and contract tests without a registry.

Frozen product dataclasses use ordinary dataclasses.field metadata: role, axes,
units, dtype and stored (CONTRACTS_40 §0; WP_D1 §4.7). Stored defaults to True.
Adapters encode only stored fields; derived fields and views declare stored=False
and are never written. Their record documentation names the reconstruction inputs.
Products carry a stored integer schema field; readers check supported versions.
Array ownership and cache-view lifetimes remain part of the owning record's contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields
from enum import StrEnum
from typing import Any

import numpy as np

from ..core.errors import InputError

__all__ = [
    "FieldRole",
    "FieldSpec",
    "ROLES",
    "record_field",
    "schema_field",
    "field_specs",
    "require_schema",
    "freeze_record_arrays",
]


class FieldRole(StrEnum):
    """The six field roles shared by contracts and scientific comparison."""

    OBS = "obs"
    EVIDENCE = "evidence"
    STATE = "state"
    CALL = "call"
    REPORT = "report"
    DIAGNOSTIC = "diagnostic"


ROLES = tuple(role.value for role in FieldRole)


@dataclass(frozen=True)
class FieldSpec:
    """Declare a field's comparison meaning and whether it belongs in the payload."""

    role: FieldRole
    axes: tuple[str, ...]
    units: str
    dtype: str
    stored: bool = True


def record_field(
    role: FieldRole | str,
    *,
    axes: tuple[str, ...] = (),
    units: str = "1",
    dtype: str,
    stored: bool = True,
    default: Any = MISSING,
    default_factory: Any = MISSING,
    kw_only: bool = False,
    init: bool = True,
    repr: bool = True,
    compare: bool = True,
) -> Any:
    """Declare field meaning using standard dataclass metadata; invalid roles raise ValueError.

    Axes follow CONTRACTS_40 §0 (N, L, B, K, ...); fixed sizes may be strings such as
    '2'. Dtype is an explicit storage type or a documented structured type name.
    stored=False marks derived fields/views that adapters must omit from payloads;
    it does not remove their comparison metadata. Defaults retain ordinary dataclass
    behavior. Nothing is registered, wrapped or automatically serialized.
    The Any return annotation follows dataclasses.field: its runtime Field becomes
    the annotated record attribute, so static checkers must not treat it as Field.
    """
    return field(
        default=default,
        default_factory=default_factory,
        kw_only=kw_only,
        init=init,
        repr=repr,
        compare=compare,
        metadata={"role": FieldRole(role).value, "axes": tuple(axes), "units": units, "dtype": dtype, "stored": stored},
    )


def schema_field(version: int) -> Any:
    """Declare a keyword-only integer schema field on a product, with diagnostic role.

    The version is checked separately from scientific comparison. Keyword-only avoids
    imposing field-order restrictions on the scientific payload of a dataclass.
    """
    return record_field(FieldRole.DIAGNOSTIC, dtype="int", default=version, kw_only=True)


def field_specs(record: Any) -> dict[str, FieldSpec]:
    """Read all field specs, including derived fields; absent stored metadata defaults to True.

    Missing role/axes/units/dtype metadata raises ValueError. Inputs are not mutated.
    """
    result = {}
    for item in fields(record):
        try:
            result[item.name] = FieldSpec(
                FieldRole(item.metadata["role"]),
                tuple(item.metadata["axes"]),
                item.metadata["units"],
                item.metadata["dtype"],
                item.metadata.get("stored", True),
            )
        except KeyError as error:
            raise ValueError(f"{item.name}: missing field metadata {error.args[0]}") from error
    return result


def require_schema(record: Any, supported: int | tuple[int, ...]) -> int:
    """Check a decoded product's schema and return it, or raise InputError naming the version.

    Accepts a payload mapping or a dataclass instance. Readers choose supported
    versions explicitly; there is no implicit migration or schema registry.
    """
    version = record.get("schema") if isinstance(record, Mapping) else getattr(record, "schema", None)
    versions = (supported,) if isinstance(supported, int) else supported
    if type(version) is not int or version not in versions:
        raise InputError(f"schema {version!r} is unsupported; expected {versions}")
    return version


def freeze_record_arrays(record: Any) -> None:
    """Publish read-only array views in a just-constructed frozen dataclass.

    Dtypes, values and order are unchanged. Writable producer buffers retain their
    flags; callers that mutate published data must copy. Already immutable arrays
    are shared by identity. Nested records enforce their own boundaries.
    """

    def freeze(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if not value.flags.writeable:
                return value
            result = value.view()
            result.flags.writeable = False
            return result
        if isinstance(value, tuple):
            return tuple(freeze(item) for item in value)
        if isinstance(value, list):
            return [freeze(item) for item in value]
        if isinstance(value, Mapping):
            return {key: freeze(item) for key, item in value.items()}
        return value

    for item in fields(record):
        object.__setattr__(record, item.name, freeze(getattr(record, item.name)))
