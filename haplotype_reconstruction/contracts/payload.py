"""Translate declared stored fields to plain payloads and rebuild typed records without I/O.

Owns the generic codec for ordinary dataclasses (WP_D1 §4.7). Field annotations
supply nested types; no class names, registry or package instances enter payloads.
Products put schema first. Derived fields are supplied by the stage adapter and
are never encoded. Products with hoisted items, such as Painting, have explicit
codecs in their own contract module. This module does not read storage or infer
hoisted items. Arrays decoded here are read-only views; no input buffer is changed.
"""

from __future__ import annotations

import functools
import types
from collections.abc import Callable, Mapping
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

import numpy as np
from numpy.typing import NDArray

from ..core.errors import InputError
from .roles import require_schema

__all__ = ["to_payload", "from_payload"]

T = TypeVar("T")


def _encode(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_payload(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise InputError("payload mappings require string keys")
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    raise InputError(f"unsupported payload value {type(value).__qualname__}")


def to_payload(record: Any) -> dict[str, Any]:
    """Encode the stored fields of a dataclass, schema first, without copying arrays.

    Tuples become lists, nested records become dictionaries and enums become their
    values. Unknown objects and derived-only views raise InputError. No mutation.
    """
    if not is_dataclass(record) or isinstance(record, type):
        raise InputError("to_payload needs a dataclass instance")
    if getattr(type(record), "__payload_view__", False):
        raise InputError(f"{type(record).__name__} is a derived view, not a stored record")
    if getattr(type(record), "__payload_custom__", False):
        raise InputError(f"{type(record).__name__} requires its explicit payload codec")
    names = [item.name for item in fields(record) if item.metadata.get("stored", True)]
    if "schema" in names:
        names.remove("schema")
        names.insert(0, "schema")
    return {name: _encode(getattr(record, name)) for name in names}


def _readonly(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        result = value.view()
        result.flags.writeable = False
        return result
    if isinstance(value, Mapping):
        return {key: _readonly(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_readonly(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_readonly(item) for item in value)
    return value


def _decode(annotation: Any, value: Any) -> Any:
    return _decoder(annotation)(value)


# Decoders by annotation object: id -> (annotation, decoder). Keyed by identity, not equality: typing's equality
# ignores the order of Union and Literal arguments, while decoding tries union members in their declared order (the
# first match wins and failures are reported in that order). Each entry keeps its annotation alive, so an id is never
# reused while its entry exists. The annotations reaching here are those of _record_layout's cached hints and their
# arguments: a fixed set per process.
_DECODERS: dict[int, tuple[Any, Callable[[Any], Any]]] = {}


def _decoder(annotation: Any) -> Callable[[Any], Any]:
    """The decoding function of one annotation, built once: an annotation's structure never changes, and inspecting
    it for every value (typing reflection per element) dominated loading records of many small items. Checks, their
    order and their errors are those of decoding the value against the annotation directly."""
    entry = _DECODERS.get(id(annotation))
    if entry is None or entry[0] is not annotation:
        entry = _DECODERS[id(annotation)] = (annotation, _build_decoder(annotation))
    return entry[1]


def _build_decoder(annotation: Any) -> Callable[[Any], Any]:
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        return _readonly
    if origin in (Union, types.UnionType):
        optional = type(None) in args
        choices = [arg for arg in args if arg is not type(None)]

        def decode_union(value: Any) -> Any:
            if value is None and optional:
                return None
            failures = []
            for choice in choices:
                try:
                    return _decoder(choice)(value)
                except (InputError, TypeError, ValueError) as error:
                    failures.append(str(error))
            raise InputError(f"payload value matches no union member: {'; '.join(failures)}")

        return decode_union
    if annotation is type(None):

        def decode_none(value: Any) -> None:
            if value is not None:
                raise InputError("expected None")
            return None

        return decode_none
    if isinstance(annotation, type) and is_dataclass(annotation):

        def decode_record(value: Any) -> Any:
            if not isinstance(value, Mapping):
                raise InputError(f"{annotation.__name__} needs a mapping")
            return from_payload(annotation, value)

        return decode_record
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    if annotation in (np.ndarray, NDArray) or origin in (np.ndarray, NDArray):

        def decode_array(value: Any) -> np.ndarray:
            if not isinstance(value, np.ndarray):
                raise InputError("array field needs an ndarray payload")
            return _readonly(value)

        return decode_array
    if origin in (tuple, list):
        build = tuple if origin is tuple else list
        if not args:
            items = None
        elif origin is tuple and len(args) > 1 and args[-1] is not Ellipsis:
            items = tuple(args)
        else:
            items = args[0]

        def decode_sequence(value: Any) -> Any:
            if not isinstance(value, (tuple, list)):
                raise InputError("sequence field needs a sequence payload")
            if items is None:
                return build([_readonly(item) for item in value])
            if isinstance(items, tuple):
                if len(value) != len(items):
                    raise InputError(f"tuple has {len(value)} items, expected {len(items)}")
                return build([_decoder(typ)(item) for typ, item in zip(items, value, strict=True)])
            decode_item = _decoder(items)
            return build([decode_item(item) for item in value])

        return decode_sequence
    if origin in (dict, Mapping) or annotation in (dict, Mapping):
        item_type = args[1] if args else Any

        def decode_mapping(value: Any) -> dict[str, Any]:
            if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
                raise InputError("mapping field needs string keys")
            decode_item = _decoder(item_type)
            return {key: decode_item(item) for key, item in value.items()}

        return decode_mapping
    if origin is Literal:

        def decode_literal(value: Any) -> Any:
            if value not in args:
                raise InputError(f"{value!r} is not in {args}")
            return value

        return decode_literal
    if annotation in (str, int, float, bool, bytes):

        def decode_scalar(value: Any) -> Any:
            # Python's float annotation accepts integers. Preserve their exact stored
            # representation; converting a large integer to float would lose bits.
            if annotation is float and type(value) is int:
                return value
            if type(value) is not annotation:
                raise InputError(f"expected {annotation.__name__}, got {type(value).__name__}")
            return value

        return decode_scalar

    def no_decoder(value: Any) -> Any:
        raise InputError(f"no payload decoder for annotation {annotation!r}")

    return no_decoder


@functools.cache
def _record_layout(cls: type) -> tuple[dict[str, Any], frozenset[str], dict[str, Any]]:
    """A record class's fields by name, its stored field names and its resolved annotations, once per class (classes do
    not change; resolving annotations for every record dominated decoding thousands of small records)."""
    declared = {item.name: item for item in fields(cls)}
    stored = frozenset(name for name, item in declared.items() if item.metadata.get("stored", True))
    return declared, stored, get_type_hints(cls)


def from_payload(cls: type[T], payload: Mapping[str, Any], *, derived: Mapping[str, Any] | None = None) -> T:
    """Restore a record from exactly its stored fields and optional adapter-derived fields.

    Checks schema and rejects missing, unknown or persisted derived fields. Nested
    records and tuples follow annotations; arrays are read-only. No I/O or mutation.
    """
    if not is_dataclass(cls) or getattr(cls, "__payload_view__", False):
        raise InputError(f"{cls.__name__} is not a stored dataclass record")
    if getattr(cls, "__payload_custom__", False):
        raise InputError(f"{cls.__name__} requires its explicit payload codec")
    declared, stored, hints = _record_layout(cls)
    if "schema" in stored:
        version = declared["schema"].default
        if version is MISSING:
            raise InputError(f"{cls.__name__}.schema must declare a version default")
        require_schema(payload, version)
    missing, unknown = stored - payload.keys(), payload.keys() - stored
    if missing or unknown:
        raise InputError(f"{cls.__name__} payload fields: missing {sorted(missing)}, unknown {sorted(unknown)}")
    supplied = dict(derived or {})
    invalid = supplied.keys() - {
        name for name, item in declared.items() if not item.metadata.get("stored", True) and item.init
    }
    if invalid:
        raise InputError(f"{cls.__name__} unknown derived fields {sorted(invalid)}")
    values = {name: _decode(hints[name], value) for name, value in payload.items() if declared[name].init}
    values.update({name: _readonly(value) for name, value in supplied.items()})
    record = cls(**values)
    for name in stored:
        if declared[name].init:
            continue
        expected = _encode(getattr(record, name))
        actual = payload[name]
        equal = (
            np.array_equal(expected, actual, equal_nan=True) if isinstance(expected, np.ndarray) else expected == actual
        )
        if not equal:
            raise InputError(f"{cls.__name__}.{name} disagrees with its reconstructed value")
    return record
