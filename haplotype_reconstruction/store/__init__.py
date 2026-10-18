"""The checkpoint store: container format, codec, identities and the stage vocabulary.

Public names (imported on first access, so `store.identity`/`store.registry` stay light):
CheckpointStore, WorkState, RecordScope (store.store); CheckpointFile, RecordInfo, CheckpointMismatch
(store.container); CLEVEL, FAST_CLEVEL, PAYLOAD_SUFFIX, write_payload, read_payload (store.codec).
"""
from __future__ import annotations

import importlib

_EXPORTS = {
    "CheckpointStore": "store", "WorkState": "store", "RecordScope": "store",
    "CheckpointFile": "container", "RecordInfo": "container", "CheckpointMismatch": "container",
    "CLEVEL": "codec", "FAST_CLEVEL": "codec", "PAYLOAD_SUFFIX": "codec",
    "write_payload": "codec", "read_payload": "codec",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
