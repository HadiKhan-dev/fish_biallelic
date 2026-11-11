"""Select an immutable record namespace while sharing unscoped inputs.

The application supplies the affected keys and manifest additions. The physical
container and its append lock remain shared: no arrays are copied and every
namespace retains the ordinary publication and identity checks.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

from .container import CheckpointFile


@dataclass(frozen=True)
class StoreLocation(os.PathLike):
    root: str
    namespace: str
    keys: tuple[str, ...]
    manifest_fields: dict

    def __fspath__(self):
        return self.root


class _Names(Mapping):
    def __init__(self, records, file):
        self.records, self.file = records, file

    def __getitem__(self, key):
        return self.records[self.file.physical_key(key)]

    def __iter__(self):
        for key in self.records:
            if key.startswith(self.file.prefix):
                yield key[len(self.file.prefix) :]
            elif not key.startswith("analysis/") and key not in self.file.scoped_keys:
                yield key

    def __len__(self):
        return sum(1 for _ in self)


class NamespacedFile(CheckpointFile):
    """Expose logical keys over the selected namespace and unchanged shared keys."""

    def __init__(self, *args, location: StoreLocation, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = "analysis/" + location.namespace + "/"
        self.scoped_keys = frozenset(location.keys)

    def physical_key(self, key):
        return self.prefix + key if key in self.scoped_keys else key

    def info(self, key):
        return super().info(self.physical_key(key))

    def _records(self):
        return dict(_Names(super()._records(), self))

    @contextmanager
    def _append_lock(self):
        with super()._append_lock() as (fd, index):
            yield fd, SimpleNamespace(records=_Names(index.records, self), physical=index)

    def _append(self, fd, index, key, payload, nthreads, clevel):
        return super()._append(fd, index.physical, self.physical_key(key), payload, nthreads, clevel)
