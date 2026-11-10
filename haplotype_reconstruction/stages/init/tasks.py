"""Declare and verify the initialized configuration prerequisite of the stage graph."""

from ...core.errors import SettingsError
from ...store.protocol import resume
from ...store.store import CheckpointStore
from ..framework import Task, RecordKey
from . import records


def check(root, identity):
    """Verify init's immutable publication before downstream tasks run."""
    if resume(CheckpointStore(root).run, "configuration/resolved", identity) is None:
        raise SettingsError("run has not been initialized; use haplotypes init")
    return identity


def task(key, context):
    """Declare init's checked graph node without performing initialization again."""
    return Task(key, check, args=(context.root, records.expected(context.manifest, context.resolved)),
                outputs=(RecordKey("run", "configuration/resolved"),))
