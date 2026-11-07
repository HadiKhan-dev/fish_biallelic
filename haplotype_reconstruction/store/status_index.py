"""Optional container-header snapshots for status, never scientific resume authority.

Stage publication writes these disposable indexes once, after checked completion.
Status can load them and scan only subsequent appends. Missing, partial or stale
snapshots fall back to the ordinary header scanner. Status itself writes nothing.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ..core.files import atomic_path
from .container import CheckpointFile, RecordInfo, _index, _parse_file_header, _Window

_LOG = logging.getLogger(__name__)


def snapshot_path(file: CheckpointFile) -> Path:
    """A disposable index beside its container, inside the checkpoint root."""
    return Path(file.path + ".index.json")


def save_index(file: CheckpointFile) -> None:
    """Snapshot already indexed headers at a publication boundary, not per task.

    Concurrent publications may replace a newer snapshot with an older prefix;
    either remains correct because status scans any appended tail. Cache I/O
    failure cannot invalidate a scientific publication.
    """
    index = _index(file.path)
    try:
        with index.lock:
            file._refresh(index)
            if index.header is None:
                return
            document = dict(
                schema=1, header=index.header, end=index.end, next_seq=index.next_seq,
                records=[list(info.__dict__.values()) for info in index.records.values()],
            )
        with atomic_path(snapshot_path(file)) as temporary:
            temporary.write_text(json.dumps(document, separators=(",", ":")) + "\n")
    except OSError as error:
        _LOG.warning("Could not cache status index for %s: %s", file.label, error)


def load_index(file: CheckpointFile) -> None:
    """Prime only status's process-local index; verify container identity and size.

    Checkpoint containers commit append-only prefixes and give each newly created
    file a unique header identity. A copied prefix can reuse its snapshot; replacing
    or truncating a container invalidates it. Resume never calls this function.
    """
    index = _index(file.path)
    with index.lock:
        if index.header is not None:
            return  # Already scanned in this process.
        try:
            document = json.loads(snapshot_path(file).read_text())
            if document["schema"] != 1:
                return
            with open(file.path, "rb") as handle:
                status = os.fstat(handle.fileno())
                header, _ = _parse_file_header(_Window(handle.fileno(), status.st_size), file.path)
            if header != document["header"] or status.st_size < document["end"]:
                return
            records = {info.key: info for row in document["records"] for info in (RecordInfo(*row),)}
        except (OSError, ValueError, KeyError, TypeError):
            return  # The cache is optional; the normal scanner reports container errors.
        index.reset(status.st_ino)
        index.header = header
        index.end = document["end"]
        index.next_seq = document["next_seq"]
        index.records = records
        file._check_header(index)
