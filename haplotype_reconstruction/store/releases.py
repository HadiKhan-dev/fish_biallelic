"""Identify this release's stored runs and exports before decoding their contents."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.release import CURRENT_RELEASE

if TYPE_CHECKING:
    from .container import CheckpointMismatch


def unsupported_release(path: Any, release: Any = None) -> CheckpointMismatch:
    """One actionable refusal, preserving recorded provenance without guessing missing versions."""
    from .container import CheckpointMismatch

    if release is None or isinstance(release, (int, float, bool)):
        writer = "an earlier release (exact writer release not recorded)"
        if release is not None:
            writer += f" [checkpoint layout {release!r}]"
    else:
        writer = f"release {release}"
    return CheckpointMismatch(f"{path}: written by {writer}; it must be re-run with {CURRENT_RELEASE} in a new output root")


def require_current_release(document: Mapping[str, Any], path: Any) -> None:
    """Require the writer's explicit release, independently of a record's payload schema."""
    release = document.get("release")
    if release != CURRENT_RELEASE:
        raise unsupported_release(path, release)


def check_directory_release(directory: Path) -> None:
    """Inspect only export/invocation provenance before routing a run or loading settings."""
    manifest = directory / "manifest.json"
    if manifest.is_file():
        require_current_release(json.loads(manifest.read_text()), manifest)
    for path in sorted((directory / "run_records").glob("*.json")):
        document = json.loads(path.read_text())
        require_current_release(document, path)
        if document.get("schema") != "run-record-v5" or "tasks" in document:
            raise unsupported_release(path, document.get("release"))
    checkpoints = directory / "checkpoints"
    if checkpoints.is_dir():
        from .store import CheckpointStore

        CheckpointStore(checkpoints).check_layout()
