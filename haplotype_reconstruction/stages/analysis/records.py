"""Own final-analysis declarations and derive key routing from the product owners."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from ...contracts.contigs import AnalysisContigs
from ...store.namespace import StoreLocation

if TYPE_CHECKING:
    from ...store.store import CheckpointStore

FINAL_STAGES = frozenset({"pedigree", "phase", "recombination", "export"})
DEFINITION = "configuration/analysis/"


def scoped_keys() -> tuple[str, ...]:
    """Aggregate each final stage's declared selection-dependent publications."""
    from ..export import records as export
    from ..pedigree import records as pedigree
    from ..phase import records as phase
    from ..recombination import records as recombination

    return tuple(key for owner in (pedigree, phase, recombination, export) for key in owner.ANALYSIS_KEYS)


def declare(store: CheckpointStore, selection: AnalysisContigs) -> None:
    """Bind scientific axes only; installed record declarations determine routing."""
    if selection.excluded:
        store.run.put_once(DEFINITION + selection.namespace, selection.document())


def location(root: str | PathLike, selection: AnalysisContigs) -> str | StoreLocation:
    if not selection.excluded:
        return str(root)
    return StoreLocation(str(root), selection.namespace, scoped_keys(), dict(analysis=selection.document()))


def output_directory(output: str | PathLike, selection: AnalysisContigs) -> Path:
    return Path(output) if not selection.excluded else Path(output) / "analyses" / selection.namespace
