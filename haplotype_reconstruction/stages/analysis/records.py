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
ANALYSIS_STAGES = FINAL_STAGES | {
    "observation_fits", "observation", "relatedness", "discovery", "selection", "assembly",
    "calibration_robustness", "calibration_refine", "calibration_outlier", "calibration_share",
    "calibration_divisor_fit", "calibration_divisor", "calibration",
    "painting", "contamination_cells", "contamination",
    "refit_histogram", "refit_priors", "refit", "repaint", "release",
}
# The observation pilot's rate and derivation feed relatedness and reconstruction.
# Their keys include variable block, context, and fit suffixes.
# The painting prefix also covers contamination cells and the run decision they determine.
_SCOPED_PREFIXES = (
    "observation", "relatedness", "discovery", "selection", "local_selection",
    "assembly", "painting", "identity/painting", "founder_refit", "identity/founder_refit", "release",
)
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
    return StoreLocation(
        str(root), selection.namespace, scoped_keys(), dict(analysis=selection.document()), _SCOPED_PREFIXES,
    )


def output_directory(output: str | PathLike, selection: AnalysisContigs) -> Path:
    return Path(output) if not selection.excluded else Path(output) / "analyses" / selection.namespace
