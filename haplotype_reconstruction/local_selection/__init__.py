"""Local path selection: the initial normalised-path fit of every discovery block, then context rounds l1 and
l2 (context assembly through assembly's CONTEXT role, context refit, refit of every block) (contracts/T3.md).

Public entry: :func:`run_contig`; readers in :mod:`.records` (``load_panels``, ``panels_identity``,
``founder_panels``, ``context_assembly_prefix``). Imports assembly; assembly never imports this package.
"""
from . import records
from .config import LocalSelectionConfig
from .stage import run_contig

__all__ = ["records", "LocalSelectionConfig", "run_contig"]
