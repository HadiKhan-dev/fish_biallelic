"""Assembly: path-selected local panels -> chromosome-scale founder components.

Public entry: :func:`assemble` (final and context roles; the driver's contig wrapper calls it with
the final role and the section's record scope); readers in :mod:`.records`. Products are row paths
over immutable prepared panels (``model.components``). Subpackages: completion (founder completion), linker
(transition fitting), join (beam + certified panel search), hierarchy (levels and batches),
count_rule (the shared founder-count rule), founder_alleles (founder-allele site posteriors, also
used by local selection), founder_refinement (after every final level).
"""
from .config import AssemblyConfig, JoinConfig, LinkerConfig
from .evidence import AssemblyEvidence
from .stage import AssemblyInputs, AssemblyResult, LevelRecord, Role, assemble

__all__ = ["AssemblyConfig", "JoinConfig", "LinkerConfig", "AssemblyEvidence", "AssemblyInputs",
           "AssemblyResult", "LevelRecord", "Role", "assemble"]
