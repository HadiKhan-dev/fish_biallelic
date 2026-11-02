"""Expose the discovery science that local selection reuses.

Owns only the declared selection-to-discovery boundary (WP_DI3 section 4.6).
Methods: docs/methods/discovery.md, DI03-DI10 and DI26. The fitter retains ordered
workspace history; the cavity scorer and allele release read the final membership.
This module must not schedule work, read records or introduce a second implementation.
"""

from ...core.ties import BlockQuanta
from .cavity import CavityWorkspace, score_solutions
from .fixed_k.fit import fit_panels, prepare_workspace
from .fixed_k.workspace import search_bytes
from .panel import FixedKSolution, deduplicate, log_haplotype_set_count, panel_key, solution_tie_mask, unique_rows
from .parameters import DiscoveryParameters
from .release import release_alleles

__all__ = [
    "prepare_workspace",
    "fit_panels",
    "search_bytes",
    "CavityWorkspace",
    "score_solutions",
    "release_alleles",
    "deduplicate",
    "panel_key",
    "solution_tie_mask",
    "unique_rows",
    "log_haplotype_set_count",
    "FixedKSolution",
    "BlockQuanta",
    "DiscoveryParameters",
]
