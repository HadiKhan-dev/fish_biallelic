"""Expose assembly's founder-allele boundary to selection.

Owns the declared selection-to-assembly surface; no execution, storage or implicit
policy is constructed here. Array kernels preserve their compiled entry points so
selection can call them from its per-site Numba loop. Settings are float64[13] from
SitePosteriorParameters.vector(reverse); info is int64[INFO_FIELDS] in INFO_* order.
Method: docs/methods/assembly.md#u09 and #u10.
"""

from .alleles.parameters import SitePosteriorParameters
from .alleles.site_posterior import (
    INFO_ASCENT_BACKSTOP,
    INFO_ASCENT_STEPS,
    INFO_BP_NONCONVERGED,
    INFO_COMPLEMENT_MODE,
    INFO_CORE,
    INFO_FIELDS,
    INFO_METHOD,
    INFO_MULTIMODAL,
    INFO_RADIUS,
    INFO_RECENTRE_BACKSTOP,
    INFO_RECENTRES,
    MAX_EXACT_BITS,
    METHOD_CORE_BALL,
    METHOD_CORE_CUBE,
    METHOD_CUBE,
    SETTING_HALF_BAND,
    SUMMARY_FIELDS,
    guard_withholds,
    settings,
    site_release_marginals,
    summarize_sites,
)
from .evidence import AssemblyEvidence, SelectionEvidence
from .run import AssemblyInputs, AssemblyResult, Role

__all__ = [
    "Role",
    "AssemblyInputs",
    "AssemblyResult",
    "AssemblyEvidence",
    "SelectionEvidence",
    "SitePosteriorParameters",
    "site_release_marginals",
    "guard_withholds",
    "summarize_sites",
    "settings",
    "MAX_EXACT_BITS",
    "METHOD_CUBE",
    "METHOD_CORE_CUBE",
    "METHOD_CORE_BALL",
    "INFO_METHOD",
    "INFO_CORE",
    "INFO_RADIUS",
    "INFO_BP_NONCONVERGED",
    "INFO_ASCENT_STEPS",
    "INFO_ASCENT_BACKSTOP",
    "INFO_RECENTRES",
    "INFO_RECENTRE_BACKSTOP",
    "INFO_COMPLEMENT_MODE",
    "INFO_MULTIMODAL",
    "INFO_FIELDS",
    "SETTING_HALF_BAND",
    "SUMMARY_FIELDS",
]
