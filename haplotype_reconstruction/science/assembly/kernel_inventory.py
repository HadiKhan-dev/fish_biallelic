"""Register assembly's lazy numerical entry points before engine warm-up.

Imports create kernel declarations only: no compilation, cache directories or
scientific work. The stage preload and ordinary application use this same static
closure, so a bounded replay does not silently omit join or polish kernels.
The engine owns parallel compilation, cache publication and admission.
"""

from . import pairs
from .join import certify, emissions, objective, pair_segments, proposals, splice
from .linker import emissions as linker_emissions
from .linker import kernels as linker_kernels
from .linker import scan as linker_scan
from .refinement import acceptance, beam, checkpoints, component_workspace, count, gains, guard, messages, window
from .refinement import objective as refinement_objective
from .refinement import proposals as refinement_proposals

# The module objects make the imported registration closure explicit and keep
# static code identities and runtime preload inventories on the same boundary.
MODULES = (
    pairs,
    certify,
    emissions,
    objective,
    pair_segments,
    proposals,
    splice,
    linker_emissions,
    linker_kernels,
    linker_scan,
    acceptance,
    beam,
    checkpoints,
    component_workspace,
    count,
    gains,
    guard,
    messages,
    refinement_objective,
    refinement_proposals,
    window,
)

__all__ = ["MODULES"]
