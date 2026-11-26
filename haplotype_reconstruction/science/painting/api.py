"""Expose painting emissions, state probabilities, bins and the read state's class swap and bin weights to pedigree
and phase.

Pure numerical operations only. Transition tables and released grids are consumed
through contracts.painting; no storage, environment or execution state enters here.
"""

from .emissions import (
    binned_emissions,
    design_effect,
    gather_sites,
    mix_window,
    informative_sites,
    state_alt_probabilities,
    unpack_emissions,
)
from .read_factor import donor_ratio, swap_classes
from .read_inputs import design_weights
from .read_operator import TAPE, apply_bin, empty_tape
from .states import background_alt_frequency, build_bins, build_states

__all__ = [
    "binned_emissions",
    "design_effect",
    "gather_sites",
    "mix_window",
    "informative_sites",
    "unpack_emissions",
    "state_alt_probabilities",
    "background_alt_frequency",
    "build_states",
    "build_bins",
    "swap_classes",
    "donor_ratio",
    "design_weights",
    "TAPE",
    "apply_bin",
    "empty_tape",
]
