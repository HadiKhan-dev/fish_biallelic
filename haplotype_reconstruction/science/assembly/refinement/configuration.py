"""Temporary native configuration spelling for the relocation proof.

The canonical full AS7/AS8 parameter record is RefinementParameters.
"""

from .parameters import DEFAULT_GENOTYPE_ERROR, DEFAULT_RESET_RATE, MAP_PROBABILITY_FLOOR, RefinementParameters

FounderRefinementConfig = RefinementParameters
__all__ = ["FounderRefinementConfig", "DEFAULT_GENOTYPE_ERROR", "DEFAULT_RESET_RATE", "MAP_PROBABILITY_FLOOR"]
