"""Re-export shared genotype emissions and the prepared-pair triangle address.

Genotype emissions come from science.shared.emissions; pair order comes from assembly.pairs.
Must not choose a numerical formula or perform storage. Method:
docs/methods/founder_refinement.md#emissions (RC03).
"""

from haplotype_reconstruction.science.shared.emissions import (
    CENTER,
    GENOTYPE_CODES,
    PREDICTIVE_CODES,
    normalised_likelihoods,
    pair_code,
    predictive_likelihood,
    robust_log_likelihood,
)

from ..pairs import emission_pair_index as pair_index

__all__ = [
    "CENTER",
    "GENOTYPE_CODES",
    "PREDICTIVE_CODES",
    "pair_code",
    "predictive_likelihood",
    "normalised_likelihoods",
    "robust_log_likelihood",
    "pair_index",
]
