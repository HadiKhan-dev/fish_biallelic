"""Neutral result payload used by pedigree inference."""


class PedigreeResult:
    """Mutable inference result with the historical core field schema.

    Engines may attach model-specific diagnostics after construction. The
    container deliberately does not implement the retired cutoff or cycle heuristics.
    """

    def __init__(
        self,
        samples,
        relationships,
        parent_candidates,
        recombination_map,
        systematic_errors,
        kinship_matrix,
        ibd0_matrix,
        trio_scores=None,
        total_bins=0,
    ):
        self.samples = samples
        self.relationships = relationships
        self.parent_candidates = parent_candidates
        self.recombination_map = recombination_map
        self.systematic_errors = systematic_errors
        self.kinship_matrix = kinship_matrix
        self.ibd0_matrix = ibd0_matrix
        self.trio_scores = trio_scores if trio_scores is not None else {}
        self.total_bins = total_bins
