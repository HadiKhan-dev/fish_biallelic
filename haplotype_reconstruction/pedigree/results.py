"""Pedigree result tables, support summaries and readable exports."""
from __future__ import annotations


class PedigreeResult:
    """Primary relationships and attached model/evidence diagnostics.

    Parent fields refer only to observed candidates. Zero observed parents is
    distinct from unresolved parent state and from biological founder status.
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
