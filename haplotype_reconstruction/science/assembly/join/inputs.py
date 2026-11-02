"""Expose the canonical numerical input of one joined batch.

Owns: the canonical JoinInputs re-export.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: WP_AS4 §5; AJ01.
Identity: Input sample/site order and readonly array contracts come from contracts.assembly.
"""

from haplotype_reconstruction.contracts.assembly import JoinInputs

__all__ = ["JoinInputs"]
