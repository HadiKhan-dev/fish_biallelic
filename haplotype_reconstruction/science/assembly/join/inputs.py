"""Expose the canonical numerical input of one joined batch.

Owns: the canonical JoinInputs re-export.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: docs/methods/assembly.md#aj01.
Invariant: Input sample/site order and readonly array contracts come from contracts.assembly.
"""

from haplotype_reconstruction.contracts.assembly import JoinInputs

__all__ = ["JoinInputs"]
