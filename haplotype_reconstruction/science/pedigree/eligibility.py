"""Expose the unrestricted observed-parent universe without using sample metadata (PD01).

The immutable mask contract owns its construction and queries. No scoring or I/O.
Method: docs/methods/pedigree.md#parent-candidates.
"""

from ...contracts.pedigree import POLICY, ParentEligibility

__all__ = ["POLICY", "ParentEligibility"]
