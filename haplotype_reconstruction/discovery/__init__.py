"""Raw block discovery: per discovery block, an adaptive reversible search over hard fixed-K panels scored by
a leave-one-out cavity predictive, then the raw allele release (contracts/T3.md).

Public entry: :func:`run` (all contigs on one block pool); :func:`discover_block` for one block's
:class:`BlockEvidence`; readers in :mod:`.records` (``load``, ``complete``, ``founder_panels``).
"""
from . import records
from .config import DiscoveryConfig
from .release import DiscoveredBlock
from .stage import BlockEvidence, block_evidence, discover_block, run

__all__ = ["records", "DiscoveryConfig", "DiscoveredBlock", "BlockEvidence", "block_evidence", "discover_block",
           "run"]
