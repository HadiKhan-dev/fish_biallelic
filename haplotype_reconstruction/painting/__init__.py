"""Painting: every sample of each assembled component as a diploid mosaic of founder rows.

Entry points: :func:`run_contig` (store-centric, one contig in a lane) and
:func:`paint_chromosome` (in memory). Product: :class:`ChromosomePainting`
(records: painting.records). Shared with pedigree: the state space
(:mod:`painting.states`), the transition table (:mod:`painting.transitions`)
and the emission kernel (:func:`painting.hmm.binned_emissions`).
"""
from .config import PaintingConfig
from .release import (ChromosomePainting, ComponentPainting, LabelContinuity, PaintedChunk,
                      RelabelMove, TrackStatus)
from .stage import paint_chromosome, run_contig
from .states import PaintingBins, PaintingStates, background_alt_frequency
from .transitions import PaintingTransitions

__all__ = [
    "ChromosomePainting", "ComponentPainting", "LabelContinuity", "PaintedChunk",
    "PaintingBins", "PaintingConfig", "PaintingStates", "PaintingTransitions", "RelabelMove",
    "TrackStatus", "background_alt_frequency", "paint_chromosome", "run_contig",
]
