"""What one joined batch reads (built by the hierarchy for every batch it joins)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...model.components import ComponentPanel


@dataclass(frozen=True)
class JoinInputs:
    """The input of one join: the batch's input components and their evidence.

    blocks: the batch's input components (n >= 2), materialised over the prepared panels, in
        genomic order. A join reads from each block: ``positions``, ``kept``, ``inference`` (every
        likelihood and the count-rule geometry), ``source_rows`` (two founders copying the same
        prepared row share one latent allele) and ``founders``; never the published ``released``
        calls (completion fills).
    gl: float32[S, L, 3] the neutral evidence (``AssemblyEvidence.scoring``) of the batch's
        informative samples (rows in ascending sample index) at the blocks' sites, the blocks
        concatenated in order (L = total sites of the blocks). S is also the count rule's sample
        count (founder cost and bootstrap population): only informative samples enter it.
    offsets: int64[n + 1]; block j's sites are the columns offsets[j] .. offsets[j + 1] - 1 of gl.
    samples: int64[S] the informative samples' indices on the contig sample axis (S >= 1).
    """

    blocks: tuple[ComponentPanel, ...]
    gl: np.ndarray
    offsets: np.ndarray
    samples: np.ndarray

    def block_gl(self, index: int) -> np.ndarray:
        """float32[S, L_j, 3] view of block ``index``'s evidence columns."""
        return self.gl[:, self.offsets[index]:self.offsets[index + 1]]

    @property
    def sample_count(self) -> int:
        return int(self.gl.shape[0])
