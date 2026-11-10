"""Hold a component's immutable arrays and caller-supplied numerical storage capabilities.

Owns the numeric workspace view and the sample-range order (RS01, RC04–RC07).
Must not choose storage routes, inspect memory, or configure runtime threads.
Method: docs/methods/founder_refinement.md#messages.
Dense message views last until update; reconstructed views last until another read
of the same segment parity. Proposal flanks are copied before parallel workers run.
"""

import numpy as np
from numba.typed import List

from haplotype_reconstruction.core.kernels import sample_team, samples_per_thread

from .capabilities import ComponentWorkspaceView, RefinementExecution
from .objective import selected_alleles

__all__ = ["Orientation", "sample_ranges", "sample_threads"]

# A search's messages and per-sample scores run over consecutive sample ranges and re-read its thread share before
# each range, so a search whose share grows while it computes them (its peers finished) takes the freed threads
# within the phase. At most SAMPLE_RANGES ranges per phase (each range rebuilds the phase's shared pair-dosage tables),
# each sized by the team's grain (core.kernels.sample_team, from the component's work estimate): a wide share runs one
# range, and a range never wakes more threads than its samples can occupy. Sample order and values do not depend on
# either.
SAMPLE_RANGES = 8


def sample_threads(samples: int, team: RefinementExecution) -> int:
    """Threads for one parallel region over ``samples`` samples: the team's live share read now, at most one thread per
    samples_per_thread(team.sample_work) samples (core.kernels.sample_team)."""
    return sample_team(samples, team.threads(), team.sample_work)


def sample_ranges(samples: int, team: RefinementExecution):
    """Consecutive ranges (s0, s1, n) covering samples 0 .. samples - 1, in order; n threads for the range: the team's
    live share read when the range starts, at most one thread per samples_per_thread(team.sample_work) samples of
    the range."""
    start = 0
    work = team.sample_work
    while start < samples:
        share = max(1, int(team.threads()))
        stop = min(samples, start + max(-(-samples // SAMPLE_RANGES), samples_per_thread(work) * share))
        yield start, stop, sample_team(stop - start, share, work)
        start = stop


class Orientation:
    """View coded leaves and offsets in one pass direction, retaining forward evidence.

    Reverse creates an int16[R_b,L_b] leaf copy and int64[B+1] offsets. haps returns
    a private int16[K,L] materialisation; window returns a workspace-owned view.
    """

    def __init__(self, workspace: ComponentWorkspaceView, reverse: bool):
        sites = int(workspace.offsets[-1])
        self.reverse = reverse
        self.workspace = workspace
        if reverse:
            self.offsets = np.ascontiguousarray(sites - workspace.offsets[::-1], np.int64)
            self.leaves = List([np.ascontiguousarray(leaf[:, ::-1]) for leaf in list(workspace.leaves)[::-1]])
        else:
            self.offsets = np.ascontiguousarray(workspace.offsets, np.int64)
            self.leaves = workspace.leaves
        self.blocks = len(self.offsets) - 1

    def haps(self, panel):
        """Return private int16[K,L] coded alleles of int64[K,B] oriented founder rows."""
        return selected_alleles(self.leaves, self.offsets, np.ascontiguousarray(panel, np.int64))

    def window(self, founders):
        """Read K's evidence/cost capability; its view lifetime is the workspace lease."""
        return self.workspace.windows.view(founders, self.reverse)
