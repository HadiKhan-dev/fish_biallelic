"""Hold a sweep's immutable arrays and caller-supplied numerical storage capabilities.

Owns the numeric workspace view and the sample-range order (RS01, RC04–RC07).
Must not choose storage routes, inspect memory, or configure runtime threads.
Method: docs/methods/founder_refinement.md#messages.
Dense message views last until update; reconstructed views last until another read
of the same segment parity. Proposal flanks are copied before parallel workers run.
"""

from numba.typed import List

from haplotype_reconstruction.contracts.assembly import RefinementProblem

from .capabilities import EvidenceWindows, RefinementExecution

__all__ = ["SweepWorkspace", "sample_ranges"]

SAMPLE_RANGES = 8  # 3.5 workspace scheduling; sample order and values do not depend on it.
SAMPLES_PER_THREAD = 8


def sample_ranges(samples: int, team: RefinementExecution):
    """Yield consecutive (start,stop,threads) ranges over N samples, without reordering."""
    start = 0
    while start < samples:
        n = max(1, int(team.threads()))
        stop = min(samples, start + max(-(-samples // SAMPLE_RANGES), SAMPLES_PER_THREAD * n))
        yield start, stop, n
        start = stop


class SweepWorkspace:
    """Borrow immutable problem arrays and caller-owned windows for one sweep.

    message_step is the stage's declared dense/checkpoint capability. leaves is a
    typed list of existing int16[R_b,L_b] arrays; it owns references, not copies.
    Only window buffers and messages created during the sweep mutate.
    """

    def __init__(self, problem: RefinementProblem, windows: EvidenceWindows, message_step: int):
        self.leaves = List(list(problem.leaves))
        self.offsets = problem.offsets
        self.samples = problem.evidence.shape[0]
        self.normalisers = problem.normalisers
        self.windows = windows
        self.message_step = message_step
