"""Hold a sweep's immutable arrays and caller-supplied numerical storage capabilities.

Owns the numeric workspace view and the sample-range order (RS01, RC04–RC07).
Must not choose storage routes, inspect memory, or configure runtime threads.
Method: docs/methods/founder_refinement.md#messages.
Dense message views last until update; reconstructed views last until another read
of the same segment parity. Proposal flanks are copied before parallel workers run.
"""

from numba.typed import List

from haplotype_reconstruction.contracts.assembly import RefinementProblem
from haplotype_reconstruction.core.kernels import SAMPLES_PER_THREAD, sample_team

from .capabilities import EvidenceWindows, RefinementExecution

__all__ = ["SweepWorkspace", "sample_ranges", "sample_threads"]

# A search's messages and per-sample scores run over consecutive sample ranges and re-read its thread share before
# each range, so a search whose share grows while it computes them (its peers finished) takes the freed threads
# within the phase. At most SAMPLE_RANGES ranges per phase (each range rebuilds the phase's shared pair-dosage tables)
# and at least SAMPLES_PER_THREAD samples per thread of a range (core.kernels.sample_team): a wide share runs one
# range, and a range never wakes more threads than its samples can occupy. Sample order and values do not depend on
# either.
SAMPLE_RANGES = 8


def sample_threads(samples: int, team: RefinementExecution) -> int:
    """Threads for one parallel region over ``samples`` samples: the team's live share read now, at most one thread per
    SAMPLES_PER_THREAD samples (core.kernels.sample_team)."""
    return sample_team(samples, team.threads())


def sample_ranges(samples: int, team: RefinementExecution):
    """Consecutive ranges (s0, s1, n) covering samples 0 .. samples - 1, in order; n threads for the range: the team's
    live share read when the range starts, at most one thread per SAMPLES_PER_THREAD samples of the range."""
    start = 0
    while start < samples:
        share = max(1, int(team.threads()))
        stop = min(samples, start + max(-(-samples // SAMPLE_RANGES), SAMPLES_PER_THREAD * share))
        yield start, stop, sample_team(stop - start, share)
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
