"""Count released phase sets and retain native solver diagnostics without execution timing.

The native history-step count is len(history)-1; repeated sweeps() calls each
append their initial objective, so this is deliberately not an iteration count.
"""

from __future__ import annotations

import numpy as np

from ...contracts.phase import MirroredSolveDiagnostics, PairingDiagnostics, PhaseComponentState
from .parameters import PhaseParameters


def phase_set_count(phased: np.ndarray, component_ids: np.ndarray, breaks: list[list[int]]) -> int:
    """Count released sets containing a phased heterozygote, split at components and starts (FP24)."""
    length = len(component_ids)
    edge = np.zeros(length, dtype=bool)
    edge[1:] = component_ids[1:] != component_ids[:-1]
    count = 0
    for sample, sample_breaks in enumerate(breaks):
        sites = phased[sample] & (component_ids >= 0)
        if not sites.any():
            continue
        cut = edge.copy()
        starts = np.asarray(sorted(set(sample_breaks)), dtype=np.int64) + 1
        cut[starts[(starts > 0) & (starts < length)]] = True
        count += len(np.unique(np.cumsum(cut)[sites]))
    return count


def solve_diagnostics(
    state: PhaseComponentState, parameters: PhaseParameters
) -> tuple[MirroredSolveDiagnostics, PairingDiagnostics]:
    """Borrow both final histories and count actual pairing decisions at the native ambiguity cut."""
    first, mirror = state.result.solves
    cutoff = -float(np.log(parameters.set_posterior / (1.0 - parameters.set_posterior)))
    mirrored = MirroredSolveDiagnostics(
        len(mirror.history) - 1,
        mirror.settled,
        mirror.history,
        len(state.junctions[1]),
        first.settled,
        len(state.junctions[0]),
    )
    pairings = PairingDiagnostics(
        len(first.pairings),
        len(first.pairing_moves),
        len(mirror.pairing_moves),
        len({(p, j) for solve in (first, mirror) for p, j, gain in solve.pairings if gain > cutoff}),
        sum(len(bins) for bins in state.pairing.values()),
    )
    return mirrored, pairings
