"""Select the RS12 fixed-K representative without collapsing equally supported panels.

Owns canonical selection after two directional searches. Must not rerun a search,
choose K, consult labels, or align founders. Method: docs/methods/founder_refinement.md#best-of-two.
Identity: highest objective class, then minimum exact-optimal switches, then the
explicit content convention. Distinct tied panels survive as biological alternatives.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from haplotype_reconstruction.contracts.assembly import RefinementSelection
from haplotype_reconstruction.core.ties import TiePolicy, policy

from ..orders import canonical_rows, panel_key

__all__ = ["select_representative", "select_sweeps"]


def select_representative(
    forward: NDArray[np.int64],
    reverse: NDArray[np.int64],
    scores: tuple[float, float],
    minimum_switches: Callable[[NDArray[np.int64]], int],
    convention: TiePolicy,
) -> RefinementSelection:
    """Choose between fwd/rev panels using their exact F values (nats).

    Rows are int64[K,B] in the searches' label frames. The supplied exact switch
    counter is called only for distinct tied panels, as in 3.5. Returns canonical
    private rows plus ordered alternatives, scores and switch counts; no mutation.
    """
    convention = policy("assembly.refine_converge", kind="refine_F", reverse=convention.reverse)
    panels = (forward, reverse)
    contents = [panel_key(panel) for panel in panels]
    values = [float(value) for value in scores]
    result = convention.classes(values, scale=max(abs(v) for v in values), content=contents, descending=True)
    chosen = int(result.winner)
    tied_panels: tuple[tuple[tuple[int, ...], ...], ...] = ()
    tied_scores: tuple[float, ...] = ()
    tied_switches: tuple[int, ...] = ()
    fewest = None
    if result.tied_winner and contents[0] != contents[1]:
        tied_panels = tuple(sorted(contents))
        counts = [minimum_switches(panel) for panel in panels]
        fewest = min(counts)
        chosen = next(int(i) for i in result.order if counts[i] == fewest)
        by_panel = {content: (value, count) for content, value, count in zip(contents, values, counts, strict=True)}
        tied_scores = tuple(by_panel[panel][0] for panel in tied_panels)
        tied_switches = tuple(by_panel[panel][1] for panel in tied_panels)
    rows, _ = canonical_rows(np.asarray(panels[chosen], np.int64))
    return RefinementSelection(rows, ("fwd", "rev")[chosen], tied_panels, tied_scores, tied_switches, fewest)


def select_sweeps(problem, forward, reverse, convention, workspace, team) -> RefinementSelection:
    """Apply RS12 to two completed sweeps of this int64[K,B] problem.

    The exact switch-count prior uses RC02's forward unordered-state recurrence,
    retaining the fewest switches at equal scores. Only distinct score-tied panels
    incur this score pass. Workspace buffers are scratch; inputs never mutate.
    """
    from .objective import score_switches_direct, selected_alleles

    def minimum_switches(rows):
        with workspace.windows.lease(len(rows)), team.scope():
            haps = selected_alleles(workspace.leaves, workspace.offsets, rows)
            total = 0
            for start, stop in workspace.windows.sample_slices(0, workspace.samples):
                win = workspace.windows.samples_view(len(rows), start, stop)
                _, counts = score_switches_direct(haps, win, start, stop)
                total += sum(map(int, counts))
            return total

    return select_representative(forward.rows, reverse.rows, (forward.F, reverse.F), minimum_switches, convention)
