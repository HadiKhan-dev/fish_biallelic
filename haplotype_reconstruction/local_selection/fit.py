"""Fixed-K GEM fit of a panel and its frequencies under the copying model.

Each update performs a frequency EM step (MAP with one extra count spread uniformly; kept when
the objective does not drop by more than the tolerance), then flips one founder's sites with
positive expected complete-emission gains, or, at a GEM stationary point, the single exact
conditional one-bit flip with the largest gain. The fit stops at a conditional fixed point; a
flip that would duplicate a row stops it (a K change belongs to the outer search). The update
backstop is reported (termination "update_backstop").

Release 2.1 (noise stability): a fit is returned with its rows in canonical (lexicographic) order
(frequencies and per-row posteriors permuted with them). GEM flips move rows past one another, so
2.0's final row order depended on the trajectory: two fits ending at the same founders and
frequencies were different panels to every byte-keyed comparison, tie-break and record.

Release 2.2: founders (or single exact flips) whose gains lie within FLIP_TIE (relative) of the
largest are compared by the exact objective after each flip; a remaining tie between different
panels stops the ascent (keep-current, a stated prior: termination "tied_flips" /
"tied_exact_flips"). 2.1 took the first candidate in row, then site order (a convention).
"""
from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np
from scipy.special import expit

from ..execution import claim_threads
from ..maths.ties import census, reverse_conventions
from .objective import edit_ball_code, regularizer
from .path_model import COUNTS, EXACT, FLIPS, PathWorkspace

# Relative tie band of the GEM's flip choices: candidates whose gain is within FLIP_TIE x |largest
# gain| of the largest are tied (release 2.2: decided by the exact objective after each flip, see
# _tied_move). Far above the rounding of the gains (~1e-16 relative) and their motion under 1e-9
# relative GL noise.
FLIP_TIE = 2.0 ** -20


def tied_within(values, tie=FLIP_TIE) -> np.ndarray:
    """Flat indices of every element within tie x |max| of the maximum of ``values``."""
    flat = np.ravel(values)
    best = float(flat.max())
    return np.flatnonzero(flat >= best - tie * abs(best))


def first_within(values, tie=FLIP_TIE) -> int:
    """Flat index of the first element within tie x |max| of the maximum of ``values`` (the last
    one under maths.ties.reverse_conventions: the row / site order is a convention)."""
    flat = np.ravel(values)
    best = float(flat.max())
    within = np.flatnonzero(flat >= best - tie * abs(best))
    return int(within[-1] if reverse_conventions() else within[0])


def _tied_move(panel, frequencies, moves, workspace, objective):
    """Among tied GEM moves ((founder, sites to flip), gains within FLIP_TIE), the one whose flipped
    panel has the best exact objective, as (founder, changed, trial panel, trial log likelihood).
    Moves that duplicate a row are skipped ("collision" when every move does). Objectives within
    FLIP_TIE x |best objective gain| of the best are tied: if they lead to one panel (the same rows
    and frequencies in canonical order, i.e. a relabelling) any of them is that move; if they lead
    to different panels, None (the data cannot decide: the caller stops the ascent, keep-current
    being the stated prior)."""
    candidates = []
    for founder, changed in moves:
        trial = panel.copy()
        trial[founder, changed] = 1 - trial[founder, changed]
        if has_duplicate_rows(trial):
            continue
        score = workspace.state(trial, frequencies, COUNTS)["log_likelihood"]
        gain = score + regularizer(trial, frequencies) - objective
        candidates.append((gain, founder, changed, trial, score))
    if not candidates:
        return "collision"
    best = max(candidate[0] for candidate in candidates)
    tied = [candidate for candidate in candidates if candidate[0] >= best - FLIP_TIE * abs(best)]
    identities = set()
    for _gain, _founder, _changed, trial, _score in tied:
        rows, mass = canonical(trial, frequencies)
        identities.add((rows.tobytes(), mass.tobytes()))
    if len(identities) > 1:
        return None
    _gain, founder, changed, trial, score = tied[0]
    return founder, changed, trial, score


def canonical_order(panel) -> np.ndarray:
    """Permutation putting distinct binary rows in lexicographic order (stable)."""
    panel = np.asarray(panel)
    if len(panel) <= 1:
        return np.arange(len(panel))
    return np.lexsort(panel.T[::-1])


@dataclass(frozen=True)
class PathFit:
    """A fitted (panel, frequencies) of a block.

    panel: int64[K, L] binary rows in canonical (lexicographic) order; frequencies: float64[K]
    (sum 1). allele_probability: float64[K, L] probability of ALT per founder allele (the exact
    one-bit conditional posterior); occupancy: float64[K, L] expected copies per site (both are
    None while a fit is restored from a memo). log_likelihood, objective (log_likelihood +
    regularizer): nats. converged: ended at a conditional fixed point; termination:
    "conditional_fixed_point", "row_collision_requires_outer_selection", "update_backstop", or
    (release 2.2) "tied_flips" / "tied_exact_flips": the best moves tied and led to different panels,
    so the ascent stopped (keep-current; converged is True). termination always describes the trajectory.
    selected: "endpoint" (the trajectory's last state) or "trajectory_best" (an earlier state that the
    caller's selection objective prefers by more than the decision tolerance; it is then refitted in its
    frequencies at the fixed panel and is NOT a conditional fixed point: converged is False).
    selected_update: the GEM update after which the selected state was reached (-1: the start).
    endpoint_selection, selected_selection: the selection objective of the endpoint and of the returned
    state (None without a selection objective).
    context_snaps (release 2.12): one-bit flips of a context round's fit taken by the selection objective
    after the GEM (``_context_snaps``); a fit with snaps is not a conditional fixed point (converged False).
    snap_ties: the snap steps decided inside a tie class (by content).
    """

    panel: np.ndarray
    frequencies: np.ndarray
    allele_probability: np.ndarray | None
    occupancy: np.ndarray | None
    log_likelihood: float
    objective: float
    converged: bool
    termination: str
    selected: str = "endpoint"
    selected_update: int = -1
    endpoint_selection: float | None = None
    selected_selection: float | None = None
    context_snaps: int = 0
    snap_ties: int = 0

    @property
    def update_backstop(self) -> bool:
        return self.termination == "update_backstop"


def unique_rows(matrix, *, return_index=False, return_inverse=False):
    """``np.unique(matrix, axis=0, ...)`` of a 2-D integer matrix (rows in lexicographic order; index
    = first occurrence; inverse 1-D), computed with a stable lexsort. numpy 2.4's ``axis=0`` path
    builds a new structured dtype on every call and never frees it (~56 KB per call at 200 sites):
    called in every GEM update, it grew a worker's memory by ~0.2 GB per AulStu block and exhausted
    the node in base's AulStu run."""
    rows = np.asarray(matrix)
    n = rows.shape[0]
    if n == 0 or rows.shape[1] == 0:
        unique = rows[:min(n, 1)].copy()
        first = np.zeros(len(unique), dtype=np.intp)
        inverse = np.zeros(n, dtype=np.intp)
    else:
        order = np.lexsort(rows.T[::-1])
        ordered = rows[order]
        new = np.ones(n, dtype=bool)
        new[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
        unique = ordered[new]
        first = order[new]
        inverse = np.empty(n, dtype=np.intp)
        inverse[order] = np.cumsum(new) - 1
    if not (return_index or return_inverse):
        return unique
    return (unique,) + ((first,) if return_index else ()) + ((inverse,) if return_inverse else ())


def has_duplicate_rows(matrix) -> bool:
    """Two rows of a 2-D array are equal."""
    rows = np.ascontiguousarray(matrix)
    return len({row.tobytes() for row in rows}) < len(rows)


def canonical(panel, frequencies=None):
    """Distinct rows in lexicographic order with their summed (renormalised) frequencies."""
    panel = np.ascontiguousarray(panel, dtype=np.int64)
    unique, inverse = unique_rows(panel, return_inverse=True)
    if frequencies is None:
        frequencies = np.full(len(panel), 1. / len(panel))
    mass = np.bincount(inverse, weights=frequencies, minlength=len(unique))
    return unique, mass / mass.sum()


def posteriors(state, panel) -> tuple[np.ndarray, np.ndarray]:
    """(allele_probability, occupancy) of a fit from its final E-step."""
    current = expit(-state["conditional_flip_gains"])
    return np.where(panel == 1, current, 1 - current), state["occupancy"][:-1]


def fit_posteriors(panel, frequencies, prepared, buffers=None):
    """The (allele_probability, occupancy) a fit ending at (panel, frequencies) reports."""
    workspace = PathWorkspace(len(panel), prepared, buffers)
    return posteriors(workspace.state(panel, frequencies, EXACT), panel)


def _census_top(kind, key, values):
    """Valley census of a GEM choice: the two largest candidates (negated, best first)."""
    flat = np.sort(np.ravel(values))[::-1][:2]
    census(kind, key, [-float(value) for value in flat])


def _context_snaps(panel, frequencies, score, workspace, context, selection, search_config):
    """Selection-aware one-bit completion of a context round's fit (release 2.12).

    The GEM climbs the context-free objective, so an allele the data do not inform (exact one-bit likelihood gain
    ~0) stays wherever the trajectory left it, although the selection objective prefers it to agree with the claiming
    founder's context call. Here each step is judged by the selection objective. For each (founder f, row k) pair of
    one shortest description of the panel (``ContextFounders.matched``, canonical choice), every called site where k
    differs from f is a proposal. Its lower bound is the exact one-bit likelihood gain (fixed frequencies) plus the
    edit-code saving of that matched pair, log V(n_f, d) - log V(n_f, d - 1): the current description stays feasible
    after the flip, so the selection value rises by at least the bound. A step needs the largest bound above the
    decision tolerance; the bounds within the tolerance of that largest one are one tie class (one band, not a chain),
    decided by content: the flip whose resulting panel, rows in canonical order, has the smallest bytes (streamed:
    only that trial is kept). Its bound is strictly positive and the step is accepted only if the recomputed value
    rises. Flips that would make two rows equal are not proposed (changing K belongs to the outer search). Each step
    solves one matching and scores one panel, O(K^3 + K L), plus O(K L log K) per tie-class member. After
    each flip the frequencies are refitted at the fixed panel while the selection objective rises. Returns (panel,
    frequencies, log likelihood, selection value, snaps, tied steps) or None when no step is taken."""
    tolerance = search_config.decision_tolerance
    value = selection(panel, frequencies, score)
    snaps = tied_steps = 0
    rows, length = panel.shape
    for _ in range(search_config.update_backstop(rows, length)):
        claim_threads()
        exact = workspace.state(panel, frequencies, EXACT)["conditional_flip_gains"]
        distance = context._columns(panel)
        # Flips making two rows equal: row pairs at Hamming distance one, at their differing site.
        collide = set()
        for a in range(rows):
            for b in range(a + 1, rows):
                differ = np.flatnonzero(panel[a] != panel[b])
                if len(differ) == 1:
                    collide.add((a, int(differ[0])))
                    collide.add((b, int(differ[0])))
        proposals = []
        for f, k in context.matched(panel):
            n, d = int(context.n_called[f]), int(distance[f, k])
            if d <= 0:
                continue
            saving = edit_ball_code(n, d) - edit_ball_code(n, d - 1)
            for site in np.flatnonzero(context.called[f] & (context.founders[f] != panel[k])):
                if (k, int(site)) not in collide:
                    proposals.append((float(exact[k, site]) + saving, k, int(site)))
        if not proposals:
            break
        top = max(bound for bound, _, _ in proposals)
        if not top > tolerance:
            break
        # Stream the tie class: keep only its canonically smallest trial panel (one candidate panel in memory).
        members, best_key, best_move = 0, None, None
        for bound, k, site in proposals:
            if bound >= top - tolerance:
                panel[k, site] = 1 - panel[k, site]
                key = panel[canonical_order(panel)].tobytes()
                panel[k, site] = 1 - panel[k, site]
                members += 1
                if best_key is None or key < best_key:
                    best_key, best_move = key, (k, site)
        tied_steps += int(members > 1)
        trial = panel.copy()
        trial[best_move] = 1 - trial[best_move]
        trial_score = workspace.state(trial, frequencies, COUNTS)["log_likelihood"]
        trial_value = selection(trial, frequencies, trial_score)
        if not trial_value > value:
            break                                     # the bound guarantees a rise; stop at numerical noise
        panel, score, value = trial, trial_score, trial_value
        snaps += 1
        for _ in range(search_config.update_backstop(rows, length)):
            claim_threads()
            state = workspace.state(panel, frequencies, COUNTS)
            counts = (state["initial_copy_counts"] + state["redraw_destination_counts"])[:-1]
            trial_frequencies = (counts + 1. / rows) / (counts.sum() + 1.)
            trial_score = workspace.state(panel, trial_frequencies, COUNTS)["log_likelihood"]
            trial_value = selection(panel, trial_frequencies, trial_score)
            if not trial_value > value + search_config.tolerance:
                break
            frequencies, score, value = trial_frequencies, trial_score, trial_value
    if not snaps:
        return None
    return panel, frequencies, score, value, snaps, tied_steps


def fit_fixed(panel, frequencies, prepared, search_config, buffers=None, census_key=None,
              selection=None, context=None) -> PathFit:
    """GEM fit from (panel, frequencies) (frequencies None = uniform) at fixed K. ``buffers``: the
    block search's shared trajectory storage (path_model.WorkspaceBuffers). ``census_key``: when the
    valley census is on, the fit's key for its flip choices (diagnostics only). ``selection``: the
    objective the caller selects fits by, (panel, frequencies, log likelihood) -> value (the
    context-coded selection objective of a context round). The GEM climbs the context-free objective,
    so it can pass through and then leave a state the selection prefers; with ``selection`` the best
    state under it along the trajectory (the start and every accepted state) is returned when it beats
    the endpoint by more than ``search_config.decision_tolerance`` (the outer decisions' tie quantum), its
    frequencies refitted at its fixed panel (frequency-only EM steps, each kept only when the selection
    objective rises; the context code of a fixed panel does not depend on the frequencies). The returned
    state is then not a conditional fixed point (converged False; PathFit.selected says which state)."""
    tolerance = search_config.tolerance
    panel, frequencies = canonical(panel, frequencies)
    workspace = PathWorkspace(len(panel), prepared, buffers)
    # Every likelihood evaluation computes, in the same pass, the statistics its next use needs.
    score = workspace.state(panel, frequencies, COUNTS)["log_likelihood"]
    objective = score + regularizer(panel, frequencies)
    best = None
    if selection is not None:
        best = (selection(panel, frequencies, score), panel.copy(), frequencies.copy(), score, -1)
    converged = False
    termination = "update_backstop"
    # After an update without a positive expected flip the next one usually has none either
    # (the frequency-only tail): its E-step then computes the exact gains in the same pass.
    stationary = False
    for update in range(search_config.update_backstop(len(panel), panel.shape[1])):
        claim_threads()
        state = workspace.state(panel, frequencies, COUNTS)
        counts = (state["initial_copy_counts"] + state["redraw_destination_counts"])[:-1]
        trial_frequencies = (counts + 1. / len(panel)) / (counts.sum() + 1.)
        level = EXACT if stationary else FLIPS
        trial_score = workspace.state(panel, trial_frequencies, level)["log_likelihood"]
        trial_objective = trial_score + regularizer(panel, trial_frequencies)
        frequency_gain = 0.
        if trial_objective >= objective - tolerance:
            frequency_gain = trial_objective - objective
            frequencies, score, objective = trial_frequencies, trial_score, trial_objective
            if best is not None:
                value = selection(panel, frequencies, score)
                if value > best[0]:
                    best = (value, panel.copy(), frequencies.copy(), score, update)
        state = workspace.state(panel, frequencies, level)
        gains = state["flip_gains"]
        positive = np.where(gains > tolerance, gains, 0.)
        sums = positive.sum(axis=1)
        if census_key is not None and len(sums) > 1 and sums.max() > 0:
            _census_top("ls.gem.founder", f"{census_key}/u{update}", sums)
        stationary = not (sums.max() > 0)
        trial = trial_score = None
        if not stationary:
            founders = tied_within(sums)
            if len(founders) == 1:
                founder = int(founders[0])
                changed = positive[founder] > 0
            else:
                # Tied founders (release 2.2): the data decide by the exact objective after each
                # founder's flip; a remaining tie between different panels stops the ascent here
                # (keep-current, a stated prior), never a row-order choice.
                moves = [(int(a), positive[int(a)] > 0) for a in founders]
                choice = _tied_move(panel, frequencies, moves, workspace, objective)
                if choice == "collision":
                    termination = "row_collision_requires_outer_selection"
                    break
                if choice is None:
                    converged = True
                    termination = "tied_flips"
                    break
                founder, changed, trial, trial_score = choice
        if stationary:
            # A conditional one-bit improvement can escape a stationary GEM point (checked with
            # the exact marginal likelihood; computed only here and for the reported posteriors).
            exact = workspace.state(panel, frequencies, EXACT)["conditional_flip_gains"]
            if census_key is not None and exact.max() > tolerance:
                _census_top("ls.gem.exact", f"{census_key}/u{update}", exact)
            cells = [int(cell) for cell in tied_within(exact) if exact.flat[int(cell)] > tolerance]
            if not cells:
                if frequency_gain > tolerance:
                    continue
                converged = True
                termination = "conditional_fixed_point"
                break
            if len(cells) == 1:
                founder, site = np.unravel_index(int(cells[0]), exact.shape)
                changed = np.zeros(panel.shape[1], dtype=bool)
                changed[site] = True
            else:
                moves = []
                for cell in cells:
                    a, site = np.unravel_index(int(cell), exact.shape)
                    one = np.zeros(panel.shape[1], dtype=bool)
                    one[site] = True
                    moves.append((int(a), one))
                choice = _tied_move(panel, frequencies, moves, workspace, objective)
                if choice == "collision":
                    termination = "row_collision_requires_outer_selection"
                    break
                if choice is None:
                    converged = True
                    termination = "tied_exact_flips"
                    break
                founder, changed, trial, trial_score = choice
        if trial is None:
            trial = panel.copy()
            trial[founder, changed] = 1 - trial[founder, changed]
            if has_duplicate_rows(trial):
                termination = "row_collision_requires_outer_selection"
                break
            trial_score = workspace.state(trial, frequencies, COUNTS)["log_likelihood"]
        panel, score = trial, trial_score
        objective = trial_score + regularizer(trial, frequencies)
        if best is not None:
            value = selection(panel, frequencies, score)
            if value > best[0]:
                best = (value, panel.copy(), frequencies.copy(), score, update)
    selected, selected_update, endpoint_selection, selected_selection = "endpoint", -1, None, None
    if best is not None:
        endpoint_selection = selected_selection = selection(panel, frequencies, score)
        if best[0] > endpoint_selection + search_config.decision_tolerance:
            selected_selection, panel, frequencies, score, selected_update = best
            selected, converged = "trajectory_best", False
            for _ in range(search_config.update_backstop(len(panel), panel.shape[1])):
                claim_threads()
                state = workspace.state(panel, frequencies, COUNTS)
                counts = (state["initial_copy_counts"] + state["redraw_destination_counts"])[:-1]
                trial_frequencies = (counts + 1. / len(panel)) / (counts.sum() + 1.)
                trial_score = workspace.state(panel, trial_frequencies, COUNTS)["log_likelihood"]
                value = selection(panel, trial_frequencies, trial_score)
                if not value > selected_selection + tolerance:
                    break
                frequencies, score, selected_selection = trial_frequencies, trial_score, value
            objective = score + regularizer(panel, frequencies)
    snaps = snap_ties = 0
    if selection is not None and context is not None and len(context):
        snapped = _context_snaps(panel, frequencies, score, workspace, context, selection, search_config)
        if snapped is not None:
            panel, frequencies, score, selected_selection, snaps, snap_ties = snapped
            converged = False
            if endpoint_selection is None:
                endpoint_selection = selection(panel, frequencies, score)
            objective = score + regularizer(panel, frequencies)
    allele_probability, occupancy = posteriors(workspace.state(panel, frequencies, EXACT), panel)
    # Canonical row order on output (the likelihood and the objective are invariant to it).
    order = canonical_order(panel)
    return PathFit(panel=np.ascontiguousarray(panel[order]),
                   frequencies=np.ascontiguousarray(frequencies[order]),
                   allele_probability=np.ascontiguousarray(allele_probability[order]),
                   occupancy=np.ascontiguousarray(occupancy[order]), log_likelihood=float(score),
                   objective=float(objective), converged=converged, termination=termination,
                   selected=selected, selected_update=int(selected_update),
                   endpoint_selection=None if endpoint_selection is None else float(endpoint_selection),
                   selected_selection=None if selected_selection is None else float(selected_selection),
                   context_snaps=int(snaps), snap_ties=int(snap_ties))
