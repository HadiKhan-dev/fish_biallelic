"""Fixed-K GEM fit of a panel and its frequencies under the copying model.

Each update performs a frequency EM step (MAP with one extra count spread uniformly; kept when
the objective does not drop by more than the tolerance), then flips one founder's sites with
positive expected complete-emission gains, or, at a GEM stationary point, the single exact
conditional one-bit flip with the largest gain. The fit stops at a conditional fixed point; a
flip that would duplicate a row stops it (a K change belongs to the outer search). The update
backstop is reported (termination "update_backstop").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from ..execution import claim_threads
from .objective import regularizer
from .path_model import COUNTS, EXACT, FLIPS, PathWorkspace


@dataclass(frozen=True)
class PathFit:
    """A fitted (panel, frequencies) of a block.

    panel: int64[K, L] binary rows in canonical (lexicographic) order; frequencies: float64[K]
    (sum 1). allele_probability: float64[K, L] probability of ALT per founder allele (the exact
    one-bit conditional posterior); occupancy: float64[K, L] expected copies per site (both are
    None while a fit is restored from a memo). log_likelihood, objective (log_likelihood +
    regularizer): nats. converged: ended at a conditional fixed point; termination:
    "conditional_fixed_point", "row_collision_requires_outer_selection" or "update_backstop".
    """

    panel: np.ndarray
    frequencies: np.ndarray
    allele_probability: np.ndarray | None
    occupancy: np.ndarray | None
    log_likelihood: float
    objective: float
    converged: bool
    termination: str

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


def fit_fixed(panel, frequencies, prepared, search_config, buffers=None) -> PathFit:
    """GEM fit from (panel, frequencies) (frequencies None = uniform) at fixed K. ``buffers``: the
    block search's shared trajectory storage (path_model.WorkspaceBuffers)."""
    tolerance = search_config.tolerance
    panel, frequencies = canonical(panel, frequencies)
    workspace = PathWorkspace(len(panel), prepared, buffers)
    # Every likelihood evaluation computes, in the same pass, the statistics its next use needs.
    score = workspace.state(panel, frequencies, COUNTS)["log_likelihood"]
    objective = score + regularizer(panel, frequencies)
    converged = False
    termination = "update_backstop"
    # After an update without a positive expected flip the next one usually has none either
    # (the frequency-only tail): its E-step then computes the exact gains in the same pass.
    stationary = False
    for _ in range(search_config.update_backstop(len(panel), panel.shape[1])):
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
        state = workspace.state(panel, frequencies, level)
        gains = state["flip_gains"]
        positive = np.where(gains > tolerance, gains, 0.)
        founder = int(np.argmax(positive.sum(axis=1)))
        changed = positive[founder] > 0
        stationary = not changed.any()
        if stationary:
            # A conditional one-bit improvement can escape a stationary GEM point (checked with
            # the exact marginal likelihood; computed only here and for the reported posteriors).
            exact = workspace.state(panel, frequencies, EXACT)["conditional_flip_gains"]
            founder, site = np.unravel_index(np.argmax(exact), exact.shape)
            if exact[founder, site] <= tolerance:
                if frequency_gain > tolerance:
                    continue
                converged = True
                termination = "conditional_fixed_point"
                break
            changed = np.zeros(panel.shape[1], dtype=bool)
            changed[site] = True
        trial = panel.copy()
        trial[founder, changed] = 1 - trial[founder, changed]
        if has_duplicate_rows(trial):
            termination = "row_collision_requires_outer_selection"
            break
        trial_score = workspace.state(trial, frequencies, COUNTS)["log_likelihood"]
        panel, score = trial, trial_score
        objective = trial_score + regularizer(trial, frequencies)
    allele_probability, occupancy = posteriors(workspace.state(panel, frequencies, EXACT), panel)
    return PathFit(panel=panel, frequencies=frequencies, allele_probability=allele_probability,
                   occupancy=occupancy, log_likelihood=float(score), objective=float(objective),
                   converged=converged, termination=termination)
