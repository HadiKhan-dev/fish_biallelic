"""Fits of the nested read models on training histograms (folds 0 and 1).

Candidates, from simplest (the order is the tie rule of the selection):

* ``binomial``: one shared binomial model, error e and heterozygote ALT fraction q, fitted by
  EM from several ordered starts (per-sample genotype mixtures are nuisances);
* ``sample_binomial``: per-sample logit-error and logit-balance effects pooled by an
  empirical-Bayes normal measurement-error approximation (diagonal Laplace: each sample's
  uncertainty from the observed profile Hessian of its own likelihood; hyperparameters from
  training data only), binomial genotype kernels;
* ``sample_joint``: those effects refitted jointly with one shared heterozygote concentration
  kappa (beta-binomial heterozygotes, binomial homozygotes).

Genotype mixtures are fitting nuisances only: they are never returned as genotype priors.
Optimiser stop reasons (convergence, iteration cap, line-search stop) are recorded as diagnostics
(``ReadModelFit.success``) and never decide which fit is used (class-B fix B-RM). Every L-BFGS-B fit is finished by
a Newton polish to its stationary point (release 2.1, design A-5): the optimiser's stopping point depends on its
path, which rounding-level arithmetic changes (another CPU, BLAS kernel or SIMD path) alter, while the optimum does
not; successful shared-binomial EM endpoints are likewise finished at stationarity before the
endpoint-equivalence and training-likelihood policy is applied. Homozygotes are
binomial in every candidate: the shared homozygote-overdispersion extension of earlier versions
was removed by the owner's decision (17 Oct 2026; tickets/B-RM).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, cast

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import logit

from haplotype_reconstruction.contracts.evidence import (
    FixedFitDetails,
    JointFitDetails,
    PooledFitDetails,
    ReadModelFit,
    SharedDistance,
    SharedFitDetails,
    SharedFitDiagnostics,
    SharedStart,
    StationaryStart,
)

from .emission import em_statistics, samples_objective_parallel, samples_objective_serial, shared_nuisance_curvature
from .parameters import (
    EM_DECREASE_PER_OBSERVATION,
    HYPERFIT_VARIANCE_CEILING_FACTOR,
    POLISH_BACKTRACK_TRIALS,
    POLISH_STAGNATION_RATIO,
    POSITIVE_WEIGHT_FLOOR,
    PROFILE_BRACKET_GROWTH,
    PROFILE_BRACKET_INITIAL_WIDTH,
    PROFILE_BRACKET_TRIALS,
    PROFILE_ROOT_EPS_FACTOR,
    PROFILE_ROOT_ITERATIONS,
    PROFILE_ROOT_XTOL,
    ReadModelParameters,
)
from .parameters import (
    FIT_BALANCE_LOGIT_BOUNDS as _BALANCE_LOGIT_BOUNDS,
)
from .parameters import (
    FIT_EM_START_BAND_PER_OBSERVATION as _EM_START_BAND_PER_OBSERVATION,
)
from .parameters import (
    FIT_ERROR_LOGIT_BOUNDS as _ERROR_LOGIT_BOUNDS,
)
from .parameters import (
    FIT_HESSIAN_EIGENVALUE_FLOOR as _HESSIAN_EIGENVALUE_FLOOR,
)
from .parameters import (
    FIT_HESSIAN_STEP as _HESSIAN_STEP,
)
from .parameters import (
    FIT_HYPERFIT_BOUNDARY_BAND as _HYPERFIT_BOUNDARY_BAND,
)
from .parameters import (
    FIT_HYPERFIT_MINIMUM_SAMPLES as _HYPERFIT_MINIMUM_SAMPLES,
)
from .parameters import (
    FIT_JOINT_FIT_OPTIONS as _JOINT_FIT_OPTIONS,
)
from .parameters import (
    FIT_LIKELIHOOD_TIE_NATS as _LIKELIHOOD_TIE_NATS,
)
from .parameters import (
    FIT_LOG_CONCENTRATION_BOUNDS as _LOG_CONCENTRATION_BOUNDS,
)
from .parameters import (
    FIT_MIXING_BOUNDS as _MIXING_BOUNDS,
)
from .parameters import (
    FIT_POLISH_CURVATURE_FLOOR as _POLISH_CURVATURE_FLOOR,
)
from .parameters import (
    FIT_POLISH_DECREASE as _POLISH_DECREASE,
)
from .parameters import (
    FIT_POLISH_ITERATIONS as _POLISH_ITERATIONS,
)
from .parameters import (
    FIT_POLISH_STAGNATION as _POLISH_STAGNATION,
)
from .parameters import (
    FIT_POLISH_STEP as _POLISH_STEP,
)
from .parameters import (
    FIT_POLISH_TOLERANCE as _POLISH_TOLERANCE,
)
from .parameters import (
    FIT_SAMPLE_FIT_OPTIONS as _SAMPLE_FIT_OPTIONS,
)

# Model names in candidate order (simplest first).
BINOMIAL = "binomial"
SAMPLE_BINOMIAL = "sample_binomial"
SAMPLE_JOINT = "sample_joint"

# Box constraints of the optimisers (logit error, logit balance, log kappa, mixing logits).
# Observed-Hessian finite-difference step and the positive-definiteness floor.
# Samples with an identifiable effect needed for the empirical-Bayes hyperfit.
# Newton polish of every L-BFGS-B fit (release 2.1, design A-5 / review R5): central-difference Hessian columns of the
# analytic gradient (relative step), at most this many Newton steps, done when a step moves no coordinate by more
# than the tolerance (relative to max(1, |x|)), or when steps below _POLISH_STAGNATION stop shrinking (a flat
# direction, e.g. log kappa near the binomial limit, locates its optimum only to the gradient's rounding divided by
# its curvature: TroMau chr15 ~1e-10 in log kappa). Measured on actm3 chr13 (sample_joint, 465 parameters): L-BFGS-B
# stops 8e-3 (logits) and 3e-4 (log kappa) short of the optimum and two paths differ by 2e-4; after 3 Newton steps
# both agree to 1e-14 (logits) and 2e-13 (log kappa); on TroMau chr15 log kappa moves 12.49 -> 15.23 in 12 steps.
# A step whose predicted decrease is above the objective's rounding (relative _POLISH_DECREASE) must decrease it
# (backtracking); smaller steps are taken as they are.
# A coordinate whose Hessian diagonal is below this fraction of the largest one has no curvature (it is held).
# EM starts whose converged log likelihoods lie within this many nats per observation of the best are the same
# optimum (EM stops at a gain of em_tolerance_per_observation per observation; three starts of real contigs end
# within 1e-11 per observation of each other, measured on actm3). Kept for diagnostics after stationary polishing;
# distinct stationary models use the training-likelihood rule below.
# One nat is the existing read-model likelihood tie band, shared with held-out selection.
# Empirical-Bayes hyperfit: the boundary tau = 0 is kept when its profile value is within this relative band of the
# interior optimum's (the two coincide when the optimum is the boundary; a band at that exact tie, design A-7).


def fixed_model(error: float, n_samples: int) -> ReadModelFit:
    """The fallback model: binomial, error ``error`` and heterozygote ALT fraction 1/2 for all."""
    return ReadModelFit(
        name="fixed",
        error_logit=np.full(n_samples, logit(error)),
        balance_logit=np.zeros(n_samples),
        log_concentration=None,
        mixing=np.zeros((n_samples, 2)),
        success=True,
        details=FixedFitDetails(error=float(error)),
    )


# ---------------------------------------------------------------------------------------------
# Objective of the sample models (normalised negative log likelihood and its gradient)
# ---------------------------------------------------------------------------------------------


def _objective(local, het_inverse, hist, depth, alt, mean, variance):
    """(-sum of sample values / scale, -gradients / scale); gradients (n, 5) as sample_objective."""
    kernel = samples_objective_serial if len(hist) == 1 else samples_objective_parallel
    values, gradients = kernel(local, het_inverse, hist, depth, alt, mean, variance)
    scale = max(1.0, float(hist.sum()))
    return -float(values.sum()) / scale, -gradients / scale


def joint_objective(vector, hist, depth, alt, mean, variance):
    """Per-sample (logit e, logit balance, mixing 0, mixing 1) and one shared log kappa."""
    local = vector[:-1].reshape(len(hist), 4)
    value, gradients = _objective(local, np.exp(-vector[-1]), hist, depth, alt, mean, variance)
    return value, np.r_[gradients[:, :4].ravel(), gradients[:, 4].sum()]


def log_predictive(fit: ReadModelFit, depth, alt) -> np.ndarray:
    """float64[N, C]: each sample's log predictive likelihood of the cells (depth[c], alt[c]), depth >= 1, under a
    fit (its training mixture nuisances, no prior): the per-cell terms that ``score`` sums."""
    from .emission import prefix

    depth, alt = np.asarray(depth, np.int64), np.asarray(alt, np.int64)
    error, heterozygote = fit.error_and_heterozygote()
    inverse = fit.heterozygote_inverse_concentration
    deepest = int(depth.max()) if len(depth) else 0
    reference = depth - alt
    out = np.empty((fit.n_samples, len(depth)))
    for sample in range(fit.n_samples):
        hom, het = prefix(error[sample], 0.0, deepest), prefix(heterozygote[sample], inverse, deepest)
        m0, m1 = fit.mixing[sample]
        normaliser = np.logaddexp(np.logaddexp(m0, m1), 0.0)
        l0 = hom[0, alt] + hom[1, reference] - hom[2, depth] + m0 - normaliser
        l1 = het[0, alt] + het[1, reference] - het[2, depth] + m1 - normaliser
        l2 = hom[0, reference] + hom[1, alt] - hom[2, depth] - normaliser
        out[sample] = np.logaddexp(np.logaddexp(l0, l1), l2)
    return out


# ---------------------------------------------------------------------------------------------
# Newton polish (release 2.1): every fit is moved to its optimum, independently of the optimiser's path
# ---------------------------------------------------------------------------------------------


def _held(x, gradient, lower, upper):
    """Coordinates that do not move: fixed (lower == upper) or at a bound with the gradient pushing outward."""
    return (lower == upper) | ((x <= lower) & (gradient > 0)) | ((x >= upper) & (gradient < 0))


def _polish_step(evaluate, x, f, gradient, step, free, lower, upper):
    """Apply a Newton step on the ``free`` coordinates (clipped to the bounds); a step whose predicted decrease is
    above the objective's rounding must decrease it (halving up to 12 times). Returns (x, f, gradient, moved) or None
    when no acceptable step exists."""
    predicted = -0.5 * float(gradient[free] @ step)
    alpha = 1.0
    for _ in range(POLISH_BACKTRACK_TRIALS):
        candidate = x.copy()
        candidate[free] = np.clip(x[free] + alpha * step, lower[free], upper[free])
        value, slope = evaluate(candidate)
        if not np.isfinite(value):
            alpha /= 2.0
            continue
        if predicted <= _POLISH_DECREASE * max(1.0, abs(f)) or value <= f:
            moved = float(np.max(np.abs(candidate - x) / np.maximum(1.0, np.abs(x)), initial=0.0))
            return candidate, value, slope, moved
        alpha /= 2.0
    return None


def _polish(evaluate, x, lower, upper):
    """Newton polish of a small fit: minimise evaluate(x) -> (f, gradient) from ``x`` (near the optimum), holding
    fixed and bound-active coordinates. Returns (x, iterations, converged); a Hessian that is not positive definite
    or a failed step keeps the last point (converged False)."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    x = np.clip(np.asarray(x, float).copy(), lower, upper)
    f, gradient = evaluate(x)
    previous = np.inf
    for iteration in range(1, _POLISH_ITERATIONS + 1):
        free = np.flatnonzero(~_held(x, gradient, lower, upper))
        if not len(free):
            return x, iteration - 1, True
        hessian = np.empty((len(free), len(free)))
        for column, j in enumerate(free):
            h = _POLISH_STEP * max(1.0, abs(x[j]))
            up, down = x.copy(), x.copy()
            up[j] += h
            down[j] -= h
            hessian[:, column] = (evaluate(up)[1][free] - evaluate(down)[1][free]) / (2.0 * h)
        hessian = (hessian + hessian.T) / 2.0
        try:
            factor = cho_factor(hessian)
        except (LinAlgError, ValueError):
            # A coordinate without curvature (e.g. the balance of a sample with no heterozygote mass) does not move
            # the objective: hold it and polish the others.
            diagonal = np.diag(hessian)
            kept = diagonal > _POLISH_CURVATURE_FLOOR * max(float(diagonal.max(initial=0.0)), 0.0)
            if kept.all() or not kept.any():
                return x, iteration - 1, False
            free, hessian = free[kept], hessian[np.ix_(kept, kept)]
            try:
                factor = cho_factor(hessian)
            except (LinAlgError, ValueError):
                return x, iteration - 1, False
        result = _polish_step(evaluate, x, f, gradient, -cho_solve(factor, gradient[free]), free, lower, upper)
        if result is None:
            return x, iteration - 1, False
        x, f, gradient, moved = result
        if moved <= _POLISH_TOLERANCE or (moved <= _POLISH_STAGNATION and moved > previous / POLISH_STAGNATION_RATIO):
            return x, iteration, True
        previous = moved
    return x, _POLISH_ITERATIONS, False


def _polish_joint(vector, hist, depth, alt, mean, variance, lower, upper):
    """Newton polish of the joint fit (N x 4 sample effects and one shared log kappa).

    The Hessian is an arrowhead: one 4 x 4 block per sample (samples are independent given kappa), each coupled to log
    kappa. Central differences of the per-sample gradient rows give every block from 4 x 2 evaluations (one effect of
    every sample moved at once) and the kappa column from 2; the step solves the arrowhead system through the Schur
    complement of log kappa. Returns (vector, iterations, converged).
    """
    n = len(hist)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    x = np.clip(np.asarray(vector, float).copy(), lower, upper)
    scale = max(1.0, float(hist.sum()))
    kernel = samples_objective_serial if n == 1 else samples_objective_parallel

    def evaluate(point):
        return joint_objective(point, hist, depth, alt, mean, variance)

    def rows(local, log_kappa):
        return -kernel(local, np.exp(-log_kappa), hist, depth, alt, mean, variance)[1] / scale

    f, gradient = evaluate(x)
    previous = np.inf
    for iteration in range(1, _POLISH_ITERATIONS + 1):
        held = _held(x, gradient, lower, upper)
        if held.all():
            return x, iteration - 1, True
        local, log_kappa = x[:-1].reshape(n, 4), x[-1]
        blocks = np.empty((n, 4, 4))
        coupling = np.empty((n, 4))
        for j in range(4):
            h = _POLISH_STEP * np.maximum(1.0, np.abs(local[:, j]))
            up, down = local.copy(), local.copy()
            up[:, j] += h
            down[:, j] -= h
            difference = (rows(up, log_kappa) - rows(down, log_kappa)) / (2.0 * h[:, None])
            blocks[:, :, j] = difference[:, :4]
            coupling[:, j] = difference[:, 4]
        h = _POLISH_STEP * max(1.0, abs(log_kappa))
        difference = (rows(local, log_kappa + h) - rows(local, log_kappa - h)) / (2.0 * h)
        coupling = (coupling + difference[:, :4]) / 2.0
        corner = float(difference[:, 4].sum())
        blocks = (blocks + blocks.transpose(0, 2, 1)) / 2.0
        # Held coordinates: identity rows/columns, zero gradient and coupling (their step is exactly 0).
        held_local = held[:-1].reshape(n, 4)
        local_gradient = np.where(held_local, 0.0, gradient[:-1].reshape(n, 4))
        coupling = np.where(held_local, 0.0, coupling)
        blocks = np.where(held_local[:, :, None] | held_local[:, None, :], 0.0, blocks)
        blocks[:, np.arange(4), np.arange(4)] += held_local
        try:
            np.linalg.cholesky(blocks)
            solved_gradient = np.linalg.solve(blocks, local_gradient[..., None])[..., 0]
            solved_coupling = np.linalg.solve(blocks, coupling[..., None])[..., 0]
        except np.linalg.LinAlgError:
            return x, iteration - 1, False
        kappa_step = 0.0
        if not held[-1]:
            schur = corner - float(np.sum(coupling * solved_coupling))
            if not schur > 0.0:
                return x, iteration - 1, False
            kappa_step = -(float(gradient[-1]) - float(np.sum(coupling * solved_gradient))) / schur
        step = np.r_[(-solved_gradient - solved_coupling * kappa_step).ravel(), kappa_step]
        free = np.flatnonzero(~held)
        result = _polish_step(evaluate, x, f, gradient, step[free], free, lower, upper)
        if result is None:
            return x, iteration - 1, False
        x, f, gradient, moved = result
        if moved <= _POLISH_TOLERANCE or (moved <= _POLISH_STAGNATION and moved > previous / POLISH_STAGNATION_RATIO):
            return x, iteration, True
        previous = moved
    return x, _POLISH_ITERATIONS, False


# ---------------------------------------------------------------------------------------------
# Shared binomial model (EM)
# ---------------------------------------------------------------------------------------------


def _em(hist, depth, alt, initial, config: ReadModelParameters):
    """EM for P(ALT count | depth, sample) with ordered components e < q < 1 - e."""
    error, heterozygote = initial
    mixture = np.full((len(hist), 3), 1 / 3.0)
    sample_weight = hist.sum(axis=1)
    scale = max(1.0, float(sample_weight.sum()))
    old = -np.inf
    last_gain = None
    for iteration in range(config.em_maximum_iterations):
        values, mass, moments = em_statistics(hist, depth, alt, error, heterozygote, mixture)
        value = float(values.sum())  # the binomial coefficient cancels in comparisons
        if not np.isfinite(value) or value < old - EM_DECREASE_PER_OBSERVATION * scale:
            raise RuntimeError("read-model EM likelihood decreased or became non-finite")
        if iteration:
            last_gain = value - old
            if last_gain <= config.em_tolerance_per_observation * scale:
                break
        old = value
        mixture = np.divide(
            mass, sample_weight[:, None], out=np.full_like(mass, 1 / 3.0), where=sample_weight[:, None] > 0
        )
        totals = moments.sum(axis=0)
        error = float(totals[0] / max(totals[1], POSITIVE_WEIGHT_FLOOR))
        heterozygote = float(totals[2] / max(totals[3], POSITIVE_WEIGHT_FLOOR))
        if not (0 < error < heterozygote < 1 - error < 1):
            raise ValueError("EM left the ordered-component domain")
    else:
        raise RuntimeError("read-model EM did not converge")
    return dict(
        error=error,
        heterozygote_alt=heterozygote,
        mixture=mixture,
        log_likelihood=value,
        iterations=iteration + 1,
        last_gain=last_gain,
    )


def shared_objective(vector, hist, depth, alt):
    """Shared binomial error/balance and per-sample nuisance mixture logits."""
    local = np.empty((len(hist), 4))
    local[:, :2] = vector[:2]
    local[:, 2:] = vector[2:].reshape(len(hist), 2)
    value, gradients = _objective(local, 0.0, hist, depth, alt, np.zeros(2), np.zeros(2))
    return value, np.r_[gradients[:, :2].sum(axis=0), gradients[:, 2:4].ravel()]


def _shared_em_vector(endpoint):
    error, heterozygote = endpoint["error"], endpoint["heterozygote_alt"]
    mixture = np.maximum(endpoint["mixture"], POSITIVE_WEIGHT_FLOOR)
    return np.r_[
        logit(error),
        logit((heterozygote - error) / (1 - 2 * error)),
        (np.log(mixture[:, :2]) - np.log(mixture[:, 2, None])).ravel(),
    ]


class _ActiveSetCycle(RuntimeError):
    """The constrained Newton subproblem repeated an active face."""


def _shared_box_step(shared, blocks, coupling, gradient, lower, upper, fixed):
    """Minimise the structured Newton quadratic over its displacement box.

    Each face is solved by the same shared/nuisance Schur complement. Travel
    towards its minimiser until the first bound, adding all exactly tied
    blockers. At a feasible face minimiser, release the largest wrong-sign
    multiplier class. Neither incoming sample order nor a numerical threshold
    selects a constraint. Nonpositive curvature and repeated faces fail.
    """
    n = len(blocks)

    def product(step):
        top, local = step[:2], step[2:].reshape(n, 2)
        return np.r_[
            shared @ top + np.einsum("nli,nl->i", coupling, local),
            (np.einsum("nij,nj->ni", blocks, local) + np.einsum("nli,i->nl", coupling, top)).ravel(),
        ]

    def face_minimum(active, point):
        hs, hl = active[:2], active[2:].reshape(n, 2)
        a = np.where(hs[:, None] | hs[None, :], 0.0, shared)
        a[np.arange(2), np.arange(2)] += hs
        b = np.where(hl[:, :, None] | hl[:, None, :], 0.0, blocks)
        b[:, np.arange(2), np.arange(2)] += hl
        c = np.where(hl[:, :, None] | hs[None, None, :], 0.0, coupling)
        fixed_step = np.where(active, point, 0.0)
        rhs = gradient + product(fixed_step)
        gs = np.where(hs, 0.0, rhs[:2])
        gl = np.where(hl, 0.0, rhs[2:].reshape(n, 2))
        np.linalg.cholesky(b)
        bg = np.linalg.solve(b, gl[..., None])[..., 0]
        bc = np.linalg.solve(b, c)
        factor = cho_factor(a - np.einsum("nli,nlj->ij", c, bc))
        top = -cho_solve(factor, gs - np.einsum("nli,nl->i", c, bg))
        local = -bg - np.einsum("nli,i->nl", bc, top)
        return fixed_step + np.r_[top, local.ravel()]

    step = np.zeros_like(gradient)
    active = fixed | ((lower == 0.0) & (gradient > 0.0)) | ((upper == 0.0) & (gradient < 0.0))
    seen = set()
    while True:
        face = np.where(fixed, 2, np.where(active, np.where(step == lower, -1, 1), 0))
        key = face.tobytes()
        if key in seen:
            raise _ActiveSetCycle("Shared Newton quadratic repeated an active face")
        seen.add(key)
        target = face_minimum(active, step)
        below, above = (~active & (target < lower)), (~active & (target > upper))
        if np.any(below | above):
            direction = target - step
            fractions = np.full(len(step), np.inf)
            fractions[below] = (lower[below] - step[below]) / direction[below]
            fractions[above] = (upper[above] - step[above]) / direction[above]
            fraction = float(fractions.min())
            blocking = fractions == fraction
            step += fraction * direction
            step[blocking & below] = lower[blocking & below]
            step[blocking & above] = upper[blocking & above]
            active |= blocking
            continue
        step = target
        slope = gradient + product(step)
        wrong = np.where(
            active & ~fixed & (step == lower), -slope, np.where(active & ~fixed & (step == upper), slope, 0.0)
        )
        largest = float(wrong.max(initial=0.0))
        if largest <= 0.0:
            return step
        active[wrong == largest] = False


def _polish_shared(vector, hist, depth, alt, *, diagnostics=None):
    """Existing Newton policy with a two-shared-coordinate block Hessian.

    Samples have independent two-coordinate nuisance mixtures conditional on the
    shared error and balance. The shared block uses central differences; analytic
    covariance terms preserve nearly absent nuisance components and their shared
    coupling. Each Newton direction solves the box-constrained quadratic via
    the structured Schur system. Bounds, backtracking and stopping are unchanged.
    """

    def finish(point, iterations, converged, reason):
        if diagnostics is not None:
            diagnostics["termination"] = reason
        return point, iterations, converged

    n = len(hist)
    bounds = [_ERROR_LOGIT_BOUNDS, _BALANCE_LOGIT_BOUNDS] + [_MIXING_BOUNDS] * (2 * n)
    lower, upper = np.asarray(bounds, float).T
    x = np.clip(np.asarray(vector, float).copy(), lower, upper)
    scale = max(1.0, float(hist.sum()))
    kernel = samples_objective_serial if n == 1 else samples_objective_parallel

    def evaluate(point):
        return shared_objective(point, hist, depth, alt)

    def rows(local):
        return -kernel(local, 0.0, hist, depth, alt, np.zeros(2), np.zeros(2))[1][:, :4] / scale

    f, gradient = evaluate(x)
    previous = np.inf
    for iteration in range(1, _POLISH_ITERATIONS + 1):
        held = _held(x, gradient, lower, upper)
        if held.all():
            return finish(x, iteration - 1, True, "bound_active")
        local = np.column_stack((np.broadcast_to(x[:2], (n, 2)), x[2:].reshape(n, 2)))
        shared = np.empty((2, 2))
        for j in range(2):
            h = _POLISH_STEP * np.maximum(1.0, np.abs(local[:, j]))
            up, down = local.copy(), local.copy()
            up[:, j] += h
            down[:, j] -= h
            difference = (rows(up) - rows(down)) / (2.0 * h[:, None])
            shared[:, j] = difference[:, :2].sum(axis=0)
        shared = (shared + shared.T) / 2.0
        blocks, coupling = shared_nuisance_curvature(local, hist, depth, alt)
        blocks /= scale
        coupling /= scale
        diagonal = np.r_[np.diag(shared), np.diagonal(blocks, axis1=1, axis2=2).ravel()]

        fixed = lower == upper

        try:
            step = _shared_box_step(shared, blocks, coupling, gradient, lower - x, upper - x, fixed)
        except _ActiveSetCycle:
            return finish(x, iteration - 1, False, "active_set_cycle")
        except (LinAlgError, ValueError):
            # The existing coordinate-curvature floor/fallback remains unchanged.
            free = ~held
            kept = diagonal > _POLISH_CURVATURE_FLOOR * max(float(diagonal[free].max(initial=0.0)), 0.0)
            if kept[free].all() or not kept[free].any():
                return finish(x, iteration - 1, False, "curvature")
            fixed = fixed | ~kept
            try:
                step = _shared_box_step(shared, blocks, coupling, gradient, lower - x, upper - x, fixed)
            except _ActiveSetCycle:
                return finish(x, iteration - 1, False, "active_set_cycle")
            except (LinAlgError, ValueError):
                return finish(x, iteration - 1, False, "curvature")
        free = np.flatnonzero(~fixed)
        result = _polish_step(evaluate, x, f, gradient, step[free], free, lower, upper)
        if result is None:
            return finish(x, iteration - 1, False, "backtracking")
        x, f, gradient, moved = result
        if moved <= _POLISH_TOLERANCE:
            return finish(x, iteration, True, "small_step")
        if moved <= _POLISH_STAGNATION and moved > previous / POLISH_STAGNATION_RATIO:
            return finish(x, iteration, True, "stagnation")
        previous = moved
    return finish(x, _POLISH_ITERATIONS, False, "iteration_limit")


def _shared_diagnostics(record):
    """Retain native start/failure fields as the D1 scientific fallback contract."""
    if record is None:
        return None
    starts = []
    for row in record["starts"]:
        values = dict(row)
        for name in ("initial", "em_vector", "vector"):
            if name in values:
                values[name] = tuple(values[name])
        starts.append(SharedStart(**values))
    return SharedFitDiagnostics(
        starts=tuple(starts),
        pairwise_distances=tuple(SharedDistance(**r) for r in record["pairwise_distances"]),
        reason=record.get("reason"),
        successful_start_indices=tuple(record.get("successful_start_indices", ())),
        best_successful_start=record.get("best_successful_start"),
        likelihood_tie_nats=record.get("likelihood_tie_nats"),
    )


class SharedStationarityError(RuntimeError):
    """No resolved shared fit; calibration records diagnostics and uses the fixed model."""

    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.diagnostics = _shared_diagnostics(diagnostics)


def fit_shared(hist, depth, alt, config: ReadModelParameters) -> ReadModelFit:
    """Polish every start and retain a representative of equivalent stationary fits.

    Shared logits use the existing small-step resolution, nuisance logits the
    existing stagnation resolution. This is numerical equivalence, not an error
    bound. Maximum training likelihood then lexicographic parameters chooses the
    representative. Distinct competitors must lose by more than the likelihood
    tie band; otherwise the contig uses the fixed model. A failed polish is a failed start.
    """
    # Fix only the floating-point evaluation order, using training data content.
    # The objective and weights are unchanged. Equal histogram rows are identical
    # terms; no sample label or incoming row position chooses a numerical endpoint.
    order = np.asarray(sorted(range(len(hist)), key=lambda i: hist[i].tobytes()), dtype=np.int64)
    undo = np.argsort(order)
    hist = np.ascontiguousarray(hist[order])
    candidates, failed, start_records = [], [], []
    for initial in config.em_starts:
        start: dict[str, Any] = dict(initial=list(initial))
        start_records.append(start)
        try:
            endpoint = _em(hist, depth, alt, initial, config)
        except (ValueError, RuntimeError) as error:
            failed.append(str(error))
            start.update(failure=str(error), phase="em")
            continue
        polish: dict[str, Any] = {}
        try:
            initial_vector = _shared_em_vector(endpoint)
            start.update(em_vector=initial_vector.tolist(), em_log_likelihood=endpoint["log_likelihood"])
            vector, steps, converged = _polish_shared(initial_vector, hist, depth, alt, diagnostics=polish)
            value, gradient = shared_objective(vector, hist, depth, alt)
        except (ValueError, RuntimeError, LinAlgError) as error:
            failed.append(str(error))
            start.update(failure=str(error), phase="polish", em_iterations=endpoint["iterations"])
            continue
        candidate = dict(
            vector=vector,
            endpoint=endpoint,
            polish_steps=steps,
            polish_converged=converged,
            polish_termination=polish.get("termination"),
            gradient_max=float(np.max(np.abs(gradient), initial=0.0)),
            log_likelihood=-value * max(1.0, float(hist.sum())),
        )
        start.update(
            vector=vector.tolist(),
            train_log_likelihood=candidate["log_likelihood"],
            em_iterations=endpoint["iterations"],
            polish_iterations=steps,
            polish_converged=converged,
            polish_termination=polish.get("termination"),
        )
        if not converged:
            reason = f"Shared-binomial stationary polish did not converge: {polish.get('termination')}"
            failed.append(reason)
            start.update(failure=reason, phase="polish")
            continue
        candidates.append(candidate)
    if not candidates:
        reason = f"No shared-binomial EM fit converged: {failed}"
        raise SharedStationarityError(
            reason, dict(reason="no_shared_start_converged", starts=start_records, pairwise_distances=[])
        )
    vectors = np.asarray([c["vector"] for c in candidates])
    # Numerical equivalence, not an error bound: shared observation parameters
    # retain the small-step resolution; nuisance logits use the existing Newton
    # stagnation resolution. Both distances are checked over the complete family.
    shared_distance = nuisance_distance = 0.0
    distances = []
    distinct = np.zeros((len(vectors), len(vectors)), dtype=bool)
    for i in range(len(vectors)):
        for j in range(i):
            scale = np.maximum(1.0, np.maximum(np.abs(vectors[i]), np.abs(vectors[j])))
            difference = np.abs(vectors[i] - vectors[j]) / scale
            shared = float(np.max(difference[:2], initial=0.0))
            nuisance = float(np.max(difference[2:], initial=0.0))
            shared_distance = max(shared_distance, shared)
            nuisance_distance = max(nuisance_distance, nuisance)
            distinct[i, j] = distinct[j, i] = shared > _POLISH_TOLERANCE or nuisance > _POLISH_STAGNATION
            distances.append(dict(first=i, second=j, shared=shared, nuisance=nuisance))
    # Largest training likelihood, then lexicographically smallest parameter vector.
    # Start order, duplicates, convention reversal and held-out data have no vote.
    best_index = min(
        range(len(candidates)), key=lambda i: (-candidates[i]["log_likelihood"], tuple(candidates[i]["vector"]))
    )
    best = candidates[best_index]
    warning = None
    diagnostics_policy = None
    if failed or distinct.any():
        warning = "shared_start_failed" if failed else None
        diagnostics_policy = dict(
            starts=start_records,
            pairwise_distances=distances,
            successful_start_indices=[i for i, row in enumerate(start_records) if "failure" not in row],
            best_successful_start=best_index,
            likelihood_tie_nats=_LIKELIHOOD_TIE_NATS,
        )
        if distinct.any():
            resolved = all(
                best["log_likelihood"] - candidate["log_likelihood"] > _LIKELIHOOD_TIE_NATS
                for i, candidate in enumerate(candidates)
                if distinct[best_index, i]
            )
            if not resolved:
                diagnostics_policy["reason"] = "shared_endpoints_unresolved"
                raise SharedStationarityError("shared_endpoints_unresolved", diagnostics_policy)
            warning = "shared_endpoints_resolved" + ("; shared_start_failed" if failed else "")
        diagnostics_policy["reason"] = warning
    vector = best["vector"]
    n = len(hist)
    top = max(c["log_likelihood"] for c in candidates)
    band = _EM_START_BAND_PER_OBSERVATION * max(1.0, float(hist.sum()))
    diagnostics = [
        dict(
            train_log_likelihood=c["log_likelihood"],
            em_iterations=c["endpoint"]["iterations"],
            polish_iterations=c["polish_steps"],
            polish_converged=c["polish_converged"],
            polish_termination=c["polish_termination"],
            gradient_max=c["gradient_max"],
        )
        for c in candidates
    ]
    return ReadModelFit(
        name=BINOMIAL,
        error_logit=np.full(n, vector[0]),
        balance_logit=np.full(n, vector[1]),
        log_concentration=None,
        mixing=vector[2:].reshape(n, 2)[undo].copy(),
        success=True,
        details=SharedFitDetails(
            fit_diagnostics=_shared_diagnostics(
                dict(
                    starts=start_records,
                    pairwise_distances=distances,
                    successful_start_indices=[i for i, row in enumerate(start_records) if "failure" not in row],
                    best_successful_start=best_index,
                    likelihood_tie_nats=_LIKELIHOOD_TIE_NATS,
                    reason=warning,
                )
            ),
            message="shared-binomial EM with stationary polish",
            iterations=best["endpoint"]["iterations"],
            failed_starts=tuple(failed),
            train_log_likelihood=best["log_likelihood"],
            starts_in_band=sum(c["log_likelihood"] >= top - band for c in candidates),
            stationary_relative_distance=max(shared_distance, nuisance_distance),
            stationary_shared_relative_distance=shared_distance,
            stationary_nuisance_relative_distance=nuisance_distance,
            stationary_shared_resolution=_POLISH_TOLERANCE,
            stationary_nuisance_resolution=_POLISH_STAGNATION,
            stationary_representative="maximum_training_ll_then_lexicographic_parameters",
            stationary_sample_order="training_histogram_content",
            stationary_starts=tuple(StationaryStart(**row) for row in diagnostics),
            **(
                {}
                if warning is None
                else dict(shared_fit_warning=warning, shared_fit_diagnostics=_shared_diagnostics(diagnostics_policy))
            ),
        ),
    )


# ---------------------------------------------------------------------------------------------
# Pooled sample effects (empirical Bayes) and the joint concentration
# ---------------------------------------------------------------------------------------------


def _fit_sample(task, depth, alt, shared: ReadModelFit, prior=None):
    """One sample's binomial fit from the shared model; returns (vector, variances, success, polished).

    vector = (logit e, logit balance, mixing 0, mixing 1). Without a prior, the variances of the
    two logits come from the observed Hessian (genotype-mixture uncertainty profiled out); with
    a prior (mean, variance), effects of zero prior variance stay at the prior mean.
    """
    index, counts = task
    vector = shared.local()[index].copy()
    if counts.sum() == 0:
        # With no training reads the normal prior is the entire objective: use its mode,
        # including exactly pooled (zero-variance) effects.
        if prior is not None:
            vector[:2] = prior[0]
        return vector, np.full(2, np.inf), True, True
    free = np.ones(len(vector), bool)
    mean, variance = (np.zeros(2), np.zeros(2)) if prior is None else prior
    if prior is not None:
        vector[:2] = prior[0]
        free[:2] = prior[1] > 0
    bounds = [_ERROR_LOGIT_BOUNDS, _BALANCE_LOGIT_BOUNDS, _MIXING_BOUNDS, _MIXING_BOUNDS]
    histogram = counts[None]

    def reduced(x):
        current = vector.copy()
        current[free] = x
        value, gradients = _objective(current[None], 0.0, histogram, depth, alt, mean, variance)
        return value, gradients[0, :4][free]

    active_bounds = [bound for bound, active in zip(bounds, free, strict=False) if active]
    fitted = minimize(
        reduced, vector[free], jac=True, method="L-BFGS-B", bounds=active_bounds, options=_SAMPLE_FIT_OPTIONS
    )
    # Newton polish to the optimum itself (release 2.1): the result no longer depends on L-BFGS-B's path.
    polished, _, polish_converged = _polish(
        reduced, fitted.x, [bound[0] for bound in active_bounds], [bound[1] for bound in active_bounds]
    )
    vector[free] = polished
    variances = np.full(2, np.inf)
    if prior is None:
        # A small observed Hessian profiles out the sample's genotype-mixture uncertainty.
        indices = np.flatnonzero(free)
        hessian = np.empty((len(indices), len(indices)))
        zeros = np.zeros(2)
        for column, j in enumerate(indices):
            upper, lower = vector.copy(), vector.copy()
            upper[j] += _HESSIAN_STEP
            lower[j] -= _HESSIAN_STEP
            gu = _objective(upper[None], 0.0, histogram, depth, alt, zeros, zeros)[1][0, :4]
            gl = _objective(lower[None], 0.0, histogram, depth, alt, zeros, zeros)[1][0, :4]
            hessian[:, column] = (gu[free] - gl[free]) / (2 * _HESSIAN_STEP) * counts.sum()
        hessian = (hessian + hessian.T) / 2
        if np.linalg.eigvalsh(hessian).min(axis=0) > _HESSIAN_EIGENVALUE_FLOOR:
            variances = np.diag(np.linalg.inv(hessian))[:2]
    return vector, variances, bool(fitted.success), bool(polish_converged)


def _normal_hyperfit(values, variances, fallback):
    """Normal measurement-error hyperparameters (mean, tau) by profile likelihood."""
    valid = np.isfinite(variances) & (variances > 0) & np.isfinite(values)
    if valid.sum() < _HYPERFIT_MINIMUM_SAMPLES:
        return float(fallback), 0.0
    y, v = values[valid], variances[valid]

    def profile(tau):
        weights = 1 / (v + tau)
        mu = np.sum(weights * y) / weights.sum()
        return 0.5 * np.sum(np.log(v + tau) + (y - mu) ** 2 / (v + tau)), mu

    def slope(tau):
        """d profile / d tau at the profiled mean (envelope theorem)."""
        weights = 1 / (v + tau)
        mu = np.sum(weights * y) / weights.sum()
        return 0.5 * float(np.sum(weights - (y - mu) ** 2 * weights**2))

    ceiling = max(1.0, HYPERFIT_VARIANCE_CEILING_FACTOR * float(np.var(y)))
    optimum = minimize_scalar(lambda t: profile(t)[0], bounds=(0.0, ceiling), method="bounded")
    tau = _profile_root(slope, float(optimum.x), ceiling)
    band = _HYPERFIT_BOUNDARY_BAND * max(1.0, abs(float(profile(tau)[0])))
    tau = 0.0 if profile(0.0)[0] <= profile(tau)[0] + band else tau
    return float(profile(tau)[1]), tau


def _profile_root(slope, start, ceiling):
    """The stationary point of the hyperfit profile near Brent's bounded minimum ``start`` (xatol 1e-5): the root of
    its derivative, bracketed around ``start`` and solved to rounding precision (release 2.1), so tau does not depend
    on Brent's path; ``start`` when no sign change brackets it (a minimum at a bound)."""
    if slope(ceiling) < 0.0:
        return float(ceiling)
    width = PROFILE_BRACKET_INITIAL_WIDTH * max(1.0, start)
    for _ in range(PROFILE_BRACKET_TRIALS):
        low, high = max(0.0, start - width), min(ceiling, start + width)
        below, above = slope(low), slope(high)
        if below <= 0.0 <= above:
            if below == 0.0:
                return float(low)
            if above == 0.0:
                return float(high)
            return float(
                brentq(
                    slope,
                    low,
                    high,
                    xtol=PROFILE_ROOT_XTOL,
                    rtol=PROFILE_ROOT_EPS_FACTOR * np.finfo(float).eps,
                    maxiter=PROFILE_ROOT_ITERATIONS,
                )
            )
        if low == 0.0 and high == ceiling:
            break
        width *= PROFILE_BRACKET_GROWTH
    return float(start)


def fit_pooled_samples(hist, depth, alt, shared: ReadModelFit, team: int) -> ReadModelFit:
    """Per-sample binomial effects shrunk by the empirical-Bayes prior (one sample per thread)."""
    tasks = list(enumerate(hist))
    worker = partial(_fit_sample, depth=depth, alt=alt, shared=shared)
    with ThreadPoolExecutor(max_workers=team) as pool:
        preliminary = list(pool.map(worker, tasks))
    estimates = np.asarray([x[0] for x in preliminary])
    variances = np.asarray([x[1] for x in preliminary])
    shared_local = shared.local()
    hyper = [_normal_hyperfit(estimates[:, j], variances[:, j], shared_local[0, j]) for j in range(2)]
    mean, variance = np.asarray(hyper).T
    worker = partial(worker, prior=(mean, variance))
    with ThreadPoolExecutor(max_workers=team) as pool:
        final = list(pool.map(worker, tasks))
    fitted = np.asarray([x[0] for x in final])
    return ReadModelFit(
        name=SAMPLE_BINOMIAL,
        error_logit=fitted[:, 0],
        balance_logit=fitted[:, 1],
        log_concentration=None,
        mixing=fitted[:, 2:],
        success=all(x[2] for x in final),
        details=PooledFitDetails(
            sample_fit_failures=sum(not x[2] for x in final),
            polish_failures=sum(not x[3] for x in (*preliminary, *final)),
            identifiable_sample_effects=tuple(np.isfinite(variances).sum(axis=0).tolist()),
            hyper_mean=mean,
            hyper_variance=variance,
            approximation="diagonal Laplace empirical Bayes",
        ),
    )


def _local_bounds(mean, variance, n):
    """Per-sample box constraints; an effect with zero prior variance is fixed at its mean."""
    per_sample = [_ERROR_LOGIT_BOUNDS, _BALANCE_LOGIT_BOUNDS, _MIXING_BOUNDS, _MIXING_BOUNDS]
    for j in range(2):
        if variance[j] == 0:
            per_sample[j] = (mean[j], mean[j])
    return per_sample * n


def fit_joint(hist, depth, alt, pooled: ReadModelFit, config: ReadModelParameters) -> ReadModelFit:
    """Pooled sample effects refitted with one shared heterozygote concentration.

    Starts from the pooled sample-binomial fit and kappa = ``initial_heterozygote_concentration``;
    the empirical-Bayes prior of the pooled fit is kept. Uses the caller's Numba team.
    """
    n = len(hist)
    details = cast(PooledFitDetails, pooled.details)
    mean = np.asarray(details.hyper_mean)
    variance = np.asarray(details.hyper_variance)
    initial = np.r_[pooled.local().ravel(), np.log(config.initial_heterozygote_concentration)]
    bounds = _local_bounds(mean, variance, n) + [_LOG_CONCENTRATION_BOUNDS]
    result = minimize(
        joint_objective,
        initial,
        args=(hist, depth, alt, mean, variance),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options=_JOINT_FIT_OPTIONS,
    )
    # Newton polish to the optimum itself (release 2.1): the result no longer depends on L-BFGS-B's path.
    polished, steps, converged = _polish_joint(
        result.x, hist, depth, alt, mean, variance, [bound[0] for bound in bounds], [bound[1] for bound in bounds]
    )
    local = polished[:-1].reshape(n, 4)
    return ReadModelFit(
        name=SAMPLE_JOINT,
        error_logit=local[:, 0],
        balance_logit=local[:, 1],
        log_concentration=float(polished[-1]),
        mixing=local[:, 2:],
        success=bool(result.success),
        details=JointFitDetails(
            message=str(result.message),
            iterations=int(result.nit),
            hyper_mean=mean,
            hyper_variance=variance,
            polish_steps=int(steps),
            polish_converged=bool(converged),
            polish_shift=float(np.max(np.abs(polished - result.x), initial=0.0)),
        ),
    )


__all__ = [
    "ReadModelFit",
    "fixed_model",
    "joint_objective",
    "log_predictive",
    "shared_objective",
    "SharedStationarityError",
    "fit_shared",
    "fit_pooled_samples",
    "fit_joint",
]
