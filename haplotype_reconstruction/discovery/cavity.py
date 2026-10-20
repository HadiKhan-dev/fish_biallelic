"""The regularised fixed-assignment leave-one-sample-out cavity score of a panel.

For a complete panel H with assignments A (from the full data), each sample is held out in
turn: its uniformly weighted tied-state counts are deleted from the Dirichlet state prior (concentration alpha over the
row-pair, row-unknown and unknown-unknown states, prior base masses from
``CavityScoreConfig``) and all its weighted likelihood terms are removed from the founder-allele Markov
random field; all distinct mean-field endpoints tied under the best converged training ELBO
predict the held-out sample's genotype likelihood under every state. These likelihoods are
averaged before the floor and logarithm; founder marginals are never averaged. Exact-distinct
endpoints connected within the existing mean-field convergence tolerance form one equivalence
class. Classes have equal weight, with a uniform likelihood average inside each class. The
sample's cavity log predictive is log sum over states of prior x predictive. The panel's score
is the sum over samples plus the telescoping K prior -log K - log(K + 1) plus minus the code
length log C(2^L, K) of an unordered set of K distinct binary rows. Missing cells are neutral
(the evidence there is exactly 1/3 for every state, a constant that cancels across K).

Scores are uncalibrated pseudo-scores: the panel and A were selected from the full data.

Release 2.3 deliberately changes the approximate predictive functional in tied-endpoint cases.
Eligibility and the solution's ``nll`` ELBO tie quantum are unchanged (``FixedKSolution.quanta``;
the floors for solutions without quanta). No held-out evidence chooses the endpoint weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..execution import claim_threads
from .cavity_kernels import (_build_evidence_emission_cache_kernel, _floor_log_likelihood_kernel,
                             _score_mode_cavity_kernel)
from ..maths.ties import BlockQuanta
from .conventions import rev
from .panel import FixedKSolution, log_haplotype_set_count, restricted, solution_tie_mask


def shared_likelihood_triples(likelihood: np.ndarray) -> np.ndarray:
    """bool[N, L]: the (sample, site) likelihood triple equals another sample's at that site
    (only such cells can join an exact held-out group)."""
    order = np.lexsort((likelihood[:, :, 2], likelihood[:, :, 1], likelihood[:, :, 0]), axis=0)
    ordered = np.take_along_axis(likelihood, order[:, :, None], axis=0)
    equal_to_next = np.all(ordered[1:] == ordered[:-1], axis=2)
    shared_in_order = np.zeros(order.shape, dtype=np.bool_)
    shared_in_order[1:] |= equal_to_next
    shared_in_order[:-1] |= equal_to_next
    shared = np.empty_like(shared_in_order)
    np.put_along_axis(shared, order, shared_in_order, axis=0)
    return shared


def _state_geometry(k: int, config) -> tuple[np.ndarray, ...]:
    """(kind, first, second, prior base) of the states: row pairs (kind 0, row-major), row with
    an unknown copy (kind 1), two unknown copies (kind 2); prior base normalised to one."""
    pair_i, pair_j = np.triu_indices(k)
    n_rr = len(pair_i)
    n_states = n_rr + k + 1
    kind = np.empty(n_states, dtype=np.int64)
    first = np.full(n_states, -1, dtype=np.int64)
    second = np.full(n_states, -1, dtype=np.int64)
    kind[:n_rr] = 0
    first[:n_rr] = pair_i
    second[:n_rr] = pair_j
    kind[n_rr:n_rr + k] = 1
    first[n_rr:n_rr + k] = np.arange(k)
    kind[-1] = 2
    omega = config.unknown_prior_mass
    delta = config.double_unknown_fraction
    base = np.empty(n_states, dtype=np.float64)
    base[:n_rr] = (1.0 - omega) * np.where(pair_i == pair_j, 1.0, 2.0) / float(k * k)
    base[n_rr:n_rr + k] = omega * (1.0 - delta) / k
    base[-1] = omega * delta
    base /= np.sum(base)
    return tuple(np.ascontiguousarray(value) for value in (kind, first, second, base))


class CavityWorkspace:
    """Evidence-only inputs of the cavity score of one block (the samples with depth)."""

    def __init__(self, likelihood: np.ndarray, config):
        self.config = config
        self.likelihood = np.ascontiguousarray(likelihood, dtype=np.float64)
        floor = float(config.likelihood_floor)
        self.log_likelihood = _floor_log_likelihood_kernel(self.likelihood, floor)
        self.rw_log_likelihood, self.ww_log_emission = _build_evidence_emission_cache_kernel(
            self.likelihood, floor)
        self.shared_triple = shared_likelihood_triples(self.likelihood)
        self._geometry: dict[int, tuple] = {}
        # Cumulative held-out sample/site counts by family size 1,2,3.
        # Rows are exact-distinct endpoints and endpoint-equivalence classes.
        self.endpoint_family_counts = np.zeros((2, 3), dtype=np.int64)

    def geometry(self, k: int):
        value = self._geometry.get(k)
        if value is None:
            value = self._geometry[k] = _state_geometry(k, self.config)
        return value


@dataclass(frozen=True)
class PanelScore:
    """The cavity score of one solution (``solution`` keeps every sample).

    log_score = cavity_log_predictive + log_k_prior + log_set_code (in that order);
    mean_field_failures: held-out site solves whose starts all failed to converge.
    """

    solution: FixedKSolution
    cavity_log_predictive: float
    log_k_prior: float
    log_set_code: float
    log_score: float
    mean_field_failures: int

    @property
    def k(self) -> int:
        return self.solution.k


def _elbo_tie(solution) -> float:
    """The start-choice tie quantum of a solution's held-out solves: its block's ``nll`` quantum."""
    quanta = solution.quanta if solution.quanta is not None else BlockQuanta.floors()
    return quanta.nll


def score_solutions(workspace: CavityWorkspace, solutions, active: np.ndarray, tie_masks=None):
    """Cavity scores of solutions of the block (``active``: bool[N] samples with depth; the
    workspace holds exactly those samples). By default the solution carries its tied A-step states.
    ``tie_masks`` optionally supplies those uint8[N, states] masks explicitly. Each member of a
    sample's winning tied category contributes 1/|T| to both the state prior and founder field."""
    # A phase boundary before the parallel held-out kernel: late blocks claim freed cores.
    claim_threads()
    if not solutions:
        return ()
    config = workspace.config
    all_active = bool(np.all(active))
    samples = None if all_active else np.flatnonzero(active)
    n_sites = workspace.likelihood.shape[1]
    scores = []
    for index, solution in enumerate(solutions):
        scored = solution if samples is None else restricted(solution, samples)
        kind, first, second, base = workspace.geometry(solution.k)
        if tie_masks is None:
            mask = solution_tie_mask(scored)
        else:
            mask = tie_masks[index] if samples is None else tie_masks[index][samples]
            mask = np.ascontiguousarray(mask, dtype=np.uint8)
        sample_scores, failures = _score_mode_cavity_kernel(
            workspace.likelihood, workspace.log_likelihood, workspace.rw_log_likelihood,
            workspace.ww_log_emission, np.ascontiguousarray(scored.haplotypes, dtype=np.int64),
            np.ascontiguousarray(scored.assignments, dtype=np.int64), kind, first, second, base,
            float(config.state_concentration), int(config.mean_field_max_iterations),
            float(config.mean_field_tolerance), float(config.likelihood_floor),
            workspace.shared_triple, float(_elbo_tie(solution)), rev("d12"), mask,
            workspace.endpoint_family_counts, float(config.mean_field_tolerance))
        total = math.fsum(float(value) for value in sample_scores)
        k = solution.k
        log_k_prior = -math.log(k) - math.log(k + 1)
        log_set_code = -log_haplotype_set_count(n_sites, k)
        scores.append(PanelScore(
            solution=solution, cavity_log_predictive=total, log_k_prior=log_k_prior,
            log_set_code=log_set_code, log_score=total + log_k_prior + log_set_code,
            mean_field_failures=int(failures)))
    return tuple(scores)
