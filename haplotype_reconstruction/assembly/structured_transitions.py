"""Sparse-plus-background homologue transitions for structured assembly.

T[i,j] = S[i,j] + u[i] q[j], with nonnegative sparse S, q.sum()=1,
and S.sum(axis=1)+u=1. The diploid distribution is NOT factorized.
All contractions, including the adjoint and EM sufficient statistics, cost
O(N*K**2*d) for O(K*d) explicit edges. Dense arrays are materialized only
for mesh export or independent validation, never for an HMM contraction.
This is a restricted statistical model, not an approximation advertised as
equivalent to an arbitrary learned dense transition.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numba import njit, prange


@dataclass(frozen=True)
class StructuredTransitionConfig:
    minimum_degree: int = 2
    degree_log_multiplier: float = 1.0
    initial_specific_mass: float = 0.5
    row_prior_strength: float = 1.0
    prior_specific_mass: float = 0.1
    destination_prior_strength: float = 1.0
    minimum_background_mass: float = 0.01

    def __post_init__(self):
        if self.minimum_degree < 1 or self.degree_log_multiplier <= 0:
            raise ValueError("structured transition degree must be positive")
        for value in (self.initial_specific_mass, self.prior_specific_mass,
                      self.minimum_background_mass):
            if not 0 < value < 1:
                raise ValueError("structured transition masses must be in (0,1)")
        if self.row_prior_strength <= 0 or self.destination_prior_strength <= 0:
            raise ValueError("structured transition prior strengths must be positive")

    def degree(self, destinations):
        return min(destinations, max(self.minimum_degree,
            int(math.ceil(self.degree_log_multiplier * math.log2(max(2, destinations))))))


def configured_transition():
    from ..core.environment import assembly_transition_model
    return StructuredTransitionConfig() if assembly_transition_model() == "structured" else None


@dataclass(frozen=True)
class StructuredTransition:
    offsets: np.ndarray
    destinations: np.ndarray
    log_specific: np.ndarray
    log_source_background: np.ndarray
    log_destination: np.ndarray

    @property
    def shape(self):
        return len(self.log_source_background), len(self.log_destination)

    def dense(self):
        result = np.exp(self.log_source_background[:, None] + self.log_destination[None, :])
        for source in range(self.shape[0]):
            lo, hi = self.offsets[source:source+2]
            result[source, self.destinations[lo:hi]] += np.exp(self.log_specific[lo:hi])
        return result

    def transpose(self):
        """The adjoint operator, NOT a row-normalized reverse conditional."""
        left, right = self.shape
        source = np.repeat(np.arange(left), np.diff(self.offsets))
        order = np.argsort(self.destinations, kind="stable")
        offsets = np.r_[0, np.cumsum(np.bincount(self.destinations, minlength=right))]
        return StructuredTransition(np.ascontiguousarray(offsets, dtype=np.int64),
            np.ascontiguousarray(source[order], dtype=np.int64),
            np.ascontiguousarray(self.log_specific[order]),
            self.log_destination, self.log_source_background)


@njit(cache=True)
def _transpose_apply(values, offsets, destinations, specific, source_bg, destination_bg):
    """log(T.T @ exp(values)); works for either orientation of T."""
    left, columns = values.shape
    right = len(destination_bg)
    result = np.empty((right, columns))
    for column in range(columns):
        background = -np.inf
        for source in range(left):
            background = np.logaddexp(background, source_bg[source] + values[source, column])
        for target in range(right):
            result[target, column] = destination_bg[target] + background
        for source in range(left):
            value = values[source, column]
            for edge in range(offsets[source], offsets[source+1]):
                target = destinations[edge]
                result[target, column] = np.logaddexp(
                    result[target, column], specific[edge] + value)
    return result


@njit(cache=True, parallel=True)
def _propagate(scores, offsets, destinations, specific, source_bg, destination_bg):
    left, right = len(source_bg), len(destination_bg)
    result = np.empty((len(scores), right*right))
    for sample in prange(len(scores)):
        matrix = scores[sample].reshape((left, left))
        first = _transpose_apply(matrix, offsets, destinations, specific, source_bg, destination_bg)
        second = _transpose_apply(first.T, offsets, destinations, specific, source_bg, destination_bg)
        result[sample] = second.T.flatten()
    return result


def propagate(scores, transition):
    safe_background=(np.min(transition.log_source_background)
                     +np.min(transition.log_destination)>-100.)
    kernel=_propagate_probability if safe_background else _propagate
    return kernel(np.ascontiguousarray(scores), transition.offsets,
        transition.destinations, transition.log_specific,
        transition.log_source_background, transition.log_destination)


@njit(cache=True)
def _log_matvec(matrix, vector, transpose=False):
    rows = matrix.shape[1] if transpose else matrix.shape[0]
    columns = matrix.shape[0] if transpose else matrix.shape[1]
    result = np.full(rows, -np.inf)
    for row in range(rows):
        for column in range(columns):
            entry = matrix[column, row] if transpose else matrix[row, column]
            result[row] = np.logaddexp(result[row], entry + vector[column])
    return result


@njit(cache=True, parallel=True)
def _statistics(forward, backward, offsets, destinations, specific, source_bg, destination_bg):
    """Latent specific-edge counts and background source/target marginals.

    Let M=A*T and U=A.T*T. Derivative W=M*B.T + U*B is required only
    on explicit edges; background counts require W*q and u.T*W, computed
    through matrix-vector contractions without ever constructing dense W.
    Specific counts plus background source counts sum to two per sample.
    """
    samples, left, right = len(forward), len(source_bg), len(destination_bg)
    edge_count = len(specific)
    edges = np.empty((samples, edge_count))
    source = np.empty((samples, left))
    target = np.empty((samples, right))
    log_normalizer = np.empty(samples)
    for sample in prange(samples):
        a = forward[sample].reshape((left, left))
        b = backward[sample].reshape((right, right))
        m = _transpose_apply(a.T, offsets, destinations, specific, source_bg, destination_bg).T
        u = _transpose_apply(a, offsets, destinations, specific, source_bg, destination_bg).T
        predicted = _transpose_apply(m, offsets, destinations, specific, source_bg, destination_bg)
        z = -np.inf
        for first in range(right):
            for second in range(right):
                z = np.logaddexp(z, predicted[first, second] + b[first, second])
        log_normalizer[sample] = z
        for first in range(left):
            for edge in range(offsets[first], offsets[first+1]):
                second = destinations[edge]
                derivative = -np.inf
                for partner in range(right):
                    derivative = np.logaddexp(derivative, m[first, partner] + b[second, partner])
                    derivative = np.logaddexp(derivative, u[first, partner] + b[partner, second])
                edges[sample, edge] = math.exp(specific[edge] + derivative - z)
        bq = _log_matvec(b, destination_bg)
        btq = _log_matvec(b, destination_bg, True)
        m_btq = _log_matvec(m, btq)
        u_bq = _log_matvec(u, bq)
        for first in range(left):
            source[sample, first] = math.exp(source_bg[first]
                + np.logaddexp(m_btq[first], u_bq[first]) - z)
        mtu = _log_matvec(m, source_bg, True)
        utu = _log_matvec(u, source_bg, True)
        b_mtu = _log_matvec(b, mtu)
        bt_utu = _log_matvec(b, utu, True)
        for second in range(right):
            target[sample, second] = math.exp(destination_bg[second]
                + np.logaddexp(b_mtu[second], bt_utu[second]) - z)
    return edges, source, target, log_normalizer


@njit(cache=True)
def _right_apply_probability(a, offsets, destinations, specific, source_bg, destination_bg):
    """A @ T with a sparse accumulation and one outer product."""
    left=len(source_bg);right=len(destination_bg)
    background=a @ source_bg
    result=background[:,None]*destination_bg[None,:]
    for source in range(left):
        for edge in range(offsets[source],offsets[source+1]):
            target=destinations[edge]
            for row in range(a.shape[0]):
                result[row,target]+=a[row,source]*specific[edge]
    return result


@njit(cache=True,parallel=True)
def _propagate_probability(scores,offsets,destinations,log_specific,log_u,log_q):
    samples,left,right=len(scores),len(log_u),len(log_q)
    specific=np.exp(log_specific);u=np.exp(log_u);q=np.exp(log_q)
    result=np.empty((samples,right*right))
    for sample in prange(samples):
        shift=np.max(scores[sample])
        a=np.exp(scores[sample]-shift).reshape((left,left))
        first=_right_apply_probability(a,offsets,destinations,specific,u,q)
        second=_right_apply_probability(first.T,offsets,destinations,specific,u,q).T
        result[sample]=np.log(second).flatten()+shift
    return result


@njit(cache=True,parallel=True)
def _statistics_probability(forward,backward,offsets,destinations,log_specific,log_u,log_q):
    """Scaled positive contractions; no dense matrix-matrix products."""
    samples,left,right=len(forward),len(log_u),len(log_q)
    specific=np.exp(log_specific);bg=np.exp(log_u);q=np.exp(log_q)
    edges=np.empty((samples,len(specific)))
    source=np.empty((samples,left));target=np.empty((samples,right));zs=np.empty(samples)
    for sample in prange(samples):
        ashift=np.max(forward[sample]);bshift=np.max(backward[sample])
        a=np.exp(forward[sample]-ashift).reshape((left,left))
        b=np.exp(backward[sample]-bshift).reshape((right,right))
        m=_right_apply_probability(a,offsets,destinations,specific,bg,q)
        u=_right_apply_probability(a.T,offsets,destinations,specific,bg,q)
        source1=m @ (b.T @ q);source2=u @ (b @ q)
        z=np.dot(bg,source1)
        for first in range(left):
            for edge in range(offsets[first],offsets[first+1]):
                second=destinations[edge];w1=0.;w2=0.
                for partner in range(right):
                    w1+=m[first,partner]*b[second,partner]
                    w2+=u[first,partner]*b[partner,second]
                edges[sample,edge]=specific[edge]*(w1+w2)
                z+=specific[edge]*w1
        edges[sample]/=z
        source[sample]=bg*(source1+source2)/z
        target[sample]=q*(b @ (m.T @ bg)+b.T @ (u.T @ bg))/z
        zs[sample]=math.log(z)+ashift+bshift
    return edges,source,target,zs


def sufficient_statistics(forward, backward, transition, *, per_sample=False):
    # With T_ij >= exp(-100), Z after max-scaling is >= exp(-200).
    # Underflowed A/B terms (< exp(-745)) cannot contribute materially.
    # If the positive background becomes smaller, use the log-domain
    # structured kernel: both routes retain the same near-quadratic bound.
    safe_background=(np.min(transition.log_source_background)
                     +np.min(transition.log_destination)>-100.)
    kernel=_statistics_probability if safe_background else _statistics
    values = kernel(np.ascontiguousarray(forward), np.ascontiguousarray(backward),
        transition.offsets, transition.destinations, transition.log_specific,
        transition.log_source_background, transition.log_destination)
    if np.any(~np.isfinite(values[3])):
        raise FloatingPointError("structured boundary has no finite sample likelihood")
    if per_sample:
        return values
    return (*(value.sum(axis=0) for value in values[:3]), float(values[3].sum()))


def _homologue_marginals(log_pairs):
    k = math.isqrt(log_pairs.shape[1])
    normalizer = np.logaddexp.reduce(log_pairs, axis=1)
    probability = np.exp(log_pairs - normalizer[:, None]).reshape((-1, k, k))
    return 0.5 * (probability.sum(axis=1) + probability.sum(axis=2))


def initialize(left_scores, right_scores, config=StructuredTransitionConfig()):
    """Screen all K_left*K_right links using independent carrier profiles.

    Positive sparse initial mass avoids absorbing zero-parameter edges.
    All destinations stay possible through the shared background. The
    O(N*K**2) screen never computes a dense diploid edge-gradient tensor.
    """
    a, b = _homologue_marginals(left_scores), _homologue_marginals(right_scores)
    left, right = a.shape[1], b.shape[1]
    ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
    scale = np.sqrt(np.sum(ac*ac, axis=0)[:, None] * np.sum(bc*bc, axis=0)[None, :])
    association = np.divide(ac.T @ bc, scale, out=np.zeros((left, right)), where=scale > 0)
    degree = config.degree(right)
    destinations = np.argsort(-association, axis=1, kind="stable")[:, :degree]
    destinations.sort(axis=1)
    q = (b.sum(axis=0) + config.destination_prior_strength / right)
    q /= q.sum()
    return StructuredTransition(np.arange(left+1, dtype=np.int64)*degree,
        np.ascontiguousarray(destinations.ravel(), dtype=np.int64),
        np.full(left*degree, math.log(config.initial_specific_mass / degree)),
        np.full(left, math.log1p(-config.initial_specific_mass)), np.log(q))


def update(transition, statistics, config=StructuredTransitionConfig(), learning_rate=1.0):
    """Regularized latent-mixture EM update, with parameter damping.

    The row prior has fixed total strength (not K-dependent total mass).
    Background destination counts are sufficient for the shared q update.
    This prior and parameterization intentionally differ from dense-edge EM.
    """
    edges, source, target = statistics[:3]
    specific = np.empty_like(edges)
    background = np.empty_like(source)
    row_mass = np.empty_like(source)
    for first in range(transition.shape[0]):
        lo, hi = transition.offsets[first:first+2]
        prior_edge = config.row_prior_strength * config.prior_specific_mass / (hi-lo)
        row = edges[lo:hi] + prior_edge
        bg = source[first] + config.row_prior_strength * (1-config.prior_specific_mass)
        row_mass[first] = row.sum() + bg
        mass = max(config.minimum_background_mass, bg / row_mass[first])
        specific[lo:hi] = (1-mass) * row / row.sum()
        background[first] = mass
    q = target + config.destination_prior_strength / len(target)
    q /= q.sum()
    specific = (1-learning_rate)*np.exp(transition.log_specific) + learning_rate*specific
    background = ((1-learning_rate)*np.exp(transition.log_source_background)
                  + learning_rate*background)
    q = (1-learning_rate)*np.exp(transition.log_destination) + learning_rate*q
    return StructuredTransition(transition.offsets, transition.destinations,
        np.log(specific), np.log(background), np.log(q)), row_mass


def export_mesh(transitions, row_masses, hap_keys, gap):
    forward, backward = {}, {}
    for first, transition in enumerate(transitions):
        second = first + gap
        dense = transition.dense()
        joint = row_masses[first][:, None] * dense
        reverse = (joint / joint.sum(axis=0)[None, :]).T
        forward[first] = {((first, a), (second, b)): float(dense[i, j])
            for i, a in enumerate(hap_keys[first]) for j, b in enumerate(hap_keys[second])}
        backward[second] = {((second, b), (first, a)): float(reverse[j, i])
            for j, b in enumerate(hap_keys[second]) for i, a in enumerate(hap_keys[first])}
    return [forward, backward]


def fit_gap(raw_blocks, prepared_scans, hap_keys, gap, *, max_iterations=20,
            config=StructuredTransitionConfig(), dynamic_cores_fn=None,
            min_change=0.001, diagnostics=None):
    """Fit one residue-chain layer; backward operators are true adjoints."""
    import numba

    blocks = len(raw_blocks)
    independent_forward, independent_backward = [], []
    for block in range(blocks):
        if dynamic_cores_fn is not None:
            numba.set_num_threads(dynamic_cores_fn())
        k = raw_blocks[block].num_haps
        prior = np.full((raw_blocks[block].log_emissions.shape[0], k*k), -2*math.log(k))
        independent_forward.append(prepared_scans[block].scan(prior))
        independent_backward.append(prepared_scans[block].scan(None, backward=True))
    transitions = [initialize(independent_forward[i], independent_backward[i+gap], config)
                   for i in range(blocks-gap)]
    row_masses = [np.ones(item.shape[0]) for item in transitions]
    for iteration in range(max_iterations):
        if dynamic_cores_fn is not None:
            numba.set_num_threads(dynamic_cores_fn())
        forward, backward = [None]*blocks, [None]*blocks
        for block in range(blocks):
            forward[block] = (independent_forward[block] if block < gap else
                prepared_scans[block].scan(propagate(forward[block-gap], transitions[block-gap])))
        for block in range(blocks-1, -1, -1):
            backward[block] = (independent_backward[block] if block >= blocks-gap else
                prepared_scans[block].scan(propagate(backward[block+gap],
                    transitions[block].transpose()), backward=True))
        log_likelihood = sum(float(np.logaddexp.reduce(forward[i], axis=1).sum())
                             for i in range(blocks-gap, blocks))
        change = 0.0
        for first, old in enumerate(transitions):
            if dynamic_cores_fn is not None:
                numba.set_num_threads(dynamic_cores_fn())
            stats = sufficient_statistics(forward[first], backward[first+gap], old)
            new, row_masses[first] = update(old, stats, config, max(0.1, 0.9**iteration))
            # O(K**2) export/comparison is allowed, unlike a cubic contraction.
            change = max(change, float(np.max(np.abs(new.dense()-old.dense()))))
            transitions[first] = new
        if diagnostics is not None:
            diagnostics.append(dict(iteration=iteration+1, log_likelihood=log_likelihood,
                max_change=change, explicit_edges=sum(len(t.destinations) for t in transitions),
                dense_edge_gradient_evaluations=0))
        if change <= min_change:
            break
    return export_mesh(transitions, row_masses, hap_keys, gap)
