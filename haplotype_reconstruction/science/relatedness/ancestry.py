"""Source populations, local ancestry and the ancestry-aware terms of the family model (rl21).

With two source populations A and B (site frequencies a, b of the ALT allele) every position of an individual's
genome is in one of three ancestry states: both homologues from A, one from each, both from B. A first-generation
hybrid, always in the middle state, is heterozygous at a fixed difference with probability 1; no Hardy-Weinberg law
at one frequency says so. In a cross these states run along the genome in long blocks, so they are read per sample
from its own reads and conditioned on, not scored as thousands of independent sites.

Fit, from the genotype likelihoods alone (no sample name, order or label):

1. shares and frequencies: an EM over each sample's genome-wide state shares z and the sites' source frequencies on
   a subsample of family sites, from seeded starts (best likelihood kept), then every family site's frequencies at
   the fitted shares;
2. local ancestry: per sample and contig, a three-state chain along the working map (stay with probability
   exp(-rate x Morgans), else redraw from z), emissions the state laws at the site's source frequencies; each
   site's state probabilities are taken from the sample's other sites (leave-site-out).

Family terms. A parent's configuration at a site is its state with the alleles on its homologues (CONFIGURATIONS:
A/A and B/B by genotype, A/B by the allele on each homologue). A child's term for a pair of parental
configurations is its reads' probability given which homologues its own local state says it received (an A/A child
received an A homologue from each parent; an A/B child one of each); a transmission its state does not allow gets
the floor ``source_state_error``. Conditioning on the child's state means its ancestry composition is no evidence
about its parents site by site (only infeasibility is), and a fixed difference is exactly neutral for an F2
sibship. What a child's ancestry along the map does say about its parents is counted once, at the scale of its
ancestry blocks: the likelihood of its reads when its two homologues come from parents of given transmitted shares
(two-state source chains along the map) against its own chain (composition). A latent
parent has one of a few ancestry classes, each the state shares of a kind of individual whose two homologues have
types (a share m of source B constant over the genome; A: 0, B: 1, H: 1/2, a homologue transmitted by a
first-generation hybrid): z = ((1 - m1)(1 - m2), m1 (1 - m2) + m2 (1 - m1), m1 m2); the family kernels marginalise
a latent slot over the classes (class_kernels). A sequenced slot's law is its own configuration posterior.

One source population runs none of this.
Method: docs/methods/relatedness.md#rl21.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numba
import numpy as np

from ...core.kernels import kernel
from .parameters import FamilyParameters

__all__ = [
    "CONFIGURATIONS",
    "SourceModel",
    "read_depth",
    "source_warnings",
    "class_types",
    "class_shares",
    "state_laws",
    "configuration_laws",
    "local_states",
    "state_logliks",
    "transmission_logliks",
    "composition",
    "child_terms",
    "fit_sources",
    "apply_sources",
]

# A parent's configurations at a site: (ancestry state 0 = A/A, 1 = A/B, 2 = B/B; genotype; the probability that a
# transmitted A homologue carries ALT; the same for a B homologue; None where the state has no such homologue).
# A/A and B/B are listed by genotype (their homologues are exchangeable), A/B by (allele on A, allele on B).
CONFIGURATIONS = (
    (0, 0, 0.0, None), (0, 1, 0.5, None), (0, 2, 1.0, None),
    (1, 0, 0.0, 0.0), (1, 1, 0.0, 1.0), (1, 1, 1.0, 0.0), (1, 2, 1.0, 1.0),
    (2, 0, None, 0.0), (2, 1, None, 0.5), (2, 2, None, 1.0),
)
_STATE = np.asarray([c[0] for c in CONFIGURATIONS], dtype=np.int64)
_GENOTYPE = np.asarray([c[1] for c in CONFIGURATIONS], dtype=np.int64)
_GIVES_A = np.asarray([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])  # probability of transmitting an A homologue

_TINY = 1e-300


@dataclass(frozen=True)
class SourceModel:
    """The fitted two-source model on the family sites.

    shares [T]: the homologue types' shares of source B. classes [K, 2]: each class's two type indices.
    frequencies: one float64 [2, M_c] per contig (ALT frequency in source A, B). sample_shares [N, 3]: each
    sample's shares of the ancestry states (A/A, A/B, B/B). loglik: the kept start's log likelihood on the fit
    sites; start_logliks: every start's; iterations: EM iterations of the kept start; fit_sites: sites of the fit.
    depth: the cohort's estimated reads per family site and sample (the median over samples of -ln(1 - observed
    share), the Poisson estimate from the share of sites with a read)."""

    shares: np.ndarray
    classes: np.ndarray
    frequencies: tuple[np.ndarray, ...]
    sample_shares: np.ndarray
    loglik: float
    start_logliks: tuple[float, ...]
    iterations: int
    fit_sites: int
    depth: float = float("nan")


def read_depth(terms) -> float:
    """The cohort's estimated reads per family site and sample: the median over samples of -ln(1 - observed
    share of family sites) (a Poisson depth from the share of sites with at least one read; capped where every
    site is observed)."""
    observed = np.zeros(terms[0].observed.shape[0])
    sites = 0
    for t in terms:
        observed += np.asarray(t.observed, dtype=np.float64).sum(axis=1)
        sites += t.observed.shape[1]
    if sites == 0:
        return float("nan")
    share = np.minimum(observed / sites, 1.0 - 0.5 / sites)
    return float(np.median(-np.log1p(-share)))


def class_types(shares) -> np.ndarray:
    """int64 [K, 2]: the unordered pairs of homologue types, in order (0, 0), (0, 1), ..., (T - 1, T - 1)."""
    n = len(shares)
    return np.asarray([(s, t) for s in range(n) for t in range(s, n)], dtype=np.int64)


def class_shares(shares, classes: np.ndarray) -> np.ndarray:
    """float64 [K, 3]: every class's shares of the ancestry states (A/A, A/B, B/B)."""
    m = np.asarray(shares, dtype=np.float64)
    m1, m2 = m[classes[:, 0]], m[classes[:, 1]]
    return np.stack(((1 - m1) * (1 - m2), m1 * (1 - m2) + m2 * (1 - m1), m1 * m2), axis=1)


def state_laws(frequencies: np.ndarray) -> np.ndarray:
    """float64 [3, M, 3]: the genotype law of each ancestry state (A/A, A/B, B/B) at source frequencies [2, M]."""
    a, b = frequencies[0], frequencies[1]
    out = np.empty((3, len(a), 3))
    for s, (f1, f2) in enumerate(((a, a), (a, b), (b, b))):
        out[s, :, 0] = (1 - f1) * (1 - f2)
        out[s, :, 1] = f1 * (1 - f2) + f2 * (1 - f1)
        out[s, :, 2] = f1 * f2
    return out


def _expectations(g, weight, z, a, b):
    """One E step on scaled likelihoods g [N, S, 3] (unobserved cells 1, weight 0): (log likelihood of the observed
    cells, state responsibilities summed over sites [N, 3], expected ALT and REF copies attributed to each source
    [2, S] each)."""
    alt = np.zeros((2, g.shape[1]))
    ref = np.zeros((2, g.shape[1]))
    mass = np.zeros_like(z)
    total = np.zeros(g.shape[:2])
    parts = []
    for s, (f1, f2) in enumerate(((a, a), (a, b), (b, b))):
        j00 = ((1 - f1) * (1 - f2))[None] * g[..., 0]
        j10 = (f1 * (1 - f2))[None] * g[..., 1]
        j01 = ((1 - f1) * f2)[None] * g[..., 1]
        j11 = (f1 * f2)[None] * g[..., 2]
        like = j00 + j10 + j01 + j11
        joint = z[:, s, None] * like
        total += joint
        parts.append((joint, (j10 + j11) / np.maximum(like, _TINY), (j01 + j11) / np.maximum(like, _TINY)))
    total = np.maximum(total, _TINY)
    loglik = float(np.sum(weight * np.log(total)))
    for s, (joint, h1, h2) in enumerate(parts):
        r = weight * joint / total
        mass[:, s] = r.sum(axis=1)
        for source, dose in ((0 if s < 2 else 1, h1), (0 if s < 1 else 1, h2)):
            alt[source] += (r * dose).sum(axis=0)
            ref[source] += (r * (1 - dose)).sum(axis=0)
    return loglik, mass, alt, ref


def _frequencies(alt, ref, clip):
    """Source frequencies [2, S] from attributed copies; a source no observed homologue draws from at a site keeps
    the pooled frequency."""
    mass = alt + ref
    pooled = alt.sum(axis=0) / np.maximum(mass.sum(axis=0), _TINY)
    return np.clip(np.where(mass > 1e-9, alt / np.maximum(mass, _TINY), pooled[None]), clip, 1 - clip)


def _fit(g, observed, *, starts, iterations, tolerance, seed, clip):
    """(z [N, 3], frequencies [2, S], log likelihood, iterations, every start's log likelihood): the EM over the
    samples' state shares and the sites' source frequencies, best of ``starts`` seeded starts, each until its log
    likelihood gains less than ``tolerance`` (relative) or for ``iterations``."""
    weight = observed.astype(np.float64)
    calls = np.maximum(weight.sum(axis=1), 1.0)
    rng = np.random.default_rng(seed)
    best, logliks = None, []
    for _ in range(int(starts)):
        z = rng.dirichlet(np.ones(3), size=g.shape[0])
        f = rng.uniform(0.05, 0.95, size=(2, g.shape[1]))
        previous, used = -np.inf, 0
        for used in range(1, int(iterations) + 1):
            loglik, mass, alt, ref = _expectations(g, weight, z, f[0], f[1])
            z = mass / calls[:, None]
            f = _frequencies(alt, ref, clip)
            if loglik - previous <= tolerance * abs(loglik):
                break
            previous = loglik
        loglik = _expectations(g, weight, z, f[0], f[1])[0]
        logliks.append(loglik)
        if best is None or loglik > best[2]:
            best = (z, f, loglik, used)
    return (*best, tuple(logliks))


def _site_frequencies(g, observed, z, start, iterations, clip, chunk):
    """float64 [2, M]: EM of every site's source frequencies at fixed sample shares z, from ``start`` [2, M]."""
    out = np.empty_like(start)
    for lo in range(0, g.shape[1], chunk):
        gc = np.asarray(g[:, lo : lo + chunk], dtype=np.float64)
        weight = np.asarray(observed[:, lo : lo + chunk], dtype=np.float64)
        f = start[:, lo : lo + chunk].copy()
        for _ in range(int(iterations)):
            _, _, alt, ref = _expectations(gc, weight, z, f[0], f[1])
            f = _frequencies(alt, ref, clip)
        out[:, lo : lo + chunk] = f
    return out


def configuration_laws(frequencies: np.ndarray) -> np.ndarray:
    """float64 [3, M, G]: each ancestry state's law over the configurations at source frequencies [2, M] (zero
    outside the state's own configurations)."""
    a, b = frequencies[0], frequencies[1]
    out = np.zeros((3, len(a), len(CONFIGURATIONS)))
    out[0, :, 0], out[0, :, 1], out[0, :, 2] = (1 - a) ** 2, 2 * a * (1 - a), a * a
    out[1, :, 3], out[1, :, 4], out[1, :, 5], out[1, :, 6] = (1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b
    out[2, :, 7], out[2, :, 8], out[2, :, 9] = (1 - b) ** 2, 2 * b * (1 - b), b * b
    return out


_f1 = numba.types.Array(numba.float64, 1, "C", readonly=False)
_f2 = numba.types.Array(numba.float64, 2, "C", readonly=False)
_f3 = numba.types.Array(numba.float64, 3, "C", readonly=False)


@kernel(overloads=((_f3, _f2, _f1, numba.float64, _f3, _f3),), parallel=True, cache=True, nogil=True)
def local_states(emission, shares, coordinates, rate, others, smoothed):
    """Per sample n, the three-state ancestry chain along one contig: emission [N, M, 3] (each state's probability of
    the sample's reads at the site), start and redraw law shares[n], coordinates [M] in Morgans (stay with
    probability exp(-rate x distance), else redraw from the shares). others[n, i]: the state probabilities at site
    i given the sample's reads at every other site of the contig; smoothed[n, i]: given all of them."""
    N, M = emission.shape[0], emission.shape[1]
    for n in numba.prange(N):
        z = shares[n]
        predicted = np.empty((M, 3))
        backward = np.empty((M, 3))
        filtered = np.empty(3)
        for i in range(M):
            if i == 0:
                for s in range(3):
                    predicted[i, s] = z[s]
            else:
                stay = np.exp(-rate * (coordinates[i] - coordinates[i - 1]))
                for s in range(3):
                    predicted[i, s] = stay * filtered[s] + (1.0 - stay) * z[s]
            total = 0.0
            for s in range(3):
                filtered[s] = predicted[i, s] * emission[n, i, s]
                total += filtered[s]
            if total <= 0.0:
                for s in range(3):
                    filtered[s] = predicted[i, s]
            else:
                for s in range(3):
                    filtered[s] /= total
        for s in range(3):
            backward[M - 1, s] = 1.0
        for i in range(M - 2, -1, -1):
            stay = np.exp(-rate * (coordinates[i + 1] - coordinates[i]))
            mixed = 0.0
            for s in range(3):
                mixed += z[s] * emission[n, i + 1, s] * backward[i + 1, s]
            top = 0.0
            for s in range(3):
                backward[i, s] = stay * emission[n, i + 1, s] * backward[i + 1, s] + (1.0 - stay) * mixed
                if backward[i, s] > top:
                    top = backward[i, s]
            if top > 0.0:
                for s in range(3):
                    backward[i, s] /= top
            else:
                for s in range(3):
                    backward[i, s] = 1.0
        for i in range(M):
            total = 0.0
            for s in range(3):
                others[n, i, s] = predicted[i, s] * backward[i, s]
                total += others[n, i, s]
            both = 0.0
            for s in range(3):
                others[n, i, s] = others[n, i, s] / total if total > 0.0 else z[s]
                smoothed[n, i, s] = others[n, i, s] * emission[n, i, s]
                both += smoothed[n, i, s]
            for s in range(3):
                smoothed[n, i, s] = smoothed[n, i, s] / both if both > 0.0 else others[n, i, s]


_i2 = numba.types.Array(numba.int64, 2, "C", readonly=False)
_f4 = numba.types.Array(numba.float64, 4, "C", readonly=False)


@kernel(overloads=((_f3, _f2, _f1, numba.float64, _i2, _f2),), parallel=True, cache=True, nogil=True)
def state_logliks(emission, shares, gaps, rate, ranges, out):
    """out[r, n] = the log-likelihood of sample n's reads over sites ranges[r] = [lo, hi) under its own ancestry
    chain (local_states' chain: start and redraw law shares[n]; gaps[i] the Morgans from site i - 1, infinite at a
    contig's first site). A range starts from the redraw law."""
    N = emission.shape[0]
    for n in numba.prange(N):
        z = shares[n]
        p = np.zeros(3)  # read at a range's first site as 0 * p: uninitialised memory there gave NaN (4.22 FLAKY-NAN)
        for r in range(ranges.shape[0]):
            total = 0.0
            for i in range(ranges[r, 0], ranges[r, 1]):
                stay = 0.0 if i == ranges[r, 0] else np.exp(-rate * gaps[i])
                norm = 0.0
                for s in range(3):
                    p[s] = (stay * p[s] + (1.0 - stay) * z[s]) * emission[n, i, s]
                    norm += p[s]
                if norm <= 0.0:
                    total += -690.0
                    for s in range(3):
                        p[s] = z[s]
                else:
                    total += np.log(norm)
                    for s in range(3):
                        p[s] /= norm
            out[r, n] = total


@kernel(overloads=((_f3, _f1, _f1, numba.float64, _i2, _f4),), parallel=True, cache=True, nogil=True)
def transmission_logliks(emission, types, gaps, rate, ranges, out):
    """out[r, n, a, b] = the log-likelihood of sample n's reads over sites ranges[r] when it received one homologue
    from a parent whose transmitted homologues have share types[a] of source B and one from a parent of share
    types[b]. Each received homologue's source is a two-state chain along the map (redrawn at ``rate`` per Morgan
    from (1 - share, share); a first-generation hybrid's gamete switches source at each crossover, a pure parent's
    never), the two chains independent; the sample's ancestry state is their pair. gaps as in state_logliks."""
    N, T = emission.shape[0], types.shape[0]
    for n in numba.prange(N):
        p = np.empty((2, 2))
        q = np.empty((2, 2))
        for r in range(ranges.shape[0]):
            for a in range(T):
                for b in range(a, T):
                    ha, hb = types[a], types[b]
                    total = 0.0
                    for i in range(ranges[r, 0], ranges[r, 1]):
                        stay = 0.0 if i == ranges[r, 0] else np.exp(-rate * gaps[i])
                        if stay > 0.0:
                            all_ = p[0, 0] + p[0, 1] + p[1, 0] + p[1, 1]
                            for x in range(2):
                                wx = ha if x == 1 else 1.0 - ha
                                for y in range(2):
                                    wy = hb if y == 1 else 1.0 - hb
                                    q[x, y] = (
                                        stay * stay * p[x, y]
                                        + stay * (1.0 - stay) * (wx * (p[0, y] + p[1, y]) + wy * (p[x, 0] + p[x, 1]))
                                        + (1.0 - stay) * (1.0 - stay) * wx * wy * all_
                                    )
                        else:
                            for x in range(2):
                                for y in range(2):
                                    q[x, y] = (ha if x == 1 else 1.0 - ha) * (hb if y == 1 else 1.0 - hb)
                        norm = 0.0
                        for x in range(2):
                            for y in range(2):
                                p[x, y] = q[x, y] * emission[n, i, x + y]
                                norm += p[x, y]
                        if norm <= 0.0:
                            total += -690.0
                            for x in range(2):
                                for y in range(2):
                                    p[x, y] = q[x, y]
                        else:
                            total += np.log(norm)
                            for x in range(2):
                                for y in range(2):
                                    p[x, y] /= norm
                    out[r, n, a, b] = total
                    out[r, n, b, a] = total


def composition(emission, sample_shares, gaps, ranges, shares, classes, rate):
    """(float64 [R, N, T, T], class types int64 [K], sample types int64 [N], transmitted shares [T]).

    The first array is, for each site range and sample, the log-likelihood ratio of the sample's reads between
    having received homologues of transmitted shares (types[a], types[b]) and its own ancestry chain: what the
    sample's ancestry along the map says about its parents' classes, counted at the scale of its ancestry blocks
    (not per site). A class transmits homologues of share (m1 + m2) / 2; a sequenced sample transmits its own
    hybrid index, rounded to the nearest listed share."""
    shares = np.asarray(shares, dtype=np.float64)
    transmitted = (shares[classes[:, 0]] + shares[classes[:, 1]]) / 2.0
    types = np.unique(transmitted)
    class_type = np.searchsorted(types, transmitted).astype(np.int64)
    index = sample_shares[:, 1] / 2.0 + sample_shares[:, 2]
    sample_type = np.abs(index[:, None] - types[None, :]).argmin(axis=1).astype(np.int64)
    ranges = np.ascontiguousarray(ranges, dtype=np.int64).reshape(-1, 2)
    out = np.empty((len(ranges), emission.shape[0], len(types), len(types)))
    own = np.empty((len(ranges), emission.shape[0]))
    transmission_logliks(emission, np.ascontiguousarray(types), gaps, rate / 2.0, ranges, out)
    state_logliks(emission, np.ascontiguousarray(sample_shares), gaps, rate, ranges, own)
    return out - own[:, :, None, None], class_type, sample_type, types


def _two_allele(g: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """The reads' probability [N, M] when the two received alleles are ALT with probabilities t1, t2."""
    return (
        (1 - t1) * (1 - t2) * g[..., 0] + (t1 * (1 - t2) + t2 * (1 - t1)) * g[..., 1] + t1 * t2 * g[..., 2]
    )


def child_terms(g: np.ndarray, states: np.ndarray, error: float, floor: float) -> np.ndarray:
    """float64 [N, M, G x G]: each sample's log term for every pair of parental configurations (h_u, h_v): the
    probability of its reads (scaled likelihoods g [N, M, 3], unobserved cells 1) given the configurations and its
    own local ancestry state, averaged over ``states`` [N, M, 3] (its state probabilities from its other sites).
    An A/A child received an A homologue from each parent, a B/B child a B homologue from each, an A/B child an A
    homologue from one and a B homologue from the other (both ways weighted by the parents' chances of giving
    them); a state the configurations cannot produce contributes ``error``."""
    G = len(CONFIGURATIONS)
    out = np.empty(g.shape[:2] + (G * G,))
    for hu, (_, _, ua, ub) in enumerate(CONFIGURATIONS):
        for hv, (_, _, va, vb) in enumerate(CONFIGURATIONS):
            term = np.zeros(g.shape[:2])
            # child A/A
            term += states[..., 0] * (_two_allele(g, ua, va) if ua is not None and va is not None else error)
            # child B/B
            term += states[..., 2] * (_two_allele(g, ub, vb) if ub is not None and vb is not None else error)
            # child A/B: A from u and B from v, or B from u and A from v
            w1 = _GIVES_A[hu] * (1 - _GIVES_A[hv])
            w2 = (1 - _GIVES_A[hu]) * _GIVES_A[hv]
            if w1 + w2 > 0:
                mixed = 0.0
                if w1 > 0:
                    mixed = mixed + w1 * _two_allele(g, ua, vb)
                if w2 > 0:
                    mixed = mixed + w2 * _two_allele(g, ub, va)
                term += states[..., 1] * mixed / (w1 + w2)
            else:
                term += states[..., 1] * error
            out[..., G * hu + hv] = np.log(np.maximum(term, floor))
    return out


def _subsample(terms, limit):
    """Per contig, the indices of every step-th family site in genome order (at most about ``limit`` in all)."""
    total = sum(len(t.coordinates) for t in terms)
    step = max(1, -(-total // max(1, int(limit))))
    picks, offset = [], 0
    for t in terms:
        m = len(t.coordinates)
        picks.append(np.arange((-offset) % step, m, step))
        offset += m
    return picks


def fit_sources(terms, parameters: FamilyParameters) -> SourceModel:
    """Fit the two-source model on every contig's family sites (ContigTerms with ``scaled`` and ``observed``)."""
    terms = tuple(terms)
    shares = np.asarray(parameters.source_shares, dtype=np.float64)
    clip = float(parameters.source_clip)
    picks = _subsample(terms, parameters.source_fit_sites)
    g = np.concatenate(
        [np.asarray(t.scaled[:, i], dtype=np.float64) for t, i in zip(terms, picks, strict=True)], axis=1
    )
    observed = np.concatenate(
        [np.asarray(t.observed[:, i], dtype=np.bool_) for t, i in zip(terms, picks, strict=True)], axis=1
    )
    z, fitted, loglik, used, logliks = _fit(
        g, observed, starts=parameters.source_starts, iterations=parameters.source_iterations,
        tolerance=parameters.source_tolerance, seed=parameters.source_seed, clip=clip,
    )
    del g, observed
    frequencies, offset = [], 0
    for t, pick in zip(terms, picks, strict=True):
        # every site starts from its nearest fit site's frequencies
        m = len(t.coordinates)
        if len(pick):
            nearest = np.clip(np.searchsorted(pick, np.arange(m), side="right") - 1, 0, len(pick) - 1)
            start = np.ascontiguousarray(fitted[:, offset + nearest])
        else:
            start = np.full((2, m), 0.5)
        offset += len(pick)
        frequencies.append(
            _site_frequencies(
                t.scaled, t.observed, z, start, parameters.source_site_iterations, clip, int(parameters.source_chunk)
            )
        )
    return SourceModel(
        shares, class_types(shares), tuple(frequencies), np.ascontiguousarray(z), float(loglik), logliks, int(used),
        int(sum(len(i) for i in picks)), read_depth(terms),
    )


def apply_sources(terms, model: SourceModel, parameters: FamilyParameters):
    """The contigs' terms for the family model with two source populations: ``ancestry_child`` [N, M, G x G] (child
    terms over parental configuration pairs, conditional on the child's local state), ``ancestry_prior`` [K, M, G]
    (log; each class's law over the configurations), ``ancestry_posterior`` [N, M, G] (log; each sample's
    configuration posterior, its law as a sequenced slot), ``ancestry_emission`` [N, M, 3] (each ancestry state's
    probability of the sample's reads), ``class_prior`` [K, M, 3] (log; each class's genotype law) and
    ``log_posterior`` [N, M, 3] rebuilt as the genotype marginal of the configuration posterior."""
    mix = class_shares(model.shares, model.classes)
    out = []
    for t, f in zip(terms, model.frequencies, strict=True):
        g = np.asarray(t.scaled, dtype=np.float64)
        n, m = g.shape[:2]
        if m == 0:
            out.append(
                replace(
                    t, class_prior=np.empty((len(mix), 0, 3)), ancestry_prior=np.empty((len(mix), 0, len(_STATE))),
                    ancestry_child=np.empty((n, 0, len(_STATE) ** 2)), ancestry_posterior=np.empty((n, 0, len(_STATE))),
                    ancestry_emission=np.empty((n, 0, 3)),
                )
            )
            continue
        states = state_laws(f)  # [3, M, 3]
        emission = np.ascontiguousarray(np.einsum("nmg,smg->nms", g, states))
        others, smoothed = np.empty((n, m, 3)), np.empty((n, m, 3))
        local_states(
            emission, np.ascontiguousarray(model.sample_shares), np.ascontiguousarray(t.coordinates, dtype=np.float64),
            float(parameters.source_switch_rate), others, smoothed,
        )
        configurations = configuration_laws(f)  # [3, M, G]
        # a sample's configuration posterior: its state from its other sites, the alleles given the state, its reads
        joint = others[:, :, _STATE] * configurations[_STATE, :, np.arange(len(_STATE))].T[None] * g[:, :, _GENOTYPE]
        joint /= np.maximum(joint.sum(axis=2, keepdims=True), _TINY)
        genotype = np.zeros((n, m, 3))
        for h, value in enumerate(_GENOTYPE):
            genotype[..., value] += joint[..., h]
        out.append(
            replace(
                t,
                log_posterior=np.ascontiguousarray(np.log(np.maximum(genotype, parameters.term_floor))),
                class_prior=np.ascontiguousarray(
                    np.log(np.maximum(np.einsum("ks,smg->kmg", mix, states), parameters.term_floor))
                ),
                ancestry_prior=np.ascontiguousarray(
                    np.log(np.maximum(np.einsum("ks,smg->kmg", mix, configurations), parameters.term_floor))
                ),
                ancestry_child=np.ascontiguousarray(
                    child_terms(g, others, float(parameters.source_state_error), parameters.term_floor)
                ),
                ancestry_posterior=np.ascontiguousarray(np.log(np.maximum(joint, parameters.term_floor))),
                ancestry_emission=emission,
            )
        )
    return tuple(out)


def source_warnings(strong_fraction: float, depth: float, parameters: FamilyParameters) -> tuple[str, ...]:
    """The warnings of a two-source fit (reported in the run log and relatedness/summary.json; no result depends on
    them): the cohort does not look like a cross between two diverged sources, or its sources are not identified
    (too few strongly differentiated sites); the depth is below what the setting was validated at."""
    out = []
    if not strong_fraction >= parameters.source_minimum_strong:
        out.append(
            f"[input] source_populations = 2, but only {100 * strong_fraction:.2f} % of the family sites differ by "
            f"more than {parameters.source_strong_differentiation:g} in allele frequency between the fitted sources "
            f"(expected at least {100 * parameters.source_minimum_strong:g} %). The cohort does not look like a "
            "cross between two diverged populations, or its sources are not identified at this depth. Declaring two "
            "sources for a one-population cohort splits sibships: use source_populations = 1 unless the cross is "
            "known to be between two diverged sources."
        )
    if depth < parameters.source_validated_depth:
        out.append(
            f"[input] source_populations = 2 at an estimated {depth:.1f} reads per family site and sample. The "
            f"setting is validated at about 5 reads; below {parameters.source_validated_depth:g} it is not shown "
            "to be better than one source: in simulations at 1 read, sibships whose parents are sibs merged around "
            "an aunt or uncle in a parent slot. Read families, parent slots and member supports with caution."
        )
    return tuple(out)
