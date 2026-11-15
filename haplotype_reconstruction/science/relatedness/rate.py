"""Estimate one genome-wide recombination rate from full-sib families and decide the working fallback rate (rl20).

Unit: a pair (i, j) of one relatedness family f with parent slots u, v (a sequenced sample or latent). Per site the
law of the parents' genotype pair leaves the pair out:

    w(G_u, G_v) ∝ slot_u(G_u) slot_v(G_v) prod_{k in f, k != i, j} C_k(G_u, G_v)

(slot: a sequenced slot's log genotype posterior or the latent Hardy-Weinberg prior; C: the stage's child terms), so
site selection uses no read of the pair, whose meioses are independent of the other members'.

- At least one sequenced slot: the two-indicator chain (I_u, I_v) (whether the pair received the same copy of each
  parent; each flips at 2c per bp) on sites whose most likely parental pair has leave-pair-out posterior >=
  site_posterior with a heterozygous parent, and that the other members alone (both slots latent) support with
  posterior >= segregation_posterior.
- Both slots latent: the slots are exchangeable, so only both-heterozygous sites are label-free; the XOR chain
  Z = I_u xor I_v (flips at 4c) on sites whose (1, 1) posterior is >= site_posterior.
- Emission: the pair's likelihood given each chain state, averaged over the parental genotype law w (all nine pairs);
  the selection only chooses the sites. Conditioning on the most likely pair instead treats its residual posterior as
  zero, and in a simulated sib-ship whose parents were sequenced at depth 10.5 that read the parents' miscalls (which
  parent is heterozygous) as inheritance changes (+17 %).
- Genotype errors the likelihoods do not represent: each site's state emissions are mixed with their state mean,
  e' = (1 - eps) e + eps mean(e); eps is fitted by maximum likelihood on the error_rates grid.
- Artefact stretches (start rate rho per bp, mean length ell; inside a stretch the emission is the state average).
- Composite log-likelihood: pairs weighted 1 / (n_f - 1); (eps, rho, ell) are shared nuisances of the data set, fitted
  with c. Inconsistent families are dropped one at a time. The interval is a two-level percentile bootstrap: families
  with replacement, then each drawn family's chromosomes with replacement (families only with fewer contigs).

Method: docs/methods/relatedness.md#rl20.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

from ...contracts.relatedness import RecombinationRateEstimate
from ...core.concurrency import ThreadSource, thread_count
from ...core.kernels import kernel
from .class_kernels import class_family_table
from .parameters import RateParameters

__all__ = ["RateFit", "estimate_rate", "decide", "rate_grid"]

_DEFAULT_RATE = RateParameters()
_PER_BP = 1e-8  # cM/Mb to Morgans per bp

_b1C = numba.types.Array(numba.boolean, 1, "C", readonly=False)
_b2C = numba.types.Array(numba.boolean, 2, "C", readonly=False)
_f1C = numba.types.Array(numba.float64, 1, "C", readonly=False)
_f2C = numba.types.Array(numba.float64, 2, "C", readonly=False)
_f3C = numba.types.Array(numba.float64, 3, "C", readonly=False)
_f4C = numba.types.Array(numba.float64, 4, "C", readonly=False)
_i1C = numba.types.Array(numba.int64, 1, "C", readonly=False)
_i2C = numba.types.Array(numba.int64, 2, "C", readonly=False)
_f5C = numba.types.Array(numba.float64, 5, "C", readonly=False)


@kernel(
    overloads=(
        (
            _f2C, _f2C, _f2C, _f2C, _f2C, _f2C, _b1C, _b1C, _f1C, _i1C, numba.float64, numba.float64, numba.boolean,
            _f2C, _f1C, _i1C,
        ),
    ),
    cache=True,
    nogil=True,
)
def _site_emissions(base, base_kids, Ci, Cj, gi, gj, oi, oj, pos, unit, tau, tau_kids, xor, em, x, u):
    """Fill em[:n], x[:n] and u[:n] (each selected site's material unit, from ``unit``) for the pair's jointly
    observed selected sites; return (n, summed log row maxima).

    Selection, from the leave-pair-out parental law w: xor, the (1, 1) posterior >= tau; else the most likely pair has
    posterior >= tau and a heterozygous parent and, with tau_kids > 0, the other members alone (both slots latent:
    base_kids) support that unordered pair with posterior >= tau_kids. Emission: the pair's likelihood of each state
    (I_u I_v = 00, 01, 10, 11) averaged over w; xor stores (e00, (e01 + e10) / 2, e11) in em[:, 0:3]. Rows are
    divided by their maxima."""
    n = 0
    logm = 0.0
    M = pos.shape[0]
    w = np.empty(9)
    wk = np.empty(9)
    e = np.empty(4)
    for site in range(M):
        if not (oi[site] and oj[site]):
            continue
        mx = -1e308
        km = 0
        for k in range(9):
            w[k] = base[site, k] - Ci[site, k] - Cj[site, k]
            if w[k] > mx:
                mx = w[k]
                km = k
        tot = 0.0
        for k in range(9):
            w[k] = np.exp(w[k] - mx)
            tot += w[k]
        for k in range(9):
            w[k] /= tot
        if xor:
            if w[4] < tau:
                continue
        else:
            if w[km] < tau:
                continue
            gu0 = km // 3
            gv0 = km % 3
            if gu0 != 1 and gv0 != 1:
                continue
            if tau_kids > 0.0:
                mk = -1e308
                for k in range(9):
                    wk[k] = base_kids[site, k] - Ci[site, k] - Cj[site, k]
                    if wk[k] > mk:
                        mk = wk[k]
                tk = 0.0
                for k in range(9):
                    wk[k] = np.exp(wk[k] - mk)
                    tk += wk[k]
                support = wk[km] if gu0 == gv0 else wk[km] + wk[3 * gv0 + gu0]
                if support / tk < tau_kids:
                    continue
        for s in range(4):
            e[s] = 0.0
        for gu in range(3):
            for gv in range(3):
                weight = w[3 * gu + gv]
                if weight < 1e-300:
                    continue
                for iu in range(2):
                    for iv in range(2):
                        acc = 0.0
                        for su in range(2):
                            au_i = 0 if gu == 0 else (1 if gu == 2 else su)
                            sj = su if iu == 1 else 1 - su
                            au_j = 0 if gu == 0 else (1 if gu == 2 else sj)
                            for sv in range(2):
                                av_i = 0 if gv == 0 else (1 if gv == 2 else sv)
                                tj = sv if iv == 1 else 1 - sv
                                av_j = 0 if gv == 0 else (1 if gv == 2 else tj)
                                acc += 0.25 * gi[site, au_i + av_i] * gj[site, au_j + av_j]
                        e[2 * iu + iv] += weight * acc
        if xor:
            e0, e1, e2 = e[0], 0.5 * (e[1] + e[2]), e[3]
            m = max(e0, max(e1, e2))
            if m <= 0:
                continue
            em[n, 0] = e0 / m
            em[n, 1] = e1 / m
            em[n, 2] = e2 / m
        else:
            m = max(max(e[0], e[1]), max(e[2], e[3]))
            if m <= 0:
                continue
            for s in range(4):
                em[n, s] = e[s] / m
        x[n] = pos[site]
        u[n] = unit[site]
        logm += np.log(m)
        n += 1
    return n, logm


@kernel(
    overloads=(
        (_f2C, _f1C, _i1C, numba.int64, numba.float64, numba.float64, numba.float64, numba.boolean, numba.float64,
         _f1C),
    ),
    cache=True,
    nogil=True,
)
def _chain_loglik(em, x, u, n, c, rho, ell, xor, eps, out):
    """Forward log-likelihood of the inheritance chain at rate c (Morgans per bp) with the artefact-stretch process
    (start rate rho per bp, mean length ell bp) and the isolated-error mixture eps. xor: em [n, 3] read as the XOR
    chain on both-heterozygous sites (P(d | Z = 1) = e1, P(d | Z = 0) = (e0 + e2) / 2); else em [n, 4]. Each site's
    predictive term log p(d_site | d_<site) is added to out[u[site]] (its material unit): the units' values sum to the
    chain's log-likelihood. Returns that sum."""
    if n == 0:
        return 0.0
    K = 2 if xor else 4
    k = rho + 1.0 / ell
    pin = rho / k
    a = np.empty((K, 2))
    e = np.empty(K)
    b = np.empty((K, 2))
    ll = 0.0
    for site in range(n):
        if xor:
            e[0] = 0.5 * (em[site, 0] + em[site, 2])
            e[1] = em[site, 1]
        else:
            for s in range(4):
                e[s] = em[site, s]
        bar = 0.0
        for s in range(K):
            bar += e[s] / K
        if eps > 0.0:
            for s in range(K):
                e[s] = (1.0 - eps) * e[s] + eps * bar
        if site == 0:
            tot = 0.0
            for s in range(K):
                a[s, 0] = (1.0 / K) * (1.0 - pin) * e[s]
                a[s, 1] = (1.0 / K) * pin * bar
                tot += a[s, 0] + a[s, 1]
        else:
            d = x[site] - x[site - 1]
            if xor:
                h = 0.5 * (1.0 - np.exp(-8.0 * c * d))
                for nu in range(2):
                    b[0, nu] = a[0, nu] * (1.0 - h) + a[1, nu] * h
                    b[1, nu] = a[0, nu] * h + a[1, nu] * (1.0 - h)
            else:
                f = 0.5 * (1.0 - np.exp(-4.0 * c * d))
                g = 1.0 - f
                for nu in range(2):
                    t0 = g * a[0, nu] + f * a[2, nu]
                    t1 = g * a[1, nu] + f * a[3, nu]
                    t2 = f * a[0, nu] + g * a[2, nu]
                    t3 = f * a[1, nu] + g * a[3, nu]
                    b[0, nu] = g * t0 + f * t1
                    b[1, nu] = f * t0 + g * t1
                    b[2, nu] = g * t2 + f * t3
                    b[3, nu] = f * t2 + g * t3
            q = np.exp(-k * d)
            p01 = pin * (1.0 - q)
            p11 = pin + (1.0 - pin) * q
            tot = 0.0
            for s in range(K):
                n0 = b[s, 0] * (1.0 - p01) + b[s, 1] * (1.0 - p11)
                n1 = b[s, 0] * p01 + b[s, 1] * p11
                a[s, 0] = n0 * e[s]
                a[s, 1] = n1 * bar
                tot += a[s, 0] + a[s, 1]
        step = np.log(tot)
        ll += step
        out[u[site]] += step
        for s in range(K):
            a[s, 0] /= tot
            a[s, 1] /= tot
    return ll


@kernel(
    overloads=(
        (
            numba.boolean, _f2C, _f2C, _f3C, _f3C, _b2C, _f1C, _i1C, _i2C, _f1C, _f1C, _f1C, _f1C, numba.float64,
            numba.float64, _f5C, _f1C,
        ),
    ),
    parallel=True,
    cache=True,
    nogil=True,
)
def pair_tables(xor, base, base_kids, C, g, obs, pos, unit, pairs, rates, rhos, ells, epss, tau, tau_kids, ll, nsites):
    """ll[t, e, a, r, k]: pair t's chain log-likelihood at rates[r] (Morgans per bp), nuisance (rhos[a], ells[a]) and
    error mixture epss[e], split by the material unit k (``unit``: each site's unit, 0-based on this contig) of each
    site's predictive term; nsites[t]: its selected sites. The emission rows' maxima are a constant of each pair and
    drop out of every comparison. Pairs are independent; every value is computed by one thread in site order."""
    P = pairs.shape[0]
    M = pos.shape[0]
    R = rates.shape[0]
    A = rhos.shape[0]
    E = epss.shape[0]
    for t in numba.prange(P):
        i = pairs[t, 0]
        j = pairs[t, 1]
        x = np.empty(M)
        em = np.empty((M, 4))
        u = np.empty(M, np.int64)
        n, _lm = _site_emissions(base, base_kids, C[i], C[j], g[i], g[j], obs[i], obs[j], pos, unit, tau, tau_kids,
                                 xor, em, x, u)
        nsites[t] = n
        for z in range(E):
            for q in range(A):
                for r in range(R):
                    _chain_loglik(em, x, u, n, rates[r], rhos[q], ells[q], xor, epss[z], ll[t, z, q, r])


def rate_grid(parameters: RateParameters = _DEFAULT_RATE) -> np.ndarray:
    """The log-spaced rate grid (cM/Mb)."""
    return np.geomspace(parameters.rate_low, parameters.rate_high, parameters.rate_points)


def _grid_argmax(rates: np.ndarray, ll: np.ndarray) -> float:
    """Maximiser of a log-likelihood tabulated on a log-spaced grid: the parabola through the best point and its two
    neighbours on log(rate); a grid edge (or a non-concave triple) returns the grid point."""
    k = int(np.argmax(ll))
    if k == 0 or k == len(rates) - 1:
        return float(rates[k])
    xs = np.log(rates[k - 1 : k + 2])
    ys = ll[k - 1 : k + 2]
    denom = (xs[0] - xs[1]) * (xs[0] - xs[2]) * (xs[1] - xs[2])
    a = (xs[2] * (ys[1] - ys[0]) + xs[1] * (ys[0] - ys[2]) + xs[0] * (ys[2] - ys[1])) / denom
    b = (xs[2] ** 2 * (ys[0] - ys[1]) + xs[1] ** 2 * (ys[2] - ys[0]) + xs[0] ** 2 * (ys[1] - ys[2])) / denom
    if a >= 0:
        return float(rates[k])
    return float(np.exp(-b / (2 * a)))


def _fit(table: np.ndarray, rates: np.ndarray) -> tuple[float, int, int, bool]:
    """(rate, error index, nuisance index, at a grid edge) of a composite table [E, A, R]."""
    e, a, r = np.unravel_index(int(np.argmax(table)), table.shape)
    return _grid_argmax(rates, table[e, a]), int(e), int(a), r in (0, len(rates) - 1)


@dataclass(frozen=True)
class _Family:
    family: int
    members: int
    model: str
    pairs: int
    table: np.ndarray | None  # [C, E, A, R], pair-weighted; None when not tabulated
    sites: float  # median selected sites per pair over the genome
    units: np.ndarray | None = None  # [U, E, A, R]: the same split by equal-material unit of the support axis


@dataclass(frozen=True)
class RateFit:
    """The estimate and its gate, before the working-map decision."""

    estimate: float
    interval: tuple[float, float]
    interval_kind: str
    meioses: int
    determined_sites: float
    error_rate: float
    artefact: tuple[float, float]
    at_grid_edge: bool
    per_chromosome: np.ndarray
    families: tuple[_Family, ...]
    family_estimate: dict[int, float]
    status: dict[int, str]
    inconsistency: dict[int, float]
    gate_passed: bool
    gate_reason: str

    @property
    def used_families(self) -> int:
        return sum(1 for value in self.status.values() if value == "used")


# Pairs per kernel call (execution grain only: bounds the [pairs, E, A, R, units] workspace).
_PAIR_CHUNK = 256


def _family_tables(terms, family: np.ndarray, parents: np.ndarray, parameters, threads,
                   contig_mask: np.ndarray | None = None) -> list[_Family]:
    """Each family's pair-weighted composite tables, per contig [C, E, A, R] and per equal-material unit of the
    support axis [U, E, A, R] (FamilyTerms.ranges: 1-Morgan units of the working map, DESIGN_417 D5), and its median
    selected sites per pair. Contigs outside ``contig_mask`` are not tabulated (zero rows). Contigs are the outer loop
    (one copy of a contig's terms), families the inner; every value is independent of the order."""
    rates = rate_grid(parameters) * _PER_BP
    rhos = np.asarray([start / 1e6 for start, _ in parameters.artefacts], dtype=np.float64)
    ells = np.asarray([length * 1e3 for _, length in parameters.artefacts], dtype=np.float64)
    epss = np.asarray(parameters.error_rates, dtype=np.float64)
    bounds = np.concatenate(([0], np.cumsum(np.asarray(terms.chromosome_sites, dtype=np.int64))))
    contigs = len(bounds) - 1
    members: dict[int, list[int]] = {}
    for sample, f in enumerate(np.asarray(family)):
        if f >= 0:
            members.setdefault(int(f), []).append(sample)
    plans = []  # (family, rows, slots, model, pairs) of tabulated families
    result: dict[int, _Family] = {}
    for f in sorted(members):
        m = members[f]
        n = len(m)
        if n < 2:
            continue
        u, v = (int(s) for s in parents[f])
        sequenced = (u >= 0) + (v >= 0)
        model = "two_indicator" if sequenced else "xor"
        pairs = np.asarray([(m[a], m[b]) for a in range(n) for b in range(a + 1, n)], dtype=np.int64)
        if n - 2 + 2 * sequenced < parameters.minimum_information:
            result[f] = _Family(f, n, model, len(pairs), None, float("nan"))
            continue
        plans.append((f, np.asarray(m, dtype=np.int64), (u, v), model, pairs))
    classed = getattr(terms, "class_prior", None) is not None
    weights = {f: _class_weights(terms, rows, slots) for f, rows, slots, _, _ in plans} if classed else {}
    shape = (contigs, len(epss), len(rhos), len(rates))
    unit_of = np.zeros(int(bounds[-1]), np.int64)
    for k, (first, stop) in enumerate(terms.ranges):
        unit_of[first:stop] = k
    n_units = max(1, len(terms.ranges))
    tables = {f: np.zeros(shape) for f, *_ in plans}
    unit_tables = {f: np.zeros((n_units,) + shape[1:]) for f, *_ in plans}
    sites = {f: np.zeros(len(pairs)) for f, *_, pairs in plans}
    for c in range(contigs):
        lo, hi = int(bounds[c]), int(bounds[c + 1])
        if hi == lo or not plans or (contig_mask is not None and not contig_mask[c]):
            continue
        first_unit = int(unit_of[lo:hi].min())
        local_unit = np.ascontiguousarray(unit_of[lo:hi] - first_unit)
        span = int(local_unit.max()) + 1
        M = hi - lo
        C = np.array(terms.child[:, lo:hi], dtype=np.float64)
        prior = np.array(terms.log_prior[lo:hi], dtype=np.float64)
        g = np.array(terms.scaled[:, lo:hi], dtype=np.float64)
        obs = np.array(terms.observed[:, lo:hi], dtype=np.bool_)
        pos = np.array(terms.positions[lo:hi], dtype=np.float64)
        latent = (prior[:, :, None] + prior[:, None, :]).reshape(M, 9)
        laws = np.exp(np.asarray(terms.class_prior[:, lo:hi], dtype=np.float64)) if classed else None
        for f, rows, (u, v), model, pairs in plans:
            thread_count(threads)  # each family's pairs on the task's current share
            member_sum = C[rows].sum(0)
            if classed:
                # rl21: a latent slot's law is the classes' mean law under the family's class posterior
                slots, both = weights[f]
                lu = np.exp(np.asarray(terms.log_posterior[u, lo:hi], dtype=np.float64))[None] if u >= 0 else laws
                lv = np.exp(np.asarray(terms.log_posterior[v, lo:hi], dtype=np.float64))[None] if v >= 0 else laws
                base = np.ascontiguousarray(_mixed_law(slots, lu, lv) + member_sum)
                kids = base if model == "xor" else np.ascontiguousarray(_mixed_law(both, laws, laws) + member_sum)
            else:
                su = np.array(terms.log_posterior[u, lo:hi], dtype=np.float64) if u >= 0 else prior
                sv = np.array(terms.log_posterior[v, lo:hi], dtype=np.float64) if v >= 0 else prior
                base = np.ascontiguousarray((su[:, :, None] + sv[:, None, :]).reshape(M, 9) + member_sum)
                kids = base if model == "xor" else np.ascontiguousarray(latent + member_sum)
            nsites = np.zeros(len(pairs))
            acc = np.zeros(shape[1:] + (span,))
            for start in range(0, len(pairs), _PAIR_CHUNK):
                chunk = np.ascontiguousarray(pairs[start:start + _PAIR_CHUNK])
                ll = np.zeros((len(chunk),) + shape[1:] + (span,))
                counted = np.zeros(len(chunk))
                pair_tables(
                    model == "xor", base, kids, C, g, obs, pos, local_unit, chunk, rates, rhos, ells, epss,
                    float(parameters.site_posterior),
                    0.0 if model == "xor" else float(parameters.segregation_posterior),
                    ll, counted,
                )
                acc += ll.sum(0)
                nsites[start:start + len(chunk)] = counted
            acc /= len(rows) - 1
            unit_tables[f][first_unit:first_unit + span] += np.moveaxis(acc, -1, 0)
            tables[f][c] = acc.sum(-1)
            sites[f] += nsites
    for f, rows, _, model, pairs in plans:
        result[f] = _Family(f, len(rows), model, len(pairs), tables[f], float(np.median(sites[f])), unit_tables[f])
    return [result[f] for f in sorted(result)]


def _class_weights(terms, rows, slots) -> tuple[np.ndarray, np.ndarray]:
    """rl21: a family's class posteriors from all its members over every family site, on the family fit's own terms
    (child terms over parental configurations): (the weights of its slots' laws [n_u, n_v], n = 1 for a sequenced
    slot; the weights [K, K] with both slots latent). Classes are constant over the genome, so the posterior is that
    of the whole genome; it does not use which sites a pair shares."""
    u, v = slots
    member_sum = np.ascontiguousarray(np.asarray(terms.ancestry_child[rows], dtype=np.float64).sum(0))
    laws = np.ascontiguousarray(np.exp(np.asarray(terms.ancestry_prior, dtype=np.float64)))
    classes, M, G = laws.shape
    posterior = np.zeros((2, M, G))
    for k, slot in enumerate((u, v)):
        if slot >= 0:
            posterior[k] = terms.ancestry_posterior[slot]

    held = None
    if terms.ancestry_composition is not None:
        held, class_type, sample_type = terms.ancestry_composition
        held = held[0][rows].sum(axis=0)  # the members' ancestry along the map, by the parents' transmitted shares

    def weights(a: int, b: int, su: int, sv: int) -> np.ndarray:
        table = np.empty((classes if a < 0 else 1, classes if b < 0 else 1))
        class_family_table(member_sum, member_sum, 0.0, a, b, laws, posterior, 0, M, table)
        if held is not None:
            tu = class_type if a < 0 else sample_type[[su]]
            tv = class_type if b < 0 else sample_type[[sv]]
            table += held[tu[:, None], tv[None, :]]
        table = np.exp(table - table.max())  # the classes' prior weights are equal
        return table / table.sum()

    both = weights(-1, -1, u, v)
    return (both if u < 0 and v < 0 else weights(0 if u >= 0 else -1, 1 if v >= 0 else -1, u, v)), both


def _mixed_law(weights: np.ndarray, laws_u: np.ndarray, laws_v: np.ndarray) -> np.ndarray:
    """float64 [M, 9]: log sum_{a, b} weights[a, b] laws_u[a, i, g_u] laws_v[b, i, g_v], the slots' joint law."""
    joint = np.einsum("ab,amg,bmh->mgh", weights, laws_u, laws_v)
    return np.log(np.maximum(joint, 1e-300)).reshape(joint.shape[0], 9)


def _exclude_inconsistent(families: list[_Family], threshold: float):
    """Drop, one at a time, the family whose composite log-likelihood ratio between its own maximum and the
    leave-one-family-out pooled maximiser (shared nuisances) is largest, while it exceeds ``threshold`` nats and at
    least three families remain."""
    keep = {item.family: item for item in families}
    dropped = {}
    while len(keep) >= 3:
        keys = sorted(keep)
        X = np.stack([keep[k].table.sum(0) for k in keys])
        total = X.sum(0)
        worst, worst_lr = None, threshold
        for i, k in enumerate(keys):
            loo = total - X[i]
            index = np.unravel_index(int(np.argmax(loo)), loo.shape)
            lr = float(X[i].max() - X[i][index])
            if lr > worst_lr:
                worst, worst_lr = k, lr
        if worst is None:
            break
        dropped[worst] = worst_lr
        del keep[worst]
    return [keep[k] for k in sorted(keep)], dropped


def _bootstrap(X: np.ndarray, rates: np.ndarray, parameters: RateParameters):
    """Two-level percentile bootstrap of the estimate from family tables X [F, U, E, A, R] split by equal-material
    unit (nuisances refitted): with at least minimum_interval_units units, families with replacement and then each
    drawn family's units with replacement ("family_unit"); with less material but that many families, families only
    ("family"). No count of chromosomes enters (owner, 13 Nov 2026: material, not chromosomes)."""
    F = X.shape[0]
    units = X.shape[1]
    rng = np.random.default_rng(parameters.bootstrap_seed)
    if units >= parameters.minimum_interval_units:
        kind, by_family = "family_unit", X
    elif F >= parameters.minimum_interval_units:
        kind, by_family = "family", X.sum(1, keepdims=True)
    else:
        return (float("nan"), float("nan")), "none"
    Cn = by_family.shape[1]
    draws = []
    for _ in range(parameters.bootstrap_replicates):
        acc = np.zeros(X.shape[2:])
        for f in rng.integers(0, F, F):
            acc += by_family[f][rng.integers(0, Cn, Cn)].sum(0) if Cn > 1 else by_family[f, 0]
        draws.append(_fit(acc, rates)[0])
    draws = np.asarray(draws)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))), kind


def estimate_rate(
    terms, family: np.ndarray, parents: np.ndarray, parameters: RateParameters = _DEFAULT_RATE, *,
    threads: ThreadSource = 1, contig_mask: np.ndarray | None = None,
) -> RateFit:
    """Fit the genome-wide rate on ``terms`` (a FamilyTerms with positions, scaled likelihoods and observations)
    from the relatedness families (``family`` per sample, ``parents`` per family) and apply the gate.
    ``contig_mask``: the contigs the fit uses (the map-free contigs of a genome that also has map files: the estimate
    stands in for the fallback rate only); None: every contig. Per-chromosome values of the others are NaN."""
    rates = rate_grid(parameters)
    contigs = len(terms.chromosome_sites)
    families = _family_tables(terms, family, parents, parameters, threads, contig_mask)
    status = {item.family: "too_little_information" for item in families if item.table is None}
    informative = [item for item in families if item.table is not None and item.sites > 0]
    status.update(
        {item.family: "no_informative_sites" for item in families if item.table is not None and not item.sites > 0}
    )
    kept, dropped = _exclude_inconsistent(informative, parameters.inconsistency_nats)
    status.update({f: "inconsistent" for f in dropped})
    status.update({item.family: "used" for item in kept})
    family_estimate = {item.family: _fit(item.table.sum(0), rates)[0] for item in informative}
    nan2 = (float("nan"), float("nan"))
    if not kept:
        return RateFit(
            float("nan"), nan2, "none", 0, float("nan"), float("nan"), nan2, False, np.full(contigs, np.nan),
            tuple(families), family_estimate, status, dropped, False, "no informative family",
        )
    X = np.stack([item.table for item in kept])  # [F, C, E, A, R]
    estimate, e, a, edge = _fit(X.sum((0, 1)), rates)
    start, length = parameters.artefacts[a]
    used = np.ones(X.shape[1], bool) if contig_mask is None else np.asarray(contig_mask, bool)
    per_chromosome = np.asarray([_grid_argmax(rates, X[:, c, e, a].sum(0)) if used[c] else float("nan")
                                 for c in range(X.shape[1])])
    Xu = np.stack([item.units for item in kept])  # [F, U, E, A, R]
    occupied = np.flatnonzero(np.any(Xu != 0, axis=(0, 2, 3, 4)))
    interval, kind = _bootstrap(np.ascontiguousarray(Xu[:, occupied]), rates, parameters)
    weights = np.asarray([2 * item.members for item in kept], dtype=np.float64)
    sites = float(np.sum(weights * np.asarray([item.sites for item in kept])) / np.sum(weights))
    meioses = int(weights.sum())
    if edge or estimate <= parameters.edge_low or estimate >= parameters.edge_high:
        passed, reason = False, "estimate at grid edge"
    elif sites < parameters.minimum_determined_sites:
        passed, reason = False, (
            f"weighted determined sites per pair {sites:.0f} < {parameters.minimum_determined_sites:.0f}"
        )
    elif kind == "none":
        passed, reason = False, f"{len(occupied)} material units and {len(kept)} families: no interval"
    else:
        half = (interval[1] - interval[0]) / 2 / estimate
        label = kind.replace("_", " x ")
        if half > parameters.maximum_half_width:
            passed, reason = False, f"{label} 95 % half-width {half:.0%} > {parameters.maximum_half_width:.0%}"
        else:
            passed, reason = True, f"{label} interval [{interval[0]:.2f}, {interval[1]:.2f}]"
    return RateFit(
        estimate, interval, kind, meioses, sites, float(parameters.error_rates[e]), (float(start), float(length)),
        bool(edge), per_chromosome, tuple(families), family_estimate, status, dropped, passed, reason,
    )


def _rate_text(value: float) -> str:
    return f"{value:.3g}"


def decide(
    fit: RateFit,
    contigs: tuple[str, ...],
    *,
    supplied_rate: float,
    map_free_contigs: tuple[str, ...],
    rate_estimate: str,
    parameters: RateParameters = _DEFAULT_RATE,
) -> RecombinationRateEstimate:
    """The working-map decision. Unless ``rate_estimate`` is adopt the supplied rate is kept (veto and ignore: the
    early estimate is a diagnostic). With adopt, the estimate replaces the supplied fallback rate (contigs without a
    map file and the stretches beyond every file's knots) when the gate passes, the supplied rate lies outside the
    estimate's 95 % interval and at least minimum_adoption_families informative families were used: with fewer,
    between-family variation in the rate cannot be measured, whatever the interval. A map file's own distances never
    change. The warning (empty when none) is reported once in the run log and in the report."""
    infer_rate = rate_estimate == "adopt"
    low, high = fit.interval
    outside = fit.gate_passed and not (low <= supplied_rate <= high)
    enough = fit.used_families >= parameters.minimum_adoption_families
    kind = fit.interval_kind.replace("_", " x ")
    summary = (
        f"genome-wide recombination rate estimated from relatives {_rate_text(fit.estimate)} cM/Mb, 95 % {kind} "
        f"interval [{low:.2f}, {high:.2f}] ({fit.meioses} meioses, {fit.used_families} families)"
    )
    supplied = f"the supplied fallback rate {_rate_text(supplied_rate)} cM/Mb"
    few = (
        f"the estimate rests on {fit.used_families} informative "
        f"famil{'y' if fit.used_families == 1 else 'ies'}; --rate-estimate adopt adopts it only with at least "
        f"{parameters.minimum_adoption_families}"
    )
    decision, working, warning = "kept", float(supplied_rate), ""
    if not fit.gate_passed:
        reason = f"gate failed: {fit.gate_reason}"
    elif not map_free_contigs:
        reason = "every contig has a map file"
    elif not outside:
        reason = "the supplied rate lies inside the interval"
    elif infer_rate and enough:
        digits = parameters.working_digits
        decision, working = "adopted", float(f"{fit.estimate:.{digits}g}")
        reason = "rate_estimate = adopt: the supplied rate lies outside the interval"
    elif infer_rate:
        reason = f"rate_estimate = adopt: fewer than {parameters.minimum_adoption_families} informative families"
        warning = f"{summary}; {supplied} lies outside it, but {few}, so the supplied rate is kept."
    else:
        reason = f"the supplied rate lies outside the interval (rate_estimate = {rate_estimate})"
        advice = (
            "set [genetic_map] rate_cm_per_mb, or run with --rate-estimate adopt ([genetic_map] rate_estimate = "
            "\"adopt\") to use the estimate" if enough else f"set [genetic_map] rate_cm_per_mb ({few})"
        )
        warning = f"{summary}; {supplied} lies outside it. Contigs without a map file keep the supplied rate: {advice}."
    rows = fit.families
    return RecombinationRateEstimate(
        estimate_cm_per_mb=float(fit.estimate),
        interval_low_cm_per_mb=float(low),
        interval_high_cm_per_mb=float(high),
        interval_kind=fit.interval_kind,
        meioses=int(fit.meioses),
        determined_sites=float(fit.determined_sites),
        error_rate=float(fit.error_rate),
        artefact_rate_per_mb=float(fit.artefact[0]),
        artefact_length_kb=float(fit.artefact[1]),
        at_grid_edge=bool(fit.at_grid_edge),
        contigs=tuple(contigs),
        per_chromosome_cm_per_mb=np.asarray(fit.per_chromosome, dtype=np.float64),
        family=np.asarray([item.family for item in rows], dtype=np.int64),
        family_members=np.asarray([item.members for item in rows], dtype=np.int64),
        family_model=tuple(item.model for item in rows),
        family_pairs=np.asarray([item.pairs for item in rows], dtype=np.int64),
        family_sites=np.asarray([item.sites for item in rows], dtype=np.float64),
        family_estimate_cm_per_mb=np.asarray(
            [fit.family_estimate.get(item.family, float("nan")) for item in rows], dtype=np.float64
        ),
        family_status=tuple(fit.status[item.family] for item in rows),
        family_inconsistency=np.asarray(
            [fit.inconsistency.get(item.family, float("nan")) for item in rows], dtype=np.float64
        ),
        gate_passed=bool(fit.gate_passed),
        gate_reason=fit.gate_reason,
        supplied_rate_cm_per_mb=float(supplied_rate),
        rate_estimate=str(rate_estimate),
        map_free_contigs=tuple(map_free_contigs),
        decision=decision,
        decision_reason=reason,
        working_rate_cm_per_mb=working,
        warning=warning,
    )
