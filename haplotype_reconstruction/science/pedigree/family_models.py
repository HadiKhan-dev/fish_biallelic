"""Search the family transmission models across independent optimizer starts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable

import numpy as np

from ...core.random import stream
from .family_fit import _prepare, fit, init_latent


def _closest(paths, latent_pair, cpairs):
    """Index (0/1) of the latent diploid whose unordered homologue pair matches C's schedule in more bins."""
    agree = []
    for l in range(2):
        a, b = paths[l, 0], paths[l, 1]
        agree.append(int(np.sum(((a == cpairs[0]) & (b == cpairs[1])) | ((a == cpairs[1]) & (b == cpairs[0])))))
    return int(np.argmax(agree))


def _complement(p, cpairs):
    """PAR start from a REL/UNR solution: X = C's pair, Y = per bin the homologue of each latent diploid that C's
    schedule does not use (the first homologue when neither or both match)."""
    q = np.empty_like(p)
    q[0] = cpairs
    for l in range(2):
        a, b = p[l, 0], p[l, 1]
        uses_a = (a == cpairs[0]) | (a == cpairs[1])
        q[1, l] = np.where(uses_a, b, a)
    return q


# A caller-owned map factory: mapper(width) -> map(function, items), the results in input order, on up to ``width``
# threads (the fits release the interpreter lock). The stage supplies it (science owns no concurrency); every
# function here reduces the results in input order, so they depend neither on the factory nor on ``width``.
Mapper = Callable[[int], Callable]


def _serial(function, items):
    return [function(item) for item in items]


def _maps(mapper: Mapper | None, width: int):
    return mapper(width) if mapper is not None and width > 1 else _serial


def family_models(tc, mcls, mknown, cpairs, theta, floor, mdelta, max_iter=30, rounds=3, rho=0.01, n_random=3,
                  seed=None, width=1, mapper=None):
    """Profile log likelihoods and per-bin increments of PAR, REL and UNR for members [C, S] + witnesses.

    cpairs int64 [2, B]: C's scheduled trajectory pair. S (member 1) is always a child; each witness (members 2..)
    is a child of the latent pair, or (log prior rho each) a direct observation of a latent diploid (a sequenced
    parent of the family; under PAR only latent 1, the other parent) or an outsider explained by its own mosaic
    (a relative that is not a sib). Every model is refitted from starts derived from every other model's best
    latent paths (direct, homologues crossed between the diploids, maternal pair swapped, and for PAR the
    complement of C's homologues) until no model improves (at most ``rounds``). Returns dict(name -> (loglik,
    inc[B], paths, iterations, roles)).

    ``width``, ``mapper``: threads for the search's independent fits (each member alone, the member-seeded starts,
    and each round's starts of one model, which are fixed before that model's loop), and the caller's map factory
    for them (None: in sequence). They are fitted at once and compared in their sequential order, so the result
    does not depend on ``width``. The adaptive steps stay sequential: adopt prefetches the moves it can know in
    advance onto spare threads (only with more than one thread), perturbed draws one start at a time."""
    map_fits = _maps(mapper, width)
    B = tc.shape[1]
    n = len(mcls)
    loge, representatives, transition = _prepare(tc, mcls, mknown, theta, floor, mdelta)
    k = n - 1
    yes = np.array([True, True])
    one = np.array([True, False])
    p0 = np.zeros((2, 2, B), dtype=np.int64)
    # each member alone (one latent diploid observed directly; exact by a single Viterbi step)
    alone_inc = np.zeros((n, B))
    alone_paths = np.zeros((n, 2, B), dtype=np.int64)
    none_inc = np.zeros((n, B))
    nflex = np.zeros(n, dtype=np.bool_)
    lpr0 = np.zeros(7)
    allow0 = np.zeros(7, dtype=np.bool_)
    def alone(m):
        q = p0.copy()
        r = np.full(n, 3, dtype=np.int64)
        r[m] = 0
        return fit(tc, mcls, mknown, r, q, one, theta, floor, mdelta, max_iter, none_inc, nflex, allow0, lpr0, loge,
                   representatives, transition)

    for m, res in enumerate(map_fits(alone, range(n))):
        alone_inc[m] = res[1]
        alone_paths[m] = res[2][0]
    calone = (float(alone_inc[0].sum()), alone_inc[0].copy(), None, 1, None)
    # v8b: half child of latent 0 or 1 (roles 5, 6) are flexible roles too; child prior 1 - 5 rho
    lc = math.log(1.0 - 5.0 * rho)
    lr = math.log(rho)
    lpr = np.array([lr, lr, lc, 0.0, lr, lr, lr])
    flex = np.zeros(n, dtype=np.bool_)
    flex[2:] = True
    allowed = dict(PAR=np.array([False, True, True, False, True, True, True]),
                   UNR=np.array([True, True, True, False, True, True, True]),
                   REL=np.array([True, True, True, False, True, True, True]))
    roles = dict(PAR=np.array([0] + [2] * k, dtype=np.int64), UNR=np.array([3] + [2] * k, dtype=np.int64),
                 REL=np.array([2] + [2] * k, dtype=np.int64))
    p0[0] = cpairs
    p0[1] = cpairs
    pp = p0.copy()
    pp[1] = init_latent(tc, mcls, mknown, roles["PAR"], pp, 1, theta, floor, mdelta, loge, representatives,
                        transition)

    fitted = {}

    def run(name, start, start_roles):
        # All other fit inputs are fixed for this family search. Keep exact bytes, including the roles, so only
        # identical deterministic fits are reused. Cached arrays are read only throughout the search.
        key = (name, start.tobytes(), start_roles.tobytes())
        if key not in fitted:
            fitted[key] = fit(tc, mcls, mknown, start_roles, start, yes, theta, floor, mdelta, max_iter, alone_inc,
                              flex, allowed[name], lpr, loge, representatives, transition)
        return fitted[key]

    def run_all(requests):
        # Fit every (name, start, roles) not yet fitted at once (distinct keys, first occurrence), then serve all in
        # order: the same fits as calling run() on each in turn.
        keys = [(name, start.tobytes(), start_roles.tobytes()) for name, start, start_roles in requests]
        missing = {}
        for key, request in zip(keys, requests, strict=True):
            if key not in fitted and key not in missing:
                missing[key] = request
        for key, value in zip(missing, map_fits(lambda r: run(*r), list(missing.values())), strict=True):
            fitted[key] = value
        return [fitted[key] for key in keys]

    best = dict(PAR=run("PAR", pp, roles["PAR"]))

    def crossed(p):
        q = p.copy()
        q[0, 1], q[1, 0] = p[1, 0].copy(), p[0, 1].copy()
        return q

    def swapped(p):
        q = p.copy()
        q[1, 0], q[1, 1] = p[1, 1].copy(), p[1, 0].copy()
        return q

    def starts_for(name):
        out = []
        for src, res in best.items():
            p = res[2]
            if name == "PAR":
                j = _closest(p, None, cpairs)
                s2 = np.empty_like(p)
                s2[0] = p[j]
                s2[1] = p[1 - j]
                out += [s2, _complement(p, cpairs)]
            else:
                out += [p, crossed(p), crossed(swapped(p))]
        return out

    def adopt(name):
        # joint moves: one witness becomes a direct observation of a latent diploid whose paths are reset to that
        # witness's own best mosaic (a sequenced parent inside the family, e.g. an unreleased co-parent)
        res0 = best[name]
        changed = False
        targets = (1,) if name == "PAR" else (0, 1)
        if map_fits is not _serial:
            # Prefetch on spare threads: the direct-observation moves depend only on res0, and the half-child moves
            # on best[name], which is res0 unless a move below improves it. Fitting them now only fills run()'s
            # cache; the sequential loop below is unchanged, so the result is the same (an improvement makes some
            # prefetched half-child fits unused).
            requests = []
            for m in range(2, n):
                for j in targets:
                    q = res0[2].copy()
                    q[j] = alone_paths[m]
                    r = res0[4].copy()
                    r[m] = j
                    requests.append((name, q, r))
                for j in (0, 1):
                    r = res0[4].copy()
                    if r[m] != 5 + j:
                        r[m] = 5 + j
                        requests.append((name, res0[2].copy(), r))
            run_all(requests)
        for m in range(2, n):
            for j in targets:
                q = res0[2].copy()
                q[j] = alone_paths[m]
                r = res0[4].copy()
                r[m] = j
                res = run(name, q, r)
                if res[0] > best[name][0] + 1e-9:
                    best[name] = res
                    changed = True
            # v8b: the witness becomes a half child of either latent diploid (paths kept; EM fills the free slot)
            for j in (0, 1):
                r = best[name][4].copy()
                if r[m] == 5 + j:
                    continue
                r[m] = 5 + j
                res = run(name, best[name][2].copy(), r)
                if res[0] > best[name][0] + 1e-9:
                    best[name] = res
                    changed = True
        return changed

    def seeded():
        # v7: member-seeded starts (S3): latent diploids initialised at members' own best mosaics, so the
        # alternatives are not reached only through PAR's solution
        k_seed = min(n, 4)
        requests = []
        for name in ("UNR", "REL"):
            for i in range(k_seed):
                for j in range(i + 1, k_seed):
                    q = np.empty_like(p0)
                    q[0] = alone_paths[i]
                    q[1] = alone_paths[j]
                    requests.append((name, q, roles[name]))
        for m in range(1, k_seed):
            q = p0.copy()
            q[1] = alone_paths[m]
            requests.append(("PAR", q, roles["PAR"]))
        for (name, _, _), res in zip(requests, run_all(requests), strict=True):
            if res[0] > best.get(name, (-np.inf,))[0] + 1e-9:
                best[name] = res

    def perturbed(rng):
        # v7: deterministic random block perturbations of each model's best paths (S3), seeded from the data
        changed = False
        T = tc.shape[0]
        for name in ("UNR", "REL", "PAR"):
            for _ in range(n_random):
                q = best[name][2].copy()
                lo = int(rng.integers(0, B))
                hi = min(B, lo + max(1, int(B * rng.uniform(0.1, 0.4))))
                lat = 1 if name == "PAR" else int(rng.integers(0, 2))
                h = int(rng.integers(0, 2))
                q[lat, h, lo:hi] = int(rng.integers(0, T))
                res = run(name, q, best[name][4])
                if res[0] > best[name][0] + 1e-9:
                    best[name] = res
                    changed = True
        return changed

    seeded()
    digest = hashlib.sha256(np.ascontiguousarray(mcls).tobytes() + np.ascontiguousarray(mknown).tobytes()).digest()
    rng = stream("pedigree.family_transmission.starts", seed=seed, key=int.from_bytes(digest[:8], "little"))
    tried = set()
    for _ in range(rounds):
        changed = False
        for name in ("UNR", "REL", "PAR"):
            # this model's starts are fixed before its loop: fit the untried ones at once, compare them in order
            starts = []
            for s0 in starts_for(name):
                key = (name, s0.tobytes())
                if key in tried:
                    continue
                tried.add(key)
                starts.append(s0)
            for res in run_all([(name, s0, roles[name]) for s0 in starts]):
                if name not in best or res[0] > best[name][0] + 1e-9:
                    best[name] = res
                    changed = True
        if n > 2:
            for name in ("UNR", "REL", "PAR"):
                changed = adopt(name) or changed
        changed = perturbed(rng) or changed
        if not changed:
            break
    return dict(C=calone, PAR=best["PAR"], UNR=best["UNR"], REL=best["REL"])
