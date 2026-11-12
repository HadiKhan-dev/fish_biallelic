"""Structured SMC' branch measure and replayed mutations, in relative Morgans.

Serial traversals take a caller-owned numpy Generator. Save its bit_generator.state
before separating_length, and restore that state before mutations (use a separate
stream for site clocks and mutation uniforms). Neither pass seeds a global RNG.
The interval record grows by GENEALOGY_RECORD_BLOCK up to GENEALOGY_INTERVAL_LIMIT.
No stock, source-panel, settings, I/O or engine dependency is required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from ..contracts.roles import frozen_array, record_field, schema_field
from ..core.errors import InputError
from ..core.kernels import kernel
from .parameters import GENEALOGY_INTERVAL_LIMIT, GENEALOGY_RECORD_BLOCK, GENEALOGY_WORK_LIMIT

__all__ = ['separating_length', 'mutations', 'tree_summaries', 'TreeSummary', 'validate_population_tree']

_I = nb.types.Array(nb.int64, 1, 'C')
_F = nb.types.Array(nb.float64, 1, 'C')
_IR = nb.types.Array(nb.int64, 1, 'C', readonly=True)
_FR = nb.types.Array(nb.float64, 1, 'C', readonly=True)
_BR = nb.types.Array(nb.boolean, 2, 'C', readonly=True)
_G = nb.types.NumPyRandomGeneratorType('NumPyRandomGeneratorType')
_TREE = (_I, _I, _I, _F, _I)


@dataclass(frozen=True, eq=False)
class TreeSummary:
    """Unascertained tree diagnostics; spectrum[i] measures branches below i leaves."""

    height: float = record_field('diagnostic', units='generations', dtype='float64')
    total_length: float = record_field('diagnostic', units='generations', dtype='float64')
    pair_time: float = record_field('diagnostic', units='generations', dtype='float64')
    spectrum: np.ndarray = record_field('diagnostic', axes=('leaves_below',), units='generations', dtype='float64')
    schema: int = schema_field(1)

    def __post_init__(self):
        object.__setattr__(self, 'spectrum', frozen_array(self.spectrum, np.float64))


def validate_population_tree(leaf_pop, pop_parent, pop_split, pop_size):
    """Validate all structure before the first random draw; return contiguous frozen inputs."""
    for values, name in ((leaf_pop, 'sample populations'), (pop_parent, 'population parents')):
        raw = np.asarray(values)
        if raw.ndim != 1 or raw.dtype.kind not in 'iu':
            raise InputError(f'{name} must be an integer vector')
    lp = frozen_array(leaf_pop, np.int64)
    pp = frozen_array(pop_parent, np.int64)
    ps = frozen_array(pop_split, np.float64)
    pz = frozen_array(pop_size, np.float64)
    P = len(pp)
    if not P or ps.shape != (P,) or pz.shape != (P,):
        raise InputError('population parent, split and size must be matching nonempty vectors')
    if len(lp) < 2 or np.any(lp < 0) or np.any(lp >= P):
        raise InputError('at least two sample populations, all known, are required')
    if np.any(pp < -1) or np.any(pp >= P) or np.count_nonzero(pp == -1) != 1:
        raise InputError('population tree requires exactly one root and known parents')
    if np.any(~np.isfinite(pz)) or np.any(pz <= 0):
        raise InputError('population sizes must be positive and finite (diploid individuals)')
    nonroot = pp >= 0
    if np.any(~np.isfinite(ps[nonroot])) or np.any(ps[nonroot] < 0):
        raise InputError('non-root splits must be finite and nonnegative (generations)')
    # The root has no split. Its placeholder is deliberately ignored.
    state = np.zeros(P, np.int8)
    for start in range(P):
        path = []
        p = start
        while p >= 0 and state[p] == 0:
            state[p] = 1
            path.append(p)
            p = pp[p]
        if p >= 0 and state[p] == 1:
            raise InputError('population tree contains a cycle')
        for p in path:
            state[p] = 2
    for p in np.flatnonzero(nonroot):
        q = pp[p]
        if pp[q] >= 0 and ps[p] > ps[q]:
            raise InputError('child split must not exceed parent split (generations)')
    return lp, pp, ps, pz


@kernel(overloads=((nb.int64, nb.float64, _IR, _FR),), cache=True)
def _ancestor(p, t, pp, ps):
    while pp[p] >= 0 and t >= ps[p]:
        p = pp[p]
    return p


@kernel(overloads=((_F, nb.float64),), cache=True)
def _positive_choice(masses, u):
    """Residual upper-tail mass selects the last positive entry, including u == sum."""
    last = -1
    for i in range(len(masses)):
        if masses[i] > 0:
            last = i
            if u < masses[i]:
                return i
            u -= masses[i]
    if last < 0:
        raise ValueError('cumulative choice has no positive mass')
    return last


@kernel(overloads=((_IR, _IR, _FR, _FR, _G, *_TREE),), cache=True)
def _first_tree(lp, pp, ps, pz, rng, parent, left, right, time, npop):
    n = len(lp)
    P = len(pp)
    live = np.arange(n)
    cur = lp.copy()
    for i in range(n):
        cur[i] = _ancestor(cur[i], 0.0, pp, ps)
    k, nxt, t = n, n, 0.0
    while k > 1:
        cnt = np.zeros(P, np.int64)
        for i in range(k):
            cnt[cur[i]] += 1
        rate = np.empty(P)
        total = 0.0
        split = np.inf
        for p in range(P):
            rate[p] = (cnt[p] * (cnt[p] - 1) / 4.0) / pz[p]
            total += rate[p]
            if pp[p] >= 0 and t < ps[p] < split:
                split = ps[p]
        dt = rng.exponential(1.0 / total) if total > 0 else np.inf
        if t + dt >= split:
            if not np.isfinite(split):
                raise ValueError('nonfinite coalescence time')
            t = split
            for i in range(k):
                cur[i] = _ancestor(cur[i], t, pp, ps)
            continue
        t += dt
        p = _positive_choice(rate, rng.random() * total)
        idx = np.empty(cnt[p], np.int64)
        m = 0
        for i in range(k):
            if cur[i] == p:
                idx[m] = i
                m += 1
        a = rng.integers(0, m)
        b = rng.integers(0, m - 1)
        if b >= a:
            b += 1
        ia, ib = idx[a], idx[b]
        na, nc = live[ia], live[ib]
        parent[na] = nxt
        parent[nc] = nxt
        left[nxt], right[nxt], time[nxt], npop[nxt] = na, nc, t, p
        lo, hi = min(ia, ib), max(ia, ib)
        live[lo], cur[lo] = nxt, p
        live[hi], cur[hi] = live[k - 1], cur[k - 1]
        k -= 1
        nxt += 1
    return live[0]


@kernel(overloads=((nb.int64, _I, _I),), cache=True)
def _postorder(root, left, right):
    """Topology order remains valid at zero-length edges and equal node times."""
    M = len(left)
    stack = np.empty(M, np.int64)
    order = np.empty(M, np.int64)
    stack[0], top, k = root, 0, 0
    while top >= 0:
        v = stack[top]
        top -= 1
        order[k] = v
        k += 1
        if left[v] >= 0:
            top += 1
            stack[top] = left[v]
            top += 1
            stack[top] = right[v]
    return order[::-1].copy()


@kernel(overloads=((nb.int64, _I, _I, _I, _I),), cache=True)
def _postorder_into(root, left, right, stack, order):
    """Write the same topology order into traversal-owned scratch arrays."""
    stack[0], top, k = root, 0, len(order) - 1
    while top >= 0:
        v = stack[top]
        top -= 1
        order[k] = v
        k -= 1
        if left[v] >= 0:
            top += 1
            stack[top] = left[v]
            top += 1
            stack[top] = right[v]


@kernel(overloads=((nb.int64, _I, _F, _I, _I),), cache=True)
def _time_order_into(n, postorder, time, order, scratch):
    """Stable internal-node time order; topology order breaks equal-time ties."""
    # Only one internal node changes time per cut. Keep the preceding order and
    # insert moved nodes, using topology ranks to preserve exact equal-time ties.
    for rank in range(len(postorder)):
        scratch[postorder[rank]] = rank
    for i in range(1, len(order)):
        node = order[i]
        t, rank = time[node], scratch[node]
        j = i
        while j > 0:
            previous = order[j - 1]
            if time[previous] < t or (time[previous] == t and scratch[previous] < rank):
                break
            order[j] = previous
            j -= 1
        order[j] = node


@kernel(overloads=((nb.int64, nb.int64, _I, _I, _I, _F, _BR, nb.int64,
                    _I, _I, _F, nb.boolean),), cache=True)
def _masses_into(n, root, parent, left, right, time, carried, j, order, below, masses, recount):
    """Refresh counts only for a changed topology or carried set; reuse storage."""
    if recount:
        for c in range(n):
            below[c] = int(carried[j, c])
        for v in order:
            if v >= n:
                below[v] = below[left[v]] + below[right[v]]
    total = below[root]
    for c in range(len(parent)):
        masses[c] = 0.0
        if parent[c] >= 0 and 0 < below[c] < total:
            masses[c] = time[parent[c]] - time[c]


@kernel(overloads=((nb.int64, nb.int64, _I, _I, _I, _F),), cache=True)
def _summary(n, root, parent, left, right, time):
    below = np.zeros(len(parent), np.int64)
    below[:n] = 1
    for v in _postorder(root, left, right):
        if v >= n:
            below[v] = below[left[v]] + below[right[v]]
    out = np.zeros(n + 4)
    out[0] = time[root]
    for c in range(len(parent)):
        if parent[c] >= 0:
            d = time[parent[c]] - time[c]
            out[1] += d
            out[2] += d * below[c] * (n - below[c]) / (n * (n - 1))
            out[3 + below[c]] += d
    return out


@kernel(overloads=((nb.int64, nb.int64, _I, _I, _I, _F, _BR, nb.int64),), cache=True)
def _separating_masses(n, root, parent, left, right, time, carried, j):
    below = np.zeros(len(parent), np.int64)
    total = 0
    for c in range(n):
        below[c] = int(carried[j, c])
        total += below[c]
    for v in _postorder(root, left, right):
        if v >= n:
            below[v] = below[left[v]] + below[right[v]]
    masses = np.zeros(len(parent))
    for c in range(len(parent)):
        if parent[c] >= 0 and 0 < below[c] < total:
            masses[c] = time[parent[c]] - time[c]
    return masses


@kernel(overloads=((_I, _F, nb.float64),), cache=True)
def _cut(parent, time, u):
    b, t = -1, 0.0
    for c in range(len(parent)):
        if parent[c] >= 0:
            d = time[parent[c]] - time[c]
            if d > 0:
                b, t = c, time[c] + min(u, d)
                if u < d:
                    return b, t
                u -= d
    if b < 0:
        raise ValueError('tree has no positive branch')
    return b, t


@kernel(overloads=((nb.int64, nb.int64, nb.int64, _I, _I),), cache=True)
def _swap_child(q, old, new, left, right):
    if left[q] == old:
        left[q] = new
    else:
        right[q] = new


@kernel(overloads=((nb.int64, nb.int64, nb.float64, _IR, _FR, _FR, _G, *_TREE, _I, _F, _I),), cache=True)
def _reattach_cached(n, root, lam, pp, ps, pz, rng, parent, left, right, time, npop, order, cnt, targets):
    """One cut; return tree-change kind 0 unchanged, 1 times, 2 topology."""
    M, P = len(parent), len(pp)
    b, t = _cut(parent, time, rng.random() * lam)
    p = parent[b]
    my = _ancestor(npop[b], t, pp, ps)
    cnt[:] = 0.0
    for c in range(M):
        tp = time[parent[c]] if parent[c] >= 0 else np.inf
        if time[c] <= t < tp:
            cnt[_ancestor(npop[c], t, pp, ps)] += 1
    iu = 0
    while iu < n - 1 and time[order[iu]] <= t:
        iu += 1
    tau = t
    while True:
        nn = time[order[iu]] if iu < n - 1 else np.inf
        ns = np.inf
        for q in range(P):
            if pp[q] >= 0 and tau < ps[q] < ns:
                ns = ps[q]
        bound = min(nn, ns)
        rate = (cnt[my] / 2.0) / pz[my]
        dt = rng.exponential(1.0 / rate) if rate > 0 else np.inf
        if tau + dt < bound:
            tstar = tau + dt
            break
        if not np.isfinite(bound):
            raise ValueError('nonfinite reattachment time')
        tau = bound
        if ns <= nn:
            for q in range(P):
                if pp[q] >= 0 and ps[q] == tau and cnt[q] > 0:
                    cnt[_ancestor(q, tau, pp, ps)] += cnt[q]
                    cnt[q] = 0.0
            my = _ancestor(my, tau, pp, ps)
        else:
            v = order[iu]
            cnt[_ancestor(npop[v], time[v], pp, ps)] -= 1
            iu += 1
    m = 0
    for c in range(M):
        tp = time[parent[c]] if parent[c] >= 0 else np.inf
        if time[c] <= tstar < tp and _ancestor(npop[c], tstar, pp, ps) == my:
            targets[m] = c
            m += 1
    if m == 0:
        raise AssertionError('reattachment has no target')
    c = targets[rng.integers(0, m)]
    if c == b:
        return root, 0
    s = right[p] if left[p] == b else left[p]
    if c == p or c == s:
        time[p], npop[p] = tstar, my
        return root, 1
    g = parent[p]
    parent[s] = g
    if g >= 0:
        _swap_child(g, p, s, left, right)
    else:
        root = s
    q = parent[c]
    parent[p], left[p], right[p], parent[c] = q, b, c, p
    time[p], npop[p] = tstar, my
    if q >= 0:
        _swap_child(q, c, p, left, right)
    else:
        root = p
    return root, 2


@kernel(overloads=((nb.int64, nb.int64, nb.float64, _IR, _FR, _FR, _G, *_TREE),), cache=True)
def _reattach(n, root, lam, pp, ps, pz, rng, parent, left, right, time, npop):
    """Standalone transition helper, with the traversal's identical stable order."""
    postorder = _postorder(root, left, right)
    internal = postorder[postorder >= n]
    order = internal[np.argsort(time[internal], kind='mergesort')]
    counts, targets = np.empty(len(pp)), np.empty(len(parent), np.int64)
    return _reattach_cached(
        n, root, lam, pp, ps, pz, rng, parent, left, right, time, npop, order, counts, targets)[0]


@kernel(overloads=((nb.int64, nb.float64, _IR, _FR, _FR, _G, _FR, _BR, _FR, _FR,
                    nb.boolean, nb.int64, nb.int64, *_TREE),), cache=True)
def _traverse(root, length, pp, ps, pz, rng, seg_x, carried, site_x, branch_u,
              record, block, ceiling, parent, left, right, time, npop):
    n, L = carried.shape[1], len(site_x)
    cap = min(block, ceiling) if record else 0
    rx, rs = np.empty(cap), np.empty(cap)
    out = np.zeros((n, L), np.int8)
    stack = np.empty(len(parent), np.int64)
    postorder = np.empty(len(parent), np.int64)
    order, order_scratch = np.arange(n, len(parent)), np.empty(len(parent), np.int64)
    below, masses = np.empty(len(parent), np.int64), np.empty(len(parent))
    population_counts, targets = np.empty(len(pp)), np.empty(len(parent), np.int64)
    topology_changed = tree_changed = below_changed = masses_changed = True
    lam, sep = 0.0, 0.0
    x, k, cuts, nrec = 0.0, 0, 0, 0
    j = np.int64(0)
    while True:
        if topology_changed:
            _postorder_into(root, left, right, stack, postorder)
            topology_changed = False
        if tree_changed:
            lam = 0.0
            for c in range(len(parent)):
                if parent[c] >= 0:
                    lam += time[parent[c]] - time[c]
            if not np.isfinite(lam) or lam <= 0:
                raise ValueError('tree length must be positive and finite')
            _time_order_into(n, postorder, time, order, order_scratch)
            tree_changed = False
        change = min(length, x + rng.exponential(1.0 / lam)) if length > x else length
        while True:
            x1 = min(change, seg_x[j + 1])
            if masses_changed:
                _masses_into(
                    n, root, parent, left, right, time, carried, j, postorder, below, masses, np.bool_(below_changed))
                sep = masses.sum()
                below_changed = masses_changed = False
            if record:
                if nrec == cap:
                    if cap >= ceiling:
                        raise ValueError('genealogy interval record reached its stated ceiling')
                    newcap = min(cap + block, ceiling)
                    nx, ns = np.empty(newcap), np.empty(newcap)
                    nx[:nrec], ns[:nrec] = rx, rs
                    rx, rs, cap = nx, ns, newcap
                rx[nrec], rs[nrec] = x, sep
                nrec += 1
            while k < L and (site_x[k] < x1 or x1 == length):
                if sep <= 0:
                    raise ValueError('mutation requested where separating length S is zero')
                pick = _positive_choice(masses, branch_u[k] * sep)
                stack[0], top = pick, 0
                while top >= 0:
                    v = stack[top]
                    top -= 1
                    if v < n:
                        out[v, k] = 1
                    else:
                        top += 1
                        stack[top] = left[v]
                        top += 1
                        stack[top] = right[v]
                k += 1
            x = x1
            if x1 >= seg_x[j + 1] and j < len(seg_x) - 2:
                j += 1
                for c in range(n):
                    if carried[j, c] != carried[j - 1, c]:
                        below_changed = masses_changed = True
                        break
            if x1 >= change or x1 >= length:
                break
        if change >= length:
            break
        cuts += 1
        root, changed = _reattach_cached(
            n, root, lam, pp, ps, pz, rng, parent, left, right, time, npop, order, population_counts, targets)
        if changed > 0:
            tree_changed = masses_changed = True
        if changed == 2:
            topology_changed = below_changed = True
    if k != L:
        raise AssertionError('not every requested mutation site was consumed')
    return rx[:nrec].copy(), rs[:nrec].copy(), cuts, out, _summary(n, root, parent, left, right, time)


def _inputs(leaf_pop, pop_parent, pop_split, pop_size, length, rng, seg_x, carried, site_x, u):
    lp, pp, ps, pz = validate_population_tree(leaf_pop, pop_parent, pop_split, pop_size)
    if not isinstance(rng, np.random.Generator):
        raise InputError('genealogy requires a numpy Generator')
    length = float(length)
    sx, car = frozen_array(seg_x, np.float64), frozen_array(carried, np.bool_)
    sites, uniforms = frozen_array(site_x, np.float64), frozen_array(u, np.float64)
    if not np.isfinite(length) or length < 0:
        raise InputError('contig length must be finite and nonnegative (Morgans)')
    if sx.ndim != 1 or len(sx) < 2 or sx[0] != 0 or sx[-1] != length or np.any(~np.isfinite(sx)):
        raise InputError('carried-set boundaries must span [0, length] in relative Morgans')
    if (length > 0 and np.any(np.diff(sx) <= 0)) or (length == 0 and len(sx) != 2):
        raise InputError('carried-set boundaries must increase; a zero-length extent uses [0, 0]')
    if car.shape != (len(sx) - 1, len(lp)):
        raise InputError('carried must have shape (intervals, sample leaves)')
    if sites.ndim != 1 or uniforms.shape != sites.shape or np.any(~np.isfinite(sites)) or np.any(np.diff(sites) < 0):
        raise InputError('site_x and u must be matching vectors with nondecreasing finite coordinates')
    if (np.any(sites < 0) or np.any(sites > length) or np.any(~np.isfinite(uniforms))
            or np.any(uniforms < 0) or np.any(uniforms > 1)):
        raise InputError('sites must be in [0, length] and mutation uniforms in [0, 1]')
    return lp, pp, ps, pz, length, sx, car, sites, uniforms


def _run(leaf_pop, pop_parent, pop_split, pop_size, length, rng, seg_x, carried, site_x, u, record):
    lp, pp, ps, pz, length, sx, car, sites, uniforms = _inputs(
        leaf_pop, pop_parent, pop_split, pop_size, length, rng, seg_x, carried, site_x, u)
    n = len(lp)
    parent, left, right = (np.full(2 * n - 1, -1, np.int64) for _ in range(3))
    time, npop = np.zeros(2 * n - 1), np.zeros(2 * n - 1, np.int64)
    npop[:n] = lp
    try:
        root = _first_tree(lp, pp, ps, pz, rng, parent, left, right, time, npop)
        first = _summary(n, root, parent, left, right, time)
        proxy = first[1] * length
        if not np.isfinite(first[1]) or proxy > GENEALOGY_WORK_LIMIT:
            raise InputError(f'first-tree length {first[1]} generations x contig length {length} Morgans '
                             f'= work proxy {proxy}; limit {GENEALOGY_WORK_LIMIT}')
        result = _traverse(root, length, pp, ps, pz, rng, sx, car, sites, uniforms,
                           record, GENEALOGY_RECORD_BLOCK, GENEALOGY_INTERVAL_LIMIT,
                           parent, left, right, time, npop)
    except ValueError as error:
        if 'interval record reached' in str(error):
            raise InputError(f'genealogy interval record reached ceiling {GENEALOGY_INTERVAL_LIMIT} intervals '
                             f'(allocation steps of {GENEALOGY_RECORD_BLOCK} intervals)') from error
        raise InputError(str(error)) from error
    return result, first


def separating_length(leaf_pop, pop_parent, pop_split, pop_size, length, rng, seg_x, carried):
    """Pass 1: (starts float64[I], S float64[I] in generations, cuts int).

    Record every tree/carried-set interval, including self-return cuts. Coordinates
    are relative Morgans; the last interval includes length, even if length is zero.
    """
    empty = np.empty(0)
    (starts, S, cuts, _, _), _ = _run(leaf_pop, pop_parent, pop_split, pop_size, length,
                                    rng, seg_x, carried, empty, empty, True)
    return starts, S, cuts


def mutations(leaf_pop, pop_parent, pop_split, pop_size, length, rng, seg_x, carried, site_x, u):
    """Pass 2: int8[n,L] derived alleles, after the caller restores the pass-1 RNG state.

    Equal site coordinates are legal (flat maps). An S=0 site is an InputError;
    every supplied site is consumed. Reference orientation belongs to the caller.
    """
    result, _ = _run(leaf_pop, pop_parent, pop_split, pop_size,
                     length, rng, seg_x, carried, site_x, u, False)
    return result[3]


def tree_summaries(leaf_pop, pop_parent, pop_split, pop_size, length, rng, seg_x, carried):
    """Test helper: first and last unascertained TreeSummary, plus cut count."""
    empty = np.empty(0)
    result, first = _run(leaf_pop, pop_parent, pop_split, pop_size,
                         length, rng, seg_x, carried, empty, empty, False)
    last = result[4]
    return (TreeSummary(*first[:3], first[3:]), TreeSummary(*last[:3], last[3:]), result[2])
