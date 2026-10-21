"""Bounded, order-free proposal geometry (A256).

Exact descriptor duplicates share an anchor identity, never an observation weight.
Selection takes complete probability-q distance classes: nearest to the population
prototype first, then farthest from the current set. An oversized class stops
selection; it is not broken by content order. Hierarchies see at most A anchors.
Lifted groups contain each original record once, including every nearest-anchor tie.
This changes proposal density and geometry, not the final likelihood or priors.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba import njit

QUANTUM = 2.0 ** -20


@njit(cache=True)
def _agreement_to(rows, targets, ceiling):
    n, sites, _ = rows.shape
    out = np.empty((n, len(targets)), dtype=np.float64)
    for i in range(n):
        for j in range(len(targets)):
            total = 0.0
            for s in range(sites):
                total += (rows[i,s,0]*targets[j,s,0] + rows[i,s,1]*targets[j,s,1]
                          + rows[i,s,2]*targets[j,s,2])
            out[i,j] = max(0.0, ceiling - total / sites)
    return out


@njit(cache=True)
def _agreement_ceiling(rows):
    ceiling = 0.0
    for i in range(len(rows)):
        total = 0.0
        for s in range(rows.shape[1]):
            total += rows[i,s,0]**2 + rows[i,s,1]**2 + rows[i,s,2]**2
        ceiling = max(ceiling, total / rows.shape[1])
    return ceiling


@njit(cache=True)
def _pack(rows):
    known = np.zeros((len(rows), (rows.shape[1]+63)//64), dtype=np.uint64)
    alt = np.zeros_like(known)
    for i in range(len(rows)):
        for s in range(rows.shape[1]):
            if rows[i,s] >= 0:
                bit = np.uint64(1) << np.uint64(s & 63)
                known[i,s>>6] |= bit
                if rows[i,s] == 1:
                    alt[i,s>>6] |= bit
    return known, alt


@njit(cache=True)
def _popcount(x):
    n = 0
    while x:
        x &= x - np.uint64(1)
        n += 1
    return n


@njit(cache=True, nogil=True)
def _hamming_to(known, alt, target_known, target_alt, minimum_joint):
    out = np.empty((len(known),len(target_known)), dtype=np.float64)
    for i in range(len(known)):
        for j in range(len(target_known)):
            joint_count, mismatch_count = 0, 0
            for w in range(known.shape[1]):
                joint = known[i,w] & target_known[j,w]
                joint_count += _popcount(joint)
                mismatch_count += _popcount((alt[i,w]^target_alt[j,w]) & joint)
            out[i,j] = 1.0 if joint_count < minimum_joint else mismatch_count / joint_count
    return out


def _winning(values, descending=False):
    # Only complete class membership is used, never within-class order, spans,
    # or diagnostics. Preserve the exact chained-gap definition without building
    # and sorting every Python TieClasses record at each of A selections.
    values = np.asarray(values, dtype=np.float64)
    ordered = np.sort(values)
    if descending:
        ordered = ordered[::-1]
        gaps = ordered[:-1] - ordered[1:]
    else:
        gaps = ordered[1:] - ordered[:-1]
    breaks = np.flatnonzero((ordered[1:] != ordered[:-1]) & ~(gaps <= QUANTUM))
    boundary = ordered[int(breaks[0])] if len(breaks) else ordered[-1]
    return np.flatnonzero(values >= boundary if descending else values <= boundary)


def _nearest_ties(distance):
    """Every member of the first chained probability-q class, per row."""
    if distance.shape[1] <= 1:
        # Zero anchors: empty membership; one anchor: every record belongs to it.
        return np.ones(distance.shape, dtype=bool)
    ordered = np.sort(distance, axis=1)
    split = np.diff(ordered, axis=1) > QUANTUM
    first = np.where(split.any(axis=1), np.argmax(split,axis=1), distance.shape[1]-1)
    return distance <= ordered[np.arange(len(ordered)),first,None]


@njit(cache=True, nogil=True)
def _pack_memberships(nearest, inverse):
    """Each original record once per nearest-anchor tie; duplicates keep their indices."""
    result = np.zeros((nearest.shape[1], (len(inverse)+63)//64), dtype=np.uint64)
    for record in range(len(inverse)):
        bit = np.uint64(1) << np.uint64(record % 64)
        for anchor in range(nearest.shape[1]):
            if nearest[inverse[record], anchor]:
                result[anchor, record//64] |= bit
    return result


@dataclass
class LandmarkSet:
    rows: np.ndarray
    inverse: np.ndarray
    anchors: np.ndarray
    distances: np.ndarray
    nearest: np.ndarray
    stats: dict

    def lift(self, group):
        if not len(group):
            return ()
        included = self.nearest[:,np.asarray(group,dtype=np.int64)].any(axis=1)
        return tuple(np.flatnonzero(included[self.inverse]).tolist())

    def packed_membership(self):
        return _pack_memberships(self.nearest, self.inverse)

    @property
    def matrix(self):
        return np.ascontiguousarray(self.distances[self.anchors])


def select(rows, budget, *, kind, minimum_joint=1, omissions=None, scope='global'):
    """Select complete landmark classes under the block timing clock."""
    from .search import diagnostic_phase
    with diagnostic_phase('landmarks'):
        return _select(rows, budget, kind=kind, minimum_joint=minimum_joint,
                       omissions=omissions, scope=scope)


def _select(rows, budget, *, kind, minimum_joint=1, omissions=None, scope='global'):
    """O(R log R * L + A*R log R + R*A*L) work; linear-in-R live arrays.

    Selection computes each unique-record/anchor distance once. Hamming distances
    use packed words; genotype agreement uses the original fixed site/genotype sum.
    The Cauchy-Schwarz self-agreement maximum supplies the K1 distance offset.
    No R-by-R array is constructed, including when unique count is <= A.
    """
    from .residual_births import sample_sum
    rows = np.ascontiguousarray(rows)
    shape = rows.shape[1:]
    unique, inverse = np.unique(rows.reshape(len(rows),-1), axis=0, return_inverse=True)
    unique = np.ascontiguousarray(unique.reshape((-1,)+shape))
    if kind == 'agreement':
        prototype = (sample_sum(rows.reshape(len(rows),-1))/len(rows)).reshape((1,)+shape)
        ceiling = _agreement_ceiling(unique)
        initial = _agreement_to(unique, prototype, ceiling)[:,0]
        def columns(indices):
            out = _agreement_to(unique, unique[indices], ceiling)
            out[indices,np.arange(len(indices))] = 0.0
            return out
    elif kind == 'hamming':
        called = (rows >= 0).sum(axis=0)
        alt_count = (rows == 1).sum(axis=0)
        fraction = np.divide(alt_count, called, out=np.full(shape,.5), where=called>0)
        prototype = np.where(fraction > .5+QUANTUM,1,np.where(fraction < .5-QUANTUM,0,-1)).astype(np.int8)
        known, alt = _pack(unique)
        pk, pa = _pack(prototype[None,:])
        initial = _hamming_to(known,alt,pk,pa,minimum_joint)[:,0]
        def columns(indices):
            out = _hamming_to(known,alt,known[indices],alt[indices],minimum_joint)
            out[indices,np.arange(len(indices))] = 0.0
            return out
    else:
        raise ValueError(kind)
    anchors, pieces = [], []
    available = np.ones(len(unique),dtype=bool)
    nearest = np.full(len(unique),np.inf)
    omitted = 0
    while available.any() and len(anchors) < budget:
        eligible = np.flatnonzero(available)
        scores = initial[eligible] if not anchors else nearest[eligible]
        group = eligible[_winning(scores, descending=bool(anchors))]
        if len(anchors)+len(group) > budget:
            omitted = len(group)
            if omissions is not None:
                omissions.append(dict(kind='tie_class_at_budget',cause=kind+'_landmark_tie_class_at_budget',
                    count=int(len(group)),count_kind='exact',scope=f'{scope}; A={budget}; admitted={len(anchors)}'))
            break
        distance = columns(group)
        nearest = np.minimum(nearest, distance.min(axis=1))
        pieces.append(distance); anchors.extend(group.tolist()); available[group] = False
    matrix = np.concatenate(pieces,axis=1) if pieces else np.empty((len(unique),0))
    ties = _nearest_ties(matrix)
    return LandmarkSet(unique,inverse,np.asarray(anchors,dtype=np.int64),matrix,ties,
        dict(records=len(rows),unique=len(unique),anchors=len(anchors),budget=int(budget),
             omitted_class=int(omitted),unrepresented=int(available.sum()),
             distance_entries=int(len(unique)*(1+len(anchors))),kind=kind,scope=scope,
             retained_array_bytes=int(unique.nbytes+inverse.nbytes+matrix.nbytes+ties.nbytes)))


def seed_groups(likelihoods, budget, min_cluster_size, omissions=None):
    from .residual_births import exact_hdbscan_labels
    chosen = select(likelihoods,budget,kind='agreement',omissions=omissions,scope='K1 samples')
    if len(chosen.anchors) < min_cluster_size:
        return [], chosen.stats
    labels = exact_hdbscan_labels(chosen.matrix,min_cluster_size=min_cluster_size,
                                 min_samples=min_cluster_size,allow_single_cluster=False)
    groups = {chosen.lift(np.flatnonzero(labels==label)) for label in np.unique(labels) if label != -1}
    return [np.asarray(g,dtype=np.int64) for g in sorted(groups) if g], chosen.stats


def residual_groups(hard_calls, budget, minimum_joint, config, omissions=None, scope='global', cache=None):
    from .residual_births import _cluster
    from hashlib import blake2b
    value = np.ascontiguousarray(hard_calls)
    key = (budget, minimum_joint, config.max_cluster_hamming, config.max_alternative_groups,
           config.max_clique_enumeration, config.max_clique_calls, scope, value.shape,
           blake2b(memoryview(value).cast("B"),digest_size=32).digest())
    if cache is not None and key in cache:
        groups, overflow, stats, details = cache[key]
        if omissions is not None:
            omissions.extend(details)
        return groups, overflow, dict(stats, cache_hit=True)
    details = []
    chosen = select(hard_calls,budget,kind='hamming',minimum_joint=minimum_joint,
                    omissions=details,scope=scope)
    matrix = chosen.matrix
    pairs = len(matrix)*(len(matrix)-1)
    edges = int(np.count_nonzero(matrix <= config.max_cluster_hamming+1e-12))-len(matrix)
    chosen.stats.update(anchor_edges=edges//2, anchor_edge_density=edges/pairs if pairs else 0.0)
    lifted, overflow = _cluster(matrix,config.max_cluster_hamming,config.max_alternative_groups,
        config.max_clique_enumeration,config.max_clique_calls,details,
        lifted=(chosen.packed_membership(), len(hard_calls)))
    result = lifted, bool(overflow or chosen.stats['omitted_class']), chosen.stats
    if omissions is not None:
        omissions.extend(details)
    if cache is not None:
        _remember_residual_groups(cache, key, (*result, tuple(details)))
    return result


def _remember_residual_groups(cache, key, value):
    """Merge recomputation-only pool cache entries in source order."""
    if key in cache:
        return
    groups, _, _, details = value
    size = 512 + sum(64+36*len(group) for group in groups) + 512*len(details)
    if size <= 8*1024*1024:
        if cache.get('bytes', 0)+size > 8*1024*1024:
            cache.clear()
        cache[key] = value
        cache['bytes'] = cache.get('bytes', 0)+size
