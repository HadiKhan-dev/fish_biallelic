"""Reusable block geometry for a one-sided bound on the unchanged M1 HMM.

Each block uses a shared founder emission envelope and a transition envelope
covering every member of a painted-word class. Taking the smaller bound from
the two aligned block partitions preserves validity. Length eight is a workspace
choice, never a biological threshold: uncertified scores are evaluated exactly.

Method: docs/methods/pedigree.md#candidate-panel
PE1–PE3 preserve this 3.5 method.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from . import certificates as kernel
from .parameters import (
    SELECTIVE_BOUND_GROUP_BINS,
    SELECTIVE_BOUND_ULPS,
    SELECTIVE_UPPER_STATE_FACTOR,
    SELECTIVE_UPPER_ULPS,
)


@dataclass
class Block:
    first: int
    stop: int
    which: np.ndarray
    refs: np.ndarray
    tv: np.ndarray
    mass: np.ndarray
    reference_mass: np.ndarray


class Geometry:
    def __init__(self, component, projection, candidates, claim_threads):
        p = projection
        n = len(p.available)
        self.partitions = []
        self.all_unavailable = not np.any(p.available & candidates)
        if self.all_unavailable:
            return
        claim_threads()
        dense = kernel.dense_bridges(p.stay, p.alpha, p.beta, p.row_arm, p.column_arm, p.pivot, p.branch, p.excursion)
        masses = dense.sum(-1)
        normal = np.divide(dense, masses[:, :, :, None], out=np.zeros_like(dense), where=masses[:, :, :, None] > 0)
        del dense
        labels = np.sort(component.state_classes, axis=2)
        keys = [
            hashlib.sha256(component.genotype_likelihoods[i].tobytes() + component.observed[i].tobytes()).digest()
            for i in range(n)
        ]
        # A group's reference is its member with the smallest key, the lowest sample index among equal keys (min()'s
        # first minimum over members in ascending order): one rank per sample orders both.
        by_key = sorted(range(n), key=lambda q: (keys[q], q))
        rank = np.empty(n, np.int64)
        rank[by_key] = np.arange(n)
        by_key = np.asarray(by_key, np.int64)
        parents = np.flatnonzero(candidates & p.available)
        edges = list(range(0, p.n_bins, SELECTIVE_BOUND_GROUP_BINS)) + [p.n_bins]
        for bounds in (edges, sorted({p.n_bins - e for e in edges})):
            blocks = []
            for first, stop in zip(bounds[:-1], bounds[1:], strict=True):
                word = np.ascontiguousarray(labels[:, first:stop]).reshape(n, -1)
                _, inverse = np.unique(word[parents], axis=0, return_inverse=True)
                inverse = inverse.reshape(-1)
                which = np.full(n, -1, np.int64)
                which[parents] = inverse
                groups = int(inverse.max() + 1)
                lowest = np.full(groups, n, np.int64)
                np.minimum.at(lowest, inverse, rank[parents])
                refs = by_key[lowest]
                tv = np.zeros((groups, stop - first, p.n_states))
                mass = np.ones_like(tv)
                reference = np.ones_like(tv)
                j = max(first, 1)
                target = slice(j - first, stop - first)
                source = slice(j - 1, stop - 1)
                # Release 4.10: every parent's distance to its group's reference at once (each sum over the last
                # axis is the same reduction as for one group's members), then the groups' maxima (exact in any
                # order); the per-group Python loop dominated the screen set-up of a long chromosome.
                distance = 0.5 * np.abs(normal[parents, source] - normal[refs[inverse], source]).sum(-1)
                order = np.argsort(inverse, kind="stable")
                starts = np.searchsorted(inverse[order], np.arange(groups))
                tv[:, target] = np.nextafter(
                    np.maximum.reduceat(distance[order], starts, axis=0) + SELECTIVE_BOUND_ULPS * np.finfo(float).eps,
                    np.inf,
                )
                mass[:, target] = np.nextafter(
                    np.maximum.reduceat(masses[parents, source][order], starts, axis=0)
                    * (1 + SELECTIVE_BOUND_ULPS * np.finfo(float).eps),
                    np.inf,
                )
                held = masses[refs, source]
                reference[:, target] = np.where(held > 0, held, 1.0)
                blocks.append(Block(first, stop, which, refs, tv, mass, reference))
            self.partitions.append(blocks)

    def upper(self, p, t, children, zero, claim_threads):
        if self.all_unavailable:
            return np.broadcast_to(zero[children, None], (len(children), len(p.available))).copy()
        claim_threads()
        common = kernel.global_common_logs(
            children, p.state_alt, t.gl, t.coefficient, t.site_index, t.site_start, t.site_stop, t.mismatch
        )
        claim_threads()
        powered, scales = kernel.power_common(common, children, t.exponent)
        answers = []
        for partition in self.partitions:
            upper = np.zeros((len(children), len(p.available)))
            for b in partition:
                claim_threads()
                values = kernel.block_upper(
                    children,
                    b.first,
                    b.stop,
                    b.refs,
                    b.tv,
                    b.mass,
                    b.reference_mass,
                    t.null_diagonal,
                    t.null_off,
                    t.null_frequency,
                    p.stay,
                    p.alpha,
                    p.beta,
                    p.row_arm,
                    p.column_arm,
                    p.pivot,
                    p.branch,
                    p.excursion,
                    powered,
                    scales,
                )
                active = np.flatnonzero(b.which >= 0)
                upper[:, active] += values[:, b.which[active]]
            with np.errstate(divide="ignore", invalid="ignore"):
                upper += np.log(np.sum(p.pi[:, 0], axis=1))[None]
                pad = (
                    SELECTIVE_UPPER_ULPS
                    * np.finfo(float).eps
                    * (t.gl.shape[1] + SELECTIVE_UPPER_STATE_FACTOR * p.n_bins * p.n_states)
                    * np.maximum(1, np.abs(upper))
                )
                upper += pad
            upper[~np.isfinite(upper)] = np.inf
            upper[:, ~p.available] = zero[children, None]
            answers.append(upper)
        return np.minimum(*answers)
