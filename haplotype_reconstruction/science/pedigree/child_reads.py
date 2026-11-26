"""Every sample's read state as a child at its observed selected markers (LR phase 2, Tier 2; preparation side).

The child's configuration chains (science.pedigree.read_forward) carry the painting's read class c of the child
(science.shared.reads: which homologue the covering reads come from, carried along the contig by read survival). At
an observed selected marker the child's emission of the transmitted ALT probabilities (a, b) of its two slots (slot 1
= homologue 1) is

    E_c(a, b) = G(a, b) R_c(a, b),   R_c(a, b) = sum_g W_g(a, b) L'_g R_c(g) / sum_g W_g(a, b) L'_g,

G the child's scalar site likelihood (forward.site_likelihood: 3 sum_g W_g L'_g, L' its mixed genotype likelihoods,
W_g the product law of the ordered configuration g = (g1, g2)), and R_c(g) the GL-anchored read ratio of
painting.read_factor at full strength,

    R_c(g) = [(1 - eps) L_g kappa_c(g) + eps B] / [(1 - eps) L_g + eps B],
    kappa_c(g) = A_c(g) / sum_c' pi(c') A_c'(g),

L the raw genotype likelihoods (L_01 = L_10 = the heterozygote's), B the cohort genotype law's likelihood (the outlier
branch), A the class likelihoods of the marker's REF/ALT counts and pi the class law at the marker. sum_c pi(c) R_c = 1,
so a chain with no read sharing (operator rows = the next marker's class law) reduces to the scalar chain at full
strength. Between the child's observed markers the class moves by the composed operators of the painting's selected
sites (the read state is defined on the evidence grid; unobserved markers and unselected sites only carry it). Its
first marker's class law starts the chain.

Stored per sample (ChildReads, child-major over its observed markers): the operator into each observed marker from the
previous one (from the first selected marker for the first), the four configuration ratios per class, and the class law
at the first marker. A sample bypassing the read state (deep at every site of the component: identity operators, unit
ratios) stores none: its chains are the scalar chains exactly.

Method: docs/methods/pedigree.md#child-read-state
"""

from __future__ import annotations

import numpy as np
from numba import prange

from ...contracts.pedigree import ChildReads
from ...core.errors import InvariantError
from ...core.kernels import kernel
from .kernels.read_transfer import OVERLOADS
from .read_gaps import append_site, configuration_ratios, identity_gap

__all__ = ["batch_reads", "join_reads", "resident_terms", "scalar_batch"]


@kernel(overloads=OVERLOADS["child_reads._fill"], cache=True, parallel=True, nogil=True)
def _fill(composed, laws, class_lik, gl, cohort, epsilon, chosen, observed, offsets, operator, ratio,
          alt, counts, error, loss, survival, absent):
    """Fill ``operator`` [K, C, C] and ``ratio`` [K, 4, C] (float32) of the batch's samples (prange) at their observed
    markers (``offsets`` [n + 1] each sample's slots; zero slots: no read state). ``composed`` [L, n, C, C] the class
    operators between consecutive painting sites, ``laws`` [L, n, C], ``class_lik`` [n, L, 4, C], ``gl`` [n, L, 3] raw,
    ``cohort`` [L, 3]; ``chosen`` [M] the markers' painting sites, ``observed`` [n, M]."""
    samples = observed.shape[0]
    markers = len(chosen)
    own = laws.shape[2]
    classes = ratio.shape[2]
    donors = classes - own
    for sample in prange(samples):
        slot = offsets[sample]
        if offsets[sample + 1] == slot:
            continue
        product = np.empty((own + 1, own + 2))
        work = np.empty_like(product)
        kappa = np.empty((4, classes))
        identity_gap(product)
        previous = chosen[0]
        for marker in range(markers):
            site = chosen[marker]
            for step in range(previous + 1, site + 1):
                phi = survival[sample, step] if donors else 1.0
                q = absent[sample, step] if donors else 1.0
                append_site(product, composed[step, sample], laws[step, sample], phi, q, work)
                product, work = work, product
            previous = site
            if not observed[sample, marker]:
                continue
            for i in range(own):
                for j in range(own):
                    operator[slot, i, j] = product[i, j]
            if donors:
                for i in range(own):
                    operator[slot, own, i] = product[own, i]
                    operator[slot, i, own] = product[i, own] / donors
                off = product[own, own + 1] / donors
                operator[slot, own, own] = product[own, own] + off
                operator[slot, own, own + 1] = off
            identity_gap(product)
            g0 = gl[sample, site, 0]
            g1 = gl[sample, site, 1]
            g2 = gl[sample, site, 2]
            outlier = cohort[site, 0] * g0 + cohort[site, 1] * g1 + cohort[site, 2] * g2
            configuration_ratios(class_lik[sample, site], laws[site, sample], alt, site,
                                 counts, error, loss, sample, kappa)
            for config in range(4):
                likelihood = g0 if config == 0 else (g2 if config == 3 else g1)
                mixed = (1.0 - epsilon) * likelihood + epsilon * outlier
                for c in range(classes):
                    value = (((1.0 - epsilon) * likelihood * kappa[config, c] + epsilon * outlier) / mixed
                             if mixed > 0.0 else 1.0)
                    ratio[slot, config, c] = value
            slot += 1


def batch_reads(sites, chosen: np.ndarray, observed: np.ndarray):
    """(counts int64 [n], initial float64 [n, C], operator float32 [K, C, C], ratio float32 [K, 4, C]) of one sample
    batch from its per-site read parts at the painting's selected sites (``sites``: read_sources.SiteParts),
    ``chosen`` int64 [M] the selected markers' indices among those sites and ``observed`` bool [n, M] the batch's
    observed markers (PreparedComponent.observed)."""
    chosen = np.ascontiguousarray(chosen, dtype=np.int64)
    observed = np.ascontiguousarray(observed, dtype=np.bool_)
    deep = np.asarray(sites.deep, dtype=np.bool_)
    own = int(sites.laws.shape[2])
    foreign = getattr(sites, "foreign", None)
    donors = len(sites.alt) if foreign is not None else 0
    classes = own + donors
    counts = np.where(deep, 0, observed.sum(axis=1)).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    gap_shape = (own + 1, own + 2) if donors else (own, own)
    operator = np.zeros((int(offsets[-1]), *gap_shape), dtype=np.float32)
    ratio = np.empty((int(offsets[-1]), 4, classes), dtype=np.float32)
    initial = np.zeros((len(deep), classes), dtype=np.float64)
    initial[:, 0] = 1.0
    if len(chosen):
        first = np.asarray(sites.laws[chosen[0]], dtype=np.float64)
        alt = np.empty((0, 0))
        if foreign is None:
            initial[~deep] = first[~deep]
            args = (np.empty((0, 0, 0)), np.empty(0), np.empty(0), np.empty((0, 0)), np.empty((0, 0)))
        else:
            absence = np.exp(-foreign.depth[~deep])
            initial[~deep, :own] = absence[:, None] * first[~deep]
            initial[~deep, own:] = ((1.0 - absence) / donors)[:, None]
            alt = np.require(sites.alt, dtype=np.float64, requirements=["C", "W"])
            args = (foreign.counts, foreign.error, foreign.loss, foreign.survival, foreign.absent)
        _fill(np.ascontiguousarray(sites.composed, dtype=np.float64),
              np.ascontiguousarray(sites.laws, dtype=np.float64),
              np.ascontiguousarray(sites.class_lik, dtype=np.float64), np.ascontiguousarray(sites.gl, dtype=np.float64),
              np.ascontiguousarray(sites.cohort, dtype=np.float64), float(sites.epsilon), chosen, observed, offsets,
              operator, ratio, alt, *args)
    return counts, initial, operator, ratio


def scalar_batch(samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A bypassed batch, with no read slots and its entering law in class zero."""
    return (np.zeros(samples, dtype=np.int64), np.ones((samples, 1), dtype=np.float64),
            np.empty((0, 1, 1), dtype=np.float32), np.empty((0, 4, 1), dtype=np.float32))


def join_reads(
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], classes: int | None = None
) -> ChildReads:
    """Gather sample batches into the prepared record's configured class width and sample order."""
    if classes is None:
        classes = max(batch[1].shape[1] for batch in batches)
    counts = np.concatenate([b[0] for b in batches])
    initial = np.zeros((len(counts), classes), dtype=np.float64)
    operators, ratios = [], []
    gap_shape = max((b[2].shape[1:] for b in batches))
    first = 0
    for batch_counts, entry, operator, ratio in batches:
        width = entry.shape[1]
        if width != classes and (width != 1 or len(operator)):
            raise InvariantError("child read batches differ in class width")
        stop = first + len(batch_counts)
        initial[first:stop, :width] = entry
        first = stop
        if len(operator):
            operators.append(operator)
            ratios.append(ratio)
    return ChildReads(
        offsets=np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
        initial=initial,
        operator=(
            np.concatenate(operators, dtype=np.float32) if operators else np.empty((0, *gap_shape), np.float32)
        ),
        ratio=np.concatenate(ratios, dtype=np.float32) if ratios else np.empty((0, 4, classes), np.float32),
    )


@kernel(overloads=OVERLOADS["child_reads.class_layout"], cache=True, parallel=True, nogil=True)
def class_layout(operator, offsets, site_start, site_stop, classes=0):
    """Exact equality of the rounded gap rows, and the class width at each bin boundary."""
    children, bins = site_stop.shape
    if classes == 0:
        classes = operator.shape[1]
    stored_rows, stored_columns = operator.shape[1:]
    structured = stored_columns > stored_rows
    own = stored_rows - 1 if structured else stored_rows
    reset = np.empty(len(operator), np.bool_)
    width = np.ones((children, bins + 1), np.int64)
    for child in prange(children):
        width[child, 0] = classes
        first, stop = offsets[child], offsets[child + 1]
        for k in range(first, stop):
            identical = True
            for i in range(1, stored_rows):
                for j in range(own):
                    if operator[k, i, j] != operator[k, 0, j]:
                        identical = False
                        break
                if structured and operator[k, i, own] != operator[k, 0, own]:
                    identical = False
                if not identical:
                    break
            if structured and classes - own > 1 and operator[k, own, own + 1] != operator[k, 0, own]:
                identical = False
            reset[k] = identical
        for b in range(bins):
            nxt = first + site_stop[child, b] - site_start[child, 0]
            if nxt < stop and not reset[nxt]:
                width[child, b + 1] = classes
    return reset, width


def resident_terms(reads: ChildReads | None, children: int, site_start: np.ndarray, site_stop: np.ndarray):
    """The child chains' read arguments (offsets int64 [N + 1], initial float64 [N, C], operator float32 [K, C, C],
    ratio float32 [K, 4, C]) of a component's ChildReads (None: one class, no slots) on the children's observed-marker
    CSR (forward.observed_sites). Owned writable copies (the kernels' declared argument types)."""
    if reads is None:
        return (np.zeros(children + 1, dtype=np.int64), np.ones((children, 1), dtype=np.float64),
                np.zeros((0, 1, 1), dtype=np.float32), np.zeros((0, 4, 1), dtype=np.float32))
    offsets = np.array(reads.offsets, dtype=np.int64, copy=True)
    if len(offsets) != children + 1:
        raise InvariantError("child read terms do not cover the component's samples")
    slots = np.diff(offsets)
    observed = site_stop[:, -1] - site_start[:, 0] if site_start.shape[1] else np.zeros(children, dtype=np.int64)
    carried = slots > 0
    if np.any(slots[carried] != observed[carried]):
        raise InvariantError("child read slots differ from the children's observed markers")
    operator = np.array(reads.operator, dtype=np.float32, order="C", copy=True)
    ratio = np.array(reads.ratio, dtype=np.float32, order="C", copy=True)
    return offsets, np.array(reads.initial, dtype=np.float64, order="C", copy=True), operator, ratio
