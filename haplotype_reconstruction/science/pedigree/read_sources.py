"""Every sample's source chain as a candidate parent, with the painting's read state (LR phase 2, Tier 1).

The chain is an HMM over (ordered diplotype u = (u1, u2), read class c): S^2 C states, S = classes + BACKGROUND, C the
painting's read classes (C = 1 without read state). Between bins b and b + 1 the state moves from (u, c) to (v, c')
with weight A_b(u, v) E_{b+1}(v)[c, c'], where

    E_b(v)[c, c'] = exp(e_b(v) + l_b(v)) M_b(v)[c, c'],

A_b is the painting's label-level Hamming transition (same / one / two founders change; applied in O(S^2) per class),
e_b the tempered scalar bin emission at the painting's PATH strength on the reference grid, and M_b with log scale l_b
the bin's structured read product (painting.read_operator): it carries the class at the previous bin's last selected site through
the bin's sites to the class at its last site; the first bin starts from the class law at the component's first
selected site. An ordered v with v1 > v2 reads the unordered pair's matrix with both class indices permuted by J
(read_factor.swap_classes), so the chain is invariant under the homologue relabelling (a, b, c) <-> (b, a, J c). The
diplotype prior of the first bin is uniform; it leaves out the painting's panel-error state, so the chain is not the
marginal of the painting's own chain. Without read state (C = 1, M = 1, l = 0) it is the scalar chain.

Exact forward-backward per sample: the backward class vectors beta_b(u, c) (_backward, max-normalised per bin), then
the transmission-oriented law (_oriented): a parent transmits one homologue per bin, which one switching between
adjacent bins with the meiotic switch probability tau_b. Over the oriented state (x, y, c) (x transmitted, c in x's
homologue frame) the propagation is

    K_b((u, c), (v, c')) = A_b(u, v) E_{b+1}(v)[c, c'] beta_{b+1}(v, c') / beta_b(u, c),
    P_{b+1}(x, y, c') = (1 - tau_b) D(x, y, c') + tau_b D(y, x, J c'),   D(v, c') = sum_{u, c} P_b(u, c) K_b(...),

from P_0, the homologue-symmetrised first-bin posterior. Its class-free law is the homologue-symmetrised marginal
of the chain at every bin (the selector is stationary at 1/2 and independent of the parent's data); the three flows
of the transmitted coordinate (same state, leaving a state, entering a state) depend on tau. Per class they are the
C = 1 formulas with the class's right weights r_b(v, c) = sum_c' E_{b+1}(v)[c, c'] beta_{b+1}(v, c'); summed over the
classes. Stored per sample and bin (class-free): the unordered pair masses (exposure; they determine the oriented
law), the transmitted marginal pi, the three flows, and the transmitted ALT probability of BACKGROUND at each marker.

BACKGROUND (no panel haplotype) transmits an allele the panel does not give; it is conditioned on the candidate's own
reads at the marker: P(a_x = ALT | x = BACKGROUND, data) = sum_{y, c} P(y, c_m | x) P(a_x = ALT | x, y, c_m, the
site's reads), the class c_m at the marker's site from the read state's within-bin chain (_background_alt: the entering
class vector of the oriented pair, the bin's sites' operators and tempered factors, the bin's end vector beta_b) and
the site's configuration evidence (1 - eps) L_a kappa_c(a) + eps B untempered (L the genotype likelihoods, kappa the
class likelihood ratio, B the cohort law; painting.read_factor's terms). A sample without read state (C = 1, or deep
at every site, where the painting bypasses it) has kappa = 1: its mixed genotype likelihoods.
Named classes transmit their panel allele (BACKGROUND's frequency where uncalled), never conditioned.

Method: docs/methods/pedigree.md#ps05
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ..painting.api import empty_tape, swap_classes
from .read_gaps import ForeignReads
from .read_source_kernels import _background_alt, _backward, _oriented
from .sources import SourceInferenceError


@dataclass(frozen=True)
class MarkerInputs:
    """The component's selected markers: state_alt float64 [S, M] (BACKGROUND last), mixed genotype likelihoods
    float64 [n, M, 3] and observed bool [n, M] of the batch, site int64 [M] (each marker's index among the painting's
    selected sites), bin_sizes int64 [B] (markers per bin, bin-major)."""

    state_alt: np.ndarray
    gl: np.ndarray
    observed: np.ndarray
    site: np.ndarray
    bin_sizes: np.ndarray


@dataclass(frozen=True)
class ReadInputs:
    """One sample batch's read state at the path strength (painting.read_decode.ReadEmissions): read float32
    [n, U, B, C, C] and read_log float64 [n, U, B] per unordered label pair (upper-triangle order), initial float64
    [n, C] the class law at the first selected site, minor the split-class parameter (C = 2 minor + 3)."""

    read: np.ndarray
    read_log: np.ndarray
    initial: np.ndarray
    minor: int

    def chain_inputs(self):
        return (self.read, self.read_log, self.initial, swap_classes(self.minor), empty_tape(),
                np.empty((0, 0)))


@dataclass(frozen=True)
class SiteParts:
    """Per-site read-state parts of the batch at the painting's selected sites (ReadEmissionBuilder.site_parts):
    deep bool [n] (read state bypassed), composed float64 [L, n, C, C] class operators from the previous selected site,
    laws float64 [L, n, C] class laws, class_lik float64 [n, L, 4, C]; gl float64 [n, L, 3] raw genotype likelihoods,
    seen bool [n, L] observed flags, cohort float64 [L, 3] cohort genotype law, alt float64 [S, L] label ALT
    probabilities (BACKGROUND last), weight float64 [n, B] the bins' tempering weights, site_start / site_stop int64
    [B] the bins' site ranges, epsilon the outlier rate."""

    deep: np.ndarray
    composed: np.ndarray
    laws: np.ndarray
    class_lik: np.ndarray
    gl: np.ndarray
    seen: np.ndarray
    cohort: np.ndarray
    alt: np.ndarray
    weight: np.ndarray
    site_start: np.ndarray
    site_stop: np.ndarray
    epsilon: float
    foreign: ForeignReads | None = None


def pair_features(states: int) -> tuple[np.ndarray, np.ndarray]:
    """(first, second) int64 [S (S + 1) / 2]: the unordered pairs in upper-triangle order (pair masses' features)."""
    first, second = np.triu_indices(states)
    return first.astype(np.int64), second.astype(np.int64)


def _pair_index(states: int, pairs: int) -> np.ndarray:
    """int64 [S, S]: the unordered pair (upper-triangle index) of each ordered diplotype; all 0 for one shared pair."""
    if pairs == 1:
        return np.zeros((states, states), dtype=np.int64)
    first, second = np.triu_indices(states)
    index = np.empty((states, states), dtype=np.int64)
    index[first, second] = np.arange(len(first))
    index[second, first] = np.arange(len(first))
    return index


def _writable(array: np.ndarray, dtype) -> np.ndarray:
    """A C-contiguous writable array of ``dtype`` (a copy only when needed): the kernels' declared argument types."""
    return np.require(array, dtype=dtype, requirements=["C", "W", "A"])


def batch_laws(
    emissions: np.ndarray,
    transition,
    selector: np.ndarray,
    available: np.ndarray,
    markers: MarkerInputs,
    *,
    read: ReadInputs | None = None,
    sites: SiteParts | None = None,
    component_index: int = -1,
    sample_offset: int = 0,
    sample_rows: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Class-free source laws of one sample batch (keys pair_mass [n, B, F], pi [n, B, S], diagonal_flow /
    outgoing_flow / incoming_flow [n, B-1, S], background_alt [n, M]).

    ``emissions`` float64 [n, S*S, B] the tempered log bin emissions (ordered pair first * S + second); ``read`` the
    batch's read state (None: C = 1), ``sites`` its per-site parts (needed with ``read``). Samples without
    ``available`` evidence keep the uniform chain (no read state, zero emissions). Raises
    SourceInferenceError (original sample index ``sample_offset`` + i) when an available sample loses all mass."""
    candidates, diplotypes, bins = emissions.shape
    states = int(round(math.sqrt(diplotypes)))
    emission = np.ascontiguousarray(np.transpose(emissions, (0, 2, 1)).reshape(candidates, bins, states, states))
    available = _writable(available, np.bool_)
    if read is None:
        read_matrix = np.ones((candidates, 1, bins, 1, 1), dtype=np.float32)
        read_log = np.zeros((candidates, 1, bins), dtype=np.float64)
        initial = np.ones((candidates, 1), dtype=np.float64)
        perm = np.zeros(1, dtype=np.int64)
        tape, exponent = empty_tape(), np.empty((0, 0))
        has_reads = np.zeros(candidates, dtype=np.bool_)
    else:
        if sites is None:
            raise ValueError("the read state's source laws need its per-site parts")
        read_matrix, read_log, initial, perm, tape, exponent = read.chain_inputs()
        read_matrix = _writable(read_matrix, read_matrix.dtype)
        read_log = _writable(read_log, np.float64)
        initial = _writable(initial, np.float64)
        has_reads = available & ~np.asarray(sites.deep, dtype=np.bool_)
    if not np.all(available):
        initial = initial.copy()
        initial[~available] = 0.0
        initial[~available, 0] = 1.0
    classes = initial.shape[1]
    pair_index = _pair_index(states, read_log.shape[1])
    bin_args = (emission, read_matrix, read_log, tape, exponent, pair_index, perm, available)
    same = _writable(transition.same, np.float64)
    one = _writable(transition.one_change, np.float64)
    two = _writable(transition.two_changes, np.float64)
    beta = np.empty((candidates, bins, classes, states, states), dtype=np.float64)
    failed = np.full(candidates, -1, dtype=np.int64)
    valid = _backward(*bin_args, same, one, two, beta, failed)
    _raise_lost(valid, failed, sample_offset, component_index, "candidate source chain lost all supported mass",
                sample_rows)
    pair_first, pair_second = pair_features(states)
    out = dict(
        pair_mass=np.empty((candidates, bins, len(pair_first)), dtype=np.float64),
        pi=np.empty((candidates, bins, states), dtype=np.float64),
        diagonal_flow=np.empty((candidates, max(0, bins - 1), states), dtype=np.float64),
        outgoing_flow=np.empty((candidates, max(0, bins - 1), states), dtype=np.float64),
        incoming_flow=np.empty((candidates, max(0, bins - 1), states), dtype=np.float64),
    )
    background_law = np.empty((candidates, bins, states), dtype=np.float64)
    entering = np.empty((candidates, bins, classes, states), dtype=np.float64)
    failed[:] = -1
    valid = _oriented(
        *bin_args, initial, same, one, two,
        _writable(selector, np.float64), beta, pair_first, pair_second, out["pair_mass"], out["pi"],
        out["diagonal_flow"], out["outgoing_flow"], out["incoming_flow"], background_law, entering, failed,
    )
    _raise_lost(valid, failed, sample_offset, component_index, "transmission-oriented source propagation lost mass",
                sample_rows)
    sizes = np.asarray(markers.bin_sizes, dtype=np.int64)
    marker_stop = np.cumsum(sizes).astype(np.int64)
    marker_start = marker_stop - sizes
    if sites is None:
        empty2 = np.zeros((0, 0), dtype=np.float64)
        site_args = (
            np.zeros(bins, dtype=np.int64), np.zeros(bins, dtype=np.int64), empty2,
            np.zeros((0, candidates, classes, classes), dtype=np.float64),
            np.zeros((0, candidates, classes), dtype=np.float64),
            np.zeros((candidates, 0, 4, classes), dtype=np.float64),
            np.zeros((candidates, 0, 3), dtype=np.float64), np.zeros((candidates, 0), dtype=np.bool_),
            np.zeros((0, 3), dtype=np.float64), 0.0, np.zeros((candidates, bins), dtype=np.float64),
        )
    else:
        site_args = (
            _writable(sites.site_start, np.int64), _writable(sites.site_stop, np.int64),
            _writable(sites.alt, np.float64), _writable(sites.composed, np.float64),
            _writable(sites.laws, np.float64), _writable(sites.class_lik, np.float64),
            _writable(sites.gl, np.float64), _writable(sites.seen, np.bool_), _writable(sites.cohort, np.float64),
            float(sites.epsilon), _writable(sites.weight, np.float64),
        )
    foreign = sites.foreign if sites is not None else None
    if foreign is None:
        foreign_args = (np.empty((0, 0, 0)), np.empty(0), np.empty(0), np.empty((0, 0)), np.empty((0, 0)), np.empty(0))
    else:
        foreign_args = (foreign.counts, foreign.error, foreign.loss, foreign.survival, foreign.absent,
                        _writable(foreign.depth, np.float64))
    out["background_alt"] = np.empty((candidates, len(markers.site)), dtype=np.float64)
    lost = np.zeros((candidates, bins), dtype=np.bool_)
    _background_alt(
        background_law, entering, beta, has_reads, _writable(markers.gl, np.float64),
        _writable(markers.observed, np.bool_), _writable(markers.state_alt, np.float64),
        _writable(markers.site, np.int64), marker_start, marker_stop, *site_args, *foreign_args, out["background_alt"], lost,
    )
    if np.any(lost):
        sample, block = map(int, np.argwhere(lost)[0])
        raise SourceInferenceError(
            "BACKGROUND's within-bin read-class posterior lost all mass",
            _sample_number(sample, sample_offset, sample_rows), component_index, block,
        )
    return out


def mixed_batch_laws(emissions, transition, selector, available, markers, *, read, sites, component_index,
                     sample_offset) -> dict[str, np.ndarray]:
    """Use the scalar and read-state source chains on their own rows of one contiguous evidence batch."""
    scalar = np.flatnonzero(sites.deep)
    if not len(scalar):
        return batch_laws(emissions, transition, selector, available, markers, read=read, sites=sites,
                          component_index=component_index, sample_offset=sample_offset)
    carried = np.flatnonzero(~sites.deep)
    result: dict[str, np.ndarray] = {}
    for rows in (scalar, carried):
        if not len(rows):
            continue
        group_read = group_sites = None
        if rows is carried:
            group_read = ReadInputs(read.read[rows], read.read_log[rows], read.initial[rows], read.minor)
            group_sites = replace(
                sites, deep=sites.deep[rows], composed=sites.composed[:, rows], laws=sites.laws[:, rows],
                class_lik=sites.class_lik[rows], gl=sites.gl[rows], seen=sites.seen[rows], weight=sites.weight[rows],
            )
        laws = batch_laws(emissions[rows], transition, selector, available[rows],
                          replace(markers, gl=markers.gl[rows], observed=markers.observed[rows]),
                          read=group_read, sites=group_sites, component_index=component_index,
                          sample_offset=sample_offset, sample_rows=rows)
        for name, values in laws.items():
            if name not in result:
                result[name] = np.empty((len(available), *values.shape[1:]), dtype=values.dtype)
            result[name][rows] = values
    return result


def _sample_number(row: int, offset: int, rows: np.ndarray | None) -> int:
    return offset + (row if rows is None else int(rows[row]))


def _raise_lost(valid: np.ndarray, failed: np.ndarray, offset: int, component: int, message: str,
                rows: np.ndarray | None) -> None:
    if np.all(valid):
        return
    sample = int(np.flatnonzero(~valid)[0])
    raise SourceInferenceError(message, _sample_number(sample, offset, rows), component, int(failed[sample]))
