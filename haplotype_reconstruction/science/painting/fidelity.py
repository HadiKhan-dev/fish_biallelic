"""Painting's copying chain with panel errors: class x fidelity track states.

Model (docs/methods/painting.md). A track state is (label, fidelity): the label a trajectory class or BACKGROUND
(state S - 1 of the S labels), the fidelity faithful (0) or panel error (1; anchored classes only). A faithful track
emits its class's calls (two faithful tracks of one class share their latent allele, as in the robust bin emission);
an error track emits as BACKGROUND (alleles from the population frequencies): its row is locally wrong for this
sample, or the sample's haplotype is locally absent from the panel, while the founder label continues.

Per track and bin edge, independently of the other track: the label follows the jump process of painting.transitions
(it stays with probability 1 - theta, else moves to one of the other S - 1 labels uniformly); a label that stays keeps
or changes its fidelity by the two-state chain faithful -> error with probability a, error -> faithful with
probability b (panel-error runs along the panel; 1 / b bins is the mean length of a run that keeps its label); a
label that moves draws its fidelity from the chain's stationary law (error with probability a / (a + b)); BACKGROUND
and unanchored classes are always faithful. A panel-error run thus contributes log(1/a) at its entry and log(1/b) at
its exit (besides its duration and label-stay factors) and keeps the founder label, where a label-level model pays two
ancestry changes (an excursion) to escape a locally wrong row. Painted switches are label
changes only.

Emissions are the robust bin emissions, each sample's bin divided by its design effect 1 + (n - 1) r
(painting.emissions.design_effect, PaintingTransitions.site_correlation), the composite-likelihood adjustment of the
per-site product.

The first bin's prior is the chain's stationary law (labels uniform; an anchored label's fidelity error with
probability a / (a + b)); the mirrored traceback convention decodes the same model backwards (transposed transition,
that prior at its last step).

Track states are ordered: the S faithful states (index = label), then one error state per anchored class in label
order. The pair transition is the product of the two tracks' transitions, so every recursion runs in two stages
(track 1, then track 2), each O(S') per state from per-label sums: O(S'^2) per bin and sample with S' <= 2 S - 1
track states (K = 30: S' <= 61).

Parameters (rho, a, b) are estimated per component by EM over all its eligible samples (estimate_chain).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numba import prange, types

from ...contracts.roles import freeze_record_arrays, record_field
from ...core.errors import InvariantError
from ...core.kernels import kernel
from ...core.numerics import ordered_sum
from . import hmm
from .parameters import (
    EM_DECREASE_NATS, EM_DECREASE_RELATIVE, ENTRY_BOUNDS, EXIT_BOUNDS, GOLDEN_STEPS, RATE_BOUNDS, RATE_GRID_POINTS,
    VITERBI_TIE,
)
from .transitions import MINIMUM_SWITCH_PROBABILITY, switch_probabilities

@dataclass(frozen=True)
class TrackStates:
    """The track states of a component: label, fidelity and emission label int64[S'] (an error track emits as
    BACKGROUND), and error_of int64[S] (the error state of each label, -1 when it has none)."""

    labels: np.ndarray = record_field("state", axes=("S_prime",), dtype="int64", units="1")
    fidelity: np.ndarray = record_field("state", axes=("S_prime",), dtype="int64", units="1")
    emission: np.ndarray = record_field("state", axes=("S_prime",), dtype="int64", units="1")
    error_of: np.ndarray = record_field("state", axes=("S",), dtype="int64", units="1")

    @property
    def n_track(self) -> int:
        return len(self.labels)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


def track_states(capable: np.ndarray) -> TrackStates:
    """The track states for ``capable`` int[S] (1 = anchored class with an error state; BACKGROUND last, 0)."""
    capable = np.asarray(capable, dtype=np.int64)
    n_labels = len(capable)
    errors = np.flatnonzero(capable)
    labels = np.concatenate((np.arange(n_labels), errors)).astype(np.int64)
    fids = np.concatenate((np.zeros(n_labels), np.ones(len(errors)))).astype(np.int64)
    emission = np.where(fids == 1, n_labels - 1, labels).astype(np.int64)
    error_of = np.full(n_labels, -1, dtype=np.int64)
    error_of[errors] = n_labels + np.arange(len(errors))
    return TrackStates(labels, fids, emission, error_of)


# Supported layouts for typed stage routes.
_OVERLOADS = {
    "_structured_emissions": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.int64,
        ),
    ),
    "_reversed_structured_emissions": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.int64,
        ),
    ),
    "_track_labels": (
        (
            types.Array(types.int32, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.boolean,
        ),
        (
            types.Array(types.int32, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.boolean,
        ),
    ),
    "_log_terms": (
        (
            types.float64,
            types.int64,
            types.float64,
            types.float64,
        ),
    ),
    "_first_within": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.int64,
            types.boolean,
        ),
    ),
    "_max_stage": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.boolean,
            types.boolean,
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.boolean,
            types.boolean,
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_viterbi_kernel": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.float64,
            types.boolean,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.float64,
            types.boolean,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=True, aligned=True),
            types.float64,
            types.float64,
            types.float64,
            types.boolean,
        ),
    ),
    "_sum_stage": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.float64,
            types.float64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.float64,
            types.float64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_transposed_parts": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.float64,
            types.float64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.float64,
            types.float64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_sum_stage_probabilities": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_transposed_parts_probabilities": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_chain_kernel": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
            types.boolean,
            types.boolean,
            types.boolean,
            types.boolean,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
            types.boolean,
            types.boolean,
            types.boolean,
            types.boolean,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=True, aligned=True),
            types.float64,
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
            types.boolean,
            types.boolean,
            types.boolean,
            types.boolean,
        ),
    ),
    "masses_from_pairs": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int32, 1, "C", readonly=False, aligned=True),
            types.Array(types.int32, 2, "C", readonly=False, aligned=True),
        ),
    ),
    "label_errors": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int32, 3, "C", readonly=False, aligned=True),
            types.Array(types.int32, 2, "C", readonly=False, aligned=True),
        ),
    ),
}


@kernel(overloads=_OVERLOADS["_structured_emissions"], cache=True, parallel=True, nogil=True)
def _structured_emissions(raw, emission_label, n_labels):
    n_samples, _, n_bins = raw.shape
    n_track = len(emission_label)
    out = np.empty((n_samples, n_track * n_track, n_bins))
    for task in prange(n_samples * n_track):
        sample, first = task // n_track, task % n_track
        for second in range(n_track):
            source = emission_label[first] * n_labels + emission_label[second]
            for bin_index in range(n_bins):
                out[sample, first * n_track + second, bin_index] = raw[sample, source, bin_index]
    return out


@kernel(overloads=_OVERLOADS["_reversed_structured_emissions"], cache=True, parallel=True, nogil=True)
def _reversed_structured_emissions(raw, emission_label, n_labels):
    """_structured_emissions of the bin-reversed label emissions, read in place."""
    n_samples, _, n_bins = raw.shape
    n_track = len(emission_label)
    out = np.empty((n_samples, n_track * n_track, n_bins))
    for task in prange(n_samples * n_track):
        sample, first = task // n_track, task % n_track
        for second in range(n_track):
            source = emission_label[first] * n_labels + emission_label[second]
            for bin_index in range(n_bins):
                out[sample, first * n_track + second, bin_index] = raw[sample, source, n_bins - 1 - bin_index]
    return out


def structured_emissions(raw: np.ndarray, tracks: TrackStates, *, reverse: bool = False) -> np.ndarray:
    """float64[N, S'*S', B] pair emissions of the track states from the (tempered) label emissions raw [N, S*S, B];
    ``reverse``: of the bins in reverse order (the mirrored convention), without copying ``raw``."""
    kernel_function = _reversed_structured_emissions if reverse else _structured_emissions
    return kernel_function(np.ascontiguousarray(raw), tracks.emission, len(tracks.error_of))


@kernel(overloads=_OVERLOADS["_track_labels"], cache=True, parallel=True, nogil=True)
def _track_labels(states, labels, reverse):
    """int32[N, 2, B] labels of the Viterbi track states ``states``; ``reverse``: states of the bin-reversed chain
    (the mirrored convention), returned in the original bin order. The same values as labels[states]."""
    n_samples, n_tracks, n_bins = states.shape
    out = np.empty((n_samples, n_tracks, n_bins), dtype=np.int32)
    for sample in prange(n_samples):
        for track in range(n_tracks):
            for bin_index in range(n_bins):
                source = n_bins - 1 - bin_index if reverse else bin_index
                out[sample, track, bin_index] = labels[states[sample, track, source]]
    return out


@kernel(overloads=_OVERLOADS["_log_terms"], cache=True, inline="always", nogil=True)
def _log_terms(theta, n_labels, a, b):
    """(log stay, log move to one particular other label, log F[2, 2], log pi[2])."""
    log_stay = math.log1p(-theta)
    log_move = math.log(theta) - math.log(n_labels - 1.0) if n_labels > 1 else -np.inf
    fidelity = np.empty((2, 2))
    fidelity[0, 0] = math.log1p(-a)
    fidelity[0, 1] = math.log(a)
    fidelity[1, 0] = math.log(b)
    fidelity[1, 1] = math.log1p(-b)
    error = a / (a + b)
    stationary = np.empty(2)
    stationary[0] = math.log1p(-error)
    stationary[1] = math.log(error)
    return log_stay, log_move, fidelity, stationary


def initial_log(tracks: TrackStates, capable: np.ndarray, a: float, b: float) -> np.ndarray:
    """float64[S']: log of the chain's stationary law of a track's fidelity given its label (labels uniform, a
    constant left out): log(1 - a/(a + b)) and log(a/(a + b)) for an anchored label's faithful and error states, 0 for
    the other labels. The first bin's prior (the chain is stationary, so its marginal at either end
    of a component is this law; a uniform law over track states would give the error states as much prior mass as the
    faithful ones). The mirrored Viterbi convention applies it at the original first bin, its last step."""
    capable = np.asarray(capable, dtype=np.int64)
    error = a / (a + b)
    result = np.zeros(tracks.n_track)
    anchored = capable[tracks.labels] == 1
    result[anchored & (tracks.fidelity == 0)] = math.log1p(-error)
    result[anchored & (tracks.fidelity == 1)] = math.log(error)
    return result


@kernel(overloads=_OVERLOADS["_first_within"], cache=True, inline="always", nogil=True)
def _first_within(values, labels, threshold, excluded, reverse):
    """The smallest index y (the largest under ``reverse``) with values[y] >= threshold and labels[y] != excluded
    (-1 when there is none)."""
    n = len(values)
    for rank in range(n):
        y = n - 1 - rank if reverse else rank
        if labels[y] != excluded and values[y] >= threshold:
            return y
    return -1


@kernel(overloads=_OVERLOADS["_max_stage"], cache=True, nogil=True)
def _max_stage(
    values,
    labels,
    fids,
    error_of,
    capable,
    log_stay,
    log_move,
    fidelity,
    stationary,
    tie,
    reverse,
    transpose,
    adjusted,
    out_value,
    out_index,
):
    """out[z] = max_y values[y] + log T(y -> z) over track states (exact maxima; with ``transpose``, log T(z -> y): the
    mirrored chain of the other traceback convention, since the chain is not time-symmetric), with the argmax under
    the stated tie rule (VITERBI_TIE): stay in z when within ``tie`` of the maximum, else the label's other fidelity
    state when within ``tie`` of a label change, else a label change from the smallest index (the largest under
    ``reverse``) whose value is within ``tie`` of the best other label's. A label change has one weight for every
    source of another label up to a term of the target's fidelity (forward: its stationary probability) or of the
    source's (transposed), so the best change comes from the largest ``adjusted`` value (values plus the source term)
    among the other labels: the global best M1 (label L1), or, for z of label L1, the best M2 of the labels other than
    L1. ``adjusted`` is a work array of length S'. O(S')."""
    n = len(values)
    for y in range(n):
        adjusted[y] = values[y]
        if transpose and capable[labels[y]]:
            adjusted[y] += stationary[fids[y]]
    best = -np.inf
    best_label = -1
    for y in range(n):
        if adjusted[y] > best:
            best = adjusted[y]
            best_label = labels[y]
    second = -np.inf
    for y in range(n):
        if labels[y] != best_label and adjusted[y] > second:
            second = adjusted[y]
    # Tie-chosen sources of a label change: for targets not of label L1 (threshold M1 - tie), the first index within
    # the tie, or the first within it among labels other than that index's; for targets of label L1 (M2 - tie).
    first = _first_within(adjusted, labels, best - tie, -1, reverse)
    first_other = _first_within(adjusted, labels, best - tie, labels[first], reverse) if first >= 0 else -1
    second_first = _first_within(adjusted, labels, second - tie, best_label, reverse) if second > -np.inf else -1
    for z in range(n):
        label = labels[z]
        fid = fids[z]
        # Same-label predecessors: the label's faithful state (index = label) and its error state.
        top = -np.inf
        top_index = -1
        stay_value = -np.inf
        for y_fid in range(2):
            y = label if y_fid == 0 else error_of[label]
            if y < 0:
                continue
            if capable[label]:
                value = values[y] + log_stay + (fidelity[fid, y_fid] if transpose else fidelity[y_fid, fid])
            elif fid == 0:
                value = values[y] + log_stay
            else:
                value = -np.inf
            if y == z:
                stay_value = value
            if value > top:
                top = value
                top_index = y
        # A label change: the exact best value among the other labels, from the tie-chosen source.
        if label != best_label:
            change = best
            source = first if labels[first] != label else first_other
        else:
            change = second
            source = second_first
        if source >= 0 and change > -np.inf:
            target = 0.0
            if not transpose:
                if capable[label]:
                    target = stationary[fid]
                elif fid != 0:
                    target = -np.inf
            value = change + log_move + target
            if value > top + tie:
                top = value
                top_index = source
            elif value > top:
                top = value  # within the tie: keep the same-label predecessor
        if stay_value >= top - tie and stay_value > -np.inf:
            top_index = z
        out_value[z] = top
        out_index[z] = top_index


@kernel(overloads=_OVERLOADS["_viterbi_kernel"], cache=True, parallel=True, nogil=True)
def _viterbi_kernel(emissions, labels, fids, error_of, capable, initial, theta, a, b, tie, reverse):
    """Max-product recursion over the bins in the given order; ``reverse`` (the mirrored convention, emissions and
    theta reversed by the caller) uses the transposed transition and the reversed index ties, and takes the initial
    law at its last step (the original first bin), so both conventions maximise the same path score."""
    n_samples, n_pairs, n_bins = emissions.shape
    n = len(labels)
    n_labels = len(capable)
    result = np.empty((n_samples, 2, n_bins), dtype=np.int32)
    for sample in prange(n_samples):
        back = np.empty((n_bins, n_pairs), dtype=np.int32)
        previous = np.empty(n_pairs)
        for first in range(n):
            for second in range(n):
                state = first * n + second
                previous[state] = emissions[sample, state, 0]
                if not reverse:
                    previous[state] += initial[first] + initial[second]
        previous -= np.max(previous)
        column = np.empty(n)
        adjusted = np.empty(n)
        stage_value = np.empty((n, n))
        stage_index = np.empty((n, n), dtype=np.int64)
        out_value = np.empty(n)
        out_index = np.empty(n, dtype=np.int64)
        current = np.empty(n_pairs)
        for bin_index in range(1, n_bins):
            log_stay, log_move, fidelity, stationary = _log_terms(theta[bin_index - 1], n_labels, a, b)
            # Stage 1: track 1, for every predecessor state of track 2.
            for y2 in range(n):
                for y1 in range(n):
                    column[y1] = previous[y1 * n + y2]
                _max_stage(
                    column,
                    labels,
                    fids,
                    error_of,
                    capable,
                    log_stay,
                    log_move,
                    fidelity,
                    stationary,
                    tie,
                    reverse,
                    reverse,
                    adjusted,
                    out_value,
                    out_index,
                )
                for z1 in range(n):
                    stage_value[z1, y2] = out_value[z1]
                    stage_index[z1, y2] = out_index[z1]
            # Stage 2: track 2.
            for z1 in range(n):
                for y2 in range(n):
                    column[y2] = stage_value[z1, y2]
                _max_stage(
                    column,
                    labels,
                    fids,
                    error_of,
                    capable,
                    log_stay,
                    log_move,
                    fidelity,
                    stationary,
                    tie,
                    reverse,
                    reverse,
                    adjusted,
                    out_value,
                    out_index,
                )
                for z2 in range(n):
                    y2 = out_index[z2]
                    current[z1 * n + z2] = out_value[z2] + emissions[sample, z1 * n + z2, bin_index]
                    back[bin_index, z1 * n + z2] = stage_index[z1, y2] * n + y2
            current -= np.max(current)
            previous[:] = current
        if reverse:
            for first in range(n):
                for second in range(n):
                    previous[first * n + second] += initial[first] + initial[second]
        top = np.max(previous)
        state = 0
        for rank in range(n_pairs):
            candidate = n_pairs - 1 - rank if reverse else rank
            if previous[candidate] >= top - tie:
                state = candidate
                break
        result[sample, 0, n_bins - 1] = state // n
        result[sample, 1, n_bins - 1] = state % n
        for bin_index in range(n_bins - 1, 0, -1):
            state = back[bin_index, state]
            result[sample, 0, bin_index - 1] = state // n
            result[sample, 1, bin_index - 1] = state % n
    return result


def viterbi_track_states(
    emissions: np.ndarray,
    tracks: TrackStates,
    capable: np.ndarray,
    theta: np.ndarray,
    a: float,
    b: float,
    *,
    tie: float = VITERBI_TIE,
) -> np.ndarray:
    """Default-convention track indices; label_paths owns mirrored expansion."""
    capable = np.ascontiguousarray(capable, dtype=np.int64)
    arguments = (tracks.labels, tracks.fidelity, tracks.error_of, capable, initial_log(tracks, capable, a, b))
    return _viterbi_kernel(
        np.ascontiguousarray(emissions), *arguments, np.ascontiguousarray(theta), float(a), float(b), float(tie), False
    )


@kernel(overloads=_OVERLOADS["_sum_stage_probabilities"], cache=True, inline="always", nogil=True)
def _sum_stage_probabilities(values, labels, fids, error_of, capable, stay, move, fidelity, stationary, per_label, out):
    """out[z] = sum_y values[y] T(y -> z) (probabilities, values scaled), O(S')."""
    n = len(values)
    total = 0.0
    per_label[:] = 0.0
    for y in range(n):
        total += values[y]
        per_label[labels[y]] += values[y]
    for z in range(n):
        label = labels[z]
        fid = fids[z]
        if capable[label]:
            faithful = values[label]
            error = values[error_of[label]]
            same = stay * (faithful * fidelity[0, fid] + error * fidelity[1, fid])
            moved = move * (total - per_label[label]) * stationary[fid]
        else:
            same = stay * values[label] if fid == 0 else 0.0
            moved = move * (total - per_label[label]) if fid == 0 else 0.0
        out[z] = same + moved


@kernel(overloads=_OVERLOADS["_transposed_parts_probabilities"], cache=True, inline="always", nogil=True)
def _transposed_parts_probabilities(
    values,
    labels,
    fids,
    error_of,
    capable,
    stay,
    move,
    fidelity,
    stationary,
    scratch,
    total,
    moved,
    started,
    ended,
    kept,
    moved_error,
    moved_faithful,
):
    """For every track state y: total[y] = sum_z T(y -> z) values[z], and its parts moved (label changes), started
    (faithful -> error, same label), ended (error -> faithful, same label), kept (same label and fidelity, for an
    anchored label), and of moved the parts into an anchored label's error state (moved_error) and faithful state
    (moved_faithful). O(S')."""
    n = len(values)
    weighted = 0.0  # sum_z pi(z) values[z]
    weighted_error = 0.0  # the same over anchored labels' error states
    weighted_faithful = 0.0  # and over anchored labels' faithful states
    scratch[:, :] = 0.0
    per_label, per_label_error, per_label_faithful = scratch[0], scratch[1], scratch[2]
    for z in range(n):
        label = labels[z]
        if capable[label]:
            value = stationary[fids[z]] * values[z]
            if fids[z] == 1:
                weighted_error += value
                per_label_error[label] += value
            else:
                weighted_faithful += value
                per_label_faithful[label] += value
        else:
            value = values[z] if fids[z] == 0 else 0.0
        weighted += value
        per_label[label] += value
    for y in range(n):
        label = labels[y]
        fid = fids[y]
        moved[y] = move * (weighted - per_label[label])
        moved_error[y] = move * (weighted_error - per_label_error[label])
        moved_faithful[y] = move * (weighted_faithful - per_label_faithful[label])
        started[y] = 0.0
        ended[y] = 0.0
        kept[y] = 0.0
        if capable[label]:
            faithful = values[label]
            error = values[error_of[label]]
            keep = stay * fidelity[fid, fid] * (faithful if fid == 0 else error)
            flip = stay * fidelity[fid, 1 - fid] * (error if fid == 0 else faithful)
            if fid == 0:
                started[y] = flip
            else:
                ended[y] = flip
            kept[y] = keep
            total[y] = moved[y] + keep + flip
        else:
            total[y] = moved[y] + stay * values[label]


@kernel(
    overloads=((types.float64[:, ::1], types.float64[::1]),),
    cache=True, inline="always", nogil=True,
)
def _legacy_fidelity_probabilities(fidelity, stationary):
    probabilities, law = np.empty((2, 2)), np.empty(2)
    for fid in range(2):
        law[fid] = math.exp(stationary[fid])
        for target in range(2):
            probabilities[fid, target] = math.exp(fidelity[fid, target])
    return probabilities, law


# The archived scalar-chain oracle uses the original log-probability contractions. Its wrappers share
# the contraction arithmetic with the current chain, which supplies probabilities and scratch directly.
@kernel(overloads=_OVERLOADS["_sum_stage"], cache=True, inline="always", nogil=True)
def _sum_stage(values, labels, fids, error_of, capable, n_labels, log_stay, log_move, fidelity, stationary, out):
    probabilities, law = _legacy_fidelity_probabilities(fidelity, stationary)
    _sum_stage_probabilities(
        values, labels, fids, error_of, capable, math.exp(log_stay), math.exp(log_move),
        probabilities, law, np.empty(n_labels), out,
    )


@kernel(overloads=_OVERLOADS["_transposed_parts"], cache=True, inline="always", nogil=True)
def _transposed_parts(
    values, labels, fids, error_of, capable, n_labels, log_stay, log_move, fidelity, stationary,
    total, moved, started, ended, kept, moved_error, moved_faithful,
):
    probabilities, law = _legacy_fidelity_probabilities(fidelity, stationary)
    _transposed_parts_probabilities(
        values, labels, fids, error_of, capable, math.exp(log_stay), math.exp(log_move),
        probabilities, law, np.empty((3, n_labels)),
        total, moved, started, ended, kept, moved_error, moved_faithful,
    )


@kernel(overloads=_OVERLOADS["_chain_kernel"], cache=True, parallel=True, nogil=True)
def _chain_kernel(
    emissions, labels, fids, error_of, capable, initial, theta, a, b, eligible, want_masses, emission_masses,
    label_masses, bin_likelihood,
):
    """Exact scaled forward-backward of the chain, per eligible sample: expected label changes per edge (both
    tracks); expected error-run starts and ends and kept faithful and kept error transitions of anchored labels that
    stay, and label moves into an anchored label's error and faithful states (summed over edges); the first bin's
    expected anchored tracks in error and faithful; the log-likelihood; (``want_masses``) the posterior masses of
    every label pair with both tracks faithful, float64[N, S*S, B]; and (``bin_likelihood``) the log-likelihood's
    per-bin terms float64[N, B], the log normaliser of the scaled forward step at bin b, log P(x_b | x_1..x_b-1): the
    log-likelihood is their sum in bin order (0 for an ineligible sample)."""
    n_samples, n_pairs, n_bins = emissions.shape
    n = len(labels)
    n_labels = len(capable)
    label_changes = np.zeros((n_samples, max(n_bins - 1, 0)))
    starts = np.zeros(n_samples)
    ends = np.zeros(n_samples)
    kept_faithful = np.zeros(n_samples)
    kept_error = np.zeros(n_samples)
    into_error = np.zeros(n_samples)
    into_faithful = np.zeros(n_samples)
    first_error = np.zeros(n_samples)
    first_faithful = np.zeros(n_samples)
    log_likelihood = np.zeros(n_samples)
    # with label_masses the second half holds every label pair's mass summed over all fidelity states (an error track
    # keeps its label: the painting's label-error statistic, ADVICE2 11.1)
    faithful_mass = np.zeros(
        (n_samples if want_masses or emission_masses else 0, n_labels * n_labels * (2 if label_masses else 1), n_bins)
    )
    bin_log_likelihood = np.zeros((n_samples if bin_likelihood else 0, n_bins))
    # Both sweeps and every sample use the same probabilities. Keep the original log-to-probability
    # conversion so this change only moves work out of the recurrent bin and track-pair loops.
    _, _, log_fidelity, log_stationary = _log_terms(0.5, n_labels, a, b)
    fidelity, stationary = np.empty((2, 2)), np.empty(2)
    for fid in range(2):
        stationary[fid] = math.exp(log_stationary[fid])
        for target in range(2):
            fidelity[fid, target] = math.exp(log_fidelity[fid, target])
    stays, moves = np.empty(len(theta)), np.empty(len(theta))
    log_labels = math.log(n_labels - 1.0) if n_labels > 1 else 0.0
    for edge in prange(len(theta)):
        stays[edge] = math.exp(math.log1p(-theta[edge]))
        moves[edge] = math.exp(math.log(theta[edge]) - log_labels) if n_labels > 1 else 0.0
    # Each sample's bin message depends on the preceding bin in that sweep.
    for sample in prange(n_samples):
        if not eligible[sample]:
            continue
        forward = np.empty((n_bins, n_pairs))
        column = np.empty(n)
        out = np.empty(n)
        moved = np.empty(n)
        started = np.empty(n)
        ended = np.empty(n)
        kept = np.empty(n)
        moved_error = np.empty(n)
        moved_faithful = np.empty(n)
        stage = np.empty((n, n))
        scratch = np.empty((3, n_labels))
        per_label = scratch[0]
        top = -np.inf
        for p in range(n_pairs):
            value = emissions[sample, p, 0] + initial[p // n] + initial[p % n]
            forward[0, p] = value
            if value > top:
                top = value
        for p in range(n_pairs):
            forward[0, p] = math.exp(forward[0, p] - top)
        scale = np.sum(forward[0])
        term = top + math.log(scale)
        log_likelihood[sample] += term
        if bin_likelihood:
            bin_log_likelihood[sample, 0] = term
        forward[0] /= scale
        for bin_index in range(1, n_bins):
            stay, move = stays[bin_index - 1], moves[bin_index - 1]
            for y2 in range(n):
                for y1 in range(n):
                    column[y1] = forward[bin_index - 1, y1 * n + y2]
                _sum_stage_probabilities(
                    column, labels, fids, error_of, capable, stay, move, fidelity, stationary, per_label, out
                )
                for z1 in range(n):
                    stage[z1, y2] = out[z1]
            top = np.max(emissions[sample, :, bin_index])
            for z1 in range(n):
                for y2 in range(n):
                    column[y2] = stage[z1, y2]
                _sum_stage_probabilities(
                    column, labels, fids, error_of, capable, stay, move, fidelity, stationary, per_label, out
                )
                for z2 in range(n):
                    forward[bin_index, z1 * n + z2] = out[z2] * math.exp(
                        emissions[sample, z1 * n + z2, bin_index] - top
                    )
            scale = np.sum(forward[bin_index])
            term = top + math.log(scale)
            log_likelihood[sample] += term
            if bin_likelihood:
                bin_log_likelihood[sample, bin_index] = term
            forward[bin_index] /= scale
        backward = np.ones(n_pairs)
        weight = np.empty(n_pairs)
        new_backward = np.empty(n_pairs)
        for bin_index in range(n_bins - 1, -1, -1):
            if bin_index == 0:
                total = 0.0
                for p in range(n_pairs):
                    weight[p] = forward[0, p] * backward[p]
                    total += weight[p]
                for p in range(n_pairs):
                    mass = weight[p] / total
                    for track_state in (p // n, p % n):
                        if capable[labels[track_state]]:
                            if fids[track_state] == 1:
                                first_error[sample] += mass
                            else:
                                first_faithful[sample] += mass
            if want_masses or emission_masses:
                total = 0.0
                for p in range(n_pairs):
                    weight[p] = forward[bin_index, p] * backward[p]
                    total += weight[p]
                for p in range(n_pairs):
                    mass = weight[p] / total
                    first, second = p // n, p % n
                    pair = labels[first] * n_labels + labels[second]
                    if emission_masses:
                        i = n_labels - 1 if fids[first] else labels[first]
                        j = n_labels - 1 if fids[second] else labels[second]
                        faithful_mass[sample, i * n_labels + j, bin_index] += mass
                    elif fids[first] == 0 and fids[second] == 0:
                        faithful_mass[sample, pair, bin_index] += mass
                    if label_masses:
                        faithful_mass[sample, n_labels * n_labels + pair, bin_index] += mass
            if bin_index == 0:
                continue
            edge = bin_index - 1
            stay, move = stays[edge], moves[edge]
            top = np.max(emissions[sample, :, bin_index])
            for p in range(n_pairs):
                weight[p] = math.exp(emissions[sample, p, bin_index] - top) * backward[p]
            # Track 1's parts: contract track 2 first, stage[z1, y2] = sum_z2 T(y2 -> z2) w(z1, z2).
            for z1 in range(n):
                for z2 in range(n):
                    column[z2] = weight[z1 * n + z2]
                _transposed_parts_probabilities(
                    column,
                    labels,
                    fids,
                    error_of,
                    capable,
                    stay,
                    move,
                    fidelity,
                    stationary,
                    scratch,
                    out,
                    moved,
                    started,
                    ended,
                    kept,
                    moved_error,
                    moved_faithful,
                )
                for y2 in range(n):
                    stage[z1, y2] = out[y2]
            changes = 0.0
            run_starts = 0.0
            run_ends = 0.0
            keep_faithful = 0.0
            keep_error = 0.0
            enter_error = 0.0
            enter_faithful = 0.0
            for y2 in range(n):
                for z1 in range(n):
                    column[z1] = stage[z1, y2]
                _transposed_parts_probabilities(
                    column,
                    labels,
                    fids,
                    error_of,
                    capable,
                    stay,
                    move,
                    fidelity,
                    stationary,
                    scratch,
                    out,
                    moved,
                    started,
                    ended,
                    kept,
                    moved_error,
                    moved_faithful,
                )
                for y1 in range(n):
                    alpha = forward[edge, y1 * n + y2]
                    new_backward[y1 * n + y2] = out[y1]
                    changes += alpha * moved[y1]
                    enter_error += alpha * moved_error[y1]
                    enter_faithful += alpha * moved_faithful[y1]
                    run_starts += alpha * started[y1]
                    run_ends += alpha * ended[y1]
                    if fids[y1] == 0:
                        keep_faithful += alpha * kept[y1]
                    else:
                        keep_error += alpha * kept[y1]
            # Track 2's parts: contract track 1 first, stage[y1, z2] = sum_z1 T(y1 -> z1) w(z1, z2).
            for z2 in range(n):
                for z1 in range(n):
                    column[z1] = weight[z1 * n + z2]
                _transposed_parts_probabilities(
                    column,
                    labels,
                    fids,
                    error_of,
                    capable,
                    stay,
                    move,
                    fidelity,
                    stationary,
                    scratch,
                    out,
                    moved,
                    started,
                    ended,
                    kept,
                    moved_error,
                    moved_faithful,
                )
                for y1 in range(n):
                    stage[y1, z2] = out[y1]
            for y1 in range(n):
                for z2 in range(n):
                    column[z2] = stage[y1, z2]
                _transposed_parts_probabilities(
                    column,
                    labels,
                    fids,
                    error_of,
                    capable,
                    stay,
                    move,
                    fidelity,
                    stationary,
                    scratch,
                    out,
                    moved,
                    started,
                    ended,
                    kept,
                    moved_error,
                    moved_faithful,
                )
                for y2 in range(n):
                    alpha = forward[edge, y1 * n + y2]
                    changes += alpha * moved[y2]
                    enter_error += alpha * moved_error[y2]
                    enter_faithful += alpha * moved_faithful[y2]
                    run_starts += alpha * started[y2]
                    run_ends += alpha * ended[y2]
                    if fids[y2] == 0:
                        keep_faithful += alpha * kept[y2]
                    else:
                        keep_error += alpha * kept[y2]
            norm = 0.0
            for p in range(n_pairs):
                norm += forward[edge, p] * new_backward[p]
            label_changes[sample, edge] = changes / norm
            starts[sample] += run_starts / norm
            ends[sample] += run_ends / norm
            kept_faithful[sample] += keep_faithful / norm
            kept_error[sample] += keep_error / norm
            into_error[sample] += enter_error / norm
            into_faithful[sample] += enter_faithful / norm
            scale = np.sum(new_backward)
            for p in range(n_pairs):
                backward[p] = new_backward[p] / scale
    return (
        label_changes,
        starts,
        ends,
        kept_faithful,
        kept_error,
        into_error,
        into_faithful,
        first_error,
        first_faithful,
        log_likelihood,
        faithful_mass,
        bin_log_likelihood,
    )


def chain_statistics(
    emissions: np.ndarray,
    tracks: TrackStates,
    capable: np.ndarray,
    theta: np.ndarray,
    a: float,
    b: float,
    eligible: np.ndarray,
    *,
    want_masses: bool = False,
    emission_masses: bool = False,
    label_masses: bool = False,
    bin_likelihood: bool = False,
):
    """(label changes float64[N, B - 1], error-run starts [N], ends [N], kept faithful [N], kept error [N], moves into
    an anchored error state [N] and faithful state [N], first-bin anchored tracks in error [N] and faithful [N],
    log-likelihood [N], faithful label-pair masses [N, S*S, B]) of the structured
    emissions (see _chain_kernel). With emission_masses=True the masses output instead carries normalized
    emission-label masses, mapping each error track separately to BACKGROUND. With label_masses=True (and
    want_masses) the masses output has S*S more rows: every label pair's mass summed over all fidelity states (ADVICE2
    11.1). With bin_likelihood=True a twelfth output follows: the log-likelihood's per-bin terms [N, B] (the
    calibration's material units, ADVICE2 11.2). Existing modes are unchanged."""
    capable = np.ascontiguousarray(capable, dtype=np.int64)
    result = _chain_kernel(
        np.ascontiguousarray(emissions),
        tracks.labels,
        tracks.fidelity,
        tracks.error_of,
        capable,
        initial_log(tracks, capable, a, b),
        np.ascontiguousarray(theta, dtype=np.float64),
        float(a),
        float(b),
        np.ascontiguousarray(eligible, dtype=np.bool_),
        bool(want_masses),
        bool(emission_masses),
        bool(label_masses),
        bool(bin_likelihood),
    )
    return result if bin_likelihood else result[:11]


@kernel(overloads=_OVERLOADS["masses_from_pairs"], cache=True, parallel=True, nogil=True)
def masses_from_pairs(faithful_mass, masks, other_masks, public_ids, local_keys):
    """qF and qAF float64[N,B], preserving the native first/second state summation.

    Both tracks must be faithful; retain the smaller mass of the two conventions.
    Unread qV/qA arrays and their additions are absent.
    """
    n_samples, n_pairs, n_bins = faithful_mass.shape
    n_states = public_ids.shape[0]
    qf = np.zeros((n_samples, n_bins))
    qaf = np.zeros((n_samples, n_bins))
    for task in prange(n_samples * n_bins):
        sample, bin_index = task // n_bins, task % n_bins
        for decoding in range(2):
            source = masks if decoding == 0 else other_masks
            mask_first = source[sample, 0, bin_index]
            mask_second = source[sample, 1, bin_index]
            pf = np.int64(0)
            ps = np.int64(0)
            lf = np.int64(0)
            ls = np.int64(0)
            for state in range(n_states):
                if (mask_first >> state) & 1:
                    pf |= np.int64(1) << public_ids[state]
                    lf |= np.int64(1) << local_keys[bin_index, state]
                if (mask_second >> state) & 1:
                    ps |= np.int64(1) << public_ids[state]
                    ls |= np.int64(1) << local_keys[bin_index, state]
            f = 0.0
            af = 0.0
            for first in range(n_states):
                for second in range(n_states):
                    faithful = faithful_mass[sample, first * n_states + second, bin_index]
                    a1 = public_ids[first]
                    a2 = public_ids[second]
                    if (((pf >> a1) & 1) and ((ps >> a2) & 1)) or (((ps >> a1) & 1) and ((pf >> a2) & 1)):
                        f += faithful
                    k1 = local_keys[bin_index, first]
                    k2 = local_keys[bin_index, second]
                    if (((lf >> k1) & 1) and ((ls >> k2) & 1)) or (((ls >> k1) & 1) and ((lf >> k2) & 1)):
                        af += faithful
            f = min(1.0, max(0.0, f))
            af = min(1.0, max(0.0, af))
            if decoding == 0:
                qf[sample, bin_index] = f
                qaf[sample, bin_index] = af
            else:
                qf[sample, bin_index] = min(qf[sample, bin_index], f)
                qaf[sample, bin_index] = min(qaf[sample, bin_index], af)
    return qf, qaf


@kernel(overloads=_OVERLOADS["label_errors"], cache=True, parallel=True, nogil=True)
def label_errors(label_mass, paths, local_keys):
    """(u float64 [N, B], v float64 [N, 2, B]): the label-error statistic of the released labels ``paths`` (ADVICE2
    11.1), from ``label_mass`` [N, S*S, B] the posterior mass of every ordered label pair summed over all fidelity
    states (an error track keeps its label). With c_0, c_1 the released labels' local classes in a bin:
    u = 1 - mass of pairs whose classes are the multiset {c_0, c_1} (the released pair is wrong); for c_0 != c_1,
    v_t = 1 - mass of pairs carrying c_t (track t's label is wrong); for c_0 == c_1 = c, v_0 = v_1 = 1 - (1/2) sum of
    mass x min(copies of c, 2) (the expected share of the two released copies that are wrong: dosage-aware).
    Independent bins (parallel over sample and bin)."""
    n_samples, n_pairs, n_bins = label_mass.shape
    n_labels = local_keys.shape[1]
    u = np.empty((n_samples, n_bins))
    v = np.empty((n_samples, 2, n_bins))
    for task in prange(n_samples * n_bins):
        sample, b = task // n_bins, task % n_bins
        c0 = local_keys[b, paths[sample, 0, b]]
        c1 = local_keys[b, paths[sample, 1, b]]
        total = 0.0
        pair = 0.0
        first = 0.0
        second = 0.0
        dose = 0.0
        for i in range(n_labels):
            k1 = local_keys[b, i]
            for j in range(n_labels):
                m = label_mass[sample, i * n_labels + j, b]
                if m == 0.0:
                    continue
                k2 = local_keys[b, j]
                total += m
                if (k1 == c0 and k2 == c1) or (k1 == c1 and k2 == c0):
                    pair += m
                if c0 != c1:
                    if k1 == c0 or k2 == c0:
                        first += m
                    if k1 == c1 or k2 == c1:
                        second += m
                else:
                    dose += m * ((1.0 if k1 == c0 else 0.0) + (1.0 if k2 == c0 else 0.0))
        if total <= 0.0:
            u[sample, b] = np.nan
            v[sample, 0, b] = np.nan
            v[sample, 1, b] = np.nan
            continue
        u[sample, b] = min(1.0, max(0.0, 1.0 - pair / total))
        if c0 != c1:
            v[sample, 0, b] = min(1.0, max(0.0, 1.0 - first / total))
            v[sample, 1, b] = min(1.0, max(0.0, 1.0 - second / total))
        else:
            v[sample, 0, b] = min(1.0, max(0.0, 1.0 - 0.5 * dose / total))
            v[sample, 1, b] = v[sample, 0, b]
    return u, v


# ---------------------------------------------------------------------------------------------- the copying model


@dataclass(frozen=True)
class ChainModel:
    """A component's copying model: rate rho (per Morgan per track), a, b, the emissions' site correlation r (the
    design effect's), the per-edge label change probabilities theta float64[B - 1] and the EM trajectory ((rho, a,
    b, log-likelihood) after each update, unrounded; the log-likelihood is of the tempered emissions under the
    values before that update)."""

    rate: float = record_field("state", axes=(), dtype="float64", units="per Morgan")
    entry: float = record_field("state", axes=(), dtype="float64", units="probability")
    exit: float = record_field("state", axes=(), dtype="float64", units="probability")
    site_correlation: float = record_field("state", axes=(), dtype="float64", units="1")
    theta: np.ndarray = record_field("state", axes=("B_minus_1",), dtype="float64", units="probability")
    trajectory: tuple[tuple[float, float, float, float], ...] = record_field(
        "diagnostic", axes=(), dtype="tuple[tuple[float, float, float, float]]", units="nats", default=()
    )
    fitted_log_likelihood: float = record_field(
        "diagnostic", axes=(), dtype="float64", units="nats", default=float("nan"), kw_only=True
    )
    """The log-likelihood of the tempered emissions at the returned (rounded) values, one pass after the last update
    (estimate_chain with final_likelihood; NaN otherwise): the calibration's support curve of the eight-update fit is
    evaluated at the fit it returns, not before its last update (ADVICE2 14.1 item 6)."""
    fitted_sample_log_likelihood: np.ndarray | None = record_field(
        "diagnostic", axes=("N",), dtype="float64", units="nats", default=None, kw_only=True
    )
    """Its per-sample terms (0 for an ineligible sample), the same pass: A11's per-sample rates (ADVICE2 14.1 item 9)."""
    fitted_bin_log_likelihood: np.ndarray | None = record_field(
        "diagnostic", axes=("B",), dtype="float64", units="nats", default=None, kw_only=True
    )
    """Its per-bin terms summed over the samples, the same pass: bin b's term is log P(x_b | x_1..x_b-1), the log
    normaliser of the scaled forward step, so the terms of a stretch of bins are that stretch's share of the
    likelihood (the calibration's material units, ADVICE2 11.2). Each bin's sum over the samples is correctly rounded
    (ordered_sum), the same on any sample batch; the bins together give fitted_log_likelihood up to rounding. None
    without final_likelihood."""
    fitted_counts: tuple[float, ...] = record_field(
        "diagnostic", axes=(), dtype="tuple[float, float, float]", units="1", default=(), kw_only=True
    )
    """The same pass's expected (label changes, anchored track-bins in a faithful state, anchored track-bins in an error
    state), summed over the eligible samples' tracks: the regime of the returned fit, so that a change of regime
    between the calibration's candidate outlier rates is visible (ADVICE2 11.2: per-rate switch counts and error-track
    occupancy; error occupancy = error bins / (faithful + error bins), among anchored track-bins). A bin is counted by
    the transition that reaches it (kept, started or ended on the same label, or a move into the state) or as the first
    bin. Empty without final_likelihood."""

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


def _round_log(value: float, step: float, low: float, high: float) -> float:
    value = min(high, max(low, float(value)))
    return float(min(high, max(low, math.exp(round(math.log(value) / step) * step))))


def rate_objective(rate: float, changes: np.ndarray, tracks: float, morgans: np.ndarray, n_labels: int) -> float:
    """sum_e [n_e log theta_e + (tracks - n_e) log(1 - theta_e)] over the edges of positive map length (the rate's part
    of the expected complete-data log-likelihood), theta_e = (S - 1)/S (1 - exp(-rho d_e S/(S - 1)))."""
    positive = np.asarray(morgans, dtype=np.float64) > 0.0
    n = np.minimum(np.asarray(changes, dtype=np.float64)[positive], tracks)
    d = np.asarray(morgans, dtype=np.float64)[positive]
    theta = np.clip(
        switch_probabilities(d, rate, n_labels), MINIMUM_SWITCH_PROBABILITY, 1.0 - MINIMUM_SWITCH_PROBABILITY
    )
    return float(np.sum(n * np.log(theta) + (tracks - n) * np.log1p(-theta)))


def rate_update(
    changes: np.ndarray, tracks: float, morgans: np.ndarray, n_labels: int, previous: float | None = None
) -> float:
    """The rho in RATE_BOUNDS maximising rate_objective: a log-spaced scan of 241 points, then a golden-section
    refinement around the best. The objective can have more than one local maximum (each edge's term is unimodal in
    rho, their sum need not be), so this is a numerical search, not a certificate; ``previous`` is kept when the
    search does not improve on it (the update never lowers the objective)."""
    low, high = RATE_BOUNDS
    positive = np.asarray(morgans, dtype=np.float64) > 0.0
    n = np.minimum(np.asarray(changes, dtype=np.float64)[positive], tracks)
    if tracks <= 0.0 or not np.any(positive) or n_labels < 2 or float(np.sum(n)) <= 0.0:
        return low if previous is None else float(previous)

    def objective(log_rate: float) -> float:
        return rate_objective(math.exp(log_rate), changes, tracks, morgans, n_labels)

    grid = np.linspace(math.log(low), math.log(high), RATE_GRID_POINTS)
    values = [objective(x) for x in grid]
    best = int(np.argmax(values))
    lo, hi = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(GOLDEN_STEPS):
        if f1 >= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - ratio * (hi - lo)
            f1 = objective(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + ratio * (hi - lo)
            f2 = objective(x2)
    found = math.exp(0.5 * (lo + hi))
    if previous is not None and objective(math.log(found)) < objective(math.log(previous)):
        return float(previous)
    return found


def fidelity_objective(a: float, b: float, counts) -> float:
    """The (a, b) part of the expected complete-data log-likelihood: with s, e the expected error-run starts and ends,
    kf, ke the expected kept faithful and kept error transitions of anchored labels that stay, and ie, if_ the
    expected anchored tracks drawn into error or faithful fidelity from the stationary law (at the first bin and after
    a label move: pi_error = a / (a + b)),
        s log a + kf log(1 - a) + e log b + ke log(1 - b) + ie log a + if_ log b - (ie + if_) log(a + b)."""
    s, kf, e, ke, ie, if_ = counts
    return float(
        (s + ie) * math.log(a)
        + kf * math.log1p(-a)
        + (e + if_) * math.log(b)
        + ke * math.log1p(-b)
        - (ie + if_) * math.log(a + b)
    )


def fidelity_update(counts, previous: tuple[float, float]) -> tuple[float, float]:
    """(a, b) in ENTRY_BOUNDS x EXIT_BOUNDS maximising fidelity_objective: bounded quasi-Newton (L-BFGS-B) in
    (log a, log b) from the previous values and from the stay-only ratios s / (s + kf), e / (e + ke); the previous
    values are kept unless the best start improves on them."""
    from scipy.optimize import minimize

    s, kf, e, ke, _, _ = counts
    bounds = [
        (math.log(ENTRY_BOUNDS[0]), math.log(ENTRY_BOUNDS[1])),
        (math.log(EXIT_BOUNDS[0]), math.log(EXIT_BOUNDS[1])),
    ]

    def clip(value, low_high):
        return min(low_high[1], max(low_high[0], value))

    starts = [(math.log(previous[0]), math.log(previous[1]))]
    if s + kf > 0.0 and e + ke > 0.0 and s > 0.0 and e > 0.0:
        starts.append((clip(math.log(s / (s + kf)), bounds[0]), clip(math.log(e / (e + ke)), bounds[1])))
    best = (fidelity_objective(*previous, counts), (previous[0], previous[1]))
    for x0 in starts:
        result = minimize(
            lambda x: -fidelity_objective(math.exp(x[0]), math.exp(x[1]), counts),
            np.asarray(x0),
            method="L-BFGS-B",
            bounds=bounds,
        )
        candidate = (float(math.exp(result.x[0])), float(math.exp(result.x[1])))
        value = fidelity_objective(*candidate, counts)
        if value > best[0]:
            best = (value, candidate)
    return best[1]


def model_for(
    morgans: np.ndarray, n_labels: int, rate: float, entry: float, exit_: float, site_correlation: float, trajectory=()
) -> ChainModel:
    return ChainModel(
        rate=float(rate),
        entry=float(entry),
        exit=float(exit_),
        site_correlation=float(site_correlation),
        theta=np.ascontiguousarray(switch_probabilities(morgans, rate, n_labels)),
        trajectory=tuple(trajectory),
    )


def check_em_log_likelihoods(log_likelihoods, *, site_correlation: float = math.nan, first_update: int = 0) -> None:
    """Fail fast on a corrupted copying-model EM (AGENTS 5.3): raise InvariantError when a log-likelihood of the
    sequence (the k-th after ``first_update`` + k updates) is not finite, or is below its predecessor by more than
    EM_DECREASE_RELATIVE x |log-likelihood| + EM_DECREASE_NATS. estimate_chain's updates never lower it beyond rounding
    (parameters.EM_DECREASE_RELATIVE)."""
    values = [float(value) for value in log_likelihoods]
    for k, current in enumerate(values):
        updates = first_update + k
        if not math.isfinite(current):
            raise InvariantError(
                f"copying-model EM (site correlation {site_correlation:g}): the log-likelihood after {updates} "
                f"updates is {current}"
            )
        if k:
            previous = values[k - 1]
            tolerance = EM_DECREASE_RELATIVE * max(abs(previous), abs(current)) + EM_DECREASE_NATS
            if previous - current > tolerance:
                raise InvariantError(
                    f"copying-model EM (site correlation {site_correlation:g}): update {updates} lowered the "
                    f"log-likelihood by {previous - current:.6g} nats ({previous:.12g} -> {current:.12g}; tolerance "
                    f"{tolerance:.3g}), so its E-step is numerically corrupted; the fit is not returned"
                )


def estimate_chain(
    raw_batches, capable: np.ndarray, morgans: np.ndarray, config, *, site_correlation: float, final_likelihood: bool = False
) -> ChainModel:
    """EM estimate of (rho, a, b) over all eligible samples of a component. ``raw_batches()`` yields (tempered label
    emissions float64[n, S*S, B], eligible bool[n]) batch by batch, for every sample; it is called once per update.

    From the fixed starting values (PaintingParameters.initial_*), ``estimation_iterations`` updates, each an exact
    forward-backward pass under the current values (chain_statistics) followed by the M-step of the expected
    complete-data log-likelihood: rho by rate_update, (a, b) by fidelity_update (both include every term of the model
    in them: the label changes, the fidelity chain of labels that stay, and the stationary fidelity draws at the first
    bin and after label moves); neither update lowers its part of the objective. A fixed number of updates from fixed
    starts: a deterministic, truncated estimator, not a convergence test. The results are clipped to their bounds and
    rounded to ``estimate_grid`` in natural-log space. The trajectory records, per update, the new values and the
    log-likelihood of the tempered emissions under the values before it (up to the constant 2 log S per sample of
    the uniform label law). A log-likelihood that is not finite or falls beyond rounding raises InvariantError
    (check_em_log_likelihoods).
    """
    capable = np.ascontiguousarray(capable, dtype=np.int64)
    n_labels = len(capable)
    tracks_states = track_states(capable)
    rate = float(config.initial_switch_rate)
    entry = float(config.initial_error_entry)
    exit_ = float(config.initial_error_exit)
    if not np.any(capable):  # no anchored class, no error state: a and b unused, reported as no runs
        entry, exit_ = ENTRY_BOUNDS[0], EXIT_BOUNDS[1]
    site_correlation = float(site_correlation)
    trajectory = []
    for _ in range(int(config.estimation_iterations)):
        theta = switch_probabilities(morgans, rate, n_labels)
        sample_statistics = []
        tracks = 0
        for raw, eligible in raw_batches():
            if hasattr(raw, "path_only"):          # read state carried (painting.read_decode)
                from .read_decode import chain_statistics as read_chain_statistics
                statistics = read_chain_statistics(raw, tracks_states, capable, theta, entry, exit_, eligible)
                (c, s, e, kf, ke, into_error, into_faithful, first_error, first_faithful, ll, _) = statistics
            else:
                emissions = structured_emissions(raw, tracks_states)
                (c, s, e, kf, ke, into_error, into_faithful, first_error, first_faithful, ll, _) = chain_statistics(
                    emissions, tracks_states, capable, theta, entry, exit_, eligible
                )
                del emissions
            sample_statistics.append((c, s, e, kf, ke, into_error, into_faithful, first_error, first_faithful, ll))
            tracks += 2 * int(np.count_nonzero(eligible))
        if tracks == 0:
            break
        # Concatenate contiguous sample batches, then fsum in sample order.
        # Stationary draws are ONE 2N-term sum, never two separately rounded sums.
        c, s, e, kf, ke, into_error, into_faithful, first_error, first_faithful, ll = (
            np.concatenate([batch[index] for batch in sample_statistics], axis=0) for index in range(10)
        )
        changes = np.asarray([ordered_sum(c[:, edge]) for edge in range(len(morgans))], dtype=np.float64)
        counts = (
            ordered_sum(s),
            ordered_sum(kf),
            ordered_sum(e),
            ordered_sum(ke),
            ordered_sum(np.concatenate((into_error, first_error))),
            ordered_sum(np.concatenate((into_faithful, first_faithful))),
        )
        log_likelihood = ordered_sum(ll)
        check_em_log_likelihoods((trajectory[-1][3], log_likelihood) if trajectory else (log_likelihood,),
                                 site_correlation=site_correlation, first_update=max(0, len(trajectory) - 1))
        rate = rate_update(changes, float(tracks), morgans, n_labels, previous=rate)
        if np.any(capable):
            entry, exit_ = fidelity_update(tuple(counts), (entry, exit_))
        trajectory.append((rate, entry, exit_, log_likelihood))
    step = float(config.estimate_grid)
    model = model_for(
        morgans,
        n_labels,
        _round_log(rate, step, *RATE_BOUNDS),
        _round_log(entry, step, *ENTRY_BOUNDS),
        _round_log(exit_, step, *EXIT_BOUNDS),
        site_correlation,
        trajectory,
    )
    if not final_likelihood:
        return model
    # one more exact forward-backward pass, at the returned values (the calibration's support curve, ADVICE2 14.1(6))
    parts, regime, bins = [], [], []
    theta = switch_probabilities(morgans, model.rate, n_labels)  # writable, as in the updates
    for raw, eligible in raw_batches():
        if hasattr(raw, "path_only"):
            from .read_decode import chain_statistics as read_chain_statistics
            statistics = read_chain_statistics(
                raw, tracks_states, capable, theta, model.entry, model.exit, eligible, bin_likelihood=True
            )
        else:
            emissions = structured_emissions(raw, tracks_states)
            statistics = chain_statistics(
                emissions, tracks_states, capable, theta, model.entry, model.exit, eligible, bin_likelihood=True
            )
            del emissions
        parts.append(statistics[9])
        bins.append(statistics[11])
        c, starts, ends, kept_faithful, kept_error, into_error, into_faithful, first_error, first_faithful = statistics[:9]
        # per sample (the same values on any batch size): label changes, anchored faithful bins, anchored error bins
        faithful_bins = kept_faithful + ends + into_faithful + first_faithful
        error_bins = kept_error + starts + into_error + first_error
        regime.append(np.stack((c.sum(axis=1), faithful_bins, error_bins), axis=1))
    per_sample = np.concatenate(parts).astype(np.float64) if parts else np.zeros(0)
    fitted = ordered_sum(per_sample) if parts else 0.0
    rows = np.concatenate(regime, axis=0) if regime else np.zeros((0, 3))
    counts = tuple(float(ordered_sum(np.ascontiguousarray(rows[:, k]))) for k in range(3)) if parts else (0.0, 0.0, 0.0)
    # each bin's term summed over the samples, correctly rounded: the same on any sample batch
    per_bin = (
        ordered_sum(np.concatenate(bins, axis=0), axis=0) if bins else np.zeros(len(morgans) + 1)
    ).astype(np.float64)
    return replace(
        model, fitted_log_likelihood=float(fitted), fitted_sample_log_likelihood=per_sample, fitted_counts=counts,
        fitted_bin_log_likelihood=per_bin,
    )


# ------------------------------------------------------------------------------------------------------- decoding


def label_paths(
    raw: np.ndarray, model: ChainModel, capable: np.ndarray, *, reverse: bool, tie: float = VITERBI_TIE
) -> np.ndarray:
    """int32[N, 2, B] Viterbi labels of the (tempered) label emissions ``raw`` [N, S*S, B] under ``model``. The mirrored
    convention expands the reversed label emissions directly, so one expanded table (8 S'^2 B bytes per sample) is
    alive at a time (viterbi_track_states would copy the expanded table)."""
    if hasattr(raw, "path_only"):
        # a read-state batch reaching label continuity's painted-switch comparison: label continuity compares
        # relabellings on the emissions without read state throughout (its part emissions are built without it), so
        # its check uses the batch's scalar part too; the painting's own decode goes through painting.read_decode
        raw = raw.raw
    tracks = track_states(capable)
    capable = np.ascontiguousarray(capable, dtype=np.int64)
    if not reverse:
        states = viterbi_track_states(
            structured_emissions(raw, tracks), tracks, capable, model.theta, model.entry, model.exit, tie=tie
        )
    else:
        states = _viterbi_kernel(
            structured_emissions(raw, tracks, reverse=True),
            tracks.labels,
            tracks.fidelity,
            tracks.error_of,
            capable,
            initial_log(tracks, capable, model.entry, model.exit),
            np.ascontiguousarray(model.theta[::-1]),
            float(model.entry),
            float(model.exit),
            float(tie),
            True,
        )
    # Map the mirrored chain's states to labels in the original bin order inside one kernel.
    return _track_labels(states, np.ascontiguousarray(tracks.labels, dtype=np.int64), bool(reverse))


def painted_switches(
    raw: np.ndarray, model: ChainModel, capable: np.ndarray, eligible: np.ndarray, *, tie: float = VITERBI_TIE
) -> np.ndarray:
    """int64[N] painted switches exactly as the painting reports them: the minimum unordered label changes of the two
    traceback conventions' Viterbi paths, the fewer of the two (0 for ineligible samples)."""
    switches = np.minimum(
        hmm.minimum_switch_counts(label_paths(raw, model, capable, reverse=False, tie=tie)),
        hmm.minimum_switch_counts(label_paths(raw, model, capable, reverse=True, tie=tie)),
    )
    switches[~np.asarray(eligible, dtype=np.bool_)] = 0
    return switches


def decode_candidates(
    raw,
    model: ChainModel,
    capable,
    local_keys,
    named_alleles,
    observed,
    bin_starts,
    bin_stops,
    n_classes,
    *,
    reverse: bool,
    tie: float = VITERBI_TIE,
):
    """(paths int32[N, 2, B], candidate masks int64[N, 2, B], switch counts int64[N], the mirrored convention's
    paths int32[N, 2, B]) of one batch: painting.fidelity.decode_candidates for the chain's Viterbi labels (the run's
    convention, with the mirrored one; switch intervals, combined masks and the canonical track order as there)."""
    decoded = []
    for convention in (bool(reverse), not bool(reverse)):
        paths = hmm.parsimonious_track_order(
            label_paths(raw, model, capable, reverse=convention, tie=tie), local_keys, n_classes
        )
        masks = hmm.switch_interval_masks(paths, named_alleles, observed, bin_starts, bin_stops, n_classes)
        decoded.append((paths, masks))
    (paths, primary), (other_paths, secondary) = decoded
    counts = np.minimum(hmm.minimum_switch_counts(paths), hmm.minimum_switch_counts(other_paths))
    masks = hmm.combine_candidate_masks(primary, secondary)
    hmm.apply_track_order(hmm.canonical_track_order(masks, local_keys, n_classes), paths, masks)
    return paths, masks, counts, other_paths


def release_posteriors(
    raw, model: ChainModel, capable, paths, other_paths, public_ids, local_keys, *, errors: bool = False
):
    """qF and qAF float64[N,B], the smaller faithful Viterbi-pair mass over both conventions; with ``errors`` also the
    label-error statistic (u [N, B], v [N, 2, B]) of the released convention's labels ``paths`` (label_errors), from
    the same forward-backward pass."""
    tracks = track_states(capable)
    emissions = structured_emissions(raw, tracks)
    *_, masses = chain_statistics(
        emissions,
        tracks,
        capable,
        model.theta,
        model.entry,
        model.exit,
        np.ones(raw.shape[0], dtype=np.bool_),
        want_masses=True,
        label_masses=bool(errors),
    )
    del emissions
    width = len(capable) * len(capable)
    qf, qaf = masses_from_pairs(
        np.ascontiguousarray(masses[:, :width]),
        np.left_shift(np.int64(1), paths.astype(np.int64)),
        np.left_shift(np.int64(1), other_paths.astype(np.int64)),
        np.ascontiguousarray(public_ids),
        np.ascontiguousarray(local_keys),
    )
    if not errors:
        return qf, qaf
    u, v = label_errors(
        np.ascontiguousarray(masses[:, width:]), np.ascontiguousarray(paths, dtype=np.int32),
        np.ascontiguousarray(local_keys, dtype=np.int32),
    )
    return qf, qaf, u, v
