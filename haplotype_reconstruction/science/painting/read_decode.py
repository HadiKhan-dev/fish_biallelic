"""Read-state counterparts of painting.fidelity's batch wrappers (PaintingParameters.read_classes > 1).

A batch's emissions with the read state carried (ReadEmissions): the tempered label emissions of the model without
read state plus, per sample, unordered label pair and bin, the read matrix of tempered per-configuration ratios and
its log scale (painting.read_operator), and the class law at the component's first selected site. Each wrapper mirrors
fidelity's (chain_statistics, label_paths, decode_candidates, release_posteriors) through painting.read_chain's
kernels; with no read sharing they return fidelity's results. painting.fidelity and painting.run dispatch a batch here
when it carries the read state (``hasattr(raw, "path_only")``)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np

from . import hmm
from .fidelity import (
    VITERBI_TIE,
    ChainModel,
    _track_labels,
    initial_log,
    label_errors,
    masses_from_pairs,
    structured_emissions,
    track_states,
)
from .read_chain import read_chain_kernel, read_viterbi_kernel, track_pair_maps
from .read_factor import swap_classes
from .read_operator import compiled_own_block, empty_tape, own_projection

__all__ = ["ReadEmissions", "chain_statistics", "label_paths", "decode_candidates", "release_posteriors"]


@dataclass(frozen=True)
class ReadEmissions:
    """One sample batch's emissions with the read state (an in-memory batch, never stored)."""

    raw: np.ndarray          # float64[N, S*S, B] tempered label emissions (the model without read state)
    _read: np.ndarray         # [N, U, B, C, C] read matrices per unordered label pair (triu order)
    _read_log: np.ndarray     # float64[N, U, B] their log scales
    initial: np.ndarray      # float64[N, C] class law at the component's first selected site
    minor: int               # split-class parameter (C = 2 minor + 3)
    # When the confidence strength differs from the path's, release decoding can ask this callback for
    # (read, read_log, operator). Copying-model fits and label continuity use the path strength.
    confidence: Callable[[], tuple] | None = None
    # when the label-error strength (parameters.LABEL_ERROR_CORRELATION) differs from both run strengths: a callable
    # returning its (read, read_log, operator), built only when the reference decode asks (at_label)
    label: Callable[[], tuple] | None = None
    # float64[N, U, B]: the bin-level tempering correction read_log carries;
    # calibration gives the hidden blocks' visible part the same correction (read_calibration)
    read_shift: np.ndarray | None = None

    operator: tuple | None = None   # site tape (empty when compiled), exponent, extended entry law
    bypass: np.ndarray | None = None  # bool[N], rows supported on class zero alone by design

    @cached_property
    def _projection(self):
        if self.operator is None:
            return self._read, self._read_log
        tape, exponent, _ = self.operator
        if not tape[0].size:
            return compiled_own_block(self._read, self._read_log, self.initial.shape[1])
        return own_projection(tape, exponent, self._read_log)

    @property
    def read(self):
        return self._projection[0]

    @property
    def read_log(self):
        return self._projection[1]

    def _at_strength(self, scaled_raw, build):
        if build is None:
            return replace(self.path_only(), raw=scaled_raw)
        read, read_log, operator = build()
        return ReadEmissions(
            scaled_raw, read, read_log, self.initial, self.minor, operator=operator, bypass=self.bypass,
        )

    def at_confidence(self, scaled_raw):
        return self._at_strength(scaled_raw, self.confidence)

    def at_label(self, scaled_raw):
        return self._at_strength(scaled_raw, self.label)

    def path_only(self):
        """Drop deferred strengths when holding the batch for repeated copying fits."""
        return replace(self, confidence=None, label=None)

    @property
    def matrix_width(self):
        return self._read.shape[-1]

    def fit_buffer(self, samples):
        """Allocate one contiguous fit input; builder batches will fill its sample slices."""
        raw, read, read_log, own = (
            np.empty((samples, *array.shape[1:]), array.dtype)
            for array in (self.raw, self._read, self._read_log, self.initial)
        )
        operator = None
        if self.operator is not None:
            tape, exponent, entry = self.operator
            operator = (tape, exponent, np.empty((samples, entry.shape[1])))
        shift = None if self.read_shift is None else np.empty_like(read_log)
        bypass = None if self.bypass is None else np.empty(samples, dtype=np.bool_)
        return ReadEmissions(raw, read, read_log, own, self.minor, read_shift=shift, operator=operator, bypass=bypass)

    def copy_fit_slice(self, batch, start, stop):
        """Fill a range of the fit buffer and return its view for a possible mixed-width fit."""
        rows = slice(start, stop)
        raw, read, log, own = self.raw[rows], self._read[rows], self._read_log[rows], self.initial[rows]
        raw[:] = batch.raw
        read[:] = batch._read
        log[:] = batch._read_log
        own[:] = batch.initial
        operator = None
        if self.operator is not None:
            tape, exponent, entry = self.operator
            initial = entry[rows]
            initial[:] = batch.operator[2]
            operator = (tape, exponent, initial)
        shift = None
        if self.read_shift is not None:
            shift = self.read_shift[rows]
            shift[:] = batch.read_shift
        bypass = None if self.bypass is None else self.bypass[rows]
        if bypass is not None:
            bypass[:] = batch.bypass
        return ReadEmissions(raw, read, log, own, self.minor, read_shift=shift, operator=operator, bypass=bypass)

    def chain_inputs(self):
        """The bin action's storage and the track-oriented chain's full class axis."""
        if self.operator is None:
            tape, exponent, initial = empty_tape(), np.empty((0, 0)), self.initial
        else:
            tape, exponent, initial = self.operator
        own = swap_classes(self.minor)
        perm = np.concatenate((own, np.arange(len(own), initial.shape[1])))
        return self._read, self._read_log, np.ascontiguousarray(initial), perm, tape, exponent

    @property
    def shape(self):
        return self.raw.shape


def _maps(tracks, n_labels):
    emission_label = np.where(tracks.fidelity == 1, n_labels - 1, tracks.labels).astype(np.int64)
    unordered, swap = track_pair_maps(tracks.labels, emission_label, n_labels)
    return unordered, swap


def chain_statistics(emissions: ReadEmissions, tracks, capable, theta, a, b, eligible, *, want_masses=False,
                     emission_masses=False, label_masses=False, bin_likelihood=False):
    """fidelity.chain_statistics with the read classes carried (same outputs)."""
    capable = np.ascontiguousarray(capable, dtype=np.int64)
    n_labels = len(capable)
    unordered, swap = _maps(tracks, n_labels)
    read, logs, entry, perm, tape, exponent = emissions.chain_inputs()
    result = read_chain_kernel(
        np.ascontiguousarray(structured_emissions(emissions.raw, tracks)),
        read, logs, unordered, swap, perm, entry,
        tracks.labels, tracks.fidelity, tracks.error_of, capable, initial_log(tracks, capable, a, b),
        np.require(theta, dtype=np.float64, requirements=("C", "W")), float(a), float(b),
        np.ascontiguousarray(eligible, dtype=np.bool_), bool(want_masses), bool(emission_masses), bool(label_masses),
        bool(bin_likelihood), tape=tape, exponent=exponent,
    )
    return result if bin_likelihood else result[:11]


def label_paths(emissions: ReadEmissions, model: ChainModel, capable, *, reverse: bool, tie: float = VITERBI_TIE):
    return _label_paths(emissions, model, capable, reverse, tie, both=False)


def _label_paths(emissions, model, capable, reverse, tie, *, both):
    """Decode with the retained-history path search and the read state summed along each retained path."""
    tracks = track_states(capable)
    capable = np.ascontiguousarray(capable, dtype=np.int64)
    unordered, swap = _maps(tracks, len(capable))
    # the read matrices in their own bin order in both conventions (the kernel reads the mirrored bin)
    read, logs, entry, perm, tape, exponent = emissions.chain_inputs()
    if both or not reverse:
        expanded = structured_emissions(emissions.raw, tracks)
        theta = model.theta
    else:
        expanded = structured_emissions(emissions.raw, tracks, reverse=True)
        theta = model.theta[::-1]
    states = read_viterbi_kernel(
        np.ascontiguousarray(expanded), np.ascontiguousarray(read), np.ascontiguousarray(logs), unordered, swap,
        perm, entry, tracks.labels, tracks.fidelity,
        tracks.error_of, capable, initial_log(tracks, capable, model.entry, model.exit),
        np.require(theta, dtype=np.float64, requirements=("C", "W")), float(model.entry), float(model.exit), float(tie), bool(reverse), tape=tape, exponent=exponent,
        own_width=emissions.initial.shape[1], both=both, bypass=emissions.bypass,
    )
    labels = np.ascontiguousarray(tracks.labels, dtype=np.int64)
    if both:
        return (_track_labels(states[0], labels, False), _track_labels(states[1], labels, True))
    return _track_labels(states, labels, bool(reverse))


def decode_candidates(emissions: ReadEmissions, model, capable, local_keys, named_alleles, observed, bin_starts,
                      bin_stops, n_classes, *, reverse: bool, tie: float = VITERBI_TIE):
    """fidelity.decode_candidates through the read-state Viterbi of both conventions."""
    decoded = []
    paths_by_direction = _label_paths(emissions, model, capable, reverse, tie, both=True)
    for convention in (bool(reverse), not bool(reverse)):
        paths = hmm.parsimonious_track_order(paths_by_direction[convention], local_keys, n_classes)
        masks = hmm.switch_interval_masks(paths, named_alleles, observed, bin_starts, bin_stops, n_classes)
        decoded.append((paths, masks))
    (paths, primary), (other_paths, secondary) = decoded
    counts = np.minimum(hmm.minimum_switch_counts(paths), hmm.minimum_switch_counts(other_paths))
    masks = hmm.combine_candidate_masks(primary, secondary)
    hmm.apply_track_order(hmm.canonical_track_order(masks, local_keys, n_classes), paths, masks)
    return paths, masks, counts, other_paths


def release_posteriors(emissions: ReadEmissions, model, capable, paths, other_paths, public_ids, local_keys, *,
                       errors: bool = False):
    """fidelity.release_posteriors from the read-state forward-backward."""
    tracks = track_states(capable)
    *_, masses = chain_statistics(
        emissions, tracks, capable, model.theta, model.entry, model.exit,
        np.ones(emissions.raw.shape[0], dtype=np.bool_), want_masses=True, label_masses=bool(errors),
    )
    width = len(capable) * len(capable)
    qf, qaf = masses_from_pairs(
        np.ascontiguousarray(masses[:, :width]),
        np.left_shift(np.int64(1), paths.astype(np.int64)),
        np.left_shift(np.int64(1), other_paths.astype(np.int64)),
        np.ascontiguousarray(public_ids), np.ascontiguousarray(local_keys),
    )
    if not errors:
        return qf, qaf
    u, v = label_errors(np.ascontiguousarray(masses[:, width:]), np.ascontiguousarray(paths, dtype=np.int32),
                        np.ascontiguousarray(local_keys, dtype=np.int32))
    return qf, qaf, u, v
