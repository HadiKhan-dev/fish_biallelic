"""Read-state counterparts of painting.fidelity's batch wrappers (PaintingParameters.read_classes > 1).

A batch's emissions with the read state carried (ReadEmissions): the tempered label emissions of the model without
read state plus, per sample, unordered label pair and bin, the read matrix of tempered per-configuration ratios and
its log scale (painting.read_factor), and the class law at the component's first selected site. Each wrapper mirrors
fidelity's (chain_statistics, label_paths, decode_candidates, release_posteriors) through painting.read_chain's
kernels; with no read sharing they return fidelity's results. painting.fidelity and painting.run dispatch a batch here
when it carries the read state (``hasattr(raw, "read")``)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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

__all__ = ["ReadEmissions", "chain_statistics", "label_paths", "decode_candidates", "release_posteriors"]


@dataclass(frozen=True)
class ReadEmissions:
    """One sample batch's emissions with the read state (an in-memory batch, never stored)."""

    raw: np.ndarray          # float64[N, S*S, B] tempered label emissions (the model without read state)
    read: np.ndarray         # float32[N, U, B, C, C] read matrices per unordered label pair (triu order)
    read_log: np.ndarray     # float64[N, U, B] their log scales
    initial: np.ndarray      # float64[N, C] class law at the component's first selected site
    minor: int               # split-class parameter (C = 2 minor + 3)
    # when the confidence strength differs from the path's: a callable returning its (read, read_log), built only when
    # a caller asks (the decode's release posteriors; the copying-model fits and label continuity never do)
    confidence: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None
    # when the label-error strength (parameters.LABEL_ERROR_CORRELATION) differs from both run strengths: a callable
    # returning its (read, read_log), built only when the reference decode asks (at_label)
    label: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None
    # float64[N, U, B]: the bin-level tempering correction read_log carries (read_factor.read_matrices ``shift``);
    # calibration gives the hidden blocks' visible part the same correction (read_calibration)
    read_shift: np.ndarray | None = None

    def at_confidence(self, scaled_raw):
        """The batch at the confidence strength: the label emissions rescaled by the caller (emissions.confidence_scale)
        and the read matrices built at the confidence weights."""
        if self.confidence is None:
            return ReadEmissions(scaled_raw, self.read, self.read_log, self.initial, self.minor)
        read, read_log = self.confidence()
        return ReadEmissions(scaled_raw, read, read_log, self.initial, self.minor)

    def at_label(self, scaled_raw):
        """The batch at the label-error strength: the label emissions rescaled by the caller (emissions.confidence_scale
        to that strength) and the read matrices built at its weights (the path strength's when it has none: a unit
        batch, whose matrices do not depend on the strength, or a label strength equal to the path's)."""
        if self.label is None:
            return ReadEmissions(scaled_raw, self.read, self.read_log, self.initial, self.minor)
        read, read_log = self.label()
        return ReadEmissions(scaled_raw, read, read_log, self.initial, self.minor)

    def path_only(self):
        """The batch without its deferred confidence matrices (and the inputs they keep alive), for holding."""
        return ReadEmissions(self.raw, self.read, self.read_log, self.initial, self.minor)

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
    result = read_chain_kernel(
        np.ascontiguousarray(structured_emissions(emissions.raw, tracks)),
        np.ascontiguousarray(emissions.read), np.ascontiguousarray(emissions.read_log), unordered, swap,
        swap_classes(emissions.minor), np.ascontiguousarray(emissions.initial),
        tracks.labels, tracks.fidelity, tracks.error_of, capable, initial_log(tracks, capable, a, b),
        np.ascontiguousarray(theta, dtype=np.float64), float(a), float(b),
        np.ascontiguousarray(eligible, dtype=np.bool_), bool(want_masses), bool(emission_masses), bool(label_masses),
        bool(bin_likelihood),
    )
    return result if bin_likelihood else result[:11]


def label_paths(emissions: ReadEmissions, model: ChainModel, capable, *, reverse: bool, tie: float = VITERBI_TIE):
    """fidelity.label_paths with the read state summed along each path (greedy; read_chain.read_viterbi_kernel)."""
    tracks = track_states(capable)
    capable = np.ascontiguousarray(capable, dtype=np.int64)
    unordered, swap = _maps(tracks, len(capable))
    # the read matrices in their own bin order in both conventions (the kernel reads the mirrored bin)
    read, logs = emissions.read, emissions.read_log
    if not reverse:
        expanded = structured_emissions(emissions.raw, tracks)
        theta = model.theta
    else:
        expanded = structured_emissions(emissions.raw, tracks, reverse=True)
        theta = model.theta[::-1]
    states = read_viterbi_kernel(
        np.ascontiguousarray(expanded), np.ascontiguousarray(read), np.ascontiguousarray(logs), unordered, swap,
        swap_classes(emissions.minor), np.ascontiguousarray(emissions.initial), tracks.labels, tracks.fidelity,
        tracks.error_of, capable, initial_log(tracks, capable, model.entry, model.exit),
        np.ascontiguousarray(theta, dtype=np.float64), float(model.entry), float(model.exit), float(tie), bool(reverse),
    )
    return _track_labels(states, np.ascontiguousarray(tracks.labels, dtype=np.int64), bool(reverse))


def decode_candidates(emissions: ReadEmissions, model, capable, local_keys, named_alleles, observed, bin_starts,
                      bin_stops, n_classes, *, reverse: bool, tie: float = VITERBI_TIE):
    """fidelity.decode_candidates through the read-state Viterbi of both conventions."""
    decoded = []
    for convention in (bool(reverse), not bool(reverse)):
        paths = hmm.parsimonious_track_order(
            label_paths(emissions, model, capable, reverse=convention, tie=tie), local_keys, n_classes
        )
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
