"""Painting's original, all-sample evidence, read in bin windows from checked record chunks.

Emissions keep their original per-bin summation order. Only their construction is tiled: the HMM and all
cross-sample reductions see the same complete arrays. Observed/informative boolean masks survive gathering;
float64 likelihood windows do not. Decoded-array callers use the same kernels directly.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numba
import numpy as np

from ..evidence import records
from ..store.slices import WindowArray
from . import hmm

# Decoded likelihoods + observed per window; at least one whole bin, plus masks/emissions and one sparse row.
WINDOW_BYTES = 64 << 20


@dataclass(frozen=True)
class RecordEvidence:
    """All-sample raw likelihoods and observations while read_evidence's record reader is open."""

    contig: str
    sample_ids: tuple[str, ...]
    likelihoods: WindowArray
    observed: WindowArray


@contextmanager
def read_evidence(file):
    """Painting's raw record view. Founder exclusions and discovery masks are deliberately not applied."""
    metadata = file.get(records.EVIDENCE, items=("contig", "sample_ids"))
    with file.open_arrays(records.EVIDENCE, ("likelihoods", "observed")) as arrays:
        yield RecordEvidence(metadata["contig"], tuple(metadata["sample_ids"]),
                             arrays["likelihoods"], arrays["observed"])


def tempered_emissions(likelihoods, observed, sites, samples, alt_probability, background_frequency,
                       bin_starts, bin_stops, config, *, parts=None):
    """(tempered float64[n, S*S, bins/parts], observed bool[n,L], informative bool[n,L]).

    ``parts`` optionally supplies (starts, stops, bin indices) for label-continuity parts. Every part uses its
    whole bin's design effect. Windows end only at whole-bin boundaries, including the last short bin.
    """
    starts, stops = (bin_starts, bin_stops) if parts is None else parts[:2]

    def calculate(values, flags, first_site, first_bin, last_bin, first_part, last_part):
        local_starts = starts[first_part:last_part] - first_site
        local_stops = stops[first_part:last_part] - first_site
        last_site = first_site + values.shape[1]
        raw = hmm.binned_emissions(values, flags, alt_probability[:, first_site:last_site],
                                   background_frequency[first_site:last_site], local_starts, local_stops,
                                   robustness_epsilon=config.robustness_epsilon, log_floor=config.emission_log_floor)
        design = hmm.design_effect(values, flags, bin_starts[first_bin:last_bin] - first_site,
                                   config.emission_site_correlation)
        if parts is not None:
            design = design[:, parts[2][first_part:last_part] - first_bin]
        raw /= design[:, None, :]
        return raw, hmm.informative_sites(values, flags)

    if isinstance(likelihoods, np.ndarray):
        values, flags = hmm.gather_sites(likelihoods, observed, sites, samples)
        raw, informative = calculate(values, flags, 0, 0, len(bin_starts), 0, len(starts))
        return raw, flags, informative

    start, stop, step = samples.indices(likelihoods.shape[0])
    if step != 1:
        raise ValueError("painting batches require contiguous sample slices")
    n_samples = stop - start
    n_states = alt_probability.shape[0] + 1
    raw = np.empty((n_samples, n_states * n_states, len(starts)), dtype=np.float64)
    flags = np.empty((n_samples, len(sites)), dtype=np.bool_)
    informative = np.empty_like(flags)
    # Locality in the sample-major record: finish a small sample tile's windows before advancing across chunks.
    sample_tile = max(1, numba.get_num_threads())
    largest_bin = int(np.max(bin_stops - bin_starts))
    for a in range(0, n_samples, sample_tile):
        b = min(n_samples, a + sample_tile)
        window_bins = max(1, WINDOW_BYTES // max(1, 25 * (b - a) * largest_bin))
        for first_bin in range(0, len(bin_starts), window_bins):
            last_bin = min(len(bin_starts), first_bin + window_bins)
            left, right = int(bin_starts[first_bin]), int(bin_stops[last_bin - 1])
            first_part, last_part = (first_bin, last_bin) if parts is None else (
                int(np.searchsorted(parts[2], first_bin)), int(np.searchsorted(parts[2], last_bin)))
            values = likelihoods.gather(slice(start + a, start + b), sites[left:right])
            seen = observed.gather(slice(start + a, start + b), sites[left:right])
            emissions, info = calculate(values, seen, left, first_bin, last_bin, first_part, last_part)
            raw[a:b, :, first_part:last_part] = emissions
            flags[a:b, left:right] = seen
            informative[a:b, left:right] = info
            del values, seen, emissions, info
    return raw, flags, informative
