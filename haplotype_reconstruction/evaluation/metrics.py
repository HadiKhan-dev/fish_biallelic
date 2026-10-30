"""Phase metrics of the released family phase against the simulated truth (evaluation only; release 3.2).

Per sample, a released phase set is a maximal run of the sample's sites inside one painting component that none of the
sample's released phase breaks separates (family phase's per-sample ``phase_breaks``: boundary i lies between sites
i and i + 1; sites outside every component belong to no set). Phase is scored only at the sample's scored sites:
correctly called true heterozygotes (both alleles called, called genotype equal to the true genotype, true genotype
heterozygous) that are released phased (inside a component, displayed phase not undetermined: frame_undetermined
bit 1 clear). A scored site's orientation is flipped when its first released allele differs from the first (maternal)
true allele.

- ``set_aligned_errors`` (sites): per set, the scored sites on its minority orientation (one strand alignment per set);
- ``switch_errors`` (events): orientation changes between consecutive scored sites of a set;
- ``long_switches`` (events): within a set, the orientation runs (maximal stretches of consecutive scored sites of one
  orientation) of at least LONG_SWITCH_MIN_RUN sites are kept in order, and each orientation change between
  consecutive kept runs is one long switch (a short opposite run between two long runs of one orientation is none);
- ``phase_sets``: sets holding at least one scored site; a set's span (bp) runs from its first to its last scored
  site, and the N50 is taken over these spans;
- ``genotype_errors``: called genotypes (both alleles called) that differ from the true genotype, at every site. They
  are reported separately and never enter the phase counts;
- ``phased_true_heterozygotes`` / ``correct_true_heterozygotes``: scored sites over correctly called true
  heterozygotes at every site (the phased share).
Released (truth-free) sets, what a reader of the exported phase sets sees: sets holding at least one called
heterozygote released phased, spanning first to last such site (``released_phase_sets``, their N50).

These are the definitions of the lab scorer the releases since 3.0 were judged with. Samples are scored one at a time;
every count is an integer, so results do not depend on the thread count.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# Long switches: both orientation runs around the change hold at least this many scored sites.
LONG_SWITCH_MIN_RUN = 20
# Per-sample counts returned by released_phase_counts, in this column order.
SAMPLE_COLUMNS = ("called_genotypes", "genotype_errors", "correct_true_heterozygotes", "phased_true_heterozygotes",
                  "phase_sets", "set_aligned_errors", "switch_errors", "long_switches", "released_phase_sets")


@njit(cache=True)
def _sample_phase(calls, truth, set_ids, undetermined, positions, minimum_run, spans, released_spans):
    """One sample's released-phase counts (module docstring).

    calls, truth int8 [L, 2]; set_ids int64 [L] (phase_set_ids: the sample's released set of each site, -1 outside
    components); undetermined bool [L] (displayed phase undetermined); positions int64 [L]; minimum_run int (long
    switches); spans, released_spans int64 [L] output buffers. Returns int64 [9] in SAMPLE_COLUMNS order; the first
    ``phase_sets`` entries of ``spans`` and the first ``released_phase_sets`` of ``released_spans`` are written.
    """
    out = np.zeros(9, dtype=np.int64)
    # The open scored set: its id (-1: none), scored sites, flipped sites, first/last scored position, the previous
    # scored site's orientation, the current orientation run (orientation, length) and the last kept long run's
    # orientation (-1: none yet in this set).
    current = -1
    scored = flipped = 0
    first = last = 0
    previous = 0
    run_flip = run_length = 0
    kept = -1
    # The open released set: id (-1: none), first/last phased called heterozygote position.
    released = -1
    released_first = released_last = 0
    for site in range(calls.shape[0] + 1):
        end = site == calls.shape[0]
        a = b = x = y = 0
        set_id = -1
        if not end:
            a, b = calls[site, 0], calls[site, 1]
            x, y = truth[site, 0], truth[site, 1]
            set_id = set_ids[site]
        called = (not end) and a >= 0 and b >= 0
        correct = called and a + b == x + y
        if called:
            out[0] += 1
            out[1] += int(not correct)
        phased = called and set_id >= 0 and a != b and not undetermined[site]
        if released >= 0 and (end or (phased and set_id != released)):
            released_spans[out[8]] = released_last - released_first
            out[8] += 1
            released = -1
        if phased:
            if released < 0:
                released = set_id
                released_first = positions[site]
            released_last = positions[site]
        score = correct and x != y
        if score:
            out[2] += 1
        score = score and phased
        if current >= 0 and (end or (score and set_id != current)):
            # Close the open set: its last run, then the set's counts and span.
            if run_length >= minimum_run:
                out[7] += int(kept >= 0 and kept != run_flip)
            out[4] += 1
            out[5] += min(flipped, scored - flipped)
            spans[out[4] - 1] = last - first
            current = -1
        if not score:
            continue
        flip = int(a != x)
        out[3] += 1
        if current < 0:
            current = set_id
            scored = flipped = 0
            first = positions[site]
            run_flip, run_length, kept = flip, 0, -1
        elif flip != previous:
            out[6] += 1
            if run_length >= minimum_run:
                out[7] += int(kept >= 0 and kept != run_flip)
                kept = run_flip
            run_flip, run_length = flip, 0
        run_length += 1
        scored += 1
        flipped += flip
        last = positions[site]
        previous = flip
    return out


def phase_set_ids(components, boundaries):
    """int64 [L]: a sample's released phase set of every site (-1 outside components): a new set starts at each
    component change and after each released boundary (marker i separates sites i and i + 1)."""
    components = np.asarray(components)
    cut = np.zeros(len(components), dtype=np.bool_)
    cut[1:] = components[1:] != components[:-1]
    indices = np.asarray(boundaries, dtype=np.int64) + 1
    cut[indices[(indices > 0) & (indices < len(components))]] = True
    return np.where(components >= 0, np.cumsum(cut), -1).astype(np.int64)


def released_phase_counts(calls, truth, components, undetermined, positions, phase_breaks=None, *,
                          minimum_run=LONG_SWITCH_MIN_RUN):
    """Per-sample released-phase counts of one chromosome (module docstring).

    ``calls`` int8 [N, L, 2] (released alleles, -1 unknown); ``truth`` int8 [N, L, 2] (maternal, paternal);
    ``components`` int [L] (-1 between components); ``undetermined`` bool [N, L] or None (none undetermined);
    ``positions`` int64 [L]; ``phase_breaks``: one array of released marker boundaries per sample (calls' order) or
    None. Returns (counts int64 [N, len(SAMPLE_COLUMNS)], [per-sample scored-set spans], [per-sample released-set
    spans]); spans in bp.
    """
    calls, truth = np.asarray(calls), np.asarray(truth)
    samples, sites, _ = calls.shape
    if truth.shape != calls.shape:
        raise ValueError("released and true allele axes differ")
    positions = np.ascontiguousarray(positions, dtype=np.int64)
    components = np.asarray(components)
    counts = np.zeros((samples, len(SAMPLE_COLUMNS)), dtype=np.int64)
    spans, released_spans = [], []
    buffer = np.empty(sites, dtype=np.int64)
    released_buffer = np.empty(sites, dtype=np.int64)
    no_undetermined = np.zeros(sites, dtype=np.bool_)
    for sample in range(samples):
        boundaries = () if phase_breaks is None else phase_breaks[sample]
        result = _sample_phase(np.ascontiguousarray(calls[sample], dtype=np.int8),
                               np.ascontiguousarray(truth[sample], dtype=np.int8),
                               phase_set_ids(components, boundaries),
                               no_undetermined if undetermined is None else
                               np.ascontiguousarray(undetermined[sample], dtype=np.bool_),
                               positions, int(minimum_run), buffer, released_buffer)
        counts[sample] = result
        spans.append(buffer[:int(result[4])].copy())
        released_spans.append(released_buffer[:int(result[8])].copy())
    return counts, spans, released_spans


def n50(lengths) -> int:
    """The N50 of a set of lengths (the length at which the longest ones reach half the total); 0 if empty."""
    values = np.sort(np.asarray(lengths, dtype=np.int64))[::-1]
    if len(values) == 0 or values.sum() == 0:
        return 0
    cumulative = np.cumsum(values)
    return int(values[int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])
