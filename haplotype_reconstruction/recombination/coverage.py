"""Per-sample observable coverage: painted site ranges minus long read-free tracts, and decoder run ids.

Final calls inside a long run of sites without reads are imputed from the painting, not observed. Removing the
run from that sample's coverage treats it as a gap: no observation, independent restarts on both sides, so no
exposure and no crossover inside or across it for any meiosis involving the sample, in both the
shared-orientation fit and the per-meiosis decoder.
"""
from __future__ import annotations

import numpy as np
from numba import njit


def read_free_tracts(observed, minimum_sites):
    """Per sample, int64 (n, 2) half-open site-index runs of at least ``minimum_sites`` zero-depth sites."""
    tracts = []
    for row in np.asarray(observed, dtype=bool):
        change = np.flatnonzero(np.diff(np.r_[0, (~row).view(np.int8), 0]))
        starts, stops = change[0::2], change[1::2]
        keep = stops - starts >= minimum_sites
        tracts.append(np.stack((starts[keep], stops[keep]), axis=1))
    return tracts


def observed_coverage(coverage, offsets, observed, minimum_read_free_sites):
    """(coverage int64 [R, 2], offsets int64 [N+1], read-free tracts, read-free sample-sites).

    ``coverage``/``offsets`` are the family-phase product's painted site ranges. Without a read-depth mask
    (``observed`` None) or with ``minimum_read_free_sites`` 0 the coverage is unchanged.
    """
    if observed is None or minimum_read_free_sites <= 0:
        return coverage, offsets, 0, 0
    removed = read_free_tracts(observed, minimum_read_free_sites)
    packed, packed_offsets = [], [0]
    for sample, cuts in enumerate(removed):
        for start, stop in coverage[offsets[sample]:offsets[sample + 1]]:
            for cut_start, cut_stop in cuts:
                if cut_stop <= start or cut_start >= stop:
                    continue
                if cut_start > start:
                    packed.append((start, cut_start))
                start = max(start, cut_stop)
                if start >= stop:
                    break
            if start < stop:
                packed.append((start, stop))
        packed_offsets.append(len(packed))
    return (np.asarray(packed, dtype=np.int64).reshape(-1, 2), np.asarray(packed_offsets, dtype=np.int64),
            sum(len(cuts) for cuts in removed),
            int(sum(int((cuts[:, 1] - cuts[:, 0]).sum()) for cuts in removed)))


@njit(cache=True)
def sample_runs(samples, sites, coverage, offsets, components, positions, maximum_gap):
    """int32 [N, L] run id of every covered site (-1 uncovered); a run restarts at every coverage range,
    component (segment) change and gap longer than ``maximum_gap`` bp."""
    runs = np.full((samples, sites), -1, dtype=np.int32)
    for sample in range(samples):
        run = 0
        for region in range(offsets[sample], offsets[sample + 1]):
            start, stop = coverage[region]
            previous = -1
            for j in range(start, stop):
                if components[j] < 0:
                    previous = -1
                    continue
                if (previous < 0 or components[j] != components[previous]
                        or positions[j] - positions[previous] > maximum_gap):
                    run += 1
                runs[sample, j] = run
                previous = j
    return runs
