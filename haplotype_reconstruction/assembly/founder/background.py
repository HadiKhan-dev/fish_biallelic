"""Immutable background summaries shared by a fixed panel's focal searches.

For each bin interval the best diploid state excludes a focal founder unless
the overall best state contains it. Only the one/two endpoints of that best
state need another scan. Thus all K exclusion maxima cost O(K²), not O(K³).
Messages and candidate-specific states are not shared. The owner discards this
cache when its panel/models change; insufficient RAM uses the direct kernels.
"""
import numpy as np
from numba import njit, prange
from numba.typed import List


@njit(cache=True, parallel=True, nogil=True)
def prepare(emissions, selected, reverse, backward, first, second):
    samples, founders = emissions[0].shape[0], len(selected)
    result = List()
    for emission in emissions:
        bins = emission.shape[3]
        # Empty allocations avoid a separate parallel fill for every block.
        # Each sample/block task initializes its own slice below.
        result.append((np.empty((samples, len(first), bins + 1)),
                       np.empty((samples, len(first))),
                       np.empty((samples, founders, bins + 1, bins + 1))))
    for task in prange(samples * len(emissions)):
        block, sample = task // samples, task % samples
        emission = emissions[block]
        prefix, shift, segments = result[block]
        bins = emission.shape[3]
        prefix[sample] = 0.0
        shift[sample] = 0.0
        segments[sample] = -np.inf
        initial = bins - 1 if reverse else 0
        for state in range(len(first)):
            a, b = selected[first[state], block], selected[second[state], block]
            if backward:
                shift[sample, state] = emission[sample, a, b, initial]
            for step in range(bins):
                offset = step + int(backward)
                value = 0.0
                if offset < bins:
                    site = bins - 1 - offset if reverse else offset
                    value = emission[sample, a, b, site]
                prefix[sample, state, step + 1] = prefix[sample, state, step] + value
        for start in range(bins):
            for stop in range(start + 1, bins + 1):
                best, winner = -np.inf, 0
                for state in range(len(first)):
                    value = prefix[sample, state, stop] - prefix[sample, state, start]
                    if value > best:
                        best, winner = value, state
                for focal in range(founders):
                    segments[sample, focal, start, stop] = best
                for endpoint in range(2):
                    focal = first[winner] if endpoint == 0 else second[winner]
                    if endpoint == 1 and first[winner] == focal:
                        continue
                    value = -np.inf
                    for state in range(len(first)):
                        if first[state] != focal and second[state] != focal:
                            value = max(value, prefix[sample, state, stop] - prefix[sample, state, start])
                    segments[sample, focal, start, stop] = value
    return result


class SharedBackground:
    """One immutable, memory-budgeted cache per competing fixed-panel batch."""
    def __init__(self, models, selected, *, backward=True):
        self.frames = {}
        self.founders = len(selected)
        self.first, self.second = (np.asarray(x, np.int64)
                                   for x in np.triu_indices(self.founders))
        self.lookup = np.empty((self.founders, self.founders), np.int64)
        self.lookup[self.first, self.second] = self.lookup[self.second, self.first] = np.arange(len(self.first))
        samples, states = models[0]["bin_emissions"].shape[0], len(self.first)
        sizes = [m["bin_emissions"].shape[3] + 1 for m in models]
        geometries = [(reverse, back) for back in ((False, True) if backward else (False,))
                      for reverse in (False, True)]
        required = 8 * samples * len(geometries) * (
            states * (sum(sizes) + len(sizes)) + self.founders * sum(t * t for t in sizes))
        from ...painting.model import available_process_memory_bytes
        available = available_process_memory_bytes()
        # Keep seven eighths of the remaining allowance for evidence, messages,
        # parallel candidates and traceback. No scientific fallback is used.
        if available is not None and required > available // 8:
            return
        emissions = List([m["bin_emissions"] for m in models])
        for reverse, back in geometries:
            self.frames[reverse, back] = prepare(
                emissions, selected, reverse, back, self.first, self.second)

    def view(self, focal, flipped=False):
        if not self.frames:
            return None
        order = np.r_[np.arange(focal), np.arange(focal + 1, self.founders), focal]
        mapping = np.ascontiguousarray(self.lookup[order[self.first], order[self.second]])
        return BackgroundView(self, focal, mapping, flipped)


class BackgroundView:
    def __init__(self, owner, focal, mapping, flipped=False):
        self.owner, self.focal, self.mapping = owner, focal, mapping
        self.flipped, self.views = flipped, {}

    def get(self, reverse, backward):
        key = bool(reverse), bool(backward)
        if key not in self.views:
            frames = self.owner.frames[bool(reverse) ^ self.flipped, bool(backward)]
            self.views[key] = List(list(frames)[::-1]) if self.flipped else frames
        return self.views[key]

    def reversed(self):
        return BackgroundView(self.owner, self.focal, self.mapping, not self.flipped)
