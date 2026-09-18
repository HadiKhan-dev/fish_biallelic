"""Exact panel scores from existing block emissions, without a repainted tensor."""
import numpy as np
from numba import njit, prange
from numba.typed import List

@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def score_emissions(emissions, local_indices, penalty, samples):
    k = local_indices.shape[1]
    pairs = k * k
    result = np.empty(samples, dtype=np.float64)
    for sample in prange(samples):
        scores = np.empty(pairs, dtype=np.float64)
        initialized = False
        for block in range(len(emissions)):
            values = emissions[block]
            for marker in range(values.shape[3]):
                if not initialized:
                    for pair in range(pairs):
                        scores[pair] = values[sample, local_indices[block, pair // k],
                                              local_indices[block, pair % k], marker]
                    initialized = True
                    continue
                best = -np.inf
                for pair in range(pairs):
                    if scores[pair] > best:
                        best = scores[pair]
                switch_base = best - penalty
                for pair in range(pairs):
                    emission = values[sample, local_indices[block, pair // k],
                                      local_indices[block, pair % k], marker]
                    stay = scores[pair]
                    if stay > switch_base:
                        scores[pair] = stay + emission
                    else:
                        scores[pair] = switch_base + emission
        best = -np.inf
        for pair in range(pairs):
            if scores[pair] > best:
                best = scores[pair]
        result[sample] = best
    return result

def score_panel(paths, sub, penalty, samples):
    arrays = List()
    indices = np.empty((len(sub), len(paths)), dtype=np.int64)
    for block, item in enumerate(sub):
        arrays.append(item["bin_emissions"])
        lookup = item.get("key_to_local_idx")
        if lookup is None:
            lookup = {key: index for index, key in enumerate(item["hap_keys"])}
        for path, keys in enumerate(paths):
            indices[block, path] = lookup[keys[block]]
    return score_emissions(arrays, indices, float(penalty), samples)
