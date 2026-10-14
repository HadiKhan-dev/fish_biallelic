"""Exact panel scores from existing block emissions, without a repainted tensor."""
import numpy as np
from numba import njit, prange

@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def score_emissions(flat, block_start, block_haps, block_bins, local_indices, penalty, samples):
    """Each sample's join score (emissions: a chimera_scoring.EmissionLayout's arrays)."""
    k = local_indices.shape[1]
    # Unphased emissions and a uniform off-diagonal switch penalty are
    # exchange-symmetric. Max scores need one copy of each unordered state;
    # traceback remains unchanged in the separate painting kernel. Each
    # block's state emissions are first copied into one array, contiguous
    # per bin (the same values in the same recursion).
    pairs = k * (k + 1) // 2
    first = np.empty(pairs, np.int64)
    second = np.empty(pairs, np.int64)
    index = 0
    for a in range(k):
        for b in range(a, k):
            first[index], second[index] = a, b
            index += 1
    widest = 1
    for block in range(len(block_start)):
        widest = max(widest, block_bins[block])
    result = np.empty(samples, dtype=np.float64)
    for sample in prange(samples):
        scores = np.empty(pairs, dtype=np.float64)
        gathered = np.empty((widest, pairs))
        initialized = False
        for block in range(len(block_start)):
            haps, bins = block_haps[block], block_bins[block]
            base = block_start[block] + sample * haps * haps * bins
            for pair in range(pairs):
                at = base + (local_indices[block, first[pair]] * haps
                             + local_indices[block, second[pair]]) * bins
                for marker in range(bins):
                    gathered[marker, pair] = flat[at + marker]
            for marker in range(bins):
                if not initialized:
                    for pair in range(pairs):
                        scores[pair] = gathered[marker, pair]
                    initialized = True
                    continue
                best = -np.inf
                for pair in range(pairs):
                    if scores[pair] > best:
                        best = scores[pair]
                switch_base = best - penalty
                for pair in range(pairs):
                    stay = scores[pair]
                    scores[pair] = (stay if stay > switch_base else switch_base) + gathered[marker, pair]
        best = -np.inf
        for pair in range(pairs):
            if scores[pair] > best:
                best = scores[pair]
        result[sample] = best
    return result

def panel_indices(paths, sub):
    """Each path's local row per block (blocks x paths)."""
    indices = np.empty((len(sub), len(paths)), dtype=np.int64)
    for block, item in enumerate(sub):
        lookup = item.get("key_to_local_idx")
        if lookup is None:
            lookup = {key: index for index, key in enumerate(item["hap_keys"])}
        for path, keys in enumerate(paths):
            indices[block, path] = lookup[keys[block]]
    return indices


def score_panel(paths, sub, penalty, samples, layout):
    """Per-sample join scores of a panel (layout: the EmissionLayout of sub)."""
    return score_emissions(*layout.arrays(), panel_indices(paths, sub), float(penalty), samples)


@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def paint_emissions(flat, block_start, block_haps, block_bins, local_indices, penalty, start, stop):
    """Viterbi painting of samples start..stop-1 (the join objective's traceback), reported per
    bin as the ordered state a * k + b of panel paths a <= b (emissions: an EmissionLayout's
    arrays).

    Unphased emissions and the uniform switch penalty make the ordered states (a, b) and (b, a)
    carry equal scores at every bin, so the ordered-pair programme's first maximiser and every
    back pointer it follows are pairs with a <= b (the first of each symmetric pair in state
    order); this programme runs over those pairs alone, in the same order, with the same
    operations and tie rules (first maximiser; stay when stay >= switch).
    """
    k = local_indices.shape[1]
    pairs = k * (k + 1) // 2
    first = np.empty(pairs, np.int64)
    second = np.empty(pairs, np.int64)
    code = np.empty(pairs, np.int32)
    index = 0
    for a in range(k):
        for b in range(a, k):
            first[index], second[index], code[index] = a, b, a * k + b
            index += 1
    total = 0
    widest = 1
    for block in range(len(block_start)):
        total += block_bins[block]
        widest = max(widest, block_bins[block])
    sample_paths = np.zeros((stop - start, total), dtype=np.int32)
    for position in prange(stop - start):
        sample = start + position
        current = np.empty(pairs, dtype=np.float64)
        new_scores = np.empty(pairs, dtype=np.float64)
        backptrs = np.zeros((total, pairs), dtype=np.int32)
        gathered = np.empty((widest, pairs))
        t = 0
        for block in range(len(block_start)):
            haps, bins = block_haps[block], block_bins[block]
            base = block_start[block] + sample * haps * haps * bins
            for pair in range(pairs):
                at = base + (local_indices[block, first[pair]] * haps
                             + local_indices[block, second[pair]]) * bins
                for marker in range(bins):
                    gathered[marker, pair] = flat[at + marker]
            for marker in range(bins):
                if t == 0:
                    for pair in range(pairs):
                        current[pair] = gathered[marker, pair]
                else:
                    best_prev = -np.inf
                    best_prev_idx = 0
                    for pair in range(pairs):
                        if current[pair] > best_prev:
                            best_prev = current[pair]
                            best_prev_idx = pair
                    switch_base = best_prev - penalty
                    for pair in range(pairs):
                        emission = gathered[marker, pair]
                        stay = current[pair] + emission
                        switch = switch_base + emission
                        if stay >= switch:
                            new_scores[pair] = stay
                            backptrs[t, pair] = pair
                        else:
                            new_scores[pair] = switch
                            backptrs[t, pair] = best_prev_idx
                    for pair in range(pairs):
                        current[pair] = new_scores[pair]
                t += 1
        best_final = -np.inf
        state = 0
        for pair in range(pairs):
            if current[pair] > best_final:
                best_final = current[pair]
                state = pair
        sample_paths[position, total - 1] = code[state]
        for t in range(total - 1, 0, -1):
            state = backptrs[t, state]
            sample_paths[position, t - 1] = code[state]
    return sample_paths
