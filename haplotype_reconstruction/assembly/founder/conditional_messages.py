"""Reuse numerical flank messages under founder-label permutations.

Only state axes are permuted. Tracebacks, tie ordering and switch counts are
not reused. The cache belongs to one immutable evidence-model workspace.
"""
from threading import Lock
import numpy as np
from numba import njit, prange
from .packing import PreparedModels

@njit(cache=True, parallel=True, nogil=True)
def reorder(messages, mapping):
    result = np.empty_like(messages)
    blocks, samples, states = messages.shape
    for task in prange(blocks * samples):
        block, sample = task // samples, task % samples
        for state in range(states):
            result[block, sample, state] = messages[block, sample, mapping[state]]
    return result


class SharedSuffix:
    def __init__(self, native):
        self.native = native
        self.creation_lock = Lock()
        self.hits = 0
        self.misses = 0

    def __call__(self, models, known, incumbent, penalty, reverse, first, second):
        if not isinstance(models, PreparedModels):
            return self.native(models, known, incumbent, penalty, reverse, first, second)
        # Lifetime is exactly that of this immutable model workspace. A single
        # panel (two directions) is retained, not a chromosome-history cache.
        with self.creation_lock:
            if not hasattr(models, '_shared_suffix'):
                models._shared_suffix = dict(lock=Lock(), key=None, values={})
        cache = models._shared_suffix
        panel = np.concatenate((known, incumbent[None, :]), axis=0)
        order = sorted(range(len(panel)), key=lambda i: panel[i].tobytes())
        canonical = np.ascontiguousarray(panel[order])
        key = (panel.shape, float(penalty), canonical.tobytes())
        inverse = np.empty(len(order), np.int64)
        inverse[order] = np.arange(len(order))
        lookup = np.empty((len(panel), len(panel)), np.int64)
        left, right = np.triu_indices(len(panel))
        lookup[left, right] = lookup[right, left] = np.arange(len(left))
        mapping = lookup[inverse[first], inverse[second]]
        # One focal worker builds a missing direction; peers share it after
        # the native call. They retain their normal dynamic thread budgets.
        with cache['lock']:
            if cache['key'] != key:
                cache['key'], cache['values'] = key, {}
            if bool(reverse) not in cache['values']:
                cache['values'][bool(reverse)] = self.native(models, canonical[:-1],
                    canonical[-1], penalty, reverse, left, right)
                self.misses += 1
            else:
                self.hits += 1
            messages = cache['values'][bool(reverse)]
        if np.array_equal(mapping, np.arange(len(mapping))):
            return messages
        return reorder(messages, mapping)

# One registry lock; actual message caches remain workspace-local.
_shared = SharedSuffix(lambda *args: _native_suffix(*args))


def _native_suffix(models, known, incumbent, penalty, reverse, first, second):
    from .path_search import _packed_incumbent_suffix
    from .packing import packed_emissions
    return _packed_incumbent_suffix(*packed_emissions(models), known, incumbent,
        penalty, reverse, first, second)


def incumbent_suffix(*args):
    return _shared(*args)
