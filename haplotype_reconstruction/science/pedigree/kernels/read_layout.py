"""Argument layouts shared by the read-carried pedigree kernels."""

from numba import types

# Chain declarations share the read arguments that follow the scalar model inputs.
# A class-transfer cache is an owned float64 [n, B, S, S, C, C] array or None.
_F32_3 = types.Array(types.float32, 3, "C")
_F64_1 = types.Array(types.float64, 1, "C")
_F64_2 = types.Array(types.float64, 2, "C")
_F64_3 = types.Array(types.float64, 3, "C")
_F64_4 = types.Array(types.float64, 4, "C")
_I64_1 = types.Array(types.int64, 1, "C")
_I64_2 = types.Array(types.int64, 2, "C")
_BOOL_2 = types.Array(types.boolean, 2, "C")
_CLASS_CACHE = types.Array(types.float64, 6, "C")
_READ_TERMS = (_I64_1, _F64_2, _F32_3, _F32_3)
_CACHE_ARGUMENT = (types.none, _CLASS_CACHE)

# An exact reset flag per read slot and the live widths at each bin boundary.
_reset = types.Array(types.boolean, 1, "C")
_width = types.Array(types.int64, 2, "C")
_READ_TAILS = tuple((*_READ_TERMS, cache) for cache in _CACHE_ARGUMENT)


def extend(signatures, *tails):
    return tuple(dict.fromkeys(tuple(signature) + tuple(tail) for signature in signatures for tail in tails))
