"""Kernel declarations for the birth-order prior's exact Gibbs sampler (science/pedigree/order.py).

order.adjust hands every kernel owned, C-contiguous, writable work arrays (it copies its inputs), so one overload
per kernel covers every call.
"""

from numba import types

_i1 = types.Array(types.int64, 1, "C")
_i2 = types.Array(types.int64, 2, "C")
_b1 = types.Array(types.boolean, 1, "C")
_f1 = types.Array(types.float64, 1, "C")
_f2 = types.Array(types.float64, 2, "C")
_n, _x, _q = types.int64, types.float64, types.boolean

OVERLOADS = {
    "order._index": ((_i2, _i1, _f1, _b1, _n, _x),),
    "order._allowed": ((_i2, _b1, _i1, _n, _n),),
    "order._exact_z": ((_i2, _i1, _b1, _f1, _i1, _f1),),
    "order._chain": ((_i2, _i1, _b1, _f1, _i1, _i1, _i1, _f1, _i1, _i1, _i2, _f2, _n, _x, _q),),
}
