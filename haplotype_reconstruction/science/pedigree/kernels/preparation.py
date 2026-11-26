"""Kernel declarations for preparation, projection and candidate-parent source laws."""

from numba import types

from ...painting.api import TAPE

_F1 = types.float64[::1]
_F2 = types.float64[:, ::1]
_F3 = types.float64[:, :, ::1]
_M5 = types.float32[:, :, :, :, ::1]
_D5 = types.float64[:, :, :, :, ::1]
_I = types.int64
_B = types.boolean
_B1 = types.boolean[::1]

OVERLOADS = {
    "prepare._informative_counts": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
        ),
    ),
    "projection._bisect_root": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.float64,
            types.int64,
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.literal(-1),
            types.float64,
            types.int64,
        ),
    ),
    "projection._branch_function": (
        (
            types.float64,
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.int64,
        ),
        (
            types.float64,
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.int64,
        ),
    ),
    "projection._bridges_kernel": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.float64,
            types.int64,
            types.boolean,
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.boolean,
            types.boolean,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.float64,
            types.int64,
            types.boolean,
            types.Omitted(None),
            types.Omitted(None),
            types.Omitted(None),
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.float64,
            types.int64,
            types.boolean,
            types.none,
            types.boolean,
            types.boolean,
        ),
    ),
    "projection._small_diagonal_product": ((types.float64, types.float64, types.float64),),
    "read_source_kernels._background_alt": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 4, "C", readonly=False, aligned=True),
            types.Array(types.float64, 5, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 4, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 4, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.float64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            _F3, _F1, _F1, _F2, _F2, _F1,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
        ),
    ),
    "read_source_kernels._backward": tuple(
        (
            types.Array(types.float64, 4, "C", readonly=False, aligned=True),
            matrix,
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            TAPE,
            _F2,
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            _B1,
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 5, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ) for matrix in (_M5, _D5)
    ),
    "read_source_kernels._apply_pairs": tuple((
        types.float64[:, :, :, ::1], matrix, _F3, TAPE, _F2,
        types.int64[:, ::1], types.int64[::1], _B1, _I, block, _F3, _F3, direction, _F3,
    ) for matrix in (_M5, _D5)
      for block, direction in ((_I, _B), (_I, types.literal(True)), (_I, types.literal(False)),
                               (types.literal(0), types.literal(False)))),
    "read_source_kernels._class_flows": (
        (
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.float64,
            types.float64,
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
        ),
    ),
    "read_source_kernels._exclusion_sums": (
        (
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
        ),
    ),
    "read_source_kernels._oriented": tuple(
        (
            types.Array(types.float64, 4, "C", readonly=False, aligned=True),
            matrix,
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            TAPE,
            _F2,
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            _B1,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 5, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 4, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ) for matrix in (_M5, _D5)
    ),
    "read_source_kernels._site_factor": ((
        types.float64,
        types.float64,
        types.Array(types.float64, 1, "C"),
        types.float64,
        types.Array(types.float64, 2, "C"),
        types.float64,
        types.float64,
        types.Array(types.float64, 1, "C"),
    ),),
    "sources._hamming_contract": (
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.float64,
            types.float64,
            types.float64,
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
    ),
}
