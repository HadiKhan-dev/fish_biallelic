"""Kernel declarations for block and tile scores and transmission evidence."""

from numba import types

from .read_chain import _PATH_OVERLOADS
from .read_layout import _CACHE_ARGUMENT, _F64_3, _I64_1, _I64_2, _READ_TAILS, _READ_TERMS, _reset, _width, extend

OVERLOADS = {
    "transmission._log_likelihood": (
        (
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 3, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.int64, 2, "C", readonly=True, aligned=True),
            types.Array(types.int64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 3, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
        ),
    ),
}


# Low-material forwards use the same production arrays, with no common-product cache.
_block_forward = tuple(dict.fromkeys(
    (types.int64,) * 4 + signature[4:25] + (types.none, first_cached, types.none)
    for signature in _PATH_OVERLOADS
    for first_cached in (types.int64, types.literal(0))))
_block_axis = (types.Array(types.int64, 1, "C"), types.int64)
OVERLOADS["read_block_forward.score_path_blocks"] = tuple(signature + _block_axis for signature in _block_forward)
OVERLOADS["read_block_forward.score_block_rows"] = tuple(dict.fromkeys(
    (types.Array(types.int64, 2, "C"), *_block_axis, types.Array(types.boolean, 1, "C")) + signature[4:25]
    for signature in _block_forward))
OVERLOADS["transmission.blocks"] = ((
    types.Array(types.int64, 2, "C"), types.Array(types.int64, 3, "C"),
    types.Array(types.int64, 3, "C"), types.Array(types.boolean, 2, "C"),
    types.Array(types.float64, 1, "C"), types.int64, types.float64, types.Array(types.float64, 3, "C"),
    types.Array(types.int64, 1, "C"), types.int64),)


OVERLOADS["read_block_forward.score_path_blocks"] = extend(OVERLOADS["read_block_forward.score_path_blocks"], *_READ_TAILS)
# score_block_rows takes its tiles' row offsets after the rows (read_block_forward.row_tiles)
OVERLOADS["read_block_forward.score_block_rows"] = tuple(
    (signature[0], _I64_1) + signature[1:]
    for signature in extend(OVERLOADS["read_block_forward.score_block_rows"],
                             *((*_READ_TERMS, cache, types.int64) for cache in _CACHE_ARGUMENT)))


_T2_OBSERVED = {
    "read_block_forward.score_path_blocks": (
        (
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.int64, 2, "C", readonly=True, aligned=True),
            types.Array(types.float64, 2, "C", readonly=True, aligned=True),
            types.Array(types.int8, 2, "C", readonly=True, aligned=True),
            types.Array(types.float64, 2, "C", readonly=True, aligned=True),
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 2, "C", readonly=True, aligned=True),
            types.Array(types.int64, 2, "C", readonly=True, aligned=True),
            types.Array(types.float64, 2, "C", readonly=True, aligned=True),
            types.Array(types.float64, 2, "C", readonly=True, aligned=True),
            types.Array(types.float64, 2, "C", readonly=True, aligned=True),
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.float64,
            types.none,
            types.int64,
            types.none,
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float32, 3, "C", readonly=False, aligned=True),
            types.Array(types.float32, 3, "C", readonly=False, aligned=True),
            types.Array(types.float32, 3, "C", readonly=False, aligned=True),
            types.none,
        ),
    ),
}
for _name, _signatures in _T2_OBSERVED.items():
    OVERLOADS[_name] = tuple(dict.fromkeys(tuple(OVERLOADS.get(_name, ())) + _signatures))


# read_block_forward._score_tile (a tile of score_block_rows, or score_path_blocks' one row): (resolved rows, start, stop,
# block axis) + score_path_blocks' model, cache and read arguments + the views output float64 [R, H, 2]
OVERLOADS["read_block_forward._score_tile"] = tuple(dict.fromkeys(
    (_I64_2, types.int64, types.int64, signature[28], signature[29]) + tuple(signature[4:28]) + tuple(signature[30:36])
    + (_F64_3,)
    for signature in OVERLOADS["read_block_forward.score_path_blocks"]))


OVERLOADS["read_block_forward.score_block_rows"] = tuple(
    sig[:28] + (_reset, _width) + sig[28:] for sig in OVERLOADS["read_block_forward.score_block_rows"])
OVERLOADS["read_block_forward._score_tile"] = tuple(
    sig[:31] + (_reset, _width) + sig[31:] for sig in OVERLOADS["read_block_forward._score_tile"])
OVERLOADS["read_block_forward.score_path_blocks"] = tuple(
    sig[:32] + (_reset, _width) + sig[32:] for sig in OVERLOADS["read_block_forward.score_path_blocks"])
