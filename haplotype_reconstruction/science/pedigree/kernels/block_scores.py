"""Kernel declarations for block and tile scores and transmission evidence."""

from numba import types

from .read_chain import _PATH_OVERLOADS
from .read_layout import (
    _CACHE_ARGUMENT,
    _F64_2,
    _F64_4,
    _I64_1,
    _I64_2,
    _READ_TAILS,
    _READ_TERMS,
    _reset,
    _width,
    extend,
)

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


# Low-material forwards use the same production arrays, with no common-product cache: the chain of one row of
# blocks (score_path's arguments without its common-product cache, then the block axis and the read arguments), the
# source of the tile declarations.
_block_forward = tuple(dict.fromkeys(
    (types.int64,) * 4 + signature[4:24] + (types.none, first_cached)
    for signature in _PATH_OVERLOADS
    for first_cached in (types.int64, types.literal(0))))
_block_axis = (types.Array(types.int64, 1, "C"), types.int64)
_OBSERVED_PATH_BLOCKS = (
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
        types.Array(types.float64, 3, "C", readonly=True, aligned=True),
        types.float64,
        types.none,
        types.int64,
        types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        types.int64,
        types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        types.Array(types.float64, 2, "C", readonly=False, aligned=True),
        types.Array(types.float32, 3, "C", readonly=False, aligned=True),
        types.Array(types.float32, 3, "C", readonly=False, aligned=True),
        types.none,
    ),
    )
_path_blocks = tuple(dict.fromkeys(
    extend(tuple(signature + _block_axis for signature in _block_forward), *_READ_TAILS) + _OBSERVED_PATH_BLOCKS))
OVERLOADS["read_block_forward.score_block_rows"] = tuple(dict.fromkeys(
    (types.Array(types.int64, 2, "C"), *_block_axis, types.Array(types.boolean, 1, "C")) + signature[4:24]
    for signature in _block_forward))
OVERLOADS["transmission.blocks"] = ((
    types.Array(types.int64, 2, "C"), types.Array(types.int64, 3, "C"),
    types.Array(types.int64, 3, "C"), types.Array(types.boolean, 2, "C"),
    types.Array(types.float64, 1, "C"), types.int64, types.float64, types.Array(types.float64, 3, "C"),
    types.Array(types.int64, 1, "C"), types.int64),)


# score_block_rows takes its tiles' row offsets after the rows (read_block_forward.row_tiles)
OVERLOADS["read_block_forward.score_block_rows"] = tuple(
    (signature[0], _I64_1) + signature[1:]
    for signature in extend(OVERLOADS["read_block_forward.score_block_rows"],
                             *((*_READ_TERMS, cache, types.int64) for cache in _CACHE_ARGUMENT)))


# read_block_forward._score_tile (a tile of score_block_rows): (resolved rows, start, stop, block axis) + the blocks
# chain's model, cache and read arguments + the scores output float64 [R, H]
OVERLOADS["read_block_forward._score_tile"] = tuple(dict.fromkeys(
    (_I64_2, types.int64, types.int64, signature[26], signature[27]) + tuple(signature[4:24])
    + (products, signature[25]) + tuple(signature[28:33])
    + (_F64_2,)
    for signature in _path_blocks for products in (types.none, _F64_4)))


OVERLOADS["read_block_forward.score_block_rows"] = tuple(
    sig[:27] + (_reset, _width) + sig[27:] + (products,)
    for sig in OVERLOADS["read_block_forward.score_block_rows"]
    for products in (types.none, types.Omitted(None), _F64_4))
OVERLOADS["read_block_forward._score_tile"] = tuple(
    sig[:29] + (_reset, _width) + sig[29:] for sig in OVERLOADS["read_block_forward._score_tile"])

# The block wrapper takes the live team size and immutable child work before the block axis.
OVERLOADS["read_block_forward.score_block_rows"] = tuple(
    sig[:2] + (types.int64, _I64_1) + sig[2:] for sig in OVERLOADS["read_block_forward.score_block_rows"])
