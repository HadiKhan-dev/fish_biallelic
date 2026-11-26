"""Kernel declarations for child read factors, class movement and transfer caches."""

from numba import types

from .read_layout import _BOOL_2, _F32_3, _F64_1, _F64_2, _F64_3, _F64_4, _I64_1, _I64_2, _reset, _width

OVERLOADS = {}

_GL_ROWS = (_F64_1, types.Array(types.float64, 1, "C", readonly=True))
OVERLOADS.update({
    "child_reads._fill": tuple(
        (_F64_4, _F64_3, _F64_4, _F64_3, _F64_2, types.float64, _I64_1, observed, _I64_1, _F32_3, _F32_3,
         _F64_2, _F64_3, _F64_1, _F64_1, _F64_2, _F64_2)
        for observed in (_BOOL_2, types.Array(types.boolean, 2, "C", readonly=True))),
    "read_transfer.read_ratio": tuple((_F64_1, _F32_3, types.int64, types.float64, types.float64, gl)
                                     for gl in _GL_ROWS),
    "read_transfer.class_transfers": tuple(
        (types.int64, types.int64, _F64_2, gl, _I64_1, _I64_2, _I64_2, _I64_1, _F32_3, _F32_3)
        for gl in (_F64_3, types.Array(types.float64, 3, "C", readonly=True))),
})
OVERLOADS["read_transfer.carry_bin"] = ()


OVERLOADS["read_transfer.class_transfers"] = tuple(
    sig[:8] + (_reset, _width) + sig[8:] for sig in OVERLOADS["read_transfer.class_transfers"])
OVERLOADS["read_transfer.carry_bin"] = tuple(
    sig[:15] + (_reset, _width) + sig[15:] for sig in OVERLOADS["read_transfer.carry_bin"])


OVERLOADS["child_reads.class_layout"] = tuple((
    types.Array(types.float32, 3, "C"), types.Array(types.int64, 1, "C"), _width, _width, count,
) for count in (types.int64, types.Omitted(0)))
OVERLOADS["read_transfer.multiply_classes"] = ((
    types.Array(types.float64, 2, "C"), types.Array(types.float32, 2, "C"), types.Array(types.float64, 1, "C"),
    types.Array(types.float64, 2, "C"), types.int64, types.boolean,
),)


# The inline call boundary normalizes signed indices before entering the shared
# carry kernel. Prepared arrays keep their read-only views.
_carry_gl = types.Array(types.float64, 3, "C", readonly=True)
_carry_state = types.Array(types.float64, 2, "C", readonly=True)
_carry_site = types.Array(types.int64, 1, "C", readonly=True)
_carry_bins = types.Array(types.int64, 2, "C", readonly=True)
_carry_layouts = (
    (_F64_3, _F64_2, _F64_3, _I64_1, _I64_2, _I64_2),
    (_F64_3, _F64_2, _carry_gl, _I64_1, _I64_2, _I64_2),
    (_carry_gl, _carry_state, _carry_gl, _I64_1, _I64_2, _I64_2),
    (_carry_gl, _carry_state, _carry_gl, _carry_site, _carry_bins, _carry_bins),
)
OVERLOADS["read_transfer._carry_bin"] = tuple(
    (_F64_3, _F64_3, _F64_1, _F64_1,
     types.int64, types.int64, types.int64, types.int64, *layout,
     _I64_1, _reset, _width, _F32_3, _F32_3, cache, types.int64, _F64_1)
    for layout in _carry_layouts
    for cache in (types.none, types.Array(types.float64, 6, "C"))
)

OVERLOADS["read_transfer._carry_background"] = tuple(
    (_F64_3, _F64_3, _F64_1, types.int64, types.int64, types.int64, ca, state, gl, site,
     _reset, _F32_3, _F32_3, *(types.int64,) * 5)
    for ca, state, gl, site, _, _ in _carry_layouts
)
# Every pair of an uncached structured bin in class planes: the background carrier's arguments without its factor.
OVERLOADS["read_transfer._carry_lanes"] = tuple(
    sig[:2] + sig[3:] for sig in OVERLOADS["read_transfer._carry_background"])
