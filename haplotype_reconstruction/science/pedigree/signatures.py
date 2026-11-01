"""Declared pedigree kernel inputs, transcribed from the frozen 3.5 release manifest.

Preserves dtype, rank, layout, mutability and literal specialization. C-layout
variants implement the L2 seam normalization. Declarations do not compile at import.
"""

from numba import types

OVERLOADS = {
    "family._cavity_iterations": (
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int8, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.int64,
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.int64,
            types.int64,
            types.float64,
            types.float64,
            types.int64,
            types.int64,
            types.float64,
            types.int64,
        ),
    ),
    "family._cavity_passes": (
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int8, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.int64,
            types.int64,
            types.int64,
        ),
    ),
    "family._child_cavity": (
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int8, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
        ),
    ),
    "family._child_log_odds": (
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int8, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.float64, 2, "C", readonly=False),
        ),
    ),
    "family._configuration_panels": (
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int8, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.int64,
            types.int64,
            types.int64,
        ),
    ),
    "family._family_log_odds": (
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int8, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.int64,
            types.int64,
        ),
    ),
    "family._lane_cavity": (
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int8, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.bool, 1, "C", readonly=False),
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int8, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.int64, 1, "C", readonly=False),
            types.literal(0),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.bool, 1, "C", readonly=False),
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int8, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.int64,
            types.uint64,
            types.int64,
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.bool, 1, "C", readonly=False),
        ),
    ),
    "family._lane_count": (
        (
            types.int64,
            types.int64,
        ),
    ),
    "family._path_correction": (
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int8, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.int64,
            types.int64,
        ),
    ),
    "family._path_singles": (
        (
            types.int64,
            types.int64,
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.int64,
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
        ),
    ),
    "family._path_target": (
        (
            types.int64,
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.int64,
        ),
    ),
    "family._sample_panel": (
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int8, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
    ),
    "family._with_incoming": (
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
        ),
    ),
    "graph._is_acyclic": (
        (
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
        ),
    ),
    "orientation._order_probability": (
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
    ),
    "orientation._pair_log_odds": (
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.float64,
        ),
    ),
    "orientation._weighted_log_odds": (
        (
            types.Array(types.float64, 2, "C", readonly=False),
            types.Array(types.bool, 2, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.int64,
        ),
    ),
    "resampling._accumulate": (
        (
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int8, 2, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
        ),
    ),
    "selection._pack_presence": ((types.Array(types.bool, 2, "C", readonly=False),),),
    "selection._units_present": (
        (
            types.Array(types.uint64, 2, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 2, "C", readonly=False),
        ),
    ),
    "ties._top_two_finite": (
        (
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
    ),
    "family._exp_or_zero": ((types.float64,),),
    "family._log_or_zero": ((types.float64,),),
    "family._softplus_tail": ((types.float64,),),
    "family._logaddexp": (
        (
            types.float64,
            types.float64,
        ),
    ),
}

# The framework borrows fixture arrays read-only. np.ascontiguousarray preserves
# that flag when no copy is needed; intermediate pair axes and weights stay owned.
for _name, _borrowed in (
    ("orientation._order_probability", (0, 1, 2)),
    ("orientation._pair_log_odds", (0, 5, 6)),
    ("orientation._weighted_log_odds", (4,)),
):
    _original = OVERLOADS[_name][0]
    _readonly = tuple(
        types.Array(value.dtype, value.ndim, value.layout, readonly=True) if index in _borrowed else value
        for index, value in enumerate(_original)
    )
    OVERLOADS[_name] = (*OVERLOADS[_name], _readonly)
