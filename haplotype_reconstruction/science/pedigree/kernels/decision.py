"""Kernel declarations for genome-wide exposure, orientation, family fitting and resampling."""

from numba import types

OVERLOADS = {
    "exposure._edges": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 1, "C", readonly=True, aligned=True),
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
        ),
    ),
    "exposure._pairs": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 2, "C", readonly=True, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
        ),
    ),
    "exposure._presence": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 1, "C", readonly=True, aligned=True),
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "exposure._unit": ((types.float64,),),
    "family._cavity_iterations": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
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
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
            types.int64,
        ),
    ),
    "family._child_cavity": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
        ),
    ),
    "family._child_log_odds": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
        ),
    ),
    "family_paths._configuration_panels": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
            types.int64,
        ),
    ),
    "family_numerics._exp_or_zero": ((types.float64,),),
    "family._family_log_odds": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
        ),
    ),
    "family._lane_cavity": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.literal(0),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.uint64,
            types.int64,
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "family_numerics._lane_count": ((types.int64, types.int64),),
    "family_numerics._log_or_zero": ((types.float64,),),
    "family_paths._path_correction": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
        ),
    ),
    "family_paths._path_singles": (
        (
            types.int64,
            types.int64,
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
        ),
    ),
    "family_paths._path_target": (
        (
            types.int64,
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.int64,
        ),
    ),
    "family_paths._sample_panel": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int8, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
            types.int64,
            types.int64,
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "family_numerics._softplus_tail": ((types.float64,),),
    "family._with_incoming": (
        (
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
        ),
    ),
    "junctions._local_classes": (
        (
            types.Array(types.int8, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.int8, 2, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "junctions._minimum_switches": (
        (
            types.Array(types.int16, 3, "C", readonly=False, aligned=True),
            types.Array(types.int16, 2, "C", readonly=False, aligned=True),
        ),
    ),
    "junctions._pooled_labels": (
        (
            types.Array(types.int16, 3, "C", readonly=False, aligned=True),
            types.Array(types.int16, 2, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.int16, 3, "C", readonly=True, aligned=True),
            types.Array(types.int16, 2, "C", readonly=False, aligned=True),
        ),
    ),
    "junctions._trace_schedule": (
        (
            types.Array(types.int16, 2, "C", readonly=False, aligned=True),
            types.Array(types.int16, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.boolean,
        ),
        (
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.boolean,
        ),
    ),
    "junctions._trace_schedules": (
        (
            types.Array(types.int16, 3, "C", readonly=False, aligned=True),
            types.Array(types.int16, 2, "C", readonly=False, aligned=True),
            types.boolean,
        ),
    ),
    "junctions._unique_trajectories": ((types.Array(types.int16, 2, "C", readonly=False, aligned=True),),),
    "orientation._order_probability": (
        (
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 2, "C", readonly=True, aligned=True),
            types.Array(types.float64, 2, "C", readonly=True, aligned=True),
            types.Array(types.float64, 1, "C", readonly=True, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "orientation._pair_log_odds": (
        (
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 2, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=True, aligned=True),
            types.Array(types.float64, 1, "C", readonly=True, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "orientation._weighted_log_odds": (
        (
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.int64,
        ),
        (
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=True, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.int64,
        ),
    ),
    "graph._is_acyclic": (
        (
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
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
        ),
    ),
    "ties._top_two_finite": (
        (
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
    ),
}


# Relationship kernels receive owned contiguous work arrays through public wrappers.
_rel_i1 = types.Array(types.int64, 1, "C")
_rel_i2 = types.Array(types.int64, 2, "C")
_rel_i3 = types.Array(types.int64, 3, "C")
_rel_b2 = types.Array(types.boolean, 2, "C")
_rel_x2 = types.Array(types.int8, 2, "C")
_rel_f1 = types.Array(types.float64, 1, "C")
_rel_f2 = types.Array(types.float64, 2, "C")
OVERLOADS.update(
    {
        "relationship._pair_chance": ((_rel_i3, _rel_b2),),
        "relationship._pair_value": ((_rel_i3, _rel_b2, types.int64, types.int64, types.int64),),
        "relationship._feasible3": ((_rel_i2,),),
        "relationship._pair_sequence": ((_rel_i3, _rel_b2, _rel_i2),),
        "relationship._trio_sequence": ((_rel_i3, _rel_b2, _rel_i2),),
        "relationship._pair_persistence": ((_rel_i3, _rel_b2),),
        "relationship._process_log_likelihoods": (
            (_rel_x2, _rel_f1, _rel_i1, _rel_i1, _rel_f1, _rel_f1, _rel_f2, _rel_f2),
        ),
    }
)


OVERLOADS["orientation._order_statistics"] = tuple(
    (types.Array(types.float64, 2, "C", readonly=readonly),
     types.Array(types.float64, 2, "C", readonly=readonly),
     types.Array(types.float64, 1, "C", readonly=readonly)) for readonly in (False, True))
OVERLOADS["orientation._omit_order_probability"] = tuple(
    (types.Array(types.float64, 3, "C", readonly=totals_readonly),
     types.Array(types.float64, 1, "C", readonly=rows_readonly),
     types.Array(types.float64, 1, "C", readonly=rows_readonly), types.float64)
    for totals_readonly in (False, True) for rows_readonly in (False, True))


def _read_only(signature, positions):
    """``signature`` with the arrays at ``positions`` read-only."""
    return tuple(kind.copy(readonly=True) if index in positions else kind for index, kind in enumerate(signature))


# Resampling: the workers' arguments (the shared decision inputs) and the results the parent gathers are read-only
# (engine.workers), so the kernels of a resampling fit and of the gather run with these arguments read-only as well
# as writable (the decision fit builds its own). Undeclared, each was compiled on first use: one worker compiled for
# about ten seconds while every other worker of the stage waited for it.
RESAMPLING_READ_ONLY_ARGUMENTS = {
    "family._cavity_iterations": (1, 2),
    "family._cavity_passes": (1, 2),
    "family._child_cavity": (1, 2),
    "family._child_log_odds": (1, 2),
    "family_paths._configuration_panels": (1, 2),
    "family._family_log_odds": (1, 2),
    "family._lane_cavity": (1, 2),
    "family_paths._path_correction": (1, 2),
    "family_paths._path_singles": (2,),
    "family_paths._path_target": (1,),
    "family_paths._sample_panel": (1, 2),
    "family._with_incoming": (1,),
    "graph._is_acyclic": (1,),
    "orientation._order_probability": (0, 1, 2, 3),
    "orientation._weighted_log_odds": (0, 1, 2, 3, 4, 5),
    "resampling._accumulate": (0, 1, 2, 3),
    "selection._units_present": (0, 1),
}
for _name, _positions in RESAMPLING_READ_ONLY_ARGUMENTS.items():
    OVERLOADS[_name] = tuple(dict.fromkeys(
        OVERLOADS[_name] + tuple(_read_only(signature, _positions) for signature in OVERLOADS[_name])
    ))
