"""Kernel declarations for the exclusion test and its upper bounds."""

from numba import types

OVERLOADS = {
    "exclusion._log_bet": ((types.float64, types.float64),),
    "exclusion._ratio": ((types.float64,) * 11,),
    # 4.22 A11: likelihoods and observations may arrive frozen (record arrays) or writable; the cohort law, rates,
    # rows and certification flags are always passed as writable contiguous copies (exclusion_bound.score_contig).
    "exclusion_bound._completed_windows": tuple(
        (
            types.Array(types.float64, 3, "C", readonly=gl_readonly, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=observed_readonly, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
            types.float64,
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
        )
        for gl_readonly in (True, False)
        for observed_readonly in (True, False)
    ),
    "exclusion_bound._observed_counts": (
        (
            types.Array(types.uint64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.uint64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=True, aligned=True),
        ),
    ),
    "exclusion_bound._popcount": ((types.uint64,),),
}
