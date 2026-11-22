"""Kernel declarations for family transmission likelihoods and latent-path fitting."""

from numba import types

OVERLOADS = {}


# Family transmission (PD29) kernels receive owned contiguous work arrays through family_transmission's wrappers.
_ft_i1 = types.Array(types.int64, 1, "C")
_ft_i2 = types.Array(types.int64, 2, "C")
_ft_i3 = types.Array(types.int64, 3, "C")
_ft_b1 = types.Array(types.boolean, 1, "C")
_ft_b2 = types.Array(types.boolean, 2, "C")
_ft_f1 = types.Array(types.float64, 1, "C")
_ft_f2 = types.Array(types.float64, 2, "C")
_ft_f3 = types.Array(types.float64, 3, "C")
_ft_f4 = types.Array(types.float64, 4, "C")
_ft_f, _ft_n, _ft_q = types.float64, types.int64, types.boolean
OVERLOADS.update(
    {
        "family_likelihood._f": ((_ft_n, _ft_n, _ft_f),),
        "family_likelihood._emit": ((_ft_n, _ft_n, _ft_n, _ft_n, _ft_f, _ft_f),),
        "family_likelihood._switch": ((_ft_f, _ft_f),),
        "family_likelihood._child_fb": tuple(
            (_ft_i2, _ft_i2, _ft_b1, _ft_i3, _ft_f1, _ft_f, _ft_f2, want, _ft_f2, _ft_f1)
            for want in (_ft_q, types.literal(True), types.literal(False))
        ),
        "family_likelihood._half_fb": tuple(
            (_ft_i2, _ft_i2, _ft_b1, _ft_i3, latent, _ft_f1, _ft_f, _ft_f2, want, _ft_f3, _ft_f1)
            for latent in (_ft_n, types.literal(0), types.literal(1))
            for want in (_ft_q, types.literal(True), types.literal(False))
        ),
        "family_likelihood._prior_inc": ((_ft_i3, _ft_b1, _ft_n, _ft_f1, _ft_f, _ft_f1),),
        "family_likelihood._direct_inc": tuple(
            (_ft_i2, _ft_i2, _ft_b1, _ft_i3, latent, _ft_f2, _ft_f1)
            for latent in (_ft_n, types.literal(0), types.literal(1))
        ),
        "family_likelihood.evaluate": (
            (_ft_i2, _ft_i3, _ft_b2, _ft_i1, _ft_i3, _ft_b1, _ft_f1, _ft_f, _ft_f3, _ft_f1, _ft_f2, _ft_f1),
        ),
        "family_likelihood._prior_cached": ((_ft_i3, _ft_b1, _ft_n, _ft_f2, _ft_f1),),
        "family_likelihood._member_value": (
            (_ft_i2, _ft_i3, _ft_b2, _ft_i3, _ft_f1, _ft_f, _ft_f3, _ft_f2, _ft_n, _ft_n,
             _ft_b2, _ft_f2, _ft_f3),
        ),
        "family_likelihood._evaluate_cached": (
            (_ft_i2, _ft_i3, _ft_b2, _ft_i1, _ft_i3, _ft_b1, _ft_f1, _ft_f, _ft_f3, _ft_f1, _ft_f2,
             _ft_f1, _ft_b2, _ft_f2, _ft_f3, _ft_f2),
        ),
        "family_fit._prepare": ((_ft_i2, _ft_i3, _ft_b2, _ft_f1, _ft_f, _ft_f3),),
        "family_fit._pair_transition": tuple(
            (_ft_f2, _ft_f2, types.Array(types.int32, 3, "C"), _ft_n, _ft_f, _ft_f, _ft_f1, _ft_i1,
             _ft_f1, _ft_i1, _ft_f2, size)
            for size in (_ft_n, types.literal(6), types.literal(8))
        ),
        "family_fit._mstep": tuple(
            (_ft_i2, _ft_i3, _ft_b2, _ft_i1, _ft_f3, _ft_i3, _ft_n, _ft_f1, _ft_f, _ft_f3, maxmode, _ft_f4, _ft_f4, _ft_i2, _ft_f2)
            for maxmode in (_ft_q, types.literal(True), types.literal(False))
        ),
        "family_fit._emis4": ((_ft_i2, _ft_i2, _ft_b1, _ft_i3, _ft_n, _ft_f2, _ft_f1),),
        "family_fit._half_flip": ((_ft_i2, _ft_i2, _ft_b1, _ft_i3, _ft_n, _ft_f1, _ft_f, _ft_f2, _ft_f1),),
        "family_fit._flip_gains_both": ((_ft_i2, _ft_i3, _ft_b2, _ft_i1, _ft_i3, _ft_f1, _ft_f, _ft_f3),),
        "family_fit.flip_gains": ((_ft_i2, _ft_i3, _ft_b2, _ft_i1, _ft_i3, _ft_n, _ft_f1, _ft_f, _ft_f3),),
        "family_fit.fit": (
            (_ft_i2, _ft_i3, _ft_b2, _ft_i1, _ft_i3, _ft_b1, _ft_f1, _ft_f, _ft_f3, _ft_n, _ft_f2, _ft_b1, _ft_b1,
             _ft_f1, _ft_f4, _ft_i2, _ft_f2),
        ),
        "family_fit.init_latent": ((_ft_i2, _ft_i3, _ft_b2, _ft_i1, _ft_i3, _ft_n, _ft_f1, _ft_f, _ft_f3, _ft_f4, _ft_i2, _ft_f2),),
    }
)
