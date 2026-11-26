"""Label-free read transitions shared by the child and map chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import types

from ...core.kernels import kernel
from ..painting.api import donor_ratio

_F1 = types.float64[::1]
_F2 = types.float64[:, ::1]
_F3 = types.float64[:, :, ::1]
_F = types.float64
_I = types.int64


@dataclass(frozen=True)
class ForeignReads:
    """Selected-site library inputs for the foreign class. Depth is its Poisson mean."""

    counts: np.ndarray
    error: np.ndarray
    loss: np.ndarray
    survival: np.ndarray
    absent: np.ndarray
    depth: np.ndarray


def foreign_reads(builder, sites, samples):
    """Use the painting builder's sample/library inputs on the same selected-site grid, with its foreign depth and
    donor-ratio counts (both zero for QC A's corrected rows, ReadEmissionBuilder.foreign_depth and foreign_counts)."""
    if not builder.foreign_read_share:
        return None
    positions = builder.positions[sites]
    gaps = np.diff(positions, prepend=positions[0]).astype(np.float64)
    libraries = builder.library[samples]
    survival = np.empty((len(libraries), len(sites)))
    loss = np.empty(len(libraries))
    for library in np.unique(libraries):
        use = libraries == library
        model = builder.survivals[library]
        survival[use] = model.survival(gaps)
        loss[use] = model.loss
    survival[:, 0] = 1.0
    depth = builder.foreign_depth(samples)
    return ForeignReads(
        builder.foreign_counts(samples, sites),
        np.ascontiguousarray(builder.error_rate[samples]),
        loss, survival, np.exp(-depth[:, None] * (1.0 - survival)), depth,
    )


@kernel(overloads=((_F2, _F1, _F2, _I, _F3, _F1, _F1, _I, _F2),), cache=True, inline="always", nogil=True)
def configuration_ratios(likelihood, law, alt, site, counts, error, loss, sample, out):
    """Own count ratios normalized by the own law, followed by the GL-anchored donor ratios."""
    own = len(law)
    for config in range(4):
        mean = 0.0
        for c in range(own):
            mean += law[c] * likelihood[config, c]
        for c in range(own):
            out[config, c] = likelihood[config, c] / mean if mean > 0.0 else 1.0
    if out.shape[1] > own:
        ref, alternate = counts[sample, site]
        depth = ref + alternate
        e, delta = error[sample], loss[sample]
        for config in range(4):
            expected = e if config == 0 else (1.0 - e if config == 3 else 0.5)
            # The painting's donor ratios (painting.read_factor.donor_ratio): one for an allele the configuration
            # carries; a donor label's fractional allele mixes the two linearly.
            ratio_ref = donor_ratio(config, 0, alternate, depth, e, delta, expected)
            ratio_alt = donor_ratio(config, 1, alternate, depth, e, delta, expected)
            for k in range(out.shape[1] - own):
                out[config, own + k] = alt[k, site] * ratio_alt + (1.0 - alt[k, site]) * ratio_ref


@kernel(overloads=((_F2, _F1, _F, _F, _F1, _F1),), cache=True, inline="always", nogil=True)
def site_row(transition, law, survival, absent, row, out):
    """One selected-site edge with no emission."""
    own = len(law)
    donors = len(row) - own
    foreign_mass = 0.0
    arrival = 0.0
    if donors:
        own_mass = 0.0
        for c in range(own):
            own_mass += row[c]
        for c in range(own, len(row)):
            foreign_mass += row[c]
        arrival = (1.0 - absent) * (own_mass + (1.0 - survival) * foreign_mass) / donors
    restart = (1.0 - survival) * absent * foreign_mass
    for j in range(own):
        value = 0.0
        for i in range(own):
            value += row[i] * transition[i, j]
        out[j] = absent * value + restart * law[j]
    for j in range(own, len(row)):
        out[j] = survival * row[j] + arrival


@kernel(overloads=((_F2, _F1, _F, _F, _F1, _F1),), cache=True, inline="always", nogil=True)
def site_adjoint(transition, law, survival, absent, column, out):
    """Adjoint of one selected-site edge with no emission."""
    own = len(law)
    donors = len(column) - own
    arrival = 0.0
    restart = 0.0
    if donors:
        for c in range(own, len(column)):
            arrival += column[c]
        arrival *= (1.0 - absent) / donors
        for c in range(own):
            restart += law[c] * column[c]
        restart = (1.0 - survival) * (absent * restart + arrival)
    for i in range(own):
        value = 0.0
        for j in range(own):
            value += transition[i, j] * column[j]
        out[i] = absent * value + arrival
    for i in range(own, len(column)):
        out[i] = survival * column[i] + restart


@kernel(overloads=((_F2,),), cache=True, inline="always", nogil=True)
def identity_gap(gap):
    """Initialize a gap descriptor with blocks A, e, l and scalars s, r."""
    own = gap.shape[0] - 1
    gap[:] = 0.0
    for i in range(own + 1):
        gap[i, i] = 1.0


@kernel(overloads=((_F2, _F2, _F1, _F, _F, _F2),), cache=True, inline="always", nogil=True)
def append_site(gap, transition, law, survival, absent, out):
    """Compose a label-free gap and one edge. Store the foreign renewal directly."""
    own = len(law)
    arrival = 1.0 - absent
    renew = (1.0 - survival) * arrival
    restart = (1.0 - survival) * absent
    foreign = gap[own, own] + gap[own, own + 1]
    for i in range(own + 1):
        mass = 0.0
        for k in range(own):
            mass += gap[i, k]
        entering = gap[i, own] if i < own else foreign
        for j in range(own):
            value = 0.0
            for k in range(own):
                value += gap[i, k] * transition[k, j]
            out[i, j] = absent * value + entering * restart * law[j]
        if i < own:
            out[i, own + 1] = 0.0
            out[i, own] = mass * arrival + entering * (survival + renew)
        else:
            out[own, own] = gap[own, own] * survival
            out[own, own + 1] = mass * arrival + gap[own, own] * renew + gap[own, own + 1] * (survival + renew)


@kernel(overloads=((_F2, _F2, _F2),), cache=True, inline="always", nogil=True)
def compose_gap(first, second, out):
    """Compose two label-free gaps in site order."""
    own = first.shape[0] - 1
    second_foreign = second[own, own] + second[own, own + 1]
    first_foreign = first[own, own] + first[own, own + 1]
    for i in range(own + 1):
        entering = first[i, own] if i < own else first_foreign
        for j in range(own + 1):
            value = 0.0
            for k in range(own):
                value += first[i, k] * second[k, j]
            if j < own:
                out[i, j] = value + entering * second[own, j]
            elif i < own:
                out[i, own + 1] = 0.0
                out[i, own] = value + entering * second_foreign
            else:
                out[own, own] = first[own, own] * second[own, own]
                out[own, own + 1] = (
                    value + first[own, own] * second[own, own + 1] + first[own, own + 1] * second_foreign
                )


@kernel(overloads=((_F2, _F1, _F1, types.boolean),), cache=True, inline="always", nogil=True)
def apply_gap(gap, vector, out, backward):
    """Apply an unrounded gap in the row or adjoint direction."""
    own = gap.shape[0] - 1
    donors = len(vector) - own
    foreign_mass = 0.0
    for j in range(own, len(vector)):
        foreign_mass += vector[j]
    same = gap[own, own]
    renew = gap[own, own + 1]
    if backward:
        mean = foreign_mass / donors
        foreign = renew * mean
        for i in range(own):
            value = gap[i, own] * mean
            for j in range(own):
                value += gap[i, j] * vector[j]
            out[i] = value
            foreign += gap[own, i] * vector[i]
    else:
        foreign = renew * foreign_mass
        for j in range(own):
            value = foreign_mass * gap[own, j]
            for i in range(own):
                value += vector[i] * gap[i, j]
            out[j] = value
            foreign += vector[j] * gap[j, own]
        foreign /= donors
    for j in range(own, len(vector)):
        out[j] = same * vector[j] + foreign
