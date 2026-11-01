"""Compact cube/ball supports preserving every assignment and its order.

Cubes have at most 1,024 assignments. Balls store ordered flip sets instead of
C-by-K rows. Positive interval sums preserve tiny complement masses, including
in log space for cavity odds, in O(C*r + K**2) work.


Owns the completion support boundary (U04-U05). Must not read files, clocks or
environment, allocate execution resources or publish diagnostics.
Method: docs/methods/assembly.md#completion.
"""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numba import types

from haplotype_reconstruction.contracts.roles import record_field
from haplotype_reconstruction.core.kernels import kernel


@dataclass(frozen=True)
class SiteSupport:
    """Ordered assignment support: a centre and varying bits, with sparse ball flips.

    Arrays are borrowed read-only; cube construction owns its centre. The number
    of configurations is C. An empty flip width denotes an exact binary cube.
    """

    centre: np.ndarray = record_field("state", dtype="int8", axes=("K",))
    varying: np.ndarray = record_field("state", dtype="int64", axes=("U",))
    flips: np.ndarray = record_field("state", dtype="int64|int8", axes=("C", "r"))
    sizes: np.ndarray = record_field("state", dtype="int64|int8", axes=("C",))
    radius: int = record_field("state", dtype="int64", axes=())

    def __len__(self) -> int:
        return len(self.sizes)


def cube(centre: np.ndarray, varying: np.ndarray) -> SiteSupport:
    centre = np.asarray(centre, dtype=np.int8).copy()
    varying = np.asarray(varying, dtype=np.int64)
    centre[varying] = 0
    count = 1 << len(varying)
    return SiteSupport(centre, varying, np.empty((count, 0), np.int8), np.zeros(count, np.int8), 0)


def from_core(
    fixed: np.ndarray,
    unknown: np.ndarray,
    b: np.ndarray,
    core: np.ndarray,
    flips: np.ndarray,
    sizes: np.ndarray,
    radius: int,
) -> SiteSupport:
    centre = np.asarray(fixed, dtype=np.int8).copy()
    centre[unknown] = b
    varying = np.ascontiguousarray(unknown[core], dtype=np.int64)
    if not radius:
        return cube(centre, varying)
    return SiteSupport(centre, varying, flips, sizes, int(radius))


@kernel(
    overloads=(
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.literal(-1),
            types.literal(False),
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.int64,
            types.int64,
            types.boolean,
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int8, 2, "C", readonly=False),
            types.Array(types.int8, 1, "C", readonly=False),
            types.int64,
            types.literal(-1),
            types.literal(False),
        ),
        (
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.int8, 2, "C", readonly=False),
            types.Array(types.int8, 1, "C", readonly=False),
            types.int64,
            types.int64,
            types.boolean,
        ),
    ),
    cache=True,
)
def ball_sums(
    values: np.ndarray, flips: np.ndarray, sizes: np.ndarray, bits: int, radius: int, logarithmic: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip/stay/shell masses, using <=r+1 unflipped intervals per support row."""
    zero = -np.inf if logarithmic else 0.0
    changed = np.full(bits, zero)
    gaps = np.full((bits + 1, bits + 1), zero)
    shell = np.full(bits, zero)
    for row in range(len(sizes)):
        value = values[row]
        first: Any = 0
        for a in range(sizes[row]):
            bit = flips[row, a]
            if logarithmic:
                changed[bit] = np.logaddexp(changed[bit], value)
                if first < bit:
                    gaps[first, bit] = np.logaddexp(gaps[first, bit], value)
                if sizes[row] == radius:
                    shell[bit] = np.logaddexp(shell[bit], value)
            else:
                changed[bit] += value
                if first < bit:
                    gaps[first, bit] += value
                if sizes[row] == radius:
                    shell[bit] += value
            first = bit + 1
        if first < bits:
            if logarithmic:
                gaps[first, bits] = np.logaddexp(gaps[first, bits], value)
            else:
                gaps[first, bits] += value
    stayed = np.full(bits, zero)
    for first in range(bits):
        total = zero
        for stop in range(bits, first, -1):
            if logarithmic:
                total = np.logaddexp(total, gaps[first, stop])
                stayed[stop - 1] = np.logaddexp(stayed[stop - 1], total)
            else:
                total += gaps[first, stop]
                stayed[stop - 1] += total
    return changed, stayed, shell


def allele_sums(
    support: SiteSupport, values: np.ndarray, *, logarithmic: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Per-founder REF/ALT and outer-shell sums over the unchanged support."""
    zero = -np.inf if logarithmic else 0.0
    # NumPy 2.5's reduce stubs omit the plain allocating overload (np.ufunc: same object, generic type).
    total = cast(np.ufunc, np.logaddexp).reduce(values) if logarithmic else np.sum(values)
    sums = np.full((len(support.centre), 2), zero)
    sums[np.arange(len(support.centre)), support.centre] = total
    outer = np.full(len(support.centre), zero)
    if support.radius:
        changed, stayed, shell = ball_sums(
            values, support.flips, support.sizes, len(support.varying), support.radius, logarithmic
        )
        for bit, founder in enumerate(support.varying):
            centre = support.centre[founder]
            sums[founder, centre] = stayed[bit]
            sums[founder, 1 - centre] = changed[bit]
        outer[support.varying] = shell
    else:
        # This cube has <=MAX_EXACT_BITS free bits, independently of K.
        codes = np.arange(len(values))
        for bit, founder in enumerate(support.varying):
            alt = ((codes >> bit) & 1) != 0
            if logarithmic:
                # NumPy 2.5's reduce stubs omit the plain allocating overload (np.ufunc: same object, generic type).
                sums[founder, 0] = cast(np.ufunc, np.logaddexp).reduce(values[~alt])
                sums[founder, 1] = cast(np.ufunc, np.logaddexp).reduce(values[alt])
            else:
                sums[founder, 0] = np.sum(values[~alt])
                sums[founder, 1] = np.sum(values[alt])
    return sums, outer


@kernel(
    overloads=(
        (
            types.Array(types.int8, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int64, 2, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
        (
            types.Array(types.int8, 1, "C", readonly=False),
            types.Array(types.int64, 1, "C", readonly=False),
            types.Array(types.int8, 2, "C", readonly=False),
            types.Array(types.int8, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
            types.Array(types.float64, 1, "C", readonly=False),
        ),
    ),
    cache=True,
)
def ball_log_prior(
    centre: np.ndarray, varying: np.ndarray, flips: np.ndarray, sizes: np.ndarray, ref: np.ndarray, alt: np.ndarray
) -> np.ndarray:
    """Bernoulli priors including certain bits; zero priors avoid inf-inf."""
    total = 0.0
    impossible = 0
    for bit in range(len(centre)):
        value: Any = alt[bit] if centre[bit] else ref[bit]
        if value == -np.inf:
            impossible += 1
        else:
            total += value
    out = np.empty(len(sizes))
    for row in range(len(sizes)):
        value, zeros = total, impossible
        for a in range(sizes[row]):
            bit = varying[flips[row, a]]
            before = alt[bit] if centre[bit] else ref[bit]
            after = ref[bit] if centre[bit] else alt[bit]
            if before == -np.inf:
                zeros -= 1
            else:
                value -= before
            if after == -np.inf:
                zeros += 1
            else:
                value += after
        out[row] = -np.inf if zeros else value
    return out


def log_priors(support: SiteSupport, ref: np.ndarray, alt: np.ndarray) -> np.ndarray:
    if support.radius:
        return ball_log_prior(support.centre, support.varying, support.flips, support.sizes, ref, alt)
    rows = np.repeat(support.centre[None, :], len(support), axis=0)
    for bit, founder in enumerate(support.varying):
        rows[:, founder] = (np.arange(len(support)) >> bit) & 1
    return np.sum(np.where(rows == 1, alt, ref), axis=1)


__all__ = ["SiteSupport", "cube", "from_core", "ball_sums", "allele_sums", "ball_log_prior", "log_priors"]
