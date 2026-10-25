"""Compact cube/ball supports preserving every assignment and its order.

Cubes have at most 1,024 assignments. Balls store ordered flip sets instead of
C-by-K rows. Positive interval sums preserve tiny complement masses, including
in log space for cavity odds, in O(C*r + K**2) work.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit


@dataclass(frozen=True)
class SiteSupport:
    centre: np.ndarray
    varying: np.ndarray
    flips: np.ndarray
    sizes: np.ndarray
    radius: int

    def __len__(self):
        return len(self.sizes)


def cube(centre, varying):
    centre = np.asarray(centre, dtype=np.int8).copy()
    varying = np.asarray(varying, dtype=np.int64)
    centre[varying] = 0
    count = 1 << len(varying)
    return SiteSupport(centre, varying, np.empty((count, 0), np.int8),
                       np.zeros(count, np.int8), 0)


def from_core(fixed, unknown, b, core, flips, sizes, radius):
    centre = np.asarray(fixed, dtype=np.int8).copy()
    centre[unknown] = b
    varying = np.ascontiguousarray(unknown[core], dtype=np.int64)
    if not radius:
        return cube(centre, varying)
    return SiteSupport(centre, varying, flips, sizes, int(radius))


@njit(cache=True)
def ball_sums(values, flips, sizes, bits, radius, logarithmic=False):
    """Flip/stay/shell masses, using <=r+1 unflipped intervals per support row."""
    zero = -np.inf if logarithmic else 0.0
    changed = np.full(bits, zero)
    gaps = np.full((bits + 1, bits + 1), zero)
    shell = np.full(bits, zero)
    for row in range(len(sizes)):
        value = values[row]
        first = 0
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


def allele_sums(support, values, *, logarithmic=False):
    """Per-founder REF/ALT and outer-shell sums over the unchanged support."""
    zero = -np.inf if logarithmic else 0.0
    total = np.logaddexp.reduce(values) if logarithmic else np.sum(values)
    sums = np.full((len(support.centre), 2), zero)
    sums[np.arange(len(support.centre)), support.centre] = total
    outer = np.full(len(support.centre), zero)
    if support.radius:
        changed, stayed, shell = ball_sums(
            values, support.flips, support.sizes, len(support.varying), support.radius,
            logarithmic)
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
                sums[founder, 0] = np.logaddexp.reduce(values[~alt])
                sums[founder, 1] = np.logaddexp.reduce(values[alt])
            else:
                sums[founder, 0] = np.sum(values[~alt])
                sums[founder, 1] = np.sum(values[alt])
    return sums, outer


@njit(cache=True)
def ball_log_prior(centre, varying, flips, sizes, ref, alt):
    """Bernoulli priors including certain bits; zero priors avoid inf-inf."""
    total = 0.0
    impossible = 0
    for bit in range(len(centre)):
        value = alt[bit] if centre[bit] else ref[bit]
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


def log_priors(support, ref, alt):
    if support.radius:
        return ball_log_prior(support.centre, support.varying, support.flips,
                              support.sizes, ref, alt)
    rows = np.repeat(support.centre[None, :], len(support), axis=0)
    for bit, founder in enumerate(support.varying):
        rows[:, founder] = (np.arange(len(support)) >> bit) & 1
    return np.sum(np.where(rows == 1, alt, ref), axis=1)
