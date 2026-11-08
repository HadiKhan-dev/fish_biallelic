"""Painting emissions and design effect reused by family phase (P0 PR04, FP01).

Parallel loops own independent sample/pair cells; the caller supplies the thread
team through the runtime. Site sums retain ascending order inside whole bins.
"""

import math
from typing import Any, Protocol, cast

import numpy as np
from numba import prange, types

from ...contracts.likelihoods import gather_likelihoods
from ...contracts.painting import state_alt_probabilities
from ...core.kernels import kernel
from .parameters import INFORMATIVE_SPREAD, PaintingParameters

# Baseline release-manifest signatures; all calls remain lazily compiled.
_OVERLOADS = {
    "informative_sites": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=True, aligned=True),
        ),
    ),
    "_gather_kernel": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ),
        (
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
        ),
    ),
    "_emission_kernel": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.float64,
            types.float64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=True, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=True, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.float64,
            types.float64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.Array(types.boolean, 2, "C", readonly=True, aligned=True),
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.Array(types.int64, 1, "C", readonly=True, aligned=True),
            types.float64,
            types.float64,
        ),
    ),
    "pack_emissions": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.int64,
            types.int64,
        ),
    ),
    "unpack_emissions": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.int64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=True, aligned=True),
            types.int64,
        ),
    ),
}


@kernel(overloads=_OVERLOADS["informative_sites"], cache=True, parallel=True, nogil=True)
def informative_sites(likelihoods, observed):
    """bool[N, L]: observed and max - min of the normalised GL row > 16 eps."""
    n_samples, n_sites, _ = likelihoods.shape
    result = np.empty((n_samples, n_sites), dtype=np.bool_)
    for row in prange(n_samples * n_sites):
        sample, site = row // n_sites, row % n_sites
        a = likelihoods[sample, site, 0]
        b = likelihoods[sample, site, 1]
        c = likelihoods[sample, site, 2]
        result[sample, site] = observed[sample, site] and max(a, b, c) - min(a, b, c) > INFORMATIVE_SPREAD
    return result


@kernel(overloads=_OVERLOADS["_gather_kernel"], cache=True, parallel=True, nogil=True)
def _gather_kernel(likelihoods, observed, sites):
    n_samples = likelihoods.shape[0]
    values = np.empty((n_samples, len(sites), 3), dtype=np.float64)
    flags = np.empty((n_samples, len(sites)), dtype=np.bool_)
    for sample in prange(n_samples):
        for site in range(len(sites)):
            source = sites[site]
            values[sample, site, 0] = likelihoods[sample, source, 0]
            values[sample, site, 1] = likelihoods[sample, source, 1]
            values[sample, site, 2] = likelihoods[sample, source, 2]
            flags[sample, site] = observed[sample, source]
    return values, flags


def design_effect(
    likelihoods: np.ndarray, observed: np.ndarray, bin_starts: np.ndarray, correlation: float
) -> np.ndarray:
    """float64[N, B]: 1 + (n - 1) r for the n informative sites (informative_sites) of each sample in each bin (1 when
    n <= 1), r = ``correlation``; ``likelihoods``/``observed`` are gathered at the component's selected sites and
    ``bin_starts`` are the bins' first selected sites (release 2.14). The variance inflation of a sum of n
    equal-variance terms with common correlation r (Kish's design effect), used as a tuned working weight of a bin's
    log emission: under that assumption the bin counts as n / (1 + (n - 1) r) variance-equivalent sites. Not an
    estimated dependence correction; the weight tends to 1 as n -> 1, and a bin without informative sites has 1."""
    counts = np.add.reduceat(
        informative_sites(likelihoods, observed),
        np.asarray(bin_starts, dtype=np.int64),
        axis=1,
        dtype=np.int64,
    )
    return 1.0 + np.maximum(counts - 1, 0) * float(correlation)


def gather_sites(
    likelihoods: Any, observed: Any, sites: np.ndarray, samples: slice = slice(None)
) -> tuple[np.ndarray, np.ndarray]:
    """C-contiguous float64[n, len(sites), 3] GL rows and bool[n, len(sites)] flags of ``samples``.

    ``sites`` are strictly increasing evidence site indices: a consecutive run is
    copied as one slice, other selections with a parallel gather.
    """
    sites = np.asarray(sites, dtype=np.int64)
    if not isinstance(likelihoods, np.ndarray) or not isinstance(observed, np.ndarray):
        values = gather_likelihoods(likelihoods, sites, samples)
        flags = (
            np.ascontiguousarray(observed[samples][:, sites], dtype=np.bool_)
            if isinstance(observed, np.ndarray)
            else observed.gather(samples, sites)
        )
        return values, flags
    if len(sites) == 0 or int(sites[-1]) - int(sites[0]) + 1 == len(sites):
        window = slice(int(sites[0]), int(sites[-1]) + 1) if len(sites) else slice(0, 0)
        return (
            np.ascontiguousarray(likelihoods[samples, window], dtype=np.float64),
            np.ascontiguousarray(observed[samples, window], dtype=np.bool_),
        )
    return _gather_kernel(likelihoods[samples], observed[samples], sites)


# The running product of a bin's site probabilities is folded into its log below this value (no underflow: every
# factor is at least min(1, e^floor) and factors below this value are logged directly).
_RESCALE_BELOW = 1e-150


@kernel(overloads=_OVERLOADS["_emission_kernel"], cache=True, parallel=True, nogil=True)
def _emission_kernel(
    likelihoods, observed, alt_probability, background_frequency, bin_starts, bin_stops, robustness_epsilon, log_floor
):
    """float64[N, S*S, B] binned log emissions, one task per sample x unordered pair.

    The ordered states (a, b) and (b, a) have equal emissions (the genotype law
    is symmetric in the two homologues), so each unordered pair is evaluated
    once and stored at both.
    """
    n_samples = likelihoods.shape[0]
    n_classes = alt_probability.shape[0]
    n_states = n_classes + 1
    n_pairs = n_states * (n_states + 1) // 2
    n_bins = len(bin_starts)
    result = np.empty((n_samples, n_states * n_states, n_bins), dtype=np.float64)
    uniform_log = max(math.log(1.0 / 3.0), log_floor)
    # One log per bin (release 4.7): the clamped site probabilities are multiplied and the product's log taken once,
    # rescaled before it can underflow; log(max(p, e^floor)) summed over sites, in another rounding.
    floor_probability = math.exp(log_floor)
    for task in prange(n_samples * n_pairs):
        sample = task // n_pairs
        # Unordered pairs first <= second in row-major upper-triangle order.
        pair = task % n_pairs
        first = 0
        while pair >= n_states - first:
            pair -= n_states - first
            first += 1
        second = first + pair
        state = first * n_states + second
        mirror = second * n_states + first
        for bin_index in range(n_bins):
            logs = 0.0
            product = 1.0
            count = 0
            for site in range(bin_starts[bin_index], bin_stops[bin_index]):
                if not observed[sample, site]:
                    continue
                first_q = background_frequency[site] if first == n_classes else alt_probability[first, site]
                second_q = background_frequency[site] if second == n_classes else alt_probability[second, site]
                if first == second and first < n_classes:
                    # Two homologues of one class share one latent allele.
                    p0 = 1.0 - first_q
                    p1 = 0.0
                    p2 = first_q
                else:
                    p0 = (1.0 - first_q) * (1.0 - second_q)
                    p2 = first_q * second_q
                    p1 = 1.0 - p0 - p2
                likelihood = (
                    likelihoods[sample, site, 0] * p0
                    + likelihoods[sample, site, 1] * p1
                    + likelihoods[sample, site, 2] * p2
                )
                probability = (1.0 - robustness_epsilon) * likelihood + robustness_epsilon / 3.0
                if probability < floor_probability:  # also p <= 0; NaN propagates, as before
                    probability = floor_probability
                count += 1
                if probability < _RESCALE_BELOW:
                    logs += math.log(probability)  # only under a floor below log(_RESCALE_BELOW)
                    continue
                product *= probability
                if product < _RESCALE_BELOW:
                    logs += math.log(product)
                    product = 1.0
            score = logs + math.log(product) - count * uniform_log
            result[sample, state, bin_index] = score
            result[sample, mirror, bin_index] = score
    return result


def binned_emissions(
    likelihoods: np.ndarray,
    observed: np.ndarray,
    alt_probability: np.ndarray,
    background_frequency: np.ndarray,
    bin_starts: np.ndarray,
    bin_stops: np.ndarray,
    *,
    robustness_epsilon: float,
    log_floor: float,
    sample_block: int = 0,
    before_block=None,
) -> np.ndarray:
    """float64[N, S*S, B] log emissions of contiguous site runs [bin_starts, bin_stops).

    ``likelihoods`` float64[N, L, 3] normalised GL rows and ``observed``
    bool[N, L] at the component's selected sites; ``alt_probability``
    float64[C, L] and ``background_frequency`` float64[L] from the states.
    O(N S^2 L / 2). ``sample_block`` > 0 evaluates that many samples at a time
    (values are identical) and calls ``before_block()`` before each block,
    e.g. to let a chromosome worker's CPU lease grow.
    """
    arguments = (
        np.ascontiguousarray(alt_probability, dtype=np.float64),
        np.ascontiguousarray(background_frequency, dtype=np.float64),
        np.ascontiguousarray(bin_starts, dtype=np.int64),
        np.ascontiguousarray(bin_stops, dtype=np.int64),
        float(robustness_epsilon),
        float(log_floor),
    )
    n_samples = likelihoods.shape[0]
    if sample_block <= 0 or sample_block >= n_samples:
        return _emission_kernel(likelihoods, observed, *arguments)
    n_states = alt_probability.shape[0] + 1
    result = np.empty((n_samples, n_states * n_states, len(bin_starts)), dtype=np.float64)
    for start in range(0, n_samples, sample_block):
        if before_block is not None:
            before_block()
        stop = min(start + sample_block, n_samples)
        result[start:stop] = _emission_kernel(likelihoods[start:stop], observed[start:stop], *arguments)
    return result


@kernel(overloads=_OVERLOADS["pack_emissions"], cache=True, parallel=True, nogil=True)
def pack_emissions(emissions, packed, sample_start, n_states):
    """Store the upper triangle (first <= second) of symmetric ordered-pair emissions
    into ``packed`` [N, S(S+1)/2, B] at rows sample_start + i."""
    for task in prange(emissions.shape[0] * n_states):
        sample, first = task // n_states, task % n_states
        row_start = first * n_states - first * (first - 1) // 2
        for second in range(first, n_states):
            for block in range(emissions.shape[2]):
                packed[sample_start + sample, row_start + second - first, block] = emissions[
                    sample, first * n_states + second, block
                ]


@kernel(overloads=_OVERLOADS["unpack_emissions"], cache=True, parallel=True, nogil=True)
def unpack_emissions(packed, n_states):
    """float64[N, S*S, B] ordered-pair emissions from their packed upper triangle."""
    output = np.empty((packed.shape[0], n_states * n_states, packed.shape[2]))
    for task in prange(packed.shape[0] * n_states):
        sample, first = task // n_states, task % n_states
        row_start = first * n_states - first * (first - 1) // 2
        for second in range(first, n_states):
            for block in range(packed.shape[2]):
                value = packed[sample, row_start + second - first, block]
                output[sample, first * n_states + second, block] = value
                output[sample, second * n_states + first, block] = value
    return output


__all__ = [
    "informative_sites",
    "design_effect",
    "gather_sites",
    "binned_emissions",
    "pack_emissions",
    "unpack_emissions",
    "state_alt_probabilities",
    "ArraySource",
    "EmissionBuilder",
    "tempered_window",
    "tempered_emissions",
]


class ArraySource(Protocol):
    """Shape of a read-only observation source; a stage owns any physical reader."""

    @property
    def shape(self) -> tuple[int, ...]: ...


class EmissionBuilder(Protocol):
    """Explicit whole-bin emission capability; sample and site axes stay canonical."""

    def __call__(
        self,
        likelihoods: ArraySource,
        observed: ArraySource,
        sites: np.ndarray,
        samples: slice,
        alt_probability: np.ndarray,
        background_frequency: np.ndarray,
        bin_starts: np.ndarray,
        bin_stops: np.ndarray,
        config: PaintingParameters,
        *,
        parts: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def tempered_window(
    values: np.ndarray,
    flags: np.ndarray,
    alt_probability: np.ndarray,
    background_frequency: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    whole_bin_starts: np.ndarray,
    config: PaintingParameters,
    *,
    part_bins: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Temper complete bins or parts by the whole bin's design effect, in site order.

    All indices are local to this whole-bin window. Returned emissions and informative
    masks are owned; inputs are unchanged. Cost O(samples * selected sites * S**2).
    """
    raw = binned_emissions(
        values,
        flags,
        alt_probability,
        background_frequency,
        starts,
        stops,
        robustness_epsilon=config.robustness_epsilon,
        log_floor=config.emission_log_floor,
    )
    design = design_effect(values, flags, whole_bin_starts, config.emission_site_correlation)
    if part_bins is not None:
        design = design[:, part_bins]
    raw /= design[:, None, :]
    return raw, informative_sites(values, flags)


def tempered_emissions(
    likelihoods: ArraySource,
    observed: ArraySource,
    sites: np.ndarray,
    samples: slice,
    alt_probability: np.ndarray,
    background_frequency: np.ndarray,
    bin_starts: np.ndarray,
    bin_stops: np.ndarray,
    config: PaintingParameters,
    *,
    parts: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resident-array emissions, observed and informative masks for a contiguous sample batch.

    Parts use their whole bin's design effect. No storage or live resource access;
    a stage can provide an equivalent EmissionBuilder over physical windows.
    """
    values, flags = gather_sites(cast(np.ndarray, likelihoods), cast(np.ndarray, observed), sites, samples)
    starts, stops = (bin_starts, bin_stops) if parts is None else parts[:2]
    raw, informative = tempered_window(
        values,
        flags,
        alt_probability,
        background_frequency,
        starts,
        stops,
        bin_starts,
        config,
        part_bins=None if parts is None else parts[2],
    )
    return raw, flags, informative
