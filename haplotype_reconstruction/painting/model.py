"""Ragged diploid founder HMM with an explicit unknown state."""
from __future__ import annotations


from dataclasses import dataclass
from enum import IntEnum
import math
from pathlib import Path
from typing import Any
import numpy as np
import numba
from numba import njit, prange
from.import evidence as painting_evidence


T09_EMISSION_CACHE_MAX_BYTES = 1024 ** 3  # Bounded reusable T10 evidence, per chromosome.


RAGGED_MIN_WORKING_MEMORY_BYTES = 512 * 1024 * 1024


RAGGED_MEMORY_RESERVE_BYTES = 4 * 1024 * 1024 * 1024


class PaintingTrackStatus(IntEnum):
    """Scientific meaning of one released painting track and bin."""

    INELIGIBLE_NO_EVIDENCE = 0
    LOW_POSTERIOR_ABSTENTION = 1
    SINGLETON_NAMED = 2
    POOLED_EQUIVALENCE = 3
    BACKGROUND = 4
    UNANCHORED_TRAJECTORY = 5


def available_process_memory_bytes() -> int | None:
    """Return a conservative current process/cgroup memory allowance."""

    candidates = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass

    for maximum_path, current_path in (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
         Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")),
    ):
        try:
            maximum_text = maximum_path.read_text().strip()
            if maximum_text == "max":
                continue
            maximum = int(maximum_text)
            current = int(current_path.read_text().strip())
            # Very large v1 sentinels mean effectively unlimited.
            if 0 < maximum < (1 << 60):
                candidates.append(max(0, maximum - current))
        except (OSError, ValueError):
            continue
    return min(candidates) if candidates else None


def resolve_ragged_working_memory_budget(
        estimated_bytes_per_sample: int,
        fixed_working_bytes: int,
        thread_count: int,
        *,
        requested_bytes: int | None=None,
        available_bytes: int | None=None,
) -> int:
    """Choose a bounded scheduling budget large enough to occupy threads.

    ``requested_bytes`` is an explicit runtime override. Otherwise the target
    is one concurrent sample per Numba thread, capped by currently available
    process/cgroup memory after a 10% (at least 4 GiB, at most half) reserve.
    """

    for name, value in (
        ("estimated_bytes_per_sample", estimated_bytes_per_sample),
        ("fixed_working_bytes", fixed_working_bytes),
        ("thread_count", thread_count),
    ):
        if isinstance(value, bool) or int(value) != value or value < (0 if name == "fixed_working_bytes" else 1):
            raise ValueError(f"{name} is invalid")
    estimated_bytes_per_sample = int(estimated_bytes_per_sample)
    fixed_working_bytes = int(fixed_working_bytes)
    thread_count = int(thread_count)
    if requested_bytes is not None:
        if (isinstance(requested_bytes, bool)
                or int(requested_bytes) != requested_bytes
                or requested_bytes < 1):
            raise ValueError("requested_bytes must be a positive integer")
        return int(requested_bytes)
    if available_bytes is None:
        available_bytes = available_process_memory_bytes()
    if available_bytes is None:
        return max(
            RAGGED_MIN_WORKING_MEMORY_BYTES,
            fixed_working_bytes + estimated_bytes_per_sample,
        )
    if (isinstance(available_bytes, bool)
            or int(available_bytes) != available_bytes
            or available_bytes < 1):
        raise ValueError("available_bytes must be a positive integer")
    available_bytes = int(available_bytes)
    reserve = min(
        available_bytes // 2,
        max(RAGGED_MEMORY_RESERVE_BYTES, available_bytes // 10),
    )
    usable = max(1, available_bytes - reserve)
    thread_target = (
        fixed_working_bytes + thread_count * estimated_bytes_per_sample
    )
    return min(
        usable,
        max(RAGGED_MIN_WORKING_MEMORY_BYTES, thread_target),
    )


def choose_ragged_batch_size(
        sample_count: int,
        requested_batch_size: int,
        thread_count: int,
        estimated_bytes_per_sample: int,
        fixed_working_bytes: int,
        working_memory_budget_bytes: int,
) -> int:
    """Choose the actual bounded batch size under an injected byte budget."""

    values = (
        sample_count, requested_batch_size, thread_count,
        estimated_bytes_per_sample, working_memory_budget_bytes,
    )
    if any(isinstance(value, bool) or int(value) != value or value < 1
           for value in values):
        raise ValueError("batch sizing inputs must be positive integers")
    if (isinstance(fixed_working_bytes, bool)
            or int(fixed_working_bytes) != fixed_working_bytes
            or fixed_working_bytes < 0):
        raise ValueError("fixed_working_bytes must be a non-negative integer")
    remaining = max(0, int(working_memory_budget_bytes) - int(fixed_working_bytes))
    memory_limited = max(1, remaining // int(estimated_bytes_per_sample))
    desired = max(
        int(requested_batch_size), min(int(thread_count), int(sample_count))
    )
    return min(int(sample_count), desired, memory_limited)


@dataclass(frozen=True)
class RaggedStateSpace:
    """Per-site allele evidence for trajectory classes plus BACKGROUND."""

    positions: np.ndarray
    site_indices: np.ndarray
    q: np.ndarray
    called: np.ndarray
    active: np.ndarray
    equivalence_classes: tuple[tuple[int, ...], ...]
    background_index: int
    background_alt_probability: np.ndarray

    @property
    def unknown_index(self) -> int:
        """Compatibility alias; new code should use ``background_index``."""

        return self.background_index


@dataclass(frozen=True)
class RaggedBinning:
    """Bins that never cross a change in the named-founder active set."""

    indices: tuple[np.ndarray, ...]
    centers: np.ndarray
    edges: np.ndarray
    sizes: np.ndarray


def normalise_genotype_likelihoods(genotype_likelihoods: np.ndarray) -> np.ndarray:
    """Normalize raw non-negative GL rows; zero-mass rows become uniform."""

    raw = np.asarray(genotype_likelihoods)
    if raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError("genotype likelihoods must have shape (samples, sites, 3)")
    if raw.dtype in (np.dtype("float32"), np.dtype("float64")) and raw.size >= 196608:
        result, invalid = painting_evidence.normalize_rows(raw)
        if invalid:
            raise ValueError("genotype likelihoods must be finite and non-negative")
        return result
    evidence = np.asarray(raw, dtype=np.float64)
    if np.any(~np.isfinite(evidence)) or np.any(evidence < 0.0):
        raise ValueError("genotype likelihoods must be finite and non-negative")
    totals = np.sum(evidence, axis=2, keepdims=True)
    result = np.full(evidence.shape, 1.0 / 3.0, dtype=np.float64)
    np.divide(evidence, totals, out=result, where=totals > 0.0)
    return result


def build_ragged_state_space(
        panel,
        retained: np.ndarray,
        *,
        active: np.ndarray | None=None,
) -> RaggedStateSpace:
    """Construct fixed whole-component trajectory classes plus BACKGROUND.

    Sites lacking every named-founder call are excluded: q=0.5 diagonal versus
    off-diagonal structure must not create ancestry evidence from no founder
    observation. Exact duplicate {-1,0,1} trajectories are pooled into one
    equivalence class, preventing arbitrary duplicate-row hard calls.
    """

    retained = np.asarray(retained, dtype=np.bool_)
    panel_positions = np.asarray(panel.positions)
    full_called = np.asarray(panel.called, dtype=np.bool_)
    full_q = np.asarray(panel.q, dtype=np.float64)
    if retained.shape != panel_positions.shape:
        raise ValueError("retained mask must match founder-panel positions")
    site_used = retained & np.any(full_called, axis=0)
    site_indices = np.flatnonzero(site_used)

    discrete = np.full(full_called.shape, -1, dtype=np.int8)
    if np.any(full_called):
        called_values = full_q[full_called]
        if np.any(~np.isin(called_values, (0.0, 1.0))):
            raise ValueError("called founder alleles must be hard 0/1 values")
        discrete[full_called] = called_values.astype(np.int8)
    explicit_active = None
    if active is not None:
        explicit_active = np.asarray(active, dtype=np.bool_)
        if explicit_active.shape != full_called.shape:
            raise ValueError("active mask must match the full founder panel")
    class_members = []
    representatives = []
    seen = {}
    for founder in range(discrete.shape[0]):
        signature = discrete[founder, site_used].tobytes()
        if explicit_active is not None:
            signature += explicit_active[founder, site_used].tobytes()
        class_index = seen.get(signature)
        if class_index is None:
            seen[signature] = len(class_members)
            class_members.append([founder])
            representatives.append(founder)
        else:
            class_members[class_index].append(founder)

    representatives = np.asarray(representatives, dtype=np.int64)
    called = full_called[representatives][:, site_used].copy()
    q = full_q[representatives][:, site_used].copy()
    unique_called = full_called[representatives][:, site_used]
    unique_q = full_q[representatives][:, site_used]
    n_called = np.sum(unique_called, axis=0)
    alt_called = np.sum(unique_q * unique_called, axis=0)
    background_frequency = (1.0 + alt_called) / (2.0 + n_called)
    q[~called] = np.broadcast_to(background_frequency, q.shape)[~called]

    if explicit_active is None:
        active_subset = np.ones(called.shape, dtype=np.bool_)
    else:
        active_subset = explicit_active[representatives][:, site_used].copy()
    return RaggedStateSpace(
        panel_positions[site_used], site_indices, q, called, active_subset,
        tuple(tuple(members) for members in class_members), len(class_members),
        background_frequency,
    )


def build_ragged_bins(
        positions: np.ndarray,
        named_active: np.ndarray,
        snps_per_bin: int,
) -> RaggedBinning:
    """Build approximate SNP-count bins without crossing activity boundaries."""

    positions = np.asarray(positions)
    active = np.asarray(named_active, dtype=np.bool_)
    if positions.ndim != 1 or active.ndim != 2 or active.shape[1] != len(positions):
        raise ValueError("active mask must be founder-by-position aligned")
    if isinstance(snps_per_bin, bool) or int(snps_per_bin) != snps_per_bin:
        raise ValueError("snps_per_bin must be a positive integer")
    snps_per_bin = int(snps_per_bin)
    if snps_per_bin < 1:
        raise ValueError("snps_per_bin must be a positive integer")
    if len(positions) == 0:
        return RaggedBinning((), np.array([]), np.array([], dtype=np.int64),
                             np.array([], dtype=np.int64))

    # Accumulate site boundaries in native vector operations, using O(L)
    # temporary memory rather than a K-by-L difference grid.
    changed = np.zeros(len(positions) - 1, dtype=np.bool_)
    for trajectory in active:
        changed |= trajectory[1:] != trajectory[:-1]
    boundaries = np.r_[0, np.flatnonzero(changed) + 1, len(positions)]

    bins: list[np.ndarray] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        length = stop - start
        # Match the established painter's approximate bin count within each
        # constant-state run.  The complete-data path bypasses this function.
        n_bins = max(1, length // snps_per_bin)
        bins.extend(np.array_split(np.arange(start, stop), n_bins))

    centers = np.asarray(
        [np.mean(positions[index]) for index in bins], dtype=np.float64
    )
    edges = np.empty(len(bins) + 1, dtype=np.int64)
    edges[0] = int(positions[0])
    for index in range(len(bins) - 1):
        edges[index + 1] = (
            int(positions[bins[index][-1]])
            + int(positions[bins[index + 1][0]]) + 1
        ) // 2
    edges[-1] = int(positions[-1]) + 1
    sizes = np.asarray([len(index) for index in bins], dtype=np.int64)
    return RaggedBinning(tuple(bins), centers, edges, sizes)


def _state_definitions(n_states: int) -> np.ndarray:
    first, second = np.unravel_index(
        np.arange(n_states * n_states), (n_states, n_states)
    )
    return np.stack((first, second), axis=1).astype(np.int32)


@njit(cache=True, parallel=True, fastmath=False)
def _ragged_binned_emission_kernel(
        evidence, observed, named_q, background_q, active,
        bin_starts, bin_stops, robustness_epsilon, log_floor):
    """Direct parallel sample-by-diplotype binned emission kernel."""

    n_samples, _, _ = evidence.shape
    n_named, _ = named_q.shape
    n_states = n_named + 1
    n_diplotypes = n_states * n_states
    n_bins = len(bin_starts)
    result = np.empty((n_samples, n_diplotypes, n_bins), dtype=np.float64)
    uniform_log = max(math.log(1.0 / 3.0), log_floor)
    for task in prange(n_samples * n_diplotypes):
        sample = task // n_diplotypes
        state = task % n_diplotypes
        first = state // n_states
        second = state % n_states
        for bin_index in range(n_bins):
            if ((first < n_named and not active[first, bin_index])
                    or (second < n_named and not active[second, bin_index])):
                result[sample, state, bin_index] = -np.inf
                continue
            score = 0.0
            for site in range(bin_starts[bin_index], bin_stops[bin_index]):
                if not observed[sample, site]:
                    continue
                first_q = (
                    background_q[site] if first == n_named
                    else named_q[first, site]
                )
                second_q = (
                    background_q[site] if second == n_named
                    else named_q[second, site]
                )
                if first == second and first < n_named:
                    p0 = 1.0 - first_q
                    p1 = 0.0
                    p2 = first_q
                else:
                    p0 = (1.0 - first_q) * (1.0 - second_q)
                    p2 = first_q * second_q
                    p1 = 1.0 - p0 - p2
                likelihood = (
                    evidence[sample, site, 0] * p0
                    + evidence[sample, site, 1] * p1
                    + evidence[sample, site, 2] * p2
                )
                probability = (
                    (1.0 - robustness_epsilon) * likelihood
                    + robustness_epsilon / 3.0
                )
                if probability <= 0.0:
                    log_probability = log_floor
                else:
                    log_probability = math.log(probability)
                    if log_probability < log_floor:
                        log_probability = log_floor
                score += log_probability - uniform_log
            result[sample, state, bin_index] = score
    return result


def calculate_ragged_binned_emissions(
        genotype_likelihoods: np.ndarray,
        observed: np.ndarray,
        state_space: RaggedStateSpace,
        binning: RaggedBinning,
        *, robustness_epsilon: float=0.01,
        log_floor: float=-50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact emissions in O(N*S²*L) with sample×state parallelism."""

    if not 0.0 <= robustness_epsilon <= 1.0:
        raise ValueError("robustness_epsilon must lie in [0, 1]")
    if log_floor > 0.0:
        raise ValueError("log_floor must be non-positive")
    evidence = normalise_genotype_likelihoods(genotype_likelihoods)
    observed = np.asarray(observed, dtype=np.bool_)
    if observed.shape != evidence.shape[:2]:
        raise ValueError("observed mask must match samples and sites")
    if evidence.shape[1] != len(state_space.positions):
        raise ValueError("evidence and state space disagree on sites")

    bin_starts = np.empty(len(binning.indices), dtype=np.int64)
    bin_stops = np.empty(len(binning.indices), dtype=np.int64)
    active = np.empty(
        (state_space.q.shape[0], len(binning.indices)), dtype=np.bool_
    )
    for bin_index, indices in enumerate(binning.indices):
        start = int(indices[0])
        stop = int(indices[-1]) + 1
        if not np.array_equal(indices, np.arange(start, stop)):
            raise ValueError("ragged emission bins must contain consecutive sites")
        bin_starts[bin_index] = start
        bin_stops[bin_index] = stop
        active[:, bin_index] = state_space.active[:, start]
    result = _ragged_binned_emission_kernel(
        np.ascontiguousarray(evidence),
        np.ascontiguousarray(observed),
        np.ascontiguousarray(state_space.q),
        np.ascontiguousarray(state_space.background_alt_probability),
        active,
        bin_starts,
        bin_stops,
        float(robustness_epsilon),
        float(log_floor),
    )
    return result, _state_definitions(state_space.q.shape[0] + 1)


@njit(cache=True, parallel=True, fastmath=False)
def viterbi_label_grid(
        emissions, centers, n_states, recomb_rate,
        switch_penalty_per_snp, snps_per_bin, double_recomb_factor,
        genetic_distances=None):
    """Exact O(B*S²) Viterbi for ordered two-track Hamming transitions.

    Exact mathematical ties use a canonical rule: the smallest previous
    flat-state index wins. The legacy uncentered fastmath reference can drift
    by a few ULP at exact ties. Per-bin score centering prevents
    long-path overflow without affecting paths or tie order.
    """

    n_samples, n_diplotypes, n_bins = emissions.shape
    back = np.full(
        (n_samples, n_bins, n_diplotypes), -1, dtype=np.int32
    )
    final_scores = np.empty((n_samples, n_diplotypes), dtype=np.float64)
    log_n_minus_one = math.log(float(n_states - 1)) if n_states > 1 else 0.0
    for sample in prange(n_samples):
        previous = emissions[sample,:, 0].copy()
        previous -= np.max(previous)
        current_scores = np.empty(n_diplotypes, dtype=np.float64)
        row_best_value = np.empty(n_states, dtype=np.float64)
        row_second_value = np.empty(n_states, dtype=np.float64)
        row_best_index = np.empty(n_states, dtype=np.int32)
        row_second_index = np.empty(n_states, dtype=np.int32)
        column_best_value = np.empty(n_states, dtype=np.float64)
        column_second_value = np.empty(n_states, dtype=np.float64)
        column_best_index = np.empty(n_states, dtype=np.int32)
        column_second_index = np.empty(n_states, dtype=np.int32)

        for bin_index in range(1, n_bins):
            distance = centers[bin_index] - centers[bin_index - 1]
            if distance < 1.0:
                distance = 1.0
            theta = min(0.5, max(1e-15, distance * recomb_rate))
            if genetic_distances is not None:
                theta = min(0.5, max(1e-15, genetic_distances[bin_index - 1]))
            log_switch = math.log(theta) - switch_penalty_per_snp * snps_per_bin
            log_stay = math.log(1.0 - theta)
            cost_zero = 2.0 * log_stay
            cost_one = log_switch + log_stay - log_n_minus_one
            cost_two = (
                double_recomb_factor * log_switch - 2.0 * log_n_minus_one
            )

            for row in range(n_states):
                best_value = -np.inf
                second_value = -np.inf
                best_index = -1
                second_index = -1
                for column in range(n_states):
                    index = row * n_states + column
                    value = previous[index]
                    if (value > best_value
                            or (value == best_value and index < best_index)):
                        second_value = best_value
                        second_index = best_index
                        best_value = value
                        best_index = index
                    elif (value > second_value
                          or (value == second_value and index < second_index)):
                        second_value = value
                        second_index = index
                row_best_value[row] = best_value
                row_second_value[row] = second_value
                row_best_index[row] = best_index
                row_second_index[row] = second_index

            for column in range(n_states):
                best_value = -np.inf
                second_value = -np.inf
                best_index = -1
                second_index = -1
                for row in range(n_states):
                    index = row * n_states + column
                    value = previous[index]
                    if (value > best_value
                            or (value == best_value and index < best_index)):
                        second_value = best_value
                        second_index = best_index
                        best_value = value
                        best_index = index
                    elif (value > second_value
                          or (value == second_value and index < second_index)):
                        second_value = value
                        second_index = index
                column_best_value[column] = best_value
                column_second_value[column] = second_value
                column_best_index[column] = best_index
                column_second_index[column] = second_index

            for current_row in range(n_states):
                complement_best_value = -np.inf
                complement_second_value = -np.inf
                complement_best_index = -1
                complement_second_index = -1
                complement_best_column = -1
                for previous_column in range(n_states):
                    index = column_best_index[previous_column]
                    value = column_best_value[previous_column]
                    if index // n_states == current_row:
                        index = column_second_index[previous_column]
                        value = column_second_value[previous_column]
                    if (value > complement_best_value
                            or (value == complement_best_value
                                and index < complement_best_index)):
                        complement_second_value = complement_best_value
                        complement_second_index = complement_best_index
                        complement_best_value = value
                        complement_best_index = index
                        complement_best_column = previous_column
                    elif (value > complement_second_value
                          or (value == complement_second_value
                              and index < complement_second_index)):
                        complement_second_value = value
                        complement_second_index = index

                for current_column in range(n_states):
                    current = current_row * n_states + current_column
                    best_value = previous[current] + cost_zero
                    best_index = current

                    column_index = column_best_index[current_column]
                    column_value = column_best_value[current_column]
                    if column_index // n_states == current_row:
                        column_index = column_second_index[current_column]
                        column_value = column_second_value[current_column]
                    row_index = row_best_index[current_row]
                    row_value = row_best_value[current_row]
                    if row_index % n_states == current_column:
                        row_index = row_second_index[current_row]
                        row_value = row_second_value[current_row]
                    one_value = column_value
                    one_index = column_index
                    if (row_value > one_value
                            or (row_value == one_value and row_index < one_index)):
                        one_value = row_value
                        one_index = row_index
                    candidate = one_value + cost_one
                    if (candidate > best_value
                            or (candidate == best_value and one_index < best_index)):
                        best_value = candidate
                        best_index = one_index

                    complement_value = complement_best_value
                    complement_index = complement_best_index
                    if complement_best_column == current_column:
                        complement_value = complement_second_value
                        complement_index = complement_second_index
                    candidate = complement_value + cost_two
                    if (candidate > best_value
                            or (candidate == best_value
                                and complement_index < best_index)):
                        best_value = candidate
                        best_index = complement_index

                    current_scores[current] = (
                        best_value + emissions[sample, current, bin_index]
                    )
                    back[sample, bin_index, current] = best_index
            current_scores -= np.max(current_scores)
            previous, current_scores = current_scores, previous
        final_scores[sample] = previous

    result = np.empty((n_samples, 2, n_bins), dtype=np.int32)
    for sample in prange(n_samples):
        current = int(np.argmax(final_scores[sample]))
        result[sample, 0, n_bins - 1] = current // n_states
        result[sample, 1, n_bins - 1] = current % n_states
        for bin_index in range(n_bins - 1, 0, -1):
            current = back[sample, bin_index, current]
            result[sample, 0, bin_index - 1] = current // n_states
            result[sample, 1, bin_index - 1] = current % n_states
    return result


@njit(cache=True, parallel=True, fastmath=False)
def posterior_class_summaries(
        emissions, centers, n_states, recomb_rate, switch_penalty_per_snp,
        snps_per_bin, double_recomb_factor, viterbi_state_grid,
        public_state_ids, genetic_distances=None):
    """Exact O(B*S²) scaled forward-backward unordered-class summaries.

    The Hamming-category transition contraction uses the same-cell value,
    row/column sums excluding that cell, and the complementary submatrix sum.
    Forward messages are normalized at every bin and backward messages reuse
    those scales. Only one sample's B*S² forward table is live per worker.
    """

    n_samples, n_diplotypes, n_bins = emissions.shape
    maximum_class_mass = np.zeros((n_samples, n_bins), dtype=np.float64)
    class_entropy = np.zeros((n_samples, n_bins), dtype=np.float64)
    background_mass = np.zeros((n_samples, n_bins), dtype=np.float64)
    public_unknown_mass = np.zeros((n_samples, n_bins), dtype=np.float64)
    viterbi_public_class_mass = np.zeros((n_samples, n_bins), dtype=np.float64)
    log_n_minus_one = math.log(float(n_states - 1)) if n_states > 1 else 0.0
    for sample in prange(n_samples):
        forward = np.zeros((n_bins, n_states, n_states), dtype=np.float64)
        scales = np.empty(n_bins, dtype=np.float64)
        emission_max = np.max(emissions[sample,:, 0])
        scale = 0.0
        for row in range(n_states):
            for column in range(n_states):
                index = row * n_states + column
                value = math.exp(emissions[sample, index, 0] - emission_max)
                forward[0, row, column] = value
                scale += value
        scales[0] = scale
        forward[0] /= scale

        for bin_index in range(1, n_bins):
            distance = centers[bin_index] - centers[bin_index - 1]
            if distance < 1.0:
                distance = 1.0
            theta = min(0.5, max(1e-15, distance * recomb_rate))
            if genetic_distances is not None:
                theta = min(0.5, max(1e-15, genetic_distances[bin_index - 1]))
            log_switch = math.log(theta) - switch_penalty_per_snp * snps_per_bin
            log_stay = math.log(1.0 - theta)
            cost_zero = 2.0 * log_stay
            cost_one = log_switch + log_stay - log_n_minus_one
            cost_two = (
                double_recomb_factor * log_switch - 2.0 * log_n_minus_one
            )
            maximum_cost = max(cost_zero, cost_one, cost_two)
            weight_zero = math.exp(cost_zero - maximum_cost)
            weight_one = math.exp(cost_one - maximum_cost)
            weight_two = math.exp(cost_two - maximum_cost)

            row_sums = np.zeros(n_states, dtype=np.float64)
            column_sums = np.zeros(n_states, dtype=np.float64)
            total = 0.0
            for row in range(n_states):
                for column in range(n_states):
                    value = forward[bin_index - 1, row, column]
                    row_sums[row] += value
                    column_sums[column] += value
                    total += value
            emission_max = np.max(emissions[sample,:, bin_index])
            scale = 0.0
            for row in range(n_states):
                for column in range(n_states):
                    cell = forward[bin_index - 1, row, column]
                    one_change = max(
                        0.0,
                        row_sums[row] + column_sums[column] - 2.0 * cell,
                    )
                    two_changes = max(
                        0.0,
                        total - row_sums[row] - column_sums[column] + cell,
                    )
                    prediction = (
                        weight_zero * cell
                        + weight_one * one_change
                        + weight_two * two_changes
                    )
                    index = row * n_states + column
                    emission_weight = math.exp(
                        emissions[sample, index, bin_index] - emission_max
                    )
                    value = prediction * emission_weight
                    forward[bin_index, row, column] = value
                    scale += value
            scales[bin_index] = scale
            forward[bin_index] /= scale

        backward = np.ones((n_states, n_states), dtype=np.float64)
        for bin_index in range(n_bins - 1, -1, -1):
            posterior_total = 0.0
            for row in range(n_states):
                for column in range(n_states):
                    posterior_total += forward[bin_index, row, column] * backward[row, column]
            background = n_states - 1
            public_mass = np.zeros(n_diplotypes, dtype=np.float64)
            bg_mass = 0.0
            for first in range(n_states):
                public_first = public_state_ids[first]
                for second in range(n_states):
                    public_second = public_state_ids[second]
                    lower = min(public_first, public_second)
                    upper = max(public_first, public_second)
                    mass = (
                        forward[bin_index, first, second]
                        * backward[first, second] / posterior_total
                    )
                    public_mass[lower * n_states + upper] += mass
                    if first == background or second == background:
                        bg_mass += mass
            entropy = 0.0
            largest = 0.0
            unknown_mass = 0.0
            for first in range(n_states):
                for second in range(first, n_states):
                    mass = public_mass[first * n_states + second]
                    largest = max(largest, mass)
                    if mass > 0.0:
                        entropy -= mass * math.log(mass)
                    if first == background or second == background:
                        unknown_mass += mass
            query_first = public_state_ids[
                viterbi_state_grid[sample, 0, bin_index]
            ]
            query_second = public_state_ids[
                viterbi_state_grid[sample, 1, bin_index]
            ]
            lower = min(query_first, query_second)
            upper = max(query_first, query_second)
            query_mass = public_mass[lower * n_states + upper]
            maximum_class_mass[sample, bin_index] = min(1.0, max(0.0, largest))
            class_entropy[sample, bin_index] = max(0.0, entropy)
            background_mass[sample, bin_index] = min(1.0, max(0.0, bg_mass))
            public_unknown_mass[sample, bin_index] = min(
                1.0, max(0.0, unknown_mass)
            )
            viterbi_public_class_mass[sample, bin_index] = min(
                1.0, max(0.0, query_mass)
            )

            if bin_index == 0:
                continue
            distance = centers[bin_index] - centers[bin_index - 1]
            if distance < 1.0:
                distance = 1.0
            theta = min(0.5, max(1e-15, distance * recomb_rate))
            if genetic_distances is not None:
                theta = min(0.5, max(1e-15, genetic_distances[bin_index - 1]))
            log_switch = math.log(theta) - switch_penalty_per_snp * snps_per_bin
            log_stay = math.log(1.0 - theta)
            cost_zero = 2.0 * log_stay
            cost_one = log_switch + log_stay - log_n_minus_one
            cost_two = (
                double_recomb_factor * log_switch - 2.0 * log_n_minus_one
            )
            maximum_cost = max(cost_zero, cost_one, cost_two)
            weight_zero = math.exp(cost_zero - maximum_cost)
            weight_one = math.exp(cost_one - maximum_cost)
            weight_two = math.exp(cost_two - maximum_cost)

            emission_max = np.max(emissions[sample,:, bin_index])
            weighted_next = np.empty((n_states, n_states), dtype=np.float64)
            row_sums = np.zeros(n_states, dtype=np.float64)
            column_sums = np.zeros(n_states, dtype=np.float64)
            total = 0.0
            for row in range(n_states):
                for column in range(n_states):
                    index = row * n_states + column
                    value = (
                        math.exp(emissions[sample, index, bin_index] - emission_max)
                        * backward[row, column]
                    )
                    weighted_next[row, column] = value
                    row_sums[row] += value
                    column_sums[column] += value
                    total += value
            previous_backward = np.empty((n_states, n_states), dtype=np.float64)
            for row in range(n_states):
                for column in range(n_states):
                    cell = weighted_next[row, column]
                    one_change = max(
                        0.0,
                        row_sums[row] + column_sums[column] - 2.0 * cell,
                    )
                    two_changes = max(
                        0.0,
                        total - row_sums[row] - column_sums[column] + cell,
                    )
                    previous_backward[row, column] = (
                        weight_zero * cell
                        + weight_one * one_change
                        + weight_two * two_changes
                    ) / scales[bin_index]
            backward = previous_backward
    return (
        maximum_class_mass, class_entropy, background_mass,
        public_unknown_mass, viterbi_public_class_mass,
    )


def minimum_unordered_switch_counts(label_grid: np.ndarray) -> np.ndarray:
    """Count minimum homolog changes, treating pure track swaps as gauge."""

    labels = np.asarray(label_grid)
    if labels.ndim != 3 or labels.shape[1] != 2:
        raise ValueError("label grid must have shape (samples, 2, bins)")
    if labels.shape[2] < 2:
        return np.zeros(labels.shape[0], dtype=np.int64)
    previous = labels[:,:,:-1]
    current = labels[:,:, 1:]
    direct = ((previous[:, 0] != current[:, 0]).astype(np.int8)
              + (previous[:, 1] != current[:, 1]).astype(np.int8))
    swapped = ((previous[:, 0] != current[:, 1]).astype(np.int8)
               + (previous[:, 1] != current[:, 0]).astype(np.int8))
    return np.sum(np.minimum(direct, swapped), axis=1, dtype=np.int64)


@dataclass(frozen=True)
class RaggedPaintingDiagnostics:
    """Compact T10-facing evidence, MAP, and uncertainty seam."""

    selected_site_indices: np.ndarray
    selected_positions: np.ndarray
    bin_centers: np.ndarray
    bin_edges: np.ndarray
    named_alleles: np.ndarray
    equivalence_classes: tuple[tuple[int, ...], ...]
    map_label_grid: np.ndarray
    map_state_class_grid: np.ndarray
    map_direct_callability: np.ndarray
    viterbi_state_class_grid: np.ndarray
    track_status_grid: np.ndarray
    posterior_max_class_mass: np.ndarray
    posterior_viterbi_public_class_mass: np.ndarray
    posterior_class_entropy: np.ndarray
    posterior_background_mass: np.ndarray
    posterior_public_unknown_mass: np.ndarray
    biological_switch_counts: np.ndarray
    structural_handoff_counts: np.ndarray
    hmm_batch_size: int
    hmm_thread_count: int
    hmm_working_memory_budget_bytes: int
    hmm_estimated_bytes_per_sample: int
    minimum_viterbi_public_class_posterior: float
    # Upper triangle, including the diagonal: samples x S*(S+1)/2 x bins.
    # Diploid emissions are exactly symmetric; transitions remain ordered.
    source_log_emission_upper: np.ndarray | None = None


@dataclass(frozen=True)
class RaggedPainting:
    """A component painting together with explicit ragged-state provenance."""

    painting: Any
    state_space: RaggedStateSpace
    binning: RaggedBinning
    evidence_eligible_sample_mask: np.ndarray
    diagnostics: RaggedPaintingDiagnostics | None = None


def ragged_public_state_ids(state_space: RaggedStateSpace) -> np.ndarray:
    """Map internal states to anchored classes or one public UNKNOWN class."""

    class_is_anchored = np.any(state_space.called, axis=1)
    result = np.arange(state_space.background_index + 1, dtype=np.int32)
    result[:-1][~class_is_anchored] = state_space.background_index
    return result


def release_ragged_states(
        state_space: RaggedStateSpace,
        viterbi_state_class_grid: np.ndarray,
        evidence_eligible_sample_mask: np.ndarray,
        posterior_viterbi_public_class_mass: np.ndarray,
        minimum_viterbi_public_class_posterior: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Release qV-qualified classes, singleton labels, and typed statuses."""

    internal_grid = np.asarray(viterbi_state_class_grid)
    eligible = np.asarray(evidence_eligible_sample_mask, dtype=np.bool_)
    qv = np.asarray(posterior_viterbi_public_class_mass, dtype=np.float64)
    threshold = float(minimum_viterbi_public_class_posterior)
    if (internal_grid.ndim != 3 or internal_grid.shape[1] != 2
            or eligible.shape != (internal_grid.shape[0],)
            or qv.shape != (internal_grid.shape[0], internal_grid.shape[2])):
        raise ValueError("ragged release arrays are not sample/track/bin aligned")
    if (np.any((internal_grid < 0)
               | (internal_grid > state_space.background_index))
            or np.any(~np.isfinite(qv[eligible]))
            or not math.isfinite(threshold) or not 0.5 < threshold <= 1.0):
        raise ValueError("ragged release inputs are invalid")

    class_is_anchored = np.any(state_space.called, axis=1)
    class_output_labels = np.asarray([
        members[0] if len(members) == 1 and class_is_anchored[class_index]
        else -1
        for class_index, members in enumerate(state_space.equivalence_classes)
    ] + [-1], dtype=np.int32)
    class_grid = internal_grid.astype(np.int32, copy=True)
    label_grid = class_output_labels[internal_grid]
    status_by_class = np.full(
        state_space.background_index + 1, int(PaintingTrackStatus.BACKGROUND),
        dtype=np.uint8,
    )
    for class_index, members in enumerate(state_space.equivalence_classes):
        if not class_is_anchored[class_index]:
            status_by_class[class_index] = int(PaintingTrackStatus.UNANCHORED_TRAJECTORY)
        elif len(members) > 1:
            status_by_class[class_index] = int(PaintingTrackStatus.POOLED_EQUIVALENCE)
        else:
            status_by_class[class_index] = int(PaintingTrackStatus.SINGLETON_NAMED)
    status_grid = status_by_class[internal_grid]
    low = np.broadcast_to((qv < threshold)[:, None,:], internal_grid.shape)
    class_grid[low] = -1
    label_grid[low] = -1
    status_grid[low] = int(PaintingTrackStatus.LOW_POSTERIOR_ABSTENTION)
    class_grid[~eligible] = -1
    label_grid[~eligible] = -1
    status_grid[~eligible] = int(PaintingTrackStatus.INELIGIBLE_NO_EVIDENCE)
    return label_grid, class_grid, status_grid


def _released_chunks(labels, classes, statuses, edges):
    """Coalesce unordered release keys, keeping each run's first ordered pair."""
    bins = labels.shape[1]
    if not bins:
        return []
    changed = np.zeros(bins, dtype=np.bool_)
    changed[0] = True
    for grid in (labels, classes, statuses):
        first = np.minimum(grid[0], grid[1])
        second = np.maximum(grid[0], grid[1])
        changed[1:] |= (first[1:] != first[:-1]) | (second[1:] != second[:-1])
    starts = np.flatnonzero(changed)
    stops = np.r_[starts[1:], bins]
    return [painting_components.PaintedChunk(
        int(edges[start]), int(edges[stop]),
        int(labels[0, start]), int(labels[1, start]))
        for start, stop in zip(starts, stops)]


def paint_ragged_component(
        panel,
        genotype_likelihoods: np.ndarray,
        observed: np.ndarray,
        retained: np.ndarray,
        *,
        active: np.ndarray | None=None,
        recomb_rate: float=1e-8,
        switch_penalty_per_snp: float=1.0,
        robustness_epsilon: float=0.01,
        double_recomb_factor: float=1.5,
        snps_per_bin: int=100,
        batch_size: int=32,
        working_memory_bytes: int | None=None,
        minimum_viterbi_public_class_posterior: float=0.90,
        chromosome_map=None,
        source_emission_cache_bytes: int=T09_EMISSION_CACHE_MAX_BYTES,
) -> RaggedPainting:
    """Paint one component in bounded sample batches with exact recurrences.

    This function deliberately has no cross-component state: invoking it once
    per released component gives the required unconditional HMM reset.
    """


    if (isinstance(batch_size, bool) or int(batch_size) != batch_size
            or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    batch_size = int(batch_size)
    minimum_viterbi_public_class_posterior = float(
        minimum_viterbi_public_class_posterior
    )
    if (not math.isfinite(minimum_viterbi_public_class_posterior)
            or not 0.5 < minimum_viterbi_public_class_posterior <= 1.0):
        raise ValueError(
            "minimum_viterbi_public_class_posterior must lie in (0.5, 1]"
        )
    state_space = build_ragged_state_space(panel, retained, active=active)
    if active is not None and not np.all(state_space.active):
        raise NotImplementedError(
            "inactive-state posterior painting requires an explicit continuation model"
        )
    full_evidence = np.asarray(genotype_likelihoods)
    full_observed = np.asarray(observed, dtype=np.bool_)
    if full_evidence.ndim != 3 or full_evidence.shape[2] != 3:
        raise ValueError("component evidence must have shape (samples, sites, 3)")
    if full_evidence.shape[:2] != full_observed.shape:
        raise ValueError("observed mask must match component evidence")
    if full_evidence.shape[1] != len(panel.positions):
        raise ValueError("component evidence must align with the full founder panel")
    evidence, observed = painting_evidence.select_site_evidence(
        full_evidence, full_observed, state_space.site_indices
    )
    binning = build_ragged_bins(
        state_space.positions, state_space.active, snps_per_bin
    )
    eligible = samples_with_ragged_evidence(evidence, observed)
    if not np.any(state_space.called):
        eligible[:] = False
    eligible = np.asarray(eligible, dtype=np.bool_)
    eligible.setflags(write=False)
    if len(state_space.positions) == 0:
        painting = painting_components.BlockPainting(
            (0, 0),
            [painting_components.SamplePainting(index, []) for index in range(evidence.shape[0])],
        )
        return RaggedPainting(painting, state_space, binning, eligible)

    n_states = state_space.background_index + 1
    n_diplotypes = n_states * n_states
    n_bins = len(binning.indices)
    thread_count = max(1, int(numba.get_num_threads()))
    # With D=S², B bins, and L selected sites, 20*D*B bounds the
    # concurrent emission/HMM arrays per sample and 24*L covers normalized GLs.
    # The direct emission kernel has no S²-by-site founder temporary.
    n_symmetric = n_states * (n_states + 1) // 2
    cache_bytes = 8 * evidence.shape[0] * n_symmetric * n_bins
    fixed_working_bytes = cache_bytes if cache_bytes <= source_emission_cache_bytes else 0
    estimated_bytes_per_sample = max(
        1, 20 * n_diplotypes * n_bins + 24 * len(state_space.positions)
    )
    working_memory_budget = resolve_ragged_working_memory_budget(
        estimated_bytes_per_sample,
        fixed_working_bytes,
        thread_count,
        requested_bytes=working_memory_bytes,
    )
    if fixed_working_bytes + estimated_bytes_per_sample > working_memory_budget:
        fixed_working_bytes = 0
    source_log_emission_upper = (
        np.empty((evidence.shape[0], n_symmetric, n_bins), dtype=np.float64)
        if fixed_working_bytes else None
    )
    effective_batch_size = choose_ragged_batch_size(
        evidence.shape[0], batch_size, thread_count,
        estimated_bytes_per_sample, fixed_working_bytes,
        working_memory_budget,
    )
    public_state_ids = ragged_public_state_ids(state_space)
    internal_batches = []
    posterior_max_batches = []
    posterior_entropy_batches = []
    posterior_background_batches = []
    posterior_public_unknown_batches = []
    posterior_viterbi_public_batches = []
    genetic_distances = None
    if chromosome_map is not None:
        recomb_rate = chromosome_map.fallback_rate_per_bp
        if chromosome_map.has_map:
            genetic_distances = np.ascontiguousarray(chromosome_map.interval_morgans(
                binning.centers[:-1], binning.centers[1:]))
    # Map integration is outside sample batches and all numerical recurrences.
    for batch_start in range(0, evidence.shape[0], effective_batch_size):
        batch_stop = min(batch_start + effective_batch_size, evidence.shape[0])
        emissions, _ = calculate_ragged_binned_emissions(
            evidence[batch_start:batch_stop],
            observed[batch_start:batch_stop],
            state_space,
            binning,
            robustness_epsilon=robustness_epsilon,
        )
        if source_log_emission_upper is not None:
            painting_evidence.store_symmetric_emissions(
                emissions, source_log_emission_upper, batch_start, n_states)
        viterbi_grid = viterbi_label_grid(
            emissions, binning.centers, n_states, recomb_rate,
            switch_penalty_per_snp, snps_per_bin, double_recomb_factor, genetic_distances,
        )
        (
            posterior_max, posterior_entropy, posterior_background,
            posterior_public_unknown, posterior_viterbi_public,
        ) = posterior_class_summaries(
            emissions, binning.centers, n_states, recomb_rate,
            switch_penalty_per_snp, snps_per_bin, double_recomb_factor,
            viterbi_grid, public_state_ids, genetic_distances,
        )
        internal_batches.append(viterbi_grid)
        posterior_max_batches.append(posterior_max)
        posterior_entropy_batches.append(posterior_entropy)
        posterior_background_batches.append(posterior_background)
        posterior_public_unknown_batches.append(posterior_public_unknown)
        posterior_viterbi_public_batches.append(posterior_viterbi_public)
    internal_grid = np.concatenate(internal_batches, axis=0)
    posterior_max = np.concatenate(posterior_max_batches, axis=0)
    posterior_entropy = np.concatenate(posterior_entropy_batches, axis=0)
    posterior_background = np.concatenate(posterior_background_batches, axis=0)
    posterior_public_unknown = np.concatenate(
        posterior_public_unknown_batches, axis=0
    )
    posterior_viterbi_public = np.concatenate(
        posterior_viterbi_public_batches, axis=0
    )

    map_grid, class_grid, status_grid = release_ragged_states(
        state_space,
        internal_grid,
        eligible,
        posterior_viterbi_public,
        minimum_viterbi_public_class_posterior,
    )
    samples = []
    for sample_index in range(evidence.shape[0]):
        if not eligible[sample_index]:
            samples.append(painting_components.SamplePainting(sample_index, []))
            continue
        chunks = _released_chunks(
            map_grid[sample_index], class_grid[sample_index],
            status_grid[sample_index], binning.edges)
        samples.append(painting_components.SamplePainting(sample_index, chunks))
    painting = painting_components.BlockPainting(
        (int(state_space.positions[0]), int(state_space.positions[-1])), samples
    )
    posterior_max[~eligible] = np.nan
    posterior_entropy[~eligible] = np.nan
    posterior_background[~eligible] = np.nan
    posterior_public_unknown[~eligible] = np.nan
    posterior_viterbi_public[~eligible] = np.nan
    if evidence.dtype in (np.dtype("float32"), np.dtype("float64")):
        # NumPy's raw-array arithmetic uses this dtype (unlike the float64
        # normalized eligibility calculation); retain that threshold rounding.
        epsilon = np.asarray(16.0 * np.finfo(np.float64).eps, dtype=evidence.dtype)[()]
        nonuniform = painting_evidence.raw_nonuniform(evidence, epsilon)
    else:
        totals = np.sum(evidence, axis=2)
        spread = np.max(evidence, axis=2) - np.min(evidence, axis=2)
        nonuniform = (totals > 0.0) & (spread > 16.0 * np.finfo(np.float64).eps * totals)
    starts = np.asarray([indices[0] for indices in binning.indices], dtype=np.int64)
    stops = np.asarray([indices[-1] + 1 for indices in binning.indices], dtype=np.int64)
    direct = painting_evidence.direct_callability(
        internal_grid, class_grid, state_space.called, observed, nonuniform,
        starts, stops, state_space.background_index)
    biological = minimum_unordered_switch_counts(internal_grid)
    biological[~eligible] = 0
    named_alleles = np.full(state_space.called.shape, -1, dtype=np.int8)
    named_alleles[state_space.called] = state_space.q[state_space.called].astype(np.int8)
    diagnostics = RaggedPaintingDiagnostics(
        selected_site_indices=state_space.site_indices.copy(),
        selected_positions=state_space.positions.copy(),
        bin_centers=binning.centers.copy(),
        bin_edges=binning.edges.copy(),
        named_alleles=named_alleles,
        equivalence_classes=state_space.equivalence_classes,
        map_label_grid=map_grid,
        map_state_class_grid=class_grid,
        map_direct_callability=direct,
        viterbi_state_class_grid=internal_grid.copy(),
        track_status_grid=status_grid,
        posterior_max_class_mass=posterior_max,
        posterior_viterbi_public_class_mass=posterior_viterbi_public,
        posterior_class_entropy=posterior_entropy,
        posterior_background_mass=posterior_background,
        posterior_public_unknown_mass=posterior_public_unknown,
        biological_switch_counts=biological,
        structural_handoff_counts=np.zeros(evidence.shape[0], dtype=np.int64),
        hmm_batch_size=effective_batch_size,
        hmm_thread_count=thread_count,
        hmm_working_memory_budget_bytes=working_memory_budget,
        hmm_estimated_bytes_per_sample=estimated_bytes_per_sample,
        minimum_viterbi_public_class_posterior=minimum_viterbi_public_class_posterior,
        source_log_emission_upper=source_log_emission_upper,
    )
    return RaggedPainting(painting, state_space, binning, eligible, diagnostics)


def samples_with_ragged_evidence(
        genotype_likelihoods: np.ndarray,
        observed: np.ndarray,
) -> np.ndarray:
    """Return samples having at least one observed non-uniform GL row."""

    raw = np.asarray(genotype_likelihoods)
    if (raw.ndim == 3 and raw.shape[2] == 3 and raw.size >= 196608
            and raw.dtype in (np.dtype("float32"), np.dtype("float64"))):
        mask = np.broadcast_to(np.asarray(observed, dtype=np.bool_), raw.shape[:2])
        eligible, invalid = painting_evidence.eligible_samples(raw, mask)
        if invalid:
            raise ValueError("genotype likelihoods must be finite and non-negative")
        return eligible
    evidence = normalise_genotype_likelihoods(genotype_likelihoods)
    observed = np.asarray(observed, dtype=np.bool_)
    spread = np.max(evidence, axis=2) - np.min(evidence, axis=2)
    return np.any(observed & (spread > 16.0 * np.finfo(np.float64).eps), axis=1)

import haplotype_reconstruction.painting.components as painting_components
