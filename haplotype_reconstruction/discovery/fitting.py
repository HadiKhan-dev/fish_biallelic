"""discovery / fitting for the canonical reconstruction pipeline."""
from __future__ import annotations


import numpy as np
import math
import numba
from collections import OrderedDict

from threading import RLock
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def _prepare_fit_cost_tables(cost_WW, log_probs, lam):
    """Precompute evidence-only costs reused by every A/H fit iteration."""
    N, L = cost_WW.shape
    ww_total_cost = np.empty(N, dtype=np.float64)
    genotype_cost = np.empty((N, L, 3), dtype=np.float64)
    real_wildcard_cost = np.empty((N, L, 2), dtype=np.float64)
    for s in prange(N):
        total = 0.0
        for l in range(L):
            cap = cost_WW[s, l]
            total += cap
            lp0 = log_probs[s, l, 0]
            lp1 = log_probs[s, l, 1]
            lp2 = log_probs[s, l, 2]
            for genotype in range(3):
                value = -log_probs[s, l, genotype]
                if value > cap:
                    value = cap
                genotype_cost[s, l, genotype] = value
            best_lp0 = lp0 if lp0 > lp1 else lp1
            best_lp1 = lp1 if lp1 > lp2 else lp2
            value0 = -best_lp0 + lam
            value1 = -best_lp1 + lam
            if value0 > cap:
                value0 = cap
            if value1 > cap:
                value1 = cap
            real_wildcard_cost[s, l, 0] = value0
            real_wildcard_cost[s, l, 1] = value1
        ww_total_cost[s] = total
    return ww_total_cost, genotype_cost, real_wildcard_cost


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def _prepare_depth_mask_fit_evidence(probs_k, observed_mask, lam):
    """Return zero-contribution missing emissions and depth-aware WW costs."""
    N, L, _ = probs_k.shape
    log_probs = np.empty((N, L, 3), dtype=np.float64)
    cost_WW = np.empty((N, L), dtype=np.float64)
    eps = 1e-12
    for s in prange(N):
        for l in range(L):
            if not observed_mask[s, l]:
                log_probs[s, l, 0] = 0.0
                log_probs[s, l, 1] = 0.0
                log_probs[s, l, 2] = 0.0
                cost_WW[s, l] = 0.0
                continue
            maximum = -math.inf
            for genotype in range(3):
                value = probs_k[s, l, genotype]
                if value < eps:
                    value = eps
                log_value = math.log(value)
                log_probs[s, l, genotype] = log_value
                if log_value > maximum:
                    maximum = log_value
            cost_WW[s, l] = -maximum + 2.0 * lam
    return log_probs, cost_WW


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def _prepare_fit_cost_tables_depth_mask(
        cost_WW, log_probs, observed_mask, lam):
    """Prepare H-update costs with wildcard charges only at observed cells."""
    N, L = cost_WW.shape
    ww_total_cost = np.empty(N, dtype=np.float64)
    genotype_cost = np.empty((N, L, 3), dtype=np.float64)
    real_wildcard_cost = np.empty((N, L, 2), dtype=np.float64)
    for s in prange(N):
        total = 0.0
        for l in range(L):
            cap = cost_WW[s, l]
            total += cap
            lp0 = log_probs[s, l, 0]
            lp1 = log_probs[s, l, 1]
            lp2 = log_probs[s, l, 2]
            for genotype in range(3):
                value = -log_probs[s, l, genotype]
                if value > cap:
                    value = cap
                genotype_cost[s, l, genotype] = value
            best_lp0 = lp0 if lp0 > lp1 else lp1
            best_lp1 = lp1 if lp1 > lp2 else lp2
            penalty = lam if observed_mask[s, l] else 0.0
            value0 = -best_lp0 + penalty
            value1 = -best_lp1 + penalty
            if value0 > cap:
                value0 = cap
            if value1 > cap:
                value1 = cap
            real_wildcard_cost[s, l, 0] = value0
            real_wildcard_cost[s, l, 1] = value1
        ww_total_cost[s] = total
    return ww_total_cost, genotype_cost, real_wildcard_cost


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def _ww_total_cost_kernel(cost_WW):
    """Exact left-to-right WW row sums for standalone assignment updates."""
    N, L = cost_WW.shape
    totals = np.empty(N, dtype=np.float64)
    for s in prange(N):
        total = 0.0
        for l in range(L):
            total += cost_WW[s, l]
        totals[s] = total
    return totals


class _FixedKFitWorkspace:
    """Immutable evidence-only inputs shared by repeated fixed-K fits.

    The posterior successor evaluates thousands of initial haplotype matrices
    against the same evidence tensor.  The wildcard costs, log probabilities,
    binned wildcard emissions, and BLAS inputs do not depend on ``H_init``.
    Keeping them in a call-scoped workspace removes that repeated work without
    changing any coordinate-descent operation or floating-point reduction.

    ``probs_reference`` deliberately keeps the exact source object alive and
    lets :func:`_fit_at_fixed_K` reject accidental reuse with another tensor.
    """

    __slots__ = (
        "probs_reference",
        "lam",
        "fit_config_key",
        "cost_WW",
        "log_probs",
        "WW_bin_emis",
        "WW_total_cost",
        "h_genotype_cost",
        "h_wildcard_cost",
        "blas_lp_cache",
        "binary_pattern_cache",
        "observed_mask",
        "uninformative_samples",
        "fit_result_cache",
        "fixed_point_keys",
        "assignment_cache",
        "transition_cache",
        "cache_lock",
    )

    def __init__(
        self,
        probs_reference,
        lam,
        cost_WW,
        log_probs,
        WW_bin_emis,
        WW_total_cost,
        h_genotype_cost,
        h_wildcard_cost,
        blas_lp_cache,
        binary_pattern_cache,
        observed_mask,
        uninformative_samples,
    ):
        self.probs_reference = probs_reference
        self.lam = float(lam)
        self.observed_mask = observed_mask
        self.uninformative_samples = uninformative_samples
        if self.observed_mask is not None:
            self.observed_mask.setflags(write=False)
            self.uninformative_samples.setflags(write=False)
        self.fit_config_key = (
            bool(core_config._VITERBI_BIC_ENABLED),
            int(core_config.VITERBI_SNPS_PER_BIN),
            float(core_config.VITERBI_SWITCH_PENALTY),
            self.observed_mask is not None,
        )
        self.cost_WW = cost_WW
        self.log_probs = log_probs
        self.WW_bin_emis = WW_bin_emis
        self.WW_total_cost = WW_total_cost
        self.h_genotype_cost = h_genotype_cost
        self.h_wildcard_cost = h_wildcard_cost
        self.blas_lp_cache = blas_lp_cache
        self.binary_pattern_cache = binary_pattern_cache
        # This cache is intentionally scoped to one exact evidence workspace.
        # Its key additionally contains every per-fit input that can change the
        # coordinate-descent trajectory.  Stored arrays are private read-only
        # copies so callers retain the historical freedom to mutate returned
        # arrays without corrupting a later reuse.
        self.fit_result_cache = OrderedDict()
        # Fixed-point certificates retain exact founder-row order because
        # assignment tie-breaking is index ordered.  A key enters this set only
        # after a complete A/H pass has proved zero H changes for this evidence
        # and wildcard penalty in that exact order.
        self.fixed_point_keys = set()
        # Coordinate descent is a deterministic map on an *ordered* binary
        # founder panel for fixed evidence. Different proposal starts often
        # enter the same trajectory, so retain the expensive A(H) evaluation
        # and the following H(A) transition separately from complete-fit
        # results. Keys bit-pack values in row-major order: founder order is
        # preserved (tie-breaking depends on it) without retaining an int64
        # K-by-L matrix for every visited state.
        self.assignment_cache = OrderedDict()
        self.transition_cache = OrderedDict()
        self.cache_lock = RLock()

    @staticmethod
    def _ordered_h_key(haplotypes):
        matrix = np.asarray(haplotypes)
        if matrix.ndim != 2:
            raise ValueError("fixed-K haplotypes must be two-dimensional")
        contiguous = np.ascontiguousarray(matrix)
        return (
            contiguous.dtype.str,
            tuple(int(value) for value in contiguous.shape),
            contiguous.tobytes(),
        )

    @staticmethod
    def _binary_state_key(haplotypes):
        """Compact exact key for an ordered binary founder panel."""
        matrix = np.asarray(haplotypes)
        if matrix.ndim != 2 or matrix.dtype.kind not in "biu":
            return None
        if np.any((matrix != 0) & (matrix != 1)):
            return None
        flat = np.ascontiguousarray(matrix, dtype=np.uint8).reshape(-1)
        packed = np.packbits(flat, bitorder="little")
        return (int(matrix.shape[0]), int(matrix.shape[1]), packed.tobytes())

    @staticmethod
    def _binary_state_from_key(key):
        """Decode a compact state key to the production int64 H dtype."""
        K, L, packed = key
        bits = np.unpackbits(
            np.frombuffer(packed, dtype=np.uint8),
            count=int(K) * int(L),
            bitorder="little",
        )
        return np.ascontiguousarray(
            bits.reshape((int(K), int(L))), dtype=np.int64
        )

    @staticmethod
    def _store_assignment_record(A, cost, uncapped_cost, wildcard_slots):
        stored_A = np.array(A, copy=True, order="K")
        stored_cost = np.array(cost, copy=True, order="K")
        stored_wildcard = np.array(wildcard_slots, copy=True, order="K")
        if uncapped_cost is cost:
            stored_uncapped = stored_cost
        else:
            stored_uncapped = np.array(
                uncapped_cost, copy=True, order="K"
            )
        for array in (stored_A, stored_cost, stored_uncapped, stored_wildcard):
            array.setflags(write=False)
        return (
            stored_A, stored_cost, stored_uncapped, stored_wildcard,
            float(stored_uncapped.sum()),
        )

    def apply_missing_assignment_policy(
            self, assignments, wildcard_slots, wildcard_index):
        """Force wholly uninformative samples to the unresolved WW state."""
        if self.observed_mask is None:
            return
        unresolved = self.uninformative_samples
        assignments[unresolved, 0] = int(wildcard_index)
        assignments[unresolved, 1] = int(wildcard_index)
        wildcard_slots[unresolved] = 2

    def cached_assignment(self, state_key):
        with self.cache_lock:
            record = self.assignment_cache.get(state_key)
            if record is not None:
                self.assignment_cache.move_to_end(state_key)
            return record

    def cached_transition(self, state_key):
        with self.cache_lock:
            record = self.transition_cache.get(state_key)
            if record is not None:
                self.transition_cache.move_to_end(state_key)
            return record

    def remember_transition(self, state_key, assignment_record,
                            next_state_key, h_changes):
        """Remember one exact A(H), H(A) step using immutable records."""
        transition = (assignment_record, next_state_key, int(h_changes))
        with self.cache_lock:
            self.assignment_cache[state_key] = assignment_record
            self.assignment_cache.move_to_end(state_key)
            self.transition_cache[state_key] = transition
            self.transition_cache.move_to_end(state_key)
            while len(self.assignment_cache) > 16384:
                self.assignment_cache.popitem(last=False)
            while len(self.transition_cache) > 16384:
                self.transition_cache.popitem(last=False)
        return transition

    def remember_assignment(self, state_key, assignment_record):
        with self.cache_lock:
            self.assignment_cache[state_key] = assignment_record
            self.assignment_cache.move_to_end(state_key)
            while len(self.assignment_cache) > 16384:
                self.assignment_cache.popitem(last=False)

    def supports_binary_transition_engine(self, haplotypes):
        matrix = np.asarray(haplotypes)
        return (
            matrix.ndim == 2
            and matrix.dtype == np.dtype(np.int64)
            and matrix.shape[0] >= 1
            and self.binary_pattern_cache is not None
            and self._binary_state_key(matrix) is not None
        )

    def compute_transition_batch(self, haplotypes, *, state_keys=None):
        """Compute and cache exact transitions for int64 binary starts."""
        matrices = tuple(haplotypes)
        if not matrices:
            return []
        H_batch = np.ascontiguousarray(np.stack(matrices), dtype=np.int64)
        K = H_batch.shape[1]
        rr_i, rr_j = discovery_assignments._pair_indices_for_K(K)
        cache = self.binary_pattern_cache
        C0b, _diff, _w, kW_Cb, _kwdiff = self.blas_lp_cache
        # This kernel parallelizes only over independent starts. Waking a
        # much larger Numba team adds scheduler cost without exposing more
        # work; four threads is the measured low-frontier saturation point.
        active_threads = numba.get_num_threads()
        batch_threads = min(active_threads, max(4, H_batch.shape[0]))
        if batch_threads != active_threads:
            numba.set_num_threads(batch_threads)
        try:
            (H_next, A_batch, cost_batch, wildcard_batch,
             h_changes) = discovery_frontier_emissions.transition_batch(
                C0b,
                cache.diff1_table, cache.w_table,
                kW_Cb, cache.kWdiff_table,
                self.WW_bin_emis, self.WW_total_cost,
                rr_i, rr_j,
                float(core_config.VITERBI_SWITCH_PENALTY),
                int(cache.snps_per_bin), int(cache.n_bins),
                self.h_genotype_cost, self.h_wildcard_cost,
                self.uninformative_samples, H_batch,
            )
        finally:
            if batch_threads != active_threads:
                numba.set_num_threads(active_threads)
        results = []
        for index in range(H_batch.shape[0]):
            if int(h_changes[index]) < 0:
                raise ValueError("batch transition received a nonbinary start")
            state_key = (
                self._binary_state_key(H_batch[index])
                if state_keys is None else state_keys[index]
            )
            # The fused binary Viterbi path has no capped/uncapped split.
            # Reuse the one immutable stored array instead of copying an
            # identical second B x N slab for every frontier transition.
            cost = cost_batch[index]
            assignment = self._store_assignment_record(
                A_batch[index], cost, cost, wildcard_batch[index]
            )
            next_key = self._binary_state_key(H_next[index])
            transition = self.remember_transition(
                state_key, assignment, next_key, int(h_changes[index])
            )
            results.append(transition)
        return results


    def compute_assignment_scalar(
            self, haplotypes, *, state_key=None, known_missing=False):
        """Compute and cache A(H) without performing an unused H sweep."""
        matrix = np.ascontiguousarray(haplotypes, dtype=np.int64)
        if state_key is None:
            state_key = self._binary_state_key(matrix)
        if not known_missing:
            cached = self.cached_assignment(state_key)
            if cached is not None:
                return cached
        A, cost, uncapped_cost, wildcard = discovery_assignments._update_A(
            self.probs_reference, matrix, self.lam,
            cost_WW=self.cost_WW,
            WW_bin_emis=self.WW_bin_emis,
            log_probs=self.log_probs,
            blas_lp_cache=self.blas_lp_cache,
            binary_pattern_cache=self.binary_pattern_cache,
            WW_total_cost=self.WW_total_cost,
        )
        self.apply_missing_assignment_policy(A, wildcard, matrix.shape[0])
        assignment = self._store_assignment_record(
            A, cost, uncapped_cost, wildcard
        )
        self.remember_assignment(state_key, assignment)
        return assignment


    def compute_transition_scalar(
            self, haplotypes, *, state_key=None, known_missing=False):
        """Compute one transition with the live internally parallel kernels."""
        matrix = np.ascontiguousarray(haplotypes, dtype=np.int64)
        if state_key is None:
            state_key = self._binary_state_key(matrix)
        if not known_missing:
            cached = self.cached_transition(state_key)
            if cached is not None:
                return cached
        A, cost, uncapped_cost, wildcard = discovery_assignments._update_A(
            self.probs_reference, matrix, self.lam,
            cost_WW=self.cost_WW,
            WW_bin_emis=self.WW_bin_emis,
            log_probs=self.log_probs,
            blas_lp_cache=self.blas_lp_cache,
            binary_pattern_cache=self.binary_pattern_cache,
            WW_total_cost=self.WW_total_cost,
        )
        self.apply_missing_assignment_policy(A, wildcard, matrix.shape[0])
        H_next = matrix.copy()
        h_changes = discovery_founder_updates._update_H(
            self.probs_reference, H_next, A, self.lam,
            cost_WW=self.cost_WW,
            log_probs=self.log_probs,
            h_genotype_cost=self.h_genotype_cost,
            h_wildcard_cost=self.h_wildcard_cost,
        )
        assignment = self._store_assignment_record(
            A, cost, uncapped_cost, wildcard
        )
        next_key = self._binary_state_key(H_next)
        return self.remember_transition(
            state_key, assignment, next_key, h_changes
        )

    def _fit_key(self, haplotypes, max_iter):
        matrix = np.asarray(haplotypes)
        contiguous = np.ascontiguousarray(matrix)
        # The A/H kernels partition independent samples or sites and retain
        # identical scalar reduction order within each item. Thread count is
        # therefore an execution detail, not part of fit identity; including
        # it fragmented the cache whenever a tail block gained free cores.
        return (
            self.fit_config_key,
            int(max_iter),
            contiguous.dtype.str,
            tuple(int(value) for value in contiguous.shape),
            contiguous.tobytes(),
        )

    @staticmethod
    def _store_fit_result(result):
        stored = []
        for index, value in enumerate(result):
            if index < 4:
                array = np.array(value, copy=True, order="K")
                array.setflags(write=False)
                stored.append(array)
            elif index == 4:
                stored.append(int(value))
            else:
                stored.append(float(value))
        return tuple(stored)

    @staticmethod
    def _copy_fit_result(result):
        return (
            np.array(result[0], copy=True, order="K"),
            np.array(result[1], copy=True, order="K"),
            np.array(result[2], copy=True, order="K"),
            np.array(result[3], copy=True, order="K"),
            int(result[4]),
            float(result[5]),
        )

    def cached_fit(self, haplotypes, max_iter):
        key = self._fit_key(haplotypes, max_iter)
        with self.cache_lock:
            result = self.fit_result_cache.get(key)
            if result is None:
                return key, None
            self.fit_result_cache.move_to_end(key)
            return key, self._copy_fit_result(result)

    def remember_fit(self, key, result, *, fixed_point):
        stored = self._store_fit_result(result)
        with self.cache_lock:
            self.fit_result_cache[key] = stored
            self.fit_result_cache.move_to_end(key)
            # A bounded cache prevents a long-lived caller from retaining an
            # unbounded number of sample-level arrays.  Complete-mode searches
            # normally stay well below this evidence-local ceiling.
            while len(self.fit_result_cache) > 4096:
                self.fit_result_cache.popitem(last=False)
            if fixed_point:
                self.fixed_point_keys.add(
                    self._ordered_h_key(result[0])
                )

    def certifies_fixed_point(self, haplotypes):
        key = self._ordered_h_key(haplotypes)
        with self.cache_lock:
            return key in self.fixed_point_keys


def _prepare_fixed_k_fit_workspace(
        probs_k, lam, *, binary_patterns=True, observed_mask=None):
    """Prepare quantities invariant across fixed-K initializations.

    When an observed mask is supplied, missing cells contribute zero
    emission/cost and wildcard penalties are charged only where a sample has
    allele-depth evidence. An absent mask remains useful as a numerical
    reference for fully observed inputs.
    """

    probs_reference = probs_k
    L = probs_k.shape[1]
    probs_c = discovery_objectives._maybe_c_contig(probs_k, np.float64)
    if observed_mask is None:
        mask_value = None
        uninformative_samples = None
        cost_WW = discovery_objectives._per_site_cost_W_W(probs_k, lam)
        log_probs = discovery_objectives._log_probs_kernel(probs_c)
        (
            WW_total_cost,
            h_genotype_cost,
            h_wildcard_cost,
        ) = _prepare_fit_cost_tables(
            discovery_objectives._maybe_c_contig(cost_WW, np.float64),
            discovery_objectives._maybe_c_contig(log_probs, np.float64),
            float(lam),
        )
    else:
        mask_value = np.array(
            observed_mask, dtype=np.bool_, order="C", copy=True
        )
        if mask_value.shape != probs_c.shape[:2]:
            raise ValueError(
                "observed_mask must match evidence samples and sites"
            )
        uninformative_samples = np.ascontiguousarray(
            ~np.any(mask_value, axis=1)
        )
        log_probs, cost_WW = _prepare_depth_mask_fit_evidence(
            probs_c, mask_value, float(lam)
        )
        (
            WW_total_cost,
            h_genotype_cost,
            h_wildcard_cost,
        ) = _prepare_fit_cost_tables_depth_mask(
            discovery_objectives._maybe_c_contig(cost_WW, np.float64),
            discovery_objectives._maybe_c_contig(log_probs, np.float64),
            mask_value,
            float(lam),
        )
    h_genotype_cost = np.ascontiguousarray(
        h_genotype_cost.transpose(1, 0, 2)
    )
    h_wildcard_cost = np.ascontiguousarray(
        h_wildcard_cost.transpose(1, 0, 2)
    )
    if core_config._VITERBI_BIC_ENABLED:
        if core_config.VITERBI_SNPS_PER_BIN > 1 and core_config.VITERBI_SNPS_PER_BIN < L:
            snps_per_bin = core_config.VITERBI_SNPS_PER_BIN
            n_bins = int(math.ceil(L / core_config.VITERBI_SNPS_PER_BIN))
        else:
            snps_per_bin = 1
            n_bins = L
        cost_WW_c = discovery_objectives._maybe_c_contig(cost_WW, np.float64)
        WW_bin_emis = discovery_objectives._ww_bin_emis_from_cost_ww(
            cost_WW_c, int(snps_per_bin), int(n_bins)
        )
        if mask_value is None:
            blas_lp_cache = discovery_assignments._update_A_blas_lp_precompute(
                discovery_objectives._maybe_c_contig(log_probs, np.float64),
                float(lam),
                int(snps_per_bin),
                int(n_bins),
            )
        else:
            blas_lp_cache = discovery_assignments._update_A_blas_lp_precompute_depth_mask(
                discovery_objectives._maybe_c_contig(log_probs, np.float64),
                mask_value,
                float(lam),
                int(snps_per_bin),
                int(n_bins),
            )
        binary_pattern_cache = (
            discovery_assignments._prepare_binary_pattern_contractions(
                blas_lp_cache, int(snps_per_bin), int(n_bins)
            )
            if binary_patterns
            else None
        )
    else:
        WW_bin_emis = None
        blas_lp_cache = None
        binary_pattern_cache = None
    return _FixedKFitWorkspace(
        probs_reference,
        lam,
        cost_WW,
        log_probs,
        WW_bin_emis,
        WW_total_cost,
        h_genotype_cost,
        h_wildcard_cost,
        blas_lp_cache,
        binary_pattern_cache,
        mask_value,
        uninformative_samples,
    )


def _fit_result_from_assignment(haplotypes, assignment, n_iter):
    """Build the public fit tuple from one immutable assignment record."""
    return (
        np.array(haplotypes, copy=True, order="K"),
        np.array(assignment[0], copy=True, order="K"),
        np.array(assignment[1], copy=True, order="K"),
        np.array(assignment[3], copy=True, order="K"),
        int(n_iter),
        float(assignment[4]),
    )


def _fit_binary_cached_trajectory(
        H_init, max_iter, workspace, capture_initial, *,
        initial_key=None, initial_transition=None):
    """Follow the exact coordinate map, reusing transitions and local cycles."""
    if initial_key is None:
        initial_key = workspace._binary_state_key(H_init)
    state_key = initial_key
    path = [state_key]
    first_visit = {state_key: 0}
    previous_A = None
    captured_initial = None
    completed = 0
    fixed_point = False

    def transition_for(key, matrix=None):
        transition = initial_transition if key == initial_key else None
        if transition is None:
            transition = workspace.cached_transition(key)
        if transition is None:
            if matrix is None:
                matrix = workspace._binary_state_from_key(key)
            transition = workspace.compute_transition_scalar(
                matrix, state_key=key, known_missing=True)
        return transition

    if int(max_iter) <= 0:
        assignment = workspace.cached_assignment(state_key)
        if assignment is None:
            assignment = workspace.compute_assignment_scalar(
                H_init, state_key=state_key, known_missing=True
            )
        result = _fit_result_from_assignment(H_init, assignment, 0)
        captured = (_FixedKFitWorkspace._copy_fit_result(result)
                    if capture_initial else None)
        return captured, result, False

    result = None
    while completed < int(max_iter):
        assignment, next_key, h_changes = transition_for(
            state_key, H_init if completed == 0 else None
        )
        if capture_initial and captured_initial is None:
            captured_initial = _fit_result_from_assignment(
                H_init, assignment, 0
            )

        completed += 1
        if h_changes == 0:
            final_H = workspace._binary_state_from_key(state_key)
            reported_iterations = completed
            if previous_A is None or not np.array_equal(
                    assignment[0], previous_A):
                if completed < int(max_iter):
                    reported_iterations += 1
            result = _fit_result_from_assignment(
                final_H, assignment, reported_iterations
            )
            fixed_point = True
            break

        previous_A = assignment[0]
        cycle_start = first_visit.get(next_key)
        if cycle_start is not None:
            period = completed - int(cycle_start)
            if period <= 1:
                raise AssertionError(
                    "nonzero H changes produced a one-state transition cycle"
                )
            remaining = int(max_iter) - completed
            endpoint_key = path[
                int(cycle_start) + (remaining % period)
            ]
            endpoint_assignment = workspace.cached_assignment(endpoint_key)
            if endpoint_assignment is None:
                endpoint_assignment = transition_for(endpoint_key)[0]
            final_H = workspace._binary_state_from_key(endpoint_key)
            result = _fit_result_from_assignment(
                final_H, endpoint_assignment, int(max_iter)
            )
            break

        state_key = next_key
        first_visit[state_key] = completed
        path.append(state_key)

    if result is None:
        endpoint_assignment = workspace.cached_assignment(state_key)
        if endpoint_assignment is None:
            endpoint_H = workspace._binary_state_from_key(state_key)
            endpoint_assignment = workspace.compute_assignment_scalar(endpoint_H)
        final_H = workspace._binary_state_from_key(state_key)
        result = _fit_result_from_assignment(
            final_H, endpoint_assignment, int(max_iter)
        )
    return captured_initial, result, fixed_point


def _fit_at_fixed_K(
        probs_k, H_init, lam, max_iter=50, *, workspace=None,
        _return_initial=False):
    """Run discrete coordinate descent at the K determined by H_init.shape[0].

    Alternates updating A (pair assignments) and H (founder bits) until
    no changes are made in a full pass, or max_iter is reached.

    Arguments:
        probs_k: (N, L_kept, 3) — kept-site posteriors
        H_init:  (K, L_kept) discrete {0, 1} — initial founder values
        lam:     wildcard penalty
        max_iter: cap on coordinate descent iterations
        workspace: optional evidence-local precomputation shared across fits
                   of different H initializations against this exact probs_k
                   object and lambda

    Returns:
        H:               final (K, L_kept)
        A:               final (N, 2)
        per_sample_cost: (N,) total CAPPED cost per sample under final state
                          (used in worst-fit-sample selection for K-growth seed)
        wildcard_slots:  (N,) wildcard strand count per sample
        n_iter:          how many iterations were used
        total_NLL:       scalar — UNCAPPED NLL summed across samples (used
                          by K-growth as the improvement signal; the cap
                          would mask improvements where adding a founder
                          converts samples from "way over cost_WW" to
                          "still over cost_WW but less so")
    """
    shared_workspace = workspace is not None
    H = H_init.copy()
    A_prev = None
    n_iter = 0
    fixed_point_certified = False

    # Initialise result variables so the post-loop block can refer to
    # them whether or not the loop body executed (max_iter=0 edge case)
    # or completed any iterations.
    A = None
    per_sample_cost = None
    per_sample_cost_unc = None
    wildcard_slots = None

    # ---------------------------------------------------------------------
    # Pre-bake invariant quantities ONCE per CD invocation.
    #
    # cost_WW depends only on (probs_k, lam) — neither changes inside the
    # CD loop — so it's wasteful to recompute it in every _update_A and
    # _update_H call.  Same for WW_bin_emis, which is a fixed binning of
    # cost_WW.  Same for log_probs (Tier 0), which depends only on
    # probs_k.  We compute all three here and thread them through.
    #
    # The fused _update_A_fused_blas_kernel takes the WW state's bin
    # emissions from WW_bin_emis (no inner WW site-loop), while the rr/kW
    # state emissions come from GEMM contractions of log_probs-derived
    # arrays (built once by _update_A_blas_lp_precompute) rather than
    # per-visit log() calls.  The M-step kernel _update_one_founder_kernel
    # reads cost_WW[s, l] for the cap value and log_probs[s, l, g] for the
    # per-genotype cost.
    # ---------------------------------------------------------------------
    if workspace is None:
        workspace = _prepare_fixed_k_fit_workspace(
            probs_k, lam, binary_patterns=False
        )
    elif not isinstance(workspace, _FixedKFitWorkspace):
        raise TypeError("workspace must be a _FixedKFitWorkspace")
    elif workspace.probs_reference is not probs_k:
        raise ValueError(
            "fixed-K workspace was prepared for a different evidence tensor"
        )
    elif workspace.lam != float(lam):
        raise ValueError(
            "fixed-K workspace wildcard penalty does not match this fit"
        )
    fit_cache_key = None
    initial_cache_key = None
    initial_result = None
    if shared_workspace:
        fit_cache_key, cached_result = workspace.cached_fit(
            H_init, max_iter
        )
        if not _return_initial and cached_result is not None:
            return cached_result
        if _return_initial:
            initial_cache_key, initial_result = workspace.cached_fit(
                H_init, 0
            )
            if cached_result is not None:
                if initial_result is None:
                    initial_result = _fit_at_fixed_K(
                        probs_k, H_init, lam, max_iter=0,
                        workspace=workspace,
                    )
                return initial_result, cached_result
    cost_WW = workspace.cost_WW
    log_probs = workspace.log_probs
    WW_bin_emis = workspace.WW_bin_emis
    WW_total_cost = workspace.WW_total_cost
    h_genotype_cost = workspace.h_genotype_cost
    h_wildcard_cost = workspace.h_wildcard_cost
    # K=0 never enters the BLAS founder-state path.
    _blas_lp_cache = (
        workspace.blas_lp_cache if H.shape[0] > 0 else None
    )
    _binary_pattern_cache = (
        workspace.binary_pattern_cache if H.shape[0] > 0 else None
    )

    # Cold scalar fits retain the direct recurrence. Enter the trajectory
    # engine only when this exact ordered panel has a reusable transition.
    initial_transition_key = None
    initial_transition = None
    if (shared_workspace and workspace.transition_cache
            and H.dtype == np.dtype(np.int64)
            and H.shape[0] >= 1
            and workspace.binary_pattern_cache is not None):
        initial_transition_key = workspace._binary_state_key(H)
        if initial_transition_key is not None:
            initial_transition = workspace.cached_transition(
                initial_transition_key)
    if initial_transition is not None:
        captured, result, certified = _fit_binary_cached_trajectory(
            H, max_iter, workspace, _return_initial,
            initial_key=initial_transition_key,
            initial_transition=initial_transition,
        )
        if _return_initial:
            if initial_result is None:
                initial_result = captured
                workspace.remember_fit(
                    initial_cache_key, initial_result, fixed_point=False
                )
            workspace.remember_fit(
                fit_cache_key, result, fixed_point=certified
            )
            return initial_result, result
        workspace.remember_fit(
            fit_cache_key, result, fixed_point=certified
        )
        return result


    # Tracks whether A and the per-sample costs are stale after the final
    # H update.  max_iter=0 starts stale because no A update has run.
    need_recompute = True

    captured_initial = initial_result
    for it in range(max_iter):
        # Update A given H — pass precomputed WW arrays and log_probs
        # for the fused kernel to consume; identical results to
        # recomputing inside.
        if it == 0 and initial_result is not None:
            # A cached max_iter=0 result is exactly the A(H_init) evaluation
            # that begins coordinate descent.  Reuse private copies rather
            # than evaluating the same assignments a second time.
            A = np.array(initial_result[1], copy=True, order="K")
            per_sample_cost = np.array(
                initial_result[2], copy=True, order="K"
            )
            wildcard_slots = np.array(
                initial_result[3], copy=True, order="K"
            )
            # The public fit tuple exposes only total uncapped NLL.  If H is
            # already fixed, that exact total is reused below; if H changes,
            # the final A(H) recomputation supplies per-sample uncapped costs.
            per_sample_cost_unc = None
        else:
            A, per_sample_cost, per_sample_cost_unc, wildcard_slots = discovery_assignments._update_A(
                probs_k, H, lam, cost_WW=cost_WW, WW_bin_emis=WW_bin_emis,
                log_probs=log_probs, blas_lp_cache=_blas_lp_cache,
                binary_pattern_cache=_binary_pattern_cache,
                WW_total_cost=WW_total_cost)
            workspace.apply_missing_assignment_policy(
                A, wildcard_slots, H.shape[0])

        if _return_initial and captured_initial is None:
            captured_initial = (
                np.array(H, copy=True, order="K"),
                np.array(A, copy=True, order="K"),
                np.array(per_sample_cost, copy=True, order="K"),
                np.array(wildcard_slots, copy=True, order="K"),
                0,
                float(per_sample_cost_unc.sum()),
            )
            if shared_workspace:
                workspace.remember_fit(
                    initial_cache_key,
                    captured_initial,
                    fixed_point=False,
                )

        # Retain the preceding assignments by reference.  Equality is needed
        # only after H makes zero changes; comparing on every nonterminal
        # iteration was pure memory traffic.
        previous_A = A_prev
        A_prev = A

        # Update H given A — pass precomputed cost_WW and log_probs for
        # the kernel to use directly.
        h_changes = discovery_founder_updates._update_H(probs_k, H, A, lam, cost_WW=cost_WW,
                              log_probs=log_probs,
                              h_genotype_cost=h_genotype_cost,
                              h_wildcard_cost=h_wildcard_cost)

        n_iter = it + 1
        if h_changes == 0:
            a_changed = (
                previous_A is None
                or not np.array_equal(A, previous_A)
            )
            # H is unchanged, so this iteration's A and costs already match
            # the final H even when this is the max_iter boundary.  This is a
            # complete A(H), H(A) fixed-point certificate, including when the
            # iteration cap is reached on this pass.
            need_recompute = False
            fixed_point_certified = True
            if not a_changed:
                break
            if it + 1 < max_iter:
                # With identical H, the old next iteration deterministically
                # repeated this A, then made zero H changes and converged.
                # Skip that redundant A/H pass while preserving its reported
                # iteration count exactly.
                n_iter = it + 2
                break

    if need_recompute:
        # Either the loop never ran (max_iter=0), or its last H update
        # changed H and left A stale.
        A, per_sample_cost, per_sample_cost_unc, wildcard_slots = discovery_assignments._update_A(
            probs_k, H, lam, cost_WW=cost_WW, WW_bin_emis=WW_bin_emis,
            log_probs=log_probs, blas_lp_cache=_blas_lp_cache,
            binary_pattern_cache=_binary_pattern_cache,
            WW_total_cost=WW_total_cost)
        workspace.apply_missing_assignment_policy(
            A, wildcard_slots, H.shape[0])
    # Use UNCAPPED NLL as the K-growth signal (see docstring).  When the
    # synchronized endpoint came from the workspace and H did not change,
    # that cached endpoint already holds the same uncapped total.
    total_NLL = (
        float(initial_result[5])
        if per_sample_cost_unc is None
        else float(per_sample_cost_unc.sum())
    )

    result = (H, A, per_sample_cost, wildcard_slots, n_iter, total_NLL)
    if _return_initial and captured_initial is None:
        captured_initial = _FixedKFitWorkspace._copy_fit_result(result)
    if shared_workspace:
        workspace.remember_fit(
            fit_cache_key,
            result,
            fixed_point=fixed_point_certified,
        )
    if _return_initial:
        return captured_initial, result
    return result


def _fit_at_fixed_K_with_initial(
        probs_k, H_init, lam, max_iter=50, *, workspace=None):
    """Fit once and return the synchronized start plus the final endpoint.

    The first A update of the ordinary positive-iteration fit is exactly the
    ``max_iter=0`` endpoint.  Returning it here avoids evaluating that same
    update twice while retaining the normal cache entries for both iteration
    budgets.
    """
    return _fit_at_fixed_K(
        probs_k,
        H_init,
        lam,
        max_iter=max_iter,
        workspace=workspace,
        _return_initial=True,
    )


def _fit_at_fixed_K_many_binary_frontier(
        starts, max_iter, workspace, return_initial):
    """Fit a same-K batch by sharing exact deterministic trajectory states."""
    count = len(starts)
    iteration_limit = max(0, int(max_iter))
    final_results = [None] * count
    initial_results = [None] * count
    cached_finals = [None] * count
    fit_keys = [None] * count
    initial_keys = [None] * count
    state_keys = [None] * count
    previous_assignments = [None] * count
    completed = [0] * count
    paths = [None] * count
    first_visits = [None] * count
    active = []

    for index, start in enumerate(starts):
        fit_key, cached_final = workspace.cached_fit(start, max_iter)
        fit_keys[index] = fit_key
        cached_finals[index] = cached_final
        cached_initial = None
        if return_initial:
            initial_key, cached_initial = workspace.cached_fit(start, 0)
            initial_keys[index] = initial_key
            initial_results[index] = cached_initial
        if cached_final is not None and (
                not return_initial or cached_initial is not None):
            final_results[index] = cached_final
            continue
        state_key = workspace._binary_state_key(start)
        state_keys[index] = state_key
        paths[index] = [state_key]
        first_visits[index] = {state_key: 0}
        active.append(index)

    def finish(index, result, fixed_point):
        if return_initial and initial_results[index] is None:
            raise AssertionError("batch fit finished before initial assignment")
        final_results[index] = result
        workspace.remember_fit(
            fit_keys[index], result, fixed_point=fixed_point
        )
        if return_initial:
            workspace.remember_fit(
                initial_keys[index], initial_results[index],
                fixed_point=False,
            )

    assignment_only = iteration_limit == 0
    while active:
        missing = OrderedDict()
        for index in active:
            key = state_keys[index]
            cached = (
                workspace.cached_assignment(key)
                if assignment_only
                else workspace.cached_transition(key)
            )
            if cached is None:
                missing.setdefault(key, None)
        if missing:
            matrices = [
                workspace._binary_state_from_key(key) for key in missing
            ]
            if assignment_only:
                for key, matrix in zip(missing, matrices):
                    workspace.compute_assignment_scalar(
                        matrix, state_key=key, known_missing=True)
            else:
                # Fuse multi-state frontiers even on one-core workers. A
                # singleton retains internal sample/site parallelism instead
                # of the fused kernel's outer B=1 parallel dimension.
                if len(matrices) == 1:
                    for key, matrix in zip(missing, matrices):
                        workspace.compute_transition_scalar(
                            matrix, state_key=key, known_missing=True)
                else:
                    workspace.compute_transition_batch(
                        matrices, state_keys=tuple(missing))

        next_active = []
        for index in active:
            state_key = state_keys[index]
            if assignment_only:
                assignment = workspace.cached_assignment(state_key)
                next_key = None
                h_changes = 0
            else:
                assignment, next_key, h_changes = (
                    workspace.cached_transition(state_key)
                )
            if return_initial and initial_results[index] is None:
                initial_results[index] = _fit_result_from_assignment(
                    starts[index], assignment, 0
                )

            if cached_finals[index] is not None:
                finish(index, cached_finals[index], False)
                continue

            if completed[index] >= iteration_limit:
                final_H = workspace._binary_state_from_key(state_key)
                result = _fit_result_from_assignment(
                    final_H, assignment, iteration_limit
                )
                finish(index, result, False)
                continue

            completed[index] += 1
            if h_changes == 0:
                reported = completed[index]
                previous_A = previous_assignments[index]
                if previous_A is None or not np.array_equal(
                        assignment[0], previous_A):
                    if completed[index] < iteration_limit:
                        reported += 1
                final_H = workspace._binary_state_from_key(state_key)
                result = _fit_result_from_assignment(
                    final_H, assignment, reported
                )
                finish(index, result, True)
                continue

            previous_assignments[index] = assignment[0]
            cycle_start = first_visits[index].get(next_key)
            if cycle_start is not None:
                period = completed[index] - int(cycle_start)
                if period <= 1:
                    raise AssertionError(
                        "nonzero H changes produced a one-state batch cycle"
                    )
                remaining = iteration_limit - completed[index]
                endpoint_key = paths[index][
                    int(cycle_start) + (remaining % period)
                ]
                endpoint_assignment = workspace.cached_assignment(endpoint_key)
                if endpoint_assignment is None:
                    endpoint_H = workspace._binary_state_from_key(endpoint_key)
                    endpoint_assignment = workspace.compute_transition_scalar(
                        endpoint_H
                    )[0]
                final_H = workspace._binary_state_from_key(endpoint_key)
                result = _fit_result_from_assignment(
                    final_H, endpoint_assignment, iteration_limit
                )
                finish(index, result, False)
                continue

            state_keys[index] = next_key
            first_visits[index][next_key] = completed[index]
            paths[index].append(next_key)
            next_active.append(index)
        active = next_active

    if return_initial:
        return list(zip(initial_results, final_results))
    return final_results


def _fit_at_fixed_K_many_impl(
        probs_k, H_starts, lam, max_iter=50, *, workspace=None,
        return_initial=False):
    """Fit an ordered batch of same-shaped binary starts at one fixed K.

    Exact duplicate starts are fitted once and replayed with independent result
    arrays. Production int64 binary starts use a frontier over the deterministic
    ordered-H coordinate map: equal trajectory states are evaluated once, and
    uncached states in the same frontier are processed by one compiled kernel
    whose only parallel dimension is the independent start. Each start retains
    the scalar sample/site accumulation, stable founder-update order, strict tie
    rules, convergence semantics, and max-iteration cycle endpoint.

    Unsupported dtypes/workspaces use the established scalar path, whose A/H
    kernels retain their internal sample/site parallelism. Results are exact
    scalar ``_fit_at_fixed_K`` tuples in input order; evidence and the
    workspace are shared read-only.
    """
    starts = tuple(H_starts)
    if not starts:
        return []

    first = np.asarray(starts[0])
    if first.ndim != 2:
        raise ValueError("fixed-K starts must be two-dimensional")
    start_shape = first.shape
    unique_starts = []
    unique_by_key = {}
    replay_indices = []
    for start in starts:
        matrix = np.asarray(start)
        if matrix.ndim != 2 or matrix.shape != start_shape:
            raise ValueError(
                "all fixed-K starts must have the same two-dimensional shape"
            )
        if np.any((matrix != 0) & (matrix != 1)):
            raise ValueError("fixed-K starts must be binary")
        contiguous = np.ascontiguousarray(matrix)
        start_key = (
            contiguous.dtype.str,
            tuple(int(value) for value in contiguous.shape),
            contiguous.tobytes(),
        )
        unique_index = unique_by_key.get(start_key)
        if unique_index is None:
            unique_index = len(unique_starts)
            unique_by_key[start_key] = unique_index
            unique_starts.append(start)
        replay_indices.append(unique_index)

    if workspace is None:
        # Share inexpensive evidence invariants across the batch.  Do not
        # construct the 54 MiB binary-pattern table implicitly for a small
        # one-off batch; callers that amortise it pass their existing workspace.
        workspace = _prepare_fixed_k_fit_workspace(
            probs_k, lam, binary_patterns=False
        )

    n_unique = len(unique_starts)
    # Refresh at this phase boundary so tail blocks can consume cores released
    # by completed workers.  Parallelism stays inside each numerical fit: a
    # matched N=116/N=320 benchmark showed no gain from competing Python
    # executor tasks on the shared OpenMP runtime.
    core_parallel.apply_dynamic_threads(
        max_threads=core_config.FIXED_K_FIT_MAX_THREADS
    )
    use_binary_frontier = all(
        workspace.supports_binary_transition_engine(start)
        for start in unique_starts
    )
    if use_binary_frontier:
        unique_results = _fit_at_fixed_K_many_binary_frontier(
            unique_starts, max_iter, workspace, return_initial
        )
    else:
        scalar_fit = (
            _fit_at_fixed_K_with_initial
            if return_initial else _fit_at_fixed_K
        )
        unique_results = [
            scalar_fit(
                probs_k, start, lam, max_iter=max_iter, workspace=workspace
            ) for start in unique_starts
        ]

    if n_unique == len(starts):
        return unique_results

    # Preserve the historical non-aliasing of results from separate scalar
    # calls.  The first occurrence can use the computed tuple directly; later
    # duplicate positions receive independent array copies.
    occurrence_counts = [0] * n_unique
    results = []
    for unique_index in replay_indices:
        result = unique_results[unique_index]
        if occurrence_counts[unique_index] == 0:
            results.append(result)
        else:
            if return_initial:
                initial_result, final_result = result
                results.append(
                    (
                        _FixedKFitWorkspace._copy_fit_result(initial_result),
                        _FixedKFitWorkspace._copy_fit_result(final_result),
                    )
                )
            else:
                results.append(_FixedKFitWorkspace._copy_fit_result(result))
        occurrence_counts[unique_index] += 1
    return results


def _fit_at_fixed_K_many(
        probs_k, H_starts, lam, max_iter=50, *, workspace=None):
    """Fit an ordered batch of same-shaped binary starts at one fixed K."""
    return _fit_at_fixed_K_many_impl(
        probs_k,
        H_starts,
        lam,
        max_iter=max_iter,
        workspace=workspace,
    )


def _fit_at_fixed_K_many_with_initial(
        probs_k, H_starts, lam, max_iter=50, *, workspace=None):
    """Fit a batch once, returning synchronized and final endpoints.

    Each result is ``(initial_fit, final_fit)`` in input order.  The initial
    endpoint has exactly the public ``max_iter=0`` fit-tuple representation.
    """
    return _fit_at_fixed_K_many_impl(
        probs_k,
        H_starts,
        lam,
        max_iter=max_iter,
        workspace=workspace,
        return_initial=True,
    )

import haplotype_reconstruction.core.config as core_config
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.discovery.assignments as discovery_assignments
import haplotype_reconstruction.discovery.founder_updates as discovery_founder_updates
import haplotype_reconstruction.discovery.objectives as discovery_objectives
import haplotype_reconstruction.discovery.frontier_emissions as discovery_frontier_emissions
