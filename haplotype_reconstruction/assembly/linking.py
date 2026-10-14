"""Distance-aware sum-product block scans and shared L1–L4 transition fitting."""
from __future__ import annotations


import numpy as np
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
import numba
from numba import njit, prange


np.seterr(divide='ignore', invalid='ignore')


warnings.filterwarnings("ignore", category=RuntimeWarning)


DEFAULT_ROBUSTNESS_EPSILON = 1e-2
# One default fitting cap for every hierarchy level; convergence may stop earlier.
MAX_LINKING_ITERATIONS = 20


from .micro_hmm import PreparedBlockScans
from .macro_hmm import propagate_homologue_priors
from .edge_counts import homologue_edge_log_counts
from . import partial_emissions


class TransitionMesh:
    """
    A specialized container for transition probability meshes across different gap sizes.
    This structure allows efficient lookups of Forward and Backward transition
    probabilities between genomic blocks separated by variable distances.

    Attributes:
        forward (dict): Maps gap_size (int) -> Forward Transition Dictionary.
                        Structure: { gap_size: { block_index: { ((curr_idx, curr_hap), (next_idx, next_hap)): prob } } }
        backward (dict): Maps gap_size (int) -> Backward Transition Dictionary.
                         Structure: { gap_size: { block_index: { ((curr_idx, curr_hap), (prev_idx, prev_hap)): prob } } }
    """
    def __init__(self, raw_gap_results=None):
        """
        Initializes the TransitionMesh.

        Args:
            raw_gap_results (dict, optional): A dictionary where keys are gap sizes and
                                            values are [forward_dict, backward_dict] lists.
        """
        self.forward = {}
        self.backward = {}

        if raw_gap_results:
            for gap, probs_pair in raw_gap_results.items():
                self.forward[gap] = probs_pair[0]
                self.backward[gap] = probs_pair[1]

    def __getitem__(self, gap):
        """
        Retrieve the [Forward, Backward] transition dictionaries for a specific gap size.

        Args:
            gap (int): The distance (in number of blocks) between connected nodes.

        Returns:
            list: [forward_transition_dict, backward_transition_dict]
        """
        return [self.forward.get(gap), self.backward.get(gap)]

    def __contains__(self, gap):
        """Checks if a specific gap size has been computed in this mesh."""
        return gap in self.forward

    def keys(self):
        """Returns an iterator over the gap sizes available in the mesh."""
        return self.forward.keys()

    def items(self):
        """Yields (gap, [forward, backward]) tuples."""
        for gap in self.forward:
            yield gap, [self.forward[gap], self.backward[gap]]


def _uniform_transitions(hap_keys, space_gap):
    """Uniform initial transitions as arrays (rows: haplotypes in sorted key order).

    forward[i][u, v] links block i to i + space_gap and reverse[i][u, v]
    block i to i - space_gap: every haplotype connects to every haplotype of
    the linked block with probability 1 / (its haplotype count).
    """
    forward = {i: np.full((len(hap_keys[i]), len(hap_keys[i + space_gap])),
                          1 / len(hap_keys[i + space_gap]))
               for i in range(len(hap_keys) - space_gap)}
    reverse = {i: np.full((len(hap_keys[i]), len(hap_keys[i - space_gap])),
                          1 / len(hap_keys[i - space_gap]))
               for i in range(len(hap_keys) - 1, space_gap - 1, -1)}
    return forward, reverse


def _entrywise_log(probabilities):
    """math.log of every transition probability (scalar libm logs)."""
    return np.array([math.log(value) for value in probabilities.ravel().tolist()],
                    dtype=np.float64).reshape(probabilities.shape)


def _transition_dicts(forward, reverse, hap_keys, space_gap):
    """The mesh's [forward, reverse] dicts of fitted transition arrays.

    forward: {i: {((i, src), (i + space_gap, dst)): probability}};
    reverse: {i + space_gap: {((i + space_gap, dst), (i, src)): summary}}.
    """
    forward_dicts, reverse_dicts = {}, {}
    for i in range(len(hap_keys) - space_gap):
        next_idx = i + space_gap
        curr_keys, next_keys = hap_keys[i], hap_keys[next_idx]
        values = reverse[next_idx].tolist()
        reverse_dicts[next_idx] = {
            ((next_idx, next_keys[v]), (i, curr_keys[u])): values[v][u]
            for v in range(len(next_keys)) for u in range(len(curr_keys))
        }
        values = forward[i].tolist()
        forward_dicts[i] = {
            ((i, curr_keys[u]), (next_idx, next_keys[v])): values[u][v]
            for u in range(len(curr_keys)) for v in range(len(next_keys))
        }
    return [forward_dicts, reverse_dicts]


class ViterbiBlockLikelihood:
    """Compact emissions indexed by founder-pair genotype distribution.

    Production stores three (complete) or seven (partial) values and a uint8 index per
    ordered founder pair. Dense tensors are materialized only for diagnostics.
    Scan arithmetic exponentiates the rounded float32 cells in float64.
    """
    def __init__(self, tensor, positions, state_defs, num_haps, dosages=None):
        self.log_emissions = tensor
        self.positions = positions
        self.state_defs = state_defs
        self.num_haps = num_haps
        self.dosages = dosages

    @property
    def tensor(self):
        if self.dosages is None:
            return self.log_emissions
        return np.take_along_axis(
            self.log_emissions, self.dosages[None,:,:], axis=2)


class ViterbiBlockList:
    """Simple container for ViterbiBlockLikelihood objects."""
    def __init__(self, blocks_list):
        self.blocks = blocks_list
    def __len__(self):
        return len(self.blocks)
    def __getitem__(self, idx):
        return self.blocks[idx]
    def __iter__(self):
        return iter(self.blocks)


@njit(cache=True, parallel=True, nogil=True)
def _genotype_emission_kernel(samples, epsilon):
    """Uniform genotype mixture, evaluated once per dosage without extra clipping.

    The mixture already gives positive likelihood to incompatible observations;
    an additional log floor would suppress their distinguishing evidence.
    """
    result = np.empty(samples.shape, dtype=np.float32)
    one_minus_eps = 1.0 - epsilon
    eps_third = epsilon * (1.0 / 3.0)
    for s in prange(samples.shape[0]):
        for site in range(samples.shape[1]):
            for dosage in range(3):
                probability = np.float64(samples[s, site, dosage]) * one_minus_eps + eps_third
                result[s, site, dosage] = np.log(max(probability, 1e-300))
    return result


def _worker_generate_viterbi_emissions(args):
    """
    Worker function to calculate P(Data_site | State) for all sites and states.
    Uses the Robust Mixture Model: P = (1-e)*Model + e*Uniform.
    """
    samples_matrix, block_hap, params = args
    # Robustness parameter to prevent outlier sites from zeroing out the likelihood
    epsilon = params.get('robustness_epsilon', DEFAULT_ROBUSTNESS_EPSILON)
    hap_dict = block_hap.haplotypes
    # Handling Flags
    if block_hap.keep_flags is not None:
        keep_flags = np.array(block_hap.keep_flags, dtype=bool)
    else:
        keep_flags = np.ones(len(block_hap.positions), dtype=bool)


    panel = assembly_observations.founder_inference_panel_from_block_result(block_hap)
    # Unknown alleles are integrated predictively, never cast to a called 0/1.
    # Sites unsupported by every founder cannot establish local linkage.
    emission_keep_flags = keep_flags & np.any(panel.called, axis=0)

    hap_keys = sorted(list(hap_dict.keys()))
    num_haps = len(hap_keys)
    if panel.keys != tuple(hap_keys):
        raise ValueError("founder panel and haplotype key order disagree")

    # State Definitions: Map flattened index 0..K-1 to (h1, h2)
    # Full Directed State Space (no symmetry folding)
    idx_i, idx_j = np.unravel_index(np.arange(num_haps ** 2), (num_haps, num_haps))
    state_defs = np.stack([idx_i, idx_j], axis=1).astype(np.int32)

    samples_masked = samples_matrix[:, emission_keep_flags,:]
    valid_positions = np.array(block_hap.positions)[emission_keep_flags].astype(np.int64)

    # --- PROBABILISTIC MIXTURE CALCULATION ---
    called = panel.called[:, emission_keep_flags]
    q = panel.q[:, emission_keep_flags]
    if np.all(called):
        alleles = q.astype(np.uint8)
        dosages = np.ascontiguousarray(
            (alleles[:, None,:] + alleles[None,:,:]).reshape(num_haps ** 2, -1).T)
        ll_per_site = _genotype_emission_kernel(
            np.ascontiguousarray(samples_masked), float(epsilon))
    else:
        alleles = np.ascontiguousarray(np.where(called, q, -1), dtype=np.int8)
        source_rows = partial_emissions.source_row_ids(block_hap)[:, emission_keep_flags]
        dosages = partial_emissions.pair_codes(alleles, np.ascontiguousarray(source_rows))
        ll_per_site = partial_emissions.compact_log_emissions(
            np.ascontiguousarray(samples_masked), float(epsilon))
    return ViterbiBlockLikelihood(
        ll_per_site, valid_positions, state_defs, num_haps, dosages)


def _single_threaded_emission_worker(args):
    # Numba masks are thread-local: the caller's mask is not inherited by
    # a fresh ThreadPoolExecutor thread.
    with core_parallel.numba_thread_scope(1):
        return _worker_generate_viterbi_emissions(args)


def generate_viterbi_block_emissions(
        samples_matrix, sample_sites, block_results, num_processes=16):
    """
    Parallel generator for ViterbiBlockLikelihood objects used in the scan.

    Uses ThreadPoolExecutor (not Pool) because the worker's heavy compute
    is a numba kernel decorated nogil=True, so threads release the GIL
    during the kernel and parallelize effectively without pickling the
    large sample arrays through a process boundary.

    Partial sites use predictive marginalization, never output allele filling.
    Completely unsupported sites are excluded.
    """
    params = {'robustness_epsilon': DEFAULT_ROBUSTNESS_EPSILON}

    if num_processes > 1 and len(block_results) > 1:
        # Prepare tasks (references only, no pickling needed with threads)
        tasks = []
        for block in block_results:
            indices = np.searchsorted(sample_sites, block.positions)
            block_samples = samples_matrix[:, indices,:]
            tasks.append((block_samples, block, params))

        with ThreadPoolExecutor(max_workers=min(num_processes, len(tasks))) as executor:
            results = list(executor.map(_single_threaded_emission_worker, tasks))
        del tasks
    else:
        # Sequential: process one block at a time, free each before the next
        results = []
        for block in block_results:
            indices = np.searchsorted(sample_sites, block.positions)
            block_samples = samples_matrix[:, indices,:]
            result = _worker_generate_viterbi_emissions((block_samples, block, params))
            del block_samples
            results.append(result)

    core_parallel.malloc_trim()

    return ViterbiBlockList(results)


def _prepare_block_scans(raw_blocks, recomb_rate, chromosome_map=None,
                         dynamic_cores_fn=None):
    """Fixed scan data of every block; blocks are independent and prepared concurrently
    (assembly_chimera_kernels.concurrent_map) when dynamic_cores_fn gives several threads."""
    if chromosome_map is not None:
        recomb_rate = chromosome_map.fallback_rate_per_bp

    def prepare(block, threads):
        if threads is not None:
            numba.set_num_threads(threads())
        distances = None
        if chromosome_map is not None and chromosome_map.has_map:
            distances = np.ascontiguousarray(chromosome_map.interval_morgans(
                block.positions[:-1], block.positions[1:]))
        return PreparedBlockScans(
            block.log_emissions, block.positions, recomb_rate, block.state_defs,
            block.num_haps, distances, dosages=block.dosages)
    return assembly_chimera_kernels.concurrent_map(prepare, raw_blocks, dynamic_cores_fn)


def global_forward_backward_pass(raw_blocks, log_transitions, space_gap, recomb_rate,
                                 genetic_distances_by_block=None,
                                 prepared_scans=None, dynamic_cores_fn=None,
                                 edge_messages_only=False):
    """Propagate through gap-residue chains with fixed block scans reused.

    ``log_transitions[i]`` is the log forward transition matrix from block i to
    block i + space_gap (rows and columns: haplotypes in sorted key order).
    Boundary priors are exactly uniform and their scan results are cached.
    State-dependent interior priors depend on the current fitted transitions.
    Forward and backward likelihoods use the same forward transition factors.
    Reverse mesh summaries are not backward likelihood operators.
    With edge_messages_only, omit terminal S and initial R messages unused by
    transition fitting and return None for the uncomputed likelihood diagnostic.
    The forward and backward recursions read only the fixed transitions, so they
    run concurrently, each with a share of dynamic_cores_fn's threads.
    """
    num_blocks = len(raw_blocks)
    if prepared_scans is None:
        prepared_scans = [
            PreparedBlockScans(
                block.log_emissions, block.positions, recomb_rate, block.state_defs,
                block.num_haps,
                None if genetic_distances_by_block is None
                else genetic_distances_by_block[i], dosages=block.dosages)
            for i, block in enumerate(raw_blocks)
        ]
    forward_stop = num_blocks - space_gap if edge_messages_only else num_blocks
    backward_stop = space_gap - 1 if edge_messages_only else -1

    def forward(threads):
        S_results = [None] * num_blocks
        for i in range(forward_stop):
            if threads is not None:
                numba.set_num_threads(threads())
            priors = None
            if i >= space_gap:
                previous = i - space_gap
                priors = propagate_homologue_priors(
                    S_results[previous], np.ascontiguousarray(log_transitions[previous]))
            if priors is None:
                # Normalized initial diploid prior for each independent chain.
                k = raw_blocks[i].num_haps
                priors = np.full((raw_blocks[i].log_emissions.shape[0], k * k),
                                 -2.0 * math.log(k))
            S_results[i] = prepared_scans[i].scan(priors)
        return S_results

    def backward(threads):
        R_results = [None] * num_blocks
        for i in range(num_blocks - 1, backward_stop, -1):
            if threads is not None:
                numba.set_num_threads(threads())
            priors = None
            if i < num_blocks - space_gap:
                following = i + space_gap
                # Transpose for the contraction, NOT a reverse conditional.
                priors = propagate_homologue_priors(
                    R_results[following], np.ascontiguousarray(log_transitions[i].T))
            R_results[i] = prepared_scans[i].scan(priors, backward=True)
        return R_results

    S_results, R_results = assembly_chimera_kernels.concurrent_map(
        lambda recursion, threads: recursion(threads), (forward, backward), dynamic_cores_fn)

    # The gap partitions the batch into independent residue-class chains.
    total_ll = None
    if not edge_messages_only:
        total_ll = 0.0
        for terminal in range(max(0, num_blocks - space_gap), num_blocks):
            total_ll += float(np.sum(core_numerics.lse_axis_last(S_results[terminal])))
    return S_results, R_results, total_ll


@njit(cache=True, nogil=True)
def _smooth_normalize_kernel(data_log_count, log_pseudo, min_log_prob, mix_rate):
    """Per-row smoothing + normalize + robust-mixture kernel.

    For each row u_i of data_log_count, computes the production-equivalent:

      1. smoothed[v_i] = np.logaddexp(data_log_count[u_i, v_i], log_pseudo)
         # Add Pseudocounts (prevents death spiral).  np.logaddexp(-inf, x)
         # returns x, so cells with data_log_count == -inf become log_pseudo.
      2. row_total = logsumexp(smoothed)
         # Equivalent to analysis_utils.lse_scalar(list(targets.values())).
      3. log_p[v_i] = max(smoothed[v_i] - row_total, min_log_prob)
         # Clip at MIN_LOG_PROB to prevent numerical underflow.
      4. p[v_i] = exp(log_p[v_i])
      5. norm_p[v_i] = p[v_i] / sum(p)  (or p[v_i] if sum is 0; never
         happens in practice since smoothed cells are >= log_pseudo
         > -inf, but matches the original's `if renorm_sum == 0:
         renorm_sum = 1.0` defensive branch).
      6. final_p[u_i, v_i] = norm_p[v_i] * (1 - mix_rate)
                              + (1 / n_n) * mix_rate
         # Robust Mixture (1% Uniform when mix_rate=0.01).

    Output is a dense (n_c, n_n) matrix; the wrapper does the dict-write
    back to the production {((i, src), (next_idx, dst)): final_p} format
    because the key tuples involve arbitrary objects that numba can't
    construct.

    Mathematically identical (verified byte-equivalent at 1 ULP across all
    K) to the original python loops; the kernel just collapses the per-cell
    interpreter overhead.  Per-call speedup ranges 27x at K=5 to 78x at
    K=36 (measured microbench); aggregate over all gaps and EM iterations,
    this saves ~0.2 sec per pipeline run at K=10 and ~2.6 sec at K=36.

    Args:
        data_log_count: (n_c, n_n) float64 -- homologue edge log counts,
            with -inf for missing cells.
        log_pseudo: float -- smoothing pseudocount in log space (typically
            math.log(PSEUDO_COUNT) where PSEUDO_COUNT = 0.1).
        min_log_prob: float -- numerical floor on log-probabilities
            (typically -10.0).
        mix_rate: float -- robust-mixture uniform weight (typically 0.01).

    Returns:
        (n_c, n_n) float64 -- final probabilities ready for dict-write.
    """
    n_c, n_n = data_log_count.shape
    final_p = np.zeros((n_c, n_n), dtype=np.float64)
    if n_n == 0:
        return final_p
    uniform_val = 1.0 / n_n

    for u_i in range(n_c):
        # Step 1: smoothing via logaddexp(d, log_pseudo).
        smoothed_row = np.empty(n_n, dtype=np.float64)
        for v_i in range(n_n):
            d = data_log_count[u_i, v_i]
            if d == -np.inf:
                # logaddexp(-inf, log_pseudo) == log_pseudo
                smoothed_row[v_i] = log_pseudo
            else:
                m = d if d > log_pseudo else log_pseudo
                smoothed_row[v_i] = m + np.log(np.exp(d - m)
                                                + np.exp(log_pseudo - m))

        # Step 2: row_total = logsumexp(smoothed_row).  Done via two-pass
        # (find max, then sum exp differences) for numerical stability,
        # matching analysis_utils.lse_scalar.
        m = -np.inf
        for v_i in range(n_n):
            if smoothed_row[v_i] > m:
                m = smoothed_row[v_i]
        if m == -np.inf:
            row_total = -np.inf
        else:
            s = 0.0
            for v_i in range(n_n):
                s += np.exp(smoothed_row[v_i] - m)
            row_total = m + np.log(s)

        # Steps 3-4: log_p = max(smoothed - row_total, min_log_prob); p = exp(log_p).
        renorm_sum = 0.0
        temp_p = np.empty(n_n, dtype=np.float64)
        for v_i in range(n_n):
            if row_total == -np.inf:
                log_p = -np.inf
            else:
                log_p = smoothed_row[v_i] - row_total
            if log_p < min_log_prob:
                log_p = min_log_prob
            p = np.exp(log_p)
            temp_p[v_i] = p
            renorm_sum += p

        # Defensive: matches the original `if renorm_sum == 0: renorm_sum = 1.0`.
        # In practice renorm_sum can't be 0 given that log_p is clipped at
        # min_log_prob > -inf, but preserved for exact behavioural parity.
        if renorm_sum == 0.0:
            renorm_sum = 1.0

        # Steps 5-6: re-normalize + robust mixture.
        for v_i in range(n_n):
            norm_p = temp_p[v_i] / renorm_sum
            final_p[u_i, v_i] = norm_p * (1.0 - mix_rate) + uniform_val * mix_rate

    return final_p


def update_transitions_layered_hmm(S_results, R_results, log_transitions, space_gap,
                                   dynamic_cores_fn=None, learning_rate=1.0):
    """Fit forward transitions from posterior homologue counts once per edge.

    S and R are forward and adjoint backward likelihood messages from the same
    gap-residue-chain HMM. The cubic contraction counts both homologues. Retain
    pseudocounts, the probability floor, uniform robustness and forward damping.
    Reverse mesh summaries are column-normalizations of this shared regularized
    edge evidence; their source masses are posterior-support summaries, not
    unconditional chain-state marginals. They are never used as beta operators.
    ``log_transitions[i]`` holds the current log forward transitions from
    block i to i + space_gap. Returns (forward, reverse): forward[i] the new
    (n_i, n_{i+gap}) transition probabilities and reverse[i + gap] the
    (n_{i+gap}, n_i) reverse summaries, rows and columns in sorted key order.

    dynamic_cores_fn refreshes the numerical thread allocation before each
    edge, allowing this M-step to use cores released by peer batch workers.
    """
    new_trans_fwd = {}
    new_trans_bwd = {}
    num_blocks = len(S_results)

    MIN_LOG_PROB = -10.0
    PSEUDO_COUNT = 0.1
    LOG_PSEUDO = math.log(PSEUDO_COUNT)

    # -----------------------------------------------------
    # LOOP 1: FORWARD TRANSITION UPDATE (Earlier -> Later)
    # -----------------------------------------------------
    for i in range(num_blocks - space_gap):
        # Dynamic thread reallocation hook: rescale numba threads
        # before processing this block.  See the function docstring.
        if dynamic_cores_fn is not None:
            try:
                numba.set_num_threads(dynamic_cores_fn())
            except Exception:
                pass

        next_idx = i + space_gap
        hap_log_t = log_transitions[i]
        data_log_count = homologue_edge_log_counts(
            np.ascontiguousarray(S_results[i]), np.ascontiguousarray(R_results[next_idx]),
            np.ascontiguousarray(hap_log_t))

        # Smoothing + per-row normalize + robust-mixture, delegated to a
        # numba kernel.  Mathematically: for each row u_i,
        #   smoothed[v] = logaddexp(data_log_count[u_i, v], LOG_PSEUDO)
        #   row_total   = logsumexp(smoothed)
        #   log_p[v]    = max(smoothed[v] - row_total, MIN_LOG_PROB)
        #   norm_p[v]   = exp(log_p[v]) / sum(exp(log_p))
        #   final[u_i, v] = norm_p[v] * (1 - mix_rate) + (1/n_n) * mix_rate
        # See _smooth_normalize_kernel for details.
        final_p_mat = _smooth_normalize_kernel(
            data_log_count, LOG_PSEUDO, MIN_LOG_PROB, 0.01)

        # Preserve the established damping of forward parameters. The
        # reverse table is a local evidence summary, never a beta operator.
        final_p_mat = ((1.0 - learning_rate) * np.exp(hap_log_t)
                       + learning_rate * final_p_mat)
        log_row_mass = np.logaddexp.reduce(
            np.logaddexp(data_log_count, LOG_PSEUDO), axis=1)
        log_joint = log_row_mass[:, None] + np.log(final_p_mat)
        new_trans_bwd[next_idx] = np.exp((log_joint - np.logaddexp.reduce(
            log_joint, axis=0)[None,:]).T)
        new_trans_fwd[i] = final_p_mat

    return new_trans_fwd, new_trans_bwd


def _max_transition_probability_change(old_probs, new_probs):
    """Return the largest absolute change across transition entries (finite
    probabilities, so the maximum does not depend on the entry order)."""
    max_change = 0.0
    for old_tables, new_tables in zip(old_probs, new_probs):
        for block_idx, new_table in new_tables.items():
            if new_table.size:
                change = float(np.max(np.abs(new_table - old_tables[block_idx])))
                if change > max_change:
                    max_change = change
    return max_change


def calculate_hap_transition_probabilities(haps_data,
                                           max_num_iterations=MAX_LINKING_ITERATIONS, space_gap=1,
                                           recomb_rate=5e-7, learning_rate=1.0,
                                           min_cutoff_change=0.001,
                                           precalculated_viterbi_emissions=None,
                                           dynamic_cores_fn=None, chromosome_map=None,
                                           prepared_scans=None, diagnostics=None):
    """Fit a regularized gap-residue-chain HMM from expected edge counts.

    Runs forward/backward inference and damped count-based updates for up to
    max_num_iterations or until
    every smoothed transition probability changes by at most
    ``min_cutoff_change``.  Parameter-space convergence is invariant to a
    state-independent offset in the emission log scores.

    Args:
        min_cutoff_change: Maximum absolute transition-probability change
            allowed at convergence.
        precalculated_viterbi_emissions: Required ViterbiBlockList.
        dynamic_cores_fn: Optional callable returning the current core
            allocation for this worker.  Called at the top of each EM
            iteration (here) and the top of each M-step block loop (in
            update_transitions_layered_hmm); the returned value is
            passed to numba.set_num_threads so parallel kernels pick
            up the live thread count.  Mirrors the in-flight rescaling
            allocation of the outer batch pool. No-op when None.
    """
    if precalculated_viterbi_emissions is None:
        raise ValueError(
            "precalculated_viterbi_emissions is required (pass a "
            "ViterbiBlockList from generate_viterbi_block_emissions)"
        )
    raw_blocks = precalculated_viterbi_emissions
    # Shared across all EM iterations, and all gaps when invoked by the mesh.
    if prepared_scans is None:
        prepared_scans = _prepare_block_scans(
            raw_blocks, recomb_rate, chromosome_map, dynamic_cores_fn)

    # Cache sorted hap keys once (never change across EM iterations). The
    # transitions stay arrays in this key order during the EM; the mesh
    # dicts are built once from the final fit.
    hap_keys_cache = [sorted(list(b.haplotypes.keys())) for b in haps_data]

    current_trans = _uniform_transitions(hap_keys_cache, space_gap)

    for it in range(max_num_iterations):
        # Dynamic thread rescaling: pick up freed cores from peer
        # workers that have finished.  No-op when dynamic_cores_fn is
        # None.  Errors are silently absorbed (matches the robustness
        # posture in block_haplotypes._update_dynamic_threads).
        if dynamic_cores_fn is not None:
            try:
                numba.set_num_threads(dynamic_cores_fn())
            except Exception:
                pass

        # Match decay schedule
        effective_lr = learning_rate * (0.9 ** it)
        effective_lr = max(effective_lr, 0.1)
        log_transitions = {i: _entrywise_log(value) for i, value in current_trans[0].items()}

        # E-Step
        S_res, R_res, _current_ll = global_forward_backward_pass(
            raw_blocks, log_transitions, space_gap, recomb_rate,
            prepared_scans=prepared_scans, dynamic_cores_fn=dynamic_cores_fn,
            edge_messages_only=diagnostics is None,
        )

        # M-Step
        next_trans = update_transitions_layered_hmm(
            S_res, R_res, log_transitions, space_gap,
            dynamic_cores_fn=dynamic_cores_fn,
            learning_rate=effective_lr,
        )

        # The M-step already applies damping. Stop in parameter space, which
        # is invariant to a state-independent offset in emission log scores.
        max_transition_change = _max_transition_probability_change(
            current_trans, next_trans)
        current_trans = next_trans
        if diagnostics is not None:
            diagnostics.append(dict(iteration=it + 1, log_likelihood=_current_ll,
                                    max_change=max_transition_change))
        if max_transition_change <= min_cutoff_change:
            break

    return _transition_dicts(*current_trans, hap_keys_cache, space_gap)


def generate_transition_probability_mesh(
        haps_data,
        max_num_iterations=MAX_LINKING_ITERATIONS, recomb_rate=5e-7,
        precalculated_viterbi_emissions=None,
        dynamic_cores_fn=None, chromosome_map=None, max_gap=None):
    """Fit consumed gaps, reusing fixed emissions/zero-prior scans within this mesh.

    Gaps retain independent EM fits and the original convergence settings.
    Process parallelism belongs to the outer batch pool. Inside a batch the gap
    fits share only the fixed block scans, so they run concurrently, each with a
    share of dynamic_cores_fn's threads (assembly_chimera_kernels.concurrent_map;
    a fit that converges leaves its threads to the rest); numerical thread
    allocation is refreshed at each EM iteration and block boundary.
    """
    if precalculated_viterbi_emissions is None:
        raise ValueError("precalculated_viterbi_emissions is required")
    prepared = _prepare_block_scans(
        precalculated_viterbi_emissions, recomb_rate, chromosome_map,
        dynamic_cores_fn)
    last_gap = len(haps_data) - 1
    if max_gap is not None:
        last_gap = min(last_gap, max_gap)
    gaps = range(1, last_gap + 1)

    def fit(gap, threads):
        return calculate_hap_transition_probabilities(
            haps_data, max_num_iterations=max_num_iterations,
            space_gap=gap, recomb_rate=recomb_rate,
            precalculated_viterbi_emissions=precalculated_viterbi_emissions,
            dynamic_cores_fn=threads, chromosome_map=chromosome_map,
            prepared_scans=prepared)
    results = dict(zip(gaps, assembly_chimera_kernels.concurrent_map(fit, gaps, dynamic_cores_fn)))
    core_parallel.malloc_trim()
    mesh = TransitionMesh(results)
    mesh.scan_diagnostics = {
        "computed_scans": sum(block.scan_calls for block in prepared),
        "fixed_prior_cache_hits": sum(block.cached_calls for block in prepared),
        "log_fallback_samples": sum(block.fallback_samples for block in prepared),
    }
    return mesh


import haplotype_reconstruction.assembly.chimera_kernels as assembly_chimera_kernels
import haplotype_reconstruction.assembly.observations as assembly_observations
import haplotype_reconstruction.core.numerics as core_numerics
import haplotype_reconstruction.core.parallel as core_parallel
