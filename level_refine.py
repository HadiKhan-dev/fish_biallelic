"""
level_refine — per-level haplotype refinement (truth-free, refinement-only, SELF-SCORING)
=========================================================================================

A SINGLE level-generic operation, run at the END of each assembly level, that makes
that level's haplotypes more accurate using ONLY that level's own haplotypes and the
genotype probabilities, and returns the refined blocks for the caller to SAVE as that
level's checkpoint.

SELF-CONTAINED (no access to blocks below this level)
-----------------------------------------------------
A level-n super-block is scored AGAINST ITSELF: each sample is painted as a diploid
mosaic of the super-block's OWN haplotypes across bins of the super-block's window,
directly from global_probs.  We do NOT decompose into the underlying L0 blocks and we
do NOT recover keypaths through them.  compute_subblock_emissions / score_path_set /
paint_samples_viterbi all operate on a single synthetic "block" = the super-block
itself (it already has .positions and .haplotypes).  Consequences:
  * the refinement at level n uses ONLY level-n haps + global_probs -- a level never
    reaches below itself;
  * no l0_blocks argument; integrates identically at L1, L2, L3, L4 with no need to
    reload pruned lower-level blocks.

NOTE on the objective: this self-block likelihood is a DIFFERENT (and, for refining a
level-n hap, more appropriate) objective than the earlier L0-keypath scoring used by
the L1->L2 prototype.  It asks "does this level-n haplotype, as a single sequence,
explain the data better?" rather than "does the best L0-mosaic through this window
improve?".  Because it is a new objective, it is RE-VALIDATED on truth (the test
runner's truth-accuracy check) before integration -- the prototype's results are not
inherited.

WHAT IT DOES (per block, in parallel across blocks)
---------------------------------------------------
  1. paint all N samples against the block's own haps over its window (self-block).
  2. FIXED-POINT rep loop (RE-PAINT each pass so carriers stay consistent with the
     evolving haps): for each rep hap, gather its carriers and re-derive a cleaner
     version H (v2 Viterbi-partner re-derivation).
  3. score dLL = LL(block with H at estar) - LL(block with current hap), where LL is
     the self-block painting log-likelihood over all N samples (score_path_set on the
     single super-block).
  4. if H is close to an existing hap (pmin < EPS_PRESENT) AND dLL > MIN_DLL, REPLACE.
     Repeat passes until a full pass makes no replace (true fixed point), capped at
     max_block_iter (default 3x #haps); cap-hit is flagged loudly (should not happen).
     REFINEMENT ONLY: never ADDs a new haplotype (recovery is a separate operation).

CONVERGENCE: a replace is accepted only if dLL > MIN_DLL > 0, so each accepted step
strictly raises the (bounded) self-block LL; cur is a finite binary matrix, so a
strictly increasing sequence cannot revisit a state and must terminate.

Forward-only checkpoint policy: nothing is written to any other level's checkpoint;
the refinement is saved as THIS level's checkpoint.  Truth is used ONLY for the test
runner's validation labels; the refinement itself is truth-free.

Public API
----------
  refine_level(blocks, global_probs, global_sites,
               n_workers=112, eps_present=2.0, paint_penalty=10.0,
               clean_span_frac=0.6, min_carriers=2, switch_pen=4.0,
               log_invalid=-7.0, near_bonus=0.3, support_thresh=0.75,
               max_block_iter=None, min_dll=1e-6,
               rep_selector=None, return_actions=False, pool=None, verbose=True)
      -> refined_blocks   (or (refined_blocks, actions_by_block) if return_actions)

Note: NO l0_blocks argument (self-scoring).  paint_penalty default 10.0 is the L1
value; the caller may pass compute_penalty([block])-style values per level, but for
self-block scoring the penalty governs bin-to-bin recombination within the block.

Workers MUST live in an importable module (forkserver re-imports modules, not
__main__).  The pool/forkserver bootstrap lives in the caller (pipeline or test).
"""
import os
import copy

import thread_config  # noqa: F401  (forkserver preload + single-thread BLAS; before numpy/numba)

import numpy as np


# Worker-side per-call data (set in the pool initializer; small + shared)
_WK = {}
_OPT = {}


def _hamming_pct(a, b):
    return 100.0 * float(np.mean(a != b))


def _collapse(hap):
    arr = np.asarray(hap)
    if arr.ndim > 1:
        return np.argmax(arr, axis=1).astype(np.int8)
    return arr.astype(np.int8)


def _seq_to_prob_block(seq_segment):
    seg = np.asarray(seq_segment).astype(np.int8)
    onehot = np.zeros((seg.shape[0], 2), dtype=np.float64)
    onehot[seg == 0, 0] = 1.0
    onehot[seg == 1, 1] = 1.0
    return onehot


class _SynblockView:
    """Minimal block-like object (positions + haplotypes) so chimera_resolution's
    compute_subblock_emissions / paint can treat a super-block (or a candidate hap
    matrix over a window) as a single 'block'.  haplotypes is {key: (n_sites,2)}."""
    __slots__ = ('positions', 'haplotypes')

    def __init__(self, positions, hap_seqs):
        self.positions = np.asarray(positions)
        self.haplotypes = {i: _seq_to_prob_block(hap_seqs[i]) for i in range(hap_seqs.shape[0])}


def _viterbi_partner(E_c, switch_pen):
    nbins, m = E_c.shape
    V = np.empty((nbins, m), dtype=np.float64)
    back = np.zeros((nbins, m), dtype=np.int64)
    V[0] = E_c[0]
    ar = np.arange(m)
    for b in range(1, nbins):
        Vp = V[b - 1]
        order = np.argsort(Vp)
        am, sec = int(order[-1]), int(order[-2])
        m1, m2 = Vp[am], Vp[sec]
        best_other = np.full(m, m1)
        best_other[am] = m2
        switch_src = np.full(m, am, dtype=np.int64)
        switch_src[am] = sec
        stay = Vp
        switch_val = best_other - switch_pen
        take_stay = stay >= switch_val
        V[b] = E_c[b] + np.where(take_stay, stay, switch_val)
        back[b] = np.where(take_stay, ar, switch_src)
    path = np.empty(nbins, dtype=np.int64)
    path[-1] = int(np.argmax(V[-1]))
    for b in range(nbins - 1, 0, -1):
        path[b - 1] = back[b, path[b]]
    return path


def _emissions_for(positions, hap_seqs, spb, num_threads=None):
    """compute_subblock_emissions for ONE synthetic block (the super-block itself).
    num_threads: int or callable forwarded to the numba kernel (dynamic-thread scaling
    so a worker that holds many cores at the upper levels actually uses them)."""
    import chimera_resolution
    gp = _WK['global_probs']; gs = _WK['global_sites']
    block = _SynblockView(positions, hap_seqs)
    bi = np.searchsorted(gs, np.asarray(positions))
    lo, hi = int(bi.min()), int(bi.max())
    bprobs = np.ascontiguousarray(gp[:, lo:hi + 1, :])
    bsites = np.ascontiguousarray(gs[lo:hi + 1])
    sub_em = chimera_resolution.compute_subblock_emissions(
        [block], bprobs, bsites, spb, num_threads=num_threads)
    return sub_em


def _paint_self(positions, hap_seqs, spb, penalty, num_threads=None):
    """Paint all N samples against hap_seqs over `positions` as a SINGLE block.
    Returns (sp [N, n_bins], K, bof [n_sites]) or None."""
    import chimera_resolution
    gp = _WK['global_probs']
    sub_em = _emissions_for(positions, hap_seqs, spb, num_threads=num_threads)
    total_bins = sub_em[0]['n_bins']
    N = gp.shape[0]
    # single block -> keypath per sample is over this one block; paint_samples_viterbi
    # expects keypaths as the per-hap key lists; with one block each hap key is itself.
    keypaths = [[i] for i in range(hap_seqs.shape[0])]
    sp, K = chimera_resolution.paint_samples_viterbi(
        keypaths, sub_em, penalty, N, num_threads=num_threads)
    n_sites = len(np.asarray(positions))
    bof = (np.arange(n_sites) // spb).astype(np.int32)
    if sp.shape[1] != total_bins or bof.max() + 1 != total_bins:
        return None
    return sp.astype(np.int32), int(K), bof


def _score_self(positions, hap_seqs, spb, penalty, num_threads=None):
    """Self-block painting LL over all N samples for hap_seqs over `positions`."""
    import chimera_resolution
    gp = _WK['global_probs']
    sub_em = _emissions_for(positions, hap_seqs, spb, num_threads=num_threads)
    N = gp.shape[0]
    keypaths = [[i] for i in range(hap_seqs.shape[0])]
    return float(chimera_resolution.score_path_set(
        keypaths, sub_em, float(penalty), N, num_threads=num_threads))


def _carriers_of(sp, K, rep_idx):
    fi = sp // K; fj = sp % K
    use = (fi == rep_idx) | (fj == rep_idx)
    frac = use.mean(axis=1)
    return set(np.nonzero(frac >= _OPT['clean_span_frac'])[0].tolist())


def _refine_one_block(task):
    """TASK (worker): refine one super-block, SELF-SCORING (no L0 blocks).
    task = (j, positions, seqs, rep_indices_or_None)."""
    import hierarchical_assembly as _ha
    try:
        import chimera_resolution
        import numba
        # Register this worker as active and take an initial dynamic thread allocation,
        # mirroring hierarchical_assembly._process_single_batch.  This is what lets the
        # inner numba kernels (emissions/paint/score) use total_cores//active_workers
        # threads -- essential at L2/L3/L4 where blocks < workers (e.g. L4: 1-2 blocks,
        # so 1-2 workers must each use ~all cores, not 1 thread).
        if _ha._ACTIVE_COUNTER is not None:
            with _ha._ACTIVE_COUNTER.get_lock():
                _ha._ACTIVE_COUNTER.value += 1
        try:
            numba.set_num_threads(max(1, _ha._get_dynamic_threads()))
        except Exception:
            pass
        # callable re-resolved by the chimera kernels each phase, so as peers finish a
        # surviving worker scales UP its threads (the remainder-pool reassignment).
        dyn_fn = _ha._get_dynamic_threads if _ha._ACTIVE_COUNTER is not None else None

        j, pos, seqs, rep_indices = task
        gp = _WK['global_probs']; N = gp.shape[0]
        site_to_idx = _WK['site_to_idx']
        pos = np.asarray(pos)
        present0 = np.asarray(seqs, dtype=np.int8)
        cur = present0.copy()
        actions = []
        idx_g = np.array([site_to_idx[int(p)] for p in pos])

        # dosage over this window -- from genotype probs, INDEPENDENT of cur.
        d = gp[:, idx_g, 1] * 1.0 + gp[:, idx_g, 2] * 2.0
        d_r = np.clip(np.rint(d), 0, 2).astype(np.int16)

        eps = _OPT['eps_present']
        support_thresh = _OPT['support_thresh']
        switch_pen = _OPT['switch_pen']
        log_invalid = _OPT['log_invalid']
        near_bonus = _OPT['near_bonus']
        min_carriers = _OPT['min_carriers']
        min_dll = _OPT.get('min_dll', 1e-6)
        penalty = float(_OPT['paint_penalty'])

        # binning for THIS single super-block window; used for BOTH paint and score
        # so the bin structure (bof) is consistent throughout.
        synblock = _SynblockView(pos, cur)
        spb = chimera_resolution.compute_spb([synblock])

        max_block_iter = _OPT.get('max_block_iter', None)
        if max_block_iter is None:
            max_block_iter = 3 * cur.shape[0]

        converged = False
        _bi_iter = 0
        while _bi_iter < max_block_iter:
            _bi_iter += 1

            # RE-PAINT current haps (self-block) -> consistent carrier assignment
            pj = _paint_self(pos, cur, spb, penalty, num_threads=dyn_fn)
            if pj is None:
                if _bi_iter == 1:
                    return dict(j=j, modified=False, refined_seqs=None, actions=[],
                                status='paint_failed')
                break
            sp, K_p, bof = pj

            pass_reps = rep_indices if rep_indices is not None else list(range(cur.shape[0]))
            changed_this_pass = False
            for rep in pass_reps:
                if rep < 0 or rep >= cur.shape[0]:
                    continue
                carriers = _carriers_of(sp, K_p, rep)
                if len(carriers) < min_carriers:
                    continue

                # ---- re-derive H from this carrier set (v2 Viterbi partner) ----
                m, L = cur.shape
                C = np.array(sorted(carriers), dtype=np.int64)
                d_rC = d_r[C]
                fi = (sp // K_p)[C]; fj = (sp % K_p)[C]
                counts = np.array([(fi == h).sum() + (fj == h).sum() for h in range(K_p)])
                f_side = int(np.argmax(counts))
                N_tilde = cur[f_side] if f_side < cur.shape[0] else cur[0]
                nbins = int(bof.max()) + 1
                Bmat = np.zeros((L, nbins), dtype=np.float64)
                Bmat[np.arange(L), bof] = 1.0
                E = np.zeros((C.size, nbins, m), dtype=np.float64)
                for k in range(m):
                    diff = d_rC - cur[k][None, :]
                    invalid = (diff < 0) | (diff > 1)
                    extracted = np.clip(diff, 0, 1)
                    nearmatch = (extracted == N_tilde[None, :]) & (~invalid)
                    Lsite = invalid * log_invalid + nearmatch * near_bonus
                    E[:, :, k] = Lsite @ Bmat
                partner_site = np.empty((C.size, L), dtype=np.int64)
                for c in range(C.size):
                    path = _viterbi_partner(E[c], switch_pen)
                    partner_site[c] = path[bof]
                present_by_partner = np.take_along_axis(cur, partner_site, axis=0)
                Fdiff = d_rC - present_by_partner
                Fest = np.clip(Fdiff, 0, 1).astype(np.int16)
                contribute = (Fdiff == 0) | (Fdiff == 1)
                mask = contribute.astype(np.float64)
                denom = mask.sum(0); num = (Fest * mask).sum(0)
                H = N_tilde.copy(); has = denom > 0
                frac1 = np.zeros(L); frac1[has] = num[has] / denom[has]
                H[has] = (frac1[has] >= 0.5).astype(np.int8)
                win_frac = np.where(H == 1, frac1, 1.0 - frac1)
                dists_e = np.array([_hamming_pct(cur[k], H) for k in range(m)])
                estar = int(np.argmin(dists_e)); pmin = float(dists_e[estar])
                deciding = (H != cur[estar])
                n_supported = int((deciding & has & (win_frac >= support_thresh)).sum())

                # ---- REFINEMENT ONLY: only consider a replace (H close to an existing hap) ----
                if pmin >= eps:
                    continue
                if not np.any(deciding):
                    continue

                # score dLL: self-block LL with H at estar vs current haps
                cand = cur.copy()
                cand[estar] = H
                base_ll = _score_self(pos, cur, spb, penalty, num_threads=dyn_fn)
                cand_ll = _score_self(pos, cand, spb, penalty, num_threads=dyn_fn)
                dLL = cand_ll - base_ll

                if dLL > min_dll:
                    cur[estar] = H
                    actions.append(dict(rep=int(rep), estar=int(estar), dLL=float(dLL),
                                        pmin=float(pmin), n_supported=int(n_supported),
                                        block_iter=int(_bi_iter)))
                    changed_this_pass = True

            if not changed_this_pass:
                converged = True
                break

        modified = len(actions) > 0
        refined = cur.astype(np.int8).tolist() if modified else None
        status = 'ok' if converged else 'NOT_CONVERGED_cap_hit'
        return dict(j=j, modified=modified, refined_seqs=refined, actions=actions,
                    status=status, passes=int(_bi_iter), converged=bool(converged))
    except Exception as e:
        return dict(j=task[0] if isinstance(task, tuple) else -1, modified=False,
                    refined_seqs=None, actions=[],
                    status=f'error: {type(e).__name__}: {e}')
    finally:
        # Unregister this worker (release any held extra thread first, so peers see the
        # freed slot before the decremented active count), mirroring
        # hierarchical_assembly._process_single_batch.  Runs on EVERY exit path,
        # including the early paint-failed return inside the loop.
        try:
            _ha._try_release_extra()
        except Exception:
            pass
        if _ha._ACTIVE_COUNTER is not None:
            try:
                with _ha._ACTIVE_COUNTER.get_lock():
                    _ha._ACTIVE_COUNTER.value -= 1
            except Exception:
                pass


def _init_worker(meta, global_sites, opt, total_cores, active_counter, extra_counter):
    """Pool initializer: attach shared probs; build site_to_idx from global_sites
    (not pickled); stash accept options; AND wire hierarchical_assembly's dynamic-
    thread state (active/extra counters + total_cores + numba ceiling) exactly as its
    _init_worker_meta does, so _get_dynamic_threads works in this worker and the inner
    kernels can scale up their thread count as peers finish.  No L0 blocks anywhere."""
    import hierarchical_assembly as _ha
    # --- dynamic-thread state (mirror _ha._init_worker_meta) ---
    os.environ['NUMBA_NUM_THREADS'] = str(total_cores)
    try:
        import numba
        numba.config.NUMBA_NUM_THREADS = total_cores  # ceiling so set_num_threads scales up
        numba.set_num_threads(1)                       # start conservative
    except Exception:
        pass
    _ha._ACTIVE_COUNTER = active_counter
    _ha._TOTAL_CORES = total_cores
    _ha._EXTRA_COUNTER = extra_counter
    _ha._I_HAVE_EXTRA = False
    # --- our shared data ---
    shm, gp = _ha._attach_shared_array(meta['probs'])
    _WK.clear()
    _WK['_shm'] = shm
    _WK['global_probs'] = gp
    _WK['global_sites'] = np.asarray(global_sites)
    _WK['site_to_idx'] = {int(s): i for i, s in enumerate(np.asarray(global_sites))}
    _OPT.clear()
    _OPT.update(opt)


def refine_level(blocks, global_probs, global_sites,
                 n_workers=112, eps_present=2.0, paint_penalty=10.0,
                 clean_span_frac=0.6, min_carriers=2, switch_pen=4.0,
                 log_invalid=-7.0, near_bonus=0.3, support_thresh=0.75,
                 max_block_iter=None, min_dll=1e-6,
                 rep_selector=None, return_actions=False, pool=None, verbose=True):
    """Refine the haplotypes of `blocks` (any level) using SELF-BLOCK scoring against
    global_probs, and return a refined copy.  REFINEMENT ONLY (replaces haps where
    dLL>min_dll; never adds).  No access to lower-level blocks.

    blocks       : list of BlockResult-like (the level-n super-blocks; each has
                   .positions and .haplotypes).  Their haps are scored against the
                   genotype probs over their OWN positions.
    global_probs : [N, n_sites, 3] genotype probs (this contig)
    global_sites : [n_sites] sorted positions matching global_probs
    rep_selector : OPTIONAL callable(block_index, cur_seqs, positions)->list[int] of
                   hap indices to use as carrier reps.  None => truth-free (every hap).
    max_block_iter : per-block fixed-point cap.  None => 3 x (#haps).  Cap-hit means
                   the block did NOT converge and is flagged loudly.
    min_dll      : positive-margin acceptance threshold (default 1e-6).  Guarantees
                   the fixed-point loop terminates (strictly-increasing bounded LL).
    pool         : OPTIONAL pre-existing forkserver pool (with _init_worker installed).
                   If None, this function creates and tears down its own pool, which
                   requires the forkserver to have been started by the caller.

    Returns refined_blocks, or (refined_blocks, actions_by_block) if return_actions.
    """
    import hierarchical_assembly

    opt = dict(eps_present=eps_present, paint_penalty=paint_penalty,
               clean_span_frac=clean_span_frac, min_carriers=min_carriers,
               switch_pen=switch_pen, log_invalid=log_invalid, near_bonus=near_bonus,
               support_thresh=support_thresh, max_block_iter=max_block_iter,
               min_dll=min_dll)

    block_pos = [np.asarray(sb.positions) for sb in blocks]
    block_seqs = [np.stack([_collapse(sb.haplotypes[k]) for k in sorted(sb.haplotypes.keys())], axis=0)
                  for sb in blocks]

    # build small per-task payloads (positions + hap matrix + optional reps) -- NO blocks
    tasks = []
    for j in range(len(blocks)):
        reps = None
        if rep_selector is not None:
            reps = list(rep_selector(j, block_seqs[j], block_pos[j]))
        tasks.append((j, block_pos[j], block_seqs[j].astype(np.int8), reps))

    shm_probs, probs_meta = hierarchical_assembly._create_shared_array(
        np.ascontiguousarray(global_probs), 'global_probs')
    shared_meta = {'probs': probs_meta}

    # Two-level parallelism (mirrors hierarchical_assembly): size the OUTER pool to the
    # work so we never spawn idle workers, and give each worker total_cores // workers
    # INNER numba threads.  When blocks are MANY (L1: 753) -> ~n_workers workers x 1
    # thread.  When blocks are FEW (L2: 13-76, L3: ~8, L4: 1-2) -> few workers each with
    # MANY threads, so the whole machine stays busy instead of (#blocks) cores doing all
    # the work while the rest idle.  The active/extra counters drive _get_dynamic_threads
    # so a surviving worker scales UP as peers finish (remainder reassignment).
    total_cores = int(n_workers)
    n_blocks = len(blocks)
    own_pool = pool is None
    if own_pool:
        outer_workers = max(1, min(total_cores, n_blocks))
        inner_threads = max(1, total_cores // outer_workers)
        active_counter = hierarchical_assembly._forkserver_ctx.Value('i', 0)
        extra_counter = hierarchical_assembly._forkserver_ctx.Value('i', 0)
        if verbose:
            print(f"      refine_level: {outer_workers} outer workers x ~{inner_threads} "
                  f"inner threads = {outer_workers * inner_threads} cores "
                  f"({n_blocks} blocks)")
        pool = hierarchical_assembly.NoDaemonPool(
            processes=outer_workers, initializer=_init_worker,
            initargs=(shared_meta, np.asarray(global_sites), opt,
                      total_cores, active_counter, extra_counter))

    refined_seqs = {}
    actions_by_block = {}
    n_modified = 0
    n_changes = 0
    n_not_converged = 0
    n_err = 0
    max_passes = 0
    not_converged_blocks = []
    try:
        done = 0
        # chunksize=1 (matches production hierarchical_assembly's imap_unordered):
        # per-block work is UNEVEN (a multi-pass fixed-point block costs far more than
        # one that converges in a single pass), so dispatching one block at a time lets
        # a freed worker immediately steal the next task -- no tail idling.
        for r in pool.imap_unordered(_refine_one_block, tasks, chunksize=1):
            done += 1
            st = r.get('status', 'ok')
            if isinstance(st, str) and st.startswith('error'):
                n_err += 1
                if verbose and n_err <= 5:
                    print(f"      refine_level: worker error on block {r['j']}: {st}")
            if st == 'NOT_CONVERGED_cap_hit':
                n_not_converged += 1
                not_converged_blocks.append(r['j'])
            max_passes = max(max_passes, int(r.get('passes', 0)))
            if r['modified']:
                refined_seqs[r['j']] = np.asarray(r['refined_seqs'], dtype=np.int8)
                actions_by_block[r['j']] = r['actions']
                n_modified += 1
                n_changes += len(r['actions'])
            if verbose and done % 100 == 0:
                print(f"      refine_level: {done}/{len(blocks)} blocks processed")
    finally:
        if own_pool:
            pool.close(); pool.join()
        try:
            shm_probs.close(); shm_probs.unlink()
        except Exception:
            pass

    if verbose:
        print(f"      refine_level: {n_modified}/{len(blocks)} blocks modified, "
              f"{n_changes} hap replacements, max passes {max_passes}")
        if n_err:
            print(f"      refine_level: {n_err} worker error(s)")
        if n_not_converged:
            print(f"      refine_level: *** {n_not_converged} block(s) DID NOT CONVERGE "
                  f"(hit cap): {not_converged_blocks[:20]}"
                  f"{' ...' if len(not_converged_blocks) > 20 else ''}")

    # build refined block list: replace haps where modified, else keep original block
    refined_blocks = []
    for j, sb in enumerate(blocks):
        if j not in refined_seqs:
            refined_blocks.append(sb)
            continue
        seqs = refined_seqs[j]
        blk = copy.deepcopy(sb)
        old_keys = sorted(blk.haplotypes.keys())
        newhaps = {}
        for hi in range(seqs.shape[0]):
            key = old_keys[hi] if hi < len(old_keys) else (max(old_keys) + 1 + (hi - len(old_keys)))
            newhaps[key] = _seq_to_prob_block(seqs[hi])
        blk.haplotypes = newhaps
        refined_blocks.append(blk)

    if return_actions:
        return refined_blocks, actions_by_block
    return refined_blocks