#!/usr/bin/env python3
"""
recombination_map.py  --  Marey-style recombination map for the BHD pipeline.

Builds, per chromosome, a genetic map (cumulative genetic distance in cM versus
physical position in Mb) with two overlaid curves:

    * TRUTH    -- from the simulation's true ancestry (stage 02 `truth_painting`)
                  + the TRUE pedigree.
    * INFERRED -- from the pipeline's recovered painting (stage 10
                  `tolerance_result`) + the INFERRED pedigree.

This is the panel-B "Marey map" adapted as a pipeline recovery check: the
simulation uses a constant recombination rate (5e-8 / bp), so the true map is
~linear in physical distance, and the figure shows whether the pipeline recovers
that (linear) map.  The same construction carries to real data, where only the
inferred curve exists and the shape is no longer linear.

================================ METHOD ======================================
A recombination map is a per-meiosis crossover rate per physical interval,
cumulated into cM.  We obtain per-meiosis crossovers from parent->child trios
using a phase-aware trio HMM (this is the BHD pipeline's own pedigree-scoring
model, `run_trio_phase_aware_hmm`, here COPIED and extended with backtracking):

  Per eligible child + its two parents, we reconstruct diploid alleles for the
  child and both parents at every marker (from each one's painting + the shared
  founder set), then Viterbi-decode the 16-state trio HMM.  Its 8 inheritance
  states jointly encode, at every marker, which parental haplotype each child
  strand follows (and the phase); its 8 burst states absorb error/IBD runs.
  Reading the decoded path:
    * a change in which p1-haplotype the p1-derived strand follows = a
      p1-meiosis crossover,
    * a change in which p2-haplotype the p2-derived strand follows = a
      p2-meiosis crossover,
    * phase flips (free at homozygous markers) are NOT crossovers (they only
      relabel which child strand is "strand 0"),
    * burst-state runs are uninformative gaps (no crossover counted inside).
  One decode thus yields both parental gametes' crossovers; there is no separate
  assignment step.  Each crossover is localized between its flanking markers
  (midpoint used for binning).  Crossovers are aggregated per physical bin per
  chromosome, divided by the meiosis count -> Morgans -> cM; the Marey curve is
  the cumulative cM along the chromosome.

PER-SNP, not binned: we lift the per-SNP trio HMM (`run_trio_phase_aware_hmm`),
not the binned `_multisnp` variant the pipeline uses for pedigree scoring, so
crossovers localize between adjacent markers rather than adjacent bins -- finer,
which is what a map wants.  Both maps use it identically.

ELIGIBLE TRANSMISSIONS: only offspring (F1/F2/F3) are painted; founders are not.
So a gamete can only be followed against a parent that is itself a painted
sample: F1->F2 and F2->F3 (every F2 and F3 individual's two gametes, ~600
meioses).  Both maps use the transmissions their own pedigree makes eligible.

UNDERCOUNTING: with only ~3 founders, by F2->F3 a parent's two haplotypes are
often identical-by-descent in a region; a crossover between two IBD segments is
invisible to ANY method (allele or label), so the F2->F3 contribution is
slightly deflated.  This is identical for both maps, so the comparison holds.

PEDIGREE: truth curve uses the TRUE pedigree (stage 02 `truth_pedigree`);
inferred curve uses the INFERRED pedigree (results CSV
`pedigree_inference_discovered.csv`).  Crossover detection needs parent links,
and on real data only the inferred pedigree exists, so the inferred curve is
built end-to-end from the pipeline's own pedigree.

============================ PROVENANCE ======================================
This file is deliberately SELF-CONTAINED: the crossover-detection machinery is
COPIED here (not imported from the pipeline) so future changes to the map's
method live in one place and cannot perturb production pedigree inference.
Copied verbatim (with light adaptation noted at each):
  * concretify_haps, pairup_haps                      <- simulate_sequences.py
  * get_snp_level_founder_ids, build_founder_allele_lookup,
    run_trio_phase_aware_hmm (the scorer)             <- pedigree_inference.py
  * the switch/stay cost formula + compute_hom_mask   <- paint_samples.py
  * penalty constants (error/phase/mismatch, recomb_rate) <- pedigree_inference.py
Written fresh: run_trio_phase_aware_hmm_backtrack (backtracking sibling of the
scorer), decode_crossovers_from_path, the loader/driver/plot.

Only `paint_samples` is imported (for the SamplePainting/PaintedChunk/
BlockPainting classes needed to unpickle the checkpoints); none of the
detection machinery is imported.

=============================== USAGE ========================================
    python recombination_map.py                  # uses ./.pipeline_checkpoints
    python recombination_map.py --ckpt-dir PATH
    python recombination_map.py --bin-mb 1.0     # physical bin width (Mb)
    python recombination_map.py --out-dir my_map # output directory (see below)
    python recombination_map.py --true-pedigree-for-inferred  # ablation
    python recombination_map.py --selftest       # synthetic validation only

Outputs (under --out-dir, default ./recombination_map/):
    composite.png              faceted all-chromosome figure (the panel-B view)
    map_data.pkl               all cM curves (re-style without re-running the HMM)
    chromosomes/<contig>.png   per-chromosome Marey figure
    chromosomes/<contig>.csv   per-chromosome curve: position_mb, cum_cM_truth,
                               cum_cM_inferred
"""

import math

import numpy as np

# numba is optional: if unavailable, @njit degrades to a no-op so the pure-Python
# path still runs (slower).  Mirrors the guard in the pipeline modules.
try:
    from numba import njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator


# =============================================================================
# Penalty / rate constants  (COPIED from pedigree_inference.py, verbatim values)
# =============================================================================
RECOMB_RATE = 5e-8                 # per-bp recombination rate (matches the sim)
ERROR_PENALTY = -math.log(1e-2)    # = 4.605170 ; pedigree_inference error_pen
PHASE_PENALTY = 50.0               # pedigree_inference phase_pen
MISMATCH_PENALTY = -4.605170       # = math.log(0.01) ; DEFAULT_MISMATCH_PENALTY


# =============================================================================
# Founder preparation  (COPIED from simulate_sequences.py)
# Needed only for the TRUTH side: the simulation's founder haplotypes are rebuilt
# the same way pipeline.py builds `founders_list`, so the founder-hap IDs in
# `truth_painting` (founder i -> ids 2i, 2i+1) map back to allele arrays.
# =============================================================================
def concretify_haps(haps_list):
    """
    Takes a list of probabalistic haps and turns each of them
    into a list of 0s and 1s by taking the highest probability
    allele at each site
    """
    concreted = []
    for hap in haps_list:
        concreted.append(np.argmax(hap, axis=1))
    return concreted


def pairup_haps(haps_list, shuffle=False):
    """
    Pair up a list of concrete haps (made up of 0s and 1s)
    """
    # NOTE (copy): original used pickle round-trip for a deep copy; we keep a
    # deep copy but via list/np copies to avoid importing pickle for this.
    haps_copy = [np.array(h, copy=True) for h in haps_list]

    if shuffle:
        import random
        random.shuffle(haps_copy)

    num_pairs = len(haps_list) // 2
    haps_paired = []

    for i in range(num_pairs):
        first = haps_copy[2 * i]
        second = haps_copy[2 * i + 1]
        haps_paired.append([first, second])

    return haps_paired


# =============================================================================
# Allele reconstruction from a painting + founder set  (COPIED from
# pedigree_inference.py: get_snp_level_founder_ids, build_founder_allele_lookup)
# =============================================================================
def get_snp_level_founder_ids(painting_chunks, snp_positions):
    n_snps = len(snp_positions)
    id_array = np.full((n_snps, 2), -1, dtype=np.int32)
    if not painting_chunks:
        return id_array
    c_ends = np.array([c.end for c in painting_chunks])
    c_h1 = np.array([c.hap1 for c in painting_chunks])
    c_h2 = np.array([c.hap2 for c in painting_chunks])
    c_starts = np.array([c.start for c in painting_chunks])
    indices = np.searchsorted(c_ends, snp_positions)
    indices = np.clip(indices, 0, len(painting_chunks) - 1)
    valid_mask = snp_positions >= c_starts[indices]
    id_array[:, 0] = np.where(valid_mask, c_h1[indices], -1)
    id_array[:, 1] = np.where(valid_mask, c_h2[indices], -1)
    return id_array


def build_founder_allele_lookup(positions, haplotypes):
    """COPIED from pedigree_inference.build_founder_allele_lookup, adapted to take
    (positions, haplotypes-dict) directly instead of a FounderBlock object so we
    don't depend on the pipeline's block class.

    haplotypes: {int founder_id: hap_array}, each 1-D (alleles) or 2-D (prob,
    argmax'd here).  founder_ids are used as row indices (may be non-contiguous).
    """
    snp_positions = positions
    n_snps = len(snp_positions)
    hap_keys = sorted(list(haplotypes.keys()))
    max_id = max(hap_keys) if hap_keys else 0
    allele_lookup = np.full((max_id + 1, n_snps), -1, dtype=np.int8)
    for fid, h_arr in haplotypes.items():
        h_arr = np.asarray(h_arr)
        if h_arr.ndim == 2:
            raw_alleles = np.argmax(h_arr, axis=1)
        else:
            raw_alleles = h_arr
        allele_lookup[fid, :] = raw_alleles.astype(np.int8)
    return allele_lookup, snp_positions


def diploid_alleles_for_sample(painting_chunks, allele_lookup, snp_positions):
    """Reconstruct a sample's (n_snps, 2) diploid alleles from its painting.

    founder id per strand per SNP (get_snp_level_founder_ids) -> allele via
    allele_lookup; missing/uncovered founder id (-1) stays -1.
    """
    ids = get_snp_level_founder_ids(painting_chunks, snp_positions)  # (n_snps, 2)
    n_snps = len(snp_positions)
    out = np.full((n_snps, 2), -1, dtype=np.int8)
    for strand in range(2):
        fid = ids[:, strand]
        valid = fid >= 0
        # allele_lookup row -1 would wrap; guard with valid mask.
        rows = np.where(valid, fid, 0)
        looked = allele_lookup[rows, np.arange(n_snps)]
        out[:, strand] = np.where(valid, looked, -1)
    return out


def build_switch_stay_costs(snp_positions, recomb_rate=RECOMB_RATE):
    """COPIED formula from paint_samples.process_contig_for_pedigree, applied
    per-SNP (the pipeline applies it per-bin):

        dists[0]=0, dists[i]=pos[i]-pos[i-1]
        theta = clip(1 - exp(-dist*recomb_rate), 1e-15, 0.5)
        sw_costs = log(theta) ; st_costs = log(1-theta)
    """
    n = len(snp_positions)
    dists = np.zeros(n)
    if n > 1:
        dists[1:] = np.diff(np.asarray(snp_positions, dtype=np.float64))
    theta = np.clip(1.0 - np.exp(-dists * recomb_rate), 1e-15, 0.5)
    sw_costs = np.log(theta)
    st_costs = np.log(1.0 - theta)
    return sw_costs, st_costs


def compute_hom_mask(child_dip_alleles):
    """Per-SNP form of paint_samples.compute_ibs_hom_mask: a marker is
    phase-ambiguous (free phase flip) when the child's two strands carry the
    same allele (true hom OR IBS).  Missing (-1) on either strand -> treat as
    ambiguous (conservative: allows free flip).
    """
    c0 = child_dip_alleles[:, 0]
    c1 = child_dip_alleles[:, 1]
    return (c0 == c1) | (c0 < 0) | (c1 < 0)


# =============================================================================
# Trio HMM -- SCORER  (COPIED VERBATIM from pedigree_inference.run_trio_phase_aware_hmm)
# Kept here unchanged so the backtracking variant below can be checked against it
# (their best-path scores must agree).  16 states = 8 inheritance + 8 burst.
# =============================================================================
@njit(fastmath=True, cache=True)
def run_trio_phase_aware_hmm(child_dip_alleles, child_potential_hom_mask,
                             p1_dip_alleles, p2_dip_alleles,
                             switch_costs, stay_costs, error_penalty, phase_penalty,
                             mismatch_penalty=-4.6):
    n_sites = len(child_dip_alleles)
    BURST_EMISSION = -1.386
    scores = np.zeros(16)
    for k in range(8, 16):
        scores[k] = -error_penalty
    for i in range(n_sites):
        c0, c1 = child_dip_alleles[i, 0], child_dip_alleles[i, 1]
        p1_h0, p1_h1 = p1_dip_alleles[i, 0], p1_dip_alleles[i, 1]
        p2_h0, p2_h1 = p2_dip_alleles[i, 0], p2_dip_alleles[i, 1]

        def soft_match(parent_allele, child_allele):
            if parent_allele == -1 or child_allele == -1:
                return 0.0
            elif parent_allele == child_allele:
                return 0.0
            else:
                return mismatch_penalty
        m_p1h0_c0 = soft_match(p1_h0, c0); m_p1h1_c0 = soft_match(p1_h1, c0)
        m_p1h0_c1 = soft_match(p1_h0, c1); m_p1h1_c1 = soft_match(p1_h1, c1)
        m_p2h0_c0 = soft_match(p2_h0, c0); m_p2h1_c0 = soft_match(p2_h1, c0)
        m_p2h0_c1 = soft_match(p2_h0, c1); m_p2h1_c1 = soft_match(p2_h1, c1)
        e = np.zeros(8)
        e[0] = m_p1h0_c0 + m_p2h0_c1
        e[1] = m_p1h0_c0 + m_p2h1_c1
        e[2] = m_p1h1_c0 + m_p2h0_c1
        e[3] = m_p1h1_c0 + m_p2h1_c1
        e[4] = m_p1h0_c1 + m_p2h0_c0
        e[5] = m_p1h0_c1 + m_p2h1_c0
        e[6] = m_p1h1_c1 + m_p2h0_c0
        e[7] = m_p1h1_c1 + m_p2h1_c0
        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        c_phase = 0.0 if child_potential_hom_mask[i] else -phase_penalty
        prev = scores.copy()
        new_scores = np.zeros(16)
        for k in range(8):
            burst_idx = k + 8
            from_burst = prev[burst_idx]
            from_normal = prev[k] - error_penalty
            new_scores[burst_idx] = max(from_burst, from_normal) + BURST_EMISSION
        cc_0 = 2 * c_stay
        cc_1 = c_recomb + c_stay
        cc_2 = 2 * c_recomb
        p0, p1, p2, p3 = prev[0], prev[1], prev[2], prev[3]
        a0 = max(p0+cc_0, p1+cc_1, p2+cc_1, p3+cc_2)
        a1 = max(p0+cc_1, p1+cc_0, p2+cc_2, p3+cc_1)
        a2 = max(p0+cc_1, p1+cc_2, p2+cc_0, p3+cc_1)
        a3 = max(p0+cc_2, p1+cc_1, p2+cc_1, p3+cc_0)
        p4, p5, p6, p7 = prev[4], prev[5], prev[6], prev[7]
        b4 = max(p4+cc_0, p5+cc_1, p6+cc_1, p7+cc_2)
        b5 = max(p4+cc_1, p5+cc_0, p6+cc_2, p7+cc_1)
        b6 = max(p4+cc_1, p5+cc_2, p6+cc_0, p7+cc_1)
        b7 = max(p4+cc_2, p5+cc_1, p6+cc_1, p7+cc_0)
        pb = prev[8:16]
        new_scores[0] = max(a0 + c_stay, b4 + c_stay + c_phase, pb[0]) + e[0]
        new_scores[1] = max(a1 + c_stay, b5 + c_stay + c_phase, pb[1]) + e[1]
        new_scores[2] = max(a2 + c_stay, b6 + c_stay + c_phase, pb[2]) + e[2]
        new_scores[3] = max(a3 + c_stay, b7 + c_stay + c_phase, pb[3]) + e[3]
        new_scores[4] = max(b4 + c_stay, a0 + c_stay + c_phase, pb[4]) + e[4]
        new_scores[5] = max(b5 + c_stay, a1 + c_stay + c_phase, pb[5]) + e[5]
        new_scores[6] = max(b6 + c_stay, a2 + c_stay + c_phase, pb[6]) + e[6]
        new_scores[7] = max(b7 + c_stay, a3 + c_stay + c_phase, pb[7]) + e[7]
        scores = new_scores
    best_final = -np.inf
    for k in range(16):
        if scores[k] > best_final:
            best_final = scores[k]
    return best_final


# =============================================================================
# Trio HMM -- BACKTRACKING variant  (NEW)
# Reproduces run_trio_phase_aware_hmm's emissions and transitions EXACTLY, but
# stores a backpointer per (site, state) and tracebacks the Viterbi path.  The
# transition is written in explicit "aggregate + argmax" form so each max in the
# scorer has a recorded winner; the resulting best-path score must equal the
# scorer's best_final (asserted in selftest).
#
# State layout (matches the scorer's emission indexing e[0..7]):
#   inheritance state s in 0..7 : phase = s//4, config = s%4,
#                                 p1hap = config//2 (which p1-hap the p1-derived
#                                 strand follows), p2hap = config%2 (which p2-hap
#                                 the p2-derived strand follows).
#   burst state s+8 for s in 0..7.
# =============================================================================
@njit(fastmath=True, cache=True)
def run_trio_phase_aware_hmm_backtrack(child_dip_alleles, child_potential_hom_mask,
                                       p1_dip_alleles, p2_dip_alleles,
                                       switch_costs, stay_costs, error_penalty,
                                       phase_penalty, mismatch_penalty=-4.6):
    n_sites = len(child_dip_alleles)
    BURST_EMISSION = -1.386

    scores = np.zeros(16)
    for k in range(8, 16):
        scores[k] = -error_penalty

    bp = np.full((n_sites, 16), -1, dtype=np.int8)  # backpointer: prev state

    for i in range(n_sites):
        c0, c1 = child_dip_alleles[i, 0], child_dip_alleles[i, 1]
        p1_h0, p1_h1 = p1_dip_alleles[i, 0], p1_dip_alleles[i, 1]
        p2_h0, p2_h1 = p2_dip_alleles[i, 0], p2_dip_alleles[i, 1]

        def soft_match(parent_allele, child_allele):
            if parent_allele == -1 or child_allele == -1:
                return 0.0
            elif parent_allele == child_allele:
                return 0.0
            else:
                return mismatch_penalty
        m_p1h0_c0 = soft_match(p1_h0, c0); m_p1h1_c0 = soft_match(p1_h1, c0)
        m_p1h0_c1 = soft_match(p1_h0, c1); m_p1h1_c1 = soft_match(p1_h1, c1)
        m_p2h0_c0 = soft_match(p2_h0, c0); m_p2h1_c0 = soft_match(p2_h1, c0)
        m_p2h0_c1 = soft_match(p2_h0, c1); m_p2h1_c1 = soft_match(p2_h1, c1)
        e = np.zeros(8)
        e[0] = m_p1h0_c0 + m_p2h0_c1
        e[1] = m_p1h0_c0 + m_p2h1_c1
        e[2] = m_p1h1_c0 + m_p2h0_c1
        e[3] = m_p1h1_c0 + m_p2h1_c1
        e[4] = m_p1h0_c1 + m_p2h0_c0
        e[5] = m_p1h0_c1 + m_p2h1_c0
        e[6] = m_p1h1_c1 + m_p2h0_c0
        e[7] = m_p1h1_c1 + m_p2h1_c0

        c_recomb = switch_costs[i]
        c_stay = stay_costs[i]
        c_phase = 0.0 if child_potential_hom_mask[i] else -phase_penalty
        prev = scores.copy()

        # Transition cost between two configs (p1hap,p2hap): per parent, c_stay if
        # that parent's hap is unchanged else c_recomb.  config bit1 = p1hap,
        # bit0 = p2hap.  (Matches cc_0/cc_1/cc_2 in the scorer.)
        # a_cfg[c] = best arrival at config c from phase-0 prevs (states 0..3),
        # b_cfg[c] = ... from phase-1 prevs (states 4..7); with argmax sources.
        a_val = np.empty(4); a_src = np.empty(4, dtype=np.int8)
        b_val = np.empty(4); b_src = np.empty(4, dtype=np.int8)
        for c in range(4):
            best_a = -np.inf; arg_a = 0
            best_b = -np.inf; arg_b = 0
            for pc in range(4):
                d_p1 = (pc >> 1) != (c >> 1)
                d_p2 = (pc & 1) != (c & 1)
                tcost = (c_recomb if d_p1 else c_stay) + (c_recomb if d_p2 else c_stay)
                va = prev[pc] + tcost
                if va > best_a:
                    best_a = va; arg_a = pc
                vb = prev[4 + pc] + tcost
                if vb > best_b:
                    best_b = vb; arg_b = 4 + pc
            a_val[c] = best_a; a_src[c] = arg_a
            b_val[c] = best_b; b_src[c] = arg_b

        new_scores = np.zeros(16)

        # Burst states first (depend only on prev).
        for k in range(8):
            burst_idx = k + 8
            from_burst = prev[burst_idx]
            from_normal = prev[k] - error_penalty
            if from_burst >= from_normal:
                new_scores[burst_idx] = from_burst + BURST_EMISSION
                bp[i, burst_idx] = burst_idx
            else:
                new_scores[burst_idx] = from_normal + BURST_EMISSION
                bp[i, burst_idx] = k

        # Inheritance states: phase 0 -> s in 0..3, phase 1 -> s in 4..7.
        for s in range(8):
            phase = s // 4
            c = s % 4
            if phase == 0:
                stay_v = a_val[c] + c_stay;            stay_src = a_src[c]
                flip_v = b_val[c] + c_stay + c_phase;  flip_src = b_src[c]
            else:
                stay_v = b_val[c] + c_stay;            stay_src = b_src[c]
                flip_v = a_val[c] + c_stay + c_phase;  flip_src = a_src[c]
            burst_v = prev[8 + s]                       # pb[s]
            # argmax over {stay, flip, burst-self}
            best_v = stay_v; best_src = stay_src
            if flip_v > best_v:
                best_v = flip_v; best_src = flip_src
            if burst_v > best_v:
                best_v = burst_v; best_src = np.int8(8 + s)
            new_scores[s] = best_v + e[s]
            bp[i, s] = best_src

        scores = new_scores

    # Best final state + traceback.
    best_final = -np.inf
    best_state = 0
    for k in range(16):
        if scores[k] > best_final:
            best_final = scores[k]
            best_state = k

    path = np.empty(n_sites, dtype=np.int8)
    s = best_state
    for i in range(n_sites - 1, -1, -1):
        path[i] = s
        s = bp[i, s]
        if s < 0:
            s = 0  # reached the virtual initial state
    return best_final, path


# =============================================================================
# Crossover decode  (NEW)
# Reads p1- and p2-meiosis crossovers off the decoded path.  Tracks the last
# DETERMINED (p1hap, p2hap) across phase flips and burst gaps; a change in p1hap
# is a p1 crossover, a change in p2hap a p2 crossover, each localized to the
# interval between the flanking determined markers.
# =============================================================================
def decode_crossovers_from_path(path, snp_positions):
    """Return (p1_spans, p2_spans): lists of (left_pos, right_pos) bounding each
    crossover.  A burst state (>=8) is undetermined and carries the last state
    through (no crossover counted inside it)."""
    p1_spans = []
    p2_spans = []
    last_p1 = -1
    last_p2 = -1
    last_pos = None
    for i in range(len(path)):
        s = int(path[i])
        if s >= 8:
            continue  # burst / undetermined -> carry last determined through
        c = s % 4
        p1hap = c >> 1
        p2hap = c & 1
        if last_pos is not None:
            if p1hap != last_p1:
                p1_spans.append((last_pos, snp_positions[i]))
            if p2hap != last_p2:
                p2_spans.append((last_pos, snp_positions[i]))
        last_p1 = p1hap
        last_p2 = p2hap
        last_pos = snp_positions[i]
    return p1_spans, p2_spans


# =============================================================================
# Checkpoint loading + painting normalization
# Layout (pipeline.py): <ckpt>/<stage>/<contig>.pkl and <ckpt>/<stage>/_global.pkl,
# each a pickle.  Unpickling the painting objects needs `paint_samples` importable
# (it defines SamplePainting/PaintedChunk/BlockPainting); run this from the
# haplotype_reconstruction dir with bio-env active.  `paint_samples` is imported
# ONLY for those data classes -- no detection machinery is imported.
# =============================================================================
import os
import pickle

STAGE_VCF = "01_vcf_discovery"       # naive_long_haps (true founders)
STAGE_SIM = "02_simulation"          # truth_painting + truth_pedigree
STAGE_L4 = "09_assembly_L4"          # super_blocks_L4 (discovered founders)
STAGE_PAINT = "10_viterbi_painting"  # tolerance_result (inferred painting)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_global(ckpt_dir, stage):
    return _load_pickle(os.path.join(ckpt_dir, stage, "_global.pkl"))


def load_contig(ckpt_dir, stage, contig):
    return _load_pickle(os.path.join(ckpt_dir, stage, f"{contig}.pkl"))


def _samples_of(painting_obj):
    """Normalize a painting checkpoint to a list of per-sample painting objects.

    Stage 02 `truth_painting` is a list of SamplePainting; stage 10
    `tolerance_result` is a BlockPainting whose `.samples` is that list.
    """
    if isinstance(painting_obj, (list, tuple)):
        return list(painting_obj)
    if hasattr(painting_obj, "samples"):
        return list(painting_obj.samples)
    raise TypeError(f"Unrecognized painting container: {type(painting_obj)!r}")


def _index_by_name(samples, sample_names):
    """Map painted samples to pedigree sample names.

    Painted samples are offspring-only, in `sample_names` order, each carrying a
    0-based `sample_index`.  Map by that index (fall back to enumerate order).
    """
    by_name = {}
    for k, sp in enumerate(samples):
        idx = getattr(sp, "sample_index", k)
        if not (isinstance(idx, (int, np.integer)) and 0 <= idx < len(sample_names)):
            idx = k
        if idx < len(sample_names):
            by_name[sample_names[idx]] = sp
    return by_name


# =============================================================================
# Founder allele lookups for each side
# =============================================================================
def build_truth_founder_lookup(naive_long_haps):
    """True founders: rebuilt exactly as pipeline.py builds `founders_list`.

    `naive_long_haps` = (sites, haps_data).  simulate_pedigree consumes
    pairup_haps(concretify_haps(haps_data)) and assigns founder-hap IDs
    sequentially (founder i -> ids 2i, 2i+1); with shuffle=False the pairing
    preserves order, so founder-hap id j corresponds to concretify_haps(...)[j].
    """
    sites, haps_data = naive_long_haps
    concrete = concretify_haps(haps_data)        # list of (n_snps,) arrays
    parents = pairup_haps(concrete)              # mirror the pipeline's exact call
    haplotypes = {}
    fid = 0
    for pair in parents:
        haplotypes[fid] = pair[0]; fid += 1
        haplotypes[fid] = pair[1]; fid += 1
    return build_founder_allele_lookup(np.asarray(sites), haplotypes)


def build_inferred_founder_lookup(super_blocks_L4):
    """Discovered founders: the L4 founder block the painting was painted against
    (`super_blocks_L4[0]`), whose `.haplotypes` keys are the founder ids used as
    hap1/hap2 in `tolerance_result`."""
    fb = super_blocks_L4[0]
    return build_founder_allele_lookup(np.asarray(fb.positions), dict(fb.haplotypes))


# =============================================================================
# Per-painting crossover collection
# =============================================================================
def collect_crossovers(samples_by_name, links, allele_lookup, snp_positions,
                       recomb_rate=RECOMB_RATE, verbose=False):
    """Collect per-meiosis crossover midpoints for one painting on one contig.

    For every child whose BOTH parents (per `links`) are painted samples,
    reconstruct child/parent diploid alleles, Viterbi-decode the trio HMM, and
    read off p1- and p2-meiosis crossovers.  Children with a founder parent (not
    painted) are skipped -- this is what restricts the map to F1->F2 and F2->F3.

    Returns (midpoints, n_meioses, n_children).
    """
    sw_costs, st_costs = build_switch_stay_costs(snp_positions, recomb_rate)
    snp_positions = np.asarray(snp_positions, dtype=np.float64)

    midpoints = []
    n_meioses = 0
    n_children = 0

    for child, (p1, p2) in links.items():
        cs = samples_by_name.get(child)
        ps1 = samples_by_name.get(p1)
        ps2 = samples_by_name.get(p2)
        if cs is None or ps1 is None or ps2 is None:
            continue

        child_dip = diploid_alleles_for_sample(cs.chunks, allele_lookup, snp_positions)
        p1_dip = diploid_alleles_for_sample(ps1.chunks, allele_lookup, snp_positions)
        p2_dip = diploid_alleles_for_sample(ps2.chunks, allele_lookup, snp_positions)
        hom = compute_hom_mask(child_dip)

        _score, path = run_trio_phase_aware_hmm_backtrack(
            child_dip, hom, p1_dip, p2_dip, sw_costs, st_costs,
            ERROR_PENALTY, PHASE_PENALTY, MISMATCH_PENALTY)
        p1_spans, p2_spans = decode_crossovers_from_path(path, snp_positions)

        for (l, r) in p1_spans:
            midpoints.append((l + r) / 2.0)
        for (l, r) in p2_spans:
            midpoints.append((l + r) / 2.0)
        n_meioses += 2     # two gametes (one per parent) per child
        n_children += 1

    if verbose:
        print(f"    {n_children} children, {n_meioses} meioses, "
              f"{len(midpoints)} crossovers")
    return midpoints, n_meioses, n_children


def cumulative_cM(midpoints, n_meioses, lo, hi, bin_bp):
    """Bin crossover midpoints -> cumulative cM along [lo, hi].

    cM per bin = 100 * (#crossovers in bin) / n_meioses  (Morgans -> cM);
    cumulative along the chromosome.  Returns (edges_bp, cum_cM) for a step plot.
    """
    if n_meioses == 0 or hi <= lo:
        return np.array([lo, hi], dtype=float), np.array([0.0, 0.0])
    n_bins = max(1, int(np.ceil((hi - lo) / float(bin_bp))))
    edges = lo + np.arange(n_bins + 1) * float(bin_bp)
    if edges[-1] < hi:
        edges[-1] = hi
    counts, _ = np.histogram(midpoints, bins=edges)
    cm_per_bin = 100.0 * counts / float(n_meioses)
    cum = np.concatenate([[0.0], np.cumsum(cm_per_bin)])
    return edges, cum


# =============================================================================
# Inferred pedigree (results CSV)
# =============================================================================
def read_inferred_pedigree(ckpt_dir, csv_path=None, results_dirname="results_simulation"):
    """Load the pipeline's discovered pedigree (Sample, Parent1, Parent2).

    Defaults to searching for `pedigree_inference_discovered.csv` near the
    checkpoint tree; override with --inferred-pedigree-csv.
    """
    import pandas as pd
    if csv_path is not None:
        candidates = [csv_path]
    else:
        base = os.path.dirname(os.path.abspath(ckpt_dir.rstrip("/"))) or "."
        candidates = [
            os.path.join(base, results_dirname, "pedigree_inference_discovered.csv"),
            os.path.join(results_dirname, "pedigree_inference_discovered.csv"),
            os.path.join(base, "pedigree_inference_discovered.csv"),
            "pedigree_inference_discovered.csv",
        ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            need = {"Sample", "Parent1", "Parent2"}
            if not need.issubset(df.columns):
                raise ValueError(
                    f"{p} is missing columns {need - set(df.columns)}; "
                    f"found {list(df.columns)}")
            return df
    raise FileNotFoundError(
        "Could not find the inferred pedigree CSV (pedigree_inference_discovered.csv). "
        f"Looked in: {candidates}. Pass --inferred-pedigree-csv PATH.")


# =============================================================================
# Map driver
# =============================================================================
def build_maps(ckpt_dir, bin_bp, use_inferred_pedigree=True, inferred_csv=None,
               contigs=None, verbose=True):
    """Build truth + inferred Marey maps for every contig.

    Returns {contig: {'truth': (edges, cum), 'inferred': (edges, cum),
                      'n_meioses_truth', 'n_meioses_inferred', 'lo', 'hi'}}.
    """
    g = load_global(ckpt_dir, STAGE_SIM)
    truth_pedigree = g["truth_pedigree"]
    region_keys = g["region_keys"]
    sample_names = list(truth_pedigree["Sample"])

    true_links = {r.Sample: (r.Parent1, r.Parent2)
                  for r in truth_pedigree.itertuples(index=False)}
    if use_inferred_pedigree:
        inf_ped = read_inferred_pedigree(ckpt_dir, inferred_csv)
        inferred_links = {r.Sample: (r.Parent1, r.Parent2)
                          for r in inf_ped.itertuples(index=False)}
    else:
        inferred_links = true_links

    if contigs is None:
        contigs = region_keys

    out = {}
    for contig in contigs:
        # Founder allele lookups (truth: rebuilt sim founders; inferred: L4).
        naive = load_contig(ckpt_dir, STAGE_VCF, contig)["naive_long_haps"]
        truth_lookup, truth_pos = build_truth_founder_lookup(naive)
        l4 = load_contig(ckpt_dir, STAGE_L4, contig)["super_blocks_L4"]
        inf_lookup, inf_pos = build_inferred_founder_lookup(l4)

        # Paintings.
        truth_samples = _samples_of(load_contig(ckpt_dir, STAGE_SIM, contig)["truth_painting"])
        truth_by_name = _index_by_name(truth_samples, sample_names)
        inf_obj = load_contig(ckpt_dir, STAGE_PAINT, contig)["tolerance_result"]
        inf_by_name = _index_by_name(_samples_of(inf_obj), sample_names)

        # Crossovers (truth uses true links + sim founders; inferred uses inferred
        # links + L4 founders).
        t_mid, t_n, _ = collect_crossovers(truth_by_name, true_links, truth_lookup, truth_pos)
        i_mid, i_n, _ = collect_crossovers(inf_by_name, inferred_links, inf_lookup, inf_pos)

        lo = float(min(truth_pos[0], inf_pos[0]))
        hi = float(max(truth_pos[-1], inf_pos[-1]))
        t_edges, t_cum = cumulative_cM(t_mid, t_n, lo, hi, bin_bp)
        i_edges, i_cum = cumulative_cM(i_mid, i_n, lo, hi, bin_bp)

        out[contig] = {
            "truth": (t_edges, t_cum), "inferred": (i_edges, i_cum),
            "n_meioses_truth": t_n, "n_meioses_inferred": i_n,
            "lo": lo, "hi": hi,
        }
        if verbose:
            print(f"  {contig}: truth {t_cum[-1]:6.1f} cM ({t_n} meioses) / "
                  f"inferred {i_cum[-1]:6.1f} cM ({i_n} meioses)")
    return out


# =============================================================================
# Plotting (basic faceted Marey map; detailed styling is a later pass)
# =============================================================================
def plot_maps(maps, out_path, title="Recombination map (cumulative genetic distance)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contigs = [c for c in maps.keys()]
    n = len(contigs)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for j, contig in enumerate(contigs):
        ax = axes[j // ncols][j % ncols]
        ax.set_visible(True)
        t_edges, t_cum = maps[contig]["truth"]
        i_edges, i_cum = maps[contig]["inferred"]
        ax.step(np.asarray(t_edges) / 1e6, t_cum, where="post",
                color="#1f4e8c", lw=1.4, label="Truth")
        ax.step(np.asarray(i_edges) / 1e6, i_cum, where="post",
                color="#c0392b", lw=1.4, label="Inferred")
        ax.set_title(str(contig), fontsize=9)
        ax.tick_params(labelsize=7)

    fig.supxlabel("Physical position (Mb)")
    fig.supylabel("Genetic distance (cM)")
    fig.suptitle(title, fontsize=12)
    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote figure: {out_path}")


def plot_one_chromosome(contig, data, path):
    """Single-chromosome Marey map (truth + inferred), larger than a facet."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_edges, t_cum = data["truth"]
    i_edges, i_cum = data["inferred"]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.step(np.asarray(t_edges) / 1e6, t_cum, where="post", color="#1f4e8c",
            lw=1.6, label=f"Truth ({data['n_meioses_truth']} meioses)")
    ax.step(np.asarray(i_edges) / 1e6, i_cum, where="post", color="#c0392b",
            lw=1.6, label=f"Inferred ({data['n_meioses_inferred']} meioses)")
    ax.set_xlabel("Physical position (Mb)")
    ax.set_ylabel("Genetic distance (cM)")
    ax.set_title(f"Recombination map - {contig}")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_chromosome_csv(contig, data, path):
    """Per-chromosome cM curve as CSV: position_mb, cum_cM_truth, cum_cM_inferred.

    Truth and inferred share the same bin edges (same lo/hi/bin_bp), so one
    position column suffices; aligned on the shorter length defensively.
    """
    t_edges, t_cum = data["truth"]
    i_edges, i_cum = data["inferred"]
    n = min(len(t_edges), len(i_edges), len(t_cum), len(i_cum))
    arr = np.column_stack([np.asarray(t_edges[:n]) / 1e6,
                           np.asarray(t_cum[:n]),
                           np.asarray(i_cum[:n])])
    np.savetxt(path, arr, delimiter=",", comments="",
               header="position_mb,cum_cM_truth,cum_cM_inferred", fmt="%.6g")


def save_map_data(maps, path):
    """Pickle the cM curves so the figure can be re-styled without re-running the
    (expensive) HMM decode."""
    with open(path, "wb") as f:
        pickle.dump(maps, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote map data: {path}")


def save_outputs(maps, out_dir,
                 composite_title="Recombination map (cumulative genetic distance)"):
    """Write the full output tree:

        <out_dir>/composite.png              faceted all-chromosome figure (panel-B)
        <out_dir>/map_data.pkl               all cM curves (for re-styling)
        <out_dir>/chromosomes/<contig>.png   per-chromosome Marey figure
        <out_dir>/chromosomes/<contig>.csv   per-chromosome cM curve (truth+inferred)
    """
    os.makedirs(out_dir, exist_ok=True)
    chrom_dir = os.path.join(out_dir, "chromosomes")
    os.makedirs(chrom_dir, exist_ok=True)

    save_map_data(maps, os.path.join(out_dir, "map_data.pkl"))
    plot_maps(maps, os.path.join(out_dir, "composite.png"), title=composite_title)

    for contig, data in maps.items():
        plot_one_chromosome(contig, data, os.path.join(chrom_dir, f"{contig}.png"))
        write_chromosome_csv(contig, data, os.path.join(chrom_dir, f"{contig}.csv"))
    print(f"Wrote composite + {len(maps)} per-chromosome figures/CSVs under {out_dir}/")


# =============================================================================
# Synthetic driver self-test (exercises painting->alleles->HMM->decode->cM,
# bypassing only the pickle loader; the HMM core is validated separately).
# =============================================================================
class _Chunk:
    __slots__ = ("start", "end", "hap1", "hap2")

    def __init__(self, start, end, hap1, hap2):
        self.start, self.end, self.hap1, self.hap2 = start, end, hap1, hap2


class _Sample:
    def __init__(self, sample_index, chunks):
        self.sample_index = sample_index
        self.chunks = chunks


def _selftest_driver(seed=3):
    """Constant-rate gametes (5e-8/bp) from two clean founder-parents; the
    recovered aggregate map should be ~linear at ~5 cM/Mb."""
    rng = np.random.default_rng(seed)
    n_snps = 4000
    L = 100_000_000
    snp_positions = np.sort(rng.choice(np.arange(1, L), size=n_snps, replace=False)).astype(np.float64)

    # Four distinct founder haplotypes (ids 0..3): 0/1 for parent P1, 2/3 for P2.
    f0 = rng.integers(0, 2, size=n_snps).astype(np.int8)
    f1 = 1 - f0
    f2 = rng.integers(0, 2, size=n_snps).astype(np.int8)
    f3 = 1 - f2
    haplotypes = {0: f0, 1: f1, 2: f2, 3: f3}
    allele_lookup, _ = build_founder_allele_lookup(snp_positions, haplotypes)

    # Painted parents: P1 = (founder 0, founder 1); P2 = (founder 2, founder 3).
    P1 = _Sample(0, [_Chunk(int(snp_positions[0]), int(L), 0, 1)])
    P2 = _Sample(1, [_Chunk(int(snp_positions[0]), int(L), 2, 3)])

    def gamete_chunks(fa, fb):
        """Chunks for one gamete: constant-rate crossovers between founders fa/fb."""
        chunks = []
        pos = int(snp_positions[0])
        cur = fa if rng.random() < 0.5 else fb
        while True:
            gap = rng.exponential(1.0 / RECOMB_RATE)
            nxt = min(L, pos + gap)
            chunks.append((pos, int(nxt), cur))
            if nxt >= L:
                break
            pos = int(nxt)
            cur = fb if cur == fa else fa
        return chunks

    samples_by_name = {"P1": P1, "P2": P2}
    links = {}
    n_children = 400
    for i in range(n_children):
        g1 = gamete_chunks(0, 1)   # from P1
        g2 = gamete_chunks(2, 3)   # from P2
        # Merge the two single-haplotype gametes into diploid PaintedChunks.
        bounds = sorted({b for (s, e, _f) in g1 for b in (s, e)} |
                        {b for (s, e, _f) in g2 for b in (s, e)})
        chunks = []
        for k in range(len(bounds) - 1):
            s, e = bounds[k], bounds[k + 1]
            if s >= e:
                continue
            mid = (s + e) / 2
            h1 = next(f for (cs, ce, f) in g1 if cs <= mid < ce)
            h2 = next(f for (cs, ce, f) in g2 if cs <= mid < ce)
            chunks.append(_Chunk(s, e, h1, h2))
        name = f"C{i}"
        samples_by_name[name] = _Sample(2 + i, chunks)
        links[name] = ("P1", "P2")

    mids, n_mei, n_ch = collect_crossovers(samples_by_name, links, allele_lookup,
                                           snp_positions)
    edges, cum = cumulative_cM(mids, n_mei, float(snp_positions[0]),
                               float(snp_positions[-1]), bin_bp=2_000_000)
    span_mb = (edges[-1] - edges[0]) / 1e6
    slope = cum[-1] / span_mb     # cM per Mb
    # Linearity: cumulative cM should track physical fraction.
    frac_pos = (edges[1:-1] - edges[0]) / (edges[-1] - edges[0])
    frac_cm = cum[1:-1] / cum[-1] if cum[-1] > 0 else np.zeros_like(frac_pos)
    max_dev = float(np.max(np.abs(frac_cm - frac_pos))) if len(frac_pos) else 1.0

    cond_slope = 3.5 < slope < 6.5      # target ~5 cM/Mb
    cond_lin = max_dev < 0.08
    print(f"[driver] {n_ch} children, {n_mei} meioses; total {cum[-1]:.0f} cM over "
          f"{span_mb:.0f} Mb -> {slope:.2f} cM/Mb (target ~5) "
          f"-> {'OK' if cond_slope else 'FAIL'}")
    print(f"[driver] linearity max dev {max_dev:.3f} -> {'OK' if cond_lin else 'FAIL'}")
    return cond_slope and cond_lin


# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Build a Marey recombination map "
                                            "(truth vs inferred) from BHD checkpoints.")
    p.add_argument("--ckpt-dir", default=".pipeline_checkpoints",
                   help="completed pipeline.py checkpoint tree (NOT a sweep combo)")
    p.add_argument("--bin-mb", type=float, default=1.0, help="physical bin width (Mb)")
    p.add_argument("--out-dir", default="recombination_map",
                   help="output directory: composite.png + map_data.pkl + "
                        "chromosomes/<contig>.{png,csv}")
    p.add_argument("--contigs", nargs="*", default=None,
                   help="subset of contigs (default: all in stage 02 global)")
    p.add_argument("--true-pedigree-for-inferred", action="store_true",
                   help="ablation: build the inferred map with the TRUE pedigree links")
    p.add_argument("--inferred-pedigree-csv", default=None,
                   help="explicit path to pedigree_inference_discovered.csv")
    p.add_argument("--selftest", action="store_true",
                   help="run synthetic validation and exit (no checkpoints needed)")
    args = p.parse_args(argv)

    if args.selftest:
        ok = _selftest_driver()
        print("=== DRIVER SELF-TEST", "PASSED" if ok else "FAILED", "===")
        return 0 if ok else 1

    maps = build_maps(
        args.ckpt_dir, bin_bp=args.bin_mb * 1e6,
        use_inferred_pedigree=not args.true_pedigree_for_inferred,
        inferred_csv=args.inferred_pedigree_csv,
        contigs=args.contigs,
    )
    save_outputs(maps, args.out_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())