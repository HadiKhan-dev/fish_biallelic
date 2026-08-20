#!/usr/bin/env python3
"""
Marey-style recombination-rate recovery analysis for BHD simulations.

The map separates five scientifically distinct sources of information (six
when the complete raw event log is present):

* analytic process expectation from Stage-2 ``recombination_model``;
* raw realized simulator events, both all gametes and the F2/F3 subset that the
  painting decoder can observe;
* true painting decoded with the true pedigree;
* reconstructed painting decoded with the true pedigree (painting ablation);
* reconstructed painting decoded with the inferred pedigree (end-to-end map).

Decoded tracks use the same phase-aware 16-state trio HMM and backtracking
crossover localization as the original map.  The HMM always receives one
scalar mean recombination rate.  It is intentionally blind to any simulated
piecewise profile, so recovery of elevated chromosome ends is not built into
the decoder.  Analytic and regional calculations use exact profile boundaries;
physical plotting bins always terminate at the recorded modeled span.

Production reconstructed tracks consume the atomic Stage-11 H1/Z1 bundle:
the pre-phase ``tolerance_result`` painting (Z1) and the single canonical
founder block (H1) it was painted against.
The bundle's ordered sample IDs must exactly match the Stage-2 global order.
The map deliberately does not use the Stage-13 Q1 painting because its trio
decoder already models phase.

Legacy checkpoints without model metadata remain supported as a constant
5e-8/bp process and retain the historic ``process``, ``truth``, and ``inferred``
map keys and CSV column names.

Usage::

    python recombination_map.py --ckpt-dir .pipeline_checkpoints --workers 22
    python recombination_map.py --bin-mb 1.0 --out-dir recombination_map
    python recombination_map.py --true-pedigree-for-inferred
    python recombination_map.py --selftest

Outputs under ``--out-dir`` include ``composite.png``, ``map_data.pkl``,
per-chromosome PNG/CSV files, a normalized-position local-rate plot and CSV,
regional rate/ratio CSV+JSON summaries, and ``README.txt``.  ``map_data.pkl``
stores compact crossover midpoints and explicit meiosis exposure so aggregate
analyses can be reproduced without rerunning the HMM.
"""

import math

import numpy as np
from numba import njit

from founder_alleles import founder_allele_matrix
from pedigree_hmm import poisson_switch_stay_terms


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
    """Build the founder-ID-indexed deterministic allele lookup."""
    snp_positions = positions
    return founder_allele_matrix(haplotypes, len(snp_positions)), snp_positions


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
    _, switch_costs, stay_costs = poisson_switch_stay_terms(
        snp_positions, recomb_rate
    )
    return switch_costs, stay_costs


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
# Layout (pipeline.py): <ckpt>/<stage>/<contig>.p5.b2 and
# <ckpt>/<stage>/_global.p5.b2, each a protocol-5/Blosc frame read via the
# shared checkpoint_io module.  Unpickling the painting objects needs
# `paint_samples` importable (it defines SamplePainting/PaintedChunk/
# BlockPainting); run this from the haplotype_reconstruction dir with bio-env
# active.  `paint_samples` is imported ONLY for those data classes -- no
# detection machinery is imported.
# =============================================================================
import csv
import json
import os
import pickle
import multiprocessing as mp

import checkpoint_io
import pipeline_runtime

STAGE_VCF = "01_vcf_discovery"       # naive_long_haps (true founders)
STAGE_SIM = "02_simulation"          # truth_painting + truth_pedigree
STAGE_PAINT = "11_viterbi_painting"  # atomic H1 founder block + Z1 painting


def load_global(ckpt_dir, stage):
    return checkpoint_io.read(checkpoint_io.global_path(ckpt_dir, stage))


def load_contig(ckpt_dir, stage, contig):
    return checkpoint_io.read(checkpoint_io.contig_path(ckpt_dir, stage, contig))


def load_stage11_painting_payload(
    ckpt_dir, contig, *, expected_sample_ids=None, contig_loader=load_contig
):
    """Load the required atomic H1/Z1 painting bundle for one contig.

    The Stage-11 checkpoint is the sole production source for both objects.
    An incomplete bundle is rejected rather than mixing a Z1 painting with an
    assembly-stage or other founder panel. When expected sample IDs are given,
    their content and order must exactly match the bundled identifiers.
    """
    payload = contig_loader(ckpt_dir, STAGE_PAINT, contig)
    path = checkpoint_io.contig_path(ckpt_dir, STAGE_PAINT, contig)
    pipeline_runtime.validate_painting_bundle(
        payload,
        expected_sample_ids=expected_sample_ids,
        context=f"Painting checkpoint {path}",
    )
    return payload


def _samples_of(painting_obj):
    """Normalize a painting checkpoint to a list of per-sample painting objects.

    Stage 02 `truth_painting` is a list of SamplePainting; Stage 11 Z1
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


def build_inferred_founder_lookup(founder_block):
    """Discovered founders from the single Stage-11 H1 block bundled with Z1."""
    return build_founder_allele_lookup(
        np.asarray(founder_block.positions), dict(founder_block.haplotypes)
    )


# =============================================================================
# Per-painting crossover collection
# =============================================================================
def collect_crossovers(samples_by_name, links, allele_lookup, snp_positions,
                       recomb_rate=RECOMB_RATE, verbose=False,
                       diploid_by_name=None):
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

    if diploid_by_name is None:
        diploid_by_name = {
            name: diploid_alleles_for_sample(sample.chunks, allele_lookup,
                                              snp_positions)
            for name, sample in samples_by_name.items()
        }

    for child, (p1, p2) in links.items():
        cs = samples_by_name.get(child)
        ps1 = samples_by_name.get(p1)
        ps2 = samples_by_name.get(p2)
        if cs is None or ps1 is None or ps2 is None:
            continue

        child_dip = diploid_by_name[child]
        p1_dip = diploid_by_name[p1]
        p2_dip = diploid_by_name[p2]
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


def _profile_segments(recombination_model):
    """Return fractional ``(start, end, rate_per_bp)`` process segments.

    Stage-2 checkpoints created before spatially varying simulations have no
    model metadata; those remain a single constant-rate segment.  New-schema
    ``rate_multiplier`` values are already normalized and directly multiply
    ``mean_rate_per_bp``.
    """
    if not recombination_model:
        return [(0.0, 1.0, RECOMB_RATE)]

    mean_rate = float(recombination_model.get("mean_rate_per_bp", RECOMB_RATE))
    raw = recombination_model.get("profile_segments") or []
    if not raw:
        return [(0.0, 1.0, mean_rate)]

    parsed = []
    for item in raw:
        if isinstance(item, dict):
            start = float(item.get("start_fraction", item.get("start", 0.0)))
            end = float(item.get("end_fraction", item.get("end", 1.0)))
            absolute = item.get("rate_per_bp")
            multiplier = item.get("rate_multiplier", item.get("multiplier", 1.0))
        else:
            if len(item) != 3:
                raise ValueError(f"Invalid recombination profile segment: {item!r}")
            start, end, multiplier = map(float, item)
            absolute = None
        if not (0.0 <= start < end <= 1.0):
            raise ValueError(f"Invalid recombination profile interval [{start}, {end}]")
        rate = float(absolute) if absolute is not None else mean_rate * float(multiplier)
        if rate < 0.0:
            raise ValueError("Recombination rates must be non-negative")
        parsed.append((start, end, rate))

    parsed.sort(key=lambda x: x[0])
    if abs(parsed[0][0]) > 1e-12 or abs(parsed[-1][1] - 1.0) > 1e-12:
        raise ValueError("Recombination profile must cover fractional interval [0, 1]")
    for left, right in zip(parsed[:-1], parsed[1:]):
        if abs(left[1] - right[0]) > 1e-12:
            raise ValueError("Recombination profile segments must be contiguous")
    return parsed


def _physical_edges(lo, hi, bin_bp, recombination_model=None):
    """Regular bins plus exact process boundaries, ending exactly at ``hi``."""
    if hi <= lo:
        return np.asarray([lo, hi], dtype=np.float64)
    n_bins = max(1, int(np.ceil((hi - lo) / float(bin_bp))))
    regular = lo + np.arange(n_bins + 1, dtype=np.float64) * float(bin_bp)
    # ceil normally places the final regular edge above hi.  Always replacing
    # it fixes the supported final-bin overshoot and its excess exposure.
    regular[-1] = hi
    span = hi - lo
    segments = _profile_segments(recombination_model)
    boundaries = [lo + start * span for start, _, _ in segments]
    boundaries.extend(lo + end * span for _, end, _ in segments)
    return np.unique(np.clip(np.concatenate((regular, boundaries)), lo, hi))


def cumulative_cM(midpoints, n_meioses, lo, hi, bin_bp,
                  recombination_model=None):
    """Bin crossover midpoints -> cumulative cM along [lo, hi].

    cM per bin = 100 * (#crossovers in bin) / n_meioses  (Morgans -> cM);
    cumulative along the chromosome.  Returns (edges_bp, cum_cM) for a step plot.
    """
    if n_meioses == 0 or hi <= lo:
        return np.array([lo, hi], dtype=float), np.array([0.0, 0.0])
    edges = _physical_edges(lo, hi, bin_bp, recombination_model)
    counts, _ = np.histogram(midpoints, bins=edges)
    cm_per_bin = 100.0 * counts / float(n_meioses)
    cum = np.concatenate([[0.0], np.cumsum(cm_per_bin)])
    return edges, cum


def process_truth_cM(lo, hi, bin_bp, recomb_rate=RECOMB_RATE,
                     recombination_model=None):
    """Analytic cumulative genetic map under the recorded simulation model.

    Legacy checkpoints use a constant ``recomb_rate``.  New checkpoints carry
    fractional piecewise-constant segments in ``recombination_model``.  Exact
    segment boundaries are included in the returned edges, so the curve and
    regional analysis do not smear the 20/80-percent changes across a bin.
    """
    if hi <= lo:
        return np.array([lo, hi], dtype=float), np.array([0.0, 0.0])
    if recombination_model is None:
        recombination_model = {"mean_rate_per_bp": recomb_rate}
    edges = _physical_edges(lo, hi, bin_bp, recombination_model)
    segments = _profile_segments(recombination_model)
    span = hi - lo
    cum = np.zeros(len(edges), dtype=np.float64)
    for j in range(1, len(edges)):
        a, b = edges[j - 1], edges[j]
        midpoint_fraction = ((a + b) * 0.5 - lo) / span
        rate = next(rate for seg_start, seg_end, rate in segments
                    if seg_start - 1e-12 <= midpoint_fraction <= seg_end + 1e-12)
        cum[j] = cum[j - 1] + 100.0 * rate * (b - a)
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
def _warmup_hmm():
    """Compile the numba trio-HMM once in the parent (with the SAME dtypes the
    real decode uses: int8 diploid alleles, bool hom mask, float64 costs) so the
    forked workers load the cached compile instead of each triggering it in
    parallel.  Pure compilation warm-up; no effect on any result.  Best-effort:
    `cache=True` already persists the compile to disk, so this is just to avoid a
    first-run compile storm across workers.
    """
    n = 8
    snp = (np.arange(1, n + 1, dtype=np.float64) * 1000.0)
    child = np.zeros((n, 2), dtype=np.int8)
    par = np.zeros((n, 2), dtype=np.int8)
    hom = compute_hom_mask(child)
    sw, st = build_switch_stay_costs(snp)
    try:
        run_trio_phase_aware_hmm_backtrack(
            child, hom, par, par, sw, st,
            ERROR_PENALTY, PHASE_PENALTY, MISMATCH_PENALTY)
    except Exception:  # pragma: no cover -- warm-up must never be fatal
        pass


TRACK_ORDER = (
    "process", "raw_all", "raw", "truth", "reconstructed_true", "inferred",
)
TRACK_LABELS = {
    "process": "Analytic process expectation",
    "raw_all": "Raw realized (all simulated meioses)",
    "raw": "Raw realized (decoder-eligible meioses)",
    "truth": "True painting + true pedigree",
    "reconstructed_true": "Reconstructed painting + true pedigree",
    "inferred": "Reconstructed painting + inferred pedigree",
}
TRACK_STYLES = {
    "process": dict(color="#2e7d32", linestyle="--", linewidth=1.5),
    "raw_all": dict(color="#777777", linestyle=":", linewidth=1.1),
    "raw": dict(color="#111111", linestyle="-.", linewidth=1.3),
    "truth": dict(color="#1f4e8c", linestyle="-", linewidth=1.4),
    "reconstructed_true": dict(color="#d17c00", linestyle="-", linewidth=1.3),
    "inferred": dict(color="#c0392b", linestyle="-", linewidth=1.4),
}


def _modeled_span(recombination_model, contig, truth_positions):
    spans = (recombination_model or {}).get("modeled_spans_bp", {})
    span = spans.get(contig) if hasattr(spans, "get") else None
    if span is None:
        return float(truth_positions[0]), float(truth_positions[-1])
    lo, hi = map(float, span)
    if hi <= lo:
        raise ValueError(f"Invalid modeled span for {contig}: {span!r}")
    return lo, hi


def _raw_event_track(records, eligible_children=None):
    """Flatten one event-log record per gamete, retaining zero-event meioses."""
    if records is None:
        return None
    selected = []
    for record in records:
        if eligible_children is not None:
            if record.get("child") not in eligible_children:
                continue
            # Explicit generation filter documents the current decoder's scope.
            if record.get("generation") not in (None, "F2", "F3"):
                continue
        selected.append(record)
    arrays = [np.asarray(r.get("crossover_positions_bp", ()), dtype=np.float64)
              for r in selected]
    nonempty = [a for a in arrays if a.size]
    midpoints = np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float64)
    return midpoints, len(selected), len({r.get("child") for r in selected})


def _compact_midpoints(midpoints):
    """Float32 is sub-marker precision here while halving stored map-data size."""
    return np.asarray(midpoints, dtype=np.float32)


def _build_one_contig(args):
    """Build all process, raw-event, painting, and pedigree ablation tracks."""
    (contig, ckpt_dir, bin_bp, true_links, inferred_links, sample_names,
     recombination_model, raw_records) = args

    vcf_payload = load_contig(ckpt_dir, STAGE_VCF, contig)
    naive = vcf_payload["naive_long_haps"]
    truth_lookup, truth_pos = build_truth_founder_lookup(naive)
    del naive, vcf_payload

    # Load Z1 and its exact H1 founder panel atomically from Stage 11.  Keep
    # only the compact allele lookup and normalized paintings afterwards.
    painting_payload = load_stage11_painting_payload(
        ckpt_dir, contig, expected_sample_ids=sample_names
    )
    founder_block = painting_payload[pipeline_runtime.FOUNDER_BLOCK_KEY]
    inf_lookup, inf_pos = build_inferred_founder_lookup(founder_block)
    inf_obj = painting_payload["tolerance_result"]
    inf_samples = _samples_of(inf_obj)
    del founder_block, inf_obj, painting_payload
    inf_by_name = _index_by_name(inf_samples, sample_names)

    simulation_payload = load_contig(ckpt_dir, STAGE_SIM, contig)
    truth_painting = simulation_payload["truth_painting"]
    truth_samples = _samples_of(truth_painting)
    del truth_painting, simulation_payload
    truth_by_name = _index_by_name(truth_samples, sample_names)

    decoder_rate = float((recombination_model or {}).get(
        "decoder_rate_per_bp", RECOMB_RATE))
    # Deliberately scalar: the decoder remains blind to the simulated profile.
    truth_dip = {
        name: diploid_alleles_for_sample(sample.chunks, truth_lookup, truth_pos)
        for name, sample in truth_by_name.items()
    }
    inf_dip = {
        name: diploid_alleles_for_sample(sample.chunks, inf_lookup, inf_pos)
        for name, sample in inf_by_name.items()
    }
    t_mid, t_n, t_nc = collect_crossovers(
        truth_by_name, true_links, truth_lookup, truth_pos, decoder_rate,
        diploid_by_name=truth_dip)
    rt_mid, rt_n, rt_nc = collect_crossovers(
        inf_by_name, true_links, inf_lookup, inf_pos, decoder_rate,
        diploid_by_name=inf_dip)
    i_mid, i_n, i_nc = collect_crossovers(
        inf_by_name, inferred_links, inf_lookup, inf_pos, decoder_rate,
        diploid_by_name=inf_dip)

    lo, hi = _modeled_span(recombination_model, contig, truth_pos)
    track_events = {
        "truth": (t_mid, t_n, t_nc),
        "reconstructed_true": (rt_mid, rt_n, rt_nc),
        "inferred": (i_mid, i_n, i_nc),
    }
    eligible_children = {
        child for child, (p1, p2) in true_links.items()
        if child in truth_by_name and p1 in truth_by_name and p2 in truth_by_name
    }
    raw_all = _raw_event_track(raw_records)
    raw_eligible = _raw_event_track(raw_records, eligible_children)
    if raw_all is not None:
        track_events["raw_all"] = raw_all
        track_events["raw"] = raw_eligible

    curves = {}
    compact_midpoints = {}
    n_meioses = {}
    n_children = {}
    for key, (midpoints, meioses, children) in track_events.items():
        curves[key] = cumulative_cM(midpoints, meioses, lo, hi, bin_bp,
                                    recombination_model)
        compact_midpoints[key] = _compact_midpoints(midpoints)
        n_meioses[key] = int(meioses)
        n_children[key] = int(children)
    curves["process"] = process_truth_cM(
        lo, hi, bin_bp, recombination_model=recombination_model)

    profile_boundaries = np.asarray(
        [lo + f * (hi - lo)
         for segment in _profile_segments(recombination_model)
         for f in segment[:2]], dtype=np.float64)
    out = dict(curves)
    out.update({
        "track_labels": {
            **{key: TRACK_LABELS[key] for key in curves},
            "process": (f"Analytic expectation ({recombination_model.get('name', 'legacy_constant')})"
                        if recombination_model else TRACK_LABELS["process"]),
        },
        "track_provenance": {
            "process": "Stage-2 analytic recombination_model",
            "raw_all": "Stage-2 raw event log; all simulated gametes",
            "raw": "Stage-2 raw event log; F2/F3 decoder-eligible gametes",
            "truth": "Stage-2 truth painting decoded with the true pedigree",
            "reconstructed_true": "Stage-11 pre-phase Z1 painting + bundled H1 founder block, decoded with the true pedigree",
            "inferred": "Stage-11 pre-phase Z1 painting + bundled H1 founder block, decoded with the selected pedigree",
        },
        "crossover_midpoints_bp": compact_midpoints,
        "n_meioses_by_track": n_meioses,
        "n_children_by_track": n_children,
        "exposure_bp_by_track": {
            key: float(value) * (hi - lo) for key, value in n_meioses.items()
        },
        "profile_boundaries_bp": np.unique(profile_boundaries),
        "recombination_model": recombination_model,
        "decoder_rate_per_bp": decoder_rate,
        "lo": lo,
        "hi": hi,
        # Backward-compatible scalar fields used by earlier plotting notebooks.
        "n_meioses_truth": t_n,
        "n_meioses_inferred": i_n,
        "n_crossovers_truth": len(t_mid),
        "n_crossovers_inferred": len(i_mid),
        "n_children_truth": t_nc,
        "n_children_inferred": i_nc,
    })
    return contig, out


# =============================================================================
# Map driver
# =============================================================================
def build_maps(ckpt_dir, bin_bp, use_inferred_pedigree=True, inferred_csv=None,
               contigs=None, verbose=True, n_workers=None):
    """Build backward-compatible maps plus raw and pedigree/painting ablations."""
    g = load_global(ckpt_dir, STAGE_SIM)
    truth_pedigree = g["truth_pedigree"]
    region_keys = g["region_keys"]
    sample_names = list(truth_pedigree["Sample"])
    recombination_model = g.get("recombination_model") or {
        "name": "legacy_constant",
        "mean_rate_per_bp": RECOMB_RATE,
        "decoder_rate_per_bp": RECOMB_RATE,
        "profile_segments": [
            {"start_fraction": 0.0, "end_fraction": 1.0,
             "rate_multiplier": 1.0}
        ],
        "normalization": "legacy_constant",
    }
    raw_by_contig = g.get("raw_recombination_events") or {}

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
    contigs = list(contigs)
    if n_workers is None:
        n_workers = min(len(contigs), os.cpu_count() or 1)
    n_workers = max(1, int(n_workers))
    tasks = [
        (contig, ckpt_dir, bin_bp, true_links, inferred_links, sample_names,
         recombination_model, raw_by_contig.get(contig))
        for contig in contigs
    ]

    if n_workers == 1 or len(tasks) == 1:
        results = [_build_one_contig(task) for task in tasks]
    else:
        _warmup_hmm()
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_build_one_contig, tasks)

    by_contig = {contig: data for contig, data in results}
    out = {}
    for contig in contigs:
        data = by_contig[contig]
        if use_inferred_pedigree:
            data["pedigree_source"] = (inferred_csv or
                                         "pedigree_inference_discovered.csv")
        else:
            data["pedigree_source"] = "true_pedigree_cli_ablation"
            data["track_labels"]["inferred"] = (
                "Reconstructed painting + true pedigree (CLI ablation duplicate)")
        out[contig] = data
        if verbose:
            span_mb = (data["hi"] - data["lo"]) / 1e6
            pieces = [f"{contig}: process {data['process'][1][-1]:.1f} cM"]
            for key in ("raw", "truth", "reconstructed_true", "inferred"):
                if key not in data:
                    continue
                total_cm = data[key][1][-1]
                meioses = data["n_meioses_by_track"][key]
                count = len(data["crossover_midpoints_bp"][key])
                rate = total_cm / span_mb if span_mb > 0 else 0.0
                pieces.append(f"{key} {rate:.2f} cM/Mb ({count} xo/{meioses} mei)")
            print("  " + " | ".join(pieces))
    return out


# =============================================================================
# Plotting (basic faceted Marey map; detailed styling is a later pass)
# =============================================================================
def _available_tracks(data):
    return [key for key in TRACK_ORDER
            if key in data and isinstance(data[key], tuple) and len(data[key]) == 2]


def _label(data, key):
    return data.get("track_labels", {}).get(key, TRACK_LABELS.get(key, key))


def plot_maps(maps, out_path, title="Recombination map (cumulative genetic distance)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contigs = list(maps)
    ncols = 5
    nrows = int(np.ceil(len(contigs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.15 * ncols, 2.5 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for j, contig in enumerate(contigs):
        ax = axes[j // ncols][j % ncols]
        ax.set_visible(True)
        data = maps[contig]
        for key in _available_tracks(data):
            edges, cumulative = data[key]
            style = TRACK_STYLES[key]
            if key == "process":
                ax.plot(np.asarray(edges) / 1e6, cumulative,
                        label=_label(data, key), **style)
            else:
                ax.step(np.asarray(edges) / 1e6, cumulative, where="post",
                        label=_label(data, key), **style)
        ax.set_title(str(contig), fontsize=9)
        ax.tick_params(labelsize=7)

    fig.supxlabel("Physical position (Mb)")
    fig.supylabel("Genetic distance (cM)")
    fig.suptitle(title, fontsize=12)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=7.5, frameon=False)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote figure: {out_path}")


def plot_one_chromosome(contig, data, path):
    """Single-chromosome map with every available provenance track."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for key in _available_tracks(data):
        edges, cumulative = data[key]
        style = TRACK_STYLES[key]
        label = _label(data, key)
        if key in data.get("n_meioses_by_track", {}):
            label += f" ({data['n_meioses_by_track'][key]} meioses)"
        if key == "process":
            ax.plot(np.asarray(edges) / 1e6, cumulative, label=label, **style)
        else:
            ax.step(np.asarray(edges) / 1e6, cumulative, where="post",
                    label=label, **style)
    for boundary in data.get("profile_boundaries_bp", ())[1:-1]:
        ax.axvline(boundary / 1e6, color="#bbbbbb", linewidth=0.7, zorder=0)
    ax.set_xlabel("Physical position (Mb)")
    ax.set_ylabel("Genetic distance (cM)")
    ax.set_title(f"Recombination map - {contig}")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_chromosome_csv(contig, data, path):
    """Write aligned cumulative curves, retaining legacy column names."""
    ordered = [key for key in ("process", "truth", "inferred", "raw_all", "raw",
                               "reconstructed_true") if key in data]
    all_edges = np.unique(np.concatenate([np.asarray(data[key][0], dtype=float)
                                          for key in ordered]))
    columns = [all_edges / 1e6]
    names = ["position_mb"]
    legacy_names = {
        "process": "cum_cM_process",
        "truth": "cum_cM_truth",
        "inferred": "cum_cM_inferred",
    }
    for key in ordered:
        edges, cumulative = data[key]
        columns.append(np.interp(all_edges, np.asarray(edges), np.asarray(cumulative)))
        names.append(legacy_names.get(key, f"cum_cM_{key}"))
    np.savetxt(path, np.column_stack(columns), delimiter=",", comments="",
               header=",".join(names), fmt="%.8g")


def _expected_rate(model, start_fraction, end_fraction):
    width = end_fraction - start_fraction
    integrated = 0.0
    for seg_start, seg_end, rate in _profile_segments(model):
        overlap = max(0.0, min(end_fraction, seg_end) - max(start_fraction, seg_start))
        integrated += overlap * rate
    return 100.0 * 1e6 * integrated / width if width > 0.0 else float("nan")


def _empirical_region_components(data, key, intervals):
    mids = np.asarray(data.get("crossover_midpoints_bp", {}).get(key, ()), dtype=float)
    n_meioses = int(data.get("n_meioses_by_track", {}).get(key, 0))
    lo, hi = float(data["lo"]), float(data["hi"])
    span = hi - lo
    count = 0
    width_bp = 0.0
    for start_fraction, end_fraction in intervals:
        left = lo + start_fraction * span
        right = lo + end_fraction * span
        is_last = abs(end_fraction - 1.0) < 1e-12
        count += int(np.count_nonzero((mids >= left) &
                                     ((mids <= right) if is_last else (mids < right))))
        width_bp += right - left
    return count, n_meioses * width_bp


def _bootstrap_rate_ci(components, n_bootstrap=2000, seed=104729):
    if len(components) < 2:
        return (float("nan"), float("nan"))
    counts = np.asarray([item[0] for item in components], dtype=float)
    exposures = np.asarray([item[1] for item in components], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(components), size=(n_bootstrap, len(components)))
    boot_counts = counts[indices].sum(axis=1)
    boot_exposure = exposures[indices].sum(axis=1)
    valid = boot_exposure > 0.0
    values = 100.0 * 1e6 * boot_counts[valid] / boot_exposure[valid]
    if values.size == 0:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(values, [2.5, 97.5]))


def _bootstrap_ratio_ci(end_components, middle_components,
                        n_bootstrap=2000, seed=130363):
    if len(end_components) < 2:
        return (float("nan"), float("nan"))
    ec = np.asarray([item[0] for item in end_components], dtype=float)
    ee = np.asarray([item[1] for item in end_components], dtype=float)
    mc = np.asarray([item[0] for item in middle_components], dtype=float)
    me = np.asarray([item[1] for item in middle_components], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ec), size=(n_bootstrap, len(ec)))
    ec_b, ee_b = ec[indices].sum(axis=1), ee[indices].sum(axis=1)
    mc_b, me_b = mc[indices].sum(axis=1), me[indices].sum(axis=1)
    valid = (ee_b > 0.0) & (me_b > 0.0) & (mc_b > 0.0)
    end_rate = np.divide(ec_b, ee_b, out=np.full_like(ec_b, np.nan), where=ee_b > 0.0)
    mid_rate = np.divide(mc_b, me_b, out=np.full_like(mc_b, np.nan), where=me_b > 0.0)
    ratios = end_rate[valid] / mid_rate[valid]
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(ratios, [2.5, 97.5]))


def regional_summary(maps, n_bootstrap=2000):
    """Pooled regional rates and chromosome-bootstrap uncertainty."""
    model = next(iter(maps.values())).get("recombination_model") or {}
    regions = {
        "left20": [(0.0, 0.2)],
        "middle60": [(0.2, 0.8)],
        "right20": [(0.8, 1.0)],
        "pooled_ends": [(0.0, 0.2), (0.8, 1.0)],
    }
    tracks = _available_tracks(next(iter(maps.values())))
    rate_rows = []
    ratio_rows = []
    components = {}
    for key in tracks:
        components[key] = {}
        for region, intervals in regions.items():
            if key == "process":
                if region == "pooled_ends":
                    rate = 0.5 * (_expected_rate(model, 0.0, 0.2) +
                                  _expected_rate(model, 0.8, 1.0))
                else:
                    rate = _expected_rate(model, intervals[0][0], intervals[0][1])
                count = exposure = None
                ci = (None, None)
                comps = []
            else:
                comps = [_empirical_region_components(data, key, intervals)
                         for data in maps.values()]
                count = sum(item[0] for item in comps)
                exposure = sum(item[1] for item in comps)
                rate = 100.0 * 1e6 * count / exposure if exposure > 0 else float("nan")
                ci = _bootstrap_rate_ci(comps, n_bootstrap=n_bootstrap)
            components[key][region] = comps
            rate_rows.append({
                "track": key,
                "label": _label(next(iter(maps.values())), key),
                "region": region,
                "n_crossovers": count,
                "exposure_meiosis_bp": exposure,
                "rate_cM_per_mb": rate,
                "bootstrap_ci95_low": ci[0],
                "bootstrap_ci95_high": ci[1],
            })

        by_region = {row["region"]: row for row in rate_rows if row["track"] == key}
        for numerator_region in ("left20", "right20", "pooled_ends"):
            ratio = (by_region[numerator_region]["rate_cM_per_mb"] /
                     by_region["middle60"]["rate_cM_per_mb"])
            if key == "process":
                ratio_ci = (None, None)
            else:
                ratio_ci = _bootstrap_ratio_ci(
                    components[key][numerator_region],
                    components[key]["middle60"],
                    n_bootstrap=n_bootstrap)
            ratio_rows.append({
                "track": key,
                "label": _label(next(iter(maps.values())), key),
                "contrast": f"{numerator_region}/middle60",
                "ratio": ratio,
                "bootstrap_ci95_low": ratio_ci[0],
                "bootstrap_ci95_high": ratio_ci[1],
            })
    return rate_rows, ratio_rows


def normalized_local_rates(maps, n_bins=20):
    """Pool chromosomes on fractional position using meiosis-length exposure."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    tracks = _available_tracks(next(iter(maps.values())))
    rows = []
    for key in tracks:
        for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            if key == "process":
                rate = _expected_rate(
                    next(iter(maps.values())).get("recombination_model") or {},
                    left, right)
                count = exposure = None
            else:
                pieces = [_empirical_region_components(data, key, [(left, right)])
                          for data in maps.values()]
                count = sum(piece[0] for piece in pieces)
                exposure = sum(piece[1] for piece in pieces)
                rate = 100.0 * 1e6 * count / exposure if exposure > 0 else float("nan")
            rows.append({
                "track": key,
                "label": _label(next(iter(maps.values())), key),
                "bin_start_fraction": left,
                "bin_end_fraction": right,
                "bin_center_fraction": centers[idx],
                "n_crossovers": count,
                "exposure_meiosis_bp": exposure,
                "rate_cM_per_mb": rate,
            })
    return rows


def _write_dict_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_normalized_local_rates(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    tracks = []
    for row in rows:
        if row["track"] not in tracks:
            tracks.append(row["track"])
    for key in tracks:
        subset = [row for row in rows if row["track"] == key]
        ax.plot([100.0 * row["bin_center_fraction"] for row in subset],
                [row["rate_cM_per_mb"] for row in subset],
                marker=None if key == "process" else "o",
                markersize=3.0, label=subset[0]["label"], **TRACK_STYLES[key])
    ax.axvspan(0, 20, color="#e8f5e9", alpha=0.45, zorder=0)
    ax.axvspan(80, 100, color="#e8f5e9", alpha=0.45, zorder=0)
    ax.axvline(20, color="#888888", linewidth=0.8)
    ax.axvline(80, color="#888888", linewidth=0.8)
    ax.set_xlabel("Normalized chromosome position (%)")
    ax.set_ylabel("Local recombination rate (cM/Mb)")
    ax.set_title("Recombination-rate recovery pooled across chromosomes")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_analysis_outputs(maps, out_dir):
    rate_rows, ratio_rows = regional_summary(maps)
    local_rows = normalized_local_rates(maps)
    _write_dict_csv(os.path.join(out_dir, "regional_rates.csv"), rate_rows)
    _write_dict_csv(os.path.join(out_dir, "regional_ratios.csv"), ratio_rows)
    _write_dict_csv(os.path.join(out_dir, "normalized_local_rates.csv"), local_rows)
    plot_normalized_local_rates(
        local_rows, os.path.join(out_dir, "normalized_local_rates.png"))

    model = next(iter(maps.values())).get("recombination_model") or {}
    payload = {
        "recombination_model": model,
        "decoder_prior": {
            "type": "scalar_poisson_switch_prior",
            "rate_per_bp": next(iter(maps.values())).get("decoder_rate_per_bp", RECOMB_RATE),
            "profile_blind": True,
        },
        "regional_rates": rate_rows,
        "regional_ratios": ratio_rows,
        "bootstrap": {"unit": "chromosome", "replicates": 2000, "ci": 0.95},
    }
    with open(os.path.join(out_dir, "regional_summary.json"), "w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)

    ratio_lookup = {
        (row["track"], row["contrast"]): row for row in ratio_rows
    }
    lines = [
        "Variable recombination-rate recovery analysis",
        "============================================",
        "",
        f"Simulation model: {model.get('name', 'legacy_constant')}",
        f"Mean process rate: {float(model.get('mean_rate_per_bp', RECOMB_RATE)) * 1e8:.4g} cM/Mb",
        "Decoder prior: one scalar mean rate (profile-blind; no oracle use of the 20/80 boundaries).",
        "Reconstructed tracks use the atomic Stage-11 pre-phase Z1 painting and bundled H1 founder block;",
        "its ordered sample IDs must exactly match the Stage-2 global pedigree order.",
        "Stage-13 Q1 is intentionally not used because the map applies its own phase-aware trio HMM.",
        "Decoded-eligible raw events use the same F2/F3 meioses accessible to the trio decoder;",
        "raw-all additionally checks the complete simulated event generator, including F1 meioses.",
        "",
        "Regional end/middle rate ratios (95% chromosome-bootstrap CI):",
    ]
    contrasts = (
        "left20/middle60", "right20/middle60", "pooled_ends/middle60",
    )
    for key in TRACK_ORDER:
        rows = [ratio_lookup.get((key, contrast)) for contrast in contrasts]
        rows = [row for row in rows if row is not None]
        if not rows:
            continue
        lines.append(f"  {rows[0]['label']}:")
        for row in rows:
            low, high = row["bootstrap_ci95_low"], row["bootstrap_ci95_high"]
            ci_text = "analytic" if low is None else f"95% CI {low:.3f}-{high:.3f}"
            lines.append(
                f"    {row['contrast']}: {row['ratio']:.3f} ({ci_text})"
            )

    inferred_left = ratio_lookup.get(("inferred", "left20/middle60"))
    inferred_right = ratio_lookup.get(("inferred", "right20/middle60"))
    if inferred_left is not None and inferred_right is not None:
        separate_end_lows = (
            inferred_left["bootstrap_ci95_low"],
            inferred_right["bootstrap_ci95_low"],
        )
        both_supported = all(
            low is not None and np.isfinite(low) and low > 1.0
            for low in separate_end_lows
        )
        if both_supported:
            lines.extend([
                "",
                "Conclusion: both chromosome ends are enriched with chromosome-level support "
                "(each separate 95% CI excludes 1).",
            ])
        else:
            lines.extend([
                "",
                "Conclusion: both separate chromosome ends are not yet supported as enriched; "
                "at least one end/middle 95% CI includes 1.",
            ])
    lines.extend(["", "See regional_rates.csv, regional_ratios.csv, normalized_local_rates.csv,",
                  "normalized_local_rates.png, composite.png, and chromosomes/ for full results."])
    with open(os.path.join(out_dir, "README.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def save_map_data(maps, path):
    """Store curves, compact crossover midpoints, and exposures for re-analysis."""
    with open(path, "wb") as handle:
        pickle.dump(maps, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote map data: {path}")


def save_outputs(maps, out_dir,
                 composite_title="Recombination map (cumulative genetic distance)"):
    """Write comparable chromosome maps plus aggregate rate-recovery analysis."""
    os.makedirs(out_dir, exist_ok=True)
    chrom_dir = os.path.join(out_dir, "chromosomes")
    os.makedirs(chrom_dir, exist_ok=True)

    save_map_data(maps, os.path.join(out_dir, "map_data.pkl"))
    plot_maps(maps, os.path.join(out_dir, "composite.png"), title=composite_title)
    for contig, data in maps.items():
        plot_one_chromosome(contig, data, os.path.join(chrom_dir, f"{contig}.png"))
        write_chromosome_csv(contig, data, os.path.join(chrom_dir, f"{contig}.csv"))
    write_analysis_outputs(maps, out_dir)
    print(f"Wrote composite, aggregate analysis, and {len(maps)} chromosome outputs under {out_dir}/")


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


def _selftest_strict_stage11_loader():
    """Reject incomplete or sample-misaligned Stage-11 bundles without fallback."""
    requested_stages = []
    expected_sample_ids = ["sample_a", "sample_b"]

    def loader_for(payload):
        def load_payload(_ckpt_dir, stage, _contig):
            requested_stages.append(stage)
            if stage != STAGE_PAINT:
                raise AssertionError(f"unexpected fallback checkpoint stage: {stage}")
            return payload
        return load_payload

    missing_founder = {
        "tolerance_result": object(),
        pipeline_runtime.SAMPLE_IDS_KEY: expected_sample_ids,
    }
    try:
        load_stage11_painting_payload(
            "/unused", "chr_test", expected_sample_ids=expected_sample_ids,
            contig_loader=loader_for(missing_founder),
        )
    except KeyError as exc:
        founder_rejected = pipeline_runtime.FOUNDER_BLOCK_KEY in str(exc)
    else:
        founder_rejected = False

    missing_sample_ids = {
        "tolerance_result": object(),
        pipeline_runtime.FOUNDER_BLOCK_KEY: object(),
    }
    try:
        load_stage11_painting_payload(
            "/unused", "chr_test", expected_sample_ids=expected_sample_ids,
            contig_loader=loader_for(missing_sample_ids),
        )
    except KeyError as exc:
        sample_ids_rejected = pipeline_runtime.SAMPLE_IDS_KEY in str(exc)
    else:
        sample_ids_rejected = False

    mismatched_sample_ids = {
        "tolerance_result": [object(), object()],
        pipeline_runtime.FOUNDER_BLOCK_KEY: object(),
        pipeline_runtime.SAMPLE_IDS_KEY: list(reversed(expected_sample_ids)),
    }
    try:
        load_stage11_painting_payload(
            "/unused", "chr_test", expected_sample_ids=expected_sample_ids,
            contig_loader=loader_for(mismatched_sample_ids),
        )
    except ValueError:
        mismatch_rejected = True
    else:
        mismatch_rejected = False

    ok = (
        founder_rejected
        and sample_ids_rejected
        and mismatch_rejected
        and requested_stages == [STAGE_PAINT, STAGE_PAINT, STAGE_PAINT]
    )
    print("[loader] incomplete/misaligned Stage-11 H1/Z1 bundle rejected "
          f"without fallback -> {'OK' if ok else 'FAIL'}")
    return ok


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


def _selftest_variable_profile():
    """Check variable-profile math, event denominators, and aggregate outputs."""
    import tempfile

    model = {
        "name": "ends_2x_mean_preserving",
        "mean_rate_per_bp": RECOMB_RATE,
        "decoder_rate_per_bp": RECOMB_RATE,
        "profile_segments": [
            {"start_fraction": 0.0, "end_fraction": 0.2,
             "rate_multiplier": 10.0 / 7.0},
            {"start_fraction": 0.2, "end_fraction": 0.8,
             "rate_multiplier": 5.0 / 7.0},
            {"start_fraction": 0.8, "end_fraction": 1.0,
             "rate_multiplier": 10.0 / 7.0},
        ],
        "normalization": "mean_preserving_weighted_mean_1",
    }
    edges = _physical_edges(0.0, 10.5, 4.0, model)
    edge_ok = (edges[-1] == 10.5 and 2.1 in edges and 8.4 in edges)
    process_edges, process_cum = process_truth_cM(
        0.0, 100_000_000.0, 7_000_000.0, recombination_model=model)
    process_ok = (
        process_edges[-1] == 100_000_000.0 and
        np.isclose(process_cum[-1], 500.0) and
        np.isclose(_expected_rate(model, 0.0, 0.2), 50.0 / 7.0) and
        np.isclose(_expected_rate(model, 0.2, 0.8), 25.0 / 7.0)
    )

    records = []
    for generation, child in (("F1", "C1"), ("F2", "C2"), ("F3", "C3")):
        for parent_slot in (0, 1):
            records.append({
                "generation": generation,
                "child": child,
                "parent_slot": parent_slot,
                "crossover_positions_bp": np.asarray([10.0 + parent_slot]),
            })
    raw_all = _raw_event_track(records)
    raw_eligible = _raw_event_track(records, {"C2", "C3"})
    denominator_ok = raw_all[1] == 6 and raw_eligible[1] == 4

    def synthetic_contig():
        lo, hi = 0.0, 100_000_000.0
        # Pooled ends: 40 events in 40% of the exposure; middle: 30 in 60%.
        mids = np.concatenate((
            np.linspace(1.0, 19_000_000.0, 20),
            np.linspace(21_000_000.0, 79_000_000.0, 30),
            np.linspace(81_000_000.0, 99_000_000.0, 20),
        ))
        data = {
            "lo": lo, "hi": hi,
            "recombination_model": model,
            "decoder_rate_per_bp": RECOMB_RATE,
            "profile_boundaries_bp": np.asarray([lo, 0.2 * hi, 0.8 * hi, hi]),
            "track_labels": {"process": TRACK_LABELS["process"],
                             "inferred": TRACK_LABELS["inferred"]},
            "track_provenance": {},
            "crossover_midpoints_bp": {"inferred": mids.astype(np.float32)},
            "n_meioses_by_track": {"inferred": 100},
            "n_children_by_track": {"inferred": 50},
            "process": process_truth_cM(lo, hi, 7_000_000.0,
                                         recombination_model=model),
            "inferred": cumulative_cM(mids, 100, lo, hi, 7_000_000.0, model),
            "n_meioses_truth": 100, "n_meioses_inferred": 100,
            "n_crossovers_truth": 70, "n_crossovers_inferred": 70,
            "n_children_truth": 50, "n_children_inferred": 50,
        }
        return data

    maps = {"chr1": synthetic_contig(), "chr2": synthetic_contig()}
    _, ratios = regional_summary(maps, n_bootstrap=100)
    inferred_ratios = [row["ratio"] for row in ratios
                       if row["track"] == "inferred"]
    ratio_ok = (len(inferred_ratios) == 3 and
                all(np.isclose(ratio, 2.0) for ratio in inferred_ratios))
    with tempfile.TemporaryDirectory(prefix="recombination_map_selftest_") as out_dir:
        save_outputs(maps, out_dir)
        required = {
            "composite.png", "map_data.pkl", "normalized_local_rates.png",
            "regional_rates.csv", "regional_ratios.csv", "regional_summary.json",
            "README.txt",
        }
        output_ok = required.issubset(set(os.listdir(out_dir)))

    ok = edge_ok and process_ok and denominator_ok and ratio_ok and output_ok
    print("[variable] exact edge/profile slopes/raw denominators/ratio/output "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Build process, realized, truth-painting, reconstructed, and "
                    "end-to-end Marey maps from BHD simulation checkpoints.")
    p.add_argument("--ckpt-dir", default=".pipeline_checkpoints",
                   help="completed pipeline.py checkpoint tree with atomic Stage-11 "
                        "H1/Z1 bundles (NOT a sweep combo)")
    p.add_argument("--bin-mb", type=float, default=1.0, help="physical bin width (Mb)")
    p.add_argument("--out-dir", default="recombination_map",
                   help="output directory for chromosome maps, aggregate local-rate plot, "
                        "regional summaries, and map_data.pkl")
    p.add_argument("--contigs", nargs="*", default=None,
                   help="subset of contigs (default: all in stage 02 global)")
    p.add_argument("--true-pedigree-for-inferred", action="store_true",
                   help="ablation: build the inferred map with the TRUE pedigree links")
    p.add_argument("--inferred-pedigree-csv", default=None,
                   help="explicit path to pedigree_inference_discovered.csv")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel worker processes for the per-chromosome decode "
                        "(default: min(#contigs, CPU count); 1 = serial)")
    p.add_argument("--selftest", action="store_true",
                   help="run synthetic validation and exit (no checkpoints needed)")
    args = p.parse_args(argv)

    if args.selftest:
        results = (
            _selftest_strict_stage11_loader(),
            _selftest_driver(),
            _selftest_variable_profile(),
        )
        ok = all(results)
        print("=== DRIVER SELF-TEST", "PASSED" if ok else "FAILED", "===")
        return 0 if ok else 1

    maps = build_maps(
        args.ckpt_dir, bin_bp=args.bin_mb * 1e6,
        use_inferred_pedigree=not args.true_pedigree_for_inferred,
        inferred_csv=args.inferred_pedigree_csv,
        contigs=args.contigs,
        n_workers=args.workers,
    )
    save_outputs(maps, args.out_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
