"""refinement / polish for the canonical reconstruction pipeline."""
from __future__ import annotations


from dataclasses import dataclass
import numpy as np
from dataclasses import asdict
import math
import time
from numba import njit, prange

from dataclasses import replace
import haplotype_reconstruction.refinement.model as refinement_model

@dataclass(frozen=True)
class FinishedFamilyPhase:
    schema: str
    allele_calls: np.ndarray
    call_provenance: np.ndarray
    phase_map: np.ndarray
    changed_phase_sites: np.ndarray
    conditional_result: PhasePolishResult
    summary: dict


@dataclass(frozen=True)
class PhasePolishConfig:
    recombination_rate: float = 5e-8
    copy_error: float = .001
    phase_switch_probability: float = .005
    max_sweeps: int = 6
    correct_incomplete_parent_phase: bool = False
    adaptive_window_sites: int = 0
    minimum_score_gain: float = 1e-6
    transmission_only: bool = False
    minimum_biological_gain: float | None = None


@njit(cache=True, inline='always')
def _copy(a, b, match, mismatch):
    return 0. if a < 0 or b < 0 else (match if a == b else mismatch)


@dataclass(frozen=True)
class PhaseReleaseAssessment:
    phase: FinishedFamilyPhase
    family: object
    phase_stable: bool
    latent_converged: bool
    checks: tuple


def family_phase_context(reference, family):
    """Complete an internal scaffold from the existing fit, without new data.

Four-genotype uncertainty is retained in the source family product. Only its
>=.98 supported gap fills are used as this conditional model's context; they
do not become published allele calls unless imputation is explicitly enabled.
No called allele is overwritten, and posterior states are moved to the emitted
frame before completion then back to the original reference frame.
"""
    initial = family.phase_map
    posterior = family.ordered_genotype_probability.copy()
    reverse = family.inferred_phase_map != initial
    posterior[:,:,1] = np.where(reverse,family.ordered_genotype_probability[:,:,2],family.ordered_genotype_probability[:,:,1])
    posterior[:,:,2] = np.where(reverse,family.ordered_genotype_probability[:,:,1],family.ordered_genotype_probability[:,:,2])
    corrected = np.where(initial[:,:,None],reference[:,:,::-1],reference)
    filled,_ = refinement_model.complete_scaffold(corrected,family.phase_flip,posterior,.98)
    context = np.where(initial[:,:,None],filled[:,:,::-1],filled)
    if not np.array_equal(context[reference>=0],reference[reference>=0]):
        raise RuntimeError('internal family context changed a called allele')
    return context


@dataclass
class PhasePolishResult:
    phase_map: np.ndarray
    selectors: np.ndarray
    phase_theta: np.ndarray
    trace: list
    initial_score: float
    final_score: float
    converged: bool
    elapsed_seconds: float
    config: dict


@njit(cache=True)
def interval_gain(i, a, b, path, reference, flips, selectors, incoming, offsets, outgoing,
                  parents, logstay, logjump, phase_stay, phase_jump, error):
    e0, e1 = incoming[i, 0], incoming[i, 1]
    match, mismatch = math.log(2*(1-error)), math.log(2*error)
    biological, regularisation = 0., 0.
    for j in range(a, b):
        new_f, old_f = path[j] & 1, flips[i,j]
        for slot, e in ((0,e0),(1,e1)):
            if e < 0:
                continue
            parent = parents[e]
            new_s = (path[j] >> (slot+1)) & 1
            old_s = selectors[e,j]
            biological += _copy(reference[parent,j,new_s ^ flips[parent,j]],
                                reference[i,j,slot ^ new_f],match,mismatch)
            biological -= _copy(reference[parent,j,old_s ^ flips[parent,j]],
                                reference[i,j,slot ^ old_f],match,mismatch)
    for j in range(max(1,a), min(reference.shape[1], b+1)):
        new_left = (path[j-1] & 1) if a <= j-1 < b else flips[i,j-1]
        new_right = (path[j] & 1) if a <= j < b else flips[i,j]
        old_jump = flips[i,j-1] != flips[i,j]
        new_jump = new_left != new_right
        regularisation += (phase_jump[j-1] if new_jump else phase_stay[j-1])
        regularisation -= (phase_jump[j-1] if old_jump else phase_stay[j-1])
        for slot,e in ((0,e0),(1,e1)):
            if e < 0:
                continue
            left = ((path[j-1] >> (slot+1)) & 1) if a <= j-1 < b else selectors[e,j-1]
            right = ((path[j] >> (slot+1)) & 1) if a <= j < b else selectors[e,j]
            biological += logjump[j-1] if left != right else logstay[j-1]
            biological -= logjump[j-1] if selectors[e,j-1] != selectors[e,j] else logstay[j-1]
        for k in range(offsets[i],offsets[i+1]):
            e = outgoing[k]
            left = selectors[e,j-1] ^ flips[i,j-1] ^ new_left
            right = selectors[e,j] ^ flips[i,j] ^ new_right
            biological += logjump[j-1] if left != right else logstay[j-1]
            biological -= logjump[j-1] if selectors[e,j-1] != selectors[e,j] else logstay[j-1]
    return biological, regularisation


def assess_phase_release(gl, observed, reference, positions, phase_bins, component_ids,
                         relationships, sample_ids, parents, children, slots, family, *,
                         phase_config, family_config=refinement_model.FamilyRefinementConfig(),
                         required_unchanged=5, maximum_checks=20, callback=None, chromosome_map=None):
    if required_unchanged < 2 or maximum_checks < required_unchanged:
        raise ValueError('invalid consecutive phase-stability window')
    phase=finish_family_phase(reference,family,positions,phase_bins,component_ids,
                             parents,children,slots,config=phase_config,impute_missing=False,
                             chromosome_map=chromosome_map)
    if family.converged:
        return PhaseReleaseAssessment(phase,family,True,True,())
    checks=[];unchanged=0
    for check in range(maximum_checks):
        # Resume preserves iteration scheduling, existing clusters and message
        # state. It does not reinitialize a fit or change its biological model.
        config=replace(family_config,max_iterations=family.messages.iteration+1)
        following=refinement_model.refine_family(gl,observed,reference,positions,phase_bins,component_ids,
            relationships,sample_ids,config=config,resume=family.messages,chromosome_map=chromosome_map)
        proposed=finish_family_phase(reference,following,positions,phase_bins,component_ids,
                                    parents,children,slots,config=phase_config,impute_missing=False,
                                    chromosome_map=chromosome_map)
        changed=int(np.count_nonzero(np.any(proposed.allele_calls!=phase.allele_calls,axis=2)))
        unchanged=unchanged+1 if changed==0 else 0
        item={'check':check+1,'family_iteration':following.messages.iteration,
              'changed_called_phase_sites':changed,'consecutive_unchanged':unchanged,
              'latent_delta':following.messages.deltas[-1],
              'latent_converged':bool(following.converged)}
        checks.append(item);family,phase=following,proposed
        if callback is not None:callback(family,phase,item)
        if family.converged or unchanged>=required_unchanged:
            break
    stable=bool(family.converged or unchanged>=required_unchanged)
    return PhaseReleaseAssessment(phase,family,stable,bool(family.converged),tuple(checks))


def finish_family_phase(reference, family, positions, phase_bins, component_ids,
                        parents, children, slots, *, config, impute_missing=False,
                        callback=None, resume=None, chromosome_map=None):
    context = family_phase_context(reference,family)
    result = polish_phase(context,family.inferred_phase_map,positions,phase_bins,
        component_ids,parents,children,slots,config=config,callback=callback,resume=resume,
        chromosome_map=chromosome_map)
    if not result.converged:
        raise RuntimeError('conditional phase finalization has not converged; retain its iteration checkpoint')
    emitted = result.phase_map.copy()
    incomplete = np.bincount(children,minlength=len(reference)) < 2
    if not config.correct_incomplete_parent_phase:
        emitted[incomplete] = family.phase_map[incomplete]
    source = context if impute_missing else reference
    calls = np.where(emitted[:,:,None],source[:,:,::-1],source)
    original = np.where(emitted[:,:,None],reference[:,:,::-1],reference)
    provenance = (calls >= 0).astype(np.uint8)
    provenance[(calls>=0)&(original<0)] = 2
    if not impute_missing and not np.array_equal(np.sort(calls,axis=2),np.sort(reference,axis=2)):
        raise RuntimeError('phase-only finalization changed genotype or missingness')
    observable = reference[:,:,0] != reference[:,:,1]
    changed = (emitted != family.phase_map) & observable
    summary = {'schema':'finished-family-phase-view-v1','called_alleles':int(np.sum(calls>=0)),
        'original_called_alleles':int(np.sum(reference>=0)),
        'filled_alleles':int(np.sum(provenance==2)),
        'changed_observable_phase_sites':int(changed.sum()),
        'known_genotypes_changed':0,'imputation_enabled':bool(impute_missing),
        'phase_estimate':'conditional point path, not an independently calibrated posterior',
        'phase_probability_policy':'source-family probabilities remain in the source product; none are transferred to this changed path',
        'selector_frame':'conditional_result.phase_map internal family frame; M0/M1 emitted tracks can differ',
        'pedigree_updated':False,'founder_namespaces_merged':False,
        'conditional_score_gain':result.final_score-result.initial_score,
        'elapsed_numerical_seconds':result.elapsed_seconds,'config':result.config}
    return FinishedFamilyPhase('finished-family-phase-view-v1',calls,provenance,emitted,
                               changed,result,summary)


def pedigree_layout(samples, parents, children, slots):
    incoming = np.full((samples, 2), -1, dtype=np.int64)
    outgoing = [[] for _ in range(samples)]
    adjacent = [set() for _ in range(samples)]
    for e, (p, c, s) in enumerate(zip(parents, children, slots)):
        if p == c or not (0 <= p < samples and 0 <= c < samples and s in (0, 1)):
            raise ValueError("invalid pedigree edge")
        if incoming[c, s] >= 0:
            raise ValueError("duplicate parental slot")
        incoming[c, s] = e
        outgoing[p].append(e)
        adjacent[p].add(int(c)); adjacent[c].add(int(p))
    colours = np.full(samples, -1, dtype=np.int64)
    for i in sorted(range(samples), key=lambda j: (-len(adjacent[j]), j)):
        used = {colours[j] for j in adjacent[i] if colours[j] >= 0}
        colour = 0
        while colour in used:
            colour += 1
        colours[i] = colour
    groups = tuple(np.flatnonzero(colours == c) for c in range(int(colours.max())+1))
    offsets = np.r_[0, np.cumsum([len(x) for x in outgoing])].astype(np.int64)
    edges = np.asarray([e for group in outgoing for e in group], dtype=np.int64)
    return incoming, offsets, edges, groups


@njit(cache=True)
def accept_supported_intervals(i, path, reference, flips, selectors, incoming, offsets, outgoing,
                               parents, logstay, logjump, phase_stay, phase_jump, error,
                               minimum_gain, minimum_biological_gain):
    e0, e1 = incoming[i,0], incoming[i,1]
    old = flips[i].astype(np.int8).copy()
    if e0 >= 0:
        old += 2*selectors[e0]
    if e1 >= 0:
        old += 4*selectors[e1]
    total_gain, changed = 0., 0
    site = 0
    while site < len(path):
        if path[site] == old[site]:
            site += 1
            continue
        a = site
        meaningful_phase = False
        while site < len(path) and path[site] != old[site]:
            if (path[site] & 1) != (old[site] & 1) and reference[i,site,0] != reference[i,site,1]:
                meaningful_phase = True
            site += 1
        b = site
        biological, prior = interval_gain(i,a,b,path,reference,flips,selectors,incoming,offsets,
                                         outgoing,parents,logstay,logjump,phase_stay,phase_jump,error)
        gain = biological+prior
        if gain <= minimum_gain or (meaningful_phase and biological < minimum_biological_gain):
            continue
        total_gain += gain
        for j in range(a,b):
            f = path[j] & 1
            different = f ^ flips[i,j]
            changed += different
            for k in range(offsets[i],offsets[i+1]):
                selectors[outgoing[k],j] ^= different
            if e0 >= 0:
                selectors[e0,j] = (path[j] >> 1) & 1
            if e1 >= 0:
                selectors[e1,j] = (path[j] >> 2) & 1
            flips[i,j] = f
    return total_gain, changed


def phase_transitions(bins, components, probability, phase_map, selectors, window):
    """Optional fine boundaries preserve each coarse bin's flip probability.

For n allowed intervals within one coarse bin, theta=(1-(1-2p)**(1/n))/2.
Thus their composed binary transition has the original coarse probability p.
Windows are selected from current paths, never simulation truth.
"""
    bins = np.asarray(bins)
    components = np.asarray(components)
    selected = bins[1:] != bins[:-1]
    if window > 0:
        changes = np.any(phase_map[:, 1:] != phase_map[:, :-1], axis=0)
        if len(selectors):
            changes |= np.any(selectors[:, 1:] != selectors[:, :-1], axis=0)
        delta = np.zeros(len(bins), dtype=np.int64)
        for site in np.flatnonzero(changes):
            delta[max(0, site-window)] += 1
            delta[min(len(bins)-1, site+window+1)] -= 1
        selected |= np.cumsum(delta[:-1]) > 0
    theta = np.zeros(len(bins)-1, dtype=np.float64)
    # Contiguous bin groups, not global numerical bin IDs (ragged gaps may be -1).
    cuts = np.r_[0, np.flatnonzero(bins[2:] != bins[1:-1])+1, len(theta)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        sites = np.flatnonzero(selected[a:b])+a
        if len(sites):
            theta[sites] = -.5*np.expm1(np.log1p(-2*probability)/len(sites))
    theta[components[1:] != components[:-1]] = .5
    return theta


@njit(cache=True, inline="always")
def copy_log(a, b, match, mismatch):
    if a < 0 or b < 0:
        return 0.
    return match if a == b else mismatch


@njit(cache=True, parallel=True)
def initialize_selectors(reference, flips, parents, children, slots, logstay, logjump, error):
    edges, sites = len(parents), reference.shape[1]
    result = np.zeros((edges, sites), dtype=np.int8)
    match, mismatch = math.log(2.)+math.log1p(-error), math.log(2.)+math.log(error)
    for e in prange(edges):
        p, c, slot = parents[e], children[e], slots[e]
        back = np.zeros((sites, 2), dtype=np.int8)
        score = np.full(2, -math.log(2.))
        current = np.empty(2)
        for j in range(sites):
            child = reference[c, j, slot ^ flips[c, j]]
            for s in range(2):
                value = score[s]
                if j:
                    stay = score[s]+logstay[j-1]
                    jump = score[1-s]+logjump[j-1]
                    if jump > stay:
                        value = jump; back[j, s] = 1-s
                    else:
                        value = stay; back[j, s] = s
                current[s] = value+copy_log(reference[p, j, s ^ flips[p, j]], child, match, mismatch)
            score, current = current, score
        state = 0 if score[0] >= score[1] else 1
        for j in range(sites-1, -1, -1):
            result[e, j] = state
            if j:
                state = back[j, state]
    return result


@njit(cache=True, parallel=True)
def conditional_score(reference, flips, selectors, parents, children, slots,
                      logstay, logjump, phase_stay, phase_jump, error):
    samples, sites = reference.shape[:2]
    values = np.zeros(samples+len(parents), dtype=np.float64)
    match, mismatch = math.log(2.)+math.log1p(-error), math.log(2.)+math.log(error)
    for task in prange(len(values)):
        value = -math.log(2.)
        if task < samples:
            for j in range(1, sites):
                value += phase_jump[j-1] if flips[task,j] != flips[task,j-1] else phase_stay[j-1]
        else:
            e = task-samples; p, c, slot = parents[e], children[e], slots[e]
            for j in range(sites):
                value += copy_log(reference[p,j,selectors[e,j] ^ flips[p,j]],
                                  reference[c,j,slot ^ flips[c,j]], match, mismatch)
                if j:
                    value += logjump[j-1] if selectors[e,j] != selectors[e,j-1] else logstay[j-1]
        values[task] = value
    return np.sum(values)


@njit(cache=True)
def update_vertex(i, reference, flips, selectors, incoming, offsets, outgoing,
                  parents, logstay, logjump, phase_stay, phase_jump, error, minimum_gain,
                  minimum_biological_gain=-np.inf):
    sites = reference.shape[1]
    e0, e1 = incoming[i,0], incoming[i,1]
    match, mismatch = math.log(2.)+math.log1p(-error), math.log(2.)+math.log(error)
    scores = np.zeros(8, dtype=np.float64)
    values = np.empty(8, dtype=np.float64)
    origin = np.empty(8, dtype=np.int8)
    origins = np.empty(8, dtype=np.int8)
    back = np.zeros((sites, 8), dtype=np.int8)
    old_score = 0.
    for j in range(sites):
        trans_stay, trans_jump = 0., 0.
        if j:
            trans_stay, trans_jump = phase_stay[j-1], phase_jump[j-1]
            old_flip = flips[i,j] != flips[i,j-1]
            for k in range(offsets[i], offsets[i+1]):
                edge = outgoing[k]
                source_changed = (selectors[edge,j] != selectors[edge,j-1]) ^ old_flip
                trans_stay += logjump[j-1] if source_changed else logstay[j-1]
                trans_jump += logstay[j-1] if source_changed else logjump[j-1]
            old_score += trans_jump if old_flip else trans_stay
            if e0 >= 0:
                old_score += logjump[j-1] if selectors[e0,j] != selectors[e0,j-1] else logstay[j-1]
            if e1 >= 0:
                old_score += logjump[j-1] if selectors[e1,j] != selectors[e1,j-1] else logstay[j-1]
        if j:
            for state in range(8):
                origin[state] = state
            # Factorized max-product transition: 3 binary axes, O(8 log 8).
            for bit in range(3):
                stay = trans_stay if bit == 0 else logstay[j-1]
                jump = trans_jump if bit == 0 else logjump[j-1]
                absent = (bit == 1 and e0 < 0) or (bit == 2 and e1 < 0)
                if absent:
                    stay, jump = 0., -np.inf
                for state in range(8):
                    other = state ^ (1 << bit)
                    a, b = scores[state]+stay, scores[other]+jump
                    if b > a:
                        values[state], origins[state] = b, origin[other]
                    else:
                        values[state], origins[state] = a, origin[state]
                scores, values = values, scores
                origin, origins = origins, origin
            back[j] = origin
        for state in range(8):
            f, s0, s1 = state & 1, (state >> 1) & 1, (state >> 2) & 1
            if (e0 < 0 and s0) or (e1 < 0 and s1):
                scores[state] = -np.inf
                continue
            emission = 0.
            if e0 >= 0:
                p = parents[e0]
                emission += copy_log(reference[p,j,s0 ^ flips[p,j]], reference[i,j,f], match, mismatch)
            if e1 >= 0:
                p = parents[e1]
                emission += copy_log(reference[p,j,s1 ^ flips[p,j]], reference[i,j,1 ^ f], match, mismatch)
            scores[state] += emission
        if e0 >= 0:
            p = parents[e0]
            old_score += copy_log(reference[p,j,selectors[e0,j] ^ flips[p,j]], reference[i,j,flips[i,j]], match, mismatch)
        if e1 >= 0:
            p = parents[e1]
            old_score += copy_log(reference[p,j,selectors[e1,j] ^ flips[p,j]], reference[i,j,1 ^ flips[i,j]], match, mismatch)
    state = int(np.argmax(scores)); gain = scores[state]-old_score
    if gain <= minimum_gain:
        return 0., 0
    path = np.empty(sites, dtype=np.int8)
    for j in range(sites-1, -1, -1):
        path[j] = state
        if j:
            state = back[j,state]
    # A DP sum and the same fixed path can differ slightly in evaluation
    # order over long chromosomes. An unchanged configuration is not a move,
    # irrespective of the apparent floating-point score gain.
    different_configuration = False
    for j in range(sites):
        if ((path[j] & 1) != flips[i,j]
                or (e0 >= 0 and ((path[j] >> 1) & 1) != selectors[e0,j])
                or (e1 >= 0 and ((path[j] >> 2) & 1) != selectors[e1,j])):
            different_configuration = True
            break
    if not different_configuration:
        return 0., 0
    if np.isfinite(minimum_biological_gain):
        return accept_supported_intervals(i,path,reference,flips,selectors,incoming,offsets,outgoing,
            parents,logstay,logjump,phase_stay,phase_jump,error,minimum_gain,
            minimum_biological_gain)
    changed = 0
    for j in range(sites):
        state = path[j]; f = state & 1; different = f ^ flips[i,j]
        changed += different
        for k in range(offsets[i], offsets[i+1]):
            selectors[outgoing[k],j] ^= different
        if e0 >= 0:
            selectors[e0,j] = (state >> 1) & 1
        if e1 >= 0:
            selectors[e1,j] = (state >> 2) & 1
        flips[i,j] = f
    return gain, changed


@njit(cache=True, parallel=True)
def update_colour(vertices, reference, flips, selectors, incoming, offsets, outgoing,
                  parents, logstay, logjump, phase_stay, phase_jump, error, minimum_gain,
                  minimum_biological_gain):
    gains = np.zeros(len(vertices)); changed = np.zeros(len(vertices), dtype=np.int64)
    for v in prange(len(vertices)):
        gains[v], changed[v] = update_vertex(vertices[v], reference, flips, selectors,
            incoming, offsets, outgoing, parents, logstay, logjump, phase_stay, phase_jump,
            error, minimum_gain, minimum_biological_gain)
    return gains, changed


def polish_phase(reference, initial_phase, positions, bins, component_ids, parents, children, slots,
                 *, config=PhasePolishConfig(), callback=None, resume=None, chromosome_map=None):
    started = time.monotonic()
    reference = np.ascontiguousarray(reference, dtype=np.int8)
    flips = np.ascontiguousarray(initial_phase, dtype=np.int8).copy()
    positions = np.asarray(positions, dtype=np.int64)
    bins = np.asarray(bins, dtype=np.int64); components = np.asarray(component_ids, dtype=np.int64)
    if reference.ndim != 3 or reference.shape[2] != 2 or flips.shape != reference.shape[:2]:
        raise ValueError("phase and allele axes differ")
    if positions.shape != (reference.shape[1],) or bins.shape != positions.shape or components.shape != positions.shape:
        raise ValueError("marker axes differ")
    if len(positions) == 0 or np.any(np.diff(positions) <= 0):
        raise ValueError("positions must be nonempty and increasing")
    if np.any(~np.isin(reference, (-1,0,1))) or np.any(~np.isin(flips, (0,1))):
        raise ValueError("invalid allele or phase values")
    if not (0 < config.copy_error < .5 and 0 < config.phase_switch_probability < .5
            and np.isfinite(config.recombination_rate) and config.recombination_rate >= 0):
        raise ValueError("invalid model probabilities")
    if config.max_sweeps < 1 or config.adaptive_window_sites < 0:
        raise ValueError("invalid iteration/window setting")
    parents, children, slots = (np.ascontiguousarray(x, dtype=np.int64) for x in (parents, children, slots))
    if not (len(parents) == len(children) == len(slots)):
        raise ValueError("pedigree axes differ")
    incoming, offsets, outgoing, colours = pedigree_layout(len(reference), parents, children, slots)
    eligible = np.sum(incoming >= 0, axis=1) == 2
    if config.correct_incomplete_parent_phase:
        eligible |= (np.diff(offsets) > 0) | np.any(incoming >= 0, axis=1)
    if chromosome_map is None or not chromosome_map.has_map:
        rate = config.recombination_rate if chromosome_map is None else chromosome_map.fallback_rate_per_bp
        theta = -.5*np.expm1(-2*rate*np.diff(positions))
    else:
        theta = -.5*np.expm1(-2*chromosome_map.interval_morgans(positions[:-1], positions[1:]))
    theta[components[1:] != components[:-1]] = .5
    logstay, logjump = np.log1p(-theta), np.full_like(theta, -np.inf)
    np.log(theta, out=logjump, where=theta > 0)
    if resume is None:
        selectors = initialize_selectors(reference, flips, parents, children, slots, logstay, logjump, config.copy_error)
        phase_theta = phase_transitions(bins, components, config.phase_switch_probability,
                                        flips, selectors, config.adaptive_window_sites)
        if config.transmission_only:
            phase_theta[:] = .5
    else:
        flips = np.asarray(resume['phase_map'], dtype=np.int8).copy()
        selectors = np.asarray(resume['selectors'], dtype=np.int8).copy()
        phase_theta = np.asarray(resume['phase_theta'], dtype=np.float64).copy()
    phase_stay = np.log1p(-phase_theta)
    phase_jump = np.full_like(phase_theta, -np.inf)
    np.log(phase_theta, out=phase_jump, where=phase_theta > 0)
    if np.any((flips[:,1:] != flips[:,:-1]) & (phase_theta[None,:] == 0)):
        raise ValueError("initial phase path crosses a prohibited boundary")
    current_score = float(conditional_score(reference, flips, selectors, parents, children, slots,
                                      logstay, logjump, phase_stay, phase_jump, config.copy_error))
    initial = current_score if resume is None else float(resume['initial_score'])
    trace = [] if resume is None else list(resume['trace'])
    score = current_score
    converged = bool(trace and trace[-1]['accepted_vertex_updates'] == 0)
    for sweep in range(len(trace), config.max_sweeps):
        if converged:
            break
        gain = 0.; changed = 0; updates = 0
        for vertices in (colours if sweep % 2 == 0 else colours[::-1]):
            vertices = vertices[eligible[vertices]]
            if not len(vertices):
                continue
            gains, changes = update_colour(vertices, reference, flips, selectors, incoming, offsets,
                outgoing, parents, logstay, logjump, phase_stay, phase_jump,
                config.copy_error, config.minimum_score_gain,
                -np.inf if config.minimum_biological_gain is None else config.minimum_biological_gain)
            gain += float(gains.sum()); changed += int(changes.sum()); updates += int((gains > 0).sum())
        score += gain
        item = {"sweep": sweep+1, "conditional_score": score, "score_gain": gain,
                "changed_phase_sites": changed, "accepted_vertex_updates": updates,
                "elapsed_seconds": time.monotonic()-started}
        trace.append(item)
        if callback is not None:
            callback(item, flips, selectors, phase_theta, initial, trace)
        if updates == 0:
            converged = True
            break
    final = float(conditional_score(reference, flips, selectors, parents, children, slots,
                                    logstay, logjump, phase_stay, phase_jump, config.copy_error))
    if not np.isclose(final, score, rtol=0., atol=max(1e-5, abs(final)*1e-9)):
        raise RuntimeError("local score gains do not equal the global conditional score change")
    model_config = asdict(config)
    if chromosome_map is not None:
        model_config['recombination_rate'] = chromosome_map.fallback_rate_per_bp
        model_config['recombination_map'] = chromosome_map.identity()
    return PhasePolishResult(flips, selectors, phase_theta, trace, initial, final,
                             converged, time.monotonic()-started, model_config)


